"""§6: score every checkpoint the same way, and never confuse three models' 72 rows.

Round 2 reported one good seed, and that was the end of it. Round 3's unit of evidence is the
*training seed*, so this file reports the three separately -- the final checkpoint and the one the
pre-declared validation rule picks -- against the rules, on the 24 validation streams, and then
repeats the comparison over a 30 % loss channel and with the environment's own sensing switched
back on.

Two pairings are kept apart on purpose. Differences are paired **within a training seed** across
the 24 validation streams, which is where the scenario uncertainty lives; the spread **between**
the three training runs is a different quantity, reported as mean, range, standard deviation and
how often each beat the comparator, and never folded into a 72-sample bootstrap.

Three modes, because the second two need the first one's answer:

* ``curve``    -- every checkpoint on the primary setting, and the rule-selected checkpoint;
* ``settings`` -- the final and selected checkpoints, plus the rule comparators, on one channel;
* ``report``  -- combine the fragments into ``COMM_TRAINING_STABILITY_RESULTS.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import attribution
import stability_common as common

import train_pilot

#: Fixed before any continuation checkpoint was scored. Validation completions first, airtime
#: second, earliest step last: the order the research question states, not one fitted to a curve.
SELECTION_RULE = "max mean_on_time, then min mean_airtime_bytes, then earliest step"
PRIMARY = ("isolated", 1.0)
#: A 30 % loss channel is delivery 0.7 -- the channel round 2 measured, so the two are comparable.
SECONDARY = (("isolated", 0.7), ("deployed", 1.0))
#: The rules a learned arm has to be measured against. ``immediate`` and ``hold_to_bound`` are the
#: two points on the attribution table's frontier; the others are the pre-registered comparators
#: that a reader may want to see anyway.
COMPARATORS = ("immediate", "hold_to_bound", "periodic_5", "event", "event_priority", "random_send")
RULE_SPECS = {spec["arm"]: spec for spec in common.ARMS}
BOOTSTRAP_NOTE = "seed-level paired percentile bootstrap, B=10 000, RNG 20260919"


def checkpoint_paths(seed: int) -> dict[str, Path]:
    directory = common.TRAIN_OUT / f"seed{seed}"
    found: dict[str, Path] = {"pilot_100k": common.PILOT_CHECKPOINTS[seed]}
    for path in sorted(directory.glob(f"stability_seed{seed}*.pt")):
        found[path.stem] = path
    return {key: value for key, value in found.items() if value.exists()}


def load_controller(path: Path, training_seed: int) -> train_pilot.Controller:
    controller = train_pilot.Controller(training_seed)
    controller.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["controller"])
    return controller


def steps_of(path: Path) -> int:
    return int(torch.load(path, map_location="cpu", weights_only=False)["state"].get("environment_steps", -1))


def unique_checkpoints(seed: int) -> dict[int, Path]:
    """One checkpoint per distinct step count -- the milestone files duplicate the resumable one."""
    by_step: dict[int, Path] = {}
    for name, path in sorted(checkpoint_paths(seed).items()):
        step_count = steps_of(path)
        by_step.setdefault(step_count, path)
    return by_step


def score_arm(
    spec: dict[str, Any],
    name: str,
    controller: Any,
    seeds: list[int],
    *,
    directory: Path,
    budget: int | None,
    setting: tuple[str, float],
) -> list[dict[str, Any]]:
    information, probability = setting
    rows = []
    for seed in seeds:
        row = attribution.run_arm(
            {**spec, "arm": name},
            seed,
            controller=controller,
            provenance={"kind": spec["kind"], "checkpoint": name},
            directory=directory,
            budget=budget,
            information=information,
            delivery_probability=probability,
            label=f"{name}_{information}_{probability}",
        )
        row["checkpoint"] = name
        rows.append(row)
    return rows


def learned_spec(name: str, training_seed: int, path: Path) -> dict[str, Any]:
    return {
        "arm": name,
        "kind": "checkpoint",
        "policy": "learned",
        "params": "",
        "training_seed": str(training_seed),
        "checkpoint_path": str(path),
    }


def means(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        return round(float(np.mean([float(row[key]) for row in rows])), 4)

    reasons: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        for reason, bucket in row["rollup"]["by_reason"].items():
            for key, value in bucket.items():
                reasons[reason][key] += int(value)
    return {
        "episodes": len(rows),
        "on_time_per_episode": mean("on_time_targets_per_episode"),
        "on_time_rate_per_episode": mean("on_time_rate_within_detected_this_episode"),
        "expired_per_episode": mean("expired_targets_per_episode"),
        "airtime_bytes": mean("airtime_bytes"),
        "attempts": mean("transmission_attempts"),
        "applied": mean("deliveries_applied"),
        "emissions_policy": reasons["policy"]["emissions"],
        "emissions_forced": reasons["forced_maxwait"]["emissions"],
        "bytes_policy": reasons["policy"]["bytes"],
        "bytes_forced": reasons["forced_maxwait"]["bytes"],
        "autonomous_share_of_bytes": mean("autonomous_share_of_bytes"),
        "inference_milliseconds_per_decision": mean("inference_milliseconds_per_decision"),
        "seconds_per_episode": mean("seconds"),
    }


def selected_checkpoint(curve: dict[str, Any]) -> dict[str, Any]:
    ranked = sorted(
        curve.items(),
        key=lambda item: (-float(item[1]["on_time_per_episode"]), float(item[1]["airtime_bytes"]), int(item[0])),
    )
    return {"rule": SELECTION_RULE, "at_steps": ranked[0][0], **ranked[0][1]}


def curve_mode(args: argparse.Namespace) -> int:
    torch.set_num_threads(max(1, int(args.threads)))
    directory = common.HERE / "stability_eval"
    directory.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {}
    for seed in (1, 2, 3):
        points: dict[str, Any] = {}
        for step_count, path in sorted(unique_checkpoints(seed).items()):
            controller = load_controller(path, seed)
            rows = score_arm(
                learned_spec(f"learned{seed}", seed, path),
                f"learned{seed}",
                controller,
                args.seeds,
                directory=directory / "curve",
                budget=args.budget_bytes,
                setting=PRIMARY,
            )
            points[str(step_count)] = {**means(rows), "path": str(path)}
            print(f"[seed {seed}] {step_count:>6} onT={points[str(step_count)]['on_time_per_episode']:>7} "
                  f"bytes={points[str(step_count)]['airtime_bytes']:>9} "
                  f"auton={points[str(step_count)]['autonomous_share_of_bytes']:>6}", flush=True)
        output[str(seed)] = {"curve": points, "selected": selected_checkpoint(points)}
        print(f"[seed {seed}] selected {output[str(seed)]['selected']['at_steps']} by {SELECTION_RULE}", flush=True)
    common.write_json(directory / "stability_curve.json", output)
    return 0


def settings_mode(args: argparse.Namespace) -> int:
    torch.set_num_threads(max(1, int(args.threads)))
    setting = (args.information, float(args.delivery_probability))
    directory = common.HERE / "stability_eval"
    curve = json.loads((directory / "stability_curve.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for arm in COMPARATORS:
        spec = RULE_SPECS[arm]
        rows.extend(
            score_arm(
                spec,
                arm,
                None,
                args.seeds,
                directory=directory / "settings",
                budget=None if spec["kind"] == "rule_ceiling" else args.budget_bytes,
                setting=setting,
            )
        )
        print(f"[{arm}] done", flush=True)
    for seed, block in curve.items():
        wanted = {int(block["selected"]["at_steps"]), max(int(step) for step in block["curve"])}
        for step_count in sorted(wanted):
            path = Path(block["curve"][str(step_count)]["path"])
            rows.extend(
                score_arm(
                    learned_spec(f"learned{seed}_at{step_count}", int(seed), path),
                    f"learned{seed}_at{step_count}",
                    load_controller(path, int(seed)),
                    args.seeds,
                    directory=directory / "settings",
                    budget=args.budget_bytes,
                    setting=setting,
                )
            )
            print(f"[learned {seed} at {step_count}] done", flush=True)
    with (directory / f"rows_{'_'.join(str(part) for part in setting)}.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


def paired_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import evaluate_pilot

    arms = sorted({str(row["arm"]) for row in rows})
    comparison: dict[str, Any] = {}
    for baseline in ("immediate", "hold_to_bound"):
        if baseline not in arms:
            continue
        comparison[baseline] = {
            metric: evaluate_pilot.paired(
                rows, baseline=baseline, extract=lambda row, metric=metric: float(row[metric])
            )["arms"]
            for metric in ("on_time_targets_per_episode", "airtime_bytes")
        }
    return comparison


def judgement(rows: list[dict[str, Any]], *, budget: int) -> dict[str, Any]:
    """The four conditions frozen in ``COMM_TRAINING_STABILITY_PROTOCOL.md`` §7, per learned arm."""
    comparison = paired_block(rows)
    table = attribution.M.summarise(rows)
    verdict: dict[str, Any] = {"statistic": BOOTSTRAP_NOTE, "budget_bytes": budget, "arms": {}}
    for arm, values in table.items():
        if not arm.startswith("learned"):
            continue
        against_immediate = comparison.get("immediate", {})
        completion = against_immediate.get("on_time_targets_per_episode", {}).get(arm)
        cost = against_immediate.get("airtime_bytes", {}).get(arm)
        against_bound = comparison.get("hold_to_bound", {}).get("airtime_bytes", {}).get(arm)
        policy_bytes = int(values["by_reason"]["policy"]["bytes"])
        checks = {
            "1_completion_non_inferior_to_immediate": bool(
                completion and float(completion["ci95"][0]) >= -0.05
            ),
            "2_bytes_strictly_below_immediate": bool(cost and float(cost["ci95"][1]) < 0),
            "3_policy_sent_something_it_was_not_told_to": bool(policy_bytes > 0),
            "4_not_more_expensive_than_the_bound": bool(
                against_bound and float(against_bound["ci95"][1]) <= 0
            ),
        }
        verdict["arms"][arm] = {
            "costs_more_than_the_bound": bool(
                against_bound and float(against_bound["mean_difference"]) > 0
            ),
            "checks": checks,
            "passed": all(checks.values()),
            "numbers": {
                "on_time_vs_immediate": completion,
                "bytes_vs_immediate": cost,
                "bytes_vs_hold_to_bound": against_bound,
                "policy_bytes": policy_bytes,
            },
        }
    passing = [arm for arm, value in verdict["arms"].items() if value["passed"]]
    verdict["at_least_two_of_three_seeds_pass"] = len({arm.split("_at")[0] for arm in passing}) >= 2
    verdict["seeds_passing"] = sorted(passing)
    return verdict


def report_mode(args: argparse.Namespace) -> int:
    directory = common.HERE / "stability_eval"
    curve = json.loads((directory / "stability_curve.json").read_text(encoding="utf-8"))
    per_setting: dict[str, Any] = {}
    for path in sorted(directory.glob("rows_*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        key = path.stem[len("rows_") :]
        per_setting[key] = {
            "per_arm": attribution.M.summarise(rows),
            "paired": paired_block(rows),
            "judgement": judgement(rows, budget=args.budget_bytes),
        }
    payload = {
        "validation_seeds": sorted(int(seed) for seed in args.seeds),
        "statistic": BOOTSTRAP_NOTE,
        "budget_bytes": args.budget_bytes,
        "settings": {"primary": list(PRIMARY), "secondary": [list(item) for item in SECONDARY]},
        "learning_curves": {
            seed: {
                "points": {step: values for step, values in block["curve"].items()},
                "selected_by_rule": block["selected"],
            }
            for seed, block in sorted(curve.items())
        },
        "between_training_seeds": between_seed_spread(curve, per_setting),
        "per_setting": per_setting,
    }
    common.write_json(common.json_path("COMM_TRAINING_STABILITY_RESULTS.json"), payload)
    for key, block in per_setting.items():
        print(f"== {key} ==")
        attribution.print_table(block["per_arm"])
        print(json.dumps(block["judgement"], ensure_ascii=False, indent=1))
    print(json.dumps(payload["between_training_seeds"], ensure_ascii=False, indent=1))
    return 0


def between_seed_spread(curve: dict[str, Any], per_setting: dict[str, Any]) -> dict[str, Any]:
    """The training-seed variance, stated separately from any within-seed interval."""
    spread: dict[str, Any] = {}
    for key, block in per_setting.items():
        table = block["per_arm"]
        outcomes, costs, used = {}, {}, []
        for seed in ("1", "2", "3"):
            # The checkpoint the declared rule selected for this seed -- not "whichever arm name
            # sorts first", which would silently mix a milestone with a final.
            step = str(curve[seed]["selected"]["at_steps"])
            chosen = f"learned{seed}_at{step}"
            if chosen not in table:
                continue
            outcomes[f"seed{seed}"] = float(table[chosen]["on_time_targets_per_episode_mean"])
            costs[f"seed{seed}"] = float(table[chosen]["airtime_bytes_mean"])
            used.append(chosen)
        if not outcomes:
            continue

        def block_of(values: dict[str, float]) -> dict[str, Any]:
            numbers = list(values.values())
            return {
                "per_training_seed": {name: round(value, 4) for name, value in sorted(values.items())},
                "mean": round(float(np.mean(numbers)), 4),
                "range": [round(min(numbers), 4), round(max(numbers), 4)],
                "standard_deviation": round(float(np.std(numbers, ddof=1)), 4) if len(numbers) > 1 else None,
            }

        spread[key] = {
            "on_time_targets_per_episode": block_of(outcomes),
            "airtime_bytes": block_of(costs),
            "arms_used": sorted(used),
        }
    return spread


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("curve", "settings", "report"), default="report")
    parser.add_argument("--budget-bytes", type=int, default=common.BUDGET_BYTES)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(common.VALIDATION_SEEDS))
    parser.add_argument("--information", default="isolated")
    parser.add_argument("--delivery-probability", type=float, default=0.7)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.mode == "curve":
        return curve_mode(args)
    if args.mode == "settings":
        return settings_mode(args)
    return report_mode(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
