"""§7: score the corrected runs, and keep the two uncertainties apart.

Three things this file refuses to do:

* **Pool 72 correlated rows.** Three models evaluated on the same 24 validation streams produce
  72 numbers with 24 shared scenarios in them. Within-seed differences are paired across those
  streams; the spread between training seeds is a separate table, never folded into a bootstrap.
* **Compare a learned arm to a rule from another tier.** Every arm is scored against rules run at
  the same credit and under the same shared silence bound, per budget tier.
* **Score a checkpoint the trainer did not produce.** Each row carries the checkpoint path, its
  sha256, the environment steps recorded inside it, and a weight fingerprint, so a table row can be
  traced back to the file that made it.

Modes: ``curve`` (every checkpoint at one tier on the primary setting), ``settings`` (the
comparators plus the selected and final checkpoints at one tier on one channel), ``report``
(merge, apply the pre-frozen judgement).
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

import tight_common as common

import attribution_metrics as M
import onpolicy
import pilot_env
import train_pilot
from integrated_sar.comm_policy import parse_parameters

#: The validation selection rule carried over from round 3, unchanged: completions first, then
#: airtime, then the earlier checkpoint. Declared here so the report can name it next to its choice.
SELECTION_RULE = "max mean_on_time, then min mean_airtime_bytes, then earliest step"
PRIMARY = ("isolated", 1.0)
#: A 30 % loss channel is delivery probability 0.7 -- the same channel rounds 2 and 3 measured.
SECONDARY = (("isolated", 0.7), ("deployed", 1.0))
BOOTSTRAP_NOTE = "seed-level paired percentile bootstrap, B=10 000, RNG 20260919"
#: The §四 thresholds, copied from COMM_TIGHT_BUDGET_PROTOCOL.md rather than re-derived.
NON_INFERIORITY_TARGETS_PER_EPISODE = -0.05
RULE_ARMS = {spec["arm"]: spec for spec in common.RULE_ARMS}
#: `send_once_no_bound` drops the shared bound, so it is protocol-cost diagnostics, never a
#: comparator for a threshold (protocol §五). Same for the unbudgeted ceiling.
COMPARATORS = ("immediate", "hold_to_bound", "periodic_5", "event", "event_priority")


def score_row(
    name: str,
    kind: str,
    seed: int,
    *,
    controller: Any,
    policy: str,
    params: str,
    budget: int | None,
    bound: str | None,
    directory: Path,
    setting: tuple[str, float],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """One arm, one stream, one row: task units plus the reason-split ledger and its residuals."""
    information, probability = setting
    learned = controller is not None
    collector = (
        onpolicy.OnPolicyCollector(controller, forced=train_pilot.forced, greedy=True)
        if learned
        else None
    )
    unbounded = bound == "none"
    row = pilot_env.run_episode(
        seed,
        directory=directory,
        label=f"{name}_{information}_{probability}",
        policy=policy if not learned else "learned",
        policy_params=parse_parameters(params) or None,
        budget_bytes=budget,
        controller=collector.policy() if learned else None,
        information=information,
        delivery_probability=probability,
        max_wait_steps=None if unbounded else pilot_env.MAX_WAIT_STEPS,
        urgent_wait_steps=None if unbounded else pilot_env.MAX_WAIT_WHEN_URGENT_STEPS,
    )
    records = common.metrics.load_transcript(row["artefacts"]["transcript"])
    rollup = M._per_reason_rollup(records)
    ledger = row["ledger"]
    closure = M._closure(ledger, rollup, spent_bytes=row["spent_bytes"])
    emissions = int(ledger["emissions"])
    return {
        "arm": name,
        "kind": kind,
        "seed": int(seed),
        "budget_bytes": budget,
        "information": information,
        "delivery_probability": float(probability),
        "provenance": provenance,
        "on_time_targets_per_episode": int(row["task"]["on_time"]),
        "detected_targets_this_episode": int(row["task"]["targets"]),
        "on_time_rate_within_detected_this_episode": (
            round(int(row["task"]["on_time"]) / int(row["task"]["targets"]), 5)
            if int(row["task"]["targets"])
            else None
        ),
        "expired_targets_per_episode": int(row["task"]["expired"]),
        "airtime_bytes": int(row["spent_bytes"]),
        "credit_used_fraction": float(row["budget_used_fraction"] or 0.0),
        "over_budget": int(row["spent_bytes"]) > int(budget) if budget else False,
        "emissions": emissions,
        "redundant_emissions": int(ledger["redundant_emissions"]),
        "generated_items": int(ledger["generated_items"]),
        "coalesced_changes": int(ledger["coalesced_changes"]),
        "suppressed_terminal": int(ledger["emissions_suppressed_terminal"]),
        "emissions_by_policy": int(ledger["emissions_policy"]),
        "emissions_forced": int(ledger["emissions_forced_maxwait"]),
        "optional_emissions_share": round((emissions - int(ledger["emissions_forced_maxwait"])) / emissions, 4)
        if emissions
        else None,
        "autonomous_share_of_bytes": rollup["autonomous_share_of_bytes"],
        "deliveries_applied": int(ledger["deliveries_applied"]),
        "budget_refusals": int(ledger["budget_refused_sends"]),
        "belief_age_steps": float(row["belief"].get("belief_age_steps_mean") or 0.0),
        "unknown_to_everyone_steps": float(row["belief"].get("unknown_to_everyone_steps") or 0.0),
        "inference_milliseconds_per_decision": round(
            1000.0 * collector.decide_seconds / max(1, collector.decisions), 4
        )
        if learned
        else None,
        "rollup": rollup,
        "closure": closure,
        "residuals": {key: value for key, value in closure["residuals"].items() if value},
        "seconds": float(row["seconds"]),
        "config_hash": row["config_hash"],
    }


def score_rule(arm: str, seed: int, *, budget: int, directory: Path, setting: tuple[str, float]) -> dict[str, Any]:
    spec = RULE_ARMS[arm]
    return score_row(
        arm,
        "rule",
        seed,
        controller=None,
        policy=spec["policy"],
        params=spec["params"],
        budget=None if spec.get("ceiling") else budget,
        bound=spec.get("bound"),
        directory=directory,
        setting=setting,
        provenance={"kind": "rule", "policy": spec["policy"], "params": spec["params"]},
    )


def inventory(tier: str, training_seed: int) -> dict[str, dict[str, Any]]:
    """Every checkpoint this run owns, keyed by the environment steps recorded inside it."""
    points: dict[str, dict[str, Any]] = {}
    initial = train_pilot.Controller(training_seed)
    points["init"] = {
        "path": None,
        "steps": 0,
        "controller": initial,
        "provenance": {
            "kind": "reconstructed-initialisation",
            "weight_fingerprint": common.weight_fingerprint(initial),
            "archived_checkpoint": False,
        },
    }
    directory = common.TRAIN_OUT / f"{tier}_seed{training_seed}"
    for path in sorted(directory.glob(f"tight_{tier}_seed{training_seed}*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("trainer") != "onpolicy":
            raise RuntimeError(f"{path} was not produced by the corrected trainer")
        controller = train_pilot.Controller(training_seed)
        controller.load_state_dict(payload["controller"])
        steps = int(payload["state"].get("environment_steps", -1))
        entry = {
            "path": str(path),
            "steps": steps,
            "controller": controller,
            "provenance": {
                "kind": "checkpoint",
                "file": path.name,
                "sha256": common.sha256(path),
                "environment_steps": steps,
                "weight_fingerprint": common.weight_fingerprint(controller),
                "collector": payload["state"].get("collector"),
                "optimizer_steps": sum(
                    int(item.get("optimizer_steps", 0)) for item in payload["state"].get("updates", [])
                ),
            },
        }
        points[path.stem] = entry
    by_step: dict[int, dict[str, Any]] = {}
    for name, entry in points.items():
        by_step.setdefault(int(entry["steps"]), entry)
    return {str(step): entry for step, entry in sorted(by_step.items())}


def means(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        return round(float(np.mean([float(row[key] or 0.0) for row in rows])), 4)

    reasons: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        for reason, bucket in row["rollup"]["by_reason"].items():
            for key, value in bucket.items():
                if key != "mean_wait_steps":
                    reasons[reason][key] += int(value)
    return {
        "episodes": len(rows),
        "on_time_targets_per_episode_mean": mean("on_time_targets_per_episode"),
        "on_time_rate_per_episode_mean": mean("on_time_rate_within_detected_this_episode"),
        "pooled_on_time_over_detected_targets": round(
            sum(int(row["on_time_targets_per_episode"]) for row in rows)
            / max(1, sum(int(row["detected_targets_this_episode"]) for row in rows)),
            5,
        ),
        "expired_per_episode_mean": mean("expired_targets_per_episode"),
        "airtime_bytes_mean": mean("airtime_bytes"),
        "credit_used_fraction_mean": mean("credit_used_fraction"),
        "episodes_over_budget": sum(1 for row in rows if row["over_budget"]),
        "budget_refusals_mean": mean("budget_refusals"),
        "emissions_mean": mean("emissions"),
        "redundant_emissions_mean": mean("redundant_emissions"),
        "optional_emissions_share_mean": mean("optional_emissions_share"),
        "emissions_policy_total": reasons["policy"]["emissions"],
        "emissions_forced_total": reasons["forced_maxwait"]["emissions"],
        "bytes_policy_total": reasons["policy"]["bytes"],
        "bytes_forced_total": reasons["forced_maxwait"]["bytes"],
        "autonomous_share_of_bytes_mean": mean("autonomous_share_of_bytes"),
        "applied_mean": mean("deliveries_applied"),
        "belief_age_mean": mean("belief_age_steps"),
        "unknown_to_everyone_mean": mean("unknown_to_everyone_steps"),
        "inference_ms_mean": mean("inference_milliseconds_per_decision"),
        "ledger_residual_rows": sum(1 for row in rows if row["residuals"]),
        # generated - (said once) - merged - silenced, with "repeat" taken from the relay itself.
        # A positive value is news the sender knew and never spoke before the episode ended.
        "generation_identity_unspoken_total": sum(
            int(row["generated_items"])
            - (int(row["emissions"]) - int(row["redundant_emissions"]))
            - int(row["coalesced_changes"])
            - int(row["suppressed_terminal"])
            for row in rows
        ),
        "generation_identity_negative_episodes": sum(
            1
            for row in rows
            if int(row["generated_items"])
            - (int(row["emissions"]) - int(row["redundant_emissions"]))
            - int(row["coalesced_changes"])
            - int(row["suppressed_terminal"])
            < 0
        ),
        "seconds_mean": mean("seconds"),
    }


def paired(rows: list[dict[str, Any]], *, baseline: str, key: str) -> dict[str, Any]:
    import evaluate_pilot

    return evaluate_pilot.paired(rows, baseline=baseline, extract=lambda row, key=key: float(row[key]))["arms"]


def evaluation_root(tier: str) -> Path:
    """One scratch tree per tier.

    The episode runner names its lifecycle and transcript files by arm label and seed only, so two
    tiers scored in parallel in one directory append to the same file and read back each other's
    half-written lines -- the failure round 3 already documented for parallel training seeds.
    """
    return common.HERE / "evaluation" / tier


def curve_mode(args: argparse.Namespace) -> int:
    torch.set_num_threads(max(1, int(args.threads)))
    directory = evaluation_root(args.tier)
    (directory / "curve").mkdir(parents=True, exist_ok=True)
    seeds = list(common.VALIDATION_SEEDS)
    budget = common.BUDGETS_BYTES[args.tier]
    output: dict[str, Any] = {}
    for training_seed in (1, 2, 3):
        points: dict[str, Any] = {}
        for step, entry in inventory(args.tier, training_seed).items():
            rows = [
                score_row(
                    f"learned{training_seed}",
                    "checkpoint" if entry["path"] else "init",
                    seed,
                    controller=entry["controller"],
                    policy="learned",
                    params="",
                    budget=budget,
                    bound=None,
                    directory=directory / "curve",
                    setting=PRIMARY,
                    provenance=entry["provenance"],
                )
                for seed in seeds
            ]
            points[step] = {**means(rows), "path": entry["path"]}
            print(
                f"[{args.tier} seed {training_seed}] {step:>6} onT={points[step]['on_time_targets_per_episode_mean']:>7}"
                f" bytes={points[step]['airtime_bytes_mean']:>9}"
                f" auton={points[step]['autonomous_share_of_bytes_mean']:>7}",
                flush=True,
            )
        selected = selected_point(points)
        output[str(training_seed)] = {"curve": points, "selected": selected}
        print(f"[{args.tier} seed {training_seed}] selected {selected['at_steps']} by {SELECTION_RULE}", flush=True)
    common.write_json(directory / f"curve_{args.tier}.json", output)
    return 0


def selected_point(points: dict[str, Any]) -> dict[str, Any]:
    ranked = sorted(
        points.items(),
        key=lambda item: (
            -float(item[1]["on_time_targets_per_episode_mean"]),
            float(item[1]["airtime_bytes_mean"]),
            int(item[0]),
        ),
    )
    return {"rule": SELECTION_RULE, "at_steps": ranked[0][0], **ranked[0][1]}


def settings_mode(args: argparse.Namespace) -> int:
    torch.set_num_threads(max(1, int(args.threads)))
    setting = (args.information, float(args.delivery_probability))
    directory = evaluation_root(args.tier)
    curve = json.loads((directory / f"curve_{args.tier}.json").read_text(encoding="utf-8"))
    seeds = list(common.VALIDATION_SEEDS)
    budget = common.BUDGETS_BYTES[args.tier]
    rows: list[dict[str, Any]] = []
    for arm in COMPARATORS:
        rows.extend(
            score_rule(arm, seed, budget=budget, directory=directory / "settings", setting=setting)
            for seed in seeds
        )
        print(f"[{args.tier}/{arm}] done", flush=True)
    for training_seed, block in sorted(curve.items()):
        wanted = sorted({int(block["selected"]["at_steps"]), max(int(step) for step in block["curve"])})
        available = inventory(args.tier, int(training_seed))
        for step in wanted:
            entry = available.get(str(step))
            if entry is None:
                continue
            rows.extend(
                score_row(
                    f"learned{training_seed}_at{step}",
                    "checkpoint" if entry["path"] else "init",
                    seed,
                    controller=entry["controller"],
                    policy="learned",
                    params="",
                    budget=budget,
                    bound=None,
                    directory=directory / "settings",
                    setting=setting,
                    provenance=entry["provenance"],
                )
                for seed in seeds
            )
            print(f"[{args.tier}/learned{training_seed}_at{step}] done", flush=True)
    path = directory / f"rows_{args.tier}_{'_'.join(str(part) for part in setting)}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


def judgement(
    rows: list[dict[str, Any]], *, tier: str, selected: dict[str, str] | None = None
) -> dict[str, Any]:
    """The conditions of COMM_TIGHT_BUDGET_PROTOCOL.md section 4, applied per training seed.

    A seed contributes one arm -- the checkpoint its validation rule selected -- because the gate is
    stated over training seeds, and letting it contribute both the selected and the final point would
    quietly test six seeds and call the result three. Thresholds are copied from the protocol.
    """
    budget = common.BUDGETS_BYTES[tier]
    comparator = "immediate"
    margin = NON_INFERIORITY_TARGETS_PER_EPISODE
    completions = paired(rows, baseline=comparator, key="on_time_targets_per_episode")
    bytes_vs_rule = paired(rows, baseline=comparator, key="airtime_bytes")
    bytes_vs_bound = paired(rows, baseline="hold_to_bound", key="airtime_bytes")
    table = {name: values for name, values in sorted(_grouped_means(rows).items())}
    verdict = {
        "tier": tier,
        "budget_bytes": budget,
        "comparator": comparator,
        "statistic": BOOTSTRAP_NOTE,
        "non_inferiority_targets_per_episode": margin,
        "checkpoint_used_per_seed": dict(selected or {}),
        "seeds": {},
    }
    for seed in ("1", "2", "3"):
        arm = "learned%s_at%s" % (seed, (selected or {}).get(seed, ""))
        values = table.get(arm)
        if values is None:
            verdict["seeds"][seed] = {"arm": arm, "present": False}
            continue
        completion = completions.get(arm)
        cost = bytes_vs_rule.get(arm)
        difference = None if not completion else float(completion["mean_difference"])
        hygiene = bool(values["episodes_over_budget"] == 0 and values["ledger_residual_rows"] == 0)
        verdict["seeds"][seed] = {
            "arm": arm,
            "present": True,
            "hygiene_clean": hygiene,
            "completion_gain_ci_lower_above_zero": bool(
                completion and difference > 0 and float(completion["ci95"][0]) > 0
            ),
            "completion_point_estimate": difference,
            "no_regression_below_margin": bool(completion and difference >= margin),
            "same_effect_cheaper": bool(
                completion
                and float(completion["ci95"][0]) >= margin
                and cost
                and float(cost["ci95"][1]) < 0
            ),
            "numbers": {
                "on_time_vs_immediate": completion,
                "bytes_vs_immediate": cost,
                "bytes_vs_hold_to_bound": bytes_vs_bound.get(arm),
                "autonomous_bytes": int(values["bytes_policy_total"]),
                "autonomous_share_of_bytes": float(values["autonomous_share_of_bytes_mean"]),
                "optional_emissions_share": float(values["optional_emissions_share_mean"]),
                "redundant_emissions_mean": float(values["redundant_emissions_mean"]),
            },
        }
    present = [value for value in verdict["seeds"].values() if value.get("present")]
    gaining = [value for value in present if value["completion_gain_ci_lower_above_zero"] and value["hygiene_clean"]]
    others = [value for value in present if value not in gaining]
    verdict["training_seeds_reaching_upgrade_bar"] = len(gaining)
    verdict["upgrade_gate_met"] = bool(
        len(gaining) >= 2
        and all(value["no_regression_below_margin"] for value in others)
        and bool(present)
        and all(value["hygiene_clean"] for value in present)
    )
    verdict["cost_saving_arms"] = sorted(
        value["arm"] for value in present if value["same_effect_cheaper"] and value["hygiene_clean"]
    )
    verdict["training_seeds_reaching_cost_saving_bar"] = len(verdict["cost_saving_arms"])
    return verdict


def _group(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm"])].append(row)
    return grouped


def between_seed_spread(curve: dict[str, Any]) -> dict[str, Any]:
    """Training-seed variance at this tier, stated apart from any within-seed interval."""
    outcomes: dict[str, float] = {}
    costs: dict[str, float] = {}
    autonomy: dict[str, float] = {}
    for seed, block in sorted(curve.items()):
        chosen = block["curve"].get(str(block["selected"]["at_steps"]))
        if chosen is None:
            continue
        outcomes[f"seed{seed}"] = float(chosen["on_time_targets_per_episode_mean"])
        costs[f"seed{seed}"] = float(chosen["airtime_bytes_mean"])
        autonomy[f"seed{seed}"] = float(chosen["autonomous_share_of_bytes_mean"])

    def block_of(values: dict[str, float]) -> dict[str, Any]:
        numbers = list(values.values())
        if not numbers:
            return {"per_training_seed": {}}
        return {
            "per_training_seed": {name: round(value, 4) for name, value in sorted(values.items())},
            "mean": round(float(np.mean(numbers)), 4),
            "range": [round(min(numbers), 4), round(max(numbers), 4)],
            "standard_deviation": round(float(np.std(numbers, ddof=1)), 4) if len(numbers) > 1 else None,
        }

    return {
        "on_time_targets_per_episode": block_of(outcomes),
        "airtime_bytes": block_of(costs),
        "autonomous_share_of_bytes": block_of(autonomy),
    }


def report_mode(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "protocol": {
            "selection_rule": SELECTION_RULE,
            "statistic": BOOTSTRAP_NOTE,
            "non_inferiority_targets_per_episode": NON_INFERIORITY_TARGETS_PER_EPISODE,
            "comparators_always_reported": ["immediate", "hold_to_bound"],
            "diagnostics_excluded_from_thresholds": ["send_once_no_bound", "event_unbudgeted"],
            "primary_setting": list(PRIMARY),
            "secondary_settings": [list(item) for item in SECONDARY],
            "final_test_block": "800000..800399 reserved, not generated, not read",
        },
        "tiers": {},
    }
    for tier in args.tiers:
        directory = evaluation_root(tier)
        curve_path = directory / f"curve_{tier}.json"
        if not curve_path.exists():
            continue
        curve = json.loads(curve_path.read_text(encoding="utf-8"))
        block: dict[str, Any] = {
            "learning_curves": {
                seed: {"points": values["curve"], "selected_by_rule": values["selected"]}
                for seed, values in sorted(curve.items())
            },
            "selected_checkpoint_per_seed": {
                seed: str(values["selected"]["at_steps"]) for seed, values in sorted(curve.items())
            },
            "between_training_seeds": between_seed_spread(curve),
            "settings": {},
        }
        for path in sorted(directory.glob(f"rows_{tier}_*.jsonl")):
            key = path.stem[len(f"rows_{tier}_") :]
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            block["settings"][key] = {
                "per_arm": {name: values for name, values in sorted(_grouped_means(rows).items())},
                "paired_vs_immediate": {
                    metric: paired(rows, baseline="immediate", key=metric)
                    for metric in ("on_time_targets_per_episode", "airtime_bytes")
                },
                "paired_vs_hold_to_bound": {
                    metric: paired(rows, baseline="hold_to_bound", key=metric)
                    for metric in ("on_time_targets_per_episode", "airtime_bytes")
                },
                "judgement": judgement(
                    rows, tier=tier, selected=block["selected_checkpoint_per_seed"]
                ),
            }
        payload["tiers"][tier] = block
    common.write_json(common.json_path("COMM_TIGHT_BUDGET_RESULTS.json"), payload)
    _print(payload)
    return 0


def _grouped_means(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {name: means(group) for name, group in sorted(_group(rows).items())}


def _print(payload: dict[str, Any]) -> None:
    for tier, block in payload["tiers"].items():
        print(f"\n===== tier {tier} ({common.BUDGETS_BYTES[tier]} B) =====")
        for seed, curve in block["learning_curves"].items():
            print(f"  seed {seed}: selected {curve['selected_by_rule']['at_steps']} steps")
            for step, values in sorted(curve["points"].items(), key=lambda item: int(item[0])):
                print(
                    f"    {step:>6} onT={values['on_time_targets_per_episode_mean']:>7}"
                    f" bytes={values['airtime_bytes_mean']:>9}"
                    f" auton={values['autonomous_share_of_bytes_mean']:>7}"
                    f" optional={values['optional_emissions_share_mean']:>7}"
                )
        print("  between seeds:", json.dumps(block["between_training_seeds"], ensure_ascii=False))
        for key, setting in block["settings"].items():
            print(f"  -- setting {key}")
            print(
                f"     {'arm':<26}{'eps':>4}{'onT/ep':>8}{'rate':>9}{'bytes':>9}{'auton':>7}"
                f"{'over':>5}{'resid':>6}{'unspoken':>9}"
            )
            for arm, values in setting["per_arm"].items():
                print(
                    f"     {arm:<26}{values['episodes']:>4}"
                    f"{values['on_time_targets_per_episode_mean']:>8}"
                    f"{values['on_time_rate_per_episode_mean']:>9}"
                    f"{values['airtime_bytes_mean']:>9}"
                    f"{values['autonomous_share_of_bytes_mean']:>7}"
                    f"{values['episodes_over_budget']:>5}"
                    f"{values['ledger_residual_rows']:>6}"
                    f"{values['generation_identity_unspoken_total']:>9}"
                )
            print(f"     judgement: {json.dumps(setting['judgement'], ensure_ascii=False)[:1200]}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("curve", "settings", "report"), default="report")
    parser.add_argument("--tier", default="25pct", choices=tuple(common.BUDGETS_BYTES))
    parser.add_argument("--tiers", nargs="*", default=list(common.BUDGET_TIERS))
    parser.add_argument("--information", default="isolated")
    parser.add_argument("--delivery-probability", type=float, default=1.0)
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
