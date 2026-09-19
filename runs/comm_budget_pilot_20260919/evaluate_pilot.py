"""§7 and §8: fair comparison of the rules and the learned controller, then the same check
with the environment's own sensing switched back on.

Every arm runs the same recorded detection stream and decides its own emissions; nobody reuses
another arm's delivery schedule. Lossless airtime is the primary comparison. The lossy run is a
diagnostic in its own right, because with packets dying the cheap-looking arm is simply the one
whose packets did not arrive -- the ledger reports attempts, deliveries and applications side by
side so that confusion cannot be read into the numbers.

Paired differences are per seed, with a percentile bootstrap over seeds (B and the RNG seed
fixed here, before any evaluation was run).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

import pilot_common as common
import pilot_env
import train_pilot

BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_SEED = 20260919


def paired(
    rows: list[dict[str, Any]],
    *,
    baseline: str,
    extract: Callable[[dict[str, Any]], float],
) -> dict[str, Any]:
    """Seed-level paired difference of every arm against the baseline arm."""
    by_arm: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        by_arm[row["arm"]][int(row["seed"])] = float(extract(row))
    if baseline not in by_arm:
        return {"baseline": baseline, "arms": {}}
    shared = sorted(set(by_arm[baseline]) & set().union(*[set(values) for values in by_arm.values()]))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output: dict[str, Any] = {}
    for arm, values in by_arm.items():
        if arm == baseline:
            continue
        seeds = [seed for seed in shared if seed in values and seed in by_arm[baseline]]
        if not seeds:
            continue
        deltas = np.array([values[seed] - by_arm[baseline][seed] for seed in seeds], dtype=float)
        draws = rng.choice(deltas, size=(BOOTSTRAP_REPLICATIONS, deltas.size), replace=True)
        means = draws.mean(axis=1)
        output[arm] = {
            "pairs": len(seeds),
            "mean_difference": round(float(deltas.mean()), 5),
            "ci95": [round(float(np.percentile(means, 2.5)), 5), round(float(np.percentile(means, 97.5)), 5)],
            "seeds_better": int((deltas > 0).sum()),
            "seeds_worse": int((deltas < 0).sum()),
            "seeds_equal": int((deltas == 0).sum()),
        }
    return {"baseline": baseline, "shared_seeds": len(shared), "arms": output}


def load_controller(path: Path, seed: int) -> train_pilot.Controller:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    controller = train_pilot.Controller(seed)
    controller.load_state_dict(payload["controller"])
    return controller


def collect(
    *,
    tier: str,
    seeds: list[int],
    checkpoints: list[Path],
    directory: Path,
    information: str,
    delivery_probability: float,
    episode_limit: int,
    budget_bytes: int | None = None,
) -> list[dict[str, Any]]:
    locked = json.loads((common.HERE / "COMM_BUDGET_LOCKED.json").read_text(encoding="utf-8"))
    credit = int(budget_bytes or locked["budgets_bytes"][tier])
    arms: list[tuple[str, dict[str, Any]]] = [
        # The ceiling read: the same rule allowed to spend as much airtime as it wants. Every
        # other arm is judged against what it managed inside a fraction of that.
        ("free_event_reference", {"policy": "event", "unbounded": True}),
        ("periodic_5", {"policy": "periodic", "params": {"interval": 5}}),
        ("event", {"policy": "event"}),
        ("event_priority", {"policy": "event_priority"}),
    ]
    rows: list[dict[str, Any]] = []
    directory.mkdir(parents=True, exist_ok=True)
    for name, spec in arms:
        # The unbudgeted reference is the ceiling read: it is the only arm allowed to spend
        # whatever it wants, and every other arm is judged against what it managed for less.
        for index, seed in enumerate(seeds):
            if episode_limit and index >= episode_limit:
                break
            row = pilot_row(
                seed,
                label=f"{name}_{information}_{delivery_probability}",
                policy=spec["policy"],
                params=spec.get("params"),
                budget=None if spec.get("unbounded") else credit,
                directory=directory,
                information=information,
                delivery_probability=delivery_probability,
                arm=name,
                tier="free" if spec.get("unbounded") else tier,
            )
            rows.append(row)
    for path in checkpoints:
        trained_seed = int(path.stem.split("seed")[-1])
        controller = load_controller(path, trained_seed)
        for index, seed in enumerate(seeds):
            if episode_limit and index >= episode_limit:
                break
            rows.append(
                pilot_row(
                    seed,
                    label=f"learned{trained_seed}_{information}_{delivery_probability}",
                    policy="learned",
                    params=None,
                    budget=credit,
                    directory=directory,
                    information=information,
                    delivery_probability=delivery_probability,
                    arm=f"learned_seed{trained_seed}",
                    tier=tier,
                    controller=controller,
                )
            )
    return rows


def pilot_row(seed: int, **kwargs: Any) -> dict[str, Any]:
    """One evaluation episode, through the same runner the trainer uses."""
    controller = kwargs.pop("controller", None)
    directory = kwargs.pop("directory")
    label = kwargs.pop("label")
    arm = kwargs.pop("arm")
    tier = kwargs.pop("tier")
    budget = kwargs.pop("budget")
    policy = kwargs.pop("policy")
    params = kwargs.pop("params")
    information = kwargs.pop("information")
    delivery_probability = kwargs.pop("delivery_probability")
    collector = train_pilot.Collector(controller, greedy=True)
    row = pilot_env_run(
        seed,
        collector=collector,
        budget_bytes=budget,
        directory=directory,
        label=label,
        information=information,
        delivery_probability=delivery_probability,
        policy=policy,
        params=params,
    )
    ledger = row["ledger"]
    return {
        "arm": arm,
        "tier": tier,
        "seed": int(seed),
        "information": information,
        "delivery_probability": delivery_probability,
        "on_time": int(row["task"]["on_time"]),
        "targets": int(row["task"]["targets"]),
        "expired": int(row["task"]["expired"]),
        "airtime_bytes": int(row["spent_bytes"]),
        "credit_used_fraction": float(row["budget_used_fraction"] or 0.0),
        "attempts": int(ledger["transmission_attempts"]),
        "delivered": int(ledger["packets_delivered"]),
        "applied": int(ledger["deliveries_applied"]),
        "duplicate": int(ledger["delivery_duplicate"]),
        "lost": int(ledger["packets_lost"]),
        "budget_refusals": int(ledger["budget_refused_sends"]),
        "forced_sends": int(collector.forced_sends),
        "belief_age_steps": float(row["belief"].get("belief_age_steps_mean") or 0.0),
        "requirement_error_steps": float(row["belief"].get("requirement_error_steps") or 0.0),
        "unknown_to_everyone_steps": float(row["belief"].get("unknown_to_everyone_steps") or 0.0),
        "decide_milliseconds_per_call": collector.timing()["decide_milliseconds_per_call"],
        "decisions": int(collector.timing()["decisions"]),
        "seconds": float(row["seconds"]),
        "arm_label": row["arm_label"],
    }


def pilot_env_run(seed: int, *, collector, budget_bytes, directory, label, information, delivery_probability, policy, params):
    return pilot_env.run_episode(
        seed,
        directory=directory,
        label=label,
        policy=policy,
        policy_params=params,
        budget_bytes=budget_bytes,
        controller=collector.policy() if policy == "learned" else None,
        information=information,
        delivery_probability=delivery_probability,
    )


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for arm in sorted({row["arm"] for row in rows}):
        values = [row for row in rows if row["arm"] == arm]
        table[arm] = {
            "episodes": len(values),
            "mean_on_time": round(float(np.mean([row["on_time"] for row in values])), 4),
            "mean_expired": round(float(np.mean([row["expired"] for row in values])), 4),
            "mean_airtime_bytes": round(float(np.mean([row["airtime_bytes"] for row in values])), 1),
            "mean_attempts": round(float(np.mean([row["attempts"] for row in values])), 1),
            "mean_applied": round(float(np.mean([row["applied"] for row in values])), 1),
            "mean_lost": round(float(np.mean([row["lost"] for row in values])), 1),
            "mean_budget_refusals": round(float(np.mean([row["budget_refusals"] for row in values])), 1),
            "credit_used_fraction": round(float(np.mean([row["credit_used_fraction"] for row in values])), 4),
            "mean_belief_age_steps": round(float(np.mean([row["belief_age_steps"] for row in values])), 2),
            "mean_unknown_to_everyone": round(
                float(np.mean([row["unknown_to_everyone_steps"] for row in values])), 1
            ),
            "decide_milliseconds_per_call": round(
                float(np.mean([row["decide_milliseconds_per_call"] for row in values])), 4
            ),
        }
    return table


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default=None, help="locked tier name; defaults to the calibrated one")
    parser.add_argument("--seeds", type=int, nargs="*", default=list(common.VALIDATION_SEEDS))
    parser.add_argument("--checkpoints", nargs="*", default=[], help="pilot_seed<N>.pt files")
    parser.add_argument("--episode-limit", type=int, default=0)
    parser.add_argument("--out", default=str(common.HERE / "COMM_POLICY_PILOT_RESULTS.json"))
    parser.add_argument("--include-deployed", action="store_true", help="also score the same arms with the environment's own sensing switched back on")
    parser.add_argument("--include-lossy", action="store_true", help="also score the arms over a 30% loss channel, as a diagnostic")
    parser.add_argument("--threads", type=int, default=1, help="torch threads; 1 keeps the inference timing meaningful")
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, int(args.threads)))

    directory = common.HERE / "eval"
    tier = args.tier
    if tier is None:
        calibration = json.loads((common.HERE / "COMM_BUDGET_CALIBRATION.json").read_text(encoding="utf-8"))
        tier = calibration["tier_selection"]["selected_tier"]
    checkpoints = [Path(path) for path in args.checkpoints]
    settings = [("isolated", 1.0)]
    if args.include_lossy:
        settings.append(("isolated", 0.7))
    if args.include_deployed:
        settings.append(("deployed", 1.0))
    all_rows: list[dict[str, Any]] = []
    for information, probability in settings:
        rows = collect(
            tier=tier,
            seeds=args.seeds,
            checkpoints=checkpoints,
            directory=directory / f"{information}_{probability}",
            information=information,
            delivery_probability=probability,
            episode_limit=args.episode_limit,
        )
        all_rows.extend(rows)
        key = f"{information}_{probability}"
        payload = {
            "tier": tier,
            "setting": {"information": information, "delivery_probability": probability},
            "episodes": len(rows),
            "per_arm": summarise(rows),
            "paired_vs_event": paired(rows, baseline="event", extract=lambda row: row["on_time"]),
            "paired_bytes_vs_event": paired(rows, baseline="event", extract=lambda row: row["airtime_bytes"]),
            "per_seed": sorted(
                ({"seed": row["seed"], "arm": row["arm"], "on_time": row["on_time"], "bytes": row["airtime_bytes"]} for row in rows),
                key=lambda item: (item["seed"], item["arm"]),
            ),
        }
        common.write_json(common.HERE / f"COMM_PILOT_{key}.json", payload)
        print(f"== {key} ==")
        for arm, values in payload["per_arm"].items():
            print(
                f"{arm:<20} eps={values['episodes']:>3} on_time={values['mean_on_time']:>6} "
                f"bytes={values['mean_airtime_bytes']:>8} attempts={values['mean_attempts']:>6} "
                f"applied={values['mean_applied']:>6} lost={values['mean_lost']:>5} "
                f"refusals={values['mean_budget_refusals']:>5} age={values['mean_belief_age_steps']:>6} "
                f"unk={values['mean_unknown_to_everyone']:>7} ms={values['decide_milliseconds_per_call']}"
            )
        print(json.dumps(payload["paired_vs_event"], ensure_ascii=False, indent=1))
    common.write_json(
        Path(args.out),
        {"tier": tier, "settings": [list(item) for item in settings], "rows": all_rows},
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
