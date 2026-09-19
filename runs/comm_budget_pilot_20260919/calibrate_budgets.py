"""§4: calibrate the budget tiers on training seeds, then lock them as absolute numbers.

Three phases, in this order, so a budget is never derived from the episode it constrains:

* ``reference`` -- the published rules run with unbounded airtime over the training seeds. The
  event-triggered arm's mean cost is the reference the tiers are cut from.
* ``lock`` -- turn that mean into 75 / 50 / 25 percent absolute byte credits, rounded down to
  hundreds, and write them once to ``COMM_BUDGET_LOCKED.json``. Every later arm reads that
  file; nothing re-derives a credit from a seed's own traffic, which would be a per-episode
  oracle over the future.
* ``tiered`` -- replay the rules at the locked credits, so the pilot's tier is chosen from
  measured contention rather than a guess.

``BUDGET_TIER_RULE`` states how the tier is selected, and was written before any tier ran.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import pilot_env
import pilot_common as common

HERE = common.HERE
ROWS = HERE / "rows_calibration.jsonl"
LOCKED = HERE / "COMM_BUDGET_LOCKED.json"

#: The published rules. The budget of each arm is read from the locked file, never recomputed.
RULES: dict[str, dict[str, Any]] = {
    "immediate": {"policy": "immediate"},
    "periodic_5": {"policy": "periodic", "params": {"interval": 5}},
    "event": {"policy": "event"},
    "event_priority": {"policy": "event_priority"},
}

TIERS = ("75pct", "50pct", "25pct")

#: How the pilot's tier is picked, written before any tiered arm ran: the tier must refuse at
#: least one send on most seeds for the event arm (otherwise there is no competition to learn
#: from), and some rule must still be leaving completions on the table (otherwise there is
#: nothing to win). Among the tiers that satisfy both, take the most generous -- a tighter tier
#: is a later experiment, not the pilot's job.
BUDGET_TIER_RULE = (
    "largest tier with budget_refused_sends > 0 on >= 50% of training seeds for the event arm "
    "and mean on_time across arms < the unbudgeted event arm's mean on_time"
)


def arm_spec(arm: str) -> dict[str, Any] | None:
    """Resolve an arm name -- ``event`` or ``event@50pct`` -- into its runner arguments."""
    name, _, tier = arm.partition("@")
    if name not in RULES:
        return None
    spec = dict(RULES[name])
    budget = None
    if tier:
        budgets = json.loads(LOCKED.read_text(encoding="utf-8"))["budgets_bytes"]
        if tier not in budgets:
            raise SystemExit(f"unknown tier {tier!r}; locked tiers are {sorted(budgets)}")
        budget = int(budgets[tier])
    return {"policy": spec["policy"], "params": spec.get("params"), "budget": budget}


def already_done() -> set[tuple[str, int]]:
    if not ROWS.exists():
        return set()
    seen = set()
    for line in ROWS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            seen.add((row["arm"], int(row["seed"])))
    return seen


def collect(arms: list[str], seeds: list[int], directory: Path, shard: int, shards: int) -> int:
    done = already_done()
    cells = [(arm, int(seed)) for arm in arms for seed in seeds]
    cells = [
        cell
        for index, cell in enumerate(cells)
        if index % max(1, shards) == shard % max(1, shards) and cell not in done
    ]
    directory.mkdir(parents=True, exist_ok=True)
    failures = 0
    with ROWS.open("a", encoding="utf-8") as handle:
        for arm, seed in cells:
            spec = arm_spec(arm)
            if spec is None:
                raise SystemExit(f"unknown arm {arm!r}")
            try:
                row = pilot_env.run_episode(
                    seed,
                    directory=directory,
                    label=arm.replace("@", "_"),
                    policy=spec["policy"],
                    policy_params=spec.get("params"),
                    budget_bytes=spec.get("budget"),
                )
            except Exception as error:  # a failed cell is recorded, not silently skipped
                failures += 1
                row = {"arm": arm, "seed": seed, "technical_failure": f"{type(error).__name__}: {error}"}
            row["arm"] = arm
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            task = row.get("task", {})
            ledger = row.get("ledger", {})
            print(
                f"[{arm} s{seed}] on_time={task.get('on_time')}/{task.get('targets')} "
                f"spent={row.get('spent_bytes')} refused={ledger.get('budget_refused_sends')} "
                f"{row.get('seconds')}s"
                + (f" FAILED {row.get('technical_failure')}" if "technical_failure" in row else ""),
                flush=True,
            )
    return failures


def lock() -> dict[str, Any]:
    rows = [json.loads(line) for line in ROWS.read_text(encoding="utf-8").splitlines() if line.strip()]
    reference = [
        row
        for row in rows
        if row.get("arm") == "event" and row.get("spent_bytes") is not None and not row.get("technical_failure")
    ]
    if not reference:
        raise SystemExit("run --phase reference for the event arm before locking tiers")
    mean_bytes = statistics.mean(float(row["spent_bytes"]) for row in reference)
    payload = {
        "reference_arm": "event, unbounded airtime, training seeds",
        "reference_mean_airtime_bytes": round(mean_bytes, 1),
        "reference_episodes": len(reference),
        "budgets_bytes": {tier: int(mean_bytes * float(tier[:2]) / 100.0) // 100 * 100 for tier in TIERS},
        "tier_selection_rule": BUDGET_TIER_RULE,
        "seeds": sorted({int(row["seed"]) for row in reference}),
        "why_constant": (
            "a budget derived from the episode being scored would be knowledge of the future; "
            "these credits are fixed numbers now, and every arm reads the same file"
        ),
    }
    common.write_json(LOCKED, payload)
    return payload


def pick_tier() -> dict[str, Any]:
    rows = [json.loads(line) for line in ROWS.read_text(encoding="utf-8").splitlines() if line.strip()]
    locked = json.loads(LOCKED.read_text(encoding="utf-8"))
    reference = statistics.mean(
        float(row["task"]["on_time"]) for row in rows if row.get("arm") == "event"
    )
    verdict: dict[str, Any] = {"free_reference_mean_on_time": round(reference, 4), "tiers": {}}
    for tier in TIERS:
        arms = {
            name: [
                row
                for row in rows
                if row.get("arm") == f"{name}@{tier}" and not row.get("technical_failure")
            ]
            for name in RULES
        }
        events = arms["event"]
        if not events:
            verdict["tiers"][tier] = {"measured": False}
            continue
        means = {
            name: (
                round(statistics.mean(float(row["task"]["on_time"]) for row in values), 4)
                if values
                else None
            )
            for name, values in arms.items()
        }
        contention = sum(1 for row in events if int(row["ledger"]["budget_refused_sends"]) > 0)
        headroom = reference - (max(value for value in means.values() if value is not None))
        verdict["tiers"][tier] = {
            "measured": True,
            "credit_bytes": locked["budgets_bytes"][tier],
            "event_seeds_with_refusals": contention,
            "event_seeds": len(events),
            "mean_on_time_by_arm": means,
            "best_rule_gap_to_free_reference": round(headroom, 4),
            "contention_ok": contention >= len(events) / 2.0,
            "headroom_ok": headroom > 0.0,
        }
    eligible = [
        tier
        for tier in TIERS
        if verdict["tiers"][tier].get("measured")
        and verdict["tiers"][tier]["contention_ok"]
        and verdict["tiers"][tier]["headroom_ok"]
    ]
    verdict["selected_tier"] = max(eligible, key=lambda tier: locked["budgets_bytes"][tier]) if eligible else None
    verdict["rule"] = BUDGET_TIER_RULE
    return verdict


def aggregate() -> dict[str, Any]:
    rows = [json.loads(line) for line in ROWS.read_text(encoding="utf-8").splitlines() if line.strip()]
    clean = [row for row in rows if not row.get("technical_failure")]
    table: dict[str, Any] = {}
    for arm in sorted({row["arm"] for row in clean}):
        values = [row for row in clean if row["arm"] == arm]
        table[arm] = {
            "episodes": len(values),
            "mean_on_time": round(statistics.mean(float(row["task"]["on_time"]) for row in values), 4),
            "mean_expired": round(statistics.mean(float(row["task"]["expired"]) for row in values), 4),
            "mean_targets": round(statistics.mean(float(row["task"]["targets"]) for row in values), 4),
            "mean_airtime_bytes": round(statistics.mean(float(row["spent_bytes"]) for row in values), 1),
            "mean_attempts": round(
                statistics.mean(float(row["ledger"]["transmission_attempts"]) for row in values), 1
            ),
            "mean_deliveries_applied": round(
                statistics.mean(float(row["ledger"]["deliveries_applied"]) for row in values), 1
            ),
            "mean_duplicate_deliveries": round(
                statistics.mean(float(row["ledger"]["delivery_duplicate"]) for row in values), 1
            ),
            "seeds_with_budget_refusals": sum(
                1 for row in values if int(row["ledger"]["budget_refused_sends"]) > 0
            ),
            "mean_budget_refusals": round(
                statistics.mean(float(row["ledger"]["budget_refused_sends"]) for row in values), 1
            ),
            "mean_belief_age_steps": round(
                statistics.mean(float(row["belief"]["belief_age_steps_mean"] or 0.0) for row in values), 2
            ),
            # Null rather than zero: rows written before the measure existed simply lack it.
            "mean_unknown_to_everyone_steps": (
                round(
                    statistics.mean(
                        float(row["belief"]["unknown_to_everyone_steps"])
                        for row in values
                        if "unknown_to_everyone_steps" in row["belief"]
                    ),
                    1,
                )
                if any("unknown_to_everyone_steps" in row["belief"] for row in values)
                else None
            ),
            "mean_budget_used_fraction": round(
                statistics.mean(float(row["budget_used_fraction"] or 0.0) for row in values), 4
            ),
        }
    return {
        "episodes": len(rows),
        "technical_failures": len(rows) - len(clean),
        "locked": json.loads(LOCKED.read_text(encoding="utf-8")) if LOCKED.exists() else None,
        "tier_selection": pick_tier() if any("@75pct" in row["arm"] for row in clean) else None,
        "arms": table,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("reference", "lock", "tiered", "aggregate"),
        default="aggregate",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=list(common.TRAIN_SEEDS))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args(argv)
    if args.phase == "reference":
        return collect(list(RULES), args.seeds, HERE / "calib", args.shard, args.shards)
    if args.phase == "lock":
        print(json.dumps(lock(), ensure_ascii=False, indent=1))
        return 0
    if args.phase == "tiered":
        arms = [f"{name}@{tier}" for name in RULES for tier in TIERS]
        return collect(arms, args.seeds, HERE / "calib", args.shard, args.shards)
    payload = aggregate()
    common.write_json(HERE / "COMM_BUDGET_CALIBRATION.json", payload)
    for arm, values in payload["arms"].items():
        print(
            f"{arm:<24} eps={values['episodes']:>3} on_time={values['mean_on_time']:>6} "
            f"exp={values['mean_expired']:>5} bytes={values['mean_airtime_bytes']:>9.0f} "
            f"attempts={values['mean_attempts']:>6} applied={values['mean_deliveries_applied']:>6} "
            f"dup={values['mean_duplicate_deliveries']:>5} refused_seeds={values['seeds_with_budget_refusals']:>3} "
            f"age={values['mean_belief_age_steps']:>6} "
            f"unk={values['mean_unknown_to_everyone_steps'] if values['mean_unknown_to_everyone_steps'] is not None else '-':>7}"
        )
    if payload["tier_selection"]:
        print(json.dumps(payload["tier_selection"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
