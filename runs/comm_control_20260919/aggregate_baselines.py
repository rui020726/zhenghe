"""Aggregate the development rows: per-condition accounting and paired per-seed contrasts.

Units are carried everywhere because a count without a unit is a rumour:
targets, on-time targets, emissions (batches), packets, bytes, robot-task-steps, steps of
world time, milliseconds.

The contrasts are seed-level and paired -- every condition ran the same recorded detection
stream on the same seed, so the comparison is a per-seed difference, and its uncertainty is a
seed-level percentile bootstrap (B and the RNG seed fixed here, before looking at any row).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import common
import conditions

HERE = common.HERE
ROWS = HERE / "rows"

BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_SEED = 20260919

# (row path, display name) for the numbers a reader will be shown.
METRICS = {
    "performance.on_time": ("targets", "on-time targets completed before their deadline"),
    "performance.on_time_rate": ("ratio", "share of discovered targets completed on time"),
    "performance.expired": ("targets", "targets that expired"),
    "performance.unresolved": ("targets", "targets neither completed nor expired by the horizon"),
    "transport.emissions": ("batches", "task views put on the air"),
    "transport.transmission_attempts": (
        "packets",
        "one emission addressed to one subscribed robot, survived the channel or not",
    ),
    "transport.packets_lost": ("packets", "attempted packets the channel dropped"),
    "transport.deliveries_applied": ("packets", "deliveries that changed a belief"),
    "transport.airtime_bytes": ("bytes", "the billed cost: payload bytes x addressed robots, per attempt"),
    "transport.applied_bytes": ("bytes", "of that airtime, bytes that actually changed a belief"),
    "transport.refresh_emissions": ("batches", "emissions after the first for that target"),
    "transport.deliveries_duplicate": ("packets", "delivered but already believed (same time+revision)"),
    "transport.deliveries_superseded": ("packets", "delivered but older than what the robot believes"),
    "belief_all_channels.belief_age_steps_mean": ("steps", "mean receiver information age, all channels"),
    "belief_all_channels.stage_error_steps": ("robot-target-steps", "believed stage differs from the registry"),
    "belief_all_channels.requirement_error_steps": ("robot-target-steps", "believed remaining requirements differ"),
    "belief_all_channels.unknown_to_everyone_steps": ("task-steps", "active tasks no robot knew about"),
    "belief_via_uplink.belief_age_steps_mean": ("steps", "mean information age from uplink packets only"),
    "decisions.replan_decisions": ("decisions", "allocation decisions flagged as a replan"),
    "decisions.allocation_milliseconds_per_call": ("ms", "mean wall-clock per allocator call"),
    "decisions.allocation_seconds": ("s", "total wall-clock in the allocator"),
    "decisions.coordinator_seconds": ("s", "total wall-clock in the coordinator"),
    "decisions.idle_robot_steps": ("robot-steps", "robot-steps spent idle"),
}


def load_rows(directory: Path) -> list[dict[str, Any]]:
    """Every row written by the shards. One row per (condition, seed) or the batch is broken."""
    rows = []
    seen: set[tuple[str, int]] = set()
    for path in sorted(directory.glob("rows_shard*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["condition"], int(row["seed"]))
                if key in seen:
                    raise RuntimeError(
                        f"{key} appears twice in {directory}: the shards overlapped, so the "
                        "paired statistics would weight some seeds twice"
                    )
                seen.add(key)
                rows.append(row)
    return rows


def dig(row: dict[str, Any], dotted: str) -> Any:
    value: Any = row
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def failed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("technical_failure")]


def per_condition(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("technical_failure"):
            by_condition[row["condition"]].append(row)
    summary: dict[str, dict[str, Any]] = {}
    for name, values in by_condition.items():
        seeds = sorted(int(row["seed"]) for row in values)
        metrics: dict[str, Any] = {}
        for dotted in METRICS:
            observed = [dig(row, dotted) for row in values]
            numbers = [float(item) for item in observed if isinstance(item, (int, float))]
            if not numbers:
                continue
            metrics[dotted] = {
                "total": round(float(sum(numbers)), 4),
                "mean_per_episode": round(float(np.mean(numbers)), 4),
                "median": round(float(np.median(numbers)), 4),
                "max": round(float(max(numbers)), 4),
                "episodes_reporting": len(numbers),
            }
        summary[name] = {
            "role": values[0].get("role"),
            "information": values[0].get("information"),
            "policy": conditions.BY_NAME[name].policy if name in conditions.BY_NAME else None,
            "policy_parameters": (
                conditions.BY_NAME[name].params if name in conditions.BY_NAME else None
            ),
            "episodes": len(values),
            "seeds_first_last": [seeds[0], seeds[-1]] if seeds else None,
            "metrics": metrics,
        }
    return summary


def paired_contrast(
    rows: list[dict[str, Any]], *, baseline: str, metric: str, pool: Sequence[str] | None = None
) -> dict[str, Any]:
    """Seed-level paired difference with a percentile bootstrap over seeds."""
    allowed = None if pool is None else set(pool)
    left: dict[int, float] = {}
    right: dict[int, float] = {}
    for row in rows:
        if row.get("technical_failure"):
            continue
        value = dig(row, metric)
        if not isinstance(value, (int, float)):
            continue
        if row["condition"] == baseline:
            left[int(row["seed"])] = float(value)
        elif row["condition"] in conditions.BY_NAME:
            right.setdefault(int(row["seed"]), float(value))
    shared = sorted(set(left) & set(right))
    if not shared:
        return {"baseline": baseline, "metric": metric, "pairs": 0, "conditions": {}}
    by_condition_values: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        if row.get("technical_failure"):
            continue
        if allowed is not None and row["condition"] not in allowed:
            continue
        value = dig(row, metric)
        if isinstance(value, (int, float)):
            by_condition_values[row["condition"]][int(row["seed"])] = float(value)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    outcomes: dict[str, Any] = {}
    for name, values in by_condition_values.items():
        if name == baseline:
            continue
        deltas = np.array([values.get(seed, np.nan) - left[seed] for seed in shared], dtype=float)
        usable = ~np.isnan(deltas)
        if not usable.any():
            continue
        sample = deltas[usable]
        draws = rng.choice(sample, size=(BOOTSTRAP_REPLICATIONS, int(sample.size)), replace=True)
        means = draws.mean(axis=1)
        outcomes[name] = {
            "pairs": int(sample.size),
            "mean_difference": round(float(sample.mean()), 5),
            "ci95": [round(float(np.percentile(means, 2.5)), 5), round(float(np.percentile(means, 97.5)), 5)],
            "seeds_higher": int((sample > 0).sum()),
            "seeds_lower": int((sample < 0).sum()),
            "seeds_equal": int((sample == 0).sum()),
        }
    return {"baseline": baseline, "metric": metric, "shared_seeds": len(shared), "conditions": outcomes}


def headroom(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    pool: Sequence[str],
    reference: str = "isolated_immediate",
) -> dict[str, Any]:
    """What the best *fixed* rule per seed buys over the reference rule.

    Ties are not handed to whoever was simulated first: among the rules that reach the best
    value on a seed, the cheapest in packets wins, and a seed where several rules tied is
    counted. That cost is the point of the number -- a ceiling that costs 8000 packets is not
    a prize a controller would be trained for.
    """
    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    cost: dict[tuple[str, int], float] = {}
    for row in rows:
        if row.get("technical_failure"):
            continue
        value = dig(row, metric)
        if isinstance(value, (int, float)):
            by_seed[int(row["seed"])][row["condition"]] = float(value)
            cost[(row["condition"], int(row["seed"]))] = float(
                dig(row, "transport.transmission_attempts") or 0.0
            )
    winners: dict[str, int] = defaultdict(int)
    gains: list[float] = []
    winner_costs: list[float] = []
    tied_seeds = 0
    worse_seeds = 0
    for seed, values in sorted(by_seed.items()):
        available = {name: value for name, value in values.items() if name in pool}
        if not available or reference not in values:
            continue
        best = max(available.values())
        tied = [name for name, value in available.items() if value == best]
        tied_seeds += int(len(tied) > 1)
        winner = min(tied, key=lambda name: cost.get((name, seed), float("inf")))
        winners[winner] += 1
        winner_costs.append(cost.get((winner, seed), 0.0))
        gains.append(best - values[reference])
        worse_seeds += int(min(available.values()) < values[reference])
    return {
        "metric": metric,
        "pool": list(pool),
        "reference": reference,
        "seeds": len(gains),
        "mean_gain_over_reference": round(float(np.mean(gains)), 5) if gains else None,
        "seeds_where_a_rule_ties_at_the_top": tied_seeds,
        "seeds_where_any_rule_is_worse_than_reference": worse_seeds,
        "winners_by_condition": dict(sorted(winners.items(), key=lambda item: -item[1])),
        "mean_packets_of_the_winning_rule": (
            round(float(np.mean(winner_costs)), 1) if winner_costs else None
        ),
    }


def spread(rows: list[dict[str, Any]], *, metric: str, pool: Sequence[str]) -> dict[str, Any]:
    """How many seeds the rules actually disagree on, and how wide the disagreement is.

    A dev set on which every rule finishes every target proves nothing about headroom in
    either direction, so a ceiling claim has to be reported next to this number.
    """
    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("technical_failure") or row["condition"] not in pool:
            continue
        value = dig(row, metric)
        if isinstance(value, (int, float)):
            by_seed[int(row["seed"])][row["condition"]] = float(value)
    widths = [max(v.values()) - min(v.values()) for v in by_seed.values() if v]
    tied = [sum(1 for x in v.values() if x == max(v.values())) for v in by_seed.values() if v]
    return {
        "metric": metric,
        "seeds": len(widths),
        "seeds_where_rules_disagree": int(sum(1 for value in widths if value > 0)),
        "seeds_every_rule_agrees": int(sum(1 for value in widths if value == 0)),
        "mean_spread": round(float(np.mean(widths)), 4) if widths else None,
        "max_spread": round(float(max(widths)), 4) if widths else None,
        "mean_rules_achieving_the_best": round(float(np.mean(tied)), 2) if tied else None,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default=str(ROWS))
    parser.add_argument("--out", default=str(HERE / "COMMUNICATION_BASELINE_STATS.json"))
    parser.add_argument("--baseline", default="isolated_immediate")
    args = parser.parse_args(argv)

    rows = load_rows(Path(args.rows))
    present = {row["condition"] for row in rows}
    primary = [
        name
        for name in present
        if name in conditions.BY_NAME and conditions.BY_NAME[name].role != "diagnostic"
    ]
    diagnostics = [
        name
        for name in present
        if name in conditions.BY_NAME and conditions.BY_NAME[name].role == "diagnostic"
    ]
    uplink_only = [
        name
        for name in primary
        if name in conditions.BY_NAME and conditions.BY_NAME[name].information == "isolated"
    ]
    compared = (
        "performance.on_time",
        "performance.expired",
        "performance.unresolved",
        "transport.transmission_attempts",
        "transport.airtime_bytes",
        "transport.refresh_emissions",
        "transport.deliveries_duplicate",
        "belief_all_channels.belief_age_steps_mean",
        "belief_all_channels.requirement_error_steps",
        "belief_all_channels.unknown_to_everyone_steps",
        "decisions.replan_decisions",
        "decisions.allocation_seconds",
    )
    payload = {
        "generated_from": str(Path(args.rows)),
        "episodes": len(rows),
        "conditions": len(present),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "primary_conditions": sorted(primary),
        "diagnostic_conditions": sorted(diagnostics),
        "units": {key: value[0] for key, value in METRICS.items()},
        "definitions": {key: value[1] for key, value in METRICS.items()},
        "bootstrap": {
            "replications": BOOTSTRAP_REPLICATIONS,
            "rng_seed": BOOTSTRAP_SEED,
            "scheme": "seed-level paired percentile bootstrap, 95% CI",
        },
        "technical_failures": [
            {"condition": row["condition"], "seed": row["seed"], "error": row["technical_failure"]}
            for row in failed_rows(rows)
        ],
        "zero_detection_seeds": sorted(
            {
                int(row["seed"])
                for row in rows
                if not row.get("technical_failure")
                and dig(row, "performance.targets") == 0
            }
        ),
        "per_condition": per_condition(rows),
        "paired": {
            "primary_vs_isolated_immediate": [
                paired_contrast(rows, baseline=args.baseline, metric=metric, pool=primary)
                for metric in compared
            ],
            "primary_vs_deployed_immediate": [
                paired_contrast(rows, baseline="deployed_immediate", metric=metric, pool=primary)
                for metric in ("performance.on_time", "transport.transmission_attempts")
            ],
            "diagnostic_vs_deployed_immediate": [
                paired_contrast(rows, baseline="deployed_immediate", metric=metric, pool=diagnostics)
                for metric in ("performance.on_time", "transport.transmission_attempts")
            ]
            if diagnostics
            else [],
        },
        "spread": [
            spread(rows, metric=metric, pool=primary)
            for metric in ("performance.on_time", "transport.transmission_attempts")
        ],
        "headroom": [
            headroom(rows, metric=metric, pool=pool)
            for metric in ("performance.on_time", "performance.on_time_rate")
            for pool in (uplink_only, sorted(primary))
        ],
    }
    common.write_json(Path(args.out), payload)
    print(
        f"episodes={payload['episodes']} conditions={payload['conditions']} "
        f"technical_failures={len(payload['technical_failures'])} -> {args.out}"
    )
    for name, values in sorted(payload["per_condition"].items()):
        metrics = values["metrics"]
        print(
            f"{name:<28} eps={values['episodes']:>3} "
            f"on_time={metrics.get('performance.on_time', {}).get('total')} "
            f"expired={metrics.get('performance.expired', {}).get('total')} "
            f"emissions={metrics.get('transport.emissions', {}).get('total'):>7} "
            f"attempts={metrics.get('transport.transmission_attempts', {}).get('total'):>6} "
            f"bytes={metrics.get('transport.airtime_bytes', {}).get('total'):>12.0f} "
            f"age={metrics.get('belief_all_channels.belief_age_steps_mean', {}).get('mean_per_episode')}"
        )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
