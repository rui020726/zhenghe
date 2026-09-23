"""Aggregate one pressure layer's 144 episodes into the quantities §5 of the protocol names.

Three comparisons are kept separate on purpose, because round 3 learned that merging them is how
a conclusion gets manufactured: the sample-mean frontier, the per-stream dominance counts, and
the paired confidence intervals can each say something different about the same nine arms. The
hindsight envelope is a per-stream maximum over the seven competitors -- an experience bound, not
a strategy -- and it is labelled as such everywhere it appears.

    python pressure_metrics.py --goals 6
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from typing import Any, Callable

import pressure_common as common
import pressure_matrix

#: The axes that get their own single-draw bootstrap block (protocol §6).
AXES: dict[str, Callable[[dict[str, Any]], float]] = {
    "on_time_per_episode": lambda row: float(row["on_time"]),
    "bytes_per_episode": lambda row: float(row["spent_bytes"]),
}

BASELINES: tuple[str, ...] = ("always_hold", "best_fixed")


def load(goals: int) -> list[dict[str, Any]]:
    rows = pressure_matrix.load_rows(goals)
    expected = pressure_matrix.cells(goals)
    missing = [f"{arm} s{seed}" for arm, seed in expected if (arm, seed) not in rows]
    if missing:
        raise SystemExit(
            f"layer {goals} is incomplete: {len(missing)} cells missing, first "
            f"{missing[:5]}; aggregates are never taken over a partial matrix"
        )
    return [rows[key] for key in expected]


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 5) if values else 0.0


def _quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": _mean(ordered),
        "min": round(ordered[0], 4),
        "p25": round(ordered[int(0.25 * (len(ordered) - 1))], 4),
        "median": round(statistics.median(ordered), 4),
        "p75": round(ordered[int(0.75 * (len(ordered) - 1))], 4),
        "max": round(ordered[-1], 4),
    }


def arm_tables(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per arm: means over the 16 streams, and the per-stream maps the statistics run on."""
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[str(row["arm"])].append(row)
    tables: dict[str, dict[str, Any]] = {}
    for arm, values in by_arm.items():
        probes = [row["probe"] or {} for row in values]
        tables[arm] = {
            "role": common.ARM_ROLES[arm],
            "streams": len(values),
            "on_time_per_episode": _mean([float(row["on_time"]) for row in values]),
            "bytes_per_episode": _mean([float(row["spent_bytes"]) for row in values]),
            "on_time_rate_of_detected": _mean(
                [
                    float(row["on_time"] / row["detected"])
                    for row in values
                    if int(row["detected"]) > 0
                ]
            ),
            "per_stream_on_time": {int(row["seed"]): float(row["on_time"]) for row in values},
            "per_stream_bytes": {int(row["seed"]): float(row["spent_bytes"]) for row in values},
            "per_stream_on_time_rate": {
                int(row["seed"]): (
                    round(float(row["on_time"]) / float(row["detected"]), 5)
                    if int(row["detected"])
                    else None
                )
                for row in values
            },
            "per_stream_budget_fraction": {
                int(row["seed"]): row["budget_used_fraction"] for row in values
            },
            "credit_used_ge_90_streams": sum(
                1
                for row in values
                if row["budget_used_fraction"] is not None
                and float(row["budget_used_fraction"]) >= 0.90
            ),
            "emissions_per_episode": _mean([float(row["emissions"]) for row in values]),
            "emission_types": {
                key: sum(int(row["emission_types"][key]) for row in values)
                for key in ("first", "refresh")
            },
            "send_step_quantiles": _quantiles(
                [float(step) for row in values for step in row["send_steps"]]
            ),
            "budget_refusals_per_episode": _mean(
                [float(row["refusals"]["budget_exhausted"]) for row in values]
            ),
            "capacity_refusals_per_episode": _mean(
                [float(row["refusals"]["capacity_step"]) for row in values]
            ),
            "first_budget_refusal_step_quantiles": _quantiles(
                [
                    float(row["refusals"]["first_budget_refusal_step"])
                    for row in values
                    if row["refusals"]["first_budget_refusal_step"] is not None
                ]
            ),
            "competition_streams": sum(
                1 for probe in probes if int(probe.get("competition_points", 0)) > 0
            ),
            "competition_points_quantiles": _quantiles(
                [float(probe.get("competition_points", 0)) for probe in probes]
            ),
            "competition_points_total": sum(
                int(probe.get("competition_points", 0)) for probe in probes
            ),
            "joint_only_points_total": sum(
                int(probe.get("joint_only_points", 0)) for probe in probes
            ),
            "competition_points_credit_cause_total": sum(
                int(probe.get("competition_points_credit_cause", 0)) for probe in probes
            ),
            "competition_points_capacity_only_total": sum(
                int(probe.get("competition_points_capacity_only", 0)) for probe in probes
            ),
            "payable_candidate_points_total": sum(
                int(probe.get("payable_total", 0)) for probe in probes
            ),
            "per_stream_competition_points": {
                int(row["seed"]): int((row["probe"] or {}).get("competition_points", 0))
                for row in values
            },
            "credit_exhausted_with_candidates_streams": sum(
                1 for probe in probes if int(probe.get("credit_blocked_steps", 0)) > 0
            ),
            "mean_steps_offered": _mean([float(probe.get("steps_offered", 0)) for probe in probes]),
            "mean_visible_at_competition": _mean(
                [
                    float(probe["mean_visible_at_competition"])
                    for probe in probes
                    if probe.get("mean_visible_at_competition") is not None
                ]
            ),
            "mean_payable_at_competition": _mean(
                [
                    float(probe["mean_payable_at_competition"])
                    for probe in probes
                    if probe.get("mean_payable_at_competition") is not None
                ]
            ),
            "mean_slack_at_competition": _mean(
                [
                    float(probe["mean_slack_at_competition"])
                    for probe in probes
                    if probe.get("mean_slack_at_competition") is not None
                ]
            ),
            "belief_age_mean": _mean(
                [
                    float(row["belief"].get("belief_age_steps_mean") or 0.0)
                    for row in values
                    if int(row["emissions"]) > 0
                ]
            ),
            "traffic": {
                key: sum(int(row["ledger"][key]) for row in values)
                for key in (
                    "transmission_attempts",
                    "deliveries_applied",
                    "delivery_duplicate",
                    "delivery_superseded",
                    "generated_items",
                    "redundant_emissions",
                    "coalesced_changes",
                )
                if key in values[0]["ledger"]
            },
        }
    return tables


def best_fixed_rule(arms: dict[str, dict[str, Any]]) -> str:
    """Most on-time per episode; tie goes to fewer bytes; then the protocol's row order."""
    competitors = [arm for arm in common.COMPETITION_ARMS if arm in arms]
    if not competitors:
        raise RuntimeError("no competitor arm ran on this layer; the best rule cannot be chosen")
    return min(
        competitors,
        key=lambda arm: (
            -float(arms[arm]["on_time_per_episode"]),
            float(arms[arm]["bytes_per_episode"]),
            common.ARM_ORDER.index(arm),
        ),
    )


def envelope(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per stream: the best of the seven competitors, most on-time then fewer bytes."""
    competitors = [arm for arm in common.COMPETITION_ARMS if arm in arms]
    seeds = sorted(int(seed) for seed in arms[competitors[0]]["per_stream_on_time"])
    per_stream: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        ranked = sorted(
            competitors,
            key=lambda arm: (
                -float(arms[arm]["per_stream_on_time"][seed]),
                float(arms[arm]["per_stream_bytes"][seed]),
                common.ARM_ORDER.index(arm),
            ),
        )
        winner = ranked[0]
        tied = [
            arm
            for arm in ranked
            if float(arms[arm]["per_stream_on_time"][seed])
            == float(arms[winner]["per_stream_on_time"][seed])
            and float(arms[arm]["per_stream_bytes"][seed])
            == float(arms[winner]["per_stream_bytes"][seed])
        ]
        per_stream[seed] = {
            "on_time": float(arms[winner]["per_stream_on_time"][seed]),
            "bytes": float(arms[winner]["per_stream_bytes"][seed]),
            "winners": sorted(tied),
            "unique_winner": len(tied) == 1,
        }
    winners_count: dict[str, int] = defaultdict(int)
    unique_winners_count: dict[str, int] = defaultdict(int)
    for entry in per_stream.values():
        for arm in entry["winners"]:
            winners_count[arm] += 1
        if entry["unique_winner"]:
            unique_winners_count[entry["winners"][0]] += 1
    return {
        "definition": (
            "per-stream maximum of the seven budgeted competitors on on-time targets, ties broken "
            "towards fewer bytes; an experience bound, not a deployable strategy"
        ),
        "per_stream": {str(seed): entry for seed, entry in per_stream.items()},
        "per_stream_on_time": {seed: entry["on_time"] for seed, entry in per_stream.items()},
        "per_stream_bytes": {seed: entry["bytes"] for seed, entry in per_stream.items()},
        "mean_on_time": _mean([entry["on_time"] for entry in per_stream.values()]),
        "mean_bytes": _mean([entry["bytes"] for entry in per_stream.values()]),
        "unique_winner_counts": dict(sorted(unique_winners_count.items())),
        "any_winner_counts": dict(sorted(winners_count.items())),
    }


def value_maps(
    arms: dict[str, dict[str, Any]], envelope_data: dict[str, Any], best: str
) -> dict[str, dict[int, float]]:
    """Label -> {seed: value} for every label the bootstrap walks, including the two composites."""
    values: dict[str, dict[int, float]] = {}
    for arm, table in arms.items():
        values[arm] = {
            int(seed): float(value)
            for seed, value in table["per_stream_on_time"].items()
        }
    values["envelope"] = {int(k): float(v) for k, v in envelope_data["per_stream_on_time"].items()}
    values["best_fixed"] = dict(values[best])
    return values


def bootstrap_block(
    arms: dict[str, dict[str, Any]], envelope_data: dict[str, Any], best: str, axis: str
) -> dict[str, Any]:
    """One draw per (axis, baseline); the returned dict is sliced by every downstream table."""
    maps = (
        value_maps(arms, envelope_data, best)
        if axis == "on_time_per_episode"
        else _byte_maps(arms, envelope_data, best)
    )
    return {
        baseline: common.paired_differences(maps, baseline=baseline)
        for baseline in BASELINES
    }


def _byte_maps(
    arms: dict[str, dict[str, Any]], envelope_data: dict[str, Any], best: str
) -> dict[str, dict[int, float]]:
    maps = {
        arm: {int(seed): float(value) for seed, value in table["per_stream_bytes"].items()}
        for arm, table in arms.items()
    }
    maps["envelope"] = {int(k): float(v) for k, v in envelope_data["per_stream_bytes"].items()}
    maps["best_fixed"] = dict(maps[best])
    return maps


def pareto_frontier(arms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Sample-mean Pareto over the competitors: most on-time, fewest bytes."""
    points = [
        {
            "arm": arm,
            "on_time_per_episode": table["on_time_per_episode"],
            "bytes_per_episode": table["bytes_per_episode"],
        }
        for arm, table in arms.items()
        if table["role"] == "competitor"
    ]
    front = [
        point
        for point in points
        if not any(
            other["arm"] != point["arm"]
            and float(other["on_time_per_episode"]) >= float(point["on_time_per_episode"])
            and float(other["bytes_per_episode"]) <= float(point["bytes_per_episode"])
            and (
                float(other["on_time_per_episode"]) > float(point["on_time_per_episode"])
                or float(other["bytes_per_episode"]) < float(point["bytes_per_episode"])
            )
            for other in points
        )
    ]
    return sorted(front, key=lambda point: float(point["bytes_per_episode"]))


def per_stream_dominance(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """How often each arm is strictly better than each other arm on a single stream."""
    competitors = [arm for arm in common.COMPETITION_ARMS if arm in arms]
    seeds = sorted(int(seed) for seed in arms[competitors[0]]["per_stream_on_time"])
    pairwise: dict[str, dict[str, int]] = {}
    for left in competitors:
        for right in competitors:
            if left >= right:
                continue
            left_better = sum(
                1
                for seed in seeds
                if float(arms[left]["per_stream_on_time"][seed])
                > float(arms[right]["per_stream_on_time"][seed])
            )
            right_better = sum(
                1
                for seed in seeds
                if float(arms[right]["per_stream_on_time"][seed])
                > float(arms[left]["per_stream_on_time"][seed])
            )
            key = f"{left} vs {right}"
            pairwise[key] = {
                "left_strictly_more_on_time": left_better,
                "right_strictly_more_on_time": right_better,
                "bidirectional": left_better >= 3 and right_better >= 3,
            }
    return pairwise


def incomparable_pairs(dominance: dict[str, Any], arms: dict[str, dict[str, Any]]) -> list[str]:
    """Pairs that neither dominates at the layer mean, so a double win is a real trade-off."""
    pairs: list[str] = []
    for key, value in dominance.items():
        left, right = (part.strip() for part in key.split(" vs "))
        a, b = arms[left], arms[right]
        dominates = (
            float(a["on_time_per_episode"]) >= float(b["on_time_per_episode"])
            and float(a["bytes_per_episode"]) <= float(b["bytes_per_episode"])
        ) or (
            float(b["on_time_per_episode"]) >= float(a["on_time_per_episode"])
            and float(b["bytes_per_episode"]) <= float(a["bytes_per_episode"])
        )
        if not dominates and value["bidirectional"]:
            pairs.append(key)
    return sorted(pairs)


def lambda_slopes(front: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Adjacent-point break-even rate, with the undefined case kept visible."""
    slopes: list[dict[str, Any]] = []
    for cheaper, dearer in zip(front, front[1:]):
        delta_effect = float(dearer["on_time_per_episode"]) - float(cheaper["on_time_per_episode"])
        delta_bytes = float(dearer["bytes_per_episode"]) - float(cheaper["bytes_per_episode"])
        denominator = delta_bytes / float(budget)
        slopes.append(
            {
                "from": cheaper["arm"],
                "to": dearer["arm"],
                "delta_on_time_per_episode": round(delta_effect, 5),
                "delta_bytes_per_episode": round(delta_bytes, 1),
                "delta_budget_fraction": round(denominator, 5),
                "lambda_break_even": (
                    "undefined"
                    if denominator <= 0
                    else round(delta_effect / denominator, 4)
                ),
                "rule": (
                    "targets gained per unit of the 29 300 B credit spent; undefined when the "
                    "dearer point is not actually dearer"
                ),
            }
        )
    return slopes


def sender_visible_features(rows: list[dict[str, Any]], envelope_data: dict[str, Any]) -> dict[str, Any]:
    """Descriptive only: does the winner of a stream differ in what the sender can see?

    No classifier is fitted and no receiver-side belief or ground truth is used -- the point is to
    see whether the existing 19-dim observation could distinguish the cases at all.
    """
    by_winner: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_seed = {int(row["seed"]): row for row in rows if row["arm"] == common.COMPETITION_ARMS[0]}
    for seed, entry in envelope_data["per_stream"].items():
        seed = int(seed)
        if not entry["unique_winner"] or seed not in by_seed:
            continue
        arm = entry["winners"][0]
        probe = (by_seed[seed]["probe"] or {})
        targets = by_seed[seed]["targets"]
        slack = [
            float(row["deadline_slack_before_work"])
            for row in targets
            if row.get("deadline_slack_before_work") is not None
        ]
        by_winner[arm]["streams"].append(1.0)
        by_winner[arm]["placed"].append(float(len(targets)))
        by_winner[arm]["mean_slack"].append(_mean(slack))
        by_winner[arm]["max_backlog"].append(float(probe.get("max_backlog_observed", 0)))
        by_winner[arm]["visible_at_competition"].append(
            float(probe.get("mean_visible_at_competition") or 0.0)
        )
        by_winner[arm]["credit_used_fraction"].append(
            float(by_seed[seed]["budget_used_fraction"] or 0.0)
        )
    return {
        "definition": (
            "streams where exactly one competitor is the per-stream winner, grouped by that winner; "
            "means of sender-visible quantities only, descriptive, no model fitted"
        ),
        "groups": {
            arm: {
                key: _mean(values) if key != "streams" else int(sum(values))
                for key, values in sorted(fields.items())
            }
            for arm, fields in sorted(by_winner.items())
        },
        "streams_without_unique_winner": sum(
            1 for entry in envelope_data["per_stream"].values() if not entry["unique_winner"]
        ),
    }


def step_geometry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The g12 extension protocol §5 diagnostics, per arm, from the probe counters already on disk.

    A step with no candidate never reaches the probe (``comm.py:700`` returns before ``decide``),
    so the zero-candidate count is the episode's steps minus the steps the probe was offered --
    exact, because ``run_episode`` asserts the step count. The four multi-candidate outcomes plus
    ``competition_points`` and ``joint_only_points`` are an exhaustive partition of the
    multi-candidate steps, which is what makes "the credit never offered a subset to choose" a
    statement about counts rather than a reading.
    """
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[str(row["arm"])].append(row)
    out: dict[str, dict[str, Any]] = {}
    for arm, values in by_arm.items():
        steps = sum(int(row["steps"]) for row in values)
        probes = [row["probe"] or {} for row in values]
        offered = sum(int(probe.get("steps_offered", 0)) for probe in probes)
        one = sum(int(probe.get("steps_with_one_candidate", 0)) for probe in probes)
        multi = sum(int(probe.get("steps_with_multiple_candidates", 0)) for probe in probes)
        competition = sum(int(probe.get("competition_points", 0)) for probe in probes)
        joint = sum(int(probe.get("joint_only_points", 0)) for probe in probes)
        within = sum(int(probe.get("multi_all_payable_within_credit_steps", 0)) for probe in probes)
        none_payable = sum(int(probe.get("multi_none_payable_steps", 0)) for probe in probes)
        unclassified = sum(int(probe.get("multi_unclassifiable_steps", 0)) for probe in probes)
        priced = [probe.get("multi_candidate_prices") or {} for probe in probes]
        counted = [entry for entry in priced if int(entry.get("count", 0) or 0)]
        minima = [int(entry["min"]) for entry in counted]
        maxima = [int(entry["max"]) for entry in counted]
        prices_total = sum(int(entry["sum"]) for entry in counted)
        price_count = sum(int(entry["count"]) for entry in counted)
        out[arm] = {
            "episodes": len(values),
            "steps": steps,
            #: The three delivered layers ran before the geometry counters existed. Their L0-L4
            #: verdicts are untouched by that; only this diagnostic is asymmetric, and saying so
            #: is what keeps a g12 distribution from being read back into a delivered layer.
            "geometry_recorded": any("steps_with_multiple_candidates" in probe for probe in probes),
            "candidate_histogram": {
                "zero": steps - offered,
                "one": one,
                "two_or_more": multi,
            },
            "multi_step_outcomes": {
                "all_payable_within_credit": within,
                "every_candidate_payable_but_not_jointly": joint,
                "strict_subset_payable": competition,
                "none_payable": none_payable,
                "unclassifiable": unclassified,
                "closes_on_multi_candidate_steps": within + joint + competition + none_payable
                + unclassified
                == multi,
            },
            "multi_candidate_price_bytes": {
                "candidates_priced": price_count,
                "min": min(minima) if minima else None,
                "max": max(maxima) if maxima else None,
                "mean": round(prices_total / price_count, 1) if price_count else None,
                "spread": (max(maxima) - min(minima)) if minima else None,
                "spread_as_fraction_of_credit": (
                    round((max(maxima) - min(minima)) / common.BUDGET_BYTES, 5) if minima else None
                ),
            },
            "first_credit_exhaustion": [
                probe.get("first_credit_exhaustion_facts")
                for probe in probes
                if probe.get("first_credit_exhaustion_facts")
            ],
        }
    return out


def layer_report(goals: int) -> dict[str, Any]:
    rows = load(goals)
    arms = arm_tables(rows)
    best = best_fixed_rule(arms)
    env = envelope(arms)
    front = pareto_frontier(arms)
    dominance = per_stream_dominance(arms)
    return {
        "goals": int(goals),
        "budget_bytes": common.BUDGET_BYTES,
        "streams": len({int(row["seed"]) for row in rows}),
        "arms": arms,
        "best_fixed_rule": {
            "arm": best,
            "rule": "most on-time per episode; tie to fewer bytes; tie to the protocol's row order",
        },
        "hindsight_envelope": env,
        "mean_pareto_frontier": front,
        "per_stream_pairwise": dominance,
        "incomparable_bidirectional_pairs": incomparable_pairs(dominance, arms),
        "step_geometry": step_geometry(rows),
        "lambda_break_even_adjacent_front": lambda_slopes(front, common.BUDGET_BYTES),
        "sender_visible_features": sender_visible_features(rows, env),
        "bootstrap": {
            axis: bootstrap_block(arms, env, best, axis) for axis in AXES
        },
        "bootstrap_audit": {
            "seed": common.BOOTSTRAP_SEED,
            "replications": common.BOOTSTRAP_REPLICATIONS,
            "label_order": list(common.BOOTSTRAP_LABEL_ORDER),
            "calls": [
                {"axis": axis, "baseline": baseline}
                for axis in AXES
                for baseline in BASELINES
            ],
            "rule": "one call per (layer, axis, baseline); every CI in the report is sliced from it",
        },
    }


def search_pressure(goals: int) -> dict[str, Any]:
    """The layer's search-side facts, which belong to the streams and not to any arm."""
    rows = load(goals)
    first = {int(row["seed"]): row for row in rows if row["arm"] == "send_once"}
    per_stream = {}
    for seed, row in sorted(first.items()):
        targets = row["targets"]
        detected_at = [float(entry["detected_at"]) for entry in targets]
        per_stream[str(seed)] = {
            "placed": int(row["placed"]),
            "detected": int(row["detected"]),
            "registered": int(row["registered"]),
            "detected_over_placed": round(int(row["detected"]) / float(row["placed"] or 1), 5),
            "last_detection_step": row["stream_last_detection_step"],
            "mean_detection_time": _mean(detected_at),
            "on_time": int(row["on_time"]),
            "expired": int(row["expired"]),
            "unresolved": int(row["unresolved"]),
            **row["occupancy"],
        }
    ratios = [entry["detected_over_placed"] for entry in per_stream.values()]
    return {
        "goals": int(goals),
        "streams": len(per_stream),
        "per_stream": per_stream,
        "mean_detected_over_placed": _mean(ratios),
        "worst_stream_detected_over_placed": min(ratios) if ratios else None,
        "worst_stream": (
            min(per_stream, key=lambda key: per_stream[key]["detected_over_placed"])
            if per_stream
            else None
        ),
        "illegal_streams": sorted(
            seed for seed, entry in per_stream.items() if entry["placed"] != int(goals)
        ),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goals", type=int, required=True, choices=sorted(common.LAYERS))
    args = parser.parse_args()
    report = layer_report(args.goals)
    # Per-layer files: the combined RULE_MATRIX.json / PRESSURE_METRICS.json are written by
    # report_pressure.py over every layer that ran, so re-aggregating one layer cannot quietly
    # replace the whole round's matrix with one layer's.
    common.write_json(common.json_path(f"RULE_MATRIX_g{args.goals}.json"), report)
    common.write_json(
        common.json_path(f"PRESSURE_METRICS_g{args.goals}.json"), search_pressure(args.goals)
    )
    print(
        json.dumps(
            {
                "goals": args.goals,
                "best_fixed_rule": report["best_fixed_rule"]["arm"],
                "best_on_time": report["arms"][report["best_fixed_rule"]["arm"]][
                    "on_time_per_episode"
                ],
                "best_bytes": report["arms"][report["best_fixed_rule"]["arm"]][
                    "bytes_per_episode"
                ],
                "envelope_on_time": report["hindsight_envelope"]["mean_on_time"],
                "frontier": [
                    (point["arm"], point["on_time_per_episode"], point["bytes_per_episode"])
                    for point in report["mean_pareto_frontier"]
                ],
                "unique_winners": report["hindsight_envelope"]["unique_winner_counts"],
                "competition_streams": {
                    arm: table["competition_streams"] for arm, table in report["arms"].items()
                },
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
