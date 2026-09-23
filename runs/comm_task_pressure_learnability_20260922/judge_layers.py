"""L0-L4 for one pressure layer, applied mechanically to the numbers the metrics module produced.

Every threshold here is copied from the protocol, which was committed before any stream of this
layer existed. Each gate returns its evidence next to its verdict, including the readings that
fail: the point of a pre-registered gate is that it can say no, and a report that only shows the
passing side of a boundary is not evidence.

    python judge_layers.py --goals 6
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pressure_common as common
import pressure_metrics
import pressure_matrix

#: Protocol §7 constants, in one place so a later reader can see nothing was retuned.
L1_MIN_DETECTED_RATIO = 0.75
L2_ON_TIME_BAND = (0.40, 0.90)
L3_MIN_COMPETITION_STREAMS = 8
L3_MIN_CREDIT_HEAVY_STREAMS = 4
L3_MIN_CREDIT_FRACTION = 0.90
L4_MIN_ENVELOPE_GAIN = 0.25
L4_MIN_UNIQUE_WINNER_STREAMS = 3
L4_MIN_RULES_WITH_UNIQUE_WINS = 2
L4_MIN_BIDIRECTIONAL_STREAMS = 3
L4_MIN_FRONTIER_POINTS = 2
L4_MIN_FRONTIER_EFFECT_GAP = 0.25
L4_MIN_FRONTIER_BYTE_GAP = 2_930

#: The configuration keys a cell may differ on. Anything else is a mixed matrix, not an arm.
ALLOWED_CELL_DIFFERENCES = ("comm_policy", "comm_policy_params", "comm_budget_bytes")


def ledger_failures(goals: int) -> dict[str, Any]:
    """Split the ledger's failures into recovered and unrecovered, mechanically.

    A launch that crashed and was later re-run successfully for the same (kind, seed) still spent
    an episode, and that cost stays in the totals. What L0 asks is whether any data this layer
    aggregates came from a launch that never produced a good counterpart.
    """
    entries: list[dict[str, Any]] = []
    for path in sorted(common.HERE.glob(common.LEDGER_PATTERN)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload.get("entries", []):
            if int(entry.get("goals", -1)) == int(goals):
                entries.append(entry)
    succeeded = {
        (str(entry["kind"]), int(entry["seed"])) for entry in entries if entry["ok"]
    }
    failed = [(str(entry["kind"]), int(entry["seed"]), str(entry["detail"])) for entry in entries if not entry["ok"]]
    unrecovered = [
        {"kind": kind, "seed": seed, "detail": detail}
        for kind, seed, detail in failed
        if (kind, seed) not in succeeded
    ]
    return {
        "entries": len(entries),
        "failures": len(failed),
        "recovered": [
            {"kind": kind, "seed": seed, "detail": detail}
            for kind, seed, detail in failed
            if (kind, seed) in succeeded
        ],
        "unrecovered": unrecovered,
        "matrix_failures": [item for item in failed if item[0] == "matrix"],
    }


def _verdict(name: str, passed: bool, evidence: dict[str, Any], meaning: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "meaning": meaning, "evidence": evidence}


def l0(goals: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Data and interface trustworthiness."""
    coverage = pressure_matrix.coverage(goals)
    manifest = json.loads(
        common.json_path(f"STREAM_MANIFEST_g{goals}.json").read_text(encoding="utf-8")
    )
    ledger = common.ledger_totals()
    failure_account = ledger_failures(goals)
    provenance = pressure_matrix.row_provenance(goals)
    agreement = (
        json.loads(common.json_path("AGREEMENT_GATE.json").read_text(encoding="utf-8"))
        if common.json_path("AGREEMENT_GATE.json").exists()
        else {}
    )
    per_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        per_seed.setdefault(int(row["seed"]), []).append(row)
    wrong_arm_count = sum(
        1
        for group in per_seed.values()
        if sorted(str(row["arm"]) for row in group) != sorted(common.ARM_ORDER)
    )
    over_budget = [
        f"{row['arm']} s{row['seed']}"
        for row in rows
        if row["budget_bytes"] is not None and int(row["spent_bytes"]) > int(row["budget_bytes"])
    ]
    closure_failures = [
        f"{row['arm']} s{row['seed']}"
        for row in rows
        if int(row["emission_types"]["first"]) + int(row["emission_types"]["refresh"])
        != int(row["emissions"])
        or int(row["ledger"]["emissions"]) != int(row["emissions"])
    ]
    refusal_mismatches = [
        f"{row['arm']} s{row['seed']}"
        for row in rows
        if int(row["refusals"]["budget_exhausted"])
        + int(row["refusals"]["capacity_step"])
        != int(row["refusals"]["total"])
        or int(row["refusals"]["budget_exhausted"])
        != int(row["ledger"]["budget_refused_sends"])
        or int(row["refusals"]["capacity_step"])
        != int(row["ledger"]["capacity_refused_sends"])
    ]
    steps = sorted({int(row["steps"]) for row in rows})
    probed = sorted({bool(row["probed"]) for row in rows})
    mixed_settings = {
        str(seed): sorted({str(row["setting_signature"]) for row in group})
        for seed, group in per_seed.items()
        if len({str(row["setting_signature"]) for row in group}) != 1
    }
    # The transport facts every arm of every stream must agree on. The raw setting signature cannot
    # be compared layer-wide, because it contains the seed and so differs per stream by design:
    # asking for one signature across 144 cells fails for a reason that has nothing to do with
    # whether the arms share a configuration.
    invariants = sorted(
        {
            (
                str(row["timing_mode"]),
                str(row["delivery_probability"]),
                str(row["ack_enabled"]),
                str(row["steps"]),
                str(row["bound_enforced_by_transport"]),
                str(row["goals"]),
                str(row["probed"]),
            )
            for row in rows
        }
    )
    placements = sorted(
        {int(entry["placed"]) for entry in manifest["streams"] if entry["illegal"]}
    )
    illegal = manifest["illegal_streams"]
    return _verdict(
        "L0",
        not (
            coverage["missing_count"]
            or wrong_arm_count
            or over_budget
            or closure_failures
            or refusal_mismatches
            or failure_account["unrecovered"]
            or not provenance["single_episode_code_state"]
            or illegal
            or steps != [common.STREAM_STEPS]
            or probed != [True]
            or len(invariants) != 1
            or mixed_settings
            or not agreement.get("agree", False)
        ),
        {
            "cells_expected": coverage["cells_expected"],
            "cells_present": coverage["cells_present"],
            "missing_cells": coverage["missing_cells"],
            "streams_with_wrong_arm_count": wrong_arm_count,
            "budget_violations": over_budget,
            "ledger_closure_failures": closure_failures,
            "refusal_ledger_mismatches": refusal_mismatches[:10],
            "cells_where_the_credit_bit": sum(
                1 for row in rows if int(row["refusals"]["budget_exhausted"]) > 0
            ),
            "note_on_refusals": (
                "a fixed rule being refused is the credit binding, which is what L3 asks for; L0 "
                "only requires that the transcript and the relay agree about how often it happened"
            ),
            "ledger_failures": failure_account,
            "technical_failures_this_layer": failure_account["failures"],
            "episode_code_states": provenance["distinct_episode_code_states"],
            "stamped_rows": provenance["stamped"],
            "un_stamped_rows": provenance["un_stamped"],
            "single_episode_code_state": provenance["single_episode_code_state"],
            "technical_failures_total": ledger["failures"],
            "launches": {
                "matrix": ledger["matrix_starts"],
                "matrix_cap": ledger["matrix_cap"],
                "matrix_cap_in_force_for_this_layer": common.launch_grant_for(goals)[
                    "effective_ceiling"
                ],
                "extension_grant_for_this_layer": common.launch_grant_for(goals)[
                    "extension_grant"
                ],
                "gate": ledger["gate_starts"],
                "gate_cap": ledger["gate_cap"],
            },
            "illegal_streams": illegal,
            "steps_per_episode": steps,
            "all_cells_probed": probed,
            "distinct_layer_wide_settings": invariants,
            "streams_with_mixed_settings": mixed_settings,
            "harness_agreement_with_archived_rows": agreement.get("agree", False),
            "harness_agreement_provenance": (
                "inherited: this layer was licensed by the delivered round's AGREEMENT_GATE.json, "
                "which ran under CODE_FREEZE.json. The extension may not spend new verification "
                "episodes, so the probe counters it added are proven transparent by zero-episode "
                "tests, and the runner needs no new gate because this layer's freeze names the same "
                "episode code identity the delivered layers do"
                if common.freeze_file(goals) != "CODE_FREEZE.json"
                else "this round's own AGREEMENT_GATE.json"
            ),
            "allowed_configuration_differences": list(ALLOWED_CELL_DIFFERENCES),
        },
        "the layer's data, ledger and interface are trustworthy enough to be judged at all",
    )


def l1(metrics: dict[str, Any]) -> dict[str, Any]:
    """The pressure is real search pressure, not missing detections."""
    mean_ratio = float(metrics["mean_detected_over_placed"])
    worst = metrics["worst_stream_detected_over_placed"]
    return _verdict(
        "L1",
        not metrics["illegal_streams"] and mean_ratio >= L1_MIN_DETECTED_RATIO,
        # ^ placed targets must be legal AND the search must still be finding them
        {
            "mean_detected_over_placed": round(mean_ratio, 5),
            "threshold": L1_MIN_DETECTED_RATIO,
            "worst_stream": metrics["worst_stream"],
            "worst_stream_ratio": worst,
            "illegal_streams": metrics["illegal_streams"],
            "per_stream": {
                seed: {
                    "placed": entry["placed"],
                    "detected": entry["detected"],
                    "registered": entry["registered"],
                    "ratio": entry["detected_over_placed"],
                }
                for seed, entry in sorted(metrics["per_stream"].items())
            },
        },
        "the layer's targets are actually being found, so a communication gate is not measuring "
        "a search hole",
    )


def band_edge_readings(rows: list[dict[str, Any]], best: str) -> dict[str, Any]:
    """Three readings of the L2 ratio, because g10's sat exactly on the band edge.

    The harness's aggregate is ``round(sum(values)/len(values), 5)``; summing floats in one order or
    another gives ``0.9000000000000001`` for a quantity whose exact rational value is 9/10. A gate
    that turns on a boundary must show the boundary three ways, so the reading can be checked
    instead of argued (extension protocol §1.4). The verdict still uses only ``rate``.
    """
    from fractions import Fraction

    pairs = [
        (int(row["on_time"]), int(row["detected"]))
        for row in rows
        if str(row["arm"]) == best and int(row["detected"]) > 0
    ]
    raw = [numerator / denominator for numerator, denominator in pairs]
    exact = sum((Fraction(n, d) for n, d in pairs), Fraction(0))
    exact = exact / len(pairs) if pairs else Fraction(0)
    return {
        "streams": len(pairs),
        "naive_float_mean": repr(sum(raw) / len(raw)) if raw else None,
        "exact_sum_float_mean": repr(math.fsum(raw) / len(raw)) if raw else None,
        "exact_rational_mean": str(exact),
        "note": (
            "the closed interval is applied to the gate's own ``best_on_time_rate_of_detected``; the "
            "other columns say whether a number sitting on a boundary is there for a reason"
        ),
    }


def l2(report: dict[str, Any], rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Communication is worth something here, and the effect is not pinned at a ceiling."""
    best = report["best_fixed_rule"]["arm"]
    on_time = report["bootstrap"]["on_time_per_episode"]["always_hold"]["arms"].get(best, {})
    lower = (on_time.get("ci95") or [None, None])[0]
    rate = float(report["arms"][best]["on_time_rate_of_detected"])
    near_edge = min(abs(rate - edge) for edge in L2_ON_TIME_BAND) <= 0.005
    return _verdict(
        "L2",
        lower is not None
        and lower > 0
        and L2_ON_TIME_BAND[0] <= rate <= L2_ON_TIME_BAND[1],
        {
            "best_fixed_rule": best,
            "best_minus_always_hold_on_time": on_time.get("mean_difference"),
            "ci95": on_time.get("ci95"),
            "ci_lower_above_zero": lower is not None and lower > 0,
            "pairs": on_time.get("pairs"),
            "best_on_time_rate_of_detected": rate,
            "band": list(L2_ON_TIME_BAND),
            "band_holds": L2_ON_TIME_BAND[0] <= rate <= L2_ON_TIME_BAND[1],
            "band_edge_readings": band_edge_readings(rows, best) if near_edge and rows else None,
            "reading_if_above_band": "effect ceiling: communication already buys nearly everything",
            "reading_if_below_band": "capacity/search floor: sending cannot recover the tasks",
        },
        "the uplink still buys tasks, and the best rule is neither at the floor nor at the "
        "ceiling, so a cost/effect trade-off can exist at all",
    )


def l3(report: dict[str, Any]) -> dict[str, Any]:
    """The credit forces a choice inside a step, not just a refusal at the end."""
    best = report["best_fixed_rule"]["arm"]
    table = report["arms"][best]
    competition_streams = int(table["competition_streams"])
    heavy = int(table["credit_used_ge_90_streams"])
    return _verdict(
        "L3",
        competition_streams >= L3_MIN_COMPETITION_STREAMS
        and heavy >= L3_MIN_CREDIT_HEAVY_STREAMS,
        {
            "best_fixed_rule": best,
            "definition_of_a_point": (
                "visible candidates >= 2 and 1 <= payable < visible at the step's opening credit"
            ),
            "streams_with_competition_points": competition_streams,
            "required_streams": L3_MIN_COMPETITION_STREAMS,
            "streams_at_or_above_90_percent_credit": heavy,
            "required_streams_90_percent": L3_MIN_CREDIT_HEAVY_STREAMS,
            "per_stream_points": {
                str(seed): int(value)
                for seed, value in sorted(table["per_stream_competition_points"].items())
            },
            "secondary_readings_not_used_for_the_gate": {
                "joint_only_points_total": table["joint_only_points_total"],
                "payable_candidate_points_total": table["payable_candidate_points_total"],
                "credit_exhausted_with_candidates_streams":
                    table["credit_exhausted_with_candidates_streams"],
                "mean_visible_at_competition": table["mean_visible_at_competition"],
                "mean_payable_at_competition": table["mean_payable_at_competition"],
            },
            "all_arms_competition_streams": {
                arm: value["competition_streams"] for arm, value in report["arms"].items()
            },
            #: Extension protocol §5: why the gate is where it is. Counts only -- the verdict above
            #: never reads them, and the L3 definition is still ``competition_points`` alone.
            "step_geometry_of_the_best_rule": report.get("step_geometry", {}).get(best, {}),
        },
        "the budget is binding in a way that forces selection among candidates the sender can see",
    )


def l4(report: dict[str, Any]) -> dict[str, Any]:
    """Fixed rules do not cover the situations uniformly, and there is a real frontier."""
    best = report["best_fixed_rule"]["arm"]
    block = report["bootstrap"]["on_time_per_episode"]["best_fixed"]
    envelope_difference = block["arms"].get("envelope", {})
    lower = (envelope_difference.get("ci95") or [None, None])[0]
    mean_gain = envelope_difference.get("mean_difference")
    unique = report["hindsight_envelope"]["unique_winner_counts"]
    rules_with_three = sorted(arm for arm, count in unique.items() if count >= L4_MIN_UNIQUE_WINNER_STREAMS)
    bidirectional = report["incomparable_bidirectional_pairs"]
    front = report["mean_pareto_frontier"]
    adjacent = _adjacent_pairs(front)
    conditions = {
        "1_envelope_gain": {
            "value": mean_gain,
            "ci95": envelope_difference.get("ci95"),
            "threshold": L4_MIN_ENVELOPE_GAIN,
            "holds": mean_gain is not None
            and float(mean_gain) >= L4_MIN_ENVELOPE_GAIN
            and lower is not None
            and lower > 0,
        },
        "2_two_rules_win_three_streams_each": {
            "unique_winner_counts": unique,
            "rules": rules_with_three,
            "threshold": L4_MIN_RULES_WITH_UNIQUE_WINS,
            "holds": len(rules_with_three) >= L4_MIN_RULES_WITH_UNIQUE_WINS,
        },
        "3_bidirectional_pair": {
            "pairs": bidirectional,
            "threshold_per_side": L4_MIN_BIDIRECTIONAL_STREAMS,
            "holds": bool(bidirectional),
        },
        "4_frontier": {
            "points": [
                (point["arm"], point["on_time_per_episode"], point["bytes_per_episode"])
                for point in front
            ],
            "adjacent_pairs": adjacent,
            "qualifying_pairs": [
                item
                for item in adjacent
                if float(item["effect_gap"]) >= L4_MIN_FRONTIER_EFFECT_GAP
                and float(item["byte_gap"]) >= L4_MIN_FRONTIER_BYTE_GAP
            ],
            "required_points": L4_MIN_FRONTIER_POINTS,
            "required_effect_gap": L4_MIN_FRONTIER_EFFECT_GAP,
            "required_byte_gap": L4_MIN_FRONTIER_BYTE_GAP,
            "holds": len(front) >= L4_MIN_FRONTIER_POINTS
            and any(
                float(item["effect_gap"]) >= L4_MIN_FRONTIER_EFFECT_GAP
                and float(item["byte_gap"]) >= L4_MIN_FRONTIER_BYTE_GAP
                for item in adjacent
            ),
        },
    }
    return _verdict(
        "L4",
        all(item["holds"] for item in conditions.values()),
        {
            "conditions": conditions,
            "envelope_definition": report["hindsight_envelope"]["definition"],
            "note": (
                "L4 is a necessary-condition signal only: it says fixed rules cannot cover the "
                "measured situations uniformly. It does not show a network could learn the "
                "envelope."
            ),
        },
        "there is situation-dependent value that no single fixed rule in the library captures",
    )


def _adjacent_pairs(front: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gaps between neighbours on the byte axis of the sample-mean frontier."""
    return [
        {
            "from": left["arm"],
            "to": right["arm"],
            "effect_gap": round(
                float(right["on_time_per_episode"]) - float(left["on_time_per_episode"]), 5
            ),
            "byte_gap": round(
                float(right["bytes_per_episode"]) - float(left["bytes_per_episode"]), 1
            ),
        }
        for left, right in zip(front, front[1:])
    ]


def judge(goals: int) -> dict[str, Any]:
    rows = pressure_metrics.load(goals)
    report = pressure_metrics.layer_report(goals)
    metrics = pressure_metrics.search_pressure(goals)
    gates = [
        l0(goals, rows),
        l1(metrics),
        l2(report, rows),
        l3(report),
        l4(report),
    ]
    passed = all(gate["passed"] for gate in gates)
    stopped, reason = stop_reading(gates)
    return {
        "goals": int(goals),
        "gates": {gate["gate"]: gate for gate in gates},
        "passed": passed,
        "stopped_here": stopped,
        "stop_reason": reason,
        "budget_bytes": common.BUDGET_BYTES,
        "launch_ledger": common.ledger_totals(),
    }


def stop_reading(gates: list[dict[str, Any]]) -> tuple[bool, str]:
    """The protocol's stop rules, expressed once so the report cannot paraphrase them."""
    by_name = {gate["gate"]: gate for gate in gates}
    if not by_name["L0"]["passed"]:
        return True, "L0 failed: the layer's data or interface is not trustworthy, so the round stops"
    if not by_name["L1"]["passed"]:
        return True, (
            "L1 failed: the pressure has moved into search coverage rather than communication "
            "scheduling, so higher goal counts are not run"
        )
    if all(by_name[name]["passed"] for name in ("L2", "L3", "L4")):
        return True, "L0-L4 all passed: this is the layer to design the next round on, and no " \
                     "higher layer is generated"
    return False, "L2/L3/L4 not all passed and the data is sound: proceed to the next layer"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goals", type=int, required=True, choices=sorted(common.LAYERS))
    args = parser.parse_args()
    result = judge(args.goals)
    common.write_json(common.json_path(f"GATES_g{args.goals}.json"), result)
    print(
        json.dumps(
            {
                "goals": args.goals,
                "passed": result["passed"],
                "stop_reason": result["stop_reason"],
                "gates": {
                    name: gate["passed"] for name, gate in result["gates"].items()
                },
                "detail": {
                    name: gate["evidence"] for name, gate in result["gates"].items()
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
