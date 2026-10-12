"""Deterministic paired summaries for the fixed ROUND64 ten-cell plan."""
from __future__ import annotations

import json
import hashlib
import struct
from pathlib import Path

from r64_execute import DELTAS, PLAN, RAW, TARGET, read_json, write_exclusive


def _feature_key(vector):
    return tuple(struct.pack("<f", float(value)).hex() for value in vector)


def _pair(send, skip):
    send_ids = set(send["metrics"]["on_time_ids"])
    skip_ids = set(skip["metrics"]["on_time_ids"])
    gained = sorted(send_ids - skip_ids)
    lost = sorted(skip_ids - send_ids)
    return {
        "target": TARGET,
        "Y_SEND": int(TARGET in send_ids),
        "Y_SKIP": int(TARGET in skip_ids),
        "target_delta_Y_SEND_minus_Y_SKIP": int(TARGET in send_ids) - int(TARGET in skip_ids),
        "whole_system_send_ids": sorted(send_ids),
        "whole_system_skip_ids": sorted(skip_ids),
        "gained_under_SEND": gained,
        "lost_under_SEND": lost,
        "whole_system_task_delta": len(send_ids) - len(skip_ids),
        "system_id_set_changed": bool(gained or lost),
        "spent_bytes_SEND": int(send["metrics"]["spent_bytes"]),
        "spent_bytes_SKIP": int(skip["metrics"]["spent_bytes"]),
        "cost_delta_bytes_SEND_minus_SKIP": (
            int(send["metrics"]["spent_bytes"]) - int(skip["metrics"]["spent_bytes"])),
        "SEND_emission_sequence": send["metrics"]["emission_sequence"],
        "SKIP_emission_sequence": skip["metrics"]["emission_sequence"],
        "SEND_target_chain": send["focus"]["target_chain"],
        "SKIP_target_chain": skip["focus"]["target_chain"],
        "SEND_target_transcript": send["focus"]["target_transcript"],
        "SKIP_target_transcript": skip["focus"]["target_transcript"],
        "features_19d_SEND": send["focus"]["features_19d"],
        "features_19d_SKIP": skip["focus"]["features_19d"],
        "prefix_gate_equal": True,
    }


def _prefix_summary(cell):
    tape_path = Path(cell["artifacts"]["tape"]["path"])
    tape = [json.loads(line) for line in tape_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    target = cell["focus"]["target"]
    ready = int(cell["focus"]["ready_step"])
    prior = [row for row in tape if int(row["step"]) < ready]
    same = next(row for row in tape if int(row["step"]) == ready and target in row["targets"])
    index = same["targets"].index(target)
    prefix_bytes = json.dumps(prior, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False).encode("utf-8")
    return {
        "prior_tape_rows": len(prior),
        "prior_tape_sha256": hashlib.sha256(prefix_bytes).hexdigest(),
        "same_tick_preceding_targets": same["targets"][:index],
        "same_tick_preceding_actions": {
            key: value for key, value in same["actions"].items()
            if key in same["targets"][:index]
        },
        "same_tick_preceding_features": same["features"][:index],
    }


def analyze(raw=RAW):
    raw = Path(raw).resolve()
    gate = read_json(raw / "ROUND64_GATE.json")
    prefix_gate = read_json(raw / "ROUND64_PAIR_PREFIX_GATE.json")
    if gate.get("status") != "passed" or not gate.get("matches_R60_focus"):
        raise RuntimeError("R64 neutral gate is not qualified")
    if any(not item.get("common_prefix_equal") for item in prefix_gate.values()):
        raise RuntimeError("one or more SEND/SKIP pairs lack the frozen common pre-action prefix")

    cells = {}
    for spec in PLAN:
        path = raw / "cells" / f"{spec['logical']}.json"
        cell = read_json(path)
        if cell.get("logical") != spec["logical"] or int(cell.get("delta", -999)) != spec["delta"]:
            raise RuntimeError(f"cell identity does not match fixed plan: {spec['logical']}")
        if cell.get("arm") != spec["arm"] or cell.get("metrics", {}).get("steps") != 1000:
            raise RuntimeError(f"cell arm or horizon differs from frozen plan: {spec['logical']}")
        cells[spec["logical"]] = cell

    ledger = [json.loads(line) for line in (raw / "ROUND64_LEDGER.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    task_events = [row for row in ledger if row.get("task") == "ROUND64" and
                   row.get("execution_version") == "round64-controlled-deadline-intervention/2"]
    if len(task_events) != len(ledger):
        raise RuntimeError("R64 ledger contains a foreign task or execution version")
    slots = [row for row in task_events if row["event"] == "slot_consumed"]
    if len(slots) > 12 or len(slots) < 10:
        raise RuntimeError(f"R64 consumed slot count is outside the fixed 10+2 plan: {len(slots)}")
    normal = sum(row.get("category") == "normal" for row in slots)
    reserve = sum(row.get("category") == "reserve" for row in slots)
    if normal != 10 or reserve > 2:
        raise RuntimeError(f"R64 slot classes violate frozen authorization: normal={normal}, reserve={reserve}")
    qualified = [row for row in task_events if row["event"] == "result_qualified"]
    if len(qualified) != 10:
        raise RuntimeError(f"expected ten qualified R64 cells, found {len(qualified)}")

    baseline = {
        "L": cells["L"]["metrics"],
        "L0": cells["L0"]["metrics"],
        "L_equals_L0": gate["L_equals_L0"],
        "L_matches_R60_focus": gate["matches_R60_focus"],
        "metadata_differences": gate["metadata_differences"],
    }
    pairs = {}
    for delta in DELTAS:
        send = cells[f"d{delta:+d}_FORCE_SEND"]
        skip = cells[f"d{delta:+d}_FORCE_SKIP"]
        summary = _pair(send, skip)
        summary["common_prefix_equal"] = prefix_gate[str(delta)]["common_prefix_equal"]
        summary["SEND_focus_intervention"] = send["focus"]["intervention"]
        summary["SKIP_focus_intervention"] = skip["focus"]["intervention"]
        summary["deadline_profile_generation_SEND"] = send["profile_generation_records"]
        summary["deadline_profile_generation_SKIP"] = skip["profile_generation_records"]
        summary["consumer_consistency_SEND"] = send["consumer_consistency"]
        summary["consumer_consistency_SKIP"] = skip["consumer_consistency"]
        if summary["features_19d_SEND"] != summary["features_19d_SKIP"]:
            raise RuntimeError(f"the paired action inputs differ at delta {delta}")
        pairs[str(delta)] = summary

    cross_delta = []
    by_feature = {}
    prefixes = []
    for delta in DELTAS:
        cell = cells[f"d{delta:+d}_FORCE_SEND"]
        focus = cell["focus"]
        intervention = focus["intervention"]
        vector = focus["features_19d"]
        prefix = _prefix_summary(cell)
        prefixes.append(prefix)
        key = _feature_key(vector)
        effect = pairs[str(delta)]
        by_feature.setdefault(key, []).append({
            "delta": delta,
            "target_delta": effect["target_delta_Y_SEND_minus_Y_SKIP"],
            "system_delta": effect["whole_system_task_delta"],
            "gained": effect["gained_under_SEND"],
            "lost": effect["lost_under_SEND"],
        })
        actual_deadline = float(focus["absolute_deadline"])
        current_time = float(intervention["current_time"])
        cross_delta.append({
            "delta": delta,
            "detection_step": focus["detection_step"],
            "ready_step": focus["ready_step"],
            "absolute_deadline": actual_deadline,
            "send_item_remaining_time": intervention["remaining_time"],
            "slack_at_actor_decision": actual_deadline - current_time,
            "current_time": current_time,
            "quote_bytes": intervention["quote_bytes"],
            "features_19d": vector,
            "original_action": intervention["original_greedy_action"],
            "actual_action": intervention["actual_action"],
            "send_probabilities": focus["send_probabilities"],
            "pre_action_prefix": prefix,
        })
    conflicts = []
    for key, observations in by_feature.items():
        effects = {(row["target_delta"], row["system_delta"],
                    tuple(row["gained"]), tuple(row["lost"])) for row in observations}
        if len(effects) > 1:
            conflicts.append({"float32_feature_key": list(key), "observations": observations})
    feature_keys = [_feature_key(row["features_19d"]) for row in cross_delta]
    trajectory_audit = {
        "detection_step_same_across_delta": len({row["detection_step"] for row in cross_delta}) == 1,
        "ready_step_same_across_delta": len({row["ready_step"] for row in cross_delta}) == 1,
        "ready_step": cross_delta[0]["ready_step"],
        "float32_actor_input_same_across_delta": len(set(feature_keys)) == 1,
        "prior_tape_prefix_same_across_delta": len({row["prior_tape_sha256"] for row in prefixes}) == 1,
        "per_delta_prefixes": {str(delta): prefix for delta, prefix in zip(DELTAS, prefixes, strict=True)},
    }

    run_returns = [row for row in task_events if row["event"] == "run_returned"]
    run_failures = [row for row in task_events if row["event"] == "run_failed"]
    run_calls = [row for row in task_events if row["event"] == "run_called"]
    builders = [row for row in task_events if row["event"] == "builder_called"]
    constructed = [row for row in task_events if row["event"] == "env_constructed"]
    missing_run_terminal = len(run_calls) - len(run_returns) - len(run_failures)
    consumption = {
        "planned_normal_cells": 10, "normal_slots_consumed": normal,
        "reserve_slots_consumed": reserve, "hard_cap": 12,
        "attempt_slots_consumed": len(slots), "headroom": 12 - len(slots),
        "builder_calls": len(builders), "environment_constructions": len(constructed),
        "run_calls_at_joint_run_entry": len(run_calls), "run_returns": len(run_returns),
        "run_failures": len(run_failures),
        "returned_steps": sum(int(row.get("steps", 0)) for row in run_returns),
        "failed_run_steps": sum(int(row.get("steps", 0)) for row in run_failures),
        "run_calls_without_return_or_failure_receipt": missing_run_terminal,
        "qualified_cells": len(qualified),
        "interrupted_slots": sum(row["event"] == "interrupted" for row in task_events),
    }
    return {
        "kind": "round64-paired-deadline-analysis/1",
        "interpretation": "Fixed single-scene, single frozen-policy, one realization per deadline condition; descriptive intervention only.",
        "baseline_gate": baseline,
        "pairs_by_deadline_delta": pairs,
        "cross_delta_actor_state_and_slack": cross_delta,
        "cross_delta_trajectory_audit": trajectory_audit,
        "identical_float32_input_effect_conflicts": conflicts,
        "consumption": consumption,
        "training_side_effects": {
            "optimizer_updates": 0, "ppo_replays": 0,
            "rng_restores": 0, "checkpoint_writes": 0,
        },
        "boundaries": [
            "Deadline is a condition intervention that changes feasibility, expiry, scheduling, and potentially quote serialization.",
            "Same scenario seed does not imply identical packet-drop sequence after SEND/SKIP changes RNG consumption.",
            "Accepted is not task benefit; equal episode totals can conceal target-set exchange.",
            "No small-sample significance, separability, 19D sufficiency, generalization, non-inferiority, or deployment claim.",
        ],
    }


if __name__ == "__main__":
    result = analyze()
    write_exclusive(Path(__file__).resolve().parent / "ROUND64_PAIRED_ANALYSIS.json", result)
    print(json.dumps({"pairs": len(result["pairs_by_deadline_delta"]),
                      "qualified": result["consumption"]["qualified_cells"],
                      "attempts": result["consumption"]["attempt_slots_consumed"]}))
