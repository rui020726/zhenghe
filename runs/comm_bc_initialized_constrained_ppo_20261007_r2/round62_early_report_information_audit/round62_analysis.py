"""Read-only ROUND62 reconstruction from frozen ROUND61 results and raw side logs."""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS_ROOT = HERE.parents[0]
R61 = RUNS_ROOT / "round61_early_report_value_intervention"
R61_RAW = R61 / "raw"
R60 = RUNS_ROOT / "round60_partitioned_actor_head_ablation"
FEATURE_NAMES = [
    "age_steps", "since_last_emission_time", "sent_count", "pending_event",
    "changed_view", "first_emission", "stage_changed", "requirements_changed",
    "members_changed", "progress_delta_abs", "revision_delta",
    "remaining_time_over_horizon", "urgency_now", "has_future_requirements",
    "queue_backlog_targets", "budget_remaining_fraction",
    "capacity_remaining_fraction", "subscribers", "step_over_horizon",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_ids(events):
    return sorted({e["target_id"] for e in events
                   if e.get("event_type") == "final" and e.get("status") == "completed"
                   and e.get("target_id") not in (None, "__run__")})


def run_metrics(events):
    for event in events:
        if event.get("event_type") == "final" and event.get("target_id") == "__run__":
            return event.get("detail", {}).get("metrics", {})
    raise ValueError("lifecycle has no final __run__ record")


def emission_sequence(events):
    return [[int(e["step"]), str(e["target"]), int(e["revision"]),
             int(e["cost_bytes"]), int(e["subscribers"]), bool(e["first"])]
            for e in events if e.get("kind") == "send"]


def paired_delta(send_ids, skip_ids):
    send, skip = set(send_ids), set(skip_ids)
    return {
        "send_minus_skip_count": len(send) - len(skip),
        "gained_under_send": sorted(send - skip),
        "lost_under_send": sorted(skip - send),
    }


def float32_signature(vector):
    return struct.pack("<" + "f" * len(vector), *[float(v) for v in vector])


def clipped_slack(remaining_time):
    return max(-1.0, min(1.0, float(remaining_time) / 50.0))


def result_file(arm, case):
    return R61_RAW / "cells" / f"r61_{arm}_s{case['scene']}_{case['target'].replace('piloc-e1-', '')}.json"


def tape_for(arm, case):
    logical = f"r61_{arm}_s{case['scene']}_{case['target'].replace('piloc-e1-', '')}"
    matches = list((R61_RAW / "logs").glob(f"*{logical}*_tape.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"expected one tape for {logical}, found {len(matches)}")
    return matches[0]


def paired_prefix(send_tape, skip_tape, target, decision_step):
    send_rows, skip_rows = read_jsonl(send_tape), read_jsonl(skip_tape)
    s_pre = [r for r in send_rows if int(r["step"]) < decision_step]
    k_pre = [r for r in skip_rows if int(r["step"]) < decision_step]
    strict_pre = s_pre == k_pre
    s_at = [r for r in send_rows if int(r["step"]) == decision_step]
    k_at = [r for r in skip_rows if int(r["step"]) == decision_step]
    same_other_actions = True
    same_target_input = False
    if len(s_at) != len(k_at):
        same_other_actions = False
    else:
        for sr, kr in zip(s_at, k_at):
            if sr.get("targets") != kr.get("targets"):
                same_other_actions = False
                continue
            if sr.get("features") != kr.get("features"):
                same_other_actions = False
            if sr.get("attempt_charge_bytes", {}).get(target) != kr.get("attempt_charge_bytes", {}).get(target):
                same_other_actions = False
            if sr.get("actions", {}).get(target) != 1 or kr.get("actions", {}).get(target) != 0:
                same_other_actions = False
            for other in set(sr.get("targets", [])) - {target}:
                if sr.get("actions", {}).get(other) != kr.get("actions", {}).get(other):
                    same_other_actions = False
            if target in sr.get("targets", []) and target in kr.get("targets", []):
                si, ki = sr["targets"].index(target), kr["targets"].index(target)
                same_target_input = sr["features"][si] == kr["features"][ki]
    return {
        "pre_decision_tape_rows_identical": strict_pre,
        "pre_decision_rows": len(s_pre),
        "same_tick_candidate_order_and_non_target_actions": same_other_actions,
        "same_tick_target_feature_vector": same_target_input,
        "opportunity_step": decision_step,
        "opportunity_reproduced": same_target_input and same_other_actions,
    }


def first_target_event(events, target, kind=None):
    for event in events:
        if event.get("target_id") == target and (kind is None or event.get("event_type") == kind):
            return event
    return None


def summarize_case(case, cell_results):
    target = case["target"]
    arms = {}
    for arm in ("FORCE_SEND", "FORCE_SKIP"):
        cell = cell_results[f"{arm}:{case['scene']}:{target}"]
        result = read_json(result_file(arm, case))
        lifecycle_path = Path(result["lifecycle"])
        transcript_path = Path(result["transcript"])
        life, transcript = read_jsonl(lifecycle_path), read_jsonl(transcript_path)
        ids = final_ids(life)
        metrics = run_metrics(life)
        emissions = emission_sequence(transcript)
        if ids != sorted(cell["on_time_ids"]):
            raise AssertionError(f"stored on-time IDs disagree with raw lifecycle: {case['scene']} {arm}")
        if int(metrics["communication"]["budget"]["spent_bytes"]) != int(cell["spent_bytes"]):
            raise AssertionError(f"stored cost disagrees with lifecycle: {case['scene']} {arm}")
        if emissions != cell["emission_sequence"]:
            raise AssertionError(f"stored emission sequence disagrees with transcript: {case['scene']} {arm}")
        target_events = [e for e in life if e.get("target_id") == target]
        assignment_chain = [
            {k: e.get(k) for k in ("event_type", "timestamp", "assigned_agents", "status", "stage_index", "progress", "failure_reason", "detail")}
            for e in target_events if e.get("event_type") in ("assignment", "final")
        ]
        target_tx = [e for e in transcript if e.get("target") == target]
        detected = first_target_event(life, target, "detection")
        send_event = next((e for e in target_tx if e.get("kind") == "send"), None)
        arms[arm] = {
            "on_time_ids": ids,
            "target_on_time": target in ids,
            "spent_bytes": int(metrics["communication"]["budget"]["spent_bytes"]),
            "emission_sequence": emissions,
            "target_delivery_records": [
                {k: e.get(k) for k in ("step", "agent", "accepted", "reason", "sent_at", "sent_step")}
                for e in target_tx if e.get("kind") == "deliver"
            ],
            "target_send_record": send_event,
            "target_assignment_and_final_chain": assignment_chain,
            "target_final_status": next((e.get("status") for e in reversed(target_events)
                                          if e.get("event_type") == "final"), None),
            "target_detection_source": ({
                "timestamp": detected.get("timestamp"),
                "deadline_offset": detected.get("detail", {}).get("deadline_offset"),
                "severity": detected.get("detail", {}).get("severity"),
                "remaining_requirements": detected.get("remaining_requirements"),
                "total_stages": detected.get("detail", {}).get("total_stages"),
            } if detected else None),
            "raw_source_files": {
                "lifecycle": str(lifecycle_path), "lifecycle_sha256": sha256(lifecycle_path),
                "transcript": str(transcript_path), "transcript_sha256": sha256(transcript_path),
            },
        }
    s, k = arms["FORCE_SEND"], arms["FORCE_SKIP"]
    pair = paired_delta(s["on_time_ids"], k["on_time_ids"])
    send_cell = cell_results[f"FORCE_SEND:{case['scene']}:{target}"]
    skip_cell = cell_results[f"FORCE_SKIP:{case['scene']}:{target}"]
    decision_step = int(case["decision_step"])
    send_tape, skip_tape = tape_for("FORCE_SEND", case), tape_for("FORCE_SKIP", case)
    prefix = paired_prefix(send_tape, skip_tape, target, decision_step)
    if not prefix["opportunity_reproduced"]:
        raise AssertionError(f"paired prefix/opportunity failure for {case['scene']}")
    iv_send, iv_skip = send_cell["iv_record"], skip_cell["iv_record"]
    if iv_send["pre_action_features"] != iv_skip["pre_action_features"]:
        raise AssertionError(f"paired intervention input differs at scene {case['scene']}")
    if iv_send["pre_action_features"] != case["features_19d"]:
        raise AssertionError(f"case freeze differs from intervention input at scene {case['scene']}")
    s_ids, k_ids = set(s["on_time_ids"]), set(k["on_time_ids"])
    return {
        "role": case["role"], "scene": case["scene"], "target": target,
        "ready_step": case["ready_step"], "decision_step": decision_step,
        "pre_action_features_19d": iv_send["pre_action_features"],
        "prefix_and_opportunity": prefix,
        "intervention": {
            "original_greedy_action_send": iv_send["original_greedy_action"],
            "original_greedy_action_skip": iv_skip["original_greedy_action"],
            "p_send": iv_send["original_model_p_send"],
            "p_hold": iv_send["original_model_p_hold"],
            "quote_bytes": iv_send["attempt_charge_bytes"],
            "send_legal_and_payable": iv_send["sequentially_affordable"],
            "actual_action_send": iv_send["intervened_action"],
            "actual_action_skip": iv_skip["intervened_action"],
        },
        "Y_SEND": int(target in s_ids), "Y_SKIP": int(target in k_ids),
        "target_delta": int(target in s_ids) - int(target in k_ids),
        "system_delta": pair["send_minus_skip_count"],
        "gained_ids_under_send": pair["gained_under_send"],
        "lost_ids_under_send": pair["lost_under_send"],
        "SEND_bytes": s["spent_bytes"], "SKIP_bytes": k["spent_bytes"],
        "cost_delta_bytes": s["spent_bytes"] - k["spent_bytes"],
        "arms": arms,
    }


def eligible_early_rows():
    out = []
    tape_root = R60 / "raw" / "logs"
    tapes = sorted(tape_root.glob("v2_r60_eval_new_s33_greedy_s*_a1_tape.jsonl"))
    for tape in tapes:
        scene = int(tape.name.split("_s")[-1].split("_a1")[0])
        for row in read_jsonl(tape):
            targets = row.get("free_first_reports", [])
            for target in targets:
                ix = row.get("targets", []).index(target)
                f = row["features"][ix]
                audit = row.get("candidate_audit", {}).get(target, {})
                ready = round(float(f[18]) * 1000) - round(float(f[0]))
                if (ready <= 147 and bool(f[5]) and audit.get("sequentially_affordable_before_action")
                        and row.get("send_executable_initial", {}).get(target) is True
                        and row.get("send_block_reason_initial", {}).get(target) == "allowed"):
                    out.append({"scene": scene, "target": target, "ready_step": ready,
                                "action": int(row["actions"][target]), "features": [float(v) for v in f]})
    return out


def main():
    freeze = read_json(R61 / "ROUND61_CASE_FREEZE.json")
    cell_results = read_json(R61_RAW / "ROUND61_CELL_RESULTS.json")
    guard_snapshot = read_json(R61_RAW / "ROUND61_GUARD_SNAPSHOT.json")
    case_rows = [summarize_case(c, cell_results) for c in freeze["cases"]]
    vectors = [r["pre_action_features_19d"] for r in case_rows]
    distinct = len({float32_signature(v) for v in vectors})
    changed_dims = [i for i in range(19) if len({float(v[i]) for v in vectors}) > 1]

    eligible = eligible_early_rows()
    if len(eligible) != freeze["candidate_counts"]["total_early"]:
        raise AssertionError(f"early candidates {len(eligible)} != frozen {freeze['candidate_counts']['total_early']}")
    all32 = {
        "source": "read-only R60 seed33 greedy stored eval tapes; no outcome labels consumed",
        "tapes": len(list((R60 / "raw" / "logs").glob("v2_r60_eval_new_s33_greedy_s*_a1_tape.jsonl"))),
        "eligible_early_candidates": len(eligible),
        "baseline_send": sum(x["action"] == 1 for x in eligible),
        "baseline_skip": sum(x["action"] == 0 for x in eligible),
        "ready_step_range": [min(x["ready_step"] for x in eligible), max(x["ready_step"] for x in eligible)],
        "feature_ranges": [{"index": i, "name": FEATURE_NAMES[i],
                            "min": min(x["features"][i] for x in eligible),
                            "max": max(x["features"][i] for x in eligible),
                            "unique_values": len({x["features"][i] for x in eligible}),
                            "count_equal_1": sum(x["features"][i] == 1.0 for x in eligible),
                            "count_equal_minus_1": sum(x["features"][i] == -1.0 for x in eligible)}
                           for i in range(19)],
        "target_ids_used_as_outcome_labels": False,
    }

    panel = {
        "kind": "round62-decision-information-panel/1",
        "model": freeze["model"],
        "case_count": len(case_rows),
        "actor_input_layer_A": {"feature_names": FEATURE_NAMES,
                                 "all_vectors_float32_distinct": distinct == len(vectors),
                                 "identical_vector_different_observed_delta_count": 0,
                                 "changed_dimensions_across_eight_cases": changed_dims,
                                 "cases": [{k: row[k] for k in ("scene", "target", "ready_step", "decision_step", "pre_action_features_19d")}
                                           for row in case_rows]},
        "sender_available_layer_B": {
            "schema_source": "integrated_sar/comm_policy.py SendItem + integrated_sar/comm_learning.py features()",
            "fields": ["ready_step", "age_steps", "pending_event", "changed_view", "first_emission",
                       "revision_delta", "deadline", "remaining_time", "since_last_emission_time",
                       "sent_count", "progress_delta", "stage_index", "stage_changed",
                       "requirements_changed", "members_changed", "has_future_requirements",
                       "attempt_charge_bytes", "send_executable", "send_block_reason",
                       "sender remaining budget", "backlog_targets", "capacity_remaining", "subscribers"],
            "per_case": [],
            "observability_note": "Raw SendItem fields are not serialized in the skipped arm. Deadline is recoverable from the same episode's detection event + source formula; remaining_time is derived at current_time=decision_step-1. These are legal sender-side fields, not part of the stored 19-D actor row.",
        },
        "receiver_allocator_future_outcome_layer_C": {
            "warning": "analysis only; excluded from actor input and case selection",
            "per_case_and_arm": [{"scene": row["scene"], "target": row["target"],
                                  "FORCE_SEND": {"delivery": row["arms"]["FORCE_SEND"]["target_delivery_records"],
                                                 "assignment_final": row["arms"]["FORCE_SEND"]["target_assignment_and_final_chain"],
                                                 "final": row["arms"]["FORCE_SEND"]["target_final_status"]},
                                  "FORCE_SKIP": {"delivery": row["arms"]["FORCE_SKIP"]["target_delivery_records"],
                                                 "assignment_final": row["arms"]["FORCE_SKIP"]["target_assignment_and_final_chain"],
                                                 "final": row["arms"]["FORCE_SKIP"]["target_final_status"]}}
                                 for row in case_rows],
        },
        "paired_cases": case_rows,
        "R60_32_scene_input_summary": all32,
    }

    for row, case in zip(case_rows, freeze["cases"]):
        send = row["arms"]["FORCE_SEND"]
        det = send["target_detection_source"]
        offset = float(det["deadline_offset"])
        ready = float(det["timestamp"])
        deadline = ready + offset
        now = float(row["decision_step"] - 1)
        remaining = deadline - now
        # Pull raw collector tape metadata from the actual SEND arm; it records inputs and quote.
        tape = read_jsonl(tape_for("FORCE_SEND", case))
        decision = next(r for r in tape if int(r["step"]) == row["decision_step"]
                        and case["target"] in r.get("free_first_reports", []))
        ix = decision["targets"].index(case["target"])
        candidate = decision.get("candidate_audit", {}).get(case["target"], {})
        row["sender_available_B"] = {
            "deadline": deadline, "deadline_offset_at_detection": offset,
            "remaining_time_at_decision_derived": remaining, "current_time_formula": "decision_step-1",
            "age_steps": decision["features"][ix][0],
            "since_last_emission_time": decision["features"][ix][1],
            "sent_count": decision["features"][ix][2],
            "has_future_requirements": bool(decision["features"][ix][13]),
            "quote_bytes": decision.get("attempt_charge_bytes", {}).get(case["target"]),
            "sequentially_affordable": candidate.get("sequentially_affordable_before_action"),
            "relay_initial_executable": candidate.get("relay_initial_executable"),
            "relay_initial_block_reason": candidate.get("relay_initial_block_reason"),
            "remaining_budget_bytes": decision.get("sender_state", {}).get("remaining_budget_bytes"),
            "backlog_targets": decision.get("sender_state", {}).get("backlog_targets"),
            "subscribers": decision.get("sender_state", {}).get("subscribers"),
            "feature_deadline_encoding": clipped_slack(remaining),
            "feature_urgency_now": bool(0.0 <= remaining <= 25.0),
            "source": "R61 action tape for actor/quote; R61 detection lifecycle + integrated_sar/comm.py source for deadline; remaining_time derived, not directly taped",
        }

    panel["sender_available_layer_B"]["per_case"] = [
        {"scene": row["scene"], "target": row["target"], **row["sender_available_B"]}
        for row in case_rows
    ]
    out_panel = HERE / "DECISION_INFORMATION_PANEL.json"
    out_panel.write_text(json.dumps(panel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ledger = read_jsonl(R61_RAW / "guard_ledger.jsonl")
    slots_dir = R61_RAW / "guard_slots"
    slots = sorted(slots_dir.glob("*.slot"))
    by_slot = {}
    for event in ledger:
        by_slot.setdefault(int(event["slot"]), []).append(event)
    failed_slot = next(i for i, events in by_slot.items() if any(e["event"] == "interrupted" for e in events))
    partial_rp = R61_RAW / "logs" / "v2_r61_r61_L0_focus_s1700027_s1700027_a1_rp.jsonl"
    partial_events = read_jsonl(partial_rp)
    run_config = next((e for e in partial_events if e.get("event_type") == "run_config"), None)
    max_observed_time = max((float(e.get("timestamp", 0.0)) for e in partial_events), default=0.0)
    completed_returns = [e for e in ledger if e.get("event") == "run_returned"]
    slot2 = by_slot[failed_slot]
    actual_env_constructions_proven = sum(e["event"] == "env_constructed" for e in ledger) + int(run_config is not None)
    model_path = RUNS_ROOT / freeze["model"]["path"]
    protected = {
        "ROUND46_CURRENT_TASK.json": RUNS_ROOT / "ROUND46_CURRENT_TASK.json",
        "ROUND46_R2_V6_STOP_STATE.json": RUNS_ROOT / "ROUND46_R2_V6_STOP_STATE.json",
    }
    protected_expected = {
        "ROUND46_CURRENT_TASK.json": "cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b",
        "ROUND46_R2_V6_STOP_STATE.json": "93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18",
    }
    source_files = {
        "r61_execute.py": R61 / "r61_execute.py",
        "r61_intervene.py": R61 / "r61_intervene.py",
        "integrated_sar/comm_learning.py": HERE.parents[2] / "integrated_sar" / "comm_learning.py",
        "integrated_sar/comm_policy.py": HERE.parents[2] / "integrated_sar" / "comm_policy.py",
        "integrated_sar/comm.py": HERE.parents[2] / "integrated_sar" / "comm.py",
    }
    recheck = {
        "kind": "round61-evidence-recheck/1",
        "source_head": "e14a7fe7ad24b888e81fcf4485f576b9268661c18",
        "model": freeze["model"],
        "case_freeze": {"cases": len(freeze["cases"]), "early_candidates": freeze["candidate_counts"],
                        "outcome_blind_declared": freeze["selection_algorithm"]["outcome_blind"],
                        "case_action_inputs_equal_both_arms": True,
                        "all_eight_prefix_and_opportunity_checks_pass": all(r["prefix_and_opportunity"]["opportunity_reproduced"] for r in case_rows)},
        "gate": read_json(R61_RAW / "ROUND61_GATE.json"),
        "independent_pair_recomputation": {
            "all_16_force_cells_recomputed_from_raw_lifecycle_and_transcript": True,
            "focus": {"Y_SEND": case_rows[0]["Y_SEND"], "Y_SKIP": case_rows[0]["Y_SKIP"],
                      "task_delta": case_rows[0]["target_delta"], "system_delta": case_rows[0]["system_delta"],
                      "cost_delta_bytes": case_rows[0]["cost_delta_bytes"],
                      "gained": case_rows[0]["gained_ids_under_send"], "lost": case_rows[0]["lost_ids_under_send"]},
            "controls": [{"scene": r["scene"], "target_delta": r["target_delta"],
                          "system_delta": r["system_delta"], "gained": r["gained_ids_under_send"],
                          "lost": r["lost_ids_under_send"], "cost_delta_bytes": r["cost_delta_bytes"]}
                         for r in case_rows[1:]],
            "raw_source_files_by_case": [{"scene": row["scene"], "target": row["target"],
                                           "arms": {arm: data["raw_source_files"]
                                                    for arm, data in row["arms"].items()}}
                                          for row in case_rows],
        },
        "identity_hashes": {
            "case_freeze_sha256": sha256(R61 / "ROUND61_CASE_FREEZE.json"),
            "cell_results_sha256": sha256(R61_RAW / "ROUND61_CELL_RESULTS.json"),
            "model_file_sha256_actual": sha256(model_path),
            "model_file_sha256_matches_freeze": sha256(model_path) == freeze["model"]["file_sha256"],
            "source_files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in source_files.items()},
            "protected_file_hashes": {name: {"sha256": sha256(path),
                                               "matches_expected": sha256(path) == protected_expected[name]}
                                      for name, path in protected.items()},
        },
        "consumption_reconstruction": {
            "slot_files": len(slots), "slot_ids": [int(p.stem) for p in slots],
            "normal_slots": sum(json.loads(p.read_text(encoding="utf-8"))["category"] == "normal" for p in slots),
            "reserve_slots": sum(json.loads(p.read_text(encoding="utf-8"))["category"] == "reserve" for p in slots),
            "hard_cap": 20, "headroom": 20 - len(slots), "builder_calls_ledger": sum(e["event"] == "builder_called" for e in ledger),
            "env_constructed_events": sum(e["event"] == "env_constructed" for e in ledger),
            "env_constructions_proven": actual_env_constructions_proven,
            "runner_invocations_proven": len(completed_returns) + int(run_config is not None),
            "run_returned_count": len(completed_returns),
            "successful_return_steps": sum(int(e["steps"]) for e in completed_returns),
            "failed_attempt": {
                "slot": failed_slot, "logical": slot2[0].get("logical"),
                "slot_events": [e["event"] for e in slot2],
                "builder_called": any(e["event"] == "builder_called" for e in slot2),
                "guard_run_called_hook_recorded": any(e["event"] == "run_called" for e in slot2),
                "runner_entered_proved_by_run_config": run_config is not None,
                "run_configured_horizon": run_config.get("detail", {}).get("configuration", {}).get("steps") if run_config else None,
                "max_lifecycle_timestamp_observed": max_observed_time,
                "lower_bound_steps_reached": int(max_observed_time),
                "exact_steps": None,
                "upper_bound_steps": 1000 if run_config else None,
                "partial_lifecycle_event_count": len(partial_events),
                "final_run_record_present": any(e.get("target_id") == "__run__" and e.get("event_type") == "final" for e in partial_events),
                "separate_claim_file_for_failed_slot": False,
                "note": "Attempt slot 2 claim path was reused by reserve retry slot 3; original slot 2 ledger+partial lifecycle remain. Executor calls run_called only after _run_episode returns, so the hook omitted this failed invocation. No traceback artifact was found; reported shadow_gate cause is not independently traceback-verified.",
            },
            "total_steps": {"known_successful_returns": sum(int(e["steps"]) for e in completed_returns),
                            "minimum_proven_including_failed_attempt": sum(int(e["steps"]) for e in completed_returns) + int(max_observed_time),
                            "maximum_from_horizon_if_all_invocations_respect_config": 19 * 1000,
                            "exact": None},
            "qualified_results": sum(e["event"] == "result_qualified" for e in ledger),
            "technical_failures": sum(e["event"] == "interrupted" for e in ledger),
            "reserve_retries": sum(e["event"] == "slot_consumed" and e.get("category") == "reserve" for e in ledger),
            "claim_overwrite_note": "The retry wrote the same attempt_0001.json path as the failed logical attempt; the slot ledger and partial raw logs are the surviving evidence. No claim has been fabricated.",
        },
        "side_effects_this_round": {"environment_constructions": 0, "simulations": 0, "new_claims": 0,
                                     "new_episodes": 0, "optimizer_updates_real": 0,
                                     "optimizer_updates_synthetic": 0, "ppo_replays": 0,
                                     "training_or_rng_restores": 0, "checkpoint_writes": 0,
                                     "new_evaluations": 0, "cpu_forward_count": 0},
        "protected_hashes": {
            "ROUND46_CURRENT_TASK.json": "cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b",
            "ROUND46_R2_V6_STOP_STATE.json": "93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18",
        },
    }
    (HERE / "ROUND61_EVIDENCE_RECHECK.json").write_text(json.dumps(recheck, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(case_rows), "prefixes_pass": sum(r["prefix_and_opportunity"]["opportunity_reproduced"] for r in case_rows),
                      "early_candidates": len(eligible), "send": all32["baseline_send"], "skip": all32["baseline_skip"],
                      "slots": len(slots), "successful_steps": recheck["consumption_reconstruction"]["successful_return_steps"],
                      "failed_attempt_step_lower_bound": int(max_observed_time), "failed_attempt_exact_steps": None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
