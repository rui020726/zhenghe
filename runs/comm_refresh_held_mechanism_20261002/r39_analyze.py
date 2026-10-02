"""Build compact per-cell, paired, opportunity, mechanism-chain, and source manifests."""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
R37 = ROOT / "runs" / "comm_final_evaluation_20261002"
R38 = ROOT / "runs" / "comm_research_recovery_20261002"
ROUND18 = ROOT / "runs" / "search_trace_and_policy_20260925"
sys.path[:0] = [str(ROOT), str(ROOT / "runs" / "comm_cost_v2_rerun_20260928"),
                str(ROUND18)]
SEEDS = [11, 22, 33]
SCENES = [1600000, 1600001, 1600002, 1600003, 1600004, 1600005, 1600006,
          1600011, 1600018, 1600034, 1600036, 1600040, 1600049, 1600056,
          1600060, 1600062]
FAILED = {1600000, 1600011, 1600018, 1600034, 1600036,
          1600040, 1600049, 1600056, 1600060, 1600062}
ARMS = [
    "s11_original_greedy", "s11_refresh_held",
    "s22_original_greedy", "s22_refresh_held",
    "s33_original_greedy", "s33_refresh_held",
    "immediate", "late_discovery_gate_147", "always_hold",
]
LEARNING_ARMS = ["original_greedy", "refresh_held"]
REFERENCE_ARMS = ["immediate", "late_discovery_gate_147", "always_hold"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(name: str, data: Any) -> None:
    path = HERE / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def add_path(entries: dict[str, dict[str, Any]], path_value: str | Path, kind: str) -> None:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        return
    key = str(path).casefold()
    if key in entries:
        return
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        rel = None
    entries[key] = {
        "kind": kind,
        "absolute_path": str(path),
        "repository_relative_path": rel,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def row_path(arm: str, scene: int) -> Path:
    if arm.startswith("s") and "_original_greedy" in arm:
        seed = int(arm[1:3])
        return HERE / "rows" / f"v2_eval_s{seed}_final_greedy_s{scene}_a0.json"
    if arm.startswith("s") and "_refresh_held" in arm:
        seed = int(arm[1:3])
        return HERE / "rows" / f"v2_eval_s{seed}_refresh_held_s{scene}_a0.json"
    rule = {"late_discovery_gate_147": "late_discovery_gate"}.get(arm, arm)
    return HERE / "rows" / f"v2_{rule}_s{scene}_a1.json"


def task_set(lifecycle_path: str | Path, window: int = 1000) -> set[str]:
    records = read_jsonl(Path(lifecycle_path))
    found: dict[str, tuple[float, float]] = {}
    completed: dict[str, float] = {}
    for record in records:
        event = record.get("event_type")
        stamp = float(record.get("timestamp") or 0.0)
        target = str(record.get("target_id"))
        if event == "detection" and stamp <= window:
            deadline_offset = float((record.get("detail") or {}).get("deadline_offset", 0.0))
            found[target] = (stamp, stamp + deadline_offset)
        elif event == "progress" and record.get("status") == "completed" and stamp <= window:
            completed.setdefault(target, stamp)
    return {
        target for target, (_detected, deadline) in found.items()
        if target in completed and completed[target] <= deadline
    }


def cell_data(arm: str, scene: int) -> dict[str, Any]:
    row_file = row_path(arm, scene)
    if not row_file.is_file():
        raise RuntimeError(f"missing completed result row {row_file}")
    row = read_json(row_file)
    if int(row.get("steps", 0)) != 1000 or int(row.get("optimizer_steps", -1)) != 0:
        raise RuntimeError(f"run closure or no-update condition failed in {row_file}")
    lifecycle = Path(row["artefacts"]["lifecycle"])
    transcript = Path(row["artefacts"]["transcript"])
    if not lifecycle.is_file() or not transcript.is_file():
        raise RuntimeError(f"missing lifecycle/transcript artifact for {arm} s{scene}")
    lifecycle_records = read_jsonl(lifecycle)
    targets = sorted(task_set(lifecycle, int(row["steps"])))
    discovered_targets = sorted({
        str(record.get("target_id")) for record in lifecycle_records
        if record.get("event_type") == "detection"
        and float(record.get("timestamp") or 0.0) <= int(row["steps"])
    })
    if len(targets) != int(row["on_time"]):
        raise RuntimeError(f"target-set count differs from Y for {arm} s{scene}")
    emissions = [list(item) for item in row["emission_sequence"]]
    first = [item for item in emissions if bool(item[5])]
    refresh = [item for item in emissions if not bool(item[5])]
    first_bytes = sum(int(item[3]) for item in first)
    refresh_bytes = sum(int(item[3]) for item in refresh)
    if first_bytes + refresh_bytes != int(row["spent_bytes"]):
        raise RuntimeError(f"first/refresh byte split does not close for {arm} s{scene}")
    transcript_records = read_jsonl(transcript)
    sends = [record for record in transcript_records if record.get("kind") == "send"]
    first_deliveries: dict[str, dict[str, Any]] = {}
    for record in sends:
        if bool(record.get("first")):
            target = str(record["target"])
            if target in first_deliveries:
                raise RuntimeError(f"multiple first-send transcript records for {target} in {arm} s{scene}")
            first_deliveries[target] = {
                "step": int(record["step"]),
                "revision": int(record.get("revision", 0)),
                "packets_surviving_channel_loss": int(record.get("packets", 0)),
                "packets_lost": int(record.get("packets_lost", 0)),
                "recipient_attempts": int(record.get("attempts", 0)),
                "cost_bytes": int(record.get("cost_bytes", 0)),
                "recipient_delivery_records": [],
            }
    for record in transcript_records:
        if record.get("kind") != "deliver":
            continue
        target = str(record.get("target"))
        first = first_deliveries.get(target)
        if first is None:
            continue
        if (int(record.get("revision", -1)) == first["revision"]
                and int(record.get("sent_step", -1)) == first["step"]):
            first["recipient_delivery_records"].append({
                "step": int(record["step"]),
                "sent_step": int(record["sent_step"]),
                "agent": str(record["agent"]),
                "accepted": bool(record.get("accepted")),
                "reason": str(record.get("reason", "")),
                "payload_bytes": int(record.get("payload_bytes", 0)),
            })
    for first in first_deliveries.values():
        first["recipient_packets_delivered"] = len(first["recipient_delivery_records"])
        first["recipient_packets_accepted"] = sum(
            int(item["accepted"]) for item in first["recipient_delivery_records"]
        )
        first["delivered_to_any_receiver"] = first["recipient_packets_delivered"] > 0
        first["accepted_by_any_receiver"] = first["recipient_packets_accepted"] > 0

    target_opportunities: dict[str, dict[str, Any]] = {}
    if arm.startswith("s") and "_" in arm:
        if arm.endswith("original_greedy") or arm.endswith("refresh_held"):
            tape_path = Path(row["artefacts"]["tape"])
            tape_records = read_jsonl(tape_path)
            target_opportunities = learned_target_opportunities(arm, tape_records, first_deliveries)

    return {
        "scenario": int(scene),
        "group": "failed" if scene in FAILED else "control",
        "arm": arm,
        "on_time": int(row["on_time"]),
        "on_time_target_ids": targets,
        "discovered": int(row["discovered"]),
        "completed": int(row["completed"]),
        "expired": int(row["expired"]),
        "discovered_target_ids": discovered_targets,
        "spent_bytes": int(row["spent_bytes"]),
        "first_report_bytes": first_bytes,
        "refresh_bytes": refresh_bytes,
        "remaining_budget_bytes": int(row["budget_bytes"]) - int(row["spent_bytes"]),
        "emissions": int(row["emissions"]),
        "first_report_emissions": len(first),
        "refresh_emissions": len(refresh),
        "sender_recipient_attempts": int(row["relay_counters"].get("transmission_attempts", 0)),
        "first_report_actual_delivery_targets": sorted(
            target for target, evidence in first_deliveries.items()
            if evidence["delivered_to_any_receiver"]
        ),
        "first_report_accepted_targets": sorted(
            target for target, evidence in first_deliveries.items()
            if evidence["accepted_by_any_receiver"]
        ),
        "first_report_opportunities": target_opportunities,
        "row_path": str(row_file.relative_to(ROOT)).replace("\\", "/"),
        "lifecycle_path": str(lifecycle),
        "transcript_path": str(transcript),
        "tape_path": row["artefacts"].get("tape"),
    }


def learned_target_opportunities(
    arm: str,
    tape_records: list[dict[str, Any]],
    first_deliveries: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    held = arm.endswith("refresh_held")
    for tape in tape_records:
        for index, target in enumerate(tape["targets"]):
            is_first = bool(float(tape["features"][index][5]) >= 0.5)
            if held:
                audit = tape["candidate_budget_audit"][target]
                sequential = bool(audit["effective_run"]["sequential_affordable_before_action"])
                raw_action = int(tape["raw_model_actions"][target])
                effective_action = int(tape["effective_actions"][target])
                source = tape["decision_source"][target]
                initial = bool(tape["send_executable_initial"][target])
                charge = int(tape["attempt_charge_bytes"][target])
                budget_before = float(audit["effective_run"]["budget_before"])
                budget_after = float(audit["effective_run"]["budget_after"])
                mask_reason = audit.get("relay_initial_block_reason") if not initial else None
            else:
                shadow = tape["candidate_shadow"][target]
                sequential = bool(shadow["honours_this_candidate"])
                effective_action = int(tape["actions"][target])
                raw_action = effective_action if target in tape["actionable"] else None
                source = "frozen_greedy_policy" if target in tape["actionable"] \
                    else "existing_sequential_budget_mask"
                initial = bool(tape["send_executable_initial"][target])
                charge = int(tape["attempt_charge_bytes"][target])
                budget_before = float(shadow["budget_before"])
                budget_after = float(shadow["budget_after"])
                mask_reason = tape.get("masked_reasons", {}).get(target)
            target = str(target)
            row = result.setdefault(target, {
                "candidate_rows_seen": 0,
                "first_report_candidate_rows": 0,
                "refresh_candidate_rows": 0,
                "sequentially_affordable_first_report_rows": 0,
                "budget_masked_first_report_rows": 0,
                "raw_policy_action_counts": {"SEND": 0, "HOLD": 0, "missing": 0},
                "effective_action_counts": {"SEND": 0, "HOLD": 0},
                "decision_source_counts": {},
                "intervention_reason_counts": {},
                "ever_sequentially_affordable": False,
                "key_decisions": {},
            })
            step = int(tape["step"])
            candidate_kind = "first_report" if is_first else "refresh"
            row["candidate_rows_seen"] += 1
            row["first_report_candidate_rows" if is_first else "refresh_candidate_rows"] += 1
            if is_first:
                row["sequentially_affordable_first_report_rows"] += int(sequential)
                row["budget_masked_first_report_rows"] += int(not sequential or not initial)
                row["ever_sequentially_affordable"] |= sequential
            row["raw_policy_action_counts"][
                "missing" if raw_action is None else ("SEND" if raw_action == 1 else "HOLD")
            ] += 1
            row["effective_action_counts"]["SEND" if effective_action == 1 else "HOLD"] += 1
            row["decision_source_counts"][source] = row["decision_source_counts"].get(source, 0) + 1
            intervention_reason = tape.get("intervention_reason", {}).get(target)
            if intervention_reason:
                row["intervention_reason_counts"][intervention_reason] = (
                    row["intervention_reason_counts"].get(intervention_reason, 0) + 1
                )
            detail = {
                "step": step,
                "candidate_type": candidate_kind,
                "relay_initial_executable": initial,
                "sequential_affordable": sequential,
                "mask_reason": mask_reason,
                "budget_before": budget_before,
                "budget_after_effective_action": budget_after,
                "attempt_charge_bytes": charge,
                "raw_model_action": raw_action,
                "effective_action": effective_action,
                "decision_source": source,
                "intervention_reason": intervention_reason,
            }
            if "first_candidate" not in row["key_decisions"]:
                row["key_decisions"]["first_candidate"] = detail
            if is_first and "first_first_report_candidate" not in row["key_decisions"]:
                row["key_decisions"]["first_first_report_candidate"] = detail
            if is_first and sequential and "first_affordable_first_report" not in row["key_decisions"]:
                row["key_decisions"]["first_affordable_first_report"] = detail
            if is_first and (not sequential or not initial) \
                    and "first_masked_first_report" not in row["key_decisions"]:
                row["key_decisions"]["first_masked_first_report"] = detail
            if raw_action == 1 and "first_model_send_request" not in row["key_decisions"]:
                row["key_decisions"]["first_model_send_request"] = detail
            if effective_action == 1 and "first_effective_send" not in row["key_decisions"]:
                row["key_decisions"]["first_effective_send"] = detail
            if not is_first and "first_refresh_candidate" not in row["key_decisions"]:
                row["key_decisions"]["first_refresh_candidate"] = detail
            if intervention_reason and "first_refresh_intervention" not in row["key_decisions"]:
                row["key_decisions"]["first_refresh_intervention"] = detail
    for target, row in result.items():
        row["first_report_emitted"] = target in first_deliveries
        row["first_report_delivery"] = first_deliveries.get(target)
        row["first_report_delivered_to_any_receiver"] = bool(
            first_deliveries.get(target, {}).get("delivered_to_any_receiver", False)
        )
        row["first_report_accepted_by_any_receiver"] = bool(
            first_deliveries.get(target, {}).get("accepted_by_any_receiver", False)
        )
    return result


def median(values: list[int | float]) -> int | float | None:
    return None if not values else statistics.median(values)


def summarize(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"n_scenes": 0, "sum": 0, "mean": None, "median": None,
                "positive_delta_scenes": 0, "zero_delta_scenes": 0,
                "negative_delta_scenes": 0}
    return {
        "n_scenes": len(values),
        "sum": sum(values),
        "mean": sum(values) / len(values),
        "median": median(values),
        "positive_delta_scenes": sum(value > 0 for value in values),
        "zero_delta_scenes": sum(value == 0 for value in values),
        "negative_delta_scenes": sum(value < 0 for value in values),
    }


def paired_matrix(cells: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    main: dict[str, Any] = {}
    absolute: dict[str, Any] = {}
    target_changes: list[dict[str, Any]] = []
    opportunity_results: dict[str, Any] = {}
    for seed in SEEDS:
        main[f"s{seed}"] = {}
        opportunity_results[f"s{seed}"] = {}
        for group in ("failed", "control"):
            scenarios = [scene for scene in SCENES
                         if ("failed" if scene in FAILED else "control") == group]
            comparisons = []
            task_deltas: list[int] = []
            byte_deltas: list[int] = []
            first_byte_deltas: list[int] = []
            refresh_byte_deltas: list[int] = []
            emissions_deltas: list[int] = []
            rescued_total = lost_total = swaps = 0
            original_y_total = held_y_total = 0
            original_bytes_total = held_bytes_total = 0
            original_refresh_bytes_total = held_refresh_bytes_total = 0
            original_refresh_emissions_total = held_refresh_emissions_total = 0
            original_affordable: set[tuple[int, str]] = set()
            held_affordable: set[tuple[int, str]] = set()
            original_emitted: set[tuple[int, str]] = set()
            held_emitted: set[tuple[int, str]] = set()
            original_delivered: set[tuple[int, str]] = set()
            held_delivered: set[tuple[int, str]] = set()
            original_accepted: set[tuple[int, str]] = set()
            held_accepted: set[tuple[int, str]] = set()
            for scene in scenarios:
                original = cells[(f"s{seed}_original_greedy", scene)]
                held = cells[(f"s{seed}_refresh_held", scene)]
                original_set = set(original["on_time_target_ids"])
                held_set = set(held["on_time_target_ids"])
                rescued = sorted(held_set - original_set)
                newly_lost = sorted(original_set - held_set)
                delta_y = int(held["on_time"] - original["on_time"])
                delta_bytes = int(held["spent_bytes"] - original["spent_bytes"])
                original_y_total += int(original["on_time"])
                held_y_total += int(held["on_time"])
                original_bytes_total += int(original["spent_bytes"])
                held_bytes_total += int(held["spent_bytes"])
                original_refresh_bytes_total += int(original["refresh_bytes"])
                held_refresh_bytes_total += int(held["refresh_bytes"])
                original_refresh_emissions_total += int(original["refresh_emissions"])
                held_refresh_emissions_total += int(held["refresh_emissions"])
                task_deltas.append(delta_y)
                byte_deltas.append(delta_bytes)
                first_byte_deltas.append(held["first_report_bytes"] - original["first_report_bytes"])
                refresh_byte_deltas.append(held["refresh_bytes"] - original["refresh_bytes"])
                emissions_deltas.append(held["emissions"] - original["emissions"])
                rescued_total += len(rescued)
                lost_total += len(newly_lost)
                swaps += int(bool(rescued) and bool(newly_lost) and delta_y == 0)
                comparisons.append({
                    "scenario": scene,
                    "group": group,
                    "original_y": original["on_time"],
                    "refresh_held_y": held["on_time"],
                    "delta_y": delta_y,
                    "rescued_target_ids": rescued,
                    "newly_lost_target_ids": newly_lost,
                    "net_same_with_target_swap": bool(rescued and newly_lost and delta_y == 0),
                    "original_spent_bytes": original["spent_bytes"],
                    "refresh_held_spent_bytes": held["spent_bytes"],
                    "delta_spent_bytes": delta_bytes,
                    "original_first_report_bytes": original["first_report_bytes"],
                    "refresh_held_first_report_bytes": held["first_report_bytes"],
                    "original_refresh_bytes": original["refresh_bytes"],
                    "refresh_held_refresh_bytes": held["refresh_bytes"],
                })
                for target, opp in original["first_report_opportunities"].items():
                    key = (scene, target)
                    if opp["ever_sequentially_affordable"]:
                        original_affordable.add(key)
                    if opp["first_report_emitted"]:
                        original_emitted.add(key)
                    if opp["first_report_delivered_to_any_receiver"]:
                        original_delivered.add(key)
                    if opp["first_report_accepted_by_any_receiver"]:
                        original_accepted.add(key)
                for target, opp in held["first_report_opportunities"].items():
                    key = (scene, target)
                    if opp["ever_sequentially_affordable"]:
                        held_affordable.add(key)
                    if opp["first_report_emitted"]:
                        held_emitted.add(key)
                    if opp["first_report_delivered_to_any_receiver"]:
                        held_delivered.add(key)
                    if opp["first_report_accepted_by_any_receiver"]:
                        held_accepted.add(key)
                target_changes.append({"seed": seed, **comparisons[-1]})
            main[f"s{seed}"][group] = {
                "n_scenes": len(scenarios),
                "original_on_time_total": original_y_total,
                "refresh_held_on_time_total": held_y_total,
                "original_spent_bytes_total": original_bytes_total,
                "refresh_held_spent_bytes_total": held_bytes_total,
                "original_refresh_bytes_total": original_refresh_bytes_total,
                "refresh_held_refresh_bytes_total": held_refresh_bytes_total,
                "original_refresh_emissions_total": original_refresh_emissions_total,
                "refresh_held_refresh_emissions_total": held_refresh_emissions_total,
                "task_delta_refresh_held_minus_original": summarize(task_deltas),
                "byte_delta_refresh_held_minus_original": summarize(byte_deltas),
                "first_report_byte_delta": summarize(first_byte_deltas),
                "refresh_byte_delta": summarize(refresh_byte_deltas),
                "emissions_delta": summarize(emissions_deltas),
                "rescued_target_occurrences": rescued_total,
                "newly_lost_target_occurrences": lost_total,
                "net_same_with_target_swap_scenes": swaps,
                "per_scene": comparisons,
            }
            opportunity_results[f"s{seed}"][group] = {
                "original": {
                    "discovered_target_occurrences": sum(len(cells[
                        (f"s{seed}_original_greedy", scene)]["discovered_target_ids"])
                        for scene in scenarios),
                    "target_opportunities_seen": len({(scene, target)
                        for scene in scenarios
                        for target in cells[(f"s{seed}_original_greedy", scene)]
                        ["first_report_opportunities"]}),
                    "targets_with_any_affordable_first_report_opportunity": len(original_affordable),
                    "targets_first_report_emitted": len(original_emitted),
                    "targets_with_any_first_report_packet_delivered": len(original_delivered),
                    "targets_with_any_first_report_packet_accepted": len(original_accepted),
                },
                "refresh_held": {
                    "discovered_target_occurrences": sum(len(cells[
                        (f"s{seed}_refresh_held", scene)]["discovered_target_ids"])
                        for scene in scenarios),
                    "target_opportunities_seen": len({(scene, target)
                        for scene in scenarios
                        for target in cells[(f"s{seed}_refresh_held", scene)]
                        ["first_report_opportunities"]}),
                    "targets_with_any_affordable_first_report_opportunity": len(held_affordable),
                    "targets_first_report_emitted": len(held_emitted),
                    "targets_with_any_first_report_packet_delivered": len(held_delivered),
                    "targets_with_any_first_report_packet_accepted": len(held_accepted),
                },
                "new_affordable_target_opportunities": [
                    {"scene": scene, "target_id": target}
                    for scene, target in sorted(held_affordable - original_affordable)
                ],
                "no_longer_affordable_target_opportunities": [
                    {"scene": scene, "target_id": target}
                    for scene, target in sorted(original_affordable - held_affordable)
                ],
                "new_first_emissions": [
                    {"scene": scene, "target_id": target}
                    for scene, target in sorted(held_emitted - original_emitted)
                ],
                "first_emissions_no_longer_made": [
                    {"scene": scene, "target_id": target}
                    for scene, target in sorted(original_emitted - held_emitted)
                ],
                "new_first_reports_delivered_to_any_receiver": [
                    {"scene": scene, "target_id": target}
                    for scene, target in sorted(held_delivered - original_delivered)
                ],
                "first_reports_no_longer_delivered_to_any_receiver": [
                    {"scene": scene, "target_id": target}
                    for scene, target in sorted(original_delivered - held_delivered)
                ],
                "new_first_reports_accepted_by_any_receiver": [
                    {"scene": scene, "target_id": target}
                    for scene, target in sorted(held_accepted - original_accepted)
                ],
                "first_reports_no_longer_accepted_by_any_receiver": [
                    {"scene": scene, "target_id": target}
                    for scene, target in sorted(original_accepted - held_accepted)
                ],
                "denominator_definition": (
                    "discovered target occurrences in the corresponding arm and scene group; "
                    "target-opportunity counts are a subset, not candidate-row counts"
                ),
            }

    # Absolute paired comparisons are kept separate by seed, group, and learned condition.
    for seed in SEEDS:
        absolute[f"s{seed}"] = {}
        for group in ("failed", "control"):
            scenarios = [scene for scene in SCENES
                         if ("failed" if scene in FAILED else "control") == group]
            absolute[f"s{seed}"][group] = {}
            for condition in LEARNING_ARMS:
                arm = f"s{seed}_{condition}"
                absolute[f"s{seed}"][group][condition] = {}
                for ref in REFERENCE_ARMS:
                    y_delta, b_delta = [], []
                    for scene in scenarios:
                        candidate = cells[(arm, scene)]
                        reference = cells[(ref, scene)]
                        y_delta.append(candidate["on_time"] - reference["on_time"])
                        b_delta.append(candidate["spent_bytes"] - reference["spent_bytes"])
                    absolute[f"s{seed}"][group][condition][ref] = {
                        "candidate_on_time_total": sum(
                            cells[(arm, scene)]["on_time"] for scene in scenarios
                        ),
                        "reference_on_time_total": sum(
                            cells[(ref, scene)]["on_time"] for scene in scenarios
                        ),
                        "candidate_spent_bytes_total": sum(
                            cells[(arm, scene)]["spent_bytes"] for scene in scenarios
                        ),
                        "reference_spent_bytes_total": sum(
                            cells[(ref, scene)]["spent_bytes"] for scene in scenarios
                        ),
                        "task_delta_candidate_minus_reference": summarize(y_delta),
                        "byte_delta_candidate_minus_reference": summarize(b_delta),
                        "scene_level_task_and_byte_deltas": [
                            {"scene": scene,
                             "task_delta": cells[(arm, scene)]["on_time"]
                                 - cells[(ref, scene)]["on_time"],
                             "byte_delta": cells[(arm, scene)]["spent_bytes"]
                                 - cells[(ref, scene)]["spent_bytes"]}
                            for scene in scenarios
                        ],
                    }

    return {"main_paired_refresh_held_minus_original": main,
            "first_report_opportunity_changes": opportunity_results,
            "absolute_vs_rule_references": absolute,
            "target_set_changes_by_pair": target_changes}


def designated_chains(cells: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    missing = read_json(R38 / "MISSING_TARGET_CHAINS.json")
    chain_list = []
    for case in missing["cases"]:
        scene = int(case["scenario"])
        target = str(case["target_id"])
        for seed in SEEDS:
            arms = [f"s{seed}_original_greedy", f"s{seed}_refresh_held", *REFERENCE_ARMS]
            for arm in arms:
                cell = cells[(arm, scene)]
                row = read_json(ROOT / cell["row_path"])
                lifecycle = read_jsonl(Path(cell["lifecycle_path"]))
                events = [record for record in lifecycle
                          if str(record.get("target_id")) == target]
                selected_events: list[dict[str, Any]] = []
                progress_segment: list[dict[str, Any]] = []

                def flush_progress_segment() -> None:
                    if not progress_segment:
                        return
                    selected_events.append(progress_segment[0])
                    if progress_segment[-1] is not progress_segment[0]:
                        selected_events.append(progress_segment[-1])
                    progress_segment.clear()

                last_progress_key: tuple[Any, Any] | None = None
                for event in events:
                    if event.get("event_type") != "progress":
                        flush_progress_segment()
                        selected_events.append(event)
                        last_progress_key = None
                        continue
                    key = (event.get("status"), event.get("stage_index"))
                    if last_progress_key is not None and key != last_progress_key:
                        flush_progress_segment()
                    if not progress_segment:
                        progress_segment.extend((event, event))
                    else:
                        progress_segment[-1] = event
                    last_progress_key = key
                flush_progress_segment()
                communications = read_jsonl(Path(cell["transcript_path"]))
                communications = [record for record in communications
                                  if str(record.get("target")) == target]
                tape_path = cell.get("tape_path")
                opportunity_summary = cell.get("first_report_opportunities", {}).get(target)
                decisions = [] if not opportunity_summary else sorted(
                    [{"selection": label, **detail} for label, detail in
                     opportunity_summary.get("key_decisions", {}).items()],
                    key=lambda item: (int(item["step"]), item["selection"]),
                )
                event_counts = Counter(
                    f"{event.get('event_type')}:{event.get('status', '')}".rstrip(":")
                    for event in events
                )
                chain_list.append({
                    "scene": scene,
                    "target_id": target,
                    "training_seed": seed if arm.startswith("s") else None,
                    "arm": arm,
                    "on_time_target": target in set(cell["on_time_target_ids"]),
                    "lifecycle_event_counts": dict(sorted(event_counts.items())),
                    "lifecycle_timeline_compacted_from_full_log": selected_events,
                    "communication_decision_keypoints_compacted_from_tape": decisions,
                    "sender_transmission_and_delivery_counts": communications,
                    "sender_emission_target_rows": [item for item in row["emission_sequence"]
                                                     if str(item[1]) == target],
                    "source_paths": {
                        "row": cell["row_path"],
                        "lifecycle": cell["lifecycle_path"],
                        "transcript": cell["transcript_path"],
                        "decision_tape": tape_path,
                    },
                    "information_limit": (
                        "successful recipient delivery rows identify agent and acceptance; the send "
                        "summary has only an aggregate loss count, so dropped-recipient identity is absent"
                    ),
                })
    return chain_list


def external_file_manifest(cells: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    plan = read_json(HERE / "FROZEN_SCENE_AND_ARM_PLAN.json")
    for seed, model in plan["models"]["models"].items():
        add_path(entries, ROOT / model["path"], f"ep128 frozen communication checkpoint {seed}")
    for key, value in plan["configuration"]["round37_frozen_configuration"][
        "frozen_search_and_allocation_weights"
    ].items():
        add_path(entries, ROOT / key, "frozen search/allocation weights")
    import sp_common
    archive_config = Path(sp_common.d1_archive_path("A3", 1100000))
    add_path(entries, archive_config, "archived D1 A3 configuration source")

    # R37 scene used as the gate reproduction control, including its non-Git sidecars.
    for arm in ["s11_original_greedy", "s22_original_greedy", "s33_original_greedy",
                *REFERENCE_ARMS]:
        old_arm = {"s11_original_greedy": "model_s11_greedy",
                   "s22_original_greedy": "model_s22_greedy",
                   "s33_original_greedy": "model_s33_greedy"}.get(arm, arm)
        file_name = (
            f"v2_eval_s{arm[1:3]}_final_greedy_s1600000_a0.json"
            if arm.endswith("original_greedy")
            else f"v2_{({'late_discovery_gate_147': 'late_discovery_gate'}.get(arm, arm))}"
                 f"_s1600000_a1.json"
        )
        old_row_path = R37 / "rows" / file_name
        add_path(entries, old_row_path, f"ROUND37 gate comparison row {old_arm}")
        if old_row_path.is_file():
            old_row = read_json(old_row_path)
            for artifact_name, artifact_path in old_row.get("artefacts", {}).items():
                add_path(entries, artifact_path, f"ROUND37 gate {old_arm} {artifact_name}")

    # New run rows, lifecycle logs, transcripts, tapes, and the real launch ledger stay outside
    # Git; their byte identities make the audit reproducible without committing the raw files.
    for cell in cells.values():
        add_path(entries, cell["row_path"], "ROUND39 result row (Git-external)")
        add_path(entries, cell["lifecycle_path"], "ROUND39 lifecycle log (Git-external)")
        add_path(entries, cell["transcript_path"], "ROUND39 communication transcript (Git-external)")
        if cell.get("tape_path"):
            add_path(entries, cell["tape_path"], "ROUND39 policy decision tape (Git-external)")
    for path in (HERE / "logs").glob("START_LEDGER_p*.json"):
        add_path(entries, path, "ROUND39 real Stage C launch ledger (Git-external)")

    tracked_text = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                                  capture_output=True).stdout
    tracked = {item.decode("utf-8", "surrogateescape").casefold()
               for item in tracked_text.split(b"\0") if item}
    for entry in entries.values():
        rel = entry["repository_relative_path"]
        entry["git_tracked"] = bool(rel and rel.casefold() in tracked)
    return {
        "schema": "round39-git-external-originals/1",
        "files": sorted(entries.values(), key=lambda item: (item["kind"],
                                                              item["repository_relative_path"] or
                                                              item["absolute_path"])),
        "raw_round39_rows_logs_transcripts_tapes_are_not_git_committed": True,
    }


def main() -> None:
    launch = read_json(HERE / "LAUNCH_BOOK.json")
    if launch.get("status") != "complete" or launch.get("starts") != 144 \
            or launch.get("completed") != 144 or launch.get("technical_failures"):
        raise RuntimeError("ROUND39 launch book is not a fully completed 144-cell matrix")
    gate = read_json(HERE / "FIRST_SCENE_GATE.json")
    if not gate.get("all_passed"):
        raise RuntimeError("the first-scene technical gate did not pass")

    cells: dict[tuple[str, int], dict[str, Any]] = {}
    for scene in SCENES:
        for arm in ARMS:
            cells[(arm, scene)] = cell_data(arm, scene)
    matrix = paired_matrix(cells)
    chains = designated_chains(cells)

    # Separate calculations for the 11 predetermined missing target occurrences; no pooled
    # target IDs across scenes, and no claim that a repeated target id is a new independent case.
    r38_cases = read_json(R38 / "MISSING_TARGET_CHAINS.json")["cases"]
    designated = []
    for seed in SEEDS:
        by_condition = {}
        for condition in LEARNING_ARMS:
            arm = f"s{seed}_{condition}"
            rescued = []
            still_missing = []
            for case in r38_cases:
                key = (arm, int(case["scenario"]))
                present = str(case["target_id"]) in set(
                    cells[key]["on_time_target_ids"]
                )
                record = {"scene": int(case["scenario"]),
                          "target_id": str(case["target_id"]),
                          "on_time": present}
                (rescued if present else still_missing).append(record)
            by_condition[condition] = {
                "n_designated_target_occurrences": len(r38_cases),
                "rescued_vs_round37_greedy": len(rescued) if condition == "refresh_held" else 0,
                "present_in_this_arm": len([item for item in rescued + still_missing if item["on_time"]]),
                "still_missing": still_missing,
            }
        designated.append({"seed": seed, "conditions": by_condition})

    s22_equal_all = all(
        cells[("s22_original_greedy", scene)]["on_time_target_ids"]
        == cells[("s22_refresh_held", scene)]["on_time_target_ids"]
        and cells[("s22_original_greedy", scene)]["spent_bytes"]
        == cells[("s22_refresh_held", scene)]["spent_bytes"]
        and all(
            read_json(row_path(f"s22_{condition}", scene))[field]
            == read_json(row_path(f"s22_{other}", scene))[field]
            for field in ("behaviour_sha256", "emission_sequence", "refusal_sequence",
                          "cost_stream", "task_stream")
            for condition, other in (("original_greedy", "refresh_held"),)
        )
        for scene in SCENES
    )
    verdict = {
        "schema": "round39-machine-verdict/1",
        "status": "mechanism_diagnostic_complete",
        "inference_boundary": (
            "post-selected 16-scene mechanism diagnostic; no independent confirmation, "
            "population estimate, significance, noninferiority, or deployment claim"
        ),
        "normal_cells": 144,
        "retry_starts": launch["retries_used"],
        "total_environment_starts": launch["starts"],
        "optimizer_steps": 0,
        "s22_negative_control_equal_all_selected_scenes": s22_equal_all,
        "main_pairwise": matrix["main_paired_refresh_held_minus_original"],
        "designated_11_round37_missing_target_occurrences": designated,
        "first_report_opportunity_changes": matrix["first_report_opportunity_changes"],
        "absolute_vs_immediate_late147_always_hold": matrix["absolute_vs_rule_references"],
        "four_layer_readout": {
            "refresh_suppression": {
                "from_real_tapes_and_transcripts": True,
                "per_seed_scene_counts": "see paired/cell result tables and raw manifest",
            },
            "affordable_first_report_opportunity": (
                "per target occurrence, ever-sequentially-affordable versus original path"
            ),
            "first_report_emitted_and_delivered": (
                "sender first-emission, recipient deliver events, and accepted receiver outcomes "
                "are counted separately"
            ),
            "task_and_cost": "on-time target IDs, new losses, byte splits, and rule gaps are explicit",
        },
        "claim_limits": [
            "the scene block was selected after previous observation for mechanism diagnosis",
            "same scene seed does not force identical packet-loss draws across different send paths",
            "refresh-held is frozen model plus an external refresh constraint",
            "results do not justify enabling the constraint in training or setting a default",
        ],
    }
    write_json("ROUND39_RESULTS.json", {
        "schema": "round39-cell-results/1",
        "cells": [cells[(arm, scene)] for scene in SCENES for arm in ARMS],
    })
    write_json("PAIRED_ANALYSIS.json", matrix)
    write_json("MECHANISM_CHAINS.json", {
        "schema": "round39-designated-mechanism-chains/1",
        "designated_round37_missing_occurrences": len(r38_cases),
        "chains": chains,
        "known_information_gaps": [
            "dropped packet recipient identities are absent from the aggregate sender send record; "
            "successful delivery rows identify recipient and acceptance outcome",
            "task-side gossip/LOS may also supply state, so receipt alone does not establish the "
            "exclusive source of downstream knowledge",
            "different emission paths use their own packet-loss sequence even for the same scenario seed",
        ],
    })
    write_json("GIT_EXTERNAL_ORIGINALS.json", external_file_manifest(cells))
    write_json("MACHINE_VERDICT.json", verdict)
    print(json.dumps({
        "status": verdict["status"],
        "cells": len(cells),
        "mechanism_chains": len(chains),
        "s22_negative_control_equal_all_selected_scenes": s22_equal_all,
        "written": ["ROUND39_RESULTS.json", "PAIRED_ANALYSIS.json",
                    "MECHANISM_CHAINS.json", "GIT_EXTERNAL_ORIGINALS.json",
                    "MACHINE_VERDICT.json"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
