#!/usr/bin/env python3
"""Read-only analysis of the frozen ROUND37 evaluation artefacts.

This file intentionally uses only the Python standard library. It reads saved JSON/JSONL,
checks the ROUND37 per-file SHA256 registry, and writes research reports. It does not import
the simulator, a trainer, a checkpoint loader, or any production module.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVAL = ROOT / "runs" / "comm_final_evaluation_20261002"
OUT = HERE
SCENES = list(range(1_600_000, 1_600_064))
ARMS = [
    "model_s11_greedy",
    "model_s11_sampled",
    "model_s22_greedy",
    "model_s22_sampled",
    "model_s33_greedy",
    "model_s33_sampled",
    "immediate",
    "late_discovery_gate",
    "always_hold",
]
REFERENCES = ["immediate", "late_discovery_gate", "always_hold"]
GREEDY_ARMS = ["model_s11_greedy", "model_s22_greedy", "model_s33_greedy"]
MODEL_ARMS = [arm for arm in ARMS if arm.startswith("model_")]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arm_name(row: dict[str, Any]) -> str:
    if row.get("training_seed") is not None:
        return f"model_s{int(row['training_seed'])}_{row['eval_mode']}"
    label = str(row.get("label", ""))
    rule = re.match(r"^rule_(.+)_s\d+$", label)
    if rule:
        return rule.group(1)
    if row.get("policy") == "immediate":
        return "immediate"
    raise RuntimeError(f"cannot identify arm from row label={label!r}")


def credited_targets(row: dict[str, Any]) -> set[str]:
    targets = [
        str(event["task_id"])
        for event in row.get("task_reward_events", [])
        if event.get("credited") is True
    ]
    if len(targets) != len(set(targets)):
        raise RuntimeError(f"duplicate credited target in {row.get('label')}")
    if int(row["on_time"]) != len(targets):
        raise RuntimeError(
            f"on_time/target identity mismatch in {row.get('label')}: "
            f"{row['on_time']} != {len(targets)}"
        )
    return set(targets)


def load_rows() -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted((EVAL / "rows").glob("*.json")):
        row = read_json(path)
        key = (arm_name(row), int(row["scenario_seed"]))
        if key in result:
            raise RuntimeError(f"duplicate arm/scene row {key}")
        row["_source_path"] = path
        row["_credited_targets"] = sorted(credited_targets(row))
        result[key] = row
    expected = {(arm, scene) for arm in ARMS for scene in SCENES}
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise RuntimeError(f"ROUND37 matrix mismatch: missing={missing[:8]} extra={extra[:8]}")
    return result


def verify_raw_manifest() -> dict[str, Any]:
    registry_path = EVAL / "delivery_draft" / "ARTEFACT_HASH_REGISTRY.json"
    registry = read_json(registry_path)
    raw = registry["raw_artefacts_kept_outside_git"]
    expected_hashes: dict[str, str] = raw["files_with_sha256"]
    expected_sizes: dict[str, int] = raw["bytes"]
    if set(expected_hashes) != set(expected_sizes):
        raise RuntimeError("ROUND37 artifact hash and byte registries have different paths")

    checked = 0
    size_mismatches: list[str] = []
    hash_mismatches: list[str] = []
    missing: list[str] = []
    categories: Counter[str] = Counter()
    byte_totals: Counter[str] = Counter()
    for relative, expected_hash in expected_hashes.items():
        path = EVAL / Path(relative)
        category = relative.split("/", 1)[0]
        categories[category] += 1
        if not path.is_file():
            missing.append(relative)
            continue
        checked += 1
        byte_totals[category] += path.stat().st_size
        if path.stat().st_size != int(expected_sizes[relative]):
            size_mismatches.append(relative)
        if sha256_file(path) != expected_hash:
            hash_mismatches.append(relative)

    ledgers_checked = 0
    ledger_missing: list[str] = []
    ledger_mismatches: list[str] = []
    for name, metadata in registry.get("raw_artefacts_kept_outside_git", {}).get(
        "episode_start_ledgers", {}
    ).items():
        path = EVAL / "logs" / name
        if not path.is_file():
            ledger_missing.append(name)
            continue
        ledgers_checked += 1
        if sha256_file(path) != metadata["sha256"]:
            ledger_mismatches.append(name)

    if missing or size_mismatches or hash_mismatches or ledger_missing or ledger_mismatches:
        raise RuntimeError(
            "ROUND37 raw source manifest verification failed: "
            f"missing={missing[:5]}, size={size_mismatches[:5]}, "
            f"hash={hash_mismatches[:5]}, ledger_missing={ledger_missing}, "
            f"ledger_hash={ledger_mismatches}"
        )
    return {
        "registry": "runs/comm_final_evaluation_20261002/delivery_draft/ARTEFACT_HASH_REGISTRY.json",
        "registry_round": registry["round"],
        "registered_head": registry["head_at_registration"],
        "tracked_deliverables_not_hashed_here": registry["git_tracked_deliverables"],
        "registered_raw_file_count": len(expected_hashes),
        "registered_raw_bytes": int(raw["total_bytes"]),
        "checked_raw_file_count": checked,
        "checked_raw_file_bytes": dict(byte_totals),
        "checked_raw_file_counts": dict(categories),
        "all_raw_file_sha256_match": True,
        "all_raw_file_sizes_match": True,
        "separate_start_ledgers_registered": len(
            raw.get("episode_start_ledgers", {})
        ),
        "separate_start_ledgers_sha256_checked": ledgers_checked,
        "all_start_ledger_sha256_match": True,
        "note": (
            "The check covers the frozen rows, lifecycle/transcript/tape JSONL, claims, and "
            "shards. It does not imply that a file hash alone establishes model identity; "
            "checkpoint identities are inherited from ROUND37's explicit checkpoint registry."
        ),
    }


def totals_for_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counter_keys = [
        "emissions",
        "first_emissions",
        "refresh_emissions",
        "redundant_emissions",
        "transmission_attempts",
        "packets_delivered",
        "packets_lost",
        "deliveries_applied",
        "delivery_superseded",
        "delivery_duplicate",
        "delivery_unknown_task",
        "delivery_refused",
        "budget_refused_sends",
        "capacity_refused_sends",
        "emissions_suppressed_terminal",
        "emissions_forced_maxwait",
        "emissions_other",
        "bytes_policy",
        "bytes_forced_maxwait",
        "bytes_other",
    ]
    summed = {key: 0 for key in counter_keys}
    for row in rows:
        for key in counter_keys:
            summed[key] += int(row.get("relay_counters", {}).get(key, 0))
    return {
        "scenarios": len(rows),
        "on_time_tasks_total": sum(int(row["on_time"]) for row in rows),
        "on_time_tasks_per_episode": sum(int(row["on_time"]) for row in rows) / len(rows),
        "measured_payload_attempt_bytes_total": sum(
            int(row["spent_bytes"]) for row in rows
        ),
        "measured_payload_attempt_bytes_per_episode": sum(
            int(row["spent_bytes"]) for row in rows
        ) / len(rows),
        "expired_targets_total": sum(int(row["expired"]) for row in rows),
        "discovered_targets_total": sum(int(row["discovered"]) for row in rows),
        "spent_bytes_min": min(int(row["spent_bytes"]) for row in rows),
        "spent_bytes_max": max(int(row["spent_bytes"]) for row in rows),
        "budget_bytes_per_episode": int(rows[0]["budget_bytes"]),
        "remaining_budget_bytes_min": min(
            int(row["budget_bytes"]) - int(row["spent_bytes"]) for row in rows
        ),
        "remaining_budget_bytes_max": max(
            int(row["budget_bytes"]) - int(row["spent_bytes"]) for row in rows
        ),
        "budget_spend_ge_90_percent_scenarios": sum(
            int(row["spent_bytes"]) * 10 >= int(row["budget_bytes"]) * 9
            for row in rows
        ),
        **summed,
    }


def metric_ci(statistics: dict[str, Any], arm: str) -> dict[str, Any]:
    summary = statistics["absolute_per_arm"][arm]
    selected = [
        "timely_tasks_per_episode",
        "metered_bytes_per_episode",
        "transmission_attempts_per_episode",
        "emissions_per_episode",
        "packets_delivered_per_episode",
        "packets_lost_per_episode",
        "budget_refused_sends_per_episode",
        "expired_targets_per_episode",
    ]
    return {
        key: summary[key]
        for key in selected
        if key in summary
    }


def paired_differences(
    rows: dict[tuple[str, int], dict[str, Any]], arm: str, reference: str
) -> dict[str, Any]:
    scenes = []
    task_signs: Counter[str] = Counter()
    byte_signs: Counter[str] = Counter()
    missing_targets: list[dict[str, Any]] = []
    added_targets: list[dict[str, Any]] = []
    difference_scenes: set[int] = set()
    for scene in SCENES:
        candidate = rows[(arm, scene)]
        baseline = rows[(reference, scene)]
        candidate_targets = set(candidate["_credited_targets"])
        baseline_targets = set(baseline["_credited_targets"])
        dy = int(candidate["on_time"]) - int(baseline["on_time"])
        dc = int(candidate["spent_bytes"]) - int(baseline["spent_bytes"])
        task_signs["improved" if dy > 0 else "degraded" if dy < 0 else "tied"] += 1
        byte_signs["cheaper" if dc < 0 else "costlier" if dc > 0 else "same"] += 1
        only_reference = sorted(baseline_targets - candidate_targets)
        only_candidate = sorted(candidate_targets - baseline_targets)
        if only_reference or only_candidate:
            difference_scenes.add(scene)
        if only_reference:
            missing_targets.append({"scenario": scene, "target_ids": only_reference})
        if only_candidate:
            added_targets.append({"scenario": scene, "target_ids": only_candidate})
        scenes.append(
            {
                "scenario": scene,
                "task_delta_candidate_minus_reference": dy,
                "measured_bytes_delta_candidate_minus_reference": dc,
                "transmission_attempts_delta_candidate_minus_reference": (
                    int(candidate["relay_counters"]["transmission_attempts"])
                    - int(baseline["relay_counters"]["transmission_attempts"])
                ),
                "emissions_delta_candidate_minus_reference": (
                    int(candidate["emissions"]) - int(baseline["emissions"])
                ),
                "only_reference_on_time_targets": only_reference,
                "only_candidate_on_time_targets": only_candidate,
            }
        )
    return {
        "comparison_direction": f"{arm} minus {reference}",
        "task_scene_counts": {
            "improved": task_signs["improved"],
            "tied": task_signs["tied"],
            "degraded": task_signs["degraded"],
        },
        "byte_scene_counts": {
            "cheaper": byte_signs["cheaper"],
            "same": byte_signs["same"],
            "costlier": byte_signs["costlier"],
        },
        "scenarios_with_identical_on_time_target_sets": 64 - len(difference_scenes),
        "scenarios_with_target_set_difference": len(difference_scenes),
        "only_reference_targets": missing_targets,
        "only_candidate_targets": added_targets,
        "per_scenario": scenes,
    }


def make_absolute_performance(
    rows: dict[tuple[str, int], dict[str, Any]], statistics: dict[str, Any]
) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [rows[(arm, scene)] for scene in SCENES]
        by_arm[arm] = {
            "computed_from_round37_rows": totals_for_arm(arm_rows),
            "precomputed_descriptive_bootstrap_from_round37_statistics": metric_ci(
                statistics, arm
            ),
            "per_scene_absolute_values": [
                {
                    "scenario": scene,
                    "on_time_tasks": int(rows[(arm, scene)]["on_time"]),
                    "on_time_target_ids": rows[(arm, scene)]["_credited_targets"],
                    "measured_payload_attempt_bytes": int(rows[(arm, scene)]["spent_bytes"]),
                    "transmission_attempts": int(
                        rows[(arm, scene)]["relay_counters"]["transmission_attempts"]
                    ),
                    "emissions": int(rows[(arm, scene)]["emissions"]),
                    "budget_refused_sends": int(
                        rows[(arm, scene)]["relay_counters"]["budget_refused_sends"]
                    ),
                }
                for scene in SCENES
            ],
            "paired_vs_reference_arms": {
                reference: paired_differences(rows, arm, reference)
                for reference in REFERENCES
            },
        }

    greedy_target_set_checks = {}
    for arm in GREEDY_ARMS:
        mismatch = []
        for scene in SCENES:
            if set(rows[(arm, scene)]["_credited_targets"]) != set(
                rows[("always_hold", scene)]["_credited_targets"]
            ):
                mismatch.append(scene)
        greedy_target_set_checks[arm] = {
            "identical_to_always_hold_in_all_64_scenes": not mismatch,
            "scenes_compared": 64,
            "mismatch_scenes": mismatch,
        }

    return {
        "schema": "round38-r-absolute-performance/1",
        "source": {
            "rows": "runs/comm_final_evaluation_20261002/rows/*.json",
            "existing_summary": (
                "runs/comm_final_evaluation_20261002/ROUND37_STATISTICS.json"
            ),
            "scenario_block": [SCENES[0], SCENES[-1]],
            "cells": 576,
        },
        "metric_semantics": {
            "on_time_tasks": "number of targets completed before their deadline per episode",
            "measured_bytes": (
                "metered payload-attempt bytes charged by the relay; packet loss is still charged"
            ),
            "transmission_attempts": (
                "recipient transmission attempts; not the number of broadcast emissions"
            ),
            "emissions": (
                "sender-side broadcast events; one event may address multiple recipients"
            ),
            "budget_refusals": "send requests refused for lack of credit, as recorded by the relay",
        },
        "arms": by_arm,
        "greedy_target_set_checks_vs_always_hold": greedy_target_set_checks,
        "analysis_boundary": (
            "The 64-scene block is now used for method analysis and is not an untouched "
            "confirmation block. Paired scene counts do not imply identical message loss streams."
        ),
    }


def summarize_tape(row: dict[str, Any]) -> dict[str, Any]:
    tape_path = Path(row["artefacts"]["tape"])
    if not tape_path.is_file():
        raise RuntimeError(f"learned decision tape missing: {tape_path}")
    decision_rows = 0
    candidates = 0
    actionable_send = 0
    actionable_hold = 0
    masked_budget = 0
    masked_capacity = 0
    masked_first = 0
    masked_refresh = 0
    legal_first_send = 0
    legal_refresh_send = 0
    legal_first_hold = 0
    legal_refresh_hold = 0
    argmax_decision_deviations = 0
    p_send_values: list[float] = []
    masked_first_earliest: dict[str, dict[str, Any]] = {}
    first_budget_mask_step: int | None = None
    refusal_steps = [int(x[0]) for x in row.get("refusal_sequence", [])]
    first_refusal_step = min(refusal_steps) if refusal_steps else None
    legal_candidates_after_refusal = 0
    after_first_budget_mask = False
    legal_candidates_after_budget_mask = 0
    sends_after_budget_mask = 0
    holds_after_budget_mask = 0
    first_report_candidates_after_budget_mask = 0
    refresh_candidates_after_budget_mask = 0

    # The feature layout is comm-send-state/1; index 5 is first_emission in
    # integrated_sar/comm_learning.py::FEATURE_NAMES.
    for record in iter_jsonl(tape_path):
        decision_rows += 1
        targets = record["targets"]
        features = record["features"]
        if len(targets) != len(features):
            raise RuntimeError(f"tape feature rows do not align at {tape_path}")
        actionable = set(record["actionable"])
        masked = set(record["masked"])
        actions = record["actions"]
        requested = record["requested"]
        for index, target in enumerate(targets):
            candidates += 1
            is_first = len(features[index]) > 5 and float(features[index][5]) >= 0.5
            if target in actionable:
                action = int(actions[target])
                if after_first_budget_mask:
                    legal_candidates_after_budget_mask += 1
                    if is_first:
                        first_report_candidates_after_budget_mask += 1
                    else:
                        refresh_candidates_after_budget_mask += 1
                    if action == 1:
                        sends_after_budget_mask += 1
                    else:
                        holds_after_budget_mask += 1
                probability = float(record["send_probabilities"][target])
                p_send_values.append(probability)
                expected_action = 1 if probability >= 0.5 else 0
                argmax_decision_deviations += int(action != expected_action)
                if action == 1 and requested.get(target) is True:
                    actionable_send += 1
                    if is_first:
                        legal_first_send += 1
                    else:
                        legal_refresh_send += 1
                elif action == 0 and requested.get(target) is False:
                    actionable_hold += 1
                    if is_first:
                        legal_first_hold += 1
                    else:
                        legal_refresh_hold += 1
                else:
                    raise RuntimeError(
                        f"action/request mismatch in {tape_path} step {record['step']} target {target}"
                    )
                if first_refusal_step is not None and int(record["step"]) > first_refusal_step:
                    legal_candidates_after_refusal += 1
            elif target in masked:
                reason = str(record.get("masked_reasons", {}).get(target, "unknown"))
                if reason == "budget_exhausted":
                    after_first_budget_mask = True
                    masked_budget += 1
                    masked_first += int(is_first)
                    masked_refresh += int(not is_first)
                    first_budget_mask_step = (
                        int(record["step"])
                        if first_budget_mask_step is None
                        else min(first_budget_mask_step, int(record["step"]))
                    )
                    if is_first:
                        candidate = {
                            "target_id": target,
                            "first_masked_step": int(record["step"]),
                            "remaining_budget_bytes": float(record["shadow_budget_before"]),
                            "attempt_charge_bytes": int(
                                record["attempt_charge_bytes"][target]
                            ),
                            "reason": reason,
                        }
                        prior = masked_first_earliest.get(target)
                        if prior is None or candidate["first_masked_step"] < prior[
                            "first_masked_step"
                        ]:
                            masked_first_earliest[target] = candidate
                elif reason == "capacity_step":
                    masked_capacity += 1

    return {
        "decision_tape_path": str(tape_path.relative_to(ROOT)).replace("\\", "/"),
        "decision_tape_rows": decision_rows,
        "candidate_opportunities": candidates,
        "actionable_send_choices": actionable_send,
        "actionable_hold_choices": actionable_hold,
        "deviations_from_greedy_argmax_rule": argmax_decision_deviations,
        "all_actionable_choices_match_greedy_argmax_rule": (
            argmax_decision_deviations == 0
        ),
        "budget_masked_candidates": masked_budget,
        "capacity_masked_candidates": masked_capacity,
        "budget_masked_first_report_candidates": masked_first,
        "budget_masked_refresh_candidates": masked_refresh,
        "first_budget_mask_step": first_budget_mask_step,
        "earliest_budget_masked_first_report_per_target": [
            masked_first_earliest[target]
            for target in sorted(masked_first_earliest)
        ],
        "legal_choice_probabilities": {
            "count": len(p_send_values),
            "min_p_send": min(p_send_values) if p_send_values else None,
            "median_p_send": statistics.median(p_send_values) if p_send_values else None,
            "max_p_send": max(p_send_values) if p_send_values else None,
            "count_p_send_ge_0_5": sum(value >= 0.5 for value in p_send_values),
            "count_p_send_lt_0_5": sum(value < 0.5 for value in p_send_values),
        },
        "legal_action_by_message_type": {
            "first_report_send": legal_first_send,
            "first_report_hold": legal_first_hold,
            "refresh_send": legal_refresh_send,
            "refresh_hold": legal_refresh_hold,
        },
        "legal_candidate_choices_after_first_actual_refusal": legal_candidates_after_refusal,
        "first_actual_refusal_step": first_refusal_step,
        "legal_candidate_choices_after_first_budget_mask": legal_candidates_after_budget_mask,
        "send_choices_after_first_budget_mask": sends_after_budget_mask,
        "hold_choices_after_first_budget_mask": holds_after_budget_mask,
        "first_report_choices_after_first_budget_mask": first_report_candidates_after_budget_mask,
        "refresh_choices_after_first_budget_mask": refresh_candidates_after_budget_mask,
    }


def read_transcript(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transcript_path = Path(row["artefacts"]["transcript"])
    sends: list[dict[str, Any]] = []
    deliveries: list[dict[str, Any]] = []
    for event in iter_jsonl(transcript_path):
        if event.get("kind") == "send":
            sends.append(event)
        elif event.get("kind") == "deliver":
            deliveries.append(event)
    return sends, deliveries


def summarize_arm_communications(
    rows: dict[tuple[str, int], dict[str, Any]], arm: str
) -> dict[str, Any]:
    per_scene = []
    arm_totals: Counter[str] = Counter()
    receiver_reasons: Counter[str] = Counter()
    send_kind_counts: Counter[str] = Counter()
    send_kind_bytes: Counter[str] = Counter()
    send_kind_attempts: Counter[str] = Counter()
    send_kind_delivered: Counter[str] = Counter()
    send_kind_lost: Counter[str] = Counter()
    send_kind_applied: Counter[str] = Counter()
    send_reason_counts: Counter[str] = Counter()
    decision_aggregate: Counter[str] = Counter()
    decision_probabilities: list[float] = []
    episodes_with_first_report_masked = 0
    episodes_with_budget_refusal = 0
    episodes_with_candidate_after_refusal = 0
    episodes_with_candidate_after_budget_mask = 0

    for scene in SCENES:
        row = rows[(arm, scene)]
        counters = row["relay_counters"]
        sends, deliveries = read_transcript(row)
        send_delivery_map: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
        for delivery in deliveries:
            send_delivery_map[
                (
                    str(delivery["target"]),
                    int(delivery["sent_step"]),
                    int(delivery["revision"]),
                )
            ].append(delivery)

        flags = [bool(item[5]) for item in row.get("emission_sequence", [])]
        if len(flags) != len(sends):
            raise RuntimeError(f"row/transcript emission count mismatch: {arm} {scene}")
        charged_from_sequence = sum(int(item[3]) for item in row["emission_sequence"])
        if charged_from_sequence != int(row["spent_bytes"]):
            raise RuntimeError(f"row emission byte sequence does not close: {arm} {scene}")
        transcript_attempts = sum(int(event.get("attempts", 0)) for event in sends)
        transcript_delivered = sum(int(event.get("packets", 0)) for event in sends)
        transcript_lost = sum(int(event.get("packets_lost", 0)) for event in sends)
        reason_counts = Counter(str(event.get("reason", "unknown")) for event in deliveries)
        if transcript_attempts != int(counters["transmission_attempts"]):
            raise RuntimeError(f"attempt total does not close: {arm} {scene}")
        if transcript_delivered != int(counters["packets_delivered"]):
            raise RuntimeError(f"delivery total does not close: {arm} {scene}")
        if transcript_lost != int(counters["packets_lost"]):
            raise RuntimeError(f"loss total does not close: {arm} {scene}")
        if transcript_attempts != transcript_delivered + transcript_lost:
            raise RuntimeError(f"recipient attempts do not split into delivery/loss: {arm} {scene}")
        if reason_counts.get("accepted", 0) != int(counters["deliveries_applied"]):
            raise RuntimeError(f"accepted/applied total does not close: {arm} {scene}")
        for outcome, counter_key in [
            ("delivery_duplicate", "delivery_duplicate"),
            ("delivery_superseded", "delivery_superseded"),
            ("delivery_unknown_task", "delivery_unknown_task"),
            ("delivery_refused", "delivery_refused"),
        ]:
            if reason_counts.get(outcome, 0) != int(counters[counter_key]):
                raise RuntimeError(
                    f"recipient outcome {outcome} does not close: {arm} {scene}"
                )

        emissions_by_type = {"first_report": [], "refresh": []}
        if sum(flags) != int(counters["first_emissions"]):
            raise RuntimeError(f"first emission flags do not close: {arm} {scene}")
        if len(flags) - sum(flags) != int(counters["refresh_emissions"]):
            raise RuntimeError(f"refresh emission flags do not close: {arm} {scene}")
        for seq, send in zip(row["emission_sequence"], sends):
            first = bool(seq[5])
            kind = "first_report" if first else "refresh"
            send_kind_counts[kind] += 1
            send_kind_bytes[kind] += int(seq[3])
            send_kind_attempts[kind] += int(send.get("attempts", 0))
            send_kind_delivered[kind] += int(send.get("packets", 0))
            send_kind_lost[kind] += int(send.get("packets_lost", 0))
            send_reason_counts[str(send.get("send_reason", "unknown"))] += 1
            send_delivery_events = send_delivery_map.get(
                (str(send["target"]), int(send["step"]), int(send["revision"])), []
            )
            outcome_counts = Counter(str(event.get("reason", "unknown")) for event in send_delivery_events)
            receiver_reasons.update(outcome_counts)
            send_kind_applied[kind] += outcome_counts.get("accepted", 0)
            emissions_by_type[kind].append(
                {
                    "step": int(send["step"]),
                    "target_id": str(send["target"]),
                    "revision": int(send["revision"]),
                    "first_report": first,
                    "charged_bytes": int(seq[3]),
                    "recipient_attempts": int(send.get("attempts", 0)),
                    "delivered_recipient_records": len(send_delivery_events),
                    "delivered_outcomes": dict(outcome_counts),
                    "lost_recipient_count_without_recipient_identity": int(
                        send.get("packets_lost", 0)
                    ),
                    "send_reason": str(send.get("send_reason", "unknown")),
                }
            )

        refused = Counter(str(item[2]) for item in row.get("refusal_sequence", []))
        episodes_with_budget_refusal += refused.get("budget_exhausted", 0) > 0
        tape_summary = summarize_tape(row)
        decision_aggregate["decision_tape_rows"] += int(tape_summary["decision_tape_rows"])
        for name in [
            "candidate_opportunities",
            "actionable_send_choices",
            "actionable_hold_choices",
            "deviations_from_greedy_argmax_rule",
            "budget_masked_candidates",
            "capacity_masked_candidates",
            "budget_masked_first_report_candidates",
            "budget_masked_refresh_candidates",
            "legal_candidate_choices_after_first_actual_refusal",
            "legal_candidate_choices_after_first_budget_mask",
            "send_choices_after_first_budget_mask",
            "hold_choices_after_first_budget_mask",
            "first_report_choices_after_first_budget_mask",
            "refresh_choices_after_first_budget_mask",
        ]:
            decision_aggregate[name] += int(tape_summary[name])
        if tape_summary["earliest_budget_masked_first_report_per_target"]:
            episodes_with_first_report_masked += 1
        if int(tape_summary["legal_candidate_choices_after_first_actual_refusal"]) > 0:
            episodes_with_candidate_after_refusal += 1
        if int(tape_summary["legal_candidate_choices_after_first_budget_mask"]) > 0:
            episodes_with_candidate_after_budget_mask += 1
        pstats = tape_summary["legal_choice_probabilities"]
        if pstats["count"]:
            decision_probabilities.extend(
                [float(pstats["min_p_send"]), float(pstats["max_p_send"])]
            )
        for key, value in reason_counts.items():
            receiver_reasons[key] += 0  # reasons are counted once below at arm scope

        arm_totals["episodes"] += 1
        for key in [
            "spent_bytes",
            "emissions",
        ]:
            arm_totals[key] += int(row[key])
        for key, value in counters.items():
            if isinstance(value, (int, float)):
                arm_totals[f"relay_{key}"] += int(value)
        arm_totals["budget_refusals_from_sequence"] += sum(refused.values())

        per_scene.append(
            {
                "scenario": scene,
                "on_time_tasks": int(row["on_time"]),
                "measured_payload_attempt_bytes": int(row["spent_bytes"]),
                "budget_bytes": int(row["budget_bytes"]),
                "remaining_budget_bytes": int(row["budget_bytes"]) - int(row["spent_bytes"]),
                "sender_emissions": len(sends),
                "sender_emissions_by_type": {
                    "first_report": emissions_by_type["first_report"],
                    "refresh": emissions_by_type["refresh"],
                },
                "recipient_attempts": transcript_attempts,
                "recipient_deliveries": transcript_delivered,
                "recipient_losses_without_agent_identity": transcript_lost,
                "recipient_delivery_outcomes": dict(reason_counts),
                "budget_refusals": dict(refused),
                "decision": tape_summary,
            }
        )

    total_episodes = len(SCENES)
    receiver_reasons = Counter()
    for scene in SCENES:
        _, deliveries = read_transcript(rows[(arm, scene)])
        receiver_reasons.update(str(event.get("reason", "unknown")) for event in deliveries)

    return {
        "training_seed": int(arm.split("_")[1][1:]),
        "mode": "greedy",
        "scenario_count": total_episodes,
        "sender_side": {
            "charged_payload_attempt_bytes_total": arm_totals["spent_bytes"],
            "charged_payload_attempt_bytes_per_episode": arm_totals["spent_bytes"] / total_episodes,
            "emissions_total": arm_totals["emissions"],
            "emissions_per_episode": arm_totals["emissions"] / total_episodes,
            "first_report_emissions": send_kind_counts["first_report"],
            "first_report_bytes": send_kind_bytes["first_report"],
            "first_report_recipient_attempts": send_kind_attempts["first_report"],
            "first_report_delivered": send_kind_delivered["first_report"],
            "first_report_lost": send_kind_lost["first_report"],
            "first_report_accepted": send_kind_applied["first_report"],
            "refresh_emissions": send_kind_counts["refresh"],
            "refresh_bytes": send_kind_bytes["refresh"],
            "refresh_recipient_attempts": send_kind_attempts["refresh"],
            "refresh_delivered": send_kind_delivered["refresh"],
            "refresh_lost": send_kind_lost["refresh"],
            "refresh_accepted": send_kind_applied["refresh"],
            "automatic_packet_retries": (
                "not recorded as a separate event type; each send is a broadcast emission "
                "with recipient attempts, and later non-first sends are classified as refreshes"
            ),
            "send_reason_counts": dict(send_reason_counts),
            "budget_refused_sends": int(arm_totals["relay_budget_refused_sends"]),
            "capacity_refused_sends": int(arm_totals["relay_capacity_refused_sends"]),
            "episodes_with_budget_refusal": int(episodes_with_budget_refusal),
            "budget_exhaustion_treatment": (
                "The collector masks non-affordable candidates to HOLD before execution; "
                "this differs from an executor refusal."
            ),
        },
        "receiver_side": {
            "delivered_recipient_outcomes": dict(receiver_reasons),
            "delivery_applied_accepted_count": int(
                arm_totals["relay_deliveries_applied"]
            ),
            "superseded_count": int(arm_totals["relay_delivery_superseded"]),
            "duplicate_count": int(arm_totals["relay_delivery_duplicate"]),
            "unknown_task_count": int(arm_totals["relay_delivery_unknown_task"]),
            "refused_delivery_count": int(arm_totals["relay_delivery_refused"]),
            "lost_recipient_count": int(arm_totals["relay_packets_lost"]),
            "loss_identity_limit": (
                "Losses are recorded per emission as aggregate recipient counts. The logs do not "
                "identify which receiver lost each packet."
            ),
            "broadcast_accounting_note": (
                "Sender emissions and charged bytes are counted once per broadcast at the sender; "
                "recipient attempts/deliveries/losses are separate receiver-side counts."
            ),
        },
        "decision_and_budget_mask": {
            "candidate_opportunities_total": int(decision_aggregate["candidate_opportunities"]),
            "actionable_send_choices": int(decision_aggregate["actionable_send_choices"]),
            "actionable_hold_choices": int(decision_aggregate["actionable_hold_choices"]),
            "deviations_from_greedy_argmax_rule": int(
                decision_aggregate["deviations_from_greedy_argmax_rule"]
            ),
            "all_actionable_choices_match_greedy_argmax_rule": (
                decision_aggregate["deviations_from_greedy_argmax_rule"] == 0
            ),
            "budget_masked_candidates": int(decision_aggregate["budget_masked_candidates"]),
            "capacity_masked_candidates": int(decision_aggregate["capacity_masked_candidates"]),
            "budget_masked_first_report_candidates": int(
                decision_aggregate["budget_masked_first_report_candidates"]
            ),
            "budget_masked_refresh_candidates": int(
                decision_aggregate["budget_masked_refresh_candidates"]
            ),
            "episodes_with_first_report_masked_by_budget": int(
                episodes_with_first_report_masked
            ),
            "actual_refusal_count": int(arm_totals["relay_budget_refused_sends"]),
            "legal_candidate_choices_after_first_actual_refusal": int(
                decision_aggregate["legal_candidate_choices_after_first_actual_refusal"]
            ),
            "legal_candidate_choices_after_first_budget_mask": int(
                decision_aggregate["legal_candidate_choices_after_first_budget_mask"]
            ),
            "episodes_with_legal_candidate_after_first_budget_mask": int(
                episodes_with_candidate_after_budget_mask
            ),
            "send_choices_after_first_budget_mask": int(
                decision_aggregate["send_choices_after_first_budget_mask"]
            ),
            "hold_choices_after_first_budget_mask": int(
                decision_aggregate["hold_choices_after_first_budget_mask"]
            ),
            "first_report_choices_after_first_budget_mask": int(
                decision_aggregate["first_report_choices_after_first_budget_mask"]
            ),
            "refresh_choices_after_first_budget_mask": int(
                decision_aggregate["refresh_choices_after_first_budget_mask"]
            ),
            "post_refusal_opportunity_status": (
                "not applicable: there were no budget refusals"
                if int(arm_totals["relay_budget_refused_sends"]) == 0
                else "see per-scene decision summaries"
            ),
            "p_send_on_legal_candidates_min": (
                min(decision_probabilities) if decision_probabilities else None
            ),
            "p_send_on_legal_candidates_max": (
                max(decision_probabilities) if decision_probabilities else None
            ),
            "episodes_with_legal_first_report_candidate_masked": (
                int(episodes_with_first_report_masked)
            ),
            "episodes_with_legal_candidate_after_first_actual_refusal": int(
                episodes_with_candidate_after_refusal
            ),
        },
        "per_scene": per_scene,
    }


def path_for_target(
    rows: dict[tuple[str, int], dict[str, Any]],
    arm: str,
    scene: int,
    target: str,
) -> dict[str, Any]:
    row = rows[(arm, scene)]
    lifecycle_path = Path(row["artefacts"]["lifecycle"])
    transcript_path = Path(row["artefacts"]["transcript"])
    tape_path = Path(row["artefacts"]["tape"])
    lifecycle = [
        event
        for event in iter_jsonl(lifecycle_path)
        if str(event.get("target_id")) == target
    ]
    lifecycle.sort(
        key=lambda event: (
            float(event.get("timestamp") or 0.0),
            str(event.get("event_type") or ""),
            int(event.get("task_revision") or 0),
        )
    )
    detected = next(
        (event for event in lifecycle if event.get("event_type") == "detection"),
        None,
    )
    related_reward_events = [
        event
        for event in row.get("task_reward_events", [])
        if str(event.get("task_id")) == target
    ]

    sends: list[dict[str, Any]] = []
    deliveries_by_send: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    if transcript_path.is_file():
        for event in iter_jsonl(transcript_path):
            if event.get("kind") == "deliver" and str(event.get("target")) == target:
                deliveries_by_send[
                    (
                        target,
                        int(event.get("sent_step", event.get("step", -1))),
                        int(event.get("revision", -1)),
                    )
                ].append(event)
    emission_sequence = {
        (str(item[1]), int(item[0]), int(item[2])): item
        for item in row.get("emission_sequence", [])
    }
    if transcript_path.is_file():
        for event in iter_jsonl(transcript_path):
            if event.get("kind") != "send" or str(event.get("target")) != target:
                continue
            key = (target, int(event["step"]), int(event["revision"]))
            recipient_events = deliveries_by_send.get(key, [])
            seq = emission_sequence.get(key)
            sends.append(
                {
                    "step": int(event["step"]),
                    "clock_now": event.get("now"),
                    "target_id": target,
                    "revision": int(event["revision"]),
                    "first_report": bool(event.get("first")),
                    "send_reason": event.get("send_reason"),
                    "charged_bytes": int(event.get("cost_bytes", 0)),
                    "recipient_attempts": int(event.get("attempts", 0)),
                    "delivered_count": int(event.get("packets", len(recipient_events))),
                    "lost_count_without_receiver_identity": int(
                        event.get("packets_lost", 0)
                    ),
                    "delivered_receiver_results": [
                        {
                            "agent": str(item.get("agent")),
                            "accepted": bool(item.get("accepted")),
                            "reason": str(item.get("reason", "unknown")),
                        }
                        for item in recipient_events
                    ],
                    "row_emission_sequence_matches": seq is not None,
                }
            )
    sends.sort(key=lambda event: (event["step"], event["revision"]))

    decision_events = []
    first_affordable = None
    first_budget_masked = None
    first_policy_send_request = None
    first_observed_decision = None
    masked_opportunities: Counter[str] = Counter()
    masked_steps: list[int] = []
    if tape_path.is_file():
        for record in iter_jsonl(tape_path):
            if target not in record.get("targets", []):
                continue
            index = record["targets"].index(target)
            feature = record["features"][index]
            is_first = len(feature) > 5 and float(feature[5]) >= 0.5
            actionable = target in record.get("actionable", [])
            masked = target in record.get("masked", [])
            reason = record.get("masked_reasons", {}).get(target)
            event = {
                "step": int(record["step"]),
                "candidate_type": "first_report" if is_first else "refresh",
                "actionable_and_affordable": actionable,
                "masked": masked,
                "mask_reason": reason,
                "p_send": record.get("send_probabilities", {}).get(target),
                "action": (
                    "SEND"
                    if int(record.get("actions", {}).get(target, 0)) == 1
                    else "HOLD"
                ),
                "policy_requested_send": bool(
                    record.get("requested", {}).get(target, False)
                ),
                "remaining_budget_before_decision": float(
                    record.get("shadow_budget_before", 0.0)
                ),
                "attempt_charge_bytes": record.get("attempt_charge_bytes", {}).get(target),
            }
            if first_observed_decision is None:
                first_observed_decision = event
            if actionable and first_affordable is None:
                first_affordable = event
            if masked and reason == "budget_exhausted" and first_budget_masked is None:
                first_budget_masked = event
            if masked and reason:
                masked_opportunities[str(reason)] += 1
                masked_steps.append(int(record["step"]))
            if event["policy_requested_send"] and first_policy_send_request is None:
                first_policy_send_request = event
            decision_events.append(event)

    final_lifecycle = next(
        (event for event in reversed(lifecycle) if event.get("event_type") == "final"),
        lifecycle[-1] if lifecycle else None,
    )
    on_time = target in set(row["_credited_targets"])
    return {
        "arm": arm,
        "scenario": scene,
        "target_id": target,
        "on_time": on_time,
        "row_outcome": {
            "episode_on_time_tasks": int(row["on_time"]),
            "episode_expired_targets": int(row["expired"]),
            "spent_bytes": int(row["spent_bytes"]),
            "budget_bytes": int(row["budget_bytes"]),
        },
        "discovery": (
            {
                "timestamp": detected.get("timestamp"),
                "observed_by": detected.get("discovered_by"),
                "source_event_id": detected.get("source_event_id"),
                "detection_event_id": (detected.get("detail") or {}).get(
                    "detection_event_id"
                ),
                "deadline": (
                    (final_lifecycle.get("detail") or {}).get("deadline")
                    if final_lifecycle
                    else None
                ),
            }
            if detected
            else None
        ),
        "first_affordable_decision_opportunity": (
            {
                key: first_affordable[key]
                for key in [
                    "step",
                    "candidate_type",
                    "p_send",
                    "action",
                    "policy_requested_send",
                    "remaining_budget_before_decision",
                    "attempt_charge_bytes",
                ]
            }
            if first_affordable
            else None
        ),
        "first_observed_policy_decision": (
            {
                key: first_observed_decision[key]
                for key in [
                    "step",
                    "candidate_type",
                    "actionable_and_affordable",
                    "masked",
                    "mask_reason",
                    "p_send",
                    "action",
                    "policy_requested_send",
                    "remaining_budget_before_decision",
                    "attempt_charge_bytes",
                ]
            }
            if first_observed_decision
            else None
        ),
        "first_budget_masked_decision_opportunity": first_budget_masked,
        "first_policy_send_request": (
            {
                key: first_policy_send_request[key]
                for key in [
                    "step",
                    "candidate_type",
                    "p_send",
                    "action",
                    "policy_requested_send",
                    "remaining_budget_before_decision",
                    "attempt_charge_bytes",
                ]
            }
            if first_policy_send_request
            else None
        ),
        "learned_decision_tape_summary_for_target": {
            "candidate_observations": len(decision_events),
            "first_mask_step": min(masked_steps) if masked_steps else None,
            "last_mask_step": max(masked_steps) if masked_steps else None,
            "masked_opportunities_by_reason": dict(masked_opportunities),
            "first_actual_send_step": min(
                (event["step"] for event in sends), default=None
            ),
            "policy_send_request_count": sum(
                bool(event["policy_requested_send"]) for event in decision_events
            ),
        },
        "actual_transmissions": sends,
        "task_reward_events": [
            {
                "event_id": event.get("event_id"),
                "event_time": event.get("event_time"),
                "relay_tick": event.get("relay_tick"),
                "clock_now": event.get("clock_now"),
                "accepted": event.get("accepted"),
                "on_time": event.get("on_time"),
                "credited": event.get("credited"),
            }
            for event in related_reward_events
        ],
        "assignment_and_job_timeline": [
            {
                "timestamp": event.get("timestamp"),
                "event_type": event.get("event_type"),
                "assigned_agents": event.get("assigned_agents"),
                "stage_index": event.get("stage_index"),
                "progress": event.get("progress"),
                "status": event.get("status"),
                "failure_reason": event.get("failure_reason"),
                "task_revision": event.get("task_revision"),
                "detail": event.get("detail"),
            }
            for event in lifecycle
        ],
        "terminal": (
            {
                "event_type": final_lifecycle.get("event_type"),
                "timestamp": final_lifecycle.get("timestamp"),
                "status": final_lifecycle.get("status"),
                "failure_reason": final_lifecycle.get("failure_reason"),
                "stage_index": final_lifecycle.get("stage_index"),
                "progress": final_lifecycle.get("progress"),
            }
            if final_lifecycle
            else None
        ),
        "evidence_limits": [
            (
                "Receiver losses are counted in aggregate on each send; the identity of the "
                "specific receiver losing a packet is not recorded."
            ),
            (
                "The relay transcript records accepted/superseded delivery results but does not "
                "record which gossip or line-of-sight source supplied a newer receiver belief."
            ),
            (
                "Paired scenario seeds do not imply identical per-message losses after arms send "
                "different numbers of packets."
            ),
            (
                "Only learned runs carry a candidate decision tape. Rule runs expose actual sends "
                "and lifecycle events, not per-candidate HOLD/affordability rows."
            ),
        ],
    }


def make_missing_target_chains(
    rows: dict[tuple[str, int], dict[str, Any]]
) -> dict[str, Any]:
    by_case: dict[tuple[int, str], list[str]] = defaultdict(list)
    per_model_differences = {}
    for arm in GREEDY_ARMS:
        differences = []
        for scene in SCENES:
            immediate_targets = set(rows[("immediate", scene)]["_credited_targets"])
            learned_targets = set(rows[(arm, scene)]["_credited_targets"])
            for target in sorted(immediate_targets - learned_targets):
                by_case[(scene, target)].append(arm)
                differences.append({"scenario": scene, "target_id": target})
        per_model_differences[arm] = differences

    cases = []
    for (scene, target), missing_models in sorted(by_case.items()):
        learned = {}
        sampled_context = {}
        for seed in (11, 22, 33):
            greedy = f"model_s{seed}_greedy"
            sampled = f"model_s{seed}_sampled"
            learned[str(seed)] = {
                "is_missing_from_immediate_on_time_set": greedy in missing_models,
                "path": path_for_target(rows, greedy, scene, target),
            }
            sampled_context[str(seed)] = path_for_target(
                rows, sampled, scene, target
            )
        cases.append(
            {
                "paired_occurrence_id": f"{scene}:{target}",
                "scenario": scene,
                "target_id": target,
                "missing_from_greedy_seeds": sorted(
                    int(arm.split("_")[1][1:]) for arm in missing_models
                ),
                "reference_paths": {
                    reference: path_for_target(rows, reference, scene, target)
                    for reference in [
                        "immediate",
                        "late_discovery_gate",
                        "always_hold",
                    ]
                },
                "greedy_paths_by_training_seed": learned,
                "sampled_context_by_training_seed": sampled_context,
                "interpretation_limit": (
                    "These are paired path observations from separate runs, not a causal "
                    "attribution. Accepted delivery, superseded delivery, or budget availability "
                    "alone does not establish task benefit or counterfactual rescue."
                ),
            }
        )

    return {
        "schema": "round38-r-missing-target-chains/1",
        "comparison": (
            "For each greedy model: immediate on-time target IDs minus that model's IDs, "
            "computed from all 64 row-level task_reward_events."
        ),
        "scene_block": [SCENES[0], SCENES[-1]],
        "greedy_target_differences": per_model_differences,
        "unique_paired_scene_target_occurrences": len(cases),
        "cases": cases,
        "information_gaps": [
            "No receiver-belief/gossip trace identifies who supplied a higher revision before a superseded delivery.",
            "Rule arms have no per-candidate decision tape, so their HOLD and unaffordable opportunities cannot be separated from saved output alone.",
            "Loss recipient identities are not stored; only aggregate loss count per emission is available.",
            "Same scenario seed is not the same channel realization when arms send different packet counts.",
        ],
    }


def make_source_inventory(
    manifest_check: dict[str, Any],
    rows: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    input_paths = [
        "runs/comm_final_evaluation_20261002/ROUND37_EVALUATION_REPORT.md",
        "runs/comm_final_evaluation_20261002/RESULTS_TABLES.md",
        "runs/comm_final_evaluation_20261002/ROUND37_STATISTICS.json",
        "runs/comm_final_evaluation_20261002/ROUND37_REPORT_CHECK.json",
        "runs/comm_final_evaluation_20261002/ROUND37_GATE.json",
        "runs/comm_final_evaluation_20261002/EVALUATION_IDENTITY.json",
        "runs/comm_final_evaluation_20261002/EVALUATION_LEDGER_RECONCILIATION.json",
        "runs/comm_final_evaluation_20261002/SCENE_USAGE_AUDIT.json",
        "runs/comm_final_evaluation_20261002/delivery_draft/MODEL_AND_CONFIG_INVENTORY.md",
        "runs/comm_final_evaluation_20261002/delivery_draft/ARCHITECTURE_AND_EVENT_ORDER.md",
        "runs/comm_final_evaluation_20261002/delivery_draft/FIX_BOUNDARY_AND_HISTORY.md",
        "runs/comm_final_train_20261001/ROUND36_TRAIN_REPORT.md",
        "runs/comm_final_train_20261001/delivery_draft/ARTEFACT_HASH_REGISTRY.json",
        "runs/comm_final_train_20261001/seed11/CHECKPOINT_HASHES.json",
        "runs/comm_final_train_20261001/seed22/CHECKPOINT_HASHES.json",
        "runs/comm_final_train_20261001/seed33/CHECKPOINT_HASHES.json",
        "runs/comm_final_train_20261001/seed11/FREEZE_MANIFEST.json",
        "runs/comm_final_train_20261001/seed22/FREEZE_MANIFEST.json",
        "runs/comm_final_train_20261001/seed33/FREEZE_MANIFEST.json",
        "runs/round26_analysis_20260929/ROUND26_ANALYSIS.md",
        "runs/round26_analysis_20260929/SCENE_1400031_DIAGNOSIS.json",
        "runs/comm_cost_firstreport_intervention_20260929/ROUND27_REPORT.md",
        "runs/comm_cost_firstreport_intervention_20260929/ROUND27_VERDICT.json",
        "runs/comm_cost_firstreport_diagnosis_20260929/ROUND28_REPORT.md",
        "runs/comm_cost_firstreport_diagnosis_20260929/ROUND28_VERDICT.json",
        "integrated_sar/comm.py",
        "integrated_sar/comm_policy.py",
        "integrated_sar/comm_learning.py",
        "runs/comm_budget_action_mask_fix_20260922/budget_masked.py",
        "runs/comm_cost_stage_b_20260927/sb_collector.py",
        "runs/comm_cost_v2_rerun_20260928/v2_rules.py",
        "runs/comm_cost_v2_rerun_20260928/v2_run.py",
        "runs/comm_final_evaluation_20261002/r37_run.py",
        "runs/comm_final_evaluation_20261002/r37_stats.py",
    ]
    inputs = []
    missing = []
    for relative in input_paths:
        path = ROOT / Path(relative)
        if not path.is_file():
            missing.append(relative)
            continue
        inputs.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": "round38-r-source-inventory/1",
        "round37_raw_manifest_verification": manifest_check,
        "analysis_inputs": inputs,
        "analysis_script": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "missing_analysis_inputs": missing,
        "round37_checkpoint_byte_checks_without_deserialization": [
            {
                "training_seed": seed,
                "checkpoint_path": rows[(f"model_s{seed}_greedy", SCENES[0])][
                    "checkpoint_load_identity"
                ]["checkpoint"],
                "checkpoint_bytes": (
                    ROOT
                    / rows[(f"model_s{seed}_greedy", SCENES[0])][
                        "checkpoint_load_identity"
                    ]["checkpoint"]
                ).stat().st_size,
                "sha256_from_current_file": sha256_file(
                    ROOT
                    / rows[(f"model_s{seed}_greedy", SCENES[0])][
                        "checkpoint_load_identity"
                    ]["checkpoint"]
                ),
                "sha256_recorded_by_round37_read_only_load": rows[
                    (f"model_s{seed}_greedy", SCENES[0])
                ]["checkpoint_load_identity"]["checkpoint_bytes_sha256"],
                "byte_hash_matches": (
                    sha256_file(
                        ROOT
                        / rows[(f"model_s{seed}_greedy", SCENES[0])][
                            "checkpoint_load_identity"
                        ]["checkpoint"]
                    )
                    == rows[(f"model_s{seed}_greedy", SCENES[0])][
                        "checkpoint_load_identity"
                    ]["checkpoint_bytes_sha256"]
                ),
                "parameter_summary_from_round37": rows[
                    (f"model_s{seed}_greedy", SCENES[0])
                ]["checkpoint_load_identity"]["parameter_sha256"],
                "parameter_content_recomputed_in_this_analysis": False,
                "round37_recorded_load_mode": rows[
                    (f"model_s{seed}_greedy", SCENES[0])
                ]["checkpoint_load_identity"]["load_mode"],
                "episodes_done": rows[
                    (f"model_s{seed}_greedy", SCENES[0])
                ]["checkpoint_load_identity"]["episodes_done"],
                "updates": rows[
                    (f"model_s{seed}_greedy", SCENES[0])
                ]["checkpoint_load_identity"]["updates"],
                "lambda_in_state": rows[
                    (f"model_s{seed}_greedy", SCENES[0])
                ]["checkpoint_load_identity"]["lambda_in_state"],
            }
            for seed in (11, 22, 33)
        ],
        "source_reading_notes": [
            "ROUND37 rows are the primary per-cell absolute results and task target identities.",
            "ROUND37 lifecycle JSONL carries detection, assignment, progress, deadline and terminal records.",
            "ROUND37 communication transcript JSONL carries sender emissions and successful per-recipient outcomes.",
            "ROUND37 learned decision tapes carry per-candidate action, probability, executability and mask state.",
            "The fixed evaluation used decision_trace=null; allocation/belief internals beyond lifecycle records are not present.",
        ],
        "analysis_runtime": {
            "python_version": sys.version.split()[0],
            "interpreter": r"D:\project_env\python.exe",
            "dependencies": "Python standard library only",
            "project_modules_imported": False,
            "simulator_or_environment_entrypoints_called": False,
            "checkpoint_deserialization_or_forward_called": False,
            "network_used": False,
        },
        "execution_counts_for_this_round": {
            "new_environment_starts": 0,
            "new_streams": 0,
            "real_optimizer_steps": 0,
            "synthetic_optimizer_steps": 0,
        },
    }


def main() -> None:
    manifest_check = verify_raw_manifest()
    rows = load_rows()
    statistics = read_json(EVAL / "ROUND37_STATISTICS.json")
    absolute = make_absolute_performance(rows, statistics)
    communication = {
        "schema": "round38-r-communication-usage/1",
        "source_rows": "runs/comm_final_evaluation_20261002/rows/*.json",
        "source_sender_receiver_trace": (
            "runs/comm_final_evaluation_20261002/logs/*_ct.jsonl"
        ),
        "source_learned_action_trace": (
            "runs/comm_final_evaluation_20261002/logs/*_tape.jsonl"
        ),
        "metric_semantics": {
            "charged_bytes": (
                "sender-side payload-attempt bytes charged by the relay, independent of "
                "whether an individual recipient receives the packet"
            ),
            "emission": "one sender-side broadcast",
            "attempt": "one recipient attempt within a broadcast",
            "first_report": (
                "an emission whose recorded first flag is true for that target; it does not "
                "guarantee a successful or accepted receiver delivery"
            ),
            "refresh": (
                "a later non-first emission for the target; the transcript has no separate "
                "automatic retry event class"
            ),
        },
        "by_greedy_training_seed": {
            arm: summarize_arm_communications(rows, arm) for arm in GREEDY_ARMS
        },
        "cross_seed_notes": [
            (
                "Per-receiver delivery outcomes are counted separately from sender emissions; "
                "a broadcast can create multiple recipient outcomes."
            ),
            (
                "A superseded result shows that the receiver rejected that arriving revision as "
                "older than its known view; the source and time of that other view are not logged."
            ),
            (
                "An accepted/applied message does not by itself establish assignment or task benefit."
            ),
        ],
    }
    chains = make_missing_target_chains(rows)
    source_inventory = make_source_inventory(manifest_check, rows)
    if source_inventory["missing_analysis_inputs"]:
        raise RuntimeError(
            f"missing required analysis sources: {source_inventory['missing_analysis_inputs']}"
        )
    if not all(
        checkpoint["byte_hash_matches"]
        for checkpoint in source_inventory[
            "round37_checkpoint_byte_checks_without_deserialization"
        ]
    ):
        raise RuntimeError("a current ROUND36 checkpoint byte hash differs from ROUND37 identity")

    write_json(OUT / "ABSOLUTE_PERFORMANCE.json", absolute)
    write_json(OUT / "COMMUNICATION_USAGE.json", communication)
    write_json(OUT / "MISSING_TARGET_CHAINS.json", chains)
    write_json(OUT / "INPUT_SOURCE_INVENTORY.json", source_inventory)

    print(
        json.dumps(
            {
                "row_count": len(rows),
                "matrix_complete": True,
                "raw_hashes_checked": manifest_check["checked_raw_file_count"],
                "raw_hashes_match": manifest_check["all_raw_file_sha256_match"],
                "missing_target_occurrences_per_greedy": {
                    arm: len(chains["greedy_target_differences"][arm])
                    for arm in GREEDY_ARMS
                },
                "unique_paired_missing_target_instances": chains[
                    "unique_paired_scene_target_occurrences"
                ],
                "outputs": [
                    "ABSOLUTE_PERFORMANCE.json",
                    "COMMUNICATION_USAGE.json",
                    "MISSING_TARGET_CHAINS.json",
                    "INPUT_SOURCE_INVENTORY.json",
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
