"""ROUND46-R2 V7: eval-only adoption, fixed quota recovery, and reporting.

The imported R40/R42/R44/R45 evaluation machinery remains the execution path.
This module owns only this task's adoption decision, private output namespace,
fixed claim map, cumulative ledger, and analysis handoff.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


TASK_ID = "ROUND46-R2-V7-EVAL-ONLY-COMPLETE"
BRANCH = "fix/round46-r2-v7-eval-only"
OLD_HEAD = "92bf455ea3a85cae727110a9552e92a5804bcd25"
CODE_VERSION = "round46-r2-v7-eval-only/1"
BASE_NORMAL = 386
BASE_STEPS = 386_000
BASE_PPO = 96
BASE_ADAM = 688
BASE_CHECKPOINT_WRITES = 318
NORMAL_SLOTS = tuple(range(387, 481))
RESERVE_CAP = 24
HARD_CAP = 504
SEEDS = (11, 22, 33)
MODES = ("greedy", "sampled")
SCENES = tuple(range(1_301_000, 1_301_016))
OLD_KEY = (11, "greedy", 1_301_000)
OLD_CLAIM_SHA = "4500b67dcce2665d78dbf547a491ea79bbf2a96cc27a0e24b3d65d8c9a9f255d"
OLD_ROW_SHA = "43b86041adbf5d7c5e72962ddb018be6b4deeb4aa6a339a515bb584d6c69dcdd"
OLD_ARTIFACT_SHAS = {
    "lifecycle": "62bb46e7f76b210097d8ba337e82d59c30b249eb9e836a0e4f3e352aea42c838",
    "transcript": "590355f8a077ff790007463d1a2f76eb79c1c92ffc04f9117d1f74b2c36c5579",
    "tape": "18f0e12cd41cfe0a395c74e6d5cbc4386010cb8cd3617977fd020a98112a1e8f",
}
MODEL_FILE_SHA = "0f161663aaf963552f8f02d6363f3c52e6f67119217f835c501cb2d0705bfc10"
MODEL_PARAMETER_SHA = "8b87f4d2591a3a0c573b71a08ec8444db1e3f91850edbf2f2b958dddb31e5670"
OLD_EXECUTOR_SHA = "ac857a7ebaa4ab425111a135590b7c5061e93773a8eaa4d2aa962704cb423180"
OLD_INPUT_HEAD = "4e989a9088111ce6d9171cb1a21629a4b2b1e861"
PROTECTED = {
    "runs/comm_bc_initialized_constrained_ppo_20261007_r2/ROUND46_CURRENT_TASK.json":
        "cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b",
    "runs/comm_bc_initialized_constrained_ppo_20261007_r2/ROUND46_R2_V6_STOP_STATE.json":
        "93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18",
}

HERE = Path(__file__).resolve().parent
V7_ROOT = HERE / "v7_eval_only"
V6_ROOT = HERE / "v6"
V7_RUNTIME = V7_ROOT / "runtime"
V7_ROWS = V7_ROOT / "rows"
V7_LOGS = V7_ROOT / "logs"
V7_CLAIMS = V7_ROOT / "launch_claims"
V7_SLOTS = V7_ROOT / "launch_slots"
V7_RECEIPTS = V7_ROOT / "receipts"
V7_LEDGER = V7_RUNTIME / "ROUND46_R2_V7_EXECUTION_LEDGER.json"
V7_LEDGER_EVENTS = V7_RUNTIME / "ledger_events"
V7_CURSOR = HERE / "ROUND46_R2_V7_CURRENT_CURSOR.json"
ENTRY = HERE / "ROUND46_R2_V7_EXECUTION_ENTRY.md"


class V7Error(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_new_json(path: Path, value: Any) -> str:
    """Create an immutable receipt; an identical replay is an idempotent read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    raw = payload.encode("utf-8")
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = path.read_bytes()
        if existing != raw:
            raise V7Error(f"immutable V7 receipt conflicts: {path}")
        return sha256_bytes(existing)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_bytes(raw)


def write_json_atomic(path: Path, value: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256_bytes(raw)


def fixed_eval_keys() -> tuple[tuple[int, str, int], ...]:
    return tuple((seed, mode, scene) for seed in SEEDS
                 for mode in MODES for scene in SCENES)


def logical_job(key: tuple[int, str, int]) -> str:
    seed, mode, scene = key
    return f"model_eval_new_s{seed}_{mode}_scene{scene}"


def build_quota_plan(*, adoption_passed: bool) -> dict[str, Any]:
    """Freeze the authorized 95-cell or 96-cell recovery map before claims."""
    keys = list(fixed_eval_keys())
    entries: list[dict[str, Any]] = []
    if adoption_passed:
        remaining = keys[1:]
        for index, key in enumerate(remaining):
            reserve = index == 94
            entries.append({
                "key": list(key), "logical_job": logical_job(key), "attempt": 1,
                "category": "reserve" if reserve else "normal",
                "slot": 1 if reserve else NORMAL_SLOTS[index],
                "purpose": "technical-capacity-recovery" if reserve else "planned-eval",
            })
        adopted = list(keys[0])
        branch = "adoption"
    else:
        retry = keys[0]
        entries.append({
            "key": list(retry), "logical_job": logical_job(retry), "attempt": 2,
            "category": "reserve", "slot": 1, "purpose": "authorized-old-key-retry",
        })
        remaining = keys[1:]
        for index, key in enumerate(remaining):
            reserve = index == 94
            entries.append({
                "key": list(key), "logical_job": logical_job(key), "attempt": 1,
                "category": "reserve" if reserve else "normal",
                "slot": 2 if reserve else NORMAL_SLOTS[index],
                "purpose": "technical-capacity-recovery" if reserve else "planned-eval",
            })
        adopted = None
        branch = "retry_attempt2"
    normal = [x for x in entries if x["category"] == "normal"]
    reserve = [x for x in entries if x["category"] == "reserve"]
    plan = {
        "kind": "round46-r2-v7-quota-plan/1", "branch": branch,
        "adopted_key": adopted, "effective_key_count": len(keys),
        "new_claim_count": len(entries), "base_normal_used": BASE_NORMAL,
        "new_normal_slots": [x["slot"] for x in normal],
        "new_reserve_slots": [x["slot"] for x in reserve],
        "normal_used_after": BASE_NORMAL + len(normal),
        "reserve_used_after": len(reserve),
        "total_used_after": BASE_NORMAL + len(entries),
        "hard_cap": HARD_CAP, "technical_reserve_cap": RESERVE_CAP,
        "entries": entries,
    }
    if (plan["new_normal_slots"] != list(NORMAL_SLOTS)
            or plan["new_claim_count"] not in (95, 96)
            or plan["total_used_after"] > HARD_CAP
            or (adoption_passed and (len(normal), len(reserve), plan["total_used_after"])
                != (94, 1, 481))
            or (not adoption_passed and (len(normal), len(reserve), plan["total_used_after"])
                != (94, 2, 482))):
        raise V7Error("V7 quota plan does not match the preauthorized 95/96-cell map")
    if len({(tuple(x["key"]), x["attempt"]) for x in entries}) != len(entries):
        raise V7Error("V7 quota plan has duplicate logical attempts")
    return plan


def _recursive_find(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _recursive_find(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _recursive_find(child, key)
            if found is not None:
                return found
    return None


def validate_adoption_manifest(candidate: dict[str, Any]) -> dict[str, Any]:
    """Shared production/test adoption seam. Missing identity never defaults to zero."""
    key = candidate.get("key")
    if key != list(OLD_KEY):
        raise V7Error("adoption key is not the authorized slot386 cell")
    required = {
        "claim_sha256": OLD_CLAIM_SHA, "row_sha256": OLD_ROW_SHA,
        "lifecycle_sha256": OLD_ARTIFACT_SHAS["lifecycle"],
        "transcript_sha256": OLD_ARTIFACT_SHAS["transcript"],
        "tape_sha256": OLD_ARTIFACT_SHAS["tape"],
        "model_file_sha256": MODEL_FILE_SHA,
        "model_parameter_sha256": MODEL_PARAMETER_SHA,
        "old_executor_sha256": OLD_EXECUTOR_SHA,
        "old_source_head": OLD_HEAD,
        "old_input_head": OLD_INPUT_HEAD,
    }
    for field, expected in required.items():
        if candidate.get(field) != expected:
            raise V7Error(f"adoption identity mismatch or missing field: {field}")
    if candidate.get("old_claim_status") != "failed_technical":
        raise V7Error("old failed claim was rewritten or is not preserved")
    if candidate.get("old_claim_category") != "normal" or candidate.get("old_slot") != 386:
        raise V7Error("slot386 original quota identity differs")
    if candidate.get("row_identity") != {
        "model_arm": "NEW", "training_seed": 11, "eval_mode": "greedy",
        "scenario_seed": 1_301_000, "attempt": 1, "steps": 1000,
    }:
        raise V7Error("old evaluation row key or completion identity differs")
    for field in ("on_time_count", "expired_count", "placed_target_count",
                  "registry_target_count", "never_discovered_count"):
        if type(candidate.get(field)) is not int:
            raise V7Error(f"adoption outcome description is missing: {field}")
    if any(candidate[field] < 0 or candidate[field] > 6 for field in (
            "on_time_count", "expired_count", "placed_target_count",
            "registry_target_count", "never_discovered_count")):
        raise V7Error("adoption outcome description is outside the six-task block")
    if (candidate.get("episode_final_status") != "passed"
            or candidate.get("event_closure_status") != "passed"
            or candidate.get("cost_closure_status") != "passed"):
        raise V7Error("old episode window/event/cost closure is incomplete")
    if candidate.get("spent_bytes") != candidate.get("cost_stream_airtime_bytes"):
        raise V7Error("old row cost stream does not close")
    if candidate.get("model_load_mode") != "read_only" or candidate.get("optimizer_restored") is not False or candidate.get("rng_restored") is not False:
        raise V7Error("old model load was not read-only")
    if candidate.get("runtime_forward_status") != "passed":
        raise V7Error("real-model greedy forward audit is missing or failed")
    if candidate.get("configuration_status") != "passed" or candidate.get("source_gate_status") != "passed":
        raise V7Error("old scientific source/config identity is incomplete")
    if candidate.get("performance_used_for_adoption") is not False:
        raise V7Error("adoption decision may not depend on performance")
    return {
        "kind": "round46-r2-v7-adoption-decision/1", "status": "accepted",
        "key": list(OLD_KEY), "normal_slot": 386,
        "old_claim_remains_failed_technical": True,
        "new_claim_created": False, "new_environment_created": False,
        "performance_used_for_adoption": False,
        "artifact_hashes": {k: candidate[k] for k in (
            "claim_sha256", "row_sha256", "lifecycle_sha256",
            "transcript_sha256", "tape_sha256")},
        "model_file_sha256": candidate["model_file_sha256"],
        "model_parameter_sha256": candidate["model_parameter_sha256"],
        "real_model_greedy_forward": candidate["runtime_forward_status"],
        "on_time_count_description": candidate["on_time_count"],
        "expired_count_description": candidate["expired_count"],
        "placed_target_count": candidate["placed_target_count"],
        "registry_target_count": candidate["registry_target_count"],
        "never_discovered_count": candidate["never_discovered_count"],
        "registry_status_counts_description": candidate.get("registry_status_counts", {}),
        "window_task_counts_description": candidate.get("window_task_counts", {}),
        "spent_bytes": candidate["spent_bytes"],
        "old_scientific_source_head": candidate["old_source_head"],
    }


def verify_greedy_tape(controller: Any, tape: list[dict[str, Any]],
                       one_shot_module: Any) -> dict[str, Any]:
    """Recompute every free action through the actual frozen controller.forward path."""
    import numpy as np
    import torch

    send_index = int(one_shot_module.SEND)
    hold_index = int(one_shot_module.HOLD)
    free_seen: list[tuple[str, int]] = []
    checked: list[dict[str, Any]] = []
    for record in tape:
        free = record.get("free")
        if not isinstance(free, list):
            raise V7Error("tape free-choice field is malformed")
        if not free:
            continue
        targets = record.get("targets")
        features = record.get("features")
        if not isinstance(targets, list) or not isinstance(features, list) or len(targets) != len(features):
            raise V7Error("tape target/features alignment is malformed")
        matrix = torch.as_tensor(np.asarray(features, dtype=np.float32))
        with torch.no_grad():
            logits, _values = controller.forward(matrix)
            probabilities = torch.softmax(logits, dim=-1)
        for target in free:
            if target not in targets:
                raise V7Error("free tape target is not in candidate order")
            index = targets.index(target)
            probs = probabilities[index]
            expected = send_index if float(probs[send_index]) >= float(probs[hold_index]) else hold_index
            logged = record.get("actions", {}).get(target)
            effective = record.get("effective_actions", {}).get(target)
            stored_send = record.get("send_probabilities", {}).get(target)
            audit = record.get("candidate_audit", {}).get(target, {})
            if (logged != expected or effective != expected or stored_send is None
                    or abs(float(stored_send) - float(probs[send_index])) > 1e-6
                    or audit.get("relay_initial_executable") is not True
                    or audit.get("shadow_charge_applied") != (expected == send_index)):
                raise V7Error("old free action differs from real greedy forward or sequential gate")
            free_seen.append((str(target), int(expected)))
            checked.append({"step": int(record["step"]), "target_id": str(target),
                            "action": int(expected),
                            "send_probability": float(probs[send_index])})
    if len({target for target, _ in free_seen}) != len(free_seen):
        raise V7Error("old one-shot tape repeats a free target decision")
    counts = {"send": sum(action == send_index for _, action in free_seen),
              "skip": sum(action == hold_index for _, action in free_seen)}
    return {"status": "passed", "free_choice_count": len(free_seen), "choice_counts": counts,
            "checked_action_sha256": sha256_bytes(canonical_json_bytes(checked))}


def validate_episode_closure(row: dict[str, Any], lifecycle: list[dict[str, Any]],
                             transcript: list[dict[str, Any]],
                             tape: list[dict[str, Any]], *,
                             placed_target_ids: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Close a full execution window and its task census without requiring terminal tasks.

    ``placed_target_ids`` comes from the fixed scene/adapter identity. A placed target is
    either present in the registry census (a detection occurred) or still in the search
    side's remaining set. The four window metrics then partition only the discovered set.
    """
    if type(row.get("steps")) is not int or row.get("steps") != 1000:
        raise V7Error("episode lacks a full-horizon final row")
    final = [x for x in lifecycle if x.get("event_type") == "final"]
    run_final = [x for x in final if x.get("target_id") == "__run__"
                 or x.get("status") == "run_finished"]
    if len(run_final) != 1:
        raise V7Error("episode lifecycle lacks exactly one run final")
    run_record = run_final[0]
    metrics = run_record.get("detail", {}).get("metrics")
    if (run_record.get("target_id") != "__run__"
            or run_record.get("status") != "run_finished"
            or run_record.get("detail", {}).get("step") != 1000
            or not isinstance(metrics, dict) or metrics.get("steps") != 1000):
        raise V7Error("run final does not close the full 1000-step execution window")

    placed_ids = list(placed_target_ids)
    if (not placed_ids or any(not isinstance(value, str) or not value for value in placed_ids)
            or len(set(placed_ids)) != len(placed_ids)):
        raise V7Error("placed target identity set is empty, malformed, or duplicated")
    placed = set(placed_ids)
    task_final = [x for x in final if x is not run_record]
    registry_ids = [x.get("target_id") for x in task_final]
    if (any(not isinstance(value, str) or not value or value == "__run__"
            for value in registry_ids)
            or len(set(registry_ids)) != len(registry_ids)):
        raise V7Error("registry final census contains a missing or duplicate target identity")
    registry = set(registry_ids)
    if not registry.issubset(placed):
        raise V7Error("registry final census contains a target outside the placed set")
    allowed_statuses = {
        "detected", "verified", "assigned", "en_route", "in_progress", "partial",
        "replan_required", "completed", "expired", "lost", "failed",
    }
    if any(x.get("status") not in allowed_statuses for x in task_final):
        raise V7Error("registry final census contains an unknown task status")

    detections = [x for x in lifecycle if x.get("event_type") == "detection"]
    detected_ids = [x.get("target_id") for x in detections]
    if (any(not isinstance(value, str) or not value for value in detected_ids)
            or len(set(detected_ids)) != len(detected_ids)
            or set(detected_ids) != registry):
        raise V7Error("detection events and registry final census do not close uniquely")
    never_discovered_ids = placed - registry
    metrics_discovered = metrics.get("targets_discovered")
    metrics_remaining = metrics.get("remaining_search_targets")
    if (type(metrics_discovered) is not int or metrics_discovered != len(registry)
            or type(metrics_remaining) is not int or metrics_remaining != len(never_discovered_ids)
            or len(registry) + metrics_remaining != len(placed)
            or row.get("discovered") != len(registry)):
        raise V7Error("placed, registry, and never-discovered target sets do not close")
    status_counts: dict[str, int] = {}
    for record in task_final:
        status = str(record["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    metric_status_counts = metrics.get("status_counts")
    if metric_status_counts != status_counts:
        raise V7Error("run final registry status counts do not match the lifecycle census")
    nonterminal_statuses = allowed_statuses - {"completed", "expired", "lost", "failed"}
    registry_active = sum(x.get("status") in nonterminal_statuses for x in task_final)
    if metrics.get("targets_active") != registry_active:
        raise V7Error("run final active-registry count does not match the lifecycle census")

    # Recompute the frozen window partition from published lifecycle events. This mirrors
    # lifecycle_metrics.window_metrics semantics: complete first, then deadline expiry,
    # then no assignment, with the remaining discovered targets counted active.
    detection_data: dict[str, tuple[float, float]] = {}
    for event in detections:
        timestamp = float(event.get("timestamp") or 0.0)
        detail = event.get("detail") or {}
        offset = detail.get("deadline_offset")
        if type(offset) not in (int, float):
            raise V7Error("detection event lacks its deadline offset")
        detection_data[str(event["target_id"])] = (timestamp, timestamp + float(offset))
    completed_at: dict[str, float] = {}
    assigned: set[str] = set()
    for event in lifecycle:
        timestamp = float(event.get("timestamp") or 0.0)
        target_id = event.get("target_id")
        if timestamp > 1000 or target_id not in registry:
            continue
        if event.get("event_type") == "progress" and event.get("status") == "completed":
            completed_at.setdefault(str(target_id), timestamp)
        elif event.get("event_type") == "assignment":
            assigned.add(str(target_id))
    window_completed = set(completed_at)
    window_expired = {target for target, (_, deadline) in detection_data.items()
                      if target not in window_completed and deadline <= 1000}
    window_unassigned = {target for target in registry - window_completed - window_expired
                         if target not in assigned}
    window_active = registry - window_completed - window_expired - window_unassigned
    on_time_ids = {target for target, timestamp in completed_at.items()
                   if timestamp <= detection_data[target][1]}
    window_counts = {
        "discovered": len(registry), "completed": len(window_completed),
        "expired": len(window_expired), "active": len(window_active),
        "unassigned": len(window_unassigned), "on_time": len(on_time_ids),
    }
    if sum(window_counts[key] for key in ("completed", "expired", "active", "unassigned")) != len(registry):
        raise V7Error("discovered target window states are not mutually exclusive and exhaustive")
    if any(type(row.get(key)) is not int or row.get(key) != value
           for key, value in window_counts.items()):
        raise V7Error("row window state counters do not close to lifecycle events")
    events = row.get("task_reward_events")
    if not isinstance(events, list):
        raise V7Error("task reward event list is missing")
    event_ids = [x.get("event_id") for x in events]
    event_task_ids = [x.get("task_id") for x in events]
    if (any(not isinstance(x, str) or not x for x in event_ids)
            or len(set(event_ids)) != len(event_ids)
            or len(set(event_task_ids)) != len(event_task_ids)
            or any(x not in registry for x in event_task_ids)):
        raise V7Error("task reward events do not map uniquely to discovered registry tasks")
    credited = sum(x.get("credited") is True for x in events)
    on_time = sum(x.get("credited") is True and x.get("on_time") is True for x in events)
    credited_ids = {str(x.get("task_id")) for x in events if x.get("credited") is True}
    if (row.get("task_reward_total") != credited or credited != len(on_time_ids)
            or credited_ids != on_time_ids or on_time != len(on_time_ids)):
        raise V7Error("row task summary does not close to published reward events")
    cost_stream = row.get("cost_stream")
    if not isinstance(cost_stream, dict) or len(cost_stream) != 1000:
        raise V7Error("per-tick cost stream is incomplete")
    relays = row.get("relay_counters")
    if not isinstance(relays, dict):
        raise V7Error("relay cost counters are missing")
    counters = ("airtime_bytes", "transmission_attempts", "emissions", "packets_lost",
                "packets_delivered", "deliveries_applied", "budget_refused_sends",
                "capacity_refused_sends")
    tick_totals = {}
    for field in counters:
        if field not in relays or any(field not in tick for tick in cost_stream.values()):
            raise V7Error(f"cost counter is incomplete: {field}")
        tick_totals[field] = sum(int(tick[field]) for tick in cost_stream.values())
        if tick_totals[field] != int(relays[field]):
            raise V7Error(f"per-tick cost counter does not close to relay total: {field}")
    spent = row.get("spent_bytes")
    if (type(spent) is not int or spent != tick_totals["airtime_bytes"]
            or spent != row.get("airtime_tap_total") or spent != relays["airtime_bytes"]):
        raise V7Error("measured payload-attempt bytes do not close")
    transcript_cost = sum(int(x.get("cost_bytes", 0) or 0) for x in transcript)
    transcript_bytes = sum(int(x.get("bytes_total", 0) or 0) for x in transcript)
    transcript_attempts = sum(int(x.get("attempts", 0) or 0) for x in transcript)
    if (transcript_cost != spent or transcript_bytes != spent
            or transcript_attempts != relays["transmission_attempts"]):
        raise V7Error("transcript byte/attempt totals do not close to the measured relay stream")
    collector = row.get("collector_summary", {})
    one = collector.get("one_shot_counts", {})
    free_targets = [str(t) for item in tape for t in item.get("free", [])]
    tape_sends = sum(1 for item in tape for target in item.get("free", [])
                     if item.get("requested", {}).get(target) is True)
    tape_skips = len(free_targets) - tape_sends
    if (collector.get("decision_steps") != len(tape)
            or one.get("free_choices") != len(free_targets)
            or one.get("send_now_choices") != tape_sends
            or one.get("skip_episode_choices") != tape_skips
            or len(set(free_targets)) != len(free_targets)):
        raise V7Error("one-shot tape and collector totals do not close")
    return {
        "status": "passed", "final_step": 1000,
        "placed_target_count": len(placed), "registry_target_count": len(registry),
        "never_discovered_count": len(never_discovered_ids),
        "terminal_task_count_description": sum(
            1 for x in task_final if x.get("status") in ("completed", "expired", "lost", "failed")),
        "registry_status_counts": status_counts,
        "window_task_counts": window_counts,
        "task_reward_event_count": len(events), "credited_event_count": credited,
        "cost_closure": "passed", "spent_bytes_description": spent,
        "transmission_attempts": int(relays["transmission_attempts"]),
        "emissions_description": int(relays["emissions"]),
        "tape_decisions": len(tape), "free_choice_count_description": len(free_targets),
        "choice_counts_description": {"send": tape_sends, "skip": tape_skips},
        "tick_cost_totals": tick_totals,
        "cost_stream_sha256": sha256_bytes(canonical_json_bytes(cost_stream)),
    }


def _git_bytes(repo: Path, spec: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), "show", spec],
                          check=True, capture_output=True).stdout


def validate_old_source_gate(gate: dict[str, Any], executor_bytes: bytes,
                             frozen_bytes: bytes, input_bytes: bytes,
                             *, current_branch: str, source_head: str = OLD_HEAD,
                             preflight_head: str | None = None,
                             actual_input_head: str | None = None,
                             expected_executor_sha: str = OLD_EXECUTOR_SHA,
                             expected_input_head: str = OLD_INPUT_HEAD,
                             expected_historical_branch: str = "fix/comm-bc-initialization-readback",
                             strict: bool = True) -> dict[str, Any]:
    """Validate source-era records separately from the current V7 branch.

    ``source_head`` identifies the old execution claim; the prelaunch gate points to a
    distinct ``head_at_preflight`` whose Git objects supplied the runner and frozen inputs.
    The current on-disk V6 accounting freeze is never used as a substitute.
    """
    try:
        input_identity_valid = isinstance(json.loads(input_bytes), dict)
    except (UnicodeDecodeError, json.JSONDecodeError):
        input_identity_valid = False
    checks = {
        "old_gate_passed": gate.get("status") == "passed",
        "source_head": source_head == OLD_HEAD,
        "preflight_head": (preflight_head is None
                           or preflight_head == gate.get("head_at_preflight")),
        "historical_branch": gate.get("branch") == expected_historical_branch,
        "old_executor_bytes": sha256_bytes(executor_bytes) == expected_executor_sha,
        "gate_executor_identity": gate.get("runner_source_sha256") == expected_executor_sha,
        "original_frozen_config": sha256_bytes(frozen_bytes) == gate.get("frozen_config_sha256"),
        "original_input_identity": (input_identity_valid
                                     and sha256_bytes(input_bytes) == gate.get("input_identity_sha256")),
        "input_source_head": actual_input_head == expected_input_head,
        "current_v7_branch_separate": current_branch == BRANCH,
    }
    passed = all(checks.values())
    if strict and not passed:
        raise V7Error(f"historical slot386 source gate mismatch: {checks}")
    return {
        "status": "passed" if passed else "failed_source_authenticity",
        "old_source_head": source_head,
        "preflight_head": gate.get("head_at_preflight"),
        "historical_branch": gate["branch"], "current_v7_branch": current_branch,
        "runner_source_sha256": expected_executor_sha, "input_head": actual_input_head,
        "gate_sha256": sha256_bytes(canonical_json_bytes(gate)),
        "original_frozen_config_sha256": gate["frozen_config_sha256"],
        "original_input_identity_sha256": gate["input_identity_sha256"],
        "checks": checks,
    }


def verify_old_source_identity(repo: Path) -> dict[str, Any]:
    r46_rel = "runs/comm_bc_initialized_constrained_ppo_20261007_r2"
    gate_bytes = _git_bytes(repo, f"{OLD_HEAD}:{r46_rel}/ROUND46_R2_V6_PRELAUNCH_GATE.json")
    gate = json.loads(gate_bytes)
    preflight_head = gate.get("head_at_preflight")
    if not isinstance(preflight_head, str) or not preflight_head:
        raise V7Error("historical gate is missing its source preflight commit")
    executor = _git_bytes(repo, f"{preflight_head}:{r46_rel}/r46_execute_r2.py")
    frozen = _git_bytes(repo, f"{preflight_head}:{r46_rel}/ROUND46_R2_V6_FROZEN_CONFIG.json")
    input_bytes = _git_bytes(repo, f"{preflight_head}:{r46_rel}/ROUND46_R2_V6_INPUT_IDENTITY.json")
    current_branch = _git(repo, "branch", "--show-current")
    input_repo = Path(repo).parent / "zhenghe"
    try:
        actual_input_head = _git(input_repo, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        actual_input_head = None
    audit = validate_old_source_gate(gate, executor, frozen, input_bytes,
                                     current_branch=current_branch,
                                     source_head=OLD_HEAD,
                                     preflight_head=preflight_head,
                                     actual_input_head=actual_input_head,
                                     strict=False)
    audit["old_prelaunch_gate_raw_sha256"] = sha256_bytes(gate_bytes)
    audit["preflight_git_object_sha256"] = {
        "executor": sha256_bytes(executor),
        "frozen_config": sha256_bytes(frozen),
        "input_identity": sha256_bytes(input_bytes),
    }
    # The current postlaunch accounting freeze is inventoried separately; it is
    # never substituted for the original source-era gate/freeze objects above.
    audit["current_postlaunch_freeze_sha256"] = sha256_file(HERE / "ROUND46_R2_V6_FROZEN_CONFIG.json")
    audit["original_and_postlaunch_freezes_are_separate_records"] = True
    return audit


def audit_adoption_candidate(base: Any, runtime: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read-only source/model/forward audit of the one authorized slot386 row."""
    import numpy as np
    import torch

    repo = base.REPO
    root = HERE
    source_audit = verify_old_source_identity(repo)
    checkpoint = V6_ROOT / "checkpoints/v2_train_s11_ep128.pt"
    if sha256_file(checkpoint) != MODEL_FILE_SHA:
        raise V7Error("fixed s11 ep128 checkpoint bytes changed before adoption audit")
    protected_before = {str(HERE / Path(rel).name): sha256_file(repo / rel)
                        for rel in PROTECTED}
    old_claim = root / "attempts/runner_v5_ep0_receipt_conflict/stale_claim/v6/launch_claims/model_eval_new_s11_greedy_scene1301000/attempt_0001.json"
    old_row = root / "v6/rows/v2_r44_eval_new_s11_greedy_s1301000_a1.json"
    companion_paths = [root / f"v6/logs/v2_r44_eval_new_s11_greedy_s1301000_a1_{s}.jsonl"
                       for s in ("rp", "ct", "tape")]
    checkpoint_before = sha256_file(checkpoint)
    journal_path = base.CHECKPOINT_JOURNAL_PATH
    journal_before = sha256_file(journal_path)
    row_before = sha256_file(old_row)
    claim_before = sha256_file(old_claim)
    artifacts_before = {str(p): sha256_file(p) for p in companion_paths}
    rng_python_before = random.getstate()
    rng_numpy_before = np.random.get_state()
    rng_torch_before = torch.get_rng_state().clone()
    run = runtime["r42_run"]
    optimizer_calls_before = int(getattr(run, "_OPTIMIZER_STEP_CALLS", 0))
    ppo_calls_before = int(getattr(run, "_PPO_UPDATE_CALLS", 0))
    checkpoint_calls_before = int(runtime["r42_versions"].CHECKPOINT_WRITE_CALLS)
    controller, model_identity = run.load_eval_model("NEW", 11)
    one_shot = runtime.get("r42_one_shot") or sys.modules.get("r42_one_shot")
    if one_shot is None:
        raise V7Error("actual one-shot collector module is unavailable")
    tape = read_jsonl(companion_paths[2])
    forward = verify_greedy_tape(controller, tape, one_shot)
    identity_fields = ("training_seed", "episodes_done", "updates", "load_mode",
                       "optimizer_restored", "rng_restored", "file_sha256")
    model_gate = {k: model_identity.get(k) for k in identity_fields}
    model_gate["parameter_identity"] = copy.deepcopy(model_identity.get("parameter_identity"))
    if (model_identity.get("training_seed") != 11
            or model_identity.get("episodes_done") != 128
            or model_identity.get("updates") != 32
            or model_identity.get("load_mode") != "read_only"
            or model_identity.get("optimizer_restored") is not False
            or model_identity.get("rng_restored") is not False
            or model_identity.get("file_sha256") != MODEL_FILE_SHA
            or model_identity.get("parameter_identity", {}).get("parameter_sha256")
            != MODEL_PARAMETER_SHA):
        raise V7Error("actual read-only s11 ep128 model identity does not match V6")
    del controller
    unchanged = {
        "old_claim": sha256_file(old_claim) == claim_before,
        "old_row": sha256_file(old_row) == row_before,
        "old_companions": all(sha256_file(Path(p)) == digest
                               for p, digest in artifacts_before.items()),
        "checkpoint": sha256_file(checkpoint) == checkpoint_before,
        "checkpoint_journal": sha256_file(journal_path) == journal_before,
        "python_rng": random.getstate() == rng_python_before,
        "numpy_rng": (
            np.random.get_state()[0] == rng_numpy_before[0]
            and np.array_equal(np.random.get_state()[1], rng_numpy_before[1])
            and np.random.get_state()[2:] == rng_numpy_before[2:]
        ),
        "torch_rng": torch.equal(torch.get_rng_state(), rng_torch_before),
        "optimizer_counter": int(getattr(run, "_OPTIMIZER_STEP_CALLS", 0)) == optimizer_calls_before,
        "ppo_counter": int(getattr(run, "_PPO_UPDATE_CALLS", 0)) == ppo_calls_before,
        "checkpoint_counter": int(runtime["r42_versions"].CHECKPOINT_WRITE_CALLS)
                             == checkpoint_calls_before,
        "protected_files": all(sha256_file(repo / rel) == digest
                               for rel, digest in PROTECTED.items()),
    }
    if not all(unchanged.values()):
        raise V7Error(f"read-only adoption audit changed protected state: {unchanged}")
    decision, detail = verify_old_candidate(root, source_audit=source_audit,
                                            model_identity=model_gate,
                                            forward_audit=forward)
    detail["read_only_nonmutation_checks"] = unchanged
    detail["historical_source_gate"] = source_audit
    detail["current_v7_branch"] = _git(repo, "branch", "--show-current")
    detail["current_postlaunch_freeze_sha256"] = source_audit["current_postlaunch_freeze_sha256"]
    detail["action_rng_seed"] = 7566968475784137584
    detail["old_slot_claim_category_and_charge_preserved"] = True
    return decision, detail


def verify_old_candidate(root: Path, *, source_audit: dict[str, Any],
                         model_identity: dict[str, Any],
                         forward_audit: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read-only content gate for the sole V7 adoption candidate."""
    root = Path(root)
    claim_path = root / "attempts/runner_v5_ep0_receipt_conflict/stale_claim/v6/launch_claims/model_eval_new_s11_greedy_scene1301000/attempt_0001.json"
    row_path = root / "v6/rows/v2_r44_eval_new_s11_greedy_s1301000_a1.json"
    artifact_paths = {
        "lifecycle": root / "v6/logs/v2_r44_eval_new_s11_greedy_s1301000_a1_rp.jsonl",
        "transcript": root / "v6/logs/v2_r44_eval_new_s11_greedy_s1301000_a1_ct.jsonl",
        "tape": root / "v6/logs/v2_r44_eval_new_s11_greedy_s1301000_a1_tape.jsonl",
    }
    actual = {"claim_sha256": sha256_file(claim_path), "row_sha256": sha256_file(row_path)}
    actual.update({f"{name}_sha256": sha256_file(path)
                   for name, path in artifact_paths.items()})
    if actual["claim_sha256"] != OLD_CLAIM_SHA or actual["row_sha256"] != OLD_ROW_SHA:
        raise V7Error("slot386 source claim/row bytes do not match the frozen candidate")
    for name, path in artifact_paths.items():
        if actual[f"{name}_sha256"] != OLD_ARTIFACT_SHAS[name]:
            raise V7Error(f"slot386 {name} bytes do not match the frozen candidate")
    claim = read_json(claim_path)
    row = read_json(row_path)
    slot_path = root / "v6/launch_slots/normal/0386.json"
    slot = read_json(slot_path)
    if (slot.get("category"), slot.get("slot"), slot.get("attempt"),
            slot.get("logical_job")) != ("normal", 386, 1, logical_job(OLD_KEY)):
        raise V7Error("slot386 immutable slot record does not match old claim")
    art = row.get("artefacts", {})
    for name, path in {"row": row_path, **artifact_paths}.items():
        if Path(art.get(name, "")).resolve() != path.resolve():
            raise V7Error(f"slot386 row artifact path mismatch: {name}")
    if (claim.get("logical_job") != logical_job(OLD_KEY)
            or claim.get("status") != "failed_technical"
            or claim.get("category") != "normal" or claim.get("attempt") != 1
            or claim.get("launch_slot") != 386 or claim.get("prelaunch_head") != OLD_HEAD
            or claim.get("environment_constructor_calls") != 1
            or claim.get("run_return_calls") != 1 or claim.get("completed_steps") != 1000
            or claim.get("run_failure_calls") != 0 or claim.get("optimizer_step_calls") != 0
            or claim.get("checkpoint_write_calls") != 0):
        raise V7Error("slot386 failed claim does not match its documented completed-run booking failure")
    collector = row.get("collector_summary", {})
    one = collector.get("one_shot_counts", {})
    task_events = row.get("task_reward_events")
    if not isinstance(task_events, list):
        raise V7Error("slot386 task completion events are missing")
    credited = [x for x in task_events if x.get("credited") is True and x.get("on_time") is True]
    cost_stream = row.get("cost_stream")
    if not isinstance(cost_stream, dict) or len(cost_stream) != 1000:
        raise V7Error("slot386 1000-tick cost stream is incomplete")
    tick_airtime = sum(int(x.get("airtime_bytes", -1)) for x in cost_stream.values())
    transcript = read_jsonl(artifact_paths["transcript"])
    lifecycle = read_jsonl(artifact_paths["lifecycle"])
    tape = read_jsonl(artifact_paths["tape"])
    cost_bytes = sum(int(x.get("cost_bytes", 0) or 0) for x in transcript)
    bytes_total = sum(int(x.get("bytes_total", 0) or 0) for x in transcript)
    attempts = sum(int(x.get("attempts", 0) or 0) for x in transcript)
    if (row.get("model_arm"), row.get("training_seed"), row.get("eval_mode"),
            row.get("scenario_seed"), row.get("attempt"), row.get("steps")) != (
            "NEW", 11, "greedy", 1_301_000, 1, 1000):
        raise V7Error("slot386 row is not the exact fixed evaluation cell")
    # The frozen world uses six placed goals. The adapter's reset identity is
    # piloc-e1-target-{index}; derive the complete placed set from that source contract,
    # so undiscovered goals remain in the denominator without requiring task finals.
    placed_target_ids = tuple(f"piloc-e1-target-{index}" for index in range(6))
    closure = validate_episode_closure(
        row, lifecycle, transcript, tape, placed_target_ids=placed_target_ids)
    if (row.get("optimizer_steps") != 0
            or row.get("transport_parameters", {}).get("ack_enabled") is not False
            or row.get("ack_ledger", {}).get("ack_attempts") != 0):
        raise V7Error("slot386 communication/action identity differs from one-shot evaluation")
    model = row.get("checkpoint_load_identity", {})
    if (model.get("file_sha256") != MODEL_FILE_SHA
            or model.get("parameter_identity", {}).get("parameter_sha256") != MODEL_PARAMETER_SHA
            or model.get("episodes_done") != 128 or model.get("updates") != 32
            or model.get("load_mode") != "read_only"
            or model.get("optimizer_restored") is not False
            or model.get("rng_restored") is not False):
        raise V7Error("slot386 recorded checkpoint identity is not the fixed read-only ep128")
    sig = _recursive_find(lifecycle, "run_signature")
    if (not isinstance(sig, dict) or sig.get("code_commit") != OLD_INPUT_HEAD
            or sig.get("config_hash") != row.get("config_hash")):
        raise V7Error("slot386 actual run signature lacks the frozen input/config identity")
    transport = row.get("transport_parameters", {})
    config_ok = (
        row.get("config_hash") == "b4adedda698c235d"
        and row.get("reward_timing_version") == "published-completion-event-at-relay-tick/2"
        and row.get("comm_action_semantics") == "one-shot-send-skip/1"
        and row.get("comm_action_eligibility") == "first-report-only/1"
        and row.get("b0_bytes") == 10000
        and transport.get("policy") == "learned"
        and transport.get("policy_parameters", {}).get("horizon") == 1000.0
        and transport.get("policy_parameters", {}).get("feature_layout") == "comm-send-state/1"
        and transport.get("timing_mode") == "policy_owned_timing"
        and transport.get("ack_enabled") is False
        and transport.get("delay_ticks") == 0
        and transport.get("delivery_probability") == 0.7
        and transport.get("budget", {}).get("bytes_per_episode") == 29300
        and sig.get("checkpoints", {}).get("piloc") ==
            "d57aed25b3d46dffb2bd269ea87140953828ba7f51a960bb01995ec054dfca33"
        and sig.get("checkpoints", {}).get("ctde") ==
            "c7c256da0fc20451c1104996ce494148f567e4f83acf265d9baae8f0f053500d"
    )
    if not config_ok:
        raise V7Error("slot386 row does not close against the frozen world/action configuration")
    candidate = {
        **actual, "key": list(OLD_KEY), "old_claim_status": claim.get("status"),
        "old_claim_category": claim.get("category"), "old_slot": slot.get("slot"),
        "row_identity": {k: row.get(k) for k in (
            "model_arm", "training_seed", "eval_mode", "scenario_seed", "attempt", "steps")},
        "on_time_count": int(row.get("on_time")), "expired_count": int(row.get("expired")),
        "placed_target_count": closure["placed_target_count"],
        "registry_target_count": closure["registry_target_count"],
        "never_discovered_count": closure["never_discovered_count"],
        "registry_status_counts": closure["registry_status_counts"],
        "window_task_counts": closure["window_task_counts"],
        "episode_final_status": closure["status"],
        "event_closure_status": closure["status"],
        "cost_closure_status": closure["cost_closure"],
        "spent_bytes": row.get("spent_bytes"), "cost_stream_airtime_bytes": tick_airtime,
        "model_file_sha256": model_identity.get("file_sha256"),
        "model_parameter_sha256": model_identity.get("parameter_identity", {}).get("parameter_sha256"),
        "model_load_mode": model_identity.get("load_mode"),
        "optimizer_restored": model_identity.get("optimizer_restored"),
        "rng_restored": model_identity.get("rng_restored"),
        "old_executor_sha256": source_audit.get("runner_source_sha256"),
        "old_source_head": source_audit.get("old_source_head"),
        "old_input_head": source_audit.get("input_head"),
        "runtime_forward_status": forward_audit.get("status"),
        "configuration_status": "passed",
        "source_gate_status": source_audit.get("status"),
        "performance_used_for_adoption": False,
    }
    if source_audit.get("status") == "passed":
        decision = validate_adoption_manifest(candidate)
    else:
        # The content, scientific configuration, model, real forward, event closure, and
        # cost closure all passed above. Only the historical source freeze/input binding is
        # unverified, so the authorized path is a fresh attempt-2 reserve execution; never
        # adopt this old row and never use its performance to choose the path.
        decision = {
            "kind": "round46-r2-v7-adoption-decision/1",
            "status": "rejected_source_authenticity",
            "key": list(OLD_KEY), "normal_slot": 386,
            "old_claim_remains_failed_technical": True,
            "new_claim_created": False, "new_environment_created": False,
            "performance_used_for_adoption": False,
            "fallback": {"category": "reserve", "slot": 1, "attempt": 2,
                         "purpose": "authorized-old-key-retry"},
            "source_gate_status": source_audit.get("status"),
            "source_gate_checks": source_audit.get("checks", {}),
            "content_identity_status": "passed",
            "model_identity_status": "passed",
            "real_model_greedy_forward": forward_audit.get("status"),
            "event_and_cost_closure_status": "passed",
            "old_claim_sha256": actual["claim_sha256"],
            "old_row_sha256": actual["row_sha256"],
        }
    detail = {
        **decision, "claim_path": str(claim_path), "row_path": str(row_path),
        "artifact_paths": {k: str(v) for k, v in artifact_paths.items()},
        "lifecycle_records": len(lifecycle), "transcript_records": len(transcript),
        "tape_records": len(tape), "transmission_attempts": attempts,
        "completion_closure": closure,
        "cost_stream_sha256": closure["cost_stream_sha256"],
        "task_ids": sorted(str(x["task_id"]) for x in credited),
        "greedy_forward": forward_audit,
        "old_claim_sha256_before_after": [actual["claim_sha256"], actual["claim_sha256"]],
        "original_row_and_companions_unmodified": True,
    }
    return decision, detail


def adopt_candidate(root: Path, *, decision: dict[str, Any], detail: dict[str, Any]) -> str:
    receipt = {"kind": "round46-r2-v7-adoption-receipt/1", **decision, "evidence": detail}
    return write_new_json(Path(root) / "receipts/ROUND46_R2_V7_ADOPTION_RECEIPT.json", receipt)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise V7Error(f"cannot load frozen runtime source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def function_sha256(source: bytes, function_name: str) -> str:
    import ast
    text = source.decode("utf-8").replace("\r\n", "\n")
    tree = ast.parse(text)
    node = next((x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and x.name == function_name), None)
    if node is None:
        raise V7Error(f"frozen function missing: {function_name}")
    segment = ast.get_source_segment(text, node)
    return sha256_bytes(segment.encode("utf-8"))


def _verify_private_runtime_aliases(base: Any, _identity: dict[str, Any]) -> None:
    pairs = [
        (base.ROUND45_DIR / "ROUND45_CONFIGURATION_DIFF.json",
         base.R46_DIR / "ROUND45_CONFIGURATION_DIFF.json"),
        (base.ROUND45_DIR / "ROUND45_R2_BC_TRAINING_SUMMARY.json",
         base.R46_DIR / "ROUND45_R2_BC_TRAINING_SUMMARY.json"),
    ]
    for name in ("r40_train_resume_adapter.py", "r42_one_shot.py",
                 "r42_versions.py", "r43_call_counter_hooks.py"):
        pairs.append((base.ROUND45_DIR / name, base.R46_DIR / name))
    for source, alias in pairs:
        if not source.is_file() or not alias.is_file() or sha256_file(source) != sha256_file(alias):
            raise V7Error(f"read-only runtime alias missing or changed: {alias.name}")


def _prepare_v7_dirs() -> None:
    for path in (V7_ROOT, V7_RUNTIME, V7_ROWS, V7_LOGS, V7_ROOT / "rejected",
                 V7_CLAIMS, V7_SLOTS / "normal", V7_SLOTS / "reserve",
                 V7_RECEIPTS, V7_LEDGER_EVENTS):
        path.mkdir(parents=True, exist_ok=True)


def _read_only_checkpoint_route(base: Any, runtime: dict[str, Any]) -> dict[str, Any]:
    audit = base._checkpoint_journal_route_audit(create=False)
    wrapper = runtime["trainer"].save_checkpoint
    cells = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__ or (), strict=True))
    journal_cell = cells.get("checkpoint_journal")
    if journal_cell is None:
        raise V7Error("checkpoint save hook has no auditable journal closure")
    captured = Path(journal_cell.cell_contents).resolve()
    allowed = {base.LEGACY_CHECKPOINT_JOURNAL_PATH.resolve(),
               base.CHECKPOINT_JOURNAL_PATH.resolve()}
    if captured not in allowed:
        raise V7Error("checkpoint journal is outside the previously audited V6 route")
    lines = sum(1 for line in base.CHECKPOINT_JOURNAL_PATH.open("r", encoding="utf-8") if line.strip())
    return {
        "r42_run_output_dir": str(base.V6_DIR.resolve()),
        "journal_closure_before": str(captured),
        "journal_closure_after": str(captured),
        "journal_entry_count": lines,
        "journal_sha256": sha256_file(captured),
        "legacy_journal_sha256_preserved": audit["legacy_sha256"],
        "route_audit_sha256": sha256_file(base.CHECKPOINT_JOURNAL_ROUTE_AUDIT_PATH),
        "read_only_eval_no_rebind": True,
    }


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _verify_v7_freeze(base: Any, _runtime: dict[str, Any], *,
                      require_gate: bool = True) -> None:
    freeze_path = V7_ROOT / "ROUND46_R2_V7_FROZEN_CONFIG.json"
    gate_path = V7_ROOT / "ROUND46_R2_V7_PRELAUNCH_GATE.json"
    if not freeze_path.is_file() or not gate_path.is_file():
        raise V7Error("V7 prelaunch freeze/gate is missing")
    freeze, gate = read_json(freeze_path), read_json(gate_path)
    if (freeze.get("task_id") != TASK_ID or gate.get("task_id") != TASK_ID
            or freeze.get("status") != "frozen_before_environment"
            or gate.get("status") != "passed"):
        raise V7Error("V7 freeze/gate identity is not active")
    if require_gate and gate.get("zero_environment_regression") != "passed":
        raise V7Error("V7 zero-environment regression is not registered as passed")
    if _git(base.REPO, "branch", "--show-current") != BRANCH:
        raise V7Error("V7 runtime is on the wrong branch")
    if (sha256_file(freeze_path) != gate.get("frozen_config_sha256")
            or sha256_file(V7_ROOT / "ROUND46_R2_V7_QUOTA_PLAN.json")
            != gate.get("quota_plan_sha256")
            or sha256_file(Path(__file__)) != gate.get("adapter_source_sha256")):
        raise V7Error("V7 frozen code or quota map changed after prelaunch")
    source_commit = freeze.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise V7Error("V7 source commit identity is missing")
    ancestor = subprocess.run(["git", "-C", str(base.REPO), "merge-base", "--is-ancestor",
                               source_commit, "HEAD"], capture_output=True)
    if ancestor.returncode != 0:
        raise V7Error("V7 source commit is not an ancestor of the current execution head")
    source_sha = gate.get("source_hashes", {}).get(Path(__file__).name)
    if source_sha != sha256_file(Path(__file__)):
        raise V7Error("V7 adapter source hash differs from the prelaunch gate")
    status = _git(base.REPO, "status", "--porcelain")
    allowed = {
        " M " + path for path in PROTECTED
    }
    lines = set(status.splitlines())
    if lines != allowed:
        raise V7Error(f"worktree status changed outside the two preserved ROUND46 files: {sorted(lines)}")
    for relative, digest in PROTECTED.items():
        if sha256_file(base.REPO / relative) != digest:
            raise V7Error(f"protected prior-session file changed: {relative}")
    for path, digest in freeze.get("checkpoint_files", {}).items():
        if sha256_file(Path(path)) != digest:
            raise V7Error("a frozen ep128 checkpoint changed")
    quota = read_json(V7_ROOT / "ROUND46_R2_V7_QUOTA_PLAN.json")
    if (quota.get("branch") != freeze.get("quota_branch")
            or quota.get("new_claim_count") != freeze.get("new_claim_count")
            or sha256_bytes(canonical_json_bytes(quota)) != gate.get("quota_plan_canonical_sha256")):
        raise V7Error("V7 claim map differs from its immutable prelaunch identity")
    route = _runtime_journal_identity(base)
    if (route.get("read_only_eval_no_rebind") is not True
            or route.get("journal_sha256") != gate.get("checkpoint_journal_sha256")):
        raise V7Error("checkpoint journal changed or V7 evaluation attempted to rebind it")


def _runtime_journal_identity(base: Any) -> dict[str, Any]:
    path = base.CHECKPOINT_JOURNAL_PATH
    return {"read_only_eval_no_rebind": True,
            "journal_sha256": sha256_file(path)}


def configure_eval_runtime(base: Any | None = None) -> tuple[Any, dict[str, Any]]:
    """Load actual frozen R45/R44/R42 evaluation code with V7-only output roots."""
    _prepare_v7_dirs()
    if base is None:
        base = _load_module(HERE / "r46_execute_r2.py", "round46_r2_v7_base")

    def v7_seed_namespace() -> dict[str, Any]:
        _prepare_v7_dirs()
        for path in (base.V6_DIR, base.V6_RUNTIME_DIR, base.V6_CHECKPOINTS_DIR,
                     base.V6_LOGS_DIR, base.V6_ROWS_DIR, base.V6_REJECTED_DIR):
            if not path.is_dir():
                raise V7Error(f"read-only V6 input namespace is missing: {path.name}")
        return {"v7_root": str(V7_ROOT), "v6_write": False}

    base._seed_v6_output_namespace = v7_seed_namespace
    base._copy_runtime_aliases = lambda identity: _verify_private_runtime_aliases(base, identity)
    base._rebind_checkpoint_journal = lambda runtime: _read_only_checkpoint_route(base, runtime)
    base._verify_prelaunch_identity = lambda runtime, **_kw: _verify_v7_freeze(base, runtime)
    runtime_module = base._configure_runtime_module()
    runtime = base._RUNTIME
    if runtime is None:
        raise V7Error("frozen runtime did not initialize")

    common, r40 = runtime["common"], runtime["r40_common"]
    common.OUTPUT_DIR, common.LOGS, common.ROWS = V7_ROOT, V7_LOGS, V7_ROWS
    common.REJECTED = V7_ROOT / "rejected"
    common.CHECKPOINTS = base.V6_CHECKPOINTS_DIR
    common.NORMAL_LAUNCHES, common.RESERVE_LAUNCHES, common.LAUNCH_CAP = 480, 24, HARD_CAP
    common.DEVICE = runtime["common"].torch.device("cpu") if hasattr(common, "torch") else __import__("torch").device("cpu")
    r40.ROUND_DIR, r40.NORMAL_CAP, r40.RESERVE_CAP, r40.HARD_CAP = V7_ROOT, 480, 24, HARD_CAP
    r40.plan = lambda: {"normal_total": 480, "technical_reserve_max": 24,
                        "hard_cap": HARD_CAP, "task_id": TASK_ID}
    run = runtime["r42_run"]
    run.RUN_DIR = V7_ROOT
    for key in ("r44_runtime", "r44_prepare", "r42_analysis"):
        mod = runtime.get(key)
        if mod is not None and hasattr(mod, "RUN_DIR"):
            mod.RUN_DIR = V7_ROOT
    r45 = runtime_module
    r45.RUN_DIR = V7_ROOT
    v7_runner_version = CODE_VERSION + "+" + str(r45.RUNNER_VERSION)
    r45.RUNNER_VERSION = v7_runner_version
    for name in ("runner", "r42_run", "r44_runtime", "r44_prepare"):
        mod = runtime.get(name)
        if mod is not None and hasattr(mod, "RUNNER_VERSION"):
            mod.RUNNER_VERSION = v7_runner_version
    r45.R3_EXECUTION_LEDGER_PATH = V7_LEDGER
    r45.R3_REPORT_PATH = V7_ROOT / "ROUND46_R2_V7_REPORT.md"
    r45.R3_FINAL_RESULTS_PATH = V7_ROOT / "ROUND46_R2_V7_FINAL_RESULTS.json"
    r45._task_state = lambda stage, **kw: v7_task_state(stage, **kw)
    base.CURRENT_TASK_PATH = V7_ROOT / "ROUND46_R2_V7_TASK_STATE.json"
    base._task_state = lambda stage, **kw: v7_task_state(stage, **kw)
    runtime["common"].OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (V7_LOGS, V7_ROWS, V7_ROOT / "rejected"):
        path.mkdir(parents=True, exist_ok=True)
    save = runtime["trainer"].save_checkpoint
    def forbidden_checkpoint_save(*_args: Any, **_kwargs: Any) -> None:
        raise V7Error("checkpoint save is forbidden in V7 eval-only runtime")
    runtime["trainer"].save_checkpoint = forbidden_checkpoint_save
    runtime["v7_base"] = base
    runtime["v7_runner"] = r45
    runtime["v7_original_checkpoint_save_hook"] = save
    runtime["v7_checkpoint_save_calls"] = 0
    return base, runtime


def v7_task_state(stage: str, **updates: Any) -> None:
    path = V7_ROOT / "ROUND46_R2_V7_TASK_STATE.json"
    state = read_json(path) if path.is_file() else {}
    state.update({"kind": "round46-r2-v7-task-state/1", "task_id": TASK_ID,
                  "stage": stage, "branch": BRANCH,
                  "head": _git(HERE.parents[1], "rev-parse", "HEAD"),
                  "updated_at_local": time.strftime("%Y-%m-%d %H:%M:%S")})
    state.update(updates)
    write_json_atomic(path, state)


def install_quota_adapter(runtime: dict[str, Any], plan: dict[str, Any], *,
                          output_root: Path = V7_ROOT) -> dict[str, Any]:
    """Route real R40 atomic claims through the pre-frozen V7 slot map."""
    r40 = runtime["r40_common"]
    expected = {(entry["logical_job"], int(entry["attempt"])): entry
                for entry in plan["entries"]}
    if len(expected) != len(plan["entries"]):
        raise V7Error("quota adapter received duplicate attempts")
    original_claim = r40.claim_launch
    original_slot = r40._reserve_slot
    original_next = r40.next_attempt
    records: list[dict[str, Any]] = []

    def reserve_slot(category: str, *, logical_job: str, attempt: int) -> int:
        item = expected.get((logical_job, int(attempt)))
        if item is None or item["category"] != category:
            raise V7Error("real R40 slot seam reached an unfrozen V7 claim")
        path = Path(output_root) / "launch_slots" / category / f"{int(item['slot']):04d}.json"
        payload = {"task_id": TASK_ID, "category": category, "slot": int(item["slot"]),
                   "logical_job": logical_job, "attempt": int(attempt),
                   "purpose": item["purpose"], "claimed_at_unix": time.time()}
        raw = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise V7Error(f"V7 fixed launch slot already consumed: {category}/{item['slot']}") from error
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        return int(item["slot"])

    def claim_launch(logical_job: str, *, category: str, attempt: int,
                     metadata: dict[str, Any] | None = None) -> Path:
        item = expected.get((logical_job, int(attempt)))
        if item is None:
            raise V7Error(f"unplanned real R40 claim rejected: {logical_job}#{attempt}")
        if category != item["category"]:
            if not (item["purpose"] == "technical-capacity-recovery"
                    and int(attempt) == 1 and category == "normal"):
                raise V7Error("R40 claim category differs from the frozen V7 quota map")
        user_meta = dict(metadata or {})
        user_meta.update({"v7_task_id": TASK_ID, "v7_purpose": item["purpose"],
                          "v7_slot": int(item["slot"]),
                          "original_base_normal_used": BASE_NORMAL})
        path = original_claim(logical_job, category=item["category"],
                              attempt=int(attempt), metadata=user_meta)
        claim = read_json(path)
        if (claim.get("category") != item["category"]
                or int(claim.get("launch_slot", -1)) != int(item["slot"])):
            raise V7Error("R40 atomic claim did not bind the planned V7 slot")
        r40._update_claim(path, round46_r2_v7={
            "task_id": TASK_ID, "code_version": CODE_VERSION,
            "category": item["category"], "slot": int(item["slot"]),
            "purpose": item["purpose"], "quota_plan_sha256": sha256_bytes(canonical_json_bytes(plan)),
        })
        records.append({"logical_job": logical_job, "attempt": int(attempt),
                        "category": item["category"], "slot": int(item["slot"])})
        return path

    def next_attempt(logical_job: str) -> int:
        retry = expected.get((logical_job, 2))
        if retry is not None and retry.get("purpose") == "authorized-old-key-retry":
            if original_next(logical_job) != 1:
                raise V7Error("V7 retry key unexpectedly already has a new attempt")
            return 2
        return original_next(logical_job)

    r40._reserve_slot = reserve_slot
    r40.claim_launch = claim_launch
    r40.next_attempt = next_attempt
    return {"expected": expected, "claim_records": records,
            "original_claim_launch": original_claim, "original_reserve_slot": original_slot,
            "original_next_attempt": original_next}


def _claim_records(root: Path = V7_ROOT) -> list[tuple[Path, dict[str, Any]]]:
    result = []
    for path in sorted((Path(root) / "launch_claims").glob("*/*.json")):
        result.append((path, read_json(path)))
    return result


def build_cumulative_ledger(plan: dict[str, Any], *, strict: bool = True,
                            root: Path = V7_ROOT) -> dict[str, Any]:
    root = Path(root)
    claims = _claim_records(root)
    planned = {(x["logical_job"], int(x["attempt"])): x for x in plan["entries"]}
    seen: set[tuple[str, int]] = set()
    delta = {"environment_constructor_calls": 0, "joint_run_constructed_calls": 0,
             "run_invocation_calls": 0, "joint_run_method_calls": 0,
             "run_return_calls": 0, "run_failure_calls": 0,
             "completed_steps": 0, "technical_failed_claims": 0}
    event_ids: list[str] = []
    for path, claim in claims:
        ident = (str(claim.get("logical_job")), int(claim.get("attempt", -1)))
        if ident in seen or ident not in planned:
            raise V7Error(f"claim ledger has duplicate/unplanned V7 attempt: {ident}")
        seen.add(ident)
        item = planned[ident]
        if (claim.get("category") != item["category"]
                or int(claim.get("launch_slot", -1)) != int(item["slot"])):
            raise V7Error(f"V7 claim category/slot differs from frozen map: {ident}")
        for field in delta:
            delta[field] += int(claim.get(field, 0) or 0)
        if claim.get("status") == "failed_technical":
            delta["technical_failed_claims"] += 1
        if claim.get("status") in ("complete", "failed_technical"):
            event = {
                "kind": "round46-r2-v7-claim-event/1", "task_id": TASK_ID,
                "logical_job": ident[0], "attempt": ident[1],
                "claim_path": str(path), "claim_sha256": sha256_file(path),
                "status": claim.get("status"),
                "environment_constructor_calls": int(claim.get("environment_constructor_calls", 0)),
                "run_return_calls": int(claim.get("run_return_calls", 0)),
                "completed_steps": int(claim.get("completed_steps", 0)),
                "optimizer_step_calls": int(claim.get("optimizer_step_calls", 0)),
                "checkpoint_write_calls": int(claim.get("checkpoint_write_calls", 0)),
            }
            event_path = V7_LEDGER_EVENTS / f"{path.parent.name}_a{ident[1]:04d}.json"
            event_ids.append(write_new_json(event_path, event))
    if strict and not seen.issubset(planned):
        raise V7Error("V7 claim set exceeds the frozen plan")
    slot_counts = {category: sum(1 for p in (root / "launch_slots" / category).glob("*.json"))
                   for category in ("normal", "reserve")}
    expected_slots = {"normal": len([x for x in plan["entries"] if x["category"] == "normal"]),
                      "reserve": len([x for x in plan["entries"] if x["category"] == "reserve"])}
    if slot_counts != expected_slots:
        raise V7Error(f"V7 slot delta differs from its frozen map: {slot_counts}")
    normal_used = BASE_NORMAL + slot_counts["normal"]
    reserve_used = slot_counts["reserve"]
    cumulative_counts = {
        "environment_constructor_calls": BASE_NORMAL + delta["environment_constructor_calls"],
        "joint_run_constructed_calls": BASE_NORMAL + delta["joint_run_constructed_calls"],
        "run_invocation_calls": BASE_NORMAL + delta["run_invocation_calls"],
        "joint_run_method_calls": BASE_NORMAL + delta["joint_run_method_calls"],
        "run_return_calls": BASE_NORMAL + delta["run_return_calls"],
        "run_failure_calls": delta["run_failure_calls"],
        "completed_steps": BASE_STEPS + delta["completed_steps"],
        "ppo_update_calls": BASE_PPO,
        "optimizer_step_calls": BASE_ADAM,
        "checkpoint_write_calls": BASE_CHECKPOINT_WRITES,
        "technical_failed_claims": 1 + delta["technical_failed_claims"],
        "new_optimizer_step_calls": sum(int(c.get("optimizer_step_calls", 0) or 0) for _, c in claims),
        "new_checkpoint_write_calls": sum(int(c.get("checkpoint_write_calls", 0) or 0) for _, c in claims),
    }
    return {
        "kind": "round46-r2-v7-cumulative-ledger/1", "task_id": TASK_ID,
        "runner_version": CODE_VERSION, "training_semantics_reused": True,
        "base_identity": {
            "normal_claims": BASE_NORMAL, "environment_runs": BASE_NORMAL,
            "completed_steps": BASE_STEPS, "ppo_updates": BASE_PPO,
            "optimizer_steps": BASE_ADAM,
            "old_execution_ledger_path": str(HERE / "runtime/ROUND46_EXECUTION_LEDGER.json"),
            "old_execution_ledger_sha256": "d433892e98e3466a305644afd5b6669ca75985fd3f0db3c0e6f94ee43051fee6",
            "old_ledger_claim_count_disagrees_with_raw_audit": True,
            "raw_claim_counter_source": "ROUND46_R2_V6_AUDIT_RECOVERY_ENTRY.md; immutable prior read-only audit",
        },
        "normal_cap": 480, "technical_reserve_max": RESERVE_CAP,
        "hard_cap": HARD_CAP, "normal_used": normal_used,
        "reserve_used": reserve_used, "total_used": normal_used + reserve_used,
        "new_claims": len(claims), "new_plan_claims": plan["new_claim_count"],
        "new_delta_counts": delta, "counts": cumulative_counts,
        "claim_event_receipt_count": len(event_ids),
        "claim_event_receipt_sha256": sha256_bytes(canonical_json_bytes(event_ids)),
        "quota_plan_sha256": sha256_bytes(canonical_json_bytes(plan)),
        "claims": [{"path": str(path), "sha256": sha256_file(path),
                    "logical_job": claim.get("logical_job"), "attempt": claim.get("attempt"),
                    "category": claim.get("category"), "slot": claim.get("launch_slot"),
                    "status": claim.get("status")} for path, claim in claims],
    }


def install_eval_only_runtime_guards(runtime: dict[str, Any], plan: dict[str, Any]) -> None:
    base = runtime["v7_base"]
    r45 = runtime["v7_runner"]
    run = runtime["r42_run"]
    r40 = runtime["r40_common"]
    adapter_info = install_quota_adapter(runtime, plan)
    runtime["v7_quota_adapter"] = adapter_info
    original_context = r45._context_for_cell

    def eval_context(kind: str, logical: str, scene: int, metadata: dict[str, Any], kwargs: Any):
        context = original_context(kind, logical, scene, metadata, kwargs)
        if kind == "eval":
            context["allow_technical_retry"] = False
        return context

    r45._context_for_cell = eval_context
    original_record_claim = r45._record_claim

    def record_claim(context: dict[str, Any], logical: str, attempt: int,
                     metadata: dict[str, Any], base_claim: Any):
        path = original_record_claim(context, logical, attempt, metadata, base_claim)
        item = adapter_info["expected"][(logical, int(attempt))]
        r40._update_claim(path, round46_r2_v7={
            "task_id": TASK_ID, "code_version": CODE_VERSION,
            "category": item["category"], "slot": int(item["slot"]),
            "purpose": item["purpose"], "quota_plan_sha256": sha256_bytes(canonical_json_bytes(plan)),
        }, round46_identity={"round": 46, "runner_version": CODE_VERSION,
                             "training_version": runtime["r42_versions"].TRAINING_VERSION},
           round46_attempt_kind=item["category"],
           round46_artifact_paths=r45._claim_paths(context, attempt),
           runner_version=r45.RUNNER_VERSION)
        return path

    r45._record_claim = record_claim

    def ledger_snapshot(*, strict: bool = True) -> dict[str, Any]:
        return build_cumulative_ledger(plan, strict=strict)

    def write_ledger(*, strict: bool = True) -> dict[str, Any]:
        ledger = ledger_snapshot(strict=strict)
        write_json_atomic(V7_LEDGER, ledger)
        return ledger

    run.ledger_snapshot = ledger_snapshot
    run.write_execution_ledger = write_ledger
    runtime["v7_write_ledger"] = write_ledger
    runtime["v7_retry_entry"] = None
    if plan["branch"] == "retry_attempt2":
        retry_item = next(x for x in plan["entries"] if x["purpose"] == "authorized-old-key-retry")
        retry_key = tuple(retry_item["key"])
        original_eval = run.run_model_eval
        original_guarded_execute = run._execute_cell
        closure = dict(zip(original_guarded_execute.__code__.co_freevars,
                           original_guarded_execute.__closure__ or (), strict=True))
        base_cell = closure.get("base_execute")
        base_execute = getattr(base_cell, "cell_contents", None)
        if not callable(base_execute):
            raise V7Error("R45 guarded eval seam has no callable captured R42 execute function")

        def eval_with_exact_retry(*, arm: str, training_seed: int, mode: str,
                                  scenario: int, controller: Any, identity: dict[str, Any]):
            key = (int(training_seed), str(mode), int(scenario))
            if key != retry_key:
                return original_eval(arm=arm, training_seed=training_seed, mode=mode,
                                     scenario=scenario, controller=controller, identity=identity)
            logical = logical_job(retry_key)
            label = f"eval_new_s{retry_key[0]}_{retry_key[1]}_s{retry_key[2]}_r44"
            metadata = {"arm": "NEW", "training_seed": retry_key[0],
                        "mode": retry_key[1], "scenario_seed": retry_key[2]}
            kwargs = {"policy": "learned", "tag": f"r44_eval_new_s{retry_key[0]}_{retry_key[1]}",
                      "controller": controller, "checkpoint_load_identity": identity}
            context = r45._context_for_cell("eval", logical, retry_key[2], metadata, kwargs)
            context.update({"allow_technical_retry": True, "logical_job": logical,
                            "kind": "eval", "scenario_seed": retry_key[2], "tag": kwargs["tag"]})
            r45._planned_key(context)
            prior_claim = read_json(HERE / "attempts/runner_v5_ep0_receipt_conflict/stale_claim/v6/launch_claims/model_eval_new_s11_greedy_scene1301000/attempt_0001.json")
            if (prior_claim.get("status") != "failed_technical"
                    or prior_claim.get("run_return_calls") != 1
                    or prior_claim.get("completed_steps") != 1000
                    or retry_item["attempt"] != 2 or retry_item["category"] != "reserve"):
                raise V7Error("narrow attempt2 retry lacks the exact authorized predecessor")
            context_token = r45.CURRENT_CONTEXT.set(context)
            current_execute = run._execute_cell
            run._execute_cell = base_execute
            try:
                row = original_eval(arm=arm, training_seed=training_seed, mode=mode,
                                    scenario=scenario, controller=controller, identity=identity)
            finally:
                run._execute_cell = current_execute
                r45.CURRENT_CONTEXT.reset(context_token)
            return row

        run.run_model_eval = eval_with_exact_retry
        runtime["v7_retry_entry"] = {"key": list(retry_key), "attempt": 2,
                                      "captured_r42_execute": base_execute}


def run_fixed_matrix(runtime: dict[str, Any], plan: dict[str, Any], *,
                     adoption_row: dict[str, Any] | None) -> list[dict[str, Any]]:
    run = runtime["r42_run"]
    original_eval = run.run_model_eval
    if adoption_row is not None:
        if plan["branch"] != "adoption":
            raise V7Error("adoption row supplied with retry quota branch")
        adopted = copy.deepcopy(adoption_row)
        used = False

        def eval_or_adopt(*, arm: str, training_seed: int, mode: str,
                          scenario: int, controller: Any, identity: dict[str, Any]):
            nonlocal used
            key = (int(training_seed), str(mode), int(scenario))
            if key == OLD_KEY:
                if used:
                    raise V7Error("old slot386 adoption was requested twice")
                used = True
                return copy.deepcopy(adopted)
            return original_eval(arm=arm, training_seed=training_seed, mode=mode,
                                 scenario=scenario, controller=controller, identity=identity)

        run.run_model_eval = eval_or_adopt
        runtime["v7_adoption_wrapper"] = {"original_eval": original_eval, "used": lambda: used}
    rows = runtime["v7_runner"]._run_final_eval_matrix(runtime)
    if adoption_row is not None and not runtime["v7_adoption_wrapper"]["used"]():
        raise V7Error("fixed matrix never consumed the one authorized adoption")
    return rows


def _analysis_union_rows(runtime: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    source_dir = V6_ROOT / "rows"
    target_dir = V7_ROOT / "analysis_inputs" / "rows"
    target_dir.mkdir(parents=True, exist_ok=True)
    records = []
    training_version = runtime["r42_versions"].TRAINING_VERSION
    copied = set()
    for source in sorted(source_dir.glob("*.json")):
        row = read_json(source)
        if row.get("training_arm") != "NEW" or row.get("training_version") != training_version:
            continue
        digest = sha256_file(source)
        dest = target_dir / ("reused_" + source.name)
        if dest.exists() and sha256_file(dest) != digest:
            raise V7Error("analysis input staging row conflicts with the reused source")
        if not dest.exists():
            shutil.copyfile(source, dest)
        records.append({"role": "reused_training_row", "source": str(source),
                        "source_sha256": digest, "staged": str(dest)})
        copied.add(source.name)
    if len(records) != 384:
        raise V7Error(f"analysis reuse found {len(records)}/384 training rows")
    for source in sorted(V7_ROWS.glob("*.json")):
        digest = sha256_file(source)
        dest = target_dir / ("eval_" + source.name)
        if dest.exists() and sha256_file(dest) != digest:
            raise V7Error("analysis input staging eval row conflicts with V7 source")
        if not dest.exists():
            shutil.copyfile(source, dest)
        records.append({"role": "new_eval_row", "source": str(source),
                        "source_sha256": digest, "staged": str(dest)})
    if len(records) != 480:
        raise V7Error(f"analysis input union contains {len(records)}/480 rows")
    manifest = {"kind": "round46-r2-v7-analysis-input-manifest/1",
                "reused_training_rows": 384, "new_eval_rows": 96,
                "rows": records}
    return target_dir, manifest


def analyze_and_report(base: Any, runtime: dict[str, Any], plan: dict[str, Any],
                       rows: list[dict[str, Any]], adoption_detail: dict[str, Any] | None) -> dict[str, Any]:
    matrix_gate = validate_full_matrix(rows, adoption_passed=adoption_detail is not None,
                                       plan=plan)
    union, input_manifest = _analysis_union_rows(runtime)
    input_manifest_sha = write_json_atomic(V7_ROOT / "ROUND46_R2_V7_ANALYSIS_INPUT_MANIFEST.json",
                                           input_manifest)
    previous_rows = runtime["common"].ROWS
    runtime["common"].ROWS = union
    base.REPORT_PATH = V7_ROOT / "ROUND46_R2_V7_REPORT.md"
    base.FINAL_RESULTS_PATH = V7_ROOT / "ROUND46_R2_V7_FINAL_RESULTS.json"
    base.PAIRED_RESULTS_PATH = V7_ROOT / "ROUND46_R2_V7_PAIRED_RESULTS.json"
    base.TARGET_DIFFS_PATH = V7_ROOT / "ROUND46_R2_V7_TARGET_ID_DIFFS.json"
    base.ARTIFACT_MANIFEST_PATH = V7_ROOT / "ROUND46_R2_V7_ARTIFACT_MANIFEST.json"
    base.CHECKPOINT_INDEX_PATH = V7_ROOT / "ROUND46_R2_V7_CHECKPOINT_INDEX.json"
    base.BOOTSTRAP_SEED, base.BOOTSTRAP_REPLICATES = 20261007, 10_000
    try:
        results, detail = base._build_analysis(runtime["v7_runner"], runtime)
    finally:
        runtime["common"].ROWS = previous_rows
    ledger = build_cumulative_ledger(plan, strict=True)
    ledger_sha = write_json_atomic(V7_LEDGER, ledger)
    results["counts"] = ledger["counts"]
    results["budget"] = {
        "normal_planned": 480, "normal_used": ledger["normal_used"],
        "reserve_used": ledger["reserve_used"], "total_used": ledger["total_used"],
        "hard_cap": HARD_CAP, "remaining_hard_cap": HARD_CAP-ledger["total_used"],
    }
    # _build_analysis writes the same stats source to the V7 paths; refresh its
    # final result after adding the cumulative, independently reconciled ledger.
    write_json_atomic(base.FINAL_RESULTS_PATH, results)
    write_report(results, matrix_gate, plan, ledger, adoption_detail,
                 input_manifest_sha, ledger_sha)
    verdict = {
        "kind": "round46-r2-v7-machine-verdict/1", "task_id": TASK_ID,
        "status": results["status"],
        "development_go": bool(results["interpretation"]["development_block_joint_improvement"]),
        "seed_results": results["seed_results"],
        "effective_matrix": matrix_gate,
        "adoption_status": "accepted" if adoption_detail is not None else "retried_attempt2",
        "quota_branch": plan["branch"], "budget": results["budget"],
        "new_training_episodes": 0, "new_ppo_updates": 0,
        "new_optimizer_steps": 0, "new_checkpoint_writes": 0,
        "independent_confirmation": False, "generalization_claim": False,
    }
    verdict_sha = write_json_atomic(V7_ROOT / "ROUND46_R2_V7_MACHINE_VERDICT.json", verdict)
    return {"results": results, "detail": detail, "ledger": ledger,
            "verdict": verdict, "verdict_sha256": verdict_sha,
            "analysis_manifest_sha256": input_manifest_sha,
            "matrix_gate": matrix_gate}


def write_report(results: dict[str, Any], matrix_gate: dict[str, Any],
                 plan: dict[str, Any], ledger: dict[str, Any],
                 adoption: dict[str, Any] | None, input_manifest_sha: str,
                 ledger_sha: str) -> None:
    lines = [
        "# ROUND46-R2 V7 报告：固定 ep128 评估恢复",
        "",
        f"任务：`{TASK_ID}`。本次只做固定模型评估，不恢复训练状态。",
        f"计划分支：`{plan['branch']}`；有效格：{matrix_gate['effective_rows']}/96；新 claim：{matrix_gate['new_claims']}。",
        f"slot386 adoption：{'已接纳' if adoption is not None else '未接纳，按授权 attempt2 重试'}。旧 claim 仍为 `failed_technical`。",
        f"额度累计：normal {ledger['normal_used']}/480，reserve {ledger['reserve_used']}/24，总计 {ledger['total_used']}/504。",
        f"累计实际环境返回步数：{ledger['counts']['completed_steps']}；本次新 PPO/Adam/checkpoint 写入均为0。",
        "",
        "## 科学判定",
        "",
        f"开发块联合门禁：`{results['status']}`。只有三个 seed 各自 greedy 的逐目标按时集合逐场景等于 immediate，且每个 seed 的平均计费严格低于 late 的 4,838.25 B/集，才判 development_go。",
        "开发场景已用于方案选择；结果不构成独立确认、正式非劣、总体零损失或泛化证据。",
        "",
        "| Seed | Greedy 按时任务均值 | Greedy 字节均值 | 与 immediate 目标集合逐场景一致 | 低于 late 成本 | Sampled 任务均值 | Sampled 字节均值 |",
        "|---:|---:|---:|---|---|---:|---:|",
    ]
    for seed in SEEDS:
        item = results["seed_results"][str(seed)]
        lines.append(
            f"| {seed} | {item['greedy_mean_on_time_task_count']:.4f} | "
            f"{item['greedy_mean_spent_bytes']:.2f} | "
            f"{item['immediate_exact_task_id_sets_preserved_greedy']} | "
            f"{item['greedy_cost_strictly_below_late_rule']} | "
            f"{item['sampled_mean_on_time_task_count']:.4f} | "
            f"{item['sampled_mean_spent_bytes']:.2f} |"
        )
    lines += [
        "", "## 原件与增量",
        "", "旧 386 normal claim 与失败记录均保留；V7 claims/slots 为新原子增量。原始 ledger 对环境计数存在已登记的 384/386 差异，本报告的基数采用既有只读逐 claim 审计 386，不重写原账本。",
        f"分析输入清单 SHA256：`{input_manifest_sha}`。累计账本 SHA256：`{ledger_sha}`。",
        "配对 bootstrap、目标身份差集和绝对结果分别见 `ROUND46_R2_V7_PAIRED_RESULTS.json`、`ROUND46_R2_V7_TARGET_ID_DIFFS.json` 与 `ROUND46_R2_V7_FINAL_RESULTS.json`。",
        "原始权重、tape、日志和分析输入副本留在 Git 外；本轮不推送、不合并、不部署。",
    ]
    path = V7_ROOT / "ROUND46_R2_V7_REPORT.md"
    raw = ("\n".join(lines) + "\n").encode("utf-8")
    path.write_bytes(raw)


def validate_full_matrix(rows: list[dict[str, Any]], *, adoption_passed: bool,
                         plan: dict[str, Any], root: Path = V7_ROOT) -> dict[str, Any]:
    expected = set(fixed_eval_keys())
    by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["training_seed"]), str(row["eval_mode"]), int(row["scenario_seed"]))
        if key in by_key:
            raise V7Error(f"duplicate effective evaluation row: {key}")
        by_key[key] = row
    if set(by_key) != expected or len(by_key) != 96:
        raise V7Error(f"effective matrix is not the fixed 96 keys: {len(by_key)}/96")
    for key, row in by_key.items():
        if (row.get("model_arm") != "NEW" or row.get("steps") != 1000
                or row.get("training_version") is None
                or row.get("comm_action_semantics") != "one-shot-send-skip/1"
                or row.get("optimizer_steps") != 0):
            raise V7Error(f"evaluation row identity or eval-only invariant failed: {key}")
    if adoption_passed:
        oldrow = read_json(root.parent / "v6/rows/v2_r44_eval_new_s11_greedy_s1301000_a1.json")
        if by_key[OLD_KEY] != oldrow:
            raise V7Error("adopted row differs from the immutable original row")
    slots = {(x["logical_job"], x["attempt"]): x for x in plan["entries"]}
    claims = {}
    for path in V7_CLAIMS.glob("*/*.json"):
        claim = read_json(path)
        key = (str(claim.get("logical_job")), int(claim.get("attempt", -1)))
        if key in claims:
            raise V7Error("duplicate V7 claim file")
        claims[key] = (path, claim)
    if set(claims) != set(slots):
        raise V7Error(f"V7 claim schedule mismatch: {len(claims)}/{len(slots)}")
    for key, (path, claim) in claims.items():
        if claim.get("status") != "complete" or claim.get("completed_steps") != 1000:
            raise V7Error(f"V7 planned claim is not a completed 1000-step eval: {key}")
        expected_slot = slots[key]
        if (claim.get("category") != expected_slot["category"]
                or int(claim.get("launch_slot", -1)) != expected_slot["slot"]):
            raise V7Error(f"V7 claim slot identity mismatch: {key}")
    return {"status": "passed", "effective_rows": len(by_key),
            "new_claims": len(claims), "adopted_old_key": adoption_passed,
            "row_keys_sha256": sha256_bytes(canonical_json_bytes(sorted([list(x) for x in expected])))}


def main() -> None:
    raise SystemExit("V7 execution is gated: use the committed prelaunch freeze and run command.")


if __name__ == "__main__":
    main()
