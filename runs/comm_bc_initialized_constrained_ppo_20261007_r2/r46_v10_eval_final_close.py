"""ROUND46-R2 V10: final close — adopt the completed 1301002 cell, fix the
progress/final gate contract, and run the last 93 cells.

Core facts (exclusive claims, consumed slots, original artifacts, receipts) are
permanent. Mid-run ledger snapshots are derived outputs: partial-tolerant but
verified real/unique/under-cap. The final gate requires the full frozen plan.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import r46_v7_eval_only as V7
import r46_v8_eval_booking_close as V8
import r46_v9_eval_runtime_close as V9

TASK_ID = "ROUND46-R2-V10-EVAL-FINAL-CLOSE"
BRANCH = "fix/round46-r2-v10-eval-final-close"
CODE_VERSION = "round46-r2-v10-eval-final-close/1"

HERE = V7.HERE
V10_ROOT = HERE / "v10_eval_final_close"
V10_RUNTIME = V10_ROOT / "runtime"
V10_ROWS = V10_ROOT / "rows"
V10_LOGS = V10_ROOT / "logs"
V10_CLAIMS = V10_ROOT / "launch_claims"
V10_SLOTS = V10_ROOT / "launch_slots"
V10_RECEIPTS = V10_ROOT / "receipts"
V10_LEDGER = V10_RUNTIME / "ROUND46_R2_V10_EXECUTION_LEDGER.json"
V10_LEDGER_EVENTS = V10_RUNTIME / "ledger_events"
V10_WARNINGS = V10_RUNTIME / "derived_snapshot_warnings"

ADOPTED_KEYS = ((11, "greedy", 1_301_000), (11, "greedy", 1_301_001),
                (11, "greedy", 1_301_002))
ADOPTED_KEYS_LOGICAL = ("model_eval_new_s11_greedy_scene1301000",
                        "model_eval_new_s11_greedy_scene1301001",
                        "model_eval_new_s11_greedy_scene1301002")
V9_PRELAUNCH_SHORT = "2518344"

FROZEN_MODELS = {
    11: (V7.MODEL_FILE_SHA, V7.MODEL_PARAMETER_SHA),
    22: ("aae4aa1b24b7e7ffadffeb7a09d45935f44970ac62a8cfd4344630248472cc62",
         "b88a6b377d9e18c3cef17ab5205f7652b1394fcfbc6b28e5642140025028caf4"),
    33: ("10f3fc8d2c2a788b7e48e1f9021b5b234c39a00706bfe486e6f2bab84a0fb9b9",
         "dc2903397865c853c59ed631e7b0a699fcb2d55b7dee8956101c53984b36b821"),
}
PLACED_TARGET_IDS = tuple(f"piloc-e1-target-{index}" for index in range(6))

V9Error = V9.V9Error
sha256_bytes = V7.sha256_bytes
sha256_file = V7.sha256_file
canonical_json_bytes = V7.canonical_json_bytes
read_json = V7.read_json
read_jsonl = V7.read_jsonl
write_new_json = V7.write_new_json
write_json_atomic = V7.write_json_atomic
fixed_eval_keys = V7.fixed_eval_keys
logical_job = V7.logical_job
validate_episode_closure = V7.validate_episode_closure
verify_greedy_tape = V7.verify_greedy_tape


class V10Error(RuntimeError):
    pass


def _prepare_v10_dirs() -> None:
    for path in (V10_ROOT, V10_RUNTIME, V10_ROWS, V10_LOGS, V10_CLAIMS, V10_SLOTS,
                 V10_RECEIPTS, V10_LEDGER_EVENTS, V10_WARNINGS, V10_ROOT / "rejected",
                 V10_ROOT / "adoption", V10_ROOT / "analysis_inputs"):
        path.mkdir(parents=True, exist_ok=True)
    for category in ("normal", "reserve"):
        (V10_SLOTS / category).mkdir(parents=True, exist_ok=True)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def build_quota_plan() -> dict[str, Any]:
    """Final 93-cell map: 92 normal slots 389..480 plus reserve slot 2."""
    keys = list(fixed_eval_keys())
    adopted = list(ADOPTED_KEYS)
    remaining = [k for k in keys if k not in adopted]
    if len(remaining) != 93:
        raise V10Error(f"remaining key count is {len(remaining)}/93")
    entries: list[dict[str, Any]] = []
    for index, key in enumerate(remaining):
        reserve = index == 92
        entries.append({
            "key": list(key), "logical_job": logical_job(key), "attempt": 1,
            "category": "reserve" if reserve else "normal",
            "slot": 2 if reserve else 389 + index,
            "purpose": "technical-capacity-recovery" if reserve else "planned-eval",
        })
    normal = [x for x in entries if x["category"] == "normal"]
    reserve = [x for x in entries if x["category"] == "reserve"]
    plan = {
        "kind": "round46-r2-v10-quota-plan/1", "branch": "adoption_1301002",
        "adopted_keys": [list(k) for k in ADOPTED_KEYS],
        "effective_key_count": 96,
        "new_claim_count": len(entries), "base_normal_used": V7.BASE_NORMAL,
        "previously_consumed": {"reserve_slot_1": "adopted (V8)",
                                "normal_slot_387": "adopted (V9)",
                                "normal_slot_388": "adopted (this round)"},
        "new_normal_slots": [x["slot"] for x in normal],
        "new_reserve_slots": [x["slot"] for x in reserve],
        "normal_used_after": 388 + len(normal),
        "reserve_used_after": 1 + len(reserve),
        "total_used_after": 389 + len(entries),
        "hard_cap": V7.HARD_CAP, "technical_reserve_cap": V7.RESERVE_CAP,
        "entries": entries,
    }
    if (plan["new_normal_slots"] != list(range(389, 481))
            or plan["new_claim_count"] != 93
            or (len(normal), len(reserve), plan["total_used_after"]) != (92, 1, 482)
            or len({(tuple(x["key"]), x["attempt"]) for x in entries}) != len(entries)):
        raise V10Error("V10 quota plan does not match the authorized 93-cell map")
    return plan


def _plan_slot_expectations(plan: dict[str, Any]) -> dict[str, int]:
    return {"normal": sum(1 for x in plan["entries"] if x["category"] == "normal"),
            "reserve": sum(1 for x in plan["entries"] if x["category"] == "reserve")}


def verify_core_ledger_state(plan: dict[str, Any], *, root: Path) -> dict[str, Any]:
    """Core-facts check over the atomic claim files: real, unique, planned, under cap."""
    root = Path(root)
    planned = {(str(x["logical_job"]), int(x["attempt"])): x for x in plan["entries"]}
    claims = []
    seen: set[tuple[str, int]] = set()
    problems: list[str] = []
    for path in sorted((root / "launch_claims").glob("*/*.json")):
        try:
            claim = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            problems.append(f"unparsable claim {path}: {error}")
            continue
        ident = (str(claim.get("logical_job")), int(claim.get("attempt", -1)))
        if ident in seen:
            problems.append(f"duplicate claim attempt: {ident}")
        seen.add(ident)
        if ident not in planned:
            problems.append(f"unplanned claim: {ident}")
            continue
        item = planned[ident]
        if (claim.get("category") != item["category"]
                or int(claim.get("launch_slot", -1)) != int(item["slot"])):
            problems.append(f"claim slot identity mismatch: {ident}")
        if claim.get("status") not in ("claimed_before_environment", "complete",
                                       "failed_technical", "not_started_quota_exhausted"):
            problems.append(f"unknown claim status: {ident} {claim.get('status')!r}")
        claims.append({"path": str(path), "logical_job": ident[0],
                       "attempt": ident[1], "status": claim.get("status"),
                       "completed_steps": int(claim.get("completed_steps", 0) or 0),
                       "optimizer_step_calls": int(claim.get("optimizer_step_calls", 0) or 0),
                       "checkpoint_write_calls": int(claim.get("checkpoint_write_calls", 0) or 0)})
    slot_counts = {category: len(list((root / "launch_slots" / category).glob("*.json")))
                   for category in ("normal", "reserve")}
    expectations = _plan_slot_expectations(plan)
    for category, count in slot_counts.items():
        if count > expectations[category]:
            problems.append(f"over-cap {category} slots: {count}/{expectations[category]}")
    return {
        "ok": not problems,
        "problems": problems,
        "claims": claims,
        "claim_count": len(claims),
        "plan_claim_count": plan["new_claim_count"],
        "slot_counts": slot_counts,
        "slot_expectations": expectations,
    }


def build_cumulative_ledger(plan: dict[str, Any], *, strict: bool = False,
                            root: Path | None = None) -> dict[str, Any]:
    """Derived cumulative ledger.

    strict=False (mid-run snapshots): partial progress allowed; every recorded claim
    must be real, unique, planned and under cap.
    strict=True (final gate): the full frozen plan must be complete and all
    cumulative counts closed (==, not merely under cap).
    """
    root = V10_ROOT if root is None else Path(root)
    core = verify_core_ledger_state(plan, root=root)
    if not core["ok"]:
        raise V10Error(f"core ledger state is broken: {core['problems']}")
    claims = core["claims"]
    if strict and core["claim_count"] != plan["new_claim_count"]:
        raise V10Error(f"final ledger requires all {plan['new_claim_count']} planned "
                       f"claims complete: found {core['claim_count']}")
    if strict:
        not_complete = [c for c in claims if c["status"] != "complete"]
        if not_complete:
            raise V10Error(f"final ledger has non-complete planned claims: {not_complete}")
    planned = {(str(x["logical_job"]), int(x["attempt"])): x for x in plan["entries"]}
    delta = {"environment_constructor_calls": 0, "joint_run_constructed_calls": 0,
             "run_invocation_calls": 0, "joint_run_method_calls": 0,
             "run_return_calls": 0, "run_failure_calls": 0,
             "completed_steps": 0, "technical_failed_claims": 0}
    event_ids: list[str] = []
    for path in sorted((root / "launch_claims").glob("*/*.json")):
        claim = read_json(path)
        ident = (str(claim.get("logical_job")), int(claim.get("attempt", -1)))
        for field in delta:
            delta[field] += int(claim.get(field, 0) or 0)
        if claim.get("status") == "failed_technical":
            delta["technical_failed_claims"] += 1
        if claim.get("status") in ("complete", "failed_technical"):
            event = {
                "kind": "round46-r2-v10-claim-event/1", "task_id": TASK_ID,
                "logical_job": ident[0], "attempt": ident[1],
                "claim_path": str(path), "claim_sha256": sha256_file(path),
                "status": claim.get("status"),
                "environment_constructor_calls": int(claim.get("environment_constructor_calls", 0)),
                "run_return_calls": int(claim.get("run_return_calls", 0)),
                "completed_steps": int(claim.get("completed_steps", 0)),
                "optimizer_step_calls": int(claim.get("optimizer_step_calls", 0)),
                "checkpoint_write_calls": int(claim.get("checkpoint_write_calls", 0)),
            }
            events_dir = root / "runtime" / "ledger_events"
            events_dir.mkdir(parents=True, exist_ok=True)
            event_path = events_dir / f"{path.parent.name}_a{ident[1]:04d}.json"
            event_ids.append(write_new_json(event_path, event))
    slot_counts = core["slot_counts"]
    adopted_normal = 2
    adopted_reserve = 1
    counts = {
        "environment_constructor_calls": 386 + 2 + delta["environment_constructor_calls"],
        "joint_run_constructed_calls": 386 + 2 + delta["joint_run_constructed_calls"],
        "run_invocation_calls": 386 + 2 + delta["run_invocation_calls"],
        "joint_run_method_calls": 386 + 2 + delta["joint_run_method_calls"],
        "run_return_calls": 386 + 2 + delta["run_return_calls"],
        "run_failure_calls": delta["run_failure_calls"],
        "completed_steps": V7.BASE_STEPS + 3000 + delta["completed_steps"],
        "ppo_update_calls": V7.BASE_PPO,
        "optimizer_step_calls": V7.BASE_ADAM,
        "checkpoint_write_calls": V7.BASE_CHECKPOINT_WRITES,
        "technical_failed_claims": 2 + delta["technical_failed_claims"],
        "new_optimizer_step_calls": sum(c["optimizer_step_calls"] for c in claims),
        "new_checkpoint_write_calls": sum(c["checkpoint_write_calls"] for c in claims),
    }
    return {
        "kind": "round46-r2-v10-cumulative-ledger/1", "task_id": TASK_ID,
        "runner_version": CODE_VERSION, "training_semantics_reused": True,
        "snapshot_mode": "final" if strict else "mid_run_partial",
        "base_identity": {
            "normal_claims": V7.BASE_NORMAL, "environment_runs": V7.BASE_NORMAL,
            "completed_steps": V7.BASE_STEPS, "ppo_updates": V7.BASE_PPO,
            "optimizer_steps": V7.BASE_ADAM,
            "adopted_cells": [
                {"logical_job": ADOPTED_KEYS_LOGICAL[0], "attempt": 2, "slot": 1,
                 "category": "reserve", "receipt": "V8"},
                {"logical_job": ADOPTED_KEYS_LOGICAL[1], "attempt": 1, "slot": 387,
                 "category": "normal", "receipt": "V9"},
                {"logical_job": ADOPTED_KEYS_LOGICAL[2], "attempt": 1, "slot": 388,
                 "category": "normal", "receipt": "V10"},
            ],
            "raw_claim_counter_source": "ROUND46_R2_V6_AUDIT_RECOVERY_ENTRY.md; immutable prior read-only audit",
        },
        "normal_cap": 480, "technical_reserve_max": V7.RESERVE_CAP,
        "hard_cap": V7.HARD_CAP,
        "normal_used": 386 + adopted_normal + slot_counts["normal"],
        "reserve_used": adopted_reserve + slot_counts["reserve"],
        "total_used": (386 + adopted_normal + slot_counts["normal"])
                      + (adopted_reserve + slot_counts["reserve"]),
        "new_claims": core["claim_count"], "new_plan_claims": plan["new_claim_count"],
        "new_delta_counts": delta, "counts": counts,
        "claim_event_receipt_count": len(event_ids),
        "claim_event_receipt_sha256": sha256_bytes(canonical_json_bytes(event_ids)),
        "quota_plan_sha256": sha256_bytes(canonical_json_bytes(plan)),
        "claims": claims,
    }


def write_ledger_closure(plan: dict[str, Any], root: Path) -> dict[str, Any]:
    """Standalone derived-snapshot refresh guard.

    Same contract as the per-cell wrapper: a derived snapshot may fail without
    touching already-persisted core claims; genuinely broken core state re-raises.
    """
    root = Path(root)
    try:
        ledger = build_cumulative_ledger(plan, strict=False, root=root)
        write_json_atomic(root / "runtime" / "ROUND46_R2_V10_EXECUTION_LEDGER.json", ledger)
        return ledger
    except Exception as snapshot_error:
        core = verify_core_ledger_state(plan, root=root)
        warning = {
            "kind": "round46-r2-v10-derived-snapshot-warning/1",
            "task_id": TASK_ID,
            "error": f"{type(snapshot_error).__name__}: {snapshot_error}",
            "core_state": core,
            "at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if not core["ok"]:
            raise
        warnings_dir = root / "runtime" / "derived_snapshot_warnings"
        warnings_dir.mkdir(parents=True, exist_ok=True)
        write_new_json(warnings_dir / f"derived_snapshot_warning_{int(time.time()*1000)}.json",
                       warning)
        return {"kind": "round46-r2-v10-cumulative-ledger/1",
                "task_id": TASK_ID, "status": "deferred",
                "warning_recorded": True,
                "core_state": {"claim_count": core["claim_count"],
                               "slot_counts": core["slot_counts"]}}


def configure_eval_runtime(base: Any | None = None) -> tuple[Any, dict[str, Any]]:
    """Load the actual frozen runtime, bind every registration-chain dependency on
    the live module objects, and keep all writes inside the private V10 root."""
    _prepare_v10_dirs()
    if base is None:
        base = V9.V7._load_module(HERE / "r46_execute_r2.py", "round46_r2_v10_base")

    def v10_seed_namespace() -> dict[str, Any]:
        _prepare_v10_dirs()
        for path in (base.V6_DIR, base.V6_RUNTIME_DIR, base.V6_CHECKPOINTS_DIR,
                     base.V6_LOGS_DIR, base.V6_ROWS_DIR, base.V6_REJECTED_DIR):
            if not path.is_dir():
                raise V10Error(f"read-only V6 input namespace is missing: {path.name}")
        return {"v10_root": str(V10_ROOT), "v9_root": str(V9.V9_ROOT),
                "v8_root": str(V8.V8_ROOT), "v6_write": False}

    base._seed_v6_output_namespace = v10_seed_namespace
    base._copy_runtime_aliases = lambda identity: V9.V7._verify_private_runtime_aliases(base, identity)
    base._rebind_checkpoint_journal = lambda runtime: V9.V7._read_only_checkpoint_route(base, runtime)
    base._verify_prelaunch_identity = lambda runtime, **_kw: _verify_v10_freeze(base, runtime)
    runtime_module = base._configure_runtime_module()
    runtime = base._RUNTIME
    if runtime is None:
        raise V10Error("frozen runtime did not initialize")

    common, r40 = runtime["common"], runtime["r40_common"]
    common.OUTPUT_DIR, common.LOGS, common.ROWS = V10_ROOT, V10_LOGS, V10_ROWS
    common.REJECTED = V10_ROOT / "rejected"
    common.CHECKPOINTS = base.V6_CHECKPOINTS_DIR
    common.NORMAL_LAUNCHES, common.RESERVE_LAUNCHES, common.LAUNCH_CAP = 480, 24, V7.HARD_CAP
    r40.ROUND_DIR, r40.NORMAL_CAP, r40.RESERVE_CAP, r40.HARD_CAP = V10_ROOT, 480, 24, V7.HARD_CAP
    r40.plan = lambda: {"normal_total": 480, "technical_reserve_max": 24,
                        "hard_cap": V7.HARD_CAP, "task_id": TASK_ID}
    run = runtime["r42_run"]
    run.RUN_DIR = V10_ROOT
    for key in ("r44_runtime", "r44_prepare", "r42_analysis"):
        mod = runtime.get(key)
        if mod is not None and hasattr(mod, "RUN_DIR"):
            mod.RUN_DIR = V10_ROOT
    r45 = runtime_module
    r45.RUN_DIR = V10_ROOT
    v10_runner_version = CODE_VERSION + "+" + str(base.ROUND46_RUNNER_VERSION)
    r45.RUNNER_VERSION = v10_runner_version
    for name in ("runner", "r42_run", "r44_runtime", "r44_prepare"):
        mod = runtime.get(name)
        if mod is not None and hasattr(mod, "RUNNER_VERSION"):
            mod.RUNNER_VERSION = v10_runner_version
    # Explicit binding entry on the LIVE r44_runtime module object.
    r44_module = sys.modules.get("r44_runtime")
    if r44_module is None:
        raise V10Error("live r44_runtime booking module is unavailable")

    def _find_booking_reader(fn: Any, seen: set[int]) -> Any | None:
        stack = [fn]
        while stack:
            current = stack.pop()
            code = getattr(current, "__code__", None)
            if (getattr(current, "__globals__", None) is r44_module.__dict__
                    and code is not None
                    and "create_claim_artifact_receipt" in code.co_names):
                return current
            for cell in (getattr(current, "__closure__", None) or ()):
                value = getattr(cell, "cell_contents", None)
                if callable(value) and hasattr(value, "__code__") and id(value) not in seen:
                    seen.add(id(value))
                    stack.append(value)
        return None

    booked = _find_booking_reader(runtime["runner"]._booked, set())
    if booked is None:
        raise V10Error("live r44 booking function that reads the runner version was not found")
    if booked.__globals__ is not r44_module.__dict__:
        raise V10Error("the r44 booking reader does not read the live module globals")
    r44_module.RUNNER_VERSION = v10_runner_version
    r44_module.RUN_DIR = V10_ROOT
    for mod_name in ("r44_prepare",):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            if hasattr(mod, "RUNNER_VERSION"):
                mod.RUNNER_VERSION = v10_runner_version
            if hasattr(mod, "RUN_DIR"):
                mod.RUN_DIR = V10_ROOT
    r45.R3_EXECUTION_LEDGER_PATH = V10_LEDGER
    r45.R3_REPORT_PATH = V10_ROOT / "ROUND46_R2_V10_REPORT.md"
    r45.R3_FINAL_RESULTS_PATH = V10_ROOT / "ROUND46_R2_V10_FINAL_RESULTS.json"
    r45._task_state = lambda stage, **kw: v10_task_state(stage, **kw)
    base.CURRENT_TASK_PATH = V10_ROOT / "ROUND46_R2_V10_TASK_STATE.json"
    base._task_state = lambda stage, **kw: v10_task_state(stage, **kw)
    runtime["common"].OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (V10_LOGS, V10_ROWS, V10_ROOT / "rejected"):
        path.mkdir(parents=True, exist_ok=True)
    save = runtime["trainer"].save_checkpoint
    def forbidden_checkpoint_save(*_args: Any, **_kwargs: Any) -> None:
        raise V10Error("checkpoint save is forbidden in the V10 eval-only runtime")
    runtime["trainer"].save_checkpoint = forbidden_checkpoint_save
    runtime["v10_base"] = base
    runtime["v10_runner"] = r45
    runtime["v10_runner_version"] = v10_runner_version
    runtime["v10_r44_module"] = r44_module
    runtime["v10_r44_booked"] = booked
    return base, runtime


def v10_task_state(stage: str, **updates: Any) -> None:
    path = V10_ROOT / "ROUND46_R2_V10_TASK_STATE.json"
    state = read_json(path) if path.is_file() else {}
    state.update({"kind": "round46-r2-v10-task-state/1", "task_id": TASK_ID,
                  "stage": stage, "branch": BRANCH,
                  "head": _git(HERE.parents[1], "rev-parse", "HEAD"),
                  "updated_at_local": time.strftime("%Y-%m-%d %H:%M:%S")})
    state.update(updates)
    write_json_atomic(path, state)


def assert_binding(runtime: dict[str, Any], *, key: tuple[int, str, int],
                   arm: str, identity: dict[str, Any], plan: dict[str, Any]) -> None:
    """Cheap pre-construction assertion over the REAL booking consumers."""
    composite = runtime["v10_runner_version"]
    rowgen = runtime["r42_run"].RUNNER_VERSION
    r44 = runtime["v10_r44_module"]
    if rowgen != composite or r44.RUNNER_VERSION != composite:
        raise V10Error("row generation or booking disagrees with the V10 execution identity")
    if Path(r44.RUN_DIR) != V10_ROOT:
        raise V10Error(f"booking RUN_DIR does not point at the private V10 root: {r44.RUN_DIR}")
    if Path(runtime["v10_runner"].RUN_DIR) != V10_ROOT:
        raise V10Error("receipt RUN_DIR does not point at the private V10 root")
    if str(runtime["common"].OUTPUT_DIR) != str(V10_ROOT) \
            or str(runtime["common"].ROWS) != str(V10_ROWS):
        raise V10Error("runtime output namespace is not the private V10 root")
    expected = {tuple(x["key"]) for x in plan["entries"]}
    expected.update(ADOPTED_KEYS)
    if key not in expected:
        raise V10Error(f"unplanned evaluation cell rejected before construction: {key}")
    if arm != "NEW":
        raise V10Error("V10 evaluation must use the NEW model arm")
    expected_file, expected_params = FROZEN_MODELS[key[0]]
    if identity.get("file_sha256") != expected_file:
        raise V10Error(f"controller identity does not bind the frozen s{key[0]} ep128 file")
    if (identity.get("parameter_identity", {}).get("parameter_sha256") != expected_params):
        raise V10Error(f"controller identity does not bind the frozen s{key[0]} parameters")


def install_quota_adapter(runtime: dict[str, Any], plan: dict[str, Any], *,
                          output_root: Path = V10_ROOT) -> dict[str, Any]:
    r40 = runtime["r40_common"]
    expected = {(str(x["logical_job"]), int(x["attempt"])): x for x in plan["entries"]}
    records: list[dict[str, Any]] = []
    original_claim = r40.claim_launch
    original_slot = r40._reserve_slot

    def reserve_slot(category: str, *, logical_job: str, attempt: int) -> int:
        item = expected.get((str(logical_job), int(attempt)))
        if item is None or item["category"] != category:
            raise V10Error("real R40 slot seam reached an unfrozen V10 claim")
        path = Path(output_root) / "launch_slots" / category / f"{int(item['slot']):04d}.json"
        payload = {"task_id": TASK_ID, "category": category, "slot": int(item["slot"]),
                   "logical_job": logical_job, "attempt": int(attempt),
                   "purpose": item["purpose"], "claimed_at_unix": time.time()}
        raw = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        try:
            fd = __import__("os").open(str(path), __import__("os").O_CREAT
                                       | __import__("os").O_EXCL | __import__("os").O_WRONLY)
        except FileExistsError as error:
            raise V10Error(f"V10 fixed launch slot already consumed: {category}/{item['slot']}") from error
        with __import__("os").fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(raw); stream.flush()
            __import__("os").fsync(stream.fileno())
        return int(item["slot"])

    def claim_launch(logical_job: str, *, category: str, attempt: int,
                     metadata: dict[str, Any] | None = None) -> Path:
        item = expected.get((str(logical_job), int(attempt)))
        if item is None:
            raise V10Error(f"unplanned real R40 claim rejected: {logical_job}#{attempt}")
        if category != item["category"]:
            raise V10Error("R40 claim category differs from the frozen V10 quota map")
        user_meta = dict(metadata or {})
        user_meta.update({"v10_task_id": TASK_ID, "v10_purpose": item["purpose"],
                          "v10_slot": int(item["slot"])})
        path = original_claim(logical_job, category=item["category"],
                              attempt=int(attempt), metadata=user_meta)
        claim = read_json(path)
        if (claim.get("category") != item["category"]
                or int(claim.get("launch_slot", -1)) != int(item["slot"])):
            raise V10Error("R40 atomic claim did not bind the planned V10 slot")
        r40._update_claim(path, round46_r2_v10={
            "task_id": TASK_ID, "code_version": CODE_VERSION,
            "category": item["category"], "slot": int(item["slot"]),
            "purpose": item["purpose"],
            "quota_plan_sha256": sha256_bytes(canonical_json_bytes(plan)),
        })
        records.append({"logical_job": logical_job, "attempt": int(attempt),
                        "category": item["category"], "slot": int(item["slot"])})
        return path

    r40._reserve_slot = reserve_slot
    r40.claim_launch = claim_launch
    return {"expected": expected, "claim_records": records,
            "original_claim_launch": original_claim, "original_reserve_slot": original_slot}


def install_eval_only_runtime_guards(runtime: dict[str, Any], plan: dict[str, Any]) -> None:
    base = runtime["v10_base"]
    r45 = runtime["v10_runner"]
    run = runtime["r42_run"]
    r40 = runtime["r40_common"]
    adapter_info = install_quota_adapter(runtime, plan)
    runtime["v10_quota_adapter"] = adapter_info
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
        item = adapter_info["expected"][(str(logical), int(attempt))]
        r40._update_claim(path, round46_r2_v10={
            "task_id": TASK_ID, "code_version": CODE_VERSION,
            "category": item["category"], "slot": int(item["slot"]),
            "purpose": item["purpose"],
            "quota_plan_sha256": sha256_bytes(canonical_json_bytes(plan)),
        }, round46_identity={"round": 46, "runner_version": CODE_VERSION,
                             "training_version": runtime["r42_versions"].TRAINING_VERSION},
           round46_attempt_kind=item["category"],
           round46_artifact_paths=r45._claim_paths(context, attempt),
           runner_version=r45.RUNNER_VERSION)
        return path

    r45._record_claim = record_claim

    def ledger_snapshot(*, strict: bool = False) -> dict[str, Any]:
        # Mid-run snapshots are partial-tolerant; the final gate re- verifies with
        # strict=True in analyze_and_report.
        return build_cumulative_ledger(plan, strict=strict)

    def write_ledger(*, strict: bool = True) -> dict[str, Any]:
        """Derived-snapshot refresh, isolated from the per-cell core registration.

        A failure here never flips an already-persisted core claim: core facts are
        re-verified and a warning receipt is recorded; only genuinely broken core
        state re-raises and stops the run.
        """
        try:
            ledger = ledger_snapshot(strict=False)
            write_json_atomic(V10_LEDGER, ledger)
            return ledger
        except Exception as snapshot_error:
            core = verify_core_ledger_state(plan, root=V10_ROOT)
            warning = {
                "kind": "round46-r2-v10-derived-snapshot-warning/1",
                "task_id": TASK_ID,
                "error": f"{type(snapshot_error).__name__}: {snapshot_error}",
                "core_state": core,
                "at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if not core["ok"]:
                raise
            write_new_json(V10_WARNINGS / f"derived_snapshot_warning_{int(time.time())}.json",
                           warning)
            return {"kind": "round46-r2-v10-cumulative-ledger/1",
                    "task_id": TASK_ID, "status": "deferred",
                    "warning_recorded": True,
                    "core_state": {"claim_count": core["claim_count"],
                                   "slot_counts": core["slot_counts"]}}

    run.ledger_snapshot = ledger_snapshot
    run.write_execution_ledger = write_ledger
    runtime["v10_write_ledger"] = write_ledger

    original_eval = run.run_model_eval
    adopted_rows = {
        ADOPTED_KEYS[0]: read_json(V9.V8.V8_ROOT / "adoption/attempt2_effective_row.json"),
        ADOPTED_KEYS[1]: read_json(V9.V9_ROOT / "adoption/cell1301001_effective_row.json"),
        ADOPTED_KEYS[2]: read_json(V10_ROOT / "adoption/cell1301002_effective_row.json"),
    }

    def eval_or_adopt(*, arm: str, training_seed: int, mode: str,
                      scenario: int, controller: Any, identity: dict[str, Any]):
        key = (int(training_seed), str(mode), int(scenario))
        assert_binding(runtime, key=key, arm=arm, identity=identity, plan=plan)
        if key in adopted_rows:
            if marker["used"].get(key):
                raise V10Error(f"the adopted cell {key} was requested twice")
            marker["used"][key] = True
            return copy.deepcopy(adopted_rows[key])
        return original_eval(arm=arm, training_seed=training_seed, mode=mode,
                             scenario=scenario, controller=controller, identity=identity)

    marker = {"used": {}}

    def adoption_used() -> dict[str, bool]:
        return dict(marker["used"])

    run.run_model_eval = eval_or_adopt
    runtime["v10_adoption_wrapper"] = {"original_eval": original_eval,
                                       "used": adoption_used, "marker": marker}


# --------------------------------------------------------------------------
# technical adoption of the completed 1301002 cell (performance-neutral)
# --------------------------------------------------------------------------

CELL_1301002_IDENTITY = {
    "claim_sha256": "8c19bcc8ca0e5af887b1cdc2a92fc9644dfe34a9c2e1d557b8c90d35e6481556",
    "row_sha256": "a0c5b0debd0a44e2312e487784bdc2644b9e8e471409fb14367dd852a3eba70a",
    "lifecycle_sha256": "8a1d032a2743e498ae809ad92511f1092bdac0e1e3d0567ac7c262eedd2960ba",
    "transcript_sha256": "663a6f6954707c21a773a2832d8ea99d942262f9f3b516d2d495765095456af9",
    "tape_sha256": "f6d38c10d493c9de605f25249e11c4c403132c4c58cbba2bf4e6d11feaf2d465",
    "slot_record_sha256": "9d318fa948640d895eb0c13fe9063579adcb2b02baa29030481bc07c536dcce2",
    "v9_stop_state_sha256": "4b0adbcd9f0c9b3acc5f055b29dca49a50bda09b5b05ba4a0189a56bcf6be1a2",
}
CELL_1301002_REL = {
    "claim": Path("v9_eval_runtime_close/launch_claims/model_eval_new_s11_greedy_scene1301002/attempt_0001.json"),
    "row": Path("v9_eval_runtime_close/rows/v2_r44_eval_new_s11_greedy_s1301002_a1.json"),
    "lifecycle": Path("v9_eval_runtime_close/logs/v2_r44_eval_new_s11_greedy_s1301002_a1_rp.jsonl"),
    "transcript": Path("v9_eval_runtime_close/logs/v2_r44_eval_new_s11_greedy_s1301002_a1_ct.jsonl"),
    "tape": Path("v9_eval_runtime_close/logs/v2_r44_eval_new_s11_greedy_s1301002_a1_tape.jsonl"),
    "slot": Path("v9_eval_runtime_close/launch_slots/normal/0388.json"),
}


def audit_cell_1301002(base: Any, runtime: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read-only technical audit of the completed-but-unbooked 1301002 cell."""
    import numpy as np
    import torch

    repo = base.REPO
    prelaunch = _git(repo, "rev-parse", V9_PRELAUNCH_SHORT)
    ancestor = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                               prelaunch, "HEAD"], capture_output=True)
    if ancestor.returncode != 0:
        raise V10Error("the V9 prelaunch commit is not an ancestor of the current head")
    rel_prefix = "runs/comm_bc_initialized_constrained_ppo_20261007_r2"
    prelaunch_binding = {"prelaunch_commit": prelaunch}
    for name, rel in (
        ("gate", "v9_eval_runtime_close/ROUND46_R2_V9_PRELAUNCH_GATE.json"),
        ("frozen_config", "v9_eval_runtime_close/ROUND46_R2_V9_FROZEN_CONFIG.json"),
        ("quota_plan", "v9_eval_runtime_close/ROUND46_R2_V9_QUOTA_PLAN.json"),
        ("adapter", "r46_v9_eval_runtime_close.py"),
    ):
        blob = subprocess.run(
            ["git", "-C", str(repo), "show", f"{prelaunch}:{rel_prefix}/{rel}"],
            check=True, capture_output=True).stdout
        disk = (HERE / rel).read_bytes()
        if blob != disk:
            raise V10Error(f"V9 prelaunch object changed after freeze: {name}")
        prelaunch_binding[f"{name}_blob_sha256"] = sha256_bytes(blob)
    gate = read_json(V9.V9_ROOT / "ROUND46_R2_V9_PRELAUNCH_GATE.json")
    if (prelaunch_binding["adapter_blob_sha256"] != gate.get("adapter_source_sha256")
            or gate.get("zero_environment_regression") != "passed"):
        raise V10Error("the V9 prelaunch gate does not bind its frozen code identity")

    claim_path = HERE / CELL_1301002_REL["claim"]
    row_path = HERE / CELL_1301002_REL["row"]
    artifact_paths = {name: HERE / CELL_1301002_REL[name]
                      for name in ("lifecycle", "transcript", "tape")}
    slot_path = HERE / CELL_1301002_REL["slot"]
    actual = {"claim_sha256": sha256_file(claim_path), "row_sha256": sha256_file(row_path)}
    actual.update({f"{name}_sha256": sha256_file(p) for name, p in artifact_paths.items()})
    actual["slot_record_sha256"] = sha256_file(slot_path)
    actual["v9_stop_state_sha256"] = sha256_file(
        V9.V9_ROOT / "ROUND46_R2_V9_STOP_STATE.json")
    for field, expected in CELL_1301002_IDENTITY.items():
        if actual.get(field) != expected:
            raise V10Error(f"1301002 preserved bytes differ from the registered identity: {field}")
    stop = read_json(V9.V9_ROOT / "ROUND46_R2_V9_STOP_STATE.json")
    preserved = stop["preserved_artifacts"]
    for name in ("claim", "row", "lifecycle", "transcript", "tape"):
        if preserved[name]["sha256"] != actual[f"{name}_sha256"]:
            raise V10Error(f"the V9 stop-state registration disagrees with disk: {name}")

    claim = read_json(claim_path)
    row = read_json(row_path)
    slot = read_json(slot_path)
    if (slot.get("category"), slot.get("slot"), slot.get("attempt"),
            slot.get("logical_job")) != ("normal", 388, 1, "model_eval_new_s11_greedy_scene1301002"):
        raise V10Error("the 1301002 slot record does not match normal slot 388")
    art = row.get("artefacts", {})
    for name, path in {"row": row_path, **artifact_paths}.items():
        if Path(art.get(name, "")).resolve() != path.resolve():
            raise V10Error(f"1301002 row artifact path mismatch: {name}")
    if (claim.get("logical_job") != "model_eval_new_s11_greedy_scene1301002"
            or claim.get("status") != "failed_technical"
            or claim.get("category") != "normal" or claim.get("attempt") != 1
            or claim.get("launch_slot") != 388 or claim.get("prelaunch_head") != prelaunch
            or claim.get("environment_constructor_calls") != 1
            or claim.get("run_return_calls") != 1 or claim.get("completed_steps") != 1000
            or claim.get("run_failure_calls") != 0 or claim.get("optimizer_step_calls") != 0
            or claim.get("checkpoint_write_calls") != 0):
        raise V10Error("the 1301002 claim does not match its documented derived-snapshot failure")
    if claim.get("round45_artifact_receipt") is None or claim.get("round44_eval_artifact_receipt") is None:
        raise V10Error("the 1301002 claim is missing its persisted booking receipts")
    if sha256_file(base.CHECKPOINT_JOURNAL_PATH) != V9.JOURNAL_SHA:
        raise V10Error("the v6 checkpoint journal changed since the 1301002 execution")

    task_events = row.get("task_reward_events")
    if not isinstance(task_events, list):
        raise V10Error("1301002 task completion events are missing")
    credited = [x for x in task_events if x.get("credited") is True and x.get("on_time") is True]
    cost_stream = row.get("cost_stream")
    if not isinstance(cost_stream, dict) or len(cost_stream) != 1000:
        raise V10Error("1301002 1000-tick cost stream is incomplete")
    tick_airtime = sum(int(x.get("airtime_bytes", -1)) for x in cost_stream.values())
    transcript = read_jsonl(artifact_paths["transcript"])
    lifecycle = read_jsonl(artifact_paths["lifecycle"])
    tape = read_jsonl(artifact_paths["tape"])
    if (row.get("model_arm"), row.get("training_seed"), row.get("eval_mode"),
            row.get("scenario_seed"), row.get("attempt"), row.get("steps")) != (
            "NEW", 11, "greedy", 1_301_002, 1, 1000):
        raise V10Error("1301002 row is not the exact fixed evaluation cell")
    closure = validate_episode_closure(row, lifecycle, transcript, tape,
                                       placed_target_ids=PLACED_TARGET_IDS)
    if (row.get("optimizer_steps") != 0
            or row.get("transport_parameters", {}).get("ack_enabled") is not False
            or row.get("ack_ledger", {}).get("ack_attempts") != 0):
        raise V10Error("1301002 communication/action identity differs from one-shot evaluation")
    model = row.get("checkpoint_load_identity", {})
    if (model.get("file_sha256") != V7.MODEL_FILE_SHA
            or model.get("parameter_identity", {}).get("parameter_sha256") != V7.MODEL_PARAMETER_SHA
            or model.get("episodes_done") != 128 or model.get("updates") != 32
            or model.get("load_mode") != "read_only"
            or model.get("optimizer_restored") is not False
            or model.get("rng_restored") is not False):
        raise V10Error("1301002 recorded checkpoint identity is not the fixed read-only ep128")
    sig = V7._recursive_find(lifecycle, "run_signature")
    if (not isinstance(sig, dict) or sig.get("code_commit") != V7.OLD_INPUT_HEAD
            or sig.get("config_hash") != row.get("config_hash")):
        raise V10Error("1301002 actual run signature lacks the frozen input/config identity")
    if sig.get("checkpoints") != {"piloc": "d57aed25b3d46dffb2bd269ea87140953828ba7f51a960bb01995ec054dfca33",
                                  "ctde": "c7c256da0fc20451c1104996ce494148f567e4f83acf265d9baae8f0f053500d"}:
        raise V10Error("1301002 run signature checkpoint identity differs from the frozen search/assignment models")
    transport = row.get("transport_parameters", {})
    config_ok = (
        row.get("reward_timing_version") == "published-completion-event-at-relay-tick/2"
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
    )
    if not config_ok:
        raise V10Error("1301002 row does not close against the frozen world/action configuration")
    if row.get("spent_bytes") != tick_airtime:
        raise V10Error("1301002 cost stream does not close against the row bytes")

    def _configuration_of(records: list[dict[str, Any]]) -> dict[str, Any]:
        for record in records:
            detail = record.get("detail") if isinstance(record, dict) else None
            config = detail.get("configuration") if isinstance(detail, dict) else None
            if isinstance(config, dict) and "seed" in config:
                return config
        raise V10Error("lifecycle run_config record is missing")

    def _config_hash_recomputed(configuration: dict[str, Any]) -> str:
        payload = {k: v for k, v in configuration.items()
                   if k not in {"output", "decision_trace", "reservation_log"}}
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    cfg = _configuration_of(lifecycle)
    if _config_hash_recomputed(cfg) != row.get("config_hash"):
        raise V10Error("1301002 config_hash does not close against its own full configuration")
    cfg_1301001 = _configuration_of(read_jsonl(
        V8.V8_ROOT / "logs/v2_r44_eval_new_s11_greedy_s1301001_a1_rp.jsonl"))
    differing = sorted(k for k in set(cfg) | set(cfg_1301001)
                       if cfg.get(k) != cfg_1301001.get(k))
    if differing != ["comm_transcript", "output", "seed"]:
        raise V10Error(f"1301002 run configuration differs beyond the per-scene seed "
                       f"and the two per-run invocation paths: {differing}")

    rng_python_before = random.getstate()
    rng_numpy_before = np.random.get_state()
    rng_torch_before = torch.get_rng_state().clone()
    run = runtime["r42_run"]
    controller, model_identity = run.load_eval_model("NEW", 11)
    one_shot = runtime.get("r42_one_shot") or sys.modules.get("r42_one_shot")
    if one_shot is None:
        raise V10Error("actual one-shot collector module is unavailable")
    forward = verify_greedy_tape(controller, tape, one_shot)
    del controller
    if (model_identity.get("training_seed") != 11
            or model_identity.get("episodes_done") != 128
            or model_identity.get("updates") != 32
            or model_identity.get("load_mode") != "read_only"
            or model_identity.get("file_sha256") != V7.MODEL_FILE_SHA
            or model_identity.get("parameter_identity", {}).get("parameter_sha256")
            != V7.MODEL_PARAMETER_SHA):
        raise V10Error("actual read-only s11 ep128 model identity does not match V6")
    if (random.getstate() != rng_python_before
            or np.random.get_state()[0] != rng_numpy_before[0]
            or not np.array_equal(np.random.get_state()[1], rng_numpy_before[1])
            or not torch.equal(torch.get_rng_state(), rng_torch_before)):
        raise V10Error("the read-only forward audit disturbed evaluator RNG state")

    decision = {
        "kind": "round46-r2-v10-cell1301002-adoption-decision/1",
        "task_id": TASK_ID,
        "status": "accepted",
        "key": [11, "greedy", 1_301_002],
        "object": "V9 normal slot388 attempt1 (completed episode, zero new claim/environment)",
        "original_booking_failure_stands": True,
        "original_claim_status_unchanged": claim.get("status"),
        "booking_receipts_persisted_before_snapshot_failure": True,
        "performance_used_for_adoption": False,
        "prelaunch_binding": prelaunch_binding,
        "identity": {k: actual.get(k, CELL_1301002_IDENTITY.get(k))
                     for k in CELL_1301002_IDENTITY},
        "model_identity_status": "passed",
        "real_greedy_forward_status": forward.get("status"),
        "episode_and_event_closure": closure["status"],
        "cost_closure_status": closure["cost_closure"],
        "science_configuration_status": "passed",
        "journal_sha256": V9.JOURNAL_SHA,
        "descriptive_only": {
            "on_time": int(row.get("on_time")),
            "spent_bytes": int(row.get("spent_bytes")),
            "note": "descriptive only, never an adoption gate",
        },
    }
    detail = {
        "claim_path": str(claim_path), "row_path": str(row_path),
        "artifact_paths": {k: str(v) for k, v in artifact_paths.items()},
        "lifecycle_records": len(lifecycle), "transcript_records": len(transcript),
        "tape_records": len(tape),
        "completion_closure": closure,
        "greedy_forward": forward,
        "credited_task_ids": sorted(str(x.get("task_id")) for x in credited),
        "model_identity": {k: model_identity.get(k) for k in (
            "training_seed", "episodes_done", "updates", "load_mode",
            "optimizer_restored", "rng_restored", "file_sha256")},
    }
    return decision, detail


def write_cell1301002_adoption(decision: dict[str, Any], detail: dict[str, Any]) -> str:
    receipt = {"kind": "round46-r2-v10-cell1301002-adoption-receipt/1",
               **decision, "evidence": detail}
    write_new_json(V10_ROOT / "receipts/ROUND46_R2_V10_CELL1301002_ADOPTION_RECEIPT.json", receipt)
    adopted_row = read_json(HERE / CELL_1301002_REL["row"])
    adopted_row["v10_technical_adoption"] = {
        "task_id": TASK_ID, "code_version": CODE_VERSION,
        "original_claim_status": "failed_technical",
        "original_booking_failure_stands": True,
        "adoption_receipt": "receipts/ROUND46_R2_V10_CELL1301002_ADOPTION_RECEIPT.json",
        "performance_used_for_adoption": False,
    }
    write_json_atomic(V10_ROOT / "adoption/cell1301002_effective_row.json", adopted_row)
    return sha256_file(V10_ROOT / "adoption/cell1301002_effective_row.json")


# --------------------------------------------------------------------------
# frozen execution identity
# --------------------------------------------------------------------------

PROTECTED = V7.PROTECTED


def _verify_v10_freeze(base: Any, _runtime: dict[str, Any], *,
                       require_gate: bool = True) -> None:
    freeze_path = V10_ROOT / "ROUND46_R2_V10_FROZEN_CONFIG.json"
    gate_path = V10_ROOT / "ROUND46_R2_V10_PRELAUNCH_GATE.json"
    if not freeze_path.is_file() or not gate_path.is_file():
        raise V10Error("V10 prelaunch freeze/gate is missing")
    freeze, gate = read_json(freeze_path), read_json(gate_path)
    if (freeze.get("task_id") != TASK_ID or gate.get("task_id") != TASK_ID
            or freeze.get("status") != "frozen_before_environment"
            or gate.get("status") != "passed"):
        raise V10Error("V10 freeze/gate identity is not active")
    if require_gate and gate.get("zero_environment_regression") != "passed":
        raise V10Error("V10 zero-environment regression is not registered as passed")
    if _git(base.REPO, "branch", "--show-current") != BRANCH:
        raise V10Error("V10 runtime is on the wrong branch")
    if (sha256_file(freeze_path) != gate.get("frozen_config_sha256")
            or sha256_file(V10_ROOT / "ROUND46_R2_V10_QUOTA_PLAN.json")
            != gate.get("quota_plan_sha256")
            or sha256_file(Path(__file__)) != gate.get("adapter_source_sha256")):
        raise V10Error("V10 frozen code or quota map changed after prelaunch")
    source_commit = freeze.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise V10Error("V10 source commit identity is missing")
    ancestor = subprocess.run(["git", "-C", str(base.REPO), "merge-base", "--is-ancestor",
                               source_commit, "HEAD"], capture_output=True)
    if ancestor.returncode != 0:
        raise V10Error("V10 source commit is not an ancestor of the current execution head")
    source_sha = gate.get("source_hashes", {}).get(Path(__file__).name)
    if source_sha != sha256_file(Path(__file__)):
        raise V10Error("V10 adapter source hash differs from the prelaunch gate")
    status = subprocess.run(["git", "-C", str(base.REPO), "status", "--porcelain"],
                            check=True, capture_output=True, text=True).stdout.rstrip("\n")
    allowed = {" M " + path for path in PROTECTED}
    lines = set(status.splitlines())
    if lines != allowed:
        raise V10Error(f"worktree status changed outside the two preserved ROUND46 files: {sorted(lines)}")
    for relative, digest in PROTECTED.items():
        if sha256_file(base.REPO / relative) != digest:
            raise V10Error(f"protected prior-session file changed: {relative}")
    for path, digest in freeze.get("checkpoint_files", {}).items():
        if sha256_file(Path(path)) != digest:
            raise V10Error("a frozen ep128 checkpoint changed")
    quota = read_json(V10_ROOT / "ROUND46_R2_V10_QUOTA_PLAN.json")
    if (quota.get("branch") != freeze.get("quota_branch")
            or quota.get("new_claim_count") != freeze.get("new_claim_count")
            or sha256_bytes(canonical_json_bytes(quota)) != gate.get("quota_plan_canonical_sha256")):
        raise V10Error("V10 claim map differs from its immutable prelaunch identity")
    if sha256_file(base.CHECKPOINT_JOURNAL_PATH) != gate.get("checkpoint_journal_sha256"):
        raise V10Error("checkpoint journal changed or V10 evaluation attempted to rebind it")


def run_fixed_matrix(runtime: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = runtime["v10_runner"]._run_final_eval_matrix(runtime)
    used = runtime["v10_adoption_wrapper"]["used"]()
    if set(used) != set(ADOPTED_KEYS):
        raise V10Error(f"adopted cells were not consumed exactly once: {used}")
    if len(rows) != 96:
        raise V10Error(f"fixed matrix returned {len(rows)}/96 effective cells")
    return rows


def validate_full_matrix(rows: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    expected = set(fixed_eval_keys())
    by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["training_seed"]), str(row["eval_mode"]), int(row["scenario_seed"]))
        if key in by_key:
            raise V10Error(f"duplicate effective evaluation row: {key}")
        by_key[key] = row
    if set(by_key) != expected or len(by_key) != 96:
        raise V10Error(f"effective matrix is not the fixed 96 keys: {len(by_key)}/96")
    adopted_sources = {
        ADOPTED_KEYS[0]: V8.V8_ROOT / "adoption/attempt2_effective_row.json",
        ADOPTED_KEYS[1]: V9.V9_ROOT / "adoption/cell1301001_effective_row.json",
        ADOPTED_KEYS[2]: V10_ROOT / "adoption/cell1301002_effective_row.json",
    }
    for key, source in adopted_sources.items():
        if by_key[key] != read_json(source):
            raise V10Error(f"adopted row differs from its registered effective row: {key}")
    for key, row in by_key.items():
        if (row.get("model_arm") != "NEW" or row.get("steps") != 1000
                or row.get("training_version") is None
                or row.get("comm_action_semantics") != "one-shot-send-skip/1"
                or row.get("optimizer_steps") != 0):
            raise V10Error(f"evaluation row identity or eval-only invariant failed: {key}")
    slots = {(str(x["logical_job"]), int(x["attempt"])): x for x in plan["entries"]}
    claims = {}
    for path in V10_CLAIMS.glob("*/*.json"):
        claim = read_json(path)
        key = (str(claim.get("logical_job")), int(claim.get("attempt", -1)))
        if key in claims:
            raise V10Error("duplicate V10 claim file")
        claims[key] = (path, claim)
    if set(claims) != set(slots):
        raise V10Error(f"V10 claim schedule mismatch: {len(claims)}/{len(slots)}")
    for key, (path, claim) in claims.items():
        if claim.get("status") != "complete" or claim.get("completed_steps") != 1000:
            raise V10Error(f"V10 planned claim is not a completed 1000-step eval: {key}")
        expected_slot = slots[key]
        if (claim.get("category") != expected_slot["category"]
                or int(claim.get("launch_slot", -1)) != expected_slot["slot"]):
            raise V10Error(f"V10 claim slot identity mismatch: {key}")
    return {"status": "passed", "effective_rows": len(by_key),
            "new_claims": len(claims),
            "adopted_cells": [list(k) for k in ADOPTED_KEYS],
            "row_keys_sha256": sha256_bytes(canonical_json_bytes(
                sorted([list(x) for x in expected])))}


def _analysis_union_rows() -> tuple[Path, dict[str, Any]]:
    import shutil
    source_dir = V7.V6_ROOT / "rows"
    target_dir = V10_ROOT / "analysis_inputs" / "rows"
    target_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted(source_dir.glob("*.json")):
        row = read_json(source)
        if row.get("training_arm") != "NEW" or not row.get("training_version"):
            continue
        digest = sha256_file(source)
        dest = target_dir / ("reused_" + source.name)
        if dest.exists() and sha256_file(dest) != digest:
            raise V10Error("analysis input staging row conflicts with the reused source")
        if not dest.exists():
            shutil.copyfile(source, dest)
        records.append({"role": "reused_training_row", "source": str(source),
                        "source_sha256": digest, "staged": str(dest)})
    if len(records) != 384:
        raise V10Error(f"analysis reuse found {len(records)}/384 training rows")
    adopted_sources = [
        (V8.V8_ROOT / "adoption/attempt2_effective_row.json", "adopted_a2_"),
        (V9.V9_ROOT / "adoption/cell1301001_effective_row.json", "adopted_c1_"),
        (V10_ROOT / "adoption/cell1301002_effective_row.json", "adopted_c2_"),
    ]
    for source, prefix in adopted_sources + [(p, "eval_") for p in sorted(V10_ROWS.glob("*.json"))]:
        digest = sha256_file(source)
        dest = target_dir / (prefix + source.name)
        if dest.exists() and sha256_file(dest) != digest:
            raise V10Error("analysis input staging eval row conflicts with the V10 source")
        if not dest.exists():
            shutil.copyfile(source, dest)
        records.append({"role": "new_eval_row", "source": str(source),
                        "source_sha256": digest, "staged": str(dest)})
    if len(records) != 480:
        raise V10Error(f"analysis input union contains {len(records)}/480 rows")
    manifest = {"kind": "round46-r2-v10-analysis-input-manifest/1",
                "reused_training_rows": 384, "adopted_rows": 3, "new_eval_rows": 93,
                "rows": records}
    return target_dir, manifest


def analyze_and_report(base: Any, runtime: dict[str, Any], plan: dict[str, Any],
                       rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix_gate = validate_full_matrix(rows, plan)
    union, input_manifest = _analysis_union_rows()
    input_manifest_sha = write_json_atomic(
        V10_ROOT / "ROUND46_R2_V10_ANALYSIS_INPUT_MANIFEST.json", input_manifest)
    previous_rows = runtime["common"].ROWS
    runtime["common"].ROWS = union
    base.REPORT_PATH = V10_ROOT / "ROUND46_R2_V10_REPORT.md"
    base.FINAL_RESULTS_PATH = V10_ROOT / "ROUND46_R2_V10_FINAL_RESULTS.json"
    base.PAIRED_RESULTS_PATH = V10_ROOT / "ROUND46_R2_V10_PAIRED_RESULTS.json"
    base.TARGET_DIFFS_PATH = V10_ROOT / "ROUND46_R2_V10_TARGET_ID_DIFFS.json"
    base.ARTIFACT_MANIFEST_PATH = V10_ROOT / "ROUND46_R2_V10_ARTIFACT_MANIFEST.json"
    base.CHECKPOINT_INDEX_PATH = V10_ROOT / "ROUND46_R2_V10_CHECKPOINT_INDEX.json"
    base.BOOTSTRAP_SEED, base.BOOTSTRAP_REPLICATES = 20261007, 10_000
    try:
        results, detail = base._build_analysis(runtime["v10_runner"], runtime)
    finally:
        runtime["common"].ROWS = previous_rows
    ledger = build_cumulative_ledger(plan, strict=True)
    ledger_sha = write_json_atomic(V10_LEDGER, ledger)
    results["counts"] = ledger["counts"]
    results["budget"] = {
        "normal_planned": 480, "normal_used": ledger["normal_used"],
        "reserve_used": ledger["reserve_used"], "total_used": ledger["total_used"],
        "hard_cap": V7.HARD_CAP, "remaining_hard_cap": V7.HARD_CAP - ledger["total_used"],
    }
    write_json_atomic(base.FINAL_RESULTS_PATH, results)
    write_report(results, matrix_gate, plan, ledger, input_manifest_sha, ledger_sha)
    verdict = {
        "kind": "round46-r2-v10-machine-verdict/1", "task_id": TASK_ID,
        "status": results["status"],
        "development_go": bool(results["interpretation"]["development_block_joint_improvement"]),
        "seed_results": results["seed_results"],
        "effective_matrix": matrix_gate,
        "adopted_cells": [list(k) for k in ADOPTED_KEYS],
        "quota_branch": plan["branch"], "budget": results["budget"],
        "new_training_episodes": 0, "new_ppo_updates": 0,
        "new_optimizer_steps": 0, "new_checkpoint_writes": 0,
        "independent_confirmation": False, "generalization_claim": False,
    }
    verdict_sha = write_json_atomic(V10_ROOT / "ROUND46_R2_V10_MACHINE_VERDICT.json", verdict)
    return {"results": results, "detail": detail, "ledger": ledger,
            "verdict": verdict, "verdict_sha256": verdict_sha,
            "analysis_manifest_sha256": input_manifest_sha,
            "matrix_gate": matrix_gate}


def write_report(results: dict[str, Any], matrix_gate: dict[str, Any],
                 plan: dict[str, Any], ledger: dict[str, Any],
                 input_manifest_sha: str, ledger_sha: str) -> None:
    lines = [
        "# ROUND46-R2 V10 报告：固定 ep128 96 格开发评估闭合",
        "",
        f"任务：`{TASK_ID}`。本次只做固定模型评估，不恢复训练状态。",
        f"计划分支：`{plan['branch']}`；有效格：{matrix_gate['effective_rows']}/96。",
        "三格技术接纳（attempt2、1301001、1301002）原 booking 技术失败仍成立；科学结果经独立技术核验后接纳，旧 claim 状态未改。",
        f"额度累计：normal {ledger['normal_used']}/480，reserve {ledger['reserve_used']}/24，总计 {ledger['total_used']}/504。",
        f"累计实际环境返回步数：{ledger['counts']['completed_steps']}；本次新 PPO/Adam/checkpoint 写入均为0。",
        "",
        "## 科学判定",
        "",
        "开发块联合门禁：只有三个 seed 各自 greedy 的逐目标按时集合逐场景等于 immediate，且每个 seed 的平均计费严格低于 4,838.25 B/集，才判 development_go。",
        "开发场景已用于方案选择；结果不构成独立确认、正式非劣、总体零损失或泛化证据。",
        "",
        "| Seed | Greedy 按时任务均值 | Greedy 字节均值 | 与 immediate 目标集合逐场景一致 | 低于 late 成本 | Sampled 任务均值 | Sampled 字节均值 |",
        "|---:|---:|---:|---|---|---:|---:|",
    ]
    for seed in V7.SEEDS:
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
        "", "## 原件与增量", "",
        "旧 386 normal claim、三次 booking 失败记录与三份接纳 receipt 均保留；V10 claims/slots 为新原子增量。",
        f"分析输入清单 SHA256：`{input_manifest_sha}`。累计账本 SHA256：`{ledger_sha}`。",
        "原始权重、tape、日志和分析输入副本留在 Git 外；本轮不推送、不合并、不部署。",
    ]
    path = V10_ROOT / "ROUND46_R2_V10_REPORT.md"
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def main() -> None:
    raise SystemExit("V10 execution is gated: use the committed prelaunch freeze and run command.")


if __name__ == "__main__":
    main()
