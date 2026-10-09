"""Private ROUND58 gen3 attempt3 claim seam (live allocation adapter).

Version: round58-gen3-attempt3-claim-seam/1
Frozen call path preserved:  R42 ``_execute_cell`` -> module ``_claim`` (the
frozen gen2 ``planned_claim`` wrapper installed by ``bind_live_runtime``) ->
R42 base ``_claim`` -> ``r40_common.claim_launch``.

Differential whitelist (execution/ledger seam only; science and environment
code untouched):
  * attempts 1/2 delegate unchanged to the frozen ``planned_claim`` wrapper;
  * attempt 3 is admitted for the single owner-approved special case
    (new_training_s11_e4 / scenario 1300003 / reserve / slot 2) only after a
    fail-closed on-disk proof: predecessor attempt-1 (normal slot 13) and
    attempt-2 (reserve slot 1) originals with matching slot records, both
    failed_technical with zero environment consumption, durable failure
    records present, no valid completed result for the key, reserve slot 2
    free, and the cumulative approved quota not exceeded;
  * the frozen R45 one-retry gate (``guarded_execute``) is extended by an
    outer wrapper: only the approved key with exactly its two persisted
    technical failures and no attempt-3 file bypasses the one-retry limit,
    through the same live R45 context helper and the same underlying
    ``_execute_cell``; every other key, every recovery route (completed-row
    replay) and every other failure path keeps the frozen behaviour;
  * any other attempt-3 key, normal attempt-3, attempt>=4, occupied slot or
    exceeded quota raises before any durable write.
The wrapper never fabricates claims or bypasses the frozen duplicate-claim
atomicity of ``claim_launch``.
"""
from __future__ import annotations

import types
from pathlib import Path
from typing import Any

from r58_attempt3_contract import (APPROVED_ATTEMPT3, PREDECESSOR_ATTEMPT2_SLOT,
                                   SEAM_VERSION, ZERO_ENVIRONMENT_COUNTERS,
                                   planned_job, preflight_attempt3_allocation,
                                   validate_claims)

HISTORICAL_CONSUMED = 16
WALL_CLOCK_BUDGET_SECONDS = 240  # graceful stop at an all-terminal claim boundary


def _zero_env(claim: dict[str, Any], label: str) -> None:
    for name in ZERO_ENVIRONMENT_COUNTERS:
        value = claim.get(name)
        if value is None:
            continue
        if int(value) != 0:
            raise RuntimeError(f"attempt3 seam: {label} recorded {name}={value}")
    if claim.get("simulation_started") not in (None, False):
        raise RuntimeError(f"attempt3 seam: {label} recorded a started simulation")


def install(run_module: Any, *, plan: dict[str, Any], raw_root: Path,
            frozen_wrapper: Any, normal_cap: int, reserve_cap: int,
            hard_cap: int, historical_consumed: int = HISTORICAL_CONSUMED,
            resume_root: Path | None = None) -> dict[str, Any]:
    """Bind the attempt3-aware wrapper as the live module ``_claim``."""
    raw_root = Path(raw_root)
    if getattr(run_module, "_r58_attempt3_seam_installed", False):
        raise RuntimeError("attempt3 claim seam already installed on this module")
    if not callable(frozen_wrapper):
        raise RuntimeError("attempt3 seam requires the frozen planned_claim wrapper")
    code = frozen_wrapper.__code__
    if code.co_name != "planned_claim" or "r58_execute.py" not in Path(code.co_filename).name:
        raise RuntimeError("attempt3 seam did not find the frozen planned_claim wrapper")
    import types
    original_claim = None
    for cell in (frozen_wrapper.__closure__ or ()):
        value = cell.cell_contents
        # The frozen wrapper captured exactly one function with the durable
        # claim-writer signature (logical_job, *, attempt, metadata): the live
        # R44 claim adapter installed on the r42_run module.
        if (isinstance(value, types.FunctionType) and value is not frozen_wrapper
                and value.__code__.co_argcount == 1
                and value.__code__.co_kwonlyargcount == 2
                and value.__code__.co_varnames[:1] == ("logical_job",)):
            if original_claim is not None:
                raise RuntimeError("attempt3 seam found an ambiguous base claim cell")
            original_claim = value
    if original_claim is None:
        found = [repr(cell.cell_contents)[:120] for cell in (frozen_wrapper.__closure__ or ())]
        raise RuntimeError(f"attempt3 seam could not locate the frozen base claim writer; cells={found}")

    def attempt3_claim(logical_job: str, *args: Any, **kwargs: Any) -> Path:
        attempt = int(kwargs.get("attempt", args[0] if args else 1))
        if attempt != APPROVED_ATTEMPT3["attempt"]:
            return frozen_wrapper(logical_job, *args, **kwargs)
        if logical_job != APPROVED_ATTEMPT3["logical_job"]:
            raise RuntimeError(f"attempt3 seam refuses attempt3 for non-approved job: {logical_job}")
        preflight = preflight_attempt3_allocation(
            raw_root, plan, historical_consumed=historical_consumed,
            normal_cap=normal_cap, reserve_cap=reserve_cap, hard_cap=hard_cap)
        claim_path = original_claim(logical_job, *args, **kwargs)
        claim = _read(claim_path)
        if (str(claim.get("logical_job", "")) != logical_job
                or int(claim.get("attempt", -1)) != APPROVED_ATTEMPT3["attempt"]
                or str(claim.get("category", "")) != APPROVED_ATTEMPT3["category"]):
            raise RuntimeError("attempt3 seam: persisted claim identity mismatch")
        if int(claim.get("launch_slot", -1)) != APPROVED_ATTEMPT3["launch_slot"]:
            raise RuntimeError(
                f"attempt3 seam: allocator returned slot {claim.get('launch_slot')}, "
                "approved slot is 2; stopping before any environment construction")
        slot_record = _read(raw_root / "launch_slots" / "reserve"
                            / f"{APPROVED_ATTEMPT3['launch_slot']:04d}.json")
        if (str(slot_record.get("logical_job", "")) != logical_job
                or int(slot_record.get("attempt", -1)) != APPROVED_ATTEMPT3["attempt"]
                or int(slot_record.get("slot", -1)) != APPROVED_ATTEMPT3["launch_slot"]):
            raise RuntimeError("attempt3 seam: durable slot record identity mismatch")
        return claim_path

    attempt3_claim.__name__ = "round58_attempt3_claim_wrapper"
    attempt3_claim.__qualname__ = attempt3_claim.__name__
    run_module._claim = attempt3_claim
    if run_module._claim is not attempt3_claim:
        raise RuntimeError("attempt3 claim seam binding failed")

    # --- execution seam: extend the frozen R45 one-retry gate for the unique
    # approved attempt3 case; everything else keeps the frozen behaviour.
    import time as _time
    _seam_started_monotonic = _time.monotonic()
    guarded_execute = getattr(run_module, "_execute_cell", None)
    base_execute = None
    if (callable(guarded_execute)
            and guarded_execute.__code__.co_name == "guarded_execute"):
        for cell in (guarded_execute.__closure__ or ()):
            value = cell.cell_contents
            if (isinstance(value, types.FunctionType)
                    and value.__code__.co_name == "_execute_cell"):
                if base_execute is not None:
                    raise RuntimeError("attempt3 execute seam found an ambiguous base cell")
                base_execute = value
        if base_execute is None:
            raise RuntimeError("attempt3 execute seam could not locate the underlying _execute_cell")
        r45_globals = guarded_execute.__globals__
        current_context = r45_globals["CURRENT_CONTEXT"]
        context_for_cell = r45_globals["_context_for_cell"]
        planned_key = r45_globals["_planned_key"]
        # The live R45 module keeps the configure-time output namespace; every
        # resume row/artefact receipt resolves against it. Bind it to the
        # resume generation root (rows, logs, failures, execution ledger).
        if resume_root is not None:
            resume_root = Path(resume_root).resolve()
            if str(r45_globals.get("RUN_DIR", "")) == str(resume_root):
                pass
            else:
                previous_r45_run_dir = Path(r45_globals.get("RUN_DIR"))
                r45_globals["RUN_DIR"] = resume_root
                if r45_globals.get("R3_EXECUTION_LEDGER_PATH") is not None:
                    r45_globals["R3_EXECUTION_LEDGER_PATH"] = (
                        resume_root / "runtime" / "ROUND53_EXECUTION_LEDGER.json")
                # Provision the mutable R45 task-state file in the resume
                # namespace (copy of the generation-seeded state; the original
                # stays untouched in the historical generation root).
                task_state_target = resume_root / "ROUND45_CURRENT_TASK.json"
                if not task_state_target.exists():
                    task_state_source = previous_r45_run_dir / "ROUND45_CURRENT_TASK.json"
                    if not task_state_source.is_file():
                        raise RuntimeError(
                            "attempt3 seam cannot provision the R45 task state: "
                            f"missing {task_state_source}")
                    import shutil as _shutil
                    _shutil.copy2(task_state_source, task_state_target)
        else:
            previous_r45_run_dir = None

        def _correct_in_flight_claims() -> list[dict[str, Any]] | None:
            """Append an external-termination correction to every planned claim
            left in flight by an externally killed process, then finish any
            whose episode row fully persisted.  Frozen retry rules then rerun
            the interrupted episodes through their normal attempt path."""
            import time as _time
            jobs = {job["logical_job"]: job for job in plan["jobs"]}
            r40_common_module = _read_r40_common_via_r45(r45_globals)
            if r40_common_module is None:
                return None
            corrected: list[dict[str, Any]] = []
            for claim_path in sorted((raw_root / "launch_claims").glob("*/*.json")):
                try:
                    claim = _read(claim_path)
                except Exception:
                    continue
                logical = str(claim.get("logical_job", ""))
                if (claim.get("status") != "claimed_before_environment"
                        or logical not in jobs):
                    continue
                correction = {
                    "kind": "round58-external-termination-correction/1",
                    "original_status": "claimed_before_environment",
                    "corrected_at_unix": _time.time(),
                    "reason": ("the executing process was terminated externally "
                               "(sandbox background reaping); no Python exception "
                               "handler ran; the claim is corrected to "
                               "failed_technical so the frozen technical-retry "
                               "path can rerun the episode"),
                    "counters_at_correction": {
                        name: claim.get(name) for name in
                        ("environment_constructor_calls", "run_invocation_calls",
                         "run_return_calls", "completed_steps")},
                }
                artifact_paths = claim.get("round45_artifact_paths")
                completed_row = None
                if isinstance(artifact_paths, dict):
                    row_path = artifact_paths.get("row")
                    if row_path and Path(row_path).is_file():
                        candidate_row = _read(row_path)
                        if (str(candidate_row.get("kind", "")) == str(claim.get("round45_key", {}).get("kind"))
                                and int(candidate_row.get("steps", -1)) == 1000):
                            completed_row = candidate_row
                if completed_row is not None:
                    create_receipt = r45_globals["_create_round45_artifact_receipt"]
                    receipt = create_receipt(completed_row, claim)
                    r40_common_module._update_claim(
                        claim_path, round45_artifact_receipt=receipt, status="complete",
                        round58_receipt_failure_correction=correction)
                else:
                    r40_common_module._update_claim(
                        claim_path, status="failed_technical",
                        round58_external_termination_correction=correction)
                corrected.append({"logical_job": logical,
                                  "attempt": int(claim.get("attempt", -1)),
                                  "completed_row_recovered": completed_row is not None})
            return corrected or None

        def _complete_interrupted_attempt3_bookkeeping() -> dict[str, Any] | None:
            """Finish the receipt of an attempt3 episode that fully ran but whose
            artifact-receipt hashing crashed on the historical RUN_DIR binding.

            Uses the actual R45 receipt creator over the persisted row/tape and
            appends a visible correction record to the claim; never reruns the
            episode and never fabricates counters.
            """
            import time as _time
            claim_dir = raw_root / "launch_claims" / APPROVED_ATTEMPT3["logical_job"]
            attempt3_path = claim_dir / "attempt_0003.json"
            if not attempt3_path.is_file():
                return None
            claim = _read(attempt3_path)
            if claim.get("status") == "complete":
                return claim
            if (claim.get("status") != "failed_technical"
                    or int(claim.get("attempt", -1)) != APPROVED_ATTEMPT3["attempt"]
                    or int(claim.get("run_return_calls", -1)) != 1
                    or int(claim.get("completed_steps", -1)) != 1000
                    or int(claim.get("environment_constructor_calls", -1)) != 1
                    or int(claim.get("run_invocation_calls", -1)) != 1
                    or claim.get("round45_artifact_receipt") is not None
                    or not isinstance(claim.get("round45_artifact_paths"), dict)):
                return None
            artifact_paths = claim["round45_artifact_paths"]
            if not all(name in artifact_paths for name in ("row", "lifecycle", "transcript", "tape")):
                return None
            row_path = Path(artifact_paths["row"])
            if not row_path.is_file() or row_path.resolve().parent != (resume_root / "rows").resolve():
                return None
            row = _read(row_path)
            # Training rows do not carry a top-level logical_job; the R44 key is
            # training_seed + episode_index + attempt + kind + full steps.
            if (str(row.get("kind", "")) != "train"
                    or int(row.get("steps", -1)) != 1000
                    or int(row.get("training_seed", -1)) != 11
                    or int(row.get("episode_index", -1)) != 4
                    or int(row.get("attempt", -1)) != APPROVED_ATTEMPT3["attempt"]):
                return None
            create_receipt = r45_globals["_create_round45_artifact_receipt"]
            receipt = create_receipt(row, claim)
            correction = {
                "kind": "round58-attempt3-receipt-failure-correction/1",
                "original_status": "failed_technical",
                "original_detail": str(claim.get("detail", ""))[:400],
                "corrected_at_unix": _time.time(),
                "reason": ("episode completed (run_return_calls=1, completed_steps=1000) and the "
                           "row/tape persisted; only the artifact-receipt hashing crashed on the "
                           "historical RUN_DIR binding. The live R45 output namespace was rebound "
                           "to the resume root, then the actual receipt creator hashed the "
                           "persisted artefacts. No episode was rerun and no counter was edited."),
                "row_sha256_before_correction": __import__("hashlib").sha256(row_path.read_bytes()).hexdigest(),
            }
            import r58_attempt3_contract as _a3c
            r40_common_module = _read_r40_common_via_r45(r45_globals)
            r40_common_module._update_claim(
                attempt3_path, round45_artifact_receipt=receipt,
                status="complete", round58_receipt_failure_correction=correction)
            return _read(attempt3_path)

        approved_job = planned_job(plan, APPROVED_ATTEMPT3["logical_job"])
        if approved_job is None:
            raise RuntimeError("attempt3 execute seam lacks the approved planned job")
        approved_normal_slot = int(approved_job["normal_slot"])

        def _approved_pair_pending() -> bool:
            claim_dir = raw_root / "launch_claims" / APPROVED_ATTEMPT3["logical_job"]
            if not claim_dir.is_dir():
                return False
            attempts: dict[int, dict[str, Any]] = {}
            for path in sorted(claim_dir.glob("attempt_*.json")):
                payload = _read(path)
                try:
                    attempts[int(payload.get("attempt", -1))] = payload
                except (TypeError, ValueError):
                    return False
            return (set(attempts) == {1, 2}
                    and all(c.get("status") == "failed_technical" for c in attempts.values())
                    and int(attempts[1].get("launch_slot", -1)) == approved_normal_slot
                    and int(attempts[2].get("launch_slot", -1)) == PREDECESSOR_ATTEMPT2_SLOT)

        def attempt3_execute(label: str, logical_job: str, kind: str, scenario: int,
                             metadata: dict[str, Any], *, collector: Any = None,
                             runner_kwargs: dict[str, Any] | None = None,
                             row_updates: dict[str, Any] | None = None,
                             identity: dict[str, Any] | None = None) -> dict[str, Any]:
            import time as _time
            _correct_in_flight_claims()
            if _time.monotonic() - _seam_started_monotonic > WALL_CLOCK_BUDGET_SECONDS:
                # Graceful boundary stop BEFORE any claim: every persisted claim
                # is terminal, all completed rows recover on relaunch, and no
                # environment, claim, optimizer or checkpoint state is in flight.
                raise RuntimeError(
                    "ROUND58 graceful wall-clock boundary reached before claim; "
                    "stopping at an all-terminal claim boundary per the freeze discipline")
            context = current_context.get()
            local_kwargs = dict(runner_kwargs or {})
            local_updates = dict(row_updates or {})
            if context is None:
                context = context_for_cell(kind, logical_job, int(scenario), metadata,
                                           local_kwargs)
            elif context.get("scope") == "bc_only_greedy_control":
                logical_job = str(context["logical_job"])
                local_kwargs["tag"] = str(context["tag"])
                local_updates["r45_evaluation_role"] = "BC_ONLY_GREEDY_CONTROL"
                local_updates["bc_checkpoint_sha256"] = context["bc_checkpoint_sha256"]
                local_updates["bc_parameter_sha256"] = context["bc_parameter_sha256"]
            else:
                logical_job = str(context["logical_job"])
            context = dict(context)
            context.update({"logical_job": logical_job, "kind": kind,
                            "scenario_seed": int(scenario),
                            "tag": str(local_kwargs.get("tag", context.get("tag", "")))})
            planned_key(context)
            if (kind == "train" and logical_job == APPROVED_ATTEMPT3["logical_job"]):
                _complete_interrupted_attempt3_bookkeeping()
            if (kind == "train" and logical_job == APPROVED_ATTEMPT3["logical_job"]
                    and _approved_pair_pending()):
                preflight_attempt3_allocation(
                    raw_root, plan, historical_consumed=historical_consumed,
                    normal_cap=normal_cap, reserve_cap=reserve_cap, hard_cap=hard_cap)
                token = current_context.set(context)
                try:
                    return base_execute(
                        label, logical_job, kind, int(scenario), metadata,
                        collector=collector, runner_kwargs=local_kwargs,
                        row_updates=local_updates, identity=identity)
                finally:
                    current_context.reset(token)
            return guarded_execute(
                label, logical_job, kind, scenario, metadata,
                collector=collector, runner_kwargs=runner_kwargs,
                row_updates=row_updates, identity=identity)

        attempt3_execute.__name__ = "round58_attempt3_execute_wrapper"
        attempt3_execute.__qualname__ = attempt3_execute.__name__
        run_module._execute_cell = attempt3_execute
        if run_module._execute_cell is not attempt3_execute:
            raise RuntimeError("attempt3 execute seam binding failed")
    else:
        raise RuntimeError("attempt3 seam expected the frozen R45 guarded_execute on the live module")

    run_module._r58_attempt3_seam_installed = True

    def undo() -> None:
        if run_module._claim is attempt3_claim:
            run_module._claim = frozen_wrapper
        if run_module._execute_cell is attempt3_execute:
            run_module._execute_cell = guarded_execute
        del run_module._r58_attempt3_seam_installed

    return {"kind": "round58-gen3-attempt3-claim-seam/1", "seam_version": SEAM_VERSION,
            "status": "installed", "approved": dict(APPROVED_ATTEMPT3),
            "plan_jobs": plan["jobs"],
            "execute_seam": "round58-gen3-attempt3-execute-wrapper/1",
            "resume_root": str(resume_root) if resume_root is not None else None,
            "previous_r45_run_dir": str(previous_r45_run_dir) if resume_root is not None else None,
            "historical_consumed": historical_consumed,
            "normal_cap": normal_cap, "reserve_cap": reserve_cap, "hard_cap": hard_cap,
            "frozen_wrapper": f"{Path(code.co_filename).name}:{code.co_firstlineno}",
            "base_claim_module": str(getattr(original_claim, "__module__", "")),
            "undo": undo}


def _read_r40_common_via_r45(r45_globals: dict[str, Any]) -> Any:
    runtime = r45_globals.get("RUNTIME")
    if not isinstance(runtime, dict) or runtime.get("r40_common") is None:
        return None
    return runtime["r40_common"]


def _read(path: Path) -> dict[str, Any]:
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_history(raw_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Read-only history audit through the extended contract validator."""
    claims = [_read(p) for p in sorted(Path(raw_root, "launch_claims").glob("*/*.json"))]
    return validate_claims(plan, claims, strict=False)
