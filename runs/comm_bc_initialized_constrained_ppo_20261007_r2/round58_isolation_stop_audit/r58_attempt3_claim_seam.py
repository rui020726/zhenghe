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
            hard_cap: int, historical_consumed: int = HISTORICAL_CONSUMED) -> dict[str, Any]:
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
            "historical_consumed": historical_consumed,
            "normal_cap": normal_cap, "reserve_cap": reserve_cap, "hard_cap": hard_cap,
            "frozen_wrapper": f"{Path(code.co_filename).name}:{code.co_firstlineno}",
            "base_claim_module": str(getattr(original_claim, "__module__", "")),
            "undo": undo}


def _read(path: Path) -> dict[str, Any]:
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_history(raw_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Read-only history audit through the extended contract validator."""
    claims = [_read(p) for p in sorted(Path(raw_root, "launch_claims").glob("*/*.json"))]
    return validate_claims(plan, claims, strict=False)
