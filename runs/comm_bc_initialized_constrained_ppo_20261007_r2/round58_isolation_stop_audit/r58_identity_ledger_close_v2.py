"""Private, fail-closed identity and derived-ledger helpers for R58 gen3 audit.

This is a new audit/repair identity. It does not alter the canonical trainer and
is not a training/resume entry point.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


IDENTITY_VERSION = "round58-gen3-identity-close/1"


def attested_freeze(base_freeze: dict[str, Any], identity: dict[str, Any],
                    training_version: str) -> dict[str, Any]:
    """Construct the exact freeze metadata written by the repaired serializer."""
    result = deepcopy(base_freeze)
    result["training_version"] = training_version
    result["round58_repair_attestation"] = deepcopy(identity)
    return result


def identity_differences(payload: Any, *, training_version: str,
                         action_semantics: str, base_freeze: dict[str, Any],
                         repair_identity: dict[str, Any]) -> list[dict[str, Any]]:
    """Return field-specific mismatches; never accept a set of legacy versions."""
    if not isinstance(payload, dict):
        return [{"field": "payload", "expected": "mapping", "actual": type(payload).__name__}]
    expected_freeze = attested_freeze(base_freeze, repair_identity, training_version)
    checks = [
        ("training_version", training_version, payload.get("training_version")),
        ("comm_action_semantics", action_semantics, payload.get("comm_action_semantics")),
        ("round58_repair_attestation", repair_identity,
         payload.get("round58_repair_attestation")),
        ("freeze", expected_freeze, payload.get("freeze")),
    ]
    state = payload.get("training_state")
    if not isinstance(state, dict):
        checks.append(("training_state", "mapping", type(state).__name__))
    else:
        checks.extend([
            ("training_state.training_version", training_version,
             state.get("training_version")),
            ("training_state.comm_action_semantics", action_semantics,
             state.get("comm_action_semantics")),
            ("training_state.round58_repair_attestation", repair_identity,
             state.get("round58_repair_attestation")),
        ])
    return [{"field": field, "expected": expected, "actual": actual}
            for field, expected, actual in checks if expected != actual]


def require_identity(payload: Any, **expected: Any) -> None:
    differences = identity_differences(payload, **expected)
    if differences:
        fields = ", ".join(item["field"] for item in differences)
        raise RuntimeError(f"R58 checkpoint identity mismatch: {fields}")


def call_first_batch_gate_with_attested_freeze(gate: Any, seed: int,
                                               runtime: dict[str, Any],
                                               frozen: dict[str, Any], *,
                                               repair_identity: dict[str, Any],
                                               training_version: str) -> Any:
    """Adapt only the expected freeze operand before calling the frozen gate.

    Version, semantics, and all other checks remain in the actual R42 gate. A
    mismatched candidate version is rejected before the gate is called.
    """
    declared = frozen.get("round42", {}).get("training_version")
    if declared != training_version:
        raise RuntimeError("R58 gate adapter refused: frozen training_version mismatch")
    base = runtime.get("round42_freeze")
    if not isinstance(base, dict):
        raise RuntimeError("R58 gate adapter refused: runtime round42_freeze missing")
    expected = attested_freeze(base, repair_identity, training_version)
    local_runtime = dict(runtime)
    local_runtime["round42_freeze"] = expected
    return gate(seed, runtime=local_runtime, frozen=frozen)


def bridge_ppo_update_counter(trainer: Any, run_module: Any) -> dict[str, Any]:
    """Preserve the R42 call-count seam when a private adapter replaces ppo_update.

    The bridge counts an invocation before delegating. It does not synthesize
    optimizer-step events and is installed only on a newly frozen execution identity.
    """
    current = getattr(trainer, "ppo_update", None)
    if not callable(current):
        raise RuntimeError("R58 counter bridge requires callable trainer.ppo_update")
    if getattr(current, "_r58_counter_bridge", False):
        raise RuntimeError("R58 PPO counter bridge is already installed")

    def counted(*args: Any, **kwargs: Any) -> Any:
        run_module._PPO_UPDATE_CALLS = int(run_module._PPO_UPDATE_CALLS) + 1
        return current(*args, **kwargs)

    counted.__name__ = "r58_counted_ppo_update"
    counted._r58_counter_bridge = True
    counted.__wrapped__ = current
    trainer.ppo_update = counted
    return {"kind": "round58-ppo-counter-bridge/1", "status": "installed",
            "wraps_private_adapter": current.__name__,
            "counter_owner": "r42_run._PPO_UPDATE_CALLS"}


def reconcile_claims_slots(claims: Iterable[dict[str, Any]],
                           slots: Iterable[dict[str, Any]], *,
                           expected_normal: int | None = None,
                           expected_reserve: int | None = None,
                           final: bool = False) -> dict[str, Any]:
    """Reconcile unique consumed attempts without rewriting authoritative records."""
    claim_rows = list(claims)
    slot_rows = list(slots)
    claim_keys = [(str(c.get("logical_job")), int(c.get("attempt", 0))) for c in claim_rows]
    slot_keys = [(str(s.get("logical_job")), int(s.get("attempt", 0))) for s in slot_rows]
    duplicate_claims = sorted({key for key in claim_keys if claim_keys.count(key) > 1})
    duplicate_slots = sorted({key for key in slot_keys if slot_keys.count(key) > 1})
    slot_by_key = {key: row for key, row in zip(slot_keys, slot_rows)}
    missing_slots = [key for key in claim_keys if key not in slot_by_key]
    unmatched_slots = [key for key in slot_keys if key not in set(claim_keys)]
    category_mismatches = []
    for key, claim in zip(claim_keys, claim_rows):
        slot = slot_by_key.get(key)
        if slot is not None and claim.get("category") != slot.get("category"):
            category_mismatches.append({"key": key, "claim": claim.get("category"),
                                        "slot": slot.get("category")})
    normal = sum(c.get("category") == "normal" for c in claim_rows)
    reserve = sum(c.get("category") == "reserve" for c in claim_rows)
    errors = bool(duplicate_claims or duplicate_slots or missing_slots or unmatched_slots
                  or category_mismatches)
    if expected_normal is not None and normal > expected_normal:
        errors = True
    if expected_reserve is not None and reserve > expected_reserve:
        errors = True
    if final and expected_normal is not None and normal != expected_normal:
        errors = True
    if final and expected_reserve is not None and reserve != expected_reserve:
        errors = True
    return {
        "kind": "round58-derived-claim-slot-reconciliation/2",
        "mode": "final" if final else "partial_snapshot",
        "status": "failed" if errors else ("complete" if final else "partial_valid"),
        "claim_attempts": len(claim_rows), "slot_attempts": len(slot_rows),
        "normal_consumed": normal, "reserve_consumed": reserve,
        "total_consumed": len(claim_rows), "duplicate_claim_keys": duplicate_claims,
        "duplicate_slot_keys": duplicate_slots, "claims_without_slots": missing_slots,
        "unmatched_slots": unmatched_slots, "category_mismatches": category_mismatches,
        "expected_normal": expected_normal, "expected_reserve": expected_reserve,
        "errors": errors,
    }


def derive_snapshot(roots: Iterable[dict[str, Any]], *, expected_normal: int,
                    expected_reserve: int, ppo_update_calls_observed: int,
                    optimizer_step_calls_observed: int, final: bool = False) -> dict[str, Any]:
    """Aggregate explicit roots; separate committed batches from invocation hooks."""
    root_rows = list(roots)
    claims = [row for root in root_rows for row in root.get("claims", [])]
    slots = [row for root in root_rows for row in root.get("slots", [])]
    batches = [row for root in root_rows for row in root.get("committed_batches", [])]
    batch_keys = [(int(row.get("seed", -1)), int(row.get("batch", -1))) for row in batches]
    duplicate_batches = sorted({key for key in batch_keys if batch_keys.count(key) > 1})
    closure = reconcile_claims_slots(claims, slots, expected_normal=expected_normal,
                                     expected_reserve=expected_reserve, final=final)
    counter_gap = int(ppo_update_calls_observed) != len(batches)
    errors = closure["errors"] or bool(duplicate_batches) or (final and counter_gap)
    status = "failed" if errors else ("partial_with_counter_gap" if counter_gap
                                      else ("partial_valid" if not final else "complete"))
    return {
        "kind": "round58-derived-runtime-snapshot/2", "status": status,
        "final_closure": final, "roots": [str(r.get("root_id", "unnamed")) for r in root_rows],
        "root_count": len(root_rows), "claim_slot_reconciliation": closure,
        "committed_batch_count": len(batches), "duplicate_batch_keys": duplicate_batches,
        "ppo_update_calls_observed": int(ppo_update_calls_observed),
        "ppo_hook_matches_committed_batches": not counter_gap,
        "optimizer_step_calls_observed": int(optimizer_step_calls_observed),
        "normal_consumed": closure["normal_consumed"],
        "reserve_consumed": closure["reserve_consumed"],
        "total_consumed": closure["total_consumed"],
        "success": not errors and not counter_gap,
        "warnings": (["PPO invocation hook count differs from durable batch commits; values remain separate"]
                     if counter_gap else []),
    }
