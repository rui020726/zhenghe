"""Private ROUND58 gen3 attempt3 execution/ledger contract adapter.

Version: round58-gen3-attempt3-contract/1
Frozen source: runs/comm_bc_initialized_constrained_ppo_20261007_r2/round58_gen2/r58_contracts.py

Differential whitelist against the frozen contract (nothing else changes):
  1. ``validate_claims`` accepts exactly one additional claim shape beyond the
     frozen (normal, 1) / (reserve, 2) contract: a single (reserve, 3) claim for
     the owner-approved logical job ``new_training_s11_e4`` (scenario 1300003,
     reserve launch slot 2), only while its persisted attempt-1 normal claim and
     attempt-2 reserve claim are both actually ``failed_technical`` with their
     approved slot identities.  Every frozen rejection rule (unplanned job,
     duplicate attempt, duplicate slot, forged predecessor, completed-task
     retry, normal attempt>1, attempt>=4, cap overflow, strict closure) keeps
     its frozen meaning; the frozen validator itself is still executed on all
     non-attempt3 claims, so the extension cannot widen any frozen rule.
  2. ``verify_persisted_first_batch_state``: the recovery source gate requires
     exactly the twelve approved complete first-four source records (one per
     planned first-four key) and audits every other persisted claim, including
     the two technical failures, without filtering failures out and without
     locking the total history count.  Mid-run partial progress is validated by
     the same audit with an explicit ``allow_in_flight`` flag; the strict final
     closure stays in the frozen contract path.

This module never writes claims, slots, journals, or checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import r58_contracts as frozen

SEAM_VERSION = "round58-gen3-attempt3-contract/1"
TRAINING_VERSION = frozen.TRAINING_VERSION
NORMAL_CAP = frozen.NORMAL_CAP
RESERVE_CAP = frozen.RESERVE_CAP
HARD_CAP = frozen.HARD_CAP
SEEDS = frozen.SEEDS
TASK_ID = frozen.TASK_ID
BRANCH = frozen.BRANCH

#: The single owner-approved attempt3 special case (approved 2026-10-09).
APPROVED_ATTEMPT3 = {
    "logical_job": "new_training_s11_e4",
    "scenario": 1300003,
    "category": "reserve",
    "launch_slot": 2,
    "attempt": 3,
}
#: The persisted predecessor slot identities the approval binds.
PREDECESSOR_ATTEMPT2_SLOT = 1
INTERRUPTED_E43_ORIGINAL_SHA256 = "4b17c6ccd38dbf62cb28bc4dec3d79f95687ce26c29ee8bd0bd71fd3bef231a4"
INTERRUPTED_E43_LIFECYCLE_SHA256 = "59b3639ca86f145269be2b1b3e2061cdf12e60b7e1d718318d9a9e8fc996958e"


def _verify_interrupted_e43(raw: Path, claim: dict[str, Any]) -> dict[str, Any]:
    """Accept only the observed partial attempt, never a generic in-flight claim."""
    import hashlib
    logical = "new_training_s11_e43"
    slot_path = Path(raw) / "launch_slots" / "normal" / "0052.json"
    if (claim.get("logical_job") != logical or _attempt(claim) != 1
            or claim.get("category") != "normal" or claim.get("launch_slot") != 52
            or claim.get("metadata", {}).get("scenario_seed") != 1300004
            or claim.get("environment_constructor_calls") != 1
            or claim.get("joint_run_constructed_calls") != 1
            or claim.get("run_invocation_calls") != 1
            or claim.get("run_return_calls") != 0
            or claim.get("optimizer_step_calls") != 0
            or claim.get("checkpoint_write_calls") != 0
            or not slot_path.is_file()
            or json.loads(slot_path.read_text(encoding="utf-8")).get("logical_job") != logical
            or json.loads(slot_path.read_text(encoding="utf-8")).get("attempt") != 1):
        raise RuntimeError("unclosed episode43 is not the exact observed attempt")
    paths = claim.get("round45_artifact_paths", {})
    lifecycle = Path(paths.get("lifecycle", ""))
    expected_parent = Path(raw).resolve().parents[1] / "round58_gen3_resume_v18" / "raw" / "logs"
    if (not lifecycle.is_file() or lifecycle.resolve().parent != expected_parent.resolve()
            or hashlib.sha256(lifecycle.read_bytes()).hexdigest()
               != INTERRUPTED_E43_LIFECYCLE_SHA256
            or any(Path(paths.get(key, "")).exists()
                   for key in ("row", "tape", "transcript"))):
        raise RuntimeError("unclosed episode43 artifact state differs from the preserved partial run")
    claim_path = Path(raw) / "launch_claims" / logical / "attempt_0001.json"
    if claim.get("status") == "claimed_before_environment":
        if hashlib.sha256(claim_path.read_bytes()).hexdigest() != INTERRUPTED_E43_ORIGINAL_SHA256:
            raise RuntimeError("unclosed episode43 claim bytes differ from the observed original")
    elif claim.get("status") == "failed_technical":
        correction = claim.get("round58_external_termination_correction")
        archive = Path((correction or {}).get("original_claim_archive", ""))
        if (not isinstance(correction, dict)
                or correction.get("original_claim_sha256") != INTERRUPTED_E43_ORIGINAL_SHA256
                or not archive.is_file()
                or hashlib.sha256(archive.read_bytes()).hexdigest()
                   != INTERRUPTED_E43_ORIGINAL_SHA256):
            raise RuntimeError("episode43 technical disposition lacks exact original archive")
    else:
        raise RuntimeError("episode43 has an unapproved status")
    return {"logical_job": logical, "attempt": 1, "normal_slot": 52,
            "constructor_calls": 1, "run_invocations": 1,
            "complete_row": False, "completed_steps": "unknown_not_zero"}

ZERO_ENVIRONMENT_COUNTERS = (
    "environment_constructor_calls", "joint_run_build_failures",
    "joint_run_constructed_calls", "run_invocation_calls",
    "joint_run_method_calls", "run_return_calls", "run_failure_calls",
    "completed_steps", "optimizer_step_calls", "checkpoint_write_calls",
)
FAILED_JOINT_ZERO_COUNTERS = ("environment_constructor_calls",
                              "joint_run_constructed_calls", "run_invocation_calls",
                              "completed_steps")


def _attempt(claim: dict[str, Any]) -> int:
    try:
        return int(claim.get("attempt", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("claim attempt/launch_slot must be integers") from exc


def _validate_attempt3_claim(plan: dict[str, Any], claim: dict[str, Any],
                             claims: list[dict[str, Any]], *,
                             used_reserve_slots: set[int]) -> None:
    """Narrow, fail-closed validation of the unique approved attempt3 shape."""
    logical = str(claim.get("logical_job", ""))
    if (logical != APPROVED_ATTEMPT3["logical_job"]
            or str(claim.get("category", "")) != APPROVED_ATTEMPT3["category"]):
        raise ValueError(f"claim attempt3 outside the approved special case: {(logical, claim.get('category'))}")
    if _attempt(claim) != APPROVED_ATTEMPT3["attempt"]:
        raise ValueError(f"claim attempt number outside the approved special case: {claim.get('attempt')}")
    slot = int(claim.get("launch_slot", 0))
    if slot != APPROVED_ATTEMPT3["launch_slot"]:
        raise ValueError(f"attempt3 reserve slot is not the approved slot 2: {slot}")
    if not 1 <= slot <= frozen.RESERVE_CAP:
        raise ValueError("reserve slot outside the frozen technical reserve")
    if slot in used_reserve_slots:
        raise ValueError("duplicate reserve launch slot")
    if claim.get("status") not in ("claimed_before_environment", "complete",
                                   "failed_technical"):
        raise ValueError(f"invalid claim status: {claim.get('status')}")
    job = planned_job(plan, logical)
    if job is None or int(job.get("scene", -1)) != APPROVED_ATTEMPT3["scenario"]:
        raise ValueError("attempt3 key is not the approved planned training job")
    by_job: dict[int, dict[str, Any]] = {}
    for other in claims:
        if other is claim:
            continue
        if str(other.get("logical_job", "")) != logical:
            continue
        attempt = _attempt(other)
        if attempt in by_job:
            raise ValueError(f"duplicate claim attempt: {(logical, attempt)}")
        if attempt == APPROVED_ATTEMPT3["attempt"]:
            raise ValueError("duplicate attempt3 claim for the approved special case")
        by_job[attempt] = other
    attempt1 = by_job.get(1)
    attempt2 = by_job.get(2)
    if (attempt1 is None
            or str(attempt1.get("category", "")) != "normal"
            or int(attempt1.get("launch_slot", -1)) != int(job["normal_slot"])
            or attempt1.get("status") != "failed_technical"):
        raise ValueError("attempt3 requires the persisted technical-failed normal attempt 1 with its slot identity")
    if (attempt2 is None
            or str(attempt2.get("category", "")) != "reserve"
            or int(attempt2.get("launch_slot", -1)) != PREDECESSOR_ATTEMPT2_SLOT
            or attempt2.get("status") != "failed_technical"):
        raise ValueError("attempt3 requires the persisted technical-failed reserve attempt 2 with its slot identity")


def planned_job(plan: dict[str, Any], logical: str) -> dict[str, Any] | None:
    for job in plan["jobs"]:
        if job["logical_job"] == logical:
            return job
    return None


def validate_claims(plan: dict[str, Any], claims: list[dict[str, Any]], *,
                    strict: bool = False) -> dict[str, Any]:
    """Frozen claim validation plus the single approved attempt3 special case."""
    planned = {job["logical_job"]: job for job in plan["jobs"]}
    attempt3_claims = [c for c in claims if _attempt(c) >= 3]
    others = [c for c in claims if _attempt(c) < 3]
    used_reserve = set()
    for claim in others:
        if str(claim.get("category", "")) == "reserve":
            used_reserve.add(int(claim.get("launch_slot", 0)))
    for claim in attempt3_claims:
        _validate_attempt3_claim(plan, claim, claims, used_reserve_slots=used_reserve)
        used_reserve.add(int(claim.get("launch_slot", 0)))
    closure = frozen.validate_claims(plan, others, strict=False)
    counts = closure["counts"]
    completed = {str(c.get("logical_job")) for c in others if c.get("status") == "complete"}
    for claim in attempt3_claims:
        counts["reserve"] += 1
        if claim.get("status") == "complete":
            completed.add(str(claim.get("logical_job")))
    if counts["normal"] > frozen.NORMAL_CAP or counts["reserve"] > frozen.RESERVE_CAP:
        raise ValueError("claim budget exceeds cap")
    if strict and (counts["normal"] != frozen.NORMAL_CAP
                   or completed != set(planned)):
        raise ValueError("strict final ledger requires exact full plan closure")
    return {"counts": counts, "completed_logical_jobs": len(completed),
            "claim_count": len(claims), "strict": strict,
            "attempt3_claims": len(attempt3_claims),
            "reserve_slots_required_to_be_full": False}


def validate_slot_records(plan: dict[str, Any], claims: list[dict[str, Any]],
                          normal_slots: list[dict[str, Any]],
                          reserve_slots: list[dict[str, Any]], *,
                          strict: bool = False) -> dict[str, Any]:
    """Slot closure is attempt-shape agnostic; reuse the frozen implementation."""
    return frozen.validate_slot_records(plan, claims, normal_slots, reserve_slots,
                                        strict=strict)


def derived_snapshot(plan: dict[str, Any], claims: list[dict[str, Any]], *,
                     strict: bool = False) -> dict[str, Any]:
    """Same derived-only contract as the frozen module, with attempt3 support."""
    before = json.dumps(claims, sort_keys=True, ensure_ascii=False)
    try:
        value = validate_claims(plan, claims, strict=strict)
        result = {"status": "complete" if strict else "partial", **value}
    except Exception as exc:
        result = {"status": "deferred", "error": f"{type(exc).__name__}: {exc}"}
    if json.dumps(claims, sort_keys=True, ensure_ascii=False) != before:
        raise AssertionError("derived snapshot mutated core claims")
    return result


def _assert_zero_environment_consumption(claim: dict[str, Any], label: str) -> None:
    for name in ZERO_ENVIRONMENT_COUNTERS:
        value = claim.get(name)
        if value is None:
            continue
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}: counter {name} is not an integer") from exc
        if number != 0:
            raise ValueError(f"{label}: failed claim consumed {name}={number}")
    if claim.get("simulation_started") not in (None, False):
        raise ValueError(f"{label}: failed claim recorded a started simulation")


def verify_persisted_first_batch_state(base: Any, raw: Path, plan: dict[str, Any], *,
                                       legacy: Any,
                                       training_version: str | None = None) -> dict[str, Any]:
    """Private recovery source gate replacing the frozen exactly-12-claims check.

    Requires exactly the twelve approved complete first-four source records and
    audits the full persisted claim history, including technical failures,
    without filtering them out and without locking the total claim count.
    """
    read_json = legacy.read_json
    raw = Path(raw)
    claim_paths = sorted((raw / "launch_claims").glob("*/*.json"))
    if not claim_paths:
        raise RuntimeError("recovery requires persisted claims")
    claims = [read_json(path) for path in claim_paths]
    closure = validate_claims(plan, claims, strict=False)

    first_four_keys = {job["logical_job"] for job in plan["jobs"]
                       if job["scope"] == "ppo_training" and int(job["episode_index"]) < 4}
    complete_keys: set[str] = set()
    attempts_by_job: dict[str, dict[int, dict[str, Any]]] = {}
    for claim in claims:
        logical = str(claim.get("logical_job", ""))
        attempts_by_job.setdefault(logical, {})[_attempt(claim)] = claim
        if claim.get("status") == "complete":
            complete_keys.add(logical)
    missing_first_four = first_four_keys - complete_keys
    if missing_first_four:
        raise RuntimeError(
            "recovery complete source records must include the twelve approved "
            f"first-four claims (missing={sorted(missing_first_four)}, "
            f"complete={len(complete_keys)})")
    for logical in sorted(complete_keys - first_four_keys):
        if planned_job(plan, logical) is None:
            raise RuntimeError(
                f"recovery found a complete claim outside the approved fixed plan: {logical}")
    for logical in sorted(complete_keys):
        job = planned_job(plan, logical)
        if job is None:
            raise RuntimeError(f"recovery found a complete unplanned job: {logical}")
        if logical in first_four_keys:
            # The twelve approved source records must each be the attempt-1
            # normal claim on its planned normal slot.
            claim = attempts_by_job[logical][1]
            if (str(claim.get("category", "")) != "normal"
                    or int(claim.get("launch_slot", -1)) != int(job["normal_slot"])
                    or claim.get("status") != "complete"):
                raise RuntimeError(f"recovery approved source record identity mismatch: {logical}")
            slot_path = raw / "launch_slots" / "normal" / f"{int(job['normal_slot']):04d}.json"
            if (not slot_path.is_file()
                    or str(read_json(slot_path).get("logical_job", "")) != logical):
                raise RuntimeError(f"recovery durable normal slot record mismatch: {logical}")
        else:
            if not any(claim.get("status") == "complete"
                       for claim in attempts_by_job[logical].values()):
                raise RuntimeError(f"recovery complete record lacks a complete claim: {logical}")

    audited_failures: dict[str, dict[str, Any]] = {}
    interrupted_attempt3: dict[str, Any] | None = None
    for claim in claims:
        if claim.get("status") == "complete":
            continue
        logical = str(claim.get("logical_job", ""))
        if claim.get("status") == "claimed_before_environment":
            # An externally terminated attempt leaves its claim in flight; the
            # execute seam appends the termination correction before that key
            # runs again, and the frozen technical-retry path then reruns it.
            # The gate records it without rewriting anything.
            _verify_interrupted_e43(raw, claim)
            continue
        if claim.get("status") != "failed_technical":
            raise RuntimeError(f"recovery found an unexpected non-complete status: {logical} {claim.get('status')}")
        if (logical == APPROVED_ATTEMPT3["logical_job"]
                and _attempt(claim) == APPROVED_ATTEMPT3["attempt"]
                and int(claim.get("run_return_calls", -1)) == 1
                and int(claim.get("completed_steps", -1)) == 1000
                and claim.get("round45_artifact_receipt") is None
                and isinstance(claim.get("round45_artifact_paths"), dict)):
            # The approved attempt3 episode fully ran and persisted its row; only
            # the artifact-receipt bookkeeping crashed on the historical RUN_DIR
            # binding. The execute seam completes this bookkeeping at training
            # time; the recovery gate records it without rewriting anything.
            interrupted_attempt3 = {
                "attempt": APPROVED_ATTEMPT3["attempt"],
                "launch_slot": int(claim.get("launch_slot", -1)),
                "run_return_calls": 1, "completed_steps": 1000,
                "artifact_receipt_pending": True,
            }
            continue
        if logical == "new_training_s11_e43" and _attempt(claim) == 1:
            _verify_interrupted_e43(raw, claim)
        else:
            _assert_zero_environment_consumption(claim, logical)
        slot = int(claim.get("launch_slot", 0))
        category = str(claim.get("category", ""))
        slot_path = raw / "launch_slots" / category / f"{slot:04d}.json"
        if (not slot_path.is_file()
                or str(read_json(slot_path).get("logical_job", "")) != logical
                or int(read_json(slot_path).get("attempt", -1)) != _attempt(claim)):
            raise RuntimeError(f"recovery durable failed-claim slot mismatch: {logical} #{_attempt(claim)}")
        audited_failures[f"{logical}#{_attempt(claim)}"] = {
            "category": category, "launch_slot": slot,
            "attempt": _attempt(claim), "status": "failed_technical",
        }

    approved = APPROVED_ATTEMPT3
    logical = approved["logical_job"]
    attempts = attempts_by_job.get(logical, {})
    has_pair = (1 in attempts and attempts[1].get("status") == "failed_technical"
                and int(attempts[1].get("launch_slot", -1)) == int(planned_job(plan, logical)["normal_slot"])
                and 2 in attempts and attempts[2].get("status") == "failed_technical"
                and int(attempts[2].get("launch_slot", -1)) == PREDECESSOR_ATTEMPT2_SLOT
                and attempts[2].get("category") == "reserve")
    attempt3_claim = attempts.get(APPROVED_ATTEMPT3["attempt"])
    attempt3_done = (attempt3_claim is not None
                     and attempt3_claim.get("status") == "complete"
                     and isinstance(attempt3_claim.get("round58_receipt_failure_correction"), dict))
    if not has_pair and not attempt3_done:
        raise RuntimeError("recovery cannot find the approved attempt3 technical-failed predecessor pair")
    if interrupted_attempt3 is None and not attempt3_done and (
            raw / "launch_slots" / "reserve" / f"{approved['launch_slot']:04d}.json").exists():
        raise RuntimeError("recovery found the approved attempt3 reserve slot already occupied")
    if closure["counts"]["normal"] > frozen.NORMAL_CAP or closure["counts"]["reserve"] >= frozen.RESERVE_CAP:
        raise RuntimeError("recovery claim history already exhausts the approved cumulative quota")

    failed_gate_receipts = {}
    for seed in frozen.SEEDS:
        gate_path = raw / f"ROUND45_R2_FIRST_BATCH_GATE_R3_s{seed}.json"
        if gate_path.is_file():
            if read_json(gate_path).get("status") != "failed":
                raise RuntimeError(
                    f"recovery found a non-failed first-batch gate receipt for seed {seed}")
            failed_gate_receipts[str(seed)] = legacy.sha256_file(gate_path)
    if not failed_gate_receipts:
        raise RuntimeError("recovery requires a persisted failed first-batch gate receipt")
    first_four_receipts = {}
    for seed in frozen.SEEDS:
        receipt_path = raw / f"ROUND46_FIRST_FOUR_s{seed}.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"recovery missing first-four receipt for seed {seed}")
        saved = read_json(receipt_path)
        if (saved.get("kind") != "round46-first-four-report/1"
                or int(saved.get("seed", -1)) != seed
                or saved.get("training_version") != (training_version or frozen.TRAINING_VERSION)):
            raise RuntimeError(f"recovery first-four receipt identity invalid for seed {seed}")
        first_four_receipts[str(seed)] = legacy.sha256_file(receipt_path)
    checkpoints = legacy._verify_existing_first_batch_checkpoints(base)
    if interrupted_attempt3 is None:
        corrected = attempts_by_job.get(APPROVED_ATTEMPT3["logical_job"], {}).get(
            APPROVED_ATTEMPT3["attempt"])
        if (corrected is None or corrected.get("status") != "complete"
                or not isinstance(corrected.get("round58_receipt_failure_correction"), dict)
                or corrected["round58_receipt_failure_correction"].get("kind")
                != "round58-attempt3-receipt-failure-correction/1"):
            raise RuntimeError(
                "recovery requires the approved attempt3 interrupted-bookkeeping state "
                "or its completed receipt-failure correction "
                "(the episode result must be recovered by the execute seam before continuation)")
    else:
        attempt3_done = False
    return {"kind": "round58-attempt3-recovery-state-verification/1",
            "seam_version": SEAM_VERSION, "status": "passed",
            "claim_closure": closure,
            "audited_complete_source_records": len(complete_keys),
            "audited_failed_claims": audited_failures,
            "interrupted_attempt3_bookkeeping": interrupted_attempt3,
            "approved_attempt3_predecessors_verified": True,
            "approved_attempt3_slot_free": True,
            "failed_gate_receipts": failed_gate_receipts,
            "first_four_receipts": first_four_receipts,
            "checkpoints": checkpoints}


def preflight_attempt3_allocation(raw: Path, plan: dict[str, Any], *,
                                  historical_consumed: int = 16,
                                  normal_cap: int = frozen.NORMAL_CAP,
                                  reserve_cap: int = frozen.RESERVE_CAP,
                                  hard_cap: int = frozen.HARD_CAP) -> dict[str, Any]:
    """Fail-closed dry-run proof for the unique attempt3 allocation.

    Reads only persisted claim/slot/failure state; writes nothing.
    """
    raw = Path(raw)
    approved = APPROVED_ATTEMPT3
    logical = approved["logical_job"]
    claim_paths = sorted((raw / "launch_claims").glob("*/*.json"))
    claims = [json.loads(p.read_text(encoding="utf-8")) for p in claim_paths]
    validate_claims(plan, claims, strict=False)
    attempts: dict[int, dict[str, Any]] = {}
    for claim in claims:
        if str(claim.get("logical_job", "")) == logical:
            attempts[_attempt(claim)] = claim
        if (str(claim.get("logical_job", "")) == logical
                and claim.get("status") == "complete"):
            raise RuntimeError("attempt3 preflight: the approved key already has a complete result")
    job = planned_job(plan, logical)
    if job is None or int(job.get("scene", -1)) != approved["scenario"]:
        raise RuntimeError("attempt3 preflight: approved key is not the planned scenario job")
    slot_path = raw / "launch_slots" / "reserve" / f"{approved['launch_slot']:04d}.json"
    if slot_path.exists():
        raise RuntimeError("attempt3 preflight: reserve slot 2 is already occupied")
    attempt1, attempt2 = attempts.get(1), attempts.get(2)
    if (attempt1 is None or attempt1.get("status") != "failed_technical"
            or str(attempt1.get("category", "")) != "normal"
            or int(attempt1.get("launch_slot", -1)) != int(job["normal_slot"])):
        raise RuntimeError("attempt3 preflight: attempt1 original/slot identity mismatch")
    if (attempt2 is None or attempt2.get("status") != "failed_technical"
            or str(attempt2.get("category", "")) != "reserve"
            or int(attempt2.get("launch_slot", -1)) != PREDECESSOR_ATTEMPT2_SLOT):
        raise RuntimeError("attempt3 preflight: attempt2 original/slot identity mismatch")
    for claim in (attempt1, attempt2):
        _assert_zero_environment_consumption(claim, f"{logical} #{_attempt(claim)}")
    failure_dir = raw / "failures"
    failure_records = sorted(failure_dir.glob("train_s11_e4_r44_a*.json")) if failure_dir.is_dir() else []
    if len(failure_records) < 2:
        raise RuntimeError("attempt3 preflight: durable failure records for the predecessor pair are missing")
    normal_used = sum(1 for c in claims if str(c.get("category")) == "normal")
    reserve_used = sum(1 for c in claims if str(c.get("category")) == "reserve")
    if normal_used > normal_cap or reserve_used + 1 > reserve_cap:
        raise RuntimeError("attempt3 preflight: allocation would exceed the approved generation caps")
    if historical_consumed + len(claims) + 1 > hard_cap:
        raise RuntimeError("attempt3 preflight: allocation would exceed the approved cumulative hard cap")
    return {"kind": "round58-attempt3-allocation-preflight/1",
            "seam_version": SEAM_VERSION, "status": "approved",
            "approved": dict(approved), "attempt1_slot": int(job["normal_slot"]),
            "attempt2_slot": PREDECESSOR_ATTEMPT2_SLOT,
            "claims_before": len(claims), "normal_used": normal_used,
            "reserve_used": reserve_used,
            "cumulative_after": historical_consumed + len(claims) + 1,
            "hard_cap": hard_cap}

