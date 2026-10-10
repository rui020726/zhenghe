"""Pure read-only claim/slot and constructor accounting before V26 execution."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R2 = HERE.parent
RAW = R2 / "round58_gen3/raw"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reconcile(claims: list[dict], slots: list[dict], plan: dict,
              quota: dict, historical: dict) -> dict:
    expected_historical = (int(historical["environment_consumption"]["gen1_complete_claims"])
                           + int(historical["environment_consumption"]["gen2_complete_claims"]))
    if (expected_historical != 16
            or int(historical["environment_consumption"]["known_constructor_and_run_calls"]) != 16
            or int(quota["historical_consumed"]) != expected_historical
            or int(quota["new_normal_cap"]) != 1024
            or int(quota["new_reserve_cap"]) != 24
            or int(quota["hard_cap"]) != 1064):
        raise RuntimeError("historical evidence/quota unit mismatch")
    jobs = {job["logical_job"]: job for job in plan["jobs"]}
    if len(jobs) != 1024 or len(plan["jobs"]) != 1024:
        raise RuntimeError("fixed plan keys are not unique and complete")
    key_claims = {}
    by_slot = {}
    for claim in claims:
        logical = claim.get("logical_job")
        if logical not in jobs:
            raise RuntimeError("claim outside frozen plan")
        category, slot, attempt = claim.get("category"), int(claim["launch_slot"]), int(claim["attempt"])
        if category not in ("normal", "reserve") or slot < 1:
            raise RuntimeError("claim has invalid category/slot")
        key = (category, slot)
        if key in by_slot:
            raise RuntimeError("duplicate claim slot")
        by_slot[key] = claim
        key_claims.setdefault(logical, []).append(claim)
        if category == "normal" and (attempt != 1 or slot != int(jobs[logical]["normal_slot"])):
            raise RuntimeError("normal claim differs from its planned first attempt")
        if category == "reserve" and attempt not in (2, 3):
            raise RuntimeError("unapproved reserve attempt number")
        if category == "reserve" and attempt == 3 and (logical, slot) != ("new_training_s11_e4", 2):
            raise RuntimeError("unapproved attempt3")
        if int(claim.get("environment_constructor_calls", -1)) not in (0, 1):
            raise RuntimeError("claim constructor count is unknown")
        if int(claim.get("run_invocation_calls", -1)) not in (0, 1):
            raise RuntimeError("claim run count is unknown")
    seen_slots = {}
    for slot in slots:
        key = (slot.get("category"), int(slot["slot"]))
        if key in seen_slots:
            raise RuntimeError("duplicate slot record")
        seen_slots[key] = slot
        claim = by_slot.get(key)
        if claim is None or (slot.get("logical_job"), int(slot.get("attempt", -1))) != (
                claim.get("logical_job"), int(claim["attempt"])):
            raise RuntimeError("slot without exact matching claim")
    if set(seen_slots) != set(by_slot):
        raise RuntimeError("claim without exact matching slot")
    normal_slots = sorted(slot for category, slot in seen_slots if category == "normal")
    reserve_slots = sorted(slot for category, slot in seen_slots if category == "reserve")
    if (normal_slots != list(range(1, len(normal_slots) + 1))
            or reserve_slots != list(range(1, len(reserve_slots) + 1))
            or len(normal_slots) > 1024 or len(reserve_slots) > 24
            or expected_historical + len(claims) > 1064):
        raise RuntimeError("slot allocation or hard cap does not close")
    complete = {}
    for logical, attempts in key_claims.items():
        done = [claim for claim in attempts if claim.get("status") == "complete"]
        if len(done) > 1:
            raise RuntimeError("duplicate completed logical task")
        if done:
            complete[logical] = done[0]
    constructor = sum(int(claim["environment_constructor_calls"]) for claim in claims)
    run = sum(int(claim["run_invocation_calls"]) for claim in claims)
    return {"historical_environment_starts": expected_historical,
            "gen3_claim_attempts": len(claims),
            "gen3_normal_slots": len(normal_slots), "gen3_reserve_slots": len(reserve_slots),
            "cumulative_attempt_slots": expected_historical + len(claims),
            "gen3_actual_constructor_calls": constructor,
            "gen3_actual_run_invocations": run,
            "cumulative_actual_constructor_calls": expected_historical + constructor,
            "cumulative_actual_run_invocations": expected_historical + run,
            "complete_logical_jobs": len(complete),
            "remaining_plan_keys": 1024 - len(complete),
            "failed_technical_claims": sum(c.get("status") == "failed_technical" for c in claims),
            "inflight_claims": sum(c.get("status") == "claimed_before_environment" for c in claims)}


def require_derived_environment_consumed(derived: dict, audit: dict) -> None:
    if int(derived.get("environment_consumed", -1)) != audit["cumulative_actual_constructor_calls"]:
        raise RuntimeError("derived ledger environment_consumed does not equal raw physical starts")


def audit() -> dict:
    quota_path = HERE / "ROUND58_GEN3_CUMULATIVE_QUOTA.json"
    quota = read(quota_path)
    historical_path = Path(quota["historical_manifest_path"])
    if sha(historical_path) != quota["historical_manifest_sha256"]:
        raise RuntimeError("historical consumption evidence changed")
    claims = [read(p) for p in sorted((RAW / "launch_claims").glob("*/*.json"))]
    slots = [read(p) for p in sorted((RAW / "launch_slots").glob("*/*.json"))]
    plan = read(HERE / "ROUND58_GEN3_FIXED_PLAN.json")
    result = reconcile(claims, slots, plan, quota, read(historical_path))
    # Negative controls are in memory only; no historical claim or slot is edited.
    with_missing_failure = [c for c in claims if not (
        c["logical_job"] == "new_training_s11_e4" and int(c["attempt"]) == 1)]
    try:
        reconcile(with_missing_failure, slots, plan, quota, read(historical_path))
    except RuntimeError as exc:
        missing_failure_rejected = str(exc)
    else:
        raise RuntimeError("missing historical failure negative control passed unexpectedly")
    bad_quota = copy.deepcopy(quota)
    bad_quota["historical_consumed"] = 0
    try:
        reconcile(claims, slots, plan, bad_quota, read(historical_path))
    except RuntimeError as exc:
        zero_history_rejected = str(exc)
    else:
        raise RuntimeError("zero historical consumption negative control passed unexpectedly")
    for wrong in (0, result["cumulative_actual_constructor_calls"] - 1):
        try:
            require_derived_environment_consumed({"environment_consumed": wrong}, result)
        except RuntimeError:
            continue
        raise RuntimeError("derived ledger undercount negative control passed unexpectedly")
    result.update({"kind": "round58-v26-independent-prelaunch-ledger/1",
                   "status": "raw_claim_slot_and_historical_accounting_closed",
                   "quota_sha256": sha(quota_path),
                   "historical_manifest_sha256": sha(historical_path),
                   "negative_controls": {"missing_failed_claim": missing_failure_rejected,
                                         "zero_historical_consumption": zero_history_rejected,
                                         "derived_zero_or_minus_one_physical_start": "rejected"},
                   "actual_cli_derived_ledger_write_spy": "not_invoked_in_V26_restore_gate"})
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2, sort_keys=True))
