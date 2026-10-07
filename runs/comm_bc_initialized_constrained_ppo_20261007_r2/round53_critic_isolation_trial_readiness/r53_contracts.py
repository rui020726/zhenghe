"""Pure ROUND53 job, quota, authorization, ledger, and checkpoint contracts."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

TASK_ID = "ROUND53-ZERO-LAUNCH-ISOLATED-TRIAL-READINESS"
BRANCH = "feat/round53-critic-isolation-trial-readiness"
TRAINING_VERSION = "constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/dual-eta-lambda-0p5/1/rule-bc-late147-init/1/published-completion-event-at-relay-tick/2+r53-critic-representation-stop-gradient/1"
SEEDS = (11, 22, 33)
SCENES = tuple(range(1_301_000, 1_301_016))
NORMAL_CAP, RESERVE_CAP, HARD_CAP = 528, 24, 552


def build_fixed_plan(training_order: list[int]) -> dict[str, Any]:
    if len(training_order) != 128 or set(training_order) != set(range(1_300_000, 1_300_048)):
        raise ValueError("frozen training sequence must be 128 entries over the exact 48 scenes")
    jobs = []
    # Matches the existing first-four barrier (seed order), then per-seed
    # continuation (episode 4..127), without adding smoke episodes.
    for seed in SEEDS:
        for episode, scene in enumerate(training_order[:4]):
            jobs.append({"scope": "ppo_training", "logical_job": f"new_training_s{seed}_e{episode}",
                         "seed": seed, "episode_index": episode, "scene": scene,
                         "normal_slot": len(jobs) + 1})
    for seed in SEEDS:
        for episode, scene in enumerate(training_order[4:], start=4):
            jobs.append({"scope": "ppo_training", "logical_job": f"new_training_s{seed}_e{episode}",
                         "seed": seed, "episode_index": episode, "scene": scene,
                         "normal_slot": len(jobs) + 1})
    for seed in SEEDS:
        for mode in ("greedy", "sampled"):
            for scene in SCENES:
                jobs.append({"scope": "candidate_eval", "logical_job": f"model_eval_new_s{seed}_{mode}_scene{scene}",
                             "seed": seed, "mode": mode, "scene": scene,
                             "normal_slot": len(jobs) + 1})
    for seed in SEEDS:
        for scene in SCENES:
            jobs.append({"scope": "bc_sampled_eval", "logical_job": f"model_eval_bc_s{seed}_sampled_scene{scene}",
                         "seed": seed, "mode": "sampled", "scene": scene,
                         "normal_slot": len(jobs) + 1})
    if len(jobs) != NORMAL_CAP or len({j["logical_job"] for j in jobs}) != NORMAL_CAP:
        raise AssertionError("fixed normal jobs are not exactly 528 unique keys")
    return {"kind": "round53-fixed-job-plan/1", "status": "proposed_not_authorized",
            "task_id": TASK_ID, "branch": BRANCH, "training_version": TRAINING_VERSION,
            "training_episode_count": 384, "candidate_eval_count": 96,
            "bc_sampled_eval_count": 48, "normal_count": NORMAL_CAP,
            "technical_reserve_count": RESERVE_CAP, "hard_cap": HARD_CAP,
            "jobs": jobs,
            "reserve_slots": [{"category": "reserve", "slot": i} for i in range(1, RESERVE_CAP + 1)],
            "eval_scenes": list(SCENES), "development_block_already_used": True,
            "sealed_confirmation_opened": False}


def authorize_receipt(receipt: dict[str, Any], *, plan_sha256: str) -> None:
    """Fail closed; a readiness report or a user's code request is not a receipt."""
    required = {"status": "explicitly_authorized", "task_id": TASK_ID, "branch": BRANCH,
                "normal_cap": NORMAL_CAP, "reserve_cap": RESERVE_CAP,
                "hard_cap": HARD_CAP, "plan_sha256": plan_sha256}
    for key, value in required.items():
        if receipt.get(key) != value:
            raise PermissionError(f"explicit execution authorization missing/mismatched: {key}")
    if not all(isinstance(receipt.get(k), str) and receipt[k].strip()
               for k in ("authorized_by", "approved_at", "approval_reference")):
        raise PermissionError("authorization requires human identity, timestamp, and approval reference")


def validate_claims(plan: dict[str, Any], claims: list[dict[str, Any]], *, strict: bool = False) -> dict[str, Any]:
    """Validate authoritative claims independent of directory iteration order.

    Reserve attempts are accepted only when the persisted attempt-1 core claim
    for the same planned key is actually technical-failed.  A copied
    ``original_status`` field is corroboration, never the authority.
    """
    planned = {job["logical_job"]: job for job in plan["jobs"]}
    by_job: dict[str, dict[int, dict[str, Any]]] = {}
    counts = {"normal": 0, "reserve": 0}
    completed: set[str] = set()
    used_slots = {"normal": set(), "reserve": set()}
    for claim in claims:
        logical = str(claim.get("logical_job", ""))
        try:
            attempt = int(claim.get("attempt", 1))
            slot = int(claim.get("launch_slot", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("claim attempt/launch_slot must be integers") from exc
        category = str(claim.get("category", ""))
        status = claim.get("status")
        if logical not in planned:
            raise ValueError(f"unplanned logical job: {logical}")
        if attempt not in (1, 2) or (category, attempt) not in (("normal", 1), ("reserve", 2)):
            raise ValueError(f"claim category/attempt mismatch: {(logical, attempt)} {category}")
        attempts = by_job.setdefault(logical, {})
        if attempt in attempts:
            raise ValueError(f"duplicate claim attempt: {(logical, attempt)}")
        if status not in ("claimed_before_environment", "complete", "failed_technical",
                          "not_started_quota_exhausted"):
            raise ValueError(f"invalid claim status: {status}")
        if category == "normal":
            if slot != int(planned[logical]["normal_slot"]):
                raise ValueError(f"normal claim slot mismatch: {logical}")
        elif not 1 <= slot <= RESERVE_CAP:
            raise ValueError("reserve slot outside the frozen technical reserve")
        if slot in used_slots[category]:
            raise ValueError(f"duplicate {category} launch slot")
        used_slots[category].add(slot)
        counts[category] += 1
        attempts[attempt] = claim
        if status == "complete":
            completed.add(logical)

    for logical, attempts in by_job.items():
        retry = attempts.get(2)
        if retry is None:
            continue
        original = attempts.get(1)
        if (original is None or original.get("category") != "normal"
                or int(original.get("attempt", -1)) != 1
                or original.get("status") != "failed_technical"
                or int(original.get("launch_slot", -1)) != int(planned[logical]["normal_slot"])):
            raise ValueError("reserve may retry only a persisted technical-failed normal core claim")
        if retry.get("original_status") not in (None, "failed_technical"):
            raise ValueError("reserve original_status contradicts the persisted core claim")

    if counts["normal"] > NORMAL_CAP or counts["reserve"] > RESERVE_CAP:
        raise ValueError("claim budget exceeds cap")
    if strict and (counts["normal"] != NORMAL_CAP or completed != set(planned)):
        raise ValueError("strict final ledger requires exact full plan closure")
    return {"counts": counts, "completed_logical_jobs": len(completed),
            "claim_count": len(claims), "strict": strict,
            "reserve_slots_required_to_be_full": False}


def validate_slot_records(plan: dict[str, Any], claims: list[dict[str, Any]],
                          normal_slots: list[dict[str, Any]],
                          reserve_slots: list[dict[str, Any]], *,
                          strict: bool = False) -> dict[str, Any]:
    """Close durable slot files against the authoritative normal/reserve claims."""
    expected: dict[str, dict[int, tuple[str, int]]] = {"normal": {}, "reserve": {}}
    for claim in claims:
        category = str(claim.get("category", ""))
        if category not in expected:
            raise ValueError(f"unknown claim slot category: {category}")
        slot = int(claim.get("launch_slot", 0))
        logical = str(claim.get("logical_job", ""))
        attempt = int(claim.get("attempt", 0))
        if slot in expected[category]:
            raise ValueError(f"duplicate claim slot: {category}/{slot}")
        expected[category][slot] = (logical, attempt)

    observed: dict[str, dict[int, tuple[str, int]]] = {"normal": {}, "reserve": {}}
    for category, records in (("normal", normal_slots), ("reserve", reserve_slots)):
        for record in records:
            slot = int(record.get("slot", 0))
            pair = (str(record.get("logical_job", "")), int(record.get("attempt", 0)))
            if record.get("category") != category or slot <= 0:
                raise ValueError(f"invalid durable slot record: {category}/{slot}")
            if slot in observed[category]:
                raise ValueError(f"duplicate durable slot record: {category}/{slot}")
            observed[category][slot] = pair
        if observed[category] != expected[category]:
            raise ValueError(f"durable {category} slots do not match authoritative claims")
    if strict and (len(observed["normal"]) != NORMAL_CAP
                   or len(observed["reserve"]) > RESERVE_CAP):
        raise ValueError("strict durable slot ledger exceeds or fails frozen slot closure")
    return {"normal_slots": len(observed["normal"]),
            "reserve_slots": len(observed["reserve"]),
            "reserve_capacity_required_to_be_full": False,
            "claims_and_slots_match": True}


def derived_snapshot(plan: dict[str, Any], claims: list[dict[str, Any]], *, strict: bool = False) -> dict[str, Any]:
    """Snapshot failure is derived-only and never mutates authoritative claims."""
    before = json.dumps(claims, sort_keys=True, ensure_ascii=False)
    try:
        value = validate_claims(plan, claims, strict=strict)
        result = {"status": "complete" if strict else "partial", **value}
    except Exception as exc:
        result = {"status": "deferred", "error": f"{type(exc).__name__}: {exc}"}
    if json.dumps(claims, sort_keys=True, ensure_ascii=False) != before:
        raise AssertionError("derived snapshot mutated core claims")
    return result


def nonfatal_derived_writer(original, warning_sink):
    """Wrap a derived ledger writer so snapshot faults cannot rewrite core claims."""
    def write(*, strict: bool = True):
        try:
            return original(strict=strict)
        except Exception as exc:
            warning = {"kind": "round53-derived-snapshot-warning/1", "status": "deferred",
                       "strict_requested": bool(strict),
                       "error": f"{type(exc).__name__}: {exc}",
                       "core_claims_mutated": False}
            try: warning_sink(warning)
            except Exception: pass
            return warning
    return write


def validate_checkpoint_identity(payload: dict[str, Any], *, freeze: dict[str, Any],
                                 allow_model_only: bool = False) -> str:
    """Old optimizer/RNG states reject; model-only BC initialization is separate."""
    if allow_model_only and payload.get("kind") == "bc_model_only" and "optimizer" not in payload and "rng" not in payload:
        return "accepted_model_only_initialization"
    if payload.get("training_version") != TRAINING_VERSION or payload.get("freeze") != freeze:
        raise ValueError("checkpoint training-version/freeze mismatch; resume refused")
    state = payload.get("training_state", {})
    if state.get("training_version") != TRAINING_VERSION:
        raise ValueError("checkpoint training state has an old route version")
    return "accepted_same_round53_resume"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
