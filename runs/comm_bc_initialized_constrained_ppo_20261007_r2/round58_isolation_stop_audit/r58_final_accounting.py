"""Independent original-claim and cross-root operation accounting for R58."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import r58_v26_independent_ledger_preflight as ledger

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RAW = BASE / "round58_gen3/raw"
ROOTS = [BASE / "round58_gen3/raw"] + [
    BASE / f"round58_gen3_resume_v{v}/raw" for v in (16, 18, 22, 26, 27, 29, 32)]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def events(name: str):
    answer, sources = [], []
    for root in ROOTS:
        path = root / "runtime" / name
        if not path.is_file():
            continue
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        answer.extend(lines)
        sources.append({"path": str(path), "sha256": sha(path), "events": len(lines)})
    return answer, sources


def main() -> None:
    counts = ledger.audit()
    if counts["complete_logical_jobs"] != 1024 or counts["remaining_plan_keys"] != 0:
        raise RuntimeError("final accounting requires 1024 unique completed tasks")
    claims = [(path, read(path)) for path in sorted((RAW / "launch_claims").glob("*/*.json"))]
    failed = []
    recorded_steps = 0
    unknown_step_claims = []
    for path, claim in claims:
        status = claim.get("status")
        steps = claim.get("completed_steps")
        if status not in ("complete", "failed_technical"):
            raise RuntimeError(f"nonterminal or unknown claim status: {path}")
        if type(steps) is int and steps >= 0:
            recorded_steps += steps
        else:
            unknown_step_claims.append(str(path))
        if status == "failed_technical":
            failed.append({"path": str(path), "sha256": sha(path),
                           "logical_job": claim["logical_job"],
                           "attempt": int(claim["attempt"]),
                           "category": claim["category"],
                           "launch_slot": int(claim["launch_slot"]),
                           "environment_constructor_calls": int(claim.get("environment_constructor_calls", 0)),
                           "run_invocation_calls": int(claim.get("run_invocation_calls", 0)),
                           "recorded_completed_steps": steps,
                           "note": ("actual simulation step count unknown after external interruption"
                                    if claim["logical_job"] == "new_training_s11_e43" else None)})
            if (int(claim.get("environment_constructor_calls", 0)) == 1
                    and int(claim.get("run_invocation_calls", 0)) == 1
                    and steps == 0):
                unknown_step_claims.append(str(path))
    if len(failed) != counts["failed_technical_claims"]:
        raise RuntimeError("failed-claim count differs from original slot ledger")
    historical_manifest = HERE / "HISTORICAL_CONSUMPTION_RECONCILIATION.json"
    historical_audit = HERE / "EXECUTION_STOP_AUDIT.json"
    historical = read(historical_manifest)
    historical_details = read(historical_audit)
    historical_calls = historical["optimizer_calls"]
    if (historical_calls["total_evidence_supported_actual_calls"] != 48
            or sum(historical_calls[key]["count"] for key in
                   ("gen1_original_unpersisted_update", "gen1_recovery_attempts", "gen2_updates")) != 48):
        raise RuntimeError("historical optimizer evidence does not support 48 separate calls")
    historical_journals = []
    for item in historical_details["optimizer_hook_journals"].values():
        path = HERE.parents[2] / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise RuntimeError("historical optimizer hook journal changed")
        historical_journals.append({"path": str(path), "sha256": sha(path),
                                    "original_line_count": item["line_count"],
                                    "reconstructed_true_count": item["reconstructed_true_count"]})
    historical_steps = 0
    historical_claims = []
    for generation in historical_details["generations"].values():
        for claim_item in generation["claims"]:
            path = HERE.parents[2] / claim_item["claim_path"]
            original = read(path)
            if sha(path) != claim_item["claim_sha256"] or original.get("status") != "complete":
                raise RuntimeError("historical complete claim source changed")
            steps = original.get("completed_steps")
            if type(steps) is not int or steps < 0 or steps != claim_item["completed_steps"]:
                raise RuntimeError("historical completed-step source changed")
            historical_steps += steps
            historical_claims.append({"path": str(path), "sha256": sha(path),
                                      "completed_steps": steps})
    if len(historical_claims) != 16:
        raise RuntimeError("historical claim graph does not contain 16 original operations")
    optimizer, optimizer_sources = events("optimizer_step_calls.jsonl")
    direct_ppo, ppo_sources = events("ppo_update_calls.jsonl")
    checkpoints, checkpoint_sources = events("checkpoint_writes.jsonl")
    optimizer_indices = [int(item["call_index"]) for item in optimizer]
    ppo_indices = [int(item["call_index"]) for item in direct_ppo]
    if optimizer_indices != list(range(1, 557)) or ppo_indices != list(range(4, 97)):
        raise RuntimeError("actual cross-root optimizer/PPO hook event sequence has a gap or duplicate")
    early_commits = []
    for seed in (11, 22, 33):
        path = RAW / "checkpoints" / f"r40_batch_commit_s{seed}_b0.json"
        receipt = read(path)
        if (int(receipt["optimizer_steps"]) != 8
                or int(receipt["updates"]) != 1
                or int(receipt["episodes_done"]) != 4):
            raise RuntimeError("original committed first batch does not support three missing PPO hook entries")
        early_commits.append({"seed": seed, "path": str(path), "sha256": sha(path)})
    source_map = read(HERE / "ROUND58_GEN3_APPROVED_RESUME_SOURCES_V35.json")
    completed_training = []
    for seed, item in sorted(source_map["completed_training_sources"].items()):
        endpoint = HERE.parents[2] / item["checkpoint_path"]
        summary = HERE.parents[2] / item["summary_path"]
        commit = HERE.parents[2] / item["commit_path"]
        for path, key in ((endpoint, "checkpoint_sha256"), (summary, "summary_sha256"),
                          (commit, "commit_sha256")):
            if not path.is_file() or sha(path) != item[key]:
                raise RuntimeError(f"seed{seed} completed source {key} differs from approved map")
        completed_training.append({"seed": int(seed), "checkpoint": str(endpoint),
                                   "checkpoint_sha256": sha(endpoint),
                                   "summary": str(summary), "summary_sha256": sha(summary),
                                   "batch31_commit": str(commit), "commit_sha256": sha(commit),
                                   "optimizer_steps": int(item["optimizer_steps"])})
    protected = {}
    for name, expected in (
        ("ROUND46_CURRENT_TASK.json", "cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b"),
        ("ROUND46_R2_V6_STOP_STATE.json", "93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18")):
        path = BASE / name
        protected[name] = sha(path)
        if protected[name] != expected:
            raise RuntimeError(f"protected dirty file changed: {path}")
    output = {"kind": "round58-final-original-operation-accounting/1",
              "status": "all_1024_logical_jobs_complete",
              "raw_claim_slot_audit": counts,
              "quota": {"historical_consumed": 16, "normal_cap": 1024,
                        "reserve_cap": 24, "hard_cap": 1064,
                        "normal_used": counts["gen3_normal_slots"],
                        "reserve_used": counts["gen3_reserve_slots"],
                        "cumulative_attempt_slots": counts["cumulative_attempt_slots"],
                        "cumulative_actual_environment_constructions": counts["cumulative_actual_constructor_calls"],
                        "reserve_remaining": 24 - counts["gen3_reserve_slots"],
                        "hard_attempt_headroom": 1064 - counts["cumulative_attempt_slots"]},
              "failed_attempts": failed,
              "simulation_steps": {"source_scope": "gen3 master launch claims only; historical gen1/gen2 steps separate",
                                   "gen3_sum_recorded_completed_steps_known_lower_bound": recorded_steps,
                                   "unknown_actual_step_claims": unknown_step_claims,
                                   "not_claimed_exact_for_interrupted_attempt": True,
                                   "historical_gen1_gen2_recorded_completed_steps": historical_steps,
                                   "historical_claim_sources": historical_claims,
                                   "combined_known_completed_steps_lower_bound": historical_steps + recorded_steps},
              "optimizer": {"gen3_actual_hook_calls": len(optimizer), "journal_sources": optimizer_sources,
                            "effective_ep128_seed_steps": {str(item["seed"]): item["optimizer_steps"]
                                                         for item in completed_training},
                            "historical_gen1_gen2_supported_calls_separate": 48,
                            "historical_evidence": {"manifest": str(historical_manifest),
                                                    "manifest_sha256": sha(historical_manifest),
                                                    "audit": str(historical_audit),
                                                    "audit_sha256": sha(historical_audit),
                                                    "hook_journals": historical_journals,
                                                    "qualification": "48 evidence-supported actual calls, including derived/reconstructed historical evidence; not 48 independent raw hook lines"}},
              "ppo_updates": {"direct_hook_rows": len(direct_ppo), "journal_sources": ppo_sources,
                              "first_three_commits_without_original_direct_hook_rows": early_commits,
                              "effective_committed_batches": 96,
                              "direct_hook_gap_not_fabricated": True},
              "checkpoint_write_journal": {"raw_event_rows": len(checkpoints),
                                           "sources": checkpoint_sources,
                                           "interpretation": "write events, not distinct checkpoint files"},
              "completed_training_sources": completed_training,
              "protected_sha256": protected}
    (HERE / "ROUND58_FINAL_ACCOUNTING.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"quota": output["quota"], "failed_attempts": failed,
                      "steps": output["simulation_steps"],
                      "optimizer_hook_calls": len(optimizer),
                      "ppo_direct_hook_rows": len(direct_ppo),
                      "checkpoint_write_events": len(checkpoints)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
