"""Build the immutable ROUND43 one-variable configuration and source freeze."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = Path(__file__).resolve().parent
ROUND42_DIR = ROOT / "runs" / "comm_one_shot_gate_resume_20261003_v2"
ROUND42_CONFIG = ROUND42_DIR / "ROUND42_FROZEN_CONFIG.json"
ROUND42_INDEX = ROUND42_DIR / "ROUND42_CHECKPOINT_INDEX.json"
ROUND42_MANIFEST = ROUND42_DIR / "ROUND42_RAW_ARTIFACT_MANIFEST.json"
ROUND42_RESULT = ROUND42_DIR / "ROUND42_RESUME_FINAL_RESULT.json"
ROUND42_PAIRED = ROUND42_DIR / "ROUND42_PAIRED_RESULTS.json"
ROUND42_AUDIT = ROUND42_DIR / "ROUND42R_COMPLETION_AUDIT.json"
FREEZE_PATH = RUN_DIR / "ROUND43_FROZEN_CONFIG.json"
DIFF_PATH = RUN_DIR / "ROUND43_CONFIGURATION_DIFF.json"
RECOVERY_GATE_PATH = RUN_DIR / "ROUND43_RESUME_RECOVERY_GATE.json"
STATE_AUDIT_PATH = RUN_DIR / "ROUND43_SEED11_BATCH0_STATE_RECONCILIATION.json"
LEDGER_AUDIT_PATH = RUN_DIR / "ROUND43_LEDGER_RECONCILIATION.json"
BRIDGE_PATH = RUN_DIR / "runtime" / "ROUND43_RESUME_BRIDGE.json"
OLD_LEDGER_COPY = RUN_DIR / "runtime" / "ROUND43_EXECUTION_LEDGER_FREEZE1_AT_STOP.json"
OLD_PROGRESS_COPY = RUN_DIR / "runtime" / "ROUND43_EXECUTION_PROGRESS_FREEZE1.json"
OLD_RECEIPT_COPY = RUN_DIR / "runtime" / "ROUND43_PRELAUNCH_IDENTITY_FREEZE1.json"
OLD_PREFLIGHT_COPY = RUN_DIR / "runtime" / "ROUND43_PREFLIGHT_FREEZE1.json"
FAILURE_PATH = RUN_DIR / "failures" / "ROUND43_RUNCALL_RESUME_FAILURE_FREEZE1.json"

CODE_FILES = (
    "r42_run.py",
    "r42_versions.py",
    "r42_one_shot.py",
    "r42_analysis.py",
    "r43_preflight.py",
    "r43_analysis.py",
    "r43_execute.py",
    "r43_report.py",
    "r43_freeze.py",
    "r40_train_resume_adapter.py",
    "r43_call_counter_hooks.py",
    "test_r43_call_counter_hooks.py",
    "test_r43_resume_adapter.py",
    "r43_recovery_gate.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def canonical_sha(value) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _artifact_identity(path: Path) -> dict:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    frozen = read(FREEZE_PATH)
    old_freeze = read(ROUND42_CONFIG)
    checkpoint_index = read(ROUND42_INDEX)
    assert old_freeze["round42"]["objective"]["gae_lambda"] == 0.95
    assert old_freeze["round42"]["objective"]["gamma"] == 1.0
    assert old_freeze["round42"]["training_order_sha256"] == (
        frozen["round42"]["training_order_sha256"]
    )
    assert set(checkpoint_index["new_checkpoints"]) == {"11", "22", "33"}
    assert checkpoint_index["read_only_inspection"] is True

    round42_source_version = old_freeze["round42"]["training_version"]
    round43_training_version = frozen["round43"]["training_version"]
    old_objective = copy.deepcopy(old_freeze["round42"]["objective"])
    new_objective = copy.deepcopy(old_objective)
    new_objective["gae_lambda"] = 1.0

    # OLD for this ablation is ROUND42 NEW ep128. Keep the registered checkpoint
    # index identities and validate the corresponding bytes without rewriting them.
    old_models = {}
    new_initial_models = {}
    for seed in (11, 22, 33):
        seed_key = str(seed)
        old_record = checkpoint_index["new_checkpoints"][seed_key]["128"]
        old_path = Path(old_record["path"])
        assert old_path.is_file()
        assert sha256(old_path) == old_record["file_sha256_after"]
        ep0_record = checkpoint_index["new_checkpoints"][seed_key]["0"]
        ep0_path = Path(ep0_record["path"])
        assert ep0_path.is_file()
        assert sha256(ep0_path) == ep0_record["file_sha256_after"]
        old_models[seed_key] = {
            "path": str(old_path),
            "size_bytes": old_path.stat().st_size,
            "file_sha256": old_record["file_sha256_after"],
            "parameter_identity": old_record["parameter_identity"],
            "module_parameter_sha256": old_record["module_parameter_sha256"],
            "episodes_done": old_record["episodes_done"],
            "updates": old_record["updates"],
            "optimizer_steps": old_record["optimizer_steps"],
            "lambda_dual": old_record["lambda"],
            "training_version": old_record["training_version"],
            "reward_timing_version": old_record["reward_timing_version"],
            "comm_action_semantics": old_record["comm_action_semantics"],
            "read_only": True,
            "round42_arm": "NEW",
            "round43_role": "OLD",
            "protocol_adaptation": "none; same one-shot SEND/SKIP protocol",
        }
        new_initial_models[seed_key] = {
            "path": str(ep0_path),
            "file_sha256": ep0_record["file_sha256_after"],
            "parameter_sha256": ep0_record["parameter_identity"]["parameter_sha256"],
            "module_parameter_sha256": ep0_record["module_parameter_sha256"],
            "episodes_done": 0,
            "updates": 0,
            "optimizer_steps": 0,
            "lambda_dual": ep0_record["lambda"],
            "training_version": ep0_record["training_version"],
            "read_only_identity_only": True,
        }

    frozen["round42"]["source_training_version"] = round42_source_version
    frozen["round42"]["training_version"] = round43_training_version
    frozen["round42"]["objective"] = new_objective
    frozen["round42"]["old_models"] = old_models
    frozen["round42"]["new_initial_models"] = new_initial_models
    frozen["round42"]["plan"] = {
        "reused_immediate_reference": {"cells": 64, "scheduled_new_launches": 0},
        "new_training": {"seeds": 3, "episodes_each": 128, "total": 384},
        "round42_new_ep128_reused_as_old": {
            "seeds": 3, "modes": ["greedy", "sampled"], "scenes": 16,
            "total": 96,
        },
        "new_final_evaluation": {
            "seeds": 3, "modes": ["greedy", "sampled"], "scenes": 16,
            "total": 96,
        },
        "reused_rules": {"arms": ["late_discovery_gate", "always_hold"],
                         "scenes": 16, "cells": 32, "scheduled_new_launches": 0},
        "baseline_repair": {"maximum": 96, "scheduled": 0},
        "normal_total": 480,
        "technical_reserve_max": 24,
        "hard_cap": 600,
        "new_streams": 0,
        "confirmation_scenes": 0,
    }

    # The only cross-freeze continuation accepted by ROUND43 is the exact ep4/batch0
    # committed state that stopped before episode 4 acquired a launch claim.
    recovery_gate = read(RECOVERY_GATE_PATH)
    state_audit = read(STATE_AUDIT_PATH)
    ledger_audit = read(LEDGER_AUDIT_PATH)
    prior_ledger = read(OLD_LEDGER_COPY)
    prior_progress = read(OLD_PROGRESS_COPY)
    live_progress = read(RUN_DIR / "runtime" / "ROUND43_EXECUTION_PROGRESS.json")
    prior_receipt = read(OLD_RECEIPT_COPY)
    if recovery_gate.get("status") != "passed" or state_audit.get("status") != "passed" \
            or ledger_audit.get("status") != "passed":
        raise RuntimeError("ROUND43 exact-resume recovery evidence is incomplete")
    if (sha256(RUN_DIR / "runtime" / "ROUND43_EXECUTION_LEDGER.json")
            != sha256(OLD_LEDGER_COPY)):
        raise RuntimeError("ROUND43 pre-recovery ledger changed before it was preserved")
    if (sha256(RUN_DIR / "runtime" / "ROUND43_EXECUTION_PROGRESS.json")
            != sha256(OLD_PROGRESS_COPY)):
        raise RuntimeError("ROUND43 pre-recovery progress changed before it was preserved")
    if live_progress.get("prelaunch_head") != prior_receipt["head"] \
            or prior_progress.get("prelaunch_head") != prior_receipt["head"]:
        raise RuntimeError("ROUND43 prior progress and source receipt disagree")

    source_code_sha256 = {name: sha256(RUN_DIR / name) for name in CODE_FILES}
    if recovery_gate.get("source_code_sha256_for_next_freeze") != source_code_sha256:
        raise RuntimeError("ROUND43 recovery gate was not run against the final continuation source")

    batch_record = read(RUN_DIR / "checkpoints" / "r40_batch_s11_b0_attempt01.json")
    logical_jobs = {str(row["logical_job"]) for row in prior_ledger["claims"]}
    artifact_paths = {
        "ROUND43_FIRST_BATCH_GATE_SEED11.json",
        "ROUND43_SEED11_BATCH0_STATE_RECONCILIATION.json",
        "ROUND43_LEDGER_RECONCILIATION.json",
        "ROUND43_RESUME_RECOVERY_GATE.json",
        "training_reports/ROUND43_SEED11_FIRST4.json",
        "runtime/ROUND43_EXECUTION_LEDGER.json",
        "runtime/ROUND43_EXECUTION_LEDGER_FREEZE1_AT_STOP.json",
        "runtime/ROUND43_EXECUTION_PROGRESS_FREEZE1.json",
        "runtime/ROUND43_PRELAUNCH_IDENTITY_FREEZE1.json",
        "runtime/ROUND43_PREFLIGHT_FREEZE1.json",
        "runtime/checkpoint_writes.jsonl",
        "runtime/optimizer_step_calls.jsonl",
        "runtime/ppo_update_calls.jsonl",
        "failures/ROUND43_RUNCALL_RESUME_FAILURE_FREEZE1.json",
        "checkpoints/r40_batch_commit_s11_b0.json",
        "checkpoints/r40_batch_s11_b0_attempt01.json",
        "checkpoints/v2_train_s11.pt",
        "checkpoints/v2_train_s11_after_b0.pt",
        "checkpoints/v2_train_s11_before_b0.pt",
        "checkpoints/v2_train_s11_ep0.pt",
    }
    for episode in batch_record["episode_artifacts"]:
        for key in ("row_path", "tape_path"):
            artifact_path = Path(episode[key]).resolve()
            artifact_paths.add(artifact_path.relative_to(RUN_DIR.resolve()).as_posix())
    for path in (RUN_DIR / "launch_claims").rglob("attempt_*.json"):
        claim = read(path)
        if claim.get("logical_job") in logical_jobs:
            artifact_paths.add(path.relative_to(RUN_DIR).as_posix())
    for path in (RUN_DIR / "training_progress").glob("seed11_*.json"):
        artifact_paths.add(path.relative_to(RUN_DIR).as_posix())
    immutable_artifacts = []
    for relative_path in sorted(artifact_paths):
        path = RUN_DIR / relative_path
        if not path.is_file():
            raise RuntimeError(f"ROUND43 exact-resume evidence is missing: {relative_path}")
        immutable_artifacts.append({
            "relative_path": relative_path,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    bridge = {
        "kind": "round43-exact-resume-bridge/1",
        "status": "passed",
        "prior_prelaunch_head": prior_receipt["head"],
        "prior_branch": prior_receipt["branch"],
        "target_source_code_sha256": source_code_sha256,
        "mechanical_change": "ROUND42-validated training-entry adapter moves only the parameter_identity import to train_seed function entry; counter ledger flush is bookkeeping-only",
        "training_version": round43_training_version,
        "resume_cursor": {
            "training_seed": 11,
            "episodes_done": int(state_audit["state_cursor"]["episodes_done"]),
            "updates": int(state_audit["state_cursor"]["updates"]),
            "optimizer_steps": int(state_audit["state_cursor"]["optimizer_steps"]),
            "lambda_dual": float(state_audit["state_cursor"]["lambda"]),
            "next_episode_index": int(state_audit["state_cursor"]["episodes_done"]),
        },
        "reconciled_counts_at_pause": ledger_audit["corrected_cumulative_snapshot"],
        "prior_progress_sha256": sha256(RUN_DIR / "runtime" / "ROUND43_EXECUTION_PROGRESS.json"),
        "prior_ledger_snapshot_sha256": sha256(OLD_LEDGER_COPY),
        "state_audit": {"path": STATE_AUDIT_PATH.name, "sha256": sha256(STATE_AUDIT_PATH)},
        "ledger_audit": {"path": LEDGER_AUDIT_PATH.name, "sha256": sha256(LEDGER_AUDIT_PATH)},
        "recovery_gate": {"path": RECOVERY_GATE_PATH.name, "sha256": sha256(RECOVERY_GATE_PATH)},
        "failed_resume_invocation": {
            "path": FAILURE_PATH.relative_to(RUN_DIR).as_posix(),
            "sha256": sha256(FAILURE_PATH),
            "new_claims": 0,
            "new_environment_constructors": 0,
            "new_optimizer_updates": 0,
        },
        "normal_plan": {"total": 480, "used_before_resume": 4,
                        "remaining": 476, "technical_reserve": 24, "hard_cap": 600},
        "completed_seed11_first_four_and_batch0_are_not_replayed": True,
        "round42_trained_checkpoints_accepted_for_training_resume": False,
        "immutable_artifacts": immutable_artifacts,
    }
    write(BRIDGE_PATH, bridge)

    config_diff = {
        "kind": "round43-single-variable-configuration-diff/2",
        "scope": "one-shot-send-skip full-trace GAE trace ablation",
        "training_version_change": {
            "old": round42_source_version,
            "new": round43_training_version,
            "checkpoint_policy": "ROUND42 trained states are rejected for ROUND43 training restore; ROUND42 NEW ep128 is read-only OLD",
        },
        "only_research_configuration_change": {
            "path": "shared_gae_lambda",
            "old": old_objective["gae_lambda"],
            "new": new_objective["gae_lambda"],
            "used_by": ["task_advantage_gae", "positive_cost_advantage_gae"],
            "actual_call_path": "v2_trainer.batch_samples -> common.GAE_LAMBDA -> cm.gae for task and positive cost",
        },
        "objective_and_ppo": {
            "old": old_objective,
            "new": new_objective,
            "old_ppo_hyperparameters": old_freeze["round42"]["ppo_hyperparameters"],
            "new_ppo_hyperparameters": old_freeze["round42"]["ppo_hyperparameters"],
            "actor_objective": "lambda_dual*A_task - A_positive_cost",
            "critic_target_consequence": "GAE=1 changes A+V critic targets as a direct consequence of the trace intervention; no extra loss or weight change",
        },
        "preserved_scientific_configuration": {
            key: copy.deepcopy(old_freeze["round42"][key])
            for key in (
                "runner_version", "reward_timing_version", "comm_action_eligibility",
                "comm_action_semantics", "training_seeds", "episodes_per_seed",
                "episodes_per_update", "batches_per_seed", "training_order",
                "training_order_sha256", "training_scenes", "development_scenes",
                "steps_per_episode", "world_config_source", "candidate_order",
                "actor_feature_width", "row_filtering",
                "frozen_search_allocation_checkpoints",
            )
        },
        "initialization_and_comparison": {
            "new_seed_initial_state": "fresh same-seed controller plus empty Adam; parameters, initial optimizer state and lambda_dual checked against registered ROUND42 ep0; registered full RNG state restored after construction before first rollout",
            "registered_round42_ep0": frozen["round43"]["registered_ep0_inputs"],
            "old_model": "ROUND42 NEW ep128, read-only, same one-shot SEND/SKIP protocol",
            "old_model_identities": old_models,
            "new_initial_parameter_identities": new_initial_models,
            "no_round42_ep64_or_ep128_hot_start": True,
        },
        "reused_baselines": {
            "round42_artifacts": {
                "completion_audit": _artifact_identity(ROUND42_AUDIT),
                "final_result": _artifact_identity(ROUND42_RESULT),
                "paired_results": _artifact_identity(ROUND42_PAIRED),
                "checkpoint_index": _artifact_identity(ROUND42_INDEX),
                "raw_artifact_manifest": _artifact_identity(ROUND42_MANIFEST),
            },
            "immediate_reference_cells": 64,
            "rule_cells": 32,
            "round42_new_ep128_evaluation_cells": 96,
            "baseline_repair_cells_scheduled": 0,
            "development_scenes_are_exploratory_and_already_used": True,
        },
        "launch_quota": {
            "normal_training": 384,
            "normal_new_final_evaluation": 96,
            "normal_total": 480,
            "reserve": 24,
            "hard_cap": 600,
        },
        "claim_limit": "GAE=1 is a method ablation, not a correction of an old mathematical error or proof that trace attenuation alone caused the previous result.",
    }

    diff_sha = canonical_sha(config_diff)
    identity = frozen["execution_identity"]
    identity["source_code_sha256"] = source_code_sha256
    identity["inherited_dependency_sha256"] = copy.deepcopy(
        frozen["round42"]["freeze_identity"]["active_dependency_sha256"]
    )
    identity["prelaunch_head"] = None
    identity["execution_code_version"] = "round43-resume-adapter-and-ledger-reconciliation/1"
    identity["resume_bridge"] = {
        "path": "runtime/ROUND43_RESUME_BRIDGE.json",
        "sha256": sha256(BRIDGE_PATH),
    }
    identity["resume_recovery_gate"] = {
        "path": "ROUND43_RESUME_RECOVERY_GATE.json",
        "sha256": sha256(RECOVERY_GATE_PATH),
    }
    identity["prior_prelaunch_head"] = prior_receipt["head"]
    freeze_identity = frozen["round42"]["freeze_identity"]
    freeze_identity["configuration_diff_sha256"] = diff_sha
    freeze_identity["round43_gae_lambda"] = 1.0
    freeze_identity["source_round42_training_version"] = round42_source_version
    freeze_identity["round42_reuse_source_sha256"] = {
        "completion_audit": sha256(ROUND42_AUDIT),
        "final_result": sha256(ROUND42_RESULT),
        "paired_results": sha256(ROUND42_PAIRED),
        "checkpoint_index": sha256(ROUND42_INDEX),
        "raw_artifact_manifest": sha256(ROUND42_MANIFEST),
    }
    freeze_identity.pop("freeze_sha256", None)
    freeze_identity["freeze_sha256"] = canonical_sha(freeze_identity)
    frozen["status"] = "prelaunch_frozen"

    write(DIFF_PATH, config_diff)
    write(FREEZE_PATH, frozen)
    print(json.dumps({
        "status": frozen["status"],
        "old_gae_lambda": old_objective["gae_lambda"],
        "new_gae_lambda": new_objective["gae_lambda"],
        "normal_total": frozen["round42"]["plan"]["normal_total"],
        "technical_reserve": frozen["round42"]["plan"]["technical_reserve_max"],
        "hard_cap": frozen["round42"]["plan"]["hard_cap"],
        "source_code_sha256": identity["source_code_sha256"],
        "configuration_diff_sha256": diff_sha,
        "freeze_sha256": freeze_identity["freeze_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
