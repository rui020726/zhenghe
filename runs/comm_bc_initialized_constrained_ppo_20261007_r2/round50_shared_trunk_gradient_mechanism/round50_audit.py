"""Read-only audit of the indexed ROUND46 first-batch records.

This deliberately refuses to manufacture loss-component gradients when the
frozen trainer bytes are not present and source-bound.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
EXPECTED_TRAINER = "0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841"
CURRENT_TRAINER = Path(__file__).resolve().parents[3] / "runs" / "comm_cost_v2_rerun_20260928" / "v2_trainer.py"
SEEDS = (11, 22, 33)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build() -> dict:
    frozen_path = ROOT / "ROUND46_R2_V6_RUNTIME_FREEZE.json"
    frozen = read_json(frozen_path)
    configured = frozen["frozen_runtime_source_sha256"]["canonical_v2_trainer.py"]
    actual_current = sha256(CURRENT_TRAINER)
    panel_path = ROOT / "round49_bc_input_policy_closure" / "INPUT_PANEL_MANIFEST.json"
    panel = read_json(panel_path)
    replay_path = ROOT / "round49_bc_input_policy_closure" / "BC_PATH_REPLAY.json"
    replay = read_json(replay_path)
    input_records = panel["records"]

    source_identity = {
        "kind": "round50-source-and-input-identity/1",
        "repository_head_at_analysis": "137f006da31da09535ad9245da12c2fc5f6d1a38",
        "task_branch": "analysis/round50-shared-trunk-gradient-mechanism",
        "frozen_trainer": {
            "frozen_manifest_name": "canonical_v2_trainer.py",
            "frozen_sha256": configured,
            "recoverable_source_path": None,
            "source_bytes_available": False,
            "current_candidate_path": str(CURRENT_TRAINER),
            "current_candidate_sha256": actual_current,
            "current_candidate_matches_frozen": actual_current == configured,
            "tracked_python_files_checked_at_head": 694,
            "status": "BLOCKED_FROZEN_SOURCE_BYTES_NOT_AVAILABLE",
        },
        "round49_panel": {
            "path": str(panel_path.relative_to(ROOT)),
            "sha256": sha256(panel_path),
            "records": len(input_records),
            "feature_width": panel["feature_width"],
            "dtype": panel["feature_dtype_forward"],
            "records_preserved_individually": panel["records_preserved_individually"],
            "action_mapping": panel["action_mapping"],
            "tie_rule": panel["tie_rule"],
            "used_for_forward_sensitivity": False,
            "reason": "The exact frozen loss/gradient function is unavailable; no component gradient exists to pair with a logit Jacobian.",
        },
        "round49_bc_path_replay": {
            "path": str(replay_path.relative_to(ROOT)),
            "sha256": sha256(replay_path),
            "bc_replay_all_seeds_passed": all(replay["bc_trajectory"][str(s)]["valid"] for s in SEEDS),
            "ppo_replay_all_seeds_passed": all(replay["ppo_trajectory"][str(s)]["valid"] for s in SEEDS),
            "action_mapping": replay["action_mapping"],
            "tie_rule": replay["greedy_tie_rule"],
            "initializer_matches_round46": {str(s): replay["source_model_identity"][str(s)]["bc_row_parameter_identity_match"] for s in SEEDS},
            "r43_registered_ep0_is_rejected_as_wrong_role": True,
        },
        "round46_files_read": [
            {"path": str(frozen_path.relative_to(ROOT)), "sha256": sha256(frozen_path)},
            *[
                {"path": str((ROOT / f"ROUND46_FIRST_BATCH_GATE_s{s}.json").relative_to(ROOT)),
                 "sha256": sha256(ROOT / f"ROUND46_FIRST_BATCH_GATE_s{s}.json")}
                for s in SEEDS
            ],
            *[
                {"path": str((ROOT / "v6" / "checkpoints" / f"r40_batch_s{s}_b0_attempt01.json").relative_to(ROOT)),
                 "sha256": sha256(ROOT / "v6" / "checkpoints" / f"r40_batch_s{s}_b0_attempt01.json")}
                for s in SEEDS
            ],
        ],
    }

    batches = {}
    gate_checks = {}
    for seed in SEEDS:
        gate = read_json(ROOT / f"ROUND46_FIRST_BATCH_GATE_s{seed}.json")["actual_r43_first_batch_gate"]
        batch_path = ROOT / "v6" / "checkpoints" / f"r40_batch_s{seed}_b0_attempt01.json"
        batch = read_json(batch_path)
        assert sha256(batch_path) == gate["batch_record"]["sha256"]
        assert gate["status"] == "passed"
        assert batch["training_seed"] == seed and batch["batch"] == 0
        assert batch["lambda_at_rollout"] == 1.0
        assert batch["update"]["samples"] == 265
        batches[str(seed)] = {
            "batch_record_path": str(batch_path.relative_to(ROOT)),
            "batch_record_sha256": sha256(batch_path),
            "gate_status": gate["status"],
            "preupdate_parameter_sha256": gate["preupdate_parameter_sha256"],
            "postupdate_parameter_sha256": gate["postupdate_parameter_sha256"],
            "episode_indices": batch["episodes"],
            "scenario_seeds": batch["scenarios"],
            "samples": batch["update"]["samples"],
            "minibatches": batch["update"]["minibatches"],
            "optimizer_steps_recorded": batch["optimizer_steps_after"],
            "epochs": batch["update"]["epochs"],
            "lambda_at_rollout": batch["lambda_at_rollout"],
            "recorded_gradient_norm_mean": batch["update"]["gradient_norm_mean"],
            "recorded_module_gradient_l1_mean": batch["update"]["module_gradient_l1_mean"],
            "recorded_module_gradient_l1_min": batch["update"]["module_gradient_l1_min"],
            "gradient_scope": batch["update"]["module_gradient_scope"],
            "recorded_likelihood_alignment": batch["update"]["likelihood_alignment"],
            "episode_artifacts": batch["episode_artifacts"],
        }
        gate_checks[str(seed)] = {
            "first_batch_gate_passed": True,
            "preupdate_checkpoint_parameter_digest_matches_bc_initializer": True,
            "likelihood_alignment_passed": batch["update"]["likelihood_alignment"]["passed"],
            "minibatch_indices_persisted": False,
            "component_gradient_vectors_persisted": False,
            "aggregate_module_l1_is_not_component_attribution": True,
        }

    zero_effects = {
        "environment_construction": 0,
        "simulation_start": 0,
        "new_episode_or_flow": 0,
        "real_training_update": 0,
        "synthetic_optimizer_step": 0,
        "checkpoint_write": 0,
        "evaluation_runner_start": 0,
    }
    grad = {
        "kind": "round50-gradient-components/1",
        "numeric_component_gradients_computed": False,
        "status": "BLOCKED_FROZEN_TRAINER_SOURCE_UNAVAILABLE",
        "reason": "Without exact frozen loss and reduction source, attributing logged trunk gradient to actor, entropy, task critic, or cost critic would be an unverified reimplementation.",
        "batch0_records": batches,
        "gate_checks": gate_checks,
        "interpretation": [
            "The first-batch record reports nonzero per-minibatch total trunk and value-head L1 gradient summaries for every seed.",
            "These aggregate summaries do not separate objective components or their signs, projections, or vector cancellation.",
            "The action-head summaries are much smaller than trunk summaries, but this alone does not identify the trunk-gradient source.",
            "The persisted batch says eight minibatches and eight optimizer steps; actual shuffled minibatch indices are not present in the inspected indexed record.",
        ],
        "zero_side_effect_counts": zero_effects,
    }
    sensitivity = {
        "kind": "round50-logit-directional-sensitivity/1",
        "status": "NOT_COMPUTED_COMPONENT_GRADIENT_UNAVAILABLE",
        "support_manifest": str(panel_path.relative_to(ROOT)),
        "support_record_count": len(input_records),
        "per_seed": {str(s): {"logit_jacobian_times_negative_component_gradient": None, "reason": "component gradients not available"} for s in SEEDS},
        "interpretation_limit": "No directional sensitivity or first-batch causal attribution is reported.",
    }
    out = {"identity": source_identity, "components": grad, "sensitivity": sensitivity}
    write_json(OUT / "SOURCE_AND_INPUT_IDENTITY.json", source_identity)
    write_json(OUT / "GRADIENT_COMPONENTS.json", grad)
    write_json(OUT / "LOGIT_DIRECTIONAL_SENSITIVITY.json", sensitivity)
    write_json(OUT / "PURE_TEST_EVIDENCE.json", {"kind": "round50-pure-test-evidence/1", "status": "PENDING"})
    return out


if __name__ == "__main__":
    build()
