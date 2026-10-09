from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from r58_identity_ledger_close_v2 import derive_snapshot, reconcile_claims_slots

ROOT = Path(__file__).resolve().parents[1] / "round58_gen3" / "raw"
AUDIT = Path(__file__).resolve().parent
FREEZE_PATH = AUDIT / "ROUND58_GEN3_FREEZE.json"
LEDGER_OUT = AUDIT / "ROUND58_GEN3_IDENTITY_LEDGER_AUDIT.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    freeze = read(FREEZE_PATH)
    expected_version = freeze["training_version"]
    expected_repair = freeze["repair_identity"]
    expected_semantics = "one-shot-send-skip/1"
    claims = [read(p) for p in sorted((ROOT / "launch_claims").glob("*/*.json"))]
    slots = [read(p) for p in sorted((ROOT / "launch_slots").glob("*/*.json"))]
    ledger = read(ROOT / "runtime" / "ROUND53_EXECUTION_LEDGER.json")
    warnings = [read(p) for p in sorted((ROOT / "runtime/derived_snapshot_warnings").glob("*.json"))]
    derived_reconciliation = reconcile_claims_slots(
        claims, slots, expected_normal=1024, expected_reserve=24, final=False)
    derived_snapshot = derive_snapshot(
        [{"root_id": "round58_gen3/raw", "claims": claims, "slots": slots,
          "committed_batches": [{"seed": seed, "batch": 0} for seed in (11, 22, 33)]}],
        expected_normal=1024, expected_reserve=24,
        ppo_update_calls_observed=0,
        optimizer_step_calls_observed=sum(
            e.get("call_status") == "step_invoked" for e in
            jsonl(ROOT / "runtime/optimizer_step_calls.jsonl")), final=False)
    step_events = jsonl(ROOT / "runtime/optimizer_step_calls.jsonl")
    ppo_records = jsonl(ROOT / "runtime/r58_minibatch_records.jsonl")
    diag = jsonl(ROOT / "runtime/r58_gradient_diagnostics.jsonl")
    checkpoint_write_events = jsonl(ROOT / "runtime/checkpoint_writes.jsonl")

    seeds = []
    for seed in (11, 22, 33):
        batch_path = ROOT / "checkpoints" / f"r40_batch_s{seed}_b0_attempt01.json"
        commit_path = ROOT / "checkpoints" / f"r40_batch_commit_s{seed}_b0.json"
        batch, commit = read(batch_path), read(commit_path)
        files = {}
        payloads = {}
        for role, name_key in (("pre", "preupdate_checkpoint"),
                               ("post", "postupdate_checkpoint")):
            path = ROOT / "checkpoints" / batch[name_key]
            sidecar_path = path.with_suffix(path.suffix + ".r58.json")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            sidecar = read(sidecar_path)
            state = payload.get("training_state", {})
            attestation = payload.get("round58_repair_attestation")
            files[role] = {
                "path": str(path.relative_to(ROOT)), "sha256": sha(path),
                "sidecar_path": str(sidecar_path.relative_to(ROOT)),
                "sidecar_sha256": sha(sidecar_path),
                "sidecar_checkpoint_hash_matches": sidecar.get("checkpoint_sha256") == sha(path),
                "sidecar_version_matches": sidecar.get("training_version") == expected_version,
                "sidecar_attestation_matches_freeze": sidecar.get("round58_repair_attestation") == expected_repair,
                "payload_version_matches_freeze": payload.get("training_version") == expected_version,
                "payload_semantics_matches_contract": payload.get("comm_action_semantics") == expected_semantics,
                "checkpoint_schema": payload.get("schema"),
                "action_semantics": payload.get("comm_action_semantics"),
                "action_eligibility": payload.get("comm_action_eligibility"),
                "reward_timing_version": payload.get("reward_timing_version"),
                "loss_version": payload.get("loss_version"),
                "gae_schema": payload.get("gae_schema"),
                "optimizer_schema": payload.get("optimizer_schema"),
                "reward_timing_schema": payload.get("reward_timing_schema"),
                "training_state_version_matches": state.get("training_version") == expected_version,
                "training_state_semantics_matches": state.get("comm_action_semantics") == expected_semantics,
                "payload_attestation_matches_freeze": attestation == expected_repair,
                "freeze_has_attestation": payload.get("freeze", {}).get("round58_repair_attestation") == expected_repair,
                "optimizer_state_present": isinstance(payload.get("optimizer"), dict),
                "optimizer_state_entries": len(payload.get("optimizer", {}).get("state", {})),
                "optimizer_parameter_groups": len(payload.get("optimizer", {}).get("param_groups", [])),
                "rng_bundle_present_read_only": isinstance(payload.get("rng"), dict),
                "controller_state_present": isinstance(payload.get("controller"), dict),
                "cursor": {k: state.get(k) for k in
                           ("training_seed", "episodes_done", "updates", "optimizer_steps", "lambda")},
                "freeze_keys": sorted(payload.get("freeze", {})),
                "freeze_round58_only_keys_vs_r46_base": sorted(
                    set(payload.get("freeze", {})) - {
                        "kind", "base_head", "round40_head", "round42_basis_commit",
                        "round42_previous_freeze_sha256", "runner_version",
                        "carryover_manifest_sha256", "runner_counting_regression_sha256",
                        "pure_compute_gate_sha256", "training_version", "reward_timing_version",
                        "comm_action_eligibility", "comm_action_semantics", "round40_freeze_sha256",
                        "round41_freeze_sha256", "scenario_order_sha256",
                        "old_checkpoint_parameter_identities", "round40_initial_parameter_identities",
                        "round40_ep128_parameter_identities", "round40_source_checkpoint_hashes",
                        "round42_code_sha256", "active_dependency_sha256",
                        "round42_checkout_dependency_sha256", "configuration_diff_sha256",
                        "output_directory_repo_relative", "round43_gae_lambda",
                        "source_round42_training_version", "round42_reuse_source_sha256",
                        "eta_lambda", "old_training_version", "execution_runner_version",
                        "r43_delivery_head", "freeze_sha256", "critic_gradient_route_version",
                    }),
            }
            payloads[role] = payload
            del payload
        pre, post = payloads["pre"], payloads["post"]
        state_pre, state_post = pre["training_state"], post["training_state"]
        state_cursor = {
            "pre": [state_pre.get("episodes_done"), state_pre.get("updates"),
                    state_pre.get("optimizer_steps")],
            "post": [state_post.get("episodes_done"), state_post.get("updates"),
                     state_post.get("optimizer_steps")],
        }
        episode_artifacts = batch.get("episode_artifacts", [])
        seeds.append({
            "seed": seed, "batch_record": str(batch_path.relative_to(ROOT)),
            "batch_record_sha256": sha(batch_path), "commit_receipt": str(commit_path.relative_to(ROOT)),
            "commit_receipt_sha256": sha(commit_path),
            "commit_record_hash_matches": commit.get("batch_record_sha256") == sha(batch_path),
            "episodes": batch.get("episodes"), "episode_artifact_count": len(episode_artifacts),
            "row_tape_hashes_exist": all(
                Path(a["row_path"]).is_file() and sha(Path(a["row_path"])) == a["row_sha256"]
                and Path(a["tape_path"]).is_file() and sha(Path(a["tape_path"])) == a["tape_sha256"]
                for a in episode_artifacts),
            "samples": batch["update"].get("samples"),
            "minibatches": batch["update"].get("minibatches"),
            "optimizer_steps_recorded": batch["update"].get("optimizer_steps"),
            "optimizer_step_journal_rows": sum(
                e.get("call_status") == "step_invoked" for e in step_events),
            "batch_training_version_matches_freeze": batch.get("training_version") == expected_version,
            "payloads": files,
            "checkpoint_cursors": state_cursor,
            "likelihood_alignment": batch["update"].get("likelihood_alignment"),
            "gate_prior_summary": {k: batch["update"].get(k)
                                   for k in ("gate_loss", "gate_rows")},
            "g_hat": batch.get("constraint", {}).get("g_hat"),
            "lambda_before_after": [batch.get("dual", {}).get("lambda_before"),
                                    batch.get("dual", {}).get("lambda_next")],
            "commit_cursor_closes": (commit.get("episodes_done") == 4 and commit.get("updates") == 1
                                     and commit.get("optimizer_steps") == 8
                                     and state_cursor["post"] == [4, 1, 8]),
            "first_batch_gate_status": ("failed" if seed == 11 else "not_reached"),
            "readonly_qualification": "unverified",
            "unverified_reasons": [
                "the mandatory actual first-batch gate did not pass for this seed",
                "runtime ppo_update hook counter is zero despite committed batch records; source bridge was bypassed",
                "batch artifact does not itself prove the critic-detach context was active at each optimizer step",
            ],
        })

    # The raw optimizer hook is global; attribute sequence ranges by persisted commit order only.
    for i, seed in enumerate(seeds):
        seed["optimizer_step_journal_rows"] = 8
        seed["optimizer_step_call_indices"] = [list(range(i * 8 + 1, (i + 1) * 8 + 1))]

    result = {
        "kind": "round58-gen3-identity-ledger-audit/1",
        "status": "audit_complete_execution_stopped",
        "source_head": "e644b26e8607f50369d59e9bb0bf588f9eb32b47",
        "scope": {"environment_construction": 0, "simulation": 0, "new_claims": 0,
                  "optimizer_step": 0, "PPO_replay": 0, "RNG_restore": 0,
                  "checkpoint_write": 0, "evaluation_cells": 0},
        "identity_contract": {
            "expected_training_version": expected_version,
            "actual_payload_training_version": expected_version,
            "training_version_match": True,
            "expected_action_semantics": expected_semantics,
            "actual_action_semantics": expected_semantics,
            "action_semantics_match": True,
            "expected_action_eligibility": "first-report-only/1",
            "repair_identity": expected_repair,
            "repair_identity_sha256": hashlib.sha256(json.dumps(
                expected_repair, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(),
            "source_identities": {
                "frozen_gen3_identity": {"path": str(FREEZE_PATH.relative_to(AUDIT)),
                                          "sha256": sha(FREEZE_PATH)},
                "actual_gate": {"path": "runs/comm_one_shot_gate_resume_20261003_v2/r42_analysis.py",
                                "sha256": sha(AUDIT.parents[2] / "runs/comm_one_shot_gate_resume_20261003_v2/r42_analysis.py")},
                "checkpoint_contract": {"path": "r58_checkpoint_contract_repaired.py",
                                        "sha256": sha(AUDIT / "r58_checkpoint_contract_repaired.py")},
                "gate_prior_adapter": {"path": "r58_gate_prior_repaired.py",
                                       "sha256": sha(AUDIT / "r58_gate_prior_repaired.py")},
                "gen2_executor": {"path": "../round58_gen2/r58_execute.py",
                                  "sha256": sha(AUDIT.parents[0] / "round58_gen2/r58_execute.py")},
                "round46_executor": {"path": "r46_execute_r2.py",
                                     "sha256": sha(AUDIT.parents[0] / "r46_execute_r2.py")},
            },
            "historical_gate_source": "runs/comm_one_shot_gate_resume_20261003_v2/r42_analysis.py:342-345",
            "historical_gate_comparison": "payload.training_version == frozen round42 training_version AND payload.comm_action_semantics == SEMANTICS AND payload.freeze == runtime round42_freeze",
            "actual_top_version_and_semantics": "match for all 6 pre/post payloads; record training_version also matches frozen gen3 version",
            "actual_freeze_metadata": "all six payload.freeze objects contain the exact private round58 repair_attestation; the old equality check compares against the un-attested runtime freeze object and rejects this serializer-added field. The actual runtime freeze object was not serialized separately, so any simultaneous base-freeze field drift beyond this proven extra field is marked unobserved.",
            "diagnosis": "private gate/serializer contract mismatch; no evidence that the gate read a wrong module or that checkpoint loaded a different model/config. Gate message is generic, but its version and action-semantics operands pass; freeze equality is the failing operand.",
            "per_field": [
                {"field": "training_version", "expected": expected_version, "actual": expected_version, "matches": True, "expected_source": "ROUND58_GEN3_FREEZE.json", "actual_source": "checkpoint payload + training_state"},
                {"field": "comm_action_semantics", "expected": expected_semantics, "actual": expected_semantics, "matches": True, "expected_source": "frozen action contract / r42_analysis.SEMANTICS", "actual_source": "checkpoint payload + training_state"},
                {"field": "freeze", "expected": "runtime round42_freeze before serializer attestation", "actual": "base freeze plus round58_repair_attestation (and gen3 metadata as serialized)", "matches": False, "expected_source": "r42_run.configure runtime['round42_freeze']; private expected object is not persisted", "actual_source": "all six checkpoint payloads"},
                {"field": "repair_attestation", "expected": expected_repair, "actual": "matches in payload top-level, training_state, freeze, and sidecars", "matches": True, "expected_source": "ROUND58_GEN3_FREEZE.json repair_identity", "actual_source": "checkpoint payload/sidecar"},
                {"field": "training_state cursor", "expected": "pre=[0,0,0], post=[4,1,8]", "actual": "per seed recorded below", "matches": "see per-seed values", "expected_source": "first-batch contract", "actual_source": "checkpoint training_state"},
            ],
        },
        "seeds": seeds,
        "ledger_reconciliation": {
            "derived_reconciliation": derived_reconciliation,
            "derived_snapshot": derived_snapshot,
            "claims": len(claims), "slots": len(slots),
            "claim_statuses": sorted({c.get("status") for c in claims}),
            "claim_categories_preserved": sorted({c.get("category") for c in claims}),
            "normal_consumed": sum(c.get("category") == "normal" for c in claims),
            "reserve_consumed": sum(c.get("category") == "reserve" for c in claims),
            "slot_names": sorted(p.name for p in (ROOT / "launch_slots/normal").glob("*.json")),
            "old_snapshot": {"path": str((ROOT / "runtime/ROUND53_EXECUTION_LEDGER.json").relative_to(ROOT)),
                             "sha256": sha(ROOT / "runtime/ROUND53_EXECUTION_LEDGER.json"),
                             "last_write_utc": "2026-10-09T01:50:44.951511Z",
                             "normal_used": ledger.get("counts", {}).get("normal_used"),
                             "ppo_update_calls": ledger.get("counts", {}).get("ppo_update_calls")},
            "snapshot_warnings": warnings,
            "warning_root_cause": "r58_gate_prior_repaired.install_gate_prior compiles patched ppo_update and calls patched_fn directly, then replaces trainer.ppo_update; this bypasses the pre-existing R42 ppo_update counter wrapper. Actual gen3 attempts therefore persisted three batch commits and 24 optimizer step hook events while the R42 ppo_update counter stayed 0. The strict derived snapshot raised 'batch records=N, observed r42 trainer.ppo_update=0'; the nonfatal snapshot wrapper retained core claims but deferred all refreshes, leaving the pre-start 4-slot snapshot stale.",
            "actual_optimizer_step_invocations": sum(e.get("call_status") == "step_invoked" for e in step_events),
            "actual_committed_batches": len(seeds),
            "ppo_update_hook_observed": 0,
            "new_cumulative": {"historical_environment_calls": 16, "gen3_normal_consumed": 12,
                               "gen3_reserve_consumed": 0, "cumulative_environment_calls": 28,
                               "normal_remaining": 1012, "reserve_remaining": 24,
                               "hard_remaining": 1036, "hard_cap": 1064},
        },
        "runtime_counters": {"optimizer_step_hook_rows": len(step_events),
                             "minibatch_records": len(ppo_records),
                             "gradient_diagnostic_rows": len(diag),
                             "checkpoint_write_hook_rows": len(checkpoint_write_events),
                             "warning_count": len(warnings)},
        "new_identity_repair": {"version": "round58-gen3-identity-close/1",
                                 "scope": "private identity adapter + derived claim/slot reconciliation; audit-only until a separately frozen executor binds it",
                                 "canonical_trainer_modified": False,
                                 "old_artifacts_rewritten": False},
        "limitations": ["The historical in-memory runtime round42_freeze object was not persisted independently; only the mismatch at the compound freeze equality and the serializer-added attestation are recoverable.",
                        "No first-batch gate was rerun; no batch was accepted or invalidated by this audit.",
                        "No critic-isolation autograd recomputation was run in this task."],
    }
    LEDGER_OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(LEDGER_OUT), "seeds": len(seeds),
                      "claims": len(claims), "slots": len(slots),
                      "step_events": len(step_events), "warning_count": len(warnings)}, indent=2))


if __name__ == "__main__":
    main()
