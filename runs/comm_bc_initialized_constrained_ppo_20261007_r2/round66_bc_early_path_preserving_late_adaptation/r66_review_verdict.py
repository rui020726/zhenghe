import json

BASE = r"D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2\round66_bc_early_path_preserving_late_adaptation"
pd = json.load(open(BASE + r"\raw\runtime\ROUND66_REVIEW_PARAM_DIFF.json", encoding="utf-8"))
proj = json.load(open(BASE + r"\raw\runtime\ROUND66_REVIEW_NUMERIC_PROJECTION.json", encoding="utf-8"))

def seed_mod(seed, mod, field):
    return pd["seeds"][str(seed)]["per_module"][mod][field]

mv = {
  "kind": "round66-final-review-verdict/1",
  "task_id": "ROUND66-BC-EARLY-PATH-PRESERVING-LATE-ADAPTATION",
  "branch": "exp/round66-bc-early-path-preserving-late-adaptation",
  "audited_code_head": "39ac0a41b932ebccfb744cea365cdb9dcf0fce6a",
  "delivery_head": "254ea798072646094dd05a55af53a58a85fb9824",
  "review_scope": "zero-environment final review: wording errata + read-only param diff + numeric projection + synthetic-update accounting audit. No new environment/training/optimizer/RNG/checkpoint writes.",
  "zero_side_effects": {
    "new_environment_constructions": 0,
    "real_optimizer_updates": 0,
    "synthetic_optimizer_updates": 0,
    "ppo_updates": 0,
    "rng_restores": 0,
    "checkpoint_writes": 0,
    "read_only_checkpoint_loads": 6,
    "read_only_json_reads": 2
  },
  "errata": {
    "E1_d1_wording": {
      "original_claim": "first detached launch was 'hard-killed' inside the environment constructor (ROUND66_FINAL_REPORT.md section 2; ROUND66_DEVIATION_REGISTER.json D1)",
      "correction": "D1 is an interrupted/terminated attempt; the cause (incl. whether 'hard-killed') and the executed step count are UNKNOWN/unproven. Only artifact-level facts hold (no exit record; frozen 1480-byte gate log, no traceback; claim with constructor_called_at_unix before the second launcher and completed_steps=0/simulation_started=false; process gone before the second launch).",
      "unchanged": "the deviation register's own not_proven field already says positive termination by process handle is NOT established; the +1 unregistered construction, 2 real constructions, no-refund, reserve-capped-39 accounting and the deterministic immediate value's eligibility all stand unchanged."
    },
    "E2_new_eq_bc_wording": {
      "original_claim": "'NEW == BC is behavioural equivalence of measured outcomes' (ROUND66_FINAL_REPORT.md section 6; ROUND66_MACHINE_VERDICT.json limitations)",
      "correction": "equivalence is verified ONLY on measured outcome fields: on-time task count, billed bytes, and exact on-time target-ID sets (per-scene 32/32 for s33 greedy; task delta 0, cost delta 0, zero target-set differences across all seeds x modes). No complete action-by-action/per-message behaviour check was performed, so no general behavioural-equivalence claim is made.",
      "reinforced_by": "read-only param diff shows NEW != BC at the parameter level (trainable modules moved; see param_diff), so the equivalence is strictly outcome-level on this single development block."
    }
  },
  "param_diff_ep0_to_ep128": {
    "method": "read-only torch.load (cpu) of R60 ep0 (v2_train_s{seed}_before_b0.pt, file sha verified) and R66 ep128 (v2_train_s{seed}_ep128.pt, sha matches checkpoint index); per-module bit-identity + L2 norm; no writes",
    "file_shas_verified": {s: {"ep0_match": pd["seeds"][str(s)]["ep0_file_sha_matches_expected"],
                                 "ep128_match": pd["seeds"][str(s)]["ep128_file_sha_matches_index"]} for s in (11,22,33)},
    "frozen_path_bit_identical_all_seeds": all(pd["seeds"][str(s)]["frozen_trunk_plus_early"]["all_bit_identical"] for s in (11,22,33)),
    "trainable_moved_all_seeds": all(not pd["seeds"][str(s)]["trainable_late_plus_critics"]["all_bit_identical"] for s in (11,22,33)),
    "l2_diff_norm": {
      str(s): {
        "trunk": seed_mod(s,"trunk","l2_diff_norm"),
        "early_head": seed_mod(s,"early_head","l2_diff_norm"),
        "late_head": seed_mod(s,"late_head","l2_diff_norm"),
        "value_task_head": seed_mod(s,"value_task_head","l2_diff_norm"),
        "value_cost_head": seed_mod(s,"value_cost_head","l2_diff_norm"),
        "frozen_total": pd["seeds"][str(s)]["frozen_trunk_plus_early"]["l2_diff_norm"],
        "trainable_total": pd["seeds"][str(s)]["trainable_late_plus_critics"]["l2_diff_norm"]
      } for s in (11,22,33)
    },
    "artifact": "raw/runtime/ROUND66_REVIEW_PARAM_DIFF.json"
  },
  "synthetic_updates_audit": {
    "cap": 32,
    "cap_kind": "optimizer.step-class cap (synthetic_optimizer_update_cap)",
    "synthetic_ppo_update_calls": 1,
    "bare_synthetic_optimizer_steps": 1,
    "optimizer_steps_inside_the_one_synthetic_ppo_update": ">=4 (EPOCHS_PER_BATCH=4 x ceil(samples/256)); exact value NOT recorded",
    "test_self_recorded_counter": 2,
    "test_self_recorded_sources": ["_accept_stdout.txt", "_accept_stdout_verify.txt"],
    "ledger_recorded_value": 4,
    "is_4_ppo_calls": False,
    "is_4_optimizer_steps_single_run": False,
    "only_consistent_reading": "aggregate across the two wire-up runs (original 2 + independent re-run 2 = 4 synthetic-update events); inference, not an explicitly recorded derivation",
    "missing_evidence_not_filled_by_rerun": ["exact optimizer.step inside the one synthetic PPO update", "explicit derivation of the ledger '4'"],
    "cap_breach": False,
    "cap_safety_note": "under every preserved interpretation (2 events/run; 1 PPO/run; 5-18 optimizer.step across up to two runs; or 4 aggregate events) consumption <= 32"
  },
  "numeric_projection": {
    "artifact": "raw/runtime/ROUND66_REVIEW_NUMERIC_PROJECTION.json",
    "source_sha256": proj["source_files"],
    "scene_count": len(proj["scene_order"]),
    "model_arms": 18,
    "rule_arms": 3,
    "bootstrap": proj["bootstrap"],
    "closure": proj["closure"]
  },
  "accounting_reaffirmed": {
    "valid_tasks": 1110,
    "breakdown": "48 block-A references + 384 training episodes + 678 eval cells",
    "consumed_slots": 1111,
    "d1_unregistered_extra": 1,
    "actual_environment_constructions": 1112,
    "hard_cap": 1150,
    "headroom": 38,
    "three_seed_eligibility": "VERIFIED (unchanged); param diff additionally confirms frozen early path bit-identical to R60 ep0 and trainable modules moved"
  },
  "protected_files_unchanged": {
    "ROUND46_CURRENT_TASK.json": "cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b",
    "ROUND46_R2_V6_STOP_STATE.json": "93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18"
  },
  "originals_preserved": "ROUND66_FINAL_REPORT.md, ROUND66_MACHINE_VERDICT.json, ROUND66_LEDGER_FINAL.json, ROUND66_FREEZE*.json, ROUND66_DEVIATION_REGISTER.json and all rows/claims/checkpoints/logs are byte-preserved; this review only appends new records."
}
out = BASE + r"\raw\runtime\ROUND66_REVIEW_MACHINE_VERDICT.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(mv, f, indent=1)
print("WROTE", out)
print(json.dumps(mv, indent=1)[:1500])
