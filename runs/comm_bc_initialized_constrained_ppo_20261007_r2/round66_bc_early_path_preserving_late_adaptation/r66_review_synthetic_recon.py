import json
BASE = r"D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2\round66_bc_early_path_preserving_late_adaptation"
OUT = BASE + r"\raw\runtime\ROUND66_REVIEW_SYNTHETIC_RECON.json"

rec = {
  "kind": "round66-review-synthetic-reconstruction/1",
  "supersedes": "the synthetic_updates_audit block of raw/runtime/ROUND66_REVIEW_MACHINE_VERDICT.json (committed 858c489), which is preserved byte-identical; this record corrects it",
  "status": "RECONSTRUCTED (read-only); cumulative guarantee WITHDRAWN",
  "frozen_constants": {"EPOCHS_PER_BATCH": 4, "MINIBATCH_STEPS": 256, "DATA_ORDER_SEED": 20260927},
  "early_break_rule": {
    "frozen_early_ppo_update": "no epoch/minibatch break; one optimizer.step per minibatch; optimizer_steps = minibatches = EPOCHS_PER_BATCH x ceil(N/MINIBATCH_STEPS)",
    "v2_trainer.ppo_update": "identical; no target_kl/KL early-stop",
    "the_only_continue": "r66_frozen.py line 747 skips gate-prior work for non-free targets only; it never skips a minibatch or an optimizer.step"
  },
  "batch_samples_N": {
    "rule": "one sample per tape row (decision step); N = total non-empty tape rows across the 4 reloaded R60 episodes",
    "episodes": "R60 seed 11 ep0..ep3 (the exact set test_r66_frozen_acceptance.reload_episode reloads)",
    "per_episode_rows": {"e0_s1300027": 66, "e1_s1300014": 71, "e2_s1300028": 52, "e3_s1300044": 76},
    "N": 265,
    "corroboration": "raw/checkpoints/r40_batch_s11_b0_attempt01.json records samples=265, minibatches=8, epochs=4 for the same seed-11 first batch"
  },
  "per_run": {
    "synthetic_update_events": 2,
    "event_breakdown": "1 bare opt.step() (test section 6) + 1 frozen_early_ppo_update() call (test section 10)",
    "raw_optimizer_step": 9,
    "raw_optimizer_step_breakdown": "1 bare + 8 internal (minibatches = 4 x ceil(265/256) = 8)",
    "ppo_update_calls": 1,
    "marked": "RECONSTRUCTED from frozen constants + tape row counts; the exact internal count was not itself persisted (throwaway report)"
  },
  "evidenced_runs": {
    "count": 2,
    "evidence": "_accept_stdout.txt (written 14:12:03) and _accept_stdout_verify.txt (written 14:49:05) are byte-identical (sha256 7A084853...) but 37 minutes apart -> two distinct deterministic executions; neither is git-tracked",
    "both_record": "synthetic_optimizer_updates = 2, synthetic_update_cap = 32"
  },
  "reconciliation_of_the_ledger_4": {
    "value": 4,
    "reading": "2 evidenced runs x 2 synthetic-update EVENTS = 4; the test counter counts events (bare opt.step + ppo_update call), NOT raw optimizer.step and NOT ppo calls alone",
    "is_4_optimizer_step": False,
    "is_4_ppo_calls": False,
    "presumes": "exactly 2 runs; this is the number EVIDENCED, not a proven total"
  },
  "corrections_to_prior_erratum": [
    "prior claim 'only 1 synthetic PPO update ran' is WRONG: there were 2 (one per evidenced run)",
    "prior '5-18 optimizer.step across up to two runs, <=32, no breach' is WITHDRAWN: per-run is 9 (reconstructed) and the number of runs is unbounded, so no <=32 guarantee holds"
  ],
  "cumulative": {
    "events_lower_bound": 4,
    "raw_optimizer_step_lower_bound": 18,
    "total": "UNKNOWN",
    "reason": "the number of historical test executions is not recorded (only 2 stdout snapshots survive; no test-run ledger); synthetic updates consume no resource slot, so nothing else bounds them",
    "no_breach_guarantee": "WITHDRAWN - an unbounded run count cannot establish a total <= 32"
  },
  "zero_side_effects": "read-only only: JSONL line counts + source reads; no PPO rerun, no optimizer call, no environment, no checkpoint/RNG write"
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1)
print("WROTE", OUT)
