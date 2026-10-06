# ROUND46-R2 V5 continuation entry

Task identity: `ROUND46-R2-V5-CONTINUE-20261007`.

This is the current continuation entry. Continue only on branch
`fix/comm-bc-initialization-readback` from the preserved ROUND46 R2 history. The
V4 stop and all earlier stop evidence remain immutable. The historical ROUND43
experiment is complete and is not an execution source for this task.

## Starting identity and preserved counts

- Starting HEAD: `3d6c086c7e2cc7960a40a4f9e5293ab2760b53dc`.
- Runner: `runs/comm_bc_initialized_constrained_ppo_20261007_r2/r46_execute_r2.py`.
- V5 text regression runner SHA256: `ea5597d0ccbe0562d0fc4b87a81708ea0f525747d31900ef9bf46b4a7d228b86`.
- Runtime: `D:\project_env\python.exe` (Python 3.10.19).
- V5 text identity regression: passed; six cases; zero runtime initialization, claims,
  environments, updates, and checkpoint writes.
- Current ROUND46 claims, environment starts, PPO updates, and optimizer steps: zero.
- Preserve the historical checkpoint-write/journal count of 18; never rewrite it as zero.
- Actual ledger at continuation: `normal_used=0`, `reserve_used=0`,
  `environment_constructor_calls=0`, `run_invocation_calls=0`,
  `completed_steps=0`, `ppo_update_calls=0`, `optimizer_step_calls=0`,
  `checkpoint_write_calls=18`, `checkpoint_write_journal_entries=18`.
- Current checkpoint journal SHA256:
  `8f68b0ea38e046073720b72cb9b009a55345b496769d6104b486d2c42b99c288`.

## Authorized sequence

1. Reuse unchanged V3 status-guard regression evidence only after exact source/receipt
   identity checks. The V5 text identity regression is separate evidence and does not
   stand in for a training gate.
2. Exercise the real R42 -> R45 -> R44 claim chain at the atomic claim boundary with
   zero actual claims, environments, runs, training updates, or checkpoint writes.
   Reproduce the stale binding rejection, prove the V5 verifier reaches the boundary
   once, and reject real source, frozen config, key-plan, and model-identity drift.
   Preserve all pre/post ledger and output snapshots.
3. Commit the V5 source and independent freeze/preflight evidence locally. Do not push.
4. If the actual prelaunch gate passes, execute only the fixed plan: 384 NEW training
   episodes (three seeds, 128 each) and 96 final evaluation cells. Keep 24 technical
   reserve and 504 total hard cap. Do not rerun completed claims or updates. Use one
   atomic claim per logical job and at most three training processes.
5. For each seed, run its first four episodes as part of the 128, and require all
   three real first-batch gates to pass before continuing the remaining episodes.
   Preserve exact recovery state and never repeat an already committed update.
6. Use the frozen BC four-module initialization, fresh empty Adam with ten unique
   parameters, lambda initial 1, and only the registered ROUND44 ep0 rollout RNG.
   Preserve the fixed one-shot SEND/SKIP, ALL_ROWS, actor advantage, B0, delta,
   dual step, gamma, GAE, PPO, world, search, allocation, reward timing, and evaluation
   settings in the frozen execution card.
7. Report each seed and mode separately. `development_go` requires all three greedy
   seeds to exactly match immediate's per-scene on-time task-ID sets and each seed's
   mean charged cost to be below 4838.25 B per episode. Do not pool seeds or treat
   gate/BC/hold behavior as success.
8. Stop on any validity or authorization error, preserve failure evidence, and do
   not retry by changing the plan. At completion, commit locally and stop; no push,
   merge, deployment, tuning, extra smoke, references, rules, streams, or scenarios.

## Scientific version

The training semantic version remains
`constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/dual-eta-lambda-0p5/1/rule-bc-late147-init/1/published-completion-event-at-relay-tick/2`.
This continuation changes only the versioned runner/execution identity. Preserve
ROUND45's failed gate and all V3/V4 stop materials unchanged.

## Exact initialized inputs and frozen experiment card

- BC checkpoint byte SHA256: seed 11
  `76ffc86c4aca26fb3b27380d5195a0cbd07822f7b85d1f1070a5550776dd458a`, seed 22
  `3cdb83785c51f9e40005d66a1d024d06a5ba6940616e22f3b01a9bf65b5e609d`, seed 33
  `9329ef048dfc5c1805af5756c7ea0962eedfc8d01aa26841e35af7001616803a`.
- Initializer: each seed's registered BC trunk, action head, task critic, and positive
  cost critic; new empty Adam with 10 unique parameters; lambda 1; cursor 0. Restore
  only the registered matching ROUND44 ep0 Python/NumPy/Torch rollout RNG after model
  initialization and before rollout.
- Data: ROUND44 registered 128-item order per seed; training scenes 1300000–1300047;
  development scenes 1301000–1301015. Reuse existing 64 immediate references,
  32 rule results, and 48 BC controls; add no streams, scenarios, or baseline runs.
- Frozen policy/training: 19-dimensional observation; 19→64→64 Tanh; one-shot
  SEND/SKIP; ALL_ROWS; sequential budget mask; budget 29300 B; 6 tasks; 1000 steps;
  ACK off; `policy_owned_timing`; CPU with one Torch and interop thread. Keep search,
  allocation, reward timing, and all remaining R44 world defaults unchanged.
- Frozen objective/optimizer: positive cost `C/B0`, `B0=10000`, `delta=0`,
  `A_actor=lambda_before*A_task-A_positive_cost`,
  `ghat=Y_immediate-Y_policy`, and
  `lambda_next=max(0, lambda_before + 0.5*ghat)`. Keep gamma 1, GAE lambda 1,
  clip 0.2, entropy 0.01, 4 epochs, learning rate 0.0003, grad clip 0.5,
  minibatch steps 256, value coefficient 0.5. Greedy ties select SEND.
- Each seed's real first-batch gate must verify BC/fresh-Adam initialization,
  original likelihood recomputation within `1e-6`, positive-cost/GAE/actor using
  pre-update lambda, event-time assignment, sequential mask, dual critics and clipped
  gradients, dual update direction, data IDs, execution counters, and full checkpoint,
  RNG, and schema identity. Static tests are not a substitute.
- `development_go` requires, separately for each seed, exact per-scene equality of
  the greedy on-time task-ID set with immediate and mean charged cost strictly below
  4838.25 B/episode. Use PCG64(20261007), 10000 shared 16-scene bootstrap indices,
  linear quantiles, and reuse each statistic consistently in report and verdict.
  Sampled is secondary. No pooling or exclusion of unfavorable scientific results.
