# ROUND71 Final Report - Critic-Initialisation Calibration (Approved Fixed Trial)

Task: `ROUND71-APPROVED-CALIBRATION-FIXED-TRIAL`
Branch: `exp/round71-critic-initialization-calibration`
Run dir: `runs/comm_bc_initialized_constrained_ppo_20261007_r2/round71_critic_initialization_calibration/`
Python: `D:\project_env\python.exe`

`ROUND71_MACHINE_VERDICT.json` records `delivered_head` = `af92f9787971fe7e2b8d07719112f1c6bc69d5d9`, the
HEAD that carries all ROUND71 code, gates and trial wiring. The delivery commit made afterwards adds this
report, the verdict, the consolidated resource ledger, the checkpoint index and the git-external manifest,
so HEAD advances again; the post-delivery HEAD is the one named in the delivery message.

## 1. What this session executed

ROUND71 is the only live task. The ROUND60/ROUND61 text visible in the handoff prompt is historical and was
not acted on. This session was a continuation of an approved fixed trial, not a new experiment.

Training and evaluation were already complete when this session started. Work completed here, in order:

1. Verified HEAD (`af92f97`), branch, and a clean worktree (only the untracked r59 analysis log).
2. Confirmed the offline-fit stage: 3 calibrated INIT `.pt` written, 70,693 of 72,000 offline rows used.
3. Confirmed the training stage: all 3 seeds at `ep128`, 96 PPO updates, 696 real optimizer steps, all
   three first-batch gates `passed`.
4. Confirmed the evaluation stage: 198 cells (192 model + 6 focal), `rc=0`, eval-order check `passed`.
5. Ran the zero-consumption `stats` phase, producing `raw/ROUND71_STATISTICS.json`.
6. Re-verified the whole chain read-only: calibration identity, training chain linkage, model binding of
   all 198 eval rows, ledger recomputation, gate and test receipts.
7. Recomputed the on-disk main ledger (it still held the pre-eval value of 385 attempts) and wrote the
   consolidated closing ledger, checkpoint index, git-external manifest, machine verdict and this report.

## 2. The single scientific intervention

Each seed's two critic initial affine heads - task `inner.value_head` and cost `cost_head` - were replaced
with a per-seed dual-output ridge calibration fitted on that seed's own R68 128-episode teacher data.
Everything else is inherited unchanged from the frozen R58/R60/R66/R68 living configuration.

What is verified to be unchanged, per seed, from `raw/checkpoints/r71_init_identity_s*.json`:

- `install.changed_modules` = `["value_cost_head", "value_task_head"]` and nothing else.
- `install.digest_before` equals `install.digest_after` for `trunk`, `early_head` and `late_head`.
- `rng_unchanged_by_calibration` = true, `optimizer_fresh` = true, `lambda_init` = 1.0.
- 12 parameter objects / 5,830 scalars: 6 frozen objects / 5,570 scalars (`trunk`, `early_head`) and
  6 trainable objects / 260 scalars (`late_head`, `value_task_head`, `value_cost_head`).
- Source is the registered R60 `ep0` model, `pre_scale_equals_r60_ep0` = true, with the R68 late-head
  post-scale (`late_init_alpha` = 0.10) applied before calibration.

Fit contract: ridge 1e-3, centring/scaling over all training teacher rows of that seed, constant-column
scale 1, intercept not penalised, no cross-validation, no seed selection. One fit per seed.

Pre-launch gates G1-G8 all `passed`. All three first-batch gates `passed`, with
`first_batch_behaviour_matches_r68` = true and an empty mismatch list.

## 3. Training chain verification

Read-only re-verification of the chain, per seed:

- `v2_train_s*_init.pt` and `v2_train_s*_before_b0.pt` are tensor-identical (12/12 parameter objects), so
  training started from the calibrated init.
- `v2_train_s*_ep128.pt` and `v2_train_s*_after_b31.pt` are tensor-identical (12/12), so the evaluated
  model is the committed end of batch 31.
- `trunk` (4 tensors) and `early_head` (2 tensors) are unchanged from init to `ep128`; only `late_head`,
  `value_task_head` and `value_cost_head` moved.
- `episodes_done` = 128 and `updates` = 32 in all three `ep128` checkpoints; `lambda` at `ep128` is
  1.625 (s11), 1.5 (s22), 1.625 (s33).

Real training side-effects: 96 `ppo_update_calls`, 696 `optimizer_step_calls`. Synthetic optimizer steps
4 of the 16 allowed, gate only.

## 4. Evaluation and model binding

198 cells: 192 model cells (3 seeds x 2 modes x 32 scenes in block 1800000..1800031) plus 6 focal cells on
scene 1700027. Every row has `steps` = 1000 and `optimizer_steps` = 0, so the evaluation performed no
training. The eval-order check reports `passed` over all 198 cells.

Read-only loading of the six on-disk checkpoints through the repository's own controller and
`parameter_summary()` reproduces exactly one `parameter_sha256` per seed per arm, and all 198 eval rows
carry a `controller_parameters_after.parameter_sha256` that matches one of those six values with zero
mismatches. The R68 control rows are read-only reuse of the same 32 scenes.

## 5. Results

Primary reading is greedy; `sampled` is reported separately and never pooled or cherry-picked.

### 5.1 Primary: NEW_EP128 minus R68_TRAINED_ALPHA, greedy

| seed | on-time (NEW) | on-time (R68) | task delta | CI | B/scene (NEW) | B/scene (R68) | cost delta | CI |
|---|---|---|---|---|---|---|---|---|
| 11 | 183 | 183 | 0 | [0, 0] | 4066.5 | 4066.5 | 0.0 | [0, 0] |
| 22 | 183 | 183 | 0 | [0, 0] | 4066.5 | 4066.5 | 0.0 | [0, 0] |
| 33 | 183 | 183 | 0 | [0, 0] | 4066.5 | 4066.5 | 0.0 | [0, 0] |

Zero targets lost, zero gained, zero scenes cheaper, zero scenes dearer, and zero changed first-report
events in all three seeds.

### 5.2 Secondary: NEW_EP128 minus R68_TRAINED_ALPHA, sampled

| seed | on-time (NEW) | on-time (R68) | task delta | cost delta (B/scene) | cost CI |
|---|---|---|---|---|---|
| 11 | 180 | 180 | 0 | -155.81 | [-389.63, 0.00] |
| 22 | 183 | 183 | 0 | +331.13 | [78.38, 662.63] |
| 33 | 180 | 181 | -1 | -642.19 | [-1133.81, -239.63] |

The cross-seed cost direction is not consistent: seed 22 is significantly more expensive (CI excludes 0),
and seed 33 loses one on-time target (scene 1800019 / `piloc-e1-target-0`).

### 5.3 Two-column context against the frozen rules (greedy, identical in all three seeds)

| rule | task (NEW) | task (rule) | B/scene (NEW) | B/scene (rule) | cost delta |
|---|---|---|---|---|---|
| IMMEDIATE | 183 | 183 | 4066.5 | 14356.5 | -10290.00 |
| LATE_DISCOVERY_GATE | 183 | 183 | 4066.5 | 3897.94 | +168.56 |

In `sampled`, NEW loses 3 on-time targets versus IMMEDIATE in seeds 11 and 33 and 0 in seed 22; the same
pattern holds versus LATE_DISCOVERY_GATE.

### 5.4 Focal scene 1700027 / `piloc-e1-target-5`

All three seeds, under both `NEW_INIT` and `NEW_EP128`: 6/6 on-time, 7500 B spent, target-5 first reported
at step 134 with action SEND in the early region, and the `task03-progress` event at relay tick 156 is
accepted, on-time and credited.

| comparison | task delta | cost delta | target ID sets |
|---|---|---|---|
| NEW_INIT minus R68_INIT_ALPHA | 0 | 0 | identical |
| NEW_EP128 minus R68_TRAINED_ALPHA | 0 | 0 | identical |
| NEW_EP128 minus NEW_INIT | 0 | 0 | identical |

### 5.5 First-report split (order: sends_early, skips_early, sends_late, skips_late)

| cell | NEW | R68 |
|---|---|---|
| 11 greedy | 2, 132, 49, 0 | 2, 132, 49, 0 |
| 22 greedy | 2, 132, 49, 0 | 2, 132, 49, 0 |
| 33 greedy | 2, 132, 49, 0 | 2, 132, 49, 0 |
| 11 sampled | 1, 133, 38, 11 | 1, 133, 40, 9 |
| 22 sampled | 2, 132, 44, 5 | 2, 132, 40, 9 |
| 33 sampled | 2, 132, 34, 15 | 2, 132, 42, 7 |

The early columns are identical in every cell. This is structural, not incidental: `trunk` and
`early_head` are frozen (`requires_grad=False`) by the inherited R66/R68 intervention, so the early
first-report path cannot move at all. Every observed first-report difference is confined to the late
region.

## 6. Consumption against the caps

| item | used | cap | headroom |
|---|---|---|---|
| normal planned launches | 582 | 582 | 0 |
| technical-failure reserve | 1 | 18 | 17 |
| booked attempts / environment constructions | 583 | 600 hard | 17 |
| offline real trunk-input rows | 70,693 | 72,000 | 1,307 |
| synthetic optimizer steps (gate only) | 4 | 16 | 12 |
| real optimizer steps | 696 | n/a | n/a |

Attempt breakdown from the 583 claim directories: 385 training attempts (384 complete plus 1 aborted with
zero steps), 192 model eval cells, 6 focal eval cells. 582 completed 1000-step episodes = 582,000 steps.
`hard_cap_exceeded_by` = 0. `historical_non_plan_probe` = 0: ROUND71 used a fresh, independent quota and
did not borrow from R60/R66/R68.

The single reserve use is seed 11 episode 0 attempt 1 (`r71_new_training_s11_e0`), status
`claimed_before_environment`, `environment_constructor_calls` = 1, `simulation_started` = False,
`completed_steps` = 0. It is a non-refundable permit/quota consumption with zero steps; actual builder
invocation is not asserted from the permit alone. Attempt 2 completed the episode.

Two registered bookkeeping gaps, both also in the machine verdict:

1. The seed-11 offline-fit receipt was reconstructed read-only after an ndarray serialization defect. Its
   durable ledger snapshot records 31,778 cumulative rows at the end of the seed-11 phase, while its
   reconstruction note records 20,013 rows for the successful seed-11 fit. The 11,765-row difference is
   not evidenced by any durable per-call record, because `ForwardLedger` persists aggregates only. The
   durable total of 70,693 rows is the quantity compared against the 72,000-row cap, so the cap check
   itself is unaffected.
2. The main `ROUND71_LEDGER.json` on disk still held the pre-eval value of 385 attempts. `phase_ledger()`
   recomputes it from the claim files and now writes 582 + 1 = 583. The pre-eval value is superseded, not
   deleted.

Six bookkeeping corrections are logged in `raw/runtime/ROUND71_CORRECTIONS.jsonl`; all were private wiring
or bookkeeping fixes, none changed the scientific configuration, and each records its own quota impact.

## 7. Tests

84 tests pass, 0 fail (`r71_test_summary.json`): 12 trial, 50 readiness, 22 runtime-binding.

## 8. What can and cannot be concluded

Retained:

- On the primary greedy reading, the calibrated critic init preserved the R68 task outcome exactly in all
  three seeds and produced an exactly zero cost change, with zero changed first-report events.
- The early key target scene 1700027 / `piloc-e1-target-5` is preserved in all three seeds under both the
  calibrated init and the trained model, and the calibrated init reproduces the R68 initial policy
  bit-for-bit on that scene.
- Every observed first-report difference is confined to the late region, because the early path is frozen
  by construction.

Deferred:

- There is no cost reduction. The sampled cost direction is -155.8 / +331.1 / -642.2 B per scene across
  the three seeds, so it is not consistent, and seed 22 sampled is significantly more expensive.
- The greedy bit-identity is partly structural - a frozen early path plus a two-way action head - and is
  not evidence that critic calibration is behaviourally neutral in general.
- ROUND71 added offline teacher exposure that R68 did not have: 70,693 real trunk-input rows over 78
  calls. This is not a same-training-budget race against R68 and must not be read as one.
- Block 1800000..1800031 is an already-used development block, not an unseen confirmation set. n = 32
  scenes per cell, one random realisation per cell, three seeds not pooled, and a CI containing 0 is not
  equivalence.
- These results support only the whole critic-initialisation-calibration structure on this development
  block. They do not prove the uncalibrated critic init was the only root cause, and they do not support
  generalisation, formal non-inferiority or deployment.

## 9. Protection and delivery

Both ROUND46 dirty files are unmodified and unstaged, with hashes matching the required values exactly:
`ROUND46_CURRENT_TASK.json` = `cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b` and
`ROUND46_R2_V6_STOP_STATE.json` = `93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18`.
All historical failures, models, logs and ledgers from earlier rounds and other sessions are untouched.

Committed in this delivery (no weights, no raw logs): this report, `ROUND71_MACHINE_VERDICT.json`,
`ROUND71_RESOURCE_LEDGER.json`, `ROUND71_CHECKPOINT_INDEX.json` and `ROUND71_GIT_EXTERNAL_MANIFEST.json`
(240 hashed artifacts that stay outside git by design). No push, no merge, no deployment.

Training side-effects are zero for the whole evaluation and for every read-only check in this session.
