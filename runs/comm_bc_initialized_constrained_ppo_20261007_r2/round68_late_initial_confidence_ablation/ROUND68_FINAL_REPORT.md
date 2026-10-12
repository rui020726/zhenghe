# ROUND68 Final Report — Late-Initial-Confidence Ablation

Task: `ROUND68-LATE-INITIAL-CONFIDENCE-ABLATION`
Branch: `exp/round68-late-initial-confidence-ablation`
Run dir: `runs/comm_bc_initialized_constrained_ppo_20261007_r2/round68_late_initial_confidence_ablation/`
Python: `D:\project_env\python.exe`
`ROUND68_MACHINE_VERDICT.json` records `delivered_head` = the repository HEAD at verdict-write time
(`48cfabc00e212fbd1f7367e734646a9b42e78e27`, the commit carrying the attempt-2 booking fix and this
session's zero-environment acceptance work). The delivery commit made afterwards adds the report, the
verdict and the three read-only analysis scripts, so HEAD advances again; the post-delivery HEAD is the
one named in the delivery message, not the one inside the verdict.

## 1. What this session executed

ROUND68 is the only live task; the ROUND60/ROUND61 text in the handoff prompt is historical and was
not acted on. This session was a continuation, not a new experiment.

Work completed in this session, in order:

1. Verified HEAD, worktree, and the live cursor before touching anything.
2. Observed seed 22 Phase B already running on the inherited session and **did not restart it**. It
   ran continuously from batch 22 to batch 32 and finished on its own. No duplicate update was
   issued, so no committed update was repeated.
3. Ran seed 33 Phase A (`--first-gate-only`), verified the real first-batch gate, then Phase B to
   episode 128.
4. Ran the 390-cell evaluation (`plan` -> `eval` -> `stats`).
5. Built the read-only rule-context post-processor and the independent projection.
6. Built the read-only checkpoint index, and corrected its cost-critic module name after the
   controller's real registration was confirmed (see section 4).
7. Exported a slim numeric copy of the projection for independent arithmetic audit.
8. Wrote this report and the machine verdict, both generated from the artefacts on disk.

Environment constructions performed by this session: seed 33 Phase A (4) + seed 33 Phase B (124)
+ evaluation (390) = 518, all inside the already-authorised fixed plan. No extra speed test, smoke
run, or unbooked probe was launched.

## 2. Budget accounting (measured in actual environment constructions)

| Item | Count |
| --- | --- |
| Planned normal (384 train + 384 model-arm eval + 6 focal) | 774 |
| Booked normal | 774 |
| Booked reserve (the seed-11 e1 late-registered correction, counted exactly once) | 1 |
| **Actual environment constructions** | **775** |
| Hard cap | 800 |
| **Hard headroom** | **25** |
| Hard cap exceeded by | 0 |
| Historical non-plan probe | 0 |
| Unregistered extra environment constructions | 0 |

Source: `raw/runtime/ROUND68_EVAL_QUOTA_GUARD.json`, whose live bindings were re-read at the end of
the eval phase (`r40_NORMAL_CAP=774`, `r40_HARD_CAP=800`, `common_RESERVE_LAUNCHES=26`).

**Disclosure — stale ledger file.** `raw/runtime/ROUND68_LEDGER.json` reads
`actual_environment_constructions: 385`. That is a **phase-boundary snapshot written by the training
executor after seed 22**; the eval phase does not rewrite it. The atomic truth is the slot files:
`raw/launch_slots/normal/` holds 774 files and `raw/launch_slots/reserve/` holds 1, giving 775. The
quota guard above agrees. The 385 figure must not be quoted as the final consumption.

**Claim-admission cross-check.** There are exactly 774 logical-job claim directories, every one of
them with exactly one `attempt_0001.json`, all `status: complete`, all `normal` category. There are
**zero** multi-attempt claim directories anywhere in the round, i.e. no retry ever occurred.

Acceptance/synthetic-test computation caps (separate from the 800 env budget, disclosed and not
refunded):

| Guard | Cap | Used | Status |
| --- | --- | --- | --- |
| Model forward rows | 8192 | 9319 | **EXCEEDED by 1127** (historical, preserved) |
| Synthetic optimizer steps | 32 | 18 | within cap (16 make-up sub-limit exhausted) |
| Synthetic PPO calls | 32 | 5 | within cap, recorded as 5 including one duplicate charge |

The synthetic-step figure above is read from the live guard `raw/runtime/r68_guard_synthetic_steps.json`
(18/32). `ROUND68_GUARD_RECONCILIATION.json` carries 2, but that is a point-in-time snapshot written
before the later make-up steps; the live guard is authoritative.

Training journals, final: **96 PPO update calls / 676 optimizer steps** = 3 seeds x 32 batches, and
236 + 208 + 232 = 676 optimizer steps across seeds 11/22/33. Internally consistent.

## 3. Gate results

All three seeds ran their real first-batch gate after the first update and passed before continuing:

| Seed | Gate | Bool checks | early_free | late_free | lambda after | alpha |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | passed | 21/21 | — | — | 1.0 | 0.10 |
| 22 | passed | 21/21 | 16 | 7 | 1.0 | 0.10 |
| 33 | passed | 21/21 | 16 | 6 | 1.0 | 0.10 |

Seed 22 `parameters_after_sha=5879c7a7…`, seed 33 `parameters_after_sha=e4a8eccd…`. Seed 11's gate
was written before the mid-session code patch and therefore carries no attempt-resolution fields;
see section 6 for why that is immaterial.

`raw/runtime/ROUND68_EVAL_ORDER_CHECK.json` reports `passed`, 390 cells, the executor having visited
every planned cell in exactly the planned order.

## 4. Checkpoint index (read-only)

`raw/ROUND68_CHECKPOINT_INDEX.json`. Each seed's fixed ep128, loaded read-only (no optimizer, RNG or
cursor restore):

| Seed | parameter_sha256 | file_sha256 | opt params | frozen objs | trainable objs | lambda final | episodes | updates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | `3aa8bd177846…` | `057ec6c83ec1…` | 6 | 6 | 6 | 1.625 | 128 | 32 |
| 22 | `f468ba1104d6…` | `c8545552cd7a…` | 6 | 6 | 6 | 1.375 | 128 | 32 |
| 33 | `b4f635c08dbf…` | `e90caf2c0e76…` | 6 | 6 | 6 | 1.500 | 128 | 32 |

Frozen mask on all three: `frozen_modules=[trunk, early_head]`,
`trainable_modules=[late_head, value_task_head, value_cost_head]`, `frozen_scalar_count=5570`,
`late_init_alpha=0.1`. Lambda history has 32 entries, monotone non-decreasing, on every seed.

**Module naming correction (recorded, not hidden).** An earlier draft of the checkpoint index
addressed the cost critic as `value_cost_head`, which is its *state_dict* key, and recorded it as
`unavailable`. The controller registers that object as `cost_head` at `controller.cost_head`, while
the task value head lives at `controller.inner.value_head` and is addressed as `value_task_head` by
`r68_frozen._module_sha`. Confirmed against the controller's real module objects and the payload, the
per-module parameter hashes are:

| Seed | trunk | early_head | late_head | value_task_head | cost_head |
| --- | --- | --- | --- | --- | --- |
| 11 | `49e41b7191cf…` | `aad35fc6ba3f…` | `93482c2c1f7a…` | `6dca9e009b63…` | `8aa5e189d252…` |
| 22 | `f71c6cfd30bd…` | `0d0d3581d130…` | `d25647fa0aab…` | `5ba84ba1f915…` | `37dd93535f79…` |
| 33 | `695b633569ad…` | `5fff65a080c1…` | `a5412a752ad7…` | `4e8363e98439…` | `fc21e4506ad3…` |

All five modules resolve on all three seeds. Nothing was filled with 0 and no gate was passed on a
string match. The naming difference is recorded in `module_naming_note` inside
`raw/ROUND68_CHECKPOINT_INDEX.json`.

## 5. Scientific results

Evaluation block **1800000..1800031** — an already-used development block, **not** a confirmation
set. Scenario is the paired unit; the frozen bootstrap index is
`PCG64(20261012)`, 10000 x 32, `index_sha256=fefcfebb1c1c…`, reused verbatim. Three seeds are not
pooled. greedy is the primary reading; sampled is reported separately.

### 5.1 PRIMARY: TRAINED_ALPHA(ep128) minus INIT_ALPHA(fresh-scaled-ep0)

**greedy (primary reading) — no change on any seed:**

| Seed | on-time T / I | B/scene T / I | task delta (CI95) | cost delta B/scene (CI95) | lost / gained |
| --- | --- | --- | --- | --- | --- |
| 11 | 183 / 183 | 4066.5 / 4066.5 | 0.000 [0.000, 0.000] | 0.0 [0.0, 0.0] | 0 / 0 |
| 22 | 183 / 183 | 4066.5 / 4066.5 | 0.000 [0.000, 0.000] | 0.0 [0.0, 0.0] | 0 / 0 |
| 33 | 183 / 183 | 4066.5 / 4066.5 | 0.000 [0.000, 0.000] | 0.0 [0.0, 0.0] | 0 / 0 |

**sampled (secondary reading, reported separately, not selected):**

| Seed | on-time T / I | B/scene T / I | task delta (CI95) | cost delta B/scene (CI95) | lost / gained |
| --- | --- | --- | --- | --- | --- |
| 11 | 180 / 176 | 3308.4 / 2756.6 | +0.125 [+0.031, +0.250] | +551.8 [+161.8, +1017.4] | 0 / 4 |
| 22 | 183 / 183 | 3338.2 / 3338.2 | 0.000 [0.000, 0.000] | 0.0 [0.0, 0.0] | 0 / 0 |
| 33 | 181 / 181 | 3491.1 / 3406.9 | 0.000 [0.000, 0.000] | +84.2 [0.0, +252.6] | 0 / 0 |

Seed 11 sampled gained exactly four targets, at scenes 1800014/`target-4`, 1800020/`target-4`,
1800021/`target-3`, 1800025/`target-5`, and paid for them. No target was lost on any seed in either
mode.

### 5.2 Focal mechanism diagnostic (scene 1700027, greedy, both arms x 3 seeds)

All six cells are identical: on-time 6, spent 7500 B, target mask `111111`, i.e. the full
`piloc-e1-target-0..5` set. On the focal scene the lowered late-head initial confidence produced no
difference at all, before or after late-only training.

### 5.3 Rule context (absolute, same 32 scenes)

Read-only from ROUND66's own qualified rule rows; no new rule simulation was run.

| Rule | on-time total | B/scene | union target IDs |
| --- | --- | --- | --- |
| IMMEDIATE | 183 | 14356.5 | all 6 |
| LATE_DISCOVERY_GATE | 183 | 3897.9 | all 6 |
| ALWAYS_HOLD | 170 | 0.0 | all 6 |

Both ROUND68 arms reach 183 on-time tasks, matching IMMEDIATE and LATE_DISCOVERY_GATE on task, at
4066.5 B/scene: **10290 B/scene cheaper than IMMEDIATE** and **168.6 B/scene more expensive than
LATE_DISCOVERY_GATE** (CI95 [+0.0, +505.7]). Against ALWAYS_HOLD the arms gain 13 targets
(+0.406/scene, CI95 [+0.219, +0.625]) and pay 4066.5 B/scene.

In sampled mode the arms sit slightly below both sending rules on task while staying far cheaper
than IMMEDIATE (seed 11 INIT −0.219/scene vs IMMEDIATE, losing 7 targets; TRAINED −0.094/scene,
losing 3).

## 6. Code divergence from the freeze, and why it is immaterial

`ROUND68_FREEZE.json` records `code_sha256` for eight files. Seven still match byte-for-byte.
**`r68_execute.py` and `r68_eval.py` diverge.** The freeze was written pre-launch (2026-10-12
23:46); the two files were patched at 03:04 and 03:29 and committed at 03:34 as the attempt-2
artefact-seam fix.

The patch is booking-only: it threads the live attempt number through `_run_train_episode`, the
`train_seed` reload, and `_gate_from_committed`, and routes evaluation through the shared
`base._r68_atomic_run`. It changes no science: loss, routed forward, mask, joint log-prob, combined
advantage, GAE, ALL_ROWS, clipping, the beta=1.00 prior, the dual update, the 0.10 init-only scale
and the frozen module set are untouched.

Why this does not disturb the delivered artefacts:

- Every one of the 774 logical jobs completed on **attempt 1**, with exactly one `attempt_0001.json`
  each and zero multi-attempt directories anywhere in the round.
- The pre-fix code hard-coded attempt 1 at four training sites; the post-fix code resolves attempt 1
  from the live/qualified claim. For attempt 1 the resulting artefact name is byte-identical
  (`<tag>_a1`, `r40_batch_s{seed}_b{batch}_attempt01.json`), which is exactly what the counter-example
  at `raw/runtime/ROUND68_ATTEMPT2_COUNTEREXAMPLE.json` was preserved to document.
- Seed 11's gate predates the patch and carries no attempt fields; since seed 11 has 32 single-attempt
  batch records covering episodes 0..127 exactly once, and 128 rows, nothing moves.
- The `test_r68_attempt2_paths.py` fixture proves the post-fix attempt-2 path with 95 checks and 0
  environment constructions.

Consequently all three seeds' statistics come from one consistent, single-attempt execution record.
The divergence is disclosed rather than reconciled by editing the freeze.

## 7. Eligibility of the three ep128 models

`raw/runtime/R68_SEED11_ELIGIBILITY_RECEIPT.json` — status `verified`, 125 checks, 0 failed. It
proves, for seed 11: the fresh controller is built model-only and bit-exact from the compliant
ROUND60 ep0; `pre_scale_equals_r60_ep0`; the late head is scaled exactly once; `before_b0` Adam
state is empty (no inherited momentum) with `optimizer_steps=0`, `updates=0`, `episodes_done=0`,
`lambda=1.0`, `data_order_seed=20260927`, and trunk/early_head/critics equal to ROUND60 ep0; the
batch chain covers episodes 0..127 exactly once with 4 episodes per batch and 32 single-attempt
records; ep128 has 128 episodes, 32 updates, Adam 6 x 236 steps, lambda 1.625 monotone; trunk and
early_head still equal ROUND60 ep0 at ep128 while the late head and both critics moved; 128 claims
all complete, all attempt 0001, all normal, one constructor call and one run invocation each, zero
build failures; and the receipt itself performed zero environment constructions, zero model
forwards, zero optimizer steps and zero new checkpoint files.

The same structural facts were re-derived independently for all three seeds in this session by
`_r68_checkpoint_index.py`: 6 optimizer params, 6 frozen + 6 trainable parameter objects, 32 lambda
entries monotone, 128 episodes, 32 updates. `r68_init_identity_s{11,22,33}.json` exist for all three
seeds.

**Caveat carried forward.** The seed-11 eligibility receipt covers seed 11 specifically. Seeds 22 and
33 are covered by their own first-batch gates, their 32 single-attempt batch records, their 128 rows
and the checkpoint index above, but not by a per-seed 125-check receipt of the same depth. The
scientific conclusion in section 5 rests on all three seeds being structurally equivalent; the
deepest per-seed provenance proof exists only for seed 11.

## 8. Conclusion boundaries

What the results support:

- On this development block, for this frozen policy family, lowering the late head's initial
  confidence to 0.10 and then training only the late head plus both critics **reproduced the init
  behaviour exactly in greedy on all three seeds** — identical on-time task sets and identical
  metered cost, CI95 [0, 0] on both task and cost.
- The only behavioural movement anywhere is in sampled mode, and it is not uniform across seeds:
  seed 11 gains four targets at a cost, seed 22 is unchanged, seed 33 is unchanged on task and
  slightly more expensive. There is no cross-seed consistent direction.
- Both arms match IMMEDIATE and LATE_DISCOVERY_GATE on task (183) while costing far less than
  IMMEDIATE and marginally more than LATE_DISCOVERY_GATE.

What the results do **not** support:

- This is a negative/null primary reading, not a demonstration that the late head is saturated, and
  certainly not that saturation is the only root cause. A zero greedy delta is equally consistent
  with the late-only training having moved parameters that do not change any argmax on this block.
- No generalization, no formal non-inferiority, no deployment readiness.
- Block 1800000..1800031 is a used development block; these are not held-out confirmatory numbers.
- CI containing 0 is not equivalence, and the absence of a change is not proof that no change is
  possible.
- Worse performance would not have been a technical failure; nothing was excluded, re-run or tuned.
- The 0.10 scale is init-only. It preserves greedy argmax by construction and necessarily lowers
  sampling confidence on late rows; that is the intervention, and it is why the sampled arm is the
  only place any movement appears.
- The critic->trunk isolation is installed on the live path, but its trunk effect is masked because
  the trunk is `requires_grad=False` under this intervention. The absent trunk gradient is not
  evidence about isolation.
- ROUND60's historical 608 constructions (over its own 600 cap by 8) is a separate disclosed
  deviation and is not ROUND68 headroom. ROUND68 did not borrow it.

## 9. Git-external originals

`raw/` is gitignored; the following original artefacts live outside Git and are named here so they
are not lost: `raw/ROUND68_FIXED_PLAN.json`, `raw/ROUND68_EVAL_PLAN.json`,
`raw/ROUND68_STATISTICS.json`, `raw/ROUND68_RULE_CONTEXT.json`, `raw/ROUND68_PROJECTION.json`,
`raw/ROUND68_PROJECTION_SLIM.json`, `raw/ROUND68_CHECKPOINT_INDEX.json`,
`raw/ROUND68_MACHINE_VERDICT.json`, `raw/launch_slots/` (774 normal + 1 reserve),
`raw/launch_claims/` (774 logical jobs), `raw/rows/` (384 `v2_r68eval_model_*`, 6
`v2_r68eval_focal_*`, 384 training rows), `raw/checkpoints/` (the three ep128, all `before_b*` /
`after_b*`, the three `r68_init_identity_*`, 96 `r40_batch_*` records),
`raw/training_progress/`, `raw/logs/`, `raw/rejected/`, and `raw/runtime/` (the three first-batch
gates, the eval order check, the eval quota guard, the seed-11 eligibility receipt, the synthetic-PPO
disclosure, the attempt-2 counter-example, the guard reconciliation, the pure recheck, the zero-env
acceptance/test, the guard journals `checkpoint_writes.jsonl` / `launch_records.jsonl` /
`optimizer_step_calls.jsonl` / `ppo_update_calls.jsonl`, and `ROUND68_LEDGER.json`).

Protected files verified byte-unchanged and never staged:
`ROUND46_CURRENT_TASK.json` (`cf3ad198…`) and `ROUND46_R2_V6_STOP_STATE.json` (`93a2f69f…`).
The untracked `analysis/round59-communication-decision-density/r59_analysis_run.log` was left
untouched and unstaged.

## 10. Residual limitations

- The training ledger JSON is stale at 385 actual; the live figure is 775.
- The acceptance forward-row cap was exceeded during zero-env test development (9319 vs 8192). This
  is an acceptance/test computation overrun, disclosed and preserved, and it does not consume the
  800 environment budget.
- Only seed 11 has a deep per-seed eligibility receipt.
- The projection's on-time target-ID sets are rebuilt from each row's own `task_reward_events`;
  they agree with the frozen statistics on all six primary comparisons (means, CIs, lost and gained
  counts) and the regenerated bootstrap index hash matches the recorded one. Spot-checked by hand:
  seed 11 sampled sums to 176 -> 180 on-time and 2756.625 -> 3308.4375 B/scene, a delta of
  +551.8125, matching the projection exactly.
- The slim numeric copy is a pure extraction of the full projection; it carries no per-cell row
  paths or hashes, so provenance must be taken from `ROUND68_PROJECTION.json` itself.
- Read-only analysis scripts added this session (`r68_rule_context.py`, `_r68_projection.py`,
  `_r68_slim_export.py`, `_r68_checkpoint_index.py`, `_r68_verdict.py`) construct no environment,
  load no model for policy purposes beyond read-only checkpoint identity, and perform no PPO,
  optimizer or checkpoint-write operation.