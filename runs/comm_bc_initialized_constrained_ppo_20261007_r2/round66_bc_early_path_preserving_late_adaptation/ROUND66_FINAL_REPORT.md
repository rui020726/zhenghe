# ROUND66 Final Report — BC Early-Path-Preserving Late Adaptation

**Task:** ROUND66-BC-EARLY-PATH-PRESERVING-LATE-ADAPTATION
**Branch:** `exp/round66-bc-early-path-preserving-late-adaptation`
**Audited code HEAD:** `39ac0a41b932ebccfb744cea365cdb9dcf0fce6a` (the code that produced every number below; the delivery commit HEAD is stated in the final handoff message)
**Status:** COMPLETE — training, evaluation, independent eligibility audit, statistics and local delivery all finished. No push, no merge, no deploy.

## 1. Research question and the single structural intervention

Freeze the ENTIRE early policy path (shared trunk + early_head, `requires_grad=False`) and let only
late_head + both critics learn, starting model-only from each seed's compliant ROUND60 ep0. Does the
restricted policy family preserve initial-value task performance while gaining cost reduction?

- **NEW** = R66 ep128 (frozen early path, read-only load).
- **OLD** = compliant R60 ep128 (structural contrast, runs exactly as R60 had it).
- **BC**  = compliant R60 ep0 initial value (the authorised added-learning reading).
- Routing (sampling, greedy, PPO likelihood recompute and evaluation all use the same):
  `ready_step = round(f[18]*1000) - round(f[0])`; `<=147` early head, `>147` late head.
- Unchanged vs R58/R60: beta=1.00, 19-dim obs, one-shot SEND/SKIP, sequential budget mask, dual
  critic with critic->trunk gradient isolation, `lambda*A_task - A_cost`, GAE(lambda=1.0), ALL_ROWS,
  global clip, dual update (eta=0.5), 6 targets, 29,300B budget, ACK off, greedy tie rule.
- **Initialisation:** both heads copied from each seed's R60 ep0 into two independent parameter
  objects; trunk + both critics carried over; no ep128 training init, no inherited Adam/lambda/RNG.
  First-batch gate s33 records `init_params_equal_r60_ep0 = true` and forward/loss equivalence at
  init. This is initial-policy equivalence; it does NOT claim parameter/gradient/Adam-trajectory
  equivalence after training.

## 2. Consumption and the 1110 / 1150 closure

| Item | Count |
|---|---|
| Valid logical tasks (normal) | 1110 = 48 block-A references + 384 training episodes + 678 eval cells |
| Normal slots consumed | 1110 (1109 complete + 1 failed_technical) |
| Reserve slots consumed | 1 (the retry of the one failed cell) |
| Consumed slots total | 1111 |
| D1 unregistered extra construction | 1 (step count UNKNOWN, see below) |
| **Actual environment constructions** | **1112 = 1111 consumed slots + 1 unregistered extra** |
| Hard cap | 1150 -> headroom 38, WITHIN cap |
| Real optimizer steps / PPO updates | 540 (180/seed) / 96 (32/seed) |
| Synthetic optimizer updates | 4 (pre-training acceptance only, counted separately) |

- **48 block-A references** were **newly run this round** as normal slots 1-48 (48 real environment
  constructions, logical `r66_ref_immediate_s13000xx`); they are NOT cache-reused and are separate
  from the 678 eval cells (slots 433-1110, scenes 1800000-1800031 + focal 1700027).
- **One failed_technical:** `r66eval2_rule_immediate_greedy_scene1800000`, normal slot 433. The rule
  arm attaches no collector, so `persist_result` raised "artefact tape is missing or empty". 1
  environment construction, 1000 steps. It was retried via reserve slot 1 (attempt a2) and
  completed; the original a1 row is preserved byte-identical as the single excluded orphan. No refund.
- **D1** (registered deviation, not hidden): the first detached launch was hard-killed inside the
  environment constructor for `r66_ref_immediate_s1300001` (normal slot 2); a second launch reused
  attempt-0001 and ran a complete 1000-step episode. The slot is counted once; the killed partial is
  the +1 unregistered construction. **Its step count is recorded UNKNOWN, not 0** — the claim's
  last-persisted counter reads `completed_steps=0 / simulation_started=false`, but positive
  termination by process handle is not established and the exact steps executed before the kill
  cannot be proven. No refund, no new slot, no claim rewritten.

## 3. Three-seed training eligibility: VERIFIED

- Each seed ran episodes 0..127, 32 PPO updates, 180 optimizer steps, lambda_final 1.0.
- First-batch gates **s11 / s22 / s33 all PASSED**, each confirming: dual-head routing active,
  init params equal R60 ep0, five-module structure, frozen trunk+early_head
  (`requires_grad=False`, gradient recorded None), trainable late_head + both critics
  (`requires_grad=True`, gradient recorded), optimizer owns the six trainable tensors, critic
  received gradient, late actor received gradient, likelihood alignment, finite value losses,
  lambda updated. Gate s33 early_free=64 / late_free=28.
- Three ep128 checkpoints (five-module NEW, 56,521 bytes each):
  - s11 `D5C003C6...A3D7`, s22 `77C2606D...56CF`, s33 `8C61EA7B...F06A` (full SHA in the checkpoint index).
- The compliant chain starts from R60 ep0 with both heads copied; it does not inherit any
  non-compliant weight, Adam or lambda.

## 4. Evaluation eligibility — independent auditor, 678/678

- Matrix: 96 rule_eval + 576 model_eval + 6 focal = 678 cells, on the development block
  (scenes 1800000-1800031 + focal 1700027). This block was already used for analysis; it is a
  development ablation, not an unseen confirmation set.
- **Independent auditor** (`r66_eval_eligibility_audit.py`, imports nothing from the adapter):
  **all 678 cells admissible**, 0 ambiguous, 0 excluded, 1 true orphan (the failed immediate a1),
  run-class breakdown model 582 / rule 96. Run-class contract: a MODEL row (collector attached) must
  carry a non-empty tape whose line count equals `collector_summary.decision_steps`, plus lifecycle
  and transcript; a RULE row (no collector) may have a structurally absent tape; a zero-byte file is
  reported distinctly from a missing one.
- **Cross-check:** the independent auditor and the adapter's own `_qualified_row_paths` admission
  derive the **same 678 logical jobs with 0 row-path mismatches** — two derivations, one answer.
- **Adapter audit-field defect (reported, not edited):** `_load_qualified_rows` builds its
  `orphan_row_files_excluded_from_statistics` via `resolved_path.startswith("r66eval2_<prefix>_")`,
  which is always False for absolute paths, so that field lists ALL on-disk rows (576 model / 97
  rule) instead of the true orphans. **No science impact:** every statistic comes from the
  sound-receipt qualified map, which the auditor re-derived and confirmed. The auditor is the
  authority on orphan accounting. The running adapter was not hot-edited.

## 5. Results — greedy primary, sampled separate, seeds not pooled

**Added learning (NEW - BC), the authorised reading:** IDENTICAL on task and cost for all three
seeds in both modes — task delta 0, cost delta 0, zero target-set differences; for seed 33 greedy,
NEW equals BC per-scene in 32/32 scenes (on_time and bytes). The restricted late adaptation produced
**no measured task or cost change versus the initial value** on this block.

**Structural contrast (NEW - OLD):** NEW never loses a target and gains 1-2 (mostly scene
1800014/target-2; seed 33 also 1800030/target-2), but is **more expensive than OLD in every case**
(`scenes_NEW_cheaper_than_OLD = 0/32` for all seeds; cost CI lower bounds >= 0). There is **no
cross-seed cost reduction**. Per seed, greedy:

| seed | NEW on-time | OLD on-time | task Δ (CI95) | NEW B/scene | OLD B/scene | cost Δ B/scene (CI95) |
|---|---|---|---|---|---|---|
| s11 | 183 | 182 | +0.031 [0, 0.094] | 4066.5 | 3753.6 | +312.9 [0, 722.6] |
| s22 | 183 | 182 | +0.031 [0, 0.094] | 4066.5 | 3922.1 | +144.4 [0, 361.3] |
| s33 | 183 | 181 | +0.0625 [0, 0.156] | 4066.5 | 3471.2 | +595.3 [168.6, 1095.8] |

Sampled mirrors greedy (NEW gains vs OLD, costs more; NEW == BC exactly): s11 +228.4, s22 +144.4,
s33 +623.1 B/scene vs OLD; task +0.031/+0.031/+0.0625.

**vs rules (greedy, task and cost both reported):** NEW is far cheaper than immediate
(-10290.0 B/scene, same task 183), essentially matches late_discovery_gate (+168.6 B/scene, same
task 183), and beats always_hold on task (+0.406 mean, 13 targets gained across scenes) at +4066.5
B/scene. No target is lost vs any rule arm.

**Early key target — focal scene 1700027 / target-5:** all six focal cells (NEW and BC x 3 seeds,
greedy) reach on_time=6 including target-5 (spent 7500B). **NEW preserves 1700027/target-5 for every
seed including seed 33**, identical to the initial value; the loss full R60 training caused for seed
33 is not reproduced by the restricted family.

## 6. Conclusions and limits

- **Kept:** the restricted (frozen-early-path) family preserves the initial-value task outcome
  exactly on this block (NEW == BC) and preserves the early key target 1700027/target-5 for all
  seeds; it never loses a target versus OLD and gains a few.
- **Not achieved:** no cost reduction. NEW is uniformly more expensive than OLD (0/32 scenes
  cheaper) and unchanged versus BC. The added-learning reading shows zero movement on task or cost.
- These are results for the **overall partitioned-actor restricted family on this single development
  block only**. They do not prove parameter-sharing interference was the sole root cause, do not
  establish global optimality, non-inferiority or generalisation, and are not a deployment basis.
- NEW == BC is behavioural equivalence of measured outcomes; it does not prove the trained
  parameters, gradient norms, clipping or Adam trajectories equal the initial value.
- CI containing 0 is not equivalence. Rule arms are greedy-only; sampled NEW-vs-rule is explicitly
  non-applicable.

## 7. Deliverables and provenance

Committed (code / tests / JSON summaries / reports): `r66_eval_eligibility_audit.py`,
`test_r66_eval_eligibility_audit_zero_env.py` (24/24 pass), `raw/runtime/ROUND66_MACHINE_VERDICT.json`,
`raw/runtime/ROUND66_LEDGER_FINAL.json`, `raw/runtime/ROUND66_CHECKPOINT_INDEX.json`,
`raw/runtime/INDEPENDENT_REVIEW_INPUTS.json` (678 cells, 0 closure violations),
`raw/runtime/ROUND66_EVAL_ELIGIBILITY_AUDIT.json`, `raw/ROUND66_CORRECTED_STATISTICS.json`,
this report. Git-external (weights, raw rows, tapes, lifecycle, transcripts, logs, claims, slots) are
listed in the manifest and NOT committed. Append-only registers preserved:
`ROUND66_DEVIATION_REGISTER.json` (D1), `ROUND66_EVAL_ADAPTER_PERSIST_FIX_REGISTER.json`,
`ROUND66_FREEZE.json` + addenda. Protected ROUND46 files untouched.
