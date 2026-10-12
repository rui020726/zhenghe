# ROUND65 Critical-State Training-Signal Audit (zero-environment, read-only)

Branch `analysis/round65-critical-state-training-signal`, authored against base HEAD
`a28e516bf1e3bc9a906379c521fffaf52a7d4397` (R64 final). Machine artifacts:
`TRAINING_COVERAGE.json`, `LEARNING_SIGNAL_AUDIT.json`, `FOCAL_POLICY_PROBE.json`,
`INPUT_SOURCES.json`, `ROUND65_FREEZE.json`. The delivery commit that adds this report is named
in the final message; raw tapes/rows/weights are Git-external (only summaries + path SHAs here).

Focal case: scene `1700027` / `piloc-e1-target-5`, first legal free first-report at step 134,
`ready_step=134` (early route). The 19-D float32 vector is **parsed from the R64 L baseline tape**
(`v2_L_a03_s1700027_a3_tape.jsonl`, SHA `2c3546af...`, tick 134, target index 0, recorded
`p_send=0.1937039047`) and validated equal to the `r65_common` constant at startup; the parsed
value is authoritative and matching aborts on any drift. This was exercised: an earlier hand-typed
hex error correctly triggered the abort rather than silently matching.

## Scope and side effects

Read-only only. No environment/episode/provider/claim/slot, no real or synthetic optimizer update,
no PPO replay/autograd, no training RNG/optimizer/λ/cursor restore, no checkpoint write-back, no
β/147/encoding/reward/GAE/model change, no rule/threshold fitting. The **only** model activity is
**9 read-only CPU `no_grad` forwards** (3 seeds × ep0/ep64/ep128) through the real R60 partitioned
controller, whose canonical modules are bound and SHA-verified via
`r60_partitioned.configure_source_imports()` from `D:/code/zhenghe/runs/comm_cost_v2_rerun_20260928`
(no hand-written network). Real training/eval/episode side effects: **0**.

## 1. Training support for the focal encoding (exact / partial)

Scanned all free first reports in the 384 compliant training episodes (index built from the 96
attempt01 batch records, never globbed; isolated non-compliant constructions excluded). 2286 free
first-report candidates total.

- **T0 exact (19-D float32 byte-identical AND free first report): 0.** The focal vector never
  appears in training.
- **T1 partial (same early route + `ready_step==134` + backlog==1 + has_future==1): 0.**
  `ready_step==134` occurs **0** times among 1767 early candidates (range 1..147; 600 of them sit
  at step 1). The nearest represented ready_step is 147 (6 candidates).
- **T2 context:** 1767 early free first reports (only **6 SEND / 1761 SKIP**, 99.7% SKIP) and 519
  late (484 SEND / 35 SKIP). Early budget is always full; backlog is usually 2–4 (multi-target
  episode-start discoveries).
- Diagnostic near-miss (**not** T1, no tolerance loosened): 177 early candidates share the focal's
  backlog==1 AND has_future==1; 246 share backlog==1; 1248 share has_future==1 — but none share
  `ready_step==134`.

Cross-check: persisted `update.early_free/4 = 1767` and `update.late_free/4 = 519` exactly equal the
tape-scan tier counts (the update counts are 4× epoch-multiplied); action encoding matched
`collector_summary` on all 384 episodes (0 mismatches).

**Answer:** the focal encoding has **zero exact and zero partial (same-ready-step) training
support**; the policy was never trained on this state.

## 2. Two actions, task loss, task-advantage signal

- **Two actions:** yes. Early free first reports contain both SEND (6) and SKIP (1761); late contain
  SEND (484) and SKIP (35). Early is overwhelmingly SKIP.
- **Task loss (constraint):** `g_hat==0` in **95/96** batches; `g_hat>0` in exactly **1**
  (seed 33, batch 22, scene 1300019, `y_reference=6` vs `y_policy=5`, one task lost; λ 1.0→1.125,
  eta 0.5). λ final: s11=1.0, s22=1.0, s33=1.125.
- **Task-advantage signal:** the explicit task-loss gradient (`g_hat`-driven) is essentially absent
  (1/96). The value-based PPO surrogate exists per batch, but per-sample `A_task`/`A_cost` and the
  combined advantage are **not persisted** by R60 and are marked `not_persisted` (not backfilled;
  no gradient replay). `g_hat==0` means the constraint held, **not** that per-message task value is 0.

## 3. Same-input p, ep0 → ep128 (9 forwards, real routed controller)

| seed | ep0 pSEND | ep64 pSEND | ep128 pSEND | greedy ep0 → ep128 |
|---:|---:|---:|---:|---|
| 11 | 0.919 | 0.998 | 0.027 | SEND → SKIP |
| 22 | 0.912 | 0.477 | 0.884 | SEND → SEND |
| 33 | 0.913 | 0.119 | 0.194 | SEND → SKIP |

The s33 ep128 forward reproduces the R64 recorded `p_send=0.1937039047` with `abs_gap=0.0`
(faithfulness proof); all 9 checkpoints verified against the identity chain (ep0 identity file;
batch b15/b31 `parameters_after_sha`). R60 training drove seeds 11 and 33 from a SEND-leaning ep0
init to **SKIP** on this early first report; seed 22 stayed SEND. V(s) and V_cost are state returns,
not the marginal value of SEND−SKIP.

## 4. Evidence, and what stays undecidable

- **Insufficient coverage (evidence):** T0=0, T1=0 — the focal state is unrepresented; training
  early first reports are dominated by step-1 multi-target discoveries.
- **Insufficient feedback (evidence):** the task-loss constraint channel is silent in 95/96 batches;
  per-sample advantages are not retained, so the surrogate's per-row behavior cannot be reconstructed.
- **Unused-signal (characterization):** 99.7% early-SKIP plus a silent task-loss channel means little
  pressure existed to SEND valuable early lone targets. This is a description, not a proven cause.
- **Undecidable — no loss root cause is presupposed.** This audit cannot attribute the target-5 loss
  to lack of coverage, insufficient feedback, the partitioned structure, or shared interference. It
  shows the state was never trained on and task-loss feedback was near-silent, and that the policy
  flipped SEND→SKIP across training, but not *why*. Identical 19-D would still not mean identical
  world; the 177 near-miss states overlap on non-time flags yet differ in discovery timing.

## 5. Single minimal follow-up (recommendation only — not launched)

Run a **registered SEND/SKIP paired intervention on the frozen seed-33 ep128 model at the nearest
*represented* early lone-target first reports** (`ready_step` 147, then 119), reusing existing saved
R60/R58/R64 trajectories where possible. This tests whether the learned early-SKIP is actually
costing on-time tasks on states the policy *did* see — the question the absent focal state
(`ready_step` 134) cannot answer, and one R64 already answered for the focal itself. Resources:
read-only reuse of existing cells plus a small registered intervention quota on the frozen model;
**zero training**, no β/147/model change. Any broader training change should wait for that result.

## Missing / limits

Per-sample `A_task`/`A_cost`, old-value history and minibatch ratio/clip/entropy arrays were not
persisted by R60 and are reported as `not_persisted`, not reconstructed. The 32-scene dev block
(1700000..1700031) is not an independent confirmation set. No new significance, sample-size,
non-inferiority or deployment claim is made; V and joint advantage are not treated as message value.

## Tests and accounting

`test_r65_pure.py`: 17 tests, all pass (float32 identity incl. 1-ULP rejection; ready_step/routing;
tier classification; joint row/candidate non-independence; repeated-scene non-independence; masked
exclusion; missing-not-zero; seed-role mismatch rejection; focal-constant-drift abort). Real side
effects: 0 training/eval episodes; 9 read-only forwards; files written only under the R65 audit dir.
