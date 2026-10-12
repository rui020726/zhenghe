# ROUND67 Report — Read-Only Diagnosis of the ROUND66 Late-Adaptation Null Result

**Task:** ROUND67-READONLY-LATE-ADAPTATION-DIAGNOSIS
**Branch:** `analysis/round67-late-adaptation-diagnosis`
**Base HEAD (ROUND66 closing errata):** `c962a2ceccb5ee767debef349eac9cac66c77706`
**Delivery HEAD:** set by the commit that adds this file
**Nature:** zero-side-effect mechanism analysis. 0 environment constructions, 0 new claims,
0 real or synthetic optimizer updates, 0 PPO calls, 0 RNG restores, 0 checkpoint writes,
0 parameter or threshold changes. Read-only CPU forward under `torch.no_grad()` only,
**4711 of 5000 permitted input rows** (panel 4708 + focal attribution probe 3).
**Status:** COMPLETE — the null result is explained, with one mechanism confirmed and one
explicitly left unproven. No push, no merge, no deploy.

## 1. The question and why it needed answering

ROUND66 froze `trunk` + `early_head` and trained only `late_head` and both critics, starting
model-only from each seed's compliant ROUND60 ep0. The trainable parameters demonstrably moved
(`ROUND66_REVIEW_PARAM_DIFF.json`: `late_head` L2 0.090 / 0.237 / 0.116 for s11 / s22 / s33, both
critics moved, `trunk` and `early_head` bit-identical). Yet NEW and BC agreed on on-time task
count, billed bytes and on-time target-ID sets in **all 192 eval cells** and all 6 focal cells.

That combination is not self-explanatory. Either the actions never changed, or they changed and
cancelled. Either way the interesting follow-up is the same: where do the decisions that actually
matter live, and is there direct evidence of saturated (high-confidence) decisions?

## 2. Task 1 — action and emission equivalence (no forward, pure persisted data)

Compared every NEW cell against its BC cell on the trajectory fields, not just the outcome fields,
across all 3 seeds x 2 modes x 32 scenes (192 cells) plus the 6 focal cells
(`r67_action_equivalence.py` -> `r67_action_equivalence.json`).

| Check | Result over 192 eval cells | Result over 6 focal cells |
|---|---|---|
| rows / tapes missing | 0 / 0 | 0 / 0 |
| per-target action maps equal | **192 / 192** | **6 / 6** |
| free first-report target sets equal | 192 / 192 | 6 / 6 |
| emission sequences equal | 192 / 192 | 6 / 6 |
| refusal sequences equal | 192 / 192 | 6 / 6 |
| `task_reward_events` equal | 192 / 192 | 6 / 6 |
| `behaviour_sha256` equal | 192 / 192 | 6 / 6 |
| `on_time` / `spent_bytes` / `emissions` equal | 192 / 192 | 6 / 6 |
| any outcome field differing | **189 / 192 — and the only differing field is `seconds`** (wall clock) | 3 / 3, same field |

So the answer to "did actions change at all, or change-and-cancel?" is: **they did not change at
all.** Not one free first-report action, not one emission, not one refusal, in either decoding mode.
The R66 equivalence is therefore stronger than the outcome-field equivalence the errata had already
narrowed it to: it is action-for-action and message-for-message identical on this block, while the
probabilities underneath are not (next section).

## 3. Tasks 2 and 3 — joint probability panel (4708 forward rows, guarded)

Built from the pre-action 19-dim float32 inputs persisted in the decision-step tapes. Distinct key
= (seed, exact float32 bytes of the input); the same input reached from any arm, mode or scene is
forwarded once per model and is never filtered by outcome (`r67_probability_panel.py` ->
`r67_probability_panel.json`).

| Panel | Contents | Distinct inputs | Forward rows |
|---|---|---|---|
| A | **complete census** of every distinct FREE first-report input, early and late | 374 | 748 |
| B | quota-limited sample of distinct LATE non-free (forced-hold) inputs, 620/seed | 1860 of 2888 (64.4%) | 3720 |
| C | quota-limited sample of distinct EARLY non-free inputs, 40/seed | 120 of 9075 (1.3%) | 240 |
| — | total | 2354 | **4708** (cap 5000) |

**Forward-path verification first:** every one of the 2232 saved probability records (1116 NEW +
1116 BC, i.e. 100% of the records that carry a saved probability) was recomputed with the real
routed forward and compared to the tape value: **2232 / 2232 matched, 0 mismatched, max absolute
difference 1.19e-07** against the 1e-6 tolerance. The panel is therefore measuring the same
quantity the run recorded.

**Counting note.** The dedup key is (seed, exact float32 input), so an input reached from both
decoding modes is forwarded once but appears in both mode rows of the tables below. Panel A is a
complete census: 374 distinct free first-report inputs (220 early + 154 late) over the 192 eval
cells and the 6 focal cells. The available non-free pools are 2888 distinct late and 9075 distinct
early inputs.

**Early route (frozen path) — exactly identical, as the structure requires:**

| seed / mode | distinct early free inputs | max abs dp | max abs d(logit margin) | p(SEND) range | saturated (p>=0.99 or <=0.01) |
|---|---|---|---|---|---|
| s11 greedy | 73 | **0** | **0** | [9.5e-08, 0.960] | 70 / 73 |
| s11 sampled | 69 | **0** | **0** | [9.5e-08, 0.959] | 67 / 69 |
| s22 greedy | 73 | **0** | **0** | [3.9e-07, 0.949] | 70 / 73 |
| s22 sampled | 69 | **0** | **0** | [3.9e-07, 0.949] | 67 / 69 |
| s33 greedy | 73 | **0** | **0** | [1.2e-07, 0.955] | 70 / 73 |
| s33 sampled | 69 | **0** | **0** | [1.2e-07, 0.955] | 67 / 69 |

`trunk` and `early_head` are bit-identical between the arms, so the early logits are bit-identical:
measured max abs dp = 0 and max abs dlogit = 0 on every early row forwarded, free or not. This is a
measured consequence, not an assumption.

**Late route (the head that moved) — logits moved, probabilities barely did:**

| seed / mode | distinct late free inputs | mean abs dlogit | max abs dlogit | max abs dp | rows with dp != 0 | p(SEND) range | min distance to 0.5 | argmax flips |
|---|---|---|---|---|---|---|---|---|
| s11 greedy | 51 | 0.550 | 0.986 | 0.0058 | 27 / 51 | [0.949, 1.000] | 0.449 | **0** |
| s11 sampled | 49 | 0.554 | 0.986 | 0.0058 | 25 / 49 | [0.949, 1.000] | 0.449 | **0** |
| s22 greedy | 51 | 1.422 | 2.620 | **0.0140** | 29 / 51 | [0.954, 1.000] | 0.454 | **0** |
| s22 sampled | 49 | 1.432 | 2.620 | 0.0140 | 27 / 49 | [0.954, 1.000] | 0.454 | **0** |
| s33 greedy | 51 | 0.656 | 1.230 | 0.0054 | 27 / 51 | [0.948, 1.000] | 0.448 | **0** |
| s33 sampled | 49 | 0.661 | 1.230 | 0.0054 | 25 / 49 | [0.948, 1.000] | 0.448 | **0** |

Across all late rows forwarded (n = 2014): mean p(SEND) 0.9976 (NEW) vs 0.9975 (BC), binary entropy
0.0061 vs 0.0065 bits, 1984 / 1981 rows saturated, minimum distance to the 0.5 decision boundary
0.078, and **0 argmax flips**.

**Why the logit movement does not reach the action.** The softmax gain is dp/dm = p(1-p). On these
rows p is ~0.999, so the gain is ~0.002: a mean logit move of 1.42 (s22) can shift p by at most
~0.003, and the observed maximum is 0.0140. Two further dampers are visible in the data:

- **140 of the 300 late free records sit at p = 1.0 exactly in both arms** (float32 softmax
  saturation), where the gain is literally zero and a logit move of any size leaves p bit-identical.
- The remaining 160 records moved by at most 0.0140, and **not one of the 300 late free records has
  p below 0.94 in either arm**, so none is anywhere near the 0.5 boundary.

**Sampled mode is explained too.** A uniform multinomial draw on a two-way distribution flips with
probability |dp| on an affected row, so the expected number of draw flips is the sum of |dp| over the
sampled late free rows: **0.0538 over 147 rows**. The observed number of action differences is 0.
Observing zero is the expected outcome (P(0) ~ 0.947), not a coincidence that needs explaining
(`r67_paired_probabilities.py` -> `r67_paired_probabilities.json`).

**Saturation is therefore directly evidenced, not inferred:** late free p(SEND) median 1.0,
288 / 300 records >= 0.99, 300 / 300 take SEND, binary entropy 0.006-0.019 bits; early free p(SEND)
median ~2e-06 with 802 / 816 records taking SKIP. Exploration on this block is the stochastic draw
plus the entropy bonus, and both are ~0 where it matters.

## 4. Task 4 — training exposure (`r67_training_exposure.py` -> `r67_training_exposure.json`)

Parsed all 96 committed batch records (`raw/checkpoints/r40_batch_s{seed}_b{batch}_attempt01.json`),
the three first-batch gates, the durable journals, and the 384 persisted training tapes.

| seed | updates | optimizer steps | late free rows (4 epochs summed) | early free rows | gate rows | late_head grad L1 mean | grad norm mean | clip fraction |
|---|---|---|---|---|---|---|---|---|
| s11 | 32 | 180 | 692 | 2356 | 2356 | 3.568e-03 | 7.53 | 0.0000 |
| s22 | 32 | 180 | 692 | 2356 | 2356 | 1.710e-03 | 16.11 | 0.0000 |
| s33 | 32 | 180 | 692 | 2356 | 2356 | 3.906e-03 | 11.41 | 0.0000 |

- The actor loss sees **free rows only**: `joint_log_prob` and `action_entropy` both multiply by the
  recorded `usable` mask, so a candidate the silence bound has taken over contributes no likelihood
  and no actor gradient. The late head's entire actor signal is the ~173 late free choices per seed
  across 128 episodes (5-8 per minibatch).
- Those choices were already saturated at initialization: training late free p(SEND) median
  0.99999, min 0.864, 161-163 / 173 >= 0.99; 172-173 of 173 took SEND. The early free choices were
  saturated the other way (median ~2e-06, 586-588 of 589 SKIP).
- Recorded pre-clip gradient readings (per-minibatch L1, trainable modules only; `trunk` and
  `early_head` are `None` because `requires_grad=False`, a structural fact and not a measured zero):

  | seed / update | early free rows | late free rows | base trainable L1 | late actor L1 | critic L1 | prior trainable L1 |
  |---|---|---|---|---|---|---|
  | s11 u0 | 16 | 6 | 45.80 | 5.66e-05 | — | **0.0** |
  | s11 u15 | 14 | 8 | 41.99 | 1.36e-01 | — | **0.0** |
  | s11 u31 | 17 | 5 | 36.27 | 6.05e-05 | — | **0.0** |
  | s22 u0 | 16 | 6 | 180.16 | 1.23e-04 | — | **0.0** |
  | s22 u15 | 14 | 7 | 126.52 | 4.98e-03 | — | **0.0** |
  | s22 u31 | 17 | 5 | 60.72 | 3.70e-05 | — | **0.0** |
  | s33 u0 | 16 | 6 | 157.43 | 3.95e-05 | — | **0.0** |
  | s33 u15 | 14 | 7 | 83.97 | 4.34e-03 | — | **0.0** |
  | s33 u31 | 17 | 5 | 39.79 | 1.01e-05 | — | **0.0** |

  The late actor gradient is 4-6 orders of magnitude below the critic gradients, and it is tiny
  precisely because p ~ 1 makes d(log pi)/dlogits ~ (1-p) ~ 1e-05. Adam then normalises that tiny
  but consistently signed gradient into near-lr steps, which is why `late_head` moved by L2
  0.090-0.237 while its behavioural effect stayed ~0.
- **The beta = 1.00 gate prior has no trainable role left.** The prior adds `-log p(SKIP)` over early
  first-emission free candidates only; every one of those rows routes to the frozen `early_head`, and
  the trunk is frozen too. The recorded reading is `prior_trainable_grad_l1 = 0.0` at all nine diag
  updates with `prior_output_requires_grad = true` — a *measured* zero on a live graph, not an absent
  reading. The prior still shapes the frozen early path's behaviour, but it can no longer move any
  trainable parameter.
- `clip_fraction` is 0.0000 for every seed, so the global clip never bound an update; the R66 clip
  norm is over the six trainable objects only and is therefore not comparable with R60's twelve-object
  norm. `g_hat` stayed at 0.0 and lambda stayed at 1.0 throughout, so the dual update was inert.
- Missing rather than reconstructed: per-update `lambda_history` is not carried in the batch record
  (only the per-batch `dual` record is); `gate_diag` exists only for diag updates 0, 15 and 31 and is
  absent, not zero, elsewhere. No minibatch order was fabricated and no PPO update was replayed.

## 5. Task 5 — the focal attribution probe: where the decisions that matter actually live

The 3-row read-only probe (`r67_focal_attribution_probe.py`) forwards the **ROUND60 ep128**
controller on exactly the three first-report inputs of scene 1700027 / `piloc-e1-target-5` that R61
froze (step 134, `ready_step = 134`, therefore **early-routed**), and compares with ROUND60 ep0:

| seed | route | R60 ep0 p(SEND) | R60 ep0 action | R60 ep128 p(SEND) | R60 ep128 greedy action | R60 ep128 focal on_time |
|---|---|---|---|---|---|---|
| s11 | early | 0.9191 | SEND | **0.0268** | **SKIP** | 5 (target-5 lost, both modes) |
| s22 | early | 0.9123 | SEND | 0.8839 | SEND | 6 (target-5 kept) |
| s33 | early | 0.9132 | SEND | **0.1937** | **SKIP** | 5 greedy / 6 sampled |

This is the whole story in one table. ROUND60's full training flipped the early head's decision on
the focal key target for s11 and s33 — a 0.89 and 0.72 probability swing, on a row that is *not*
saturated — and that flip reproduces R60's observed focal pattern exactly, including the asymmetry:
s33 sampled kept target-5 because p = 0.194 still draws SEND often enough, and s22 never flipped at
all. ROUND66 froze precisely this path, so the decision is bit-identical to the initial value and
target-5 is preserved for all three seeds in both modes.

The contrast with section 3 is the point: **the head that governs non-saturated decisions is the one
ROUND66 froze, and the head it let move only ever governs saturated ones.**

## 6. What this supports, and what it does not

**Supported (on this development block, for this frozen policy and these three seeds):**

1. ROUND66's NEW and BC are identical action-for-action and message-for-message, not merely on
   outcome fields: 192 / 192 eval cells and 6 / 6 focal cells agree on every free first-report
   action, every emission, every refusal, every on-time target-ID set and every billed byte.
2. The mechanism is softmax saturation, not an absence of learning. `late_head` really moved (L2
   0.090 / 0.237 / 0.116); its logit output on the visited late states moved by mean 0.55-1.43 and
   up to 2.62; but those states sit at p(SEND) ~ 0.999 where the gain p(1-p) ~ 0.002, and 140 / 300
   late free records are at p = 1.0 exactly in float32. Maximum probability change 0.0140, zero
   argmax flips, and a sampled-draw exposure of 0.0538 expected flips against 0 observed.
3. The training signal that produced that movement was thin and saturated from the start: ~173 late
   free choices per seed over 128 episodes, 5-8 per minibatch, median p(SEND) 0.99999, late actor
   gradient L1 1e-05..1e-01 against critic gradients of order 1e+02.
4. The beta = 1.00 gate prior has no remaining trainable path under the frozen early path
   (measured `prior_trainable_grad_l1 = 0.0` on a live graph at all nine diag updates). It still
   shapes the frozen early behaviour; it can no longer move a parameter.
5. The early key target 1700027 / target-5 is preserved because its first report routes to the
   frozen early head, and that is exactly where ROUND60's full training did the damage.

**Not supported / must not be claimed:**

- Nothing here shows that late-head training is *generally* inert. It shows that on the states this
  frozen policy visits, the late head was saturated. A different initialisation, a different prior
  strength, or a state distribution with late rows nearer the boundary could behave differently.
- The null result cannot be attributed to any single cause (freezing, saturation, the thin late
  signal, the inert dual update and the never-binding clip all co-occur), and it says nothing about
  whether parameter-sharing interference was or was not a root cause of anything.
- Panel B and Panel C are quota-limited (64.4% of the 2888 distinct late non-free inputs, 1.3% of
  the 9075 distinct early non-free inputs). Those rows cannot change an action, so the quota does not affect any action-level
  conclusion, but no claim is made about the unsampled majority.
- The attribution probe covers exactly three inputs (one per seed), not a census of ROUND60's
  early-path changes; it confirms the mechanism on the focal target and does not enumerate every
  early decision ROUND60 moved.
- No generalisation, non-inferiority, deployment or confirmation-set claim. The evaluation block
  1800000-1800031 remains the already-used ROUND66 development block, and the focal scene 1700027 is
  a single scene used for mechanism inspection only.
- These are post-hoc selected cases, one frozen policy and one random realisation per cell; nothing
  here is a prospective trial, and no statistical inference is drawn from the 300 late free records.

## 7. Consumption record (this round, zero side effects)

| Item | Count |
|---|---|
| Environment constructions / simulations | 0 |
| New claims / slots | 0 |
| Real optimizer updates / PPO calls | 0 |
| Synthetic optimizer updates / checkpoint writes | 0 |
| RNG restores / parameter edits / threshold edits | 0 |
| Read-only `torch.load` checkpoint reads | 6 R66 ep128 + 3 R60 ep0 + 3 R60 ep128 = 12 |
| Read-only CPU forward rows | **4711 of 5000** (panel 4708 + focal probe 3) |
| Read-only JSON / JSONL parses | rows, tapes, 96 batch records, 3 gates, journals |

The forward-row ledger is persisted in `r67_forward_row_ledger.json` and is checked before any
forward runs, so the cap is enforced across processes rather than inside one script. The unused 289
rows were deliberately left unspent: every remaining question was answerable from persisted data.

## 8. Deliverables and provenance

Committed under `runs/.../round66_bc_early_path_preserving_late_adaptation/round67_late_adaptation_diagnosis/`:
`ROUND67_REPORT.md` (this file), `raw/runtime`-independent machine verdict
`ROUND67_MACHINE_VERDICT.json`, execution basis `ROUND67_EXECUTION_BASIS.json`,
`r67_action_equivalence.py` + `.json`, `r67_probability_panel.py` + `.json`,
`r67_paired_probabilities.py` + `.json`, `r67_training_exposure.py` + `.json`,
`r67_focal_attribution_probe.py` + `.json`, `r67_forward_row_ledger.json`,
`test_r67_zero_env.py` (8 / 8 pass, 0 forward rows).

All ROUND66 originals are preserved byte-identical, including `ROUND66_FINAL_REPORT.md`,
`ROUND66_MACHINE_VERDICT.json`, the freeze and its addenda, the deviation register, every row, tape,
claim, slot and checkpoint. The two protected ROUND46 files were re-verified byte-unchanged
(`cf3ad198...`, `93a2f69f...`) before the commit, and the R66 ep128 / R60 ep0 / R60 ep128 checkpoint
bytes were re-hashed against the frozen identity and the checkpoint index.
