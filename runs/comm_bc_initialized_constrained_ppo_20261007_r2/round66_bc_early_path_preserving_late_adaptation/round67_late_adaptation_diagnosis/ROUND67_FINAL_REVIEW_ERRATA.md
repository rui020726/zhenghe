# ROUND67 Final-Review Errata (append-only, zero side effects)

**Task:** ROUND67 final effectiveness + cumulative-consumption reconciliation (incremental, no rerun)
**Branch:** `analysis/round67-late-adaptation-diagnosis`
**Reviewed delivery HEAD:** `36931efc099ef280211646b7980e0d2ecdaca009`
**Nature:** read-only reconciliation. 0 environment constructions, 0 simulations, 0 new claims,
0 real or synthetic optimizer updates, 0 PPO calls, 0 RNG restores, 0 checkpoint writes,
0 parameter or threshold edits, **0 additional forward rows**. Only read-only parsing of already
persisted artifacts plus pure arithmetic. No original ROUND67 deliverable is modified; this file and
its machine verdict are appended.
**Status:** three precise wording/attribution corrections issued; the core findings are retained.

## Why this errata exists

The final review flagged three places where the ROUND67 report/verdict assert more than the
underlying records support. Each is corrected below against actual source and persisted artifacts.
None of the three changes the primary scientific finding (NEW==BC action-for-action; mechanism =
softmax saturation); they only tighten what may be claimed about clip, sampling, and the forward guard.

## Correction 1 - "clip_fraction = 0, so the global clip never bound" (category error, and inverted)

Source of truth: `r66_frozen.py` (`frozen_early_ppo_update`).

- `clip_fraction` (recorded at `r66_frozen.py:876`) is the **PPO importance-ratio clip fraction**:
  `((ratio - 1.0).abs() > common.CLIP).float().mean()` at `r66_frozen.py:850`, with
  `common.CLIP = 0.2` (`v2_common.py:72`). It is the fraction of samples whose policy ratio fell
  outside [0.8, 1.2]; value 0.0 means no ratio sample was clipped (approx_kl ~ 0). It says nothing
  about gradient clipping.
- The **global gradient-norm clip** is separate:
  `torch.nn.utils.clip_grad_norm_(trainable_parameters(controller), common.MAX_GRAD_NORM)` at
  `r66_frozen.py:840-841`, with `common.MAX_GRAD_NORM = 0.5` (`v2_common.py:76`). Its only persisted
  trace is the pre-clip total norm over the six trainable objects, recorded as `gradient_norm_mean`
  (`r66_frozen.py:877`): **7.5252 / 16.1051 / 11.4117** for s11 / s22 / s33.
- Those norms are **15.1x / 32.2x / 22.8x** the 0.5 cap, so the global grad clip scaled the gradients
  on essentially every minibatch (implied coefficient ~ 0.066 / 0.031 / 0.044). The exact per-minibatch
  clip coefficient is **not recorded** (missing), but the direction is unambiguous: the global grad
  clip **bound constantly**, the opposite of "never bound."

Correction: the report sentence "clip_fraction is 0.0000 for every seed, so the global clip never
bound an update" is a category error and is factually inverted. The machine-verdict phrase "the
never-binding clip" (in `conclusions_deferred`) is wrong and becomes "the global grad clip bound on
essentially every minibatch (pre-clip norm 7.5-16.1 >> MAX_GRAD_NORM 0.5)." This does not change the
saturation mechanism (the null is about action saturation); it removes one wrong supporting detail and
one wrong co-occurring factor.

Adam attribution softened: "Adam then normalises that tiny but consistently signed gradient into
near-lr steps, which is why late_head moved by L2 0.090-0.237" is an inference from Adam per-parameter
normalisation, **not a measured per-step update**. The batch records carry no Adam step sizes, no
per-parameter update magnitudes, and no lr trajectory, and the (binding) global grad clip scales the
late_head gradient by ~0.03-0.07 before Adam sees it. State it as "consistent with Adam normalisation,"
not as the confirmed sole cause. The multi-cause caveat already present in the report stands.

## Correction 2 - "sum|dp| = 0.0538 expected flips, P(0) ~ 0.947" (assumption-dependent approximation)

Measured, assumption-free facts (`r67_paired_probabilities.json`, block
`sampled_late_free_expected_draw_flips`): **147** sampled late free records (49/seed), sum|dp| =
0.0537558794, per-seed max|dp| = 0.0058 / 0.0140 / 0.0054, every record has p(SEND) >= 0.939 in both
arms, **0 observed action differences**.

The "expected number of draw flips = sum|dp|" and "P(0) ~ exp(-0.0538) = 0.9477" are a **descriptive
approximation** resting on two assumptions the paired script does not verify (its own `note` states
the first):
1. Coupling: NEW and BC draw their action from the SAME uniform (common-random-number inverse-CDF), so
   the per-row flip probability equals |dp|. Same-seed reuse does not by itself establish this; the
   script reads only saved probabilities/actions and never inspects the eval sampler RNG scheme.
2. Independence: the 147 rows are independent Bernoulli trials, so #flips ~ Poisson(sum|dp|). They are
   not: they lie on coupled trajectories where the state at row k+1 depends on the action at row k.

Downgrade: do not claim "P(0) ~ 0.947 is the real execution probability of zero flips" or "0.0538 is a
general expected flip count." State instead: under the coupled-independent approximation sum|dp| ~
0.054 would predict ~0.05 expected flips (P(0) ~ 0.95); but the actual sampled run is a single coupled
realisation, and - assumption-free - observing 0 differences is expected simply because both arms sit
at p(SEND) >= 0.939, so even independent draws almost never differ. Retain the measured statement: 147
records, small probability changes, 0 action differences. The machine-verdict field
`expected_draw_flips_sum_abs_dp` is an approximation, not a measured expectation.

## Correction 3 - 4711 forward rows: the round-level ledger is retrospective, not a pre-run hard guard

The actual pre-forward hard gate for the panel is the **local in-process `RowGuard`** in
`r67_probability_panel.py`: `FORWARD_ROW_CAP = 5000` (line 49), `class RowGuard` (line 85), `charge()`
(line 93), plan pre-check `if planned_rows > FORWARD_ROW_CAP: raise` (lines 242-243),
`guard.charge(2 * mat.shape[0], ...)` (line 254) before each forward. It capped the panel own 4708 rows
within that one process. It does **not** read the round-level ledger, so its scope is the panel process
only.

The round-level ledger `r67_forward_row_ledger.json` is **retrospective**, not a live cumulative guard.
Evidence from artifact mtimes:
- panel output `r67_probability_panel.json` at 21:19:34 (forwards already done; the panel never touched
  the ledger);
- focal probe output `r67_focal_attribution_probe.json` at 21:28:04 - the focal probe reads/appends the
  ledger, but at that moment the file did not yet hold the panel rows, so its cross-process check saw
  used = 0 (it wrote a 1-event, 3-row ledger);
- `_patch2.py` at 21:29:59 first seeds the ledger by reading the panel OUTPUT json (so it post-dates the
  panel forwards) and overwrites it with only the panel event;
- `_patch4.py` at 21:34:24 rebuilds the ledger with both events (4708 + 3) from the two output jsons.

So the EXECUTION_BASIS claim "a persisted round-level ledger ... is read and appended by every script
that forwards, so the cap holds across processes" is inaccurate: the panel never read/appended it, and
when the focal probe ran the ledger did not yet contain the panel rows, so no cumulative cap was
actually enforced across the two scripts. The cap was nonetheless **not breached** (4708 and 4711 both
<= 5000); this correction is about enforcement mechanism/timing, not a violation.

Duplicate-forward from a failed or rerun invocation: the panel output records one clean run
(`forward_batches_charged = 9`, `forward_rows_used = 4708`) and there are no invocation logs for the
round67 scripts (the only .log files are earlier R66 train/eval runs). Because `RowGuard` is in-memory
and the ledger was backfilled only from the final output, an earlier crashed invocation would leave no
persisted trace. Duplicate-forward exposure is therefore **unknown** (cannot be excluded from
artifacts); no forward is run to check.

## Retained unchanged

All primary ROUND67 findings stand: NEW==BC action-for-action and message-for-message on 192 eval + 6
focal cells (only wall-clock `seconds` differs); the forward path reproduces 2232/2232 saved
probabilities within 1.19e-07; the early route is bit-identical; the late route is saturated
(p ~ 0.999, max|dp| 0.0140, 0 argmax flips, 140/300 late free at p = 1.0 exactly); the late training
signal is thin and saturated; the beta=1.00 prior has no trainable path (measured 0.0 on a live graph);
focal attribution (R60 ep128 flips the early head on scene 1700027/target-5 for s11 and s33). The three
corrections above only remove over-claims.

## Consumption of THIS reconciliation (zero side effects)

| Item | Count |
|---|---|
| Environment constructions / simulations | 0 |
| New claims / slots | 0 |
| Real or synthetic optimizer updates / PPO calls | 0 |
| RNG restores / checkpoint writes / parameter or threshold edits | 0 |
| Additional read-only forward rows | 0 |
| Read-only JSON/JSONL/source parses | report, verdict, basis, ledger, panel+paired+exposure+probe jsons, 96 batch records, gate files, r66_frozen.py, v2_common.py |
| Pure arithmetic | exp(-0.05376) = 0.947663; clip coefs 0.0664 / 0.0311 / 0.0438 |

Protected ROUND46 files re-verified byte-unchanged: `ROUND46_CURRENT_TASK.json` cf3ad198...,
`ROUND46_R2_V6_STOP_STATE.json` 93a2f69f.... No push, no merge, no deploy.

## Evidence paths
- Clip source: `.../round66_bc_early_path_preserving_late_adaptation/r66_frozen.py:840-841,850,876-877`
- Constants: `runs/comm_cost_v2_rerun_20260928/v2_common.py:72` (CLIP=0.2), `:76` (MAX_GRAD_NORM=0.5)
- Sampling: `.../round67_late_adaptation_diagnosis/r67_paired_probabilities.py` + `.json`
- Guard/ledger: `.../round67_late_adaptation_diagnosis/r67_probability_panel.py:49,85,93,242-243,254`;
  `_patch2.py`; `_patch4.py`; `r67_forward_row_ledger.json`
- Batch records: `.../raw/checkpoints/r40_batch_s{seed}_b{batch}_attempt01.json`
