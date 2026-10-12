# ROUND66 Final Review — Wording Errata and Read-Only Audit Addendum

**Task:** ROUND66-BC-EARLY-PATH-PRESERVING-LATE-ADAPTATION
**Branch:** `exp/round66-bc-early-path-preserving-late-adaptation`
**Audited code HEAD:** `39ac0a41b932ebccfb744cea365cdb9dcf0fce6a`
**Delivery HEAD:** `254ea798072646094dd05a55af53a58a85fb9824`
**Review commit HEAD:** set by the errata commit that adds this file
**Nature:** zero-environment final review. 0 new environment constructions, 0 real/synthetic
optimizer updates, 0 PPO, 0 RNG restores, 0 checkpoint writes. Read-only only: 6 `torch.load`
checkpoint reads (parameter diff) and 2 pure-JSON reads (projection). No push, no merge, no deploy.

This addendum APPENDS corrections and read-only evidence. Every original is preserved
byte-identical: `ROUND66_FINAL_REPORT.md`, `ROUND66_MACHINE_VERDICT.json`,
`ROUND66_LEDGER_FINAL.json`, `ROUND66_FREEZE*.json`, `ROUND66_DEVIATION_REGISTER.json`, and all
rows / claims / checkpoints / logs. Protected ROUND46 files re-verified byte-unchanged
(`cf3ad198...`, `93a2f69f...`).

## E1. D1 wording — "interrupted, cause and step count UNKNOWN", not "hard-killed"

The final report (section 2) and the deviation register (D1) describe the first detached launch as
**"hard-killed"** inside the environment constructor. The deviation register's own `not_proven`
field already states that positive termination by process handle is **not** established and that
termination is established at artifact level only. "Hard-killed" asserts a specific termination
mechanism/cause that the evidence does not establish.

**Correction:** D1 is an *interrupted/terminated* attempt; the cause (including whether it was
"hard-killed") and the executed step count are **UNKNOWN / unproven**. Only artifact-level facts
hold: no exit record; a frozen 1480-byte gate log with no traceback; a claim carrying
`constructor_called_at_unix` before the second launcher together with `completed_steps=0` /
`simulation_started=false`; and the process gone before the second launch began.

**Unchanged:** the accounting (2 real environment constructions for `r66_ref_immediate_s1300001`,
+1 unregistered extra, no refund, reserve capped at 39) and the value's deterministic eligibility
stand exactly as registered. Only the causal adjective is downgraded.

## E2. NEW==BC equivalence — measured-outcome fields only, not behaviour or parameter equivalence

The final report (section 6) and the machine verdict limitations call it **"behavioural
equivalence of measured outcomes."**

**Correction:** the equivalence is verified **only** on specific measured outcome fields: on-time
task count, billed bytes, and exact on-time target-ID sets (per-scene 32/32 for s33 greedy; task
delta 0, cost delta 0, zero target-set differences for all three seeds x both modes). **No**
complete action-by-action / per-message behaviour check was performed, so no general
"behavioural equivalence" is claimed. This is outcome-field equivalence on one development block.

**Reinforced by the read-only parameter diff (A3):** NEW is provably **not** equal to BC at the
parameter level, so the equivalence is strictly outcome-level.

## A3. Read-only ep0 -> ep128 parameter diff (all three seeds)

Method: `torch.load` (CPU, read-only) of R60 ep0 (`v2_train_s{seed}_before_b0.pt`, file SHA verified
against the frozen identity) and R66 ep128 (`v2_train_s{seed}_ep128.pt`, SHA matches the checkpoint
index); per-module bit-identity + L2 norm. No writes. Full numbers in
`raw/runtime/ROUND66_REVIEW_PARAM_DIFF.json`.

Identical pattern for s11 / s22 / s33:

- `trunk` — **bit-identical**, L2 = 0.0
- `early_head` — **bit-identical**, L2 = 0.0
- frozen path (trunk + early_head) — **all bit-identical**, L2 = 0.0
- `late_head` — moved (L2 0.090 / 0.237 / 0.116)
- `value_task_head` (task critic) — moved (L2 0.351 / 0.426 / 0.407)
- `value_cost_head` (cost critic) — moved (L2 0.072 / 0.084 / 0.117)
- trainable set (late + both critics) — moved (L2 0.369 / 0.494 / 0.439)

Interpretation: the frozen early path never moved (confirming the intervention), while the trainable
modules did move. So NEW's parameters differ from BC's even though the measured outcomes are
identical. This is the concrete basis for E2.

## A4. "4 synthetic updates" audit — not 4 optimizer.step, not 4 PPO calls

The synthetic cap is `synthetic_optimizer_update_cap = 32` (an optimizer.step-class cap). Evidence
from the preserved source and saved outputs:

- `test_r66_frozen_acceptance.py` (committed once at `7267fc7`, never edited) increments its
  `synthetic_updates` counter exactly twice: one bare `opt.step()` (section 6) + one
  `frozen_early_ppo_update(...)` call (section 10).
- Both saved wire-up outputs (`_accept_stdout.txt`, `_accept_stdout_verify.txt`) record
  `"synthetic_optimizer_updates": 2` and `"synthetic_update_cap": 32`.

Separately-counted totals:

- Synthetic **PPO-update calls** before training = **1** (not 4).
- Bare synthetic **optimizer.step()** outside a PPO update = **1**.
- `optimizer.step()` **inside** the one synthetic PPO update = `EPOCHS_PER_BATCH(4) x
  ceil(samples/256)` >= 4; exact value **not recorded** (the throwaway report was not saved).
- Test's own recorded counter = **2**.
- Ledger / verdict / addendum recorded value = **4**.

The "4" therefore matches none of: 4 PPO calls (only 1), the test counter (2), or a single-run
optimizer.step total (>= 5). The only arithmetically consistent reading is an aggregate across the
two wire-up runs (original 2 + independent re-run 2 = 4 synthetic-update events), which is an
inference, not an explicitly recorded derivation.

**Missing evidence (recorded as unknown; not filled by rerunning tests):** the exact optimizer.step
count inside the one synthetic PPO update, and the explicit derivation of the ledger "4".

**Cap safety:** under every preserved interpretation (2 events/run; 1 PPO/run; 5-18 optimizer.step
across up to two runs; or 4 aggregate events) consumption is <= 32. **No cap breach.** But "4" is an
unreconciled bookkeeping figure and must not be quoted as a precise optimizer.step or PPO-call count.

## A5. Compact numeric projection

`raw/runtime/ROUND66_REVIEW_NUMERIC_PROJECTION.json` (also printed to the tool output): 32 scenes;
per seed/mode/arm Y (on_time) / C (bytes) / target-bitmask vectors for 18 model arms
(NEW/OLD/BC x 3 seeds x 2 modes) and 3 rule arms (immediate, late_discovery_gate, always_hold);
bootstrap PCG64 seed 20261012, 10000 draws, index `fefcfebb...`, numpy linear 0.025/0.975; source
SHAs of `INDEPENDENT_REVIEW_INPUTS.json` (`ecb73a89...`) and `ROUND66_CORRECTED_STATISTICS.json`
(`6eea648e...`); focal cells (1700027, all six on_time=6 incl. target-5); closure 678/678, 0
violations. Per-cell long paths/hashes removed.

## Accounting, eligibility and HEADs

- This review's consumption: 0 new constructions / 0 updates / 0 RNG / 0 checkpoint writes
  (6 read-only checkpoint loads + 2 JSON reads).
- Original R66 accounting re-affirmed unchanged: 1110 valid tasks = 48 references + 384 training +
  678 eval; consumed slots 1111; +1 D1 unregistered extra = 1112 actual constructions; hard cap
  1150, headroom 38. The only number flagged as unreconciled is the synthetic "4" (see A4).
- Three-seed eligibility: VERIFIED (unchanged). The param diff additionally confirms the frozen
  early path is bit-identical to R60 ep0 for every seed and that the trainable modules moved.
- Audited code HEAD `39ac0a4...`; delivery HEAD `254ea79...`; review/errata commit HEAD set below.

## Science retained vs deferred

- **Retained:** restricted (frozen-early-path) family preserves the initial-value on-time/cost
  outcome on this block and preserves focal 1700027/target-5 for all seeds; it never loses a target
  vs OLD and gains 1-2; no cross-seed cost reduction (uniformly more expensive than OLD).
- **Deferred / not claimed:** any general behavioural equivalence of NEW and BC (E2); any claim that
  "4 synthetic updates" is a precise optimizer.step or PPO-call count (A4); generalisation,
  non-inferiority, deployment, or that parameter-sharing interference is the sole root cause.
