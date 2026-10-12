# ROUND66 Final Review — Erratum 2 (closing): corrected synthetic-update accounting

**Nature:** zero-environment, read-only only (JSONL line counts + source reads). No PPO rerun, no
optimizer call, no environment, no checkpoint/RNG write. Append-only; `ROUND66_FINAL_REVIEW_ERRATA.md`
and `raw/runtime/ROUND66_REVIEW_MACHINE_VERDICT.json` (commit `858c489`) are preserved byte-identical.
This record supersedes only the `synthetic_updates_audit` block of that machine verdict. Machine
record: `raw/runtime/ROUND66_REVIEW_SYNTHETIC_RECON.json`.

## What was wrong in Erratum 1 (A4)

Two errors, now corrected:

1. "Only 1 synthetic PPO update ran." A single call site in the source does not bound historical
   executions. There were **2** evidenced runs (see below), hence **2** `frozen_early_ppo_update`
   calls, not 1.
2. "Each run >= 5 optimizer.step, so 5-18 across up to two runs, therefore <= 32, no breach." A
   per-run lower bound and an assumed run ceiling do not establish a total <= 32. **The <=32
   no-breach guarantee is WITHDRAWN.**

## Read-only reconstruction (marked RECONSTRUCTED)

- **No early-break/skip:** neither `r66_frozen.frozen_early_ppo_update` nor the frozen
  `v2_trainer.ppo_update` breaks out of the epoch/minibatch loop; there is no KL/target early-stop.
  The only `continue` (r66_frozen.py line 747) skips gate-prior work for non-free targets and never
  skips a minibatch or an `optimizer.step`. So `optimizer_steps = minibatches =
  EPOCHS_PER_BATCH x ceil(N / MINIBATCH_STEPS)`.
- **Frozen constants:** `EPOCHS_PER_BATCH = 4`, `MINIBATCH_STEPS = 256`.
- **N (samples):** `batch_samples` emits one sample per tape row. The test reloads R60 seed-11
  episodes 0-3; their a1 tapes have 66 + 71 + 52 + 76 = **265** decision-step rows, so **N = 265**
  (cross-checked against `raw/checkpoints/r40_batch_s11_b0_attempt01.json`: samples 265, minibatches
  8, epochs 4, same seed-11 first batch).
- **minibatches = 4 x ceil(265/256) = 4 x 2 = 8** `optimizer.step` per `frozen_early_ppo_update`.
- **Per run:** 1 bare `opt.step()` (section 6) + 8 internal (section 10) = **9 raw optimizer.step**;
  the test's `synthetic_updates` counter = **2 events** (1 bare + 1 ppo_update call). RECONSTRUCTED.

## The two evidenced runs

`_accept_stdout.txt` (written 14:12:03) and `_accept_stdout_verify.txt` (written 14:49:05) are
**byte-identical** (sha256 `7A084853...`) but 37 minutes apart: two distinct deterministic
executions. Both record `synthetic_optimizer_updates = 2`, `synthetic_update_cap = 32`. Neither file
is git-tracked.

## Reconciling the ledger "4" (and its limit)

Under exactly these 2 evidenced runs: 2 x 2 events = **4**; 2 x 9 = **18** raw optimizer.step. The
recorded "4" therefore reconciles as **2 runs x 2 synthetic-update EVENTS** (the test counter counts
events, not raw optimizer.step and not PPO calls). It is **not** "4 optimizer.step" (that is 18) and
**not** "4 PPO calls" (that is 2).

## Withdrawn guarantee and what stays unknown

The number of historical test executions is **not recorded**: only 2 stdout snapshots survive and
there is no test-run ledger (synthetic updates consume no resource slot, so nothing else bounds
them). Therefore the cumulative synthetic consumption is **UNKNOWN**; 4 events / 18 raw optimizer.step
are **lower bounds** consistent with the 2 evidenced runs, not proven totals. The "<= 32, no breach"
statement from Erratum 1 is **withdrawn**. Missing evidence (recorded as unknown, not filled by any
rerun): the true number of test executions and hence the true cumulative event/step count.

## Unchanged from Erratum 1

E1 (D1 wording), E2 (NEW==BC wording), A3 (read-only ep0->ep128 param diff: trunk+early_head
bit-identical, late+critics moved), and A5 (numeric projection) all stand. Only the synthetic-update
accounting is corrected here.
