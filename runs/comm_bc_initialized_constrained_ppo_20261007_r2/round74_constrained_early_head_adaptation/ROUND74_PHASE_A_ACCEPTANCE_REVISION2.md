# ROUND74 PHASE A - Acceptance Boundary Revision 2 (third narrow delegation)

Appended. `ROUND74_PHASE_A_ACCEPTANCE.json` and `..._REVISION.json` are preserved byte-identical and are NOT overwritten. Parent HEAD `6a3fbacde75468b6f0d030cc4c52f6573c777f48` (decision-end verified commit 1). Branch `exp/round74-constrained-early-head-adaptation`.

Only `r74_guard_prod.py` is modified. The original 17 and the full 27 suites are NOT re-run; the 2 used canonical steps are NOT refunded; no push/merge/deploy.

## Three fixes (independent pure-fixture counter-examples)

1. **Reserve-category cap enforced independently.** `charge` now applies the per-category cap to BOTH normal and reserve (`base[cat_key] + n > cat_cap`); `_read_persisted` refuses a stale/foreign state whose single category exceeds its own cap even when the hard total is still under; `make_charged_step` routes to reserve only via an explicit `category="reserve"` bound to a registered attempt - a normal gate refuses at the normal cap and never auto-upgrades. Counter-example: the 5th synthetic RESERVE charge is refused while total < hard 16.
2. **Cross-check fails closed on ANY mismatch.** `_cross_check_ledger` now uses `used != ledger_max` (was `used < ledger_max`): counter-ahead (persist interruption), non-zero counter with a missing ledger, and the existing low-backup all raise `LedgerStateError`. Blocks further work, preserves consumed permission, no self-refund/recovery/rewrite; no power-loss-atomicity claim.
3. **Fixture refuses canonical.** `R74ResourcesProd.__init__` fixture branch raises before any mkdir/guard construction when the target equals the canonical durable root or is inside it. Real PHASE B must use `production()`, never an injected fixture.

## New test

`test_r74_guard_prod_revision.py` - 14/14 ALL_PASS, EXIT 0, sentinel `R74_REVISION_TESTS ALL_PASS`. Throwaway SIBLING roots, self-cleaned; never touches the canonical ledger/marker/2 real toy steps.

## Read-only confirmation

`production()` binds canonical (pinned True), 5 ledger events, synth 2/16 (n2/r0), env/offline/ppo/joint all 0. Canonical SHAs byte-unchanged before/after (empty diff). The tightened `used != ledger` cross-check still accepts the canonical pair (2 == 2).

## Consumption & boundary

This revision adds 0 real steps / 0 env / 0 offline / 0 PPO (one read-only production construction only). Cumulative PHASE A stays at 2 real toy steps; PHASE B remains NOT AUTHORIZED.