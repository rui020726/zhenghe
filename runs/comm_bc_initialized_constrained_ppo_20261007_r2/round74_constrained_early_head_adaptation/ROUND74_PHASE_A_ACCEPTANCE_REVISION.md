# ROUND74 PHASE A - Acceptance Boundary Revision (appended)

- Branch: `exp/round74-constrained-early-head-adaptation`
- HEAD: `accc47bab5b5c9aaa15b10e06919f29b0240e03e`
- This is an APPENDED revision. The original `ROUND74_PHASE_A_ACCEPTANCE.json` is preserved byte-identical.

## Boundary corrections
- PROOF_OK write omission: the original acceptance carries no top-level `toy_adam_proof_ok` (reads None).
  The actual proof_ok was True (driver printed `R74_PHASE_A OK`, marker written only after the invariants
  held, and the toy_adam_proof object shows third_attempt_refused / param_unchanged / global==2).
- Adam per-value state: UNVERIFIED. The original compared only the `exp_avg` ROW COUNT (1==1) around the
  refused 3rd attempt, not per-element values; a full per-value snapshot was not retained and is NOT
  re-run (no budget). Verified: the 3rd attempt was refused BEFORE `opt.step()` and the scalar param was
  unchanged (-0.19999998807907104); because the step was never entered, no Adam mutation happened on it.

## Hardening delivered (r74_guard_prod.py + test_r74_guard_prod.py)
- `count_target_vectors`: (19,) now counts 1 (was 0); a non-19 last dim RAISES at the trunk entry
  (never defaults to 0).
- `make_charged_step`: unified per-physical-step pre-charge seam built on the REAL r43 `count_at_entry`
  boundary (the wrapper installed around `torch.optim.Adam.step`). Multi-step each charged; refusal
  happens before the original step; original-step failure does not refund.
- `charge_trunk_forward`: trunk-entry metering over ACTUAL 19-D rows; per-batch 1024 ALWAYS enforced at
  the production entry (not bypassable); non-19 raises; only this offline seam moves the offline unit.
- `_ProdCounter`: durable-base-inside-lock + cross-instance/cross-process O_EXCL lock (the PHASE A
  threading.Lock was per-instance/in-process only); fail-closed on corrupt/partial/foreign/used>hard/
  lost-counter/low-backup/residual-lock; no refund; no lock recovery.
- `R74ResourcesProd.production()` pins the canonical root and refuses any non-canonical/redirected root;
  `.fixture(root)` records simulated consumption only and cannot reset production quota.

## Real connection points (for PHASE B to wire)
- Adam.step hook: `r43_call_counter_hooks.count_at_entry` (round43-training-call-counter/2), installed by
  `configure()` in `r46_execute_r2.py`.
- Pristine step: `_PRISTINE_ADAM_STEP` captured before the hook and restored around gate updates
  (round71 `r71_trial_gates.py`; round73 `_patch_*.py`, `test_r73_qualification_gates.py`) - these bypass
  the hook and must be wrapped by the same seam.
- Trunk entry: `controller.inner.trunk(x)`, x = 19-D rows, per the R71 `guarded_trunk_forward` pattern.
- Seam boundary: `make_charged_step` charges the NORMAL category and refuses at the normal cap (12); the
  hard cap (16) is reachable only via reserve. PHASE B sizes its own physical-step budget at decision-end.

## Verification
- New pure-fixture spy suite `test_r74_guard_prod.py`: 27/27 PASS (throwaway roots; canonical untouched).
- Read-only production pinning: `R74ResourcesProd.production()` binds canonical, synth used 2/16, others 0,
  canonical ledger unchanged (5 events before and after), no charge.

## Consumption
- This revision: real physical optimizer steps 0, env 0, forward 0, PPO 0. Canonical ledger/marker and the
  2 real toy steps are byte-unchanged.
- Cumulative PHASE A: 2 real toy Adam steps; 17 (original) + 27 (production-seam) pure spy passes.
- PHASE B NOT started; the spy passes do NOT prove the real production wiring is qualified.

## Protected (unchanged)
- ROUND46_CURRENT_TASK.json cf3ad198... ; ROUND46_R2_V6_STOP_STATE.json 93a2f69f...

Machine-readable: ROUND74_PHASE_A_ACCEPTANCE_REVISION.json.