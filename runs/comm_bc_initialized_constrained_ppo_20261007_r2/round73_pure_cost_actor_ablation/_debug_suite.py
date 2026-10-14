# -*- coding: utf-8 -*-
"""DEBUG-ONLY harness: run the qualification suite against SCRATCH guard ledgers.

The committed suite charges the ONE persisted ROUND73 ledgers.  While iterating on the gate
code, that would burn the round's 8-charge synthetic budget on failed attempts, so this
harness points the two guards at fresh scratch files under the synthetic-test-fixture
directory (never the real raw ledgers).  The guard class, the cap and the charge-before-
execute semantics are unchanged; only the ledger file differs.  NOT part of the delivery.
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN_ROOT = REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
for _d in (REPO / "analysis" / "round71-critic-calibration-readiness",
           RUN_ROOT / "round68_late_initial_confidence_ablation", HERE):
    sys.path.insert(0, str(_d))

import r68_guards
import r73_trial_gates as gates

SCRATCH_DIR = HERE / "raw" / "synthetic_test_fixtures" / "debug_scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
STEPS = SCRATCH_DIR / "synthetic_optimizer_steps.json"
ROWS = SCRATCH_DIR / "forward_rows.json"
for _f in (STEPS, ROWS):
    if _f.is_file():
        _f.unlink()
gates.synthetic_guard = lambda raw_root: r68_guards.SyntheticStepGuard(STEPS, cap=8)


def _scratch_forward(raw_root):
    from r71_critic_calibration_seam import ForwardLedger
    return ForwardLedger(cap=6000, state_path=ROWS)


gates.forward_ledger = _scratch_forward

import test_r73_qualification_gates as mod  # noqa: E402

# The suite's DERIVED outputs must also land in scratch.  Left on the real paths, a scratch
# run overwrites the real synthetic receipt and appends scratch lines to the real quarantine
# archive.  PPO_JOURNAL / STEP_JOURNAL stay REAL on purpose: proving those journals are
# restored byte-for-byte is the whole point of the suite.
mod.SYNTH_JOURNAL_RECEIPT = SCRATCH_DIR / "R73_SYNTHETIC_PPO_JOURNAL_RECEIPT.json"
mod.DIVERT_DIR = SCRATCH_DIR / "diverted_real_journal_lines"

loader = unittest.TestLoader()
suite = loader.loadTestsFromModule(mod)
result = unittest.TextTestRunner(verbosity=2).run(suite)
print("SCRATCH steps used:", r68_guards.SyntheticStepGuard(STEPS, cap=8).used())
raise SystemExit(0 if result.wasSuccessful() else 1)