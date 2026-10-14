# -*- coding: utf-8 -*-
"""ROUND73 shared gate ledgers (offline trunk rows + synthetic optimizer steps).

ROUND73 runs NO offline fit: the INIT comes straight from the ROUND71 calibrated checkpoint,
so the offline trunk-input ledger stays at zero rows and the 6,000-row cap is a pure guard.
The synthetic-optimizer-step guard caps gate-only steps at 8 and is charged before each step;
ROUND73's pre-launch gates perform none, so it also stays at zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN_ROOT = REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
R68_DIR = RUN_ROOT / "round68_late_initial_confidence_ablation"
READINESS_DIR = REPO / "analysis" / "round71-critic-calibration-readiness"
for _d in (READINESS_DIR, str(R68_DIR), str(HERE)):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import r68_guards                                                         # noqa: E402
from r71_critic_calibration_seam import ForwardLedger                    # noqa: E402

OFFLINE_ROW_CAP = 6000
SYNTHETIC_STEP_CAP = 8


def gate_paths(raw_root: Path) -> dict:
    root = Path(raw_root)
    return {"forward_ledger": root / "runtime" / "r73_forward_ledger.json",
            "synthetic_steps": root / "runtime" / "synthetic_gates"
            / "r73_synthetic_optimizer_steps.json",
            "receipts": root / "runtime" / "synthetic_gates",
            "summary": root / "ROUND73_PRELAUNCH_GATES.json"}


def forward_ledger(raw_root: Path) -> ForwardLedger:
    """The ONE persisted offline trunk-input ledger of this round (cap 6,000 rows)."""
    return ForwardLedger(cap=OFFLINE_ROW_CAP,
                         state_path=gate_paths(raw_root)["forward_ledger"])


def synthetic_guard(raw_root: Path):
    """The ONE persisted synthetic-optimizer-step counter (cap 8, charged before the step)."""
    path = gate_paths(raw_root)["synthetic_steps"]
    path.parent.mkdir(parents=True, exist_ok=True)
    return r68_guards.SyntheticStepGuard(path, cap=SYNTHETIC_STEP_CAP)