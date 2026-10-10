"""Synthetic safe-cell/terminal transaction across the actual V41 supervisor."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_round58_v41_controlled as supervisor


class ControlledSupervisorTests(unittest.TestCase):
    def test_graceful_cell_boundary_then_complete_without_duplicate_launch(self):
        with tempfile.TemporaryDirectory(prefix="r58_v41_controlled_test_", dir=HERE) as folder:
            root = Path(folder).resolve()
            self.assertEqual(root.parent, HERE)
            def audit(complete, attempts, constructors, remaining):
                return {"complete_logical_jobs": complete, "remaining_plan_keys": remaining,
                        "inflight_claims": 0, "failed_technical_claims": 5,
                        "cumulative_attempt_slots": attempts,
                        "cumulative_actual_constructor_calls": constructors}
            first = audit(960, 981, 979, 64)
            middle = audit(980, 1001, 999, 44)
            final = audit(1024, 1045, 1043, 0)
            class FakeChild:
                counter = 0
                def __init__(self, *_args, **kwargs):
                    FakeChild.counter += 1
                    self.pid = 1000 + FakeChild.counter
                    self.err = kwargs["stderr"]
                def wait(self):
                    if self.pid == 1001:
                        self.err.write("Traceback (most recent call last):\n"
                                       + supervisor.STOP + "\n")
                        self.err.flush()
                        return 1
                    return 0
            def fake_git(args, **_kwargs):
                return SimpleNamespace(stdout="" if args[1] == "status" else "frozen-head\n")
            with (mock.patch.object(supervisor, "LOCK", root / "lock.json"),
                  mock.patch.object(supervisor, "STATE", root / "state.json"),
                  mock.patch.object(supervisor, "HERE", root),
                  mock.patch.object(supervisor.ledger, "audit",
                                    side_effect=[first, first, middle, middle, final]),
                  mock.patch.object(supervisor.subprocess, "run", side_effect=fake_git),
                  mock.patch.object(supervisor.subprocess, "Popen", side_effect=FakeChild)):
                # Input receipts remain the real frozen files while output is
                # isolated in this test directory.
                for name in ("ROUND58_RESUME_FREEZE_V41.json",
                             "ROUND58_V41C_ACTUAL_EXISTING_CELLS_GATE.json"):
                    (root / name).write_bytes((HERE / name).read_bytes())
                self.assertEqual(supervisor.run(), 0)
                self.assertFalse((root / "lock.json").exists())
                state = json.loads((root / "state.json").read_text(encoding="utf-8"))
                self.assertEqual(state["status"], "fixed_plan_cli_complete")
                self.assertEqual([row["exit_code"] for row in state["cycles"]], [1, 0])
                self.assertEqual(state["cycles"][0]["after"]["complete_logical_jobs"], 980)


if __name__ == "__main__":
    unittest.main(verbosity=2)
