"""Synthetic single-instance supervisor transaction; no real child or provider."""
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
import run_round58_v36_eval_once as supervisor


class SupervisorTests(unittest.TestCase):
    def test_prechild_state_and_final_lock_release_with_spied_child(self):
        with tempfile.TemporaryDirectory(prefix="r58_v36_supervisor_test_", dir=HERE) as folder:
            root = Path(folder).resolve()
            self.assertEqual(root.parent, HERE)
            before = {"complete_logical_jobs": 384, "remaining_plan_keys": 640,
                      "inflight_claims": 0}
            after = {"complete_logical_jobs": 1024, "remaining_plan_keys": 0,
                     "inflight_claims": 0}
            class FakeChild:
                pid = 12345
                def wait(self):
                    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
                    self_test.assertEqual(state["status"], "running")
                    self_test.assertEqual(state["child_pid"], self.pid)
                    return 0
            self_test = self
            def fake_git(args, **kwargs):
                return SimpleNamespace(stdout="" if args[1] == "status" else "frozen-head\n")
            with (mock.patch.object(supervisor, "LOCK", root / "lock.json"),
                  mock.patch.object(supervisor, "STATE", root / "state.json"),
                  mock.patch.object(supervisor, "STDOUT", root / "stdout.log"),
                  mock.patch.object(supervisor, "STDERR", root / "stderr.log"),
                  mock.patch.object(supervisor.ledger, "audit", side_effect=[before, after]),
                  mock.patch.object(supervisor.subprocess, "run", side_effect=fake_git),
                  mock.patch.object(supervisor.subprocess, "Popen", return_value=FakeChild())):
                self.assertEqual(supervisor.run(), 0)
                self.assertFalse((root / "lock.json").exists())
                final = json.loads((root / "state.json").read_text(encoding="utf-8"))
                self.assertEqual(final["status"], "fixed_plan_cli_complete")


if __name__ == "__main__":
    unittest.main(verbosity=2)
