# -*- coding: utf-8 -*-
"""Narrow pure-stdlib counterexamples for the S1F Git identity repair."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "r75_s1_runner.py"


class IdentityRepairTests(unittest.TestCase):
    def load_runner(self):
        spec = importlib.util.spec_from_file_location("r75_s1_runner_s1f_test", RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def fake_git(head="test-head", parent="test-parent", branch="real-branch", fail=False):
        calls = []
        class Result:
            def __init__(self, returncode, stdout=b"", stderr=b""):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr
        def run(args, capture_output):
            calls.append(tuple(args))
            if fail:
                return Result(128, b"", b"injected git failure")
            tail = tuple(args[3:])
            outputs = {
                ("rev-parse", "HEAD"): (0, (head + "\n").encode(), b""),
                ("rev-parse", "HEAD^"): (0, (parent + "\n").encode(), b""),
                ("rev-parse", "--abbrev-ref", "HEAD"): (0, (branch + "\n").encode(), b""),
            }
            if tail == ("--abbrev-ref", "HEAD"):
                return Result(129, b"", b"unknown option: --abbrev-ref")
            if tail not in outputs:
                return Result(2, b"", b"unexpected git command")
            code, stdout, stderr = outputs[tail]
            return Result(code, stdout, stderr)
        return calls, run

    def test_execution_identity_uses_valid_git_protocol_and_real_branch(self):
        runner = self.load_runner()
        calls, run = self.fake_git(branch="exp/round75-native-ppo-qualification")
        with mock.patch.object(runner.subprocess, "run", run):
            identity = runner._execution_identity()
        self.assertEqual(identity["head"], "test-head")
        self.assertEqual(identity["parent_head"], "test-parent")
        self.assertEqual(identity["branch"], "exp/round75-native-ppo-qualification")
        self.assertEqual(identity["errors"], {"head": None, "parent_head": None, "branch": None})
        self.assertIn(("git", "-C", str(runner.HERE), "rev-parse", "--abbrev-ref", "HEAD"), calls)
        self.assertNotIn(("git", "-C", str(runner.HERE), "--abbrev-ref", "HEAD"), calls)

    def test_execution_identity_git_failure_returns_unknown_and_errors(self):
        runner = self.load_runner()
        calls, run = self.fake_git(fail=True)
        with mock.patch.object(runner.subprocess, "run", run):
            identity = runner._execution_identity()
        self.assertEqual(identity["head"], "unknown")
        self.assertEqual(identity["parent_head"], "unknown")
        self.assertEqual(identity["branch"], "unknown")
        self.assertEqual(identity["errors"]["branch"], "injected git failure")
        self.assertEqual(len(calls), 3)

    def test_default_stage_helper_uses_real_identity_wiring(self):
        runner = self.load_runner()
        calls, run = self.fake_git(head="wire-head", parent="wire-parent",
                                   branch="exp/round75-native-ppo-qualification")
        class State:
            unread = []
            record = {}
            def counter_report(self):
                return {"counted": 1}
        class FakeTorch:
            save_count = 0
            payload = None
            def save(self, payload, file_obj):
                self.save_count += 1
                self.payload = payload
                Path(file_obj).write_bytes(b"stage")
        fake_torch = FakeTorch()
        runner.EXECUTION_REQUEST_ID = "R75_S1F_GIT_IDENTITY_REPAIR_DRAFT_20261015"
        self.addCleanup(setattr, runner, "EXECUTION_REQUEST_ID", None)
        with mock.patch.object(runner.subprocess, "run", run), \
             mock.patch.dict(runner.sys.modules, {"torch": fake_torch}):
            with tempfile.TemporaryDirectory(prefix="s1f_identity_wiring_", dir=HERE) as directory:
                path = Path(directory) / "identity-wiring.pt"
                identity = runner._save_stage_snapshot(
                    path, stage="S1F_identity_wiring", label="pure_test",
                    state=State(), full_state={"data_identity": {"fixture": "non-neural only"}})
                self.assertTrue(path.is_file())
                self.assertEqual(fake_torch.save_count, 1)
        self.assertEqual(identity["execution_head"], "wire-head")
        self.assertEqual(fake_torch.payload["execution_identity"]["head"], "wire-head")
        self.assertEqual(fake_torch.payload["execution_identity"]["branch"],
                         "exp/round75-native-ppo-qualification")


if __name__ == "__main__":
    unittest.main()
