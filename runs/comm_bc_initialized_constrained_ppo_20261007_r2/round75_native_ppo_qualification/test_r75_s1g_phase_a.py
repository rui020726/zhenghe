"""Narrow pure-stdlib tests for ROUND75 S1G Phase A accounting and receipt wiring."""
from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

import r75_guard  # noqa: E402  pure stdlib; no torch import
import r75_s1_runner  # noqa: E402  import is free; paid imports stay inside run paths
import r75_s1g_outer_exec  # noqa: E402  pure stdlib outer module


FROZEN = HERE / "r75_frozen.py"
RUNNER = HERE / "r75_s1_runner.py"
OUTER = HERE / "r75_s1g_outer_exec.py"
PLAN = HERE / "ROUND75_S1G_RUN_PLAN.json"


class Round75S1GPhaseATests(unittest.TestCase):
    @staticmethod
    def source(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def test_frozen_technical_prefix_fee_route(self):
        tree = ast.parse(self.source(FROZEN))
        functions = {
            node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        update = functions["r75_native_ppo_update"]
        self.assertIn("physical_step_category", [
            arg.arg for arg in update.args.kwonlyargs])
        self.assertIn("physical_step_registration_prefix", [
            arg.arg for arg in update.args.kwonlyargs])
        text = self.source(FROZEN)
        self.assertIn('(physical_step_category == "technical") != '
                      '(physical_step_registration_prefix is not None)', text)
        self.assertIn("charge_physical_step(\n                what=physical_step_what, "
                      "category=physical_step_category)", text)
        return_nodes = [node for node in ast.walk(update) if isinstance(node, ast.Return)]
        return_keys = []
        for node in return_nodes:
            if isinstance(node.value, ast.Dict):
                return_keys.extend(
                    key.value for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str))
        self.assertIn("optimizer_steps", return_keys)
        self.assertNotIn("physical_step_category", return_keys)
        self.assertNotIn("physical_step_registration_prefix", return_keys)

    def test_runner_s1g_a3_technical_wiring(self):
        text = self.source(RUNNER)
        tree = ast.parse(text)
        functions = {
            node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("run_s1", functions)
        self.assertIn('S1G_REQUEST_ID = "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015"', text)
        self.assertIn('S1G_TECHNICAL_PREFIX = "r75_s1g_A3_technical_retry_20261015"', text)
        self.assertIn("resources.pre_register_technical_attempt(S1G_TECHNICAL_PREFIX)", text)
        self.assertIn("a3_step_category = \"technical\"", text)
        self.assertIn("physical_step_category=a3_step_category", text)
        self.assertIn("physical_step_registration_prefix=a3_step_prefix", text)
        self.assertIn('record["A3_end_stage_snapshot"] = _save_stage_snapshot(', text)
        self.assertIn('("A3_end_stage_snapshot", "A3_completed")', text)
        valid = {
            "license_state": "issued",
            "argv": ["/python", str(RUNNER), "--run-s1g",
                     "--request-id", "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015",
                     "--attempt-dir", str(HERE / "raw" / "s1g_native_retry" /
                                          "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015")],
        }
        self.assertTrue(r75_s1_runner._license_is_fresh(
            valid, "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015",
            HERE / "raw" / "s1g_native_retry" /
            "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015"))
        self.assertFalse(r75_s1_runner._license_is_fresh(
            {"license_state": "issued",
             "argv": ["/python", str(RUNNER), "--run-s1",
                      "--request-id", "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015",
                      "--attempt-dir", str(HERE / "raw" / "s1g_native_retry" /
                                           "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015")]},
            "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015",
            HERE / "raw" / "s1g_native_retry" /
            "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015"))

    def test_outer_reliable_returncode_and_frozen_hashes(self):
        text = self.source(OUTER)
        self.assertIn("child_returncode", text)
        self.assertIn("wrapper_return", text)
        self.assertNotIn("__S1G_RUNNER_SHA__", text)
        self.assertNotIn("__S1G_FROZEN_SHA__", text)
        self.assertNotIn("__S1G_PLAN_SHA__", text)
        self.assertIn("RUNNER_ARGV = [\"--run-s1g\"]", text)
        self.assertIn('OUTPUT_DIR_NAME = "raw/s1g_native_retry"', text)
        outer_fixture_parent = HERE / "_r75_s1g_phase_a_test_fixture"
        outer_fixture_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="outer_record_", dir=outer_fixture_parent) as temp:
            child = {"stdout_bytes": b"out", "stderr_bytes": b"err", "timed_out": False,
                     "started_at": {"unix": 1}, "ended_at": {"unix": 2},
                     "wall_seconds": 1, "child_pid": 123}
            payload = r75_s1g_outer_exec.record_outcome(
                Path(temp), "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015", child,
                "head", "parent", "branch", "wrapper", {"ledger": True}, ["file"])
            self.assertEqual(payload["child_returncode"], "MISSING_OR_INVALID")
            self.assertEqual(payload["wrapper_return"], 5)
            child["child_returncode"] = 0
            payload = r75_s1g_outer_exec.record_outcome(
                Path(temp), "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015", child,
                "head", "parent", "branch", "wrapper", {"ledger": True}, ["file"])
            self.assertEqual(payload["child_returncode"], 0)
            self.assertEqual(payload["wrapper_return"], 0)
        original_popen = r75_s1g_outer_exec.subprocess.Popen
        class _NoExitProcess:
            returncode = None
            pid = 123
            def communicate(self, timeout=None):
                return b"", b""
        def fake_popen(*args, **kwargs):
            return _NoExitProcess()
        r75_s1g_outer_exec.subprocess.Popen = fake_popen
        try:
            with self.assertRaises(RuntimeError):
                r75_s1g_outer_exec.launch_once(HERE, RUNNER, ["/python", str(RUNNER)], 1)
        finally:
            r75_s1g_outer_exec.subprocess.Popen = original_popen

    def test_plan_uses_new_baseline_and_technical_accounting(self):
        plan = json.loads(self.source(PLAN))
        self.assertEqual(plan["request_id"], "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015")
        contract = plan["s1g_execution_contract"]
        baseline = contract["ledger_baseline"]["guards"]
        self.assertEqual(baseline["physical_optimizer_step"]["used_normal"], 5)
        self.assertEqual(baseline["offline_trunk_target_row"]["used_normal"], 840)
        self.assertEqual(baseline["offline_trunk_target_row"]["charged_events"], 70)
        budget = contract["attempt_budget"]
        self.assertEqual(budget["A1"]["category"], "normal")
        self.assertEqual(budget["A2"]["category"], "normal")
        self.assertEqual(budget["A3"]["category"], "technical")
        totals = budget["totals"]
        self.assertEqual(totals["physical_optimizer_steps"], {
            "normal": 5, "technical": 4, "hard": 9})
        self.assertEqual(totals["offline_trunk_target_row"]["normal"], 1680)
        self.assertEqual(totals["offline_trunk_target_row"]["reserve"], 0)
        self.assertEqual(
            budget["cumulative_caps_after_attempt"]["offline_trunk_target_row"],
            {"normal": 2520, "reserve": 0, "total": 2520,
             "contract_caps": "normal 5000 + technical 1000 / hard 6000"})

    def test_guard_technical_charge_requires_pre_registration_fixture_only(self):
        fixture_parent = HERE / "_r75_s1g_phase_a_test_fixture"
        fixture_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r75_s1g_phase_a_", dir=fixture_parent) as temp:
            resources = r75_guard.R75Resources.fixture(Path(temp))
            prefix = "r75_s1g_A3_technical_retry_20261015"
            resources.pre_register_technical_attempt(prefix)
            self.assertEqual(resources.charge_physical_step(
                prefix + ":probe", category="technical"), 1)
            with self.assertRaises(r75_guard.R75LedgerError):
                resources.charge_physical_step("unregistered:probe", category="technical")
            class _ShapeList(list):
                shape = (840, 19)
            self.assertEqual(resources.charge_trunk_forward(
                _ShapeList([0.0] * 19 for _ in range(840)),
                prefix + ":normal_trunk", category="normal"), 840)
            optimizer = type("Optimizer", (), {"step": lambda self: None})()
            observer = r75_s1_runner.ObservedStepEntry(
                optimizer, resources, "fixture_old5_baseline")
            for index in range(5):
                resources.charge_physical_step(f"fixture:old5_baseline_step_{index}")
                observer.step()
            completed = [item for item in observer.records if item["phase"] == "completed"]
            self.assertEqual(len(completed), 5)
            self.assertTrue(all(item["exactly_one_new_permit"] for item in completed))
            with self.assertRaises(RuntimeError):
                observer.step()
            self.assertEqual(observer.records[-1]["phase"], "refused")
            self.assertFalse(observer.records[-1]["delegated"])
            snapshot = resources.snapshots()
            self.assertEqual(snapshot["physical_optimizer_step"]["used_normal"], 0)
            self.assertEqual(snapshot["physical_optimizer_step"]["used_reserve"], 6)
            self.assertEqual(snapshot["offline_trunk_target_row"]["used_normal"], 840)

    def test_fixture_cannot_use_canonical_production_root(self):
        with self.assertRaises(r75_guard.R75LedgerError):
            r75_guard.R75Resources.fixture(r75_guard.R75_CANONICAL_ROOT)


if __name__ == "__main__":
    unittest.main()
