"""Narrow pure-stdlib tests for ROUND75 S1G Phase A accounting and receipt wiring."""
from __future__ import annotations

import ast
import contextlib
import copy
import io
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
import r75_s1_precheck  # noqa: E402  pure stdlib; no torch import
import r75_s1g_outer_exec  # noqa: E402  pure stdlib outer module


FROZEN = HERE / "r75_frozen.py"
RUNNER = HERE / "r75_s1_runner.py"
OUTER = HERE / "r75_s1g_outer_exec.py"
PLAN = HERE / "ROUND75_S1G_AR1_RUN_PLAN.json"


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
        runner_text = self.source(RUNNER)
        self.assertLess(runner_text.index("record[\"A2_stage_snapshot\"] = _save_stage_snapshot("),
                        runner_text.index("end_diffs = compare_full_states("))
        plan = json.loads(self.source(PLAN))
        compared, diffs = r75_s1_runner.compare_scientific_returns(
            {"gate_records": 1, "gate_diag": 1},
            {"gate_records": 1, "gate_diag": 2}, plan)
        self.assertEqual([item["field"] for item in diffs], ["gate_diag"])
        self.assertIn("gate_records", compared)

    def test_runner_s1g_a3_technical_wiring(self):
        text = self.source(RUNNER)
        tree = ast.parse(text)
        functions = {
            node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("run_s1", functions)
        self.assertIn('S1G_REQUEST_ID = "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015"', text)
        self.assertIn('S1G_AR1_PLAN_PATH = HERE / "ROUND75_S1G_AR1_RUN_PLAN.json"', text)
        self.assertIn("PLAN_PATH = S1G_AR1_PLAN_PATH", text)
        self.assertIn("precheck_code = r75_s1_precheck.main(PLAN_PATH)", text)
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
        self.assertIn('ROUND75_S1G_AR1_RUN_PLAN.json', text)
        self.assertIn('"r75_s1_runner.py": "4f84184a25d9058df97abf04fb00febe1d6b8c123944fcc9814e78651d3835d4"', text)
        self.assertIn('"r75_s1_precheck.py": "ad14206200b8786fea5f09a0cc5ffe838a57255cb9d844313c7b3fc45da38703"', text)
        self.assertIn('"r75_frozen.py": "bb4bae98518cc482f4b733794ee1c3f93097c56c5b3e85d6fed2c65f2a72198e"', text)
        self.assertIn('"ROUND75_S1G_AR1_RUN_PLAN.json": "6303c093d0e19d71081e5c1c5d6393f21733d8f7c9c9fb87ec6fd21eecd7ebab"', text)
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
        for returncode, timed_out, expected in (
                (0, False, 0), (1, False, 1), (None, False, None),
                (True, False, True), ("0", False, "0"), (0, True, 0)):
            class _FakeProcess:
                pass
            process = _FakeProcess()
            process.returncode = returncode
            process.pid = 123
            def communicate(_self, timeout=None, _timed_out=timed_out):
                if _timed_out:
                    raise r75_s1g_outer_exec.subprocess.TimeoutExpired([], 1)
                return b"", b""
            process.communicate = communicate
            r75_s1g_outer_exec.subprocess.Popen = lambda *args, **kwargs: process
            try:
                if returncode is None or returncode is True or returncode == "0":
                    with self.assertRaises(RuntimeError):
                        r75_s1g_outer_exec.launch_once(HERE, RUNNER, ["/python", str(RUNNER)], 1)
                else:
                    payload = r75_s1g_outer_exec.launch_once(
                        HERE, RUNNER, ["/python", str(RUNNER)], 1)
                    self.assertEqual(payload["child_returncode"], expected)
                    self.assertEqual(payload["timed_out"], timed_out)
            finally:
                r75_s1g_outer_exec.subprocess.Popen = original_popen

    def test_ar1_plan_and_precheck_accept_real_baseline(self):
        plan = json.loads(self.source(PLAN))
        self.assertEqual(plan["request_id"], "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015")
        self.assertEqual(plan["revision_id"], "R75_S1G_AR1_PRECHECK_BASELINE_REPAIR_20261015")
        self.assertEqual(plan["active_production_baseline"]["total_ledger_events"], 88)
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

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(r75_s1_precheck.main(PLAN), 0)
        verdict = json.loads(buffer.getvalue())
        self.assertEqual(verdict["revision_id"], "R75_S1G_AR1_PRECHECK_BASELINE_REPAIR_20261015")
        self.assertEqual(verdict["plan_path"], str(PLAN))
        self.assertTrue(verdict["passed"])

        report = r75_s1_precheck.Report()
        r75_s1_precheck.check_s1g_headroom(plan, report)
        self.assertTrue(report.passed)

        excess = copy.deepcopy(plan)
        excess["s1g_execution_contract"]["attempt_budget"]["totals"][
            "physical_optimizer_steps"]["normal"] = 9
        excess["s1g_execution_contract"]["attempt_budget"]["cumulative_caps_after_attempt"][
            "physical_optimizer_step"] = {"normal": 14, "reserve": 0, "total": 14}
        excess_report = r75_s1_precheck.Report()
        r75_s1_precheck.check_s1g_headroom(excess, excess_report)
        self.assertFalse(excess_report.passed)

        for field, value in (("total_ledger_sha256", "0" * 64),
                             ("total_ledger_events", 89)):
            tampered = copy.deepcopy(plan)
            tampered["active_production_baseline"][field] = value
            tampered_report = r75_s1_precheck.Report()
            r75_s1_precheck.check_active_production_baseline(tampered, tampered_report)
            self.assertFalse(tampered_report.passed)
        for field, value in (("used_normal", 6), ("refusals", 1)):
            tampered = copy.deepcopy(plan)
            tampered["active_production_baseline"]["guards"][
                "physical_optimizer_step"][field] = value
            tampered_report = r75_s1_precheck.Report()
            r75_s1_precheck.check_active_production_baseline(tampered, tampered_report)
            self.assertFalse(tampered_report.passed)

    def test_guard_technical_charge_requires_pre_registration_fixture_only(self):
        fixture_parent = HERE / "_r75_s1g_phase_a_test_fixture"
        fixture_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r75_s1g_phase_a_", dir=fixture_parent) as temp:
            resources = r75_guard.R75Resources.fixture(Path(temp))
            prefix = "r75_s1g_A3_technical_retry_20261015"
            for index in range(5):
                resources.charge_physical_step(f"fixture:old5:{index}", category="normal")
            class _ShapeList(list):
                shape = (840, 19)
            self.assertEqual(resources.charge_trunk_forward(
                _ShapeList([0.0] * 19 for _ in range(840)),
                prefix + ":normal_trunk", category="normal"), 840)
            optimizer = type("Optimizer", (), {"step": lambda self: None})()
            observer = r75_s1_runner.ObservedStepEntry(optimizer, resources, "old5")
            with self.assertRaises(RuntimeError):
                observer.step()
            self.assertEqual(optimizer.calls, 0)
            with self.assertRaises(r75_guard.R75LedgerError):
                resources.charge_physical_step(prefix + ":unregistered", category="technical")
            self.assertEqual(optimizer.calls, 0)
            resources.pre_register_technical_attempt(prefix)
            with self.assertRaises(r75_guard.R75LedgerError):
                resources.charge_physical_step("wrong-prefix:probe", category="technical")
            resources.charge_physical_step("fixture:new-normal", category="normal")
            observer.step()
            resources.charge_physical_step(prefix + ":new-technical", category="technical")
            observer.step()
            self.assertEqual(optimizer.calls, 2)
            resources.charge_physical_step("fixture:duplicate-a", category="normal")
            resources.charge_physical_step("fixture:duplicate-b", category="normal")
            with self.assertRaises(RuntimeError):
                observer.step()
            self.assertEqual(optimizer.calls, 2)
            self.assertEqual(observer.records[-1]["new_permits_since_previous_step"], 2)
            self.assertFalse(observer.records[-1]["delegated"])
            self.assertEqual(resources.charge_trunk_forward(
                _ShapeList([0.0] * 19 for _ in range(840)),
                "fixture:normal-trunk", category="normal"), 840)
            snapshot = resources.snapshots()
            self.assertEqual(snapshot["physical_optimizer_step"]["used_normal"], 8)
            self.assertEqual(snapshot["physical_optimizer_step"]["used_reserve"], 1)
            self.assertEqual(snapshot["offline_trunk_target_row"]["used_normal"], 840)

    def test_fixture_cannot_use_canonical_production_root(self):
        with self.assertRaises(r75_guard.R75LedgerError):
            r75_guard.R75Resources.fixture(r75_guard.R75_CANONICAL_ROOT)


if __name__ == "__main__":
    unittest.main()
