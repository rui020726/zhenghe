"""Six fixed, pure-stdlib tests for ROUND75 S1G A-R2 preparation."""
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
PRECHECK = HERE / "r75_s1_precheck.py"
PLAN = HERE / "ROUND75_S1G_AR1_RUN_PLAN.json"


class Round75S1GPhaseATests(unittest.TestCase):
    @staticmethod
    def source(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def test_source_parse_and_scientific_return_contract(self):
        tree = ast.parse(self.source(FROZEN))
        for path in (RUNNER, PRECHECK, OUTER, Path(__file__)):
            ast.parse(self.source(path))
        functions = {
            node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        update = functions["r75_native_ppo_update"]
        self.assertIn("physical_step_category", [
            arg.arg for arg in update.args.kwonlyargs])
        self.assertIn("physical_step_registration_prefix", [
            arg.arg for arg in update.args.kwonlyargs])
        frozen_text = self.source(FROZEN)
        self.assertIn('(physical_step_category == "technical") != '
                      '(physical_step_registration_prefix is not None)', frozen_text)
        self.assertIn("charge_physical_step(\n                what=physical_step_what, "
                      "category=physical_step_category)", frozen_text)
        return_keys = []
        for node in ast.walk(update):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                return_keys.extend(key.value for key in node.value.keys
                                   if isinstance(key, ast.Constant) and isinstance(key.value, str))
        self.assertIn("optimizer_steps", return_keys)
        self.assertNotIn("physical_step_category", return_keys)
        self.assertNotIn("physical_step_registration_prefix", return_keys)

        runner_text = self.source(RUNNER)
        self.assertLess(runner_text.index('record["A2_stage_snapshot"] = _save_stage_snapshot('),
                        runner_text.index("end_diffs = compare_full_states("))
        plan = json.loads(self.source(PLAN))
        compared, differences = r75_s1_runner.compare_scientific_returns(
            {"gate_records": 1, "gate_diag": 1},
            {"gate_records": 1, "gate_diag": 2}, plan)
        self.assertIn("gate_records", compared)
        self.assertEqual([item["field"] for item in differences], ["gate_diag"])

    def test_runner_license_and_precheck_path_wiring(self):
        text = self.source(RUNNER)
        tree = ast.parse(text)
        self.assertIn("run_s1", {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)})
        self.assertIn('S1G_REQUEST_ID = "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015"', text)
        self.assertIn('S1G_AR1_PLAN_PATH = HERE / "ROUND75_S1G_AR1_RUN_PLAN.json"', text)
        self.assertIn("PLAN_PATH = S1G_AR1_PLAN_PATH", text)
        self.assertIn("precheck_code = r75_s1_precheck.main(PLAN_PATH)", text)
        self.assertIn('S1G_TECHNICAL_PREFIX = "r75_s1g_A3_technical_retry_20261015"', text)
        self.assertIn("resources.pre_register_technical_attempt(S1G_TECHNICAL_PREFIX)", text)
        self.assertIn('a3_step_category = "technical"', text)
        self.assertIn("physical_step_category=a3_step_category", text)
        self.assertIn("physical_step_registration_prefix=a3_step_prefix", text)

        original_request_id = "R75_S1_NATIVE_QUALIFICATION_20261015"
        original_attempt_dir = HERE / "raw" / "s1"
        self.assertTrue(r75_s1_runner._license_is_fresh({
            "license_state": "issued",
            "argv": ["/python", str(RUNNER), "--run-s1",
                     "--request-id", original_request_id,
                     "--attempt-dir", str(original_attempt_dir)],
        }, original_request_id, original_attempt_dir))

        request_id = "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015"
        attempt_dir = HERE / "raw" / "s1g_native_retry" / request_id
        self.assertTrue(r75_s1_runner._license_is_fresh({
            "license_state": "issued",
            "argv": ["/python", str(RUNNER), "--run-s1g",
                     "--request-id", request_id, "--attempt-dir", str(attempt_dir)],
        }, request_id, attempt_dir))
        self.assertFalse(r75_s1_runner._license_is_fresh({
            "license_state": "issued",
            "argv": ["/python", str(RUNNER), "--run-s1",
                     "--request-id", request_id, "--attempt-dir", str(attempt_dir)],
        }, request_id, attempt_dir))
        self.assertFalse(r75_s1_runner._license_is_fresh({
            "license_state": "issued",
            "argv": ["/python", str(RUNNER), "--run-s1g",
                     "--request-id", "wrong-id", "--attempt-dir", str(attempt_dir)],
        }, request_id, attempt_dir))
        self.assertFalse(r75_s1_runner._license_is_fresh({
            "license_state": "issued",
            "argv": ["/python", str(RUNNER), "--run-s1g",
                     "--request-id", request_id, "--attempt-dir", str(HERE / "elsewhere")],
        }, request_id, attempt_dir))

    def test_outer_process_and_outcome_fail_closed(self):
        text = self.source(OUTER)
        self.assertIn("child_returncode", text)
        self.assertIn("wrapper_return", text)
        self.assertNotIn("__S1G_RUNNER_SHA__", text)
        self.assertNotIn("__S1G_FROZEN_SHA__", text)
        self.assertNotIn("__S1G_PLAN_SHA__", text)
        self.assertIn('RUNNER_ARGV = ["--run-s1g"]', text)
        self.assertIn("ROUND75_S1G_AR1_RUN_PLAN.json", text)
        self.assertIn('"r75_s1_runner.py": '
                      '"4f84184a25d9058df97abf04fb00febe1d6b8c123944fcc9814e78651d3835d4"', text)
        self.assertIn('"r75_s1_precheck.py": '
                    '"88b88c3854f25f81a1398d37066b5d62f4941a7529b349fe5a9d20361733625b"', text)
        self.assertIn('"r75_frozen.py": '
                      '"bb4bae98518cc482f4b733794ee1c3f93097c56c5b3e85d6fed2c65f2a72198e"', text)
        self.assertIn('"ROUND75_S1G_AR1_RUN_PLAN.json": '
                      '"6303c093d0e19d71081e5c1c5d6393f21733d8f7c9c9fb87ec6fd21eecd7ebab"', text)

        class FakeProcess:
            def __init__(self, returncode, timeout_once=False):
                self.returncode = returncode
                self.pid = 777
                self.timeout_once = timeout_once
                self.communicate_calls = 0
                self.kill_calls = 0

            def communicate(self, timeout=None):
                self.communicate_calls += 1
                if self.timeout_once and self.communicate_calls == 1:
                    raise r75_s1g_outer_exec.subprocess.TimeoutExpired(["fake"], timeout)
                return b"out", b"err"

            def kill(self):
                self.kill_calls += 1
                self.returncode = -9

        argv = r75_s1g_outer_exec.build_child_argv(RUNNER, ["--run-s1g"])
        self.assertEqual(argv[:2], [sys.executable, str(RUNNER)])
        for returncode, timeout_once, expected in (
                (0, False, 0), (1, False, 1), (None, False, None),
                (True, False, True), ("0", False, "0"), (0, True, -9)):
            process = FakeProcess(returncode, timeout_once=timeout_once)
            popen_calls = []
            original_popen = r75_s1g_outer_exec.subprocess.Popen

            def fake_popen(call_argv, *, cwd=None, stdout=None, stderr=None):
                popen_calls.append({"argv": list(call_argv), "cwd": cwd,
                                    "stdout": stdout, "stderr": stderr})
                return process

            r75_s1g_outer_exec.subprocess.Popen = fake_popen
            try:
                if returncode is None or returncode is True or returncode == "0":
                    with self.assertRaises(RuntimeError):
                        r75_s1g_outer_exec.launch_once(HERE, RUNNER, argv, 1)
                    payload = None
                else:
                    payload = r75_s1g_outer_exec.launch_once(HERE, RUNNER, argv, 1)
            finally:
                r75_s1g_outer_exec.subprocess.Popen = original_popen
            self.assertEqual(popen_calls, [{"argv": list(argv), "cwd": str(HERE),
                                            "stdout": r75_s1g_outer_exec.subprocess.PIPE,
                                            "stderr": r75_s1g_outer_exec.subprocess.PIPE}])
            self.assertEqual(process.communicate_calls, 2 if timeout_once else 1)
            self.assertEqual(process.kill_calls, 1 if timeout_once else 0)
            if returncode is None or returncode is True or returncode == "0":
                self.assertIsNone(payload)
            else:
                self.assertEqual(payload["child_returncode"], expected)
                self.assertEqual(payload["timed_out"], timeout_once)
                self.assertEqual(payload["stdout_bytes"], b"out")
                self.assertEqual(payload["stderr_bytes"], b"err")

        fixture_parent = HERE / "_r75_s1g_phase_a_test_fixture"
        fixture_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ar2_outcome_", dir=fixture_parent) as temp:
            for returncode, timed_out, expected_child, expected_wrapper in (
                    (0, False, 0, 0), (1, False, 1, 5),
                    (None, False, "MISSING_OR_INVALID", 5),
                    (True, False, "MISSING_OR_INVALID", 5),
                    ("0", False, "MISSING_OR_INVALID", 5),
                    (0, True, 0, 5)):
                child = {"stdout_bytes": b"out", "stderr_bytes": b"err",
                         "timed_out": timed_out, "started_at": {"unix": 1},
                         "ended_at": {"unix": 2}, "wall_seconds": 1, "child_pid": 123}
                if returncode is not None:
                    child["child_returncode"] = returncode
                payload = r75_s1g_outer_exec.record_outcome(
                    Path(temp), "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015", child,
                    "head", "parent", "branch", "wrapper", {"ledger": True}, ["file"])
                self.assertEqual(payload["child_returncode"], expected_child)
                self.assertEqual(payload["wrapper_return"], expected_wrapper)
                self.assertEqual(payload["stdout"]["bytes"], 3)
                self.assertTrue(payload["stdout"]["utf8_decoded_losslessly"])

    def test_precheck_baseline_and_headroom_reject_tampering(self):
        plan = json.loads(self.source(PLAN))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(r75_s1_precheck.main(PLAN), 0)
        verdict = json.loads(buffer.getvalue())
        self.assertEqual(verdict["revision_id"], "R75_S1G_AR1_PRECHECK_BASELINE_REPAIR_20261015")
        self.assertEqual(verdict["plan_path"], str(PLAN))
        self.assertTrue(verdict["passed"])

        report = r75_s1_precheck.Report()
        r75_s1_precheck.check_active_production_baseline(plan, report)
        self.assertTrue(report.passed)
        mutations = (
            lambda contract: contract.__setitem__("total_ledger_sha256", "0" * 64),
            lambda contract: contract["files"]["r75_total_ledger.jsonl"].__setitem__(
                "sha256", "0" * 64),
            lambda contract: contract.__setitem__("total_ledger_events", 89),
            lambda contract: contract["guards"]["physical_optimizer_step"].__setitem__(
                "used_normal", 6),
            lambda contract: contract["guards"]["physical_optimizer_step"].__setitem__(
                "refusals", 1),
        )
        for mutation in mutations:
            tampered = copy.deepcopy(plan)
            mutation(tampered["active_production_baseline"])
            tampered_report = r75_s1_precheck.Report()
            r75_s1_precheck.check_active_production_baseline(tampered, tampered_report)
            self.assertFalse(tampered_report.passed)

        for field in ("sum_normal", "sum_reserve"):
            tampered = copy.deepcopy(plan)
            tampered["active_production_baseline"]["guards"][
                "physical_optimizer_step"].pop(field)
            with self.assertRaises(KeyError):
                r75_s1_precheck.check_active_production_baseline(
                    tampered, r75_s1_precheck.Report())

        excess = copy.deepcopy(plan)
        excess["s1g_execution_contract"]["attempt_budget"]["totals"][
            "physical_optimizer_steps"] = {"normal": 9, "technical": 0, "hard": 9}
        excess["s1g_execution_contract"]["attempt_budget"]["cumulative_caps_after_attempt"][
            "physical_optimizer_step"] = {"normal": 14, "reserve": 0, "total": 14}
        excess_report = r75_s1_precheck.Report()
        r75_s1_precheck.check_s1g_headroom(excess, excess_report)
        by_name = {item["check"]: item["ok"] for item in excess_report.checks}
        self.assertTrue(by_name["headroom:physical_by_category"])
        self.assertFalse(by_name["headroom:physical_within_caps"])

    def test_guard_technical_charge_requires_pre_registration_fixture_only(self):
        fixture_parent = HERE / "_r75_s1g_phase_a_test_fixture"
        fixture_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ar2_", dir=fixture_parent) as temp:
            resources = r75_guard.R75Resources.fixture(Path(temp))
            prefix = "r75_s1g_A3_technical_retry_20261015"
            for index in range(5):
                resources.charge_physical_step(f"fixture:old5:{index}", category="normal")

            class FakeOptimizer:
                def __init__(self):
                    self.calls = 0

                def step(self):
                    self.calls += 1

            optimizer = FakeOptimizer()
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
            self.assertEqual(optimizer.calls, 0)

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

            resources.charge_offline_rows(840, "fixture:normal-rows", category="normal")
            snapshot = resources.snapshots()
            self.assertEqual(snapshot["physical_optimizer_step"]["used_normal"], 8)
            self.assertEqual(snapshot["physical_optimizer_step"]["used_reserve"], 1)
            self.assertEqual(snapshot["offline_trunk_target_row"]["used_normal"], 840)

    def test_stage_snapshot_abi_and_isolation(self):
        class State:
            unread: list[str] = []
            record: dict[str, object] = {}

            def counter_report(self):
                return {"fixture": True}

        fixture_parent = HERE / "_r75_s1g_phase_a_test_fixture"
        fixture_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ar2_stage_", dir=fixture_parent) as temp:
            path = Path(temp) / "stage_snapshot.pt"

            def save_fn(payload, target):
                target.write_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))

            identity = r75_s1_runner._save_stage_snapshot(
                path, stage="A2_completed", label="A2_completed", state=State(),
                full_state={"data_identity": {"fixture": True}}, save_fn=save_fn,
                identity_fn=lambda: {"head": "fixture-head"})
            self.assertTrue(path.is_file())
            self.assertEqual(identity["stage"], "A2_completed")
            self.assertTrue(identity["full_state_present"])
            self.assertEqual(identity["execution_head"], "fixture-head")

        runner_text = self.source(RUNNER)
        self.assertLess(runner_text.index('record["A2_stage_snapshot"] = _save_stage_snapshot('),
                        runner_text.index("end_diffs = compare_full_states("))
        self.assertLess(runner_text.index('record["A3_end_stage_snapshot"] = _save_stage_snapshot('),
                        runner_text.index("end_diffs = compare_full_states("))
        with self.assertRaises(r75_guard.R75LedgerError):
            r75_guard.R75Resources.fixture(r75_guard.R75_CANONICAL_ROOT)


if __name__ == "__main__":
    unittest.main()
