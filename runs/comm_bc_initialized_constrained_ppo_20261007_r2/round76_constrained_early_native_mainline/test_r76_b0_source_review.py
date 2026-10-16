"""R76 B0 source-review cases: pure stdlib; never run by B0 preparation."""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import r76_b_contracts as contracts
import r76_b_driver as driver
import r76_b_entry as entry
import r76_guard


HERE = Path(__file__).resolve().parent


def _source_manifest() -> dict[str, str]:
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in contracts.SOURCE_MANIFEST_FILES}


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def _function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one function {name} in {path}")
    return ast.dump(matches[0], annotate_fields=False, include_attributes=False)


def _valid_contract(source_head: str, driver_sha256: str) -> dict:
    return {
        "kind": contracts.CONTRACT_KIND,
        "status": "issued",
        "request_id": contracts.REQUEST_ID,
        "task_id": contracts.TASK_ID,
        "branch": contracts.BRANCH,
        "source_head": source_head,
        "scope": "r71_native_restore_qualification",
        "resource_root": str(contracts.RESOURCE_ROOT),
        "driver_sha256": driver_sha256,
        "source_manifest": _source_manifest(),
        "input_manifest": {"old_report": contracts.OLD_REPORT,
                           "r71_init": contracts.R71_INIT,
                           "episodes": list(contracts.EPISODES)},
        "resource_caps": contracts.RESOURCE_CAPS,
        "qualification_limits": dict(contracts.QUALIFICATION_LIMITS),
    }


def _valid_permit(source_head: str, contract_sha256: str) -> dict:
    return {
        "kind": contracts.PERMIT_KIND,
        "status": "issued",
        "request_id": contracts.REQUEST_ID,
        "task_id": contracts.TASK_ID,
        "branch": contracts.BRANCH,
        "source_head": source_head,
        "scope": "r71_native_restore_qualification",
        "contract_sha256": contract_sha256,
        "resource_root": str(contracts.RESOURCE_ROOT),
        "contract_path": str(HERE / "r76_b_driver.py"),
    }


class _FakeB2Ops:
    """Fine-grained pure orchestration spies; no real NN, disk, or resource is touched."""

    def __init__(self, *, verify_disk_drift: bool = False):
        self.calls: list[str] = []
        self.output_root = Path("fake-output-root")
        self.steps = 0
        self.input_counts = {"M": 265, "N": 265, "T": 620}
        self.episodes: list[dict] = []
        self.a1_baseline: dict | None = None
        self.scientific_files: dict[str, dict] = {}
        self.a1_disk_sha = "a1-disk-sha"
        self.verify_disk_drift = bool(verify_disk_drift)
        self.stage: dict | None = None
        self.result: dict | None = None
        self._model_state = {"model": 1}

    def resource_snapshot(self):
        self.calls.append("resource_snapshot")
        return {"physical_optimizer_step": {"used": self.steps}}

    def build_controller(self):
        self.calls.append("build_controller")
        return "controller", {"kind": "init"}

    def build_optimizer(self, controller, lr=None):
        self.calls.append("build_optimizer" if lr is None
                          else f"build_optimizer:{lr}")
        return "optimizer"

    def build_state(self):
        self.calls.append("build_state")
        return {"optimizer_steps": 0}

    def a1(self, controller, optimizer, state):
        self.calls.append("a1")
        self.steps += 1
        return {"optimizer_steps": 1}

    def capture(self, controller, optimizer, state):
        self.calls.append("capture")
        return {"state": copy.deepcopy(self._model_state)}

    def save_checkpoint(self, controller, optimizer, state, path):
        self.calls.append("save_checkpoint")
        name = Path(path).stem
        value = {"path": str(path), "sha256": f"{name}-sha"}
        if name == "r76_b_a1_state":
            self.a1_disk_sha = value["sha256"]
        self.scientific_files["a1" if name == "r76_b_a1_state" else name] = value
        return value

    def native_update(self, controller, optimizer, state, episodes):
        self.calls.append("native_update")
        self.steps += 8
        return {"optimizer_steps": 8}

    def verify_disk(self, path, expected_sha256):
        self.calls.append("verify_disk")
        if self.verify_disk_drift:
            raise RuntimeError("injected A1 disk drift")
        if expected_sha256 != self.a1_disk_sha:
            raise RuntimeError("A1 disk drifted")

    def load_checkpoint(self, path):
        self.calls.append("load_checkpoint")
        return "restored_controller", "restored_optimizer", {"optimizer_steps": 1}

    def verify_resources(self):
        self.calls.append("verify_resources")
        return {"consistent": True}

    def operation_counts(self):
        return {
            "controller_factory": {"attempted": 2, "completed": 2, "failed": 0},
            "adam_construction": {"attempted": 2, "completed": 2, "failed": 0},
            "init_load": {"attempted": 1, "completed": 1, "failed": 0},
            "own_a1_load": {"attempted": 1, "completed": 1, "failed": 0},
            "checkpoint_save": {"attempted": 3, "completed": 3, "failed": 0},
            "checkpoint_load": {"attempted": 2, "completed": 2, "failed": 0},
        }

    def record_stage(self, value):
        self.calls.append("record_stage")
        self.stage = value

    def write_result(self, value):
        self.calls.append("write_result")
        self.result = value


class R76B0SourceReview(unittest.TestCase):
    def test_contract_identity_permit_and_resource_role_refusals(self):
        source_head = "0" * 64
        driver_sha256 = hashlib.sha256(
            (HERE / "r76_b_driver.py").read_bytes()).hexdigest()
        contract = _valid_contract(source_head, driver_sha256)
        self.assertEqual(contracts.validate_contract(
            contract, source_head=source_head, driver_sha256=driver_sha256), contract)
        for mutate in (lambda value: value.update(task_id="foreign"),
                       lambda value: value.update(source_head="1" * 64),
                       lambda value: value.update(source_manifest={})):
            bad = {**contract}
            mutate(bad)
            with self.assertRaises(contracts.BContractError):
                contracts.validate_contract(bad, source_head=source_head,
                                            driver_sha256=driver_sha256)
        permit = _valid_permit(source_head, "1" * 64)
        with self.assertRaises(contracts.BContractError):
            contracts.validate_permit(permit, source_head=source_head,
                                      contract_sha256="2" * 64)
        with self.assertRaises(contracts.BContractError):
            contracts.validate_permit({**permit, "resource_root": str(HERE)},
                                      source_head=source_head,
                                      contract_sha256="1" * 64)

    def test_b0_state_entry_and_pure_orchestrate_close_17_steps(self):
        caps = contracts.validate_stage_caps(contracts.B_STAGE_CAPS)
        self.assertEqual(caps["b0"]["python_processes"], 0)
        self.assertEqual(caps["b0"]["fixed_case_executions"], 0)
        self.assertEqual(caps["b1"]["python_processes"], 2)
        self.assertEqual(caps["b1"]["distinct_cases"], 3)
        self.assertEqual(caps["b1"]["fixed_case_executions"], 6)
        self.assertEqual(caps["b2"]["normal_optimizer_steps"], 17)
        self.assertEqual(caps["b2"]["reserve_optimizer_steps"], 0)
        self.assertEqual(caps["b2"]["hard_optimizer_steps"], 17)
        self.assertEqual(caps["b2"]["checkpoint_saves"], 3)
        self.assertEqual(caps["b2"]["checkpoint_loads"], 2)

        meter = driver.B2OperationMeter({"physical_optimizer_step": 17})
        for _ in range(17):
            with meter.operation("physical_optimizer_step"):
                pass
        self.assertEqual(meter.snapshot()["physical_optimizer_step"],
                         {"attempted": 17, "completed": 17, "failed": 0})
        with self.assertRaises(RuntimeError):
            with meter.operation("physical_optimizer_step"):
                pass
        self.assertEqual(meter.snapshot()["physical_optimizer_step"],
                         {"attempted": 17, "completed": 17, "failed": 0})

        entry_source = (HERE / "r76_b_entry.py").read_text(encoding="utf-8")
        entry_code = compile(entry_source, str(HERE / "r76_b_entry.py"), "exec")
        old_argv = sys.argv
        try:
            sys.argv = [str(HERE / "r76_b_entry.py"), "--execute"]
            namespace = {"__name__": "__main__",
                         "__file__": str(HERE / "r76_b_entry.py")}
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    exec(entry_code, namespace)
            self.assertNotEqual(caught.exception.code, 0)
        finally:
            sys.argv = old_argv

        driver_imports = _top_level_imports(HERE / "r76_b_driver.py")
        self.assertNotIn("r76_native", driver_imports)
        self.assertIn("r76_runtime", driver_imports)
        self.assertEqual(driver._differences({"key": None}, {}),
                         ["key:missing_right"])
        self.assertEqual(driver._differences({}, {"key": None}),
                         ["key:missing_left"])

        ops = _FakeB2Ops()
        return_code = driver.orchestrate(ops)
        self.assertEqual(return_code, 0)
        self.assertIn("build_optimizer", ops.calls)
        self.assertEqual(ops.calls.count("build_optimizer"), 1)
        self.assertEqual(ops.calls, [
            "resource_snapshot", "build_controller", "build_optimizer",
            "build_state", "a1", "capture", "save_checkpoint",
            "resource_snapshot", "native_update", "capture", "save_checkpoint",
            "resource_snapshot", "verify_disk", "load_checkpoint", "capture",
            "native_update", "capture", "save_checkpoint", "resource_snapshot",
            "verify_resources", "write_result",
        ])
        self.assertEqual(ops.result["branch_path_optimizer_step_entries"],
                         {"a1_a2": 9, "a1_a3": 9})
        self.assertEqual(ops.result["central_optimizer_steps"], 17)
        self.assertEqual(ops.result["optimizer_step_entry_count"], 17)

        drifted_ops = _FakeB2Ops(verify_disk_drift=True)
        with self.assertRaises(RuntimeError):
            driver.orchestrate(drifted_ops)
        self.assertEqual(drifted_ops.stage["status"], "failed")
        self.assertEqual(drifted_ops.stage["phases"]["a3_restore"]["status"], "failed")
        self.assertEqual(drifted_ops.stage["phases"]["a3"]["status"], "not_executed")
        self.assertEqual(
            drifted_ops.stage["resource_snapshots"]["failure"]
            ["physical_optimizer_step"]["used"], 9)
        self.assertNotIn("load_checkpoint", drifted_ops.calls)

    def test_native_scientific_ast_and_guard_fixture_isolation(self):
        native_ast = _function_ast(HERE / "r76_native.py", "r75_native_ppo_update")
        frozen_ast = _function_ast(
            HERE.parent / "round75_native_ppo_qualification" / "r75_frozen.py",
            "r75_native_ppo_update")
        self.assertEqual(native_ast, frozen_ast)

        fixture_parent = (HERE / "raw" / "b1_fixture_temp").resolve()
        fixture_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r76_b1_",
                                         dir=fixture_parent) as temporary:
            self.assertEqual(Path(temporary).resolve().parent, fixture_parent)
            resources = r76_guard.R75Resources.fixture(temporary)
            before = resources.physical_steps.snapshot()
            self.assertEqual(before["used"], 0)
            resources.charge_physical_step("r76-b0-fixture-normal-1")
            resources.charge_physical_step("r76-b0-fixture-normal-2")
            resources.pre_register_technical_attempt("r76-b0-fixture-reserve")
            resources.charge_physical_step("r76-b0-fixture-reserve",
                                           category="technical")
            with self.assertRaises(r76_guard.R75QuotaExhausted):
                resources.charge_physical_step("r76-b0-fixture-over-hard")
            after = resources.physical_steps.snapshot()
            self.assertEqual(after["used_normal"], 2)
            self.assertEqual(after["used_reserve"], 1)
            self.assertEqual(after["used"], 3)
            for protected in (r76_guard.R76_CANONICAL_ROOT,
                              r76_guard.R76_QUALIFICATION_ROOT,
                              r76_guard.R75_HISTORICAL_CANONICAL_ROOT):
                with self.assertRaises(r76_guard.R75LedgerError):
                    r76_guard.R75Resources.fixture(protected)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
