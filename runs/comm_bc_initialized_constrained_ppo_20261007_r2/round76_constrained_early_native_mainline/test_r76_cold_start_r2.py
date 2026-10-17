"""R2 cold-start source review: three stdlib-only cases; not run during A0."""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import sys
import tempfile
import types
import unittest
from importlib.machinery import PathFinder
from pathlib import Path
from typing import Any

import r76_b_driver as driver


HERE = Path(__file__).resolve().parent
DRIVER_PATH = HERE / "r76_b_driver.py"
DRIVER_SOURCE = DRIVER_PATH.read_text(encoding="utf-8")
DRIVER_TREE = ast.parse(DRIVER_SOURCE, filename=str(DRIVER_PATH))
R66_DIR = HERE.parent / "round66_bc_early_path_preserving_late_adaptation"
R68_DIR = HERE.parent / "round68_late_initial_confidence_ablation"
FIXTURE_ROOT = (HERE / "raw" / "r2_fixture_temp").resolve()
MISSING = object()
TARGET_MODULE_NAMES = ("r66_execute", "r68_execute", "torch", "numpy")


def _function_node(name: str) -> ast.FunctionDef:
    matches = [node for node in DRIVER_TREE.body
               if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one function {name} in {DRIVER_PATH}")
    return matches[0]


def _install_real_body(namespace: dict[str, Any], name: str) -> None:
    node = _function_node(name)
    source = ast.get_source_segment(DRIVER_SOURCE, node)
    if source is None:
        raise AssertionError(f"cannot extract source for {name}")
    code = compile(source, str(DRIVER_PATH), "exec")
    exec(code, namespace)


class R76ColdStartR2SourceReview(unittest.TestCase):
    def test_1_canonical_r68_resolves_without_import_or_execution(self):
        original_path = list(sys.path)
        for name in TARGET_MODULE_NAMES:
            self.assertNotIn(name, sys.modules)
        try:
            identity = driver.validate_r2_cold_start_paths()
            self.assertEqual(identity["kind"], "round76-r2-cold-start-path-identity/1")
            self.assertEqual(
                identity["sources"],
                {
                    "r66_execute": str((R66_DIR / "r66_execute.py").resolve()),
                    "r68_execute": str((R68_DIR / "r68_execute.py").resolve()),
                })
            for name in TARGET_MODULE_NAMES:
                self.assertNotIn(name, sys.modules)

            self.assertIsNone(
                PathFinder.find_spec("r68_execute", [str(R66_DIR)]))
            self.assertIsNotNone(
                PathFinder.find_spec("r68_execute", [str(R68_DIR)]))
        finally:
            sys.path[:] = original_path

    def test_2_missing_foreign_and_cached_module_sources_are_rejected(self):
        FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=FIXTURE_ROOT) as temporary:
            root = Path(temporary).resolve()
            self.assertEqual(root.parent, FIXTURE_ROOT)
            original_sources = driver.R2_COLD_START_SOURCES
            missing_sources = {
                name: {
                    "path": root / f"missing-{name}.py",
                    "bytes": 1,
                    "sha256": "0" * 64,
                }
                for name in original_sources
            }
            driver.R2_COLD_START_SOURCES = missing_sources
            try:
                with self.assertRaisesRegex(RuntimeError, "missing-r66_execute"):
                    driver.validate_r2_cold_start_paths()
            finally:
                driver.R2_COLD_START_SOURCES = original_sources

        for name, expected in original_sources.items():
            path = expected["path"].resolve()
            with self.subTest(module=name):
                original_module = sys.modules.get(name)
                foreign = types.ModuleType(name)
                foreign.__spec__ = types.SimpleNamespace(
                    origin=str(root / f"foreign-{name}.py"))
                sys.modules[name] = foreign
                try:
                    with self.assertRaisesRegex(RuntimeError, "foreign or opaque"):
                        driver.validate_r2_cold_start_paths()
                finally:
                    if original_module is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = original_module

                contradictory = types.ModuleType(name)
                contradictory.__spec__ = types.SimpleNamespace(origin=str(path))
                contradictory.__file__ = str(root / f"contradictory-{name}.py")
                sys.modules[name] = contradictory
                try:
                    with self.assertRaisesRegex(RuntimeError, "__file__ does not match"):
                        driver.validate_r2_cold_start_paths()
                finally:
                    if original_module is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = original_module

                opaque = types.ModuleType(name)
                opaque.__file__ = str(path)
                sys.modules[name] = opaque
                try:
                    with self.assertRaisesRegex(RuntimeError, "foreign or opaque"):
                        driver.validate_r2_cold_start_paths()
                finally:
                    if original_module is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = original_module

        with self.assertRaisesRegex(RuntimeError, "final search origin mismatch"):
            driver._resolve_final_search_origin(
                "r66_execute", (R68_DIR / "r68_execute.py").resolve())

    def test_3_configure_cold_start_calls_and_zero_env_binding_with_fakes(self):
        FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=FIXTURE_ROOT) as temporary:
            run_root = Path(temporary).resolve()
            self.assertEqual(run_root.parent, FIXTURE_ROOT)
            counts_root = run_root / "counts-root"
            counts_root.mkdir()
            original_path = list(sys.path)
            for name in TARGET_MODULE_NAMES:
                self.assertNotIn(name, sys.modules)
            events: list[str] = []
            factory_calls: list[int] = []
            seen: dict[str, Any] = {}
            contract = {
                "request_id": "R76_R2_REQ", "task_id": "R76_R2_TASK",
                "branch": "exp/round76-constrained-early-native-mainline",
                "scope": "r76-r2-qualification", "source_head": "source-head",
                "input_manifest": {"M": 265, "N": 17, "T": 3},
                "resource_caps": {
                    "environment_construction": {
                        "normal": 0, "reserve": 0, "hard": 0,
                    },
                },
            }
            counts = dict(contract["input_manifest"])
            fake_r40 = types.SimpleNamespace(
                NORMAL_CAP=999, RESERVE_CAP=999, HARD_CAP=999)
            fake_common = types.SimpleNamespace(
                NORMAL_LAUNCHES=999, RESERVE_LAUNCHES=999, LAUNCH_CAP=999)
            fake_cfg = {
                "runtime": {"r40_common": fake_r40},
                "common": fake_common,
            }
            fake_resources = types.SimpleNamespace(
                verify_or_fail=lambda: events.append("resource"))
            fake_guard = types.SimpleNamespace(
                R75Resources=types.SimpleNamespace(
                    qualification_from_contract=lambda value: fake_resources))
            fake_b_contracts = types.SimpleNamespace(
                REQUEST_ID="R76_R2_REQ", TASK_ID="R76_R2_TASK",
                RESOURCE_ROOT=run_root / "qualification")
            fake_r66 = types.SimpleNamespace()
            fake_r68 = types.SimpleNamespace()
            fake_native = types.SimpleNamespace()
            fake_r66.R66_DIR = (R66_DIR).resolve()

            def fake_configure_and_inject(raw_root: Path) -> dict[str, Any]:
                seen["r66_raw_root"] = Path(raw_root)
                events.append("r66.configure")
                return fake_cfg

            def fake_source_manifest(native: Any, r68: Any,
                                     cfg: dict[str, Any]) -> dict[str, Any]:
                seen["source_native"] = native
                seen["source_r68"] = r68
                seen["source_cfg"] = cfg
                events.append("source_manifest")
                return {"source": "manifest"}

            def fake_fresh_controller(native: Any, seed: int) -> dict[str, int]:
                seen["factory_native"] = native
                factory_calls.append(int(seed))
                return {"seed": int(seed)}

            fake_r66.configure_and_inject = fake_configure_and_inject
            fake_r66._validated_source_manifest = fake_source_manifest
            fake_r66._fresh_architecture_controller = fake_fresh_controller

            def fake_bindings(cfg: dict[str, Any], resources: Any, native: Any, *,
                              request_id: str, task_id: str) -> dict[str, Any]:
                seen["bindings_cfg"] = cfg
                seen["bindings_resources"] = resources
                seen["bindings_native"] = native
                seen["bindings_request_id"] = request_id
                seen["bindings_task_id"] = task_id
                events.append("bindings")
                return {"runtime": "identity"}

            fake_r76_runtime = types.SimpleNamespace(
                install_r76_runtime_bindings=fake_bindings)
            fake_import_modules = {
                "r66_execute": fake_r66,
                "r68_execute": fake_r68,
                "r76_guard": fake_guard,
                "r76_native": fake_native,
            }

            def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
                if level != 0 or name not in fake_import_modules:
                    raise ImportError(f"forbidden real import: {name}")
                events.append(f"import:{name}")
                return fake_import_modules[name]

            isolated_builtins = vars(builtins).copy()
            isolated_builtins["__import__"] = fake_import

            namespace: dict[str, Any] = {
                "__builtins__": isolated_builtins,
                "Any": Any,
                "Path": Path,
                "PathFinder": PathFinder,
                "argparse": argparse,
                "hashlib": hashlib,
                "json": json,
                "os": __import__("os"),
                "sys": sys,
                "HERE": run_root,
                "R2_CANONICAL_R66_DIR": R66_DIR.resolve(),
                "R2_CANONICAL_R68_DIR": R68_DIR.resolve(),
                "R2_COLD_START_SOURCES": {
                    "r66_execute": {
                        "path": (R66_DIR / "r66_execute.py").resolve(),
                        "bytes": 53131,
                        "sha256": "9164f0c989847e78d1018512bcdd26cb1d234971f61f05f03a6d58927cf14670",
                    },
                    "r68_execute": {
                        "path": (R68_DIR / "r68_execute.py").resolve(),
                        "bytes": 61201,
                        "sha256": "c123dfe753effc5b6e35b8ce680e6bb1092a80a0d0deb40a0971df09e1c2cd3e",
                    },
                },
                "r76_execute": fake_r66,
                "r76_runtime": fake_r76_runtime,
                "b_contracts": fake_b_contracts,
                "validate_b2_release": lambda gates: events.append("gate"),
                "validate_b2_bundle": lambda value: (
                    events.append("bundle"), seen.update(bundle_args=value),
                    (contract, "permit", "contract-sha"))[-1],
                "pre_model_input_counts": lambda manifest: (
                    events.append("counts"), seen.update(counts_manifest=manifest),
                    (counts, [{"episode": 0}]))[-1],
                "_read_json": lambda path: {"gates": {"all": True}},
                "_build_identity_context": lambda cfg: (
                    events.append("identity"), seen.update(identity_cfg=cfg),
                    ({"data": True}, {"reference": True}))[-1],
                "orchestrate": lambda ops: (
                    events.append("orchestrate"), seen.update(orchestrate_ops=ops), 7)[-1],
            }

            class FakeRealB2Ops:
                def __init__(self, cfg: dict[str, Any], resource_root: Path,
                             input_counts: dict[str, Any],
                             episodes: list[dict[str, Any]]) -> None:
                    seen["ops_cfg"] = cfg
                    seen["ops_resource_root"] = resource_root
                    seen["ops_input_counts"] = input_counts
                    seen["ops_episodes"] = episodes
                    events.append("ops")

            namespace["RealB2Ops"] = FakeRealB2Ops
            for name in ("sha256_file", "_resolve_final_search_origin",
                         "validate_r2_cold_start_paths", "_write_r2_input_counts",
                         "configure_qualification_runtime", "run_b2"):
                _install_real_body(namespace, name)

            real_writer = namespace["_write_r2_input_counts"]
            real_cold_start = namespace["validate_r2_cold_start_paths"]

            def wrapped_writer(counts_value: dict[str, Any],
                               contract_value: dict[str, Any]) -> dict[str, Any]:
                events.append("writer")
                return real_writer(counts_value, contract_value,
                                   capture_root=counts_root)

            def wrapped_cold_start() -> dict[str, Any]:
                events.append("cold_start")
                return real_cold_start()

            namespace["_write_r2_input_counts"] = wrapped_writer
            namespace["validate_r2_cold_start_paths"] = wrapped_cold_start
            fake_args = types.SimpleNamespace(
                contract_path="contract", permit_path="permit",
                source_head="source-head")

            try:
                result = namespace["run_b2"](fake_args)
                self.assertEqual(result, 7)
                self.assertEqual(events, [
                    "gate", "bundle", "counts", "writer", "cold_start",
                    "import:r66_execute", "import:r68_execute",
                    "import:r76_guard", "import:r76_native", "r66.configure",
                    "resource", "source_manifest", "bindings", "identity",
                    "ops", "orchestrate",
                ])
                self.assertEqual(seen["bundle_args"], fake_args)
                self.assertEqual(seen["counts_manifest"], contract["input_manifest"])
                self.assertEqual(seen["r66_raw_root"], run_root / "raw")
                self.assertEqual(seen["source_native"], fake_native)
                self.assertEqual(seen["source_r68"], fake_r68)
                self.assertEqual(seen["source_cfg"], fake_cfg)
                self.assertEqual(seen["bindings_cfg"], fake_cfg)
                self.assertEqual(seen["bindings_resources"], fake_resources)
                self.assertEqual(seen["bindings_request_id"], contract["request_id"])
                self.assertEqual(seen["bindings_task_id"], contract["task_id"])
                self.assertEqual(seen["ops_resource_root"], fake_b_contracts.RESOURCE_ROOT)
                self.assertEqual(seen["ops_input_counts"], counts)
                self.assertEqual(seen["ops_episodes"], [{"episode": 0}])
                self.assertEqual(
                    (fake_r40.NORMAL_CAP, fake_r40.RESERVE_CAP, fake_r40.HARD_CAP),
                    (0, 0, 0))
                self.assertEqual(
                    (fake_common.NORMAL_LAUNCHES, fake_common.RESERVE_LAUNCHES,
                     fake_common.LAUNCH_CAP), (0, 0, 0))
                self.assertEqual(factory_calls, [])
                self.assertEqual(
                    fake_cfg["r76_controller_factory"](11), {"seed": 11})
                self.assertIs(seen["factory_native"], fake_native)
                self.assertEqual(factory_calls, [11])
                for name in TARGET_MODULE_NAMES:
                    self.assertNotIn(name, sys.modules)

                output_path = counts_root / "input_counts.json"
                payload = json.loads(output_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["input_counts"], counts)
                self.assertEqual(payload["identity"]["request_id"], contract["request_id"])
                self.assertEqual(payload["identity"]["task_id"], contract["task_id"])
                self.assertEqual(payload["identity"]["source_head"], contract["source_head"])
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    wrapped_writer(counts, contract)
            finally:
                sys.path[:] = original_path


if __name__ == "__main__":
    raise SystemExit(unittest.main())
