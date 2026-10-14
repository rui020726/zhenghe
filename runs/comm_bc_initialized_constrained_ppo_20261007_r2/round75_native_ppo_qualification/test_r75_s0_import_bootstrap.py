"""Narrow counterexamples for the ROUND75 import-bootstrap repair (R75_S0_IMPORT_BOOTSTRAP_REPAIR_20261015).

Scope discipline, stated up front because it is easy to misread a green run here:

* This module imports ONLY the standard library, plus the two pure-stdlib files it exercises by
  path (r75_frozen.py is parsed with ``ast`` and its verification function is executed against
  FAKE module objects; r75_s1_outer_exec_v2.py is imported because it is stdlib-only).
* It never imports numpy, torch, r75_frozen, r75_guard, r66_frozen or any canonical module, never
  loads or constructs a model, never runs a forward/backward, never steps an optimizer, never
  touches the environment and never launches the production runner.
* A pass here therefore proves the WIRING and the REFUSAL LOGIC only. It is not an import
  qualification, not a real-Adam proof and not a policy result.

The fake file trees are created under a scratch directory inside this output root and removed
again; nothing is written to raw/s1/, to raw/runtime/ or to the canonical trees.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN_SRC = HERE / "r75_frozen.py"
WRAPPER_SRC = HERE / "r75_s1_outer_exec_v2.py"
PLAN_PATH = HERE / "ROUND75_S1_RUN_PLAN.json"
R66_SRC = HERE.parent / "round66_bc_early_path_preserving_late_adaptation" / "r66_frozen.py"
REAL_REPO = HERE.parents[2]

KEEP_FUNCTIONS = ("_sha256_of", "_verify_canonical_bindings")
KEEP_CONSTANTS = ("EXPECTED_TRAINER_SHA256", "EXPECTED_COMMON_SHA256",
                  "EXPECTED_CONTROLLER_SHA256", "EXPECTED_ONPOLICY_SHA256",
                  "EXPECTED_COST_MATH_SHA256", "CANONICAL_IMPORT_NAMES")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)
    return path


class Scratch(unittest.TestCase):
    """Base case giving every test its own scratch directory inside the output root."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="_r75_scratch_", dir=str(HERE))
        self.tmp = Path(self._tmp)

    def tearDown(self) -> None:
        for entry in sorted(self.tmp.rglob("*"), reverse=True):
            try:
                if entry.is_file() or entry.is_symlink():
                    entry.unlink()
                else:
                    entry.rmdir()
            except OSError:
                pass
        try:
            self.tmp.rmdir()
        except OSError:
            pass
        self.assertFalse(self.tmp.exists(), "scratch directory was not cleaned up")


class WiringAstTest(Scratch):
    """The real source must be wired the way the report claims, checked on the file on disk."""

    def setUp(self) -> None:
        super().setUp()
        self.source = FROZEN_SRC.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def _imports(self):
        found = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ("v2_trainer", "v2_common", "onpolicy", "cost_math",
                                      "v2_controller"):
                        found.append((node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.module == "onpolicy":
                found.append((node.lineno, "onpolicy"))
        return found

    def test_helper_runs_before_every_canonical_import(self) -> None:
        helper_line = None
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "CANONICAL_SOURCES":
                        helper_line = node.lineno
        imports = self._imports()
        self.assertTrue(imports, "no canonical import found")
        self.assertIsNotNone(helper_line)
        self.assertLess(helper_line, min(line for line, _ in imports),
                        "the canonical import bootstrap must run before any canonical import")

    def test_cost_math_is_never_imported_by_this_module(self) -> None:
        imported = [name for _, name in self._imports() if name == "cost_math"]
        self.assertEqual(imported, [],
                         "cost_math must be read back from sys.modules, not imported again")

    def test_bindings_verified_at_import_time(self) -> None:
        calls = [node for node in ast.walk(self.tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "_verify_canonical_bindings"]
        self.assertTrue(calls, "_verify_canonical_bindings() is not invoked at module level")

    def test_identity_check_compares_two_different_expressions(self) -> None:
        """The old defect compared one dictionary read with itself; that must be impossible now."""
        function = None
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_verify_canonical_bindings":
                function = node
        self.assertIsNotNone(function)
        segment = ast.get_source_segment(self.source, function) or ""
        self.assertIn("used = {", segment,
                      "the verification must name the bindings this module actually uses")
        self.assertIn("sys.modules.get(", segment,
                      "the verification must compare against what the import system cached")
        for line in segment.splitlines():
            self.assertNotRegex(line, r"^\s*module\s*=\s*CANONICAL_SOURCES\.get\(key\)\s*$",
                                "a bare re-read of the same dictionary entry is not a comparison")
        compares = [node for node in ast.walk(function)
                    if isinstance(node, ast.Compare) and len(node.ops) == 1
                    and isinstance(node.ops[0], (ast.Is, ast.IsNot))]
        self.assertGreaterEqual(len(compares), 6,
                                "object identity must be compared for the helper return, the used "
                                "binding and the sys.modules cache")

    def test_onpolicy_and_cost_math_pins_match_disk_and_plan(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        identity = plan["source_identity"]
        for key, constant in (("canonical_onpolicy", "EXPECTED_ONPOLICY_SHA256"),
                              ("canonical_cost_math", "EXPECTED_COST_MATH_SHA256")):
            pinned = None
            for node in ast.walk(self.tree):
                if isinstance(node, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == constant for t in node.targets):
                    pinned = ast.literal_eval(node.value)
            self.assertIsNotNone(pinned, constant + " is not pinned in r75_frozen.py")
            entry = identity[key]
            self.assertEqual(pinned, entry["expected"],
                             constant + " disagrees with the frozen plan")
            self.assertEqual(pinned, entry["sha256"],
                             constant + " disagrees with the recorded disk digest")
            self.assertEqual(sha256_bytes(Path(entry["path"]).read_bytes()), pinned,
                             constant + " does not match the file on disk")

    def test_helper_source_is_the_pinned_r66(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        entry = plan["source_identity"]["r66_frozen"]
        self.assertTrue(R66_SRC.is_file())
        self.assertEqual(sha256_bytes(R66_SRC.read_bytes()), entry["expected"])

def _extracted_verification_code():
    """The real constants plus the real verification functions, lifted out of r75_frozen.py.

    r75_frozen.py itself is never imported (it would pull numpy and torch), so the check is
    executed from the source on disk against fake module objects.
    """
    tree = ast.parse(FROZEN_SRC.read_text(encoding="utf-8"))
    keep = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in KEEP_FUNCTIONS:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in KEEP_CONSTANTS for t in node.targets):
            keep.append(node)
    module = ast.Module(body=keep, type_ignores=[])
    return compile(ast.fix_missing_locations(module), str(FROZEN_SRC), "exec")


MISSING = object()


class FakeTree:
    """A throwaway canonical tree whose digests become the pins for this one test."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.canon = root / "canon"
        self.tight = root / "tight"
        self.stage_a = root / "stage_a"
        self.checkout = root / "checkout_copy"
        self.trainer = write(self.canon / "v2_trainer.py", b"frozen canonical trainer source\n")
        self.common = write(self.canon / "v2_common.py", b"frozen canonical common source\n")
        self.controller = write(self.canon / "v2_controller.py",
                                b"frozen canonical controller source\n")
        self.onpolicy = write(self.tight / "onpolicy.py", b"frozen onpolicy math source\n")
        self.cost_math = write(self.stage_a / "cost_math.py", b"frozen cost math source\n")
        self.decoy_cost_math = write(self.checkout / "cost_math.py",
                                     b"frozen cost math source\n")
        self.decoy_trainer = write(self.checkout / "v2_trainer.py", b"a different trainer\n")

    @staticmethod
    def module(name: str, file: Path, **attributes):
        module = types.ModuleType(name)
        module.__file__ = str(file)
        for key, value in attributes.items():
            setattr(module, key, value)
        return module

    def common_module(self):
        return self.module("v2_common", self.common,
                           TIGHT_ROUND=str(self.tight), STAGE_A_DIR=str(self.stage_a),
                           GAE_LAMBDA=1.0, ETA_LAMBDA=0.5, GAMMA=1.0)

    def onpolicy_module(self):
        def entropy(*_args):
            return 0.0

        def joint(*_args):
            return 0.0

        module = self.module("onpolicy", self.onpolicy,
                             action_entropy=entropy, joint_log_prob=joint)
        return module, entropy, joint


class sys_modules_patched:
    """Install fake sys.modules entries for the duration of one check, then restore them."""

    def __init__(self, mapping) -> None:
        self.mapping = dict(mapping)
        self.saved = {}

    def __enter__(self):
        for key, value in self.mapping.items():
            if value is None:
                continue
            self.saved[key] = sys.modules.get(key)
            sys.modules[key] = value
        return self

    def __exit__(self, *_exc):
        for key, value in self.saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
        return False


class BindingVerificationTest(Scratch):
    """Execute the real _verify_canonical_bindings against fakes: one good case, many refusals."""

    def setUp(self) -> None:
        super().setUp()
        self.tree = FakeTree(self.tmp / "canonical_tree")
        self.code = _extracted_verification_code()

    def namespace(self, *, used_trainer=MISSING, used_common=MISSING, used_controller=MISSING,
                  used_onpolicy=MISSING, cost_math=MISSING, cached=None, helper_return=MISSING,
                  live_overrides=None, pins=None):
        tree = self.tree
        common = tree.common_module() if used_common is MISSING else used_common
        onpolicy, entropy, joint = tree.onpolicy_module()
        if used_onpolicy is not MISSING:
            onpolicy = used_onpolicy
        trainer = (tree.module("v2_trainer", tree.trainer) if used_trainer is MISSING
                   else used_trainer)
        controller = (tree.module("v2_controller", tree.controller)
                      if used_controller is MISSING else used_controller)
        cost = tree.module("cost_math", tree.cost_math) if cost_math is MISSING else cost_math
        returned = ({"trainer": trainer, "common": common, "controller_module": controller}
                    if helper_return is MISSING else helper_return)
        cache = {"v2_trainer": returned.get("trainer"), "v2_common": returned.get("common"),
                 "v2_controller": returned.get("controller_module"), "onpolicy": onpolicy,
                 "cost_math": cost}
        if cached:
            cache.update(cached)
        if live_overrides:
            for key, value in live_overrides.items():
                setattr(common, key, value)
        namespace = {"sys": sys, "Path": Path, "hashlib": hashlib, "Any": object,
                     "r66": types.SimpleNamespace(CANON=str(tree.canon)),
                     "CANONICAL_SOURCES": returned, "trainer": trainer, "common": common,
                     "onpolicy": onpolicy, "action_entropy": entropy, "joint_log_prob": joint,
                     "CONTROLLER_MODULE": controller, "COST_MATH_MODULE": cost}
        exec(self.code, namespace)
        namespace["EXPECTED_TRAINER_SHA256"] = sha256_bytes(tree.trainer.read_bytes())
        namespace["EXPECTED_COMMON_SHA256"] = sha256_bytes(tree.common.read_bytes())
        namespace["EXPECTED_CONTROLLER_SHA256"] = sha256_bytes(tree.controller.read_bytes())
        namespace["EXPECTED_ONPOLICY_SHA256"] = sha256_bytes(tree.onpolicy.read_bytes())
        namespace["EXPECTED_COST_MATH_SHA256"] = sha256_bytes(tree.cost_math.read_bytes())
        if pins:
            namespace.update(pins)
        return namespace, cache

    def verify(self, **kwargs):
        namespace, cache = self.namespace(**kwargs)
        with sys_modules_patched(cache):
            return namespace["_verify_canonical_bindings"]()

    def refuse(self, **kwargs):
        namespace, cache = self.namespace(**kwargs)
        with sys_modules_patched(cache):
            with self.assertRaises(RuntimeError) as caught:
                namespace["_verify_canonical_bindings"]()
        return str(caught.exception)

    def test_consistent_bindings_are_accepted(self) -> None:
        bindings = self.verify()
        self.assertEqual(bindings["gae_lambda"], 1.0)
        self.assertEqual(bindings["eta_lambda"], 0.5)
        self.assertEqual(bindings["gamma"], 1.0)
        self.assertEqual(bindings["onpolicy_sha256"],
                         sha256_bytes(self.tree.onpolicy.read_bytes()))
        self.assertEqual(bindings["cost_math_sha256"],
                         sha256_bytes(self.tree.cost_math.read_bytes()))
        for label in ("trainer", "common", "controller"):
            self.assertTrue(bindings["bindings"][label]["used_is_helper_return"], label)
            self.assertTrue(bindings["bindings"][label]["sys_modules_is_helper_return"], label)
        self.assertIn("canonical_root", bindings)

    def real_bindings(self):
        """The objects the helper would return for this fake tree."""
        tree = self.tree
        return {"trainer": tree.module("v2_trainer", tree.trainer),
                "common": tree.common_module(),
                "controller_module": tree.module("v2_controller", tree.controller)}

    def test_same_path_same_digest_different_object_is_refused(self) -> None:
        """A decoy object pointing at the very same file must not pass as the real binding."""
        tree = self.tree
        real = self.real_bindings()
        decoy = tree.module("v2_trainer", tree.trainer)
        message = self.refuse(used_trainer=decoy, used_common=real["common"],
                              used_controller=real["controller_module"],
                              helper_return=real)
        self.assertIn("trainer", message)
        self.assertIn("not the object", message)

    def test_stale_or_shadowing_sys_modules_entry_is_refused(self) -> None:
        tree = self.tree
        stranger = tree.module("v2_common", tree.common)
        message = self.refuse(cached={"v2_common": stranger})
        self.assertIn("sys.modules", message)

    def test_wrong_file_for_the_pinned_digest_is_refused(self) -> None:
        message = self.refuse(used_trainer=self.tree.module("v2_trainer", self.tree.decoy_trainer))
        self.assertIn("trainer", message)

    def test_wrong_living_value_is_refused(self) -> None:
        message = self.refuse(live_overrides={"GAE_LAMBDA": 0.95})
        self.assertIn("GAE_LAMBDA", message)

    def test_missing_cost_math_is_refused(self) -> None:
        message = self.refuse(cost_math=None)
        self.assertIn("cost_math", message)

    def test_cost_math_from_the_wrong_tree_is_refused_even_with_the_same_digest(self) -> None:
        """Both trees carry identical bytes, so only the resolved path can tell them apart."""
        tree = self.tree
        message = self.refuse(cost_math=tree.module("cost_math", tree.decoy_cost_math))
        self.assertIn("cost_math", message)

    def test_helper_returning_nothing_is_refused(self) -> None:
        message = self.refuse(helper_return={})
        self.assertIn("configure_source_imports() returned no", message)

    def test_unbound_onpolicy_helper_is_refused(self) -> None:
        tree = self.tree
        module, _entropy, _joint = tree.onpolicy_module()

        def other(*_args):
            return 0.0

        namespace, cache = self.namespace(used_onpolicy=module)
        namespace["action_entropy"] = other
        with sys_modules_patched(cache):
            with self.assertRaises(RuntimeError) as caught:
                namespace["_verify_canonical_bindings"]()
        self.assertIn("action_entropy", str(caught.exception))

def load_wrapper():
    """Import the revised wrapper. It is stdlib-only, so importing it is not a paid action."""
    spec = importlib.util.spec_from_file_location("r75_s1_outer_exec_v2_under_test",
                                                  str(WRAPPER_SRC))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_child(returncode: int, out: bytes = b"child stdout bytes\n",
               err: bytes = b"child stderr bytes\n"):
    return {"started_at": {"unix": 1.0, "iso": "2026-10-15T00:00:00+08:00"},
            "ended_at": {"unix": 2.0, "iso": "2026-10-15T00:00:01+08:00"},
            "wall_seconds": 1.0, "child_pid": 4242, "child_returncode": returncode,
            "timed_out": False, "timeout_s": 7200, "stdout_bytes": out, "stderr_bytes": err}


class WrapperBehaviourTest(Scratch):
    """Drive the real wrapper functions against fakes: no child is ever really started."""

    def setUp(self) -> None:
        super().setUp()
        self.mod = load_wrapper()
        self.here = self.tmp / "here"
        self.here.mkdir(parents=True)
        self.code = {}
        for name in ("r75_s1_runner.py", "r75_s1_precheck.py", "r75_frozen.py",
                     "r75_guard.py", "r75_replay_adapter.py", "ROUND75_S1_RUN_PLAN.json"):
            path = write(self.here / name, ("fake payload of " + name + "\n").encode("utf-8"))
            self.code[name] = sha256_bytes(path.read_bytes())
        self.runtime = self.here / "raw" / "runtime"
        self.runtime.mkdir(parents=True)
        rows = [json.dumps({"unix": 1791981043.0 + index, "guard": "fake-ledger/1",
                            "event": "ledger-opened", "unit": "physical_optimizer_step",
                            "used_normal": 0, "used_reserve": 0, "refusals": 0})
                for index in range(12)]
        write(self.runtime / "r75_total_ledger.jsonl", ("\n".join(rows) + "\n").encode("utf-8"))
        snapshot = self.mod.ledger_snapshot(self.runtime, None)
        self.plan = {"canonical_ledger_frozen_snapshot": {"files": snapshot["files"]}}
        self.data = {}
        for role in ("row", "tape", "model"):
            path = write(self.tmp / ("fake_" + role + ".json"), b"fake data payload\n")
            self.data[role] = (str(path), sha256_bytes(path.read_bytes()))
        self.first = self.here / "raw" / "s1"
        self.first.mkdir(parents=True)
        self.first_files = {}
        for name in self.mod.EXPECT_FIRST_ATTEMPT_FILES:
            path = write(self.first / name, ("preserved original " + name + "\n").encode("utf-8"))
            self.first_files[name] = sha256_bytes(path.read_bytes())
        self.mod.EXPECT_CODE = dict(self.code)
        self.mod.EXPECT_DATA = {role: (rel, want) for role, (rel, want) in self.data.items()}
        self.mod.EXPECT_FIRST_ATTEMPT_FILES = dict(self.first_files)
        self.out = self.here / "raw" / "s1_revised"
        self.out.mkdir(parents=True)
        self.launches = []

    def launch(self, returncode: int):
        def _launch(*_args, **_kwargs):
            self.launches.append(returncode)
            return fake_child(returncode)
        self.mod.launch_once = _launch

    def outcome(self):
        return json.loads((self.out / self.mod.OUTCOME_RECORD).read_text(encoding="utf-8"))

    def test_prepare_verifies_and_starts_nothing(self) -> None:
        payload, problems = self.mod.prepare(REAL_REPO, self.here, self.out,
                                            "R75_TEST_PREPARE", self.plan, self.runtime)
        self.assertEqual(problems, [])
        self.assertFalse(payload["launched"])
        self.assertFalse(payload["attempt_marker_created"])
        self.assertTrue((self.out / self.mod.PREPARED_RECORD).is_file())
        self.assertEqual(list(self.out.glob("r75_s1_outer_v2_attempt_marker_*.json")), [])
        self.assertEqual(self.launches, [])
        self.assertTrue(payload["first_attempt"]["present"])
        self.assertEqual(payload["first_attempt"]["problems"], [])

    def test_successful_child_records_the_three_facts_separately(self) -> None:
        self.launch(0)
        code, problems = self.mod.run(REAL_REPO, self.here, self.out, "R75_TEST_CHILD_OK",
                                      self.plan, self.runtime)
        self.assertEqual((code, problems), (0, []))
        self.assertEqual(self.launches, [0])
        record = self.outcome()
        self.assertEqual(record["child_returncode"], 0)
        self.assertEqual(record["wrapper_return"], 0)
        self.assertIsNone(record["wrapper_exception"])
        self.assertEqual(record["stdout"]["sha256"],
                         sha256_bytes(b"child stdout bytes\n"))
        self.assertTrue((self.out / "r75_s1_outer_v2_stdout.bin").is_file())
        markers = list(self.out.glob("r75_s1_outer_v2_attempt_marker_*.json"))
        self.assertEqual(len(markers), 1)

    def test_failing_child_keeps_the_real_exit_code(self) -> None:
        self.launch(1)
        code, problems = self.mod.run(REAL_REPO, self.here, self.out, "R75_TEST_CHILD_FAIL",
                                      self.plan, self.runtime)
        self.assertEqual(code, 5)
        record = self.outcome()
        self.assertEqual(record["child_returncode"], 1)
        self.assertEqual(record["wrapper_return"], 5)
        self.assertIsNone(record["wrapper_exception"])

    def test_the_same_request_id_cannot_run_twice(self) -> None:
        self.launch(0)
        first_code, _ = self.mod.run(REAL_REPO, self.here, self.out, "R75_TEST_DUPLICATE",
                                     self.plan, self.runtime)
        second_code, problems = self.mod.run(REAL_REPO, self.here, self.out,
                                             "R75_TEST_DUPLICATE", self.plan, self.runtime)
        self.assertEqual(first_code, 0)
        self.assertEqual(second_code, 4)
        self.assertEqual(len(self.launches), 1, "a duplicate request must not start a child")
        self.assertTrue(any("already exists" in problem for problem in problems))

    def test_the_first_attempt_id_is_refused_in_a_different_output_directory(self) -> None:
        other = self.here / "raw" / "s1_another_directory"
        other.mkdir(parents=True)
        self.launch(0)
        code, problems = self.mod.run(REAL_REPO, self.here, other,
                                      self.mod.FIRST_ATTEMPT_REQUEST_ID, self.plan, self.runtime)
        self.assertEqual(code, 4)
        self.assertEqual(self.launches, [])
        self.assertTrue(any("first failed attempt" in problem for problem in problems))
        self.assertEqual(list(other.glob("r75_s1_outer_v2_attempt_marker_*.json")), [])

    def test_a_new_request_id_is_not_refused_because_history_exists(self) -> None:
        """The old gate refused every later request once raw/s1 held a preserved failure."""
        self.launch(0)
        code, problems = self.mod.run(REAL_REPO, self.here, self.out, "R75_TEST_NEW_ID",
                                      self.plan, self.runtime)
        self.assertEqual((code, problems), (0, []))
        self.assertEqual(self.launches, [0])

    def test_a_tampered_first_attempt_is_refused(self) -> None:
        victim = self.first / "r75_s1_outer_attempt_marker.json"
        victim.write_bytes(victim.read_bytes() + b"tampered\n")
        code, problems = self.mod.run(REAL_REPO, self.here, self.out, "R75_TEST_TAMPER",
                                      self.plan, self.runtime)
        self.assertEqual(code, 3)
        self.assertEqual(self.launches, [])
        self.assertTrue(any("first attempt original changed" in problem for problem in problems))

    def test_a_missing_first_attempt_is_refused(self) -> None:
        for entry in sorted(self.first.iterdir()):
            entry.unlink()
        self.first.rmdir()
        code, problems = self.mod.run(REAL_REPO, self.here, self.out, "R75_TEST_MISSING",
                                      self.plan, self.runtime)
        self.assertEqual(code, 3)
        self.assertEqual(self.launches, [])
        self.assertTrue(any("raw/s1 is missing" in problem for problem in problems))

    def test_a_subprocess_failure_is_not_a_child_exit_code(self) -> None:
        def _explode(*_args, **_kwargs):
            raise OSError("no such interpreter")
        self.mod.launch_once = _explode
        with self.assertRaises(OSError):
            self.mod.run(REAL_REPO, self.here, self.out, "R75_TEST_SPAWN_FAIL",
                         self.plan, self.runtime)
        self.assertFalse((self.out / self.mod.OUTCOME_RECORD).exists(),
                         "a wrapper-side crash must never be written as a child outcome")

    def test_request_id_helper(self) -> None:
        parse = self.mod._request_id_from_argv
        self.assertEqual(parse(["--request-id", "R75_X"]), "R75_X")
        self.assertEqual(parse(["--request-id=R75_Y"]), "R75_Y")
        self.assertEqual(parse(["--launch"]), self.mod.FOLLOW_UP_REQUEST_ID)


    def test_the_child_receives_the_same_identity_and_directory(self) -> None:
        seen = {}

        def _launch(_repo, runner_path, argv, _timeout):
            seen["argv"] = list(argv)
            seen["runner"] = Path(runner_path)
            return fake_child(0)

        self.mod.launch_once = _launch
        code, problems = self.mod.run(REAL_REPO, self.here, self.out, "R75_TEST_ROUTE",
                                      self.plan, self.runtime)
        self.assertEqual((code, problems), (0, []))
        argv = seen["argv"]
        self.assertIn("--run-s1", argv)
        self.assertEqual(argv[argv.index("--request-id") + 1], "R75_TEST_ROUTE")
        self.assertEqual(Path(argv[argv.index("--attempt-dir") + 1]), self.out)
        self.assertEqual(seen["runner"], self.here / "r75_s1_runner.py")

    def test_the_marker_records_the_child_argv(self) -> None:
        self.launch(0)
        self.mod.run(REAL_REPO, self.here, self.out, "R75_TEST_MARKER", self.plan, self.runtime)
        markers = list(self.out.glob("r75_s1_outer_v2_attempt_marker_*.json"))
        self.assertEqual(len(markers), 1)
        recorded = json.loads(markers[0].read_text(encoding="utf-8"))
        self.assertIn("--request-id", recorded["argv"])
        self.assertIn("R75_TEST_MARKER", recorded["argv"])
        self.assertIn("--attempt-dir", recorded["argv"])
        self.assertIn(str(self.out), recorded["argv"])

    def test_two_request_ids_do_not_share_or_overwrite_a_directory(self) -> None:
        dir_a = self.here / "raw" / "s1_revised" / "R75_TEST_DIR_A"
        dir_b = self.here / "raw" / "s1_revised" / "R75_TEST_DIR_B"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)
        self.launch(0)
        code_a, problems_a = self.mod.run(REAL_REPO, self.here, dir_a, "R75_TEST_DIR_A",
                                          self.plan, self.runtime)
        code_b, problems_b = self.mod.run(REAL_REPO, self.here, dir_b, "R75_TEST_DIR_B",
                                          self.plan, self.runtime)
        self.assertEqual((code_a, problems_a), (0, []))
        self.assertEqual((code_b, problems_b), (0, []))
        names_a = sorted(p.name for p in dir_a.iterdir())
        names_b = sorted(p.name for p in dir_b.iterdir())
        self.assertIn("r75_s1_outer_v2_attempt_marker_R75_TEST_DIR_A.json", names_a)
        self.assertNotIn("r75_s1_outer_v2_attempt_marker_R75_TEST_DIR_B.json", names_a,
                         "one attempt directory must never hold another request id's marker")
        self.assertIn("r75_s1_outer_v2_attempt_marker_R75_TEST_DIR_B.json", names_b)
        self.assertNotIn("r75_s1_outer_v2_attempt_marker_R75_TEST_DIR_A.json", names_b)

    def test_prepare_creates_no_license_and_starts_no_child(self) -> None:
        attempt = self.here / "raw" / "s1_revised" / "R75_TEST_PREPARE_ONLY"
        attempt.mkdir(parents=True)
        payload, problems = self.mod.prepare(REAL_REPO, self.here, attempt,
                                             "R75_TEST_PREPARE_ONLY", self.plan, self.runtime)
        self.assertEqual(problems, [])
        self.assertFalse(payload["launched"])
        self.assertFalse(payload["attempt_marker_created"])
        self.assertEqual(self.launches, [], "prepare must never start a child")
        self.assertEqual(sorted(p.name for p in attempt.iterdir()), [self.mod.PREPARED_RECORD],
                         "prepare writes only the prepared record and no attempt license")


class WrapperMainGuardTest(Scratch):
    """The __main__ guard must let SystemExit through and never call it a wrapper crash."""

    def _guard(self, namespace, stub_name):
        """Run the real __main__ guard with a stubbed main(); return the wrapper-failure calls."""
        source = WRAPPER_SRC.read_text(encoding="utf-8")
        head, separator, tail = source.partition('if __name__ == "__main__":')
        self.assertTrue(separator, "the __main__ guard is missing")
        exec(compile(head, str(WRAPPER_SRC), "exec"), namespace)
        calls = []
        namespace["record_wrapper_failure"] = lambda *args, **kwargs: calls.append(args)
        namespace["_spy_calls"] = calls
        guard = separator + tail.replace("main(sys.argv[1:])", stub_name + "()")
        exec(compile(guard, str(WRAPPER_SRC), "exec"), namespace)
        return calls

    def test_systemexit_is_not_a_wrapper_crash(self) -> None:
        namespace = {"__name__": "__main__", "__file__": str(WRAPPER_SRC)}
        namespace["OUTPUT_DIR_NAME"] = str(self.tmp / "nowhere")

        def stub():
            raise SystemExit(3)

        namespace["_stub_main"] = stub
        with self.assertRaises(SystemExit) as caught:
            self._guard(namespace, "_stub_main")
        calls = namespace["_spy_calls"]
        self.assertEqual(caught.exception.code, 3)
        self.assertEqual(calls, [], "SystemExit is the exit transport, not a wrapper crash")

    def test_wrapper_exception_is_recorded_with_the_request_id(self) -> None:
        namespace = {"__name__": "__main__", "__file__": str(WRAPPER_SRC)}
        namespace["OUTPUT_DIR_NAME"] = str(self.tmp / "nowhere")
        saved_argv = list(sys.argv)
        sys.argv = ["r75_s1_outer_exec_v2.py", "--launch", "--request-id", "R75_TEST_GUARD_ID"]
        try:
            def stub():
                raise RuntimeError("wrapper side boom")

            namespace["_stub_main"] = stub
            with self.assertRaises(RuntimeError):
                self._guard(namespace, "_stub_main")
            calls = namespace["_spy_calls"]
        finally:
            sys.argv = saved_argv
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], "R75_TEST_GUARD_ID")


class WrapperRealPrepareTest(unittest.TestCase):
    """One genuine prepare against the real output root. No marker, no child, no launch."""

    def test_prepare_against_the_real_root_starts_nothing(self) -> None:
        module = load_wrapper()
        code = module.main(["--prepare", "--request-id", "R75_S0_IMPORT_BOOTSTRAP_REPAIR_20261015"])
        self.assertEqual(code, 0)
        here = HERE
        self.assertFalse((here / "raw" / "s1_revised" /
                          "r75_s1_outer_v2_attempt_marker_"
                          "R75_S0_IMPORT_BOOTSTRAP_REPAIR_20261015.json").exists())
        self.assertFalse((here / "raw" / "s1" / "r75_s1_outer_v2_stdout.bin").exists())


RUNNER_SRC = HERE / "r75_s1_runner.py"


def load_runner():
    """Import the real runner. Its module level is stdlib-only; the paid imports are inside."""
    spec = importlib.util.spec_from_file_location("r75_s1_runner_under_test", str(RUNNER_SRC))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def issue_fresh_license(target, request_id):
    """Register ONE real parent->child launch license into target exactly as the wrapper does.

    The marker is written by the REAL wrapper build_child_argv()+register_attempt(), so its argv is
    the genuine child vector the runner later parses; no child is started and nothing paid runs."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    wrapper = load_wrapper()
    argv = wrapper.build_child_argv(RUNNER_SRC,
                                    ["--run-s1", "--request-id", request_id,
                                     "--attempt-dir", str(target)])
    marker, marker_path = wrapper.register_attempt(target, request_id, argv,
                                                   "0" * 40, "1" * 40, "test-wrapper-sha")
    if marker is None:
        raise AssertionError("the fresh license marker already exists at %s" % marker_path)
    return marker_path


class RunnerWiringAstTest(Scratch):
    """The routing seam must exist in the real source, not only in the wrapper's dictionary."""

    def setUp(self) -> None:
        super().setUp()
        self.source = RUNNER_SRC.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_the_cli_accepts_the_execution_identity(self) -> None:
        flags = set()
        for node in ast.walk(self.tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument" and node.args
                    and isinstance(node.args[0], ast.Constant)):
                flags.add(node.args[0].value)
        self.assertIn("--request-id", flags)
        self.assertIn("--attempt-dir", flags)
        self.assertIn("--run-s1", flags)

    def test_the_identity_is_bound_before_the_paid_path(self) -> None:
        main = next(node for node in self.tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        named = sorted(((node.func.id, node.lineno) for node in ast.walk(main)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)),
                       key=lambda item: item[1])
        calls = [name for name, _line in named]
        self.assertIn("_bind_execution_identity", calls)
        self.assertIn("run_s1", calls)
        self.assertLess(calls.index("_bind_execution_identity"), calls.index("run_s1"),
                        "the identity must be bound before the paid path is entered")
        run = next(node for node in self.tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "run_s1")
        run_calls = [node.func.id for node in ast.walk(run)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertNotIn("_bind_execution_identity", run_calls,
                         "binding belongs to the CLI, ahead of the precheck and the paid imports")

    def test_the_records_keep_the_plan_identity_as_provenance(self) -> None:
        self.assertIn('"plan_request_id": PLAN["request_id"]', self.source)
        self.assertIn("EXECUTION_REQUEST_ID or PLAN[", self.source)
        self.assertIn('"attempt_dir": str(S1_DIR)', self.source)

    def test_the_central_ledger_is_not_duplicated(self) -> None:
        self.assertIn("r75_guard.R75Resources.production()", self.source)
        self.assertNotIn('S1_DIR / "runtime"', self.source)
        self.assertNotIn('S1_DIR / "raw"', self.source)
        self.assertIn('S1_DIR = HERE / "raw" / "s1"', self.source,
                      "the historical directory stays the default, so the originals keep their place")

    def test_the_historical_default_is_still_the_preserved_directory(self) -> None:
        module = load_runner()
        self.assertEqual(module.S1_DIR, HERE / "raw" / "s1")
        self.assertEqual(module.FIRST_ATTEMPT_REQUEST_ID,
                         "R75_S1_NATIVE_QUALIFICATION_20261015")
        self.assertIsNone(module.EXECUTION_REQUEST_ID)


class RunnerIdentityRoutingTest(Scratch):
    """Drive the real binding function and the real CLI. Nothing paid is ever started."""

    def setUp(self) -> None:
        super().setUp()
        self.module = load_runner()
        self.historical = sorted(path.name for path in (HERE / "raw" / "s1").iterdir())

    def test_a_command_without_an_identity_is_refused(self) -> None:
        self.assertEqual(self.module.main(["--run-s1"]), 2)
        self.assertEqual(self.module.main(["--run-s1", "--request-id", "R75_TEST_ONLY_ID"]), 2)
        self.assertEqual(self.module.main(["--run-s1", "--attempt-dir", str(self.tmp)]), 2)
        self.assertIsNone(self.module.EXECUTION_REQUEST_ID)
        self.assertEqual(self.module.S1_DIR, HERE / "raw" / "s1")
        self.assertEqual(sorted(path.name for path in (HERE / "raw" / "s1").iterdir()),
                         self.historical, "a refused command must not touch the preserved originals")

    def test_the_historical_request_id_is_refused_in_any_directory(self) -> None:
        target = self.tmp / "raw" / "s1_somewhere_else"
        with self.assertRaises(RuntimeError) as caught:
            self.module._bind_execution_identity(self.module.FIRST_ATTEMPT_REQUEST_ID, str(target))
        self.assertIn("can never run again", str(caught.exception))
        self.assertIsNone(self.module.EXECUTION_REQUEST_ID)

    def test_the_historical_directory_and_its_children_are_refused(self) -> None:
        for candidate in (HERE / "raw" / "s1", HERE / "raw" / "s1" / "nested"):
            with self.assertRaises(RuntimeError) as caught:
                self.module._bind_execution_identity("R75_TEST_ROUTE", str(candidate))
            self.assertIn("raw/s1", str(caught.exception))

    def test_a_directory_outside_the_output_root_is_refused(self) -> None:
        outside = HERE.parent / "_r75_outside_probe"
        with self.assertRaises(RuntimeError) as caught:
            self.module._bind_execution_identity("R75_TEST_ROUTE", str(outside))
        self.assertIn("output root", str(caught.exception))
        self.assertIsNone(self.module.EXECUTION_REQUEST_ID)

    def test_a_valid_identity_binds_the_directory(self) -> None:
        target = self.tmp / "raw" / "s1_independent"
        issue_fresh_license(target, "R75_TEST_ROUTE")
        bound = self.module._bind_execution_identity("R75_TEST_ROUTE", str(target))
        self.assertEqual(bound, target)
        self.assertEqual(self.module.S1_DIR, target)
        self.assertEqual(self.module.EXECUTION_REQUEST_ID, "R75_TEST_ROUTE")
        self.assertTrue((target / "r75_s1_outer_v2_license_consumed_R75_TEST_ROUTE.json").is_file(),
                        "binding must leave the non-repeatable consumption evidence")
        self.assertEqual(sorted(path.name for path in (HERE / "raw" / "s1").iterdir()),
                         self.historical)

    def test_a_repeated_request_id_in_one_directory_is_refused(self) -> None:
        target = self.tmp / "raw" / "s1_independent"
        issue_fresh_license(target, "R75_TEST_ROUTE")
        self.module._bind_execution_identity("R75_TEST_ROUTE", str(target))
        with self.assertRaises(RuntimeError) as caught:
            self.module._bind_execution_identity("R75_TEST_ROUTE", str(target))
        self.assertIn("not a fresh, unconsumed", str(caught.exception))

    def test_the_child_directory_agrees_with_the_outer_record(self) -> None:
        out = self.tmp / "raw" / "s1_revised"
        out.mkdir(parents=True)
        marker = issue_fresh_license(out, "R75_TEST_AGREE")
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        self.assertEqual(recorded["license_state"], "issued")
        bound = self.module._bind_execution_identity("R75_TEST_AGREE", str(out))
        self.assertEqual(bound, out)
        self.assertEqual(self.module.S1_DIR, out)
        self.assertEqual(self.module.EXECUTION_REQUEST_ID, "R75_TEST_AGREE")

class ParentChildLicenseSeamTest(Scratch):
    """The real wrapper registration bridged to the real runner binding, with the paid path spied.

    A fresh parent license reaches the spied run_s1() exactly once; a second child, a cross-directory
    replay, the historical request id and an already-consumed license file are each refused. No torch,
    no model, no environment, no real subprocess: run_s1 and Popen are replaced by spies."""

    def setUp(self) -> None:
        super().setUp()
        self.wrapper = load_wrapper()
        self.module = load_runner()
        self.reached = []
        self.module.run_s1 = lambda: (self.reached.append(1) or 0)

    def child_argv(self, request_id, target):
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        return self.wrapper.build_child_argv(RUNNER_SRC,
                                             ["--run-s1", "--request-id", request_id,
                                              "--attempt-dir", str(target)])

    def test_the_real_child_argv_reaches_the_spied_paid_path_exactly_once(self) -> None:
        target = self.tmp / "raw" / "s1_bridge"
        argv = self.child_argv("R75_TEST_BRIDGE", target)
        self.assertEqual(argv[0], sys.executable)
        self.assertEqual(argv[1], str(RUNNER_SRC))
        self.assertEqual(argv.count(sys.executable), 1, "the interpreter must appear exactly once")
        self.assertEqual(argv.count(str(RUNNER_SRC)), 1, "the script must appear exactly once")
        marker, _ = self.wrapper.register_attempt(target, "R75_TEST_BRIDGE", argv,
                                                  "0" * 40, "1" * 40, "test-wrapper-sha")
        self.assertIsNotNone(marker)
        code = self.module.main(argv[2:])
        self.assertEqual(code, 0)
        self.assertEqual(self.reached, [1], "one fresh license reaches the paid path exactly once")
        self.assertTrue((target / "r75_s1_outer_v2_license_consumed_R75_TEST_BRIDGE.json").is_file())

    def test_a_second_child_cannot_reuse_the_same_license(self) -> None:
        target = self.tmp / "raw" / "s1_bridge"
        argv = self.child_argv("R75_TEST_SECOND", target)
        self.wrapper.register_attempt(target, "R75_TEST_SECOND", argv,
                                      "0" * 40, "1" * 40, "test-wrapper-sha")
        self.assertEqual(self.module.main(argv[2:]), 0)
        self.assertEqual(self.reached, [1])
        self.assertEqual(self.module.main(argv[2:]), 2, "the second child is refused")
        self.assertEqual(self.reached, [1], "a refused second child never reaches the paid path")

    def test_a_license_issued_for_another_directory_is_refused(self) -> None:
        issued = self.tmp / "raw" / "s1_issued"
        replay = self.tmp / "raw" / "s1_replay"
        argv = self.child_argv("R75_TEST_REPLAY", issued)
        self.wrapper.register_attempt(issued, "R75_TEST_REPLAY", argv,
                                      "0" * 40, "1" * 40, "test-wrapper-sha")
        self.assertEqual(self.module.main(["--run-s1", "--request-id", "R75_TEST_REPLAY",
                                           "--attempt-dir", str(replay)]), 2)
        self.assertEqual(self.reached, [], "a cross-directory replay never reaches the paid path")

    def test_the_historical_request_id_is_refused_even_with_a_marker_present(self) -> None:
        target = self.tmp / "raw" / "s1_hist"
        argv = self.child_argv(self.module.FIRST_ATTEMPT_REQUEST_ID, target)
        self.wrapper.register_attempt(target, self.module.FIRST_ATTEMPT_REQUEST_ID, argv,
                                      "0" * 40, "1" * 40, "test-wrapper-sha")
        self.assertEqual(self.module.main(argv[2:]), 2)
        self.assertEqual(self.reached, [])

    def test_an_already_consumed_license_file_is_refused(self) -> None:
        target = self.tmp / "raw" / "s1_preconsumed"
        argv = self.child_argv("R75_TEST_PRECONSUMED", target)
        self.wrapper.register_attempt(target, "R75_TEST_PRECONSUMED", argv,
                                      "0" * 40, "1" * 40, "test-wrapper-sha")
        consumed = target / "r75_s1_outer_v2_license_consumed_R75_TEST_PRECONSUMED.json"
        fd = os.open(str(consumed), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("{}")
        with self.assertRaises(RuntimeError) as caught:
            self.module._bind_execution_identity("R75_TEST_PRECONSUMED", str(target))
        self.assertIn("already consumed", str(caught.exception))
        self.assertEqual(self.reached, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
