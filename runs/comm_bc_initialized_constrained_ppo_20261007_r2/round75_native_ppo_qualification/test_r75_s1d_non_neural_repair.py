# -*- coding: utf-8 -*-
"""ROUND75 S1D non-neural repair counterexamples. Pure stdlib; no NumPy/Torch import."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import struct
import random
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("r75_s1_runner_s1d_test", "r75_s1_runner.py")
probe = _load("r75_s1d_non_neural_probe_test", "r75_s1d_non_neural_probe.py")


class Scratch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="r75_s1d_test_", dir=str(HERE)))
        self.addCleanup(shutil.rmtree, self.tmp, True)


class FakePayload:
    def __init__(self, data, dtype="torch.uint8", shape=None):
        self.data = bytes(data)
        self.dtype = dtype
        self.shape = tuple(shape) if shape is not None else (len(self.data),)

    def tobytes(self):
        return self.data


class FakeTorchTensor:
    def __init__(self, data, dtype="torch.uint8", shape=None):
        self.payload = FakePayload(data, dtype, shape)
        self.dtype = dtype
        self.shape = tuple(shape) if shape is not None else (len(self.payload.data),)

    def numpy(self):
        return self.payload


class VectorEqualityObject:
    def __eq__(self, other):
        return [False, False]

    def __bool__(self):
        raise AssertionError("unknown objects must not be reduced to bool")


class RandomNamespace:
    def __init__(self, state):
        self.state = state

    def get_state(self):
        return self.state


class FakeNumpy:
    def __init__(self, state=None):
        self.state = state or ("MT19937", FakePayload(b"keys", "numpy.uint32", (5,)), 7, 0, 0.0)

    class random:
        @staticmethod
        def get_state():
            return ("MT19937", FakePayload(b"keys", "numpy.uint32", (5,)), 7, 0, 0.0)


class FakeTorchModule:
    def __init__(self, state=None):
        self.state = FakeTorchTensor(b"torch") if state is None else state

    def get_rng_state(self):
        return self.state


class State:
    def __init__(self):
        self.record = {"A1_gate": {"passed": True}}
        self.unread = ["optional"]

    def counter_report(self):
        return {"physical_optimizer_step": {"counted": 0}}


def write_identity(path, head="actual-head"):
    return lambda: {"head": head, "parent_head": head + "-parent", "branch": "test",
                    "source": "fake", "errors": {}}


class ComparatorTests(Scratch):
    def test_torch_like_without_tobytes_compares_full_bytes(self):
        live = FakeTorchTensor(b"abcd")
        same = FakeTorchTensor(b"abcd")
        different = FakeTorchTensor(b"abce")
        self.assertIsInstance(runner._raw_equal(live, same), bool)
        self.assertTrue(runner._raw_equal(live, same))
        self.assertFalse(runner._raw_equal(live, different))
        detail = runner._first_raw_difference(live, different, "torch_like")
        self.assertEqual(detail["first_diff_byte"], 3)

    def test_mixed_payload_with_torch_like_is_equal(self):
        live = {"torch": FakeTorchTensor(b"abcd"), "python": (1, 2, None)}
        same = {"torch": FakeTorchTensor(b"abcd"), "python": (1, 2, None)}
        self.assertIsInstance(runner._raw_equal(live, same), bool)
        self.assertTrue(runner._raw_equal(live, same))

    def test_tensor_like_type_dtype_shape_are_strict(self):
        live = FakeTorchTensor(b"abcd")
        self.assertFalse(runner._raw_equal(live, FakePayload(b"abcd")))
        self.assertFalse(runner._raw_equal(live, FakeTorchTensor(b"abcd", "torch.int8")))
        self.assertFalse(runner._raw_equal(live, FakeTorchTensor(b"abcd", shape=(2, 2))))

    def test_rng_payload_differences_handle_torch_like(self):
        np_mod = FakeNumpy()
        torch_mod = FakeTorchModule()
        saved = {"_numpy_state": np_mod.random.get_state(),
                 "_torch_state": FakeTorchTensor(b"torch"),
                 "_python_state": random.getstate()}
        self.assertEqual(runner._rng_payload_differences(saved, np_mod, torch_mod), [])
        torch_mod.state = FakeTorchTensor(b"torcm")
        problems = runner._rng_payload_differences(saved, np_mod, torch_mod)
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0]["where"], "rng_payload:torch_rng_state")

    def test_unknown_vector_object_is_safe(self):
        left = VectorEqualityObject()
        self.assertIs(runner._raw_equal(left, VectorEqualityObject()), False)
        self.assertIs(runner._raw_equal(left, left), True)

    def test_nested_containers_and_scalars_remain_strict(self):
        self.assertTrue(runner._raw_equal((1, (b"x", None)), (1, (b"x", None))))
        self.assertFalse(runner._raw_equal((1,), [1]))
        self.assertFalse(runner._raw_equal(1, True))


class RngIdentityTests(Scratch):
    def test_numpy_position_and_cache_are_recorded(self):
        first = runner._rng_identity_full(FakeNumpy(), FakeTorchModule())
        changed_state = ("MT19937", FakePayload(b"keys", "numpy.uint32", (5,)), 8, 1, 0.25)
        changed_np = FakeNumpy()
        changed_np.random.get_state = lambda: changed_state
        second = runner._rng_identity_full(changed_np, FakeTorchModule())
        self.assertNotEqual(first, second)
        self.assertEqual(first["numpy"]["position"], 7)
        self.assertEqual(second["numpy"]["position"], 8)
        self.assertEqual(second["numpy"]["has_gauss"], 1)
        self.assertEqual(second["numpy"]["cached_gaussian"], 0.25)

    def test_complete_identity_includes_keys_python_and_torch(self):
        identity = runner._rng_identity_full(FakeNumpy(), FakeTorchModule())
        self.assertEqual(identity["numpy"]["keys"]["bytes_len"], 4)
        self.assertEqual(identity["numpy"]["keys"]["values_sha256"],
                         hashlib.sha256(b"keys").hexdigest())
        self.assertEqual(identity["torch"]["values_sha256"],
                         hashlib.sha256(b"torch").hexdigest())
        self.assertIn("repr_sha256", identity["python"])


class StageSaveTests(Scratch):
    def test_default_save_fn_uses_torch_save_obj_file_abi(self):
        calls = []
        class FakeTorch:
            @staticmethod
            def save(obj, file_obj):
                calls.append((type(obj).__name__, type(file_obj).__name__))
                file_obj.write_text(json.dumps(obj), encoding="utf-8")
        original = runner.sys.modules.get("torch")
        runner.sys.modules["torch"] = FakeTorch
        try:
            runner._save_stage_snapshot(self.tmp / "stage.pt", stage="A2", label="A2",
                                        state=State(), full_state={"x": 1},
                                        identity_fn=write_identity(self.tmp))
        finally:
            if original is None:
                runner.sys.modules.pop("torch", None)
            else:
                runner.sys.modules["torch"] = original
        self.assertEqual(calls, [("dict", "WindowsPath")])

    def test_injected_save_fn_uses_same_obj_file_abi(self):
        calls = []
        def save_fn(obj, file_obj):
            calls.append((obj, file_obj))
            file_obj.write_text(json.dumps(obj), encoding="utf-8")
        runner._save_stage_snapshot(self.tmp / "stage.pt", stage="A2", label="A2",
                                    state=State(), full_state={"x": 1}, save_fn=save_fn,
                                    identity_fn=write_identity(self.tmp))
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0][0], dict)
        self.assertIsInstance(calls[0][1], Path)

    def test_stage_snapshot_is_atomic_and_cites_identity(self):
        path = self.tmp / "stage.pt"
        identity_fn = write_identity(self.tmp, "actual-123")
        identity = runner._save_stage_snapshot(path, stage="A2_completed", label="A2",
                                               state=State(), full_state={"x": 1},
                                               save_fn=lambda obj, file_obj: file_obj.write_text(
                                                   json.dumps(obj), encoding="utf-8"),
                                               identity_fn=identity_fn)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["execution_identity"]["head"], "actual-123")
        self.assertNotEqual(payload["plan_frozen_head"], "actual-123")
        self.assertTrue(payload["plan_frozen_head_is_not_execution_head"])
        self.assertEqual(identity["execution_head"], "actual-123")
        self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_data_identity_is_explicit_present_or_missing(self):
        present = runner._save_stage_snapshot(self.tmp / "present.pt", stage="A2", label="A2",
                                              state=State(),
                                              full_state={"data_identity": {"row": "sha"}},
                                              save_fn=lambda obj, file_obj: file_obj.write_text(
                                                  json.dumps(obj), encoding="utf-8"),
                                              identity_fn=write_identity(self.tmp))
        missing = runner._save_stage_snapshot(self.tmp / "missing.pt", stage="A2", label="A2",
                                              state=State(), full_state={"x": 1},
                                              save_fn=lambda obj, file_obj: file_obj.write_text(
                                                  json.dumps(obj), encoding="utf-8"),
                                              identity_fn=write_identity(self.tmp))
        self.assertTrue(present["data_identity_present"])
        self.assertFalse(missing["data_identity_present"])
        self.assertEqual(json.loads((self.tmp / "missing.pt").read_text(encoding="utf-8"))
                         ["data_identity"], "missing")

    def test_failed_save_leaves_no_temporary_or_target(self):
        path = self.tmp / "failed.pt"
        def save_fn(obj, file_obj):
            file_obj.write_text("partial", encoding="utf-8")
            raise RuntimeError("injected")
        with self.assertRaises(RuntimeError):
            runner._save_stage_snapshot(path, stage="A2", label="A2", state=State(),
                                        full_state={"x": 1}, save_fn=save_fn,
                                        identity_fn=write_identity(self.tmp))
        self.assertFalse(path.exists())
        self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_execution_identity_helper_uses_git_rev_parse(self):
        calls = []
        class Completed:
            def __init__(self, out, code=0):
                self.stdout = out
                self.stderr = b""
                self.returncode = code
        def fake_git(argv, capture_output=True):
            calls.append(argv)
            if "rev-parse" in argv and argv[-1] == "HEAD":
                return Completed(b"actual-head\n")
            if "rev-parse" in argv and argv[-1] == "HEAD^":
                return Completed(b"actual-parent\n")
            if any("abbrev-ref" in item for item in argv):
                return Completed(b"branch\n")
            raise AssertionError("unexpected git argv: " + repr(argv))
        original = runner.subprocess.run
        runner.subprocess.run = fake_git
        try:
            identity = runner._execution_identity()
        finally:
            runner.subprocess.run = original
        self.assertEqual(identity["head"], "actual-head")
        self.assertEqual(identity["parent_head"], "actual-parent")
        self.assertEqual(identity["branch"], "branch")
        self.assertEqual(len(calls), 3)


class ProbeHelperTests(Scratch):
    def test_capture_process_retains_actual_stdio_bytes(self):
        class Completed:
            stdout = b"actual-out"
            stderr = b"actual-err"
            returncode = 2
        captured = probe.capture_process(["fake", "argv"], runner=lambda argv, **kwargs: Completed())
        self.assertEqual(captured["returncode"], 2)
        self.assertEqual(captured["stdout_bytes"], b"actual-out")
        self.assertEqual(captured["stderr_bytes"], b"actual-err")
        self.assertEqual(captured["stdout_sha256"], hashlib.sha256(b"actual-out").hexdigest())

    def test_consumption_record_is_o_excl_and_separate(self):
        path = self.tmp / "consumed.json"
        self.assertTrue(probe.consume_once(path, {"attempt": 1}))
        self.assertFalse(probe.consume_once(path, {"attempt": 2}))
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["attempt"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
