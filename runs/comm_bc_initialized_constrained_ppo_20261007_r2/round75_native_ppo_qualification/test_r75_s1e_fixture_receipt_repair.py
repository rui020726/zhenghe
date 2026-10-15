# -*- coding: utf-8 -*-
"""S1E fixture/receipt repair A gates. Pure stdlib; no NumPy/Torch import."""
from __future__ import annotations

import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT_SCRIPT = HERE / "r75_s1e_parent_capture.ps1"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("r75_s1_runner_s1e_test", "r75_s1_runner.py")
probe = _load("r75_s1e_probe_test", "r75_s1e_fixture_receipt_repair.py")


class Payload:
    def __init__(self, data):
        self.data = bytes(data)

    def tobytes(self):
        return self.data


class FakeTensor:
    def __init__(self, data, dtype="torch.uint8", shape=None, device="cpu"):
        self.payload = Payload(data)
        self.dtype = dtype
        self.shape = tuple(shape) if shape is not None else (len(self.payload.data),)
        self.device = device

    def cpu(self):
        return self

    def clone(self):
        return FakeTensor(self.payload.data, self.dtype, self.shape, self.device)

    def __getitem__(self, index):
        if index == 0:
            return self.payload.data[0]
        raise IndexError

    def __setitem__(self, index, value):
        if index != 0:
            raise IndexError
        self.payload = Payload(bytes([value]) + self.payload.data[1:])


class FakeKeys:
    def __init__(self, values, dtype="numpy.uint32", shape=None):
        self.values = tuple(values)
        self.dtype = dtype
        self.shape = shape or (len(self.values),)

    def __iter__(self):
        return iter(self.values)

    def reshape(self, shape):
        if tuple(shape) != tuple(self.shape):
            raise ValueError("fake reshape shape mismatch")
        return self

    def tobytes(self):
        return b"".join(bytes([value & 0xff]) for value in self.values)


class FakeNumpyModule:
    def __init__(self):
        self.state = ("MT19937", FakeKeys((3, 1, 4, 1, 5)), 5, 0, 0.0)
        self.__version__ = "fake-1"

    def random_state(self):
        return self.state

    class random:
        @staticmethod
        def get_state():
            return ("MT19937", FakeKeys((3, 1, 4, 1, 5)), 5, 0, 0.0)

    def asarray(self, values, dtype):
        return FakeKeys(tuple(values), dtype=dtype)


class FakeTorchModule:
    def __init__(self):
        self.state = FakeTensor(b"torch-rng")
        self.last_saved = None
        self.__version__ = "fake-2"
        self.int16 = "torch.int16"
        self.int32 = "numpy.int32"

    def get_rng_state(self):
        return self.state.clone()

    def save(self, obj, file_obj):
        self.last_saved = obj
        if hasattr(file_obj, "write"):
            file_obj.write(b"fake-serialized")
        else:
            Path(file_obj).write_bytes(b"fake-serialized")

    def load(self, path, map_location, weights_only):
        if map_location != "cpu" or weights_only is not True:
            raise AssertionError("load must use map_location=cpu and weights_only=True")
        return self.last_saved


class State:
    record = {"ok": True}
    unread: list[str] = []

    def counter_report(self):
        return {"non_neural_fixture": {"counted": 1}}


def identity_fn(head="s1e-test-head"):
    return lambda: {"head": head, "parent_head": head + "-parent", "branch": "test",
                    "source": "fake", "errors": {}}


class Scratch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="r75_s1e_test_", dir=str(HERE)))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def patch_output_root(self):
        original = probe.OUTPUT_ROOT
        probe.OUTPUT_ROOT = self.tmp
        self.addCleanup(setattr, probe, "OUTPUT_ROOT", original)


class CodecTests(Scratch):
    def test_numpy_codec_preserves_complete_identity(self):
        state = ("MT19937", FakeKeys((9, 2, 6), dtype="numpy.uint32", shape=(3,)), 7, 1, 2.5)
        encoded = probe.encode_numpy_mt19937_state(state)
        self.assertEqual(encoded["algorithm"], "MT19937")
        self.assertEqual(encoded["keys_dtype"], "numpy.uint32")
        self.assertEqual(encoded["keys_shape"], [3])
        self.assertEqual(encoded["keys"], [9, 2, 6])
        self.assertEqual(encoded["position"], 7)
        self.assertEqual(encoded["has_gauss"], 1)
        self.assertEqual(encoded["cached_gaussian"], 2.5)

    def test_numpy_codec_output_is_supported(self):
        encoded = probe.encode_numpy_mt19937_state(FakeNumpyModule().state)
        probe._assert_supported_payload(encoded, "encoded")

    def test_decode_rebuilds_without_set_state_or_seed(self):
        np_mod = FakeNumpyModule()
        encoded = probe.encode_numpy_mt19937_state(np_mod.random.get_state())
        rebuilt = probe.decode_numpy_mt19937_state(encoded, np_mod)
        self.assertEqual(rebuilt[0], "MT19937")
        self.assertEqual(rebuilt[1].values, (3, 1, 4, 1, 5))
        self.assertEqual(rebuilt[2], 5)
        self.assertEqual(rebuilt[4], 0.0)

    def test_numpy_module_object_is_rejected(self):
        class FakeNumpyArray:
            dtype = "numpy.uint32"
            shape = (1,)
            def tobytes(self):
                return b"x"
        FakeNumpyArray.__module__ = "numpy.ndarray"
        with self.assertRaises(TypeError):
            probe._assert_supported_payload([FakeNumpyArray()], "array")

    def test_path_and_custom_objects_are_rejected(self):
        with self.assertRaises(TypeError):
            probe._assert_supported_payload({"path": self.tmp}, "path")

    def test_numpy_scalar_subclasses_are_rejected_before_python_builtins(self):
        class FakeNumpyFloat(float):
            pass
        FakeNumpyFloat.__module__ = "numpy.float64"
        with self.assertRaises(TypeError):
            probe._assert_supported_payload({"value": FakeNumpyFloat(1.0)}, "scalar")

    def test_stage_fixture_keeps_torch_and_removes_numpy_objects(self):
        state = probe.build_stage_full_state(FakeNumpyModule(), FakeTorchModule())
        probe._assert_supported_payload(state, "fixture")
        self.assertTrue(hasattr(state["torch_rng_state"], "dtype"))
        self.assertIn("numpy_rng_state", state)
        self.assertNotIsInstance(state["numpy_rng_state"], FakeKeys)

    def test_supported_payload_contract_covers_scalars_containers_and_tensor(self):
        value = {"a": [1, 2.0, None, True], "b": (b"raw", "text"),
                 "c": FakeTensor(b"bytes"), "d": {"nested": []}}
        probe._assert_supported_payload(value, "contract")


class SaveAndStageTests(Scratch):
    def test_default_save_uses_torch_save_object_file_abi(self):
        fake_torch = FakeTorchModule()
        original = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            identity = runner._save_stage_snapshot(
                self.tmp / "default.pt", stage="test", label="abi", state=State(),
                full_state={"x": 1, "tensor": FakeTensor(b"z")}, identity_fn=identity_fn("test-head"))
        finally:
            if original is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original
        self.assertIsNotNone(fake_torch.last_saved)
        self.assertEqual(identity["execution_head"], "test-head")
        self.assertTrue((self.tmp / "default.pt").is_file())

    def test_failed_default_save_leaves_no_target_or_temporary(self):
        fake_torch = FakeTorchModule()
        def fail(obj, file_obj):
            if hasattr(file_obj, "write"):
                file_obj.write(b"partial")
            else:
                Path(file_obj).write_bytes(b"partial")
            raise RuntimeError("injected")
        fake_torch.save = fail
        original = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            with self.assertRaises(RuntimeError):
                runner._save_stage_snapshot(self.tmp / "failed.pt", stage="test", label="fail",
                                            state=State(), full_state={"x": 1},
                                            identity_fn=identity_fn("test-head"))
        finally:
            if original is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original
        self.assertFalse((self.tmp / "failed.pt").exists())
        self.assertFalse((self.tmp / "failed.pt.tmp").exists())

    def test_fake_weights_only_stage_case_saves_and_loads_once(self):
        self.patch_output_root()
        fake_torch = FakeTorchModule()
        original = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        runner.EXECUTION_REQUEST_ID = probe.REQUEST_ID
        counters = {key: 0 for key in ("fixture_save_attempt", "fixture_save_completed",
                                       "fixture_load_attempt", "fixture_load_completed",
                                       "fixture_read_attempt", "fixture_read_completed")}
        try:
            result = probe._stage_case(runner, FakeNumpyModule(), fake_torch, counters)
        finally:
            if original is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original
        runner.EXECUTION_REQUEST_ID = None
        self.assertTrue(result["passed"])
        self.assertEqual(result["got_type"], "bool")
        self.assertEqual(result["weights_only"], True)
        self.assertEqual(counters["fixture_save_attempt"], 1)
        self.assertEqual(counters["fixture_save_completed"], 1)
        self.assertEqual(counters["fixture_load_completed"], 1)
        self.assertEqual(counters["fixture_read_completed"], 1)

    def test_stage_case_refuses_existing_fixture(self):
        self.patch_output_root()
        fixture = probe.OUTPUT_ROOT / "non_neural_fixture" / "rng_fixture_stage.pt"
        fixture.parent.mkdir(parents=True)
        fixture.write_bytes(b"existing")
        counters = {key: 0 for key in ("fixture_save_attempt", "fixture_save_completed",
                                       "fixture_load_attempt", "fixture_load_completed",
                                       "fixture_read_attempt", "fixture_read_completed")}
        result = probe._stage_case(runner, FakeNumpyModule(), FakeTorchModule(), counters)
        self.assertFalse(result["passed"])

    def test_stage_case_handles_load_failure_and_counts_it(self):
        self.patch_output_root()
        fake_torch = FakeTorchModule()
        def fail_load(path, map_location, weights_only):
            raise RuntimeError("weights-only fixture rejected")
        fake_torch.load = fail_load
        original = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        counters = {key: 0 for key in ("fixture_save_attempt", "fixture_save_completed",
                                       "fixture_load_attempt", "fixture_load_completed",
                                       "fixture_read_attempt", "fixture_read_completed")}
        try:
            with self.assertRaises(RuntimeError):
                probe._stage_case(runner, FakeNumpyModule(), fake_torch, counters)
        finally:
            if original is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original
        self.assertEqual(counters["fixture_load_attempt"], 1)
        self.assertEqual(counters["fixture_load_completed"], 0)


class ReceiptTests(Scratch):
    def make_recorder(self):
        return probe.CaseRecorder(self.tmp / "cases.jsonl")

    def test_recorder_appends_each_case_immediately(self):
        recorder = self.make_recorder()
        first = recorder.append({"case_id": 1, "passed": True})
        self.assertTrue((self.tmp / "cases.jsonl").is_file())
        second = recorder.append({"case_id": 2, "passed": False})
        lines = (self.tmp / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["case_id"], 1)
        self.assertEqual(json.loads(lines[1])["case_id"], 2)
        recorder.close()
        self.assertEqual([first, second][0]["case_id"], 1)

    def test_run_case_success_is_durable_and_counts_completed(self):
        counters = {"case_attempt": 0, "case_completed": 0, "case_failed": 0}
        recorder = self.make_recorder()
        fake_runner = type("R", (), {"_rng_identity_full": staticmethod(lambda np, torch: {"ok": True})})()
        record = probe.run_case(1, "ok", "direct", lambda: {"passed": True}, recorder,
                                counters, fake_runner, None, None)
        recorder.close()
        self.assertTrue(record["passed"])
        self.assertEqual(record["rng_after"]["status"], "present")
        self.assertEqual(counters["case_completed"], 1)

    def test_run_case_exception_is_recorded_with_type_and_message(self):
        counters = {"case_attempt": 0, "case_completed": 0, "case_failed": 0}
        recorder = self.make_recorder()
        fake_runner = type("R", (), {"_rng_identity_full": staticmethod(lambda np, torch: {"ok": True})})()
        def fail():
            raise ValueError("case exploded")
        record = probe.run_case(2, "fail", "stage_save", fail, recorder, counters,
                                fake_runner, None, None)
        recorder.close()
        self.assertFalse(record["passed"])
        self.assertEqual(record["exception"]["type"], "ValueError")
        self.assertIn("case exploded", record["exception"]["message"])
        self.assertEqual(counters["case_failed"], 1)

    def test_run_case_marks_missing_rng_without_fabricating_after(self):
        counters = {"case_attempt": 0, "case_completed": 0, "case_failed": 0}
        recorder = self.make_recorder()
        def bad_rng(np, torch):
            raise OSError("rng unavailable")
        fake_runner = type("R", (), {"_rng_identity_full": staticmethod(bad_rng)})()
        record = probe.run_case(3, "rng-missing", "direct", lambda: {"passed": True},
                                recorder, counters, fake_runner, None, None)
        recorder.close()
        self.assertEqual(record["rng_after"]["status"], "missing")
        self.assertEqual(record["rng_after"]["error"]["type"], "OSError")

    def test_run_case_rejects_non_bool_passed(self):
        counters = {"case_attempt": 0, "case_completed": 0, "case_failed": 0}
        recorder = self.make_recorder()
        fake_runner = type("R", (), {"_rng_identity_full": staticmethod(lambda np, torch: {"ok": True})})()
        record = probe.run_case(4, "bad-type", "direct", lambda: {"passed": "yes"},
                                recorder, counters, fake_runner, None, None)
        recorder.close()
        self.assertFalse(record["passed"])

    def test_result_writer_is_atomic_json(self):
        original = probe.RESULT_PATH
        probe.RESULT_PATH = self.tmp / "result.json"
        try:
            probe._write_result({"passed": True})
            payload = json.loads((self.tmp / "result.json").read_text(encoding="utf-8"))
        finally:
            probe.RESULT_PATH = original
        self.assertTrue(payload["passed"])
        self.assertFalse((self.tmp / "result.json.tmp").exists())


class ParentAndConsumptionTests(Scratch):
    def test_consumption_is_o_excl_once(self):
        path = self.tmp / "consumed.json"
        self.assertTrue(probe.consume_once(path, {"attempt": 1}))
        self.assertFalse(probe.consume_once(path, {"attempt": 2}))
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["attempt"], 1)

    def test_main_without_run_b_does_not_write_outputs(self):
        self.patch_output_root()
        self.assertEqual(probe.main([]), 2)
        self.assertFalse((self.tmp / "phase_b_consumed.json").exists())

    def test_parent_uses_powershell_start_process_hidden_once(self):
        text = PARENT_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("Start-Process", text)
        self.assertIn("-WindowStyle Hidden", text)
        self.assertIn("-PassThru", text)
        self.assertNotIn("spawnSync", text)
        self.assertIn("refuses a second launch", text)

    def test_parent_captures_stdio_as_binary_file_hashes(self):
        text = PARENT_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("-RedirectStandardOutput", text)
        self.assertIn("-RedirectStandardError", text)
        self.assertIn("Get-FileHash -Algorithm SHA256", text)
        self.assertNotIn("StreamReader", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
