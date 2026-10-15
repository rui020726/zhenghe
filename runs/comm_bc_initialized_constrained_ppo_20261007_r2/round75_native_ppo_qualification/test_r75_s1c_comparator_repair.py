# -*- coding: utf-8 -*-
"""ROUND75 S1C comparator and ledger reconciliation counterexamples.

Pure stdlib. This file does not import numpy, torch, r75_frozen, r75_guard, r75_native, the
replay adapter or canonical production material. Real model files are never loaded.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("r75_s1_runner_s1c_test", "r75_s1_runner.py")
wrapper = _load("r75_s1_outer_exec_v2_s1c_test", "r75_s1_outer_exec_v2.py")


class Scratch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="r75_s1c_test_", dir=str(HERE)))
        self.addCleanup(shutil.rmtree, self.tmp, True)


class FakePayload:
    def __init__(self, data, dtype="torch.uint8", shape=None):
        self.data = bytes(data)
        self.dtype = dtype
        self.shape = tuple(shape) if shape is not None else (len(self.data),)

    def tobytes(self):
        return self.data


class VectorEqualityObject:
    def __eq__(self, other):
        return [False, False, False]

    def __bool__(self):
        raise AssertionError("the comparator must not reduce this object to a bool")


class FakeTorchTensor:
    def __init__(self, data, dtype="torch.uint8", shape=None):
        self.payload = FakePayload(data, dtype, shape)
        self.dtype = dtype
        self.shape = tuple(shape) if shape is not None else (len(self.payload.data),)

    def numpy(self):
        return self.payload


class RandomNamespace:
    def __init__(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def get_state_tuple(self):
        return ("MT19937", FakePayload(b"numpy", "numpy.uint32", (5,)), 1, 2, None)


class NumpyNamespace:
    def __init__(self):
        self.random = RandomNamespace(("MT19937", FakePayload(b"numpy", "numpy.uint32", (5,)), 1, 2, None))


class TorchNamespace:
    def __init__(self, state=None):
        self.state = FakePayload(b"torch", "torch.uint8", (5,)) if state is None else state

    def get_rng_state(self):
        return self.state


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def guard_payload(unit, used_normal=0, used_reserve=0, refusals=0, events=None):
    return {"kind": "r75-test-guard/1", "unit": unit, "normal_cap": 12, "reserve_cap": 4,
            "hard_cap": 16, "used_normal": used_normal, "used_reserve": used_reserve,
            "refusals": refusals, "events": [] if events is None else events}


def ledger_charged(unit, category, n, used_after, what="test"):
    return {"unix": 1, "guard": "r75-test-guard/1", "event": "charged", "unit": unit,
            "category": category, "n": n, "what": what, "used_after": used_after}


def ledger_opened(unit):
    return {"unix": 0, "guard": "r75-test-guard/1", "event": "ledger-opened", "unit": unit,
            "used_normal": 0, "used_reserve": 0, "refusals": 0}


class ComparatorTests(Scratch):
    def test_tensor_payloads_use_dtype_shape_and_full_bytes(self):
        equal = self.assertEqual(runner._raw_equal(FakePayload(b"abcd"), FakePayload(b"abcd")), True)
        different = self.assertIs(runner._raw_equal(FakePayload(b"abce"), FakePayload(b"abcd")), False)
        dtype = self.assertIs(runner._raw_equal(FakePayload(b"abcd", "torch.int8"),
                                                FakePayload(b"abcd", "torch.uint8")), False)
        shape = self.assertIs(runner._raw_equal(FakePayload(b"abcd", shape=(4,)),
                                                FakePayload(b"abcd", shape=(2, 2))), False)
        return [equal, different, dtype, shape]

    def test_nested_containers_and_scalar_types_are_strict(self):
        self.assertTrue(runner._raw_equal((1, (b"x", None)), (1, (b"x", None))))
        self.assertFalse(runner._raw_equal((1,), [1]))
        self.assertFalse(runner._raw_equal(1, True))
        self.assertFalse(runner._raw_equal({"a": 1}, {"a": "1"}))
        self.assertTrue(runner._raw_equal({"a": (1, b"x")}, {"a": (1, b"x")}))

    def test_torch_like_payload_without_tobytes_is_compared_through_numpy_view(self):
        live = FakeTorchTensor(b"abcd")
        same = FakeTorchTensor(b"abcd")
        different = FakeTorchTensor(b"abce")
        self.assertTrue(runner._raw_equal(live, same))
        self.assertFalse(runner._raw_equal(live, different))
        detail = runner._first_raw_difference(live, different, "torch_like")
        self.assertEqual(detail["first_diff_byte"], 3)

    def test_vector_equality_object_is_never_reduced_to_bool(self):
        left, right = VectorEqualityObject(), VectorEqualityObject()
        self.assertIs(runner._raw_equal(left, right), False)
        self.assertIs(runner._raw_equal(left, left), True)

    def test_rng_payload_differences_are_empty_when_all_three_states_match(self):
        np_mod = NumpyNamespace()
        torch_mod = TorchNamespace()
        saved = {"_numpy_state": np_mod.random.get_state(),
                 "_torch_state": FakePayload(b"torch", "torch.uint8", (5,)),
                 "_python_state": random.getstate()}
        torch_mod.state = FakePayload(b"torch", "torch.uint8", (5,))
        self.assertEqual(runner._rng_payload_differences(saved, np_mod, torch_mod), [])

    def test_one_differing_torch_byte_reports_first_position(self):
        np_mod = NumpyNamespace()
        torch_mod = TorchNamespace(FakePayload(b"torcm", "torch.uint8", (5,)))
        saved = {"_numpy_state": np_mod.random.get_state(),
                 "_torch_state": FakePayload(b"torch", "torch.uint8", (5,)),
                 "_python_state": random.getstate()}
        problems = runner._rng_payload_differences(saved, np_mod, torch_mod)
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0]["where"], "rng_payload:torch_rng_state")
        self.assertEqual(problems[0]["detail"]["first_diff_byte"], 4)


class LedgerSnapshotTests(Scratch):
    def _runtime(self, unit, guard, events):
        runtime = self.tmp / "runtime"
        units = ["environment_construction", "joint_record",
                 "offline_trunk_target_row", "physical_optimizer_step"]
        events_by_unit = {name: [ledger_opened(name)] for name in units}
        for name in units:
            write_json(runtime / ("r75_" + name + "_guard.json"), guard_payload(name))
        if guard is not None:
            write_json(runtime / ("r75_" + unit + "_guard.json"), guard)
            events_by_unit[unit] = events
        with (runtime / "r75_total_ledger.jsonl").open("w", encoding="utf-8") as handle:
            for name in units:
                for event in events_by_unit[name]:
                    handle.write(json.dumps(event) + "\n")
        return runtime

    def _guard_name(self, unit):
        return "r75_" + unit + "_guard.json"

    def test_zero_guard_and_opened_ledger_reconcile_to_zero(self):
        unit = "physical_optimizer_step"
        runtime = self._runtime(unit, guard_payload(unit), [ledger_opened(unit)])
        snapshot = wrapper.ledger_snapshot(runtime, None)
        summary = snapshot["per_unit_last_known"][unit]
        self.assertEqual(summary["used_normal"], 0)
        self.assertEqual(summary["charged_events"], 0)
        self.assertTrue(summary["schema_ok"] and summary["guard_ledger_match"])
        self.assertTrue(snapshot["guard_ledger_reconciliation"]["ok"],
                        str(summary["problems"]))

    def test_normal_charged_events_sum_n_and_used_after(self):
        unit = "offline_trunk_target_row"
        events = [ledger_charged(unit, "normal", 4, 4), ledger_charged(unit, "normal", 5, 9)]
        runtime = self._runtime(unit, guard_payload(unit, used_normal=9, events=events), events)
        summary = wrapper.ledger_snapshot(runtime, None)["per_unit_last_known"][unit]
        self.assertEqual(summary["used_normal"], 9)
        self.assertEqual(summary["used_after_normal"], 9)
        self.assertEqual(summary["charged_events"], 2)

    def test_reserve_events_are_not_merged_into_normal(self):
        unit = "physical_optimizer_step"
        events = [ledger_charged(unit, "normal", 1, 1), ledger_charged(unit, "reserve", 2, 2)]
        guard = guard_payload(unit, used_normal=1, used_reserve=2, events=events)
        runtime = self._runtime(unit, guard, events)
        summary = wrapper.ledger_snapshot(runtime, None)["per_unit_last_known"][unit]
        self.assertEqual(summary["used_normal"], 1)
        self.assertEqual(summary["used_reserve"], 2)

    def test_refused_events_reconcile_with_guard_refusals(self):
        unit = "physical_optimizer_step"
        refusal = {"unix": 2, "guard": "r75-test-guard/1", "event": "refused", "unit": unit,
                   "what": "test-refusal", "category": "normal"}
        guard = guard_payload(unit, refusals=1, events=[refusal])
        runtime = self._runtime(unit, guard, [refusal])
        summary = wrapper.ledger_snapshot(runtime, None)["per_unit_last_known"][unit]
        self.assertEqual(summary["refusals"], 1)
        self.assertTrue(summary["guard_ledger_match"])

    def test_guard_top_level_normal_mismatch_becomes_unknown(self):
        unit = "physical_optimizer_step"
        events = [ledger_charged(unit, "normal", 1, 1)]
        guard = guard_payload(unit, used_normal=2, events=events)
        runtime = self._runtime(unit, guard, events)
        snapshot = wrapper.ledger_snapshot(runtime, None)
        summary = snapshot["per_unit_last_known"][unit]
        self.assertEqual(summary["used_normal"], "unknown")
        self.assertFalse(snapshot["guard_ledger_reconciliation"]["ok"])

    def test_guard_used_after_mismatch_becomes_unknown(self):
        unit = "physical_optimizer_step"
        events = [ledger_charged(unit, "normal", 1, 2)]
        guard = guard_payload(unit, used_normal=1, events=events)
        runtime = self._runtime(unit, guard, events)
        snapshot = wrapper.ledger_snapshot(runtime, None)
        self.assertEqual(snapshot["per_unit_last_known"][unit]["used_normal"], "unknown")

    def test_ledger_n_sum_mismatch_becomes_unknown(self):
        unit = "physical_optimizer_step"
        guard_events = [ledger_charged(unit, "normal", 1, 1)]
        ledger_events = [ledger_charged(unit, "normal", 4, 4)]
        guard = guard_payload(unit, used_normal=1, events=guard_events)
        runtime = self._runtime(unit, guard, ledger_events)
        snapshot = wrapper.ledger_snapshot(runtime, None)
        self.assertEqual(snapshot["per_unit_last_known"][unit]["used_normal"], "unknown")

    def test_invalid_guard_n_becomes_unknown(self):
        unit = "physical_optimizer_step"
        bad = ledger_charged(unit, "normal", "1", 1)
        guard = guard_payload(unit, used_normal=1, events=[bad])
        runtime = self._runtime(unit, guard, [bad])
        snapshot = wrapper.ledger_snapshot(runtime, None)
        self.assertEqual(snapshot["per_unit_last_known"][unit]["used_normal"], "unknown")

    def test_missing_guard_blocks_even_at_zero(self):
        runtime = self.tmp / "runtime"
        runtime.mkdir(parents=True)
        unit = "physical_optimizer_step"
        with (runtime / "r75_total_ledger.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(ledger_opened(unit)) + "\n")
        snapshot = wrapper.ledger_snapshot(runtime, None)
        summary = snapshot["per_unit_last_known"][unit]
        self.assertEqual(summary["used_normal"], "unknown")
        self.assertFalse(snapshot["guard_ledger_reconciliation"]["ok"])


class StageSnapshotTests(Scratch):
    def _state(self, record=None, unread=None):
        class State:
            def __init__(self):
                self.record = {} if record is None else record
                self.unread = [] if unread is None else unread
            def counter_report(self):
                return {"physical_optimizer_step": {"counted": 1}}
        return State()

    def test_completed_stage_snapshot_is_atomic_and_carries_full_state(self):
        path = self.tmp / "stage.pt"
        state = self._state({"A1_gate": {"passed": True}})
        state.unread.append("test:optional")
        identity = runner._save_stage_snapshot(path, stage="A2_completed", label="A2",
                                               state=state, full_state={"modules": {"trunk": 1}},
                                               save_fn=lambda target, payload: target.write_text(
                                                   json.dumps(payload), encoding="utf-8"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["stage"], "A2_completed")
        self.assertEqual(payload["full_state"]["modules"]["trunk"], 1)
        self.assertEqual(identity["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertFalse(path.with_name(path.name + ".tmp").exists())
        self.assertEqual(identity["unread"], ["test:optional"])

    def test_failed_stage_snapshot_leaves_no_temporary_or_target(self):
        path = self.tmp / "failed.pt"
        def failing_save(target, payload):
            target.write_text("partial", encoding="utf-8")
            raise RuntimeError("injected save failure")
        with self.assertRaises(RuntimeError):
            runner._save_stage_snapshot(path, stage="A2_completed", label="A2",
                                        state=self._state(), full_state={"x": 1},
                                        save_fn=failing_save)
        self.assertFalse(path.exists())
        self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_failure_receipt_cites_verified_completed_stage_artifact(self):
        artifact = self.tmp / "stage.pt"
        artifact.write_text("payload", encoding="utf-8")
        state = self._state({"A2_stage_snapshot": {"path": str(artifact),
                                                  "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}})
        artifacts = runner._completed_stage_artifacts(state)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["status"], "verified")

    def test_missing_and_hash_changed_stage_artifacts_are_not_reclassified(self):
        artifact = self.tmp / "stage.pt"
        artifact.write_text("payload", encoding="utf-8")
        missing = {"path": str(self.tmp / "missing.pt"), "sha256": "0" * 64}
        changed = {"path": str(artifact), "sha256": "1" * 64}
        state = self._state({"A2_stage_snapshot": missing, "A3_start_stage_snapshot": changed})
        artifacts = runner._completed_stage_artifacts(state)
        self.assertEqual(artifacts[0]["status"], "missing")
        self.assertEqual(artifacts[1]["status"], "hash_mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
