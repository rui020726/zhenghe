# -*- coding: utf-8 -*-
"""ROUND75 S0 restore / fail-fast counterexamples (request R75_S0_RESTORE_FAILFAST_20261015).

Pure stdlib. No torch, no numpy NN path, no model load or construction, no forward or backward,
no optimizer or PPO step, no environment, no sampling, no checkpoint write, no canonical-ledger
write. Every assertion uses FAKE objects, and the fake base is imported from the closeout suite so
the same seam behaviour is exercised rather than a parallel re-implementation.

What this suite pins down, on top of the seven closeout gaps:
  1. a COMPLETE bundle-shaped payload (assembled by hand, not produced by this runner's own
     capture) round-trips through _bundle_to_saved -> restore_full_state -> verify_restore;
  2. the expected RNG report is rebuilt FROM THE LOADED PAYLOADS, so a restore cannot supply its
     own expectation, and a live RNG state that differs from the payload is caught value by value
     instead of being hidden by a truncated repr;
  3. a step without a permit, or with two, is refused before delegation: zero delegations, one
     terminal record, and the consumed permits are never refunded; one permit gives exactly one
     delegation and exactly one record;
  4. the three fail-fast gates raise where the problem is found, and the source order proves a
     broken A1, a bundle that is not the file this run wrote, or an unproven A3 start cannot reach
     a paid native update;
  5. a failed attempt leaves a durable, appendable, fsynced receipt carrying the real phase, the
     raw exception and traceback, the real attempt counters, the observed steps, the meters and
     the durable counters, with anything unreadable marked UNKNOWN and never defaulted to zero.

Run:  python test_r75_s0_restore_failfast.py        (exit 0 = all counterexamples hold)
"""
from __future__ import annotations

import ast
import json
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import test_r75_s0_driver_closeout as base                       # noqa: E402  (stdlib-only fakes)
import r75_s1_runner as runner                                   # noqa: E402  (stdlib-only)

FakeTensor = base.FakeTensor
FakeRngState = base.FakeRngState
FakeNumpy = base.FakeNumpy
FakeTorch = base.FakeTorch
FakeResources = base.FakeResources
_controller_and_optimizer = base._controller_and_optimizer
_capture = base._capture


def _scratch():
    """A scratch directory INSIDE the R75 output root: the sandbox temp root is not writable."""
    return tempfile.mkdtemp(prefix="_r75_scratch_", dir=str(HERE))


def _stepped_pair():
    """A controller/optimizer pair whose fakes already carry one real Adam state."""
    controller, optimizer, names = _controller_and_optimizer()
    optimizer.step()                       # in-place step increment, exactly like torch Adam
    return controller, optimizer, names


def _bundle_of(controller, optimizer):
    """A COMPLETE bundle-shaped payload, built from a capture but consumed as a plain bundle."""
    return runner.build_state_bundle(_capture(controller, optimizer))


def _installed(np_mod, torch_mod, bundle):
    """Install the bundle's three RNG payloads into the live fakes, as a restore would."""
    np_mod.random.set_state(bundle["numpy_state"])
    torch_mod.set_rng_state(bundle["torch_state"])
    random.setstate(bundle["python_state"])


def _a1_record(**overrides):
    """An A1 record shaped exactly like run_a1 writes it, so the gate reads the same keys."""
    _controller, _optimizer, names = _controller_and_optimizer()
    adam_record = {name: {"state_present": False, "step": None,
                          "exp_avg_present": False, "exp_avg_sq_present": False}
                   for name in names}
    for name in names:
        if name.startswith("value_"):
            adam_record[name] = {"state_present": True, "step": 1,
                                 "exp_avg_present": True, "exp_avg_sq_present": True}
    record = {"loss": 1.0, "gradient_norm": 0.5, "critic_input_width": 4, "objects": names,
              "object_coverage": {"named_trainable_objects": len(names),
                                  "optimizer_parameters": len(names),
                                  "coverage_complete": True},
              "grad_record": {}, "objects_with_gradient": [], "objects_without_gradient": [],
              "adam_record": adam_record, "trunk_parameters_measured": 2,
              "trunk_parameters_in_optimizer": 0, "trunk_requires_grad": [False, False],
              "trunk_moved_parameters": [], "trunk_moved": False, "moved_objects": [],
              "max_grad_norm_used": 0.5}
    record.update(overrides)
    return record


# ======================================================================================
# 1. a complete bundle-shaped payload round-trips through the real bridge
# ======================================================================================
class CompleteBundleBridgeTests(unittest.TestCase):
    def test_a_hand_built_complete_bundle_round_trips_through_the_real_bridge(self):
        controller, optimizer, names = _stepped_pair()
        bundle = _bundle_of(controller, optimizer)
        fresh, fresh_optimizer, fresh_names = _controller_and_optimizer()
        np_mod, torch_mod = FakeNumpy(), FakeTorch()
        _installed(np_mod, torch_mod, bundle)

        saved = runner._bundle_to_saved(bundle)
        report = runner.restore_full_state(fresh, fresh_optimizer, saved,
                                           np_mod=np_mod, torch_mod=torch_mod)
        self.assertTrue(report["restored_exactly"], report["problems"])
        self.assertEqual(len(fresh_optimizer.load_state_dict_calls), 1,
                         "the restore must go through the real optimizer.load_state_dict")
        self.assertEqual(fresh_names, names)
        self.assertEqual(runner.compare_full_states(saved, saved, "probe"), [])
        self.assertEqual(sorted(runner.build_state_bundle(saved)), sorted(runner.BUNDLE_FIELDS))

    def test_the_expected_rng_report_comes_from_the_payload_not_the_live_state(self):
        controller, optimizer, _names = _stepped_pair()
        bundle = _bundle_of(controller, optimizer)
        np_mod, torch_mod = FakeNumpy(), FakeTorch()
        # a live state that differs from the payload in every one of the three streams
        torch_mod.set_rng_state(FakeRngState(b"a-totally-different-live-torch-state"))
        np_mod.random.set_state(("MT19937", (7, 7, 7, 7)))
        saved = runner._bundle_to_saved(bundle)
        self.assertEqual(saved["rng"]["torch_rng_state_sha256"],
                         runner._sha256_bytes(bundle["torch_state"].tobytes()),
                         "the expectation must be rebuilt from the loaded payload")
        self.assertEqual(saved["rng"]["numpy_random_state"], repr(bundle["numpy_state"]))
        self.assertEqual(saved["rng"]["python_random_state"], repr(bundle["python_state"]))

    def test_a_live_rng_state_that_differs_from_the_payload_is_detected(self):
        controller, optimizer, _names = _stepped_pair()
        bundle = _bundle_of(controller, optimizer)
        saved = runner._bundle_to_saved(bundle)
        np_mod, torch_mod = FakeNumpy(), FakeTorch()
        _installed(np_mod, torch_mod, bundle)
        self.assertEqual(runner._rng_payload_differences(saved, np_mod, torch_mod), [])
        torch_mod.set_rng_state(FakeRngState(b"a-different-live-state"))
        problems = runner._rng_payload_differences(saved, np_mod, torch_mod)
        self.assertEqual([item["where"] for item in problems],
                         ["rng_payload:torch_rng_state"])
        self.assertEqual(problems[0]["detail"]["first_diff_byte"], 0)

    def test_a_tampered_module_payload_is_caught_against_the_baseline(self):
        controller, optimizer, _names = _stepped_pair()
        baseline = _capture(controller, optimizer)
        bundle = _bundle_of(controller, optimizer)
        bundle["module_payloads"]["trunk"]["trunk.weight"] = FakeTensor([9.0, 9.0])
        fresh, fresh_optimizer, _fresh_names = _controller_and_optimizer()
        np_mod, torch_mod = FakeNumpy(), FakeTorch()
        _installed(np_mod, torch_mod, bundle)
        report = runner.restore_full_state(fresh, fresh_optimizer,
                                           runner._bundle_to_saved(bundle),
                                           np_mod=np_mod, torch_mod=torch_mod)
        # the restore reproduces the BUNDLE exactly - that is what a restore is for
        self.assertTrue(report["restored_exactly"], report["problems"])
        a3_start = runner.capture_full_state(fresh, fresh_optimizer, np_mod=np_mod,
                                             torch_mod=torch_mod,
                                             data_identity=bundle["data_identity"],
                                             configuration=bundle["configuration"],
                                             cursor=bundle["update_cursor"])
        differences = runner.compare_full_states(baseline, a3_start, "probe")
        self.assertTrue(any(item["where"] == "probe:module:trunk:trunk.weight"
                            for item in differences), differences)


# ======================================================================================
# 2. raw payload differences are never hidden by a printed abbreviation
# ======================================================================================
class RawPayloadDifferenceTests(unittest.TestCase):
    def test_a_single_differing_byte_at_the_end_of_a_long_payload_is_located(self):
        left = FakeRngState(bytes([7]) * 2048)
        right = FakeRngState(bytes([7]) * 2047 + bytes([8]))
        self.assertFalse(runner._raw_equal(left, right))
        detail = runner._first_raw_difference(left, right, "torch")
        self.assertEqual(detail["first_diff_byte"], 2047)
        self.assertEqual(detail["live"], 7)
        self.assertEqual(detail["expected"], 8)
        self.assertEqual(detail["live_len"], 2048)
        self.assertEqual(detail["expected_len"], 2048)

    def test_a_difference_in_the_last_element_of_a_long_tuple_is_located(self):
        left = ("MT19937", tuple(range(624)) + (0,))
        right = ("MT19937", tuple(range(624)) + (1,))
        self.assertFalse(runner._raw_equal(left, right))
        detail = runner._first_raw_difference(left, right, "numpy")
        self.assertEqual(detail["where"], "numpy[1][624]")

    def test_a_prefix_payload_is_reported_as_a_prefix_not_as_equal(self):
        left = FakeRngState(bytes([1, 2, 3, 4]))
        right = FakeRngState(bytes([1, 2, 3]))
        self.assertFalse(runner._raw_equal(left, right))
        detail = runner._first_raw_difference(left, right, "torch")
        self.assertEqual(detail["live_len"], 4)
        self.assertEqual(detail["expected_len"], 3)
        self.assertIn("prefix", detail["note"])

    def test_a_dtype_difference_is_a_difference(self):
        left = FakeRngState(b"abcd")
        right = FakeRngState(b"abcd")
        right.dtype = "torch.uint8"
        self.assertFalse(runner._raw_equal(left, right))

    def test_identical_payloads_are_equal(self):
        self.assertTrue(runner._raw_equal(("MT19937", (1, 2, 3)), ("MT19937", (1, 2, 3))))
        self.assertTrue(runner._raw_equal(FakeRngState(b"xy"), FakeRngState(b"xy")))


# ======================================================================================
# 3. the permit contract: refuse before delegation, never refund
# ======================================================================================
class PermitRefusalTests(unittest.TestCase):
    def test_no_permit_means_zero_delegations_and_no_refund(self):
        _controller, optimizer, _names = _controller_and_optimizer()
        observer = runner.ObservedStepEntry(optimizer, FakeResources(), "probe")
        with self.assertRaises(RuntimeError):
            observer.step()
        self.assertEqual(optimizer.step_calls, 0)
        self.assertFalse(observer.records[-1]["delegated"])
        self.assertEqual(observer._resources.used, 0)

    def test_two_permits_means_zero_delegations_and_no_refund(self):
        _controller, optimizer, _names = _controller_and_optimizer()
        observer = runner.ObservedStepEntry(optimizer, FakeResources(), "probe")
        observer._resources.charge_physical_step("permit")
        observer._resources.charge_physical_step("permit-again")
        with self.assertRaises(RuntimeError):
            observer.step()
        self.assertEqual(optimizer.step_calls, 0)
        self.assertEqual(observer._resources.used, 2, "a refusal must NOT be refunded")

    def test_one_permit_gives_exactly_one_delegation_and_one_record(self):
        _controller, optimizer, _names = _controller_and_optimizer()
        observer = runner.ObservedStepEntry(optimizer, FakeResources(), "probe")
        observer._resources.charge_physical_step("permit")
        observer.step()
        self.assertEqual(optimizer.step_calls, 1)
        self.assertEqual(len(observer.records), 1, "one real step, exactly one terminal record")
        self.assertEqual(observer.records[-1]["phase"], "completed")
        self.assertTrue(observer.records[-1]["delegated"])

    def test_a_refused_step_is_still_counted_as_an_attempt(self):
        _controller, optimizer, _names = _controller_and_optimizer()
        state = runner._RunState()
        observer = runner.ObservedStepEntry(optimizer, FakeResources(), "probe")
        state.observers["probe"] = observer
        with self.assertRaises(RuntimeError):
            observer.step()
        counted = state.counter_report()["optimizer_step_attempts"]
        self.assertEqual(counted["counted"], 1, "a refused step is an attempt that really happened")
        self.assertEqual(counted["entries"][0]["phase"], "refused")

    def test_the_frozen_nine_steps_come_from_one_record_each(self):
        state = runner._RunState()
        for label in ("A1_A2", "A3"):
            _controller, optimizer, _names = _controller_and_optimizer()
            observer = runner.ObservedStepEntry(optimizer, FakeResources(), label)
            state.observers[label] = observer
        # 1 A1 control step + 4 A2 on the first observer, 4 A3 on the second: 9 in total,
        # each with its own permit and its own single terminal record
        for _ in range(5):
            state.observers["A1_A2"]._resources.charge_physical_step("permit")
            state.observers["A1_A2"].step()
        for _ in range(4):
            state.observers["A3"]._resources.charge_physical_step("permit")
            state.observers["A3"].step()
        counted = state.counter_report()["optimizer_step_attempts"]
        self.assertEqual(counted["counted"], 9)
        self.assertEqual(runner.PLAN["accounting"]["planned_totals"]["physical_optimizer_steps"], 9)
        self.assertTrue(counted["within_conservative_bound"])


# ======================================================================================
# 4. the fail-fast gates raise where the problem is found
# ======================================================================================
class FailFastGateTests(unittest.TestCase):
    def setUp(self):
        self._dirs = []

    def tearDown(self):
        for path in self._dirs:
            shutil.rmtree(path, ignore_errors=True)

    def _scratch(self):
        path = _scratch()
        self._dirs.append(path)
        return Path(path)

    def test_a_clean_a1_passes_the_gate(self):
        gate = runner._gate_a1(_a1_record())
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["mapped_trainable_objects"], gate["expected_trainable_objects"])
        self.assertEqual(len(gate["critics_with_nonempty_adam"]), 4,
                         "both critic modules carry two trainable parameters each")

    def test_a_moved_trunk_is_refused_by_the_gate(self):
        with self.assertRaises(RuntimeError):
            runner._gate_a1(_a1_record(trunk_moved=True, trunk_moved_parameters=["trunk_p0"]))

    def test_a_trunk_parameter_inside_the_optimizer_is_refused(self):
        with self.assertRaises(RuntimeError):
            runner._gate_a1(_a1_record(trunk_parameters_in_optimizer=1))

    def test_an_incomplete_object_mapping_is_refused(self):
        with self.assertRaises(RuntimeError):
            runner._gate_a1(_a1_record(object_coverage={"named_trainable_objects": 7,
                                                        "optimizer_parameters": 8,
                                                        "coverage_complete": False}))

    def test_a_duplicated_object_name_is_refused(self):
        record = _a1_record()
        record["objects"] = record["objects"][:-1] + [record["objects"][0]]
        with self.assertRaises(RuntimeError):
            runner._gate_a1(record)

    def test_a_critic_without_adam_state_is_refused(self):
        record = _a1_record()
        critics = [name for name in record["objects"] if name.startswith("value_")]
        self.assertEqual(len(critics), 4, "both critic modules carry two parameters each")
        record["adam_record"][critics[0]]["state_present"] = False
        with self.assertRaises(RuntimeError):
            runner._gate_a1(record)

    def test_a_bundle_that_is_not_the_file_this_run_wrote_is_refused(self):
        controller, optimizer, _names = _stepped_pair()
        bundle = _bundle_of(controller, optimizer)
        scratch = self._scratch()
        path = scratch / "bundle.pt"
        path.write_bytes(b"not-the-bytes-that-were-saved")
        with self.assertRaises(RuntimeError):
            runner._gate_loaded_bundle(bundle, path, "0" * 64)

    def test_a_bundle_missing_a_resume_field_is_refused(self):
        controller, optimizer, _names = _stepped_pair()
        bundle = _bundle_of(controller, optimizer)
        del bundle["python_state"]
        scratch = self._scratch()
        path = scratch / "bundle.pt"
        payload = b"the-bytes"
        path.write_bytes(payload)
        with self.assertRaises(RuntimeError):
            runner._gate_loaded_bundle(bundle, path, runner._sha256_bytes(payload))

    def test_a_bundle_with_a_foreign_code_identity_is_refused(self):
        controller, optimizer, _names = _stepped_pair()
        bundle = _bundle_of(controller, optimizer)
        bundle["code_identity"] = {"r75_s1_runner.py": "0" * 64}
        scratch = self._scratch()
        path = scratch / "bundle.pt"
        payload = b"the-bytes"
        path.write_bytes(payload)
        with self.assertRaises(RuntimeError):
            runner._gate_loaded_bundle(bundle, path, runner._sha256_bytes(payload))

    def test_a_clean_bundle_passes_the_gate(self):
        controller, optimizer, _names = _stepped_pair()
        bundle = _bundle_of(controller, optimizer)
        scratch = self._scratch()
        path = scratch / "bundle.pt"
        payload = b"the-bytes"
        path.write_bytes(payload)
        gate = runner._gate_loaded_bundle(bundle, path, runner._sha256_bytes(payload))
        self.assertTrue(gate["passed"])
        self.assertEqual(len(gate["fields_present"]), len(runner.BUNDLE_FIELDS))

    def test_an_unproven_a3_start_is_refused_before_the_update(self):
        controller, optimizer, _names = _stepped_pair()
        baseline = _capture(controller, optimizer)
        drifted = _capture(controller, optimizer)
        drifted["update_cursor"] = int(baseline["update_cursor"]) + 5
        with self.assertRaises(RuntimeError):
            runner._gate_a3_start(baseline, drifted, {"restored_exactly": True, "problems": []},
                                  {"update_cursor": int(baseline["update_cursor"])},
                                  controller=controller, optimizer=optimizer, episode=object(),
                                  update_index=0, np_mod=FakeNumpy(), torch_mod=FakeTorch())

    def test_a_failed_restore_is_refused_even_when_the_states_look_equal(self):
        controller, optimizer, _names = _stepped_pair()
        baseline = _capture(controller, optimizer)
        with self.assertRaises(RuntimeError):
            runner._gate_a3_start(baseline, baseline,
                                  {"restored_exactly": False,
                                   "problems": [{"where": "adam:early_head.weight"}]},
                                  {"update_cursor": int(baseline["update_cursor"])},
                                  controller=controller, optimizer=optimizer, episode=object(),
                                  update_index=0, np_mod=FakeNumpy(), torch_mod=FakeTorch())

    def test_a_proven_start_passes_and_binds_the_update_to_the_verified_objects(self):
        controller, optimizer, _names = _stepped_pair()
        baseline = _capture(controller, optimizer)
        np_mod, torch_mod = FakeNumpy(), FakeTorch()
        np_mod.random.set_state(baseline["_numpy_state"])
        torch_mod.set_rng_state(baseline["_torch_state"])
        random.setstate(baseline["_python_state"])
        episode = object()
        gate = runner._gate_a3_start(baseline, baseline, {"restored_exactly": True, "problems": []},
                                     {"update_cursor": int(baseline["update_cursor"])},
                                     controller=controller, optimizer=optimizer, episode=episode,
                                     update_index=0, np_mod=np_mod, torch_mod=torch_mod)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["differences"], [])
        self.assertEqual(gate["rng_problems"], [])
        self.assertEqual(gate["re_read_drift"], [])
        self.assertEqual(gate["binding"]["update_cursor"], int(baseline["update_cursor"]))
        self.assertEqual(gate["binding"]["controller_id"], id(controller))
        self.assertEqual(gate["binding"]["optimizer_id"], id(optimizer))
        self.assertEqual(gate["binding"]["episode_id"], id(episode))

    def test_the_source_order_proves_no_gate_can_be_reached_after_a_paid_update(self):
        source = (HERE / "r75_s1_runner.py").read_text(encoding="utf-8")
        function = next(node for node in ast.walk(ast.parse(source))
                        if isinstance(node, ast.FunctionDef) and node.name == "run_s1")
        watched = ("_gate_a1", "_gate_loaded_bundle", "_gate_a3_start", "load_state_bundle",
                   "restore_full_state", "r75_frozen.r75_native_ppo_update")
        seen = sorted((node.lineno, ast.unparse(node.func)) for node in ast.walk(function)
                      if isinstance(node, ast.Call) and ast.unparse(node.func) in watched)
        lines = {}
        for lineno, name in seen:
            lines.setdefault(name, []).append(lineno)
        self.assertEqual(len(lines["r75_frozen.r75_native_ppo_update"]), 2,
                         "exactly two paid native updates are planned: A2 and A3")
        first_update = lines["r75_frozen.r75_native_ppo_update"][0]
        second_update = lines["r75_frozen.r75_native_ppo_update"][1]
        self.assertEqual(len(lines["load_state_bundle"]), 1)
        self.assertEqual(len(lines["restore_full_state"]), 1)
        self.assertLess(lines["_gate_a1"][0], first_update,
                        "the A1 gate must run before the FIRST paid native update")
        self.assertLess(lines["_gate_loaded_bundle"][0], lines["restore_full_state"][0],
                        "the bundle gate must run before its restore is trusted")
        self.assertLess(lines["_gate_a3_start"][0], second_update,
                        "the A3 start gate must run before the SECOND paid native update")
        self.assertLess(lines["load_state_bundle"][0], lines["restore_full_state"][0],
                        "the disk file must be read before it is restored")
        self.assertLess(lines["restore_full_state"][0], second_update,
                        "the restore must complete before the A3 update is issued")


# ======================================================================================
# 5. the durable failure receipt
# ======================================================================================
class FailureReceiptTests(unittest.TestCase):
    def setUp(self):
        self._original_s1_dir = runner.S1_DIR
        self._dir = _scratch()
        runner.S1_DIR = Path(self._dir)

    def tearDown(self):
        runner.S1_DIR = self._original_s1_dir
        shutil.rmtree(self._dir, ignore_errors=True)

    def _state(self):
        _controller, optimizer, _names = _controller_and_optimizer()
        state = runner._RunState()
        state.phase = "A3_native_update"
        state.count("gate_controller_constructions", "build_gate_only_controller", label="A3")
        state.count("native_ppo_update_attempts", "r75_native_ppo_update", label="A2")
        observer = runner.ObservedStepEntry(optimizer, FakeResources(), "probe")
        observer._resources.charge_physical_step("permit")
        observer.step()
        state.observers["A1_A2"] = observer
        state.resources = FakeResources()
        state.resources.used = 5
        return state

    def _events(self):
        path = runner.S1_DIR / "r75_s1_failure_receipts.jsonl"
        return [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_a_failure_leaves_a_durable_receipt_with_every_required_field(self):
        state = self._state()
        try:
            raise ValueError("the paid path broke here")
        except ValueError as error:
            path = runner._write_failure_receipt(state, error)
        self.assertTrue(path.exists())
        self.assertEqual(len(self._events()), 1)
        event = self._events()[0]
        self.assertEqual(event["kind"], "round75-s1-failure-receipt/1")
        self.assertEqual(event["phase"], "A3_native_update")
        self.assertEqual(event["error"]["type"], "ValueError")
        self.assertIn("the paid path broke here", event["error"]["message"])
        self.assertIn("ValueError", event["traceback"])
        self.assertIn("Traceback", event["traceback"])
        self.assertEqual(event["counters"]["gate_controller_constructions"]["counted"], 1)
        self.assertEqual(event["counters"]["native_ppo_update_attempts"]["counted"], 1)
        self.assertEqual(event["step_observations"]["A1_A2"][0]["phase"], "completed")
        self.assertEqual(event["durable_counters"]["physical_optimizer_step"]["used"], 5)
        self.assertEqual(event["unread"], [])

    def test_a_second_failure_appends_and_never_overwrites(self):
        state = self._state()
        for message in ("first", "second"):
            try:
                raise RuntimeError(message)
            except RuntimeError as error:
                runner._write_failure_receipt(state, error)
        self.assertEqual([event["error"]["message"] for event in self._events()],
                         ["first", "second"])

    def test_an_unreadable_meter_or_counter_is_unknown_and_never_zero(self):
        class Exploding:
            def snapshots(self):
                raise OSError("the durable counter cannot be read")

            def snapshot(self):
                raise OSError("the meter cannot be read")

        state = self._state()
        state.meters["A3"] = Exploding()
        state.resources = Exploding()
        try:
            raise KeyError("boom")
        except KeyError as error:
            runner._write_failure_receipt(state, error)
        event = self._events()[-1]
        self.assertEqual(event["meter_snapshots"]["A3"]["unknown"], "OSError: the meter cannot be read")
        self.assertEqual(event["durable_counters"]["unknown"],
                         "OSError: the durable counter cannot be read")
        self.assertIn("meter:A3", event["unread"])
        self.assertIn("durable_counters", event["unread"])

    def test_a_state_with_no_open_resources_is_unknown_not_zero(self):
        state = self._state()
        state.resources = None
        try:
            raise RuntimeError("stopped before the ledger was opened")
        except RuntimeError as error:
            runner._write_failure_receipt(state, error)
        event = self._events()[-1]
        self.assertIn("unknown", event["durable_counters"])
        self.assertNotEqual(event["durable_counters"].get("physical_optimizer_step", {}).get("used"),
                            0)

    def test_the_counters_report_units_bounds_and_the_real_entries_together(self):
        state = self._state()
        state.count("paid_module_imports", "torch")
        report = state.counter_report()
        self.assertEqual(report["paid_module_imports"]["counted"], 1)
        self.assertEqual(report["paid_module_imports"]["unit"], "modules")
        self.assertTrue(report["paid_module_imports"]["within_conservative_bound"])
        self.assertEqual(report["optimizer_step_attempts"]["counted"], 1)
        self.assertEqual(report["optimizer_step_attempts"]["unit"], "steps")
        bounds = runner._conservative_attempt_bounds()
        self.assertEqual(bounds["optimizer_step_attempts"]["bound"],
                         runner.PLAN["accounting"]["planned_totals"]["physical_optimizer_steps"])
        self.assertEqual(set(bounds), set(runner.COUNTER_UNITS))

    def test_the_runner_body_registers_the_phase_before_each_paid_step(self):
        source = (HERE / "r75_s1_runner.py").read_text(encoding="utf-8")
        body = source[source.index("def run_s1() -> int:"):source.index("def _repo_root()")]
        for phase in ("precheck", "paid_module_import", "A2_controller_construction",
                      "replay_data_load", "A1_critic_only_control_step", "A1_baseline_capture",
                      "A2_native_update", "A3_controller_construction", "A3_bundle_load",
                      "A3_restore", "A3_start_gate", "A3_native_update", "closeout"):
            self.assertIn(f'state.phase = "{phase}"', body, f"the phase {phase} is not registered")
        self.assertIn("except BaseException as error:", body)
        self.assertIn("_write_failure_receipt(state, error)", body)
        self.assertIn("finally:", body)
        self.assertIn("r75_s1_attempt_counters.jsonl", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)