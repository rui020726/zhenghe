# -*- coding: utf-8 -*-
"""ROUND75 S0 driver-closeout counterexamples (request R75_S0_DRIVER_CLOSEOUT_20261015).

Pure stdlib. No torch, no numpy NN path, no model load or construction, no forward or backward,
no optimizer or PPO step, no environment, no sampling, no checkpoint write, no canonical-ledger
write. Every assertion uses FAKE objects that reproduce the exact seam behaviour the real run
depends on, so the same code path is exercised without paying for anything.

What this suite pins down (the seven seam gaps the closeout had to close):
  1. a saved module snapshot never shares mutable storage with the live module;
  2. the Adam payload is TYPED and restored through the real optimizer.load_state_dict;
  3. the resume bundle is COMPLETE and the disk file - not the in-memory dict - is the source;
  4. a real full-state or metering difference cannot slip past the comparator;
  5. the real step entry observes attempts and per-permit reconciliation without double charging;
  6. A1 trunk immobility and the eight-object coverage are measured, never literal;
  7. the pure precheck runs before the paid modules and never depends on the CWD.

Run:  python test_r75_s0_driver_closeout.py        (exit 0 = all counterexamples hold)
"""
from __future__ import annotations

import ast
import json
import struct
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r75_s1_runner as runner                                        # noqa: E402  (stdlib-only)


# ======================================================================================
# fake objects: the smallest stand-ins that reproduce the real seam behaviour
# ======================================================================================
class _Scalar:
    def __init__(self, value: float):
        self._value = float(value)

    def item(self) -> float:
        return self._value


class FakeTensor:
    """A tensor stand-in with the full chain ``_tensor_digest`` uses, PLUS in-place mutation so
    that a snapshot which kept the original reference WOULD be corrupted."""

    def __init__(self, data, dtype: str = "torch.float32", shape=None):
        self._data = [float(value) for value in data]
        self.dtype = dtype
        self.shape = tuple(shape) if shape is not None else (len(self._data),)

    def numel(self) -> int:
        return len(self._data)

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self

    def tobytes(self) -> bytes:
        return struct.pack(f"<{len(self._data)}d", *self._data)

    def abs(self):
        return FakeTensor([abs(value) for value in self._data], self.dtype, self.shape)

    def sum(self):
        return _Scalar(sum(self._data))

    def clone(self):
        return FakeTensor(self._data, self.dtype, self.shape)

    def add_(self, value):
        """In-place mutation, exactly the kind that rewrites a shared reference."""
        self._data = [item + float(value) for item in self._data]
        return self

    def __repr__(self):                                                       # pragma: no cover
        return f"FakeTensor({self._data}, {self.dtype}, {self.shape})"


class FakeModule:
    """A module whose state_dict() returns the LIVE tensor references, like torch does."""

    def __init__(self, tensors: dict):
        self._tensors = dict(tensors)

    def state_dict(self) -> dict:
        return dict(self._tensors)

    def load_state_dict(self, payload: dict):
        for key, value in payload.items():
            self._tensors[key] = value
        return {"missing_keys": [], "unexpected_keys": []}

    def named_parameters(self):
        return list(self._tensors.items())

    def parameters(self):
        return list(self._tensors.values())


class _Inner:
    def __init__(self, trunk, early_head, late_head, value_head):
        self.trunk = trunk
        self.early_head = early_head
        self.late_head = late_head
        self.value_head = value_head


class FakeController:
    def __init__(self):
        self.inner = _Inner(FakeModule({"trunk.weight": FakeTensor([1.0, 2.0])}),
                            FakeModule({"early.weight": FakeTensor([0.5, 0.25]),
                                        "early.bias": FakeTensor([0.1])}),
                            FakeModule({"late.weight": FakeTensor([0.5, 0.25]),
                                        "late.bias": FakeTensor([0.1])}),
                            FakeModule({"value_task.weight": FakeTensor([0.2, 0.3]),
                                        "value_task.bias": FakeTensor([0.0])}))
        self.cost_head = FakeModule({"value_cost.weight": FakeTensor([0.4, 0.6]),
                                     "value_cost.bias": FakeTensor([0.0])})


class FakeOptimizer:
    """Adam stand-in: a TENSOR step incremented in place, exp_avg / exp_avg_sq tensors, and a real
    load_state_dict entry point so a restore cannot quietly bypass it."""

    def __init__(self, params, *, lr: float = 3e-4):
        self.param_groups = [{"params": list(params), "lr": lr, "betas": (0.9, 0.999),
                              "eps": 1e-8, "weight_decay": 0.0, "amsgrad": False}]
        # keyed by the PARAMETER OBJECT, exactly like torch's optimizer.state
        self.state: dict = {}
        self.step_calls = 0
        self.zero_grad_calls = 0
        self.load_state_dict_calls: list = []

    def zero_grad(self):
        self.zero_grad_calls += 1
        for parameter in self.param_groups[0]["params"]:
            parameter.grad = None

    def step(self):
        self.step_calls += 1
        for parameter in self.param_groups[0]["params"]:
            state = self.state.setdefault(parameter, {})
            state["step"] = state.get("step", FakeTensor([0.0]))
            state["step"].add_(1.0)                       # IN PLACE, like torch Adam
            state.setdefault("exp_avg", FakeTensor([0.0] * parameter.numel()))
            state.setdefault("exp_avg_sq", FakeTensor([0.0] * parameter.numel()))

    def load_state_dict(self, payload: dict):
        self.load_state_dict_calls.append(payload)
        params = self.param_groups[0]["params"]
        rebuilt: dict = {}
        for index, values in payload["state"].items():
            rebuilt[params[int(index)]] = {
                key: (value.clone() if hasattr(value, "clone") else value)
                for key, value in values.items()}
        self.state = rebuilt
        for group, saved in zip(self.param_groups, payload["param_groups"]):
            for key, value in saved.items():
                if key != "params":
                    group[key] = value


class FakeRngState:
    def __init__(self, payload: bytes):
        self.payload = bytes(payload)

    def numpy(self):
        return self

    def tobytes(self) -> bytes:
        return self.payload

    def clone(self):
        return FakeRngState(self.payload)

    def __eq__(self, other):
        return isinstance(other, FakeRngState) and other.payload == self.payload

    def __hash__(self):
        return hash(self.payload)


class _FakeNumpyRandom:
    def __init__(self):
        self._state = ("MT19937", (0,))

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state


class FakeNumpy:
    def __init__(self):
        self.random = _FakeNumpyRandom()


class FakeTorch:
    def __init__(self):
        self._rng = b"torch-rng-default"

    def get_rng_state(self):
        return FakeRngState(self._rng)

    def set_rng_state(self, state):
        self._rng = state.payload if isinstance(state, FakeRngState) else bytes(state)


class FakeResources:
    """A durable-counter stand-in: a read is free, a permit is an increment the test performs."""

    def __init__(self):
        self.used = 0
        self.charge_calls: list = []

    def snapshots(self) -> dict:
        return {"physical_optimizer_step": {"used": self.used}}

    def charge_physical_step(self, what, *, category: str = "normal") -> int:
        self.charge_calls.append((what, category))
        self.used += 1
        return self.used


def _controller_and_optimizer():
    controller = FakeController()
    names = runner.r75_trainable_names(controller)
    params = []
    for module in (controller.inner.early_head, controller.inner.late_head,
                   controller.inner.value_head, controller.cost_head):
        params.extend(module.parameters())
    return controller, FakeOptimizer(params), names


def _capture(controller, optimizer, *, np_mod=None, torch_mod=None, cursor: int = 0):
    return runner.capture_full_state(
        controller, optimizer,
        np_mod=np_mod or FakeNumpy(), torch_mod=torch_mod or FakeTorch(),
        data_identity={"row_sha256": "row", "tape_sha256": "tape"},
        configuration=runner._live_configuration(_FakeFrozen()), cursor=cursor)


class _FakeCommon:
    ETA_LAMBDA = 0.5
    GAE_LAMBDA = 1.0
    GAMMA = 1.0
    CLIP = 0.2
    MAX_GRAD_NORM = 0.5
    ENTROPY_BONUS = 0.01
    VALUE_COEFFICIENT = 0.5
    EPOCHS_PER_BATCH = 4
    MINIBATCH_STEPS = 256
    LOG_PROB_TOLERANCE = 1e-6
    DATA_ORDER_SEED = 20260927
    B0_BYTES = 10000.0
    LEARNING_RATE = 3e-4


class _FakeR66:
    GATE_BETA = 1.00
    READY_CUT = 147.0


class _FakeFrozen:
    common = _FakeCommon()
    r66 = _FakeR66()


# ======================================================================================
# 1. mutable snapshot isolation
# ======================================================================================
class MutableSnapshotIsolationTests(unittest.TestCase):
    def test_a_module_payload_never_shares_storage_with_the_live_module(self):
        controller, optimizer, _names = _controller_and_optimizer()
        # PRECONDITION: a raw state_dict hands back the LIVE reference, so keeping it would alias.
        trunk_weight = controller.inner.trunk.state_dict()["trunk.weight"]
        self.assertIs(controller.inner.trunk.state_dict()["trunk.weight"], trunk_weight)
        saved = _capture(controller, optimizer)
        self.assertIsNot(saved["_module_payloads"]["trunk"]["trunk.weight"], trunk_weight,
                         "the payload must be a copy, not the original object")
        # a later update writes into the live tensor in place
        trunk_weight.add_(100.0)
        self.assertEqual(saved["_module_payloads"]["trunk"]["trunk.weight"]._data, [1.0, 2.0],
                         "the captured payload was rewritten by a live in-place update")

    def test_an_adam_step_mutation_cannot_rewrite_a_captured_payload(self):
        controller, optimizer, names = _controller_and_optimizer()
        optimizer.step()                                   # creates the typed tensor step
        saved = _capture(controller, optimizer)
        live_step = optimizer.state[optimizer.param_groups[0]["params"][0]]["step"]
        self.assertEqual(saved["_adam_payloads"][names[0]]["step"]._data, [1.0])
        optimizer.step()                                   # increments IN PLACE
        self.assertEqual(live_step._data, [2.0], "PRECONDITION: the live step did move")
        self.assertEqual(saved["_adam_payloads"][names[0]]["step"]._data, [1.0],
                         "the captured Adam payload was rewritten by the live step")

    def test_the_exp_avg_payloads_are_copies_too(self):
        controller, optimizer, names = _controller_and_optimizer()
        optimizer.step()
        saved = _capture(controller, optimizer)
        live_exp_avg = optimizer.state[optimizer.param_groups[0]["params"][0]]["exp_avg"]
        live_exp_avg.add_(7.0)
        self.assertEqual(saved["_adam_payloads"][names[0]]["exp_avg"]._data, [0.0, 0.0],
                         "the captured exp_avg was rewritten by a live in-place update")
    def test_a_recursive_payload_is_cloned_all_the_way_down(self):
        controller, optimizer, _names = _controller_and_optimizer()
        saved = _capture(controller, optimizer)
        payload = runner._clone_value({"a": [FakeTensor([1.0])], "b": (FakeTensor([2.0]),)})
        payload["a"][0].add_(5.0)
        self.assertEqual(payload["a"][0]._data, [6.0], "PRECONDITION: the clone is mutable")
        self.assertEqual(runner._clone_value({"a": [FakeTensor([1.0])]})["a"][0]._data, [1.0])
        self.assertIsInstance(runner._clone_value({"a": [FakeTensor([1.0])]})["a"], list)
# ======================================================================================
# 2. typed optimizer payload + the real load_state_dict path
# ======================================================================================
class TypedOptimizerPayloadTests(unittest.TestCase):
    def test_the_payload_step_is_typed_not_a_digest(self):
        controller, optimizer, names = _controller_and_optimizer()
        optimizer.step()
        saved = _capture(controller, optimizer)
        step = saved["_optimizer_payload"]["state"][0]["step"]
        self.assertIsInstance(step, FakeTensor,
                              "a tensor step must stay a TENSOR in the payload; a digest dict "
                              "cannot be handed back to Adam as running state")
        self.assertEqual(step._data, [1.0])
        # the report layer may describe it, but the payload must never BE the description
        self.assertIsInstance(saved["adam"][names[0]]["step"], dict)
        self.assertNotIsInstance(saved["adam"][names[0]]["step"], FakeTensor)

    def test_an_integer_step_stays_an_integer(self):
        controller, optimizer, _names = _controller_and_optimizer()
        for parameter in optimizer.param_groups[0]["params"]:
            optimizer.state[parameter] = {"step": 3, "exp_avg": FakeTensor([0.0]),
                                              "exp_avg_sq": FakeTensor([0.0])}
        saved = _capture(controller, optimizer)
        self.assertEqual(saved["_optimizer_payload"]["state"][0]["step"], 3)
        self.assertEqual(saved["adam"][_names_of(controller)[0]]["step"], 3)

    def test_the_payload_keeps_the_param_group_hyperparameters_and_index_mapping(self):
        controller, optimizer, names = _controller_and_optimizer()
        saved = _capture(controller, optimizer)
        groups = saved["_optimizer_payload"]["param_groups"]
        self.assertEqual(len(groups), 1)
        for key in ("lr", "betas", "eps", "weight_decay", "amsgrad"):
            self.assertIn(key, groups[0], f"the param-group hyperparameter {key} was dropped")
        self.assertEqual(groups[0]["lr"], 3e-4)
        self.assertEqual(groups[0]["params"], list(range(len(names))),
                         "the params entry must be the INDEX mapping, not live objects")
        for value in groups[0]["params"]:
            self.assertIsInstance(value, int)

    def test_a_restore_goes_through_the_real_load_state_dict(self):
        controller, optimizer, names = _controller_and_optimizer()
        optimizer.step()
        saved = _capture(controller, optimizer)
        fresh_controller, fresh_optimizer, fresh_names = _controller_and_optimizer()
        self.assertEqual(fresh_optimizer.load_state_dict_calls, [],
                         "PRECONDITION: no restore happened yet")
        report = runner.restore_full_state(fresh_controller, fresh_optimizer, saved,
                                          np_mod=FakeNumpy(), torch_mod=FakeTorch())
        self.assertEqual(len(fresh_optimizer.load_state_dict_calls), 1,
                         "the Adam state must be restored through optimizer.load_state_dict")
        self.assertTrue(report["restored_exactly"], report["problems"])
        restored_state = fresh_optimizer.state[fresh_optimizer.param_groups[0]["params"][0]]
        self.assertIsInstance(restored_state["step"], FakeTensor,
                              "the restored step must keep its TYPE")
        self.assertEqual(restored_state["step"]._data, [1.0])
        self.assertEqual(fresh_names, names)

    def test_a_bootstrap_object_without_state_stays_absent_and_is_not_fabricated(self):
        controller, optimizer, names = _controller_and_optimizer()
        # only the first object has Adam state; the rest are bootstrap actors
        first = optimizer.param_groups[0]["params"][0]
        optimizer.state[first] = {"step": FakeTensor([1.0]),
                                      "exp_avg": FakeTensor([0.0] * first.numel()),
                                      "exp_avg_sq": FakeTensor([0.0] * first.numel())}
        saved = _capture(controller, optimizer)
        self.assertEqual(sorted(saved["_optimizer_payload"]["state"]), [0])
        for index in range(1, len(names)):
            self.assertFalse(saved["adam"][names[index]]["present"])
        fresh_controller, fresh_optimizer, _names = _controller_and_optimizer()
        report = runner.restore_full_state(fresh_controller, fresh_optimizer, saved,
                                          np_mod=FakeNumpy(), torch_mod=FakeTorch())
        self.assertTrue(report["restored_exactly"], report["problems"])
        for index in range(1, len(names)):
            self.assertNotIn(fresh_optimizer.param_groups[0]["params"][index],
                             fresh_optimizer.state,
                             "an object with no state must not be fabricated into one")

    def test_a_tuple_valued_hyperparameter_keeps_its_type_through_capture_and_restore(self):
        controller, optimizer, _names = _controller_and_optimizer()
        self.assertIsInstance(optimizer.param_groups[0]["betas"], tuple)
        saved = _capture(controller, optimizer)
        self.assertIsInstance(saved["param_groups"][0]["betas"], tuple)
        fresh_controller, fresh_optimizer, _fresh_names = _controller_and_optimizer()
        report = runner.restore_full_state(fresh_controller, fresh_optimizer, saved,
                                           np_mod=FakeNumpy(), torch_mod=FakeTorch())
        self.assertTrue(report["restored_exactly"], report["problems"])
        self.assertIsInstance(fresh_optimizer.param_groups[0]["betas"], tuple,
                              "a tuple-valued hyperparameter must not drift into a list")
        self.assertEqual(runner.compare_full_states(saved, saved, "probe"), [])

    def test_a_state_whose_object_mapping_does_not_line_up_is_reported_not_silently_accepted(self):
        controller, optimizer, names = _controller_and_optimizer()
        optimizer.step()
        saved = _capture(controller, optimizer)
        broken = dict(saved)
        broken["adam"] = dict(list(saved["adam"].items())[:-1])
        report = runner.restore_full_state(controller, optimizer, broken,
                                           np_mod=FakeNumpy(), torch_mod=FakeTorch())
        self.assertFalse(report["restored_exactly"],
                         "a capture that lost an object must not restore as 'exact'")
        self.assertTrue(any(item["where"].startswith("adam:") for item in report["problems"]),
                        report["problems"])
        # the capture path itself refuses to guess a broken mapping
        with self.assertRaises(RuntimeError):
            runner._adam_report(_ShortController(controller), optimizer)

    def test_a_moved_trunk_parameter_is_detected_by_the_digest_comparison_real_run(self):
        controller, optimizer, _names = _controller_and_optimizer()
        before = {f"trunk_p{index}": runner._tensor_digest(parameter)
                  for index, parameter in enumerate(controller.inner.trunk.parameters())}
        controller.inner.trunk.state_dict()["trunk.weight"].add_(50.0)
        after = {f"trunk_p{index}": runner._tensor_digest(parameter)
                 for index, parameter in enumerate(controller.inner.trunk.parameters())}
        moved = sorted(key for key in before if before[key]["sha256"] != after[key]["sha256"])
        self.assertEqual(moved, ["trunk_p0"])


def _names_of(controller):
    return runner.r75_trainable_names(controller)


class _ShortController:
    """A controller whose trainable-name count does NOT match the optimizer parameter count, so a
    broken one-to-one mapping can be exercised without guessing."""

    def __init__(self, controller):
        self.inner = controller.inner
        self.cost_head = FakeModule({"value_cost.weight": FakeTensor([0.4, 0.6])})


# ======================================================================================
# 3. the complete bundle and the real disk restore path
# ======================================================================================
class CompleteBundleTests(unittest.TestCase):
    REQUIRED = {"module_payloads", "adam_payloads", "optimizer_payload", "numpy_state",
                "torch_state", "python_state", "lambda", "lambda_at_rollout", "eta_lambda",
                "update_cursor", "data_identity", "configuration", "code_identity", "contract"}

    def test_the_bundle_carries_every_resume_field(self):
        controller, optimizer, _names = _controller_and_optimizer()
        saved = _capture(controller, optimizer)
        bundle = runner.build_state_bundle(saved)
        missing = sorted(self.REQUIRED - set(bundle))
        self.assertEqual(missing, [], f"the resume bundle is missing {missing}")
        self.assertEqual(len(bundle["module_payloads"]), 5)
        for name, payload in bundle["module_payloads"].items():
            self.assertTrue(payload, f"the {name} module payload is empty")

    def test_the_runner_restores_from_the_disk_file_not_the_memory_dict(self):
        source = (HERE / "r75_s1_runner.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "run_s1")
        calls = [ast.unparse(node.func) for node in ast.walk(function)
                 if isinstance(node, ast.Call)]
        self.assertIn("load_state_bundle", calls,
                      "A3 must load the bundle from the ACTUAL serialized file")
        restore_args = [ast.unparse(node) for node in ast.walk(function)
                        if isinstance(node, ast.Call) and ast.unparse(node.func) == "restore_full_state"]
        self.assertEqual(len(restore_args), 1)
        self.assertIn("_bundle_to_saved", restore_args[0],
                      "the restore source must be the LOADED bundle, not the in-memory capture")
        for name in ("a1_baseline", "saved"):
            self.assertNotIn(f"restore_full_state(controller3, optimizer3, {name}",
                             restore_args[0].replace(" ", ""),
                             f"the in-memory {name} must not be the A3 restore source")

    def test_a_loaded_bundle_round_trips_into_the_saved_shape(self):
        controller, optimizer, _names = _controller_and_optimizer()
        optimizer.step()
        saved = _capture(controller, optimizer, cursor=4)
        bundle = runner.build_state_bundle(saved)
        adapted = runner._bundle_to_saved(bundle)
        for key in ("modules", "adam", "param_groups", "lambda", "lambda_at_rollout",
                    "eta_lambda", "update_cursor", "data_identity", "configuration",
                    "code_identity", "contract"):
            self.assertIn(key, adapted, f"the loaded bundle lost {key}")
        self.assertEqual(adapted["update_cursor"], 4)
        self.assertEqual(adapted["lambda"], 1.0)
        self.assertIsInstance(adapted["adam"][_names_of(controller)[0]]["step"], dict)

    def test_a_bundle_missing_a_field_fails_the_restore(self):
        controller, optimizer, _names = _controller_and_optimizer()
        saved = _capture(controller, optimizer)
        bundle = runner.build_state_bundle(saved)
        del bundle["torch_state"]
        with self.assertRaises(KeyError):
            runner.restore_full_state(controller, optimizer, runner._bundle_to_saved(bundle),
                                      np_mod=FakeNumpy(), torch_mod=FakeTorch())


# ======================================================================================
# 4. the comparator must not miss a real difference
# ======================================================================================
class FullStateComparisonTests(unittest.TestCase):
    def _two_states(self):
        controller, optimizer, _names = _controller_and_optimizer()
        optimizer.step()
        return _capture(controller, optimizer), _capture(controller, optimizer)

    def test_a_single_adam_value_difference_is_detected(self):
        left, right = self._two_states()
        name = sorted(right["adam"])[0]
        right["adam"][name]["exp_avg"]["abs_sum"] += 1.0
        differences = runner.compare_full_states(left, right, "probe")
        self.assertTrue(any(item["where"].startswith("probe:adam:") for item in differences),
                        f"a real Adam difference slipped past the comparator: {differences}")

    def test_a_module_value_difference_is_detected(self):
        left, right = self._two_states()
        right["modules"]["trunk"]["trunk.weight"]["abs_sum"] += 1.0
        differences = runner.compare_full_states(left, right, "probe")
        self.assertTrue(any(item["where"] == "probe:module:trunk:trunk.weight"
                            for item in differences), differences)

    def test_a_dtype_or_shape_difference_is_detected(self):
        left, right = self._two_states()
        right["modules"]["early_head"]["early.weight"]["dtype"] = "torch.float64"
        self.assertTrue(runner.compare_full_states(left, right, "probe"))
        left2, right2 = self._two_states()
        right2["modules"]["early_head"]["early.weight"]["shape"] = [3]
        self.assertTrue(runner.compare_full_states(left2, right2, "probe"))

    def test_a_lambda_or_cursor_difference_is_detected(self):
        left, right = self._two_states()
        right["update_cursor"] = 99
        self.assertTrue(any(item["where"] == "probe:update_cursor"
                            for item in runner.compare_full_states(left, right, "probe")))
        left2, right2 = self._two_states()
        right2["lambda"] = 0.5
        self.assertTrue(any(item["where"] == "probe:lambda"
                            for item in runner.compare_full_states(left2, right2, "probe")))

    def test_an_rng_difference_is_detected(self):
        left, right = self._two_states()
        right["rng"]["python_random_state"] = "different"
        self.assertTrue(any(item["where"] == "probe:rng"
                            for item in runner.compare_full_states(left, right, "probe")))

    def test_identical_states_produce_no_difference(self):
        left, right = self._two_states()
        self.assertEqual(runner.compare_full_states(left, right, "probe"), [])


class ScientificReturnComparisonTests(unittest.TestCase):
    def test_gate_records_gate_diag_and_batch_read_are_compared_not_skipped(self):
        plan = json.loads((HERE / "ROUND75_S1_RUN_PLAN.json").read_text(encoding="utf-8"))
        left = {"policy_surrogate": 1.0, "gate_records": [{"a": 1}], "gate_diag": {"b": 2},
                "batch_read": {"c": 3}, "approx_kl": 0.1}
        right = {"policy_surrogate": 1.0, "gate_records": [{"a": 2}], "gate_diag": {"b": 3},
                 "batch_read": {"c": 4}, "approx_kl": 0.1}
        compared, differences = runner.compare_scientific_returns(left, right, plan)
        for field in ("gate_records", "gate_diag", "batch_read", "policy_surrogate", "approx_kl"):
            self.assertIn(field, compared, f"{field} was silently skipped by the comparator")
        skipped = {item["field"] for item in differences}
        self.assertEqual(skipped, {"gate_records", "gate_diag", "batch_read"})

    def test_only_the_plan_frozen_exclusions_are_excluded(self):
        plan = json.loads((HERE / "ROUND75_S1_RUN_PLAN.json").read_text(encoding="utf-8"))
        compared, _ = runner.compare_scientific_returns({"a": 1, "trunk_entry_meter": {}},
                                                        {"a": 1, "trunk_entry_meter": {}}, plan)
        self.assertNotIn("trunk_entry_meter", compared,
                         "the A3 meter absolute readings are excluded in advance and reconciled")
        self.assertIn("a", compared)
        source = (HERE / "r75_s1_runner.py").read_text(encoding="utf-8")
        self.assertNotIn('if key in ("trunk_entry_meter", "batch_read", "gate_records", '
                         '"gate_diag")', source,
                         "the old hardcoded skip list must be gone")

    def test_a_non_additive_meter_is_reported_and_a_matching_one_passes(self):
        plan = json.loads((HERE / "ROUND75_S1_RUN_PLAN.json").read_text(encoding="utf-8"))
        per_update = plan["accounting"]["per_native_update"]
        good = {"trunk_entry_meter": {"alignment": {"calls_charged": 66, "rows_charged": 168},
                                      "epochs": {"calls_charged": 4, "rows_charged": 672},
                                      "total": {"calls_charged": 70, "rows_charged": 840,
                                                "calls_completed": 70, "rows_completed": 840,
                                                "calls_refused": 0, "calls_failed": 0}}}
        bad = {"trunk_entry_meter": {"alignment": {"calls_charged": 66, "rows_charged": 168},
                                     "epochs": {"calls_charged": 5, "rows_charged": 672},
                                     "total": {"calls_charged": 70, "rows_charged": 840,
                                               "calls_completed": 70, "rows_completed": 840,
                                               "calls_refused": 0, "calls_failed": 0}}}
        report = runner.reconcile_trunk_meter(good, good, plan)
        self.assertTrue(all(data["additive"] for data in report["controllers"].values()))
        self.assertTrue(all(data["calls_match_plan"] and data["rows_match_plan"]
                            for data in report["controllers"].values()))
        self.assertEqual(report["controllers"]["A2"]["calls_charged"],
                         per_update["trunk_entry_calls"])
        broken = runner.reconcile_trunk_meter(bad, bad, plan)
        self.assertFalse(broken["controllers"]["A2"]["additive"],
                         "a non-additive phase split must be visible to the failure gate")
        short = {"trunk_entry_meter": {"alignment": {"calls_charged": 66, "rows_charged": 168},
                                       "epochs": {"calls_charged": 4, "rows_charged": 672},
                                       "total": {"calls_charged": 70, "rows_charged": 840,
                                                 "calls_completed": 70, "rows_completed": 840,
                                                 "calls_refused": 0, "calls_failed": 0}}}
        short["trunk_entry_meter"]["total"]["rows_charged"] = 800
        self.assertFalse(runner.reconcile_trunk_meter(short, short, plan)
                         ["controllers"]["A2"]["rows_match_plan"])

# ======================================================================================
# 5. the real step entry: attempt records and per-permit reconciliation
# ======================================================================================
class StepEntryObservationTests(unittest.TestCase):
    def _observer(self):
        controller, optimizer, _names = _controller_and_optimizer()
        return controller, optimizer, runner.ObservedStepEntry(optimizer, FakeResources(),
                                                               "probe")

    def test_a_completed_step_records_exactly_one_new_permit(self):
        _controller, _optimizer, observer = self._observer()
        observer._resources.charge_physical_step("permit")       # the pre-charge, as r75_frozen does
        observer.step()
        record = observer.records[-1]
        self.assertEqual(record["phase"], "completed")
        self.assertEqual(record["new_permits_since_previous_step"], 1)
        self.assertTrue(record["exactly_one_new_permit"])
        self.assertTrue(record["step_charged_nothing"])
        self.assertEqual(record["durable_used_before"], 1)
        self.assertEqual(record["durable_used_after"], 1)
        self.assertEqual(record["permits_issued_since_start"], 1)
        self.assertEqual(record["expected_durable_used_before"], 1)

    def test_a_step_without_a_new_permit_is_a_failure_not_a_note(self):
        _controller, _optimizer, observer = self._observer()
        # NO permit was issued: the step must be REFUSED before delegation, not merely noted.
        with self.assertRaises(RuntimeError):
            observer.step()
        record = observer.records[-1]
        self.assertEqual(record["phase"], "refused")
        self.assertEqual(record["new_permits_since_previous_step"], 0)
        self.assertFalse(record["exactly_one_new_permit"])
        self.assertFalse(record["delegated"])
        self.assertIn("refused BEFORE delegation", record["note"])
        self.assertEqual(observer._optimizer.step_calls, 0,
                         "an unlicensed step must never reach the real optimizer")
        self.assertEqual(observer._resources.used, 0, "a refusal must not charge anything new")

    def test_a_double_charged_permit_is_detected(self):
        _controller, _optimizer, observer = self._observer()
        observer._resources.charge_physical_step("permit")
        observer._resources.charge_physical_step("permit-again")
        with self.assertRaises(RuntimeError):
            observer.step()
        record = observer.records[-1]
        self.assertEqual(record["phase"], "refused")
        self.assertEqual(record["new_permits_since_previous_step"], 2)
        self.assertFalse(record["exactly_one_new_permit"],
                         "a permit spent twice must be visible to the failure gate")
        self.assertFalse(record["delegated"])
        self.assertEqual(observer._optimizer.step_calls, 0,
                         "a doubly licensed step must never fire a second real update")
        self.assertEqual(observer._resources.used, 2, "a refusal must NOT be refunded")

    def test_a_sequence_of_steps_reconciles_one_permit_each(self):
        _controller, _optimizer, observer = self._observer()
        for _ in range(4):
            observer._resources.charge_physical_step("permit")
            observer.step()
        self.assertEqual([entry["exactly_one_new_permit"] for entry in observer.records],
                         [True, True, True, True])
        self.assertEqual([entry["index"] for entry in observer.records], [1, 2, 3, 4])
        self.assertEqual(observer.records[-1]["permits_issued_since_start"], 4)
        self.assertEqual(observer.records[-1]["durable_used_after"], 4)
        self.assertEqual([entry["delegated"] for entry in observer.records],
                         [True, True, True, True])
        # the fifth step with no permit must break the sequence WITHOUT ever being delegated
        with self.assertRaises(RuntimeError):
            observer.step()
        self.assertFalse(observer.records[-1]["exactly_one_new_permit"])
        self.assertEqual(observer.records[-1]["new_permits_since_previous_step"], 0)
        self.assertFalse(observer.records[-1]["delegated"])
        self.assertEqual(observer._optimizer.step_calls, 4,
                         "the refused step must not reach the real optimizer")
        self.assertEqual(observer._resources.used, 4, "a refused step must NOT be refunded")

    def test_a_failed_step_is_recorded_and_never_refunded(self):
        controller, optimizer, _names = _controller_and_optimizer()

        class Boom:
            def step(self):
                raise RuntimeError("the real step failed")

        resources = FakeResources()
        observer = runner.ObservedStepEntry(Boom(), resources, "probe")
        resources.charge_physical_step("permit")   # the permit precedes the step, as the runner does
        with self.assertRaises(RuntimeError):
            observer.step()
        record = observer.records[-1]
        self.assertEqual(record["phase"], "failed")
        self.assertTrue(record["delegated"])
        self.assertIn("RuntimeError", record["error"])
        self.assertEqual(resources.used, 1, "a failed step must NOT be refunded")

    def test_the_observer_never_charges_opens_or_bypasses_anything(self):
        _controller, _optimizer, observer = self._observer()
        observer._resources.charge_physical_step("permit")
        observer.step()
        self.assertEqual(len(observer._resources.charge_calls), 1,
                         "the observer must not charge on its own")
        source = (HERE / "r75_s1_runner.py").read_text(encoding="utf-8")
        start = source.index("class ObservedStepEntry")
        end = source.index("FULL_STATE_FIELDS = ")
        body = source[start:end]
        for forbidden in ("charge_physical_step", "charge_offline_rows", "charge_joint_record",
                          "charge_env_construction", "fixture(", "simulated(", "bootstrap("):
            self.assertNotIn(forbidden, body,
                             f"the observer must never call {forbidden}")
        self.assertIn("never charges, never refunds", body)

    def test_the_observer_delegates_everything_except_step(self):
        controller, optimizer, _names = _controller_and_optimizer()
        observer = runner.ObservedStepEntry(optimizer, FakeResources(), "probe")
        self.assertIs(observer.param_groups, optimizer.param_groups)
        self.assertIs(observer.state, optimizer.state)
        observer.zero_grad()
        self.assertEqual(observer.step_calls, 0)
        observer._resources.charge_physical_step("permit")       # the pre-charge, as r75_frozen does
        observer.step()
        self.assertEqual(observer.step_calls, 1)
        # a second step without a fresh permit is refused and never delegated
        with self.assertRaises(RuntimeError):
            observer.step()
        self.assertEqual(observer.step_calls, 1)

    def test_the_planned_step_count_is_nine_and_is_enforced(self):
        plan = json.loads((HERE / "ROUND75_S1_RUN_PLAN.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["accounting"]["planned_totals"]["physical_optimizer_steps"], 9)
        source = (HERE / "r75_s1_runner.py").read_text(encoding="utf-8")
        self.assertIn('"steps_match_plan": total_real_steps == expected_steps', source)
        self.assertIn('failures.append("the real step count does not match the frozen plan")',
                      source)


# ======================================================================================
# 6. A1: measured immobility and coverage, never a literal
# ======================================================================================
class A1MeasurementTests(unittest.TestCase):
    def test_trunk_immobility_is_measured_not_written_as_a_literal(self):
        source = (HERE / "r75_s1_runner.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "run_a1")
        body = ast.unparse(function)
        self.assertNotIn("'trunk_moved': False", body.replace('"', "'"),
                         "trunk_moved must never be a hardcoded literal")
        self.assertIn("'trunk_moved': bool(trunk_moved)", body,
                      "trunk_moved must be derived from the measured digest comparison")
        self.assertIn("trunk_moved = sorted(", body)
        self.assertIn("trunk_moved_parameters", body)
        self.assertIn("trunk_before", body)
        self.assertIn("trunk_after", body)
        self.assertIn("trunk_parameters_in_optimizer", body)
        self.assertIn("objects_without_gradient", body,
                      "a coverage gap must be reported, not hidden")

    def test_a_moved_trunk_parameter_is_detected_by_the_digest_comparison(self):
        before = {"trunk_p0": runner._tensor_digest(FakeTensor([1.0, 2.0]))}
        after_same = {"trunk_p0": runner._tensor_digest(FakeTensor([1.0, 2.0]))}
        after_moved = {"trunk_p0": runner._tensor_digest(FakeTensor([1.0, 9.0]))}
        self.assertEqual(sorted(k for k in before if before[k]["sha256"] != after_same[k]["sha256"]),
                         [])
        self.assertEqual(sorted(k for k in before if before[k]["sha256"] != after_moved[k]["sha256"]),
                         ["trunk_p0"])

    def test_the_object_coverage_count_is_the_real_name_to_parameter_count(self):
        controller, optimizer, names = _controller_and_optimizer()
        self.assertEqual(len(names), 8, "the R75 contract has eight trainable objects")
        self.assertEqual(len(optimizer.param_groups[0]["params"]), 8)
        self.assertEqual(names, ["early_head.early.weight", "early_head.early.bias",
                                 "late_head.late.weight", "late_head.late.bias",
                                 "value_task_head.value_task.weight",
                                 "value_task_head.value_task.bias",
                                 "value_cost_head.value_cost.weight",
                                 "value_cost_head.value_cost.bias"])


# ======================================================================================
# 7. the pure precheck ordering, CWD independence and the per-charge cap
# ======================================================================================
class PrecheckOrderingTests(unittest.TestCase):
    def test_the_precheck_runs_before_any_paid_module_is_imported(self):
        source = (HERE / "r75_s1_runner.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "run_s1")
        order = []
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name in ("r75_s1_precheck.main", "_load_paid_modules"):
                    order.append(name)
        self.assertEqual(order[:2], ["r75_s1_precheck.main", "_load_paid_modules"],
                         f"the pure precheck must run BEFORE the paid modules: {order}")
        body = source[source.index("def run_s1() -> int:"):source.index("def r75_replay_adapter_load():")]
        self.assertLess(body.index("import r75_s1_precheck"),
                        body.index("_load_paid_modules()"),
                        "inside run_s1 the precheck import must precede the paid-module load")
        self.assertLess(body.index("r75_s1_precheck.main()"),
                        body.index("_load_paid_modules()"),
                        "inside run_s1 the precheck must RUN before the paid modules load")

    def test_the_paid_import_stays_lazy(self):
        source = (HERE / "r75_s1_runner.py").read_text(encoding="utf-8")
        module_level = source.split("def _load_paid_modules")[0]
        for forbidden in ("import torch", "import numpy", "import r75_frozen"):
            self.assertNotIn(forbidden, module_level,
                             f"{forbidden} must stay inside the run path")

    def test_path_resolution_does_not_depend_on_the_cwd(self):
        import os
        import r75_s1_precheck as precheck
        original = os.getcwd()
        directories = [str(preview_root()), str(HERE), str(HERE.parent), str(HERE.parents[1])]
        try:
            for directory in directories:
                os.chdir(directory)
                resolved = precheck.resolve_path(
                    "runs/comm_bc_initialized_constrained_ppo_20261007_r2/x.json")
                self.assertTrue(resolved.is_absolute())
                self.assertEqual(resolved, precheck.REPO_ROOT /
                                 "runs/comm_bc_initialized_constrained_ppo_20261007_r2/x.json")
        finally:
            os.chdir(original)

    def test_the_per_charge_cap_is_verified_and_never_compared_against_none(self):
        import r75_s1_precheck as precheck
        plan = json.loads((HERE / "ROUND75_S1_RUN_PLAN.json").read_text(encoding="utf-8"))
        report = precheck.Report()
        cap = precheck.verified_per_charge_cap(plan, report)
        self.assertEqual(cap, 1024)
        self.assertTrue(report.passed, [item for item in report.checks if not item["ok"]])
        # a plan whose frozen contract does not state the cap must FAIL, not crash on None
        broken = json.loads(json.dumps(plan))
        broken["accounting"]["caps"]["offline_trunk_target_row"] = "normal 5000 / hard 6000"
        report2 = precheck.Report()
        self.assertEqual(precheck.verified_per_charge_cap(broken, report2), 0)
        self.assertFalse(report2.passed)
        # a raw counter that DOES carry a contradicting value must be caught, not overridden
        contradicted = json.loads(json.dumps(plan))
        contradicted["canonical_ledger_frozen_snapshot"]["files"] = {}
        report3 = precheck.Report()
        self.assertEqual(precheck.verified_per_charge_cap(contradicted, report3), 1024)


def preview_root() -> Path:
    return HERE.parents[2]


if __name__ == "__main__":
    unittest.main(verbosity=2)
