"""R76 A0/A1 fixed eight-case source review suite (pure stdlib only)."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import r76_cli
import r76_contracts as contracts
import r76_episode_protocol as protocol
import r76_runtime

HERE = Path(__file__).resolve().parent
R75_DIR = HERE.parent / "round75_native_ppo_qualification"


def function_ast(path: Path, name: str) -> ast.AST:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"missing function {name} in {path}")


def registered_identity(seed: int) -> dict:
    path = contracts.R71_INIT_ROOT / f"r71_init_identity_s{seed}.json"
    identity = json.loads(path.read_text(encoding="utf-8"))
    return identity


def r71_source_path(seed: int) -> Path:
    filename, _bytes, _sha256 = contracts.R71_INIT_FILES[int(seed)]
    return contracts.R71_INIT_ROOT / filename


def init_payload(seed: int) -> dict:
    return {
        "schema": contracts.R71_STATE_SCHEMA,
        "trainer_tag": contracts.R71_TRAINER_TAG,
        "training_version": contracts.R71_TRAINING_VERSION,
        "loss_version": contracts.R71_LOSS_VERSION,
        "optimizer_schema": contracts.R71_OPTIMIZER_SCHEMA,
        "structure": contracts.R71_STRUCTURE,
        "controller": {name: {"seed": seed} for name in contracts.R76_MODULES},
        "optimizer": {"param_groups": [{"params": [1, 2, 3, 4, 5, 6]}]},
        "frozen_mask": {
            "frozen_modules": ["trunk", "early_head"],
            "trainable_modules": ["late_head", "value_task_head", "value_cost_head"],
            "frozen_parameter_object_count": 6,
            "trainable_parameter_object_count": 6,
            "frozen_scalar_count": 5570,
            "trainable_scalar_count": 260,
            "critic_heads_calibrated": True,
            "frozen_sha256": "0" * 64,
            "note": "model-only payload mask retains its payload-only identity fields",
        },
        "training_state": {"training_seed": seed, "episodes_done": 0, "updates": 0,
                           "optimizer_steps": 0, "lambda": 1.0},
        "rng": {"python": "initial", "numpy": "initial", "torch": "initial"},
    }


class FakeController:
    def __init__(self, seed: int):
        self.seed = seed
        self.loaded = None
        self.unfrozen = False

    def load_state_dict(self, value):
        self.loaded = value


class FakeNamedModule:
    def __init__(self, *names):
        self.names = list(names)

    def named_parameters(self):
        for name in self.names:
            yield name, object()


def make_fake_runtime_cfg():
    class Common:
        GAMMA = 1.0
        GAE_LAMBDA = 1.0
        ETA_LAMBDA = 0.5
        B0_BYTES = 10000.0
        DATA_ORDER_SEED = 20260927
        EPOCHS_PER_BATCH = 4
        MINIBATCH_STEPS = 256
        LEARNING_RATE = 3e-4

    class Trainer:
        def save_checkpoint(self, controller, optimizer, state, path, *, freeze=None):
            raise AssertionError("A1 must not serialize a checkpoint")

        def load_checkpoint(self, path, *, freeze=None):
            raise AssertionError("A1 must not deserialize a checkpoint")

        def rng_state(self):
            return {"python": "p", "numpy": "n", "torch": "t"}

        def restore_rng(self, state):
            raise AssertionError("A1 must not restore RNG")

        def batch_samples(self, episodes):
            return []

        def _unassigned_prefix_totals(self, episodes):
            return 0.0, 0.0

        ppo_update = object()

    class Runner:
        def _run_episode(self, *args, **kwargs):
            raise AssertionError("A1 must not run an episode")

    class R66:
        READY_CUT = 147.0
        HORIZON = 1000.0
        GATE_BETA = 1.0

    class R68:
        LATE_INIT_ALPHA = 0.10

    class FakeResources:
        mode = "production"
        root = HERE / "raw" / "runtime"

        def charge_joint_record(self, n=1, what="joint_record"):
            raise AssertionError("A1 must not charge")

        def ensure_trunk_meter(self, controller, *, what_prefix):
            raise AssertionError("A1 must not meter")

        def charge_physical_step(self, what, *, category="normal"):
            raise AssertionError("A1 must not charge")

        def charge_offline_rows(self, n, what, *, category="normal"):
            raise AssertionError("A1 must not charge")

        def verify_or_fail(self):
            return {}

    class FakeNative:
        def __init__(self):
            self.ppo_calls = []
            self.optimizer_calls = []

        @staticmethod
        def build_optimizer(controller, lr):
            raise AssertionError("A1 must not fabricate an optimizer")

        @staticmethod
        def r75_native_ppo_update(controller, optimizer, episodes, *, update_index,
                                  resources, physical_step_category):
            return {"controller": controller, "optimizer": optimizer,
                    "episodes": list(episodes), "update_index": update_index,
                    "resources": resources,
                    "physical_step_category": physical_step_category}

        @staticmethod
        def module_sha(controller, name):
            return f"sha-{name}"

    old_r66 = sys.modules.get("r66_frozen")
    old_r68 = sys.modules.get("r68_frozen")
    sys.modules["r66_frozen"] = R66
    sys.modules["r68_frozen"] = R68
    return {
        "trainer": Trainer(), "common": Common(), "runner": Runner(),
        "runtime": {}, "controller_module": FakeController(11), "optimizer": None,
    }, FakeResources(), FakeNative, (old_r66, old_r68)


class R76A0SourceReviewTests(unittest.TestCase):
    def test_1_native_scientific_ast_is_r75_native_update(self):
        old = function_ast(R75_DIR / "r75_frozen.py", "r75_native_ppo_update")
        new = function_ast(HERE / "r76_native.py", "r75_native_ppo_update")
        self.assertEqual(ast.dump(old, include_attributes=False),
                         ast.dump(new, include_attributes=False))
        source = (HERE / "r76_native.py").read_text(encoding="utf-8")
        self.assertIn("import r76_guard", source)
        self.assertNotIn("import r75_guard", source)

    def test_2_model_only_r71_init_adapter_verifies_before_unfreeze(self):
        payload = init_payload(11)
        identity = registered_identity(11)
        source_file = r71_source_path(11)
        actual_source_sha = r76_runtime.sha256_file(source_file)
        self.assertEqual(
            actual_source_sha, contracts.R71_INIT_FILES[11][2],
            "the registered INIT file must be verified, never deserialized, before use")
        events = []

        class EventController(FakeController):
            def load_state_dict(self, value):
                events.append("load")
                super().load_state_dict(value)

        def factory(seed):
            events.append("factory")
            return EventController(seed)

        def module_sha(controller, name):
            events.append(
                "hash_before_unfreeze" if not controller.unfrozen
                else "hash_after_unfreeze")
            return identity["calibration"][contracts.R71_CALIBRATION_SHA_KEYS[name]]

        def freeze(controller):
            controller.unfrozen = True
            events.append("freeze")
            return {
                "frozen_modules": list(contracts.R76_FROZEN_MODULES),
                "trainable_modules": list(contracts.R76_TRAINABLE_MODULES),
                "frozen_parameter_object_count": 4,
                "trainable_parameter_object_count": 8,
                "frozen_scalar_count": 5440,
                "trainable_scalar_count": 390,
            }

        controller, report = r76_runtime.build_r76_controller_from_r71_init(
            payload, identity, seed=11, controller_factory=factory,
            module_sha=module_sha, freeze_trunk_only=freeze, source_file=source_file)
        self.assertEqual(events, ["factory"]
                         + ["load"]
                         + ["hash_before_unfreeze"] * len(contracts.R76_MODULES)
                         + ["freeze"]
                         + ["hash_after_unfreeze"] * len(contracts.R76_MODULES))
        self.assertEqual(report["source_sha256"], actual_source_sha)
        self.assertTrue(controller.unfrozen)
        self.assertEqual(report["mask"]["trainable_parameter_object_count"], 8)
        self.assertFalse(report["rescaled"])
        bad = dict(payload)
        bad["training_state"] = dict(payload["training_state"], episodes_done=1)
        with self.assertRaisesRegex(RuntimeError, "untrained"):
            r76_runtime.build_r76_controller_from_r71_init(
                bad, identity, seed=11, controller_factory=factory,
                module_sha=module_sha, freeze_trunk_only=freeze, source_file=source_file)
        drifted_identity = json.loads(json.dumps(identity))
        drifted_identity["calibration"]["trunk_sha256"] = "f" * 64
        with self.assertRaisesRegex(RuntimeError, "module values drifted"):
            r76_runtime.build_r76_controller_from_r71_init(
                payload, drifted_identity, seed=11, controller_factory=factory,
                module_sha=module_sha, freeze_trunk_only=freeze, source_file=source_file)

        refusing_factory_calls = []

        def refusing_factory(seed):
            refusing_factory_calls.append(seed)
            raise AssertionError("bad INIT source must be rejected before architecture construction")

        with self.assertRaisesRegex(RuntimeError, "untrained"):
            r76_runtime.build_r76_controller_from_r71_init(
                bad, identity, seed=11, controller_factory=refusing_factory,
                module_sha=module_sha, freeze_trunk_only=freeze, source_file=source_file)
        self.assertEqual(refusing_factory_calls, [])

    def test_3_runtime_install_binds_real_cfg_objects(self):
        cfg, resources, native, _old_modules = make_fake_runtime_cfg()
        old = cfg["trainer"].ppo_update
        identity = r76_runtime.install_r76_runtime_bindings(
            cfg, resources, native, request_id=contracts.REQUEST_ID)
        self.assertIs(cfg["r76"]["resources"], resources)
        self.assertIs(cfg["r76_module_sha"], native.module_sha)
        self.assertTrue(identity["module_sha_is_r76"])
        self.assertTrue(identity["ppo_replaced"])
        self.assertTrue(identity["save_replaced"])
        self.assertTrue(identity["load_replaced"])
        self.assertIsNot(cfg["trainer"].ppo_update, old)
        self.assertIsNone(cfg.get("optimizer"))
        delegated = cfg["trainer"].ppo_update("controller", "optimizer", ["episode"],
                                              update_index=7)
        self.assertEqual(delegated["controller"], "controller")
        self.assertEqual(delegated["optimizer"], "optimizer")
        self.assertEqual(delegated["episodes"], ["episode"])
        self.assertEqual(delegated["update_index"], 7)
        self.assertIs(delegated["resources"], resources)
        self.assertEqual(delegated["physical_step_category"], "normal")
        bad = make_fake_runtime_cfg()[0]
        bad["common"].GAE_LAMBDA = 0.95
        with self.assertRaisesRegex(RuntimeError, "GAE_LAMBDA"):
            r76_runtime.install_r76_runtime_bindings(
                bad, resources, native, request_id=contracts.REQUEST_ID)
        with self.assertRaisesRegex(RuntimeError, "request id"):
            r76_runtime.install_r76_runtime_bindings(
                cfg, resources, native, request_id="not-the-request")

    def test_4_parameter_partition_and_optimizer_coverage_are_r76_exact(self):
        native_source = (HERE / "r76_native.py").read_text(encoding="utf-8")
        for name in ("partition", "assert_mask", "freeze_trunk_only",
                     "build_optimizer"):
            self.assertIsNotNone(function_ast(HERE / "r76_native.py", name))
        self.assertIn("FROZEN_MODULES = (\"trunk\",)", native_source)
        self.assertIn("TRAINABLE_MODULES = (\"early_head\", \"late_head\", "
                      "\"value_task_head\", \"value_cost_head\")", native_source)
        self.assertIn("clip_grad_norm_(trainable_parameters(controller)", native_source)
        mask = contracts.validate_r76_mask_report({
            "frozen_modules": list(contracts.R76_FROZEN_MODULES),
            "trainable_modules": list(contracts.R76_TRAINABLE_MODULES),
            "frozen_parameter_object_count": 4,
            "trainable_parameter_object_count": 8,
            "frozen_scalar_count": 5440,
            "trainable_scalar_count": 390,
        })
        self.assertEqual(mask["frozen_scalar_count"] + mask["trainable_scalar_count"], 5830)
        self.assertEqual(set(contracts.R76_SOURCE_MANIFEST_NAMES), {
            "r71_trial_execute.py", "r71_runtime_binding.py", "r68_execute.py",
            "r75_frozen.py", "v2_trainer.py", "r42_one_shot.py"})
        import r76_guard
        with self.assertRaisesRegex(RuntimeError, "self-issued training caps"):
            r76_guard.R76Resources(mode="production", training_caps={})
        bad = dict(mask, trainable_parameter_object_count=6)
        with self.assertRaisesRegex(RuntimeError, "trainable_parameter_object_count"):
            contracts.validate_r76_mask_report(bad)

        class Module:
            def __init__(self, *names):
                self.parameters_by_name = {name: object() for name in names}

            def named_parameters(self):
                return list(self.parameters_by_name.items())

            def parameters(self):
                return list(self.parameters_by_name.values())

        class Inner:
            pass

        class MappedController:
            def __init__(self):
                self.inner = Inner()
                self.inner.trunk = Module("weight_a", "bias_a", "weight_b", "bias_b")
                self.inner.early_head = Module("weight", "bias")
                self.inner.late_head = Module("weight", "bias")
                self.inner.value_head = Module("weight", "bias")
                self.cost_head = Module("weight", "bias")
                self.parameters = [parameter for module in (
                    self.inner.trunk, self.inner.early_head, self.inner.late_head,
                    self.inner.value_head, self.cost_head)
                    for parameter in module.parameters_by_name.values()]

        controller = MappedController()
        trainable = controller.parameters[4:]
        optimizer = type("Optimizer", (), {"param_groups": [{"params": trainable}]})()
        expected = ["early_head.weight", "early_head.bias", "late_head.weight",
                    "late_head.bias", "value_task_head.weight", "value_task_head.bias",
                    "value_cost_head.weight", "value_cost_head.bias"]
        self.assertEqual(r76_runtime.parameter_object_mapping(controller, optimizer), expected)
        foreign_optimizer = type("Optimizer", (), {
            "param_groups": [{"params": trainable[:-1] + [object()]}]})()
        with self.assertRaisesRegex(RuntimeError, "outside the trainable set"):
            r76_runtime.parameter_object_mapping(controller, foreign_optimizer)
        trunk_optimizer = type("Optimizer", (), {
            "param_groups": [{"params": [controller.parameters[0]] + trainable[1:]}]})()
        with self.assertRaisesRegex(RuntimeError, "frozen trunk"):
            r76_runtime.parameter_object_mapping(controller, trunk_optimizer)
        duplicate_optimizer = type("Optimizer", (), {
            "param_groups": [{"params": [trainable[0], trainable[0]] + trainable[2:]}]})()
        with self.assertRaisesRegex(RuntimeError, "more than once"):
            r76_runtime.parameter_object_mapping(controller, duplicate_optimizer)

        class FakeTensor:
            def __init__(self, numel):
                self.requires_grad = True
                self._numel = int(numel)

            def numel(self):
                return self._numel

            def requires_grad_(self, value):
                self.requires_grad = bool(value)
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def contiguous(self):
                return self

            def numpy(self):
                return type("BytesLike", (), {"tobytes": lambda value: b"x"})()

        class PureModule:
            def __init__(self, count, numel):
                self._parameters = [(f"parameter_{index}", FakeTensor(numel))
                                    for index in range(count)]

            def parameters(self):
                return [parameter for _name, parameter in self._parameters]

        class PureInner:
            pass

        class PureController:
            def __init__(self):
                self.inner = PureInner()
                self.inner.trunk = PureModule(4, 1360)
                self.inner.early_head = PureModule(2, 48)
                self.inner.late_head = PureModule(2, 48)
                self.inner.value_head = PureModule(2, 48)
                self.cost_head = PureModule(2, 51)
                self.parameters = [
                    parameter
                    for module in (self.inner.trunk, self.inner.early_head,
                                   self.inner.late_head, self.inner.value_head,
                                   self.cost_head)
                    for parameter in module.parameters()
                ]

        native_path = HERE / "r76_native.py"
        pure_namespace = {
            "FROZEN_MODULES": ("trunk",),
            "TRAINABLE_MODULES": ("early_head", "late_head", "value_task_head",
                                  "value_cost_head"),
            "_module_of": lambda value, name: getattr(value.inner, {
                "trunk": "trunk", "early_head": "early_head", "late_head": "late_head",
                "value_task_head": "value_head", "value_cost_head": "cost_head"}[name]),
            "hashlib": hashlib,
            "RuntimeError": RuntimeError,
        }
        for pure_name in ("_module_of", "partition", "assert_mask", "freeze_trunk_only"):
            pure_node = function_ast(native_path, pure_name)
            exec(compile(ast.Module(body=[pure_node], type_ignores=[]),
                         str(native_path), "exec"), pure_namespace)
        pure_controller = PureController()
        pure_mask = pure_namespace["freeze_trunk_only"](pure_controller)
        self.assertEqual(len(pure_controller.parameters), 12)
        self.assertEqual(len(pure_namespace["partition"](pure_controller)[0]), 4)
        self.assertEqual(len(pure_namespace["partition"](pure_controller)[1]), 8)
        self.assertEqual(pure_mask["frozen_parameter_object_count"], 4)
        self.assertEqual(pure_mask["trainable_parameter_object_count"], 8)
        self.assertEqual(pure_mask["frozen_scalar_count"], 5440)
        self.assertEqual(pure_mask["trainable_scalar_count"], 390)

    def test_5_full_resume_contract_rejects_foreign_or_incomplete_state(self):
        optimizer = {"param_groups": [{"params": list(range(8))}]}
        parameter_names = ["early_head.weight", "early_head.bias", "late_head.weight",
                           "late_head.bias", "value_task_head.weight", "value_task_head.bias",
                           "value_cost_head.weight", "value_cost_head.bias"]
        state = r76_runtime.build_r76_state(11, data_order_seed=20260927, b0_bytes=10000.0)
        data_identity = {"data_order_seed": 20260927, "b0_bytes": 10000.0,
                         "episodes_per_seed": 128,
                         "scenario_order_sha256": contracts.sha256_text(
                             [1, 2, 3]),
                         "seeds": [11, 22, 33]}
        reference_identity = {"kind": "round68-reused-r66-immediate-references/1",
                              "reference_row_sha256": {"11": "a" * 64, "22": "b" * 64,
                                                       "33": "c" * 64}}
        config_identity = dict(contracts.R76_LIVE_CONFIG)
        source_identity = {
            "request_id": contracts.REQUEST_ID,
            "task_id": contracts.TASK_ID,
            "module_sha256": {name: "0" * 64 for name in contracts.R76_MODULES},
            "source_manifest": dict(contracts.R76_PINNED_SOURCE_MANIFEST),
            "data_identity": data_identity,
            "reference_identity": reference_identity,
            "config_identity": config_identity,
            "resource_contract_sha256": "d" * 64,
        }
        payload = {
            "schema": contracts.R76_STATE_SCHEMA,
            "trainer_tag": contracts.R76_TRAINER_TAG,
            "training_version": contracts.R76_TRAINING_VERSION,
            "loss_version": contracts.R76_LOSS_VERSION,
            "optimizer_schema": contracts.R76_OPTIMIZER_SCHEMA,
            "structure": contracts.R76_STRUCTURE,
            "contract": contracts.R76_FULL_CONTRACT,
            "controller": {name: {"x": 1} for name in contracts.R76_MODULES},
            "optimizer": optimizer,
            "optimizer_param_groups": [8],
            "frozen_mask": {
                "frozen_modules": list(contracts.R76_FROZEN_MODULES),
                "trainable_modules": list(contracts.R76_TRAINABLE_MODULES),
                "frozen_parameter_object_count": 4,
                "trainable_parameter_object_count": 8,
                "frozen_scalar_count": 5440,
                "trainable_scalar_count": 390,
            },
            "training_state": state,
            "rng": {"python": "p", "numpy": "n", "torch": "t"},
            "parameter_object_names": parameter_names,
            "source_identity": source_identity,
        }
        self.assertEqual(r76_runtime.validate_r76_full_payload(payload)["schema"],
                         contracts.R76_STATE_SCHEMA)
        foreign = dict(payload, schema=contracts.R71_STATE_SCHEMA)
        with self.assertRaisesRegex(RuntimeError, "foreign checkpoint"):
            r76_runtime.validate_r76_full_payload(foreign)
        six = dict(payload, optimizer={"param_groups": [{"params": list(range(6))}]},
                   optimizer_param_groups=[6])
        with self.assertRaisesRegex(RuntimeError, "8-parameter"):
            r76_runtime.validate_r76_full_payload(six)
        missing_rng = dict(payload, rng={"python": "p"})
        with self.assertRaisesRegex(RuntimeError, "R76 RNG"):
            r76_runtime.validate_r76_full_payload(missing_rng)
        bad_run_identity = dict(payload, training_state=dict(state, run_identity={}))
        with self.assertRaisesRegex(RuntimeError, "run identity"):
            r76_runtime.validate_r76_full_payload(bad_run_identity)
        bad_contract_version = dict(payload, training_state=dict(
            state, contract_version="round76-constrained-early-native-full-resume/1-x"))
        with self.assertRaisesRegex(RuntimeError, "contract version"):
            r76_runtime.validate_r76_full_payload(bad_contract_version)
        bad_parameters = dict(payload, parameter_object_names=["trunk.weight"] + parameter_names[1:])
        with self.assertRaisesRegex(RuntimeError, "foreign object"):
            r76_runtime.validate_r76_full_payload(bad_parameters)
        bad_source = dict(payload, source_identity={"request_id": contracts.REQUEST_ID})
        with self.assertRaisesRegex(RuntimeError, "source identity"):
            r76_runtime.validate_r76_full_payload(bad_source)
        drifted_manifest = dict(payload, source_identity=dict(
            source_identity, source_manifest=dict(
                contracts.R76_PINNED_SOURCE_MANIFEST, **{"r68_execute.py": "0" * 64})))
        with self.assertRaisesRegex(RuntimeError, "source manifest is not pinned"):
            r76_runtime.validate_r76_full_payload(drifted_manifest)
        wrong_seed_context = {
            "seed": 22, "data_identity": data_identity,
            "reference_identity": reference_identity, "config_identity": config_identity,
            "resource_contract_sha256": "d" * 64,
        }
        with self.assertRaisesRegex(RuntimeError, "resume context seed"):
            r76_runtime.validate_r76_resume_context(payload, wrong_seed_context)

        import r76_execute

        def write_json_object(path, value):
            path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

        with tempfile.TemporaryDirectory() as temporary:
            runtime_root = Path(temporary)
            rolling = runtime_root / "rolling_identity_fixture.txt"
            rolling.write_bytes(b"rolling-payload")
            rolling_sha = hashlib.sha256(rolling.read_bytes()).hexdigest()

            seed = 11

            def make_batch_records(batch, *, break_bridge=False):
                before_state = {"updates": batch, "episodes_done": batch * 4,
                                "lambda": 1.0}
                after_state = {"updates": batch + 1,
                               "episodes_done": (batch + 1) * 4,
                               "lambda": 1.0}
                before_param = f"p{batch}" + "0" * 62
                after_param = f"p{batch + 1}" + "0" * 62
                before_sha = contracts.sha256_text(before_state)
                after_sha = contracts.sha256_text(after_state)
                if break_bridge:
                    before_param = "f" * 64
                pre = {"seed": seed, "batch": batch, "state": before_state,
                       "state_sha256": before_sha,
                       "parameters_before_sha": before_param,
                       "data_identity": data_identity,
                       "resource_contract_sha256": "d" * 64}
                ready = dict(pre, parameters_sha=before_param)
                post = {"seed": seed, "batch": batch, "state": after_state,
                        "state_sha256": after_sha,
                        "parameters_before_sha": before_param,
                        "parameters_after_sha": after_param,
                        "data_identity": data_identity,
                        "resource_contract_sha256": "d" * 64}
                committed = dict(post, rolling_checkpoint_sha256=(
                    rolling_sha if batch == 1 else f"{batch:x}" * 64))
                write_json_object(r76_execute._batch_path(seed, batch,
                                                          "preupdate", runtime_root), pre)
                write_json_object(r76_execute._batch_path(seed, batch,
                                                          "ready", runtime_root), ready)
                write_json_object(r76_execute._batch_path(seed, batch,
                                                          "postupdate", runtime_root), post)
                write_json_object(r76_execute._batch_path(seed, batch,
                                                          "committed", runtime_root), committed)

            make_batch_records(0)
            make_batch_records(1)
            write_json_object(runtime_root / "r76_first_batch_gate_s11.json",
                              {"status": "passed"})
            resume_state = {"updates": 2, "episodes_done": 8}
            r76_execute._verify_durable_history(
                seed, resume_state, data_identity, "d" * 64, rolling, runtime_root)

            make_batch_records(1, break_bridge=True)
            with self.assertRaisesRegex(RuntimeError, "chain is broken"):
                r76_execute._verify_durable_history(
                    seed, resume_state, data_identity, "d" * 64, rolling, runtime_root)

            make_batch_records(1)
            dangling_state = {"updates": 2, "episodes_done": 8}
            dangling_pre = {
                "seed": seed,
                "batch": 2,
                "state": dangling_state,
                "state_sha256": contracts.sha256_text(dangling_state),
                "parameters_sha": "p2" + "0" * 62,
                "data_identity": data_identity,
                "resource_contract_sha256": "d" * 64,
            }
            write_json_object(r76_execute._batch_path(seed, 2, "ready", runtime_root),
                              dangling_pre)
            with self.assertRaisesRegex(RuntimeError, "dangling"):
                r76_execute._verify_durable_history(
                    seed, resume_state, data_identity, "d" * 64, rolling, runtime_root)
        sys.modules.pop("r76_execute", None)

    def test_6_row_tape_persistence_reconstructs_without_filling_missing_fields(self):
        tape = {
            "schema": protocol.TAPE_SCHEMA,
            "comm_action_semantics": "one-shot-send-skip/1",
            "comm_action_eligibility": "first-report-only/1",
            "base_action_mask_schema": "budget-sequential-mask/1",
            "step": 7,
            "targets": ["target-5", "target-forced", "target-closed"],
            "features": [[0.0] * 19, [1.0] * 19, [2.0] * 19],
            "actions": {"target-5": protocol.SEND, "target-forced": protocol.SEND,
                        "target-closed": protocol.HOLD},
            "effective_actions": {"target-5": protocol.SEND, "target-forced": protocol.SEND,
                                  "target-closed": protocol.HOLD},
            "requested": {"target-5": True, "target-forced": False, "target-closed": False},
            "decision_class": {"target-5": "free_one_shot_choice",
                               "target-forced": "transport_forced",
                               "target-closed": "fixed_hold_one_shot_closed"},
            "free_one_shot_choices": ["target-5"],
            "free_first_reports": ["target-5"],
            "free": ["target-5"],
            "actionable": ["target-5"],
            "masked": [],
            "masked_reasons": {},
            "fixed_hold_one_shot_closed": ["target-closed"],
            "fixed_hold_refresh": [],
            "fixed_hold_reasons": {},
            "forced": ["target-forced"],
            "attempt_charge_bytes": {"target-5": 100},
            "send_executable_initial": {"target-5": True, "target-forced": True,
                                        "target-closed": True},
            "send_block_reason_initial": {"target-5": None, "target-forced": None,
                                          "target-closed": None},
            "candidate_audit": {
                "target-5": {
                    "first_emission": True, "attempt_charge_bytes": 100,
                    "sequentially_affordable_before_action": True,
                    "effective_block_reason": "eligible",
                    "one_shot_state_after": "send_now_chosen",
                    "effective_action": protocol.SEND,
                    "shadow_charge_applied": True,
                },
                "target-closed": {
                    "first_emission": True,
                    "one_shot_state_before": "send_now_chosen",
                    "effective_action": protocol.HOLD,
                    "fixed_reason": "one-shot-choice-already-made",
                    "shadow_charge_applied": False,
                },
            },
            "relay_observation_remaining_bytes": 29200.0,
            "shadow_budget_before": 29300.0,
            "shadow_budget_after": 29200.0,
            "shadow_capacity_before": 10,
            "shadow_capacity_after": 9,
            "log_prob": -0.7,
            "value": 0.25,
            "cost_value": -0.125,
            "send_probabilities": {"target-5": 0.9},
            "one_shot_state_after_step": {"target-5": "send_now_chosen",
                                          "target-closed": "send_now_chosen"},
        }
        row = {
            "kind": "train", "label": "r76_train_s11_e0", "scenario_seed": 1,
            "attempt": 1, "policy": "learned", "spent_bytes": 100.0,
            "budget_bytes": 29300.0, "airtime_tap_total": 100.0, "emissions": 1,
            "on_time": 6, "completed": 6, "expired": 0, "discovered": 6,
            "active": 0, "unassigned": 0, "task_reward_total": 6,
            "training_version": contracts.R76_TRAINING_VERSION,
            "reward_timing_version": "published-completion-event-at-relay-tick/2",
            "b0_bytes": 10000.0, "normalised_cost_total": 0.01,
            "emission_sequence": [[7, "target-5", 1, 100, 6, True]],
            "refusal_sequence": [],
            "cost_stream": {"7": {"airtime_bytes": 100.0}},
            "task_stream": {"7": 6.0},
            "steps": 1000,
            "artefacts": {"row": "row.json", "tape": "tape.jsonl",
                          "lifecycle": "lifecycle.jsonl", "transcript": "transcript.jsonl"},
            "training_seed": 11, "episode_index": 0, "lambda": 1.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row_path = root / "row.json"
            tape_path = root / "tape.jsonl"
            protocol.write_row(row_path, row)
            protocol.write_tape(tape_path, [tape])
            loaded_row = protocol.read_row(row_path)
            loaded_tape = protocol.read_tape(tape_path)
            episode = protocol.rebuild_episode(loaded_row, loaded_tape)
            self.assertEqual(episode["collector"].records[0]["features"][0][0], 0.0)
            self.assertEqual(episode["collector"].records[0]["actions"]["target-5"], 1)
            broken = dict(loaded_tape[0])
            del broken["cost_value"]
            with self.assertRaisesRegex(RuntimeError, "missing required fields"):
                protocol.validate_tape_record(broken, record_index=0)
            bad_features = dict(
                loaded_tape[0],
                features=[loaded_tape[0]["features"][0], [0.0] * 18,
                          loaded_tape[0]["features"][2]])
            with self.assertRaisesRegex(RuntimeError, "19 finite"):
                protocol.validate_tape_record(bad_features, record_index=0)
            missing_required_free_quote = dict(loaded_tape[0])
            del missing_required_free_quote["attempt_charge_bytes"]["target-5"]
            with self.assertRaisesRegex(RuntimeError, "relay quotes"):
                protocol.validate_tape_record(missing_required_free_quote, record_index=0)

    def test_7_real_fixture_guard_uses_real_counter_and_rejects_canonical_roots(self):
        import r76_guard

        historical_root = r76_guard.R75_HISTORICAL_CANONICAL_ROOT.resolve()
        canonical_root = r76_guard.R76_CANONICAL_ROOT.resolve()
        with self.assertRaisesRegex(RuntimeError, "R76 canonical durable root"):
            r76_guard.R76Resources.fixture(canonical_root)
        with self.assertRaisesRegex(RuntimeError, "canonical durable root"):
            r76_guard.R76Resources.fixture(canonical_root / "descendant")
        with self.assertRaisesRegex(RuntimeError, "historical R75"):
            r76_guard.R76Resources.fixture(historical_root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "isolated_fixture"
            resources = r76_guard.R76Resources.fixture(root)
            self.assertTrue(resources.r75_fixture)
            self.assertEqual(resources.physical_steps.snapshot()["used_normal"], 0)
            self.assertEqual(resources.charge_physical_step("fixture-normal-1"), 1)
            self.assertEqual(resources.charge_physical_step("fixture-normal-2"), 2)
            resources.pre_register_technical_attempt("fixture-reserve")
            self.assertEqual(resources.charge_physical_step(
                "fixture-reserve", category="technical"), 3)
            with self.assertRaises(Exception):
                resources.charge_physical_step("fixture-refused")
            self.assertEqual(resources.physical_steps.snapshot()["used_normal"], 2)
            self.assertEqual(resources.physical_steps.snapshot()["used_reserve"], 1)
            self.assertEqual(resources.physical_steps.snapshot()["refusals"], 1)
            resources.pre_register_technical_attempt("fixture-offline-technical")
            resources.pre_register_technical_attempt("fixture-rows-reserve")
            self.assertEqual(resources.charge_offline_rows(4, "fixture-rows-normal"), 4)
            self.assertEqual(resources.charge_offline_rows(2, "fixture-rows-reserve",
                                                           category="technical"), 6)
            with self.assertRaises(Exception):
                resources.charge_offline_rows(1, "fixture-rows-refused")
            self.assertEqual(resources.offline_rows.snapshot()["used_normal"], 4)
            self.assertEqual(resources.offline_rows.snapshot()["used_reserve"], 2)
            self.assertEqual(resources.offline_rows.snapshot()["refusals"], 1)
            with self.assertRaisesRegex(RuntimeError, "non-production"):
                r76_guard.assert_native_production_resources(resources)

    def test_8_cli_refuses_without_real_permit_before_lazy_real_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing-permit.json"
            argv = ["--request-id", contracts.REQUEST_ID,
                    "--source-head", "dc363721cc2954eabce04e34e2ccc578866ee468",
                    "--permit", str(missing),
                    "--attempt-dir", str(Path(temporary) / "attempt")]
            with self.assertRaisesRegex(RuntimeError, "real execution refused"):
                r76_cli.main(argv)
            self.assertFalse((Path(temporary) / "attempt").exists())
            self.assertNotIn("r76_execute", sys.modules)
        driver_sha256 = hashlib.sha256(
            (HERE / "r76_execute.py").read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            permit_path = Path(temporary) / "complete-but-unbacked-permit.json"
            permit_path.write_text(json.dumps({
                "kind": "round76-real-training-permission/1",
                "status": "issued",
                "request_id": contracts.REQUEST_ID,
                "task_id": contracts.TASK_ID,
                "branch": contracts.BRANCH,
                "source_head": "dc363721cc2954eabce04e34e2ccc578866ee468",
                "scope": "real_training_frozen_contract",
                "real_contract_sha256": driver_sha256,
                "resource_root": str(HERE / "raw" / "runtime"),
                "contract_path": str(Path(temporary) / "missing-contract.json"),
            }, indent=2), encoding="utf-8")
            argv = ["--request-id", contracts.REQUEST_ID,
                    "--source-head", "dc363721cc2954eabce04e34e2ccc578866ee468",
                    "--permit", str(permit_path),
                    "--attempt-dir", str(Path(temporary) / "attempt")]
            with self.assertRaisesRegex(RuntimeError, "separately frozen real-training contract"):
                r76_cli.main(argv)
            self.assertFalse((Path(temporary) / "attempt").exists())
            self.assertNotIn("r76_execute", sys.modules)


if __name__ == "__main__":
    unittest.main()
