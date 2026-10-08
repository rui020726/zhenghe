from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import torch


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[0]
R58_OLD = ROOT / "round58_gen2"
R53 = ROOT / "round53_critic_isolation_trial_readiness"
R45 = HERE.parents[2] / "runs" / "comm_rule_warmstart_cost_learning_20261006_r2"
CANON = HERE.parents[2].parent / "zhenghe" / "runs" / "comm_cost_v2_rerun_20260928"
sys.path[:0] = [str(R58_OLD), str(R53), str(R45), str(CANON)]

import r45_bc  # noqa: E402
r45_bc.configure_source_imports()
import v2_controller  # noqa: E402
import v2_trainer  # noqa: E402
from round53_trial_adapter import critic_input_detached  # noqa: E402


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


old_gate = load_module(R58_OLD / "r58_gate_prior.py", "r58_gate_prior_old_fixture")
fixed_gate = load_module(HERE / "r58_gate_prior_repaired.py", "r58_gate_prior_repaired_fixture")


class SpyOptimizer:
    """Counts the canonical loop's call site without performing an optimizer update."""

    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.step_calls = 0

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        self.step_calls += 1


def one_rollout(controller, count=265):
    feature = np.zeros((1, 19), dtype=np.float32)
    feature[0, 5] = 1.0
    feature[0, 9] = 0.5
    feature[0, 14] = 1.0
    feature[0, 17] = 6.0
    feature[0, 18] = 0.1
    with torch.no_grad():
        logits, _ = controller.forward(torch.as_tensor(feature))
        old_log_prob = float(v2_trainer.joint_log_prob(
            logits, torch.tensor([1]), torch.tensor([1.0])))
    records = [{
        "step": i + 1, "features": feature.copy(), "targets": ["t"],
        "actions": {"t": 1}, "actionable": ["t"], "log_prob": old_log_prob,
        "value": 0.0, "cost_value": 0.0,
    } for i in range(count)]
    collector = type("Collector", (), {"records": records})()
    return [{"collector": collector, "steps": 1000, "lambda": 1.0,
             "cost_stream": {}, "task_stream": {}, "episode_index": 0}]


def execute_source(source: str):
    namespace = dict(vars(v2_trainer))
    namespace.update({"GATE_BETA": 1.0,
                      "GATE_RECORDS": [],
                      "GATE_DIAG": {"pending": False, "done": False, "record": None}})
    exec(compile(source, "<counter-regression-source>", "exec"), namespace)
    controller = v2_controller.fresh_controller(11)
    optimizer = SpyOptimizer(controller.parameters)
    report = namespace["ppo_update"](controller, optimizer, one_rollout(controller), update_index=0)
    return report, optimizer.step_calls, namespace["GATE_RECORDS"]


class CounterRegression(unittest.TestCase):
    def test_historical_patch_reproduces_double_counter_without_real_optimizer(self):
        _, old_source, _ = old_gate.build_patched_source(v2_trainer)
        report, spy_calls, records = execute_source(old_source)
        self.assertEqual(spy_calls, 8)
        self.assertEqual(report["optimizer_steps"], 16)
        self.assertEqual([r["minibatch"] for r in records], list(range(1, 16, 2)))

    def test_repaired_patch_counts_each_minibatch_once_and_averages_correctly(self):
        _, new_source, _ = fixed_gate.build_patched_source(v2_trainer)
        report, spy_calls, records = execute_source(new_source)
        self.assertEqual(spy_calls, 8)
        self.assertEqual(report["optimizer_steps"], 8)
        self.assertEqual(report["minibatches"], 8)
        self.assertEqual([r["minibatch"] for r in records], list(range(8)))
        self.assertAlmostEqual(report["gate_loss"],
                               sum(r["gate_loss_raw"] for r in records) / 8,
                               places=6)


class CriticIsolationRegression(unittest.TestCase):
    def _gradients(self, loss, controller, modules):
        params = [p for module in modules for p in module.parameters()]
        return torch.autograd.grad(loss, params, allow_unused=True)

    def test_real_task_and_cost_heads_detach_shared_trunk(self):
        controller = v2_controller.fresh_controller(11)
        features = torch.linspace(-0.3, 0.4, 19, dtype=torch.float32).reshape(1, 19)
        trunk_params = list(controller.inner.trunk.parameters())

        hidden = controller.inner.trunk(features)
        with critic_input_detached(controller):
            task_value = controller.inner.value_head(hidden).sum()
            task_grads = torch.autograd.grad(
                task_value, trunk_params + list(controller.inner.value_head.parameters()),
                allow_unused=True)
        n_trunk = len(trunk_params)
        self.assertTrue(all(g is None for g in task_grads[:n_trunk]))
        self.assertTrue(all(g is not None and float(g.abs().sum()) > 0
                            for g in task_grads[n_trunk:]))

        hidden = controller.inner.trunk(features)
        with critic_input_detached(controller):
            cost_value = controller.cost_head(hidden).sum()
            cost_grads = torch.autograd.grad(
                cost_value, trunk_params + list(controller.cost_head.parameters()),
                allow_unused=True)
        self.assertTrue(all(g is None for g in cost_grads[:n_trunk]))
        self.assertTrue(all(g is not None and float(g.abs().sum()) > 0
                            for g in cost_grads[n_trunk:]))

        hidden = controller.inner.trunk(features)
        actor_logits = controller.inner.action_head(hidden)
        actor_loss = -torch.log_softmax(actor_logits, dim=-1)[0, 0]
        actor_grads = torch.autograd.grad(
            actor_loss,
            trunk_params + list(controller.inner.action_head.parameters()),
            allow_unused=True)
        self.assertTrue(all(g is not None and float(g.abs().sum()) > 0
                            for g in actor_grads))
        for module in (controller.inner.value_head, controller.cost_head):
            self.assertEqual(set(module._forward_pre_hooks), set())

    def _run_with_forward_input_probe(self, fn, *, update_index):
        controller = v2_controller.fresh_controller(11)
        optimizer = SpyOptimizer(controller.parameters)
        observed = {"task": [], "cost": []}
        task = controller.inner.value_head
        cost = controller.cost_head
        task_forward, cost_forward = task.forward, cost.forward

        def task_probe(x):
            observed["task"].append(bool(x.requires_grad))
            return task_forward(x)

        def cost_probe(x):
            observed["cost"].append(bool(x.requires_grad))
            return cost_forward(x)

        task.forward, cost.forward = task_probe, cost_probe
        try:
            report = fn(controller, optimizer, one_rollout(controller),
                        update_index=update_index)
        finally:
            task.forward, cost.forward = task_forward, cost_forward
        return report, optimizer.step_calls, observed

    def test_historical_compiled_callable_bypasses_installed_critic_wrapper(self):
        _, old_source, _ = old_gate.build_patched_source(v2_trainer)
        namespace = dict(vars(v2_trainer))
        namespace.update({"GATE_BETA": 1.0, "GATE_RECORDS": [],
                          "GATE_DIAG": {"pending": False, "done": False, "record": None}})
        exec(compile(old_source, "<historical-r58-source>", "exec"), namespace)
        report, spy_steps, seen = self._run_with_forward_input_probe(
            namespace["ppo_update"], update_index=1)
        self.assertEqual(report["optimizer_steps"], 16)
        self.assertEqual(spy_steps, 8)
        # likelihood_alignment runs under no_grad and contributes the leading
        # False values; the final eight are the four-epoch PPO forwards.
        self.assertEqual(len(seen["task"]), 274, seen)
        self.assertTrue(all(seen["task"][-8:]), seen)
        self.assertEqual(len(seen["cost"]), 8, seen)
        self.assertTrue(all(seen["cost"]), seen)

    def test_repaired_installed_callable_enters_real_critic_context(self):
        with tempfile.TemporaryDirectory(dir=HERE) as output:
            installed = fixed_gate.install_gate_prior(
                v2_trainer, critic_input_detached, pathlib.Path(output))
            try:
                report, spy_steps, seen = self._run_with_forward_input_probe(
                    v2_trainer.ppo_update, update_index=1)
                self.assertEqual(report["optimizer_steps"], 8)
                self.assertEqual(spy_steps, 8)
                self.assertTrue(seen["task"] and not any(seen["task"]))
                self.assertTrue(seen["cost"] and not any(seen["cost"]))
            finally:
                installed["undo"]()


if __name__ == "__main__":
    unittest.main(verbosity=2)
