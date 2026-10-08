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
        self.step_gradients = []

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        self.step_calls += 1
        self.step_gradients.append(tuple(
            None if p.grad is None else p.grad.detach().clone()
            for p in self.parameters))


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
    records = []
    namespace.update({"GATE_BETA": 1.0,
                      "GATE_RECORDS": records,
                      "record_minibatch": records.append,
                      "GATE_DIAG": {"pending": False, "done": False, "record": None}})
    exec(compile(source, "<counter-regression-source>", "exec"), namespace)
    controller = v2_controller.fresh_controller(11)
    optimizer = SpyOptimizer(controller.parameters)
    report = namespace["ppo_update"](controller, optimizer, one_rollout(controller), update_index=0)
    return report, optimizer.step_calls, records


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

    def test_beta_zero_compiled_prior_matches_canonical_ppo_loss_and_gradients(self):
        _, patched_source, _ = fixed_gate.build_patched_source(v2_trainer)
        namespace = dict(vars(v2_trainer))
        records = []
        namespace.update({"GATE_BETA": 0.0, "GATE_RECORDS": records,
                          "record_minibatch": records.append,
                          "GATE_DIAG": {"pending": False, "done": False, "record": None,
                                        "error": None, "record_errors": []}})
        exec(compile(patched_source, "<beta-zero-r58-source>", "exec"), namespace)
        canonical_controller = v2_controller.fresh_controller(11)
        patched_controller = v2_controller.fresh_controller(11)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(
            canonical_controller.parameters, patched_controller.parameters)))
        canonical_optimizer = SpyOptimizer(canonical_controller.parameters)
        patched_optimizer = SpyOptimizer(patched_controller.parameters)
        canonical_report = v2_trainer.ppo_update(
            canonical_controller, canonical_optimizer,
            one_rollout(canonical_controller), update_index=1)
        patched_report = namespace["ppo_update"](
            patched_controller, patched_optimizer,
            one_rollout(patched_controller), update_index=1)
        for key in ("samples", "minibatches", "optimizer_steps", "epochs",
                    "policy_surrogate", "value_loss_task", "value_loss_cost", "entropy"):
            self.assertEqual(canonical_report[key], patched_report[key])
        for real_step_grads, prior_step_grads in zip(
                canonical_optimizer.step_gradients, patched_optimizer.step_gradients):
            for a, b in zip(real_step_grads, prior_step_grads):
                self.assertTrue(torch.equal(a, b))


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


class DiagnosticSafetyRegression(unittest.TestCase):
    def test_prior_gradient_beta_scale_same_logits(self):
        logits = torch.tensor([[0.4, -0.2]], requires_grad=True)
        prior = -torch.log_softmax(logits, dim=-1)[0, 0]
        g01 = torch.autograd.grad(0.10 * prior, logits, retain_graph=True)[0]
        g1 = torch.autograd.grad(1.00 * prior, logits, retain_graph=True)[0]
        self.assertTrue(torch.allclose(g1, 10.0 * g01, rtol=1e-6, atol=1e-7))
        base = (logits ** 2).sum()
        g_base = torch.autograd.grad(base, logits, retain_graph=True)[0]
        g_zero = torch.autograd.grad(base + 0.0 * prior, logits)[0]
        self.assertTrue(torch.equal(g_base, g_zero))

    def test_diagnostic_handles_minibatch_with_no_eligible_prior_rows(self):
        controller = v2_controller.fresh_controller(11)
        features = torch.ones((1, 19), dtype=torch.float32)
        logits, _ = controller.forward(features)
        base_loss = torch.log_softmax(logits, dim=-1)[0, 1]
        diagnostic = fixed_gate.capture_actor_gradient_diagnostic(
            controller, base_loss, torch.zeros((), dtype=torch.float32))
        self.assertEqual(diagnostic["prior_grad_l1"], 0.0)
        self.assertEqual(diagnostic["prior_actor_grad_l2"], 0.0)
        self.assertIsNone(diagnostic["cosine"])

    def test_hashlib_diagnostics_run_at_0_15_31(self):
        self.assertTrue(hasattr(fixed_gate, "hashlib"))
        for update_index in (0, 15, 31):
            with self.subTest(update_index=update_index), tempfile.TemporaryDirectory(dir=HERE) as output:
                writes = []
                def writer(path, rows):
                    writes.append((path.name, list(rows)))
                installed = fixed_gate.install_gate_prior(
                    v2_trainer, critic_input_detached, pathlib.Path(output), writer=writer)
                try:
                    report, spy_steps, _seen = CriticIsolationRegression()._run_with_forward_input_probe(
                        v2_trainer.ppo_update, update_index=update_index)
                    self.assertEqual(report["optimizer_steps"], 8)
                    self.assertEqual(spy_steps, 8)
                    self.assertNotIn("diagnostics_error", report.get("gate_diag", {}))
                    self.assertEqual(report["gate_diag"]["update_index"], update_index)
                    self.assertEqual(len(report["gate_diag"]["params_before_sha256"]), 64)
                    self.assertEqual(len(report["gate_diag"]["params_after_sha256"]), 64)
                    self.assertIn("r58_gradient_diagnostics.jsonl", [name for name, _ in writes])
                finally:
                    installed["undo"]()

    def test_historical_missing_hashlib_is_reproduced_and_repaired(self):
        self.assertFalse(hasattr(old_gate, "hashlib"))
        with tempfile.TemporaryDirectory(dir=HERE) as output:
            old_install = old_gate.install_gate_prior(
                v2_trainer, critic_input_detached, pathlib.Path(output))
            try:
                report, spy_steps, _seen = CriticIsolationRegression()._run_with_forward_input_probe(
                    v2_trainer.ppo_update, update_index=0)
                self.assertEqual(spy_steps, 8)
                self.assertIn("NameError: name 'hashlib' is not defined",
                              report["gate_diag"]["diagnostics_error"])
            finally:
                old_install["undo"]()
        self.assertTrue(hasattr(fixed_gate, "hashlib"))

    def test_passive_diagnostic_does_not_change_gradients_or_rng(self):
        def run(update_index):
            with tempfile.TemporaryDirectory(dir=HERE) as output:
                installed = fixed_gate.install_gate_prior(
                    v2_trainer, critic_input_detached, pathlib.Path(output),
                    writer=lambda *_args: None)
                try:
                    controller = v2_controller.fresh_controller(11)
                    optimizer = SpyOptimizer(controller.parameters)
                    episodes = one_rollout(controller)
                    rng_before = torch.get_rng_state().clone()
                    report = v2_trainer.ppo_update(controller, optimizer, episodes,
                                                   update_index=update_index)
                    rng_after = torch.get_rng_state().clone()
                    return report, optimizer.step_gradients, rng_before, rng_after
                finally:
                    installed["undo"]()

        no_diag, grads_no_diag, rng0a, rng0b = run(1)
        with_diag, grads_with_diag, rng1a, rng1b = run(0)
        self.assertEqual(len(grads_no_diag), 8)
        self.assertEqual(len(grads_with_diag), 8)
        for g_no, g_yes in zip(grads_no_diag, grads_with_diag):
            self.assertEqual(len(g_no), len(g_yes))
            for a, b in zip(g_no, g_yes):
                self.assertTrue(torch.equal(a, b))
        for key in ("policy_surrogate", "value_loss_task", "value_loss_cost", "gate_loss"):
            self.assertEqual(no_diag[key], with_diag[key])
        self.assertTrue(torch.equal(rng0a, rng0b))
        self.assertTrue(torch.equal(rng1a, rng1b))

    def test_diagnostic_autograd_and_writer_errors_do_not_hide_update_return(self):
        original_diag = fixed_gate.capture_actor_gradient_diagnostic
        try:
            def fail_diagnostic(*_args, **_kwargs):
                raise ArithmeticError("injected diagnostic-only failure")
            fixed_gate.capture_actor_gradient_diagnostic = fail_diagnostic
            with tempfile.TemporaryDirectory(dir=HERE) as output:
                installed = fixed_gate.install_gate_prior(
                    v2_trainer, critic_input_detached, pathlib.Path(output),
                    writer=lambda *_args: (_ for _ in ()).throw(OSError("injected I/O failure")))
                try:
                    report, spy_steps, _seen = CriticIsolationRegression()._run_with_forward_input_probe(
                        v2_trainer.ppo_update, update_index=0)
                    committed = []
                    committed.append({"update_index": 0, "optimizer_steps": report["optimizer_steps"]})
                    self.assertEqual(spy_steps, 8)
                    self.assertEqual(len(committed), 1)
                    self.assertEqual(report["gate_diag"]["diagnostics_error"],
                                     "RuntimeError: ArithmeticError: injected diagnostic-only failure")
                    self.assertEqual(report["passive_diagnostics_errors"][0]["kind"],
                                     "minibatch_record_write")
                finally:
                    installed["undo"]()
        finally:
            fixed_gate.capture_actor_gradient_diagnostic = original_diag

    def test_core_ppo_error_still_propagates(self):
        original_batch_samples = v2_trainer.batch_samples
        def fail_core(_episodes):
            raise ValueError("injected core PPO failure")
        v2_trainer.batch_samples = fail_core
        with tempfile.TemporaryDirectory(dir=HERE) as output:
            installed = fixed_gate.install_gate_prior(
                v2_trainer, critic_input_detached, pathlib.Path(output), writer=lambda *_args: None)
            try:
                controller = v2_controller.fresh_controller(11)
                optimizer = SpyOptimizer(controller.parameters)
                with self.assertRaisesRegex(ValueError, "injected core PPO failure"):
                    v2_trainer.ppo_update(controller, optimizer, one_rollout(controller),
                                          update_index=1)
                self.assertEqual(optimizer.step_calls, 0)
            finally:
                installed["undo"]()
                v2_trainer.batch_samples = original_batch_samples


if __name__ == "__main__":
    unittest.main(verbosity=2)
