from __future__ import annotations

import importlib.util
import pathlib
import sys
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
