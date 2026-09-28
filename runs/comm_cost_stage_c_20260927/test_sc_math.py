"""Stage C 零启动合成测试：规则边界、训练器数学、数据顺序与缓存键。

不导入 driver、不构造 `JointRun`、不启动任何环境。对应 ROUND22 spec §2/§4/§6/§7 中
可在合成数据上锁定的部分：late_discovery_gate 的 tick/运算符边界、对偶更新公式与方向、
无 advantage 标准化、双 critic 损失形状、GAE 闭合、场景顺序与 reference 缓存键。

    /d/project_env/python.exe -m unittest discover -s runs/comm_cost_stage_c_20260927
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

import sc_common as common
import sc_rules as rules
import sc_trainer as trainer
from sc_controller import fresh_controller, parameter_identity

from integrated_sar.comm_policy import SendItem, SendObservation


def make_item(target: str, *, first: bool, ready_step: int, pending: bool = True,
              charge: int = 2400, executable: bool = True) -> SendItem:
    return SendItem(
        target_id=target, ready_step=ready_step, age_steps=0, pending_event=pending,
        changed_view=True, first_emission=first, revision=1, revision_delta=0,
        stage_index=0, stage_changed=False, progress=0.0, progress_delta=0.0,
        requirements_changed=False, members_changed=False, deadline=1000.0,
        remaining_time=800.0, sent_count=0, last_sent_step=None,
        since_last_emission_time=None, has_future_requirements=False,
        attempt_charge_bytes=charge, send_executable=executable,
        send_block_reason="allowed" if executable else "budget_exhausted",
    )


def observation(items: list[SendItem], *, step: int = 1) -> SendObservation:
    return SendObservation(step=step, current_time=float(step), items=tuple(items),
                           remaining_budget_bytes=29000.0, budget_bytes_per_episode=29300,
                           capacity_remaining=None, backlog_targets=len(items), subscribers=6)


class LateDiscoveryGate(unittest.TestCase):
    """The rule's boundary, operator and sender-visibility, locked before any launch."""

    def test_boundary_operator_and_predicate(self) -> None:
        policy = rules.LateDiscoveryGatePolicy(cut_steps=147.0)
        items = [make_item("early", first=True, ready_step=146),
                 make_item("at_cut", first=True, ready_step=147),
                 make_item("late", first=True, ready_step=148),
                 make_item("already_sent", first=False, ready_step=999)]
        chosen = policy.decide(observation(items))
        self.assertEqual(chosen, ("late",))      # strictly greater than 147, first report only

    def test_uses_only_sender_visible_fields(self) -> None:
        # The rule never receives the registry or a receiver's belief: its whole input is the
        # observation the relay assembled, and it selects on first_emission/ready_step only.
        policy = rules.LateDiscoveryGatePolicy(cut_steps=147.0)
        import inspect

        source = inspect.getsource(type(policy).decide)
        self.assertIn("first_emission", source)
        self.assertIn("ready_step", source)
        self.assertNotIn("registry", source)
        self.assertNotIn("belief", source)

    def test_rule_factory_builds_registered_arms(self) -> None:
        for arm in common.RULE_ARMS:
            policy = rules.build_rule(arm, 1300000)
            self.assertTrue(policy.name)
        gate = rules.build_rule("late_discovery_gate", 1300000)
        self.assertEqual(gate.parameters(), {"cut_steps": 147.0})
        periodic = rules.build_rule("periodic_20", 1300000)
        self.assertEqual(periodic.parameters(), {"interval": 20})
        random_send = rules.build_rule("random_send", 1300000)
        self.assertEqual(random_send.parameters(), {"send_probability": 0.5, "seed": 1300000})

    def test_unknown_rule_field_is_an_error(self) -> None:
        with self.assertRaises(Exception):
            rules.build_rule("no_such_arm", 1300000)


class TrainerMath(unittest.TestCase):
    """Dual update, no-standardisation, two critics, GAE closure -- on synthetic streams."""

    class FakeEpisode(dict):
        """A dict-shaped episode: the trainer reads its episodes by subscript."""

        def __getattr__(self, name: str):
            try:
                return self[name]
            except KeyError as error:
                raise AttributeError(name) from error

    def test_dual_update_formula_and_direction(self) -> None:
        record = trainer.dual_update(1.0, 0.5)
        self.assertAlmostEqual(record["lambda_next"], 1.0 + common.ETA_LAMBDA * 0.5, places=12)
        self.assertEqual(record["direction"], "increase")
        self.assertAlmostEqual(trainer.dual_update(0.02, -1.0)["lambda_next"], 0.0, places=12)
        self.assertEqual(trainer.dual_update(0.02, -1.0)["direction"], "decrease")

    def test_batch_g_hat_is_paired_and_scalar(self) -> None:
        def episode(scenario: int, on_time: int) -> "TrainerMath.FakeEpisode":
            return TrainerMath.FakeEpisode(scenario_seed=scenario, on_time=on_time)

        record = trainer.batch_g_hat([episode(1, 5), episode(2, 6)],
                                     {1: 6.0, 2: 6.0})
        self.assertAlmostEqual(record["g_hat"], 0.5, places=12)   # mean(6-5, 6-6) - 0
        self.assertTrue(all(isinstance(pair["y_reference"], float) for pair in record["pairs"]))

    def test_samples_use_combined_advantage_without_standardisation(self) -> None:
        # Two synthetic "episodes": tapes with values, cost/task streams, a lambda.
        tapes = [{"step": 2, "value": 0.1, "cost_value": 0.2, "targets": ["t"],
                  "actions": {"t": 1}, "actionable": ["t"], "features": [[0.0] * 19],
                  "log_prob": -0.7},
                 {"step": 5, "value": 0.3, "cost_value": 0.1, "targets": ["t"],
                  "actions": {"t": 0}, "actionable": ["t"], "features": [[0.0] * 19],
                  "log_prob": -0.6}]

        class Collector:
            records = tapes

        episode = TrainerMath.FakeEpisode(collector=Collector(), steps=10,
                                          cost_stream={1: {"airtime_bytes": 1000},
                                                       3: {"airtime_bytes": 2000},
                                                       6: {"airtime_bytes": 500}},
                                          task_stream={4: 1.0, 8: 2.0},
                                          **{"lambda": 2.0}, episode_index=0)
        samples = trainer.batch_samples([episode])
        self.assertEqual(len(samples), 2)
        # The combined advantage is lambda*A_task - A_cost of the same GAE arrays; with a
        # deterministic stream this equals the independently computed combination.
        import cost_math as cm

        # decision steps 2 and 5: cost 1000 lands on step 2; 2000+500 on step 5;
        # task reward 1+2 lands on step 5 (the last decision step catches the tail).
        rewards_cost = torch.tensor([1000.0 / common.B0_BYTES, 2500.0 / common.B0_BYTES],
                                    dtype=torch.float64)
        rewards_task = torch.tensor([0.0, 3.0], dtype=torch.float64)
        terminal = torch.tensor([False, True])
        values_task = torch.tensor([0.1, 0.3], dtype=torch.float64)
        values_cost = torch.tensor([0.2, 0.1], dtype=torch.float64)
        a_task, _ = cm.gae(rewards_task, values_task,
                           gamma=common.GAMMA, gae_lambda=common.GAE_LAMBDA, terminal=terminal,
                           bootstrap=torch.tensor([0.3, 0.0], dtype=torch.float64))
        a_cost, _ = cm.gae(rewards_cost, values_cost,
                           gamma=common.GAMMA, gae_lambda=common.GAE_LAMBDA, terminal=terminal,
                           bootstrap=torch.tensor([0.1, 0.0], dtype=torch.float64))
        expected = cm.actor_advantage(a_task, a_cost, 2.0)
        for sample, want in zip(samples, expected.tolist()):
            self.assertAlmostEqual(sample["advantage"], want, places=9)

    def test_update_runs_and_moves_only_comm_parameters(self) -> None:
        import cost_math as cm

        controller = fresh_controller(11)
        before = parameter_identity(controller)
        optimizer = torch.optim.Adam(controller.parameters, lr=common.LEARNING_RATE)
        # A synthetic batch whose tapes are internally consistent with the sampler.
        matrix = torch.rand(3, cm.ACTOR_FEATURE_DIM, dtype=torch.float32)
        with torch.no_grad():
            logits, values = controller.forward(matrix)
            cost_values = controller.cost_values(matrix)
        probabilities = torch.softmax(logits, dim=-1)
        actions = [int(torch.multinomial(probabilities[index], 1).item()) for index in range(3)]

        class Collector:
            records = [{
                "step": 1, "targets": ["t"], "actions": {"t": actions[0]},
                "actionable": ["t"], "free": ["t"],
                "features": [matrix[0].tolist()],
                "log_prob": float(torch.log_softmax(logits[0], dim=-1)[actions[0]]),
                "value": float(values[0]), "cost_value": float(cost_values[0]),
                "requested": {"t": bool(actions[0])}, "masked": [], "forced": [],
            }]

        episode = TrainerMath.FakeEpisode(collector=Collector(), steps=4,
                                          cost_stream={2: {"airtime_bytes": 4800}},
                                          task_stream={3: 1.0},
                                          **{"lambda": 1.0}, episode_index=0)

        report = trainer.ppo_update(controller, optimizer, [episode], update_index=0)
        after = parameter_identity(controller)
        self.assertGreater(report["optimizer_steps"], 0)
        self.assertNotEqual(before["parameter_sha256"], after["parameter_sha256"])
        self.assertLessEqual(report["likelihood_alignment"]["max_abs_gap"], common.LOG_PROB_TOLERANCE)
        self.assertTrue(np.isfinite(report["value_loss_task"]))
        self.assertTrue(np.isfinite(report["value_loss_cost"]))

    def test_all_mask_step_has_no_actor_gradient(self) -> None:
        import cost_math as cm

        controller = fresh_controller(11)
        matrix = torch.zeros(1, cm.ACTOR_FEATURE_DIM, requires_grad=True)
        logits, _ = controller.forward(matrix)
        from onpolicy import joint_log_prob

        usable = torch.zeros(1, dtype=torch.float32)      # the single candidate is masked
        picked = torch.tensor([0], dtype=torch.long)
        ll = joint_log_prob(logits, picked, usable)
        self.assertEqual(float(ll), 0.0)
        ll.backward()
        # The masked term carries zero weight, so its gradient is exactly zero: an actor
        # update in that direction is impossible, which is what "no pseudo sample" means.
        self.assertTrue(matrix.grad is None or bool((matrix.grad == 0).all()))


class DataOrderAndCache(unittest.TestCase):
    """The frozen scenario order and the reference cache key are deterministic."""

    def test_scenario_order_is_frozen_and_covers_48_scenarios(self) -> None:
        first = common.scenario_order()
        second = common.scenario_order()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 128)
        self.assertEqual(len(set(first)), 48)            # all 48 training scenarios appear
        self.assertTrue(all(scenario in common.TRAINING_SCENARIOS for scenario in first))

    def test_reference_cache_key_is_deterministic_and_scenario_specific(self) -> None:
        identity = common.code_identity()
        behaviour = {"seed": 1300000, "comm_policy": "immediate"}
        key_a = common.reference_cache_key(1300000, identity, behaviour)
        key_b = common.reference_cache_key(1300000, identity, behaviour)
        key_c = common.reference_cache_key(1300001, identity, behaviour)
        self.assertEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)

    def test_budget_arithmetic(self) -> None:
        self.assertEqual(common.NORMAL_LAUNCHES, 672)
        self.assertEqual(common.LAUNCH_CAP, 700)
        self.assertEqual(len(common.TRAINING_SCENARIOS), 48)
        self.assertEqual(len(common.DEVELOPMENT_SCENARIOS), 16)


if __name__ == "__main__":
    unittest.main()
