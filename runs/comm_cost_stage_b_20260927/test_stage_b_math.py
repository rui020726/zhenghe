"""Stage B 零启动合成测试：接线数学与 collector 行为，不导入 driver、不启动任何环境。

对应 STAGE_B_PROTOCOL.md §7。运行方式：

    /d/project_env/python.exe -m unittest discover -s runs/comm_cost_stage_b_20260927

这些用例只在合成 `SendObservation` / stub 权重上运行；唯一被触及的"真实"对象是生产
`CommBudget`（shadow gate）、`BudgetMaskedCollector` 的采样路径与 Stage A 的 `cost_math`。
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

import sb_common as common
import sb_collector as sbc
import cost_math as cm

from integrated_sar.comm_budget import CommBudget
from integrated_sar.comm_policy import ImmediatePolicy, SendItem, SendObservation, transport_order


def make_item(target: str, *, pending: bool, charge: int, executable: bool,
              remaining_time: float = 100.0, age_steps: int = 0) -> SendItem:
    return SendItem(
        target_id=target, ready_step=1, age_steps=age_steps, pending_event=pending,
        changed_view=True, first_emission=pending, revision=1, revision_delta=0,
        stage_index=0, stage_changed=False, progress=0.0, progress_delta=0.0,
        requirements_changed=False, members_changed=False, deadline=1000.0,
        remaining_time=remaining_time, sent_count=0, last_sent_step=None,
        since_last_emission_time=None, has_future_requirements=False,
        attempt_charge_bytes=charge, send_executable=executable,
        send_block_reason="allowed" if executable else "budget_exhausted",
    )


def make_observation(items: list[SendItem], *, step: int = 1, remaining: float = 10_000.0,
                     budget: int | None = 10_000, capacity: int | None = None) -> SendObservation:
    return SendObservation(
        step=step, current_time=float(step), items=tuple(items),
        remaining_budget_bytes=remaining if budget is not None else float("inf"),
        budget_bytes_per_episode=budget, capacity_remaining=capacity,
        backlog_targets=len(items), subscribers=6,
    )


class CollectorWiring(unittest.TestCase):
    """The collector's tape must describe the distribution it actually drew from."""

    def test_log_prob_recompute_within_tolerance(self) -> None:
        torch.manual_seed(common.SAMPLING_SEED)
        from sb_controller import StageBController

        collector = sbc.StageBCollector(StageBController(common.ACTOR_INIT_SEED))
        items = [make_item("t1", pending=True, charge=2400, executable=True),
                 make_item("t2", pending=True, charge=2400, executable=True),
                 make_item("t3", pending=False, charge=2400, executable=True)]
        observation = make_observation(items, remaining=10_000.0)
        asked = collector.decide(observation)
        tape = collector.records[-1]
        self.assertEqual(set(asked),
                         {t for t in tape["targets"] if tape["requested"][t]})
        # Recompute the joint log-probability from the stored features and the same weights.
        matrix = torch.as_tensor(np.asarray(tape["features"], dtype=np.float32))
        logits, _ = collector.controller.forward(matrix)
        actions = torch.tensor([tape["actions"][t] for t in tape["targets"]], dtype=torch.long)
        allowed = torch.tensor(
            [[True, True] if t in tape["actionable"] else [True, False] for t in tape["targets"]],
            dtype=torch.bool)
        ll, entropy = cm.masked_log_likelihood(logits, actions, allowed)
        self.assertTrue(torch.isfinite(ll) and torch.isfinite(entropy))
        self.assertLess(abs(float(ll) - float(tape["log_prob"])), common.LOG_PROB_TOLERANCE)

    def test_masked_candidates_carry_no_likelihood_and_close(self) -> None:
        torch.manual_seed(common.SAMPLING_SEED)
        from sb_controller import StageBController

        collector = sbc.StageBCollector(StageBController(common.ACTOR_INIT_SEED))
        # Two affordable candidates, a third one the credit cannot cover as the third send.
        items = [make_item("t1", pending=True, charge=5000, executable=True),
                 make_item("t2", pending=True, charge=5000, executable=True),
                 make_item("t3", pending=False, charge=5000, executable=False)]
        collector.decide(make_observation(items, remaining=10_000.0))
        tape = collector.records[-1]
        self.assertIn("t3", tape["masked"])
        self.assertEqual(tape["masked_reasons"]["t3"], "budget_exhausted")
        self.assertFalse(tape["requested"]["t3"])
        # masked/actionable/forced are disjoint and cover every candidate
        summary = collector.tape_summary()
        self.assertTrue(summary["candidate_classes_disjoint"])
        self.assertTrue(summary["candidate_classes_close"])
        self.assertTrue(summary["masked_carry_no_likelihood"])
        self.assertEqual(summary["forced_total"], 0)

    def test_shadow_gate_is_the_production_gate(self) -> None:
        from budget_masked import shadow_gate

        items = [make_item("t1", pending=True, charge=4000, executable=True)]
        observation = make_observation(items, remaining=6000.0, budget=10_000)
        gate = shadow_gate(observation)
        self.assertIsInstance(gate, CommBudget)
        self.assertTrue(gate.can_send(4000))
        self.assertFalse(gate.can_send(7000))
        reference = CommBudget(bytes_per_episode=10_000, sends_per_step=None)
        reference.spent_bytes = 4000
        self.assertEqual(gate.remaining_bytes, reference.remaining_bytes)

    def test_cost_value_is_finite_and_logged(self) -> None:
        torch.manual_seed(common.SAMPLING_SEED)
        from sb_controller import StageBController

        collector = sbc.StageBCollector(StageBController(common.ACTOR_INIT_SEED))
        items = [make_item("t1", pending=True, charge=2400, executable=True)]
        collector.decide(make_observation(items))
        tape = collector.records[-1]
        self.assertIn("cost_value", tape)
        self.assertTrue(np.isfinite(tape["cost_value"]))
        self.assertEqual(tape["sender_state"]["candidates"], 1)
        self.assertEqual(tape["sender_state"]["subscribers"], 6)

    def test_all_masked_step_has_no_actor_sample(self) -> None:
        torch.manual_seed(common.SAMPLING_SEED)
        from sb_controller import StageBController

        collector = sbc.StageBCollector(StageBController(common.ACTOR_INIT_SEED))
        items = [make_item("t1", pending=True, charge=5000, executable=False)]
        asked = collector.decide(make_observation(items, remaining=1000.0))
        self.assertEqual(asked, ())
        tape = collector.records[-1]
        self.assertEqual(float(tape["log_prob"]), 0.0)
        self.assertEqual(tape["actionable"], [])


class RuleEquivalence(unittest.TestCase):
    """I_send must ask for exactly what native immediate asks for, in the transport's order."""

    def test_immediate_equivalent_matches_native_immediate(self) -> None:
        items = [
            make_item("t3", pending=True, charge=100, executable=True, remaining_time=50.0),
            make_item("t1", pending=False, charge=100, executable=True, remaining_time=10.0),
            make_item("t2", pending=True, charge=100, executable=True, remaining_time=90.0,
                      age_steps=7),
            make_item("t4", pending=True, charge=100, executable=True, remaining_time=10.0),
        ]
        observation = make_observation(items)
        native = ImmediatePolicy().decide(observation)
        native_trimmed = tuple(item.target_id for item in transport_order(
            item for item in observation.items if item.target_id in set(native)))
        injected = sbc.immediate_equivalent_fn(observation)
        self.assertEqual(set(native), set(injected))
        self.assertEqual(native_trimmed, injected)

    def test_hold_equivalent_is_empty(self) -> None:
        items = [make_item("t1", pending=True, charge=100, executable=True)]
        self.assertEqual(sbc.hold_equivalent_fn(make_observation(items)), ())


class EpisodeMath(unittest.TestCase):
    """The bundled reward stream and GAE must close onto the executor's own totals."""

    def test_bundling_and_returns_close_on_totals(self) -> None:
        decision_steps = [3, 7, 10]
        cost_stream = {1: 100.0, 2: 200.0, 4: 50.0, 7: 25.0, 10: 500.0}
        task_stream = {2: 1.0, 5: 2.0, 9: 1.0}
        horizon = 10
        bundled_cost = sbc.bundle_rewards(cost_stream, decision_steps, horizon)
        bundled_task = sbc.bundle_rewards(task_stream, decision_steps, horizon)
        self.assertEqual(sum(bundled_cost.values()), sum(cost_stream.values()))
        self.assertEqual(sum(bundled_task.values()), sum(task_stream.values()))
        self.assertEqual(bundled_cost[3], 300.0)   # steps 1..3 land on decision step 3
        self.assertEqual(bundled_cost[7], 75.0)    # steps 4..7
        self.assertEqual(bundled_cost[10], 500.0)
        self.assertEqual(bundled_task[3], 1.0)     # step 2 lands on decision step 3
        self.assertEqual(bundled_task[7], 2.0)     # step 5 lands on decision step 7
        self.assertEqual(bundled_task[10], 1.0)    # step 9 lands on decision step 10
        # gamma=1 keeps the *reward stream* undiscounted; the GAE return mixes the residuals
        # with gae_lambda, so returns[0] is the undiscounted total only at gae_lambda=1.
        rewards = torch.tensor([-bundled_cost[s] / common.B0_TEST for s in decision_steps],
                               dtype=torch.float64)
        terminal = torch.zeros(3, dtype=torch.bool)
        terminal[-1] = True
        values = torch.zeros(3, dtype=torch.float64)
        advantages, returns = cm.gae(rewards, values, gamma=1.0, gae_lambda=0.95,
                                     terminal=terminal, bootstrap=torch.zeros(3, dtype=torch.float64))
        expected_return = float(rewards[0] + 0.95 * rewards[1] + 0.95 * 0.95 * rewards[2])
        self.assertAlmostEqual(float(returns[0]), expected_return, places=9)
        self.assertAlmostEqual(float(rewards.sum()), -sum(cost_stream.values()) / common.B0_TEST,
                               places=9)
        _, full_returns = cm.gae(rewards, values, gamma=1.0, gae_lambda=1.0, terminal=terminal,
                                 bootstrap=torch.zeros(3, dtype=torch.float64))
        self.assertAlmostEqual(float(full_returns[0]),
                               -sum(cost_stream.values()) / common.B0_TEST, places=9)
        self.assertTrue(torch.isfinite(advantages).all())

    def test_combined_advantage_matches_stage_a_wiring(self) -> None:
        a_task = torch.tensor([1.0, 0.0], dtype=torch.float64)
        a_cost = torch.tensor([0.2, 0.0], dtype=torch.float64)
        combined = cm.actor_advantage(a_task, a_cost, common.LAMBDA_TEST)
        self.assertTrue(torch.equal(combined, torch.tensor([0.8, 0.0], dtype=torch.float64)))

    def test_b0_round_trip(self) -> None:
        norm = cm.CostNormalization(b0_bytes=common.B0_TEST)
        self.assertAlmostEqual(norm.normalise(29300.0), 2.93, places=12)
        self.assertAlmostEqual(norm.to_bytes(2.93), 29300.0, places=6)


class ConfigurationInheritance(unittest.TestCase):
    """The per-seed configuration must inherit the archived world and this round's overrides."""

    def test_stage_b_fields_carry_frozen_overrides_and_round_paths(self) -> None:
        seed = common.SEEDS[0]
        fields, substituted = common.stage_b_fields("P_init", seed)
        for key, value in common.STAGE_B_OVERRIDES.items():
            self.assertEqual(fields[key], value, key)
        self.assertEqual(fields["seed"], seed)
        self.assertEqual(fields["comm_policy"], "learned")
        self.assertEqual(fields["comm_budget_bytes"], 29300)
        self.assertEqual(fields["comm_timing_mode"], "policy_owned_timing")
        self.assertIsNone(fields["info_probe"])
        self.assertIsNone(fields["search_probe"])
        self.assertIsNone(fields["record_stream"])
        self.assertIn(str(common.LOGS), fields["output"])
        self.assertTrue(any("D1" in line or "Stage B overrides" in line for line in substituted))

    def test_controller_initialisation_is_fixed(self) -> None:
        from sb_controller import StageBController

        first = StageBController(common.ACTOR_INIT_SEED).parameter_summary()
        second = StageBController(common.ACTOR_INIT_SEED).parameter_summary()
        self.assertEqual(first["parameter_sha256"], second["parameter_sha256"])
        other = StageBController(common.ACTOR_INIT_SEED + 1).parameter_summary()
        self.assertNotEqual(first["parameter_sha256"], other["parameter_sha256"])
        self.assertEqual(first["parameter_count"], 19 * 64 + 64 + 64 * 64 + 64 + 64 * 2 + 2
                         + 64 + 1 + 64 + 1)


if __name__ == "__main__":
    unittest.main()
