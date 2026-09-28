"""A nonterminal decision must bootstrap from the value at the next decision."""

from __future__ import annotations

import unittest

import v2_common as common
import v2_trainer as trainer
from v2_controller import fresh_controller
from v2_correctness_test_support import episode, record


class GaeNextValueCounterexample(unittest.TestCase):
    def test_batch_samples_uses_next_value_for_a_nonterminal_decision(self) -> None:
        controller = fresh_controller(11)
        records = [
            record(controller, 2, value=0.2, cost_value=0.0),
            record(controller, 5, value=0.3, cost_value=0.0),
        ]
        rollout = episode(records, steps=5, task_stream={2: 1.0})
        samples = trainer.batch_samples([rollout])

        # Tick 2's reward follows decision 2, so r=[1,0]. Direct Bellman/GAE calculation:
        # A_1 = 0 - V_1; A_0 = (1 + gamma*V_1 - V_0) + gamma*lambda_gae*A_1.
        delta_last = 0.0 - 0.3
        expected_advantage = (
            1.0 + common.GAMMA * 0.3 - 0.2
            + common.GAMMA * common.GAE_LAMBDA * delta_last
        )
        expected_return = 0.2 + expected_advantage
        self.assertAlmostEqual(samples[0]["advantage"], expected_advantage, places=9)
        self.assertAlmostEqual(samples[0]["return_task"], expected_return, places=9)

    def test_nonzero_value_two_step_example_matches_closed_form(self) -> None:
        import cost_math as cm
        import torch

        rewards = torch.tensor([0.0, 1.0], dtype=torch.float64)
        values = torch.tensor([0.4, 0.6], dtype=torch.float64)
        next_values = torch.tensor([0.6, 0.0], dtype=torch.float64)
        terminal = torch.tensor([False, True])
        advantages, returns = cm.gae(
            rewards, values, gamma=1.0, gae_lambda=0.95,
            terminal=terminal, bootstrap=next_values,
        )
        self.assertAlmostEqual(float(advantages[0]), 0.58, places=12)
        self.assertAlmostEqual(float(advantages[1]), 0.4, places=12)
        self.assertAlmostEqual(float(returns[0]), 0.98, places=12)
        self.assertAlmostEqual(float(returns[1]), 1.0, places=12)

        full_mc_advantage, full_mc_return = cm.gae(
            rewards, values, gamma=1.0, gae_lambda=1.0,
            terminal=terminal, bootstrap=next_values,
        )
        self.assertAlmostEqual(float(full_mc_return[0]), 1.0, places=12)
        self.assertAlmostEqual(float(full_mc_advantage[0] + values[0]), 1.0, places=12)

        truncated_advantage, truncated_return = cm.gae(
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            gamma=0.9, gae_lambda=0.95,
            terminal=torch.tensor([False]),
            bootstrap=torch.tensor([0.4], dtype=torch.float64),
        )
        self.assertAlmostEqual(float(truncated_advantage[0]), 0.86, places=12)
        self.assertAlmostEqual(float(truncated_return[0]), 1.36, places=12)

    def test_gae_stays_inside_each_episode(self) -> None:
        controller = fresh_controller(11)
        records_a = [
            record(controller, 2, value=0.2, cost_value=0.0),
            record(controller, 5, value=0.3, cost_value=0.0),
        ]
        records_b = [
            record(controller, 2, value=0.2, cost_value=0.0),
            record(controller, 5, value=0.3, cost_value=0.0),
        ]
        samples = trainer.batch_samples([
            episode(records_a, steps=5, task_stream={2: 1.0}),
            episode(records_b, steps=5, task_stream={2: 0.0}),
        ])
        expected_a = 1.0 + common.GAMMA * 0.3 - 0.2 \
            + common.GAMMA * common.GAE_LAMBDA * (0.0 - 0.3)
        expected_b = common.GAMMA * 0.3 - 0.2 \
            + common.GAMMA * common.GAE_LAMBDA * (0.0 - 0.3)
        self.assertAlmostEqual(samples[0]["advantage"], expected_a, places=9)
        self.assertAlmostEqual(samples[2]["advantage"], expected_b, places=9)


if __name__ == "__main__":
    unittest.main()
