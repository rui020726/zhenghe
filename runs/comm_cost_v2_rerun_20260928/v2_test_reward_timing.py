"""Post-action rewards must remain attached to the action that caused them."""

from __future__ import annotations

import unittest

import v2_common as common
import v2_check
import v2_trainer as trainer
from v2_controller import fresh_controller
from v2_correctness_test_support import episode, record


class RewardTimingCounterexample(unittest.TestCase):
    def test_same_tick_between_prefix_and_tail_rewards_follow_action_order(self) -> None:
        controller = fresh_controller(11)
        records = [
            record(controller, 7, value=0.0, cost_value=0.0),
            record(controller, 10, value=0.0, cost_value=0.0),
        ]
        # CommRelay.tick samples the action before Task03Adapter advances the world. A reward
        # on that same tick and rewards between decisions follow action 7; the tick-2 prefix
        # predates every action, while the horizon tail follows the final action at tick 10.
        rollout = episode(
            records,
            steps=12,
            cost_stream={2: {"airtime_bytes": 50},
                         7: {"airtime_bytes": 100},
                         9: {"airtime_bytes": 200},
                         12: {"airtime_bytes": 300}},
            task_stream={2: 1.0, 7: 2.0, 9: 3.0, 12: 4.0},
        )
        samples = trainer.batch_samples([rollout])

        self.assertAlmostEqual(samples[0]["return_task"], 8.8, places=9)
        self.assertAlmostEqual(samples[1]["return_task"], 4.0, places=9)
        self.assertAlmostEqual(samples[0]["unassigned_prefix_task_reward"], 1.0, places=9)
        self.assertAlmostEqual(samples[1]["unassigned_prefix_task_reward"], 0.0, places=9)
        self.assertAlmostEqual(samples[0]["return_cost"], 0.0585, places=9)
        self.assertAlmostEqual(samples[1]["return_cost"], 0.03, places=9)
        self.assertAlmostEqual(samples[0]["unassigned_prefix_cost_bytes"], 50.0, places=9)

        # Total closure is checked separately from causal attribution: 1 prefix reward is
        # intentionally absent from the action returns, while all post-action rewards close.
        self.assertAlmostEqual(sum(sample["return_task"] for sample in samples)
                               - common.GAE_LAMBDA * 4.0, 9.0, places=9)
        self.assertAlmostEqual(sum(sample["return_cost"] for sample in samples)
                               - common.GAE_LAMBDA * 0.03, 0.06, places=9)

        independent = v2_check.recompute_episode_math(
            {
                "steps": 12,
                "cost_stream": {str(step): row for step, row in rollout["cost_stream"].items()},
                "task_stream": {str(step): value for step, value in rollout["task_stream"].items()},
                "spent_bytes": 650,
                "task_reward_total": 10.0,
            },
            records,
            lam=1.0,
        )
        self.assertAlmostEqual(independent["sum_rewards_task"], 9.0, places=9)
        self.assertAlmostEqual(independent["unassigned_prefix_task_reward"], 1.0, places=9)
        self.assertAlmostEqual(independent["sum_rewards_cost"], 0.06, places=9)
        self.assertAlmostEqual(independent["unassigned_prefix_cost_bytes"], 50.0, places=9)
        self.assertAlmostEqual(independent["returns_task_first"], 8.8, places=9)
        self.assertAlmostEqual(independent["returns_cost_first"], 0.0585, places=9)


if __name__ == "__main__":
    unittest.main()
