"""The mock frontier policy must only use what the robot can actually observe.

Regression for review finding 6: the trail penalty used to crop its pheromone
window around the candidate position, which reaches one cell beyond the agent's
current view, so pheromone changes outside the visible window changed the chosen
action. It also has to stay deterministic for a seed and never read other
robots' positions.
"""

from __future__ import annotations

import unittest

import numpy as np

from integrated_sar.scenarios import build_scenario
from integrated_sar.search_runtime import FrontierSearchPolicy, build_mock_search_side


def _policy_world(view_radius: int = 2):
    """One robot on an empty 20x20 grid at (10, 10); visible rows/cols 8..12."""
    scenario = build_scenario(seed=3, size=20, num_agents=1, num_goals=0, obstacle_density=0.0)
    env, obs, _ = build_mock_search_side(scenario, view_radius=view_radius)
    env.agents_pos = np.array([[10, 10]], dtype=np.int32)
    env.global_pheromone[:] = 0.0
    policy = FrontierSearchPolicy(view_radius=view_radius)

    # Unknown cells to the right (5, visible col 13 only reachable through the
    # candidate-centred window) and to the left (4), so the right move wins on
    # frontier gain and the left move is the runner-up.
    observations = np.full((20, 20), 0.0, dtype=np.float32)
    observations[10, 10] = -4.5
    observations[8:13, 13] = 0.5
    observations[8:12, 7] = 0.5
    obs = {"agent_0": observations}
    return env, obs, policy


class ObservablePheromoneTests(unittest.TestCase):
    def test_pheromone_outside_the_visible_window_does_not_change_the_action(self) -> None:
        env, obs, policy = _policy_world()
        action_without = policy(env, obs)["agent_0"]
        self.assertEqual(action_without, 3, "the frontier gain setup regressed")

        # (10, 13) is invisible (visible columns are 8..12) but sits inside the
        # candidate-centred window of the right move.
        env.global_pheromone[10, 13] = 5.0
        action_with = policy(env, obs)["agent_0"]
        self.assertEqual(
            action_with,
            action_without,
            "a pheromone change outside the visible window changed the action",
        )

    def test_pheromone_inside_the_visible_window_still_repels(self) -> None:
        env, obs, policy = _policy_world()
        action_without = policy(env, obs)["agent_0"]

        # (10, 12) is inside the visible window and inside the right candidate's
        # penalty window: the left move must win instead.
        env.global_pheromone[10, 12] = 5.0
        action_with = policy(env, obs)["agent_0"]
        self.assertEqual(action_without, 3)
        self.assertEqual(action_with, 2)

    def test_the_policy_never_reads_other_robots_positions(self) -> None:
        env, obs, policy = _policy_world()
        action_alone = policy(env, obs)["agent_0"]
        # A second team somewhere else: pheromone is the only separation channel.
        env.agents_pos = np.vstack([env.agents_pos, np.array([[2, 17]], dtype=np.int32)])
        env.global_pheromone[2, 17] = 1.0
        self.assertEqual(policy(env, obs)["agent_0"], action_alone)

    def test_the_policy_is_deterministic(self) -> None:
        env, obs, policy = _policy_world()
        env.global_pheromone[9, 12] = 1.0
        first = policy(env, obs)
        second = policy(env, obs)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
