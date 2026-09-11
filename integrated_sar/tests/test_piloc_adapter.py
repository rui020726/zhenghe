from __future__ import annotations

import unittest

from integrated_sar.adapters.piloc import PilocSearchAdapter
from integrated_sar.contracts import WorldPosition
from integrated_sar.profiles import DeterministicTargetProfileProvider


class _FakePilocEnv:
    def __init__(self) -> None:
        self.map_size = (5, 5)
        self.map = [[0 for _ in range(5)] for _ in range(5)]
        self.agents = ["agent_0"]
        self.agents_pos = [[0, 0]]
        self.goals_pos = [[2, 2]]
        self.steps = 0

    def step(self, actions, obs, rescue_actions, two_channelmap, episode_horizon):
        del actions, rescue_actions, episode_horizon
        self.agents_pos = [[2, 1]]
        self.goals_pos = []
        self.steps += 1
        return obs, {"agent_0": 1.0}, {"agent_0": True}, [], two_channelmap

    def _is_line_of_sight_blocked(self, *args):
        del args
        return True


class PilocAdapterTests(unittest.TestCase):
    def test_visible_removed_goal_becomes_detection_event(self) -> None:
        env = _FakePilocEnv()
        adapter = PilocSearchAdapter(
            env=env,
            action_provider=lambda *_: {"agent_0": 0},
            profile_provider=DeterministicTargetProfileProvider(),
            obs={},
            two_channelmap={},
        )
        result = adapter.step(0.0, 1.0)
        self.assertEqual(len(result.detections), 1)
        event = result.detections[0]
        self.assertEqual(event.world_position, WorldPosition(0.5, 0.5))
        self.assertEqual(event.discovered_by, "agent_0")
        self.assertEqual(result.metrics["remaining_targets"], 0)
