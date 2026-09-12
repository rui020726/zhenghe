"""A full reset must also restore the real PILOC world, not just reload the map.

Regression for the third review round: ``reset_world`` only called
``Environment.load``, which replaces the map, positions, goals, action cache and
``steps`` but leaves ``global_pheromone`` / ``local_pheromone``, the position
history, the timestamp maps, ``check_repeat_pos``, ``comm_stats`` and the delayed
message buffer describing the previous episode -- and it did not reseed the two
RNGs the scenario was built from.
"""

from __future__ import annotations

import importlib.util
import random
import unittest

import numpy as np

from integrated_sar.contracts import GridPosition
from integrated_sar.piloc_runtime import reset_piloc_environment
from integrated_sar.scenarios import PILOC_OBS_RADIUS, SearchScenario

HAS_TIMEMAP = all(
    importlib.util.find_spec(name) is not None for name in ("gym", "PIL")
)


def _scenario(rows: int = 8, cols: int = 8) -> SearchScenario:
    obstacles = np.zeros((rows, cols), dtype=np.int32)
    return SearchScenario(
        obstacles=obstacles,
        agent_positions=(GridPosition(1, 1), GridPosition(2, 2)),
        goal_positions=(GridPosition(5, 5),),
        seed=5,
        source="test",
    )


class _StubTimemapEnvironment:
    """Mimics the parts of ``timemap.Environment`` a reset has to restore.

    ``load`` reproduces the real method's contract: it replaces the map, the
    positions, the goals, ``steps`` and ``last_actions``, and nothing else.
    """

    def __init__(self) -> None:
        self.obs_radius = PILOC_OBS_RADIUS
        self.num_agents = 2
        self.map = np.zeros((8, 8), dtype=np.int32)
        self.map_size = (8, 8)
        self.agents_pos = np.zeros((2, 2), dtype=np.int32)
        self.goals_pos = np.zeros((1, 2), dtype=np.int32)
        self.steps = 0
        self.last_actions = None
        self.dirty()

    def dirty(self) -> None:
        """Leave every field a previous episode would have polluted."""
        self.global_pheromone = np.full((8, 8), 7.0, dtype=np.float32)
        self.local_pheromone = {"agent_0": np.full((11, 11), 7.0, dtype=np.float32)}
        self.agent_positions_history = {"agent_0": np.array([58, 58])}
        self.global_timestamp_maps = np.full((8, 8), 0.25, dtype=np.float32)
        self.check_repeat_pos = [(3, 3)]
        self.comm_stats = {"success": 30, "failed": 0, "partial": 0, "delayed": 0, "total": 30}
        self.delayed_map_buffer = [object()]

    def calculate_pheromone(self, row: int, col: int) -> float:
        """Window sum, matching the real method's view-radius crop."""
        from integrated_sar.search_runtime import local_pheromone_window

        return float(
            local_pheromone_window(
                np.asarray(self.global_pheromone), row, col, self.obs_radius
            ).sum()
        )

    def load(self, map, agents_pos, goals_pos) -> None:
        self.map = np.copy(map)
        self.agents_pos = np.copy(agents_pos)
        self.goals_pos = np.copy(goals_pos)
        self.num_agents = agents_pos.shape[0]
        self.steps = 0
        self.last_actions = np.zeros((self.num_agents, 5, 11, 11), dtype=bool)


class PilocResetTests(unittest.TestCase):
    def test_reset_clears_pheromone_history_and_comm_state(self) -> None:
        environment = _StubTimemapEnvironment()
        scenario = _scenario()

        reset_piloc_environment(environment, scenario, seed=11)

        # The map, robots, goals and counters come from the scenario again.
        np.testing.assert_array_equal(environment.map, scenario.obstacles)
        np.testing.assert_array_equal(
            environment.agents_pos, [[1, 1], [2, 2]]
        )
        np.testing.assert_array_equal(environment.goals_pos, [[5, 5]])
        self.assertEqual(environment.steps, 0)

        # Derived state is rebuilt rather than inherited.
        expected_pheromone = np.zeros((8, 8), dtype=np.float32)
        expected_pheromone[1, 1] = 1.0
        expected_pheromone[2, 2] = 1.0
        np.testing.assert_array_equal(environment.global_pheromone, expected_pheromone)
        self.assertEqual(set(environment.local_pheromone), {"agent_0", "agent_1"})
        # _rebuild_position_state marks each start cell and measures that agent's
        # window in the same pass, so agent i only ever sees agents 0..i.
        self.assertEqual(float(environment.local_pheromone["agent_0"]), 1.0)
        self.assertEqual(
            float(environment.local_pheromone["agent_1"]), float(expected_pheromone.sum())
        )
        for index, position in enumerate(scenario.agent_positions):
            history = environment.agent_positions_history[f"agent_{index}"]
            self.assertEqual((int(history[0]), int(history[1])), (position.row, position.col))
        np.testing.assert_array_equal(
            environment.global_timestamp_maps, np.full((8, 8), 0.5, dtype=np.float32)
        )
        self.assertEqual(environment.check_repeat_pos, [])
        self.assertEqual(
            environment.comm_stats,
            {"success": 0, "failed": 0, "partial": 0, "delayed": 0, "total": 0},
        )
        self.assertEqual(environment.delayed_map_buffer, [])

    def test_reset_reseeds_both_random_generators(self) -> None:
        environment = _StubTimemapEnvironment()
        scenario = _scenario()

        reset_piloc_environment(environment, scenario, seed=13)
        draws = (random.random(), float(np.random.random()))
        reset_piloc_environment(environment, scenario, seed=13)
        again = (random.random(), float(np.random.random()))

        self.assertEqual(draws, again)


@unittest.skipUnless(HAS_TIMEMAP, "the PILOC environment needs gym and Pillow")
class RealPilocResetTests(unittest.TestCase):
    """The same checks against the real ``timemap.Environment``."""

    def test_second_episode_starts_from_a_clean_world(self) -> None:
        from integrated_sar.piloc_runtime import build_piloc_environment

        scenario = _scenario()
        environment = build_piloc_environment(scenario, seed=7)
        # Pollute it the way a finished episode would.
        environment.global_pheromone[:] = 5.0
        environment.comm_stats["success"] = 30
        environment.comm_stats["total"] = 30
        environment.delayed_map_buffer.append(object())
        environment.check_repeat_pos.append((0, 0))
        environment.steps = 42

        reset_piloc_environment(environment, scenario, seed=7)

        expected = np.zeros(scenario.obstacles.shape, dtype=np.float32)
        for position in scenario.agent_positions:
            expected[position.row, position.col] = 1.0
        np.testing.assert_array_equal(environment.global_pheromone, expected)
        self.assertEqual(environment.comm_stats["total"], 0)
        self.assertEqual(environment.delayed_map_buffer, [])
        self.assertEqual(environment.check_repeat_pos, [])
        self.assertEqual(environment.steps, 0)


if __name__ == "__main__":
    unittest.main()
