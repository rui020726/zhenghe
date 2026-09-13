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
import types
import unittest

import numpy as np

from integrated_sar.contracts import GridPosition
from integrated_sar.errors import CheckpointIncompatibleError
from integrated_sar.piloc_runtime import reset_piloc_environment
from integrated_sar.scenarios import PILOC_GRID_SIZE, PILOC_OBS_RADIUS, SearchScenario

# timemap imports gym, matplotlib and Pillow at module scope, so a machine without
# any of them cannot import the environment at all. Only that specific condition is
# a skip: once the three libraries are present, a failing import is a real
# regression in PILOC or in the loader and must fail the test rather than be
# reported as a missing dependency.
TIMEMAP_IMPORT_NAMES = ("gym", "PIL", "matplotlib")


def _timemap_skip_reason() -> str | None:
    missing = [
        name for name in TIMEMAP_IMPORT_NAMES if importlib.util.find_spec(name) is None
    ]
    if missing:
        return f"the PILOC environment needs {', '.join(missing)}"
    return None


TIMEMAP_SKIP_REASON = _timemap_skip_reason()


def _scenario(rows: int = 8, cols: int = 8) -> SearchScenario:
    obstacles = np.zeros((rows, cols), dtype=np.int32)
    return SearchScenario(
        obstacles=obstacles,
        agent_positions=(GridPosition(1, 1), GridPosition(2, 2)),
        goal_positions=(GridPosition(5, 5),),
        seed=5,
        source="test",
    )


class _StandInTimemapEnvironment:
    """Stands in for ``timemap.Environment`` when it cannot be imported.

    Only ``__init__``, ``load`` and ``calculate_pheromone`` decide what a reset has
    to restore, and this class mirrors their real contracts (see
    ``piloc/src/Map/timemap.py``): ``__init__`` builds the map, robots and goals and
    marks the drawn start cells in the pheromone map; ``load`` replaces the map,
    positions, goals, ``steps`` and the action cache and nothing else. Going through
    it lets ``build_piloc_environment`` itself be exercised -- the 60x60 guard, the
    construction and the derived-state rebuild -- on a machine that has no
    gym, matplotlib or Pillow.
    """

    def __init__(self, **kwargs: object) -> None:
        length = int(kwargs.get("map_length", PILOC_GRID_SIZE))
        self.obs_radius = int(kwargs.get("obs_radius", PILOC_OBS_RADIUS))
        self.num_agents = int(kwargs.get("num_agents", 2))
        self.num_goals = int(kwargs.get("num_goals", 1))
        self.agents = [f"agent_{index}" for index in range(self.num_agents)]
        self.map = np.zeros((length, length), dtype=np.int32)
        self.map_size = (length, length)
        self.agents_pos = np.stack(
            [np.array([length // 4, length // 4], dtype=np.int32)] * self.num_agents
        )
        self.goals_pos = np.zeros((self.num_goals, 2), dtype=np.int32)
        self.steps = 0
        self.global_pheromone = np.zeros((length, length), dtype=np.float32)
        for row, col in self.agents_pos:
            self.global_pheromone[int(row), int(col)] = 1.0
        self.local_pheromone = {
            f"agent_{index}": self.calculate_pheromone(
                int(self.agents_pos[index][0]), int(self.agents_pos[index][1])
            )
            for index in range(self.num_agents)
        }
        self.agent_positions_history = {
            f"agent_{index}": self.agents_pos[index] for index in range(self.num_agents)
        }
        self.global_timestamp_maps = np.full((length, length), 0.5, dtype=np.float32)
        self.check_repeat_pos: list[tuple[int, int]] = []
        self.comm_stats = {"success": 0, "failed": 0, "partial": 0, "delayed": 0, "total": 0}
        self.delayed_map_buffer: list[object] = []
        self.last_actions = np.zeros(
            (self.num_agents, 5, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
            dtype=bool,
        )

    def calculate_pheromone(self, row: int, col: int) -> float:
        """Window sum, matching the real method's view-radius crop."""
        from integrated_sar.search_runtime import local_pheromone_window

        return float(
            local_pheromone_window(
                np.asarray(self.global_pheromone), row, col, self.obs_radius
            ).sum()
        )

    def load(self, map, agents_pos, goals_pos) -> None:  # noqa: A002 - real signature
        self.map = np.copy(map)
        self.agents_pos = np.copy(agents_pos)
        self.goals_pos = np.copy(goals_pos)
        self.num_agents = agents_pos.shape[0]
        self.map_size = (self.map.shape[0], self.map.shape[1])
        self.steps = 0
        self.last_actions = np.zeros(
            (self.num_agents, 5, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
            dtype=bool,
        )

    def reseed(self, seed: int) -> None:
        """Mirror the real environment's per-instance RNG replay."""
        self._py_rng = random.Random(seed)


def _stand_in_timemap_module() -> types.ModuleType:
    module = types.ModuleType("integrated_sar_stand_in_timemap")
    module.Environment = _StandInTimemapEnvironment
    return module


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

    def load(self, map, agents_pos, goals_pos) -> None:  # noqa: A002 - real signature
        self.map = np.copy(map)
        self.agents_pos = np.copy(agents_pos)
        self.goals_pos = np.copy(goals_pos)
        self.num_agents = agents_pos.shape[0]
        self.steps = 0
        self.last_actions = np.zeros((self.num_agents, 5, 11, 11), dtype=bool)

    def reseed(self, seed: int) -> None:
        """Mirror the real environment's per-instance RNG replay."""
        self._py_rng = random.Random(seed)


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

    def test_reset_replays_the_environment_rng_stream(self) -> None:
        """A reset must replay the per-instance stream, not the global modules.

        The environment owns its RNG now, so ``reset_piloc_environment`` calls
        ``reseed`` instead of ``random.seed``/``np.random.seed``. Two resets on one
        seed must leave the environment's own stream in the same state, and must
        not touch the global streams the training loop consumes.
        """
        environment = _StubTimemapEnvironment()
        scenario = _scenario()

        random.seed(999)
        np.random.seed(999)
        global_before = (random.random(), float(np.random.random()))

        reset_piloc_environment(environment, scenario, seed=13)
        draws = (environment._py_rng.random(), environment._py_rng.random())
        reset_piloc_environment(environment, scenario, seed=13)
        again = (environment._py_rng.random(), environment._py_rng.random())

        self.assertEqual(draws, again)
        # The globals were not reseeded by the reset: replaying them continues the
        # 999 stream exactly, proving the fixed validation seed can't be consumed.
        random.seed(999)
        np.random.seed(999)
        self.assertEqual((random.random(), float(np.random.random())), global_before)


class BuildPilocEnvironmentTests(unittest.TestCase):
    """The production construction path, with ``timemap`` stood in for.

    ``build_piloc_environment`` refuses any scenario that is not 60x60 because the
    trained policy's timestamp maps are built for that canvas, and it installs the
    scenario through ``reset_piloc_environment``, whose ``load`` replaces the map,
    positions and goals the constructor randomly drew. Both behaviours are checked
    here without gym, matplotlib or Pillow.
    """

    def setUp(self) -> None:
        import integrated_sar.piloc_runtime as piloc_runtime

        self.piloc_runtime = piloc_runtime
        self._saved = piloc_runtime.import_timemap_module
        piloc_runtime.import_timemap_module = lambda root=None: _stand_in_timemap_module()
        self.addCleanup(
            setattr, piloc_runtime, "import_timemap_module", self._saved
        )

    def test_a_legal_scenario_builds_and_resets_from_the_shared_map(self) -> None:
        size = PILOC_GRID_SIZE
        scenario = _scenario(size, size)
        environment = self.piloc_runtime.build_piloc_environment(scenario, seed=7)

        np.testing.assert_array_equal(environment.map, scenario.obstacles)
        np.testing.assert_array_equal(
            environment.agents_pos,
            [[position.row, position.col] for position in scenario.agent_positions],
        )
        np.testing.assert_array_equal(
            environment.goals_pos,
            [[position.row, position.col] for position in scenario.goal_positions],
        )
        self.assertEqual(environment.steps, 0)

        # A finished episode leaves trails, traffic counters and a delayed buffer.
        environment.global_pheromone[:] = 5.0
        environment.comm_stats.update({"success": 30, "total": 30})
        environment.delayed_map_buffer.append(object())
        environment.check_repeat_pos.append((0, 0))
        environment.steps = 42

        reset_piloc_environment(environment, scenario, seed=7)

        expected = np.zeros((size, size), dtype=np.float32)
        for position in scenario.agent_positions:
            expected[position.row, position.col] = 1.0
        np.testing.assert_array_equal(environment.global_pheromone, expected)
        self.assertEqual(environment.comm_stats["total"], 0)
        self.assertEqual(environment.delayed_map_buffer, [])
        self.assertEqual(environment.check_repeat_pos, [])
        self.assertEqual(environment.steps, 0)

    def test_an_off_size_scenario_is_refused_not_resized(self) -> None:
        with self.assertRaises(CheckpointIncompatibleError):
            self.piloc_runtime.build_piloc_environment(_scenario(8, 8), seed=7)


@unittest.skipIf(TIMEMAP_SKIP_REASON is not None, TIMEMAP_SKIP_REASON or "")
class RealPilocResetTests(unittest.TestCase):
    """The same checks against the real ``timemap.Environment``.

    The scenario must be 60x60: ``build_piloc_environment`` refuses anything else,
    so a smaller map would fail here for a reason that has nothing to do with reset.
    """

    def test_second_episode_starts_from_a_clean_world(self) -> None:
        from integrated_sar.piloc_runtime import build_piloc_environment

        size = PILOC_GRID_SIZE
        scenario = _scenario(size, size)
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


@unittest.skipIf(TIMEMAP_SKIP_REASON is not None, TIMEMAP_SKIP_REASON or "")
class TimemapDefaultConstructionTests(unittest.TestCase):
    """``Environment()`` has to work with its own defaults.

    ``self.map`` used to be assigned only inside the ``if test:`` branch, so the
    default ``test=False`` reached ``map_partition(self.map)`` with no such attribute
    and raised ``AttributeError``. That made ``piloc/train.py`` unable to start, and
    forced every other caller to pre-assign ``map`` on a bare instance before running
    ``__init__``.
    """

    def test_default_construction_produces_a_usable_map(self) -> None:
        import random as random_module

        from integrated_sar.piloc_runtime import import_timemap_module

        timemap = import_timemap_module()
        random_module.seed(11)
        np.random.seed(11)

        environment = timemap.Environment()

        self.assertEqual(environment.map.shape, (PILOC_GRID_SIZE, PILOC_GRID_SIZE))
        self.assertTrue(
            np.isin(environment.map, (0, 1)).all(), "the obstacle map must be binary"
        )
        self.assertGreater(
            int(np.count_nonzero(environment.map != 1)),
            0,
            "a map with no free cell cannot be searched",
        )
        self.assertEqual(len(environment.agents_pos), len(environment.agents))
        self.assertGreater(len(environment.goals_pos), 0)


if __name__ == "__main__":
    unittest.main()
