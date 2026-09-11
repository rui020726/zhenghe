"""Search-side runtime helpers shared by the mock and the real PILOC environment.

Everything here is deliberately numpy-only so that the joint driver can validate the
full search -> registry -> allocation -> execution data flow on machines that have
neither torch nor the PILOC checkpoints installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .scenarios import SearchScenario, has_line_of_sight

# Mirrors ``piloc/src/Map/timemap.action_list``: up, down, left, right.
ACTION_DELTAS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

UNKNOWN_VALUE = 0.5
OBSTACLE_VALUE = 1.0
FREE_VALUE = 0.0
SELF_VALUE = -4.5


def _line_of_sight_checker(env: Any):
    """Reuse the environment's own checker when it exposes one, else fall back."""
    checker = getattr(env, "_is_line_of_sight_blocked", None)
    if checker is None:
        return lambda r0, c0, r1, c1, env_map: has_line_of_sight(
            env_map, (r0, c0), (r1, c1)
        )
    # Despite its name, PILOC's method returns True when the path is clear.
    return checker


def _as_position_array(positions: Sequence[Any]) -> np.ndarray:
    """Accept ``GridPosition`` objects or ``(row, col)`` pairs interchangeably."""
    pairs = [
        (int(position.row), int(position.col))
        if hasattr(position, "row")
        else (int(position[0]), int(position[1]))
        for position in positions
    ]
    if not pairs:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(pairs, dtype=np.int32)


def initialize_timestamp_observations(env: Any, view_radius: int = 5) -> dict[str, np.ndarray]:
    """Build the per-agent timestamp maps PILOC's policy consumes.

    Cell values follow ``piloc/test.initialize_obs``: ``0.5`` unknown, ``0`` observed
    free, ``1`` observed obstacle and ``-4.5`` the agent itself.
    """
    rows, cols = (int(value) for value in env.map_size)
    checker = _line_of_sight_checker(env)
    offsets = range(-view_radius, view_radius + 1)
    observations: dict[str, np.ndarray] = {}

    for index, (row, col) in enumerate(np.asarray(env.agents_pos).reshape(-1, 2)):
        row, col = int(row), int(col)
        agent_obs = np.full((rows, cols), UNKNOWN_VALUE, dtype=np.float32)
        agent_obs[row, col] = SELF_VALUE
        for d_row in offsets:
            for d_col in offsets:
                neighbour_row, neighbour_col = row + d_row, col + d_col
                if not (0 <= neighbour_row < rows and 0 <= neighbour_col < cols):
                    continue
                if not checker(row, col, neighbour_row, neighbour_col, env.map):
                    continue
                if env.map[neighbour_row, neighbour_col] == 1:
                    agent_obs[neighbour_row, neighbour_col] = OBSTACLE_VALUE
                elif (neighbour_row, neighbour_col) != (row, col):
                    agent_obs[neighbour_row, neighbour_col] = FREE_VALUE
        observations[f"agent_{index}"] = agent_obs

    return observations


def initialize_two_channel_map(observations: Mapping[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Derive the (obstacle, explore) channel pair from timestamp maps."""
    return {
        agent_id: two_channel_from_timestamp(agent_obs)
        for agent_id, agent_obs in observations.items()
    }


def two_channel_from_timestamp(agent_obs: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorised equivalent of ``timemap.Environment.two_channel_map``."""
    rows, cols = agent_obs.shape
    obstacle_map = np.full((rows, cols), -1.0, dtype=np.float32)
    explore_map = np.full((rows, cols), -0.5, dtype=np.float32)

    is_obstacle = agent_obs == OBSTACLE_VALUE
    is_unknown = agent_obs == UNKNOWN_VALUE
    is_visited_free = (agent_obs >= 0.0) & (agent_obs <= 0.3) & ~is_obstacle

    obstacle_map[is_obstacle] = OBSTACLE_VALUE
    explore_map[is_obstacle] = FREE_VALUE
    explore_map[is_unknown] = UNKNOWN_VALUE
    obstacle_map[is_visited_free] = FREE_VALUE
    explore_map[is_visited_free] = agent_obs[is_visited_free]

    other = ~(is_obstacle | is_unknown | is_visited_free)
    obstacle_map[other] = agent_obs[other]
    explore_map[other] = agent_obs[other]
    return {"obstacle_map": obstacle_map, "explore_map": explore_map}


def local_pheromone_window(
    global_pheromone: np.ndarray, row: int, col: int, view_radius: int = 5
) -> np.ndarray:
    """Crop a zero-padded ``(2r+1, 2r+1)`` pheromone window around one agent."""
    height, width = global_pheromone.shape
    size = 2 * view_radius + 1
    row_min, row_max = max(0, row - view_radius), min(height, row + view_radius + 1)
    col_min, col_max = max(0, col - view_radius), min(width, col + view_radius + 1)
    window = np.zeros((size, size), dtype=global_pheromone.dtype)
    window[row_min - (row - view_radius): row_max - (row - view_radius),
           col_min - (col - view_radius): col_max - (col - view_radius)] = (
        global_pheromone[row_min:row_max, col_min:col_max]
    )
    return window


class MockGridSearchEnv:
    """A numpy stand-in for ``piloc.src.Map.timemap.Environment``.

    It reproduces only the surface :class:`~integrated_sar.adapters.piloc.PilocSearchAdapter`
    relies on, with two intentional deviations from PILOC:

    * discovered targets are removed from the scene (same as PILOC), but undiscovered
      ones stay put -- PILOC random-walks its goals through the global ``random``
      module, which would make a mock episode irreproducible;
    * rewards are informational only, because nothing trains here.
    """

    def __init__(
        self,
        *,
        obstacles: np.ndarray,
        agent_positions: Sequence[Sequence[int]],
        goal_positions: Sequence[Sequence[int]],
        view_radius: int = 5,
        pheromone_decay: float = 0.95,
    ) -> None:
        self.map = np.asarray(obstacles, dtype=np.int32)
        self.map_size = (int(self.map.shape[0]), int(self.map.shape[1]))
        self.agents_pos = _as_position_array(agent_positions)
        self.goals_pos = _as_position_array(goal_positions)
        self.num_agents = int(self.agents_pos.shape[0])
        self.agents = [f"agent_{index}" for index in range(self.num_agents)]
        self.view_radius = int(view_radius)
        self.pheromone_decay = float(pheromone_decay)
        self.steps = 0
        self.current_time = 0.0
        self.global_pheromone = np.zeros(self.map_size, dtype=np.float32)
        for row, col in self.agents_pos:
            self.global_pheromone[int(row), int(col)] = 1.0

    @classmethod
    def from_scenario(cls, scenario: SearchScenario, **kwargs: Any) -> "MockGridSearchEnv":
        return cls(
            obstacles=scenario.obstacles,
            agent_positions=scenario.agent_positions,
            goal_positions=scenario.goal_positions,
            **kwargs,
        )

    def _is_line_of_sight_blocked(
        self, start_row: int, start_col: int, end_row: int, end_col: int, env_map: Any
    ) -> bool:
        return has_line_of_sight(env_map, (start_row, start_col), (end_row, end_col))

    def step(
        self,
        actions: Mapping[str, Any],
        obs: Mapping[str, np.ndarray],
        rescue_actions: Mapping[str, Any] | None,
        two_channelmap: Mapping[str, Any] | None,
        episode_horizon: Any = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], list[str], dict[str, dict[str, np.ndarray]]]:
        del rescue_actions, two_channelmap, episode_horizon
        rows, cols = self.map_size
        next_positions = self.agents_pos.copy()
        rewards: dict[str, float] = {agent_id: 0.0 for agent_id in actions}

        for agent_id, action in actions.items():
            index = int(str(agent_id).rsplit("_", 1)[-1])
            d_row, d_col = ACTION_DELTAS[int(action)]
            row, col = next_positions[index]
            new_row, new_col = int(row) + d_row, int(col) + d_col
            if not (0 <= new_row < rows and 0 <= new_col < cols) or self.map[new_row, new_col] == 1:
                rewards[agent_id] -= 1.0
                continue
            next_positions[index] = (new_row, new_col)

        self.agents_pos = next_positions
        self.global_pheromone *= self.pheromone_decay
        for row, col in self.agents_pos:
            self.global_pheromone[int(row), int(col)] = min(
                10.0, float(self.global_pheromone[int(row), int(col)]) + 1.0
            )

        discovered: set[int] = set()
        for agent_id in actions:
            index = int(str(agent_id).rsplit("_", 1)[-1])
            row, col = (int(value) for value in self.agents_pos[index])
            for goal_index, goal in enumerate(self.goals_pos):
                goal_row, goal_col = int(goal[0]), int(goal[1])
                if (
                    abs(row - goal_row) <= self.view_radius
                    and abs(col - goal_col) <= self.view_radius
                    and self._is_line_of_sight_blocked(row, col, goal_row, goal_col, self.map)
                ):
                    discovered.add(goal_index)

        if discovered:
            keep = [
                goal
                for goal_index, goal in enumerate(self.goals_pos)
                if goal_index not in discovered
            ]
            self.goals_pos = (
                np.asarray(keep, dtype=np.int32).reshape(-1, 2)
                if keep
                else np.zeros((0, 2), dtype=np.int32)
            )

        self.steps += 1
        self.current_time += 1.0
        updated_obs = self._refresh_observations(obs)
        updated_channels = initialize_two_channel_map(updated_obs)
        done = {agent_id: len(self.goals_pos) == 0 for agent_id in actions}
        return updated_obs, rewards, done, [], updated_channels

    def _refresh_observations(
        self, obs: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        rows, cols = self.map_size
        updated: dict[str, np.ndarray] = {}
        offsets = range(-self.view_radius, self.view_radius + 1)
        for index, (row, col) in enumerate(self.agents_pos):
            agent_id = f"agent_{index}"
            row, col = int(row), int(col)
            previous = np.asarray(obs.get(agent_id), dtype=np.float32).copy()
            if previous.shape != (rows, cols):
                previous = np.full((rows, cols), UNKNOWN_VALUE, dtype=np.float32)
            previous[previous == SELF_VALUE] = FREE_VALUE
            for d_row in offsets:
                for d_col in offsets:
                    neighbour_row, neighbour_col = row + d_row, col + d_col
                    if not (0 <= neighbour_row < rows and 0 <= neighbour_col < cols):
                        continue
                    if not self._is_line_of_sight_blocked(
                        row, col, neighbour_row, neighbour_col, self.map
                    ):
                        continue
                    if self.map[neighbour_row, neighbour_col] == 1:
                        previous[neighbour_row, neighbour_col] = OBSTACLE_VALUE
                    elif (neighbour_row, neighbour_col) != (row, col):
                        previous[neighbour_row, neighbour_col] = FREE_VALUE
            previous[row, col] = SELF_VALUE
            updated[agent_id] = previous
        return updated


class FrontierSearchPolicy:
    """Deterministic greedy frontier policy used when no search checkpoint is given.

    Each candidate action is scored by how many still-unknown cells it would bring
    inside the agent's view, plus a separation bonus capped at the view radius --
    every searcher keeps its own timestamp map, so without that bonus a second robot
    just trails the first through cells it has not personally observed. Illegal moves
    are only ever chosen when every move is illegal, and ties break on the action
    index so a seed fully determines the episode.
    """

    def __init__(
        self,
        *,
        view_radius: int = 5,
        separation_weight: float = 1.0,
        illegal_penalty: float = 1e6,
    ) -> None:
        self.view_radius = int(view_radius)
        self.separation_weight = float(separation_weight)
        self.illegal_penalty = float(illegal_penalty)

    def __call__(
        self,
        env: Any,
        obs: Mapping[str, np.ndarray],
        two_channelmap: Mapping[str, Any] | None = None,
    ) -> dict[str, int]:
        del two_channelmap
        rows, cols = (int(value) for value in env.map_size)
        obstacles = np.asarray(env.map)
        positions = np.asarray(env.agents_pos).reshape(-1, 2)
        candidate_positions = self._candidate_positions(positions, obstacles, rows, cols)
        actions: dict[str, int] = {}

        for index, agent_id in enumerate(env.agents):
            agent_obs = np.asarray(obs[agent_id], dtype=np.float32)
            unknown = agent_obs == UNKNOWN_VALUE
            best_action, best_score = 0, -np.inf
            for action_index, (row, col) in enumerate(candidate_positions[index]):
                if row < 0:
                    score = -self.illegal_penalty
                else:
                    score = float(self._frontier_gain(unknown, row, col, rows, cols))
                    score += self.separation_weight * self._separation(
                        index, row, col, positions, self.view_radius
                    )
                if score > best_score:
                    best_action, best_score = action_index, score
            actions[str(agent_id)] = best_action

        return actions

    def _candidate_positions(
        self,
        positions: np.ndarray,
        obstacles: np.ndarray,
        rows: int,
        cols: int,
    ) -> list[list[tuple[int, int]]]:
        candidates: list[list[tuple[int, int]]] = []
        for row, col in positions:
            row, col = int(row), int(col)
            options: list[tuple[int, int]] = []
            for d_row, d_col in ACTION_DELTAS:
                new_row, new_col = row + d_row, col + d_col
                if (
                    0 <= new_row < rows
                    and 0 <= new_col < cols
                    and obstacles[new_row, new_col] != 1
                ):
                    options.append((new_row, new_col))
                else:
                    options.append((-1, -1))
            candidates.append(options)
        return candidates

    def _frontier_gain(self, unknown: np.ndarray, row: int, col: int, rows: int, cols: int) -> int:
        radius = self.view_radius
        row_min, row_max = max(0, row - radius), min(rows, row + radius + 1)
        col_min, col_max = max(0, col - radius), min(cols, col + radius + 1)
        return int(np.count_nonzero(unknown[row_min:row_max, col_min:col_max]))

    @staticmethod
    def _separation(index: int, row: int, col: int, positions: np.ndarray, cap: int) -> float:
        distances = [
            max(abs(row - int(other[0])), abs(col - int(other[1])))
            for other_index, other in enumerate(positions)
            if other_index != index
        ]
        if not distances:
            return 0.0
        return float(min(min(distances), cap))


def build_mock_search_side(
    scenario: SearchScenario,
    *,
    view_radius: int = 5,
) -> tuple[MockGridSearchEnv, dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Create a mock environment plus its initial observations and channel maps."""
    env = MockGridSearchEnv.from_scenario(scenario, view_radius=view_radius)
    obs = initialize_timestamp_observations(env, view_radius=view_radius)
    return env, obs, initialize_two_channel_map(obs)
