"""Non-invasive adapter around piloc.src.Map.timemap.Environment."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..contracts import (
    CoordinateTransform,
    DetectionEvent,
    GridPosition,
    SearchStepResult,
)
from ..profiles import TargetProfileProvider
from ..search_runtime import initialize_timestamp_observations, initialize_two_channel_map


ActionProvider = Callable[[Any, Any, Any], Mapping[str, int]]


class PilocSearchAdapter:
    """
    Wraps the current PILOC environment without importing it directly.

    The adapter reconstructs which pre-move targets were observed, because the
    current environment deletes observed goals inside ``Environment.step``.
    A future native integration should emit DetectionEvent before that deletion.

    ``reset_world`` is an optional callback that restores the underlying search
    environment to the episode-start scenario (the mock environment resets
    itself; the real timemap environment re-loads the scenario map, positions
    and pheromone). Without it, ``reset`` can only rebuild the adapter's own
    tracking state.
    """

    def __init__(
        self,
        *,
        env: Any,
        action_provider: ActionProvider,
        profile_provider: TargetProfileProvider,
        obs: Any,
        two_channelmap: Any,
        rescue_actions: Mapping[str, Any] | None = None,
        episode_horizon: int = 250,
        detection_radius: int = 5,
        confidence: float = 1.0,
        reset_world: Callable[[], None] | None = None,
    ) -> None:
        self.env = env
        self.action_provider = action_provider
        self.profile_provider = profile_provider
        self.obs = obs
        self.two_channelmap = two_channelmap
        self.rescue_actions = dict(rescue_actions or {})
        self.episode_horizon = int(episode_horizon)
        self.detection_radius = int(detection_radius)
        self.confidence = float(confidence)
        self._reset_world = reset_world
        rows, cols = tuple(int(value) for value in env.map_size)
        self.transform = CoordinateTransform(rows=rows, cols=cols)
        self._episode = 0
        self._track_ids: list[str] = []
        self._untracked_counter = 0
        self.reset()

    def _view_radius(self) -> int:
        for name in ("view_radius", "obs_radius"):
            value = getattr(self.env, name, None)
            if value is not None:
                return int(value)
        return self.detection_radius

    def reset(self) -> None:
        """Start a new episode: track ids are namespaced by episode, so they never collide.

        When a ``reset_world`` callback is wired, the search environment is also
        restored to the original scenario -- initial robot positions, the full
        set of hidden targets, zeroed pheromone and counters -- and the adapter's
        timestamp observations and channel maps are rebuilt from it.
        """
        self._episode += 1
        if self._reset_world is not None:
            self._reset_world()
            self.obs = initialize_timestamp_observations(
                self.env, view_radius=self._view_radius()
            )
            self.two_channelmap = initialize_two_channel_map(self.obs)
        # Untracked-target numbering is per episode, like the track ids themselves.
        self._untracked_counter = 0
        self._track_ids = [
            f"piloc-e{self._episode}-target-{index}"
            for index in range(len(self.env.goals_pos))
        ]
        # The placed set has to be captured here and not later: timemap removes a goal once
        # it is found, so reading goals_pos after step one would under-count placement and
        # silently lose the denominator for "never discovered".
        self._placed_at_reset = [
            (track_id, tuple(position))
            for track_id, position in zip(self._track_ids, self.env.goals_pos)
        ]

    def placed_goals(self) -> list[tuple[str, tuple[Any, ...]]]:
        """Every goal the scenario contains at reset, with the id it will be found under."""
        return list(getattr(self, "_placed_at_reset", ()))

    def step(self, current_time: float, dt: float) -> SearchStepResult:
        before_goals = [
            GridPosition(int(position[0]), int(position[1]))
            for position in self.env.goals_pos
        ]
        before_tracks = list(self._track_ids)
        actions = dict(self.action_provider(self.env, self.obs, self.two_channelmap))

        (
            self.obs,
            rewards,
            done,
            repeat_mark,
            self.two_channelmap,
        ) = self.env.step(
            actions,
            self.obs,
            self.rescue_actions,
            self.two_channelmap,
            self.episode_horizon,
        )

        detected: list[tuple[int, str]] = []
        post_agents = list(self.env.agents_pos)
        for goal_index, goal in enumerate(before_goals):
            for agent_index, agent_position in enumerate(post_agents):
                row, col = int(agent_position[0]), int(agent_position[1])
                if (
                    abs(row - goal.row) <= self.detection_radius
                    and abs(col - goal.col) <= self.detection_radius
                    and self._line_of_sight(row, col, goal.row, goal.col)
                ):
                    agent_id = (
                        str(self.env.agents[agent_index])
                        if agent_index < len(self.env.agents)
                        else f"agent_{agent_index}"
                    )
                    detected.append((goal_index, agent_id))
                    break

        detected_indices = {index for index, _ in detected}
        map_version = int(getattr(self.env, "steps", 0))
        observed_at = current_time + dt
        events: list[DetectionEvent] = []
        for goal_index, agent_id in detected:
            if goal_index >= len(before_tracks):
                continue
            target_id = before_tracks[goal_index]
            grid_position = before_goals[goal_index]
            events.append(
                DetectionEvent(
                    event_id=f"detection:{target_id}:{map_version}",
                    target_id=target_id,
                    grid_position=grid_position,
                    world_position=self.transform.to_world(grid_position),
                    detected_at=observed_at,
                    discovered_by=agent_id,
                    confidence=self.confidence,
                    profile=self.profile_provider.profile_for(
                        target_id, grid_position, observed_at
                    ),
                    map_version=map_version,
                )
            )

        remaining_tracks = [
            track_id
            for index, track_id in enumerate(before_tracks)
            if index not in detected_indices
        ]
        post_goal_count = len(self.env.goals_pos)
        if post_goal_count > len(remaining_tracks):
            # A goal the adapter never tracked (the environment can add or move goals).
            # The id must not depend on map_version, or the same physical target would
            # receive a fresh id every tick and be rescued several times over.
            for _ in range(post_goal_count - len(remaining_tracks)):
                self._untracked_counter += 1
                remaining_tracks.append(
                    f"piloc-e{self._episode}-untracked-{self._untracked_counter}"
                )
        self._track_ids = remaining_tracks[:post_goal_count]

        return SearchStepResult(
            detections=tuple(events),
            metrics={
                "search_rewards": dict(rewards),
                "remaining_targets": post_goal_count,
                "search_done": all(bool(value) for value in done.values()) if done else False,
                "repeat_agents": tuple(repeat_mark),
                "map_version": map_version,
            },
        )

    def _line_of_sight(self, row: int, col: int, goal_row: int, goal_col: int) -> bool:
        checker = getattr(self.env, "_is_line_of_sight_blocked", None)
        if checker is None:
            return True
        # Despite its name, the existing method returns True when the path is clear.
        return bool(checker(row, col, goal_row, goal_col, self.env.map))
