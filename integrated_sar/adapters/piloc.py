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
from ..search_probe import closure_flags
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
        search_probe: Any = None,
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
        # B1: a read-only side recorder. None (the default) leaves the environment's
        # ``search_trace_enabled`` flag untouched, so an untouched run is the same code path
        # every recorded run was produced on.
        self.search_probe = search_probe
        rows, cols = tuple(int(value) for value in env.map_size)
        self.transform = CoordinateTransform(rows=rows, cols=cols)
        self._episode = 0
        self._track_ids: list[str] = []
        self._untracked_counter = 0
        # Reset-index -> id table for the current episode. ``goals_pos`` is delete-only, so
        # the array index drifts; this table is what makes an id mean one target all episode.
        self._id_by_reset_index: dict[int, str] = {}
        self._probe_alive: list[int] = []
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
        if self.search_probe is not None:
            # 探针接上才打开环境的只读快照开关，并按这一份放置数组重建 reset 序号表。
            # 卸载探针（默认）时这里一行都不执行，走的还是原来的代码路径。
            self.search_probe.note_placed(self._placed_at_reset)
            self._id_by_reset_index = {
                index: str(track_id) for index, (track_id, _) in enumerate(self._placed_at_reset)
            }
            self._probe_alive = list(range(len(self._placed_at_reset)))
            if hasattr(self.env, "search_trace_enabled"):
                self.env.search_trace_enabled = True
                self.env._trace_alive = list(range(len(self.env.goals_pos)))

    def _reset_index(self, track_id: Any) -> int | None:
        """The reset-order index behind a track id, or None when the id carries none.

        ``piloc-e{episode}-untracked-{n}`` ids are minted when the environment hands back
        more goals than the adapter was tracking; they name a real target but not a placement
        index, so they are reported instead of being guessed at.
        """
        text = str(track_id)
        marker = "-target-"
        position = text.rfind(marker)
        if position < 0:
            if self.search_probe is not None:
                self.search_probe.anomaly(f"id_without_reset_index:{text}")
            return None
        tail = text[position + len(marker):]
        return int(tail) if tail.lstrip("-").isdigit() else None

    def search_side_snapshot(self) -> dict[str, Any]:
        """Where the search side stands at the end of an episode (probe closing record)."""
        targets = []
        positions = list(getattr(self.env, "goals_pos", ()))
        for index, track_id in enumerate(self._track_ids):
            if index >= len(positions):
                continue
            position = positions[index]
            targets.append({"target_id": str(track_id),
                            "cell": [int(position[0]), int(position[1])]})
        return {
            "episode": int(self._episode),
            "agents": [[int(p[0]), int(p[1])] for p in getattr(self.env, "agents_pos", ())],
            "remaining_targets": targets,
            "remaining_count": len(positions),
            "all_found": len(positions) == 0,
            "env_steps": int(getattr(self.env, "steps", 0)),
        }

    def _note_probe_step(self, *, step: int, actions: Mapping[str, int],
                         alive_start: list[int], before_tracks: list[str],
                         detected: list[tuple[int, str]],
                         events: tuple[DetectionEvent, ...]) -> None:
        """Write one detection-instant record: the environment's own verdict, the pairs that
        satisfy its predicate, and what this adapter actually published."""
        probe = self.search_probe
        if probe is None:
            return
        trace = dict(getattr(self.env, "last_detection_trace", None) or {})
        env_deleted = [int(i) for i in trace.get("env_deleted_reset_indices", ())]
        pairs = list(trace.get("observed_pairs", ()))
        pair_reset = sorted({int(row["reset_index"]) for row in pairs})
        adapter_detected = [goal_index for goal_index, _ in detected]
        adapter_reset = sorted({
            alive_start[goal_index] for goal_index in adapter_detected
            if goal_index < len(alive_start)
        })
        alive_after = self._probe_alive
        env_alive_after = [int(i) for i in trace.get("alive_reset_indices_after", ())]
        trace_goals = [int(row["reset_index"]) for row in trace.get("goals_at_detection", ())]
        to_id = lambda indices: [self._id_by_reset_index.get(int(i), f"unknown:{i}")
                                 for i in sorted(set(indices))]
        closure = closure_flags(
            env_ids=to_id(env_deleted),
            pair_ids=to_id(pair_reset),
            adapter_ids=to_id(adapter_reset),
            alive_views_agree=(not trace
                               or (trace_goals == alive_start
                                   and env_alive_after == alive_after)),
        )
        probe.note_step({
            "kind": "step",
            "episode": int(self._episode),
            "step": int(step),
            "actions": {str(key): int(value) for key, value in actions.items()},
            "detected_by": {str(self._id_by_reset_index.get(alive_start[g], f"unknown:{g}")):
                            str(agent) for g, agent in detected
                            if g < len(alive_start)},
            "adapter_event_count": len(events),
            "adapter_event_target_ids": sorted(str(event.target_id) for event in events),
            "adapter_alive_before": [self._id_by_reset_index.get(i, f"unknown:{i}")
                                     for i in alive_start],
            "adapter_alive_after": [str(track) for track in self._track_ids],
            "environment_trace_present": bool(trace),
            "env_deleted_target_ids": to_id(env_deleted),
            "pair_observed_target_ids": to_id(pair_reset),
            "observed_pairs": pairs,
            "goals_at_detection": trace.get("goals_at_detection", []),
            "agents_before_move": trace.get("agents_before_move", []),
            "agents_after_move": trace.get("agents_after_move", []),
            "remaining_after_move": trace.get("remaining_after_move", []),
            "detection_rule_radius": trace.get("detection_rule_radius"),
            "reward_mode": trace.get("reward_mode"),
            "done": trace.get("done", {}),
            "rewards": trace.get("rewards", {}),
            "alive_bookkeeping_reinitialised": bool(
                trace.get("alive_bookkeeping_reinitialised", False)),
            "closure": closure,
        })

    def placed_goals(self) -> list[tuple[str, tuple[Any, ...]]]:
        """Every goal the scenario contains at reset, with the id it will be found under."""
        return list(getattr(self, "_placed_at_reset", ()))

    def step(self, current_time: float, dt: float) -> SearchStepResult:
        before_goals = [
            GridPosition(int(position[0]), int(position[1]))
            for position in self.env.goals_pos
        ]
        before_tracks = list(self._track_ids)
        # The reset-index list aligned with this step's array order, snapshotted before the
        # environment mutates it: every id translation in the record goes through this.
        alive_start = list(self._probe_alive)
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
        if self.search_probe is not None:
            self._probe_alive = [
                index for index in (self._reset_index(track) for track in self._track_ids)
                if index is not None
            ]
            self._note_probe_step(step=map_version, actions=actions, alive_start=alive_start,
                                  before_tracks=before_tracks, detected=detected,
                                  events=tuple(events))

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
