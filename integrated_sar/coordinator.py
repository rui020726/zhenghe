"""Coordination policies that optimize, but do not replace, task allocation."""

from __future__ import annotations

import math

from .contracts import CoordinationAdvice, CoordinatorInput, TaskStatus


def _clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


class PassthroughCoordinator:
    """No-op baseline used for coordinator ablations."""

    def __init__(self, validity: float = 1.0) -> None:
        self.validity = float(validity)
        self._revision = 0

    def advise(self, state: CoordinatorInput) -> CoordinationAdvice:
        self._revision += 1
        return CoordinationAdvice(
            revision=self._revision,
            generated_at=state.current_time,
            valid_until=state.current_time + self.validity,
        )

    def reset(self) -> None:
        """Start a new episode: advice numbering restarts, so revisions stay comparable."""
        self._revision = 0


class HeuristicCoordinator:
    """Interpretable first-stage coordinator; later replaceable by a learned policy."""

    def __init__(
        self,
        validity: float = 5.0,
        urgency_horizon: float = 100.0,
        urgency_weight: float = 0.45,
        progress_weight: float = 0.25,
        severity_weight: float = 0.20,
        confidence_weight: float = 0.10,
        compatibility_weight: float = 0.65,
        distance_weight: float = 0.35,
        busy_penalty: float = 0.40,
    ) -> None:
        self.validity = float(validity)
        self.urgency_horizon = max(float(urgency_horizon), 1e-6)
        self.urgency_weight = float(urgency_weight)
        self.progress_weight = float(progress_weight)
        self.severity_weight = float(severity_weight)
        self.confidence_weight = float(confidence_weight)
        self.compatibility_weight = float(compatibility_weight)
        self.distance_weight = float(distance_weight)
        self.busy_penalty = float(busy_penalty)
        self._revision = 0

    def reset(self) -> None:
        """Start a new episode: advice numbering restarts, so revisions stay comparable."""
        self._revision = 0

    def advise(self, state: CoordinatorInput) -> CoordinationAdvice:
        """Score every (robot, task) pair from that robot's own belief.

        Invariant 9: the coordinator is an online module, so it may only use
        information that reached a robot over the communication chain. Every
        number here is derived from ``AgentSnapshot.task_beliefs`` -- never from
        the registry's task snapshots -- so a robot with a stale belief, or no
        belief at all, cannot be biased towards a target it has not heard about
        and cannot receive a task update that never arrived.
        """
        self._revision += 1
        task_by_id = {task.task_id: task for task in state.tasks if task.active}
        task_bias: dict[str, float] = {}
        pair_bias: dict[tuple[str, str], float] = {}
        replan_agents: set[str] = set()

        for agent in state.agents:
            if agent.current_task and agent.current_task not in task_by_id:
                replan_agents.add(agent.agent_id)

            known = agent.known_tasks
            for belief in agent.task_beliefs:
                task_id = belief.task_id
                if task_id not in task_by_id:
                    # Unheard-of or already closed in the registry: no bias at all.
                    continue
                if known is not None and task_id not in known:
                    continue

                time_left = belief.deadline - state.current_time
                urgency = 1.0 - _clamp(time_left / self.urgency_horizon, 0.0, 1.0)
                unfinished = 1.0 - _clamp(belief.progress, 0.0, 1.0)
                score = (
                    self.urgency_weight * urgency
                    + self.progress_weight * unfinished
                    + self.severity_weight * belief.severity
                    + self.confidence_weight * task_by_id[task_id].confidence
                )
                task_bias[task_id] = max(task_bias.get(task_id, -1.0), _clamp(score))

                required = belief.remaining_requirements
                denominator = max(1.0, float(sum(required)))
                overlap = sum(
                    min(float(ability), float(need))
                    for ability, need in zip(agent.abilities, required)
                ) / denominator
                distance = math.hypot(
                    agent.position.x - belief.position.x,
                    agent.position.y - belief.position.y,
                ) / math.sqrt(2.0)
                busy = (
                    self.busy_penalty
                    if agent.status not in {"idle", "available"}
                    and agent.current_task != task_id
                    else 0.0
                )
                pair_bias[(agent.agent_id, task_id)] = _clamp(
                    self.compatibility_weight * overlap
                    - self.distance_weight * distance
                    - busy
                )

        return CoordinationAdvice(
            revision=self._revision,
            generated_at=state.current_time,
            valid_until=state.current_time + self.validity,
            task_bias=task_bias,
            pair_bias=pair_bias,
            replan_agents=frozenset(replan_agents),
            metadata={"policy": "heuristic-residual-v2-belief-only"},
        )
