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

    def advise(self, state: CoordinatorInput) -> CoordinationAdvice:
        self._revision += 1
        active_tasks = [task for task in state.tasks if task.active]
        task_by_id = {task.task_id: task for task in active_tasks}
        task_bias: dict[str, float] = {}
        pair_bias: dict[tuple[str, str], float] = {}
        replan_agents: set[str] = set()

        for task in active_tasks:
            time_left = task.deadline - state.current_time
            urgency = 1.0 - _clamp(time_left / self.urgency_horizon, 0.0, 1.0)
            unfinished = 1.0 - task.progress
            score = (
                self.urgency_weight * urgency
                + self.progress_weight * unfinished
                + self.severity_weight * task.severity
                + self.confidence_weight * task.confidence
            )
            if task.expired:
                score -= 0.25
            task_bias[task.task_id] = _clamp(score)

        for agent in state.agents:
            if agent.current_task and agent.current_task not in task_by_id:
                replan_agents.add(agent.agent_id)

            # Invariant 9: only bias pairs this robot can actually have heard about.
            # A robot that never received the detection cannot act on it, and scoring
            # it anyway would leak registry truth into a decentralized decision.
            known = agent.known_tasks
            for task in active_tasks:
                if known is not None and task.task_id not in known:
                    continue
                required = task.remaining_requirements
                denominator = max(1.0, float(sum(required)))
                overlap = sum(
                    min(float(ability), float(need))
                    for ability, need in zip(agent.abilities, required)
                ) / denominator
                distance = math.hypot(
                    agent.position.x - task.position.x,
                    agent.position.y - task.position.y,
                ) / math.sqrt(2.0)
                busy = (
                    self.busy_penalty
                    if agent.status not in {"idle", "available"}
                    and agent.current_task != task.task_id
                    else 0.0
                )
                pair_bias[(agent.agent_id, task.task_id)] = _clamp(
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
            metadata={"policy": "heuristic-residual-v1"},
        )
