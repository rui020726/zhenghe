"""Coordination policies that optimize, but do not replace, task allocation."""

from __future__ import annotations

import math
from collections import defaultdict

from .contracts import AgentSnapshot, CoordinationAdvice, CoordinatorInput


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

        A task's priority is emitted as the per-agent ``agent_task_bias``, not as
        the pool-level ``task_bias``: the pool-level map is handed to every robot,
        so a belief-derived number placed there would broadcast one robot's
        knowledge to robots the message never reached.
        """
        self._revision += 1
        agent_task_bias: dict[tuple[str, str], float] = {}
        pair_bias: dict[tuple[str, str], float] = {}
        replan_agents: set[str] = set()

        for agent in state.agents:
            if agent.current_task and self._models_belief(agent):
                # Replan only when this robot's own knowledge says its commitment
                # ended, or when it holds no knowledge of the task it is committed
                # to. The registry closing a task is not a robot-visible fact, so
                # it must not pull a robot off a target it still believes in.
                belief = agent.belief_for(agent.current_task)
                if belief is None or belief.finished:
                    replan_agents.add(agent.agent_id)

            known = agent.known_tasks
            for belief in agent.task_beliefs:
                task_id = belief.task_id
                if belief.finished:
                    # This robot itself has learned that the task is over.
                    continue
                if known is not None and task_id not in known:
                    continue

                time_left = belief.deadline - state.current_time
                urgency = 1.0 - _clamp(time_left / self.urgency_horizon, 0.0, 1.0)
                unfinished = 1.0 - _clamp(belief.progress, 0.0, 1.0)
                agent_task_bias[(agent.agent_id, task_id)] = _clamp(
                    self.urgency_weight * urgency
                    + self.progress_weight * unfinished
                    + self.severity_weight * belief.severity
                    + self.confidence_weight * belief.confidence
                )

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
            agent_task_bias=agent_task_bias,
            pair_bias=pair_bias,
            replan_agents=frozenset(replan_agents),
            metadata={"policy": "heuristic-residual-v3-belief-only"},
        )

    @staticmethod
    def _models_belief(agent: AgentSnapshot) -> bool:
        """True when the agent source reports task knowledge for this robot at all."""
        return agent.known_tasks is not None or bool(agent.task_beliefs)


def _covered_by_others(
    requirements: tuple[float, ...],
    committed: dict[str, tuple[float, ...]],
    candidate_id: str,
) -> tuple[float, ...]:
    """What the other robots committed to this task already supply, per ability.

    The candidate is skipped: a robot that is already under orders is *inside* the
    committed sum, so counting it as a filler too would credit the same ability
    twice and invite it to work it is already doing.
    """
    covered = [0.0] * len(requirements)
    for agent_id, abilities in committed.items():
        if agent_id == candidate_id:
            continue
        for index, ability in enumerate(abilities[: len(covered)]):
            covered[index] += ability
    return tuple(
        max(need - got, 0.0) for need, got in zip(requirements, covered)
    )


class CoalitionAwareCoordinator:
    """Convene the robots that fill a task's *missing* capability, not the ones repeating it.

    The uncalibrated post-mortem left one failure mode behind: a stage requirement
    that needs a mix of abilities never assembles on site, and the fleet waits a
    median of 67 steps for a group that covers it. ``HeuristicCoordinator`` scores a
    robot by how much of the *whole* remaining requirement its abilities can satisfy,
    which keeps rewarding ability the team has already committed -- with two of the
    three needed units already en route, the third robot is ranked by the same margin
    as when it was the first, and nothing in the score is about the gap.

    This policy scores the gap instead. For every (robot, task) pair it computes

    * ``uncovered`` -- the belief's ``remaining_requirements`` minus the abilities of
      the robots already committed to that task, the candidate excluded so its own
      commitment is never counted twice;
    * ``candidate_fill_ratio`` -- how much of that uncovered part this candidate's own
      abilities would add, as a fraction of the gap.

    The coalition term *is* that ratio, so a robot that fills nothing scores exactly
    zero: distance and the busy penalty can then only push it away, and the positive
    part of the residual can never be earned by repeating a capability the task
    already has. A task with no gap left invites nobody.

    Invariant 9: the pair's task view is the candidate's own belief, read exactly as
    ``HeuristicCoordinator`` reads it -- never the registry. The committed set is not
    a belief about the world either: it is the team's own assignment plan, which
    robot is under orders for which task, and the central allocator that issued those
    commands is the module that reads it. No other robot's requirements, deadline,
    progress, severity or position is ever used to score a candidate's row, so one
    robot's knowledge cannot leak into another's bias. A robot holding a stale belief
    gets a stale ratio -- its own belief still shows the gap open, so it keeps being
    convened until newer knowledge reaches it.
    """

    def __init__(
        self,
        validity: float = 5.0,
        urgency_horizon: float = 100.0,
        urgency_weight: float = 0.45,
        progress_weight: float = 0.25,
        severity_weight: float = 0.20,
        confidence_weight: float = 0.10,
        fill_weight: float = 0.65,
        distance_weight: float = 0.35,
        busy_penalty: float = 0.40,
    ) -> None:
        self.validity = float(validity)
        self.urgency_horizon = max(float(urgency_horizon), 1e-6)
        self.urgency_weight = float(urgency_weight)
        self.progress_weight = float(progress_weight)
        self.severity_weight = float(severity_weight)
        self.confidence_weight = float(confidence_weight)
        # Same default as the heuristic's compatibility weight: the two policies are
        # meant to differ in *what* the dominant positive term measures, not in how
        # much weight it carries, so a comparison isolates the coalition term.
        self.fill_weight = float(fill_weight)
        self.distance_weight = float(distance_weight)
        self.busy_penalty = float(busy_penalty)
        self._revision = 0

    def reset(self) -> None:
        """Start a new episode: advice numbering restarts, so revisions stay comparable."""
        self._revision = 0

    def advise(self, state: CoordinatorInput) -> CoordinationAdvice:
        self._revision += 1
        agent_task_bias: dict[tuple[str, str], float] = {}
        pair_bias: dict[tuple[str, str], float] = {}
        replan_agents: set[str] = set()

        # The team's own plan: which robot is under orders for which task. Not a
        # belief, and not a registry fact -- see the class docstring.
        committed: dict[str, dict[str, tuple[float, ...]]] = defaultdict(dict)
        for agent in state.agents:
            if agent.current_task:
                committed[agent.current_task][agent.agent_id] = tuple(
                    float(ability) for ability in agent.abilities
                )

        filled_pairs = 0
        for agent in state.agents:
            if agent.current_task and self._models_belief(agent):
                # Same commitment rule as the heuristic: only this robot's own belief
                # can tell it that its commitment ended.
                belief = agent.belief_for(agent.current_task)
                if belief is None or belief.finished:
                    replan_agents.add(agent.agent_id)

            known = agent.known_tasks
            abilities = tuple(float(ability) for ability in agent.abilities)
            for belief in agent.task_beliefs:
                task_id = belief.task_id
                if belief.finished:
                    continue
                if known is not None and task_id not in known:
                    continue

                time_left = belief.deadline - state.current_time
                urgency = 1.0 - _clamp(time_left / self.urgency_horizon, 0.0, 1.0)
                unfinished = 1.0 - _clamp(belief.progress, 0.0, 1.0)
                agent_task_bias[(agent.agent_id, task_id)] = _clamp(
                    self.urgency_weight * urgency
                    + self.progress_weight * unfinished
                    + self.severity_weight * belief.severity
                    + self.confidence_weight * belief.confidence
                )

                uncovered = _covered_by_others(
                    tuple(float(need) for need in belief.remaining_requirements),
                    committed.get(task_id, {}),
                    agent.agent_id,
                )
                gap = sum(uncovered)
                fill = (
                    sum(
                        min(ability, need)
                        for ability, need in zip(abilities, uncovered)
                    )
                    / gap
                    if gap > 0.0
                    else 0.0
                )
                if fill > 0.0:
                    filled_pairs += 1

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
                    self.fill_weight * fill
                    - self.distance_weight * distance
                    - busy
                )

        return CoordinationAdvice(
            revision=self._revision,
            generated_at=state.current_time,
            valid_until=state.current_time + self.validity,
            agent_task_bias=agent_task_bias,
            pair_bias=pair_bias,
            replan_agents=frozenset(replan_agents),
            metadata={
                "policy": "coalition-aware-v1-belief-only",
                "filled_pairs": filled_pairs,
            },
        )

    @staticmethod
    def _models_belief(agent: AgentSnapshot) -> bool:
        """True when the agent source reports task knowledge for this robot at all."""
        return agent.known_tasks is not None or bool(agent.task_beliefs)
