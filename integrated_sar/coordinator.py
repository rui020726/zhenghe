"""Coordination policies that optimize, but do not replace, task allocation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .allocation_trace import ROBOT_VELOCITY
from .contracts import (
    AgentSnapshot,
    AgentTaskBelief,
    CoalitionReservation,
    CoordinationAdvice,
    CoordinatorInput,
    WorldPosition,
)

# The statuses a robot has when it has no task and can take one. In task-03 these are
# exactly the statuses that carry ``need_decision = True``: a robot becomes "moving"
# with ``need_decision = False`` the moment it is dispatched, and "working" once it is
# on site, so "free" and "will be asked to decide" coincide.
IDLE_STATUSES = frozenset({"idle", "available"})


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


def _unit_slots(requirements: Sequence[float]) -> list[int]:
    """A task's requirement as whole unit counts, one per ability dimension."""
    return [max(int(value), 0) for value in requirements]


def _covered_units(
    requirements: Sequence[float], committed: Sequence[Sequence[float]]
) -> list[int]:
    """Whole units of each dimension the given robots already supply.

    Truncated to the requirement: a robot carrying two units into a one-unit slot
    supplies one, and the surplus must not drive the gap negative.
    """
    needs = _unit_slots(requirements)
    supplied = [0.0] * len(needs)
    for abilities in committed:
        for dimension in range(min(len(needs), len(abilities))):
            supplied[dimension] += float(abilities[dimension])
    return [min(int(value), need) for value, need in zip(supplied, needs)]


def _open_units(
    requirements: Sequence[float], committed: Sequence[Sequence[float]]
) -> list[int]:
    """The unit slots of each dimension that nobody has covered yet."""
    needs = _unit_slots(requirements)
    return [
        need - got for need, got in zip(needs, _covered_units(requirements, committed))
    ]


def _others(
    committed: Mapping[str, Sequence[float]], candidate_id: str
) -> list[Sequence[float]]:
    """Committed abilities with the candidate's own entry removed.

    A robot already inside the committed sum must not be counted as its own filler,
    or it would be credited twice for one capability.
    """
    return [
        abilities for agent_id, abilities in committed.items() if agent_id != candidate_id
    ]


def _gap_fill_ratio(
    requirements: Sequence[float],
    committed: Sequence[Sequence[float]],
    abilities: Sequence[float],
) -> float:
    """The share of a task's still-open units this robot would add, in [0, 1]."""
    open_units = _open_units(requirements, committed)
    total = float(sum(open_units))
    if total <= 0.0:
        return 0.0
    added = 0.0
    for dimension, available in enumerate(open_units):
        ability = float(abilities[dimension]) if dimension < len(abilities) else 0.0
        added += min(ability, float(available))
    return added / total


def _would_have_filled(
    requirements: Sequence[float], abilities: Sequence[float]
) -> bool:
    """True when this robot's abilities overlap the task's requirement at all."""
    return any(
        int(need) > 0
        and float(abilities[dimension] if dimension < len(abilities) else 0.0) > 0.0
        for dimension, need in enumerate(requirements)
    )


def _claim_slot(
    requirements: Sequence[float],
    committed: Sequence[Sequence[float]],
    abilities: Sequence[float],
) -> tuple[int, int] | None:
    """The next open unit slot this robot fills, or None when it fills nothing.

    Returns ``(ability_slot, slot_dimension)``, where ``ability_slot`` is the unit's
    index in the flattened requirement vector -- a requirement of ``(3, 0, ...)`` has
    slots 0, 1 and 2, all in dimension 0 -- and ``slot_dimension`` names the ability
    dimension it belongs to. The dimension with the largest contribution wins, ties
    going to the lowest index, so the answer is a function of the numbers alone.

    Claiming a slot is what makes a reservation exclusive: the caller adds the
    robot's abilities to the committed set, which moves ``_covered_units`` on and
    hands the next robot working that dimension the next unit index.
    """
    needs = _unit_slots(requirements)
    covered = _covered_units(requirements, committed)
    best_dimension, best_gain = -1, 0.0
    for dimension, need in enumerate(needs):
        available = need - covered[dimension]
        if available <= 0:
            continue
        ability = float(abilities[dimension]) if dimension < len(abilities) else 0.0
        gain = min(ability, float(available))
        if gain > best_gain:
            best_dimension, best_gain = dimension, gain
    if best_dimension < 0:
        return None
    return sum(needs[:best_dimension]) + covered[best_dimension], best_dimension


@dataclass(frozen=True, slots=True)
class _PlanRow:
    """One scored (free robot, believed task) candidate inside a planning round."""

    agent_id: str
    task_id: str
    order: int
    value: float
    fill: float
    revision: int
    abilities: tuple[float, ...]
    requirements: tuple[float, ...]


class CoalitionPlannerCoordinator:
    """Plan the group first, then let the residual commit each robot to its slot.

    A per-pair interface cannot convene a group: two robots that would fill the same
    capability slot are scored identically, so nothing in the allocator's menu
    distinguishes "the missing unit" from "one more unit of what is already coming".
    This policy answers the composition question on the team's behalf -- a greedy
    set-cover over the *open* unit slots of every task each robot believes in -- and
    publishes the answer as a :class:`CoalitionReservation` per selected robot, naming
    the slot it owns. Occupancy is exclusive by construction, so the robots a
    reservation sends are the robots the plan chose and no others.

    How a plan is built, once per coordination cycle:

    1. Every robot under orders is folded into the plan's committed abilities, so a
       task's open slots already reflect what the fleet has on the way.
    2. Free robots (``IDLE_STATUSES``, no current task) are candidates, paired with
       every task *their own* belief contains. A task past its believed deadline is
       nobody's candidate. A robot that is not free is left out of the pair channel
       entirely -- a stronger guarantee than a busy penalty, which a large enough
       positive term could still outweigh. Only the priority channel
       (``agent_task_bias``) reaches it, exactly as the heuristic's does.
    3. Rounds: score every remaining candidate, take the highest, claim its best open
       slot, add its abilities to that task's committed set, repeat. Re-scoring each
       round makes the cover greedy on the *live* gap -- a robot whose only
       contribution an earlier pick already supplied scores zero and is passed over,
       which is how redundant dispatch is prevented rather than penalised.
    4. Order is ``fill * fill_weight + urgency * urgency_weight + ... - distance *
       distance_weight``: the priority terms the heuristic publishes, with the
       gap-fill ratio standing in for plain compatibility overlap.

    The policy never reads ``CoordinatorInput.tasks``. Every number on a robot's row
    comes from that robot's own :class:`AgentTaskBelief`; the only other input is the
    team's own assignment plan (``AgentSnapshot.current_task``) -- which robot is
    under orders for which task, the same plan the allocator issued. One robot's
    deadline, requirement, position or progress therefore cannot change another
    robot's row, and a robot holding a stale belief is planned against its stale view
    until newer knowledge reaches it (invariant 9).

    The output is advice, not a decision: reservations ride the allocator's ordinary
    bounded residual, the allocator keeps the final choice, and the environment's hard
    mask is still applied last (invariants 5 and 6). The advice never outlives the
    shortest commitment in it, so a cached advice cannot apply a stale commitment.
    """

    def __init__(
        self,
        validity: float = 5.0,
        reservation_validity: float = 5.0,
        urgency_horizon: float = 100.0,
        urgency_weight: float = 0.45,
        progress_weight: float = 0.25,
        severity_weight: float = 0.20,
        confidence_weight: float = 0.10,
        fill_weight: float = 0.65,
        distance_weight: float = 0.35,
        commitment_weight: float = 0.35,
    ) -> None:
        if reservation_validity <= 0:
            raise ValueError("reservation_validity must be positive")
        if not 0.0 <= commitment_weight <= 1.0:
            raise ValueError("commitment_weight must be in [0, 1]")
        self.validity = float(validity)
        self.reservation_validity = float(reservation_validity)
        self.urgency_horizon = max(float(urgency_horizon), 1e-6)
        self.urgency_weight = float(urgency_weight)
        self.progress_weight = float(progress_weight)
        self.severity_weight = float(severity_weight)
        self.confidence_weight = float(confidence_weight)
        # Same default as the heuristic's compatibility weight, so the two policies
        # differ in what the dominant positive term measures, not in how much weight
        # it carries -- the comparison then isolates the coalition structure.
        self.fill_weight = float(fill_weight)
        self.distance_weight = float(distance_weight)
        # What a commitment adds on top of the pair bias. Kept well below the fill
        # weight so a reserved pair stays inside the [-1, 1] band the other policies
        # respect: the reservation breaks ties between plans, it is not a second vote.
        self.commitment_weight = float(commitment_weight)
        self._revision = 0
        self._reservations: dict[tuple[str, str], CoalitionReservation] = {}

    def reset(self) -> None:
        """Start a new episode: numbering restarts and no commitment survives it."""
        self._revision = 0
        self._reservations.clear()

    def advise(self, state: CoordinatorInput) -> CoordinationAdvice:
        self._revision += 1
        now = float(state.current_time)
        previous = dict(self._reservations)
        agents = {agent.agent_id: agent for agent in state.agents}

        agent_task_bias: dict[tuple[str, str], float] = {}
        replan_agents: set[str] = set()
        # The team's own plan: which robot is under orders for which task, and what
        # abilities that already puts on each task. Not a belief, and not a registry
        # fact -- see the class docstring.
        committed: dict[str, dict[str, tuple[float, ...]]] = defaultdict(dict)
        candidates: list[tuple[int, str, tuple[float, ...], tuple[AgentTaskBelief, ...]]] = []
        beliefs_by_key: dict[tuple[str, str], AgentTaskBelief] = {}

        for order, agent in enumerate(state.agents):
            abilities = tuple(float(value) for value in agent.abilities)
            if agent.current_task:
                committed[agent.current_task][agent.agent_id] = abilities
                if self._models_belief(agent):
                    # Only this robot's own belief can tell it its commitment ended;
                    # the registry closing a task is not a robot-visible fact.
                    belief = agent.belief_for(agent.current_task)
                    if belief is None or belief.finished:
                        replan_agents.add(agent.agent_id)

            known = agent.known_tasks
            usable: list[AgentTaskBelief] = []
            for belief in agent.task_beliefs:
                task_id = belief.task_id
                if belief.finished:
                    # This robot itself has learned that the task is over.
                    continue
                if known is not None and task_id not in known:
                    continue
                beliefs_by_key[(agent.agent_id, task_id)] = belief
                if belief.deadline < now:
                    # This robot's own belief says the task is over. Publishing a
                    # priority for it would be the strongest possible pull -- urgency
                    # saturates at 1 for a task already past its deadline -- so it
                    # leaves both channels rather than being biased downwards.
                    continue
                agent_task_bias[(agent.agent_id, task_id)] = self._priority(belief, now)
                usable.append(belief)

            if usable and self._is_free(agent):
                candidates.append((order, agent.agent_id, abilities, tuple(usable)))

        reservations, planned = self._plan(agents, candidates, committed, now)
        pair_bias, redundant = self._pair_biases(
            agents, candidates, committed, planned
        )
        counts = self._lifecycle(previous, reservations, beliefs_by_key, now)
        self._reservations = dict(reservations)

        window = now + self.validity
        if reservations:
            # A cached advice is the only way an expired commitment could still be
            # applied, so the advice may not outlive the shortest commitment in it.
            window = min(window, min(held.valid_until for held in reservations.values()))
        return CoordinationAdvice(
            revision=self._revision,
            generated_at=now,
            valid_until=window,
            agent_task_bias=agent_task_bias,
            pair_bias=pair_bias,
            replan_agents=frozenset(replan_agents),
            reservations=dict(reservations),
            metadata={
                "policy": "coalition-planner-v1-belief-only",
                "planned_reservations": len(reservations),
                "redundant_candidates": redundant,
                "held_slot_indices": sorted(
                    held.ability_slot for held in reservations.values()
                ),
                **counts,
            },
        )

    def _plan(
        self,
        agents: Mapping[str, AgentSnapshot],
        candidates: Sequence[
            tuple[int, str, tuple[float, ...], tuple[AgentTaskBelief, ...]]
        ],
        committed: Mapping[str, Mapping[str, tuple[float, ...]]],
        now: float,
    ) -> tuple[
        dict[tuple[str, str], CoalitionReservation], dict[str, dict[str, tuple[float, ...]]]
    ]:
        """Greedy set-cover over the live gap: score, take the best, claim its slot.

        The scale is deliberately small -- one round per selected robot, each round
        walking the free robots times the tasks they believe in -- which is why the
        whole plan is recomputed rather than patched.
        """
        covered: dict[str, dict[str, tuple[float, ...]]] = defaultdict(dict)
        for task_id, holders in committed.items():
            covered[task_id] = dict(holders)

        picked: set[str] = set()
        chosen: dict[tuple[str, str], CoalitionReservation] = {}
        planned: dict[str, dict[str, tuple[float, ...]]] = defaultdict(dict)
        while True:
            rows: list[_PlanRow] = []
            for order, agent_id, abilities, beliefs in candidates:
                if agent_id in picked:
                    continue
                for belief in beliefs:
                    task_id = belief.task_id
                    committed_here = _others(covered.get(task_id, {}), agent_id)
                    fill = _gap_fill_ratio(
                        belief.remaining_requirements, committed_here, abilities
                    )
                    if fill <= 0.0:
                        continue
                    rows.append(
                        _PlanRow(
                            agent_id=agent_id,
                            task_id=task_id,
                            order=order,
                            value=self._row_value(
                                agents[agent_id], belief, fill, now
                            ),
                            fill=fill,
                            revision=int(belief.revision),
                            abilities=abilities,
                            requirements=tuple(
                                float(need) for need in belief.remaining_requirements
                            ),
                        )
                    )
            if not rows:
                break
            # Total order: value first, then a deterministic tie-break that does not
            # depend on how a set or a dict happened to be laid out.
            rows.sort(key=lambda row: (-row.value, row.order, row.task_id))
            row = rows[0]
            picked.add(row.agent_id)
            claim = _claim_slot(
                row.requirements,
                _others(covered.get(row.task_id, {}), row.agent_id),
                row.abilities,
            )
            if claim is None:
                # Unreachable for a row with a positive fill; skipping the robot keeps
                # the loop moving instead of dropping the round.
                continue
            slot_index, dimension = claim
            covered[row.task_id][row.agent_id] = row.abilities
            planned[row.task_id][row.agent_id] = row.abilities
            chosen[(row.agent_id, row.task_id)] = CoalitionReservation(
                agent_id=row.agent_id,
                task_id=row.task_id,
                ability_slot=slot_index,
                slot_dimension=dimension,
                task_revision=row.revision,
                issued_at=now,
                valid_until=now + self.reservation_validity,
                score=_clamp(self.commitment_weight * row.fill),
            )
        return chosen, planned

    def _pair_biases(
        self,
        agents: Mapping[str, AgentSnapshot],
        candidates: Sequence[
            tuple[int, str, tuple[float, ...], tuple[AgentTaskBelief, ...]]
        ],
        committed: Mapping[str, Mapping[str, tuple[float, ...]]],
        planned: Mapping[str, Mapping[str, tuple[float, ...]]],
    ) -> tuple[dict[tuple[str, str], float], int]:
        """Score every free (robot, believed task) pair against the *planned* cover.

        This is what steers the robots the plan did not pick: a task whose slots the
        plan already filled leaves them a fill ratio of zero, so their pair bias is
        distance only and they look elsewhere. A robot the plan did pick keeps a
        positive ratio, because its own commitment is never counted as already
        supplied to it.
        """
        covered: dict[str, dict[str, tuple[float, ...]]] = defaultdict(dict)
        for task_id, holders in committed.items():
            covered[task_id] = dict(holders)
        for task_id, holders in planned.items():
            covered[task_id].update(holders)

        pair_bias: dict[tuple[str, str], float] = {}
        redundant = 0
        for _, agent_id, abilities, beliefs in candidates:
            agent = agents[agent_id]
            for belief in beliefs:
                task_id = belief.task_id
                fill = _gap_fill_ratio(
                    belief.remaining_requirements,
                    _others(covered.get(task_id, {}), agent_id),
                    abilities,
                )
                if fill <= 0.0 and _would_have_filled(
                    belief.remaining_requirements, abilities
                ):
                    # This robot could have helped, but the plan already has the unit
                    # on the way: the duplicate the planner exists to prevent.
                    redundant += 1
                pair_bias[(agent_id, task_id)] = _clamp(
                    self.fill_weight * fill
                    - self.distance_weight * self._distance(agent, belief)
                )
        return pair_bias, redundant

    def _row_value(
        self,
        agent: AgentSnapshot,
        belief: AgentTaskBelief,
        fill: float,
        now: float,
    ) -> float:
        """Selection order: the heuristic's priority terms plus the gap-fill ratio."""
        urgency, unfinished, severity, confidence = self._priority_terms(belief, now)
        return (
            self.fill_weight * fill
            + self.urgency_weight * urgency
            + self.progress_weight * unfinished
            + self.severity_weight * severity
            + self.confidence_weight * confidence
            - self.distance_weight * self._distance(agent, belief)
        )

    def _priority(self, belief: AgentTaskBelief, now: float) -> float:
        urgency, unfinished, severity, confidence = self._priority_terms(belief, now)
        return _clamp(
            self.urgency_weight * urgency
            + self.progress_weight * unfinished
            + self.severity_weight * severity
            + self.confidence_weight * confidence
        )

    def _priority_terms(
        self, belief: AgentTaskBelief, now: float
    ) -> tuple[float, float, float, float]:
        """Urgency, unfinished share, severity and confidence -- all from the belief."""
        time_left = belief.deadline - now
        urgency = 1.0 - _clamp(time_left / self.urgency_horizon, 0.0, 1.0)
        unfinished = 1.0 - _clamp(belief.progress, 0.0, 1.0)
        return (urgency, unfinished, float(belief.severity), float(belief.confidence))

    @staticmethod
    def _distance(agent: AgentSnapshot, belief: AgentTaskBelief) -> float:
        return math.hypot(
            agent.position.x - belief.position.x,
            agent.position.y - belief.position.y,
        ) / math.sqrt(2.0)

    def _lifecycle(
        self,
        previous: Mapping[tuple[str, str], CoalitionReservation],
        planned: Mapping[tuple[str, str], CoalitionReservation],
        beliefs_by_key: Mapping[tuple[str, str], AgentTaskBelief],
        now: float,
    ) -> dict[str, int]:
        """Why the commitments held before this cycle are not held any more.

        Three endings, kept apart because they mean different things to the
        experiment: one the clock ran out on (``expired``), one this robot's own
        belief moved on from -- a revision bump, or knowledge that left its view
        (``superseded``) -- and one still nominally valid that the new plan declined
        to re-issue (``cancelled``).
        """
        expired = superseded = cancelled = 0
        for key, held in previous.items():
            if not held.active_at(now):
                expired += 1
                continue
            if key in planned:
                continue
            belief = beliefs_by_key.get(key)
            if belief is None or int(belief.revision) != held.task_revision:
                superseded += 1
            else:
                cancelled += 1
        return {
            "expired_reservations": expired,
            "superseded_reservations": superseded,
            "cancelled_reservations": cancelled,
        }

    @staticmethod
    def _is_free(agent: AgentSnapshot) -> bool:
        return agent.status in IDLE_STATUSES and not agent.current_task

    @staticmethod
    def _models_belief(agent: AgentSnapshot) -> bool:
        """True when the agent source reports task knowledge for this robot at all."""
        return agent.known_tasks is not None or bool(agent.task_beliefs)


def _travel_steps(
    position: WorldPosition, believed: WorldPosition, velocity: float
) -> float:
    """Steps a robot needs to reach a believed task position at the allocator's speed.

    The same speed ``old_task_env`` moves robots by, so this is the ETA the allocator is
    effectively choosing between, not a new model of travel.
    """
    return math.hypot(position.x - believed.x, position.y - believed.y) / max(
        float(velocity), 1e-9
    )


class DeadlineAwareCoalitionCoordinator(CoalitionPlannerCoordinator):
    """Same candidates, same slots, a different question: can this group arrive in time?

    A diagnostic policy, built to answer one question before any network is trained: is
    the rule planner's *ordering* the binding constraint, or is the information it sees
    not enough to order better? It generates exactly the candidates the rule planner
    does -- the same free robots, the same believed tasks, the same unit slots, the same
    exclusivity -- and changes only how it chooses among them. A difference between the
    two is therefore a difference in ordering and nothing else, which is what makes it a
    usable oracle: whatever it fixes, a reordering of legal candidates fixed.

    Three changes, all computable from what an online coordinator already sees:

    1. **Cover, not fill.** The rule planner takes the highest-scoring (robot, task)
       pair each round and assembles a group member by member without ever asking
       whether the group can work. This policy enumerates the coalitions that cover the
       believed requirement outright and keeps the one whose *last needed member*
       arrives soonest -- the moment the group can actually start.
    2. **Believed budget.** A task whose fastest covering group still arrives after that
       robot's own believed deadline is skipped and its robots stay free for tasks that
       can still be saved. The post-mortem says the rule planner never does this: it
       spends robots on work that cannot be finished and calls the spend a plan.
    3. **One-step lookahead.** A robot that is the only remaining way an
       earlier-deadline task can be covered is not spent on a later one while an
       alternative coalition exists. One step, over the team's own plan, not over the
       simulator.

    Two things it deliberately cannot do, because the belief does not carry them. It
    cannot subtract the stage's work duration from the budget -- ``AgentTaskBelief``
    keeps the requirement and the deadline but drops the duration the
    ``DetectionMessage`` carried -- so its budget check is optimistic by the work time.
    And it cannot see a detection that has not reached a robot yet. Both limits are the
    point: if this policy still leaves the failure, the failure is not an ordering
    failure, and no network trained on these features can fix it.

    Invariant 9 holds exactly as in the parent: every number comes from that robot's own
    belief or from the team's own assignment plan, ``CoordinatorInput.tasks`` is never
    read, and the advice never outlives the shortest commitment in it.
    """

    def __init__(self, robot_velocity: float = ROBOT_VELOCITY, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.robot_velocity = float(robot_velocity)
        self.last_plan_notes: dict[str, int] = {}

    def _plan(
        self,
        agents: Mapping[str, AgentSnapshot],
        candidates: Sequence[
            tuple[int, str, tuple[float, ...], tuple[AgentTaskBelief, ...]]
        ],
        committed: Mapping[str, Mapping[str, tuple[float, ...]]],
        now: float,
    ) -> tuple[
        dict[tuple[str, str], CoalitionReservation],
        dict[str, dict[str, tuple[float, ...]]],
    ]:
        """The rule plan, minus the tasks it cannot finish inside the believed budget.

        The first attempt at this oracle built coalitions from scratch and demanded a
        complete cover from the robots free *right now*, which is strictly more brittle
        than the rule planner's incremental filling: a task that needed three robots and
        had two free got no reservation at all instead of two, and the fleet waited. On
        the legacy analysis block that made expiry *worse* (16 to 30), so the design was
        corrected to the one below, which cannot be more brittle than the rule planner
        because it starts from it.

        What remains of the oracle is the part that is a pure reordering of the rule
        planner's own output: a task whose reserved group still cannot arrive inside the
        budget its own members believe in has its commitments returned, and the robots
        are re-planned onto the tasks that can still be saved. Nothing is added, nothing
        is invented, and no robot is held back from work the rule planner would have
        sent it to.
        """
        chosen, planned = super()._plan(agents, candidates, committed, now)
        beliefs = {
            (agent_id, belief.task_id): belief
            for _, agent_id, _, agent_beliefs in candidates
            for belief in agent_beliefs
        }
        hopeless = self._hopeless_tasks(chosen, beliefs, agents, now)
        if not hopeless:
            self.last_plan_notes = {"tasks_returned_on_believed_budget": 0, "robots_replanned": 0}
            return chosen, planned

        freed = {agent_id for agent_id, task_id in chosen if task_id in hopeless}
        kept = {key: value for key, value in chosen.items() if key[1] not in hopeless}
        kept_planned = {
            task_id: dict(members)
            for task_id, members in planned.items()
            if task_id not in hopeless
        }
        # The robots the returned commitments release are planned again, over the tasks
        # that are still worth serving, against a cover that already includes what the
        # plan kept.
        busy = {agent_id for agent_id, _ in kept}
        carriers = [
            (
                order,
                agent_id,
                abilities,
                tuple(b for b in beliefs if b.task_id not in hopeless),
            )
            for order, agent_id, abilities, beliefs in candidates
            if agent_id in freed and agent_id not in busy
        ]
        carriers = [
            (order, agent_id, abilities, beliefs) for order, agent_id, abilities, beliefs in carriers if beliefs
        ]
        covering: dict[str, dict[str, tuple[float, ...]]] = defaultdict(dict)
        for task_id, holders in committed.items():
            covering[task_id] = dict(holders)
        for task_id, members in kept_planned.items():
            covering[task_id].update(members)
        extra, extra_planned = super()._plan(agents, carriers, covering, now)

        merged = dict(kept)
        for key, value in extra.items():
            merged[key] = value
        merged_planned: dict[str, dict[str, tuple[float, ...]]] = defaultdict(dict)
        for task_id, members in kept_planned.items():
            merged_planned[task_id].update(members)
        for task_id, members in extra_planned.items():
            merged_planned[task_id].update(members)
        self.last_plan_notes = {
            "tasks_returned_on_believed_budget": len(hopeless),
            "robots_replanned": len(carriers),
        }
        return merged, merged_planned

    def _hopeless_tasks(
        self,
        chosen: Mapping[tuple[str, str], CoalitionReservation],
        beliefs: Mapping[tuple[str, str], AgentTaskBelief],
        agents: Mapping[str, AgentSnapshot],
        now: float,
    ) -> frozenset[str]:
        """Tasks whose reserved group cannot arrive inside what its members believe.

        The budget check is deliberately optimistic: the belief carries no stage work
        duration, so only travel is subtracted. A task returned here was hopeless even
        before work time is taken into account, which is what makes returning it safe.
        """
        by_task: dict[str, list[str]] = defaultdict(list)
        for agent_id, task_id in chosen:
            by_task[task_id].append(agent_id)
        hopeless: set[str] = set()
        for task_id, members in by_task.items():
            arrivals = []
            deadlines = []
            for agent_id in members:
                belief = beliefs.get((agent_id, task_id))
                if belief is None:
                    continue
                arrivals.append(
                    _travel_steps(
                        agents[agent_id].position, belief.position, self.robot_velocity
                    )
                )
                deadlines.append(belief.deadline)
            if not arrivals:
                continue
            if max(arrivals) > min(deadlines) - now:
                hopeless.add(task_id)
        return frozenset(hopeless)
