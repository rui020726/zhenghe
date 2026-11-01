"""Deterministic assembly planning for a coalition: who, in what order, arriving together.

Phase 10B of the coalition-formation analysis. The measured finding behind it: on the six
remaining binding failures every robot already held the detection letter at detection time
(``belief_spread = 0``) and no capable robot was busy, yet the covering coalition completed
0-157 steps later against budgets of 40-77 steps. Two of the six additionally waited 66 and
85 steps for the *next* stage after the previous one completed. The planner chose the right
robots but never asked when they would arrive.

This module answers that question and nothing else. It is a pure function over what each
robot was told:

* candidates are only robots whose **own** belief contains the task (invariant 9) -- a robot
  that has not been told is never planned for, and its position is never assumed;
* travel is the straight-line ETA at ``ROBOT_VELOCITY`` from the robot's **believed**
  position to the task's **believed** position, the same primitive
  :func:`integrated_sar.coordinator._travel_steps` already uses;
* the covering set is minimal by construction (greedy over the live gap, like the existing
  planner's set-cover), and the *synchronisation* term is what is new: among the sets that
  cover the requirement, prefer the one whose members land closest together, because a
  stage clock cannot start until the last of them arrives.

Nothing here is stateful and nothing is random: the same input produces the same plan.

The plan is advisory, exactly like a coalition reservation. It is returned as data; the
allocator still makes the final choice and the environment's hard mask is still applied
last (invariants 5 and 6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .contracts import AgentSnapshot, AgentTaskBelief
from .coordinator import ROBOT_VELOCITY

# A member whose ETA is this far from the earliest member's is "late": the set will not be
# complete until it lands, and the stage clock cannot start before that. Chosen from the
# measured spreads (controls assemble in 1-71 steps; the failures in 42-157) so the term
# separates ragged arrivals from tight ones without dominating the covering decision.
SYNC_TOLERANCE = 5.0


@dataclass(frozen=True, slots=True)
class AssemblyMember:
    """One robot's role in an assembly plan, with the numbers that put it there."""

    agent_id: str
    eta_steps: float
    contribution: tuple[float, ...]
    claimed_slot: int | None
    claimed_dimension: int | None


@dataclass(frozen=True, slots=True)
class AssemblyPlan:
    """A deterministic, minimal, arrival-synchronised coalition for one task."""

    task_id: str
    members: tuple[AssemblyMember, ...]
    covered: tuple[float, ...]
    missing: tuple[float, ...]
    earliest_arrival: float | None
    latest_arrival: float | None
    arrival_spread: float | None
    feasible: bool
    score: float

    @property
    def complete(self) -> bool:
        """The plan covers the whole believed requirement."""
        return bool(self.members) and not any(value > 0 for value in self.missing)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "members": [
                {
                    "agent_id": member.agent_id,
                    "eta_steps": round(member.eta_steps, 3),
                    "claimed_slot": member.claimed_slot,
                    "claimed_dimension": member.claimed_dimension,
                }
                for member in self.members
            ],
            "covered": [int(value) for value in self.covered],
            "missing": [int(value) for value in self.missing],
            "earliest_arrival": None if self.earliest_arrival is None else round(self.earliest_arrival, 3),
            "latest_arrival": None if self.latest_arrival is None else round(self.latest_arrival, 3),
            "arrival_spread": None if self.arrival_spread is None else round(self.arrival_spread, 3),
            "feasible": self.feasible,
            "complete": self.complete,
            "score": round(self.score, 4),
        }


def _travel_steps(origin: Any, target: Any) -> float:
    """Straight-line ETA in steps from one believed position to another.

    The same model the offline attribution uses: robots move at ``ROBOT_VELOCITY``
    normalized units per step in the unit square, so this is a lower bound on travel and
    the only travel number an online module may use (it needs no map, no path and no
    future position).
    """
    dx = float(origin.x) - float(target.x)
    dy = float(origin.y) - float(target.y)
    return math.hypot(dx, dy) / ROBOT_VELOCITY


def _slack(belief: AgentTaskBelief, now: float) -> float:
    """Steps until this robot's own believed deadline, never negative."""
    return max(0.0, float(belief.deadline) - now)


def _claim(requirement: Sequence[float], covered: Sequence[float], abilities: Sequence[float]):
    """The best open unit slot this robot can fill, or None.

    Mirrors ``coordinator._claim_slot``: scan dimensions by descending remaining need and
    return the first the robot actually has. Returning None is what keeps a redundant
    robot out of the plan rather than penalising it.
    """
    best: tuple[float, int] | None = None
    for dimension, need in enumerate(requirement):
        if need <= 0:
            continue
        if dimension >= len(abilities) or abilities[dimension] <= 0:
            continue
        already = float(covered[dimension]) if dimension < len(covered) else 0.0
        gap = float(need) - already
        if gap <= 0:
            continue
        if best is None or gap > best[0]:
            best = (gap, dimension)
    if best is None:
        return None
    return best[1]


def plan_assembly(
    belief: AgentTaskBelief,
    agents: Mapping[str, AgentSnapshot],
    *,
    now: float,
    tolerance: float = SYNC_TOLERANCE,
    max_members: int | None = None,
) -> AssemblyPlan:
    """Build the minimal covering coalition for ``belief`` that arrives most together.

    ``agents`` is keyed by id and every entry is used only through *its own* belief for this
    task: a robot whose beliefs do not contain the task is not a candidate, and a robot the
    caller has not been told about cannot be planned for. Deterministic: ties break on
    ``agent_id``, so the same input always yields the same plan.
    """
    requirement = tuple(float(value) for value in belief.remaining_requirements)
    covered = [0.0] * len(requirement)

    # Candidates: robots that are free and whose own belief contains this task. A robot
    # already under orders is not a candidate here -- the caller folds committed abilities
    # into ``covered`` before calling, exactly as the coalition planner does.
    pool: list[tuple[str, float, tuple[float, ...]]] = []
    for agent_id, agent in sorted(agents.items()):
        if agent.current_task:
            continue
        own = next((b for b in agent.task_beliefs if b.task_id == belief.task_id), None)
        if own is None or own.finished:
            continue
        if own.deadline < now:
            # This robot's own belief says the task is over; planning for it would be the
            # strongest possible pull towards a task it believes is closed.
            continue
        abilities = tuple(float(value) for value in agent.abilities)
        if not any(value > 0 for value in abilities):
            continue
        eta = _travel_steps(agent.position, own.position)
        pool.append((agent_id, eta, abilities))

    members: list[AssemblyMember] = []
    remaining = list(requirement)
    while any(value > 0 for value in remaining):
        if max_members is not None and len(members) >= max_members:
            break
        best: tuple[float, str, float, tuple[float, ...], int] | None = None
        for agent_id, eta, abilities in pool:
            if any(member.agent_id == agent_id for member in members):
                continue
            dimension = _claim(remaining, covered, abilities)
            if dimension is None:
                continue
            # Score: fill as much of the live gap as possible first, then prefer the robot
            # that lands soonest, then the robot already closest to the current set's
            # arrival time. The last term is the synchronisation: two robots that can each
            # fill the slot are not equivalent if one lands 40 steps after the other.
            gain = min(remaining[dimension], abilities[dimension])
            current_latest = max((member.eta_steps for member in members), default=eta)
            raggedness = abs(eta - current_latest)
            score = gain * 100.0 - eta - (raggedness if raggedness > tolerance else 0.0)
            # Ties break on agent_id ascending, so the plan is reproducible: comparing
            # (score, id) with "greater" would silently prefer the lexicographically
            # larger id, which is arbitrary but stable -- this makes the intent explicit.
            better = best is None or score > best[0] or (score == best[0] and agent_id < best[1])
            if better:
                best = (score, agent_id, eta, abilities, dimension)
        if best is None:
            break
        _, agent_id, eta, abilities, dimension = best
        gain = min(remaining[dimension], abilities[dimension])
        remaining[dimension] = max(0.0, remaining[dimension] - gain)
        covered[dimension] += gain
        members.append(
            AssemblyMember(
                agent_id=agent_id,
                eta_steps=eta,
                contribution=abilities,
                claimed_slot=len(members),
                claimed_dimension=dimension,
            )
        )

    etas = [member.eta_steps for member in members]
    earliest = min(etas) if etas else None
    latest = max(etas) if etas else None
    spread = None if earliest is None or latest is None else latest - earliest
    missing = tuple(int(value) for value in remaining)

    # A plan is feasible when it covers the requirement and the last member still lands
    # inside the robot's own believed budget. No stage duration is used anywhere: the
    # budget is the believed deadline, which is what the robot was told.
    feasible = bool(members) and not any(value > 0 for value in missing) and (
        latest is not None and latest <= _slack(belief, now)
    )
    return AssemblyPlan(
        task_id=belief.task_id,
        members=tuple(members),
        covered=tuple(int(value) for value in covered),
        missing=missing,
        earliest_arrival=earliest,
        latest_arrival=latest,
        arrival_spread=spread,
        feasible=feasible,
        score=float(len(members)),
    )


def order_tasks(
    plans: Sequence[AssemblyPlan],
    beliefs: Mapping[str, AgentTaskBelief],
    *,
    now: float,
    active_load: Mapping[str, int] | None = None,
) -> list[AssemblyPlan]:
    """Rank assembly plans: feasible first, then deadline slack, then the lighter task.

    Explicitly **not** ranked by stage duration. The duration-relevance gate already showed
    that ordering by work time rescues none of the six; what matters here is whether the
    coalition can be assembled inside the believed budget and how much slack is left. Ties
    break on ``task_id`` so the order is deterministic.
    """
    load = dict(active_load or {})

    def key(plan: AssemblyPlan):
        belief = beliefs.get(plan.task_id)
        slack = _slack(belief, now) if belief is not None else 0.0
        # The last member's arrival is what the deadline has to absorb.
        cost = plan.latest_arrival if plan.latest_arrival is not None else float("inf")
        return (
            0 if plan.feasible else 1,
            -(slack - cost),
            load.get(plan.task_id, 0),
            plan.task_id,
        )

    return sorted(plans, key=key)
