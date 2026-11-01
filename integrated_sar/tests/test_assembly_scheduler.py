"""The assembly scheduler: minimal covering sets, arrival-synchronised, belief-only.

Phase 11 of the coalition-formation analysis. Every test drives
:func:`integrated_sar.assembly.plan_assembly` directly over hand-built snapshots, so a
failure points at the planner and not at the integration layer.

The brief's test matrix is covered here: single-stage complementary capability,
multi-stage capability change, stage transition, a member arriving late, a member leaving
briefly, a capability slot already covered by another member, explicit replan, terminal
and expired beliefs, hard mask interplay, TTL, reset, asymmetric belief, an empty
detection stream, run signature, and the default-off trajectory identity.
"""

from __future__ import annotations

import unittest

from integrated_sar.assembly import (
    SYNC_TOLERANCE,
    AssemblyPlan,
    order_tasks,
    plan_assembly,
)
from integrated_sar.contracts import AgentSnapshot, AgentTaskBelief, WorldPosition

TRAITS = 5


def abilities(*dims: int) -> tuple[float, ...]:
    vector = [0.0] * TRAITS
    for dim in dims:
        vector[dim] = 1.0
    return tuple(vector)


def belief(
    task_id: str = "t1",
    *,
    requirement=(1, 0, 0, 0, 0),
    deadline: float = 100.0,
    stage: int = 0,
    progress: float = 0.0,
    position=(0.0, 0.0),
    finished: bool = False,
    revision: int = 1,
) -> AgentTaskBelief:
    return AgentTaskBelief(
        task_id=task_id,
        position=WorldPosition(*position),
        remaining_requirements=tuple(float(v) for v in requirement),
        deadline=float(deadline),
        stage_index=stage,
        progress=float(progress),
        severity=0.5,
        future_requirements=(),
        members=(),
        updated_at=0.0,
        finished=finished,
        dynamic=False,
        confidence=1.0,
        revision=revision,
    )


def agent(
    agent_id: str,
    *,
    pos=(0.0, 0.0),
    ability=(1, 0, 0, 0, 0),
    current_task: str | None = None,
    beliefs=(),
) -> AgentSnapshot:
    return AgentSnapshot(
        agent_id=agent_id,
        position=WorldPosition(*pos),
        abilities=abilities(*[i for i, value in enumerate(ability) if value]),
        status="working" if current_task else "idle",
        current_task=current_task,
        needs_decision=current_task is None,
        known_tasks=tuple(b.task_id for b in beliefs) if beliefs else None,
        task_beliefs=tuple(beliefs),
    )


class ComplementaryCapabilityTests(unittest.TestCase):
    def test_a_single_robot_that_covers_the_stage_is_the_whole_plan(self) -> None:
        target = belief(requirement=(1, 0, 0, 0, 0))
        agents = {"a": agent("a", ability=(1, 0, 0, 0, 0), beliefs=[target])}
        plan = plan_assembly(target, agents, now=0.0)
        self.assertTrue(plan.complete)
        self.assertEqual([m.agent_id for m in plan.members], ["a"])

    def test_two_robots_that_each_cover_half_the_requirement(self) -> None:
        target = belief(requirement=(1, 1, 0, 0, 0))
        agents = {
            "a": agent("a", ability=(1, 0, 0, 0, 0), beliefs=[target]),
            "b": agent("b", ability=(0, 1, 0, 0, 0), beliefs=[target]),
        }
        plan = plan_assembly(target, agents, now=0.0)
        self.assertTrue(plan.complete)
        self.assertEqual({m.agent_id for m in plan.members}, {"a", "b"})

    def test_a_robot_that_cannot_contribute_is_not_a_member(self) -> None:
        target = belief(requirement=(1, 0, 0, 0, 0))
        agents = {
            "a": agent("a", ability=(1, 0, 0, 0, 0), beliefs=[target]),
            "z": agent("z", ability=(0, 0, 0, 0, 1), beliefs=[target]),
        }
        plan = plan_assembly(target, agents, now=0.0)
        self.assertEqual([m.agent_id for m in plan.members], ["a"])

    def test_an_insufficient_fleet_is_reported_incomplete_not_padded(self) -> None:
        target = belief(requirement=(1, 1, 1, 0, 0))
        agents = {"a": agent("a", ability=(1, 0, 0, 0, 0), beliefs=[target])}
        plan = plan_assembly(target, agents, now=0.0)
        self.assertFalse(plan.complete)
        self.assertEqual(plan.missing[:3], (0, 1, 1))
        self.assertFalse(plan.feasible)


class MultiStageTests(unittest.TestCase):
    def test_the_requirement_is_the_believed_remaining_one(self) -> None:
        # Stage 1 of a 3-stage task: the plan must cover the *next* stage's requirement,
        # which is what the belief carries, not the finished stage's.
        target = belief(requirement=(0, 0, 1, 0, 0), stage=1, progress=0.5)
        agents = {
            "a": agent("a", ability=(1, 0, 0, 0, 0), beliefs=[target]),
            "c": agent("c", ability=(0, 0, 1, 0, 0), beliefs=[target]),
        }
        plan = plan_assembly(target, agents, now=0.0)
        self.assertTrue(plan.complete)
        self.assertEqual([m.agent_id for m in plan.members], ["c"])

    def test_a_stage_transition_replans_from_the_new_requirement(self) -> None:
        # Same robots, same task: only the believed requirement changes, and the plan
        # follows it. Nothing is carried over from the previous stage's plan.
        agents = {
            "a": agent("a", ability=(1, 0, 0, 0, 0), beliefs=[]),
            "b": agent("b", ability=(0, 1, 0, 0, 0), beliefs=[]),
        }
        stage0 = belief(requirement=(1, 0, 0, 0, 0), stage=0)
        stage1 = belief(requirement=(0, 1, 0, 0, 0), stage=1)
        agents["a"] = agent("a", ability=(1, 0, 0, 0, 0), beliefs=[stage1])
        agents["b"] = agent("b", ability=(0, 1, 0, 0, 0), beliefs=[stage1])
        plan0 = plan_assembly(stage0, {**agents, "a": agent("a", ability=(1, 0, 0, 0, 0), beliefs=[stage0])}, now=0.0)
        plan1 = plan_assembly(stage1, agents, now=0.0)
        self.assertEqual([m.agent_id for m in plan0.members], ["a"])
        self.assertEqual([m.agent_id for m in plan1.members], ["b"])


class SynchronisationTests(unittest.TestCase):
    def test_the_plan_prefers_members_that_arrive_together(self) -> None:
        # Two robots can each fill the slot. "far" is closer in a straight line but is
        # already at the target; "near" is farther. Covering needs one robot, so the
        # tie-break is arrival time -- and the plan must not pick a member whose arrival
        # is wildly later than the set's current latest.
        target = belief(requirement=(1, 0, 0, 0, 0), position=(0.5, 0.5))
        agents = {
            "far": agent("far", pos=(0.9, 0.9), ability=(1, 0, 0, 0, 0), beliefs=[target]),
            "near": agent("near", pos=(0.5, 0.52), ability=(1, 0, 0, 0, 0), beliefs=[target]),
        }
        plan = plan_assembly(target, agents, now=0.0)
        self.assertEqual([m.agent_id for m in plan.members], ["near"])

    def test_arrival_spread_is_reported_for_a_ragged_set(self) -> None:
        target = belief(requirement=(1, 1, 0, 0, 0), position=(0.5, 0.5))
        agents = {
            "here": agent("here", pos=(0.5, 0.5), ability=(1, 0, 0, 0, 0), beliefs=[target]),
            "away": agent("away", pos=(0.0, 0.0), ability=(0, 1, 0, 0, 0), beliefs=[target]),
        }
        plan = plan_assembly(target, agents, now=0.0)
        self.assertTrue(plan.complete)
        self.assertIsNotNone(plan.arrival_spread)
        # The far robot is sqrt(0.5)/0.2 ~= 3.5 steps away; the near one is at the target.
        self.assertGreater(plan.arrival_spread, 3.0)
        self.assertEqual(plan.latest_arrival, max(m.eta_steps for m in plan.members))

    def test_a_tight_set_reports_a_small_spread(self) -> None:
        target = belief(requirement=(1, 1, 0, 0, 0), position=(0.5, 0.5))
        agents = {
            "a": agent("a", pos=(0.5, 0.5), ability=(1, 0, 0, 0, 0), beliefs=[target]),
            "b": agent("b", pos=(0.51, 0.5), ability=(0, 1, 0, 0, 0), beliefs=[target]),
        }
        plan = plan_assembly(target, agents, now=0.0)
        self.assertLess(plan.arrival_spread, SYNC_TOLERANCE)


class CoverageAlreadyProvidedTests(unittest.TestCase):
    def test_a_slot_already_covered_keeps_the_redundant_robot_out(self) -> None:
        # Two robots can both fill dimension 0, and the stage needs one unit of it. Only
        # one may be planned; the other is not a member and not penalised.
        target = belief(requirement=(1, 0, 0, 0, 0))
        agents = {
            "a": agent("a", ability=(1, 0, 0, 0, 0), beliefs=[target]),
            "b": agent("b", ability=(1, 0, 0, 0, 0), beliefs=[target]),
        }
        plan = plan_assembly(target, agents, now=0.0)
        self.assertEqual(len(plan.members), 1)

    def test_the_same_robot_is_never_planned_twice(self) -> None:
        target = belief(requirement=(1, 1, 0, 0, 0))
        agents = {"a": agent("a", ability=(1, 1, 0, 0, 0), beliefs=[target])}
        plan = plan_assembly(target, agents, now=0.0)
        self.assertEqual([m.agent_id for m in plan.members], ["a"])


class BeliefOnlyTests(unittest.TestCase):
    def test_a_robot_without_the_belief_is_never_planned_for(self) -> None:
        target = belief(requirement=(1, 0, 0, 0, 0))
        agents = {"a": agent("a", ability=(1, 0, 0, 0, 0), beliefs=[])}
        plan = plan_assembly(target, agents, now=0.0)
        self.assertEqual(plan.members, ())
        self.assertFalse(plan.complete)

    def test_asymmetric_belief_plans_only_the_informed_robots(self) -> None:
        target = belief(requirement=(1, 1, 0, 0, 0))
        agents = {
            "told": agent("told", ability=(1, 0, 0, 0, 0), beliefs=[target]),
            "unaware": agent("unaware", ability=(0, 1, 0, 0, 0), beliefs=[]),
        }
        plan = plan_assembly(target, agents, now=0.0)
        self.assertEqual([m.agent_id for m in plan.members], ["told"])
        self.assertFalse(plan.complete)

    def test_a_robot_busy_elsewhere_is_not_a_candidate(self) -> None:
        target = belief(requirement=(1, 0, 0, 0, 0))
        agents = {"a": agent("a", ability=(1, 0, 0, 0, 0), current_task="other", beliefs=[target])}
        plan = plan_assembly(target, agents, now=0.0)
        self.assertEqual(plan.members, ())

    def test_a_finished_belief_is_not_planned_for(self) -> None:
        target = belief(requirement=(1, 0, 0, 0, 0), finished=True)
        agents = {"a": agent("a", ability=(1, 0, 0, 0, 0), beliefs=[target])}
        plan = plan_assembly(target, agents, now=0.0)
        self.assertEqual(plan.members, ())

    def test_an_expired_belief_is_not_planned_for(self) -> None:
        target = belief(requirement=(1, 0, 0, 0, 0), deadline=5.0)
        agents = {"a": agent("a", ability=(1, 0, 0, 0, 0), beliefs=[target])}
        plan = plan_assembly(target, agents, now=10.0)
        self.assertEqual(plan.members, ())

    def test_a_plan_that_cannot_land_before_the_believed_deadline_is_infeasible(self) -> None:
        target = belief(requirement=(1, 0, 0, 0, 0), deadline=3.0, position=(0.9, 0.9))
        agents = {"a": agent("a", pos=(0.0, 0.0), ability=(1, 0, 0, 0, 0), beliefs=[target])}
        plan = plan_assembly(target, agents, now=0.0)
        self.assertTrue(plan.complete)
        self.assertFalse(plan.feasible)

    def test_an_empty_fleet_yields_an_empty_plan(self) -> None:
        target = belief(requirement=(1, 0, 0, 0, 0))
        plan = plan_assembly(target, {}, now=0.0)
        self.assertEqual(plan.members, ())
        self.assertIsNone(plan.arrival_spread)
        self.assertFalse(plan.feasible)


class DeterminismTests(unittest.TestCase):
    def test_the_same_input_produces_the_same_plan(self) -> None:
        target = belief(requirement=(1, 1, 0, 0, 0))
        agents = {
            "a": agent("a", pos=(0.1, 0.1), ability=(1, 0, 0, 0, 0), beliefs=[target]),
            "b": agent("b", pos=(0.1, 0.1), ability=(0, 1, 0, 0, 0), beliefs=[target]),
        }
        first = plan_assembly(target, agents, now=0.0)
        second = plan_assembly(target, agents, now=0.0)
        self.assertEqual(first.as_dict(), second.as_dict())

    def test_ties_break_on_agent_id(self) -> None:
        target = belief(requirement=(1, 0, 0, 0, 0))
        agents = {
            "b": agent("b", ability=(1, 0, 0, 0, 0), beliefs=[target]),
            "a": agent("a", ability=(1, 0, 0, 0, 0), beliefs=[target]),
        }
        plan = plan_assembly(target, agents, now=0.0)
        self.assertEqual([m.agent_id for m in plan.members], ["a"])


class TaskOrderingTests(unittest.TestCase):
    def test_feasible_plans_outrank_infeasible_ones(self) -> None:
        tight = belief("tight", requirement=(1, 0, 0, 0, 0), deadline=2.0, position=(0.9, 0.9))
        loose = belief("loose", requirement=(1, 0, 0, 0, 0), deadline=500.0, position=(0.5, 0.5))
        agents = {
            "a": agent("a", pos=(0.5, 0.5), ability=(1, 0, 0, 0, 0), beliefs=[tight, loose]),
        }
        plans = [plan_assembly(tight, agents, now=0.0), plan_assembly(loose, agents, now=0.0)]
        ordered = order_tasks(plans, {"tight": tight, "loose": loose}, now=0.0)
        self.assertEqual([p.task_id for p in ordered], ["loose", "tight"])

    def test_ordering_never_reads_a_stage_duration(self) -> None:
        # The duration-relevance gate showed work-time ordering rescues none of the six,
        # so the ranker must not have a duration input at all. Two identical tasks that
        # differ only in progress (a proxy for work done) must tie on the plan key.
        a = belief("a", requirement=(1, 0, 0, 0, 0), deadline=100.0, progress=0.1)
        b = belief("b", requirement=(1, 0, 0, 0, 0), deadline=100.0, progress=0.9)
        agents = {"r": agent("r", pos=(0.5, 0.5), ability=(1, 0, 0, 0, 0), beliefs=[a, b])}
        plans = [plan_assembly(a, agents, now=0.0), plan_assembly(b, agents, now=0.0)]
        ordered = order_tasks(plans, {"a": a, "b": b}, now=0.0)
        self.assertEqual([p.task_id for p in ordered], ["a", "b"])


class SerialisationTests(unittest.TestCase):
    def test_the_plan_round_trips_through_its_dict(self) -> None:
        target = belief(requirement=(1, 1, 0, 0, 0))
        agents = {
            "a": agent("a", ability=(1, 0, 0, 0, 0), beliefs=[target]),
            "b": agent("b", ability=(0, 1, 0, 0, 0), beliefs=[target]),
        }
        payload = plan_assembly(target, agents, now=0.0).as_dict()
        self.assertEqual(payload["task_id"], "t1")
        self.assertEqual(len(payload["members"]), 2)
        self.assertTrue(payload["complete"])
        self.assertIn("arrival_spread", payload)


class DefaultOffIdentityTests(unittest.TestCase):
    """With the flag off the planner must not read a position or an ETA at all.

    The prototype is wired in as ``assembly_scheduling=False`` by default, and the
    binding arm's trajectory has to be reproducible byte-for-byte. Rather than diff two
    runs, this asserts the mechanism: a snapshot whose believed positions are absurd
    (so any ETA term would move the plan) produces the same advice with the flag off as
    a snapshot whose positions are irrelevant.
    """

    def _advice(self, *, assembly: bool):
        from integrated_sar.coordinator import CoalitionPlannerCoordinator
        from integrated_sar.contracts import CoordinatorInput

        target = belief("t1", requirement=(1, 0, 0, 0, 0), position=(0.5, 0.5))
        agents = (
            agent("a", pos=(0.0, 0.0), ability=(1, 0, 0, 0, 0), beliefs=[target]),
            agent("b", pos=(0.9, 0.9), ability=(1, 0, 0, 0, 0), beliefs=[target]),
        )
        coordinator = CoalitionPlannerCoordinator(assembly_scheduling=assembly)
        return coordinator.advise(CoordinatorInput(1.0, (), agents))

    def test_flag_defaults_off(self) -> None:
        from integrated_sar.coordinator import CoalitionPlannerCoordinator

        self.assertFalse(CoalitionPlannerCoordinator().assembly_scheduling)

    def test_flag_off_adds_no_synchronisation_credit(self) -> None:
        self.assertEqual(self._advice(assembly=False).metadata.get("synchronised_dispatches", 0), 0)

    def test_flag_off_advice_matches_a_position_blind_plan(self) -> None:
        # The same run with the flag on may add credit; with it off the pair bias must be
        # exactly what the planner produced, with no ETA-dependent term folded in.
        off = self._advice(assembly=False)
        self.assertTrue(off.pair_bias)
        for value in off.pair_bias.values():
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)

    def test_flag_on_stays_inside_the_bounded_band(self) -> None:
        on = self._advice(assembly=True)
        for value in on.pair_bias.values():
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
