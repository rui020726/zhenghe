"""The deterministic coalition planner: who goes with whom, and on what grounds.

The pair interface cannot express a group, so this policy plans one and publishes
it as a reservation per selected robot. What has to hold:

* two robots with complementary abilities are convened on the same task, while a
  robot whose unit the plan already has is passed over as redundant;
* no two robots hold the same capability slot of one task;
* the plan is built from each robot's *own* belief, so asymmetric or incomplete
  knowledge is planned against -- never averaged away and never borrowed from a
  neighbour;
* a robot that is not free is not reassigned, a task past its believed deadline is
  nobody's candidate, and a task the robot has not heard of produces no bias at all;
* committed state has a lifecycle: a bumped revision supersedes, a changed plan
  cancels, the clock expires, and ``reset`` clears everything;
* the option is off by default, so the existing policies and their trajectories are
  untouched.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from typing import Any

from integrated_sar.contracts import (
    AgentSnapshot,
    AgentTaskBelief,
    CoordinatorInput,
    TaskSnapshot,
    TaskStageSpec,
    TaskStatus,
    WorldPosition,
)
from integrated_sar.coordinator import (
    IDLE_STATUSES,
    CoalitionAwareCoordinator,
    CoalitionPlannerCoordinator,
    HeuristicCoordinator,
    PassthroughCoordinator,
    _claim_slot,
    _gap_fill_ratio,
    _open_units,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None

TASK = "target-a"
OTHER = "target-b"


def belief(
    task_id: str = TASK,
    *,
    requirements: tuple[float, ...] = (3, 0, 0, 0, 0),
    deadline: float = 1000.0,
    progress: float = 0.0,
    severity: float = 0.5,
    position: tuple[float, float] = (0.5, 0.5),
    finished: bool = False,
    confidence: float = 0.5,
    revision: int = 1,
) -> AgentTaskBelief:
    return AgentTaskBelief(
        task_id=task_id,
        position=WorldPosition(*position),
        remaining_requirements=tuple(requirements),
        deadline=deadline,
        stage_index=0,
        progress=progress,
        severity=severity,
        finished=finished,
        confidence=confidence,
        revision=revision,
    )


def agent(
    agent_id: str,
    abilities: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0),
    *,
    current_task: str | None = None,
    status: str = "idle",
    position: tuple[float, float] = (0.5, 0.5),
    beliefs: tuple[AgentTaskBelief, ...] = (),
    known: tuple[str, ...] | None = None,
) -> AgentSnapshot:
    return AgentSnapshot(
        agent_id=agent_id,
        position=WorldPosition(*position),
        abilities=abilities,
        status=status,
        current_task=current_task,
        known_tasks=known if known is not None else tuple(b.task_id for b in beliefs),
        task_beliefs=beliefs,
    )


def registry_truth(
    task_id: str = TASK, *, requirements: tuple[int, ...] = (3, 0, 0, 0, 0)
) -> TaskSnapshot:
    return TaskSnapshot(
        task_id=task_id,
        source_event_id=f"detection:{task_id}",
        position=WorldPosition(0.1, 0.1),
        stages=(TaskStageSpec(requirements, 3.0),),
        current_stage=0,
        remaining_requirements=requirements,
        progress=0.0,
        status=TaskStatus.DETECTED,
        deadline=999.0,
        confidence=1.0,
        severity=0.5,
        discovered_at=0.0,
        updated_at=0.0,
        revision=7,
    )


def state(
    *agents: AgentSnapshot, current_time: float = 0.0, truth: bool = False
) -> CoordinatorInput:
    return CoordinatorInput(
        current_time=current_time,
        tasks=(registry_truth(),) if truth else (),
        agents=agents,
    )


class SlotArithmeticTests(unittest.TestCase):
    """The unit-slot view of a requirement, tested where it is defined."""

    def test_a_requirement_flattens_to_unit_slots(self) -> None:
        # (3, 0, ...) is three slots, all in dimension 0: indices 0, 1 and 2.
        self.assertEqual(_open_units((3, 0, 0, 0, 0), []), [3, 0, 0, 0, 0])
        self.assertEqual(
            _claim_slot((3, 0, 0, 0, 0), [], (1.0, 0.0, 0.0, 0.0, 0.0)), (0, 0)
        )
        self.assertEqual(
            _claim_slot((3, 0, 0, 0, 0), [(1.0, 0.0, 0.0, 0.0, 0.0)], (1.0, 0.0, 0.0, 0.0, 0.0)),
            (1, 0),
        )

    def test_slots_of_a_later_dimension_are_offset_by_the_earlier_ones(self) -> None:
        # Two units of dimension 0, then dimension 1: its first slot is index 2.
        self.assertEqual(
            _claim_slot((2, 1, 0, 0, 0), [], (0.0, 1.0, 0.0, 0.0, 0.0)), (2, 1)
        )

    def test_a_robot_that_fills_nothing_claims_no_slot(self) -> None:
        self.assertIsNone(_claim_slot((2, 0, 0, 0, 0), [], (0.0, 1.0, 0.0, 0.0, 0.0)))
        self.assertIsNone(
            _claim_slot((1, 0, 0, 0, 0), [(1.0, 0.0, 0.0, 0.0, 0.0)], (1.0, 0.0, 0.0, 0.0, 0.0))
        )

    def test_a_surplus_ability_cannot_drive_the_gap_negative(self) -> None:
        self.assertEqual(
            _open_units((1, 0, 0, 0, 0), [(2.0, 0.0, 0.0, 0.0, 0.0)]), [0, 0, 0, 0, 0]
        )

    def test_the_fill_ratio_is_the_share_of_the_open_units(self) -> None:
        self.assertAlmostEqual(_gap_fill_ratio((4, 0, 0, 0, 0), [], (1.0, 0.0, 0.0, 0.0, 0.0)), 0.25)
        self.assertAlmostEqual(_gap_fill_ratio((2, 0, 0, 0, 0), [], (9.0, 0.0, 0.0, 0.0, 0.0)), 1.0)
        self.assertAlmostEqual(
            _gap_fill_ratio((2, 0, 0, 0, 0), [(2.0, 0.0, 0.0, 0.0, 0.0)], (1.0, 0.0, 0.0, 0.0, 0.0)),
            0.0,
        )


class PlanFormationTests(unittest.TestCase):
    """Who the plan convenes, from beliefs alone."""

    def setUp(self) -> None:
        self.planner = CoalitionPlannerCoordinator()

    def test_complementary_robots_are_reserved_for_the_same_task(self) -> None:
        # The stage needs one unit of two different abilities, so the group is one
        # robot of each -- and the third robot, a second unit of an ability the stage
        # already has, is not sent.
        advice = self.planner.advise(
            state(
                agent("0", (1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(requirements=(1, 1, 0, 0, 0)),)),
                agent("1", (0.0, 1.0, 0.0, 0.0, 0.0), beliefs=(belief(requirements=(1, 1, 0, 0, 0)),)),
                agent("2", (1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(requirements=(1, 1, 0, 0, 0)),)),
            )
        )
        self.assertEqual(set(advice.reservations), {("0", TASK), ("1", TASK)})
        self.assertEqual(advice.reservations[("0", TASK)].slot_dimension, 0)
        self.assertEqual(advice.reservations[("1", TASK)].slot_dimension, 1)
        self.assertNotIn(("2", TASK), advice.reservations)

    def test_no_two_robots_hold_the_same_slot(self) -> None:
        # Three units of one ability: three robots, three distinct unit slots, and a
        # fourth identical robot left out because the plan has the units on the way.
        robots = (
            agent("0", (1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(),)),
            agent("1", (1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(),)),
            agent("2", (1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(),)),
            agent("3", (1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(),)),
        )
        advice = self.planner.advise(state(*robots))
        held = [entry.ability_slot for entry in advice.reservations.values()]
        self.assertEqual(sorted(held), [0, 1, 2])
        self.assertEqual(len(set(held)), len(held))
        self.assertNotIn(("3", TASK), advice.reservations)
        self.assertEqual(advice.metadata["redundant_candidates"], 1)

    def test_a_robot_already_under_order_counts_toward_the_cover(self) -> None:
        # The committed robot brings two of the three units, so exactly one more
        # robot is convened -- for the third slot, not the first.
        advice = self.planner.advise(
            state(
                agent("0", (1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(),)),
                agent("9", (2.0, 0.0, 0.0, 0.0, 0.0), current_task=TASK, status="moving", beliefs=(belief(),)),
            )
        )
        self.assertEqual(set(advice.reservations), {("0", TASK)})
        self.assertEqual(advice.reservations[("0", TASK)].ability_slot, 2)

    def test_the_closer_robot_is_chosen_and_the_other_steered_away(self) -> None:
        # One slot, two identical robots, different distances: the near one is
        # reserved, and the far one's pair bias turns negative for that task.
        advice = self.planner.advise(
            state(
                agent("near", (1.0, 0.0, 0.0, 0.0, 0.0), position=(0.5, 0.5), beliefs=(belief(requirements=(1, 0, 0, 0, 0)),)),
                agent("far", (1.0, 0.0, 0.0, 0.0, 0.0), position=(1.0, 1.0), beliefs=(belief(requirements=(1, 0, 0, 0, 0)),)),
            )
        )
        self.assertEqual(set(advice.reservations), {("near", TASK)})
        self.assertGreater(advice.pair_bias[("near", TASK)], 0.0)
        self.assertLess(advice.pair_bias[("far", TASK)], 0.0)

    def test_the_task_past_its_deadline_is_nobodys_candidate(self) -> None:
        advice = self.planner.advise(
            state(agent("0", beliefs=(belief(deadline=3.0),)), current_time=5.0)
        )
        self.assertEqual(advice.reservations, {})
        # Neither channel carries it: a priority for a task the robot believes is over
        # would pull it out of the field at saturated urgency.
        self.assertNotIn(("0", TASK), advice.pair_bias)
        self.assertNotIn(("0", TASK), advice.agent_task_bias)

    def test_a_busy_robot_is_not_reassigned(self) -> None:
        # A robot on site for another task is not a candidate for this one, gets no
        # slot and no pair bias -- only the priority channel still reaches it.
        advice = self.planner.advise(
            state(
                agent("0", (1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(),)),
                agent("1", (1.0, 0.0, 0.0, 0.0, 0.0), current_task=OTHER, status="working", beliefs=(belief(),)),
            )
        )
        self.assertNotIn(("1", TASK), advice.reservations)
        self.assertNotIn(("1", TASK), advice.pair_bias)
        self.assertIn(("1", TASK), advice.agent_task_bias)

    def test_an_unheard_of_task_produces_no_bias_or_reservation(self) -> None:
        # This robot holds a belief about the task but its knowledge list says it has
        # not heard of it, so nothing may be biased towards it.
        advice = self.planner.advise(
            state(agent("0", beliefs=(belief(),), known=()))
        )
        self.assertEqual(advice.reservations, {})
        self.assertEqual(advice.pair_bias, {})
        self.assertEqual(advice.agent_task_bias, {})

    def test_a_finished_task_is_dropped_from_the_plan(self) -> None:
        advice = self.planner.advise(
            state(agent("0", beliefs=(belief(finished=True),)))
        )
        self.assertEqual(advice.reservations, {})
        self.assertEqual(advice.pair_bias, {})


class AsymmetricBeliefTests(unittest.TestCase):
    """Every row is built from its own robot's view, never a neighbour's."""

    def setUp(self) -> None:
        self.planner = CoalitionPlannerCoordinator()

    def test_a_fuller_belief_reserves_more_and_a_thinner_one_less(self) -> None:
        # Robot 0 believes three units are still needed, robot 1 believes one. Each
        # is planned against its own view, so robot 1 -- whose single unit covers its
        # whole believed gap -- carries the larger fill, and both are convened.
        advice = self.planner.advise(
            state(
                agent("0", (1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(requirements=(3, 0, 0, 0, 0), revision=4),)),
                agent("1", (1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(requirements=(1, 0, 0, 0, 0), revision=9),)),
            )
        )
        self.assertGreater(
            advice.reservations[("1", TASK)].score,
            advice.reservations[("0", TASK)].score,
        )
        # Each commitment carries the revision of the robot whose belief produced it.
        self.assertEqual(advice.reservations[("0", TASK)].task_revision, 4)
        self.assertEqual(advice.reservations[("1", TASK)].task_revision, 9)

    def test_an_incomplete_belief_still_plans_what_that_robot_can_see(self) -> None:
        # Robot 2 knows nothing about the task: it is not planned for it. Robot 0
        # still gets its own reservation rather than none because nobody else knows.
        advice = self.planner.advise(
            state(
                agent("0", (1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(requirements=(1, 0, 0, 0, 0)),)),
                agent("2", (0.0, 1.0, 0.0, 0.0, 0.0), beliefs=(belief(task_id=OTHER),)),
            )
        )
        self.assertEqual(set(advice.reservations), {("0", TASK)})
        self.assertEqual(advice.reservations[("0", TASK)].task_revision, 1)

    def test_one_robots_belief_cannot_move_another_robots_row(self) -> None:
        base = agent("0", (1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(requirements=(1, 0, 0, 0, 0)),))
        neighbour = agent(
            "1", (0.0, 1.0, 0.0, 0.0, 0.0), beliefs=(belief(task_id=OTHER, deadline=1000.0),)
        )
        moved = agent(
            "1", (0.0, 1.0, 0.0, 0.0, 0.0), beliefs=(belief(task_id=OTHER, deadline=2.0),)
        )
        before = self.planner.advise(state(base, neighbour))
        after = self.planner.advise(state(base, moved))

        # The neighbour's own priority moved, so the change is real ...
        self.assertNotEqual(
            before.agent_task_bias[("1", OTHER)], after.agent_task_bias[("1", OTHER)]
        )
        # ... and robot 0's row for the task it believes in did not.
        self.assertEqual(
            before.pair_bias[("0", TASK)], after.pair_bias[("0", TASK)]
        )
        self.assertEqual(
            before.reservations[("0", TASK)], after.reservations[("0", TASK)]
        )

    def test_the_stale_view_is_the_view_that_gets_planned(self) -> None:
        # The robot's belief still shows one unit open, so it is convened even though
        # the fleet already has half the stage on the way. Belief-only means
        # belief-only: the plan cannot borrow a fresher view it never received.
        advice = self.planner.advise(
            state(
                agent("0", (1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(requirements=(2, 0, 0, 0, 0), progress=0.0, revision=2),)),
                agent("9", (1.0, 0.0, 0.0, 0.0, 0.0), current_task=TASK, status="moving"),
            )
        )
        self.assertIn(("0", TASK), advice.reservations)


class ReservationLifecycleTests(unittest.TestCase):
    """Expiry, revision bumps, plan changes and reset."""

    def setUp(self) -> None:
        self.planner = CoalitionPlannerCoordinator(validity=100.0, reservation_validity=2.0)

    def _single(self, *, revision: int = 1, requirements: tuple[float, ...] = (3, 0, 0, 0, 0), now: float = 0.0, status: str = "idle", current_task: str | None = None) -> Any:
        return self.planner.advise(
            state(
                agent(
                    "0",
                    (1.0, 0.0, 0.0, 0.0, 0.0),
                    status=status,
                    current_task=current_task,
                    beliefs=(belief(requirements=requirements, revision=revision),),
                ),
                current_time=now,
            )
        )

    def test_the_advice_never_outlives_its_shortest_commitment(self) -> None:
        advice = self._single()
        self.assertIn(("0", TASK), advice.reservations)
        # validity alone would allow 100; the commitment caps it at 2.
        self.assertEqual(advice.valid_until, 2.0)
        self.assertEqual(advice.reservations[("0", TASK)].valid_until, 2.0)
        self.assertTrue(advice.reservations[("0", TASK)].active_at(2.0))
        self.assertFalse(advice.reservations[("0", TASK)].active_at(2.5))

    def test_the_clock_expires_the_held_commitment(self) -> None:
        self._single()
        after = self._single(now=5.0)
        self.assertEqual(after.metadata["expired_reservations"], 1)
        # The plan is otherwise unchanged, so the same robot is committed afresh.
        self.assertEqual(after.reservations[("0", TASK)].issued_at, 5.0)
        self.assertEqual(after.reservations[("0", TASK)].valid_until, 7.0)

    def test_a_valid_advice_does_not_report_an_expiry(self) -> None:
        self._single()
        after = self._single(now=1.0)
        self.assertEqual(after.metadata["expired_reservations"], 0)
        self.assertEqual(after.metadata["cancelled_reservations"], 0)
        self.assertEqual(after.metadata["superseded_reservations"], 0)

    def test_a_bumped_revision_supersedes_the_held_commitment(self) -> None:
        self._single(revision=1)
        # The robot's own view moves on and the work is now covered, so the plan no
        # longer picks it. The old commitment is superseded, not silently reused.
        after = self.planner.advise(
            state(
                agent("0", (1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(requirements=(0, 0, 0, 0, 0), revision=2),)),
                current_time=1.0,
            )
        )
        self.assertEqual(after.reservations, {})
        self.assertEqual(after.metadata["superseded_reservations"], 1)
        self.assertEqual(after.metadata["cancelled_reservations"], 0)

    def test_a_changed_plan_cancels_without_a_belief_change(self) -> None:
        self._single()
        # Same belief, same revision, but the robot is now on its way somewhere else,
        # so it is no longer free: the commitment is dropped as cancelled.
        after = self._single(now=1.0, status="moving", current_task=OTHER)
        self.assertEqual(after.metadata["cancelled_reservations"], 1)
        self.assertEqual(after.metadata["superseded_reservations"], 0)

    def test_a_lost_belief_supersedes_rather_than_cancels(self) -> None:
        self._single()
        after = self.planner.advise(
            state(agent("0", (1.0, 0.0, 0.0, 0.0, 0.0), known=()), current_time=1.0)
        )
        self.assertEqual(after.metadata["superseded_reservations"], 1)

    def test_reset_clears_every_commitment(self) -> None:
        self._single()
        self.planner.reset()
        after = self._single(now=1.0)
        # A fresh episode: no leaked lifecycle counts and numbering restarted.
        self.assertEqual(after.revision, 1)
        self.assertEqual(after.metadata["expired_reservations"], 0)
        self.assertEqual(after.metadata["cancelled_reservations"], 0)
        self.assertEqual(after.metadata["superseded_reservations"], 0)
        self.assertIn(("0", TASK), after.reservations)


class PlanDeterminismTests(unittest.TestCase):
    """Same input, same plan -- including from a cold instance."""

    def _state(self) -> CoordinatorInput:
        return state(
            agent("0", (1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(requirements=(2, 1, 0, 0, 0)),)),
            agent("1", (1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(requirements=(2, 1, 0, 0, 0)),)),
            agent("2", (0.0, 1.0, 0.0, 0.0, 0.0), beliefs=(belief(requirements=(2, 1, 0, 0, 0)),)),
            agent("3", (0.0, 0.0, 1.0, 0.0, 0.0), beliefs=(belief(requirements=(2, 1, 0, 0, 0)),)),
        )

    def test_the_same_state_plans_the_same_way_twice(self) -> None:
        first = CoalitionPlannerCoordinator().advise(self._state())
        second = CoalitionPlannerCoordinator().advise(self._state())
        self.assertEqual(first.reservations, second.reservations)
        self.assertEqual(first.pair_bias, second.pair_bias)
        self.assertEqual(first.agent_task_bias, second.agent_task_bias)
        self.assertEqual(first.metadata, second.metadata)

    def test_a_warm_instance_plans_the_same_as_a_cold_one(self) -> None:
        warm = CoalitionPlannerCoordinator()
        warm.advise(self._state())
        self.assertEqual(warm.advise(self._state()).reservations, CoalitionPlannerCoordinator().advise(self._state()).reservations)

    def test_the_plan_covers_the_open_slots_and_no_more(self) -> None:
        advice = CoalitionPlannerCoordinator().advise(self._state())
        # (2, 1) is three units, and four robots are free with three units each.
        self.assertEqual(len(advice.reservations), 3)
        self.assertEqual(
            sorted(entry.ability_slot for entry in advice.reservations.values()), [0, 1, 2]
        )


class PlannerLeakTests(unittest.TestCase):
    """Registry truth in, nothing different out."""

    def test_the_registry_snapshot_cannot_change_the_advice(self) -> None:
        robots = (
            agent("0", beliefs=(belief(),)),
            agent("1", (1.0, 0.0, 0.0, 0.0, 0.0), current_task=TASK, status="moving"),
        )
        planner = CoalitionPlannerCoordinator()
        without = planner.advise(state(*robots))
        with_truth = CoalitionPlannerCoordinator().advise(state(*robots, truth=True))
        self.assertEqual(with_truth.reservations, without.reservations)
        self.assertEqual(with_truth.pair_bias, without.pair_bias)
        self.assertEqual(with_truth.agent_task_bias, without.agent_task_bias)

    def test_nothing_is_published_on_the_pool_level(self) -> None:
        # A pool-level bias reaches every robot, so a belief-derived number must never
        # land there.
        advice = CoalitionPlannerCoordinator().advise(
            state(agent("0", beliefs=(belief(),)))
        )
        self.assertEqual(advice.task_bias, {})


class PlannerDefaultOffTests(unittest.TestCase):
    """The option is off unless it is asked for, at every layer."""

    def test_the_driver_default_is_still_the_heuristic(self) -> None:
        from integrated_sar.joint_driver import RunConfiguration, _build_coordinator

        configuration = RunConfiguration()
        self.assertEqual(configuration.coordinator_policy, "heuristic")
        self.assertIsInstance(_build_coordinator(configuration), HeuristicCoordinator)

    def test_the_driver_can_select_the_planner(self) -> None:
        from integrated_sar.joint_driver import RunConfiguration, _build_coordinator

        configuration = RunConfiguration(coordinator_policy="coalition_planner")
        self.assertIsInstance(
            _build_coordinator(configuration), CoalitionPlannerCoordinator
        )

    def test_no_pre_existing_policy_emits_a_reservation(self) -> None:
        robots = (
            agent("0", beliefs=(belief(requirements=(3, 0, 0, 0, 0)),)),
            agent("1", (0.0, 1.0, 0.0, 0.0, 0.0), beliefs=(belief(),)),
        )
        sample = state(*robots)
        for policy in (
            PassthroughCoordinator(),
            HeuristicCoordinator(),
            CoalitionAwareCoordinator(),
        ):
            with self.subTest(policy=type(policy).__name__):
                self.assertEqual(policy.advise(sample).reservations, {})

    def test_the_idle_statuses_are_the_ones_task03_reports_as_free(self) -> None:
        # Pinned because a status task-03 reports for a free robot but that is missing
        # here would silently stop the planner from ever reserving anyone.
        self.assertEqual(IDLE_STATUSES, frozenset({"idle", "available"}))


class PlannerDriverTests(unittest.TestCase):
    """End to end through the mock pipeline."""

    def _run(self, directory: str, policy: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        from integrated_sar.joint_driver import (
            RunConfiguration,
            run_from_configuration,
        )
        from integrated_sar.tests.test_joint_driver import read_jsonl

        configuration = RunConfiguration(
            mock=True,
            steps=20,
            seed=42,
            quiet=True,
            output=os.path.join(directory, f"lifecycle-{policy}.jsonl"),
            coordinator_policy=policy,
        )
        summary = run_from_configuration(configuration)
        return summary, read_jsonl(configuration.output)

    def test_the_planner_runs_the_mock_episode_without_a_rejected_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary, records = self._run(directory, "coalition_planner")

        self.assertEqual(summary["rejected_assignments"], 0)
        self.assertGreater(summary["targets_discovered"], 0)
        self.assertGreater(summary["assignment_commands"], 0)
        kinds = [record["event_type"] for record in records]
        self.assertIn("assignment", kinds)
        self.assertIn("final", kinds)

    def test_the_default_run_is_unchanged_by_the_new_policy_being_available(self) -> None:
        # Two default runs must agree bit for bit: adding a policy must not perturb
        # the default trajectory, and the pipeline stays reproducible either way.
        with tempfile.TemporaryDirectory() as directory:
            first, first_records = self._run(directory, "heuristic")
            second, second_records = self._run(directory, "heuristic")
        self.assertEqual(first, second)
        self.assertEqual(first_records, second_records)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class PlannerHardMaskTests(unittest.TestCase):
    """The plan is advice: it can reorder legal actions, never legalise one."""

    def test_a_masked_task_is_not_re_legalised_by_a_planned_slot(self) -> None:
        import numpy as np
        import torch

        from integrated_sar.tests.test_ctde_task_logit_offset import (
            ACTION_MAP,
            _make_adapter,
        )

        mask = np.array([False, False, True], dtype=bool)
        logits = torch.tensor([[0.30, 0.05, 0.05]])
        # One unit per task, so the plan splits the two robots across the two tasks
        # instead of stacking them on the first. Robot 1 takes target-b, which the
        # environment masks: a robot holds one task at a time, so a second robot is
        # what puts a commitment on the masked action at all.
        one_unit = (1, 0, 0, 0, 0)
        advice = CoalitionPlannerCoordinator().advise(
            state(
                agent(
                    "0",
                    (1.0, 0.0, 0.0, 0.0, 0.0),
                    beliefs=(belief(TASK, requirements=one_unit), belief(OTHER, requirements=one_unit)),
                ),
                agent(
                    "1",
                    (1.0, 0.0, 0.0, 0.0, 0.0),
                    beliefs=(belief(TASK, requirements=one_unit), belief(OTHER, requirements=one_unit)),
                ),
            )
        )
        self.assertIn(("1", OTHER), advice.reservations)

        adapter = _make_adapter(offset=0.0, mask=mask, logits=logits)
        adapter.residual_scale = 1.0
        verdict = adapter._network_verdict(1, list(ACTION_MAP), advice)
        self.assertIsNone(verdict.final[2])
        self.assertNotEqual(verdict.index, 2)
        self.assertAlmostEqual(verdict.final[0], 0.30)


if __name__ == "__main__":
    unittest.main()
