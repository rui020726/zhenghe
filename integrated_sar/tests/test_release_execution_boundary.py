"""What the execution boundary really verifies before a release runs.

A release command has two ends, and the boundary between planning and execution can only see
the state of the world, not the reasoning. This module pins each of the six ways the world can
differ between the moment a release was judged and the moment its command is applied:

1. the source task moved to another stage;
2. the stage held, but a teammate the verdict counted on is gone;
3. two commands in one batch, each legal on its own, that together strip the source;
4. the robot itself already left the task the command says it is leaving;
5. the destination check fails (expired, unknown task, stale revision);
6. the belief carried no next-stage vector, so no future-stage protection was applied.

The first five are checked where commands are validated; the sixth is a planner-side fact and
is counted there, because "no information" must not read as "safe at every future moment".

    D:/project_env/python.exe -m unittest integrated_sar.tests.test_release_execution_boundary
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from integrated_sar.assignment_gate import AssignmentGate
from integrated_sar.clock import GlobalClock
from integrated_sar.contracts import (
    AgentSnapshot,
    AgentTaskBelief,
    AssignmentCommand,
    CoordinatorInput,
    WorldPosition,
)
from integrated_sar.coordinator import CoalitionPlannerCoordinator

TRAITS = 5


class _RecordingExecutor:
    def __init__(self) -> None:
        self.applied: list[AssignmentCommand] = []

    def apply_assignments(self, commands) -> None:
        self.applied.extend(commands)

    def step(self, current_time: float, dt: float):
        raise AssertionError("the gate must not step the executor while applying commands")

    def reset(self) -> None:
        self.applied.clear()


def command(
    *,
    agent_id: str = "0",
    task_id: str = "t-new",
    revision: int | None = 5,
    valid_until: float = 20.0,
    release_from: str | None = "t-old",
    release_stage: int = 0,
    support: tuple[str, ...] = ("1",),
) -> AssignmentCommand:
    metadata: dict[str, object] = {}
    if release_from is not None:
        metadata = {
            "release_from_task": release_from,
            "release_stage_index": release_stage,
            "release_revision": 3,
            "release_support_members": support,
        }
    return AssignmentCommand(
        command_id=f"assignment:{agent_id}",
        agent_id=agent_id,
        task_id=task_id,
        issued_at=10.0,
        valid_until=valid_until,
        task_revision=revision,
        metadata=metadata,
    )


def gate_for(
    executor,
    *,
    stage=0,
    revision=9,
    roster: tuple[str, ...] | None = ("0", "1"),
    now: float = 10.0,
    with_roster: bool = True,
):
    clock = GlobalClock()
    clock.advance_to(now)
    return AssignmentGate(
        executor,
        clock=clock,
        revision_of=lambda task_id: 5 if task_id in ("t-new", "t-other") else None,
        release_state_of=lambda task_id: (int(stage), int(revision)),
        release_roster_of=(lambda task_id: roster) if with_roster else None,
    )


def reasons(gate: AssignmentGate) -> list[str]:
    return [row.reason for row in gate.drain_rejections()]


class ReleaseExecutionBoundaryTests(unittest.TestCase):
    def test_1_the_source_moved_to_another_stage(self) -> None:
        executor = _RecordingExecutor()
        gate = gate_for(executor, stage=1)
        gate.apply_assignments([command(release_stage=0)])
        self.assertEqual(executor.applied, [])
        self.assertEqual(reasons(gate), ["stale_release_stage"])
        self.assertEqual(gate.pending_replan, {"0"})

    def test_2_a_supporter_left_the_source_after_the_verdict(self) -> None:
        # The verdict said "robot 1 covers what robot 0 was supplying". Robot 1 is gone, so
        # the judgement describes a team that no longer stands on the task.
        executor = _RecordingExecutor()
        gate = gate_for(executor, roster=("0",))
        gate.apply_assignments([command(agent_id="0", support=("1",))])
        self.assertEqual(executor.applied, [])
        self.assertEqual(reasons(gate), ["stale_release_support"])
        self.assertEqual(gate.pending_replan, {"0"})

    def test_3_two_legal_releases_that_together_strip_the_source(self) -> None:
        # Each command is legal on its own: 0 is spare because 1 is there, 1 is spare because
        # 0 is there. Applied together they leave nothing. The batch is therefore validated
        # against the state each earlier acceptance produces, not against a frozen snapshot.
        executor = _RecordingExecutor()
        gate = gate_for(executor, roster=("0", "1"))
        first = command(agent_id="0", support=("1",))
        second = command(agent_id="1", task_id="t-other", support=("0",))
        gate.apply_assignments([first, second])
        self.assertEqual([row.agent_id for row in executor.applied], ["0"])
        self.assertEqual(reasons(gate), ["stale_release_support"])
        self.assertEqual(gate.pending_replan, {"1"})

    def test_4_the_robot_already_left_the_task_it_is_said_to_release(self) -> None:
        executor = _RecordingExecutor()
        gate = gate_for(executor, roster=("1", "2"))
        gate.apply_assignments([command(agent_id="0", support=("1",))])
        self.assertEqual(executor.applied, [])
        self.assertEqual(reasons(gate), ["stale_release_agent_moved"])

    def test_5_the_destination_checks_keep_their_order_and_strength(self) -> None:
        for kwargs, expected in (
            ({"valid_until": 1.0}, "expired_command"),
            ({"task_id": "t-gone"}, "unknown_task"),
            ({"revision": 4}, "stale_task_revision"),
        ):
            executor = _RecordingExecutor()
            gate = gate_for(executor, stage=7, roster=())
            gate.apply_assignments([command(**kwargs)])
            self.assertEqual(reasons(gate), [expected])
            self.assertEqual(executor.applied, [])

    def test_a_still_supported_release_is_executed_and_nothing_is_recorded(self) -> None:
        executor = _RecordingExecutor()
        gate = gate_for(executor, roster=("0", "1", "2"))
        issued = command(agent_id="0", support=("1", "2"))
        gate.apply_assignments([issued])
        self.assertEqual(executor.applied, [issued])
        self.assertEqual(gate.drain_rejections(), ())
        self.assertEqual(gate.pending_replan, set())

    def test_a_source_the_boundary_cannot_read_is_left_to_the_stage_check(self) -> None:
        # Refusing because a roster cannot be read would park the robot for a bookkeeping
        # gap. Without a readable roster the gate behaves exactly as it did before this check
        # existed: the stage comparison still applies.
        executor = _RecordingExecutor()
        gate = gate_for(executor, roster=None)
        issued = command(agent_id="0", support=("1",))
        gate.apply_assignments([issued])
        self.assertEqual(executor.applied, [issued])
        gate = gate_for(executor, stage=1, roster=None)
        gate.apply_assignments([command(release_stage=0)])
        self.assertEqual(reasons(gate), ["stale_release_stage"])

    def test_without_the_roster_callback_the_gate_is_exactly_what_it_was(self) -> None:
        executor = _RecordingExecutor()
        gate = gate_for(executor, roster=("1",), with_roster=False)
        issued = command(agent_id="0", support=("1",))
        gate.apply_assignments([issued])
        self.assertEqual(executor.applied, [issued])
        self.assertEqual(gate.drain_rejections(), ())

    def test_an_ordinary_assignment_is_never_screened_as_a_release(self) -> None:
        executor = _RecordingExecutor()

        def forbidden(task_id):
            raise AssertionError("a command that releases nobody must not read any roster")

        gate = AssignmentGate(
            executor,
            clock=GlobalClock(),
            revision_of=lambda task_id: 5,
            release_state_of=forbidden,
            release_roster_of=forbidden,
        )
        issued = command(release_from=None)
        gate.apply_assignments([issued])
        self.assertEqual(executor.applied, [issued])


def belief(
    task_id: str,
    *,
    gross=(1, 0, 0, 0, 0),
    future=None,
    stage: int = 0,
    deadline: float = 100.0,
    members=(),
    remaining=None,
) -> AgentTaskBelief:
    return AgentTaskBelief(
        task_id=task_id,
        position=WorldPosition(0.0, 0.0),
        remaining_requirements=tuple(
            float(value) for value in (gross if remaining is None else remaining)
        ),
        deadline=deadline,
        stage_index=stage,
        progress=0.0,
        severity=0.5,
        future_requirements=() if future is None else tuple(int(value) for value in future),
        members=tuple(str(member) for member in members),
        updated_at=0.0,
        finished=False,
        dynamic=False,
        confidence=1.0,
        revision=1,
        stage_requirements=tuple(int(value) for value in gross),
    )


def robot(agent_id: str, *, dims, current_task: str, beliefs) -> AgentSnapshot:
    # A robot standing on a task is on that task's own believed roster.
    beliefs = tuple(
        belief_object
        if belief_object.task_id != current_task
        else replace(belief_object, members=(*belief_object.members, agent_id))
        for belief_object in beliefs
    )
    return AgentSnapshot(
        agent_id=agent_id,
        position=WorldPosition(0.1, 0.1),
        abilities=tuple(1.0 if index in dims else 0.0 for index in range(TRAITS)),
        status="working",
        current_task=current_task,
        needs_decision=False,
        known_tasks=tuple(belief_object.task_id for belief_object in beliefs) or None,
        task_beliefs=tuple(beliefs),
    )


class NextStageInformationTests(unittest.TestCase):
    """6: what the planner does when the belief carries nothing about the next stage."""

    def world(self, future) -> tuple[AgentSnapshot, ...]:
        """t-old's believed stage needs only dim 4, which b supplies: a is spare by that reading.

        Only a can fill t-new, which is what gives the release somewhere to go, so the three
        readings of the next-stage field -- absent, all-zero, and a real requirement -- differ
        in nothing but the protection itself.
        """
        old = belief("t-old", gross=(0, 0, 0, 0, 1), future=future, stage=2, members=("b",))
        new = belief("t-new", gross=(1, 0, 0, 0, 0))
        return (
            robot("a", dims=(0, 4), current_task="t-old", beliefs=(old, new)),
            robot("b", dims=(4,), current_task="t-old", beliefs=(old,)),
        )

    def advise(self, future) -> CoalitionPlannerCoordinator:
        coordinator = CoalitionPlannerCoordinator(
            renew_commitments=True, release_redundant_members=True
        )
        advice = coordinator.advise(
            CoordinatorInput(current_time=10.0, tasks=(), agents=self.world(future))
        )
        self.payload = advice
        return coordinator

    def test_a_missing_vector_still_lets_the_robot_go_and_says_so(self) -> None:
        coordinator = self.advise(None)
        self.assertEqual(sorted(self.payload.releases), ["a"])
        self.assertEqual(
            coordinator.releases_without_future_information,
            1,
            "the release carried no evidence about the following stage",
        )

    def test_a_last_stage_reads_as_no_further_requirement_not_as_missing_information(self) -> None:
        coordinator = self.advise((0, 0, 0, 0, 0))
        self.assertEqual(sorted(self.payload.releases), ["a"])
        self.assertEqual(coordinator.releases_without_future_information, 0)

    def test_the_next_stage_requirement_holds_the_robot_where_it_is(self) -> None:
        coordinator = self.advise((1, 0, 0, 0, 0))
        self.assertEqual(self.payload.releases, {})
        self.assertEqual(coordinator.releases_without_future_information, 0)

    def test_the_count_is_per_decision_and_published_with_the_advice(self) -> None:
        coordinator = self.advise(None)
        state = CoordinatorInput(current_time=11.0, tasks=(), agents=self.world((0, 0, 0, 0, 0)))
        advice = coordinator.advise(state)
        self.assertEqual(coordinator.releases_without_future_information, 0)
        self.assertEqual(
            advice.metadata["releases_without_future_information"],
            0,
            "the previous decision's count must not carry over",
        )


if __name__ == "__main__":
    unittest.main()
