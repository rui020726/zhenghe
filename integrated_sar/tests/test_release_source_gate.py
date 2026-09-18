"""The release precondition is checked where commands are validated.

A release has two ends. The gate already checked the destination -- TTL, and the
destination's task revision -- and never the task the robot was being taken off, so an
advice that judged a stage covered could execute after that stage had moved on. The advice
now carries what the planner believed about the source, and the gate compares it.

Pinned here: a refused release runs nothing and leaves the robot's current work alone; a
stage move and a revision move are named apart; an unchanged source is executed; the gate
without the callback behaves exactly as before; the destination checks keep their order and
their strength; and two releases from one source in the same batch are both refused once the
source has moved.

    D:/project_env/python.exe -m unittest integrated_sar.tests.test_release_source_gate
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
    ReleaseProvenance,
    WorldPosition,
)
from integrated_sar.coordinator import CoalitionPlannerCoordinator

TRAITS = 5


class _RecordingExecutor:
    def __init__(self) -> None:
        self.applied: list[AssignmentCommand] = []
        self.steps = 0

    def apply_assignments(self, commands) -> None:
        self.applied.extend(commands)

    def step(self, current_time: float, dt: float):
        self.steps += 1
        raise AssertionError("the gate must not step the executor while applying commands")

    def reset(self) -> None:
        self.applied.clear()


def command(
    *,
    agent_id: str = "0",
    task_id: str = "t-new",
    revision: int | None = 5,
    issued_at: float = 10.0,
    valid_until: float = 20.0,
    release_from: str | None = None,
    release_stage: int = 0,
    release_revision: int = 3,
) -> AssignmentCommand:
    metadata: dict[str, object] = {}
    if release_from is not None:
        metadata = {
            "release_from_task": release_from,
            "release_stage_index": release_stage,
            "release_revision": release_revision,
        }
    return AssignmentCommand(
        command_id=f"assignment:{agent_id}",
        agent_id=agent_id,
        task_id=task_id,
        issued_at=issued_at,
        valid_until=valid_until,
        task_revision=revision,
        metadata=metadata,
    )


def gate_for(executor, source_state, *, now=10.0):
    clock = GlobalClock()
    clock.advance_to(now)
    return AssignmentGate(
        executor,
        clock=clock,
        revision_of=lambda task_id: 5 if task_id == "t-new" else None,
        release_state_of=source_state,
    )


class SourceStageGateTests(unittest.TestCase):
    def test_a_release_whose_source_stage_moved_is_refused_and_runs_nothing(self) -> None:
        executor = _RecordingExecutor()
        seen: list[str] = []
        gate = gate_for(
            executor,
            lambda task_id: (seen.append(task_id) or (1, 4)) if task_id == "t-old" else None,
        )

        gate.apply_assignments([command(release_from="t-old", release_stage=0, release_revision=3)])

        self.assertEqual(executor.applied, [])
        self.assertEqual(seen, ["t-old"])
        rejections = gate.drain_rejections()
        self.assertEqual([row.reason for row in rejections], ["stale_release_stage"])
        self.assertEqual(rejections[0].release_from_task, "t-old")
        self.assertEqual(gate.pending_replan, {"0"}, "the robot must be asked to decide again")

    def test_a_source_revision_move_is_named_apart_from_a_stage_move(self) -> None:
        executor = _RecordingExecutor()
        gate = gate_for(executor, lambda task_id: (0, 9))
        gate.apply_assignments([command(release_from="t-old", release_stage=0, release_revision=3)])
        self.assertEqual([row.reason for row in gate.drain_rejections()], ["stale_release_revision"])

    def test_an_unchanged_source_is_executed(self) -> None:
        executor = _RecordingExecutor()
        gate = gate_for(executor, lambda task_id: (0, 3))
        issued = command(release_from="t-old", release_stage=0, release_revision=3)
        gate.apply_assignments([issued])
        self.assertEqual(executor.applied, [issued])
        self.assertEqual(gate.drain_rejections(), ())
        self.assertEqual(gate.pending_replan, set())

    def test_a_command_without_a_release_carries_no_source_check(self) -> None:
        executor = _RecordingExecutor()

        def forbidden(task_id):
            raise AssertionError("an ordinary assignment must not be screened as a release")

        gate = gate_for(executor, forbidden)
        issued = command()
        gate.apply_assignments([issued])
        self.assertEqual(executor.applied, [issued])

    def test_the_destination_checks_still_come_first(self) -> None:
        # An expired command must still read as expired, and a stale destination as stale,
        # whatever the source says: the new check may not weaken or reorder what exists.
        executor = _RecordingExecutor()
        gate = gate_for(executor, lambda task_id: (7, 7), now=10.0)
        gate.apply_assignments(
            [command(valid_until=1.0, release_from="t-old", release_stage=0, release_revision=3)]
        )
        self.assertEqual([row.reason for row in gate.drain_rejections()], ["expired_command"])

        executor = _RecordingExecutor()
        gate = gate_for(executor, lambda task_id: (0, 3))
        gate.apply_assignments([command(revision=4, release_from="t-old")])
        self.assertEqual([row.reason for row in gate.drain_rejections()], ["stale_task_revision"])

    def test_without_the_callback_the_gate_is_exactly_what_it_was(self) -> None:
        executor = _RecordingExecutor()
        gate = gate_for(executor, None)
        issued = command(release_from="t-old", release_stage=0, release_revision=3)
        gate.apply_assignments([issued])
        self.assertEqual(executor.applied, [issued])
        self.assertEqual(gate.drain_rejections(), ())

    def test_two_releases_from_one_source_in_the_same_batch_are_both_refused(self) -> None:
        # The planner's joint accounting is what makes a pair of releases safe; this check
        # cannot re-derive it. What it must guarantee is that when the source has moved,
        # neither half of the batch slips through.
        executor = _RecordingExecutor()
        gate = gate_for(executor, lambda task_id: (1, 4))
        gate.apply_assignments(
            [
                command(agent_id="0", release_from="t-old"),
                command(agent_id="1", release_from="t-old"),
            ]
        )
        self.assertEqual(executor.applied, [])
        self.assertEqual(
            sorted(row.reason for row in gate.drain_rejections()),
            ["stale_release_stage", "stale_release_stage"],
        )
        self.assertEqual(gate.pending_replan, {"0", "1"})


def belief(task_id, *, requirement, gross, stage=0, revision=1, deadline=100.0, position=(0.0, 0.0)):
    return AgentTaskBelief(
        task_id=task_id,
        position=WorldPosition(*position),
        remaining_requirements=tuple(float(v) for v in requirement),
        deadline=deadline,
        stage_index=stage,
        progress=0.0,
        severity=0.5,
        members=(),
        updated_at=0.0,
        finished=False,
        dynamic=False,
        confidence=1.0,
        revision=revision,
        stage_requirements=tuple(int(v) for v in gross),
    )


def robot(agent_id, *, dims, current_task, beliefs, roster=True):
    if roster:
        # A robot standing on a task is on that task's own believed roster.
        beliefs = tuple(
            replace(object_, members=(*object_.members, agent_id))
            if object_.task_id == current_task
            else object_
            for object_ in beliefs
        )
    return AgentSnapshot(
        agent_id=agent_id,
        position=WorldPosition(0.1, 0.1),
        abilities=tuple(1.0 if index in dims else 0.0 for index in range(TRAITS)),
        status="working" if current_task else "idle",
        current_task=current_task,
        needs_decision=current_task is None,
        known_tasks=tuple(belief_object.task_id for belief_object in beliefs) or None,
        task_beliefs=tuple(beliefs),
    )


class PublishedProvenanceTests(unittest.TestCase):
    """The advice has to say what each release left behind, as it was believed."""

    def setUp(self) -> None:
        old = belief("t-old", requirement=(0, 0, 0, 0, 0), gross=(0, 0, 0, 0, 1), stage=2, revision=7)
        new = belief("t-new", requirement=(1, 0, 0, 0, 0), gross=(1, 0, 0, 0, 0))
        # b alone covers t-old's believed stage, so a is spare -- and a is the only robot
        # that can fill t-new, which is what gives the release somewhere to go.
        self.world = (
            robot("a", dims=(0, 4), current_task="t-old", beliefs=(old, new)),
            robot("b", dims=(4,), current_task="t-old", beliefs=(old,)),
        )

    def coordinator(self):
        return CoalitionPlannerCoordinator(renew_commitments=True, release_redundant_members=True)

    def test_only_a_released_robot_gets_a_provenance_and_it_names_the_source(self) -> None:
        advice = self.coordinator().advise(
            CoordinatorInput(current_time=10.0, tasks=(), agents=self.world)
        )
        self.assertEqual(sorted(advice.releases), ["a"], "only the robot the plan moved elsewhere")
        provenance = advice.releases["a"]
        self.assertEqual(provenance.task_id, "t-old")
        self.assertEqual(
            (provenance.stage_index, provenance.revision), (2, 7),
            "the provenance must be the believed source state, not anything newer",
        )

    def test_no_release_means_no_provenance(self) -> None:
        kept = CoalitionPlannerCoordinator(renew_commitments=True, release_redundant_members=False)
        self.assertEqual(kept.advise(CoordinatorInput(10.0, (), self.world)).releases, {})

    def test_provenance_is_a_per_agent_record(self) -> None:
        # A reservation is keyed by agent so one robot's belief cannot be broadcast to
        # robots it never reached; the same must hold for the release it came from.
        advice = self.coordinator().advise(CoordinatorInput(10.0, (), self.world))
        self.assertTrue(all(isinstance(key, str) for key in advice.releases))
        self.assertTrue(all(isinstance(value, ReleaseProvenance) for value in advice.releases.values()))


if __name__ == "__main__":
    unittest.main()
