"""Commitment continuity: what the allocator may do to a commitment it is holding.

The learnability gate attributed 12 of 14 residual expiries to execution breaks -- a
commitment the plan held and the allocator quietly overrode. This file pins the fix:

* a commitment that survives the validity conditions is re-commanded, not re-decided, and
  a robot already on it produces no second command;
* every condition that can release one does, with a named reason, so a release is never
  mistaken for an override;
* with the option off, or with no reservations in play, nothing moves.
"""

from __future__ import annotations

import importlib.util
import unittest
from typing import Any

from integrated_sar.contracts import (
    AgentSnapshot,
    AgentTaskBelief,
    CoalitionReservation,
    CoordinationAdvice,
    WorldPosition,
)
from integrated_sar.tests.test_coalition_reservation_contract import (
    ACTION_MAP,
    _make_adapter,
    reservation,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
TASK = "target-a"
OTHER = "target-b"


def belief(
    task_id: str = TASK,
    *,
    requirements: tuple[int, ...] = (1, 0, 0, 0, 0),
    deadline: float = 1000.0,
    revision: int = 3,
    finished: bool = False,
    position: tuple[float, float] = (0.5, 0.0),
) -> AgentTaskBelief:
    return AgentTaskBelief(
        task_id=task_id,
        position=WorldPosition(*position),
        remaining_requirements=requirements,
        deadline=deadline,
        stage_index=0,
        progress=0.0,
        severity=0.5,
        finished=finished,
        revision=revision,
    )


def robot(
    *,
    current_task: str | None = None,
    status: str = "idle",
    beliefs: tuple[AgentTaskBelief, ...] = (),
) -> AgentSnapshot:
    return AgentSnapshot(
        agent_id="0",
        position=WorldPosition(0.4, 0.0),
        abilities=(1.0, 0.0, 0.0, 0.0, 0.0),
        status=status,
        current_task=current_task,
        known_tasks=tuple(b.task_id for b in beliefs) or (TASK,),
        task_beliefs=beliefs,
    )


def advice(
    commitments: dict[tuple[str, str], CoalitionReservation] | None = None,
    *,
    replan: bool = False,
) -> CoordinationAdvice:
    return CoordinationAdvice(
        revision=7,
        generated_at=0.0,
        valid_until=100.0,
        reservations=commitments or {},
        replan_agents=frozenset({"0"}) if replan else frozenset(),
    )


def binding_adapter(mask: Any, *, binding: bool = True) -> Any:
    import numpy as np
    import torch

    return _make_adapter(
        mask=mask,
        logits=torch.tensor([[0.50, 0.10, 0.05]]),
        residual_scale=0.5,
        binding_commitments=binding,
    )


def held_commitment_for(agent_id: str = "0", task_id: str = TASK, **kwargs: Any) -> dict[tuple[str, str], CoalitionReservation]:
    return {(agent_id, task_id): reservation(agent_id=agent_id, task_id=task_id, **kwargs)}


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class ContinuityTests(unittest.TestCase):
    """A commitment that still stands is re-commanded, not re-decided."""

    def _mask(self) -> Any:
        import numpy as np

        return np.array([False, False, True], dtype=bool)

    def _traced(self, mask: Any, robots: Any, adv: Any):
        """Allocate once with a real trace attached, and return commands plus rows.

        The commitment fields are computed on the way into the record, so asserting on a
        hand-built :class:`DecisionFacts` would test nothing; reading them back from the
        writer is the same path a run takes.
        """
        import io
        import json

        from integrated_sar.allocation_trace import DecisionTraceWriter

        sink = io.StringIO()
        adapter = binding_adapter(mask)
        adapter.decision_trace = DecisionTraceWriter(
            sink, meta={"arm": "coalplan0.25sticky", "seed": 1}
        )
        commands = adapter.allocate((), robots, adv, 3.0)
        adapter.decision_trace.close()
        rows = [
            json.loads(line)
            for line in sink.getvalue().splitlines()
            if line.strip()
            and json.loads(line).get("event_type") == "decision"
        ]
        return commands, rows

    def test_repeated_decisions_do_not_break_a_valid_commitment(self) -> None:
        mask = self._mask()
        robots = (robot(beliefs=(belief(),)),)
        held = advice(held_commitment_for())
        for _ in range(3):
            commands, rows = self._traced(mask, robots, held)
            self.assertEqual([command.task_id for command in commands], [TASK])
        row = rows[0]
        self.assertTrue(row["commitment_present"])
        self.assertTrue(row["commitment_held"])
        self.assertIsNone(row["commitment_release_reason"])
        self.assertEqual(row["commitment_task"], TASK)
        self.assertEqual(row["commitment_slot_dimension"], 0)
        self.assertEqual(row["commitment_belief_status"], "active")
        self.assertEqual(row["commitment_belief_revision"], 3)
        self.assertTrue(row["commitment_mask_legal"])
        # Three cycles, three commands, all for the committed task. The allocator's own
        # verdict wanted the depot, so the commitment *did* override it -- and that is the
        # override the acceptance criterion counts as prevented, not unjustified: the
        # command that went out is the committed one.
        self.assertTrue(row["commitment_overridden"])
        self.assertFalse(row["commitment_suppressed_command"])
        self.assertEqual(row["chosen_task"], TASK)

    def test_an_explicit_replan_releases_the_commitment(self) -> None:
        mask = self._mask()
        adapter = binding_adapter(mask)
        binding = adapter._evaluate_commitment(
            robot(beliefs=(belief(),)),
            reservation(),
            action_index=1,
            mask=mask,
            current_time=3.0,
            replanned=True,
        )
        self.assertFalse(binding.held)
        self.assertEqual(binding.reason, "replan")

    def test_a_finished_belief_releases_the_commitment(self) -> None:
        mask = self._mask()
        adapter = binding_adapter(mask)
        binding = adapter._evaluate_commitment(
            robot(beliefs=(belief(finished=True),)),
            reservation(),
            action_index=1,
            mask=mask,
            current_time=3.0,
            replanned=False,
        )
        self.assertFalse(binding.held)
        self.assertEqual(binding.reason, "terminal_belief")

    def test_an_expired_commitment_releases(self) -> None:
        mask = self._mask()
        adapter = binding_adapter(mask)
        binding = adapter._evaluate_commitment(
            robot(beliefs=(belief(),)),
            reservation(valid_until=2.0),
            action_index=1,
            mask=mask,
            current_time=3.0,
            replanned=False,
        )
        self.assertFalse(binding.held)
        self.assertEqual(binding.reason, "expired")

    def test_a_masked_task_releases_the_commitment(self) -> None:
        import numpy as np

        mask = np.array([False, True, True], dtype=bool)
        adapter = binding_adapter(mask)
        binding = adapter._evaluate_commitment(
            robot(beliefs=(belief(),)),
            reservation(),
            action_index=1,
            mask=mask,
            current_time=3.0,
            replanned=False,
        )
        self.assertFalse(binding.held)
        self.assertEqual(binding.reason, "hard_mask")

    def test_a_task_off_the_menu_releases_the_commitment(self) -> None:
        mask = self._mask()
        adapter = binding_adapter(mask)
        binding = adapter._evaluate_commitment(
            robot(beliefs=(belief(),)),
            reservation(),
            action_index=None,
            mask=mask,
            current_time=3.0,
            replanned=False,
        )
        self.assertFalse(binding.held)
        self.assertEqual(binding.reason, "not_on_menu")

    def test_a_fulfilled_slot_releases_the_commitment(self) -> None:
        mask = self._mask()
        adapter = binding_adapter(mask)
        for requirement in ((0, 0, 0, 0, 0), (0, 1, 0, 0, 0)):
            with self.subTest(requirement=requirement):
                binding = adapter._evaluate_commitment(
                    robot(beliefs=(belief(requirements=requirement),)),
                    reservation(slot_dimension=0),
                    action_index=1,
                    mask=mask,
                    current_time=3.0,
                    replanned=False,
                )
                self.assertFalse(binding.held)
                self.assertEqual(binding.reason, "slot_filled")

    def test_a_robot_working_elsewhere_is_not_yanked(self) -> None:
        mask = self._mask()
        adapter = binding_adapter(mask)
        binding = adapter._evaluate_commitment(
            robot(
                current_task=OTHER,
                status="working",
                beliefs=(belief(), belief(OTHER, finished=False)),
            ),
            reservation(),
            action_index=1,
            mask=mask,
            current_time=3.0,
            replanned=False,
        )
        self.assertFalse(binding.held)
        self.assertEqual(binding.reason, "not_executable")

    def test_no_second_command_while_already_on_the_committed_task(self) -> None:
        mask = self._mask()
        robots = (robot(current_task=TASK, status="moving", beliefs=(belief(),)),)
        commands, rows = self._traced(mask, robots, advice(held_commitment_for()))
        # A robot already executing its commitment gets no command at all: re-issuing one
        # would restart work it is part way through.
        self.assertEqual(commands, ())
        self.assertTrue(rows[0]["commitment_suppressed_command"])
        self.assertTrue(rows[0]["commitment_held"])

    def test_complementary_robots_are_committed_to_distinct_slots(self) -> None:
        import numpy as np
        import torch

        mask = np.array([False, False, False], dtype=bool)
        adapter = binding_adapter(mask)
        first = AgentSnapshot(
            agent_id="0",
            position=WorldPosition(0.4, 0.0),
            abilities=(1.0, 0.0, 0.0, 0.0, 0.0),
            status="idle",
            known_tasks=(TASK,),
            task_beliefs=(belief(requirements=(1, 1, 0, 0, 0)),),
        )
        second = AgentSnapshot(
            agent_id="1",
            position=WorldPosition(0.4, 0.0),
            abilities=(0.0, 1.0, 0.0, 0.0, 0.0),
            status="idle",
            known_tasks=(TASK,),
            task_beliefs=(belief(requirements=(1, 1, 0, 0, 0)),),
        )
        commitments = {
            ("0", TASK): reservation(agent_id="0", task_id=TASK, slot_dimension=0),
            ("1", TASK): reservation(agent_id="1", task_id=TASK, slot_dimension=1),
        }
        commands = adapter.allocate((), (first, second), advice(commitments=commitments), 3.0)
        self.assertEqual(sorted(command.task_id for command in commands), [TASK, TASK])
        self.assertEqual(
            [command.agent_id for command in commands], ["0", "1"]
        )

    def test_a_command_past_its_ttl_is_not_applied(self) -> None:
        from integrated_sar.contracts import AssignmentCommand

        mask = self._mask()
        adapter = binding_adapter(mask)
        adapter.env.current_time = 50.0
        stale = AssignmentCommand(
            command_id="assignment:1",
            agent_id="0",
            task_id=TASK,
            issued_at=0.0,
            valid_until=10.0,
            task_revision=None,
        )
        adapter.apply_assignments((stale,))
        # The robot never moved onto it: the depot is where an expired command leaves it.
        self.assertEqual(adapter.env.agent_dic[0]["current_task"], -1)


class DefaultOffTests(unittest.TestCase):
    """The option is off unless it is asked for, and off means nothing moves."""

    def test_the_default_configuration_does_not_bind(self) -> None:
        from integrated_sar.joint_driver import RunConfiguration, _build_coordinator

        configuration = RunConfiguration()
        self.assertFalse(configuration.sticky_commitments)
        planner = _build_coordinator(configuration)
        self.assertFalse(getattr(planner, "renew_commitments", False))

    def test_binding_requires_a_coordinator_that_issues_reservations(self) -> None:
        from integrated_sar.errors import ConfigurationError
        from integrated_sar.joint_driver import RunConfiguration

        # Validation is explicit, so a bad combination is refused at the gate and not
        # silently tolerated by the dataclass.
        with self.assertRaises(ConfigurationError):
            RunConfiguration(sticky_commitments=True, use_coordinator=False).validate()
        with self.assertRaises(ConfigurationError):
            RunConfiguration(sticky_commitments=True).validate()
        RunConfiguration(
            sticky_commitments=True, coordinator_policy="coalition_planner"
        ).validate()

    def test_the_label_marks_a_binding_arm(self) -> None:
        from integrated_sar.allocation_trace import arm_name

        self.assertEqual(
            arm_name(
                use_coordinator=True,
                residual_scale=0.25,
                coordinator_policy="coalition_planner",
                binding_commitments=True,
            ),
            "coalplan0.25sticky",
        )
        self.assertEqual(
            arm_name(
                use_coordinator=True,
                residual_scale=0.25,
                coordinator_policy="coalition_planner",
            ),
            "coalplan0.25",
        )

    def test_an_advisory_adapter_ignores_the_commitment(self) -> None:
        mask = binding_adapter.__wrapped__ if False else None  # pragma: no cover
        import numpy as np
        import torch

        adapter = _make_adapter(
            mask=np.array([False, False, True], dtype=bool),
            logits=torch.tensor([[0.50, 0.10, 0.05]]),
            residual_scale=0.5,
            binding_commitments=False,
        )
        robots = (robot(beliefs=(belief(),)),)
        plain = adapter.allocate((), robots, advice(), 3.0)
        with_commitment = adapter.allocate(
            (), robots, advice(held_commitment_for()), 3.0
        )
        self.assertEqual(
            [command.task_id for command in plain],
            [command.task_id for command in with_commitment],
        )

    def test_renewal_is_off_unless_asked_for(self) -> None:
        from integrated_sar.coordinator import CoalitionPlannerCoordinator
        from integrated_sar.tests.test_coalition_planner import state

        robots = (
            __import__(
                "integrated_sar.tests.test_coalition_planner", fromlist=["agent"]
            ).agent(
                "0",
                (1.0, 0.0, 0.0, 0.0, 0.0),
                current_task=TASK,
                status="moving",
                beliefs=(belief(),),
            ),
        )
        advisory = CoalitionPlannerCoordinator().advise(state(*robots))
        self.assertEqual(advisory.reservations, {})
        sticky = CoalitionPlannerCoordinator(renew_commitments=True).advise(state(*robots))
        self.assertIn(("0", TASK), sticky.reservations)


class ResetTests(unittest.TestCase):
    """No commitment survives an episode boundary."""

    def test_reset_clears_renewed_commitments(self) -> None:
        from integrated_sar.coordinator import CoalitionPlannerCoordinator
        from integrated_sar.tests.test_coalition_planner import agent as make_agent
        from integrated_sar.tests.test_coalition_planner import state

        planner = CoalitionPlannerCoordinator(renew_commitments=True)
        busy = make_agent(
            "0",
            (1.0, 0.0, 0.0, 0.0, 0.0),
            current_task=TASK,
            status="moving",
            beliefs=(belief(),),
        )
        held = planner.advise(state(busy))
        self.assertIn(("0", TASK), held.reservations)
        planner.reset()
        # A robot still under orders is committed again -- that is the point of renewal --
        # but nothing from the previous episode is carried in: numbering restarted and the
        # store was empty before this cycle.
        after = planner.advise(state(busy))
        self.assertEqual(after.revision, 1)
        self.assertEqual(after.metadata["expired_reservations"], 0)
        self.assertEqual(after.metadata["cancelled_reservations"], 0)
        # And with nothing to renew at all, a reset episode holds nothing.
        self.assertEqual(planner.advise(state()).reservations, {})


if __name__ == "__main__":
    unittest.main()
