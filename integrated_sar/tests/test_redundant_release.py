"""Redundant-member release: the planner may re-decide a provably useless presence.

The dispatch-trigger audit measured the wall behind the six assembly failures: working
robots carry ``need_decision=False``, the planner plans free robots only, and the only
release from a stage nobody present can complete was the environment's random 50-120-step
working timeout. The fix is planner-side and default off: a working robot whose *own
belief* says it fills no slot its teammates do not already fill may be planned onto
another task and re-decided this tick.

Pinned here:

* the default-off world is byte-identical (no redundancy, no difference; no alternative
  task, no release);
* the release fires only for a redundant robot with a real alternative, and names it in
  ``replan_agents`` exactly once per decision;
* a robot another member needs, a finished belief, or a believed-dead task never releases;
* the redirect is stable: once re-decided, the new commitment renews with the same
  identity and no further replan fires;
* nothing about the mask, the bias band or determinism moves.
"""

from __future__ import annotations

import unittest

from integrated_sar.coordinator import CoalitionPlannerCoordinator
from integrated_sar.contracts import (
    AgentSnapshot,
    AgentTaskBelief,
    CoordinatorInput,
    WorldPosition,
)

TRAITS = 5


def abilities(*dims: int) -> tuple[float, ...]:
    vector = [0.0] * TRAITS
    for dim in dims:
        vector[dim] = 1.0
    return tuple(vector)


def belief(
    task_id: str,
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


def robot(
    agent_id: str,
    *,
    dims: tuple[int, ...],
    current_task: str | None = None,
    beliefs=(),
    status: str | None = None,
) -> AgentSnapshot:
    working = current_task is not None
    return AgentSnapshot(
        agent_id=agent_id,
        position=WorldPosition(*position_of(agent_id)),
        abilities=abilities(*dims),
        status=status or ("working" if working else "idle"),
        current_task=current_task,
        needs_decision=not working,
        known_tasks=tuple(b.task_id for b in beliefs) or None,
        task_beliefs=tuple(beliefs),
    )


def position_of(agent_id: str) -> tuple[float, float]:
    # Fixed per-id positions so ordering never depends on dict layout.
    spots = {"a": (0.1, 0.1), "b": (0.2, 0.2), "c": (0.3, 0.3)}
    return spots.get(agent_id, (0.0, 0.0))


def planner(*, release: bool, sticky: bool = True) -> CoalitionPlannerCoordinator:
    return CoalitionPlannerCoordinator(
        renew_commitments=sticky,
        release_redundant_members=release,
    )


def state(agents: tuple[AgentSnapshot, ...], now: float = 10.0) -> CoordinatorInput:
    return CoordinatorInput(current_time=now, tasks=(), agents=agents)


# The measured 20008 shape: two dim-4 robots parked on a task whose believed stage
# needs one dim-4 unit, and an open task needing the dim-0 unit only one of them has.
OLD = "t-old"
NEW = "t-new"
OLD_REQ = (0, 0, 0, 0, 1)
NEW_REQ = (1, 0, 0, 0, 0)


class DefaultOffIdentityTests(unittest.TestCase):
    def test_the_flag_defaults_off(self) -> None:
        self.assertFalse(
            CoalitionPlannerCoordinator().release_redundant_members
        )

    def test_without_redundancy_the_flag_changes_nothing(self) -> None:
        world = (
            robot("a", dims=(4,), current_task=OLD,
                  beliefs=(belief(OLD, requirement=OLD_REQ),)),
        )
        off = planner(release=False).advise(state(world))
        on = planner(release=True).advise(state(world))
        self.assertEqual(off, on)

    def test_off_and_reset_report_zero_releases(self) -> None:
        coordinator = planner(release=False)
        coordinator.advise(state(()))
        self.assertEqual(coordinator.redundant_releases, 0)
        coordinator.redundant_releases = 3
        coordinator.reset()
        self.assertEqual(coordinator.redundant_releases, 0)


class ReleaseTests(unittest.TestCase):
    def advice_for(self, coordinator: CoalitionPlannerCoordinator) -> object:
        world = (
            robot("a", dims=(0, 4), current_task=OLD,
                  beliefs=(belief(OLD, requirement=OLD_REQ),
                           belief(NEW, requirement=NEW_REQ, position=(0.9, 0.9)))),
            robot("b", dims=(4,), current_task=OLD,
                  beliefs=(belief(OLD, requirement=OLD_REQ),)),
        )
        return coordinator.advise(state(world))

    def test_a_redundant_member_with_an_alternative_is_released(self) -> None:
        # "b" already fills the dim-4 unit "a" was parked on, and "a" is the only robot
        # that can fill the new task's dim-0 unit.
        coordinator = planner(release=True)
        advice = self.advice_for(coordinator)
        self.assertIn(("a", NEW), advice.reservations)
        self.assertIn("a", advice.replan_agents)
        self.assertEqual(coordinator.redundant_releases, 1)
        # "b" is redundant on the old task too once "a"'s slot is counted, but the plan
        # has nowhere else for it: no reservation, no replan, and it stays where it is.
        self.assertNotIn(("b", OLD), advice.reservations)
        self.assertNotIn("b", advice.replan_agents)

    def test_the_same_world_with_the_flag_off_releases_nothing(self) -> None:
        coordinator = planner(release=False)
        advice = self.advice_for(coordinator)
        self.assertNotIn(("a", NEW), advice.reservations)
        self.assertEqual(advice.replan_agents, frozenset())
        self.assertEqual(coordinator.redundant_releases, 0)

    def test_a_needed_member_is_never_released(self) -> None:
        # "b" has no teammate: its dim-4 unit is the only coverage of the stage.
        world = (
            robot("b", dims=(4,), current_task=OLD,
                  beliefs=(belief(OLD, requirement=OLD_REQ),
                           belief(NEW, requirement=NEW_REQ, position=(0.9, 0.9)))),
        )
        coordinator = planner(release=True)
        advice = coordinator.advise(state(world))
        self.assertNotIn("b", advice.replan_agents)
        self.assertEqual(coordinator.redundant_releases, 0)
        self.assertIn(("b", OLD), advice.reservations)

    def test_a_redundant_member_without_an_alternative_stays_put(self) -> None:
        # "a" is redundant on the old task but holds no belief for any other task:
        # replanning it into the identical task would just re-issue its own command.
        world = (
            robot("a", dims=(0, 4), current_task=OLD,
                  beliefs=(belief(OLD, requirement=OLD_REQ),)),
            robot("b", dims=(4,), current_task=OLD,
                  beliefs=(belief(OLD, requirement=OLD_REQ),)),
        )
        coordinator = planner(release=True)
        advice = coordinator.advise(state(world))
        self.assertNotIn("a", advice.replan_agents)
        self.assertEqual(coordinator.redundant_releases, 0)

    def test_a_believed_dead_task_does_not_release(self) -> None:
        # The deadline guard keeps a stale belief from flinging a robot around; the
        # believed-finished case is the pre-existing replan path, not a release.
        world = (
            robot("a", dims=(0, 4), current_task=OLD,
                  beliefs=(belief(OLD, requirement=OLD_REQ, deadline=5.0),
                           belief(NEW, requirement=NEW_REQ, position=(0.9, 0.9)))),
            robot("b", dims=(4,), current_task=OLD,
                  beliefs=(belief(OLD, requirement=OLD_REQ),)),
        )
        coordinator = planner(release=True)
        advice = coordinator.advise(state(world))
        self.assertNotIn("a", advice.replan_agents)
        self.assertEqual(coordinator.redundant_releases, 0)

    def test_a_finished_belief_uses_the_existing_path_not_the_release(self) -> None:
        world = (
            robot("a", dims=(0, 4), current_task=OLD,
                  beliefs=(belief(OLD, requirement=OLD_REQ, finished=True),
                           belief(NEW, requirement=NEW_REQ, position=(0.9, 0.9)))),
            robot("b", dims=(4,), current_task=OLD,
                  beliefs=(belief(OLD, requirement=OLD_REQ),)),
        )
        coordinator = planner(release=True)
        advice = coordinator.advise(state(world))
        self.assertIn("a", advice.replan_agents)
        self.assertEqual(coordinator.redundant_releases, 0)


class StabilityTests(unittest.TestCase):
    def test_after_the_redirect_the_new_commitment_renews_without_a_new_replan(self) -> None:
        world_before = (
            robot("a", dims=(0, 4), current_task=OLD,
                  beliefs=(belief(OLD, requirement=OLD_REQ),
                           belief(NEW, requirement=NEW_REQ, position=(0.9, 0.9)))),
            robot("b", dims=(4,), current_task=OLD,
                  beliefs=(belief(OLD, requirement=OLD_REQ),)),
        )
        coordinator = planner(release=True)
        first = coordinator.advise(state(world_before))
        released_id = first.reservations[("a", NEW)].commitment_id

        # The command was applied: "a" now works the new task. The plan should renew
        # that commitment with the same identity and leave the robot alone.
        world_after = (
            robot("a", dims=(0, 4), current_task=NEW,
                  beliefs=(belief(OLD, requirement=OLD_REQ),
                           belief(NEW, requirement=NEW_REQ, position=(0.9, 0.9)))),
            robot("b", dims=(4,), current_task=OLD,
                  beliefs=(belief(OLD, requirement=OLD_REQ),)),
        )
        second = coordinator.advise(state(world_after))
        self.assertIn(("a", NEW), second.reservations)
        self.assertEqual(
            second.reservations[("a", NEW)].commitment_id, released_id
        )
        self.assertNotIn("a", second.replan_agents)
        self.assertEqual(coordinator.redundant_releases, 0)

    def test_two_releases_in_one_cycle_stay_deterministic(self) -> None:
        # Two redundant pairs, two alternatives: the plan must pick the same robot for
        # the same task on repeated construction, with no cross-task conflict.
        def world() -> tuple[AgentSnapshot, ...]:
            return (
                robot("a", dims=(0, 4), current_task=OLD,
                      beliefs=(belief(OLD, requirement=OLD_REQ),
                               belief(NEW, requirement=NEW_REQ, position=(0.9, 0.9)))),
                robot("b", dims=(4,), current_task=OLD,
                      beliefs=(belief(OLD, requirement=OLD_REQ),)),
                robot("c", dims=(0, 4), current_task="t-old2",
                      beliefs=(belief("t-old2", requirement=OLD_REQ),
                               belief("t-new2", requirement=NEW_REQ, position=(0.5, 0.5)))),
                robot("d", dims=(4,), current_task="t-old2",
                      beliefs=(belief("t-old2", requirement=OLD_REQ),)),
            )

        first = planner(release=True).advise(state(world()))
        second = planner(release=True).advise(state(world()))
        self.assertEqual(first, second)
        released = {agent for agent, _ in first.reservations} & {"a", "c"}
        for agent in released:
            self.assertIn(agent, first.replan_agents)


class InvariantsTests(unittest.TestCase):
    def test_biases_stay_inside_the_band_with_releases_active(self) -> None:
        world = (
            robot("a", dims=(0, 4), current_task=OLD,
                  beliefs=(belief(OLD, requirement=OLD_REQ),
                           belief(NEW, requirement=NEW_REQ, position=(0.9, 0.9)))),
            robot("b", dims=(4,), current_task=OLD,
                  beliefs=(belief(OLD, requirement=OLD_REQ),)),
        )
        advice = planner(release=True).advise(state(world))
        for value in advice.pair_bias.values():
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)

    def test_no_registry_truth_is_needed_to_decide_a_release(self) -> None:
        # The release decision is a function of beliefs and snapshots alone: the
        # CoordinatorInput built with an empty task registry must behave exactly like
        # one with tasks present.
        world = (
            robot("a", dims=(0, 4), current_task=OLD,
                  beliefs=(belief(OLD, requirement=OLD_REQ),
                           belief(NEW, requirement=NEW_REQ, position=(0.9, 0.9)))),
            robot("b", dims=(4,), current_task=OLD,
                  beliefs=(belief(OLD, requirement=OLD_REQ),)),
        )
        bare = planner(release=True).advise(CoordinatorInput(current_time=10.0, tasks=(), agents=world))
        self.assertIn(("a", NEW), bare.reservations)


class RecommandSuppressionTests(unittest.TestCase):
    """A replan-woke robot that re-picks its own task must not be re-commanded.

    Re-issuing the command restarts the environment's work clock and disarms the
    working timeout, which turns a released robot into a per-tick churn loop. The
    suppression is allocator-side and default off.
    """

    def _scenario(self, *, suppress: bool):
        import numpy as np
        import torch

        from integrated_sar.adapters.task03 import Task03Adapter
        from integrated_sar.contracts import CoordinationAdvice
        from integrated_sar.tests.test_coalition_reservation_contract import (
            _fixed_network,
            _stub_env,
        )

        # Menu: depot, t-old, t-new. The network prefers the robot's own task, so a
        # replan that woke it produces exactly the re-command the suppression targets.
        mask = np.array([False, False, False], dtype=bool)
        logits = torch.tensor([[0.1, 0.9, 0.2]])
        adapter = Task03Adapter(
            env=_stub_env(mask),
            network=_fixed_network(logits),
            deterministic=True,
            clear_initial_tasks=False,
            residual_scale=0.25,
            ctde_task_logit_offset=0.0,
            binding_commitments=True,
            suppress_recommand_own_task=suppress,
        )
        adapter.internal_to_external = {7: OLD, 8: NEW}
        working = robot("0", dims=(0, 4), current_task=OLD,
                        beliefs=(belief(OLD, requirement=OLD_REQ),))
        advice = CoordinationAdvice(
            revision=1,
            generated_at=0.0,
            valid_until=10.0,
            replan_agents=frozenset({"0"}),
        )
        return adapter.allocate((), (working,), advice, 3.0)

    def test_without_suppression_a_replan_recommand_restarts_the_robot(self) -> None:
        commands = self._scenario(suppress=False)
        self.assertEqual([c.task_id for c in commands], [OLD])

    def test_with_suppression_the_robot_keeps_working_without_a_command(self) -> None:
        commands = self._scenario(suppress=True)
        self.assertEqual(commands, ())


if __name__ == "__main__":
    unittest.main()
