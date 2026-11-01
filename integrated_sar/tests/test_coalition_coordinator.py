"""Unit tests for the coalition-aware residual: score the gap, not the overlap.

The policy exists to answer one question the heuristic coordinator cannot: with a
stage requirement needing three units of ability and two already committed, does
the residual prefer the robot that supplies the missing unit, or does it keep
rewarding robots for ability the team already has? Each test below pins one of the
properties that decision depends on, plus the two safety properties the whole
architecture rests on: the residual is bounded, and it can never re-legalise a
hard-masked action.
"""

from __future__ import annotations

import importlib.util
import unittest

from integrated_sar.contracts import (
    AgentSnapshot,
    AgentTaskBelief,
    CoordinatorInput,
    TaskSnapshot,
    TaskStageSpec,
    TaskStatus,
    WorldPosition,
)
from integrated_sar.coordinator import CoalitionAwareCoordinator, HeuristicCoordinator

HAS_TORCH = importlib.util.find_spec("torch") is not None

TASK = "target-a"


def belief(
    task_id: str = TASK,
    *,
    requirements: tuple[int, ...] = (3, 0, 0, 0, 0),
    deadline: float = 100.0,
    progress: float = 0.0,
    severity: float = 0.5,
    position: tuple[float, float] = (0.5, 0.5),
    finished: bool = False,
    confidence: float = 0.5,
    revision: int = 0,
) -> AgentTaskBelief:
    return AgentTaskBelief(
        task_id=task_id,
        position=WorldPosition(*position),
        remaining_requirements=requirements,
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
    *,
    abilities: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0),
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
        position=WorldPosition(0.5, 0.5),
        stages=(TaskStageSpec(requirements, 3.0),),
        current_stage=0,
        remaining_requirements=requirements,
        progress=0.0,
        status=TaskStatus.DETECTED,
        deadline=100.0,
        confidence=1.0,
        severity=0.5,
        discovered_at=0.0,
        updated_at=0.0,
        revision=1,
    )


def state(*agents: AgentSnapshot, current_time: float = 0.0, truth: bool = False) -> CoordinatorInput:
    return CoordinatorInput(
        current_time=current_time,
        tasks=(registry_truth(),) if truth else (),
        agents=agents,
    )


class CoalitionGapTests(unittest.TestCase):
    """Who gets convened, and who does not."""

    def setUp(self) -> None:
        self.coordinator = CoalitionAwareCoordinator()

    def _pair(self, advice, agent_id: str, task_id: str = TASK) -> float:
        return advice.pair_bias[(agent_id, task_id)]

    def test_the_missing_ability_is_the_one_that_gets_convened(self) -> None:
        # Two of the three units the stage needs are already committed, so the gap
        # is exactly the single unit robot A can supply and robot B cannot.
        advice = self.coordinator.advise(
            state(
                agent("0", abilities=(1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(),)),
                agent("1", abilities=(0.0, 1.0, 0.0, 0.0, 0.0), beliefs=(belief(),)),
                agent("2", abilities=(2.0, 0.0, 0.0, 0.0, 0.0), current_task=TASK),
            )
        )
        self.assertGreater(self._pair(advice, "0"), 0.0)
        # The robot that fills nothing must not be invited: no positive part of the
        # residual can be earned by repeating a capability the task already has.
        self.assertLessEqual(self._pair(advice, "1"), 0.0)
        self.assertGreater(self._pair(advice, "0"), self._pair(advice, "1"))

    def test_the_specialist_outranks_the_partial_filler(self) -> None:
        # Requirement 4, one unit committed: the gap is 3. A robot holding all three
        # units covers the whole gap; three robots holding one unit each cover a
        # third. The specialist is the one the group is missing.
        advice = self.coordinator.advise(
            state(
                agent("specialist", abilities=(3.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(requirements=(4, 0, 0, 0, 0)),)),
                agent("single", abilities=(1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(requirements=(4, 0, 0, 0, 0)),)),
                agent("committed", abilities=(1.0, 0.0, 0.0, 0.0, 0.0), current_task=TASK),
            )
        )
        specialist = self._pair(advice, "specialist")
        single = self._pair(advice, "single")
        self.assertAlmostEqual(specialist, 0.65, places=6)
        self.assertAlmostEqual(single, 0.65 / 3.0, places=6)
        self.assertGreater(specialist, single)

    def test_an_already_committed_robot_is_not_counted_as_a_filler_twice(self) -> None:
        # Robot 1 is committed and holds half of what the stage needs. Its own unit
        # is already inside the committed sum, so the gap it may be credited with is
        # what is left beyond that unit -- never the unit itself.
        advice = self.coordinator.advise(
            state(
                agent(
                    "1",
                    abilities=(1.0, 0.0, 0.0, 0.0, 0.0),
                    current_task=TASK,
                    beliefs=(belief(requirements=(2, 0, 0, 0, 0)),),
                ),
                agent("2", abilities=(1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(requirements=(2, 0, 0, 0, 0)),)),
            )
        )
        # Gap for robot 1 is 2 (its own unit excluded) and it fills 1 of it; robot 2
        # faces the unit robot 1 already carries and fills that one completely.
        self.assertAlmostEqual(self._pair(advice, "1"), 0.65 * 0.5, places=6)
        self.assertAlmostEqual(self._pair(advice, "2"), 0.65, places=6)

    def test_a_task_with_no_gap_left_convenes_nobody(self) -> None:
        advice = self.coordinator.advise(
            state(
                agent("0", abilities=(1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(requirements=(2, 0, 0, 0, 0)),)),
                agent("1", abilities=(1.0, 0.0, 0.0, 0.0, 0.0), current_task=TASK),
                agent("2", abilities=(1.0, 0.0, 0.0, 0.0, 0.0), current_task=TASK),
            )
        )
        self.assertLessEqual(self._pair(advice, "0"), 0.0)

    def test_a_finished_belief_is_not_scored_at_all(self) -> None:
        advice = self.coordinator.advise(
            state(agent("0", beliefs=(belief(finished=True),)))
        )
        self.assertNotIn(("0", TASK), advice.pair_bias)
        self.assertNotIn(("0", TASK), advice.agent_task_bias)

    def test_a_task_the_robot_never_heard_of_is_not_scored(self) -> None:
        advice = self.coordinator.advise(
            state(agent("0", beliefs=(belief(),), known=()))
        )
        self.assertNotIn(("0", TASK), advice.pair_bias)


class CoalitionBeliefBoundaryTests(unittest.TestCase):
    """Invariant 9: beliefs do not cross robots, and registry truth is never read."""

    def setUp(self) -> None:
        self.coordinator = CoalitionAwareCoordinator()

    def test_one_robots_belief_cannot_change_another_robots_bias(self) -> None:
        def run(other: tuple[int, ...]) -> object:
            return self.coordinator.advise(
                state(
                    agent("a", beliefs=(belief(requirements=other),)),
                    agent("b", beliefs=(belief(requirements=(2, 0, 0, 0, 0)),)),
                    agent("c", abilities=(1.0, 0.0, 0.0, 0.0, 0.0), current_task=TASK),
                )
            )

        # Robot a believes there is nothing left to do; robot b believes there is.
        empty = run((0, 0, 0, 0, 0))
        full = run((2, 0, 0, 0, 0))
        for key in (("b", TASK),):
            self.assertEqual(empty.pair_bias[key], full.pair_bias[key])
            self.assertEqual(empty.agent_task_bias[key], full.agent_task_bias[key])

    def test_a_stale_belief_produces_a_stale_decision(self) -> None:
        stale = belief(requirements=(2, 0, 0, 0, 0), deadline=40.0, progress=0.0)
        # The registry has moved on -- the stage it is describing no longer needs
        # anything -- but that is not knowledge this robot has received.
        fresh_truth = registry_truth(requirements=(0, 0, 0, 0, 0))
        advice = self.coordinator.advise(
            CoordinatorInput(
                current_time=0.0,
                tasks=(fresh_truth,),
                agents=(
                    agent("0", beliefs=(stale,)),
                    agent("1", abilities=(1.0, 0.0, 0.0, 0.0, 0.0), current_task=TASK),
                ),
            )
        )
        # The gap the robot can see is still open, so it is still convened: acting on
        # newer truth than the robot holds would be a belief leak, not a repair.
        self.assertGreater(advice.pair_bias[("0", TASK)], 0.0)
        without_truth = self.coordinator.advise(
            state(
                agent("0", beliefs=(stale,)),
                agent("1", abilities=(1.0, 0.0, 0.0, 0.0, 0.0), current_task=TASK),
            )
        )
        self.assertEqual(advice.pair_bias, without_truth.pair_bias)
        self.assertEqual(advice.agent_task_bias, without_truth.agent_task_bias)

    def test_the_urgency_term_comes_from_the_belief_the_robot_holds(self) -> None:
        soon = self.coordinator.advise(
            state(agent("0", beliefs=(belief(deadline=10.0),)), current_time=0.0)
        )
        late = self.coordinator.advise(
            state(agent("0", beliefs=(belief(deadline=1000.0),)), current_time=0.0)
        )
        self.assertGreater(
            soon.agent_task_bias[("0", TASK)], late.agent_task_bias[("0", TASK)]
        )

    def test_a_busy_robot_that_is_not_committed_here_is_pushed_away(self) -> None:
        advice = self.coordinator.advise(
            state(
                agent(
                    "0",
                    abilities=(3.0, 0.0, 0.0, 0.0, 0.0),
                    status="moving",
                    current_task="other",
                    beliefs=(belief(requirements=(3, 0, 0, 0, 0)),),
                )
            )
        )
        # Full gap filled (0.65) minus the busy penalty, so still positive -- but a
        # robot idle elsewhere would score strictly higher.
        idle = self.coordinator.advise(
            state(agent("0", abilities=(3.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(),)))
        )
        self.assertLess(
            advice.pair_bias[("0", TASK)], idle.pair_bias[("0", TASK)]
        )


class CoalitionBoundedOutputTests(unittest.TestCase):
    def test_every_bias_stays_inside_the_unit_interval(self) -> None:
        # Absurd weights, so the clamp is what keeps the residual bounded rather than
        # the weights happening to be small.
        coordinator = CoalitionAwareCoordinator(
            urgency_weight=10.0,
            progress_weight=10.0,
            severity_weight=10.0,
            confidence_weight=10.0,
            fill_weight=10.0,
            distance_weight=10.0,
            busy_penalty=10.0,
        )
        advice = coordinator.advise(
            state(
                agent("0", abilities=(3.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(deadline=0.0, severity=1.0),)),
                agent("1", abilities=(3.0, 0.0, 0.0, 0.0, 0.0), status="moving", current_task="other", beliefs=(belief(deadline=0.0, severity=1.0),)),
            )
        )
        for table in (advice.pair_bias, advice.agent_task_bias):
            for key, value in table.items():
                self.assertGreaterEqual(value, -1.0, key)
                self.assertLessEqual(value, 1.0, key)
        self.assertEqual(advice.pair_bias[("0", TASK)], 1.0)

    def test_revisions_advance_and_reset(self) -> None:
        coordinator = CoalitionAwareCoordinator()
        first = coordinator.advise(state(agent("0", beliefs=(belief(),))))
        second = coordinator.advise(state(agent("0", beliefs=(belief(),))))
        self.assertEqual((first.revision, second.revision), (1, 2))
        coordinator.reset()
        self.assertEqual(coordinator.advise(state(agent("0", beliefs=(belief(),)))).revision, 1)


class CoalitionInterfaceTests(unittest.TestCase):
    """The new policy must remain a drop-in for the heuristic one."""

    def test_both_policies_publish_the_same_task_scoped_keys(self) -> None:
        coordinator = CoalitionAwareCoordinator()
        heuristic = HeuristicCoordinator()
        world = state(
            agent("0", abilities=(1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(),)),
            agent("1", abilities=(0.0, 0.0, 1.0, 0.0, 0.0), beliefs=(belief(),)),
        )
        coalition_advice = coordinator.advise(world)
        heuristic_advice = heuristic.advise(world)
        self.assertEqual(
            set(coalition_advice.pair_bias), set(heuristic_advice.pair_bias)
        )
        self.assertEqual(
            set(coalition_advice.agent_task_bias), set(heuristic_advice.agent_task_bias)
        )
        self.assertEqual(coalition_advice.task_bias, {})
        self.assertEqual(
            coalition_advice.metadata["policy"], "coalition-aware-v1-belief-only"
        )

    def test_the_coalition_term_retargets_the_residual(self) -> None:
        # One of the three needed units is already committed, so the one-unit robot
        # covers the entire open gap: the coalition policy must score it strictly
        # higher than the heuristic, which only ever sees one third of a requirement.
        with_gap = state(
            agent("0", abilities=(1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(),)),
            agent("1", abilities=(1.0, 0.0, 0.0, 0.0, 0.0), current_task=TASK),
        )
        self.assertGreater(
            CoalitionAwareCoordinator().advise(with_gap).pair_bias[("0", TASK)],
            HeuristicCoordinator().advise(with_gap).pair_bias[("0", TASK)],
        )

        # With the requirement already met, the same robot fills nothing, while the
        # heuristic still pays it a third of the weight for ability nobody needs.
        covered = state(
            agent("0", abilities=(1.0, 0.0, 0.0, 0.0, 0.0), beliefs=(belief(),)),
            agent("1", abilities=(2.0, 0.0, 0.0, 0.0, 0.0), current_task=TASK),
            agent("2", abilities=(1.0, 0.0, 0.0, 0.0, 0.0), current_task=TASK),
        )
        self.assertEqual(CoalitionAwareCoordinator().advise(covered).pair_bias[("0", TASK)], 0.0)
        self.assertGreater(
            HeuristicCoordinator().advise(covered).pair_bias[("0", TASK)], 0.0
        )


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class CoalitionHardMaskTests(unittest.TestCase):
    """The residual is advice: it can reorder the legal actions, never legalise one."""

    def test_a_masked_task_is_never_re_legalised_by_a_coalition_bias(self) -> None:
        import numpy as np
        import torch

        from integrated_sar.adapters.task03 import MASKED_LOGIT
        from integrated_sar.tests.test_ctde_task_logit_offset import (
            ACTION_MAP,
            _fixed_network,
            _make_adapter,
        )

        # Both tasks are known and both would earn a bias; the environment only
        # allows the first of them.
        mask = np.array([False, False, True], dtype=bool)
        logits = torch.tensor([[0.30, 0.05, 0.05]])
        advice = CoalitionAwareCoordinator().advise(
            state(
                agent(
                    "0",
                    abilities=(3.0, 0.0, 0.0, 0.0, 0.0),
                    beliefs=(belief("target-a"), belief("target-b")),
                )
            )
        )
        self.assertGreater(advice.bias_for("0", "target-b"), 0.0)

        adapter = _make_adapter(offset=0.0, mask=mask, logits=logits)
        # A residual of the full size the CLI allows, so it is the mask and not the
        # weight that has to keep the illegal task out.
        adapter.residual_scale = 1.0
        verdict = adapter._network_verdict(0, list(ACTION_MAP), advice)
        self.assertIsNone(verdict.final[2])
        self.assertNotEqual(verdict.index, 2)
        self.assertAlmostEqual(verdict.final[0], 0.30)


if __name__ == "__main__":
    unittest.main()