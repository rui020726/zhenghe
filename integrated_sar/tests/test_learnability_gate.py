"""The learnability gate: attribution, the same-information oracle, and the split contract.

Three things are pinned here, because each one is easy to get wrong in a way that would
make the gate say yes when it should say no:

* the oracle may only *reorder the rule planner's own output* -- it starts from that plan
  and returns commitments, so if it ever invents a candidate the rule planner could not
  have produced, the comparison stops measuring ordering;
* the six failure classes are assigned by a deterministic procedure, so a class means the
  same thing on every split;
* the split registry is the only place a seed's role is written down, and the hindsight
  ceiling stays out of every module the driver can reach.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from integrated_sar.coordinator import (
    CoalitionPlannerCoordinator,
    DeadlineAwareCoalitionCoordinator,
)
from integrated_sar.data_splits import (
    RETIRED_NAMES,
    SPLITS,
    all_seeds,
    split_of,
)
from integrated_sar.learnability import FAILURE_CLASSES, _classify
from integrated_sar.tests.test_coalition_planner import belief, state  # noqa: F401

TASK = "target-a"
OTHER = "target-b"


def _agent(agent_id: str, *, position: tuple[float, float], abilities: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0), **kwargs: Any):
    from integrated_sar.tests.test_coalition_planner import agent as make_agent

    return make_agent(agent_id, abilities, position=position, **kwargs)


class OracleOrderingTests(unittest.TestCase):
    """The oracle is the rule plan, pruned and re-planned -- nothing else."""

    def setUp(self) -> None:
        self.planner = CoalitionPlannerCoordinator()
        self.oracle = DeadlineAwareCoalitionCoordinator()

    def test_a_comfortable_plan_is_reproduced_exactly(self) -> None:
        # Both tasks are reachable well inside their believed budgets, so the oracle has
        # nothing to return: it must hand back the rule planner's commitments unchanged.
        robots = (
            _agent("0", position=(0.5, 0.5), beliefs=(belief(TASK, requirements=(1, 0, 0, 0, 0), deadline=1000.0),)),
            _agent("1", position=(0.5, 0.5), beliefs=(belief(OTHER, requirements=(1, 0, 0, 0, 0), deadline=1000.0),)),
        )
        planner = self.planner.advise(state(*robots))
        oracle = self.oracle.advise(state(*robots))
        self.assertEqual(set(oracle.reservations), set(planner.reservations))
        for key, value in planner.reservations.items():
            self.assertEqual(oracle.reservations[key].agent_id, value.agent_id)
            self.assertEqual(oracle.reservations[key].task_id, value.task_id)
            self.assertEqual(oracle.reservations[key].ability_slot, value.ability_slot)

    def test_a_task_that_cannot_be_reached_in_time_is_returned_and_replanned(self) -> None:
        # The planner sends the only free robot to the *urgent* task, which stands far
        # away and has almost no budget left; the oracle hands that commitment back and
        # sends the same robot to the task that can still be saved.
        robots = (
            _agent(
                "0",
                position=(1.0, 1.0),
                beliefs=(
                    belief(TASK, requirements=(1, 0, 0, 0, 0), deadline=3.0, position=(0.0, 0.0)),
                    belief(OTHER, requirements=(1, 0, 0, 0, 0), deadline=1000.0, position=(0.9, 0.9)),
                ),
            ),
        )
        planner = self.planner.advise(state(robots[0]))
        oracle = self.oracle.advise(state(robots[0]))
        # Travel to the urgent task is 1.41 / 0.2 = 7.07 steps against a budget of 3.
        self.assertIn(("0", TASK), planner.reservations)
        self.assertNotIn(("0", TASK), oracle.reservations)
        self.assertIn(("0", OTHER), oracle.reservations)
        self.assertEqual(
            oracle.metadata["policy"], "coalition-planner-v1-belief-only"
        )
        self.assertGreater(
            self.oracle.last_plan_notes["tasks_returned_on_believed_budget"], 0
        )
        self.assertGreater(self.oracle.last_plan_notes["robots_replanned"], 0)

    def test_the_oracle_never_adds_a_candidate_the_planner_did_not_have(self) -> None:
        # Every (robot, task) the oracle commits to must be one the rule planner also
        # considered: the oracle reorders and returns, it does not invent.
        robots = (
            _agent(
                "0",
                position=(0.2, 0.2),
                beliefs=(
                    belief(TASK, requirements=(2, 0, 0, 0, 0), deadline=40.0, position=(0.9, 0.9)),
                    belief(OTHER, requirements=(1, 0, 0, 0, 0), deadline=60.0, position=(0.1, 0.1)),
                ),
            ),
            _agent(
                "1",
                position=(0.8, 0.8),
                beliefs=(
                    belief(TASK, requirements=(2, 0, 0, 0, 0), deadline=40.0, position=(0.9, 0.9)),
                    belief(OTHER, requirements=(1, 0, 0, 0, 0), deadline=60.0, position=(0.1, 0.1)),
                ),
            ),
        )
        planner_keys = set(self.planner.advise(state(*robots)).reservations)
        oracle_keys = set(self.oracle.advise(state(*robots)).reservations)
        # Both policies draw from the same candidate set, so any commitment the oracle
        # holds is one the planner could have held.
        self.assertTrue(oracle_keys <= planner_keys | {(a, t) for a, t in planner_keys})

    def test_registry_truth_cannot_change_the_oracle(self) -> None:
        from integrated_sar.tests.test_coalition_planner import registry_truth

        robots = (
            _agent("0", position=(0.5, 0.5), beliefs=(belief(TASK, requirements=(1, 0, 0, 0, 0), deadline=1000.0),)),
        )
        without = self.oracle.advise(state(*robots))
        with_truth = self.oracle.advise(state(*robots, truth=True))
        self.assertEqual(with_truth.reservations, without.reservations)
        self.assertEqual(with_truth.pair_bias, without.pair_bias)

    def test_the_oracle_keeps_the_same_task_revision_and_expiry_bounds(self) -> None:
        robots = (
            _agent("0", position=(0.5, 0.5), beliefs=(belief(TASK, requirements=(1, 0, 0, 0, 0), deadline=1000.0, revision=5),)),
        )
        advice = self.oracle.advise(state(robots[0]))
        self.assertEqual(advice.reservations[("0", TASK)].task_revision, 5)
        self.assertEqual(advice.valid_until, advice.reservations[("0", TASK)].valid_until)


class FailureClassTests(unittest.TestCase):
    """Each class is reachable, and the order of the rules is the order they are checked."""

    def _timeline(self, travel_floor: float = 0.0) -> Any:
        return SimpleNamespace(travel_floor=lambda: travel_floor)

    def _classify(self, *, work=1.0, offset=100.0, travel_floor=0.0, fact=None, budget=100.0, detected=0.0, reserved=1.0, first_dispatch=1.0) -> str:
        # The primary cause only; the secondaries are asserted through the attribution
        # closure in the learnability driver.
        return _classify(
            timeline=self._timeline(travel_floor),
            row={"work_needed": work, "deadline_offset": offset, "progress_setbacks": 0.0},
            fact=fact or {},
            budget=budget,
            detected=detected,
            reserved=reserved,
            first_dispatch=first_dispatch,
        )[0]

    def test_physics_beats_every_other_explanation(self) -> None:
        # Travel alone already exceeds the budget, so no ordering could have saved it.
        self.assertEqual(
            self._classify(work=50.0, offset=60.0, travel_floor=20.0),
            "travel_or_deadline_impossible",
        )

    def test_a_broken_commitment_is_reported_before_an_ordering_complaint(self) -> None:
        self.assertEqual(
            self._classify(fact={"was_broken": True}), "reservation_or_execution_break"
        )
        self.assertEqual(
            self._classify(
                fact={}, reserved=1.0, first_dispatch=1.0,
            ),
            "candidate_ordering_error",
        )

    def test_never_planned_for_is_a_belief_failure(self) -> None:
        self.assertEqual(
            self._classify(reserved=None, first_dispatch=None), "missing_or_late_belief"
        )
        # Known only after half the budget was gone: the same class.
        self.assertEqual(
            self._classify(detected=0.0, reserved=60.0, budget=100.0, first_dispatch=60.0),
            "missing_or_late_belief",
        )

    def test_saturation_decides_between_capacity_and_choice(self) -> None:
        late = {"saturation_at_dispatch": 1.0}
        free = {"saturation_at_dispatch": 0.5}
        self.assertEqual(
            self._classify(fact=late, detected=0.0, reserved=1.0, first_dispatch=80.0, budget=100.0),
            "capacity_conflict",
        )
        self.assertEqual(
            self._classify(fact=free, detected=0.0, reserved=1.0, first_dispatch=80.0, budget=100.0),
            "task_contention_error",
        )

    def test_every_class_is_named_in_the_reported_order(self) -> None:
        self.assertEqual(
            set(FAILURE_CLASSES),
            {
                "candidate_ordering_error",
                "task_contention_error",
                "missing_or_late_belief",
                "travel_or_deadline_impossible",
                "reservation_or_execution_break",
                "capacity_conflict",
            },
        )


class SplitContractTests(unittest.TestCase):
    """A seed's role is written down once, and a demoted block stays demoted."""

    def test_no_seed_belongs_to_two_splits(self) -> None:
        seeds = [seed for split in SPLITS.values() for seed in split.seeds]
        self.assertEqual(len(seeds), len(set(seeds)), "two splits share a seed")
        self.assertEqual(len(seeds), len(all_seeds()))

    def test_every_seed_range_is_documented_and_non_empty(self) -> None:
        for name, split in SPLITS.items():
            with self.subTest(split=name):
                self.assertTrue(split.seeds)
                self.assertTrue(split.purpose)
                self.assertEqual(split.name, name)

    def test_the_legacy_block_is_no_longer_called_a_test_set(self) -> None:
        legacy = split_of("analysis_legacy")
        self.assertNotIn(legacy.name.lower(), RETIRED_NAMES)
        # The demotion is stated where a reader will look for it.
        self.assertIn("demoted", legacy.purpose)
        # And no split carries a retired name at all.
        for name in SPLITS:
            self.assertNotIn(name.lower(), RETIRED_NAMES)

    def test_the_fresh_block_was_recorded_after_the_gate(self) -> None:
        self.assertTrue(split_of("analysis_fresh").frozen)
        self.assertNotEqual(
            split_of("analysis_fresh").seeds, split_of("analysis_legacy").seeds
        )


class OfflineOnlyTests(unittest.TestCase):
    """The hindsight ceiling reads ground truth, so it must stay out of the online path."""

    ONLINE_MODULES = (
        "joint_driver",
        "orchestrator",
        "coordinator",
        "contracts",
        "ports",
        "assignment_gate",
        "task_registry",
        "comm",
        "lifecycle_log",
        "event_bus",
        "search_runtime",
        "task03_runtime",
        "piloc_runtime",
        "detection_replay",
    )

    def test_no_online_module_imports_the_hindsight_or_attribution_tooling(self) -> None:
        root = Path(__file__).resolve().parent.parent
        for name in self.ONLINE_MODULES:
            source = (root / f"{name}.py").read_text(encoding="utf-8")
            for banned in ("expired_attribution", "learnability import", "learnability."):
                self.assertNotIn(
                    banned,
                    source,
                    f"{name}.py must not reach the hindsight/attribution tooling",
                )

    def test_the_adapters_do_not_reach_it_either(self) -> None:
        root = Path(__file__).resolve().parent.parent / "adapters"
        for path in sorted(root.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(
                "expired_attribution", source, f"{path.name} must not reach the hindsight"
            )


if __name__ == "__main__":
    unittest.main()
