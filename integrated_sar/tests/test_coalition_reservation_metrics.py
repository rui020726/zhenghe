"""The metrics that read a coalition plan out of an allocation trace.

The planner publishes a commitment, the allocator may or may not take it, and the
target may or may not be reached. Those are three different things and the trace has
to keep them apart, so this file pins the arithmetic that separates them:

* a default-off feature must not move any pre-existing number -- the arm label, the
  depot share, the group gaps and the commitment walk all have to read exactly as they
  did before the new columns existed;
* a commitment is counted once per decision that was made under it, and its cycle
  counters are counted once per advice revision, not once per row they appear on;
* the three timings are measured from detection, so a plan that reacts in the tick it
  can act reports zero rather than being silently dropped;
* the missing-capability vector is read per file, because the pooled view keeps the
  element-wise minimum and would otherwise compare vectors lexicographically.
"""

from __future__ import annotations

import unittest
from typing import Any

from integrated_sar.allocation_trace import (
    PLAN_COUNTERS,
    analyse,
    arm_name,
    report,
    summarise,
)


def decision(
    *,
    timestamp: float,
    agent_id: str = "0",
    chosen_index: int = 1,
    action_task_ids: tuple[str | None, ...] = ("depot", "target-a"),
    reservations: tuple[str, ...] = (),
    held: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    revision: int | None = None,
    truth: dict[str, Any] | None = None,
    status: str = "idle",
) -> dict[str, Any]:
    return {
        "event_type": "decision",
        "agent_id": agent_id,
        "status": status,
        "timestamp": timestamp,
        "step": int(timestamp),
        "action_task_ids": list(action_task_ids),
        "mask": [False] * len(action_task_ids),
        "chosen_index": chosen_index,
        "chosen_kind": "depot" if chosen_index == 0 else "task",
        "chosen_task": action_task_ids[chosen_index],
        "current_kind": None,
        "current_task": None,
        "needs_decision": True,
        "replanned": False,
        "truth": truth or {"targets": {}},
        "advice_revision": revision,
        "advice_reservations": list(reservations),
        "held_reservation": held,
        "advice_plan": plan or {},
    }


def target_truth(
    *,
    detected_at: float,
    requirements: list[int],
    remaining: list[int],
    on_site: list[str] | None = None,
    en_route: list[str] | None = None,
    capacity: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "requirements": requirements,
        "remaining": remaining,
        "on_site": on_site or [],
        "en_route": en_route or [],
        "capacity": capacity or [0.0] * len(requirements),
        "detected_at": detected_at,
        "deadline": 500.0,
        "covered": all(value <= 0 for value in remaining),
        "en_route_covers": False,
        "expired": False,
        "stage_index": 0,
    }


class ArmLabelTests(unittest.TestCase):
    """Three policies, three labels -- a collision would pool two of them."""

    def test_the_planner_gets_its_own_label(self) -> None:
        self.assertEqual(
            arm_name(
                use_coordinator=True, residual_scale=0.25, coordinator_policy="coalition_planner"
            ),
            "coalplan0.25",
        )

    def test_the_existing_labels_are_untouched(self) -> None:
        self.assertEqual(
            arm_name(use_coordinator=True, residual_scale=0.25), "rs0.25"
        )
        self.assertEqual(
            arm_name(
                use_coordinator=True, residual_scale=0.25, coordinator_policy="coalition"
            ),
            "coal0.25",
        )
        self.assertEqual(
            arm_name(use_coordinator=False, residual_scale=0.5), "nocoord"
        )
        self.assertEqual(
            arm_name(
                use_coordinator=True,
                residual_scale=0.25,
                task_logit_offset=0.1,
                coordinator_policy="coalition_planner",
            ),
            "coalplan0.25+cal0.1",
        )

    def test_an_unlisted_policy_keeps_the_historical_prefix(self) -> None:
        self.assertEqual(
            arm_name(
                use_coordinator=True, residual_scale=0.5, coordinator_policy="something-else"
            ),
            "coal0.5",
        )


class CommitmentAccountingTests(unittest.TestCase):
    """A commitment the allocator took is not the same as one it ignored."""

    def _rows(self) -> list[dict[str, Any]]:
        honoured = {
            "task_id": "target-a",
            "ability_slot": 0,
            "slot_dimension": 0,
            "task_revision": 3,
            "issued_at": 5.0,
            "valid_until": 10.0,
            "score": 0.35,
            "honoured": True,
        }
        broken = dict(honoured, honoured=False)
        plan = {"planned_reservations": 2, "redundant_candidates": 1}
        return [
            decision(timestamp=5.0, reservations=("target-a",), held=honoured, plan=plan, revision=2),
            # Same revision again: the counters describe the transition into revision 2
            # and must not be summed twice.
            decision(timestamp=6.0, reservations=("target-a",), held=honoured, plan=plan, revision=2),
            decision(timestamp=7.0, reservations=("target-a",), held=broken, plan=plan, revision=3),
            decision(timestamp=8.0),
        ]

    def test_commitments_are_split_into_taken_and_ignored(self) -> None:
        metrics = analyse(self._rows(), arm="coalplan0.25", seed=1)
        self.assertEqual(metrics.reserved_decisions, 3)
        self.assertEqual(metrics.honoured_reservations, 2)
        self.assertEqual(metrics.broken_reservations, 1)

    def test_cycle_counters_are_summed_once_per_advice_revision(self) -> None:
        metrics = analyse(self._rows(), arm="coalplan0.25", seed=1)
        totals = metrics.plan_totals()
        # Two distinct revisions, each carrying the same counters.
        self.assertEqual(totals["planned_reservations"], 4)
        self.assertEqual(totals["redundant_candidates"], 2)
        self.assertEqual(set(totals), set(PLAN_COUNTERS))

    def test_an_arm_without_commitments_reports_zeros(self) -> None:
        metrics = analyse(
            [decision(timestamp=5.0), decision(timestamp=6.0)],
            arm="rs0.25",
            seed=1,
        )
        self.assertEqual(metrics.reserved_decisions, 0)
        self.assertIsNone(summarise([metrics])["rs0.25"]["honoured_share"])
        self.assertEqual(
            summarise([metrics])["rs0.25"]["plan_totals"],
            {key: 0 for key in PLAN_COUNTERS},
        )


class TimingTests(unittest.TestCase):
    """Detection to reservation, to arrival, to the first step of work."""

    def _rows(self) -> list[dict[str, Any]]:
        # The plan holds the commitment from two steps after the detection; the group
        # lands on site ten steps later and the stage's requirement is met by then.
        return [
            decision(timestamp=2.0, reservations=("target-a",), revision=1),
            decision(timestamp=4.0, reservations=("target-a",), revision=2),
            decision(
                timestamp=12.0,
                reservations=("target-a",),
                truth={
                    "targets": {
                        "target-a": target_truth(
                            detected_at=0.0,
                            requirements=[1, 0, 0, 0, 0],
                            remaining=[0, 0, 0, 0, 0],
                            on_site=["0", "1"],
                            capacity=[1.0, 1.0, 0.0, 0.0, 0.0],
                        )
                    }
                },
            ),
            decision(timestamp=14.0),
        ]

    def test_the_gaps_are_measured_from_detection(self) -> None:
        metrics = analyse(self._rows(), arm="coalplan0.25", seed=1)
        self.assertEqual(metrics.dispatched_targets, 1)
        self.assertEqual(metrics.reservation_gaps, [2.0])
        self.assertEqual(metrics.arrival_gaps, [12.0])
        self.assertEqual(metrics.group_arrival_gaps, [12.0])
        # Arrival and the first step of work coincide for a stage that starts the moment
        # its members are on site.
        self.assertEqual(metrics.progress_gaps, [12.0])

    def test_a_target_never_reserved_is_counted_as_such(self) -> None:
        rows = [
            decision(
                timestamp=3.0,
                truth={"targets": {"target-a": target_truth(detected_at=0.0, requirements=[1, 0, 0, 0, 0], remaining=[1, 0, 0, 0, 0])}},
            )
        ]
        metrics = analyse(rows, arm="rs0.25", seed=1)
        self.assertEqual(metrics.dispatched_targets, 1)
        self.assertEqual(metrics.never_reserved, 1)
        self.assertEqual(metrics.reservation_gaps, [])
        summary = summarise([metrics])["rs0.25"]
        self.assertEqual(summary["never_reserved_share"], 1.0)
        self.assertIsNone(summary["reservation_gap_median"])

    def test_the_lead_time_is_not_tied_to_when_the_truth_block_arrives(self) -> None:
        # The commitment is held from t=7, seven steps after the detection at t=0, and
        # the first ground-truth block naming the target only shows up at t=40. Reading
        # the reservation out of that block instead of its own table would report a
        # lead time of zero.
        rows = [
            decision(timestamp=1.0, revision=1),
            decision(timestamp=7.0, reservations=("target-a",), revision=2),
            decision(
                timestamp=40.0,
                truth={"targets": {"target-a": target_truth(detected_at=0.0, requirements=[1, 0, 0, 0, 0], remaining=[1, 0, 0, 0, 0])}},
            ),
        ]
        metrics = analyse(rows, arm="coalplan0.25", seed=1)
        self.assertEqual(metrics.reservation_gaps, [7.0])
        self.assertEqual(metrics.never_reserved, 0)

    def test_a_commitment_cannot_precede_the_detection_it_answers(self) -> None:
        # Clamped rather than recorded as a negative: the advice can never carry a
        # target before the registry accepted its detection, so a negative gap would be
        # an artefact of the trace, not a measurement.
        rows = [
            decision(timestamp=1.0, reservations=("target-a",), revision=1),
            decision(
                timestamp=5.0,
                truth={"targets": {"target-a": target_truth(detected_at=6.0, requirements=[1, 0, 0, 0, 0], remaining=[1, 0, 0, 0, 0])}},
            ),
        ]
        self.assertEqual(analyse(rows, arm="coalplan0.25", seed=1).reservation_gaps, [0.0])


class MissingCapabilityTests(unittest.TestCase):
    """What never arrived, per ability dimension."""

    def test_the_vector_is_read_per_file_not_from_the_pooled_minimum(self) -> None:
        rows = [
            decision(
                timestamp=1.0,
                truth={"targets": {"target-a": target_truth(detected_at=0.0, requirements=[1, 0, 1, 0, 0], remaining=[0, 0, 1, 0, 0])}},
            ),
            decision(
                timestamp=2.0,
                chosen_index=2,
                action_task_ids=("depot", "target-a", "target-b"),
                truth={"targets": {"target-b": target_truth(detected_at=0.0, requirements=[1, 0, 1, 0, 0], remaining=[0, 0, 0, 0, 0])}},
            ),
        ]
        metrics = analyse(rows, arm="coalplan0.25", seed=1)
        self.assertEqual(metrics.dispatched_targets, 2)
        self.assertEqual(metrics.open_dimensions, [0, 0, 1, 0, 0])
        summary = summarise([metrics])["coalplan0.25"]
        self.assertEqual(summary["missing_capability_targets"], [0, 0, 1, 0, 0])
        self.assertEqual(summary["missing_capability_target_share"][2], 0.5)

    def test_over_dispatch_is_measured_at_the_target(self) -> None:
        rows = [
            decision(
                timestamp=1.0,
                truth={
                    "targets": {
                        "target-a": target_truth(
                            detected_at=0.0,
                            requirements=[1, 0, 0, 0, 0],
                            remaining=[0, 0, 0, 0, 0],
                            on_site=["0", "1"],
                            capacity=[2.0, 0.0, 0.0, 0.0, 0.0],
                        )
                    }
                },
            )
        ]
        metrics = analyse(rows, arm="coalplan0.25", seed=1)
        self.assertEqual(metrics.oversupplied_targets, 1)
        self.assertEqual(metrics.excess_units, [1.0])

    def test_a_target_without_a_truth_block_is_not_a_target(self) -> None:
        metrics = analyse([decision(timestamp=1.0)], arm="coalplan0.25", seed=1)
        self.assertEqual(metrics.dispatched_targets, 0)
        self.assertEqual(metrics.open_dimensions, [])
        summary = summarise([metrics])["coalplan0.25"]
        self.assertEqual(summary["missing_capability_target_share"], [])
        self.assertIn("missing capability vector", report(summarise([metrics])))


class BackwardCompatibilityTests(unittest.TestCase):
    """A trace written before these columns existed must still analyse."""

    def test_a_row_without_the_new_keys_is_read_as_absent(self) -> None:
        legacy: dict[str, Any] = {
            "event_type": "decision",
            "agent_id": "0",
            "status": "idle",
            "timestamp": 5.0,
            "step": 5,
            "action_task_ids": ["depot", "target-a"],
            "mask": [False, False],
            "chosen_index": 1,
            "chosen_task": "target-a",
            "truth": {"targets": {"target-a": {"deadline": 100.0, "covered": False}}},
        }
        metrics = analyse([legacy], arm="rs0.25", seed=1)
        self.assertEqual(metrics.reserved_decisions, 0)
        self.assertEqual(metrics.dispatched_targets, 1)
        self.assertEqual(metrics.never_reserved, 1)
        # No ``detected_at`` in the legacy truth block means no timing, not a zero.
        self.assertEqual(metrics.arrival_gaps, [])
        self.assertEqual(metrics.open_dimensions, [])

    def test_the_arm_level_report_still_renders(self) -> None:
        metrics = analyse([decision(timestamp=1.0)], arm="rs0.25", seed=1)
        text = report(summarise([metrics]))
        self.assertIn("== rs0.25", text)
        self.assertIn("coalition commitments", text)


if __name__ == "__main__":
    unittest.main()
