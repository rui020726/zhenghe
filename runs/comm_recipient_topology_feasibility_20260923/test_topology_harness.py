"""Zero-episode tests for the recipient/topology audit harness.

Run:  python -m unittest test_topology_harness -v

These test the instruments: that the episode tripwire has teeth, that the pricing identity the
whole audit rests on is checked rather than assumed, that the scale-equivalence claims are
arithmetically enforced, and that the projection distinguishes "priced differently" from "delivered"
-- the failure mode that would let a coalition-routing idea look like progress while silently
deleting every first report.
"""

from __future__ import annotations

import unittest
from typing import Any

import topology_common as T

CALLS = T.install_env_tripwire()

import topology_projection as P  # noqa: E402
import topology_scalar_proof as S  # noqa: E402


def message(payload: int, members: list[str], first: bool, step: int = 1,
            target: str = "t", seed: int = 1, layer: str = "g12", arm: str = "x") -> dict:
    return {
        "layer": layer, "arm": arm, "seed": seed, "step": step, "target": target,
        "class": "first_report" if first else "refresh",
        "payload_bytes": payload, "members": members, "subscribers_recorded": 6,
    }


class TripwireTests(unittest.TestCase):
    def test_every_episode_entry_point_raises_and_is_logged(self) -> None:
        import pressure_env
        import pressure_matrix
        import record_layer

        before = len(CALLS)
        for call in (
            lambda: pressure_env.run_episode(6, 1),
            lambda: pressure_env.run_arm(6, 1, "send_once"),
            lambda: record_layer.record(6, 1),
            lambda: pressure_matrix.run(6, 0, 1),
            lambda: pressure_env.agreement_with_archived_rows((1,), ("send_once",)),
        ):
            with self.assertRaises(T.EnvironmentCallForbidden):
                call()
        self.assertEqual(len(CALLS), before + 5)

    def test_write_json_cannot_escape_this_round(self) -> None:
        with self.assertRaises(ValueError):
            T.write_json(T.SOURCE_ROUND / "GATES_g12.json", {"would": "overwrite a verdict"})

    def test_every_source_anchor_resolves_in_the_current_code(self) -> None:
        #: The source map cites line numbers rather than prose. If an anchor moved, the map is
        #: wrong and must say so, so this test is the map's own freshness check.
        facts = T.source_facts()
        missing = sorted(name for name, fact in facts.items() if not fact["found"])
        self.assertEqual(missing, [], f"anchors no longer present in the code: {missing}")
        for name, fact in facts.items():
            self.assertTrue(fact["lines"], name)
            self.assertIsInstance(fact["text"], str)


class PricingIdentityTests(unittest.TestCase):
    def test_cost_is_exactly_payload_times_fanout(self) -> None:
        # mirrors comm.py:895 `cost_bytes = payload_bytes * len(self._subscribers)`
        self.assertEqual(446 * 6, 2676)

    def test_projection_pricies_broadcast_as_payload_times_six(self) -> None:
        records = [message(400, ["0", "1"], first=False)]
        out = P.project(records)
        self.assertEqual(out["broadcast"]["projected_charge_bytes"]["min"], 2_400.0)
        self.assertEqual(out["coalition"]["projected_charge_bytes"]["min"], 800.0)

    def test_a_first_report_with_no_coalition_is_priced_to_zero_and_counted(self) -> None:
        #: This is the trap the whole R4 question turns on: fan-out 0 looks like a bargain unless
        #: the projection also records that the message reaches nobody.
        records = [message(400, [], first=True), message(400, ["0", "1"], first=False)]
        out = P.project(records)
        self.assertEqual(out["coalition"]["fanout"]["min"], 0.0)
        self.assertEqual(out["coalition"]["fanout_distinct_values"], [0, 2])
        self.assertAlmostEqual(out["coalition"]["share_of_messages_with_empty_recipient_set"], 0.5)
        self.assertEqual(out["coalition"]["bytes_that_would_never_be_sent_under_this_rule"], 400)

    def test_empty_recipient_set_is_not_reported_as_a_saving(self) -> None:
        records = [message(400, [], first=True) for _ in range(3)]
        out = P.project(records)
        self.assertEqual(out["coalition"]["projected_charge_bytes"]["max"], 0.0)
        self.assertEqual(
            out["coalition"]["share_of_messages_with_empty_recipient_set"], 1.0
        )


class PayabilityTests(unittest.TestCase):
    def test_classes_are_mutually_exclusive_and_use_the_stored_definition(self) -> None:
        cases = [
            ([100, 120], 500, "all_payable_within_credit"),
            ([100, 120], 150, "joint_over_credit"),
            ([100, 900], 500, "strict_subset_payable"),
            ([900, 950], 100, "none_payable"),
        ]
        for charges, remaining, expected in cases:
            verdict = S.pay_at([float(c) for c in charges], remaining)
            self.assertEqual(verdict["kind"], expected, (charges, remaining))

    def test_a_refusal_is_never_counted_as_a_strict_subset(self) -> None:
        #: The distinction the source protocol forbids collapsing, checked on the projection's own
        #: classifier so this round cannot reintroduce it.
        verdict = S.pay_at([900.0, 950.0], 100.0)
        self.assertFalse(verdict["strict_subset"])
        self.assertEqual(verdict["payable"], 0)


class ScaleInvarianceTests(unittest.TestCase):
    def test_window_over_packet_is_independent_of_fanout(self) -> None:
        charges = [2_300.0, 2_400.0, 2_900.0]
        ratios = set()
        for scale in (1 / 6.0, 1.0, 2.0, 6.0):
            scaled = [charge * scale for charge in charges]
            window = max(scaled) - min(scaled)
            ratios.add(round(window / (sum(scaled) / len(scaled)), 9))
        self.assertEqual(len(ratios), 1, f"ratio moved with scale: {ratios}")

    def test_raising_fanout_equals_lowering_the_budget_on_every_step(self) -> None:
        charges = [2_300.0, 2_900.0]
        remaining = 2_500.0
        for scale in (2.0, 3.0, 0.5):
            widened = S.pay_at([charge * scale for charge in charges], remaining)
            tightened = S.pay_at(charges, remaining / scale)
            self.assertEqual(widened["kind"], tightened["kind"], scale)
            self.assertEqual(widened["payable"], tightened["payable"], scale)

    def test_first_report_pair_choices_are_detected_when_they_exist(self) -> None:
        #: Broadcast: two new targets priced by their payloads -> a real spread exists.
        #: Coalition: both are pre-coalition -> both price to zero -> no choice, and the helper
        #: must say so rather than reporting the pair count as opportunity.
        records = [
            message(400, [], first=True, step=5, target="a"),
            message(440, [], first=True, step=5, target="b"),
        ]
        broad = P.first_report_choices(records, "broadcast")
        coalition = P.first_report_choices(records, "coalition")
        self.assertEqual(broad["steps_where_a_first_report_would_be_priced_to_zero"], 0)
        self.assertEqual(
            broad["streams_where_two_first_reports_ever_cost_different_amounts"], 1
        )
        self.assertEqual(coalition["steps_where_a_first_report_would_be_priced_to_zero"], 1)
        self.assertEqual(
            coalition["streams_where_two_first_reports_ever_cost_different_amounts"], 0
        )


class ClassCountingTests(unittest.TestCase):
    def test_retry_is_reported_orthogonally_rather_than_as_a_third_class(self) -> None:
        records = [
            message(400, [], first=True, step=1, target="a"),
            message(410, ["0"], first=False, step=9, target="a"),
        ]
        out = P.project(records)
        classes = set(out["coalition"]["fanout_by_message_class"])
        self.assertEqual(classes, {"first_report", "refresh"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
