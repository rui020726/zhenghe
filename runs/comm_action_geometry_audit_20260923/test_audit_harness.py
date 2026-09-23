"""Zero-episode tests for the action-geometry audit harness.

Run:  python -m unittest test_audit_harness -v

These test the *measuring instruments*, not the episodes: the tripwire that makes "0 new episodes"
enforceable, the step-class partition that must not let a refusal masquerade as a strict-subset
competition point, and the ledger-only counterfactual, which has to be checked against a hand
computed case before any of its numbers are quoted.

One of these tests is deliberately about a failure: it proves the tripwire has teeth rather than
being a decorator nobody noticed was a no-op. The delivered round established that convention
(learnability protocol section 5.3), and an audit that forbids itself from running episodes needs
the same kind of evidence as an audit that runs them.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

import audit_common as A

CALLS = A.install_env_tripwire()
A.enter_source_round()

import audit_cells as C  # noqa: E402
import audit_run as R  # noqa: E402


def make_cell(
    *,
    emissions: list[tuple[int, str, int, int, bool]],
    refusals: list[tuple[int, str, int, float]],
    probe: dict[str, Any],
    budget: int | None = A.BUDGET_BYTES,
    targets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a cell the way the row + transcript pair would, without touching an environment."""
    row = {
        "goals": 12,
        "arm": "unit_test_arm",
        "role": "competitor",
        "seed": 999999,
        "budget_bytes": budget,
        "spent_bytes": sum(item[3] for item in emissions),
        "budget_used_fraction": (
            round(sum(item[3] for item in emissions) / budget, 5) if budget else None
        ),
        "on_time": 0,
        "detected": len({item[1] for item in emissions}),
        "placed": 12,
        "probe": probe,
        "emission_sequence": [
            [step, target, revision, cost, 6, first]
            for step, target, revision, cost, first in emissions
        ],
        "emission_types": {
            "first": sum(1 for item in emissions if item[4]),
            "refresh": sum(1 for item in emissions if not item[4]),
        },
        "refusal_sequence": [
            [step, target, "budget_exhausted", "policy"] for step, target, _, _ in refusals
        ],
        "refusals": {
            "budget_exhausted": len(refusals),
            "capacity_step": 0,
            "first_budget_refusal_step": min((item[0] for item in refusals), default=None),
            "reasons": ["budget_exhausted"] if refusals else [],
            "total": len(refusals),
        },
        "targets": list((targets or {}).values()),
        "ledger": {},
    }
    transcript = [
        {"kind": "send", "step": step, "target": target, "cost_bytes": cost, "first": first,
         "revision": revision}
        for step, target, revision, cost, first in emissions
    ] + [
        {"kind": "refused", "step": step, "target": target, "cost_bytes": cost,
         "remaining_bytes": remaining, "reason": "budget_exhausted"}
        for step, target, cost, remaining in refusals
    ]
    return row, transcript


class TripwireTests(unittest.TestCase):
    def test_every_episode_entry_point_raises(self) -> None:
        import pressure_env
        import pressure_matrix
        import record_layer

        for label, call in (
            ("pressure_env.run_episode", lambda: pressure_env.run_episode(6, 1)),
            ("pressure_env.run_arm", lambda: pressure_env.run_arm(6, 1, "send_once")),
            ("record_layer.record", lambda: record_layer.record(6, 1)),
            ("pressure_matrix.run", lambda: pressure_matrix.run(6, 0, 1)),
            (
                "pressure_env.agreement_with_archived_rows",
                lambda: pressure_env.agreement_with_archived_rows((1,), ("send_once",)),
            ),
        ):
            with self.assertRaises(A.EnvironmentCallForbidden, msg=label):
                call()
        self.assertEqual(
            len(CALLS), 5,
            "the tripwire must log each blocked call; an empty log here means it wrapped nothing",
        )

    def test_tripwire_is_installed_before_the_round_modules_are_used(self) -> None:
        #: A tripwire installed after the episode functions were already imported-and-bound
        #: somewhere else would report a clean log while being unreachable. Check it is live.
        import pressure_env

        self.assertTrue(getattr(pressure_env.run_episode, "_audit_tripwire", False))
        self.assertTrue(callable(getattr(pressure_env.run_episode, "_audit_original", None)))

    def test_write_json_refuses_to_escape_the_audit_directory(self) -> None:
        with self.assertRaises(ValueError):
            A.write_json(A.SOURCE_ROUND / "GATES_g6.json", {"would": "overwrite a verdict"})


class QuantileTests(unittest.TestCase):
    def test_empty_and_single_value_inputs_are_not_confused(self) -> None:
        self.assertEqual(A.quantiles([])["n"], 0)
        self.assertIsNone(A.quantiles([])["median"])
        one = A.quantiles([7.0])
        self.assertEqual((one["n"], one["median"], one["min"], one["max"]), (1, 7.0, 7.0, 7.0))

    def test_normalization_makes_int_and_str_keys_equal(self) -> None:
        #: The aggregators key per-stream maps by int(seed); the files on disk have strings. Without
        #: normalization every one of those maps reads as a reproduction failure.
        published = {"per_stream": {"1000000": 5.0}}
        recomputed = {"per_stream": {1000000: 5.0}}
        self.assertEqual(A.deep_diff(published, recomputed), [])
        self.assertTrue(A.deep_diff({"a": 1}, {"a": 2}))

    def test_diff_directions_are_distinguishable(self) -> None:
        self.assertIn("only-in-published", A.deep_diff({"a": 1}, {})[0])
        self.assertIn("only-in-recomputed", A.deep_diff({}, {"a": 1})[0])


class StepClassTests(unittest.TestCase):
    """The partition the whole audit stands on, checked on numbers with a known answer."""

    def setUp(self) -> None:
        self.probe = {
            "steps_offered": 100,
            "visible_total": 250,
            "payable_total": 90,
            "block_reasons": {"allowed": 60, "budget_exhausted": 60},
            "competition_points": 10,
            "joint_only_points": 20,
            "credit_blocked_steps": 40,
            "capacity_blocked_steps": 0,
            "max_visible": 5,
            "steps_with_multiple_candidates": 50,
            "steps_with_one_candidate": 50,
            "multi_all_payable_within_credit_steps": 5,
            "multi_none_payable_steps": 15,
            "multi_unclassifiable_steps": 0,
            "multi_candidate_prices": {"count": 100, "min": 2200, "max": 2900, "sum": 255000},
            "first_credit_exhaustion_step": 30,
            "first_credit_exhaustion_facts": None,
            "competition_sample": [],
        }

    def _analyse(self, probe: dict[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
        row, transcript = make_cell(
            emissions=[(1, "t1", 1, 2600, True)],
            refusals=[(30, "t1", 2700, 1000.0)],
            probe=probe if probe is not None else {**self.probe, **overrides},
        )
        original = A.read_transcript
        A.read_transcript = lambda *args, **kwargs: transcript
        try:
            return C.analyse_cell("g12", row)
        finally:
            A.read_transcript = original

    def test_a_refusal_never_becomes_a_strict_subset_step(self) -> None:
        #: The error the protocol forbids (learnability section 7, L3) is calling any refusal a
        #: competition point. Here 40 steps have nothing payable and 60 have a refusal, yet only
        #: the 10 declared strict-subset steps may be counted as competition.
        cell = self._analyse()
        self.assertEqual(cell["steps_strict_subset"], 10)
        self.assertEqual(cell["steps_none_payable_two_or_more"], 15)
        self.assertEqual(cell["steps_joint_over_credit"], 20)
        self.assertNotEqual(
            cell["steps_strict_subset"], cell["steps_none_payable_any_visible"]
        )

    def test_multi_candidate_partition_closes(self) -> None:
        cell = self._analyse()
        self.assertEqual(
            cell["steps_ge_2_candidates_exact"],
            cell["steps_all_payable_within_credit"]
            + cell["steps_joint_over_credit"]
            + cell["steps_strict_subset"]
            + cell["steps_none_payable_two_or_more"],
        )

    def test_dual_action_steps_come_from_the_block_reason_identity(self) -> None:
        #: allowed + blocked - offered = steps that had both a payable and a blocked candidate.
        cell = self._analyse()
        self.assertEqual(cell["steps_both_payable_and_blocked"], 60 + 60 - 100)
        self.assertEqual(cell["steps_at_least_one_payable"], 60)

    def test_missing_geometry_is_reported_as_missing_not_as_zero(self) -> None:
        stripped = {
            key: value
            for key, value in self.probe.items()
            if key not in (
                "steps_with_multiple_candidates", "steps_with_one_candidate",
                "multi_all_payable_within_credit_steps", "multi_none_payable_steps",
                "multi_unclassifiable_steps",
            )
        }
        cell = self._analyse(probe=stripped)
        self.assertFalse(cell["geometry_recorded"])
        self.assertIsNone(cell["steps_ge_2_candidates_exact"])
        self.assertIsNone(cell["steps_all_payable_within_credit"])
        #: The delivered layers still bound the multi-candidate count from the stored sums.
        self.assertLessEqual(
            cell["steps_ge_2_candidates_lower_bound"], cell["steps_ge_2_candidates_upper_bound"]
        )
        records = [R.compact_cell(cell)]
        denominators = R._denominators(records)
        self.assertEqual(denominators["all_payable"]["streams_missing"], 1)
        self.assertEqual(denominators["all_payable"]["total_steps"], 0)

    def test_zero_candidate_steps_are_the_complement_of_offered_steps(self) -> None:
        cell = self._analyse()
        self.assertEqual(cell["steps_zero_candidate"], 1000 - 100)


class CounterfactualTests(unittest.TestCase):
    """The ledger-only saving has to be right on a case whose answer is known by hand."""

    def _cell(self, emissions: list[tuple[int, str, int, int, bool]],
              refusals: list[tuple[int, str, int, float]],
              targets: dict[str, Any]) -> dict[str, Any]:
        row, transcript = make_cell(
            emissions=emissions, refusals=refusals, probe={"steps_offered": 50},
            targets=targets,
        )
        original = A.read_transcript
        A.read_transcript = lambda *args, **kwargs: transcript
        try:
            return C.analyse_cell("g12", row)
        finally:
            A.read_transcript = original

    def test_one_held_refresh_unlocks_exactly_the_refusal_it_funds(self) -> None:
        # Credit 29 300. Step 1 sends a first report (2 600), step 2 sends a refresh (2 600),
        # leaving 24 100. A new target's first report of 24 800 is refused at step 3 with 24 100
        # left: short by 700. Holding the step-2 refresh releases 2 600, which funds it.
        cell = self._cell(
            emissions=[(1, "t1", 1, 2600, True), (2, "t1", 2, 2600, False)],
            refusals=[(3, "t2", 24800, 24100.0)],
            targets={"t2": {"target": "t2", "deadline": 120.0, "outcome": "expired"}},
        )
        savings = R.ledger_only_savings(cell)
        self.assertEqual(len(savings), 1)
        held = savings[0]
        self.assertEqual(held["held_step"], 2)
        self.assertEqual(held["released_bytes"], 2600)
        self.assertEqual(held["unlocked_attempts"], 1)
        self.assertEqual(held["unlocked_first_report_targets"], 1)
        self.assertFalse(held["unlock_is_refresh_only"])
        self.assertEqual(held["earliest_delay_steps"], 1)
        self.assertAlmostEqual(held["earliest_slack_at_unlock"], 117.0, places=3)

    def test_a_saving_too_small_unlocks_nothing_and_says_so(self) -> None:
        cell = self._cell(
            emissions=[(1, "t1", 1, 2600, True), (2, "t1", 2, 300, False)],
            refusals=[(3, "t2", 24800, 24100.0)],
            targets={"t2": {"target": "t2", "deadline": 120.0, "outcome": "expired"}},
        )
        self.assertEqual(R.ledger_only_savings(cell)[0]["unlocked_attempts"], 0)

    def test_refusals_before_the_held_packet_are_never_counted_as_unlocked(self) -> None:
        #: Time order is the whole point: a saving made at step 20 cannot explain step 5.
        cell = self._cell(
            emissions=[(20, "t1", 1, 2600, True), (30, "t1", 2, 2600, False)],
            refusals=[(5, "t2", 2600, 100.0), (40, "t2", 2600, 100.0)],
            targets={"t2": {"target": "t2", "deadline": 120.0, "outcome": "expired"}},
        )
        held = R.ledger_only_savings(cell)[0]
        self.assertEqual(held["unlocked_attempts"], 1)
        self.assertEqual(held["earliest_delay_steps"], 10)

    def test_the_rule_that_never_refreshes_has_nothing_to_hold(self) -> None:
        #: The audit's most consequential negative result, checked rather than asserted in prose.
        cell = self._cell(
            emissions=[(1, "t1", 1, 2600, True), (2, "t2", 1, 2600, True)],
            refusals=[(3, "t3", 2600, 100.0)],
            targets={"t3": {"target": "t3", "deadline": 90.0, "outcome": "expired"}},
        )
        self.assertEqual(R.ledger_only_savings(cell), [])

    def test_unbudgeted_arm_is_skipped_rather_than_dividing_by_none(self) -> None:
        row, transcript = make_cell(
            emissions=[(1, "t1", 1, 2600, True)], refusals=[],
            probe={"steps_offered": 10}, budget=None,
        )
        original = A.read_transcript
        A.read_transcript = lambda *args, **kwargs: transcript
        try:
            cell = C.analyse_cell("g12", row)
        finally:
            A.read_transcript = original
        self.assertIsNone(cell["budget_bytes"])
        self.assertEqual(R.ledger_only_savings(cell), [])


class QuoteGeometryTests(unittest.TestCase):
    """`strict_step_measurements` takes the probe's candidate rows already expanded into dicts,
    which is the shape ``analyse_cell`` hands it; the tests feed it that same shape."""

    @staticmethod
    def candidate(target: str, charge: int, executable: bool, first: int) -> dict[str, Any]:
        return {
            "target": target, "charge": charge, "executable": executable,
            "reason": "allowed" if executable else "budget_exhausted",
            "first_emission": first, "remaining_time": 10.0,
        }

    def test_credit_position_and_seam_are_computed_as_defined(self) -> None:
        sample = [{
            "step": 17, "remaining_bytes": 2534, "visible": 3, "payable": 2,
            "candidates": [
                self.candidate("t1", 2300, True, 0),
                self.candidate("t2", 2400, True, 0),
                self.candidate("t3", 2900, False, 0),
            ],
        }]
        raw, checks = R.strict_step_measurements(sample)
        self.assertEqual(raw["charge_range_bytes"], [600.0])
        self.assertEqual(raw["seam_bytes_to_next_unaffordable"], [500.0])
        self.assertEqual(raw["payable_count"], [2.0])
        self.assertEqual(raw["first_report_candidates"], [0.0])
        self.assertEqual(raw["refresh_candidates"], [3.0])
        #: 2534 sits 234 of the way through a 600-byte band.
        self.assertAlmostEqual(raw["position_of_credit_within_charge_band"][0], 234 / 600)
        self.assertEqual(checks["credit_test_explains_payable"], 1)

    def test_a_step_where_the_credit_test_disagrees_is_flagged_not_silently_counted(self) -> None:
        sample = [{
            "step": 5, "remaining_bytes": 2534, "visible": 2, "payable": 2,
            "candidates": [
                self.candidate("t1", 2300, True, 1),
                #: payable claims two of two are affordable, but this charge is above the credit.
                self.candidate("t2", 9000, True, 0),
            ],
        }]
        _raw, checks = R.strict_step_measurements(sample)
        self.assertEqual(checks["credit_test_explains_payable"], 0)
        self.assertEqual(checks["steps"], 1)


class TransitionTests(unittest.TestCase):
    def _record(self, **overrides: Any) -> dict[str, Any]:
        base = {
            "strict_sample": [], "first_none_payable_step": None, "steps_strict_subset": 0,
            "strict_sample_covers_all_points": True,
        }
        return {**base, **overrides}

    def test_kinds_separate_no_band_from_band_then_starvation(self) -> None:
        self.assertEqual(
            R.transition_reading(self._record())["kind"], "never_ran_out"
        )
        self.assertEqual(
            R.transition_reading(self._record(first_none_payable_step=50))["kind"],
            "no_band_visited",
        )
        visited = self._record(
            steps_strict_subset=3,
            first_none_payable_step=50,
            strict_sample=[{"step": 20}, {"step": 40}, {"step": 60}],
        )
        self.assertEqual(R.transition_reading(visited)["kind"], "band_visited_before_starvation")
        self.assertEqual(R.transition_reading(visited)["first_strict_subset_step"], 20)

    def test_a_truncated_sample_is_marked_as_a_bound(self) -> None:
        visited = self._record(
            steps_strict_subset=147,
            first_none_payable_step=None,
            strict_sample=[{"step": 17}],
            strict_sample_covers_all_points=False,
        )
        reading = R.transition_reading(visited)
        self.assertFalse(reading["first_strict_is_exact"])
        self.assertEqual(reading["first_strict_subset_step"], 17)


class RefusalClassTests(unittest.TestCase):
    def test_first_report_refresh_and_retry_are_distinguished(self) -> None:
        cell_rows, transcript = make_cell(
            emissions=[(1, "t1", 1, 2600, True)],
            refusals=[
                (5, "t2", 2600, 100.0),   # never emitted: a refused first report
                (6, "t2", 2600, 100.0),   # same target again: a retry of that first report
                (7, "t1", 2600, 100.0),   # emitted at step 1: a refused refresh
            ],
            probe={"steps_offered": 10},
        )
        original = A.read_transcript
        A.read_transcript = lambda *args, **kwargs: transcript
        try:
            cell = C.analyse_cell("g12", cell_rows)
        finally:
            A.read_transcript = original
        composition = R.refusal_composition(cell)
        self.assertEqual(composition["attempts"], 3)
        self.assertEqual(composition["first_report_attempts"], 2)
        self.assertEqual(composition["refresh_attempts"], 1)
        self.assertEqual(composition["retry_attempts"], 1)
        self.assertEqual(composition["distinct_targets"], 2)
        self.assertFalse(composition["first_refusal_is_first_report"] is None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
