"""Tests for the machinery that reads the data, not for the data.

Round 7's lesson: two defects in that machinery (an aggregator that ignored the setting key, a
validator that raised on a missing probability instead of failing closed) invalidated evidence
without moving a single number. Everything here runs on synthetic rows and constructed
observations; no episode is started, so the round's launch ceiling is untouched by its own tests.

The probe tests are the interesting ones. A measurement wrapper is only trustworthy if it provably
cannot change what it measures, and round 6 lost a counterfactual run to a wrapper that turned an
ordered tuple into a set. So the suite asserts both halves: the probe returns the wrapped rule's
own object, and a deliberately tampering probe *fails* the same assertion.

    python -m unittest discover -s runs/comm_task_pressure_learnability_20260922 -t . -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import judge_layers
import pressure_common as common
import pressure_env
import pressure_metrics
import record_layer
import pressure_matrix
import seed_audit

from integrated_sar.comm import CommRelay
from integrated_sar.comm_budget import CommBudget
from integrated_sar.comm_policy import build_comm_policy, transport_order
from integrated_sar.contracts import TaskStageSpec
from integrated_sar.task_registry import TaskRegistry

from integrated_sar.tests.test_comm_budget_action_mask import RecordingWriter, detection


def item(
    target: str,
    charge: int | None,
    executable: bool,
    reason: str = "allowed",
    *,
    remaining_time: float = 100.0,
    first: bool = True,
) -> Any:
    """A stand-in SendItem carrying exactly the fields the probe reads."""
    return type(
        "Item",
        (),
        {
            "target_id": target,
            "attempt_charge_bytes": charge,
            "send_executable": executable,
            "send_block_reason": reason,
            "first_emission": first,
            "remaining_time": remaining_time,
        },
    )()


def observation(items: list[Any], remaining: int, step: int = 1) -> Any:
    return type(
        "Observation",
        (),
        {
            "items": tuple(items),
            "step": step,
            "remaining_budget_bytes": remaining,
            "budget_bytes_per_episode": common.BUDGET_BYTES,
            "capacity_remaining": None,
            "backlog_targets": len(items),
            "subscribers": 6,
        },
    )()


class FixedRule:
    """A rule that answers in a given order and claims that order."""

    name = "fixed_for_tests"
    orders_selection = True

    def __init__(self, answer: tuple[str, ...]) -> None:
        self.answer = answer
        self.calls = 0

    def parameters(self) -> dict[str, Any]:
        return {"answer": list(self.answer)}

    def decide(self, obs: Any) -> tuple[str, ...]:
        self.calls += 1
        return self.answer


class ProbeContractTests(unittest.TestCase):
    def test_probe_returns_the_wrapped_rules_own_object(self) -> None:
        rule = FixedRule(("target-2", "target-1"))
        probe = pressure_env.CandidateProbe(rule)
        result = probe.decide(observation([item("target-1", 10, True), item("target-2", 10, True)], 100))
        self.assertIs(result, rule.answer, "the probe must hand back the same object it was given")
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, ("target-2", "target-1"), "and must not reorder it")

    def test_probe_delegates_identity_and_ordering(self) -> None:
        probe = pressure_env.CandidateProbe(FixedRule(()))
        self.assertEqual(probe.name, "fixed_for_tests")
        self.assertTrue(probe.orders_selection)
        self.assertEqual(dict(probe.parameters()), {"answer": []})

    def test_a_reordering_wrapper_fails_the_same_assertion(self) -> None:
        """The meta-test: without this, the transparency assertion above proves nothing.

        Round 6 lost a counterfactual run to a probe that turned a rule's ordered selection into a
        set and back. The order decides who gets paid when the credit cannot cover everyone, so a
        wrapper that reorders is not measuring the same episode.
        """

        class Tampering:
            name = "fixed_for_tests"
            orders_selection = True

            def parameters(self) -> dict[str, Any]:
                return {}

            def decide(self, obs: Any) -> tuple[str, ...]:
                return tuple(sorted(set(obs.reordered_answer)))

        obs = observation([item("target-1", 10, True), item("target-2", 10, True)], 100)
        obs.reordered_answer = ("target-2", "target-1")
        probe = pressure_env.CandidateProbe(Tampering())
        result = probe.decide(obs)
        self.assertNotEqual(
            result,
            ("target-2", "target-1"),
            "the assertion has no teeth if a set round-trip cannot be caught",
        )
        honest = pressure_env.CandidateProbe(FixedRule(("target-2", "target-1")))
        self.assertEqual(honest.decide(obs), ("target-2", "target-1"))

    def test_empty_observation_is_not_counted_as_a_step(self) -> None:
        probe = pressure_env.CandidateProbe(FixedRule(()))
        probe.decide(observation([], 100))
        self.assertEqual(probe.summary()["steps_offered"], 0)
        self.assertEqual(probe.summary()["competition_points"], 0)

    def test_competition_point_needs_a_nonempty_strict_subset(self) -> None:
        # (flags, remaining credit, expected competition / joint-only / two-or-more payable)
        cases = [
            ([True, False], 10, 1, 0, 0),  # a non-empty strict subset is payable
            ([True, True], 15, 0, 1, 1),  # every candidate individually payable, not all jointly
            ([False, False], 0, 0, 0, 0),  # no choice left: the credit is gone
            ([True], 15, 0, 0, 0),  # one candidate is not a subset decision
            ([True, True, False], 15, 1, 0, 1),
            ([True, True, True], 25, 0, 1, 1),  # three wanted, credit for two and a half
        ]
        for flags, remaining, competition, joint, multi in cases:
            with self.subTest(flags=flags, remaining=remaining):
                probe = pressure_env.CandidateProbe(FixedRule(("target-1",)))
                items = [
                    item(f"target-{index}", 10, flag, "allowed" if flag else "budget_exhausted")
                    for index, flag in enumerate(flags)
                ]
                probe.decide(observation(items, remaining))
                summary = probe.summary()
                self.assertEqual(summary["competition_points"], competition)
                self.assertEqual(summary["joint_only_points"], joint)
                self.assertEqual(summary["multi_payable_points"], multi)

    def test_a_capacity_block_is_separate_from_a_credit_block(self) -> None:
        """The frozen arm sets no per-step capacity, so this must never be confused with it.

        Counting the two causes separately is what lets the report say that every competition
        point in the round was produced by the byte credit and not by a queueing limit.
        """
        probe = pressure_env.CandidateProbe(FixedRule(()))
        probe.decide(
            observation([item("target-1", 10, True), item("target-2", 10, False, "capacity_step")], 500)
        )
        summary = probe.summary()
        self.assertEqual(summary["competition_points"], 1)
        self.assertEqual(summary["competition_points_credit_cause"], 0)
        self.assertEqual(summary["competition_points_capacity_only"], 1)

    def test_joint_only_point_is_separate_from_a_credit_strict_subset(self) -> None:
        probe = pressure_env.CandidateProbe(FixedRule(()))
        # two candidates, both individually affordable (10 <= 15), together unaffordable (20 > 15)
        probe.decide(
            observation([item("target-1", 10, True), item("target-2", 10, True)], 15)
        )
        summary = probe.summary()
        self.assertEqual(summary["joint_only_points"], 1)
        self.assertEqual(summary["competition_points"], 0)

    def test_no_credit_with_candidates_counts_as_exhaustion_not_competition(self) -> None:
        probe = pressure_env.CandidateProbe(FixedRule(()))
        probe.decide(
            observation(
                [
                    item("target-1", 10, False, "budget_exhausted"),
                    item("target-2", 10, False, "budget_exhausted"),
                ],
                0,
                step=37,
            )
        )
        summary = probe.summary()
        self.assertEqual(summary["competition_points"], 0)
        self.assertEqual(summary["credit_blocked_steps"], 1)
        self.assertEqual(summary["first_credit_exhaustion_step"], 37)

    def test_probe_reads_without_writing(self) -> None:
        probe = pressure_env.CandidateProbe(FixedRule(("target-1",)))
        obs = observation([item("target-1", 10, True), item("target-2", 5, False)], 12, step=9)
        before = (obs.step, obs.remaining_budget_bytes, tuple(i.target_id for i in obs.items))
        probe.decide(obs)
        self.assertEqual(
            (obs.step, obs.remaining_budget_bytes, tuple(i.target_id for i in obs.items)), before
        )

    def test_sample_of_competition_steps_is_bounded(self) -> None:
        probe = pressure_env.CandidateProbe(FixedRule(()))
        for step in range(pressure_env.COMPETITION_SAMPLE_LIMIT + 30):
            probe.decide(
                observation([item("target-1", 10, True), item("target-2", 10, False)], 12, step=step)
            )
        summary = probe.summary()
        self.assertEqual(summary["competition_points"], 30 + pressure_env.COMPETITION_SAMPLE_LIMIT)
        self.assertEqual(len(summary["competition_sample"]), pressure_env.COMPETITION_SAMPLE_LIMIT)


class RealSeamProbeTests(unittest.TestCase):
    """The probe must read the fields the relay actually builds, not a stand-in shape."""

    def _relay(self, budget: CommBudget, policy_name: str = "immediate") -> CommRelay:
        writer = RecordingWriter()
        registry = TaskRegistry()
        relay = CommRelay(
            subscribers=("rescue_1", "rescue_2", "rescue_3"),
            writer=writer,
            snapshot_of=registry.get,
            accepted_of=registry.was_accepted,
            delivery_probability=1.0,
            delay_ticks=0,
            seed=5,
            policy=build_comm_policy(policy_name),
            budget=budget,
            timing_mode="policy_owned_timing",
            now_of=lambda: 1.0,
            freshness_of=writer.belief_freshness,
        )
        for index in (1, 2):
            event = detection(f"target-{index}")
            registry.handle_detection(event)
            relay.handle_detection(event)
        return relay

    def _relay_with(self, budget: CommBudget, events: list[Any]) -> CommRelay:
        writer = RecordingWriter()
        registry = TaskRegistry()
        relay = CommRelay(
            subscribers=("rescue_1", "rescue_2", "rescue_3"),
            writer=writer,
            snapshot_of=registry.get,
            accepted_of=registry.was_accepted,
            delivery_probability=1.0,
            delay_ticks=0,
            seed=5,
            policy=build_comm_policy("immediate"),
            budget=budget,
            timing_mode="policy_owned_timing",
            now_of=lambda: 1.0,
            freshness_of=writer.belief_freshness,
        )
        for event in events:
            registry.handle_detection(event)
            relay.handle_detection(event)
        return relay

    def test_probe_sees_the_relays_own_price_and_payability(self) -> None:
        charge_probe = self._relay(CommBudget(bytes_per_episode=10**6))
        probe = pressure_env.CandidateProbe(charge_probe.policy)
        charge_probe._policy = probe
        probe.decide(charge_probe._observation())
        summary = probe.summary()
        self.assertEqual(summary["visible_total"], 2, "two detected targets are two candidates")
        self.assertEqual(summary["payable_total"], 2, "an unspent credit can pay both")
        self.assertEqual(summary["competition_points"], 0)
        self.assertGreater(summary["visible_total"], 0)

    def test_a_tight_credit_turns_one_candidate_into_a_competition_point(self) -> None:
        """Two candidates, exactly one payable, priced by the transport itself.

        The prices are read out of an unbounded relay first, so the test cannot pass by assuming a
        packet size, and the cheaper message comes from a shorter stage list rather than from a
        number written here.
        """
        cheap = detection("target-1", stages=(TaskStageSpec((1, 0, 0, 0, 0), 3.0),))
        dear = detection(
            "target-2",
            stages=(
                TaskStageSpec((1, 1, 1, 1, 1), 3.0),
                TaskStageSpec((1, 1, 1, 1, 1), 3.0),
                TaskStageSpec((0, 1, 1, 1, 0), 2.0),
            ),
        )
        priced = self._relay_with(CommBudget(bytes_per_episode=10**6), [cheap, dear])
        prices = {
            str(entry.target_id): int(entry.attempt_charge_bytes)
            for entry in priced._observation().items
        }
        self.assertEqual(len(prices), 2, "the fixture needs two candidates in view")
        low, high = min(prices.values()), max(prices.values())
        self.assertLess(low, high, f"differently priced candidates are required, got {prices}")

        tight = self._relay_with(CommBudget(bytes_per_episode=high - 1), [cheap, dear])
        probe = pressure_env.CandidateProbe(tight.policy)
        probe.decide(tight._observation())
        summary = probe.summary()
        self.assertEqual(summary["visible_total"], 2)
        self.assertEqual(summary["payable_total"], 1)
        self.assertEqual(summary["competition_points"], 1, "the credit now forces a choice")
        self.assertEqual(summary["competition_points_credit_cause"], 1)
        self.assertEqual(summary["competition_points_capacity_only"], 0)
        sample = summary["competition_sample"][0]["candidates"]
        self.assertEqual(sorted(bool(row[2]) for row in sample), [False, True])

    def test_send_item_really_carries_the_fields_the_probe_reads(self) -> None:
        relay = self._relay(CommBudget(bytes_per_episode=10**6))
        for entry in relay._observation().items:
            self.assertIsNotNone(entry.attempt_charge_bytes)
            self.assertIn(entry.send_block_reason, ("allowed", "budget_exhausted", "capacity_step"))
            self.assertIsInstance(entry.send_executable, bool)


def _table(on_time: dict[int, float], bytes_: dict[int, float], **extra: Any) -> dict[str, Any]:
    base = {
        "role": "competitor",
        "per_stream_on_time": on_time,
        "per_stream_bytes": bytes_,
        "on_time_per_episode": sum(on_time.values()) / len(on_time),
        "bytes_per_episode": sum(bytes_.values()) / len(bytes_),
        "on_time_rate_of_detected": 0.6,
        "competition_streams": 0,
        "credit_used_ge_90_streams": 0,
    }
    base.update(extra)
    return base


def _report(arms: dict[str, dict[str, Any]], envelope_counts: dict[str, int], ci: list[float], gain: float, frontier: list[dict[str, Any]], pairs: list[str]) -> dict[str, Any]:
    best = max(
        (arm for arm, table in arms.items() if table["role"] == "competitor"),
        key=lambda arm: (arms[arm]["on_time_per_episode"], -arms[arm]["bytes_per_episode"]),
    )
    return {
        "best_fixed_rule": {"arm": best},
        "arms": arms,
        "hindsight_envelope": {
            "definition": "synthetic",
            "mean_on_time": sum(arms[best]["per_stream_on_time"].values()) / 16 + gain,
            "unique_winner_counts": envelope_counts,
        },
        "incomparable_bidirectional_pairs": pairs,
        "mean_pareto_frontier": frontier,
        "bootstrap": {
            "on_time_per_episode": {
                "always_hold": {"arms": {best: {"mean_difference": gain, "ci95": ci, "pairs": 16}}},
                "best_fixed": {"arms": {"envelope": {"mean_difference": gain, "ci95": ci}}},
            }
        },
    }


class GateBoundaryTests(unittest.TestCase):
    def _metrics(self, ratios: list[float]) -> dict[str, Any]:
        return {
            "mean_detected_over_placed": sum(ratios) / len(ratios),
            "worst_stream_detected_over_placed": min(ratios),
            "worst_stream": "0",
            "illegal_streams": [],
            "per_stream": {},
        }

    def test_l1_boundary_is_inclusive(self) -> None:
        self.assertTrue(judge_layers.l1(self._metrics([0.75] * 16))["passed"])
        self.assertFalse(judge_layers.l1(self._metrics([0.7499] * 16))["passed"])
        self.assertFalse(judge_layers.l1({**self._metrics([1.0] * 16), "illegal_streams": ["7"]})["passed"])

    def _arms_for_l2(self, rate: float) -> dict[str, dict[str, Any]]:
        seeds = {seed: float(seed) for seed in range(16)}
        arms = {
            "always_hold": _table(seeds, seeds),
            "send_once": {**_table(seeds, seeds), "on_time_rate_of_detected": rate},
        }
        arms["always_hold"]["role"] = "control"
        return arms

    def test_l2_needs_a_positive_ci_floor_and_a_band(self) -> None:
        report = _report(self._arms_for_l2(0.6), {}, [0.0, 1.0], 0.5, [], [])
        self.assertFalse(judge_layers.l2(report)["passed"], "CI lower exactly 0 is not > 0")
        report = _report(self._arms_for_l2(0.6), {}, [0.001, 1.0], 0.5, [], [])
        self.assertTrue(judge_layers.l2(report)["passed"])
        for rate in (0.39, 0.91):
            too_extreme = _report(self._arms_for_l2(rate), {}, [0.001, 1.0], 0.5, [], [])
            self.assertFalse(judge_layers.l2(too_extreme)["passed"], f"rate {rate} is outside the band")
        for edge in (0.40, 0.90):
            at_edge = _report(self._arms_for_l2(edge), {}, [0.001, 1.0], 0.5, [], [])
            self.assertTrue(judge_layers.l2(at_edge)["passed"], f"{edge} is inside the closed band")

    def _l3_arms(self, competition: int, heavy: int) -> dict[str, dict[str, Any]]:
        seeds = {seed: 1.0 for seed in range(16)}
        arms = {
            "send_once": {
                **_table(seeds, seeds),
                "competition_streams": competition,
                "credit_used_ge_90_streams": heavy,
                "per_stream_competition_points": {seed: 0 for seed in range(16)},
                "joint_only_points_total": 0,
                "payable_candidate_points_total": 0,
                "credit_exhausted_with_candidates_streams": 0,
                "mean_visible_at_competition": 0,
                "mean_payable_at_competition": 0,
            }
        }
        return arms

    def test_l3_counts_streams_not_points(self) -> None:
        self.assertFalse(judge_layers.l3(_report(self._l3_arms(7, 8), {}, [1, 2], 1, [], []))["passed"])
        self.assertFalse(judge_layers.l3(_report(self._l3_arms(8, 3), {}, [1, 2], 1, [], []))["passed"])
        self.assertTrue(judge_layers.l3(_report(self._l3_arms(8, 4), {}, [1, 2], 1, [], []))["passed"])

    def test_l4_needs_all_four_conditions(self) -> None:
        seeds = {seed: 1.0 for seed in range(16)}
        arms = {"send_once": _table(seeds, seeds)}
        frontier = [
            {"arm": "send_once", "on_time_per_episode": 5.0, "bytes_per_episode": 10_000},
            {"arm": "event", "on_time_per_episode": 5.3, "bytes_per_episode": 13_000},
        ]
        good = _report(
            {**arms, "event": _table(seeds, seeds, role="competitor")},
            {"send_once": 3, "event": 4},
            [0.10, 0.5],
            0.30,
            frontier,
            ["event vs send_once"],
        )
        self.assertTrue(judge_layers.l4(good)["passed"])
        missing_winners = {**good, "hindsight_envelope": {**good["hindsight_envelope"], "unique_winner_counts": {"send_once": 3}}}
        self.assertFalse(judge_layers.l4(missing_winners)["passed"])
        no_bidirectional = {**good, "incomparable_bidirectional_pairs": []}
        self.assertFalse(judge_layers.l4(no_bidirectional)["passed"])
        one_point = {**good, "mean_pareto_frontier": frontier[:1]}
        self.assertFalse(judge_layers.l4(one_point)["passed"])
        small_gaps = {
            **good,
            "mean_pareto_frontier": [
                {"arm": "send_once", "on_time_per_episode": 5.0, "bytes_per_episode": 10_000},
                {"arm": "event", "on_time_per_episode": 5.1, "bytes_per_episode": 12_000},
            ],
        }
        self.assertFalse(judge_layers.l4(small_gaps)["passed"])
        ci_touching_zero = {
            **good,
            "bootstrap": {
                "on_time_per_episode": {
                    "always_hold": {"arms": {"send_once": {"mean_difference": 1.0, "ci95": [0.0, 2.0]}}},
                    "best_fixed": {"arms": {"envelope": {"mean_difference": 0.3, "ci95": [0.0, 0.6]}}},
                }
            },
        }
        self.assertFalse(judge_layers.l4(ci_touching_zero)["passed"])

    def test_byte_gap_threshold_is_ten_percent_of_the_credit(self) -> None:
        self.assertEqual(judge_layers.L4_MIN_FRONTIER_BYTE_GAP, common.BUDGET_BYTES // 10)


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seeds = {seed: 0.0 for seed in range(16)}

    def test_best_fixed_breaks_ties_on_bytes_then_on_protocol_order(self) -> None:
        arms = {
            "event": _table({s: 5.0 for s in range(16)}, {s: 20_000.0 for s in range(16)}),
            "send_once": _table({s: 5.0 for s in range(16)}, {s: 20_000.0 for s in range(16)}),
            "periodic_5": _table({s: 5.0 for s in range(16)}, {s: 19_000.0 for s in range(16)}),
            "always_hold": _table(
                {s: 9.0 for s in range(16)}, {s: 0.0 for s in range(16)}, role="control"
            ),
        }
        self.assertEqual(pressure_metrics.best_fixed_rule(arms), "periodic_5")
        del arms["periodic_5"]
        self.assertEqual(
            pressure_metrics.best_fixed_rule(arms),
            "send_once",
            "a tie must go to the protocol's row order, not to dict order",
        )
        self.assertNotIn("always_hold", [pressure_metrics.best_fixed_rule(arms)])

    def test_envelope_picks_the_per_stream_best_and_counts_unique_winners(self) -> None:
        arms = {
            "send_once": _table(
                {0: 6.0, 1: 4.0, 2: 5.0, 3: 5.0}, {0: 9_000.0, 1: 9_000.0, 2: 9_000.0, 3: 8_000.0}
            ),
            "event": _table(
                {0: 6.0, 1: 5.0, 2: 5.0, 3: 5.0}, {0: 9_000.0, 1: 9_000.0, 2: 9_000.0, 3: 8_500.0}
            ),
        }
        env = pressure_metrics.envelope(arms)
        self.assertEqual(env["per_stream"]["1"]["winners"], ["event"], "one more task on time")
        self.assertEqual(
            sorted(env["per_stream"]["0"]["winners"]),
            ["event", "send_once"],
            "same effect and same cost is a tie, not a winner",
        )
        self.assertFalse(env["per_stream"]["0"]["unique_winner"])
        self.assertEqual(sorted(env["per_stream"]["2"]["winners"]), ["event", "send_once"])
        self.assertEqual(env["per_stream"]["3"]["winners"], ["send_once"], "the cheaper tie-break")
        self.assertEqual(env["unique_winner_counts"], {"event": 1, "send_once": 1})
        self.assertEqual(env["mean_on_time"], 5.25)

    def test_a_tie_on_both_axes_is_not_a_unique_winner(self) -> None:
        arms = {
            "send_once": _table({0: 5.0}, {0: 9_000.0}),
            "event": _table({0: 5.0}, {0: 9_000.0}),
        }
        env = pressure_metrics.envelope(arms)
        self.assertEqual(sorted(env["per_stream"]["0"]["winners"]), ["event", "send_once"])
        self.assertFalse(env["per_stream"]["0"]["unique_winner"])
        self.assertEqual(env["unique_winner_counts"], {})

    def test_pareto_excludes_dominated_points_and_keeps_equal_effects(self) -> None:
        front = pressure_metrics.pareto_frontier(
            {
                "a": _table(self.seeds, self.seeds, role="competitor", on_time_per_episode=5.0, bytes_per_episode=10_000),
                "b": _table(self.seeds, self.seeds, role="competitor", on_time_per_episode=5.0, bytes_per_episode=12_000),
                "c": _table(self.seeds, self.seeds, role="competitor", on_time_per_episode=6.0, bytes_per_episode=20_000),
                "hold": _table(self.seeds, self.seeds, role="control"),
            }
        )
        self.assertEqual([point["arm"] for point in front], ["a", "c"], "b is dominated by a")

    def test_lambda_is_reported_undefined_rather_than_dropped(self) -> None:
        slopes = pressure_metrics.lambda_slopes(
            [
                {"arm": "a", "on_time_per_episode": 5.0, "bytes_per_episode": 10_000},
                {"arm": "b", "on_time_per_episode": 5.0, "bytes_per_episode": 10_000},
                {"arm": "c", "on_time_per_episode": 6.0, "bytes_per_episode": 19_300},
            ],
            common.BUDGET_BYTES,
        )
        self.assertEqual(slopes[0]["lambda_break_even"], "undefined")
        self.assertAlmostEqual(
            slopes[1]["lambda_break_even"], 1.0 / (9_300 / common.BUDGET_BYTES), places=3
        )
        self.assertEqual(len(slopes), 2, "the undefined pair stays in the report")

    def test_bidirectional_requires_three_streams_each_side(self) -> None:
        left = {seed: 1.0 for seed in range(16)}
        right = dict(left)
        for seed in (0, 1, 2):
            left[seed] = 2.0
        for seed in (3, 4):
            right[seed] = 2.0
        arms = {
            "send_once": _table(left, left),
            "event": _table(right, right),
        }
        dominance = pressure_metrics.per_stream_dominance(arms)
        entry = dominance["event vs send_once"]
        self.assertEqual(entry["left_strictly_more_on_time"], 2)
        self.assertEqual(entry["right_strictly_more_on_time"], 3)
        self.assertFalse(entry["bidirectional"])


class StatisticsTests(unittest.TestCase):
    def _maps(self) -> dict[str, dict[int, float]]:
        seeds = list(range(16))
        return {
            "always_hold": {seed: 0.0 for seed in seeds},
            "send_once": {seed: float(seed % 5) for seed in seeds},
            "event": {seed: float(seed % 5 + 1) for seed in seeds},
            "envelope": {seed: float(seed % 5 + 1) for seed in seeds},
            "best_fixed": {seed: float(seed % 5 + 1) for seed in seeds},
        }

    def test_a_statistic_has_exactly_one_confidence_interval(self) -> None:
        first = common.paired_differences(self._maps(), baseline="always_hold")
        again = common.paired_differences(self._maps(), baseline="always_hold")
        self.assertEqual(first["arms"], again["arms"], "identical input must redraw identical bounds")
        self.assertEqual(first["label_order"], list(common.BOOTSTRAP_LABEL_ORDER))
        self.assertEqual(first["baseline"], "always_hold")
        self.assertEqual(first["arms"]["send_once"]["pairs"], 16)

    def test_a_subset_row_set_would_move_the_bounds_and_is_therefore_forbidden(self) -> None:
        whole = common.paired_differences(self._maps(), baseline="always_hold")
        narrowed = {
            label: {seed: value for seed, value in table.items() if seed != 3}
            for label, table in self._maps().items()
        }
        partial = common.paired_differences(narrowed, baseline="always_hold")
        self.assertNotEqual(
            whole["arms"]["send_once"]["ci95"], partial["arms"]["send_once"]["ci95"],
            "this test documents why one draw per (axis, baseline) is required",
        )

    def test_missing_baseline_fails_rather_than_silently_returning_nothing(self) -> None:
        with self.assertRaises(KeyError):
            common.paired_differences(self._maps(), baseline="not_present")


class SeedAuditTests(unittest.TestCase):
    def test_the_audit_audits_exactly_the_rounds_blocks(self) -> None:
        self.assertEqual(
            {goals: (seeds[0], seeds[-1] + 1) for goals, seeds in seed_audit.BLOCKS.items()},
            {goals: (seeds[0], seeds[-1] + 1) for goals, seeds in common.LAYERS.items()},
            "an audit of a different block than the round runs would be worthless",
        )

    def test_seed_patterns_are_recognised_in_names_and_text(self) -> None:
        self.assertEqual(seed_audit.SEED_IN_NAME.search("ct_send_once_s1000000.jsonl").group(1), "1000000")
        self.assertEqual(seed_audit.STREAM_IN_NAME.match("s1000315.json").group(1), "1000315")
        self.assertIsNone(seed_audit.SEED_IN_NAME.search("notes.txt"))
        self.assertEqual(seed_audit.BARE_SEED.findall("1000100..1000115"), ["1000100", "1000115"])
        self.assertEqual(seed_audit.BARE_SEED.findall("21000100"), [])

    def test_declared_blocks_are_checked_arithmetically(self) -> None:
        original = dict(seed_audit.DECLARED_BLOCKS)
        try:
            seed_audit.DECLARED_BLOCKS["synthetic collision"] = (1_000_100, 1_000_102)
            conflicts = seed_audit.declared_range_conflicts()
            self.assertTrue(any(item["goals"] == 8 for item in conflicts))
            self.assertTrue(all(item["count"] for item in conflicts))
        finally:
            del seed_audit.DECLARED_BLOCKS["synthetic collision"]
            seed_audit.DECLARED_BLOCKS.update(original)
        self.assertEqual(seed_audit.declared_range_conflicts(), [])

    def test_two_files_with_the_same_seed_number_are_a_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "s1000000.json"
            second = Path(directory) / "other_s1000000.json"
            first.write_text("{}", encoding="utf-8")
            second.write_text("{\"a\": 1}", encoding="utf-8")
            collisions = seed_audit.stream_content_collisions(
                {1_000_000}, {1_000_000: [first, second]}
            )
            self.assertIn("1000000", collisions)
            second.write_text("{}", encoding="utf-8")
            self.assertEqual(
                seed_audit.stream_content_collisions({1_000_000}, {1_000_000: [first, second]}), {}
            )

    def test_only_the_brief_itself_may_name_a_block_without_conflicting(self) -> None:
        self.assertEqual(
            seed_audit.PRE_REGISTRATION_FILES, ("COMM_TASK_PRESSURE_LEARNABILITY_NEXT_ROUND.md",)
        )
        self.assertNotIn("PROJECT_CHARTER.md", seed_audit.PRE_REGISTRATION_FILES)


class RecorderLegalityTests(unittest.TestCase):
    def _facts(self, **overrides: Any) -> dict[str, Any]:
        facts = {
            "placed": 8,
            "configured_goals": 8,
            "goal_cells": [[1, 2], [3, 4]],
            "goal_cells_distinct": True,
            "goal_cells_on_agent_start": [],
            "map_sha256_reproduced": True,
            "detected": 6,
            "steps_recorded": common.STREAM_STEPS,
        }
        facts.update(overrides)
        return facts

    def test_each_failure_mode_is_named(self) -> None:
        cases = {
            "placed": 7,
            "goal_cells_distinct": False,
            "goal_cells_on_agent_start": [[1, 2]],
            "map_sha256_reproduced": False,
            "steps_recorded": 999,
        }
        for key, value in cases.items():
            with self.subTest(key=key):
                problems = record_layer.legality(self._facts(**{key: value}))
                self.assertTrue(problems, f"{key} must make the stream illegal")
        self.assertEqual(record_layer.legality(self._facts()), [])

    def test_more_detections_than_targets_is_impossible_and_rejected(self) -> None:
        self.assertTrue(record_layer.legality(self._facts(detected=9)))

    def test_recording_config_may_only_move_num_goals(self) -> None:
        for goals in common.LAYER_ORDER:
            report = common.assert_only_num_goals_changed(goals)
            self.assertTrue(set(report["changed"]) <= {"num_goals"})
            self.assertEqual(report["world_conflicts"], {})

    def test_a_second_field_change_is_refused(self) -> None:
        original = dict(common.RECORD_CONFIGURATION)
        try:
            common.RECORD_CONFIGURATION["search_agents"] = 4
            with self.assertRaises(RuntimeError):
                common.assert_only_num_goals_changed(8)
        finally:
            common.RECORD_CONFIGURATION.clear()
            common.RECORD_CONFIGURATION.update(original)


class MatrixHygieneTests(unittest.TestCase):
    def _write(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_duplicate_cells_are_deduplicated_only_when_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_rows, original_layer = pressure_matrix.rows_path, common.layer_rows
            try:
                common.layer_rows = lambda goals: root  # type: ignore[assignment]
                pressure_matrix.rows_path = lambda goals, shard: root / f"rows_g{goals}_shard{shard}.jsonl"
                row = {"arm": "send_once", "seed": 7, "on_time": 3}
                self._write(root / "rows_g6_shard0.jsonl", [row])
                self._write(root / "rows_g6_shard1.jsonl", [row])
                self.assertEqual(len(pressure_matrix.load_rows(6)), 1)
                self._write(root / "rows_g6_shard2.jsonl", [{**row, "on_time": 4}])
                with self.assertRaises(RuntimeError):
                    pressure_matrix.load_rows(6)
            finally:
                pressure_matrix.rows_path = original_rows  # type: ignore[assignment]
                common.layer_rows = original_layer  # type: ignore[assignment]

    def test_sharding_partitions_the_cells_exactly(self) -> None:
        cells = pressure_matrix.cells(6)
        shards = [common.shard_of(cells, 4, index) for index in range(4)]
        self.assertEqual(sum(len(shard) for shard in shards), len(cells))
        self.assertEqual(len({cell for shard in shards for cell in shard}), len(cells))

    def test_a_cell_key_is_arm_and_seed_not_arm_alone(self) -> None:
        cells = pressure_matrix.cells(6)
        self.assertEqual(len(cells), 9 * 16)
        self.assertEqual(len({arm for arm, _ in cells}), 9, "every arm must run on every stream")

    def test_layer_paths_are_isolated_so_two_layers_cannot_share_rows(self) -> None:
        names = {goals: str(common.layer_rows(goals)) for goals in common.LAYER_ORDER}
        self.assertEqual(len(set(names.values())), len(names))


class StopRuleTests(unittest.TestCase):
    def _gates(self, **passed: bool) -> list[dict[str, Any]]:
        order = ("L0", "L1", "L2", "L3", "L4")
        return [
            {"gate": name, "passed": bool(passed.get(name, True))} for name in order
        ]

    def test_stop_rules_follow_the_protocol_table(self) -> None:
        cases = {
            "everything passes": (self._gates(), (True, "all passed")),
            "L0 fails": (self._gates(L0=False), (True, "L0")),
            "L1 fails": (self._gates(L1=False), (True, "L1")),
            "L4 fails only": (self._gates(L4=False), (False, "")),
            "L2 fails only": (self._gates(L2=False), (False, "")),
        }
        for label, (gates, expected) in cases.items():
            with self.subTest(label):
                stopped, reason = judge_layers.stop_reading(gates)
                self.assertEqual(stopped, expected[0])
                if expected[1]:
                    self.assertIn(expected[1].lower(), reason.lower())
                else:
                    self.assertIn("next layer", reason.lower())


class FrozenSettingTests(unittest.TestCase):
    def test_credit_is_the_locked_25_percent_tier(self) -> None:
        self.assertEqual(common.BUDGET_BYTES, 29_300)
        self.assertEqual(common.TIER, "25pct")

    def test_arms_roles_and_table_membership(self) -> None:
        self.assertEqual(len(common.ARM_ORDER), 9)
        self.assertEqual(len(common.COMPETITION_ARMS), 7)
        for arm in common.ARM_ORDER:
            spec = common.arm_spec(arm)
            self.assertEqual(spec["role"], common.ARM_ROLES[arm])
            self.assertEqual(spec["unbudgeted"], arm == "event_unbudgeted")
        self.assertEqual(common.ARM_ROLES["always_hold"], "control")

    def test_no_arm_is_random_or_unlisted(self) -> None:
        self.assertNotIn("random_send", common.ARM_ORDER)
        with self.assertRaises(ValueError):
            common.arm_spec("immediate")

    def test_the_frozen_channel_and_ack_state(self) -> None:
        self.assertEqual(common.DELIVERY_PROBABILITY, 1.0)
        self.assertFalse(common.ACK_ENABLED)
        self.assertEqual(common.INFORMATION, "isolated")
        self.assertEqual(common.TIMING_MODE, "policy_owned_timing")

    def test_production_defaults_are_untouched(self) -> None:
        report = common.production_defaults_ok()
        self.assertTrue(report["ok"], report)
        self.assertFalse(report["comm_ack"])
        self.assertEqual(report["comm_timing_mode"], "legacy_shared_bound")
        self.assertEqual(report["num_goals"], 6)

    def test_the_sealed_learning_block_is_still_empty(self) -> None:
        scan = common.sealed_block_artefacts()
        self.assertEqual(scan["count"], 0, scan["artefacts"][:3])

    def test_layers_are_disjoint_and_ordered(self) -> None:
        self.assertEqual(list(common.LAYERS), [6, 8, 10, 12])
        blocks = [seed for seeds in common.LAYERS.values() for seed in seeds]
        self.assertEqual(len(blocks), 64)
        self.assertEqual(len(set(blocks)), 64, "two layers may not share a stream seed")
        for goals, seeds in common.LAYERS.items():
            self.assertEqual(len(seeds), 16, f"layer {goals}")

    def test_launch_ceiling_matches_the_briefs_arithmetic(self) -> None:
        self.assertEqual(common.MATRIX_LAUNCH_CAP, 4 * (16 + 16 * 9))
        self.assertEqual(common.ledger_totals()["matrix_cap"], common.MATRIX_LAUNCH_CAP)

    def test_the_ledger_refuses_to_start_past_its_own_ceiling(self) -> None:
        """Two categories, two ceilings, one count shared across processes.

        The fixture replaces the cross-process total rather than the local file, because the point
        of the design is that a shard cannot see how much the rest of the round has already spent.
        """
        state = {"matrix": 0, "gate": 0}
        real = common.ledger_totals
        common.ledger_totals = lambda: {  # type: ignore[assignment]
            "matrix_starts": state["matrix"],
            "gate_starts": state["gate"],
            "files": ["test"],
        }
        path = common.HERE / "LAUNCH_LEDGER_test_ceiling.json"
        try:
            ledger = common.LaunchLedger(path.name)
            for index in range(common.GATE_LAUNCH_CAP):
                ledger.reserve("gate", 6, 1_000_000 + index)
                state["gate"] += 1
            with self.assertRaises(RuntimeError):
                ledger.reserve("gate", 6, 1_000_999)
            state["matrix"] = common.MATRIX_LAUNCH_CAP
            with self.assertRaises(RuntimeError):
                ledger.reserve("matrix", 6, 1_000_000, arm="send_once")
            self.assertLess(
                ledger.starts,
                common.MATRIX_LAUNCH_CAP,
                "the local file counts one process; the ceiling reads every process",
            )
            with self.assertRaises(ValueError):
                ledger.reserve("training", 6, 1_000_000)
        finally:
            common.ledger_totals = real  # type: ignore[assignment]
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
