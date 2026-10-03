from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
R40_OUTPUT = Path(
    r"D:\code\整合\zhenghe-round40\runs\comm_first_report_training_ablation_20261003"
)
STAGE_B = ROOT / "runs" / "comm_cost_stage_b_20260927"
MASK_FIX = ROOT / "runs" / "comm_budget_action_mask_fix_20260922"
TIGHT = ROOT / "runs" / "comm_tight_budget_20260919"
for path in (str(ROOT), str(STAGE_B), str(MASK_FIX), str(TIGHT), str(R40_OUTPUT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from integrated_sar.comm_learning import features  # noqa: E402
from integrated_sar.comm_policy import SendItem, SendObservation  # noqa: E402
from r40_eligibility import FirstReportOnlyCollector  # noqa: E402
from r41_hazard_median import (  # noqa: E402
    HOLD,
    SEND,
    TAU,
    HazardMedianCollector,
    HazardMedianState,
)


class FakeController:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = list(probabilities)
        self.forward_inputs: list[torch.Tensor] = []
        self.parameter_identity = "constant-test-controller"

    def forward(self, matrix: torch.Tensor):
        self.forward_inputs.append(matrix.detach().clone())
        if len(matrix) != len(self.probabilities):
            raise AssertionError((len(matrix), self.probabilities))
        probs = torch.tensor(self.probabilities, dtype=torch.float32)
        logits = torch.stack((torch.log1p(-probs), torch.log(probs)), dim=1)
        return logits, torch.zeros((len(matrix),), dtype=torch.float32)

    def cost_values(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.zeros((len(matrix),), dtype=torch.float32)

    def parameter_summary(self) -> str:
        return self.parameter_identity


def item(target: str, *, first: bool = True, revision: int = 1,
         charge: int = 10, executable: bool = True,
         block: str = "allowed") -> SendItem:
    return SendItem(
        target_id=target,
        ready_step=1,
        age_steps=0,
        pending_event=True,
        changed_view=True,
        first_emission=first,
        revision=revision,
        revision_delta=0,
        stage_index=0,
        stage_changed=False,
        progress=0.0,
        progress_delta=0.0,
        requirements_changed=False,
        members_changed=False,
        deadline=100.0,
        remaining_time=100.0,
        sent_count=0 if first else 1,
        last_sent_step=None if first else 0,
        since_last_emission_time=None if first else 1.0,
        has_future_requirements=False,
        attempt_charge_bytes=charge,
        send_executable=executable,
        send_block_reason=block,
    )


def observation(tick: int, candidates: list[SendItem], *, budget: int = 100,
                remaining: int | None = None, capacity: int | None = 4) -> SendObservation:
    return SendObservation(
        step=tick,
        current_time=float(tick),
        items=tuple(candidates),
        remaining_budget_bytes=float(budget if remaining is None else remaining),
        budget_bytes_per_episode=budget,
        capacity_remaining=capacity,
        backlog_targets=len(candidates),
        subscribers=1,
    )


def collect(collector: HazardMedianCollector, obs: SendObservation) -> tuple[str, ...]:
    return collector.decide(obs)


class HazardMathTests(unittest.TestCase):
    def test_A_disabled_decoder_matches_round40_greedy(self) -> None:
        obs = observation(4, [item("a"), item("b")])
        old = FirstReportOnlyCollector(FakeController([0.35, 0.8]), greedy=True)
        compat = HazardMedianCollector(FakeController([0.35, 0.8]), enabled=False)
        old_asked = old.decide(obs)
        compat_asked = compat.decide(obs)
        old_tape, new_tape = old.records[-1], compat.records[-1]
        self.assertEqual(old_asked, compat_asked)
        self.assertEqual(old_tape["targets"], new_tape["targets"])
        self.assertEqual(old_tape["actions"], new_tape["actions"])
        self.assertEqual(old_tape["features"], new_tape["features"])
        self.assertEqual(old_tape["send_probabilities"], new_tape["send_probabilities"])
        self.assertEqual(old_tape["shadow_budget_after"], new_tape["shadow_budget_after"])

    def test_B_hand_calculated_035_and_01_sequences(self) -> None:
        state = HazardMedianState()
        actions = [state.decide("x", tick, 0.35, eligible=True, first_emission=True)
                   ["effective_action"] for tick in (1, 2)]
        self.assertEqual(actions, [HOLD, SEND])
        self.assertGreater(0.65, 0.5)
        self.assertLessEqual(0.65 ** 2, 0.5)
        self.assertAlmostEqual(state.value("x"), -math.log(0.65 ** 2), places=14)

        state = HazardMedianState()
        actions = [state.decide("y", tick, 0.1, eligible=True, first_emission=True)
                   ["effective_action"] for tick in range(1, 8)]
        self.assertEqual(actions, [HOLD] * 6 + [SEND])
        self.assertGreater(0.9 ** 6, 0.5)
        self.assertLessEqual(0.9 ** 7, 0.5)
        self.assertAlmostEqual(state.value("y"), -math.log(0.9 ** 7), places=14)

    def test_C_hazard_matches_independent_survival_product(self) -> None:
        probabilities = [0.03, 0.12, 0.0, 0.21, 0.07, 0.18]
        survival = 1.0
        state = HazardMedianState()
        for tick, p in enumerate(probabilities, start=1):
            survival *= 1.0 - p
            state.decide("z", tick, p, eligible=True, first_emission=True)
            self.assertAlmostEqual(state.value("z"), -math.log(survival), places=13)

    def test_D_masked_budget_and_capacity_opportunities_do_not_update_H(self) -> None:
        by_budget = HazardMedianCollector(FakeController([0.9]))
        collect(by_budget, observation(
            1, [item("budget", charge=101)], budget=100, remaining=100, capacity=2
        ))
        self.assertEqual(by_budget.hazard_state.value("budget"), 0.0)
        self.assertFalse(by_budget.records[-1]["hazard_audit"]["budget"]["H_updated"])

        by_capacity = HazardMedianCollector(FakeController([0.9, 0.9]))
        collect(by_capacity, observation(
            1, [item("fills_capacity"), item("capacity")],
            budget=100, remaining=100, capacity=1,
        ))
        self.assertEqual(by_capacity.hazard_state.value("capacity"), 0.0)
        self.assertEqual(by_capacity.records[-1]["masked"], ["capacity"])

    def test_E_targets_are_independent_and_candidate_order_is_preserved(self) -> None:
        collector = HazardMedianCollector(FakeController([0.35, 0.1]))
        first = observation(1, [item("b"), item("a")])
        collect(collector, first)
        self.assertEqual(collector.records[-1]["targets"], ["b", "a"])
        self.assertAlmostEqual(collector.hazard_state.value("b"), -math.log(0.65))
        self.assertAlmostEqual(collector.hazard_state.value("a"), -math.log(0.9))
        collector.controller.probabilities = [0.35, 0.1]
        collect(collector, observation(2, [item("b"), item("a")]))
        self.assertEqual(collector.records[-1]["requested"], {"b": True, "a": False})
        self.assertAlmostEqual(collector.hazard_state.value("a"), -math.log(0.9 ** 2))

    def test_F_duplicate_tick_revision_and_episode_reset(self) -> None:
        collector = HazardMedianCollector(FakeController([0.35]))
        collect(collector, observation(7, [item("same")]))
        before = collector.hazard_state.value("same")
        collector.controller.probabilities = [0.99]
        collect(collector, observation(7, [item("same", revision=2)]))
        self.assertEqual(collector.hazard_state.value("same"), before)
        self.assertEqual(collector.records[-1]["hazard_audit"]["same"]["duplicate_tick"], True)
        collector.controller.probabilities = [0.35]
        collect(collector, observation(8, [item("same", revision=3)]))
        self.assertAlmostEqual(collector.hazard_state.value("same"), -math.log(0.65 ** 2))
        fresh_episode = HazardMedianCollector(FakeController([0.35]))
        collect(fresh_episode, observation(7, [item("same", revision=4)]))
        self.assertAlmostEqual(fresh_episode.hazard_state.value("same"), -math.log(0.65))

    def test_G_real_emission_closes_but_refusal_does_not(self) -> None:
        refused = HazardMedianCollector(FakeController([0.8]))
        self.assertEqual(collect(refused, observation(1, [item("refused")])), ("refused",))
        # No real send: the relay still marks this target first_emission on the next tick.
        self.assertEqual(collect(refused, observation(2, [item("refused", revision=2)])),
                         ("refused",))
        self.assertTrue(refused.records[-1]["hazard_audit"]["refused"]["H_updated"])
        self.assertFalse(refused.records[-1]["hazard_audit"]["refused"]["closed_by_real_emission"])

        sent = HazardMedianCollector(FakeController([0.8]))
        self.assertEqual(collect(sent, observation(1, [item("sent")])), ("sent",))
        self.assertEqual(collect(sent, observation(2, [item("sent", first=False)])), ())
        self.assertEqual(sent.records[-1]["actions"]["sent"], HOLD)
        self.assertTrue(sent.records[-1]["hazard_audit"]["sent"]["closed_by_real_emission"])

    def test_H_probability_boundaries_and_invalid_values(self) -> None:
        state = HazardMedianState()
        zero = state.decide("zero", 1, 0.0, eligible=True, first_emission=True)
        self.assertEqual(zero["H_after"], 0.0)
        self.assertEqual(zero["effective_action"], HOLD)
        one = state.decide("one", 1, 1.0, eligible=True, first_emission=True)
        self.assertEqual(one["effective_action"], SEND)
        self.assertEqual(one["H_after"], TAU)
        self.assertTrue(one["p_one_saturated"])
        self.assertTrue(math.isfinite(one["H_after"]))
        for bad in (float("nan"), float("inf"), -0.01, 1.01, True):
            with self.subTest(probability=bad):
                with self.assertRaises(ValueError):
                    state.decide("bad", 1, bad, eligible=True, first_emission=True)

    def test_I_only_legal_first_report_send_spends_shadow_budget(self) -> None:
        collector = HazardMedianCollector(FakeController([0.9, 0.9, 0.9]))
        obs = observation(1, [
            item("send", charge=10),
            item("masked", charge=100, revision=1),
            item("refresh", first=False, revision=2),
        ], budget=50, remaining=50, capacity=2)
        collect(collector, obs)
        tape = collector.records[-1]
        self.assertEqual(tape["requested"], {"send": True, "masked": False, "refresh": False})
        self.assertEqual(tape["shadow_budget_after"], 40.0)
        self.assertTrue(tape["candidate_audit"]["send"]["shadow_charge_applied"])
        self.assertFalse(tape["candidate_audit"]["masked"]["shadow_charge_applied"])
        self.assertFalse(tape["candidate_audit"]["refresh"]["shadow_charge_applied"])
        self.assertFalse(tape["hazard_audit"]["masked"]["H_updated"])
        self.assertFalse(tape["hazard_audit"]["refresh"]["H_updated"])

    def test_J_actor_features_parameters_and_rng_are_unchanged(self) -> None:
        controller = FakeController([0.35, 0.1])
        collector = HazardMedianCollector(controller)
        obs = observation(1, [item("b"), item("a")])
        original_features = [list(features(obs, candidate)) for candidate in obs.items]
        original_identity = controller.parameter_summary()
        rng_before = torch.get_rng_state().clone()
        collect(collector, obs)
        rng_after = torch.get_rng_state().clone()
        self.assertEqual(collector.records[-1]["features"], original_features)
        self.assertTrue(all(len(row) == 19 for row in collector.records[-1]["features"]))
        self.assertEqual(controller.parameter_summary(), original_identity)
        self.assertTrue(torch.equal(rng_before, rng_after))
        self.assertEqual(collector.records[-1]["log_prob"], 0.0)
        self.assertEqual(collector.records[-1]["action_log_prob_semantics"],
                         "not_a_sampling_likelihood")


if __name__ == "__main__":
    unittest.main(verbosity=2)
