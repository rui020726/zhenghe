"""ROUND41 deterministic cumulative-risk decoder for frozen first-report policies.

This module is an experiment-local collector.  It leaves the ROUND40 collector and all
production defaults untouched.  The external state is keyed only by target id and relay tick;
it never enters the controller feature matrix.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from budget_masked import HOLD, SEND, shadow_gate
from integrated_sar.comm_learning import features
from r40_eligibility import COMM_ACTION_ELIGIBILITY, FirstReportOnlyCollector

DECODER = "hazard_median"
DECODER_SCHEMA = "round41-cumulative-risk-decoder/1"
TAU = math.log(2.0)


class HazardMedianState:
    """Per-episode sender-side cumulative hazard, without actor or RNG state."""

    def __init__(self) -> None:
        self.tau = TAU
        self.hazard: dict[str, float] = {}
        self.closed_targets: set[str] = set()
        self.last_observed_tick: dict[str, int] = {}
        self.last_eligible_tick: dict[str, int] = {}
        self.last_eligible_probability: dict[str, float] = {}
        self.last_eligible_action: dict[str, int] = {}

    def value(self, target_id: str) -> float:
        return self.hazard.get(str(target_id), 0.0)

    def threshold_reached(self, target_id: str) -> bool:
        return self.value(target_id) >= self.tau

    def decide(
        self,
        target_id: str,
        relay_tick: int,
        probability_send: float,
        *,
        eligible: bool,
        first_emission: bool,
    ) -> dict[str, Any]:
        target = str(target_id)
        tick = int(relay_tick)
        if target in self.last_observed_tick and tick < self.last_observed_tick[target]:
            raise ValueError(f"relay tick moved backwards for {target}: {tick}")
        self.last_observed_tick[target] = tick
        before = self.value(target)

        # A real successful first emission is the only event that closes this state.  A
        # request/refusal pair leaves first_emission true and therefore stays eligible.
        if not first_emission:
            self.closed_targets.add(target)
        if target in self.closed_targets:
            return self._result(before, before, False, False, HOLD, False, None,
                                closed_by_real_emission=True)
        if not eligible:
            return self._result(before, before, False, before >= self.tau, HOLD, False,
                                None, closed_by_real_emission=False)

        p = _validated_probability(probability_send)
        if self.last_eligible_tick.get(target) == tick:
            # Idempotent for a duplicate collector call or a repeated/revised candidate in
            # this relay tick.  Reuse its first action and never accumulate twice.
            action = self.last_eligible_action[target]
            return self._result(
                before, before, False, before >= self.tau, action, True,
                None, closed_by_real_emission=False,
            )

        if p == 1.0:
            # The mathematical increment is +inf.  Saturate the finite stored H at tau and
            # retain the threshold flag through H>=tau; no JSON Infinity is produced.
            after = max(before, self.tau)
            increment: float | None = None
        else:
            increment = -math.log1p(-p)
            after = before + increment
            if not math.isfinite(after):
                raise OverflowError(f"finite probability produced non-finite H for {target}")

        reached = after >= self.tau
        action = SEND if reached else HOLD
        self.hazard[target] = after
        self.last_eligible_tick[target] = tick
        self.last_eligible_probability[target] = p
        self.last_eligible_action[target] = action
        return self._result(before, after, True, reached, action, False, increment,
                            closed_by_real_emission=False)

    @staticmethod
    def _result(before: float, after: float, updated: bool, reached: bool, action: int,
                duplicate_tick: bool, increment: float | None, *,
                closed_by_real_emission: bool, probability_used: float | None = None) -> dict[str, Any]:
        result = {
            "H_before": float(before),
            "H_after": float(after),
            "H_updated": bool(updated),
            "hazard_increment": None if increment is None else float(increment),
                "threshold": float(TAU),
            "threshold_reached": bool(reached),
            "effective_action": int(action),
            "duplicate_tick": bool(duplicate_tick),
            "closed_by_real_emission": bool(closed_by_real_emission),
            "p_one_saturated": increment is None and updated,
        }
        if probability_used is not None:
            result["probability_used_for_update"] = float(probability_used)
        return result


def _validated_probability(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("P(SEND) must be a finite real probability, not bool")
    try:
        p = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid P(SEND): {value!r}") from error
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        raise ValueError(f"P(SEND) must be finite and in [0,1], got {value!r}")
    return p


class HazardMedianCollector(FirstReportOnlyCollector):
    """ROUND40 qualification, feature and shadow-budget path with a new decoder only."""

    def __init__(self, controller: Any, *, enabled: bool = True) -> None:
        super().__init__(controller, greedy=not enabled)
        self.enabled = bool(enabled)
        self.hazard_state = HazardMedianState()
        self.mode = DECODER if enabled else "greedy_compatibility"
        self.hazard_updates = 0
        self.hazard_duplicate_ticks = 0

    def _choose_masked(self, observation: Any):
        items = list(observation.items)
        targets = [str(item.target_id) for item in items]
        if len(targets) != len(set(targets)):
            raise RuntimeError(f"relay observation contains duplicate targets: {targets}")
        must = {str(item.target_id) for item in items if self.mask and self.forced(item)}
        self.forced_sends += len(must)
        gate = shadow_gate(observation)
        budget_before = float(gate.remaining_bytes)
        capacity_before = observation.capacity_remaining

        # The original 19-D rows, full candidate list, and relay order go to the same frozen
        # network.  H is deliberately absent from rows and from controller inputs.
        rows = [features(observation, item) for item in items]
        matrix = torch.as_tensor(np.asarray(rows, dtype=np.float32))
        logits, values = self.controller.forward(matrix)
        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=-1)

        actions: dict[str, int] = {}
        requested: dict[str, bool] = {}
        charges: dict[str, int] = {}
        relay_executable: dict[str, bool | None] = {}
        relay_block_reason: dict[str, str | None] = {}
        candidate_audit: dict[str, dict[str, Any]] = {}
        decoder_audit: dict[str, dict[str, Any]] = {}
        decision_class: dict[str, str] = {}
        free_first_reports: list[str] = []
        masked_first_reports: list[str] = []
        fixed_hold_refresh: list[str] = []
        forced: list[str] = []
        send_probabilities: dict[str, float] = {}

        for index, item in enumerate(items):
            target = str(item.target_id)
            requested[target] = False
            is_first = bool(item.first_emission)
            charge = item.attempt_charge_bytes
            executable = item.send_executable
            block_reason = item.send_block_reason
            relay_executable[target] = None if executable is None else bool(executable)
            relay_block_reason[target] = None if block_reason is None else str(block_reason)

            if target in must:
                actions[target] = SEND
                decision_class[target] = "transport_forced"
                forced.append(target)
                continue

            if not is_first:
                closed = self.hazard_state.decide(
                    target, observation.step, 0.0, eligible=False, first_emission=False
                )
                actions[target] = HOLD
                decision_class[target] = "fixed_hold_refresh"
                fixed_hold_refresh.append(target)
                self.eligibility_counts["fixed_hold_refreshes"] += 1
                if charge is not None:
                    charges[target] = int(charge)
                candidate_audit[target] = {
                    "first_emission": False,
                    "relay_initial_executable": relay_executable[target],
                    "relay_initial_block_reason": relay_block_reason[target],
                    "effective_action": HOLD,
                    "fixed_reason": COMM_ACTION_ELIGIBILITY,
                    "shadow_budget_before": float(gate.remaining_bytes),
                    "shadow_budget_after": float(gate.remaining_bytes),
                    "shadow_capacity_before": gate.sends_this_step,
                    "shadow_capacity_after": gate.sends_this_step,
                    "shadow_charge_applied": False,
                }
                decoder_audit[target] = closed
                continue

            if charge is None or executable is None or block_reason is None:
                raise RuntimeError(
                    f"first-report candidate {target!r} has no relay-owned price/gate metadata: "
                    f"charge={charge!r}, executable={executable!r}, reason={block_reason!r}"
                )
            charge = int(charge)
            charges[target] = charge
            self.eligibility_counts["first_report_candidates"] += 1
            remaining_before = float(gate.remaining_bytes)
            capacity_sent_before = gate.sends_this_step
            affordable = gate.can_send(charge)
            effective_reason = gate.why_blocked(charge)
            candidate_audit[target] = {
                "first_emission": True,
                "attempt_charge_bytes": charge,
                "relay_initial_executable": bool(executable),
                "relay_initial_block_reason": str(block_reason),
                "sequentially_affordable_before_action": bool(affordable),
                "effective_block_reason": str(effective_reason),
                "shadow_budget_before": remaining_before,
                "shadow_budget_after": remaining_before,
                "shadow_capacity_before": capacity_sent_before,
                "shadow_capacity_after": capacity_sent_before,
                "shadow_charge_applied": False,
            }
            if not affordable:
                actions[target] = HOLD
                decision_class[target] = "existing_budget_or_capacity_mask"
                masked_first_reports.append(target)
                key = ("masked_by_credit" if effective_reason == "budget_exhausted"
                       else "masked_by_capacity")
                self.distribution_reads[key] += 1
                self.eligibility_counts[
                    "budget_masked_first_reports" if effective_reason == "budget_exhausted"
                    else "capacity_masked_first_reports"
                ] += 1
                before = self.hazard_state.value(target)
                decoder_audit[target] = {
                    "H_before": before, "H_after": before, "H_updated": False,
                    "hazard_increment": None, "threshold": TAU,
                    "threshold_reached": self.hazard_state.threshold_reached(target),
                    "effective_action": HOLD, "duplicate_tick": False,
                    "closed_by_real_emission": False, "masked_no_update": True,
                }
                continue

            p = _validated_probability(probabilities[index, SEND].item())
            if self.enabled:
                decoded = self.hazard_state.decide(
                    target, observation.step, p, eligible=True, first_emission=True
                )
                action = int(decoded["effective_action"])
                self.hazard_updates += int(decoded["H_updated"])
                self.hazard_duplicate_ticks += int(decoded["duplicate_tick"])
                decoder_audit[target] = decoded
            else:
                action = SEND if probabilities[index, SEND] >= probabilities[index, HOLD] else HOLD
                decoder_audit[target] = {
                    "H_before": self.hazard_state.value(target),
                    "H_after": self.hazard_state.value(target),
                    "H_updated": False,
                    "threshold": TAU,
                    "threshold_reached": False,
                    "effective_action": int(action),
                    "duplicate_tick": False,
                    "closed_by_real_emission": False,
                    "decoder_disabled": True,
                }
            actions[target] = action
            requested[target] = action == SEND
            decision_class[target] = (
                "free_first_report_hazard_median" if self.enabled else "free_first_report_greedy"
            )
            free_first_reports.append(target)
            send_probabilities[target] = p
            candidate_audit[target]["effective_action"] = int(action)
            candidate_audit[target]["model_p_send"] = p
            self.distribution_reads["free_candidates"] += 1
            self.distribution_reads["wanted_send"] += int(action == SEND)
            self.eligibility_counts["free_first_reports"] += 1
            self.eligibility_counts["requested_first_report_sends"] += int(action == SEND)
            if action == SEND:
                gate.charge(charge)
                candidate_audit[target]["shadow_charge_applied"] = True
            candidate_audit[target]["shadow_budget_after"] = float(gate.remaining_bytes)
            candidate_audit[target]["shadow_capacity_after"] = gate.sends_this_step

        categories = [set(forced), set(masked_first_reports), set(fixed_hold_refresh),
                      set(free_first_reports)]
        if set(targets) != set().union(*categories):
            raise RuntimeError("ROUND41 candidate classes do not cover all relay candidates")
        if any(categories[i] & categories[j]
               for i in range(len(categories)) for j in range(i + 1, len(categories))):
            raise RuntimeError("ROUND41 candidate classes overlap")

        asked = [target for target in free_first_reports if actions[target] == SEND]
        tape = {
            "schema": "round41-hazard-median-tape/1",
            "decoder": DECODER if self.enabled else "greedy_compatibility",
            "decoder_schema": DECODER_SCHEMA,
            "comm_action_eligibility": COMM_ACTION_ELIGIBILITY,
            "base_action_mask_schema": "budget-sequential-mask/1",
            "action_log_prob_semantics": "not_a_sampling_likelihood",
            "step": int(observation.step),
            "targets": targets,
            "features": [list(row) for row in rows],
            "actions": actions,
            "effective_actions": actions,
            "requested": requested,
            "decision_class": decision_class,
            "free_first_reports": list(free_first_reports),
            "free": list(free_first_reports),
            "actionable": list(free_first_reports),
            "masked": list(masked_first_reports),
            "masked_reasons": {
                target: candidate_audit[target]["effective_block_reason"]
                for target in masked_first_reports
            },
            "fixed_hold_refresh": list(fixed_hold_refresh),
            "fixed_hold_reasons": {
                target: COMM_ACTION_ELIGIBILITY for target in fixed_hold_refresh
            },
            "forced": sorted(forced),
            "attempt_charge_bytes": charges,
            "send_executable_initial": relay_executable,
            "send_block_reason_initial": relay_block_reason,
            "candidate_audit": candidate_audit,
            "hazard_audit": decoder_audit,
            "relay_observation_remaining_bytes": float(observation.remaining_budget_bytes),
            "shadow_budget_before": budget_before,
            "shadow_budget_after": float(gate.remaining_bytes),
            "shadow_capacity_before": capacity_before,
            "shadow_capacity_after": (
                None if capacity_before is None
                else int(capacity_before) - int(gate.sends_this_step)
            ),
            # StageBCollector keeps this numeric key for its rollup.  It is deliberately zero
            # and explicitly labelled inapplicable: the deterministic decoder has no sampled
            # action likelihood.
            "log_prob": 0.0,
            "value": float(values.mean().item()) if items else 0.0,
            "send_probabilities": send_probabilities,
        }
        return asked, actions, tape["free"], tape["log_prob"], tape["value"], tape

    def timing(self) -> dict[str, Any]:
        result = super().timing()
        result.update({
            "decoder": self.mode,
            "decoder_schema": DECODER_SCHEMA,
            "hazard_threshold": TAU,
            "hazard_updates": self.hazard_updates,
            "duplicate_eligible_tick_calls": self.hazard_duplicate_ticks,
            "action_log_prob_semantics": "not_a_sampling_likelihood",
            "decoder_rng_calls": 0,
        })
        return result

    def tape_summary(self) -> dict[str, Any]:
        result = super().tape_summary()
        result["timing"] = self.timing()
        result["decoder"] = self.mode
        result["action_log_prob_semantics"] = "not_a_sampling_likelihood"
        return result


__all__ = [
    "DECODER", "DECODER_SCHEMA", "TAU", "HazardMedianCollector", "HazardMedianState",
]
