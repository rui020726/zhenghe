"""ROUND39-local refresh-held wrapper around the frozen greedy collector.

The only intervention is a HOLD for a candidate whose authoritative relay item says the
target has already had a successful emission. This class has no per-target memory and does
not alter the relay, registry, RNG, feature vector, or candidate ordering.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from budget_masked import BudgetMaskedCollector, HOLD, SEND, shadow_gate
from integrated_sar.comm_learning import features
from integrated_sar.comm_policy import SendObservation


SCHEMA = "round39-refresh-held-decision-tape/1"
INTERVENTION_REASON = "target_has_authoritative_prior_real_emission"


class RefreshHeldCollector(BudgetMaskedCollector):
    """Use the ROUND37 masked greedy path, with refresh suppression as an optional wrapper."""

    def __init__(
        self,
        controller: Any,
        *,
        forced: Callable[[Any], bool],
        mask: bool = False,
        refresh_held: bool = True,
    ) -> None:
        super().__init__(controller, forced=forced, mask=mask, greedy=True, budget_mask=True)
        self.refresh_held = bool(refresh_held)
        self.refresh_counts = {
            "candidates": 0,
            "raw_model_send_recommendations_on_refresh_candidates": 0,
            "effective_shadow_charges": 0,
        }

    def timing(self) -> dict[str, Any]:
        summary = super().timing()
        summary["refresh_held_schema"] = SCHEMA
        summary["refresh_held_enabled"] = self.refresh_held
        summary["refresh_held_counts"] = dict(self.refresh_counts)
        return summary

    def _choose_masked(self, observation: SendObservation):
        if not self.refresh_held:
            return super()._choose_masked(observation)

        items = list(observation.items)
        targets = [item.target_id for item in items]
        if len(targets) != len(set(targets)):
            raise RuntimeError(f"relay observation contains duplicate target ids: {targets}")

        must = {item.target_id for item in items if self.mask and self.forced(item)}
        self.forced_sends += len(must)
        effective_gate = shadow_gate(observation)
        effective_budget_at_start = float(effective_gate.remaining_bytes)
        rows = [features(observation, item) for item in items]
        matrix = torch.as_tensor(np.asarray(rows, dtype=np.float32))
        logits, values = self.controller.forward(matrix)
        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=-1)
            log_probabilities = torch.log_softmax(logits, dim=-1)

        actions: dict[str, int] = {}
        requested: dict[str, bool] = {}
        raw_actions: dict[str, int] = {}
        decision_source: dict[str, str] = {}
        intervention_reason: dict[str, str] = {}
        charges: dict[str, int] = {}
        relay_initial_executable: dict[str, bool] = {}
        relay_initial_reason: dict[str, str] = {}
        candidate_shadow: dict[str, dict[str, Any]] = {}
        actionable: list[str] = []
        masked: list[str] = []
        intervened: list[str] = []
        forced_targets: list[str] = []
        terms: list[float] = []
        send_probabilities: dict[str, float] = {}

        for index, item in enumerate(items):
            target = item.target_id
            requested[target] = False
            raw_action = (
                SEND
                if probabilities[index, SEND] >= probabilities[index, HOLD]
                else HOLD
            )
            raw_actions[target] = raw_action

            if target in must:
                # Preserve the base collector's external forced-send handling. ROUND39 evals
                # use mask=False and forced=False, so this branch is not expected in the matrix.
                actions[target] = SEND
                decision_source[target] = "preexisting_transport_forced_send"
                forced_targets.append(target)
                continue

            charge = item.attempt_charge_bytes
            executable = item.send_executable
            block_reason = item.send_block_reason
            if charge is None or executable is None or block_reason is None:
                raise RuntimeError(
                    f"candidate {target!r} lacks relay-owned price/gate metadata "
                    f"(charge={charge!r}, executable={executable!r}, "
                    f"reason={block_reason!r})"
                )
            charge = int(charge)
            charges[target] = charge
            relay_initial_executable[target] = bool(executable)
            relay_initial_reason[target] = str(block_reason)

            effective_before = float(effective_gate.remaining_bytes)
            effective_capacity_before = effective_gate.sends_this_step
            effective_affordable = effective_gate.can_send(charge)
            refresh_candidate = not bool(item.first_emission)

            if refresh_candidate:
                # The intervention is applied before the only shadow gate that controls the
                # effective run can charge this candidate.
                effective_action = HOLD
                intervened.append(target)
                intervention_reason[target] = INTERVENTION_REASON
                decision_source[target] = "external_refresh_intervention"
                self.refresh_counts["candidates"] += 1
                self.refresh_counts["raw_model_send_recommendations_on_refresh_candidates"] += int(
                    raw_action == SEND
                )
            elif not effective_affordable:
                effective_action = HOLD
                masked.append(target)
                decision_source[target] = "existing_sequential_budget_mask"
                reason = effective_gate.why_blocked(charge)
                self.distribution_reads[
                    "masked_by_credit" if reason == "budget_exhausted" else "masked_by_capacity"
                ] += 1
            else:
                effective_action = raw_action
                actionable.append(target)
                decision_source[target] = "frozen_greedy_policy"
                terms.append(float(log_probabilities[index, raw_action].item()))
                send_probabilities[target] = float(probabilities[index, SEND].item())
                self.distribution_reads["free_candidates"] += 1
                self.distribution_reads["wanted_send"] += int(raw_action == SEND)
                if raw_action == SEND:
                    effective_gate.charge(charge)
                    self.refresh_counts["effective_shadow_charges"] += 1

            actions[target] = effective_action
            requested[target] = effective_action == SEND
            candidate_shadow[target] = {
                "attempt_charge_bytes": charge,
                "relay_initial_state": {
                    "remaining_budget_bytes": float(observation.remaining_budget_bytes),
                    "capacity_remaining": observation.capacity_remaining,
                    "send_executable": bool(executable),
                    "send_block_reason": str(block_reason),
                },
                "relay_initial_executable": bool(executable),
                "relay_initial_block_reason": str(block_reason),
                "effective_run": {
                    "budget_before": effective_before,
                    "capacity_sent_before": effective_capacity_before,
                    "sequential_affordable_before_action": bool(effective_affordable),
                    "action": int(effective_action),
                    "budget_after": float(effective_gate.remaining_bytes),
                    "capacity_sent_after": effective_gate.sends_this_step,
                    "charge_applied": bool(
                        effective_action == SEND and not refresh_candidate
                    ),
                },
            }

        if set(targets) != set(forced_targets) | set(masked) | set(intervened) | set(actionable):
            raise RuntimeError("ROUND39 effective decision classes do not cover the candidates")
        classes = [set(forced_targets), set(masked), set(intervened), set(actionable)]
        if any(classes[i] & classes[j] for i in range(len(classes)) for j in range(i + 1, 4)):
            raise RuntimeError("ROUND39 effective decision classes overlap")

        asked = [target for target in actionable if actions[target] == SEND]
        tape: dict[str, Any] = {
            "schema": SCHEMA,
            "base_action_mask_schema": "budget-sequential-mask/1",
            "step": int(observation.step),
            "targets": targets,
            "features": [list(row) for row in rows],
            "raw_model_actions": raw_actions,
            "raw_model_action_status": {
                target: "computed_from_frozen_logits_diagnostic" for target in targets
            },
            "effective_actions": actions,
            "actions": actions,
            "decision_source": decision_source,
            "intervened": intervened,
            "intervention_reason": intervention_reason,
            "forced": sorted(forced_targets),
            "masked": list(masked),
            "actionable": list(actionable),
            "free": list(actionable),
            "requested": requested,
            "attempt_charge_bytes": charges,
            "send_executable_initial": relay_initial_executable,
            "send_block_reason_initial": relay_initial_reason,
            "candidate_budget_audit": candidate_shadow,
            "relay_observation_remaining_bytes": float(observation.remaining_budget_bytes),
            "shadow_budget_before": effective_budget_at_start,
            "shadow_budget_after": float(effective_gate.remaining_bytes),
            "shadow_capacity_before": observation.capacity_remaining,
            "shadow_capacity_after": (
                None
                if observation.capacity_remaining is None
                else int(observation.capacity_remaining) - int(effective_gate.sends_this_step)
            ),
            "log_prob": float(sum(terms)) if terms else 0.0,
            "value": float(values.mean().item()) if items else 0.0,
            "send_probabilities": send_probabilities,
        }
        return asked, actions, tape["free"], tape["log_prob"], tape["value"], tape


__all__ = ["INTERVENTION_REASON", "RefreshHeldCollector", "SCHEMA"]
