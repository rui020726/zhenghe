"""ROUND40 training/evaluation collector with one shared first-report action space.

The qualification bit comes only from ``SendItem.first_emission``, which CommRelay derives
from its successful-emission ledger.  The wrapper keeps every relay candidate and its feature
row.  A refresh is fixed to HOLD before the sequential shadow budget is consulted, so it spends
neither bytes nor per-step capacity and contributes no actor likelihood or entropy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from budget_masked import HOLD, SEND, shadow_gate
from sb_collector import StageBCollector
from integrated_sar.comm_learning import features
from integrated_sar.comm_policy import SendObservation

COMM_ACTION_ELIGIBILITY = "first-report-only/1"
TAPE_SCHEMA = "round40-first-report-only-tape/1"


class FirstReportOnlyCollector(StageBCollector):
    """Sample or greedily decode first reports; hold every post-emission candidate."""

    def __init__(self, controller: Any, *, greedy: bool = False) -> None:
        super().__init__(controller)
        self.greedy = bool(greedy)
        self.eligibility_counts: dict[str, int] = {
            "first_report_candidates": 0,
            "free_first_reports": 0,
            "budget_masked_first_reports": 0,
            "capacity_masked_first_reports": 0,
            "fixed_hold_refreshes": 0,
            "requested_first_report_sends": 0,
        }

    def _choose_masked(
        self, observation: SendObservation
    ) -> tuple[list[str], dict[str, int], list[str], float, float, dict[str, Any]]:
        items = list(observation.items)
        targets = [str(item.target_id) for item in items]
        if len(targets) != len(set(targets)):
            raise RuntimeError(f"relay observation contains duplicate targets: {targets}")

        must = {item.target_id for item in items if self.mask and self.forced(item)}
        self.forced_sends += len(must)
        gate = shadow_gate(observation)
        budget_before = float(gate.remaining_bytes)
        capacity_before = observation.capacity_remaining

        # Keep the full observation and original candidate order in the actor input, including
        # refresh rows.  Only the action set below changes.
        rows = [features(observation, item) for item in items]
        matrix = torch.as_tensor(np.asarray(rows, dtype=np.float32))
        logits, values = self.controller.forward(matrix)
        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=-1)
            log_probabilities = torch.log_softmax(logits, dim=-1)

        actions: dict[str, int] = {}
        requested: dict[str, bool] = {}
        charges: dict[str, int] = {}
        relay_executable: dict[str, bool | None] = {}
        relay_block_reason: dict[str, str | None] = {}
        candidate_audit: dict[str, dict[str, Any]] = {}
        decision_class: dict[str, str] = {}
        free_first_reports: list[str] = []
        masked_first_reports: list[str] = []
        fixed_hold_refresh: list[str] = []
        forced: list[str] = []
        log_terms: list[float] = []
        send_probabilities: dict[str, float] = {}

        for index, item in enumerate(items):
            target = str(item.target_id)
            requested[target] = False
            is_first = bool(item.first_emission)
            if target in must:
                actions[target] = SEND
                decision_class[target] = "transport_forced"
                forced.append(target)
                continue

            charge = item.attempt_charge_bytes
            executable = item.send_executable
            block_reason = item.send_block_reason
            relay_executable[target] = None if executable is None else bool(executable)
            relay_block_reason[target] = None if block_reason is None else str(block_reason)

            if not is_first:
                # This is the method intervention.  Do not ask the shadow gate whether a
                # suppressed refresh could be afforded: it has no action and no reservation.
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
                continue

            if self.greedy:
                action = (SEND if probabilities[index, SEND] >= probabilities[index, HOLD]
                          else HOLD)
            else:
                action = int(torch.multinomial(probabilities[index], num_samples=1).item())
            actions[target] = action
            requested[target] = action == SEND
            decision_class[target] = "free_first_report"
            free_first_reports.append(target)
            log_terms.append(float(log_probabilities[index, action].item()))
            send_probabilities[target] = float(probabilities[index, SEND].item())
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
            raise RuntimeError("ROUND40 action classes do not cover all relay candidates")
        if any(categories[i] & categories[j]
               for i in range(len(categories)) for j in range(i + 1, len(categories))):
            raise RuntimeError("ROUND40 action classes overlap")
        if any(actions[target] != HOLD or requested[target]
               for target in fixed_hold_refresh):
            raise RuntimeError("a refresh candidate escaped its fixed HOLD class")

        asked = [target for target in free_first_reports if actions[target] == SEND]
        tape: dict[str, Any] = {
            "schema": TAPE_SCHEMA,
            "comm_action_eligibility": COMM_ACTION_ELIGIBILITY,
            "base_action_mask_schema": "budget-sequential-mask/1",
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
            "relay_observation_remaining_bytes": float(observation.remaining_budget_bytes),
            "shadow_budget_before": budget_before,
            "shadow_budget_after": float(gate.remaining_bytes),
            "shadow_capacity_before": capacity_before,
            "shadow_capacity_after": (
                None if capacity_before is None
                else int(capacity_before) - int(gate.sends_this_step)
            ),
            "log_prob": float(sum(log_terms)) if log_terms else 0.0,
            "value": float(values.mean().item()) if items else 0.0,
            "send_probabilities": send_probabilities,
        }
        return asked, actions, tape["free"], tape["log_prob"], tape["value"], tape

    def timing(self) -> dict[str, Any]:
        result = super().timing()
        result["comm_action_eligibility"] = COMM_ACTION_ELIGIBILITY
        result["first_report_only_counts"] = dict(self.eligibility_counts)
        result["decoder"] = "greedy" if self.greedy else "sampled"
        return result

    def tape_summary(self) -> dict[str, Any]:
        result = super().tape_summary()
        categories_close = True
        categories_disjoint = True
        for tape in self.records:
            categories = [set(tape["forced"]), set(tape["masked"]),
                          set(tape["fixed_hold_refresh"]), set(tape["actionable"])]
            categories_close &= set(tape["targets"]) == set().union(*categories)
            categories_disjoint &= all(
                not (categories[i] & categories[j])
                for i in range(len(categories)) for j in range(i + 1, len(categories))
            )
        result.update({
            "comm_action_eligibility": COMM_ACTION_ELIGIBILITY,
            "fixed_hold_refresh_total": sum(
                len(tape["fixed_hold_refresh"]) for tape in self.records
            ),
            "first_report_candidates_total": sum(
                len(tape["free_first_reports"]) + len(tape["masked"])
                for tape in self.records
            ),
            "candidate_classes_close": bool(categories_close),
            "candidate_classes_disjoint": bool(categories_disjoint),
            "refreshes_excluded_from_likelihood": all(
                not (set(tape["fixed_hold_refresh"]) & set(tape["free"]))
                and all(tape["actions"][target] == HOLD
                        for target in tape["fixed_hold_refresh"])
                for tape in self.records
            ),
            "eligibility_counts": dict(self.eligibility_counts),
        })
        return result


__all__ = ["COMM_ACTION_ELIGIBILITY", "FirstReportOnlyCollector", "TAPE_SCHEMA"]
