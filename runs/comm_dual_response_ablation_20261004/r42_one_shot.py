"""ROUND42 one-shot SEND_NOW / SKIP_EPISODE collector.

This versioned adapter keeps ROUND40's feature rows, relay quote, sequential shadow budget,
candidate order, value reads, and Stage C tape interface.  Its only new state is the per-target,
per-episode decision latch: after the first legal affordable choice, both SEND and SKIP close
that target for the rest of the episode.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from budget_masked import HOLD, SEND, shadow_gate
from integrated_sar.comm_learning import features
from integrated_sar.comm_policy import SendObservation
from r40_eligibility import FirstReportOnlyCollector


COMM_ACTION_SEMANTICS = "one-shot-send-skip/1"
TAPE_SCHEMA = "round42-one-shot-send-skip-tape/1"


class UnexpectedPolicySendRefusal(RuntimeError):
    """A policy SEND passed its quote gate but the real relay refused it."""


@contextmanager
def stop_on_policy_refusal():
    """Stop a one-shot run at the first real budget/capacity refusal.

    The wrapper calls the unchanged production ``CommRelay._emit`` first, preserving its
    refusal transcript and counters, then raises before the relay advances to delivery or the
    next environment tick. Successful emissions and packet loss take the original path.
    """
    from integrated_sar.comm import CommRelay

    original = CommRelay._emit

    def guarded(relay: Any, item: Any, *, first: bool, reason: str = "policy") -> None:
        before = (int(relay.counters["budget_refused_sends"]),
                  int(relay.counters["capacity_refused_sends"]))
        original(relay, item, first=first, reason=reason)
        after = (int(relay.counters["budget_refused_sends"]),
                 int(relay.counters["capacity_refused_sends"]))
        if reason == "policy" and after != before:
            raise UnexpectedPolicySendRefusal(
                f"one-shot SEND for target {item.target_id!r} was refused after the "
                f"collector passed its quote gate: budget {before[0]}->{after[0]}, "
                f"capacity {before[1]}->{after[1]}"
            )

    CommRelay._emit = guarded
    try:
        yield
    finally:
        CommRelay._emit = original


class OneShotSendSkipCollector(FirstReportOnlyCollector):
    """Make one real policy choice at the first sequentially affordable first report."""

    def __init__(self, controller: Any, *, greedy: bool = False) -> None:
        super().__init__(controller, greedy=greedy)
        self.target_state: dict[str, str] = {}
        self.one_shot_counts = {
            "undecided_first_report_candidates": 0,
            "masked_budget_or_capacity": 0,
            "free_choices": 0,
            "send_now_choices": 0,
            "skip_episode_choices": 0,
            "fixed_holds_after_choice": 0,
            "closed_by_prior_real_emission": 0,
        }

    def _choose_masked(self, observation: SendObservation):
        items = list(observation.items)
        targets = [str(item.target_id) for item in items]
        if len(targets) != len(set(targets)):
            raise RuntimeError(f"relay observation contains duplicate targets: {targets}")

        forced_targets = {str(item.target_id) for item in items
                          if self.mask and self.forced(item)}
        self.forced_sends += len(forced_targets)
        gate = shadow_gate(observation)
        budget_before = float(gate.remaining_bytes)
        capacity_before = observation.capacity_remaining

        # Keep every relay candidate and its original 19-D row, including closed targets.
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
        free_choices: list[str] = []
        masked_first_reports: list[str] = []
        fixed_closed: list[str] = []
        fixed_refresh: list[str] = []
        forced: list[str] = []
        log_terms: list[float] = []
        send_probabilities: dict[str, float] = {}

        for index, item in enumerate(items):
            target = str(item.target_id)
            requested[target] = False
            executable = item.send_executable
            block_reason = item.send_block_reason
            relay_executable[target] = None if executable is None else bool(executable)
            relay_block_reason[target] = None if block_reason is None else str(block_reason)

            if target in forced_targets:
                actions[target] = SEND
                decision_class[target] = "transport_forced"
                forced.append(target)
                continue

            prior_state = self.target_state.get(target, "undecided")
            if prior_state != "undecided":
                actions[target] = HOLD
                decision_class[target] = "fixed_hold_one_shot_closed"
                fixed_closed.append(target)
                fixed_refresh.extend([target] if not bool(item.first_emission) else [])
                self.one_shot_counts["fixed_holds_after_choice"] += 1
                if item.attempt_charge_bytes is not None:
                    charges[target] = int(item.attempt_charge_bytes)
                candidate_audit[target] = {
                    "first_emission": bool(item.first_emission),
                    "one_shot_state_before": prior_state,
                    "effective_action": HOLD,
                    "fixed_reason": "one-shot-choice-already-made",
                    "shadow_budget_before": float(gate.remaining_bytes),
                    "shadow_budget_after": float(gate.remaining_bytes),
                    "shadow_capacity_before": gate.sends_this_step,
                    "shadow_capacity_after": gate.sends_this_step,
                    "shadow_charge_applied": False,
                }
                continue

            # A real earlier emission is authoritative for sender-side closure.  In normal
            # ROUND42 collection this only occurs if a candidate was emitted outside this
            # collector; it is kept distinct from a policy SKIP.
            if not bool(item.first_emission):
                self.target_state[target] = "closed_by_prior_real_emission"
                actions[target] = HOLD
                decision_class[target] = "fixed_hold_prior_emission"
                fixed_closed.append(target)
                fixed_refresh.append(target)
                self.one_shot_counts["closed_by_prior_real_emission"] += 1
                if item.attempt_charge_bytes is not None:
                    charges[target] = int(item.attempt_charge_bytes)
                candidate_audit[target] = {
                    "first_emission": False,
                    "one_shot_state_before": "undecided",
                    "one_shot_state_after": self.target_state[target],
                    "effective_action": HOLD,
                    "fixed_reason": "prior-real-emission",
                    "shadow_budget_before": float(gate.remaining_bytes),
                    "shadow_budget_after": float(gate.remaining_bytes),
                    "shadow_capacity_before": gate.sends_this_step,
                    "shadow_capacity_after": gate.sends_this_step,
                    "shadow_charge_applied": False,
                }
                continue

            if item.attempt_charge_bytes is None or executable is None or block_reason is None:
                raise RuntimeError(
                    f"first-report candidate {target!r} lacks relay-owned quote/gate metadata: "
                    f"charge={item.attempt_charge_bytes!r}, executable={executable!r}, "
                    f"reason={block_reason!r}"
                )

            charge = int(item.attempt_charge_bytes)
            charges[target] = charge
            self.eligibility_counts["first_report_candidates"] += 1
            self.one_shot_counts["undecided_first_report_candidates"] += 1
            remaining_before = float(gate.remaining_bytes)
            capacity_sent_before = gate.sends_this_step
            affordable = bool(gate.can_send(charge))
            effective_reason = gate.why_blocked(charge)
            candidate_audit[target] = {
                "first_emission": True,
                "attempt_charge_bytes": charge,
                "relay_initial_executable": bool(executable),
                "relay_initial_block_reason": str(block_reason),
                "sequentially_affordable_before_action": affordable,
                "effective_block_reason": str(effective_reason),
                "one_shot_state_before": "undecided",
                "one_shot_state_after": "undecided",
                "shadow_budget_before": remaining_before,
                "shadow_budget_after": remaining_before,
                "shadow_capacity_before": capacity_sent_before,
                "shadow_capacity_after": capacity_sent_before,
                "shadow_charge_applied": False,
            }
            if not affordable:
                # A mask is not a choice: it leaves the target undecided and spends no RNG.
                actions[target] = HOLD
                decision_class[target] = "existing_budget_or_capacity_mask"
                masked_first_reports.append(target)
                self.distribution_reads[
                    "masked_by_credit" if effective_reason == "budget_exhausted"
                    else "masked_by_capacity"
                ] += 1
                self.one_shot_counts["masked_budget_or_capacity"] += 1
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

            # Both alternatives are genuine free actions.  Latch immediately, before the
            # executor call, so an unexpected refusal cannot cause resampling or a retry.
            state_after = "send_now_chosen" if action == SEND else "skip_episode_chosen"
            self.target_state[target] = state_after
            actions[target] = int(action)
            requested[target] = action == SEND
            decision_class[target] = "free_one_shot_choice"
            free_choices.append(target)
            log_terms.append(float(log_probabilities[index, action].item()))
            send_probabilities[target] = float(probabilities[index, SEND].item())
            self.distribution_reads["free_candidates"] += 1
            self.distribution_reads["wanted_send"] += int(action == SEND)
            self.one_shot_counts["free_choices"] += 1
            self.eligibility_counts["free_first_reports"] += 1
            self.eligibility_counts["requested_first_report_sends"] += int(action == SEND)
            self.one_shot_counts[
                "send_now_choices" if action == SEND else "skip_episode_choices"
            ] += 1
            if action == SEND:
                gate.charge(charge)
                candidate_audit[target]["shadow_charge_applied"] = True
            candidate_audit[target]["one_shot_state_after"] = state_after
            candidate_audit[target]["effective_action"] = int(action)
            candidate_audit[target]["shadow_budget_after"] = float(gate.remaining_bytes)
            candidate_audit[target]["shadow_capacity_after"] = gate.sends_this_step

        categories = [set(forced), set(masked_first_reports), set(fixed_closed),
                      set(free_choices)]
        if set(targets) != set().union(*categories):
            raise RuntimeError("ROUND42 candidate classes do not cover the relay observation")
        if any(categories[i] & categories[j]
               for i in range(len(categories)) for j in range(i + 1, len(categories))):
            raise RuntimeError("ROUND42 candidate classes overlap")
        if any(actions[target] != HOLD or requested[target] for target in fixed_closed):
            raise RuntimeError("a closed one-shot target escaped its fixed HOLD class")

        asked = [target for target in free_choices if actions[target] == SEND]
        tape: dict[str, Any] = {
            "schema": TAPE_SCHEMA,
            "comm_action_semantics": COMM_ACTION_SEMANTICS,
            "comm_action_eligibility": "first-report-only/1",
            "base_action_mask_schema": "budget-sequential-mask/1",
            "step": int(observation.step),
            "targets": targets,
            "features": [list(row) for row in rows],
            "actions": actions,
            "effective_actions": actions,
            "requested": requested,
            "decision_class": decision_class,
            "free_one_shot_choices": list(free_choices),
            "free_first_reports": list(free_choices),
            "free": list(free_choices),
            "actionable": list(free_choices),
            "masked": list(masked_first_reports),
            "masked_reasons": {
                target: candidate_audit[target]["effective_block_reason"]
                for target in masked_first_reports
            },
            "fixed_hold_one_shot_closed": list(fixed_closed),
            "fixed_hold_refresh": list(fixed_refresh),
            "fixed_hold_reasons": {
                target: str(candidate_audit[target]["fixed_reason"])
                for target in fixed_closed
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
            "one_shot_state_after_step": dict(self.target_state),
        }
        return asked, actions, tape["free"], tape["log_prob"], tape["value"], tape

    def timing(self) -> dict[str, Any]:
        result = super().timing()
        result.update({
            "comm_action_semantics": COMM_ACTION_SEMANTICS,
            "one_shot_counts": dict(self.one_shot_counts),
            "closed_target_count": len(self.target_state),
        })
        return result

    def tape_summary(self) -> dict[str, Any]:
        close = True
        disjoint = True
        for tape in self.records:
            categories = [set(tape["forced"]), set(tape["masked"]),
                          set(tape["fixed_hold_one_shot_closed"]),
                          set(tape["actionable"])]
            close &= set(tape["targets"]) == set().union(*categories)
            disjoint &= all(not (categories[i] & categories[j])
                            for i in range(len(categories))
                            for j in range(i + 1, len(categories)))
        return {
            "comm_action_semantics": COMM_ACTION_SEMANTICS,
            "decision_steps": len(self.records),
            "candidate_classes_close": bool(close),
            "candidate_classes_disjoint": bool(disjoint),
            "one_shot_counts": dict(self.one_shot_counts),
            "eligibility_counts": dict(self.eligibility_counts),
            "fixed_hold_one_shot_total": sum(
                len(tape["fixed_hold_one_shot_closed"]) for tape in self.records),
            "free_one_shot_choices_total": sum(
                len(tape["free_one_shot_choices"]) for tape in self.records),
            "skip_choices_in_free_set": all(
                target in tape["free"] and tape["actions"][target] == HOLD
                for tape in self.records for target in tape["free"]
                if tape["candidate_audit"][target]["one_shot_state_after"]
                == "skip_episode_chosen"),
            "closed_targets_excluded_from_likelihood": all(
                not (set(tape["fixed_hold_one_shot_closed"]) & set(tape["free"]))
                for tape in self.records),
            "timing": self.timing(),
        }


__all__ = [
    "COMM_ACTION_SEMANTICS", "OneShotSendSkipCollector", "TAPE_SCHEMA",
    "UnexpectedPolicySendRefusal", "stop_on_policy_refusal",
]
