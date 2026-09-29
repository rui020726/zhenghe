"""ROUND27 intervention: advance one target's first report inside the real sequential decision path.

``FirstReportAdvancer`` subclasses the archived ``BudgetMaskedCollector`` and inserts exactly one
hook into the candidate walk: after the sequential shadow gate has said a candidate is payable
(the ``honours`` verdict) and the original greedy action has been computed, a named target whose
greedy action is HOLD may be overridden to SEND. Everything else about the decision is the
parent's own code, copied verbatim below with the insertion marked; the copy is checked against
the parent in ``r27_preflight`` (same synthetic observation stream, same controller, identical
tape when no named target qualifies).

What the override is not:
* not a free send -- the overridden SEND charges the same shadow gate and the same real budget,
  in the same candidate order, at the same price the relay quoted;
* not an appended emission -- it happens while the collector is still building its ask, so later
  candidates of the same step see the spent credit;
* not a bypass of the mask -- a candidate the shadow cannot pay for never reaches the hook;
* not a forced send -- ``forced`` stays disabled, the row keeps its actionable bookkeeping, and
  the tape records that its log-probability describes the forced action;
* not a retransmission rule -- at most one override per target, and a lost packet adds nothing.

The interceptor is a local wrapper in the experimental harness: no production default, transport
semantics or channel RNG mechanism is touched.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from budget_masked import HOLD, SEND, BudgetMaskedCollector, shadow_gate  # noqa: F401  (parent seam)
from integrated_sar.comm_learning import features
from integrated_sar.comm_policy import SendObservation

ACTION_MASK_SCHEMA = "budget-sequential-mask/1"
TAPE_SCHEMA = "onpolicy-tape/2"
OVERRIDE_SCHEMA = "round27-first-report-override/1"

__all__ = ["FirstReportAdvancer", "OVERRIDE_SCHEMA"]


class FirstReportAdvancer(BudgetMaskedCollector):
    """The ROUND25 candidate collector plus a first-report advance hook for named targets."""

    def __init__(self, controller: Any, *, targets: tuple[str, ...] = (), enabled: bool = True,
                 forced: Any = lambda item: False, mask: bool = False, greedy: bool = True,
                 budget_mask: bool = True) -> None:
        super().__init__(controller, forced=forced, mask=mask, greedy=greedy,
                         budget_mask=budget_mask)
        self.override_targets = frozenset(str(target) for target in targets)
        self.override_enabled = bool(enabled)
        #: targets whose single override has been used (or proved unnecessary); never reset.
        self.consumed: set[str] = set()
        #: targets the relay has really put on the air (its own ledger view, read off items).
        self.really_sent: set[str] = set()

    # ------------------------------------------------------------------ identity
    def intervention_identity(self) -> dict[str, Any]:
        return {
            "schema": OVERRIDE_SCHEMA,
            "class": type(self).__name__,
            "enabled": bool(self.override_enabled),
            "targets": sorted(self.override_targets),
            "rule": "at the first decision opportunity after discovery where the candidate is "
                    "legal and payable in the real sequential budget walk, an original greedy "
                    "HOLD for a named target is overridden to SEND; if the original action is "
                    "already SEND there is no action difference; at most one override per "
                    "target; after the first emission the original learned logic resumes",
            "consumed": sorted(self.consumed),
            "really_sent": sorted(self.really_sent),
            "not_a_forced_send": True,
            "not_an_appended_emission": True,
            "log_prob_semantics": "an overridden row's log-probability describes the forced "
                                  "action, not the model's draw; the tape is inert (no training)",
        }

    # ------------------------------------------------------------------ the hook
    def _named(self, target: str) -> bool:
        return bool(self.override_enabled) and target in self.override_targets

    def _pending(self, target: str) -> bool:
        """A named target whose single override is still outstanding: the only one we record."""
        return self._named(target) and target not in self.consumed

    def _first_report_hook(self, *, step: int, item: Any, action: int, send_probability: float,
                           charge: int, budget_before: float) -> tuple[int, dict[str, Any] | None]:
        """The one inserted decision: maybe turn this named target's HOLD into its first report.

        Only reachable for a candidate the sequential shadow gate has already honoured, so the
        override spends real credit in real candidate order like any other send.
        """
        target = str(item.target_id)
        if not self._pending(target):
            # Either not named, or this target's single override is already settled: the
            # original learned logic continues untouched and nothing is recorded.
            return action, None
        note: dict[str, Any] = {
            "step": int(step),
            "target": target,
            "class": "actionable",
            "greedy_action": "SEND" if action == SEND else "HOLD",
            "greedy_send_probability": round(float(send_probability), 6),
            "attempt_charge_bytes": int(charge),
            "budget_before_bytes": float(budget_before),
            "ready_step": int(getattr(item, "ready_step", -1)),
            "sent_count_before": int(getattr(item, "sent_count", 0)),
            "first_emission_flag_before": bool(getattr(item, "first_emission", False)),
            "override_applied": False,
        }
        if (target in self.really_sent or int(getattr(item, "sent_count", 0)) > 0
                or not bool(getattr(item, "first_emission", False))):
            self.really_sent.add(target)
            note["outcome"] = "already_sent_no_duplicate"
            self.consumed.add(target)
            return action, note
        if action == SEND:
            note["outcome"] = "greedy_already_sends_no_action_difference"
            self.consumed.add(target)
            return action, note
        note["outcome"] = "overridden_hold_to_send"
        note["override_applied"] = True
        note["override_action"] = "SEND"
        self.consumed.add(target)
        return SEND, note

    # ------------------------------------------------------------------ the decision
    def _choose_masked(self, observation: SendObservation) -> tuple[Any, ...]:
        # Disabled overrides delegate to the archived collector verbatim: "unchanged" is a
        # delegation, not a second copy of the loop to keep in step.
        if not self.override_enabled:
            return super()._choose_masked(observation)
        return self._choose_masked_with_override(observation)

    def _choose_masked_with_override(self, observation: SendObservation) -> tuple[Any, ...]:
        """The archived masked walk, verbatim, with one inserted hook after the greedy action.

        Every line below is the parent's code except the marked hook block; ``r27_preflight``
        proves the copy reproduces the parent's tape byte for byte when no named target
        qualifies.
        """
        items = list(observation.items)
        must = {item.target_id for item in items if self.mask and self.forced(item)}
        self.forced_sends += len(must)
        gate = shadow_gate(observation)
        budget_at_start = float(gate.remaining_bytes)
        capacity_at_start = observation.capacity_remaining
        rows = [features(observation, item) for item in items]
        matrix = torch.as_tensor(np.asarray(rows, dtype=np.float32))
        logits, values = self.controller.forward(matrix)
        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=-1)
            log_probabilities = torch.log_softmax(logits, dim=-1)
        actions: dict[str, int] = {}
        requested: dict[str, bool] = {}
        charges: dict[str, int] = {}
        executable_initial: dict[str, bool] = {}
        block_initial: dict[str, str] = {}
        block_effective: dict[str, str] = {}
        candidate_shadow: dict[str, dict[str, Any]] = {}
        actionable: list[str] = []
        masked: list[str] = []
        terms: list[float] = []
        send_probabilities: dict[str, float] = {}
        candidate_notes: list[dict[str, Any]] = []      # ROUND27: named-target opportunities
        overrides_this_step: list[dict[str, Any]] = []  # ROUND27: overrides that fired here
        for index, item in enumerate(items):
            target = item.target_id
            requested[target] = False
            if target in must:
                # The silence bound's decision, not ours: it goes out and it is billed, and it
                # carries no likelihood because nothing chose it.
                actions[target] = SEND
                requested[target] = True
                if self._pending(target):
                    candidate_notes.append({
                        "step": int(observation.step), "target": target, "class": "forced",
                        "greedy_action": "SEND", "override_applied": False,
                        "outcome": "forced_by_bound_no_override",
                    })
                continue
            charge = item.attempt_charge_bytes
            executable = item.send_executable
            reason = item.send_block_reason
            if charge is None or executable is None or reason is None:
                raise RuntimeError(
                    f"candidate {target!r} has no attempt charge metadata "
                    f"(charge={charge!r}, executable={executable!r}, reason={reason!r}); a "
                    "budget-masked collector cannot know its action set without it, and "
                    "guessing 'affordable' is the defect this class exists to remove"
                )
            charges[target] = int(charge)
            executable_initial[target] = bool(executable)
            block_initial[target] = str(reason)
            before = (float(gate.remaining_bytes), gate.sends_this_step)
            honours = gate.can_send(int(charge))
            block_effective[target] = gate.why_blocked(int(charge))
            candidate_shadow[target] = {
                "budget_before": before[0],
                "capacity_sent_before": before[1],
                "budget_after": float(gate.remaining_bytes),
                "capacity_sent_after": gate.sends_this_step,
                "honours_this_candidate": bool(honours),
            }
            if not honours:
                # Unaffordable in the order the transport would take this candidate: HOLD is not
                # a choice, it is the only legal action, so it must not earn actor gradient.
                actions[target] = HOLD
                masked.append(target)
                self.distribution_reads[
                    "masked_by_credit"
                    if block_effective[target] == "budget_exhausted"
                    else "masked_by_capacity"
                ] += 1
                if self._pending(target):
                    candidate_notes.append({
                        "step": int(observation.step), "target": target, "class": "masked",
                        "block_reason": block_effective[target],
                        "attempt_charge_bytes": int(charge),
                        "budget_before_bytes": before[0],
                        "greedy_action": "not_computed_no_legal_action",
                        "override_applied": False,
                        "outcome": "no_legal_opportunity_no_override",
                    })
                continue
            actionable.append(target)
            if self.greedy:
                action = SEND if probabilities[index, SEND] >= probabilities[index, HOLD] else HOLD
            else:
                action = int(torch.multinomial(probabilities[index], num_samples=1).item())
            # >>> ROUND27 intervention hook (the only inserted block in the parent loop) <<<
            action, note = self._first_report_hook(
                step=int(observation.step), item=item, action=action,
                send_probability=float(probabilities[index, SEND].item()),
                charge=int(charge), budget_before=before[0])
            if note is not None:
                if note["override_applied"]:
                    overrides_this_step.append(note)
                else:
                    candidate_notes.append(note)
            # >>> end of the ROUND27 hook <<<
            actions[target] = action
            requested[target] = bool(action == SEND)
            terms.append(float(log_probabilities[index, action].item()))
            send_probabilities[target] = float(probabilities[index, SEND].item())
            self.distribution_reads["free_candidates"] += 1
            self.distribution_reads["wanted_send"] += int(action == SEND)
            if action == SEND:
                # Only a sampled send spends the shadow. The gate's own charge() is the rule for
                # how much, so an arithmetic mistake here shows up as a refusal at execution.
                gate.charge(int(charge))
            candidate_shadow[target]["budget_after"] = float(gate.remaining_bytes)
            candidate_shadow[target]["capacity_sent_after"] = gate.sends_this_step
        asked = [target for target in actionable if actions[target] == SEND]
        tape: dict[str, Any] = {
            "schema": TAPE_SCHEMA,
            "action_mask_schema": ACTION_MASK_SCHEMA,
            "step": int(observation.step),
            "targets": [item.target_id for item in items],
            "features": [list(row) for row in rows],
            "actions": actions,
            # ``free`` keeps the parent's key and the parent's meaning as a set of *actionable*
            # targets: the two must not diverge, or the update and the audit read different sets.
            "free": list(actionable),
            "actionable": list(actionable),
            "forced": sorted(must),
            "masked": list(masked),
            "masked_reasons": {target: block_effective[target] for target in masked},
            "attempt_charge_bytes": charges,
            "send_executable_initial": executable_initial,
            "send_block_reason_initial": block_initial,
            "candidate_shadow": candidate_shadow,
            "shadow_budget_before": budget_at_start,
            "shadow_budget_after": float(gate.remaining_bytes),
            "shadow_capacity_before": capacity_at_start,
            "shadow_capacity_after": (
                None
                if capacity_at_start is None
                else int(capacity_at_start) - int(gate.sends_this_step)
            ),
            "requested": requested,
            "log_prob": float(sum(terms)) if terms else 0.0,
            "value": float(values.mean().item()) if items else 0.0,
            "send_probabilities": send_probabilities,
        }
        if candidate_notes or overrides_this_step:
            tape["first_report_overrides"] = {
                "schema": OVERRIDE_SCHEMA,
                "enabled": bool(self.override_enabled),
                "targets": sorted(self.override_targets),
                "consumed": sorted(self.consumed),
                "opportunities": candidate_notes + overrides_this_step,
                "overrides_applied_this_step": [note["target"] for note in overrides_this_step],
                "log_prob_note": (
                    "the overridden row's log-probability describes the forced action, not the "
                    "model's draw; this tape is inert and no optimizer consumes it"
                    if overrides_this_step else
                    "no override fired at this step"),
            }
        self._assert_closure(tape)
        return asked, actions, tape["free"], tape["log_prob"], tape["value"], tape
