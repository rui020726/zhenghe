"""A collector that samples only from the actions the transport could actually execute.

The archived sampler (``runs/comm_tight_budget_20260919/onpolicy.py``, round 4's fix for
*which* distribution gets logged) proved that the recorded probability describes the draw. It never
asked the other question: whether the draw was of something the relay could pay for. Under
policy-owned timing every non-forced candidate is handed to `multinomial`, so a SEND the credit
cannot cover is sampled, logged into the joint likelihood, trained on, and only then refused by
``CommRelay._emit`` -- the optimiser is spending gradient on an action that does not exist.

Deleting refused samples after the episode is not a fix: whether a send was refused depends on
which SENDs earlier in the same step were drawn, so filtering by the realised actions changes the
joint probability selectively and the logged and recomputed numbers stop describing one
distribution. The correct move is to apply the constraint *before* the draw, in the order the
transport will execute in -- which is what this class does:

* each candidate carries the price ``_emit`` would charge it, computed by the relay itself;
* a shadow credit walks the candidates in that same order and **is the same gate object**
  (``CommBudget``), so there is no second implementation of "affordable" to drift;
* a candidate the shadow cannot pay for has exactly one legal action, HOLD, and contributes no
  likelihood and no entropy;
* a candidate that can be paid for is sampled SEND/HOLD, and only a sampled SEND spends credit.

``budget_mask=False`` delegates to the archived collector without touching it, so the historical
tape, actions and log-probabilities stay reproducible byte for byte rather than being redefined.
The module imports no checkpoint and no stream: the property "the tape's actionable set is the set
the executor would honour" must be checkable from a synthetic candidate list alone.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch

from integrated_sar.comm_budget import CommBudget
from integrated_sar.comm_learning import features
from integrated_sar.comm_policy import SendObservation

SEND, HOLD = 1, 0

#: Identity of this sampler. A checkpoint may only be resumed by a trainer that says the same
#: three things, because all three change which actions the policy was ever shown.
TRAINER_TAG = "onpolicy-v3-budget-mask"
ACTION_MASK_SCHEMA = "budget-sequential-mask/1"
TAPE_SCHEMA = "onpolicy-tape/2"

__all__ = [
    "SEND",
    "HOLD",
    "TRAINER_TAG",
    "ACTION_MASK_SCHEMA",
    "TAPE_SCHEMA",
    "BudgetMaskedCollector",
    "identity",
    "shadow_gate",
]


def identity() -> dict[str, Any]:
    """What a run trained with this collector must record about itself."""
    return {
        "trainer_tag": TRAINER_TAG,
        "action_mask_schema": ACTION_MASK_SCHEMA,
        "tape_schema": TAPE_SCHEMA,
        "collector": "budget-masked-onpolicy",
        "candidate_order": "relay._observation order (sorted target ids), which is also the "
        "order the transport emits the policy's selection in",
        "feature_layout_unchanged": True,
    }


def shadow_gate(observation: SendObservation) -> CommBudget:
    """A throwaway copy of the step's remaining credit and capacity, built from the observation.

    The gate is the production object, not a restatement of its arithmetic: ``can_send`` and
    ``why_blocked`` here answer exactly as the relay's own budget answers them for the rest of
    this step. What is *carried over* is the state, not a fresh credit -- the observation reports
    the episode's remaining bytes, so the shadow starts at the amount already spent rather than at
    the full allowance. Starting from the allowance would make the mask see a budget that has
    already gone, which is the same blindness this round exists to remove, only one step earlier.
    """
    gate = CommBudget(
        bytes_per_episode=observation.budget_bytes_per_episode,
        sends_per_step=observation.capacity_remaining,
    )
    if observation.budget_bytes_per_episode is not None:
        credit = int(observation.budget_bytes_per_episode)
        spent = credit - int(float(observation.remaining_budget_bytes))
        if not 0 <= spent <= credit:
            raise RuntimeError(
                f"observation reports {observation.remaining_budget_bytes!r} remaining of a {credit} byte "
                f"credit, which reads as {spent} bytes spent"
            )
        gate.spent_bytes = spent
    return gate


class BudgetMaskedCollector:
    """Decide send or hold per candidate, but only over what the credit can actually pay for.

    ``choose`` returns the same six-tuple as the archived collector so the pilot's rollout loop
    does not change shape; the tape it appends gains the executability bookkeeping the PPO update
    and the audit both read. With ``budget_mask`` off the class is a thin pass-through to the
    archived collector, tape included.
    """

    def __init__(
        self,
        controller: Any,
        *,
        forced: Callable[[Any], bool],
        mask: bool = True,
        greedy: bool = False,
        budget_mask: bool = True,
    ) -> None:
        if controller is None:
            raise ValueError("a learned collector needs a controller")
        # The archived sampler is the off-switch's implementation: with the mask disabled this
        # class hands the whole decision to it, so "unchanged" is a delegation rather than a
        # promise to keep two copies of the old loop in step.
        import onpolicy

        self._delegate = onpolicy.OnPolicyCollector(
            controller, forced=forced, mask=mask, greedy=greedy
        )
        self.controller = controller
        self.forced = forced
        self.mask = bool(mask)
        self.greedy = bool(greedy)
        self.budget_mask = bool(budget_mask)
        self.records: list[dict[str, Any]] = []
        self.send_steps: list[tuple[int, tuple[str, ...]]] = []
        self.decide_seconds = 0.0
        self.decisions = 0
        self.candidates = 0
        self.forced_sends = 0
        #: The two numbers this round exists to make measurable: how often a candidate was
        #: genuinely a choice, and how often the credit had already decided it.
        self.distribution_reads: dict[str, float] = {
            "free_candidates": 0,
            "wanted_send": 0,
            "masked_by_credit": 0,
            "masked_by_capacity": 0,
        }

    # ------------------------------------------------------------------ the policy seam
    def policy(self) -> Any:
        from integrated_sar.comm_learning import LearnedSendPolicy

        return LearnedSendPolicy(self.decide)

    # ------------------------------------------------------------------ sampling
    def choose(
        self, observation: SendObservation
    ) -> tuple[list[str], dict[str, int], list[str], float, float, dict[str, Any] | None]:
        if not self.budget_mask:
            asked, actions, free, log_prob, value, tape = self._delegate.choose(observation)
            self.forced_sends += len(tape["forced"]) if tape else 0
            self.distribution_reads["free_candidates"] += len(free)
            self.distribution_reads["wanted_send"] += sum(
                1 for target in free if actions.get(target) == SEND
            )
            return asked, actions, free, log_prob, value, tape
        return self._choose_masked(observation)

    def _choose_masked(
        self, observation: SendObservation
    ) -> tuple[list[str], dict[str, int], list[str], float, float, dict[str, Any]]:
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
        for index, item in enumerate(items):
            target = item.target_id
            requested[target] = False
            if target in must:
                # The silence bound's decision, not ours: it goes out and it is billed, and it
                # carries no likelihood because nothing chose it.
                actions[target] = SEND
                requested[target] = True
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
                continue
            actionable.append(target)
            if self.greedy:
                action = SEND if probabilities[index, SEND] >= probabilities[index, HOLD] else HOLD
            else:
                action = int(torch.multinomial(probabilities[index], num_samples=1).item())
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
        self._assert_closure(tape)
        return asked, actions, tape["free"], tape["log_prob"], tape["value"], tape

    @staticmethod
    def _assert_closure(tape: dict[str, Any]) -> None:
        """Forced, masked and actionable are disjoint and together are every candidate."""
        targets = list(tape["targets"])
        forced = set(tape["forced"])
        masked = set(tape["masked"])
        actionable = set(tape["actionable"])
        if len(targets) != len(set(targets)):
            raise RuntimeError(f"one candidate id appears twice: {targets}")
        if forced & masked or forced & actionable or masked & actionable:
            raise RuntimeError(
                f"candidate classes overlap: forced={sorted(forced)} masked={sorted(masked)} "
                f"actionable={sorted(actionable)}"
            )
        if forced | masked | actionable != set(targets):
            raise RuntimeError(
                "candidate closure broken: "
                f"{len(targets)} candidates, {len(actionable)} actionable + {len(forced)} forced "
                f"+ {len(masked)} masked = {len(actionable) + len(forced) + len(masked)}"
            )
        if any(tape["requested"][target] for target in forced | masked):
            raise RuntimeError("a non-actionable candidate was requested for emission")

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        started = time.perf_counter()
        self.decisions += 1
        self.candidates += len(observation.items)
        asked, _actions, _free, _log_prob, _value, tape = self.choose(observation)
        if tape is not None:
            self.records.append(tape)
        if asked:
            self.send_steps.append((int(observation.step), tuple(asked)))
        self.decide_seconds += time.perf_counter() - started
        return tuple(asked)

    # ------------------------------------------------------------------ reporting
    def timing(self) -> dict[str, float]:
        return {
            "decisions": self.decisions,
            "candidates_seen": self.candidates,
            "forced_sends": self.forced_sends,
            "free_candidates": int(self.distribution_reads["free_candidates"]),
            "masked_by_credit": int(self.distribution_reads["masked_by_credit"]),
            "masked_by_capacity": int(self.distribution_reads["masked_by_capacity"]),
            "actionable_share": round(
                self.distribution_reads["free_candidates"] / max(1, self.candidates), 6
            ),
            "voluntary_send_rate": round(
                self.distribution_reads["wanted_send"]
                / max(1, self.distribution_reads["free_candidates"]),
                4,
            ),
            "decide_milliseconds_per_call": round(
                1000.0 * self.decide_seconds / max(1, self.decisions), 4
            ),
            "trainer_tag": TRAINER_TAG,
            "action_mask_schema": ACTION_MASK_SCHEMA,
            "tape_schema": TAPE_SCHEMA,
        }
