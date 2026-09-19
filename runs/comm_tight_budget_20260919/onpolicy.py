"""A sampler that records the probability distribution it actually drew from.

Round 2 trained with ``mu(a|s) = 0.9 * pi(a|s) + 0.1 * Uniform(a)`` and wrote ``log pi(a|s)`` into
the rollout. Round 3 registered the mismatch in its own protocol and trained on anyway, because
that round was allowed to change only duration. This module is the fix: a free candidate is drawn
from the masked policy itself, and the number stored beside it is the log-probability of that
same policy, so the PPO ratio compares a sampling distribution with itself.

Two properties are load-bearing and are tested directly:

* **Nothing is logged that was not sampled.** The epsilon-greedy overlay is gone; exploration is
  the stochastic draw plus the entropy bonus, which is what the update actually optimises.
* **Nothing is sampled that was not chosen.** A candidate the silence bound has already taken over
  is not the policy's action, so it contributes no likelihood and no entropy, and -- because the
  transport applies the bound to every arm alike -- it is not even handed back as a selection.

The module imports no experiment fixtures on purpose: the correctness of a probability statement
must be checkable without a checkpoint, a recorded stream or an episode runner.
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable

import numpy as np
import torch

from integrated_sar.comm_learning import features
from integrated_sar.comm_policy import SendObservation

SEND, HOLD = 1, 0

__all__ = [
    "SEND",
    "HOLD",
    "OnPolicyCollector",
    "action_entropy",
    "joint_log_prob",
    "rng_state",
    "restore_rng",
]


def joint_log_prob(
    logits: torch.Tensor, picked: torch.Tensor, usable: torch.Tensor
) -> torch.Tensor:
    """Log-probability of one step's joint action: the free candidates' terms, summed.

    ``usable`` is the actionable set recorded at selection time, so a candidate the bound had
    taken over multiplies in as zero -- no likelihood, and therefore no actor gradient.
    """
    log_probabilities = torch.log_softmax(logits, dim=-1)
    rows = torch.arange(len(picked))
    return (log_probabilities[rows, picked] * usable).sum()


def action_entropy(logits: torch.Tensor, usable: torch.Tensor) -> torch.Tensor:
    """Entropy of the same set and only the same set, so bonus and likelihood agree."""
    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    per_candidate = -(probabilities * log_probabilities).sum(dim=-1)
    return (per_candidate * usable).sum()


def rng_state() -> dict[str, Any]:
    """Every stream a run draws randomness from: the coin, the shuffle, the network."""
    python = random.getstate()
    numpy = np.random.get_state()
    return {
        "python": [python[0], list(python[1]), python[2]],
        "numpy": [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]],
        "torch": torch.get_rng_state().tolist(),
    }


def restore_rng(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    python = payload["python"]
    random.setstate((python[0], tuple(python[1]), python[2]))
    numpy = payload["numpy"]
    np.random.set_state(
        (numpy[0], np.asarray(numpy[1], dtype=np.uint32), numpy[2], numpy[3], numpy[4])
    )
    torch.set_rng_state(torch.tensor(payload["torch"], dtype=torch.uint8))
    return True


class OnPolicyCollector:
    """Decide send or hold for every candidate the sender is shown, and keep the tape.

    The interface is the one the relay's policy seam and the pilot's ``run_one`` already expect
    (``policy()``, ``decide()``, ``records``, ``timing()``), so swapping this in changes how an
    action is drawn and logged -- nothing about the protocol around it.
    """

    def __init__(
        self,
        controller: Any | None = None,
        *,
        forced: Callable[[Any], bool],
        mask: bool = True,
        greedy: bool = False,
        always_send: bool = False,
        never_send: bool = False,
    ) -> None:
        if not always_send and not never_send and controller is None:
            raise ValueError("a learned collector needs a controller; a rule needs a mode")
        self.controller = controller
        self.forced = forced
        self.mask = bool(mask)
        self.greedy = bool(greedy)
        self.always_send = bool(always_send)
        self.never_send = bool(never_send)
        self.records: list[dict[str, Any]] = []
        self.send_steps: list[tuple[int, tuple[str, ...]]] = []
        self.decide_seconds = 0.0
        self.decisions = 0
        self.candidates = 0
        self.forced_sends = 0
        #: Filled by :meth:`choose` on every learned decision: how often the policy wanted the
        #: air, counted separately from what the bound handed it.
        self.distribution_reads: dict[str, float] = {"free_candidates": 0, "wanted_send": 0}

    # ------------------------------------------------------------------ the policy seam
    def policy(self) -> Any:
        from integrated_sar.comm_learning import LearnedSendPolicy

        return LearnedSendPolicy(self.decide)

    # ------------------------------------------------------------------ sampling
    def choose(
        self, observation: SendObservation
    ) -> tuple[list[str], dict[str, int], list[str], float, float, dict[str, Any] | None]:
        """Return (asked, actions, actionable, log_prob, value, tape).

        ``asked`` deliberately excludes bound-forced candidates: they are the transport's
        decision, and handing them back as ours both billed them as policy and put them ahead of
        a rule's equivalent sends in the queue when the credit ran out. ``tape`` is None for a
        rule arm, which has no likelihood to train on.
        """
        items = list(observation.items)
        must = {item.target_id for item in items if self.mask and self.forced(item)}
        self.forced_sends += len(must)
        free = [item.target_id for item in items if item.target_id not in must]
        if self.controller is None or self.always_send or self.never_send:
            if self.never_send:
                actions = {
                    item.target_id: (SEND if item.target_id in must else HOLD) for item in items
                }
            else:
                actions = {item.target_id: SEND for item in items}
            asked = [key for key, value in actions.items() if value == SEND and key not in must]
            return asked, actions, free, 0.0, 0.0, None
        rows = [features(observation, item) for item in items]
        matrix = torch.as_tensor(np.asarray(rows, dtype=np.float32))
        logits, values = self.controller.forward(matrix)
        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=-1)
            log_probabilities = torch.log_softmax(logits, dim=-1)
        actions: dict[str, int] = {}
        send_probabilities: dict[str, float] = {}
        terms: list[float] = []
        for index, item in enumerate(items):
            if item.target_id in must:
                # Not our action: recorded as SEND because it will go out, logged as nothing
                # because we did not choose it.
                actions[item.target_id] = SEND
                continue
            if self.greedy:
                action = SEND if probabilities[index, SEND] >= probabilities[index, HOLD] else HOLD
            else:
                action = int(torch.multinomial(probabilities[index], num_samples=1).item())
            actions[item.target_id] = action
            terms.append(float(log_probabilities[index, action].item()))
            send_probabilities[item.target_id] = float(probabilities[index, SEND].item())
            self.distribution_reads["free_candidates"] += 1
            self.distribution_reads["wanted_send"] += int(action == SEND)
        asked = [
            item.target_id
            for item in items
            if actions[item.target_id] == SEND and item.target_id not in must
        ]
        tape: dict[str, Any] = {
            "step": int(observation.step),
            "targets": [item.target_id for item in items],
            "features": [list(row) for row in rows],
            "actions": actions,
            "free": free,
            "log_prob": float(sum(terms)) if terms else 0.0,
            "value": float(values.mean().item()) if items else 0.0,
            "send_probabilities": send_probabilities,
            "forced": sorted(must),
        }
        return asked, actions, free, tape["log_prob"], tape["value"], tape

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
            "free_candidates": self.distribution_reads["free_candidates"],
            "voluntary_send_rate": round(
                self.distribution_reads["wanted_send"]
                / max(1, self.distribution_reads["free_candidates"]),
                4,
            ),
            "decide_milliseconds_per_call": round(
                1000.0 * self.decide_seconds / max(1, self.decisions), 4
            ),
        }
