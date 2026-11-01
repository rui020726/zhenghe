"""The reverse channel: acknowledgements that cost bytes and can be lost.

This module is the whole ACK protocol in one place, so the contract
(``COMM_ACK_CONTRACT.md``) is checkable without a relay, an episode or a checkpoint:

* an ACK exists only because a receiver *applied* an uplink packet -- the writer's
  ``deliver_detection`` returning True is the one and only trigger;
* the ACK carries the receiver's highest applied ``(sent_at, revision, content_fingerprint)``
  and is *cumulative*: a repeated packet re-confirms the same view without moving the state,
  and an out-of-order ACK can never move ``highest`` backwards;
* every ACK send *attempt* is billed at its serialized size, lost ones included, on a
  separate reverse ledger that never touches the task uplink credit;
* the reverse channel has the same declared delay and loss parameters as the uplink and is
  driven by a counter-based key, so two strategies facing the same key draw the same
  random number (common-random-number pairing) rather than merely the same distribution;
* when ACK is off, none of this is constructed: no state table, no reverse rng, no
  transcript writes, no change to the candidate order or the rng consumption.

The sender-visible side of this state is :class:`AckState`, summarised per subscriber by
:class:`AckView`. Nothing here reads a robot's belief, the registry, or a transcript: the
only inputs are what the receiver reported when it applied a packet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

#: Schema names. A checkpoint or artefact carrying a different one is a different experiment
#: and is refused, never silently reshaped.
ACK_SCHEMA = "comm-ack/1"
OBSERVATION_SCHEMA = "comm-send-state/2"
BILLING_VERSION = "ack-billing/1"

#: Normalisation constants of the ``/2`` feature block. Declared here, frozen before training:
#: changing one invalidates every checkpoint trained against it.
ACK_AGE_SCALE_STEPS = 100.0
VIEW_GAP_SCALE = 5.0
UNACKED_SCALE = 10.0

#: The "unknown" encoding. A negative value, so no feature built from it can read as a real
#: fraction or age: a receiver that has never confirmed is unknown, not confirmed-empty.
UNKNOWN = -1.0


def content_fingerprint(message: Any) -> str:
    """What the receiver actually applied, as a short stable string.

    The same content rule the uplink uses for "did this packet say anything new"
    (``comm._View``): fields a robot acts on. Two packets whose payload differs only in a
    field no robot reads must not look like two different views to the acknowledgement
    logic, or a redundant refresh would buy a confirmation it did not earn.
    """
    payload = {
        "stage_index": int(message.current_stage),
        "progress": round(float(message.progress), 6),
        "remaining": [int(value) for value in message.remaining_requirements],
        "members": [str(member) for member in message.members],
        "deadline": round(float(message.deadline), 6),
        "position": [round(float(message.position.x), 6), round(float(message.position.y), 6)],
        "confidence": round(float(message.confidence), 6),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def estimate_ack_bytes(ack: "AckMessage") -> int:
    """The reverse channel's price: UTF-8 length of the ACK's canonical JSON encoding.

    Same narrow rule as the uplink's ``estimate_message_bytes``: payload only, no radio
    header or framing. Billed per *attempt*, so a lost ACK costs what a delivered one costs.
    """
    payload = {
        "subscriber": ack.subscriber,
        "target_id": ack.target_id,
        "sent_at": ack.sent_at,
        "revision": ack.revision,
        "content_fingerprint": ack.content_fingerprint,
        "confirmed_at": ack.confirmed_at,
        "kind": ack.kind,
    }
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


@dataclass(frozen=True)
class AckMessage:
    """One acknowledgement frame: what one receiver applied, and when it said so."""

    subscriber: str
    target_id: str
    sent_at: float
    revision: int
    content_fingerprint: str
    confirmed_at: float
    kind: str = "cumulative"

    def bytes(self) -> int:
        return estimate_ack_bytes(self)

    def view_key(self) -> tuple[float, int, str]:
        """The cumulative triple. Ordering by this is what makes ACK monotone."""
        return (float(self.sent_at), int(self.revision), str(self.content_fingerprint))


@dataclass
class AckState:
    """One (subscriber, target) pair's confirmation state, on the sending side.

    ``highest`` is the last view this receiver reported as applied. ``last_arrival_step`` is
    when that report finished the reverse channel; ``reaffirmations`` counts reports that
    confirmed the same highest view again.
    """

    highest: tuple[float, int, str] | None = None
    last_arrival_step: int | None = None
    reaffirmations: int = 0

    def offer(self, ack: AckMessage, step: int) -> bool:
        """Record one arrived ACK; True when it advanced the highest confirmed view."""
        key = ack.view_key()
        if self.highest is None or key > self.highest:
            self.highest = key
            self.last_arrival_step = int(step)
            return True
        self.reaffirmations += 1
        return False

    def age_steps(self, step: int) -> float:
        if self.last_arrival_step is None:
            return UNKNOWN
        return float(max(0, int(step) - int(self.last_arrival_step)))


class AckLedger:
    """The reverse channel's own cost ledger: counted, never capped, never shared.

    Deliberately not a :class:`~integrated_sar.comm_budget.CommBudget`: the uplink credit is
    a hard constraint on the task channel, and an ACK that could be refused for lack of
    reverse credit would make "confirm everything" a strategy with a price the sender could
    feel. Here every attempt is billed, the total is reported, and nothing is ever dropped
    for money.
    """

    def __init__(self) -> None:
        self.attempt_bytes = 0
        self.attempts = 0
        self.delivered = 0
        self.lost = 0
        self.frames_offered = 0
        self.frames_advanced = 0
        self.frames_reaffirmed = 0

    def note_attempt(self, cost_bytes: int, *, delivered: bool) -> None:
        self.attempts += 1
        self.attempt_bytes += int(cost_bytes)
        if delivered:
            self.delivered += 1
        else:
            self.lost += 1

    def note_offer(self, *, advanced: bool) -> None:
        self.frames_offered += 1
        if advanced:
            self.frames_advanced += 1
        else:
            self.frames_reaffirmed += 1

    def usage(self) -> dict[str, int]:
        return {
            "attempts": self.attempts,
            "attempt_bytes": self.attempt_bytes,
            "delivered": self.delivered,
            "lost": self.lost,
            "frames_offered": self.frames_offered,
            "frames_advanced": self.frames_advanced,
            "frames_reaffirmed": self.frames_reaffirmed,
        }

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]


def reverse_rng_key(
    *, episode_seed: int, direction: str, subscriber: str, target_id: str, ordinal: int
) -> tuple[int, str, str, str, int]:
    """The counter-based key every channel draw is named by.

    Two strategies facing the same ``(episode_seed, direction, subscriber, target, ordinal)``
    draw the same number, so a loss or delay is a property of the *channel event*, not of the
    strategy that happened to reach it. A strategy that sends more simply names more keys.
    """
    return (int(episode_seed), str(direction), str(subscriber), str(target_id), int(ordinal))


def _stable_hash(text: str) -> int:
    """A stable 32-bit digest of a name (FNV-1a), for counter-based seeding.

    ``hash()`` is salted per process in CPython, so it cannot seed a stream two runs must
    agree on; this one is stable everywhere and needs no dependency.
    """
    digest = 2166136261
    for byte in str(text).encode("utf-8"):
        digest = ((digest ^ byte) * 16777619) & 0xFFFFFFFF
    return digest


def draw_channel(seed: int, direction: str, *, delivery_probability: float) -> bool:
    """One counter-based Bernoulli draw for a named direction of the reverse channel."""
    stream = np.random.default_rng([int(seed), _stable_hash(direction)])
    return bool(stream.random() < float(delivery_probability))


@dataclass
class AckFeedback:
    """Everything the sender has learned from the reverse channel, for one step.

    ``states`` is keyed ``(subscriber, target_id)``. ``broadcast_attempts`` counts the
    emissions the sender made for one target since its view last changed, which is what the
    ``unacked_broadcasts`` feature reads; it is the sender's own history, not a receiver's.
    """

    states: dict[tuple[str, str], AckState] = field(default_factory=dict)
    ledger: AckLedger = field(default_factory=AckLedger)
    broadcast_attempts: dict[str, int] = field(default_factory=dict)

    def state(self, subscriber: str, target_id: str) -> AckState:
        return self.states.setdefault((str(subscriber), str(target_id)), AckState())

    def reset(self) -> None:
        self.states.clear()
        self.ledger.reset()
        self.broadcast_attempts.clear()

    # ------------------------------------------------------------------ sender-visible reads
    def confirmed_fraction(self, target_id: str, subscribers: Iterable[str]) -> float:
        """How many configured receivers have confirmed the view the sender last sent."""
        names = [str(name) for name in subscribers]
        if not names:
            return UNKNOWN
        confirmed = sum(
            1
            for name in names
            if self.state(name, target_id).highest is not None
        )
        return confirmed / len(names)

    def max_ack_age(self, target_id: str, subscribers: Iterable[str], step: int) -> float:
        ages = [
            self.state(str(name), target_id).age_steps(step)
            for name in subscribers
        ]
        known = [age for age in ages if age >= 0.0]
        if not known:
            return UNKNOWN
        return max(known)

    def max_view_gap(self, target_id: str, subscribers: Iterable[str], current_key: tuple[float, int, str]) -> float:
        """Revision distance between the view in hand and the best confirmed one."""
        gaps: list[float] = []
        for name in subscribers:
            highest = self.state(str(name), target_id).highest
            if highest is None:
                continue
            gaps.append(float(max(0, int(current_key[1]) - int(highest[1]))))
        if not gaps:
            return UNKNOWN
        return max(gaps)

    def unacked_broadcasts(self, target_id: str) -> int:
        return int(self.broadcast_attempts.get(str(target_id), 0))
