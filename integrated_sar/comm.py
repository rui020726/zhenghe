"""The communication bridge between search detections and rescue robots' beliefs.

Architecture invariant 9: actors and the online coordinator may only use beliefs that
actually arrived over a communication link. This module is that link. The event bus fans a
published :class:`~integrated_sar.contracts.DetectionEvent` out once, the relay records what
the registry accepted, and a :mod:`send policy <integrated_sar.comm_policy>` decides which
task views go on the air. Delivery then obeys an explicit topology, a per-packet delay and a
seeded packet-loss process.

What the relay guarantees, independently of the policy attached to it:

* at most one emission per (target, step), and its payload is the newest registry snapshot at
  the step of emission -- a change that is held is *superseded* by the next emission, never
  sent stale, so a policy that sends less sends later and fresher rather than thinner;
* only subscribed robots receive it, every packet is delivered or lost independently, and the
  relay never retransmits on its own -- only a policy that emits again can recover a loss;
* the relay never reads a robot's position, so search and rescue robots remain two
  independent entity sets (invariant 10) and co-location with the search start is never a
  precondition for the loop to start;
* the stochastic process (which packets survive) is driven entirely by the seed;
* a task the registry has already closed is not announced: "there is a task here" would be
  false information at the moment of sending, and the suppressed emission is counted.

Once a robot knows a task, TaskEnv's own line-of-sight refresh and one-hop gossip can still
reach it without a packet from here. That bypass belongs to the environment; it is measured
(``runs/comm_control_20260919/INFORMATION_BOUNDARY.md``), not hidden.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from .contracts import DetectionEvent, DetectionMessage, TaskSnapshot
from .comm_budget import CommBudget
from .comm_ack import (
    ACK_AGE_SCALE_STEPS,
    ACK_SCHEMA,
    UNACKED_SCALE,
    VIEW_GAP_SCALE,
    AckFeedback,
    AckLedger,
    AckMessage,
    content_fingerprint,
    estimate_ack_bytes,
    reverse_rng_key,
)
from .comm_policy import (
    CommSendPolicy,
    ImmediatePolicy,
    SendItem,
    SendObservation,
    transport_order,
)

SnapshotLookup = Callable[[str], TaskSnapshot | None]
AcceptanceLookup = Callable[[str], bool]
NowLookup = Callable[[], float]


def _stable_hash(text: str) -> int:
    """A stable 32-bit digest of a name, for counter-based seeding.

    ``hash()`` is salted per process in CPython, so it cannot seed a stream two runs must
    agree on; this one is a plain FNV-1a over the UTF-8 bytes, which is stable everywhere.
    """
    digest = 2166136261
    for byte in str(text).encode("utf-8"):
        digest = ((digest ^ byte) * 16777619) & 0xFFFFFFFF
    return digest


# Read-only "what does this robot already believe" lookup, used only to classify a refused
# delivery as duplicate or superseded. Never consulted before deciding what to send.
FreshnessLookup = Callable[[str, str], "tuple[float, int] | None"]

#: Who decides that a change which has sat unsaid for ``max_wait_steps`` goes on the air.
#:
#: ``legacy_shared_bound`` is every experiment recorded since the bound existed: the transport
#: appends the overdue changes to whatever the arm asked for, bills them, and refuses them when
#: the credit cannot pay. ``policy_owned_timing`` hands that decision to the arm -- nothing is
#: emitted because a clock ran out, and a request is never rewritten into a send.
#:
#: Only that one mechanism differs. Candidate generation, registry polling, the per-step call,
#: message merging, the byte credit, the per-step capacity, the terminal-suppression rule and the
#: payload's "newest snapshot at the moment of sending" are identical in both modes, and so is the
#: loss process. A default-constructed relay is the legacy one, so an untouched configuration
#: cannot drift into the new protocol by omission.
LEGACY_SHARED_BOUND = "legacy_shared_bound"
POLICY_OWNED_TIMING = "policy_owned_timing"
TIMING_MODES: tuple[str, ...] = (LEGACY_SHARED_BOUND, POLICY_OWNED_TIMING)


def estimate_message_bytes(message: DetectionMessage) -> int:
    """Payload size in bytes: the UTF-8 length of the message's canonical JSON encoding.

    An estimate, and an explicitly narrow one. It counts the fields a receiver has to apply
    and nothing else: no radio header, no framing, no link-layer overhead, no ack. Those are
    constant per packet, so a comparison that holds here holds after adding them, but the
    absolute number is payload only. One emission to ``n`` subscribers costs ``n`` packets of
    this size, which is why batch size (emissions per step) and information volume (bytes)
    are reported separately.
    """
    payload = {
        "event_id": message.event_id,
        "target_id": message.target_id,
        "sent_at": message.sent_at,
        "position": [message.position.x, message.position.y],
        "stages": [[list(stage.requirements), stage.duration] for stage in message.stages],
        "current_stage": message.current_stage,
        "remaining_requirements": list(message.remaining_requirements),
        "deadline": message.deadline,
        "severity": message.severity,
        "confidence": message.confidence,
        "progress": message.progress,
        "members": list(message.members),
        "discovered_by": message.discovered_by,
        "revision": message.revision,
    }
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


@dataclass(frozen=True)
class _View:
    """The part of a task view a receiver would actually apply, for "have I said this?".

    Two snapshots that differ only in fields no robot reads are not worth a packet, and two
    that differ here are. ``revision`` is deliberately excluded: it says the registry moved,
    not that anything a robot would act on moved.
    """

    stage_index: int
    progress: float
    remaining: tuple[int, ...]
    members: tuple[str, ...]
    deadline: float
    position: tuple[float, float]
    confidence: float

    @classmethod
    def of(cls, state: Any) -> "_View":
        """From either side of the link: a registry snapshot and a sent message agree here."""
        return cls(
            stage_index=int(state.current_stage),
            progress=round(float(state.progress), 6),
            remaining=tuple(int(value) for value in state.remaining_requirements),
            members=tuple(str(member) for member in state.members),
            deadline=round(float(state.deadline), 6),
            position=(round(float(state.position.x), 6), round(float(state.position.y), 6)),
            confidence=round(float(state.confidence), 6),
        )


@dataclass
class _TargetLedger:
    """The sending node's own record for one target: what it said, and what moved since."""

    ready_step: int | None = None
    pending_message: DetectionMessage | None = None
    last_sent: DetectionMessage | None = None
    last_sent_view: "_View | None" = None
    #: The newest view the sender has already booked an item against -- ``last_sent_view`` says
    #: what went out, this says what it came to know. Without it a change that arrives while an
    #: older one is still unsent is invisible to both ``generated_items`` and
    #: ``coalesced_changes``, because neither counter can tell it from the one already waiting.
    ready_view: "_View | None" = None
    last_sent_step: int | None = None
    sent_count: int = 0
    pending_event: bool = False
    closed: bool = False
    last_sent_now: float | None = None


class CommTranscript:
    """Append-only record of what the relay was asked to do, for offline measurement.

    Written after a step and never read back, so attaching one cannot change what a policy or
    a robot does. Deliberately per-event, not per-tick: queue depths are aggregated in
    :attr:`CommRelay.counters` so a 1000-step episode stays a few hundred lines.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add(self, kind: str, **fields: Any) -> None:
        self.records.append({"kind": kind, **fields})

    def write(self, path: Any) -> Any:
        """Write one JSON object per line; returns the path so a caller can record it."""
        target = str(path)
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return target


class CommRelay:
    """Holds the task views the registry accepted and emits the ones its policy selects.

    Both happen in :meth:`tick`, which the executor calls once per joint step before the
    simulated world advances: the policy picks, then packets whose due step has come are
    delivered. A packet emitted on step ``N`` is due at ``N + delay_ticks``, so
    ``delay_ticks=0`` arrives within the same joint step and ``delay_ticks=1`` on the next
    one -- which is what the relay did before a policy was involved.

    ``accepted_of`` is the registry's verdict lookup: an event the registry rejected (unknown
    task, late timestamp, terminal task) is not news, so the relay must not treat it as a
    change worth sending. Subscribing the relay after the registry guarantees the verdict is
    already recorded when the relay is called.
    """

    def __init__(
        self,
        *,
        subscribers,
        writer,
        snapshot_of: SnapshotLookup | None = None,
        accepted_of: AcceptanceLookup | None = None,
        delivery_probability: float = 1.0,
        delay_ticks: int = 0,
        seed: int | None = None,
        policy: CommSendPolicy | None = None,
        sends_per_tick: int | None = None,
        budget: CommBudget | None = None,
        max_wait_steps: int | None = None,
        urgent_wait_steps: int | None = None,
        urgency_time: float = 25.0,
        timing_mode: str = LEGACY_SHARED_BOUND,
        freshness_of: FreshnessLookup | None = None,
        now_of: NowLookup | None = None,
        transcript: CommTranscript | None = None,
        ack_enabled: bool = False,
        ack_delay_ticks: int | None = None,
        ack_delivery_probability: float | None = None,
        ack_seed: int | None = None,
    ) -> None:
        if not 0.0 <= float(delivery_probability) <= 1.0:
            raise ValueError(
                f"delivery_probability must stay in [0, 1], got {delivery_probability}"
            )
        if int(delay_ticks) < 0:
            raise ValueError(f"delay_ticks cannot be negative, got {delay_ticks}")
        if sends_per_tick is not None and int(sends_per_tick) < 1:
            raise ValueError(f"sends_per_tick must be >= 1 or None, got {sends_per_tick}")
        if budget is not None and sends_per_tick is not None and budget.sends_per_step != int(
            sends_per_tick
        ):
            raise ValueError(
                "pass the emission capacity either as sends_per_tick or inside budget, not as "
                f"two different values ({sends_per_tick} vs {budget.sends_per_step})"
            )
        self._subscribers = tuple(str(agent_id) for agent_id in subscribers)
        self._writer = writer
        self._snapshot_of = snapshot_of
        self._accepted_of = accepted_of
        self._delivery_probability = float(delivery_probability)
        self._delay_ticks = int(delay_ticks)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._policy = policy if policy is not None else ImmediatePolicy()
        self._budget = budget if budget is not None else CommBudget(sends_per_step=sends_per_tick)
        # One silence bound for every arm, transport-side rather than policy-side: a change may
        # not be held past max_wait_steps, or past urgent_wait_steps while a task is inside
        # urgency_time. Unset leaves the relay exactly as it was. It is a floor on what goes
        # out, still billed and still refused when the credit cannot pay for it.
        if max_wait_steps is not None and int(max_wait_steps) < 1:
            raise ValueError(f"max_wait_steps must be >= 1 or None, got {max_wait_steps}")
        mode = str(timing_mode or "").strip().lower()
        if mode not in TIMING_MODES:
            raise ValueError(
                f"unknown timing mode {timing_mode!r}; available: {', '.join(TIMING_MODES)}"
            )
        # The bound constants are *declared* under either mode. What the mode decides is whether
        # the transport acts on them: under policy_owned_timing they stay here as the numbers a
        # rule may quote (and as a record of what the legacy floor was), and no emission follows
        # from them by itself.
        self._timing_mode = mode
        self._max_wait_steps = None if max_wait_steps is None else int(max_wait_steps)
        self._urgent_wait_steps = (
            None if urgent_wait_steps is None else max(0, int(urgent_wait_steps))
        )
        self._urgency_time = float(urgency_time)
        self._freshness_of = freshness_of
        self._now_of = now_of
        self.transcript = transcript
        # The reverse channel. Off by default and constructed only when asked for: with ACK off
        # there is no state table, no reverse rng, no extra transcript kind and no extra rng
        # consumption, which is what keeps the historical path byte-identical. The declared
        # delay and loss are the *same frozen numbers* the uplink runs with unless a caller
        # passes its own, so the reverse channel is not a second, friendlier uplink.
        self._ack_enabled = bool(ack_enabled)
        self._ack_delay_ticks = (
            self._delay_ticks if ack_delay_ticks is None else int(ack_delay_ticks)
        )
        self._ack_delivery_probability = (
            float(delivery_probability)
            if ack_delivery_probability is None
            else float(ack_delivery_probability)
        )
        if self._ack_enabled:
            if not 0.0 <= self._ack_delivery_probability <= 1.0:
                raise ValueError(
                    f"ack_delivery_probability must stay in [0, 1], got {ack_delivery_probability}"
                )
            if self._ack_delay_ticks < 0:
                raise ValueError(
                    f"ack delay_ticks cannot be negative, got {ack_delay_ticks}"
                )
            self._ack_seed = self._seed if ack_seed is None else int(ack_seed)
            self.ack = AckFeedback(ledger=AckLedger())
            self._ack_pending: list[tuple[int, int, str, str, AckMessage, int]] = []
            self._ack_ordinal = 0
        else:
            self.ack = AckFeedback(ledger=AckLedger())
            self._ack_pending = []
            self._ack_ordinal = 0
        self._pending: list[tuple[int, int, str, DetectionMessage, int, int]] = []
        self._ledger: dict[str, _TargetLedger] = {}
        self._seen_event_ids: set[str] = set()
        self._tick = 0
        self._sequence = 0
        # The transport ledger, reported once per episode by the run harness. The naming is
        # deliberate because these quantities are routinely confused: an *attempt* is radio
        # time spent and is what a budget is charged against, a *delivery* is a packet that
        # reached a receiver, and an *application* is a delivery that changed a belief.
        # Losing packets reduces deliveries, never attempts, so a lossy channel cannot be
        # reported as savings.
        self.counters: dict[str, int] = {
            key: 0
            for key in (
                # what the sender came to know, and what it said
                "generated_items",
                "events_accepted",
                "coalesced_changes",
                "emissions",
                "first_emissions",
                # Three different things that all sound like "the same thing twice", counted
                # separately on purpose: ``coalesced_changes`` is production-side (two pieces of
                # news, one letter), ``redundant_emissions`` is send-side (a letter whose content
                # was already said unchanged), and ``delivery_duplicate`` is one receiver
                # refusing a packet it already believed.
                "redundant_emissions",
                "refresh_emissions",
                # what went on the air, and what came out the other end
                "transmission_attempts",
                "packets_lost",
                "packets_delivered",
                "deliveries_applied",
                "delivery_duplicate",
                "delivery_superseded",
                "delivery_unknown_task",
                "delivery_refused",
                # what the transport rules refused to say
                "emissions_suppressed_terminal",
                "capacity_refused_sends",
                "budget_refused_sends",
                # Who each send was really for. The three reasons are mutually exclusive and
                # cover every attempt, so an arm that only ever speaks because the bound ran
                # out cannot be read as an arm that learned when to talk. ``policy`` counts a
                # send the sender asked for even when the change was also overdue: it asked.
                *[
                    f"{metric}_{reason}"
                    for metric in ("emissions", "attempts", "bytes", "refused")
                    for reason in ("policy", "forced_maxwait", "other")
                ],
                # cost: primary is airtime over attempts; applied bytes is an efficiency read
                "airtime_bytes",
                "applied_bytes",
                # Two different waits. backlog_steps is how long the packets that *did* go
                # out sat first; unsent_change_steps is the integral of everything the sender
                # knew and had not said, which a policy that never refreshes keeps growing.
                "backlog_steps",
                "unsent_change_steps",
                "flight_steps",
                "max_in_flight",
                "max_backlog_age",
                # The reverse channel, all zero unless ACK is on. ``ack_emitted`` counts
                # frames the receivers actually asked to send (one per applied packet, merged
                # per receiver/task/tick); ``ack_attempts``/``ack_attempt_bytes`` are the
                # billed attempts on the reverse channel, losses included; ``ack_delivered``
                # and ``ack_lost`` split them; ``ack_frames_advanced``/``ack_frames_reaffirmed``
                # say whether an arrived frame moved the highest confirmed view or re-confirmed
                # the same one.
                "ack_emitted",
                "ack_attempts",
                "ack_attempt_bytes",
                "ack_delivered",
                "ack_lost",
                "ack_frames_advanced",
                "ack_frames_reaffirmed",
            )
        }

    # ------------------------------------------------------------------ production side
    def handle_detection(self, event: DetectionEvent) -> None:
        """Bus handler: record that the registry accepted news about one target.

        Nothing is sent here. Whether anything goes on the air, and when, is the policy's
        decision, taken once per step inside :meth:`tick`.
        """
        if event.event_id in self._seen_event_ids:
            return
        self._seen_event_ids.add(event.event_id)
        if self._accepted_of is not None and not self._accepted_of(event.event_id):
            # The registry refused this event: it is not a new lifecycle fact, so
            # forwarding it would let a rejected detection re-send an old task
            # view or reopen a terminal one in robot belief.
            return
        ready_step = self._tick + 1
        ledger = self._ledger.setdefault(str(event.target_id), _TargetLedger())
        if ledger.ready_step is not None:
            # Something about this target was already waiting to be said. This event joins it
            # instead of queueing a second letter: the emission will carry one newer view.
            self.counters["coalesced_changes"] += 1
        ledger.pending_event = True
        ledger.closed = False
        # The event has now been booked as one item (and as one merge, if something was already
        # waiting). What it actually says is only known when the registry is read, so clear the
        # content marker: the next poll records that content without counting it a second time.
        ledger.ready_view = None
        if ledger.ready_step is None:
            ledger.ready_step = ready_step
        self.counters["events_accepted"] += 1
        self.counters["generated_items"] += 1
        if self._snapshot_of is None:
            # No registry lookup wired: the event itself -- the same profile the registry
            # would have stored -- defines the payload, as it always did.
            ledger.pending_message = self._message_from_event(event)
        if self.transcript is not None:
            snapshot = self._snapshot_of(event.target_id) if self._snapshot_of else None
            self.transcript.add(
                "event",
                step=ready_step,
                target=str(event.target_id),
                event_id=str(event.event_id),
                revision=int(getattr(snapshot, "revision", 0) or 0),
            )

    # ------------------------------------------------------------------ transport step
    def tick(self) -> tuple[str, ...]:
        """One send opportunity and one delivery opportunity; returns belief changes."""
        self._tick += 1
        self._emit_selected()
        due = [item for item in self._pending if item[0] <= self._tick]
        self._pending = [item for item in self._pending if item[0] > self._tick]
        changed: list[str] = []
        if due:
            due.sort(key=lambda item: item[1])
            for due_tick, _, agent_id, message, sent_step, payload_bytes in due:
                self.counters["packets_delivered"] += 1
                self.counters["flight_steps"] += int(self._tick - due_tick)
                accepted, reason = self._deliver(message, agent_id)
                if accepted:
                    changed.append(agent_id)
                    self.counters["deliveries_applied"] += 1
                    self.counters["applied_bytes"] += int(payload_bytes)
                    # The one and only ACK trigger: the receiver applied this packet. A lost
                    # packet never reaches here, and a refused one returns False, so neither
                    # can produce an acknowledgement.
                    self._ack_applied(agent_id, message)
                elif reason in self.counters:
                    self.counters[reason] += 1
                if self.transcript is not None:
                    self.transcript.add(
                        "deliver",
                        step=int(self._tick),
                        target=message.target_id,
                        agent=agent_id,
                        revision=int(message.revision),
                        sent_at=float(message.sent_at),
                        sent_step=int(sent_step),
                        now=float(self._now_value()),
                        payload_bytes=int(payload_bytes),
                        accepted=bool(accepted),
                        reason="accepted" if accepted else str(reason),
                    )
        # The reverse channel runs after this step's deliveries and before the next send
        # decision, so an ACK generated here is visible no earlier than the next step.
        self._deliver_acks()
        self._note_queue_depth()
        return tuple(changed)

    def _deliver(self, message: DetectionMessage, agent_id: str) -> tuple[bool, str]:
        """Hand one packet to the receiver, naming the reason when the receiver refuses it.

        Without a freshness lookup the relay does not interpret a refusal -- it records that
        the writer returned False, which is all it ever knew.
        """
        accepted = bool(self._writer.deliver_detection(message, agent_id))
        if accepted:
            return True, "accepted"
        if self._freshness_of is None:
            return False, "delivery_refused"
        known = self._freshness_of(agent_id, message.target_id)
        if known is None:
            return False, "delivery_unknown_task"
        update_time, revision = known
        if update_time > float(message.sent_at):
            return False, "delivery_superseded"
        if update_time == float(message.sent_at) and revision >= int(message.revision):
            return False, "delivery_duplicate"
        return False, "delivery_superseded"

    # ------------------------------------------------------------------ reverse channel
    def _ack_applied(self, agent_id: str, message: DetectionMessage) -> None:
        """A receiver applied one uplink packet: queue its cumulative acknowledgement.

        At most one ACK per (receiver, task, tick), merged with the existing queue rules -- a
        receiver that applies two packets for the same task in one step confirms the newer view
        once, not twice. The ACK itself is billed on the reverse ledger whether or not it
        survives the reverse channel.
        """
        if not self._ack_enabled:
            return
        for index, (_, _, subscriber, target, _ack, _cost) in enumerate(self._ack_pending):
            if subscriber == str(agent_id) and target == str(message.target_id):
                # Same receiver, same task, same step: the newer application supersedes the
                # older frame, exactly as two pieces of news share one letter on the uplink.
                self._ack_pending[index] = (
                    self._tick + self._ack_delay_ticks,
                    self._ack_ordinal,
                    str(agent_id),
                    str(message.target_id),
                    AckMessage(
                        subscriber=str(agent_id),
                        target_id=str(message.target_id),
                        sent_at=float(message.sent_at),
                        revision=int(message.revision),
                        content_fingerprint=content_fingerprint(message),
                        confirmed_at=float(self._now_value()),
                    ),
                    0,
                )
                return
        self._ack_ordinal += 1
        ack = AckMessage(
            subscriber=str(agent_id),
            target_id=str(message.target_id),
            sent_at=float(message.sent_at),
            revision=int(message.revision),
            content_fingerprint=content_fingerprint(message),
            confirmed_at=float(self._now_value()),
        )
        cost = estimate_ack_bytes(ack)
        self._ack_pending.append(
            (
                self._tick + self._ack_delay_ticks,
                self._ack_ordinal,
                str(agent_id),
                str(message.target_id),
                ack,
                int(cost),
            )
        )
        self.counters["ack_emitted"] += 1

    def _deliver_acks(self) -> None:
        """Deliver the ACK frames whose reverse-channel delay has come; bill every attempt.

        The draw is counter-based on ``(episode seed, direction, subscriber, target, ordinal)``,
        so two strategies facing the same key lose the same frame -- common random numbers, not
        merely a common distribution. A lost frame is billed in full and changes nothing else.
        """
        if not self._ack_enabled:
            return
        due = [item for item in self._ack_pending if item[0] <= self._tick]
        self._ack_pending = [item for item in self._ack_pending if item[0] > self._tick]
        if not due:
            return
        due.sort(key=lambda item: item[1])
        for _due_tick, ordinal, subscriber, target, ack, cost in due:
            key = reverse_rng_key(
                episode_seed=int(self._ack_seed),
                direction="uplink_ack",
                subscriber=subscriber,
                target_id=target,
                ordinal=ordinal,
            )
            # Counter-based seeding needs integers, so the name components are hashed into
            # the seed sequence rather than passed through: the same key still assembles the
            # same stream, and two strategies naming the same key still draw the same number.
            stream = np.random.default_rng(
                [
                    key[0],
                    _stable_hash(key[1]),
                    _stable_hash(key[2]),
                    _stable_hash(key[3]),
                    key[4],
                    int(self._tick),
                ]
            )
            delivered = bool(stream.random() < self._ack_delivery_probability)
            self.counters["ack_attempts"] += 1
            self.counters["ack_attempt_bytes"] += int(cost)
            self.ack.ledger.note_attempt(int(cost), delivered=delivered)
            if delivered:
                self.counters["ack_delivered"] += 1
                state = self.ack.state(subscriber, target)
                advanced = state.offer(ack, self._tick)
                self.ack.ledger.note_offer(advanced=advanced)
                if advanced:
                    self.counters["ack_frames_advanced"] += 1
                else:
                    self.counters["ack_frames_reaffirmed"] += 1
                self._settle_unacked(target)
            else:
                self.counters["ack_lost"] += 1
            if self.transcript is not None:
                self.transcript.add(
                    "ack",
                    step=int(self._tick),
                    subscriber=subscriber,
                    target=target,
                    revision=int(ack.revision),
                    sent_at=float(ack.sent_at),
                    confirmed_at=float(ack.confirmed_at),
                    attempt_bytes=int(cost),
                    delivered=bool(delivered),
                    ordinal=int(ordinal),
                )

    def _settle_unacked(self, target_id: str) -> None:
        """Clear the unacknowledged-broadcast count once every receiver has caught up.

        The count is the sender's own "I have said this and not everyone has confirmed it"
        counter, so it is zeroed by confirmations, not by further sends.
        """
        if self.ack.broadcast_attempts.get(target_id, 0) <= 0:
            return
        if all(
            self.ack.state(name, target_id).highest is not None for name in self._subscribers
        ):
            self.ack.broadcast_attempts[target_id] = 0

    def ack_features(
        self, item: SendItem, subscribers: Iterable[str] | None = None
    ) -> tuple[float, ...]:
        """The four ``/2`` features for one candidate, from arrived ACKs and own history only.

        The view gap compares what the sender *last put on the air* (its own record) with what
        each receiver has confirmed, so both sides of the subtraction are the sender's own
        history. Every term is ``UNKNOWN`` when ACK is off, so a ``/1``-trained controller
        reading this block sees a constant rather than a plausible-looking zero.
        """
        if not self._ack_enabled:
            return (-1.0, -1.0, -1.0, -1.0)
        names = [
            str(name)
            for name in (subscribers if subscribers is not None else self._subscribers)
        ]
        ledger = self._ledger.get(item.target_id)
        last_sent = ledger.last_sent if ledger is not None else None
        if last_sent is None:
            # Nothing has gone out yet, so there is no confirmed view to be behind: the gap
            # term is unknown rather than zero.
            return (self.ack.confirmed_fraction(item.target_id, names), -1.0, -1.0,
                    min(1.0, self.ack.unacked_broadcasts(item.target_id) / UNACKED_SCALE))
        gaps: list[float] = []
        for name in names:
            highest = self.ack.state(name, item.target_id).highest
            if highest is None:
                continue
            revision_gap = max(0, int(item.revision) - int(highest[1]))
            content_gap = 0.0 if highest[2] == content_fingerprint(last_sent) else 1.0
            gaps.append(float(revision_gap) + content_gap)
        gap = max(gaps) if gaps else -1.0
        confirmed = self.ack.confirmed_fraction(item.target_id, names)
        age = self.ack.max_ack_age(item.target_id, names, self._tick)
        return (
            float(confirmed),
            -1.0 if age < 0 else min(1.0, age / ACK_AGE_SCALE_STEPS),
            -1.0 if gap < 0 else min(1.0, gap / VIEW_GAP_SCALE),
            min(1.0, self.ack.unacked_broadcasts(item.target_id) / UNACKED_SCALE),
        )

    def _note_queue_depth(self) -> None:
        self.counters["max_in_flight"] = max(self.counters["max_in_flight"], len(self._pending))
        waiting = [
            ledger.ready_step
            for ledger in self._ledger.values()
            if ledger.ready_step is not None and ledger.ready_step <= self._tick
        ]
        self.counters["unsent_change_steps"] += len(waiting)
        if waiting:
            self.counters["max_backlog_age"] = max(
                self.counters["max_backlog_age"], int(self._tick - min(waiting))
            )

    # ------------------------------------------------------------------ scheduling
    def _emit_selected(self) -> None:
        self._budget.begin_step()
        observation = self._observation()
        if not observation.items:
            return
        chosen_targets = self._policy.decide(observation)
        by_target = {item.target_id: item for item in observation.items}
        if getattr(self._policy, "orders_selection", False):
            # The policy named its own priority: keep it, first mention first, and ignore any
            # target it was not actually offered.
            requested = dict.fromkeys(str(target) for target in chosen_targets)
            chosen = [by_target[target] for target in requested if target in by_target]
        else:
            # A rule that does not claim an ordering is trimmed by the transport, identically
            # for every such rule: it is judged by what it asks to say, not by a queueing
            # advantage the executor happened to give it.
            wanted = set(chosen_targets)
            chosen = transport_order(
                item for item in observation.items if item.target_id in wanted
            )
        forced: set[str] = set()
        if self._max_wait_steps is not None and self._timing_mode == LEGACY_SHARED_BOUND:
            overdue = {
                item.target_id
                for item in observation.items
                if item.age_steps >= self._wait_limit(item)
            }
            present = {item.target_id for item in chosen}
            forced = overdue - present
            chosen = list(chosen) + [
                item for item in observation.items if item.target_id in forced
            ]
        for item in chosen:
            reason = "forced_maxwait" if item.target_id in forced else "policy"
            self._emit(item, first=item.first_emission, reason=reason)

    def _wait_limit(self, item: SendItem) -> int:
        if self._urgent_wait_steps is not None and 0.0 <= item.remaining_time <= self._urgency_time:
            return self._urgent_wait_steps
        assert self._max_wait_steps is not None
        return self._max_wait_steps

    def _now_value(self) -> float:
        """The sending node's own reading of the clock; the step counter if none is wired."""
        return float(self._now_of()) if self._now_of is not None else float(self._tick)

    def _observation(self) -> SendObservation:
        now = self._now_value()
        items: list[SendItem] = []
        for target_id in sorted(self._ledger):
            ledger = self._ledger[target_id]
            if ledger.ready_step is None and ledger.last_sent is None:
                continue
            if ledger.closed and ledger.ready_step is None:
                continue
            snapshot = self._snapshot_of(target_id) if self._snapshot_of else None
            if snapshot is not None and snapshot.status.terminal:
                # The task closed before the news went out. Say nothing, and record that a
                # policy asked: a rule that keeps trying to announce a dead task is visible.
                ledger.closed = True
                if ledger.pending_event:
                    self.counters["emissions_suppressed_terminal"] += 1
                    ledger.pending_event = False
                    ledger.ready_step = None
                    ledger.ready_view = None
                    ledger.pending_message = None
                continue
            reference = ledger.last_sent or ledger.pending_message
            # A registry snapshot is what the sending node would say now; without one (no
            # lookup wired, or a test stub) the event it was told about is the state.
            state = snapshot if snapshot is not None else reference
            if state is None:
                # Neither the registry nor a previous packet describes this target: there is
                # nothing to say about it, and the ledger stays as it is.
                continue
            view = _View.of(state)
            changed_view = ledger.last_sent_view != view
            if changed_view and ledger.ready_step is None:
                # The clock on an unsaid change starts here, whether or not the attached
                # policy intends to send it: a held change is what a "queue wait" number has
                # to count, and the historical policy holds every one of them by design.
                ledger.ready_step = self._tick
                ledger.ready_view = view
                self.counters["generated_items"] += 1
            elif changed_view and ledger.ready_view is None:
                # Something is already waiting and this is its content for the first time: an
                # event booked it on the way in, so record what it says and count nothing again.
                ledger.ready_view = view
            elif changed_view and ledger.ready_view != view:
                # The registry moved again while the first change sat unsent. This is the same
                # situation the event path calls a coalesce -- two pieces of news, one letter --
                # and it has to be counted the same way, or the merge the relay really performed
                # disappears from the ledger.
                ledger.ready_view = view
                self.counters["generated_items"] += 1
                self.counters["coalesced_changes"] += 1
            sent = ledger.last_sent
            ready_step = ledger.ready_step if ledger.ready_step is not None else self._tick
            items.append(
                SendItem(
                    target_id=target_id,
                    ready_step=int(ready_step),
                    age_steps=max(0, self._tick - int(ready_step)),
                    pending_event=ledger.pending_event,
                    changed_view=changed_view,
                    first_emission=sent is None,
                    revision=int(state.revision),
                    revision_delta=int(state.revision) - int(sent.revision if sent else 0),
                    stage_index=int(state.current_stage),
                    stage_changed=bool(sent is not None and sent.current_stage != state.current_stage),
                    progress=float(state.progress),
                    progress_delta=float(state.progress) - float(sent.progress if sent else 0.0),
                    requirements_changed=bool(
                        sent is not None
                        and tuple(sent.remaining_requirements) != tuple(state.remaining_requirements)
                    ),
                    members_changed=bool(
                        sent is not None and tuple(sent.members) != tuple(state.members)
                    ),
                    deadline=float(state.deadline),
                    remaining_time=float(state.deadline) - now,
                    sent_count=ledger.sent_count,
                    last_sent_step=ledger.last_sent_step,
                    since_last_emission_time=(
                        None if ledger.last_sent_now is None else now - float(ledger.last_sent_now)
                    ),
                    has_future_requirements=bool(len(state.stages) > int(state.current_stage) + 1),
                )
            )
        return SendObservation(
            step=self._tick,
            current_time=now,
            items=tuple(items),
            remaining_budget_bytes=self._budget.remaining_bytes,
            budget_bytes_per_episode=self._budget.bytes_per_episode,
            capacity_remaining=(
                None
                if self._budget.sends_per_step is None
                else max(0, self._budget.sends_per_step - self._budget.sends_this_step)
            ),
            backlog_targets=self.backlog,
            subscribers=len(self._subscribers),
        )

    def _emit(self, item: SendItem, *, first: bool, reason: str = "policy") -> None:
        if reason not in ("policy", "forced_maxwait", "other"):
            raise ValueError(f"unknown send reason {reason!r}")
        ledger = self._ledger[item.target_id]
        snapshot = self._snapshot_of(item.target_id) if self._snapshot_of else None
        message = self._message(item.target_id, ledger, snapshot)
        payload_bytes = estimate_message_bytes(message)
        # One emission addresses the whole subscribed set, so its price is the payload times
        # that fan-out. A broadcast is paid for per receiver because the airtime is per
        # receiver; a per-emission cost would let an arm look cheap by widening the team.
        cost_bytes = payload_bytes * len(self._subscribers)
        if not self._budget.can_send(cost_bytes):
            blocked = self._budget.why_blocked(cost_bytes)
            self._budget.note_refusal(blocked, cost_bytes)
            key = (
                "capacity_refused_sends"
                if blocked == "capacity_step"
                else "budget_refused_sends"
            )
            self.counters[key] += 1
            self.counters[f"refused_{reason}"] += 1
            if self.transcript is not None:
                self.transcript.add(
                    "refused",
                    step=int(self._tick),
                    target=item.target_id,
                    reason=blocked,
                    send_reason=reason,
                    cost_bytes=int(cost_bytes),
                    remaining_bytes=(
                        None if self._budget.bytes_per_episode is None else self._budget.remaining_bytes
                    ),
                )
            # The change stays queued and keeps ageing: an unaffordable send is not a send,
            # and nothing here marks the ledger as informed.
            return
        due_tick = self._tick + self._delay_ticks
        sent = 0
        lost = 0
        for agent_id in self._subscribers:
            # The uplink drops packets independently per robot; a dropped packet is never
            # retransmitted by the relay itself, so that robot's belief stays as it was and
            # only a policy that emits again can recover it. The drop is billed all the same.
            if self._rng.random() > self._delivery_probability:
                lost += 1
                continue
            self._sequence += 1
            sent += 1
            self._pending.append(
                (due_tick, self._sequence, agent_id, message, self._tick, payload_bytes)
            )
        self._budget.charge(cost_bytes)
        ready_step = ledger.ready_step if ledger.ready_step is not None else self._tick
        ledger.last_sent = message
        ledger.last_sent_view = _View.of(message)
        ledger.last_sent_step = self._tick
        ledger.last_sent_now = self._now_value()
        ledger.sent_count += 1
        ledger.pending_event = False
        ledger.pending_message = None
        ledger.ready_step = None
        ledger.ready_view = None
        attempts = sent + lost
        self.counters["emissions"] += 1
        self.counters["redundant_emissions"] += 0 if item.changed_view else 1
        self.counters[f"emissions_{reason}"] += 1
        self.counters[f"attempts_{reason}"] += attempts
        self.counters[f"bytes_{reason}"] += int(cost_bytes)
        self.counters["first_emissions"] += 1 if first else 0
        self.counters["refresh_emissions"] += 0 if first else 1
        self.counters["transmission_attempts"] += attempts
        self.counters["packets_lost"] += lost
        self.counters["airtime_bytes"] += cost_bytes
        self.counters["backlog_steps"] += max(0, self._tick - int(ready_step))
        # The sender's own count of emissions for this target since its view last changed:
        # what the ``unacked_broadcasts`` feature reads, and what an ACK resets to zero.
        if self._ack_enabled and item.changed_view:
            self.ack.broadcast_attempts[item.target_id] = 1
        elif self._ack_enabled:
            self.ack.broadcast_attempts[item.target_id] = (
                self.ack.broadcast_attempts.get(item.target_id, 0) + 1
            )
        if self.transcript is not None:
            self.transcript.add(
                "send",
                step=int(self._tick),
                target=item.target_id,
                send_reason=reason,
                revision=int(message.revision),
                payload_bytes=int(payload_bytes),
                attempts=int(attempts),
                packets=int(sent),
                packets_lost=int(lost),
                subscribers=len(self._subscribers),
                cost_bytes=int(cost_bytes),
                spent_after_bytes=int(self._budget.spent_bytes),
                due_step=int(due_tick),
                ready_step=int(ready_step),
                now=float(self._now_value()),
                backlog_steps=max(0, self._tick - int(ready_step)),
                first=bool(first),
                progress=float(message.progress),
                stage_index=int(message.current_stage),
                remaining=[int(value) for value in message.remaining_requirements],
                members=[str(member) for member in message.members],
                bytes_total=int(cost_bytes),
                deadline=float(message.deadline),
            )

    def reset(self) -> None:
        """Start a new episode: drop backlog, outbox, ledgers, dedup state and rng draws."""
        self._pending.clear()
        self._ledger.clear()
        self._seen_event_ids.clear()
        self._tick = 0
        self._sequence = 0
        self._rng = np.random.default_rng(self._seed)
        # The reverse channel starts empty too: no queued frames, no confirmation state, no
        # billed bytes, and the counter-based draws begin again from ordinal zero.
        self._ack_pending.clear()
        self._ack_ordinal = 0
        self.ack.reset()
        self._budget.reset()
        for key in self.counters:
            self.counters[key] = 0

    @property
    def queued(self) -> int:
        return len(self._pending)

    @property
    def backlog(self) -> int:
        """Targets with a change the policy has not emitted yet."""
        return sum(1 for ledger in self._ledger.values() if ledger.ready_step is not None)

    @property
    def policy(self) -> CommSendPolicy:
        return self._policy
    @property
    def budget(self) -> CommBudget:
        """The executor this relay spends from. A harness asserts on it, never writes it."""
        return self._budget

    def attach_policy(self, policy: CommSendPolicy) -> None:
        """Install the decision-maker before the episode starts.

        Refusing after the first step is the point: swapping a controller mid-episode would
        make the recorded transcript a mixture of two policies and the run label a lie.
        """
        if self._tick or self._pending or self._ledger:
            raise RuntimeError(
                f"the relay has already run {self._tick} steps; a policy can only be attached "
                "before the episode starts"
            )
        self._policy = policy
        self._attach_ack_source(policy)

    def _attach_ack_source(self, policy: Any) -> None:
        """Give a learned sender the relay's own reverse-channel reader, if ACK is on.

        The reader is a closure over this relay, so the four extra dims can only ever come
        from ACKs that arrived here. With ACK off nothing is attached and the policy keeps the
        19-dim layout it was trained on.
        """
        source = getattr(policy, "_ack_source", None)
        if self._ack_enabled and hasattr(policy, "last_features"):
            policy._ack_source = self.ack_features
        elif source is not None and not self._ack_enabled:
            policy._ack_source = None

    def transport_parameters(self) -> dict[str, Any]:
        """The knobs the relay itself owns, for a configuration truth table."""
        return {
            "subscribers": list(self._subscribers),
            "delivery_probability": self._delivery_probability,
            "delay_ticks": self._delay_ticks,
            "budget": self._budget.usage(),
            "policy": self._policy.name,
            "policy_parameters": dict(self._policy.parameters()),
            "max_wait_steps": self._max_wait_steps,
            "urgent_wait_steps": self._urgent_wait_steps,
            "urgency_time": self._urgency_time if self._max_wait_steps is not None else None,
            # Which of the two protocols this relay runs, and whether the declared bound is
            # something the transport acts on. A truth table has to be able to say that the
            # bound is declared *and not enforced* before it reads a zero forced-send count as
            # evidence of anything.
            "timing_mode": self._timing_mode,
            "bound_enforced_by_transport": self._max_wait_steps is not None
            and self._timing_mode == LEGACY_SHARED_BOUND,
            # The reverse channel's declared parameters. ``ack_enabled`` False means none of
            # the rest was ever constructed, so a zero ACK byte count on a no-ACK run is a
            # statement about the protocol, not about the channel.
            "ack_enabled": self._ack_enabled,
            "ack_schema": ACK_SCHEMA if self._ack_enabled else None,
            "ack_delay_ticks": self._ack_delay_ticks if self._ack_enabled else None,
            "ack_delivery_probability": (
                self._ack_delivery_probability if self._ack_enabled else None
            ),
            "ack_seed": self._ack_seed if self._ack_enabled else None,
            "ack_rng_scheme": (
                "counter-based:(episode_seed,direction,subscriber,target,ordinal)"
                if self._ack_enabled
                else None
            ),
            "ack_billing": "attempt_bytes_on_separate_reverse_ledger"
            if self._ack_enabled
            else None,
            "ack_usage": self.ack.ledger.usage() if self._ack_enabled else None,
        }

    # ------------------------------------------------------------------ payloads
    def _message(
        self, target_id: str, ledger: _TargetLedger, snapshot: TaskSnapshot | None
    ) -> DetectionMessage:
        """The packet for one emission: the newest registry view, or the recorded event.

        A first emission keeps the detection's own event id, so a discovery message stays
        traceable to the search event behind it. A refresh has no new event to name, so it is
        labelled by what it repeats.
        """
        if snapshot is None:
            pending = ledger.pending_message
            if pending is None:
                raise RuntimeError(
                    f"relay has no registry lookup and no stored event for {target_id}"
                )
            return pending
        event_id = (
            ledger.pending_message.event_id
            if ledger.pending_message is not None
            else f"refresh-{target_id}-{self._tick}"
        )
        return self._message_from_snapshot(target_id, event_id, snapshot)

    def _message_from_event(self, event: DetectionEvent) -> DetectionMessage:
        stages = event.profile.stages
        return DetectionMessage(
            event_id=event.event_id,
            target_id=event.target_id,
            sent_at=event.detected_at,
            position=event.world_position,
            stages=stages,
            current_stage=0,
            remaining_requirements=stages[0].requirements if stages else (),
            deadline=event.detected_at + event.profile.deadline_offset,
            severity=event.profile.severity,
            confidence=event.confidence,
            progress=0.0,
            discovered_by=event.discovered_by,
        )

    def _message_from_snapshot(
        self, target_id: str, event_id: str, snapshot: TaskSnapshot
    ) -> DetectionMessage:
        return DetectionMessage(
            event_id=event_id,
            target_id=target_id,
            sent_at=snapshot.updated_at,
            position=snapshot.position,
            stages=snapshot.stages,
            current_stage=snapshot.current_stage,
            remaining_requirements=snapshot.remaining_requirements,
            deadline=snapshot.deadline,
            severity=snapshot.severity,
            confidence=snapshot.confidence,
            progress=snapshot.progress,
            members=snapshot.members,
            discovered_by=snapshot.metadata.get("discovered_by"),
            revision=snapshot.revision,
        )
