"""The communication bridge between search detections and rescue robots' beliefs.

Architecture invariant 9: actors and the online coordinator may only use beliefs
that actually arrived over a communication link. This module is that link. The
event bus fans a published :class:`~integrated_sar.contracts.DetectionEvent` out
once, the relay snapshots what the registry accepted and queues one transmission
attempt per subscribed rescue robot, and delivery obeys an explicit topology, a
configurable per-packet delay and a seeded packet-loss process.

The uplink is a radio channel, not omniscience:

* only subscribed robots receive it, and every packet is delivered or lost
  independently;
* a lost packet is gone -- the robot's belief stays exactly as it was;
* the relay never reads anyone's position, so the search and rescue robots remain
  two independent entity sets (invariant 10) and co-location with the search start
  is never a precondition for the loop to start;
* the stochastic process (which packets survive) is driven entirely by the seed.

Once a robot knows a task, further updates still travel through TaskEnv's own
sensing and one-hop gossip; this relay only carries what the search side found.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .contracts import DetectionEvent, DetectionMessage, TaskSnapshot

SnapshotLookup = Callable[[str], TaskSnapshot | None]
AcceptanceLookup = Callable[[str], bool]


class CommRelay:
    """Queues published detections and delivers them to subscribed robots.

    Delivery happens in :meth:`tick`, which the executor calls once per joint step
    before the simulated world advances. A message queued during tick ``N`` is due
    at ``N + 1 + delay_ticks``, so ``delay_ticks=0`` arrives within the same joint
    step and ``delay_ticks=1`` on the next one.

    ``accepted_of`` is the registry's verdict lookup: an event the registry
    rejected (unknown task, late timestamp, terminal task) is not news, so the
    relay must not send it. Subscribing the relay after the registry guarantees
    the verdict is already recorded when the relay is called.
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
    ) -> None:
        if not 0.0 <= float(delivery_probability) <= 1.0:
            raise ValueError(
                f"delivery_probability must stay in [0, 1], got {delivery_probability}"
            )
        if int(delay_ticks) < 0:
            raise ValueError(f"delay_ticks cannot be negative, got {delay_ticks}")
        self._subscribers = tuple(str(agent_id) for agent_id in subscribers)
        self._writer = writer
        self._snapshot_of = snapshot_of
        self._accepted_of = accepted_of
        self._delivery_probability = float(delivery_probability)
        self._delay_ticks = int(delay_ticks)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._pending: list[tuple[int, int, str, DetectionMessage]] = []
        self._seen_event_ids: set[str] = set()
        self._tick = 0
        self._sequence = 0

    def handle_detection(self, event: DetectionEvent) -> None:
        """Bus handler: queue one transmission attempt per subscriber."""
        if event.event_id in self._seen_event_ids:
            return
        self._seen_event_ids.add(event.event_id)
        if self._accepted_of is not None and not self._accepted_of(event.event_id):
            # The registry refused this event: it is not a new lifecycle fact, so
            # forwarding it would let a rejected detection re-send an old task
            # view or reopen a terminal one in robot belief.
            return
        message = self._build_message(event)
        due_tick = self._tick + 1 + self._delay_ticks
        for agent_id in self._subscribers:
            # The uplink drops packets independently per robot; a dropped packet is
            # never retransmitted, so that robot's belief stays as it was.
            if self._rng.random() > self._delivery_probability:
                continue
            self._sequence += 1
            self._pending.append((due_tick, self._sequence, agent_id, message))

    def tick(self) -> tuple[str, ...]:
        """One delivery opportunity; returns the agents whose belief changed."""
        self._tick += 1
        due = [item for item in self._pending if item[0] <= self._tick]
        if not due:
            return ()
        self._pending = [item for item in self._pending if item[0] > self._tick]
        due.sort(key=lambda item: item[1])
        changed: list[str] = []
        for _, _, agent_id, message in due:
            if self._writer.deliver_detection(message, agent_id):
                changed.append(agent_id)
        return tuple(changed)

    @property
    def queued(self) -> int:
        return len(self._pending)

    def reset(self) -> None:
        """Start a new episode: drop outbox, dedup state and the seeded rng draws."""
        self._pending.clear()
        self._seen_event_ids.clear()
        self._tick = 0
        self._sequence = 0
        self._rng = np.random.default_rng(self._seed)

    def _build_message(self, event: DetectionEvent) -> DetectionMessage:
        snapshot = self._snapshot_of(event.target_id) if self._snapshot_of else None
        if snapshot is None:
            # The registry has not been consulted (or does not know the target), so
            # the event itself -- the same profile the registry would have stored --
            # defines the payload.
            stages = event.profile.stages
            return DetectionMessage(
                event_id=event.event_id,
                target_id=event.target_id,
                sent_at=event.detected_at,
                position=event.world_position,
                stages=stages,
                current_stage=0,
                remaining_requirements=(
                    stages[0].requirements if stages else ()
                ),
                deadline=event.detected_at + event.profile.deadline_offset,
                severity=event.profile.severity,
                confidence=event.confidence,
                progress=0.0,
                discovered_by=event.discovered_by,
            )
        return DetectionMessage(
            event_id=event.event_id,
            target_id=event.target_id,
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
        )
