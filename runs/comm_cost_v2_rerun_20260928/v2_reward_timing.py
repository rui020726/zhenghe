"""Versioned capture of accepted on-time completion events for Stage C task reward."""

from __future__ import annotations

from typing import Any

from integrated_sar.contracts import ProgressEvent, TaskStatus


REWARD_TIMING_VERSION = "published-completion-event-at-relay-tick/2"
TRAINING_VERSION = "constrained-cost-v2/published-completion-event-at-relay-tick/2"


class PublishedProgressRewardCollector:
    """Observe progress after registry commit and credit new on-time completions at that tick.

    The observer only reads the event, registry, relay tick, and clock. It does not publish,
    mutate registry state, advance time, or draw randomness. The task-id set makes the reward
    transition idempotent even if the same accepted event is dispatched again.
    """

    def __init__(self, *, registry: Any, relay: Any, clock: Any,
                 task_stream: dict[int, float]) -> None:
        self.registry = registry
        self.relay = relay
        self.clock = clock
        self.task_stream = task_stream
        self.credited_task_ids: set[str] = set()
        self.records: list[dict[str, Any]] = []

    def handle_progress(self, event: ProgressEvent) -> None:
        if event.status is not TaskStatus.COMPLETED:
            return

        accepted = bool(self.registry.was_accepted(event.event_id))
        snapshot = self.registry.get(event.task_id)
        deadline = float(snapshot.deadline) if snapshot is not None else None
        on_time = bool(
            accepted
            and snapshot is not None
            and snapshot.status is TaskStatus.COMPLETED
            and not event.expired
            and not snapshot.expired
            and float(event.observed_at) <= deadline
        )
        already_credited = event.task_id in self.credited_task_ids
        credited = on_time and not already_credited
        relay_tick = int(self.relay._tick)

        self.records.append({
            "event_id": str(event.event_id),
            "task_id": str(event.task_id),
            "event_time": float(event.observed_at),
            "relay_tick": relay_tick,
            "clock_now": float(self.clock.now),
            "accepted": accepted,
            "deadline": deadline,
            "event_expired": bool(event.expired),
            "snapshot_expired": bool(snapshot.expired) if snapshot is not None else None,
            "on_time": on_time,
            "credited": credited,
        })

        if credited:
            self.credited_task_ids.add(event.task_id)
            self.task_stream[relay_tick] = self.task_stream.get(relay_tick, 0.0) + 1.0
