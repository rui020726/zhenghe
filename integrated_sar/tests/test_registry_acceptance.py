"""The registry must classify events, and the log must follow its verdicts.

Regression for review finding 5: the registry used to mark an event id
"accepted" before checking whether the event was legal, so late, unknown and
terminal-state events silently inflated ``accepted_event_count``, and the
lifecycle logger wrote records for transitions the registry never applied
(e.g. a late ``completed`` record next to a registry status of ``assigned``).
"""

from __future__ import annotations

import io
import json
import unittest
from typing import Any

from integrated_sar.contracts import (
    DetectionEvent,
    GridPosition,
    ProgressEvent,
    TargetProfile,
    TaskStageSpec,
    TaskStatus,
    WorldPosition,
)
from integrated_sar.event_bus import InMemoryEventBus
from integrated_sar.lifecycle_log import LifecycleLogger
from integrated_sar.task_registry import TaskRegistry


def make_detection(
    target_id: str = "target-a",
    *,
    event_id: str = "detection:target-a",
    detected_at: float = 1.0,
) -> DetectionEvent:
    return DetectionEvent(
        event_id=event_id,
        target_id=target_id,
        grid_position=GridPosition(20, 30),
        world_position=WorldPosition(30 / 59, 20 / 59),
        detected_at=detected_at,
        discovered_by="agent_0",
        confidence=1.0,
        profile=TargetProfile(
            stages=(TaskStageSpec((1, 0, 0, 0, 0), 3.0),),
            deadline_offset=500.0,
        ),
    )


def make_progress(
    task_id: str = "target-a",
    *,
    event_id: str,
    observed_at: float = 2.0,
    status: TaskStatus = TaskStatus.ASSIGNED,
) -> ProgressEvent:
    return ProgressEvent(
        event_id=event_id,
        task_id=task_id,
        observed_at=observed_at,
        status=status,
        stage_index=0,
        remaining_requirements=(1, 0, 0, 0, 0),
        progress=0.0,
        members=("0",),
    )


class RegistryClassificationHarness(unittest.TestCase):
    """Bus + registry + logger wired exactly like build_joint_run does."""

    def setUp(self) -> None:
        self.registry = TaskRegistry()
        self.stream = io.StringIO()
        self.logger = LifecycleLogger(
            self.stream,
            snapshot_of=self.registry.get,
            accepted_of=self.registry.was_accepted,
        )
        self.bus = InMemoryEventBus()
        # Registry first: the logger and any other consumer must see the verdict.
        self.bus.subscribe(DetectionEvent, self.registry.handle_detection)
        self.bus.subscribe(ProgressEvent, self.registry.handle_progress)
        self.bus.subscribe(DetectionEvent, self.logger.handle_detection)
        self.bus.subscribe(ProgressEvent, self.logger.handle_progress)

    def records(self) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in self.stream.getvalue().splitlines()
            if line.strip()
        ]

    def publish(self, event: Any) -> None:
        self.bus.publish(event)


class LateEventTests(RegistryClassificationHarness):
    """The review repro: an out-of-order completion must not become history."""

    def test_late_progress_event_is_rejected_everywhere(self) -> None:
        self.publish(make_detection())
        self.publish(make_progress(event_id="progress:assigned", observed_at=2.0))
        self.publish(
            make_progress(
                event_id="progress:late-completed",
                observed_at=1.5,
                status=TaskStatus.COMPLETED,
            )
        )

        snapshot = self.registry.get("target-a")
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.status, TaskStatus.ASSIGNED)
        self.assertEqual(snapshot.progress, 0.0)
        self.assertEqual(self.registry.accepted_event_count, 2)
        self.assertEqual(self.registry.rejected_event_count, 1)
        self.assertEqual(
            self.registry.rejection_reason("progress:late-completed"), "late_event"
        )
        records = self.records()
        self.assertEqual(
            [(record["event_type"], record["status"]) for record in records],
            [("detection", "detected"), ("progress", "assigned")],
        )
        self.assertEqual(self.logger.records_written, 2)

    def test_a_new_detection_older_than_the_record_is_rejected(self) -> None:
        self.publish(make_detection())
        self.publish(make_progress(event_id="progress:assigned", observed_at=3.0))
        self.publish(
            make_detection(
                event_id="detection:target-a:late", detected_at=2.0
            )
        )

        snapshot = self.registry.get("target-a")
        self.assertEqual(snapshot.revision, 2)
        self.assertEqual(self.registry.accepted_event_count, 2)
        self.assertEqual(self.registry.rejection_reason("detection:target-a:late"), "late_event")
        self.assertEqual(len(self.records()), 2)


class UnknownTaskTests(RegistryClassificationHarness):
    def test_progress_for_an_unknown_task_is_rejected(self) -> None:
        self.publish(make_progress(task_id="ghost", event_id="progress:ghost"))

        self.assertEqual(len(self.registry), 0)
        self.assertEqual(self.registry.accepted_event_count, 0)
        self.assertEqual(self.registry.rejection_reason("progress:ghost"), "unknown_task")
        self.assertEqual(self.records(), [])


class TerminalTaskTests(RegistryClassificationHarness):
    def test_completed_task_rejects_late_detections_and_progress(self) -> None:
        self.publish(make_detection())
        self.publish(
            make_progress(
                event_id="progress:done",
                observed_at=2.0,
                status=TaskStatus.COMPLETED,
            )
        )
        self.publish(
            make_detection(event_id="detection:target-a:again", detected_at=3.0)
        )
        self.publish(
            make_progress(
                event_id="progress:after-done", observed_at=4.0
            )
        )

        snapshot = self.registry.get("target-a")
        self.assertEqual(snapshot.status, TaskStatus.COMPLETED)
        self.assertEqual(snapshot.revision, 2)
        self.assertEqual(self.registry.accepted_event_count, 2)
        self.assertEqual(self.registry.rejected_event_count, 2)
        self.assertEqual(
            self.registry.rejection_reason("detection:target-a:again"), "terminal_task"
        )
        self.assertEqual(
            self.registry.rejection_reason("progress:after-done"), "terminal_task"
        )
        self.assertEqual(
            [record["event_type"] for record in self.records()],
            ["detection", "progress"],
        )


class ConsistencyTests(RegistryClassificationHarness):
    def test_log_accepted_count_and_registry_status_agree(self) -> None:
        self.publish(make_detection())
        self.publish(make_detection())  # exact re-publication: seen, not re-counted
        self.publish(make_progress(event_id="progress:1", observed_at=2.0))
        self.publish(
            make_progress(
                event_id="progress:late", observed_at=1.0
            )
        )
        self.publish(make_progress(task_id="ghost", event_id="progress:ghost"))

        snapshot = self.registry.get("target-a")
        records = self.records()
        self.assertEqual(self.registry.accepted_event_count, 2)
        self.assertEqual(self.registry.rejected_event_count, 2)
        self.assertEqual(self.logger.records_written, len(records))
        self.assertEqual(len(records), 2)
        self.assertEqual(snapshot.status, TaskStatus.ASSIGNED)
        self.assertTrue(
            all(record["event_type"] in ("detection", "progress") for record in records)
        )


if __name__ == "__main__":
    unittest.main()
