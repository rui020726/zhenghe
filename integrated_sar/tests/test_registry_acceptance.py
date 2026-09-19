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

import numpy as np

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
from integrated_sar.tests.test_joint_driver import (
    _NullAgentSource,
    _NullAllocator,
    _NullTaskSink,
    _RecordingExecutor,
)


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


class RelayAcceptanceTests(unittest.TestCase):
    """A rejected detection must not be forwarded to any robot's belief.

    Third review round, finding 4: CommRelay only deduplicated event ids
    locally, so a registry-rejected late event was still queued and delivered
    -- effectively re-sending an old task view, and letting a terminal task be
    rewritten as pending in a robot's belief.
    """

    @staticmethod
    def _wire():
        from integrated_sar.adapters.task03 import Task03Adapter
        from integrated_sar.comm import CommRelay
        from integrated_sar.task03_runtime import build_task_env, load_task_env_module

        env = build_task_env(
            load_task_env_module(),
            seed=7,
            species=1,
            per_species=1,
            comm_range=0.3,
            depot_positions=[(0.0, 0.0)],
        )
        adapter = Task03Adapter(env=env, clear_initial_tasks=True, seed=7)
        registry = TaskRegistry()
        relay = CommRelay(
            subscribers=("0",),
            writer=adapter,
            snapshot_of=registry.get,
            accepted_of=registry.was_accepted,
            seed=7,
        )
        adapter.comm_relay = relay
        return env, adapter, registry, relay

    def test_a_late_event_is_not_queued_or_delivered(self) -> None:
        env, adapter, registry, relay = self._wire()
        first = make_detection(detected_at=1.0)
        registry.handle_detection(first)
        adapter.sync_tasks(registry.active_snapshots())
        relay.handle_detection(first)
        relay.tick()
        internal_id = adapter.external_to_internal["target-a"]
        belief_before = env.agent_knowledge[0]["task_locations"][internal_id].copy()

        late = make_detection(event_id="detection:target-a:late", detected_at=0.5)
        registry.handle_detection(late)
        self.assertEqual(registry.rejection_reason("detection:target-a:late"), "late_event")

        relay.handle_detection(late)
        self.assertEqual(relay.queued, 0, "a rejected detection entered the uplink")
        relay.tick()
        np.testing.assert_array_equal(
            env.agent_knowledge[0]["task_locations"][internal_id], belief_before
        )

    def test_a_terminal_task_is_not_reopened_in_belief(self) -> None:
        env, adapter, registry, relay = self._wire()
        first = make_detection()
        registry.handle_detection(first)
        adapter.sync_tasks(registry.active_snapshots())
        relay.handle_detection(first)
        relay.tick()
        registry.handle_progress(
            ProgressEvent(
                event_id="progress:done",
                task_id="target-a",
                observed_at=2.0,
                status=TaskStatus.COMPLETED,
                stage_index=0,
                remaining_requirements=(0, 0, 0, 0, 0),
                progress=1.0,
            )
        )

        again = make_detection(event_id="detection:target-a:again", detected_at=3.0)
        registry.handle_detection(again)
        self.assertEqual(registry.rejection_reason("detection:target-a:again"), "terminal_task")

        internal_id = adapter.external_to_internal["target-a"]
        knowledge = env.agent_knowledge[0]
        status_before = knowledge["task_current_status"][internal_id].copy()
        finished_before = set(knowledge["known_finished"])

        relay.handle_detection(again)
        relay.tick()
        self.assertEqual(relay.queued, 0, "a terminal-task re-detection entered the uplink")
        # The robot's belief is exactly as it was: not reopened as pending, and not
        # told the task finished either (that news travels via sensing/gossip).
        np.testing.assert_array_equal(
            knowledge["task_current_status"][internal_id], status_before,
            "a rejected re-detection rewrote the robot's belief",
        )
        self.assertEqual(set(knowledge["known_finished"]), finished_before)

    def test_an_accepted_event_is_still_forwarded(self) -> None:
        env, adapter, registry, relay = self._wire()
        event = make_detection()
        registry.handle_detection(event)
        adapter.sync_tasks(registry.active_snapshots())
        relay.handle_detection(event)
        # Recording news and putting it on the air are separate now: the relay holds the
        # change until the step's tick, where the send policy decides whether it goes out.
        self.assertEqual(relay.backlog, 1)
        self.assertEqual(relay.queued, 0)
        relay.tick()
        self.assertEqual(relay.counters["emissions"], 1)
        self.assertIn(
            adapter.external_to_internal["target-a"],
            env.agent_knowledge[0]["known_tasks"],
        )


class SchedulerVerdictTests(unittest.TestCase):
    """Coordination must not be triggered by an event the registry refused."""

    def test_a_rejected_detection_does_not_change_the_scheduling_signal(self) -> None:
        from integrated_sar.contracts import SearchStepResult
        from integrated_sar.orchestrator import JointSAROrchestrator

        registry = TaskRegistry()
        bus = InMemoryEventBus()
        bus.subscribe(DetectionEvent, registry.handle_detection)
        # A task the registry already knows, updated at t=10.
        registry.handle_detection(make_detection(event_id="detection:fresh", detected_at=10.0))

        class _StaleSearch:
            """Publishes a late re-detection, which the registry must refuse."""

            def step(self, current_time: float, dt: float) -> SearchStepResult:
                del current_time, dt
                return SearchStepResult(
                    (make_detection(event_id="detection:late", detected_at=1.0),)
                )

        class _CountingCoordinator:
            def __init__(self) -> None:
                self.calls = 0

            def advise(self, state: Any) -> Any:
                self.calls += 1
                from integrated_sar.contracts import CoordinationAdvice

                # Long validity, so the periodic/expiry triggers cannot mask the
                # event trigger this test is about.
                return CoordinationAdvice(
                    self.calls, state.current_time, state.current_time + 100.0
                )

        coordinator = _CountingCoordinator()
        orchestrator = JointSAROrchestrator(
            search=_StaleSearch(),
            task_sink=_NullTaskSink(),
            agent_source=_NullAgentSource(),
            coordinator=coordinator,
            allocator=_NullAllocator(),
            executor=_RecordingExecutor(),
            registry=registry,
            event_bus=bus,
            coordination_interval=100,
            allocation_interval=100,
        )
        # Tick 1 always coordinates: step_index 0 satisfies the periodic rule.
        # From tick 2 on, only a real state change may trigger another call.
        for _ in range(3):
            orchestrator.step()
        self.assertEqual(registry.rejection_reason("detection:late"), "late_event")
        # Only the pre-seeded detection was accepted; the stale re-detections were not.
        self.assertEqual(registry.accepted_event_count, 1)
        self.assertEqual(
            coordinator.calls, 1,
            "a rejected detection triggered coordination on a non-periodic tick",
        )


class ReDetectionLogTests(RegistryClassificationHarness):
    """A legal re-detection must not be logged as a fresh 'detected' task.

    Third review round, finding 6: the detection record hardcoded stage 0,
    progress 0 and status 'detected', and stamped its own event id as
    source_event_id, so a re-detection of an already-assigned task wrote a
    lifecycle state the registry never held and broke the per-target
    correlation key.
    """

    def test_re_detection_logs_the_registry_state_and_keeps_the_first_source(self) -> None:
        self.publish(make_detection(detected_at=1.0))
        self.publish(
            make_progress(
                event_id="progress:assigned", observed_at=2.0, status=TaskStatus.ASSIGNED
            )
        )
        self.publish(
            make_detection(event_id="detection:target-a:v2", detected_at=3.0)
        )

        snapshot = self.registry.get("target-a")
        self.assertEqual(snapshot.status, TaskStatus.ASSIGNED)
        self.assertEqual(snapshot.revision, 3)

        records = self.records()
        detections = [r for r in records if r["event_type"] == "detection"]
        self.assertEqual(len(detections), 2)
        first, second = detections
        self.assertEqual(first["source_event_id"], "detection:target-a")
        self.assertEqual(
            second["source_event_id"],
            "detection:target-a",
            "a re-detection replaced the target's correlation key",
        )
        self.assertEqual(second["status"], "assigned")
        self.assertEqual(second["progress"], 0.0)
        self.assertEqual(second["task_revision"], 3)
        self.assertEqual(
            second["detail"]["detection_event_id"],
            "detection:target-a:v2",
            "the new observation's own id belongs in detail",
        )
        # Every record for this target still joins on one detection id.
        sources = {r["source_event_id"] for r in records}
        self.assertEqual(sources, {"detection:target-a"})

    def test_a_first_detection_still_reads_as_detected(self) -> None:
        self.publish(make_detection(detected_at=1.0))
        record = self.records()[0]
        self.assertEqual(record["status"], "detected")
        self.assertEqual(record["stage_index"], 0)
        self.assertEqual(record["progress"], 0.0)
        self.assertEqual(record["remaining_requirements"], [1, 0, 0, 0, 0])
        self.assertEqual(record["detail"]["detection_event_id"], "detection:target-a")


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
