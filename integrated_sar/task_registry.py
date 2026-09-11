"""Canonical task state derived from search detections and execution feedback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import (
    DetectionEvent,
    ProgressEvent,
    TaskSnapshot,
    TaskStageSpec,
    TaskStatus,
    WorldPosition,
)


@dataclass(slots=True)
class _TaskRecord:
    task_id: str
    source_event_id: str
    position: WorldPosition
    stages: tuple[TaskStageSpec, ...]
    current_stage: int
    remaining_requirements: tuple[int, ...]
    progress: float
    status: TaskStatus
    deadline: float
    confidence: float
    severity: float
    discovered_at: float
    updated_at: float
    revision: int = 1
    members: tuple[str, ...] = ()
    expired: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> TaskSnapshot:
        return TaskSnapshot(
            task_id=self.task_id,
            source_event_id=self.source_event_id,
            position=self.position,
            stages=self.stages,
            current_stage=self.current_stage,
            remaining_requirements=self.remaining_requirements,
            progress=self.progress,
            status=self.status,
            deadline=self.deadline,
            confidence=self.confidence,
            severity=self.severity,
            discovered_at=self.discovered_at,
            updated_at=self.updated_at,
            revision=self.revision,
            members=self.members,
            expired=self.expired,
            metadata=dict(self.metadata),
        )


class TaskRegistry:
    """Single source of truth for task lifecycle and event deduplication."""

    def __init__(self) -> None:
        self._tasks: dict[str, _TaskRecord] = {}
        self._accepted_event_ids: set[str] = set()

    def handle_detection(self, event: DetectionEvent) -> None:
        if event.event_id in self._accepted_event_ids:
            return
        self._accepted_event_ids.add(event.event_id)

        record = self._tasks.get(event.target_id)
        if record is None:
            first_stage = event.profile.stages[0]
            metadata = dict(event.profile.metadata)
            metadata.update(event.metadata)
            metadata.update(
                {
                    "discovered_by": event.discovered_by,
                    "map_version": event.map_version,
                    "grid_position": (event.grid_position.row, event.grid_position.col),
                }
            )
            self._tasks[event.target_id] = _TaskRecord(
                task_id=event.target_id,
                source_event_id=event.event_id,
                position=event.world_position,
                stages=event.profile.stages,
                current_stage=0,
                remaining_requirements=first_stage.requirements,
                progress=0.0,
                status=TaskStatus.DETECTED,
                deadline=event.detected_at + event.profile.deadline_offset,
                confidence=event.confidence,
                severity=event.profile.severity,
                discovered_at=event.detected_at,
                updated_at=event.detected_at,
                metadata=metadata,
            )
            return

        if record.status.terminal or event.detected_at < record.updated_at:
            return
        record.position = event.world_position
        record.confidence = max(record.confidence, event.confidence)
        record.updated_at = event.detected_at
        record.revision += 1
        record.metadata["map_version"] = event.map_version
        record.metadata["last_observed_by"] = event.discovered_by

    def handle_progress(self, event: ProgressEvent) -> None:
        if event.event_id in self._accepted_event_ids:
            return
        self._accepted_event_ids.add(event.event_id)

        record = self._tasks.get(event.task_id)
        if record is None or event.observed_at < record.updated_at:
            return

        record.current_stage = min(event.stage_index, len(record.stages) - 1)
        record.remaining_requirements = tuple(event.remaining_requirements)
        record.progress = max(record.progress, event.progress)
        record.status = event.status
        record.members = tuple(event.members)
        record.expired = bool(event.expired)
        record.updated_at = event.observed_at
        record.revision += 1
        if event.failure_reason:
            record.metadata["failure_reason"] = event.failure_reason
        record.metadata.update(event.metadata)

        if event.status == TaskStatus.COMPLETED:
            record.progress = 1.0
            record.remaining_requirements = tuple(0 for _ in record.remaining_requirements)

    def get(self, task_id: str) -> TaskSnapshot | None:
        record = self._tasks.get(task_id)
        return record.snapshot() if record else None

    def snapshots(self, include_terminal: bool = True) -> tuple[TaskSnapshot, ...]:
        records = self._tasks.values()
        if not include_terminal:
            records = (record for record in records if not record.status.terminal)
        return tuple(record.snapshot() for record in sorted(records, key=lambda item: item.task_id))

    def active_snapshots(self) -> tuple[TaskSnapshot, ...]:
        return self.snapshots(include_terminal=False)

    def clear(self) -> None:
        self._tasks.clear()
        self._accepted_event_ids.clear()

    def __len__(self) -> int:
        return len(self._tasks)

    @property
    def accepted_event_count(self) -> int:
        return len(self._accepted_event_ids)
