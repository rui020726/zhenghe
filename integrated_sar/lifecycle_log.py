"""JSONL lifecycle log for every target that moves through the joint pipeline.

One line is written per lifecycle event, so a run can be replayed offline to answer
"who found this target, who was sent, how far did they get and why did it stop".
The field names are fixed by the integration contract; anything run-specific goes
into ``detail``.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, TextIO

from .contracts import AssignmentCommand, DetectionEvent, ProgressEvent, TaskSnapshot

LIFECYCLE_FIELDS: tuple[str, ...] = (
    "target_id",
    "source_event_id",
    "event_type",
    "timestamp",
    "discovered_by",
    "assigned_agents",
    "task_revision",
    "stage_index",
    "progress",
    "remaining_requirements",
    "status",
    "failure_reason",
)

RUN_TARGET_ID = "__run__"


@dataclass(frozen=True, slots=True)
class LifecycleRecord:
    """One JSONL line describing what happened to one target at one instant."""

    target_id: str
    event_type: str
    timestamp: float
    source_event_id: str | None = None
    discovered_by: str | None = None
    assigned_agents: tuple[str, ...] = ()
    task_revision: int | None = None
    stage_index: int | None = None
    progress: float | None = None
    remaining_requirements: tuple[int, ...] | None = None
    status: str | None = None
    failure_reason: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "target_id": self.target_id,
            "source_event_id": self.source_event_id,
            "event_type": self.event_type,
            "timestamp": round(float(self.timestamp), 6),
            "discovered_by": self.discovered_by,
            "assigned_agents": list(self.assigned_agents),
            "task_revision": self.task_revision,
            "stage_index": self.stage_index,
            "progress": None if self.progress is None else round(float(self.progress), 6),
            "remaining_requirements": (
                None if self.remaining_requirements is None else list(self.remaining_requirements)
            ),
            "status": self.status,
            "failure_reason": self.failure_reason,
        }
        if self.detail:
            payload["detail"] = _jsonable(self.detail)
        return payload


SnapshotLookup = Callable[[str], TaskSnapshot | None]
AcceptanceLookup = Callable[[str], bool]


class LifecycleLogger:
    """Writes lifecycle records to a file path or to an already-open text stream.

    ``accepted_of`` is the registry's verdict lookup: a published event the
    registry rejected (unknown task, late timestamp, terminal task) is never a
    lifecycle record, so the logger drops it instead of writing history the
    registry does not stand behind. Bus subscription order matters -- subscribe
    the logger after the registry, exactly as ``build_joint_run`` does.
    """

    def __init__(
        self,
        sink: str | os.PathLike[str] | TextIO,
        *,
        snapshot_of: SnapshotLookup | None = None,
        accepted_of: AcceptanceLookup | None = None,
    ) -> None:
        self.snapshot_of = snapshot_of
        self.accepted_of = accepted_of
        if isinstance(sink, (str, os.PathLike)):
            self._owns_stream = str(sink) != "-"
            self._stream: TextIO = (
                sys.stdout if not self._owns_stream else open(os.fspath(sink), "w", encoding="utf-8")
            )
        else:
            self._owns_stream = False
            self._stream = sink
        self._closed = False
        self.records_written = 0
        # The registry already ignores a re-published event; a log consumer has to as
        # well, or one duplicated publication inflates every downstream statistic.
        self._seen_event_ids: set[str] = set()

    def _first_time(self, event_id: str) -> bool:
        if event_id in self._seen_event_ids:
            return False
        self._seen_event_ids.add(event_id)
        return True

    def _registry_accepted(self, event_id: str) -> bool:
        """A rejected event is marked seen here so it can never be logged later."""
        return self.accepted_of is None or self.accepted_of(event_id)

    def reset(self) -> None:
        """Clear the per-episode dedup state so a second episode logs afresh.

        Records already written stay in the stream; the next episode's events are
        namespaced by their adapters' episode counters, so nothing is confused
        with this episode's records.
        """
        self._seen_event_ids.clear()
        self.records_written = 0

    def write(self, record: LifecycleRecord) -> LifecycleRecord:
        if self._closed:
            raise RuntimeError("lifecycle log is already closed")
        self._stream.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
        self._stream.flush()
        self.records_written += 1
        return record

    def log_run_config(self, configuration: Mapping[str, Any], scenario: Mapping[str, Any] | None = None) -> LifecycleRecord:
        detail: dict[str, Any] = {"configuration": dict(configuration)}
        if scenario is not None:
            detail["scenario"] = dict(scenario)
        return self.write(
            LifecycleRecord(
                target_id=RUN_TARGET_ID,
                event_type="run_config",
                timestamp=0.0,
                detail=detail,
            )
        )

    def handle_detection(self, event: DetectionEvent) -> LifecycleRecord | None:
        if not self._first_time(event.event_id):
            return None
        if not self._registry_accepted(event.event_id):
            return None
        snapshot = self._snapshot(event.target_id)
        return self.write(
            LifecycleRecord(
                target_id=event.target_id,
                event_type="detection",
                timestamp=event.detected_at,
                source_event_id=event.event_id,
                discovered_by=event.discovered_by,
                task_revision=snapshot.revision if snapshot else 1,
                stage_index=0,
                progress=0.0,
                remaining_requirements=tuple(
                    event.profile.stages[0].requirements if event.profile.stages else ()
                ),
                status="detected",
                detail={
                    "grid_position": [event.grid_position.row, event.grid_position.col],
                    "world_position": [event.world_position.x, event.world_position.y],
                    "confidence": event.confidence,
                    "severity": event.profile.severity,
                    "total_stages": len(event.profile.stages),
                    "deadline_offset": event.profile.deadline_offset,
                    "map_version": event.map_version,
                },
            )
        )

    def handle_progress(self, event: ProgressEvent) -> LifecycleRecord | None:
        if not self._first_time(event.event_id):
            return None
        if not self._registry_accepted(event.event_id):
            return None
        snapshot = self._snapshot(event.task_id)
        return self.write(
            LifecycleRecord(
                target_id=event.task_id,
                event_type="progress",
                timestamp=event.observed_at,
                # The originating detection, so every record for one target joins on the
                # same key; the progress event's own id stays available in detail.
                source_event_id=snapshot.source_event_id if snapshot else None,
                discovered_by=snapshot.metadata.get("discovered_by") if snapshot else None,
                assigned_agents=event.members,
                task_revision=snapshot.revision if snapshot else None,
                stage_index=event.stage_index,
                progress=event.progress,
                remaining_requirements=event.remaining_requirements,
                status=event.status.value,
                failure_reason=_failure_reason(event),
                detail={"expired": event.expired, "progress_event_id": event.event_id},
            )
        )

    def log_assignment(self, command: AssignmentCommand, *, step: int) -> LifecycleRecord:
        snapshot = self._snapshot(command.task_id) if command.task_id else None
        return self.write(
            LifecycleRecord(
                target_id=command.task_id or RUN_TARGET_ID,
                event_type="assignment",
                timestamp=command.issued_at,
                source_event_id=snapshot.source_event_id if snapshot else None,
                discovered_by=snapshot.metadata.get("discovered_by") if snapshot else None,
                assigned_agents=(command.agent_id,),
                task_revision=command.task_revision,
                stage_index=snapshot.current_stage if snapshot else None,
                progress=snapshot.progress if snapshot else None,
                remaining_requirements=(
                    snapshot.remaining_requirements if snapshot else None
                ),
                status=snapshot.status.value if snapshot else None,
                detail={
                    "command_id": command.command_id,
                    "reason": command.reason,
                    "valid_until": command.valid_until,
                    "step": step,
                },
            )
        )

    def log_rejection(self, rejection: Any, *, step: int) -> LifecycleRecord:
        snapshot = self._snapshot(rejection.task_id) if rejection.task_id else None
        return self.write(
            LifecycleRecord(
                target_id=rejection.task_id or RUN_TARGET_ID,
                event_type="assignment_rejected",
                timestamp=rejection.rejected_at,
                source_event_id=snapshot.source_event_id if snapshot else None,
                discovered_by=snapshot.metadata.get("discovered_by") if snapshot else None,
                assigned_agents=(rejection.agent_id,),
                task_revision=rejection.current_revision,
                stage_index=snapshot.current_stage if snapshot else None,
                progress=snapshot.progress if snapshot else None,
                remaining_requirements=(
                    snapshot.remaining_requirements if snapshot else None
                ),
                status=snapshot.status.value if snapshot else None,
                failure_reason=rejection.reason,
                detail={
                    "command_id": rejection.command_id,
                    "issued_at": rejection.issued_at,
                    "valid_until": rejection.valid_until,
                    "command_task_revision": rejection.task_revision,
                    "step": step,
                },
            )
        )

    def log_final(
        self,
        snapshots: Iterable[TaskSnapshot],
        *,
        step: int,
        current_time: float,
        metrics: Mapping[str, Any] | None = None,
    ) -> list[LifecycleRecord]:
        """Emit one closing record per target so the log ends with a full census."""
        written: list[LifecycleRecord] = []
        for snapshot in snapshots:
            written.append(
                self.write(
                    LifecycleRecord(
                        target_id=snapshot.task_id,
                        event_type="final",
                        timestamp=current_time,
                        source_event_id=snapshot.source_event_id,
                        discovered_by=snapshot.metadata.get("discovered_by"),
                        assigned_agents=snapshot.members,
                        task_revision=snapshot.revision,
                        stage_index=snapshot.current_stage,
                        progress=snapshot.progress,
                        remaining_requirements=snapshot.remaining_requirements,
                        status=snapshot.status.value,
                        failure_reason=(
                            "deadline_expired"
                            if snapshot.expired
                            else snapshot.metadata.get("failure_reason")
                        ),
                        detail={
                            "step": step,
                            "deadline": snapshot.deadline,
                            "discovered_at": snapshot.discovered_at,
                            "updated_at": snapshot.updated_at,
                            "position": [snapshot.position.x, snapshot.position.y],
                        },
                    )
                )
            )
        if metrics is not None:
            written.append(
                self.write(
                    LifecycleRecord(
                        target_id=RUN_TARGET_ID,
                        event_type="final",
                        timestamp=current_time,
                        status="run_finished",
                        detail={"step": step, "metrics": dict(metrics)},
                    )
                )
            )
        return written

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_stream and self._stream is not sys.stdout:
            self._stream.close()

    def __enter__(self) -> "LifecycleLogger":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _snapshot(self, target_id: str | None) -> TaskSnapshot | None:
        if target_id is None or self.snapshot_of is None:
            return None
        return self.snapshot_of(target_id)


def _failure_reason(event: ProgressEvent) -> str | None:
    if event.failure_reason:
        return str(event.failure_reason)
    if event.expired:
        return "deadline_expired"
    return None


def _jsonable(value: Any) -> Any:
    """Make numpy scalars and tuples safe for ``json.dumps`` without a custom encoder."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)
