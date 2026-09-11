"""Shared, implementation-independent data contracts for integrated SAR."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class TaskStatus(str, Enum):
    DETECTED = "detected"
    VERIFIED = "verified"
    ASSIGNED = "assigned"
    EN_ROUTE = "en_route"
    IN_PROGRESS = "in_progress"
    PARTIAL = "partial"
    REPLAN_REQUIRED = "replan_required"
    COMPLETED = "completed"
    EXPIRED = "expired"
    LOST = "lost"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {
            TaskStatus.COMPLETED,
            TaskStatus.EXPIRED,
            TaskStatus.LOST,
            TaskStatus.FAILED,
        }


@dataclass(frozen=True, slots=True)
class GridPosition:
    row: int
    col: int


@dataclass(frozen=True, slots=True)
class WorldPosition:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.x <= 1.0 or not 0.0 <= self.y <= 1.0:
            raise ValueError("world coordinates must be normalized to [0, 1]")


@dataclass(frozen=True, slots=True)
class CoordinateTransform:
    """Converts PILOC (row, col) positions to TaskEnv normalized (x, y)."""

    rows: int
    cols: int

    def __post_init__(self) -> None:
        if self.rows < 2 or self.cols < 2:
            raise ValueError("grid must contain at least two rows and two columns")

    def to_world(self, position: GridPosition) -> WorldPosition:
        if not 0 <= position.row < self.rows or not 0 <= position.col < self.cols:
            raise ValueError(f"grid position outside {self.rows}x{self.cols}: {position}")
        return WorldPosition(
            x=position.col / float(self.cols - 1),
            y=position.row / float(self.rows - 1),
        )

    def to_grid(self, position: WorldPosition) -> GridPosition:
        return GridPosition(
            row=int(round(position.y * (self.rows - 1))),
            col=int(round(position.x * (self.cols - 1))),
        )


@dataclass(frozen=True, slots=True)
class TaskStageSpec:
    requirements: tuple[int, ...]
    duration: float

    def __post_init__(self) -> None:
        if not self.requirements:
            raise ValueError("a stage must contain at least one ability dimension")
        if any(value < 0 for value in self.requirements):
            raise ValueError("stage requirements cannot be negative")
        if self.duration <= 0:
            raise ValueError("stage duration must be positive")


@dataclass(frozen=True, slots=True)
class TargetProfile:
    stages: tuple[TaskStageSpec, ...]
    deadline_offset: float
    severity: float = 0.5
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a target profile must contain at least one stage")
        width = len(self.stages[0].requirements)
        if any(len(stage.requirements) != width for stage in self.stages):
            raise ValueError("all task stages must use the same ability dimension")
        if self.deadline_offset <= 0:
            raise ValueError("deadline_offset must be positive")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class DetectionEvent:
    event_id: str
    target_id: str
    grid_position: GridPosition
    world_position: WorldPosition
    detected_at: float
    discovered_by: str
    confidence: float
    profile: TargetProfile
    map_version: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id or not self.target_id:
            raise ValueError("event_id and target_id are required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    event_id: str
    task_id: str
    observed_at: float
    status: TaskStatus
    stage_index: int
    remaining_requirements: tuple[int, ...]
    progress: float
    members: tuple[str, ...] = ()
    expired: bool = False
    failure_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError("progress must be in [0, 1]")
        if self.stage_index < 0:
            raise ValueError("stage_index cannot be negative")
        if any(value < 0 for value in self.remaining_requirements):
            raise ValueError("remaining requirements cannot be negative")


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
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
    revision: int
    members: tuple[str, ...] = ()
    expired: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return not self.status.terminal


@dataclass(frozen=True, slots=True)
class AgentSnapshot:
    agent_id: str
    position: WorldPosition
    abilities: tuple[float, ...]
    status: str = "idle"
    current_task: str | None = None
    needs_decision: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AssignmentCommand:
    command_id: str
    agent_id: str
    task_id: str | None
    issued_at: float
    valid_until: float
    task_revision: int | None
    reason: str = "allocator"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CoordinationAdvice:
    revision: int
    generated_at: float
    valid_until: float
    task_bias: Mapping[str, float] = field(default_factory=dict)
    pair_bias: Mapping[tuple[str, str], float] = field(default_factory=dict)
    replan_agents: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def bias_for(self, agent_id: str, task_id: str) -> float:
        return float(self.task_bias.get(task_id, 0.0)) + float(
            self.pair_bias.get((agent_id, task_id), 0.0)
        )


@dataclass(frozen=True, slots=True)
class CoordinatorInput:
    current_time: float
    tasks: tuple[TaskSnapshot, ...]
    agents: tuple[AgentSnapshot, ...]
    search_metrics: Mapping[str, Any] = field(default_factory=dict)
    execution_metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchStepResult:
    detections: tuple[DetectionEvent, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionStepResult:
    progress_events: tuple[ProgressEvent, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StepReport:
    step: int
    current_time: float
    detections: tuple[DetectionEvent, ...]
    progress_events: tuple[ProgressEvent, ...]
    assignments: tuple[AssignmentCommand, ...]
    advice: CoordinationAdvice
    active_tasks: tuple[TaskSnapshot, ...]


DomainEvent = DetectionEvent | ProgressEvent
