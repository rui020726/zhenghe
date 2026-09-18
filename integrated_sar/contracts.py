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
class DetectionMessage:
    """The payload a communication relay carries over the search-to-rescue link.

    This is not a second discovery event: :class:`DetectionEvent` is the lifecycle
    event published once on the bus, while a message is the letter itself -- a
    snapshot of the task as the registry accepted it at send time, delivered later
    (or never) to each subscribed robot's belief over an explicit topology with
    delay and loss.
    """

    event_id: str
    target_id: str
    sent_at: float
    position: WorldPosition
    stages: tuple[TaskStageSpec, ...]
    current_stage: int
    remaining_requirements: tuple[int, ...]
    deadline: float
    severity: float
    confidence: float
    progress: float = 0.0
    members: tuple[str, ...] = ()
    discovered_by: str | None = None
    # The registry revision this snapshot was taken from. ``sent_at`` alone cannot
    # order two updates that carry the same timestamp, which the registry accepts
    # (it only rejects strictly older ones), so delivery is decided on
    # (sent_at, revision) together.
    revision: int = 0

    def __post_init__(self) -> None:
        if not self.event_id or not self.target_id:
            raise ValueError("event_id and target_id are required")
        if self.current_stage < 0 or self.current_stage >= len(self.stages):
            raise ValueError("current_stage must index into stages")


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
class AgentTaskBelief:
    """What one robot believes about one task, as delivered by communication.

    This is the only task view a decentralized decision may use: the online
    coordinator scores pairs from it (invariant 9), and the executor navigates
    to ``position`` here rather than to registry truth.
    """

    task_id: str
    position: WorldPosition
    remaining_requirements: tuple[int, ...]
    deadline: float
    stage_index: int
    progress: float
    severity: float
    future_requirements: tuple[int, ...] = ()
    members: tuple[str, ...] = ()
    updated_at: float = 0.0
    finished: bool = False
    dynamic: bool = False
    # Keyword-only, and appended after every pre-existing field: a new positional
    # parameter would re-bind the arguments the earlier call sites already pass by
    # position (a requirements tuple landing in ``confidence``, for one).
    # Detection confidence as it arrived with the message; a robot that learned the
    # task by its own sensing simply keeps the default.
    confidence: float = field(default=0.5, kw_only=True)
    # Registry revision of this belief, so a newer fact with an unchanged timestamp
    # can still be ordered.
    revision: int = field(default=0, kw_only=True)
    # What the believed stage asks for in total, as delivered with the same message as
    # ``remaining_requirements``. The two are not interchangeable: ``remaining`` is what
    # the task still lacked *including this robot* among its members, so a robot that is
    # the sole provider of a requirement sees that requirement as zero. Only the gross
    # figure can say whether removing one member would uncover a unit.
    # Empty means the belief carries no stage requirement at all.
    stage_requirements: tuple[int, ...] = field(default=(), kw_only=True)


@dataclass(frozen=True, slots=True)
class AgentSnapshot:
    agent_id: str
    position: WorldPosition
    abilities: tuple[float, ...]
    status: str = "idle"
    current_task: str | None = None
    needs_decision: bool = True
    # External ids of the tasks this robot has actually heard about, by sensing or by
    # gossip. None means the agent source does not model belief, so nothing is filtered.
    known_tasks: tuple[str, ...] | None = None
    # The beliefs behind those ids. Empty means the source models no belief at all,
    # in which case consumers fall back to the id filter only.
    task_beliefs: tuple[AgentTaskBelief, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def belief_for(self, task_id: str) -> AgentTaskBelief | None:
        for belief in self.task_beliefs:
            if belief.task_id == task_id:
                return belief
        return None


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
class CoalitionReservation:
    """One robot's short-horizon commitment to one capability slot of one task.

    A per-pair bias scores a robot against a task in isolation, so two robots that
    would fill the *same* slot are scored identically and nothing stops the allocator
    from sending both. A reservation names the slot, so occupancy is exclusive and
    the group that assembles is the one the plan actually asked for -- the
    composition lever the pair interface cannot reach.

    ``ability_slot`` reads the task's remaining requirement as a flat list of unit
    slots: a requirement of ``(3, 0, 0, 0, 0)`` has three slots, in dimension 0, and
    a robot with one unit of that ability covers exactly one of them.
    ``slot_dimension`` names which ability dimension the slot belongs to, so a
    consumer never has to re-derive it from the requirement vector. Within one task
    two reservations may not share a slot index.

    ``task_revision`` is the revision of the task *as this robot believes it*, so the
    commitment is dropped as soon as that robot's own view moves on. It is never a
    registry revision: the coordinator does not read the registry (invariant 9).

    The commitment is deliberately short-horizon: ``valid_until`` bounds it, and a
    reserving planner must also bound its advice's ``valid_until`` by the earliest
    reservation expiry, because the advice is what the allocator reads and a cached
    advice is the only way an expired commitment could still be applied.

    ``commitment_id`` and ``created_at`` are the commitment's identity, as distinct from
    the instant it was last published (``issued_at``). A planner that keeps a robot on the
    task it already sent it to publishes the *same* commitment again with a later
    ``valid_until``; minting a new id there would make one continuous commitment look like
    a series of unrelated ones in the trace, and would hide the very churn the renewal
    exists to prevent. Only a fresh plan -- a robot that was not already committed to this
    task, in this slot -- creates a new id.
    """

    agent_id: str
    task_id: str
    ability_slot: int
    slot_dimension: int
    task_revision: int
    issued_at: float
    valid_until: float
    score: float
    # Identity, preserved across renewals. Keyword-only and appended so every positional
    # call site written before identity existed keeps binding what it meant.
    commitment_id: str = field(default="", kw_only=True)
    created_at: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not self.agent_id or not self.task_id:
            raise ValueError("agent_id and task_id are required")
        if self.ability_slot < 0 or self.slot_dimension < 0:
            raise ValueError("slot indices cannot be negative")
        if self.task_revision < 0:
            raise ValueError("task_revision cannot be negative")
        if self.valid_until < self.issued_at:
            raise ValueError("a reservation cannot expire before it was issued")
        if not -1.0 <= self.score <= 1.0:
            raise ValueError("reservation score must be bounded to [-1, 1]")
        # Defaulted rather than required: a caller that names a commitment by its slot
        # already has an identity, so it does not have to invent one to construct this.
        object.__setattr__(
            self,
            "commitment_id",
            self.commitment_id or f"{self.agent_id}|{self.task_id}|{self.ability_slot}",
        )
        if self.created_at is None:
            object.__setattr__(self, "created_at", float(self.issued_at))
        if float(self.created_at) > float(self.issued_at):
            raise ValueError("a reservation cannot be created after it was issued")

    def active_at(self, now: float) -> bool:
        """True while the commitment still stands. Expiry is the issuer's to enforce."""
        return now <= self.valid_until

    def renewed(self, *, issued_at: float, valid_until: float, task_revision: int, score: float) -> "CoalitionReservation":
        """The same commitment, published again with a later window.

        ``commitment_id``, ``created_at`` and the slot are carried over untouched: the
        robot is still under the order it was given. What describes this instant -- the
        publication instant, the window, the revision the robot now believes and the score
        the planner now assigns -- advances, so the identity stays and the facts move.
        """
        return CoalitionReservation(
            agent_id=self.agent_id,
            task_id=self.task_id,
            ability_slot=self.ability_slot,
            slot_dimension=self.slot_dimension,
            task_revision=int(task_revision),
            issued_at=float(issued_at),
            valid_until=float(valid_until),
            score=float(score),
            commitment_id=self.commitment_id,
            created_at=float(self.created_at),
        )


@dataclass(frozen=True, slots=True)
class CoordinationAdvice:
    revision: int
    generated_at: float
    valid_until: float
    # Pool-level bias for one task. Every robot that scores the task sees this number,
    # so a policy must only put facts that are common knowledge here -- never a value
    # derived from one robot's belief, which would broadcast that robot's knowledge to
    # robots it never reached (invariant 9).
    task_bias: Mapping[str, float] = field(default_factory=dict)
    # Per-robot task compatibility: ability overlap, believed distance, busy penalty.
    pair_bias: Mapping[tuple[str, str], float] = field(default_factory=dict)
    replan_agents: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Per-robot priority for one task: what this robot's own belief says about
    # urgency, remaining work, severity and confidence. Separate from ``pair_bias``
    # so a task-level term cannot leak across robots. Keyword-only and appended so
    # the pre-existing positional call sites keep binding replan_agents to
    # replan_agents.
    agent_task_bias: Mapping[tuple[str, str], float] = field(
        default_factory=dict, kw_only=True
    )
    # Short-horizon coalition commitments keyed by (agent, task): which robot covers
    # which capability slot of which task, until when. Empty by default, which is what
    # keeps a run without a reserving planner identical to one from before this field
    # existed. The key names a single robot, so unlike ``task_bias`` a reservation
    # cannot carry one robot's knowledge to robots it never reached (invariant 9).
    # Keyword-only and appended, so the earlier positional call sites keep binding
    # their arguments to the fields they already meant.
    reservations: Mapping[tuple[str, str], CoalitionReservation] = field(
        default_factory=dict, kw_only=True
    )

    def bias_for(self, agent_id: str, task_id: str) -> float:
        return (
            float(self.task_bias.get(task_id, 0.0))
            + float(self.pair_bias.get((agent_id, task_id), 0.0))
            + float(self.agent_task_bias.get((agent_id, task_id), 0.0))
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
    # Robots whose belief changed during this step (a message arrived, gossip or
    # sensing updated a task view). A belief revision is an explicit scheduling
    # signal: knowledge propagation must not be disguised as a ProgressEvent.
    # Appended after ``metrics`` so the earlier positional call sites keep meaning
    # what they meant.
    belief_changed_agents: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StepReport:
    step: int
    current_time: float
    detections: tuple[DetectionEvent, ...]
    progress_events: tuple[ProgressEvent, ...]
    assignments: tuple[AssignmentCommand, ...]
    advice: CoordinationAdvice
    active_tasks: tuple[TaskSnapshot, ...]
    belief_changed_agents: tuple[str, ...] = ()


DomainEvent = DetectionEvent | ProgressEvent
