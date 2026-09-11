"""Search–coordination–allocation–execution integration framework."""

from .contracts import (
    AgentSnapshot,
    AssignmentCommand,
    CoordinationAdvice,
    DetectionEvent,
    GridPosition,
    ProgressEvent,
    TaskSnapshot,
    TaskStageSpec,
    TaskStatus,
    TargetProfile,
    WorldPosition,
)
from .coordinator import HeuristicCoordinator, PassthroughCoordinator
from .event_bus import InMemoryEventBus
from .orchestrator import JointSAROrchestrator
from .task_registry import TaskRegistry

__all__ = [
    "AgentSnapshot",
    "AssignmentCommand",
    "CoordinationAdvice",
    "DetectionEvent",
    "GridPosition",
    "HeuristicCoordinator",
    "InMemoryEventBus",
    "JointSAROrchestrator",
    "PassthroughCoordinator",
    "ProgressEvent",
    "TaskRegistry",
    "TaskSnapshot",
    "TaskStageSpec",
    "TaskStatus",
    "TargetProfile",
    "WorldPosition",
]
