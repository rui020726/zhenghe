"""Ports that keep PILOC, coordination, CTDE allocation and execution replaceable."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .contracts import (
    AgentSnapshot,
    AssignmentCommand,
    CoordinationAdvice,
    CoordinatorInput,
    ExecutionStepResult,
    SearchStepResult,
    TaskSnapshot,
)


class SearchModule(Protocol):
    def step(self, current_time: float, dt: float) -> SearchStepResult: ...


class TaskSink(Protocol):
    def sync_tasks(self, tasks: Sequence[TaskSnapshot]) -> None: ...


class AgentSource(Protocol):
    def agent_snapshots(self) -> Sequence[AgentSnapshot]: ...


class Coordinator(Protocol):
    def advise(self, state: CoordinatorInput) -> CoordinationAdvice: ...


class Allocator(Protocol):
    def allocate(
        self,
        tasks: Sequence[TaskSnapshot],
        agents: Sequence[AgentSnapshot],
        advice: CoordinationAdvice,
        current_time: float,
    ) -> Sequence[AssignmentCommand]: ...


class Executor(Protocol):
    def apply_assignments(self, commands: Sequence[AssignmentCommand]) -> None: ...

    def step(self, current_time: float, dt: float) -> ExecutionStepResult: ...


class ReplanSource(Protocol):
    """
    Reports robots whose last assignment was dropped and must decide again.

    A rejection is only known after the executor has applied assignments, which is
    later in the tick than coordination, so the scheduler has to ask before it can
    decide whether the next tick needs fresh advice.
    """

    def pending_replans(self) -> frozenset[str]: ...


@runtime_checkable
class Resettable(Protocol):
    """
    A collaborator holding per-episode state that must be cleared between episodes.

    Adapters map external ids to internal ones, fingerprint progress to suppress
    repeats and number their commands; none of that may survive into the next
    episode, because TaskEnv restarts its own counters and a repeated event id
    would be dropped as a duplicate.
    """

    def reset(self) -> None: ...
