"""Ports that keep PILOC, coordination, CTDE allocation and execution replaceable."""

from __future__ import annotations

from typing import Protocol, Sequence

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
