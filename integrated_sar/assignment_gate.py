"""Guards that keep stale or expired allocation decisions away from the executor."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from .clock import GlobalClock
from .contracts import (
    AssignmentCommand,
    CoordinationAdvice,
    CoordinatorInput,
    ExecutionStepResult,
)
from .ports import Coordinator, Executor


@dataclass(frozen=True, slots=True)
class RejectedAssignment:
    command_id: str
    agent_id: str
    task_id: str | None
    issued_at: float
    valid_until: float
    task_revision: int | None
    current_revision: int | None
    reason: str
    rejected_at: float


class AssignmentGate:
    """
    Executor decorator that enforces command TTL and task-revision consistency.

    ``TaskRegistry`` is the only source of truth, so a command stamped with an older
    revision describes a task that has since changed. Applying it would let a stale
    decision overwrite newer state, so the gate drops it and asks for a replan
    instead of executing it.
    """

    def __init__(
        self,
        executor: Executor,
        *,
        clock: GlobalClock,
        revision_of: Callable[[str], int | None],
    ) -> None:
        self.executor = executor
        self.clock = clock
        self.revision_of = revision_of
        self.pending_replan: set[str] = set()
        self._rejections: list[RejectedAssignment] = []

    def apply_assignments(self, commands: Sequence[AssignmentCommand]) -> None:
        now = self.clock.now
        accepted: list[AssignmentCommand] = []
        for command in commands:
            reason, current_revision = self._inspect(command, now)
            if reason is None:
                accepted.append(command)
                continue
            self._rejections.append(
                RejectedAssignment(
                    command_id=command.command_id,
                    agent_id=command.agent_id,
                    task_id=command.task_id,
                    issued_at=command.issued_at,
                    valid_until=command.valid_until,
                    task_revision=command.task_revision,
                    current_revision=current_revision,
                    reason=reason,
                    rejected_at=now,
                )
            )
            self.pending_replan.add(command.agent_id)
        if accepted:
            self.executor.apply_assignments(accepted)

    def step(self, current_time: float, dt: float) -> ExecutionStepResult:
        return self.executor.step(current_time, dt)

    def drain_rejections(self) -> tuple[RejectedAssignment, ...]:
        rejections = tuple(self._rejections)
        self._rejections.clear()
        return rejections

    def pending_replans(self) -> frozenset[str]:
        """Robots whose command was dropped; the coordinator drains them into its advice."""
        return frozenset(self.pending_replan)

    @property
    def rejections(self) -> tuple[RejectedAssignment, ...]:
        return tuple(self._rejections)

    def _inspect(
        self, command: AssignmentCommand, now: float
    ) -> tuple[str | None, int | None]:
        if command.valid_until < now:
            return "expired_command", None
        if command.task_id is None:
            return None, None

        current_revision = self.revision_of(command.task_id)
        if current_revision is None:
            return "unknown_task", None
        if command.task_revision is None:
            return "missing_task_revision", current_revision
        if command.task_revision != current_revision:
            return "stale_task_revision", current_revision
        return None, current_revision


class ReplanCoordinator:
    """
    Coordinator decorator that turns gate rejections into ``replan_agents`` entries.

    The wrapped coordinator stays responsible for every bias value; this only widens
    the set of robots that must re-decide, which makes the orchestrator allocate
    again on the next tick instead of leaving a robot idle after a dropped command.
    """

    def __init__(self, inner: Coordinator, pending_replan: set[str]) -> None:
        self.inner = inner
        self.pending_replan = pending_replan

    def advise(self, state: CoordinatorInput) -> CoordinationAdvice:
        advice = self.inner.advise(state)
        if not self.pending_replan:
            return advice
        merged = frozenset(advice.replan_agents | self.pending_replan)
        self.pending_replan.clear()
        return replace(advice, replan_agents=merged)
