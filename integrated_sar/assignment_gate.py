"""Guards that keep stale or expired allocation decisions away from the executor."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
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
    # Which task the refused command was leaving, when the refusal was about the release
    # rather than the destination. None for every pre-existing reason.
    release_from_task: str | None = None


class AssignmentGate:
    """
    Executor decorator that enforces command TTL, destination freshness and release safety.

    ``TaskRegistry`` is the only source of truth, so a command stamped with an older
    revision describes a task that has since changed. Applying it would let a stale
    decision overwrite newer state, so the gate drops it and asks for a replan
    instead of executing it.

    A release has a second end to check. The advice that moved a robot off the task it was
    working decided that was safe against a *stage* and against the *team* standing on that
    stage; if either has moved in the meantime, the judgement describes a state that no
    longer exists, so the command is dropped rather than quietly upgraded to the newer
    revision. The robot keeps the work it had, which is the conservative outcome the
    planner itself cannot see.

    ``release_state_of`` supplies the source task's live stage, ``release_roster_of`` its
    live member list. Both are execution-side truth that no actor, planner or observation
    reads: the gate only ever uses them to refuse, and a refusal feeds back as a
    ``replan_agents`` entry, which is the same channel every other rejection already uses.
    A batch is screened in order against the state each earlier acceptance produces, so two
    commands that are individually legal cannot jointly strip the source.
    """

    def __init__(
        self,
        executor: Executor,
        *,
        clock: GlobalClock,
        revision_of: Callable[[str], int | None],
        release_state_of: Callable[[str], tuple[int, int] | None] | None = None,
        release_roster_of: Callable[[str], Sequence[str] | None] | None = None,
    ) -> None:
        self.executor = executor
        self.clock = clock
        self.revision_of = revision_of
        # Supplied only when the wiring can report a task's live stage and revision; the
        # gate stays exactly as strict without it.
        self.release_state_of = release_state_of
        # Same condition: without a readable roster the support check simply does not run.
        self.release_roster_of = release_roster_of
        self.pending_replan: set[str] = set()
        self._rejections: list[RejectedAssignment] = []

    def apply_assignments(self, commands: Sequence[AssignmentCommand]) -> None:
        now = self.clock.now
        accepted: list[AssignmentCommand] = []
        # source task -> the robots this batch has already been allowed to take off it
        leaving: dict[str, set[str]] = defaultdict(set)
        for command in commands:
            reason, current_revision, release_from = self._inspect(command, now)
            if reason is None:
                reason, release_from = self._inspect_release(command, leaving)
            if reason is None:
                accepted.append(command)
                source = (command.metadata or {}).get("release_from_task")
                if source:
                    leaving[str(source)].add(command.agent_id)
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
                    release_from_task=release_from,
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

    def reset(self) -> None:
        """Drop the previous episode's rejections and outstanding replan requests.

        The wrapped executor is deliberately left alone: the orchestrator already
        visits every collaborator it was given, so delegating would reset the same
        adapter twice.
        """
        self._rejections.clear()
        self.pending_replan.clear()

    @property
    def rejections(self) -> tuple[RejectedAssignment, ...]:
        return tuple(self._rejections)

    def _inspect(
        self, command: AssignmentCommand, now: float
    ) -> tuple[str | None, int | None, str | None]:
        if command.valid_until < now:
            return "expired_command", None, None
        if command.task_id is None:
            return None, None, None

        current_revision = self.revision_of(command.task_id)
        if current_revision is None:
            return "unknown_task", None, None
        if command.task_revision is None:
            return "missing_task_revision", current_revision, None
        if command.task_revision != current_revision:
            return "stale_task_revision", current_revision, None

        metadata = command.metadata or {}
        source = metadata.get("release_from_task")
        if source and self.release_state_of is not None:
            state = self.release_state_of(str(source))
            if state is not None:
                stage_now, revision_now = state
                # The planner judged the release against a stage of the source task. If
                # that stage has advanced, the coverage the judgement relied on is gone.
                # Revision is not compared here: the registry bumps it on every accepted
                # event, so a belief is nearly always older than the live value and
                # refusing on it rejects the mechanism rather than the unsafe release.
                if int(metadata.get("release_stage_index", stage_now)) != int(stage_now):
                    return "stale_release_stage", revision_now, str(source)
        return None, current_revision, None

    def _inspect_release(
        self, command: AssignmentCommand, leaving: Mapping[str, set[str]]
    ) -> tuple[str | None, str | None]:
        """Does the team this release was judged against still stand on the source task?

        The verdict said "these teammates cover what this robot supplies". A supporter that
        has since left -- on its own timeout, on an earlier command, or on a companion
        release inside this very batch -- takes that reasoning with it, and the stage
        comparison alone would not notice.
        """
        metadata = command.metadata or {}
        source = metadata.get("release_from_task")
        if not source or self.release_roster_of is None:
            return None, None
        listed = self.release_roster_of(str(source))
        if listed is None:
            # Nothing to compare against. Parking a robot because a bookkeeping field cannot
            # be read would be a new failure mode, so this stays the stage check it was.
            return None, None
        roster = {str(member) for member in listed}
        if command.agent_id not in roster:
            return "stale_release_agent_moved", str(source)
        support = {str(member) for member in metadata.get("release_support_members", ())}
        if not support <= roster - leaving.get(str(source), set()):
            return "stale_release_support", str(source)
        return None, None


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

    def reset(self) -> None:
        """Advice numbering restarts with the wrapped coordinator.

        ``pending_replan`` is the gate's own set and is cleared by the gate's own
        reset, so it is deliberately left alone here.
        """
        reset = getattr(self.inner, "reset", None)
        if reset is not None:
            reset()
