"""The event-driven fast path and slower coordination loop."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .clock import GlobalClock
from .contracts import (
    CoordinationAdvice,
    CoordinatorInput,
    DetectionEvent,
    ProgressEvent,
    StepReport,
)
from .event_bus import InMemoryEventBus
from .ports import (
    AgentSource,
    Allocator,
    Coordinator,
    Executor,
    ReplanSource,
    SearchModule,
    TaskSink,
)
from .task_registry import TaskRegistry


class JointSAROrchestrator:
    def __init__(
        self,
        *,
        search: SearchModule,
        task_sink: TaskSink,
        agent_source: AgentSource,
        coordinator: Coordinator,
        allocator: Allocator,
        executor: Executor,
        registry: TaskRegistry | None = None,
        event_bus: InMemoryEventBus | None = None,
        clock: GlobalClock | None = None,
        replan_source: ReplanSource | None = None,
        dt: float = 1.0,
        coordination_interval: int = 5,
        allocation_interval: int = 5,
    ) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if coordination_interval < 1 or allocation_interval < 1:
            raise ValueError("decision intervals must be positive integers")

        self.search = search
        self.task_sink = task_sink
        self.agent_source = agent_source
        self.coordinator = coordinator
        self.allocator = allocator
        self.executor = executor
        # Explicit None checks: an empty TaskRegistry is falsy via __len__ and a freshly
        # started GlobalClock is falsy via __float__, so `x or Default()` would silently
        # discard the instance the caller injected.
        self.registry = registry if registry is not None else TaskRegistry()
        self.event_bus = event_bus if event_bus is not None else InMemoryEventBus()
        self.clock = clock if clock is not None else GlobalClock()
        self.replan_source = replan_source
        self.dt = float(dt)
        self.coordination_interval = int(coordination_interval)
        self.allocation_interval = int(allocation_interval)
        self.step_index = 0
        self._advice = CoordinationAdvice(0, 0.0, 0.0)

        self.event_bus.subscribe(DetectionEvent, self.registry.handle_detection)
        self.event_bus.subscribe(ProgressEvent, self.registry.handle_progress)

    @property
    def current_time(self) -> float:
        return self.clock.now

    def step(self) -> StepReport:
        now = self.current_time
        decision_time = now + self.dt

        search_result = self.search.step(now, self.dt)
        for event in search_result.detections:
            self.event_bus.publish(event)

        # Fast path: new detections are visible to the execution environment at once.
        self.task_sink.sync_tasks(self.registry.snapshots())

        execution_result = self.executor.step(now, self.dt)
        for event in execution_result.progress_events:
            self.event_bus.publish(event)

        # The physical step has happened: every later decision in this tick judges
        # command validity against the same global time the simulators now hold.
        self.clock.advance_to(decision_time)

        tasks = self.registry.active_snapshots()
        agents = tuple(self.agent_source.agent_snapshots())
        state_changed = bool(search_result.detections or execution_result.progress_events)
        # Rejections from the previous tick surface only now: the executor drops a
        # command while applying assignments, which is later than this tick's advice.
        replan_requested = bool(self._pending_replans())

        advice_due = (
            state_changed
            or replan_requested
            or self.step_index % self.coordination_interval == 0
            or self._advice.valid_until < decision_time
        )
        if advice_due:
            self._advice = self.coordinator.advise(
                CoordinatorInput(
                    current_time=decision_time,
                    tasks=tasks,
                    agents=agents,
                    search_metrics=search_result.metrics,
                    execution_metrics=execution_result.metrics,
                )
            )

        allocation_due = (
            state_changed
            or replan_requested
            or self.step_index % self.allocation_interval == 0
            or bool(self._advice.replan_agents)
        )
        assignments = ()
        advice_used = self._advice
        if allocation_due:
            assignments = tuple(
                self.allocator.allocate(tasks, agents, advice_used, decision_time)
            )
            self.executor.apply_assignments(assignments)
            # One-shot: that allocation is the answer to those replan requests. Leaving
            # them in the cached advice would re-trigger allocation on every following
            # tick even though nothing has changed since.
            if advice_used.replan_agents:
                self._advice = replace(advice_used, replan_agents=frozenset())

        self.step_index += 1
        return StepReport(
            step=self.step_index,
            current_time=self.current_time,
            detections=tuple(search_result.detections),
            progress_events=tuple(execution_result.progress_events),
            assignments=assignments,
            advice=advice_used,
            active_tasks=self.registry.active_snapshots(),
        )

    def _pending_replans(self) -> frozenset[str]:
        if self.replan_source is None:
            return frozenset()
        return frozenset(self.replan_source.pending_replans())

    def run(self, steps: int) -> Iterable[StepReport]:
        for _ in range(max(0, int(steps))):
            yield self.step()

    def reset(self) -> None:
        self.registry.clear()
        self.event_bus.clear_history()
        self.step_index = 0
        self.clock.reset()
        self._advice = CoordinationAdvice(0, 0.0, 0.0)
