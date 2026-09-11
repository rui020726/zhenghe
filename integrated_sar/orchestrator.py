"""The event-driven fast path and slower coordination loop."""

from __future__ import annotations

from typing import Iterable

from .contracts import (
    CoordinationAdvice,
    CoordinatorInput,
    DetectionEvent,
    ProgressEvent,
    StepReport,
)
from .event_bus import InMemoryEventBus
from .ports import AgentSource, Allocator, Coordinator, Executor, SearchModule, TaskSink
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
        self.registry = registry or TaskRegistry()
        self.event_bus = event_bus or InMemoryEventBus()
        self.dt = float(dt)
        self.coordination_interval = int(coordination_interval)
        self.allocation_interval = int(allocation_interval)
        self.step_index = 0
        self.current_time = 0.0
        self._advice = CoordinationAdvice(0, 0.0, 0.0)

        self.event_bus.subscribe(DetectionEvent, self.registry.handle_detection)
        self.event_bus.subscribe(ProgressEvent, self.registry.handle_progress)

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

        tasks = self.registry.active_snapshots()
        agents = tuple(self.agent_source.agent_snapshots())
        state_changed = bool(search_result.detections or execution_result.progress_events)

        advice_due = (
            state_changed
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
            or self.step_index % self.allocation_interval == 0
            or bool(self._advice.replan_agents)
        )
        assignments = ()
        if allocation_due:
            assignments = tuple(
                self.allocator.allocate(tasks, agents, self._advice, decision_time)
            )
            self.executor.apply_assignments(assignments)

        self.current_time = decision_time
        self.step_index += 1
        return StepReport(
            step=self.step_index,
            current_time=self.current_time,
            detections=tuple(search_result.detections),
            progress_events=tuple(execution_result.progress_events),
            assignments=assignments,
            advice=self._advice,
            active_tasks=self.registry.active_snapshots(),
        )

    def run(self, steps: int) -> Iterable[StepReport]:
        for _ in range(max(0, int(steps))):
            yield self.step()

    def reset(self) -> None:
        self.registry.clear()
        self.event_bus.clear_history()
        self.step_index = 0
        self.current_time = 0.0
        self._advice = CoordinationAdvice(0, 0.0, 0.0)
