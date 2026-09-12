"""Small dependency-free demonstration of the complete information loop."""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import (
    AgentSnapshot,
    AgentTaskBelief,
    AssignmentCommand,
    CoordinationAdvice,
    DetectionEvent,
    ExecutionStepResult,
    GridPosition,
    ProgressEvent,
    SearchStepResult,
    TaskSnapshot,
    TaskStageSpec,
    TaskStatus,
    TargetProfile,
    WorldPosition,
)
from .coordinator import HeuristicCoordinator
from .orchestrator import JointSAROrchestrator


class OneShotSearch:
    def __init__(self) -> None:
        self.emitted = False

    def step(self, current_time: float, dt: float) -> SearchStepResult:
        if self.emitted:
            return SearchStepResult(metrics={"coverage": 0.8})
        self.emitted = True
        profile = TargetProfile(
            stages=(TaskStageSpec((1, 1, 0, 0, 0), duration=2.0),),
            deadline_offset=20.0,
            severity=0.9,
        )
        event = DetectionEvent(
            event_id="demo-detection-1",
            target_id="demo-target-1",
            grid_position=GridPosition(30, 42),
            world_position=WorldPosition(42 / 59, 30 / 59),
            detected_at=current_time + dt,
            discovered_by="search-0",
            confidence=0.95,
            profile=profile,
            map_version=1,
        )
        return SearchStepResult((event,), {"coverage": 0.65})


class SimpleResponseTeam:
    """Tiny executor/allocator used only to prove the framework wiring."""

    def __init__(self) -> None:
        self.tasks: dict[str, TaskSnapshot] = {}
        self.progress: dict[str, float] = {}
        self.assignments: dict[str, str] = {}
        self._event_counter = 0
        self._command_counter = 0
        self._agents = {
            "rescue-0": {
                "position": WorldPosition(0.1, 0.1),
                "abilities": (1.0, 0.0, 0.0, 0.0, 0.0),
            },
            "rescue-1": {
                "position": WorldPosition(0.2, 0.1),
                "abilities": (0.0, 1.0, 0.0, 0.0, 0.0),
            },
        }

    def sync_tasks(self, tasks: Sequence[TaskSnapshot]) -> None:
        self.tasks = {task.task_id: task for task in tasks}
        for task in tasks:
            self.progress.setdefault(task.task_id, task.progress)

    def agent_snapshots(self) -> tuple[AgentSnapshot, ...]:
        """The team, with the task belief each robot holds.

        This demo's response team is also the task sink, so every robot here has been
        told about the pool; the beliefs below are that knowledge. In a real joint run
        the rescue side learns detections through the relay in ``integrated_sar.comm``
        instead, and the same belief fields gate its decisions.
        """
        known = tuple(sorted(self.tasks))
        beliefs = tuple(
            AgentTaskBelief(
                task_id=task.task_id,
                position=task.position,
                remaining_requirements=task.remaining_requirements,
                deadline=task.deadline,
                stage_index=task.current_stage,
                progress=self.progress.get(task.task_id, task.progress),
                severity=task.severity,
                confidence=task.confidence,
                revision=task.revision,
                members=task.members,
                updated_at=task.updated_at,
                finished=task.status.terminal,
            )
            for task in sorted(self.tasks.values(), key=lambda item: item.task_id)
        )
        snapshots = []
        for agent_id, data in self._agents.items():
            current_task = self.assignments.get(agent_id)
            snapshots.append(
                AgentSnapshot(
                    agent_id=agent_id,
                    position=data["position"],
                    abilities=data["abilities"],
                    status="working" if current_task else "idle",
                    current_task=current_task,
                    needs_decision=current_task is None,
                    known_tasks=known,
                    task_beliefs=beliefs,
                )
            )
        return tuple(snapshots)

    def allocate(
        self,
        tasks: Sequence[TaskSnapshot],
        agents: Sequence[AgentSnapshot],
        advice: CoordinationAdvice,
        current_time: float,
    ) -> tuple[AssignmentCommand, ...]:
        active = [task for task in tasks if task.active]
        commands = []
        for agent in agents:
            if not agent.needs_decision or not active:
                continue
            task = max(active, key=lambda item: advice.bias_for(agent.agent_id, item.task_id))
            self._command_counter += 1
            commands.append(
                AssignmentCommand(
                    command_id=f"demo-command-{self._command_counter}",
                    agent_id=agent.agent_id,
                    task_id=task.task_id,
                    issued_at=current_time,
                    valid_until=current_time + 5.0,
                    task_revision=task.revision,
                    reason="demo-greedy+coordination",
                )
            )
        return tuple(commands)

    def apply_assignments(self, commands: Sequence[AssignmentCommand]) -> None:
        for command in commands:
            if command.task_id is not None:
                self.assignments[command.agent_id] = command.task_id

    def step(self, current_time: float, dt: float) -> ExecutionStepResult:
        grouped: dict[str, list[str]] = {}
        for agent_id, task_id in self.assignments.items():
            grouped.setdefault(task_id, []).append(agent_id)

        events = []
        for task_id, members in grouped.items():
            task = self.tasks.get(task_id)
            if task is None or task.status.terminal:
                continue
            team_abilities = [0.0] * len(task.remaining_requirements)
            for agent_id in members:
                for index, value in enumerate(self._agents[agent_id]["abilities"]):
                    team_abilities[index] += value
            feasible = all(
                have >= need
                for have, need in zip(team_abilities, task.remaining_requirements)
            )
            increment = 0.5 * dt if feasible else 0.0
            self.progress[task_id] = min(1.0, self.progress.get(task_id, 0.0) + increment)
            complete = self.progress[task_id] >= 1.0
            self._event_counter += 1
            events.append(
                ProgressEvent(
                    event_id=f"demo-progress-{self._event_counter}",
                    task_id=task_id,
                    observed_at=current_time + dt,
                    status=TaskStatus.COMPLETED if complete else TaskStatus.IN_PROGRESS,
                    stage_index=0,
                    remaining_requirements=(0, 0, 0, 0, 0) if feasible else task.remaining_requirements,
                    progress=self.progress[task_id],
                    members=tuple(members),
                )
            )
            if complete:
                for agent_id in members:
                    self.assignments.pop(agent_id, None)

        return ExecutionStepResult(
            tuple(events), metrics={"assigned_agents": len(self.assignments)}
        )


def build_demo() -> JointSAROrchestrator:
    response_team = SimpleResponseTeam()
    return JointSAROrchestrator(
        search=OneShotSearch(),
        task_sink=response_team,
        agent_source=response_team,
        coordinator=HeuristicCoordinator(validity=2.0),
        allocator=response_team,
        executor=response_team,
        dt=1.0,
        coordination_interval=2,
        allocation_interval=2,
    )


def main() -> None:
    orchestrator = build_demo()
    for report in orchestrator.run(4):
        print(
            f"step={report.step} time={report.current_time:.1f} "
            f"detections={len(report.detections)} "
            f"assignments={len(report.assignments)} "
            f"progress_events={len(report.progress_events)} "
            f"active_tasks={len(report.active_tasks)}"
        )
    task = orchestrator.registry.get("demo-target-1")
    print(f"final_status={task.status.value if task else 'missing'}")


if __name__ == "__main__":
    main()
