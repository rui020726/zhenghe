"""Adapter around task-03 old_task_env and optionally old_network."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ..contracts import (
    AgentSnapshot,
    AssignmentCommand,
    CoordinationAdvice,
    ExecutionStepResult,
    ProgressEvent,
    TaskSnapshot,
    TaskStatus,
    WorldPosition,
)


class Task03Adapter:
    """
    Makes ``old_task_env.TaskEnv`` serve as task sink, allocator and executor.

    If ``network`` is omitted, an interpretable greedy allocator is used. When
    a CTDE network is supplied, coordinator advice is added as a bounded
    residual to its action logits.
    """

    def __init__(
        self,
        *,
        env: Any,
        network: Any | None = None,
        deterministic: bool = True,
        residual_scale: float = 0.5,
        command_ttl: float = 10.0,
        clear_initial_tasks: bool = True,
        notify_mode: str = "all",
    ) -> None:
        if notify_mode not in {"all", "nearest"}:
            raise ValueError("notify_mode must be 'all' or 'nearest'")
        self.env = env
        self.network = network
        self.deterministic = bool(deterministic)
        self.residual_scale = float(residual_scale)
        self.command_ttl = float(command_ttl)
        self.notify_mode = notify_mode
        self.external_to_internal: dict[str, int] = {}
        self.internal_to_external: dict[int, str] = {}
        self._task_revisions: dict[str, int] = {}
        self._progress_fingerprints: dict[str, tuple[Any, ...]] = {}
        self._command_counter = 0

        self.env.enable_tidal_tasks = False
        self.env.enable_discovered_tasks = False
        if clear_initial_tasks:
            self._clear_initial_tasks()

    def _clear_initial_tasks(self) -> None:
        self.env.task_dic = {}
        self.env.task_source = {}
        self.env.tasks_num = 0
        self.env.next_dynamic_task_id = 0
        self.env.init_agent_memory()
        for agent in self.env.agent_dic.values():
            agent["need_decision"] = True
            agent["current_task"] = -1

    def sync_tasks(self, tasks: Sequence[TaskSnapshot]) -> None:
        for task in tasks:
            internal_id = self.external_to_internal.get(task.task_id)
            if internal_id is None:
                if task.status.terminal:
                    continue
                self._inject_task(task)
                continue

            self._task_revisions[task.task_id] = task.revision
            env_task = self.env.task_dic.get(internal_id)
            if env_task is None:
                continue
            env_task["deadline"] = float(task.deadline)
            env_task["location"] = self._np_array((task.position.x, task.position.y), dtype=float)
            if task.status.terminal:
                env_task["finished"] = True
                env_task["expired"] = task.status == TaskStatus.EXPIRED or task.expired

    def _inject_task(self, task: TaskSnapshot) -> None:
        internal_id = int(self.env.next_dynamic_task_id)
        while internal_id in self.env.task_dic:
            internal_id += 1
        self.env.next_dynamic_task_id = internal_id + 1

        stages: list[dict[str, Any]] = []
        for index, spec in enumerate(task.stages):
            requirements = self._np_array(spec.requirements, dtype=int)
            finished = index < task.current_stage
            stages.append(
                {
                    "stage_id": index,
                    "requirements": requirements,
                    "status": self._np_array(
                        tuple(0 for _ in spec.requirements)
                        if finished
                        else (
                            task.remaining_requirements
                            if index == task.current_stage
                            else spec.requirements
                        ),
                        dtype=int,
                    ),
                    "time": float(spec.duration),
                    "time_left": 0.0 if finished else float(spec.duration),
                    "finished": finished,
                }
            )

        current_stage = min(task.current_stage, len(stages) - 1)
        current = stages[current_stage]
        self.env.task_dic[internal_id] = {
            "ID": internal_id,
            "stages": stages,
            "current_stage_index": current_stage,
            "total_stages": len(stages),
            "requirements": current["requirements"].copy(),
            "status": current["status"].copy(),
            "time": current["time"],
            "location": self._np_array((task.position.x, task.position.y), dtype=float),
            "scene_complexity": float(task.severity),
            "finished": False,
            "feasible_assignment": False,
            "members": [],
            "time_start": float(task.discovered_at),
            "time_finish": 0.0,
            "deadline": float(task.deadline),
            "expired": bool(task.expired),
            "discovered_at": float(task.discovered_at),
            "discovered_by": task.metadata.get("discovered_by"),
            "external_task_id": task.task_id,
            "confidence": float(task.confidence),
        }
        self.env.task_source[internal_id] = "piloc"
        self.env.tasks_num = len(self.env.task_dic)
        self.external_to_internal[task.task_id] = internal_id
        self.internal_to_external[internal_id] = task.task_id
        self._task_revisions[task.task_id] = task.revision
        self._notify_agents(internal_id)

    def _notify_agents(self, internal_id: int) -> None:
        task = self.env.task_dic[internal_id]
        agent_ids = list(self.env.agent_dic.keys())
        if self.notify_mode == "nearest" and agent_ids:
            agent_ids = [
                min(
                    agent_ids,
                    key=lambda aid: float(
                        self._norm(self.env.agent_dic[aid]["location"] - task["location"])
                    ),
                )
            ]

        for agent_id in agent_ids:
            knowledge = self.env.agent_knowledge[agent_id]
            knowledge["known_tasks"].add(internal_id)
            knowledge["task_locations"][internal_id] = task["location"].copy()
            knowledge["task_requirements"][internal_id] = task["requirements"].copy()
            knowledge["task_deadlines"][internal_id] = task["deadline"]
            knowledge["dynamic_tasks"].add(internal_id)
            knowledge["task_current_status"][internal_id] = task["status"].copy()
            knowledge["task_members"][internal_id] = []
            knowledge["task_last_update_time"][internal_id] = self.env.current_time

    def agent_snapshots(self) -> tuple[AgentSnapshot, ...]:
        snapshots: list[AgentSnapshot] = []
        for internal_id, agent in self.env.agent_dic.items():
            task_id = self.internal_to_external.get(int(agent.get("current_task", -1)))
            snapshots.append(
                AgentSnapshot(
                    agent_id=str(internal_id),
                    position=WorldPosition(
                        float(agent["location"][0]), float(agent["location"][1])
                    ),
                    abilities=tuple(float(value) for value in agent["abilities"]),
                    status=str(agent.get("status", "idle")),
                    current_task=task_id,
                    needs_decision=bool(agent.get("need_decision", False)),
                    metadata={"species": int(agent.get("species", -1))},
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
        task_by_id = {task.task_id: task for task in tasks}
        commands: list[AssignmentCommand] = []
        for agent in agents:
            if not agent.needs_decision and agent.agent_id not in advice.replan_agents:
                continue
            internal_agent_id = int(agent.agent_id)
            if internal_agent_id not in self.env.agent_dic:
                continue
            _, _, _, mask, action_map = self.env.get_local_obs(internal_agent_id)
            if self.network is None:
                action_index = self._heuristic_action(
                    internal_agent_id, mask, action_map, task_by_id, advice
                )
                reason = "heuristic+coordination"
            else:
                action_index = self._network_action(
                    internal_agent_id, action_map, advice
                )
                reason = "ctde+coordination"

            kind, internal_task_id = action_map[action_index]
            external_task_id = (
                self.internal_to_external.get(int(internal_task_id))
                if kind == "task" and internal_task_id is not None
                else None
            )
            revision = (
                task_by_id[external_task_id].revision
                if external_task_id in task_by_id
                else None
            )
            self._command_counter += 1
            commands.append(
                AssignmentCommand(
                    command_id=f"assignment:{self._command_counter}",
                    agent_id=agent.agent_id,
                    task_id=external_task_id,
                    issued_at=current_time,
                    valid_until=current_time + self.command_ttl,
                    task_revision=revision,
                    reason=reason,
                )
            )
        return tuple(commands)

    def _heuristic_action(
        self,
        agent_id: int,
        mask: Any,
        action_map: Sequence[tuple[str, int | None]],
        task_by_id: dict[str, TaskSnapshot],
        advice: CoordinationAdvice,
    ) -> int:
        agent = self.env.agent_dic[agent_id]
        best_index, best_score = 0, -math.inf
        for index, (kind, internal_task_id) in enumerate(action_map):
            if index < len(mask) and bool(mask[index]):
                continue
            if kind == "depot" or internal_task_id is None:
                score = -0.25
            else:
                external_id = self.internal_to_external.get(int(internal_task_id))
                task = task_by_id.get(external_id or "")
                env_task = self.env.task_dic[int(internal_task_id)]
                requirements = env_task["status"]
                denominator = max(1.0, float(sum(requirements)))
                compatibility = sum(
                    min(float(ability), float(need))
                    for ability, need in zip(agent["abilities"], requirements)
                ) / denominator
                distance = float(self._norm(agent["location"] - env_task["location"])) / math.sqrt(2.0)
                residual = advice.bias_for(str(agent_id), external_id or "")
                score = compatibility - 0.35 * distance + self.residual_scale * residual
                if task is not None and task.expired:
                    score -= 0.25
            if score > best_score:
                best_index, best_score = index, score
        return best_index

    def _network_action(
        self,
        agent_id: int,
        action_map: Sequence[tuple[str, int | None]],
        advice: CoordinationAdvice,
    ) -> int:
        import torch

        state, neighbors, tasks, mask, _ = self.env.get_local_obs(agent_id)
        try:
            device = next(self.network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        with torch.no_grad():
            distribution = self.network.get_action(
                torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor(neighbors, dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor(tasks, dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0),
            )
            logits = distribution.logits.clone()
            for index, (kind, internal_task_id) in enumerate(action_map):
                if kind != "task" or internal_task_id is None:
                    continue
                external_id = self.internal_to_external.get(int(internal_task_id))
                if external_id is not None:
                    logits[0, index] += self.residual_scale * advice.bias_for(
                        str(agent_id), external_id
                    )
            residual_distribution = torch.distributions.Categorical(logits=logits)
            if self.deterministic:
                return int(torch.argmax(logits, dim=-1).item())
            return int(residual_distribution.sample().item())

    def apply_assignments(self, commands: Sequence[AssignmentCommand]) -> None:
        for command in commands:
            if command.valid_until < float(self.env.current_time):
                continue
            agent_id = int(command.agent_id)
            if agent_id not in self.env.agent_dic:
                continue
            _, _, _, _, action_map = self.env.get_local_obs(agent_id)
            desired_internal = (
                self.external_to_internal.get(command.task_id)
                if command.task_id is not None
                else None
            )
            desired_kind = "task" if desired_internal is not None else "depot"
            action_index = next(
                (
                    index
                    for index, (kind, task_id) in enumerate(action_map)
                    if kind == desired_kind
                    and (desired_kind == "depot" or int(task_id) == desired_internal)
                ),
                0,
            )
            self.env.apply_action(agent_id, action_index, action_map)

    def step(self, current_time: float, dt: float) -> ExecutionStepResult:
        del current_time
        self.env.step(dt)
        events: list[ProgressEvent] = []
        for external_id, internal_id in self.external_to_internal.items():
            task = self.env.task_dic.get(internal_id)
            if task is None:
                continue
            status, stage_index, remaining, progress, members = self._read_progress(task)
            fingerprint = (
                status.value,
                stage_index,
                remaining,
                round(progress, 6),
                members,
                bool(task.get("expired", False)),
            )
            if self._progress_fingerprints.get(external_id) == fingerprint:
                continue
            self._progress_fingerprints[external_id] = fingerprint
            events.append(
                ProgressEvent(
                    event_id=(
                        f"task03-progress:{external_id}:"
                        f"{int(getattr(self.env, '_inner_step_counter', 0))}"
                    ),
                    task_id=external_id,
                    observed_at=float(self.env.current_time),
                    status=status,
                    stage_index=stage_index,
                    remaining_requirements=remaining,
                    progress=progress,
                    members=members,
                    expired=bool(task.get("expired", False)),
                )
            )
        return ExecutionStepResult(
            progress_events=tuple(events),
            metrics={
                "resource_utilization": float(self.env.get_resource_utilization()),
                "environment_time": float(self.env.current_time),
            },
        )

    def _read_progress(
        self, task: dict[str, Any]
    ) -> tuple[TaskStatus, int, tuple[int, ...], float, tuple[str, ...]]:
        stages = task["stages"]
        total_stages = max(1, len(stages))
        raw_index = int(task.get("current_stage_index", 0))
        stage_index = min(raw_index, total_stages - 1)
        members = tuple(str(member) for member in task.get("members", []))

        if bool(task.get("finished", False)):
            status = TaskStatus.EXPIRED if bool(task.get("expired", False)) else TaskStatus.COMPLETED
            remaining = tuple(0 for _ in stages[-1]["requirements"])
            return status, stage_index, remaining, 1.0, members

        stage = stages[stage_index]
        remaining = tuple(int(value) for value in stage["status"])
        stage_fraction = 0.0
        if all(value <= 0 for value in remaining):
            duration = max(float(stage["time"]), 1e-6)
            stage_fraction = 1.0 - max(0.0, float(stage["time_left"])) / duration
        progress = max(0.0, min(1.0, (stage_index + stage_fraction) / total_stages))

        if bool(task.get("feasible_assignment", False)):
            status = TaskStatus.IN_PROGRESS
        elif members:
            status = TaskStatus.ASSIGNED
        elif any(
            agent.get("target_type") == "task"
            and agent.get("target_id") == task["ID"]
            for agent in self.env.agent_dic.values()
        ):
            status = TaskStatus.EN_ROUTE
        else:
            status = TaskStatus.DETECTED
        return status, stage_index, remaining, progress, members

    @staticmethod
    def _np_array(values: Any, dtype: Any) -> Any:
        import numpy as np

        return np.asarray(values, dtype=dtype)

    @staticmethod
    def _norm(values: Any) -> float:
        import numpy as np

        return float(np.linalg.norm(values))
