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

# Same sentinel old_network.CTDE_Network.get_action uses for illegal actions, so
# re-applying the mask after the coordinator residual cannot change which actions
# are legal -- only restore the constraint the residual was allowed to weaken.
MASKED_LOGIT = -1e9


class Task03Adapter:
    """
    Makes ``old_task_env.TaskEnv`` serve as task sink, allocator and executor.

    If ``network`` is omitted, an interpretable greedy allocator is used. When
    a CTDE network is supplied, coordinator advice is added as a bounded
    residual to its action logits.

    Injected targets go into the environment's ground truth only. How a robot
    learns about one is decided by explicit communication: an attached
    :class:`~integrated_sar.comm.CommRelay` routes published detections into
    robot beliefs over a seeded, lossy, delayed uplink, and TaskEnv's own
    ``comm_range`` sensing plus one-hop gossip remain the robot-to-robot
    channels. No decision here can outrun communication.
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
        seed: int | None = None,
    ) -> None:
        self.env = env
        self.network = network
        self.deterministic = bool(deterministic)
        self.residual_scale = float(residual_scale)
        self.command_ttl = float(command_ttl)
        self.comm_relay: Any | None = None
        self.external_to_internal: dict[str, int] = {}
        self.internal_to_external: dict[int, str] = {}
        self._task_revisions: dict[str, int] = {}
        self._progress_fingerprints: dict[str, tuple[Any, ...]] = {}
        self._command_counter = 0
        self._episode = 0
        self._seed = seed

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

    def reset(self) -> None:
        """Start a new episode on the same environment.

        Mappings, fingerprints and command numbering restart, and the episode counter
        keeps progress event ids from colliding with the previous episode's.
        """
        self._episode += 1
        self.external_to_internal.clear()
        self.internal_to_external.clear()
        self._task_revisions.clear()
        self._progress_fingerprints.clear()
        self._command_counter = 0
        self._clear_initial_tasks()

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
        # "discovered" is the label TaskEnv whitelists for tasks that appear at
        # runtime, and it drives the dynamic_tasks observation feature. Provenance
        # stays on the task record itself, in external_task_id and discovered_by.
        self.env.task_source[internal_id] = "discovered"
        self.env.tasks_num = len(self.env.task_dic)
        self.external_to_internal[task.task_id] = internal_id
        self.internal_to_external[internal_id] = task.task_id
        self._task_revisions[task.task_id] = task.revision
        # Deliberately no notification here: detections reach robot beliefs only
        # through the attached comm relay (topology, delay, loss, seed) and the
        # environment's own sensing and gossip.

    def deliver_detection(self, message: Any, agent_id: str) -> bool:
        """Merge one relayed detection into one robot's belief.

        Returns True when the belief actually changed. A message for a task the
        registry has not injected yet is dropped, and a message older than what
        the robot already believes never overwrites fresher knowledge.
        """
        agent_key = int(agent_id)
        knowledge = self.env.agent_knowledge.get(agent_key)
        if knowledge is None:
            return False
        internal_id = self.external_to_internal.get(message.target_id)
        if internal_id is None or internal_id not in self.env.task_dic:
            return False
        if (
            internal_id in knowledge["known_tasks"]
            and float(message.sent_at) <= knowledge["task_last_update_time"].get(internal_id, 0.0)
        ):
            return False

        stage_index = min(int(message.current_stage), len(message.stages) - 1)
        next_stage = message.stages[stage_index + 1].requirements if stage_index + 1 < len(message.stages) else ()
        view = {
            "task_id": internal_id,
            "location": (message.position.x, message.position.y),
            "deadline": float(message.deadline),
            "status": tuple(message.remaining_requirements),
            "requirements": tuple(message.stages[stage_index].requirements),
            "members": tuple(message.members),
            "update_time": float(message.sent_at),
            "stage_index": stage_index,
            "future_requirements": tuple(next_stage),
            "scene_complexity": float(message.severity),
            # Detections are dynamic tasks by provenance, matching what TaskEnv's
            # own sensing would have recorded for this task_source.
            "dynamic": True,
        }
        self.env.learn_task_view(agent_key, view)
        return True

    def agent_snapshots(self) -> tuple[AgentSnapshot, ...]:
        snapshots: list[AgentSnapshot] = []
        for internal_id, agent in self.env.agent_dic.items():
            task_id = self.internal_to_external.get(int(agent.get("current_task", -1)))
            knowledge = self.env.agent_knowledge.get(internal_id) or {}
            # Sorted so advice does not depend on set iteration order.
            known_tasks = tuple(
                sorted(
                    external
                    for external in (
                        self.internal_to_external.get(int(known))
                        for known in knowledge.get("known_tasks", ())
                    )
                    if external is not None
                )
            )
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
                    known_tasks=known_tasks,
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
                    internal_agent_id, mask, action_map, current_time, advice
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
        current_time: float,
        advice: CoordinationAdvice,
    ) -> int:
        """Score legal actions from this robot's belief, never from environment truth."""
        agent = self.env.agent_dic[agent_id]
        knowledge = self.env.agent_knowledge.get(agent_id) or {}
        believed_status = knowledge.get("task_current_status") or {}
        believed_locations = knowledge.get("task_locations") or {}
        believed_deadlines = knowledge.get("task_deadlines") or {}
        best_index, best_score = 0, -math.inf
        for index, (kind, internal_task_id) in enumerate(action_map):
            if index < len(mask) and bool(mask[index]):
                continue
            if kind == "depot" or internal_task_id is None:
                score = -0.25
            else:
                internal_task_id = int(internal_task_id)
                requirements = believed_status.get(internal_task_id)
                location = believed_locations.get(internal_task_id)
                if requirements is None or location is None:
                    continue  # not in this robot's belief, so it cannot act on it
                external_id = self.internal_to_external.get(internal_task_id)
                denominator = max(1.0, float(sum(requirements)))
                compatibility = sum(
                    min(float(ability), float(need))
                    for ability, need in zip(agent["abilities"], requirements)
                ) / denominator
                distance = float(self._norm(agent["location"] - location)) / math.sqrt(2.0)
                residual = advice.bias_for(str(agent_id), external_id or "")
                score = compatibility - 0.35 * distance + self.residual_scale * residual
                deadline = believed_deadlines.get(internal_task_id)
                if deadline is not None and float(deadline) < current_time:
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

        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            distribution = self.network.get_action(
                torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor(neighbors, dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor(tasks, dtype=torch.float32, device=device).unsqueeze(0),
                mask_tensor,
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
            # Invariant 6: the environment's hard mask is the last thing applied.
            # get_action already masked these logits, but the residual lands afterwards
            # and CoordinationAdvice bounds nothing, so without this a large enough bias
            # would make an illegal action selectable again.
            logits = logits.masked_fill(mask_tensor, MASKED_LOGIT)
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
        # The relay's delivery opportunity comes before the world advances, so a
        # message that arrives this tick can be acted on in this tick's allocation.
        if self.comm_relay is not None:
            self.comm_relay.tick()
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
                        f"task03-progress:e{self._episode}:{external_id}:"
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
                    failure_reason=self._failure_reason(task),
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

    def _failure_reason(self, task: dict[str, Any]) -> str | None:
        """Why this task is not succeeding, or None while it is progressing normally."""
        if bool(task.get("expired", False)):
            return "deadline_expired"
        if bool(task.get("finished", False)):
            return None
        internal_id = int(task["ID"])
        for agent in self.env.agent_dic.values():
            for record in agent.get("failed_tasks", []):
                if int(record.get("task_id", -1)) == internal_id:
                    return str(record.get("reason") or "assignment_failed")
        return None

    @staticmethod
    def _np_array(values: Any, dtype: Any) -> Any:
        import numpy as np

        return np.asarray(values, dtype=dtype)

    @staticmethod
    def _norm(values: Any) -> float:
        import numpy as np

        return float(np.linalg.norm(values))
