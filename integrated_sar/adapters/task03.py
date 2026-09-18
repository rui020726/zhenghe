"""Adapter around task-03 old_task_env and optionally old_network."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..allocation_trace import EPS_ARRIVE, ROBOT_VELOCITY, DecisionFacts
from ..contracts import (
    AgentSnapshot,
    AgentTaskBelief,
    AssignmentCommand,
    CoalitionReservation,
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

# TaskEnv moves its robots by repeated floating-point addition, so a robot that
# reaches the edge of the scene can land one ulp outside the closed unit square
# (observed at -3.469446951953614e-18 after 80 steps of travel). ``WorldPosition``
# requires [0, 1], and the violation is pure rounding noise, not a lost robot.
# Anything inside this tolerance is clamped onto the boundary; anything larger is
# a real coordinate bug and must keep raising.
POSITION_TOLERANCE = 1e-9


def _unit_coordinate(value: float, *, what: str, axis: str) -> float:
    """Clamp a TaskEnv coordinate onto [0, 1] within rounding tolerance."""
    if -POSITION_TOLERANCE <= value < 0.0:
        return 0.0
    if 1.0 < value <= 1.0 + POSITION_TOLERANCE:
        return 1.0
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{what} {axis} coordinate {value!r} is outside [0, 1] by more than "
            f"{POSITION_TOLERANCE:g}; this is not rounding noise, so it is not clamped"
        )
    return value


def _world_position(x: float, y: float, *, what: str) -> WorldPosition:
    return WorldPosition(
        _unit_coordinate(x, what=what, axis="x"),
        _unit_coordinate(y, what=what, axis="y"),
    )


def _reservation_score(advice: CoordinationAdvice, agent_id: str, task_id: str) -> float:
    """The coalition commitment this robot holds for this task, or 0.0 if none.

    Reading the commitment here is what lets the same bounded residual channel carry
    both the per-pair bias and the team plan. Staleness is bounded by the advice
    window, not re-checked per pair: the planner that issues reservations also bounds
    its advice's ``valid_until`` by the earliest reservation expiry, and both the
    heuristic and the network path must treat a commitment the same way.
    """
    commitment = advice.reservations.get((agent_id, task_id))
    return 0.0 if commitment is None else float(commitment.score)


_RELEASE_REASONS = (
    "replan",
    "terminal_belief",
    "expired",
    "slot_filled",
    "not_on_menu",
    "hard_mask",
    "not_executable",
)


# The validity conditions, in the order they are checked. The first that fails is the
# release reason; each one is named so the trace can record the state a release was made
# from instead of leaving a reader to infer it from the reason alone.
_CONDITIONS = (
    ("replan", "not_replanning"),
    ("terminal_belief", "task_active_in_belief"),
    ("expired", "not_expired"),
    ("slot_filled", "slot_still_needed"),
    ("not_on_menu", "on_menu"),
    ("hard_mask", "hard_mask_legal"),
    ("not_executable", "agent_can_continue"),
)


class _Binding:
    """The outcome of testing one commitment against the conditions that keep it.

    ``held`` is the only thing the allocator acts on. ``reason`` is the first condition
    that failed, or None while the commitment stands; both go to the trace, so a released
    commitment is never mistaken for an ignored one. ``conditions`` is the prefix of the
    check that actually ran: the test stops at the first failure, so a condition after it
    was never evaluated and is absent rather than reported as false.
    """

    __slots__ = ("held", "reason", "action_index", "conditions")

    def __init__(
        self,
        held: bool,
        reason: str | None,
        action_index: int | None,
        conditions: Mapping[str, bool] | None = None,
    ) -> None:
        self.held = held
        self.reason = reason
        self.action_index = action_index
        self.conditions: dict[str, bool] = dict(conditions or {})


def _index_of_task(action_task_ids: Sequence[str | None], task_id: str) -> int | None:
    """The menu position of one task, or None when the robot is not offered it."""
    for index, external in enumerate(action_task_ids):
        if external == task_id:
            return index
    return None


def held_commitment(
    advice: CoordinationAdvice, agent_id: str, action_task_ids: Sequence[str | None]
) -> CoalitionReservation | None:
    """The commitment this robot holds for a task on its own menu, if any.

    Only tasks the robot could actually choose count. A commitment for a task that is
    absent from the menu, or masked, cannot be honoured or broken by this decision, so
    treating it as either would invent evidence about whether the plan was followed.
    """
    for external_id in action_task_ids:
        if external_id is None:
            continue
        commitment = advice.reservations.get((agent_id, external_id))
        if commitment is not None:
            return commitment
    return None


@dataclass(frozen=True, slots=True)
class _Verdict:
    """One allocator decision: the winning index plus the logits behind it.

    ``ctde`` and ``bias`` are ``None`` for a backend without a network, in which case
    ``final`` carries the scoring function's own values. Every vector is aligned with
    the environment's ``action_map`` and carries ``None`` where an action is masked, so
    a reader can tell "illegal" from "very unlikely" without knowing the -1e9 sentinel.
    """

    index: int
    ctde: tuple[float | None, ...] | None = None
    bias: tuple[float | None, ...] | None = None
    final: tuple[float | None, ...] | None = None


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
        ctde_task_logit_offset: float = 0.0,
        binding_commitments: bool = False,
        suppress_recommand_own_task: bool = False,
    ) -> None:
        self.env = env
        self.network = network
        self.deterministic = bool(deterministic)
        self.residual_scale = float(residual_scale)
        self.command_ttl = float(command_ttl)
        # Whether a commitment the advice still holds binds the allocator. Off by default:
        # the advisory arm is what every earlier number was measured on, and binding a
        # commitment changes what the allocator commands.
        self.binding_commitments = bool(binding_commitments)
        # Whether a replan-woke robot that re-picks the task it is already executing gets
        # a command at all. Re-issuing one would restart its work clock for no change --
        # the exact harm the commitment path's suppression exists to prevent -- so the
        # redundant-member release turns this on to keep the working timeout armed as the
        # natural next escape. Off by default: the baseline never reaches this branch.
        self.suppress_recommand_own_task = bool(suppress_recommand_own_task)
        # Scalar calibration of the CTDE task/depot preference: a constant added to
        # every *legal task* logit before the coordinator residual, never to the
        # depot and never to a masked action. Zero by default, so a run that does
        # not ask for calibration is bit-for-bit what it was before this existed.
        self.ctde_task_logit_offset = float(ctde_task_logit_offset)
        self.comm_relay: Any | None = None
        # A sink for per-decision facts, or None. Attached by the driver only when a run
        # asks for a trace; it is written to after the decision is made and is never read
        # back, so attaching it cannot change what the allocator does.
        self.decision_trace: Any | None = None
        self._commitments: dict[int, tuple[Any, str | None, float]] = {}
        self.external_to_internal: dict[str, int] = {}
        self.internal_to_external: dict[int, str] = {}
        self._task_revisions: dict[str, int] = {}
        self._progress_fingerprints: dict[str, tuple[Any, ...]] = {}
        self._command_counter = 0
        self._episode = 0
        self._seed = seed
        # Captured for reset(): a TaskEnv.reset regenerates depots from its own
        # rng, so the staging points this episode was built with must be reapplied.
        # Environments without a depot dictionary (stubs in tests) are tolerated.
        self._depot_positions = {
            species_id: depot["location"].copy()
            for species_id, depot in getattr(self.env, "depot_dic", {}).items()
        }

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
        """Start a new episode: restart the underlying simulator, then the mappings.

        ``TaskEnv.reset`` regenerates robots, depots, tasks, knowledge, the
        communication graph and every counter (including its clock and gossip
        counters), so no stale target reference, cooldown or working state can
        survive into the next episode. The rescue teams are then staged back on
        the depot positions this episode was built with, initial tasks (none in
        joint runs) are cleared again, and the attached comm relay drops its
        outbox. The episode counter keeps progress event ids from colliding with
        the previous episode's.
        """
        self._episode += 1
        env_reset = getattr(self.env, "reset", None)
        if env_reset is not None:
            env_reset(seed=self._seed)
        self._stage_rescue_teams()
        self._clear_initial_tasks()
        self.external_to_internal.clear()
        self.internal_to_external.clear()
        self._task_revisions.clear()
        self._progress_fingerprints.clear()
        self._commitments.clear()
        self._command_counter = 0
        if self.comm_relay is not None:
            self.comm_relay.reset()
        if self.decision_trace is not None:
            self.decision_trace.reset()

    def _stage_rescue_teams(self) -> None:
        """Move each species depot -- and its robots -- back to its staging point."""
        for species_id, point in self._depot_positions.items():
            depot = getattr(self.env, "depot_dic", {}).get(species_id)
            if depot is None:
                continue
            depot["location"] = self._np_array(point, dtype=float)
            for member in depot.get("members", ()):
                agent = self.env.agent_dic.get(member)
                if agent is None:
                    continue
                agent["location"] = depot["location"].copy()
                agent["depot"] = depot["location"].copy()

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
            env_task["confidence"] = float(task.confidence)
            env_task["revision"] = int(task.revision)
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
            "revision": int(task.revision),
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
        registry has not injected yet is dropped, and a message older than what the
        robot already believes never overwrites fresher knowledge.

        Freshness is decided on ``(sent_at, revision)`` together, because the
        registry accepts an update that carries the same timestamp as its record
        when the event id differs -- it only rejects a strictly older one. Ordering
        by time alone would silently discard exactly those revisions the registry
        just accepted.
        """
        agent_key = int(agent_id)
        knowledge = self.env.agent_knowledge.get(agent_key)
        if knowledge is None:
            return False
        internal_id = self.external_to_internal.get(message.target_id)
        if internal_id is None or internal_id not in self.env.task_dic:
            return False
        if internal_id in knowledge["known_tasks"]:
            known_time = float(knowledge["task_last_update_time"].get(internal_id, 0.0))
            known_revision = int(self._knowledge_revisions(knowledge).get(internal_id, 0))
            sent_at = float(message.sent_at)
            revision = int(getattr(message, "revision", 0))
            if sent_at < known_time:
                return False
            if sent_at == known_time and revision <= known_revision:
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
            "confidence": float(message.confidence),
            "revision": int(getattr(message, "revision", 0)),
            "progress": float(message.progress),
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
                    position=_world_position(
                        float(agent["location"][0]),
                        float(agent["location"][1]),
                        what=f"rescue robot {internal_id}",
                    ),
                    abilities=tuple(float(value) for value in agent["abilities"]),
                    status=str(agent.get("status", "idle")),
                    current_task=task_id,
                    needs_decision=bool(agent.get("need_decision", False)),
                    known_tasks=known_tasks,
                    task_beliefs=self._task_beliefs(knowledge),
                    metadata={"species": int(agent.get("species", -1))},
                )
            )
        return tuple(snapshots)

    def _knowledge_revisions(self, knowledge: Mapping[str, Any]) -> Mapping[Any, Any]:
        """The belief's registry revisions, tolerating environments that predate them."""
        return knowledge.get("task_revision") or {}

    def _task_beliefs(self, knowledge: Mapping[str, Any]) -> tuple[AgentTaskBelief, ...]:
        """The per-task beliefs behind one robot's ``known_tasks``, sorted by id."""
        beliefs: list[AgentTaskBelief] = []
        for internal_id in sorted(knowledge.get("known_tasks", ())):
            external_id = self.internal_to_external.get(int(internal_id))
            if external_id is None:
                continue
            location = knowledge.get("task_locations", {}).get(internal_id)
            remaining = knowledge.get("task_current_status", {}).get(internal_id)
            if location is None or remaining is None:
                continue
            beliefs.append(
                AgentTaskBelief(
                    task_id=external_id,
                    position=WorldPosition(float(location[0]), float(location[1])),
                    remaining_requirements=tuple(int(value) for value in remaining),
                    deadline=float(knowledge["task_deadlines"].get(internal_id, 0.0)),
                    stage_index=int(knowledge.get("task_stage_index", {}).get(internal_id, 0)),
                    progress=float(knowledge.get("task_progress", {}).get(internal_id, 0.0)),
                    severity=float(
                        knowledge.get("task_scene_complexity", {}).get(internal_id, 0.5)
                    ),
                    confidence=float(
                        knowledge.get("task_confidence", {}).get(internal_id, 0.5)
                    ),
                    revision=int(self._knowledge_revisions(knowledge).get(internal_id, 0)),
                    future_requirements=tuple(
                        int(value)
                        for value in knowledge.get("task_future_requirements", {}).get(
                            internal_id, ()
                        )
                    ),
                    members=tuple(
                        str(member)
                        for member in knowledge.get("task_members", {}).get(internal_id, ())
                    ),
                    stage_requirements=tuple(
                        int(value)
                        for value in knowledge.get("task_requirements", {}).get(internal_id, ())
                    ),
                    updated_at=float(knowledge["task_last_update_time"].get(internal_id, 0.0)),
                    finished=internal_id in knowledge.get("known_finished", ()),
                    dynamic=internal_id in knowledge.get("dynamic_tasks", ()),
                )
            )
        return tuple(beliefs)

    def allocate(
        self,
        tasks: Sequence[TaskSnapshot],
        agents: Sequence[AgentSnapshot],
        advice: CoordinationAdvice,
        current_time: float,
    ) -> tuple[AssignmentCommand, ...]:
        # ``tasks`` is read for one thing only: stamping each command with the
        # registry revision the assignment gate validates against, so a command
        # issued before a revision change is rejected rather than executed. Which
        # task a robot is sent to comes from that robot's own belief.
        task_by_id = {task.task_id: task for task in tasks}
        commands: list[AssignmentCommand] = []
        for agent in agents:
            if not agent.needs_decision and agent.agent_id not in advice.replan_agents:
                continue
            internal_agent_id = int(agent.agent_id)
            if internal_agent_id not in self.env.agent_dic:
                continue
            _, _, _, mask, action_map = self.env.get_local_obs(internal_agent_id)
            action_task_ids = tuple(
                self.internal_to_external.get(int(task_id)) if task_id is not None else None
                for _, task_id in action_map
            )
            commitment = held_commitment(advice, str(agent.agent_id), action_task_ids)
            binding = _Binding(False, None, None)
            if commitment is not None:
                binding = self._evaluate_commitment(
                    agent,
                    commitment,
                    action_index=_index_of_task(action_task_ids, commitment.task_id),
                    mask=mask,
                    current_time=current_time,
                    replanned=agent.agent_id in advice.replan_agents,
                )
            belief_status = belief_revision = None
            mask_legal = None
            if commitment is not None:
                believed = agent.belief_for(commitment.task_id)
                belief_status = (
                    "unknown"
                    if believed is None
                    else "finished" if believed.finished else "active"
                )
                belief_revision = None if believed is None else int(believed.revision)
                if binding.action_index is None:
                    mask_legal = False
                else:
                    mask_legal = not bool(mask[binding.action_index])
            if self.network is None:
                verdict = self._heuristic_verdict(
                    internal_agent_id, mask, action_map, current_time, advice
                )
                reason = "heuristic+coordination"
            else:
                verdict = self._network_verdict(internal_agent_id, action_map, advice)
                reason = "ctde+coordination"
            action_index = verdict.index
            forced_index = None
            if self.binding_commitments and binding.held and commitment is not None:
                forced = _index_of_task(action_task_ids, commitment.task_id)
                if forced is not None and forced != action_index:
                    # The commitment wins, and the fact that the allocator would have gone
                    # elsewhere is recorded rather than erased: that difference is the
                    # number the continuity fix exists to drive to zero.
                    action_index = forced
                    forced_index = forced
            commitment_suppressed = bool(
                self.binding_commitments
                and binding.held
                and commitment is not None
                and agent.current_task == commitment.task_id
            )
            # The command is built here, from the values the decision produced, and the
            # trace is given those values -- not a second derivation of them. Recording
            # first and commanding afterwards is what let a trace disagree with the log it
            # was supposed to explain.
            kind, internal_task_id = action_map[action_index]
            external_task_id = (
                self.internal_to_external.get(int(internal_task_id))
                if kind == "task" and internal_task_id is not None
                else None
            )
            own_task_suppressed = bool(
                not commitment_suppressed
                and self.suppress_recommand_own_task
                and external_task_id is not None
                and external_task_id == agent.current_task
            )
            # The decision re-picked the task this robot is already executing. Re-issuing
            # the command would restart its work clock for no change -- the same harm the
            # commitment path's suppression prevents -- so the robot simply keeps working,
            # and the trace names which of the two reasons held the command back.
            suppressed = commitment_suppressed or own_task_suppressed
            revision = self._command_revision(external_task_id, task_by_id)
            command_id = None
            if not suppressed:
                self._command_counter += 1
                command_id = f"assignment:{self._command_counter}"
            # The commitment was executed when the robot ends this decision on the task it
            # is committed to: either because the commitment forced the choice, or because
            # the allocator's own choice already was that task -- including the case where
            # it is already under way there and no new command is needed.
            reservation_executed = bool(
                commitment is not None
                and binding.held
                and binding.action_index is not None
                and action_index == binding.action_index
            )
            if self.decision_trace is not None:
                self.decision_trace.record(
                    self._decision_facts(
                        agent_id=str(agent.agent_id),
                        response=verdict,
                        action_map=action_map,
                        mask=mask,
                        current_time=current_time,
                        needs_decision=bool(agent.needs_decision),
                        replanned=agent.agent_id in advice.replan_agents,
                        belief_task_count=len(agent.known_tasks or ()),
                        advice=advice,
                        reserved_commitment=commitment,
                        binding=binding,
                        overridden=forced_index is not None,
                        belief_status=belief_status,
                        belief_revision=belief_revision,
                        mask_legal=mask_legal,
                        effective_index=action_index,
                        raw_index=int(verdict.index),
                        forced_index=forced_index,
                        actual_assignment=external_task_id,
                        command_id=command_id,
                        reservation_executed=reservation_executed,
                        suppressed_command=commitment_suppressed,
                        own_task_suppressed=own_task_suppressed,
                    )
                )

            if suppressed:
                # Already executing the committed task: re-issuing the command would
                # restart it, so this decision produces no command at all.
                continue

            release = advice.releases.get(agent.agent_id)
            command_metadata: dict[str, Any] = {}
            if release is not None and release.task_id != external_task_id:
                # The robot is being taken off the task it was working. Carry what the
                # coordinator believed about that task when it decided, so the gate can
                # check the release itself and not only where the command is going.
                command_metadata = {
                    "release_from_task": release.task_id,
                    "release_stage_index": int(release.stage_index),
                    "release_revision": int(release.revision),
                    "release_support_members": tuple(release.supporting_members),
                }
            commands.append(
                AssignmentCommand(
                    command_id=command_id,
                    agent_id=agent.agent_id,
                    task_id=external_task_id,
                    issued_at=current_time,
                    valid_until=current_time + self.command_ttl,
                    task_revision=revision,
                    reason=reason,
                    metadata=command_metadata,
                )
            )
        return tuple(commands)

    def _command_revision(
        self,
        external_task_id: str | None,
        active_by_id: Mapping[str, TaskSnapshot],
    ) -> int | None:
        """The registry revision to stamp on a command, or None for a depot action.

        A task the registry has closed is absent from the active snapshots, but it is
        not absent from this adapter's cache: ``sync_tasks`` records the revision of
        every injected task on every tick, terminal ones included. Stamping the cached
        revision is what lets such a command through the assignment gate instead of
        being refused as ``missing_task_revision``.

        That matters because a robot may hold a belief the registry has since
        closed -- invariant 9 forbids telling it otherwise, and the coordinator is
        not allowed to read the registry to compensate. Without the stamp the robot
        would re-propose the same closed task on every allocation cycle and never
        travel, so on-scene verification would never run and the task would stay in
        its belief forever, outbidding tasks that are still alive. With it, the robot
        goes, `verify_task_on_arrival` settles the question from the ground truth it
        is standing on, and its belief catches up.
        """
        if external_task_id is None:
            return None
        active = active_by_id.get(external_task_id)
        if active is not None:
            return active.revision
        return self._task_revisions.get(external_task_id)

    def _heuristic_action(
        self,
        agent_id: int,
        mask: Any,
        action_map: Sequence[tuple[str, int | None]],
        current_time: float,
        advice: CoordinationAdvice,
    ) -> int:
        return self._heuristic_verdict(agent_id, mask, action_map, current_time, advice).index

    def _heuristic_verdict(
        self,
        agent_id: int,
        mask: Any,
        action_map: Sequence[tuple[str, int | None]],
        current_time: float,
        advice: CoordinationAdvice,
    ) -> _Verdict:
        """Score legal actions from this robot's belief, never from environment truth."""
        agent = self.env.agent_dic[agent_id]
        knowledge = self.env.agent_knowledge.get(agent_id) or {}
        believed_status = knowledge.get("task_current_status") or {}
        believed_locations = knowledge.get("task_locations") or {}
        believed_deadlines = knowledge.get("task_deadlines") or {}
        best_index, best_score = 0, -math.inf
        scores: list[float | None] = [None] * len(action_map)
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
                residual += _reservation_score(advice, str(agent_id), external_id or "")
                score = compatibility - 0.35 * distance + self.residual_scale * residual
                deadline = believed_deadlines.get(internal_task_id)
                if deadline is not None and float(deadline) < current_time:
                    score -= 0.25
            scores[index] = score
            if score > best_score:
                best_index, best_score = index, score
        return _Verdict(index=best_index, final=tuple(scores))

    def _network_action(
        self,
        agent_id: int,
        action_map: Sequence[tuple[str, int | None]],
        advice: CoordinationAdvice,
    ) -> int:
        return self._network_verdict(agent_id, action_map, advice).index

    def _network_verdict(
        self,
        agent_id: int,
        action_map: Sequence[tuple[str, int | None]],
        advice: CoordinationAdvice,
    ) -> _Verdict:
        import torch

        state, neighbors, tasks, mask, _ = self.env.get_local_obs(agent_id)
        try:
            device = next(self.network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        illegal = [bool(value) for value in mask][: len(action_map)]
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            distribution = self.network.get_action(
                torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor(neighbors, dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor(tasks, dtype=torch.float32, device=device).unsqueeze(0),
                mask_tensor,
            )
            ctde = distribution.logits.clone()
            logits = ctde.clone()
            # Scalar calibration comes first: a constant added to legal task logits
            # only. The depot action keeps its logit, and a masked task is skipped,
            # so no offset -- however large -- can make an illegal action selectable
            # (invariant 6 is enforced again at the end regardless). The offset is
            # deliberately not folded into ``bias``: the trace reports calibration
            # and coordinator residual separately, so the recorded final logit can
            # be re-derived from all three numbers.
            if self.ctde_task_logit_offset:
                for position, (kind, internal_task_id) in enumerate(action_map):
                    if kind != "task" or internal_task_id is None:
                        continue
                    if position < len(illegal) and illegal[position]:
                        continue
                    logits[0, position] += self.ctde_task_logit_offset
            # The residual is zero on the depot action by construction: advice is only
            # ever produced for (robot, task) pairs, so "go home" cannot be biased.
            # A coalition commitment rides the same channel and the same bound, so the
            # recorded ``bias`` vector still satisfies final = ctde + offset + bias.
            bias = torch.zeros_like(logits)
            for position, (kind, internal_task_id) in enumerate(action_map):
                if kind != "task" or internal_task_id is None:
                    continue
                external_id = self.internal_to_external.get(int(internal_task_id))
                if external_id is not None:
                    agent_key = str(agent_id)
                    residual = advice.bias_for(agent_key, external_id)
                    residual += _reservation_score(advice, agent_key, external_id)
                    value = self.residual_scale * residual
                    bias[0, position] = value
                    logits[0, position] += value
            # Invariant 6: the environment's hard mask is the last thing applied.
            # get_action already masked these logits, but the residual lands afterwards
            # and CoordinationAdvice bounds nothing, so without this a large enough bias
            # would make an illegal action selectable again.
            logits = logits.masked_fill(mask_tensor, MASKED_LOGIT)
            residual_distribution = torch.distributions.Categorical(logits=logits)
            if self.deterministic:
                index = int(torch.argmax(logits, dim=-1).item())
            else:
                index = int(residual_distribution.sample().item())

        def visible(values: Any) -> tuple[float | None, ...]:
            return tuple(
                None if position < len(illegal) and illegal[position]
                else float(values[0, position])
                for position in range(len(action_map))
            )

        return _Verdict(
            index=index, ctde=visible(ctde), bias=visible(bias), final=visible(logits)
        )

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
            self._note_commitment(agent_id)

    def _note_commitment(self, agent_id: int) -> None:
        """Remember when a robot's navigation target last changed.

        Only the trace needs this: a commitment's age is that of the target change, not
        of the last decision, and a command applied here is the only place a target
        changes before the robot arrives. The bookkeeping is skipped entirely while no
        trace is attached.
        """
        if self.decision_trace is None:
            return
        agent = self.env.agent_dic.get(int(agent_id))
        if agent is None:
            return
        internal = agent.get("target_id")
        self._commitments[int(agent_id)] = (
            agent.get("target_type"),
            self.internal_to_external.get(int(internal)) if internal is not None else None,
            float(self.env.current_time),
        )

    def _evaluate_commitment(
        self,
        agent: AgentSnapshot,
        commitment: Any,
        *,
        action_index: int | None,
        mask: Any,
        current_time: float,
        replanned: bool,
    ) -> _Binding:
        """The conditions a held commitment must satisfy to keep this robot.

        Ordered so ``reason`` is the first condition that failed, and every one of them is
        checkable from what this adapter already holds: the robot's own belief, the advice
        it arrived in, and the hard mask it is about to be judged against. Nothing here
        reads the registry -- whether the task is still open is the robot's own belief
        (invariant 9), and the mask is the last word on legality (invariant 6).
        """
        conditions: dict[str, bool] = {}

        def check(name: str, holds: bool) -> bool:
            conditions[name] = bool(holds)
            return bool(holds)

        if not check("not_replanning", not replanned):
            return _Binding(False, "replan", action_index, conditions)
        belief = agent.belief_for(commitment.task_id)
        if not check(
            "task_active_in_belief", belief is not None and not belief.finished
        ):
            return _Binding(False, "terminal_belief", action_index, conditions)
        if not check("not_expired", commitment.active_at(float(current_time))):
            return _Binding(False, "expired", action_index, conditions)
        requirement = tuple(int(need) for need in belief.remaining_requirements)
        dimension = int(commitment.slot_dimension)
        needed = not (
            all(need <= 0 for need in requirement)
            or dimension >= len(requirement)
            or requirement[dimension] <= 0
        )
        if not check("slot_still_needed", needed):
            return _Binding(False, "slot_filled", action_index, conditions)
        # On the menu is the precondition that makes the mask check meaningful: an action
        # the robot is not offered has no mask entry and cannot be executed at all.
        if not check("on_menu", action_index is not None):
            return _Binding(False, "not_on_menu", action_index, conditions)
        if not check("hard_mask_legal", not bool(mask[action_index])):
            return _Binding(False, "hard_mask", action_index, conditions)
        can_continue = True
        if agent.current_task and agent.current_task != commitment.task_id:
            current = agent.belief_for(agent.current_task)
            # Mid-commitment elsewhere: yanking it would discard the work it has already
            # done. That other commitment has to be released first.
            can_continue = not (current is not None and not current.finished)
        if not check("agent_can_continue", can_continue):
            return _Binding(False, "not_executable", action_index, conditions)
        return _Binding(True, None, action_index, conditions)

    def _decision_facts(
        self,
        *,
        agent_id: str,
        response: _Verdict,
        action_map: Sequence[tuple[str, int | None]],
        mask: Any,
        current_time: float,
        needs_decision: bool,
        replanned: bool,
        belief_task_count: int,
        advice: CoordinationAdvice,
        reserved_commitment: Any = None,
        binding: Any = None,
        overridden: bool = False,
        belief_status: str | None = None,
        belief_revision: int | None = None,
        mask_legal: bool | None = None,
        effective_index: int | None = None,
        raw_index: int | None = None,
        forced_index: int | None = None,
        actual_assignment: str | None = None,
        command_id: str | None = None,
        reservation_executed: bool | None = None,
        suppressed_command: bool = False,
        own_task_suppressed: bool = False,
    ) -> DecisionFacts:
        """Gather one already-made decision for the trace.

        Nothing here can reach a decision: the action was chosen before this runs and the
        trace is a sink. The ``truth`` block is the exception that proves the rule -- it
        reads the environment's ground truth deliberately, because the questions this
        trace exists to answer ("is a sufficient group already on site", "how much of the
        deadline is left") are about the world, and a robot's belief lags the world by
        exactly the delay those questions are about.
        """
        import numpy as np

        raw = self.env.agent_dic[int(agent_id)]
        kind = raw.get("target_type")
        external_current = (
            self.internal_to_external.get(int(raw.get("current_task", -1)))
            if int(raw.get("current_task", -1)) >= 0
            else None
        )
        location = np.asarray(raw["location"], dtype=float)
        target_pos = raw.get("target_pos")
        distance = None
        if target_pos is not None:
            distance = float(np.linalg.norm(np.asarray(target_pos, dtype=float) - location))

        commitment = self._commitments.get(int(agent_id))
        if commitment is None or commitment[0] != kind or commitment[1] != external_current:
            # The target changed without a command (arrival, timeout, release), so the
            # change is visible only now: age is measured from here and understates by at
            # most one allocation interval.
            commitment = (kind, external_current, float(current_time))
            self._commitments[int(agent_id)] = commitment

        action_task_ids = tuple(
            self.internal_to_external.get(int(task_id)) if task_id is not None else None
            for _, task_id in action_map
        )
        truth: dict[str, Any] = {"targets": {}}
        # The executed action, not the allocator's own preference: the per-target timings
        # this block feeds are about where the robot was sent.
        executed_index = int(response.index if effective_index is None else effective_index)
        for external_id in (action_task_ids[executed_index], external_current):
            if external_id and external_id not in truth["targets"]:
                entry = self._target_truth(external_id)
                if entry is not None:
                    truth["targets"][external_id] = entry

        reserved = held_commitment(advice, str(agent_id), action_task_ids)
        reservation = (
            None
            if reserved is None
            else {
                "reservation_id": reserved.commitment_id,
                "task_id": reserved.task_id,
                "ability_slot": int(reserved.ability_slot),
                "slot_dimension": int(reserved.slot_dimension),
                "task_revision": int(reserved.task_revision),
                "created_at": float(reserved.created_at),
                "issued_at": float(reserved.issued_at),
                "valid_until": float(reserved.valid_until),
                "score": float(reserved.score),
                # Two different questions about one commitment, both answered here so a
                # reader never has to reconstruct either. ``executed`` says the decision
                # ended on the committed task -- the plan was carried out.
                # ``raw_choice_taken`` says the allocator's own preference was already
                # that task, which is the part of the fidelity that cost nothing.
                "executed": bool(reservation_executed),
                "raw_choice_taken": bool(
                    reserved.task_id == action_task_ids[response.index]
                ),
            }
        )

        return DecisionFacts(
            step=int(getattr(self.env, "_inner_step_counter", 0)),
            timestamp=float(current_time),
            agent_id=str(agent_id),
            status=str(raw.get("status", "idle")),
            abilities=tuple(float(value) for value in raw["abilities"]),
            position=(float(location[0]), float(location[1])),
            travel_dist=float(raw.get("travel_dist", 0.0)),
            needs_decision=bool(needs_decision),
            replanned=bool(replanned),
            belief_task_count=int(belief_task_count),
            action_task_ids=action_task_ids,
            mask=tuple(bool(value) for value in mask),
            chosen_index=int(
                response.index if effective_index is None else effective_index
            ),
            raw_choice=int(response.index if raw_index is None else raw_index),
            forced_choice=None if forced_index is None else int(forced_index),
            actual_assignment=actual_assignment,
            command_id=command_id,
            reservation_id=None if reserved is None else reserved.commitment_id,
            commitment_validity=dict(binding.conditions) if binding is not None else {},
            reservation_executed=(
                None if reservation_executed is None else bool(reservation_executed)
            ),
            ctde_logits=response.ctde,
            bias=response.bias,
            final_logits=response.final,
            task_logit_offset=(
                self.ctde_task_logit_offset if self.network is not None else None
            ),
            current_kind=str(kind) if kind is not None else None,
            current_task=external_current,
            current_task_age=max(0.0, float(current_time) - commitment[2]),
            current_target_distance=distance,
            current_target_eta=(
                None if distance is None else distance / ROBOT_VELOCITY
            ),
            current_target_taken=None if distance is None else distance <= EPS_ARRIVE,
            truth=truth,
            advice_revision=int(getattr(advice, "revision", 0)),
            advice_reservations=tuple(
                sorted(task_id for _, task_id in advice.reservations)
            ),
            held_reservation=reservation,
            advice_plan=dict(advice.metadata),
            commitment_present=reserved_commitment is not None,
            commitment_held=bool(binding is not None and binding.held),
            commitment_release_reason=None if binding is None else binding.reason,
            commitment_task=None if reserved_commitment is None else reserved_commitment.task_id,
            commitment_slot=(
                None if reserved_commitment is None else int(reserved_commitment.ability_slot)
            ),
            commitment_slot_dimension=(
                None
                if reserved_commitment is None
                else int(reserved_commitment.slot_dimension)
            ),
            commitment_belief_status=belief_status,
            commitment_belief_revision=belief_revision,
            commitment_mask_legal=mask_legal,
            commitment_overridden=bool(overridden),
            commitment_suppressed_command=bool(suppressed_command),
            own_task_suppressed=bool(own_task_suppressed),
        )

    def _target_truth(self, external_id: str) -> dict[str, Any] | None:
        """Ground truth about one target, written to the trace and read by nobody else."""
        import numpy as np

        internal_id = self.external_to_internal.get(external_id)
        task = self.env.task_dic.get(internal_id) if internal_id is not None else None
        if task is None:
            return None
        stages = task["stages"]
        stage = stages[min(int(task.get("current_stage_index", 0)), len(stages) - 1)]
        on_site = [int(member) for member in task.get("members", ())]
        internal_task_id = int(task["ID"])
        en_route = [
            int(other)
            for other, agent in self.env.agent_dic.items()
            if agent.get("target_type") == "task"
            and int(agent.get("target_id", -1)) == internal_task_id
        ]
        requirements = np.asarray(stage["requirements"], dtype=float)
        capacity = np.zeros_like(requirements)
        for other in on_site + en_route:
            capacity = capacity + np.asarray(self.env.agent_dic[other]["abilities"], dtype=float)
        remaining = np.asarray(stage["status"], dtype=float)
        return {
            "on_site": [str(member) for member in on_site],
            "en_route": [str(other) for other in en_route],
            # The stage's own status is the environment's answer to "do the robots on
            # site already cover the requirement": it holds requirements - member
            # abilities, floored at zero, and is all-zero exactly when they cover it.
            "remaining": [int(value) for value in remaining],
            "covered": bool((remaining <= 0).all()),
            "en_route_covers": bool((np.maximum(requirements - capacity, 0.0) <= 0).all()),
            # The stage requirement and the abilities on the way, so a reader can tell
            # "still missing" from "never asked for" and measure over-supply without a
            # second source. Both are ground truth: they are written, never read back.
            "requirements": [int(value) for value in requirements],
            "capacity": [float(value) for value in capacity],
            "detected_at": float(task.get("discovered_at", 0.0)),
            "deadline": float(task.get("deadline", 0.0)),
            "stage_index": int(task.get("current_stage_index", 0)),
            "expired": bool(task.get("expired", False)),
        }

    def step(self, current_time: float, dt: float) -> ExecutionStepResult:
        del current_time
        before = self._belief_fingerprints()
        # The relay's delivery opportunity comes before the world advances, so a
        # message that arrives this tick can be acted on in this tick's allocation.
        if self.comm_relay is not None:
            self.comm_relay.tick()
        self.env.step(dt)
        after = self._belief_fingerprints()
        changed = sorted(
            agent_id
            for agent_id in set(before) | set(after)
            if before.get(agent_id) != after.get(agent_id)
        )
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
            belief_changed_agents=tuple(str(agent_id) for agent_id in changed),
            metrics={
                "resource_utilization": float(self.env.get_resource_utilization()),
                "environment_time": float(self.env.current_time),
            },
        )

    def _belief_fingerprints(self) -> dict[int, tuple[Any, ...]]:
        """Content fingerprints of every robot's belief, for change detection.

        Every field that reaches a decision input is part of the fingerprint --
        remaining status, full requirements, stage index, future requirements,
        location, deadline, scene complexity, confidence, members, progress and the
        finished/assigned/dynamic flags -- not only the subset an earlier version
        happened to compare. Missing one means a delivered update does not trigger
        coordination until the periodic interval fires.

        Raw timestamps and the raw revision are deliberately excluded: a robot
        standing next to an idle task refreshes its timestamps every tick without
        learning anything new, and a revision bump with unchanged content is already
        a registry-accepted event, which the orchestrator schedules on separately.
        """
        import numpy as np

        def as_tuple(values: Any) -> tuple[Any, ...] | None:
            return None if values is None else tuple(np.asarray(values).ravel().tolist())

        fingerprints: dict[int, tuple[Any, ...]] = {}
        for agent_id, knowledge in self.env.agent_knowledge.items():
            status_of = knowledge.get("task_current_status", {})
            requirements_of = knowledge.get("task_requirements", {})
            stage_of = knowledge.get("task_stage_index", {})
            future_of = knowledge.get("task_future_requirements", {})
            locations_of = knowledge.get("task_locations", {})
            deadlines_of = knowledge.get("task_deadlines", {})
            complexity_of = knowledge.get("task_scene_complexity", {})
            confidence_of = knowledge.get("task_confidence", {})
            members_of = knowledge.get("task_members", {})
            progress_of = knowledge.get("task_progress", {})
            finished = knowledge.get("known_finished", ())
            assigned = knowledge.get("known_assigned", ())
            dynamic = knowledge.get("dynamic_tasks", ())

            entries = []
            for tid in knowledge.get("known_tasks", ()):
                entries.append(
                    (
                        int(tid),
                        as_tuple(status_of.get(tid)),
                        as_tuple(requirements_of.get(tid)),
                        int(stage_of.get(tid, -1)),
                        as_tuple(future_of.get(tid)),
                        as_tuple(locations_of.get(tid)),
                        None if tid not in deadlines_of else float(deadlines_of[tid]),
                        None if tid not in complexity_of else float(complexity_of[tid]),
                        None if tid not in confidence_of else float(confidence_of[tid]),
                        tuple(members_of.get(tid, ())),
                        float(progress_of.get(tid, 0.0)),
                        int(tid in finished),
                        int(tid in assigned),
                        int(tid in dynamic),
                    )
                )
            fingerprints[agent_id] = tuple(sorted(entries))
        return fingerprints

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
