"""No online decision path may read registry truth instead of a robot's belief.

Third review round, finding 2: ``get_local_obs`` and ``get_hard_mask`` were
already belief-based, but the coordinator still scored pairs from the registry's
task snapshots (position, remaining requirements), ``apply_action`` navigated to
the truth position, and the en-route replan check used the truth status. A robot
with a stale belief therefore received task updates it never heard.

Fourth review round, findings 1 and 2: the coordinator still filtered beliefs
against the registry's active pool, read the registry's ``confidence``, triggered
a replan when the registry closed a task the robot still believed in, and
aggregated every robot's task priority into the pool-level ``task_bias`` -- a map
handed to every robot, which broadcast one robot's knowledge to robots the message
never reached. The coordinator now scores from beliefs alone and emits the priority
per robot.
"""

from __future__ import annotations

import unittest

import numpy as np

from integrated_sar.adapters.task03 import Task03Adapter
from integrated_sar.comm import CommRelay
from integrated_sar.contracts import (
    AgentSnapshot,
    AgentTaskBelief,
    CoordinatorInput,
    DetectionEvent,
    GridPosition,
    TargetProfile,
    TaskStageSpec,
    WorldPosition,
)
from integrated_sar.coordinator import HeuristicCoordinator
from integrated_sar.task03_runtime import build_task_env, load_task_env_module
from integrated_sar.task_registry import TaskRegistry

BELIEVED_POSITION = (0.9, 0.9)
TRUTH_ONLY_POSITION = (0.25, 0.25)


def _wire():
    env = build_task_env(
        load_task_env_module(),
        seed=7,
        species=1,
        per_species=1,
        comm_range=0.3,
        depot_positions=[(0.0, 0.0)],
    )
    adapter = Task03Adapter(env=env, clear_initial_tasks=True, seed=7)
    registry = TaskRegistry()
    event = DetectionEvent(
        event_id="d1",
        target_id="t1",
        grid_position=GridPosition(59, 59),
        world_position=WorldPosition(*BELIEVED_POSITION),
        detected_at=1.0,
        discovered_by="a0",
        confidence=1.0,
        profile=TargetProfile(
            stages=(TaskStageSpec((1, 0, 0, 0, 0), 3.0),), deadline_offset=500.0
        ),
    )
    registry.handle_detection(event)
    adapter.sync_tasks(registry.active_snapshots())
    relay = CommRelay(subscribers=("0",), writer=adapter, snapshot_of=registry.get, seed=7)
    adapter.comm_relay = relay
    relay.handle_detection(event)
    relay.tick()
    return env, adapter, registry


class CoordinatorBeliefTests(unittest.TestCase):
    def test_registry_only_updates_do_not_move_the_advice(self) -> None:
        env, adapter, registry = _wire()
        agents = adapter.agent_snapshots()
        before = HeuristicCoordinator().advise(
            CoordinatorInput(2.0, registry.active_snapshots(), agents)
        )

        # Move the registry truth and its confidence; the robot received nothing.
        registry._tasks["t1"].position = WorldPosition(*TRUTH_ONLY_POSITION)
        registry._tasks["t1"].confidence = 0.1
        after = HeuristicCoordinator().advise(
            CoordinatorInput(2.0, registry.active_snapshots(), agents)
        )

        self.assertAlmostEqual(before.pair_bias[("0", "t1")], after.pair_bias[("0", "t1")])
        self.assertAlmostEqual(
            before.agent_task_bias[("0", "t1")], after.agent_task_bias[("0", "t1")]
        )
        # The pool-level map is handed to every robot, so it stays empty: a
        # belief-derived priority placed there would be broadcast to robots the
        # message never reached.
        self.assertEqual(after.task_bias, {})
        self.assertAlmostEqual(
            float(env.agent_knowledge[0]["task_locations"][adapter.external_to_internal["t1"]][0]),
            BELIEVED_POSITION[0],
        )

    def test_a_registry_terminal_task_does_not_force_a_replan(self) -> None:
        from integrated_sar.contracts import TaskStatus

        env, adapter, registry = _wire()
        internal_id = adapter.external_to_internal["t1"]
        env.agent_dic[0].update({"current_task": internal_id, "status": "working"})
        agents = adapter.agent_snapshots()
        self.assertFalse(agents[0].belief_for("t1").finished)
        advice = HeuristicCoordinator().advise(
            CoordinatorInput(2.0, registry.active_snapshots(), agents)
        )
        self.assertEqual(advice.replan_agents, frozenset())

        # The registry closing the task is not a fact any robot has observed.
        registry._tasks["t1"].status = TaskStatus.COMPLETED
        advice = HeuristicCoordinator().advise(
            CoordinatorInput(2.0, registry.active_snapshots(), agents)
        )
        self.assertEqual(advice.replan_agents, frozenset())

        # Only the robot learning it replans, and the closed task is no longer scored.
        env.agent_knowledge[0]["known_finished"].add(internal_id)
        agents = adapter.agent_snapshots()
        advice = HeuristicCoordinator().advise(
            CoordinatorInput(2.0, registry.active_snapshots(), agents)
        )
        self.assertEqual(advice.replan_agents, frozenset({"0"}))
        self.assertEqual(advice.agent_task_bias, {})
        self.assertEqual(advice.pair_bias, {})

    def test_the_belief_position_is_what_moves_the_pair_bias(self) -> None:
        env, adapter, registry = _wire()
        agents = adapter.agent_snapshots()
        before = HeuristicCoordinator().advise(
            CoordinatorInput(2.0, registry.active_snapshots(), agents)
        )

        # A genuinely delivered update does move it: the belief is live, not frozen.
        env.agent_knowledge[0]["task_locations"][adapter.external_to_internal["t1"]] = (
            np.array(TRUTH_ONLY_POSITION)
        )
        after = HeuristicCoordinator().advise(
            CoordinatorInput(2.0, registry.active_snapshots(), adapter.agent_snapshots())
        )
        self.assertGreater(after.pair_bias[("0", "t1")], before.pair_bias[("0", "t1")])

    def test_belief_confidence_is_what_moves_the_priority(self) -> None:
        # The registry's confidence is not a robot-visible number, so the belief
        # carries its own copy and that copy is what the coordinator reads.
        env, adapter, registry = _wire()
        internal_id = adapter.external_to_internal["t1"]
        before = HeuristicCoordinator().advise(
            CoordinatorInput(2.0, registry.active_snapshots(), adapter.agent_snapshots())
        )

        # The registry moves its confidence: nothing moves.
        registry._tasks["t1"].confidence = 0.1
        unchanged = HeuristicCoordinator().advise(
            CoordinatorInput(2.0, registry.active_snapshots(), adapter.agent_snapshots())
        )
        self.assertAlmostEqual(before.bias_for("0", "t1"), unchanged.bias_for("0", "t1"))

        # A delivered value does move it, in both directions.
        env.agent_knowledge[0]["task_confidence"][internal_id] = 0.05
        lowered = HeuristicCoordinator().advise(
            CoordinatorInput(2.0, registry.active_snapshots(), adapter.agent_snapshots())
        )
        self.assertLess(lowered.bias_for("0", "t1"), before.bias_for("0", "t1"))

        env.agent_knowledge[0]["task_confidence"][internal_id] = 1.0
        raised = HeuristicCoordinator().advise(
            CoordinatorInput(2.0, registry.active_snapshots(), adapter.agent_snapshots())
        )
        self.assertAlmostEqual(raised.bias_for("0", "t1"), before.bias_for("0", "t1"))

    def test_a_belief_derived_priority_never_reaches_another_robot(self) -> None:
        # One robot learns the deadline changed. The other robot's belief is
        # untouched, so its bias must not move with it.
        def pool(informed_deadline: float) -> tuple[AgentSnapshot, ...]:
            return (
                AgentSnapshot(
                    "informed",
                    WorldPosition(0.9, 0.9),
                    (1.0, 0, 0, 0, 0),
                    known_tasks=("t1",),
                    task_beliefs=(
                        AgentTaskBelief(
                            "t1", WorldPosition(0.9, 0.9), (1, 0, 0, 0, 0),
                            informed_deadline, 0, 0.0, 0.5,
                        ),
                    ),
                ),
                AgentSnapshot(
                    "stale",
                    WorldPosition(0.4, 0.5),
                    (1.0, 0, 0, 0, 0),
                    known_tasks=("t1",),
                    task_beliefs=(
                        AgentTaskBelief(
                            "t1", WorldPosition(0.9, 0.9), (1, 0, 0, 0, 0),
                            1000.0, 0, 0.0, 0.5,
                        ),
                    ),
                ),
            )

        relaxed = HeuristicCoordinator().advise(CoordinatorInput(10.0, (), pool(1000.0)))
        urgent = HeuristicCoordinator().advise(CoordinatorInput(10.0, (), pool(10.0)))

        self.assertEqual(relaxed.task_bias, {})
        self.assertEqual(urgent.task_bias, {})
        self.assertAlmostEqual(
            relaxed.bias_for("stale", "t1"), urgent.bias_for("stale", "t1")
        )
        self.assertGreater(
            urgent.bias_for("informed", "t1"), relaxed.bias_for("informed", "t1")
        )
        self.assertEqual(
            sorted(agent for agent, _ in urgent.agent_task_bias), ["informed", "stale"]
        )

    def test_the_registry_task_list_is_not_consulted_at_all(self) -> None:
        from integrated_sar.contracts import TaskSnapshot, TaskStatus

        env, adapter, registry = _wire()
        agents = adapter.agent_snapshots()
        active = registry.get("t1")
        without_pool = HeuristicCoordinator().advise(CoordinatorInput(2.0, (), agents))
        with_pool = HeuristicCoordinator().advise(
            CoordinatorInput(2.0, (active,), agents)
        )
        self.assertEqual(without_pool.bias_for("0", "t1"), with_pool.bias_for("0", "t1"))

        # And a registry record the robot never heard of earns it nothing.
        unheard = TaskSnapshot(
            task_id="t2", source_event_id="d2", position=WorldPosition(0.2, 0.2),
            stages=(TaskStageSpec((1, 0, 0, 0, 0), 3.0),), current_stage=0,
            remaining_requirements=(1, 0, 0, 0, 0), progress=0.0, status=TaskStatus.DETECTED,
            deadline=500.0, confidence=1.0, severity=0.9, discovered_at=1.0, updated_at=1.0,
            revision=1,
        )
        advice = HeuristicCoordinator().advise(
            CoordinatorInput(2.0, (unheard,), agents)
        )
        self.assertEqual(advice.task_bias, {})
        self.assertNotIn(("0", "t2"), advice.agent_task_bias)
        self.assertNotIn(("0", "t2"), advice.pair_bias)


class ExecutionBeliefTests(unittest.TestCase):
    def test_navigation_target_comes_from_the_belief(self) -> None:
        env, adapter, registry = _wire()
        internal_id = adapter.external_to_internal["t1"]

        registry._tasks["t1"].position = WorldPosition(*TRUTH_ONLY_POSITION)
        adapter.sync_tasks(registry.active_snapshots())
        agents = adapter.agent_snapshots()
        advice = HeuristicCoordinator().advise(
            CoordinatorInput(2.0, registry.active_snapshots(), agents)
        )
        adapter.apply_assignments(
            adapter.allocate(registry.active_snapshots(), agents, advice, 2.0)
        )

        agent = env.agent_dic[0]
        np.testing.assert_allclose(agent["target_pos"], BELIEVED_POSITION)
        self.assertEqual(agent["target_id"], internal_id)
        # The believed expected state, not the truth's.
        np.testing.assert_array_equal(
            agent["target_task_expected_state"],
            env.agent_knowledge[0]["task_current_status"][internal_id],
        )

    def test_enroute_replan_follows_the_belief_not_the_truth(self) -> None:
        env, adapter, registry = _wire()
        internal_id = adapter.external_to_internal["t1"]
        env.agent_dic[0].update(
            {"target_type": "task", "target_id": internal_id, "status": "moving"}
        )

        # Truth says finished, the robot has not heard it: keep going.
        env.task_dic[internal_id]["finished"] = True
        self.assertEqual(env.check_agent_enroute_replan(0), (False, "OK"))

        # Once the belief catches up, the robot replans.
        env.agent_knowledge[0]["known_finished"].add(internal_id)
        needs_replan, reason = env.check_agent_enroute_replan(0)
        self.assertTrue(needs_replan)
        self.assertEqual(reason, "Task finished")

    def test_belief_carries_progress_for_the_coordinator(self) -> None:
        env, adapter, registry = _wire()
        snapshot = adapter.agent_snapshots()[0]
        belief = snapshot.belief_for("t1")
        self.assertIsNotNone(belief)
        self.assertEqual(belief.progress, 0.0)
        internal_id = adapter.external_to_internal["t1"]
        self.assertIn(internal_id, env.agent_knowledge[0]["task_progress"])


class StaleTerminalBeliefTests(unittest.TestCase):
    """A robot that still believes in a closed task must not starve a live one.

    Fifth review round, finding 1. The coordinator no longer reads the registry, so
    it keeps biasing a task the registry has closed, and the live task competes with
    it on belief alone. That command used to be stamped ``task_revision=None`` and
    refused with ``missing_task_revision`` on every allocation round -- reproduced at
    02d29e9, where the live task never ran at all. The stamp now comes from the
    adapter's own cache, so the robot travels, verifies on scene, and its belief
    catches up.
    """

    def _wire(self):
        env = build_task_env(
            load_task_env_module(),
            seed=7,
            species=1,
            per_species=1,
            comm_range=0.02,
            depot_positions=[(0.05, 0.05)],
        )
        adapter = Task03Adapter(env=env, clear_initial_tasks=True, seed=7)
        registry = TaskRegistry()
        events = (
            # The closed task sits next to the robot, so its belief-derived score wins
            # over the live task on the far side of the map.
            _event("closed", (0.12, 0.12), "d-closed"),
            _event("live", (0.95, 0.95), "d-live"),
        )
        for event in events:
            registry.handle_detection(event)
        adapter.sync_tasks(registry.snapshots())
        relay = CommRelay(
            subscribers=("0",), writer=adapter, snapshot_of=registry.get, seed=7
        )
        adapter.comm_relay = relay
        for event in events:
            relay.handle_detection(event)
        relay.tick()
        return env, adapter, registry

    def test_a_closed_task_is_settled_on_scene_and_the_live_task_runs(self) -> None:
        from integrated_sar.assignment_gate import AssignmentGate
        from integrated_sar.clock import GlobalClock
        from integrated_sar.contracts import TaskStatus

        env, adapter, registry = self._wire()
        self.assertEqual(adapter.agent_snapshots()[0].known_tasks, ("closed", "live"))
        closed_internal = adapter.external_to_internal["closed"]
        live_internal = adapter.external_to_internal["live"]

        # The registry closes one task; no message about it reaches the robot.
        registry._tasks["closed"].status = TaskStatus.COMPLETED
        registry._tasks["closed"].revision += 1
        adapter.sync_tasks(registry.snapshots())
        self.assertFalse(adapter.agent_snapshots()[0].belief_for("closed").finished)

        clock = GlobalClock()
        gate = AssignmentGate(
            adapter,
            clock=clock,
            revision_of=lambda task_id: (
                registry.get(task_id).revision if registry.get(task_id) else None
            ),
        )
        coordinator = HeuristicCoordinator()

        first_command = None
        for tick in range(1, 30):
            clock.advance_to(float(tick))
            agents = adapter.agent_snapshots()
            advice = coordinator.advise(
                CoordinatorInput(float(tick), registry.active_snapshots(), agents)
            )
            commands = adapter.allocate(
                registry.active_snapshots(), agents, advice, float(tick)
            )
            gate.apply_assignments(commands)
            self.assertEqual(
                gate.drain_rejections(),
                (),
                "the command was refused instead of letting the robot verify on scene",
            )
            if first_command is None and commands:
                first_command = commands[0]
            adapter.step(float(tick), 1.0)
            if env.task_dic[live_internal].get("finished"):
                break

        # The closed task was proposed first, and it travelled with the revision the
        # adapter cached for it rather than with None.
        self.assertEqual(first_command.task_id, "closed")
        self.assertIsNotNone(first_command.task_revision)

        # On scene the robot learned the truth, and the live task then ran to
        # completion instead of being starved.
        self.assertIn(closed_internal, env.agent_knowledge[0]["known_finished"])
        self.assertTrue(env.task_dic[live_internal].get("finished"))
        self.assertTrue(
            adapter.agent_snapshots()[0].belief_for("closed").finished,
            "the robot's belief never caught up with the closed task",
        )

    def test_a_task_completing_mid_tick_is_refused_once_and_then_travelled(self) -> None:
        """The one stale-revision shape the 400-seed block produced, reproduced.

        ``sync_tasks`` caches a closed task's revision at the top of the tick, the
        completion event then moves the registry on, and the command built later in the
        same tick carries the cached, one-behind stamp. The refusal must run nothing,
        must not repeat, and must not strand the robot.
        """
        from integrated_sar.assignment_gate import AssignmentGate
        from integrated_sar.clock import GlobalClock
        from integrated_sar.contracts import TaskStatus

        env, adapter, registry = self._wire()
        closed_internal = adapter.external_to_internal["closed"]
        clock = GlobalClock()
        gate = AssignmentGate(
            adapter,
            clock=clock,
            revision_of=lambda task_id: (
                registry.get(task_id).revision if registry.get(task_id) else None
            ),
        )
        coordinator = HeuristicCoordinator()
        rejections: list = []
        ticks_at_the_closed_task: list[int] = []

        for tick in range(1, 12):
            clock.advance_to(float(tick))
            adapter.sync_tasks(registry.snapshots())
            if tick == 1:
                # The task finishes after the cache was written and before the gate reads.
                registry._tasks["closed"].status = TaskStatus.COMPLETED
                registry._tasks["closed"].revision += 1
            agents = adapter.agent_snapshots()
            advice = coordinator.advise(
                CoordinatorInput(float(tick), registry.active_snapshots(), agents)
            )
            commands = adapter.allocate(
                registry.active_snapshots(), agents, advice, float(tick)
            )
            proposed = [command for command in commands if command.task_id == "closed"]
            if proposed:
                ticks_at_the_closed_task.append(tick)
            # The robot's own state is what a refused command must leave alone.
            untouched = (env.agent_dic[0]["status"], env.agent_dic[0]["current_task"])
            gate.apply_assignments(commands)
            round_rejections = gate.drain_rejections()
            rejections.extend(round_rejections)
            if round_rejections:
                self.assertEqual(
                    (env.agent_dic[0]["status"], env.agent_dic[0]["current_task"]),
                    untouched,
                    "a refused command changed the robot's execution state",
                )
            adapter.step(float(tick), 1.0)
            if closed_internal in env.agent_knowledge[0]["known_finished"]:
                break

        self.assertEqual([rejection.reason for rejection in rejections], ["stale_task_revision"])
        self.assertEqual(len(rejections), 1, "the same stale command must not be attempted twice")
        self.assertEqual((rejections[0].agent_id, rejections[0].task_id), ("0", "closed"))
        self.assertEqual(rejections[0].task_revision + 1, rejections[0].current_revision)
        # It proposed the closed task, was refused, and travelled to it one tick later.
        self.assertEqual(ticks_at_the_closed_task[:2], [1, 2])
        self.assertIn(closed_internal, env.agent_knowledge[0]["known_finished"])


def _event(target_id: str, world: tuple[float, float], event_id: str) -> DetectionEvent:
    return DetectionEvent(
        event_id=event_id,
        target_id=target_id,
        grid_position=GridPosition(59, 59),
        world_position=WorldPosition(*world),
        detected_at=1.0,
        discovered_by="a0",
        confidence=1.0,
        profile=TargetProfile(
            stages=(TaskStageSpec((1, 0, 0, 0, 0), 3.0),), deadline_offset=500.0
        ),
    )


if __name__ == "__main__":
    unittest.main()
