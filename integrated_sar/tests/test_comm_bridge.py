"""Regression tests for the DetectionEvent -> rescue belief communication bridge.

A search detection must reach rescue robots as an explicit, seed-controlled
communication (topology, delay, packet loss) instead of requiring a robot to
physically approach the target, and instead of an unconditional broadcast.
"""

from __future__ import annotations

import unittest

import numpy as np

from integrated_sar.comm import CommRelay
from integrated_sar.contracts import (
    CoordinatorInput,
    DetectionEvent,
    GridPosition,
    TargetProfile,
    TaskStageSpec,
    WorldPosition,
)
from integrated_sar.coordinator import HeuristicCoordinator
from integrated_sar.task_registry import TaskRegistry


def make_detection(
    target_id: str = "remote-target",
    *,
    event_id: str | None = None,
    world: tuple[float, float] = (1.0, 1.0),
    detected_at: float = 1.0,
) -> DetectionEvent:
    return DetectionEvent(
        event_id=event_id or f"detection:{target_id}",
        target_id=target_id,
        grid_position=GridPosition(59, 59),
        world_position=WorldPosition(*world),
        detected_at=detected_at,
        discovered_by="agent_0",
        confidence=1.0,
        profile=TargetProfile(
            stages=(TaskStageSpec((1, 0, 0, 0, 0), 3.0),),
            deadline_offset=500.0,
        ),
    )


def wire_rescue_side(*, species: int, per_species: int, depot_positions, comm_range=0.3):
    from integrated_sar.adapters.task03 import Task03Adapter
    from integrated_sar.task03_runtime import build_task_env, load_task_env_module

    env = build_task_env(
        load_task_env_module(),
        seed=7,
        species=species,
        per_species=per_species,
        comm_range=comm_range,
        depot_positions=list(depot_positions),
    )
    return env, Task03Adapter(env=env, clear_initial_tasks=True, seed=7)


class RemoteDetectionDeliveryTests(unittest.TestCase):
    """A far-away PILOC detection must still reach the rescue team."""

    def test_remote_detection_reaches_a_rescue_robot_over_the_uplink(self) -> None:
        # Review repro: robot at (0, 0), target at (1, 1), comm_range 0.3. The robot
        # can never sense the target and no neighbour can gossip it; only the
        # uplink message closes the loop.
        env, adapter = wire_rescue_side(
            species=1, per_species=1, depot_positions=[(0.0, 0.0)]
        )
        registry = TaskRegistry()
        event = make_detection()
        registry.handle_detection(event)
        tasks = registry.active_snapshots()
        adapter.sync_tasks(tasks)

        relay = CommRelay(subscribers=("0",), writer=adapter, snapshot_of=registry.get, seed=7)
        adapter.comm_relay = relay
        relay.handle_detection(event)
        relay.tick()

        knowledge = env.agent_knowledge[0]
        internal_id = adapter.external_to_internal["remote-target"]
        self.assertIn(internal_id, knowledge["known_tasks"])
        self.assertAlmostEqual(
            float(knowledge["task_locations"][internal_id][0]), 1.0, places=9
        )

        # The coordinator may now bias the pair, and the allocator offers the task.
        agents = adapter.agent_snapshots()
        advice = HeuristicCoordinator().advise(CoordinatorInput(1.0, tasks, agents))
        self.assertIn(("0", "remote-target"), advice.pair_bias)
        commands = adapter.allocate(tasks, agents, advice, 1.0)
        self.assertEqual(commands[0].task_id, "remote-target")

        # Closed loop: the robot travels to the target and finishes the task.
        adapter.apply_assignments(commands)
        finished = False
        for tick in range(2, 60):
            agents = adapter.agent_snapshots()
            advice = HeuristicCoordinator().advise(CoordinatorInput(float(tick), tasks, agents))
            adapter.apply_assignments(
                adapter.allocate(tasks, agents, advice, float(tick))
            )
            adapter.step(float(tick), 1.0)
            if env.task_dic[internal_id]["finished"]:
                finished = True
                break
        self.assertTrue(finished, "the remotely discovered target was never completed")

    def test_a_robot_outside_the_uplink_topology_never_sees_or_picks_the_task(self) -> None:
        env, adapter = wire_rescue_side(
            species=2, per_species=1, depot_positions=[(0.1, 0.1), (0.9, 0.9)]
        )
        registry = TaskRegistry()
        event = make_detection()
        registry.handle_detection(event)
        tasks = registry.active_snapshots()
        adapter.sync_tasks(tasks)

        # Only robot 0 carries the uplink receiver; the teams are 1.13 apart, so
        # gossip cannot bridge the gap either.
        relay = CommRelay(subscribers=("0",), writer=adapter, snapshot_of=registry.get, seed=7)
        adapter.comm_relay = relay
        relay.handle_detection(event)
        relay.tick()

        snapshots = adapter.agent_snapshots()
        knowing = {
            snapshot.agent_id: snapshot.known_tasks for snapshot in snapshots
        }
        self.assertEqual(knowing["0"], ("remote-target",))
        self.assertEqual(knowing["1"], ())

        advice = HeuristicCoordinator().advise(CoordinatorInput(1.0, tasks, snapshots))
        self.assertNotIn(("1", "remote-target"), advice.pair_bias)
        commands = {
            command.agent_id: command.task_id
            for command in adapter.allocate(tasks, snapshots, advice, 1.0)
        }
        self.assertIsNone(commands["1"], "a robot that never heard of the task picked it")


class GossipHopTests(unittest.TestCase):
    """One gossip call propagates a task at most one hop."""

    def test_gossip_spreads_at_most_one_hop_per_tick(self) -> None:
        from integrated_sar.task03_runtime import build_task_env, load_task_env_module

        env = build_task_env(
            load_task_env_module(),
            seed=7,
            species=1,
            per_species=3,
            comm_range=0.3,
        )
        env.task_dic.clear()
        env.task_source.clear()
        env.next_dynamic_task_id = 0
        env.init_agent_memory()
        # Chain A(0) - B(1) - C(2); the task sits far from every robot so the
        # sensing part of propagation cannot shortcut the test.
        for agent_id, x in ((0, 0.0), (1, 0.2), (2, 0.4)):
            env.agent_dic[agent_id]["location"] = np.array([x, 0.0])
        env._spawn_new_task(0, location=np.array([0.9, 0.9]), source="tidal")
        env.update_communication_graph()
        self.assertEqual(env.comm_graph[0], [1])
        self.assertEqual(env.comm_graph[1], [0, 2])
        self.assertEqual(env.comm_graph[2], [1])

        env.learn_task_view(
            0,
            {
                "task_id": 0,
                "location": env.task_dic[0]["location"],
                "deadline": env.task_dic[0]["deadline"],
                "status": env.task_dic[0]["status"],
                "requirements": env.task_dic[0]["requirements"],
                "members": [],
                "update_time": 0.0,
                "dynamic": True,
            },
        )

        env.propagate_task_completion()
        self.assertIn(0, env.agent_knowledge[1]["known_tasks"], "B did not hear A")
        self.assertNotIn(
            0,
            env.agent_knowledge[2]["known_tasks"],
            "C heard the task in the same tick it reached B: gossip ran two hops",
        )

        env.propagate_task_completion()
        self.assertIn(0, env.agent_knowledge[2]["known_tasks"], "C never heard B")


class PacketLossTests(unittest.TestCase):
    """A dropped packet never changes anyone's belief."""

    def test_a_lost_packet_leaves_belief_untouched(self) -> None:
        env, adapter = wire_rescue_side(
            species=1, per_species=1, depot_positions=[(0.0, 0.0)]
        )
        registry = TaskRegistry()
        first = make_detection("first-target", event_id="detection:first")
        registry.handle_detection(first)
        adapter.sync_tasks(registry.active_snapshots())

        reliable = CommRelay(subscribers=("0",), writer=adapter, snapshot_of=registry.get, seed=7)
        adapter.comm_relay = reliable
        reliable.handle_detection(first)
        reliable.tick()

        knowledge = env.agent_knowledge[0]
        first_internal = adapter.external_to_internal["first-target"]
        belief_before = (
            set(knowledge["known_tasks"]),
            knowledge["task_current_status"][first_internal].copy(),
            knowledge["task_last_update_time"][first_internal],
        )

        # A second detection is published over a link that drops everything.
        lossy = CommRelay(
            subscribers=("0",),
            writer=adapter,
            snapshot_of=registry.get,
            delivery_probability=0.0,
            seed=7,
        )
        second = make_detection("second-target", event_id="detection:second")
        registry.handle_detection(second)
        adapter.sync_tasks(registry.active_snapshots())
        lossy.handle_detection(second)
        lossy.tick()

        belief_after = (
            set(knowledge["known_tasks"]),
            knowledge["task_current_status"][first_internal].copy(),
            knowledge["task_last_update_time"][first_internal],
        )
        self.assertEqual(belief_after[0], belief_before[0])
        np.testing.assert_array_equal(belief_after[1], belief_before[1])
        self.assertEqual(belief_after[2], belief_before[2])
        self.assertNotIn(
            adapter.external_to_internal.get("second-target"),
            knowledge["known_tasks"],
        )


class DelayAndDeterminismTests(unittest.TestCase):
    """Delay holds a message back for the configured number of ticks; seeds replay."""

    def test_delayed_messages_arrive_on_the_configured_tick(self) -> None:
        env, adapter = wire_rescue_side(
            species=1, per_species=1, depot_positions=[(0.0, 0.0)]
        )
        registry = TaskRegistry()
        event = make_detection()
        registry.handle_detection(event)
        adapter.sync_tasks(registry.active_snapshots())

        relay = CommRelay(
            subscribers=("0",), writer=adapter, snapshot_of=registry.get,
            delay_ticks=2, seed=7,
        )
        adapter.comm_relay = relay
        relay.handle_detection(event)

        relay.tick()
        self.assertEqual(env.agent_knowledge[0]["known_tasks"], set())
        relay.tick()
        self.assertEqual(env.agent_knowledge[0]["known_tasks"], set())
        relay.tick()
        internal_id = adapter.external_to_internal["remote-target"]
        self.assertIn(internal_id, env.agent_knowledge[0]["known_tasks"])

    def test_same_seed_replays_the_same_delivery_pattern(self) -> None:
        delivered: list[list[str]] = []
        for _ in range(2):
            env, adapter = wire_rescue_side(
                species=1, per_species=2, depot_positions=[(0.0, 0.0)]
            )
            registry = TaskRegistry()
            relay = CommRelay(
                subscribers=tuple(str(aid) for aid in env.agent_dic),
                writer=adapter,
                snapshot_of=registry.get,
                delivery_probability=0.5,
                seed=11,
            )
            for index in range(30):
                event = make_detection(f"target-{index}", event_id=f"detection:{index}")
                registry.handle_detection(event)
                adapter.sync_tasks(registry.active_snapshots())
                relay.handle_detection(event)
            delivered.append(sorted(relay.tick()))

        self.assertEqual(delivered[0], delivered[1])
        # A 50% link over 60 transmission attempts must both deliver and drop.
        self.assertTrue(delivered[0], "nothing was ever delivered")
        self.assertLess(len(delivered[0]), 60, "nothing was ever dropped")


class BeliefSchedulingTests(unittest.TestCase):
    """A first message arrival is an explicit scheduling signal (review finding 3)."""

    def test_first_message_arrival_coordinates_and_allocates_immediately(self) -> None:
        from integrated_sar.adapters.task03 import Task03Adapter
        from integrated_sar.clock import GlobalClock
        from integrated_sar.coordinator import HeuristicCoordinator
        from integrated_sar.contracts import SearchStepResult
        from integrated_sar.event_bus import InMemoryEventBus
        from integrated_sar.orchestrator import JointSAROrchestrator
        from integrated_sar.task03_runtime import build_task_env, load_task_env_module

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
        bus = InMemoryEventBus()
        event = make_detection()
        relay = CommRelay(
            subscribers=("0",), writer=adapter, snapshot_of=registry.get,
            delay_ticks=1, seed=7,
        )

        class _OneShotSearch:
            def __init__(self) -> None:
                self.emitted = False

            def step(self, current_time: float, dt: float) -> SearchStepResult:
                del current_time, dt
                if self.emitted:
                    return SearchStepResult()
                self.emitted = True
                return SearchStepResult((event,), {})

        orchestrator = JointSAROrchestrator(
            search=_OneShotSearch(),
            task_sink=adapter,
            agent_source=adapter,
            coordinator=HeuristicCoordinator(),
            allocator=adapter,
            executor=adapter,
            registry=registry,
            event_bus=bus,
            clock=GlobalClock(),
            dt=1.0,
            coordination_interval=5,
            allocation_interval=5,
        )
        bus.subscribe(type(event), relay.handle_detection)
        adapter.comm_relay = relay

        reports = list(orchestrator.run(6))

        # Tick 1: the task exists, but no robot has heard of it -> nobody may be
        # assigned, and no belief moved yet.
        self.assertEqual(reports[0].belief_changed_agents, ())
        self.assertEqual(
            [command.task_id for command in reports[0].assignments if command.task_id], []
        )
        # Tick 2: the delayed message arrives -> immediate coordination (fresh
        # advice, not the cached one) and exactly one allocation of the task,
        # without waiting for the five-step allocation interval.
        self.assertEqual(reports[1].belief_changed_agents, ("0",))
        self.assertEqual(reports[1].advice.generated_at, 2.0)
        self.assertEqual(
            [command.task_id for command in reports[1].assignments if command.task_id],
            ["remote-target"],
        )
        self.assertEqual(len(reports[1].assignments), 1)
        # Ticks 3-5: the same arrival must not re-trigger allocation every tick.
        for report in reports[2:5]:
            self.assertEqual(
                [command.task_id for command in report.assignments if command.task_id], []
            )


class RelayResetTests(unittest.TestCase):
    def test_reset_clears_outbox_dedup_and_reseeds(self) -> None:
        env, adapter = wire_rescue_side(
            species=1, per_species=1, depot_positions=[(0.0, 0.0)]
        )
        registry = TaskRegistry()
        event = make_detection()
        registry.handle_detection(event)
        adapter.sync_tasks(registry.active_snapshots())

        relay = CommRelay(subscribers=("0",), writer=adapter, snapshot_of=registry.get, seed=7)
        relay.handle_detection(event)
        relay.reset()
        self.assertEqual(relay.queued, 0)
        relay.tick()
        self.assertEqual(env.agent_knowledge[0]["known_tasks"], set())

        # Dedup state was cleared too, so the same detection can be re-delivered.
        relay.handle_detection(event)
        relay.tick()
        internal_id = adapter.external_to_internal["remote-target"]
        self.assertIn(internal_id, env.agent_knowledge[0]["known_tasks"])


if __name__ == "__main__":
    unittest.main()
