"""No online decision path may read registry truth instead of a robot's belief.

Third review round, finding 2: ``get_local_obs`` and ``get_hard_mask`` were
already belief-based, but the coordinator still scored pairs from the registry's
task snapshots (position, remaining requirements), ``apply_action`` navigated to
the truth position, and the en-route replan check used the truth status. A robot
with a stale belief therefore received task updates it never heard.
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
    def test_pair_bias_ignores_registry_only_position_updates(self) -> None:
        env, adapter, registry = _wire()
        agents = adapter.agent_snapshots()
        before = HeuristicCoordinator().advise(
            CoordinatorInput(2.0, registry.active_snapshots(), agents)
        )

        # Move only the registry truth; the robot received nothing.
        registry._tasks["t1"].position = WorldPosition(*TRUTH_ONLY_POSITION)
        after = HeuristicCoordinator().advise(
            CoordinatorInput(2.0, registry.active_snapshots(), agents)
        )

        self.assertAlmostEqual(before.pair_bias[("0", "t1")], after.pair_bias[("0", "t1")])
        self.assertAlmostEqual(before.task_bias["t1"], after.task_bias["t1"])
        self.assertAlmostEqual(
            float(env.agent_knowledge[0]["task_locations"][adapter.external_to_internal["t1"]][0]),
            BELIEVED_POSITION[0],
        )

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

    def test_stale_belief_does_not_inherit_a_fresher_task_pool_priority(self) -> None:
        # A second, better-informed robot learns the deadline changed; the stale
        # robot must keep its own believed priority, not the other robot's.
        task_pool = (
            AgentSnapshot(
                "informed",
                WorldPosition(0.9, 0.9),
                (1.0, 0, 0, 0, 0),
                known_tasks=("t1",),
                task_beliefs=(
                    AgentTaskBelief("t1", WorldPosition(0.9, 0.9), (1, 0, 0, 0, 0), 1000.0, 0, 0.0, 0.5),
                ),
            ),
        )
        advice = HeuristicCoordinator().advise(
            CoordinatorInput(
                10.0,
                (),
                task_pool
                + (
                    AgentSnapshot("stale", WorldPosition(0.0, 0.0), (1.0, 0, 0, 0, 0),
                                  known_tasks=("t1",), task_beliefs=()),
                ),
            )
        )
        # No active registry task -> no bias at all, even though a belief exists.
        self.assertEqual(advice.task_bias, {})
        self.assertEqual(advice.pair_bias, {})

        # With the registry agreeing the task is active, only the believing robot
        # is scored: the stale one has no belief to score from.
        from integrated_sar.contracts import TaskSnapshot, TaskStatus

        active = TaskSnapshot(
            task_id="t1", source_event_id="d1", position=WorldPosition(0.9, 0.9),
            stages=(TaskStageSpec((1, 0, 0, 0, 0), 3.0),), current_stage=0,
            remaining_requirements=(1, 0, 0, 0, 0), progress=0.0, status=TaskStatus.DETECTED,
            deadline=1000.0, confidence=1.0, severity=0.5, discovered_at=1.0, updated_at=1.0,
            revision=1,
        )
        advice = HeuristicCoordinator().advise(
            CoordinatorInput(
                10.0,
                (active,),
                task_pool
                + (
                    AgentSnapshot("stale", WorldPosition(0.0, 0.0), (1.0, 0, 0, 0, 0),
                                  known_tasks=("t1",), task_beliefs=()),
                ),
            )
        )
        self.assertEqual(sorted(agent for agent, _ in advice.pair_bias), ["informed"])


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


if __name__ == "__main__":
    unittest.main()
