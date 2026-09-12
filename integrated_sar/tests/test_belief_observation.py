"""CTDE observations and hard masks must be built entirely from per-agent belief.

Regression for review finding 2: ``get_local_obs`` and ``get_hard_mask`` used to
read ``task_dic`` truth (stage index, future requirements, scene complexity and
several fallbacks), so advancing the truth of a task a robot had not heard about
changed that robot's decision inputs. Invariant 9: only a message that actually
arrives may update them.
"""

from __future__ import annotations

import unittest

import numpy as np

from integrated_sar.adapters.task03 import Task03Adapter
from integrated_sar.comm import CommRelay
from integrated_sar.contracts import (
    DetectionEvent,
    GridPosition,
    TargetProfile,
    TaskStageSpec,
    WorldPosition,
)
from integrated_sar.task03_runtime import build_task_env, load_task_env_module
from integrated_sar.task_registry import TaskRegistry


def _wire(abilities):
    """One robot staged at (0, 0), far from every task, with fixed abilities."""
    env = build_task_env(
        load_task_env_module(),
        seed=7,
        species=1,
        per_species=1,
        comm_range=0.3,
        depot_positions=[(0.0, 0.0)],
    )
    adapter = Task03Adapter(env=env, clear_initial_tasks=True, seed=7)
    env.agent_dic[0]["abilities"] = np.asarray(abilities, dtype=int)
    return env, adapter


def _two_stage_detection(target_id="two-stage"):
    return DetectionEvent(
        event_id=f"detection:{target_id}",
        target_id=target_id,
        grid_position=GridPosition(59, 59),
        world_position=WorldPosition(1.0, 1.0),
        detected_at=1.0,
        discovered_by="agent_0",
        confidence=1.0,
        profile=TargetProfile(
            stages=(
                TaskStageSpec((0, 1, 0, 0, 0), 3.0),
                TaskStageSpec((1, 0, 0, 0, 0), 3.0),
            ),
            deadline_offset=500.0,
            severity=0.7,
        ),
    )


def _register_and_deliver(env, adapter, event):
    """Publish the detection through the comm bridge so robot 0 believes it."""
    registry = TaskRegistry()
    registry.handle_detection(event)
    adapter.sync_tasks(registry.active_snapshots())
    relay = CommRelay(subscribers=("0",), writer=adapter, snapshot_of=registry.get, seed=7)
    adapter.comm_relay = relay
    relay.handle_detection(event)
    relay.tick()
    return registry


class BeliefObservationTests(unittest.TestCase):
    def test_truth_advances_do_not_change_a_disconnected_robot(self) -> None:
        env, adapter = _wire(abilities=(1, 0, 0, 0, 0))
        _register_and_deliver(env, adapter, _two_stage_detection())
        internal_id = adapter.external_to_internal["two-stage"]

        obs_before = env.get_local_obs(0)
        mask_before = env.get_hard_mask(0)

        # Another team completes stage 0 while this robot is disconnected; the
        # belief timestamp must not move and neither may the robot's inputs.
        task = env.task_dic[internal_id]
        task["current_stage_index"] = 1
        task["status"] = task["stages"][1]["requirements"].copy()
        task["requirements"] = task["stages"][1]["requirements"].copy()

        obs_after = env.get_local_obs(0)
        mask_after = env.get_hard_mask(0)
        self.assertEqual(
            obs_before[2].tobytes(), obs_after[2].tobytes(),
            "the actor observation changed although no message arrived",
        )
        self.assertEqual(
            obs_before[0].tobytes(), obs_after[0].tobytes(),
            "the self state changed although no message arrived",
        )
        self.assertEqual(
            mask_before.tobytes(), mask_after.tobytes(),
            "the hard mask changed although no message arrived",
        )
        self.assertEqual(
            float(env.agent_knowledge[0]["task_last_update_time"][internal_id]), 1.0
        )

    def test_a_new_belief_message_updates_observation_and_mask(self) -> None:
        env, adapter = _wire(abilities=(1, 0, 0, 0, 0))
        _register_and_deliver(env, adapter, _two_stage_detection())
        internal_id = adapter.external_to_internal["two-stage"]

        obs_before = env.get_local_obs(0)
        mask_before = env.get_hard_mask(0)

        # Truth advances (as if another team finished stage 0)...
        task = env.task_dic[internal_id]
        task["current_stage_index"] = 1
        task["status"] = task["stages"][1]["requirements"].copy()
        task["requirements"] = task["stages"][1]["requirements"].copy()
        # ...and this time one fresh belief update is propagated to the robot.
        env.refresh_task_belief(0, internal_id, update_time=2.0)

        obs_after = env.get_local_obs(0)
        mask_after = env.get_hard_mask(0)
        # The believed stage moved from 0 to 1, so the believed remaining status
        # goes from (0,1,0,0,0) to (1,0,0,0,0): the observation feature moves.
        self.assertNotEqual(
            obs_before[2].tobytes(), obs_after[2].tobytes(),
            "a delivered belief update did not reach the actor observation",
        )
        self.assertEqual(
            mask_before.tobytes(), mask_after.tobytes(),
            "this robot can contribute to both stages, so its mask should not move",
        )

    def test_mask_follows_believed_future_requirements_only(self) -> None:
        # Stage 0 needs dim 1 (robot cannot contribute), stage 1 needs dim 0 (the
        # robot could). With a stale belief the mask must be decided by the
        # BELIEVED future requirement, not by the truth's.
        env, adapter = _wire(abilities=(1, 0, 0, 0, 0))
        _register_and_deliver(env, adapter, _two_stage_detection())
        internal_id = adapter.external_to_internal["two-stage"]
        mask = env.get_hard_mask(0)
        self.assertFalse(bool(mask[1]), "the believed future stage made the task look useless")

        # Truth says stage 0 is done and the robot can contribute to stage 1; the
        # robot has not heard that, so its mask must not change.
        task = env.task_dic[internal_id]
        task["current_stage_index"] = 1
        task["status"] = task["stages"][1]["requirements"].copy()
        task["requirements"] = task["stages"][1]["requirements"].copy()
        self.assertFalse(
            bool(env.get_hard_mask(0)[1]),
            "truth advanced but the mask changed without a message",
        )

        # Once the belief catches up the mask is recomputed from the new stage.
        env.refresh_task_belief(0, internal_id, update_time=2.0)
        self.assertFalse(bool(env.get_hard_mask(0)[1]))

    def test_belief_carries_stage_and_complexity_into_the_features(self) -> None:
        env, adapter = _wire(abilities=(1, 1, 1, 1, 1))
        _register_and_deliver(env, adapter, _two_stage_detection())
        internal_id = adapter.external_to_internal["two-stage"]

        knowledge = env.agent_knowledge[0]
        self.assertEqual(int(knowledge["task_stage_index"][internal_id]), 0)
        np.testing.assert_array_equal(
            knowledge["task_future_requirements"][internal_id], (1, 0, 0, 0, 0)
        )
        self.assertAlmostEqual(float(knowledge["task_scene_complexity"][internal_id]), 0.7)

        _, _, tasks_obs, _, action_map = env.get_local_obs(0)
        index = next(
            i for i, (kind, tid) in enumerate(action_map)
            if kind == "task" and tid == internal_id
        )
        # Feature layout: believed status(5) future(5) location(2) urgency(1)
        # complexity(1) members(1) | info(5).
        np.testing.assert_allclose(tasks_obs[index, 0:5], (0, 1, 0, 0, 0))
        np.testing.assert_allclose(tasks_obs[index, 5:10], (1, 0, 0, 0, 0))
        self.assertAlmostEqual(float(tasks_obs[index, 13]), 0.7, places=6)


if __name__ == "__main__":
    unittest.main()
