"""Every decision-relevant belief change must be reported as a scheduling signal.

Third review round, finding 3: the change fingerprint only compared status,
members, stage and the finished flag, so a delivered update that moved a task's
position, deadline, future requirements, scene complexity or dynamic/assigned
flags produced an empty ``belief_changed_agents`` and allocation waited for the
periodic interval instead of firing on arrival.
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
        world_position=WorldPosition(0.9, 0.9),
        detected_at=1.0,
        discovered_by="a0",
        confidence=1.0,
        profile=TargetProfile(
            stages=(
                TaskStageSpec((1, 0, 0, 0, 0), 3.0),
                TaskStageSpec((0, 1, 0, 0, 0), 3.0),
            ),
            deadline_offset=500.0,
            severity=0.7,
        ),
    )
    registry.handle_detection(event)
    adapter.sync_tasks(registry.active_snapshots())
    relay = CommRelay(subscribers=("0",), writer=adapter, snapshot_of=registry.get, seed=7)
    adapter.comm_relay = relay
    relay.handle_detection(event)
    relay.tick()
    return env, adapter


class BeliefChangeDetectionTests(unittest.TestCase):
    """Each mutation below reaches an observation, a mask or an advice input."""

    def _mutations(self, env, internal_id):
        knowledge = env.agent_knowledge[0]
        return {
            "position": lambda: knowledge["task_locations"].__setitem__(
                internal_id, np.array([0.8, 0.9])
            ),
            "deadline": lambda: knowledge["task_deadlines"].__setitem__(internal_id, 777.0),
            "remaining status": lambda: knowledge["task_current_status"].__setitem__(
                internal_id, np.array([0, 1, 0, 0, 0])
            ),
            "requirements": lambda: knowledge["task_requirements"].__setitem__(
                internal_id, np.array([0, 1, 0, 0, 0])
            ),
            "stage index": lambda: knowledge["task_stage_index"].__setitem__(internal_id, 1),
            "future requirements": lambda: knowledge["task_future_requirements"].__setitem__(
                internal_id, np.array([0, 0, 1, 0, 0])
            ),
            "scene complexity": lambda: knowledge["task_scene_complexity"].__setitem__(
                internal_id, 0.2
            ),
            "members": lambda: knowledge["task_members"].__setitem__(internal_id, [1]),
            "progress": lambda: knowledge["task_progress"].__setitem__(internal_id, 0.5),
            "dynamic flag": lambda: knowledge["dynamic_tasks"].discard(internal_id),
            "assigned flag": lambda: knowledge["known_assigned"].add(internal_id),
            "finished flag": lambda: knowledge["known_finished"].add(internal_id),
        }

    def test_every_decision_relevant_field_change_is_detected(self) -> None:
        for name, mutate in self._mutations(*_wire()).items():
            with self.subTest(field=name):
                env, adapter = _wire()
                internal_id = adapter.external_to_internal["t1"]
                before = adapter._belief_fingerprints()
                self._mutations(env, internal_id)[name]()
                after = adapter._belief_fingerprints()
                changed = sorted(
                    agent
                    for agent in set(before) | set(after)
                    if before.get(agent) != after.get(agent)
                )
                self.assertEqual(
                    changed, [0], f"a change to {name} was not reported as a belief revision"
                )

    def test_a_timestamp_only_refresh_is_not_a_revision(self) -> None:
        # Standing next to an idle task refreshes timestamps without learning
        # anything; that must not look like new knowledge.
        env, adapter = _wire()
        internal_id = adapter.external_to_internal["t1"]
        before = adapter._belief_fingerprints()
        env.agent_knowledge[0]["task_last_update_time"][internal_id] = 99.0
        after = adapter._belief_fingerprints()
        self.assertEqual(before, after)

    def test_a_delivered_position_update_reports_the_agent(self) -> None:
        env, adapter = _wire()
        internal_id = adapter.external_to_internal["t1"]
        task = env.task_dic[internal_id]
        task["location"] = np.array([0.2, 0.2])
        # Truth moved, the belief has not: nothing is reported yet.
        self.assertAlmostEqual(
            float(env.agent_knowledge[0]["task_locations"][internal_id][0]), 0.9
        )
        before = adapter._belief_fingerprints()
        self.assertEqual(before, adapter._belief_fingerprints())

        # The sensing/arrival channel delivers the new position.
        env.refresh_task_belief(0, internal_id, update_time=5.0)
        self.assertAlmostEqual(
            float(env.agent_knowledge[0]["task_locations"][internal_id][0]), 0.2
        )
        self.assertNotEqual(before, adapter._belief_fingerprints())


if __name__ == "__main__":
    unittest.main()
