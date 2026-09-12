"""A public reset must restore the whole joint episode, not just the mappings.

Regression for review finding 4: Task03Adapter.reset only cleared its mappings
and the task dictionary, so the TaskEnv clock kept running, robots stayed
working/moving with stale target ids (a following step raised KeyError), and
the mock search environment kept its explored world.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from typing import Any

import numpy as np

from integrated_sar.joint_driver import RunConfiguration, build_joint_run
from integrated_sar.search_runtime import build_mock_search_side


def make_configuration(directory: str, **overrides: Any) -> RunConfiguration:
    settings: dict[str, Any] = {
        "mock": True,
        "steps": 8,
        "seed": 42,
        "quiet": True,
        "output": os.path.join(directory, "lifecycle.jsonl"),
    }
    settings.update(overrides)
    return RunConfiguration(**settings)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize_ids(record: dict[str, Any]) -> dict[str, Any]:
    """Blank out episode-numbered ids so two episodes can be compared."""
    normalized = json.loads(json.dumps(record))
    for key in ("target_id", "source_event_id"):
        if isinstance(normalized.get(key), str):
            normalized[key] = re.sub(r"\d+", "N", normalized[key])
    detail = normalized.get("detail") or {}
    for key in ("progress_event_id", "command_id"):
        if isinstance(detail.get(key), str):
            detail[key] = re.sub(r"\d+", "N", detail[key])
    return normalized


class FullResetTests(unittest.TestCase):
    def test_reset_restores_clocks_scene_and_rescue_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = make_configuration(directory)
            run = build_joint_run(configuration)
            try:
                run.run()
                scenario = run.scenario
                radius = configuration.detection_radius

                run.reset()

                # Three clocks back to zero.
                self.assertEqual(run.orchestrator.current_time, 0.0)
                self.assertEqual(run.search_adapter.env.current_time, 0.0)
                self.assertEqual(run.search_adapter.env.steps, 0)
                self.assertEqual(run.task_adapter.env.current_time, 0.0)
                # The registry is empty again.
                self.assertEqual(len(run.registry), 0)

                # The search world matches a freshly built episode-start scenario.
                fresh_env, fresh_obs, _ = build_mock_search_side(
                    scenario, view_radius=radius, dt=configuration.dt
                )
                search_env = run.search_adapter.env
                np.testing.assert_array_equal(search_env.agents_pos, fresh_env.agents_pos)
                np.testing.assert_array_equal(search_env.goals_pos, fresh_env.goals_pos)
                np.testing.assert_array_equal(
                    search_env.global_pheromone, fresh_env.global_pheromone
                )
                for agent_id, observation in fresh_obs.items():
                    np.testing.assert_array_equal(
                        run.search_adapter.obs[agent_id], observation,
                        f"{agent_id}'s timestamp map was not restored",
                    )

                # The rescue side is empty and idle again, staged on its depots.
                task_env = run.task_adapter.env
                self.assertEqual(task_env.task_dic, {})
                self.assertEqual(task_env.agent_knowledge[0]["known_tasks"], set())
                for species_id, depot in task_env.depot_dic.items():
                    for member in depot["members"]:
                        agent = task_env.agent_dic[member]
                        self.assertEqual(agent["status"], "idle")
                        self.assertIsNone(agent["target_id"])
                        np.testing.assert_allclose(agent["location"], depot["location"])
                        self.assertEqual(agent["failed_tasks"], [])
                        self.assertEqual(agent["cooldown_tasks"], {})

                # A second episode runs cleanly and discovers targets again.
                summary = run.run()
            finally:
                run.logger.close()

        self.assertGreater(summary["targets_discovered"], 0)
        self.assertGreater(summary["assignment_commands"], 0)

    def test_reset_with_a_working_robot_never_crashes_the_next_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = make_configuration(directory)
            run = build_joint_run(configuration)
            try:
                run.run()
                task_env = run.task_adapter.env
                busy = any(
                    agent["status"] in ("moving", "working")
                    for agent in task_env.agent_dic.values()
                )
                self.assertTrue(busy, "the scenario produced no moving/working robot")

                run.reset()
                self.assertTrue(
                    all(
                        agent["target_id"] is None
                        for agent in task_env.agent_dic.values()
                    ),
                    "a robot kept a stale target id across the reset",
                )
                # The old failure was a KeyError when a robot arrived on a task
                # that the reset had deleted.
                run.run()
            finally:
                run.logger.close()

    def test_reset_then_rerun_replays_the_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = make_configuration(directory)
            run = build_joint_run(configuration)
            try:
                run.run()
                first_episode = read_jsonl(configuration.output)
                run.reset()
                run.run()
                records = read_jsonl(configuration.output)
            finally:
                run.logger.close()

        starts = [
            index
            for index, record in enumerate(records)
            if record["event_type"] == "run_config"
        ]
        self.assertEqual(len(starts), 2)
        second_episode = records[starts[1]:]
        self.assertEqual(
            [_normalize_ids(record) for record in first_episode],
            [_normalize_ids(record) for record in second_episode],
            "the second episode did not replay the first episode's trajectory",
        )

    def test_event_ids_do_not_collide_across_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = make_configuration(directory)
            run = build_joint_run(configuration)
            try:
                run.run()
                first = read_jsonl(configuration.output)
                run.reset()
                run.run()
                records = read_jsonl(configuration.output)
            finally:
                run.logger.close()

        def episode_event_ids(items):
            ids = set()
            for record in items:
                if record.get("source_event_id"):
                    ids.add(record["source_event_id"])
                detail_id = (record.get("detail") or {}).get("progress_event_id")
                if detail_id:
                    ids.add(detail_id)
            return ids

        starts = [
            index
            for index, record in enumerate(records)
            if record["event_type"] == "run_config"
        ]
        first_ids = episode_event_ids(first)
        second_ids = episode_event_ids(records[starts[1]:])
        self.assertTrue(first_ids and second_ids)
        self.assertEqual(
            first_ids & second_ids,
            set(),
            "the two episodes reused an event id",
        )


if __name__ == "__main__":
    unittest.main()
