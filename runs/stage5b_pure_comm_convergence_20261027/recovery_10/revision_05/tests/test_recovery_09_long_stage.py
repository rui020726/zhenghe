"""Pure tests for the one-stage 5000-completed-episode continuation."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from stage5b_comm import long_stage as L
from stage5b_comm import recovery_descendant as D
from stage5b_comm import recovery_runtime as RT

ROSTER = L.ROSTER
POST_R8 = {
    ("T1", 91001): (2164, 2160),
    ("T1", 91002): (2160, 2160),
    ("T1", 91003): (1800, 1800),
    ("T2", 91001): (1800, 1800),
    ("T2", 91002): (1800, 1800),
    ("T2", 91003): (1440, 1440),
}


class LongStagePlanTests(unittest.TestCase):
    def test_target_uses_completed_episodes_not_cursor_or_updates(self):
        states = {key: {"cursor": cursor, "completed": completed}
                  for key, (cursor, completed) in POST_R8.items()}
        plan = L.plan_roster(states)
        self.assertEqual(
            [(plan[key]["new_world_cap"], plan[key]["end_episode_cap"])
             for key in ROSTER],
            [(2840, 5004), (2840, 5000), (3200, 5000),
             (3200, 5000), (3200, 5000), (3560, 5000)])
        self.assertEqual(sum(x["new_world_cap"] for x in plan.values()), 18840)
        self.assertEqual(plan[("T1", 91001)]["registered_missing_episode_count_before_stage"], 4)

    def test_checkpoint_schedule_is_every_120_actual_episodes_plus_target(self):
        schedule = L.checkpoint_schedule(2160)
        self.assertEqual(schedule[:2], [2280, 2400])
        self.assertEqual(schedule[-2:], [4920, 5000])
        self.assertEqual(len(schedule), 24)
        self.assertTrue(all((value - 2160) % 120 == 0 or value == 5000
                            for value in schedule))
        self.assertEqual(L.checkpoint_schedule(1800)[0], 1920)

    def test_interrupted_checkpoint_resumes_without_cursor_confusion(self):
        plan = L.plan_run(2164, 2160)
        resumed = L.resume_from_stage_checkpoint(plan, 2284)
        self.assertEqual(resumed["resume_completed_episode_count"], 2280)
        self.assertEqual(resumed["resume_episode_cursor"], 2284)
        self.assertEqual(resumed["remaining_episodes_to_target"], 2720)
        self.assertEqual(resumed["end_episode_cap"], 5004)
        with self.assertRaisesRegex(ValueError, "120-episode stage interval"):
            L.resume_from_stage_checkpoint(plan, 2280)

    def test_next_runs_keeps_ppo_update_cursor_separate(self):
        runs = [{"arm": arm, "training_seed": seed} for arm, seed in ROSTER]
        states = {}
        for key in ROSTER:
            cursor, completed = POST_R8[key]
            states[key] = {
                "cursor": cursor, "next_update": cursor // 4,
                "certified": False, "loss_stability_stable": False,
                "donor": {"synthetic": True},
                "completed_episode_count": completed,
                "registered_missing_episode_count": cursor - completed,
            }
        result = D.next_runs(
            runs, states, include_completion_counts=True,
            include_minimum_target=True, target_completed_episodes=5000)
        self.assertEqual(result[0]["next_update"], 541)
        self.assertEqual(result[-1]["next_update"], 360)
        self.assertTrue(all(item["target_basis"] == "ACTUAL_COMPLETED_EPISODE_COUNT"
                            for item in result))
        self.assertTrue(all(item["checkpoint_interval_episodes"] == 120
                            for item in result))

    def test_long_stage_seed_plan_is_not_limited_to_720_worlds(self):
        runs = []
        for arm, seed in ROSTER:
            cursor, completed = POST_R8[(arm, seed)]
            runs.append({
                "arm": arm, "training_seed": seed,
                "start_episode": cursor,
                "end_episode_cap": cursor + 5000 - completed,
                "new_world_cap": 5000 - completed, "active": True,
                "target_completed_episodes": 5000,
                "target_basis": "ACTUAL_COMPLETED_EPISODE_COUNT",
                "checkpoint_interval_episodes": 120,
                "projected_completed_episode_count_after_stage": 5000,
            })
        plan = D.seed_plan(runs, {("__sentinel__", 0, 0, 0)})
        self.assertEqual(len(plan), 18840)
        self.assertEqual(plan[0]["episode_index"], 2164)
        self.assertEqual(plan[-1]["episode_index"], 4999)

    def test_task_certificate_continues_only_in_completed_target_mode(self):
        self.assertTrue(L.certificate_stop_required(None, 1560, 1560))
        self.assertFalse(L.certificate_stop_required(5000, 1560, 1560))
        self.assertFalse(L.certificate_stop_required(None, 1560, 2040))
        with self.assertRaisesRegex(ValueError, "completed episode"):
            L.certificate_stop_required(None, 1560, True)

    def test_long_target_payload_and_world_scope_are_explicit(self):
        ordinary = SimpleNamespace(payload={
            "target_completed_episodes": None,
            "checkpoint_interval_episodes": None,
        })
        long = SimpleNamespace(payload={
            "target_completed_episodes": 5000,
            "checkpoint_interval_episodes": 120,
        })
        self.assertIsNone(RT.completed_episode_target(ordinary))
        self.assertEqual(RT.completed_episode_target(long), 5000)
        self.assertEqual(
            RT.expected_world_authorization_scope(ordinary),
            "FINITE_RECOVERY_STAGE_MAX_720_WORLDS_ONE_ROOT_ONLY")
        self.assertEqual(
            RT.expected_world_authorization_scope(long),
            "LONG_RECOVERY_STAGE_TO_COMPLETED_EPISODE_TARGET_ONE_ROOT_ONLY")
        with self.assertRaisesRegex(ValueError, "120-episode"):
            L.completed_target_from_payload({
                "target_completed_episodes": 5000,
                "checkpoint_interval_episodes": 60,
            })
        with self.assertRaisesRegex(ValueError, "requires"):
            L.completed_target_from_payload({
                "target_completed_episodes": None,
                "checkpoint_interval_episodes": 120,
            })

    def test_runtime_run_spec_accepts_long_interval(self):
        runs = []
        for arm, seed in ROSTER:
            cursor, completed = POST_R8[(arm, seed)]
            remaining = 5000 - completed
            runs.append({"arm": arm, "training_seed": seed,
                         "start_episode": cursor, "end_episode_cap": cursor + remaining,
                         "new_world_cap": remaining})
        context = SimpleNamespace(runs=runs)
        self.assertEqual(RT.run_spec(context, "T1", 91001), (2164, 5004))
        self.assertEqual(RT.run_spec(context, "T2", 91003), (1440, 5000))


if __name__ == "__main__":
    unittest.main()