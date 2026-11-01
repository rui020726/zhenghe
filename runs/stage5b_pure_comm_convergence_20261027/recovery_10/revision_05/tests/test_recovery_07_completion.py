"""Focused pure-file tests for recovery_07's completed R6 continuation."""
import unittest
from pathlib import Path

from stage5b_comm import recovery_descendant as D
from stage5b_comm import recovery_protocol as R
from stage5b_comm import recovery_runtime as RT
from stage5b_comm.paths import Layout

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path(r"D:\code\整合\zhenghe-round42\runs\stage5b_pure_comm_convergence_20261027\recovery_06\revision_02")


class Recovery07CompletedParentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = D.build_next_stage(PARENT, ROOT)
        cls.context = RT.strict_json_bytes(
            cls.bundle["files"][RT.CONTEXT_RELATIVE], "recovery_07 context")
        cls.admission = RT.strict_json_bytes(
            cls.bundle["files"][D.ADMISSION], "recovery_07 admission")
        cls.seeds = RT.strict_json_bytes(
            cls.bundle["files"]["protocol/RECOVERY_NEXT_SEEDS.json"],
            "recovery_07 seeds")

    def test_real_r6_parent_generates_exact_six_intervals(self):
        self.assertEqual(self.bundle["world_cap"], 720)
        self.assertEqual(
            [(r["arm"], r["training_seed"], r["start_episode"], r["end_episode_cap"])
             for r in self.bundle["runs"]],
            [("T1", 91001, 1924, 2044), ("T1", 91002, 1920, 2040),
             ("T1", 91003, 1560, 1680), ("T2", 91001, 1560, 1680),
             ("T2", 91002, 1560, 1680), ("T2", 91003, 1200, 1320)])
        self.assertTrue(all(r["new_world_cap"] == 120 and r["active"] is True
                            for r in self.bundle["runs"]))

    def test_episode_cursor_and_actual_completed_count_are_distinct(self):
        self.assertEqual(
            [(r["episode_cursor_before_stage"],
              r["completed_episode_count_before_stage"],
              r["registered_missing_episode_count_before_stage"])
             for r in self.bundle["runs"]],
            [(1924, 1920, 4), (1920, 1920, 0), (1560, 1560, 0),
             (1560, 1560, 0), (1560, 1560, 0), (1200, 1200, 0)])

    def test_registered_gaps_remain_unknown_and_never_retired(self):
        for arm, seed in (("T1", 91003), ("T2", 91001), ("T2", 91002)):
            self.assertGreaterEqual(len(R.known_gap(arm, seed)["updates"]), 2)
        for arm, seed in (("T1", 91003), ("T2", 91001), ("T2", 91002)):
            self.assertEqual(R.known_gap(arm, seed),
                             {"episodes": (600, 960), "updates": (150, 240)})
        self.assertIsNone(R.known_gap("T1", 91002))
        self.assertIsNone(R.known_gap("T2", 91003))
        self.assertEqual(R.known_gap("T1", 91001),
                         {"episodes": (1800, 1804), "updates": (450, 451)})
        self.assertEqual(self.admission["unknown_unfilled_updates"], 271)
        self.assertEqual(self.admission["world_cap"], 720)
        self.assertTrue(all(r["task_certified"] is False for r in self.bundle["runs"]))

    def test_frozen_context_reload_preserves_inherited_r5_gap(self):
        context = D.load_context(
            Layout(ROOT), ROOT / "protocol" / "RECOVERY_RUNTIME_CONTEXT.json")
        self.assertEqual(context.stage_index, 6)
        self.assertEqual(context.world_cap, 720)
        self.assertEqual(tuple(context.runs), tuple(self.bundle["runs"]))
        self.assertIsNone(context.payload["g3_report"])
        self.assertIsNone(context.payload["world_authorization"])
        self.assertEqual(self.admission["unknown_unfilled_updates"], 271)

    def test_seed_plan_is_exact_720_and_context_has_no_execution_approval(self):
        self.assertEqual(len(self.seeds), 720)
        keys = [(x["arm"], x["training_seed"], x["scene_seed"], x["action_rng_seed"])
                for x in self.seeds]
        self.assertEqual(len(set(keys)), 720)
        self.assertEqual(self.bundle["authorization"],
                         "G3_NOT_RUN_WORLD_NOT_AUTHORIZED")
        self.assertIsNone(self.context["g3_report"])
        self.assertIsNone(self.context["world_authorization"])
        self.assertEqual(self.context["stage_index"], 6)
        self.assertEqual(self.context["namespace"], R.WORLD_SEED_NAMESPACE)


if __name__ == "__main__":
    unittest.main()