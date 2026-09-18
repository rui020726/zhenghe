"""Member-roster consistency in the rescue environment.

The roster is how the environment decides which abilities a stage sees, so a robot that has
been re-tasked away must stop appearing on the task it left -- otherwise that task keeps
being covered, and keeps spending its work clock, on a body that is not there. These tests
assert the consequences (coverage status, the work clock, who is released when a task
closes), never the roster list on its own, and they pin the default-off path so every number
produced before this switch existed stays reproducible.

    D:/project_env/python.exe -m unittest integrated_sar.tests.test_task_membership_consistency
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from integrated_sar.task03_runtime import build_task_env, load_task_env_module

A, B = 0, 1
A_NEED = (0, 0, 0, 0, 1)   # only robot 0 carries dim 4
B_NEED = (1, 0, 0, 0, 0)   # only robot 0 carries dim 0
STAGE_TIME = 60.0
EPS = 1e-9


def scenario(membership_consistency: bool):
    env = build_task_env(
        load_task_env_module(),
        seed=11,
        species=2,
        per_species=1,
        dt=1.0,
        depot_positions=[(0.5, 0.5), (0.5, 0.5)],
        membership_consistency=membership_consistency,
    )
    env.reset(seed=11)
    for tid, location, need in ((A, (0.60, 0.50), A_NEED), (B, (0.60, 0.90), B_NEED)):
        env._spawn_new_task(tid, location=np.array(location, dtype=float))
        task = env.task_dic[tid]
        requirement = np.array(need, dtype=int)
        task["stages"] = [
            {
                "stage_id": 0,
                "requirements": requirement.copy(),
                "status": requirement.copy(),
                "time": STAGE_TIME,
                "time_left": STAGE_TIME,
                "finished": False,
            }
        ]
        task["total_stages"] = 1
        task["current_stage_index"] = 0
        task["requirements"] = requirement.copy()
        task["status"] = requirement.copy()
        task["members"] = []
        task["deadline"] = 400.0
    env.agent_dic[0]["location"] = np.array([0.55, 0.50])
    env.agent_dic[1]["location"] = np.array([0.55, 0.50])
    return env


ACTION_MAP = [("depot", None), ("task", A), ("task", B)]
INDEX_OF = {"depot": 0, A: 1, B: 2}


def stage(task):
    return task["stages"][min(int(task["current_stage_index"]), len(task["stages"]) - 1)]


def arrive(env, agent_id, tid, limit=12):
    env.apply_action(agent_id, INDEX_OF[tid], ACTION_MAP)
    for _ in range(limit):
        env.step()
        if env.agent_dic[agent_id].get("status") == "working":
            return True
    return False


class DefaultOffUnchangedTests(unittest.TestCase):
    """With the switch off, the old bookkeeping is exactly reproduced."""

    def test_a_reassigned_robot_still_credits_the_task_it_left(self) -> None:
        env = scenario(membership_consistency=False)
        self.assertTrue(arrive(env, 0, A))
        covered = stage(env.task_dic[A])["time_left"]
        env.apply_action(0, INDEX_OF[B], ACTION_MAP)
        env.step()
        self.assertIn(0, env.task_dic[A]["members"], "the stale roster entry should remain off")
        # The consequence, not just the list: the task still reads as covered.
        self.assertEqual(int(np.asarray(env.task_dic[A]["status"], dtype=int).sum()), 0)
        self.assertLess(float(stage(env.task_dic[A])["time_left"]), covered)


class RosterConsistencyTests(unittest.TestCase):
    def test_an_accepted_reassignment_stops_the_old_task_being_covered(self) -> None:
        env = scenario(membership_consistency=True)
        self.assertTrue(arrive(env, 0, A))
        self.assertEqual(int(np.asarray(env.task_dic[A]["status"], dtype=int).sum()), 0)
        env.apply_action(0, INDEX_OF[B], ACTION_MAP)
        env.step()
        gap = np.asarray(env.task_dic[A]["status"], dtype=int)
        self.assertEqual(int(gap.sum()), 1, "the stage must go back to uncovered once its only provider leaves")
        self.assertEqual(int(gap[4]), 1)
        self.assertNotIn(0, env.task_dic[A]["members"])

    def test_the_old_task_makes_no_further_progress_on_absent_credit(self) -> None:
        env = scenario(membership_consistency=True)
        arrive(env, 0, A)
        self.assertLess(float(stage(env.task_dic[A])["time_left"]), STAGE_TIME)
        env.apply_action(0, INDEX_OF[B], ACTION_MAP)
        env.step()
        # The environment's own coverage-interrupt rule: an uncovered stage's clock goes
        # back to the full stage duration. What must not happen is it then running down
        # with nobody there to do the work.
        self.assertEqual(float(stage(env.task_dic[A])["time_left"]), STAGE_TIME)
        for _ in range(6):
            env.step()
        self.assertEqual(float(stage(env.task_dic[A])["time_left"]), STAGE_TIME)

    def test_a_task_is_never_finished_by_a_robot_that_left_it(self) -> None:
        env = scenario(membership_consistency=True)
        arrive(env, 0, A)
        env.apply_action(0, INDEX_OF[B], ACTION_MAP)
        for _ in range(80):
            env.step()
        self.assertFalse(env.task_dic[A]["finished"], "A closed on ghost capability")

    def test_one_robot_never_supplies_two_tasks_at_once(self) -> None:
        env = scenario(membership_consistency=True)
        arrive(env, 0, A)
        arrive(env, 0, B)
        for _ in range(5):
            env.step()
        homes = [tid for tid, task in env.task_dic.items() if 0 in task["members"]]
        self.assertEqual(homes, [B])
        # And B really has the capability now: its own clock is moving.
        self.assertLess(float(stage(env.task_dic[B])["time_left"]), STAGE_TIME)

    def test_re_picking_the_current_task_is_a_true_no_op(self) -> None:
        env = scenario(membership_consistency=True)
        arrive(env, 0, A)
        started = env.agent_dic[0]["working_start_time"]
        clock = float(stage(env.task_dic[A])["time_left"])
        env.step()
        env.apply_action(0, INDEX_OF[A], ACTION_MAP)
        self.assertEqual(env.agent_dic[0]["status"], "working")
        self.assertEqual(env.agent_dic[0]["working_start_time"], started, "the work clock was reset")
        self.assertEqual(env.agent_dic[0]["need_decision"], False)
        self.assertIn(0, env.task_dic[A]["members"])
        env.step()
        self.assertLess(float(stage(env.task_dic[A])["time_left"]), clock, "coverage was lost by a no-op")

    def test_a_depottbound_release_clears_the_roster_too(self) -> None:
        env = scenario(membership_consistency=True)
        arrive(env, 0, A)
        env.apply_action(0, INDEX_OF["depot"], ACTION_MAP)
        env.step()
        self.assertNotIn(0, env.task_dic[A]["members"])
        self.assertEqual(int(np.asarray(env.task_dic[A]["status"], dtype=int).sum()), 1)


class LifecycleStillConsistentTests(unittest.TestCase):
    def test_working_timeout_releases_from_the_current_task_only(self) -> None:
        env = scenario(membership_consistency=True)
        arrive(env, 0, A)
        arrive(env, 0, B)
        # Force the timeout on robot 0 while it works at B.
        env.agent_dic[0]["working_start_time"] = env.current_time - 10_000.0
        env.agent_dic[0]["working_timeout_value"] = 0.0
        env.step()
        self.assertNotIn(0, env.task_dic[B]["members"], "timeout left it on the task it was working")
        self.assertNotIn(0, env.task_dic[A]["members"], "it was never supposed to be on A again")

    def test_reset_leaves_no_roster_behind(self) -> None:
        env = scenario(membership_consistency=True)
        arrive(env, 0, A)
        env.reset(seed=11)
        self.assertEqual(list(env.task_dic), [])

    def test_stage_transition_keeps_only_overlapping_members(self) -> None:
        env = scenario(membership_consistency=True)
        arrive(env, 0, A)
        task = env.task_dic[A]
        requirement = np.array([0, 0, 0, 0, 1], dtype=int)
        task["stages"].append(
            {
                "stage_id": 1,
                "requirements": requirement.copy(),
                "status": requirement.copy(),
                "time": 30.0,
                "time_left": 30.0,
                "finished": False,
            }
        )
        task["total_stages"] = 2
        stage(task)["time_left"] = EPS  # completes on this tick
        env.step()
        self.assertEqual(int(task["current_stage_index"]), 1)
        self.assertIn(0, task["members"], "robot 0 overlaps the next stage and should be kept")
        env.apply_action(0, INDEX_OF[B], ACTION_MAP)
        env.step()
        self.assertNotIn(0, task["members"], "the new stage lost its provider silently")


if __name__ == "__main__":
    unittest.main()
