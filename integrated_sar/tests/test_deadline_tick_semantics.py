"""The stage clock's boundary behaviour, pinned as tests rather than described in prose.

These tests exist because the six remaining binding-planner failures all expire within a
fraction of one tick of finishing, which makes the exact tick semantics of
``old_task_env`` load-bearing for the attribution. Every assertion here records what the
environment *currently* does; none of them changes it. Where the current behaviour looks
wrong it is marked as such in the docstring and reported in
``runs/coalition_formation_latency_20260918/DEADLINE_TICK_TESTS.md`` instead of fixed.

The environment is driven directly (``task_update()`` / ``step()`` on a real ``TaskEnv``)
rather than through the adapter, so a failure here points at the environment and not at
the integration layer. Tasks are injected by hand into ``env.task_dic`` with the same
shape ``Task03Adapter._inject_task`` produces, because the stock generator cannot be told
to produce one task with one stage of a chosen duration.
"""

from __future__ import annotations

import unittest

import numpy as np

from integrated_sar.task03_runtime import load_task_env_module

TRAITS = 5


def _requirements(*dims: int) -> np.ndarray:
    vector = np.zeros(TRAITS, dtype=int)
    for dim in dims:
        vector[dim] = 1
    return vector


class _Fixture(unittest.TestCase):
    """A one-task, two-robot environment whose clock is driven by hand."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_task_env_module()

    def build(self, *, durations, requirements, deadline=1000.0, dt=1.0, hard_expire=False):
        env = self.module.TaskEnv(
            tasks_range=(0, 0),
            species_range=(2, 2),
            per_species_range=(1, 1),
            traits_dim=TRAITS,
            enable_tidal_tasks=False,
            enable_discovered_tasks=False,
            dynamic_task_discovery=False,
            dt=dt,
            hard_expire=hard_expire,
            seed=11,
        )
        stages = []
        for index, (duration, req) in enumerate(zip(durations, requirements)):
            stages.append(
                {
                    "stage_id": index,
                    "requirements": req.copy(),
                    "status": req.copy(),
                    "time": float(duration),
                    "time_left": float(duration),
                    "finished": False,
                }
            )
        task_id = 900
        env.task_dic[task_id] = {
            "ID": task_id,
            "stages": stages,
            "current_stage_index": 0,
            "total_stages": len(stages),
            "requirements": stages[0]["requirements"].copy(),
            "status": stages[0]["status"].copy(),
            "time": stages[0]["time"],
            "location": np.array([0.5, 0.5]),
            "scene_complexity": 0.5,
            "finished": False,
            "feasible_assignment": False,
            "members": [],
            "time_start": 0.0,
            "time_finish": 0.0,
            "deadline": float(deadline),
            "expired": False,
        }
        return env, task_id

    def set_abilities(self, env, agent_id, *dims):
        """Pin one robot's ability vector so a stage's coverage is exactly known.

        The stock generator guarantees every ability dimension is covered by *some*
        species, so two generated robots can silently satisfy a requirement this file
        means to leave unmet.
        """
        vector = np.zeros(TRAITS)
        for dim in dims:
            vector[dim] = 1.0
        env.agent_dic[agent_id]["abilities"] = vector

    def cover(self, env, task_id, *agent_ids):
        """Put these robots on the task as arrived members, without simulating travel."""
        task = env.task_dic[task_id]
        for agent_id in agent_ids:
            agent = env.agent_dic[agent_id]
            agent.update(
                {
                    "status": "working",
                    "assigned": True,
                    "current_task": task_id,
                    "target_type": None,
                    "target_id": None,
                    "target_pos": None,
                    "working_start_time": env.current_time,
                    "working_timeout_value": 1e9,
                }
            )
            if agent_id not in task["members"]:
                task["members"].append(agent_id)

    def uncover(self, env, task_id, agent_id):
        """Send one robot away, leaving the task short of its requirement."""
        task = env.task_dic[task_id]
        if agent_id in task["members"]:
            task["members"].remove(agent_id)
        env.agent_dic[agent_id].update(
            {"status": "idle", "assigned": False, "current_task": -1}
        )

    def tick(self, env):
        """Advance one tick the way ``step`` does: clock first, then the task clock."""
        env.current_time += env.dt
        env.task_update()


class SingleStageClockTests(_Fixture):
    def test_partial_final_tick_still_completes_the_stage(self) -> None:
        # time_left = 0.2 with dt = 1.0: the decrement overshoots to -0.8 and the stage
        # finishes. There is no partial credit and no leftover carried to the next stage.
        env, task_id = self.build(durations=[0.2], requirements=[_requirements(0)])
        self.cover(env, task_id, 0)
        self.tick(env)
        task = env.task_dic[task_id]
        self.assertTrue(task["stages"][0]["finished"])
        self.assertTrue(task["finished"])
        self.assertLessEqual(task["stages"][0]["time_left"], 0.0)

    def test_exact_one_tick_stage_finishes_on_that_tick(self) -> None:
        # time_left = 1.0 with dt = 1.0: time_left lands exactly on 0.0 and `<= 0` fires.
        env, task_id = self.build(durations=[1.0], requirements=[_requirements(0)])
        self.cover(env, task_id, 0)
        self.tick(env)
        self.assertTrue(env.task_dic[task_id]["finished"])

    def test_a_stage_needing_two_ticks_needs_both(self) -> None:
        env, task_id = self.build(durations=[2.0], requirements=[_requirements(0)])
        self.cover(env, task_id, 0)
        self.tick(env)
        self.assertFalse(env.task_dic[task_id]["finished"])
        self.assertAlmostEqual(env.task_dic[task_id]["stages"][0]["time_left"], 1.0)
        self.tick(env)
        self.assertTrue(env.task_dic[task_id]["finished"])


class CoverageResetTests(_Fixture):
    def test_breaking_coverage_at_95_percent_restores_the_full_duration(self) -> None:
        # The reset is to `stage["time"]`, not to a partial value: every elapsed step on
        # the stage is discarded. A 10-step stage at 9.5 steps loses all 9.5.
        env, task_id = self.build(durations=[10.0], requirements=[_requirements(0)])
        self.cover(env, task_id, 0)
        for _ in range(9):
            self.tick(env)
        stage = env.task_dic[task_id]["stages"][0]
        self.assertAlmostEqual(stage["time_left"], 1.0)

        self.uncover(env, task_id, 0)
        self.tick(env)
        self.assertAlmostEqual(stage["time_left"], 10.0)

    def test_no_members_at_all_also_resets(self) -> None:
        env, task_id = self.build(durations=[4.0], requirements=[_requirements(0)])
        self.cover(env, task_id, 0)
        self.tick(env)
        self.assertAlmostEqual(env.task_dic[task_id]["stages"][0]["time_left"], 3.0)
        self.uncover(env, task_id, 0)
        self.tick(env)
        self.assertAlmostEqual(env.task_dic[task_id]["stages"][0]["time_left"], 4.0)

    def test_the_reset_pins_at_the_full_duration_across_a_long_gap(self) -> None:
        # Once broken, `time_left` is re-assigned every tick, so it stays at the full
        # duration for the whole gap rather than draining.
        env, task_id = self.build(durations=[6.0], requirements=[_requirements(0)])
        for _ in range(5):
            self.tick(env)
        for _ in range(20):
            self.tick(env)
        self.assertAlmostEqual(env.task_dic[task_id]["stages"][0]["time_left"], 6.0)

    def test_work_is_not_destroyed_while_coverage_holds(self) -> None:
        # The complement of the reset: with coverage unbroken the clock only decreases,
        # which is the property DURATION_RELEVANCE.md checked on the committed traces.
        env, task_id = self.build(durations=[12.0], requirements=[_requirements(0)])
        self.cover(env, task_id, 0)
        seen = []
        for _ in range(8):
            self.tick(env)
            seen.append(env.task_dic[task_id]["stages"][0]["time_left"])
        self.assertEqual(seen, sorted(seen, reverse=True))

    def test_partial_ability_does_not_start_the_clock(self) -> None:
        # The stage needs dimensions 0 and 1; both robots carry only dimension 0, so the
        # requirement stays unmet and no work happens however many robots stand there.
        env, task_id = self.build(
            durations=[3.0], requirements=[_requirements(0, 1)]
        )
        self.set_abilities(env, 0, 0)
        self.set_abilities(env, 1, 0)
        self.cover(env, task_id, 0, 1)
        for _ in range(10):
            self.tick(env)
        self.assertAlmostEqual(env.task_dic[task_id]["stages"][0]["time_left"], 3.0)
        self.assertFalse(env.task_dic[task_id]["feasible_assignment"])

    def test_extra_agents_do_not_speed_the_stage_up(self) -> None:
        # Two robots whose abilities together over-cover a one-dimension stage: the
        # decrement is a flat dt, so a surplus member buys nothing but redundancy.
        env, task_id = self.build(durations=[5.0], requirements=[_requirements(0)])
        self.set_abilities(env, 0, 0)
        self.set_abilities(env, 1, 0, 1)
        self.cover(env, task_id, 0, 1)
        self.tick(env)
        self.assertAlmostEqual(env.task_dic[task_id]["stages"][0]["time_left"], 4.0)


class AssignedIsNotOnSiteTests(_Fixture):
    def test_an_en_route_robot_does_not_count_as_coverage(self) -> None:
        # `members` is only appended on arrival, so a robot that is one tick away when
        # the last member leaves does not protect the clock.
        env, task_id = self.build(durations=[8.0], requirements=[_requirements(0)])
        self.cover(env, task_id, 0)
        for _ in range(4):
            self.tick(env)
        self.assertAlmostEqual(env.task_dic[task_id]["stages"][0]["time_left"], 4.0)

        self.uncover(env, task_id, 0)
        env.agent_dic[1].update(
            {"status": "moving", "target_type": "task", "target_id": task_id}
        )
        self.tick(env)
        self.assertAlmostEqual(env.task_dic[task_id]["stages"][0]["time_left"], 8.0)


class DeadlineBoundaryTests(_Fixture):
    def test_deadline_is_not_crossed_by_equality(self) -> None:
        # `current_time > deadline`, strictly. At equality the task is still live.
        env, task_id = self.build(
            durations=[50.0], requirements=[_requirements(0)], deadline=5.0
        )
        for _ in range(5):
            self.tick(env)
        self.assertEqual(env.current_time, 5.0)
        self.assertFalse(env.task_dic[task_id]["expired"])

        self.tick(env)
        self.assertEqual(env.current_time, 6.0)
        self.assertTrue(env.task_dic[task_id]["expired"])

    def test_expiry_is_soft_and_the_work_continues(self) -> None:
        # Soft expiry marks the flag only: no release, no delete, the clock keeps running.
        # With deadline 1.0 the flag first fires on the tick that reaches current_time 2.0,
        # because the test is `>` and not `>=`.
        env, task_id = self.build(
            durations=[4.0], requirements=[_requirements(0)], deadline=1.0
        )
        self.cover(env, task_id, 0)
        self.tick(env)
        self.assertEqual(env.current_time, 1.0)
        self.assertFalse(env.task_dic[task_id]["expired"])

        self.tick(env)
        self.assertEqual(env.current_time, 2.0)
        self.assertTrue(env.task_dic[task_id]["expired"])
        self.assertAlmostEqual(env.task_dic[task_id]["stages"][0]["time_left"], 2.0)
        self.assertEqual(env.task_dic[task_id]["members"], [0])
        self.assertFalse(env.task_dic[task_id]["finished"])

    def test_hard_expire_finishes_and_removes_the_task_from_work(self) -> None:
        env, task_id = self.build(
            durations=[4.0], requirements=[_requirements(0)], deadline=1.0, hard_expire=True
        )
        self.cover(env, task_id, 0)
        self.tick(env)
        self.assertFalse(env.task_dic[task_id]["expired"])
        self.tick(env)
        task = env.task_dic[task_id]
        self.assertTrue(task["expired"])
        self.assertTrue(task["finished"])

    def test_a_stage_finishing_on_the_crossing_tick_is_reported_expired(self) -> None:
        # FINDING (reported, not fixed): loop A sets `expired` before loop B sets
        # `finished`, so a task that completes on the very tick its deadline is crossed
        # ends with both flags. The adapter reads `expired` first
        # (`task03.py:1258`), so this run is recorded as expired despite completing.
        env, task_id = self.build(
            durations=[3.0], requirements=[_requirements(0)], deadline=2.0
        )
        self.cover(env, task_id, 0)
        self.tick(env)
        self.tick(env)
        task = env.task_dic[task_id]
        # The crossing tick is the third one: current_time becomes 3.0 > 2.0 while the
        # stage clock also reaches zero.
        self.assertFalse(task["finished"])
        self.tick(env)
        self.assertEqual(env.current_time, 3.0)
        self.assertTrue(task["expired"])
        self.assertTrue(task["finished"])


class MultiStageTransitionTests(_Fixture):
    def test_transition_keeps_only_contributing_members(self) -> None:
        # On a stage transition the members that cannot contribute to the *next* stage
        # are released, and the new stage starts at its own full duration.
        env, task_id = self.build(
            durations=[2.0, 2.0],
            requirements=[_requirements(0), _requirements(1)],
        )
        self.cover(env, task_id, 0)
        self.tick(env)
        self.tick(env)
        task = env.task_dic[task_id]
        self.assertEqual(task["current_stage_index"], 1)
        self.assertEqual(task["members"], [])
        self.assertAlmostEqual(task["stages"][1]["time_left"], 2.0)

    def test_transition_and_deadline_on_the_same_tick(self) -> None:
        # Same-tick shape as the single-stage case: expired and advanced together.
        env, task_id = self.build(
            durations=[2.0, 2.0],
            requirements=[_requirements(0), _requirements(0)],
            deadline=1.0,
        )
        self.cover(env, task_id, 0)
        self.tick(env)
        self.tick(env)
        task = env.task_dic[task_id]
        self.assertEqual(env.current_time, 2.0)
        self.assertTrue(task["expired"])
        self.assertEqual(task["current_stage_index"], 1)
        self.assertFalse(task["finished"])


class ClockDecouplingTests(_Fixture):
    def test_task_update_uses_self_dt_not_the_step_argument(self) -> None:
        # `step(dt)` advances the world by the argument but `task_update` drains the
        # stage clock by `self.dt`. Passing a different dt desynchronises the two clocks;
        # recorded runs never do (joint_driver pins dt == self.dt == 1.0).
        env, task_id = self.build(durations=[10.0], requirements=[_requirements(0)], dt=1.0)
        self.cover(env, task_id, 0)
        env.step(0.25)
        # The world moved 0.25 but the stage clock moved a full self.dt of 1.0.
        self.assertAlmostEqual(env.current_time, 0.25)
        self.assertAlmostEqual(env.task_dic[task_id]["stages"][0]["time_left"], 9.0)

    def test_matching_dt_keeps_the_clocks_together(self) -> None:
        env, task_id = self.build(durations=[10.0], requirements=[_requirements(0)], dt=0.5)
        self.cover(env, task_id, 0)
        env.step(0.5)
        self.assertAlmostEqual(env.current_time, 0.5)
        self.assertAlmostEqual(env.task_dic[task_id]["stages"][0]["time_left"], 9.5)


if __name__ == "__main__":
    unittest.main()
