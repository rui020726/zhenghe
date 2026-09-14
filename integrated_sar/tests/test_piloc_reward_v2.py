"""Reward V2, the corrected target-discovery flow, and the finish-accumulation fix.

Every test here drives the real ``timemap.Environment`` in ``v2`` mode with a
controlled, obstacle-free scene so a single reward term can be isolated. The
invariants locked are the ones the reward rework exists to establish:

* a middle target discovery pays ``new_target`` immediately (the old flow deleted
  the goal before scoring it, so this was always 0);
* a target seen by both robots is counted once for the team;
* discovering a target does not end the episode unless it was the last one;
* ``finish`` adds to the step reward instead of overwriting it;
* the components sum exactly to what ``step`` returns;
* the pheromone term cannot produce NaN/Inf;
* a single discovery event is worth the same whatever the episode limit ``E`` is.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PILOC_DIR = _REPO_ROOT / "piloc"

TIMEMAP_IMPORT_NAMES = ("gym", "PIL", "matplotlib")


def _skip_reason() -> str | None:
    missing = [n for n in TIMEMAP_IMPORT_NAMES if importlib.util.find_spec(n) is None]
    if missing:
        return f"the PILOC environment needs {', '.join(missing)}"
    return None


SKIP_REASON = _skip_reason()


def _load_modules():
    for entry in (_PILOC_DIR, _PILOC_DIR / "src" / "DDQN"):
        if str(entry) not in sys.path:
            sys.path.append(str(entry))
    from src.Map import timemap  # type: ignore
    from src.Map import reward_modes  # type: ignore

    return timemap, reward_modes


def _controlled_env(timemap, *, goals, agents, reward_mode="v2", seed=5):
    """A 60x60 obstacle-free environment with hand-placed robots and targets."""
    env = timemap.Environment(
        num_agents=len(agents),
        num_goals=len(goals),
        track_reward_terms=True,
        reward_mode=reward_mode,
        enable_comm_noise=False,
        seed=seed,
    )
    env.map = np.zeros((60, 60), dtype=np.int32)
    env.agents_pos = np.asarray(agents, dtype=np.int32)
    env.goals_pos = np.asarray(goals, dtype=np.int32).reshape(-1, 2)
    env.num_agents = len(agents)
    # Mirror the constructor: start cells carry pheromone, so the legacy
    # relative-pheromone term has a non-zero denominator.
    env.global_pheromone = np.zeros((60, 60), dtype=np.float32)
    for row, col in env.agents_pos:
        env.global_pheromone[int(row), int(col)] = 1
    env.local_pheromone = {
        f"agent_{i}": env.calculate_pheromone(int(p[0]), int(p[1]))
        for i, p in enumerate(env.agents_pos)
    }
    env.agent_positions_history = {
        f"agent_{i}": env.agents_pos[i] for i in range(len(agents))
    }
    obs = {f"agent_{i}": np.full((60, 60), 0.5, dtype=np.float32) for i in range(len(agents))}
    return env, obs


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class TargetDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.timemap, self.reward_modes = _load_modules()

    def test_one_discovery_pays_new_target_immediately(self) -> None:
        env, obs = _controlled_env(self.timemap, goals=[(30, 31)], agents=[(30, 30), (5, 5)])
        actions = {"agent_0": 0, "agent_1": 0}  # stay-ish; discovery is about position

        _obs, reward, done, _marks, _tc = env.step(actions, obs, {}, {}, 60)

        self.assertEqual(env.last_step_stats["new_targets"], 1)
        self.assertAlmostEqual(env.last_reward_terms["agent_0"]["new_target"], 1.0)
        # The team reward is shared: the robot that did not see it still gets it once.
        self.assertAlmostEqual(env.last_reward_terms["agent_1"]["new_target"], 1.0)
        self.assertTrue(all(np.isfinite(list(reward.values()))))

    def test_both_robots_seeing_one_target_counts_once(self) -> None:
        env, obs = _controlled_env(
            self.timemap, goals=[(30, 31)], agents=[(30, 30), (30, 32)]
        )
        actions = {"agent_0": 0, "agent_1": 0}

        env.step(actions, obs, {}, {}, 60)

        # new_targets is a set over goal indices, so two viewers of one goal == 1.
        self.assertEqual(env.last_step_stats["new_targets"], 1)
        self.assertAlmostEqual(env.last_reward_terms["agent_0"]["new_target"], 1.0)

    def test_discovery_does_not_end_episode_until_last_target(self) -> None:
        env, obs = _controlled_env(
            self.timemap, goals=[(30, 31), (5, 5)], agents=[(30, 30), (5, 6)]
        )
        # agent_0 sees goal (30,31); agent_1 sees goal (5,5). Both found -> all done.
        actions = {"agent_0": 0, "agent_1": 0}
        _obs, _reward, done_two, _m, _tc = env.step(actions, obs, {}, {}, 60)
        self.assertTrue(all(done_two.values()))

        # Now a scene where only one of two goals is seen: episode must NOT end.
        env2, obs2 = _controlled_env(
            self.timemap, goals=[(30, 31), (50, 50)], agents=[(30, 30), (5, 5)]
        )
        _obs, _r, done_one, _m, _tc = env2.step(
            {"agent_0": 0, "agent_1": 0}, obs2, {}, {}, 60
        )
        self.assertFalse(any(done_one.values()))
        self.assertEqual(len(env2.goals_pos), 1)

    def test_finish_is_added_not_overwritten(self) -> None:
        # agent_0 finds the only goal (finish + new_target); agent_1 is at the top
        # edge moving up, so it also collides on the same step. If finish overwrote
        # the step reward (the old bug), agent_1 would get exactly 3.0 with its
        # collision dropped; accumulation keeps the collision visible.
        env, obs = _controlled_env(self.timemap, goals=[(30, 31)], agents=[(30, 30), (0, 5)])
        actions = {"agent_0": 0, "agent_1": 0}  # agent_1 action 0 = move up -> out of bounds
        _obs, reward, _done, _m, _tc = env.step(actions, obs, {}, {}, 60)

        terms = env.last_reward_terms["agent_0"]
        self.assertIn("finish", terms)
        self.assertAlmostEqual(terms["finish"], 3.0)
        self.assertAlmostEqual(terms["new_target"], 1.0)
        # finish (3) + new_target (1) + time_cost (-0.006) => ~3.994, not 3.0.
        self.assertGreater(reward["agent_0"], 3.0)
        self.assertAlmostEqual(sum(terms.values()), reward["agent_0"], places=4)

        collision_terms = env.last_reward_terms["agent_1"]
        self.assertAlmostEqual(collision_terms["collision"], -0.4)
        self.assertAlmostEqual(sum(collision_terms.values()), reward["agent_1"], places=4)
        self.assertLess(reward["agent_1"], reward["agent_0"])


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class CoverageRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.timemap, self.reward_modes = _load_modules()

    def test_new_union_coverage_is_rewarded(self) -> None:
        env, obs = _controlled_env(self.timemap, goals=[(50, 50)], agents=[(30, 30), (5, 5)])
        # agent_0 moves right into fresh cells; coverage_after > coverage_before.
        _obs, _r, _d, _m, _tc = env.step({"agent_0": 3, "agent_1": 0}, obs, {}, {}, 60)
        self.assertGreater(env.last_step_stats["coverage_after"], env.last_step_stats["coverage_before"])
        self.assertGreater(env.last_reward_terms["agent_0"]["new_union_coverage"], 0.0)

    def test_two_robots_same_cells_do_not_double_count(self) -> None:
        # team_union_coverage is a set-union over agents, so overlapping views add once.
        env, _obs = _controlled_env(self.timemap, goals=[(50, 50)], agents=[(30, 30), (30, 31)])
        shared = {
            "agent_0": np.full((60, 60), 0.5, dtype=np.float32),
            "agent_1": np.full((60, 60), 0.5, dtype=np.float32),
        }
        for agent_map in shared.values():
            agent_map[10, 10] = 0.0
            agent_map[10, 11] = 0.0
        both = env.team_union_coverage(shared)
        single = env.team_union_coverage({"agent_0": shared["agent_0"]})
        self.assertAlmostEqual(both, single)

    def test_pure_recoverage_has_no_positive_reward(self) -> None:
        # obs already shows every cell the robot can see; a step that adds no new
        # free cell must not earn any positive exploration in V2.
        env, _obs = _controlled_env(self.timemap, goals=[(50, 50)], agents=[(30, 30), (5, 5)])
        primed = {a: np.full((60, 60), 0.5, dtype=np.float32) for a in env.agents}
        for agent_map in primed.values():
            agent_map[:, :] = 0.0  # everything already explored
        _obs, reward, _d, _m, _tc = env.step({"agent_0": 0, "agent_1": 0}, primed, {}, {}, 60)
        terms = env.last_reward_terms["agent_0"]
        self.assertNotIn("reexploration_shaped", terms)  # V2 emits no re-exploration term
        self.assertAlmostEqual(terms["new_union_coverage"], 0.0)
        self.assertLessEqual(reward["agent_0"], 0.0)  # only the time cost remains


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class RewardInvariantsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.timemap, self.reward_modes = _load_modules()

    def test_components_sum_to_total_for_every_step(self) -> None:
        random.seed(1)
        np.random.seed(1)
        env, obs = _controlled_env(
            self.timemap, goals=[(30, 31), (40, 40)], agents=[(30, 30), (5, 5)]
        )
        for _ in range(15):
            actions = {a: random.randrange(4) for a in env.agents}
            obs, reward, _d, _m, _tc = env.step(actions, obs, {}, {}, 60)
            for agent_id, parts in env.last_reward_terms.items():
                self.assertAlmostEqual(sum(parts.values()), reward[agent_id], places=4)
            if len(env.goals_pos) == 0:
                break

    def test_pheromone_term_never_produces_nan_or_inf(self) -> None:
        rm = self.reward_modes
        weights = rm.RewardWeights(pheromone=1.0)
        for bad in (float("inf"), float("-inf"), float("nan"), 1e18, -1e18):
            value = rm.pheromone_term(bad, 1.0, clip=weights.pheromone_clip)
            self.assertTrue(np.isfinite(value), f"{bad!r} produced {value!r}")

    def test_pheromone_weight_zero_disables_the_term(self) -> None:
        rm = self.reward_modes
        weights = rm.RewardWeights(pheromone=0.0)
        self.assertEqual(rm.pheromone_term(1e9, weights.pheromone, clip=weights.pheromone_clip), 0.0)

    def test_single_discovery_reward_is_independent_of_episode_limit(self) -> None:
        for limit in (10, 250):
            env, obs = _controlled_env(
                self.timemap, goals=[(30, 31)], agents=[(30, 30), (5, 5)], seed=9
            )
            _obs, _r, _d, _m, _tc = env.step({"agent_0": 0, "agent_1": 0}, obs, {}, {}, limit)
            self.assertAlmostEqual(env.last_reward_terms["agent_0"]["new_target"], 1.0,
                                   msg=f"discovery reward changed with E={limit}")


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class LegacyModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.timemap, self.reward_modes = _load_modules()

    def test_legacy_still_runs_and_is_finite(self) -> None:
        random.seed(2)
        np.random.seed(2)
        env, obs = _controlled_env(
            self.timemap, goals=[(30, 31), (40, 40)], agents=[(30, 30), (5, 5)],
            reward_mode="legacy",
        )
        for _ in range(10):
            actions = {a: random.randrange(4) for a in env.agents}
            obs, reward, _d, _m, _tc = env.step(actions, obs, {}, {}, 60)
            for agent_id, parts in env.last_reward_terms.items():
                self.assertAlmostEqual(sum(parts.values()), reward[agent_id], places=3)
            self.assertTrue(all(np.isfinite(list(reward.values()))))

    def test_step_return_contract_is_unchanged(self) -> None:
        env, obs = _controlled_env(self.timemap, goals=[(30, 31)], agents=[(30, 30), (5, 5)])
        result = env.step({"agent_0": 0, "agent_1": 0}, obs, {}, {}, 60)
        self.assertEqual(len(result), 5)  # obs, reward, done, repeat_mark, two_channelmap
        _obs, reward, done, marks, _tc = result
        self.assertEqual(set(reward), {"agent_0", "agent_1"})
        self.assertEqual(set(done), {"agent_0", "agent_1"})
        self.assertIsInstance(marks, list)


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class PotentialModeTests(unittest.TestCase):
    """Formula-level checks for the potential-based coverage shaping."""

    def setUp(self) -> None:
        self.timemap, self.reward_modes = _load_modules()

    def test_potential_shaping_matches_the_formula(self) -> None:
        rm = self.reward_modes
        weights = rm.RewardWeights(potential_beta=1.0, gamma=0.99, coverage=2.0, time_cost=0.0)
        events = rm.StepEvents(potential_current=0.20, potential_next=0.25, all_found=False)
        terms = rm.compute_potential_rewards(["agent_0"], events, weights)["agent_0"]
        expected = 1.0 * (0.99 * 0.25 - 0.20)
        self.assertAlmostEqual(terms["potential"], expected)
        self.assertNotIn("new_union_coverage", terms)  # replaced by the shaping term

    def test_terminal_potential_is_zeroed(self) -> None:
        rm = self.reward_modes
        weights = rm.RewardWeights(potential_beta=1.0, gamma=0.99, time_cost=0.0)
        # Real termination: the caller passes potential_next=0 (Phi_terminal := 0).
        events = rm.StepEvents(potential_current=0.3, potential_next=0.0, all_found=True)
        terms = rm.compute_potential_rewards(["agent_0"], events, weights)["agent_0"]
        self.assertAlmostEqual(terms["potential"], 1.0 * (0.99 * 0.0 - 0.3))
        self.assertAlmostEqual(terms["finish"], weights.finish)


if __name__ == "__main__":
    unittest.main()
