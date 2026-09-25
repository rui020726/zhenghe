"""Synthetic tests for the B1 search-side probe. No joint-driver episode is launched here.

The environment is the real ``piloc.src.Map.timemap.Environment`` on a hand-built map, because
the thing under test is the interaction between the environment's own detection step, its
delete-only goal array, and the adapter's event translation. A stand-in would only prove that
two fakes agree with each other.

Boundary this file sits on: stepping an environment inside a unit test is a synthetic test, not
an episode launch, and it produces no experimental evidence -- it uses maps and positions no
scenario in any batch contains. The launch ledger counts joint-driver episodes.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from integrated_sar.adapters.piloc import PilocSearchAdapter
from integrated_sar.profiles import DeterministicTargetProfileProvider
from integrated_sar.piloc_runtime import build_piloc_environment, import_timemap_module
from integrated_sar.scenarios import build_scenario
from integrated_sar.search_probe import SearchProbe, closure_flags
from integrated_sar.search_runtime import (
    initialize_timestamp_observations,
    initialize_two_channel_map,
)

SIZE = 60
# timemap's action_list: up, down, left, right.
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


def _environment(goals, agents, *, seed: int = 11, obstacles=()) -> object:
    """The real PILOC environment forced onto hand-placed cells.

    ``build_piloc_environment`` has to be used (its constructor draws a map and needs the 60x60
    guard satisfied), then ``load`` installs the world the test actually wants -- the same
    order the production reset uses.
    """
    scenario = build_scenario(size=SIZE, seed=seed, num_agents=len(agents),
                              num_goals=max(1, len(goals)), obstacle_density=0.0)
    environment = build_piloc_environment(scenario, seed=seed)
    grid = np.zeros((SIZE, SIZE), dtype=np.int32)
    for row, col in obstacles:
        grid[row, col] = 1
    environment.load(
        grid,
        np.asarray(agents, dtype=np.int32),
        np.asarray(goals, dtype=np.int32).reshape(-1, 2),
    )
    environment.reseed(seed)
    from integrated_sar.piloc_runtime import _rebuild_position_state

    _rebuild_position_state(environment)
    return environment


def _adapter(environment, actions: dict, *, probe=None, horizon: int = 40):
    provider = lambda env, obs, channels: dict(actions)  # noqa: E731 - a fixed script, not a policy
    obs = initialize_timestamp_observations(environment, view_radius=5)
    return PilocSearchAdapter(
        env=environment,
        action_provider=provider,
        profile_provider=DeterministicTargetProfileProvider(),
        obs=obs,
        two_channelmap=initialize_two_channel_map(obs),
        episode_horizon=horizon,
        detection_radius=5,
        search_probe=probe,
    )


def _probe(directory: Path, name: str = "sp.jsonl") -> SearchProbe:
    return SearchProbe(Path(directory) / name)


def _steps(adapter, *, time=0.0, dt=1.0):
    return adapter.step(time, dt)


class TwoObserversTests(unittest.TestCase):
    """One target seen by two robots is one discovery, recorded as two observers."""

    def test_both_pairs_are_recorded_and_only_one_event_is_emitted(self) -> None:
        # Both robots are pinned at row 0 by an out-of-bounds move, so they stay put and both
        # remain within Chebyshev 5 of the target at (0, 31).
        environment = _environment(goals=[(0, 31)], agents=[(0, 30), (0, 32)])
        with tempfile.TemporaryDirectory() as directory:
            adapter = _adapter(environment, {"agent_0": UP, "agent_1": UP},
                               probe=_probe(directory))
            result = _steps(adapter)
            row = adapter.search_probe.rows[-1]
            self.assertEqual(len(result.detections), 1, "two observers must not mean two events")
            self.assertEqual(len(row["observed_pairs"]), 2, "both observers must be recorded")
            self.assertEqual(sorted(pair["agent"] for pair in row["observed_pairs"]),
                             ["agent_0", "agent_1"])
            self.assertEqual(row["env_deleted_target_ids"], row["adapter_event_target_ids"])
            self.assertTrue(row["closure"]["all_views_agree"])


class SameCellTargetsTests(unittest.TestCase):
    def test_two_targets_in_one_cell_get_two_ids_and_two_events(self) -> None:
        environment = _environment(goals=[(0, 31), (0, 31)], agents=[(0, 30), (0, 32)])
        with tempfile.TemporaryDirectory() as directory:
            adapter = _adapter(environment, {"agent_0": UP, "agent_1": UP},
                               probe=_probe(directory))
            result = _steps(adapter)
            row = adapter.search_probe.rows[-1]
            self.assertEqual(len(result.detections), 2)
            self.assertEqual(len(set(event.target_id for event in result.detections)), 2,
                             "two targets sharing a cell must not collapse into one id")
            self.assertEqual(len(row["env_deleted_target_ids"]), 2)


class IndexShiftTests(unittest.TestCase):
    """Deleting array entry 0 must not hand entry 1's discovery to entry 0's id."""

    def test_survivor_keeps_its_reset_id_after_the_array_shifts(self) -> None:
        environment = _environment(goals=[(0, 31), (0, 45)], agents=[(0, 30), (0, 32)])
        with tempfile.TemporaryDirectory() as directory:
            adapter = _adapter(environment, {"agent_0": UP, "agent_1": UP},
                               probe=_probe(directory))
            result = _steps(adapter)
            row = adapter.search_probe.rows[-1]
            self.assertEqual([event.target_id for event in result.detections],
                             ["piloc-e1-target-0"])
            self.assertEqual(row["adapter_alive_after"], ["piloc-e1-target-1"])
            self.assertEqual([entry["reset_index"] for entry in row["remaining_after_move"]], [1])
            self.assertEqual(len(environment.goals_pos), 1)


class DetectionInstantTests(unittest.TestCase):
    """The record must show the target where it stood when discovery was judged."""

    def test_detection_cell_is_the_pre_move_cell_and_survivors_are_reported_moved(self) -> None:
        environment = _environment(goals=[(0, 45)], agents=[(0, 30), (0, 32)])
        with tempfile.TemporaryDirectory() as directory:
            adapter = _adapter(environment, {"agent_0": UP, "agent_1": UP},
                               probe=_probe(directory))
            moved_steps = 0
            for step in range(5):
                _steps(adapter, time=float(step))
                row = adapter.search_probe.rows[-1]
                detection_cell = row["goals_at_detection"][0]["cell"]
                after_cell = row["remaining_after_move"][0]["cell_after"]
                if step == 0:
                    self.assertEqual(detection_cell, [0, 45],
                                     "the first detection must use the placement cell")
                else:
                    previous = adapter.search_probe.rows[-2]
                    self.assertEqual(detection_cell,
                                     previous["remaining_after_move"][0]["cell_after"],
                                     "this step's detection cell must be last step's moved cell, "
                                     "not the post-deletion-and-movement array of this step")
                self.assertLessEqual(max(abs(after_cell[0] - detection_cell[0]),
                                         abs(after_cell[1] - detection_cell[1])), 1,
                                     "the hidden target walks at most one cell per step")
                moved_steps += int(after_cell != detection_cell)
            self.assertGreaterEqual(moved_steps, 1,
                                     "a target that never moved would not test this property")


class VisibleAtResetNotAfterTheMoveTests(unittest.TestCase):
    """"It was in range at the start" is not "the detector saw it"."""

    def test_the_first_detection_step_uses_the_post_move_position(self) -> None:
        environment = _environment(goals=[(0, 15)], agents=[(0, 10)])
        # At reset the target sits at exactly Chebyshev 5: a legal observation would exist.
        self.assertEqual(environment._observed_goal_indices(environment.agents_pos), {0},
                         "the setup is no longer visible-at-reset")
        with tempfile.TemporaryDirectory() as directory:
            adapter = _adapter(environment, {"agent_0": LEFT},
                               probe=_probe(directory))
            result = _steps(adapter)
            row = adapter.search_probe.rows[-1]
            self.assertEqual(result.detections, (),
                             "the robot walked away before the detection step ran")
            self.assertEqual(row["agents_after_move"][0], [0, 9])
            self.assertEqual(row["observed_pairs"], [])
            self.assertEqual(row["env_deleted_target_ids"], [])


class EmptyGoalSetTests(unittest.TestCase):
    def test_no_targets_is_a_clean_termination_not_a_crash(self) -> None:
        environment = _environment(goals=[(0, 45)], agents=[(0, 10), (0, 12)])
        environment.goals_pos = np.zeros((0, 2), dtype=np.int32)
        with tempfile.TemporaryDirectory() as directory:
            probe = _probe(directory)
            adapter = _adapter(environment, {"agent_0": UP, "agent_1": DOWN}, probe=probe)
            result = _steps(adapter)
            snapshot = adapter.search_side_snapshot()
            self.assertTrue(snapshot["all_found"])
            self.assertEqual(snapshot["remaining_count"], 0)
            self.assertEqual(result.metrics["remaining_targets"], 0)
            probe.note_final(**snapshot, termination_reason="all_found")
            probe.write()
            rows = [line for line in probe.path.read_text(encoding="utf-8").splitlines() if line]
            self.assertTrue(any('"kind": "final"' in line for line in rows))


class UninstalledProbeTests(unittest.TestCase):
    """Without a probe the environment must not even enter the recording branch."""

    def test_probe_absent_leaves_the_trace_switch_off_and_positions_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            goals = [(0, 45), (30, 30)]
            agents = [(0, 40), (30, 25)]
            actions = {"agent_0": DOWN, "agent_1": RIGHT}
            quiet = _adapter(_environment(goals=goals, agents=agents), actions)
            loud = _adapter(_environment(goals=goals, agents=agents), actions,
                            probe=_probe(directory))
            self.assertIs(quiet.env.search_trace_enabled, False)
            self.assertIs(loud.env.search_trace_enabled, True)
            quiet_steps, loud_steps = [], []
            for step in range(6):
                quiet.env._py_rng.setstate(_INITIAL_PY)
                quiet_steps.append(_observe(quiet, step))
                loud.env._py_rng.setstate(_INITIAL_PY)
                loud_steps.append(_observe(loud, step))
            self.assertEqual(quiet_steps, loud_steps,
                             "installing the probe changed the stepped world")
            self.assertEqual(loud.search_probe.steps_recorded, 6)
            self.assertTrue(all(row["closure"]["all_views_agree"] for row in loud.search_probe.rows
                                 if row["kind"] == "step"))

    def test_recovery_after_a_raising_step_restarts_bookkeeping_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            probe = _probe(directory)
            environment = _environment(goals=[(0, 55), (0, 56)], agents=[(0, 40), (0, 41)])
            calls = {"n": 0}

            def provider(env, obs, channels):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise RuntimeError("interrupted episode")
                return {"agent_0": UP, "agent_1": UP}

            adapter = _adapter(environment, {}, probe=probe)
            adapter.action_provider = provider
            _steps(adapter)
            with self.assertRaises(RuntimeError):
                _steps(adapter)
            self.assertEqual(environment._trace_alive, [0, 1],
                             "the aborted step must not have deleted anything")
            adapter.reset()
            self.assertEqual(environment._trace_alive, [0, 1])
            _steps(adapter)
            self.assertEqual(probe.rows[-1]["kind"], "step")
            self.assertTrue(probe.rows[-1]["closure"]["all_views_agree"])


class ClosureFlagTests(unittest.TestCase):
    def test_pair_list_collapses_by_target_id_not_by_pair_count(self) -> None:
        flags = closure_flags(env_ids=["t0"], pair_ids=["t0", "t0"], adapter_ids=["t0"],
                              alive_views_agree=True)
        self.assertTrue(flags["all_views_agree"])
        self.assertEqual(flags["view_pair_observed"], ["t0"])

    def test_a_missing_event_is_reported_as_a_disagreement(self) -> None:
        flags = closure_flags(env_ids=["t0"], pair_ids=["t0"], adapter_ids=[],
                              alive_views_agree=True)
        self.assertFalse(flags["all_views_agree"])
        self.assertFalse(flags["env_matches_adapter"])


def _observe(adapter, step: int) -> dict:
    result = adapter.step(float(step), 1.0)
    env = adapter.env
    return {
        "agents": [[int(p[0]), int(p[1])] for p in env.agents_pos],
        "goals": [[int(p[0]), int(p[1])] for p in env.goals_pos],
        "py_rng": str(env._py_rng.getstate()),
        "np_rng": str(env._np_rng.get_state()[1]),
        "steps": int(env.steps),
        "detections": sorted(str(event.target_id) for event in result.detections),
        "rewards": {key: round(float(value), 9) for key, value in result.metrics["search_rewards"].items()},
    }


try:
    _INITIAL_PY = _environment(goals=[(0, 45)], agents=[(0, 40)])._py_rng.getstate()
except Exception:  # pragma: no cover - the guard is the module-level import
    _INITIAL_PY = None


@unittest.skipIf(_INITIAL_PY is None, "pilot environment module unavailable")
class PairedSteppingTests(unittest.TestCase):
    """The two conditions above stay comparable only if both start from one rng state."""

    def test_the_initial_state_helper_is_usable(self) -> None:
        self.assertIsInstance(_INITIAL_PY, tuple)
        self.assertEqual(_INITIAL_PY[0], 3)


if __name__ == "__main__":
    unittest.main()
