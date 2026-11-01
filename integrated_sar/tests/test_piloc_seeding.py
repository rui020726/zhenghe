"""Same-seed reproducibility of the PILOC environment on CPU.

Two runs on one seed used to diverge because the environment drew from the
global ``random``/``np.random`` streams, which the training loop also consumes
(and the policy consumes torch streams). The environment now owns independent
RNGs seeded per instance, so:

* two identically-seeded environments produce identical rollouts;
* consuming the *global* RNG between constructions cannot change that;
* a fixed validation seed survives any amount of training-time randomness.
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


def _timemap():
    for entry in (_PILOC_DIR, _PILOC_DIR / "src" / "DDQN"):
        if str(entry) not in sys.path:
            sys.path.append(str(entry))
    from src.Map import timemap  # type: ignore

    return timemap


def _rollout(timemap, seed: int, steps: int, actions_per_step):
    """Build a seeded v2 environment and replay a fixed action sequence."""
    env = timemap.Environment(num_agents=2, num_goals=4, reward_mode="v2", seed=seed)
    obs = {a: np.full((60, 60), 0.5, dtype=np.float32) for a in env.agents}
    two_channelmap = env.two_channel_map(obs, {})
    trace = []
    for step in range(steps):
        actions = {a: actions_per_step[step % len(actions_per_step)][i]
                   for i, a in enumerate(env.agents)}
        obs, reward, done, marks, two_channelmap = env.step(actions, obs, {}, two_channelmap, 60)
        trace.append(
            (
                env.agents_pos.copy(),
                env.goals_pos.copy(),
                {k: float(v) for k, v in reward.items()},
                list(marks),
                dict(done),
            )
        )
        if any(done.values()):
            break
    return env, trace


def _fixed_actions(count: int):
    rng = random.Random(1234)
    return [[rng.randrange(4), rng.randrange(4)] for _ in range(count)]


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class SeededRolloutReproducibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.timemap = _timemap()
        self.actions = _fixed_actions(20)

    def _compare(self, first, second) -> None:
        self.assertEqual(len(first), len(second))
        for (pos_a, goals_a, rew_a, marks_a, done_a), (
            pos_b, goals_b, rew_b, marks_b, done_b,
        ) in zip(first, second):
            np.testing.assert_array_equal(pos_a, pos_b)
            np.testing.assert_array_equal(goals_a, goals_b)
            self.assertEqual(rew_a, rew_b)
            self.assertEqual(marks_a, marks_b)
            self.assertEqual(done_a, done_b)

    def test_the_same_seed_replays_identically(self) -> None:
        _env1, trace1 = _rollout(self.timemap, seed=7, steps=15, actions_per_step=self.actions)
        _env2, trace2 = _rollout(self.timemap, seed=7, steps=15, actions_per_step=self.actions)
        self._compare(trace1, trace2)

    def test_global_rng_churn_between_runs_changes_nothing(self) -> None:
        _env1, trace1 = _rollout(self.timemap, seed=7, steps=15, actions_per_step=self.actions)
        # A training loop consuming global randomness must not shift the env stream.
        random.seed(999)
        np.random.seed(999)
        for _ in range(500):
            random.random()
            np.random.random()
        _env2, trace2 = _rollout(self.timemap, seed=7, steps=15, actions_per_step=self.actions)
        self._compare(trace1, trace2)

    def test_a_different_seed_diverges(self) -> None:
        _env1, trace1 = _rollout(self.timemap, seed=7, steps=15, actions_per_step=self.actions)
        _env2, trace2 = _rollout(self.timemap, seed=8, steps=15, actions_per_step=self.actions)
        self.assertNotEqual(
            [tuple(map(tuple, t[0])) for t in trace1],
            [tuple(map(tuple, t[0])) for t in trace2],
        )

    def test_a_validation_env_seed_survives_training_rollouts(self) -> None:
        """The fixed validation world must be rebuildable after arbitrary training."""
        validation = _rollout(self.timemap, seed=100007, steps=10, actions_per_step=self.actions)
        for seed in range(20):  # simulate 20 training episodes consuming randomness
            _rollout(self.timemap, seed=seed, steps=8, actions_per_step=self.actions)
            random.random()
            np.random.random()
        replay = _rollout(self.timemap, seed=100007, steps=10, actions_per_step=self.actions)
        self._compare(validation[1], replay[1])


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class TorchSeedingTests(unittest.TestCase):
    """``train.seed_all`` must touch every stream a rollout draws from."""

    def test_seed_all_pins_python_numpy_and_torch(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed")
        spec = importlib.util.spec_from_file_location(
            "piloc_train_seed_probe", _PILOC_DIR / "train.py"
        )
        assert spec and spec.loader
        train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train)

        import torch

        train.seed_all(42)
        draws = (random.random(), float(np.random.random()), float(torch.rand(())))
        train.seed_all(42)
        again = (random.random(), float(np.random.random()), float(torch.rand(())))
        self.assertEqual(draws, again)


if __name__ == "__main__":
    unittest.main()
