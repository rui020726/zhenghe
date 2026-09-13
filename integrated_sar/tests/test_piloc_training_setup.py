"""Tests for the PILOC training entry point's setup helpers.

``piloc/train.py`` imports ``MAPPO`` at module scope, which pulls in torch, and
``src.Map.timemap``, which pulls in gym, Pillow and matplotlib; some tests here build
that real environment, so the whole module is gated on all of those being present.
The helpers themselves are pure configuration and bookkeeping: they are what make a
run reproducible and its artefacts identifiable, which is why they are worth testing
separately from a training run that takes hours.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PILOC_DIR = _REPO_ROOT / "piloc"


def _missing_dependency() -> str | None:
    for name in ("torch", "gym", "PIL", "matplotlib"):
        if importlib.util.find_spec(name) is None:
            return f"{name} is not installed"
    return None


_SKIP_REASON = _missing_dependency()
_MODULE: Any = None


def _load_train_module() -> Any:
    """Import ``piloc/train.py`` by path, the way the integration layer loads PILOC."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    for entry in (_PILOC_DIR, _PILOC_DIR / "src" / "DDQN"):
        if str(entry) not in sys.path:
            sys.path.append(str(entry))
    spec = importlib.util.spec_from_file_location(
        "piloc_train_under_test", _PILOC_DIR / "train.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE = module
    return module


class _FakeSearchEnv:
    """Just enough of ``timemap.Environment`` for the metrics accumulator."""

    def __init__(self, obstacles: np.ndarray, agents_pos: list, goals_pos: list) -> None:
        self.map = obstacles
        self.agents_pos = agents_pos
        self.goals_pos = goals_pos


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class CheckpointCadenceTests(unittest.TestCase):
    """The episode-0 bug rewrote a checkpoint on every step of the first episode."""

    def setUp(self) -> None:
        self.train = _load_train_module()

    def test_episode_zero_never_saves(self) -> None:
        self.assertFalse(self.train.should_save_checkpoint(0, 1000))
        self.assertFalse(self.train.should_save_checkpoint(0, 1))

    def test_saves_only_on_a_multiple_of_the_interval(self) -> None:
        self.assertFalse(self.train.should_save_checkpoint(999, 1000))
        self.assertTrue(self.train.should_save_checkpoint(1000, 1000))
        self.assertTrue(self.train.should_save_checkpoint(2000, 1000))
        self.assertFalse(self.train.should_save_checkpoint(2001, 1000))

    def test_a_non_positive_interval_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.train.should_save_checkpoint(10, 0)


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class DeviceResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = _load_train_module()

    def test_cpu_is_always_available(self) -> None:
        self.assertEqual(self.train.resolve_device("cpu").type, "cpu")

    def test_auto_falls_back_to_cpu_without_a_gpu(self) -> None:
        import torch

        if torch.cuda.is_available():
            self.skipTest("a GPU is present, so auto legitimately picks cuda")
        self.assertEqual(self.train.resolve_device("auto").type, "cpu")

    def test_asking_for_cuda_without_a_gpu_is_an_error_not_a_fallback(self) -> None:
        import torch

        if torch.cuda.is_available():
            self.skipTest("a GPU is present")
        with self.assertRaises(ValueError):
            self.train.resolve_device("cuda")

    def test_an_unknown_device_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.train.resolve_device("tpu")


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class OutputLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = _load_train_module()

    def test_output_directories_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "experiment"
            directories = self.train.prepare_output_dir(target)
            self.assertTrue(directories["checkpoints"].is_dir())
            self.assertTrue(directories["tb_logs"].is_dir())

    def test_run_meta_records_what_identifies_the_run(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = self.train.write_run_meta(
                directory,
                settings={"num_agents": 2, "gamma": 0.95},
                seed=7,
                device=torch.device("cpu"),
                dim_info={"agent_0": [121, 4]},
            )
            meta = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(meta["seed"], 7)
        self.assertEqual(meta["device"], "cpu")
        self.assertEqual(meta["dim_info"], {"agent_0": [121, 4]})
        self.assertEqual(meta["settings"]["num_agents"], 2)
        self.assertIn("torch_version", meta)
        # None outside a checkout, but the key must exist so a missing commit is visible.
        self.assertIn("git_commit", meta)

    def test_episode_is_read_back_from_the_checkpoint_name(self) -> None:
        self.assertEqual(
            self.train.episode_from_checkpoint_name("shared_mappo_model_21_9232.pth"), 9232
        )
        self.assertEqual(self.train.episode_from_checkpoint_name("something_else.pth"), 0)


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class SearchMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = _load_train_module()

    def test_metrics_follow_the_episode(self) -> None:
        obstacles = np.zeros((6, 6), dtype=np.int32)
        obstacles[0, 0] = 1  # one obstacle, 35 free cells
        env = _FakeSearchEnv(obstacles, [(2, 2), (4, 4)], [(3, 3), (5, 5), (1, 4)])

        metrics = self.train.SearchEpisodeMetrics()
        metrics.reset(len(env.goals_pos))

        # A step where nobody moves and nothing is found.
        metrics.observe(env, [])
        # One robot moves, still nothing found.
        env.agents_pos = [(2, 3), (4, 4)]
        metrics.observe(env, [])
        # A repeat mark, and one of the three targets disappears.
        env.goals_pos = [(5, 5), (1, 4)]
        metrics.observe(env, ["agent_0"])

        observed = {"agent_0": np.full((6, 6), 0.5, dtype=np.float32)}
        observed["agent_0"][2, 2:5] = 0.0  # three free cells observed out of 35
        report = metrics.as_dict(env, observed)

        self.assertEqual(report["targets_discovered"], 1)
        self.assertAlmostEqual(report["discovery_rate"], 1 / 3)
        self.assertEqual(report["first_discovery_step"], 3)
        self.assertEqual(report["episode_steps"], 3)
        self.assertEqual(report["travel"], 1)
        self.assertAlmostEqual(report["repeat_ratio"], 1 / 3)
        self.assertAlmostEqual(report["coverage"], 3 / 35)


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class CommandLineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = _load_train_module()

    def test_defaults_are_usable_and_overridable(self) -> None:
        parser = self.train.build_parser()
        defaults = parser.parse_args([])
        self.assertEqual(defaults.device, "auto")
        self.assertEqual(defaults.seed, 0)
        self.assertIsNone(defaults.resume)
        self.assertGreater(defaults.save_every, 0)

        parsed = parser.parse_args(
            ["--device", "cpu", "--seed", "11", "--episodes", "5", "--save-every", "2"]
        )
        self.assertEqual(parsed.device, "cpu")
        self.assertEqual(parsed.seed, 11)
        self.assertEqual(parsed.episodes, 5)
        self.assertEqual(parsed.save_every, 2)


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class SearchEnvironmentSettingsTests(unittest.TestCase):
    """The episode-boundary rebuild must keep the run's own settings.

    ``main`` builds the environment once before training and rebuilds it at every
    episode boundary. The rebuild used to pass ``num_agents`` alone, so a non-default
    ``--num-goals`` was dropped and every episode after the first silently fell back
    to ``configs.num_goals`` -- the run trained against a different world than the
    one ``write_run_meta`` recorded.
    """

    def setUp(self) -> None:
        self.train = _load_train_module()

    def _args(self, argv: list[str]) -> Any:
        import random

        random.seed(11)
        np.random.seed(11)
        return self.train.build_parser().parse_args(argv)

    def test_the_command_line_agents_and_goals_reach_the_environment(self) -> None:
        environment = self.train.build_search_environment(
            self._args(["--num-agents", "4", "--num-goals", "3"])
        )

        self.assertEqual(len(environment.agents_pos), 4)
        self.assertEqual(len(environment.goals_pos), 3)

    def test_the_defaults_are_the_configs_values(self) -> None:
        configs = self.train.configs

        environment = self.train.build_search_environment(self._args([]))

        self.assertEqual(len(environment.agents_pos), configs.num_agents)
        self.assertEqual(len(environment.goals_pos), configs.num_goals)

    def test_the_reward_term_flag_reaches_the_environment(self) -> None:
        environment = self.train.build_search_environment(
            self._args(["--track-reward-terms"])
        )

        self.assertTrue(environment.track_reward_terms)

    def test_reward_term_tracking_stays_off_unless_requested(self) -> None:
        environment = self.train.build_search_environment(self._args([]))

        self.assertFalse(environment.track_reward_terms)


if __name__ == "__main__":
    unittest.main()
