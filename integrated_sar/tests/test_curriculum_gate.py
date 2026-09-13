"""CurriculumGate: the upgrade rule reads task metrics, never cumulative reward.

The old gate advanced after "200 episodes without a new all-time best cumulative
reward". With the implicit per-step cost that made the ladder a pure timer, and
comparing returns across different episode lengths was never meaningful. These
tests lock the replacement contract:

* the gate refuses any metric outside the task-metric whitelist (reward included);
* waiting long without meeting thresholds never advances the stage;
* the stage advances exactly once after the required consecutive passes;
* the window resets on stage change and on a failed validation;
* the curriculum can be switched off for a fixed episode limit.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PILOC_DIR = _REPO_ROOT / "piloc"

# src.curriculum is pure stdlib; only the package import path needs wiring.
if str(_PILOC_DIR) not in sys.path:
    sys.path.append(str(_PILOC_DIR))

from src.curriculum import CurriculumGate, CurriculumStage, default_stages  # noqa: E402


def _passing(**overrides):
    metrics = {
        "coverage": 0.6,
        "coverage_auc": 0.4,
        "discovery_rate": 0.8,
        "collision_rate": 0.01,
        "repeat_rate": 0.2,
    }
    metrics.update(overrides)
    return metrics


def _gate(**kwargs):
    options = dict(eval_every=5, required_passes=2)
    options.update(kwargs)
    return CurriculumGate(
        stages=default_stages([10, 20, 30]),
        **options,
    )


class GateRejectsRewardTests(unittest.TestCase):
    def test_cumulative_reward_is_not_an_allowed_input(self) -> None:
        gate = _gate()
        with self.assertRaises(ValueError):
            gate.submit_validation({"episode_return": 1234.0, "coverage": 0.6})

    def test_reward_named_keys_are_rejected_even_alongside_valid_metrics(self) -> None:
        gate = _gate()
        for key in ("reward", "return", "mean_reward_per_step", "cumulative_reward"):
            with self.assertRaises(ValueError, msg=key):
                gate.metrics_pass({key: 1.0, **_passing()})

    def test_the_gate_whitelist_contains_no_reward_metric(self) -> None:
        from src.curriculum import ALLOWED_METRICS

        joined = " ".join(ALLOWED_METRICS).lower()
        self.assertNotIn("reward", joined)
        self.assertNotIn("return", joined)


class AdvancementTests(unittest.TestCase):
    def test_no_advancement_without_passes_however_long_one_waits(self) -> None:
        gate = _gate()
        failing = _passing(coverage=0.1)  # below the 0.5 threshold
        for _ in range(100):  # 20 evaluation windows of pure waiting
            gate.register_episode()
            if gate.should_evaluate():
                self.assertFalse(gate.submit_validation(failing))
        self.assertEqual(gate.stage_index, 0)
        self.assertEqual(gate.episode_limit(), 10)

    def test_advances_exactly_once_after_required_consecutive_passes(self) -> None:
        gate = _gate(required_passes=2, eval_every=1)
        self.assertFalse(gate.submit_validation(_passing()))  # 1st pass
        self.assertTrue(gate.submit_validation(_passing()))   # 2nd pass -> advance
        self.assertEqual(gate.stage_index, 1)
        self.assertEqual(gate.episode_limit(), 20)
        # A third pass must not advance again without a fresh window of 2.
        self.assertFalse(gate.submit_validation(_passing()))
        self.assertEqual(gate.stage_index, 1)
        self.assertTrue(gate.submit_validation(_passing()))
        self.assertEqual(gate.stage_index, 2)

    def test_a_failed_validation_resets_the_consecutive_window(self) -> None:
        gate = _gate(required_passes=3, eval_every=1)
        gate.submit_validation(_passing())
        gate.submit_validation(_passing())
        gate.submit_validation(_passing(coverage=0.0))  # fail resets
        self.assertEqual(gate.stage_index, 0)
        self.assertFalse(gate.submit_validation(_passing()))  # needs 3 again
        gate.submit_validation(_passing())
        self.assertTrue(gate.submit_validation(_passing()))
        self.assertEqual(gate.stage_index, 1)

    def test_the_final_stage_never_advances_further(self) -> None:
        gate = _gate(required_passes=1)
        gate.submit_validation(_passing())
        self.assertEqual(gate.stage_index, 1)
        gate.submit_validation(_passing())
        self.assertEqual(gate.stage_index, 2)
        for _ in range(10):
            self.assertFalse(gate.submit_validation(_passing()))
        self.assertEqual(gate.stage_index, 2)
        self.assertEqual(gate.episode_limit(), 30)

    def test_mixed_thresholds_all_required(self) -> None:
        gate = _gate(required_passes=1)
        # discovery fine but collision above bound -> no pass
        self.assertFalse(gate.submit_validation(_passing(collision_rate=0.5)))
        # repeat above bound -> no pass
        self.assertFalse(gate.submit_validation(_passing(repeat_rate=0.9)))
        self.assertEqual(gate.stage_index, 0)


class EvaluationCadenceTests(unittest.TestCase):
    def test_eval_is_due_only_every_eval_every_episodes(self) -> None:
        gate = _gate(eval_every=5)
        due = []
        for episode in range(1, 13):
            gate.register_episode()
            if gate.should_evaluate():
                due.append(episode)
                gate.submit_validation(_passing(coverage=0.0))
        self.assertEqual(due, [5, 10])

    def test_the_window_resets_after_an_advancement(self) -> None:
        gate = _gate(eval_every=5, required_passes=1)
        for _ in range(5):
            gate.register_episode()
        self.assertTrue(gate.should_evaluate())
        self.assertTrue(gate.submit_validation(_passing()))
        self.assertEqual(gate.stage_index, 1)
        # Right after advancing, the next eval is again 5 episodes away.
        counter = 0
        while not gate.should_evaluate():
            gate.register_episode()
            counter += 1
        self.assertEqual(counter, 5)


class DisabledCurriculumTests(unittest.TestCase):
    def test_off_uses_the_fixed_limit_and_never_evaluates(self) -> None:
        gate = CurriculumGate(stages=None, fixed_limit=123)
        self.assertEqual(gate.episode_limit(), 123)
        self.assertFalse(gate.should_evaluate())
        for _ in range(10):
            gate.register_episode()
        self.assertFalse(gate.should_evaluate())
        self.assertFalse(gate.submit_validation(_passing()))

    def test_thresholds_on_unknown_metrics_are_refused_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            CurriculumStage(10, {"episode_return": 5.0})


class MetricSemanticsTests(unittest.TestCase):
    def test_higher_is_better_and_lower_is_better_are_not_swapped(self) -> None:
        # coverage/discovery: must be >= threshold. collision/repeat: must be <=.
        gate = _gate(required_passes=1)
        self.assertFalse(gate.metrics_pass(_passing(coverage=0.49)))
        self.assertTrue(gate.metrics_pass(_passing(coverage=0.5)))
        self.assertFalse(gate.metrics_pass(_passing(collision_rate=0.06)))
        self.assertTrue(gate.metrics_pass(_passing(collision_rate=0.05)))


if __name__ == "__main__":
    unittest.main()
