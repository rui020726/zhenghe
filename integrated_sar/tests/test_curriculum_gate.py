"""CurriculumGate: the upgrade rule reads task metrics, never cumulative reward.

The old gate advanced after "200 episodes without a new all-time best cumulative
reward". With the implicit per-step cost that made the ladder a pure timer, and
comparing returns across different episode lengths was never meaningful. These
tests lock the replacement contract:

* the gate refuses any metric outside the task-metric whitelist (reward included);
* waiting long without meeting thresholds never advances the stage;
* the stage advances exactly once after the required consecutive passes;
* the window resets on stage change and on a failed validation;
* coverage/discovery thresholds scale with each stage's budget share, so a
  10-step stage is never asked for the final stage's coverage (the first v2
  run sat 5000 episodes at stage 0 precisely because it was);
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
        "coverage_gain": 0.6,
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
        # Half the stage-0 bound: clearly below whatever the scaling asks for.
        failing = _passing(coverage_gain=default_stages([10, 20, 30])[0].thresholds["coverage_gain"] / 2)
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
        gate.submit_validation(_passing(coverage_gain=0.0))  # fail resets
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
        self.assertFalse(gate.submit_validation(_passing(collision_rate=0.9)))
        # repeat above bound -> no pass
        self.assertFalse(gate.submit_validation(_passing(repeat_rate=0.95)))
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
    def test_off_uses_the_fixed_limit_and_never_advances(self) -> None:
        gate = CurriculumGate(stages=None, fixed_limit=123)
        self.assertEqual(gate.episode_limit(), 123)
        for _ in range(10):
            gate.register_episode()
        self.assertFalse(gate.submit_validation(_passing()))
        self.assertEqual(gate.stage_index, 0)

    def test_off_keeps_the_evaluation_cadence_for_baseline_early_stop(self) -> None:
        # --curriculum off 的对照实验仍按 eval_every 跑验证与随机基线早停，
        # 只是不升级；评估节奏必须与 gate 模式一致。
        gate = CurriculumGate(stages=None, fixed_limit=123, eval_every=5)
        for _ in range(4):
            gate.register_episode()
        self.assertFalse(gate.should_evaluate())
        gate.register_episode()
        self.assertTrue(gate.should_evaluate())
        self.assertFalse(gate.submit_validation(_passing()))  # never advances

    def test_thresholds_on_unknown_metrics_are_refused_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            CurriculumStage(10, {"episode_return": 5.0})


class MetricSemanticsTests(unittest.TestCase):
    def test_higher_is_better_and_lower_is_better_are_not_swapped(self) -> None:
        # coverage_gain: must be >= threshold. collision/repeat: must be <=.
        # The bound is whatever stage 0 of this ladder actually asks for.
        gate = _gate(required_passes=1)
        bound = default_stages([10, 20, 30])[0].thresholds["coverage_gain"]
        self.assertFalse(gate.metrics_pass(_passing(coverage_gain=bound - 1e-9)))
        self.assertTrue(gate.metrics_pass(_passing(coverage_gain=bound)))
        self.assertFalse(gate.metrics_pass(_passing(collision_rate=0.51)))
        self.assertTrue(gate.metrics_pass(_passing(collision_rate=0.5)))


class StageScaledThresholdsTests(unittest.TestCase):
    """Regression for the stuck first v2 run: demanding the final stage's
    coverage (0.5) from a 10-step stage is geometrically impossible -- the
    validation plateaued at coverage ~0.06 / discovery 1/6 and the ladder
    never left stage 0 in 5000 episodes."""

    def test_coverage_gain_scales_with_the_budget_share(self) -> None:
        stages = default_stages([10, 20, 30])
        self.assertAlmostEqual(stages[0].thresholds["coverage_gain"], 0.5 * 10 / 30)
        self.assertAlmostEqual(stages[1].thresholds["coverage_gain"], 0.5 * 20 / 30)
        self.assertAlmostEqual(stages[2].thresholds["coverage_gain"], 0.5)

    def test_discovery_is_not_gated(self) -> None:
        # E 较短时 discovery 主要由目标出生位置运气决定；门控只看覆盖增益。
        for stage in default_stages([10, 20, 30]):
            self.assertNotIn("discovery_rate", stage.thresholds)

    def test_per_step_rates_are_not_scaled(self) -> None:
        for stage in default_stages([10, 20, 30]):
            self.assertEqual(stage.thresholds["collision_rate"], 0.5)
            self.assertEqual(stage.thresholds["repeat_rate"], 0.9)

    def test_production_stage0_is_reachable_by_the_plateaued_policy(self) -> None:
        # The stuck run's validation plateau: coverage ~0.06, discovery 1/6.
        # Stage 0 of the production ladder must sit below both, otherwise the
        # curriculum is blocked at E=10 again no matter how good the policy is.
        stage0 = default_stages(list(range(10, 251, 10)))[0].thresholds
        self.assertLess(stage0["coverage_gain"], 0.06)

    def test_an_intermediate_policy_advances_only_to_where_it_qualifies(self) -> None:
        gate = _gate(eval_every=1, required_passes=1)
        # coverage_gain 0.25 clears stage 0 (bound 0.5*10/30) but not stage 1
        # (bound 0.5*20/30): the ladder advances once, then holds.
        middle = _passing(coverage_gain=0.25)
        self.assertTrue(gate.submit_validation(middle))
        self.assertEqual(gate.stage_index, 1)
        self.assertFalse(gate.submit_validation(middle))
        self.assertEqual(gate.stage_index, 1)
        self.assertEqual(gate.episode_limit(), 20)

    def test_empty_limits_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            default_stages([])


if __name__ == "__main__":
    unittest.main()
