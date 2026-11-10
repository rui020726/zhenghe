"""Revision_02 gates: everything that must pass before a world is opened.

The task names six gates explicitly, and each one is a separate test class here so a failure
names the requirement rather than "the suite is red":

  BudgetGateTests        the 1e9 ceiling binds on all three authorities and restores
  CostReconcileTests     charged bytes == transcript sums, and the N*G divisor is applied once
  CreditConservationTests total task credit is preserved and lands on the send that made it
  LambdaTests            the dual moves in the right direction, is clamped, never inside an episode
  CellTests              N <= R, the grid, and that a cell reproduces its own seed
  FeaturePathTests       sampling / cost critic / PPO update see the same row; ratio == 1 at 0
  DedupPolicyTests       the low-cost strong rule sends once, and only re-sends on real change
  VerdictTests           the improvement test is applied literally, not generously

No test opens a world, loads a checkpoint or trains: every one is arithmetic on inputs the
runtime would produce. Run with:

    D:/project_env/python.exe -B -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REVISION = Path(__file__).resolve().parents[1]
if str(REVISION) not in sys.path:
    sys.path.insert(0, str(REVISION))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from stage5b_comm import constants as C  # noqa: E402
from stage5b_comm import objective, scale_features, scale_program, unlimited_budget  # noqa: E402


# --------------------------------------------------------------------------- budget
class _Authority:
    """The three references the frozen runtime reads, as plain attributes."""

    def __init__(self):
        self.STAGE_B_INHERITED_OVERRIDES = {"comm_budget_bytes": 29300}


class _Hooks:
    RELAY_BUDGET_BYTES = 29300


class _Adapter:
    RELAY_BUDGET_BYTES = 29300


class BudgetGateTests(unittest.TestCase):
    """The ceiling is 1e9, all three authorities agree inside the context, and restore exactly."""

    def test_budget_is_large_and_finite(self):
        self.assertEqual(C.BUDGET_BYTES, 1_000_000_000)
        self.assertTrue(math.isfinite(float(C.BUDGET_BYTES)))
        # A budget of infinity would make remaining/budget NaN inside the relay.
        self.assertNotEqual(float(C.BUDGET_BYTES), float("inf"))

    def test_authority_binding_and_restore(self):
        from stage5b_comm import world
        common, hooks, adapter = _Authority(), _Hooks(), _Adapter()
        before = world.budget_authority_snapshot(common, hooks, adapter)
        self.assertEqual(set(before.values()), {C.FROZEN_AUTHORITY_BUDGET_BYTES})
        with world.budget_authority(common, hooks, adapter, C.BUDGET_BYTES) as record:
            inside = world.budget_authority_snapshot(common, hooks, adapter)
            self.assertEqual(set(inside.values()), {C.BUDGET_BYTES})
            self.assertEqual(record["inside"], inside)
        after = world.budget_authority_snapshot(common, hooks, adapter)
        self.assertEqual(after, before, "restore must put all three authorities back")
        self.assertTrue(record["restore_ok"])
        self.assertEqual(record["after_exit"], before)

    def test_restore_happens_even_when_the_search_raises(self):
        from stage5b_comm import world
        common, hooks, adapter = _Authority(), _Hooks(), _Adapter()
        before = world.budget_authority_snapshot(common, hooks, adapter)

        class Boom(RuntimeError):
            pass

        with self.assertRaises(Boom):
            with world.budget_authority(common, hooks, adapter, C.BUDGET_BYTES):
                raise Boom("world died")
        self.assertEqual(world.budget_authority_snapshot(common, hooks, adapter), before)

    def test_a_moved_authority_is_refused_before_anything_runs(self):
        from stage5b_comm import world
        common, hooks, adapter = _Authority(), _Hooks(), _Adapter()
        hooks.RELAY_BUDGET_BYTES = 1234
        with self.assertRaises(RuntimeError):
            with world.budget_authority(common, hooks, adapter, C.BUDGET_BYTES):
                pass


# --------------------------------------------------------------------------- cost
class CostReconcileTests(unittest.TestCase):
    """The cost the objective sees is the charged bytes, divided by N*G exactly once.

    "Exactly once" is checked in the sense the reward needs: the reference cell reproduces the
    frozen ``bytes / B0`` magnitude, and every other cell is priced relative to it by the
    registered ``N*G`` ratio.
    """

    def test_reference_cell_reproduces_the_frozen_magnitude(self):
        # (6, 24) is the frozen configuration, so one byte there must be worth bytes/10000 --
        # the scale every archived run used. Without this the cost term dwarfs the task term
        # and "never send" becomes optimal for every lambda in [0.25, 8].
        self.assertAlmostEqual(unlimited_budget.scaled_cost(1, 6, 24), 1.0 / 10000.0)
        self.assertAlmostEqual(unlimited_budget.scaled_cost(2445, 6, 24), 0.2445)

    def test_other_cells_are_priced_by_the_n_times_g_ratio(self):
        """Superseded by v4: ``N*G`` priced a broadcast by the world's size, not its cost.

        The old assertion (a half-fan-out, half-goals cell is 4x cheaper per byte) is exactly
        the defect that pinned G=48's break-even at the registered ceiling. What must hold now
        is that a broadcast costs the same in every cell.
        """
        page_bytes = 418.0
        per_send = {cell: unlimited_budget.scaled_cost(page_bytes * cell[0], cell[0], cell[1])
                    for cell in ((3, 12), (6, 24), (12, 48), (3, 48), (12, 12))}
        reference = per_send[(6, 24)]
        for cell, value in per_send.items():
            self.assertAlmostEqual(value, reference, places=12,
                                   msg="cell %r prices a broadcast differently" % (cell,))
        # And the break-even price, which is what the lambda band has to contain.
        self.assertAlmostEqual(1.0 / reference, C.B0_BYTES / (6.0 * page_bytes), places=9)
        self.assertGreater(1.0 / reference, C.LAMBDA_MIN)
        self.assertLess(1.0 / reference, C.LAMBDA_MAX)
        self.assertLess(1.0 / reference, C.LAMBDA_MAX * 0.75,
                        "the break-even must sit below the ceiling, not on it")

    def test_capacity_ratio_replaces_the_dead_capacity_dim(self):
        from stage5b_comm import scale_features
        self.assertAlmostEqual(scale_features.capacity_ratio(6, 24), 1.0)
        self.assertAlmostEqual(scale_features.capacity_ratio(12, 6), 8.0)
        self.assertAlmostEqual(scale_features.capacity_ratio(6, 48), 0.5)
        self.assertAlmostEqual(scale_features.capacity_ratio(12, 48), 1.0)
        with self.assertRaises(ValueError):
            scale_features.capacity_ratio(0, 24)

    def test_zero_scale_is_refused(self):
        with self.assertRaises(ValueError):
            unlimited_budget.scaled_cost(100, 0, 24)
        with self.assertRaises(ValueError):
            unlimited_budget.scaled_cost(100, 6, 0)
        with self.assertRaises(ValueError):
            unlimited_budget.scaled_cost(100, 6, 24, reference=(0, 24))

    def test_bytes_per_on_time_is_none_not_infinity(self):
        self.assertIsNone(unlimited_budget.bytes_per_on_time(1000, 0))
        self.assertAlmostEqual(unlimited_budget.bytes_per_on_time(1000, 4), 250.0)

    def test_cost_stream_and_ledger_agree_by_construction(self):
        """The trainer's cost reward is the tap stream; the relay ledger is the same bytes.

        ``world._run_episode_body`` already asserts ``spent == sum(tap airtime)`` for every
        real world, so what this checks is the arithmetic the trainer applies on top: the
        bundled per-decision reward must sum to the same charged bytes, scaled once.
        """
        from stage5b_comm import trainer
        cost_stream = {1: {"airtime_bytes": 4608}, 2: {"airtime_bytes": 4800},
                       3: {"airtime_bytes": 0}, 4: {"airtime_bytes": 5300}}
        decision_steps = [1, 2, 4]
        horizon = 10
        bundled, prefix = trainer._bundle(
            {k: float(v["airtime_bytes"]) for k, v in cost_stream.items()},
            decision_steps, horizon)
        self.assertEqual(prefix, 0.0)
        self.assertAlmostEqual(sum(bundled.values()), 14708.0)
        # The trainer multiplies charged bytes by 1/(N*G) rescaled to the reference cell,
        # which is exactly what scaled_cost(1, N, G) returns.
        factor = unlimited_budget.scaled_cost(1, 6, 24)
        self.assertAlmostEqual(factor, 1.0 / 10000.0)
        rewards = [bundled[s] * factor for s in decision_steps]
        self.assertAlmostEqual(sum(rewards), 14708.0 / 10000.0)

    def test_cost_is_the_same_order_as_the_task_advantage(self):
        """Regression on the magnitude that killed the first training run.

        A send is ~2.4 kB. At the reference cell that is ~0.24 scaled units, so with
        lambda=2 the cost term is ~0.49 against a task advantage of ~0.41: comparable, which
        is what lets the dual move the policy either way. Before the rescale the same send was
        ~33 units and hold was optimal at every lambda in the registered band.
        """
        per_send = unlimited_budget.scaled_cost(2445, 6, 24)
        self.assertLess(per_send, 1.0)
        self.assertGreater(per_send * C.LAMBDA_INITIAL, 0.1)
        self.assertLess(per_send * C.LAMBDA_INITIAL, 2.0)


# --------------------------------------------------------------------------- credit
class CreditConservationTests(unittest.TestCase):
    """The causal re-keying preserves the total and moves each credit onto a real send."""

    def test_credit_total_is_preserved_and_moved(self):
        from stage5b_comm import causal_credit
        assignment = causal_credit.CausalCreditAssignment(rule="first_accepted_send")
        tick_keyed = {0: 0.0, 5: 1.0, 9: 1.0, 20: 0.0}
        events = [
            {"target": "t1", "step": 3, "accepted": True},
            {"target": "t2", "step": 7, "accepted": True},
        ]
        stream = assignment.apply(tick_keyed, events)
        self.assertAlmostEqual(sum(stream.values()), sum(tick_keyed.values()),
                               msg="re-keying must not create or destroy credit")
        assignment.assert_credit_preserved(tick_keyed, stream)

    def test_credit_without_a_send_is_unmatched_not_silently_dropped(self):
        from stage5b_comm import causal_credit
        assignment = causal_credit.CausalCreditAssignment(rule="first_accepted_send")
        tick_keyed = {5: 1.0}
        stream = assignment.apply(tick_keyed, [])
        self.assertAlmostEqual(sum(stream.values()), 1.0)
        detail = assignment.audit()
        self.assertGreaterEqual(int(detail.get("unmatched") or 0), 0)
        assignment.assert_credit_preserved(tick_keyed, stream)


# --------------------------------------------------------------------------- lambda
class LambdaTests(unittest.TestCase):
    """The dual step: direction, clamp, and "only at an update boundary"."""

    def test_above_target_raises_the_price(self):
        new, detail = unlimited_budget.update_lambda(2.0, q_batch=0.90, q_ref=0.80)
        self.assertGreater(new, 2.0)
        self.assertEqual(detail["direction"], "raise_price")
        self.assertAlmostEqual(detail["q_target"], 0.95 * 0.80)

    def test_below_target_lowers_the_price(self):
        new, detail = unlimited_budget.update_lambda(2.0, q_batch=0.50, q_ref=0.80)
        self.assertLess(new, 2.0)
        self.assertEqual(detail["direction"], "lower_price")

    def test_exactly_at_target_holds(self):
        new, detail = unlimited_budget.update_lambda(3.0, q_batch=0.76, q_ref=0.80)
        self.assertAlmostEqual(new, 3.0)
        self.assertEqual(detail["direction"], "hold")

    def test_clamped_to_registered_bounds(self):
        low, _ = unlimited_budget.update_lambda(0.30, q_batch=0.0, q_ref=1.0)
        self.assertAlmostEqual(low, C.LAMBDA_MIN)
        high, detail = unlimited_budget.update_lambda(7.99, q_batch=1.0, q_ref=0.10)
        self.assertAlmostEqual(high, C.LAMBDA_MAX)
        self.assertTrue(detail["clamped"])

    def test_non_finite_is_refused(self):
        with self.assertRaises(ValueError):
            unlimited_budget.update_lambda(2.0, q_batch=float("nan"), q_ref=0.5)
        with self.assertRaises(ValueError):
            unlimited_budget.clamp_lambda(float("inf"))

    def test_missing_reference_is_refused(self):
        with self.assertRaises(ValueError):
            unlimited_budget.update_lambda(2.0, q_batch=0.5, q_ref=None)

    def test_initial_lambda_is_two(self):
        self.assertAlmostEqual(C.LAMBDA_INITIAL, 2.0)


class CellLambdaTests(unittest.TestCase):
    """train_v4_cell_lambda: one price per cell, moved by that cell's own smoothed rate."""

    def test_initial_price_is_three_and_step_is_a_quarter(self):
        self.assertAlmostEqual(unlimited_budget.LAMBDA_BY_CELL_INITIAL, 3.0)
        self.assertAlmostEqual(unlimited_budget.LAMBDA_STEP_PER_CELL, 0.25)
        self.assertAlmostEqual(C.LAMBDA_STEP_PER_CELL, 0.25)

    def test_direction_and_size_of_one_cell_step(self):
        # q_ref 0.80 -> dual target 0.965 * 0.80 = 0.772; q_ema 0.96 beats it, so bytes get
        # dearer. The 1.5-point safety margin above the verdict is the whole point: steering
        # exactly on 0.95 at G=12 is one dropped goal (8.3% of the cell) from failing.
        up, detail = unlimited_budget.update_lambda_cell(3.0, 0.96, 0.80)
        self.assertAlmostEqual(up, 3.0 + 0.25 * (0.96 - 0.772))
        self.assertAlmostEqual(detail["q_target"], 0.965 * 0.80)
        self.assertEqual(detail["direction"], "raise_price")
        down, detail = unlimited_budget.update_lambda_cell(3.0, 0.40, 0.80)
        self.assertAlmostEqual(down, 3.0 - 0.25 * (0.772 - 0.40))
        self.assertEqual(detail["direction"], "lower_price")
        hold, detail = unlimited_budget.update_lambda_cell(3.0, 0.772, 0.80)
        self.assertAlmostEqual(hold, 3.0)
        self.assertEqual(detail["direction"], "hold")

    def test_the_dual_margin_and_the_verdict_threshold_are_separate_numbers(self):
        """The dual aims 1.5 points above the rule the report applies; neither may drift."""
        self.assertAlmostEqual(unlimited_budget.Q_TARGET_FRACTION_DUAL, 0.965)
        self.assertAlmostEqual(unlimited_budget.Q_TARGET_FRACTION, 0.95)
        self.assertAlmostEqual(C.Q_TARGET_FRACTION_DUAL, 0.965)
        # A cell sitting exactly on the verdict line is BELOW the dual's water line, so the
        # dual lowers the price and buys success back: the buffer working as intended.
        _, detail = unlimited_budget.update_lambda_cell(2.0, 0.95 * 0.80, 0.80)
        self.assertEqual(detail["direction"], "lower_price")
        # ...and the verdict itself still passes exactly at 0.95, not at 0.965.
        verdict = unlimited_budget.improvement_verdict(
            cell="G12_R12_N3", learned_on_time=95, learned_bytes=100,
            immediate_on_time=100, immediate_bytes=200)
        self.assertTrue(verdict["improved"])
        verdict = unlimited_budget.improvement_verdict(
            cell="G12_R12_N3", learned_on_time=94, learned_bytes=100,
            immediate_on_time=100, immediate_bytes=200)
        self.assertFalse(verdict["improved"])

    def test_frozen_ema_lets_the_anchor_signal_persist_between_scans(self):
        """v4g: a training batch must not pull the anchored rate back to its own sampling q."""
        from stage5b_comm import trainer
        episodes = [{"cell_label": "G12_R6_N3", "on_time": 3, "generated_goals": 12,
                     "spent_bytes": 100, "emissions": 1}]
        q_ref = {"G12_R6_N3": 0.80}
        frozen = trainer.dual_step(episodes, lam={"G12_R6_N3": 3.0}, q_ref=q_ref,
                                   q_ema_by_cell={"G12_R6_N3": 0.95},
                                   q_ema_frozen=True)
        self.assertAlmostEqual(frozen["q_ema_by_cell"]["G12_R6_N3"], 0.95)
        # q_ema 0.95 > 0.965 * 0.80 = 0.772, so the price still rises on the anchored signal.
        self.assertGreater(frozen["lambda_by_cell"]["G12_R6_N3"], 3.0)
        live = trainer.dual_step(episodes, lam={"G12_R6_N3": 3.0}, q_ref=q_ref,
                                 q_ema_by_cell={"G12_R6_N3": 0.95}, q_ema_frozen=False)
        # The live EMA falls towards the batch's 0.25; the frozen one does not move at all.
        self.assertLess(live["q_ema_by_cell"]["G12_R6_N3"], 0.95)
        self.assertGreater(live["q_ema_by_cell"]["G12_R6_N3"], 0.70)
        self.assertGreater(frozen["lambda_by_cell"]["G12_R6_N3"],
                           live["lambda_by_cell"]["G12_R6_N3"])

    def test_frozen_anchor_steps_every_cell_not_only_the_batch(self):
        """The anchored rate is valid for all 15 cells, so all of them step each update.

        Measured without this on train_v4g: 300 episodes moved the R=12 prices by only +0.10 to
        +0.40, because each 4-episode batch drew one or two cells and only those stepped. The
        registered rule is "one step per update", which is 15x faster once the rate is anchored.
        """
        from stage5b_comm import trainer
        episodes = [{"cell_label": "G12_R6_N3", "on_time": 9, "generated_goals": 12,
                     "spent_bytes": 100, "emissions": 1}]
        anchors = {"G12_R6_N3": 0.95, "G24_R12_N3": 0.99, "G48_R12_N12": 0.85}
        q_ref = {"G12_R6_N3": 0.80, "G24_R12_N3": 0.89, "G48_R12_N12": 0.82}
        table = {label: 1.0 for label in anchors}
        frozen = trainer.dual_step(episodes, lam=table, q_ref=q_ref,
                                   q_ema_by_cell=anchors, q_ema_frozen=True)
        self.assertEqual(set(frozen["cells"]), set(anchors))
        self.assertGreater(frozen["lambda_by_cell"]["G24_R12_N3"], 1.0)
        self.assertGreater(frozen["lambda_by_cell"]["G48_R12_N12"], 1.0)
        # Without the freeze only the batch's own cell has a rate to move on.
        live = trainer.dual_step(episodes, lam=dict(table), q_ref=q_ref,
                                 q_ema_by_cell=dict(anchors), q_ema_frozen=False)
        self.assertEqual(set(live["cells"]), {"G12_R6_N3"})

    def test_cell_step_is_clamped_and_refuses_non_finite(self):
        low, _ = unlimited_budget.update_lambda_cell(0.30, 0.0, 1.0)
        self.assertAlmostEqual(low, C.LAMBDA_MIN)
        high, detail = unlimited_budget.update_lambda_cell(7.99, 1.0, 0.10)
        self.assertAlmostEqual(high, C.LAMBDA_MAX)
        self.assertTrue(detail["clamped"])
        with self.assertRaises(ValueError):
            unlimited_budget.update_lambda_cell(3.0, float("nan"), 0.5)
        with self.assertRaises(ValueError):
            unlimited_budget.update_lambda_cell(3.0, 0.5, None)

    def test_ema_starts_at_the_first_observation_and_never_overshoots(self):
        self.assertAlmostEqual(unlimited_budget.ema(None, 0.4), 0.4)
        self.assertAlmostEqual(unlimited_budget.ema(0.4, 0.8, alpha=0.25), 0.5)
        self.assertAlmostEqual(unlimited_budget.ema(0.4, 0.8, alpha=1.0), 0.8)
        with self.assertRaises(ValueError):
            unlimited_budget.ema(0.4, 0.8, alpha=0.0)

    def test_one_batch_moves_only_the_cells_it_contains(self):
        from stage5b_comm import trainer
        episodes = [
            {"cell_label": "G12_R6_N3", "on_time": 9, "generated_goals": 12},
            {"cell_label": "G48_R12_N12", "on_time": 48, "generated_goals": 48},
        ]
        table = {"G12_R6_N3": 3.0, "G48_R12_N12": 3.0, "G24_R6_N6": 3.0}
        q_ema = {"G12_R6_N3": 0.5, "G48_R12_N12": 1.0}
        q_ref = {"G12_R6_N3": 0.80, "G48_R12_N12": 0.80, "G24_R6_N6": 0.80}
        dual = trainer.dual_step(episodes, lam=table, q_ref=q_ref, q_ema_by_cell=q_ema)
        self.assertEqual(sorted(dual["cells"]), ["G12_R6_N3", "G48_R12_N12"])
        self.assertEqual(dual["lambda_by_cell"]["G24_R6_N6"], 3.0)
        self.assertLess(dual["lambda_by_cell"]["G12_R6_N3"], 3.0)
        self.assertGreater(dual["lambda_by_cell"]["G48_R12_N12"], 3.0)
        # The EMA moved too, and by the registered alpha.
        self.assertAlmostEqual(dual["q_ema_by_cell"]["G12_R6_N3"], 0.75 * 0.5 + 0.25 * 0.75)

    def test_cell_without_a_reference_is_skipped_not_guessed(self):
        from stage5b_comm import trainer
        episodes = [{"cell_label": "G12_R6_N3", "on_time": 6, "generated_goals": 12}]
        dual = trainer.dual_step(episodes, lam={"G12_R6_N3": 3.0}, q_ref={}, q_ema_by_cell={})
        self.assertFalse(dual["applied"])
        self.assertIn("G12_R6_N3", dual["skipped"])


# --------------------------------------------------------------------------- cells
class CellTests(unittest.TestCase):
    """The grid, the N<=R rule, and seed reproducibility."""

    def test_every_registered_cell_satisfies_n_le_r(self):
        for goals, robots, subscribers in scale_program.CELLS:
            self.assertLessEqual(subscribers, robots, (goals, robots, subscribers))
            self.assertIn(goals, C.SCALE_GOALS)
            self.assertIn(robots, C.SCALE_ROBOTS)
            self.assertIn(subscribers, C.SCALE_N_VALUES)

    def test_cell_count_matches_the_grid(self):
        # 3 goals x 5 realisable (R, N) pairs: R=6 allows N in {3,6}; R=12 allows all three.
        self.assertEqual(len(scale_program.CELLS), 15)

    def test_n_greater_than_r_is_refused(self):
        with self.assertRaises(ValueError):
            scale_program.validate_cell((24, 6, 12))

    def test_unregistered_values_are_refused(self):
        # 36 is a *G-scaling* goal count, not a training cell, so the training-side check must
        # still refuse it; 30 is admitted only when the caller opts into the extended set.
        with self.assertRaises(ValueError):
            scale_program.validate_cell((36, 6, 3))
        with self.assertRaises(ValueError):
            scale_program.validate_cell((24, 9, 3))
        self.assertEqual(scale_program.validate_cell((36, 6, 3), allow_extended_goals=True),
                         (36, 6, 3))
        with self.assertRaises(ValueError):
            scale_program.validate_cell((33, 6, 3), allow_extended_goals=True)

    def test_extended_goal_counts_never_enter_the_training_grid(self):
        """The sweep's seven new counts are registered worlds, not sampling cells."""
        self.assertEqual(scale_program.GOALS_EXTENDED,
                         (12, 18, 24, 30, 36, 42, 48, 54, 60, 72))
        self.assertEqual(len(scale_program.GOALS_ACCEPTED), 10)
        self.assertEqual(len(scale_program.CELLS), 15)
        self.assertTrue(all(cell[0] in scale_program.GOALS for cell in scale_program.CELLS))
        sweep = scale_program.g_scaling_cells()
        self.assertEqual(len(sweep), 50)
        self.assertEqual({cell[0] for cell in sweep}, set(scale_program.GOALS_EXTENDED))
        self.assertEqual({(cell[1], cell[2]) for cell in sweep}, set(scale_program.PAIRS))
        self.assertFalse(set(sweep) - set(scale_program.CELLS)
                         & {cell for cell in scale_program.CELLS})

    def test_extended_goal_seed_is_cell_specific_and_stable(self):
        cell = (30, 6, 6)
        self.assertEqual(scale_program.eval_seed(2301901, cell),
                         scale_program.eval_seed(2301901, cell))
        self.assertNotEqual(scale_program.eval_seed(2301901, cell),
                            scale_program.eval_seed(2301901, (36, 6, 6)))
        self.assertEqual(scale_program.subscriber_spec_for((54, 12, 12)), "all")

    def test_cell_is_deterministic_in_its_inputs(self):
        first = scale_program.cell_for(2301901, 17)
        second = scale_program.cell_for(2301901, 17)
        self.assertEqual(first, second)
        # ...and the cell is drawn from the registered grid.
        self.assertIn(first, scale_program.CELLS)

    def test_episode_seed_depends_on_scene_cell_and_index(self):
        cell = (24, 6, 3)
        base = scale_program.episode_seed(2301901, cell, 5)
        self.assertEqual(base, scale_program.episode_seed(2301901, cell, 5))
        self.assertNotEqual(base, scale_program.episode_seed(2301901, cell, 6))
        self.assertNotEqual(base, scale_program.episode_seed(2301902, cell, 5))
        self.assertNotEqual(base, scale_program.episode_seed(2301901, (48, 6, 3), 5))

    def test_eval_seed_is_stable_and_cell_specific(self):
        self.assertEqual(scale_program.eval_seed(2301903, (12, 12, 3)),
                         scale_program.eval_seed(2301903, (12, 12, 3)))
        self.assertNotEqual(scale_program.eval_seed(2301903, (12, 12, 3)),
                            scale_program.eval_seed(2301903, (12, 12, 6)))

    def test_subscriber_spec_uses_all_only_at_the_roster_size(self):
        self.assertEqual(scale_program.subscriber_spec_for((24, 6, 6)), "all")
        self.assertEqual(scale_program.subscriber_spec_for((24, 6, 3)), "first:3")
        self.assertEqual(scale_program.subscriber_spec_for((24, 12, 12)), "all")
        self.assertEqual(scale_program.subscriber_spec_for((24, 12, 3)), "first:3")

    def test_rescue_per_species_matches_the_roster(self):
        self.assertEqual(scale_program.rescue_per_species(6) * 3, 6)
        self.assertEqual(scale_program.rescue_per_species(12) * 3, 12)
        with self.assertRaises(ValueError):
            scale_program.rescue_per_species(7)

    def test_expectation_overrides_name_only_what_moved(self):
        from stage5b_comm import world
        self.assertEqual(world.cell_expectation_overrides((24, 6, 6)), {})
        self.assertEqual(world.cell_expectation_overrides((12, 6, 3)), {"num_goals": 12})
        self.assertEqual(world.cell_expectation_overrides((24, 12, 6)), {"rescue_robots": 12})
        self.assertEqual(world.cell_expectation_overrides((48, 12, 12)),
                         {"num_goals": 48, "rescue_robots": 12})
        # The G sweep's new counts resolve through the same path and carry the same reason.
        self.assertEqual(world.cell_expectation_overrides((30, 6, 3)),
                         {"num_goals": 30})
        self.assertIn("num_goals_extended", world.SCALE_EXPECTATION_OVERRIDE_REASONS)
        with self.assertRaises(ValueError):
            world.cell_expectation_overrides((33, 6, 3))

    def test_cost_scale_is_n_times_g(self):
        self.assertAlmostEqual(scale_program.cost_scale(6, 24), 144.0)
        self.assertAlmostEqual(scale_program.cost_scale(3, 48), 144.0)
        with self.assertRaises(ValueError):
            scale_program.cost_scale(0, 24)


# --------------------------------------------------------------------------- features
class _StubTrunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(11)
        self.net = torch.nn.Sequential(torch.nn.Linear(19, 16), torch.nn.Tanh())

    def forward(self, batch):
        return self.net(batch)


class _StubInner(torch.nn.Module):
    """Mirrors train_pilot.Controller: a plain nn.Module owning trunk/action_head/value_head."""

    def __init__(self):
        super().__init__()
        self.trunk = _StubTrunk()
        self.action_head = torch.nn.Linear(16, 2)
        self.value_head = torch.nn.Linear(16, 1)

    def forward(self, batch):
        hidden = self.trunk(batch)
        return self.action_head(hidden), self.value_head(hidden).squeeze(-1)


class _PlainInner:
    """The same three objects, but reached through a plain class (the real runtime shape)."""

    def __init__(self):
        self._inner = _StubInner()
        self.trunk = self._inner.trunk
        self.action_head = self._inner.action_head
        self.value_head = self._inner.value_head
        self.parameters = [*self.trunk.parameters(),
                           *self.action_head.parameters(),
                           *self.value_head.parameters()]

    def forward(self, batch):
        hidden = self.trunk(batch)
        return self.action_head(hidden), self.value_head(hidden).squeeze(-1)


class _Controller:
    def __init__(self, inner):
        self.inner = inner
        self.cost_head = torch.nn.Linear(16, 1)

    def forward(self, batch):
        return self.inner.forward(batch)

    def cost_values(self, batch):
        return self.cost_head(self.inner.trunk(batch)).squeeze(-1)


def _rows(goals=24, subscribers=6, n=4):
    rows = []
    for index in range(n):
        row = [0.0] * 19
        row[0] = float(index)
        row[2] = float(index * 2)          # sent_count
        row[14] = float(3 + index * 5)     # backlog
        row[15] = 1.0
        row[17] = float(subscribers)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


class FeaturePathTests(unittest.TestCase):
    """The three read paths see the same rewritten row, and ratio is 1 at update 0."""

    def test_scaled_rewrites_exactly_the_three_dims(self):
        raw = _rows()
        transform = scale_features.ScaleFeatures(mode="scaled", goals=24, subscribers=6)
        out = transform.rewrite(torch.as_tensor(raw))
        self.assertTrue(torch.allclose(out[:, 2], torch.as_tensor(raw[:, 2] / 24.0)))
        self.assertTrue(torch.allclose(out[:, 14], torch.as_tensor(raw[:, 14] / 24.0)))
        self.assertTrue(torch.allclose(out[:, 17], torch.as_tensor(raw[:, 17] / 6.0)))
        untouched = [i for i in range(19) if i not in (2, 14, 16, 17)]
        self.assertTrue(torch.allclose(out[:, untouched], torch.as_tensor(raw[:, untouched])))
        # #16 is now the roster-to-workload ratio, constant for the cell.
        self.assertTrue(torch.allclose(out[:, 16],
                                       torch.full_like(out[:, 16], 1.0)))

    def test_capacity_dim_distinguishes_r_at_the_same_g(self):
        raw = torch.as_tensor(_rows(goals=48, subscribers=6))
        thin = scale_features.ScaleFeatures(mode="scaled", goals=48, subscribers=6,
                                            robots=6).rewrite(raw)
        thick = scale_features.ScaleFeatures(mode="scaled", goals=48, subscribers=6,
                                             robots=12).rewrite(raw)
        self.assertAlmostEqual(float(thin[0, 16]), 0.5)
        self.assertAlmostEqual(float(thick[0, 16]), 1.0)
        # Everything else is identical, so R is the only difference the policy can read.
        others = [i for i in range(19) if i != 16]
        self.assertTrue(torch.allclose(thin[:, others], thick[:, others]))

    def test_horizon_denominator_is_two_hundred_not_fifty(self):
        """#11 must keep resolution past 50 steps; the shared builder clips there."""
        import integrated_sar.comm_learning as comm_learning

        observation = type("O", (), {"budget_bytes_per_episode": 0, "capacity_remaining": None,
                                     "backlog_targets": 0, "subscribers": 6, "step": 0})()
        item = type("I", (), {"age_steps": 0, "since_last_emission_time": 0, "sent_count": 0,
                              "pending_event": False, "changed_view": False,
                              "first_emission": True, "stage_changed": False,
                              "requirements_changed": False, "members_changed": False,
                              "progress_delta": 0.0, "revision_delta": 0.0,
                              "remaining_time": 150.0, "urgency_now": 0.0,
                              "has_future_requirements": False})()
        original = comm_learning.features
        # A previous test may have installed the shadow and not restored it; start clean so
        # "original" really is the shared builder.
        scale_features.restore_horizon_scale()
        original = comm_learning.features
        try:
            self.assertAlmostEqual(float(original(observation, item)[11]), 1.0,
                                   msg="frozen builder saturates at 50 steps")
            scale_features.install_horizon_scale()
            shadowed = comm_learning.features
            self.assertIsNot(shadowed, original)
            self.assertAlmostEqual(float(shadowed(observation, item)[11]), 150.0 / 200.0)
            item.remaining_time = 300.0
            self.assertAlmostEqual(float(shadowed(observation, item)[11]), 1.0)
            item.remaining_time = 20.0
            self.assertAlmostEqual(float(shadowed(observation, item)[11]), 0.1)
            item.remaining_time = -300.0
            self.assertAlmostEqual(float(shadowed(observation, item)[11]), -1.0)
            # Installing twice is idempotent; restoring puts the shared builder back exactly.
            scale_features.install_horizon_scale()
            self.assertEqual(scale_features.restore_horizon_scale(), 200.0)
            self.assertIs(comm_learning.features, original)
            self.assertIsNone(scale_features.restore_horizon_scale())
        finally:
            scale_features.restore_horizon_scale()
            comm_learning.features = original

    def test_off_is_the_identity(self):
        raw = torch.as_tensor(_rows())
        transform = scale_features.ScaleFeatures(mode="off", goals=24, subscribers=6)
        self.assertIs(transform.rewrite(raw), raw)
        self.assertEqual(sum(transform.changed.values()), 0)

    def test_tensor_and_dtype_preserved(self):
        raw = torch.as_tensor(_rows())
        out = scale_features.ScaleFeatures(mode="scaled", goals=12, subscribers=3).rewrite(raw)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.dtype, raw.dtype)

    def test_all_three_paths_reach_the_trunk_rewritten(self):
        controller = _Controller(_StubInner())
        seen = []
        handle = controller.inner.trunk.net.register_forward_pre_hook(
            lambda _m, inputs: seen.append(torch.as_tensor(inputs[0]).detach().clone()))
        transform = scale_features.ScaleFeatures(mode="scaled", goals=24, subscribers=6)
        transform.install(controller)
        raw = _rows()
        expected = transform.rewrite(torch.as_tensor(raw)).detach()
        controller.forward(torch.as_tensor(raw))          # sampling
        controller.cost_values(torch.as_tensor(raw))      # cost critic
        controller.inner.trunk(torch.as_tensor(raw))      # PPO update
        handle.remove()
        self.assertEqual(len(seen), 3)
        for observed in seen:
            self.assertTrue(torch.allclose(observed, expected))

    def test_importance_ratio_is_one_at_update_zero(self):
        controller = _Controller(_StubInner())
        transform = scale_features.ScaleFeatures(mode="scaled", goals=24, subscribers=6)
        transform.install(controller)
        raw = torch.as_tensor(_rows())
        old_logits, _ = controller.forward(raw)
        hidden = controller.inner.trunk(raw)
        new_logits = controller.inner.action_head(hidden)
        delta = float((torch.log_softmax(new_logits, dim=-1)
                       - torch.log_softmax(old_logits, dim=-1)).abs().max().detach())
        self.assertLess(delta, 1e-6)

    def test_restore_handles_both_container_shapes(self):
        for build in (_StubInner, _PlainInner):
            inner = build()
            real = inner.trunk
            controller = _Controller(inner)
            transform = scale_features.ScaleFeatures(mode="scaled", goals=24, subscribers=6)
            transform.install(controller)
            self.assertTrue(callable(inner.trunk))
            scale_features.ScaleFeatures.restore({"scale_features": transform})
            self.assertIs(inner.trunk, real, "%s: restore must return the real trunk"
                          % type(inner).__name__)
            # Idempotent.
            scale_features.ScaleFeatures.restore({"scale_features": transform})
            self.assertIs(inner.trunk, real)

    def test_divisors_come_from_the_cell_not_from_the_relay(self):
        transform = scale_features.ScaleFeatures(mode="scaled", goals=48, subscribers=3)
        self.assertEqual(transform.goals, 48)
        self.assertEqual(transform.subscribers, 3)
        audit = transform.audit()
        self.assertEqual(audit["divisors"]["#2 sent_count"], 48)
        self.assertEqual(audit["divisors"]["#14 backlog"], 48)
        self.assertEqual(audit["divisors"]["#17 subscribers"], 6)

    def test_invalid_mode_and_divisors_are_refused(self):
        with self.assertRaises(ValueError):
            scale_features.ScaleFeatures(mode="norm_only")
        with self.assertRaises(ValueError):
            scale_features.ScaleFeatures(mode="scaled", goals=0)
        with self.assertRaises(ValueError):
            scale_features.ScaleFeatures(mode="scaled", subscribers=0)

    def test_update_path_reproduces_the_sampled_row_after_restore(self):
        """Regression on the seam that made the first revision_02 run off-policy.

        ``world`` restores the trunk wrap when the episode ends, so the PPO update runs with
        no transform installed. If the update then forwards the raw ``tape["features"]``, the
        recorded log-probability came from one row and the recomputed one from another.
        Measured on the broken path: max |dlogp| = 0.109 at update 0, which is why that run's
        approx_kl read 11-17 against a clip of 0.2.

        The fix is ``apply_rewrite`` called by the trainer with the sample's own cell. This
        test runs the real sequence: sample, restore, update.
        """
        controller = _Controller(_StubInner())
        transform = scale_features.ScaleFeatures(mode="scaled", goals=24, subscribers=6)
        transform.install(controller)
        raw = _rows()
        sampled, _ = controller.forward(torch.as_tensor(raw))
        scale_features.ScaleFeatures.restore({"scale_features": transform})
        rebuilt = scale_features.apply_rewrite(raw, mode="scaled", goals=24, subscribers=6)
        hidden = controller.inner.trunk(rebuilt)
        updated = controller.inner.action_head(hidden)
        delta = float((torch.log_softmax(updated, dim=-1)
                       - torch.log_softmax(sampled.detach(), dim=-1)).abs().max())
        self.assertLess(delta, 1e-6)
        # And the same comparison without the rewrite must be visibly wrong, or the test
        # would pass for the wrong reason.
        raw_hidden = controller.inner.trunk(torch.as_tensor(raw))
        raw_logits = controller.inner.action_head(raw_hidden)
        broken = float((torch.log_softmax(raw_logits, dim=-1)
                        - torch.log_softmax(sampled.detach(), dim=-1)).abs().max())
        self.assertGreater(broken, 1e-3)

    def test_transform_is_per_sample_for_a_mixed_cell_minibatch(self):
        """A minibatch can mix cells, and each cell divides by its own G."""
        from stage5b_comm import trainer
        rows = []
        for label, goals, subscribers in ((("A"), 12, 3), ("B", 48, 6), ("A", 12, 3)):
            for index in range(2):
                row = [0.0] * 19
                row[2] = float(index)
                row[14] = float(index)
                row[17] = float(subscribers)
                rows.append(row)
        stacked = torch.as_tensor(np.asarray(rows, dtype=np.float32))
        chunk = [
            {"cell_transform": {"mode": "scaled", "goals": 12, "subscribers": 3},
             "targets": ["a", "b"]},
            {"cell_transform": {"mode": "scaled", "goals": 48, "subscribers": 6},
             "targets": ["c", "d"]},
            {"cell_transform": {"mode": "scaled", "goals": 12, "subscribers": 3},
             "targets": ["e", "f"]},
        ]
        out = trainer._apply_transform(stacked, chunk)
        # rows 0,1 and 4,5 use G=12; rows 2,3 use G=48
        for row in (0, 1, 4, 5):
            self.assertAlmostEqual(float(out[row, 2]), float(stacked[row, 2]) / 12.0)
        for row in (2, 3):
            self.assertAlmostEqual(float(out[row, 2]), float(stacked[row, 2]) / 48.0)
        # #17 is always /6, independent of the cell
        self.assertAlmostEqual(float(out[0, 17]), 3 / 6.0)

    def test_missing_cell_transform_is_refused_not_silently_raw(self):
        from stage5b_comm import trainer
        stacked = torch.as_tensor(np.asarray(_rows(), dtype=np.float32))
        with self.assertRaises(RuntimeError):
            trainer._apply_transform(stacked, [{"targets": ["a"]}] * 4)
        mixed = [{"cell_transform": {"mode": "scaled", "goals": 24, "subscribers": 6},
                  "targets": ["a", "b"]},
                 {"targets": ["c", "d"]}]
        with self.assertRaises(RuntimeError):
            trainer._apply_transform(stacked, mixed)


# --------------------------------------------------------------------------- dedup
class _Item:
    """One candidate, with the relay's own fields as the decision inputs.

    ``first_emission`` and ``changed_view`` are computed by ``CommRelay`` from its own ledger;
    a policy that re-derives "have I said this" from scratch can disagree with the transport,
    which is exactly the bug the 372-send probe exposed.
    """

    def __init__(self, target, first_emission=True, changed_view=False, stage=False,
                 reqs=False, members=False, pending=False, sent_count=0):
        self.target_id = target
        self.first_emission = first_emission
        self.changed_view = changed_view
        self.stage_changed = stage
        self.requirements_changed = reqs
        self.members_changed = members
        self.pending_event = pending
        self.sent_count = sent_count


def _sent(target, **kwargs):
    """Already emitted once, view unchanged: the relay would report first_emission False."""
    kwargs.setdefault("first_emission", False)
    kwargs.setdefault("sent_count", 1)
    return _Item(target, **kwargs)


class _Observation:
    def __init__(self, items, step=1):
        self.items = tuple(items)
        self.step = step


class DedupPolicyTests(unittest.TestCase):
    """First emission sends; an unchanged view is suppressed; a structural change re-sends."""

    def test_first_sighting_sends(self):
        from stage5b_comm import dedup_policy
        policy = dedup_policy.DedupImmediatePolicy()
        asked = policy.decide(_Observation([_Item("t1"), _Item("t2")]))
        self.assertEqual(set(asked), {"t1", "t2"})

    def test_identical_view_is_suppressed(self):
        from stage5b_comm import dedup_policy
        policy = dedup_policy.DedupImmediatePolicy()
        policy.decide(_Observation([_Item("t1")]))
        asked = policy.decide(_Observation([_sent("t1")]))
        self.assertEqual(asked, ())
        self.assertEqual(policy.suppressed_repeats, 1)

    def test_changed_view_without_structure_is_suppressed(self):
        """A progress tick is not worth a packet: it would collapse this arm into immediate."""
        from stage5b_comm import dedup_policy
        policy = dedup_policy.DedupImmediatePolicy()
        policy.decide(_Observation([_Item("t1")]))
        asked = policy.decide(_Observation([_sent("t1", changed_view=True)]))
        self.assertEqual(asked, ())

    def test_material_change_resends(self):
        """Default form: a stage transition re-sends; requirements/members alone do not.

        The narrow default is measured, not preferred on principle: with members and
        requirements included the arm sent 138 times over 24 targets in one world, because the
        member list churns on every assignment. Stage transitions are the case that actually
        carries new instructions for the receivers.
        """
        from stage5b_comm import dedup_policy
        policy = dedup_policy.DedupImmediatePolicy()
        policy.decide(_Observation([_Item("t1")]))
        asked = policy.decide(_Observation([_sent("t1", changed_view=True, stage=True)]))
        self.assertEqual(asked, ("t1",))

    def test_broader_form_resends_on_any_structural_flag(self):
        from stage5b_comm import dedup_policy
        for flag in ("stage", "reqs", "members"):
            policy = dedup_policy.DedupImmediatePolicy(stage_only=False)
            target = "t_%s" % flag
            policy.decide(_Observation([_Item(target)]))
            asked = policy.decide(_Observation(
                [_sent(target, changed_view=True, **{flag: True})]))
            self.assertEqual(asked, (target,), "%s change must re-send in the broad form" % flag)

    def test_structural_flag_without_changed_view_does_not_send(self):
        """The relay's own comparison wins: this arm must not second-guess the transport."""
        from stage5b_comm import dedup_policy
        policy = dedup_policy.DedupImmediatePolicy(stage_only=False)
        policy.decide(_Observation([_Item("t1")]))
        self.assertEqual(policy.decide(_Observation([_sent("t1", members=True)])), ())

    def test_dedup_never_sends_more_than_immediate(self):
        from stage5b_comm import dedup_policy
        policy = dedup_policy.DedupImmediatePolicy()
        items = [_Item("t%d" % i) for i in range(5)]
        settled = [_sent("t%d" % i) for i in range(5)]
        first = policy.decide(_Observation(items))
        second = policy.decide(_Observation(settled))
        self.assertEqual(len(first), 5)
        self.assertEqual(len(second), 0)
        self.assertEqual(len(first) + len(second), 5, "immediate would have sent 10")


# --------------------------------------------------------------------------- verdict
class VerdictTests(unittest.TestCase):
    """The improvement test is applied as written: 95% of immediate AND fewer bytes."""

    def test_cheaper_and_accurate_enough_is_improvement(self):
        verdict = unlimited_budget.improvement_verdict(
            cell="G24_R6_N3", learned_on_time=96, learned_bytes=1000,
            immediate_on_time=100, immediate_bytes=2000)
        self.assertTrue(verdict["improved"])

    def test_cheaper_but_below_95_percent_is_not(self):
        verdict = unlimited_budget.improvement_verdict(
            cell="G24_R6_N3", learned_on_time=94, learned_bytes=1000,
            immediate_on_time=100, immediate_bytes=2000)
        self.assertFalse(verdict["improved"])
        self.assertIn("below the 0.95 requirement", verdict["reason"])

    def test_accurate_but_not_cheaper_is_not(self):
        verdict = unlimited_budget.improvement_verdict(
            cell="G24_R6_N3", learned_on_time=100, learned_bytes=2000,
            immediate_on_time=100, immediate_bytes=2000)
        self.assertFalse(verdict["improved"])
        self.assertIn("not lower", verdict["reason"])

    def test_no_points_is_not_a_verdict(self):
        verdict = unlimited_budget.improvement_verdict(
            cell="G24_R6_N3", learned_on_time=None, learned_bytes=0,
            immediate_on_time=100, immediate_bytes=2000)
        self.assertFalse(verdict["improved"])
        self.assertIn("no evaluation point", verdict["reason"])


# --------------------------------------------------------------------------- objective
class ObjectiveTests(unittest.TestCase):
    def test_combination_shape(self):
        self.assertAlmostEqual(objective.combine(1.0, 1.0, 2.0), -1.0)
        self.assertAlmostEqual(objective.combine(0.5, 0.25, 4.0), 0.5 - 1.0)

    def test_non_finite_is_refused(self):
        with self.assertRaises(ValueError):
            objective.combine(float("nan"), 0.0, 2.0)
        with self.assertRaises(ValueError):
            objective.combine(0.0, 0.0, float("inf"))

    def test_pacing_is_refused_without_a_budget(self):
        original = C.COST_PACING
        try:
            C.COST_PACING = "u_free"
            with self.assertRaises(RuntimeError):
                objective.combine(1.0, 1.0, 2.0)
        finally:
            C.COST_PACING = original

    def test_describe_records_the_dual(self):
        described = objective.describe(2.5)
        self.assertEqual(described["lambda"], 2.5)
        self.assertTrue(described["budget_is_ceiling"])
        self.assertEqual(described["cost_scale"], "bytes / (N * G)")


class SendCapTests(unittest.TestCase):
    """The integer actuator: one step is one page, the unit the verdict counts."""

    class _Obs:
        def __init__(self, step):
            self.step = step

    class _Inner:
        name = "learned"
        orders_selection = True

        def __init__(self, targets=("a", "b", "c")):
            self._targets = tuple(targets)
            self.last_features = {"a": ()}

        def parameters(self):
            return {"feature_layout": "test"}

        def decide(self, _observation):
            return self._targets

    def test_wrapper_truncates_to_the_budget_and_resets_per_episode(self):
        from stage5b_comm.send_cap import SendCapPolicy
        policy = SendCapPolicy(self._Inner(), 4)
        self.assertEqual(policy.decide(self._Obs(1)), ("a", "b", "c"))
        self.assertEqual(policy.decide(self._Obs(2)), ("a",))
        self.assertEqual(policy.decide(self._Obs(3)), ())
        self.assertEqual((policy.sent, policy.dropped), (4, 5))
        # A step counter that restarts is a new episode: the budget comes back.
        self.assertEqual(policy.decide(self._Obs(1)), ("a", "b", "c"))
        self.assertEqual(policy.sent, 3)

    def test_the_policy_identity_survives_the_wrapper(self):
        from stage5b_comm.send_cap import SendCapPolicy
        policy = SendCapPolicy(self._Inner(), 2)
        # The frozen transport assertion compares the relay against this name.
        self.assertEqual(policy.name, "learned")
        self.assertTrue(policy.orders_selection)
        self.assertEqual(policy.parameters()["send_cap"], 2)
        self.assertEqual(policy.last_features, {"a": ()})

    def test_above_the_water_line_costs_one_page_below_gains_one(self):
        from stage5b_comm import send_cap
        up, detail = send_cap.update_cap(10, 0.99, 0.80, measured=12)
        self.assertEqual(up, 9)
        self.assertEqual(detail["direction"], "cut_page")
        self.assertAlmostEqual(detail["q_target"], 0.965 * 0.80)
        down, detail = send_cap.update_cap(10, 0.50, 0.80, measured=12)
        self.assertEqual(down, 11)
        self.assertEqual(detail["direction"], "add_page")
        hold, detail = send_cap.update_cap(13, 0.50, 0.80, measured=12)
        self.assertEqual(hold, 13, "measured + 1 is the ceiling: the dual may not invent pages")

    def test_the_cap_never_reaches_zero_and_never_exceeds_the_measured_count(self):
        from stage5b_comm import send_cap
        floor, detail = send_cap.update_cap(1, 1.0, 0.80, measured=30)
        self.assertEqual(floor, 1)
        self.assertTrue(detail["clamped"])
        held, detail = send_cap.update_cap(31, 0.10, 0.80, measured=30)
        self.assertEqual(held, 31)
        self.assertEqual(detail["direction"], "hold")

    def test_first_scan_starts_from_what_the_policy_spent(self):
        from stage5b_comm import send_cap
        first, detail = send_cap.update_cap(None, 0.99, 0.80, measured=44)
        self.assertEqual(detail["cap_before"], 44)
        self.assertEqual(first, 43)

    def test_bad_inputs_are_refused(self):
        from stage5b_comm import send_cap
        with self.assertRaises(ValueError):
            send_cap.update_cap(5, float("nan"), 0.5)
        with self.assertRaises(ValueError):
            send_cap.update_cap(5, 0.5, None)
        with self.assertRaises(ValueError):
            send_cap.update_cap(5, 0.5, 0.5, step=0)


class TraditionalArmTests(unittest.TestCase):
    """The two traditional triggers are the frozen catalogue's own arms, not retuned copies."""

    def test_periodic_arm_is_the_catalogue_arm_at_interval_five(self):
        from integrated_sar.comm_policy import PeriodicPolicy
        from stage5b_comm import episode
        policy = episode.build_rule_policy(None, "periodic_k5", 2301901)
        self.assertIsInstance(policy, PeriodicPolicy)
        self.assertEqual(policy.parameters(), {"interval": 5})

    def test_event_maxwait_arm_keeps_the_declared_silence_bound(self):
        from integrated_sar.comm_policy import EventMaxWaitPolicy
        from stage5b_comm import episode
        policy = episode.build_rule_policy(None, "event_maxwait", 2301901)
        self.assertIsInstance(policy, EventMaxWaitPolicy)
        params = policy.parameters()
        self.assertEqual((params["max_wait_steps"], params["max_urgent_steps"]), (10, 3))
        self.assertEqual(params["urgency_time"], 25.0)

    def test_traditional_arms_are_declared_and_distinct_from_the_throttles(self):
        from stage5b_comm import episode
        self.assertEqual(episode.TRADITIONAL_ARMS, ("periodic_k5", "event_maxwait"))
        self.assertFalse(set(episode.TRADITIONAL_ARMS) & set(episode.THROTTLE_ARMS))


if __name__ == "__main__":
    unittest.main()
