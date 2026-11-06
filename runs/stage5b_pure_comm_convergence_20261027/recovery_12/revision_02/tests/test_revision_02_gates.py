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
        reference = unlimited_budget.scaled_cost(2400, 6, 24)
        # Half the fan-out and half the goals: four times cheaper per byte than the reference.
        self.assertAlmostEqual(unlimited_budget.scaled_cost(2400, 3, 12), reference * 4.0)
        # Double the fan-out and double the goals: four times dearer.
        self.assertAlmostEqual(unlimited_budget.scaled_cost(2400, 12, 48), reference / 4.0)

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
        with self.assertRaises(ValueError):
            scale_program.validate_cell((36, 6, 3))
        with self.assertRaises(ValueError):
            scale_program.validate_cell((24, 9, 3))

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
        untouched = [i for i in range(19) if i not in (2, 14, 17)]
        self.assertTrue(torch.allclose(out[:, untouched], torch.as_tensor(raw[:, untouched])))
        # #11 must not be touched by this revision.
        self.assertTrue(torch.allclose(out[:, 11], torch.as_tensor(raw[:, 11])))

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


if __name__ == "__main__":
    unittest.main()
