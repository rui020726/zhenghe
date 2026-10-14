# -*- coding: utf-8 -*-
"""ROUND75 S0 final-prep counterexamples (request R75_S0_FINAL_PREP_20261014).

Pure stdlib. Zero torch, zero model, zero NN, zero optimizer, zero PPO, zero environment,
zero canonical-ledger writes: the metering proof uses the SEPARATE fixture interface and the
REAL r75_guard meter / the REAL _meter_delta helper extracted from r75_frozen.py by AST.

Two REAL source defects, each with a counterexample that FAILS on the unfixed source:

  D1  r75_native_ppo_update returned
          "approx_kl": round(totals["approx_kl"] / minibatches, 6) ...
      while the accumulator is initialised as "kl" (line 303) and accumulated as
      totals["kl"] (line 448). totals["approx_kl"] is never created, so any update that
      completes at least one minibatch raises KeyError. The r66 reference
      (r66_frozen.py:875) reads totals["kl"]; the field NAME "approx_kl" is kept and only the
      accumulator it reads is restored. This is a transcription drift already present in the
      S0 delivery at 0f8574a (line 411 there), not new work of this round.

  D2  epochs_meter = _meter_delta(alignment_meter, meter.snapshot()).
      _meter_delta(before, after) = after - before is CORRECT; the CALL is wrong because it
      passes the alignment DELTA as "before" and a CUMULATIVE as "after". With a non-zero
      pre_meter the epoch phase is over-counted: pre=100, alignment_end=120, final=170 gives
      alignment=20, total=70, epochs must be 50, the old call yields 170-20=150. The fix
      captures the cumulative snapshot at the end of the alignment phase and differences from
      it, so alignment + epochs == total for EVERY component.
"""
from __future__ import annotations

import ast
import shutil
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r75_guard as guard_mod                      # noqa: E402  (stdlib-only reuse chain)

SELFTEST = HERE / "_s0_final_prep_selftest"
FROZEN_SRC = (HERE / "r75_frozen.py").read_text(encoding="utf-8")
R66_SRC = (HERE.parent / "round66_bc_early_path_preserving_late_adaptation"
           / "r66_frozen.py").read_text(encoding="utf-8")


# ------------------------------------------------------------------ AST helpers ----------
def _tree(source: str) -> ast.Module:
    return ast.parse(source)


def _function(source: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(_tree(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _segment(source: str, node: ast.AST) -> str:
    return ast.get_source_segment(source, node)


def _is_totals_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "totals"


def _string_key(node: ast.Subscript):
    slice_node = node.slice
    if isinstance(slice_node, ast.Index):            # pragma: no cover - py<3.9 shape
        slice_node = slice_node.value
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        return slice_node.value
    return None


def _totals_access(func: ast.FunctionDef) -> tuple[set, set, set]:
    """(defined, written, read) string keys of the `totals` accumulator inside func."""
    defined: set[str] = set()
    written: set[str] = set()
    read: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if _is_totals_name(target) and isinstance(node.value, ast.Dict):
                    for key in node.value.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            defined.add(key.value)
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Subscript):
            if _is_totals_name(node.target.value):
                key = _string_key(node.target)
                if key is not None:
                    written.add(key)
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            if _is_totals_name(node.value):
                key = _string_key(node)
                if key is not None:
                    read.add(key)
    return defined, written, read


def _returned_dict(func: ast.FunctionDef) -> ast.Dict:
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            return node.value
    raise AssertionError("no dict literal returned by the function")


def _returned_value_keys(func: ast.FunctionDef, field: str) -> set:
    """String keys of `totals` read by the value expression of a returned field."""
    table = _returned_dict(func)
    for key, value in zip(table.keys, table.values):
        if isinstance(key, ast.Constant) and key.value == field:
            found: set[str] = set()
            for node in ast.walk(value):
                if isinstance(node, ast.Subscript) and _is_totals_name(node.value):
                    name = _string_key(node)
                    if name is not None:
                        found.add(name)
            return found
    raise AssertionError(f"returned field {field!r} not found")


def _simple_assign(func: ast.FunctionDef, target: str):
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == target:
            return node
    return None


def _is_meter_snapshot_call(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "snapshot"
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "meter")


def _load_real_meter_delta():
    """The REAL _meter_delta + _METER_KEYS, extracted from r75_frozen.py and executed."""
    namespace: dict = {}
    for node in _tree(FROZEN_SRC).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_METER_KEYS" for t in node.targets):
            exec(compile(_segment(FROZEN_SRC, node), "<r75_frozen._METER_KEYS>", "exec"),
                 namespace)
    exec(compile(_segment(FROZEN_SRC, _function(FROZEN_SRC, "_meter_delta")),
                 "<r75_frozen._meter_delta>", "exec"), namespace)
    return namespace["_meter_delta"], namespace["_METER_KEYS"]


# ------------------------------------------------------------------ fake trunk -----------
class _Shape:
    def __init__(self, shape):
        self.shape = tuple(shape)


class _FakeTrunk:
    """nn.Module-like: __call__ dispatches to self.forward, which the meter wraps."""

    def __init__(self):
        self.calls = 0

    def __call__(self, batch):
        return self.forward(batch)

    def forward(self, batch):
        self.calls += 1
        return "hidden"


class _FakeInner:
    def __init__(self):
        self.trunk = _FakeTrunk()


class _FakeController:
    def __init__(self):
        self.inner = _FakeInner()


# ------------------------------------------------------------------ D1: totals keys ------
class TotalsKeyTests(unittest.TestCase):
    """The returned scientific numbers must read accumulators that actually exist."""

    @classmethod
    def setUpClass(cls):
        cls.func = _function(FROZEN_SRC, "r75_native_ppo_update")
        cls.defined, cls.written, cls.read = _totals_access(cls.func)

    def test_every_totals_read_is_a_defined_accumulator(self):
        missing = sorted(self.read - self.defined)
        self.assertEqual(missing, [],
                         "the update reads totals keys that are never initialised: "
                         f"{missing}; a non-empty batch raises KeyError at the return")

    def test_every_totals_write_is_a_defined_accumulator(self):
        missing = sorted(self.written - self.defined)
        self.assertEqual(missing, [], f"totals keys written but never defined: {missing}")

    def test_returned_approx_kl_reads_the_real_kl_accumulator(self):
        self.assertIn("approx_kl", {k.value for k in _returned_dict(self.func).keys
                                    if isinstance(k, ast.Constant)},
                      "the returned field name approx_kl must be preserved")
        self.assertEqual(_returned_value_keys(self.func, "approx_kl"), {"kl"},
                         "approx_kl must read the real kl accumulator (r66_frozen.py:875 "
                         "reads totals[\"kl\"]); reading a never-created key is a KeyError")

    def test_provenance_the_drift_is_a_transcription_error_not_new_math(self):
        """Read-only provenance: the r66 reference is self-consistent and reads totals["kl"].

        The drift is a transcription error already present in the S0 delivery (0f8574a line
        411); it is registered in the acceptance report, not re-derived from git history here.
        """
        r66_func = _function(R66_SRC, "frozen_early_ppo_update")
        r66_defined, r66_written, r66_read = _totals_access(r66_func)
        self.assertEqual(sorted(r66_read - r66_defined), [],
                         "the r66 reference reads only accumulators it defines")
        self.assertEqual(_returned_value_keys(r66_func, "approx_kl"), {"kl"},
                         "the r66 reference reads totals[\"kl\"]; R75 must match it")
        self.assertIn("kl", self.defined,
                      "R75 must keep the same accumulator name as the r66 reference")
        self.assertIn("kl", self.written)


# ------------------------------------------------------------------ D2: meter phases -----
class MeterPhaseTests(unittest.TestCase):
    """Per-phase trunk-entry deltas must be differences of CUMULATIVE snapshots."""

    @classmethod
    def setUpClass(cls):
        cls.func = _function(FROZEN_SRC, "r75_native_ppo_update")
        real_delta, keys = _load_real_meter_delta()
        cls.delta = staticmethod(real_delta)   # keep it unbound: it is not a method
        cls.keys = keys

    def test_epochs_meter_first_argument_is_a_cumulative_snapshot(self):
        assignment = _simple_assign(self.func, "epochs_meter")
        self.assertIsNotNone(assignment, "epochs_meter assignment not found")
        call = assignment.value
        self.assertIsInstance(call, ast.Call, "epochs_meter must be a _meter_delta call")
        self.assertEqual(getattr(call.func, "id", None), "_meter_delta")
        self.assertEqual(len(call.args), 2, "_meter_delta takes exactly (before, after)")
        before = call.args[0]
        self.assertTrue(not isinstance(before, ast.Name) or before.id != "alignment_meter",
                        "epochs_meter subtracts the alignment DELTA (alignment_meter) from a "
                        "CUMULATIVE snapshot; that over-counts the epoch phase whenever "
                        "pre_meter is non-zero")
        self.assertIsInstance(before, ast.Name,
                              "the epoch phase must start from a captured snapshot variable")
        source_of_before = _simple_assign(self.func, before.id)
        self.assertIsNotNone(source_of_before,
                             f"{before.id} is not assigned in the update function")
        self.assertTrue(_is_meter_snapshot_call(source_of_before.value),
                        f"{before.id} must be a meter.snapshot() cumulative, got "
                        f"{_segment(FROZEN_SRC, source_of_before.value)}")
        alignment_call = _simple_assign(self.func, "alignment")
        self.assertIsNotNone(alignment_call, "the alignment pass assignment was not found")
        self.assertGreater(source_of_before.lineno, alignment_call.lineno,
                           "the cumulative snapshot must be captured AFTER the alignment pass "
                           "(r66.likelihood_alignment) has run")
        self.assertTrue(_is_meter_snapshot_call(call.args[1]),
                        "the second argument must be the current cumulative meter.snapshot()")

    def test_phase_deltas_are_additive_with_nonzero_pre_meter(self):
        """Real meter + real helper: a previous update's consumption must not leak in."""
        shutil.rmtree(SELFTEST, ignore_errors=True)
        SELFTEST.mkdir(parents=True, exist_ok=True)
        try:
            resources = guard_mod.R75Resources.fixture(SELFTEST / "fx")
            controller = _FakeController()
            meter = resources.install_trunk_meter(controller, what_prefix="probe")
            trunk = controller.inner.trunk

            def call(shape):
                return trunk(_Shape(shape))

            # a PREVIOUS update on the same controller: 66 alignment rows + 4 epoch passes
            for _index in range(60):
                call((2, 19))
            for _index in range(6):
                call((8, 19))
            for _epoch in range(4):
                call((168, 19))
            pre_meter = meter.snapshot()
            self.assertGreater(pre_meter["rows_charged"], 0,
                               "this proof requires a NON-ZERO pre_meter (the bug trigger)")
            self.assertEqual(pre_meter["calls_charged"], 70)
            self.assertEqual(pre_meter["rows_charged"], 840)

            # the CURRENT update: alignment pass (66 calls / 168 rows) then 4 epochs
            for _index in range(60):
                call((2, 19))
            for _index in range(6):
                call((8, 19))
            alignment_end = meter.snapshot()
            for _epoch in range(4):
                call((168, 19))
            final = meter.snapshot()

            alignment = self.delta(pre_meter, alignment_end)
            epochs = self.delta(alignment_end, final)
            total = self.delta(pre_meter, final)
            for key in self.keys:
                self.assertEqual(alignment[key] + epochs[key], total[key],
                                 f"phase additivity broken for {key}: "
                                 f"alignment {alignment[key]} + epochs {epochs[key]} "
                                 f"!= total {total[key]}")
            self.assertEqual(alignment["calls_charged"], 66)
            self.assertEqual(alignment["rows_charged"], 168)
            self.assertEqual(epochs["calls_charged"], 4)
            self.assertEqual(epochs["rows_charged"], 672)
            self.assertEqual(total["calls_charged"], 70)
            self.assertEqual(total["rows_charged"], 840)

            # the OLD call shape (delta of a delta) is provably wrong on the same numbers:
            # it leaks the previous update's consumption into this update's epoch phase
            stale = self.delta(alignment, final)
            self.assertNotEqual(stale["rows_charged"], epochs["rows_charged"],
                                "subtracting the alignment delta from the cumulative snapshot "
                                "must not equal the epoch phase")
            self.assertEqual(stale["rows_charged"],
                             epochs["rows_charged"] + pre_meter["rows_charged"],
                             "the stale call over-counts the epoch phase by exactly the "
                             "pre-existing (previous update) consumption")

            # cross-update cumulative accounting is preserved: no uninstall, no double charge
            self.assertIs(getattr(controller.inner.trunk, "r75_trunk_meter", None), meter)
            snap = resources.snapshots()
            self.assertEqual(snap["offline_trunk_target_row"]["used"], final["rows_charged"],
                             "the durable counter must equal the cumulative meter rows")
            self.assertEqual(final["rows_charged"], 1680)
            self.assertEqual(final["calls_charged"], 140)
            resources.close()
        finally:
            shutil.rmtree(SELFTEST, ignore_errors=True)

    def test_native_update_never_uninstalls_the_meter(self):
        segment = _segment(FROZEN_SRC, self.func)
        self.assertIn("ensure_trunk_meter", segment,
                      "the update must reach the real entry through ensure_trunk_meter")
        self.assertNotIn("uninstall_trunk_meters", segment,
                         "uninstalling inside the update would drop cross-update accounting")


if __name__ == "__main__":
    unittest.main(verbosity=2)