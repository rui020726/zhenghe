# -*- coding: utf-8 -*-
"""ROUND75 S0 acceptance counterexamples (decision-end review of 0f8574a).

Pure stdlib + the real frozen JSON through the real adapter + fake callers. Zero torch, zero
model, zero NN, zero optimizer, zero PPO, zero environment, zero canonical-ledger writes.

Each test targets one ACTUAL defect found in the S0 delivery:
1. the production ledger was not pinned to a canonical root and the native entry trusted a
   forgeable boolean;
2. the counter missed the technical category cap, validated nothing durable, and its
   consistency check only reported;
3. the trunk rows were pre-charged by caller-side aggregation around an opaque alignment call
   instead of being metered at the real entry, and (19,) counted as 19;
4. the four trainable modules were never explicitly trainable-ized;
5. the plan claimed joint records = 0 while two native updates re-read the same 66 records.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r75_guard as guard_mod                      # noqa: E402  (stdlib-only reuse chain)
import r75_replay_adapter as adapter_mod           # noqa: E402  (stdlib-only)

SELFTEST = HERE / "_s0_acceptance_selftest"
FROZEN_SRC = (HERE / "r75_frozen.py").read_text(encoding="utf-8")
R66_SRC = (HERE.parent / "round66_bc_early_path_preserving_late_adaptation"
           / "r66_frozen.py").read_text(encoding="utf-8")
GUARD_SHA = guard_mod.sha256_file(HERE / "r75_guard.py")


class _PatchedCanonical:
    """Point the module-level canonical root at a throwaway dir (test-only; the shipped
    constant is untouched, so the pinning logic under test is the real one)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._saved = None

    def __enter__(self) -> Path:
        self._saved = guard_mod.R75_CANONICAL_ROOT
        guard_mod.R75_CANONICAL_ROOT = self.root
        return self.root

    def __exit__(self, *args) -> None:
        guard_mod.R75_CANONICAL_ROOT = self._saved


class _Shape:
    def __init__(self, shape):
        self.shape = shape


class _FakeTrunk:
    """nn.Module-like: __call__ dispatches to self.forward, which the meter wraps."""

    def __init__(self, fail_on: int | None = None):
        self.calls = 0
        self.fail_on = fail_on

    def __call__(self, batch):
        return self.forward(batch)

    def forward(self, batch):
        self.calls += 1
        if self.fail_on == self.calls:
            raise RuntimeError("synthetic forward failure")
        return "hidden"


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function {name} not found")


class AcceptanceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        shutil.rmtree(SELFTEST, ignore_errors=True)
        SELFTEST.mkdir(parents=True)
        self.root = SELFTEST / "canonical"

    def tearDown(self) -> None:
        shutil.rmtree(SELFTEST, ignore_errors=True)


# ======================================================================================
# issue 1: pinned canonical root, separate non-production fixture, strict native entry
# ======================================================================================
class ProductionPinningTests(AcceptanceTestCase):

    def test_production_refuses_non_canonical_root(self):
        with self.assertRaises(guard_mod.R75LedgerError):
            guard_mod.R75Resources.production(SELFTEST / "not-canonical")

    def test_production_accepts_the_pinned_canonical_root_only(self):
        with _PatchedCanonical(self.root):
            res = guard_mod.R75Resources.production()
            self.assertEqual(res.mode, "production")
            self.assertTrue(res.snapshots()["pinned_to_canonical"])
            res.close()
            with self.assertRaises(guard_mod.R75LedgerError):
                guard_mod.R75Resources.production(SELFTEST / "elsewhere")

    def test_fixture_refuses_canonical_root_and_descendants(self):
        for bad in (guard_mod.R75_CANONICAL_ROOT, guard_mod.R75_CANONICAL_ROOT / "child"):
            with self.assertRaises(guard_mod.R75LedgerError):
                guard_mod.R75Resources.fixture(bad)

    def test_fixture_is_the_separate_non_production_interface(self):
        res = guard_mod.R75Resources.fixture(SELFTEST / "fixture")
        self.assertEqual(res.mode, "fixture")
        self.assertTrue(res.r75_fixture)
        self.assertFalse(res.snapshots()["pinned_to_canonical"])
        res.close()

    def test_native_entry_refuses_fixture_and_forged_objects(self):
        fixture = guard_mod.R75Resources.fixture(SELFTEST / "fixture")
        with self.assertRaises(guard_mod.R75LedgerError):
            guard_mod.assert_native_production_resources(fixture)
        with self.assertRaises(guard_mod.R75LedgerError):
            guard_mod.assert_native_production_resources(None)

        class Forged:
            r75_central_production = True

            def charge_physical_step(self, *a, **k):
                return 0
        with self.assertRaises(guard_mod.R75LedgerError):
            guard_mod.assert_native_production_resources(Forged())
        fixture.close()

    def test_native_entry_checks_the_guard_source_pin(self):
        with _PatchedCanonical(self.root):
            res = guard_mod.R75Resources.production()
            self.assertTrue(guard_mod.assert_native_production_resources(res))
            with self.assertRaises(guard_mod.R75LedgerError):
                guard_mod.assert_native_production_resources(
                    res, expected_guard_source_sha256="0" * 64)
            self.assertTrue(guard_mod.assert_native_production_resources(
                res, expected_guard_source_sha256=GUARD_SHA))
            res.close()

    def test_simulated_classmethod_is_refused(self):
        with self.assertRaises(guard_mod.R75LedgerError):
            guard_mod.R75Resources.simulated(SELFTEST / "x")

    def test_r75_frozen_pins_the_actual_guard_source_hash(self):
        match = re.search(r'EXPECTED_R75_GUARD_SOURCE_SHA256 = "([0-9a-f]{64})"', FROZEN_SRC)
        self.assertIsNotNone(match, "the native entry must pin the guard source identity")
        self.assertEqual(match.group(1), GUARD_SHA)


# ======================================================================================
# issue 2: category caps, durable-state validation, fail-closed reconciliation
# ======================================================================================
class CounterHardeningTests(AcceptanceTestCase):

    def _fixture(self):
        return guard_mod.R75Resources.fixture(SELFTEST / "fx")

    def test_technical_category_cap_enforced_below_the_hard_cap(self):
        res = self._fixture()
        for i in range(3):
            res.charge_physical_step(f"n{i}")
        res.pre_register_technical_attempt("att-1")
        for i in range(4):
            res.charge_physical_step(f"att-1_t{i}", category="technical")
        with self.assertRaises(guard_mod.R75QuotaExhausted):     # category cap 4, hard 16 NOT reached
            res.charge_physical_step("att-1_over", category="technical")
        self.assertEqual(res.physical_steps.used(), 7)
        res.close()

    def test_normal_cap_then_hard_cap(self):
        res = self._fixture()
        for i in range(12):
            res.charge_physical_step(f"n{i}")
        with self.assertRaises(guard_mod.R75QuotaExhausted):
            res.charge_physical_step("n13")
        self.assertEqual(res.physical_steps.used(), 12)
        res.pre_register_technical_attempt("att-2")
        for i in range(4):
            res.charge_physical_step(f"att-2_t{i}", category="technical")
        with self.assertRaises(guard_mod.R75QuotaExhausted):     # hard cap 16
            res.charge_physical_step("att-2_over", category="technical")
        res.close()

    def test_technical_without_preregistration_refused(self):
        res = self._fixture()
        with self.assertRaises(guard_mod.R75LedgerError):
            res.charge_physical_step("unregistered", category="technical")
        self.assertEqual(res.physical_steps.used(), 0)
        res.close()

    def test_corrupt_durable_states_fail_closed(self):
        mutations = {
            "missing_kind": lambda o: o.pop("kind"),
            "wrong_cap": lambda o: o.update(normal_cap=99),
            "negative_used": lambda o: o.update(used_normal=-1),
            "over_normal_cap": lambda o: o.update(used_normal=13),
            "over_hard_cap": lambda o: o.update(used_normal=13, used_reserve=13),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                shutil.rmtree(self.root, ignore_errors=True)
                with _PatchedCanonical(self.root):
                    res = guard_mod.R75Resources.production()
                    res.charge_physical_step("seed_charge")
                    res.close()
                    path = self.root / "r75_physical_optimizer_step_guard.json"
                    obj = json.loads(path.read_text(encoding="utf-8"))
                    mutate(obj)
                    path.write_text(json.dumps(obj), encoding="utf-8")
                    with self.assertRaises(guard_mod.R75LedgerError):
                        guard_mod.R75Resources.production()

    def test_missing_counter_with_prior_charges_fails_closed(self):
        with _PatchedCanonical(self.root):
            res = guard_mod.R75Resources.production()
            res.charge_physical_step("seed_charge")
            res.close()
            (self.root / "r75_physical_optimizer_step_guard.json").unlink()
            with self.assertRaises(guard_mod.R75LedgerError):
                guard_mod.R75Resources.production()

    def test_counter_ahead_of_ledger_fails_closed(self):
        with _PatchedCanonical(self.root):
            res = guard_mod.R75Resources.production()
            res.charge_physical_step("seed_charge")
            res.close()
            path = self.root / "r75_physical_optimizer_step_guard.json"
            obj = json.loads(path.read_text(encoding="utf-8"))
            obj["used_normal"] = 2                       # counter ahead of the ledger maximum
            path.write_text(json.dumps(obj), encoding="utf-8")
            with self.assertRaises(guard_mod.R75LedgerError):
                guard_mod.R75Resources.production()

    def test_low_backup_counter_fails_closed(self):
        with _PatchedCanonical(self.root):
            res = guard_mod.R75Resources.production()
            res.charge_physical_step("c1")
            res.charge_physical_step("c2")
            res.close()
            path = self.root / "r75_physical_optimizer_step_guard.json"
            obj = json.loads(path.read_text(encoding="utf-8"))
            obj["used_normal"] = 1                       # stale / low backup
            path.write_text(json.dumps(obj), encoding="utf-8")
            with self.assertRaises(guard_mod.R75LedgerError):
                guard_mod.R75Resources.production()

    def test_charge_never_refunds_and_restart_keeps_consumption(self):
        res = self._fixture()
        res.charge_physical_step("work_that_fails_afterwards")
        self.assertEqual(res.physical_steps.used(), 1)   # a failed work is never refunded
        res.close()
        res2 = guard_mod.R75Resources.fixture(SELFTEST / "fx")
        self.assertEqual(res2.physical_steps.used(), 1)  # restart does not clear consumption
        res2.close()

    def test_residual_lock_fails_closed_without_recovery(self):
        res = self._fixture()
        path = SELFTEST / "fx" / "r75_physical_optimizer_step_guard.json"
        guard = guard_mod.R75PhysicalStepGuard(path, res.ledger, mode="fixture", lock_timeout=0.05)
        lock = path.with_name(path.name + ".lock")
        lock.write_text("stale", encoding="utf-8")
        with self.assertRaises(guard_mod.R75LedgerError):
            guard.charge(1, "under_stale_lock")
        self.assertTrue(lock.is_file(), "a residual lock is never auto-recovered")
        res.close()

    def test_verify_or_fail_blocks_before_work_in_production(self):
        with _PatchedCanonical(self.root):
            res = guard_mod.R75Resources.production()
            res.charge_physical_step("seed_charge")
            report = res.verify_or_fail()
            self.assertTrue(report["consistent"], report)
            # tamper the durable counter behind the open instance's back
            path = self.root / "r75_physical_optimizer_step_guard.json"
            obj = json.loads(path.read_text(encoding="utf-8"))
            obj["used_normal"] = 2
            path.write_text(json.dumps(obj), encoding="utf-8")
            with self.assertRaises(guard_mod.R75LedgerError):
                res.verify_or_fail()                      # forced reconciliation blocks pre-work
            res.close()
            with self.assertRaises(guard_mod.R75LedgerError):
                guard_mod.R75Resources.production()       # and a fresh open fails closed too

    def test_consistency_check_reports_without_blocking(self):
        res = self._fixture()
        res.charge_physical_step("s1")
        report = res.consistency_check()
        self.assertFalse(report["blocking"])
        self.assertTrue(report["consistent"], report)
        res.close()

    def test_environment_construction_is_forbidden(self):
        res = self._fixture()
        with self.assertRaises(guard_mod.R75QuotaExhausted):
            res.charge_env_construction("any_env")
        snap = res.snapshots()["environment_construction"]
        self.assertEqual((snap["used"], snap["normal_cap"], snap["hard_cap"]), (0, 0, 0))
        res.close()

    def test_offline_rows_per_charge_cap_enforced(self):
        res = self._fixture()
        with self.assertRaises(guard_mod.R75QuotaExhausted):
            res.charge_offline_rows(1025, "too_big")
        self.assertEqual(res.offline_rows.used(), 0)
        res.charge_offline_rows(1024, "max_ok")
        self.assertEqual(res.offline_rows.used(), 1024)
        res.close()


# ======================================================================================
# issue 3: verified 19-D row counting + unified real-entry trunk meter
# ======================================================================================
class TrunkEntryMeterTests(AcceptanceTestCase):

    def test_count_target_vectors_is_the_verified_r74_rule(self):
        import r74_guard_prod
        self.assertIs(guard_mod.count_target_vectors, r74_guard_prod.count_target_vectors)
        self.assertEqual(guard_mod.count_target_vectors(_Shape((19,))), 1)
        self.assertEqual(guard_mod.count_target_vectors(_Shape((4, 5, 19))), 20)
        for bad in (_Shape((18,)), _Shape((5, 18)), _Shape(())):
            with self.assertRaises(guard_mod.R75LedgerError):
                guard_mod.count_target_vectors(bad)

    def test_meter_counts_actual_shape_and_precharges(self):
        res = guard_mod.R75Resources.fixture(SELFTEST / "fx")
        trunk = _FakeTrunk()
        meter = guard_mod.TrunkEntryMeter(trunk, res.offline_rows, what_prefix="t")
        self.assertEqual(trunk(_Shape((19,))), "hidden")
        self.assertEqual(trunk(_Shape((4, 5, 19))), "hidden")
        self.assertEqual(meter.rows_charged, 21)
        self.assertEqual(meter.rows_completed, 21)
        self.assertEqual(res.offline_rows.used(), 21)
        meter.uninstall()
        res.close()

    def test_meter_refuses_before_the_forward_runs(self):
        res = guard_mod.R75Resources.fixture(SELFTEST / "fx")
        trunk = _FakeTrunk()
        meter = guard_mod.TrunkEntryMeter(trunk, res.offline_rows, what_prefix="t")
        with self.assertRaises(guard_mod.R75LedgerError):
            trunk(_Shape((5, 18)))
        with self.assertRaises(guard_mod.R75QuotaExhausted):
            trunk(_Shape((1025, 19)))
        self.assertEqual(trunk.calls, 0)                 # neither refusal reached the forward
        self.assertEqual(meter.calls_refused, 1)
        self.assertEqual(res.offline_rows.used(), 0)     # refusals consume nothing
        meter.uninstall()
        res.close()

    def test_failed_forward_is_charged_but_never_counted_as_completed(self):
        res = guard_mod.R75Resources.fixture(SELFTEST / "fx")
        trunk = _FakeTrunk(fail_on=2)
        meter = guard_mod.TrunkEntryMeter(trunk, res.offline_rows, what_prefix="t")
        trunk(_Shape((3, 19)))
        with self.assertRaises(RuntimeError):
            trunk(_Shape((4, 19)))
        self.assertEqual(meter.rows_charged, 7)          # charged, never refunded
        self.assertEqual(meter.rows_completed, 3)        # NOT passed off as a completed forward
        self.assertEqual(meter.calls_failed, 1)
        meter.uninstall()
        res.close()

    def test_double_install_refused_and_uninstall_restores(self):
        res = guard_mod.R75Resources.fixture(SELFTEST / "fx")
        trunk = _FakeTrunk()
        meter = guard_mod.TrunkEntryMeter(trunk, res.offline_rows, what_prefix="t")
        with self.assertRaises(guard_mod.R75LedgerError):
            guard_mod.TrunkEntryMeter(trunk, res.offline_rows, what_prefix="t2")
        meter.uninstall()
        self.assertFalse(hasattr(trunk, "r75_trunk_meter"))
        self.assertEqual(trunk(_Shape((2, 19))), "hidden")
        self.assertEqual(trunk.calls, 1)
        res.close()

    def test_caller_side_aggregate_charges_are_gone(self):
        self.assertNotIn("charge_offline_rows", FROZEN_SRC)
        self.assertIn("resources.ensure_trunk_meter(", FROZEN_SRC)
        self.assertIn('"trunk_entry_meter"', FROZEN_SRC)
        self.assertIn('"batch_read"', FROZEN_SRC)


# ======================================================================================
# issue 4: explicit trainable flags (+ static evidence about the r66 constructor)
# ======================================================================================
class FreezeFlagTests(AcceptanceTestCase):

    def test_freeze_trunk_only_trainables_explicitly_before_freezing(self):
        segment = _function_source(FROZEN_SRC, "freeze_trunk_only")
        self.assertIn("for parameter in trainable:", segment)
        self.assertIn("parameter.requires_grad_(True)", segment)
        self.assertLess(segment.index("requires_grad_(True)"),
                        segment.index("requires_grad_(False)"))
        mask = _function_source(FROZEN_SRC, "assert_mask")
        self.assertIn("any(p.requires_grad for p in frozen)", mask)
        self.assertIn("any(not p.requires_grad for p in trainable)", mask)

    def test_r66_constructor_sets_no_requires_grad(self):
        for name in ("fresh_partitioned_controller", "_PartitionedInner"):
            segment = _function_source(R66_SRC, name) if name.startswith("fresh") else None
            if segment is None:
                tree = ast.parse(R66_SRC)
                segment = next(ast.get_source_segment(R66_SRC, node) for node in tree.body
                               if isinstance(node, ast.ClassDef) and node.name == name)
            self.assertNotIn("requires_grad", segment,
                             f"{name} must stay flag-free so the explicit R75 flags matter")


# ======================================================================================
# issue 5: joint-record accounting for the repeated native batch reads
# ======================================================================================
class BatchReadAccountingTests(AcceptanceTestCase):

    def test_joint_record_charged_before_the_native_read(self):
        self.assertLess(FROZEN_SRC.index("resources.charge_joint_record("),
                        FROZEN_SRC.index("samples = trainer.batch_samples(episodes)"))
        self.assertIn('"unique_input_records"', FROZEN_SRC)
        self.assertIn('"record_exposures"', FROZEN_SRC)

    def test_two_native_reads_of_the_same_records_are_accounted(self):
        run_root = HERE.parent
        row = run_root / "round58_gen2" / "raw" / "rows" / "v2_r44_train_s11_e0_s1300027_a1.json"
        tape = run_root / "round58_gen2" / "raw" / "logs" / "v2_r44_train_s11_e0_s1300027_a1_tape.jsonl"
        episode, facts = adapter_mod.load_qualification_batch(row, tape)
        unique_records = len(episode["collector"].records)
        self.assertEqual(unique_records, facts["records"])
        self.assertEqual(unique_records, 66)

        res = guard_mod.R75Resources.fixture(SELFTEST / "fx")
        for update_index in range(2):                    # A2 + A3 read the SAME batch
            res.charge_joint_record(1, what=f"r75_native_update_{update_index}_batch_read")
        snap = res.snapshots()
        self.assertEqual(snap["joint_record"]["used"], 2)
        self.assertEqual(snap["environment_construction"]["used"], 0)   # never env episodes
        res.close()
        self.assertEqual(2 * unique_records, 132)        # exposures, not new episodes


if __name__ == "__main__":
    unittest.main(verbosity=2)