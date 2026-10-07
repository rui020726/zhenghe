"""Pure-computation tests for the ROUND47 analysis (no environments, read-only)."""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent
SPEC = importlib.util.spec_from_file_location("r47_analysis", HERE / "r47_analysis.py")
assert SPEC is not None and SPEC.loader is not None
R47 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = R47
SPEC.loader.exec_module(R47)


class DecompositionIdentityTests(unittest.TestCase):
    def test_identity_holds_on_synthetic_numbers(self):
        ppo = {"t1": {"cost_bytes": 500}, "t2": {"cost_bytes": 300}}
        bench = {"t2": {"cost_bytes": 300}, "t3": {"cost_bytes": 700}}
        a = sum(v["cost_bytes"] for k, v in ppo.items() if k not in bench)
        b = sum(v["cost_bytes"] for k, v in bench.items() if k not in ppo)
        c = sum(ppo[k]["cost_bytes"] - bench[k]["cost_bytes"] for k in set(ppo) & set(bench))
        delta = sum(v["cost_bytes"] for v in ppo.values()) - sum(v["cost_bytes"] for v in bench.values())
        self.assertEqual(delta, a - b + c)

    def test_bootstrap_is_deterministic_for_the_frozen_seed(self):
        first = R47.paired_bootstrap([1.0] * 8 + [2.0] * 8)
        second = R47.paired_bootstrap([1.0] * 8 + [2.0] * 8)
        self.assertEqual(first, second)
        self.assertEqual(first["index_seed"], 20261007)


class AccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recheck = json.loads((HERE / "ACCOUNTING_RECHECK.json").read_text(encoding="utf-8"))

    def test_rebuilt_totals_match_the_expected_closure(self):
        totals = self.recheck["rebuilt_totals"]
        self.assertEqual(totals["total_claims"], 482)
        self.assertEqual(totals["total_completed_steps"], 482000)
        self.assertEqual(totals["total_environment_constructor_calls"], 482)
        self.assertEqual(totals["technical_failed_claims_total"], 4)
        self.assertEqual(totals["adopted_cells"], 3)

    def test_completed_steps_finding_is_correct(self):
        finding = next(f for f in self.recheck["findings"] if f["field"] == "completed_steps")
        self.assertEqual(finding["verdict"], "correct")

    def test_per_claim_steps_all_positive_thousand(self):
        for claim in self.recheck["rebuilt_from"]["post_training_claims"]:
            self.assertEqual(claim["completed_steps"], 1000, claim["path"])
            self.assertLessEqual(claim["environment_constructor_calls"], 1)


class BaselineIdentityTests(unittest.TestCase):
    def test_late_frozen_mean_4838_25_is_recomputed_from_real_rule_rows(self):
        immediate, late = R47.load_baseline()
        mean = sum(int(row["spent_bytes"]) for row in late.values()) / len(late)
        self.assertEqual(mean, 4838.25)
        self.assertEqual(len(late), 16)
        self.assertEqual(len(immediate), 16)

    def test_bc_rows_are_48_greedy_controls_with_stable_identity(self):
        rows, identities = R47.load_bc_rows()
        for seed in R47.SEEDS:
            self.assertEqual(len(rows[seed]), 16)
            self.assertEqual(len(identities[seed]), 1)


class ComparisonCrossCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparison = json.loads((HERE / "BC_TO_PPO_COMPARISON.json").read_text(encoding="utf-8"))

    def test_report_numbers_match_the_machine_json(self):
        for seed in ("11", "22", "33"):
            block = self.comparison["per_seed"][seed]
            self.assertEqual(round(block["ppo_greedy"]["mean_spent_bytes"], 2),
                             {("11"): 5778.00, ("22"): 4694.25, ("33"): 5621.62}[seed])
            self.assertEqual(block["bc_greedy"]["mean_spent_bytes"], 5152.125)
            self.assertTrue(block["answers"]["ppo_preserves_exact_target_sets_vs_immediate"])

    def test_s22_ci_contains_zero(self):
        ci = self.comparison["per_seed"]["22"]["paired_bytes_ppo_minus_late"]["bootstrap"]
        self.assertLessEqual(ci["ci_low_linear"], 0.0)
        self.assertGreaterEqual(ci["ci_high_linear"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
