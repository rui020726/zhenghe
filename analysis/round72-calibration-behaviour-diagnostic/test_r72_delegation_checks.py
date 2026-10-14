# -*- coding: utf-8 -*-
"""Read-only locks for the ROUND72 delegation close-out (zero environment).

Every test parses a persisted JSON artifact and asserts a fact that the close-out
report and machine verdict rely on. No environment, no model load, no forward,
no RNG, no optimizer: these are pure reads of on-disk evidence.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r72_paths import R68, R71, DELIV


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


class TestBoundary(unittest.TestCase):
    def setUp(self):
        self.b = load(DELIV / "ROUND72_BOUNDARY_AUDIT.json")
        self.prim = self.b["comparisons"]["PRIMARY_R68_TRAINED_VS_R71_NEW_EP128"]["overall"]

    def test_early_bit_identical_and_no_greedy_flip(self):
        self.assertEqual(self.prim["margin"]["early_max_abs_dp"], 0.0)
        self.assertEqual(self.prim["margin"]["argmax_flips"], 0)
        self.assertEqual(self.prim["n_free"], 1098)

    def test_late_moved_but_never_crossed(self):
        m = self.prim["margin"]
        self.assertEqual(m["late_n"], 294)
        self.assertEqual(m["late_n_dp_nonzero"], 294)
        self.assertLessEqual(m["late_max_encroachment_ratio"], 0.5)
        self.assertGreater(m["late_min_abs_margin_a"], 0.0)

    def test_sampled_flips_all_late(self):
        f = self.b["sampled_flips"]
        self.assertEqual(f["n"], 14)
        self.assertTrue(f["all_in_sampled_mode"])
        self.assertTrue(all(d["region"] == "late" for d in f["detail"]))


class TestRulesAndBudget(unittest.TestCase):
    def setUp(self):
        self.ra = load(R68 / "raw" / "ROUND68_RULE_CONTEXT.json")["rule_absolute"]

    def test_immediate_cost_far_below_hard_budget(self):
        self.assertEqual(self.ra["IMMEDIATE"]["spent_bytes_per_scene_mean"], 14356.5)
        self.assertEqual(self.ra["IMMEDIATE"]["on_time_total"], 183)
        self.assertLess(self.ra["IMMEDIATE"]["spent_bytes_per_scene_mean"], 29300)

    def test_always_hold_is_170_of_183_ablation(self):
        self.assertEqual(self.ra["ALWAYS_HOLD"]["on_time_total"], 170)
        self.assertEqual(self.ra["ALWAYS_HOLD"]["spent_bytes_total"], 0)
        self.assertEqual(self.ra["LATE_DISCOVERY_GATE"]["on_time_total"], 183)

    def test_greedy_late_mean_is_single_scene(self):
        self.assertEqual(round(5394 / 32, 4), 168.5625)


class TestAdvantageContract(unittest.TestCase):
    def test_live_gamma_gae_eta_not_the_095_default(self):
        ts = load(R71 / "ROUND71_CHECKPOINT_INDEX.json")["seeds"]["s11"]["trained_ep128"]["training_state"]
        self.assertEqual(ts["gamma"], 1.0)
        self.assertEqual(ts["gae_lambda"], 1.0)
        self.assertEqual(ts["eta_lambda"], 0.5)

    def test_cost_baseline_counterexample_is_nonzero(self):
        # A_cost = return - V with V ~ E[cost]; baseline containing expected cost
        # does not cancel the cost policy gradient: -c*p*(1-p) at p=0.58.
        self.assertAlmostEqual(-0.58 * (1 - 0.58), -0.2436, places=12)


class TestPrimaryResult(unittest.TestCase):
    def setUp(self):
        self.s = load(R71 / "raw" / "ROUND71_STATISTICS.json")["per_seed_mode"]

    def test_greedy_identical_to_r68_all_seeds(self):
        for c in ("11/greedy", "22/greedy", "33/greedy"):
            p = self.s[c]["primary"]
            self.assertEqual(p["task_delta_NEW_EP128_minus_R68_TRAINED_ALPHA_mean"], 0.0, c)
            self.assertEqual(p["cost_delta_NEW_EP128_minus_R68_TRAINED_ALPHA_B_per_scene"], 0.0, c)
            self.assertEqual(p["targets_lost_NEW_EP128_vs_R68_TRAINED_ALPHA_count"], 0, c)

    def test_sampled_directions_differ_and_not_all_variance(self):
        d = {c: self.s[c]["primary"]["cost_delta_NEW_EP128_minus_R68_TRAINED_ALPHA_B_per_scene"]
             for c in ("11/sampled", "22/sampled", "33/sampled")}
        self.assertLess(d["11/sampled"], 0.0)
        self.assertGreater(d["22/sampled"], 0.0)
        self.assertLess(d["33/sampled"], 0.0)
        self.assertEqual(self.s["33/sampled"]["primary"]["targets_lost_NEW_EP128_vs_R68_TRAINED_ALPHA_count"], 1)

    def test_bootstrap_index_shared_with_r68(self):
        r68 = load(R68 / "raw" / "ROUND68_STATISTICS.json")["bootstrap"]
        r71 = load(R71 / "raw" / "ROUND71_STATISTICS.json")["bootstrap"]
        self.assertEqual(r68["index_sha256"], r71["index_sha256"])


class TestCloseOutArtifacts(unittest.TestCase):
    def test_ledger_and_verdict_record_zero_consumption(self):
        led = load(DELIV / "ROUND72_RESOURCE_LEDGER.json")
        ver = load(DELIV / "ROUND72_MACHINE_VERDICT.json")
        for key, val in led["this_session_consumption"].items():
            if isinstance(val, int):
                self.assertEqual(val, 0, key)
        self.assertEqual(ver["session_side_effects"]["environment_constructions"], 0)
        self.assertTrue(ver["runtime_identity"]["r68_r71_runtime_identical"])
        self.assertEqual(ver["evidence_head"], "fe53d476efb56baaae194d68d7f749a546b14389")


if __name__ == "__main__":
    unittest.main(verbosity=2)