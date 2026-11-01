# -*- coding: utf-8 -*-
"""Read-only locks for the ROUND72 final-acceptance errata (zero environment)."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r72_paths import DELIV


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


class TestErrata(unittest.TestCase):
    def setUp(self):
        self.e = load(DELIV / "ROUND72_ERRATA.json")

    def test_mode_split_counts(self):
        g, s = self.e["correction_1_mode_split"]["greedy"], self.e["correction_1_mode_split"]["sampled"]
        self.assertEqual((g["n_free"], g["early"], g["late"]), (549, 402, 147))
        self.assertEqual((s["n_free"], s["early"], s["late"]), (549, 402, 147))
        self.assertEqual(g["cells"], 96)
        self.assertEqual((g["steps_equal"], g["features_equal"], g["actions_equal"]), (96, 96, 96))
        self.assertEqual(g["action_flips"], 0)
        self.assertEqual(g["argmax_flips"], 0)
        self.assertEqual(s["cells"], 96)
        self.assertEqual(s["features_equal"], 83)
        self.assertEqual(s["action_flips"], 14)
        self.assertEqual(s["argmax_flips"], 0)
        self.assertLess(s["features_equal"], 96)

    def test_budget_measured_zero_all_splits(self):
        b = self.e["correction_2_budget"]
        self.assertEqual(b["budget_bytes"], 29300)
        self.assertTrue(b["measured_zero_not_missing"])
        for name, sp in b["splits"].items():
            self.assertEqual(sp["budget_bytes"], [29300], name)
            self.assertEqual(sp["budget_refused_sends"], 0, name)
            self.assertEqual(sp["capacity_refused_sends"], 0, name)
            self.assertEqual(sp["rows_with_refusal_sequence"], 0, name)
            self.assertEqual(sp["collector_masked_budget_or_capacity"], 0, name)
            self.assertEqual(sp["missing_fields"], [], name)

    def test_teacher_billing_registered(self):
        c = self.e["correction_3_teacher_billing"]
        self.assertEqual(c["offline_billing_rows"]["value"], 70693)
        self.assertEqual(c["teacher_original_scale"]["joint"], 32046)
        self.assertEqual(c["teacher_original_scale"]["target"], 58928)
        self.assertEqual(c["attribution_gap"]["value"], 11765)
        self.assertEqual(c["r70_fit_source"], "1300000..1300047")

    def test_margin_framing_softened(self):
        self.assertIn("does not rule out", self.e["correction_4_margin_framing"]["restated"])

    def test_consumption_zero_and_head(self):
        for k, v in self.e["consumption_this_session"].items():
            self.assertEqual(v, 0, k)
        self.assertEqual(self.e["evidence_head"], "fe53d476efb56baaae194d68d7f749a546b14389")
        self.assertTrue(all("sha256" in h for h in self.e["source_hashes"].values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)