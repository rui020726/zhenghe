# -*- coding: utf-8 -*-
"""Read-only locks for the ROUND72 budget/refusal coverage receipt (zero environment).

Locks ROUND72_COVERAGE_RECEIPT.json: executor-refusal fields are present (not
merely defaulted) and measured zero on every row; the collector pre-mask channel
is covered+zero for learned+training and N/A for the immediate rule path; the
immediate 33-row composition is 32 dev-block scenes + 1 duplicate re-attempt.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r72_paths import DELIV


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


class TestCoverageReceipt(unittest.TestCase):
    def setUp(self):
        self.r = load(DELIV / "ROUND72_COVERAGE_RECEIPT.json")
        self.splits = self.r["splits"]

    def test_row_counts(self):
        self.assertEqual(self.splits["dev_eval_learned"]["rows_loaded"], 192)
        self.assertEqual(self.splits["dev_eval_immediate"]["rows_loaded"], 33)
        self.assertEqual(self.splits["training"]["rows_loaded"], 384)
        for name, s in self.splits.items():
            self.assertEqual(s["rows_failed"], 0, name)
            self.assertEqual(s["rows_loaded"], s["rows_total"], name)

    def test_executor_refusal_present_and_zero_everywhere(self):
        for name, s in self.splits.items():
            self.assertEqual(s["refusal_sequence_present"], s["rows_loaded"], name)
            self.assertEqual(s["refusal_sequence_type_list"], s["rows_loaded"], name)
            self.assertEqual(s["cost_stream_present"], s["rows_loaded"], name)
            self.assertEqual(s["cost_stream_type_dict"], s["rows_loaded"], name)
            self.assertEqual(s["steps_missing_budget_refused_sends"], 0, name)
            self.assertEqual(s["steps_missing_capacity_refused_sends"], 0, name)
            self.assertEqual(s["refusal_sequence_len_total"], 0, name)
            self.assertEqual(s["measured_budget_refused_sends_total"], 0, name)
            self.assertEqual(s["measured_capacity_refused_sends_total"], 0, name)

    def test_collector_premask_split(self):
        self.assertEqual(self.splits["dev_eval_learned"]["collector_dict"], 192)
        self.assertEqual(self.splits["training"]["collector_dict"], 384)
        self.assertEqual(self.splits["dev_eval_immediate"]["collector_none"], 33)
        self.assertEqual(self.splits["dev_eval_immediate"]["collector_dict"], 0)
        for name in ("dev_eval_learned", "training"):
            s = self.splits[name]
            self.assertEqual(s["measured_masked_budget_or_capacity_total"], 0, name)
            self.assertEqual(s["measured_budget_masked_first_reports_total"], 0, name)
            self.assertEqual(s["measured_capacity_masked_first_reports_total"], 0, name)
            self.assertEqual(s["dict_rows_missing_masked_budget_or_capacity"], 0, name)
            self.assertEqual(s["dict_rows_missing_budget_masked_first_reports"], 0, name)
            self.assertEqual(s["dict_rows_missing_capacity_masked_first_reports"], 0, name)

    def test_channel_verdicts(self):
        cv = self.r["channel_verdicts"]
        self.assertTrue(cv["executor_refusal_channel"]["covered_all_splits"])
        self.assertTrue(cv["executor_refusal_channel"]["measured_zero_all_splits"])
        self.assertEqual(cv["collector_premask_channel"]["na_splits"], ["dev_eval_immediate"])
        self.assertEqual(sorted(cv["collector_premask_channel"]["covered_splits"]),
                         ["dev_eval_learned", "training"])
        self.assertTrue(cv["collector_premask_channel"]["measured_zero_where_covered"])

    def test_immediate_composition(self):
        comp = self.r["composition_notes"]["dev_eval_immediate"]
        self.assertEqual(comp["dev_block_canonical_rows"], 32)
        self.assertEqual(comp["focal_scene_1700027_rows"], 0)
        self.assertEqual(len(comp["extra_repeat_rows"]), 1)
        self.assertEqual(comp["extra_repeat_rows"][0]["scene"], 1800000)
        self.assertTrue(comp["extra_repeat_rows"][0]["spent_equals_canonical"])

    def test_zero_side_effects(self):
        for k, v in self.r["zero_side_effects"].items():
            self.assertEqual(v, 0, k)


if __name__ == "__main__":
    unittest.main(verbosity=2)