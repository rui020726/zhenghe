"""Pure-computation checks for ROUND62's independent reconstruction helpers."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import round62_analysis as audit


class Round62AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freeze = audit.read_json(audit.R61 / "ROUND61_CASE_FREEZE.json")
        cls.cells = audit.read_json(audit.R61_RAW / "ROUND61_CELL_RESULTS.json")
        cls.rows = [audit.summarize_case(case, cls.cells) for case in cls.freeze["cases"]]

    def test_focus_recomputed_from_raw_rows(self):
        focus = self.rows[0]
        self.assertEqual((focus["Y_SEND"], focus["Y_SKIP"]), (1, 0))
        self.assertEqual(focus["system_delta"], 1)
        self.assertEqual(focus["gained_ids_under_send"], ["piloc-e1-target-5"])
        self.assertEqual(focus["lost_ids_under_send"], [])
        self.assertEqual(focus["cost_delta_bytes"], 2688)

    def test_controls_have_no_target_or_system_exchange(self):
        for row in self.rows[1:]:
            with self.subTest(scene=row["scene"]):
                self.assertEqual(row["target_delta"], 0)
                self.assertEqual(row["system_delta"], 0)
                self.assertEqual(row["gained_ids_under_send"], [])
                self.assertEqual(row["lost_ids_under_send"], [])
                self.assertGreater(row["cost_delta_bytes"], 0)

    def test_all_eight_opportunities_replay_the_same_prefix(self):
        self.assertEqual(len(self.rows), 8)
        self.assertTrue(all(row["prefix_and_opportunity"]["pre_decision_tape_rows_identical"]
                            for row in self.rows))
        self.assertTrue(all(row["prefix_and_opportunity"]["opportunity_reproduced"]
                            for row in self.rows))

    def test_freeze_vectors_are_distinct_at_float32_precision(self):
        signatures = {audit.float32_signature(row["pre_action_features_19d"]) for row in self.rows}
        self.assertEqual(len(signatures), 8)

    def test_deadline_slack_is_lossily_clipped(self):
        # Distinct sender-visible deadlines beyond 50 steps collapse to the same deadline feature.
        self.assertEqual(audit.clipped_slack(98.85), audit.clipped_slack(400.0))
        self.assertEqual(audit.clipped_slack(98.85), 1.0)
        self.assertNotEqual(98.85, 400.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
