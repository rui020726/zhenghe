from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from r64_analysis import _feature_key, _pair
from r64_execute import TARGET


class PairedAnalysisTests(unittest.TestCase):
    def test_pair_reports_target_and_system_set_exchange_separately(self):
        vector = [0.0, 1.0, 0.25]
        send = {
            "metrics": {"on_time_ids": [TARGET, "send_only"], "spent_bytes": 120,
                        "emission_sequence": [[134, "focus"]]},
            "focus": {"target_chain": [{"status": "completed"}],
                      "target_transcript": [{"kind": "send"}],
                      "features_19d": vector},
        }
        skip = {
            "metrics": {"on_time_ids": ["skip_only"], "spent_bytes": 0,
                        "emission_sequence": []},
            "focus": {"target_chain": [{"status": "expired"}],
                      "target_transcript": [], "features_19d": vector},
        }
        result = _pair(send, skip)
        self.assertEqual(result["target_delta_Y_SEND_minus_Y_SKIP"], 1)
        self.assertEqual(result["gained_under_SEND"], [TARGET, "send_only"])
        self.assertEqual(result["lost_under_SEND"], ["skip_only"])
        self.assertEqual(result["whole_system_task_delta"], 1)
        self.assertEqual(result["cost_delta_bytes_SEND_minus_SKIP"], 120)

    def test_feature_key_uses_float32_identity_not_python_float_format(self):
        self.assertEqual(_feature_key([0.1]), _feature_key([float("0.10000000149011612")]))
        self.assertNotEqual(_feature_key([0.1]), _feature_key([0.1001]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
