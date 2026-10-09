"""Zero-environment counterexamples for the unfinished batch-10 prefix proof."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import r58_uncommitted_prefix_probe as probe


class PrefixProbeTests(unittest.TestCase):
    def test_actual_prefix_and_rng_control(self) -> None:
        result = probe.verify()
        self.assertTrue(result["control_episode39_rng_equals_committed_b9"])
        self.assertEqual(result["prefix_decision_samples"], 207)
        self.assertEqual({k: v["records"] for k, v in result["episodes"].items()},
                         {"40": 54, "41": 55, "42": 98})
        self.assertEqual(sum(v["action_mismatches"] for v in result["episodes"].values()), 0)

    def test_claim_artifact_hash_tamper_is_rejected(self) -> None:
        real_read = probe.read_json

        def tampered(path):
            value = real_read(path)
            if path.name == "attempt_0001.json" and "new_training_s11_e40" in str(path):
                value = copy.deepcopy(value)
                value["round45_artifact_receipt"]["artifacts"]["tape"]["sha256"] = "0" * 64
            return value

        with patch.object(probe, "read_json", side_effect=tampered):
            with self.assertRaisesRegex(RuntimeError, "tape is outside its hash-bound source"):
                probe._episode(40)


if __name__ == "__main__":
    unittest.main()
