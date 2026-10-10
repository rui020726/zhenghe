"""Read-only V29 committed-source and cumulative-journal counterexamples."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import r58_resume_seam as seam

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "runs/comm_bc_initialized_constrained_ppo_20261007_r2"


class V29ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proof = seam.validate_resume_sources(REPO)
        cls.source = cls.proof["committed_continuation"]

    def test_exact_b20_and_discarded_v26_writes_are_separate(self):
        self.assertEqual(self.source["cursor"], {
            "episodes_done": 84, "updates": 21, "optimizer_steps": 116})
        self.assertEqual(self.source["historical_ppo_calls"], 23)
        self.assertEqual(self.source["historical_optimizer_calls"], 132)
        self.assertEqual(self.source["historical_checkpoint_writes"], 79)
        self.assertEqual(self.proof["discarded_v26_identity_writes"]["rows"], 2)
        self.assertFalse(self.proof["discarded_v26_identity_writes"]["scientific_chain"])

    def test_exact_committed_batch_records_and_missing_duplicate_rejected(self):
        records = seam._batch_records(ROOT / "round58_gen3/raw",
            ROOT / "round58_gen3_resume_v29/raw/checkpoints",
            [Path(p) / "checkpoints" for p in self.source["prior_batch_roots"]])
        self.assertEqual(seam.carryover_count_from_batches(records, self.proof), 23)
        b20 = [p for p in records if p.name == "r40_batch_s11_b20_attempt01.json"]
        self.assertEqual(len(b20), 1)
        with self.assertRaises(RuntimeError):
            seam.carryover_count_from_batches([p for p in records if p != b20[0]], self.proof)
        with self.assertRaises(RuntimeError):
            seam.carryover_count_from_batches(records + [b20[0]], self.proof)

    def test_wrong_map_and_missing_v26_physical_events_rejected(self):
        with self.assertRaises(PermissionError):
            seam.validate_resume_sources(REPO, expected_map_sha256="0" * 64)
        real = seam.sha256
        def wrong_v26(path):
            if "round58_gen3_resume_v26" in str(path) and str(path).endswith("checkpoint_writes.jsonl"):
                return "0" * 64
            return real(path)
        with mock.patch.object(seam, "sha256", side_effect=wrong_v26):
            with self.assertRaises(PermissionError):
                seam.validate_resume_sources(REPO)


if __name__ == "__main__":
    unittest.main()
