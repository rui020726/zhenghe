"""Read-only two-seed committed continuation counterexamples."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import r58_resume_seam as seam

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "runs/comm_bc_initialized_constrained_ppo_20261007_r2"


class V32ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proof = seam.validate_resume_sources(REPO)

    def test_exact_two_seed_endpoints_and_physical_writes(self):
        self.assertEqual(self.proof["committed_continuation"]["cursor"], {
            "episodes_done": 128, "updates": 32, "optimizer_steps": 184})
        self.assertEqual(self.proof["additional_continuations"][22]["cursor"], {
            "episodes_done": 20, "updates": 5, "optimizer_steps": 32})
        self.assertEqual(self.proof["committed_continuation"]["historical_ppo_calls"], 38)
        self.assertEqual(self.proof["committed_continuation"]["historical_optimizer_calls"], 224)
        self.assertEqual(self.proof["committed_continuation"]["historical_checkpoint_writes"], 128)
        self.assertEqual(self.proof["discarded_v26_identity_writes"]["rows"], 2)

    def test_missing_or_duplicate_committed_batch_rejected(self):
        source = self.proof["committed_continuation"]
        rows = seam._batch_records(ROOT / "round58_gen3/raw",
            ROOT / "round58_gen3_resume_v32/raw/checkpoints",
            [Path(p) / "checkpoints" for p in source["prior_batch_roots"]])
        self.assertEqual(seam.carryover_count_from_batches(rows, self.proof), 38)
        for name in ("r40_batch_s11_b31_attempt01.json",
                     "r40_batch_s22_b4_attempt01.json"):
            target, = [p for p in rows if p.name == name]
            with self.assertRaises(RuntimeError):
                seam.carryover_count_from_batches([p for p in rows if p != target], self.proof)
            with self.assertRaises(RuntimeError):
                seam.carryover_count_from_batches(rows + [target], self.proof)

    def test_tampered_v29_journal_and_source_map_rejected(self):
        with self.assertRaises(PermissionError):
            seam.validate_resume_sources(REPO, expected_map_sha256="0" * 64)
        original = seam.sha256
        def wrong_v29(path):
            if "round58_gen3_resume_v29" in str(path) and str(path).endswith("optimizer_step_calls.jsonl"):
                return "0" * 64
            return original(path)
        with mock.patch.object(seam, "sha256", side_effect=wrong_v29):
            with self.assertRaises(PermissionError):
                seam.validate_resume_sources(REPO)


if __name__ == "__main__":
    unittest.main()
