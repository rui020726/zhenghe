"""Pure read-only controls for one failed eval and frozen booking matrix."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r58_v35_eval_booking as booking

RAW = HERE.parent / "round58_gen3/raw"
PLAN = json.loads((RAW / "ROUND53_FIXED_PLAN.json").read_text(encoding="utf-8"))


class FrozenBookingTests(unittest.TestCase):
    def test_failed_eval_is_one_hash_closed_retry(self):
        row = booking._verify_failed_first_eval(RAW, PLAN)
        self.assertEqual(row["logical_job"], booking.FAILED_KEY)
        self.assertEqual(row["retry_status"], "pending")
        self.assertEqual((row["next_attempt"], row["expected_reserve_slot"]), (2, 4))

    def test_tampered_original_failed_claim_rejected(self):
        target = RAW / "launch_claims" / booking.FAILED_KEY / "attempt_0001.json"
        old = Path.read_bytes
        def altered(path):
            return b"tampered" if Path(path).resolve() == target.resolve() else old(path)
        with mock.patch.object(Path, "read_bytes", altered):
            with self.assertRaisesRegex(RuntimeError, "original bytes changed"):
                booking._verify_failed_first_eval(RAW, PLAN)

    def test_unknown_or_duplicate_matrix_key_rejected(self):
        altered = dict(PLAN)
        altered["jobs"] = [job for job in PLAN["jobs"]
                           if job["logical_job"] != booking.FAILED_KEY]
        with self.assertRaisesRegex(RuntimeError, "registered booking defect"):
            booking._verify_failed_first_eval(RAW, altered)
        eval_jobs = [job for job in PLAN["jobs"] if job["scope"].startswith(
            ("model_eval_", "rule_eval_"))]
        self.assertEqual(len(eval_jobs), 640)
        self.assertEqual(len({job["logical_job"] for job in eval_jobs}), 640)


if __name__ == "__main__":
    unittest.main(verbosity=2)
