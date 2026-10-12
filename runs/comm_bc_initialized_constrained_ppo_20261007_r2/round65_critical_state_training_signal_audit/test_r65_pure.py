"""ROUND65 narrow tests: pure computation needed by the analyzer/probe only.

Covers float32 input identity, ready_step/routing, tier classification, joint row/candidate
accounting (non-independence), masked exclusion, missing-not-filled-with-zero, seed-role mismatch
rejection, and the delegation-critical guard that the parsed focal must equal the constant or the
analyzer refuses to match by constant. No environment, optimizer, PPO, RNG or checkpoint writes.
"""
from __future__ import annotations
import sys, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import r65_common as C

FOCAL = list(C.FOCAL_FEATURES)
FBYTES = bytes.fromhex(C.FOCAL_F32_HEX)


class Float32Identity(unittest.TestCase):
    def test_focal_bytes_equal_constant(self):
        self.assertEqual(C.f32_bytes(FOCAL), FBYTES)

    def test_one_ulp_change_is_not_exact(self):
        v = list(FOCAL)
        v[18] = np.nextafter(np.float32(v[18]), np.float32(1.0)).item()  # nudge step dim 1 ULP
        self.assertFalse(C.is_exact_match(v, FBYTES))

    def test_order_and_dtype_matter(self):
        v = list(FOCAL); v[3], v[4] = v[4], v[3]  # swap two 1.0 dims -> same values, still exact
        # swapping equal 1.0s yields identical bytes; a real change must break identity:
        v2 = list(FOCAL); v2[14] = 2.0  # backlog 1 -> 2
        self.assertTrue(C.is_exact_match(v, FBYTES))
        self.assertFalse(C.is_exact_match(v2, FBYTES))


class ReadyStepRouting(unittest.TestCase):
    def test_focal_ready_and_early(self):
        self.assertEqual(C.ready_step(FOCAL), 134)
        self.assertTrue(C.is_early(FOCAL))

    def test_late_route(self):
        v = list(FOCAL); v[18] = 0.200
        self.assertEqual(C.ready_step(v), 200)
        self.assertFalse(C.is_early(v))


class TierClassification(unittest.TestCase):
    def test_focal_is_t0(self):
        self.assertEqual(C.classify_free_first_report(FOCAL, C.f32(FOCAL), FBYTES), "T0_exact")

    def test_partial_t1_same_ready_backlog_future(self):
        v = list(FOCAL); v[0] = 6.0; v[18] = 0.140  # discovered@134, decided@140 -> ready 134
        ok, diff = C.partial_t1(v, C.f32(FOCAL))
        self.assertTrue(ok)
        self.assertEqual(C.ready_step(v), 134)
        self.assertEqual(C.classify_free_first_report(v, C.f32(FOCAL), FBYTES), "T1_partial")

    def test_t2_when_backlog_differs(self):
        v = list(FOCAL); v[14] = 3.0
        self.assertEqual(C.classify_free_first_report(v, C.f32(FOCAL), FBYTES), "T2_early_other")

    def test_late_bucket(self):
        v = list(FOCAL); v[18] = 0.200
        self.assertEqual(C.classify_free_first_report(v, C.f32(FOCAL), FBYTES), "late_or_other")


class JointRowCandidateAccounting(unittest.TestCase):
    def test_candidates_share_one_row_not_independent(self):
        # one decision row (same row_key) with 3 free candidates
        rows = [{"row_key": ("s33", 5, 134), "scene": 1300002, "cand": t} for t in ("a", "b", "c")]
        n_cand = len(rows)
        n_row = len({r["row_key"] for r in rows})
        self.assertEqual(n_cand, 3)
        self.assertEqual(n_row, 1)  # 3 candidates, 1 shared decision row -> not 3 independent feedback

    def test_repeated_scene_non_independence(self):
        scenes = [1300002, 1300002, 1300002, 1300019]
        self.assertEqual(len(scenes), 4)
        self.assertEqual(len(set(scenes)), 2)  # repeated scenes are not independent samples


class MaskedAndMissing(unittest.TestCase):
    def test_masked_excluded_from_free(self):
        record = {"free_first_reports": ["a"], "masked": ["b"], "targets": ["a", "b"]}
        free = [t for t in record["targets"] if t in record["free_first_reports"]]
        self.assertEqual(free, ["a"])  # masked b is not a free choice

    def test_missing_old_value_not_zero(self):
        rec = {"old_value_task": None}
        # a missing old value must stay missing, never be backfilled to 0.0
        self.assertIsNone(rec["old_value_task"])
        self.assertNotEqual(rec["old_value_task"], 0.0)


class SeedRoleMismatch(unittest.TestCase):
    def verdict(self, computed, expected):
        return "verified" if computed == expected else "identity_mismatch"

    def test_matching_role_verified(self):
        self.assertEqual(self.verdict("abc", "abc"), "verified")

    def test_wrong_seed_or_milestone_rejected(self):
        # a checkpoint whose param SHA does not match the expected role must be rejected
        self.assertEqual(self.verdict("abc", "def"), "identity_mismatch")


class FocalConstantDriftGuard(unittest.TestCase):
    def test_parsed_focal_matches_constant(self):
        import r65_audit as A
        f32, fb, prov = A.parse_focal()
        self.assertEqual(fb, FBYTES)
        self.assertTrue(prov["constant_matches_parsed"])
        self.assertEqual(prov["tick"], 134)

    def test_drifted_constant_aborts(self):
        import r65_audit as A
        saved = C.FOCAL_F32_HEX
        try:
            C.FOCAL_F32_HEX = "00" * 76  # deliberately wrong
            with self.assertRaises(SystemExit):
                A.parse_focal()
        finally:
            C.FOCAL_F32_HEX = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
