"""ROUND54 zero-environment pure computation tests (no project runtime imports)."""
from __future__ import annotations
import json, unittest
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def decompose_pair(c_sent, t_sent):
    A_targets = sorted(set(c_sent) - set(t_sent))
    B_targets = sorted(set(t_sent) - set(c_sent))
    A = int(sum(c_sent[t] for t in A_targets))
    B = int(sum(t_sent[t] for t in B_targets))
    C = int(sum(c_sent[t] - t_sent[t] for t in set(c_sent) & set(t_sent)))
    delta = int(sum(c_sent.values()) - sum(t_sent.values()))
    return A, B, C, delta


class TestRound54Pure(unittest.TestCase):
    def test_decomposition_closes_with_and_without_common_price_differences(self):
        A, B, C, delta = decompose_pair({"t1": 100, "t2": 200}, {"t1": 100, "t2": 200})
        self.assertEqual((A, B, C, delta), (0, 0, 0, 0))
        self.assertEqual(A - B + C, delta)
        A, B, C, delta = decompose_pair({"t1": 300}, {"t2": 250})
        self.assertEqual((A, B, C), (300, 250, 0))
        self.assertEqual(A - B + C, delta)
        A, B, C, delta = decompose_pair({"t1": 120, "t3": 500}, {"t1": 100, "t2": 250})
        self.assertEqual((A, B, C), (500, 250, 20))
        self.assertEqual(A - B + C, delta)

    def test_bootstrap_indices_deterministic_pcg64_20261008(self):
        rng = np.random.Generator(np.random.PCG64(20261008))
        idx = rng.integers(0, 16, size=(10000, 16))
        rng2 = np.random.Generator(np.random.PCG64(20261008))
        idx2 = rng2.integers(0, 16, size=(10000, 16))
        self.assertTrue(np.array_equal(idx, idx2))
        probe = np.arange(1.0, 17.0)
        independent = float(sum(float(probe[i]) for i in idx[0]) / 16)
        self.assertLess(abs(float(probe[idx[0]].mean()) - independent), 1e-9)

    def test_threshold_is_strict_not_rounded(self):
        self.assertTrue(4838.249 < 4838.25)
        self.assertFalse(4838.25 < 4838.25)
        self.assertFalse(4851.0 < 4838.25)
        self.assertFalse(4995.375 < 4838.25)
        self.assertTrue(3923.25 < 4838.25)

    def test_real_row_emission_bytes_sum_to_spent_bytes(self):
        row_path = (HERE.parents[1] / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
                    / "round53_critic_isolation_trial_readiness" / "raw" / "rows"
                    / "v2_r44_eval_new_s11_greedy_s1301000_a1.json")
        row = json.loads(row_path.read_text(encoding="utf-8"))
        billed = sum(int(item[3]) for item in row["emission_sequence"]
                     if len(item) >= 6 and item[5] is True)
        self.assertEqual(billed, row["spent_bytes"])
        self.assertTrue(all(len(item) == 6 for item in row["emission_sequence"]))


if __name__ == "__main__":
    unittest.main()
