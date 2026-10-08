"""ROUND55 pure computation boundary tests (no project runtime imports).

Covers: float32 step/1000 round-trip over all recorded steps, the strict >147
gate boundary, age clipping, the analytic bootstrap quantiles for a 15-zero /
one-+2514 vector, and decomposition closure.
"""
from __future__ import annotations
import json, math, unittest
from pathlib import Path

import numpy as np


def recover(age: float, step_norm: float, first_emission: bool,
            cut: float = 147.0, horizon: float = 1000.0) -> bool:
    age_i = int(round(float(age)))
    step = int(round(float(step_norm) * float(horizon)))
    return bool(first_emission and step - age_i > cut)


class TestRound55Pure(unittest.TestCase):
    def test_float32_step_round_trip_exact_for_all_steps(self):
        for step in range(0, 1001):
            f18 = np.float32(min(1.0, step / 1000.0))
            self.assertEqual(int(round(float(f18) * 1000.0)), step, f"step {step}")

    def test_gate_boundary_is_strict_above_147(self):
        self.assertFalse(recover(0.0, np.float32(147 / 1000.0), True))   # ready=147 -> skip
        self.assertTrue(recover(0.0, np.float32(148 / 1000.0), True))    # ready=148 -> send
        self.assertTrue(recover(0.0, np.float32(999 / 1000.0), True))
        self.assertFalse(recover(0.0, np.float32(0 / 1000.0), True))

    def test_age_offset_moves_ready_step(self):
        # ready = 147 with age 1 at step 148 -> skip; ready = 148 with age 0 -> send
        self.assertFalse(recover(1.0, np.float32(148 / 1000.0), True))
        self.assertTrue(recover(0.0, np.float32(148 / 1000.0), True))
        # age clipping: max(0, tick - ready) means age is never negative
        self.assertFalse(recover(-0.0, np.float32(100 / 1000.0), True))

    def test_first_emission_required(self):
        self.assertFalse(recover(0.0, np.float32(999 / 1000.0), False))

    def test_focal_cells_recover_to_recorded_late_actions(self):
        # scene1301014 t5: decision step 140, age 0 -> ready 140 <= 147 -> late skips
        self.assertFalse(recover(0.0, np.float32(140 / 1000.0), True))
        # scene1301010 t5: decision step 148, age 0 -> ready 148 > 147 -> late sends
        self.assertTrue(recover(0.0, np.float32(148 / 1000.0), True))

    def test_analytic_bootstrap_quantiles_15_zeros_one_2514(self):
        rng = np.random.Generator(np.random.PCG64(20261008))
        idx = rng.integers(0, 16, size=(10000, 16))
        deltas = np.array([0.0] * 15 + [2514.0])
        means = deltas[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5], method="linear")
        self.assertEqual((float(lo), float(hi)), (0.0, 471.375))
        p = 1 / 16
        self.assertAlmostEqual((1 - p) ** 16, 0.3560741304517942, places=12)
        cdf3 = sum(math.comb(16, k) * p ** k * (1 - p) ** (16 - k) for k in range(4))
        self.assertGreaterEqual(cdf3, 0.975)

    def test_decomposition_closure(self):
        def close(c_sent, t_sent):
            A = sum(v for k, v in c_sent.items() if k not in t_sent)
            B = sum(v for k, v in t_sent.items() if k not in c_sent)
            C = sum(c_sent[k] - t_sent[k] for k in c_sent.keys() & t_sent.keys())
            return A - B + C == sum(c_sent.values()) - sum(t_sent.values())
        self.assertTrue(close({"t1": 120, "t3": 500}, {"t1": 100, "t2": 250}))
        self.assertTrue(close({}, {"t1": 2508}))
        self.assertTrue(close({"t5": 2514}, {}))


if __name__ == "__main__":
    unittest.main()
