"""Pure arithmetic checks for the frozen 32-scene paired bootstrap."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from r58_final_source_aware_stats import paired_bootstrap


class FrozenArithmetic(unittest.TestCase):
    def test_shared_indices_direction_and_linear_quantile(self):
        idx = np.random.Generator(np.random.PCG64(20261008)).integers(
            0, 32, size=(10_000, 32))
        diff = np.zeros(32, dtype=np.int64)
        diff[-1] = 3200
        actual = paired_bootstrap(diff, idx)
        manual = diff[idx].sum(axis=1) / 32
        self.assertEqual(actual["mean"], 100.0)
        self.assertEqual(actual["ci95_linear"], list(np.quantile(
            manual, [0.025, 0.975], method="linear")))
        self.assertFalse(actual["strictly_positive"])
        reverse = paired_bootstrap(-diff, idx)
        self.assertEqual(reverse["mean"], -100.0)
        self.assertEqual(reverse["ci95_linear"], [-actual["ci95_linear"][1],
                                                    -actual["ci95_linear"][0]])

    def test_shape_and_constant_difference(self):
        idx = np.zeros((10_000, 32), dtype=np.int64)
        result = paired_bootstrap(np.full(32, 7, dtype=np.int64), idx)
        self.assertEqual(result["ci95_linear"], [7.0, 7.0])
        self.assertTrue(result["strictly_positive"])
        with self.assertRaises(ValueError):
            paired_bootstrap(np.zeros(16, dtype=np.int64), idx)


if __name__ == "__main__":
    unittest.main()
