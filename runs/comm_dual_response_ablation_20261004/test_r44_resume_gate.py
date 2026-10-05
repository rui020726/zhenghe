"""Pure regression tests for ROUND44's deterministic saved-RNG continuation helper."""

from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from r44_resume_bridge import MECHANICAL_VERSION, STATE_BRIDGE_VERSION
from r44_resume_gate import (
    _is_expected_old_training_checkpoint_rejection,
    _next_values_from_saved_rng,
    _validate_training_order,
)


class Round44ResumeGateTests(unittest.TestCase):
    def test_resume_bridge_is_versioned_and_mechanical(self) -> None:
        self.assertEqual(STATE_BRIDGE_VERSION, "round44-exact-resume-state-bridge/1")
        self.assertEqual(MECHANICAL_VERSION, "round44-resume-comparator-and-restore/4")

    def test_old_r43_checkpoint_rejection_accepts_exact_version_field_spelling(self) -> None:
        self.assertTrue(_is_expected_old_training_checkpoint_rejection(
            "v2_train_s11_ep128.pt: training_version mismatch; read-only"))
        self.assertTrue(_is_expected_old_training_checkpoint_rejection(
            "checkpoint: training version mismatch"))
        self.assertTrue(_is_expected_old_training_checkpoint_rejection(
            "checkpoint: freeze identity mismatch"))
        self.assertFalse(_is_expected_old_training_checkpoint_rejection(
            "checkpoint: unexpected tensor shape mismatch"))

    def test_resume_scene_comes_from_the_full_frozen_order(self) -> None:
        frozen_order = list(range(1300000, 1300128))
        frozen_order[4] = 1300003
        self.assertEqual(_validate_training_order(frozen_order.copy(), frozen_order), 1300003)
        changed_order = frozen_order.copy()
        changed_order[4] = 1300014
        with self.assertRaisesRegex(RuntimeError, "differs from the committed freeze"):
            _validate_training_order(changed_order, frozen_order)

    def test_next_values_match_independent_python_numpy_torch_streams(self) -> None:
        py = random.Random(13579)
        numpy_rng = np.random.RandomState(24680)
        torch_rng = torch.Generator(device="cpu")
        torch_rng.manual_seed(112233)
        py_state = py.getstate()
        np_state = numpy_rng.get_state()
        torch_state = torch_rng.get_state()
        expected = {
            "python_random": py.random(),
            "numpy_random": float(numpy_rng.random_sample()),
            "torch_rand_cpu": torch.rand(3, generator=torch_rng).tolist(),
        }
        payload = {
            "python": [py_state[0], list(py_state[1]), py_state[2]],
            "numpy": [np_state[0], np_state[1].tolist(), np_state[2],
                      np_state[3], np_state[4]],
            "torch": torch_state.tolist(),
        }
        self.assertEqual(_next_values_from_saved_rng(payload), expected)


if __name__ == "__main__":
    unittest.main()
