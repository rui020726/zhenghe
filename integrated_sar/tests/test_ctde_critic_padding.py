"""The critic must value a state the same way however its batch happens to be padded.

``CTDE_Network.get_value`` accepts agent and task padding masks and uses them both to
exclude padded slots from attention and to mean-pool over valid slots only. The driver
never built them, so the zero rows added by ``pad_to_batch_max`` were attended over and
averaged in. Because rollout pads to the round maximum while the update pads to a random
minibatch maximum, one identical stored state received two different values, and the
critic was trained towards ``returns = advantages + rollout_values`` -- a target it
cannot represent.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TASK_DIR = _REPO_ROOT / "task_allocation"

HAS_TORCH = importlib.util.find_spec("torch") is not None

_DRIVER: Any = None


def _load_old_driver() -> Any:
    """Import ``old_driver.py`` by path; it imports its siblings without a package."""
    global _DRIVER
    if _DRIVER is not None:
        return _DRIVER
    if str(_TASK_DIR) not in sys.path:
        sys.path.insert(0, str(_TASK_DIR))
    spec = importlib.util.spec_from_file_location(
        "ctde_old_driver_under_test", _TASK_DIR / "old_driver.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _DRIVER = module
    return module


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class PaddingMaskConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        self.torch = torch
        self.driver = _load_old_driver()

    def test_mask_marks_every_slot_past_the_real_length(self) -> None:
        torch = self.torch
        lengths = torch.tensor([3, 1, 4])
        mask = self.driver.padding_mask_from_lengths(lengths, 5, torch.device("cpu"))

        self.assertEqual(mask.shape, (3, 5))
        self.assertEqual(mask.dtype, torch.bool)
        expected = torch.tensor(
            [
                [False, False, False, True, True],
                [False, True, True, True, True],
                [False, False, False, False, True],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_no_lengths_means_no_mask(self) -> None:
        self.assertIsNone(self.driver.padding_mask_from_lengths(None, 4, self.torch.device("cpu")))


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class CriticPaddingInvarianceTests(unittest.TestCase):
    """One state, two batch widths: the value must not move."""

    def setUp(self) -> None:
        import torch

        _load_old_driver()
        from old_network import CTDE_Network

        self.torch = torch
        self.driver = _load_old_driver()
        torch.manual_seed(0)
        self.net = CTDE_Network(
            self_dim=10,
            neighbor_dim=10,
            task_dim=20,
            global_agent_dim=13,
            global_task_dim=23,
            hidden=128,
            dropout=0.15,
            use_pos_encoding=True,
        )
        self.net.eval()

    def _value(self, agent_rows, task_rows, masked: bool):
        torch = self.torch
        agents, agent_lengths = self.driver.pad_to_batch_max(agent_rows, 0.0)
        tasks, task_lengths = self.driver.pad_to_batch_max(task_rows, 0.0)
        device = torch.device("cpu")
        agent_mask = task_mask = None
        if masked:
            agent_mask = self.driver.padding_mask_from_lengths(
                agent_lengths, agents.shape[1], device
            )
            task_mask = self.driver.padding_mask_from_lengths(
                task_lengths, tasks.shape[1], device
            )
        with torch.no_grad():
            return self.net.get_value(agents, tasks, agent_mask, task_mask)

    def test_the_same_state_values_the_same_in_a_wider_batch(self) -> None:
        torch = self.torch
        agents = torch.randn(2, 13)
        tasks = torch.randn(3, 23)
        # A longer second sample forces a wider pad when they share a batch.
        other_agents = torch.randn(5, 13)
        other_tasks = torch.randn(6, 23)

        alone = self._value([agents], [tasks], masked=True)[0]
        wide = self._value([agents, other_agents], [tasks, other_tasks], masked=True)[0]
        torch.testing.assert_close(alone, wide, atol=1e-4, rtol=1e-3)

    def test_without_the_mask_the_value_does_move(self) -> None:
        """The negative control: this is the behaviour the masks remove."""
        torch = self.torch
        agents = torch.randn(2, 13)
        tasks = torch.randn(3, 23)
        other_agents = torch.randn(5, 13)
        other_tasks = torch.randn(6, 23)

        alone = self._value([agents], [tasks], masked=False)[0]
        wide = self._value([agents, other_agents], [tasks, other_tasks], masked=False)[0]
        self.assertFalse(
            torch.allclose(alone, wide, atol=1e-4),
            "padding was expected to change the unmasked value; if it no longer does, "
            "this test no longer demonstrates the defect it was written for",
        )


if __name__ == "__main__":
    unittest.main()
