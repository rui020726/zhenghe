"""Correctness tests for the CTDE trainer in ``task_allocation/old_driver.py``.

Three defects, each of which silently corrupts learning rather than crashing.

``CTDE_Network.get_value`` accepts agent and task padding masks and uses them both to
exclude padded slots from attention and to mean-pool over valid slots only. The driver
never built them, so the zero rows added by ``pad_to_batch_max`` were attended over and
averaged in. Because rollout pads to the round maximum while the update pads to a random
minibatch maximum, one identical stored state received two different values, and the
critic was trained towards ``returns = advantages + rollout_values`` -- a target it
cannot represent.

Nothing called ``eval()``, so with dropout active the stored ``old_logprobs`` came from
an unrecorded mask and the behaviour policy behind the importance ratio was not
reproducible.

GAE walked the flat buffer, which is stored round-major and agent-minor, so it treated
the next robot in the same round as the next time step.
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


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TrainingModeTests(unittest.TestCase):
    """Dropout must not be part of the recorded behaviour policy."""

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

    def _log_prob(self, action: Any) -> Any:
        torch = self.torch
        # Real values, not zeros: dropout scales zeros to zeros, so an all-zero batch
        # would show no difference between the two modes and prove nothing.
        generator = torch.Generator().manual_seed(3)
        state = torch.randn(1, 10, generator=generator)
        neighbors = torch.randn(1, 20, 10, generator=generator)
        tasks = torch.randn(1, 3, 20, generator=generator)
        mask = torch.zeros(1, 3, dtype=torch.bool)
        with torch.no_grad():
            return self.net.get_action(state, neighbors, tasks, mask).log_prob(action)

    def test_train_mode_redraws_dropout_but_eval_mode_is_stable(self) -> None:
        torch = self.torch
        action = torch.zeros(1, dtype=torch.long)

        self.net.train()
        first = self._log_prob(action)
        second = self._log_prob(action)
        self.assertFalse(
            torch.allclose(first, second),
            "dropout was expected to make two train-mode passes over identical input differ",
        )

        self.net.eval()
        third = self._log_prob(action)
        fourth = self._log_prob(action)
        torch.testing.assert_close(third, fourth)

    def test_update_switches_the_network_back_to_train_mode(self) -> None:
        self.net.eval()
        # Undersized on purpose: the mode is established before the early return, so
        # the contract is observable without building a 1024-sample buffer.
        result = self.driver.update(
            self.net, None, {"states": []}, self.torch.device("cpu"),
            self.driver.TrainParams, 0.0,
        )
        self.assertEqual(result, {})
        self.assertTrue(self.net.training, "the PPO epochs would run with dropout frozen off")


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class GaeAlignmentTests(unittest.TestCase):
    """GAE runs along each robot's decisions, not along the flat buffer."""

    def setUp(self) -> None:
        self.driver = _load_old_driver()

    @staticmethod
    def _buffer(rewards, values, agent_ids, dones) -> dict:
        count = len(rewards)
        return {
            "states": [None] * count,
            "rewards": list(rewards),
            "values": list(values),
            "agent_ids": list(agent_ids),
            "dones": list(dones),
        }

    def test_advantages_follow_each_robots_own_sequence(self) -> None:
        gamma, lam = 0.9, 1.0
        # Round-major, agent-minor: both robots in a round share that round's value,
        # which is exactly what made V(s) - V(s) cancel in the flat walk.
        v0, v1 = 2.0, 3.0
        buffer = self._buffer(
            rewards=[1.0, 5.0, 2.0, 7.0],
            values=[v0, v0, v1, v1],
            agent_ids=["A", "B", "A", "B"],
            dones=[False, False, False, True],
        )

        advantages, returns = self.driver.compute_gae_segmented(buffer, gamma, lam)

        # Robot A decides at index 0 then index 2; index 2 is its last decision of the
        # episode, so it bootstraps from zero even though dones flags index 3 instead.
        adv_a2 = 2.0 - v1
        adv_a0 = (1.0 + gamma * v1 - v0) + gamma * lam * adv_a2
        # Robot B decides at index 1 then index 3.
        adv_b3 = 7.0 - v1
        adv_b1 = (5.0 + gamma * v1 - v0) + gamma * lam * adv_b3

        self.assertAlmostEqual(advantages[0], adv_a0, places=9)
        self.assertAlmostEqual(advantages[1], adv_b1, places=9)
        self.assertAlmostEqual(advantages[2], adv_a2, places=9)
        self.assertAlmostEqual(advantages[3], adv_b3, places=9)
        self.assertAlmostEqual(returns[2], 2.0, places=9)
        self.assertAlmostEqual(returns[3], 7.0, places=9)

    def test_one_robots_reward_never_enters_another_robots_advantage(self) -> None:
        gamma, lam = 0.9, 0.95
        shape = dict(
            values=[2.0, 2.0, 3.0, 3.0],
            agent_ids=["A", "B", "A", "B"],
            dones=[False, False, False, True],
        )
        quiet = self.driver.compute_gae_segmented(
            self._buffer(rewards=[1.0, 0.0, 2.0, 0.0], **shape), gamma, lam
        )[0]
        loud = self.driver.compute_gae_segmented(
            self._buffer(rewards=[1.0, 500.0, 2.0, -400.0], **shape), gamma, lam
        )[0]

        self.assertAlmostEqual(quiet[0], loud[0], places=9)
        self.assertAlmostEqual(quiet[2], loud[2], places=9)
        self.assertNotAlmostEqual(quiet[1], loud[1], places=6)

    def test_two_episodes_do_not_leak_into_each_other(self) -> None:
        gamma, lam = 0.9, 0.95
        alone = self.driver.compute_gae_segmented(
            self._buffer(
                rewards=[1.0, 2.0],
                values=[0.5, 0.5],
                agent_ids=["A", "B"],
                dones=[False, True],
            ),
            gamma,
            lam,
        )[0]
        both = self.driver.compute_gae_segmented(
            self._buffer(
                rewards=[1.0, 2.0, 9.0, 9.0],
                values=[0.5, 0.5, 4.0, 4.0],
                agent_ids=["A", "B", "A", "B"],
                dones=[False, True, False, True],
            ),
            gamma,
            lam,
        )[0]

        self.assertAlmostEqual(alone[0], both[0], places=9)
        self.assertAlmostEqual(alone[1], both[1], places=9)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TrainerConfigurationTests(unittest.TestCase):
    """A run has to be configurable from the command line and identifiable afterwards."""

    def setUp(self) -> None:
        import torch

        self.torch = torch
        self.driver = _load_old_driver()
        self.params = self.driver.TrainParams
        self._saved = {
            name: getattr(self.params, name)
            for name in ("SEED", "DEVICE", "LOG_DIR", "CHECKPOINT_DIR", "ACTOR_KV_MODE",
                         "RESUME", "RESUME_CKPT")
        }

    def tearDown(self) -> None:
        # TrainParams is module state shared by every test in this process.
        for name, value in self._saved.items():
            setattr(self.params, name, value)

    def test_cli_overrides_land_on_train_params(self) -> None:
        args = self.driver.build_parser().parse_args(
            ["--seed", "7", "--device", "cpu", "--checkpoint-dir", "runs/exp",
             "--actor-kv-mode", "summary", "--resume"]
        )
        self.driver.apply_cli_overrides(args)

        self.assertEqual(self.params.SEED, 7)
        self.assertEqual(self.params.DEVICE, "cpu")
        self.assertEqual(self.params.CHECKPOINT_DIR, "runs/exp")
        self.assertEqual(self.params.ACTOR_KV_MODE, "summary")
        self.assertTrue(self.params.RESUME)

    def test_a_new_run_does_not_resume_by_default(self) -> None:
        args = self.driver.build_parser().parse_args([])
        self.assertIsNone(args.resume, "an unset flag must not override TrainParams")
        self.driver.apply_cli_overrides(args)
        self.assertFalse(
            self.params.RESUME,
            "resuming by default silently jumps global_episodes_done, which drives the "
            "entropy weight, the LR schedule and the curriculum",
        )
        self.driver.apply_cli_overrides(self.driver.build_parser().parse_args(["--resume"]))
        self.assertTrue(self.params.RESUME)

    def test_the_parameter_snapshot_is_serializable_and_names_the_mode(self) -> None:
        import json

        snapshot = self.driver.params_snapshot()
        restored = json.loads(json.dumps(snapshot))
        self.assertEqual(restored["ACTOR_KV_MODE"], self.params.ACTOR_KV_MODE)
        self.assertIn("GAMMA", restored)
        self.assertIn("SEED", restored)

    def test_device_resolution_refuses_what_the_machine_cannot_do(self) -> None:
        self.assertEqual(self.driver.resolve_device("cpu").type, "cpu")
        self.assertEqual(self.driver.resolve_device(None).type, "cpu" if not self.torch.cuda.is_available() else "cuda")
        with self.assertRaises(ValueError):
            self.driver.resolve_device("tpu")
        if not self.torch.cuda.is_available():
            with self.assertRaises(ValueError):
                self.driver.resolve_device("cuda")

    def test_a_saved_checkpoint_records_the_mode_that_produced_it(self) -> None:
        import tempfile

        from old_network import CTDE_Network

        net = CTDE_Network(actor_kv_mode="tokens", **self.driver.NETWORK_DIMS)
        optimizer = self.torch.optim.Adam(net.parameters(), lr=1e-4)
        run_meta = {"actor_kv_mode": "tokens", "seed": 3, "git_commit": "abc123"}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ctde_ep_1.pth"
            self.driver.save_checkpoint(path, net, optimizer, 1, run_meta=run_meta)
            loaded = self.torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(loaded["actor_kv_mode"], "tokens")
        self.assertEqual(loaded["run_meta"]["seed"], 3)
        self.assertEqual(loaded["episode"], 1)
        self.assertIn("model_state_dict", loaded)
        self.assertIn("optimizer_state_dict", loaded)


if __name__ == "__main__":
    unittest.main()
