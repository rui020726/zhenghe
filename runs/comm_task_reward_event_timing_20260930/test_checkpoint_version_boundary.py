"""Pure mocked-payload checks for the ROUND34 checkpoint restore boundary."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "comm_cost_v2_rerun_20260928"
STAGE_B = ROOT / "runs" / "comm_cost_stage_b_20260927"
sys.path[:0] = [str(RUN), str(STAGE_B)]

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_stub.optim = SimpleNamespace(Optimizer=object, Adam=object)
    torch_stub.__path__ = []
    sys.modules["torch"] = torch_stub

import v2_trainer


def legacy_payload(freeze: dict[str, str]) -> dict[str, object]:
    state = {
        "training_version": v2_trainer.TRAINER_TAG,
        "loss_version": v2_trainer.LOSS_VERSION,
        "state_schema": v2_trainer.STATE_SCHEMA,
        "gae_schema": v2_trainer.GAE_SCHEMA,
        "optimizer_schema": v2_trainer.OPTIMIZER_SCHEMA,
        "reward_timing_schema": v2_trainer.REWARD_TIMING_SCHEMA,
        "training_seed": 11,
        "episodes_done": 4,
        "updates": 0,
        "optimizer_steps": 0,
        "lambda": 1.0,
    }
    return {
        "schema": v2_trainer.STATE_SCHEMA,
        "trainer_tag": v2_trainer.TRAINER_TAG,
        "loss_version": v2_trainer.LOSS_VERSION,
        "gae_schema": v2_trainer.GAE_SCHEMA,
        "optimizer_schema": v2_trainer.OPTIMIZER_SCHEMA,
        "reward_timing_schema": v2_trainer.REWARD_TIMING_SCHEMA,
        "freeze": freeze,
        "controller": {
            "trunk": {"weight": "trunk-state"},
            "action_head": {"weight": "action-state"},
            "value_task_head": {"weight": "task-state"},
            "value_cost_head": {"weight": "cost-state"},
        },
        "optimizer": {"must_not": "restore"},
        "rng": {"must_not": "restore"},
        "training_state": state,
    }


class CheckpointVersionBoundaryTests(unittest.TestCase):
    def test_legacy_checkpoint_is_refused_by_real_resume_loader_before_restore(self) -> None:
        freeze = {"id": "legacy-source"}
        payload = legacy_payload(freeze)
        with patch.object(v2_trainer.torch, "load", return_value=payload, create=True):
            with self.assertRaisesRegex(RuntimeError, "read-only under the current"):
                v2_trainer.load_checkpoint(Path("legacy.pt"), freeze=freeze)

    def test_legacy_controller_has_explicit_read_only_load_without_optimizer_or_rng(self) -> None:
        freeze = {"id": "legacy-source"}
        payload = legacy_payload(freeze)

        class FakeModule:
            def __init__(self) -> None:
                self.state = None

            def load_state_dict(self, state) -> None:
                self.state = state

        class FakeController:
            def __init__(self) -> None:
                self.inner = SimpleNamespace(trunk=FakeModule(), action_head=FakeModule(),
                                             value_head=FakeModule())
                self.cost_head = FakeModule()

        fake_controller = FakeController()
        controller_module = types.ModuleType("v2_controller")
        controller_module.fresh_controller = lambda seed: fake_controller

        with patch.object(v2_trainer.torch, "load", return_value=payload, create=True), \
                patch.dict(sys.modules, {"v2_controller": controller_module}), \
                patch.object(v2_trainer, "restore_rng", side_effect=AssertionError), \
                patch.object(v2_trainer.torch.optim, "Adam", side_effect=AssertionError,
                             create=True):
            controller, state, identity = v2_trainer.load_checkpoint_read_only(
                Path("legacy.pt"), freeze=freeze
            )

        self.assertIs(controller, fake_controller)
        self.assertEqual(state["episodes_done"], 4)
        self.assertEqual(identity["load_mode"], "read_only")
        self.assertIsNone(identity["reward_timing_version"])
        self.assertEqual(identity["timing_interpretation"], "legacy_unversioned_capture_semantics")
        self.assertFalse(identity["optimizer_restored"])
        self.assertFalse(identity["rng_restored"])


if __name__ == "__main__":
    unittest.main()
