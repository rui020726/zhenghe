"""Audit optimizer membership, updates, zeroing, clipping, checkpoint save and restore."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import v2_common as common
import v2_trainer as trainer
from v2_controller import fresh_controller
from v2_correctness_test_support import episode, record


def expected_module_parameters(controller):
    return {
        "trunk": list(controller.inner.trunk.named_parameters()),
        "action_head": list(controller.inner.action_head.named_parameters()),
        "task_value_head": list(controller.inner.value_head.named_parameters()),
        "cost_head": list(controller.cost_head.named_parameters()),
    }


def parameter_audit(expected_by_module, candidate):
    expected = [parameter for pairs in expected_by_module.values()
                for _name, parameter in pairs]
    expected_ids = {id(parameter) for parameter in expected}
    candidate_ids = [id(parameter) for parameter in candidate]
    return {
        "missing": sorted(expected_ids - set(candidate_ids)),
        "external": sorted(set(candidate_ids) - expected_ids),
        "duplicates": len(candidate_ids) - len(set(candidate_ids)),
        "passed": (set(candidate_ids) == expected_ids
                   and len(candidate_ids) == len(expected_ids)
                   and len(candidate_ids) == len(set(candidate_ids))),
    }


class CostHeadLifecycleCounterexample(unittest.TestCase):
    def test_every_module_is_in_optimizer_and_cost_head_lifecycle_is_complete(self) -> None:
        controller = fresh_controller(11)
        modules = expected_module_parameters(controller)
        expected = [parameter for pairs in modules.values()
                    for _name, parameter in pairs]
        self.assertEqual(len(expected), 10)
        self.assertEqual(sum(parameter.numel() for parameter in expected), 5700)

        # Demonstrate that this independent object-ID audit rejects all three old gate traps.
        missing_cost = [parameter for parameter in controller.parameters
                        if id(parameter) not in {id(p) for p in controller.cost_head.parameters()}]
        duplicate_trunk = [*controller.parameters, controller.inner.trunk.parameters().__next__()]
        external_parameter = torch.nn.Parameter(torch.ones(1))
        self.assertFalse(parameter_audit(modules, missing_cost)["passed"])
        self.assertFalse(parameter_audit(modules, duplicate_trunk)["passed"])
        self.assertFalse(parameter_audit(modules, [*controller.parameters, external_parameter])["passed"])

        optimizer = torch.optim.Adam(controller.parameters, lr=common.LEARNING_RATE)
        synthetic_module_steps = 0
        module_update_evidence = {}
        problems = []
        for module_name, named_parameters in modules.items():
            module = {
                "trunk": controller.inner.trunk,
                "action_head": controller.inner.action_head,
                "task_value_head": controller.inner.value_head,
                "cost_head": controller.cost_head,
            }[module_name]
            optimizer.zero_grad(set_to_none=True)
            if module_name == "trunk":
                output = module(torch.ones((1, 19), dtype=torch.float32))
            else:
                output = module(torch.ones((1, 64), dtype=torch.float32))
            loss = (output - 3.0).square().sum()
            loss.backward()
            parameters = [parameter for _name, parameter in named_parameters]
            gradient_norm = sum(float(parameter.grad.detach().abs().sum())
                                for parameter in parameters if parameter.grad is not None)
            before = [parameter.detach().clone() for parameter in parameters]
            optimizer.step()
            synthetic_module_steps += 1
            updated = any(not torch.equal(old, new.detach())
                          for old, new in zip(before, parameters))
            state_present = all(parameter in optimizer.state for parameter in parameters)
            if gradient_norm <= 0.0:
                problems.append(f"{module_name} synthetic loss had no gradient")
            if not updated:
                problems.append(f"{module_name} parameters did not update")
            if not state_present:
                problems.append(f"{module_name} optimizer state missing")
            module_update_evidence[module_name] = {
                "gradient_l1": gradient_norm,
                "updated": updated,
                "optimizer_state": state_present,
            }

        features = torch.ones((1, 19), dtype=torch.float32).numpy()
        rollout = episode(
            [record(controller, 1, value=0.0, cost_value=0.0, action=1,
                    features=features)],
            steps=1,
            task_stream={1: 1.0},
            cost_stream={1: {"airtime_bytes": common.B0_BYTES}},
            lam=0.0,
        )
        clip_parameter_ids: set[int] = set()
        zero_grad_was_clear: list[bool] = []
        optimizer_ids_before = [id(parameter) for group in optimizer.param_groups
                                for parameter in group["params"]]
        original_clip = torch.nn.utils.clip_grad_norm_
        original_zero_grad = optimizer.zero_grad
        original_step = optimizer.step
        step_gradients = []

        def record_clip(parameters, max_norm, *args, **kwargs):
            parameters = list(parameters)
            clip_parameter_ids.update(id(parameter) for parameter in parameters)
            return original_clip(parameters, max_norm, *args, **kwargs)

        def record_zero_grad(*args, **kwargs):
            original_zero_grad(*args, **kwargs)
            zero_grad_was_clear.append(all(parameter.grad is None for parameter in expected))

        def record_step(*args, **kwargs):
            step_gradients.append({
                module_name: sum(float(parameter.grad.detach().abs().sum())
                                 for _name, parameter in pairs if parameter.grad is not None)
                for module_name, pairs in modules.items()
            })
            return original_step(*args, **kwargs)

        optimizer.zero_grad = record_zero_grad
        optimizer.step = record_step
        before_ppo = {name: [p.detach().clone() for _key, p in pairs]
                      for name, pairs in modules.items()}
        with patch("torch.nn.utils.clip_grad_norm_", side_effect=record_clip):
            trainer.ppo_update(controller, optimizer, [rollout], update_index=0)

        actual_audit = parameter_audit(
            modules, [p for group in optimizer.param_groups for p in group["params"]])
        if not actual_audit["passed"]:
            problems.append("optimizer object-ID set has missing, external, or duplicate parameters")
        if len(optimizer_ids_before) != len(set(optimizer_ids_before)):
            problems.append("optimizer contains duplicate parameter objects")
        if clip_parameter_ids != {id(parameter) for parameter in expected}:
            problems.append("gradient clipping did not receive the complete module set")
        if not zero_grad_was_clear or not all(zero_grad_was_clear):
            problems.append("optimizer.zero_grad did not clear every module")
        if len(step_gradients) != common.EPOCHS_PER_BATCH:
            problems.append("PPO update did not reach every synthetic optimizer step")
        if not all(row["cost_head"] > 0.0 for row in step_gradients):
            problems.append("cost_head did not receive a nonzero PPO gradient")
        for module_name, pairs in modules.items():
            params = [p for _key, p in pairs]
            if not any(not torch.equal(old, new.detach())
                       for old, new in zip(before_ppo[module_name], params)):
                problems.append(f"{module_name} did not update through PPO")
            if not all(parameter in optimizer.state for parameter in params):
                problems.append(f"{module_name} PPO optimizer state missing")
        ppo_optimizer_steps = len(step_gradients)
        self.assertEqual(synthetic_module_steps, 4)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.pt"
            state = trainer.fresh_training_state(11)
            trainer.save_checkpoint(controller, optimizer, state, path, freeze={"id": "test"})
            restored, restored_optimizer, restored_state = trainer.load_checkpoint(
                path, freeze={"id": "test"}
            )
            restored_modules = expected_module_parameters(restored)
            restored_parameters = [p for pairs in restored_modules.values()
                                   for _name, p in pairs]
            restored_optimizer_params = [p for group in restored_optimizer.param_groups
                                        for p in group["params"]]
            if not parameter_audit(restored_modules, restored_optimizer_params)["passed"]:
                problems.append("restored optimizer parameter set is incomplete")
            for pairs in restored_modules.values():
                if not all(parameter in restored_optimizer.state
                           for _name, parameter in pairs):
                    problems.append("restored optimizer state is incomplete")
            if len(restored_parameters) != len(expected):
                problems.append("restored controller module set changed size")
            if restored_state["gae_schema"] != trainer.GAE_SCHEMA:
                problems.append("restored checkpoint GAE schema mismatch")
            if restored_state["optimizer_schema"] != trainer.OPTIMIZER_SCHEMA:
                problems.append("restored checkpoint optimizer schema mismatch")
            if restored_state["reward_timing_schema"] != trainer.REWARD_TIMING_SCHEMA:
                problems.append("restored checkpoint reward timing schema mismatch")
            if restored_state["training_version"] != trainer.TRAINING_VERSION:
                problems.append("restored checkpoint training version mismatch")
            if restored_state["reward_timing_version"] != trainer.REWARD_TIMING_VERSION:
                problems.append("restored checkpoint reward timing version mismatch")

        self.assertEqual(synthetic_module_steps + ppo_optimizer_steps,
                         4 + common.EPOCHS_PER_BATCH)
        self.assertEqual(set(module_update_evidence),
                         {"trunk", "action_head", "task_value_head", "cost_head"})
        self.assertEqual(problems, [], "; ".join(problems))

    def test_old_stage_c_checkpoint_is_not_a_valid_repaired_checkpoint(self) -> None:
        controller = fresh_controller(11)
        optimizer = torch.optim.Adam(controller.parameters, lr=common.LEARNING_RATE)
        payload = {
            "schema": "constrained-cost-state/1",
            "trainer_tag": "constrained-cost-v1",
            "loss_version": "constrained-lagrangian/1",
            "freeze": {"id": "test"},
            "controller": controller.state_dict(),
            "optimizer": optimizer.state_dict(),
            "training_state": trainer.fresh_training_state(11),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy-stage-c.pt"
            torch.save(payload, path)
            with self.assertRaises(RuntimeError):
                trainer.load_checkpoint(path, freeze={"id": "test"})

    def test_outputs_are_isolated_to_this_round_directory(self) -> None:
        """ROUND24 §2.5: every write path points at this round's directory (output isolation).

        This round's harness lives in its own run directory, so the check is that no artefact
        path can reach an earlier round's directory -- in particular the Stage C round whose
        677 old launches stay untouched and read-only.
        """
        self.assertEqual(common.OUTPUT_DIR, common.HERE)
        for path in (common.LOGS, common.ROWS, common.CHECKPOINTS, common.REJECTED):
            self.assertTrue(str(path).startswith(str(common.OUTPUT_DIR)))
        for old in (common.STAGE_C_DIR, common.STAGE_B_DIR,
                    common.ROOT / "runs" / "comm_cost_stage_c_correctness_20260928"):
            self.assertFalse(str(common.OUTPUT_DIR).startswith(str(old)))
            for path in (common.LOGS, common.ROWS, common.CHECKPOINTS):
                self.assertFalse(str(path).startswith(str(old)))


if __name__ == "__main__":
    unittest.main()
