"""Independent cost-direction math through batch_samples and the real PPO actor path."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch

import sc_common as common
import sc_trainer as trainer
from sc_controller import fresh_controller
from sc_correctness_test_support import episode, record


class CostObjectiveCounterexample(unittest.TestCase):
    def test_positive_cost_penalty_matches_enumerated_send_hold_gradient(self) -> None:
        expected_gradients = {0.0: -0.05, 0.1: -0.025, 1.0: 0.2, 10.0: 2.45}
        epsilon = 1e-6

        for lam, expected_gradient in expected_gradients.items():
            with self.subTest(lambda_value=lam):
                controller = fresh_controller(11)
                with torch.no_grad():
                    for module in (controller.inner.action_head,
                                   controller.inner.value_head,
                                   controller.cost_head):
                        for parameter in module.parameters():
                            parameter.zero_()

                send = episode(
                    [record(controller, 1, value=0.0, cost_value=0.0, action=1)],
                    steps=1,
                    cost_stream={1: {"airtime_bytes": 0.2 * common.B0_BYTES}},
                    task_stream={1: 1.0},
                    lam=lam,
                )
                hold = episode(
                    [record(controller, 1, value=0.0, cost_value=0.0, action=0)],
                    steps=1,
                    lam=lam,
                )

                samples = trainer.batch_samples([send, hold])

                # Exact action enumeration gives J(theta)=sigmoid(theta)*(lambda-0.2).
                # This expected derivative is independent of actor_advantage / GAE code.
                def enumerated_objective(theta: float) -> float:
                    p_send = 1.0 / (1.0 + math.exp(-theta))
                    return p_send * (lam - 0.2)

                finite_difference = (enumerated_objective(epsilon)
                                     - enumerated_objective(-epsilon)) / (2.0 * epsilon)

                optimizer = torch.optim.SGD(controller.parameters, lr=0.01)
                captured_loss_gradients: list[float] = []
                original_step = optimizer.step

                def capture_then_step(*args, **kwargs):
                    # The action_head's SEND logit is theta while HOLD stays at zero.
                    captured_loss_gradients.append(
                        float(controller.inner.action_head.bias.grad[1].item()))
                    return original_step(*args, **kwargs)

                optimizer.step = capture_then_step
                with patch.object(common, "VALUE_COEFFICIENT", 0.0), \
                        patch.object(common, "ENTROPY_BONUS", 0.0), \
                        patch.object(common, "MAX_GRAD_NORM", 100.0):
                    trainer.ppo_update(controller, optimizer, [send, hold], update_index=0)

                # PPO minimizes the negative surrogate, so its loss gradient is -grad(J).
                self.assertEqual(len(captured_loss_gradients), common.EPOCHS_PER_BATCH)
                self.assertAlmostEqual(captured_loss_gradients[0], -expected_gradient,
                                       places=6)
                self.assertAlmostEqual(finite_difference, expected_gradient, places=8)
                self.assertAlmostEqual(samples[0]["return_task"], 1.0, places=9)
                self.assertAlmostEqual(samples[0]["return_cost"], 0.2, places=9)
                self.assertAlmostEqual(samples[0]["advantage"], lam - 0.2, places=9)
                self.assertAlmostEqual(samples[1]["return_task"], 0.0, places=9)
                self.assertAlmostEqual(samples[1]["return_cost"], 0.0, places=9)
                self.assertAlmostEqual(samples[1]["advantage"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
