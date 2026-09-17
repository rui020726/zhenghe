"""Scalar calibration of the CTDE task/depot preference.

The allocation trace showed that the uncalibrated CTDE actor picks the depot on
94.9% of the decisions where a legal task was on the menu, while the raw
task-vs-depot logit margin has a median of -0.03 -- a near-indifference, not a
strong wrong preference. ``--ctde-task-logit-offset`` adds a constant to every
legal task logit before the coordinator residual, so this hypothesis ("the depot
problem is scalar calibration") can be tested without retraining anything.

What must hold:

* the default (0.0) is bit-for-bit the old behaviour;
* the offset lands on legal task actions only -- never on the depot, never on a
  masked task, and no offset can make a masked action selectable;
* the coordinator residual is still applied after the offset and the environment
  hard mask is still applied last;
* deterministic inference keeps taking the argmax, the trace records the offset
  so the final logit can be re-derived, and the checkpoint path is untouched.
"""

from __future__ import annotations

import importlib.util
import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any

from integrated_sar.allocation_trace import DecisionTraceWriter
from integrated_sar.contracts import CoordinationAdvice
from integrated_sar.errors import ConfigurationError
from integrated_sar.joint_driver import (
    RunConfiguration,
    build_parser,
    configuration_from_args,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None

ACTION_MAP = (("depot", None), ("task", 7), ("task", 8))


def _make_env(mask: Any) -> Any:
    import numpy as np

    class _StubEnv:
        """Just enough TaskEnv for a network verdict and one traced decision."""

        def __init__(self) -> None:
            self.current_time = 3.0
            self._inner_step_counter = 3
            self.task_dic: dict[int, Any] = {}
            self.agent_dic = {
                0: {
                    "location": np.array([0.4, 0.0]),
                    "target_pos": np.array([0.5, 0.0]),
                    "target_type": None,
                    "target_id": None,
                    "current_task": -1,
                    "abilities": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
                    "status": "idle",
                    "travel_dist": 0.0,
                }
            }

        def get_local_obs(self, agent_id: int, max_tasks: int | None = None) -> Any:
            del agent_id, max_tasks
            return (
                np.zeros(4, dtype=np.float32),
                np.zeros((2, 4), dtype=np.float32),
                np.zeros((len(ACTION_MAP), 4), dtype=np.float32),
                mask,
                list(ACTION_MAP),
            )

    return _StubEnv()


def _fixed_network(logits: Any) -> Any:
    """A stand-in for old_network: fixed logits, mask applied inside get_action."""
    import torch

    from integrated_sar.adapters.task03 import MASKED_LOGIT

    class _Network:
        def parameters(self) -> Any:
            return iter([torch.zeros(1)])

        def get_action(self, state: Any, neighbors: Any, tasks: Any, hard_mask: Any) -> Any:
            del state, neighbors, tasks
            values = logits.clone()
            return SimpleNamespace(logits=values.masked_fill(hard_mask.bool(), MASKED_LOGIT))

    return _Network()


def _make_adapter(
    *, offset: float, mask: Any, logits: Any, deterministic: bool = True, **kwargs: Any
) -> Any:
    import numpy as np

    from integrated_sar.adapters.task03 import Task03Adapter

    adapter = Task03Adapter(
        env=_make_env(mask),
        network=_fixed_network(logits),
        deterministic=deterministic,
        clear_initial_tasks=False,
        residual_scale=0.0,
        ctde_task_logit_offset=offset,
        **kwargs,
    )
    adapter.internal_to_external = {7: "target-a", 8: "target-b"}
    return adapter


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TaskLogitOffsetSemanticsTests(unittest.TestCase):
    """Where the offset lands, and where it must never land."""

    def _verdict(self, *, offset: float, advice: CoordinationAdvice | None = None) -> Any:
        import numpy as np
        import torch

        mask = np.array([False, False, True], dtype=bool)
        logits = torch.tensor([[0.50, 0.10, 0.05]])
        adapter = _make_adapter(offset=offset, mask=mask, logits=logits)
        return adapter._network_verdict(
            0,
            list(ACTION_MAP),
            advice
            or CoordinationAdvice(revision=1, generated_at=0.0, valid_until=10.0),
        )

    def test_a_zero_offset_reproduces_the_uncalibrated_verdict_bit_for_bit(self) -> None:
        # The default value and an explicit 0.0 must agree with each other *and* with
        # the pre-calibration code path, whose only observable is these vectors.
        default = self._verdict(offset=0.0)
        self.assertAlmostEqual(default.ctde[0], 0.50)
        self.assertAlmostEqual(default.ctde[1], 0.10)
        self.assertEqual(default.final, default.ctde)
        self.assertEqual(default.bias, (0.0, 0.0, None))
        self.assertEqual(default.index, 0)

    def test_the_offset_is_added_to_legal_task_logits_only(self) -> None:
        # 0.45 is just enough to overtake a depot logit of 0.50, which stays at 0.50.
        verdict = self._verdict(offset=0.45)
        # The depot keeps its logit, the legal task rises by exactly the offset, and
        # the masked task stays masked rather than "very unlikely".
        self.assertAlmostEqual(verdict.final[0], 0.50)
        self.assertAlmostEqual(verdict.final[1], 0.10 + 0.45)
        self.assertIsNone(verdict.final[2])
        self.assertAlmostEqual(verdict.ctde[1], 0.10, places=6)
        self.assertEqual(verdict.index, 1)

    def test_an_arbitrarily_large_offset_cannot_select_a_masked_task(self) -> None:
        verdict = self._verdict(offset=1e12)
        self.assertNotEqual(verdict.index, 2, "a masked task became selectable again")
        self.assertIsNone(verdict.final[2])
        self.assertEqual(verdict.index, 1)

    def test_the_offset_can_flip_a_near_indifferent_depot_choice(self) -> None:
        import numpy as np
        import torch

        mask = np.array([False, False, True], dtype=bool)
        logits = torch.tensor([[0.30, 0.10, 0.05]])
        without = _make_adapter(offset=0.0, mask=mask, logits=logits)._network_verdict(
            0, list(ACTION_MAP), CoordinationAdvice(revision=1, generated_at=0.0, valid_until=10.0)
        )
        with_offset = _make_adapter(offset=0.25, mask=mask, logits=logits)._network_verdict(
            0, list(ACTION_MAP), CoordinationAdvice(revision=1, generated_at=0.0, valid_until=10.0)
        )
        self.assertEqual(without.index, 0)
        self.assertEqual(with_offset.index, 1)

    def test_the_coordinator_residual_is_added_after_the_offset(self) -> None:
        import numpy as np
        import torch

        mask = np.array([False, False, True], dtype=bool)
        logits = torch.tensor([[0.50, 0.10, 0.05]])
        adapter = _make_adapter(offset=0.30, mask=mask, logits=logits)
        adapter.residual_scale = 0.5
        advice = CoordinationAdvice(
            revision=1,
            generated_at=0.0,
            valid_until=10.0,
            pair_bias={("0", "target-a"): 0.4},
        )
        verdict = adapter._network_verdict(0, list(ACTION_MAP), advice)
        # final = ctde + offset + residual, and the depot only ever gets ctde.
        self.assertAlmostEqual(verdict.final[0], 0.50)
        self.assertAlmostEqual(verdict.final[1], 0.10 + 0.30 + 0.5 * 0.4)
        self.assertAlmostEqual(verdict.bias[1], 0.5 * 0.4)

    def test_deterministic_inference_still_takes_the_argmax(self) -> None:
        import numpy as np
        import torch

        mask = np.array([False, False, True], dtype=bool)
        logits = torch.tensor([[0.20, 0.90, 0.05]])
        adapter = _make_adapter(
            offset=0.30, mask=mask, logits=logits, deterministic=True
        )
        verdict = adapter._network_verdict(
            0, list(ACTION_MAP), CoordinationAdvice(revision=1, generated_at=0.0, valid_until=10.0)
        )
        self.assertEqual(verdict.index, 1)
        self.assertAlmostEqual(verdict.final[1], 1.20)

    def test_the_offset_is_visible_in_the_decision_trace(self) -> None:
        import json

        import numpy as np
        import torch

        from integrated_sar.allocation_trace import trace_meta

        mask = np.array([False, False, True], dtype=bool)
        logits = torch.tensor([[0.50, 0.10, 0.05]])
        adapter = _make_adapter(offset=0.45, mask=mask, logits=logits)
        stream = io.StringIO()
        # Built the way the driver builds it, so a header that lost the offset would
        # fail here even if the adapter still recorded it per decision.
        adapter.decision_trace = DecisionTraceWriter(
            stream,
            meta=trace_meta(RunConfiguration(ctde_task_logit_offset=0.45, seed=1)),
        )
        agent = SimpleNamespace(agent_id="0", needs_decision=True, known_tasks=("target-a",))
        commands = adapter.allocate(
            (), (agent,), CoordinationAdvice(revision=1, generated_at=3.0, valid_until=13.0), 3.0
        )
        adapter.decision_trace.close()

        rows = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
        header = next(row for row in rows if row["event_type"] == "trace_config")
        self.assertEqual(header["meta"]["ctde_task_logit_offset"], 0.45)
        decision = next(row for row in rows if row["event_type"] == "decision")
        self.assertEqual(decision["task_logit_offset"], 0.45)
        # final can be re-derived from the three recorded numbers.
        delta = decision["final_logits"][1] - decision["ctde_logits"][1]
        self.assertAlmostEqual(delta, 0.45, places=6)
        self.assertEqual(commands[0].task_id, "target-a")


class TaskLogitOffsetConfigurationTests(unittest.TestCase):
    """The switch is validated and serialised like every other run parameter."""

    def test_the_cli_accepts_the_offset_and_records_it(self) -> None:
        args = build_parser().parse_args(["--ctde-task-logit-offset", "0.15"])
        configuration = configuration_from_args(args)
        configuration.validate()
        self.assertEqual(configuration.ctde_task_logit_offset, 0.15)
        self.assertEqual(configuration.as_dict()["ctde_task_logit_offset"], 0.15)

    def test_the_default_configuration_carries_a_zero_offset(self) -> None:
        configuration = RunConfiguration()
        configuration.validate()
        self.assertEqual(configuration.ctde_task_logit_offset, 0.0)

    def test_a_negative_offset_is_refused(self) -> None:
        with self.assertRaises(ConfigurationError) as caught:
            RunConfiguration(ctde_task_logit_offset=-0.1).validate()
        self.assertIn("--ctde-task-logit-offset", str(caught.exception))

    def test_a_non_finite_offset_is_refused(self) -> None:
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ConfigurationError):
                RunConfiguration(ctde_task_logit_offset=value).validate()


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class CalibratedCheckpointRunTests(unittest.TestCase):
    """Calibration must not touch the checkpoint path: strict load still holds."""

    def test_a_calibrated_run_still_loads_the_checkpoint_strictly(self) -> None:
        import torch

        from integrated_sar.joint_driver import run_from_configuration
        from integrated_sar.task03_runtime import _load_network_module, load_ctde_network
        from integrated_sar.tests.test_joint_driver import make_configuration

        module = _load_network_module(None)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ctde_best.pth")
            torch.save(
                {"episode": 1, "model_state_dict": module.CTDE_Network().state_dict()}, path
            )
            network = load_ctde_network(path)
            self.assertEqual(network.checkpoint_episode, 1)

            configuration = make_configuration(
                directory,
                mock=False,
                task_checkpoint=path,
                output=os.path.join(directory, "ctde.jsonl"),
                ctde_task_logit_offset=0.25,
            )
            summary = run_from_configuration(configuration)

        self.assertEqual(summary["ctde_task_logit_offset"], 0.25)
        self.assertEqual(summary["allocation_backend"], "ctde-network")