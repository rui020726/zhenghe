from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path[:0] = [str(HERE), str(ROOT / "round58_gen2"),
                str(ROOT / "round53_critic_isolation_trial_readiness"),
                str(HERE.parents[2] / "runs" / "comm_rule_warmstart_cost_learning_20261006_r2")]

import r45_bc  # noqa: E402
r45_bc.configure_source_imports()
import v2_common  # noqa: E402
import v2_trainer  # noqa: E402
import r58_gate_prior_repaired as repaired_adapter  # noqa: E402
import r58_repaired_entry as entry  # noqa: E402
from round53_trial_adapter import critic_input_detached  # noqa: E402


class RepairedEntryRegression(unittest.TestCase):
    def test_zero_launch_binding_imports_repaired_adapter_on_exact_runtime_objects(self):
        self.assertEqual(entry.adapter.__file__, repaired_adapter.__file__)
        runtime = {"trainer": v2_trainer,
                   "r40_train": SimpleNamespace(trainer=None),
                   "common": v2_common,
                   "critic_input_detached": critic_input_detached}
        with tempfile.TemporaryDirectory(dir=HERE) as output:
            receipt = entry.bind_repaired_update(
                runtime, pathlib.Path(output), critic_input_detached,
                writer=lambda *_args: None)
            try:
                self.assertIs(runtime["r40_train"].trainer, runtime["trainer"])
                self.assertEqual(runtime["r58_repaired_adapter"]["kind"],
                                 "r58-gate-prior-repaired/1")
                self.assertEqual(receipt["trainer_sha256"], entry.EXPECTED_TRAINER_SHA256)
                self.assertEqual(runtime["r58_live_configuration"]["EPOCHS_PER_BATCH"], 4)
                self.assertEqual(runtime["r58_live_configuration"]["MINIBATCH_STEPS"], 256)
            finally:
                receipt["undo"]()

    def test_old_r58_checkpoint_is_model_readable_but_resume_rejected(self):
        old_checkpoint = ROOT / "round58_gen2" / "raw" / "checkpoints" / "v2_train_s11_after_b0.pt"
        payload = torch.load(old_checkpoint, map_location="cpu", weights_only=False)
        model = entry.model_only_state(payload)
        self.assertEqual(model["kind"], "r58-model-only-read/1")
        self.assertIn("trunk", model["controller"])
        self.assertNotIn("optimizer", model)
        self.assertNotIn("rng", model)
        self.assertIn("r53-critic-representation-stop-gradient", model["training_version"])
        expected = {"critic_route_version": entry.ROUTE_VERSION, **entry.adapter_identity()}
        with self.assertRaisesRegex(RuntimeError, "lacks per-update R58 repair attestation"):
            entry.require_repaired_resume(payload, expected)

    def test_resume_requires_matching_checkpoint_state_and_freeze_proof(self):
        expected = {"critic_route_version": entry.ROUTE_VERSION, **entry.adapter_identity()}
        payload = {"round58_repair_attestation": expected,
                   "training_state": {"round58_repair_attestation": expected},
                   "freeze": {"round58_repair_attestation": expected}}
        entry.require_repaired_resume(payload, expected)
        payload["freeze"] = {}
        with self.assertRaisesRegex(RuntimeError, "do not close"):
            entry.require_repaired_resume(payload, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
