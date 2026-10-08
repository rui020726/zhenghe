from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import r58_private_launcher_repaired as launcher


class PrivateLauncherRouteTests(unittest.TestCase):
    def test_default_execution_is_denied_before_runtime_configuration(self):
        with mock.patch.object(launcher, "load_legacy_launcher",
                               side_effect=AssertionError("runtime loader must not be reached")):
            with self.assertRaises(PermissionError):
                launcher.execute_authorized()

    def test_actual_configured_r40_r53_runtime_binds_repaired_adapter_without_execution(self):
        with tempfile.TemporaryDirectory(dir=launcher.GEN2) as temp:
            raw = pathlib.Path(temp) / "private_raw"
            binding = launcher.prepare_actual_runtime_for_audit(raw)
            runtime = binding["runtime"]
            try:
                self.assertIs(runtime["r40_train"].trainer, runtime["trainer"])
                self.assertEqual(binding["trainer_object_id"], binding["r40_trainer_object_id"])
                self.assertEqual(runtime["r58_repaired_adapter"]["kind"], "r58-gate-prior-repaired/1")
                self.assertEqual(runtime["r58_repaired_adapter"]["beta"], 1.0)
                self.assertTrue(runtime["trainer"].ppo_update.__name__.startswith("r58_"))
                self.assertFalse((raw / "runtime" / "checkpoint_writes.jsonl").exists())
                old_path = launcher.GEN2 / "raw" / "checkpoints" / "v2_train_s11_after_b0.pt"
                payload = torch.load(old_path, map_location="cpu", weights_only=False)
                expected = {"critic_route_version": launcher.resume_guard.ROUTE_VERSION,
                            **launcher.resume_guard.adapter_identity()}
                calls = []
                callbacks = {name: (lambda *_args, n=name: calls.append(n))
                             for name in ("state_writeback", "model", "optimizer", "rng",
                                          "environment", "step")}
                with self.assertRaisesRegex(RuntimeError, "lacks per-update R58 repair attestation"):
                    launcher.resume_from_payload_before_any_restore(
                        payload, expected,
                        write_state=callbacks["state_writeback"],
                        restore_model=callbacks["model"],
                        restore_optimizer=callbacks["optimizer"],
                        restore_rng=callbacks["rng"],
                        construct_environment=callbacks["environment"],
                        optimizer_step=callbacks["step"])
                self.assertEqual(calls, [])
                self.assertFalse(any(raw.rglob("*.pt")))
                self.assertFalse(any(raw.rglob("r40_batch_commit*.json")))
            finally:
                launcher.close_audit_runtime(binding)

    def test_legacy_checkpoint_rejected_before_restore_rng_environment_or_update(self):
        old_path = launcher.GEN2 / "raw" / "checkpoints" / "v2_train_s11_after_b0.pt"
        payload = torch.load(old_path, map_location="cpu", weights_only=False)
        expected = {"critic_route_version": launcher.resume_guard.ROUTE_VERSION,
                    **launcher.resume_guard.adapter_identity()}
        calls = []
        callbacks = {name: (lambda *_args, n=name: calls.append(n))
                     for name in ("state_writeback", "model", "optimizer", "rng", "environment", "step")}
        with self.assertRaisesRegex(RuntimeError, "lacks per-update R58 repair attestation"):
            launcher.resume_from_payload_before_any_restore(
                payload, expected,
                write_state=callbacks["state_writeback"],
                restore_model=callbacks["model"], restore_optimizer=callbacks["optimizer"],
                restore_rng=callbacks["rng"], construct_environment=callbacks["environment"],
                optimizer_step=callbacks["step"])
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
