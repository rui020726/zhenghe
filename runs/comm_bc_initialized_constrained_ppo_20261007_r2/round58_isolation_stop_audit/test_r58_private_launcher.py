from __future__ import annotations

import pathlib
import sys
import tempfile
import io
import json
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
        original_path = list(sys.path)
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
                self.assertTrue(runtime["r58_checkpoint_contract"]["wrapped_save"])
                self.assertTrue(runtime["r58_checkpoint_contract"]["wrapped_load"])
                self.assertTrue(runtime["r58_checkpoint_contract"]["wrapped_r40_train"])
                self.assertEqual(runtime["trainer"].TRAINING_VERSION,
                                 launcher.REPAIRED_TRAINING_VERSION)
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
                # Exercise the actual trainer load wrapper used by R40. A legacy checkpoint
                # is rejected before the canonical loader can construct/restore a controller.
                guarded_path = raw / "checkpoints" / "v2_train_s11.pt"
                restored = []
                deserialized = []
                old_restore = runtime["trainer"]._restore_controller
                with mock.patch.object(torch, "load",
                                       side_effect=lambda *_a, **_k: deserialized.append(True)), \
                     mock.patch.object(runtime["trainer"], "_restore_controller",
                                       side_effect=lambda *_a, **_k: restored.append(True)):
                    with self.assertRaisesRegex(RuntimeError, "before checkpoint deserialization"):
                        runtime["trainer"].load_checkpoint(
                            guarded_path, freeze=runtime["round42_freeze"])
                self.assertEqual(restored, [])
                self.assertEqual(deserialized, [])
                runtime["trainer"]._restore_controller = old_restore

            finally:
                launcher.close_audit_runtime(binding)
        self.assertEqual(sys.path, original_path)

    def test_cli_rejects_test_receipt_files_before_loading_runtime(self):
        fixture = launcher.GEN2 / "r58_execute.py"
        with mock.patch.object(launcher, "load_legacy_launcher",
                               side_effect=AssertionError("runtime loader must not run")):
            with self.assertRaisesRegex(PermissionError, "outside the repository"):
                launcher.main(["--authorization", str(fixture), "--freeze", str(fixture),
                               "--cumulative-quota", str(fixture), "--scene-audit", str(fixture),
                               "--output-root", str(launcher.PRIVATE_RAW)])
        external_fixture = pathlib.Path("C:/codex-test-only/approval.json")
        with mock.patch.object(pathlib.Path, "is_file", return_value=True), \
             mock.patch.object(pathlib.Path, "read_text",
                               return_value='{"fixture":true,"environment":"test"}'), \
             mock.patch.object(launcher, "load_legacy_launcher",
                               side_effect=AssertionError("runtime loader must not run")):
            with self.assertRaisesRegex(PermissionError, "test fixtures"):
                launcher.main(["--authorization", str(external_fixture),
                               "--freeze", str(external_fixture),
                               "--cumulative-quota", str(external_fixture),
                               "--scene-audit", str(external_fixture),
                               "--output-root", str(launcher.PRIVATE_RAW)])

    def test_validated_dispatch_calls_preserved_fixed_executor_and_stops_before_environment(self):
        # This is an in-process test-only grant; main/CLI cannot deserialize this object.
        class StopBeforeEnvironment(BaseException):
            pass
        legacy = launcher.load_legacy_launcher()
        repair = launcher._repair_identity()
        plan = legacy.plan_from_frozen_sources()
        plan.update({"training_version": launcher.REPAIRED_TRAINING_VERSION,
                     "r58_repair_identity": repair, "generation_id": "round58_gen3"})
        plan_sha = legacy.canonical_sha(plan)
        events = []
        real_fixed_gate = legacy.run_first_four_gate_then_continue
        def stop_before_episode(*_args, **_kwargs):
            events.append(("first-four-gate-called",))
            raise StopBeforeEnvironment()
        def forbidden(operation):
            def fail(*_args, **_kwargs):
                raise AssertionError(f"zero-operation route test reached forbidden {operation}")
            return fail
        def call_real_fixed_gate(base, module, runtime):
            events.append((runtime["trainer"], runtime["r40_train"].trainer,
                           runtime["r58_repaired_adapter"]["kind"],
                           runtime["r58_checkpoint_contract"]["kind"]))
            runtime["trainer"].restore_rng = forbidden("RNG restore")
            runtime["runner"].run_training_episode = forbidden("environment episode provider")
            runtime["r42_run"]._claim = forbidden("durable claim")
            base._run_first_four_and_all_gates = stop_before_episode
            return real_fixed_gate(base, module, runtime)
        legacy.run_first_four_gate_then_continue = call_real_fixed_gate
        with tempfile.TemporaryDirectory(dir=launcher.GEN2) as temp:
            raw = pathlib.Path(temp) / "private_raw"
            original_sys_path = list(sys.path)
            original_tracked_modules = {name: value for name, value in sys.modules.items()
                                        if name.startswith(("v2_", "r40_", "r42_", "r43_",
                                                            "r44_", "r45_", "r58_"))}
            auth_path = pathlib.Path("C:/codex-test-only/approval.json")
            receipt = {
                "fixture": True, "environment": "test",
                "status": "explicitly_authorized", "task_id": "ROUND58-PRIOR-STRENGTH-PROSPECTIVE-TRIAL",
                "branch": "exp/round58-prior-strength-prospective-trial",
                "normal_cap": 1024, "reserve_cap": 40, "hard_cap": 1064,
                "plan_sha256": plan_sha, "authorized_by": "unit-test",
                "approved_at": "unit-test-only", "approval_reference": "not-human-authorization",
            }
            grant = launcher._ValidatedGrant(launcher._GRANT_TOKEN, {
                "authorization_path": auth_path, "output_root": raw, "mode": "fresh",
                "repair_identity": repair, "reserve_cap": 40,
                "plan_sha256": plan_sha, "approval_receipt": receipt, "test_only": True,
            })
            writes = []
            original_write_text = pathlib.Path.write_text
            original_write_bytes = pathlib.Path.write_bytes
            original_open = pathlib.Path.open
            original_touch = pathlib.Path.touch
            def in_memory_text(path, data, *args, **kwargs):
                resolved = pathlib.Path(path).resolve()
                if raw.resolve() == resolved or raw.resolve() in resolved.parents:
                    writes.append(("text", str(resolved)))
                    return len(data)
                return original_write_text(path, data, *args, **kwargs)
            def in_memory_bytes(path, data, *args, **kwargs):
                resolved = pathlib.Path(path).resolve()
                if raw.resolve() == resolved or raw.resolve() in resolved.parents:
                    writes.append(("bytes", str(resolved)))
                    return len(data)
                return original_write_bytes(path, data, *args, **kwargs)
            def in_memory_open(path, mode="r", *args, **kwargs):
                resolved = pathlib.Path(path).resolve()
                if (raw.resolve() == resolved or raw.resolve() in resolved.parents) \
                        and any(flag in mode for flag in ("w", "a", "x", "+")):
                    writes.append(("open", str(resolved), mode))
                    return io.BytesIO() if "b" in mode else io.StringIO()
                return original_open(path, mode, *args, **kwargs)
            def in_memory_touch(path, *args, **kwargs):
                resolved = pathlib.Path(path).resolve()
                if raw.resolve() == resolved or raw.resolve() in resolved.parents:
                    writes.append(("touch", str(resolved)))
                    return path
                return original_touch(path, *args, **kwargs)
            def guarded_torch_save(*_a, **_k):
                raise AssertionError("checkpoint disk serialization forbidden in route test")
            # The actual fixed executor/configure runs, but the first-batch boundary raises
            # before its episode provider. All generation writes are memory-only.
            with mock.patch.object(pathlib.Path, "write_text", in_memory_text), \
                 mock.patch.object(pathlib.Path, "write_bytes", in_memory_bytes), \
                 mock.patch.object(pathlib.Path, "open", in_memory_open), \
                 mock.patch.object(pathlib.Path, "touch", in_memory_touch), \
                 mock.patch.object(torch, "save", guarded_torch_save), \
                 mock.patch.object(torch.optim.Adam, "step",
                                   side_effect=AssertionError("real optimizer step forbidden")):
                with self.assertRaises(StopBeforeEnvironment):
                    launcher._dispatch_validated(grant, legacy_loader=lambda: legacy)
            legacy.run_first_four_gate_then_continue = real_fixed_gate
            self.assertEqual(len(events), 2)
            self.assertIs(events[0][0], events[0][1])
            self.assertEqual(events[0][2], "r58-gate-prior-repaired/1")
            self.assertEqual(events[0][3], "round58-r40-checkpoint-contract/1")
            self.assertEqual(events[1], ("first-four-gate-called",))
            self.assertEqual(list(raw.rglob("launch_claims/**/*.json")), [])
            self.assertEqual(list(raw.rglob("*.pt")), [])
            self.assertEqual(list(raw.rglob("r40_batch_commit*.json")), [])
            output_files = [path for path in raw.rglob("*") if path.is_file()]
            self.assertTrue(all(path.suffix in (".py", ".pyc") for path in output_files),
                            f"unexpected durable task/ledger output: {output_files}")
            self.assertTrue(all(str(raw.resolve()) in event[1] for event in writes))
            self.assertEqual(sys.path, original_sys_path)
            for name, module in original_tracked_modules.items():
                self.assertIs(sys.modules.get(name), module)

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
