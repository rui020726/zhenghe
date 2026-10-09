from __future__ import annotations

import pathlib
import sys
import tempfile
import io
import json
import subprocess
import os
import runpy
import importlib.util
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import r58_private_launcher_repaired as launcher


class PrivateLauncherRouteTests(unittest.TestCase):
    def test_old_main_before_definitions_reproduces_nameerror_and_new_module_help_exits_zero(self):
        source_path = pathlib.Path(launcher.__file__).resolve()
        repo_root = source_path.parents[2]
        historical_repo_path = pathlib.Path(
            "runs/comm_bc_initialized_constrained_ppo_20261007_r2/round58_isolation_stop_audit/"
            "r58_private_launcher_repaired.py")
        old_source = subprocess.run(
            ["git", "show", "b38b8688dee7ffc2626aa2064c8af74af75e216b:" +
             historical_repo_path.as_posix()],
            cwd=repo_root, check=True, capture_output=True, text=True).stdout
        harness = (
            "import sys; from pathlib import Path; "
            "sys.path.insert(0, r'" + str(source_path.parent) + "'); "
            "Path.is_file=lambda self: True; Path.read_text=lambda self,*a,**k:'{}'; "
            "sys.argv=['r58_private_launcher_repaired.py','--authorization','C:/v/a.json',"
            "'--freeze','C:/v/f.json','--cumulative-quota','C:/v/q.json',"
            "'--scene-audit','C:/v/s.json','--output-root','C:/v/round58_gen3/raw']; "
            "exec(compile(sys.stdin.read(), sys.argv[0], 'exec'), "
            "{'__name__':'__main__','__file__':r'" + str(source_path) + "'})"
        )
        old = subprocess.run([sys.executable, "-c", harness], input=old_source,
                             capture_output=True, text=True, cwd=repo_root)
        self.assertEqual(old.returncode, 1)
        self.assertIn("NameError", old.stderr)
        self.assertIn("_validate_external_grant", old.stderr)
        new = subprocess.run([sys.executable, str(source_path), "--help"],
                             capture_output=True, text=True, cwd=repo_root)
        self.assertEqual(new.returncode, 0, new.stderr)
        self.assertIn("--authorization", new.stdout)

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
                self.assertEqual(runtime["r58_ppo_counter_bridge"]["kind"],
                                 "round58-ppo-counter-bridge/2")
                self.assertIs(runtime["r58_ppo_counter_bridge"]["counter_object"],
                              runtime["r42_run"])
                self.assertIs(runtime["r40_train"].trainer, runtime["trainer"])
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
        fixture = launcher.HERE / "virtual-fixture-approval.json"
        with mock.patch.object(pathlib.Path, "is_file", return_value=True), \
             mock.patch.object(pathlib.Path, "read_text",
                               return_value='{"fixture":true,"environment":"test"}'), \
             mock.patch.object(launcher, "load_legacy_launcher",
                               side_effect=AssertionError("runtime loader must not run")):
            with self.assertRaisesRegex(PermissionError, "test fixtures"):
                launcher.main(["--authorization", str(fixture),
                               "--freeze", str(fixture),
                               "--cumulative-quota", str(fixture),
                               "--scene-audit", str(fixture),
                               "--output-root", str(launcher.PRIVATE_RAW)])

    def test_actual_main_validates_virtual_receipts_and_hands_off_at_first_four_gate(self):
        # All receipts, including the hypothetical future approval, exist only in this
        # in-memory VFS. They are not persisted as executable authorization artifacts.
        class StopBeforeEnvironment(BaseException):
            pass
        legacy = launcher.load_legacy_launcher()
        repair = launcher._repair_identity()
        contracts = sys.modules["r58_contracts"]
        original_contract_cap, original_legacy_cap = contracts.RESERVE_CAP, legacy.RESERVE_CAP
        contracts.RESERVE_CAP = legacy.RESERVE_CAP = 24
        plan = legacy.plan_from_frozen_sources()
        contracts.RESERVE_CAP, legacy.RESERVE_CAP = original_contract_cap, original_legacy_cap
        plan.update({"training_version": launcher.REPAIRED_TRAINING_VERSION,
                     "r58_repair_identity": repair, "generation_id": "round58_gen3",
                     "status": "frozen_approved_gen3",
                     "authorization": "direct_user_owner_approval_identity_bound"})
        plan_sha = legacy.canonical_sha(plan)
        bc_models, bc_sha = launcher._bc_model_identity()
        resume_sources = launcher.resume_seam.validate_resume_sources(launcher.REPO_ROOT)
        # The disposable root is a sibling of gen2 under ROUND58's output parent,
        # matching the actual gen3 containment semantics (never a gen2 subdirectory).
        with tempfile.TemporaryDirectory(prefix="r58_gen3_route_test_", dir=launcher.RUN_ROOT) as temp:
            old_generation, old_raw = launcher.PRIVATE_GENERATION, launcher.PRIVATE_RAW
            generation = pathlib.Path(temp) / "round58_gen3"
            raw = generation / "raw"
            launcher.PRIVATE_GENERATION, launcher.PRIVATE_RAW = generation, raw
            auth_path = launcher.HERE / "virtual-future-approval.json"
            freeze_path = launcher.HERE / "virtual-future-freeze.json"
            quota_path = launcher.HERE / "virtual-future-quota.json"
            scene_path = launcher.HERE / "virtual-future-scene-audit.json"
            history_path = launcher.HERE / "virtual-future-history.json"
            scene = {"kind": "round58-scene-occupancy-audit/1", "status": "clear",
                     "scenes": list(range(1_700_000, 1_700_032)), "conflicts": [],
                     "result_blind": True,
                     "checked_sources": {k: True for k in
                                          ("registrations", "reservations", "claims", "artifacts")},
                     "audited_at_utc": "2099-01-01T00:00:00Z"}
            history = {"environment_consumption": {
                "known_constructor_and_run_calls": 16,
                "reserve_classification": "unreconciled: preserved historical category conflict",
                "gen1_complete_claims": 4, "gen2_complete_claims": 12,
                "gen1_categories": {"normal": 4, "reserve": 0},
                "gen2_claim_categories": {"normal": 12, "reserve": 0},
                "gen2_registration_says_reserve_from_gen1_failure": 4}}
            def encoded(value):
                return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")) + "\n").encode("utf-8")
            history_bytes = encoded(history)
            history_sha = __import__("hashlib").sha256(history_bytes).hexdigest()
            scene_bytes = encoded(scene)
            scene_sha = __import__("hashlib").sha256(scene_bytes).hexdigest()
            quota = {"kind": "round58-cumulative-quota-receipt/2",
                     "status": "reconciled_with_preserved_category_conflict",
                     "historical_consumed": 16,
                     "historical_category_disposition": "unresolved_preserved_no_relabel",
                     "planned_normal": 1024, "new_normal_cap": 1024,
                     "new_reserve_cap": 24, "normal_remaining": 1024,
                     "reserve_remaining": 24, "hard_remaining": 1048,
                     "hard_cap": 1064,
                     "historical_manifest_path": str(history_path),
                     "historical_manifest_sha256": history_sha}
            quota_bytes = encoded(quota)
            quota_sha = __import__("hashlib").sha256(quota_bytes).hexdigest()
            freeze = {"kind": "round58-repaired-execution-freeze/1", "status": "frozen",
                      "plan_sha256": plan_sha,
                      "training_version": launcher.REPAIRED_TRAINING_VERSION,
                      "repair_identity": repair, "bc_model_identity_sha256": bc_sha,
                      "execution_basis_sha256": launcher._execution_basis_identity(),
                      "output_root": str(raw.resolve()), "generation_id": "round58_gen3",
                      "checkpoint_root": str(resume_sources["target_checkpoint_root"]),
                      "mode": "fresh", "cumulative_quota_sha256": quota_sha,
                      "reserve_cap": 24, "scene_audit_sha256": scene_sha,
                      "bc_models": bc_models}
            freeze_bytes = encoded(freeze)
            freeze_sha = __import__("hashlib").sha256(freeze_bytes).hexdigest()
            approval = {"status": "explicitly_authorized",
                        "task_id": "ROUND58-PRIOR-STRENGTH-PROSPECTIVE-TRIAL",
                        "branch": "exp/round58-prior-strength-prospective-trial",
                        "normal_cap": 1024, "reserve_cap": 24, "hard_cap": 1064,
                        "plan_sha256": plan_sha, "generation_id": "round58_gen3",
                        "repair_identity": repair, "output_root": str(raw.resolve()),
                        "checkpoint_root": str(resume_sources["target_checkpoint_root"]),
                        "bc_model_identity_sha256": bc_sha,
                        "execution_basis_sha256": launcher._execution_basis_identity(),
                        "freeze_sha256": freeze_sha, "quota_sha256": quota_sha,
                        "scene_audit_sha256": scene_sha,
                        "authorized_by": "synthetic-test-principal",
                        "approved_at": "2099-01-01T00:00:00Z",
                        "approval_reference": "TEST-ONLY-NOT-A-REAL-APPROVAL"}
            files = {str(auth_path.resolve()): encoded(approval),
                     str(freeze_path.resolve()): freeze_bytes,
                     str(quota_path.resolve()): quota_bytes,
                     str(scene_path.resolve()): scene_bytes,
                     str(history_path.resolve()): history_bytes}
            events = []
            def stop_before_episode(*_args, **_kwargs):
                events.append(("first-four-gate-called",))
                raise StopBeforeEnvironment()
            def forbidden(operation):
                def fail(*_args, **_kwargs):
                    raise AssertionError(f"zero-operation route test reached forbidden {operation}")
                return fail
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
            original_sys_path = list(sys.path)
            original_tracked_modules = {name: value for name, value in sys.modules.items()
                                        if name.startswith(("v2_", "r40_", "r42_", "r43_",
                                                            "r44_", "r45_", "r58_"))}
            original_is_file = pathlib.Path.is_file
            original_read_text = pathlib.Path.read_text
            original_read_bytes = pathlib.Path.read_bytes
            original_resolve = pathlib.Path.resolve
            fixed_generation = original_resolve(launcher.RUN_ROOT / "round58_gen3")
            def map_gen3_resolve(path, *args, **kwargs):
                resolved = original_resolve(path, *args, **kwargs)
                try:
                    suffix = resolved.relative_to(fixed_generation)
                except ValueError:
                    return resolved
                return original_resolve(generation) / suffix
            class GateStoppingLoader:
                def __init__(self, wrapped): self.wrapped = wrapped
                def create_module(self, spec):
                    return self.wrapped.create_module(spec) if hasattr(self.wrapped, "create_module") else None
                def exec_module(self, loaded):
                    self.wrapped.exec_module(loaded)
                    real_gate = loaded.run_first_four_gate_then_continue
                    def gate(base, module, runtime):
                        common_root = pathlib.Path(runtime["common"].CHECKPOINTS).resolve()
                        self_outer.assertEqual(common_root, raw.resolve() / "checkpoints")
                        self_outer.assertEqual(loaded.HERE, launcher.GEN2)
                        for forbidden_root in (launcher.GEN2 / "raw", raw.parent.parent / "escape"):
                            with self_outer.assertRaisesRegex(RuntimeError, "output escapes the isolated root"):
                                loaded.bind_runner_paths(base, forbidden_root)
                        events.append((runtime["trainer"], runtime["r40_train"].trainer,
                                       runtime["r58_repaired_adapter"]["kind"],
                                       runtime["r58_checkpoint_contract"]["kind"], common_root))
                        runtime["trainer"].restore_rng = forbidden("RNG restore")
                        runtime["runner"].run_training_episode = forbidden("environment episode provider")
                        runtime["r42_run"]._claim = forbidden("durable claim")
                        base._run_first_four_and_all_gates = stop_before_episode
                        return real_gate(base, module, runtime)
                    loaded.run_first_four_gate_then_continue = gate
            self_outer = self
            original_spec = importlib.util.spec_from_file_location
            def guarded_spec(name, location, *args, **kwargs):
                spec = original_spec(name, location, *args, **kwargs)
                if spec is not None and str(pathlib.Path(location).resolve()) == str(
                        (launcher.GEN2 / "r58_execute.py").resolve()):
                    spec.loader = GateStoppingLoader(spec.loader)
                return spec
            def virtual_is_file(path):
                if str(path.resolve()) in files: return True
                return original_is_file(path)
            def virtual_read_text(path, *args, **kwargs):
                data = files.get(str(path.resolve()))
                if data is not None: return data.decode(kwargs.get("encoding") or "utf-8")
                return original_read_text(path, *args, **kwargs)
            def virtual_read_bytes(path):
                data = files.get(str(path.resolve()))
                return data if data is not None else original_read_bytes(path)
            # Run the source as __main__: this exercises the actual CLI entry route,
            # receipt validation, fixed-plan configure/bind and first-four handoff.
            # Path resolution maps only the approved gen3/raw namespace into this
            # disposable test location; no synthetic approval is written to disk.
            old_argv = sys.argv[:]
            sys.argv = [str(launcher.__file__), "--authorization", str(auth_path),
                        "--freeze", str(freeze_path), "--cumulative-quota", str(quota_path),
                        "--scene-audit", str(scene_path), "--output-root",
                        str(launcher.HERE.parent / "round58_gen3" / "raw")]
            try:
                with mock.patch.object(pathlib.Path, "is_file", virtual_is_file), \
                     mock.patch.object(pathlib.Path, "read_text", virtual_read_text), \
                     mock.patch.object(pathlib.Path, "read_bytes", virtual_read_bytes), \
                     mock.patch.object(pathlib.Path, "resolve", map_gen3_resolve), \
                     mock.patch.object(pathlib.Path, "write_text", in_memory_text), \
                     mock.patch.object(pathlib.Path, "write_bytes", in_memory_bytes), \
                     mock.patch.object(pathlib.Path, "open", in_memory_open), \
                     mock.patch.object(pathlib.Path, "touch", in_memory_touch), \
                     mock.patch.object(importlib.util, "spec_from_file_location", guarded_spec), \
                     mock.patch.object(torch, "save", guarded_torch_save), \
                     mock.patch.object(torch.optim.Adam, "step",
                                       side_effect=AssertionError("real optimizer step forbidden")):
                    with self.assertRaises(StopBeforeEnvironment):
                        runpy.run_path(str(launcher.__file__), run_name="__main__")
            finally:
                sys.argv = old_argv
            self.assertEqual(len(events), 2)
            self.assertIs(events[0][0], events[0][1])
            self.assertEqual(events[0][2], "r58-gate-prior-repaired/1")
            self.assertEqual(events[0][3], "round58-r40-checkpoint-contract/1")
            self.assertEqual(events[0][4], raw.resolve() / "checkpoints")
            self.assertEqual(events[1], ("first-four-gate-called",))
            self.assertEqual(list(raw.rglob("launch_claims/**/*.json")), [])
            self.assertEqual(list(raw.rglob("*.pt")), [])
            self.assertEqual(list(raw.rglob("r40_batch_commit*.json")), [])
            output_files = [path for path in raw.rglob("*") if path.is_file()]
            self.assertTrue(all(path.suffix in (".py", ".pyc") for path in output_files),
                            f"unexpected durable task/ledger output: {output_files}")
            self.assertTrue(all(str(raw.resolve()) in event[1] for event in writes))
            self.assertEqual(sys.path, original_sys_path)
            launcher.PRIVATE_GENERATION, launcher.PRIVATE_RAW = old_generation, old_raw
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


class CumulativeQuotaV2Tests(unittest.TestCase):
    def setUp(self):
        self.history = {"environment_consumption": {
            "known_constructor_and_run_calls": 16,
            "gen1_complete_claims": 4, "gen2_complete_claims": 12,
            "gen1_categories": {"normal": 4, "reserve": 0},
            "gen2_claim_categories": {"normal": 12, "reserve": 0},
            "gen2_registration_says_reserve_from_gen1_failure": 4,
            "reserve_classification": "unreconciled: preserve conflicting source fields"}}
        self.quota = {"kind": "round58-cumulative-quota-receipt/2",
                      "status": "reconciled_with_preserved_category_conflict",
                      "historical_consumed": 16,
                      "historical_category_disposition": "unresolved_preserved_no_relabel",
                      "planned_normal": 1024, "new_normal_cap": 1024,
                      "new_reserve_cap": 24, "normal_remaining": 1024,
                      "reserve_remaining": 24, "hard_remaining": 1048,
                      "hard_cap": 1064}

    def test_full_plan_closes_16_plus_1024_plus_24_at_1064(self):
        result = launcher._validate_cumulative_quota_v2(self.quota, self.history)
        self.assertEqual(result, {"historical_consumed": 16, "normal_remaining": 1024,
                                  "reserve_remaining": 24, "hard_remaining": 1048,
                                  "planned_max_total": 1064})

    def test_legacy_quota_receipt_cannot_substitute_for_v2(self):
        old = dict(self.quota, kind="round58-cumulative-quota-receipt/1",
                   status="reconciled", reserve_cap=40)
        with self.assertRaises(PermissionError):
            launcher._validate_cumulative_quota_v2(old, self.history)

    def test_history_cannot_be_omitted_duplicated_or_relabelled(self):
        for changed in (dict(self.quota, historical_consumed=0),
                        dict(self.quota, historical_consumed=32),
                        dict(self.quota, historical_category_disposition="reclassified_as_reserve")):
            with self.subTest(changed=changed), self.assertRaises(PermissionError):
                launcher._validate_cumulative_quota_v2(changed, self.history)

    def test_normal_reserve_and_hard_caps_are_independently_enforced(self):
        for changed in (dict(self.quota, planned_normal=1023),
                        dict(self.quota, new_normal_cap=1025),
                        dict(self.quota, new_reserve_cap=25),
                        dict(self.quota, hard_cap=1065)):
            with self.subTest(changed=changed), self.assertRaises(PermissionError):
                launcher._validate_cumulative_quota_v2(changed, self.history)

    def test_historical_source_counts_must_close_without_resolving_conflict(self):
        bad = json.loads(json.dumps(self.history))
        bad["environment_consumption"]["gen2_complete_claims"] = 11
        with self.assertRaises(PermissionError):
            launcher._validate_cumulative_quota_v2(self.quota, bad)


class ResumeApprovalSourceTests(unittest.TestCase):
    def setUp(self):
        self.source = json.loads(launcher.RESUME_APPROVAL_SOURCE_PATH.read_text(encoding="utf-8"))

    def test_owner_approval_source_accepts_exact_episode4_continuation_scope(self):
        launcher._validate_resume_approval_source_payload(self.source)

    def test_owner_approval_source_rejects_wrong_thread_or_quote(self):
        for key, value in (("source_thread_id", "unrelated-thread"),
                           ("approval_quote", "核验后开始全新训练")):
            bad = json.loads(json.dumps(self.source))
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(PermissionError):
                launcher._validate_resume_approval_source_payload(bad)

    def test_owner_approval_source_rejects_expanded_scope(self):
        bad = json.loads(json.dumps(self.source))
        bad["approval_scope_as_relayed_by_owner"]["remaining_training_episodes"] = 384
        with self.assertRaises(PermissionError):
            launcher._validate_resume_approval_source_payload(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
