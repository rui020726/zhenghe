from __future__ import annotations

import io
import os
import pathlib
import sys
import tempfile
import types
import traceback
import unittest

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r58_private_launcher_repaired as launcher  # noqa: E402
import r58_gate_prior_repaired as adapter  # noqa: E402
import r58_checkpoint_contract_repaired as checkpoint_contract  # noqa: E402
from test_r58_live_adapter_repair import one_rollout  # noqa: E402


class ActualR40CommitSeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current_events = []
        cls.writer_calls = []
        def writer(path, _rows):
            cls.current_events.append(("diagnostic_writer", pathlib.Path(path).name))
            cls.writer_calls.append(pathlib.Path(path).name)
            raise OSError("injected passive diagnostic log failure")
        cls.temp = tempfile.TemporaryDirectory(dir=launcher.GEN2)
        cls.raw = pathlib.Path(cls.temp.name) / "private_raw"
        cls.binding = launcher.prepare_actual_runtime_for_audit(cls.raw, writer=writer)

    @classmethod
    def tearDownClass(cls):
        launcher.close_audit_runtime(cls.binding)
        cls.temp.cleanup()

    def _run_actual_train_seed(self, *, diag_failure=None, save_failure=False,
                               receipt_failure=False):
        events = []
        type(self).current_events = events
        ppo_reports = []
        step_calls = []
        provider_calls = []
        real_torch_saves = []
        actual_torch_save = torch.save
        original_capture = adapter.capture_actor_gradient_diagnostic
        original_adapter_hashlib = adapter.hashlib
        original_rename = __import__("os").rename
        original_replace = os.replace
        if diag_failure == "gradient":
            adapter.capture_actor_gradient_diagnostic = lambda *_a, **_k: (_ for _ in ()).throw(
                ArithmeticError("injected diagnostic gradient failure"))
        elif diag_failure == "hash":
            adapter.hashlib = types.SimpleNamespace(
                sha256=lambda *_a, **_k: (_ for _ in ()).throw(
                    OSError("injected diagnostic hash failure")))
        binding = type(self).binding
        raw = type(self).raw
        runtime = binding["runtime"]
        trainer = runtime["trainer"]
        r40 = runtime["r40_train"]
        common = runtime["common"]
        r42run = runtime["r42_run"]
        controller_module = sys.modules["v2_controller"]
        r40run = sys.modules["r40_run"]

        original_ppo = trainer.ppo_update
        def capture_ppo(*args, **kwargs):
            report = original_ppo(*args, **kwargs)
            ppo_reports.append(report)
            events.append(("ppo_return", report.get("optimizer_steps")))
            return report
        trainer.ppo_update = capture_ppo

        original_latest = r40._latest_resume_source
        original_pre = r40._save_preupdate_checkpoint
        original_immutable = r40._write_immutable_snapshot
        original_json = r40._write_json_exclusive
        original_hash = r40._sha256
        original_progress = r40._save_progress
        original_provider = r40run.run_training_episode
        original_parameter_identity = controller_module.parameter_identity
        original_adam = torch.optim.Adam
        original_milestone = r40._save_milestone
        original_path_open = pathlib.Path.open
        original_fsync = os.fsync
        original_runner_sha = r42run.sha256
        original_sidecar_sha = checkpoint_contract._file_sha256

        r40._latest_resume_source = lambda *_a, **_k: None
        r40._save_milestone = lambda _c, _o, _s, _seed, episodes, _freeze: (
            events.append(("milestone_spy", episodes)) or pathlib.Path(common.CHECKPOINTS) / "spy_milestone.pt")
        r40._sha256 = lambda _path: "spy-sha"
        def json_writer(path, value):
            name = pathlib.Path(path).name
            is_receipt = name.startswith("r40_batch_commit_")
            events.append(("receipt_attempt" if is_receipt else "json_record", name))
            if is_receipt and receipt_failure:
                raise OSError("injected receipt persistence failure")
            if is_receipt:
                events.append(("receipt_spy_return", name))
        r40._write_json_exclusive = json_writer
        r40._save_progress = lambda *_a, **_k: events.append(("progress",))

        order = common.scenario_order()
        def episode_provider(_common, _runner, controller, *, training_seed,
                             episode_index, scenario_seed, lam):
            provider_calls.append((training_seed, episode_index, scenario_seed))
            episode = one_rollout(controller, count=1)[0]
            episode.update({
                "episode_index": episode_index, "scenario_seed": scenario_seed,
                "lambda": lam, "on_time": 2, "spent_bytes": 100,
                "emissions": 1, "attempt": 1,
                "artefacts": {"row": "memory-row", "tape": "memory-tape"},
            })
            return episode
        r40run.run_training_episode = episode_provider

        identity_counter = {"n": 0}
        def fake_parameter_identity(_controller):
            identity_counter["n"] += 1
            return {"parameter_sha256": f"{identity_counter['n']:064x}",
                    "parameter_count": 5700}
        controller_module.parameter_identity = fake_parameter_identity

        def no_update_adam(parameters, lr):
            optimizer = original_adam(parameters, lr=lr)
            def spy_step(*_a, **_k):
                step_calls.append("spy-no-update")
                events.append(("optimizer_spy",))
                return None
            optimizer.step = spy_step
            return optimizer

        def torch_save_spy(payload, path, *args, **kwargs):
            name = getattr(path, "audit_name", pathlib.Path(path).name
                           if isinstance(path, (str, bytes, os.PathLike)) else "memory-file")
            events.append(("core_save", name))
            real_torch_saves.append({"name": name,
                                     "training_version": payload.get("training_version"),
                                     "top_attestation": payload.get("round58_repair_attestation"),
                                     "state_attestation": payload.get("training_state", {}).get(
                                         "round58_repair_attestation"),
                                     "freeze_attestation": payload.get("freeze", {}).get(
                                         "round58_repair_attestation")})
            return None

        torch.optim.Adam = no_update_adam
        torch.save = torch_save_spy
        class MemoryWriter(io.BytesIO):
            def __init__(self, name):
                super().__init__()
                self.audit_name = name
            def fileno(self):
                return 0
        class MemoryTextWriter(io.StringIO):
            def fileno(self):
                return 0
        def path_open_spy(path, mode="r", *args, **kwargs):
            resolved = pathlib.Path(path).resolve()
            if str(resolved).startswith(str(raw.resolve()) + os.sep) and any(
                    flag in mode for flag in ("w", "a", "x", "+")):
                name = pathlib.Path(path).name
                events.append(("memory_open", name, mode))
                if "b" not in mode:
                    return MemoryTextWriter()
                return MemoryWriter(name)
            return original_path_open(path, mode, *args, **kwargs)
        pathlib.Path.open = path_open_spy
        os.fsync = lambda _fd: events.append(("fsync_spy",))
        r42run.sha256 = lambda _path: "journal-spy-sha"
        checkpoint_contract._file_sha256 = lambda _path: "memory-sidecar-sha"
        os.rename = lambda src, dst: events.append(
            ("snapshot_rename_spy", pathlib.Path(dst).name))
        def replace_spy(src, dst):
            name = pathlib.Path(dst).name
            events.append(("core_replace_spy", name))
            if save_failure and name == "v2_train_s11.pt":
                raise OSError("injected core checkpoint failure")
        os.replace = replace_spy

        try:
            references = {int(scene): {"on_time": 2} for scene in order[:4]}
            try:
                result = r40.train_seed(11, episodes=4, freeze=runtime["round42_freeze"],
                                        reference_cache=references)
                outcome = {"result": result, "exception": None}
            except Exception as exc:
                    outcome = {"result": None, "exception": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}
            self.assertFalse(any(raw.rglob("*.pt")), "spy must not persist checkpoint bytes")
            self.assertFalse(any(raw.rglob("r40_batch_commit*.json")))
            self.assertFalse((raw / "runtime" / "checkpoint_writes.jsonl").exists())
            return {"events": events, "ppo_reports": ppo_reports,
                    "step_calls": len(step_calls), "save_calls": real_torch_saves,
                    "provider_calls": len(provider_calls),
                    "outcome": outcome}
        finally:
            trainer.ppo_update = original_ppo
            r40._latest_resume_source = original_latest
            r40._save_preupdate_checkpoint = original_pre
            r40._save_milestone = original_milestone
            r40._write_immutable_snapshot = original_immutable
            r40._write_json_exclusive = original_json
            r40._sha256 = original_hash
            r40._save_progress = original_progress
            r40run.run_training_episode = original_provider
            controller_module.parameter_identity = original_parameter_identity
            torch.optim.Adam = original_adam
            torch.save = actual_torch_save
            os.rename = original_rename
            os.replace = original_replace
            pathlib.Path.open = original_path_open
            os.fsync = original_fsync
            r42run.sha256 = original_runner_sha
            checkpoint_contract._file_sha256 = original_sidecar_sha
            adapter.capture_actor_gradient_diagnostic = original_capture
            adapter.hashlib = original_adapter_hashlib

    def test_diagnostic_gradient_hash_and_log_failures_do_not_break_real_commit_call_order(self):
        for failure in ("gradient", "hash", "log"):
            with self.subTest(failure=failure):
                result = self._run_actual_train_seed(diag_failure=failure)
                self.assertIsNone(result["outcome"]["exception"])
                self.assertEqual(result["outcome"]["result"]["episodes_done"], 4)
                self.assertEqual(result["step_calls"], 4)  # spy callbacks only; no Adam mutation
                self.assertEqual(result["provider_calls"], 4)  # stubbed provider; no environment builder
                self.assertTrue(result["ppo_reports"])
                report = result["ppo_reports"][0]
                self.assertTrue(report.get("gate_diag", {}).get("diagnostics_error")
                                or report.get("passive_diagnostics_errors"))
                rolling_save = result["events"].index(("core_replace_spy", "v2_train_s11.pt"))
                immutable_post = next(i for i, event in enumerate(result["events"])
                                      if event[0] == "core_replace_spy"
                                      and "after_b0" in event[1])
                receipt = next(i for i, event in enumerate(result["events"])
                               if event[0] == "receipt_attempt"
                               and event[1].startswith("r40_batch_commit_"))
                self.assertLess(immutable_post, rolling_save)
                self.assertLess(rolling_save, receipt)
                self.assertTrue(any(event[0] == "receipt_spy_return"
                                    for event in result["events"]))
                identity = type(self).binding["runtime"]["r58_checkpoint_contract"]["identity"]
                self.assertTrue(result["save_calls"])
                self.assertTrue(all(item["training_version"] == launcher.REPAIRED_TRAINING_VERSION
                                    and item["top_attestation"] == identity
                                    and item["state_attestation"] == identity
                                    and item["freeze_attestation"] == identity
                                    for item in result["save_calls"]))

    def test_core_checkpoint_failure_propagates_and_no_commit_receipt_is_attempted(self):
        result = self._run_actual_train_seed(save_failure=True)
        self.assertIn("injected core checkpoint failure", result["outcome"]["exception"])
        self.assertEqual(result["step_calls"], 4)
        self.assertEqual(result["provider_calls"], 4)
        self.assertFalse(any(event[0] == "receipt_attempt" for event in result["events"]))

    def test_receipt_failure_propagates_without_success_receipt(self):
        result = self._run_actual_train_seed(receipt_failure=True)
        self.assertIn("injected receipt persistence failure", result["outcome"]["exception"])
        self.assertEqual(result["step_calls"], 4)
        self.assertEqual(result["provider_calls"], 4)
        receipts = [event for event in result["events"] if event[0] == "receipt_attempt"]
        self.assertEqual(len(receipts), 1)  # one failed write attempt; no successful receipt bytes
        self.assertFalse(any(event[0] == "receipt_spy_return" for event in result["events"]))
        self.assertFalse(any(event[0] == "progress" for event in result["events"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
