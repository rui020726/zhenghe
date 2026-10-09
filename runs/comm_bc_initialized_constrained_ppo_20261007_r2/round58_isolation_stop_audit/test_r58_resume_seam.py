from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r58_resume_seam_tested", HERE / "r58_resume_seam.py")
assert spec and spec.loader
seam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seam)
REPO = HERE.parents[2]


class ResumeSourceContractTests(unittest.TestCase):
    def test_only_exact_three_qualified_sources_are_mapped(self):
        proof = seam.validate_resume_sources(REPO)
        self.assertEqual(sorted(proof["sources"]), [11, 22, 33])
        self.assertEqual(proof["historical_ppo_carryover"], 3)
        self.assertEqual(proof["historical_optimizer_steps"], 24)
        self.assertEqual(proof["expected_cursor"] if "expected_cursor" in proof else
                         proof["manifest"]["expected_cursor"],
                         {"episodes_done": 4, "updates": 1,
                          "optimizer_steps": 8, "lambda": 1.0})
        for seed, item in proof["sources"].items():
            self.assertEqual(item["qualification"], seam.QUALIFIED)
            self.assertTrue(item["path"].is_file())
            self.assertTrue(item["sidecar_path"].is_file())
            self.assertNotEqual(item["path"].parent, proof["target_checkpoint_root"])

    def test_mutated_source_map_identity_is_rejected(self):
        with self.assertRaisesRegex(PermissionError, "map hash"):
            seam.validate_resume_sources(REPO, expected_map_sha256="0" * 64)

    def test_carryover_requires_exact_durable_batch_records_without_copying(self):
        proof = seam.validate_resume_sources(REPO)
        batch_dir = proof["sources"][11]["path"].parent
        prior = proof["committed_continuation"]["prior_resume_raw"] / "checkpoints"
        records = seam._batch_records(Path("Z:/absent-round58-resume-root"), batch_dir, prior)
        self.assertEqual(seam.carryover_count_from_batches(records, proof), 12)
        self.assertEqual(len(records), 12)
        with self.assertRaisesRegex(RuntimeError, "committed continuation batch missing"):
            seam.carryover_count_from_batches(records[:-1], proof)

    def test_direct_event_journal_requires_monotone_actual_call_indexes(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            path = Path(temp) / "optimizer_step_calls.jsonl"
            path.write_text(json.dumps({"call_index": 77, "call_status": "step_invoked"}) + "\n",
                            encoding="utf-8")
            self.assertEqual(seam._read_direct_journal(path, start=77)[0]["call_index"], 77)
            path.write_text(json.dumps({"call_index": 78, "call_status": "step_invoked"}) + "\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "index gap/collision"):
                seam._read_direct_journal(path, start=77)
            path.write_text(json.dumps({"call_index": 77, "call_status": "estimated"}) + "\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "non-invocation"):
                seam._read_direct_journal(path, start=77)

    def _install_call_journal(self, temp):
        import torch
        trainer = SimpleNamespace(ppo_update=lambda: None)
        run = SimpleNamespace(_OPTIMIZER_STEP_CALLS=0, _PPO_UPDATE_CALLS=0)
        version = SimpleNamespace(CHECKPOINT_WRITE_CALLS=41)
        runtime = {
            "trainer": trainer, "r42_run": run, "r42_versions": version,
            "r58_resume_source_proof": {"historical_checkpoint_journal": {
                "sha256": seam.HISTORICAL_CHECKPOINT_JOURNAL_SHA256, "rows": 12},
                "committed_continuation": {"prior_direct_journals": {
                    "checkpoint_writes.jsonl": {"rows": 29}}}},
        }
        before = torch.optim.Adam.step
        run._RUNTIME = {"restore_adam_step": before}
        receipt = seam.install_persistent_call_journals(
            runtime, resume_runtime_dir=Path(temp) / "runtime")
        receipt["_test_run"] = run
        return torch, trainer, receipt, before

    def test_call_journal_teardown_accepts_exact_active_wrappers(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            torch, trainer, receipt, before = self._install_call_journal(temp)
            try:
                self.assertIs(torch.optim.Adam.step, receipt["wrapper_step"])
                self.assertIs(trainer.ppo_update, receipt["wrapper_ppo"])
                receipt["undo"]()
                self.assertIs(torch.optim.Adam.step, before)
            finally:
                torch.optim.Adam.step = before

    def test_call_journal_teardown_accepts_exact_already_unwrapped_hooks(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            torch, trainer, receipt, before = self._install_call_journal(temp)
            try:
                torch.optim.Adam.step = receipt["wrapped_step"]
                trainer.ppo_update = receipt["wrapped_ppo"]
                receipt["undo"]()
                self.assertIs(torch.optim.Adam.step, receipt["wrapped_step"])
                self.assertIs(trainer.ppo_update, receipt["wrapped_ppo"])
            finally:
                torch.optim.Adam.step = before

    def test_call_journal_teardown_accepts_frozen_r42_adam_cleanup_only(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            torch, trainer, receipt, before = self._install_call_journal(temp)
            try:
                # install_persistent_call_journals captures the exact active R42
                # runtime object; emulate its documented final Adam-hook restore
                # while the outer persistent PPO journal is still installed.
                # The helper's closure retains that runtime via r42_run.
                run = receipt.get("_test_run")
                if run is None:
                    self.skipTest("test fixture does not expose R42 runtime")
                torch.optim.Adam.step = before
                trainer.ppo_update = receipt["wrapper_ppo"]
                receipt["undo"]()
                self.assertIs(torch.optim.Adam.step, before)
                self.assertIs(trainer.ppo_update, receipt["wrapped_ppo"])
            finally:
                torch.optim.Adam.step = before

    def test_call_journal_teardown_accepts_only_exact_pytorch_wrapped_step_chain(self):
        from functools import wraps
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            torch, trainer, receipt, before = self._install_call_journal(temp)
            try:
                target = receipt["wrapper_step"]
                @wraps(target)
                def torch_style_wrapper(*args, **kwargs):
                    return target(*args, **kwargs)
                torch.optim.Adam.step = torch_style_wrapper
                trainer.ppo_update = receipt["wrapper_ppo"]
                receipt["undo"]()
                self.assertIs(torch.optim.Adam.step, before)
                self.assertIs(trainer.ppo_update, receipt["wrapped_ppo"])
            finally:
                torch.optim.Adam.step = before

    def test_call_journal_teardown_still_rejects_unrecognized_hook(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            torch, trainer, receipt, before = self._install_call_journal(temp)
            try:
                torch.optim.Adam.step = lambda *args, **kwargs: None
                with self.assertRaisesRegex(RuntimeError, "adam_expected_wrapper=False"):
                    receipt["undo"]()
            finally:
                torch.optim.Adam.step = before


if __name__ == "__main__":
    unittest.main(verbosity=2)
