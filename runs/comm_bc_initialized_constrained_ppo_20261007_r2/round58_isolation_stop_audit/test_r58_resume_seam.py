from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

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
        records = seam._batch_records(Path("Z:/absent-round58-resume-root"), batch_dir)
        self.assertEqual(seam.carryover_count_from_batches(records, proof), 3)
        self.assertEqual(len(records), 3)

    def test_direct_event_journal_requires_monotone_actual_call_indexes(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            path = Path(temp) / "optimizer_step_calls.jsonl"
            path.write_text(json.dumps({"call_index": 25, "call_status": "step_invoked"}) + "\n",
                            encoding="utf-8")
            self.assertEqual(seam._read_direct_journal(path, start=25)[0]["call_index"], 25)
            path.write_text(json.dumps({"call_index": 26, "call_status": "step_invoked"}) + "\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "index gap/collision"):
                seam._read_direct_journal(path, start=25)
            path.write_text(json.dumps({"call_index": 25, "call_status": "estimated"}) + "\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "non-invocation"):
                seam._read_direct_journal(path, start=25)


if __name__ == "__main__":
    unittest.main(verbosity=2)
