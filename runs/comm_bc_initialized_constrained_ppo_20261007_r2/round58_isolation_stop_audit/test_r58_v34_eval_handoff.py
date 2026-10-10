"""Zero-environment negative controls for the eval-only private continuation."""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r58_private_launcher_repaired as launcher


class EvalOnlyBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(launcher.V33_SOURCE_MAP.read_text(encoding="utf-8"))

    def test_archived_dispatcher_missing_scenes_global_is_reproduced(self):
        legacy = launcher.load_legacy_launcher()
        self.assertNotIn("SCENES_NEW", legacy.execute_authorized.__globals__)
        self.assertNotIn("TV53", legacy.execute_authorized.__globals__)
        self.assertNotIn("TV56", legacy.execute_authorized.__globals__)

    def test_completed_sources_close_exact_three_seeds(self):
        result = launcher._qualified_completed_training_sources(self.manifest)
        self.assertEqual(set(result), {11, 22, 33})
        for seed, row in result.items():
            self.assertEqual(row["commit"]["batch"], 31)
            self.assertEqual(row["summary"]["fixed_ep128_report"]["episodes_done"], 128)
            self.assertEqual(row["summary"]["fixed_ep128_report"]["updates"], 32)

    def test_missing_or_tampered_endpoint_rejected(self):
        bad = copy.deepcopy(self.manifest)
        bad["completed_training_sources"]["22"]["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(PermissionError, "hash mismatch"):
            launcher._qualified_completed_training_sources(bad)
        bad = copy.deepcopy(self.manifest)
        del bad["completed_training_sources"]["33"]
        with self.assertRaisesRegex(PermissionError, "exactly three"):
            launcher._qualified_completed_training_sources(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
