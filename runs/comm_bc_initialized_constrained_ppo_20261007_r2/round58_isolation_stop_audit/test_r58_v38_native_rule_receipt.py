"""Zero-environment counterexamples for native-rule artifact receipt separation."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r58_v35_eval_booking as booking

REPO = HERE.parents[2]
BASE = REPO / "runs/comm_bc_initialized_constrained_ppo_20261007_r2"
ART = BASE / "round58_gen3_resume_v35/raw"
CLAIMS = BASE / "round58_gen3/raw/launch_claims"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class NativeRuleReceipt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = read(ART / "rows/v2_r58_rule_immediate_s1700000_s1700000_a1.json")
        cls.claim = read(CLAIMS / "rule_immediate_scene1700000/attempt_0001.json")
        cls.r45 = SimpleNamespace(RUN_DIR=ART, _load_jsonl=read_jsonl,
                                  _assert_empty_transcript_zero_closure=lambda row: None)

    def test_original_rule_has_only_three_real_artifacts(self):
        assert not Path(self.row["artefacts"]["tape"]).exists()
        receipt = booking._rule_receipt(self.r45, self.row, self.claim)
        self.assertEqual(set(receipt["artifacts"]), {"row", "lifecycle", "transcript"})
        self.assertEqual(receipt["actor_tape"], "structurally_absent_for_native_rule")
        self.assertEqual(receipt["row_identity"]["spent_bytes"], 15486)

    def test_rule_identity_and_artifacts_are_not_softened(self):
        for mutate in (
            lambda row, claim: row.__setitem__("scenario_seed", 1700001),
            lambda row, claim: row.__setitem__("model_arm", "NEW"),
            lambda row, claim: row.__setitem__("training_version", "wrong"),
            lambda row, claim: claim["round45_artifact_paths"].__setitem__("row", "outside"),
        ):
            row, claim = copy.deepcopy(self.row), copy.deepcopy(self.claim)
            mutate(row, claim)
            with self.assertRaises(RuntimeError):
                booking._rule_receipt(self.r45, row, claim)

    def test_model_receipt_remains_four_artifact_branch(self):
        src = (HERE / "r58_v35_eval_booking.py").read_text(encoding="utf-8")
        self.assertIn("r45._verify_receipt_artifacts(claim, row)", src)
        self.assertIn("r45._record_artifact_receipt(kind, row", src)
        model = read(ART / "rows/v2_r44_eval_new_s11_greedy_s1700000_a2.json")
        self.assertTrue(Path(model["artefacts"]["tape"]).is_file())
        self.assertEqual(set(model["artefacts"]), {"row", "lifecycle", "transcript", "tape"})

    def test_only_byte_authenticated_failed_rule_is_retryable(self):
        plan = {"jobs": [{"logical_job": booking.FAILED_RULE_KEY,
                          "scope": "rule_eval_immediate", "scene": 1700000,
                          "normal_slot": 961}]}
        with tempfile.TemporaryDirectory(prefix="r58_rule_receipt_", dir=HERE) as folder:
            root = Path(folder)
            claim_path = root / "launch_claims" / booking.FAILED_RULE_KEY / "attempt_0001.json"
            slot_path = root / "launch_slots/normal/0961.json"
            claim_path.parent.mkdir(parents=True)
            slot_path.parent.mkdir(parents=True)
            original_claim = CLAIMS / booking.FAILED_RULE_KEY / "attempt_0001.json"
            original_slot = BASE / "round58_gen3/raw/launch_slots/normal/0961.json"
            claim_path.write_bytes(original_claim.read_bytes())
            slot_path.write_bytes(original_slot.read_bytes())
            self.assertEqual(booking._verify_failed_first_rule(root, plan)["retry_status"], "pending")
            altered = read(claim_path)
            altered["launch_slot"] = 962
            claim_path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                booking._verify_failed_first_rule(root, plan)


if __name__ == "__main__":
    unittest.main()
