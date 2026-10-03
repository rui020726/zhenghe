"""Targeted zero-environment regressions for ROUND42 path and resume identity."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from r42_analysis import _validate_frozen_checkpoint_path_map


class FrozenCheckpointPathIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "input"
        self.file = self.root / "runs" / "model" / "frozen.pth"
        self.file.parent.mkdir(parents=True)
        self.file.write_bytes(b"registered checkpoint bytes")
        self.digest = hashlib.sha256(self.file.read_bytes()).hexdigest()
        self.frozen = {
            "piloc": {"path": str(self.file), "sha256": self.digest},
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def validate(self, record: dict[str, str], frozen=None):
        return _validate_frozen_checkpoint_path_map(
            record, frozen or self.frozen, frozen_root=self.root)

    def test_slash_and_backslash_spellings_are_equivalent(self) -> None:
        posix_spelling = "runs/model/frozen.pth"
        windows_spelling = r"runs\model\frozen.pth"
        # This reproduces why the old raw set assertion failed on Windows.
        self.assertNotEqual({windows_spelling}, {posix_spelling})
        self.assertEqual(self.validate({windows_spelling: self.digest})[0][
            "actual_sha256"], self.digest)
        self.assertEqual(self.validate({posix_spelling: self.digest})[0][
            "actual_sha256"], self.digest)

    def test_real_seed11_batch_record_and_freeze_match_by_full_identity(self) -> None:
        run_dir = Path(__file__).resolve().parent
        repo = run_dir.parents[1]
        frozen = json.loads((run_dir / "ROUND42_FROZEN_CONFIG.json").read_text(
            encoding="utf-8"))
        record = json.loads((run_dir / "checkpoints" /
                             "r40_batch_s11_b0_attempt01.json").read_text(encoding="utf-8"))
        input_root = Path(frozen["execution_identity"]["canonical_read_only_input_root"])
        old_expected = {
            Path(item["path"]).relative_to(input_root).as_posix()
            for item in frozen["round42"]["frozen_search_allocation_checkpoints"].values()
        }
        self.assertNotEqual(set(record["frozen_checkpoints"]), old_expected)
        verified = _validate_frozen_checkpoint_path_map(
            record["frozen_checkpoints"],
            frozen["round42"]["frozen_search_allocation_checkpoints"],
            frozen_root=input_root,
        )
        self.assertEqual(len(verified), 2)
        self.assertEqual({item["actual_sha256"] for item in verified},
                         {item["sha256"] for item in
                          frozen["round42"]["frozen_search_allocation_checkpoints"].values()})
        self.assertTrue((repo / "runs" / "comm_one_shot_report_selection_20261003_r"
                         / "ROUND42_STOP_STATE.json").is_file())

    def test_absolute_forward_slash_spelling_stays_below_declared_root(self) -> None:
        absolute = str(self.file).replace("\\", "/")
        self.assertEqual(self.validate({absolute: self.digest})[0]["actual_sha256"],
                         self.digest)

    def test_same_basename_in_a_different_directory_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "path set changed"):
            self.validate({"runs/other/frozen.pth": self.digest})

    def test_record_hash_mismatch_is_rejected(self) -> None:
        wrong = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "SHA256 differs"):
            self.validate({"runs/model/frozen.pth": wrong})

    def test_missing_frozen_file_is_rejected(self) -> None:
        missing = self.root / "runs" / "model" / "missing.pth"
        frozen = {"piloc": {"path": str(missing), "sha256": self.digest}}
        with self.assertRaisesRegex(RuntimeError, "missing or outside"):
            self.validate({"runs/model/missing.pth": self.digest}, frozen)

    def test_record_keys_that_normalize_to_a_collision_are_rejected(self) -> None:
        record = {
            "runs/model/frozen.pth": self.digest,
            r"runs\model\frozen.pth": self.digest,
        }
        with self.assertRaisesRegex(RuntimeError, "keys collide after normalization"):
            self.validate(record)

    def test_frozen_keys_that_normalize_to_a_collision_are_rejected(self) -> None:
        frozen = {
            "piloc_forward": {"path": str(self.file), "sha256": self.digest},
            "piloc_backslash": {
                "path": str(self.file).replace("/", "\\"), "sha256": self.digest,
            },
        }
        with self.assertRaisesRegex(RuntimeError, "frozen checkpoint keys collide"):
            self.validate({"runs/model/frozen.pth": self.digest}, frozen)

    def test_other_drive_or_absolute_path_outside_root_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "outside its declared root|wrong drive"):
            self.validate({"Z:/runs/model/frozen.pth": self.digest})


if __name__ == "__main__":
    unittest.main()
