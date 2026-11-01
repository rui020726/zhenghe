"""Pure-data tests for the recovery_07 terminal evidence adapter."""
from __future__ import annotations

import copy
import unittest
from pathlib import Path

from stage5b_comm import recovery07_terminal as A
from stage5b_comm import recovery_descendant as D
from stage5b_comm import recovery_runtime as RT

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path(r"D:\code\整合\zhenghe-round42\runs\stage5b_pure_comm_convergence_20261027\recovery_07\revision_01")
RUN_DIR = PARENT / "execution" / "recovery_train_run_01"
MONITOR = RUN_DIR / "monitor_terminal_receipt.json"
OUTER = RUN_DIR / "outer_poll_208.json"
RECEIPT = PARENT / "execution" / "receipts" / "recovery-train_terminal_recovery-train-43720-1793267363280466900.json"
R8_PARENT = Path(r"D:\code\整合\zhenghe-round42\runs\stage5b_pure_comm_convergence_20261027\recovery_08\revision_01")
R8_RUN_DIR = R8_PARENT / "execution" / "recovery_train_run_01"
R8_MONITOR = R8_RUN_DIR / "monitor_terminal_receipt.json"


class OverrideReader(D.Reads):
    def __init__(self, overrides):
        super().__init__()
        self.overrides = {Path(path).resolve(): mutate for path, mutate in overrides.items()}

    def json(self, path, pin=None, root=None):
        body = super().json(path, pin, root)
        resolved = Path(path).resolve()
        mutate = self.overrides.get(resolved)
        if mutate is None:
            return body
        changed = copy.deepcopy(body)
        mutate(changed)
        return changed


class RawOverrideReader(D.Reads):
    def __init__(self, overrides):
        super().__init__()
        self.overrides = {Path(path).resolve(): replacement
                          for path, replacement in overrides.items()}

    def raw(self, path, pin=None, root=None):
        actual = super().raw(path, pin, root)
        return self.overrides.get(Path(path).resolve(), actual)


class Recovery07TerminalAdapterTests(unittest.TestCase):
    def test_real_r7_terminal_structure_normalizes(self):
        summary, outer_name, monitor_path = A.normalize(PARENT, RUN_DIR, D.Reads())
        self.assertEqual(outer_name, "outer_poll_208.json")
        self.assertEqual(monitor_path, MONITOR)
        self.assertEqual(summary["claims"], 720)
        self.assertEqual(summary["results"], 720)
        self.assertEqual(summary["update_journals"], 180)
        self.assertEqual(len(summary["checkpoints"]), 6)
        self.assertEqual(len(summary["certificates"]), 6)
        self.assertTrue(all(item["run"].startswith(("T1_s", "T2_s"))
                            for item in summary["checkpoints"]))
        self.assertTrue(all(item["run"].startswith(("T1_s", "T2_s"))
                            for item in summary["certificates"]))
        self.assertEqual(summary["adapter"], A.ADAPTER)
        self.assertEqual(summary["child_exit"]["value"], "0")
        self.assertEqual(summary["outer_terminal"]["exit_code"], 0)
        self.assertEqual(summary["outer_terminal"]["output"], "")

    def test_real_r8_monitor_wrapper_normalizes_from_saved_terminal_files(self):
        summary, outer_name, monitor_path = A.normalize(R8_PARENT, R8_RUN_DIR, D.Reads())
        self.assertEqual(outer_name, "outer_poll_terminal.json")
        self.assertEqual(monitor_path, R8_MONITOR)
        self.assertEqual(summary["adapter"], A.WRAPPER_ADAPTER)
        self.assertEqual(summary["claims"], 720)
        self.assertEqual(summary["results"], 720)
        self.assertEqual(summary["update_journals"], 180)
        self.assertEqual(len(summary["checkpoints"]), 6)
        self.assertEqual(len(summary["certificates"]), 6)
        self.assertEqual(summary["child_exit"]["value"], "0")
        self.assertEqual(summary["outer_terminal"]["exit_code"], 0)
        self.assertIn("CHILD_EXIT=0", summary["outer_terminal"]["output"])

    def test_r8_nonzero_child_text_is_rejected(self):
        reader = RawOverrideReader({R8_RUN_DIR / "child_exit.txt": b"1\n"})
        with self.assertRaisesRegex(RT.RecoveryRuntimeError, "child exit is not zero"):
            A.normalize(R8_PARENT, R8_RUN_DIR, reader)

    def test_missing_outer_evidence_is_rejected(self):
        def remove_outer(body):
            body["evidence_files"] = [item for item in body["evidence_files"]
                                      if Path(item["path"]).name != "outer_poll_208.json"]
        reader = OverrideReader({MONITOR: remove_outer})
        with self.assertRaisesRegex(RT.RecoveryRuntimeError, "evidence missing outer_poll_208"):
            A.normalize(PARENT, RUN_DIR, reader)

    def test_missing_child_evidence_is_rejected(self):
        def remove_child(body):
            body["evidence_files"] = [item for item in body["evidence_files"]
                                      if Path(item["path"]).name != "child.exit.txt"]
        reader = OverrideReader({MONITOR: remove_child})
        with self.assertRaisesRegex(RT.RecoveryRuntimeError, "evidence missing child.exit"):
            A.normalize(PARENT, RUN_DIR, reader)

    def test_nonzero_child_exit_with_empty_outer_output_is_rejected(self):
        reader = RawOverrideReader({RUN_DIR / "child.exit.txt": b"1\n"})
        with self.assertRaisesRegex(RT.RecoveryRuntimeError, "child exit is not zero"):
            A.normalize(PARENT, RUN_DIR, reader)

    def test_nonzero_outer_exit_is_rejected(self):
        def nonzero(body):
            body["exit_code"] = 1
        reader = OverrideReader({OUTER: nonzero})
        with self.assertRaisesRegex(RT.RecoveryRuntimeError, "clean exit zero"):
            A.normalize(PARENT, RUN_DIR, reader)

    def test_nonzero_final_receipt_is_rejected(self):
        def nonzero(body):
            body["returncode"] = 1
        reader = OverrideReader({RECEIPT: nonzero})
        with self.assertRaisesRegex(RT.RecoveryRuntimeError, "not successful training"):
            A.normalize(PARENT, RUN_DIR, reader)


class MinimumTargetCompatibilityTests(unittest.TestCase):
    def test_stage6_recalculation_excludes_stage7_minimum_fields(self):
        roster = (("T1", 91001), ("T1", 91002), ("T1", 91003),
                  ("T2", 91001), ("T2", 91002), ("T2", 91003))
        runs = [{"arm": arm, "training_seed": seed} for arm, seed in roster]
        states = {key: {"cursor": 8, "next_update": 2, "certified": False,
                        "donor": {"synthetic": True},
                        "completed_episode_count": 8,
                        "registered_missing_episode_count": 0}
                  for key in roster}
        old_shape = D.next_runs(runs, states, include_completion_counts=True,
                                include_minimum_target=False)
        self.assertTrue(all("minimum_episode_target" not in item for item in old_shape))
        self.assertTrue(all("episodes_below_minimum_5000_after_stage" not in item
                            for item in old_shape))
        new_shape = D.next_runs(runs, states, include_completion_counts=True,
                                include_minimum_target=True)
        self.assertTrue(all(item["minimum_episode_target"] == 5000 for item in new_shape))
        self.assertTrue(all(item["projected_completed_episode_count_after_stage"] == 128
                            for item in new_shape))
        self.assertTrue(all(item["episodes_below_minimum_5000_after_stage"] == 4872
                            for item in new_shape))


if __name__ == "__main__":
    unittest.main()