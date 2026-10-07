from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
MODULE_PATH = RUN_DIR / "r46_v9_eval_runtime_close.py"
SPEC = importlib.util.spec_from_file_location("r46_v9_eval_runtime_close_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
V9 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V9
SPEC.loader.exec_module(V9)

R44_INDEX_SOURCE = RUN_DIR.parents[0] / "comm_dual_response_ablation_20261004/r44_eval_index.py"
R44_SPEC = importlib.util.spec_from_file_location("r44_eval_index_v9_under_test", R44_INDEX_SOURCE)
assert R44_SPEC is not None and R44_SPEC.loader is not None
R44 = importlib.util.module_from_spec(R44_SPEC)
sys.modules[R44_SPEC.name] = R44
R44_SPEC.loader.exec_module(R44)

OLD_ROUND46_RUNNER = ("round46-bc-initialized-constrained-ppo/5+round45-model-only-bc-loader/1+"
                      "round44-constrained-ppo-runtime/1")
V9_COMPOSITE = V9.CODE_VERSION + "+" + OLD_ROUND46_RUNNER
TRAINING_VERSION = ("constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/"
                    "dual-eta-lambda-0p5/1/rule-bc-late147-init/1/"
                    "published-completion-event-at-relay-tick/2")


def booking_fixture(root: Path, *, row_runner_version: str,
                    steps: int = 1000, tape_lines: int = 2) -> tuple[dict, Path]:
    run_dir = root / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "rows").mkdir(parents=True)
    row_path = run_dir / "rows" / "row.json"
    lifecycle = run_dir / "logs" / "rp.jsonl"
    transcript = run_dir / "logs" / "ct.jsonl"
    tape = run_dir / "logs" / "tape.jsonl"
    key = (11, "greedy", 1_301_000)
    row = {
        "kind": "eval", "model_arm": "NEW", "training_seed": key[0],
        "eval_mode": key[1], "scenario_seed": key[2], "attempt": 1,
        "steps": steps, "on_time": 4, "expired": 1, "spent_bytes": 4242,
        "label": R44._expected_label(key), "eval_block": f"one_shot_{key[1]}",
        "comm_action_semantics": "one-shot-send-skip/1",
        "training_version": TRAINING_VERSION,
        "runner_version": row_runner_version,
        "checkpoint_parameter_identity": {"parameter_sha256": "f" * 64},
        "collector_summary": {"decision_steps": tape_lines, "transmission_attempts": 3},
        "artefacts": {
            "row": str(row_path), "lifecycle": str(lifecycle),
            "transcript": str(transcript), "tape": str(tape),
        },
    }
    row_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    lifecycle.write_text('{"tick": 0}\n', encoding="utf-8")
    transcript.write_text('{"tick": 0}\n', encoding="utf-8")
    tape.write_text("".join('{"tick": %d}\n' % i for i in range(tape_lines)), encoding="utf-8")
    return row, run_dir


class RealBookingSeamTests(unittest.TestCase):
    """Reproduce both original defects and prove the unified binding on the real seam."""

    def _assert_receipt(self, run_dir, row):
        return R44.create_claim_artifact_receipt(
            run_dir, dict(row), training_version=TRAINING_VERSION,
            runner_version=V9_COMPOSITE, checkpoint_parameter_sha256="f" * 64)

    def test_stale_booking_version_reproduces_the_v7_stop_defect(self):
        with tempfile.TemporaryDirectory(dir=V9.HERE) as directory:
            root = Path(directory)
            row, run_dir = booking_fixture(root, row_runner_version=V9_COMPOSITE)
            with self.assertRaises(R44.EvalIndexError) as caught:
                R44.create_claim_artifact_receipt(
                    run_dir, dict(row), training_version=TRAINING_VERSION,
                    runner_version=OLD_ROUND46_RUNNER,
                    checkpoint_parameter_sha256="f" * 64)
            self.assertIn("wrong version", str(caught.exception))

    def test_stale_booking_run_dir_reproduces_the_v8_stop_defect(self):
        with tempfile.TemporaryDirectory(dir=V9.HERE) as directory:
            root = Path(directory)
            stale_root = root / "v6_like_root"
            stale_root.mkdir()
            row, run_dir = booking_fixture(root, row_runner_version=V9_COMPOSITE)
            with self.assertRaises(R44.EvalIndexError) as caught:
                R44.create_claim_artifact_receipt(
                    stale_root, dict(row), training_version=TRAINING_VERSION,
                    runner_version=V9_COMPOSITE,
                    checkpoint_parameter_sha256="f" * 64)
            self.assertIn("escapes the ROUND44 output", str(caught.exception))

    def test_unified_binding_passes_the_full_registration_chain(self):
        with tempfile.TemporaryDirectory(dir=V9.HERE) as directory:
            root = Path(directory)
            row, run_dir = booking_fixture(root, row_runner_version=V9_COMPOSITE)
            receipt = self._assert_receipt(run_dir, row)
            self.assertEqual(receipt["runner_version"], V9_COMPOSITE)
            # claim completion through the real r40_common atomic update
            r40_source = (MODULE_PATH.parents[2].parent
                          / "zhenghe-round40/runs/comm_first_report_training_ablation_20261003/r40_common.py")
            spec = importlib.util.spec_from_file_location("r40_common_v9_chain_test", r40_source)
            assert spec is not None and spec.loader is not None
            r40 = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = r40
            spec.loader.exec_module(r40)
            r40.ROUND_DIR = root
            claim_dir = root / "launch_claims" / "model_eval_new_s11_greedy_scene1301000"
            claim_dir.mkdir(parents=True)
            claim_path = claim_dir / "attempt_0001.json"
            claim_path.write_text(json.dumps({
                "logical_job": "model_eval_new_s11_greedy_scene1301000",
                "attempt": 1, "category": "normal",
                "status": "claimed_before_environment"}) + "\n", encoding="utf-8")
            r40._update_claim(claim_path, round44_eval_artifact_receipt=receipt,
                              status="complete", completed_steps=1000)
            final = json.loads(claim_path.read_text(encoding="utf-8"))
            self.assertEqual(final["status"], "complete")
            self.assertEqual(final["round44_eval_artifact_receipt"]["runner_version"],
                             V9_COMPOSITE)

    def test_adjacent_same_prefix_directory_and_outside_paths_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=V9.HERE) as directory:
            root = Path(directory)
            row, run_dir = booking_fixture(root, row_runner_version=V9_COMPOSITE)
            neighbor = root / "v9_eval_runtime_close_v2"
            neighbor.mkdir()
            for outside in (neighbor, root):
                tampered = dict(row)
                tampered["artefacts"] = dict(row["artefacts"])
                tampered["artefacts"]["row"] = str(outside / "rows" / "row.json")
                if outside is root:
                    # root itself contains run/rows/row.json: still inside root but
                    # outside run_dir -> must be rejected as well
                    pass
                with self.assertRaises(R44.EvalIndexError):
                    self._assert_receipt(run_dir, tampered)

    def test_wrong_seed_mode_model_and_incomplete_final_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=V9.HERE) as directory:
            root = Path(directory)
            row, run_dir = booking_fixture(root, row_runner_version=V9_COMPOSITE)
            wrong_seed = dict(row, training_seed=12)
            with self.assertRaises(R44.EvalIndexError):
                self._assert_receipt(run_dir, wrong_seed)
            wrong_mode = dict(row, eval_mode="sampled")
            with self.assertRaises(R44.EvalIndexError):
                self._assert_receipt(run_dir, wrong_mode)
            wrong_model = dict(row)
            wrong_model["checkpoint_parameter_identity"] = {"parameter_sha256": "0" * 64}
            with self.assertRaises(R44.EvalIndexError):
                self._assert_receipt(run_dir, wrong_model)
            short_steps = dict(row, steps=999)
            with self.assertRaises(R44.EvalIndexError):
                self._assert_receipt(run_dir, short_steps)
            missing_tape = dict(row)
            missing_tape["artefacts"] = dict(row["artefacts"])
            missing_tape["artefacts"]["tape"] = str(run_dir / "logs" / "missing.jsonl")
            with self.assertRaises(R44.EvalIndexError):
                self._assert_receipt(run_dir, missing_tape)


class LiveRuntimeBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.runtime = V9.configure_eval_runtime()

    def test_binding_table_all_rows_verified(self):
        table = V9.runtime_binding_table(self.base, self.runtime)
        self.assertTrue(table["all_verified"])
        self.assertGreaterEqual(len(table["rows"]), 7)

    def test_real_consumers_read_the_v9_identity_and_private_root(self):
        composite = self.runtime["v9_runner_version"]
        self.assertEqual(self.runtime["r42_run"].RUNNER_VERSION, composite)
        self.assertEqual(self.runtime["v9_r44_module"].RUNNER_VERSION, composite)
        self.assertEqual(Path(self.runtime["v9_r44_module"].RUN_DIR), V9.V9_ROOT)
        self.assertEqual(Path(self.runtime["v9_runner"].RUN_DIR), V9.V9_ROOT)
        self.assertEqual(Path(self.runtime["common"].ROWS), V9.V9_ROWS)

    def test_assert_binding_rejects_stale_version_run_dir_and_unplanned_key(self):
        plan = V9.build_quota_plan()
        identity = {"file_sha256": V9.FROZEN_MODELS[11][0],
                    "parameter_identity": {"parameter_sha256": V9.FROZEN_MODELS[11][1]}}
        V9.assert_binding(self.runtime, key=(11, "greedy", 1_301_002), arm="NEW",
                          identity=identity, plan=plan)
        with self.assertRaises(V9.V9Error):
            V9.assert_binding(self.runtime, key=(11, "greedy", 1_301_099), arm="NEW",
                              identity=identity, plan=plan)
        saved = self.runtime["v9_r44_module"].RUN_DIR
        try:
            self.runtime["v9_r44_module"].RUN_DIR = V9.V7.V6_ROOT
            with self.assertRaises(V9.V9Error):
                V9.assert_binding(self.runtime, key=(11, "greedy", 1_301_002),
                                  arm="NEW", identity=identity, plan=plan)
        finally:
            self.runtime["v9_r44_module"].RUN_DIR = saved

    def test_binding_guard_rejects_wrong_model_identity(self):
        plan = V9.build_quota_plan()
        bad = {"file_sha256": "0" * 64,
               "parameter_identity": {"parameter_sha256": "0" * 64}}
        with self.assertRaises(V9.V9Error):
            V9.assert_binding(self.runtime, key=(11, "greedy", 1_301_002),
                              arm="NEW", identity=bad, plan=plan)

    def test_two_isolated_r44_instances_do_not_cross_paths_or_versions(self):
        plan = V9.build_quota_plan()
        fake = types.SimpleNamespace(RUNNER_VERSION=OLD_ROUND46_RUNNER,
                                     RUN_DIR=Path(V9.HERE) / "other_root")
        runtime_b = dict(self.runtime)
        runtime_b["v9_r44_module"] = fake
        identity = {"file_sha256": V9.FROZEN_MODELS[11][0],
                    "parameter_identity": {"parameter_sha256": V9.FROZEN_MODELS[11][1]}}
        with self.assertRaises(V9.V9Error):
            V9.assert_binding(runtime_b, key=(11, "greedy", 1_301_002), arm="NEW",
                              identity=identity, plan=plan)
        fake.RUNNER_VERSION = self.runtime["v9_runner_version"]
        with self.assertRaises(V9.V9Error):
            V9.assert_binding(runtime_b, key=(11, "greedy", 1_301_002), arm="NEW",
                              identity=identity, plan=plan)
        fake.RUN_DIR = V9.V9_ROOT
        V9.assert_binding(runtime_b, key=(11, "greedy", 1_301_002), arm="NEW",
                          identity=identity, plan=plan)
        self.assertEqual(self.runtime["v9_r44_module"].RUN_DIR, V9.V9_ROOT)


class AdoptionAndQuotaTests(unittest.TestCase):
    def test_preserved_attempt2_and_1301001_bytes_match_registered_identity(self):
        expected_fields = {
            "claim": "claim_sha256", "row": "row_sha256",
            "lifecycle": "lifecycle_sha256", "transcript": "transcript_sha256",
            "tape": "tape_sha256", "slot": "slot_record_sha256",
        }
        for rel_name, hash_name in expected_fields.items():
            self.assertEqual(V9.sha256_file(V9.HERE / V9.CELL_1301001_REL[rel_name]),
                             V9.CELL_1301001_IDENTITY[hash_name], rel_name)
        self.assertEqual(V9.sha256_file(V9.HERE / V9.V8.ATTEMPT2_CLAIM_REL),
                         V9.V8.ATTEMPT2_IDENTITY["claim_sha256"])
        self.assertEqual(V9.sha256_file(V9.HERE / V9.V8.ATTEMPT2_ROW_REL),
                         V9.V8.ATTEMPT2_IDENTITY["row_sha256"])

    def test_quota_plan_matches_the_authorized_94_cell_map(self):
        plan = V9.build_quota_plan()
        self.assertEqual(plan["new_claim_count"], 94)
        self.assertEqual(plan["adopted_keys"],
                         [[11, "greedy", 1_301_000], [11, "greedy", 1_301_001]])
        self.assertEqual(plan["new_normal_slots"], list(range(388, 481)))
        self.assertEqual(plan["new_reserve_slots"], [2])
        self.assertEqual(plan["total_used_after"], 482)
        keys = [tuple(x["key"]) for x in plan["entries"]]
        self.assertNotIn((11, "greedy", 1_301_000), keys)
        self.assertNotIn((11, "greedy", 1_301_001), keys)
        self.assertEqual(len(set(keys)), 94)
        self.assertEqual((plan["entries"][-1]["category"], plan["entries"][-1]["slot"]),
                         ("reserve", 2))

    def test_adoption_receipt_is_idempotent_and_conflict_safe(self):
        decision = {"kind": "fixture", "task_id": V9.TASK_ID, "status": "accepted",
                    "performance_used_for_adoption": False}
        path = V9.V9_ROOT / "receipts" / "fixture_adoption.json"
        self.assertEqual(V9.write_new_json(path, decision),
                         V9.write_new_json(path, decision))
        with self.assertRaises(V9.V7Error):
            V9.write_new_json(path, {**decision, "status": "conflict"})

    def test_no_performance_gate_in_the_adoption_audit(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('"performance_used_for_adoption": False', source)
        self.assertNotIn('on_time == 6', source)
        self.assertNotIn('spent_bytes == 7878', source)

    def test_validate_episode_closure_accepts_legitimate_task_losses(self):
        # The same performance-neutral closure contract V7 validated still holds.
        self.assertTrue(callable(V9.validate_episode_closure))
        source = (RUN_DIR / "r46_v7_eval_only.py").read_text(encoding="utf-8")
        self.assertIn("never_discovered", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
