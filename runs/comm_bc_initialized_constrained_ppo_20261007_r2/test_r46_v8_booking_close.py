from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
MODULE_PATH = RUN_DIR / "r46_v8_eval_booking_close.py"
SPEC = importlib.util.spec_from_file_location("r46_v8_eval_booking_close_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
V8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V8)

R44_INDEX_SOURCE = RUN_DIR.parents[1] / "runs/comm_dual_response_ablation_20261004/r44_eval_index.py"
R44_SPEC = importlib.util.spec_from_file_location("r44_eval_index_under_test", R44_INDEX_SOURCE)
assert R44_SPEC is not None and R44_SPEC.loader is not None
R44 = importlib.util.module_from_spec(R44_SPEC)
import sys as _sys
_sys.modules[R44_SPEC.name] = R44
R44_SPEC.loader.exec_module(R44)

OLD_ROUND46_RUNNER = ("round46-bc-initialized-constrained-ppo/5+round45-model-only-bc-loader/1+"
                      "round44-constrained-ppo-runtime/1")
V8_COMPOSITE = V8.CODE_VERSION + "+" + OLD_ROUND46_RUNNER


def booking_fixture(root: Path, *, row_runner_version: str) -> tuple[dict, Path]:
    """A minimal but real booking fixture that passes every non-version check."""
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
        "steps": 1000, "on_time": 4, "expired": 1, "spent_bytes": 4242,
        "label": R44._expected_label(key), "eval_block": f"one_shot_{key[1]}",
        "comm_action_semantics": "one-shot-send-skip/1",
        "training_version": "frozen-training-version",
        "runner_version": row_runner_version,
        "checkpoint_parameter_identity": {"parameter_sha256": "f" * 64},
        "collector_summary": {"decision_steps": 2, "transmission_attempts": 3},
        "artefacts": {
            "row": str(row_path), "lifecycle": str(lifecycle),
            "transcript": str(transcript), "tape": str(tape),
        },
    }
    row_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    lifecycle.write_text('{"tick": 0}\n', encoding="utf-8")
    transcript.write_text('{"tick": 0}\n', encoding="utf-8")
    tape.write_text('{"tick": 0}\n{"tick": 1}\n', encoding="utf-8")
    return row, run_dir


class RealBookingSeamTests(unittest.TestCase):
    """The version binding must be proven through the real r44_eval_index entry."""

    def test_original_binding_defect_reproduces_then_unified_binding_passes(self):
        with tempfile.TemporaryDirectory(dir=V8.HERE) as directory:
            root = Path(directory)
            row, run_dir = booking_fixture(root, row_runner_version=V8_COMPOSITE)
            stale = OLD_ROUND46_RUNNER
            with self.assertRaises(R44.EvalIndexError) as caught:
                R44.create_claim_artifact_receipt(
                    run_dir, dict(row), training_version="frozen-training-version",
                    runner_version=stale,
                    checkpoint_parameter_sha256="f" * 64)
            self.assertIn("wrong version", str(caught.exception))
            # The identical science row passes once the booking module global is
            # bound to the same execution identity as row generation.
            receipt = R44.create_claim_artifact_receipt(
                run_dir, dict(row), training_version="frozen-training-version",
                runner_version=V8_COMPOSITE,
                checkpoint_parameter_sha256="f" * 64)
            self.assertEqual(receipt["runner_version"], V8_COMPOSITE)

    def test_row_version_change_is_metadata_only_not_science(self):
        with tempfile.TemporaryDirectory(dir=V8.HERE) as directory:
            root = Path(directory)
            science = {k: v for k, v in booking_fixture(
                root, row_runner_version=V8_COMPOSITE)[0].items()
                if k not in ("runner_version",)}
            self.assertNotIn("runner_version", science)


class LiveRuntimeBindingTests(unittest.TestCase):
    """Load the actual runtime and check the unified binding without any episode."""

    @classmethod
    def setUpClass(cls):
        cls.base, cls.runtime = V8.configure_eval_runtime()

    def test_row_generation_and_booking_read_the_same_v8_identity(self):
        composite = self.runtime["v8_runner_version"]
        self.assertEqual(self.runtime["r42_run"].RUNNER_VERSION, composite)
        self.assertEqual(self.runtime["v8_r44_module"].RUNNER_VERSION, composite)
        booked = self.runtime["v8_r44_booked"]
        self.assertIs(booked.__globals__, self.runtime["v8_r44_module"].__dict__)
        self.assertIn("create_claim_artifact_receipt", booked.__code__.co_names)
        self.assertEqual(self.runtime["v8_runner"].RUNNER_VERSION, composite)

    def test_private_output_namespace_is_the_v8_root(self):
        self.assertEqual(Path(self.runtime["common"].OUTPUT_DIR), V8.V8_ROOT)
        self.assertEqual(Path(self.runtime["common"].ROWS), V8.V8_ROWS)
        self.assertEqual(self.runtime["r40_common"].ROUND_DIR, V8.V8_ROOT)

    def test_binding_guard_passes_a_planned_cell_and_rejects_unplanned(self):
        plan = V8.build_quota_plan()
        runtime = self.runtime
        identity = {"file_sha256": V8.V7.MODEL_FILE_SHA,
                    "parameter_identity": {"parameter_sha256": V8.V7.MODEL_PARAMETER_SHA}}
        V8.assert_binding(runtime, key=(11, "greedy", 1_301_001), arm="NEW",
                          identity=identity, plan=plan)
        with self.assertRaises(V8.V8Error):
            V8.assert_binding(runtime, key=(11, "greedy", 1_301_099), arm="NEW",
                              identity=identity, plan=plan)
        with self.assertRaises(V8.V8Error):
            V8.assert_binding(runtime, key=(11, "greedy", 1_301_001), arm="OLD_R43_NEW_EP128",
                              identity=identity, plan=plan)

    def test_binding_guard_rejects_wrong_model_identity(self):
        plan = V8.build_quota_plan()
        bad = {"file_sha256": "0" * 64}
        with self.assertRaises(V8.V8Error):
            V8.assert_binding(self.runtime, key=(11, "greedy", 1_301_001),
                              arm="NEW", identity=bad, plan=plan)

    def test_binding_guard_rejects_stale_booking_module(self):
        plan = V8.build_quota_plan()
        r44 = self.runtime["v8_r44_module"]
        saved = r44.RUNNER_VERSION
        try:
            r44.RUNNER_VERSION = OLD_ROUND46_RUNNER
            with self.assertRaises(V8.V8Error):
                V8.assert_binding(self.runtime, key=(11, "greedy", 1_301_001),
                                  arm="NEW",
                                  identity={"file_sha256": V8.V7.MODEL_FILE_SHA,
                                            "parameter_identity": {"parameter_sha256": V8.V7.MODEL_PARAMETER_SHA}},
                                  plan=plan)
        finally:
            r44.RUNNER_VERSION = saved

    def test_two_isolated_r44_module_instances_do_not_cross_versions(self):
        """The binding lives per live module object, not in shared process state."""
        plan = V8.build_quota_plan()
        import types
        fake = types.SimpleNamespace(RUNNER_VERSION=OLD_ROUND46_RUNNER)
        runtime_b = dict(self.runtime)
        runtime_b["v8_r44_module"] = fake
        identity = {"file_sha256": V8.V7.MODEL_FILE_SHA,
                    "parameter_identity": {"parameter_sha256": V8.V7.MODEL_PARAMETER_SHA}}
        with self.assertRaises(V8.V8Error):
            V8.assert_binding(runtime_b, key=(11, "greedy", 1_301_001), arm="NEW",
                              identity=identity, plan=plan)
        fake.RUNNER_VERSION = self.runtime["v8_runner_version"]
        V8.assert_binding(runtime_b, key=(11, "greedy", 1_301_001), arm="NEW",
                          identity=identity, plan=plan)
        # The live runtime is untouched by the isolated instance.
        self.assertEqual(self.runtime["v8_r44_module"].RUNNER_VERSION,
                         self.runtime["v8_runner_version"])


class RealR40BookingSeamTests(unittest.TestCase):
    """Book the real R40 claim seam against the frozen V8 map (zero environment)."""

    def test_real_r40_claim_seam_books_planned_normal_and_reserve_once(self):
        source = (MODULE_PATH.parents[2].parent
                  / "zhenghe-round40/runs/comm_first_report_training_ablation_20261003/r40_common.py")
        spec = importlib.util.spec_from_file_location("r40_common_v8_booking_test", source)
        assert spec is not None and spec.loader is not None
        r40 = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = r40
        spec.loader.exec_module(r40)
        plan = V8.build_quota_plan()
        with tempfile.TemporaryDirectory(dir=V8.HERE) as directory:
            root = Path(directory)
            (root / "launch_slots" / "normal").mkdir(parents=True)
            (root / "launch_slots" / "reserve").mkdir(parents=True)
            old = {name: getattr(r40, name) for name in (
                "ROUND_DIR", "NORMAL_CAP", "RESERVE_CAP", "HARD_CAP",
                "_reserve_slot", "claim_launch", "next_attempt")}
            try:
                r40.ROUND_DIR = root
                r40.NORMAL_CAP, r40.RESERVE_CAP, r40.HARD_CAP = 480, 24, 504
                adapter = V8.install_quota_adapter({"r40_common": r40}, plan, output_root=root)
                first = plan["entries"][0]
                first_claim = r40.claim_launch(
                    first["logical_job"], category=first["category"], attempt=1)
                claim = V8.read_json(first_claim)
                self.assertEqual((claim["status"], claim["category"], claim["launch_slot"]),
                                 ("claimed_before_environment", "normal", 387))
                last = plan["entries"][-1]
                last_claim = r40.claim_launch(
                    last["logical_job"], category=last["category"], attempt=1)
                claim = V8.read_json(last_claim)
                self.assertEqual((claim["category"], claim["launch_slot"]),
                                 ("reserve", 2))
                with self.assertRaises(RuntimeError):
                    r40.claim_launch(first["logical_job"], category="normal", attempt=1)
                with self.assertRaises(V8.V8Error):
                    r40.claim_launch("model_eval_new_s11_greedy_scene1301000",
                                     category="reserve", attempt=1)
                self.assertEqual(len(adapter["claim_records"]), 2)
                self.assertEqual(len(list((root / "launch_claims").glob("*/*.json"))), 2)
                self.assertEqual(len(list((root / "launch_slots" / "normal").glob("*.json"))), 1)
                self.assertEqual(len(list((root / "launch_slots" / "reserve").glob("*.json"))), 1)
            finally:
                for name, value in old.items():
                    setattr(r40, name, value)


class Attempt2AdoptionTests(unittest.TestCase):
    def test_preserved_attempt2_bytes_match_the_registered_identity(self):
        for field, rel in (
            ("claim_sha256", V8.ATTEMPT2_CLAIM_REL),
            ("row_sha256", V8.ATTEMPT2_ROW_REL),
            ("lifecycle_sha256", V8.ATTEMPT2_COMPANIONS["lifecycle"]),
            ("transcript_sha256", V8.ATTEMPT2_COMPANIONS["transcript"]),
            ("tape_sha256", V8.ATTEMPT2_COMPANIONS["tape"]),
            ("failure_receipt_sha256", V8.ATTEMPT2_FAILURE_REL),
        ):
            self.assertEqual(V8.sha256_file(V8.HERE / rel), V8.ATTEMPT2_IDENTITY[field],
                             field)

    def test_adoption_decision_has_no_performance_gate(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('"performance_used_for_adoption": False', source)
        self.assertNotIn('on_time == 6', source)
        self.assertNotIn('spent_bytes == 9798', source)

    def test_attempt2_adoption_receipt_is_idempotent(self):
        decision = {"kind": "round46-r2-v8-attempt2-adoption-decision/1",
                    "task_id": V8.TASK_ID, "status": "accepted",
                    "performance_used_for_adoption": False, "marker": "fixture"}
        path = V8.V8_ROOT / "receipts" / "fixture_adoption.json"
        first = V8.write_new_json(path, decision)
        second = V8.write_new_json(path, decision)
        self.assertEqual(first, second)
        with self.assertRaises(V8.V7Error):
            V8.write_new_json(path, {**decision, "marker": "conflict"})

    def test_quota_plan_matches_the_authorized_95_cell_map(self):
        plan = V8.build_quota_plan()
        self.assertEqual(plan["new_claim_count"], 95)
        self.assertEqual(plan["adopted_key"], [11, "greedy", 1_301_000])
        self.assertEqual(plan["new_normal_slots"], list(V8.V7.NORMAL_SLOTS))
        self.assertEqual(plan["new_reserve_slots"], [2])
        self.assertEqual(plan["total_used_after"], 482)
        keys = [tuple(x["key"]) for x in plan["entries"]]
        self.assertNotIn((11, "greedy", 1_301_000), keys)
        self.assertEqual(len(set(keys)), 95)
        self.assertEqual((plan["entries"][-1]["category"],
                          plan["entries"][-1]["slot"],
                          plan["entries"][-1]["purpose"]),
                         ("reserve", 2, "technical-capacity-recovery"))


class FreezeGateTests(unittest.TestCase):
    def setUp(self):
        self._old_v8_root = V8.V8_ROOT
        self._tmp = tempfile.TemporaryDirectory(dir=V8.HERE)
        root = Path(self._tmp.name)
        self.v8_root = root / "v8_eval_booking_close"
        self.v8_root.mkdir()
        V8.V8_ROOT = self.v8_root
        V8.PROTECTED = {}
        self.sandbox = root / "sandbox_repo"
        self.sandbox.mkdir()
        subprocess.run(["git", "-C", str(self.sandbox), "-c", "user.email=v8@test",
                        "-c", "user.name=v8", "-c", "commit.gpgsign=false",
                        "init", "-b", V8.BRANCH], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(self.sandbox), "-c", "user.email=v8@test",
                        "-c", "user.name=v8", "-c", "commit.gpgsign=false",
                        "commit", "--allow-empty", "-m", "identity"],
                       check=True, capture_output=True)
        self.journal = root / "journal.jsonl"
        self.journal.write_bytes(b"{}\n")
        self.base = type("Base", (), {"REPO": self.sandbox,
                                      "CHECKPOINT_JOURNAL_PATH": self.journal})()

    def tearDown(self):
        V8.V8_ROOT = self._old_v8_root
        V8.PROTECTED = V8.V7.PROTECTED
        self._tmp.cleanup()

    def _write_freeze(self, *, regression_field):
        freeze = {"task_id": V8.TASK_ID, "status": "frozen_before_environment",
                  "source_commit": subprocess.run(
                      ["git", "-C", str(self.sandbox), "rev-parse", "HEAD"],
                      check=True, capture_output=True, text=True).stdout.strip(),
                  "quota_branch": "adoption_attempt2", "new_claim_count": 95,
                  "checkpoint_files": {}}
        plan = {"branch": "adoption_attempt2", "new_claim_count": 95}
        gate = {"task_id": V8.TASK_ID, "status": "passed",
                "frozen_config_sha256": V8.sha256_bytes(
                    (json.dumps(freeze, ensure_ascii=False, indent=2) + "\n").encode("utf-8")),
                "quota_plan_sha256": V8.sha256_bytes(
                    (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8")),
                "quota_plan_canonical_sha256": V8.sha256_bytes(V8.canonical_json_bytes(plan)),
                "adapter_source_sha256": V8.sha256_file(MODULE_PATH),
                "source_hashes": {"r46_v8_eval_booking_close.py": V8.sha256_file(MODULE_PATH)},
                "checkpoint_journal_sha256": V8.sha256_file(self.journal)}
        if regression_field is not None:
            gate["zero_environment_regression"] = regression_field
        (self.v8_root / "ROUND46_R2_V8_FROZEN_CONFIG.json").write_bytes(
            (json.dumps(freeze, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        (self.v8_root / "ROUND46_R2_V8_QUOTA_PLAN.json").write_bytes(
            (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        (self.v8_root / "ROUND46_R2_V8_PRELAUNCH_GATE.json").write_bytes(
            (json.dumps(gate, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

    def test_gate_without_regression_field_is_rejected(self):
        self._write_freeze(regression_field=None)
        with self.assertRaises(V8.V8Error):
            V8._verify_v8_freeze(self.base, {})

    def test_gate_with_consistent_identity_passes(self):
        self._write_freeze(regression_field="passed")
        V8._verify_v8_freeze(self.base, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
