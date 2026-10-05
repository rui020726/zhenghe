from __future__ import annotations

import json
import math
import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from r44_eval_index import (
    EvalIndexError,
    MODES,
    SCENES,
    SEEDS,
    audit_fixed_matrix,
    build_eval_index,
    create_claim_artifact_receipt,
    evaluation_schedule,
    expected_eval_keys,
)


RUN_DIR = Path(__file__).resolve().parent
R43_DIR = RUN_DIR.parent / "comm_one_shot_full_trace_ablation_20261004"
sys.path.insert(0, str(R43_DIR))
import r43_execute  # noqa: E402


TRAINING_VERSION = "round44-dual-eta-0.5-test/1"
REWARD_TIMING_VERSION = "published-completion-event-at-relay-tick/2"
RUNNER_VERSION = "counted-jointrun-proxy/1+eval-index/1"
PARAMETER_SHA = "a" * 64


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in values),
                    encoding="utf-8")


def make_cell(root: Path, key: tuple[int, str, int], *, attempt: int = 1,
              status: str = "complete") -> tuple[dict, Path, Path]:
    seed, mode, scene = key
    logical = f"model_eval_new_s{seed}_{mode}_scene{scene}"
    row_path = root / "rows" / (
        f"v2_r44_eval_new_s{seed}_{mode}_s{scene}_a{attempt}.json")
    lifecycle = root / "logs" / f"{row_path.stem}_rp.jsonl"
    transcript = root / "logs" / f"{row_path.stem}_ct.jsonl"
    tape = root / "logs" / f"{row_path.stem}_tape.jsonl"
    _write_jsonl(lifecycle, [{"event_type": "TaskCompleted", "tick": 1000}])
    _write_jsonl(transcript, [{"kind": "send", "step": 1}])
    _write_jsonl(tape, [{"step": 1}, {"step": 2}])
    parameter_identity = {
        "parameter_count": 5700,
        "parameter_sha256": PARAMETER_SHA,
        "init_seed": 20260927,
        "heads": ("action_head", "value_task_head", "value_cost_head"),
        "shapes": {"action_head.weight": [2, 64], "trunk.0.weight": [64, 19]},
        "note": "fixture parameter identity",
    }
    row = {
        "kind": "eval",
        "label": f"eval_new_s{seed}_{mode}_s{scene}_r44",
        "scenario_seed": scene,
        "attempt": attempt,
        "training_version": TRAINING_VERSION,
        "reward_timing_version": REWARD_TIMING_VERSION,
        "runner_version": RUNNER_VERSION,
        "comm_action_semantics": "one-shot-send-skip/1",
        "steps": 1000,
        "model_arm": "NEW",
        "training_seed": seed,
        "eval_mode": mode,
        "eval_block": f"one_shot_{mode}",
        "config_hash": "c" * 64,
        "action_rng_seed": 8675309,
        "eval_key": [seed, mode, scene],
        "checkpoint_parameter_identity": parameter_identity,
        "checkpoint_identity": {
            "file_sha256": "f" * 64, "parameter_identity": parameter_identity,
        },
        "checkpoint_load_identity": {
            "file_sha256": "f" * 64, "parameter_identity": parameter_identity,
            "optimizer_restored": False, "rng_restored": False,
        },
        "controller_parameters_after": parameter_identity,
        "collector_summary": {"decision_steps": 2},
        "budget_bytes": 29300,
        "budget_used_fraction": -0.0,
        "spent_bytes": 0,
        "normalised_cost_total": 0.0,
        "on_time": 6,
        "completed": ["target-1", "target-2"],
        "task_reward_total": 6.0,
        "task_reward_events": [{"task_id": "target-1", "reward": 1.0}],
        "task_stream": [{"task_id": "target-1", "completed": True}],
        "cost_stream": [{"bytes": 0, "cost": 0.0}],
        "policy": "learned",
        "comm_action_eligibility": "one-shot-send-skip/1",
        "action_probabilities": [{"send": 0.25, "skip": 0.75}],
        "behaviour_sha256": "b" * 64,
        "artefacts": {
            "row": str(row_path.resolve()),
            "lifecycle": str(lifecycle.resolve()),
            "transcript": str(transcript.resolve()),
            "tape": str(tape.resolve()),
        },
    }
    _write_json(row_path, row)
    claim_path = root / "launch_claims" / logical / f"attempt_{attempt:04d}.json"
    claim = {
        "logical_job": logical,
        "attempt": attempt,
        "category": "normal" if attempt == 1 else "reserve",
        "status": status,
        "round": 44,
        "training_version": TRAINING_VERSION,
        "runner_version": RUNNER_VERSION,
        "comm_action_semantics": "one-shot-send-skip/1",
        "metadata": {
            "arm": "NEW", "training_seed": seed, "mode": mode, "scenario_seed": scene,
        },
        "environment_constructor_calls": 1,
        "run_invocation_calls": 1,
        "joint_run_method_calls": 1,
        "run_return_calls": 1,
        "run_failure_calls": 0,
        "completed_steps": 1000,
    }
    if status == "complete":
        claim["round44_eval_artifact_receipt"] = create_claim_artifact_receipt(
            root, row, training_version=TRAINING_VERSION,
            runner_version=RUNNER_VERSION,
            checkpoint_parameter_sha256=PARAMETER_SHA,
        )
    _write_json(claim_path, claim)
    return row, row_path, claim_path


def build(root: Path):
    return build_eval_index(
        root,
        training_version=TRAINING_VERSION,
        reward_timing_version=REWARD_TIMING_VERSION,
        runner_version=RUNNER_VERSION,
        checkpoint_parameter_sha256_by_seed={seed: PARAMETER_SHA for seed in SEEDS},
    )


def legacy_lookup(root: Path, key: tuple[int, str, int]):
    prior = r43_execute.RUN_DIR
    try:
        r43_execute.RUN_DIR = root
        runtime = {
            "common": SimpleNamespace(ROWS=root / "rows"),
            "r40_common": SimpleNamespace(_slug=lambda logical: logical),
        }
        return r43_execute._completed_eval_row(runtime, *key)
    finally:
        r43_execute.RUN_DIR = prior


class EvaluationIndexTests(unittest.TestCase):
    def test_preserved_old_implementation_fails_on_actual_attempt1_tuple_heads(self) -> None:
        row_path = RUN_DIR / "rows" / "v2_r44_eval_new_s11_greedy_s1301000_a1.json"
        row = json.loads(row_path.read_text(encoding="utf-8"))
        tuple_paths = []

        def restore_parameter_heads(value, path="$path"):
            if isinstance(value, dict):
                restored = {}
                for key, item in value.items():
                    if key == "heads" and type(item) is list:
                        tuple_paths.append(f"{path}.heads")
                        restored[key] = tuple(item)
                    else:
                        restored[key] = restore_parameter_heads(item, f"{path}.{key}")
                return restored
            if isinstance(value, list):
                return [restore_parameter_heads(item, f"{path}[{index}]")
                        for index, item in enumerate(value)]
            return value

        memory_row = restore_parameter_heads(copy.deepcopy(row))
        self.assertGreaterEqual(len(tuple_paths), 1)
        self.assertIs(type(memory_row["checkpoint_parameter_identity"]["heads"]), tuple)
        old_path = RUN_DIR / "runtime" / "ROUND44_R44_EVAL_INDEX_PREPATCH_R1.py"
        spec = importlib.util.spec_from_file_location("r44_eval_index_prepatch_test", old_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        old_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = old_module
        spec.loader.exec_module(old_module)
        with self.assertRaisesRegex(old_module.EvalIndexError, "persisted row differs"):
            old_module.create_claim_artifact_receipt(
                RUN_DIR, memory_row,
                training_version=row["training_version"],
                runner_version=row["runner_version"],
                checkpoint_parameter_sha256=(
                    row["checkpoint_parameter_identity"]["parameter_sha256"]),
            )

    def test_actual_failed_attempt1_row_is_not_admitted_by_actual_index(self) -> None:
        partial = json.loads(
            (RUN_DIR / "ROUND44_PARTIAL_EXECUTION_LEDGER.json").read_text(encoding="utf-8"))
        failed_claim = json.loads(
            (RUN_DIR / "launch_claims" / "model_eval_new_s11_greedy_scene1301000"
             / "attempt_0001.json").read_text(encoding="utf-8"))
        actual_row = json.loads(
            (RUN_DIR / "rows" / "v2_r44_eval_new_s11_greedy_s1301000_a1.json")
            .read_text(encoding="utf-8"))
        self.assertEqual(failed_claim["status"], "failed_technical")
        self.assertEqual(actual_row["attempt"], 1)
        index = build_eval_index(
            RUN_DIR,
            training_version=actual_row["training_version"],
            reward_timing_version=actual_row["reward_timing_version"],
            runner_version=actual_row["runner_version"],
            checkpoint_parameter_sha256_by_seed={
                int(seed): item["parameter_sha256"]
                for seed, item in partial["training"].items()},
        )
        self.assertEqual(index.completed_count, 0)
        self.assertIsNone(index.get(11, "greedy", 1_301_000))

    def test_index_key_set_matches_real_legacy_lookup_and_fixed_schedule(self) -> None:
        self.assertEqual(len(expected_eval_keys()), 96)
        self.assertEqual(evaluation_schedule()[0], (11, "greedy", 1_301_000))
        self.assertEqual(evaluation_schedule()[-1], (33, "sampled", 1_301_015))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            retry_key = (11, "greedy", 1_301_000)
            for key in expected_eval_keys():
                if key == retry_key:
                    make_cell(root, key, attempt=1, status="failed_technical")
                    make_cell(root, key, attempt=2)
                else:
                    make_cell(root, key)
            _write_json(root / "rows" / "training_unrelated.json", {
                "kind": "train", "training_seed": 11, "episode_index": 0,
            })
            index = build(root)
            self.assertEqual(index.completed_keys, [list(key) for key in expected_eval_keys()])
            self.assertEqual(index.completed_count, 96)
            for key in expected_eval_keys():
                self.assertIsNotNone(legacy_lookup(root, key))
                self.assertEqual(index.get(*key)["label"],
                                 f"eval_new_s{key[0]}_{key[1]}_s{key[2]}_r44")
            audit = audit_fixed_matrix(
                root,
                training_version=TRAINING_VERSION,
                reward_timing_version=REWARD_TIMING_VERSION,
                runner_version=RUNNER_VERSION,
                checkpoint_parameter_sha256_by_seed={seed: PARAMETER_SHA for seed in SEEDS},
            )
            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["cell_count"], 96)
            self.assertEqual(set(audit["group_counts"].values()), {16})
            self.assertEqual(sum(entry["eval_key"] == [11, "greedy", 1_301_000]
                                 for entry in audit["entries"]), 1)

    def test_failed_attempt_plus_one_valid_success_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key = (11, "greedy", 1_301_000)
            make_cell(root, key, attempt=1, status="failed_technical")
            make_cell(root, key, attempt=2)
            self.assertEqual(build(root).completed_keys, [[11, "greedy", 1_301_000]])
            self.assertIsNotNone(legacy_lookup(root, key))

    def test_failed_attempt_row_alone_is_preserved_but_never_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_cell(root, (11, "greedy", 1_301_000), status="failed_technical")
            index = build(root)
            self.assertEqual(index.completed_count, 0)
            self.assertEqual(index.completed_keys, [])
            with self.assertRaisesRegex(EvalIndexError, "incomplete: missing="):
                audit_fixed_matrix(
                    root,
                    training_version=TRAINING_VERSION,
                    reward_timing_version=REWARD_TIMING_VERSION,
                    runner_version=RUNNER_VERSION,
                    checkpoint_parameter_sha256_by_seed={s: PARAMETER_SHA for s in SEEDS},
                )

    def test_tuple_heads_round_trip_is_accepted_by_receipt_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, row_path, _ = make_cell(root, (22, "sampled", 1_301_007))
            self.assertIs(type(row["checkpoint_parameter_identity"]["heads"]), tuple)
            stored = json.loads(row_path.read_text(encoding="utf-8"))
            self.assertIs(type(stored["checkpoint_parameter_identity"]["heads"]), list)
            index = build(root)
            self.assertEqual(index.completed_count, 1)

    def test_every_science_identity_and_row_field_must_match_at_receipt_creation(self) -> None:
        mutations = [
            lambda row: row["checkpoint_parameter_identity"].__setitem__(
                "heads", ("action_head", "changed", "value_cost_head")),
            lambda row: row["checkpoint_parameter_identity"].__setitem__(
                "heads", tuple(reversed(row["checkpoint_parameter_identity"]["heads"]))),
            lambda row: row["checkpoint_parameter_identity"].__setitem__(
                "parameter_sha256", "d" * 64),
            lambda row: row["checkpoint_load_identity"].__setitem__("file_sha256", "e" * 64),
            lambda row: row.__setitem__("model_arm", "OLD"),
            lambda row: row.__setitem__("training_seed", 11),
            lambda row: row.__setitem__("eval_mode", "greedy"),
            lambda row: row.__setitem__("scenario_seed", 1_301_008),
            lambda row: row.__setitem__("eval_key", [22, "sampled", 1_301_008]),
            lambda row: row.__setitem__("config_hash", "9" * 64),
            lambda row: row.__setitem__("budget_bytes", 29_301),
            lambda row: row.__setitem__("spent_bytes", 1),
            lambda row: row.__setitem__("completed", ["different-target"]),
            lambda row: row.__setitem__("policy", "forced-send"),
            lambda row: row.__setitem__("action_probabilities", [{"send": 0.75, "skip": 0.25}]),
            lambda row: row.__setitem__("task_reward_events", [{"task_id": "other", "reward": 0.0}]),
            lambda row: row.__setitem__("arbitrary_unknown_field", {"changed": True}),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    row, _, _ = make_cell(root, (22, "sampled", 1_301_007))
                    mutate(row)
                    with self.assertRaises(EvalIndexError):
                        create_claim_artifact_receipt(
                            root, row, training_version=TRAINING_VERSION,
                            runner_version=RUNNER_VERSION,
                            checkpoint_parameter_sha256=PARAMETER_SHA,
                        )

    def test_json_comparison_is_type_strict_and_preserves_negative_zero(self) -> None:
        mutations = [
            lambda row: row.__setitem__("budget_bytes", True),
            lambda row: row.__setitem__("steps", 1000.0),
            lambda row: row.__setitem__("budget_used_fraction", 0.0),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    row, _, _ = make_cell(root, (33, "greedy", 1_301_010))
                    mutate(row)
                    with self.assertRaises(EvalIndexError):
                        create_claim_artifact_receipt(
                            root, row, training_version=TRAINING_VERSION,
                            runner_version=RUNNER_VERSION,
                            checkpoint_parameter_sha256=PARAMETER_SHA,
                        )

    def test_unknown_nonfinite_and_non_string_key_values_are_rejected(self) -> None:
        mutations = [
            lambda row: row.__setitem__("nested", {"x": math.nan}),
            lambda row: row.__setitem__("nested", {"x": object()}),
            lambda row: row.__setitem__("nested", {1: "not-a-string-key"}),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    row, _, _ = make_cell(root, (33, "greedy", 1_301_010))
                    mutate(row)
                    with self.assertRaises(EvalIndexError):
                        create_claim_artifact_receipt(
                            root, row, training_version=TRAINING_VERSION,
                            runner_version=RUNNER_VERSION,
                            checkpoint_parameter_sha256=PARAMETER_SHA,
                        )

    def test_jsonl_receipt_and_index_reject_nested_overflow_and_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, _, _ = make_cell(root, (11, "greedy", 1_301_000))
            transcript = Path(row["artefacts"]["transcript"])
            transcript.write_text('{"nested":{"value":1e999}}\n', encoding="utf-8")
            with self.assertRaisesRegex(EvalIndexError, "non-finite JSON number"):
                create_claim_artifact_receipt(
                    root, row, training_version=TRAINING_VERSION,
                    runner_version=RUNNER_VERSION,
                    checkpoint_parameter_sha256=PARAMETER_SHA,
                )
            with self.assertRaisesRegex(EvalIndexError, "non-finite JSON number"):
                build(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, _, _ = make_cell(root, (11, "greedy", 1_301_000))
            transcript = Path(row["artefacts"]["transcript"])
            transcript.write_text('{"nested":{"value":1,"value":2}}\n', encoding="utf-8")
            with self.assertRaisesRegex(EvalIndexError, "unreadable JSONL artifact"):
                create_claim_artifact_receipt(
                    root, row, training_version=TRAINING_VERSION,
                    runner_version=RUNNER_VERSION,
                    checkpoint_parameter_sha256=PARAMETER_SHA,
                )
            with self.assertRaisesRegex(EvalIndexError, "unreadable JSONL artifact"):
                build(root)

    def test_duplicate_json_object_keys_and_nonstandard_numbers_are_rejected(self) -> None:
        for text in ('{"kind":"eval","kind":"eval"}', '{"kind":"eval","value":NaN}'):
            with self.subTest(text=text):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    _, row_path, _ = make_cell(root, (11, "greedy", 1_301_000))
                    row_path.write_text(text, encoding="utf-8")
                    with self.assertRaisesRegex(EvalIndexError, "unreadable JSON artifact"):
                        build(root)

    def test_missing_row_claim_receipt_and_wrong_attempt_link_fail_closed(self) -> None:
        key = (22, "greedy", 1_301_004)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, row_path, claim_path = make_cell(root, key)
            claim_path.unlink()
            with self.assertRaisesRegex(EvalIndexError, "no matching attempt claim"):
                build(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, row_path, _ = make_cell(root, key)
            row_path.unlink()
            with self.assertRaisesRegex(EvalIndexError, "incomplete"):
                build(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, claim_path = make_cell(root, key)
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
            claim.pop("round44_eval_artifact_receipt")
            _write_json(claim_path, claim)
            with self.assertRaisesRegex(EvalIndexError, "lacks ROUND44 artifact receipt"):
                build(root)

    def test_index_lookup_uses_cached_verified_row_without_filesystem_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key = (11, "greedy", 1_301_000)
            _, row_path, _ = make_cell(root, key)
            index = build(root)
            row_path.unlink()
            cached = index.get(*key)
            self.assertEqual(cached["label"], "eval_new_s11_greedy_s1301000_r44")

    def test_nonterminal_claim_fails_closed_instead_of_being_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logical = "model_eval_new_s11_greedy_scene1301000"
            _write_json(root / "launch_claims" / logical / "attempt_0001.json", {
                "logical_job": logical, "attempt": 1,
                "category": "normal",
                "status": "claimed_before_environment",
                "metadata": {"arm": "NEW", "training_seed": 11,
                             "mode": "greedy", "scenario_seed": 1_301_000},
            })
            with self.assertRaisesRegex(EvalIndexError, "nonterminal claim"):
                build(root)

    def test_duplicate_successes_missing_artifacts_and_hash_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key = (22, "sampled", 1_301_003)
            _, _, _ = make_cell(root, key, attempt=1)
            make_cell(root, key, attempt=2)
            with self.assertRaisesRegex(EvalIndexError, "duplicate successful"):
                build(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, row_path, _ = make_cell(root, key)
            row = json.loads(row_path.read_text(encoding="utf-8"))
            Path(row["artefacts"]["tape"]).unlink()
            with self.assertRaisesRegex(EvalIndexError, "missing"):
                build(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, row_path, _ = make_cell(root, key)
            row = json.loads(row_path.read_text(encoding="utf-8"))
            Path(row["artefacts"]["lifecycle"]).write_text(
                '{"event_type":"changed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(EvalIndexError, "hash/size differs"):
                build(root)

    def test_wrong_version_foreign_label_and_bad_checkpoint_hash_fail_closed(self) -> None:
        key = (33, "greedy", 1_301_010)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, row_path, claim_path = make_cell(root, key)
            row["training_version"] = "foreign/version"
            _write_json(row_path, row)
            with self.assertRaisesRegex(EvalIndexError, "row version"):
                build(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, row_path, _ = make_cell(root, key)
            row["label"] = "foreign_eval_label"
            _write_json(row_path, row)
            with self.assertRaisesRegex(EvalIndexError, "foreign or mismatched"):
                build(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_cell(root, key)
            with self.assertRaisesRegex(EvalIndexError, "parameter identity mismatch"):
                build_eval_index(
                    root, training_version=TRAINING_VERSION,
                    reward_timing_version=REWARD_TIMING_VERSION,
                    runner_version=RUNNER_VERSION,
                    checkpoint_parameter_sha256_by_seed={33: "b" * 64},
                )

    def test_empty_directory_is_empty_not_success_and_has_frozen_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = build(root)
            self.assertEqual(index.completed_count, 0)
            self.assertEqual(index.completed_keys, [])
            self.assertEqual(len(evaluation_schedule()), 96)
            self.assertIsNone(legacy_lookup(root, (11, "greedy", 1_301_000)))


if __name__ == "__main__":
    unittest.main()
