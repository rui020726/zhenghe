"""Pure schedule and eval-only guard regressions for ROUND44."""

from __future__ import annotations

import unittest
import hashlib
import json
import tempfile
import copy
import importlib.util
import sys
import types
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import r44_eval_resume as resume


class _Index:
    def __init__(self, complete=()):
        self.complete = set(complete)

    def get(self, seed, mode, scene):
        key = (seed, mode, scene)
        return {"key": key} if key in self.complete else None


class EvalScheduleTests(unittest.TestCase):
    def test_source_identity_uses_exact_preserved_and_current_revisions(self):
        self.assertEqual(
            resume.BASE_STOP_HEAD,
            "12c3bdde0c85d9cf31eb1f1fc264f397701babac",
        )
        self.assertEqual(
            resume.INITIAL_SOURCE_HEAD,
            "347da3b8b559e669c57334f51ed17bcb3e90e3b2",
        )
        self.assertEqual(
            resume.SOURCE_IDENTITY_VALIDATOR_HEAD,
            "3fbd702c398b9cc45d3495183b0ce73ac3ded178",
        )
        self.assertEqual(
            resume.SOURCE_AMBIENT_GUARD_HEAD,
            "b7c27b809cb7ac1fd3f6282a6c03265dfc44ea60",
        )
        self.assertEqual(
            resume.PRIOR_EXECUTION_COMMIT,
            "1f43d869df6200f9de3e6bd7ae139e8c6e7f7c10",
        )
        self.assertEqual(
            resume.RECEIPT_RESUME_SOURCE_HEAD,
            "c18e7d6564aac132104649dc18c3ab52acb76023",
        )
        self.assertEqual(
            resume.R2_EXECUTION_IDENTITY_HEAD,
            "a48261e5aa1def4ddf130f1377dbdff0e4cb51d1",
        )
        self.assertEqual(resume.R3_CLAIM_QUEUE_SOURCE_HEAD,
                         "9c7fb0bc61da2ca925ff51ff8658c166cf5e2ba4")
        self.assertEqual(resume.R3_INITIAL_IDENTITY_ATTEMPT_HEAD,
                         "a7e2f86392d31346a9bcfcc4cd8fe86b7068d066")
        self.assertEqual(resume.IDENTITY_ALLOWED_PATHS_R1, {
            f"{resume.RUN_REL}/runtime/ROUND44_EVAL_ONLY_GATE_R1.json",
            f"{resume.RUN_REL}/runtime/ROUND44_EVAL_ONLY_EXECUTION_IDENTITY_R1.json",
        })
        self.assertEqual(
            resume.SOURCE_REVISION_ALLOWED_PATHS,
            {
                f"{resume.RUN_REL}/ROUND44_CURRENT_TASK.json",
                f"{resume.RUN_REL}/ROUND44_EVAL_MECHANICAL_DIFF_R1.json",
                f"{resume.RUN_REL}/ROUND44_EVAL_RECEIPT_REPAIR_REGRESSION_R1.json",
                f"{resume.RUN_REL}/r44_eval_resume.py",
                f"{resume.RUN_REL}/test_r44_eval_resume.py",
            },
        )
        self.assertEqual(resume.SOURCE_AMBIENT_GUARD_REVISION_ALLOWED_PATHS,
                         resume.SOURCE_REVISION_ALLOWED_PATHS)
        self.assertEqual(resume.SOURCE_RECEIPT_RESUME_REVISION_ALLOWED_PATHS,
                         resume.SOURCE_REVISION_ALLOWED_PATHS)
        self.assertEqual(resume.SOURCE_CLAIM_QUEUE_REVISION_ALLOWED_PATHS,
                         resume.SOURCE_REVISION_ALLOWED_PATHS)
        self.assertEqual(resume.SOURCE_R3_IDENTITY_REPAIR_ALLOWED_PATHS,
                         resume.SOURCE_REVISION_ALLOWED_PATHS)
        self.assertEqual(
            resume.SOURCE_ALLOWED_PATHS,
            resume.INITIAL_SOURCE_ALLOWED_PATHS
            | resume.SOURCE_REVISION_ALLOWED_PATHS
            | resume.SOURCE_AMBIENT_GUARD_REVISION_ALLOWED_PATHS
            | resume.SOURCE_RECEIPT_RESUME_REVISION_ALLOWED_PATHS
            | resume.SOURCE_CLAIM_QUEUE_REVISION_ALLOWED_PATHS
            | resume.SOURCE_R3_IDENTITY_REPAIR_ALLOWED_PATHS,
        )

    def test_preserved_first_r3_identity_attempt_records_exact_stale_evidence_hashes(self):
        evidence = resume._verify_preserved_first_r3_identity_attempt()
        self.assertEqual(evidence["commit"], resume.R3_INITIAL_IDENTITY_ATTEMPT_HEAD)
        self.assertEqual(evidence["parent"], resume.R3_CLAIM_QUEUE_SOURCE_HEAD)
        stale = evidence["stale_evidence_hashes"]
        self.assertNotEqual(stale["receipt_regression_sha256"],
                            stale["source_receipt_regression_sha256"])
        self.assertNotEqual(stale["mechanical_diff_sha256"],
                            stale["source_mechanical_diff_sha256"])
        self.assertEqual(evidence["environment_starts"], 0)
        self.assertEqual(evidence["new_claims"], 0)

    def test_eval_row_comparison_uses_strict_persisted_json_array_semantics(self):
        in_memory = {"emission_sequence": [(1, "target-a", 0)], "attempt": 2}
        durable = {"emission_sequence": [[1, "target-a", 0]], "attempt": 2}
        self.assertNotEqual(in_memory, durable)
        self.assertTrue(resume._eval_rows_equivalent(in_memory, durable))
        self.assertFalse(resume._eval_rows_equivalent(
            {"value": 1}, {"value": 1.0}))
        self.assertFalse(resume._eval_rows_equivalent(
            {"value": -0.0}, {"value": 0.0}))

    def test_live_ledger_resume_reconciles_only_terminal_matrix_claims(self):
        retry = resume.RETRY_KEY
        normal = (11, "greedy", 1_301_001)
        old_path = Path("attempt_0001.json")
        retry_path = Path("attempt_0002.json")
        normal_path = Path("normal_attempt_0001.json")
        metadata_retry = {"arm": "NEW", "training_seed": 11,
                          "mode": "greedy", "scenario_seed": 1_301_000}
        metadata_normal = {"arm": "NEW", "training_seed": 11,
                           "mode": "greedy", "scenario_seed": 1_301_001}
        baseline_claim = {"attempt": 1, "status": "failed_technical"}
        complete = {
            "attempt": 2, "category": "reserve", "status": "complete",
            "logical_job": "model_eval_new_s11_greedy_scene1301000",
            "metadata": metadata_retry, "round": 44,
            "training_version": resume.TRAINING_VERSION,
            "comm_action_semantics": "one-shot-send-skip/1",
            "environment_constructor_calls": 1, "run_invocation_calls": 1,
            "joint_run_constructed_calls": 1, "joint_run_method_calls": 1,
            "run_return_calls": 1, "run_failure_calls": 0,
            "completed_steps": 1_000, "optimizer_step_calls": 0,
            "checkpoint_write_calls": 0,
        }
        normal_complete = {
            **complete, "attempt": 1, "category": "normal",
            "logical_job": "model_eval_new_s11_greedy_scene1301001",
            "metadata": metadata_normal,
        }
        claims = {old_path: baseline_claim, retry_path: complete,
                  normal_path: normal_complete}
        hashes = {old_path: "f" * 64, retry_path: "a" * 64,
                  normal_path: "b" * 64}
        baseline = {
            "normal_used": 385, "reserve_used": 0, "total_used": 385,
            "environment_constructor_calls": 385, "run_invocation_calls": 385,
            "completed_steps": 385_000, "completed_logical_jobs": 384,
            "technical_failed_claims": 1, "retry_claims": 0,
            "optimizer_step_calls": 1_016, "ppo_update_calls": 96,
            "checkpoint_write_calls": 297,
        }

        def claim_paths(key):
            return ([old_path, retry_path] if key == retry
                    else ([normal_path] if key == normal else []))

        with ExitStack() as stack:
            stack.enter_context(patch.object(resume, "evaluation_schedule", return_value=(retry, normal)))
            stack.enter_context(patch.object(resume, "_claim_attempt_paths", side_effect=claim_paths))
            stack.enter_context(patch.object(resume, "read_json", side_effect=claims.__getitem__))
            stack.enter_context(patch.object(resume, "sha256_file", side_effect=hashes.__getitem__))
            result = resume._counts_after_verified_eval_claims(
                {"failed_attempt1": {"artifacts": {"claim": {"sha256": "f" * 64}}}},
                baseline,
            )
        self.assertEqual(result["normal_used"], 386)
        self.assertEqual(result["reserve_used"], 1)
        self.assertEqual(result["total_used"], 387)
        self.assertEqual(result["completed_steps"], 387_000)
        self.assertEqual(result["completed_logical_jobs"], 386)
        self.assertEqual(result["retry_claims"], 1)

        incomplete = dict(normal_complete, status="claimed_before_environment")
        claims[normal_path] = incomplete
        with ExitStack() as stack:
            stack.enter_context(patch.object(resume, "evaluation_schedule", return_value=(retry, normal)))
            stack.enter_context(patch.object(resume, "_claim_attempt_paths", side_effect=claim_paths))
            stack.enter_context(patch.object(resume, "read_json", side_effect=claims.__getitem__))
            stack.enter_context(patch.object(resume, "sha256_file", side_effect=hashes.__getitem__))
            with self.assertRaisesRegex(resume.EvalResumeError, "nonterminal"):
                resume._counts_after_verified_eval_claims(
                    {"failed_attempt1": {"artifacts": {"claim": {"sha256": "f" * 64}}}},
                    baseline,
                )

    def test_fixed_matrix_has_96_cells_and_retry_is_first(self):
        schedule = resume.evaluation_schedule()
        self.assertEqual(len(schedule), 96)
        self.assertEqual(schedule[0], resume.RETRY_KEY)
        self.assertEqual(len(set(schedule)), 96)
        self.assertEqual(schedule[-1], (33, "sampled", 1_301_015))

    def test_retry_requires_attempt_two_and_reserve_metadata(self):
        metadata = {
            "arm": "NEW", "training_seed": 11,
            "mode": "greedy", "scenario_seed": 1_301_000,
        }
        self.assertEqual(
            resume.validate_eval_claim(
                "model_eval_new_s11_greedy_scene1301000", 2, metadata),
            (resume.RETRY_KEY, "reserve"),
        )
        with self.assertRaises(resume.EvalResumeError):
            resume.validate_eval_claim(
                "model_eval_new_s11_greedy_scene1301000", 1, metadata)

    def test_other_fixed_cell_requires_normal_attempt_one(self):
        metadata = {
            "arm": "NEW", "training_seed": 22,
            "mode": "sampled", "scenario_seed": 1_301_004,
        }
        self.assertEqual(
            resume.validate_eval_claim(
                "model_eval_new_s22_sampled_scene1301004", 1, metadata),
            ((22, "sampled", 1_301_004), "normal"),
        )
        with self.assertRaises(resume.EvalResumeError):
            resume.validate_eval_claim(
                "model_eval_new_s22_sampled_scene1301004", 2, metadata)

    def test_claim_rejects_wrong_metadata_and_foreign_keys(self):
        logical = "model_eval_new_s11_greedy_scene1301000"
        wrong_metadata = {
            "arm": "OLD", "training_seed": 11,
            "mode": "greedy", "scenario_seed": 1_301_000,
        }
        with self.assertRaises(resume.EvalResumeError):
            resume.validate_eval_claim(logical, 2, wrong_metadata)
        with self.assertRaises(resume.EvalResumeError):
            resume.validate_eval_claim(
                "model_eval_new_s11_greedy_scene1301016", 1,
                {"arm": "NEW", "training_seed": 11,
                 "mode": "greedy", "scenario_seed": 1_301_016})

    def test_retry_queue_must_precede_remaining_fixed_cells(self):
        queue = resume._make_claim_guard_queue(_Index())
        self.assertEqual(queue[0], resume.RETRY_KEY)
        self.assertEqual(len(queue), 96)
        completed_retry = resume._make_claim_guard_queue(
            _Index(complete=(resume.RETRY_KEY,)))
        self.assertEqual(len(completed_retry), 95)
        self.assertEqual(completed_retry[0], (11, "greedy", 1_301_001))

    def test_queue_advances_from_retry_to_next_fixed_cell(self):
        cursor = {"index": 0, "keys": resume._make_claim_guard_queue(_Index())}
        resume._consume_claim_queue_key(cursor, resume.RETRY_KEY)
        self.assertEqual(cursor["keys"][cursor["index"]], (11, "greedy", 1_301_001))
        self.assertEqual(len(cursor["keys"]), 96)
        self.assertEqual(cursor["index"], 1)

    def test_queue_rejects_consumption_out_of_order(self):
        cursor = {"index": 0, "keys": resume._make_claim_guard_queue(_Index())}
        with self.assertRaises(resume.EvalResumeError):
            resume._consume_claim_queue_key(cursor, (11, "greedy", 1_301_001))

    def test_actual_installed_claim_guard_advances_one_fixed_cursor_only(self):
        schedule = resume.evaluation_schedule()
        keys = list(schedule[1:4])
        self.assertEqual(len(keys), 3)
        self.assertTrue(all(key != resume.RETRY_KEY for key in keys))
        self.assertEqual(keys, sorted(keys, key=schedule.index))
        base_calls = []

        def base_claim(logical_job, *, attempt, metadata):
            base_calls.append((logical_job, attempt, metadata["scenario_seed"]))
            return Path(f"mock-claim-{metadata['scenario_seed']}.json")

        run_module = SimpleNamespace(_claim=base_claim)
        runtime = {}
        validator_calls = []
        cursor = resume._install_claim_order_guard(
            run_module, runtime, frozen={"frozen": True},
            identity_validator=lambda frozen: validator_calls.append(frozen),
            expected_keys=keys,
        )

        def call_claim(key):
            seed, mode, scene = key
            logical = f"model_eval_new_s{seed}_{mode}_scene{scene}"
            metadata = {"arm": "NEW", "training_seed": seed,
                        "mode": mode, "scenario_seed": scene}
            return run_module._claim(logical, attempt=1, metadata=metadata)

        with patch.object(resume, "read_json", return_value={"category": "normal"}):
            with self.assertRaisesRegex(resume.EvalResumeError, "claim order mismatch"):
                call_claim(keys[1])
            self.assertEqual(base_calls, [])
            for key in keys:
                self.assertEqual(call_claim(key), Path(f"mock-claim-{key[2]}.json"))
                resume._consume_claim_queue_key(cursor, key)
                if key == keys[0]:
                    with self.assertRaisesRegex(resume.EvalResumeError, "claim order mismatch"):
                        call_claim(key)
                    self.assertEqual(len(base_calls), 1)
            with self.assertRaisesRegex(resume.EvalResumeError, "schedule ended"):
                call_claim(schedule[4])

        self.assertEqual([item[2] for item in base_calls], [key[2] for key in keys])
        self.assertEqual(cursor["keys"], keys)
        self.assertEqual(cursor["index"], len(keys))
        self.assertEqual(len(validator_calls), 6)
        self.assertTrue(all(value == {"frozen": True} for value in validator_calls))


class EvalOnlyGuardTests(unittest.TestCase):
    def test_forbidden_training_optimizer_and_checkpoint_calls_raise(self):
        class Optimizer:
            def step(self):
                raise AssertionError("underlying optimizer must not be reached")

        class Adam(Optimizer):
            pass

        def untouched(*_args, **_kwargs):
            raise AssertionError("underlying operation must not be reached")

        run_module = SimpleNamespace(
            train_seed=untouched,
            run_training_episode=untouched,
            run_reference=untouched,
            run_rule=untouched,
        )
        ambient = {
            "python": [3, [1, 2, 3], None],
            "numpy": ["MT19937", [4, 5, 6], 7, 0, 0.0],
            "torch": [8, 9, 10],
        }
        restore_behavior = {"wrong_payload": False}

        def rng_state():
            return copy.deepcopy(ambient)

        def restore_rng(state):
            ambient.clear()
            ambient.update(copy.deepcopy(state))
            return True

        def read_only_loader(*_args, **_kwargs):
            saved = rng_state()
            ambient["torch"] = [77, 88]
            if restore_behavior["wrong_payload"]:
                trainer.restore_rng({**saved, "torch": [999]})
            else:
                trainer.restore_rng(saved)
            return {"load_mode": "read_only"}

        trainer = SimpleNamespace(
            ppo_update=untouched,
            save_checkpoint=untouched,
            load_checkpoint=untouched,
            restore_rng=restore_rng,
            rng_state=rng_state,
            load_checkpoint_read_only=read_only_loader,
        )
        r40_train = SimpleNamespace(
            train_seed=untouched,
            _save_milestone=untouched,
            _write_immutable_snapshot=untouched,
            _save_preupdate_checkpoint=untouched,
        )
        torch_module = SimpleNamespace(
            save=untouched,
            optim=SimpleNamespace(Optimizer=Optimizer, Adam=Adam),
        )
        eval_runtime = {"trainer": trainer, "r40_train": r40_train}
        guarded = resume.install_read_only_guards(
            run_module, eval_runtime, torch_module)

        self.assertIn("r42_run.train_seed", guarded)
        self.assertIn("Adam.step", guarded)
        with self.assertRaises(resume.EvalOnlyViolation):
            run_module.train_seed()
        with self.assertRaises(resume.EvalOnlyViolation):
            Adam().step()
        with self.assertRaises(resume.EvalOnlyViolation):
            torch_module.save(object(), "forbidden.pt")
        ambient_before = rng_state()
        self.assertEqual(trainer.load_checkpoint_read_only("ep128.pt")["load_mode"],
                         "read_only")
        self.assertEqual(rng_state(), ambient_before)
        self.assertEqual(eval_runtime["eval_only_rng_guard_state"]["read_only_load_calls"], 1)
        self.assertEqual(eval_runtime["eval_only_rng_guard_state"]["ambient_restore_calls"], 1)
        with self.assertRaises(resume.EvalOnlyViolation):
            trainer.restore_rng(ambient_before)
        restore_behavior["wrong_payload"] = True
        with self.assertRaisesRegex(resume.EvalOnlyViolation, "non-ambient/checkpoint RNG"):
            trainer.load_checkpoint_read_only("bad.pt")

    def test_real_versioned_read_only_loader_preserves_only_ambient_rng(self):
        import torch

        run_dir = Path(resume.RUN_DIR)
        ambient = {
            "python": [3, [1, 2, 3], None],
            "numpy": ["MT19937", [4, 5, 6], 7, 0, 0.0],
            "torch": [8, 9, 10],
        }
        behavior = {"restore_checkpoint_rng": False}
        one_shot = types.ModuleType("r42_one_shot")
        one_shot.COMM_ACTION_SEMANTICS = "one-shot-send-skip/1"
        r40_versions = types.ModuleType("r40_versions")
        r40_versions.COMM_ACTION_ELIGIBILITY = "one-shot-send-skip/1"
        r40_versions.REWARD_TIMING_VERSION = "published-completion-event-at-relay-tick/2"
        r40_versions.install = lambda *_args, **_kwargs: None
        training_state = {"comm_action_semantics": "one-shot-send-skip/1"}
        payload = {
            "training_state": training_state,
            "comm_action_semantics": "one-shot-send-skip/1",
            "freeze": {"source": "synthetic read-only fixture"},
            "rng": {"python": [999], "numpy": [999], "torch": [999]},
        }
        trainer_holder = {}

        def rng_state():
            return copy.deepcopy(ambient)

        def restore_rng(state):
            ambient.clear()
            ambient.update(copy.deepcopy(state))
            return True

        def base_read_only(_path, *, freeze):
            self.assertEqual(freeze, payload["freeze"])
            ambient["torch"] = [77, 88]
            if behavior["restore_checkpoint_rng"]:
                trainer_holder["trainer"].restore_rng(payload["rng"])
            return object(), training_state, {
                "optimizer_restored": False, "rng_restored": False,
            }

        trainer = SimpleNamespace(
            _round42_one_shot_installed=False,
            STATE_SCHEMA="schema", TRAINER_TAG="trainer",
            LOSS_VERSION="loss", GAE_SCHEMA="gae", OPTIMIZER_SCHEMA="optim",
            REWARD_TIMING_SCHEMA="timing", LEARNING_RATE=0.001,
            fresh_training_state=lambda _seed: {"eta_lambda": 0.5},
            ppo_update=lambda: None,
            load_checkpoint_read_only=base_read_only,
            rng_state=rng_state, restore_rng=restore_rng,
            _validate_checkpoint_base=lambda *_args, **_kwargs: training_state,
            _restore_controller=lambda *_args, **_kwargs: object(),
        )
        trainer_holder["trainer"] = trainer
        runner = SimpleNamespace()
        common = SimpleNamespace(ETA_LAMBDA=0.5, LEARNING_RATE=0.001)
        checkpoint_path = Path(run_dir) / "runtime" / "nonexistent_read_only_fixture.pt"

        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {
                "r42_one_shot": one_shot, "r40_versions": r40_versions,
            }))
            original_sys_path = list(sys.path)
            stack.callback(lambda: sys.path.__setitem__(slice(None), original_sys_path))
            sys.path.insert(0, str(run_dir))
            spec = importlib.util.spec_from_file_location(
                "r42_versions_eval_identity_test", run_dir / "r42_versions.py")
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            versioned = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(versioned)

            original_save = torch.save
            original_optimizer_step = torch.optim.Optimizer.step
            original_adam_step = torch.optim.Adam.step
            stack.enter_context(patch.object(torch, "save", original_save))
            stack.enter_context(patch.object(torch.optim.Optimizer, "step", original_optimizer_step))
            stack.enter_context(patch.object(torch.optim.Adam, "step", original_adam_step))
            stack.enter_context(patch.object(torch, "load", return_value=payload))

            versioned.install(
                trainer, runner, common=common,
                freeze={"comm_action_semantics": "one-shot-send-skip/1"},
                round40_freeze={},
            )
            eval_runtime = {"trainer": trainer, "r40_train": None}
            resume.install_read_only_guards(
                SimpleNamespace(train_seed=lambda: None, run_training_episode=lambda: None),
                eval_runtime, torch,
            )
            before = rng_state()
            _, _, identity = trainer.load_checkpoint_read_only(checkpoint_path)
            self.assertTrue(identity["ambient_rng_preserved"])
            self.assertIs(identity["optimizer_restored"], False)
            self.assertIs(identity["rng_restored"], False)
            self.assertEqual(rng_state(), before)

            with self.assertRaises(resume.EvalOnlyViolation):
                trainer.restore_rng(payload["rng"])
            behavior["restore_checkpoint_rng"] = True
            before_failed_load = rng_state()
            with self.assertRaisesRegex(
                    resume.EvalOnlyViolation, "non-ambient/checkpoint RNG"):
                trainer.load_checkpoint_read_only(checkpoint_path)
            self.assertEqual(rng_state(), before_failed_load)


class VerifyEvalIdentityTests(unittest.TestCase):
    SOURCE_HEAD = "f" * 40
    IDENTITY_HEAD = "e" * 40
    PRIOR_EXECUTION_HEAD = "1f43d869df6200f9de3e6bd7ae139e8c6e7f7c10"
    SOURCE_FILE = f"{resume.RUN_REL}/r44_eval_resume.py"

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _verify(self, *, wrong_parent=False, unregistered_path=False,
                dirty_status="", drift_gate=False, drift_source=False,
                drift_chain_evidence=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        relative_paths = {
            name: root / Path(name) for name in resume.IDENTITY_ALLOWED_PATHS
        }
        for path in relative_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        run_dir = root / "runs" / Path(resume.RUN_DIR.name)
        run_dir.mkdir(parents=True, exist_ok=True)
        runtime_dir = run_dir / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        source_path = root / Path(self.SOURCE_FILE)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(b"committed eval-only source\n")

        gate_rel, identity_rel = resume._identity_paths()
        gate_path, identity_path = relative_paths[gate_rel], relative_paths[identity_rel]
        readonly_path = runtime_dir / "readonly.json"
        manifest_path = runtime_dir / "manifest.json"
        post_retry_audit_path = runtime_dir / "post_retry_audit.json"
        prior_gate_rel, prior_identity_rel = sorted(resume.IDENTITY_ALLOWED_PATHS_R1)
        prior_identity_path = root / Path(prior_identity_rel)
        prior_gate_path = root / Path(prior_gate_rel)
        prior_identity_path.parent.mkdir(parents=True, exist_ok=True)
        prior_gate_path.parent.mkdir(parents=True, exist_ok=True)
        prior_progress_path = runtime_dir / "prior_progress.json"
        regression_path = run_dir / "ROUND44_EVAL_RECEIPT_REPAIR_REGRESSION_R1.json"
        diff_path = run_dir / "ROUND44_EVAL_MECHANICAL_DIFF_R1.json"
        for path in (manifest_path, regression_path, diff_path):
            path.write_text("{}\n", encoding="utf-8")
        readonly_path.write_text(json.dumps({
            "status": "passed", "source_or_execution_head": resume.SOURCE_AMBIENT_GUARD_HEAD,
        }) + "\n", encoding="utf-8")
        prior_identity = {
            "status": "passed", "source_commit": resume.SOURCE_AMBIENT_GUARD_HEAD,
            "source_commit_chain": [resume.BASE_STOP_HEAD, resume.INITIAL_SOURCE_HEAD,
                resume.SOURCE_IDENTITY_VALIDATOR_HEAD, resume.SOURCE_AMBIENT_GUARD_HEAD],
        }
        prior_gate = {"status": "passed", "source_commit": resume.SOURCE_AMBIENT_GUARD_HEAD}
        prior_progress = {
            "prelaunch_head": self.PRIOR_EXECUTION_HEAD,
            "status": "running_retry_gate", "retry_gate_passed": False,
            "completed_eval_keys": [], "evaluations_completed": 0,
        }
        post_retry_audit = {
            "status": "passed",
            "failure": {
                "execution_identity_head": self.PRIOR_EXECUTION_HEAD,
                "attempt2_claim_sha256": "c" * 64,
                "attempt2_row_sha256": "d" * 64,
            },
            "attempt2": {"claim_status": "complete", "eval_index_completed_count": 1},
            "old_attempt1": {"unchanged": True},
            "training_immutability": {
                "all_204_checkpoint_hashes_unchanged": True,
                "all_training_journal_hashes_unchanged": True,
            },
        }
        prior_identity_path.write_text(json.dumps(prior_identity) + "\n", encoding="utf-8")
        prior_gate_path.write_text(json.dumps(prior_gate) + "\n", encoding="utf-8")
        prior_progress_path.write_text(json.dumps(prior_progress) + "\n", encoding="utf-8")
        post_retry_audit_path.write_text(json.dumps(post_retry_audit) + "\n", encoding="utf-8")

        chain_evidence = {
            "base_stop_head": resume.BASE_STOP_HEAD,
            "initial_source_head": resume.INITIAL_SOURCE_HEAD,
            "source_identity_validator_head": resume.SOURCE_IDENTITY_VALIDATOR_HEAD,
            "ambient_guard_head": resume.SOURCE_AMBIENT_GUARD_HEAD,
            "prior_execution_commit": resume.PRIOR_EXECUTION_COMMIT,
            "revised_source_head": self.SOURCE_HEAD,
            "initial_source_allowed_paths": sorted(resume.INITIAL_SOURCE_ALLOWED_PATHS),
            "source_identity_validator_revision_allowed_paths": sorted(
                resume.SOURCE_REVISION_ALLOWED_PATHS),
            "ambient_guard_revision_allowed_paths": sorted(
                resume.SOURCE_AMBIENT_GUARD_REVISION_ALLOWED_PATHS),
            "receipt_resume_revision_allowed_paths": sorted(
                resume.SOURCE_RECEIPT_RESUME_REVISION_ALLOWED_PATHS),
            "source_file_sha256": {self.SOURCE_FILE: self._sha(source_path)},
        }
        failed_attempt_evidence = {
            "commit": resume.R3_INITIAL_IDENTITY_ATTEMPT_HEAD,
            "parent": resume.R3_CLAIM_QUEUE_SOURCE_HEAD,
            "gate_sha256": "4" * 64,
            "identity_sha256": "5" * 64,
        }
        chain_evidence["first_r3_identity_attempt"] = failed_attempt_evidence
        stored_chain_evidence = dict(chain_evidence)
        r2_evidence = {
            "commit": resume.R2_EXECUTION_IDENTITY_HEAD,
            "identity_sha256": "2" * 64,
            "gate_sha256": "3" * 64,
            "progress_sha256": None,
            "progress_exists": False,
        }
        history = {"r4_identity_sha256": "a" * 64}
        gate = {
            "kind": "round44-eval-only-prelaunch-gate/1",
            "status": "passed",
            "source_commit": self.SOURCE_HEAD,
            "source_commit_parent": resume.R3_INITIAL_IDENTITY_ATTEMPT_HEAD,
            "source_commit_allowed_paths": sorted(
                resume.SOURCE_R3_IDENTITY_REPAIR_ALLOWED_PATHS),
            "first_r3_identity_attempt_commit": failed_attempt_evidence["commit"],
            "first_r3_identity_attempt_gate_sha256": failed_attempt_evidence["gate_sha256"],
            "first_r3_identity_attempt_identity_sha256": failed_attempt_evidence["identity_sha256"],
            "readonly_model_gate_sha256": self._sha(readonly_path),
            "fixed_eval_keys_sha256": "b" * 64,
            "post_retry_receipt_audit_sha256": self._sha(post_retry_audit_path),
            "prior_execution_commit": self.PRIOR_EXECUTION_HEAD,
            "prior_execution_identity_sha256": self._sha(prior_identity_path),
            "prior_execution_gate_sha256": self._sha(prior_gate_path),
            "prior_progress_sha256": self._sha(prior_progress_path),
            "prior_r2_identity_commit": r2_evidence["commit"],
            "prior_r2_identity_sha256": r2_evidence["identity_sha256"],
            "prior_r2_gate_sha256": r2_evidence["gate_sha256"],
            "prior_r2_progress_sha256": r2_evidence["progress_sha256"],
            "read_only_model_gate_reused_from_source_commit": resume.SOURCE_AMBIENT_GUARD_HEAD,
            "quota": {"normal": 480, "reserve": 24, "hard_cap": 504},
        }
        gate_path.write_text(json.dumps(gate, sort_keys=True) + "\n", encoding="utf-8")
        identity = {
            "kind": "round44-eval-only-execution-identity/1",
            "status": "passed",
            "branch": resume.BRANCH,
            "source_commit": self.SOURCE_HEAD,
            "source_commit_parent": resume.R3_INITIAL_IDENTITY_ATTEMPT_HEAD,
            "source_commit_allowed_paths": sorted(
                resume.SOURCE_R3_IDENTITY_REPAIR_ALLOWED_PATHS),
            "first_r3_identity_attempt_commit": failed_attempt_evidence["commit"],
            "first_r3_identity_attempt_gate_sha256": failed_attempt_evidence["gate_sha256"],
            "first_r3_identity_attempt_identity_sha256": failed_attempt_evidence["identity_sha256"],
            "source_commit_chain": [
                resume.BASE_STOP_HEAD, resume.INITIAL_SOURCE_HEAD,
                resume.SOURCE_IDENTITY_VALIDATOR_HEAD,
                resume.SOURCE_AMBIENT_GUARD_HEAD,
                resume.PRIOR_EXECUTION_COMMIT, resume.RECEIPT_RESUME_SOURCE_HEAD,
                resume.R2_EXECUTION_IDENTITY_HEAD, resume.R3_CLAIM_QUEUE_SOURCE_HEAD,
                resume.R3_INITIAL_IDENTITY_ATTEMPT_HEAD, self.SOURCE_HEAD],
            "source_commit_chain_evidence": stored_chain_evidence,
            "gate_sha256": self._sha(gate_path),
            "fixed_eval_keys_sha256": gate["fixed_eval_keys_sha256"],
            "post_retry_receipt_audit_sha256": self._sha(post_retry_audit_path),
            "prior_execution_commit": self.PRIOR_EXECUTION_HEAD,
            "prior_execution_identity_sha256": self._sha(prior_identity_path),
            "prior_execution_gate_sha256": self._sha(prior_gate_path),
            "prior_progress_sha256": self._sha(prior_progress_path),
            "prior_r2_identity_commit": r2_evidence["commit"],
            "prior_r2_identity_sha256": r2_evidence["identity_sha256"],
            "prior_r2_gate_sha256": r2_evidence["gate_sha256"],
            "prior_r2_progress_sha256": r2_evidence["progress_sha256"],
            "attempt2_claim_sha256": post_retry_audit["failure"]["attempt2_claim_sha256"],
            "attempt2_row_sha256": post_retry_audit["failure"]["attempt2_row_sha256"],
            "readonly_model_gate_sha256": self._sha(readonly_path),
            "source_file_sha256": {self.SOURCE_FILE: self._sha(source_path)},
            "immutable_input_manifest_sha256": self._sha(manifest_path),
            "receipt_regression_sha256": self._sha(regression_path),
            "mechanical_diff_sha256": self._sha(diff_path),
            "preserved_r4_history_hashes": history,
            "retry_key": [11, "greedy", 1_301_000],
            "schedule": [[s, m, scene] for s, m, scene in resume.evaluation_schedule()],
            "counts_before_r3_resume": {
                "normal_used": 385, "reserve_used": 1,
                "completed_training_jobs": 384, "valid_new_eval_cells": 1,
                "optimizer_step_calls": 1016, "ppo_update_calls": 96,
                "checkpoint_write_calls": 297,
            },
        }
        identity_path.write_text(json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8")
        if drift_chain_evidence:
            chain_evidence = {**chain_evidence, "unexpected": True}
        blobs = {
            (self.IDENTITY_HEAD, gate_rel): self._sha(gate_path),
            (self.IDENTITY_HEAD, identity_rel): self._sha(identity_path),
            (self.SOURCE_HEAD, self.SOURCE_FILE): self._sha(source_path),
            (resume.PRIOR_EXECUTION_COMMIT, prior_gate_rel): self._sha(prior_gate_path),
            (resume.PRIOR_EXECUTION_COMMIT, prior_identity_rel): self._sha(prior_identity_path),
        }
        if drift_gate:
            gate_path.write_text("{}\n", encoding="utf-8")
        if drift_source:
            source_path.write_bytes(b"uncommitted source drift\n")
        changed_paths = set(resume.IDENTITY_ALLOWED_PATHS)
        if unregistered_path:
            changed_paths.add("unregistered/foreign.json")
        parent = "bad-parent" if wrong_parent else self.SOURCE_HEAD

        def fake_git(*args, **_kwargs):
            if args[:2] == ("branch", "--show-current"):
                return resume.BRANCH
            if args and args[0] == "status":
                return dirty_status
            if args[:2] == ("rev-parse", "HEAD"):
                return self.IDENTITY_HEAD
            raise AssertionError(f"unexpected git read in identity test: {args}")

        def fake_blob(commit, repo_relative):
            return blobs.get((commit, repo_relative), "0" * 64)

        with ExitStack() as stack:
            for name, value in {
                "ROOT": root,
                "RUN_DIR": run_dir,
                "EVAL_IDENTITY": identity_path,
                "EVAL_GATE": gate_path,
                "EVAL_IDENTITY_R1": prior_identity_path,
                "EVAL_GATE_R1": prior_gate_path,
                "PRIOR_PROGRESS_R1": prior_progress_path,
                "POST_RETRY_AUDIT": post_retry_audit_path,
                "IMMUTABLE_MANIFEST": manifest_path,
                "READONLY_GATE": readonly_path,
            }.items():
                stack.enter_context(patch.object(resume, name, value))
            stack.enter_context(patch.object(resume, "_git", side_effect=fake_git))
            stack.enter_context(patch.object(resume, "_parent", return_value=parent))
            stack.enter_context(patch.object(
                resume, "_changed_paths", return_value=changed_paths))
            chain_mock = stack.enter_context(patch.object(
                resume, "_verify_source_commit_chain", return_value=chain_evidence))
            stack.enter_context(patch.object(
                resume, "_verify_preserved_r2_identity", return_value=r2_evidence))
            stack.enter_context(patch.object(resume, "_verify_r4_history", return_value=history))
            stack.enter_context(patch.object(resume, "_blob_sha", side_effect=fake_blob))
            result = resume.verify_eval_identity()
            chain_mock.assert_called_once_with(self.SOURCE_HEAD)
            return result

    def test_actual_identity_verifier_accepts_source_head_and_identity_child(self):
        head, identity, gate = self._verify()
        self.assertEqual(head, self.IDENTITY_HEAD)
        self.assertEqual(identity["source_commit"], self.SOURCE_HEAD)
        self.assertEqual(gate["source_commit"], self.SOURCE_HEAD)

    def test_actual_identity_verifier_rejects_wrong_parent_and_unregistered_path(self):
        for options in ({"wrong_parent": True}, {"unregistered_path": True}):
            with self.subTest(options=options):
                with self.assertRaises(resume.EvalResumeError):
                    self._verify(**options)

    def test_actual_identity_verifier_rejects_worktree_and_content_drift(self):
        cases = (
            {"dirty_status": " M source.py"},
            {"drift_gate": True},
            {"drift_source": True},
            {"drift_chain_evidence": True},
        )
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(resume.EvalResumeError):
                    self._verify(**options)

    def test_historical_freeze_and_r4_sources_match_real_git_identities(self):
        evidence = resume._verify_r4_history(resume.INITIAL_SOURCE_HEAD)
        self.assertEqual(evidence["original_freeze_source_count"], 13)
        self.assertEqual(
            evidence["r4_registered_mechanical_source_delta"]["paths"],
            ["r44_execute.py", "r44_runtime.py"],
        )
        self.assertEqual(
            evidence["r4_registered_mechanical_source_delta"]["training_execution_head"],
            resume.TRAINING_EXECUTION_HEAD,
        )


if __name__ == "__main__":
    unittest.main()
