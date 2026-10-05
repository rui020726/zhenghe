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
    def test_source_identity_uses_three_exact_mechanical_commits(self):
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
        self.assertEqual(
            resume.SOURCE_ALLOWED_PATHS,
            resume.INITIAL_SOURCE_ALLOWED_PATHS
            | resume.SOURCE_REVISION_ALLOWED_PATHS
            | resume.SOURCE_AMBIENT_GUARD_REVISION_ALLOWED_PATHS,
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
        queue = resume._make_claim_guard_queue(_Index())
        resume._consume_claim_queue_key(queue, resume.RETRY_KEY)
        self.assertEqual(queue[0], (11, "greedy", 1_301_001))
        self.assertEqual(len(queue), 95)

    def test_queue_rejects_consumption_out_of_order(self):
        queue = resume._make_claim_guard_queue(_Index())
        with self.assertRaises(resume.EvalResumeError):
            resume._consume_claim_queue_key(queue, (11, "greedy", 1_301_001))


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
        regression_path = run_dir / "ROUND44_EVAL_RECEIPT_REPAIR_REGRESSION_R1.json"
        diff_path = run_dir / "ROUND44_EVAL_MECHANICAL_DIFF_R1.json"
        for path in (readonly_path, manifest_path, regression_path, diff_path):
            path.write_text("{}\n", encoding="utf-8")

        chain_evidence = {
            "base_stop_head": resume.BASE_STOP_HEAD,
            "initial_source_head": resume.INITIAL_SOURCE_HEAD,
            "revised_source_head": self.SOURCE_HEAD,
            "source_file_sha256": {self.SOURCE_FILE: self._sha(source_path)},
        }
        stored_chain_evidence = dict(chain_evidence)
        history = {"r4_identity_sha256": "a" * 64}
        gate = {
            "kind": "round44-eval-only-prelaunch-gate/1",
            "status": "passed",
            "source_commit": self.SOURCE_HEAD,
            "readonly_model_gate_sha256": self._sha(readonly_path),
            "fixed_eval_keys_sha256": "b" * 64,
            "quota": {"normal": 480, "reserve": 24, "hard_cap": 504},
        }
        gate_path.write_text(json.dumps(gate, sort_keys=True) + "\n", encoding="utf-8")
        identity = {
            "kind": "round44-eval-only-execution-identity/1",
            "status": "passed",
            "branch": resume.BRANCH,
            "source_commit": self.SOURCE_HEAD,
            "source_commit_chain": [
                resume.BASE_STOP_HEAD, resume.INITIAL_SOURCE_HEAD,
                resume.SOURCE_IDENTITY_VALIDATOR_HEAD, self.SOURCE_HEAD],
            "source_commit_chain_evidence": stored_chain_evidence,
            "gate_sha256": self._sha(gate_path),
            "fixed_eval_keys_sha256": gate["fixed_eval_keys_sha256"],
            "readonly_model_gate_sha256": self._sha(readonly_path),
            "source_file_sha256": {self.SOURCE_FILE: self._sha(source_path)},
            "immutable_input_manifest_sha256": self._sha(manifest_path),
            "receipt_regression_sha256": self._sha(regression_path),
            "mechanical_diff_sha256": self._sha(diff_path),
            "preserved_r4_history_hashes": history,
            "retry_key": [11, "greedy", 1_301_000],
            "schedule": [[s, m, scene] for s, m, scene in resume.evaluation_schedule()],
            "counts_before_first_eval": {
                "normal_used": 385, "reserve_used": 0,
                "completed_training_jobs": 384, "valid_new_eval_cells": 0,
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
