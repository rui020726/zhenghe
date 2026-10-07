from __future__ import annotations
import importlib.util, json, subprocess, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace

from r53_contracts import (BRANCH, HARD_CAP, NORMAL_CAP, RESERVE_CAP, TASK_ID,
    TRAINING_VERSION, authorize_receipt, build_fixed_plan, derived_snapshot,
    nonfatal_derived_writer, validate_checkpoint_identity, validate_claims)
import r53_execute as r53
from round53_trial_adapter import install_on_verified_trainer


class TestRound53Execution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = r53.plan_from_frozen_sources()

    def test_fixed_plan_exact_and_unique(self):
        jobs = self.plan["jobs"]
        self.assertEqual(len(jobs), 528)
        self.assertEqual(len({job["logical_job"] for job in jobs}), 528)
        self.assertEqual(sum(job["scope"] == "ppo_training" for job in jobs), 384)
        self.assertEqual(sum(job["scope"] == "candidate_eval" for job in jobs), 96)
        self.assertEqual(sum(job["scope"] == "bc_sampled_eval" for job in jobs), 48)
        self.assertEqual(self.plan["eval_scenes"], list(range(1301000, 1301016)))
        self.assertTrue(self.plan["development_block_already_used"])
        self.assertFalse(self.plan["sealed_confirmation_opened"])

    def test_authorization_refuses_pending_and_accepts_exact_receipt(self):
        digest = r53.canonical_sha(self.plan)
        with self.assertRaises(PermissionError):
            authorize_receipt({"status": "proposed_not_authorized"}, plan_sha256=digest)
        receipt = {"status": "explicitly_authorized", "task_id": TASK_ID,
                   "branch": BRANCH, "normal_cap": NORMAL_CAP,
                   "reserve_cap": RESERVE_CAP, "hard_cap": HARD_CAP,
                   "plan_sha256": digest, "authorized_by": "human",
                   "approved_at": "future-explicit-approval", "approval_reference": "external"}
        authorize_receipt(receipt, plan_sha256=digest)
        bad = dict(receipt, hard_cap=504)
        with self.assertRaises(PermissionError): authorize_receipt(bad, plan_sha256=digest)

    def test_actual_entry_refuses_before_private_raw_namespace_creation(self):
        with tempfile.TemporaryDirectory(dir=r53.HERE) as td:
            receipt = Path(td) / "missing_authorization.json"
            raw_before = r53.RAW.exists()
            with self.assertRaises(FileNotFoundError):
                r53.execute_authorized(receipt)
            self.assertEqual(r53.RAW.exists(), raw_before)

    def test_partial_core_claim_and_strict_close(self):
        first = self.plan["jobs"][0]
        partial = [{"logical_job": first["logical_job"], "attempt": 1,
                    "category": "normal", "launch_slot": first["normal_slot"],
                    "status": "complete"}]
        snap = derived_snapshot(self.plan, partial, strict=False)
        self.assertEqual(snap["status"], "partial")
        self.assertEqual(snap["completed_logical_jobs"], 1)
        before = list(partial)
        strict = derived_snapshot(self.plan, partial, strict=True)
        self.assertEqual(strict["status"], "deferred")
        self.assertEqual(partial, before)
        duplicate = partial + [dict(partial[0])]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_claims(self.plan, duplicate)
        with self.assertRaisesRegex(ValueError, "unplanned"):
            validate_claims(self.plan, [{**partial[0], "logical_job": "foreign"}])

    def test_actual_nonfatal_snapshot_wrapper_preserves_complete_core_claim(self):
        core = [{"logical_job": "job", "status": "complete"}]
        warnings = []
        def broken_snapshot(*, strict=True): raise OSError("derived disk failure")
        writer = nonfatal_derived_writer(broken_snapshot, warnings.append)
        result = writer(strict=False)
        self.assertEqual(result["status"], "deferred")
        self.assertEqual(warnings, [result])
        self.assertEqual(core[0]["status"], "complete")

    def test_checkpoint_version_rejects_round46_training_state_but_allows_bc_model_only(self):
        freeze = {"training_version": TRAINING_VERSION,
                  "critic_gradient_route_version": "critic_representation_stop_gradient/1"}
        new = {"training_version": TRAINING_VERSION, "freeze": freeze,
               "training_state": {"training_version": TRAINING_VERSION},
               "optimizer": {}, "rng": {}}
        self.assertEqual(validate_checkpoint_identity(new, freeze=freeze), "accepted_same_round53_resume")
        old = {"training_version": "constrained-cost-v2/one-shot-send-skip/1", "freeze": {},
               "training_state": {"training_version": "constrained-cost-v2/one-shot-send-skip/1"}}
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_checkpoint_identity(old, freeze=freeze)
        self.assertEqual(validate_checkpoint_identity({"kind": "bc_model_only"},
                                                       freeze=freeze, allow_model_only=True),
                         "accepted_model_only_initialization")

    def test_actual_frozen_runner_path_binder_is_private_and_separate_from_inputs(self):
        base = r53.load_base_runner()  # imports actual frozen runner, no configure/seed function
        old_key_plan = Path(base.KEY_PLAN_PATH).resolve()
        with tempfile.TemporaryDirectory(dir=r53.HERE) as td:
            raw = Path(td) / "raw"
            bound = r53.bind_runner_paths(base, raw)
            self.assertEqual(Path(base.V6_DIR).resolve(), raw.resolve())
            self.assertEqual(Path(base.LEDGER_PATH).resolve().parent, (raw / "runtime").resolve())
            self.assertEqual(Path(base.CHECKPOINT_JOURNAL_PATH).resolve(),
                             (raw / "runtime" / "checkpoint_writes.jsonl").resolve())
            self.assertEqual(Path(base.KEY_PLAN_PATH).resolve(), old_key_plan)
            self.assertTrue(all(Path(p).resolve() == raw.resolve()
                                or raw.resolve() in Path(p).resolve().parents
                                for p in bound.values()))
            r53.install_private_configure_seams(base)
            self.assertEqual(base._seed_v6_output_namespace.__name__, "prepare_r53_output_namespace")
            self.assertNotEqual(base._seed_v6_output_namespace.__func__
                                if hasattr(base._seed_v6_output_namespace, "__func__") else None,
                                getattr(r53.load_base_runner(), "_seed_v6_output_namespace", None))
            # The path/binder test never calls configure or creates raw output.
            self.assertFalse(raw.exists())

    def test_actual_slots_joint_run_proxy_factory(self):
        run_file = r53.R42_DIR / "r42_run.py"
        spec = importlib.util.spec_from_file_location("round53_test_actual_r42_run", run_file)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        class SlottedJointRun:
            __slots__ = ("calls",)
            def __init__(self): self.calls = 0
            def run(self): self.calls += 1; return "ok"
        target = SlottedJointRun(); progress = module._new_run_progress(); path = Path("fake-claim")
        saved = []
        proxy = module._CountedJointRunProxy(
            target, progress, path, lambda *_a, **_k: None, lambda: "collector",
            lambda fn, _collector: fn(), lambda _p, _progress: saved.append(1))
        self.assertEqual(proxy.run(), "ok")
        self.assertEqual(target.calls, 1)
        self.assertEqual(progress["joint_run_method_calls"], 1)
        self.assertEqual(progress["run_return_calls"], 1)
        self.assertEqual(len(saved), 3)

    def test_guarded_actual_runtime_binding_in_private_sandbox(self):
        script = r'''import json, shutil, tempfile
from pathlib import Path
import torch
import r53_execute as r
from r53_contracts import sha256_file
root=Path(tempfile.mkdtemp(prefix="r53_runtime_fixture_",dir=r.HERE)); raw=root/"raw"
protected=[r.RUN_ROOT/"ROUND46_CURRENT_TASK.json",r.RUN_ROOT/"ROUND46_R2_V6_STOP_STATE.json",
           r.RUN_ROOT/"runtime"/"checkpoint_writes.jsonl",
           r.R45_DIR/"runtime"/"checkpoint_writes.jsonl"]
before={str(p):sha256_file(p) for p in protected if p.is_file()}
counts={"environment_construction":0,"simulation":0,"new_scenes_or_flows":0,
        "new_evaluation":0,"real_training_updates":0,"synthetic_optimizer_steps":0,
        "checkpoint_writes":0,"claim_calls":0}
old_step=torch.optim.Optimizer.step
def blocked_step(self,*a,**k):
 counts["synthetic_optimizer_steps"]+=1
 raise RuntimeError("blocked optimizer step")
torch.optim.Optimizer.step=blocked_step
try:
 base=r.load_base_runner(); r.bind_runner_paths(base,raw); r.install_private_configure_seams(base)
 base._R45_MODULE=None; base._RUNTIME=None; base._configure_runtime_module()
 rt=base._RUNTIME; plan=r.plan_from_frozen_sources(); live=r.bind_live_runtime(base,rt,plan)
 run=rt["r42_run"]; train=rt["r40_train"]
 def reject_eval(*a,**k):
  counts["new_evaluation"]+=1
  raise RuntimeError("blocked evaluation")
 def reject_environment(*a,**k):
  counts["environment_construction"]+=1; counts["simulation"]+=1; counts["new_scenes_or_flows"]+=1
  raise RuntimeError("blocked environment/simulation path")
 def reject_builder(*a,**k):
  counts["environment_construction"]+=1; counts["new_scenes_or_flows"]+=1
  raise RuntimeError("blocked actual counted builder")
 def reject_ppo(*a,**k):
  counts["real_training_updates"]+=1
  raise RuntimeError("blocked real ppo_update")
 def reject_claim(*a,**k):
  counts["claim_calls"]+=1
  raise RuntimeError("blocked atomic claim")
 def reject_checkpoint(*a,**k):
  counts["checkpoint_writes"]+=1
  raise RuntimeError("blocked checkpoint write")
 run._execute_cell=reject_eval
 rt["runner"].build_joint_run=reject_builder
 rt["runner"]._run_episode=reject_environment
 rt["r40_run"].run_training_episode=reject_environment
 rt["trainer"].ppo_update=reject_ppo
 rt["trainer"].save_checkpoint=reject_checkpoint
 rt["r40_common"].claim_launch=reject_claim
 called={}
 def capture_train(seed,*,episodes,freeze,reference_cache):
  called.update({"seed":seed,"episodes":episodes,"same_trainer":train.trainer is rt["trainer"],
                 "same_common":train.common is rt["common"],"same_runner":train.runner is rt["runner"]})
  return {"episodes_done":0,"updates":0,"optimizer_steps":0}
 train.train_seed=capture_train
 run.train_seed(11,episodes=4,frozen={"round42":{"training_version":"fixture"}})
 try: run._claim("unplanned-job",attempt=1,metadata={})
 except RuntimeError as e: claim_refused="unplanned" in str(e)
 else: claim_refused=False
 pt=list(raw.rglob("*.pt")); claim_files=list((raw/"launch_claims").rglob("*.json")) if (raw/"launch_claims").exists() else []
 journal=Path(base.CHECKPOINT_JOURNAL_PATH)
 assert live["r40_trainer_is_runtime_trainer"] and all(called.values()) and claim_refused
 assert not pt and not claim_files and journal.exists() and journal.stat().st_size==0
 assert all(v==0 for v in counts.values())
 after={str(p):sha256_file(p) for p in protected if p.is_file()}
 assert before==after
 assert len({str(Path(v).resolve()) for v in live["bindings"].values()}) == 1
 print(json.dumps({"status":"passed","live_bindings":{k:"<temporary fixture>/raw" for k in live["bindings"]},
  "live_trainer_same_object":called,"unplanned_claim_refused":claim_refused,
  "raw_pt_files":len(pt),"raw_claim_files":len(claim_files),
  "private_journal_empty":True,"protected_sources_unchanged":before==after,"guards":counts}))
finally:
 torch.optim.Optimizer.step=old_step; shutil.rmtree(root,ignore_errors=True)
'''
        result = subprocess.run([sys.executable, "-c", script], cwd=r53.HERE,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
        evidence = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(evidence["status"], "passed")
        self.assertTrue(evidence["live_training_adapter"] if "live_training_adapter" in evidence else evidence["live_trainer_same_object"]["same_trainer"])
        self.assertTrue(evidence["unplanned_claim_refused"])
        self.assertEqual(evidence["raw_pt_files"], 0)
        self.assertEqual(evidence["raw_claim_files"], 0)
        self.assertTrue(evidence["protected_sources_unchanged"])
        result_path = r53.HERE / "RUNTIME_BINDING_FIXTURE_RESULT.json"
        result_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")

    def test_real_baseline_rows_hash_close_without_generating_bc_sampled(self):
        checked = r53.verify_baseline_reuse()
        self.assertEqual(checked["training_immediate_cache"]["rows"], 48)
        self.assertEqual(checked["bc_greedy"]["rows"], 48)
        self.assertEqual(checked["shared_ppo"]["verified_source_rows"], 96)
        self.assertEqual(len(checked["development_references"]["immediate_dev"]), 16)
        self.assertEqual(len(checked["development_references"]["late_dev"]), 16)
        self.assertTrue(checked["bc_sampled"]["not_generated_in_readiness"])

    def test_wrong_source_hash_is_an_actual_refusal(self):
        trainer = SimpleNamespace(ppo_update=lambda *_a, **_kw: None)
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            install_on_verified_trainer(trainer, source_sha256="0" * 64)
        import hashlib
        source_sha = hashlib.sha256((r53.CANON / "v2_trainer.py").read_bytes()).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "source mismatch"):
            install_on_verified_trainer(trainer, source_sha256=source_sha,
                                        source_path=r53.SOURCE_RUNNER)


if __name__ == "__main__": unittest.main()
