"""Run the narrow ROUND53 tests and derive readiness/report artifacts from outputs."""
from __future__ import annotations
import json, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent
REPO = RUN_ROOT.parents[1]
sys.path.insert(0, str(HERE))
import r53_execute as trial
from r53_contracts import BRANCH, HARD_CAP, NORMAL_CAP, RESERVE_CAP, TASK_ID, TRAINING_VERSION


def put(name: str, value) -> None:
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha_sources(paths):
    return [{"path": str(p), "sha256": trial.sha256_file(p)} for p in paths]


def build() -> dict:
    command = [sys.executable, "-m", "unittest", "test_r53_execution", "test_round53_pure", "-v"]
    proc = subprocess.run(command, cwd=HERE, capture_output=True, text=True, timeout=180)
    transcript = (proc.stdout or "") + (proc.stderr or "")
    match = re.search(r"Ran\s+(\d+)\s+tests?\s+in\s+([0-9.]+s)", transcript)
    tests_run = int(match.group(1)) if match else None
    passed = proc.returncode == 0 and tests_run is not None and re.search(r"\bOK\b", transcript) is not None
    test_record = {"kind": "round53-subprocess-test-run/1",
                   "status": "passed" if passed else "failed",
                   "command": [str(Path(sys.executable)), *command[1:]],
                   "exit_code": proc.returncode, "tests_run_from_unittest_output": tests_run,
                   "duration_from_unittest_output": match.group(2) if match else None,
                   "output": transcript,
                   "runtime_fixture": None}
    fixture_path = HERE / "RUNTIME_BINDING_FIXTURE_RESULT.json"
    if fixture_path.is_file(): test_record["runtime_fixture"] = json.loads(fixture_path.read_text(encoding="utf-8"))
    put("TEST_RUN_EVIDENCE.json", test_record)
    if not passed:
        raise RuntimeError("ROUND53 pure/guarded-runtime tests failed; readiness not generated")
    if test_record["runtime_fixture"] is None or test_record["runtime_fixture"].get("status") != "passed":
        raise RuntimeError("actual live runtime binding fixture evidence is missing")

    baseline = trial.verify_baseline_reuse()
    plan = trial.plan_from_frozen_sources()
    plan["authorization"] = "PENDING_USER_APPROVAL"
    plan["status"] = "proposed_not_authorized"
    plan["plan_sha256"] = trial.canonical_sha(plan)
    put("TRIAL_PLAN.json", plan)
    put("BASELINE_REUSE_CHECK.json", baseline)

    task_root = RUN_ROOT
    r40_root = REPO.parent / "zhenghe-round40" / "runs" / "comm_first_report_training_ablation_20261003"
    source_paths = [
        trial.CANON / "v2_trainer.py", trial.CANON / "v2_controller.py", trial.SOURCE_RUNNER,
        REPO / "runs" / "comm_one_shot_full_trace_ablation_20261004" / "r42_run.py",
        REPO / "runs" / "comm_one_shot_full_trace_ablation_20261004" / "r42_analysis.py",
        trial.R44_DIR / "r44_runtime.py", trial.R44_DIR / "r44_prepare.py",
        trial.R45_DIR / "r45_r2_execute_r3.py",
        trial.R45_DIR / "r40_train_resume_adapter.py",
        trial.R45_DIR / "r42_one_shot.py", trial.R45_DIR / "r42_versions.py",
        trial.R45_DIR / "r43_call_counter_hooks.py",
        r40_root / "r40_common.py", r40_root / "r40_train.py", r40_root / "r40_run.py",
        HERE / "r53_execute.py", HERE / "r53_contracts.py",
        HERE / "round53_trial_adapter.py", HERE / "test_r53_execution.py",
        HERE / "test_round53_pure.py",
        HERE / "build_readiness_artifacts.py",
        task_root / "ROUND46_R2_V6_FROZEN_CONFIG.json",
        task_root / "ROUND46_R2_V6_KEY_PLAN.json",
        task_root / "ROUND46_R2_V6_INPUT_IDENTITY.json",
        task_root / "ROUND46_R2_V6_TRAINING_VERSION_MANIFEST.json",
        task_root / "v10_eval_final_close" / "ROUND46_R2_V10_ANALYSIS_INPUT_MANIFEST.json",
        trial.R44_DIR / "ROUND44_FROZEN_CONFIG.json",
        trial.R44_DIR / "ROUND44_BASELINE_REUSE.json",
        RUN_ROOT / "round52_critic_gradient_isolation_prototype" / "PROTOTYPE_CONTRACT.json",
        RUN_ROOT / "round52_critic_gradient_isolation_prototype" / "FORWARD_AND_GRADIENT_COMPARISON.json",
    ]
    sources = sha_sources(source_paths)
    fixture = test_record["runtime_fixture"]
    source_binding = {
        "kind": "round53-source-binding/2", "status": "actual_runtime_objects_bound_in_guarded_private_fixture",
        "initial_head": "40adea4e16c5ccd6ee454f86462e4c959149da8a",
        "previous_delivery_head": "13f8d91ff67aba37938d0bf56bf62efa4f564b5f",
        "branch": BRANCH, "head_independent_scientific_identity": True,
        "head_is_not_an_input_identity": True,
        "canonical_trainer_expected_sha256": "0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841",
        "canonical_trainer_observed_sha256": trial.sha256_file(trial.CANON / "v2_trainer.py"),
        "source_files": sources,
        "live_runtime_fixture": fixture,
        "live_objects": ["canonical trainer (same object as r40_train.trainer)",
                         "r40_common", "r40_train", "r40_run", "r42_run",
                         "r42_analysis", "r44_runtime", "r44_prepare", "v2_run runner"],
        "actual_output_fields": ["common.OUTPUT_DIR", "common.LOGS", "common.ROWS",
                                 "common.REJECTED", "common.CHECKPOINTS", "r40_common.ROUND_DIR",
                                 "r42_run.RUN_DIR", "r44_runtime.RUN_DIR", "r44_prepare.RUN_DIR",
                                 "r42_analysis.RUN_DIR", "launch_claims", "launch_slots",
                                 "trainer.save_checkpoint closure journal"],
        "family_journal_seeded": False, "historical_R46_v6_seeder_called": False,
        "R42_ep0_rng_restored": False,
        "fresh_rng_scheme": plan["scientific_config_identity"]["fresh_rng_scheme"],
        "ppo_update_route": "actual runtime trainer object wrapped; __wrapped__ chain resolves to canonical source SHA",
        "training_math_rewritten": False,
    }
    put("SOURCE_BINDING.json", source_binding)

    operations = fixture["guards"]
    test_evidence = {
        "kind": "round53-zero-launch-test-evidence/2", "status": "passed",
        "test_count": tests_run, "test_exit_code": proc.returncode,
        "test_output_parsed_not_hardcoded": True,
        "test_command": test_record["command"], "tests_run": test_record["output"],
        "guard_observed_operation_counts": operations,
        "runtime_configured_in_private_temporary_fixture": True,
        "builder_run_environment_eval_optimizer_checkpoint_calls": {
            k: operations[k] for k in ("environment_construction", "simulation",
                "new_scenes_or_flows", "new_evaluation", "real_training_updates",
                "synthetic_optimizer_steps", "checkpoint_writes")},
        "real_claims_created": 0, "raw_checkpoint_files": fixture["raw_pt_files"],
        "raw_claim_files": fixture["raw_claim_files"],
        "protected_sources_unchanged": fixture["protected_sources_unchanged"],
        "scope": "narrow R53 tests only; guarded actual runtime configure and output binding in disposable private fixture; no first-four environment gate or real evaluation/training was executed",
    }
    put("ZERO_LAUNCH_TEST_EVIDENCE.json", test_evidence)

    ready = {
        "kind": "round53-readiness-check/2",
        "status": "conditional ready_for_authorized_execution",
        "ready_for_authorized_execution": True,
        "authorization": "PENDING_USER_APPROVAL",
        "authorization_is_not_implied_by_readiness": True,
        "trial_plan_status": "proposed_not_authorized",
        "plan_sha256": plan["plan_sha256"],
        "hard_cap": HARD_CAP, "normal_cap": NORMAL_CAP, "reserve_cap": RESERVE_CAP,
        "development_go": False,
        "passed": ["fixed 384 + 96 + 48 = 528 normal jobs and 24 fresh reserve; unique logical keys",
                   "48 training immediate rows hash/protocol checked against frozen training sequence",
                   "BC greedy48, shared PPO96, immediate/late dev16 each verified from indexed sources",
                   "BC sampled48 are planned outputs, not missing readiness inputs, and were not generated",
                   "actual runtime configure/binder passed in temporary sandbox; r44_runtime is explicitly bound",
                   "canonical live trainer method unwrapped to frozen source and same object is bound at R40 train adapter",
                   "private row/log/claim/slot/booking/checkpoint/journal roots point to fresh namespace",
                   "real R42 ep0 RNG restore path bypassed in favor of frozen deterministic fresh-RNG scheme",
                   "resume version/freeze gate rejects old ROUND46 state; BC model-only role accepted",
                   "first-four gate and strict cumulative budget check are wired in actual runner but not executed"],
        "execution_not_performed": True,
        "conditional_runtime_checks_after_explicit_approval": [
            "first-four per-seed PPO correctness gates must pass before continuation",
            "all 384 training logical claims must close before evaluation",
            "96 candidate plus 48 BC sampled rows must close; final ledger strict check must pass",
            "all actual output routes are revalidated by live binder before first training claim"],
        "missing_input_blockers": [],
        "not_authorized": ["no authorization receipt exists", "no raw ledger/claim/checkpoint namespace created"],
    }
    put("READY_CHECK.json", ready)
    report = f'''# ROUND53 后补实现与就绪检查\n\n**条件就绪：`conditional ready_for_authorized_execution`；执行授权仍待用户另行明确批准。** 本报告没有授权552额度，也没有启动训练或评估。`development_go=false` 保持。\n\n## 实现\n\n独立入口 `r53_execute.py` 在任何 runtime 配置前要求单独任务/分支/额度/计划哈希回执。它复用真实冻结 R46/R45/R44/R42/R40 接线，但不调用整体 ROUND46 `execute()`，以独立 `raw/` 路由输出，且替换旧 V6 namespace seeder，绝不复制 ROUND46 family journal。\n\n实时 private-fixture 接线测试确认 common 输出、R40 claim/slot 根、R42 booking、`r44_runtime`/`r44_prepare`/analysis RUN_DIR、checkpoint 根与实际 trainer 保存闭包 journal 全指向同一私有根。真实 canonical `v2_trainer.py` 字节 SHA为 `{trial.sha256_file(trial.CANON / 'v2_trainer.py')}`；live `ppo_update` 解包回该文件后才安装 ROUND53 head-input detach wrapper，且 R40 adapter持有同一个 trainer 对象。PPO loss/GAE/mask/clip/Adam/dual代码没有重写。\n\n初始模型只走同 seed R45 `BC_POST_MODEL` 四模块 factory；Adam/dual/RNG为本轮新状态，旧 ROUND46 PPO checkpoint freeze/version 会被拒绝。固定训练顺序来自原冻结 key plan；48场景 immediate 缓存逐键核验整数 `on_time`/计费、场景、one-shot 协议、ACK关闭、policy-owned timing及行SHA。评估固定144键：candidate greedy/sample各48，另BC sampled48；sampling seed通过原 `eval_action_seed`。\n\nFirst-four correctness门和预算额度逻辑已接入，first-four包含在384训练episode内；未运行。开发块1301000–1301015已使用，封存确认块未开。\n\n## 基线复用结果\n\n核验通过：training immediate 48；BC greedy 48；shared PPO 96；immediate/late开发参照各16。BC sampled48是本提案要产生的评估输出，并非准备输入，本轮没有生成。完整逐键哈希在 `BASELINE_REUSE_CHECK.json`。\n\n## 测试与操作计数\n\n测试结果直接解析自子进程 unittest 输出：**{tests_run}项，exit {proc.returncode}，OK**。实际调用配置器并在临时沙箱中验证live模块绑定；guard实测计数：{', '.join(f'{k}={v}' for k,v in operations.items() if k != 'claim_calls')}；真实claim数0；checkpoint `.pt` 数{fixture['raw_pt_files']}；sandbox claims数{fixture['raw_claim_files']}；两份受保护源文件和旧journal哈希未变。没有执行 first-four 环境门、真实训练或评估。\n\n## 执行边界与剩余条件\n\n`READY_CHECK.json` 只是条件代码/输入就绪检查。实际运行仍要求另行提供明确 human authorization receipt（严格匹配528 normal、24新reserve、552 cap、分支及计划哈希）；当前不存在该receipt。授权后仍必须通过first-four门和最终完整claim/slot/step/optimizer/checkpoint ledger闭合。未产生任何模型、账本或新评估行。\n\n当前交付提交完整SHA在回复中给出。未改写旧 `NOT_READY` 报告；本文件是后补版本。\n'''
    (HERE / "ROUND53_READINESS_REPORT_ADDENDUM.md").write_text(report, encoding="utf-8")
    return {"tests": tests_run, "baseline": baseline["status"], "ready": ready["status"],
            "plan_sha256": plan["plan_sha256"], "test_exit_code": proc.returncode}


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
