"""Run the narrow ROUND53 tests and derive readiness/report artifacts from outputs."""
from __future__ import annotations
import ast, json, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent
REPO = RUN_ROOT.parents[1]
sys.path.insert(0, str(HERE))
import r53_execute as trial
from r53_contracts import BRANCH, HARD_CAP, NORMAL_CAP, RESERVE_CAP, TASK_ID, TRAINING_VERSION
from r53_rng_audit import audit_registered_rng


def put(name: str, value) -> None:
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha_sources(paths):
    return [{"path": str(p), "sha256": trial.sha256_file(p)} for p in paths]


def source_excerpt(title: str, path: Path, start_marker: str, end_marker: str) -> str:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if start_marker in line]
    if len(starts) != 1:
        raise RuntimeError(f"source excerpt start marker not unique: {path} / {start_marker}")
    start = starts[0]
    ends = [i for i in range(start + 1, len(lines)) if end_marker in lines[i]]
    if not ends:
        raise RuntimeError(f"source excerpt end marker missing: {path} / {end_marker}")
    end = ends[0]
    selected = lines[start:end]
    if not selected or len(selected) > 120:
        raise RuntimeError(f"source excerpt must contain 1..120 lines, got {len(selected)}: {title}")
    return (f"## {title}\n\n`{path}`:{start + 1}-{end}\n\n```python\n"
            + "\n".join(selected) + "\n```\n")


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
    rng_identity = trial.read_json(RUN_ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json")["round44_registered_ep0_rng"]
    rng_audit = {str(seed): audit_registered_rng(
        Path(rng_identity[str(seed)]["path"]), expected_file_sha256=rng_identity[str(seed)]["sha256"])
        for seed in (11, 22, 33)}
    prior_ready_path = HERE / "ROUND53_READY_CHECK_V2_AS_DELIVERED.json"
    prior_report_path = HERE / "ROUND53_READINESS_REPORT_V2_AS_DELIVERED.md"
    if not prior_ready_path.exists():
        prior_ready = (HERE / "READY_CHECK.json").read_bytes()
        prior_doc = json.loads(prior_ready.decode("utf-8"))
        if prior_doc.get("kind") != "round53-readiness-check/2":
            raise RuntimeError("cannot preserve the delivered READY_CHECK /2 fact record")
        prior_ready_path.write_bytes(prior_ready)
    elif json.loads(prior_ready_path.read_text(encoding="utf-8")).get("kind") != "round53-readiness-check/2":
        raise RuntimeError("preserved READY_CHECK /2 record is malformed")
    if not prior_report_path.exists():
        prior_report_path.write_bytes((HERE / "ROUND53_READINESS_REPORT_ADDENDUM.md").read_bytes())
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
        trial.R43_DIR / "r42_run.py", trial.R44_DIR / "ROUND44_CHECKPOINT_INDEX.json",
        task_root / "ROUND46_PRELAUNCH_GATE.json",
        r40_root / "r40_common.py", r40_root / "r40_train.py", r40_root / "r40_run.py",
        HERE / "r53_execute.py", HERE / "r53_contracts.py",
        HERE / "round53_trial_adapter.py", HERE / "test_r53_execution.py",
        HERE / "r53_rng_audit.py",
        HERE / "test_round53_pure.py",
        HERE / "build_readiness_artifacts.py",
        task_root / "ROUND46_R2_V6_FROZEN_CONFIG.json",
        task_root / "ROUND46_R2_V6_KEY_PLAN.json",
        task_root / "ROUND46_R2_V6_INPUT_IDENTITY.json",
        task_root / "ROUND46_R2_V6_TRAINING_VERSION_MANIFEST.json",
        task_root / "v10_eval_final_close" / "ROUND46_R2_V10_ANALYSIS_INPUT_MANIFEST.json",
        trial.R44_DIR / "ROUND44_FROZEN_CONFIG.json",
        trial.R44_DIR / "ROUND44_BASELINE_REUSE.json",
        *[Path(rng_identity[str(seed)]["path"]) for seed in (11, 22, 33)],
        *[RUN_ROOT / f"ROUND46_TRAINING_s{seed}.json" for seed in (11, 22, 33)],
        *[Path(trial.read_json(RUN_ROOT / f"ROUND46_TRAINING_s{seed}.json")["ep128_checkpoint_path"])
          for seed in (11, 22, 33)],
        RUN_ROOT / "round52_critic_gradient_isolation_prototype" / "PROTOTYPE_CONTRACT.json",
        RUN_ROOT / "round52_critic_gradient_isolation_prototype" / "FORWARD_AND_GRADIENT_COMPARISON.json",
    ]
    sources = sha_sources(source_paths)
    fixture = test_record["runtime_fixture"]
    source_binding = {
        "kind": "round53-source-binding/3", "status": "actual_runtime_objects_and_frozen_rng_bound_in_guarded_private_fixture",
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
        "R42_ep0_rng_restored": True,
        "registered_ep0_rng_only": rng_audit,
        "rng_route": fixture.get("rng_route_identity"),
        "optimizer_and_checkpoint_guards": {"optimizer_step_targets": fixture.get("optimizer_guard_targets"),
                                             "checkpoint_save_target": fixture.get("checkpoint_guard_target")},
        "fresh_rng_scheme": plan["scientific_config_identity"]["fresh_rng_scheme"],
        "ppo_update_route": "actual runtime trainer object wrapped; __wrapped__ chain resolves to canonical source SHA",
        "training_math_rewritten": False,
    }
    put("SOURCE_BINDING.json", source_binding)

    operations = fixture["guards"]
    test_evidence = {
        "kind": "round53-zero-launch-test-evidence/3", "status": "passed",
        "test_count": tests_run, "test_exit_code": proc.returncode,
        "test_output_parsed_not_hardcoded": True,
        "test_command": test_record["command"], "tests_run": test_record["output"],
        "guard_observed_operation_counts": operations,
        "runtime_configured_in_private_temporary_fixture": True,
        "optimizer_guard_targets": fixture.get("optimizer_guard_targets"),
        "checkpoint_guard_target": fixture.get("checkpoint_guard_target"),
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
        "kind": "round53-readiness-check/3",
        "status": "preflight_ready_pending_authorization_and_first_four",
        "preflight_ready_for_authorized_first_four": True,
        "preflight_basis": {
            "frozen_source_identities": "passed",
            "indexed_baseline_reuse": "passed",
            "pure_and_guarded_runtime_tests": f"passed_{tests_run}_tests",
            "actual_runtime_binding_fixture": "passed",
            "registered_ep0_rng_route": "source_bound_to_frozen_R45_R42_R40_chain",
        },
        "prior_v2_conditional_readiness_withdrawn": True,
        "prior_v2_withdrawal_reason": "R53 v2 used an unregistered hash-seeded RNG route; its original record is preserved, and that route has been removed.",
        "first_four_dynamic_gate": "PENDING_NOT_EXECUTED",
        "full_plan_execution_ready": False,
        "full_plan_execution_dependency": "the authorized, quota-counted first-four dynamic gates must pass before continuation",
        "authorization": "PENDING_USER_APPROVAL",
        "authorization_is_not_implied_by_readiness": True,
        "trial_plan_status": "proposed_not_authorized",
        "plan_sha256": plan["plan_sha256"],
        "hard_cap": HARD_CAP, "normal_cap": NORMAL_CAP, "reserve_cap": RESERVE_CAP,
        "development_go": False,
        "prior_v2_status": "conditional ready_for_authorized_execution (historical artifact preserved; readiness withdrawn)",
        "post_fix_source_connection_status": "preflight_passed; first_four_dynamic_gate_pending",
        "prior_v2_record_preserved": str(prior_ready_path),
        "current_rng_contract": "registered R44 ep0 RNG-only restore after same-seed BC_POST model-only init via actual R42 train_seed",
        "passed": ["fixed 384 + 96 + 48 = 528 normal jobs and 24 fresh reserve; unique logical keys",
                   "48 training immediate rows hash/protocol checked against frozen training sequence",
                   "BC greedy48, shared PPO96, immediate/late dev16 each verified from indexed sources",
                   "BC sampled48 are planned outputs, not missing readiness inputs, and were not generated",
                   "actual runtime configure/binder passed in temporary sandbox; r44_runtime is explicitly bound",
                   "canonical live trainer method unwrapped to frozen source and same object is bound at R40 train adapter",
                   "private row/log/claim/slot/booking/checkpoint/journal roots point to fresh namespace",
                   "same-seed R45 BC model-only load is followed by the exact frozen R44 ep0 RNG-only restore in the upstream R42 train_seed",
                   "registered RNG file bytes and Python/NumPy/Torch component digests match the V6 and R44 source identities",
                   "all accepted PPO rows bind to the corresponding actual ep128 checkpoint parameter identity and evaluation mode",
                   "final strict claim and durable slot closure is order-independent; reserve remains optional and requires real failed attempt 1",
                   "guarded fixture blocks Optimizer/Adam/SGD.step and torch.save, not only the base optimizer method",
                   "resume version/freeze gate rejects old ROUND46 state; BC model-only role accepted",
                   "first-four gate and strict cumulative budget check are wired in actual runner but not executed"],
        "execution_not_performed": True,
        "runtime_checks_after_explicit_approval": [
            "enter the plan's quota-counted first-four episodes and per-seed PPO correctness gates",
            "continue only if every dynamic first-four gate returns passed",
            "all 384 training logical claims must close before evaluation",
            "96 candidate plus 48 BC sampled rows must close; final ledger strict check must pass",
            "all actual output routes are revalidated by live binder before first training claim"],
        "missing_input_blockers": [],
        "not_authorized": ["no authorization receipt exists", "no raw ledger/claim/checkpoint namespace created"],
    }
    put("READY_CHECK.json", ready)
    review = {
        "kind": "round53-final-connection-review/1", "status": "preflight_ready_pending_authorization_and_first_four",
        "previous_head": "88ce17dc59cdc72e13274c00056e6d17a6fe68f5",
        "rng_contract": {"finding": "old R53 hash-derived seeding was an uncontracted second factor; removed",
                          "status": "fixed_and_source_bound",
                          "registered_sources": plan["registered_r44_ep0_rng_only"],
                          "component_state_hashes": rng_audit,
                          "route_identity": fixture.get("rng_route_identity"),
                          "call_order": ["R45 same-seed BC_POST four-module model-only init",
                                         "R42 train_seed loads registered R44 ep0 payload and restores rng only after fresh_controller",
                                         "R40 fresh Adam(empty), dual lambda 1, then first training episode"],
                          "BC_optimizer_or_rng_restored": False,
                          "R44_model_or_optimizer_restored": False,
                          "same_round53_checkpoint_resume_only": True},
        "strict_claims": {"status": "fixed_and_tested", "reserve_optional": True,
                          "reserve_requires_persisted_same_key_failed_technical_attempt1": True,
                          "original_status_self_report_is_not_authority": True,
                          "slot_claims_cross_closed": True,
                          "nonfatal_snapshot_cannot_replace_direct_strict_closure": True},
        "optimizer_guard": {"status": "guarded", "step_methods": fixture.get("optimizer_guard_targets"),
                            "checkpoint_write_guard": fixture.get("checkpoint_guard_target")},
        "baseline_identity": {"status": baseline["status"], "BC_greedy_rows": baseline["bc_greedy"]["rows"],
                              "PPO_rows": baseline["shared_ppo"]["verified_source_rows"],
                              "PPO_ep128_models": baseline["shared_ppo"]["ep128_models"],
                              "BC_sampled_generated": False},
        "r53_execute_functions": [node.name for node in ast.parse(
            (HERE / "r53_execute.py").read_text(encoding="utf-8")).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))],
        "readiness": {"status": ready["status"],
                       "preflight_ready_for_authorized_first_four": ready["preflight_ready_for_authorized_first_four"],
                       "first_four_dynamic_gate": "PENDING_NOT_EXECUTED",
                       "full_plan_execution_ready": False,
                       "full_plan_execution_dependency": ready["full_plan_execution_dependency"],
                       "prior_v2_conditional_readiness_withdrawn": True,
                       "authorization": "PENDING_USER_APPROVAL", "development_go": False,
                       "actual_first_four_gate_run": False, "actual_environment_or_training_run": False,
                       "actual_evaluation_run": False},
        "seven_zero_operation_counts": {k: operations[k] for k in (
            "environment_construction", "simulation", "new_scenes_or_flows", "new_evaluation",
            "real_training_updates", "synthetic_optimizer_steps", "checkpoint_writes")},
        "real_claims_created": fixture["raw_claim_files"],
        "preserved_delivered_v2_records": {
            "ready_check_worktree_copy_sha256": trial.sha256_file(prior_ready_path),
            "report_worktree_copy_sha256": trial.sha256_file(prior_report_path),
            "copied_before_current_v3_overwrite": True},
        "protected_files_current_sha256": {
            str(path): trial.sha256_file(path) for path in (
                RUN_ROOT / "ROUND46_CURRENT_TASK.json",
                RUN_ROOT / "ROUND46_R2_V6_STOP_STATE.json",
                RUN_ROOT / "runtime" / "checkpoint_writes.jsonl",
                trial.R45_DIR / "runtime" / "checkpoint_writes.jsonl") if path.is_file()},
    }
    put("ROUND53_FINAL_CONNECTION_REVIEW.json", review)
    function_list = review["r53_execute_functions"]
    report = f'''# ROUND53 后补实现与连接复核\n\n**RNG 更正：R53 v2 的条件就绪状态已撤回并保留原件；错误 hash-seeded 路径已从当前源码移除。**\n\n- `preflight_ready_for_authorized_first_four=true`：冻结源身份、索引基线、{tests_run}项纯/守卫测试及真实绑定fixture均通过，足以在明确授权后进入首四集动态门。\n- `first_four_dynamic_gate=PENDING_NOT_EXECUTED`：首四集属于固定384集计划并计入配额；本轮没有运行，也不把未运行当作通过。\n- `full_plan_execution_ready=false`：仅当获批后的首四集动态门全部通过才可继续；失败则停止。\n- `authorization=PENDING_USER_APPROVAL`；`development_go=false`。\n\n## RNG合同复核与修复\n\nROUND46训练清单把唯一科学变化限定为 critic gradient route；它明示 `initializer=ROUND45 BC_POST_MODEL four modules, model-only, same seed`、fresh Adam、dual λ=1、`r44_ep0_rng_only=true`。冻结输入身份/ROUND44配置注册的ep0文件是 s11 `70fe64c18f61b33d17b6aa528cf830b43919a89b22210b11d7edb732bd75f1fd`、s22 `96a09f2f4796069a10f77e5472a4fa8c09ddbf1407e44a7102c022c43c5790c5`、s33 `9030be21c945a490802692569e0fb5b4608602394e43233666ff83adcc2cfe8d`。三者文件SHA均与输入身份和ROUND44冻结登记一致；Python/NumPy/Torch状态分量摘要详见 `ROUND53_FINAL_CONNECTION_REVIEW.json`。\n\n冻结真实次序为：R45 `_call_training_seed` 临时装入同seed BC四模块；R42 `train_seed` 仅读取注册ep0 payload的 `rng`；`_registered_ep0_rng_after_init` 在同seed `fresh_controller`（含BC参数覆盖）返回后调用 `trainer.restore_rng(rng_state)`；之后R40 fresh adapter建立空Adam、λ=1并写本轮ep0状态/开始首集。R42 ep0的模型权重和optimizer没有恢复。ROUND53原先在R46→R45→R42路径外自行SHA-seed Python/NumPy/Torch并直调R40，构成额外随机过程因子；不能证明与注册状态逐字节相同。已删除该支路，现binder要求真实R46 factory-scope→R45 `_call_training_seed`→R42 `train_seed` 源路径和三seed冻结RNG路径/hash精确相符。训练续跑只允许私有ROUND53同版本checkpoint；BC仅模型初始化。\n\n## Claim/预算与基线连接\n\n严格claim检查改为两遍、与目录枚举顺序无关：reserve attempt 2 必须对应同key真实持久化的 normal attempt 1=`failed_technical`，不信任自报 `original_status`；normal必须精确闭合528，reserve上限24而非必须耗尽。新增slot文件与claim逐项身份交叉闭合，strict final门直接执行，不被nonfatal派生快照warning吞掉。纯反例覆盖少用reserve、伪报失败、错slot、坏strict关闭。\n\nBC 48行已核BC_ONLY_R2角色、真实BC_POST checkpoint及参数SHA（包含 `bc_parameter_sha256`）、scene/mode/steps、world/config与行为指纹、成本和冻结通信协议。PPO 96行逐行核NEW、seed/scene/mode、ep128参数SHA；并核三份真实ep128 checkpoint摘要、SHA及训练summary。成本校验涵盖整数spent/on_time、预算、B0、1000步、one-shot/first-report、reward timing、ACK关闭及policy-owned timing。开发参照与训练immediate缓存按索引复用，未做全历史重扫，也未重放forward或评估。\n\n## 当前 `r53_execute.py` 函数清单\n\n`{', '.join(function_list)}`。关键实际函数分段源码见 `ROUND53_ACTUAL_PATH_EXCERPTS.md`，每段不超过120行。\n\n## 测试范围及未覆盖项\n\n窄测试直接报告 **{tests_run}项，exit {proc.returncode}，OK**。隔离fixture实接了实际runtime配置器/输出绑定，但把真正train/env/eval/claim/save入口全部拒绝；optimizer守卫覆盖 `Optimizer.step`、`Adam.step`、`SGD.step`，checkpoint写守卫覆盖 `torch.save` 和trainer.save入口；实际观测七项操作计数均为0。实际first-four任务门、任何环境、模型更新、评估和最终真实ledger没有执行；获批后仍须验证这些动态性质。预检就绪只表示获批后可进入计划内、计入配额的逐seed首四集动态门；它不表示动态门已通过、全计划可继续或科学成功。\n\n## 交付\n\n`ROUND53_FINAL_CONNECTION_REVIEW.json`给出风险项逐条状态和RNG字节摘要；`ROUND53_ACTUAL_PATH_EXCERPTS.md`按每段≤120源码行列出未来训练/评估/RNG/init/save/load/budget关键接线。完整代码、纯反例和基线证据保持在本目录。旧 `/2`原件保留，只更新当前 `READY_CHECK.json`为`/3`。\n'''
    (HERE / "ROUND53_READINESS_REPORT_ADDENDUM.md").write_text(report, encoding="utf-8")
    r40_root = REPO.parent / "zhenghe-round40" / "runs" / "comm_first_report_training_ablation_20261003"
    r42_run = trial.R43_DIR / "r42_run.py"
    r45_run = trial.R45_DIR / "r45_r2_execute_r3.py"
    r46_run = trial.SOURCE_RUNNER
    excerpts = [
        source_excerpt("R53 frozen plan and registered RNG identity", HERE / "r53_execute.py",
                       "def plan_from_frozen_sources", "def verify_baseline_reuse"),
        source_excerpt("R53 live runtime binding before training-route check", HERE / "r53_execute.py",
                       "def bind_live_runtime", "    # Preserve the frozen R46/R45/R42 fresh-start route."),
        source_excerpt("R53 exact RNG route and upstream function identity guards", HERE / "r53_execute.py",
                       "    # Preserve the frozen R46/R45/R42 fresh-start route.",
                       "    # A pre-claim guard validates every logical key"),
        source_excerpt("R53 planned claims and reserve precondition", HERE / "r53_execute.py",
                       "    # A pre-claim guard validates every logical key",
                       "    # The atomic claim is authoritative."),
        source_excerpt("R53 strict closure and immutable trainer route", HERE / "r53_execute.py",
                       "    # Route the exact physical trainer object", "    return {\"bindings\""),
        source_excerpt("R53 authorized execution, train/evaluation order, final ledger", HERE / "r53_execute.py",
                       "def execute_authorized", "if __name__ == \"__main__\":"),
        source_excerpt("R42 post-controller RNG restore helper", r42_run,
                       "def _registered_ep0_rng_after_init", "def train_seed("),
        source_excerpt("R42 actual train_seed ep0 payload verification and restore", r42_run,
                       "def train_seed(training_seed", "def load_eval_model("),
        source_excerpt("R42 fixed evaluation model loading", r42_run,
                       "def load_eval_model(", "def torch_load_cpu("),
        source_excerpt("R42 evaluation cell path", r42_run,
                       "def run_model_eval(", "def run_rule("),
        source_excerpt("R45 BC model-only injection before actual R42 train_seed", r45_run,
                       "def _call_training_seed(", "def _fixed_gate_baseline("),
        source_excerpt("R46 first-four/cross-seed gates and fixed continuation", r46_run,
                       "def _run_first_four_and_all_gates(", "def _load_round45_bc_comparison("),
        source_excerpt("R40 checkpoint save milestone", r40_root / "r40_train.py",
                       "def _save_milestone(", "def _batch_record_path("),
        source_excerpt("R40 same-version resume source selection", r40_root / "r40_train.py",
                       "def _latest_resume_source(", "def _save_progress("),
        source_excerpt("R40 fresh BC controller/Adam/dual initialization and first rollout", r40_root / "r40_train.py",
                       "def train_seed(training_seed", "    from r40_run import run_training_episode"),
        source_excerpt("R40 PPO update and checkpoint commit path", r40_root / "r40_train.py",
                       "        update_report = trainer.ppo_update", "__all__ ="),
        source_excerpt("R53 order-independent strict claims", HERE / "r53_contracts.py",
                       "def validate_claims(", "def validate_slot_records("),
        source_excerpt("R53 durable quota slot cross-closure", HERE / "r53_contracts.py",
                       "def validate_slot_records(", "def derived_snapshot("),
    ]
    (HERE / "ROUND53_ACTUAL_PATH_EXCERPTS.md").write_text(
        "# ROUND53 actual future-path source excerpts\n\n"
        "Generated from current source; every code segment is limited to at most 120 lines. "
        "These are actual wired functions, not pseudocode.\n\n" + "\n".join(excerpts), encoding="utf-8")
    return {"tests": tests_run, "baseline": baseline["status"], "ready": ready["status"],
            "plan_sha256": plan["plan_sha256"], "test_exit_code": proc.returncode}


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
