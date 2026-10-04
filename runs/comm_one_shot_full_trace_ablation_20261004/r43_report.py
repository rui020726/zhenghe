"""Build ROUND43 report, correction note, status, and source/raw inventories."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from r42_run import ROUND42_OUTPUT, ROUND40_ROOT, ROUND41_OUTPUT, ROUND42_ROOT, RUN_DIR


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()
    temporary.replace(path)


def _fmt_bool(value: bool) -> str:
    return "是" if value else "否"


def _fmt_group(group: dict[str, Any]) -> str:
    return (f"按时任务 {group['on_time_total']}，计费 {group['spent_bytes_total']:,} B，"
            f"首报 SEND {group['send_now_choices_total']} / SKIP "
            f"{group['skip_episode_choices_total']}（自由选择 "
            f"{group['free_choices_total']}）")


def write_final_report(*, frozen: dict[str, Any], analysis: dict[str, Any],
                       ledger: dict[str, Any], progress: dict[str, Any]) -> dict[str, Any]:
    paired = analysis["paired_results"]
    verdict = analysis["machine_verdict"]
    curves = analysis["training_curves"]
    index = analysis["checkpoint_index"]
    preflight = _read(RUN_DIR / "ROUND43_PREFLIGHT.json")
    baseline = _read(RUN_DIR / "ROUND43_BASELINE_REUSE.json")
    ep0 = _read(RUN_DIR / "ROUND43_EP0_INITIALIZATION.json")
    resume_bridge = _read(RUN_DIR / "runtime" / "ROUND43_RESUME_BRIDGE.json")
    recovery_gate = _read(RUN_DIR / "ROUND43_RESUME_RECOVERY_GATE.json")
    state_reconciliation = _read(
        RUN_DIR / "ROUND43_SEED11_BATCH0_STATE_RECONCILIATION.json")
    ledger_reconciliation = _read(RUN_DIR / "ROUND43_LEDGER_RECONCILIATION.json")
    failed_resume = _read(
        RUN_DIR / "failures" / "ROUND43_RUNCALL_RESUME_FAILURE_FREEZE1.json")
    counters = ledger["counts"]
    branch = _git(ROUND42_ROOT, "branch", "--show-current")
    source_freeze_head = _git(ROUND42_ROOT, "rev-parse", "HEAD")

    source_inventory = {
        "kind": "round43-source-and-baseline-inventory/1",
        "branch": branch,
        "source_freeze_head": source_freeze_head,
        "execution_freeze": frozen["execution_identity"],
        "training_version": frozen["round42"]["training_version"],
        "source_training_version": frozen["round42"]["source_training_version"],
        "execution_basis": {
            "path": str(RUN_DIR / "EXECUTION_BASIS.md"),
            "sha256": _sha256(RUN_DIR / "EXECUTION_BASIS.md"),
        },
        "configuration_diff": {
            "path": str(RUN_DIR / "ROUND43_CONFIGURATION_DIFF.json"),
            "sha256": _sha256(RUN_DIR / "ROUND43_CONFIGURATION_DIFF.json"),
        },
        "prelaunch_gates": {
            "path": str(RUN_DIR / "ROUND43_PREFLIGHT.json"),
            "sha256": _sha256(RUN_DIR / "ROUND43_PREFLIGHT.json"),
            "runtime_environment": preflight["runtime_environment"],
        },
        "execution_recovery": {
            "initial_freeze_head": resume_bridge["prior_prelaunch_head"],
            "recovery_freeze_head": source_freeze_head,
            "execution_code_version": frozen["execution_identity"][
                "execution_code_version"],
            "training_semantics_version_unchanged": (
                resume_bridge["training_version"] == frozen["round42"]["training_version"]),
            "resume_bridge": {
                "path": str(RUN_DIR / "runtime" / "ROUND43_RESUME_BRIDGE.json"),
                "sha256": _sha256(RUN_DIR / "runtime" / "ROUND43_RESUME_BRIDGE.json"),
            },
            "recovery_gate": {
                "path": str(RUN_DIR / "ROUND43_RESUME_RECOVERY_GATE.json"),
                "sha256": _sha256(RUN_DIR / "ROUND43_RESUME_RECOVERY_GATE.json"),
            },
        },
        "frozen_search_allocation_inputs": preflight[
            "frozen_search_allocation_inputs"],
        "registered_ep0_inputs": ep0["by_seed"],
        "baseline_reuse_manifest": {
            "path": str(RUN_DIR / "ROUND43_BASELINE_REUSE.json"),
            "sha256": _sha256(RUN_DIR / "ROUND43_BASELINE_REUSE.json"),
            "reused_counts": {key: value["count"] for key, value in baseline["reused"].items()},
        },
        "source_code_sha256": frozen["execution_identity"]["source_code_sha256"],
        "inherited_dependency_sha256": frozen["execution_identity"][
            "inherited_dependency_sha256"],
        "round40_code_sha256": frozen["execution_identity"]["round40_code_sha256"],
        "round40_delivery_head": frozen["execution_identity"]["round40_delivery_head"],
        "round41_delivery_head": frozen["execution_identity"]["round41_delivery_head"],
        "canonical_read_only_input_head": frozen["execution_identity"][
            "canonical_read_only_input_head"],
        "round42_originals_unchanged": True,
    }
    _atomic_json(RUN_DIR / "ROUND43_SOURCE_INVENTORY.json", source_inventory)

    raw_roots = ("rows", "logs", "checkpoints", "launch_claims", "launch_slots",
                 "training_progress", "runtime", "failures", "rejected")
    raw_files: list[dict[str, Any]] = []
    for name in raw_roots:
        folder = RUN_DIR / name
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            if path.relative_to(RUN_DIR).as_posix() == "runtime/ROUND43_EXECUTION_PROGRESS.json":
                continue
            raw_files.append({
                "relative_path": path.relative_to(RUN_DIR).as_posix(),
                "path": str(path), "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    raw_manifest = {
        "kind": "round43-raw-artifact-manifest/1",
        "raw_artifacts_remain_in_local_run_directory": True,
        "raw_files_count": len(raw_files),
        "files": raw_files,
    }
    _atomic_json(RUN_DIR / "ROUND43_RAW_ARTIFACT_MANIFEST.json", raw_manifest)

    version_manifest = {
        "kind": "training-version-manifest/1",
        "previous_version": frozen["round42"]["source_training_version"],
        "previous_checkpoint_policy": {
            "resume": "rejected by the actual ROUND43 restore entry before controller, optimizer, or RNG restore",
            "read_only": "allowed with explicit ROUND42 identity",
        },
        "new_version": frozen["round42"]["training_version"],
        "new_checkpoint_policy": {
            "resume": "accepted only with the exact ROUND43 version and freeze identity",
            "initialization": "fresh same-seed controller and empty Adam; ROUND42 ep0 supplies only the registered RNG state",
        },
        "only_research_variable": {"gae_lambda": {"old": 0.95, "new": 1.0}},
        "unchanged": {"gamma": 1.0, "lambda_dual_initial": 1.0,
                      "eta_lambda": 0.05, "delta": 0.0, "b0_bytes": 10000,
                      "comm_action_semantics": "one-shot-send-skip/1",
                      "reward_timing_version": frozen["round42"]["reward_timing_version"]},
        "round42_report_correction": {
            "original_text_claimed": "two greedy NEW seeds were all-SKIP",
            "corrected_from_original_table_and_rows": "all three greedy NEW seeds were all-SKIP",
            "original_round42_report_edited": False,
        },
    }
    _atomic_json(RUN_DIR / "TRAINING_VERSION_MANIFEST.json", version_manifest)

    resume_reconciliation = {
        "kind": "round43-exact-resume-reconciliation/1",
        "status": "passed",
        "initial_freeze_head": resume_bridge["prior_prelaunch_head"],
        "recovery_freeze_head": source_freeze_head,
        "execution_code_version": frozen["execution_identity"][
            "execution_code_version"],
        "training_semantics_version": frozen["round42"]["training_version"],
        "resume_cursor": resume_bridge["resume_cursor"],
        "previously_completed_normal_claims": resume_bridge["normal_plan"][
            "used_before_resume"],
        "remaining_normal_claims_at_resume": resume_bridge["normal_plan"]["remaining"],
        "technical_reserve_max": resume_bridge["normal_plan"]["technical_reserve"],
        "hard_cap": resume_bridge["normal_plan"]["hard_cap"],
        "initial_runner_failure": {
            "kind": failed_resume["kind"],
            "status": failed_resume["status"],
            "exception": failed_resume["exception"],
            "claim_delta": failed_resume["accounting_at_stop"][
                "environment_claims_delta_during_failed_resume"],
            "environment_constructor_delta": failed_resume["accounting_at_stop"][
                "environment_constructors_delta"],
            "run_call_delta": failed_resume["accounting_at_stop"][
                "environment_run_calls_delta"],
            "optimizer_update_delta": failed_resume["accounting_at_stop"][
                "PPO_or_optimizer_call_delta"],
            "source_path": str(
                RUN_DIR / "failures" / "ROUND43_RUNCALL_RESUME_FAILURE_FREEZE1.json"),
            "source_sha256": _sha256(
                RUN_DIR / "failures" / "ROUND43_RUNCALL_RESUME_FAILURE_FREEZE1.json"),
            "classification": "runner defect before a new launch claim; not a zero-result episode",
        },
        "old_ledger_snapshot": {
            "normal_used": ledger_reconciliation["original_persisted_ledger"][
                "counts"]["normal_used"],
            "ppo_update_calls": ledger_reconciliation[
                "original_persisted_ledger"]["counts"]["ppo_update_calls"],
            "optimizer_step_calls": ledger_reconciliation[
                "original_persisted_ledger"]["counts"]["optimizer_step_calls"],
            "checkpoint_save_calls": ledger_reconciliation[
                "original_persisted_ledger"]["counts"]["checkpoint_write_calls"],
            "preserved_without_rewrite": True,
        },
        "reconciled_at_pause": ledger_reconciliation[
            "corrected_cumulative_snapshot"],
        "evidence": {
            "seed11_batch0_state_reconciliation": {
                "path": str(RUN_DIR / "ROUND43_SEED11_BATCH0_STATE_RECONCILIATION.json"),
                "sha256": _sha256(
                    RUN_DIR / "ROUND43_SEED11_BATCH0_STATE_RECONCILIATION.json"),
                "status": state_reconciliation["status"],
            },
            "ledger_reconciliation": {
                "path": str(RUN_DIR / "ROUND43_LEDGER_RECONCILIATION.json"),
                "sha256": _sha256(RUN_DIR / "ROUND43_LEDGER_RECONCILIATION.json"),
                "status": ledger_reconciliation["status"],
            },
            "resume_recovery_gate": {
                "path": str(RUN_DIR / "ROUND43_RESUME_RECOVERY_GATE.json"),
                "sha256": _sha256(RUN_DIR / "ROUND43_RESUME_RECOVERY_GATE.json"),
                "status": recovery_gate["status"],
            },
        },
        "first_batch_replayed": False,
        "optimizer_step_replayed": False,
        "round42_trained_checkpoint_used_as_resume_state": False,
    }
    if (resume_reconciliation["initial_runner_failure"]["claim_delta"] != 0
            or resume_reconciliation["initial_runner_failure"][
                "environment_constructor_delta"] != 0
            or resume_reconciliation["initial_runner_failure"]["run_call_delta"] != 0
            or resume_reconciliation["initial_runner_failure"][
                "optimizer_update_delta"] != 0
            or resume_reconciliation["remaining_normal_claims_at_resume"] != 476
            or not recovery_gate["status"] == "passed"
            or not state_reconciliation["status"] == "passed"
            or not ledger_reconciliation["status"] == "passed"):
        raise RuntimeError("ROUND43 resume reconciliation did not close")
    _atomic_json(RUN_DIR / "ROUND43_RESUME_RECONCILIATION.json",
                 resume_reconciliation)

    old_eval = baseline["reused"]["old_round42_new_ep128_evaluations"]
    correction = {
        "kind": "round42-report-correction/1",
        "source_round42_report": str(ROUND42_OUTPUT / "ROUND42_RESUME_FINAL_RESULT.json"),
        "source_round42_paired_results": str(ROUND42_OUTPUT / "ROUND42_PAIRED_RESULTS.json"),
        "registered_correction": version_manifest["round42_report_correction"],
        "supporting_new_ep128_greedy_rows": {},
    }
    for seed in (11, 22, 33):
        selected = [item for item in old_eval["rows"]
                    if int(item["row"]["training_seed"]) == seed
                    and item["row"]["eval_mode"] == "greedy"]
        correction["supporting_new_ep128_greedy_rows"][str(seed)] = {
            "count": len(selected),
            "sum_send_now_choices": sum(int(item["row"]["collector_summary"][
                "one_shot_counts"]["send_now_choices"]) for item in selected),
            "sum_skip_episode_choices": sum(int(item["row"]["collector_summary"][
                "one_shot_counts"]["skip_episode_choices"]) for item in selected),
            "row_path_sha256": [{"path": item["path"], "sha256": item["sha256"]}
                                for item in selected],
        }
        if (len(selected) != 16
                or correction["supporting_new_ep128_greedy_rows"][str(seed)][
                    "sum_send_now_choices"] != 0):
            raise RuntimeError(f"ROUND42 all-SKIP correction not proven for seed {seed}")
    _atomic_json(RUN_DIR / "ROUND42_REPORT_CORRECTION.json", correction)

    # Include stable, no-self-reference manifest identity in the source/report inventory.
    raw_manifest_sha = _sha256(RUN_DIR / "ROUND43_RAW_ARTIFACT_MANIFEST.json")
    source_inventory["raw_artifact_manifest"] = {
        "path": str(RUN_DIR / "ROUND43_RAW_ARTIFACT_MANIFEST.json"),
        "sha256": raw_manifest_sha, "raw_files_count": len(raw_files),
    }
    _atomic_json(RUN_DIR / "ROUND43_SOURCE_INVENTORY.json", source_inventory)

    greedy_lines = []
    for seed in (11, 22, 33):
        item = verdict["greedy_primary"]["by_seed"][str(seed)]
        greedy_lines.append(
            f"- seed {seed}：目标集合与 immediate 完全一致={_fmt_bool(item['exact_immediate_on_time_task_set_by_scene'])}；"
            f"NEW {item['new_on_time_total']} 个按时任务 / {item['new_spent_bytes_total']:,} B，"
            f"late {item['late_on_time_total']} 个 / {item['late_spent_bytes_total']:,} B；"
            f"低于 late={_fmt_bool(item['cost_below_late_discovery_gate'])}。"
        )
    sampled_lines = []
    for seed in (11, 22, 33):
        newg = paired["group_summaries"][f"s{seed}_sampled_NEW_round43_ep128"]
        oldg = paired["group_summaries"][f"s{seed}_sampled_OLD_round42_new_ep128"]
        sampled_lines.append(
            f"- seed {seed}：NEW {_fmt_group(newg)}；OLD {_fmt_group(oldg)}。"
        )

    ledger_summary = {
        key: counters.get(key) for key in (
            "normal_used", "reserve_used", "total_used", "environment_constructor_calls",
            "joint_run_constructed_calls", "run_invocation_calls", "joint_run_method_calls",
            "run_return_calls", "run_failure_calls", "completed_steps",
            "completed_logical_jobs", "technical_failed_claims", "retry_claims",
            "ppo_update_calls", "ppo_update_journal_entries", "optimizer_step_calls",
            "optimizer_step_journal_entries", "checkpoint_write_calls", "hard_cap")
    }
    training_state_lines = []
    for seed in (11, 22, 33):
        values = index["by_seed"][str(seed)]["128"]["training_state"]
        training_state_lines.append(
            f"- seed {seed}：episodes={values['episodes_done']}、batches={values['updates']}、"
            f"optimizer steps={values['optimizer_steps']}、λ_dual={values['lambda']:.6f}、"
            f"ep128 参数 SHA256=`{index['by_seed'][str(seed)]['128']['parameter_sha256']}`。"
        )

    report = f"""# ROUND43 报告：一次性首报选择的完整 GAE trace 消融

## 结论

是否完成：是。三 seed 固定训练和固定评估均完成；总计 {counters['normal_used']} 次正常启动、{counters['reserve_used']} 次技术储备，硬上限 {counters['hard_cap']}。

任务保持是否改善：{"是" if verdict['greedy_primary']['all_three_seeds_exactly_preserve_immediate_task_set'] else "否"}。三个 greedy seed 均保持 immediate 的开发块按时任务身份集合：{_fmt_bool(verdict['greedy_primary']['all_three_seeds_exactly_preserve_immediate_task_set'])}。

是否优于强规则：{"开发块观察到联合改善" if verdict['greedy_primary']['joint_improvement_by_registered_development_rule'] else "未观察到满足预登记联合改善条件的结果"}。条件为三个 greedy seed 各自保持 immediate 按时任务集合，且计费低于 late_discovery_gate。

跨 seed 是否一致：{_fmt_bool(verdict['cross_seed_consistency']['greedy_joint_criterion_all_seeds'])}。判定只针对已用于开发的16个场景，不代表正式非劣、总体零损失或泛化。

## 研究变量与边界

唯一研究变量为共享 `gae_lambda` 从 0.95 改为 1.0；`gamma=1.0`、`lambda_dual` 初值 1.0、`eta_lambda=0.05`、`delta=0`、`B0=10000 B` 均保持冻结。task 与 positive-cost GAE 共用新 trace 系数，组合仍为 `lambda_dual*A_task - A_positive_cost`。这是一项方法消融，不是修复旧 GAE 数学错误；GAE=1 同时改变 `A+V` critic target，且可能增大方差。本结果不能证明 ROUND42 的学习失败只由信用衰减导致。

## Greedy 主结果

{chr(10).join(greedy_lines)}

联合判定：`{verdict['scientific_result']}`。

## Sampled 辅助结果

{chr(10).join(sampled_lines)}

完整逐 seed/mode 的 NEW−OLD、NEW−immediate、NEW−late、NEW−always_hold 配对结果和逐目标恢复/丢失列表见 `ROUND43_PAIRED_RESULTS.json`。绝对任务、载荷尝试字节、首报 SEND/SKIP、交付和丢包统计也在其中。

## 训练和路径完整性

{chr(10).join(training_state_lines)}

- 训练曲线和96个批记录：`ROUND43_TRAINING_CURVES.json`；每批含 `g_hat`、`lambda_dual` 更新、PPO指标、梯度摘要、场景和episode行身份。
- ep0/64/128 checkpoint 字节哈希、参数及模块摘要、optimizer/RNG摘要：`ROUND43_CHECKPOINT_INDEX.json`。
- 三个 seed 各32个批的参数链连续；每个 ep128 参数摘要与最后一批更新后状态核对。
- 首批真实 PPO 接线门禁：`ROUND43_FIRST_BATCH_GATE_SEED11.json`、`ROUND43_FIRST_BATCH_GATE_SEED22.json`、`ROUND43_FIRST_BATCH_GATE_SEED33.json`。
- 执行账本和逐 claim 计数：`runtime/ROUND43_EXECUTION_LEDGER.json`；完成逻辑任务={counters['completed_logical_jobs']}，真实完成步={counters['completed_steps']}。

启动与更新对账：{json.dumps(ledger_summary, ensure_ascii=False)}。

## 恢复与计数勘误

首次冻结 `{resume_bridge['prior_prelaunch_head']}` 下，seed11 已完成 episode0–3 并提交 batch0。继续运行时在下一次 episode claim 前，冻结 ROUND40 训练入口因 `parameter_identity` 局部导入作用域错误抛出 `UnboundLocalError`。这次是运行器缺陷：新增 claim、环境构造、run 调用和优化更新均为0，不作为零任务/零成本 episode，也不计入480个正常任务。

恢复门禁复核了 after_b0 与 rolling 状态、完整模型与 Adam 状态、12个 Adam step、λ与历史、训练版本、游标、RNG 和剩余场景顺序。ROUND42 验证过的适配器只把 `parameter_identity` 导入移到训练入口作用域；训练语义版本保持 `{resume_bridge['training_version']}`。恢复从 episode4、update1、Adam step12、λ={resume_bridge['resume_cursor']['lambda_dual']:.4f} 开始，首批没有重放。恢复冻结 `{source_freeze_head}` 使用执行代码身份 `{frozen['execution_identity']['execution_code_version']}`。

停机时原始 ledger 的快照把 PPO/Adam 计数记为0、checkpoint 保存调用记为2。该旧快照作为原件保留；批提交、逐调用日志及 checkpoint 状态核验出的累计值分别为 PPO 1、Adam 12、checkpoint 保存调用4。checkpoint 保存调用依据调用日志，不由文件数量推算。精确证据和哈希见 `ROUND43_RESUME_RECONCILIATION.json`、`ROUND43_LEDGER_RECONCILIATION.json` 与 `ROUND43_SEED11_BATCH0_STATE_RECONCILIATION.json`。

## 初值与随机状态

NEW 均由同 seed fresh controller、空 Adam 和 `lambda_dual=1.0` 开始。每 seed 的模型参数哈希与 ROUND42 对应 ep0 相同；optimizer state、cursor=0、updates=0、optimizer_steps=0、lambda 初值相符。ROUND42 ep0 checkpoint 只读提供登记的 RNG，不提供模型权重或 optimizer 状态。

门禁发现当前进程 fresh-controller 构造后的 Python/NumPy 全局状态在部分 seed 上与旧进程捕获值不同；runner 在 controller 构造之后、首次 rollout 之前，通过实际 fresh-controller 接线恢复登记的 ep0 RNG。恢复后的 Python、NumPy、Torch 状态均逐项与相应 ep0 checkpoint 一致。环境每 episode 按原 `scenario_seed` 构造并 reseed 其自身状态；通信动作仍使用原逐集 SHA256 action seed；PPO minibatch 顺序仍用 `DATA_ORDER_SEED:update_index` 的独立 `random.Random`；场景顺序 SHA256 与 ROUND42 一致。这个核验支持相同的起始 RNG 状态，不代表三 seed 轨迹在参数更新后仍逐位配对。

细项见 `ROUND43_EP0_INITIALIZATION.json`、`ROUND43_PREFLIGHT.json` 和 `ROUND43_CONFIGURATION_DIFF.json`。

## 版本与旧结果勘误

ROUND43 登记新训练版本 `{frozen['round42']['training_version']}`。实际恢复入口拒绝旧 ROUND42 已训练 checkpoint 作为 ROUND43 训练恢复源；旧模型仍允许带身份信息只读加载。ROUND42 的原始报告正文把 greedy NEW 全 SKIP 写成“两 seed”；逐行原件和配对表确认是三个 seed。勘误保存在 `ROUND42_REPORT_CORRECTION.json`，未改写 ROUND42 原报告、模型、tape 或日志。

## 基线复用与数据范围

本轮复用 64 个 immediate 行、32 个 late/always_hold 规则行，以及 ROUND42 NEW ep128 的96个 OLD评估行；三类均已校验原 manifest 哈希和对应身份。ROUND43 新增384个训练episode、96个 ep128 NEW评估。参考源、checkpoint及文件哈希见 `ROUND43_BASELINE_REUSE.json`、`ROUND43_SOURCE_INVENTORY.json` 与 `ROUND43_RAW_ARTIFACT_MANIFEST.json`。原始权重、tape、环境日志保留在本地 run 目录，没有写入报告仓库内容之外的文件。

## 预启动与复现身份

- Branch：`{branch}`
- 预启动源冻结 HEAD：`{source_freeze_head}`
- Python：`{frozen['execution_identity']['python_executable']}`；实验设备：CPU
- Runtime：Python {preflight['runtime_environment']['python_version']}、NumPy {preflight['runtime_environment']['numpy_version']}、PyTorch {preflight['runtime_environment']['torch_version']}（CUDA 可用性={preflight['runtime_environment']['cuda_available_but_unused']}，本轮运行固定 CPU）
- 执行依据 SHA256：`{source_inventory['execution_basis']['sha256']}`
- 配置差异 SHA256：`{source_inventory['configuration_diff']['sha256']}`
- 原始产物清单：`ROUND43_RAW_ARTIFACT_MANIFEST.json`（{len(raw_manifest['files'])}个文件，SHA256 `{raw_manifest_sha}`）
- 无环境正确性门禁：通过，`draft_mode=false`；记录在 `ROUND43_PREFLIGHT.json`。
- 每个 seed 首四集使用真实 batch_samples/GAE/actor/恢复路径；通过后完成其余124集。
- 固定命令：`D:\\project_env\\python.exe runs/comm_one_shot_full_trace_ablation_20261004/r43_execute.py`。

## 限制

结果来自已用于开发的固定16场景。没有独立确认实验、额外训练、调参、hazard decoder 或补充场景。通信成本仅指载荷尝试字节，丢包仍计费，不代表全系统通信或能耗。ROUND42 原始模型、tape、日志和报告均未覆盖。
"""
    report_path = RUN_DIR / "ROUND43_REPORT.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    status = {
        "kind": "round43-project-status/1",
        "round": 43,
        "status": "complete_local_delivery",
        "source_freeze_head": source_freeze_head,
        "branch": branch,
        "training_version": frozen["round42"]["training_version"],
        "result": verdict["scientific_result"],
        "prelaunch_gate": "passed",
        "training_episodes": 384,
        "final_evaluations": 96,
        "normal_launches": counters["normal_used"],
        "reserve_launches": counters["reserve_used"],
        "hard_cap": counters["hard_cap"],
        "independent_confirmation": False,
        "remote_push": False,
    }
    _atomic_json(RUN_DIR / "ROUND43_STATUS.json", status)
    return {"report_path": str(report_path), "report_sha256": _sha256(report_path),
            "status": status, "raw_manifest_sha256": raw_manifest_sha}


__all__ = ["write_final_report"]
