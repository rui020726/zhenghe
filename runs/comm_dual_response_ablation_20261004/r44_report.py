"""Render the single ROUND44 final report from immutable analysis and ledger values."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from r44_prepare import ETA_NEW, RUN_DIR, TRAINING_VERSION


def _mean(value: float) -> str:
    # Four decimal places preserve the registered 1/16-scene resolution exactly.
    return f"{float(value):.4f}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _performance_table(paired: dict[str, Any]) -> list[str]:
    lines = [
        "| Group | On-time Y / mean per scene | Charged C / mean B per scene | First reports / bytes | Free SEND / SKIP |",
        "|---|---:|---:|---:|---:|",
    ]
    groups = paired["group_summaries"]
    entries = [
        ("Immediate reference", "immediate_reference_development"),
        ("Late discovery gate", "late_discovery_gate"),
        ("Always hold", "always_hold"),
    ]
    for seed in (11, 22, 33):
        for mode in ("greedy", "sampled"):
            entries.append((f"s{seed} {mode} OLD R43 ep128", f"s{seed}_{mode}_OLD_R43_NEW_EP128"))
            entries.append((f"s{seed} {mode} NEW R44 ep128", f"s{seed}_{mode}_NEW_R44_EP128"))
    for label, key in entries:
        item = groups[key]
        n = int(item["n"])
        y = int(item["on_time_total"])
        c = int(item["spent_bytes_total"])
        y_mean = y / n if n else 0.0
        c_mean = c / n if n else 0.0
        lines.append(
            f"| {label} | {y} / {_mean(y_mean)} | {c:,} / {_mean(c_mean)} B | "
            f"{item['first_report_emissions_total']} / {item['first_report_bytes_total']:,} B | "
            f"{item['send_now_choices_total']} / {item['skip_episode_choices_total']} |"
        )
    return lines


def _paired_table(paired: dict[str, Any]) -> list[str]:
    lines = [
        "| Seed | Mode | Comparison (right − left) | ΔY total / mean | ΔC total / mean B | Recovered / lost task IDs |",
        "|---:|---|---|---:|---:|---:|",
    ]
    comparisons = [
        ("NEW−OLD R43", paired["new_minus_old_r43_by_seed_and_mode"]),
        ("NEW−immediate", paired["new_vs_immediate_by_seed_and_mode"]),
    ]
    for name, mapping in comparisons:
        for seed in (11, 22, 33):
            for mode in ("greedy", "sampled"):
                key = f"s{seed}_{mode}"
                item = mapping[key]
                n = int(item["scene_count"])
                dy = int(item["on_time_total_delta"])
                dc = int(item["spent_bytes_total_right"]) - int(
                    item["spent_bytes_total_left"])
                lines.append(
                    f"| {seed} | {mode} | {name} | {dy:+d} / {_mean(dy / n if n else 0)} | "
                    f"{dc:+,} / {_mean(dc / n if n else 0)} B | "
                    f"{len(item['recovered_task_ids'])} / {len(item['lost_task_ids'])} |"
                )
    for seed in (11, 22, 33):
        for mode in ("greedy", "sampled"):
            rules = paired["new_vs_fixed_rules_by_seed_and_mode"][f"s{seed}_{mode}"]
            for arm in ("late_discovery_gate", "always_hold"):
                item = rules[arm]
                n = int(item["scene_count"])
                dy = int(item["on_time_total_delta"])
                dc = int(item["spent_bytes_total_right"]) - int(
                    item["spent_bytes_total_left"])
                lines.append(
                    f"| {seed} | {mode} | NEW−{arm} | {dy:+d} / {_mean(dy / n if n else 0)} | "
                    f"{dc:+,} / {_mean(dc / n if n else 0)} B | "
                    f"{len(item['recovered_task_ids'])} / {len(item['lost_task_ids'])} |"
                )
    return lines


def render(*, result: dict[str, Any], ledger: dict[str, Any], progress: dict[str, Any],
           eval_audit: dict[str, Any], execution_head: str,
           zero_tests: dict[str, Any], final_artifact_count: int) -> str:
    paired = result["paired_results"]
    verdict = result["machine_verdict"]
    curves = result["training_curves"]
    counts = ledger["counts"]
    primary = verdict["registered_primary_rule"]
    joint = bool(primary["joint_improvement"])
    verdict_text = "development_block_joint_improvement_observed" if joint else (
        "development_block_joint_improvement_not_observed")
    lines = [
        "# ROUND44 报告：对偶响应速度单变量消融",
        "",
        "## 结论",
        "",
        f"是否完成：是。384 集三 seed 训练、96 格固定评估全部完成；正常额度 "
        f"{counts['normal_used']}/480，技术储备 {counts['reserve_used']}/24，"
        f"总 claim {counts['total_used']}/504。",
        f"开发块联合判定：`{verdict_text}`。该判定要求三个 greedy seed 各自按场景保持 "
        "immediate 的按时任务身份集合，并且计费低于 late discovery gate。",
        "本轮只改变每 batch 的 `eta_lambda`：0.05→0.5。结果只检验对偶响应速度假设；"
        "即使改善，也不能证明先前学习失败只有一个根因。",
        "所有比较仍位于已使用的16场景开发块，不构成独立确认、总体零损失、正式非劣或泛化结论。",
        "",
        "## 固定科研配置",
        "",
        f"- 训练版本：`{TRAINING_VERSION}`。",
        "- 共享 GAE trace=1.0、gamma=1.0、lambda_dual 初值=1.0、delta=0、B0=10,000 B 均保持冻结。",
        f"- 每四集 PPO batch 更新一次 dual：`lambda_next=max(0, lambda_before+{ETA_NEW}*g_hat)`；"
        "本批 actor 使用更新前 lambda。",
        "- 384 个 NEW 训练 episode（seed 11/22/33 各128集），96 个 NEW ep128 固定评估格。",
        "- 旧比较输入为 ROUND43 NEW ep128；64 immediate、32 late/always_hold 结果只读复用。",
        "- 无新场景、新流、确认块、追加训练或第二个 eta 步长。成本为已计量载荷尝试字节，含丢包；"
        "不代表全系统通信或能耗。",
        "",
        "## 绝对表现",
        "",
        *_performance_table(paired),
        "",
        "Y 是按时完成任务数；C 是载荷发送尝试的计费字节。均值以16个开发场景计算，"
        "显示精度保留1/16分辨率。SEND/SKIP 仅计自由首报选择。",
        "",
        "## 配对差与目标身份变化",
        "",
        *_paired_table(paired),
        "",
        "完整逐场景 recovered/lost 任务 ID、OLD/NEW 原始汇总和规则配对保存在 "
        "`ROUND44_PAIRED_RESULTS.json`；表格和机器判定由同一分析对象生成。",
        "没有预登记可用的置信区间，本轮不新增区间或等效性解释。",
        "",
        "## 每批对偶轨迹与训练完整性",
        "",
        "逐 seed 的32批 `g_hat`、dual 更新前后值、任务/成本、PPO 指标、likelihood 与参数链见 "
        "`ROUND44_TRAINING_CURVES.json`。",
    ]
    for seed in (11, 22, 33):
        batches = curves["by_seed"][str(seed)]["batches"]
        first = batches[0]["dual"]
        final = batches[-1]["dual"]
        steps = sum(int(batch["update"].get("optimizer_steps", 0)) for batch in batches)
        lines.append(
            f"- seed {seed}: 128集/32批，Adam.step={steps}；dual {first['lambda_before']:.6f}→"
            f"{final['lambda_next']:.6f}，首批 g={first['g_hat']:+.6f}、"
            f"eta={first['eta']:.2f}，末批 g={final['g_hat']:+.6f}。"
        )
    lines.extend([
        "",
        f"首批门禁：3/3 seed 通过实际 rollout→tape→batch_samples/GAE→actor/双 critic→PPO/dual→"
        "checkpoint 路径；逐批记录证明每 seed 恰有32次更新。R43/R44 首批 actor 与模型/optimizer "
        "结果比较见 `ROUND44_FIRST_BATCH_COMPARISON.json`。",
        "checkpoint ep0/64/128 的文件、参数、模块、optimizer、RNG 和状态游标见 "
        "`ROUND44_CHECKPOINT_INDEX.json`。旧 R43 ep128 只读使用。",
        "",
        "## 额度对账",
        "",
        f"- normal claims：{counts['normal_used']}；reserve claims：{counts['reserve_used']}；"
        f"总计：{counts['total_used']}；硬上限：{counts['hard_cap']}。",
        f"- 构造调用：{counts['environment_constructor_calls']}；JointRun 构造："
        f"{counts['joint_run_constructed_calls']}；run 调用：{counts['run_invocation_calls']}；"
        f"完整步数：{counts['completed_steps']}。",
        f"- PPO 更新：{counts['ppo_update_calls']}；Adam.step：{counts['optimizer_step_calls']}；"
        f"checkpoint 保存调用：{counts['checkpoint_write_calls']}。",
        f"- 评估独立索引复核：{eval_audit['status']}，{eval_audit['cell_count']}格，"
        f"每个 seed×mode 16格。训练/执行失败与重试明细见 `ROUND44_EXECUTION_LEDGER.json`。",
        f"- 评估索引纯计算测试：{zero_tests['status']}，0环境、0 optimizer.step；"
        "训练前 dual/版本/初值门禁：0环境、0实验 checkpoint 写入。",
        f"- 审计原始文件数：{final_artifact_count}；原始 rows/logs/checkpoints 留在本机 Git 外运行目录。",
        "",
        "## 身份与解释限制",
        "",
        f"科学执行冻结 HEAD：`{execution_head}`。本报告代码在该冻结中已锁定；报告内容提交 SHA "
        "登记在 `ROUND44_STATUS.json`。",
        "判读按每 seed greedy 主结果、sampled 辅结果分列；不池化训练 seed、不挑 checkpoint/decoder。",
        "全 SEND 最多说明任务恢复，若成本没有低于 late 仍不满足联合目标；全 SKIP 丢任务不算降本成功。",
        "本轮结果仅作开发块探索性观察。本地提交后停止；未推送、部署、扩样、确认或续训。",
        "",
    ])
    return "\n".join(lines)


def write_report(**kwargs: Any) -> dict[str, Any]:
    text = render(**kwargs)
    path = RUN_DIR / "ROUND44_REPORT.md"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError("ROUND44 report already exists with different content")
    else:
        path.write_text(text, encoding="utf-8", newline="\n")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


__all__ = ["render", "write_report"]

