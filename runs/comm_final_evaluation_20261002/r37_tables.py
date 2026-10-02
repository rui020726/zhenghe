"""ROUND37 表格渲染（0 环境启动）：把 `ROUND37_STATISTICS.json` 渲染成报告用的 markdown 表格。

目的只有一个：报告里的数字必须与统计件同源。渲染函数对每个量固定小数位，报告正文只引用这些表，
不另抄数字。若文字与表格冲突，以本件为准（本件由统计件机械生成）。

    /d/project_env/python.exe r37_tables.py
"""

from __future__ import annotations

import json
import sys
from typing import Any

import r37_common as r37

SEEDS = r37.TRAINING_SEEDS
MODES = ("greedy", "sampled")
RULES = ("immediate", "late_discovery_gate", "always_hold")
Y_DP, C_DP, DIFF_Y_DP, DIFF_C_DP = 4, 1, 4, 1


def interval(item: dict[str, Any], decimals: int) -> str:
    return (f"{item['mean']:.{decimals}f} "
            f"[{item['ci_low']:.{decimals}f}, {item['ci_high']:.{decimals}f}]")


def absolute_table(payload: dict[str, Any], arms: list[str], title: str) -> list[str]:
    lines = [title, "",
             "| 臂 | 场景数 | 按时任务数/集 [95% CI] | 计量字节/集 [95% CI] | 发射/集 | "
             "发送尝试/集 | 交付/集 | 丢包/集 | 预算拒付/集 | 过期目标/集 | 发现目标/集 |",
             "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        item = payload["absolute_per_arm"][arm]
        n = item["timely_tasks_per_episode"]["n_scenarios"]
        lines.append("| {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | "
                     "{:.3f} | {:.3f} |".format(
                         arm, n, interval(item["timely_tasks_per_episode"], Y_DP),
                         interval(item["metered_bytes_per_episode"], C_DP),
                         item["emissions_per_episode"]["mean"],
                         item["transmission_attempts_per_episode"]["mean"],
                         item["packets_delivered_per_episode"]["mean"],
                         item["packets_lost_per_episode"]["mean"],
                         item["budget_refused_sends_per_episode"]["mean"],
                         item["expired_targets_per_episode"]["mean"],
                         item["discovered_targets_per_episode"]["mean"]))
    return lines


def paired_table(payload: dict[str, Any], mode: str) -> list[str]:
    label = "主表（greedy）" if mode == "greedy" else "辅助表（sampled）"
    lines = [f"### {label}：相对规则臂的逐场景配对差", "",
             "| 模型 | 参照 | 任务差/集 [95% CI] | 场景级 改善/持平/退化 | 字节差/集 [95% CI] | "
             "场景级 更省/持平/更贵 | 按时集合完全相同的场景数 | 仅模型有 | 仅参照有 |",
             "|---|---|---|---:|---|---:|---:|---:|---:|"]
    for seed in SEEDS:
        arm = f"model_s{seed}_{mode}"
        for reference in (payload["paired_vs_rules_per_seed_and_mode"][arm]):
            entry = payload["paired_vs_rules_per_seed_and_mode"][arm][reference]
            task = entry["per_scenario_task_counts_vs_reference"]
            byte = entry["per_scenario_byte_counts_vs_reference"]
            exchange = entry["timely_target_set_exchange"]
            lines.append("| {} | {} | {} | {} | {} | 更省 {} / 持平 {} / 更贵 {} | {} / {} | {} | "
                         "{} |".format(
                             arm, reference,
                             interval(entry["task_difference_per_episode"], DIFF_Y_DP),
                             f"改善 {task['improved']} / 持平 {task['flat']} / "
                             f"退化 {task['declined']}",
                             interval(entry["byte_difference_per_episode"], DIFF_C_DP),
                             byte["cheaper"], byte["same"], byte["costlier"],
                             exchange["scenarios_with_identical_timely_target_set"],
                             exchange["scenarios"],
                             exchange["targets_only_in_the_model_arm"],
                             exchange["targets_only_in_the_reference_arm"]))
    return lines


def consistency_table(payload: dict[str, Any]) -> list[str]:
    lines = ["### 三个训练 seed 的符号一致性（描述性，非稳定性证明）", "",
             "| 模式 | 比较 | 每 seed 的差值均值 | 符号是否一致 |",
             "|---|---|---|---|"]
    for mode, item in payload["seed_consistency_descriptive_only"].items():
        for key, label in (("task_difference_vs_immediate_per_seed", "任务差 vs immediate（个/集）"),
                           ("byte_difference_vs_immediate_per_seed", "字节差 vs immediate（B/集）")):
            values = ", ".join(f"{name} {value:+.{4 if 'task' in key else 1}f}"
                               for name, value in item[key].items())
            sign = item["all_seeds_same_sign_on_task_difference"] if "task" in key else \
                item["all_seeds_same_sign_on_byte_difference"]
            lines.append(f"| {mode} | {label} | {values} | {'是' if sign else '否'} |")
    return lines


def quality_table(payload: dict[str, Any]) -> list[str]:
    lines = ["## 0. 矩阵质量与分母（零结果场景一律保留，不事后剔除）", "",
             f"分母：每臂 64 个场景 x 1 集；每集最多 6 个按时目标（`num_goals`），"
             f"故合计上限 {6 * r37.SCENARIO_COUNT}。", "",
             "| 臂 | 按时任务合计 | 按时任务上限 | 计量字节合计 | 零按时任务场景数 | 零发现场景数 | "
             "零计量字节场景数 | 按时任务 min–max | 计量字节 min–max |",
             "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for arm, item in payload["matrix_quality_per_arm"].items():
        lines.append("| {} | {:.0f} | {} | {:.0f} | {} / 64 | {} / 64 | {} / 64 | "
                     "{:.0f}–{:.0f} | {:.0f}–{:.0f} |".format(
                         arm, item["timely_task_total_over_the_block"],
                         6 * r37.SCENARIO_COUNT,
                         item["metered_bytes_total_over_the_block"],
                         item["scenarios_with_zero_timely_tasks"],
                         item["scenarios_with_zero_discoveries"],
                         item["scenarios_with_zero_metered_bytes"],
                         item["min_on_time"], item["max_on_time"],
                         item["min_metered_bytes"], item["max_metered_bytes"]))
    return lines


def main(argv: list[str]) -> int:
    payload = r37.read_json(r37.HERE / "ROUND37_STATISTICS.json")
    plan = r37.read_json(r37.HERE / "FROZEN_EVALUATION_PLAN.json")
    reconcile = r37.read_json(r37.HERE / "EVALUATION_LEDGER_RECONCILIATION.json")
    authoritative = reconcile["authoritative_sources"]
    book = reconcile["this_round_book"]
    lines: list[str] = [
        "# ROUND37 结果表（由 `ROUND37_STATISTICS.json` 机械渲染，报告正文只引用本件）",
        "",
        f"- 场景块：{plan['scenario_provenance']['block'][0]}..{plan['scenario_provenance']['block'][1]}"
        f"（{'九臂 x 64 = 576 格全部到齐' if payload['matrix_complete'] else '矩阵不完整'}）",
        f"- 统计方案：{payload['frozen_plan']['generator']}，seed "
        f"{payload['frozen_plan']['pcg64_seed']}，{payload['frozen_plan']['draws']} 次重抽 × "
        f"{payload['frozen_plan']['shared_indices']} 个共享索引，"
        f"{payload['frozen_plan']['quantile_method']} 分位，场景升序",
        f"- 额度对账（权威三方一致）：行 {authoritative['row_files']} / claim "
        f"{authoritative['claim_files']} / 真实 Stage C ledger "
        f"{authoritative['cell_starts_in_stage_c_ledger']}，技术失败 "
        f"{authoritative['failed_entries_in_stage_c_ledger']} 次，独立流录制 "
        f"{book['recording_launches']} 次（判定不需要，额度未挪用），训练 "
        f"{book['training_launches']} 次、optimizer.step {book['optimizer_steps']} 次",
        f"- 本轮自记 book 少计一次（{book['starts']} 而非 "
        f"{authoritative['cell_starts_in_stage_c_ledger']}）：并行分片共享计数器的丢写，"
        "见 `ROUND37_HARNESS_DEFECTS.json` 的 R37-D2；上表所有计数取自权威件",
        "- 分位数是**描述性** bootstrap（对 64 个场景重抽），不是抽样推断意义上的总体置信区间；"
        "区间包含 0 不等于等效。", ""]
    model_arms = [f"model_s{seed}_{mode}" for mode in MODES for seed in SEEDS]
    lines += quality_table(payload) + [""]
    lines += absolute_table(payload, [f"model_s{seed}_greedy" for seed in SEEDS],
                            "## 1. 主表：三个模型的 greedy 绝对量（分母 64 个新场景）") + [""]
    lines += absolute_table(payload, [f"model_s{seed}_sampled" for seed in SEEDS],
                            "## 2. 辅助表：三个模型的 sampled 绝对量") + [""]
    lines += absolute_table(payload, list(RULES), "## 3. 规则臂绝对量（同一 64 个场景）") + [""]
    for mode in MODES:
        lines += paired_table(payload, mode) + [""]
    lines += consistency_table(payload) + [""]
    lines += ["## 4. 判读限定（与授权边界 §6 逐条对应）", ""]
    lines += [f"- {item}" for item in payload["reporting_cautions"]]
    text = "\n".join(lines) + "\n"
    (r37.HERE / "RESULTS_TABLES.md").write_text(text, encoding="utf-8")
    print(json.dumps({"tables_written": True, "characters": len(text),
                      "cells": payload["cells"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
