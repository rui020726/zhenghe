"""ROUND25 判定：候选证据，不是算法总胜利。Tables, judgement and verdict read one statistics file.

执行依据 §六/§七 的判定字段：
- primary_cost_evidence：主成本 dC 的 95% CI 上界 < 0 才写 lower_cost_supported；
- observed_task_count_status：对两参照均逐场景 dY >= 0 才写 no_observed_count_loss；
- goal_identity_tradeoffs、constraint_guarantee、training_seed_stability 单列；
- 即使任务全打平也不证明 delta=0 总体非劣；目标身份替换必须披露。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import r25_common as common
import r25_stats as stats


def verdict_payload() -> dict[str, Any]:
    paired = stats.read_statistics()
    identity = json.loads((common.OUTPUT_DIR / "TASK_IDENTITY_TRADEOFFS.json").read_text(
        encoding="utf-8")) if (common.OUTPUT_DIR / "TASK_IDENTITY_TRADEOFFS.json").exists() else None
    absolute_path = common.OUTPUT_DIR / "ABSOLUTE_METRICS.json"
    absolute = json.loads(absolute_path.read_text(encoding="utf-8")) if absolute_path.exists() else None

    comparisons = paired["comparisons"]
    primary = comparisons[common.PRIMARY_COST_COMPARATOR]
    task_fields = {}
    for comparator in common.TASK_COMPARATORS:
        entry = comparisons[comparator]
        task_fields[comparator] = {
            "dY_mean": entry["task"]["mean"], "dY_ci": [entry["task"]["ci_low"],
                                                        entry["task"]["ci_high"]],
            "dC_mean": entry["cost"]["mean"], "dC_ci": [entry["cost"]["ci_low"],
                                                        entry["cost"]["ci_high"]],
            "task_counts": {key: entry["counts"].get(key, 0)
                            for key in ("task_improved", "task_tied", "task_degraded")},
            "cost_counts": {key: entry["counts"].get(key, 0)
                            for key in ("cost_lower", "cost_equal", "cost_higher")},
            "joint_counts": {key: entry["counts"].get(key, 0)
                             for key in ("joint_dominant", "both_tied",
                                         "tradeoff_task_gain_cost_loss",
                                         "tradeoff_task_loss_cost_gain", "task_only_worse",
                                         "cost_only_worse", "worse_both")},
            "task_degraded_scenes": entry["task_degraded_scenes"],
            "zero_degradation_observed": entry["zero_degradation_observed"],
            "zero_event_upper_bound_95": entry.get("zero_event_upper_bound_95"),
        }

    primary_cost_ok = primary["cost"]["ci_high"] < 0.0
    no_task_loss = all(entry["zero_degradation_observed"]
                       for entry in comparisons.values())
    primary_cost_evidence = ("lower_cost_supported" if primary_cost_ok
                             else "lower_cost_not_established")
    observed_task_count_status = ("no_observed_count_loss" if no_task_loss
                                  else "task_count_tradeoff_observed")

    identity_summary: dict[str, Any] = {}
    if identity is not None:
        for comparator, rows in identity["comparators"].items():
            identity_summary[comparator] = {
                "scenes_with_lost_goals": [row["scene"] for row in rows if row["lost_on_time_goal_ids"]],
                "scenes_with_gained_goals": [row["scene"] for row in rows
                                             if row["gained_on_time_goal_ids"]],
                "identity_swaps_despite_net_tie": [row["scene"] for row in rows
                                                   if row["identity_swap_despite_net_tie"]],
                "rows": rows,
            }

    statements: list[str] = []
    if primary_cost_ok and no_task_loss:
        statements.append(
            "相对 late_discovery_gate(147)，固定 seed33 greedy 候选的已计量载荷尝试成本降幅"
            "在这 64 个新场景得到支持（主成本 dC 的 95% CI 上界 < 0），且未观察到净任务数损失。")
    elif not primary_cost_ok:
        statements.append(
            "主成本证据不足：相对 late_discovery_gate(147) 的 dC 均值即使更低，95% CI 上界也未 < 0，"
            "不得写已证明降本。")
    if not no_task_loss:
        statements.append(
            "观察到任务数权衡：存在逐场景净按时任务数退化，逐场景公开，不用其他场景的改善抵消。")

    return {
        "round": "25",
        "run_id": common.OUTPUT_DIR.name,
        "question": "开发集事后选定的 seed33 最终 greedy 模型，在新场景上能否比固定规则少花已计量"
                    "成本，同时未观察到按时任务数损失？",
        "primary_cost_evidence": primary_cost_evidence,
        "observed_task_count_status": observed_task_count_status,
        "goal_identity_tradeoffs": identity_summary,
        "constraint_guarantee": "not_established",
        "training_seed_stability": "not_established",
        "comparisons": task_fields,
        "absolute_metrics_included": absolute is not None,
        "statements": statements,
        "limits": [
            "候选是看过 ROUND24 16 个开发场景后从事后选出的固定模型；本结果检验此固定候选的泛化，"
            "不能消除模型筛选事实，也不能估计随机训练一次的成功率",
            "64 场景是本轮资源约束下的固定探索性复验规模，不是已论证足以检验 delta=0 的功效样本",
            "任务样本全打平不证明 delta=0 总体非劣；即使 64/64 无退化，1-0.05^(1/64)≈4.57% 也只是"
            "i.i.d. 场景假设下发生净退化场景概率的单侧 95% 上限，不是期望任务损失上限，更不是部署保证",
            "bootstrap 描述此固定候选在指定生成分布上的场景不确定性，不含训练初始化不确定性",
            "训练 sampled 对偶约束不能转移为 greedy 保证；约束保证始终 not_established",
            "seed11/22 与 sampled 的 ROUND24 差异只作历史背景，不用一个候选成功宣称 3/3 成功",
            "费用口径仅为已计量载荷尝试成本（丢包照计、拒付计0并单列），不是全系统通信量或真实无线能耗",
            "192 次启动不是 192 个独立场景；未池化任何旧开发场景",
        ],
        "publication": {"remote_verification": "NOT_PERFORMED",
                        "publication_status": "PENDING_NETWORK",
                        "note": "local execution complete; remote publication pending; no network "
                                "attempt was made this session and no unpublished state is "
                                "claimed as published"},
        "planning_end_caveat": "规划端未独立取得 ROUND24 原件；本轮复用执行端已核验件并以 hash 登记",
    }


def verdict_markdown(payload: dict[str, Any]) -> str:
    primary = payload["comparisons"][common.PRIMARY_COST_COMPARATOR]
    immediate = payload["comparisons"]["immediate"]
    lines = [
        "# ROUND25 判定：固定 seed33 greedy 候选在 64 个新场景上的复验",
        "",
        f"- run_id: `{payload['run_id']}`",
        "- 问题：" + payload["question"],
        "",
        "## 判定字段",
        "",
        f"- `primary_cost_evidence`: **{payload['primary_cost_evidence']}**",
        f"- `observed_task_count_status`: **{payload['observed_task_count_status']}**",
        f"- `constraint_guarantee`: {payload['constraint_guarantee']}",
        f"- `training_seed_stability`: {payload['training_seed_stability']}",
        "",
        "## 主成本比较（seed33 greedy 对 late_discovery_gate_147）",
        "",
        f"- dC 均值 {primary['dC_mean']:.1f} B/场景，95% CI "
        f"[{primary['dC_ci'][0]:.1f}, {primary['dC_ci'][1]:.1f}]",
        f"- 成本计数：更省 {primary['cost_counts']['cost_lower']} / 相同 "
        f"{primary['cost_counts']['cost_equal']} / 更高 {primary['cost_counts']['cost_higher']}",
        f"- 任务计数（同一比较）：改善 {primary['task_counts']['task_improved']} / 相同 "
        f"{primary['task_counts']['task_tied']} / 退化 {primary['task_counts']['task_degraded']}",
        "",
        "## 任务比较（对 immediate 与对 late）",
        "",
        f"- 对 immediate：dY 均值 {immediate['dY_mean']:.4f}，CI "
        f"[{immediate['dY_ci'][0]:.4f}, {immediate['dY_ci'][1]:.4f}]；dC 均值 "
        f"{immediate['dC_mean']:.1f} B，CI [{immediate['dC_ci'][0]:.1f}, "
        f"{immediate['dC_ci'][1]:.1f}]",
        "",
        "## 陈述",
        "",
    ]
    lines.extend(f"- {line}" for line in payload["statements"])
    lines.extend(["", "## 限定", ""])
    lines.extend(f"- {line}" for line in payload["limits"])
    lines.extend([
        "",
        "## 发布状态",
        "",
        f"- remote_verification: {payload['publication']['remote_verification']}",
        f"- publication_status: {payload['publication']['publication_status']}",
        f"- {payload['publication']['note']}",
        "",
    ])
    return "\n".join(lines)


def write_verdict() -> tuple[Path, Path]:
    payload = verdict_payload()
    json_path = common.OUTPUT_DIR / "ROUND25_VERDICT.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                         encoding="utf-8")
    md_path = common.OUTPUT_DIR / "ROUND25_VERDICT.md"
    md_path.write_text(verdict_markdown(payload), encoding="utf-8")
    return json_path, md_path


def main(argv: list[str]) -> int:
    paths = write_verdict()
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    print(json.dumps({"primary_cost_evidence": payload["primary_cost_evidence"],
                      "observed_task_count_status": payload["observed_task_count_status"],
                      "verdict": [str(path) for path in paths]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
