"""ROUND37 统计（0 环境启动、0 optimizer.step）：唯一一份统计结果件，表格与文字都从它引用。

冻结方案（授权边界 §6 逐字转写）：场景升序、`PCG64(20261002)`、10000 次重抽 × 64 个**共享**索引、
`linear` 分位、统计量 `values[indices].mean(axis=1)`、区间 2.5/97.5。共享索引矩阵只生成一次，
所有臂、所有比较都用同一矩阵 ⇒ 表与表之间彼此一致。

报告的是：每 seed 每模式的按时任务数/集与计量字节/集；相对 immediate 与相对
late_discovery_gate 的任务差与成本差（逐场景配对后取均值）；逐场景改善/持平/退化计数；
按时目标集合是否交换；发送尝试、预算拒付与交付计数。

不做的事：不合并 greedy 与 sampled；不把三个训练 seed 池化成独立场景样本；不根据区间包含零说
"等效"；不把"未观察到损失"说成 δ=0 的总体保证。

    /d/project_env/python.exe r37_stats.py
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np

import r37_common as r37

SEEDS = r37.TRAINING_SEEDS
MODES = ("greedy", "sampled")
COMPARATORS = ("immediate", "late_discovery_gate")
RULE_LABEL = {"immediate": "immediate", "late_discovery_gate": "late_discovery_gate_147"}


def cells() -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for job in r37.jobs():
        arm, scenario = job["arm"], job["scenario"]
        path = r37.row_path(job)
        if not path.exists():
            raise SystemExit(f"missing cell before statistics: {path}")
        out[(arm, scenario)] = r37.read_json(path)
    return out


def vector(table: dict[tuple[str, int], dict[str, Any]], arm: str, field: str) -> np.ndarray:
    return np.array([float(table[(arm, scenario)][field]) for scenario in r37.SCENARIOS],
                    dtype=np.float64)


def target_sets(table: dict[tuple[str, int], dict[str, Any]], arm: str) -> list[frozenset[str]]:
    out = []
    for scenario in r37.SCENARIOS:
        row = table[(arm, scenario)]
        out.append(frozenset(item["task_id"] for item in (row.get("task_reward_events") or [])
                             if item["credited"]))
    return out


def shared_indices(size: int, draws: int, seed: int) -> np.ndarray:
    """The one frozen resampling matrix: PCG64(seed), `draws x size` integer indices."""
    rng = np.random.default_rng(np.random.PCG64(seed))
    return rng.integers(0, size, size=(draws, size))


def boot(value: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    statistics = value[indices].mean(axis=1)
    low, high = np.quantile(statistics, [0.025, 0.975], method="linear")
    return {"mean": float(value.mean()), "ci_low": float(low), "ci_high": float(high),
            "n_scenarios": int(value.size)}


def counts(difference: np.ndarray) -> dict[str, int]:
    return {"improved": int((difference > 0).sum()), "flat": int((difference == 0).sum()),
            "declined": int((difference < 0).sum())}


def set_exchange(model: list[frozenset[str]], reference: list[frozenset[str]]) -> dict[str, Any]:
    identical = sum(1 for left, right in zip(model, reference) if left == right)
    out = sum(len(left - right) for left, right in zip(model, reference))
    new = sum(len(right - left) for left, right in zip(model, reference))
    return {"scenarios_with_identical_timely_target_set": identical,
            "targets_only_in_the_model_arm": out, "targets_only_in_the_reference_arm": new,
            "scenarios": len(model)}


def main(argv: list[str]) -> int:
    r37.configure()          # row paths are derived from the rebound Stage C globals
    table = cells()
    indices = shared_indices(len(r37.SCENARIOS), r37.STATISTICS["draws"],
                             r37.STATISTICS["pcg64_seed"])
    arms = list(r37.ALL_ARMS)
    assert len(arms) * len(r37.SCENARIOS) == len(table) == 576, (
        f"the matrix must be complete before statistics: {len(table)} cells")

    absolute = {}
    for arm in arms:
        absolute[arm] = {
            "timely_tasks_per_episode": boot(vector(table, arm, "on_time"), indices),
            "metered_bytes_per_episode": boot(vector(table, arm, "spent_bytes"), indices),
            "emissions_per_episode": boot(vector(table, arm, "emissions"), indices),
            "transmission_attempts_per_episode": boot(
                np.array([float(table[(arm, s)]["relay_counters"]
                                    .get("transmission_attempts", 0))
                          for s in r37.SCENARIOS]), indices),
            "packets_delivered_per_episode": boot(
                np.array([float(table[(arm, s)]["relay_counters"].get("packets_delivered", 0))
                          for s in r37.SCENARIOS]), indices),
            "packets_lost_per_episode": boot(
                np.array([float(table[(arm, s)]["relay_counters"].get("packets_lost", 0))
                          for s in r37.SCENARIOS]), indices),
            "budget_refused_sends_per_episode": boot(
                np.array([float(table[(arm, s)]["relay_counters"]
                                .get("budget_refused_sends", 0)) for s in r37.SCENARIOS]), indices),
            "expired_targets_per_episode": boot(vector(table, arm, "expired"), indices),
            "discovered_targets_per_episode": boot(vector(table, arm, "discovered"), indices)}

    paired = {}
    for seed in SEEDS:
        for mode in MODES:
            arm = f"model_s{seed}_{mode}"
            model_targets = target_sets(table, arm)
            entry: dict[str, Any] = {}
            for comparator in COMPARATORS:
                reference = RULE_LABEL[comparator]
                model_y = vector(table, arm, "on_time")
                model_c = vector(table, arm, "spent_bytes")
                comparator_y = vector(table, comparator, "on_time")
                comparator_c = vector(table, comparator, "spent_bytes")
                difference_y = model_y - comparator_y
                difference_c = model_c - comparator_c
                entry[reference] = {
                    "task_difference_per_episode": boot(difference_y, indices),
                    "byte_difference_per_episode": boot(difference_c, indices),
                    "per_scenario_task_counts_vs_reference": counts(difference_y),
                    "per_scenario_byte_counts_vs_reference": {
                        "cheaper": int((difference_c < 0).sum()),
                        "same": int((difference_c == 0).sum()),
                        "costlier": int((difference_c > 0).sum())},
                    "timely_target_set_exchange": set_exchange(model_targets,
                                                               target_sets(table, comparator))}
            paired[arm] = entry

    seed_consistency: dict[str, Any] = {}
    for mode in MODES:
        means_y = {f"s{seed}": paired[f"model_s{seed}_{mode}"][RULE_LABEL["immediate"]]
                   ["task_difference_per_episode"]["mean"] for seed in SEEDS}
        means_c = {f"s{seed}": paired[f"model_s{seed}_{mode}"][RULE_LABEL["immediate"]]
                   ["byte_difference_per_episode"]["mean"] for seed in SEEDS}
        same_sign_y = len({value > 0 for value in means_y.values()}) == 1
        same_sign_c = len({value > 0 for value in means_c.values()}) == 1
        seed_consistency[mode] = {
            "task_difference_vs_immediate_per_seed": means_y,
            "byte_difference_vs_immediate_per_seed": means_c,
            "all_seeds_same_sign_on_task_difference": same_sign_y,
            "all_seeds_same_sign_on_byte_difference": same_sign_c,
            "note": "three training seeds are three fits of the same recipe; agreement in sign "
                    "is not evidence that the training algorithm is stable, and the seeds are "
                    "not independent scenario samples"}

    quality = {}
    for arm in arms:
        on_time = vector(table, arm, "on_time")
        discovered = vector(table, arm, "discovered")
        spent = vector(table, arm, "spent_bytes")
        quality[arm] = {
            "timely_task_total_over_the_block": float(on_time.sum()),
            "metered_bytes_total_over_the_block": float(spent.sum()),
            "scenarios_with_zero_timely_tasks": int((on_time == 0).sum()),
            "scenarios_with_zero_discoveries": int((discovered == 0).sum()),
            "scenarios_with_zero_metered_bytes": int((spent == 0).sum()),
            "min_on_time": float(on_time.min()), "max_on_time": float(on_time.max()),
            "min_metered_bytes": float(spent.min()), "max_metered_bytes": float(spent.max())}
    payload = {
        "kind": "round37-statistics", "schema": "round37-statistics/1", "round": "37",
        "generated_at": time.strftime("%F %T"),
        "environment_launches_used": 0, "optimizer_steps_used": 0,
        "frozen_plan": {**r37.STATISTICS, "scenarios": [r37.SCENARIOS[0], r37.SCENARIOS[-1]],
                        "shared_index_matrix": "one matrix reused by every arm and comparison"},
        "matrix_complete": True, "cells": len(table),
        "matrix_quality_per_arm": quality,
        "quality_rule": "零完成、零发现或零计量的场景是本块的合法结果，一律保留在册并计入分母；"
                        "只有技术失败（异常、缺记录）才可重跑",
        "metric_definitions": {
            "timely_tasks_per_episode": "每集 on_time（lifecycle 指标，个/集，分母 64 个场景）",
            "metered_bytes_per_episode": "每集 spent_bytes：按发送尝试计量的空气时长字节/集"
                                         "（丢包不退费，airtime_tap_total == spent_bytes 已在门禁闭合）",
            "difference": "逐场景配对后再取均值（同一场景号、同一初始状态、同一世界种子）",
            "timely_target_set": "该臂账本里 credited 的 task_id 集合"},
        "absolute_per_arm": absolute,
        "paired_vs_rules_per_seed_and_mode": paired,
        "seed_consistency_descriptive_only": seed_consistency,
        "reporting_cautions": [
            "区间包含 0 不等于等效", "未观察到任务损失不等于 δ=0 的总体保证",
            "单个 seed 成功不代表训练算法稳定", "不得挑一个显著结果宣称整体方法成立",
            "sampled 每场景一次不是完整采样不确定性评估",
            "各臂的发送尝试次数不同 ⇒ 逐消息丢包实现也不同，只可声称场景配对",
            "本块 64 个场景是新登记块，但一次终评仍不足以支持部署主张"],
        "quota_at_statistics_time": r37.book()}
    r37.write_json("ROUND37_STATISTICS.json", payload, refresh=True)
    print(json.dumps({"cells": len(table), "arms": len(absolute),
                      "primary_table_rows": [f"model_s{seed}_greedy" for seed in SEEDS],
                      "greedy_task_difference_vs_immediate": {
                          f"s{seed}": round(paired[f"model_s{seed}_greedy"]
                                            [RULE_LABEL["immediate"]]
                                            ["task_difference_per_episode"]["mean"], 4)
                          for seed in SEEDS}}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
