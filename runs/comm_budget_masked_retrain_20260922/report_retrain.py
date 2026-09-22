"""Judge the retraining run against the frozen criteria, and render what it measured.

Every number here is sliced out of an artefact another script wrote; nothing is retyped from a
document. Three habits the earlier rounds had to learn the hard way are built in:

* **One bootstrap draw per comparison.** The interval for a seed is produced by a call whose arm set
  is exactly ``{send_once, that arm}`` -- the pilot's routine consumes one draw per arm in dict
  order, so pooling arms makes a number depend on which other arms happened to be in the file.
* **Three seeds are three models, not 72 observations.** Each arm gets its own interval over the 24
  development streams; nothing pools them, and the spread between arms is reported as model variance.
* **Units are named.** A completion difference is 个/episode, a byte difference is B/episode, and a
  proportion is stated with its absolute counts beside it -- the 0.88 % figure from the previous
  round is exactly what a ratio looks like when the denominator is hidden.

    python report_retrain.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import retrain_common as common
import evaluate_retrain as evaluation
import train_retrain as entry

CHECKPOINTS = (0, 100_000, 200_000, 300_000, 400_000)


# ------------------------------------------------------------------ inputs
def training_records() -> dict[int, dict[str, Any]]:
    """One block per training seed: summary, per-update diagnostics, milestone windows."""
    out: dict[int, dict[str, Any]] = {}
    for seed in common.TRAINING_SEEDS:
        directory = common.TRAIN_OUT / f"seed{seed}"
        summary_path = directory / f"maskretrain_seed{seed}_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        updates = [
            json.loads(line)
            for line in (directory / f"updates_seed{seed}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        episodes = [
            json.loads(line)
            for line in (directory / f"episodes_seed{seed}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        milestones = {}
        for at in CHECKPOINTS:
            path = directory / f"diagnostics_at{at}.json"
            if path.exists():
                milestones[at] = json.loads(path.read_text(encoding="utf-8"))
        out[seed] = {
            "summary": summary,
            "updates": updates,
            "episodes": episodes,
            "milestones": milestones,
        }
    return out


def evaluation_rows() -> dict[str, list[dict[str, Any]]]:
    families = {"rules": evaluation.load_rows("rules")}
    for kind in evaluation.CELLS:
        families[f"learned_{kind}"] = evaluation.load_rows(f"learned_{kind}")
    return families


def index_by_cell(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["arm"]), int(row["seed"])): row for row in rows}


# ------------------------------------------------------------------ statistics
def comparisons(families: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Paired intervals: each trained arm and each untrained start against ``send_once``."""
    rules = families["rules"]
    learned = [row for family in ("learned_initial", "learned_curve", "learned_final") for row in families[family]]
    all_rows = rules + learned
    dev_only = [row for row in all_rows if int(row["seed"]) in common.VALIDATION_SEEDS]
    cells = index_by_cell(dev_only)
    output: dict[str, Any] = {
        "per_arm": {},
        "cells": len(cells),
        "development_rows": dev_only,
    }
    arms = sorted({str(row["arm"]) for row in dev_only if row["arm"] != common.MAIN_RULE})
    for arm in arms:
        metrics: dict[str, Any] = {}
        for key, name in (
            ("on_time_targets_per_episode", "on_time"),
            ("airtime_bytes", "airtime_bytes"),
        ):
            metrics[name] = common.paired(
                dev_only, baseline=common.MAIN_RULE, arm=arm, key=key
            )
        streams = sorted({int(row["seed"]) for row in dev_only if row["arm"] == arm})
        metrics["streams"] = len(streams)
        metrics["mean_on_time"] = round(
            float(np.mean([cells[(arm, seed)]["on_time_targets_per_episode"] for seed in streams])),
            5,
        )
        metrics["mean_airtime_bytes"] = round(
            float(np.mean([cells[(arm, seed)]["airtime_bytes"] for seed in streams])), 2
        )
        output["per_arm"][arm] = metrics
    output["baseline"] = {
        "arm": common.MAIN_RULE,
        "streams": len({int(row["seed"]) for row in rules}),
        "mean_on_time": round(
            float(np.mean([row["on_time_targets_per_episode"] for row in rules])), 5
        )
        if rules
        else None,
        "mean_airtime_bytes": round(float(np.mean([row["airtime_bytes"] for row in rules])), 2)
        if rules
        else None,
    }
    return output


def per_stream_movement(families: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Which streams a trained arm rescued and which it lost, against the rule, stream by stream."""
    rules = index_by_cell(families["rules"])
    learned = [
        row
        for family in ("learned_final", "learned_initial")
        for row in families[family]
    ]
    movement: dict[str, Any] = {}
    for row in learned:
        arm = str(row["arm"])
        reference = rules.get((common.MAIN_RULE, int(row["seed"])))
        if reference is None:
            continue
        delta = int(row["on_time_targets_per_episode"]) - int(
            reference["on_time_targets_per_episode"]
        )
        bucket = movement.setdefault(arm, {"rescued": [], "lost": [], "equal": 0, "detail": {}})
        if delta > 0:
            bucket["rescued"].append(int(row["seed"]))
        elif delta < 0:
            bucket["lost"].append(int(row["seed"]))
        else:
            bucket["equal"] += 1
        bucket["detail"][str(row["seed"])] = {
            "on_time_this_arm": int(row["on_time_targets_per_episode"]),
            "on_time_send_once": int(reference["on_time_targets_per_episode"]),
            "bytes_this_arm": int(row["airtime_bytes"]),
            "bytes_send_once": int(reference["airtime_bytes"]),
        }
    for arm, bucket in movement.items():
        bucket["rescued"] = sorted(bucket["rescued"])
        bucket["lost"] = sorted(bucket["lost"])
        bucket["net_streams"] = len(bucket["rescued"]) - len(bucket["lost"])
    return movement


def frontier(families: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """The byte-completion points, and which arms the rule does not strictly dominate."""
    points: dict[str, dict[str, float]] = {}
    for row in [entry for entry in families["rules"]] + [
        row
        for family in ("learned_initial", "learned_final")
        for row in families[family]
    ]:
        arm = str(row["arm"])
        current = points.setdefault(arm, {"on_time": 0.0, "bytes": 0.0, "n": 0})
        current["on_time"] += float(row["on_time_targets_per_episode"])
        current["bytes"] += float(row["airtime_bytes"])
        current["n"] += 1
    summary = {
        arm: {
            "mean_on_time": round(item["on_time"] / item["n"], 5),
            "mean_airtime_bytes": round(item["bytes"] / item["n"], 2),
            "streams": item["n"],
        }
        for arm, item in points.items()
    }
    rule = summary.get(common.MAIN_RULE)
    dominated = []
    if rule:
        for arm, item in summary.items():
            if arm == common.MAIN_RULE:
                continue
            if (
                item["mean_on_time"] <= rule["mean_on_time"]
                and item["mean_airtime_bytes"] >= rule["mean_airtime_bytes"]
            ):
                dominated.append(arm)
    return {"points": summary, "dominated_by_send_once": sorted(dominated)}


# ------------------------------------------------------------------ judgement
def correctness(pre: dict[str, Any], coverage: dict[str, Any], records: dict[int, Any]) -> dict[str, Any]:
    """Every violation this round can be failed for, counted rather than described."""
    violations: dict[str, Any] = {}
    violations["policy_refusals_in_training"] = sum(
        int(block["summary"].get("policy_refusals_total", 0)) for block in records.values()
    )
    violations["requested_minus_executed_in_training"] = sum(
        int(block["summary"].get("requested_total", 0))
        - int(block["summary"].get("executed_total", 0))
        for block in records.values()
    )
    violations["max_log_prob_gap_training"] = max(
        (float(block["summary"].get("likelihood_max_abs_gap", 0.0)) for block in records.values()),
        default=0.0,
    )
    violations["nonfinite_values_in_reports"] = sum(
        int(block["summary"].get("nonfinite_values_recorded", 0)) for block in records.values()
    )
    closure = coverage.get("closure", {})
    violations["evaluation_requested_equals_executed"] = bool(closure.get("requested_equals_executed"))
    violations["evaluation_policy_refusals_zero"] = bool(closure.get("policy_refusals_zero"))
    violations["evaluation_tape_invariants_passed"] = bool(closure.get("tape_invariants_passed"))
    violations["max_log_prob_gap_evaluation"] = float(closure.get("max_log_prob_gap", 0.0))
    violations["episode_length_all_1000"] = bool(closure.get("episode_length_all_1000"))
    violations["sealed_block_artefacts"] = int(
        (pre.get("sealed_block") or {}).get("count", -1)
    )
    violations["shipped_code_changed"] = common.production_diff()["integrated_sar_changed"]
    violations["probe_and_gates_passed"] = bool(pre.get("passed"))
    passed = (
        violations["policy_refusals_in_training"] == 0
        and violations["requested_minus_executed_in_training"] == 0
        and violations["max_log_prob_gap_training"] <= entry.LOG_PROB_TOLERANCE
        and violations["nonfinite_values_in_reports"] == 0
        and violations["max_log_prob_gap_evaluation"] <= entry.LOG_PROB_TOLERANCE
        and violations["sealed_block_artefacts"] == 0
        and violations["evaluation_requested_equals_executed"]
        and violations["evaluation_policy_refusals_zero"]
        and violations["evaluation_tape_invariants_passed"]
        and violations["episode_length_all_1000"]
        and not violations["shipped_code_changed"]
        and violations["probe_and_gates_passed"]
    )
    return {"violations": violations, "passed": bool(passed)}


def judge(stats: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    """§7's two preregistered paths, decided on the 400k checkpoint and nothing else."""
    per_arm = stats["per_arm"]
    seeds = {}
    for seed in common.TRAINING_SEEDS:
        arm = evaluation.arm_label("learned", seed, 400_000)
        initial = evaluation.arm_label("learned", seed, 0)
        if arm not in per_arm:
            seeds[seed] = {"available": False}
            continue
        completion = per_arm[arm]["on_time"]
        bytes_ = per_arm[arm]["airtime_bytes"]
        versus_initial = (
            common.paired(
                stats["development_rows"],
                baseline=initial,
                arm=arm,
                key="on_time_targets_per_episode",
            )
            if initial in per_arm
            else None
        )
        seeds[seed] = {
            "available": True,
            "arm": arm,
            "completion": completion,
            "bytes": bytes_,
            "completion_ci_lower": completion["ci95"][0],
            "completion_point": completion["mean_difference"],
            "bytes_ci_upper": bytes_["ci95"][1],
            "bytes_point": bytes_["mean_difference"],
            "beats_send_once_on_time": completion["ci95"][0] > 0,
            "cheaper_than_send_once": bytes_["ci95"][1] < 0,
            "non_inferior_completion": completion["ci95"][0] >= common.NON_INFERIORITY_MARGIN,
            "versus_own_initial": versus_initial,
        }
    present = [item for item in seeds.values() if item.get("available")]
    decisive = [item for item in present if item["beats_send_once_on_time"]]
    others_ok = all(
        item["completion_point"] >= common.NON_INFERIORITY_MARGIN
        for item in present
        if not item["beats_send_once_on_time"]
    )
    cost_decisive = [
        item for item in present
        if item["non_inferior_completion"] and item["cheaper_than_send_once"]
    ]
    cost_others_ok = all(
        item["completion_point"] >= common.NON_INFERIORITY_MARGIN
        and item["bytes_point"] <= 0
        for item in present
        if not (item["non_inferior_completion"] and item["cheaper_than_send_once"])
    )
    primary = bool(
        len(present) == len(common.TRAINING_SEEDS)
        and len(decisive) >= 2
        and others_ok
        and gates["passed"]
    )
    cost = bool(
        len(present) == len(common.TRAINING_SEEDS)
        and len(cost_decisive) >= 2
        and cost_others_ok
        and gates["passed"]
    )
    return {
        "criterion": "400k end-of-run checkpoint only; no middle checkpoint selects a model",
        "per_seed": seeds,
        "primary_effect": {
            "rule": "at least 2/3 seeds with completion-difference CI lower bound > 0, "
            "the remaining seed's point estimate >= -0.05 targets/episode, zero correctness violations",
            "seeds_with_positive_lower_bound": [item["arm"] for item in decisive],
            "seeds": len(decisive),
            "other_seeds_within_margin": others_ok,
            "passed": primary,
        },
        "non_inferiority_cost": {
            "rule": "at least 2/3 seeds with completion CI lower bound >= -0.05 and byte CI upper "
            "bound < 0, the remaining seed within the margin and no more expensive, zero violations",
            "seeds": len(cost_decisive),
            "other_seeds_within_margin": cost_others_ok,
            "passed": cost,
        },
        "mechanism_evidence_only": [
            "learning versus its own initial weights",
            "a falling training loss",
            "a changed send rate",
        ],
    }


# ------------------------------------------------------------------ curves
def curves(records: dict[int, Any], families: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """The per-checkpoint training and evaluation readings, kept as a curve and never as a choice."""
    rows_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for family in ("learned_initial", "learned_curve", "learned_final"):
        for row in families.get(family, []):
            rows_by_arm[str(row["arm"])].append(row)
    out: dict[str, Any] = {}
    for seed, block in records.items():
        points: dict[str, Any] = {}
        for at in CHECKPOINTS:
            readings = rows_by_arm.get(evaluation.arm_label("learned", seed, at), [])
            streams = sorted({int(row["stream"]) for row in readings}) if readings else []
            points[str(at)] = {
                "training_diagnostics": block["milestones"].get(at),
                "evaluation": {
                    "streams": len(streams),
                    "stream_block": (
                        "development (24)"
                        if len(streams) == len(common.VALIDATION_SEEDS)
                        else "curve subset (8)"
                        if len(streams) == common.CURVE_STREAM_COUNT
                        else "incomplete"
                    ),
                    "mean_on_time": round(
                        float(np.mean([row["on_time"] for row in readings])), 5
                    )
                    if readings
                    else None,
                    "mean_airtime_bytes": round(
                        float(np.mean([row["bytes"] for row in readings])), 2
                    )
                    if readings
                    else None,
                },
            }
        out[str(seed)] = {
            "summary": {
                key: block["summary"][key]
                for key in (
                    "environment_steps",
                    "episodes",
                    "updates",
                    "optimizer_steps_cumulative",
                    "actor_terms_total",
                    "actionable_total",
                    "candidates_total",
                    "masked_by_credit_total",
                    "episodes_without_actionable_total",
                    "requested_total",
                    "executed_total",
                    "policy_refusals_total",
                    "likelihood_max_abs_gap",
                    "initial_weights_fingerprint",
                    "final_weights_fingerprint",
                    "wall_seconds",
                    "resumed",
                )
                if key in block["summary"]
            },
            "checkpoints": points,
            "arm_family": f"masked_s{seed}",
        }
    return out


# ------------------------------------------------------------------ rendering
def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:,.0f}"
    return str(value)


def render(data: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append
    judgement = data["judgement"]
    gates = data["correctness"]
    stats = data["statistics"]
    add("# 预算 mask 修复后的从初值重训：三 seed × 400k 环境步结果")
    add("")
    add(
        f"日期：{data['date']}。分支 `{data['branch']}`，code-freeze "
        f"`{str(data['code_freeze_sha'])[:9]}`，结果记录头 `{data['head'][:9]}`，总纲 v3.1，"
        f"执行指令 `{data['brief']}`，协议 `{data['protocol']}`（先于第一个训练 episode 提交）。"
    )
    add("")
    setting = data["setting"]
    add(
        f"**唯一主设置**：{setting['budget_bytes']:,} B/episode、`{setting['timing_mode']}`、"
        f"{setting['information']} 信息路径、丢包 0（投递概率 {setting['delivery_probability']}）、"
        f"ACK 关闭、`{setting['feature_layout']}`（{setting['feature_dimension']} 维）、"
        f"trainer `{setting['trainer_tag']}` / mask `{setting['action_mask_schema']}` / "
        f"tape `{setting['tape_schema']}`。三个臂都从 `Controller(seed)` 随机初值开始，"
        f"禁止加载旧 checkpoint。"
    )
    add("")
    measured = data["measured"]
    add(
        f"**本轮实测**：训练 {measured['training_runs']} 次 × {measured['episodes_per_run']} 集 = "
        f"{measured['environment_steps']:,} 环境步；启动登记 {measured['launches_booked']}/"
        f"{measured['launch_cap']}（探针 {measured['probe']}、确定性恢复 "
        f"{measured['resume_check']}、评价 {measured['evaluation']}、规则重放 "
        f"{measured['rule_replay']}、技术失败 {measured['technical_failures']}）；"
        f"optimizer.step 合计 {measured['optimizer_steps']:,}；actor 项合计 "
        f"{measured['actor_terms']:,}；候选 {measured['candidates']:,}，其中顺序后可执行 "
        f"{measured['actionable']:,}（{measured['actionable_share']:.2%}，绝对量 "
        f"{measured['actionable_per_episode']:.1f}/episode）。新流 "
        f"{measured['new_streams']}；H=20 spillover {measured['h20']}。"
    )
    add("")
    add("## 一、判定（预登记，400k 末轮唯一主判据）")
    add("")
    add("| 门槛 | 结果 | 判据 |")
    add("|---|---|---|")
    add(
        f"| 主效果（学习 vs `send_once` 按时完成） | "
        f"**{'通过' if judgement['primary_effect']['passed'] else '未通过'}** | "
        f"{judgement['primary_effect']['rule']}；实到 "
        f"{judgement['primary_effect']['seeds']}/3 个 seed 的 CI 下界 > 0 |"
    )
    add(
        f"| 非劣降本 | **{'通过' if judgement['non_inferiority_cost']['passed'] else '未通过'}** | "
        f"{judgement['non_inferiority_cost']['rule']}；实到 "
        f"{judgement['non_inferiority_cost']['seeds']}/3 |"
    )
    add(
        f"| 正确性门禁 | **{'通过' if gates['passed'] else '未通过'}** | "
        f"policy 拒付 {gates['violations']['policy_refusals_in_training']}、"
        f"请求-执行差 {gates['violations']['requested_minus_executed_in_training']}、"
        f"log-prob 最大差（训练）{gates['violations']['max_log_prob_gap_training']:.1e}、"
        f"（评价）{gates['violations']['max_log_prob_gap_evaluation']:.1e}、"
        f"NaN/Inf {gates['violations']['nonfinite_values_in_reports']}、封存块 "
        f"{gates['violations']['sealed_block_artefacts']} |"
    )
    add("")
    add("### 逐 seed 的配对区间（24 条开发流，B=10 000，RNG=20260922）")
    add("")
    add("| 训练 seed | 完成差（个/ep） | 95% CI | 字节差（B/ep） | 95% CI | CI 下界>0 | 字节 CI 上界<0 |")
    add("|---|---:|---|---:|---|---|---|")
    for seed, item in judgement["per_seed"].items():
        if not item.get("available"):
            add(f"| {seed} | 未评价 | — | — | — | — | — |")
            continue
        add(
            f"| {seed} | {item['completion_point']:+.4f} | "
            f"[{item['completion_ci_lower']:+.4f}, "
            f"{item['completion']['ci95'][1]:+.4f}] | {item['bytes_point']:+,.1f} | "
            f"[{item['bytes']['ci95'][0]:+,.1f}, {item['bytes_ci_upper']:+,.1f}] | "
            f"{'是' if item['beats_send_once_on_time'] else '否'} | "
            f"{'是' if item['cheaper_than_send_once'] else '否'} |"
        )
    add("")
    add(
        f"- 对照 `send_once` 本轮重放：{stats['baseline']['streams']} 条流，"
        f"平均 {stats['baseline']['mean_on_time']} 个/episode、"
        f"{stats['baseline']['mean_airtime_bytes']:,} B/episode。"
    )
    agreement = data["rule_replay"]
    add(
        f"- 归档一致性：逐流任务结果/字节/发送序列差异 "
        f"{len(agreement.get('field_differences', {}))} 处、序列差异 "
        f"{len(agreement.get('emission_sequence_differences', {}))} 处，"
        f"passed=`{agreement.get('passed')}`（对照 "
        f"`{agreement.get('archived_rows')}`）。"
    )
    add("")
    add("### 学习 vs 自身 0 步初值（机制证据，不是门槛）")
    add("")
    add("| 训练 seed | 0 步完成 | 400k 完成 | 差（个/ep） | 差 95% CI | 0 步字节 | 400k 字节 |")
    add("|---|---:|---:|---:|---|---:|---:|")
    for seed, item in judgement["per_seed"].items():
        initial = data["initial_reads"].get(str(seed))
        final = data["final_reads"].get(str(seed))
        versus = item.get("versus_own_initial") if item.get("available") else None
        if not initial or not final or not versus:
            add(f"| {seed} | — | — | — | — | — | — |")
            continue
        add(
            f"| {seed} | {initial['mean_on_time']} | {final['mean_on_time']} | "
            f"{versus['mean_difference']:+.4f} | [{versus['ci95'][0]:+.4f}, "
            f"{versus['ci95'][1]:+.4f}] | {initial['mean_airtime_bytes']:,} | "
            f"{final['mean_airtime_bytes']:,} |"
        )
    add("")
    add("## 二、样本与优化诊断（绝对量）")
    add("")
    add("| 训练 seed | 环境步 | episode | update | optimizer.step | 候选 | 可执行 | actor 项 | "
        "被额度 mask | 零可执行 episode | 请求 | 执行 | policy 拒付 | log-prob 差 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for seed, block in data["training_table"].items():
        add(
            f"| {seed} | {block['environment_steps']:,} | {block['episodes']} | "
            f"{block['updates']} | {block['optimizer_steps']:,} | {block['candidates']:,} | "
            f"{block['actionable']:,} | {block['actor_terms']:,} | {block['masked']:,} | "
            f"{block['zero_actionable_episodes']} | {block['requested']:,} | "
            f"{block['executed']:,} | {block['refusals']} | {block['gap']:.1e} |"
        )
    add("")
    add("### 每 checkpoint 的分布（episode 内 actionable、消息类型、发送时间）")
    add("")
    add("| seed | checkpoint | actionable 均值 | 中位 | p25–p75 | 零可执行 episode | "
        "首报/更新/重试 | 随机 SEND 率 | greedy SEND 率 | 每候选熵 | value loss | EV | KL | clip |")
    add("|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in data["milestone_table"]:
        add(
            f"| {row['seed']} | {row['at']:,} | {row['actionable_mean']} | {row['actionable_median']} | "
            f"{row['iqr']} | {row['zero_episodes']} | {row['messages']} | {row['send_rate']} | "
            f"{row['greedy_send_rate']} | {row['entropy']} | {row['value_loss']} | {row['ev']} | "
            f"{row['kl']} | {row['clip']} |"
        )
    add("")
    add("## 三、曲线（中间 checkpoint 只作曲线，不参与选择）")
    add("")
    add("| seed | 0 步 | 100k | 200k | 300k | 400k | 评价流数 |")
    add("|---|---:|---:|---:|---:|---:|---|")
    for seed, block in data["curves_summary"].items():
        cells = []
        counts = []
        for at in CHECKPOINTS:
            point = block["checkpoints"][str(at)]["evaluation"]
            cells.append(
                "—" if point["mean_on_time"] is None else f"{point['mean_on_time']}"
            )
            counts.append(f"{at // 1000}k:{point['streams']}")
        add(f"| {seed} | {' | '.join(cells)} | {' '.join(counts)} |")
    add("")
    add("## 四、逐流得失与 Pareto")
    add("")
    for arm, bucket in data["movement"].items():
        add(
            f"- `{arm}`：救回 {len(bucket['rescued'])} 条 "
            f"{bucket['rescued']}，新增失败 {len(bucket['lost'])} 条 {bucket['lost']}，"
            f"净 {bucket['net_streams']:+d}（相等 {bucket['equal']}）。"
        )
    add("")
    add("| 臂 | 平均完成（个/ep） | 平均字节（B/ep） | 流数 |")
    add("|---|---:|---:|---:|")
    for arm, item in data["frontier"]["points"].items():
        add(
            f"| `{arm}` | {item['mean_on_time']} | {item['mean_airtime_bytes']:,} | "
            f"{item['streams']} |"
        )
    add("")
    add(
        f"- 被 `send_once` 在两个轴上同时不优于它的臂："
        f"`{data['frontier']['dominated_by_send_once']}`。"
    )
    add("")
    add("## 五、结论范围")
    add("")
    for line in data["conclusions"]:
        add(f"- {line}")
    add("")
    add("## 六、卫生与身份")
    add("")
    add(f"- code-freeze `{str(data['code_freeze_sha'])[:9]}`：执行图 {data['graph_members']} 个模块，"
        f"相对 freeze 干净 = `{data['cleanliness']}`；未跟踪执行模块 "
        f"`{data['untracked_code']}`。")
    add(
        f"- 生产代码自 `e692607` 起改动：`{data['production_changed']}`（应为空）；"
        f"生产默认值 comm_ack=`{data['defaults']['comm_ack']}`、"
        f"timing=`{data['defaults']['comm_timing_mode']}`。"
    )
    add(
        f"- 封存块 `800000..800399` 盘上 artefact "
        f"`{gates['violations']['sealed_block_artefacts']}`；`900000..900199` 未进入本轮。"
    )
    add("- checkpoint、录制流、逐 episode 原始日志均不入库。")
    add("")
    add("## 七、复跑")
    add("")
    add("```bash")
    add("cd runs/comm_budget_masked_retrain_20260922")
    add("python train_retrain.py --mode identity")
    add("python preflight.py                      # 12 探针 + 8 恢复回归 + 1 greedy 路径，0 训练")
    add("python train_retrain.py --mode train --seed 1 --steps 400000 --threads 1   # 2/3 同理")
    add("python evaluate_retrain.py --mode rules --shards 3 --shard 0")
    add("python evaluate_retrain.py --mode eval --kind initial --shards 3 --shard 0")
    add("python evaluate_retrain.py --mode eval --kind curve   --shards 3 --shard 0")
    add("python evaluate_retrain.py --mode eval --kind final   --shards 3 --shard 0")
    add("python evaluate_retrain.py --mode replay-check")
    add("python report_retrain.py")
    add("cd ../.. && python -m unittest discover -s integrated_sar/tests && python -m integrated_sar")
    add("```")
    add("")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ assembly
def assemble() -> dict[str, Any]:
    records = training_records()
    families = evaluation_rows()
    pre = common.read_json("PREFLIGHT.json") if common.PREFLIGHT_FILE.exists() else {}
    coverage = evaluation.coverage()
    stats = comparisons(families)
    gates = correctness(pre, coverage, records)
    judgement = judge(stats, gates)
    rule_replay = evaluation.rule_replay_agreement()
    movement = per_stream_movement(families)
    curve_data = curves(records, families)
    freeze = common.freeze()
    throughput = (
        common.read_json("THROUGHPUT.json")
        if common.json_path("THROUGHPUT.json").exists()
        else {}
    )
    resume = (
        common.read_json("RESUME_CHECK.json")
        if common.json_path("RESUME_CHECK.json").exists()
        else {}
    )
    ledger = json.loads(common.LAUNCH_LEDGER.read_text(encoding="utf-8"))

    training_table = {}
    milestone_table = []
    final_reads = {}
    initial_reads = {}
    for seed, block in records.items():
        summary = block["summary"]
        training_table[seed] = {
            "environment_steps": int(summary["environment_steps"]),
            "episodes": int(summary["episodes"]),
            "updates": int(summary["updates"]),
            "optimizer_steps": int(summary["optimizer_steps_cumulative"]),
            "candidates": int(summary.get("candidates_total", 0)),
            "actionable": int(summary.get("actionable_total", 0)),
            "actor_terms": int(summary.get("actor_terms_total", 0)),
            "masked": int(summary.get("masked_by_credit_total", 0)),
            "zero_actionable_episodes": int(summary.get("episodes_without_actionable_total", 0)),
            "requested": int(summary.get("requested_total", 0)),
            "executed": int(summary.get("executed_total", 0)),
            "refusals": int(summary.get("policy_refusals_total", 0)),
            "gap": float(summary.get("likelihood_max_abs_gap", 0.0)),
        }
        for at, milestone in sorted(block["milestones"].items()):
            actionable = milestone.get("actionable_per_episode") or {}
            messages = milestone.get("message_types") or {}
            last = block["updates"][-1] if block["updates"] else {}
            window = [
                item for item in block["updates"] if int(item["at_steps"]) <= int(at)
            ]
            tail = window[-1] if window else {}
            milestone_table.append(
                {
                    "seed": seed,
                    "at": int(at),
                    "actionable_mean": actionable.get("mean"),
                    "actionable_median": actionable.get("median"),
                    "iqr": f"{actionable.get('p25')}-{actionable.get('p75')}",
                    "zero_episodes": actionable.get("n", 0) and milestone.get(
                        "episodes_without_actionable"
                    ),
                    "messages": (
                        f"{messages.get('first', 0)}/{messages.get('update', 0)}/"
                        f"{messages.get('retry_same_view', 0)}"
                    ),
                    "send_rate": tail.get("stochastic_send_rate"),
                    "greedy_send_rate": tail.get("greedy_send_rate"),
                    "entropy": tail.get("stochastic_entropy_per_actionable_candidate"),
                    "value_loss": tail.get("value_loss"),
                    "ev": tail.get("value_explained_variance"),
                    "kl": tail.get("approx_kl"),
                    "clip": tail.get("clip_fraction"),
                }
            )
    for family, target in (("learned_initial", initial_reads), ("learned_final", final_reads)):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in families[family]:
            grouped[str(row["training_seed"])].append(row)
        for seed, rows in grouped.items():
            target[seed] = {
                "mean_on_time": round(
                    float(np.mean([row["on_time_targets_per_episode"] for row in rows])), 5
                ),
                "mean_airtime_bytes": round(
                    float(np.mean([row["airtime_bytes"] for row in rows])), 2
                ),
                "streams": len(rows),
            }
    candidates = sum(int(block["summary"].get("candidates_total", 0)) for block in records.values())
    actionable = sum(int(block["summary"].get("actionable_total", 0)) for block in records.values())
    episodes = sum(int(block["summary"].get("episodes", 0)) for block in records.values())
    data = {
        "document": "COMM_BUDGET_MASKED_RETRAIN_RESULTS",
        "schema": entry.SCHEMA,
        "date": "2026-09-22",
        "branch": common.git_out("rev-parse", "--abbrev-ref", "HEAD"),
        "head": common.git_out("rev-parse", "HEAD"),
        "code_freeze_sha": freeze.get("sha"),
        "configuration_hash": entry.configuration_hash(),
        "brief": "COMM_BUDGET_MASKED_RETRAIN_NEXT_ROUND.md",
        "protocol": "COMM_BUDGET_MASKED_RETRAIN_PROTOCOL.md",
        "charter_version": "v3.1",
        "setting": common.setting(),
        "measured": {
            "training_runs": len(records),
            "episodes_per_run": episodes // max(1, len(records)),
            "environment_steps": sum(
                int(block["summary"].get("environment_steps", 0)) for block in records.values()
            ),
            "optimizer_steps": sum(
                int(block["summary"].get("optimizer_steps_cumulative", 0))
                for block in records.values()
            ),
            "actor_terms": sum(
                int(block["summary"].get("actor_terms_total", 0)) for block in records.values()
            ),
            "candidates": candidates,
            "actionable": actionable,
            "actionable_share": actionable / max(1, candidates),
            "actionable_per_episode": actionable / max(1, episodes),
            "launches_booked": int(ledger.get("starts", 0)),
            "launch_cap": int(ledger.get("cap", common.LAUNCH_CAP)),
            "probe": int(ledger.get("by_kind", {}).get("probe", 0)),
            "resume_check": int(ledger.get("by_kind", {}).get("resume_check", 0)),
            "evaluation": sum(
                count
                for kind, count in ledger.get("by_kind", {}).items()
                if kind.startswith("eval_")
            ),
            "rule_replay": int(ledger.get("by_kind", {}).get("rule", 0)),
            "technical_failures": int(ledger.get("failures", 0)),
            "new_streams": 0,
            "h20": 0,
        },
        "throughput": throughput.get("throughput"),
        "deterministic_resume": {
            "passed": resume.get("passed"),
            "weights_identical": resume.get("weights_identical"),
            "optimizer_identical": resume.get("optimizer_identical"),
        },
        "statistics": stats,
        "judgement": judgement,
        "correctness": gates,
        "rule_replay": rule_replay,
        "movement": movement,
        "frontier": frontier(families),
        "training_table": training_table,
        "milestone_table": milestone_table,
        "curves": curve_data,
        "curves_summary": curve_data,
        "initial_reads": initial_reads,
        "final_reads": final_reads,
        "cleanliness": common.freeze_cleanliness(common.executed_modules())["clean"],
        "graph_members": freeze.get("module_count"),
        "untracked_code": common.untracked_code(common.executed_modules()),
        "production_changed": common.production_diff()["integrated_sar_changed"],
        "defaults": common.setting()["production_defaults"],
        "conclusions": data_conclusions(judgement, gates, rule_replay),
        "not_measured": [
            "任何独立测试集结论：800000..800399 仍封存，本轮未打开",
            "第二预算、ACK、H=20、observation 扩维：均未授权、未运行",
            "把 3 个训练 seed 当成 72 个独立样本：每个 arm 单独给区间",
        ],
    }
    return data


def data_conclusions(
    judgement: dict[str, Any], gates: dict[str, Any], replay: dict[str, Any]
) -> list[str]:
    lines: list[str] = []
    if not replay.get("passed"):
        lines.append(
            "规则对照在本轮代码下与归档不一致，本轮结论挂起：必须先归因，不得更换更弱的规则。"
        )
        return lines
    if judgement["primary_effect"]["passed"]:
        lines.append(
            "主效果门槛通过：在修复动作空间、29 300 B 主预算、原 19 维观测/奖励与 400k 步下，"
            "至少 2/3 个 seed 的按时完成差 95% CI 下界 > 0，另一个 seed 点估计不在劣侧界外。"
            "这只说明**开发集候选成立**，下一步是独立测试协议草案，不是自动打开封存块。"
        )
    if judgement["non_inferiority_cost"]["passed"]:
        lines.append(
            "非劣降本门槛通过：至少 2/3 个 seed 完成差不劣且字节差 CI 上界 < 0。"
            "这是降本结论，同样只限开发集。"
        )
    if not judgement["primary_effect"]["passed"] and not judgement["non_inferiority_cost"]["passed"]:
        lines.append(
            "两条门槛都未通过。结论限定为：**在修复动作空间、原 19 维观测/奖励与总 1.2M 训练步下，"
            "学习附加价值仍未建立。**不得据此加训练量、改特征、换预算、加 ACK 或恢复 H=20。"
        )
    if not gates["passed"]:
        lines.append(
            "正确性门禁存在违规：本轮不得给出效果结论，先修接缝，失败运行隔离后从 0 重开。"
        )
    lines.append(
        "0 步初值与 400k 的差、训练 loss 与发送率变化只作机制证据：它们说明优化器确实在动，"
        "不能替代与强规则的对照。"
    )
    return lines


def main(argv: list[str]) -> int:
    data = assemble()
    common.write_json(common.json_path("COMM_BUDGET_MASKED_RETRAIN_RESULTS.json"), data)
    common.write_json(common.json_path("TRAINING_CURVES.json"), data["curves"])
    common.json_path("COMM_BUDGET_MASKED_RETRAIN_RESULTS.md").write_text(
        render(data), encoding="utf-8"
    )
    manifest = {
        "document": "COMM_BUDGET_MASKED_RETRAIN_MANIFEST",
        "date": "2026-09-22",
        "branch": data["branch"],
        "head": data["head"],
        "code_freeze": common.freeze(),
        "cleanliness_now": common.freeze_cleanliness(common.executed_modules()),
        "configuration_hash": data["configuration_hash"],
        "trainer": {
            "trainer_tag": entry.TRAINER_TAG,
            "action_mask_schema": entry.ACTION_MASK_SCHEMA,
            "tape_schema": entry.TAPE_SCHEMA,
        },
        "setting": data["setting"],
        "streams": common.verify_streams(),
        "sealed_block": common.sealed_block_scan(),
        "production_diff": common.production_diff(),
        "launches": json.loads(common.LAUNCH_LEDGER.read_text(encoding="utf-8")),
        "evaluation_coverage": evaluation.coverage(),
        "rule_replay_agreement": data["rule_replay"],
        "preflight": {
            key: value
            for key, value in (
                json.loads(common.PREFLIGHT_FILE.read_text(encoding="utf-8"))
                if common.PREFLIGHT_FILE.exists()
                else {}
            ).items()
            if key
            in (
                "passed",
                "all_mask_gates_passed",
                "initial_weights",
                "deterministic_resume",
                "greedy_evaluation_path",
                "throughput",
                "probe_and_resume_reused_from_disk",
            )
        },
        "artefacts_in_git": sorted(
            path.name
            for path in common.HERE.glob("*")
            if path.suffix in {".py", ".md", ".json"}
        ),
        "not_committed": [
            "checkpoints (.pt)",
            "recorded streams",
            "per-episode lifecycle/decision/transcript logs (.jsonl)",
            "evaluation row files (.jsonl)",
            "other sessions' untracked files",
        ],
        "delivered": {
            "provenance_correction": "COMM_BUDGET_MASK_PROVENANCE_CORRECTION.md/.json",
            "protocol": "COMM_BUDGET_MASKED_RETRAIN_PROTOCOL.md",
            "preflight": "PREFLIGHT.json",
            "throughput": "THROUGHPUT.json",
            "resume_check": "RESUME_CHECK.json",
            "curves": "TRAINING_CURVES.json",
            "results": "COMM_BUDGET_MASKED_RETRAIN_RESULTS.md/.json",
        },
    }
    common.write_json(common.json_path("COMM_BUDGET_MASKED_RETRAIN_MANIFEST.json"), manifest)
    print(
        json.dumps(
            {
                "primary_effect_passed": data["judgement"]["primary_effect"]["passed"],
                "non_inferiority_cost_passed": data["judgement"]["non_inferiority_cost"]["passed"],
                "correctness_passed": data["correctness"]["passed"],
                "rule_replay_passed": data["rule_replay"].get("passed"),
                "launches": data["measured"]["launches_booked"],
                "environment_steps": data["measured"]["environment_steps"],
                "actor_terms": data["measured"]["actor_terms"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
