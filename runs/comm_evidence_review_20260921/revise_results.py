"""Rebuild the rule-confirmation report from ONE seeded statistics object, with corrected wording.

No episode is run here. The inputs are the archived 3,000 rows (read-only), and the outputs are
``COMM_RULE_CONFIRMATION_RESULTS_REVISED.json`` / ``.md``. The archived documents stay exactly
as they were committed; the differences between them and this revision are recorded in
``old_output_diff`` here and in ``COMM_EVIDENCE_CORRECTIONS.md``.

Three structural rules this file exists to enforce:

* one statistic, one bootstrap sample: every paired difference is drawn once per (setting,
  axis, baseline) over the whole arm set, and the judgement sections slice those same blocks --
  the archived report drew twice for the same quantity and the two draws disagreed in their
  CI bounds (``COMM_RESEARCH_REVIEW_NEXT_ROUND.md`` §3.5);
* a byte difference is never "better": cost axes render 更高/相等/更低, only the completion
  axis renders 更好/相等/更差;
* a ratio is reported beside its own raw numerator and denominator, and an interval built
  after dropping non-positive-denominator replicates is labelled a conditional interval.

    python revise_results.py

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import review_common as common

#: The archived round's own statistics module, reached through the fixture that sets sys.path:
#: the aggregates, delta vectors, ratio math, failure accounting and row validation stay in
#: exactly one implementation, the one that produced the archived numbers.
cm = common.confirm_matrix

NON_INFERIORITY_MARGIN = cm.NON_INFERIORITY_MARGIN
RECHECK_BOOTSTRAP_SEED = 20260921  # this round's own re-check seed, never the frozen one

#: The quantities the report is allowed to call "supported", in which direction.
EFFECT_LABELS = {"up": "更好", "down": "更差", "flat": "相等"}
COST_LABELS = {"up": "更高", "down": "更低", "flat": "相等"}

UNIT_LEGEND = (
    "个任务/episode（完成数差）、B/episode（每局字节差）、B/新增按时任务（比值）、"
    "完成率（[0,1] 比例，不是百分点）。四者不可互换：+0.835 是个/ep，+26 354 是 B/ep，"
    "31 562 是 B/新增任务，不是 B/ep。"
)


# --------------------------------------------------------------------------- the one object
def effect_entry(block: dict[str, Any]) -> dict[str, Any]:
    """The bootstrap block for one (arm - baseline) on the completion axis, with readings."""
    lower, upper = block["ci95_lower"], block["ci95_upper"]
    if lower is None:
        direction = "unknown"
    elif lower > 0.0:
        direction = "supported_higher"
    elif upper is not None and upper < 0.0:
        direction = "supported_lower"
    else:
        direction = "undetected"
    return {
        **block,
        "direction": direction,
        "ci_contains_zero": bool(lower is not None and upper is not None and lower <= 0.0 <= upper),
        "passes_non_inferiority": bool(lower is not None and lower >= NON_INFERIORITY_MARGIN),
        "equivalence_established": False,  # never true from a bootstrap CI alone
    }


def cost_entry(block: dict[str, Any]) -> dict[str, Any]:
    """The bootstrap block for one (arm - baseline) on a byte axis, with readings.

    ``seeds_better``/``seeds_worse`` from ``evaluate_pilot.paired`` count a positive and a
    negative difference. On a cost axis a positive difference is "dearer", so the same
    integers are re-labelled 更高/更低 and the direction never reads as an advantage.
    """
    lower, upper = block["ci95_lower"], block["ci95_upper"]
    if lower is None:
        direction = "unknown"
    elif lower > 0.0:
        direction = "supported_dearer"
    elif upper is not None and upper < 0.0:
        direction = "supported_cheaper"
    else:
        direction = "undetected"
    return {
        **block,
        "direction": direction,
        "higher_streams": int(block.get("seeds_better") or 0),
        "equal_streams": int(block.get("seeds_equal") or 0),
        "lower_streams": int(block.get("seeds_worse") or 0),
    }


def comparisons_for_setting(scoped: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Every arm against both frozen baselines, drawn once per axis over the full arm set.

    ``evaluate_pilot.paired`` reseeds per call and consumes one draw block per arm in row
    order, so the only call shape that reproduces the archived comparison tables -- and gives
    the judgement the same sample -- is the full arm set, once per axis.
    """
    blocks: dict[str, dict[str, Any]] = {}
    for baseline in ("noack_send_once", "ack_send_once", "ack_retry_2"):
        for axis in common.confirm_matrix.AXES:
            drawn = common.paired(scoped, baseline=baseline, key=axis)
            for arm, entry in drawn.get("arms", {}).items():
                key = f"{arm} - {baseline}"
                axis_entry = {
                    "arm": arm,
                    "baseline": baseline,
                    "key": axis,
                    "pairs": int(entry.get("pairs", 0)),
                    "mean_difference": entry.get("mean_difference"),
                    "ci95_lower": (entry.get("ci95") or [None, None])[0],
                    "ci95_upper": (entry.get("ci95") or [None, None])[1],
                    "seeds_better": entry.get("seeds_better"),
                    "seeds_worse": entry.get("seeds_worse"),
                    "seeds_equal": entry.get("seeds_equal"),
                }
                blocks.setdefault(key, {"axes": {}})["axes"][axis] = (
                    effect_entry(axis_entry) if axis == "on_time" else cost_entry(axis_entry)
                )
    for key, block in blocks.items():
        arm, _, baseline = key.partition(" - ")
        block["arm"] = arm
        block["baseline"] = baseline
        block["pairs"] = block["axes"]["on_time"]["pairs"]
    return blocks


def ratio_block(deltas: dict[str, np.ndarray], axes: dict[str, Any]) -> dict[str, Any]:
    """The frozen ratio, plus the flags that stop a conditional interval reading as unconditional."""
    block = dict(cm.ratio_bootstrap(deltas))
    undefined = int(block.get("undefined_replicates") or 0)
    if not block.get("defined"):
        block["interval_kind"] = "not_applicable"
        block["robustly_interpretable"] = False
        block["note"] = (
            "点估计分母非正 ⇒ undefined；成本差与效果差本身照常报告，不删除样本后给出比值"
        )
        return block
    completion = axes["on_time"]
    effect_supported = completion["direction"] == "supported_higher"
    block["interval_kind"] = "unconditional" if undefined == 0 else "conditional"
    block["robustly_interpretable"] = bool(undefined == 0 and effect_supported)
    notes: list[str] = []
    if undefined:
        notes.append(
            f"分位数只在 {10000 - undefined}/10000 个 defined replicate 上计算：条件区间，"
            "不是无条件 95 % CI"
        )
    if not effect_supported:
        notes.append(
            "分母（完成数差）的 CI 含 0：比值符号依赖少量分母的抽样，不可稳健解释，"
            "应引用原始的成本差与效果差"
        )
    if notes:
        block["note"] = "；".join(notes)
    return block


def sample_mean_frontier(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Definition 1 of three: dominance between the per-arm means of this setting."""
    arms = sorted(summary)
    dominated: dict[str, list[str]] = {}
    for arm in arms:
        for other in arms:
            if other == arm:
                continue
            if (
                summary[other]["on_time_per_episode"] >= summary[arm]["on_time_per_episode"]
                and summary[other]["total_bytes"] <= summary[arm]["total_bytes"]
                and (
                    summary[other]["on_time_per_episode"] > summary[arm]["on_time_per_episode"]
                    or summary[other]["total_bytes"] < summary[arm]["total_bytes"]
                )
            ):
                dominated.setdefault(arm, []).append(other)
    return {
        "definition": "样本均值前沿：一个臂的均值 onT 与均值 total bytes 被另一臂同时不劣且至少一项严格更优",
        "frontier": [arm for arm in arms if arm not in dominated],
        "dominated": {arm: sorted(sources) for arm, sources in sorted(dominated.items())},
    }


def supported_pairwise(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Definition 3 of three: only differences the frozen bootstrap supports in both directions."""
    supported = []
    for key, block in sorted(comparisons.items()):
        effect = block["axes"]["on_time"]
        cost = block["axes"]["total_bytes"]
        if effect["direction"] == "supported_higher" and cost["direction"] == "supported_cheaper":
            supported.append(
                {
                    "key": key,
                    "reading": (
                        f"{block['arm']} 在完成数上更高（CI 下界 > 0）且 total bytes 更低"
                        "（CI 上界 < 0）⇒ 有统计支持的双轴支配"
                    ),
                }
            )
    return {
        "definition": "有统计支持的支配：完成数差 CI 下界 > 0 且 total-byte 差 CI 上界 < 0（同一冻结 bootstrap 抽样）",
        "supported": supported,
        "note": "本定义下未被支持的配对不是'等效'，只是未检出差异",
    }


# --------------------------------------------------------------------------- the judgement
def judge(setting: str, comparisons: dict[str, dict[str, Any]], failures: dict) -> dict[str, Any]:
    """The protocol's four readings, sliced from the one object rather than recomputed."""
    verdict: dict[str, Any] = {"setting": setting}
    locked = "ack_retry_2 - noack_send_once"
    completion = comparisons[locked]["axes"]["on_time"]
    total = comparisons[locked]["axes"]["total_bytes"]
    retry_improves = completion["direction"] == "supported_higher"

    if setting == "isolated-lossless":
        checks: dict[str, Any] = {}
        for arm in cm.NEGATIVE_CONTROL_ARMS:
            key = f"{arm} - noack_send_once"
            block = comparisons[key]
            checks[arm] = {
                "completion": block["axes"]["on_time"],
                "improves": block["axes"]["on_time"]["direction"] == "supported_higher",
                "total_bytes": block["axes"]["total_bytes"],
            }
        violations = [arm for arm, block in checks.items() if block["improves"]]
        verdict["negative_control"] = {
            "expectation": "no ACK rule should gain completions where nothing is ever lost",
            "arms": checks,
            "violations": violations,
            "passed": not violations,
            "action": (
                "pause and check whether the ACK or the rule changed behaviour beyond the "
                "feedback itself"
                if violations
                else "proceed: the negative control behaved as the protocol expects"
            ),
            "note": (
                "the ACK arms' higher total bytes on this setting are the protocol's cost, "
                "not a pass and not a failure"
            ),
        }
    if setting == "isolated-lossy":
        verdict["lossy_confirmation"] = {
            "completion": completion,
            "total_bytes": total,
            "reproduced": retry_improves,
            "causal_scope": (
                "ack_retry_2 相对 noack_send_once 同时改变了三件事：打开反向信道、开始消费确认状态、"
                "以及允许第二次尝试。差值是三者的联合效应，不是'ACK 信息本身'的净价值。"
            ),
            "action": (
                "round 6's cost-effect effect gains independent support on fresh streams; "
                "report the byte increase beside the task gain"
                if retry_improves
                else "the +0.875 targets/episode measured on the training streams does not "
                "gain independent support: stop the deployment discussion"
            ),
        }
    if setting == "normal-lossy":
        pair = failures["ack_retry_2 - noack_send_once"]
        same_outcome = bool(
            completion["mean_difference"] == 0.0 and (completion["seeds_worse"] or 0) == 0
        )
        verdict["normal_path"] = {
            "completion": completion,
            "total_bytes": total,
            "new_failures": pair["new_failures"],
            "new_failure_cases": pair["new_failure_cases"],
            "recovered": pair["recovered"],
            "proposed_as_optional_mode": retry_improves,
            "same_outcome_but_dearer": bool(
                same_outcome and total["direction"] == "supported_dearer"
            ),
            "action": (
                "propose ack_retry_2 as an optional high-reliability mode candidate only; the "
                "default stays unchanged and the user decides on the bytes"
                if retry_improves
                else "do not propose deployment: the real information path gives no completion "
                "gain to justify the bytes"
            ),
        }
    frontier_checks: dict[str, Any] = {}
    for arm in ("ack_periodic_20", "ack_backoff_16"):
        block = comparisons[f"{arm} - ack_retry_2"]
        completion_block = block["axes"]["on_time"]
        cost_block = block["axes"]["total_bytes"]
        frontier_checks[arm] = {
            "completion": completion_block,
            "total_bytes": cost_block,
            "beats_the_locked_rule": completion_block["direction"] == "supported_higher",
            "reading": _frontier_reading(completion_block, cost_block),
        }
    unstable = [arm for arm, block in frontier_checks.items() if block["beats_the_locked_rule"]]
    verdict["frontier"] = {
        "checks": frontier_checks,
        "beaten_by": unstable,
        "pre_registered_action": (
            "report the rule choice as unstable on fresh data; do not re-tune or rename the "
            "locked primary rule"
            if unstable
            else "no challenger beats the locked rule with statistical support on this data; "
            "the locked primary rule stays, and this says nothing about the other two "
            "frontier definitions"
        ),
    }
    return verdict


def old_output_diff(revised: dict[str, Any], archived: dict[str, Any]) -> dict[str, Any]:
    """Every place the archived report printed two different CI bounds for one quantity."""
    differences: list[dict[str, Any]] = []
    for setting, blocks in archived["judgement"].items():
        comparisons = revised["comparisons"][setting]
        for section, key, axis, field in (
            ("lossy_confirmation", "ack_retry_2 - noack_send_once", "completion", "completion"),
            ("lossy_confirmation", "ack_retry_2 - noack_send_once", "total_bytes", "total_bytes"),
            ("normal_path", "ack_retry_2 - noack_send_once", "completion", "completion"),
            ("normal_path", "ack_retry_2 - noack_send_once", "total_bytes", "total_bytes"),
        ):
            if section not in blocks or key not in comparisons:
                continue
            differences.append(
                _pair(
                    setting,
                    section,
                    key,
                    axis,
                    blocks[section][field],
                    comparisons[key]["axes"]["on_time" if axis == "completion" else "total_bytes"],
                )
            )
        for arm, entry in blocks.get("frontier", {}).get("checks", {}).items():
            key = f"{arm} - ack_retry_2"
            if key in comparisons:
                differences.append(
                    _pair(setting, "frontier", key, "on_time", entry["completion"],
                          comparisons[key]["axes"]["on_time"])
                )
        for arm, entry in blocks.get("negative_control", {}).get("arms", {}).items():
            key = f"{arm} - noack_send_once"
            if key in comparisons:
                differences.append(
                    _pair(setting, "negative_control", key, "on_time", entry["completion"],
                          comparisons[key]["axes"]["on_time"])
                )
                differences.append(
                    _pair(setting, "negative_control", key, "total_bytes", entry["total_bytes"],
                          comparisons[key]["axes"]["total_bytes"])
                )
    changed = [item for item in differences if item["old_ci"] != item["new_ci"]]
    return {
        "cause": (
            "归档报告对同一统计量调用了两次 bootstrap：比较表用全部五条臂的行集，判读段用只含两条臂的"
            "行集。`evaluate_pilot.paired` 每次调用重新播种，但按臂序逐块消耗抽样，行集大小不同 ⇒ 同一"
            "统计量落在随机流的不同位置 ⇒ 均值相同、分位数边界略有差别。本轮把所有统计量收敛到一个对象，"
            "判读段只切片、不再计算。"
        ),
        "means_changed": bool(
            [item for item in changed if item["old_mean"] != item["new_mean"]]
        ),
        "checked_pairs": len(differences),
        "differing": changed,
        "verdict_note": (
            "旧 COMMUNICATION_LEARNING_VERDICT 与 RESULTS.md §三 引用的 CI（如 [+0.685,+0.990]、"
            "[26 018, 26 687]）属于两臂抽样；修订版统一为比较表那一套（[+0.685,+0.995]、"
            "[26 002, 26 688]）。原始 3,000 行输入的摘要未变，改判据未发生。"
        ),
    }


def _pair(setting, section, arm, axis, old_block, new_block) -> dict[str, Any]:
    return {
        "setting": setting,
        "section": section,
        "quantity": f"{arm} ({axis})",
        "old_mean": old_block.get("mean_difference"),
        "new_mean": new_block.get("mean_difference"),
        "old_ci": [old_block.get("ci95_lower"), old_block.get("ci95_upper")],
        "new_ci": [new_block.get("ci95_lower"), new_block.get("ci95_upper")],
    }


# --------------------------------------------------------------------------- build
def build_payload() -> dict[str, Any]:
    rows = common.load_rows()
    registration = cm.load_registration()
    rejected = {
        f"{row['setting']}/{row['arm']}/{row['seed']}": issue
        for row in rows
        for issue in cm.validate_row(row, registration)
    }
    if rejected:
        raise SystemExit(f"input rows fail revalidation: {list(rejected)[:5]}")
    archived = common.load_archived("COMM_RULE_CONFIRMATION_RESULTS.json")
    payload: dict[str, Any] = {
        "document": "COMM_RULE_CONFIRMATION_RESULTS_REVISED",
        "revises": "COMM_RULE_CONFIRMATION_RESULTS.json",
        "branch": "analysis/comm-evidence-review",
        "charter_version": "v2.1",
        "brief": "COMM_RESEARCH_REVIEW_NEXT_ROUND.md",
        "block": list(common.BLOCK),
        "budget_bytes": common.BUDGET_BYTES,
        "inputs": {
            "shards": [
                {"file": path.name, "bytes": path.stat().st_size, "sha256": common.confirm_common.sha256(path)}
                for path in common.shard_paths()
            ],
            "rows": len(rows),
            "rows_expected": len(cm.job_list(common.confirm_common.BLOCK_SEEDS)),
        },
        "settings": {},
        "comparisons": {},
        "ratios": {},
        "new_failures": {},
        "frontier_definitions": {},
        "judgement": {},
        "episodes": len(rows),
        "incomplete_cells": {},
        "statistics": {
            "method": "seed-level paired percentile bootstrap (evaluate_pilot.paired)",
            "replications": common.BOOTSTRAP_REPLICATIONS,
            "rng_seed": common.BOOTSTRAP_SEED,
            "settings_pooled": False,
            "draws_per_quantity": 1,
        },
        "units": UNIT_LEGEND,
        "channel_semantics": {
            "field": "delivery_probability",
            "semantics": "投递成功率：comm.py 用 rng.random() > p 判丢包，comm_ack.py 用 random() < p 判送达",
            "lossless": "成功率 1.0 ⇒ 丢包率 0.0",
            "lossy": "成功率 0.7 ⇒ 丢包率 0.3",
            "note": "归档报告把该字段写成'丢包均 1.0/0.7'，语义与源码相反；本轮修订措辞，不改配置。",
        },
        "ack_comparison_objects": list(common.COMPARISON_OBJECTS),
        "randomness": archived["randomness"],
    }
    for setting in common.SETTING_LABELS:
        scoped = [row for row in rows if row["setting"] == setting]
        summary = cm.aggregate(scoped)
        payload["settings"][setting] = summary
        payload["incomplete_cells"][setting] = {
            arm: len(common.confirm_common.BLOCK_SEEDS) - int(summary[arm]["episodes"])
            for arm in summary
            if int(summary[arm]["episodes"]) != len(common.confirm_common.BLOCK_SEEDS)
        }
        comparisons = comparisons_for_setting(scoped)
        payload["comparisons"][setting] = comparisons
        payload["ratios"][setting] = {}
        payload["new_failures"][setting] = {}
        for arm, versus, note in common.COMPARISONS:
            key = f"{arm} - {versus}"
            seeds, deltas = cm.delta_vectors(scoped, arm, versus)
            payload["ratios"][setting][key] = ratio_block(deltas, comparisons[key]["axes"])
            payload["new_failures"][setting][key] = cm.new_failures(scoped, arm, versus)
        payload["frontier_definitions"][setting] = {
            "sample_mean": sample_mean_frontier(summary),
            "per_stream": cm.pareto_frontier(scoped),
            "statistical_support": supported_pairwise(comparisons),
        }
        payload["judgement"][setting] = judge(setting, comparisons, payload["new_failures"][setting])
    payload["primary_ratio_check"] = primary_ratio_check(rows, payload)
    payload["old_output_diff"] = old_output_diff(payload, archived)
    payload["aggregate_invariance"] = aggregate_invariance(payload, archived)
    return payload


def primary_ratio_check(rows: list[dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any]:
    """Re-derive the headline B/新增按时任务 ratio under this round's own independent seed."""
    scoped = [row for row in rows if row["setting"] == "isolated-lossy"]
    seeds, deltas = cm.delta_vectors(scoped, "ack_retry_2", "noack_send_once")
    completion = deltas["on_time"]
    rng = np.random.default_rng(RECHECK_BOOTSTRAP_SEED)
    indices = rng.integers(0, completion.size, size=(common.BOOTSTRAP_REPLICATIONS, completion.size))
    denominator = completion[indices].mean(axis=1)
    numerator = deltas["total_bytes"][indices].mean(axis=1)
    defined = denominator > 0.0
    ratios = numerator[defined] / denominator[defined]
    point = float(deltas["total_bytes"].mean() / completion.mean())
    frozen = payload["ratios"]["isolated-lossy"]["ack_retry_2 - noack_send_once"]
    return {
        "quantity": "B/新增按时任务（total），isolated-lossy，ack_retry_2 − noack_send_once",
        "unit": "B per additional on-time target（不是 B/episode）",
        "point_estimate": round(point, 1),
        "raw_numerator_B_per_episode": round(float(deltas["total_bytes"].mean()), 1),
        "raw_denominator_targets_per_episode": round(float(completion.mean()), 4),
        "frozen_interval": [frozen["total_bytes_ratio_ci95"][0], frozen["total_bytes_ratio_ci95"][1]],
        "frozen_undefined_replicates": frozen["undefined_replicates"],
        "recheck_seed": RECHECK_BOOTSTRAP_SEED,
        "recheck_interval": [
            round(float(np.percentile(ratios, 2.5)), 1),
            round(float(np.percentile(ratios, 97.5)), 1),
        ],
        "recheck_undefined_replicates": int(denominator.size - defined.sum()),
        "reading": (
            "两条区间来自同一对逐 seed 差向量、不同抽样种子：区间宽度一致即说明该比值对抽样不敏感，"
            "可以引用；不一致则只引用分子分母两个原始量。"
        ),
    }


def aggregate_invariance(revised: dict[str, Any], archived: dict[str, Any]) -> dict[str, Any]:
    """The 3,000-row summary must not move: compare every archived mean field per arm per setting."""
    checked = 0
    drifted: list[str] = []
    for setting in common.SETTING_LABELS:
        for arm, entry in revised["settings"][setting].items():
            old = archived["settings"][setting].get(arm, {})
            for field, value in entry.items():
                checked += 1
                if field not in old:
                    drifted.append(f"{setting}/{arm}/{field}: missing in archived")
                elif old[field] != value:
                    drifted.append(f"{setting}/{arm}/{field}: {old[field]} -> {value}")
    return {"fields_checked": checked, "drifted": drifted, "identical": not drifted}


# --------------------------------------------------------------------------- render
def _fmt(value: Any, digits: int = 4, dash: str = "—") -> str:
    if value is None:
        return dash
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(block: dict[str, Any], digits: int = 4) -> str:
    lower, upper = block.get("ci95_lower"), block.get("ci95_upper")
    if lower is None or upper is None:
        return "—"
    return f"[{lower:.{digits}f}, {upper:.{digits}f}]"


def _effect_counts(block: dict[str, Any]) -> str:
    return (
        f"{block.get('seeds_better', 0)} 更好 / {block.get('seeds_equal', 0)} 相等 / "
        f"{block.get('seeds_worse', 0)} 更差"
    )


def _cost_counts(block: dict[str, Any]) -> str:
    return (
        f"{block.get('higher_streams', 0)} 更高 / {block.get('equal_streams', 0)} 相等 / "
        f"{block.get('lower_streams', 0)} 更低"
    )


def _ratio_row(key: str, block: dict[str, Any]) -> str:
    if not block.get("defined"):
        return (
            f"| `{key}` | {_fmt(block['denominator_mean_on_time'], 5)} | — | — | **undefined** "
            f"| n/a（未计算） | {block['interval_kind']} | 0 | 否 |"
        )
    total_ci = block.get("total_bytes_ratio_ci95") or [None, None]
    return (
        f"| `{key}` | {_fmt(block['denominator_mean_on_time'], 5)} "
        f"| {block['uplink_bytes_ratio']:,.1f} | {block['ack_bytes_ratio']:,.1f} "
        f"| **{block['total_bytes_ratio']:,.1f}** "
        f"| [{total_ci[0]:,.1f}, {total_ci[1]:,.1f}] "
        f"| {block['interval_kind']} | {block.get('undefined_replicates', 0)} "
        f"| {'是' if block['robustly_interpretable'] else '否'} |"
    )


#: How to say a bootstrap direction in prose, so no internal code reaches the reader.
DIRECTION_WORDS = {
    "supported_higher": "更高（CI 下界 > 0）",
    "supported_lower": "更低（CI 上界 < 0）",
    "undetected": "未检出差异（CI 含 0）",
    "supported_dearer": "更贵（CI 下界 > 0）",
    "supported_cheaper": "更省（CI 上界 < 0）",
    "unknown": "未计算",
}


def _direction(block: dict[str, Any]) -> str:
    return DIRECTION_WORDS.get(block.get("direction", "unknown"), "未计算")


def _frontier_reading(completion: dict[str, Any], cost: dict[str, Any]) -> str:
    """The sentence the protocol allows: what was detected, what was not, and what neither says."""
    if completion["direction"] == "supported_higher":
        effect = "完成数更高（CI 下界 > 0）"
    elif completion["direction"] == "supported_lower":
        effect = "完成数更低（CI 上界 < 0）"
    else:
        margin = (
            f"通过了 −0.05 个/ep 的非劣界（CI 下界 {completion['ci95_lower']:+.3f} ≥ −0.05），"
            "但该界只在'不显著更差'的意义上成立，不建立等效"
            if completion["passes_non_inferiority"]
            else f"未通过 −0.05 个/ep 的非劣界（CI 下界 {completion['ci95_lower']:+.3f} < −0.05）"
        )
        effect = (
            f"完成数未检出差异（CI 含 0）：只能说没测出差别，不能称效果相同；{margin}"
        )
    return f"{effect}；total bytes {_direction(cost)}，两轴结论各自独立"


def render(payload: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append
    add("# 规则独立工程确认：修订版报告（同一统计对象、统一措辞）")
    add("")
    add(
        f"日期：2026-09-21。分支 `{payload['branch']}`，对应总纲 {payload['charter_version']} 与执行指令 "
        f"`{payload['brief']}` §2/§3。修订对象：`../comm_rule_confirm_20260921/"
        f"COMM_RULE_CONFIRMATION_RESULTS.md`（该文件与其判定原样保留、可追溯）。"
    )
    add("")
    add(
        "**本文不产生任何新数据**：episode 0、新流 0、训练 0；输入是归档的 "
        f"{payload['inputs']['rows']:,} 行（8 个分片，逐文件 sha256 见附录），"
        "统计量仍为 seed 级配对百分位 bootstrap（B = "
        f"{payload['statistics']['replications']:,}，RNG {payload['statistics']['rng_seed']}），"
        "三个设置不合并样本。原停止判定与预注册判据未改；改的只有一处统计组织方式（每个量只抽一次样）"
        "与措辞口径。"
    )
    add("")
    add("## 〇、单位与口径")
    add("")
    add(f"- 单位只用这四种，不互换：{payload['units']}。")
    add(
        f"- 信道字段是 `{payload['channel_semantics']['field']}`（**成功率**）："
        f"{payload['channel_semantics']['semantics']}。{payload['channel_semantics']['lossless']}；"
        f"{payload['channel_semantics']['lossy']}。"
    )
    add("- 字节差只说更高/相等/更低；只有完成数才说更好/相等/更差。")
    add("- 比值一律配分子（B/ep）与分母（个/ep）一起给；丢掉非正分母后得到的区间标 `conditional`。")
    add("")

    add("## 一、三个设置上的任务效果与成本")
    add("")
    for setting in common.SETTING_LABELS:
        add(f"### {setting}（{'隔离' if common.SETTINGS[setting]['information'] == 'isolated' else '正常'}信息路径，{common.channel_label(setting)}）")
        add("")
        add(f"*{common.SETTINGS[setting]['role']}*")
        add("")
        add("| 臂 | onT (个/ep) | 上行 B/ep | ACK B/ep | **total B/ep** | ACK 占比 | 额度使用 | 信息年龄(步) |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|")
        for arm in common.ARM_LABELS:
            entry = payload["settings"][setting][arm]
            add(
                f"| `{arm}` | {entry['on_time_per_episode']:.4f} | {entry['uplink_bytes']:,.0f} "
                f"| {entry['ack_bytes']:,.0f} | **{entry['total_bytes']:,.0f}** "
                f"| {entry['ack_share_of_total'] * 100:.1f} % | {entry['budget_used_fraction']:.3f} "
                f"| {entry['belief_age_steps']:.1f} |"
            )
        add("")
        add("| 臂 | 条件完成率 | 全部目标完成率 | expired (个/ep) | unresolved (个/ep) | 发现 (个/ep) |")
        add("|---|---:|---:|---:|---:|---:|")
        for arm in common.ARM_LABELS:
            entry = payload["settings"][setting][arm]
            add(
                f"| `{arm}` | {entry['completion_rate_conditional']:.4f} "
                f"| {entry['completion_rate_all_goals']:.4f} | {entry['expired_per_episode']:.4f} "
                f"| {entry['unresolved_per_episode']:.4f} | {entry['discovered_per_episode']:.4f} |"
            )
        add("")
    add("## 二、四个预注册比较（每个量只抽样一次）")
    add("")
    add(
        "完成数列用'更好/相等/更差'，total-byte 列用'更高/相等/更低'——字节差的正值一律是'更贵'，"
        "不是优势。上行与 ACK 分列只用于说明成本落在哪一侧；任何降本主张只看 total。"
    )
    add("")
    for setting in common.SETTING_LABELS:
        add(f"### {setting}（{common.channel_label(setting)}）")
        add("")
        add("| 比较 | 完成数差 (个/ep) | 95 % CI | 逐流 | total-byte 差 (B/ep) | 95 % CI | 逐流 |")
        add("|---|---:|---|---:|---:|---|---|")
        for arm, versus, _ in common.COMPARISONS:
            block = payload["comparisons"][setting][f"{arm} - {versus}"]["axes"]
            completion, total = block["on_time"], block["total_bytes"]
            add(
                f"| `{arm}` − `{versus}` | {_fmt(completion['mean_difference'])} | {_ci(completion)} "
                f"| {_effect_counts(completion)} | {_fmt(total['mean_difference'], 1)} "
                f"| {_ci(total, 1)} | {_cost_counts(total)} |"
            )
        add("")
        add("| 比较 | 分母 ΔonT (个/ep) | 每任务上行 B | 每任务 ACK B | 每任务 total B | total 区间 | 区间类型 | 未定义 replicate | 可稳健解释 |")
        add("|---|---:|---:|---:|---:|---|---|---:|---|")
        for arm, versus, _ in common.COMPARISONS:
            add(_ratio_row(f"{arm} − {versus}", payload["ratios"][setting][f"{arm} - {versus}"]))
        add("")
        for arm, versus, _ in common.COMPARISONS:
            note = payload["ratios"][setting][f"{arm} - {versus}"].get("note")
            if note:
                add(f"> `{arm}` − `{versus}` 比值：{note}。")
        add("")
        add("| 比较 | 新增失败 (个目标) | 救回 (个目标) |")
        add("|---|---:|---:|")
        for arm, versus, _ in common.COMPARISONS:
            block = payload["new_failures"][setting][f"{arm} - {versus}"]
            add(f"| `{arm}` − `{versus}` | {block['new_failures']} | {block['recovered']} |")
        add("")
    add("## 三、预注册判读（切片，不再计算）")
    add("")
    for setting in common.SETTING_LABELS:
        block = payload["judgement"][setting]
        add(f"### {setting}（{common.channel_label(setting)}）")
        add("")
        if "negative_control" in block:
            control = block["negative_control"]
            add(
                f"**无损负控：{'通过' if control['passed'] else '未通过'}。** {control['action']}。"
            )
            add("")
            add("| 臂 | 完成数差 (个/ep) | 95 % CI | 判定 | total-byte 差 (B/ep) | 方向 |")
            add("|---|---:|---|---|---:|---|")
            for arm, entry in control["arms"].items():
                completion, total = entry["completion"], entry["total_bytes"]
                add(
                    f"| `{arm}` | {_fmt(completion['mean_difference'])} | {_ci(completion)} "
                    f"| {'稳定提高（需暂停检查）' if entry['improves'] else '未稳定提高'} "
                    f"| {_fmt(total['mean_difference'], 1)} | {_direction(total)} |"
                )
            add("")
            add(f"> {control['note']}。")
            add("")
        if "lossy_confirmation" in block:
            confirmation = block["lossy_confirmation"]
            completion, total = confirmation["completion"], confirmation["total_bytes"]
            add(
                f"**有损隔离确认：{'复现' if confirmation['reproduced'] else '未复现'}。** "
                f"{confirmation['action']}。"
            )
            add("")
            add(
                f"- 完成数差 **{completion['mean_difference']:+.4f} 个/ep**，CI "
                f"{_ci(completion)}，{_effect_counts(completion)}（{_direction(completion)}）；"
            )
            add(
                f"- total-byte 差 **{total['mean_difference']:+,.1f} B/ep**，CI {_ci(total, 1)}，"
                f"{_cost_counts(total)}（{_direction(total)}）——是更贵，不是更好；"
            )
            add(f"- 因果范围：{confirmation['causal_scope']}")
            add("")
        if "normal_path" in block:
            normal = block["normal_path"]
            completion, total = normal["completion"], normal["total_bytes"]
            add(
                f"**正常信息路径：{'可作为可选高可靠模式候选' if normal['proposed_as_optional_mode'] else '不可作为部署候选'}。** "
                f"{normal['action']}。"
            )
            add("")
            add(
                f"- 完成数差 **{completion['mean_difference']:+.4f} 个/ep**，CI {_ci(completion)}，"
                f"{_effect_counts(completion)}；"
            )
            add(
                f"- total-byte 差 **{total['mean_difference']:+,.1f} B/ep**，CI {_ci(total, 1)}，"
                f"{_cost_counts(total)}；"
            )
            add(
                f"- 逐目标：新增失败 **{normal['new_failures']}** 个、救回 **{normal['recovered']}** 个；"
                f"结局相同但更贵（工程上被支配）：**{'是' if normal['same_outcome_but_dearer'] else '否'}**。"
            )
            add("")
        frontier = block["frontier"]
        add("**规则比较（预注册动作：不回头调参、不重命名主规则）**")
        add("")
        for arm, entry in frontier["checks"].items():
            add(f"- `{arm}` 相对锁定的 `ack_retry_2`：{entry['reading']}。")
        add("")
        definitions = payload["frontier_definitions"][setting]
        add("**'前沿'必须写明是哪一个定义**（三个定义、三种结论）")
        add("")
        add(
            f"1. 样本均值 Pareto 前沿：{', '.join(f'`{arm}`' for arm in definitions['sample_mean']['frontier'])}；"
            f"被均值支配：{', '.join(f'`{arm}`' for arm in definitions['sample_mean']['dominated']) or '无'}。"
        )
        add(
            f"2. 逐流支配（每流双轴不劣且至少一流严格更优）：前沿 "
            f"{', '.join(f'`{arm}`' for arm in definitions['per_stream']['frontier'])}；被支配 "
            f"{', '.join(f'`{arm}`' for arm in definitions['per_stream']['dominated']) or '无'}。"
        )
        supported = definitions["statistical_support"]["supported"]
        add(
            "3. 有统计支持的支配（完成数 CI 下界 > 0 且 total-byte CI 上界 < 0）："
            + ("；".join(item["reading"] for item in supported) if supported else "无此配对")
            + "。"
        )
        add("")
        add(f"> {definitions['statistical_support']['note']}；预注册动作：{frontier['pre_registered_action']}。")
        add("")
    add("## 四、ACK 比较对象：开关、消费、重试能力与预算")
    add("")
    add(
        "第六轮的参照 `send_once` 是**反向信道打开、但策略不消费确认状态**（每视图一次上报，"
        "没有重发能力）：它的 total 里已含 ACK 字节。本轮把它拆成两条臂显式对照——`noack_send_once`"
        "（信道关）与 `ack_send_once`（信道开、不消费）。因此下面三个成本差是**三个不同的比较**，"
        "不能读成同一条基线发生了变化。"
    )
    add("")
    add("| 比较对象 | 所在轮次 | ACK 开关 | 是否消费 ACK | 重试能力 | 上行额度 |")
    add("|---|---|---|---|---|---|")
    for row in payload["ack_comparison_objects"]:
        add(
            f"| `{row['arm']}` | {row['round']} | {row['ack_switch']} | "
            f"{row['consumes_ack_feedback']} | {row['retry_capability']} | {row['budget']} |"
        )
    add("")
    lossy = payload["comparisons"]["isolated-lossy"]
    add(
        f"- +26 354.1 B/ep = `ack_retry_2` − `noack_send_once`（本轮 200 新流，基线**无 ACK 信道**）："
        "协议整体（开信道 + 消费反馈 + 允许二次尝试）的成本差；"
    )
    add(
        f"- +19 100.7 B/ep = `ack_retry_2` − `ack_send_once`（本轮 200 新流，基线**有 ACK 信道、不消费**）："
        "重发规则本身的成本增量；"
    )
    add(
        "- +19 562 B/ep = 第六轮 `retry_until_acked_2` − `send_once`（48 条**训练**流，基线同为有 ACK、"
        "不消费）：与上一行同类，但数据、轮次不同，不是'同一基线的成本变化'。"
    )
    add("")
    add("## 五、随机数与配对强度")
    add("")
    add(f"- **前向（上行）丢包**：{payload['randomness']['forward']}。")
    add(f"- **反向 ACK 丢包**：{payload['randomness']['reverse']}。")
    add(
        "- 因此：no-ACK 与 ACK 臂之间是**同分布比较**（同 seed、同机制，不是同一串事件）；"
        "ACK 臂之间在相同 (episode, subscriber, target, ordinal) 键上是**共同随机数**。"
        "跨这两类比较的数字不能互相代入解释。"
    )
    add("")
    add("## 六、主比值单独核对")
    add("")
    check = payload["primary_ratio_check"]
    add(
        f"- {check['quantity']}：**{check['point_estimate']:,.1f} {check['unit']}**"
        f"（分子 {check['raw_numerator_B_per_episode']:,.1f} B/ep、分母 "
        f"{check['raw_denominator_targets_per_episode']:.4f} 个/ep）。"
    )
    add(
        f"- 冻结区间（RNG {payload['statistics']['rng_seed']}，未定义 replicate "
        f"{check['frozen_undefined_replicates']}）：[{check['frozen_interval'][0]:,.1f}, "
        f"{check['frozen_interval'][1]:,.1f}]；独立复核区间（RNG {check['recheck_seed']}，未定义 "
        f"{check['recheck_undefined_replicates']}）：[{check['recheck_interval'][0]:,.1f}, "
        f"{check['recheck_interval'][1]:,.1f}]。"
    )
    add(f"- {check['reading']}")
    add(
        "- 经济解释只给量纲正确的条件：设新增按时任务价值 V（效用/任务）、字节成本 c（效用/B），"
        "此样本均值下净效用改善需 `V × 0.835 > c × 26354.1`。未指定 V、c 前不宣布值得部署；"
        "total 仍只含任务上行与 ACK，不含未计入的 gossip/系统通信。"
    )
    add("")
    add("## 七、与归档输出的差异登记")
    add("")
    diff = payload["old_output_diff"]
    add(f"- 原因：{diff['cause']}")
    add(
        f"- 核对 {diff['checked_pairs']} 对旧/新数值：均值是否改变 = "
        f"{'是' if diff['means_changed'] else '否（仅分位边界）'}；差异条目 "
        f"{len(diff['differing'])} 项。"
    )
    add(f"- {diff['verdict_note']}")
    add("")
    add("| 设置 | 段落 | 量 | 旧 CI | 新 CI | 均值 |")
    add("|---|---|---|---|---|---|")
    for item in diff["differing"]:
        add(
            f"| {item['setting']} | {item['section']} | {item['quantity']} | "
            f"[{_fmt(item['old_ci'][0])}, {_fmt(item['old_ci'][1])}] | "
            f"[{_fmt(item['new_ci'][0])}, {_fmt(item['new_ci'][1])}] | "
            f"{_fmt(item['old_mean'])} = {_fmt(item['new_mean'])} |"
        )
    add("")
    invariance = payload["aggregate_invariance"]
    add(
        f"- **3,000 行输入摘要不变性核对**：{invariance['fields_checked']} 个 (设置, 臂, 字段) 均值"
        f"逐项与归档 results 比对，"
        f"{'全部相同' if invariance['identical'] else '发现漂移 ' + str(invariance['drifted'][:5])}。"
    )
    add("")
    add("## 八、复跑与卫生")
    add("")
    add("```bash")
    add("cd runs/comm_evidence_review_20260921")
    add("python revise_results.py                      # 读 3,000 行 → 修订 results（唯一统计对象）")
    add("python revise_verdict.py                     # 由修订 results 渲染 verdict")
    add("python evidence_corrections.py               # 十项核对与原文→修订映射")
    add("python learning_diagnostic.py                # 仅用已有训练记录做诊断（不跑 episode）")
    add("python -m unittest discover -s . -p 'test_*.py' -v")
    add("```")
    add("")
    add(
        f"本轮实测：0 episode、0 新流、0 训练。`{common.SEALED}` 未生成、未读取、未改动；"
        "`900000..900199` 已消费，只作既有证据使用，不作为下一候选的未见测试。生产默认值未改动"
        "（`comm_ack=False`、`comm_timing_mode=legacy_shared_bound`）。"
    )
    add("")
    add("### 附录：输入分片")
    add("")
    add("| 文件 | 字节 | sha256 |")
    add("|---|---:|---|")
    for row in payload["inputs"]["shards"]:
        add(f"| `{row['file']}` | {row['bytes']:,} | `{row['sha256'][:16]}…` |")
    add("")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="build, assert invariance, print summary")
    args = parser.parse_args(argv)
    payload = build_payload()
    if not payload["aggregate_invariance"]["identical"]:
        raise SystemExit(f"aggregate drifted: {payload['aggregate_invariance']['drifted'][:5]}")
    common.write_json(common.json_path("COMM_RULE_CONFIRMATION_RESULTS_REVISED.json"), payload)
    if args.check_only:
        print(json.dumps({
            "rows": payload["inputs"]["rows"],
            "aggregate_identical": payload["aggregate_invariance"]["identical"],
            "differing_ci": len(payload["old_output_diff"]["differing"]),
        }, ensure_ascii=False))
        return 0
    text = render(payload)
    common.json_path("COMM_RULE_CONFIRMATION_RESULTS_REVISED.md").write_text(text, encoding="utf-8")
    print(f"written: COMM_RULE_CONFIRMATION_RESULTS_REVISED.md ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
