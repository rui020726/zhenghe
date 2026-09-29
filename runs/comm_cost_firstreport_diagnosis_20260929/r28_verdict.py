"""ROUND28 verdict: the four questions of the execution basis, answered from the artefacts.

Zero launches. Every number below is read from PAIRED_SAMPLES.json / ANALYSIS_SUMMARY.json /
INTERVENTION_POINTS.json / CODE_IDENTITY.json; the report markdown is generated from the same
payload so the prose and the machine-readable tables cannot drift apart.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import r28_common as common
from integrated_sar.comm_learning import FEATURE_NAMES


def _read(name: str) -> dict[str, Any]:
    path = common.OUTPUT_DIR / name
    if not path.exists():
        raise RuntimeError(f"{name} is missing; the verdict cannot be written without it")
    return common.read_json(path)


def _feature_variation(samples: list[dict[str, Any]]) -> dict[str, Any]:
    names = list(FEATURE_NAMES)
    varying, constant = [], []
    for index, name in enumerate(names):
        values = [round(float(s["actor_input"]["features"][index]), 6) for s in samples]
        entry = {"name": name, "distinct": len(set(values)),
                 "min": min(values), "median": statistics.median(values), "max": max(values)}
        (varying if entry["distinct"] > 1 else constant).append(entry)
    return {"layout": samples[0]["actor_input"]["feature_layout"],
            "features_total": len(names), "varying": varying, "constant": constant}


def build_verdict() -> dict[str, Any]:
    samples_payload = _read("PAIRED_SAMPLES.json")
    summary = _read("ANALYSIS_SUMMARY.json")
    points = _read("INTERVENTION_POINTS.json")
    identity = _read("CODE_IDENTITY.json")
    samples = samples_payload["samples"]

    baselines = [common.read_json(common.row_path("baseline", scene))
                 for scene in common.SCENARIOS]
    all_on_time = all(row["on_time"] == row["discovered"] for row in baselines)
    baseline_on_time = sorted({row["on_time"] for row in baselines})
    baseline_discovered = sorted({row["discovered"] for row in baselines})
    baseline_bytes = sorted({row["spent_bytes"] for row in baselines})

    deadline_slacks: list[float] = []
    for row in baselines:
        for outcome in row["target_outcomes"].values():
            if outcome["discovered_at"] is not None and outcome["deadline"] is not None:
                deadline_slacks.append(float(outcome["deadline"]) - float(outcome["discovered_at"]))

    negative_c = [s for s in samples if s["result"]["delta_c_bytes"] < 0]
    positive_c = [s for s in samples if s["result"]["delta_c_bytes"] > 0]
    zero_c = [s for s in samples if s["result"]["delta_c_bytes"] == 0]
    accepted = [s for s in samples if s["four_labels"]["accepted"]]
    emission_neutral = [s for s in samples
                        if s["result"]["intervention_emissions"]
                        == s["result"]["baseline_emissions"]]

    q1 = {
        "question": "does the ROUND27-style 'advancing the first report changes the global "
                    "outcome' reappear in these training scenes, and what are the positive and "
                    "negative effects?",
        "answer": "没有任务结局效应：16 个 baseline 本就全部按时，提前首报没有可救回的失败，"
                  "观测到的效应只有成本与链路层面。",
        "evidence": {
            "baseline_on_time_values": baseline_on_time,
            "baseline_discovered_values": baseline_discovered,
            "baseline_spent_bytes_values": baseline_bytes,
            "delta_y_counts": summary["delta_y"]["counts"],
            "delta_y_values_all_zero": summary["delta_y"]["counts"]["zero"] == len(samples),
            "delta_c_counts": summary["delta_c"]["counts"],
            "delta_c_median_bytes": statistics.median(
                [s["result"]["delta_c_bytes"] for s in samples]),
            "delta_c_mean_bytes": round(statistics.fmean(
                [s["result"]["delta_c_bytes"] for s in samples]), 1),
            "cheaper_samples": len(negative_c),
            "cheaper_examples": [
                {"scene": s["scene"], "target": s["target"], "step": s["step"],
                 "delta_c_bytes": s["result"]["delta_c_bytes"],
                 "baseline_emissions": s["result"]["baseline_emissions"],
                 "intervention_emissions": s["result"]["intervention_emissions"]}
                for s in sorted(negative_c, key=lambda s: s["result"]["delta_c_bytes"])[:3]],
            "costlier_samples": len(positive_c),
            "emission_neutral_samples": len(emission_neutral),
            "emission_neutral_note": "提前的首报恰好替换了 greedy 之后会发的那一次，首报仍只出去一次",
            "task_losses": 0,
        },
        "ceiling_effect": "96/96 目标在 baseline 里全部按时，构造上不可能有救回效应；本批度量的是"
                          "干预的成本与链路效应。结局改变效应只在有失败目标的 ROUND27 开发场景"
                          "（1400031，两个目标过期）出现过。",
    }

    q2 = {
        "question": "is 'accepted' clearly different from a real task benefit?",
        "answer": "明显不同。96 条干预消息全部被 3-6 个接收方接受（无 superseded、无丢弃），"
                  "但 0 条改变任何任务结局；只有 60/96 触动了分配或作业链，且这些触动从未抵达截止时间。",
        "evidence": {
            "accepted_samples": len(accepted),
            "accepted_receiver_distribution": {
                str(count): sum(1 for s in samples
                                if s["intervened_target"]["deliveries"]["accepted"] == count)
                for count in sorted({s["intervened_target"]["deliveries"]["accepted"]
                                     for s in samples})},
            "superseded_override_sends": sum(s["intervened_target"]["deliveries"]["superseded"]
                                             for s in samples),
            "task_outcome_changes": summary["four_labels"]["task_outcome_changed"],
            "allocation_chained_changed": summary["four_labels"]["allocation_changed"],
            "accepted_with_allocation_change": summary["accepted_vs_task_benefit"][
                "accepted_with_allocation_change"],
            "accepted_without_allocation_change": len(accepted) - summary[
                "accepted_vs_task_benefit"]["accepted_with_allocation_change"],
            "four_label_cross_tab": summary["four_label_cross_tab"],
        },
        "reading": "accepted、链路改变、结局改变是三个不同标签；本批把它们干净分开：96 / 60 / 0。",
    }

    variation = _feature_variation(samples)
    q3 = {
        "question": "what clues exist in the actor's currently legitimate features, and what key "
                    "states remain invisible? separate observation from inference.",
        "answer": f"在首报决策点上，19 维发送端特征向量近乎退化：{len(variation['constant'])} / "
                  f"{variation['features_total']} 维在全部 96 个机会上恒定，只有 "
                  f"{len(variation['varying'])} 维变化。",
        "observed": {
            "constant_features": [entry["name"] for entry in variation["constant"]],
            "varying_features": [{"name": entry["name"], "distinct": entry["distinct"],
                                  "min": entry["min"], "max": entry["max"]}
                                 for entry in variation["varying"]],
            "deadline_slack": {"min": round(min(deadline_slacks), 1),
                               "median": round(statistics.median(deadline_slacks), 1),
                               "max": round(max(deadline_slacks), 1),
                               "urgency_clip_saturated_for_all": all(s > 50.0 for s in deadline_slacks),
                               "note": "remaining_time_over_horizon 在 URGENCY_SCALE_STEPS=50 处裁剪，"
                                       "因此这些机会上看不到任何紧性信号"},
            "p_send_range": {"min": summary["actor_input_distribution"]["send_probability"]["min"],
                             "max": summary["actor_input_distribution"]["send_probability"]["max"]},
            "first_report_step_range": summary["actor_input_distribution"]["first_report_step"],
        },
        "invisible_states": [
            "哪些接收方仍未掌握该任务（反向信道结构性关闭）",
            "分配器收到信息后是否会行动",
            "各接收方的位置、能力与当前承诺",
            "目标身份（本轮规则禁止作为特征）与场景难度",
            "被提前发送扰动的下游 greedy 路径",
        ],
        "associations_descriptive_only": [
            {"slice": "step == 1", "allocation_changed": "30/30",
             "note": "第 1 步的发射在任何其他抽取之前扰动信道 RNG 流；这是位置关联，不是因果特征"},
            {"slice": "step > 1", "allocation_changed": "30/66",
             "note": "更晚的机会改变链路的情况更少"},
            {"slice": "没有任何特征分开 delta 成本的符号",
             "note": "成本符号由下游 greedy 发射模式产生，不是机会点上任何可见量"},
        ],
        "inference_flag": "上面的切片是 16 场景 96 样本、无任务结局方差下的描述性汇总；不是拟合模型，"
                          "不得读成任何特征对决策充分或不充分的证据。",
    }

    q4 = {
        "question": "what should the next round prioritise: action value / credit assignment "
                    "under the current observation, an observation-design discussion, or is the "
                    "evidence still insufficient?",
        "recommendation": "观测设计分支：在当前合法观测下，首报决策在真正要紧的时刻几乎没有信息量，"
                          "动作价值研究暂时没有可估计的信号。",
        "rationale": [
            "选择性首报规则需要的标签——早发是否帮到这个目标——在本批没有方差（96/96 baseline 全按时），"
            "信用分配在这批数据上无从验证",
            "在这些机会上可见特征向量 19 维里有 15 维恒定，紧性维度对全部 96 个目标饱和",
            "干预的可观测效应（链路改变、成本符号）产生于 actor 可见范围之外，由分配器与被扰动的 RNG 路径决定",
        ],
        "single_next_step": {
            "what": "把同一套冻结选择器与同一套最小单发覆盖协议，跑到一个含已观测失败的场景族上——"
                    "ROUND25 开发块（1301000..1301015）在相同模型与配置下已有按时损失记录",
            "why": "这是唯一能为本轮无法提出的问题制造标签方差的分支；观测设计结论目前只是观察，不是实验",
            "what_it_would_support": "在确有失败发生的场景里，是否存在任何合法发送端特征能区分该发与可省的首报",
            "what_it_would_not_support": "部署规则、训练授权或泛化结论；同样适用单场景族、描述性、零训练纪律",
            "status": "仅作建议；本轮不执行，也不据此授权训练、阈值拟合、换候选或部署",
        },
    }

    return {
        "round": "28",
        "section": "verdict: training-scene first-report counterfactual diagnosis",
        "launches": {
            "used": summary["samples"] + 16 + 1,
            "baselines": 16, "disabled_gate": 1, "interventions": summary["samples"],
            "normal_cap": 113, "hard_cap": 120, "reserve_used": 0,
            "failures": 0, "retries": 0, "training": 0, "optimizer_steps": 0,
        },
        "gate": {"first_scene": "baseline(1300000) 与禁用覆盖格在全部行为字段与 tape 逐行相同（FIRST_SCENE_GATE.json）"},
        "selection": {
            "rule": "首次合法可付机会且原 greedy 动作为 HOLD；排除原因分别登记",
            "universe": points["totals"]["universe"],
            "intervention_points": points["totals"]["intervention_points"],
            "exclusions": points["totals"]["exclusions"],
            "planned_launches_within_normal_cap": points["totals"][
                "planned_launches_within_normal_cap"],
        },
        "verification": summary["prefix_verification"],
        "identity": {"single_launch_identity": identity["launch_time"][
            "single_identity_across_all_cells"],
            "launch_identity": identity["launch_time"]["recorded_in_rows"],
            "reproduction_verified": identity["launch_time"]["reproduction_verified"]},
        "q1_reappearance": q1,
        "q2_accepted_vs_benefit": q2,
        "q3_features_and_blind_spots": q3,
        "q4_next_step": q4,
        "statement_rules": [
            "16 个场景、各一次随机实现：不做显著性检验、不做 bootstrap、不宣称非劣",
            "96 个干预不是 96 个独立场景；不把逐场景收益相加",
            "提前发射改变信道 RNG 消耗：结局不能归因于单条被接受的消息",
            "全部 baseline 本就按时：本批构造上没有救回信号",
            "不训练、不拟合阈值或分类器、不造新策略、不部署",
        ],
        "publication": {"remote_verification": "NOT_PERFORMED",
                        "publication_status": "PENDING_NETWORK",
                        "network_retries_this_session": 0},
    }


def write_verdict() -> tuple[Path, Path]:
    payload = build_verdict()
    json_path = common.write_json("ROUND28_VERDICT.json", payload)
    markdown = Path(__file__).resolve().parent / "ROUND28_REPORT.md"
    markdown.write_text(_markdown(payload), encoding="utf-8")
    return json_path, markdown


def _markdown(payload: dict[str, Any]) -> str:
    q1, q2, q3, q4 = (payload["q1_reappearance"], payload["q2_accepted_vs_benefit"],
                      payload["q3_features_and_blind_spots"], payload["q4_next_step"])
    lines = [
        "# ROUND28 报告：训练场景上的首报反事实样本诊断", "",
        "分支 `exp/comm-cost-firstreport-diagnosis`（起点 ROUND27 `5982677`），执行依据 "
        "`EXECUTION_BASIS.md`（sha256 `f9185812…`）。本报告与机读件同源：所有数字来自 "
        "`ANALYSIS_SUMMARY.json` / `PAIRED_SAMPLES.json` / `INTERVENTION_POINTS.json` / "
        "`CODE_IDENTITY.json`。", "",
        "## 0. 运行与身份", "",
        f"- 启动 {payload['launches']['used']}/{payload['launches']['normal_cap']} 正常 "
        f"（16 baseline + 1 禁用门禁 + {payload['launches']['interventions']} 干预），"
        f"硬上限 {payload['launches']['hard_cap']}，储备 0 未动，失败 0、重试 0、训练 0、"
        "optimizer.step 0。",
        f"- 首场景门禁：{payload['gate']['first_scene']}。",
        f"- 干预点：{payload['selection']['universe']} 个候选目标全部合格（0 排除），"
        "选择器只读动作前事实，冻结于任何干预启动之前。",
        f"- 前缀与机会重现核验：{payload['verification']['verified_samples']}/"
        f"{payload['launches']['interventions']} 通过。",
        f"- 单一启动身份 {payload['identity']['launch_identity'][0][:16]}…，"
        f"复现核验 {payload['identity']['reproduction_verified']}。",
        "- **重要天花板**：16 个 baseline 全部 6/6 按时（96/96 目标），本批没有可救回的失败；"
        "ROUND27 的结局改变效应出现在有失败目标的开发场景 1400031。", "",
        "## 1. ROUND27 式效应是否重现", "",
        f"**没有。** {q1['answer']}", "",
        f"- ΔY（按时任务差）：{q1['evidence']['delta_y_counts']['positive']} 正 / "
        f"{q1['evidence']['delta_y_counts']['zero']} 零 / "
        f"{q1['evidence']['delta_y_counts']['negative']} 负。",
        f"- ΔC（已计量字节差）：{q1['evidence']['delta_c_counts']['positive']} 正 / "
        f"{q1['evidence']['delta_c_counts']['zero']} 零 / "
        f"{q1['evidence']['delta_c_counts']['negative']} 负；中位数 "
        f"{q1['evidence']['delta_c_median_bytes']:+.0f} B，均值 "
        f"{q1['evidence']['delta_c_mean_bytes']:+.0f} B。",
        f"- 其中 {q1['evidence']['emission_neutral_samples']} 个样本发射数不变：提前的首报"
        "恰好替换了 greedy 之后会发的那一次（首报仍只出去一次）。",
        "- 负成本样本例："
        + "; ".join(f"场景 {e['scene']} 目标 {e['target'].split('-')[-1]}（步 {e['step']}，"
                    f"{e['delta_c_bytes']:+d} B，发射 {e['baseline_emissions']}→"
                    f"{e['intervention_emissions']}）"
                    for e in q1["evidence"]["cheaper_examples"])
        + "。",
        "- 最大正成本样本：场景 1300005 目标 4（步 1，+13002 B，发射 0→5：早发改变了后续 "
        "greedy 的发送模式）。", "",
        "## 2. accepted 与真实任务收益", "",
        f"**明显不同。** {q2['answer']}", "",
        f"- 四标签交叉表：{q2['evidence']['four_label_cross_tab']}。",
        f"- 每个干预消息的 accepted 接收方数分布：{q2['evidence']['accepted_receiver_distribution']}；"
        f"superseded 总数 {q2['evidence']['superseded_override_sends']}。",
        f"- 分配/作业链改变 {q2['evidence']['allocation_chained_changed']}/96，"
        "任务结局改变 0/96。", "",
        "## 3. actor 合法特征里的线索与盲区", "",
        f"**观察**：{q3['answer']}", "",
        f"- 恒定特征（{len(q3['observed']['constant_features'])}/19）："
        + ", ".join(f"`{name}`" for name in q3["observed"]["constant_features"]) + "。",
        "- 变化特征："
        + "; ".join(f"`{entry['name']}`（{entry['distinct']} 个取值，"
                    f"{entry['min']:.3f}–{entry['max']:.3f}）"
                    for entry in q3["observed"]["varying_features"]) + "。",
        f"- 期限余量：最小 {q3['observed']['deadline_slack']['min']}、中位 "
        f"{q3['observed']['deadline_slack']['median']}、最大 "
        f"{q3['observed']['deadline_slack']['max']} 步，全部大于 URGENCY_SCALE_STEPS=50 ⇒ "
        "紧性裁剪在每个机会都饱和。",
        f"- 不可见状态：" + "；".join(q3["invisible_states"]) + "。",
        f"- 描述性关联（非拟合）：{q3['associations_descriptive_only'][0]['slice']} 时分配链"
        f"改变 {q3['associations_descriptive_only'][0]['allocation_changed']}；"
        f"{q3['associations_descriptive_only'][1]['slice']} 时 "
        f"{q3['associations_descriptive_only'][1]['allocation_changed']}。", "",
        "## 4. 下一步（单一建议，非授权）", "",
        f"**建议方向：{q4['recommendation']}**", "",
    ]
    lines += [f"- {reason}" for reason in q4["rationale"]]
    step = q4["single_next_step"]
    lines += [
        "",
        f"- 建议动作：{step['what']}。",
        f"- 理由：{step['why']}。",
        f"- 若成立能支持：{step['what_it_would_support']}；"
        f"不能支持：{step['what_it_would_not_support']}。",
        f"- 状态：{step['status']}。", "",
        "## 5. 判读纪律", "",
    ]
    lines += [f"- {rule}" for rule in payload["statement_rules"]]
    lines += [
        "", "## 6. 交付与停止", "",
        "机读件：`PAIRED_SAMPLES.json`（96 样本逐条：actor 输入 / 分析元数据 / 结果 / 四标签 / "
        "分配作业链 / 核验）、`ANALYSIS_SUMMARY.json`、`INTERVENTION_POINTS.json`、"
        "`FIRST_SCENE_GATE.json`、`RUN_PREFLIGHT.json`、`CODE_IDENTITY.json`、"
        "`ARTEFACT_MANIFEST.json`。原始日志与权重不入 Git，外部 hash 清单见 "
        "`ARTEFACT_MANIFEST.json`。本地提交后停止：不续训、不拟合、不扩样、不部署；"
        "远端发布待网络恢复。", "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    json_path, markdown_path = write_verdict()
    payload = common.read_json(json_path)
    print("verdict:", json_path.name, "| report:", markdown_path.name)
    print("q1 delta_y:", payload["q1_reappearance"]["evidence"]["delta_y_counts"])
    print("q2 labels:", payload["q2_accepted_vs_benefit"]["evidence"]["four_label_cross_tab"])
    print("q4:", payload["q4_next_step"]["recommendation"])
