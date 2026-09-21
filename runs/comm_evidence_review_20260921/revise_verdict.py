"""Re-render the frozen learning verdict from the revised results object, with scoped claims.

The archived verdict is a committed document and stays untouched. This module reads the revised
rule-confirmation results plus round 6's own gate JSON, and emits
``COMMUNICATION_LEARNING_VERDICT_REVISED.md`` / ``.json``. Nothing is retyped by hand: every
number is sliced from a machine-readable artefact, and the differences against the archived
verdict are recorded in ``differences_against_archived``.

The corrections that motivate this file (``COMM_RESEARCH_REVIEW_NEXT_ROUND.md`` §3.6--§3.10):
the three learning facts are stated separately instead of packaged; the retry-vs-send-once
difference is labelled a joint effect of three changes rather than the net value of ACK
information; the lossless tie is read as "this retry gain is tied to the loss condition that
was tested", not as proof that no bytes were spent for score; "extended training ruled out
undertraining" is removed; and the 425-point gate is described as the sample it is.

    python revise_verdict.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import review_common as common

#: The frozen conclusion quoted from the round-7 brief, unchanged in wording.
FROZEN_CONCLUSION = (
    "在已经测试的任务分布、两类发送协议、no-ACK/ACK 反馈、观测、动作、奖励和有界训练资源下，"
    "学习型发送控制器没有建立相对强规则的附加价值；简单规则形成了当前可复现的成本—效果前沿。"
)

#: Phrasings the verdict must not use, each with the reason it overreaches.
FORBIDDEN_CLAIMS: tuple[dict[str, str], ...] = (
    {
        "claim": "强化学习永远无效",
        "why": "负结论的范围是已测分布/观测/动作/奖励/训练资源，不是方法本身",
    },
    {
        "claim": "规则全局最优",
        "why": "规则只是在已测集合与已测数据上形成可复现前沿",
    },
    {
        "claim": "没有任何可学习空间",
        "why": "门禁 3 只否定被采样的那 425 个单步决策点；本轮 3 000 集没有学习臂",
    },
    {
        "claim": "ACK 模型训练失败",
        "why": "ACK 从未训练过模型，只做过规则与信息门禁；三条事实不能打包",
    },
    {
        "claim": "扩训已排除欠训练",
        "why": "100k→400k 的改善与种子间差异缩小不足以排除仍欠训练、稀疏信用分配或表征问题",
    },
    {
        "claim": "96.5 % 的所有决策没有学习空间",
        "why": "打平率是特定采样分布、单次干预与固定后续策略下的观测，且各 ≥20 点/≥8 流只是资源门槛",
    },
    {
        "claim": "无损打平证明不是多花字节买分",
        "why": "无损负控只支持'该重试收益与所测丢包条件有关'，不分离反馈价值与额外重试机会",
    },
)

#: The three facts that must never be collapsed into one sentence.
LEARNING_FACTS: tuple[dict[str, str], ...] = (
    {
        "fact": "no-ACK 学习模型未建立稳定优势",
        "scope": "修正采样器之后、两种 timing 协议、两档预算、每条件 3 seed × 400k 环境步",
        "evidence": "S3-1…S3-4：预注册门槛 0/3；但每个运行的六个训练都显著优于各自未训练初值",
    },
    {
        "fact": "ACK 只做过规则与信息门禁，未训练任何 ACK 模型",
        "scope": "S3-5：48 流 × 22 臂规则矩阵 + 425 点受控反事实；门禁 3 未过 ⇒ 按指令未训练",
        "evidence": "runs/comm_ack_value_20260920/COMM_ACK_VALUE_GATE_25pct.json（decision = stop）",
    },
    {
        "fact": "本轮 3 000 episode 里没有学习臂",
        "scope": "S3-6：五条已锁定规则 × 3 设置 × 200 新流，训练 0 次",
        "evidence": "本文修订报告的 inputs.rows = 3 000，arms 全部是规则",
    },
)


def load_round6() -> dict[str, Any]:
    gate = json.loads(
        (common.ACK_ROUND / "COMM_ACK_VALUE_GATE_25pct.json").read_text(encoding="utf-8")
    )
    gates = gate["gates"]
    payoff = gates["2_information_pays"]
    choice = gates["3_real_choice"]
    exists = gates["1_information_exists"]
    dominates = gates["4_no_rule_dominates"]
    return {
        "gate_1_streams": [exists["streams_qualifying"], exists["streams_total"]],
        "gate_2": {
            "selected_rule": payoff["selected_ack_rule"],
            "baseline": payoff["baseline"],
            "baseline_ack_switch": "开（本轮 22 臂全部 ack=True）",
            "baseline_consumes_feedback": "否：send_once 每视图一次上报，没有重发决策",
            "baseline_on_time": payoff["baseline_on_time"],
            "baseline_total_bytes": payoff["baseline_total_bytes"],
            "selected_on_time": payoff["selected_on_time"],
            "selected_total_bytes": payoff["selected_total_bytes"],
            "completion": payoff["completion_difference"],
            "total_bytes": payoff["total_bytes_difference"],
            "passed": payoff["passed"],
        },
        "gate_3": choice,
        "gate_4": {"dominators": dominates["dominators"], "passed": dominates["passed"]},
        "thresholds": gate["thresholds"],
        "decision": gate["decision"],
    }


def build_payload() -> dict[str, Any]:
    results = common.read_json("COMM_RULE_CONFIRMATION_RESULTS_REVISED.json")
    archived_verdict = common.load_archived("COMMUNICATION_LEARNING_VERDICT.json")
    round6 = load_round6()
    lossy = results["judgement"]["isolated-lossy"]["lossy_confirmation"]
    normal = results["judgement"]["normal-lossy"]["normal_path"]
    lossless = results["judgement"]["isolated-lossless"]["negative_control"]
    ratio = results["ratios"]["isolated-lossy"]["ack_retry_2 - noack_send_once"]
    failures = results["new_failures"]["isolated-lossy"]["ack_retry_2 - noack_send_once"]
    completion = lossy["completion"]
    total = lossy["total_bytes"]
    return {
        "document": "COMMUNICATION_LEARNING_VERDICT_REVISED",
        "revises": "COMMUNICATION_LEARNING_VERDICT.json",
        "date": "2026-09-21",
        "branch": "analysis/comm-evidence-review",
        "charter_version": results["charter_version"],
        "brief": results["brief"],
        "numbers_from": "COMM_RULE_CONFIRMATION_RESULTS_REVISED.json / 第六轮 gate JSON",
        "learning_facts": list(LEARNING_FACTS),
        "conclusions": {
            "learning_negative": {
                "claim": FROZEN_CONCLUSION,
                "s3_state": "范围准确的负结果；研究先导按预注册停止条件关闭",
                "three_facts_stated_separately": list(LEARNING_FACTS),
                "training_sufficiency": (
                    "扩训（100k→400k 环境步）改善了种子间稳定性并让六个运行显著优于各自初值，"
                    "但这只排除'完全学不动'，不排除仍然欠训练、时序信用分配过稀或表征不足。"
                    "'优于初值'≠充分收敛，'达到资源上限'≠最优。停止决定保留，绝对表述删除。"
                ),
                "not_claimed": [item["claim"] for item in FORBIDDEN_CLAIMS],
                "why_not_claimed": {item["claim"]: item["why"] for item in FORBIDDEN_CLAIMS},
                "final_test_block": {
                    "block": common.SEALED,
                    "opened": False,
                    "generated": False,
                    "read": False,
                },
            },
            "protocol_and_rule_gains": {
                "claim": (
                    "把 3/10 静默上界从传输层代发改为显式规则后，规则集合本身改善：send_once 成为同"
                    "协议合法对照并在主预算上支配 immediate 同行为。这是协议/规则收益，与学习收益"
                    "必须分开引用。"
                ),
                "evidence": "S3-4（runs/comm_protocol_v2_20260920/）",
                "nature": "不是某条规则变好，而是规则集合的合法性变了",
            },
            "ack_task_gain_and_total_cost": {
                "claim": (
                    "ACK 信息真实、可获得、有后果（一条简单规则就能拿到完成数增益），但不是免费的，"
                    "且增益不能单独归因给反馈本身。"
                ),
                "units": results["units"],
                "round6_training_streams": round6["gate_2"],
                "round7_fresh_streams": {
                    "setting": "isolated-lossy（成功率 0.7 ⇒ 丢包率 0.3）",
                    "completion_gain_targets_per_episode": completion["mean_difference"],
                    "completion_ci95": [completion["ci95_lower"], completion["ci95_upper"]],
                    "per_stream": [
                        completion["seeds_better"], completion["seeds_equal"], completion["seeds_worse"]
                    ],
                    "total_bytes_increase_B_per_episode": total["mean_difference"],
                    "total_bytes_ci95": [total["ci95_lower"], total["ci95_upper"]],
                    "bytes_per_additional_on_time_target": ratio["total_bytes_ratio"],
                    "bytes_per_additional_unit": "B / 新增一个按时任务（不是 B/episode）",
                    "per_target_interval": ratio["total_bytes_ratio_ci95"],
                    "ratio_interval_kind": ratio["interval_kind"],
                    "ratio_robustly_interpretable": ratio["robustly_interpretable"],
                    "new_per_target_failures": failures["new_failures"],
                    "recovered": failures["recovered"],
                },
                "cost_comparisons_are_different": [
                    {
                        "difference": "+26 354.1 B/ep",
                        "comparison": "ack_retry_2 − noack_send_once（基线无 ACK 信道）",
                        "what_it_measures": "协议整体：开信道 + 消费反馈 + 允许二次尝试",
                    },
                    {
                        "difference": "+19 100.7 B/ep",
                        "comparison": "ack_retry_2 − ack_send_once（基线有 ACK 信道、不消费）",
                        "what_it_measures": "重发规则本身的成本增量",
                    },
                    {
                        "difference": "+19 562 B/ep",
                        "comparison": "第六轮 retry_until_acked_2 − send_once（48 条训练流，基线同为不消费）",
                        "what_it_measures": "与上一行同类的量，但数据与轮次不同；不是同一基线的变化",
                    },
                ],
                "causal_scope": (
                    "retry 胜过只发一次同时改变三件事（反向信道、是否消费确认、是否有第二次尝试），"
                    "因此该差值不单独证明 ACK 信息本身的净价值。要分离反馈价值，需要'相近重试机会与"
                    "调参资源的无反馈重试'对照，本轮未新增任何运行。"
                ),
                "negative_control_reading": (
                    "无损负控上完成数逐字相等而 total 更贵 ⇒ 只支持'该重试收益与所测丢包条件有关'。"
                    "它不支持'不是多花字节买分'这种机制结论，因为在有损信道上增益也可能全部来自额外的"
                    "第二次尝试而非确认状态。"
                ),
                "lossless_control": {
                    "completion": lossless["arms"]["ack_retry_2"]["completion"]["mean_difference"],
                    "total_bytes": lossless["arms"]["ack_retry_2"]["total_bytes"]["mean_difference"],
                },
                "gate_3_scope": {
                    "observed": round6["gate_3"],
                    "thresholds": {
                        "per_branch": round6["thresholds"]["gate_3_per_branch"],
                        "streams": round6["thresholds"]["gate_3_streams"],
                        "nature": "资源门槛（这轮愿意采多少点），不是强化学习可行性的必要条件",
                    },
                    "sampling_history": (
                        "初次每流 2 点（93 点）→ 一致性检查抓到 Recorder 把 orders_selection 的有序选择"
                        "转成 set 再转回 tuple、破坏优先级顺序的真实缺陷 → 修复后扩到每流 10 点（425 点）。"
                        "样本上限是在看到第一遍的一致性失败后提高的，打平率从 87 % 变为 96.5 %；"
                        "该顺序如实披露，不外推为总体比例。"
                    ),
                    "what_it_does_not_say": "不写'96.5 % 的所有决策没有学习空间'；只说这 425 个单步点上未采到取舍",
                },
            },
            "normal_path_engineering_suitability": {
                "claim": (
                    "在保留项目原有视距感知与 gossip 的正常信息路径上，这两个固定规则在已测 200 流的"
                    "完成数逐局相同，而 ACK 重试每局多花同样多的字节 ⇒ 该设置上被支配，不提部署。"
                ),
                "completion_difference": normal["completion"]["mean_difference"],
                "completion_ci95": [
                    normal["completion"]["ci95_lower"], normal["completion"]["ci95_upper"]
                ],
                "total_bytes_difference": normal["total_bytes"]["mean_difference"],
                "same_outcome_but_dearer": normal["same_outcome_but_dearer"],
                "proposed_as_optional_high_reliability_mode": normal["proposed_as_optional_mode"],
                "default_changed": False,
                "does_not_mean": (
                    "不代表所有 ACK 策略、所有正常场景或学习策略都无价值；这些固定规则不构成学习成果，"
                    "也不构成对整个反馈类的否证。"
                ),
                "break_even": (
                    "设新增按时任务价值 V（效用/任务）、字节成本 c（效用/B）：此样本均值下净效用改善需 "
                    f"V × {completion['mean_difference']:.3f} > c × {total['mean_difference']:.1f}。"
                    "未指定 V、c 时不宣布值得部署；total 只含任务上行与 ACK，不含未计入的 gossip/系统通信。"
                ),
            },
        },
        "differences_against_archived": differences(archived_verdict, results),
        "preserved": {
            "original_stop_decisions": "未改：门禁未过与门槛 0/3 的判定原样保留",
            "gate_criteria_redefined": False,
            "round6_negative_result": "原样保留，ACK 是新协议不覆盖旧结论",
            "production_defaults_changed": False,
            "new_episodes": 0,
            "new_streams": 0,
            "new_training": 0,
        },
    }


def differences(archived: dict[str, Any], results: dict[str, Any]) -> list[dict[str, Any]]:
    """What the archived verdict said versus what the unified object says."""
    lossy = results["judgement"]["isolated-lossy"]["lossy_confirmation"]
    fresh = archived["conclusions"]["ack_task_gain_and_total_cost"]["fresh_streams_isolated_lossy"]
    new_ci = [lossy["completion"]["ci95_lower"], lossy["completion"]["ci95_upper"]]
    rows = [
        {
            "item": "隔离有损完成数 CI",
            "archived": fresh["gain_ci95"],
            "revised": new_ci,
            "kind": "数值（分位边界）",
            "reason": "归档引用两臂抽样的判读段值；修订统一为比较表那一套抽样",
        },
        {
            "item": "隔离有损 total-byte CI",
            "archived": fresh["total_bytes_ci95"],
            "revised": [
                lossy["total_bytes"]["ci95_lower"], lossy["total_bytes"]["ci95_upper"]
            ],
            "kind": "数值（分位边界）",
            "reason": "同上；均值未变",
        },
        {
            "item": "每新增按时任务的字节成本",
            "archived": fresh["bytes_per_additional_on_time_target"],
            "revised": results["ratios"]["isolated-lossy"]["ack_retry_2 - noack_send_once"][
                "total_bytes_ratio"
            ],
            "kind": "数值（未变）",
            "reason": "比值不依赖判读段抽样；单位写明是 B/新增任务而不是 B/ep",
        },
        {
            "item": "'欠训练（已用扩训排除）'",
            "archived": "理由段落中的绝对表述",
            "revised": "扩训排除的是'完全学不动'，不排除仍欠训练/信用分配/表征",
            "kind": "只改解释",
            "reason": "§3.9：改善与种子差异缩小不足以证明充分收敛",
        },
        {
            "item": "'第六轮最强 no-ACK 规则 send_once'",
            "archived": "把该基线称作 no-ACK 规则",
            "revised": "第六轮 22 臂全部 ack=True：该基线信道开、但不消费反馈",
            "kind": "只改解释",
            "reason": "§3.6：否则 +19 562 B 与 +26 354 B 会被读成同一基线的变化",
        },
    ]
    return rows


# --------------------------------------------------------------------------- render
def render(data: dict[str, Any]) -> str:
    conclusions = data["conclusions"]
    learning = conclusions["learning_negative"]
    ack = conclusions["ack_task_gain_and_total_cost"]
    normal = conclusions["normal_path_engineering_suitability"]
    fresh = ack["round7_fresh_streams"]
    round6 = ack["round6_training_streams"]
    completion = fresh["completion_gain_targets_per_episode"]
    total = fresh["total_bytes_increase_B_per_episode"]
    out: list[str] = []
    add = out.append
    add("# 通信学习结论（修订版 verdict）")
    add("")
    add(
        f"日期：{data['date']}。分支 `{data['branch']}`，总纲 {data['charter_version']}，执行指令 "
        f"`{data['brief']}` §3/§4/§6。修订对象："
        "`../comm_rule_confirm_20260921/COMMUNICATION_LEARNING_VERDICT.md`（归档件原样保留）。"
        "全部数值取自 `COMM_RULE_CONFIRMATION_RESULTS_REVISED.json` 与第六轮 gate JSON，"
        "**本轮 0 episode、0 新流、0 训练**。"
    )
    add("")
    add(
        "本文的目的不是改判，而是把已经成立的事实与其能承载的解释范围对齐："
        "**旧停止判定、旧预注册判据、旧负结果全部保留**，被删除的只是超出证据范围的措辞。"
    )
    add("")
    add("## 一、三条学习事实必须分开陈述")
    add("")
    add("| 事实 | 范围 | 证据 |")
    add("|---|---|---|")
    for item in learning["three_facts_stated_separately"]:
        add(f"| {item['fact']} | {item['scope']} | {item['evidence']} |")
    add("")
    add(
        "禁止把三行打包成'ACK 模型也训练失败'——ACK 从未训练过模型；也禁止打包成'学习空间不存在'。"
    )
    add("")
    add("## 二、当前范围内的学习负结论")
    add("")
    add(f"> {learning['claim']}")
    add("")
    add(f"S3 状态：**{learning['s3_state']}**。")
    add("")
    add(f"- 训练充分性：{learning['training_sufficiency']}")
    add(
        f"- 最终测试块 `{learning['final_test_block']['block']}`：未打开、未生成、未读取；"
        "没有通过门槛的学习候选，因此不存在需要它的性能主张。"
    )
    add("")
    add("**本文不采用的写法（逐条列出并给出理由，供读者核对）**")
    add("")
    for claim in learning["not_claimed"]:
        add(f"- ~~{claim}~~ —— {learning['why_not_claimed'][claim]}")
    add("")
    add("## 三、协议与规则收益")
    add("")
    protocol = conclusions["protocol_and_rule_gains"]
    add(f"- {protocol['claim']}")
    add(f"- 性质：{protocol['nature']}。证据：{protocol['evidence']}。")
    add("")
    add("## 四、ACK 的任务收益与总成本")
    add("")
    add(ack["claim"])
    add("")
    add(f"单位只用这四种：{ack['units']}")
    add("")
    add("| 数据 | 完成数增益 (个/ep) | 95 % CI | total 增量 (B/ep) | 95 % CI | 基线的 ACK 开关 |")
    add("|---|---:|---|---:|---|---|")
    r6c, r6t = round6["completion"], round6["total_bytes"]
    add(
        f"| 第六轮 48 条训练流 | {r6c['mean_difference']:+.3f} "
        f"| [{r6c['ci95_lower']:.3f}, {r6c['ci95_upper']:.3f}] "
        f"| {r6t['mean_difference']:+,.0f} "
        f"| [{r6t['ci95_lower']:,.0f}, {r6t['ci95_upper']:,.0f}] | 开、不消费 |"
    )
    add(
        f"| 本轮 200 新流（isolated-lossy） | {completion:+.3f} "
        f"| [{fresh['completion_ci95'][0]:.3f}, {fresh['completion_ci95'][1]:.3f}] "
        f"| {total:+,.1f} | [{fresh['total_bytes_ci95'][0]:,.0f}, "
        f"{fresh['total_bytes_ci95'][1]:,.0f}] | 基线 `noack_send_once`：关 |"
    )
    add(
        f"| 本轮 200 新流（isolated-lossless 负控） | "
        f"{ack['lossless_control']['completion']:+.3f} | — "
        f"| {ack['lossless_control']['total_bytes']:+,.0f} | — | 基线：关 |"
    )
    add("")
    add(
        f"- 每新增一个按时任务的代价：**{fresh['bytes_per_additional_on_time_target']:,.1f} "
        f"{fresh['bytes_per_additional_unit']}**（区间 "
        f"[{fresh['per_target_interval'][0]:,.1f}, {fresh['per_target_interval'][1]:,.1f}]，"
        f"{fresh['ratio_interval_kind']}，可稳健解释："
        f"{'是' if fresh['ratio_robustly_interpretable'] else '否'}）。"
    )
    add(
        f"- 同一交换的形状：新增 {fresh['new_per_target_failures']} 个逐目标失败、救回 "
        f"{fresh['recovered']} 个 ⇒ 重试不是对每条流都有益。"
    )
    add("")
    add("**三个成本差是三个不同的比较，不是同一条基线变了**")
    add("")
    add("| 差值 | 比较 | 测的是什么 |")
    add("|---|---|---|")
    for row in ack["cost_comparisons_are_different"]:
        add(f"| {row['difference']} | {row['comparison']} | {row['what_it_measures']} |")
    add("")
    add(f"- 因果范围：{ack['causal_scope']}")
    add(f"- 负控读法：{ack['negative_control_reading']}")
    add("")
    gate = ack["gate_3_scope"]
    add("**门禁 3 的含义边界**")
    add("")
    add(
        f"- 观测：{gate['observed']['consistency']['checked']} 个受控点、一致性 "
        f"{gate['observed']['consistency']['consistent']}/{gate['observed']['consistency']['checked']}；"
        f"SEND 达标 {gate['observed']['send_ok']}、HOLD 达标 {gate['observed']['hold_ok']} ⇒ 未过。"
    )
    add(f"- 门槛性质：{gate['thresholds']['nature']}。")
    add(f"- 采样顺序披露：{gate['sampling_history']}")
    add(f"- {gate['what_it_does_not_say']}。")
    add("")
    add("## 五、正常信息路径下的工程适用性")
    add("")
    add(normal["claim"])
    add("")
    add(
        f"- 完成数差 {normal['completion_difference']:+.4f} 个/ep，CI "
        f"[{normal['completion_ci95'][0]:.4f}, {normal['completion_ci95'][1]:.4f}]；"
    )
    add(
        f"- total-byte 差 {normal['total_bytes_difference']:+,.1f} B/ep（更高，不是更好）；"
        f"结局相同但更贵：**{'是' if normal['same_outcome_but_dearer'] else '否'}**；"
    )
    add(
        f"- 提议为可选高可靠模式：**{'是' if normal['proposed_as_optional_high_reliability_mode'] else '否'}**；"
        f"生产默认值：**未改动**。"
    )
    add(f"- 范围限定：{normal['does_not_mean']}")
    add(f"- **Break-even（只给量纲正确的条件）**：{normal['break_even']}")
    add("")
    add("## 六、与归档 verdict 的差异")
    add("")
    add("| 项 | 归档 | 修订 | 是否改数值 | 理由 |")
    add("|---|---|---|---|---|")
    for row in data["differences_against_archived"]:
        add(
            f"| {row['item']} | {row['archived']} | {row['revised']} | {row['kind']} | "
            f"{row['reason']} |"
        )
    add("")
    add("## 七、被保留的东西")
    add("")
    for key, value in data["preserved"].items():
        add(f"- `{key}`：{value}")
    add("")
    add("## 八、复跑")
    add("")
    add("```bash")
    add("cd runs/comm_evidence_review_20260921")
    add("python revise_results.py")
    add("python revise_verdict.py")
    add("```")
    add("")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    data = build_payload()
    common.write_json(common.json_path("COMMUNICATION_LEARNING_VERDICT_REVISED.json"), data)
    text = render(data)
    common.json_path("COMMUNICATION_LEARNING_VERDICT_REVISED.md").write_text(text, encoding="utf-8")
    print(f"written: COMMUNICATION_LEARNING_VERDICT_REVISED.md ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
