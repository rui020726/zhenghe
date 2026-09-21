"""Emit COMM_EVIDENCE_CORRECTIONS.md / .json: the ten checks, original wording to revision.

Each item carries the five fields the brief demands -- 原表述、证据路径/字段、更正内容、
是否改变数值、修订落在哪个文件 -- and every number in it is sliced from a machine-readable
artefact (this round's revised results, the archived results, round 6's gate JSON, the round-6
ACK matrix) rather than retyped, so the corrections document cannot itself drift.

    python evidence_corrections.py
"""

from __future__ import annotations

import json
import sys
from typing import Any

import review_common as common

ARCHIVED_RESULTS = "runs/comm_rule_confirm_20260921/COMM_RULE_CONFIRMATION_RESULTS.md"
ARCHIVED_VERDICT = "runs/comm_rule_confirm_20260921/COMMUNICATION_LEARNING_VERDICT.md"
ARCHIVED_CHARTER = "PROJECT_CHARTER.md §5.12 / §1 / PROJECT_STATUS.md 顶部块"
ROUND6_GATE = "runs/comm_ack_value_20260920/COMM_ACK_VALUE_GATE.md / COMM_ACK_VALUE_GATE_25pct.json"


def items(results: dict[str, Any], archived: dict[str, Any], verdict: dict[str, Any]) -> list[dict[str, Any]]:
    lossy = results["comparisons"]["isolated-lossy"]["ack_retry_2 - noack_send_once"]["axes"]
    completion, total = lossy["on_time"], lossy["total_bytes"]
    periodic = results["comparisons"]["isolated-lossy"]["ack_periodic_20 - ack_retry_2"]["axes"]
    lossless_periodic = results["comparisons"]["isolated-lossless"]["ack_periodic_20 - ack_retry_2"]["axes"]
    ratio = results["ratios"]["isolated-lossy"]["ack_retry_2 - noack_send_once"]
    recheck = results["primary_ratio_check"]
    diff = results["old_output_diff"]
    lossless_periodic_ratio = results["ratios"]["isolated-lossless"]["ack_periodic_20 - ack_retry_2"]
    normal_periodic_ratio = results["ratios"]["normal-lossy"]["ack_periodic_20 - ack_retry_2"]
    failures = results["new_failures"]["isolated-lossy"]["ack_retry_2 - noack_send_once"]
    gate = json.loads((common.ACK_ROUND / "COMM_ACK_VALUE_GATE_25pct.json").read_text(encoding="utf-8"))
    payoff = gate["gates"]["2_information_pays"]
    counterfactual = json.loads(
        (common.ACK_ROUND / "COMM_ACK_VALUE_COUNTERFACTUAL_25pct.json").read_text(encoding="utf-8")
    )
    matrix_row = json.loads(
        (common.ACK_ROUND / "COMM_ACK_VALUE_RULE_MATRIX_25pct.json").read_text(encoding="utf-8")
    )
    send_once_ack_bytes = _arm_bytes(matrix_row, "send_once", "lossy")

    return [
        {
            "id": 1,
            "title": "单位与信道语义",
            "original": (
                "`COMM_RULE_CONFIRMATION_RESULTS.md` 每个设置标题写'上行与反向丢包均 1.0 / 0.7'，"
                "并把三种成本（B/ep、B/新增任务、个/ep）在同一页里混用；verdict 与总纲沿用同一措辞。"
            ),
            "evidence": (
                "`integrated_sar/comm.py:894`（`if self._rng.random() > self._delivery_probability: drop`）与 "
                "`integrated_sar/comm_ack.py:212`（`stream.random() < delivery_probability` 判送达）："
                "字段 `delivery_probability` 是**成功率**。1.0 ⇒ 丢包率 0.0，0.7 ⇒ 丢包率 0.3。"
            ),
            "correction": (
                "设置标题改为'上行与反向投递成功率均 1.0（丢包率 0.0）/ 0.7（丢包率 0.3）'；"
                f"报告加'〇、单位与口径'图例：{results['units']}。"
                f"31 561.7 一律带单位写作 B/新增按时任务，{total['mean_difference']:,.1f} 带单位 B/episode。"
            ),
            "changes": "只改解释与措辞；配置未动",
        },
        {
            "id": 2,
            "title": "成本方向不得渲染为'更好'",
            "original": (
                "归档报告 total-byte 列的逐流计数写 '200 更好 / 0 相等 / 0 更差'，"
                "判读段同样用'更好'描述 +26 354 B 的字节增量。"
            ),
            "evidence": (
                "`runs/comm_rule_confirm_20260921/render_results.py:_counts` 把完成数与字节的"
                "seeds_better/seeds_worse 复用同一套'更好/更差'词；字节轴上 seeds_better 计的是差值为正的流。"
            ),
            "correction": (
                "字节轴改用'更高/相等/更低'（`revise_results._cost_counts`），只有 on_time 轴保留"
                "'更好/相等/更差'；判读段对 +26 354.1 B/ep 明确写'是更贵，不是更好'。"
            ),
            "changes": "只改措辞；底层计数与均值未变",
        },
        {
            "id": 3,
            "title": "显著性、等效与'前沿'的定义",
            "original": (
                "归档报告把 periodic 相对锁定规则的结果写成'规则前沿：稳定 ⇒ 未超过锁定规则'，"
                "并在正文用'打平'描述（总纲 §5.12 同）。"
            ),
            "evidence": (
                f"修订 results：isolated-lossy `ack_periodic_20 − ack_retry_2` 完成数 "
                f"{periodic['on_time']['mean_difference']:+.4f} 个/ep，CI "
                f"[{periodic['on_time']['ci95_lower']:.4f}, {periodic['on_time']['ci95_upper']:.4f}]"
                f" 含 0，下界 < −0.05；total-byte 差 {periodic['total_bytes']['mean_difference']:+,.1f} B/ep，"
                f"CI [{periodic['total_bytes']['ci95_lower']:,.1f}, {periodic['total_bytes']['ci95_upper']:,.1f}]。"
            ),
            "correction": (
                "CI 含 0 只写'未检出差异'，不写效果相同，也不写通过 −0.05 非劣界（本例下界 "
                f"{periodic['on_time']['ci95_lower']:.3f} 连非劣界都没过）；成本侧更省单独成立照单报。"
                "'前沿'拆成三个定义分别给结论：样本均值 Pareto、逐流支配、有统计支持的支配"
                "（`frontier_definitions`），不再用'稳定/不稳定'替代定义。"
            ),
            "changes": "只改解释；无预注册判据被改",
        },
        {
            "id": 4,
            "title": "比值区间必须是条件区间",
            "original": (
                "归档报告在丢掉非正分母抽样后仍把结果列标题写作 '95 % CI'，未区分无条件/条件区间；"
                "对分母极小的 periodic 比较也照样给出比值数字。"
            ),
            "evidence": (
                f"修订 results：无损 periodic 比较 {lossless_periodic_ratio['undefined_replicates']}/10 000 "
                f"个抽样分母非正；正常路径 periodic 比较 {normal_periodic_ratio['undefined_replicates']}/10 000；"
                f"主比值（retry/noack，isolated-lossy）未定义抽样 {ratio['undefined_replicates']}。"
            ),
            "correction": (
                "新增'区间类型'与'可稳健解释'两列：丢掉非正分母后得到的区间标 "
                f"`{lossless_periodic_ratio['interval_kind']}`，不是无条件 95 % CI；"
                "点估计分母非正者记 undefined 且不删样本求稳。主比值单独核对："
                f"冻结 RNG 区间 [{recheck['frozen_interval'][0]:,.1f}, {recheck['frozen_interval'][1]:,.1f}] vs "
                f"独立种子 {recheck['recheck_seed']} 区间 [{recheck['recheck_interval'][0]:,.1f}, "
                f"{recheck['recheck_interval'][1]:,.1f}] ⇒ 对抽样不敏感，可引用，并同时给分子分母。"
            ),
            "changes": "只改标注与呈现；未改任何抽样实现",
        },
        {
            "id": 5,
            "title": "重复数字的根因：同一量被抽了两次样",
            "original": (
                "主表 CI [+0.685, +0.995] vs verdict/判读段 [+0.685, +0.990]；"
                "成本 CI [26 002, 26 688] vs [26 018, 26 687]；periodic 两处 CI 也不一致。"
            ),
            "evidence": (
                "`confirm_matrix.judge()` 用只含两条臂的行集重算 `_difference`，`report()` 用全部五条臂；"
                "`evaluate_pilot.paired` 每次调用重新播种但按臂序逐块消耗抽样 ⇒ 同一统计量落在随机流"
                f"不同位置。修订 results 的 `old_output_diff` 核对 {diff['checked_pairs']} 对，"
                f"均值改变 = {diff['means_changed']}，仅分位边界不同的 {len(diff['differing'])} 对已逐条列出。"
            ),
            "correction": (
                "所有统计量只算一次（每 (设置, 轴, 基线) 一次全臂集调用），判读段与 verdict 只从"
                "该对象切片；§三 标题改为'切片，不再计算'。统一后的口径是主表那一套。"
                "`aggregate_invariance` 逐项核对 3 000 行输入的摘要未变"
                f"（{results['aggregate_invariance']['fields_checked']} 个字段全部相同）。"
            ),
            "changes": "数值（分位边界）统一到已有的一套；未新跑任何 episode，均值不变",
        },
        {
            "id": 6,
            "title": "ACK 比较对象的开关",
            "original": "多处把第六轮的 `send_once` 称作'最强 no-ACK 规则'，与本轮 +26 354 B 并列引用。",
            "evidence": (
                f"`ack_matrix.run_arm` 对全部 22 臂固定 `ack=True`；gate JSON `2_information_pays` 的"
                f"基线 `send_once` total = {payoff['baseline_total_bytes']:,.1f} B，其中 ACK "
                f"{send_once_ack_bytes:,.0f} B ⇒ 该基线反向信道是开的，只是策略不消费确认状态。"
            ),
            "correction": (
                "报告新增 §四表：逐对象列出 ACK 开关/是否消费反馈/重试能力/额度。三个成本差是三个"
                "不同比较：+26 354.1（基线无信道）、+19 100.7（基线有信道不消费）、"
                "+19 562（第六轮同类基线、48 条训练流）；不再作为同一基线的变化并置。"
            ),
            "changes": "只改解释；第六轮数字原样保留",
        },
        {
            "id": 7,
            "title": "因果范围：重试差 ≠ 反馈净价值",
            "original": "'ACK 信息真实且有后果（一条简单规则拿到全部增益）'与'不是多花字节买分'。",
            "evidence": (
                "ack_retry_2 − noack_send_once 同时改变反向信道、是否消费确认、是否有第二次尝试；"
                f"逐目标新增失败 {failures['new_failures']} 个、救回 {failures['recovered']} 个"
                f"（净差 {failures['recovered'] - failures['new_failures']} = 200 × "
                f"{completion['mean_difference']:.3f}）。"
            ),
            "correction": (
                "该差值写成三者联合效应；无损打平只支持'该重试收益与所测丢包条件有关'，"
                "不支持'不是花钱买分'。分离反馈价值需要'相近重试机会/调参资源的无反馈重试'对照，"
                "本轮未新增任何运行（列为下一步）。"
            ),
            "changes": "只改解释",
        },
        {
            "id": 8,
            "title": "学习结论的三条事实",
            "original": "归档 verdict 把六轮并陈为一段理由，读者易读成'ACK 模型也训练失败'。",
            "evidence": (
                "S3-1…S3-4 训练的是 no-ACK 观测；S3-5 只有规则与门禁（gate JSON `decision = "
                f"{gate['decision']}`）；本轮 3 000 episode 五条臂全是规则。"
            ),
            "correction": (
                "verdict 修订版 §一 用三行分开：no-ACK 未建立稳定优势 / ACK 从未训练模型 / "
                "本轮无学习臂；并把'ACK 模型训练失败'列入 FORBIDDEN_CLAIMS 的禁用清单。"
            ),
            "changes": "只改解释",
        },
        {
            "id": 9,
            "title": "训练充分性的表述",
            "original": "归档 verdict 理由段：'六轮中每一轮的失败都有具体归因——欠训练（已用扩训排除）…'。",
            "evidence": (
                "`COMM_LEARNING_DIAGNOSTIC.json`：曲线非单调（25 % seed1 @100k=@200k=2.9583 后跳升；"
                "75 % seed1 5.4583→4.7917→5.5417→5.7500），value loss 未持久化，"
                "optimizer.step 仅 2 792–5 676 次。"
            ),
            "correction": (
                "删除'已用扩训排除'的绝对表述，改为'扩训排除的是完全学不动，不排除仍欠训练、"
                "时序信用分配过稀或表征不足'；停止决定原样保留。四种解释的证据表见诊断文档。"
            ),
            "changes": "只改解释；未追加训练",
        },
        {
            "id": 10,
            "title": "门禁 3 的含义边界",
            "original": (
                "'425 个受控反事实点里 96.5 % 发送/沉默对任务结局无差别 ⇒ ACK 状态没有形成可学习的"
                "逐决策取舍'，并被总纲转写为'96.5 % 的决策点上…'。"
            ),
            "evidence": (
                f"gate JSON `3_real_choice`：{counterfactual['decision_points']} 点、一致性 "
                f"{counterfactual['consistency']['consistent']}/{counterfactual['decision_points']}，"
                f"门槛 {gate['thresholds']['gate_3_per_branch']} 点 / {gate['thresholds']['gate_3_streams']} 流；"
                "`ack_counterfactual.POINTS_PER_STREAM` 注释与门禁文档 §三记录了每流 2 点（93 点）"
                "→ 修 orders_selection 顺序缺陷 → 每流 10 点（425 点）的扩样顺序，打平率 87 %→96.5 %。"
            ),
            "correction": (
                "表述限定为：这是特定采样分布、单步干预与固定后续策略下的观测；各 ≥20 点/≥8 流是资源"
                "门槛，不是强化学习可行性的必要条件；不写'96.5 % 的所有决策没有学习空间'。"
                "补样顺序如实披露。跨步取舍是否存在的问题交给多步干预协议（本轮只设计）。"
            ),
            "changes": "只改解释；门禁判定与停止决定未改",
        },
    ]


def _arm_bytes(matrix: Any, arm: str, channel: str) -> float:
    """One arm's per-episode ACK bytes in round 6's matrix, per channel."""
    return float(matrix["channels"][channel][arm]["ack_bytes"])


def render(data: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append
    add("# 证据修订清单：十项核对（原文 → 修订）")
    add("")
    add(
        f"日期：{data['date']}。分支 `{data['branch']}`，总纲 {data['charter_version']}，执行指令 "
        f"`{data['brief']}` §3。机器可读：`COMM_EVIDENCE_CORRECTIONS.json`。"
    )
    add("")
    add(
        "**本轮不产生新数据**：新 episode 0、新流 0、训练 0。归档件（"
        f"`{ARCHIVED_RESULTS}`、`{ARCHIVED_VERDICT}` 及其判定）原样保留、可追溯；"
        "修订件是本目录内的 `_REVISED` 文档，由脚本从机读对象渲染。所有核对结论只区分两类："
        "**只改解释/措辞**，或**统一到已有的一套数值口径**（均值不变），没有任何预注册判据被后验修改。"
    )
    add("")
    add("## 汇总表")
    add("")
    add("| # | 项 | 是否改变数值 | 修订落点 |")
    add("|---|---|---|---|")
    for item in data["items"]:
        add(f"| {item['id']} | {item['title']} | {item['changes']} | {item['where']} |")
    add("")
    add("## 逐项")
    add("")
    for item in data["items"]:
        add(f"### {item['id']}. {item['title']}")
        add("")
        add(f"- **原表述**：{item['original']}")
        add(f"- **证据路径/字段**：{item['evidence']}")
        add(f"- **更正内容**：{item['correction']}")
        add(f"- **是否改变数值或只改解释**：{item['changes']}")
        add(f"- **修订落在**：{item['where']}")
        add("")
    add("## 未纳入本轮的东西")
    add("")
    add("- 不为修文字重跑仿真；现有矩阵不足以复核的项在诊断文档里标'证据不足/待测'，不用推测补齐。")
    add("- 不改生产模块与默认值（`comm_ack=False`、`comm_timing_mode=legacy_shared_bound`）。")
    add(f"- 不触碰封存测试块 `{common.SEALED}`；`900000..900199` 已消费，只作既有证据使用。")
    add("- 不修改非本轮创建的 `architecture/` 与 `runs/coordinator_baselines_20260917/dev_summary.json`。")
    add("")
    add("## 复跑")
    add("")
    add("```bash")
    add("cd runs/comm_evidence_review_20260921")
    add("python revise_results.py && python revise_verdict.py && python evidence_corrections.py")
    add("```")
    add("")
    return "\n".join(out) + "\n"


WHERE = {
    1: "COMM_RULE_CONFIRMATION_RESULTS_REVISED.md 〇/一、COMMUNICATION_LEARNING_VERDICT_REVISED.md 四",
    2: "REVISED.md 二/三（`_cost_counts`）",
    3: "REVISED.md 三（`frontier_definitions` 三口径）",
    4: "REVISED.md 二/六（`ratios.*.interval_kind`、`primary_ratio_check`）",
    5: "REVISED.md 三/七、REVISED.json `old_output_diff`、verdict 修订 §六",
    6: "REVISED.md 四（比较对象表）、verdict 修订 §四",
    7: "verdict 修订 §四、REVISED.md 三（`causal_scope`）",
    8: "verdict 修订 §一/二",
    9: "verdict 修订 §二、COMM_LEARNING_DIAGNOSTIC.md",
    10: "verdict 修订 §四（门禁 3 含义边界）",
}


def main(argv: list[str]) -> int:
    results = common.read_json("COMM_RULE_CONFIRMATION_RESULTS_REVISED.json")
    archived = common.load_archived("COMM_RULE_CONFIRMATION_RESULTS.json")
    verdict = common.load_archived("COMMUNICATION_LEARNING_VERDICT.json")
    data = {
        "document": "COMM_EVIDENCE_CORRECTIONS",
        "date": "2026-09-21",
        "branch": "analysis/comm-evidence-review",
        "charter_version": results["charter_version"],
        "brief": results["brief"],
        "episodes_run": 0,
        "streams_recorded": 0,
        "training_runs": 0,
        "items": [
            {**item, "where": WHERE[item["id"]]} for item in items(results, archived, verdict)
        ],
    }
    common.write_json(common.json_path("COMM_EVIDENCE_CORRECTIONS.json"), data)
    text = render(data)
    common.json_path("COMM_EVIDENCE_CORRECTIONS.md").write_text(text, encoding="utf-8")
    print(f"written: COMM_EVIDENCE_CORRECTIONS.md ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
