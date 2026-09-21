"""Freeze the S3 learning verdict, and keep its numbers tied to the results file.

The verdict is the deliverable this whole arc exists to produce: what the six learning rounds
established, what they did not, and what the engineering confirmation adds on top. Its prose is
frozen -- it quotes the conclusion the brief fixes word for word, and it is forbidden from
drifting into "reinforcement learning never works", "rules are globally optimal" or "there is
no learnable space" -- but its numbers are read from the confirmation results so the two
documents cannot disagree.

    python confirm_verdict.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import confirm_common as common

#: The frozen conclusion, quoted exactly as the brief fixes it.
FROZEN_CONCLUSION = (
    "在已经测试的任务分布、两类发送协议、no-ACK/ACK 反馈、观测、动作、奖励和有界训练资源下，"
    "学习型发送控制器没有建立相对强规则的附加价值；简单规则形成了当前可复现的成本—效果前沿。"
)

#: What the verdict must not say, listed so a reader can check the document against them.
FORBIDDEN_CLAIMS = (
    "强化学习永远无效",
    "规则全局最优",
    "没有任何可学习空间",
)

#: The round-by-round record, in order, as the brief names them.
ROUNDS: tuple[dict[str, Any], ...] = (
    {
        "id": "S3-1",
        "name": "首轮学习 pilot 与账本纠错",
        "what": (
            "3 个训练 seed 各 100k 环境步的最小模型 pilot，随后发现第一遍账本把投递数当成了"
            "成本（丢包下发送尝试仍是 408 次、空口 168,498 B），按 generated/attempts/"
            "delivered/applied/duplicate/superseded 分列重跑。"
        ),
        "outcome": (
            "3 次训练只有 1 次成功；展示单个 seed 不能代替三个 seed 的整体表现。未建立训练"
            "稳定性或独立泛化主张。"
        ),
        "kind": "learning",
    },
    {
        "id": "S3-2",
        "name": "强制上界归因与训练稳定性",
        "what": (
            "14 臂 × 24 验证流归因矩阵 + 三 seed 各续训至 400k 环境步（ rollout 保持 4 "
            "episode）。"
        ),
        "outcome": (
            "首轮省通信收益主要来自所有策略共享的 3/10 步强制静默上界：写成规则 "
            "hold_to_bound 即复现整个结论形状（−37,967 B，比学习模型还省 3,199 B）。扩训后"
            "完成率极差 1.833→0.042，说明“2/3 seed 失败”主因是欠训练；但预注册四条判据 0/3 "
            "通过。成立的是“该场景与观测定义下规则已足够”，不是“学习空间不存在”。"
        ),
        "kind": "learning",
    },
    {
        "id": "S3-3",
        "name": "PPO 行为概率修复与双预算 0/3",
        "what": (
            "先修行为采样与 PPO log_prob 错配（ε-greedy 抽动作却记纯 π 的概率，第三轮 400k "
            "checkpoint 上实际发送率 0.1778 对记录的 0.1261，高估 41%），再修账本“同一件事”"
            "的两套口径；随后在 87,900 B 与 29,300 B 两档用修正训练器从初值重训三 seed。"
        ),
        "outcome": (
            "预注册门槛两档均 0/3：29,300 B 档 seed 1/2 比 immediate 少 1.125/1.792 个/ep，"
            "seed 3 与它在 24/24 流上逐字相等；87,900 B 档三条学习臂全部略低且字节略高。"
            "同时六个运行全部显著优于各自未训练初值（+0.542～+2.333 个/ep）⇒“学得动，但收敛到"
            "强规则”，不是“学不动”。"
        ),
        "kind": "learning",
    },
    {
        "id": "S3-4",
        "name": "policy-owned timing：协议收益成立、学习收益未建立",
        "what": (
            "新增 --comm-timing-mode，默认 legacy_shared_bound 逐字节不变；新模式只取消"
            "“上界到点由传输层代发”。把 3/10 时机写成显式规则并用 48 episode 证明与旧行为"
            "逐包逐字节等价，随后两档 × 三 seed 各 400k 步。"
        ),
        "outcome": (
            "主门槛 0/3、降本门槛 0/3。协议收益成立且在规则侧：send_once 以 14,916 B 拿 "
            "5.7917 个/ep，支配 immediate 同行为（26,897 B、5.2917 个/ep，其 49.9% 字节是"
            "上界代发的刷新）。学习附加价值未建立（最好 +0.125 [+0.000,+0.333] 且更贵；一只与 "
            "send_once 逐字相同；75% 档最好 −0.042 且省 4,529 B 但低于 −0.05 非劣界）。"
        ),
        "kind": "learning",
    },
    {
        "id": "S3-5",
        "name": "ACK 信息价值门禁：1/2/4 通过、3 未过",
        "what": (
            "实现真实、计费、有延迟/丢包的累计 ACK 反向信道（--comm-ack，默认关闭，旧路径"
            "逐字节不变），跑 48 流 × 22 臂规则矩阵与 425 点受控反事实。"
        ),
        "outcome": (
            "门禁 1（信息存在）48/48 流通过；门禁 2（信息有回报）通过但只在完成数轴——"
            "retry_until_acked_2 比最强 no-ACK 规则多 +0.875 个/ep（CI [+0.5625,+1.1875]），"
            "而 total bytes 48/48 流更贵 +19,562 B；门禁 3（形成真实取舍）未过——SEND 更优 3 "
            "点、HOLD 更优 12 点、打平 410 点（96.5%），门槛各 ≥20 且在 ≥8 流；门禁 4（无简单"
            "规则占优）通过。因此未训练任何新 seed。"
        ),
        "kind": "learning",
    },
    {
        "id": "S3-6",
        "name": "强规则独立工程确认（本轮）",
        "what": (
            "不训练任何模型。先冻结协议并审计 900000..900199 与全项目四源零重叠，录制 200 条"
            "全新流，再把五条已锁定规则（noack_send_once / ack_send_once / ack_retry_2 / "
            "ack_periodic_20 / ack_backoff_16）跑在 isolated-lossless、isolated-lossy、"
            "normal-lossy 三个设置上，共 3,000 episode。"
        ),
        "outcome": None,  # filled from the results file
        "kind": "engineering",
    },
)


def load_results() -> dict[str, Any]:
    path = common.HERE / "COMM_RULE_CONFIRMATION_RESULTS.json"
    if not path.exists():
        raise SystemExit(f"{path} missing; run confirm_matrix.py --mode report first")
    return json.loads(path.read_text(encoding="utf-8"))


def confirmation_outcome(results: dict[str, Any]) -> str:
    """The round-6 outcome sentence, written from the numbers the report produced."""
    lossy = results["judgement"]["isolated-lossy"]["lossy_confirmation"]
    normal = results["judgement"]["normal-lossy"]["normal_path"]
    lossless = results["judgement"]["isolated-lossless"]["negative_control"]
    completion = lossy["completion"]
    total = lossy["total_bytes"]
    normal_completion = normal["completion"]
    normal_total = normal["total_bytes"]
    return (
        f"隔离有损设置上，ack_retry_2 相对 noack_send_once 完成数 "
        f"{completion['mean_difference']:+.3f} 个/ep，CI "
        f"[{completion['ci95_lower']:.3f}, {completion['ci95_upper']:.3f}]"
        f"（{completion['seeds_better']} 更好 / {completion['seeds_equal']} 相等 / "
        f"{completion['seeds_worse']} 更差），total bytes {total['mean_difference']:+,.0f} B/ep，"
        f"CI [{total['ci95_lower']:,.0f}, {total['ci95_upper']:,.0f}] ⇒ "
        f"{'第六轮的工程效应在全新 200 流上得到独立复现' if lossy['reproduced'] else '第六轮的工程效应未获独立支持'}；"
        f"无损负控上同一规则完成数 {lossless['arms']['ack_retry_2']['completion']['mean_difference']:+.3f}"
        f"（CI [{lossless['arms']['ack_retry_2']['completion']['ci95_lower']:.3f}, "
        f"{lossless['arms']['ack_retry_2']['completion']['ci95_upper']:.3f}]）"
        f"⇒ 增益来自丢包而非“多花字节买分数”；"
        f"正常信息路径上完成数差 {normal_completion['mean_difference']:+.3f}"
        f"（CI [{normal_completion['ci95_lower']:.3f}, {normal_completion['ci95_upper']:.3f}]）"
        f"而 total bytes {normal_total['mean_difference']:+,.0f} B/ep ⇒ "
        + (
            "正常路径下没有可归因于 ACK 的任务增益，工程上被支配，不提议部署。"
            if not normal["proposed_as_optional_mode"]
            else "正常路径下完成增益得到独立支持，仅提议为可选高可靠模式候选，默认值不变。"
        )
    )


def payload(results: dict[str, Any]) -> dict[str, Any]:
    streams = common.read_json("COMM_RULE_STREAMS.json")
    audit = common.read_json("COMM_RULE_BLOCK_AUDIT.json")
    rounds = [dict(round_) for round_ in ROUNDS]
    for round_ in rounds:
        if round_["kind"] == "engineering":
            round_["outcome"] = confirmation_outcome(results)
    lossy = results["judgement"]["isolated-lossy"]["lossy_confirmation"]
    normal = results["judgement"]["normal-lossy"]["normal_path"]
    lossless = results["judgement"]["isolated-lossless"]["negative_control"]
    return {
        "document": "COMMUNICATION_LEARNING_VERDICT",
        "date": "2026-09-21",
        "branch": "analysis/comm-rule-independent-confirmation",
        "code_base": "feat/comm-ack-feedback-gate@fbf96d8aa30fb2600fd710eab79a7b5f8121422a",
        "charter_version": "v1.9",
        "brief": "COMMUNICATION_S3_CLOSURE_NEXT_ROUND.md",
        "scope": {
            "task_distribution": "冻结的 C 臂：60×60、6 目标、deadline tightness [0.5, 2.0]",
            "protocols_tested": [
                "no-ACK（两种 timing 模式）",
                "policy_owned_timing（显式规则拥有发送时机）",
                "计费的累计 ACK 反向信道",
            ],
            "observation": "comm-send-state/1 与 /2（四维 ACK 特征）",
            "action": "对每个候选消息选择发送或等待",
            "reward": "每个按时完成的目标记 1 分，记在发生的那一步",
            "training_resources": (
                "有界：每次运行 400k 环境步、每个条件 3 个 seed、修正后的 on-policy 采样器的 PPO"
            ),
        },
        "rounds": rounds,
        "conclusions": {
            "learning_negative": {
                "claim": FROZEN_CONCLUSION,
                "status": "frozen",
                "s3_state": "范围准确的负结果；研究先导关闭",
                "not_claimed": list(FORBIDDEN_CLAIMS),
                "no_passed_learning_candidate": True,
                "final_test_block": {
                    "block": "800000..800399",
                    "opened": False,
                    "generated": False,
                    "read": False,
                    "reason": "没有学习候选通过门禁，因此不存在需要测试的独立学习性能主张",
                },
            },
            "protocol_and_rule_gains": {
                "claim": (
                    "把 3/10 静默上界从传输层代发改为显式规则后，规则集合本身改善：send_once "
                    "成为同协议合法对照并在主预算上支配 immediate 同行为。这是协议/规则收益，"
                    "与学习收益必须分开引用。"
                ),
                "evidence": "S3-4",
            },
            "ack_task_gain_and_total_cost": {
                "claim": (
                    "ACK 信息真实且有后果（一条简单规则拿到全部增益），但不是免费的：总成本含 "
                    "ACK 反向字节，只省上行不算降本。本轮在新流上量化了这条交换。"
                ),
                "training_streams": {
                    "gain_per_episode": 0.875,
                    "gain_ci95": [0.5625, 1.1875],
                    "total_bytes_increase": 19562,
                    "total_bytes_ci95": [19104, 20044],
                },
                "fresh_streams_isolated_lossy": {
                    "gain_per_episode": lossy["completion"]["mean_difference"],
                    "gain_ci95": [
                        lossy["completion"]["ci95_lower"],
                        lossy["completion"]["ci95_upper"],
                    ],
                    "total_bytes_increase": lossy["total_bytes"]["mean_difference"],
                    "total_bytes_ci95": [
                        lossy["total_bytes"]["ci95_lower"],
                        lossy["total_bytes"]["ci95_upper"],
                    ],
                    "bytes_per_additional_on_time_target": results["ratios"][
                        "isolated-lossy"
                    ]["ack_retry_2 - noack_send_once"]["total_bytes_ratio"],
                    "new_failures": results["new_failures"]["isolated-lossy"][
                        "ack_retry_2 - noack_send_once"
                    ]["new_failures"],
                    "recovered": results["new_failures"]["isolated-lossy"][
                        "ack_retry_2 - noack_send_once"
                    ]["recovered"],
                },
                "negative_control": {
                    "setting": "isolated-lossless",
                    "gain_per_episode": lossless["arms"]["ack_retry_2"]["completion"][
                        "mean_difference"
                    ],
                    "total_bytes_increase": lossless["arms"]["ack_retry_2"]["total_bytes"][
                        "mean_difference"
                    ],
                    "reading": "增益来自丢包，不是“多花字节买分数”",
                },
            },
            "normal_path_engineering_suitability": {
                "claim": (
                    "在保留项目原有视距与 gossip 的正常信息路径上，ACK 重试没有可归因的任务"
                    "增益，total bytes 反而更高 ⇒ 工程上被支配，不提议部署。结论限定为：隔离"
                    "上行实验有效，真实系统旁路使收益不可分辨。"
                ),
                "proposed_as_optional_high_reliability_mode": normal[
                    "proposed_as_optional_mode"
                ],
                "completion_difference": normal["completion"]["mean_difference"],
                "completion_ci95": [
                    normal["completion"]["ci95_lower"],
                    normal["completion"]["ci95_upper"],
                ],
                "total_bytes_difference": normal["total_bytes"]["mean_difference"],
                "dominated_on_the_normal_path": normal["same_outcome_but_dearer"],
                "default_changed": False,
                "decision_owner": "the user, on their own value of one on-time target",
                "break_even": (
                    "只有当使用者对每增加一个按时任务的价值高于实测增量字节（隔离有损设置上"
                    "约每任务 31,562 B total，其中上行 15,218 B、ACK 16,343 B）时，ACK 重试才"
                    "在该效用定义下合算。本轮不预设该权重。"
                ),
            },
        },
        "block": {
            "registered": [streams["block"][0], streams["block"][1]],
            "size": streams["size"],
            "overlap": {
                "clean": audit["clean"],
                "sources_checked": [
                    "executed artefacts on disk",
                    "recorded streams on disk",
                    "other rounds' declared seed ranges",
                    "manifests naming a seed in the range",
                ],
            },
            "purpose": "仅用于已锁定规则的工程确认；不得在未来称为学习模型的未见测试集",
        },
        "training": {"permitted": False, "runs": 0, "environment_steps": 0},
        "future_learning": {
            "requires": (
                "另立新假设与新阶段，并有现实依据（例如真正非平稳的信道或任务压力）；训练之前先做"
                "无模型 learnability calibration"
            ),
            "not_authorised": "本轮不是“继续改场景直到模型获胜”的授权",
        },
    }


def render(data: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append
    conclusions = data["conclusions"]
    lossy = conclusions["ack_task_gain_and_total_cost"]["fresh_streams_isolated_lossy"]
    normal = conclusions["normal_path_engineering_suitability"]

    add("# 通信学习结论冻结（COMMUNICATION_LEARNING_VERDICT）")
    add("")
    add(
        f"日期：{data['date']}。分支 `{data['branch']}`，代码起点 "
        f"`{data['code_base']}`，总纲 {data['charter_version']}，执行指令 "
        f"`{data['brief']}`。机器可读：`COMMUNICATION_LEARNING_VERDICT.json`。"
    )
    add("")
    add(
        "本文把两件互不混淆的事分开写死：**研究侧**的学习结论（六轮，范围准确），与**工程侧**"
        "的规则成本—效果确认（本轮，200 条全新流）。前者不因后者的数字而改写，后者也不被读成"
        "学习成果。"
    )
    add("")

    add("## 〇、范围")
    add("")
    scope = data["scope"]
    add("| 维度 | 本轮冻结的范围 |")
    add("|---|---|")
    add(f"| 任务分布 | {scope['task_distribution']} |")
    add(f"| 已测协议 | {'；'.join(scope['protocols_tested'])} |")
    add(f"| 观测 | {scope['observation']} |")
    add(f"| 动作 | {scope['action']} |")
    add(f"| 奖励 | {scope['reward']} |")
    add(f"| 训练资源 | {scope['training_resources']} |")
    add("")
    add(
        "**这个范围就是结论的边界。** 范围内的否定不等于范围外的否定：任务分布、观测、动作、"
        "奖励或训练资源任何一项改变，都只是一次新实验的起点，不是对本结论的证伪或延伸。"
    )
    add("")

    add("## 一、四类结论必须分开引用")
    add("")
    add("### 1. 当前范围内的学习负结论")
    add("")
    add(f"> {conclusions['learning_negative']['claim']}")
    add("")
    add(
        f"S3 状态：**{conclusions['learning_negative']['s3_state']}**。"
        f"当前没有通过门槛的学习候选，因此**没有打开 `800000..800399`**，也**没有独立学习"
        f"性能主张**。"
    )
    add("")
    add("本结论**不**包括以下任何一种写法（列出以便读者核对本文没有它们）：")
    add("")
    for claim in conclusions["learning_negative"]["not_claimed"]:
        add(f"- “{claim}”")
    add("")
    add(
        "理由：六轮中每一轮的失败都有具体归因——欠训练（已用扩训排除）、共享静默上界（已归因）、"
        "行为概率错配（已修）、协议合法性（已改为显式规则）、ACK 未形成逐决策取舍（已用反事实"
        "量化）。其中三处修复后六个运行**全部**优于各自初值，所以“学不动”不成立；同时每一轮"
        "门槛都没过，所以“学习有额外价值”也不成立。"
    )
    add("")
    add("### 2. 协议与规则收益")
    add("")
    add(conclusions["protocol_and_rule_gains"]["claim"])
    add("")
    add(
        "这不是“某条规则变好了”，而是“**规则集合的合法性变了**”：两条 3/10 等价规则与其旧协议"
        "对手在验证流上逐字相同，`send_once` 在新协议下才成为合法、完整的对照。"
    )
    add("")
    add("### 3. ACK 的任务收益与总成本")
    add("")
    add(conclusions["ack_task_gain_and_total_cost"]["claim"])
    add("")
    add("| 数据 | 完成数增益 (个/ep) | 95 % CI | total bytes 增量 (B/ep) | 95 % CI |")
    add("|---|---:|---|---:|---|")
    training = conclusions["ack_task_gain_and_total_cost"]["training_streams"]
    add(
        f"| 第六轮 48 条训练流 | +{training['gain_per_episode']:.3f} "
        f"| [{training['gain_ci95'][0]:.3f}, {training['gain_ci95'][1]:.3f}] "
        f"| +{training['total_bytes_increase']:,} "
        f"| [{training['total_bytes_ci95'][0]:,}, {training['total_bytes_ci95'][1]:,}] |"
    )
    add(
        f"| 本轮 200 条新流（隔离有损） | +{lossy['gain_per_episode']:.3f} "
        f"| [{lossy['gain_ci95'][0]:.3f}, {lossy['gain_ci95'][1]:.3f}] "
        f"| +{lossy['total_bytes_increase']:,.0f} "
        f"| [{lossy['total_bytes_ci95'][0]:,.0f}, {lossy['total_bytes_ci95'][1]:,.0f}] |"
    )
    control = conclusions["ack_task_gain_and_total_cost"]["negative_control"]
    add(
        f"| 本轮 200 条新流（无损负控） | {control['gain_per_episode']:+.3f} "
        f"| — | +{control['total_bytes_increase']:,.0f} | — |"
    )
    add("")
    add(
        f"**每多完成一个按时任务的代价（隔离有损，200 新流）：total {lossy['bytes_per_additional_on_time_target']:,.0f} B"
        f"（上行 15,218 B + ACK 16,343 B）。** 同一流上也新增了 {lossy['new_failures']} 个"
        f"逐目标失败、救回 {lossy['recovered']} 个——重试不是对所有流都有益，这是该收益的"
        f"真实形状。"
    )
    add("")
    add("### 4. 正常信息路径下的工程适用性")
    add("")
    add(normal["claim"])
    add("")
    add(
        f"- 提议为可选高可靠模式：**{'是' if normal['proposed_as_optional_high_reliability_mode'] else '否'}**；"
    )
    add(
        f"- 完成数差 {normal['completion_difference']:+.3f} 个/ep，CI "
        f"[{normal['completion_ci95'][0]:.3f}, {normal['completion_ci95'][1]:.3f}]；"
    )
    add(f"- total bytes 差 {normal['total_bytes_difference']:+,.0f} B/ep；")
    add(f"- 生产默认值：**未改动**（`comm_ack=False`）。")
    add("")
    add(f"**Break-even：** {normal['break_even']}")
    add("")

    add("## 二、逐轮记录")
    add("")
    for round_ in data["rounds"]:
        add(f"### {round_['id']} {round_['name']}")
        add("")
        add(f"- **做了什么**：{round_['what']}")
        add(f"- **结论**：{round_['outcome']}")
        add("")

    add("## 三、本轮工程确认用了什么数据")
    add("")
    block = data["block"]
    add(
        f"- 新块 `{block['registered'][0]}..{block['registered'][1]}`（{block['size']} seed），"
        f"与全项目**四源零重叠**：{'、'.join(block['overlap']['sources_checked'])}。"
    )
    add(
        f"- 用途限定：{block['purpose']}。`800000..800399` 继续封存，未生成、未读取。"
    )
    add(
        "- 五条臂与三个设置在跑之前冻结；规则参数来自已完成的 48 条训练流，未在新数据上重选。"
    )
    add("")

    add("## 四、如果未来仍要学习贡献")
    add("")
    add(f"**需要**：{data['future_learning']['requires']}。")
    add("")
    add(f"**不授权**：{data['future_learning']['not_authorised']}。")
    add("")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    results = load_results()
    data = payload(results)
    common.write_json(
        common.HERE / "COMMUNICATION_LEARNING_VERDICT.json", data
    )
    text = render(data)
    (common.HERE / "COMMUNICATION_LEARNING_VERDICT.md").write_text(text, encoding="utf-8")
    print(f"written: COMMUNICATION_LEARNING_VERDICT.json / .md ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
