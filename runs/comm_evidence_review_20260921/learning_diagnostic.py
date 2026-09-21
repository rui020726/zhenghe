"""Diagnose why the learned sender has not beaten the strong rules, using only saved records.

No episode is run and no checkpoint is extended: every number here is parsed out of round 5's
training artefacts, which are the only runs trained with the corrected on-policy sampler under
``policy_owned_timing`` (six runs: two budget tiers x three seeds). Anything the training loop
never saved is written as ``missing``, not estimated.

The output is deliberately not a verdict. It is an evidence table over four explanations --
undertraining, insufficient observation, temporal credit assignment, and "the strong rule is
already at the achievable effect for this state" -- which are allowed to coexist, with what
would separate them listed as to-measure. ``COMM_MULTISTEP_DIAGNOSTIC_PROTOCOL.md`` is the
experiment that one of those lines of evidence points at.

    python learning_diagnostic.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import review_common as common

TRAIN_ROOT = common.PROTOCOL_ROUND / "train"
EVAL_ROOT = common.PROTOCOL_ROUND / "evaluation"

#: The six runs trained with the fixed sampler under policy-owned timing.
RUNS: tuple[tuple[str, int], ...] = tuple(
    (tier, seed) for tier in ("25pct", "75pct") for seed in (1, 2, 3)
)

#: One rollout line of the training log, with the quantities the loop actually printed.
UPDATE_PATTERN = re.compile(
    r"steps=\s*(?P<steps>\d+) on_time=(?P<on_time>\d+)/(?P<on_time_total>\d+) "
    r"bytes=(?P<bytes>\d+) use=(?P<use>[\d.]+) free=(?P<free>\d+)/(?P<terms>\d+) "
    r"send_rate=(?P<send_rate>[\d.]+) samples=(?P<samples>\d+) opt=(?P<opt>\d+) "
    r"kl=(?P<kl>[\d.]+) clip=(?P<clip>[\d.]+) ent=(?P<ent>[\d.]+) "
    r"ent_free=(?P<ent_free>[\d.]+) ev=(?P<ev>-?[\d.]+)"
)
VALIDATION_PATTERN = re.compile(r"validation (\{.*\})\s*$")

#: What the /1 observation offers the sender, and what only the evaluator can see.
OBSERVATION_LEGALITY: tuple[dict[str, str], ...] = (
    {
        "question": "这条消息有多紧急（离 deadline 多远）",
        "legal_features": "remaining_time_over_horizon、urgency_now（belief 里的截止时间，不是注册表真值）",
        "caveat": "截止时间随消息到达，晚到的视图带来晚到的紧急度；紧急度不是真值剩余时间",
    },
    {
        "question": "还剩多少预算",
        "legal_features": "budget_remaining_fraction、capacity_remaining_fraction",
        "caveat": "是发送端自己账本的余额，与规则臂同一条执行器；不含未来到达包的字节预估",
    },
    {
        "question": "反馈不确定性（接收端到底收到没有）",
        "legal_features": "/2 的 confirmed_fraction、max_ack_age_scaled、max_view_gap_scaled、"
        "unacked_broadcasts_scaled；无 ACK 时显式 unknown(-1.0)",
        "caveat": "只由自身发送历史与已到达 ACK 推出；在途包、接收 belief、真实丢包结果均不可见",
    },
    {
        "question": "这条视图是不是新闻",
        "legal_features": "changed_view、first_emission、stage_changed、requirements_changed、"
        "members_changed、progress_delta_abs、revision_delta",
        "caveat": "全部来自发送端可见的视图差分",
    },
    {
        "question": "队列与规模压力",
        "legal_features": "queue_backlog_targets、subscribers、step_over_horizon、age_steps、"
        "since_last_emission_time、sent_count、pending_event",
        "caveat": "无目标 id 类特征进网络（避免记忆具体场景）",
    },
)

#: Quantities the sender must never read, from the same source as the invariant tests.
EVALUATOR_ONLY: tuple[str, ...] = (
    "注册表真值状态与真 revision（TaskRegistry）",
    "未来的检测事件与未来任务结局",
    "接收端 belief 的实际内容与信息年龄",
    "本包/本 ACK 的真实丢包结果与在途队列",
    "评估器按 target_id 聚合的按时/超时结局",
)

#: The four explanations, with what each one's evidence currently says.
def explanations() -> dict:
    return {
        "允许多种解释并存：以下任何一项都不能因为'不能排除'而被写成原因。": True,
        "undertraining": {
            "claim": "在 400k 环境步的资源上限内仍未收敛到位",
            "supporting": [
                "曲线里 100k→400k 仍在改善：'last_minus_first' 见 runs[*]['curve_delta']",
                "每 run 的 optimizer.step 只有 3.1k–6.1k 次，决策样本 18.8 万–37.7 万条",
                "48 条训练流被重放 8–9 次，不是 400 个不同场景",
            ],
            "contradicting": [
                "六个运行全部显著优于各自未训练初值（CI 下界 > 0），说明梯度信号确实在起作用",
                "seed 间完成率极差在扩训后缩小（第二轮 1.833→0.042 的同型现象）",
            ],
            "missing": [
                "value loss 未持久化：只有 explained variance、KL、clip 与熵可查",
                "未保存每里程碑的贪心动作熵（只有 rollout 期间的 ent/ent_free）",
                "没有更长期限的运行可比较（资源上限是预注册的，本轮不追加）",
            ],
            "status": "不能排除，也不能确证",
        },
        "insufficient_observation": {
            "claim": "发送时机真正需要的状态量不在观测里",
            "supporting": [
                "紧急度与剩余预算都在观测里；缺的是'别的消息/别台机器人的状态'类全局量——但那是设计约束（不变量 9），不是遗漏",
                "learned2 与 send_once 逐包序列在 15/24 流上完全相同 ⇒ 观测足以复现该规则的行为",
            ],
            "contradicting": [
                "同观测下 learned1 达到全表最高均值 5.9167 个/ep（虽未过门槛）",
                "policy_owned_timing 下自由决策占比 100%（无代发掩码），不是'动作被掩码定死'的老问题",
            ],
            "missing": [
                "没有做过消融：把某个特征置零/替换后重训的对照不存在（需新训练授权）",
                "发送端能否合法推断'这次不发的代价'只有间接证据（门禁 3 的单步反事实）",
            ],
            "status": "无直接证据；需要干预实验而非再训练",
        },
        "temporal_credit_assignment": {
            "claim": "一次 HOLD 的后果在若干步之后才出现，单步奖励把它冲淡了",
            "supporting": [
                "本轮实测口径：奖励是每个按时目标记 1 分、记在发生的那一步 ⇒ 与发送决策之间隔着行程与工时",
                "单步反事实里 96.5% 的SEND/HOLD 差被后续策略补偿掉（打平），正是'后果被后续动作吸收'的画像",
                "下一轮协议（多步干预）存在的唯一理由就是把 horizon 拉长到能看见这种补偿",
            ],
            "contradicting": [
                "打平也可能意味着该步本身确实无后果，而不是后果被延迟——单步数据无法区分二者",
                "ACK 反事实中字节轴同样大量打平（363/425），若只是延迟成本，字节应先于完成数分化",
            ],
            "missing": [
                "已有 trace 不能回答'一次 HOLD 后多久被补发'：反事实 transcript 每 seed 只保留最后一个点的运行，且未标强制步",
                "需要跨步干预与补发间隔统计（见协议 §5）",
            ],
            "status": "最可疑、但当前证据最薄；是下一轮的唯一优先问题",
        },
        "strong_rule_at_reachable_effect": {
            "claim": "在这个任务分布与这套信道下，强规则已经接近可达效果，学习没有额外空间可占",
            "supporting": [
                "本轮 200 新流：noack_send_once 以 14 605 B/ep 拿 4.48/5.515/5.81 个/ep，是被反复复现的强锚点",
                "第二轮起所有'省通信'收益都可由共享静默上界或规则改写解释（hold_to_bound、send_once）",
                "本轮规则前沿里没有任何臂在有统计支持的意义上支配锁定规则",
            ],
            "contradicting": [
                "规则的合法性变化本身就带来过收益（协议收益≠学习收益，但说明'规则已够'依赖协议定义）",
                "ACK 规则在隔离有损上拿到 +0.835 个/ep ⇒ 规则集合的上界还会随反馈协议移动",
            ],
            "missing": [
                "'可达效果'需要一个不依赖学习的上界参照（如全知调度 oracle），当前没有做过",
                "正常路径上旁路（视距+gossip）使任何发送时机的边际价值被稀释，未单独量化",
            ],
            "status": "与前三项不互斥；不能作为关闭学习路线的理由",
        },
    }


# --------------------------------------------------------------------------- parsing
def parse_log(path: Path) -> dict[str, Any]:
    updates: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = UPDATE_PATTERN.search(line)
        if match:
            record = {
                key: (float(value) if "." in value else int(value))
                for key, value in match.groupdict().items()
            }
            updates.append(record)
            continue
        validation = VALIDATION_PATTERN.search(line)
        if validation:
            validations.append(json.loads(validation.group(1)))
    return {
        "log": path.name,
        "updates": updates,
        "validations": validations,
        "fields_saved": sorted(updates[0]) if updates else [],
    }


def milestone_rows(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """The rollout line nearest each saved checkpoint milestone (100k/200k/300k/400k)."""
    out = []
    for milestone in (100_000, 200_000, 300_000, 400_000):
        near = [
            row for row in parsed["updates"] if abs(row["steps"] - milestone) <= 4_000
        ]
        if near:
            row = min(near, key=lambda item: abs(item["steps"] - milestone))
            out.append({**row, "milestone": milestone})
    return out


def curve_for_seed(tier: str, seed: int) -> dict[int, dict[str, Any]]:
    """Every saved greedy evaluation point of one run, keyed by its training milestone.

    The curve file stores one arm name per milestone (``..._at0``, ``..._at100000``, ...), so
    a run's curve is gathered across those names rather than filtered to the final one.
    """
    path = EVAL_ROOT / tier / f"rows_{tier}_isolated_1.0_curve.jsonl"
    if not path.exists():
        return {}
    prefix = f"learned_{tier}_seed{seed}_at"
    points: dict[int, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not str(row["arm"]).startswith(prefix):
            continue
        points.setdefault(int(row["at_steps"]), []).append(row)
    return {steps: _summarise_point(rows) for steps, rows in sorted(points.items())}


def _summarise_point(rows: list[dict[str, Any]]) -> dict[str, Any]:
    emissions = sum(r["emissions"] for r in rows)
    return {
        "episodes": len(rows),
        "arm_label": str(rows[0]["arm"]),
        "on_time_per_episode": round(
            sum(r["on_time_targets_per_episode"] for r in rows) / len(rows), 4),
        "airtime_bytes_per_episode": round(
            sum(r["airtime_bytes"] for r in rows) / len(rows), 1),
        "emissions_per_episode": round(emissions / len(rows), 3),
        "policy_emissions_share": round(
            sum(r["emissions_by_policy"] for r in rows) / max(1, emissions), 4),
        "forced_emissions_share": (
            None if all(row.get("forced_share_of_emissions") is None for row in rows)
            else round(
                sum(float(row.get("forced_share_of_emissions") or 0.0) for row in rows) / len(rows), 4)
        ),
        "credit_used_fraction": round(sum(r["credit_used_fraction"] for r in rows) / len(rows), 4),
        "budget_refusals_per_episode": round(sum(r["budget_refusals"] for r in rows) / len(rows), 1),
        "belief_age_steps": round(sum(r["belief_age_steps"] for r in rows) / len(rows), 1),
        "greedy": bool(rows[0].get("greedy")),
    }


def resend_trace_feasibility() -> dict[str, Any]:
    """Can the saved traces say how fast a fixed follow-up policy compensates a forced HOLD?"""
    directory = common.ACK_ROUND / "evaluation" / "25pct_p70" / "counterfactual"
    counts = {label: len(list(directory.glob(f"ct_cf_{label}_s*.jsonl"))) for label in ("free", "send", "hold", "probe")}
    counterfactual = json.loads(
        (common.ACK_ROUND / "COMM_ACK_VALUE_COUNTERFACTUAL_25pct.json").read_text(encoding="utf-8")
    )
    points = counterfactual["decision_points"]
    streams = len({row["seed"] for row in counterfactual["per_stream"]})
    forced_marker = "forced_applied" in json.dumps(counterfactual["per_stream"][0])
    return {
        "decision_points": points,
        "streams": streams,
        "transcripts_per_branch": counts,
        "expected_files_if_one_per_point": points * 3,
        "per_point_transcripts_retained": bool(
            counts["free"] >= points
        ),
        "forced_step_marked_in_transcript": forced_marker,
        "answer": (
            "不能回答。每个 seed 的 transcript 被同 seed 的 10 个点反复覆盖（"
            f"{counts['free']} 份 vs 需要 {points} 份），且转录行不标强制步，"
            "无法把一次补发归因到某个干预点 ⇒ 列为待测，写入下一轮协议的记录要求。"
        ),
        "listed_as_to_measure": True,
    }


def build() -> dict[str, Any]:
    sequences = json.loads(
        (common.PROTOCOL_ROUND / "COMM_PROTOCOL_V2_SEQUENCES.json").read_text(encoding="utf-8")
    )
    runs: dict[str, Any] = {}
    for tier, seed in RUNS:
        name = f"{tier}_seed{seed}"
        log = parse_log(common.PROTOCOL_ROUND / f"train_{tier}_seed{seed}.log")
        summary_path = TRAIN_ROOT / name / f"v2_{tier}_seed{seed}_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        curve = curve_for_seed(tier, seed)
        initial = {0: curve[0]} if 0 in curve else {}
        updates = log["updates"]
        runs[name] = {
            "tier": tier,
            "seed": seed,
            "environment_steps": summary.get("environment_steps"),
            "episodes": summary.get("episodes"),
            "optimizer_steps_cumulative": summary.get("optimizer_steps_cumulative"),
            "decision_samples_recorded": summary.get("decision_samples_recorded"),
            "free_decision_terms_recorded": summary.get("free_decision_terms_recorded"),
            "likelihood_max_abs_gap": summary.get("likelihood_max_abs_gap"),
            "rollout_updates_parsed": len(updates),
            "validation_blocks_parsed": len(log["validations"]),
            "value_loss_saved": "missing" if "value_loss" not in (log["fields_saved"] or []) else True,
            "last_rollout_lines": milestone_rows(log),
            "evaluation_curve_greedy": curve,
            "initial_evaluation_greedy": initial,
            "curve_delta_on_time_per_episode": (
                round(curve[400_000]["on_time_per_episode"] - curve[0]["on_time_per_episode"], 4)
                if 400_000 in curve and 0 in curve
                else None
            ),
            "last_evaluation_saved": summary.get("last_evaluation"),
        }
    return {
        "document": "COMM_LEARNING_DIAGNOSTIC",
        "date": "2026-09-21",
        "branch": "analysis/comm-evidence-review",
        "scope": (
            "修正采样器之后、policy_owned_timing 下训练过的全部 seed = 第五轮 6 个运行"
            "（25 % 与 75 % 两档 × 3 seed，各 400k 环境步）。第四轮的 6 个运行是修正采样器 + "
            "legacy_shared_bound，不在本表口径内；ACK 反馈模型从未训练。"
        ),
        "runs_executed_this_round": {"episodes": 0, "streams_recorded": 0, "training_runs": 0},
        "runs": runs,
        "sequence_comparator": sequences["comparator"],
        "sequence_identity_vs_strong_rule": sequences["sequence_comparison"],
        "sequence_definition": "同一条验证流上逐包比对（target, step, revision）序列；不看完成数也不看字节",
        "observation_legality": list(OBSERVATION_LEGALITY),
        "evaluator_only": list(EVALUATOR_ONLY),
        "ack_resend_question": resend_trace_feasibility(),
        "explanations": explanations(),
        "reading_rules": [
            "环境步、候选决策数、真正自由决策数、optimizer.step 是四个不同的量，不得互为代理",
            "explained variance / KL / clip fraction / 熵只作健康度诊断，任何单一阈值都不构成收敛证明",
            "完成数相同不等于策略相同：逐消息序列一致率必须单独给",
            "'不能排除某解释'不等于'该解释是原因'；本表不允许被引用为训练追加依据",
        ],
    }


def render(data: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append
    add("# 已有训练记录的学习诊断（不跑任何 episode）")
    add("")
    add(
        f"日期：{data['date']}。分支 `{data['branch']}`。{data['scope']} "
        f"本轮实测：episode {data['runs_executed_this_round']['episodes']}、"
        f"新流 {data['runs_executed_this_round']['streams_recorded']}、"
        f"训练 {data['runs_executed_this_round']['training_runs']}。"
    )
    add("")
    add("## 一、六个运行的训练量：四个计数分开列")
    add("")
    add(
        "| 运行 | 预算档 | 环境步 | episode | 记录决策样本 | 自由决策项 | optimizer.step | "
        "value loss | likelihood 偏差 |"
    )
    add("|---|---|---:|---:|---:|---:|---:|---|---:|")
    for name, run in data["runs"].items():
        add(
            f"| `{name}` | {run['tier']} | {run['environment_steps']:,} | {run['episodes']} "
            f"| {run['decision_samples_recorded']:,} | {run['free_decision_terms_recorded']:,} "
            f"| {run['optimizer_steps_cumulative']:,} | "
            f"{'**缺失（未持久化）**' if run['value_loss_saved'] == 'missing' else run['value_loss_saved']} "
            f"| {run['likelihood_max_abs_gap']} |"
        )
    add("")
    add(
        "- 环境步 400k = 400 episode × 1 000 联合步（无提前终止），但真正驱动更新的是 "
        "optimizer.step（3.1k–6.1k 次）与约 19 万–38 万条决策样本 ⇒ 用环境步表征学习量会高估近一个数量级。"
    )
    add(
        "- `policy_owned_timing` 下自由决策项与候选决策项**同数**（传输层不再代发），"
        "这是把 3/10 时机还给策略的直接后果；旧协议下该比例曾被掩码压到 14–34 %。"
    )
    add("")
    add("## 二、同一口径的学习曲线（贪心评估，24 条验证流，隔离无损）")
    add("")
    add(
        "| 运行 | @0 (个/ep) | @100k | @200k | @300k | @400k | Δ(400k−0) | "
        "@400k 字节 B/ep | @400k 额度使用 | @400k 拒付/ep | @400k 政策发射占比 |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, run in data["runs"].items():
        curve = run["evaluation_curve_greedy"]
        at400 = curve.get(400_000, {})
        cells = []
        for milestone in (0, 100_000, 200_000, 300_000, 400_000):
            point = curve.get(milestone)
            cells.append(f"{point['on_time_per_episode']:.4f}" if point else "—")
        delta = run["curve_delta_on_time_per_episode"]
        add(
            f"| `{name}` | " + " | ".join(cells) + f" | {delta:+.4f} "
            f"| {at400.get('airtime_bytes_per_episode', float('nan')):,.0f} "
            f"| {at400.get('credit_used_fraction', float('nan')):.3f} "
            f"| {at400.get('budget_refusals_per_episode', float('nan')):.1f} "
            f"| {at400.get('policy_emissions_share', float('nan')):.3f} |"
        )
    add("")
    add("### 与曲线对齐的优化健康度（rollout 侧，非贪心评估；只作诊断）")
    add("")
    add(
        "| 运行 | 里程碑 | on_time (个/rollout) | 字节 | 额度使用 | send_rate | 自由决策/候选 | "
        "samples | opt.step | KL | clip | ent | ent_free | explained variance |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, run in data["runs"].items():
        for row in run["last_rollout_lines"]:
            add(
                f"| `{name}` | {row['milestone']:,} | {row['on_time']}/{row['on_time_total']} "
                f"| {row['bytes']:,} | {row['use']:.3f} | {row['send_rate']:.4f} "
                f"| {row['free']}/{row['terms']} | {row['samples']:,} | {row['opt']} "
                f"| {row['kl']:.4f} | {row['clip']:.4f} | {row['ent']:.3f} "
                f"| {row['ent_free']:.3f} | {row['ev']:.3f} |"
            )
    add("")
    add(
        "- **缺失项如实登记**：贪心评估的动作熵、value loss、每里程碑的策略概率分布均未保存；"
        "上表熵/KL/clip/ev 来自 rollout（ε 已去除后的 on-policy 采样），与评估策略同权重但不同状态分布。"
        "任何单一阈值都不构成收敛证明。"
    )
    add("")
    add("## 三、模型与强规则是否真的相同（逐消息序列）")
    add("")
    add(
        f"口径：{data['sequence_definition']}。对照 = `{data['sequence_comparator']}`，"
        "24 条验证流，隔离无损，29 300 B。"
    )
    add("")
    add("| 臂 | 比较流数 | 逐包完全相同 | 同批包不同时刻 | 多发 | 少发 |")
    add("|---|---:|---:|---:|---:|---:|")
    for arm, block in data["sequence_identity_vs_strong_rule"].items():
        add(
            f"| `{arm}` | {block['streams_compared']} | {block['streams_identical']} "
            f"| {block['streams_same_packets_different_timing']} | {block['packets_extra']} "
            f"| {block['packets_missing']} |"
        )
    add("")
    add(
        "- `learned_..._seed2_at400000` 在 15/24 流上与 `send_once` 逐包相同、9/24 同批包不同时刻 "
        "⇒ 完成数与字节相同并不等于策略相同；反之亦然。"
    )
    add("")
    add("## 四、观测合法性：发送端能推断什么")
    add("")
    add("| 决策需要的问题 | 观测里合法可用的量 | 边界 |")
    add("|---|---|---|")
    for row in data["observation_legality"]:
        add(f"| {row['question']} | {row['legal_features']} | {row['caveat']} |")
    add("")
    add("**只有评估器/模拟器知道的量（不得进在线 actor）**")
    add("")
    for item in data["evaluator_only"]:
        add(f"- {item}")
    add("")
    add("## 五、ACK 反事实之后是否被补发抵消（已有 trace 能否回答）")
    add("")
    resend = data["ack_resend_question"]
    add(
        f"- 已有受控点 {resend['decision_points']} 个 / {resend['streams']} 条流；"
        f"三个分支的 transcript 各 {resend['transcripts_per_branch']['free']} 份 "
        f"（每点一份需要 {resend['expected_files_if_one_per_point']}）；"
        f"强制步是否标在转录里：{'是' if resend['forced_step_marked_in_transcript'] else '否'}。"
    )
    add(f"- {resend['answer']}")
    add("")
    add("## 六、四种解释的证据表（允许并存）")
    add("")
    for key, block in data["explanations"].items():
        if key.startswith("允许"):
            continue
        add(f"### {key} — {block['claim']}")
        add("")
        add(f"- **支持**：{'；'.join(block['supporting'])}")
        add(f"- **反对**：{'；'.join(block['contradicting'])}")
        add(f"- **仍缺**：{'；'.join(block['missing'])}")
        add(f"- **当前状态**：{block['status']}")
        add("")
    add("## 七、读这张表的规则")
    add("")
    for rule in data["reading_rules"]:
        add(f"- {rule}")
    add("")
    add("## 八、复跑")
    add("")
    add("```bash")
    add("cd runs/comm_evidence_review_20260921")
    add("python learning_diagnostic.py")
    add("```")
    add("")
    add(
        "本文件所有数字来自 `runs/comm_protocol_v2_20260920/`（训练日志、summary、曲线评价行、"
        "序列比较 JSON）与 `runs/comm_ack_value_20260920/`（反事实 JSON 与转录文件计数），"
        "均为只读引用；本轮未生成任何新 episode。"
    )
    add("")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    data = build()
    common.write_json(common.json_path("COMM_LEARNING_DIAGNOSTIC.json"), data)
    text = render(data)
    common.json_path("COMM_LEARNING_DIAGNOSTIC.md").write_text(text, encoding="utf-8")
    print(f"written: COMM_LEARNING_DIAGNOSTIC.md ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
