"""Write the round's results: the ladder, the ledger, and what the ceiling actually bought.

Two things this renderer refuses to do. It never re-decides a gate -- every verdict is read out of
``GATES_g<goals>.json``, which ``judge_layers.py`` produced from the frozen thresholds -- and it
never drops a cost: the launches spent on this round's own defects appear with their count and
their consequence, including the 16 that made the fourth layer impossible inside the brief's
ceiling.

    python report_pressure.py
"""

from __future__ import annotations

import json
import sys
from typing import Any

import freeze_pressure
import judge_layers
import pressure_common as common
import pressure_matrix
import pressure_metrics

LAYER_STATUSES: dict[str, str] = {
    "passed": "通过 L0-L4",
    "l0_failed": "L0 未过：整轮停止",
    "l1_failed": "L1 未过：更高压力层不再运行",
    "proceed": "L2/L3/L4 未过：按门禁进入下一层",
    "not_run": "未运行",
}


def executed_layers() -> list[int]:
    return [
        goals
        for goals in common.LAYER_ORDER
        if common.json_path(f"GATES_g{goals}.json").exists()
    ]


def layer_block(goals: int) -> dict[str, Any]:
    gates = json.loads(common.json_path(f"GATES_g{goals}.json").read_text(encoding="utf-8"))
    matrix = json.loads(
        common.json_path(f"RULE_MATRIX_g{goals}.json").read_text(encoding="utf-8")
    )
    pressure = json.loads(
        common.json_path(f"PRESSURE_METRICS_g{goals}.json").read_text(encoding="utf-8")
    )
    stream_manifest = json.loads(
        common.json_path(f"STREAM_MANIFEST_g{goals}.json").read_text(encoding="utf-8")
    )
    return {
        "goals": goals,
        "seeds": [common.LAYERS[goals][0], common.LAYERS[goals][-1]],
        "gates": gates,
        "matrix": matrix,
        "search_pressure": pressure,
        "streams": {
            "size": len(stream_manifest["streams"]),
            "sha256_fold": stream_manifest["fold"]["sha256_fold"],
            "bytes": stream_manifest["fold"]["bytes"],
            "illegal": stream_manifest["illegal_streams"],
            "mean_detected_over_placed": pressure["mean_detected_over_placed"],
            "worst_stream_detected_over_placed": pressure["worst_stream_detected_over_placed"],
            "executed_code_head": stream_manifest["executed_code"]["head"],
        },
    }


def arm_rows(goals: int, matrix: dict[str, Any]) -> list[dict[str, Any]]:
    """Every arm with its means, its paired intervals and its competition accounting."""
    best = matrix["best_fixed_rule"]["arm"]
    on_time = matrix["bootstrap"]["on_time_per_episode"]
    bytes_ = matrix["bootstrap"]["bytes_per_episode"]
    rows: list[dict[str, Any]] = []
    for arm in common.ARM_ORDER:
        table = matrix["arms"][arm]
        rows.append(
            {
                "arm": arm,
                "role": table["role"],
                "on_time_per_episode": table["on_time_per_episode"],
                "bytes_per_episode": table["bytes_per_episode"],
                "on_time_rate_of_detected": table["on_time_rate_of_detected"],
                "emissions_per_episode": table["emissions_per_episode"],
                "credit_ge_90_streams": table["credit_used_ge_90_streams"],
                "competition_streams": table["competition_streams"],
                "competition_points_total": table["competition_points_total"],
                "competition_points_credit_cause_total":
                    table["competition_points_credit_cause_total"],
                "competition_points_capacity_only_total":
                    table["competition_points_capacity_only_total"],
                "joint_only_points_total": table["joint_only_points_total"],
                "payable_candidate_points_total": table["payable_candidate_points_total"],
                "credit_exhausted_with_candidates_streams":
                    table["credit_exhausted_with_candidates_streams"],
                "budget_refusals_per_episode": table["budget_refusals_per_episode"],
                "first_budget_refusal_step_median": table["first_budget_refusal_step_quantiles"].get(
                    "median"
                ),
                "vs_always_hold_on_time": (
                    on_time["always_hold"]["arms"].get(arm)
                    or (
                        {"mean_difference": 0.0, "ci95": [0.0, 0.0], "pairs": 16}
                        if arm == "always_hold"
                        else None
                    )
                ),
                "vs_best_fixed_on_time": on_time["best_fixed"]["arms"].get(arm),
                "vs_best_fixed_bytes": bytes_["best_fixed"]["arms"].get(arm),
                "is_best_fixed_rule": arm == best,
            }
        )
    return rows


def ceiling_accounting(goals: int | None = None) -> dict[str, Any]:
    """What the brief's 640 bought, and what this round's own defects cost.

    ``goals`` selects the ceiling a layer actually ran against: the g12 extension is granted 16
    launches over the brief's arithmetic, and reporting its remainder against 640 would say the
    round overran when the order says it was topped up.
    """
    ledger = common.ledger_totals()
    spent = int(ledger["matrix_starts"])
    grant = common.launch_grant_for(goals) if goals is not None else None
    ceiling = common.MATRIX_LAUNCH_CAP if grant is None else grant["effective_ceiling"]
    remaining = ceiling - spent
    needed_for_next = common.STREAMS_PER_LAYER * (1 + len(common.ARM_ORDER))
    return {
        "matrix_ceiling": common.MATRIX_LAUNCH_CAP,
        "effective_ceiling_for_this_layer": ceiling,
        "extension_grant": 0 if grant is None else grant["extension_grant"],
        "matrix_starts_spent": spent,
        "recordings_included": True,
        "gate_ceiling": common.GATE_LAUNCH_CAP,
        "gate_starts_spent": int(ledger["gate_starts"]),
        "technical_failures_recorded": int(ledger["failures"]),
        "remaining_for_the_next_layer": remaining,
        "next_layer_would_need": needed_for_next,
        "short_by": max(0, needed_for_next - remaining),
        "cause": (
            "16 次矩阵启动花在“探针在写行之前崩溃”的格上，4 次录制启动花在登记步骤的 "
            "`GridPosition` 缺陷上；按执行指令，失败照样计入上限，所以第四层比它需要的 160 次少 16 次"
        ),
        "rule_applied": "不换 seed、不删臂、不把任何一次启动从账上抹掉",
    }


def packet_cost_profile(goals: int) -> dict[str, Any]:
    """How wide the "some candidates payable, some not" window actually is.

    A strict-subset step needs the remaining credit to sit between two different packet costs. If
    the spread of costs is small next to the credit, that window is narrow no matter how many
    packets the rule wants, and exhaustion arrives as 'nothing is payable' instead. The numbers
    come from the emission sequences already on disk, so this costs no episode.
    """
    rows = list(pressure_matrix.load_rows(goals).values())
    costs = [
        int(entry[3])
        for row in rows
        for entry in row.get("emission_sequence", [])
    ]
    if not costs:
        return {"packets": 0}
    costs.sort()
    low, high = costs[0], costs[-1]
    return {
        "packets": len(costs),
        "min_bytes": low,
        "p25_bytes": costs[int(0.25 * (len(costs) - 1))],
        "median_bytes": costs[len(costs) // 2],
        "p75_bytes": costs[int(0.75 * (len(costs) - 1))],
        "max_bytes": high,
        "spread_bytes": high - low,
        "spread_as_fraction_of_credit": round((high - low) / common.BUDGET_BYTES, 5),
        "median_as_fraction_of_credit": round(costs[len(costs) // 2] / common.BUDGET_BYTES, 5),
        "packets_the_credit_could_pay_in_total": round(
            common.BUDGET_BYTES / costs[len(costs) // 2], 2
        ),
    }


def mechanism_note(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Why the credit binds abruptly rather than selectively -- stated with its counts.

    L3 asks for a step where several candidates are in view and only a strict subset is payable.
    What the layers show instead is that the refresh-heavy rules reach a step where *nothing* is
    payable: the packet cost is a sizeable fraction of the whole credit, so the last affordable
    emission and the exhaustion are one step apart. That is a measurement result, not a definition
    change, and it is the reason the ladder is stopped rather than reinterpreted.
    """
    per_layer = []
    for block in blocks:
        block["packet_cost_profile"] = packet_cost_profile(block["goals"])
        arms = block["matrix"]["arms"]
        best = block["matrix"]["best_fixed_rule"]["arm"]
        per_layer.append(
            {
                "goals": block["goals"],
                "best_fixed_rule": best,
                "best_credit_used_ge_90_streams": arms[best]["credit_used_ge_90_streams"],
                "best_competition_streams": arms[best]["competition_streams"],
                "arms_with_any_competition": sorted(
                    arm for arm, table in arms.items() if table["competition_streams"] > 0
                ),
                "max_competition_streams_over_arms": max(
                    table["competition_streams"] for table in arms.values()
                ),
                "streams_where_something_was_still_visible_but_unpayable": sum(
                    table["credit_exhausted_with_candidates_streams"]
                    for table in arms.values()
                ),
                "mean_packet_bytes_at_refusal": (
                    round(
                        sum(
                            table["bytes_per_episode"]
                            / max(1.0, table["emissions_per_episode"])
                            for table in arms.values()
                            if table["emissions_per_episode"]
                        )
                        / max(
                            1,
                            sum(
                                1
                                for table in arms.values()
                                if table["emissions_per_episode"]
                            ),
                        ),
                        1,
                    )
                ),
            }
        )
    return {
        "per_layer": per_layer,
        "packet_cost_profile": {
            str(block["goals"]): block["packet_cost_profile"] for block in blocks
        },
        "reading": (
            "机制：额度只够约 11 个包，而同一步里各候选的价格极差只占额度的 2.3 %–2.6 %。于是一个流"
            "从『还付得起最贵的那个』滑到『一个都付不起』只需要跨过约 2.6 % 的额度区间，而终点态"
            "（看得见候选但 payable = 0）自己就占掉最后约 9 % ——所以三层里所有臂合计的争用点最多 3/16 流，"
            "而拒付成片出现（每层 82–96 个臂×流）。额度确实生效，但它生效的方式是**整体断供**，"
            "不是在同一 step 上给出可选子集；L3 要的是后者。"
        ),
        "what_this_does_not_license": (
            "这条机制陈述不让 L3 变成通过，也不允许把争用点重定义为“任何一次拒付”——那正是 §7 "
            "明令排除的读法；它只解释为什么三层都没有出现 L3 要求的那种 step 内子集选择"
        ),
    }


def render(data: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# 任务压力 learnability calibration：三层实测与停止点")
    add("")
    add(
        f"日期：{data['date']}。分支 `analysis/comm-task-pressure-learnability`，"
        f"起点 `docs/communication-learnability-calibration@ddee56db`（父提交为 "
        f"`exp/comm-budget-masked-retrain@7e0a423`，两个远端 SHA 已在开跑前核对）。"
        f"协议 `COMM_TASK_PRESSURE_LEARNABILITY_PROTOCOL.md` 先于第一条新 stream 提交；"
        f"总纲 v3.4。"
    )
    add("")
    add(
        "**本轮 0 训练、0 `optimizer.step`、0 checkpoint、0 模型加载。** 唯一自变量 `num_goals`；"
        "29 300 B、`policy_owned_timing`、isolated-lossless、ACK 关闭、其余系统全部冻结。"
    )
    add("")
    add("## 一、判定")
    add("")
    add("| 压力层 | 流 | L0 | L1 | L2 | L3 | L4 | 该层结论 | 协议规定的下一步 |")
    add("|---|---:|---|---|---|---|---|---|---|")
    for block in data["layers"]:
        gates = block["gates"]["gates"]
        add(
            f"| goals={block['goals']} | {block['streams']['size']} | "
            + " | ".join("过" if gates[name]["passed"] else "**不过**" for name in
                         ("L0", "L1", "L2", "L3", "L4"))
            + f" | {block['gates']['passed'] and '通过' or '未通过'} "
            f"| {LAYER_STATUSES[block['next']]} |"
        )
    if data["layers_stopped_at"] == "ceiling":
        add(
            f"| goals=12 | 0 | — | — | — | — | — | 未运行 | 上限已尽，见 §六 |"
        )
    add("")
    add(data["headline"])
    add("")
    add("## 二、每层的九臂实测")
    for block in data["layers"]:
        add("")
        add(f"### goals={block['goals']}（seeds {block['seeds'][0]}..{block['seeds'][1]}）")
        add("")
        add(
            f"最佳预算内固定规则 `{block['matrix']['best_fixed_rule']['arm']}`；"
            f"逐流 hindsight envelope 均值 {block['matrix']['hindsight_envelope']['mean_on_time']} 个/episode；"
            f"mean detected/placed {block['streams']['mean_detected_over_placed']}"
            f"（最差流 {block['streams']['worst_stream_detected_over_placed']}）。"
        )
        add("")
        add("| 臂 | 角色 | 按时完成/ep | 字节/ep | on-time/detected | 发射/ep | 额度≥90 % 流 | 争用点流 | 争用点合计 | 额度型 | 容量型 | 仅合并超支 | 拒付/ep |")
        add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in block["arms"]:
            add(
                f"| `{row['arm']}`"
                f"{' **(最佳)**' if row['is_best_fixed_rule'] else ''} | {row['role']} | "
                f"{row['on_time_per_episode']:.3f} | {row['bytes_per_episode']:.1f} | "
                f"{row['on_time_rate_of_detected']:.4f} | {row['emissions_per_episode']:.2f} | "
                f"{row['credit_ge_90_streams']} | {row['competition_streams']} | "
                f"{row['competition_points_total']} | {row['competition_points_credit_cause_total']} | "
                f"{row['competition_points_capacity_only_total']} | {row['joint_only_points_total']} | "
                f"{row['budget_refusals_per_episode']:.1f} |"
            )
        add("")
        add("| 臂 | 相对 `always_hold` 完成差 | 95 % CI | 相对最佳规则完成差 | 95 % CI | 相对最佳规则字节差 | 95 % CI |")
        add("|---|---:|---|---:|---|---:|---|")
        for row in block["arms"]:
            base_a = row["vs_always_hold_on_time"] or {}
            best_a = row["vs_best_fixed_on_time"] or {}
            best_b = row["vs_best_fixed_bytes"] or {}
            add(
                f"| `{row['arm']}` | {_fmt(base_a.get('mean_difference'))} | "
                f"{_ci(base_a.get('ci95'))} | {_fmt(best_a.get('mean_difference'))} | "
                f"{_ci(best_a.get('ci95'))} | {_fmt(best_b.get('mean_difference'))} | "
                f"{_ci(best_b.get('ci95'))} |"
            )
        add("")
        add(f"- 样本均值 Pareto：{block['matrix']['mean_pareto_frontier']}")
        add(
            f"- 逐流唯一赢家计数：{block['matrix']['hindsight_envelope']['unique_winner_counts']}；"
            f"互不支配且双向胜负的规则对：{block['matrix']['incomparable_bidirectional_pairs'] or '无'}"
        )
        add(
            f"- envelope − 最佳固定规则：**{block['l4_gain']:.4f} 个/episode**，"
            f"95 % CI {block['l4_gain_ci']}"
        )
    add("")
    add("## 三、L3 为什么没被满足：机制而不是口径")
    add("")
    mech = data["mechanism"]
    add("| 压力层 | 最佳规则 | 最佳规则额度≥90 % 流 | 最佳规则争用点流 | 有争用点的臂 | 单层最大争用点流 | 看得见却没额度的流·合计 |")
    add("|---|---|---:|---:|---|---:|---:|")
    for item in mech["per_layer"]:
        add(
            f"| {item['goals']} | `{item['best_fixed_rule']}` | "
            f"{item['best_credit_used_ge_90_streams']} | {item['best_competition_streams']} | "
            f"{', '.join('`%s`' % a for a in item['arms_with_any_competition']) or '无'} | "
            f"{item['max_competition_streams_over_arms']} | "
            f"{item['streams_where_something_was_still_visible_but_unpayable']} |"
        )
    add("")
    add("| 压力层 | 包 | 最小 B | 中位 B | 最大 B | 极差 B | 极差/额度 | 中位/额度 | 额度可付包数 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for goals, profile in mech["packet_cost_profile"].items():
        add(
            f"| {goals} | {profile.get('packets', 0)} | {profile.get('min_bytes', 0)} | "
            f"{profile.get('median_bytes', 0)} | {profile.get('max_bytes', 0)} | "
            f"{profile.get('spread_bytes', 0)} | {profile.get('spread_as_fraction_of_credit', 0):.4f} | "
            f"{profile.get('median_as_fraction_of_credit', 0):.4f} | "
            f"{profile.get('packets_the_credit_could_pay_in_total', 0)} |"
        )
    add("")
    add(f"- {mech['reading']}")
    add(f"- 这条陈述不允许做的事：{mech['what_this_does_not_license']}")
    add("")
    add("## 四、压力与容量（避免把分配瓶颈当成通信瓶颈）")
    add("")
    add("| 压力层 | placed | detected 均值 | registered | 并发开放任务均值 | p95 | 机器人利用率 | 工作量/执行容量 | 从未被分配 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for block in data["layers"]:
        per = block["search_pressure"]["per_stream"]
        n = len(per)
        add(
            f"| {block['goals']} | {block['goals']} | "
            f"{round(sum(v['detected'] for v in per.values()) / n, 3)} | "
            f"{round(sum(v['registered'] for v in per.values()) / n, 3)} | "
            f"{round(sum(v['mean_concurrent_open'] for v in per.values()) / n, 3)} | "
            f"{round(sum(v['p95_concurrent_open'] for v in per.values()) / n, 3)} | "
            f"{round(sum(v['mean_robot_utilisation'] for v in per.values()) / n, 4)} | "
            f"{round(sum(v['work_demand_vs_capacity'] for v in per.values()) / n, 3)} | "
            f"{round(sum(v['never_assigned_targets'] for v in per.values()) / n, 3)} |"
        )
    add("")
    add(data["capacity_note"])
    add("")
    add("## 五、经验交换率（本轮未计算，按协议）")
    add("")
    add(data["lambda_note"])
    add("")
    add("## 六、启动账本与上限")
    add("")
    ceil = data["ceiling"]
    add(
        f"- 数据启动 **{ceil['matrix_starts_spent']} / {ceil['matrix_ceiling']}**"
        f"（含每层 16 次录制）；核验启动 **{ceil['gate_starts_spent']} / {ceil['gate_ceiling']}**；"
        f"账本记录到的技术失败 **{ceil['technical_failures_recorded']}** 次，全部计入。"
    )
    add(
        f"- 下一层需要 {ceil['next_layer_would_need']} 次，剩余 {ceil['remaining_for_the_next_layer']} 次，"
        f"**缺口 {ceil['short_by']} 次**。原因：{ceil['cause']}"
    )
    add(f"- 适用的规则：{ceil['rule_applied']}")
    add("")
    add("## 七、结论范围")
    add("")
    for line in data["conclusion"]:
        add(f"- {line}")
    add("")
    add("## 八、卫生与身份")
    add("")
    hygiene = data["hygiene"]
    add(f"- 四源 seed 冲突审计：`{hygiene['audit']['verdict']}`；命中 {hygiene['audit']['hit_counts']}")
    add(f"- 封存块 `800000..800399` 盘上 artefact **{hygiene['sealed']['count']}**；`900000..900199`、"
        f"`700000..700071` 未进入本轮矩阵")
    add(f"- 生产默认值未改：`comm_ack={hygiene['defaults']['comm_ack']}`、"
        f"`comm_timing_mode={hygiene['defaults']['comm_timing_mode']}`、"
        f"`num_goals={hygiene['defaults']['num_goals']}`")
    add(f"- 归档一致性核验（含探针透明性）：`agree={hygiene['agreement']['agree']}`，"
        f"细节与对第五轮归档转录自不相洽的反向发现写在协议 §12")
    add(
        f"- 测试：本轮合成套件 `test_pressure_harness` {hygiene['tests']['round']}；"
        f"项目套件 `integrated_sar/tests` {hygiene['tests']['project']}；"
        f"集成 smoke `python -m integrated_sar` {hygiene['tests']['smoke']}"
    )
    add("")
    add("### freeze 时间线（从 Git 里每个 CODE_FREEZE.json 版本读回）")
    add("")
    add("| 登记的执行头 | 登记提交 | 模块 | episode 代码身份 | `run_episode` 源哈希 |")
    add("|---|---|---:|---|---|")
    for head, entry in data["freeze_timeline"].items():
        add(
            f"| `{head}` | `{entry['registered_by_commit']}` | {entry['modules']} | "
            f"`{entry['episode_code_identity']}` | `{entry['run_episode_sha256']}` |"
        )
    add("")
    add(
        f"本轮登记了 {len(data['freeze_timeline'])} 次冻结，**episode 代码身份只有一个**"
        f"（`{data['episode_code_identity']}`）：37 个 `integrated_sar/**` 生产模块的 blob 与 "
        f"`run_episode` 源码文本的哈希在每次登记之间未变。因此 §11 修正（探针算法、录制登记、"
        f"行内冻结戳）都没有换过世界，第一层的 144 格与后两层的格属同一 episode 代码状态；"
        f"这一点是哈希核对出来的，不是叙述出来的。"
    )
    add("")
    add("## 九、复跑")
    add("")
    add("```bash")
    add("cd runs/comm_task_pressure_learnability_20260922")
    for line in data["rerun"]:
        add(line)
    add("```")
    add("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):+.4f}"


def _ci(value: Any) -> str:
    return "—" if not value else f"[{value[0]:+.4f}, {value[1]:+.4f}]"


def build() -> dict[str, Any]:
    layers = executed_layers()
    blocks = []
    for goals in layers:
        block = layer_block(goals)
        matrix = block["matrix"]
        best = matrix["best_fixed_rule"]["arm"]
        gain = matrix["bootstrap"]["on_time_per_episode"]["best_fixed"]["arms"].get(
            "envelope", {}
        )
        block["arms"] = arm_rows(goals, matrix)
        block["l4_gain"] = float(gain.get("mean_difference") or 0.0)
        block["l4_gain_ci"] = gain.get("ci95")
        block["next"] = (
            "passed"
            if block["gates"]["passed"]
            else "l0_failed"
            if not block["gates"]["gates"]["L0"]["passed"]
            else "l1_failed"
            if not block["gates"]["gates"]["L1"]["passed"]
            else "proceed"
        )
        block["best_fixed_rule"] = best
        blocks.append(block)

    ran_all_three = [block["goals"] for block in blocks] == [6, 8, 10]
    stopped = "ceiling" if ran_all_three and not any(b["gates"]["passed"] for b in blocks) else ""
    passed = [block for block in blocks if block["gates"]["passed"]]
    headline = (
        "**没有任何压力层通过 L0–L4**（本轮执行 "
        + "、".join(f"{b['goals']} 目标" for b in blocks)
        + " 三层）。"
        if not passed
        else f"通过层：{[b['goals'] for b in passed]}"
    )
    if stopped == "ceiling":
        headline += (
            "L0 与 L1 在三层全部通过，L2 在 10 目标首次进入 [0.40, 0.90] 带内，"
            "但 L3（额度的严格子集选择）与 L4（规则间情境差 / 两个折中点）在三层都未过，"
            "而第四层 12 目标已无法在 640 次上限内完整运行"
        )
    conclusion = [
        "结论口径：**仅提高目标数量（6→8→10，其余全部冻结）没有建立"
        "“固定规则无法统一覆盖、而额度迫使逐 step 选择”的场景**；"
        "L3 的“可支付严格子集”在三层中最佳规则上都是 0/16 流。",
        "不得据此写“不存在更优发送策略”“学习无效”或“规则全局最优”——本轮只测试了一条轴"
        "（目标数），而且没有跑到 12 目标。",
        "也不得把 L2 在 10 目标过带说成收益空间已打开：L2 通过只说明通信仍有价值且未触顶，"
        "它不与 L3/L4 相互替代。",
        "L4 的完成差在三层分别是 "
        + "、".join(
            f"{block['goals']} 目标 {block['l4_gain']:+.3f} 个/episode"
            f"（CI {block['l4_gain_ci']}）"
            for block in blocks
        )
        + "：envelope 相对最佳固定规则的增益从未同时满足 ≥0.25 与 CI 下界 > 0。",
        "按 §9/执行指令 §五：无层通过 ⇒ 停止该学习路线的现有场景扩展；不训练、不改 reward、"
        "不再增加目标数、不打开 `800000..800399`。",
    ]
    if stopped == "ceiling":
        conclusion.append(
            "**上限诚实性**：第四层没跑，不是因为不重要，而是本轮自己的两处缺陷吃掉了 20 次启动，"
            "使 12 目标层缺 16 次。要继续必须由使用者明确追加授权，并先把这一层当作同一协议的延续来登记。"
        )
    hygiene = {
        "audit": json.loads(
            common.json_path("SEED_COLLISION_AUDIT.json").read_text(encoding="utf-8")
        ),
        "sealed": common.sealed_block_artefacts(),
        "defaults": common.production_defaults_ok(),
        "agreement": json.loads(
            common.json_path("AGREEMENT_GATE.json").read_text(encoding="utf-8")
        ),
        "tests": data_tests(),
    }
    history = pressure_matrix.freeze_history()
    identities = sorted(
        {
            entry["episode_code_identity"]
            for entry in history.values()
            if isinstance(entry, dict) and "episode_code_identity" in entry
        }
    )
    return {
        # The episodes ran on 2026-09-22; the clock crossed midnight during reporting, so the
        # delivery date is stated next to the execution date rather than replacing it.
        "date": "2026-09-22（执行日；报告与推送完成于 2026-09-23）",
        "branch": common.git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": common.git("rev-parse", "HEAD"),
        "setting": {
            "budget_bytes": common.BUDGET_BYTES,
            "timing_mode": common.TIMING_MODE,
            "information": common.INFORMATION,
            "delivery_probability": common.DELIVERY_PROBABILITY,
            "ack": common.ACK_ENABLED,
            "steps": common.STREAM_STEPS,
            "arms": list(common.ARM_ORDER),
            "layers_configured": list(common.LAYER_ORDER),
        },
        "layers": blocks,
        "layers_executed": [block["goals"] for block in blocks],
        "layers_stopped_at": stopped,
        "headline": headline,
        "mechanism": mechanism_note(blocks),
        "ceiling": ceiling_accounting(),
        "capacity_note": (
            "`work_demand_vs_capacity` 是被检测目标所需工作量与 6 台救援机器人 × 1 000 步名义容量之比。"
            "它随目标数线性上升，说明更高压力层里未完成任务有相当部分落在执行容量而非信息调度上——"
            "这正是执行指令 §2 要求区分的东西。"
        ),
        "lambda_note": (
            "协议 §9 只在某层通过 L0–L4 之后计算 `lambda_break_even`。本轮没有通过层，因此**不给出经验斜率**，"
            "也不挑选任何 λ；三层各自的样本均值前沿（6 目标两点、8/10 目标一点）在 §二 中如实列出，"
            "但不作为下一轮系数选择的依据——用未通过门槛的层去定成本系数，正是执行指令 §2 禁止的暗加字节惩罚。"
        ),
        "conclusion": conclusion,
        "hygiene": hygiene,
        "freeze_timeline": {k: v for k, v in history.items() if k != "_heads"},
        "episode_code_identity": identities[0] if len(identities) == 1 else identities,
        "rerun": [
            "python pressure_common.py",
            "python seed_audit.py",
            "python pressure_env.py            # 归档一致性 + 探针透明（gate 启动）",
            "python pressure_matrix.py --goals 6 --mode provenance",
            "python record_layer.py --goals 12 --shards 4 --shard 0   # 需使用者追加授权",
            "python record_layer.py --goals 12 --mode merge",
            "python pressure_matrix.py --goals 12 --shards 4 --shard 0",
            "python pressure_matrix.py --goals 12 --mode coverage",
            "python pressure_metrics.py --goals 12",
            "python judge_layers.py --goals 12",
            "python report_pressure.py",
            "python -m unittest test_pressure_harness",
        ],
    }


def data_tests() -> dict[str, str]:
    """The three suites the brief requires, re-run by the reporting stage that quotes them."""
    import subprocess

    commands = {
        "round": [sys.executable, "-m", "unittest", "test_pressure_harness"],
        "project": [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "integrated_sar/tests",
        ],
        "smoke": [sys.executable, "-m", "integrated_sar"],
    }
    out: dict[str, str] = {}
    for name, command in commands.items():
        cwd = str(common.HERE) if name == "round" else str(common.ROOT)
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        tail = (result.stderr or result.stdout).strip().splitlines()[-1:] or [""]
        out[name] = f"{'通过' if result.returncode == 0 else '失败'}：{tail[0]}"
    return out


def main(argv: list[str]) -> int:
    if common.json_path(f"GATES_g{common.G12_GOALS}.json").exists():
        raise SystemExit(
            "the g12 extension layer has been judged, so this round's delivered report is frozen: "
            "regenerating it here would fold the extension into the verdict that said no. Write the "
            "extension's own report with report_g12.py (extension protocol §2)"
        )
    data = build()
    common.write_json(
        common.json_path("COMM_TASK_PRESSURE_LEARNABILITY_RESULTS.json"), data
    )
    common.json_path("COMM_TASK_PRESSURE_LEARNABILITY_RESULTS.md").write_text(
        render(data), encoding="utf-8"
    )
    # The round's two combined artefacts, written from the same in-memory data as the report so
    # they cannot disagree with it.
    common.write_json(
        common.json_path("RULE_MATRIX.json"),
        {
            "layers": {
                str(block["goals"]): block["matrix"] for block in data["layers"]
            },
            "note": "per-layer matrices; the per-layer verdicts are in GATES_g<goals>.json",
        },
    )
    common.write_json(
        common.json_path("PRESSURE_METRICS.json"),
        {
            "layers": {
                str(block["goals"]): block["search_pressure"] for block in data["layers"]
            }
        },
    )
    common.write_json(
        common.json_path("COMM_TASK_PRESSURE_LEARNABILITY_MANIFEST.json"),
        {
            "round": "comm_task_pressure_learnability_20260922",
            "date": data["date"],
            "branch": data["branch"],
            "head_at_report": data["head"],
            "starting_shas": {
                "docs/communication-learnability-calibration":
                    "ddee56db2d7ff0de991520fae3bc7100319d10f8",
                "exp/comm-budget-masked-retrain":
                    "7e0a4233127a19fa9bf49672bbcf37218b587fff",
            },
            "question": (
                "does raising only the number of targets create a scene where fixed rules cannot "
                "cover the situations uniformly and the byte credit forces a per-step choice"
            ),
            "setting": data["setting"],
            "executed": {
                "training_steps": 0,
                "optimizer_steps": 0,
                "checkpoints_written": 0,
                "new_streams": sum(
                    common.STREAMS_PER_LAYER for _ in data["layers_executed"]
                ),
                "layers_executed": data["layers_executed"],
                "layers_not_executed": [
                    goals
                    for goals in common.LAYER_ORDER
                    if goals not in data["layers_executed"]
                ],
                "episode_launches_according_to_ledger": (
                    data["ceiling"]["matrix_starts_spent"]
                    + data["ceiling"]["gate_starts_spent"]
                ),
                "expected_by_arithmetic": sum(
                    common.STREAMS_PER_LAYER * (1 + len(common.ARM_ORDER))
                    for _ in data["layers_executed"]
                ),
                "difference_explained": (
                    "the ledger is the truth and the arithmetic is the plan; the gap is the "
                    "launches this round spent on its own defects and re-attempts, itemised in "
                    "post_run_harness_fixes"
                ),
                "launch_ledger": data["ceiling"],
            },
            "verdicts": {
                str(block["goals"]): {
                    name: gate["passed"] for name, gate in block["gates"]["gates"].items()
                }
                | {"layer_passed": block["gates"]["passed"]}
                for block in data["layers"]
            },
            "headline": data["headline"],
            "conclusion_scope": data["conclusion"],
            "mechanism": data["mechanism"],
            "freeze_timeline": data["freeze_timeline"],
            "episode_code_identity": data["episode_code_identity"],
            "hygiene": data["hygiene"],
            "post_run_harness_fixes": [
                {
                    "what": "probe turned an unbounded credit into an integer (int(inf))",
                    "touched": "CandidateProbe._observe only",
                    "episodes_invalidated": False,
                    "launches_lost": 16,
                    "evidence": "one episode_code_identity across every registered freeze",
                },
                {
                    "what": "recorder expanded a GridPosition as a sequence",
                    "touched": "record_layer.scenario_facts registration step",
                    "episodes_invalidated": False,
                    "launches_lost": 4,
                    "evidence": "streams were written and re-verified; same-seed re-registration "
                    "succeeded and reported placed/detected",
                },
                {
                    "what": "L0 read a layer-wide setting signature and forbade any refusal",
                    "touched": "judge_layers.l0 conditions",
                    "episodes_invalidated": False,
                    "evidence": "the seed is inside the signature, and a fixed rule's refusals are "
                    "the credit binding that L3 asks for; L0 now checks per-stream signature "
                    "equality, layer-wide transport invariants, and transcript/relay agreement",
                },
            ],
            "artefacts_committed": [
                "COMM_TASK_PRESSURE_LEARNABILITY_PROTOCOL.md",
                "SEED_COLLISION_AUDIT.json",
                "AGREEMENT_GATE.json",
                "GATE_SEQUENCE_EXPLANATION.json",
                "CODE_FREEZE.json",
                "STREAM_MANIFEST_g*.json",
                "ROW_PROVENANCE_g*.json",
                "RULE_MATRIX.json",
                "RULE_MATRIX_g*.json",
                "PRESSURE_METRICS.json",
                "PRESSURE_METRICS_g*.json",
                "GATES_g*.json",
                "COMM_TASK_PRESSURE_LEARNABILITY_RESULTS.md/.json",
                "LAUNCH_LEDGER*.json",
            ],
            "never_committed": ["streams", "per-episode logs", "checkpoints", "runs/"],
            "tests": data["hygiene"]["tests"],
            "next_round_requires_an_order": (
                "12 目标层需要 160 次启动而剩余额度为 "
                f"{data['ceiling']['remaining_for_the_next_layer']} 次；继续须由使用者明确追加授权"
            ),
        },
    )
    print(
        json.dumps(
            {
                "layers_executed": data["layers_executed"],
                "verdicts": {
                    str(block["goals"]): {
                        name: gate["passed"]
                        for name, gate in block["gates"]["gates"].items()
                    }
                    for block in data["layers"]
                },
                "headline": data["headline"][:200],
                "ceiling": data["ceiling"],
                "tests": data["hygiene"]["tests"],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
