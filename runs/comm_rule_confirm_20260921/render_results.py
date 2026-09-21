"""Render COMM_RULE_CONFIRMATION_RESULTS.md from the machine-readable results.

The markdown and the JSON are two views of the same run, so the markdown is rendered rather
than typed: a number cannot drift between the two, and a column the protocol asked for cannot
be quietly dropped. Every judgement sentence comes from the judgement blocks the report wrote,
so the prose cannot outrun the pre-registered rules.

    python confirm_matrix.py --mode report
    python render_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import confirm_common as common

SEALED = "800000..800399"


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


def _counts(block: dict[str, Any]) -> str:
    return (
        f"{block.get('seeds_better', 0)} 更好 / "
        f"{block.get('seeds_equal', 0)} 相等 / {block.get('seeds_worse', 0)} 更差"
    )


def _arm_table(payload: dict[str, Any], setting: str) -> list[str]:
    summary = payload["settings"][setting]
    lines = [
        "| 臂 | onT/ep | 上行 B | ACK B | **total B** | ACK 占比 | 额度使用 | 信息年龄 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in common.ARM_LABELS:
        if arm not in summary:
            continue
        entry = summary[arm]
        lines.append(
            f"| `{arm}` | {entry['on_time_per_episode']:.4f} "
            f"| {entry['uplink_bytes']:,.0f} | {entry['ack_bytes']:,.0f} "
            f"| **{entry['total_bytes']:,.0f}** | {entry['ack_share_of_total'] * 100:.1f} % "
            f"| {entry['budget_used_fraction']:.3f} | {entry['belief_age_steps']:.1f} 步 |"
        )
    return lines


def _task_table(payload: dict[str, Any], setting: str) -> list[str]:
    summary = payload["settings"][setting]
    lines = [
        "| 臂 | 条件完成率 | 全部目标完成率 | expired/ep | unresolved/ep | 发现/ep |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in common.ARM_LABELS:
        if arm not in summary:
            continue
        entry = summary[arm]
        lines.append(
            f"| `{arm}` | {entry['completion_rate_conditional']:.4f} "
            f"| {entry['completion_rate_all_goals']:.4f} "
            f"| {entry['expired_per_episode']:.4f} "
            f"| {entry['unresolved_per_episode']:.4f} "
            f"| {entry['discovered_per_episode']:.4f} |"
        )
    return lines


def _latency_table(payload: dict[str, Any], setting: str) -> list[str]:
    summary = payload["settings"][setting]
    keys = (
        ("detection_to_assignment", "发现→首次分配"),
        ("detection_to_work", "发现→首次工作"),
        ("dispatch_to_work", "派遣→首次工作"),
        ("detection_to_completion", "发现→按时完成"),
    )
    lines = ["| 臂 | " + " | ".join(name for _, name in keys) + " |", "|---|" + "---:|" * len(keys)]
    for arm in common.ARM_LABELS:
        if arm not in summary:
            continue
        entry = summary[arm]
        lines.append(
            f"| `{arm}` | " + " | ".join(f"{entry[key]:.2f} 步" for key, _ in keys) + " |"
        )
    return lines


def _comparison_table(payload: dict[str, Any], setting: str) -> list[str]:
    lines = [
        "| 比较 | 完成数差 (个/ep) | 95 % CI | 逐流 | total-byte 差 (B/ep) | 95 % CI | 逐流 |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for key, block in payload["comparisons"][setting].items():
        completion = block["axes"]["on_time"]
        total = block["axes"]["total_bytes"]
        lines.append(
            f"| `{key}` | {_fmt(completion['mean_difference'])} "
            f"| {_ci(completion)} | {_counts(completion)} "
            f"| {_fmt(total['mean_difference'], 1)} | {_ci(total, 1)} | {_counts(total)} |"
        )
    return lines


def _ratio_table(payload: dict[str, Any], setting: str) -> list[str]:
    lines = [
        "| 比较 | ΔonT/ep (分母) | 每任务上行 B | 每任务 ACK B | 每任务 total B | total CI | 未定义 replicate |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for key, block in payload["ratios"][setting].items():
        if not block.get("defined"):
            lines.append(
                f"| `{key}` | {block['denominator_mean_on_time']:.5f} | — | — | **undefined** "
                f"| — | {block.get('undefined_replicates', '—')} |"
            )
            continue
        total_ci = block.get("total_bytes_ratio_ci95")
        lines.append(
            f"| `{key}` | {block['denominator_mean_on_time']:.5f} "
            f"| {block['uplink_bytes_ratio']:,.1f} | {block['ack_bytes_ratio']:,.1f} "
            f"| **{block['total_bytes_ratio']:,.1f}** "
            f"| {f'[{total_ci[0]:,.1f}, {total_ci[1]:,.1f}]' if total_ci else '—'} "
            f"| {block.get('undefined_replicates', 0)} |"
        )
    return lines


def _failure_table(payload: dict[str, Any], setting: str) -> list[str]:
    lines = ["| 比较 | 新增失败目标 | 救回目标 | 明细 |", "|---|---:|---:|---|"]
    for key, block in payload["new_failures"][setting].items():
        detail = ", ".join(
            f"s{case['seed']}/{case['target_id']}({case['outcome']})"
            for case in block["new_failure_cases"][:8]
        )
        if len(block["new_failure_cases"]) > 8:
            detail += f" …共 {len(block['new_failure_cases'])} 个"
        lines.append(
            f"| `{key}` | {block['new_failures']} | {block['recovered']} | {detail or '—'} |"
        )
    return lines


def render(payload: dict[str, Any]) -> str:
    throughput = common.read_json("COMM_RULE_THROUGHPUT.json")
    streams = common.read_json("COMM_RULE_STREAMS.json")
    audit = common.read_json("COMM_RULE_BLOCK_AUDIT.json")
    out: list[str] = []
    add = out.append

    add("# 强规则独立工程确认：五条锁定规则 × 三个设置 × 200 条全新流")
    add("")
    add(
        "日期：2026-09-21。分支 `analysis/comm-rule-independent-confirmation`，起点 "
        "`feat/comm-ack-feedback-gate@fbf96d8`（只引入文档提交 `2e341b1`），协议冻结提交 "
        "`5ddd7f5`，流登记提交 `fbeb0b8`。对应总纲 v1.9 与执行指令 "
        "`COMMUNICATION_S3_CLOSURE_NEXT_ROUND.md`。"
    )
    add("")
    add(
        "机器可读：`COMM_RULE_CONFIRMATION_RESULTS.json`（全部逐流配对统计、比值、判读与 "
        "Pareto 前沿）。本文只报告，不改判据；判据在运行前写定于 "
        "`COMM_RULE_CONFIRMATION_PROTOCOL.md`。"
    )
    add("")
    add(
        f"**本轮不训练任何模型**，不延长旧 checkpoint，不确认单个最好 seed，不改奖励、网络、"
        f"任务分布、deadline 或生产默认配置。学习最终测试块 `{SEALED}` 继续封存：未生成、"
        f"未读取、盘上无 artefact。"
    )
    add("")

    add("## 〇、本轮跑了什么")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(
        f"| 新工程块 | `{streams['block'][0]}..{streams['block'][1]}`（{streams['size']} seed），"
        f"与全项目四源零重叠 |"
    )
    add(
        f"| 流 | {streams['fold']['streams']} 条、{streams['fold']['bytes']:,} B，"
        f"fold `{streams['fold']['sha256_fold'][:16]}…`，检测数合计 "
        f"{sum(int(row['detections']) for row in streams['streams'])} |"
    )
    add(
        f"| 规模 | 200 seed × 3 设置 × 5 臂 = **{payload['episodes']:,} / "
        f"{payload['episodes_expected']:,} episode**，未合并设置样本 |"
    )
    add(
        f"| 吞吐 | 实测 {throughput['seconds_per_episode_mean']:.2f} s/episode、"
        f"{throughput['bytes_per_episode'] / 1e6:.2f} MB/episode（探测 {throughput['episodes']} 集，"
        f"5 个非新块 seed） |"
    )
    add(
        f"| 统计 | 同 seed 配对百分位 bootstrap，B = {payload['statistics']['replications']:,}，"
        f"RNG {payload['statistics']['rng_seed']}，逐流胜/平/负 |"
    )
    add("| 成本口径 | total = 任务上行尝试字节 + ACK 反向尝试字节，两项分列，ACK 丢失也计费 |")
    add("| 训练 | **未执行**（0 个 seed、0 环境步） |")
    add("")
    incomplete = {
        setting: value for setting, value in payload["incomplete_cells"].items() if value
    }
    if incomplete:
        add(f"> 注意：有未满格单元 {incomplete}，本表的均值按实际 episode 数计算。")
    else:
        add("> 全部 15 个 (设置, 臂) 单元均为 200/200 集，无缺格、无重跑剔除。")
    add("")

    add("## 一、三个设置上的任务效果与成本")
    add("")
    for setting in common.SETTING_LABELS:
        channel = common.SETTINGS[setting]
        add(
            f"### {setting}（{'隔离' if channel['information'] == 'isolated' else '正常'}信息路径，"
            f"上行与反向丢包均 {channel['delivery_probability']}）"
        )
        add("")
        add(f"*{channel['role']}*")
        add("")
        out.extend(_arm_table(payload, setting))
        add("")
        out.extend(_task_table(payload, setting))
        add("")
        out.extend(_latency_table(payload, setting))
        add("")
        add("**通信明细（每 episode 均值）**")
        add("")
        summary = payload["settings"][setting]
        add(
            "| 臂 | 发射 | 首发 | 刷新 | 重复发射 | 尝试 | 丢失 | 交付应用 | 接收判重 | 拒付 |"
            " ACK 尝试 | ACK 丢失 | ACK 推进 | ACK 重申 |"
        )
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for arm in common.ARM_LABELS:
            if arm not in summary:
                continue
            entry = summary[arm]
            add(
                f"| `{arm}` | {entry['emissions']:.2f} | {entry['first_emissions']:.2f} "
                f"| {entry['refresh_emissions']:.2f} | {entry['redundant_emissions']:.2f} "
                f"| {entry['transmission_attempts']:.2f} | {entry['packets_lost']:.2f} "
                f"| {entry['deliveries_applied']:.2f} | {entry['delivery_duplicate']:.2f} "
                f"| {entry['budget_refused_sends'] + entry['capacity_refused_sends']:.2f} "
                f"| {entry['ack_attempts']:.2f} | {entry['ack_lost']:.2f} "
                f"| {entry['ack_frames_advanced']:.2f} | {entry['ack_frames_reaffirmed']:.2f} |"
            )
        add("")

    add("## 二、四个预注册比较")
    add("")
    add(
        "每个比较同时给出**完成数差**与**total-byte 差**的均值、95 % CI 与逐流胜/平/负。"
        "任何“降本”只能引用 total 列；上行与 ACK 分列只用于说明成本在哪一侧。"
    )
    add("")
    for setting in common.SETTING_LABELS:
        add(f"### {setting}")
        add("")
        out.extend(_comparison_table(payload, setting))
        add("")
        out.extend(_ratio_table(payload, setting))
        add("")
        out.extend(_failure_table(payload, setting))
        add("")

    add("## 三、预注册判读")
    add("")
    for setting in common.SETTING_LABELS:
        block = payload["judgement"][setting]
        add(f"### {setting}")
        add("")
        if "negative_control" in block:
            control = block["negative_control"]
            add(
                f"**无损负控：{'通过' if control['passed'] else '未通过'}。** "
                f"{control['action']}。"
            )
            add("")
            add("| 臂 | 完成数差 | 95 % CI | 判定 | total-byte 差 |")
            add("|---|---:|---|---|---:|")
            for arm, entry in control["arms"].items():
                add(
                    f"| `{arm}` | {_fmt(entry['completion']['mean_difference'])} "
                    f"| {_ci(entry['completion'])} "
                    f"| {'稳定提高（需暂停检查）' if entry['improves'] else '未稳定提高'} "
                    f"| {_fmt(entry['total_bytes']['mean_difference'], 1)} B |"
                )
            add("")
            add(f"> {control['note']}。")
            add("")
        if "lossy_confirmation" in block:
            confirmation = block["lossy_confirmation"]
            add(
                f"**有损隔离确认：{'复现' if confirmation['reproduced'] else '未复现'}。** "
                f"{confirmation['action']}。"
            )
            add("")
            completion = confirmation["completion"]
            total = confirmation["total_bytes"]
            add(
                f"- 完成数差 **{_fmt(completion['mean_difference'])} 个/ep**，"
                f"CI {_ci(completion)}，逐流 {_counts(completion)}；"
            )
            add(
                f"- total-byte 差 **{_fmt(total['mean_difference'], 1)} B/ep**，"
                f"CI {_ci(total, 1)}，逐流 {_counts(total)}。"
            )
            add("")
        if "normal_path" in block:
            normal = block["normal_path"]
            add(
                f"**正常信息路径：{'可作为可选高可靠模式候选' if normal['proposed_as_optional_mode'] else '不可作为部署候选'}。** "
                f"{normal['action']}。"
            )
            add("")
            completion = normal["completion"]
            total = normal["total_bytes"]
            add(
                f"- 完成数差 **{_fmt(completion['mean_difference'])} 个/ep**，"
                f"CI {_ci(completion)}，逐流 {_counts(completion)}；"
            )
            add(
                f"- total-byte 差 **{_fmt(total['mean_difference'], 1)} B/ep**，"
                f"CI {_ci(total, 1)}，逐流 {_counts(total)}；"
            )
            add(
                f"- 逐目标：新增失败 **{normal['new_failures']}** 个、救回 "
                f"**{normal['recovered']}** 个；结局相同但更贵："
                f"**{'是' if normal['same_outcome_but_dearer'] else '否'}**。"
            )
            add("")
        frontier = block["frontier"]
        add(
            f"**规则前沿：{'不稳定' if frontier['unstable'] else '稳定'}。** {frontier['action']}。"
        )
        add("")
        for arm, entry in frontier["checks"].items():
            completion = entry["completion"]
            add(
                f"- `{arm} − ack_retry_2`：{_fmt(completion['mean_difference'])} 个/ep，"
                f"CI {_ci(completion)}，逐流 {_counts(completion)} ⇒ "
                f"{'超过锁定规则' if entry['beats_the_locked_rule'] else '未超过锁定规则'}；"
            )
        add("")
        pareto = block["pareto"]
        add(
            f"**Pareto 前沿（逐流双轴）：{', '.join(f'`{arm}`' for arm in pareto['frontier']) or '空'}。** "
            f"被支配：{', '.join(f'`{arm}`' for arm in pareto['dominated']) or '无'}。"
        )
        add("")

    add("## 四、随机数与配对强度")
    add("")
    add(f"- **前向丢包**：{payload['randomness']['forward']}。")
    add(f"- **反向 ACK 丢包**：{payload['randomness']['reverse']}。")
    add(
        "- 因此：no-ACK 与 ACK 臂之间是**同分布比较**（同 seed、同机制，不是同一串事件）；"
        "ACK 臂之间在相同 (episode, subscriber, target, ordinal) 键上是**共同随机数**。"
    )
    add("")
    add("## 五、复跑")
    add("")
    add("```bash")
    add("cd runs/comm_rule_confirm_20260921")
    add("python block_audit.py")
    add("python record_streams.py --shards 4 --shard 0")
    add("python record_streams.py --mode merge")
    add("python confirm_preflight.py --threads 1")
    add("python confirm_matrix.py --mode probe")
    add("python confirm_matrix.py --mode matrix --shards 8 --shard 0")
    add("python confirm_matrix.py --mode report")
    add("python render_results.py")
    add("python confirm_manifest.py --threads 1")
    add("```")
    add("")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    path = common.HERE / "COMM_RULE_CONFIRMATION_RESULTS.json"
    if not path.exists():
        raise SystemExit(f"{path} missing; run confirm_matrix.py --mode report first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = render(payload)
    out = common.HERE / "COMM_RULE_CONFIRMATION_RESULTS.md"
    out.write_text(text, encoding="utf-8")
    print(f"written: {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
