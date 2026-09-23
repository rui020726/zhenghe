"""Report the g12 extension layer on its own, without touching the delivered round's verdicts.

The three delivered layers said no. This renderer exists so the fourth layer can be finished
without that "no" being rewritten by association: every file it writes carries the
``COMM_TASK_PRESSURE_G12_EXTENSION_`` or ``G12_`` prefix, the delivered artefacts are read but never
written, and the branch decision follows the pre-registered table in the extension protocol §8
rather than a sentence composed here.

    python report_g12.py
"""

from __future__ import annotations

import json
import sys
from typing import Any

import judge_layers
import pressure_common as common
import pressure_matrix
import pressure_metrics
import report_pressure

#: Every name this script may write. Anything else is the delivered round's evidence.
OWN_NAMES = (
    "COMM_TASK_PRESSURE_G12_EXTENSION_PROTOCOL.md",
    "COMM_TASK_PRESSURE_G12_EXTENSION_RESULTS.md",
    "COMM_TASK_PRESSURE_G12_EXTENSION_RESULTS.json",
    "COMM_TASK_PRESSURE_G12_EXTENSION_VERDICT.json",
    "COMM_TASK_PRESSURE_G12_EXTENSION_MANIFEST.json",
    "G12_FREEZE.json",
    "G12_START_LEDGER.json",
    "G12_SEED_COLLISION_AUDIT.json",
)

DELIVERED_LAYERS = (6, 8, 10)


def _remote_sha(branch: str, attempts: int = 3) -> str:
    """Ask the remote itself. If it stays unreachable, say so instead of quoting a local ref as remote.

    Retried because this round's own network history is intermittent: a single failed lookup is not
    evidence about the remote, and a single success is not either -- the caller sees which one it got.
    """
    import subprocess
    import time

    last = ""
    for attempt in range(attempts):
        try:
            result = subprocess.run(
                ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
                cwd=str(common.ROOT),
                capture_output=True,
                text=True,
                timeout=180,
                check=True,
            )
        except Exception as error:  # noqa: BLE001 - an unreachable remote is reported, never glossed
            last = f"{type(error).__name__}: {str(error)[:120]}"
            time.sleep(5)
            continue
        fields = result.stdout.split()
        return fields[0] if fields else "<远端没有该分支>"
    return f"<未能在线核对（{attempts} 次尝试）：{last}>"


def require_writable(name: str) -> None:
    if not (name in OWN_NAMES or name.startswith("G12_") or name.endswith("_g12.json")):
        raise SystemExit(
            f"report_g12 may not write {name!r}: the extension publishes its own files and leaves "
            "the delivered round's verdicts alone (extension protocol §2)"
        )


def write(name: str, payload: Any) -> None:
    require_writable(name)
    common.write_json(common.json_path(name), payload)


def geometry_reading(matrix: dict[str, Any], best: str) -> dict[str, Any]:
    """The §5 mechanism numbers for one layer, reduced to the three that decide the reading.

    L3 asks for a step where several candidates are in view and only a strict subset can be paid.
    When that never happens, the three counts that say *why* are: how often more than one candidate
    was ever in view, how far apart their prices were, and how often the step instead arrived with
    nothing payable at all.
    """
    geometry = (matrix.get("step_geometry") or {}).get(best) or {}
    multi = int(geometry.get("candidate_histogram", {}).get("two_or_more", 0))
    steps = int(geometry.get("steps", 0)) or 1
    outcomes = geometry.get("multi_step_outcomes", {})
    prices = geometry.get("multi_candidate_price_bytes", {})
    return {
        "geometry_recorded": geometry.get("geometry_recorded", False),
        "candidate_histogram": geometry.get("candidate_histogram"),
        "multi_step_outcomes": outcomes,
        "multi_candidate_price_bytes": prices,
        "indicators": {
            "share_of_steps_with_two_or_more_candidates": round(multi / steps, 5),
            "price_spread_as_fraction_of_credit": prices.get("spread_as_fraction_of_credit"),
            "share_of_multi_steps_with_nothing_payable": (
                round(int(outcomes.get("none_payable", 0)) / multi, 5) if multi else None
            ),
            "share_of_multi_steps_where_any_subset_was_payable": (
                round(int(outcomes.get("strict_subset_payable", 0)) / multi, 5) if multi else None
            ),
        },
    }


def branch(gates: dict[str, Any], coverage_complete: bool) -> dict[str, Any]:
    """The extension protocol §8 table, expressed once."""
    passed = {name: bool(gates["gates"][name]["passed"]) for name in ("L0", "L1", "L2", "L3", "L4")}
    if not coverage_complete:
        return {
            "branch": "incomplete_matrix",
            "layer_verdict": "缺格：聚合未出，判定不成立",
            "next_round": "如实报告缺格并停止；不补 seed、不重跑（执行指令 §3）",
            "training_authorized": False,
        }
    if not passed["L0"] or not passed["L1"]:
        return {
            "branch": "invalid_data",
            "layer_verdict": "L0/L1 未过：该层实验无效，只报告失效原因",
            "next_round": "停止。不把数据问题解释成任何通信结论",
            "training_authorized": False,
        }
    if passed["L2"] and not passed["L3"]:
        return {
            "branch": "l2_pass_l3_fail",
            "layer_verdict": "L2 过、L3 未过",
            "next_round": (
                "正式关闭“仅增加任务数量即可建立逐步预算选择”这条路线；不再往 14/16 个目标加。"
                "下一轮只做只读 action-geometry audit，不训练"
            ),
            "training_authorized": False,
        }
    if passed["L3"] and not passed["L4"]:
        return {
            "branch": "l3_pass_l4_fail",
            "layer_verdict": "L3 过、L4 未过：真实选择空间出现了，但固定规则间没有稳定的情境差",
            "next_round": "下一轮做发送动作价值归因，不训练",
            "training_authorized": False,
        }
    if all(passed.values()):
        return {
            "branch": "all_layers_pass",
            "layer_verdict": "L0-L4 全过",
            "next_round": (
                "才允许进入成本感知目标（cost-aware objective）协议设计：按原协议 §9 给出规则 Pareto "
                "相邻点的 lambda break-even 区间与逐流 bootstrap 支持，但不据此挑 λ、不改 reward"
            ),
            "training_authorized": False,
        }
    return {
        "branch": "l2_fail_too",
        "layer_verdict": "L2 未过（且 L0/L1 过）：12 目标把效果推到了带外（天花板或地板）",
        "next_round": (
            "与前三层同向：这条轴已经测到底。停止增加目标数；下一轮只读 action-geometry audit，不训练"
        ),
        "training_authorized": False,
    }


def build() -> dict[str, Any]:
    goals = common.G12_GOALS
    gates = json.loads(
        common.json_path(f"GATES_g{goals}.json").read_text(encoding="utf-8")
    )
    matrix = json.loads(
        common.json_path(f"RULE_MATRIX_g{goals}.json").read_text(encoding="utf-8")
    )
    pressure = json.loads(
        common.json_path(f"PRESSURE_METRICS_g{goals}.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        common.json_path(f"STREAM_MANIFEST_g{goals}.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        common.json_path(f"ROW_PROVENANCE_g{goals}.json").read_text(encoding="utf-8")
    )
    coverage = pressure_matrix.coverage(goals)
    best = matrix["best_fixed_rule"]["arm"]
    gain = matrix["bootstrap"]["on_time_per_episode"]["best_fixed"]["arms"].get("envelope", {})
    delivered = {
        str(earlier): json.loads(
            common.json_path(f"GATES_g{earlier}.json").read_text(encoding="utf-8")
        )
        for earlier in DELIVERED_LAYERS
    }
    freeze = json.loads(
        common.json_path(common.freeze_file(goals)).read_text(encoding="utf-8")
    )
    audit = json.loads(
        common.json_path(common.audit_file(goals)).read_text(encoding="utf-8")
    )
    decision = branch(gates, bool(coverage["complete"]))
    return {
        "goals": goals,
        "seeds": [common.LAYERS[goals][0], common.LAYERS[goals][-1]],
        "gates": gates,
        "matrix": matrix,
        "arms": report_pressure.arm_rows(goals, matrix),
        "search_pressure": pressure,
        "coverage": coverage,
        "provenance": provenance,
        "streams": {
            "size": len(manifest["streams"]),
            "sha256_fold": manifest["fold"]["sha256_fold"],
            "bytes": manifest["fold"]["bytes"],
            "illegal": manifest["illegal_streams"],
            "seed_audit": manifest["seed_audit"],
        },
        "freeze": {
            "file": common.freeze_file(goals),
            "head": freeze["head_at_registration"],
            "modules": freeze["module_count"],
            "episode_code_identity": freeze.get("episode_code_identity"),
            "run_episode_sha256": freeze.get("run_episode_sha256"),
            "layer": freeze.get("layer"),
        },
        "audit": audit,
        "delivered_gates": {
            layer: {name: gate["passed"] for name, gate in block["gates"].items()}
            for layer, block in delivered.items()
        },
        "best_fixed_rule": best,
        "l4_gain": float(gain.get("mean_difference") or 0.0),
        "l4_gain_ci": gain.get("ci95"),
        "geometry": geometry_reading(matrix, best),
        "delivered_geometry": {
            str(earlier): geometry_reading(
                json.loads(common.json_path(f"RULE_MATRIX_g{earlier}.json").read_text(encoding="utf-8")),
                json.loads(common.json_path(f"RULE_MATRIX_g{earlier}.json").read_text(encoding="utf-8"))[
                    "best_fixed_rule"
                ]["arm"],
            )
            for earlier in DELIVERED_LAYERS
        },
        "ceiling": report_pressure.ceiling_accounting(goals),
        "grant_audit": common.launch_grant_audit(),
        #: Disclosures the round owes the reader that are not gates. The first two were found after
        #: the first data launch, so each one says what it touched and what it could not touch.
        "post_launch_disclosures": [
            {
                "what": "the round's own grant invariant could only pass before the layer spent anything",
                "found": "after the matrix ran, when `python -m unittest test_pressure_harness` failed "
                "under the UTF-8 environment the report stage runs in",
                "symptom": "grant == 160 - (640 - matrix_starts) reads 176 once matrix_starts is 656: "
                "a live ledger cannot restate what was left at authorisation time",
                "fixed": "pressure_common.launch_grant_audit() answers the question against "
                "G12_START_LEDGER.json (matrix_starts = 496 before the layer), and the self-check and "
                "the test read that instead",
                "touched": "a self-check and one test; no episode, no row, no aggregate, no gate",
            },
            {
                "what": "episode_code_identity has two routes and only one of them is stable",
                "detail": "the route that stamps rows (inspect.getsource, locale-independent) gives "
                "ff0b4139b1fff700 for the 288 stamped cells of the delivered round (g8, g10) and for "
                "all 144 g12 cells; g6's 144 cells carry no stamp at all, as its delivered provenance "
                "file already says. The route "
                "the freeze timeline uses (git show, decoded in the ambient locale) gives "
                "c922ca100665609c under cp936 and 23390f55a6b90afa under UTF-8 for the *same* commits",
                "consequence": "the single-code-state claim is checked on the stamping route, which "
                "agrees across all four layers; a timeline hash is only comparable inside one pinned "
                "environment, and this report is generated in the one the delivered round used",
                "not_fixed": "the hashing route is left as the delivered round froze it: changing it "
                "after the launch would move an identity number without moving an episode",
            },
            {
                "what": "GATES_g12.json's stop_reason says 'proceed to the next layer'",
                "detail": "that sentence was written when layer 12 was not the last layer; the judge "
                "is frozen from the first data launch, so the wording was not edited",
                "consequence": "the next step the round actually prescribes is the §8 branch recorded in this "
                "report's section 1, and there is no fifth layer",
            },
        ],
        "packet_cost_profile": report_pressure.packet_cost_profile(goals),
        "decision": decision,
        #: Where the round started, as the remote itself reported it at delivery time. The protocol
        #: disclosed an offline start because github.com was unreachable then; this is the repair of
        #: that gap rather than a restatement of it.
        "starting_branch_remote_sha": _remote_sha("analysis/comm-task-pressure-learnability"),
        "tests": report_pressure.data_tests(),
        "hygiene": {
            "sealed": common.sealed_block_artefacts(),
            "defaults": common.production_defaults_ok(),
            "agreement": json.loads(
                common.json_path("AGREEMENT_GATE.json").read_text(encoding="utf-8")
            ),
        },
    }


def render(data: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# 任务压力 g12 独立扩展闭环：第四层实测与分支判定")
    add("")
    add(
        "日期：2026-09-23。分支 `analysis/comm-task-pressure-g12-extension`，起点 "
        "`analysis/comm-task-pressure-learnability@6b4452f5de88f7cfd0c17ce475de269eb13dd6b7`。"
        "协议 `COMM_TASK_PRESSURE_G12_EXTENSION_PROTOCOL.md` 先于任何一条 g12 episode 单独提交。"
    )
    add("")
    add(
        "**本轮 0 训练、0 `optimizer.step`、0 checkpoint、0 模型加载、0 新增验证 episode。** "
        "唯一自变量 `num_goals = 12`；额度 29 300 B、`policy_owned_timing`、isolated-lossless、"
        "ACK 关闭、九臂规则库、bootstrap 口径与前三层逐项相同。"
    )
    add("")
    add(
        "**地位**：这是对原协议预登记却没跑完的第四层的容量扩展（探索性闭环）。"
        "它不倒推修改原三层的 confirmatory 判定；原判据文件一字未动。"
    )
    add("")
    add("## 一、判定")
    add("")
    add("| 压力层 | 流 | L0 | L1 | L2 | L3 | L4 | 该层结论 |")
    add("|---|---:|---|---|---|---|---|---|")
    for layer in DELIVERED_LAYERS:
        marks = data["delivered_gates"][str(layer)]
        add(
            f"| goals={layer}（已交付） | 16 | "
            + " | ".join(("过" if marks[name] else "**不过**") for name in ("L0", "L1", "L2", "L3", "L4"))
            + " | 未通过（原判据，本轮不改） |"
        )
    marks = {name: gate["passed"] for name, gate in data["gates"]["gates"].items()}
    add(
        f"| **goals=12（本轮）** | {data['streams']['size']} | "
        + " | ".join(("过" if marks[name] else "**不过**") for name in ("L0", "L1", "L2", "L3", "L4"))
        + f" | {data['decision']['layer_verdict']} |"
    )
    add("")
    add(f"**下一步（协议 §8 规定的分支）**：{data['decision']['next_round']}")
    add("")
    add(f"本轮不训练：`training_authorized = {str(data['decision']['training_authorized']).lower()}`。")
    add("")
    add("## 二、12 目标的九臂实测")
    add("")
    add(
        f"最佳预算内固定规则 `{data['best_fixed_rule']}`；逐流 hindsight envelope 均值 "
        f"{data['matrix']['hindsight_envelope']['mean_on_time']} 个/episode；"
        f"mean detected/placed {data['search_pressure']['mean_detected_over_placed']}"
        f"（最差流 {data['search_pressure']['worst_stream_detected_over_placed']}，"
        f"seed {data['search_pressure']['worst_stream']}）。"
    )
    add("")
    add("| 臂 | 角色 | 按时完成/ep | 字节/ep | on-time/detected | 发射/ep | 额度≥90 % 流 | 争用点流 | 争用点合计 | 仅合并超支 | 拒付/ep | 多候选步 | 其中全不可付 |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in data["arms"]:
        geometry = (data["matrix"]["step_geometry"] or {}).get(row["arm"], {})
        hist = geometry.get("candidate_histogram") or {}
        outcomes = geometry.get("multi_step_outcomes") or {}
        add(
            f"| `{row['arm']}`"
            f"{' **(最佳)**' if row['is_best_fixed_rule'] else ''} | {row['role']} | "
            f"{row['on_time_per_episode']:.3f} | {row['bytes_per_episode']:.1f} | "
            f"{row['on_time_rate_of_detected']:.4f} | {row['emissions_per_episode']:.2f} | "
            f"{row['credit_ge_90_streams']} | {row['competition_streams']} | "
            f"{row['competition_points_total']} | {row['joint_only_points_total']} | "
            f"{row['budget_refusals_per_episode']:.1f} | {hist.get('two_or_more', '—')} | "
            f"{outcomes.get('none_payable', '—')} |"
        )
    add("")
    add("| 臂 | 相对 `always_hold` 完成差 | 95 % CI | 相对最佳规则完成差 | 95 % CI | 相对最佳规则字节差 | 95 % CI |")
    add("|---|---:|---|---:|---|---:|---|")
    for row in data["arms"]:
        base_a = row["vs_always_hold_on_time"] or {}
        best_a = row["vs_best_fixed_on_time"] or {}
        best_b = row["vs_best_fixed_bytes"] or {}
        add(
            f"| `{row['arm']}` | {report_pressure._fmt(base_a.get('mean_difference'))} | "
            f"{report_pressure._ci(base_a.get('ci95'))} | "
            f"{report_pressure._fmt(best_a.get('mean_difference'))} | "
            f"{report_pressure._ci(best_a.get('ci95'))} | "
            f"{report_pressure._fmt(best_b.get('mean_difference'))} | "
            f"{report_pressure._ci(best_b.get('ci95'))} |"
        )
    add("")
    add(f"- 样本均值 Pareto：{data['matrix']['mean_pareto_frontier']}")
    add(
        f"- 逐流唯一赢家计数：{data['matrix']['hindsight_envelope']['unique_winner_counts']}；"
        f"互不支配且双向胜负的规则对：{data['matrix']['incomparable_bidirectional_pairs'] or '无'}"
    )
    add(
        f"- envelope − 最佳固定规则：**{data['l4_gain']:.4f} 个/episode**，95 % CI {data['l4_gain_ci']}"
    )
    add(
        f"- 相邻点 lambda break-even（只在 L0–L4 全过时才是可引用口径）："
        f"{data['matrix']['lambda_break_even_adjacent_front']}"
    )
    l2evidence = data["gates"]["gates"]["L2"]["evidence"]
    add("")
    add("## 三、L2 的带边界读法")
    add("")
    add(
        f"- `best_on_time_rate_of_detected = {l2evidence['best_on_time_rate_of_detected']}`，"
        f"闭带 `{l2evidence['band']}`，`band_holds = {l2evidence['band_holds']}`"
    )
    if l2evidence.get("band_edge_readings"):
        readings = l2evidence["band_edge_readings"]
        add(
            f"- 该比值落在边界 ±0.005 内，按协议 §1.4 三种读法并报：朴素浮点和 "
            f"`{readings['naive_float_mean']}`、精确求和 `{readings['exact_sum_float_mean']}`、"
            f"精确有理数 `{readings['exact_rational_mean']}`（{readings['streams']} 条流）"
        )
    else:
        add("- 该比值不在任何边界的 ±0.005 之内，因此不需要 §1.4 的三读并报。")
    add("")
    add("## 四、机制诊断：额度以什么形式 binding（不参加任何判据）")
    add("")
    geometry = data["geometry"]
    add(f"- 几何量已记录：`{geometry['geometry_recorded']}`")
    add(f"- 每步候选数分布（16 流 × 1 000 步合计）：`{geometry['candidate_histogram']}`")
    add(f"- 多候选步的结局划分：`{geometry['multi_step_outcomes']}`")
    add(f"- 多候选步的包价：`{geometry['multi_candidate_price_bytes']}`")
    add(f"- 三个成因指标：`{geometry['indicators']}`")
    add("")
    profile = data["packet_cost_profile"]
    add(
        f"- 发包口径（**已发出的包**，与前三层同算法、来自已存转录，不花 episode）："
        f"{profile.get('packets', 0)} 包，"
        f"最小 {profile.get('min_bytes', 0)} B、中位 {profile.get('median_bytes', 0)} B、"
        f"最大 {profile.get('max_bytes', 0)} B、极差 {profile.get('spread_bytes', 0)} B "
        f"= 额度的 {profile.get('spread_as_fraction_of_credit', 0):.2%}；"
        f"额度总共约 {profile.get('packets_the_credit_could_pay_in_total', 0)} 个包。"
        "上面那行是**候选报价**口径（多候选步上所有候选的价格，含没发出去的），两个数不同是量纲不同，"
        "不是矛盾。"
    )
    facts = (
        data["matrix"]["step_geometry"].get(data["best_fixed_rule"], {}).get("first_credit_exhaustion", [])
    )
    add(
        f"- 首次额度耗尽发生在哪类消息：最佳规则 {len(facts)} 条流走到过\"看得见候选但一个都付不起\"，"
        f"其中 **{sum(1 for f in facts if f['first_emissions'] == 0 and f['refreshes'] > 0)} 条**"
        "的第一个 such 步上所有候选都是**刷新**（`first_emissions = 0`）——额度是在已经发过消息的目标"
        "续报时耗尽的，不是在新目标的第一条消息上。逐条明细（步号、候选数、各候选包价、首发/刷新计数）"
        "在 `GATES_g12.json` 的 `gates.L3.evidence.step_geometry_of_the_best_rule.first_credit_exhaustion`。"
    )
    outcomes = geometry["multi_step_outcomes"]
    indicators = geometry["indicators"]
    add("")
    add(
        "**L3 为什么仍未过（成因归类，按协议 §5 的三个指标，不设阈值）**：候选并发**不是**限制——"
        f"16 流合计 {geometry['candidate_histogram']['two_or_more']} 个多候选步"
        f"（占全部步的 {indicators['share_of_steps_with_two_or_more_candidates']:.1%}）；"
        f"包价极差 {indicators['price_spread_as_fraction_of_credit']:.2%} 额度，"
        "『剩余额度恰好落在两个候选报价之间』这条缝本来就窄；"
        f"而多候选步里 {indicators['share_of_multi_steps_with_nothing_payable']:.1%} 直接是"
        f"一个都付不起，只有 {indicators['share_of_multi_steps_where_any_subset_was_payable']:.1%} "
        "出现真正的严格子集。**三种成因里 binding 的是第三种：额度成片跨进全拒付区**，"
        "与前三层的读法同向，只是 12 目标把它推到了能量化的程度（严格子集步从三层的 0 变成 "
        f"{outcomes['strict_subset_payable']} 步、集中在 3 条流）。"
    )
    add("")
    add("| 压力层 | 几何量已记录 | 多候选步 | 严格子集步 | 全不可付步 | 极差/额度 |")
    add("|---|---|---:|---:|---:|---:|")
    for layer in [*(str(g) for g in DELIVERED_LAYERS), "12"]:
        item = data["delivered_geometry"].get(layer, data["geometry"] if layer == "12" else {})
        hist = item.get("candidate_histogram") or {}
        outcomes = item.get("multi_step_outcomes") or {}
        prices = item.get("multi_candidate_price_bytes") or {}
        add(
            f"| {layer} | {item.get('geometry_recorded')} | {hist.get('two_or_more', '—')} | "
            f"{outcomes.get('strict_subset_payable', '—')} | {outcomes.get('none_payable', '—')} | "
            f"{prices.get('spread_as_fraction_of_credit', '—')} |"
        )
    add("")
    add(
        "- **这张表的不对称是预先登记的**（协议 §5）：逐 step 的候选几何只有 g12 记了，因为前三层的行里"
        "没有这些键。因此不得用 g12 的分布回头改写前三层已判完的 L3；能跨层比的仍是"
        "`competition_points`、包价极差与 `credit_exhausted_with_candidates_streams`。"
    )
    add("")
    add("## 五、压力与容量")
    add("")
    per = data["search_pressure"]["per_stream"]
    n = len(per)
    add(
        f"- placed 12（`illegal_streams = {data['streams']['illegal']}`）、detected 均值 "
        f"{round(sum(v['detected'] for v in per.values()) / n, 3)}、registered 均值 "
        f"{round(sum(v['registered'] for v in per.values()) / n, 3)}、并发开放任务均值 "
        f"{round(sum(v['mean_concurrent_open'] for v in per.values()) / n, 3)}、p95 "
        f"{round(sum(v['p95_concurrent_open'] for v in per.values()) / n, 3)}、机器人利用率 "
        f"{round(sum(v['mean_robot_utilisation'] for v in per.values()) / n, 4)}、工作量/执行容量 "
        f"{round(sum(v['work_demand_vs_capacity'] for v in per.values()) / n, 3)}、从未被分配 "
        f"{round(sum(v['never_assigned_targets'] for v in per.values()) / n, 3)}"
    )
    add(
        "- 无预算参考的 `event_unbudgeted` 完成 "
        f"{[row['on_time_per_episode'] for row in data['arms'] if row['arm'] == 'event_unbudgeted'][0]} 个/ep："
        "若它也不高，未完成任务里有一部分属于搜索/执行容量，不能记在通信账上（原协议 §8）。"
    )
    add("")
    add("## 六、启动账本与上限诚实性")
    add("")
    ceiling = data["ceiling"]
    add(
        f"- 原协议上限 **{ceiling['matrix_ceiling']}**，本轮追加授权 "
        f"**{common.G12_EXTENSION_GRANT}** 次 ⇒ 有效上限 **{common.MATRIX_LAUNCH_CAP_WITH_GRANT}**；"
        f"实到数据启动 **{ceiling['matrix_starts_spent']}**（含前三层的 496 次与本层的 16 录制 + 144 矩阵）。"
    )
    add(
        f"- 核验启动 **{ceiling['gate_starts_spent']} / {ceiling['gate_ceiling']}**：本轮没有新增核验 "
        f"episode（执行指令 §3 不允许增加验证 episode），探针扩展只用零 episode 测试证明其透明。"
    )
    add(
        f"- 账本记录的技术失败 **{ceiling['technical_failures_recorded']}** 次（含前三层的 20 次），"
        "全部计入，不抹账。"
    )
    add(f"- 格覆盖：{data['coverage']['cells_present']}/{data['coverage']['cells_expected']}，"
        f"缺格 {data['coverage']['missing_count']}")
    add("")
    add("## 七、身份与卫生")
    add("")
    add(
        f"- 冻结文件 `{data['freeze']['file']}`，登记头 `{str(data['freeze']['head'])[:9]}`，模块 "
        f"{data['freeze']['modules']} 个。**给行打戳的那条身份路径**（`inspect.getsource`，与环境无关）"
        f"给出 `{data['freeze']['episode_code_identity']}`，"
        f"`run_episode` 源哈希 `{data['freeze']['run_episode_sha256']}`——"
        "g8 与 g10 的全部 288 格行戳是同一个值，本轮 144 格也是，所以这三层加本层同属一个 episode "
        "代码状态（由哈希核对）。g6 的 144 格**没有行戳**，因为戳是在那层跑完之后才加的——这一点"
        "已交付轮的 `ROW_PROVENANCE_g6.json` 就写明了，本层不替它补叙述性归因。"
    )
    add(
        "- 另一条按 commit 算的路径（报告里 freeze 时间线用的那条，`git show` 按当前 locale 解码）在 "
        "cp936 下给出 `c922ca100665609c`、在 UTF-8 下给出 `23390f55a6b90afa`，**同一批 commit 也是这两个值**"
        "——它是环境相关的，本文全程用与已交付报告相同的 cp936 环境生成，两个数都写在这里以免被当成"
        "\"代码换了\"。"
    )
    add(
        f"- 行戳：{data['provenance']['stamped']}，未戳 {data['provenance']['un_stamped']}，"
        f"单一 episode 代码状态 = {data['provenance']['single_episode_code_state']}"
    )
    add(
        f"- 四源冲突审计（{data['audit']['scope']['goals_audited']} 块）：命中 {data['audit']['hit_counts']}，"
        f"`clean = {data['audit']['clean']}`"
    )
    add(
        f"- 封存块 `800000..800399` 盘上 artefact **{data['hygiene']['sealed']['count']}**；"
        f"生产默认值 `comm_ack={data['hygiene']['defaults']['comm_ack']}`、"
        f"`comm_timing_mode={data['hygiene']['defaults']['comm_timing_mode']}`、"
        f"`num_goals={data['hygiene']['defaults']['num_goals']}`"
    )
    add(
        f"- 归档一致性（继承自原轮）：`agree = {data['hygiene']['agreement']['agree']}`；"
        "本轮探针扩展的透明性由 `test_pressure_harness` 的零 episode 测试证明"
    )
    add(
        f"- 测试：本轮合成套件 {data['tests']['round']}；项目套件 {data['tests']['project']}；"
        f"集成 smoke {data['tests']['smoke']}"
    )
    expected_start = "6b4452f5de88f7cfd0c17ce475de269eb13dd6b7"
    remote_start = data["starting_branch_remote_sha"]
    if remote_start == expected_start:
        verification = (
            f"在线核对通过：`git ls-remote` 报起点分支 "
            f"`analysis/comm-task-pressure-learnability` 的远端 HEAD = `{remote_start}`，"
            "与执行指令要求的 SHA 一致"
        )
    elif remote_start.startswith("<"):
        verification = (
            f"`git ls-remote` 此刻仍不可用（`{remote_start[1:-1]}`），所以起点分支的远端 HEAD "
            f"**没有在交付时在线核到** `{expected_start}`；已核对的只有本地远程跟踪 ref。"
            "这一条保持未闭合，不写成已核对。"
        )
    else:
        verification = (
            f"**不一致**：远端报 `{remote_start}`，要求的却是 `{expected_start}` ⇒ 起点不对，"
            "本轮所有结论都不成立，必须停止并报告"
        )
    add(
        "- 起点核对的补记：协议 §1.1 如实写过\"开跑时 github.com:443 不可达，只核对了本地远程跟踪 ref\"；"
        f"交付时再核一次：{verification}。"
        "本分支自身的远端 HEAD 无法写进本报告（报告所在的提交就是被核对的对象），"
        "它随最后一次推送单独报出。"
    )
    add("")
    add("## 八、本轮欠读者的披露（不是判据，逐条写明动了什么）")
    add("")
    for item in data["post_launch_disclosures"]:
        add(f"- **{item['what']}**")
        for key in ("found", "symptom", "fixed", "touched", "detail", "consequence", "not_fixed"):
            if item.get(key):
                add(f"  - {key}: {item[key]}")
    add("")
    grant = data["grant_audit"]
    add(
        f"- 追加授权的算术（对着开跑前的快照核）：授权前已花 "
        f"{grant['spent_before_the_layer']} / {grant['brief_ceiling']}，有效上限 "
        f"{grant['effective_ceiling']} ⇒ 空出 {grant['room_the_grant_opens']} 次，第四层需要 "
        f"{grant['layer_needs']} 次，`exact = {grant['exact']}`；本轮实到 "
        f"{data['ceiling']['matrix_starts_spent']} 次，未越线。"
    )
    add("")
    add("## 九、结论范围与下一步")
    add("")
    for line in conclusion(data):
        add(f"- {line}")
    add("")
    add("## 十、复跑")
    add("")
    add("```bash")
    add("cd runs/comm_task_pressure_learnability_20260922")
    for line in [
        "python pressure_common.py",
        "python seed_audit.py --goals 12 --out G12_SEED_COLLISION_AUDIT.json",
        "python -m unittest test_pressure_harness",
        "python freeze_pressure.py --mode register --goals 12",
        "python record_layer.py --goals 12 --shards 4 --shard 0   # 1..3 同理",
        "python record_layer.py --goals 12 --mode merge",
        "python pressure_matrix.py --goals 12 --shards 4 --shard 0   # 1..3 同理",
        "python pressure_matrix.py --goals 12 --mode coverage",
        "python pressure_matrix.py --goals 12 --mode provenance",
        "python pressure_metrics.py --goals 12",
        "python judge_layers.py --goals 12",
        "python report_g12.py",
        "cd ../.. && python -m unittest discover -s integrated_sar/tests && python -m integrated_sar",
    ]:
        add(line)
    add("```")
    add("")
    return "\n".join(lines)


def conclusion(data: dict[str, Any]) -> list[str]:
    marks = {name: gate["passed"] for name, gate in data["gates"]["gates"].items()}
    best = data["best_fixed_rule"]
    lines = [
        f"**地位**：g12 是原协议 §2 预登记的第四层，本轮按使用者追加授权（640 → 656）补跑；"
        "它是容量扩展 / 探索性闭环，不是对原三层门禁的事后改写。"
        "原判据与结果文件未被覆盖、未被重写（`report_g12.py` 与 `common.write_json` 双重守卫）。",
        f"12 目标层：最佳固定规则 `{best}`，按时完成 {data['matrix']['arms'][best]['on_time_per_episode']} 个/episode，"
        f"bytes {data['matrix']['arms'][best]['bytes_per_episode']}，"
        f"on-time/detected {data['matrix']['arms'][best]['on_time_rate_of_detected']}，"
        f"L0 {marks['L0']} / L1 {marks['L1']} / L2 {marks['L2']} / L3 {marks['L3']} / L4 {marks['L4']}。",
        f"L3 实测：最佳规则争用点流 "
        f"{data['matrix']['arms'][best]['competition_streams']}/16，额度 ≥90 % 流 "
        f"{data['matrix']['arms'][best]['credit_used_ge_90_streams']}/16。",
        f"L4 实测：envelope − 最佳固定规则 {data['l4_gain']:+.4f} 个/episode，"
        f"95 % CI {data['l4_gain_ci']}。",
        f"分支判定：**{data['decision']['branch']}** ⇒ {data['decision']['next_round']}",
        "不得据本轮写“学习无效”“规则全局最优”“不存在更优策略”：本轮仍只测了目标数这一条轴，"
        "且 L2/L3/L4 是必要条件门禁，不是策略优劣的判决。",
        "本轮未训练、未改 reward、未新增规则/seed/验证 episode、未打开 `800000..800399`、"
        "未修改生产默认值；λ 只在 L0–L4 全过时才是可引用口径。",
        "跑完即停止：下一轮方向由使用者按 §8 的分支决定，本轮不自动展开。",
    ]
    return lines


def main(argv: list[str]) -> int:
    data = build()
    results_json = {
        key: value for key, value in data.items() if key != "matrix"
    }
    results_json["matrix_headline"] = {
        "best_fixed_rule": data["matrix"]["best_fixed_rule"],
        "arms": {
            arm: {
                key: table[key]
                for key in (
                    "role",
                    "on_time_per_episode",
                    "bytes_per_episode",
                    "on_time_rate_of_detected",
                    "competition_streams",
                    "credit_used_ge_90_streams",
                )
            }
            for arm, table in data["matrix"]["arms"].items()
        },
        "mean_pareto_frontier": data["matrix"]["mean_pareto_frontier"],
        "hindsight_envelope": {
            "mean_on_time": data["matrix"]["hindsight_envelope"]["mean_on_time"],
            "unique_winner_counts": data["matrix"]["hindsight_envelope"]["unique_winner_counts"],
        },
        "lambda_break_even_adjacent_front": data["matrix"][
            "lambda_break_even_adjacent_front"
        ],
        "bootstrap_audit": data["matrix"]["bootstrap_audit"],
    }
    write("COMM_TASK_PRESSURE_G12_EXTENSION_RESULTS.json", results_json)
    require_writable("COMM_TASK_PRESSURE_G12_EXTENSION_RESULTS.md")
    common.json_path("COMM_TASK_PRESSURE_G12_EXTENSION_RESULTS.md").write_text(
        render(data), encoding="utf-8"
    )
    verdict = {
        "round": "comm_task_pressure_g12_extension",
        "date": "2026-09-23",
        "branch_name": common.git("rev-parse", "--abbrev-ref", "HEAD"),
        "starting_head": "6b4452f5de88f7cfd0c17ce475de269eb13dd6b7",
        "layer": common.G12_GOALS,
        "seeds": [common.LAYERS[common.G12_GOALS][0], common.LAYERS[common.G12_GOALS][-1]],
        "status": (
            "capacity extension of the pre-registered fourth pressure layer; exploratory closure, "
            "not a revision of the delivered round's confirmatory gates"
        ),
        "training_steps": 0,
        "optimizer_steps": 0,
        "checkpoints_written": 0,
        "new_streams": data["streams"]["size"],
        "verification_episodes_added": 0,
        "coverage": {
            "cells_expected": data["coverage"]["cells_expected"],
            "cells_present": data["coverage"]["cells_present"],
            "missing": data["coverage"]["missing_count"],
        },
        "gates": {
            name: {"passed": gate["passed"], "evidence": gate["evidence"]}
            for name, gate in data["gates"]["gates"].items()
        },
        "layer_passed": data["gates"]["passed"],
        "delivered_layers_unchanged": data["delivered_gates"],
        "mechanism_geometry": data["geometry"],
        "packet_cost_profile": data["packet_cost_profile"],
        "decision": data["decision"],
        "launch_ceiling": {
            "brief_ceiling": common.MATRIX_LAUNCH_CAP,
            "extension_grant": common.G12_EXTENSION_GRANT,
            "effective_ceiling": common.MATRIX_LAUNCH_CAP_WITH_GRANT,
            "spent_after_this_round": common.ledger_totals()["matrix_starts"],
            "gate_cap": common.GATE_LAUNCH_CAP,
            "gate_spent": common.ledger_totals()["gate_starts"],
            "technical_failures_recorded": common.ledger_totals()["failures"],
        },
        "episode_code_identity": data["freeze"]["episode_code_identity"],
        "stops_after_this_round": True,
    }
    write("COMM_TASK_PRESSURE_G12_EXTENSION_VERDICT.json", verdict)
    write(
        "COMM_TASK_PRESSURE_G12_EXTENSION_MANIFEST.json",
        {
            "round": "comm_task_pressure_g12_extension",
            "date": "2026-09-23",
            "branch": common.git("rev-parse", "--abbrev-ref", "HEAD"),
            "head_at_report": common.git("rev-parse", "HEAD"),
            "starting_head": "6b4452f5de88f7cfd0c17ce475de269eb13dd6b7",
            "extends": {
                "protocol": "COMM_TASK_PRESSURE_LEARNABILITY_PROTOCOL.md",
                "results": "COMM_TASK_PRESSURE_LEARNABILITY_RESULTS.md",
                "delivered_layers": list(DELIVERED_LAYERS),
                "delivered_layer_verdicts": data["delivered_gates"],
            },
            "authorization": {
                "granted": "16 additional data launches (640 -> 656) for the unrun fourth layer",
                "by": "the user's order of 2026-09-23",
                "recorded_in": "COMM_TASK_PRESSURE_G12_EXTENSION_PROTOCOL.md §2",
                "not_granted": [
                    "training",
                    "reward changes",
                    "new arms or seeds",
                    "extra verification episodes",
                    "opening 800000..800399",
                    "raising the gate ceiling",
                ],
            },
            "setting": {
                "budget_bytes": common.BUDGET_BYTES,
                "timing_mode": common.TIMING_MODE,
                "information": common.INFORMATION,
                "delivery_probability": common.DELIVERY_PROBABILITY,
                "ack": common.ACK_ENABLED,
                "steps": common.STREAM_STEPS,
                "arms": list(common.ARM_ORDER),
                "only_variable": "num_goals = 12",
            },
            "artefacts_committed": [
                "COMM_TASK_PRESSURE_G12_EXTENSION_PROTOCOL.md",
                "COMM_TASK_PRESSURE_G12_EXTENSION_RESULTS.md/.json",
                "COMM_TASK_PRESSURE_G12_EXTENSION_VERDICT.json",
                "COMM_TASK_PRESSURE_G12_EXTENSION_MANIFEST.json",
                "G12_FREEZE.json",
                "G12_START_LEDGER.json",
                "G12_SEED_COLLISION_AUDIT.json",
                "STREAM_MANIFEST_g12.json",
                "RULE_MATRIX_g12.json",
                "PRESSURE_METRICS_g12.json",
                "GATES_g12.json",
                "ROW_PROVENANCE_g12.json",
                "LAUNCH_LEDGER*g12*.json",
            ],
            "never_committed": ["streams", "per-episode logs", "checkpoints", "rows/", "runs/"],
            "delivered_artefacts_left_byte_identical": sorted(
                name for name in common.DELIVERED_ARTEFACTS
            ),
            "freeze": data["freeze"],
            "audit": {
                "clean": data["audit"]["clean"],
                "hit_counts": data["audit"]["hit_counts"],
                "scope": data["audit"]["scope"],
            },
            "tests": data["tests"],
            "decision": data["decision"],
        },
    )
    print(
        json.dumps(
            {
                "gates": {
                    name: gate["passed"] for name, gate in data["gates"]["gates"].items()
                },
                "best_fixed_rule": data["best_fixed_rule"],
                "coverage": data["coverage"]["cells_present"],
                "decision": data["decision"],
                "geometry": data["geometry"]["indicators"],
                "tests": data["tests"],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
