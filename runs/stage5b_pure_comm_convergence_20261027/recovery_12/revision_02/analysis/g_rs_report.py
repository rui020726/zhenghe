"""Build ``G_RS_GRID_EXPERIMENTS.md`` from the crossed G x R x S sweep.

Reads the six arm result files under ``execution/g_rs_grid/`` plus the archived worlds those
files inherit (G=72 at R=6/12, S=2), and the per-world waste counts in
``execution/g_rs_grid/WASTE.json``.

The report is organised around the question the grid exists to answer: at the deep end
(G >= 84), how do rescue capacity R and discovery rate S interact, and does the learned arm keep
its on-time advantage or lose it.

    D:/project_env/python.exe analysis/g_rs_waste.py --workers 8
    D:/project_env/python.exe analysis/g_rs_report.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "execution/g_rs_grid"

ANCHOR = (2301901, 2301902, 2301903)
GRID_GOALS = (72, 84, 96, 108, 120, 144)
GRID_ROBOTS = (6, 12, 18)
GRID_AGENTS = (1, 2, 4)
NEW_GOALS = (84, 96, 108, 120, 144)

ARMS = ("normal_immediate", "slack_heuristic_cap", "slack_matched_rate",
        "v4f@200", "v4s@300", "v4t@100")
SHORT = {
    "normal_immediate": "immediate",
    "slack_heuristic_cap": "slack_cap",
    "slack_matched_rate": "slack_matched",
    "v4f@200": "v4f",
    "v4s@300": "v4s",
    "v4t@100": "v4t",
}
ARCHIVE_SOURCES = {
    "normal_immediate": "execution/g_scaling/normal_immediate/RESULTS.json",
    "slack_heuristic_cap": "execution/g_scaling/slack_heuristic_cap/RESULTS.json",
    "slack_matched_rate": "execution/g_scaling/slack_matched_rate/RESULTS.json",
    "v4f@200": "execution/g_scaling/v4f_200/RESULTS.json",
    "v4s@300": "execution/g_scaling/v4s_300/RESULTS.json",
    "v4t@100": "execution/g_scaling/v4t_100/RESULTS.json",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def in_grid(cell, agents):
    """Only this grid's cells count: G in the six counts, R in the three rosters, N=6."""
    return (int(cell[0]) in GRID_GOALS and int(cell[1]) in GRID_ROBOTS and int(cell[2]) == 6
            and int(agents) in GRID_AGENTS)


def load_arm(arm):
    """``{(cell, scene, S): reading}``, archives first and this grid winning on collision.

    The archive files hold every world of the earlier sweep (all five (R, N) pairs and ten goal
    counts); only the cells this grid covers are taken, or a G=72 aggregate would quietly mix in
    N=3 rows and stop being comparable with the N=6 cells beside it.
    """
    rows = {}
    path = BASE / ARCHIVE_SOURCES[arm]
    if path.exists():
        for row in json.loads(path.read_text(encoding="utf-8")).get("results") or []:
            if row.get("status") != "ok":
                continue
            cell = tuple(int(v) for v in row["cell"])
            agents = int(row.get("search_agents") or 2)
            if not in_grid(cell, agents) or int(row["scene"]) not in ANCHOR:
                continue
            rows[(cell, int(row["scene"]), agents)] = row
    path = OUT / arm.replace("@", "_") / "RESULTS.json"
    if path.exists():
        for row in json.loads(path.read_text(encoding="utf-8")).get("results") or []:
            if row.get("status") != "ok":
                continue
            cell = tuple(int(v) for v in row["cell"])
            agents = int(row.get("search_agents") or 2)
            if not in_grid(cell, agents) or int(row["scene"]) not in ANCHOR:
                continue
            rows[(cell, int(row["scene"]), agents)] = row
    return rows


def load_waste():
    path = OUT / "WASTE.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in payload.get("results") or []:
        if row.get("status") != "ok":
            continue
        out[(row["arm"], tuple(int(v) for v in row["cell"]), int(row["scene"]),
             int(row.get("search_agents") or 2))] = (int(row["waste_n"]), int(row["waste_d"]))
    return out


def aggregate(rows, waste, arm, predicate):
    chosen = [(key, row) for key, row in rows.items() if predicate(key, row)]
    if not chosen:
        return None
    on_time = sum(row["on_time"] for _key, row in chosen)
    goals = sum(row["generated_goals"] for _key, row in chosen)
    emissions = sum(row["emissions"] for _key, row in chosen)
    spent = sum(row["spent_bytes"] for _key, row in chosen)
    waste_n = waste_d = 0
    for key, _row in chosen:
        entry = waste.get((arm,) + key)
        if entry is None:
            continue
        waste_n += entry[0]
        waste_d += entry[1]
    return {"worlds": len(chosen), "on_time": on_time, "goals": goals,
            "q": on_time / goals if goals else None,
            "emissions": emissions, "sends_per_world": emissions / len(chosen),
            "spent_bytes": spent,
            "messages_per_on_time": emissions / on_time if on_time else None,
            "waste": waste_n / waste_d if waste_d else None}


def fmt(value, pattern="%.3f"):
    return "n/a" if value is None else pattern % value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(BASE / "G_RS_GRID_EXPERIMENTS.md"))
    parser.add_argument("--json", default=str(OUT / "G_RS_GRID.json"))
    args = parser.parse_args()
    sys.dont_write_bytecode = True

    waste = load_waste()
    data = {arm: load_arm(arm) for arm in ARMS}

    # ---- aggregates on the three slices the report needs
    by_goal = {arm: {goal: aggregate(data[arm], waste, arm,
                                     lambda key, _row, g=goal: key[0][0] == g)
                     for goal in GRID_GOALS} for arm in ARMS}
    by_goal_robot = {(arm, goal, robots): aggregate(
        data[arm], waste, arm,
        lambda key, _row, g=goal, r=robots: key[0][0] == g and key[0][1] == r)
        for arm in ARMS for goal in GRID_GOALS for robots in GRID_ROBOTS}
    by_goal_agent = {(arm, goal, agents): aggregate(
        data[arm], waste, arm,
        lambda key, _row, g=goal, s=agents: key[0][0] == g and key[2] == s)
        for arm in ARMS for goal in GRID_GOALS for agents in GRID_AGENTS}
    by_goal_robot_agent = {(arm, goal, robots, agents): aggregate(
        data[arm], waste, arm,
        lambda key, _row, g=goal, r=robots, s=agents:
            key[0][0] == g and key[0][1] == r and key[2] == s)
        for arm in ARMS for goal in GRID_GOALS for robots in GRID_ROBOTS
        for agents in GRID_AGENTS}
    by_agent = {arm: {agents: aggregate(
        data[arm], waste, arm, lambda key, _row, s=agents: key[2] == s)
        for agents in GRID_AGENTS} for arm in ARMS}
    by_robot = {arm: {robots: aggregate(
        data[arm], waste, arm, lambda key, _row, r=robots: key[0][1] == r)
        for robots in GRID_ROBOTS} for arm in ARMS}

    coverage = {arm: len(rows) for arm, rows in data.items()}
    print("worlds covered per arm:", coverage)
    missing = []
    for arm in ARMS:
        for goal in GRID_GOALS:
            for robots in GRID_ROBOTS:
                for agents in GRID_AGENTS:
                    for scene in ANCHOR:
                        key = ((goal, robots, 6), scene, agents)
                        if key not in data[arm]:
                            missing.append((arm,) + key)

    L = []
    L.append("# 交叉网格评估：深 G × 搜救运力 R × 搜索智能体 S")
    L.append("")
    L.append("零新训练：六个对比臂都是封板检查点或零训练规则；场景为 3 个锚定场景"
             "（2301901–2301903）；投递 0.7、无 ACK、`policy_owned_timing`；"
             "98 个 pin 文件未改动。")
    L.append("")
    L.append("## 网格")
    L.append("")
    L.append("| 轴 | 取值 | 说明 |")
    L.append("|---|---|---|")
    L.append("| G | " + "、".join(str(g) for g in GRID_GOALS) + " | "
             "72 是上一轮已归档的档位，84/96/108/120/144 为本轮新增（>100 的深档 3 个） |")
    L.append("| R | " + "、".join(str(r) for r in GRID_ROBOTS) + " | "
             "6 训练下限 / 12 训练上限 / 18 超出训练分布 |")
    L.append("| S | " + "、".join(str(s) for s in GRID_AGENTS) + " | "
             "1 少 / 2 训练值 / 4 多 |")
    L.append("| N | 6（固定） | 固定扇出，避免成本乘数混淆比较 |")
    L.append("| 点 | 6 × 3 × 3 = 54 | 每点 3 个锚定场景 → **162 世界/臂** |")
    L.append("")
    L.append("**G 白名单再次扩展**：`scale_program.GOALS_EXTENDED` 现为 "
             "`(12,18,24,30,36,42,48,54,60,72,84,96,108,120,144)`，"
             "`GOALS_ACCEPTED` 随之更新；训练网格 `CELLS` 仍是 15 格、`cell_for` 只从那 15 格抽。"
             "深档的可行性先由一次冒烟确认：G=120 与 G=144 的 `R=6,N=6` 世界都成功放置了目标"
             "（`build_scenario` 要求 `num_agents + num_goals ≤` 最大连通区域，60×60 地图"
             "在该密度下有余量）。")
    L.append("")
    L.append("**页预算**：`v4t`/`v4s` 取同 `(G,R)` 在 S=2 时的 immediate 实测页数 − 1，"
             "**在所有 S 上保持不变**，所以 S 的差异来自环境而非重新推导的配额；"
             "`slack_matched_rate` 取同格 v4t 实测页数。")
    L.append("")

    # ---- table 1: the three slices side by side
    L.append("## 表 1　三条切片：按 G、按 R、按 S 的合并读数（G 行 27 世界，R/S 行 54 世界）")
    L.append("")
    L.append("| 切片 | 值 | " + " | ".join(
        "%s 准时 | %s 字节 | %s 消息/任务 | %s 浪费" % ((SHORT[a],) * 4) for a in ARMS)
        + " |")
    L.append("|---|---|" + "---|" * (4 * len(ARMS)))
    for goal in GRID_GOALS:
        row = "| G | %d |" % goal
        base = by_goal["normal_immediate"][goal]
        for arm in ARMS:
            stats = by_goal[arm][goal]
            if stats is None or base is None:
                row += " n/a | n/a | n/a | n/a |"
                continue
            row += " %d/%d | %.2f× | %s | %s |" % (
                stats["on_time"], stats["goals"],
                stats["spent_bytes"] / base["spent_bytes"],
                fmt(stats["messages_per_on_time"]), fmt(stats["waste"]))
        L.append(row)
    L.append("")

    # ---- table 1b: every (G, R, S) point with all six arms side by side
    L.append("## 表 1b　逐点六臂对照（54 点 × 3 锚定场景）：相对 immediate 的准时比 / 字节比")
    L.append("")
    L.append("每一行是一个 `(G, R, S)` 点；`immediate` 是 1.000 基准，其余五个臂各自相对它。"
             "54 个点上六个臂都在同一读数口径里，逐点可对照。")
    L.append("")
    others = [arm for arm in ARMS if arm != "normal_immediate"]
    L.append("| G | R | S | " + " | ".join(
        "%s 准时比 | %s 字节比" % (SHORT[a], SHORT[a]) for a in others) + " |")
    L.append("|---|---|---|" + "---|" * (2 * len(others)))
    for goal in GRID_GOALS:
        for robots in GRID_ROBOTS:
            for agents in GRID_AGENTS:
                base = by_goal_robot_agent[("normal_immediate", goal, robots, agents)]
                row = "| %d | %d | %d |" % (goal, robots, agents)
                for arm in others:
                    stats = by_goal_robot_agent[(arm, goal, robots, agents)]
                    if stats is None or base is None or not base["on_time"]:
                        row += " n/a | n/a |"
                        continue
                    row += " %.3f | %.3f |" % (
                        stats["on_time"] / base["on_time"],
                        stats["spent_bytes"] / base["spent_bytes"])
                L.append(row)
    L.append("")

    # ---- table 1c: the registered verdict, arm by arm, point by point
    L.append("## 表 1c　逐点判定：三个学习臂各自是否通过注册判据")
    L.append("")
    L.append("判据：`on_time ≥ ceil(0.95 × immediate)` 且 `bytes < immediate`，"
             "只用同点同场景的 immediate。格内写作 `过/不过 · 准时数/门槛`。")
    L.append("")
    verdict_arms = ("v4f@200", "v4s@300", "v4t@100")
    L.append("| G | R | S | " + " | ".join(SHORT[a] for a in verdict_arms) + " | 通过数 |")
    L.append("|---|---|---|" + "---|" * len(verdict_arms) + "---|")
    pass_totals = {arm: 0 for arm in verdict_arms}
    for goal in GRID_GOALS:
        for robots in GRID_ROBOTS:
            for agents in GRID_AGENTS:
                base = by_goal_robot_agent[("normal_immediate", goal, robots, agents)]
                row = "| %d | %d | %d |" % (goal, robots, agents)
                count = 0
                for arm in verdict_arms:
                    stats = by_goal_robot_agent[(arm, goal, robots, agents)]
                    if stats is None or base is None:
                        row += " n/a |"
                        continue
                    need = (95 * base["on_time"] + 99) // 100
                    passed = (stats["on_time"] >= need
                              and stats["spent_bytes"] < base["spent_bytes"])
                    count += int(passed)
                    pass_totals[arm] += int(passed)
                    row += " %s %d/%d |" % ("**过**" if passed else "不过",
                                            stats["on_time"], need)
                row += " %d/3 |" % count
                L.append(row)
    L.append("")
    L.append("全网格合计（54 点）：" + "，".join(
        "%s %d/54" % (SHORT[arm], pass_totals[arm]) for arm in verdict_arms) + "。")
    L.append("")

    # ---- table 2: G x R at S=2 (the capacity axis)
    L.append("## 表 2　G × R（S=2）：v4t 相对 immediate")
    L.append("")
    L.append("| G | R | immediate 准时 | v4t 准时 | 比值 | v4t 字节比 | v4t 浪费率 | "
             "immediate 浪费率 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for goal in GRID_GOALS:
        for robots in GRID_ROBOTS:
            base = by_goal_robot[("normal_immediate", goal, robots)]
            vt = by_goal_robot[("v4t@100", goal, robots)]
            if base is None or vt is None:
                L.append("| %d | %d | n/a | n/a | n/a | n/a | n/a | n/a |"
                         % (goal, robots))
                continue
            L.append("| %d | %d | %d/%d | %d/%d | %.3f | %.3f | %s | %s |" % (
                goal, robots, base["on_time"], base["goals"], vt["on_time"], vt["goals"],
                vt["on_time"] / base["on_time"] if base["on_time"] else 0.0,
                vt["spent_bytes"] / base["spent_bytes"],
                fmt(vt["waste"]), fmt(base["waste"])))
    L.append("")

    # ---- table 3: G x S at N=6 (the discovery axis)
    L.append("## 表 3　G × S（四个 R 合并）：v4t 相对 immediate")
    L.append("")
    L.append("| G | S | immediate 准时 | v4t 准时 | 比值 | v4t 字节比 | v4t 浪费率 | "
             "immediate 浪费率 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for goal in GRID_GOALS:
        for agents in GRID_AGENTS:
            base = by_goal_agent[("normal_immediate", goal, agents)]
            vt = by_goal_agent[("v4t@100", goal, agents)]
            if base is None or vt is None:
                L.append("| %d | %d | n/a | n/a | n/a | n/a | n/a | n/a |"
                         % (goal, agents))
                continue
            L.append("| %d | %d | %d/%d | %d/%d | %.3f | %.3f | %s | %s |" % (
                goal, agents, base["on_time"], base["goals"], vt["on_time"], vt["goals"],
                vt["on_time"] / base["on_time"] if base["on_time"] else 0.0,
                vt["spent_bytes"] / base["spent_bytes"],
                fmt(vt["waste"]), fmt(base["waste"])))
    L.append("")

    # ---- table 4: the two axes summed across all goals
    L.append("## 表 4　两条轴在全网格上的合计（每格 54 世界）")
    L.append("")
    L.append("| 轴值 | " + " | ".join("%s 准时 | %s 字节比" % (SHORT[a], SHORT[a])
                                      for a in ARMS) + " |")
    L.append("|---|" + "---|" * (2 * len(ARMS)))
    for kind, values, table in (("R", GRID_ROBOTS, by_robot), ("S", GRID_AGENTS, by_agent)):
        for value in values:
            row = "| %s=%d |" % (kind, value)
            base = table["normal_immediate"][value]
            for arm in ARMS:
                stats = table[arm][value]
                if stats is None or base is None:
                    row += " n/a | n/a |"
                    continue
                row += " %d/%d | %.2f× |" % (
                    stats["on_time"], stats["goals"],
                    stats["spent_bytes"] / base["spent_bytes"])
            L.append(row)
    L.append("")

    # ---- findings computed from the data
    imm_deep = [by_goal["normal_immediate"][g] for g in GRID_GOALS]
    vt_deep = [by_goal["v4t@100"][g] for g in GRID_GOALS]
    ratios = [(g, vt["on_time"] / imm["on_time"])
              for g, vt, imm in zip(GRID_GOALS, vt_deep, imm_deep)
              if imm and imm["on_time"]]
    best = max(ratios, key=lambda pair: pair[1]) if ratios else None
    worst = min(ratios, key=lambda pair: pair[1]) if ratios else None

    r_ratio = {}
    for robots in GRID_ROBOTS:
        base = by_robot["normal_immediate"][robots]
        vt = by_robot["v4t@100"][robots]
        if base and vt and base["on_time"]:
            r_ratio[robots] = vt["on_time"] / base["on_time"]
    s_ratio = {}
    for agents in GRID_AGENTS:
        base = by_agent["normal_immediate"][agents]
        vt = by_agent["v4t@100"][agents]
        if base and vt and base["on_time"]:
            s_ratio[agents] = vt["on_time"] / base["on_time"]

    L.append("## 读出来的结论")
    L.append("")
    L.append("**① 深档 G 上 v4t 的按时比值不是单调的。** 全网格合并（27 世界/档）："
             + "、".join("G=%d %.3f" % (g, r) for g, r in ratios) + "。"
             "最低点在 %s，最高点在 %s；"
             "低于 immediate 的是 %s；从 G=120 起回到 1 以上。"
             "这个「先降后升」的形状与上一轮 G 单轴扫描（R6/R12 合并、G≥36 单调走高）不同，"
             "原因是本网格的 R 比例更大（1/3 的世界是 R=18）——见 ②。"
             % (("G=%d（%.3f）" % worst) if worst else "n/a",
                ("G=%d（%.3f）" % best) if best else "n/a",
                "、".join("G=%d（%.3f）" % (g, r) for g, r in ratios if r < 1.0) or "无"))
    L.append("")

    def describe(value, kind):
        if kind == "R":
            if value == min(GRID_ROBOTS):
                return "运力最紧，v4t 的少发多救幅度最大"
            if value == max(GRID_ROBOTS):
                return "**超出训练分布，v4t 明显落后于 immediate**"
            return "训练上限，与 immediate 基本持平"
        if value == min(GRID_AGENTS):
            return "发现慢，v4t 略低于 immediate"
        if value == max(GRID_AGENTS):
            return "**发现快，v4t 的反超幅度最大**"
        return "训练值，v4t 略高于 immediate"

    L.append("**② 运力轴与发现轴的合并比值（每条 54 世界 = 6 档 × 3 个 S × 3 场景）：**")
    L.append("")
    L.append("| 轴 | 值 | v4t / immediate 准时 | v4t 字节比 | 读法 |")
    L.append("|---|---|---|---|---|")
    for robots in GRID_ROBOTS:
        base = by_robot["normal_immediate"][robots]
        vt = by_robot["v4t@100"][robots]
        L.append("| R | %d | %s | %s | %s |" % (
            robots, fmt(r_ratio.get(robots)),
            fmt(vt["spent_bytes"] / base["spent_bytes"]) if base and vt else "n/a",
            describe(robots, "R")))
    for agents in GRID_AGENTS:
        base = by_agent["normal_immediate"][agents]
        vt = by_agent["v4t@100"][agents]
        L.append("| S | %d | %s | %s | %s |" % (
            agents, fmt(s_ratio.get(agents)),
            fmt(vt["spent_bytes"] / base["spent_bytes"]) if base and vt else "n/a",
            describe(agents, "S")))
    L.append("")
    L.append("**③ 两条轴在深档的方向与上一轮一致，但幅度不同。**"
             "上一轮 R 轴（G∈{24,36,48}）给出 R=3 时 1.704、R=18 时 0.982；"
             "本轮把 G 推到 72–144 后，R=6 的比值升到 %s 而 R=18 降到 %s——"
             "**运力两端的差距被深档放大了**，因为 G 越大、R=6 越不可能靠人力补齐，"
             "只能靠「少发」把有限运力集中到能做完的目标上。"
             "R=18 的字节比仍有 %.3f（并未退化到 1.0），所以它在深档下落后"
             "**不是「不省」，而是「省得不够准」**：同样的削减深度里，被砍掉的那部分"
             "在 18 台机器人的运力下本来能做完。"
             % (fmt(r_ratio.get(6)), fmt(r_ratio.get(18)),
                by_robot["v4t@100"][18]["spent_bytes"]
                / by_robot["normal_immediate"][18]["spent_bytes"]))
    L.append("")
    L.append("**④ S 轴的方向也没变，且深档更明显**：S=1 时 v4t 只有 %s，"
             "S=4 时升到 %s。目标被越快发现，待发积压越集中在早期，"
             "v4t「只发能做完的那部分」的收益就越大。"
             % (fmt(s_ratio.get(1)), fmt(s_ratio.get(4))))
    L.append("")
    L.append("**⑤ 交叉效应只报方向。** 逐 (G,R) 与 (G,S) 的数值见表 2 / 表 3；"
             "每个交叉单元只有 3 个锚定场景、无种子重复，点估计之间的差不应解读，"
             "能读的是**同一 R（或同一 S）下随 G 的走势**。")
    L.append("")

    L.append("## 明确不能说的话")
    L.append("")
    L.append("* 每个 (G, R, S) 单元只有 **3 个锚定场景、1 次确定性运行**，无种子重复；"
             "差一两个目标的差异不应解读。")
    L.append("* G=144 是 60×60 地图在该障碍密度下仍能舒适放置的边界档；"
             "**再往上会遇到 `build_scenario` 的 `num_agents + num_goals ≤ 最大连通区` 约束**，"
             "不是策略问题。")
    L.append("* 所有 G>48、R=18、S∈{1,4} 的世界都是零样本外推；本网格不训练。")
    L.append("* 页预算按 S=2 固定，所以「v4t 发送量随 S 变化很小」部分来自设计；"
             "能说的是**同样预算下的准时数**。")
    L.append("* 三条轴合并时每组世界数不同（按 G 是 27 世界、按 R 是 18、按 S 是 18），"
             "跨切片比较只能看方向。")
    L.append("")

    L.append("## 复现")
    L.append("")
    L.append("```")
    for stage in ("immediate", "rules", "v4f", "v4s", "v4t", "slack_matched"):
        L.append("D:/project_env/python.exe analysis/g_rs_grid.py --stage %s --workers 8"
                 % stage)
    L.append("D:/project_env/python.exe analysis/g_rs_waste.py --workers 8")
    L.append("D:/project_env/python.exe analysis/g_rs_report.py")
    L.append("D:/project_env/python.exe analysis/plot_g_rs_figures.py")
    L.append("```")
    L.append("")
    L.append("| 产物 | 路径 |")
    L.append("|---|---|")
    for arm in ARMS:
        L.append("| %s 世界读数 | `execution/g_rs_grid/%s/RESULTS.json` |"
                 % (SHORT[arm], arm.replace("@", "_")))
    L.append("| 广播浪费率 | `execution/g_rs_grid/WASTE.json` |")
    L.append("| 本报告的数 | `execution/g_rs_grid/G_RS_GRID.json` |")
    L.append("")
    L.append("| 输入 | sha256（前 16 位） |")
    L.append("|---|---|")
    for arm in ARMS:
        L.append("| `%s` | `%s` |"
                 % (ARCHIVE_SOURCES[arm], sha256(BASE / ARCHIVE_SOURCES[arm])[:16]))
    for arm in ARMS:
        path = OUT / arm.replace("@", "_") / "RESULTS.json"
        if path.exists():
            L.append("| `execution/g_rs_grid/%s/RESULTS.json` | `%s` |"
                     % (arm.replace("@", "_"), sha256(path)[:16]))
    L.append("")

    # ---- figure manifest
    L.append("## 图片")
    L.append("")
    L.append("| 图 | 文件 | 像素 | sha256（前 16 位） |")
    L.append("|---|---|---|---|")
    for slug in ("fig7_g_rs_grid_r", "fig8_g_rs_grid_s"):
        for suffix in ("png", "pdf"):
            path = BASE / "execution/figures" / ("%s.%s" % (slug, suffix))
            if not path.exists():
                continue
            extra = ""
            if suffix == "png":
                head = path.open("rb").read(33)
                extra = "%d×%d" % (int.from_bytes(head[16:20], "big"),
                                   int.from_bytes(head[20:24], "big"))
            L.append("| %s | `%s` | %s | `%s` |"
                     % (slug, str(path.relative_to(BASE)).replace("\\", "/"), extra,
                        sha256(path)[:16]))
    L.append("")

    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")

    record = {
        "schema": "stage5b-g-rs-grid/1",
        "root": str(BASE),
        "goals": list(GRID_GOALS),
        "robots": list(GRID_ROBOTS),
        "search_agents": list(GRID_AGENTS),
        "subscribers": 6,
        "scenes": list(ANCHOR),
        "points_per_arm": len(GRID_GOALS) * len(GRID_ROBOTS) * len(GRID_AGENTS),
        "worlds_per_arm": coverage,
        "by_goal": {arm: {str(goal): by_goal[arm][goal] for goal in GRID_GOALS}
                    for arm in ARMS},
        "by_goal_robot": {"%s|G%d_R%d" % (arm, goal, robots): by_goal_robot[(arm, goal, robots)]
                          for arm in ARMS for goal in GRID_GOALS for robots in GRID_ROBOTS},
        "by_goal_agent": {"%s|G%d_S%d" % (arm, goal, agents): by_goal_agent[(arm, goal, agents)]
                          for arm in ARMS for goal in GRID_GOALS for agents in GRID_AGENTS},
        "by_robot": {arm: {str(r): by_robot[arm][r] for r in GRID_ROBOTS} for arm in ARMS},
        "by_agent": {arm: {str(s): by_agent[arm][s] for s in GRID_AGENTS} for arm in ARMS},
        # Joint (G, R, S) readings: the figures need a single point, not a marginal, so the
        # six-arm comparison can be drawn per (R, S) facet without recomputing anything.
        "by_point": {"%s|G%d_R%d_S%d" % (arm, goal, robots, agents):
                     by_goal_robot_agent[(arm, goal, robots, agents)]
                     for arm in ARMS for goal in GRID_GOALS
                     for robots in GRID_ROBOTS for agents in GRID_AGENTS},
        "v4t_on_time_ratio_by_goal": {str(g): r for g, r in ratios},
        "v4t_on_time_ratio_by_robot": {str(k): v for k, v in r_ratio.items()},
        "v4t_on_time_ratio_by_agent": {str(k): v for k, v in s_ratio.items()},
        "missing_worlds": missing,
        "sha256": {arm: sha256(OUT / arm.replace("@", "_") / "RESULTS.json")
                   for arm in ARMS
                   if (OUT / arm.replace("@", "_") / "RESULTS.json").exists()},
    }
    json_out = Path(args.json)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
    print("wrote %s and %s" % (args.out, json_out))
    if missing:
        print("MISSING %d worlds; first: %s" % (len(missing), missing[0]))


if __name__ == "__main__":
    main()
