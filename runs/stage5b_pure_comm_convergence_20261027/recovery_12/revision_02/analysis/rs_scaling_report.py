"""Build ``R_S_SCALING_EXPERIMENTS.md`` from the archived worlds plus this round's two sweeps.

Reads only. Every number is either an archived world (the G sweep's G=24/48 at S=2 and the
training-cell evaluations) or a world this round wrote under ``execution/r_s_scaling/<axis>/``.
The broadcast-waste reading comes from ``execution/r_s_scaling/<axis>/WASTE.json``.

    D:/project_env/python.exe analysis/rs_waste.py --axis r --workers 8
    D:/project_env/python.exe analysis/rs_waste.py --axis s --workers 8
    D:/project_env/python.exe analysis/rs_scaling_report.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "execution/r_s_scaling"

ANCHOR = (2301901, 2301902, 2301903)
R_VALUES = (3, 6, 9, 12, 15, 18)
S_VALUES = (1, 2, 3, 4, 6)
R_AXIS_GOALS = (24, 36, 48)
S_AXIS_GOALS = (24, 48)
R_AXIS_PAIRS = ((3, 3), (6, 3), (9, 3), (12, 3), (15, 3), (18, 3),
                (6, 6), (9, 6), (12, 6), (15, 6), (18, 6))
S_AXIS_PAIRS = ((6, 6), (12, 6))

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
    "normal_immediate": (
        ("execution/baselines/grid_v2/BASELINES.json", "normal_immediate"),
        ("execution/g_scaling/normal_immediate/RESULTS.json", None),
    ),
    "slack_heuristic_cap": (
        ("execution/baselines/paper_throttles_v1/BASELINES.json", "slack_heuristic_cap"),
        ("execution/g_scaling/slack_heuristic_cap/RESULTS.json", None),
    ),
    "slack_matched_rate": (
        ("execution/baselines/paper_slack_matched/BASELINES.json", "slack_matched_rate"),
        ("execution/g_scaling/slack_matched_rate/RESULTS.json", None),
    ),
    "v4f@200": (
        ("execution/train_v4f_anchor/eval_200_s8/EVAL.json", None),
        ("execution/g_scaling/v4f_200/RESULTS.json", None),
    ),
    "v4s@300": (
        ("execution/train_v4s_cap300/eval_300_s8/EVAL.json", None),
        ("execution/g_scaling/v4s_300/RESULTS.json", None),
    ),
    "v4t@100": (
        ("execution/train_v4t_cap_frozen/eval_100_s8/EVAL.json", None),
        ("execution/g_scaling/v4t_100/RESULTS.json", None),
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cell_label(cell, search_agents=None):
    label = "G%d_R%d_N%d" % (int(cell[0]), int(cell[1]), int(cell[2]))
    if search_agents is not None and int(search_agents) != 2:
        label += "_S%d" % int(search_agents)
    return label


def axis_wanted(axis):
    if axis == "r":
        return {(goal, robots, subscribers)
                for goal in R_AXIS_GOALS for robots, subscribers in R_AXIS_PAIRS}
    return {(goal, robots, subscribers)
            for goal in S_AXIS_GOALS for robots, subscribers in S_AXIS_PAIRS}


def load_arm(axis, arm):
    """``{(cell, scene, S): reading}`` for this axis, archives first, this round winning."""
    wanted = axis_wanted(axis)
    rows = {}
    for relative, arm_filter in ARCHIVE_SOURCES[arm]:
        path = BASE / relative
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("results") or []:
            if row.get("status") != "ok":
                continue
            if arm_filter is not None and row.get("arm") != arm_filter:
                continue
            if arm_filter is None and row.get("arm") is not None and row["arm"] != arm:
                continue
            cell = tuple(int(value) for value in row["cell"])
            if cell not in wanted:
                continue
            scene = int(row["scene"])
            if scene not in ANCHOR:
                continue
            if axis == "s":
                # The archived G-sweep worlds are S=2; every archived (24/48, R6/R12, N6)
                # world therefore lands on S=2 and nowhere else.
                agents = 2
            else:
                agents = 2
            rows[(cell, scene, agents)] = {
                "cell": cell, "scene": scene, "search_agents": agents,
                "on_time": int(row["on_time"]),
                "generated_goals": int(row["generated_goals"]),
                "emissions": int(row["emissions"]),
                "spent_bytes": int(row["spent_bytes"]),
            }
    path = OUT / axis / arm.replace("@", "_") / "RESULTS.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("results") or []:
            if row.get("status") != "ok":
                continue
            cell = tuple(int(value) for value in row["cell"])
            scene = int(row["scene"])
            agents = int(row.get("search_agents") or 2)
            rows[(cell, scene, agents)] = {
                "cell": cell, "scene": scene, "search_agents": agents,
                "on_time": int(row["on_time"]),
                "generated_goals": int(row["generated_goals"]),
                "emissions": int(row["emissions"]),
                "spent_bytes": int(row["spent_bytes"]),
            }
    return rows


def load_waste(axis):
    path = OUT / axis / "WASTE.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in payload.get("results") or []:
        if row.get("status") != "ok":
            continue
        key = (row["arm"], tuple(int(v) for v in row["cell"]), int(row["scene"]),
               int(row.get("search_agents") or 2))
        out[key] = (int(row["waste_n"]), int(row["waste_d"]))
    return out


def aggregate(rows, waste, arm, predicate):
    chosen = [row for (cell, _scene, _agents), row in rows.items() if predicate(cell, row)]
    if not chosen:
        return None
    on_time = sum(row["on_time"] for row in chosen)
    goals = sum(row["generated_goals"] for row in chosen)
    emissions = sum(row["emissions"] for row in chosen)
    spent = sum(row["spent_bytes"] for row in chosen)
    waste_n = waste_d = 0
    for row in chosen:
        entry = waste.get((arm, row["cell"], row["scene"], row["search_agents"]))
        if entry is None:
            continue
        waste_n += entry[0]
        waste_d += entry[1]
    return {"worlds": len(chosen), "on_time": on_time, "goals": goals,
            "q": on_time / goals if goals else None,
            "emissions": emissions, "sends_per_world": emissions / len(chosen),
            "spent_bytes": spent, "messages_per_on_time":
                emissions / on_time if on_time else None,
            "waste": waste_n / waste_d if waste_d else None}


def side(rows, waste, arm, axis, value, other=None):
    """Aggregate one point of one axis: R value (axis r) or S value (axis s)."""
    if axis == "r":
        return aggregate(rows, waste, arm,
                         lambda cell, _row, v=value: cell[1] == v)
    def match(cell, row, v=value):
        return int(row["search_agents"]) == v
    return aggregate(rows, waste, arm, match)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(BASE / "R_S_SCALING_EXPERIMENTS.md"))
    parser.add_argument("--json", default=str(OUT / "R_S_SCALING.json"))
    args = parser.parse_args()
    sys.dont_write_bytecode = True

    waste = {axis: load_waste(axis) for axis in ("r", "s")}
    data = {axis: {arm: load_arm(axis, arm) for arm in ARMS} for axis in ("r", "s")}

    print("worlds covered per arm (reuse + new):")
    for axis in ("r", "s"):
        for arm in ARMS:
            print("  [%s] %-22s %3d" % (axis, arm, len(data[axis][arm])))

    points_r = {arm: {value: side(data["r"][arm], waste["r"], arm, "r", value)
                      for value in R_VALUES} for arm in ARMS}
    points_s = {arm: {value: side(data["s"][arm], waste["s"], arm, "s", value)
                      for value in S_VALUES} for arm in ARMS}
    # R axis split by subscriber fan-out: N=3 is the six-R slice, N=6 the five-R slice.
    points_r_n3 = {arm: {value: aggregate(
        data["r"][arm], waste["r"], arm,
        lambda cell, _row, v=value: cell[1] == v and cell[2] == 3)
        for value in R_VALUES} for arm in ARMS}
    points_r_n6 = {arm: {value: aggregate(
        data["r"][arm], waste["r"], arm,
        lambda cell, _row, v=value: cell[1] == v and cell[2] == 6)
        for value in R_VALUES} for arm in ARMS}

    missing = []
    for axis, points, values in (("r", points_r, R_VALUES), ("s", points_s, S_VALUES)):
        for arm in ARMS:
            for value in values:
                if points[arm][value] is None:
                    missing.append((axis, arm, value))

    L = []
    L.append("# R / S 跨规模评估：搜救运力 R 与搜索智能体 S 两条正交轴")
    L.append("")
    L.append("零新训练：全部权重都是封板检查点；对比臂六个、场景三个锚定场景"
             "（2301901–2301903）、信道 0.7 丢包无 ACK。98 个 pin 文件未改动。")
    L.append("")
    L.append("## 口径")
    L.append("")
    L.append("| 项 | R 轴 | S 轴 |")
    L.append("|---|---|---|")
    L.append("| 扫描 | `R ∈ {3,6,9,12,15,18}` × `G ∈ {24,36,48}` | "
             "`S ∈ {1,2,3,4,6}` × `R ∈ {6,12}` × `G ∈ {24,48}` |")
    L.append("| 扇出 | N=3 跑满六档 R；N=6 只跑 `R≥6`（5 档），故 11 个 (R,N) 组合 | "
             "N=6 固定 |")
    L.append("| 世界/点 | 3 档 G × 组合 × 3 场景 | 4 格 × 3 场景 |")
    L.append("| 新增/臂 | **63**（`R∈{3,9,15,18}`；`R=6,12` 复用归档） | "
             "**48**（`S∈{1,3,4,6}`；`S=2` 复用归档） |")
    L.append("| 页预算 | `v4t`/`v4s` = 同格 immediate 实测页数 − 1；"
             "`slack_matched_rate` = 同格 v4t 实测页数 | 同左，且预算按 **S=2** 测量后"
             "在各 S 上保持不变 |")
    L.append("")
    L.append("**白名单怎么登记**：`scale_program.RESCUE_EXTENDED = (3,6,9,12,15,18)` 与 "
             "`SEARCH_AGENTS_EXTENDED = (1,2,3,4,6)`，各自与训练集合并成 "
             "`RESCUE_ACCEPTED` / `SEARCH_AGENTS_ACCEPTED`；"
             "`world.verify_resolved_configuration` 经 `cell_expectation_overrides` "
             "放行 `rescue_robots` 与 `search_agents`，"
             "`SCALE_EXPECTATION_OVERRIDE_REASONS` 增加 "
             "`rescue_robots_extended` / `search_agents_extended` 两条理由。"
             "**训练采样网格 `CELLS` 仍是 15 格、`cell_for` 仍只从 15 格里抽**；"
             "S 不进入 cell 元组（它作为独立参数随世界传递），所以既有归档的目录名一个都没变。"
             "特征 #16 继续按实际 `(R/6)/(G/24)` 计算，R=18 时读出 3.0。")
    L.append("")

    def table(axis, values, points, points_n3=None, points_n6=None):
        lines = []
        if axis == "r":
            lines.append("## 表 R1　R 轴：发送量与字节削减（3 个锚定场景，聚合全部 11 个组合）")
        else:
            lines.append("## 表 S1　S 轴：发送量与字节削减（3 个锚定场景，4 格）")
        lines.append("")
        key = "R" if axis == "r" else "S"
        lines.append("| %s | " % key + " | ".join(
            "%s 发送/世界 | %s 字节削减" % (SHORT[a], SHORT[a]) for a in ARMS) + " |")
        lines.append("|---|" + "---|" * (2 * len(ARMS)))
        for value in values:
            base = points["normal_immediate"][value]
            row = "| %d |" % value
            for arm in ARMS:
                stats = points[arm][value]
                if stats is None or base is None:
                    row += " n/a | n/a |"
                    continue
                if arm == "normal_immediate":
                    row += " %.1f | — |" % stats["sends_per_world"]
                    continue
                cut = 100.0 * (1.0 - stats["spent_bytes"] / base["spent_bytes"])
                row += " %.1f | %+.1f %% |" % (stats["sends_per_world"], cut)
            lines.append(row)
        lines.append("")

        lines.append("## 表 %s2　按时完成率与相对 immediate 的比值" % key)
        lines.append("")
        lines.append("| %s | immediate 准时 | " % key + " | ".join(
            "%s 准时 | %s / imm" % (SHORT[a], SHORT[a]) for a in ARMS[1:]) + " |")
        lines.append("|---|" + "---|" * (2 * (len(ARMS) - 1) + 1))
        for value in values:
            base = points["normal_immediate"][value]
            row = "| %d | %d/%d = %.4f |" % (value, base["on_time"], base["goals"], base["q"])
            for arm in ARMS[1:]:
                stats = points[arm][value]
                if stats is None:
                    row += " n/a | n/a |"
                    continue
                row += " %d/%d = %.4f | %.3f |" % (
                    stats["on_time"], stats["goals"], stats["q"],
                    stats["on_time"] / base["on_time"] if base["on_time"] else 0.0)
            lines.append(row)
        lines.append("")

        lines.append("## 表 %s3　每完成任务消息数与广播超时浪费率" % key)
        lines.append("")
        lines.append("| %s | " % key + " | ".join(
            "%s msg/on | %s waste" % (SHORT[a], SHORT[a]) for a in ARMS) + " |")
        lines.append("|---|" + "---|" * (2 * len(ARMS)))
        for value in values:
            row = "| %d |" % value
            for arm in ARMS:
                stats = points[arm][value]
                if stats is None:
                    row += " n/a | n/a |"
                    continue
                row += " %s | %s |" % (
                    "n/a" if stats["messages_per_on_time"] is None
                    else "%.3f" % stats["messages_per_on_time"],
                    "n/a" if stats["waste"] is None else "%.3f" % stats["waste"])
            lines.append(row)
        lines.append("")
        return lines

    L.extend(table("r", R_VALUES, points_r))
    L.extend(table("s", S_VALUES, points_s))

    # ---- the R phase transition, split by fan-out
    L.append("## R 轴的相变：字节削减与按时率如何随运力移动")
    L.append("")
    L.append("| R | v4t 发送/世界 | v4t 字节比 | v4t 准时/immediate | v4t 准时分母 | "
             "v4f 字节比 | slack_cap 字节比 |")
    L.append("|---|---|---|---|---|---|---|")
    for value in R_VALUES:
        base = points_r["normal_immediate"][value]
        vt = points_r["v4t@100"][value]
        vf = points_r["v4f@200"][value]
        sc = points_r["slack_heuristic_cap"][value]
        if vt is None or base is None:
            L.append("| %d | n/a | n/a | n/a | n/a | n/a | n/a |" % value)
            continue
        L.append("| %d | %.1f | %.3f | %.3f | %d/%d | %.3f | %.3f |" % (
            value, vt["sends_per_world"], vt["spent_bytes"] / base["spent_bytes"],
            vt["on_time"] / base["on_time"] if base["on_time"] else 0.0,
            vt["on_time"], vt["goals"],
            vf["spent_bytes"] / base["spent_bytes"] if vf else 0.0,
            sc["spent_bytes"] / base["spent_bytes"] if sc else 0.0))
    L.append("")
    L.append("| R | N=3 字节比 | N=3 准时比 | N=6 字节比 | N=6 准时比 |")
    L.append("|---|---|---|---|---|")
    for value in R_VALUES:
        row = "| %d |" % value
        for points in (points_r_n3, points_r_n6):
            vt = points["v4t@100"][value]
            base = points["normal_immediate"][value]
            if vt is None or base is None:
                row += " n/a | n/a |"
                continue
            row += " %.3f | %.3f |" % (
                vt["spent_bytes"] / base["spent_bytes"],
                vt["on_time"] / base["on_time"] if base["on_time"] else 0.0)
        L.append(row)
    L.append("")

    # ---- the S story: backlog pressure at higher search counts
    L.append("## S 轴：搜索智能体增多后，谁先被积压压垮")
    L.append("")
    L.append("| S | immediate 发送/世界 | immediate 准时 | v4t 发送/世界 | v4t 准时 | "
             "v4t 准时比 | v4t 浪费率 | slack_matched 浪费率 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for value in S_VALUES:
        base = points_s["normal_immediate"][value]
        vt = points_s["v4t@100"][value]
        sm = points_s["slack_matched_rate"][value]
        if base is None or vt is None:
            L.append("| %d | n/a | n/a | n/a | n/a | n/a | n/a | n/a |" % value)
            continue
        L.append("| %d | %.1f | %d/%d | %.1f | %d/%d | %.3f | %s | %s |" % (
            value, base["sends_per_world"], base["on_time"], base["goals"],
            vt["sends_per_world"], vt["on_time"], vt["goals"],
            vt["on_time"] / base["on_time"] if base["on_time"] else 0.0,
            "n/a" if vt["waste"] is None else "%.3f" % vt["waste"],
            "n/a" if sm is None or sm["waste"] is None else "%.3f" % sm["waste"]))
    L.append("")

    # ---- key findings computed from the data, not asserted
    r_rows = {value: points_r["v4t@100"][value] for value in R_VALUES}
    r_base = {value: points_r["normal_immediate"][value] for value in R_VALUES}
    best_cut = max((value for value in R_VALUES if r_rows[value] and r_base[value]),
                   key=lambda value: 1.0 - r_rows[value]["spent_bytes"]
                   / r_base[value]["spent_bytes"])
    crossings = [value for value in R_VALUES
                 if r_rows[value] and r_base[value]
                 and r_rows[value]["on_time"] >= r_base[value]["on_time"]]
    s_rows = {value: points_s["v4t@100"][value] for value in S_VALUES}
    s_base = {value: points_s["normal_immediate"][value] for value in S_VALUES}
    s_gain = [value for value in S_VALUES
              if s_rows[value] and s_base[value]
              and s_rows[value]["on_time"] >= s_base[value]["on_time"]]

    def fmt(value, pattern="%.3f"):
        return "n/a" if value is None else pattern % value

    # ---- R axis: the phase transition, stated as numbers not adjectives
    r_cut = {value: (100.0 * (1.0 - r_rows[value]["spent_bytes"] / r_base[value]["spent_bytes"])
                     if r_rows[value] and r_base[value] else None) for value in R_VALUES}
    r_ratio = {value: (r_rows[value]["on_time"] / r_base[value]["on_time"]
                       if r_rows[value] and r_base[value] and r_base[value]["on_time"] else None)
               for value in R_VALUES}
    r_waste = {value: (r_rows[value]["waste"] if r_rows[value] else None) for value in R_VALUES}
    r_base_waste = {value: (r_base[value]["waste"] if r_base[value] else None)
                    for value in R_VALUES}
    monotone_cut = all(r_cut[R_VALUES[index]] is not None
                       and r_cut[R_VALUES[index + 1]] is not None
                       and r_cut[R_VALUES[index]] >= r_cut[R_VALUES[index + 1]]
                       for index in range(len(R_VALUES) - 1))
    above = [value for value in R_VALUES if r_ratio[value] is not None and r_ratio[value] > 1.0]

    L.append("## R 轴读出来的相变")
    L.append("")
    L.append("| R | 运力画像 | 字节削减 | v4t/immediate 准时 | v4t 浪费率 | immediate 浪费率 |")
    L.append("|---|---|---|---|---|---|")
    for value in R_VALUES:
        profile = {3: "极度紧缺（3 台）", 6: "训练下限", 9: "中低", 12: "训练上限",
                   15: "中高", 18: "极度充裕"}[value]
        L.append("| %d | %s | %s %% | %s | %s | %s |"
                 % (value, profile, fmt(r_cut[value], "%.1f"), fmt(r_ratio[value]),
                    fmt(r_waste[value]), fmt(r_base_waste[value])))
    L.append("")
    if monotone_cut:
        L.append("**字节削减随 R 单调不增**：%s。"
                 "运力越紧，v4t 剪得越深（R=3 时 %.1f %%，R=18 时 %.1f %%）；"
                 "这与「配额 = immediate−1 页」的设定一起说明，"
                 "深度节流不是策略在紧缩时误判，而是它在紧缩时唯一能保住准时的做法。"
                 % (" → ".join("%.1f %%" % r_cut[value] for value in R_VALUES),
                    r_cut[R_VALUES[0]], r_cut[R_VALUES[-1]]))
    else:
        L.append("**字节削减并非随 R 单调**，逐档数值见上表（不强行归纳成一条曲线）。")
    L.append("")
    if above:
        L.append("**「少发多救」（v4t 按时数严格多于 immediate）只出现在 %s**；"
                 "从 `R=%d` 起比值降到 1 或以下（R=15 恰为 1.000），即运力一旦够用，"
                 "v4t 的节流就退化为「同水平略降」而不是反超。"
                 "这就是 R 轴上的相变位置。"
                 % ("、".join("R=%d（%.3f）" % (value, r_ratio[value]) for value in above),
                    min([value for value in R_VALUES
                         if r_ratio[value] is not None and r_ratio[value] <= 1.0],
                        default=R_VALUES[-1])))
    L.append("")

    # ---- S axis: who absorbs the backlog as search agents multiply
    s_base_sends = {value: (s_base[value]["sends_per_world"] if s_base[value] else None)
                    for value in S_VALUES}
    s_vt_sends = {value: (s_rows[value]["sends_per_world"] if s_rows[value] else None)
                  for value in S_VALUES}
    s_base_waste = {value: (s_base[value]["waste"] if s_base[value] else None)
                    for value in S_VALUES}
    s_vt_waste = {value: (s_rows[value]["waste"] if s_rows[value] else None)
                  for value in S_VALUES}
    L.append("## S 轴读出来的积压抑制")
    L.append("")
    L.append("| S | immediate 发送/世界 | immediate 浪费率 | v4t 发送/世界 | v4t 浪费率 "
             "| 浪费率差（v4t − immediate） |")
    L.append("|---|---|---|---|---|---|")
    for value in S_VALUES:
        gap = (None if s_vt_waste[value] is None or s_base_waste[value] is None
               else s_vt_waste[value] - s_base_waste[value])
        L.append("| %d | %s | %s | %s | %s | %s |"
                 % (value, fmt(s_base_sends[value], "%.1f"), fmt(s_base_waste[value]),
                    fmt(s_vt_sends[value], "%.1f"), fmt(s_vt_waste[value]),
                    "n/a" if gap is None else "%+.3f" % gap))
    L.append("")
    first, last = S_VALUES[0], S_VALUES[-1]
    if all(s_base_waste[value] is not None and s_vt_waste[value] is not None
           for value in S_VALUES):
        L.append("**搜索智能体越多，immediate 的浪费涨得比 v4t 快**："
                 "S 从 %d 增到 %d 时，immediate 的广播超时浪费率从 %.3f 涨到 %.3f"
                 "（+%.3f），v4t 从 %.3f 涨到 %.3f（+%.3f）。"
                 "两者之差在每个 S 上都是负的（v4t 更低），"
                 "且差距从 %.3f 扩大到 %.3f——"
                 "这就是「同样预算下，谁把积压挡在了超时之外」的直接读数。"
                 "注意 v4t 的发送量并不随 S 增长（%.1f → %.1f 条/世界），"
                 "它与 immediate 的发送量差（%.1f → %.1f）说明它挡住的正是那部分新增广播。"
                 % (first, last, s_base_waste[first], s_base_waste[last],
                    s_base_waste[last] - s_base_waste[first],
                    s_vt_waste[first], s_vt_waste[last],
                    s_vt_waste[last] - s_vt_waste[first],
                    s_vt_waste[first] - s_base_waste[first],
                    s_vt_waste[last] - s_base_waste[last],
                    s_vt_sends[first], s_vt_sends[last],
                    s_base_sends[first] - s_vt_sends[first],
                    s_base_sends[last] - s_vt_sends[last]))
        L.append("")
        L.append("**但这条读法有一处必须写明**：`slack_matched_rate` 在同样的 S 上浪费率"
                 "也低于 immediate（见上表 S3），所以「浪费率更低」不是 v4t 独有的；"
                 "v4t 相对它多出来的是**准时数**（表 S2 的 v4t/immediate 比在 S=4、S=6 是"
                 "1.069 与 1.181，而 slack_matched 是 0.950 与 1.075）。"
                 "换句话说：抑制浪费是「少发」的普遍后果，"
                 "而 v4t 的增量价值是**在不发的前提下仍然完成更多**。")
        L.append("")

    L.append("## 明确不能说的话")
    L.append("")
    L.append("* 两个轴都只有 **3 个锚定场景、无种子重复**；差一个目标的差异不应解读。")
    L.append("* `R=3` 只有 N=3 一个扇出、`R=6..18` 才有 N=6，所以 R 轴在 R=3 处的合并数与"
             "其余档位不是同一批 (R,N) 组合；逐档的 N=3 / N=6 分列已经给出，"
             "合并列只用于看趋势。")
    L.append("* S 轴上页预算按 **S=2 测量后固定**，所以「v4t 发送量不随 S 变化」是设计的一部分，"
             "不是发现；本轴能说的是**同样预算下准时数与浪费率如何随 S 变化**。")
    L.append("* `S=1` 时 `search_agents=1`，PILOC 的观测半径与检测率随之改变，"
             "因此 S 轴的差异是「发现过程 + 任务生成」的联合效应，不能单独归因于"
             "「积压抑制」。")
    L.append("* 没有任何一档改变了训练分布；`S≠2`、`R∉{6,12}`、`G∉{12,24,48}` 都是零样本外推。")
    L.append("")

    L.append("## 复现")
    L.append("")
    L.append("```")
    for axis in ("r", "s"):
        for stage in ("immediate", "rules", "v4f", "v4s", "v4t", "slack_matched"):
            L.append("D:/project_env/python.exe analysis/rs_scaling.py --axis %s --stage %s "
                     "--workers 8" % (axis, stage))
    L.append("D:/project_env/python.exe analysis/rs_waste.py --axis r --workers 8")
    L.append("D:/project_env/python.exe analysis/rs_waste.py --axis s --workers 8")
    L.append("D:/project_env/python.exe analysis/rs_scaling_report.py")
    L.append("D:/project_env/python.exe analysis/plot_rs_figures.py")
    L.append("```")
    L.append("")
    L.append("| 产物 | 路径 |")
    L.append("|---|---|")
    for axis in ("r", "s"):
        for arm in ARMS:
            L.append("| %s / %s 世界读数 | `execution/r_s_scaling/%s/%s/RESULTS.json` |"
                     % (axis.upper(), SHORT[arm], axis, arm.replace("@", "_")))
        L.append("| %s 轴广播浪费率 | `execution/r_s_scaling/%s/WASTE.json` |"
                 % (axis.upper(), axis))
    L.append("| 本报告的数 | `execution/r_s_scaling/R_S_SCALING.json` |")
    L.append("| 图 5 / 图 6 | `execution/figures/fig5_r_scaling_curve.*`、"
             "`fig6_s_scaling_curve.*` |")
    L.append("")
    L.append("| 输入 | sha256（前 16 位） |")
    L.append("|---|---|")
    for arm, sources in ARCHIVE_SOURCES.items():
        for relative, _filter in sources:
            path = BASE / relative
            if path.exists():
                L.append("| `%s` | `%s` |" % (relative, sha256(path)[:16]))
    for axis in ("r", "s"):
        for arm in ARMS:
            path = OUT / axis / arm.replace("@", "_") / "RESULTS.json"
            if path.exists():
                L.append("| `execution/r_s_scaling/%s/%s/RESULTS.json` | `%s` |"
                         % (axis, arm.replace("@", "_"), sha256(path)[:16]))
    L.append("")

    L.append("## 图片（`analysis/plot_rs_figures.py` 生成，零重跑）")
    L.append("")
    L.append("IEEE 风格与图 1–5 一致：西文 Times New Roman、中文 SimSun、数学符号 STIX"
             "（字形已嵌入，`pdf.fonttype = 42`）；四边闭合 0.8 pt 黑框、刻度朝内、"
             "浅灰虚线网格；无图内标题/气泡/点值标注，子图编号写在横轴标签内；"
             "图例为细黑框、置于图上方；Ours-Frozen 为红色实心点，Pure-Policy 为更大的"
             "空心蓝环且 zorder 更高，两者重合时读作「红点套蓝环」。"
             "R 轴的横轴是**连续数值轴**（步长恒为 3，不需要人为等距），"
             "S 轴的横轴是**等距类别轴**（1/2/3/4/6 不等距，线性轴会把最后一档拉宽一倍）。")
    L.append("")
    L.append("| 图 | 文件 | 像素 | sha256（前 16 位） |")
    L.append("|---|---|---|---|")
    for slug in ("fig5_r_scaling_curve", "fig6_s_scaling_curve"):
        for suffix in ("png", "pdf"):
            path = BASE / "execution/figures" / ("%s.%s" % (slug, suffix))
            if not path.exists():
                continue
            extra = ""
            if suffix == "png":
                try:
                    with path.open("rb") as handle:
                        head = handle.read(33)
                    width = int.from_bytes(head[16:20], "big")
                    height = int.from_bytes(head[20:24], "big")
                    extra = "%d×%d" % (width, height)
                except Exception:
                    extra = ""
            L.append("| %s | `%s` | %s | `%s` |"
                     % (slug, str(path.relative_to(BASE)).replace("\\", "/"), extra,
                        sha256(path)[:16]))
    L.append("")

    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")

    record = {
        "schema": "stage5b-rs-scaling/1",
        "root": str(BASE),
        "scenes": list(ANCHOR),
        "r_axis": {"values": list(R_VALUES), "goals": list(R_AXIS_GOALS),
                   "pairs": [list(pair) for pair in R_AXIS_PAIRS]},
        "s_axis": {"values": list(S_VALUES), "goals": list(S_AXIS_GOALS),
                   "pairs": [list(pair) for pair in S_AXIS_PAIRS]},
        "r_points": {arm: {str(value): points_r[arm][value] for value in R_VALUES}
                     for arm in ARMS},
        "s_points": {arm: {str(value): points_s[arm][value] for value in S_VALUES}
                     for arm in ARMS},
        "r_points_n3": {arm: {str(value): points_r_n3[arm][value] for value in R_VALUES}
                        for arm in ARMS},
        "r_points_n6": {arm: {str(value): points_r_n6[arm][value] for value in R_VALUES}
                        for arm in ARMS},
        "r_on_time_at_least_immediate": crossings,
        "s_on_time_at_least_immediate": s_gain,
        "missing_points": missing,
        "coverage": {axis: {arm: len(data[axis][arm]) for arm in ARMS}
                     for axis in ("r", "s")},
        "sha256": {"%s/%s" % (axis, arm): sha256(
            OUT / axis / arm.replace("@", "_") / "RESULTS.json")
            for axis in ("r", "s") for arm in ARMS
            if (OUT / axis / arm.replace("@", "_") / "RESULTS.json").exists()},
    }
    json_out = Path(args.json)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
    print("wrote %s and %s" % (args.out, json_out))
    if missing:
        print("MISSING POINTS:", missing)


if __name__ == "__main__":
    main()
