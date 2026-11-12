"""One table for every measured cell: the 15-cell grid, the G sweep, and the deep G x R x S grid.

Three sources, one schema. Every row is one (experiment, axis point, arm) aggregate, and every
ratio is computed against ``normal_immediate`` **in the same point and the same scenes** — never
against a neighbouring point, because a 12-goal world and a 144-goal world are different problems.

    A. 15-cell grid      execution/COMM_EFFICIENCY.json        15 cells x 8 scenes, 14 arms
    B. G sweep           execution/g_scaling/G_SCALING.json     7 new goals x 5 (R,N), 6 arms
    C. deep G x R x S    execution/g_rs_grid/G_RS_GRID.json     54 points, 6 arms

Columns (identical for all three):

    experiment   A / B / C
    axis         "N (subscribers)" for A/B, "S (search agents)" for C
    G, R, N, S   the point's own axes; S=2 wherever it was not swept
    scenes       how many DEV scenes that point ran
    arm          the method
    worlds       worlds behind the row
    on_time, goals, q
    sends_per_world, bytes
    byte_ratio   bytes / immediate bytes, same point and scenes
    msg_per_task emissions / on_time
    waste        broadcast waste ratio (None when the logs carry no broadcast)

    D:/project_env/python.exe analysis/master_table.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT_CSV = BASE / "execution/MASTER_TABLE.csv"
OUT_MD = BASE / "MASTER_TABLE.md"

ARMS_A = ("normal_immediate", "dedup_immediate", "always_hold", "periodic_k5",
          "event_maxwait", "fifo_cap_k1", "random_matched_rate", "slack_heuristic_cap",
          "slack_matched_rate", "v4f@200", "v4g@300", "v4s@300", "v4t@100", "v4u@100")
ARMS_BC = ("normal_immediate", "slack_heuristic_cap", "slack_matched_rate",
           "v4f@200", "v4s@300", "v4t@100")
SCENES_A = 8
#: ``{(experiment, arm): pooled waste}``, filled in :func:`main` before the summary is built.
POOLED = {}


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def pct(value):
    return None if value is None else 100.0 * value


def rows_from_a():
    """15 cells x 14 arms, eight scenes each, from the efficiency reading."""
    payload = load(BASE / "execution/COMM_EFFICIENCY.json")
    per_arm = payload["per_arm"]
    base = per_arm["normal_immediate"]["cells"]
    out = []
    for arm in ARMS_A:
        if arm not in per_arm:
            continue
        for label, stats in sorted(per_arm[arm]["cells"].items()):
            ref = base.get(label)
            if ref is None:
                continue
            parts = label.split("_")
            goals, robots, subscribers = (int(p[1:]) for p in parts)
            out.append({
                "experiment": "A",
                "axis": "N (subscribers)",
                "G": goals, "R": robots, "N": subscribers, "S": 2,
                "scenes": SCENES_A,
                "arm": arm,
                "worlds": stats["worlds"],
                "on_time": stats["on_time"],
                "goals": stats["goals"],
                "q": stats["q"],
                "sends_per_world": stats["emissions"] / stats["worlds"],
                "bytes": stats["spent_bytes"],
                "byte_ratio": (stats["spent_bytes"] / ref["spent_bytes"]
                               if ref["spent_bytes"] else None),
                "on_ratio": (stats["on_time"] / ref["on_time"] if ref["on_time"] else None),
                "msg_per_task": stats["messages_per_on_time"],
                "waste": stats["broadcast_waste_ratio"],
                "source": "execution/COMM_EFFICIENCY.json",
            })
    return out


def rows_from_b():
    """Seven new goal counts x five (R, N) pairs x six arms, 25 worlds per point-set."""
    payload = load(BASE / "execution/g_scaling/G_SCALING.json")
    cells = payload["cells"]
    out = []
    for key, stats in cells.items():
        arm, label = key.split("|", 1)
        if arm not in ARMS_BC:
            continue
        if int(label.split("_")[0][1:]) in (12, 24, 48):
            # Those three goal counts are experiment A's; including them here would report the
            # same cells twice under two different scene sets.
            continue
        ref = cells.get("normal_immediate|" + label)
        if ref is None:
            continue
        parts = label.split("_")
        goals, robots, subscribers = (int(p[1:]) for p in parts)
        out.append({
            "experiment": "B",
            "axis": "N (subscribers)",
            "G": goals, "R": robots, "N": subscribers, "S": 2,
            "scenes": 8 if (robots, subscribers) in ((6, 6), (12, 6)) else 3,
            "arm": arm,
            "worlds": stats["worlds"],
            "on_time": stats["on_time"],
            "goals": stats["goals"],
            "q": stats["q"],
            "sends_per_world": stats["sends_per_world"],
            "bytes": stats["spent_bytes"],
            "byte_ratio": (stats["spent_bytes"] / ref["spent_bytes"]
                           if ref["spent_bytes"] else None),
            "on_ratio": (stats["on_time"] / ref["on_time"] if ref["on_time"] else None),
            "msg_per_task": stats["messages_per_on_time"],
            "waste": stats["waste"],
            "source": "execution/g_scaling/G_SCALING.json",
        })
    return out


def rows_from_c():
    """54 deep points x six arms, three anchor scenes each."""
    payload = load(BASE / "execution/g_rs_grid/G_RS_GRID.json")
    points = payload["by_point"]
    out = []
    for key, stats in points.items():
        arm, label = key.split("|", 1)
        if arm not in ARMS_BC:
            continue
        ref = points.get("normal_immediate|" + label)
        if ref is None:
            continue
        goals, robots, agents = (int(p[1:]) for p in label.split("_"))
        out.append({
            "experiment": "C",
            "axis": "S (search agents)",
            "G": goals, "R": robots, "N": payload.get("subscribers", 6), "S": agents,
            "scenes": 3,
            "arm": arm,
            "worlds": stats["worlds"],
            "on_time": stats["on_time"],
            "goals": stats["goals"],
            "q": stats["q"],
            "sends_per_world": stats["sends_per_world"],
            "bytes": stats["spent_bytes"],
            "byte_ratio": (stats["spent_bytes"] / ref["spent_bytes"]
                           if ref["spent_bytes"] else None),
            "on_ratio": (stats["on_time"] / ref["on_time"] if ref["on_time"] else None),
            "msg_per_task": stats["messages_per_on_time"],
            "waste": stats["waste"],
            "source": "execution/g_rs_grid/G_RS_GRID.json",
        })
    return out


FIELDS = ["experiment", "axis", "G", "R", "N", "S", "scenes", "arm", "worlds",
          "on_time", "goals", "q", "on_ratio", "sends_per_world", "bytes", "byte_ratio",
          "msg_per_task", "waste", "source"]


def pooled_waste():
    """``{(experiment, arm): ratio}`` pooled over the same worlds the rows cover."""
    counts = {}

    def add(experiment, arm, rows):
        for row in rows:
            if row.get("status") != "ok":
                continue
            key = (experiment, row["arm"])
            entry = counts.setdefault(key, [0, 0])
            entry[0] += int(row["waste_n"])
            entry[1] += int(row["waste_d"])

    add("A", None, load(BASE / "execution/COMM_EFFICIENCY.json")["per_world"])
    # Re-key A's rows by their own arm field.
    counts.pop(("A", None), None)
    for row in load(BASE / "execution/COMM_EFFICIENCY.json")["per_world"]:
        entry = counts.setdefault(("A", row["arm"]), [0, 0])
        entry[0] += int(row["waste_n"])
        entry[1] += int(row["waste_d"])
    new_goals = (18, 30, 36, 42, 54, 60, 72)
    for row in load(BASE / "execution/g_scaling/WASTE.json")["results"]:
        if int(row["cell"][0]) in (12, 24, 48):
            continue
        add("B", row["arm"], [row])
    for row in load(BASE / "execution/g_rs_grid/WASTE.json")["results"]:
        add("C", row["arm"], [row])
    return {key: (n / d if d else None) for key, (n, d) in counts.items()}


def aggregate_by_arm(rows, experiment):
    """One row per (experiment, arm): the summary a reader scans first."""
    out = []
    for arm in ARMS_A if experiment == "A" else ARMS_BC:
        chosen = [r for r in rows if r["experiment"] == experiment and r["arm"] == arm]
        if not chosen:
            continue
        base = [r for r in rows if r["experiment"] == experiment
                and r["arm"] == "normal_immediate"]
        base_bytes = sum(r["bytes"] for r in base)
        base_on = sum(r["on_time"] for r in base)
        on = sum(r["on_time"] for r in chosen)
        goals = sum(r["goals"] for r in chosen)
        by = sum(r["bytes"] for r in chosen)
        em = sum(r["sends_per_world"] * r["worlds"] for r in chosen)
        worlds = sum(r["worlds"] for r in chosen)
        out.append({
            "experiment": experiment, "arm": arm, "worlds": worlds,
            "on_time": on, "goals": goals, "q": on / goals if goals else None,
            "on_ratio": on / base_on if base_on else None,
            "sends_per_world": em / worlds if worlds else None,
            "bytes": by,
            "byte_ratio": by / base_bytes if base_bytes else None,
            "msg_per_task": em / on if on else None,
            "waste": POOLED.get((experiment, arm)),
        })
    return out


def fmt(value, pattern="%.3f"):
    return "n/a" if value is None else pattern % value


def write_markdown(rows, summaries, path):
    L = []
    L.append("# 全部数据总表")
    L.append("")
    L.append("三套实验、同一套列名、同一套分母：每一行的比值都用**同一实验、同一格、同一批场景**的 "
             "`normal_immediate` 作分母，绝不跨格。机器可读版见 `execution/MASTER_TABLE.csv`。")
    L.append("")
    L.append("| 实验 | 什么在变 | 固定什么 | 点 | 场景/点 | 臂 | 世界 | 数据源 |")
    L.append("|---|---|---|---|---|---|---|---|")
    L.append("| A | N（订阅者 3/6/12） | G ∈ {12,24,48}、R ∈ {6,12} | 15 格 | 8 | 14 | %d | `execution/COMM_EFFICIENCY.json` |"
             % len([r for r in rows if r["experiment"] == "A"]))
    L.append("| B | N（3/6/12）× G（18–72 七档） | S=2 | 35 格 | 8 或 3 | 6 | %d | `execution/g_scaling/G_SCALING.json` |"
             % len([r for r in rows if r["experiment"] == "B"]))
    L.append("| C | S（搜索智能体 1/2/4）× R（6/12/18）× G（72–144） | N=6 | 54 点 | 3 | 6 | %d | `execution/g_rs_grid/G_RS_GRID.json` |"
             % len([r for r in rows if r["experiment"] == "C"]))
    L.append("")
    L.append("## 表 0　按实验 × 臂的汇总（先看这张）")
    L.append("")
    L.append("| 实验 | 臂 | 世界 | 准时/目标 | 准时率 | 相对 immediate | 发送/世界 | 字节 | 字节比 | 消息/任务 | 浪费率 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    order = {a: i for i, a in enumerate(ARMS_A)}
    for experiment in ("A", "B", "C"):
        for s in sorted([x for x in summaries if x["experiment"] == experiment],
                        key=lambda x: order.get(x["arm"], 99)):
            L.append("| %s | `%s` | %d | %d/%d | %.4f | %s | %.1f | %d | %s | %s | %s |"
                     % (experiment, s["arm"], s["worlds"], s["on_time"], s["goals"], s["q"],
                        fmt(s["on_ratio"]), s["sends_per_world"], s["bytes"],
                        fmt(s["byte_ratio"]), fmt(s["msg_per_task"]), fmt(s["waste"])))
    L.append("")
    L.append("## 表 1　逐点长表（A：15 格 × 8 场景）")
    L.append("")
    L += point_table([r for r in rows if r["experiment"] == "A"], with_axis=False)
    L.append("## 表 2　逐点长表（B：G 18–72 × 5 个 (R,N)）")
    L.append("")
    L += point_table([r for r in rows if r["experiment"] == "B"], with_axis=False)
    L.append("## 表 3　逐点长表（C：G 72–144 × R × S）")
    L.append("")
    L += point_table([r for r in rows if r["experiment"] == "C"], with_axis=False)
    L.append("## 读这一张表时必须记住的三件事")
    L.append("")
    L.append("1. **分母不同**：A 每格 8 个场景，B 的 `R=6_N=3`/`R=12_N=3`/`R=12_N=12` 只有 3 个锚定场景，"
             "C 全部 3 个锚定场景。所以**跨实验的绝对准时率不可直接比**；同一实验内、同一列才可比。")
    L.append("2. **臂的集合不同**：A 有 14 个臂（含去重、周期、事件等），B/C 是 6 个臂。"
             "A 里多出来的臂没有在 B/C 上跑，不能推断它们在深档的表现。")
    L.append("3. **`v4f` 与 `v4t` 同权重**：C 的 54 个点里 52 个两行逐位相同（页预算未用满），"
             "它们不是两次独立实验。")
    L.append("")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def point_table(rows, with_axis=True):
    L = []
    L.append("| 实验 | G | R | N | S | 场景 | 臂 | 世界 | 准时/目标 | 相对 imm | 发送/世界 | 字节 | 字节比 | 消息/任务 | 浪费率 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append("| %s | %d | %d | %d | %d | %d | `%s` | %d | %d/%d | %s | %.1f | %d | %s | %s | %s |"
                 % (r["experiment"], r["G"], r["R"], r["N"], r["S"], r["scenes"], r["arm"],
                    r["worlds"], r["on_time"], r["goals"],
                    "1.000" if r["arm"] == "normal_immediate" else fmt(r["on_ratio"]),
                    r["sends_per_world"], r["bytes"], fmt(r["byte_ratio"]),
                    fmt(r["msg_per_task"]), fmt(r["waste"])))
    L.append("")
    return L


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(OUT_CSV))
    parser.add_argument("--md", default=str(OUT_MD))
    args = parser.parse_args()
    sys.dont_write_bytecode = True

    global POOLED
    POOLED = pooled_waste()
    rows = rows_from_a() + rows_from_b() + rows_from_c()
    summaries = (aggregate_by_arm(rows, "A") + aggregate_by_arm(rows, "B")
                 + aggregate_by_arm(rows, "C"))
    for row in rows:
        row["q"] = round(row["q"], 6) if row["q"] is not None else None
        row["byte_ratio"] = round(row["byte_ratio"], 6) if row["byte_ratio"] is not None else None
        row["on_ratio"] = round(row["on_ratio"], 6) if row.get("on_ratio") is not None else None
        row["msg_per_task"] = (round(row["msg_per_task"], 6)
                               if row["msg_per_task"] is not None else None)
        row["waste"] = round(row["waste"], 6) if row["waste"] is not None else None
        row["sends_per_world"] = round(row["sends_per_world"], 4)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["experiment"], r["G"], r["R"], r["N"], r["S"],
                                               r["arm"])):
            writer.writerow(row)
    write_markdown(rows, summaries, args.md)
    counts = {e: len([r for r in rows if r["experiment"] == e]) for e in ("A", "B", "C")}
    print("rows:", counts, "total", len(rows))
    print("wrote", args.csv, "and", args.md)


if __name__ == "__main__":
    main()
