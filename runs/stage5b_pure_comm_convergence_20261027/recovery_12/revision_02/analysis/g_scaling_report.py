"""Build ``G_SCALING_EXPERIMENTS.md`` from the archived cells plus the seven new goal counts.

Reads only. Archived goal counts (12/24/48) come from the sealed baseline/evaluation files; the
new ones from ``execution/g_scaling/<arm>/RESULTS.json``; the broadcast-waste reading comes from
``execution/g_scaling/WASTE.json`` (both goal-count sets). Every row aggregates over the worlds
its arm actually ran: ``R=6_N=6`` and ``R=12_N=6`` over all eight DEV scenes, the other three
pairs over the three anchor scenes.

    D:/project_env/python.exe analysis/g_waste.py --workers 8
    D:/project_env/python.exe analysis/g_scaling_report.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "execution/g_scaling"

GOALS = (12, 18, 24, 30, 36, 42, 48, 54, 60, 72)
PAIRS = ((6, 3), (6, 6), (12, 3), (12, 6), (12, 12))
R6_PAIRS = ((6, 3), (6, 6))
R12_PAIRS = ((12, 3), (12, 6), (12, 12))
FULL_SCENE_PAIRS = ((6, 6), (12, 6))
ANCHOR = (2301901, 2301902, 2301903)
ALL_SCENES = tuple(range(2301901, 2301909))
ARCHIVED_GOALS = (12, 24, 48)

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
ARCHIVED = {
    "normal_immediate": "execution/baselines/grid_v2/BASELINES.json",
    "slack_heuristic_cap": "execution/baselines/paper_throttles_v1/BASELINES.json",
    "slack_matched_rate": "execution/baselines/paper_slack_matched/BASELINES.json",
    "v4f@200": "execution/train_v4f_anchor/eval_200_s8/EVAL.json",
    "v4s@300": "execution/train_v4s_cap300/eval_300_s8/EVAL.json",
    "v4t@100": "execution/train_v4t_cap_frozen/eval_100_s8/EVAL.json",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scenes_for(pair):
    return ALL_SCENES if tuple(pair) in FULL_SCENE_PAIRS else ANCHOR


def load_archived(arm):
    path = BASE / ARCHIVED[arm]
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for row in payload.get("results") or []:
        if row.get("status") != "ok":
            continue
        if row.get("arm") is not None and row["arm"] != arm:
            continue
        cell = tuple(int(value) for value in row["cell"])
        if cell[0] not in ARCHIVED_GOALS:
            continue
        scene = int(row["scene"])
        if scene not in scenes_for((cell[1], cell[2])):
            continue
        rows[(cell, scene)] = {
            "cell": cell, "scene": scene,
            "on_time": int(row["on_time"]),
            "generated_goals": int(row["generated_goals"]),
            "emissions": int(row["emissions"]),
            "spent_bytes": int(row["spent_bytes"]),
        }
    return rows


def load_new(arm):
    path = OUT / arm.replace("@", "_") / "RESULTS.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for row in payload.get("results") or []:
        if row.get("status") != "ok":
            continue
        cell = tuple(int(value) for value in row["cell"])
        scene = int(row["scene"])
        rows[(cell, scene)] = {
            "cell": cell, "scene": scene,
            "on_time": int(row["on_time"]),
            "generated_goals": int(row["generated_goals"]),
            "emissions": int(row["emissions"]),
            "spent_bytes": int(row["spent_bytes"]),
        }
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
        key = (row["arm"], tuple(int(v) for v in row["cell"]), int(row["scene"]))
        out[key] = (int(row["waste_n"]), int(row["waste_d"]))
    return out


def aggregate(rows, waste, arm, goal, pairs=None):
    chosen = [row for (cell, _scene), row in rows.items()
              if cell[0] == goal and (pairs is None or (cell[1], cell[2]) in pairs)]
    if not chosen:
        return None
    on_time = sum(row["on_time"] for row in chosen)
    goals = sum(row["generated_goals"] for row in chosen)
    emissions = sum(row["emissions"] for row in chosen)
    spent = sum(row["spent_bytes"] for row in chosen)
    waste_n = waste_d = 0
    for row in chosen:
        entry = waste.get((arm, row["cell"], row["scene"]))
        if entry is None:
            continue
        waste_n += entry[0]
        waste_d += entry[1]
    return {
        "worlds": len(chosen), "on_time": on_time, "goals": goals,
        "q": on_time / goals if goals else None,
        "emissions": emissions, "sends_per_world": emissions / len(chosen),
        "spent_bytes": spent, "bytes_per_world": spent / len(chosen),
        "messages_per_on_time": emissions / on_time if on_time else None,
        "waste": waste_n / waste_d if waste_d else None,
    }


def need_for(base):
    """``ceil(0.95 * immediate on_time)`` in integer arithmetic."""
    return (95 * base["on_time"] + 99) // 100


def verdict(learned, base):
    need = need_for(base)
    return learned["on_time"] >= need and learned["spent_bytes"] < base["spent_bytes"], need


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(BASE / "G_SCALING_EXPERIMENTS.md"))
    parser.add_argument("--json", default=str(OUT / "G_SCALING.json"))
    args = parser.parse_args()
    sys.dont_write_bytecode = True

    waste = load_waste()
    data = {}
    for arm in ARMS:
        rows = load_archived(arm)
        rows.update(load_new(arm))
        data[arm] = rows
    per_goal = {arm: {goal: aggregate(data[arm], waste, arm, goal) for goal in GOALS}
                for arm in ARMS}
    per_cell = {(arm, goal, pair): aggregate(data[arm], waste, arm, goal, (pair,))
                for arm in ARMS for goal in GOALS for pair in PAIRS}

    coverage = {arm: sum(1 for goal in GOALS if per_goal[arm][goal]) for arm in ARMS}
    print("worlds per goal:")
    for arm in ARMS:
        print("  %-22s %s" % (arm, " ".join(
            "%3d" % (per_goal[arm][goal]["worlds"] if per_goal[arm][goal] else 0)
            for goal in GOALS)))
    if any(count < len(GOALS) for count in coverage.values()):
        print("WARNING: incomplete arms:",
              {arm: count for arm, count in coverage.items() if count < len(GOALS)})

    L = []
    L.append("# G 跨规模评估：10 档任务数量下的通信—完成曲线")
    L.append("")
    L.append("零新训练：全部权重都是封板检查点，`G = 12 / 24 / 48` 复用既有归档，"
             "只补跑 `G = 18/30/36/42/54/60/72`。98 个 pin 文件未改动。")
    L.append("")
    L.append("## 口径（先读，再看表）")
    L.append("")
    L.append("| 项 | 取值 |")
    L.append("|---|---|")
    L.append("| G 档位 | " + "、".join(str(g) for g in GOALS) + "（步长 6） |")
    L.append("| (R, N) | (6,3)、(6,6)、(12,3)、(12,6)、(12,12)，N ≤ R |")
    L.append("| 场景数 | `R=6_N=6`、`R=12_N=6` 跑满 8 个 DEV 场景；"
             "`R=6_N=3`、`R=12_N=3`、`R=12_N=12` 跑 3 个锚定场景 → **25 世界/臂/档位** |")
    L.append("| 信道 | 投递率 0.7、无 ACK、`policy_owned_timing`（传输层不代发） |")
    L.append("| 页预算 | `v4t`、`v4s` 用同格 `normal_immediate` 实测页数 − 1；"
             "`slack_matched_rate` 用同格 `v4t` 实测页数 |")
    L.append("| 判定 | 逐格：`on_time ≥ ceil(0.95 × immediate)` 且 `bytes < immediate` |")
    L.append("| 复用 | G=12/24/48 的六个臂全部来自归档；新档位的六个臂本轮只读评估 |")
    L.append("")
    L.append("**新档位怎么被放行**：`scale_program.GOALS_EXTENDED` 登记十个档位，"
             "`GOALS_ACCEPTED` 是其与训练集 `GOALS` 的并集；"
             "`world.verify_resolved_configuration` 按 `GOALS_ACCEPTED` 校验，"
             "`SCALE_EXPECTATION_OVERRIDE_REASONS` 增加 `num_goals_extended` 说明理由；"
             "训练网格 `CELLS` 仍是 15 格、`cell_for` 只从那 15 格里抽，"
             "所以扩展档位**不可能**进入训练采样。"
             "特征层按**实际 G** 动态算 #2 `sent_count/G`、#14 `backlog/G`、"
             "#16 `(R/6)/(G/24)`（`scale_features.capacity_ratio`），"
             "G=72 的世界进网络时是比例量而不是原始计数。")
    L.append("")

    L.append("## 表 1　发送量与字节削减率（25 世界/点；% 相对同档位 immediate）")
    L.append("")
    L.append("| G | " + " | ".join("%s 发送/世界 | %s 字节削减"
                                   % (SHORT[a], SHORT[a]) for a in ARMS) + " |")
    L.append("|---|" + "---|" * (2 * len(ARMS)))
    for goal in GOALS:
        base = per_goal["normal_immediate"][goal]
        row = "| %d |" % goal
        for arm in ARMS:
            stats = per_goal[arm][goal]
            if stats is None or base is None:
                row += " n/a | n/a |"
                continue
            if arm == "normal_immediate":
                row += " %.1f | — |" % stats["sends_per_world"]
                continue
            cut = 100.0 * (1.0 - stats["spent_bytes"] / base["spent_bytes"])
            row += " %.1f | %+.1f %% |" % (stats["sends_per_world"], cut)
        L.append(row)
    L.append("")

    L.append("## 表 2　按时完成率（绝对）与相对 immediate 的比值")
    L.append("")
    L.append("| G | immediate 准时 | " + " | ".join(
        "%s 准时 | %s / imm" % (SHORT[a], SHORT[a]) for a in ARMS[1:]) + " |")
    L.append("|---|" + "---|" * (2 * (len(ARMS) - 1) + 1))
    for goal in GOALS:
        base = per_goal["normal_immediate"][goal]
        row = "| %d | %d/%d = %.4f |" % (goal, base["on_time"], base["goals"], base["q"])
        for arm in ARMS[1:]:
            stats = per_goal[arm][goal]
            if stats is None:
                row += " n/a | n/a |"
                continue
            row += " %d/%d = %.4f | %.3f |" % (
                stats["on_time"], stats["goals"], stats["q"],
                stats["on_time"] / base["on_time"] if base["on_time"] else 0.0)
        L.append(row)
    L.append("")

    L.append("## 表 3　每完成任务消息数与广播超时浪费率")
    L.append("")
    L.append("| G | " + " | ".join("%s msg/on | %s waste" % (SHORT[a], SHORT[a])
                                   for a in ARMS) + " |")
    L.append("|---|" + "---|" * (2 * len(ARMS)))
    for goal in GOALS:
        row = "| %d |" % goal
        for arm in ARMS:
            stats = per_goal[arm][goal]
            if stats is None:
                row += " n/a | n/a |"
                continue
            row += " %s | %s |" % (
                "n/a" if stats["messages_per_on_time"] is None
                else "%.3f" % stats["messages_per_on_time"],
                "n/a" if stats["waste"] is None else "%.3f" % stats["waste"])
        L.append(row)
    L.append("")

    L.append("## 表 4　逐 (G, R, N) 判定：v4t 是否通过注册判据")
    L.append("")
    L.append("| G | R=6 N=3 | R=6 N=6 | R=12 N=3 | R=12 N=6 | R=12 N=12 | 通过数 |")
    L.append("|---|---|---|---|---|---|---|")
    pass_by_goal = {}
    for goal in GOALS:
        row = "| %d |" % goal
        count = 0
        for pair in PAIRS:
            learned = per_cell[("v4t@100", goal, pair)]
            base = per_cell[("normal_immediate", goal, pair)]
            if learned is None or base is None:
                row += " n/a |"
                continue
            passed, need = verdict(learned, base)
            count += int(passed)
            row += " %s %d/%d · %.2fx |" % ("**过**" if passed else "不过",
                                           learned["on_time"], need,
                                           learned["spent_bytes"] / base["spent_bytes"])
        pass_by_goal[goal] = count
        row += " %d/5 |" % count
        L.append(row)
    L.append("")

    L.append("## 表 5　R=6（运力紧缺）与 R=12（运力充裕）分离：v4t 相对 immediate")
    L.append("")
    L.append("| G | R=6 准时 imm | R=6 准时 v4t | R=6 比值 | R=6 字节比 "
             "| R=12 准时 imm | R=12 准时 v4t | R=12 比值 | R=12 字节比 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for goal in GOALS:
        row = "| %d |" % goal
        for group in (R6_PAIRS, R12_PAIRS):
            base = aggregate(data["normal_immediate"], waste, "normal_immediate", goal, group)
            vt = aggregate(data["v4t@100"], waste, "v4t@100", goal, group)
            if base is None or vt is None:
                row += " n/a | n/a | n/a | n/a |"
                continue
            row += " %d/%d | %d/%d | %.3f | %.3f |" % (
                base["on_time"], base["goals"], vt["on_time"], vt["goals"],
                vt["on_time"] / base["on_time"] if base["on_time"] else 0.0,
                vt["spent_bytes"] / base["spent_bytes"])
        L.append(row)
    L.append("")

    r6 = {}
    for goal in GOALS:
        base = aggregate(data["normal_immediate"], waste, "normal_immediate", goal, R6_PAIRS)
        vt = aggregate(data["v4t@100"], waste, "v4t@100", goal, R6_PAIRS)
        r6[goal] = (base["on_time"], vt["on_time"])
    crossings = [goal for goal in GOALS if r6[goal][1] >= r6[goal][0]]
    gstar = None
    for index, goal in enumerate(GOALS):
        if all(r6[later][1] >= r6[later][0] for later in GOALS[index:]):
            gstar = goal
            break

    L.append("## 拐点与达标区间")
    L.append("")
    L.append("### R=6（紧缺运力：R=6_N=3 与 R=6_N=6，13 世界/点）")
    L.append("")
    L.append("| G | immediate 准时 | v4t 准时 | 差 | 少发多救？ |")
    L.append("|---|---|---|---|---|")
    for goal in GOALS:
        imm_on, vt_on = r6[goal]
        L.append("| %d | %d | %d | %+d | %s |"
                 % (goal, imm_on, vt_on, vt_on - imm_on,
                    "是" if vt_on >= imm_on else "否"))
    L.append("")
    if gstar is None:
        L.append("**R=6 未出现持续反超**：没有任何一档使 v4t 的按时数从此一直 ≥ immediate。")
    else:
        L.append("**G\\* = %d**：从这一档起，v4t 在 R=6 的按时数不再低于 immediate，"
                 "并且其后每一档都保持。单档反超出现在 %s，只有从 G\\* 起才持续。"
                 % (gstar, "、".join("G=%d" % g for g in crossings) if crossings else "无"))
    L.append("")
    L.append("### R=12（运力充裕：R=12_N=3 / N=6 / N=12，12 世界/点）")
    L.append("")
    L.append("| G | 通过格数（3 格） | v4t/immediate 准时比 | 备注 |")
    L.append("|---|---|---|---|")
    stable = []
    for goal in GOALS:
        count = 0
        for pair in R12_PAIRS:
            learned = per_cell[("v4t@100", goal, pair)]
            base = per_cell[("normal_immediate", goal, pair)]
            if learned is None or base is None:
                continue
            passed, _need = verdict(learned, base)
            count += int(passed)
        base = aggregate(data["normal_immediate"], waste, "normal_immediate", goal, R12_PAIRS)
        vt = aggregate(data["v4t@100"], waste, "v4t@100", goal, R12_PAIRS)
        ratio = vt["on_time"] / base["on_time"] if base["on_time"] else 0.0
        if count == len(R12_PAIRS):
            stable.append(goal)
        note = ("G=12 每格只有 12 个目标，一个目标就是 8.3 个百分点"
                if goal == 12 else "")
        L.append("| %d | %d/3 | %.3f | %s |" % (goal, count, ratio, note))
    L.append("")
    if stable:
        L.append("**R=12 三格全过的档位：G ∈ {%s}**（%d/10 档）。"
                 % ("、".join(str(g) for g in stable), len(stable)))
    else:
        L.append("**R=12 没有任何一档让三格全部通过。**")
    L.append("")

    L.append("## 明确不能说的话")
    L.append("")
    L.append("* 新档位（18/30/36/42/54/60/72）**没有参与任何训练或调参**，"
             "是这个 checkpoint 的零样本外推，不是训练分布的延续。")
    L.append("* 每档 25 个世界（两个组合 8 场景 + 三个组合 3 场景），**没有种子重复**；"
             "差一个目标的差异不应解读。")
    L.append("* `R=6_N=3`、`R=12_N=3`、`R=12_N=12` 只有 3 个锚定场景，"
             "其合并数不能与两个 8 场景组合直接比。")
    L.append("* 页预算按本轮规定统一为 `immediate − 1`；G=12/24/48 归档里 v4t 用的是它"
             "**训练得到的**页预算（多数格等于 `immediate − 1`，个别格 ±1–3 页）。"
             "表 4 的归档列用归档口径，新档位列用本轮统一口径，这是一处已知的口径接缝。")
    L.append("* **G=12 的 `2/5` 与 `PAPER_EXPERIMENTS.md` 表 1 的 `9/15` 不是矛盾，是两张不同的表**："
             "前者是 5 个 (R,N) 组合 ×（2 组 8 场景 + 3 组 3 场景）= 25 个世界；"
             "后者是 15 个训练格 × 3 个锚定场景 = 45 个世界。场景集合与格数都不同，"
             "两者的通过数不可互相校验，只能各自读。")
    L.append("")

    L.append("## 复现")
    L.append("")
    L.append("```")
    L.append("D:/project_env/python.exe analysis/g_scaling.py --stage immediate --workers 8")
    L.append("D:/project_env/python.exe analysis/g_scaling.py --stage rules --workers 8")
    L.append("D:/project_env/python.exe analysis/g_scaling.py --stage v4f --workers 8")
    L.append("D:/project_env/python.exe analysis/g_scaling.py --stage v4s --workers 8")
    L.append("D:/project_env/python.exe analysis/g_scaling.py --stage v4t --workers 8")
    L.append("D:/project_env/python.exe analysis/g_scaling.py --stage slack_matched --workers 8")
    L.append("D:/project_env/python.exe analysis/g_waste.py --workers 8")
    L.append("D:/project_env/python.exe analysis/g_scaling_report.py")
    L.append("```")
    L.append("")
    L.append("| 产物 | 路径 |")
    L.append("|---|---|")
    for arm in ARMS:
        L.append("| %s 世界读数 | `execution/g_scaling/%s/RESULTS.json` |"
                 % (SHORT[arm], arm.replace("@", "_")))
    L.append("| 广播浪费率（十个档位） | `execution/g_scaling/WASTE.json` |")
    L.append("| 本报告的数 | `execution/g_scaling/G_SCALING.json` |")
    L.append("")
    L.append("| 输入 | sha256（前 16 位） |")
    L.append("|---|---|")
    for arm in ARMS:
        L.append("| `%s`（归档） | `%s` |"
                 % (ARCHIVED[arm], sha256(BASE / ARCHIVED[arm])[:16]))
    for arm in ARMS:
        path = OUT / arm.replace("@", "_") / "RESULTS.json"
        if path.exists():
            L.append("| `execution/g_scaling/%s/RESULTS.json` | `%s` |"
                     % (arm.replace("@", "_"), sha256(path)[:16]))
    L.append("")

    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")

    cells_json = {}
    for arm in ARMS:
        for goal in GOALS:
            for pair in PAIRS:
                stats = per_cell[(arm, goal, pair)]
                if stats is not None:
                    cells_json["%s|G%d_R%d_N%d" % (arm, goal, pair[0], pair[1])] = stats
    record = {
        "schema": "stage5b-g-scaling/1",
        "root": str(BASE),
        "goals": list(GOALS),
        "pairs": [list(pair) for pair in PAIRS],
        "full_scene_pairs": [list(pair) for pair in FULL_SCENE_PAIRS],
        "archive_scenes": {"anchor": list(ANCHOR), "all": list(ALL_SCENES)},
        "archived_goals": list(ARCHIVED_GOALS),
        "new_goals": [goal for goal in GOALS if goal not in ARCHIVED_GOALS],
        "aggregate": {arm: {str(goal): per_goal[arm][goal] for goal in GOALS}
                      for arm in ARMS},
        "cells": cells_json,
        "passes_by_goal": {str(goal): pass_by_goal[goal] for goal in GOALS},
        "r6_pairs": [list(pair) for pair in R6_PAIRS],
        "r6_series": {str(goal): {"immediate": r6[goal][0], "v4t": r6[goal][1]}
                      for goal in GOALS},
        "g_star_r6": gstar,
        "r12_all_pass_goals": stable,
        "missing_goal_counts": {arm: count for arm, count in coverage.items()
                                if count < len(GOALS)},
        "sha256": {arm: sha256(OUT / arm.replace("@", "_") / "RESULTS.json")
                   for arm in ARMS
                   if (OUT / arm.replace("@", "_") / "RESULTS.json").exists()},
    }
    json_out = Path(args.json)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
    print("wrote %s and %s" % (args.out, json_out))
    print("G* (R=6) =", gstar, "| R=12 all-pass goals =", stable)


if __name__ == "__main__":
    main()
