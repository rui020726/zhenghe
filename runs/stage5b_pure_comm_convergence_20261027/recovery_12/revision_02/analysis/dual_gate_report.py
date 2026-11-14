"""Both gates, four arms plus the new uncapped v4v, from archived worlds only.

Old gate (the paper's): every cell must have ``on_time >= ceil(0.95 * immediate)`` **and**
``bytes < immediate``.
Relaxed gate (this round's proposal): the on-time clause stays per cell, and the cost clause
becomes global -- total bytes ratio < 0.85 and broadcast waste < 0.10 over the same 15 cells x
3 anchor scenes.

Every arm is read the same way: rows from its own EVAL/BASELINES file, waste re-derived from the
two logs of each world (the paper's definition), so no arm is scored on a different convention.

    D:/project_env/python.exe analysis/dual_gate_report.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "execution/relaxed_gate"
OUT_MD = OUT_DIR / "DUAL_GATE.md"
OUT_JSON = OUT_DIR / "DUAL_GATE.json"

ANCHOR = (2301901, 2301902, 2301903)
BYTES_LIMIT = 0.85
WASTE_LIMIT = 0.10
CELL_TAG = re.compile(r"(G\d+R\d+N\d+)")

ARMS = (
    ("normal_immediate", "execution/baselines/grid_v2",
     "execution/baselines/grid_v2/BASELINES.json", True),
    ("v4f@200", "execution/train_v4f_anchor/eval_200_s8",
     "execution/train_v4f_anchor/eval_200_s8/EVAL.json", False),
    ("v4v_pure", "execution/train_v4v_pure/eval_200",
     "execution/train_v4v_pure/eval_200/EVAL.json", False),
    ("v4t@100", "execution/train_v4t_cap_frozen/eval_100_s8",
     "execution/train_v4t_cap_frozen/eval_100_s8/EVAL.json", False),
    ("slack_heuristic_cap", "execution/baselines/paper_throttles_v1",
     "execution/baselines/paper_throttles_v1/BASELINES.json", True),
)
LABEL = {"normal_immediate": "Immediate", "v4f@200": "v4f@200（旧纯网络）",
         "v4v_pure": "v4v_pure（新纯网络）", "v4t@100": "v4t@100（旧网络+配额表）",
         "slack_heuristic_cap": "Slack-Cap（零训练启发式）"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def need(base):
    return (95 * int(base) + 99) // 100


def rows_of(path, arm, shared):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {}
    for row in payload.get("results") or []:
        if row.get("status") != "ok":
            continue
        if "arm" in row and row["arm"] != arm:
            continue
        cell = tuple(int(value) for value in row["cell"])
        scene = int(row["scene"])
        if scene not in ANCHOR:
            continue
        rows[(cell, scene)] = row
    return rows


def waste_by_cell(root, arm, shared):
    sys.path.insert(0, str(BASE / "analysis"))
    from comm_efficiency import measure
    logs = BASE / root / "world_logs"
    totals = {}
    for directory in sorted(logs.iterdir()):
        if not directory.is_dir():
            continue
        # Only evaluation leaves: a training run's own world_logs also sit under its tag
        # (train__T2__91002__3900), and those worlds are not part of the reported evaluation.
        if not directory.name.startswith("eval__"):
            continue
        if shared and not directory.name.startswith("eval__%s_" % arm):
            continue
        parts = directory.name.split("__")
        match = CELL_TAG.search(parts[1]) if len(parts) == 4 else None
        if match is None:
            continue
        scene = int(parts[3])
        if scene not in ANCHOR:
            continue
        inner = [path for path in directory.iterdir() if path.is_dir()]
        if len(inner) != 1:
            continue
        # World leaves spell the cell without underscores (G12R6N3); the report keys cells as
        # G12_R6_N3, so normalise instead of string-comparing the two spellings.
        parts_cell = [int(value) for value in re.findall(r"\d+", match.group(1))]
        label = "G%d_R%d_N%d" % tuple(parts_cell)
        reading = measure(inner[0])
        entry = totals.setdefault(label, [0, 0])
        entry[0] += reading["distinct_broadcast"]
        entry[1] += reading["waste_n"]
    return {label: (distinct, waste) for label, (distinct, waste) in totals.items()}


def main():
    base_rows = rows_of(ARMS[0][2], ARMS[0][0], ARMS[0][3])
    cells = sorted({cell for cell, _scene in base_rows}, key=lambda c: (c[0], c[1], c[2]))
    base = {}
    for cell in cells:
        own = [row for (c, _s), row in base_rows.items() if c == cell]
        base[cell] = {"on_time": sum(int(r["on_time"]) for r in own),
                      "bytes": sum(int(r["spent_bytes"]) for r in own),
                      "emissions": sum(int(r["emissions"]) for r in own)}

    state = {"schema": "stage5b-dual-gate/1", "cells": len(cells), "scenes": list(ANCHOR),
             "gate": {"per_cell": "on_time >= ceil(0.95 * immediate)",
                      "global": {"bytes_ratio_below": BYTES_LIMIT,
                                 "waste_below": WASTE_LIMIT}},
             "arms": {}}
    lines = ["# 双口径对照：旧严苛判据 vs 新合理判据", "",
             "15 格 × 3 锚定场景（45 世界/臂）。旧判据要求**每格** `on_time ≥ ⌈0.95×immediate⌉` "
             "且 `bytes < immediate`；新判据保留逐格准时条款，把字节条款移到全局："
             "总字节比 < %.2f 且广播浪费率 < %.2f。" % (BYTES_LIMIT, WASTE_LIMIT), "",
             "| 臂 | 旧判据通过 | 准时条款通过 | 全局字节比 | 全局浪费率 | 新判据通过 |",
             "|---|---|---|---|---|---|"]

    for arm, root, results, shared in ARMS:
        rows = rows_of(results, arm, shared)
        waste = waste_by_cell(root, arm, shared)
        per_cell = {}
        for cell in cells:
            label = "G%d_R%d_N%d" % cell
            own = [row for (c, _s), row in rows.items() if c == cell]
            if not own:
                continue
            on_time = sum(int(r["on_time"]) for r in own)
            spent = sum(int(r["spent_bytes"]) for r in own)
            emissions = sum(int(r["emissions"]) for r in own)
            distinct, useless = waste.get(label, (0, 0))
            per_cell[label] = {
                "on_time": on_time, "immediate_on_time": base[cell]["on_time"],
                "threshold": need(base[cell]["on_time"]),
                "on_time_ok": on_time >= need(base[cell]["on_time"]),
                "bytes": spent, "immediate_bytes": base[cell]["bytes"],
                "bytes_ok": spent < base[cell]["bytes"],
                "emissions": emissions,
                "waste_n": useless, "waste_d": distinct,
                "waste_ratio": (useless / distinct) if distinct else None,
            }
        total_bytes = sum(v["bytes"] for v in per_cell.values())
        total_base = sum(v["immediate_bytes"] for v in per_cell.values())
        total_distinct = sum(v["waste_d"] for v in per_cell.values())
        total_waste = sum(v["waste_n"] for v in per_cell.values())
        total_on_time = sum(v["on_time"] for v in per_cell.values())
        goals = sum(int(r["generated_goals"]) for (c, _s), r in rows.items() if c in cells)
        bytes_ratio = total_bytes / total_base
        waste_ratio = total_waste / total_distinct if total_distinct else None
        global_ok = bytes_ratio < BYTES_LIMIT and waste_ratio is not None and waste_ratio < WASTE_LIMIT
        old_pass = sum(1 for v in per_cell.values() if v["on_time_ok"] and v["bytes_ok"])
        on_time_pass = sum(1 for v in per_cell.values() if v["on_time_ok"])
        record = {
            "old_passes": old_pass, "on_time_passes": on_time_pass,
            "relaxed_passes": on_time_pass if global_ok else 0,
            "global_ok": global_ok,
            "bytes_ratio": bytes_ratio, "waste_ratio": waste_ratio,
            "on_time": total_on_time, "goals": goals,
            "on_time_ratio": total_on_time / sum(base[c]["on_time"] for c in cells),
            "bytes": total_bytes,
            "emissions": sum(v["emissions"] for v in per_cell.values()),
            "messages_per_on_time": (sum(v["emissions"] for v in per_cell.values())
                                     / total_on_time) if total_on_time else None,
            "blocked_by_bytes_only": [label for label, v in per_cell.items()
                                      if v["on_time_ok"] and not v["bytes_ok"]],
            "on_time_failures": {label: (v["on_time"], v["threshold"])
                                 for label, v in per_cell.items() if not v["on_time_ok"]},
            "per_cell": per_cell,
        }
        state["arms"][arm] = record
        lines.append("| %s | %d/15 | %d/15 | %.3f | %s | %s |" % (
            LABEL[arm], old_pass, on_time_pass, bytes_ratio,
            "n/a" if waste_ratio is None else "%.3f" % waste_ratio,
            ("%d/15" % on_time_pass) if global_ok else "0/15（全局条款不过）"))

    lines += ["", "## 旧判据下「只因为字节 == immediate」被否掉的格子", ""]
    for arm, _root, _results, _shared in ARMS:
        record = state["arms"][arm]
        if record["blocked_by_bytes_only"]:
            lines.append("- **%s**：%s" % (LABEL[arm],
                                           "、".join(record["blocked_by_bytes_only"])))
    lines += ["", "## 准时条款仍不过的格子（两套判据下都不过）", ""]
    for arm, _root, _results, _shared in ARMS:
        record = state["arms"][arm]
        if record["on_time_failures"]:
            lines.append("- **%s**：%s" % (LABEL[arm], "；".join(
                "%s %d/%d" % (label, on_time, threshold)
                for label, (on_time, threshold) in sorted(record["on_time_failures"].items()))))
    lines += ["", "## 四个臂的读数汇总（45 世界）", "",
              "| 臂 | 按时 | 相对 immediate | 发送 | 字节 | 字节比 | 每按时任务消息 | 浪费率 |",
              "|---|---|---|---|---|---|---|---|"]
    for arm, _root, _results, _shared in ARMS:
        r = state["arms"][arm]
        lines.append("| %s | %d/%d | %.3f | %d | %d | %.3f | %.3f | %s |" % (
            LABEL[arm], r["on_time"], r["goals"], r["on_time_ratio"], r["emissions"],
            r["bytes"], r["bytes_ratio"], r["messages_per_on_time"] or 0.0,
            "n/a" if r["waste_ratio"] is None else "%.3f" % r["waste_ratio"]))
    checkpoint = BASE / "execution/train_v4v_pure/checkpoints/milestone_200.pth"
    if checkpoint.exists():
        import torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_block = payload.get("state") or {}
        lambdas = state_block.get("lambda_by_cell") or {}
        caps = state_block.get("cap_by_cell") or {}
        state["v4v_mechanism"] = {"dual_mode": state_block.get("dual_mode"),
                                  "lambda_by_cell": {k: float(v) for k, v in lambdas.items()},
                                  "cap_by_cell": {k: int(v) for k, v in caps.items()}}
        lines += ["", "## v4v 训练结束时的每格价格（机制读数）", "",
                  "无配额表（`cap_by_cell` 为空 = %s），只有价格 λ；λ = 0 表示该格训练时被判定为"
                  "「准时不足、成本免费」。" % (not bool(caps)), "",
                  "| 格 | λ |", "|---|---|"]
        for label in sorted(lambdas):
            lines.append("| %s | %.3f |" % (label, float(lambdas[label])))
        zero = [label for label, value in sorted(lambdas.items()) if float(value) == 0.0]
        lines += ["", "λ = 0 的格：%s" % ("、".join(zero) if zero else "无")]
    if "v4v_pure" in state["arms"] and "v4f@200" in state["arms"]:
        new_arm = state["arms"]["v4v_pure"]
        old_arm = state["arms"]["v4f@200"]
        lines += ["", "## 读法：新对偶把策略推回了 immediate", "",
                  "本轮新增的 v4v_pure 没有配额表、只在「某格准时不足」时把价格衰减到约 0。"
                  "结果它把每条候选都发了出去：45 个锚定世界上发送 **%d** 条，与 immediate 的 "
                  "%d 条相同，字节比 **%.3f**（v4f 是 %.3f），浪费率 **%.3f**（v4f %.3f）。"
                  % (new_arm["emissions"], state["arms"]["normal_immediate"]["emissions"],
                     new_arm["bytes_ratio"], old_arm["bytes_ratio"],
                     new_arm["waste_ratio"] or 0.0, old_arm["waste_ratio"] or 0.0),
                  "",
                  "按时数确实从 %d 升到 %d（%.3f× immediate），但代价是字节回到 baseline，"
                  "而全局条款（字节比 < %.2f）因此不过——新判据下 v4v 是 0/15，v4f 是 %d/15。"
                  "逐格看更清楚：三个「痛点格」里，多发的页数买到的按时数是递减的："
                  % (old_arm["on_time"], new_arm["on_time"], new_arm["on_time_ratio"], BYTES_LIMIT,
                     old_arm["relaxed_passes"]), "",
                  "| 格 | v4f 发送 / 按时 | v4v 发送 / 按时 | 多花页数 | 多买到按时 | 是否过线 |",
                  "|---|---|---|---|---|---|"]
        for label in ("G24_R6_N3", "G48_R12_N3", "G48_R12_N6"):
            a = old_arm["per_cell"].get(label)
            b = new_arm["per_cell"].get(label)
            if not a or not b:
                continue
            lines.append("| %s | %d / %d | %d / %d | %+d | %+d | %s |" % (
                label, a["emissions"], a["on_time"], b["emissions"], b["on_time"],
                b["emissions"] - a["emissions"], b["on_time"] - a["on_time"],
                "过" if b["on_time_ok"] else "仍不过（%d/%d）" % (b["on_time"], b["threshold"])))
        lines += ["",
                  "机制读数：训练中 λ 只落在 0.06–2.44（`dual_anchor_log`），而发送/保留的 "
                  "logit 间距是 12–14；两处评估点（第 100、200 回合，各 45 世界）的贪心读数"
                  "完全相同（q 0.7190、字节 208 352），说明策略在第 100 回合前就已经到不动点。"
                  "所以「把价格关掉」不是「更聪明地砍废广播」，而是「不再砍」。"]
    lines += ["", "## 证据", "", "| 文件 | sha256（前 16 位） |", "|---|---|"]
    for arm, _root, results, _shared in ARMS:
        path = BASE / results
        if path.exists():
            lines.append("| `%s` | `%s` |" % (results, sha256(path)[:16]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines[:14]))
    print("wrote %s and %s" % (OUT_MD.name, OUT_JSON.name))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
