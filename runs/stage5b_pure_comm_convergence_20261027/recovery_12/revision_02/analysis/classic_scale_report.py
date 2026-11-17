"""The four classical arms on the two scale grids (G >= 48), against Ours and Immediate.

Two grids, both already run for the sealed arms:

* **G sweep** ``execution/g_scaling`` -- G in {48, 54, 60, 72} x the five (R, N) pairs;
  ``(6,6)`` and ``(12,6)`` run all eight DEV scenes, the other three pairs the three anchors
  (25 worlds per goal).  G=48 for the classical arms is read from the 120-world benchmark
  (``classic_v1``), filtered to exactly this scene rule; G=54/60/72 are the new runs.
* **deep G x R x S grid** ``execution/g_rs_grid`` -- G in {72, 84, 96, 108, 120, 144} x
  R in {6, 12, 18} x S in {1, 2, 4}, N=6, three anchor scenes (27 worlds per goal).

The classical arms keep the operating point chosen on the 15-cell benchmark **unchanged**;
only the token bucket's per-cell rate is recomputed from that cell's own immediate pages.
So this section measures *transfer*, not a re-tuned baseline.

Writes ``execution/classic/SCALE_RESULTS.json`` and appends nothing by itself; the markdown
section and the LaTeX table are produced here too.

    D:/project_env/python.exe analysis/classic_scale_report.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "analysis"))

from comm_efficiency import measure, world_key  # noqa: E402

CLASSIC = ("red_aqm", "token_bucket", "ahp_utility", "linucb_bandit")
OLD = ("normal_immediate", "slack_heuristic_cap", "slack_matched_rate", "v4f@200")
LABEL = {"normal_immediate": "Immediate", "slack_heuristic_cap": "EDF-Slack",
         "slack_matched_rate": "EDF-Matched", "v4f@200": "本文方法 (Ours)",
         "red_aqm": "RED-AQM", "token_bucket": "Token-Bucket",
         "ahp_utility": "AHP-Utility", "linucb_bandit": "LinUCB-Bandit"}
ORDER = ("normal_immediate",) + CLASSIC + ("v4f@200",)

G_GOALS = (24, 30, 36, 42, 48, 54, 60, 72)
G_PAIRS = ((6, 3), (6, 6), (12, 3), (12, 6), (12, 12))
G_FULL = ((6, 6), (12, 6))
ANCHOR3 = (2301901, 2301902, 2301903)
DEV8 = tuple(range(2301901, 2301909))
DEEP_GOALS = (72, 84, 96, 108, 120, 144)

OUT_JSON = BASE / "execution/classic/SCALE_RESULTS.json"
OUT_TEX = BASE / "execution/latex/table4_scale_classical.tex"
CELL_TAG = re.compile(r"(G\d+R\d+N\d+)(?:S(\d+))?")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads((BASE / path).read_text(encoding="utf-8"))


def scenes_for(cell):
    return DEV8 if (int(cell[1]), int(cell[2])) in G_FULL else ANCHOR3


def rows_of(path, arm=None):
    if not (BASE / path).exists():
        return []
    payload = load(path)
    return [row for row in payload.get("results") or []
            if row.get("status") == "ok" and (arm is None or row.get("arm") == arm)]


def totals(rows):
    uses = sum(int(row["emissions"]) * int(row["cell"][2]) for row in rows)
    return {"worlds": len(rows),
            "on_time": sum(int(row["on_time"]) for row in rows),
            "goals": sum(int(row["generated_goals"]) for row in rows),
            "spent_bytes": sum(int(row["spent_bytes"]) for row in rows),
            "emissions": sum(int(row["emissions"]) for row in rows),
            "cu": uses / len(rows) if rows else None}


def waste_from_logs(roots, want, arm):
    """``want(cell, scene, agents) -> bool``; sums waste counts over the matching leaves."""
    distinct = waste = 0
    for root in roots:
        logs = BASE / root / "world_logs"
        if not logs.is_dir():
            continue
        for directory in sorted(logs.iterdir()):
            if not directory.is_dir() or directory.name.startswith("_"):
                continue
            parts = directory.name.split("__")
            if len(parts) != 4 or not parts[1].startswith(arm + "_"):
                continue
            match = CELL_TAG.search(parts[1])
            if match is None:
                continue
            numbers = [int(value) for value in re.findall(r"\d+", match.group(1))]
            agents = int(match.group(2)) if match.group(2) else 2
            scene = int(parts[3])
            if not want(tuple(numbers), scene, agents):
                continue
            inner = [path for path in directory.iterdir() if path.is_dir()]
            if len(inner) != 1:
                continue
            reading = measure(inner[0])
            distinct += reading["distinct_broadcast"]
            waste += reading["waste_n"]
    return distinct, waste


def waste_from_json(path, arm, want):
    distinct = waste = 0
    for row in load(path).get("results") or []:
        if row.get("status") != "ok" or row.get("arm") != arm:
            continue
        if not want(tuple(row["cell"]), int(row["scene"]), row.get("search_agents") or 2):
            continue
        distinct += int(row["waste_d"])
        waste += int(row["waste_n"])
    return distinct, waste


def g_sweep():
    scaling = load("execution/g_scaling/G_SCALING.json")
    out = {}
    for goal in G_GOALS:
        out[goal] = {}
        for arm in ORDER:
            if arm in OLD:
                cells, uses = [], 0
                for r, n in G_PAIRS:
                    cell = scaling["cells"].get("%s|G%d_R%d_N%d" % (arm, goal, r, n))
                    if cell:
                        cells.append(cell)
                        uses += cell["emissions"] * n
                if not cells:
                    continue
                entry = {"worlds": sum(cell["worlds"] for cell in cells),
                         "on_time": sum(cell["on_time"] for cell in cells),
                         "goals": sum(cell["goals"] for cell in cells),
                         "spent_bytes": sum(cell["spent_bytes"] for cell in cells),
                         "emissions": sum(cell["emissions"] for cell in cells)}
                entry["cu"] = uses / entry["worlds"] if entry["worlds"] else None
                want = lambda cell, scene, agents, g=goal: (
                    cell[0] == g and (cell[1], cell[2]) in G_PAIRS and scene in scenes_for(cell))
                distinct, waste = waste_from_json("execution/g_scaling/WASTE.json", arm, want)
            else:
                if goal == 48:
                    rows = [row for row in rows_of("execution/baselines/classic_v1/BASELINES.json",
                                                   arm)
                            if int(row["cell"][0]) == 48
                            and (row["cell"][1], row["cell"][2]) in G_PAIRS
                            and int(row["scene"]) in scenes_for(row["cell"])]
                    roots = ["execution/baselines/classic_v1"]
                else:
                    rows = [row for row in rows_of("execution/g_scaling/%s/RESULTS.json" % arm)
                            if int(row["cell"][0]) == goal]
                    roots = ["execution/g_scaling/%s" % arm]
                if not rows:
                    continue
                entry = totals(rows)
                want = lambda cell, scene, agents, g=goal: (
                    cell[0] == g and (cell[1], cell[2]) in G_PAIRS and scene in scenes_for(cell))
                distinct, waste = waste_from_logs(roots, want, arm)
            entry["waste_d"], entry["waste_n"] = distinct, waste
            out[goal][arm] = entry
    return out


def deep_grid():
    grid = load("execution/g_rs_grid/G_RS_GRID.json")
    out = {}
    for goal in DEEP_GOALS:
        out[goal] = {}
        for arm in ORDER:
            if arm in OLD:
                entry = dict(grid["by_goal"][arm][str(goal)])
                entry = {key: entry[key] for key in
                         ("worlds", "on_time", "goals", "spent_bytes", "emissions")}
                entry["cu"] = (entry["emissions"] * 6 / entry["worlds"]
                               if entry["worlds"] else None)
                want = lambda cell, scene, agents, g=goal: cell[0] == g
                distinct, waste = waste_from_json("execution/g_rs_grid/WASTE.json", arm, want)
            else:
                rows = [row for row in rows_of("execution/g_rs_grid/%s/RESULTS.json" % arm)
                        if int(row["cell"][0]) == goal]
                if not rows:
                    continue
                entry = totals(rows)
                want = lambda cell, scene, agents, g=goal: cell[0] == g
                distinct, waste = waste_from_logs(["execution/g_rs_grid/%s" % arm], want, arm)
            entry["waste_d"], entry["waste_n"] = distinct, waste
            out[goal][arm] = entry
    return out


def readings(entry, reference):
    if not entry or not reference or not reference["on_time"]:
        return None
    return {"worlds": entry["worlds"], "on_time": entry["on_time"], "goals": entry["goals"],
            "q": entry["on_time"] / entry["goals"] if entry["goals"] else None,
            "rel_q": entry["on_time"] / reference["on_time"],
            "byte_ratio": (entry["spent_bytes"] / reference["spent_bytes"]
                           if reference["spent_bytes"] else None),
            "messages": (entry["emissions"] / entry["on_time"] if entry["on_time"] else None),
            "waste": (entry["waste_n"] / entry["waste_d"] if entry["waste_d"] else None),
            "cu": entry.get("cu")}


def main():
    grids = {"g_sweep": g_sweep(), "deep": deep_grid()}
    table = {}
    for name, data in grids.items():
        table[name] = {}
        for goal, arms in data.items():
            reference = arms.get("normal_immediate")
            table[name][goal] = {arm: readings(entry, reference)
                                 for arm, entry in arms.items()}
    payload = {"schema": "stage5b-classic-scale/1", "table": table,
               "operating_points": load("execution/classic/CHOSEN.json")["chosen"],
               "note": "classical arms keep the 15-cell operating point unchanged; only the "
                       "token-bucket rate is recomputed per cell",
               "sources": {
                   "g_scaling": sha256(BASE / "execution/g_scaling/G_SCALING.json"),
                   "g_rs_grid": sha256(BASE / "execution/g_rs_grid/G_RS_GRID.json")}}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")

    lines = []
    for name, goals, title in (("g_sweep", G_GOALS, "G 扫描（5 个 (R,N) 对，25 世界/档）"),
                               ("deep", DEEP_GOALS, "深档 G×R×S（N=6，27 世界/档）")):
        lines.append("### %s" % title)
        lines.append("")
        lines.append("| 方法 | " + " | ".join("G=%d" % goal for goal in goals) + " |")
        lines.append("|---" * (len(goals) + 1) + "|")
        for arm in ORDER:
            cells = []
            for goal in goals:
                record = (table[name].get(goal) or {}).get(arm)
                cells.append("—" if record is None else "%.3f / %.3f"
                             % (record["rel_q"], record["byte_ratio"]))
            lines.append("| %s | %s |" % (LABEL[arm], " | ".join(cells)))
        lines.append("")
        lines.append("每格为“相对按时率 / 字节比”，分母是同档同世界的 Immediate。")
        lines.append("")
    (BASE / "execution/classic/SCALE_TABLE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    section = ["<!-- scale-section:start -->", "## 9. 大规模扩展（G ≥ 48）：工作点是否迁移", ""]
    section.append("四个经典臂**保持 15 格基准上选定的工作点不变**（RED max_p、AHP 阈值、"
                   "LinUCB 阈值均冻结），只有 Token-Bucket 的速率按该格自身的 immediate 页面数"
                   "重算。因此本节读数是“迁移”，不是重新调参后的基线。")
    section.append("")
    section.append("**图的口径**：图 19 按作者决定不画 LinUCB-Bandit，只画 Immediate / RED / "
                   "Token / AHP / 本文方法；LinUCB 的逐档读数保留在本节表格与表 4 中，"
                   "“LinUCB 与本文方法打平/更高”的结论不因图而消失。")
    section.append("")
    section += lines
    section.append("### 迁移读数")
    section.append("")
    for arm in CLASSIC:
        first = (table["g_sweep"].get(48) or {}).get(arm)
        last_g = (table["g_sweep"].get(72) or {}).get(arm)
        last_d = (table["deep"].get(144) or {}).get(arm)
        if not first:
            continue
        def show(record, key):
            return "—" if not record or record.get(key) is None else "%.3f" % record[key]
        text = "- %s：字节比 G48 %s → G72 %s → G144 %s；相对按时率 %s → %s → %s。" % (
            LABEL[arm], show(first, "byte_ratio"), show(last_g, "byte_ratio"),
            show(last_d, "byte_ratio"), show(first, "rel_q"), show(last_g, "rel_q"),
            show(last_d, "rel_q"))
        section.append(text)
    ours_first = (table["g_sweep"].get(48) or {}).get("v4f@200")
    ours_g72 = (table["g_sweep"].get(72) or {}).get("v4f@200")
    ours_d144 = (table["deep"].get(144) or {}).get("v4f@200")
    if ours_first:
        def show(record, key):
            return "—" if not record or record.get(key) is None else "%.3f" % record[key]
        section.append("- 本文方法：字节比 G48 %s → G72 %s → G144 %s；相对按时率 %s → %s → %s。"
                       % (show(ours_first, "byte_ratio"), show(ours_g72, "byte_ratio"),
                          show(ours_d144, "byte_ratio"), show(ours_first, "rel_q"),
                          show(ours_g72, "rel_q"), show(ours_d144, "rel_q")))
    section.append("")
    wins = []
    for name, goals in (("g_sweep", G_GOALS), ("deep", DEEP_GOALS)):
        for goal in goals:
            row = table[name].get(goal) or {}
            ours, lin = row.get("v4f@200"), row.get("linucb_bandit")
            if ours and lin and lin["rel_q"] > ours["rel_q"] + 1e-9:
                wins.append((name, goal, lin["rel_q"] - ours["rel_q"]))
    if wins:
        deltas = [d for _n, _g, d in wins]
        byte_deltas = []
        for name, goals in (("g_sweep", G_GOALS), ("deep", DEEP_GOALS)):
            for goal in goals:
                row = table[name].get(goal) or {}
                ours, lin = row.get("v4f@200"), row.get("linucb_bandit")
                if ours and lin:
                    byte_deltas.append(lin["byte_ratio"] - ours["byte_ratio"])
        section.append("- **不利结果（必须写）**：在全部 %d 个 G≥48 档位上 LinUCB 的相对按时率都高于"
                       "本文方法（+%.3f…+%.3f），代价是字节比高 %.3f…%.3f。即大 G 下本文方法"
                       "砍得更狠、丢得更多；线性 UCB 的外推比训练在 G≤48 的原对偶策略更稳。"
                       % (len(wins), min(deltas), max(deltas),
                          min(byte_deltas), max(byte_deltas)))
        section.append("- 逐档明细：%s。"
                       % "、".join("%s G%d (+%.3f)" % (n, g, d) for n, g, d in wins))
    else:
        section.append("- 没有任何档位上 LinUCB 的相对按时率高于本文方法。")
    section.append("- 不能说：经典算法在大 G 下仍与本文方法同预算。工作点冻结后字节比会漂移，"
                   "上表逐档给出漂移后的实际字节比；比较时必须连同字节比一起读。")
    section.append("<!-- scale-section:end -->")
    doc = BASE / "CLASSIC_ALGORITHM_COMPARISON.md"
    text = doc.read_text(encoding="utf-8")
    block = "\n".join(section) + "\n"
    if "<!-- scale-section:start -->" in text:
        head, rest = text.split("<!-- scale-section:start -->", 1)
        _, tail = rest.split("<!-- scale-section:end -->", 1)
        text = head + block + tail
    else:
        text = text.rstrip() + "\n\n" + block
    doc.write_text(text, encoding="utf-8", newline="\n")
    print("updated %s §9" % doc.name)

    success = ["<!-- success-section:start -->",
               "## 10. 成功率与信道使用次数/世界（G = 24…108）", ""]
    success.append("成功率 = on_time / generated_goals（绝对口径，分母为该档全部生成目标）；"
                   "信道使用次数/世界 = Σ(发射数 × 订阅者数) / 世界数。图 20 把两段画在"
                   "**同一根 G 轴（24…108）**上：G≤72 取 G 扫描口径（25 世界/档，N 随 (R,N) 对"
                   "变化），G≥84 取深档口径（27 世界/档，N=6），72 与 84 之间以点竖线标"
                   "“口径切换”；G=72 用扫描值。下表仍按两口径分块列全读数（含深档 G=72），"
                   "跨口径的数值差属设计差异，不是模型变化。")
    success.append("")
    success.append("**图的口径**：图 20 同样不画 LinUCB-Bandit（五臂）；其成功率与信道读数"
                   "保留在下表，便于与图对照时补回第六臂。")
    success.append("")
    for key, title, pattern in (("q", "成功率（绝对）", "%.3f"),
                                ("cu", "信道使用次数/世界", "%.1f"),
                                ("waste", "广播超时浪费率", "%.3f")):
        success.append("### %s" % title)
        success.append("")
        success.append("| 方法 | " + " | ".join("G=%d" % g for g in G_GOALS)
                       + " | " + " | ".join("深G=%d" % g for g in (72, 84, 96, 108)) + " |")
        success.append("|---" * (len(G_GOALS) + 4 + 1) + "|")
        for arm in ORDER:
            cells = []
            for name, goals in (("g_sweep", G_GOALS), ("deep", (72, 84, 96, 108))):
                for goal in goals:
                    record = (table[name].get(goal) or {}).get(arm)
                    cells.append("—" if record is None or record.get(key) is None
                                 else pattern % record[key])
            success.append("| %s | %s |" % (LABEL[arm], " | ".join(cells)))
        success.append("")
    success.append("浪费率 = 被广播的不同目标中未在其时限内完成的比例（分母为被广播目标数，"
                   "不是全部生成目标）；从不发送的臂没有该读数。")
    success.append("")
    success.append("图 `fig20_success_and_channel_uses`（单轴 G=24…108 三联：成功率 / "
                   "信道使用 / 浪费率）。")
    success.append("<!-- success-section:end -->")
    text = doc.read_text(encoding="utf-8")
    block = "\n".join(success) + "\n"
    if "<!-- success-section:start -->" in text:
        head, rest = text.split("<!-- success-section:start -->", 1)
        _, tail = rest.split("<!-- success-section:end -->", 1)
        text = head + block + tail
    else:
        text = text.rstrip() + "\n\n" + block
    doc.write_text(text, encoding="utf-8", newline="\n")
    print("updated %s §10" % doc.name)

    tex = [r"% Generated by analysis/classic_scale_report.py -- do not edit by hand.",
           r"\begin{table*}[t]", r"\centering",
           r"\caption{大规模扩展（$G \geq 48$）：四个具名经典算法保持 15 格基准上的工作点不变，"
           r"仅 Token-Bucket 的速率按该格自身的 immediate 页面数重算。每格为“相对按时率 / 字节比”，"
           r"分母为同档同世界的 Immediate。上：G 扫描（25 世界/档）；下：深档 G$\times$R$\times$S"
           r"（N=6，27 世界/档）。}",
           r"\label{tab:scale_classical}"]
    for name, goals, panel in (("g_sweep", G_GOALS, "(a) G 扫描"),
                               ("deep", DEEP_GOALS, "(b) 深档 G$\\times$R$\\times$S")):
        tex += [r"\par\smallskip", r"\textbf{%s}" % panel,
                r"\begin{tabular}{@{}l" + "c" * len(goals) + "@{}}", r"\toprule",
                r"方法 & " + " & ".join("$G=%d$" % goal for goal in goals) + r" \\",
                r"\midrule"]
        for arm in ORDER:
            cells = []
            for goal in goals:
                record = (table[name].get(goal) or {}).get(arm)
                cells.append("--" if record is None else "%.3f / %.3f"
                             % (record["rel_q"], record["byte_ratio"]))
            name_tex = LABEL[arm].replace("本文方法 (Ours)", r"\textbf{本文方法 (Ours)}")
            tex.append("%s & %s \\\\" % (name_tex, " & ".join(cells)))
        tex += [r"\bottomrule", r"\end{tabular}"]
    tex += [r"\end{table*}", ""]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(tex) + "\n", encoding="utf-8", newline="\n")
    print("wrote %s" % OUT_JSON)
    print("wrote %s" % OUT_TEX)
    for name, goals in (("g_sweep", G_GOALS), ("deep", DEEP_GOALS)):
        for goal in goals:
            row = table[name].get(goal) or {}
            ours = row.get("v4f@200")
            lin = row.get("linucb_bandit")
            if ours and lin:
                print("%s G%-4d Ours %.3f/%.3f  LinUCB %.3f/%.3f"
                      % (name, goal, ours["rel_q"], ours["byte_ratio"],
                         lin["rel_q"], lin["byte_ratio"]))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
