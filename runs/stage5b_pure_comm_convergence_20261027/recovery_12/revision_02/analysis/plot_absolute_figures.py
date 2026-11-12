"""Absolute-value figures for the G sweep: counts and bytes, never a ratio to a baseline.

The published fig1 answers "how much less than Immediate", which is a relative reading. These two
figures answer "how much, in absolute units", and they are deliberately built so no axis carries a
ratio to another arm:

* fig13 -- totals per goal count over the 25 worlds of that point: on-time tasks delivered,
  bytes spent (MB), broadcast messages emitted, and the number of broadcast targets that were
  never finished on time;
* fig14 -- the two per-task costs, which are the only absolute quantities that stay comparable
  across G: bytes per on-time task and messages per on-time task.

Reading rule stated in the manifest: the totals in fig13 rise with G mechanically, because each
point is 25 worlds of G goals each; they measure throughput and spend, not efficiency. Efficiency
across G is only readable in fig14 (cost per delivered task) and in the registered verdict.

Offline: reads ``execution/g_scaling/G_SCALING.json`` and ``WASTE.json`` only.
Style is inherited from ``plot_paper_figures.py`` (Times New Roman + SimSun + STIX, closed
spines, inward ticks, no titles, categorical G axis).

    D:/project_env/python.exe analysis/plot_absolute_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "analysis"))
sys.dont_write_bytecode = True

import plot_paper_figures as base  # noqa: E402

GOALS = base.GOALS
ORDER = base.ORDER
STYLE = base.STYLE
LABEL = base.LABEL
WASTE = BASE / "execution/g_scaling/WASTE.json"
ANCHOR = base.ANCHOR
ALL_SCENES = base.ALL_SCENES
FULL_SCENE_PAIRS = base.FULL_SCENE_PAIRS


def scenes_for(pair):
    return ALL_SCENES if tuple(pair) in FULL_SCENE_PAIRS else ANCHOR


def load_absolute():
    g_scaling = base.load_json(base.G_SCALING)
    waste_payload = base.load_json(WASTE)
    rows = [row for row in waste_payload["results"] if row.get("status") == "ok"]
    waste_counts = {}
    for arm in ORDER:
        for goal in GOALS:
            chosen = [row for row in rows
                      if row["arm"] == arm and row["cell"][0] == goal
                      and row["scene"] in scenes_for((row["cell"][1], row["cell"][2]))]
            waste_counts[(arm, goal)] = (sum(r["waste_n"] for r in chosen),
                                        sum(r["waste_d"] for r in chosen))
    return g_scaling, waste_counts


def absolute(arm_series):
    """One arm's absolute series, from the report's own aggregate block."""
    return arm_series


def figure_totals(g_scaling: dict, waste_counts: dict) -> list:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))

    # (a) on-time tasks delivered -- a count, not a rate
    ax = axes[0][0]
    x = base.goal_ticks(ax)
    for arm in ORDER:
        base.plot_arm(ax, x, base.series(g_scaling, arm, "on_time"), arm)
    # Zero-based on purpose: these are counts, and an absolute figure whose axis starts at
    # 200 would read as a bigger difference than it is. The G* marker stays because the
    # crossing itself is only legible against it; the ratio view of the same crossing is fig2.
    ax.axvline(base.GOALS.index(base.G_STAR), color="#7F7F7F", lw=0.9, ls="--", zorder=1)
    ax.text(base.GOALS.index(base.G_STAR) + 0.2, 910, "𝐺*=36", color="#404040",
            fontsize=8.5, ha="left", va="top")
    ax.set_ylim(0, 950)
    ax.set_xlabel("(a) 任务规模 𝐺")
    ax.set_ylabel("按时完成任务数（个）")

    # (b) bytes spent, in MB -- the absolute cost the paper's ratio is hiding
    ax = axes[0][1]
    x = base.goal_ticks(ax)
    for arm in ORDER:
        ax.plot(x, [value / 1e6 for value in
                    base.series(g_scaling, arm, "spent_bytes")],
                **{key: STYLE[arm][key] for key in
                   ("color", "ls", "marker", "lw", "ms", "zorder")},
                markerfacecolor=STYLE[arm]["mfc"], markeredgecolor=STYLE[arm]["color"],
                markeredgewidth=STYLE[arm]["mew"])
    ax.set_ylim(0.0, 5.0)
    ax.set_xlabel("(b) 任务规模 𝐺")
    ax.set_ylabel("通信字节总量（MB）")

    # (c) broadcast messages emitted
    ax = axes[1][0]
    x = base.goal_ticks(ax)
    for arm in ORDER:
        ax.plot(x, [value / 1e3 for value in base.series(g_scaling, arm, "emissions")],
                **{key: STYLE[arm][key] for key in
                   ("color", "ls", "marker", "lw", "ms", "zorder")},
                markerfacecolor=STYLE[arm]["mfc"], markeredgecolor=STYLE[arm]["color"],
                markeredgewidth=STYLE[arm]["mew"])
    ax.set_ylim(0.0, 2.0)
    ax.set_xlabel("(c) 任务规模 𝐺")
    ax.set_ylabel("广播消息数（千条）")

    # (d) broadcast targets that were never finished on time -- a count of wasted targets
    ax = axes[1][1]
    x = base.goal_ticks(ax)
    for arm in ORDER:
        values = [waste_counts[(arm, goal)][0] for goal in GOALS]
        ax.plot(x, values, **{key: STYLE[arm][key] for key in
                              ("color", "ls", "marker", "lw", "ms", "zorder")},
                markerfacecolor=STYLE[arm]["mfc"], markeredgecolor=STYLE[arm]["color"],
                markeredgewidth=STYLE[arm]["mew"])
    ax.set_ylim(0, 1180)
    ax.set_xlabel("(d) 任务规模 𝐺")
    ax.set_ylabel("广播后未按时完成目标数（个）")

    base.place_legend(fig, [base.arm_handle(arm) for arm in ORDER], ncol=3)
    base.tighten(fig)
    return base.save(fig, "fig13_absolute_totals")


def figure_per_task(g_scaling: dict, waste_counts: dict) -> list:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    # (a) bytes per delivered task: absolute cost, comparable across G
    ax = axes[0]
    x = base.goal_ticks(ax)
    for arm in ORDER:
        spent = base.series(g_scaling, arm, "spent_bytes")
        on = base.series(g_scaling, arm, "on_time")
        ax.plot(x, [s / o if o else np.nan for s, o in zip(spent, on)],
                **{key: STYLE[arm][key] for key in
                   ("color", "ls", "marker", "lw", "ms", "zorder")},
                markerfacecolor=STYLE[arm]["mfc"], markeredgecolor=STYLE[arm]["color"],
                markeredgewidth=STYLE[arm]["mew"])
    ax.set_ylim(2200, 6600)
    ax.set_xlabel("(a) 任务规模 𝐺")
    ax.set_ylabel("每按时完成任务字节数（B/任务）")

    # (b) messages per delivered task: absolute count, comparable across G
    ax = axes[1]
    x = base.goal_ticks(ax)
    ax.axhline(1.0, color=base.PARITY_BLACK, lw=0.8, zorder=1)
    for arm in ORDER:
        base.plot_arm(ax, x, base.series(g_scaling, arm, "messages_per_on_time"), arm)
    ax.set_ylim(0.9, 2.6)
    ax.set_xlabel("(b) 任务规模 𝐺")
    ax.set_ylabel("每按时完成任务消息数（条/任务）")

    base.place_legend(fig, [base.arm_handle(arm) for arm in ORDER], ncol=3)
    base.tighten(fig)
    return base.save(fig, "fig14_absolute_per_task")


ABS_FIGURES = (
    ("fig13_absolute_totals", "图 6　G 扩展的绝对量：按时任务数、字节总量、广播数、未完成目标数"),
    ("fig14_absolute_per_task", "图 7　绝对单位成本：每按时任务字节数与消息数"),
)
ABS_DATA = (None, None)  # filled by main() so manifest_text can embed table 6
ABS_START = "<!-- ABSOLUTE-FIGURE-MANIFEST:START -->"
ABS_END = "<!-- ABSOLUTE-FIGURE-MANIFEST:END -->"


def absolute_table(g_scaling: dict, waste_counts: dict) -> str:
    """The numbers behind fig13/fig14, as one table: absolute units only, no ratios."""
    short = {"normal_immediate": "Immediate", "slack_heuristic_cap": "Slack-Cap",
             "slack_matched_rate": "Slack-Matched", "v4f@200": "Pure-Policy",
             "v4s@300": "Ours-Joint", "v4t@100": "Ours-Frozen"}
    lines = ["| 臂 | 指标 | " + " | ".join(str(goal) for goal in GOALS) + " |",
             "|---|---|" + "---|" * len(GOALS)]
    for arm in ORDER:
        on = base.series(g_scaling, arm, "on_time")
        spent = [value / 1e6 for value in base.series(g_scaling, arm, "spent_bytes")]
        sends = [value / 1e3 for value in base.series(g_scaling, arm, "emissions")]
        waste = [waste_counts[(arm, goal)][0] for goal in GOALS]
        bpt = [(s / o if o else float("nan")) for s, o in
               zip(base.series(g_scaling, arm, "spent_bytes"), on)]
        msg = base.series(g_scaling, arm, "messages_per_on_time")
        name = short[arm]
        lines.append("| %s | 按时任务（个） | %s |" % (name, " | ".join(str(v) for v in on)))
        lines.append("| | 字节（MB） | %s |" % " | ".join("%.2f" % v for v in spent))
        lines.append("| | 广播消息（千条） | %s |" % " | ".join("%.2f" % v for v in sends))
        lines.append("| | 未按时目标（个） | %s |" % " | ".join(str(v) for v in waste))
        lines.append("| | B/按时任务 | %s |" % " | ".join(
            "n/a" if v != v else "%.0f" % v for v in bpt))
        lines.append("| | 条/按时任务 | %s |" % " | ".join(
            "n/a" if v is None else "%.3f" % v for v in msg))
    return chr(10).join(lines)


def manifest_text(sources) -> str:
    lines = [ABS_START,
             "## 绝对量图（`analysis/plot_absolute_figures.py` 生成，零重跑）", "",
             "**这两张图里没有任何\"相对 immediate\"的比值**：纵轴分别是按时任务个数、"
             "字节总量（MB）、广播消息条数、未按时完成的目标个数，以及每按时任务的字节数与"
             "消息数。图 13 的四个总量随 G 上升是构造性的（每档就是 25 个 G 目标的世界），"
             "它衡量吞吐与开销，不衡量效率；跨 G 的效率只能读图 14 与注册判据。", "",
             "### 表 6　绝对量明细（与图 13/14 同一批数；25 世界/点）", ""]
    lines.append(absolute_table(*ABS_DATA))
    lines += ["",
             "| 图 | 文件 | 像素 | sha256（前 16 位） |", "|---|---|---|---|"]
    for slug, title in ABS_FIGURES:
        png = base.OUT / (slug + ".png")
        pdf = base.OUT / (slug + ".pdf")
        if not png.exists():
            continue
        width, height = base.png_size(png)
        lines.append("| %s | `%s`（300 dpi）<br>`%s`（矢量） | %d × %d | `%s` |"
                     % (title, png.relative_to(base.BASE).as_posix(),
                        pdf.relative_to(base.BASE).as_posix(), width, height,
                        base.sha256(png)[:16]))
    lines += ["", "读图注意：", "",
              "* 图 13(b)(c) 的 Immediate 与 Slack-Cap 几乎重合——Slack-Cap 只去掉每 episode "
              "最后 1 页（字节削减 0.9 %–6.8 %），两条线在这个尺度上分不开；"
              "它们的差别只在逐格判定里可读。",
              "* 图 13(d) 的未完成目标数直接取自 `WASTE.json` 的 `waste_n` 求和，"
              "分母（被广播的不同目标数）在表 3 的浪费率里给出。",
              "* 图 14 的纵轴是**绝对**单位成本（B/任务、条/任务），不是比值。"
              "十个档位的跨度：v4t 2 548–3 161 B/任务、1.019–1.136 条/任务；"
              "v4s 2 548–3 700 B/任务、1.019–1.348 条/任务；"
              "Immediate 到 G=72 升到 6 184 B/任务、2.47 条/任务。"
              "**两条训练臂的 3.2 kB 上限不是 v4s 的**：v4s 在 G=72 是 3 700 B。", "",
              "输入文件（只读）：", "", "| 输入 | sha256（前 16 位） |", "|---|---|"]
    for path in sources:
        lines.append("| `%s` | `%s` |"
                     % (path.relative_to(base.BASE).as_posix(), base.sha256(path)[:16]))
    lines += ["", "复现：", "", "```",
              "D:/project_env/python.exe analysis/plot_absolute_figures.py", "```", ABS_END]
    return chr(10).join(lines) + chr(10)


def append_manifest(sources) -> None:
    doc = base.REPORT.read_text(encoding="utf-8")
    text = manifest_text(sources)
    if ABS_START in doc:
        head, rest = doc.split(ABS_START, 1)
        _old, tail = rest.split(ABS_END, 1)
        doc = head.rstrip() + chr(10) * 2 + text + tail.lstrip(chr(10)).lstrip()
    else:
        doc = doc.rstrip() + chr(10) * 2 + "---" + chr(10) * 2 + text
    base.REPORT.write_text(doc, encoding="utf-8", newline=chr(10))


def main():
    base.require_fonts()
    base.configure()
    print("fonts: Times New Roman + SimSun + STIXGeneral (math)")
    g_scaling, waste_counts = load_absolute()
    for slug, builder in (("fig13_absolute_totals", figure_totals),
                          ("fig14_absolute_per_task", figure_per_task)):
        paths = builder(g_scaling, waste_counts)
        print("wrote", ", ".join(path.name for path in paths))
    global ABS_DATA
    ABS_DATA = (g_scaling, waste_counts)
    append_manifest([base.G_SCALING, WASTE])
    print("manifest appended to", base.REPORT)


if __name__ == "__main__":
    main()
