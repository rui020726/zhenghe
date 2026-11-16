"""Internal audit figures 11 and 12 (NOT in the paper figure list).

One figure for the master table: every experiment, every arm, on one canvas.

Reads only ``execution/MASTER_TABLE.csv`` -- the same file MASTER_TABLE.md is rendered from -- so
the figure and the table cannot drift apart.

Layout: two rows (on-time ratio to immediate, byte ratio to immediate) by three columns, one
column per experiment. The x axis is the goal count, categorical and equidistant, covering the
union of the three grids; a column simply has no point where its grid has no goal count.

    12/24/48      A, the 15-cell N-axis grid (14 arms, 8 scenes each)
    18..72        B, the G sweep at the five (R, N) pairs (6 arms)
    72..144       C, the deep G x R x S grid (6 arms, 3 anchor scenes)

Every ratio is divided by ``normal_immediate`` in the same experiment, the same cell and the same
scenes, so the y values are comparable across columns as "relative to that condition's own
baseline" -- not as absolute success rates, which the different scene counts would forbid.

    D:/project_env/python.exe analysis/plot_master_figure.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from analysis.plot_paper_figures import (  # noqa: E402
    configure, place_legend, require_fonts, save, tighten,
)

SOURCE = BASE / "execution/MASTER_TABLE.csv"
GOALS = (12, 18, 24, 30, 36, 42, 48, 54, 60, 72, 84, 96, 108, 120, 144)
COLUMNS = (("A", "（a）15 格网格，N = 3/6/12"),
           ("B", "（b）G 扩展，五个 (R, N)"),
           ("C", "（c）深档网格，R × S"))
VERDICT_RED = "#C0392B"
IMMEDIATE_GREY = "#636363"

ARM_STYLE = {
    "normal_immediate": dict(color=IMMEDIATE_GREY, ls="-.", marker=None,
                             label="Immediate（=1.0 基准）"),
    "slack_heuristic_cap": dict(color="#1F618D", ls="--", marker="s",
                                label="Slack-Cap（时限+配额）"),
    "slack_matched_rate": dict(color="#27AE60", ls=":", marker="^",
                               label="Slack-Matched（同字节时限）"),
    "v4f@200": dict(color="#E67E22", ls="-.", marker="D", label="Ours-Net（v4f）"),
    "v4f@200": dict(color=VERDICT_RED, ls="-", marker="o", label="本文方法 (Ours)"),
}
ARMS = tuple(ARM_STYLE)


def load():
    """``{(experiment, arm, goal): (on_ratio, byte_ratio)}`` over the whole point set."""
    sums = {}
    with SOURCE.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["experiment"], row["arm"], int(row["G"]))
            entry = sums.setdefault(key, [0, 0, 0.0, 0.0, 0])
            entry[0] += int(row["on_time"])
            entry[1] += int(row["goals"])
            entry[2] += float(row["bytes"])
            entry[3] += float(row["sends_per_world"]) * int(row["worlds"])
            entry[4] += int(row["worlds"])
    base = {}
    for (experiment, arm, goal), entry in sums.items():
        if arm == "normal_immediate":
            base[(experiment, goal)] = entry
    out = {}
    for (experiment, arm, goal), (on, goals, by, _sends, _worlds) in sums.items():
        ref = base.get((experiment, goal))
        if ref is None:
            continue
        out[(experiment, arm, goal)] = (on / ref[0] if ref[0] else np.nan,
                                        by / ref[2] if ref[2] else np.nan)
    return out


def draw(data):
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.2))
    x = np.arange(len(GOALS))
    for col, (experiment, _label) in enumerate(COLUMNS):
        present = [i for i, goal in enumerate(GOALS)
                   if (experiment, "v4f@200", goal) in data]
        for row, metric in enumerate((0, 1)):
            ax = axes[row][col]
            ax.axhline(1.0, color=IMMEDIATE_GREY, lw=1.1, ls="-.", zorder=1)
            if metric == 0:
                ax.axhline(0.95, color=VERDICT_RED, lw=0.9, ls=(0, (4, 3)), alpha=0.5,
                           zorder=1)
            for arm in ARMS:
                if arm == "normal_immediate":
                    continue
                values = [data.get((experiment, arm, GOALS[i]), (np.nan, np.nan))[metric]
                          for i in present]
                st = ARM_STYLE[arm]
                filled = arm == "v4f@200"
                ax.plot([x[i] for i in present], values, color=st["color"], ls=st["ls"],
                        marker=st["marker"], mfc=(st["color"] if filled else "white"),
                        mec=st["color"], lw=(1.8 if filled else 1.2), ms=4.2, mew=1.0,
                        zorder=(6 if filled else 4))
            ax.set_xticks([x[i] for i in present],
                          [str(GOALS[i]) for i in present])
            ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.set_xlim(x[present[0]] - 0.4, x[present[-1]] + 0.4)
            if row == 0:
                ax.set_xlabel("")
            else:
                # The subplot letter and its condition live in the axis label, never in an
                # in-figure title: one convention for every figure in this report.
                ax.set_xlabel("%s　任务数量 𝐺" % COLUMNS[col][1])
            if col == 0:
                ax.set_ylabel("相对 Immediate 的准时比" if metric == 0
                              else "相对 Immediate 的字节比")
    handles = []
    for arm in ARMS:
        st = ARM_STYLE[arm]
        handles.append(plt.Line2D([], [], color=st["color"], ls=st["ls"],
                                  marker=(st["marker"] or ""),
                                  mfc=(st["color"] if arm == "v4f@200" else "white"),
                                  mec=st["color"],
                                  lw=(1.8 if arm == "v4f@200" else 1.2), ms=4.2,
                                  label=st["label"]))
    handles.append(plt.Line2D([], [], color=VERDICT_RED, lw=0.9, ls=(0, (4, 3)),
                              alpha=0.5, label="0.95 判定线"))
    place_legend(fig, handles, ncol=4)
    tighten(fig)
    return save(fig, "fig11_master_all_experiments")


# ---------------------------------------------------------------- one axis, every G
# The faceted figure above keeps each experiment in its own column. This one puts all fifteen
# goal counts on a single x axis: colour says which arm, line style and marker say which
# experiment the point came from. The denominators stay per-point, so a line is a sequence of
# "relative to that condition's own immediate" readings, not one continuous measurement.
EXPERIMENT_STYLE = {
    "A": dict(ls="-", marker="o", label="A：15 格网格（N 轴，8 场景）"),
    "B": dict(ls="--", marker="s", label="B：G 扩展（18–72，五个 (R,N)）"),
    "C": dict(ls=":", marker="^", label="C：深档网格（72–144，R×S）"),
}


def draw_unified(data):
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.0))
    x = np.arange(len(GOALS))
    for row, metric in enumerate((0, 1)):
        ax = axes[row]
        ax.axhline(1.0, color=IMMEDIATE_GREY, lw=1.1, ls="-.", zorder=1)
        if metric == 0:
            ax.axhline(0.95, color=VERDICT_RED, lw=0.9, ls=(0, (4, 3)), alpha=0.5, zorder=1)
        for arm in ARMS:
            if arm == "normal_immediate":
                continue
            st = ARM_STYLE[arm]
            filled = arm == "v4f@200"
            for experiment, est in EXPERIMENT_STYLE.items():
                present = [i for i, goal in enumerate(GOALS)
                           if (experiment, arm, goal) in data]
                if not present:
                    continue
                values = [data[(experiment, arm, GOALS[i])][metric] for i in present]
                ax.plot([x[i] for i in present], values, color=st["color"], ls=est["ls"],
                        marker=est["marker"], mfc=(st["color"] if filled else "white"),
                        mec=st["color"], lw=(1.8 if filled else 1.1), ms=4.0, mew=1.0,
                        zorder=(6 if filled else 4))
        ax.set_xticks(x, [str(goal) for goal in GOALS])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlim(-0.4, len(GOALS) - 0.6)
        ax.set_xlabel("任务数量 𝐺（12 → 144，三套实验同轴）" if row == 1 else "")
        ax.set_ylabel("相对 Immediate 的准时比" if metric == 0
                      else "相对 Immediate 的字节比")
    handles = []
    for arm in ARMS:
        st = ARM_STYLE[arm]
        handles.append(plt.Line2D([], [], color=st["color"], ls="-",
                                  marker=(st["marker"] or ""),
                                  mfc=(st["color"] if arm == "v4f@200" else "white"),
                                  mec=st["color"],
                                  lw=(1.8 if arm == "v4f@200" else 1.2), ms=4.2,
                                  label=st["label"]))
    for experiment, est in EXPERIMENT_STYLE.items():
        handles.append(plt.Line2D([], [], color="#404040", ls=est["ls"],
                                  marker=est["marker"], mfc="white", mec="#404040",
                                  lw=1.1, ms=4.0, label=est["label"]))
    handles.append(plt.Line2D([], [], color=VERDICT_RED, lw=0.9, ls=(0, (4, 3)),
                              alpha=0.5, label="0.95 判定线"))
    place_legend(fig, handles, ncol=3)
    tighten(fig)
    return save(fig, "fig12_master_unified_g")


def main() -> None:
    sys.dont_write_bytecode = True
    require_fonts()
    configure()
    data = load()
    written = draw(data)
    written += draw_unified(data)
    print("wrote", ", ".join(path.name for path in written))


if __name__ == "__main__":
    main()
