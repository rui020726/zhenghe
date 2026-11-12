"""Figures 7 and 8: the crossed G x R x S grid, same IEEE style as figures 1-6.

Reads only ``execution/g_rs_grid/G_RS_GRID.json``. Nothing trains and no world is opened.

Figure 7 answers the capacity axis at the deep end: for each rescue roster, the on-time ratio and
the byte ratio against immediate across the six goal counts, plus the broadcast-waste pair.
Figure 8 answers the discovery axis the same way for the three search-agent counts.

Axis choice: G is the x axis and is uniform in steps of 12 from 72, so a categorical equidistant
axis is used to keep the six labels equally spaced (the 72 -> 84 step is the same width as
120 -> 144, which is true) and to leave room for labels at the deep end.

    D:/project_env/python.exe analysis/plot_g_rs_figures.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from analysis.plot_paper_figures import (  # noqa: E402
    configure, place_legend, require_fonts, save, tighten,
)

SOURCE = BASE / "execution/g_rs_grid/G_RS_GRID.json"
VERDICT_RED = "#C0392B"
IMMEDIATE_GREY = "#636363"

GOALS = (72, 84, 96, 108, 120, 144)
ROBOTS = (6, 12, 18)
AGENTS = (1, 2, 4)
#: One colour per value of the swept axis; the three lines are the R or S levels.
LEVEL_STYLE = {
    6: dict(color="#C0392B", ls="-", marker="o", mfc="#C0392B", label="R = 6（运力紧缺）"),
    12: dict(color="#1F618D", ls="--", marker="s", mfc="#1F618D", label="R = 12（训练上限）"),
    18: dict(color="#27AE60", ls=":", marker="^", mfc="#27AE60", label="R = 18（超出训练）"),
}
AGENT_STYLE = {
    1: dict(color="#C0392B", ls="-", marker="o", mfc="#C0392B", label="S = 1（搜索少）"),
    2: dict(color="#1F618D", ls="--", marker="s", mfc="#1F618D", label="S = 2（训练值）"),
    4: dict(color="#27AE60", ls=":", marker="^", mfc="#27AE60", label="S = 4（搜索多）"),
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit("missing input: %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def goal_ticks(ax):
    x = np.arange(len(GOALS))
    ax.set_xlim(-0.4, len(GOALS) - 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(goal) for goal in GOALS])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    return x


def level_series(rows: dict, arm_field: str, level: int, field: str):
    out = []
    for goal in GOALS:
        entry = rows.get("%s|G%d_%s%d" % (arm_field[0], goal, arm_field[1], level))
        out.append(np.nan if entry is None else entry[field])
    return out


def ratio_series(rows: dict, arm_field: str, level: int):
    own = level_series(rows, (arm_field[0], arm_field[1]), level, "on_time")
    base = level_series(rows, ("normal_immediate", arm_field[1]), level, "on_time")
    return [a / b if b else np.nan for a, b in zip(own, base)]


def byte_series(rows: dict, arm_field: str, level: int):
    own = level_series(rows, (arm_field[0], arm_field[1]), level, "spent_bytes")
    base = level_series(rows, ("normal_immediate", arm_field[1]), level, "spent_bytes")
    return [a / b if b else np.nan for a, b in zip(own, base)]


def draw(rows, levels, style, arm_field, x_label, slug) -> list:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))

    ax = axes[0][0]
    x = goal_ticks(ax)
    ax.axhline(1.0, color=IMMEDIATE_GREY, lw=1.1, ls="-.", zorder=1)
    ax.axhline(0.95, color=VERDICT_RED, lw=1.0, ls="--", zorder=1)
    for level in levels:
        st = style[level]
        ax.plot(x, ratio_series(rows, ("v4t@100", arm_field[1]), level),
                color=st["color"], ls=st["ls"], marker=st["marker"],
                mfc=st["mfc"], mec=st["color"], lw=1.6, ms=4.6, mew=1.0, zorder=5,
                label=st["label"])
    ax.set_xlabel("(a) 任务数量 𝐺")
    ax.set_ylabel("Ours-Frozen / Immediate 准时比")

    ax = axes[0][1]
    x = goal_ticks(ax)
    ax.axhline(1.0, color=IMMEDIATE_GREY, lw=1.1, ls="-.", zorder=1)
    for level in levels:
        st = style[level]
        ax.plot(x, byte_series(rows, ("v4t@100", arm_field[1]), level),
                color=st["color"], ls=st["ls"], marker=st["marker"],
                mfc="white", mec=st["color"], lw=1.2, ms=4.6, mew=1.0, zorder=4,
                label=st["label"])
    ax.set_xlabel("(b) 任务数量 𝐺")
    ax.set_ylabel("Ours-Frozen / Immediate 字节比")

    ax = axes[1][0]
    x = goal_ticks(ax)
    for level in levels:
        st = style[level]
        ax.plot(x, level_series(rows, ("v4t@100", arm_field[1]), level, "waste"),
                color=st["color"], ls=st["ls"], marker=st["marker"],
                mfc=st["mfc"], mec=st["color"], lw=1.6, ms=4.6, mew=1.0, zorder=5,
                label=st["label"])
    ax.set_xlabel("(c) 任务数量 𝐺")
    ax.set_ylabel("Ours-Frozen 广播超时浪费率")

    ax = axes[1][1]
    x = goal_ticks(ax)
    for level in levels:
        st = style[level]
        ax.plot(x, level_series(rows, ("normal_immediate", arm_field[1]), level, "waste"),
                color=st["color"], ls=st["ls"], marker=st["marker"],
                mfc="white", mec=st["color"], lw=1.2, ms=4.6, mew=1.0, zorder=4,
                label=st["label"])
    ax.set_xlabel("(d) 任务数量 𝐺")
    ax.set_ylabel("Immediate 广播超时浪费率")

    place_legend(fig, [Line2D([], [], color=style[level]["color"], ls=style[level]["ls"],
                              marker=style[level]["marker"], mfc=style[level]["mfc"],
                              mec=style[level]["color"], lw=1.6, ms=4.6,
                              label=style[level]["label"]) for level in levels], ncol=3)
    tighten(fig)
    return save(fig, slug)


# ---------------------------------------------------------------- six-arm comparison figures
# Figures 9 and 10 answer the question "does the deep grid compare the methods, or only report
# ours?". Every panel puts the five non-baseline arms in one frame at the same (R, S) facet, so
# a reader can see, at one glance, whether the learned arms and the two heuristics separate.
ARM_STYLE = {
    "slack_heuristic_cap": dict(color="#1F618D", ls="--", marker="s",
                                label="Slack-Cap（时限+配额）"),
    "slack_matched_rate": dict(color="#27AE60", ls=":", marker="^",
                               label="Slack-Matched（同字节时限）"),
    "v4f@200": dict(color="#E67E22", ls="-.", marker="D", label="Ours-Net（v4f）"),
    "v4s@300": dict(color="#8E44AD", ls=(0, (5, 1, 1, 1)), marker="v", label="Ours-Cap（v4s）"),
    "v4t@100": dict(color="#C0392B", ls="-", marker="o", label="Ours-Frozen（v4t）"),
}
COMPARE_ARMS = tuple(ARM_STYLE)


def point(rows, arm, goal, robots, agents):
    return rows.get("%s|G%d_R%d_S%d" % (arm, goal, robots, agents))


def series(rows, arm, facet_value, metric, facet, agents=None, robots=None):
    out = []
    for goal in GOALS:
        if facet == "R":
            entry = point(rows, arm, goal, facet_value, agents)
        else:
            entry = point(rows, arm, goal, robots, facet_value)
        out.append(np.nan if entry is None else entry[metric])
    return out


def ratio(rows, arm, facet_value, facet, agents=None, robots=None):
    own = series(rows, arm, facet_value, "on_time", facet, agents, robots)
    base = series(rows, "normal_immediate", facet_value, "on_time", facet, agents, robots)
    return [a / b if b else np.nan for a, b in zip(own, base)]


def byte_ratio(rows, arm, facet_value, facet, agents=None, robots=None):
    own = series(rows, arm, facet_value, "spent_bytes", facet, agents, robots)
    base = series(rows, "normal_immediate", facet_value, "spent_bytes", facet, agents, robots)
    return [a / b if b else np.nan for a, b in zip(own, base)]


def draw_comparison(rows, facet, facet_values, fixed, slug, facet_cn):
    fig, axes = plt.subplots(2, len(facet_values), figsize=(7.6, 5.2))
    for col, value in enumerate(facet_values):
        base_kw = ({"agents": fixed} if facet == "R" else {"robots": fixed})
        for row, (metric, ylabel, hollow) in enumerate((
                ("ratio", "相对 Immediate 的准时比", False),
                ("byte", "相对 Immediate 的字节比", True))):
            ax = axes[row][col]
            x = goal_ticks(ax)
            ax.axhline(1.0, color=IMMEDIATE_GREY, lw=1.1, ls="-.", zorder=1)
            if metric == "ratio":
                ax.axhline(0.95, color=VERDICT_RED, lw=0.9, ls=(0, (4, 3)), alpha=0.55,
                           zorder=1)
            for arm in COMPARE_ARMS:
                st = ARM_STYLE[arm]
                values = (ratio(rows, arm, value, facet, **base_kw) if metric == "ratio"
                          else byte_ratio(rows, arm, value, facet, **base_kw))
                filled = arm == "v4t@100"
                # v4f and v4t share one set of weights and differ only by the page cap, which
                # does not bind in these cells: the two curves coincide on 52 of 54 points.
                # They are drawn at +-0.09 around the tick so the pair is visible as a pair.
                shift = {"v4f@200": 0.09, "v4t@100": -0.09}.get(arm, 0.0)
                ax.plot([x_i + shift for x_i in x], values, color=st["color"], ls=st["ls"],
                        marker=st["marker"],
                        mfc=(st["color"] if filled else "white"), mec=st["color"],
                        lw=(1.8 if filled else 1.2), ms=4.2, mew=1.0,
                        zorder=(6 if filled else 4))
            letter = "ab"[row]
            ax.set_xlabel("(%s%d) 任务数量 𝐺%s" % (letter, col + 1,
                                                  "，%s = %d" % (facet_cn, value)))
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_ylim(0, None) if metric == "byte" else ax.set_ylim(None, None)
    handles = [Line2D([], [], color=IMMEDIATE_GREY, lw=1.1, ls="-.",
                      label="Immediate（=1.0 基准）"),
               Line2D([], [], color=VERDICT_RED, lw=0.9, ls=(0, (4, 3)), alpha=0.55,
                      label="0.95 判定线")]
    for arm in COMPARE_ARMS:
        st = ARM_STYLE[arm]
        handles.append(Line2D([], [], color=st["color"], ls=st["ls"], marker=st["marker"],
                              mfc=(st["color"] if arm == "v4t@100" else "white"),
                              mec=st["color"], lw=(1.8 if arm == "v4t@100" else 1.2),
                              ms=4.2, label=st["label"]))
    place_legend(fig, handles, ncol=3)
    tighten(fig)
    return save(fig, slug)


def main() -> None:
    sys.dont_write_bytecode = True
    require_fonts()
    configure()
    print("fonts: Times New Roman + SimSun + STIXGeneral (math)")
    payload = load_json(SOURCE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        written = draw(payload["by_goal_robot"], ROBOTS, LEVEL_STYLE, ("v4t@100", "R"),
                       "搜救机器人数量", "fig7_g_rs_grid_r")
        written += draw(payload["by_goal_agent"], AGENTS, AGENT_STYLE, ("v4t@100", "S"),
                        "搜索智能体数量", "fig8_g_rs_grid_s")
        # Six arms in one frame, S fixed at the training value: the capacity axis.
        written += draw_comparison(payload["by_point"], "R", ROBOTS, 2,
                                   "fig9_g_rs_compare_r", "R")
        # Same, R fixed at the training value: the discovery axis.
        written += draw_comparison(payload["by_point"], "S", AGENTS, 12,
                                   "fig10_g_rs_compare_s", "S")
        for paths in (written[:2], written[2:4], written[4:6]):
            print("wrote", ", ".join(path.name for path in paths))
        missed = [str(item.message) for item in caught if "missing from font" in str(item.message)]
    if missed:
        raise SystemExit("missing glyphs: %s" % missed[:3])


if __name__ == "__main__":
    main()
