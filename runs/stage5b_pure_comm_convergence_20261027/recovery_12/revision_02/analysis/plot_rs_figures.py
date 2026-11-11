"""Figures 5 and 6: the R and S sweeps in the same IEEE style as figures 1-5 of the G sweep.

Reads only ``execution/r_s_scaling/R_S_SCALING.json``. Nothing trains, no world is opened, and
the 98 pinned files are not touched.

Style matches ``analysis/plot_paper_figures.py`` exactly -- fonts, closed spines, inward ticks,
arm colours and markers, legend above the axes -- because the two files appear in the same
paper. Arm styling is imported rather than copied, so the red/blue pairing cannot drift.

Axis choice differs by honesty of spacing:

* R takes a continuous numeric axis. {3,6,9,12,15,18} is uniform in steps of 3, so a linear
  axis states the truth rather than manufacturing even spacing.
* S takes a categorical equidistant axis (indices labelled 1..6), because {1,2,3,4,6} is not
  uniform and a linear axis would stretch the 6 -> step twice as wide as the others.

    D:/project_env/python.exe analysis/plot_rs_figures.py
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
    LABEL, ORDER, STYLE, arm_handle, configure, place_legend, require_fonts, save,
    tighten,
)

SOURCE = BASE / "execution/r_s_scaling/R_S_SCALING.json"
VERDICT_RED = "#C0392B"
IMMEDIATE_GREY = "#636363"
PARITY_BLACK = "#000000"

R_VALUES = (3, 6, 9, 12, 15, 18)
S_VALUES = (1, 2, 3, 4, 6)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit("missing input: %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def plot_arm(ax, x, y, arm, **override):
    style = STYLE[arm]
    kwargs = dict(color=style["color"], ls=style["ls"], marker=style["marker"],
                  markerfacecolor=style["mfc"], markeredgecolor=style["color"],
                  markeredgewidth=style["mew"], lw=style["lw"], ms=style["ms"],
                  zorder=style["zorder"], label=LABEL[arm])
    kwargs.update(override)
    ax.plot(x, y, **kwargs)


def numeric_ticks(ax, values, lower, upper):
    ax.set_xlim(lower, upper)
    ax.set_xticks(list(values))
    ax.set_xticklabels([str(value) for value in values])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    return np.asarray(values, dtype=float)


def categorical_ticks(ax, values):
    x = np.arange(len(values), dtype=float)
    ax.set_xlim(-0.4, len(values) - 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in values])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    return x


def ratio_axis(ax):
    ax.axhspan(0.95, 1.35, color="#E8F5E9", alpha=0.4, zorder=0)
    ax.axhline(1.0, color=IMMEDIATE_GREY, lw=1.1, ls="-.", zorder=1)
    ax.axhline(0.95, color=VERDICT_RED, lw=1.0, ls="--", zorder=1)


def panel_values(points: dict, arm: str, field: str, values):
    out = []
    for value in values:
        entry = points[arm][str(value)]
        out.append(np.nan if entry is None else entry[field])
    return out


def byte_cut(points: dict, arm: str, values):
    base = panel_values(points, "normal_immediate", "spent_bytes", values)
    own = panel_values(points, arm, "spent_bytes", values)
    return [100.0 * (1.0 - value / ref) if ref else np.nan
            for value, ref in zip(own, base)]


def on_time_ratio(points: dict, arm: str, values):
    base = panel_values(points, "normal_immediate", "on_time", values)
    own = panel_values(points, arm, "on_time", values)
    return [value / ref if ref else np.nan for value, ref in zip(own, base)]


def figure_r(payload: dict) -> list:
    points = payload["r_points"]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    x_spec = lambda ax: numeric_ticks(ax, R_VALUES, 1.2, 19.8)

    ax = axes[0][0]
    x = x_spec(ax)
    ax.axhline(0.0, color=IMMEDIATE_GREY, lw=1.1, ls="-.", zorder=1)
    for arm in ("slack_heuristic_cap", "v4f@200", "v4s@300", "v4t@100"):
        plot_arm(ax, x, byte_cut(points, arm, R_VALUES), arm)
    ax.set_ylim(-6, 92)
    ax.set_xlabel("(a) 搜救机器人数量 𝑅（𝐺∈{24,36,48}）")
    ax.set_ylabel("字节削减率（%）")

    ax = axes[0][1]
    x = x_spec(ax)
    ratio_axis(ax)
    for arm in ("slack_heuristic_cap", "slack_matched_rate", "v4f@200", "v4s@300",
                "v4t@100"):
        plot_arm(ax, x, on_time_ratio(points, arm, R_VALUES), arm)
    # Wide enough for Slack-Matched's R=3 collapse (0.370) — a clipped line would hide the
    # single most informative point of the axis.
    ax.set_ylim(0.30, 1.95)
    ax.set_xlabel("(b) 搜救机器人数量 𝑅（𝐺∈{24,36,48}）")
    ax.set_ylabel("相对按时完成率比值")

    ax = axes[1][0]
    x = x_spec(ax)
    for arm in ORDER:
        plot_arm(ax, x, panel_values(points, arm, "messages_per_on_time", R_VALUES), arm)
    ax.set_ylim(0.9, 6.4)
    ax.set_xlabel("(c) 搜救机器人数量 𝑅（𝐺∈{24,36,48}）")
    ax.set_ylabel("每完成任务消息数（条/任务）")

    ax = axes[1][1]
    x = x_spec(ax)
    for arm in ORDER:
        plot_arm(ax, x, panel_values(points, arm, "waste", R_VALUES), arm)
    ax.set_ylim(0.0, 0.90)
    ax.set_xlabel("(d) 搜救机器人数量 𝑅（𝐺∈{24,36,48}）")
    ax.set_ylabel("广播超时浪费率")

    place_legend(fig, [arm_handle(arm) for arm in ORDER], ncol=3)
    tighten(fig)
    return save(fig, "fig5_r_scaling_curve")


def figure_s(payload: dict) -> list:
    points = payload["s_points"]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))

    ax = axes[0][0]
    x = categorical_ticks(ax, S_VALUES)
    ax.axhline(0.0, color=IMMEDIATE_GREY, lw=1.1, ls="-.", zorder=1)
    for arm in ("slack_heuristic_cap", "v4f@200", "v4s@300", "v4t@100"):
        plot_arm(ax, x, byte_cut(points, arm, S_VALUES), arm)
    ax.set_ylim(-6, 52)
    ax.set_xlabel("(a) 搜索智能体数量 𝑆（𝑁=6）")
    ax.set_ylabel("字节削减率（%）")

    ax = axes[0][1]
    x = categorical_ticks(ax, S_VALUES)
    ratio_axis(ax)
    for arm in ("slack_heuristic_cap", "slack_matched_rate", "v4f@200", "v4s@300",
                "v4t@100"):
        plot_arm(ax, x, on_time_ratio(points, arm, S_VALUES), arm)
    ax.set_ylim(0.70, 1.40)
    ax.set_xlabel("(b) 搜索智能体数量 𝑆（𝑁=6）")
    ax.set_ylabel("相对按时完成率比值")

    ax = axes[1][0]
    x = categorical_ticks(ax, S_VALUES)
    for arm in ORDER:
        plot_arm(ax, x, panel_values(points, arm, "messages_per_on_time", S_VALUES), arm)
    ax.set_ylim(0.9, 2.85)
    ax.set_xlabel("(c) 搜索智能体数量 𝑆（𝑁=6）")
    ax.set_ylabel("每完成任务消息数（条/任务）")

    ax = axes[1][1]
    x = categorical_ticks(ax, S_VALUES)
    for arm in ORDER:
        plot_arm(ax, x, panel_values(points, arm, "waste", S_VALUES), arm)
    ax.set_ylim(0.0, 0.70)
    ax.set_xlabel("(d) 搜索智能体数量 𝑆（𝑁=6）")
    ax.set_ylabel("广播超时浪费率")

    place_legend(fig, [arm_handle(arm) for arm in ORDER], ncol=3)
    tighten(fig)
    return save(fig, "fig6_s_scaling_curve")


def main() -> None:
    sys.dont_write_bytecode = True
    require_fonts()
    configure()
    print("fonts: Times New Roman + SimSun + STIXGeneral (math)")
    payload = load_json(SOURCE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for paths in (figure_r(payload), figure_s(payload)):
            print("wrote", ", ".join(path.name for path in paths))
        missed = [str(item.message) for item in caught if "missing from font" in str(item.message)]
    if missed:
        raise SystemExit("missing glyphs: %s" % missed[:3])


if __name__ == "__main__":
    main()
