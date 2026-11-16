"""The paper's figures with **one learned model**: v4f@200 (cap=None), drawn as Ours.

The revision's audit figures (fig1-fig10 of the earlier rounds) still exist and still show the
internal arms; this script overwrites the *paper-facing* filenames with the four-arm version:

    Immediate (baseline) | Slack-Cap | Slack-Matched | Ours (v4f@200)

Everything is read from artifacts on disk (``G_SCALING.json`` cells, ``COMM_EFFICIENCY.json``,
``WASTE.json``); no world is opened.

Style contract is the revision's (see ``plot_paper_figures.py``): Times New Roman + SimSun +
STIXGeneral with per-glyph fallback (mixed Chinese labels carry no mathtext), closed spines,
inward ticks, dashed light grid, no in-figure titles, legend above the axes, 300 dpi PNG plus
vector PDF with embedded fonts.

    D:/project_env/python.exe analysis/plot_paper_ours.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
G_SCALING = BASE / "execution/g_scaling/G_SCALING.json"
OUT = BASE / "execution/figures"

GOALS = (12, 18, 24, 30, 36, 42, 48, 54, 60, 72)
PAIRS = ((6, 3), (6, 6), (12, 3), (12, 6), (12, 12))
R6_PAIRS = ((6, 3), (6, 6))
R12_PAIRS = ((12, 3), (12, 6), (12, 12))
OURS = "v4f@200"

STYLE = {
    "normal_immediate": dict(color="#636363", ls="-.", marker="^", mfc="white",
                             lw=1.1, ms=4.5, mew=1.0, zorder=4),
    "slack_heuristic_cap": dict(color="#D97706", ls="--", marker="D", mfc="white",
                                lw=1.1, ms=4.2, mew=1.0, zorder=3),
    "slack_matched_rate": dict(color="#8E44AD", ls=":", marker="v", mfc="white",
                               lw=1.3, ms=4.5, mew=1.0, zorder=4),
    OURS: dict(color="#C0392B", ls="-", marker="o", mfc="#C0392B",
               lw=1.7, ms=4.8, mew=1.0, zorder=6),
}
LABEL = {"normal_immediate": "Immediate (全量广播基线)",
         "slack_heuristic_cap": "Slack-Cap (时限配额启发式)",
         "slack_matched_rate": "Slack-Matched (同预算启发式)",
         OURS: "本文方法 (Ours)"}
ORDER = ("normal_immediate", "slack_heuristic_cap", "slack_matched_rate", OURS)
VERDICT_RED = "#C0392B"


def require_fonts():
    available = {font.name for font in font_manager.fontManager.ttflist}
    missing = [name for name in ("Times New Roman", "SimSun", "STIXGeneral")
               if name not in available]
    if missing:
        raise SystemExit("missing fonts: %s" % missing)


def configure():
    plt.rcParams.update({
        "font.family": ["Times New Roman", "SimSun", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.2,
        "axes.linewidth": 0.8, "axes.edgecolor": "#000000",
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.grid": True, "grid.color": "#CCCCCC", "grid.linewidth": 0.4,
        "grid.linestyle": "--", "axes.axisbelow": True, "axes.unicode_minus": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.frameon": True, "legend.fancybox": False,
        "legend.edgecolor": "#000000", "legend.facecolor": "#FFFFFF",
    })


def load():
    return json.loads(G_SCALING.read_text(encoding="utf-8"))


def cell(data, arm, goal, pair):
    return data["cells"].get("%s|G%d_R%d_N%d" % (arm, goal, pair[0], pair[1]))


def series(data, arm, key, pairs, goals=GOALS):
    """One value per goal, summed over the pairs that ran that goal."""
    out = []
    for goal in goals:
        rows = [cell(data, arm, goal, pair) for pair in pairs]
        rows = [row for row in rows if row]
        if not rows:
            out.append(np.nan)
            continue
        if key == "sends_per_world":
            out.append(sum(row["sends_per_world"] for row in rows) / len(rows))
        elif key == "q":
            out.append(sum(row["on_time"] for row in rows) / sum(row["goals"] for row in rows))
        elif key == "on_time":
            out.append(sum(row["on_time"] for row in rows))
        elif key == "bytes_ratio":
            own = sum(row["spent_bytes"] for row in rows)
            base = sum(cell(data, "normal_immediate", goal, pair)["spent_bytes"]
                       for pair in pairs if cell(data, "normal_immediate", goal, pair))
            out.append(own / base if base else np.nan)
        elif key == "ratio":
            own = sum(row["on_time"] for row in rows)
            base = sum(cell(data, "normal_immediate", goal, pair)["on_time"]
                       for pair in pairs if cell(data, "normal_immediate", goal, pair))
            out.append(own / base if base else np.nan)
        elif key == "messages":
            own = sum(row["messages_per_on_time"] * row["on_time"] for row in rows)
            on_time = sum(row["on_time"] for row in rows)
            out.append(own / on_time if on_time else np.nan)
        elif key == "waste":
            # ``cells[..].waste`` is already the ratio (waste_n / distinct_broadcast).
            out.append(sum(row["waste"] for row in rows) / len(rows))
        else:
            raise KeyError(key)
    return out


def save(figure, name):
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / (name + ".png"))
    figure.savefig(OUT / (name + ".pdf"))
    plt.close(figure)


def fig1(data):
    x = np.arange(len(GOALS))
    figure, axes = plt.subplots(2, 2, figsize=(7.4, 5.0))
    panels = (("bytes_ratio", "(a) 字节比（相对 Immediate）", "字节 / 同格 Immediate", 1.0),
              ("ratio", "(b) 相对 Immediate 的按时数比", "比值（1.0 = 打平）", 1.0),
              ("messages", "(c) 消息 / 按时任务", "条数", None),
              ("waste", "(d) 广播超时浪费率", "已广播未按时 / 被广播", None))
    for axis, (key, title, ylabel, reference) in zip(axes.ravel(), panels):
        for arm in ("slack_heuristic_cap", "slack_matched_rate", OURS):
            axis.plot(x, series(data, arm, key, PAIRS), label=LABEL[arm], **STYLE[arm])
        if reference is not None:
            axis.axhline(reference, color="#636363", lw=0.8, ls="-.", zorder=1)
        if key == "ratio":
            axis.axhline(0.95, color=VERDICT_RED, lw=0.8, ls="--", zorder=1)
        axis.set_xticks(x)
        axis.set_xticklabels([str(goal) for goal in GOALS])
        axis.set_xlabel("任务规模 G，" + title.split(")")[0] + ")")
        axis.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    reference = Line2D([], [], color="#636363", ls="-.", lw=0.9, label="Immediate (参照线)")
    figure.legend(handles + [reference], labels + ["Immediate (参照线)"], loc="upper center",
                  ncol=4, bbox_to_anchor=(0.5, 1.04), columnspacing=1.0)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save(figure, "fig1_g_scaling_efficiency")


def fig2(data):
    x = np.arange(len(GOALS))
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    imm = series(data, "normal_immediate", "on_time", R6_PAIRS)
    ours = series(data, OURS, "on_time", R6_PAIRS)
    axes[0].plot(x, imm, label="Immediate", **STYLE["normal_immediate"])
    axes[0].plot(x, ours, label=LABEL[OURS], **STYLE[OURS])
    tail = [index for index, goal in enumerate(GOALS) if goal >= 36]
    axes[0].fill_between(x[tail], [imm[i] for i in tail], [ours[i] for i in tail],
                         color=STYLE[OURS]["color"], alpha=0.18, lw=0, zorder=1)
    axes[0].annotate("G* = 36", xy=(x[4], ours[4]), xytext=(x[2], ours[4] + 45),
                     arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"),
                     fontsize=8, color="#333333")
    axes[0].set_ylabel("R=6 的按时完成数")
    axes[0].set_title("")
    axes[0].set_xlabel("任务规模 G\n(a) R=6：G≥36 起少发多救")
    axes[0].legend(loc="upper left", frameon=False)

    ratio6 = series(data, OURS, "ratio", R6_PAIRS)
    ratio12 = series(data, OURS, "ratio", R12_PAIRS)
    bytes6 = series(data, OURS, "bytes_ratio", R6_PAIRS)
    bytes12 = series(data, OURS, "bytes_ratio", R12_PAIRS)
    red = STYLE[OURS]["color"]
    axes[1].plot(x, ratio6, color=red, ls="-", marker="o", ms=4.2, lw=1.6, label="R=6 按时比")
    axes[1].plot(x, ratio12, color=red, ls="--", marker="s", mfc="white", ms=4.2, lw=1.2,
                 label="R=12 按时比")
    axes[1].plot(x, bytes6, color="#1F618D", ls="-", marker="o", mfc="white", ms=4.0, lw=1.2,
                 label="R=6 字节比")
    axes[1].plot(x, bytes12, color="#1F618D", ls="--", marker="s", mfc="white", ms=4.0, lw=1.0,
                 label="R=12 字节比")
    axes[1].axhline(1.0, color="#636363", lw=0.8, ls=":")
    axes[1].axhline(0.95, color=VERDICT_RED, lw=0.8, ls="--")
    axes[1].set_ylim(0.25, 1.45)
    axes[1].set_ylabel("相对 Immediate")
    axes[1].set_xlabel("任务规模 G\n(b) 运力充裕时无损全发，重载时自主节流")
    axes[1].legend(loc="lower left", ncol=2, frameon=False)
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels([str(goal) for goal in GOALS])
    figure.tight_layout()
    save(figure, "fig2_r6_phase_transition")


def fig3(data):
    """Per-world sends and on-time counts at three (R, N) pairs, four goals each."""
    pairs = ((6, 6), (12, 6), (12, 12))
    goals = (36, 48, 60, 72)
    arms = (("normal_immediate", "Immediate"), ("slack_heuristic_cap", "Slack-Cap"),
            (OURS, "本文方法 (Ours)"))
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    width = 0.26
    index = np.arange(len(pairs) * len(goals))
    ticks, labels = [], []
    for position, (pair, goal) in enumerate([(p, g) for p in pairs for g in goals]):
        ticks.append(position)
        labels.append("G%d\nR%d_N%d" % (goal, pair[0], pair[1]))
    for offset, (arm, name) in enumerate(arms):
        sends, on_time = [], []
        for pair, goal in [(p, g) for p in pairs for g in goals]:
            row = cell(data, arm, goal, pair)
            sends.append(row["sends_per_world"] if row else np.nan)
            on_time.append(row["on_time"] / row["goals"] if row else np.nan)
        style = dict(STYLE.get(arm, STYLE["normal_immediate"]))
        style.pop("ls", None)
        axes[0].bar(index + (offset - 1) * width, sends, width, label=name,
                    color=style["color"], alpha=0.9 if arm == OURS else 0.55,
                    edgecolor="#000000", linewidth=0.4)
        axes[1].bar(index + (offset - 1) * width, on_time, width, label=name,
                    color=style["color"], alpha=0.9 if arm == OURS else 0.55,
                    edgecolor="#000000", linewidth=0.4)
    for axis, ylabel in zip(axes, ("发送次数 / 世界", "按时完成率")):
        axis.set_xticks(ticks)
        axis.set_xticklabels(labels, fontsize=6.6, rotation=55, ha="right")
        axis.set_ylabel(ylabel)
    axes[0].set_xlabel("(a) 每世界发送次数")
    axes[1].set_xlabel("(b) 按时完成率（按时 / 目标）")
    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(handles, names, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save(figure, "fig3_throttling_attribution")


def fig4(data):
    """Efficiency-vs-outcome scatter at the eight-scene grand total."""
    figure, axis = plt.subplots(figsize=(5.6, 3.4))
    for arm in ORDER:
        ratio = series(data, arm, "ratio", PAIRS)[-1]
        bytes_ratio = series(data, arm, "bytes_ratio", PAIRS)[-1]
        if arm == "normal_immediate":
            ratio, bytes_ratio = 1.0, 1.0
        style = dict(STYLE[arm])
        marker = style.pop("marker")
        mfc = style.pop("mfc")
        ls = style.pop("ls")
        axis.scatter(bytes_ratio, ratio, s=90 if arm != OURS else 150, marker=marker,
                     facecolors=mfc if mfc else style["color"], edgecolors=style["color"],
                     linewidths=1.0, zorder=style.get("zorder", 4), label=LABEL[arm])
        axis.annotate(LABEL[arm], (bytes_ratio, ratio), textcoords="offset points",
                      xytext=(8, 6), fontsize=7.6, color="#333333")
    axis.axhline(0.95, color=VERDICT_RED, lw=0.8, ls="--")
    axis.axvline(1.0, color="#636363", lw=0.8, ls=":")
    axis.set_xlabel("字节比（相对 Immediate，G=72 汇总）")
    axis.set_ylabel("相对 Immediate 的按时数比")
    axis.legend(loc="lower left", frameon=False)
    figure.tight_layout()
    save(figure, "fig4_efficiency_pareto_all8")


def fig5(data):
    x = np.arange(len(GOALS))
    figure, axis = plt.subplots(figsize=(5.8, 3.0))
    for arm in ORDER:
        q = series(data, arm, "q", PAIRS)
        axis.plot(x, q, label=LABEL[arm], **STYLE[arm])
    axis.set_xticks(x)
    axis.set_xticklabels([str(goal) for goal in GOALS])
    axis.set_xlabel("任务规模 G")
    axis.set_ylabel("绝对按时完成率（按时 / 目标总数）")
    axis.legend(loc="lower left", frameon=False)
    figure.tight_layout()
    save(figure, "fig5_absolute_on_time")


def main():
    sys.dont_write_bytecode = True
    require_fonts()
    configure()
    data = load()
    for builder in (fig1, fig2, fig3, fig4, fig5):
        builder(data)
        print("drew", builder.__name__)


if __name__ == "__main__":
    main()
