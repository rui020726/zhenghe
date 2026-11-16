"""Figures 15 and 16 for the extended probes, from the sweep JSON only. No world is opened.

fig15  delivery-probability sweep, 2x2: absolute on-time rate, on-time relative to immediate,
       bytes per on-time task, and broadcast waste, with the delivery probability on a
       categorical axis (0.3 .. 1.0).
fig16  the budget spectrum: for each of the two cells, the four selectors' trajectory over
       the nine ceilings, plotted as byte ratio (x) against on-time relative to immediate (y),
       with the ceiling printed beside each point and the non-dominated set marked.

Style contract is the one every other figure in this revision uses (see plot_paper_figures.py):
Times New Roman + SimSun + STIX with per-glyph fallback, closed spines, inward ticks, dashed
light grid, no in-figure titles, legend above the axes, 300 dpi PNG plus vector PDF with
``pdf.fonttype = 42``.

    D:/project_env/python.exe analysis/plot_extended_figures.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
EXT = BASE / "execution/extended"
OUT = BASE / "execution/figures"
STATE = EXT / "EXTENDED.json"

DELIVERY_VALUES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
BUDGET_CELLS = ("G48_R6_N6", "G48_R12_N6")
BUDGET_FRACTIONS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

STYLE = {
    "normal_immediate": dict(color="#636363", ls="-.", marker="^", mfc="white",
                             lw=1.1, ms=4.5, mew=1.0, zorder=4),
    "dedup_immediate": dict(color="#7F8C8D", ls=":", marker="x", mfc="#7F8C8D",
                            lw=1.1, ms=4.5, mew=1.0, zorder=3),
    "event_maxwait": dict(color="#D97706", ls="--", marker="D", mfc="white",
                          lw=1.1, ms=4.2, mew=1.0, zorder=3),
    "slack_matched_rate": dict(color="#8E44AD", ls=":", marker="v", mfc="white",
                               lw=1.3, ms=4.5, mew=1.0, zorder=4),
    "v4f@200": dict(color="#C0392B", ls="-", marker="o", mfc="#C0392B",
                    lw=1.7, ms=4.6, mew=1.0, zorder=6),
}
ORDER = ("normal_immediate", "dedup_immediate", "event_maxwait",
         "slack_matched_rate", "v4f@200")
LABEL = {"normal_immediate": "Immediate", "dedup_immediate": "Dedup",
         "event_maxwait": "Event-MaxWait", "slack_matched_rate": "Slack-Matched",
         "v4f@200": "本文方法 (Ours)"}

BUDGET_STYLE = {
    "v4f": dict(color="#2980B9", ls="--", marker="o", mfc="none", lw=1.2, ms=6.0,
                mew=1.15, zorder=5),
    "slack_budget": dict(color="#D97706", ls="-", marker="D", mfc="white", lw=1.1, ms=4.2,
                         mew=1.0, zorder=3),
    "fifo_budget": dict(color="#7F8C8D", ls="-.", marker="s", mfc="white", lw=1.1, ms=4.2,
                        mew=1.0, zorder=2),
    "random_budget": dict(color="#8E44AD", ls=":", marker="v", mfc="white", lw=1.2, ms=4.5,
                          mew=1.0, zorder=4),
}
BUDGET_ORDER = ("v4f", "slack_budget", "fifo_budget", "random_budget")
BUDGET_LABEL = {"v4f": "本文方法 (Ours，价值排序)", "slack_budget": "Slack (时限排序)",
                "fifo_budget": "FIFO (先到先得)", "random_budget": "Random (随机)"}
VERDICT_RED = "#C0392B"


def require_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    missing = [name for name in ("Times New Roman", "SimSun", "STIXGeneral")
               if name not in available]
    if missing:
        raise SystemExit("required fonts missing: %s" % missing)


def configure() -> None:
    plt.rcParams.update({
        "font.family": ["Times New Roman", "SimSun", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.0,
        "axes.linewidth": 0.8, "axes.edgecolor": "#000000",
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.color": "#000000", "ytick.color": "#000000",
        "axes.grid": True, "grid.color": "#CCCCCC", "grid.linewidth": 0.4,
        "grid.linestyle": "--", "axes.axisbelow": True,
        "axes.unicode_minus": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.frameon": True, "legend.fancybox": False,
        "legend.edgecolor": "#000000", "legend.facecolor": "#FFFFFF",
    })


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(figure, name):
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / (name + ".png")
    pdf = OUT / (name + ".pdf")
    figure.savefig(png)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def figure_delivery(state):
    arms = state["delivery"]["by_arm"]
    x = np.arange(len(DELIVERY_VALUES))
    figure, axes = plt.subplots(2, 2, figsize=(7.4, 5.0))
    panels = (
        ("q", "(a) 绝对按时完成率", "按时数 / 目标总数"),
        ("on_time_ratio", "(b) 相对 Immediate 的按时数比", "比值（1.0 = 打平）"),
        ("bytes_per_on_time", "(c) 字节 / 按时任务", "字节"),
        ("waste_ratio", "(d) 广播超时浪费率", "已广播未按时 / 被广播"),
    )
    for axis, (field, title, ylabel) in zip(axes.ravel(), panels):
        for arm in ORDER:
            values = [arms[arm][str(p)][field] for p in DELIVERY_VALUES]
            if all(value is None for value in values):
                continue
            order = np.argsort(x)
            axis.plot(x[order], [values[index] for index in order],
                      label=LABEL[arm], **STYLE[arm])
        if field == "on_time_ratio":
            axis.axhline(1.0, color="#636363", lw=0.8, ls=":", zorder=1)
            axis.axhline(0.95, color=VERDICT_RED, lw=0.8, ls="--", zorder=1)
        axis.set_xticks(x)
        axis.set_xticklabels(["%.1f" % p for p in DELIVERY_VALUES])
        # No mathtext in a mixed Chinese label: matplotlib does not apply per-glyph fallback
        # to the non-math segments, so "$p$" beside 汉字 turns the 汉字 into dummy boxes.
        axis.set_xlabel("投递率 p，" + title.split(")")[0] + ")")
        axis.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.03),
                  columnspacing=1.0, handlelength=2.4)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return save(figure, "fig15_delivery_prob_sweep")


def figure_budget(state):
    budget = state["budget"]
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=False)
    marked = {20, 50, 100}
    for axis, cell in zip(axes, BUDGET_CELLS):
        entry = budget[cell]["by_selector"]
        ys_all = []
        for selector in BUDGET_ORDER:
            xs, ys = [], []
            for fraction in BUDGET_FRACTIONS:
                stats = entry[selector][str(fraction)]
                if stats["byte_ratio"] is None or stats["on_time_ratio"] is None:
                    continue
                xs.append(stats["byte_ratio"])
                ys.append(stats["on_time_ratio"])
            ys_all.extend(ys)
            axis.plot(xs, ys, label=BUDGET_LABEL[selector], **BUDGET_STYLE[selector])
            # Only three ceiling labels per curve, and only on the v4f curve: nine labels per
            # line turned the panel into overlapping text without adding a reading.
            if selector == "v4f":
                offsets = {20: (4, 5), 50: (-24, -12), 100: (5, 7)}
                for value_x, value_y, fraction in zip(xs, ys, BUDGET_FRACTIONS):
                    percent = int(round(fraction * 100))
                    if percent in marked:
                        axis.annotate("%d%%" % percent, (value_x, value_y),
                                      textcoords="offset points", xytext=offsets[percent],
                                      fontsize=7.5, color="#1F4E79")
        frontier = budget[cell]["pareto"]
        if frontier:
            axis.scatter([point["spent_bytes"] / budget[cell]["reference"]["spent_bytes"]
                          for point in frontier],
                         [point["on_time"] / budget[cell]["reference"]["on_time"]
                          for point in frontier],
                         s=26, facecolors="none", edgecolors="#000000", linewidths=0.8,
                         zorder=8, label="非支配点" if cell == BUDGET_CELLS[0] else None)
        axis.axhline(1.0, color="#636363", lw=0.8, ls=":", zorder=1)
        axis.axhline(0.95, color=VERDICT_RED, lw=0.8, ls="--", zorder=1)
        axis.set_xlabel("字节比（相对同格 Immediate）\n%s" % (
            "(a) " + cell.replace("_", " ") if cell == BUDGET_CELLS[0]
            else "(b) " + cell.replace("_", " ")))
        axis.set_xlim(min(xs) - 0.04, 1.04)
        low = min(ys_all + [0.95])
        high = max(ys_all + [1.0])
        axis.set_ylim(max(0.30, low - 0.05), high + 0.06)
    axes[0].set_ylabel("相对 Immediate 的按时数比")
    axes[1].set_ylabel("相对 Immediate 的按时数比")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06),
                  columnspacing=1.0, handlelength=2.4)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return save(figure, "fig16_budget_spectrum_pareto")


def main():
    sys.dont_write_bytecode = True
    require_fonts()
    configure()
    if not STATE.exists():
        raise SystemExit("missing %s; run analysis/extended_report.py first" % STATE)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    outputs = [figure_delivery(state), figure_budget(state)]
    for png, pdf in outputs:
        print("%s  sha256 %s" % (png.name, sha256(png)[:16]))
        print("%s  sha256 %s" % (pdf.name, sha256(pdf)[:16]))


if __name__ == "__main__":
    main()
