"""Comparison figures with **v4f@200 as the only learned arm**.

Every other v4-family checkpoint (v4s / v4t / v4u / v4v / v4g) is deliberately absent: this set
answers "how does the pure network compare with the zero-training baselines", not "which of our
checkpoints is best". Outputs go to ``execution/figures/v4f_only/`` so the published figures and
their manifests keep their hashes.

Four figures, all read from JSON/CSV that already exist:

    v4f_g_scaling        ten goal counts x five (R,N) pairs   (G_SCALING.json)
    v4f_dual_gate        four arms under both gates           (DUAL_GATE.json)
    v4f_budget_pareto    nine ceilings x four selectors       (EXTENDED.json)
    v4f_g_rs_grid        deep grid: R axis and S axis         (G_RS_GRID.json)
    v4f_master_g         one G axis across the three grids    (G_SCALING + G_RS)

Style follows the revision's IEEE contract (Times New Roman + SimSun + STIX, closed spines,
inward ticks, no in-figure titles, legend above, 300 dpi + vector PDF, embedded fonts). All
canvas text is English so the figures drop into a LaTeX paper without a CJK font.

    D:/project_env/python.exe analysis/plot_v4f_only_figures.py
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
OUT = BASE / "execution/figures/v4f_only"
G_SCALING = BASE / "execution/g_scaling/G_SCALING.json"
DUAL = BASE / "execution/relaxed_gate/DUAL_GATE.json"
EXTENDED = BASE / "execution/extended/EXTENDED.json"
G_RS = BASE / "execution/g_rs_grid/G_RS_GRID.json"
R_S = BASE / "execution/r_s_scaling/R_S_SCALING.json"

ARMS = ("normal_immediate", "slack_heuristic_cap", "slack_matched_rate", "v4f@200")
LABEL = {"normal_immediate": "Immediate", "slack_heuristic_cap": "Slack-Cap",
         "slack_matched_rate": "Slack-Matched", "v4f@200": "v4f (pure network, ours)"}
STYLE = {
    "normal_immediate": dict(color="#636363", ls="-.", marker="^", mfc="white",
                             lw=1.1, ms=4.5, mew=1.0, zorder=4),
    "slack_heuristic_cap": dict(color="#D97706", ls="--", marker="D", mfc="white",
                                lw=1.1, ms=4.2, mew=1.0, zorder=3),
    "slack_matched_rate": dict(color="#8E44AD", ls=":", marker="v", mfc="white",
                               lw=1.3, ms=4.5, mew=1.0, zorder=4),
    "v4f@200": dict(color="#C0392B", ls="-", marker="o", mfc="#C0392B",
                    lw=1.7, ms=4.8, mew=1.0, zorder=6),
}
GOALS = (12, 18, 24, 30, 36, 42, 48, 54, 60, 72)
DEEP_GOALS = (72, 84, 96, 108, 120, 144)
BUDGET_CELLS = ("G48_R6_N6", "G48_R12_N6")
BUDGET_FRACTIONS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
SELECTORS = ("v4f", "slack_budget", "fifo_budget", "random_budget")
SELECTOR_LABEL = {"v4f": "v4f (value ranking)", "slack_budget": "Slack (deadline)",
                  "fifo_budget": "FIFO", "random_budget": "Random"}
SELECTOR_STYLE = {
    "v4f": dict(color="#C0392B", ls="-", marker="o", mfc="#C0392B", lw=1.7, ms=4.6, zorder=6),
    "slack_budget": dict(color="#D97706", ls="-", marker="D", mfc="white", lw=1.1, ms=4.2,
                         zorder=3),
    "fifo_budget": dict(color="#7F8C8D", ls="-.", marker="s", mfc="white", lw=1.1, ms=4.2,
                        zorder=2),
    "random_budget": dict(color="#8E44AD", ls=":", marker="v", mfc="white", lw=1.2, ms=4.5,
                          zorder=4),
}
VERDICT_RED = "#C0392B"
BAR_STYLE = {
    "normal_immediate": dict(facecolor="#D9D9D9", edgecolor="#4D4D4D"),
    "slack_heuristic_cap": dict(facecolor="#FBE3C4", edgecolor="#B9770E"),
    "slack_matched_rate": dict(facecolor="#E7D8F3", edgecolor="#7D3C98"),
    "v4f@200": dict(facecolor="#F5C6C0", edgecolor="#C0392B"),
}


def require_fonts():
    available = {font.name for font in font_manager.fontManager.ttflist}
    missing = [name for name in ("Times New Roman", "SimSun", "STIXGeneral")
               if name not in available]
    if missing:
        raise SystemExit("required fonts missing: %s" % missing)


def configure():
    plt.rcParams.update({
        "font.family": ["Times New Roman", "SimSun", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8.2, "ytick.labelsize": 8.5, "legend.fontsize": 8.0,
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(figure, name):
    OUT.mkdir(parents=True, exist_ok=True)
    png, pdf = OUT / (name + ".png"), OUT / (name + ".pdf")
    figure.savefig(png)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def legend_above(figure, axes, ncol=4):
    handles, labels = axes.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 1.03),
                  columnspacing=1.2, handlelength=2.6)


# --------------------------------------------------------------------- G scaling
def figure_g_scaling(state):
    aggregate = state["aggregate"]
    base = aggregate["normal_immediate"]
    x = np.arange(len(GOALS))
    figure, axes = plt.subplots(2, 2, figsize=(7.4, 5.0))

    def series(arm, field):
        return [aggregate[arm][str(goal)][field] for goal in GOALS]

    byte_cut = [100.0 * (1.0 - own / ref) for own, ref in
                zip(series("v4f@200", "spent_bytes"), series("normal_immediate", "spent_bytes"))]
    ratio = [own / ref for own, ref in
             zip(series("v4f@200", "on_time"), series("normal_immediate", "on_time"))]
    bpt = [own / on for own, on in
           zip(series("v4f@200", "spent_bytes"), series("v4f@200", "on_time"))]
    waste = series("v4f@200", "waste")
    panels = (
        ((byte_cut,), ("Bytes saved vs Immediate (%)",), "(a) byte saving", 0.0),
        ((ratio,), ("On-time / Immediate",), "(b) on-time relative to Immediate", 1.0),
        ((bpt,), ("Bytes per on-time task",), "(c) bytes per delivered task", None),
        ((waste,), ("Broadcast waste",), "(d) broadcast waste", 0.10),
    )
    for axis, (values, (ylabel,), title, reference) in zip(axes.ravel(), panels):
        for arm in ("slack_heuristic_cap", "slack_matched_rate"):
            own = series(arm, "spent_bytes")
            ref = series("normal_immediate", "spent_bytes")
            if "byte" in title and "per" not in title:
                data = [100.0 * (1.0 - a / b) for a, b in zip(own, ref)]
            elif "on-time" in title:
                data = [a / b for a, b in zip(series(arm, "on_time"), series("normal_immediate", "on_time"))]
            elif "per" in title:
                data = [a / b for a, b in zip(own, series(arm, "on_time"))]
            else:
                data = series(arm, "waste")
            axis.plot(x, data, label=LABEL[arm], **STYLE[arm])
        axis.plot(x, values[0], label=LABEL["v4f@200"], **STYLE["v4f@200"])
        if reference is not None:
            axis.axhline(reference, color="#636363", lw=0.8,
                         ls="--" if reference in (0.95, 0.10) else ":",
                         zorder=1)
        if reference == 0.10:
            axis.annotate("waste 0.10", (0.1, 0.10), ha="left", va="bottom", fontsize=7.5,
                          color=VERDICT_RED)
        if title.startswith("(b)"):
            axis.axhline(0.95, color=VERDICT_RED, lw=0.9, ls="--", zorder=1)
            axis.annotate("0.95", (0.1, 0.952), ha="left", va="bottom", fontsize=7.5,
                          color=VERDICT_RED)
        axis.set_xticks(x)
        axis.set_xticklabels([str(g) for g in GOALS])
        axis.set_xlabel("Goals per world \u2014 " + title)
        axis.set_ylabel(ylabel)
    legend_above(figure, axes[0, 0])
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return save(figure, "v4f_g_scaling")


# --------------------------------------------------------------------- dual gate
def figure_dual_gate(state):
    arms = state["arms"]
    x = np.arange(len(ARMS))
    panels = (
        ("on_time_passes", "Cells passing (/15)", "(a) per-cell on-time passes", "%d/15", 16.5),
        ("on_time_ratio", "On-time / Immediate", "(b) on-time relative to Immediate", "%.3f", None),
        ("bytes_ratio", "Bytes / Immediate", "(c) bytes relative to Immediate", "%.3f", None),
        ("waste_ratio", "Broadcast waste", "(d) broadcast waste", "%.3f", None),
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.4, 4.9))
    for axis, (field, ylabel, title, pattern, limit) in zip(axes.ravel(), panels):
        values = {arm: (arms[arm][field] or 0.0) for arm in ARMS}
        for index, arm in enumerate(ARMS):
            axis.bar(x[index], values[arm], width=0.62, linewidth=0.8, zorder=3,
                     **BAR_STYLE[arm])
            text = (pattern % arms[arm][field]) if field == "on_time_passes" else (pattern % values[arm])
            axis.annotate(text, (x[index], values[arm]), textcoords="offset points",
                          xytext=(0, 3), ha="center", fontsize=7.8, color="#000000")
        axis.set_xticks(x)
        axis.set_xticklabels([LABEL[a] for a in ARMS], rotation=16, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_xlim(-0.65, len(ARMS) - 0.35)
        axis.set_xlabel(title)
        if field == "on_time_passes":
            axis.set_ylim(0, limit)
            axis.set_yticks(range(0, 16, 3))
        elif field == "on_time_ratio":
            axis.set_ylim(0, max(1.15, max(values.values()) + 0.05))
            axis.axhline(1.0, color="#636363", lw=0.8, ls=":", zorder=2)
        elif field == "bytes_ratio":
            axis.set_ylim(0, 1.12)
            axis.axhline(1.0, color="#636363", lw=0.8, ls=":", zorder=2)
            axis.axhline(0.85, color=VERDICT_RED, lw=0.9, ls="--", zorder=2)
            axis.annotate("gate 0.85", (0.05, 0.852), ha="left", va="bottom", fontsize=7.5,
                          color=VERDICT_RED)
        else:
            axis.set_ylim(0, max(0.32, max(values.values()) + 0.05))
            axis.axhline(0.10, color=VERDICT_RED, lw=0.9, ls="--", zorder=2)
            axis.annotate("gate 0.10", (0.05, 0.102), ha="left", va="bottom", fontsize=7.5,
                          color=VERDICT_RED)
    figure.tight_layout(h_pad=1.4, w_pad=1.0)
    return save(figure, "v4f_dual_gate")


# --------------------------------------------------------------------- budget pareto
def figure_budget(state):
    budget = state["budget"]
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for axis, cell in zip(axes, BUDGET_CELLS):
        entry = budget[cell]["by_selector"]
        ys_all = []
        for selector in SELECTORS:
            xs, ys = [], []
            for fraction in BUDGET_FRACTIONS:
                stats = entry[selector][str(fraction)]
                if stats["byte_ratio"] is None or stats["on_time_ratio"] is None:
                    continue
                xs.append(stats["byte_ratio"])
                ys.append(stats["on_time_ratio"])
            ys_all.extend(ys)
            axis.plot(xs, ys, label=SELECTOR_LABEL[selector], **SELECTOR_STYLE[selector])
            if selector == "v4f":
                for value_x, value_y, fraction in zip(xs, ys, BUDGET_FRACTIONS):
                    percent = int(round(fraction * 100))
                    if percent in (20, 50, 100):
                        offset = {20: (4, 5), 50: (-24, -12), 100: (5, 7)}[percent]
                        axis.annotate("%d%%" % percent, (value_x, value_y),
                                      textcoords="offset points", xytext=offset,
                                      fontsize=7.5, color="#8C2F22")
        axis.axhline(1.0, color="#636363", lw=0.8, ls=":", zorder=1)
        axis.axhline(0.95, color=VERDICT_RED, lw=0.8, ls="--", zorder=1)
        axis.set_xlim(min(xs) - 0.04, 1.04)
        axis.set_ylim(max(0.0, min(ys_all) - 0.06), max(ys_all) + 0.08)
        axis.set_xlabel("Bytes / Immediate\n%s" % (
            "(a) G48_R6_N6" if cell == BUDGET_CELLS[0] else "(b) G48_R12_N6"))
        axis.set_ylabel("On-time / Immediate")
    legend_above(figure, axes[0], ncol=4)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return save(figure, "v4f_budget_pareto")


# --------------------------------------------------------------------- deep grid
def figure_g_rs(state):
    """Deep grid: on-time and bytes for the three baselines plus v4f, along R and along S."""
    robots, agents = state["robots"], state["search_agents"]
    figure, axes = plt.subplots(2, 2, figsize=(7.4, 5.0))

    def ratio(block, arm, value, field):
        own = state[block][arm][str(value)]
        ref = state[block]["normal_immediate"][str(value)]
        return (own["on_time"] / ref["on_time"] if field == "on_time"
                else own["spent_bytes"] / ref["spent_bytes"])

    panels = (
        (axes[0, 0], "by_robot", robots, "Rescue robots R", "on_time",
         "(a) on-time along R"),
        (axes[0, 1], "by_robot", robots, "Rescue robots R", "bytes", "(b) bytes along R"),
        (axes[1, 0], "by_agent", agents, "Search agents S", "on_time",
         "(c) on-time along S"),
        (axes[1, 1], "by_agent", agents, "Search agents S", "bytes", "(d) bytes along S"),
    )
    for axis, block, values, xlabel, field, title in panels:
        for arm in ARMS[1:]:
            axis.plot(range(len(values)),
                      [ratio(block, arm, value, field) for value in values],
                      label=LABEL[arm], **STYLE[arm])
        axis.set_xticks(range(len(values)))
        axis.set_xticklabels([str(value) for value in values])
        axis.set_xlabel("%s — %s" % (xlabel, title))
        if field == "on_time":
            axis.set_ylabel("On-time / Immediate")
            axis.axhline(1.0, color="#636363", lw=0.8, ls=":", zorder=1)
            axis.axhline(0.95, color=VERDICT_RED, lw=0.9, ls="--", zorder=1)
        else:
            axis.set_ylabel("Bytes / Immediate")
            axis.set_ylim(0, 1.08)
            axis.axhline(1.0, color="#636363", lw=0.8, ls=":", zorder=1)
    legend_above(figure, axes[0, 0], ncol=3)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return save(figure, "v4f_g_rs_grid")


# --------------------------------------------------------------------- one G axis
def figure_master(state, grid):
    aggregate = state["aggregate"]
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    goals_all = list(GOALS) + [g for g in DEEP_GOALS if g != 72]
    index = {goal: position for position, goal in enumerate(goals_all)}
    # Labels are the union of both grids; 72 appears once but carries two markers (B and C).
    for axis, (field, ylabel, title) in zip(axes, (
            ("on_time_ratio", "On-time / Immediate", "(a) on-time relative to Immediate"),
            ("bytes_ratio", "Bytes / Immediate", "(b) bytes relative to Immediate"))):
        for arm in ARMS[1:]:
            xs, ys = [], []
            for goal in GOALS:
                ref = aggregate["normal_immediate"][str(goal)]
                own = aggregate[arm][str(goal)]
                xs.append(index[goal])
                ys.append((own["on_time"] / ref["on_time"]) if field == "on_time_ratio"
                          else (own["spent_bytes"] / ref["spent_bytes"]))
            axis.plot(xs, ys, label=LABEL[arm], **STYLE[arm])
        xs, ys = [], []
        for goal in DEEP_GOALS:
            ref = grid["by_goal"]["normal_immediate"][str(goal)]
            own = grid["by_goal"]["v4f@200"][str(goal)]
            xs.append(index[goal])
            ys.append((own["on_time"] / ref["on_time"]) if field == "on_time_ratio"
                      else (own["spent_bytes"] / ref["spent_bytes"]))
        axis.plot(xs, ys, ls="--", marker="s", mfc="white", color="#C0392B", lw=1.2,
                  ms=4.2, zorder=5, label="v4f (deep grid, G 72-144)" if field == "on_time_ratio"
                  else None)
        if field == "on_time_ratio":
            axis.axhline(1.0, color="#636363", lw=0.8, ls=":", zorder=1)
            axis.axhline(0.95, color=VERDICT_RED, lw=0.9, ls="--", zorder=1)
        else:
            axis.axhline(1.0, color="#636363", lw=0.8, ls=":", zorder=1)
            axis.axhline(0.85, color=VERDICT_RED, lw=0.9, ls="--", zorder=1)
        axis.set_xticks(range(len(goals_all)))
        axis.set_xticklabels([str(g) for g in goals_all], rotation=45, ha="right")
        axis.set_xlabel("Goals per world — " + title)
        axis.set_ylabel(ylabel)
        if field == "bytes_ratio":
            axis.set_ylim(0, 1.1)
    legend_above(figure, axes[0], ncol=4)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return save(figure, "v4f_master_g")


# --------------------------------------------------------------------- R and S axes
def figure_r_s(state):
    """The two orthogonal axes: rescue roster R and search-agent count S."""
    r_values = [int(v) for v in state["r_axis"]["values"]]
    s_values = [int(v) for v in state["s_axis"]["values"]]
    figure, axes = plt.subplots(2, 2, figsize=(7.4, 5.0))

    def ratio(block, arm, value, field):
        own = state[block][arm][str(value)]
        ref = state[block]["normal_immediate"][str(value)]
        return (own["on_time"] / ref["on_time"] if field == "on_time"
                else own["spent_bytes"] / ref["spent_bytes"])

    panels = (
        (axes[0, 0], "r_points", r_values, "Rescue robots R", "on_time", "linear",
         "(a) on-time along R"),
        (axes[0, 1], "r_points", r_values, "Rescue robots R", "bytes", "linear",
         "(b) bytes along R"),
        (axes[1, 0], "s_points", s_values, "Search agents S", "on_time", "categorical",
         "(c) on-time along S"),
        (axes[1, 1], "s_points", s_values, "Search agents S", "bytes", "categorical",
         "(d) bytes along S"),
    )
    for axis, block, values, xlabel, field, scale, title in panels:
        positions = list(range(len(values))) if scale == "categorical" else values
        for arm in ARMS[1:]:
            axis.plot(positions, [ratio(block, arm, value, field) for value in values],
                      label=LABEL[arm], **STYLE[arm])
        axis.set_xticks(positions)
        axis.set_xticklabels([str(value) for value in values])
        axis.set_xlabel("%s — %s" % (xlabel, title))
        if field == "on_time":
            axis.set_ylabel("On-time / Immediate")
            axis.axhline(1.0, color="#636363", lw=0.8, ls=":", zorder=1)
            axis.axhline(0.95, color=VERDICT_RED, lw=0.9, ls="--", zorder=1)
        else:
            axis.set_ylabel("Bytes / Immediate")
            axis.set_ylim(0, 1.08)
            axis.axhline(1.0, color="#636363", lw=0.8, ls=":", zorder=1)
    legend_above(figure, axes[0, 0], ncol=3)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return save(figure, "v4f_r_s_scaling")


def main():
    sys.dont_write_bytecode = True
    require_fonts()
    configure()
    for path in (G_SCALING, DUAL, EXTENDED, G_RS):
        if not path.exists():
            raise SystemExit("missing %s" % path)
    scaling = load(G_SCALING)
    dual = load(DUAL)
    extended = load(EXTENDED)
    grid = load(G_RS)
    r_s = load(R_S) if R_S.exists() else None
    outputs = [
        figure_g_scaling(scaling),
        figure_dual_gate(dual),
        figure_budget(extended),
        figure_g_rs(grid),
        figure_master(scaling, grid),
    ]
    if r_s is not None:
        outputs.append(figure_r_s(r_s))
    for png, pdf in outputs:
        print("%s  sha256 %s" % (png.name, sha256(png)[:16]))
        print("%s  sha256 %s" % (pdf.name, sha256(pdf)[:16]))


if __name__ == "__main__":
    main()
