"""IEEE-style paper figures from the archived G-sweep and efficiency readings. Offline, no re-runs.

Reads only files already on disk:

* ``execution/g_scaling/G_SCALING.json`` -- the ten-goal-count aggregates (six arms) and the
  per-(G, R, N) cells behind the report's tables 4 and 5;
* ``execution/g_scaling/WASTE.json``     -- per-world broadcast-waste counts (1 770 worlds);
* ``execution/COMM_EFFICIENCY.json``     -- the fourteen arms' four efficiency readings on the
  fifteen training cells (anchor / unseen / all(8)).

Nothing here opens a world, trains, or touches the 98 pinned files.

Style contract (IEEE / control-discipline, no dashboard decoration):

* Times New Roman for Latin/digits, SimSun for CJK, STIX for math; 8.5-9.5 pt. Mixed
  Chinese+variable strings use Unicode math italics (𝐺/𝑅/𝑁) instead of ``$...$``: the
  mathtext renderer does not apply font fallback to the non-math segments of a string,
  so ``$G$`` inside a Chinese label would turn the Chinese glyphs into dummy boxes;
* all four spines closed at 0.8 pt black, ticks inward on all sides, light dashed grid;
* no figure/axes titles, no text bubbles, no numbers on data points; subplot labels live
  inside the x-axis label, e.g. ``(a) 任务规模 𝐺``;
* the goal-count axis is categorical (equidistant indices labelled 12..72), so the 60->72
  step is not stretched;
* academic legend names, no code underscores: Immediate / Slack-Cap / Slack-Matched /
  Pure-Policy (v4f) / Ours-Joint (v4s) / Ours-Frozen (v4t). Ours-Frozen is deep red with a
  filled dot; Pure-Policy is a large hollow blue ring drawn ABOVE it (zorder 7 > 6), so a
  coincident pair reads as a red dot inside a blue ring instead of one hiding the other;
* in fig1(a)/(b) Immediate is only the grey dash-dot reference line at y=0 / y=1.0 (no
  markers), and fig1(a) omits Slack-Matched (its page count matches v4t by construction,
  so the curve would sit on top of v4t/v4f);
* 300 dpi PNG + vector PDF with embedded TrueType fonts (``pdf.fonttype = 42``).

    D:/project_env/python.exe analysis/plot_paper_figures.py              # all four figures
    D:/project_env/python.exe analysis/plot_paper_figures.py --only 2,3   # a subset
    D:/project_env/python.exe analysis/plot_paper_figures.py --no-manifest

Output: ``execution/figures/fig<N>_<slug>.{png,pdf}`` plus a manifest section appended to
``G_SCALING_EXPERIMENTS.md`` carrying each file's sha256. The previous dashboard-style renders
are kept in ``execution/figures/_dashboard_v1/`` (quarantined, not deleted).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager, ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
G_SCALING = BASE / "execution/g_scaling/G_SCALING.json"
WASTE = BASE / "execution/g_scaling/WASTE.json"
EFFICIENCY = BASE / "execution/COMM_EFFICIENCY.json"
OUT = BASE / "execution/figures"
REPORT = BASE / "G_SCALING_EXPERIMENTS.md"

GOALS = (12, 18, 24, 30, 36, 42, 48, 54, 60, 72)
THROTTLE_GOALS = (36, 42, 48, 54, 60, 72)
FULL_SCENE_PAIRS = ((6, 6), (12, 6))
ANCHOR = (2301901, 2301902, 2301903)
ALL_SCENES = tuple(range(2301901, 2301909))
G_STAR = 36
R6_PAIRS = ((6, 3), (6, 6))
R12_PAIRS = ((12, 3), (12, 6), (12, 12))

#: One style per arm, fixed here so every figure says the same thing about the same line.
#: Baselines are hollow; the two trained quota arms are solid; Pure-Policy's large hollow
#: ring sits above Ours-Frozen's filled dot so coincidence stays visible.
STYLE = {
    "normal_immediate": dict(color="#636363", ls="-.", marker="^", mfc="white",
                             lw=1.1, ms=4.5, mew=1.0, zorder=4),
    "slack_heuristic_cap": dict(color="#D97706", ls="--", marker="D", mfc="white",
                                lw=1.1, ms=4.2, mew=1.0, zorder=3),
    "slack_matched_rate": dict(color="#8E44AD", ls=":", marker="v", mfc="white",
                               lw=1.3, ms=4.5, mew=1.0, zorder=4),
    "v4f@200": dict(color="#C0392B", ls="-", marker="o", mfc="#C0392B",
                    lw=1.7, ms=4.8, mew=1.0, zorder=6),
    "v4s@300": dict(color="#1F618D", ls="-", marker="s", mfc="#1F618D",
                    lw=1.35, ms=4.5, mew=1.0, zorder=5),
    "v4t@100": dict(color="#C0392B", ls="-", marker="o", mfc="#C0392B",
                    lw=1.7, ms=4.6, mew=1.0, zorder=6),
}
ORDER = ("normal_immediate", "slack_heuristic_cap", "slack_matched_rate", "v4f@200")
LABEL = {"normal_immediate": "Immediate (全量广播基线)",
         "slack_heuristic_cap": "Slack-Cap (时限配额启发式)",
         "slack_matched_rate": "Slack-Matched (同预算启发式)",
         "v4f@200": "本文方法 (Ours)"}
VERDICT_RED = "#C0392B"
PARITY_BLACK = "#000000"
IMMEDIATE_GREY = "#636363"
R12_BLUE = "#1F618D"


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
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": PARITY_BLACK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.color": PARITY_BLACK,
        "ytick.color": PARITY_BLACK,
        "axes.grid": True,
        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.4,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": PARITY_BLACK,
        "legend.facecolor": "#FFFFFF",
        "legend.framealpha": 1.0,
    })


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit("missing input: %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scenes_for(pair):
    """Same rule the sweep used: the two representative pairs run all eight DEV scenes."""
    return ALL_SCENES if tuple(pair) in FULL_SCENE_PAIRS else ANCHOR


def series(g_scaling: dict, arm: str, field: str) -> list:
    aggregate = g_scaling["aggregate"][arm]
    return [aggregate[str(goal)][field] for goal in GOALS]


def byte_cut(g_scaling: dict, arm: str) -> list:
    base = series(g_scaling, "normal_immediate", "spent_bytes")
    own = series(g_scaling, arm, "spent_bytes")
    return [100.0 * (1.0 - value / ref) for value, ref in zip(own, base)]


def ratio_to_immediate(g_scaling: dict, arm: str) -> list:
    base = series(g_scaling, "normal_immediate", "on_time")
    own = series(g_scaling, arm, "on_time")
    return [value / ref for value, ref in zip(own, base)]


def cell(g_scaling: dict, arm: str, goal: int, pair) -> dict | None:
    return g_scaling["cells"].get("%s|G%d_R%d_N%d" % (arm, goal, pair[0], pair[1]))


def grouped(g_scaling: dict, arm: str, goal: int, pairs) -> dict:
    """Aggregate one subset of (R, N) pairs at one goal count, over the worlds it ran."""
    on = goals = sent = spent = ref_spent = ref_on = 0
    for pair in pairs:
        entry = cell(g_scaling, arm, goal, pair)
        base = cell(g_scaling, "normal_immediate", goal, pair)
        if entry is None or base is None:
            raise SystemExit("missing cell %s G%d R%d N%d" % (arm, goal, pair[0], pair[1]))
        on += entry["on_time"]
        goals += entry["goals"]
        sent += entry["emissions"]
        spent += entry["spent_bytes"]
        ref_spent += base["spent_bytes"]
        ref_on += base["on_time"]
    return {"on_time": on, "goals": goals, "emissions": sent, "spent_bytes": spent,
            "ratio": on / ref_on if ref_on else None,
            "byte_ratio": spent / ref_spent if ref_spent else None}


def plot_arm(ax, x, y, arm, label=None, **override):
    style = STYLE[arm]
    kwargs = dict(color=style["color"], ls=style["ls"], marker=style["marker"],
                  markerfacecolor=style["mfc"], markeredgecolor=style["color"],
                  markeredgewidth=style["mew"], lw=style["lw"], ms=style["ms"],
                  zorder=style["zorder"], label=LABEL[arm] if label is None else label)
    kwargs.update(override)
    ax.plot(x, y, **kwargs)


def arm_handle(arm):
    style = STYLE[arm]
    return Line2D([], [], color=style["color"], ls=style["ls"], marker=style["marker"],
                  markerfacecolor=style["mfc"], markeredgecolor=style["color"],
                  markeredgewidth=style["mew"], lw=style["lw"], ms=style["ms"],
                  label=LABEL[arm])


def goal_ticks(ax, goals=GOALS):
    """Categorical, equidistant goal axis: index positions labelled with the goal counts."""
    x = np.arange(len(goals))
    ax.set_xlim(-0.4, len(goals) - 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(goal) for goal in goals])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    return x


def reference_lines(ax, label_at=None, parity="black"):
    if parity == "grey":
        ax.axhline(1.0, color=IMMEDIATE_GREY, lw=1.1, ls="-.", zorder=1)
    else:
        ax.axhline(1.0, color=PARITY_BLACK, lw=0.8, zorder=1)
    ax.axhline(0.95, color=VERDICT_RED, lw=1.0, ls="--", zorder=1)
    if label_at is not None:
        x, y, ha, va = label_at
        ax.text(x, y, "0.95 准时率红线", color=VERDICT_RED, fontsize=8,
                ha=ha, va=va, zorder=2)


def place_legend(fig, handles, ncol):
    # lower center anchored at the figure top: the legend grows UPWARD, entirely outside
    # the axes area (bbox_inches="tight" expands the canvas). upper center would hang
    # down into the rect reserved by tight_layout and cover the top of short figures.
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                        ncol=ncol, handlelength=2.4, handletextpad=0.5,
                        columnspacing=1.3, labelspacing=0.35, borderpad=0.45)
    legend.get_frame().set_linewidth(0.6)
    return legend


def tighten(fig):
    fig.tight_layout(rect=[0, 0, 1, 0.905], pad=0.5, w_pad=1.0, h_pad=0.95)


def save(fig, slug: str) -> list:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "pdf"):
        path = OUT / ("%s.%s" % (slug, suffix))
        fig.savefig(path, dpi=300)
        written.append(path)
    plt.close(fig)
    return written


def check_waste(g_scaling: dict, waste: dict) -> None:
    """``G_SCALING.json``'s waste must equal a re-aggregation of ``WASTE.json``.

    WASTE.json also holds the full 15-cell archive for G = 12/24/48 (120 worlds), while the sweep
    uses 25 of those, so the scene set has to be narrowed to what each (R, N) pair actually ran.
    """
    rows = [row for row in waste["results"] if row.get("status") == "ok"]
    for arm in ORDER:
        for goal in GOALS:
            chosen = [row for row in rows
                      if row["arm"] == arm and row["cell"][0] == goal
                      and row["scene"] in scenes_for((row["cell"][1], row["cell"][2]))]
            if not chosen:
                continue
            num = sum(row["waste_n"] for row in chosen)
            den = sum(row["waste_d"] for row in chosen)
            expected = num / den if den else None
            got = g_scaling["aggregate"][arm][str(goal)]["waste"]
            if expected is None or got is None or abs(expected - got) > 1e-9:
                raise SystemExit("waste mismatch: %s G%d %r != %r"
                                 % (arm, goal, got, expected))


# --------------------------------------------------------------------------- figure 1
def figure_one(g_scaling: dict, waste: dict) -> list:
    check_waste(g_scaling, waste)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))

    # (a) byte reduction against the same-goal immediate. Immediate itself is the y=0
    # reference line; Slack-Matched is omitted because its page count matches v4t by
    # construction and the curve would lie on top of v4t/v4f.
    ax = axes[0][0]
    x = goal_ticks(ax)
    ax.axhline(0.0, color=IMMEDIATE_GREY, lw=1.1, ls="-.", zorder=1)
    for arm in ("slack_heuristic_cap", "v4f@200", "v4s@300", "v4t@100"):
        plot_arm(ax, x, byte_cut(g_scaling, arm), arm)
    ax.set_ylim(-4, 52)
    ax.set_xlabel("(a) 任务规模 𝐺")
    ax.set_ylabel("字节削减率（%）")

    # (b) on-time ratio against the same-goal immediate. Immediate is the y=1.0 grey
    # dash-dot reference line; the registered 0.95 bar is named just below itself.
    ax = axes[0][1]
    x = goal_ticks(ax)
    ax.axhspan(0.95, 1.25, color="#E8F5E9", alpha=0.4, zorder=0)
    reference_lines(ax, label_at=(0.15, 0.935, "left", "top"), parity="grey")
    for arm in ("slack_heuristic_cap", "slack_matched_rate", "v4f@200", "v4s@300",
                "v4t@100"):
        plot_arm(ax, x, ratio_to_immediate(g_scaling, arm), arm)
    ax.set_ylim(0.60, 1.25)
    ax.set_xlabel("(b) 任务规模 𝐺")
    ax.set_ylabel("相对按时完成率比值")

    # (c) messages per on-time task
    ax = axes[1][0]
    x = goal_ticks(ax)
    for arm in ORDER:
        plot_arm(ax, x, series(g_scaling, arm, "messages_per_on_time"), arm)
    ax.set_ylim(0.9, 2.6)
    ax.set_xlabel("(c) 任务规模 𝐺")
    ax.set_ylabel("每完成任务消息数（条/任务）")

    # (d) broadcast waste ratio, its own subplot (no secondary y axis anywhere)
    ax = axes[1][1]
    x = goal_ticks(ax)
    for arm in ORDER:
        plot_arm(ax, x, series(g_scaling, arm, "waste"), arm)
    ax.set_ylim(0.0, 0.65)
    ax.set_xlabel("(d) 任务规模 𝐺")
    ax.set_ylabel("广播超时浪费率")

    place_legend(fig, [arm_handle(arm) for arm in ORDER], ncol=3)
    tighten(fig)
    return save(fig, "fig1_g_scaling_efficiency")


# --------------------------------------------------------------------------- figure 2
def figure_two(g_scaling: dict, waste: dict) -> list:
    check_waste(g_scaling, waste)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    # (a) R = 6: immediate vs v4t on-time counts, the crossing marked, the net gain filled
    ax = axes[0]
    x = goal_ticks(ax)
    r6 = g_scaling["r6_series"]
    imm = [r6[str(goal)]["immediate"] for goal in GOALS]
    v4t = [r6[str(goal)]["v4t"] for goal in GOALS]
    tail = [index for index, goal in enumerate(GOALS) if goal >= G_STAR]
    ax.fill_between(x[tail], [imm[i] for i in tail], [v4t[i] for i in tail],
                    color=STYLE["v4t@100"]["color"], alpha=0.18, lw=0, zorder=1)
    ax.axvline(GOALS.index(G_STAR), color="#7F7F7F", lw=0.9, ls="--", zorder=1)
    plot_arm(ax, x, imm, "normal_immediate")
    plot_arm(ax, x, v4t, "v4t@100")
    ax.text(GOALS.index(G_STAR) + 0.15, 290, "$G^*=36$", color="#404040", fontsize=9,
            ha="left", va="top")
    ax.set_ylim(90, 295)
    worlds = sum(cell(g_scaling, "normal_immediate", GOALS[0], pair)["worlds"]
                 for pair in R6_PAIRS)
    ax.set_xlabel("(a) 任务规模 𝐺（𝑅=6）")
    ax.set_ylabel("按时完成任务数（%d 世界）" % worlds)

    # (b) v4t only: relative on-time (solid) and byte ratio (dashed), R=6 vs R=12.
    # Both quantities are dimensionless ratios to the same-group immediate, so they share
    # one axis; no secondary y axis.
    ax = axes[1]
    x = goal_ticks(ax)
    reference_lines(ax, label_at=(len(GOALS) - 0.7, 0.958, "right", "bottom"))
    r6_rows = [grouped(g_scaling, "v4t@100", goal, R6_PAIRS) for goal in GOALS]
    r12_rows = [grouped(g_scaling, "v4t@100", goal, R12_PAIRS) for goal in GOALS]
    red = STYLE["v4t@100"]["color"]
    ax.plot(x, [row["ratio"] for row in r6_rows], color=red, ls="-", marker="o",
            ms=4.6, mfc=red, mec=red, mew=1.0, lw=1.7, zorder=6)
    ax.plot(x, [row["ratio"] for row in r12_rows], color=R12_BLUE, ls="-", marker="s",
            ms=4.5, mfc=R12_BLUE, mec=R12_BLUE, mew=1.0, lw=1.35, zorder=5)
    ax.plot(x, [row["byte_ratio"] for row in r6_rows], color=red, ls="--", marker="o",
            ms=4.6, mfc="white", mec=red, mew=1.0, lw=1.1, zorder=4)
    ax.plot(x, [row["byte_ratio"] for row in r12_rows], color=R12_BLUE, ls="--",
            marker="s", ms=4.5, mfc="white", mec=R12_BLUE, mew=1.0, lw=1.1, zorder=3)
    ax.set_ylim(0.25, 1.55)
    ax.set_xlabel("(b) 任务规模 𝐺")
    ax.set_ylabel("比值（相对同组 Immediate）")

    handles = [
        arm_handle("normal_immediate"),
        arm_handle("v4t@100"),
        Patch(facecolor=red, alpha=0.18, edgecolor="none",
              label="净增区（𝐺≥36）"),
        Line2D([], [], color=red, ls="-", marker="o", ms=4.6, mfc=red, mec=red,
               lw=1.7, label="𝑅=6 按时比"),
        Line2D([], [], color=red, ls="--", marker="o", ms=4.6, mfc="white", mec=red,
               lw=1.1, label="𝑅=6 字节比"),
        Line2D([], [], color=R12_BLUE, ls="-", marker="s", ms=4.5, mfc=R12_BLUE,
               mec=R12_BLUE, lw=1.35, label="𝑅=12 按时比"),
        Line2D([], [], color=R12_BLUE, ls="--", marker="s", ms=4.5, mfc="white",
               mec=R12_BLUE, lw=1.1, label="𝑅=12 字节比"),
    ]
    place_legend(fig, handles, ncol=4)
    tighten(fig)
    return save(fig, "fig2_r6_phase_transition")


# --------------------------------------------------------------------------- figure 3
def figure_three(g_scaling: dict, waste: dict) -> list:
    check_waste(g_scaling, waste)
    panels = (((6, 6), "(a)"), ((12, 6), "(b)"), ((12, 12), "(c)"))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))

    for index, (ax, (pair, letter)) in enumerate(zip(axes, panels)):
        x = goal_ticks(ax, THROTTLE_GOALS)
        budget, network, capped = [], [], []
        for goal in THROTTLE_GOALS:
            base = cell(g_scaling, "normal_immediate", goal, pair)
            free = cell(g_scaling, "v4f@200", goal, pair)
            learned = cell(g_scaling, "v4t@100", goal, pair)
            budget.append(max(1.0, round(base["sends_per_world"]) - 1.0))
            network.append(free["sends_per_world"])
            capped.append(learned["sends_per_world"])

        ax.plot(x, budget, color="#7F7F7F", ls=":", lw=1.2, zorder=2)
        # The large hollow ring (zorder 7) over the narrower filled dot (zorder 6): where
        # the two coincide the ring is still visible around it, which is panel (a)'s claim.
        plot_arm(ax, x, network, "v4f@200")
        plot_arm(ax, x, capped, "v4t@100")
        ax.set_ylim(0, max(budget) * 1.15)
        ax.set_xlabel("%s 𝐺（𝑅=%d, 𝑁=%d）" % (letter, pair[0], pair[1]))
        if index == 0:
            ax.set_ylabel("发送次数 / 世界")

    handles = [Line2D([], [], color="#7F7F7F", ls=":", lw=1.2,
                      label="页预算上限（Immediate−1）"),
               arm_handle("v4f@200"), arm_handle("v4t@100")]
    place_legend(fig, handles, ncol=3)
    tighten(fig)
    return save(fig, "fig3_throttling_attribution")


# --------------------------------------------------------------------------- figure 4
#: Arms drawn in the core scatter: byte ratio <= 3 keeps the two traditional triggers out
#: (periodic_k5 24.7x, event_maxwait 8.9x); they appear in panel (b) instead.
CORE_BYTE_LIMIT = 3.0
BAR_ORDER = ("normal_immediate", "dedup_immediate", "periodic_k5", "event_maxwait",
             "fifo_cap_k1", "random_matched_rate", "slack_heuristic_cap",
             "slack_matched_rate", "v4f@200", "v4g@300", "v4s@300", "v4t@100", "v4u@100")
#: Compact academic names for the scatter annotations and the bar axis (no underscores).
SHORT = {"normal_immediate": "Immediate", "dedup_immediate": "Dedup",
         "periodic_k5": "Periodic-k5", "event_maxwait": "Event-MaxWait",
         "fifo_cap_k1": "FIFO-Cap", "random_matched_rate": "Random",
         "slack_heuristic_cap": "Slack-Cap", "slack_matched_rate": "Slack-Matched",
         "v4f@200": "v4f", "v4g@300": "v4g", "v4s@300": "v4s",
         "v4t@100": "v4t", "v4u@100": "v4u"}
#: Arms outside the six-arm style sheet: neutral black, distinct hollow shapes.
EXTRA_MARKER = {"dedup_immediate": "X", "fifo_cap_k1": "P", "random_matched_rate": "*",
                "v4g@300": "h", "v4u@100": "H"}
#: Label offsets in points, picked once so no two sit on a marker or on each other.
PARETO_OFFSET = {
    "v4t@100": (-4, -13, "right", "top"),
    "v4f@200": (2, 11, "left", "bottom"),
    "v4s@300": (10, -3, "left", "center"),
    "v4g@300": (-6, 10, "right", "bottom"),
    "v4u@100": (8, -6, "left", "top"),
    "random_matched_rate": (-10, -12, "right", "top"),
    "slack_matched_rate": (8, -10, "left", "top"),
    "slack_heuristic_cap": (0, 11, "center", "bottom"),
    "fifo_cap_k1": (-6, -13, "right", "top"),
    "normal_immediate": (10, -7, "left", "top"),
    "dedup_immediate": (9, 9, "left", "bottom"),
}


def figure_four(g_scaling: dict, waste: dict, efficiency: dict) -> list:
    del g_scaling, waste
    per_arm = efficiency["per_arm"]
    base = per_arm["normal_immediate"]["groups"]["all(8)"]
    rows = {}
    for arm, entry in per_arm.items():
        group = entry["groups"]["all(8)"]
        if group["broadcast_waste_ratio"] is None or not base["on_time"]:
            continue
        rows[arm] = {"waste": group["broadcast_waste_ratio"],
                     "ratio": group["on_time"] / base["on_time"],
                     "byte_ratio": group["spent_bytes"] / base["spent_bytes"],
                     "msg": group["messages_per_on_time"]}
    core = [arm for arm in BAR_ORDER
            if arm in rows and rows[arm]["byte_ratio"] <= CORE_BYTE_LIMIT]
    missing = [arm for arm in BAR_ORDER if arm not in rows or rows[arm]["msg"] is None]
    if missing:
        raise SystemExit("COMM_EFFICIENCY.json lacks arms/readings: %s" % missing)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    # (a) waste ratio vs relative on-time, core arms only, names on thin arrows.
    # The two reference lines are named in the figure legend, not inside the axes: every
    # in-axes strip at y=0.95 is occupied by a marker or an annotation arrow.
    ax = axes[0]
    reference_lines(ax)
    for arm in core:
        row = rows[arm]
        if arm in STYLE:
            style = STYLE[arm]
            color, marker, mfc = style["color"], style["marker"], style["mfc"]
            size = {"v4f@200": 60, "v4t@100": 42, "v4s@300": 36}.get(arm, 30)
            mew = style["mew"]
        else:
            color, marker = "#404040", EXTRA_MARKER[arm]
            mfc = color if marker == "*" else "white"
            size, mew = 30, 1.1
        ax.scatter([row["waste"]], [row["ratio"]], s=size,
                   color=mfc, edgecolor=color, linewidth=mew, marker=marker, zorder=5)
    for arm in core:
        row = rows[arm]
        dx, dy, ha, va = PARETO_OFFSET[arm]
        ax.annotate(SHORT[arm], xy=(row["waste"], row["ratio"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8, color="#1A1A1A",
                    ha=ha, va=va, zorder=6,
                    arrowprops=dict(arrowstyle="->", lw=0.5, color="#666666",
                                    shrinkA=1, shrinkB=3))
    ax.set_xlim(0.02, 0.35)
    ax.set_ylim(0.82, 1.09)
    ax.set_xlabel("(a) 广播超时浪费率")
    ax.set_ylabel("相对按时完成率比值（all(8)）")

    # (b) messages per on-time task for every arm that sends, log axis so the traditional
    # triggers' 10-36x redundancy is visible next to the 1.0-1.4x core arms.
    ax = axes[1]
    values = [rows[arm]["msg"] for arm in BAR_ORDER]
    colors = [STYLE[arm]["color"] if arm in STYLE else "#8C8C8C" for arm in BAR_ORDER]
    ypos = list(range(len(BAR_ORDER)))
    ax.barh(ypos, [value - 0.8 for value in values], left=0.8, height=0.62,
            color=colors, edgecolor=PARITY_BLACK, linewidth=0.5, zorder=3)
    ax.set_yticks(ypos, [SHORT[arm] for arm in BAR_ORDER])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.8, 60)
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(axis="x")
    ax.set_xlabel("(b) 每完成任务消息数（条/任务，对数轴）")

    handles = [Line2D([], [], color=PARITY_BLACK, lw=0.8, label="1.00（与 Immediate 持平）"),
               Line2D([], [], color=VERDICT_RED, lw=1.0, ls="--", label="0.95（准时率红线）")]
    place_legend(fig, handles, ncol=2)
    tighten(fig)
    return save(fig, "fig4_efficiency_pareto_all8")


# --------------------------------------------------------------------------- figure 5
def figure_five(g_scaling: dict, waste: dict) -> list:
    """Absolute on-time rate (on_time / goals), the <=1 companion to fig1(b)'s ratio.

    No 0.95 reference line here: the registered criterion is relative to Immediate, so it
    has no meaning on the absolute axis.
    """
    del waste
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    x = goal_ticks(ax)
    for arm in ORDER:
        plot_arm(ax, x, series(g_scaling, arm, "q"), arm)
    ax.set_ylim(0.25, 1.0)
    ax.set_xlabel("任务规模 𝐺")
    ax.set_ylabel("绝对按时完成率（按时数 / 目标总数）")
    place_legend(fig, [arm_handle(arm) for arm in ORDER], ncol=3)
    tighten(fig)
    return save(fig, "fig5_absolute_on_time")


# --------------------------------------------------------------------------- manifest
FIGURES = {
    1: ("fig1_g_scaling_efficiency", "图 1　G 扩展：字节削减、按时比、消息/任务与浪费率",
        figure_one),
    2: ("fig2_r6_phase_transition", "图 2　R=6 相变（G*=36）与 R=6/R=12 比值对照",
        figure_two),
    3: ("fig3_throttling_attribution", "图 3　限流归因：页预算 vs 网络自身（G≥36）",
        figure_three),
    4: ("fig4_efficiency_pareto_all8", "图 4　效率—按时权衡与消息/任务量级（all(8)）",
        "efficiency"),
    5: ("fig5_absolute_on_time", "图 5　绝对按时完成率（按时数/目标总数）", figure_five),
}
MANIFEST_START = "<!-- FIGURE-MANIFEST:START -->"
MANIFEST_END = "<!-- FIGURE-MANIFEST:END -->"


def png_size(path: Path) -> tuple:
    import struct
    with Path(path).open("rb") as handle:
        handle.seek(16)
        width = struct.unpack(">I", handle.read(4))[0]
        height = struct.unpack(">I", handle.read(4))[0]
    return width, height


def build_manifest(sources: list) -> str:
    """Describe every figure pair currently in ``execution/figures``.

    Scanning the directory rather than the list just written matters: ``--only 2`` would
    otherwise replace the manifest with a single row and silently drop the other three.
    """
    titles = {slug: title for _number, (slug, title, _builder) in FIGURES.items()}
    lines = [MANIFEST_START,
             "## 图片清单（`analysis/plot_paper_figures.py` 生成，零重跑）", "",
             "IEEE 风格：西文 Times New Roman、中文 SimSun、数学符号 STIX，字体已嵌入"
             "（`pdf.fonttype = 42`）；每张图同时给出 300 dpi PNG 与矢量 PDF。"
             "四边封闭坐标轴、刻度朝内、无图内标题/气泡/点值标注；子图编号写在横轴标签内，"
             "图例为细黑框、置于图上方；横轴为等距类别刻度（12…72）。", "",
             "| 图 | 文件 | 像素 | sha256（前 16 位） |", "|---|---|---|---|"]
    for png in sorted(p for p in OUT.glob("*.png")):
        pdf = png.with_suffix(".pdf")
        if not pdf.exists():
            continue
        width, height = png_size(png)
        lines.append("| %s | `%s`（300 dpi）<br>`%s`（矢量） | %d × %d | `%s` |"
                     % (titles.get(png.stem, png.stem),
                        png.relative_to(BASE).as_posix(),
                        pdf.relative_to(BASE).as_posix(), width, height,
                        sha256(png)[:16]))
    lines += ["", "读图口径：", "",
              "* 图例命名与表格代码名的对应：Immediate=`normal_immediate`，"
              "Slack-Cap=`slack_heuristic_cap`，Slack-Matched=`slack_matched_rate`，"
              "Pure-Policy=`v4f@200`，Ours-Joint=`v4s@300`，Ours-Frozen=`v4t@100`。",
              "* 图 1(a) 不画 Slack-Matched（其页数按定义对齐 v4t，曲线与 v4t/v4f 几乎重合）；"
              "图 1(a)(b) 中 Immediate 只作 y=0 / y=1.0 的灰色点划参考线，不画数据点。"
              "图 1(b) 的 0.95 红线上方为极淡绿底衬（达标区）。",
              "* 图 1/图 2/图 5 每档 25 个世界（R6_N6 与 R12_N6 跑 8 场景，其余三对跑 3 锚定场景）；"
              "图 2(a) 的 R=6 子集为 11 个世界。",
              "* 图 5 是绝对按时完成率（按时数/目标总数，恒 ≤ 1），与图 1(b) 的相对比值对照；"
              "注册判据 0.95 是相对 Immediate 的，故图 5 不画该红线。",
              "* 图 2(b) 按时比与字节比都是相对同组 Immediate 的无量纲比值，共用同一纵轴；"
              "实线为按时比，虚线为字节比。",
              "* 图 4(a) 只画字节比 ≤ 3 的 11 臂；`periodic_k5`（24.7×）与 `event_maxwait`"
              "（8.9×）在图 4(b) 的对数轴条形图中给出。`always_hold` 从不发送、无消息/任务读数，"
              "不入图 4(b)。",
              "* 旧版仪表盘风格图隔离在 `execution/figures/_dashboard_v1/`，未删除。", "",
              "输入文件（只读，本轮未改动）：", "", "| 输入 | sha256（前 16 位） |",
              "|---|---|"]
    for path in sources:
        lines.append("| `%s` | `%s` |"
                     % (path.relative_to(BASE).as_posix(), sha256(path)[:16]))
    lines += ["", "复现：", "", "```",
              "D:/project_env/python.exe analysis/plot_paper_figures.py",
              "```", MANIFEST_END]
    return "\n".join(lines) + "\n"


def append_manifest(text: str) -> None:
    doc = REPORT.read_text(encoding="utf-8")
    if MANIFEST_START in doc:
        head, rest = doc.split(MANIFEST_START, 1)
        _old, tail = rest.split(MANIFEST_END, 1)
        doc = head.rstrip() + "\n\n" + text + tail.lstrip("\n").lstrip()
    else:
        doc = doc.rstrip() + "\n\n---\n\n" + text
    REPORT.write_text(doc, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma separated figure numbers, e.g. 2,3")
    parser.add_argument("--no-manifest", action="store_true",
                        help="render figures without touching G_SCALING_EXPERIMENTS.md")
    args = parser.parse_args()
    sys.dont_write_bytecode = True

    require_fonts()
    configure()
    print("fonts: Times New Roman + SimSun + STIXGeneral (math)")

    g_scaling = load_json(G_SCALING)
    waste = load_json(WASTE)
    efficiency = load_json(EFFICIENCY)

    wanted = ([int(chunk) for chunk in args.only.split(",") if chunk.strip()]
              if args.only else sorted(FIGURES))
    written = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for number in wanted:
            if number not in FIGURES:
                raise SystemExit("unknown figure %d; available: %s"
                                 % (number, sorted(FIGURES)))
            slug, title, builder = FIGURES[number]
            paths = (figure_four(g_scaling, waste, efficiency) if builder == "efficiency"
                     else builder(g_scaling, waste))
            written.append((slug, title, paths))
            print("wrote", ", ".join(path.name for path in paths))
        missed = [str(item.message) for item in caught
                  if "missing from font" in str(item.message)]
    if missed:
        raise SystemExit("missing glyphs, fix the font before shipping: %s" % missed[:3])

    if not args.no_manifest:
        append_manifest(build_manifest([G_SCALING, WASTE, EFFICIENCY]))
        print("manifest appended to", REPORT)


if __name__ == "__main__":
    main()