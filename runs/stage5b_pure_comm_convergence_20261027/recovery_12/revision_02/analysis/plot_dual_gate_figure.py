"""Figure 17: the dual-gate comparison and the ablation chain, from the archived JSON only.

Four panels, six arms, one denominator rule: every ratio divides by the same arm's
``normal_immediate`` over the fifteen training cells and their three anchor scenes (45 worlds).

    (a) per-cell on-time passes        out of 15, the clause both gates keep
    (b) total on-time relative to immediate
    (c) total bytes relative to immediate      (the relaxed gate's 0.85 line is drawn)
    (d) broadcast waste ratio                  (the relaxed gate's 0.10 line is drawn)

Style is the revision's IEEE contract: Times New Roman + SimSun + STIX with per-glyph fallback,
closed spines, inward ticks, light dashed grid, no in-figure titles, legend above the axes,
300 dpi PNG plus vector PDF with ``pdf.fonttype = 42``. All text in the canvas is English so the
figure can be dropped into a LaTeX paper without a CJK font; the captions live in the report.

    D:/project_env/python.exe analysis/plot_dual_gate_figure.py
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
STATE = BASE / "execution/relaxed_gate/DUAL_GATE.json"
OUT = BASE / "execution/figures"
NAME = "fig17_dual_gate_ablation"

ORDER = ("normal_immediate", "slack_heuristic_cap", "slack_matched_rate",
         "v4v_pure", "v4t@100", "v4f@200")
SHORT = {"normal_immediate": "Immediate", "slack_heuristic_cap": "Slack-Cap",
         "slack_matched_rate": "Slack-Matched", "v4v_pure": "v4v (price-decay)",
         "v4t@100": "v4t (net + quota)", "v4f@200": "v4f (pure net, ours)"}
STYLE = {
    "normal_immediate": dict(facecolor="#D9D9D9", edgecolor="#4D4D4D"),
    "slack_heuristic_cap": dict(facecolor="#FBE3C4", edgecolor="#B9770E"),
    "slack_matched_rate": dict(facecolor="#E7D8F3", edgecolor="#7D3C98"),
    "v4v_pure": dict(facecolor="#CDEFD9", edgecolor="#1E8449"),
    "v4t@100": dict(facecolor="#BFD9EC", edgecolor="#1F618D"),
    "v4f@200": dict(facecolor="#F5C6C0", edgecolor="#C0392B"),
}
VERDICT_RED = "#C0392B"
ZERO = "#000000"


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
        "xtick.labelsize": 8.0, "ytick.labelsize": 8.5, "legend.fontsize": 8.0,
        "axes.linewidth": 0.8, "axes.edgecolor": ZERO,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.color": ZERO, "ytick.color": ZERO,
        "axes.grid": True, "grid.color": "#CCCCCC", "grid.linewidth": 0.4,
        "grid.linestyle": "--", "axes.axisbelow": True,
        "axes.unicode_minus": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.frameon": True, "legend.fancybox": False,
        "legend.edgecolor": ZERO, "legend.facecolor": "#FFFFFF",
    })


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    sys.dont_write_bytecode = True
    require_fonts()
    configure()
    if not STATE.exists():
        raise SystemExit("missing %s; run analysis/dual_gate_report.py first" % STATE)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    arms = state["arms"]
    x = np.arange(len(ORDER))
    width = 0.62

    figure, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))

    def bars(axis, values, ylabel, empty="0"):
        for index, arm in enumerate(ORDER):
            axis.bar(x[index], values[arm], width=width, linewidth=0.8,
                     zorder=3, **STYLE[arm])
        axis.set_xticks(x)
        axis.set_xticklabels([SHORT[a] for a in ORDER], rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_xlim(-0.65, len(ORDER) - 0.35)

    # (a) per-cell on-time passes -- the clause both gates keep
    axis = axes[0, 0]
    passes = {a: arms[a]["on_time_passes"] for a in ORDER}
    bars(axis, passes, "Cells passing (/15)")
    axis.set_ylim(0, 16.5)
    axis.set_yticks(range(0, 16, 3))
    for index, arm in enumerate(ORDER):
        axis.annotate("%d/15" % passes[arm], (x[index], passes[arm]),
                      textcoords="offset points", xytext=(0, 3), ha="center",
                      fontsize=8, color=ZERO)
    axis.set_xlabel("(a) per-cell on-time passes")

    # (b) total on-time relative to immediate
    axis = axes[0, 1]
    ratio = {a: arms[a]["on_time_ratio"] for a in ORDER}
    bars(axis, ratio, "On-time / Immediate")
    axis.set_ylim(0.0, max(1.15, max(ratio.values()) + 0.05))
    axis.axhline(1.0, color="#636363", lw=0.8, ls=":", zorder=2)
    for index, arm in enumerate(ORDER):
        axis.annotate("%.3f" % ratio[arm], (x[index], ratio[arm]),
                      textcoords="offset points", xytext=(0, 3), ha="center",
                      fontsize=7.5, color=ZERO)
    axis.set_xlabel("(b) on-time relative to Immediate")

    # (c) total bytes relative to immediate, with the gate's 0.85 line
    axis = axes[1, 0]
    bytes_ratio = {a: arms[a]["bytes_ratio"] for a in ORDER}
    bars(axis, bytes_ratio, "Bytes / Immediate")
    axis.set_ylim(0.0, 1.13)
    axis.axhline(1.0, color="#636363", lw=0.8, ls=":", zorder=2)
    axis.axhline(0.85, color=VERDICT_RED, lw=0.9, ls="--", zorder=2)
    axis.annotate("gate 0.85", (0.05, 0.85), ha="left", va="bottom",
                  fontsize=7.5, color=VERDICT_RED)
    for index, arm in enumerate(ORDER):
        axis.annotate("%.3f" % bytes_ratio[arm], (x[index], bytes_ratio[arm]),
                      textcoords="offset points", xytext=(0, 3), ha="center",
                      fontsize=7.5, color=ZERO)
    axis.set_xlabel("(c) total bytes relative to Immediate")

    # (d) broadcast waste, with the gate's 0.10 line
    axis = axes[1, 1]
    waste = {a: (arms[a]["waste_ratio"] or 0.0) for a in ORDER}
    bars(axis, waste, "Broadcast waste")
    axis.set_ylim(0.0, max(0.34, max(waste.values()) + 0.05))
    axis.axhline(0.10, color=VERDICT_RED, lw=0.9, ls="--", zorder=2)
    axis.annotate("gate 0.10", (0.05, 0.10), ha="left", va="bottom",
                  fontsize=7.5, color=VERDICT_RED)
    for index, arm in enumerate(ORDER):
        axis.annotate("%.3f" % waste[arm], (x[index], waste[arm]),
                      textcoords="offset points", xytext=(0, 3), ha="center",
                      fontsize=7.5, color=ZERO)
    axis.set_xlabel("(d) broadcast waste (broadcast but not on time)")

    figure.tight_layout(h_pad=1.2, w_pad=1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / (NAME + ".png")
    pdf = OUT / (NAME + ".pdf")
    figure.savefig(png)
    figure.savefig(pdf)
    plt.close(figure)
    for path in (png, pdf):
        print("%s  sha256 %s" % (path.name, sha256(path)[:16]))


if __name__ == "__main__":
    main()
