"""Figure 19: the classical arms against Ours on the two scale grids (G >= 48).

Reads ``execution/classic/SCALE_RESULTS.json`` only; no world is opened.  Two rows of panels:
relative on-time (top) and bytes ratio (bottom); left column the G sweep, right column the deep
G x R x S grid.  Ours keeps the revision's contract: deep red, solid, filled circle.

    D:/project_env/python.exe analysis/plot_classic_scale.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "analysis"))

from plot_paper_ours import configure, require_fonts  # noqa: E402

IN = BASE / "execution/classic/SCALE_RESULTS.json"
OUT = BASE / "execution/figures"
STEM = "fig19_scale_classical_comparison"

G_GOALS = (48, 54, 60, 72)
DEEP_GOALS = (72, 84, 96, 108, 120, 144)
# LinUCB-Bandit is deliberately not drawn (author decision, R81): its readings stay in
# table 4 and in sections 9/10 of CLASSIC_ALGORITHM_COMPARISON.md, so the tie with Ours is
# still on record even though the figures carry the other arms only.
ORDER = ("normal_immediate", "red_aqm", "token_bucket", "ahp_utility", "v4f@200")
STYLE = {
    "normal_immediate": dict(color="#636363", ls="-.", marker="^", mfc="white",
                             lw=1.1, ms=4.5, mew=1.0, zorder=3),
    "red_aqm": dict(color="#2C7FB8", ls="--", marker="s", mfc="white",
                    lw=1.1, ms=4.2, mew=1.0, zorder=4),
    "token_bucket": dict(color="#2CA02C", ls=":", marker="D", mfc="white",
                         lw=1.3, ms=4.2, mew=1.0, zorder=4),
    "ahp_utility": dict(color="#8E44AD", ls="-.", marker="p", mfc="white",
                        lw=1.1, ms=4.6, mew=1.0, zorder=4),
    "linucb_bandit": dict(color="#17BECF", ls="--", marker="*", mfc="white",
                          lw=1.1, ms=5.4, mew=1.0, zorder=5),
    "v4f@200": dict(color="#C0392B", ls="-", marker="o", mfc="#C0392B",
                    lw=1.7, ms=4.8, mew=1.0, zorder=6),
}
LABEL = {"normal_immediate": "Immediate (全量广播基线)", "red_aqm": "RED-AQM",
         "token_bucket": "Token-Bucket", "ahp_utility": "AHP-Utility",
         "linucb_bandit": "LinUCB-Bandit", "v4f@200": "本文方法 (Ours)"}


def series(table, grid, goals, arm, key):
    out = []
    for goal in goals:
        record = (table[grid].get(str(goal)) or table[grid].get(goal) or {}).get(arm)
        out.append(np.nan if record is None or record.get(key) is None else record[key])
    return np.asarray(out, dtype="float64")


def main():
    require_fonts()
    configure()
    table = json.loads(IN.read_text(encoding="utf-8"))["table"]
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 4.6))
    panels = (("g_sweep", G_GOALS, "rel_q", "相对按时率（vs Immediate）"),
              ("deep", DEEP_GOALS, "rel_q", "相对按时率（vs Immediate）"),
              ("g_sweep", G_GOALS, "byte_ratio", "字节比（vs Immediate）"),
              ("deep", DEEP_GOALS, "byte_ratio", "字节比（vs Immediate）"))
    titles = ("(a) G 扫描：按时", "(b) 深档 G×R×S：按时",
              "(c) G 扫描：字节", "(d) 深档 G×R×S：字节")
    handles = []
    for index, (grid, goals, key, ylabel) in enumerate(panels):
        axis = axes[index // 2][index % 2]
        plotted = []
        for arm in ORDER:
            values = series(table, grid, goals, arm, key)
            plotted.append(values)
            line, = axis.plot(goals, values, label=LABEL[arm], **STYLE[arm])
            if index == 0:
                handles.append(line)
        finite = np.concatenate([values[np.isfinite(values)] for values in plotted])
        if finite.size:
            low, high = float(finite.min()), float(finite.max())
            span = max(high - low, 0.2)
            axis.set_ylim(low - 0.12 * span, high + 0.20 * span)
        axis.set_xlabel("任务数 G")
        axis.set_ylabel(ylabel)
        axis.set_xticks(list(goals))
        axis.set_xlim(min(goals) - 2, max(goals) + 2)
        axis.text(0.02, 0.97, titles[index], transform=axis.transAxes,
                  ha="left", va="top", fontsize=8.5)
        if key == "rel_q":
            axis.axhline(1.0, color="#999999", lw=0.6, ls=":", zorder=1)
        else:
            axis.axhline(0.85, color="#999999", lw=0.6, ls=":", zorder=1)
    figure.legend(handles=handles, labels=[LABEL[arm] for arm in ORDER],
                  loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02),
                  frameon=True, fontsize=8.2)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / (STEM + ".png")
    pdf = OUT / (STEM + ".pdf")
    figure.savefig(png)
    figure.savefig(pdf)
    plt.close(figure)
    print("wrote %s" % png)
    print("wrote %s" % pdf)


if __name__ == "__main__":
    main()
