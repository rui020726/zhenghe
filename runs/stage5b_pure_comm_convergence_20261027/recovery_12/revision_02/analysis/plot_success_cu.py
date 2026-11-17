"""Figure 20: absolute success rate and channel uses per world on ONE G axis, 24..108.

The two world designs are joined on a single numeric axis because the paper asks for one
curve per arm over the whole range: G = 24..72 comes from the G sweep (25 worlds per goal,
N varies with the (R, N) pair) and G = 84..108 from the deep G x R x S grid (27 worlds per
goal, N = 6).  The regime switch is drawn as a dotted vertical rule between 72 and 84 and
labelled in both panels, so the join is visible rather than silent; G = 72 is taken from the
sweep, whose scene rule matches the lower half of the range.

Top panel: success rate = on_time / generated_goals.  Middle panel: channel uses per world =
sum(emissions x subscribers) / worlds.  Bottom panel: broadcast waste = distinct broadcast
targets that missed their deadline / distinct broadcast targets.  Five arms: Immediate, RED-AQM, Token-Bucket,
AHP-Utility and Ours (deep red, solid, filled circle).  LinUCB-Bandit stays out of the figure
by author decision; its readings remain in table 4 and sections 9/10 of the comparison doc.

Reads ``execution/classic/SCALE_RESULTS.json`` only.

    D:/project_env/python.exe analysis/plot_success_cu.py
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

from plot_classic_scale import LABEL, ORDER, STYLE  # noqa: E402
from plot_paper_ours import configure, require_fonts  # noqa: E402

IN = BASE / "execution/classic/SCALE_RESULTS.json"
OUT = BASE / "execution/figures"
STEM = "fig20_success_and_channel_uses"

SWEEP_GOALS = (24, 30, 36, 42, 48, 54, 60, 72)
DEEP_GOALS = (84, 96, 108)
GOALS = SWEEP_GOALS + DEEP_GOALS
SWITCH = 78.0


def series(table, arm, key):
    out = []
    for goal in GOALS:
        grid = "g_sweep" if goal <= 72 else "deep"
        record = (table[grid].get(str(goal)) or table[grid].get(goal) or {}).get(arm)
        out.append(np.nan if record is None or record.get(key) is None else record[key])
    return np.asarray(out, dtype="float64")


def main():
    require_fonts()
    configure()
    table = json.loads(IN.read_text(encoding="utf-8"))["table"]
    figure, axes = plt.subplots(3, 1, figsize=(7.16, 7.6))
    panels = (("q", "成功率（on-time / 生成目标）"),
              ("cu", "信道使用次数 / 世界"),
              ("waste", "广播超时浪费率（未按时 / 被广播目标）"))
    handles = []
    for index, (key, ylabel) in enumerate(panels):
        axis = axes[index]
        plotted = []
        for arm in ORDER:
            values = series(table, arm, key)
            plotted.append(values)
            line, = axis.plot(GOALS, values, label=LABEL[arm], **STYLE[arm])
            if index == 0:
                handles.append(line)
        finite = np.concatenate([values[np.isfinite(values)] for values in plotted])
        low, high = float(finite.min()), float(finite.max())
        span = max(high - low, 0.2)
        top = high + 0.24 * span
        axis.set_ylim(max(0.0, low - 0.10 * span), top)
        axis.set_xlim(22, 110)
        axis.set_xticks(list(GOALS))
        axis.set_xlabel("任务数 G")
        axis.set_ylabel(ylabel)
        axis.axvline(SWITCH, color="#8A8A8A", lw=0.9, ls=":", zorder=1)
        label_y = top - 0.05 * span
        axis.text(SWITCH + 0.8, label_y, "口径切换", ha="left", va="top", fontsize=7.6,
                  color="#5A5A5A")
        axis.text(24, label_y, "(" + "abc"[index] + ") G 扫描：25 世界/档",
                  ha="left", va="top", fontsize=7.6, color="#5A5A5A")
        axis.text(108, label_y, "深档：27 世界/档，N=6", ha="right", va="top", fontsize=7.6,
                  color="#5A5A5A")
    figure.legend(handles=handles, labels=[LABEL[arm] for arm in ORDER],
                  loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.005),
                  frameon=True, fontsize=8.2)
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / (STEM + ".png"))
    figure.savefig(OUT / (STEM + ".pdf"))
    plt.close(figure)
    print("wrote %s" % (OUT / (STEM + ".png")))
    print("wrote %s" % (OUT / (STEM + ".pdf")))


if __name__ == "__main__":
    main()
