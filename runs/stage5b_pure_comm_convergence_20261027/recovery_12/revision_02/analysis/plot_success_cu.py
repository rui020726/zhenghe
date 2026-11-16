"""Figure 20: absolute success rate and channel uses per world, G = 24..108.

Two regimes, drawn as two columns because their world designs differ and must not be joined
into one curve:

* left  -- the G sweep (25 worlds per goal; N varies with the (R, N) pair), G = 24..72;
* right -- the deep G x R x S grid (27 worlds per goal; N = 6), G = 72..108.

Top row: success rate = on_time / generated_goals.  Bottom row: channel uses per world =
sum(emissions x subscribers) / worlds.  Six arms: Immediate, the four named classical
algorithms and Ours (deep red, solid, filled circle).

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

G_GOALS = (24, 30, 36, 42, 48, 54, 60, 72)
DEEP_GOALS = (72, 84, 96, 108)


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
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 4.7))
    panels = (("g_sweep", G_GOALS, "q", "成功率（on-time / 生成目标）"),
              ("deep", DEEP_GOALS, "q", "成功率（on-time / 生成目标）"),
              ("g_sweep", G_GOALS, "cu", "信道使用次数 / 世界"),
              ("deep", DEEP_GOALS, "cu", "信道使用次数 / 世界"))
    titles = ("(a) G 扫描：成功率", "(b) 深档 G×R×S：成功率",
              "(c) G 扫描：信道使用", "(d) 深档 G×R×S：信道使用")
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
            axis.set_ylim(max(0.0, low - 0.12 * span), high + 0.20 * span)
        axis.set_xlabel("任务数 G")
        axis.set_ylabel(ylabel)
        axis.set_xticks(list(goals))
        axis.set_xlim(min(goals) - 2, max(goals) + 2)
        axis.text(0.02, 0.97, titles[index], transform=axis.transAxes,
                  ha="left", va="top", fontsize=8.5)
    figure.legend(handles=handles, labels=[LABEL[arm] for arm in ORDER],
                  loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02),
                  frameon=True, fontsize=8.2)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / (STEM + ".png"))
    figure.savefig(OUT / (STEM + ".pdf"))
    plt.close(figure)
    print("wrote %s" % (OUT / (STEM + ".png")))
    print("wrote %s" % (OUT / (STEM + ".pdf")))


if __name__ == "__main__":
    main()
