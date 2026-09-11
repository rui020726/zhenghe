# -*- coding: utf-8 -*-
"""
Nature-family submission figures for CTDE-PPO heterogeneous MRTA project.
Generates 4 figures from experiments A/B/D.

Output: ./paper_results/figures/
  fig1_expA_dlsr.png/svg/pdf   - 3x3 grid, 8 algos DL-SR comparison
  fig2_expA_advantage.png/svg/pdf  - Ours vs Dist-CBBA / Dist-GC pp advantage
  fig3_expB_ablation.png/svg/pdf   - 5 ablation variants
  fig4_expD_comm_range.png/svg/pdf - DL-SR vs comm_range S-curve
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── MANDATORY: editable SVG text ──────────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'

# ── Layout & style (Nature compact) ───────────────────────────────────────
plt.rcParams['font.size'] = 7
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['legend.frameon'] = False
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['xtick.labelsize'] = 6.5
plt.rcParams['ytick.labelsize'] = 6.5

# ── Color palette (NMI pastel + signal accents) ───────────────────────────
PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "red_strong":     "#B64342",
    "neutral_light":  "#CFCECE",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
    "neutral_black":  "#272727",
    "gold":           "#FFD700",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
    # distributed variants - lighter shades
    "dist_cbba":      "#9EC4E0",   # light blue
    "dist_gc":        "#C4A8D8",   # light violet
    # NMI pastel family for unified look
    "baseline_dark":  "#484878",
    "baseline_mid":   "#7884B4",
    "baseline_soft":  "#B4C0E4",
}

# Algorithm colors - 8 algos
ALGO_COLORS = {
    "Hungarian":             PALETTE["neutral_mid"],
    "Greedy-Nearest":        PALETTE["baseline_mid"],
    "Greedy-Priority":       PALETTE["baseline_soft"],
    "Greedy-Coalition":      PALETTE["blue_secondary"],
    "CBBA":                  PALETTE["blue_main"],
    "Dist-CBBA":             PALETTE["dist_cbba"],
    "Dist-Greedy-Coalition": PALETTE["dist_gc"],
    "Ours":                  PALETTE["red_strong"],
}

ALGO_HATCHES = {
    "Hungarian":             "",
    "Greedy-Nearest":        "",
    "Greedy-Priority":       "",
    "Greedy-Coalition":      "",
    "CBBA":                  "",
    "Dist-CBBA":             "//",
    "Dist-Greedy-Coalition": "\\\\",
    "Ours":                  "",
}

SCALES = ["small", "medium", "large"]
LOADS = [(8, 15), (20, 30), (40, 50)]
LOAD_LABELS = ["8-15 tasks", "20-30 tasks", "40-50 tasks"]

ALGO_ORDER = [
    "Hungarian", "Greedy-Nearest", "Greedy-Priority", "Greedy-Coalition",
    "CBBA", "Dist-CBBA", "Dist-Greedy-Coalition", "Ours"
]
ALGO_SHORT = {
    "Hungarian":             "Hun",
    "Greedy-Nearest":        "GN",
    "Greedy-Priority":       "GP",
    "Greedy-Coalition":      "GC",
    "CBBA":                  "CBBA",
    "Dist-CBBA":             "D-CBBA",
    "Dist-Greedy-Coalition": "D-GC",
    "Ours":                  "Ours",
}

FIG_DIR = "./paper_results/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_cell(data, scale, task_range, algo):
    """Fetch one (scale, task_range, algo) entry from expA data."""
    for e in data:
        if e["scale"] == scale and tuple(e["task_range"]) == task_range and e["algorithm"] == algo:
            return e
    return None


# ──────────────────────────────────────────────────────────────────────────
# Figure 1: 3x3 grid of 8-algo DL-SR bar charts (Experiment A)
# ──────────────────────────────────────────────────────────────────────────
def make_fig1(data):
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.8))  # ~183mm square-ish
    fig.subplots_adjust(hspace=0.55, wspace=0.35,
                        left=0.07, right=0.98, top=0.93, bottom=0.08)

    bar_width = 0.85
    x = np.arange(len(ALGO_ORDER))

    for i, scale in enumerate(SCALES):
        for j, load in enumerate(LOADS):
            ax = axes[i, j]

            means, stds = [], []
            for algo in ALGO_ORDER:
                e = get_cell(data, scale, load, algo)
                means.append(e["dl_sr_mean"] * 100 if e else 0)
                stds.append(e["dl_sr_std"] * 100 if e else 0)

            # Colors with Ours highlighted
            colors = [ALGO_COLORS[a] for a in ALGO_ORDER]
            hatches = [ALGO_HATCHES[a] for a in ALGO_ORDER]

            bars = ax.bar(x, means, width=bar_width,
                          color=colors, edgecolor='black', linewidth=0.5,
                          yerr=stds, error_kw={'elinewidth': 0.6, 'capthick': 0.6, 'capsize': 1.5})

            # Apply hatches to distributed variants
            for bar, h in zip(bars, hatches):
                if h:
                    bar.set_hatch(h)
                    bar.set_edgecolor('#444444')

            # Highlight Ours bar with thicker edge
            bars[-1].set_edgecolor(PALETTE["red_strong"])
            bars[-1].set_linewidth(1.2)

            ax.set_xticks(x)
            ax.set_xticklabels([ALGO_SHORT[a] for a in ALGO_ORDER], rotation=45, ha='right', fontsize=5.8)
            ax.set_ylim(0, 105)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_ylabel('DL-SR (%)' if j == 0 else '')

            if i == 0:
                ax.set_title(LOAD_LABELS[j], fontsize=7.5, fontweight='bold', pad=4)
            if j == 0:
                ax.text(-0.35, 0.5, scale.capitalize(), transform=ax.transAxes,
                        fontsize=8, fontweight='bold', ha='center', va='center', rotation=90)

            # Panel label
            if i == 0 and j == 0:
                ax.text(-0.30, 1.18, 'a', transform=ax.transAxes,
                        fontsize=10, fontweight='bold', ha='left', va='bottom')

    # Shared legend at bottom
    from matplotlib.patches import Patch
    legend_elements = []
    for algo in ALGO_ORDER:
        legend_elements.append(Patch(facecolor=ALGO_COLORS[algo],
                                     edgecolor='black' if algo != "Ours" else PALETTE["red_strong"],
                                     hatch=ALGO_HATCHES[algo],
                                     linewidth=1.0 if algo == "Ours" else 0.5,
                                     label=algo))
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.005))

    # Save in multiple formats
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(f"{FIG_DIR}/fig1_expA_dlsr.{ext}", dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] fig1 saved -> {FIG_DIR}/fig1_expA_dlsr.{{png,svg,pdf}}")


# ──────────────────────────────────────────────────────────────────────────
# Figure 2: Ours vs Dist-CBBA / Dist-GC advantage (Experiment A)
# ──────────────────────────────────────────────────────────────────────────
def make_fig2(data):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))  # double-column wide, short

    cell_labels = []
    adv_dcba = []
    adv_dgc = []

    for scale in SCALES:
        for load in LOADS:
            ours = get_cell(data, scale, load, "Ours")
            dcbba = get_cell(data, scale, load, "Dist-CBBA")
            dgc = get_cell(data, scale, load, "Dist-Greedy-Coalition")
            cell_labels.append(f"{scale[0].upper()}\n{load[0]}-{load[1]}")
            adv_dcba.append((ours["dl_sr_mean"] - dcbba["dl_sr_mean"]) * 100)
            adv_dgc.append((ours["dl_sr_mean"] - dgc["dl_sr_mean"]) * 100)

    x = np.arange(len(cell_labels))
    w = 0.38

    bars1 = ax.bar(x - w/2, adv_dcba, width=w,
                   color=PALETTE["dist_cbba"], edgecolor='black', linewidth=0.6,
                   label='Ours vs Dist-CBBA')
    bars2 = ax.bar(x + w/2, adv_dgc, width=w,
                   color=PALETTE["dist_gc"], edgecolor='black', linewidth=0.6,
                   label='Ours vs Dist-GC')

    # Mean lines
    mean_dcba = np.mean(adv_dcba)
    mean_dgc = np.mean(adv_dgc)
    ax.axhline(mean_dcba, color=PALETTE["blue_main"], linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(mean_dgc, color=PALETTE["violet"], linestyle='--', linewidth=0.8, alpha=0.7)

    # Annotate means at right edge
    ax.text(len(cell_labels) - 0.3, mean_dcba + 1.5,
            f'mean +{mean_dcba:.1f}pp', fontsize=6, color=PALETTE["blue_main"], fontweight='bold')
    ax.text(len(cell_labels) - 0.3, mean_dgc - 3.5,
            f'mean +{mean_dgc:.1f}pp', fontsize=6, color=PALETTE["violet"], fontweight='bold')

    # Value labels on bars
    for bars, vals in [(bars1, adv_dcba), (bars2, adv_dgc)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.8 if v > 0 else v - 2.5,
                    f'+{v:.0f}' if v > 0 else f'{v:.0f}',
                    ha='center', va='bottom' if v > 0 else 'top',
                    fontsize=5.2, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(cell_labels, fontsize=6)
    ax.set_xlabel('Scale  /  Task load  (S=small, M=medium, L=large)', fontsize=7, labelpad=4)
    ax.set_ylabel('DL-SR advantage (pp)', fontsize=7)
    ax.set_ylim(0, max(max(adv_dcba), max(adv_dgc)) * 1.15)
    ax.legend(loc='upper right', fontsize=6.5, ncol=1, frameon=False)

    # Zero baseline
    ax.axhline(0, color='black', linewidth=0.5)

    ax.text(-0.06, 1.05, 'b', transform=ax.transAxes,
            fontsize=10, fontweight='bold', ha='left', va='bottom')

    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(f"{FIG_DIR}/fig2_expA_advantage.{ext}", dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] fig2 saved -> {FIG_DIR}/fig2_expA_advantage.{{png,svg,pdf}}")


# ──────────────────────────────────────────────────────────────────────────
# Figure 3: Ablation bar chart (Experiment B)
# ──────────────────────────────────────────────────────────────────────────
def make_fig3(data):
    fig, ax = plt.subplots(figsize=(5.0, 3.0))  # single-column-ish

    variants = ["baseline", "-gossip", "-tidal", "-soft-deadline", "+tight-deadline"]
    display_names = ["Baseline\n(full)", "-Gossip", "-Tidal", "-Soft-\ndeadline", "+Tight\ndeadline"]
    means = []
    stds = []
    deltas = []
    baseline_val = None

    for v in variants:
        entry = next(e for e in data if e["variant"] == v)
        means.append(entry["dl_sr_mean"] * 100)
        stds.append(entry["dl_sr_std"] * 100)
        if v == "baseline":
            baseline_val = entry["dl_sr_mean"] * 100
            deltas.append(0.0)
        else:
            deltas.append(entry["dl_sr_mean"] * 100 - baseline_val)

    # Color code: baseline dark, -gossip red (critical), others neutral
    colors = []
    for v in variants:
        if v == "baseline":
            colors.append(PALETTE["neutral_dark"])
        elif v == "-gossip":
            colors.append(PALETTE["red_strong"])
        elif v == "-tidal":
            colors.append(PALETTE["neutral_mid"])
        elif v == "-soft-deadline":
            colors.append(PALETTE["neutral_light"])
        elif v == "+tight-deadline":
            colors.append(PALETTE["teal"])

    x = np.arange(len(variants))
    bars = ax.bar(x, means, width=0.65,
                  color=colors, edgecolor='black', linewidth=0.6,
                  yerr=stds, error_kw={'elinewidth': 0.7, 'capthick': 0.7, 'capsize': 3})

    # Highlight -gossip with thicker edge
    bars[1].set_edgecolor(PALETTE["red_strong"])
    bars[1].set_linewidth(1.2)

    # Annotate delta on each bar
    for bar, d in zip(bars, deltas):
        y_top = bar.get_height()
        if abs(d) < 0.05:
            label = 'ref'
            color = 'black'
            y_pos = y_top + 2
        else:
            sign = '+' if d > 0 else ''
            label = f'{sign}{d:.2f}pp'
            color = PALETTE["red_strong"] if d < -5 else ('black' if abs(d) < 2 else PALETTE["neutral_dark"])
            y_pos = y_top + 2
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                ha='center', va='bottom', fontsize=6.2, fontweight='bold', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=6.5)
    ax.set_ylabel('DL-SR (%)', fontsize=7)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Ablation variant  (config: large scale, 20-30 tasks)', fontsize=6.5, labelpad=4)

    # Add a horizontal line for baseline
    ax.axhline(baseline_val, color=PALETTE["neutral_dark"], linestyle=':', linewidth=0.6, alpha=0.6)

    ax.text(-0.10, 1.05, 'c', transform=ax.transAxes,
            fontsize=10, fontweight='bold', ha='left', va='bottom')

    # Caption-style annotation
    ax.text(0.5, -0.30,
            'Reference: baseline DL-SR = {:.1f}%'.format(baseline_val),
            transform=ax.transAxes, ha='center', fontsize=5.8,
            color=PALETTE["neutral_mid"], style='italic')

    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(f"{FIG_DIR}/fig3_expB_ablation.{ext}", dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] fig3 saved -> {FIG_DIR}/fig3_expB_ablation.{{png,svg,pdf}}")


# ──────────────────────────────────────────────────────────────────────────
# Figure 4: DL-SR vs comm_range S-curve (Experiment D)
# ──────────────────────────────────────────────────────────────────────────
def make_fig4(data):
    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    comm_ranges = [e["comm_range"] for e in data]
    dlsr_mean = [e["dl_sr_mean"] * 100 for e in data]
    dlsr_std = [e["dl_sr_std"] * 100 for e in data]
    expired = [e["expired_mean"] * 100 for e in data]
    expired_std = [e["expired_std"] * 100 for e in data]

    # Primary axis: DL-SR
    line1 = ax.plot(comm_ranges, dlsr_mean, marker='o', markersize=6,
                    color=PALETTE["blue_main"], linewidth=1.8,
                    label='DL-SR', zorder=3)
    ax.fill_between(comm_ranges,
                    [m - s for m, s in zip(dlsr_mean, dlsr_std)],
                    [m + s for m, s in zip(dlsr_mean, dlsr_std)],
                    color=PALETTE["blue_main"], alpha=0.15, zorder=1)

    # Error bars
    ax.errorbar(comm_ranges, dlsr_mean, yerr=dlsr_std,
                fmt='none', color=PALETTE["blue_main"], elinewidth=0.7,
                capthick=0.7, capsize=2.5, zorder=2)

    # Secondary axis: Expired rate
    ax2 = ax.twinx()
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(False)
    line2 = ax2.plot(comm_ranges, expired, marker='s', markersize=5,
                     color=PALETTE["red_strong"], linewidth=1.5,
                     linestyle='--', label='Expired rate', zorder=3)
    ax2.fill_between(comm_ranges,
                     [m - s for m, s in zip(expired, expired_std)],
                     [m + s for m, s in zip(expired, expired_std)],
                     color=PALETTE["red_strong"], alpha=0.12, zorder=1)

    # Annotate saturation point
    sat_idx = comm_ranges.index(0.3)
    ax.annotate('Saturation\npoint',
                xy=(0.3, dlsr_mean[sat_idx]),
                xytext=(0.42, dlsr_mean[sat_idx] - 12),
                fontsize=6, color=PALETTE["neutral_dark"],
                arrowprops=dict(arrowstyle='->', color=PALETTE["neutral_dark"], lw=0.6))

    # Annotate critical transition 0->0.1
    ax.annotate('+31pp jump',
                xy=(0.1, dlsr_mean[1]),
                xytext=(0.15, dlsr_mean[1] - 18),
                fontsize=6, color=PALETTE["blue_main"], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=PALETTE["blue_main"], lw=0.6))

    ax.set_xlabel('Communication range (comm_range)', fontsize=7, labelpad=3)
    ax.set_ylabel('DL-SR (%)', fontsize=7, color=PALETTE["blue_main"])
    ax2.set_ylabel('Expired rate (%)', fontsize=7, color=PALETTE["red_strong"])
    ax.tick_params(axis='y', colors=PALETTE["blue_main"])
    ax2.tick_params(axis='y', colors=PALETTE["red_strong"])
    ax.set_xticks(comm_ranges)
    ax.set_ylim(0, 105)
    ax2.set_ylim(0, 60)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax2.set_yticks([0, 15, 30, 45, 60])

    # Combined legend
    lines = line1 + line2
    labels = ['DL-SR (left)', 'Expired rate (right)']
    ax.legend(lines, labels, loc='lower right', fontsize=6.5, frameon=False)

    # Config note
    ax.text(0.5, -0.28,
            'Config: large scale, 20-30 tasks; default comm_range = 0.28',
            transform=ax.transAxes, ha='center', fontsize=5.8,
            color=PALETTE["neutral_mid"], style='italic')

    ax.text(-0.10, 1.05, 'd', transform=ax.transAxes,
            fontsize=10, fontweight='bold', ha='left', va='bottom')

    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(f"{FIG_DIR}/fig4_expD_comm_range.{ext}", dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] fig4 saved -> {FIG_DIR}/fig4_expD_comm_range.{{png,svg,pdf}}")


# ──────────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    expA = load_json("./paper_results/expA_distributed.json")
    expB = load_json("./paper_results/expB_ablation.json")
    expD = load_json("./paper_results/expD_comm_range.json")
    print(f"  expA: {len(expA)} entries")
    print(f"  expB: {len(expB)} entries")
    print(f"  expD: {len(expD)} entries")
    print()

    print("Generating figures...")
    make_fig1(expA)
    make_fig2(expA)
    make_fig3(expB)
    make_fig4(expD)

    print()
    print("All figures generated in:", FIG_DIR)


if __name__ == "__main__":
    main()
