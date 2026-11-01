"""Offline loss-curve plots for recovery_09b training telemetry.

Reads only completed telemetry JSON files; never imports project code, torch or the env.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"D:\code\整合\zhenghe-round42\runs\stage5b_pure_comm_convergence_20261027\recovery_09b\revision_01")
TELEMETRY = ROOT / "execution" / "recovery" / "telemetry"
OUT = Path(__file__).resolve().parent
ARMS = ("T1", "T2")
SEEDS = (91001, 91002, 91003)
COLORS = {"T1": "#1f77b4", "T2": "#d62728"}
STYLES = {91001: "-", 91002: "--", 91003: ":"}
METRICS = (
    ("policy_surrogate", "policy surrogate"),
    ("value_loss_task", "value loss (task)"),
    ("value_loss_cost", "value loss (cost)"),
    ("entropy", "policy entropy"),
    ("approx_kl", "approx KL"),
    ("gradient_norm_mean", "gradient norm (mean)"),
)
ROLLING = 25


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rolling(values, window):
    out, acc = [], 0.0
    for i, value in enumerate(values):
        acc += value
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def main():
    files = [TELEMETRY / ("%s_s%d.json" % (arm, seed)) for arm in ARMS for seed in SEEDS]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise SystemExit("missing telemetry: %r" % missing)

    series = {}
    inputs = []
    rows = []
    for path in files:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        arm, seed = payload["arm"], int(payload["training_seed"])
        updates = sorted(payload["updates"], key=lambda u: int(u["update_index"]))
        episodes = [int(u["episode_indices"][0]) for u in updates]
        series[(arm, seed)] = (episodes, updates)
        inputs.append({"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        for u in updates:
            rows.append({
                "arm": arm, "seed": seed, "update_index": int(u["update_index"]),
                "episode_start": int(u["episode_indices"][0]),
                **{name: u.get(name) for name, _ in METRICS},
            })

    (OUT / "loss_updates.csv").write_text("", encoding="utf-8")
    with (OUT / "loss_updates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.0), constrained_layout=True)
    for ax, (metric, label) in zip(axes.ravel(), METRICS):
        for arm in ARMS:
            for seed in SEEDS:
                episodes, updates = series[(arm, seed)]
                values = [u.get(metric) for u in updates]
                xs = [x for x, v in zip(episodes, values) if v is not None and math.isfinite(v)]
                ys = [v for v in values if v is not None and math.isfinite(v)]
                if not ys:
                    continue
                ax.plot(xs, ys, color=COLORS[arm], linestyle=STYLES[seed],
                        linewidth=0.5, alpha=0.18)
                ax.plot(xs, rolling(ys, ROLLING), color=COLORS[arm], linestyle=STYLES[seed],
                        linewidth=1.5, alpha=0.95,
                        label="%s/%d" % (arm, seed) if metric == "policy_surrogate" else None)
        ax.set_title(label)
        ax.set_xlabel("global episode index")
        ax.grid(True, alpha=0.25)
        if metric in ("value_loss_cost", "approx_kl", "gradient_norm_mean"):
            ax.set_yscale("log")
    axes[0, 0].legend(fontsize=8, ncol=2, loc="best")
    fig.suptitle("recovery_09b PPO diagnostics by update (thin = raw, thick = %d-update mean)" % ROLLING)
    for extension in ("png", "svg"):
        fig.savefig(OUT / ("loss_curves." + extension), dpi=170)
    plt.close(fig)

    summary = {
        "schema": "r9b-loss-curves/1", "generated_utc": utc(),
        "root": str(ROOT), "rolling_window_updates": ROLLING,
        "inputs": inputs,
        "runs": [
            {"arm": arm, "seed": seed,
             "updates": len(series[(arm, seed)][1]),
             "first_episode": series[(arm, seed)][0][0],
             "last_episode": series[(arm, seed)][0][-1],
             "final_rolling": {name: (rolling([u.get(name) for u in series[(arm, seed)][1]],
                                              ROLLING)[-1])
                               for name, _ in METRICS}}
            for arm in ARMS for seed in SEEDS
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print("wrote", OUT / "loss_curves.png", "rows", len(rows))


if __name__ == "__main__":
    main()
