"""ROUND30 §8 配对比较与描述性区间（只读落盘件，0 环境启动）。

- 每个训练 seed、每种模式单独比较：ΔY = NEW − OLD（按时任务数/episode）、
  ΔC = NEW − OLD（已计量载荷尝试字节/episode），逐场景改善/持平/退化，按时目标集合是否
  交换（从 lifecycle 的 final 记录取目标级结局）。
- 描述性区间按执行依据 §8.3 预先冻结：场景升序、以 16 场景为配对抽样单位、
  numpy PCG64 seed=20260930、10000 次重采样、linear 分位、所有比较复用同一份
  10000×16 抽样索引、估计量 values[indices].mean(axis=1)，并保留独立手算反例。
- 该区间只覆盖固定权重下的场景不确定性，不覆盖训练 seed 的总体不确定性；sampled 每场景
  仅一次，不据此估计其完整采样波动。三个 seed 分别展示，不池化、不挑最佳。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import r30_common as common
import r30_encoding as enc

DEV = tuple(int(s) for s in common.DEVELOPMENT_SCENARIOS)


def eval_row(arm: str, seed: int, mode: str, scenario: int) -> dict[str, Any]:
    path = common.cell_paths("eval", scenario, 0, tag=f"eval_{arm}_s{seed}_final_{mode}")["row"]
    return json.loads(path.read_text(encoding="utf-8"))


def reference_row(scenario: int) -> dict[str, Any]:
    return json.loads(common.cell_paths("reference", scenario, 1)["row"].read_text(
        encoding="utf-8"))


def late_row(scenario: int) -> dict[str, Any]:
    return json.loads(common.cell_paths("rule", scenario, 1,
                                        tag="late_discovery_gate")["row"].read_text(
                                            encoding="utf-8"))


def on_time_goal_ids(rp_path: Path) -> list[str]:
    """On-time target ids from the lifecycle log's per-target final records."""
    finals: dict[str, str] = {}
    with rp_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("event_type") != "final" or not str(row.get("target_id", "")).startswith(
                    "piloc"):
                continue
            finals[row["target_id"]] = row["status"]
    return sorted(target for target, status in finals.items() if status == "completed")


def paired_cells(arm: str, seed: int, mode: str) -> dict[int, dict[str, Any]]:
    cells: dict[int, dict[str, Any]] = {}
    for scenario in DEV:
        row = eval_row(arm, seed, mode, scenario)
        rp_path = Path(row["artefacts"]["lifecycle"])
        cells[scenario] = {
            "on_time": int(row["on_time"]),
            "spent_bytes": int(row["spent_bytes"]),
            "expired": int(row["expired"]),
            "emissions": int(row["emissions"]),
            "goal_ids": on_time_goal_ids(rp_path),
        }
    return cells


def index_matrix() -> np.ndarray:
    """The one shared 10000 x 16 resampling index (execution basis §8.3)."""
    return np.random.Generator(np.random.PCG64(20260930)).integers(0, 16, size=(10000, 16))


def percentile95(values: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def hand_check(indices: np.ndarray) -> dict[str, Any]:
    """An independent hand-computed counterexample for the resampling path.

    values = [1, 2, 3, ..., 16]; the mean of any resample of 16 draws with replacement is
    recomputed by an explicit slow loop for the first 5 index rows and compared with the
    vectorised path. This checks ``values[indices].mean(axis=1)`` itself, not the RNG.
    """
    values = np.arange(1, 17, dtype=float)
    vectorised = values[indices].mean(axis=1)
    slow = []
    for row in indices[:5]:
        total = 0.0
        for position in row:
            total += float(values[position])
        slow.append(total / 16.0)
    return {
        "vectorised_first_five": [round(float(value), 12) for value in vectorised[:5]],
        "hand_computed_first_five": [round(value, 12) for value in slow],
        "max_abs_gap": float(max(abs(a - b) for a, b in zip(vectorised[:5], slow))),
        "passed": all(abs(a - b) <= 1e-12 for a, b in zip(vectorised[:5], slow)),
        "note": "the check uses the round's real index matrix; a second layer of defence is that "
                "the same index matrix object is passed to every comparison in this script",
    }


def compare(seed: int, mode: str, indices: np.ndarray) -> dict[str, Any]:
    new = paired_cells("new", seed, mode)
    old = paired_cells("old", seed, mode)
    reference = {scenario: int(reference_row(scenario)["on_time"]) for scenario in DEV}
    late = {scenario: int(late_row(scenario)["on_time"]) for scenario in DEV}
    delta_y, delta_c = [], []
    per_scene = {}
    gained_total, lost_total = [], []
    for scenario in DEV:
        dy = new[scenario]["on_time"] - old[scenario]["on_time"]
        dc = new[scenario]["spent_bytes"] - old[scenario]["spent_bytes"]
        delta_y.append(float(dy))
        delta_c.append(float(dc))
        gained = sorted(set(new[scenario]["goal_ids"]) - set(old[scenario]["goal_ids"]))
        lost = sorted(set(old[scenario]["goal_ids"]) - set(new[scenario]["goal_ids"]))
        gained_total.extend(gained)
        lost_total.extend(lost)
        per_scene[str(scenario)] = {
            "delta_y": int(dy), "delta_c_bytes": int(dc),
            "new_on_time": new[scenario]["on_time"], "old_on_time": old[scenario]["on_time"],
            "reference_on_time": reference[scenario], "late_on_time": late[scenario],
            "new_bytes": new[scenario]["spent_bytes"], "old_bytes": old[scenario]["spent_bytes"],
            "gained_goal_ids": gained, "lost_goal_ids": lost,
            "task_direction": ("better" if dy > 0 else ("worse" if dy < 0 else "tied")),
            "cost_direction": ("cheaper" if dc < 0 else ("costlier" if dc > 0 else "tied")),
            "cheaper_but_lost_task": bool(dc < 0 and dy < 0),
        }
    y_values = np.asarray(delta_y, dtype=float)
    c_values = np.asarray(delta_c, dtype=float)
    y_low, y_high = percentile95(y_values, indices)
    c_low, c_high = percentile95(c_values, indices)
    joint = {
        "both_better": sum(1 for s in DEV if per_scene[str(s)]["delta_y"] > 0
                           and per_scene[str(s)]["delta_c_bytes"] < 0),
        "task_better_cost_worse": sum(1 for s in DEV if per_scene[str(s)]["delta_y"] > 0
                                      and per_scene[str(s)]["delta_c_bytes"] > 0),
        "cheaper_but_lost_task": sum(1 for s in DEV
                                     if per_scene[str(s)]["cheaper_but_lost_task"]),
        "both_worse": sum(1 for s in DEV if per_scene[str(s)]["delta_y"] < 0
                          and per_scene[str(s)]["delta_c_bytes"] > 0),
        "other": None,
    }
    joint["other"] = 16 - joint["both_better"] - joint["task_better_cost_worse"] - joint[
        "cheaper_but_lost_task"] - joint["both_worse"]
    return {
        "training_seed": seed, "mode": mode, "scenes": 16,
        "delta_y": {"mean": round(float(y_values.mean()), 6), "min": float(y_values.min()),
                    "max": float(y_values.max()),
                    "counts": {"positive": int((y_values > 0).sum()),
                               "zero": int((y_values == 0).sum()),
                               "negative": int((y_values < 0).sum())},
                    "ci95": [round(y_low, 6), round(y_high, 6)]},
        "delta_c": {"mean": round(float(c_values.mean()), 3), "min": float(c_values.min()),
                    "max": float(c_values.max()),
                    "counts": {"cheaper": int((c_values < 0).sum()),
                               "equal": int((c_values == 0).sum()),
                               "costlier": int((c_values > 0).sum())},
                    "ci95": [round(c_low, 3), round(c_high, 3)]},
        "joint_counts": joint,
        "goal_set_swaps": {"gained": sorted(gained_total), "lost": sorted(lost_total),
                           "identity_swap_despite_net_tie": bool(
                               gained_total and lost_total)},
        "vs_immediate": {
            "new_mean_on_time": round(float(np.mean([new[s]["on_time"] for s in DEV])), 6),
            "old_mean_on_time": round(float(np.mean([old[s]["on_time"] for s in DEV])), 6),
            "reference_mean_on_time": round(float(np.mean([reference[s] for s in DEV])), 6),
            "new_mean_bytes": round(float(np.mean([new[s]["spent_bytes"] for s in DEV])), 3),
            "old_mean_bytes": round(float(np.mean([old[s]["spent_bytes"] for s in DEV])), 3),
            "reference_mean_bytes": round(float(np.mean(
                [int(reference_row(s)["spent_bytes"]) for s in DEV])), 3),
            "late_mean_on_time": round(float(np.mean([late[s] for s in DEV])), 6),
            "late_mean_bytes": round(float(np.mean([int(late_row(s)["spent_bytes"])
                                                    for s in DEV])), 3),
        },
        "per_scene": per_scene,
    }


def main() -> int:
    indices = index_matrix()
    payload: dict[str, Any] = {
        "round": "30",
        "section": "paired NEW - OLD comparison per training seed and mode (descriptive)",
        "statistics_plan": common.STATISTICS_PLAN,
        "hand_check": hand_check(indices),
        "comparisons": {},
        "warning": "the development block has been used many times; nothing here is a "
                   "confirmatory significance claim, a delta=0 guarantee, or an independent "
                   "generalisation claim; the three seeds are shown separately and never pooled; "
                   "a cheaper-but-lost-task scene is not an unconditional improvement",
    }
    for seed in common.TRAIN_SEEDS:
        for mode in ("greedy", "sampled"):
            payload["comparisons"][f"s{seed}_{mode}"] = compare(seed, mode, indices)
    payload["seed_consistency"] = {
        f"s{seed}": {
            mode: payload["comparisons"][f"s{seed}_{mode}"]["delta_y"]["counts"]
            for mode in ("greedy", "sampled")}
        for seed in common.TRAIN_SEEDS}
    out = common.OUTPUT_DIR / "PAIRED_COMPARISON.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    for key, value in payload["comparisons"].items():
        print(key, "dY mean", value["delta_y"]["mean"], "counts", value["delta_y"]["counts"],
              "| dC mean", value["delta_c"]["mean"], "counts", value["delta_c"]["counts"],
              "| y CI", value["delta_y"]["ci95"], "c CI", value["delta_c"]["ci95"])
    print("hand check passed:", payload["hand_check"]["passed"])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
