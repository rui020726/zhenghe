"""ROUND32 任务 F：配对比较与绝对表现（只读落盘件，0 环境启动）。

- 每个训练 seed、每种模式单独比较：`ΔY = ACTIVE_ROWS − ALL_ROWS`（按时任务数/episode）、
  `ΔC = ACTIVE_ROWS − ALL_ROWS`（已计量载荷尝试字节/episode）；
- 同时列出每臂的**绝对**任务数与字节、相对 immediate 的配对任务损失、相对 late 的任务差与
  成本差、逐场景改善/持平/退化，以及按时目标集合是否交换。执行依据 §七 明确要求不得只报
  "两臂相等"而隐藏双方都比 immediate 差，因此这些绝对量与 Δ 同等地位写入同一份统计值；
- 描述性区间**沿用** ROUND30 的冻结方案：场景升序、16 场景为配对抽样单位、numpy PCG64
  seed=20260930、10000 次、linear 分位、所有比较复用同一份 10000×16 索引、估计量
  `values[indices].mean(axis=1)`，并保留独立手算反例。方案不换、不为其重新调参。
- 区间只覆盖固定权重下的场景不确定性，不覆盖训练 seed 的总体不确定性；sampled 每场景仅一次。
  三个 seed 分别展示，不池化、不挑最佳模式。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import r32_common as common

DEV = tuple(int(s) for s in common.DEVELOPMENT_SCENARIOS)
BASE, TEST = common.ARM_ORDER      # Δ = ACTIVE_ROWS − ALL_ROWS


def eval_row(reduction: str, seed: int, mode: str, scenario: int) -> dict[str, Any]:
    path = common.cell_paths("eval", scenario, 0,
                             tag=f"eval_{reduction}_s{seed}_final_{mode}")["row"]
    return common.read_json(path)


def reference_row(scenario: int) -> dict[str, Any]:
    return common.read_json(common.cell_paths("reference", scenario, 1)["row"])


def late_row(scenario: int) -> dict[str, Any]:
    return common.read_json(common.cell_paths("rule", scenario, 1,
                                              tag=common.LATE_RULE_ARM)["row"])


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


def paired_cells(reduction: str, seed: int, mode: str) -> dict[int, dict[str, Any]]:
    cells: dict[int, dict[str, Any]] = {}
    for scenario in DEV:
        row = eval_row(reduction, seed, mode, scenario)
        if row.get("reduction") != reduction or row.get("arm") != reduction:
            raise RuntimeError(f"{reduction} s{seed} {mode} s{scenario}: row identity mismatch")
        if int(row.get("optimizer_steps", -1)) != 0:
            raise RuntimeError(f"{reduction} s{seed} {mode} s{scenario}: an evaluation rollout "
                               "must not report optimizer steps")
        cells[scenario] = {
            "on_time": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]),
            "expired": int(row["expired"]), "emissions": int(row["emissions"]),
            "goal_ids": on_time_goal_ids(Path(row["artefacts"]["lifecycle"])),
            "behaviour_sha256": row["behaviour_sha256"],
            "parameter_sha256": (row.get("controller_parameters_after") or {})
            .get("parameter_sha256"),
            "row": Path(row["artefacts"]["row"]).name,
        }
    return cells


def index_matrix() -> np.ndarray:
    """The one shared 10000 x 16 resampling index (frozen scheme inherited from ROUND30)."""
    return np.random.Generator(np.random.PCG64(int(common.STATISTICS_PLAN["resample_seed"]))
                               ).integers(0, 16, size=(int(common.STATISTICS_PLAN["resamples"]),
                                                       16))


def percentile95(values: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def hand_check(indices: np.ndarray) -> dict[str, Any]:
    """An independent hand-computed counterexample for the resampling path."""
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
        "note": "the check uses the round's real index matrix; the same index matrix object is "
                "passed to every comparison in this script",
    }


def absolute_block(cells: dict[int, dict[str, Any]], reference: dict[int, int],
                   late: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """One arm's absolute position: task count, metered bytes, paired loss vs immediate, vs late."""
    on_time = [float(cells[s]["on_time"]) for s in DEV]
    bytes_ = [float(cells[s]["spent_bytes"]) for s in DEV]
    losses = [float(reference[s]) - float(cells[s]["on_time"]) for s in DEV]
    late_y = [float(cells[s]["on_time"]) - float(late[s]["on_time"]) for s in DEV]
    late_c = [float(cells[s]["spent_bytes"]) - float(late[s]["spent_bytes"]) for s in DEV]
    return {
        "mean_on_time": round(float(np.mean(on_time)), 6),
        "mean_spent_bytes": round(float(np.mean(bytes_)), 3),
        "mean_normalised_cost_c": round(float(np.mean(bytes_)) / common.B0_BYTES, 6),
        "mean_paired_task_loss_vs_immediate": round(float(np.mean(losses)), 6),
        "counts_vs_immediate_task": {
            "below_reference": int(sum(1 for v in losses if v > 0)),
            "equal_reference": int(sum(1 for v in losses if v == 0)),
            "above_reference": int(sum(1 for v in losses if v < 0))},
        "worst_scene_task_loss": float(max(losses)),
        "scenes_with_task_loss": [str(s) for s, v in zip(DEV, losses) if v > 0],
        "vs_late": {"mean_delta_y": round(float(np.mean(late_y)), 6),
                    "mean_delta_c_bytes": round(float(np.mean(late_c)), 3),
                    "counts": {"task_better": int(sum(1 for v in late_y if v > 0)),
                               "task_equal": int(sum(1 for v in late_y if v == 0)),
                               "task_worse": int(sum(1 for v in late_y if v < 0)),
                               "cheaper": int(sum(1 for v in late_c if v < 0)),
                               "cost_equal": int(sum(1 for v in late_c if v == 0)),
                               "costlier": int(sum(1 for v in late_c if v > 0))},
                    "cheaper_but_lost_task_vs_late": [
                        str(s) for s, dy, dc in zip(DEV, late_y, late_c) if dy < 0 and dc < 0]},
    }


def compare(seed: int, mode: str, indices: np.ndarray) -> dict[str, Any]:
    test = paired_cells(TEST, seed, mode)
    base = paired_cells(BASE, seed, mode)
    reference = {scenario: int(reference_row(scenario)["on_time"]) for scenario in DEV}
    late = {scenario: late_row(scenario) for scenario in DEV}
    delta_y, delta_c = [], []
    per_scene: dict[str, Any] = {}
    gained_total: list[str] = []
    lost_total: list[str] = []
    for scenario in DEV:
        dy = test[scenario]["on_time"] - base[scenario]["on_time"]
        dc = test[scenario]["spent_bytes"] - base[scenario]["spent_bytes"]
        delta_y.append(float(dy))
        delta_c.append(float(dc))
        gained = sorted(set(test[scenario]["goal_ids"]) - set(base[scenario]["goal_ids"]))
        lost = sorted(set(base[scenario]["goal_ids"]) - set(test[scenario]["goal_ids"]))
        gained_total.extend(gained)
        lost_total.extend(lost)
        per_scene[str(scenario)] = {
            "delta_y": int(dy), "delta_c_bytes": int(dc),
            "test_on_time": test[scenario]["on_time"], "base_on_time": base[scenario]["on_time"],
            "test_bytes": test[scenario]["spent_bytes"], "base_bytes": base[scenario]["spent_bytes"],
            "reference_on_time": reference[scenario], "late_on_time": int(late[scenario]["on_time"]),
            "gained_goal_ids": gained, "lost_goal_ids": lost,
            "task_direction": ("better" if dy > 0 else ("worse" if dy < 0 else "tied")),
            "cost_direction": ("cheaper" if dc < 0 else ("costlier" if dc > 0 else "tied")),
            "cheaper_but_lost_task_vs_base": bool(dc < 0 and dy < 0),
            "both_arms_below_reference": bool(reference[scenario] > test[scenario]["on_time"]
                                              and reference[scenario] > base[scenario]["on_time"]),
            "test_behaviour_sha256": test[scenario]["behaviour_sha256"],
            "base_behaviour_sha256": base[scenario]["behaviour_sha256"],
            "identical_behaviour": (test[scenario]["behaviour_sha256"]
                                    == base[scenario]["behaviour_sha256"]),
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
        "cheaper_but_lost_task_vs_base": sum(
            1 for s in DEV if per_scene[str(s)]["cheaper_but_lost_task_vs_base"]),
        "both_worse": sum(1 for s in DEV if per_scene[str(s)]["delta_y"] < 0
                          and per_scene[str(s)]["delta_c_bytes"] > 0),
    }
    joint["other"] = 16 - sum(joint[key] for key in list(joint))
    return {
        "training_seed": seed, "mode": mode, "scenes": 16, "delta_definition": "TEST - BASE",
        "arms": {"base": BASE, "test": TEST},
        "identical_behaviour_scenes": sum(1 for s in DEV
                                          if per_scene[str(s)]["identical_behaviour"]),
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
                           "identity_swap_despite_net_tie": bool(gained_total and lost_total)},
        "absolute": {TEST: absolute_block(test, reference, late),
                     BASE: absolute_block(base, reference, late)},
        "reference_immediate": {
            "mean_on_time": round(float(np.mean([reference[s] for s in DEV])), 6),
            "mean_spent_bytes": round(float(np.mean([int(reference_row(s)["spent_bytes"])
                                                     for s in DEV])), 3)},
        "late_rule": {"mean_on_time": round(float(np.mean([int(late[s]["on_time"])
                                                            for s in DEV])), 6),
                      "mean_spent_bytes": round(float(np.mean([int(late[s]["spent_bytes"])
                                                                for s in DEV])), 3)},
        "per_scene": per_scene,
    }


def main() -> int:
    indices = index_matrix()
    payload: dict[str, Any] = {
        "round": "32",
        "section": f"paired {TEST} - {BASE} comparison per training seed and mode (descriptive)",
        "statistics_plan": common.STATISTICS_PLAN,
        "hand_check": hand_check(indices),
        "comparisons": {},
        "warning": "the development block has been used many times; nothing here is a "
                   "confirmatory significance claim, a delta=0 guarantee, or an independent "
                   "generalisation claim; the three seeds are shown separately and never pooled; "
                   "greedy and sampled are reported separately and the best mode is not picked; "
                   "a cheaper-but-lost-task scene is not an unconditional improvement",
    }
    for seed in common.TRAIN_SEEDS:
        for mode in common.MODES:
            payload["comparisons"][f"s{seed}_{mode}"] = compare(int(seed), mode, indices)
    payload["seed_consistency"] = {
        f"s{seed}": {mode: {"delta_y_counts": payload["comparisons"][f"s{seed}_{mode}"]
                            ["delta_y"]["counts"],
                            "delta_c_mean": payload["comparisons"][f"s{seed}_{mode}"]
                            ["delta_c"]["mean"],
                            "absolute_task_loss": {
                                arm: payload["comparisons"][f"s{seed}_{mode}"]["absolute"][arm]
                                ["mean_paired_task_loss_vs_immediate"]
                                for arm in (BASE, TEST)}}
                        for mode in common.MODES}
        for seed in common.TRAIN_SEEDS}
    out = common.OUTPUT_DIR / "PAIRED_COMPARISON.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    for key, value in payload["comparisons"].items():
        print(key, "dY", value["delta_y"]["mean"], value["delta_y"]["counts"], "CI",
              value["delta_y"]["ci95"], "| dC", value["delta_c"]["mean"],
              value["delta_c"]["counts"], "CI", value["delta_c"]["ci95"],
              "| abs loss", {arm: value["absolute"][arm]["mean_paired_task_loss_vs_immediate"]
                             for arm in (BASE, TEST)})
    print("hand check passed:", payload["hand_check"]["passed"])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
