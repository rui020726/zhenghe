"""ROUND25 预登记统计：固定候选的配对 bootstrap；表格、判读、verdict 共用同一机读结果。

执行依据 §六（冻结项）：
- 每场景 dY = 候选按时任务数 - 参照按时任务数（正更好）；dC = 候选成本 - 参照成本（负更省）；
- 唯一主成本比较：seed33 greedy 对 late_discovery_gate(147)；任务比较同时对 immediate 和 late；
- 以 64 场景为配对抽样单位；NumPy PCG64 seed=20260928，一次生成 10000×64 重采样索引，
  所有任务/成本比较共用该索引；95% percentile bootstrap，分位计算用 linear；
- 各计数的相等/严格不等谓词在冻结时逐项写明（见 PREDICATES），不混用。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import r25_common as common

BOOTSTRAP_SEED = 20260928
BOOTSTRAP_DRAWS = 10000
SCENE_COUNT = 64
CONFIDENCE = 0.95
QUANTILE_METHOD = "linear"

#: Frozen predicates for the joint partition (complete partition of the (dY, dC) plane).
PREDICATES: dict[str, str] = {
    "task_improved": "dY > 0",
    "task_tied": "dY == 0",
    "task_degraded": "dY < 0",
    "cost_lower": "dC < 0",
    "cost_equal": "dC == 0",
    "cost_higher": "dC > 0",
    "joint_dominant": "dY >= 0 and dC <= 0 and not (dY == 0 and dC == 0)",
    "both_tied": "dY == 0 and dC == 0",
    "tradeoff_task_gain_cost_loss": "dY > 0 and dC > 0",
    "tradeoff_task_loss_cost_gain": "dY < 0 and dC < 0",
    "task_only_worse": "dY < 0 and dC == 0",
    "cost_only_worse": "dY == 0 and dC > 0",
    "worse_both": "dY < 0 and dC > 0",
}


def resample_matrix(n_scenes: int = SCENE_COUNT, n_draws: int = BOOTSTRAP_DRAWS,
                    seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """The one shared resampling index matrix: PCG64(seed), rows are draws, columns scenes."""
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, n_scenes, size=(n_draws, n_scenes))


def matrix_digest(matrix: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(matrix, dtype=np.int64).tobytes()).hexdigest()


def bootstrap_ci(deltas: list[float], matrix: np.ndarray) -> dict[str, float]:
    """Percentile bootstrap over scenes with the shared resampling matrix (linear quantiles).

    Each resampling row holds scene indices; the statistic is the mean of the deltas at those
    indices (``values[matrix]``). The mean of bounded values stays inside their range, which is
    asserted below: a matrix product or any other index-arithmetic slip cannot pass silently.
    """
    values = np.asarray(deltas, dtype=np.float64)
    if values.shape[0] != matrix.shape[1]:
        raise RuntimeError(f"{values.shape[0]} scenes vs resampling matrix with {matrix.shape[1]}")
    means = values[matrix].mean(axis=1)
    if means.shape != (matrix.shape[0],):
        raise RuntimeError(f"resampled means have shape {means.shape}, expected "
                           f"({matrix.shape[0]},)")
    low, high = np.quantile(means, [0.5 - CONFIDENCE / 2, 0.5 + CONFIDENCE / 2],
                            method=QUANTILE_METHOD)
    if not (float(values.min()) <= float(low) <= float(high) <= float(values.max())):
        raise RuntimeError(
            f"bootstrap interval [{low}, {high}] leaves the range of the deltas "
            f"[{values.min()}, {values.max()}]; the resampling arithmetic is wrong")
    return {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high),
            "draws": int(matrix.shape[0]), "scenes": int(values.shape[0]),
            "quantile_method": QUANTILE_METHOD, "confidence": CONFIDENCE}


def classify(dy: int, dc: int) -> dict[str, bool]:
    return {
        "task_improved": dy > 0, "task_tied": dy == 0, "task_degraded": dy < 0,
        "cost_lower": dc < 0, "cost_equal": dc == 0, "cost_higher": dc > 0,
        "joint_dominant": dy >= 0 and dc <= 0 and not (dy == 0 and dc == 0),
        "both_tied": dy == 0 and dc == 0,
        "tradeoff_task_gain_cost_loss": dy > 0 and dc > 0,
        "tradeoff_task_loss_cost_gain": dy < 0 and dc < 0,
        "task_only_worse": dy < 0 and dc == 0,
        "cost_only_worse": dy == 0 and dc > 0,
        "worse_both": dy < 0 and dc > 0,
    }


def zero_event_upper_bound(n_scenes: int = SCENE_COUNT, alpha: float = 0.05) -> float:
    """One-sided 95% upper bound on the per-scenario probability of a net degradation event
    when zero degradations are observed under an i.i.d. scene assumption (1 - alpha^(1/n))."""
    return float(1.0 - alpha ** (1.0 / n_scenes))


def load_rows() -> dict[str, dict[int, dict[str, Any]]]:
    """Per arm, per scene, the final record. Missing cells are an error, never silently dropped."""
    table: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in common.ARM_ORDER}
    for path in sorted(common.ROWS.glob("r25_*_s*_a*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        arm = str(row["arm"])
        scene = int(row["scenario_seed"])
        if arm not in table:
            raise RuntimeError(f"unknown arm {arm!r} in {path.name}")
        if scene in table[arm]:
            raise RuntimeError(f"two final records for {arm} s{scene}: {path.name}")
        table[arm][scene] = row
    return table


def require_complete(table: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """Hard-fail aggregation when any cell is missing; the verdict then reports incomplete."""
    missing: list[dict[str, Any]] = []
    for arm in common.ARM_ORDER:
        for scene in common.NEW_SCENARIOS:
            if int(scene) not in table.get(arm, {}):
                missing.append({"arm": arm, "scenario": int(scene)})
    if missing:
        raise SystemExit(f"aggregation refuses to run: {len(missing)} missing cells, first="
                         f"{missing[:6]}")
    return {"cells": len(common.ARM_ORDER) * len(common.NEW_SCENARIOS),
            "scenes": len(common.NEW_SCENARIOS), "arms": list(common.ARM_ORDER),
            "missing": []}


def paired_table(table: dict[str, dict[int, dict[str, Any]]],
                 matrix: np.ndarray | None = None) -> dict[str, Any]:
    """The single machine-readable statistics artefact every table and the verdict read."""
    matrix = resample_matrix() if matrix is None else matrix
    completeness = require_complete(table)
    scenes = [int(scene) for scene in common.NEW_SCENARIOS]
    candidate = common.CANDIDATE_ARM
    comparisons: dict[str, Any] = {}
    for comparator in common.TASK_COMPARATORS:
        per_scene = []
        d_y, d_c = [], []
        for scene in scenes:
            cand, comp = table[candidate][scene], table[comparator][scene]
            dy = int(cand["on_time"]) - int(comp["on_time"])
            dc = int(cand["spent_bytes"]) - int(comp["spent_bytes"])
            d_y.append(dy)
            d_c.append(dc)
            per_scene.append({"scene": scene, "candidate_on_time": int(cand["on_time"]),
                              "comparator_on_time": int(comp["on_time"]), "dY": dy,
                              "candidate_spent_bytes": int(cand["spent_bytes"]),
                              "comparator_spent_bytes": int(comp["spent_bytes"]), "dC": dc,
                              "counts": {key: value for key, value in classify(dy, dc).items()}})
        counts: dict[str, int] = {}
        for item in per_scene:
            for key, value in item["counts"].items():
                if value:
                    counts[key] = counts.get(key, 0) + 1
        degraded_scenes = [item["scene"] for item in per_scene if item["dY"] < 0]
        entry = {
            "comparator": comparator,
            "per_scene": per_scene,
            "task": bootstrap_ci(d_y, matrix),
            "cost": bootstrap_ci(d_c, matrix),
            "counts": counts,
            "task_degraded_scenes": degraded_scenes,
            "zero_degradation_observed": not degraded_scenes,
        }
        if not degraded_scenes:
            entry["zero_event_upper_bound_95"] = zero_event_upper_bound(len(scenes))
        comparisons[comparator] = entry
    return {
        "round": "25",
        "section": "pre-registered statistics (execution basis §6)",
        "completeness": completeness,
        "shared_resampling": {
            "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "scenes": SCENE_COUNT,
            "generator": "numpy.random.Generator(numpy.random.PCG64(seed)).integers(0,64,"
                         "size=(10000,64))",
            "matrix_sha256": matrix_digest(matrix),
            "confidence": CONFIDENCE, "quantile_method": QUANTILE_METHOD,
            "shared_by": "every task/cost comparison of this round",
        },
        "predicates": PREDICATES,
        "primary_cost_comparison": common.PRIMARY_COST_COMPARATOR,
        "candidate_arm": candidate,
        "comparisons": comparisons,
        "notes": [
            "the 64 scenes are the sampling unit; the 192 launches are not 192 independent scenes",
            "the bootstrap describes scenario uncertainty of this fixed candidate under the "
            "generation distribution; it carries no training-initialisation uncertainty",
            "a CI containing 0 is not evidence of no effect, and a task tie is not proof of "
            "population non-inferiority",
            "no old development scenes are pooled into these statistics",
        ],
    }


def write_statistics(table: dict[str, dict[int, dict[str, Any]]] | None = None) -> Path:
    table = load_rows() if table is None else table
    payload = paired_table(table)
    path = common.OUTPUT_DIR / "PAIRED_COMPARISON.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def read_statistics() -> dict[str, Any]:
    """The single statistics artefact; the verdict and every table read this and only this."""
    path = common.OUTPUT_DIR / "PAIRED_COMPARISON.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = matrix_digest(resample_matrix())
    if payload["shared_resampling"]["matrix_sha256"] != expected:
        raise RuntimeError("the statistics artefact was not produced with the frozen resampling")
    return payload
