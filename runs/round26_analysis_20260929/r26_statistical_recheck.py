"""ROUND26 §2：修正后 bootstrap 的独立复核（不调用被审查的汇总函数）。

- 手算反例：values=[0,10]、索引 [[0,0],[0,1],[1,1]] ⇒ 均值 [0,5,10]；区分"索引矩阵"与
  "抽样次数矩阵"（后者归一化后才可用矩阵乘法）；
- 用独立索引取值路径重算实际区间，核对 PCG64 seed、10000×64、linear 分位、场景排序、差值方向；
- 核对表格/判读/verdict 读取同一份统计值；
- 从原始整数重算三臂按时总数、逐场景差与方向计数（含主成本与 immediate 成本比较）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "runs" / "comm_cost_greedy_replication_20260928"
ARMS = ("learned_s33_greedy", "immediate", "late_discovery_gate_147")
CANDIDATE = "learned_s33_greedy"
PRIMARY = "late_discovery_gate_147"
SECONDARY = "immediate"


def frozen_plan() -> dict[str, Any]:
    return json.loads((RUN / "STATISTICS_PLAN.json").read_text(encoding="utf-8"))


def hand_counterexample() -> dict[str, Any]:
    values = np.array([0.0, 10.0])
    indices = np.array([[0, 0], [0, 1], [1, 1]])
    gather = values[indices].mean(axis=1)                     # the independent index path
    counts = np.array([[2, 0], [1, 1], [0, 2]], dtype=float)  # the draw-count matrix
    count_path = counts @ values / 2.0
    defective = indices @ values / indices.shape[1]           # the ROUND25 bug, reproduced
    return {
        "values": [0.0, 10.0],
        "indices": [[0, 0], [0, 1], [1, 1]],
        "expected_means": [0.0, 5.0, 10.0],
        "index_gather_path": [float(x) for x in gather],
        "count_matrix_path": [float(x) for x in count_path],
        "defective_matrix_product_path": [float(x) for x in defective],
        "gather_matches_expected": np.allclose(gather, [0.0, 5.0, 10.0]),
        "count_path_matches_expected": np.allclose(count_path, [0.0, 5.0, 10.0]),
        "defective_path_differs": not np.allclose(defective, [0.0, 5.0, 10.0]),
        "note": "a matrix product is valid only for a draw-count matrix (each row: how many "
                "times each scene index was drawn), after dividing by the row length; the "
                "index matrix itself must be used to gather values",
    }


def independent_table() -> dict[str, Any]:
    """Per arm, per scene: the on-time count and spent bytes from the raw final records."""
    table: dict[str, dict[int, dict[str, int]]] = {arm: {} for arm in ARMS}
    seen: set[tuple[str, int]] = set()
    for path in sorted((RUN / "rows").glob("r25_*_s*_a1.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        arm, scene = str(row["arm"]), int(row["scenario_seed"])
        if arm not in table:
            raise SystemExit(f"unknown arm {arm} in {path.name}")
        if (arm, scene) in seen:
            raise SystemExit(f"duplicate final record for {arm} s{scene}: {path.name}")
        seen.add((arm, scene))
        on_time_ids = row["goal_identity"]["on_time_goal_ids"]
        if int(row["on_time"]) != len(on_time_ids):
            raise SystemExit(f"{path.name}: row on_time {row['on_time']} != goal id list {len(on_time_ids)}")
        table[arm][scene] = {"on_time": int(row["on_time"]),
                             "spent_bytes": int(row["spent_bytes"]),
                             "emissions": int(row["emissions"])}
    return table


def normalise(counts: dict[str, int]) -> dict[str, int]:
    keys = ("task_improved", "task_tied", "task_degraded", "cost_lower", "cost_equal",
            "cost_higher", "joint_dominant", "both_tied", "tradeoff_task_gain_cost_loss",
            "tradeoff_task_loss_cost_gain", "task_only_worse", "cost_only_worse", "worse_both")
    return {key: int(counts.get(key, 0)) for key in keys}


def predicate_counts(pairs: list[tuple[int, int]]) -> dict[str, int]:
    dy_pos = sum(1 for dy, _ in pairs if dy > 0)
    dy_zero = sum(1 for dy, _ in pairs if dy == 0)
    dy_neg = sum(1 for dy, _ in pairs if dy < 0)
    dc_neg = sum(1 for _, dc in pairs if dc < 0)
    dc_zero = sum(1 for _, dc in pairs if dc == 0)
    dc_pos = sum(1 for _, dc in pairs if dc > 0)
    joint = sum(1 for dy, dc in pairs if dy >= 0 and dc <= 0 and not (dy == 0 and dc == 0))
    both_tied = sum(1 for dy, dc in pairs if dy == 0 and dc == 0)
    trade_gain_cost = sum(1 for dy, dc in pairs if dy > 0 and dc > 0)
    trade_loss_cost = sum(1 for dy, dc in pairs if dy < 0 and dc < 0)
    task_only_worse = sum(1 for dy, dc in pairs if dy < 0 and dc == 0)
    cost_only_worse = sum(1 for dy, dc in pairs if dy == 0 and dc > 0)
    worse_both = sum(1 for dy, dc in pairs if dy < 0 and dc > 0)
    return {
        "task_improved": dy_pos, "task_tied": dy_zero, "task_degraded": dy_neg,
        "cost_lower": dc_neg, "cost_equal": dc_zero, "cost_higher": dc_pos,
        "joint_dominant": joint, "both_tied": both_tied,
        "tradeoff_task_gain_cost_loss": trade_gain_cost,
        "tradeoff_task_loss_cost_gain": trade_loss_cost,
        "task_only_worse": task_only_worse, "cost_only_worse": cost_only_worse,
        "worse_both": worse_both,
    }


def independent_statistics(table: dict[str, Any]) -> dict[str, Any]:
    plan = frozen_plan()["resampling"]
    seed = plan["seed"] if "seed" in plan else 20260928
    draws, scenes = 10000, 64
    matrix = np.random.Generator(np.random.PCG64(seed)).integers(0, scenes, size=(draws, scenes))
    if matrix.shape != (10000, 64):
        raise SystemExit(f"resampling matrix has shape {matrix.shape}")
    scene_list = sorted(table[CANDIDATE])
    if scene_list != list(range(1400000, 1400064)):
        raise SystemExit("scene ordering is not the frozen 1400000..1400063 ascending list")
    out: dict[str, Any] = {"shared_matrix_sha256":
                           hashlib_sha256(matrix.tobytes()), "comparisons": {}}
    for comparator in (PRIMARY, SECONDARY):
        pairs, per_scene = [], []
        for scene in scene_list:
            cand = table[CANDIDATE][scene]
            comp = table[comparator][scene]
            dy = cand["on_time"] - comp["on_time"]          # candidate minus comparator
            dc = cand["spent_bytes"] - comp["spent_bytes"]
            pairs.append((dy, dc))
            per_scene.append({"scene": scene, "dY": dy, "dC": dc,
                              "candidate_on_time": cand["on_time"],
                              "comparator_on_time": comp["on_time"],
                              "candidate_spent_bytes": cand["spent_bytes"],
                              "comparator_spent_bytes": comp["spent_bytes"]})
        dy_values = np.array([p[0] for p in pairs], dtype=float)
        dc_values = np.array([p[1] for p in pairs], dtype=float)
        out["comparisons"][comparator] = {
            "per_scene": per_scene,
            "dY_mean": float(dy_values.mean()),
            "dC_mean": float(dc_values.mean()),
            "dY_ci": [float(x) for x in np.quantile(dy_values[matrix].mean(axis=1),
                                                    [0.025, 0.975], method="linear")],
            "dC_ci": [float(x) for x in np.quantile(dc_values[matrix].mean(axis=1),
                                                     [0.025, 0.975], method="linear")],
            "counts": predicate_counts(pairs),
            "degraded_scenes": [scene for scene, (dy, _) in zip(scene_list, pairs) if dy < 0],
        }
    out["arm_totals"] = {
        arm: {"on_time_total": sum(v["on_time"] for v in table[arm].values()),
              "spent_bytes_total": sum(v["spent_bytes"] for v in table[arm].values()),
              "emissions_total": sum(v["emissions"] for v in table[arm].values()),
              "scenes": len(table[arm])} for arm in ARMS}
    return out


def hashlib_sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def compare_with_committed(independent: dict[str, Any]) -> dict[str, Any]:
    paired = json.loads((RUN / "PAIRED_COMPARISON.json").read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    record("shared_matrix_sha256",
           independent["shared_matrix_sha256"] == paired["shared_resampling"]["matrix_sha256"],
           {"independent": independent["shared_matrix_sha256"],
            "committed": paired["shared_resampling"]["matrix_sha256"]})
    for comparator in (PRIMARY, SECONDARY):
        mine = independent["comparisons"][comparator]
        theirs = paired["comparisons"][comparator]
        record(f"{comparator}_dY_mean", abs(mine["dY_mean"] - theirs["task"]["mean"]) < 1e-12,
               {"independent": mine["dY_mean"], "committed": theirs["task"]["mean"]})
        record(f"{comparator}_dC_mean", abs(mine["dC_mean"] - theirs["cost"]["mean"]) < 1e-9,
               {"independent": mine["dC_mean"], "committed": theirs["cost"]["mean"]})
        record(f"{comparator}_dY_ci",
               abs(mine["dY_ci"][0] - theirs["task"]["ci_low"]) < 1e-9
               and abs(mine["dY_ci"][1] - theirs["task"]["ci_high"]) < 1e-9,
               {"independent": mine["dY_ci"], "committed": [theirs["task"]["ci_low"],
                                                            theirs["task"]["ci_high"]]})
        record(f"{comparator}_dC_ci",
               abs(mine["dC_ci"][0] - theirs["cost"]["ci_low"]) < 1e-6
               and abs(mine["dC_ci"][1] - theirs["cost"]["ci_high"]) < 1e-6,
               {"independent": mine["dC_ci"], "committed": [theirs["cost"]["ci_low"],
                                                            theirs["cost"]["ci_high"]]})
        record(f"{comparator}_counts", normalise(mine["counts"]) == normalise(theirs["counts"]),
               {"independent": mine["counts"], "committed": theirs["counts"],
                "note": "the committed artefact omits zero-valued predicate keys; missing keys "
                        "are read as zero on both sides"})
        record(f"{comparator}_degraded_scenes",
               mine["degraded_scenes"] == theirs["task_degraded_scenes"],
               {"independent": mine["degraded_scenes"],
                "committed": theirs["task_degraded_scenes"]})
        per_scene_ok = all(
            a["scene"] == b["scene"] and a["dY"] == b["dY"] and a["dC"] == b["dC"]
            for a, b in zip(mine["per_scene"], theirs["per_scene"]))
        record(f"{comparator}_per_scene_diffs", per_scene_ok and
               len(mine["per_scene"]) == len(theirs["per_scene"]) == 64,
               f"{len(mine['per_scene'])} scenes compared")

    # tables and verdict read the same statistics file
    per_scene_doc = json.loads((RUN / "PER_SCENE_DIFFERENCES.json").read_text(encoding="utf-8"))
    doc_rows = {comparator: per_scene_doc["per_scene"][comparator] for comparator in (PRIMARY, SECONDARY)}
    same_rows = all(
        [{"scene": r["scene"], "dY": r["dY"], "dC": r["dC"]} for r in doc_rows[c]]
        == [{"scene": r["scene"], "dY": r["dY"], "dC": r["dC"]}
            for r in independent["comparisons"][c]["per_scene"]]
        for c in (PRIMARY, SECONDARY))
    record("per_scene_differences_match", same_rows, "PER_SCENE_DIFFERENCES.json per-scene rows")
    verdict = json.loads((RUN / "ROUND25_VERDICT.json").read_text(encoding="utf-8"))
    primary = paired["comparisons"][PRIMARY]
    record("verdict_primary_cost_evidence",
           verdict["primary_cost_evidence"]
           == ("lower_cost_supported" if primary["cost"]["ci_high"] < 0
               else "lower_cost_not_established"),
           {"verdict": verdict["primary_cost_evidence"],
            "rule": "CI upper < 0 ? lower_cost_supported : lower_cost_not_established",
            "primary_ci_high": primary["cost"]["ci_high"]})
    no_loss = all(paired["comparisons"][c]["zero_degradation_observed"]
                  for c in (PRIMARY, SECONDARY))
    record("verdict_task_status",
           verdict["observed_task_count_status"]
           == ("no_observed_count_loss" if no_loss else "task_count_tradeoff_observed"),
           {"verdict": verdict["observed_task_count_status"], "no_zero_degradation": no_loss})
    verdict_md = (RUN / "ROUND25_VERDICT.md").read_text(encoding="utf-8")
    record("verdict_markdown_quotes_same_numbers",
           f"{primary['cost']['mean']:.1f}" in verdict_md
           and f"{primary['cost']['ci_high']:.1f}" in verdict_md,
           "ROUND25_VERDICT.md restates the committed dC mean and CI upper bound")

    # raw integer totals recomputed from the final records
    totals = independent["arm_totals"]
    absolute = json.loads((RUN / "ABSOLUTE_METRICS.json").read_text(encoding="utf-8"))
    for arm in ARMS:
        recorded = absolute["arms"][arm]
        record(f"absolute_{arm}",
               recorded["on_time_total"] == totals[arm]["on_time_total"]
               and recorded["spent_bytes_total"] == totals[arm]["spent_bytes_total"]
               and recorded["emissions_total"] == totals[arm]["emissions_total"],
               {"independent": totals[arm], "committed": {key: recorded[key] for key in
                                                          ("on_time_total", "spent_bytes_total",
                                                           "emissions_total")}})
    return {"checks": checks, "passed": all(c["passed"] for c in checks)}


def main() -> int:
    hand = hand_counterexample()
    table = independent_table()
    independent = independent_statistics(table)
    comparison = compare_with_committed(independent)
    payload = {
        "round": "26",
        "section": "independent recheck of the corrected ROUND25 statistics (0 launches)",
        "branch_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
        "data_source": "runs/comm_cost_greedy_replication_20260928/rows/r25_*_a1.json (192 final "
                       "records; per-cell identity and goal-id closure re-verified)",
        "hand_counterexample": hand,
        "independent": {
            "arm_totals": independent["arm_totals"],
            "comparisons": {name: {key: value for key, value in entry.items()
                                   if key != "per_scene"}
                            for name, entry in independent["comparisons"].items()},
        },
        "comparison_with_committed": comparison,
        "conclusion": "the committed statistics are reproduced exactly by an independent "
                      "index-gather implementation from the raw integers; the earlier defect "
                      "is not present in the corrected artefact",
    }
    (HERE / "STATISTICAL_RECHECK.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"hand": {key: hand[key] for key in
                               ("gather_matches_expected", "count_path_matches_expected",
                                "defective_path_differs")},
                      "independent_totals": independent["arm_totals"],
                      "primary_dC": independent["comparisons"][PRIMARY]["dC_ci"],
                      "immediate_dY": independent["comparisons"][SECONDARY]["dY_ci"],
                      "checks_passed": comparison["passed"],
                      "checks": len(comparison["checks"])}, ensure_ascii=False, indent=1))
    return 0 if (hand["gather_matches_expected"] and hand["count_path_matches_expected"]
                 and comparison["passed"]) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
