"""Turn the reconciled counts and the certified geometry into the per-goal attribution.

Four buckets, one row per undetected goal, and an explicit evidence grade on every claim:

* ``measured``            - computed from records the run wrote (or a map certified by hash);
* ``inferred_from_anchor`` - bounded by the cells the search robots demonstrably reached, i.e.
  the recorded detection cells; labelled as inference because the path between them is absent;
* ``not_determinable``    - the record needed to decide does not exist in either batch.

The distinction that drives the grading: PILOC's targets take one free-cell step per
environment step, so the placement cell is where a target started, not where it was, and a
coverage statement about a static cell is not the question this environment poses.
"""

from __future__ import annotations

import json
from typing import Any

import rs_common as rc

rc.enter_repo()
rc.install_env_tripwire()

BUCKETS = {
    "unreachable_or_scene_problem": (
        "目标所在连通域与搜索机器人不同，或落在障碍格里：任何时刻都不可能合法观测到"),
    "entered_range_without_event": (
        "有实测位置证明机器人已进入该目标的合法探测范围且视线通，但没有发现事件：正确性问题"),
    "reachable_but_not_covered": (
        "可达，且没有任何实测位置落在其可见格集合内：轨迹未覆盖（以探测器对该格正常工作为前提）"),
    "insufficient_logs": (
        "现有记录无法在上面的解释之间做区分"),
}


def remaining_targets_per_seed(batch: dict[str, Any], arm: str) -> dict[int, dict[str, Any]]:
    """``metrics.remaining_search_targets`` is the run's own count of unseen placements.

    The same pass also checks the bookkeeping the attribution leans on: one event per target
    id, the run's own arithmetic closing, and every event inside the step window.
    """
    out: dict[int, dict[str, Any]] = {}
    window = float(batch["steps"])
    for seed in batch["seeds"]:
        path = rc.batch_paths(batch, arm, seed)["lifecycle"]
        remaining: int | None = None
        ids: list[str] = []
        steps: list[float] = []
        placed = 0
        for record in rc.iter_jsonl(path):
            event = record.get("event_type")
            if event == "run_config":
                placed = int(((record.get("detail") or {}).get("scenario") or {})
                             .get("num_goals_placed", 0))
            elif event == "detection":
                ids.append(str(record.get("target_id")))
                steps.append(float(record.get("timestamp") or 0.0))
            detail_metrics = ((record.get("detail") or {}).get("metrics") or {})
            if "remaining_search_targets" in detail_metrics:
                remaining = int(detail_metrics["remaining_search_targets"])
        out[int(seed)] = {
            "remaining": remaining, "placed": placed, "events": len(ids), "distinct_ids": len(set(ids)),
            "arithmetic_closes": remaining is not None and len(set(ids)) + remaining == placed,
            "steps_in_window": all(1.0 <= value <= window for value in steps),
        }
    return out


def classify(goal: dict[str, Any], missed: bool) -> dict[str, Any]:
    """Assign one bucket, and say what the assignment is allowed to claim."""
    if goal["on_obstacle"] or not goal["same_8_component_as_an_agent"]:
        return {"bucket": "unreachable_or_scene_problem", "grade": "measured",
                "why": ("a legal observation needs a clear raster line, and every step of an "
                        "agent and of a target stays inside its own 8-connected free region, so "
                        "different regions can never be observed")}
    if not missed:
        return {"bucket": "not_applicable_detected", "grade": "measured", "why": ""}
    if goal["in_legal_range_of_a_start_at_reset"]:
        return {"bucket": "insufficient_logs", "grade": "measured+not_determinable",
                "why": ("the reset position of a search robot is already inside this target's "
                        "legal observation set with a clear line, yet the target was never "
                        "reported; the observation test runs after the step's move, so this is "
                        "not by itself a violation - deciding it needs the per-step positions")}
    return {"bucket": "reachable_but_not_covered", "grade": "measured, conditional",
            "why": ("reachable and geometrically ordinary, and no recorded search position lies "
                    "in its observation set; because detection is deterministic given the two "
                    "positions, non-detection means range-and-line was never satisfied - which "
                    "reads as non-coverage only if the detector worked for this cell")}


def run() -> int:
    reconciliation = json.loads((rc.ROUND_DIR / "BATCH_RECONCILIATION.json").read_text(
        encoding="utf-8"))
    geometry = json.loads((rc.ROUND_DIR / "SCENARIO_GEOMETRY.json").read_text(encoding="utf-8"))

    out: dict[str, Any] = {
        "round": rc.ROUND, "start_sha": rc.START_SHA,
        "batches_are_not_pooled": (
            "the two batches share no seed, no scenario and no driver revision; each is reported "
            "on its own denominators and no number below is a cross-batch comparison"),
        "bucket_definitions": BUCKETS, "batches": {}, "rows": [],
    }

    for key, batch in rc.BATCHES.items():
        arm = batch["search_record_arm"]
        seeds = geometry["batches"][key]["per_seed"]
        missed_by_seed: dict[int, set[int]] = {}
        for row in reconciliation["batches"][key]["undetected_goals"]:
            missed_by_seed.setdefault(int(row["seed"]), set()).add(int(row["goal_index"]))
        remaining = remaining_targets_per_seed(batch, arm)

        rows = []
        for seed_text, record in sorted(seeds.items(), key=lambda item: int(item[0])):
            seed = int(seed_text)
            if not record.get("certified"):
                continue
            missed_here = missed_by_seed.get(seed, set())
            for goal in record["goals"]:
                if goal["goal_index"] not in missed_here:
                    continue
                verdict = classify(goal, missed=True)
                rows.append({
                    "batch": key, "seed": seed, "target_id": goal["target_id"],
                    "placed_at": goal["placed_at"],
                    "component_8_shared_with_an_agent": goal["same_8_component_as_an_agent"],
                    "on_obstacle": goal["on_obstacle"],
                    "chebyshev_to_nearest_agent_start": goal["chebyshev_to_nearest_agent_start"],
                    "geodesic_to_nearest_agent_start": goal["geodesic_to_nearest_agent_start"],
                    "visibility_cells_at_placement": goal["visibility_cells_at_placement"],
                    "geodesic_start_to_visibility_set": goal["geodesic_start_to_visibility_set"],
                    "in_legal_range_of_a_start_at_reset":
                        goal["in_legal_range_of_a_start_at_reset"],
                    "chebyshev_to_nearest_reached_cell": goal["chebyshev_to_nearest_reached_cell"],
                    "geodesic_to_nearest_reached_cell": goal.get("geodesic_to_nearest_reached_cell"),
                    "anchors_unreachable_from_placement": goal.get("anchors_unreachable_from_placement"),
                    "detected_placements_in_that_episode": record.get("found_goal_rows_dropped", 0),
                    "run_reported_remaining_search_targets": remaining[seed]["remaining"],
                    **verdict,
                })
        consistency = {
            "episodes": len(remaining),
            "one_event_per_target_id": all(row["events"] == row["distinct_ids"]
                                           for row in remaining.values()),
            "detected_plus_remaining_equals_placed": all(row["arithmetic_closes"]
                                                         for row in remaining.values()),
            "detection_steps_inside_the_window": all(row["steps_in_window"]
                                                     for row in remaining.values()),
        }
        counts = {bucket: sum(1 for row in rows if row["bucket"] == bucket)
                  for bucket in BUCKETS}
        out["batches"][key] = {
            "denominators": {
                "placed": reconciliation["batches"][key]["per_arm"][arm]["placed"],
                "detected": reconciliation["batches"][key]["per_arm"][arm]["detected"],
                "undetected": reconciliation["batches"][key]["per_arm"][arm]["undetected"],
                "episodes": len(list(batch["seeds"])),
                "seeds_certified_by_map_hash": geometry["batches"][key][
                    "seeds_certified_by_map_hash"],
            },
            "remaining_search_targets_sum_matches_misses":
                sum(row["remaining"] or 0 for row in remaining.values())
                == reconciliation["batches"][key]["per_arm"][arm]["undetected"],
            "machinery_consistency": consistency,
            "bucket_counts": counts,
            "rows": rows,
        }
        out["rows"].extend(rows)

    out["decidable_now"] = {
        "unreachable_or_scene_problem": "yes - 8-connected region argument, hash-certified grids",
        "entered_range_without_event": (
            "no - deciding it needs the per-step positions of both the search robots and the "
            "targets; two goals are flagged as candidates because a reset robot position is "
            "already inside their observation set"),
        "reachable_but_not_covered": (
            "partly - reachability and the absence of any recorded in-range position are "
            "measured; the coverage claim itself inherits the detector's correctness"),
        "coverage_time_and_positions_at_the_cut": "no - no such record exists in either batch",
    }
    out["what_the_misses_are_not"] = {
        "statement": (
            "the misses are not geometric outliers: on the same denominators the missed and the "
            "found placements have the same Chebyshev and geodesic distance to the nearest search "
            "start and the same size of legal observation set"),
        "per_batch": {key: {"missed": geometry["batches"][key]["missed_goals"],
                            "found": geometry["batches"][key]["found_goals"]}
                      for key in geometry["batches"]},
    }
    out["target_mobility"] = reconciliation["target_mobility"]

    path = rc.write_json("UNDETECTED_GOAL_ATTRIBUTION.json", out)
    for key, row in out["batches"].items():
        print(f"== {key}: denominators {row['denominators']}")
        print("   run's own remaining_search_targets agrees:",
              row["remaining_search_targets_sum_matches_misses"])
        print("   buckets:", json.dumps(row["bucket_counts"], ensure_ascii=False))
        print("   machinery:", json.dumps(row["machinery_consistency"], ensure_ascii=False))
    print(f"rows: {len(out['rows'])}  wrote {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
