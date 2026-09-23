"""Reconcile the two live-PILOC batches, separately, and list every undetected goal.

Read-only. Nothing here decides a bucket; it fixes the denominators, checks that the
placement record and the detection record describe the same goals, and hands the
position list to ``rs_geometry``.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import rs_common as rc

rc.enter_repo()
rc.install_env_tripwire()

from integrated_sar.lifecycle_metrics import window_metrics  # noqa: E402


def read_cell(path: Path, probe_path: Path | None, window: float) -> dict[str, Any]:
    records = list(rc.iter_jsonl(path))
    configuration: dict[str, Any] = {}
    scenario: dict[str, Any] = {}
    detections: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("event_type") == "run_config":
            configuration = dict((record.get("detail") or {}).get("configuration") or {})
            scenario = dict((record.get("detail") or {}).get("scenario") or {})
        elif record.get("event_type") == "detection":
            detail = record.get("detail") or {}
            target = str(record.get("target_id"))
            row = {
                "step": float(record.get("timestamp") or 0.0),
                "grid_position": [int(v) for v in (detail.get("grid_position") or [])],
                "discovered_by": record.get("discovered_by"),
            }
            previous = detections.get(target)
            if previous is None or row["step"] < previous["step"]:
                detections[target] = row
    metrics = window_metrics(records, window)
    probe_placed: list[dict[str, Any]] = []
    probe_summary: dict[str, Any] = {}
    if probe_path is not None and probe_path.exists():
        for row in rc.iter_jsonl(probe_path):
            if row.get("kind") == "placed":
                probe_placed.append({"target_id": str(row["target_id"]),
                                     "grid_position": [int(v) for v in row["grid_position"]]})
            elif row.get("kind") == "probe_summary":
                probe_summary = dict(row)
    return {
        "path": str(path.relative_to(rc.REPO_ROOT)).replace("\\", "/"),
        "configuration": configuration,
        "scenario": scenario,
        "detections": detections,
        "placed": int(scenario.get("num_goals_placed", 0)),
        "goal_positions": [list(cell) for cell in (scenario.get("goal_positions") or [])],
        "agent_positions": [list(cell) for cell in (scenario.get("agent_positions") or [])],
        "detected": len(detections),
        "on_time": int(metrics.on_time),
        "expired": int(metrics.expired),
        "completed": int(metrics.completed),
        "probe_placed": probe_placed,
        "probe_summary": probe_summary,
    }


def run() -> int:
    manifest: dict[str, Any] = {}
    out: dict[str, Any] = {
        "round": rc.ROUND,
        "start_sha": rc.START_SHA,
        "batches": {},
        "identity_checks": {},
        "note": (
            "the two batches share no seed, no week and no driver; they are reported side "
            "by side and never pooled into one paired experiment"
        ),
    }
    per_seed: dict[str, dict[int, dict[str, Any]]] = {}

    for key, batch in rc.BATCHES.items():
        cells: dict[tuple[str, int], dict[str, Any]] = {}
        for arm in batch["arms"]:
            for seed in batch["seeds"]:
                paths = rc.batch_paths(batch, arm, seed)
                lifecycle = paths["lifecycle"]
                if not lifecycle.exists():
                    raise FileNotFoundError(lifecycle)
                probe_path = paths.get("probe")
                cell = read_cell(lifecycle, probe_path, float(batch["steps"]))
                rc.register_input(lifecycle, manifest)
                if probe_path is not None and probe_path.exists():
                    rc.register_input(probe_path, manifest)
                cells[(arm, seed)] = cell

        arms = list(batch["arms"])
        per_arm = {}
        for arm in arms:
            rows = [cells[(arm, seed)] for seed in batch["seeds"]]
            per_arm[arm] = {
                "episodes": len(rows),
                "placed": sum(row["placed"] for row in rows),
                "detected": sum(row["detected"] for row in rows),
                "undetected": sum(row["placed"] - row["detected"] for row in rows),
                "on_time": sum(row["on_time"] for row in rows),
                "expired": sum(row["expired"] for row in rows),
                "episodes_with_a_miss": sum(1 for row in rows if row["placed"] > row["detected"]),
                "detection_step_quantiles": rc.quantiles(
                    [det["step"] for (arm_row, seed), cell in cells.items()
                     if arm_row == arms[0] for det in cell["detections"].values()]
                ),
            }

        # every arm of a batch shares one scenario and one search record: check it, do not
        # assume it, because the attribution below reads the search side from one arm only.
        # Only search-side identity belongs here; task outcomes differ by design (that is the
        # intervention D1 measured) and must not read as drift.
        drift = []
        for seed in batch["seeds"]:
            reference = cells[(arms[0], seed)]
            for arm in arms[1:]:
                row = cells[(arm, seed)]
                if (row["goal_positions"] != reference["goal_positions"]
                        or row["agent_positions"] != reference["agent_positions"]
                        or row["scenario"].get("map_sha256") != reference["scenario"].get("map_sha256")
                        or {k: (v["grid_position"], v["step"])
                            for k, v in row["detections"].items()}
                        != {k: (v["grid_position"], v["step"])
                            for k, v in reference["detections"].items()}):
                    drift.append({"seed": seed, "arm": arm})

        misses = []
        for seed in batch["seeds"]:
            reference = cells[(arms[0], seed)]
            detected_ids = set(reference["detections"])
            for index in range(len(reference["goal_positions"])):
                target = rc.goal_label(index)
                if target in detected_ids:
                    continue
                detection = reference["detections"].get(target)
                misses.append({
                    "seed": seed,
                    "goal_index": index,
                    "target_id": target,
                    "placed_at": list(reference["goal_positions"][index]),
                    "emitted_any_detection": detection is not None,
                    "also_missed_in_every_arm": all(
                        set(cells[(arm, seed)]["detections"]) == detected_ids for arm in arms),
                })

        out["batches"][key] = {
            "label": batch["label"],
            "seeds": [min(batch["seeds"]), max(batch["seeds"])],
            "seed_count": len(list(batch["seeds"])),
            "steps": batch["steps"],
            "detection_radius": cells[(arms[0], min(batch["seeds"]))]["configuration"].get(
                "detection_radius"),
            "search_backend": cells[(arms[0], min(batch["seeds"]))]["configuration"].get(
                "search_backend"),
            "piloc_checkpoint": cells[(arms[0], min(batch["seeds"]))]["configuration"].get(
                "piloc_checkpoint"),
            "scenario_source": reference_source(cells, arms[0], min(batch["seeds"]), batch),
            "per_arm": per_arm,
            "arms_share_scenario_and_detections": not drift,
            "arm_drift": drift[:10],
            "arm_drift_count": len(drift),
            "undetected_goals": misses,
            "search_record_arm": batch["search_record_arm"],
        }
        per_seed[key] = {seed: cells[(batch["search_record_arm"], seed)] for seed in batch["seeds"]}

    # --- is the id a usable handle on a placement, and do the targets move? ----------------
    reset_check = {"cells_checked": 0, "mismatched": []}
    probe_check = {"cells_checked": 0, "mismatched": []}
    mobility = {"detections": 0, "cell_equals_placement": 0, "drift_over_step_violations": [],
                "drift_by_detection_step": {"early_le_10_steps": [], "late_gt_10_steps": []},
                "placements_never_matching_any_detected_cell": 0}
    for key, rows in per_seed.items():
        for seed, cell in rows.items():
            placements = cell["goal_positions"]
            detected_cells = {tuple(row["grid_position"]) for row in cell["detections"].values()}
            mobility["placements_never_matching_any_detected_cell"] += sum(
                1 for place in placements if tuple(place) not in detected_cells)
            for row in cell["probe_placed"]:
                probe_check["cells_checked"] += 1
                index = int(row["target_id"].rsplit("-", 1)[1])
                if index >= len(placements) or list(placements[index]) != row["grid_position"]:
                    probe_check["mismatched"].append({"batch": key, "seed": seed, **row})
            for target, detection in cell["detections"].items():
                if not target.startswith("piloc-e1-target-"):
                    continue
                index = int(target.rsplit("-", 1)[1])
                reset_check["cells_checked"] += 1
                if index >= len(placements):
                    reset_check["mismatched"].append({"batch": key, "seed": seed, "target": target,
                                                      "why": "id index beyond the placement list"})
                    continue
                drift = rc.chebyshev(tuple(detection["grid_position"]), tuple(placements[index]))
                step = detection["step"]
                mobility["detections"] += 1
                mobility["cell_equals_placement"] += int(drift == 0)
                if (step - 1) < drift:
                    mobility["drift_over_step_violations"].append(
                        {"batch": key, "seed": seed, "target": target, "step": step, "drift": drift})
                (mobility["drift_by_detection_step"]["early_le_10_steps"] if step <= 10
                 else mobility["drift_by_detection_step"]["late_gt_10_steps"]).append(drift)
    mobility["drift_quantiles"] = {
        bucket: rc.quantiles(values) for bucket, values in mobility["drift_by_detection_step"].items()}
    mobility.pop("drift_by_detection_step")
    out["identity_checks"]["probe_placement_id_indexes_the_logged_list"] = probe_check
    out["identity_checks"]["every_detection_id_is_a_reset_placement_id"] = {
        "checked": reset_check["cells_checked"], "mismatched": len(reset_check["mismatched"])}
    out["target_mobility"] = mobility

    # --- the ground-truth caveat, stated as a field, not as a paragraph -------------------
    out["ground_truth_provenance"] = {
        "goal_positions_and_agent_starts_and_obstacle_grid": (
            "simulation truth written by the run for the record; the search policy at step t "
            "saw only its own observation, so any distance computed from these values is a "
            "post-hoc audit statement, not a statement about what the policy could know"
        ),
        "detection_events": (
            "what the system actually emitted; used as an anchor set, i.e. positions the "
            "search robots demonstrably reached, and the only policy-visible evidence here"
        ),
    }

    out["inputs_read"] = {"count": len(manifest), "files": manifest}
    path = rc.write_json("BATCH_RECONCILIATION.json", out)

    for key, batch in out["batches"].items():
        print(f"== {key}  {batch['label']}")
        print(f"   seeds {batch['seeds'][0]}..{batch['seeds'][1]}  detection_radius="
              f"{batch['detection_radius']}  backend={batch['search_backend']}")
        for arm, row in batch["per_arm"].items():
            print(f"   {arm:>9}: placed {row['placed']} detected {row['detected']} "
                  f"undetected {row['undetected']} on_time {row['on_time']} expired {row['expired']} "
                  f"episodes_with_a_miss {row['episodes_with_a_miss']}")
        print(f"   arms share scenario+detections: {batch['arms_share_scenario_and_detections']} "
              f"(drift {batch['arm_drift_count']}); misses {len(batch['undetected_goals'])}")
    print(json.dumps({
        "probe_id_matches_logged_list": {
            "checked": probe_check["cells_checked"], "mismatched": len(probe_check["mismatched"])},
        "detection_ids_are_reset_ids": out["identity_checks"][
            "every_detection_id_is_a_reset_placement_id"],
        "target_mobility": {
            "detections": mobility["detections"],
            "cell_still_at_placement": mobility["cell_equals_placement"],
            "drift_exceeding_elapsed_steps": len(mobility["drift_over_step_violations"]),
            "drift_quantiles": mobility["drift_quantiles"],
        },
    }, ensure_ascii=False, indent=1))
    print(f"wrote {path.name}; inputs hashed: {len(manifest)}")
    return 0


def reference_source(cells: dict, arm: str, seed: int, batch: dict) -> str:
    return str(cells[(arm, seed)]["scenario"].get("source"))


if __name__ == "__main__":
    raise SystemExit(run())
