"""Read-only ROUND63 support and matching audit over saved R60/R61/R62 artifacts."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
R2 = HERE.parents[0]
REPO = HERE.parents[2]
R60 = R2 / "round60_partitioned_actor_head_ablation"
R61 = R2 / "round61_early_report_value_intervention"
R62 = R2 / "round62_early_report_information_audit"
FOCUS_SCENE = 1700027
FOCUS_TARGET = "piloc-e1-target-5"
EARLY_CUTOFF = 147
FEATURE_NAMES = [
    "age_steps", "since_last_emission_time", "sent_count", "pending_event",
    "changed_view", "first_emission", "stage_changed", "requirements_changed",
    "members_changed", "progress_delta_abs", "revision_delta",
    "remaining_time_over_horizon", "urgency_now", "has_future_requirements",
    "queue_backlog_targets", "budget_remaining_fraction",
    "capacity_remaining_fraction", "subscribers", "step_over_horizon",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def load_saved_candidates():
    """Extract unique eligible decisions using only sender-side pre-action data."""
    tape_root = R60 / "raw" / "logs"
    tape_paths = sorted(tape_root.glob("v2_r60_eval_new_s33_greedy_s*_a1_tape.jsonl"))
    if len(tape_paths) != 32:
        raise AssertionError(f"expected 32 frozen scene tapes, found {len(tape_paths)}")
    tape_hashes = {path: sha256(path) for path in tape_paths}
    records = {}
    duplicates_removed = 0
    rejected_missing_deadline = 0
    for tape in tape_paths:
        scene = int(tape.name.split("_s")[-1].split("_a1")[0])
        rp = tape.with_name(tape.name.replace("_tape.jsonl", "_rp.jsonl"))
        if not rp.is_file():
            raise AssertionError(f"missing paired lifecycle source for scene {scene}")
        detections = {}
        for event in read_jsonl(rp):
            if event.get("event_type") == "detection" and event.get("target_id"):
                detections.setdefault(event["target_id"], event)
        for row in read_jsonl(tape):
            for target in row.get("free_first_reports", []):
                if target not in row.get("targets", []):
                    raise AssertionError("free first-report target absent from tape target ordering")
                ix = row["targets"].index(target)
                features = [float(x) for x in row["features"][ix]]
                if len(features) != 19:
                    raise AssertionError(f"unexpected feature dimension {len(features)}")
                audit = row.get("candidate_audit", {}).get(target, {})
                step = int(row["step"])
                ready = round(features[18] * 1000) - round(features[0])
                eligible = (
                    ready <= EARLY_CUTOFF
                    and features[5] == 1.0
                    and audit.get("first_emission") is True
                    and audit.get("sequentially_affordable_before_action") is True
                    and row.get("send_executable_initial", {}).get(target) is True
                    and row.get("send_block_reason_initial", {}).get(target) == "allowed"
                )
                if not eligible:
                    continue
                if ready != step:
                    raise AssertionError(f"first-report ready/decision mismatch: {scene} {target} {ready}/{step}")
                detected = detections.get(target)
                deadline = remaining = deadline_offset = None
                detected_at = None
                missing_reason = None
                if detected is None or detected.get("detail", {}).get("deadline_offset") is None:
                    missing_reason = "no_detection_deadline_record"
                    rejected_missing_deadline += 1
                else:
                    detected_at = float(detected["timestamp"])
                    deadline_offset = float(detected["detail"]["deadline_offset"])
                    deadline = detected_at + deadline_offset
                    remaining = deadline - (step - 1)
                state = row.get("sender_state", {})
                backlog = float(features[14])
                budget_fraction = float(features[15])
                if state.get("backlog_targets") is not None and float(state["backlog_targets"]) != backlog:
                    raise AssertionError(f"backlog feature/source mismatch: {scene} {target}")
                rec = {
                    "scene": scene,
                    "target": str(target),
                    "ready_step": int(ready),
                    "decision_step": step,
                    "backlog": backlog,
                    "has_future_requirements": bool(features[13]),
                    "budget_fraction": budget_fraction,
                    "remaining_budget_bytes": state.get("remaining_budget_bytes"),
                    "budget_bytes_per_episode": state.get("budget_bytes_per_episode"),
                    "quote_bytes": row.get("attempt_charge_bytes", {}).get(target),
                    "send_executable": row.get("send_executable_initial", {}).get(target),
                    "sequentially_affordable": audit.get("sequentially_affordable_before_action"),
                    "baseline_action": int(row["actions"][target]),
                    "remaining_time": remaining,
                    "deadline": deadline,
                    "deadline_offset": deadline_offset,
                    "detection_step": detected_at,
                    "remaining_time_derivation": None if remaining is None else "deadline - (decision_step - 1)",
                    "remaining_time_missing_reason": missing_reason,
                    "features_19d": features,
                    "source": {
                        "tape_path": relative(tape),
                        "tape_sha256": tape_hashes[tape],
                        "lifecycle_path": relative(rp),
                        "lifecycle_sha256": sha256(rp),
                    },
                }
                key = (scene, str(target))
                if key in records:
                    old = records[key]
                    compare = (old["ready_step"], old["features_19d"], old["quote_bytes"])
                    current = (rec["ready_step"], rec["features_19d"], rec["quote_bytes"])
                    if compare != current:
                        raise AssertionError(f"duplicate target decisions disagree for {key}")
                    duplicates_removed += 1
                    continue
                records[key] = rec
    candidates = sorted(records.values(), key=lambda r: (r["scene"], r["target"]))
    return candidates, {
        "scene_tape_count": len(tape_paths),
        "raw_eligible_rows_before_dedup": len(candidates) + duplicates_removed,
        "duplicate_scene_target_rows_removed": duplicates_removed,
        "unique_scene_target_candidates": len(candidates),
        "missing_deadline_rows": rejected_missing_deadline,
        "tapes": [{"path": relative(p), "sha256": tape_hashes[p]} for p in tape_paths],
    }


def group_summary(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, members in sorted(groups.items(), key=lambda item: item[0]):
        values = [float(r["remaining_time"]) for r in members if r["remaining_time"] is not None]
        out.append({
            "key": {name: value for name, value in zip(keys, key)},
            "candidate_count": len(members),
            "distinct_scene_count": len({r["scene"] for r in members}),
            "scenes": sorted({r["scene"] for r in members}),
            "remaining_time_known_count": len(values),
            "remaining_time_missing_count": len(members) - len(values),
            "remaining_time_min_within_group": min(values) if values else None,
            "remaining_time_max_within_group": max(values) if values else None,
            "remaining_time_distinct_count_within_group": len(set(values)),
        })
    return out


def nearest_covariate_candidates(focus, candidates):
    """Deterministic covariate-only order; extra fields such as outcomes are ignored."""
    rows = [r for r in candidates if int(r["scene"]) != int(focus["scene"])]
    return sorted(rows, key=lambda r: (
        abs(int(r["ready_step"]) - int(focus["ready_step"])),
        abs(float(r["backlog"]) - float(focus["backlog"])),
        int(bool(r["has_future_requirements"]) != bool(focus["has_future_requirements"])),
        abs(float(r["budget_fraction"]) - float(focus["budget_fraction"])),
        int(r["scene"]), str(r["target"]),
    ))


def exact_key(row, keys):
    return tuple(row[name] for name in keys)


def r61_distance_rank(candidates, focus, columns=(13, 14, 15, 18)):
    ranges = {}
    for col in columns:
        values = [r["features_19d"][col] for r in candidates]
        ranges[col] = max(values) - min(values)
    def distance(row):
        total = 0.0
        for col in columns:
            scale = ranges[col]
            if scale:
                total += ((row["features_19d"][col] - focus["features_19d"][col]) / scale) ** 2
        return total
    ordered = sorted((r for r in candidates if r["scene"] != focus["scene"]),
                     key=lambda r: (distance(r), r["scene"], r["target"]))
    return {"ranges": {str(k): v for k, v in ranges.items()},
            "ordered": [{"scene": r["scene"], "target": r["target"], "distance": distance(r)} for r in ordered]}


def main():
    candidates, source_index = load_saved_candidates()
    if len(candidates) != 146:
        raise AssertionError(f"R62 verified candidate count changed: {len(candidates)}")
    focus_matches = [r for r in candidates if r["scene"] == FOCUS_SCENE and r["target"] == FOCUS_TARGET]
    if len(focus_matches) != 1:
        raise AssertionError(f"focus candidate count is {len(focus_matches)}")
    focus = focus_matches[0]

    # Finish every match/group/order calculation before opening any R61 outcome artifact.
    base_keys = ["ready_step", "backlog"]
    extended_keys = base_keys + ["has_future_requirements", "budget_fraction"]
    base_groups = group_summary(candidates, base_keys)
    extended_groups = group_summary(candidates, extended_keys)
    focus_base = [g for g in base_groups if g["key"]["ready_step"] == focus["ready_step"]
                  and g["key"]["backlog"] == focus["backlog"]]
    exact_other_scenes = sorted({r["scene"] for r in candidates
                                 if r["scene"] != focus["scene"]
                                 and r["ready_step"] == focus["ready_step"]
                                 and r["backlog"] == focus["backlog"]})
    nearest = nearest_covariate_candidates(focus, candidates)
    r61_metric = r61_distance_rank(candidates, focus)
    r61_distance_by_key = {(r["scene"], r["target"]): r["distance"] for r in r61_metric["ordered"]}
    r61_distance_by_key[(focus["scene"], focus["target"])] = 0.0
    informative_groups = [g for g in base_groups if g["distinct_scene_count"] >= 2
                          and g["remaining_time_distinct_count_within_group"] >= 2]

    covariate_layer = {
        "focus": {k: focus[k] for k in ("scene", "target", "ready_step", "decision_step", "backlog",
                                         "has_future_requirements", "budget_fraction", "remaining_time",
                                         "deadline", "features_19d", "source")},
        "candidate_counts": source_index,
        "exact_ready_backlog_groups": base_groups,
        "exact_ready_backlog_future_budget_groups": extended_groups,
        "focus_exact_group": focus_base[0] if focus_base else None,
        "focus_exact_controls_other_scenes": len(exact_other_scenes),
        "focus_exact_control_scenes": exact_other_scenes,
        "focus_has_exact_common_support": bool(exact_other_scenes),
        "same_ready_backlog_groups_with_remaining_variation_and_multiple_scenes": informative_groups,
        "r61_frozen_distance_recomputed": {
            "metric": "sum(((x_i-focus_i)/range_i)^2), columns 13/14/15/18 only; zero-range columns contribute zero",
            "range_by_column": r61_metric["ranges"],
            "full_order_first_20": r61_metric["ordered"][:20],
        },
        "nearest_different_scene_by_requested_covariate_order_first_20": [
            {k: r[k] for k in ("scene", "target", "ready_step", "backlog", "has_future_requirements",
                               "budget_fraction", "remaining_time", "deadline", "quote_bytes", "source")}
            for r in nearest[:20]
        ],
        "all_candidates": candidates,
    }

    # Post-hoc outcome fields are joined only after the covariate-only support is frozen.
    r61_freeze = read_json(R61 / "ROUND61_CASE_FREEZE.json")
    r61_pairs = read_json(R61 / "raw" / "ROUND61_PAIRED_ANALYSIS.json")
    raw_cases = [r61_pairs["focus"], *r61_pairs["controls"]]
    case_by_key = {(int(r["scene"]), str(r["target"])): r for r in raw_cases}
    frozen_cases = []
    for case in r61_freeze["cases"]:
        key = (int(case["scene"]), str(case["target"]))
        cov = next((r for r in candidates if (r["scene"], r["target"]) == key), None)
        effect = case_by_key.get(key)
        if cov is None or effect is None:
            frozen_cases.append({"scene": key[0], "target": key[1], "candidate_available": cov is not None,
                                 "posthoc_effect_available": effect is not None})
            continue
        target = effect["target_on_time"]
        system = effect["whole_system_on_time"]
        cost = effect["cost"]
        recomputed_distance = r61_distance_by_key[key]
        frozen_distance = float(case["distance_to_focus"])
        if abs(recomputed_distance - frozen_distance) > 1e-12:
            raise AssertionError(f"R61 distance mismatch for {key}: {recomputed_distance} != {frozen_distance}")
        frozen_cases.append({
            "scene": key[0], "target": key[1], "role": case["role"],
            "ready_step_abs_difference_from_focus": abs(cov["ready_step"] - focus["ready_step"]),
            "backlog_abs_difference_from_focus": abs(cov["backlog"] - focus["backlog"]),
            "future_requirements_equal_focus": cov["has_future_requirements"] == focus["has_future_requirements"],
            "budget_fraction_abs_difference_from_focus": abs(cov["budget_fraction"] - focus["budget_fraction"]),
            "remaining_time_difference_from_focus": (cov["remaining_time"] - focus["remaining_time"]
                                                      if cov["remaining_time"] is not None and focus["remaining_time"] is not None else None),
            "distance_to_focus_recorded_by_r61": case.get("distance_to_focus"),
            "distance_to_focus_recomputed_from_146_candidates": recomputed_distance,
            "exact_ready_backlog_match": cov["ready_step"] == focus["ready_step"] and cov["backlog"] == focus["backlog"],
            "posthoc_outcome_only": {
                "Y_SEND": target["Y_SEND"],
                "Y_SKIP": target["Y_SKIP"],
                "target_delta": target["delta"],
                "system_delta": system["delta"],
                "gained_ids_under_send": effect["new_targets_under_SEND"],
                "lost_ids_under_send": effect["lost_targets_under_SEND"],
                "cost_delta_bytes": cost["delta_bytes"],
            },
        })

    panel = {
        "kind": "round63-matched-slack-feasibility/1",
        "source_identity": {
            "R62_HEAD": "62dcb2419f323461b6aa07bb79ce39e3d17a4ac7",
            "R60_model_file_sha256": "573e4c26a4fee6f01405d4386ca73b518eeb1adb9cd8409bc83661e4113f6c5a",
            "R61_case_freeze_sha256": sha256(R61 / "ROUND61_CASE_FREEZE.json"),
            "R61_raw_paired_analysis_sha256": sha256(R61 / "raw" / "ROUND61_PAIRED_ANALYSIS.json"),
            "feature_source_sha256": {
                "comm_learning.py": sha256(REPO / "integrated_sar" / "comm_learning.py"),
                "comm_policy.py": sha256(REPO / "integrated_sar" / "comm_policy.py"),
                "comm.py": sha256(REPO / "integrated_sar" / "comm.py"),
            },
        },
        "field_contract": {
            "A_actor_indices": {"future": 13, "backlog": 14, "budget_fraction": 15, "ready_step_formula": "round(feature[18]*1000)-round(feature[0])"},
            "raw_deadline_source": "R60 same-scene lifecycle detection timestamp + deadline_offset",
            "remaining_time_formula": "deadline - (decision_step - 1); R30 clock audit verified now=step-1",
            "missing_policy": "missing values stay null and are excluded from remaining-time ranges; never imputed as zero",
            "posthoc_fields_excluded_from_match": ["Y_SEND", "Y_SKIP", "task_delta", "system_delta", "cost_delta", "receiver delivery", "assignment", "final status"],
        },
        "covariate_only_analysis": covariate_layer,
        "R61_eight_cases_joined_after_matching": frozen_cases,
        "selection_guard": {
            "matching_inputs": ["ready_step", "backlog", "has_future_requirements", "budget_fraction", "scene", "target"],
            "outcome_fields_loaded_after_match_and_ranking": True,
            "remaining_time_used_in_nearest_candidate_ranking": False,
            "control_outcomes_used_for_selection": False,
        },
    }
    (HERE / "MATCHED_PANEL.json").write_text(json.dumps(panel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "candidate_count": len(candidates),
        "base_group_count": len(base_groups),
        "focus_ready_backlog": [focus["ready_step"], focus["backlog"]],
        "focus_exact_other_scene_support": len(exact_other_scenes),
        "multi_scene_same_ready_backlog_slack_variable_groups": len(informative_groups),
        "top_covariate_neighbor": {k: nearest[0][k] for k in ("scene", "target", "ready_step", "backlog", "remaining_time")},
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
