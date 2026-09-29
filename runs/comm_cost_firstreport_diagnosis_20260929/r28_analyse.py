"""ROUND28 analysis: the paired sample table, the per-sample verification and the descriptive stats.

Zero launches. Every intervention cell is compared with its own scene's baseline
(intervention minus baseline). The four labels the execution basis keeps distinct -- the message
was emitted, the message was accepted, the allocation chain changed, the task outcome changed --
are recorded separately and never merged. Actor-visible inputs are stored apart from the
analysis metadata; no receiver-side or outcome field enters the input block.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import r28_common as common
from integrated_sar.comm_learning import FEATURE_NAMES
from r28_verify import verify_sample

# ------------------------------------------------------------------ loading -------------
def _row(kind: str, scene: int, target: str | None = None) -> dict[str, Any]:
    path = common.row_path(kind, scene, target)
    if not path.exists():
        raise RuntimeError(f"{common.cell_name(kind, scene, target)}: no final row on disk")
    return common.read_json(path)


def _load_all() -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    baselines = {scene: _row("baseline", scene) for scene in common.SCENARIOS}
    disabled = _row("disabled", common.GATE_SCENARIO)
    points = common.read_json(common.OUTPUT_DIR / "INTERVENTION_POINTS.json")
    point_index = {(int(entry["scene"]), point["target"]): point
                   for entry in points["scenes"] for point in entry["intervention_points"]}
    return baselines, [disabled], point_index


# ------------------------------------------------------------------ per-target chain ------------
def _chain(row: dict[str, Any], target: str) -> dict[str, Any]:
    records = list(common.iter_jsonl(Path(row["artefacts"]["lifecycle"])))
    assignments = [{"t": float(r["timestamp"]), "agents": list(r.get("assigned_agents") or []),
                    "valid_until": (r.get("detail") or {}).get("valid_until")}
                   for r in records
                   if r.get("target_id") == target and r.get("event_type") == "assignment"]
    work: list[dict[str, Any]] = []
    for record in records:
        if record.get("target_id") != target or record.get("event_type") != "progress":
            continue
        agents = list(record.get("assigned_agents") or [])
        stamp = float(record["timestamp"])
        if work and work[-1]["agents"] == agents and work[-1]["to"] == stamp - 1.0:
            work[-1]["to"] = stamp
            work[-1]["progress_at_end"] = float(record.get("progress") or 0.0)
        else:
            work.append({"agents": agents, "from": stamp, "to": stamp,
                         "progress_at_end": float(record.get("progress") or 0.0)})
    return {"assignment_events": assignments, "work_segments": work}


def _chain_changed(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    assignments_changed = before["assignment_events"] != after["assignment_events"]
    work_changed = before["work_segments"] != after["work_segments"]
    return {"assignment_chain_changed": assignments_changed,
            "work_pattern_changed": work_changed,
            "baseline_assignments": before["assignment_events"],
            "intervention_assignments": after["assignment_events"],
            "baseline_work": before["work_segments"],
            "intervention_work": after["work_segments"]}


def _delivery_summary(entry: dict[str, Any]) -> dict[str, int]:
    deliveries = entry.get("deliveries") or []
    return {
        "accepted": sum(1 for d in deliveries if d["accepted"]),
        "superseded": sum(1 for d in deliveries if d["reason"] == "delivery_superseded"),
        "other_or_dropped": sum(1 for d in deliveries
                                if not d["accepted"] and d["reason"] != "delivery_superseded"),
        "deliveries_recorded": len(deliveries),
    }


# ------------------------------------------------------------------ one sample -------------
def build_sample(scene: int, target: str, step: int, point: dict[str, Any],
                 baseline: dict[str, Any]) -> dict[str, Any]:
    cell = _row("intervention", scene, target)
    evidence = cell["intervention_evidence"]["per_target"][target]

    delta_y = int(cell["on_time"]) - int(baseline["on_time"])
    delta_c = int(cell["spent_bytes"]) - int(baseline["spent_bytes"])

    base_outcomes = baseline["target_outcomes"]
    cell_outcomes = cell["target_outcomes"]
    base_on_time = set(baseline["goal_identity"]["on_time_goal_ids"])
    cell_on_time = set(cell["goal_identity"]["on_time_goal_ids"])
    gained = sorted(cell_on_time - base_on_time)
    lost = sorted(base_on_time - cell_on_time)

    before, after = base_outcomes.get(target, {}), cell_outcomes.get(target, {})
    sends = evidence.get("real_sends") or []
    override_sends = [s for s in sends if int(s["step"]) == int(step)]
    delivery = _delivery_summary(evidence)
    chain = _chain_changed(_chain(baseline, target), _chain(cell, target))

    verification = verify_sample(scene, target, step, point)

    outcome_change = ("on_time_to_on_time" if before.get("on_time") and after.get("on_time")
                      else "lost_to_on_time" if after.get("on_time")
                      else "on_time_to_lost" if before.get("on_time")
                      else "lost_to_lost")

    return {
        "scene": scene, "target": target, "step": int(step),
        "launches": {"baseline": baseline["attempt"], "intervention": cell["attempt"]},
        # ---- actor-visible inputs (kept strictly apart from the analysis metadata) ----
        "actor_input": {
            "feature_layout": point["feature_layout"],
            "feature_names": list(FEATURE_NAMES),
            "features": point["features"],
            "original_send_probability": point["send_probability"],
            "original_action": point["original_action"],
            "first_report_time_identity": {
                "step": int(step), "discovered_at": point["discovered_at"],
                "first_candidate_step": point["first_candidate_step"],
                "candidate_steps": point["candidate_steps"],
            },
            "sequential_budget_state": {
                "attempt_charge_bytes": point["attempt_charge_bytes"],
                "budget_before_bytes": point["budget_before_bytes"],
                "honours_this_candidate": point["honours_this_candidate"],
            },
            "note": "sender-side visible state only; no receiver truth, no target identity as a "
                    "feature, no outcome field",
        },
        # ---- analysis metadata ----
        "analysis_metadata": {
            "baseline_tape_row_sha256": point["baseline_tape_row_sha256"],
            "intervention_cell": common.cell_name("intervention", scene, target),
            "baseline_cell": common.cell_name("baseline", scene),
        },
        # ---- results (intervention minus baseline) ----
        "result": {
            "delta_y_on_time": delta_y,
            "delta_c_bytes": delta_c,
            "baseline_on_time": baseline["on_time"], "intervention_on_time": cell["on_time"],
            "baseline_spent_bytes": baseline["spent_bytes"],
            "intervention_spent_bytes": cell["spent_bytes"],
            "baseline_emissions": baseline["emissions"],
            "intervention_emissions": cell["emissions"],
            "on_time_ids_gained": gained, "on_time_ids_lost": lost,
            "other_targets_affected": sorted((set(gained) | set(lost)) - {target}),
        },
        "intervened_target": {
            "outcome_change": outcome_change,
            "baseline": {key: before.get(key) for key in
                         ("discovered_at", "deadline", "completed_at", "on_time", "expired")},
            "intervention": {key: after.get(key) for key in
                             ("discovered_at", "deadline", "completed_at", "on_time", "expired")},
            "emission": {
                "override_step": int(step),
                "override_emissions": len(override_sends),
                "emission_step": override_sends[0]["step"] if override_sends else None,
                "packets": override_sends[0]["packets"] if override_sends else None,
                "packets_lost": override_sends[0]["packets_lost"] if override_sends else None,
                "cost_bytes": override_sends[0]["cost_bytes"] if override_sends else None,
                "first": override_sends[0]["first"] if override_sends else None,
                "refusals": evidence.get("real_refusals") or [],
            },
            "deliveries": delivery,
        },
        "four_labels": {
            "emitted": bool(override_sends),
            "accepted": delivery["accepted"] > 0,
            "allocation_changed": bool(chain["assignment_chain_changed"]
                                       or chain["work_pattern_changed"]),
            "task_outcome_changed": outcome_change != "on_time_to_on_time",
        },
        "allocation_and_work": chain,
        "verification": {
            "prefix_identical": verification["prefix_identical"],
            "prefix_rows_compared": verification["prefix_rows_compared"],
            "prefix_mismatch_steps": verification["prefix_mismatch_steps"],
            "opportunity_reproduced": bool(
                verification["opportunity_reproduction"].get("matches_frozen_point")),
            "original_action_reproduced": verification["opportunity_reproduction"].get(
                "original_action") == 0,
            "override_applied_in_cell": verification["override_applied_in_cell"],
            "verified": verification["verified"],
        },
    }


# ------------------------------------------------------------------ panel + stats -------------
def build_panel() -> dict[str, Any]:
    baselines, [disabled], points = _load_all()
    samples: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for entry in common.read_json(common.OUTPUT_DIR / "INTERVENTION_POINTS.json")["scenes"]:
        scene = int(entry["scene"])
        baseline = baselines[scene]
        for point in entry["intervention_points"]:
            target = point["target"]
            path = common.row_path("intervention", scene, target)
            if not path.exists():
                missing.append({"scene": scene, "target": target})
                continue
            samples.append(build_sample(scene, target, int(point["step"]), point, baseline))
    if missing:
        raise RuntimeError(f"incomplete panel: {len(missing)} intervention cells missing, "
                           f"e.g. {missing[:3]}; an incomplete panel is reported incomplete, "
                           "never as a complete case")
    ledger = common.ledger_report()
    return {"samples": samples, "baselines": baselines, "disabled": disabled, "ledger": ledger}


def summarise(panel: dict[str, Any]) -> dict[str, Any]:
    samples = panel["samples"]
    deltas_y = [s["result"]["delta_y_on_time"] for s in samples]
    deltas_c = [s["result"]["delta_c_bytes"] for s in samples]

    def counts(values: list[float]) -> dict[str, int]:
        return {"positive": sum(1 for v in values if v > 0),
                "zero": sum(1 for v in values if v == 0),
                "negative": sum(1 for v in values if v < 0)}

    four = {
        label: sum(1 for s in samples if s["four_labels"][label])
        for label in ("emitted", "accepted", "allocation_changed", "task_outcome_changed")}
    cross = {}
    for s in samples:
        key = tuple(s["four_labels"][label] for label in
                    ("emitted", "accepted", "allocation_changed", "task_outcome_changed"))
        cross[str(key)] = cross.get(str(key), 0) + 1

    outcome_changes: dict[str, int] = {}
    for s in samples:
        outcome_changes[s["intervened_target"]["outcome_change"]] = \
            outcome_changes.get(s["intervened_target"]["outcome_change"], 0) + 1

    per_scene = []
    for scene in common.SCENARIOS:
        rows = [s for s in samples if s["scene"] == scene]
        if not rows:
            continue
        per_scene.append({
            "scene": scene,
            "interventions": len(rows),
            "delta_y_values": [s["result"]["delta_y_on_time"] for s in rows],
            "delta_c_values": [s["result"]["delta_c_bytes"] for s in rows],
            "baseline_on_time": rows[0]["result"]["baseline_on_time"],
            "baseline_spent_bytes": rows[0]["result"]["baseline_spent_bytes"],
            "outcome_changes": sorted({s["intervened_target"]["outcome_change"] for s in rows}),
        })

    probabilities = [s["actor_input"]["original_send_probability"] for s in samples]
    steps = [s["actor_input"]["first_report_time_identity"]["step"] for s in samples]
    accepted = [s for s in samples if s["four_labels"]["accepted"]]
    not_accepted = [s for s in samples if not s["four_labels"]["accepted"]]

    return {
        "round": "28",
        "section": "descriptive statistics of the paired first-report samples",
        "samples": len(samples),
        "scenes": len({s["scene"] for s in samples}),
        "independence_warning": "these are closed-loop interventions on 16 scenes under a fixed "
                                "greedy follow-up policy; they are not 96 independent scenes and "
                                "per-scene gains are not additive evidence for joint sends",
        "delta_y": {"counts": counts(deltas_y),
                    "values": deltas_y,
                    "mean": round(statistics.fmean(deltas_y), 6) if deltas_y else None,
                    "min": min(deltas_y) if deltas_y else None,
                    "max": max(deltas_y) if deltas_y else None},
        "delta_c": {"counts": counts(deltas_c),
                    "values": deltas_c,
                    "mean": round(statistics.fmean(deltas_c), 4) if deltas_c else None,
                    "min": min(deltas_c) if deltas_c else None,
                    "max": max(deltas_c) if deltas_c else None,
                    "total": sum(deltas_c)},
        "four_labels": four,
        "four_label_cross_tab": cross,
        "intervened_target_outcome_changes": outcome_changes,
        "accepted_vs_task_benefit": {
            "accepted_samples": len(accepted),
            "not_accepted_samples": len(not_accepted),
            "accepted_with_outcome_change": sum(1 for s in accepted
                                                if s["four_labels"]["task_outcome_changed"]),
            "not_accepted_with_outcome_change": sum(1 for s in not_accepted
                                                    if s["four_labels"]["task_outcome_changed"]),
            "accepted_with_allocation_change": sum(1 for s in accepted
                                                   if s["four_labels"]["allocation_changed"]),
            "not_accepted_with_allocation_change": sum(1 for s in not_accepted
                                                       if s["four_labels"]["allocation_changed"]),
        },
        "prefix_verification": {
            "verified_samples": sum(1 for s in samples if s["verification"]["verified"]),
            "prefix_identical_samples": sum(1 for s in samples
                                            if s["verification"]["prefix_identical"]),
            "opportunity_reproduced_samples": sum(1 for s in samples
                                                  if s["verification"]["opportunity_reproduced"]),
            "override_applied_samples": sum(1 for s in samples
                                            if s["verification"]["override_applied_in_cell"]),
        },
        "actor_input_distribution": {
            "send_probability": {"min": min(probabilities), "max": max(probabilities),
                                 "median": round(statistics.median(probabilities), 6)},
            "first_report_step": {"min": min(steps), "max": max(steps),
                                  "median": statistics.median(steps)},
        },
        "per_scene": per_scene,
    }


def write_analysis() -> tuple[Path, Path]:
    panel = build_panel()
    summary = summarise(panel)
    identities = {scene: row["code_identity_freeze_sha256"]
                  for scene, row in panel["baselines"].items()}
    identities[("disabled", common.GATE_SCENARIO)] = \
        panel["disabled"]["code_identity_freeze_sha256"]
    for entry in common.read_json(common.OUTPUT_DIR / "INTERVENTION_POINTS.json")["scenes"]:
        for point in entry["intervention_points"]:
            row = _row("intervention", int(entry["scene"]), point["target"])
            identities[("intervention", int(entry["scene"]), point["target"])] = \
                row["code_identity_freeze_sha256"]
    distinct = sorted(set(identities.values()))
    payload = {"summary": summary,
               "samples": panel["samples"],
               "identity": {"distinct_launch_identities": distinct,
                            "single_identity": len(distinct) == 1},
               "ledger": {key: value for key, value in panel["ledger"].items()
                          if key not in ("entries", "results")}}
    samples_path = common.write_json("PAIRED_SAMPLES.json", payload)
    summary_path = common.write_json("ANALYSIS_SUMMARY.json", summary)
    return samples_path, summary_path


if __name__ == "__main__":
    samples_path, summary_path = write_analysis()
    summary = common.read_json(summary_path)
    print(json.dumps({
        "samples": summary["samples"], "scenes": summary["scenes"],
        "delta_y_counts": summary["delta_y"]["counts"],
        "delta_c_counts": summary["delta_c"]["counts"],
        "four_labels": summary["four_labels"],
        "prefix_verified": summary["prefix_verification"]["verified_samples"],
        "outcome_changes": summary["intervened_target_outcome_changes"],
    }, ensure_ascii=False, indent=1))
