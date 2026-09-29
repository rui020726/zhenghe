"""ROUND27 comparison: the five-path table, per-target timelines and prefix consistency.

Zero launches: every artefact already exists. The comparison hard-fails when any of the five
path rows is missing (an incomplete panel is reported incomplete, never as a complete case),
reports E0-L, E5-L and E05-L on the named targets and the whole scene, and records whether the
pre-intervention prefix of each intervention path is identical to L's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import r27_common as common

TARGETS = common.INTERVENTION_TARGETS
COMPARISONS = ("E0", "E5", "E05")


# ------------------------------------------------------------------ panel loading -------------
def load_panel() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name in common.PATH_ORDER:
        path = common.row_path(name, 1)
        if not path.exists():
            raise RuntimeError(f"{name}: no final row on disk; an incomplete panel is reported "
                               "incomplete and never as a complete case")
        rows[name] = common.read_json(path)
    return rows


# ------------------------------------------------------------------ task and cost deltas -------------
def _goal_sets(row: dict[str, Any]) -> tuple[set[str], set[str]]:
    identity = row["goal_identity"]
    return set(identity["on_time_goal_ids"]), set(identity["expired_goal_ids"])


def task_and_cost_comparison(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref_on_time, ref_expired = _goal_sets(reference)
    cand_on_time, cand_expired = _goal_sets(candidate)
    gained = sorted(cand_on_time - ref_on_time)
    lost = sorted(ref_on_time - cand_on_time)
    ref_outcomes = reference["target_outcomes"]
    cand_outcomes = candidate["target_outcomes"]
    per_target: dict[str, Any] = {}
    for target in TARGETS:
        before, after = ref_outcomes.get(target, {}), cand_outcomes.get(target, {})
        per_target[target] = {
            "reference": {"on_time": before.get("on_time"), "expired": before.get("expired"),
                          "completed_at": before.get("completed_at"),
                          "discovered_at": before.get("discovered_at"),
                          "deadline": before.get("deadline")},
            "candidate": {"on_time": after.get("on_time"), "expired": after.get("expired"),
                          "completed_at": after.get("completed_at"),
                          "discovered_at": after.get("discovered_at"),
                          "deadline": after.get("deadline")},
            "task_change": _task_change(before, after),
        }
    return {
        "whole_scene": {
            "reference_on_time": reference["on_time"],
            "candidate_on_time": candidate["on_time"],
            "on_time_delta": candidate["on_time"] - reference["on_time"],
            "on_time_goal_ids_gained": gained,
            "on_time_goal_ids_lost": lost,
            "reference_expired_ids": sorted(ref_expired),
            "candidate_expired_ids": sorted(cand_expired),
            "other_targets_affected": sorted((set(gained) | set(lost)) - set(TARGETS)),
            "expired_delta": candidate["expired"] - reference["expired"],
            "completed_delta": candidate["completed"] - reference["completed"],
        },
        "per_named_target": per_target,
        "cost": {
            "reference_spent_bytes": reference["spent_bytes"],
            "candidate_spent_bytes": candidate["spent_bytes"],
            "cost_delta_bytes": candidate["spent_bytes"] - reference["spent_bytes"],
            "reference_emissions": reference["emissions"],
            "candidate_emissions": candidate["emissions"],
            "emissions_delta": candidate["emissions"] - reference["emissions"],
            "candidate_budget_refused_sends": candidate["budget_refused_sends"],
            "candidate_capacity_refused_sends": candidate["capacity_refused_sends"],
            "candidate_refusal_sequence": candidate["refusal_sequence"],
        },
    }


def _task_change(before: dict[str, Any], after: dict[str, Any]) -> str:
    if bool(before.get("on_time")) and bool(after.get("on_time")):
        return "on_time_to_on_time"
    if not bool(before.get("on_time")) and bool(after.get("on_time")):
        return "lost_to_on_time"
    if bool(before.get("on_time")) and not bool(after.get("on_time")):
        return "on_time_to_lost"
    return "lost_to_lost"


# ------------------------------------------------------------------ allocation continuity -------------
def allocation_chain(arm_row: dict[str, Any], target: str) -> dict[str, Any]:
    """Assignment events and work segments for one target of one path, from its lifecycle log."""
    records = list(common.iter_jsonl(Path(arm_row["artefacts"]["lifecycle"])))
    assignments = [{"t": float(r["timestamp"]), "agents": list(r.get("assigned_agents") or []),
                    "valid_until": (r.get("detail") or {}).get("valid_until"),
                    "command_id": (r.get("detail") or {}).get("command_id")}
                   for r in records
                   if r.get("event_type") == "assignment" and r.get("target_id") == target]
    work: list[dict[str, Any]] = []
    for record in records:
        if record.get("target_id") != target or record.get("event_type") != "progress":
            continue
        agents = list(record.get("assigned_agents") or [])
        stamp = float(record["timestamp"])
        if work and work[-1]["agents"] == agents and work[-1]["to"] == stamp - 1.0:
            work[-1]["to"] = stamp
            work[-1]["progress_at_end"] = float(record.get("progress") or 0.0)
            work[-1]["stage_at_end"] = int(record.get("stage_index") or 0)
        else:
            work.append({"agents": agents, "from": stamp, "to": stamp,
                         "progress_at_end": float(record.get("progress") or 0.0),
                         "stage_at_end": int(record.get("stage_index") or 0)})
    gap = None
    if work:
        stops = [segment["to"] for segment in work if segment["agents"]]
        if stops:
            last_stop = max(stops)
            later = [segment for segment in work if segment["from"] > last_stop]
            gap = {"work_stops_at": last_stop,
                   "work_resumes_at": min((segment["from"] for segment in later), default=None)}
    return {"target": target, "assignment_events": assignments, "work_segments": work,
            "interruption": gap}


# ------------------------------------------------------------------ prefix consistency -------------
def prefix_consistency(reference_row: dict[str, Any], candidate_row: dict[str, Any],
                       first_override_step: int | None) -> dict[str, Any]:
    """Compare the two episodes' tapes and transcripts up to the first divergence.

    The intervention's extra emission consumes channel loss draws, so the paths legitimately
    diverge from the override step onward. Everything before it must be identical.
    """
    reference_tape = _tape(Path(reference_row["artefacts"]["tape"]))
    candidate_tape = _tape(Path(candidate_row["artefacts"]["tape"]))
    divergent_step = None
    for index, (left, right) in enumerate(zip(reference_tape, candidate_tape)):
        if _tape_without_override_block(left) != _tape_without_override_block(right):
            divergent_step = int(left["step"])
            break
    reference_transcript = list(common.iter_jsonl(Path(reference_row["artefacts"]["transcript"])))
    candidate_transcript = list(common.iter_jsonl(Path(candidate_row["artefacts"]["transcript"])))
    transcript_divergence = None
    for index, (left, right) in enumerate(zip(reference_transcript, candidate_transcript)):
        if left != right:
            transcript_divergence = {"record_index": index, "step": int(left.get("step", -1)),
                                     "kind": left.get("kind"), "target": left.get("target")}
            break
    return {
        "first_divergent_tape_step": divergent_step,
        "first_override_step": first_override_step,
        "prefix_matches_until_divergence": (
            divergent_step is not None and first_override_step is not None
            and divergent_step >= first_override_step),
        "transcript_first_divergence": transcript_divergence,
        "note": "different emissions change the channel RNG consumption, so the post-override "
                "paths are not the same per-message loss batch and outcome changes are not "
                "attributable to one accepted message",
    }


def _tape(path: Path) -> list[dict[str, Any]]:
    return list(common.iter_jsonl(path))


def _tape_without_override_block(tape: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in tape.items() if key != "first_report_overrides"}


# ------------------------------------------------------------------ panel assembly -------------
def compare_panel() -> dict[str, Any]:
    rows = load_panel()
    reference = rows["L"]
    comparisons: dict[str, Any] = {}
    for name in COMPARISONS:
        candidate = rows[name]
        evidence = candidate["intervention_evidence"]
        first_override = min(
            (step for entry in evidence["per_target"].values() for step in entry["override_steps"]),
            default=None)
        comparisons[f"{name}-L"] = {
            "path": name,
            "override_steps": {target: entry["override_steps"]
                               for target, entry in evidence["per_target"].items()},
            "task_and_cost": task_and_cost_comparison(reference, candidate),
            "allocation_continuity": {
                target: {
                    "L": allocation_chain(reference, target),
                    name: allocation_chain(candidate, target),
                } for target in TARGETS},
            "prefix_consistency": prefix_consistency(reference, candidate, first_override),
            "intervention_effectiveness": evidence,
        }
    return {
        "round": "27", "section": "five-path comparison", "scene": common.SCENE,
        "paths": list(common.PATH_ORDER),
        "reference": "L",
        "rows": {name: {
            "on_time": row["on_time"], "expired": row["expired"],
            "on_time_goal_ids": row["goal_identity"]["on_time_goal_ids"],
            "expired_goal_ids": row["goal_identity"]["expired_goal_ids"],
            "spent_bytes": row["spent_bytes"], "emissions": row["emissions"],
            "target_outcomes": {target: {key: value for key, value in
                                         row["target_outcomes"].get(target, {}).items()
                                         if key != "final_record"}
                                for target in TARGETS},
        } for name, row in rows.items()},
        "comparisons": comparisons,
    }


def write_comparison() -> dict[str, Any]:
    payload = compare_panel()
    common.write_json("FIVE_PATH_COMPARISON.json", payload)
    return payload


if __name__ == "__main__":
    result = write_comparison()
    for name, row in result["rows"].items():
        print(f"{name}: on_time={row['on_time']} expired={row['expired']} "
              f"bytes={row['spent_bytes']} emissions={row['emissions']}")
    for key, comparison in result["comparisons"].items():
        whole = comparison["task_and_cost"]["whole_scene"]
        print(f"{key}: on_time delta={whole['on_time_delta']} "
              f"gained={whole['on_time_goal_ids_gained']} lost={whole['on_time_goal_ids_lost']} "
              f"bytes delta={comparison['task_and_cost']['cost']['cost_delta_bytes']} "
              f"prefix ok={comparison['prefix_consistency']['prefix_matches_until_divergence']}")
