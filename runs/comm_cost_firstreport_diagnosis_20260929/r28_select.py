"""ROUND28 selector: the frozen intervention-point list, produced from the baselines alone.

Selection rule (frozen before the first launch of this round; it may not be changed after any
effect is seen):

* universe: every target id that appears as a candidate in a scene's baseline tape, plus every
  target the baseline lifecycle recorded as discovered (so a target that was discovered but never
  offered still gets an exclusion reason);
* opportunity: the first tape step at which the target is *actionable* -- the archived collector's
  sequential shadow gate said this candidate is payable in the order the transport would take it;
* intervention point: exactly when that first legal payable opportunity exists AND the original
  greedy action there is HOLD;
* exclusions carry their own reason and never launch: ``first_legal_opportunity_greedy_send``,
  ``no_legal_payable_opportunity``, ``never_discovered``, ``terminal_before_first_candidate``;
* inputs: the pre-action candidate set, the send history (requested flags), the effective
  sequential mask (actionable/masked), the original greedy action, and time identity (ready_step,
  steps). No completion, expiry, cost or outcome field is read to decide a point. The lifecycle is
  consulted *only* to label exclusions of targets the tape never offered.

The list is written once and frozen (``INTERVENTION_POINTS.json``); the runner then reads it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import r28_common as common
from integrated_sar.comm_learning import FEATURE_NAMES

FEATURE_LAYOUT = "comm-send-state/1"
URGENCY_SCALE_STEPS = 50.0
EPISODE_HORIZON_STEPS = 1000.0

#: The scene's frozen goal-id space: ``num_goals=6`` with ids ``piloc-e1-target-0..5``.
#: Verified against ROUND24's training artefacts of the same scenes (same seeds, search agents
#: independent of the comm channel). It only gives a never-detected goal an exclusion row; it
#: never selects anything.
_NUM_GOALS = int(common.STAGE_B_INHERITED_OVERRIDES["num_goals"])
GOAL_ID_SPACE: tuple[str, ...] = tuple(f"piloc-e1-target-{index}" for index in range(_NUM_GOALS))

SELECTION_RULE = {
    "universe": "candidate ids in the baseline tape plus lifecycle-discovered ids",
    "opportunity": "first tape step where the target is actionable (sequential shadow gate: "
                   "payable in transport order)",
    "intervention_point": "first legal payable opportunity exists AND original greedy action "
                          "there is HOLD",
    "exclusions": ["first_legal_opportunity_greedy_send", "no_legal_payable_opportunity",
                   "never_discovered", "terminal_before_first_candidate"],
    "inputs_allowed": ["pre-action candidate set", "send history (requested flags)",
                       "effective sequential mask (actionable/masked)", "original greedy action",
                       "time identity (step, ready_step)"],
    "inputs_forbidden": ["completion", "expiry", "on-time", "spent bytes", "final cost",
                         "any receiver-side truth"],
    "lifecycle_use": "only to label exclusions of targets the tape never offered; never to pick",
    "frozen_before_launch": True,
}


def _baseline(scene: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row_path = common.row_path("baseline", scene)
    if not row_path.exists():
        raise RuntimeError(f"scene {scene}: no baseline row on disk; selection cannot run")
    row = common.read_json(row_path)
    if "code_identity_freeze_sha256" not in row or row.get("model_identity") is None:
        raise RuntimeError(
            f"scene {scene}: the baseline row at {row_path.name} is not a real episode row "
            "(missing code identity and model identity); refusing to select from a stub")
    tape = list(common.iter_jsonl(Path(row["artefacts"]["tape"])))
    return row, tape


def _lifecycle_targets(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Discovery facts only (time identity); no outcome field is read here."""
    records = list(common.iter_jsonl(Path(row["artefacts"]["lifecycle"])))
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        target = str(record.get("target_id") or "")
        if not target.startswith("piloc-"):
            continue
        if record.get("event_type") == "detection":
            out.setdefault(target, {})["discovered_at"] = float(record["timestamp"])
            out[target]["discovered_by"] = record.get("discovered_by")
    return out


def select_scene(scene: int) -> dict[str, Any]:
    row, tape = _baseline(scene)
    lifecycle = _lifecycle_targets(row)

    tape_targets: list[str] = []
    for line in tape:
        for target in line["targets"]:
            if target not in tape_targets:
                tape_targets.append(target)
    outside = sorted((set(tape_targets) | set(lifecycle)) - set(GOAL_ID_SPACE))
    if outside:
        raise RuntimeError(
            f"scene {scene}: the baseline names targets outside the frozen goal-id space "
            f"{GOAL_ID_SPACE}: {outside}; the space assumption is wrong, not the scene")
    universe = sorted(set(tape_targets) | set(lifecycle) | set(GOAL_ID_SPACE))

    points: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    for target in universe:
        steps = [line for line in tape if target in line["targets"]]
        if not steps:
            reason = ("never_discovered" if target not in lifecycle
                      else "terminal_before_first_candidate")
            exclusions.append({
                "target": target, "reason": reason,
                "evidence": ("no candidate row and no detection record in the baseline"
                             + ("" if target not in lifecycle else
                                f"; lifecycle detection at t={lifecycle[target]['discovered_at']} "
                                "but the relay never offered it as a candidate")),
            })
            continue
        first_actionable = None
        for line in steps:
            if target in line["actionable"]:
                first_actionable = line
                break
        if first_actionable is None:
            masked_steps = sum(1 for line in steps if target in line["masked"])
            exclusions.append({
                "target": target, "reason": "no_legal_payable_opportunity",
                "evidence": f"{len(steps)} candidate steps, {masked_steps} masked by the "
                            "sequential gate; never actionable",
            })
            continue
        step = int(first_actionable["step"])
        index = first_actionable["targets"].index(target)
        action = int(first_actionable["actions"][target])
        charge = int(first_actionable["attempt_charge_bytes"][target])
        shadow = first_actionable["candidate_shadow"][target]
        point = {
            "target": target,
            "step": step,
            "original_action": "SEND" if action == 1 else "HOLD",
            "send_probability": round(float(first_actionable["send_probabilities"][target]), 6),
            "attempt_charge_bytes": charge,
            "budget_before_bytes": float(shadow["budget_before"]),
            "honours_this_candidate": bool(shadow["honours_this_candidate"]),
            "discovered_at": (lifecycle.get(target) or {}).get("discovered_at"),
            "first_candidate_step": int(steps[0]["step"]),
            "candidate_steps": len(steps),
            "features": [float(value) for value in first_actionable["features"][index]],
            "feature_names": list(FEATURE_NAMES),
            "feature_layout": FEATURE_LAYOUT,
            "baseline_tape_row_sha256": hashlib.sha256(json.dumps(
                first_actionable, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        }
        if action == 1:
            exclusions.append({
                "target": target, "reason": "first_legal_opportunity_greedy_send",
                "evidence": f"step {step} was the first legal payable opportunity and the original "
                            f"greedy action there was already SEND (p={point['send_probability']})",
            })
            continue
        points.append(point)

    return {
        "scene": scene,
        "intervention_points": points,
        "exclusions": exclusions,
        "universe_size": len(universe),
        "selected": len(points),
        "excluded": len(exclusions),
        "cap_check": {
            "max_targets_per_scene": common.MAX_TARGETS_PER_SCENE,
            "within_cap": len(points) <= common.MAX_TARGETS_PER_SCENE,
        },
    }


def select_all() -> dict[str, Any]:
    scenes = []
    for scene in common.SCENARIOS:
        entry = select_scene(scene)
        scenes.append(entry)
        if not entry["cap_check"]["within_cap"]:
            raise RuntimeError(
                f"scene {scene}: {entry['selected']} intervention points exceed the per-scene cap "
                f"{common.MAX_TARGETS_PER_SCENE}; the plan is wrong, not the cap")
    total_points = sum(entry["selected"] for entry in scenes)
    plan_launches = 16 + 1 + total_points
    return {
        "round": "28",
        "section": "frozen intervention-point list (selected from baselines only)",
        "selection_rule": SELECTION_RULE,
        "scenes": scenes,
        "totals": {
            "scenes": len(scenes),
            "universe": sum(entry["universe_size"] for entry in scenes),
            "intervention_points": total_points,
            "exclusions": sum(entry["excluded"] for entry in scenes),
            "planned_launches": plan_launches,
            "planned_launches_within_normal_cap": plan_launches <= common.NORMAL_LAUNCHES,
        },
        "exclusion_reason_counts": _reason_counts(scenes),
        "launch_plan_check": {
            "normal_cap": common.NORMAL_LAUNCHES,
            "hard_cap": common.LAUNCH_CAP,
            "baselines": 16, "disabled_gate_paths": 1, "interventions": total_points,
            "total": plan_launches,
        },
    }


def _reason_counts(scenes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in scenes:
        for exclusion in entry["exclusions"]:
            counts[exclusion["reason"]] = counts.get(exclusion["reason"], 0) + 1
    return counts


def write_points() -> dict[str, Any]:
    payload = select_all()
    common.write_json("INTERVENTION_POINTS.json", payload)
    return payload


def load_points() -> dict[str, Any]:
    path = common.OUTPUT_DIR / "INTERVENTION_POINTS.json"
    if not path.exists():
        raise RuntimeError("INTERVENTION_POINTS.json is missing; the list must be frozen before "
                           "any intervention launch")
    return common.read_json(path)


def point_cells() -> list[tuple[int, str, int]]:
    """(scene, target, step) for every frozen intervention point, in scene order."""
    cells = []
    for entry in load_points()["scenes"]:
        for point in entry["intervention_points"]:
            cells.append((int(entry["scene"]), point["target"], int(point["step"])))
    return cells


if __name__ == "__main__":
    payload = write_points()
    print(json.dumps(payload["totals"], ensure_ascii=False, indent=1))
    print("exclusion reasons:", payload["exclusion_reason_counts"])
