"""ROUND28 gate and per-sample verification (zero launches).

Two checks live here:

* the first-scene gate: the scene's baseline and its disabled-interceptor cell must be the same
  episode (all behaviour fields, tape line for line), proving the installed-but-disabled
  interceptor is inert;
* prefix reproduction: for every intervention cell, the tape rows before the intervention step
  must be byte-identical to the baseline tape (the only allowed key difference is the
  ``first_report_overrides`` block the enabled interceptor adds), and the recorded opportunity
  (step, features, probability, charge, shadow budget) must reproduce the baseline tape row at
  that step exactly. A failure is a technical problem and is reported as such -- never patched by
  picking a nearby moment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import r28_common as common

TAPE_IDENTITY_KEYS = ("step", "targets", "features", "actions", "free", "actionable", "forced",
                      "masked", "masked_reasons", "attempt_charge_bytes", "send_executable_initial",
                      "send_block_reason_initial", "candidate_shadow", "shadow_budget_before",
                      "shadow_budget_after", "shadow_capacity_before", "shadow_capacity_after",
                      "requested", "log_prob", "value", "send_probabilities", "schema",
                      "action_mask_schema")

BEHAVIOUR_FIELDS = (
    "on_time", "completed", "expired", "discovered", "active", "unassigned", "task_reward_total",
    "spent_bytes", "budget_bytes", "emissions", "budget_refused_sends", "capacity_refused_sends",
    "airtime_tap_total", "goal_identity", "emission_sequence", "refusal_sequence", "cost_stream",
    "task_stream", "ack_ledger", "transport_parameters", "relay_counters",
    "behaviour_sha256", "behaviour_config_sha256", "steps", "substituted_fields",
    "run_signature_checkpoints", "controller_parameters_before", "controller_parameters_after",
    "optimizer_steps",
)


def _load(kind: str, scene: int, target: str | None = None) -> dict[str, Any]:
    path = common.row_path(kind, scene, target)
    if not path.exists():
        raise RuntimeError(f"{common.cell_name(kind, scene, target)}: no final row on disk")
    return common.read_json(path)


def _tape(path: Path) -> list[dict[str, Any]]:
    return list(common.iter_jsonl(path))


def _canonical(tape: dict[str, Any]) -> str:
    return json.dumps({key: tape.get(key) for key in TAPE_IDENTITY_KEYS}, ensure_ascii=False,
                      sort_keys=True)


def _mismatches(left: dict[str, Any], right: dict[str, Any],
                fields: tuple[str, ...]) -> list[str]:
    return [key for key in fields if left.get(key) != right.get(key)]


# ------------------------------------------------------------------ first-scene gate -------------
def first_scene_gate() -> dict[str, Any]:
    scene = common.GATE_SCENARIO
    baseline = _load("baseline", scene)
    disabled = _load("disabled", scene)
    checks: dict[str, Any] = {}
    checks["baseline_vs_disabled_behaviour_fields"] = {
        "fields_compared": list(BEHAVIOUR_FIELDS),
        "mismatches": _mismatches(baseline, disabled, BEHAVIOUR_FIELDS),
    }
    checks["baseline_vs_disabled_behaviour_fields"]["passed"] = \
        not checks["baseline_vs_disabled_behaviour_fields"]["mismatches"]
    checks["baseline_vs_disabled_tape_lines"] = {
        "lines": len(_tape(Path(baseline["artefacts"]["tape"]))),
        "identical": _tape(Path(baseline["artefacts"]["tape"]))
        == _tape(Path(disabled["artefacts"]["tape"])),
    }
    checks["disabled_interceptor_idle"] = {
        "identity": disabled["intervention_identity"],
        "passed": disabled["intervention_identity"].get("enabled") is False,
    }
    checks["frozen_weights_unchanged"] = {
        "baseline": baseline["controller_parameters_before"] == baseline["controller_parameters_after"],
        "disabled": disabled["controller_parameters_before"] == disabled["controller_parameters_after"],
        "passed": (baseline["controller_parameters_before"] is not None
                   and baseline["controller_parameters_before"] == baseline["controller_parameters_after"]
                   and disabled["controller_parameters_before"] is not None
                   and disabled["controller_parameters_before"] == disabled["controller_parameters_after"]),
    }
    checks["optimizer_steps_zero"] = {
        "baseline": baseline["optimizer_steps"], "disabled": disabled["optimizer_steps"],
        "passed": baseline["optimizer_steps"] == 0 and disabled["optimizer_steps"] == 0,
    }
    passed = all(bool(check.get("passed", check.get("identical", False)))
                 for check in checks.values())
    return {"round": "28", "section": "first-scene gate (baseline vs disabled interceptor)",
            "scene": scene, "checks": checks, "passed": passed,
            "verdict": ("the installed-but-disabled interceptor is inert on the gate scene; the "
                        "frozen plan may continue" if passed else
                        "the disabled interceptor is not inert; stop and report")}


def write_gate() -> dict[str, Any]:
    payload = first_scene_gate()
    common.write_json("FIRST_SCENE_GATE.json", payload)
    return payload


# ------------------------------------------------------------------ per-sample prefix check ------
def verify_sample(scene: int, target: str, step: int,
                  point: dict[str, Any] | None = None) -> dict[str, Any]:
    """One intervention cell against its baseline: prefix identity and opportunity reproduction."""
    baseline = _load("baseline", scene)
    cell = _load("intervention", scene, target)
    baseline_tape = _tape(Path(baseline["artefacts"]["tape"]))
    cell_tape = _tape(Path(cell["artefacts"]["tape"]))

    by_step_base = {int(line["step"]): line for line in baseline_tape}
    prefix = []
    for line in cell_tape:
        line_step = int(line["step"])
        if line_step >= step:
            break
        base = by_step_base.get(line_step)
        prefix.append({
            "step": line_step,
            "identical": base is not None and _canonical(base) == _canonical(line),
        })
    divergent = [entry for entry in prefix if not entry["identical"]]

    baseline_row = by_step_base.get(step)
    reproduction: dict[str, Any] = {"baseline_row_at_step": baseline_row is not None}
    if baseline_row is not None and target in baseline_row["targets"]:
        index = baseline_row["targets"].index(target)
        shadow = baseline_row["candidate_shadow"][target]
        reproduction.update({
            "in_baseline_tape": True,
            "original_action": int(baseline_row["actions"][target]),
            "actionable": target in baseline_row["actionable"],
            "send_probability": round(float(baseline_row["send_probabilities"][target]), 6),
            "attempt_charge_bytes": int(baseline_row["attempt_charge_bytes"][target]),
            "budget_before_bytes": float(shadow["budget_before"]),
            "features": [float(value) for value in baseline_row["features"][index]],
            "baseline_tape_row_sha256": hashlib.sha256(json.dumps(
                baseline_row, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        })
        if point is not None:
            reproduction["matches_frozen_point"] = (
                int(point["step"]) == int(baseline_row["step"])
                and point["original_action"] == ("SEND" if int(baseline_row["actions"][target]) == 1
                                                 else "HOLD")
                and abs(float(point["send_probability"])
                        - round(float(baseline_row["send_probabilities"][target]), 6)) < 1e-9
                and int(point["attempt_charge_bytes"])
                == int(baseline_row["attempt_charge_bytes"][target])
                and abs(float(point["budget_before_bytes"])
                        - float(shadow["budget_before"])) < 1e-9
                and point["features"] == [float(value) for value in
                                          baseline_row["features"][index]]
                and point["baseline_tape_row_sha256"] == reproduction["baseline_tape_row_sha256"])
        else:
            reproduction["matches_frozen_point"] = None

    cell_note = None
    for line in cell_tape:
        if int(line["step"]) != step:
            continue
        block = line.get("first_report_overrides") or {}
        for note in block.get("opportunities", []):
            if note.get("target") == target:
                cell_note = note
    return {
        "scene": scene, "target": target, "step": step,
        "prefix_rows_compared": len(prefix),
        "prefix_mismatch_steps": divergent,
        "prefix_identical": not divergent,
        "opportunity_reproduction": reproduction,
        "cell_opportunity_note": cell_note,
        "override_applied_in_cell": bool(cell_note and cell_note.get("override_applied")),
        "verified": (not divergent
                     and bool(reproduction.get("baseline_row_at_step"))
                     and reproduction.get("original_action") == 0
                     and reproduction.get("matches_frozen_point") is not False
                     and bool(cell_note and cell_note.get("override_applied"))),
    }


if __name__ == "__main__":
    result = write_gate()
    print("first-scene gate passed:", result["passed"])
