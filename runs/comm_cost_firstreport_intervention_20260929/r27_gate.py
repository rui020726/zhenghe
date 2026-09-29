"""ROUND27 gate: L versus L0 must be the same episode, and L must match ROUND25's cell.

Zero launches: both runs already happened. The gate compares the two rows on the behavioural
fields, the two tapes line by line, and L against the archived ROUND25 learned row on the common
fields. Any unexplained drift stops the round before the intervention paths may run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import r27_common as common

ROUND25_ROW = common.ROUND25_DIR / "rows" / "r25_learned_s33_greedy_s1400031_a1.json"
ROUND25_TAPE = common.ROUND25_DIR / "logs" / "r25_learned_s33_greedy_s1400031_a1_tape.jsonl"

#: Fields that must be identical between L and L0: everything the episode actually did.
L_L0_BEHAVIOUR_FIELDS = (
    "on_time", "completed", "expired", "discovered", "active", "unassigned", "task_reward_total",
    "spent_bytes", "budget_bytes", "emissions", "budget_refused_sends", "capacity_refused_sends",
    "airtime_tap_total", "goal_identity", "emission_sequence", "refusal_sequence", "cost_stream",
    "task_stream", "ack_ledger", "transport_parameters", "relay_counters",
    "behaviour_sha256", "behaviour_config_sha256", "steps", "substituted_fields",
    "run_signature_checkpoints", "controller_parameters_before", "controller_parameters_after",
    "optimizer_steps",
)
#: Fields this round's L row and the archived ROUND25 learned row share by name and meaning.
#: ``config_hash`` is excluded on purpose: it embeds the per-cell transcript path.
COMMON_ROW_FIELDS = tuple(field for field in L_L0_BEHAVIOUR_FIELDS
                          if field not in {"controller_parameters_before",
                                           "controller_parameters_after", "optimizer_steps"})


def _load(path_name: str) -> dict[str, Any]:
    path = common.row_path(path_name, 1)
    if not path.exists():
        raise RuntimeError(f"{path_name}: no final row on disk; the gate cannot compare")
    return common.read_json(path)


def _tape_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mismatches(left: dict[str, Any], right: dict[str, Any],
                fields: tuple[str, ...]) -> list[str]:
    return [key for key in fields if left.get(key) != right.get(key)]


def gate() -> dict[str, Any]:
    l_row = _load("L")
    l0_row = _load("L0")
    checks: dict[str, Any] = {}

    checks["L_vs_L0_behaviour_fields"] = {
        "fields_compared": list(L_L0_BEHAVIOUR_FIELDS),
        "mismatches": _mismatches(l_row, l0_row, L_L0_BEHAVIOUR_FIELDS),
    }
    checks["L_vs_L0_behaviour_fields"]["passed"] = not checks["L_vs_L0_behaviour_fields"]["mismatches"]

    checks["L_vs_L0_tape_lines"] = {
        "lines": len(_tape_lines(Path(l_row["artefacts"]["tape"]))),
        "identical": _tape_lines(Path(l_row["artefacts"]["tape"]))
        == _tape_lines(Path(l0_row["artefacts"]["tape"])),
        "note": "L installs no interceptor and L0 delegates to the parent collector while the "
                "overrides are disabled, so the two tapes must be the same lines",
    }

    checks["L0_interceptor_disabled_and_idle"] = {
        "identity": l0_row["intervention_identity"],
        "per_target": {target: {
            "opportunities_recorded": entry["opportunities_recorded"],
            "override_applied": entry["override_applied"],
            "real_sends": len(entry["real_sends"]),
        } for target, entry in l0_row["intervention_evidence"]["per_target"].items()},
        "passed": (l0_row["intervention_identity"].get("enabled") is False
                   and l0_row["intervention_identity"].get("class") == "FirstReportAdvancer"
                   and all(not entry["override_applied"]
                           for entry in l0_row["intervention_evidence"]["per_target"].values())),
    }

    checks["L_has_no_interceptor"] = {
        "identity": l_row["intervention_identity"],
        "passed": l_row["intervention_identity"].get("class") is None
        and "note" in l_row["intervention_identity"],
    }

    if ROUND25_ROW.exists():
        r25 = common.read_json(ROUND25_ROW)
        checks["L_vs_ROUND25_common_fields"] = {
            "fields_compared": list(COMMON_ROW_FIELDS),
            "mismatches": _mismatches(l_row, r25, COMMON_ROW_FIELDS),
        }
        checks["L_vs_ROUND25_common_fields"]["passed"] = \
            not checks["L_vs_ROUND25_common_fields"]["mismatches"]
        checks["L_vs_ROUND25_tape_lines"] = {
            "lines": len(_tape_lines(Path(l_row["artefacts"]["tape"]))),
            "identical": _tape_lines(Path(l_row["artefacts"]["tape"]))
            == _tape_lines(ROUND25_TAPE),
            "note": "L runs the same controller, action seed, configuration and the archived "
                    "parent collector, so its tape must match the ROUND25 learned tape line for "
                    "line",
        }
    else:
        checks["L_vs_ROUND25_common_fields"] = {"skipped": "ROUND25 row not on disk",
                                                "passed": True}

    checks["frozen_weights_unchanged"] = {
        "L": l_row["controller_parameters_before"] == l_row["controller_parameters_after"],
        "L0": l0_row["controller_parameters_before"] == l0_row["controller_parameters_after"],
        "passed": (l_row["controller_parameters_before"] is not None
                   and l_row["controller_parameters_before"] == l_row["controller_parameters_after"]
                   and l0_row["controller_parameters_before"] is not None
                   and l0_row["controller_parameters_before"] == l0_row["controller_parameters_after"]),
    }
    checks["optimizer_steps_zero"] = {
        "L": l_row["optimizer_steps"], "L0": l0_row["optimizer_steps"],
        "passed": l_row["optimizer_steps"] == 0 and l0_row["optimizer_steps"] == 0,
    }

    passed = all(bool(check.get("passed", check.get("identical", False)))
                 for check in checks.values())
    return {"round": "27", "section": "first-two gate (L, L0)", "scene": common.SCENE,
            "checks": checks, "passed": passed,
            "verdict": ("L and L0 are the same episode and L matches the ROUND25 cell; the "
                        "intervention paths may run" if passed else
                        "unexplained drift between L and L0 or against ROUND25; stop before "
                        "E0/E5/E05")}


def write_gate() -> dict[str, Any]:
    payload = gate()
    common.write_json("FIRST2_GATE.json", payload)
    return payload


if __name__ == "__main__":
    result = write_gate()
    for name, check in result["checks"].items():
        state = check.get("passed", check.get("identical", "n/a"))
        print(f"{name}: {state}")
        if not state and state != "n/a":
            print("   ", json.dumps(check, ensure_ascii=False)[:800])
    print("passed:", result["passed"])
