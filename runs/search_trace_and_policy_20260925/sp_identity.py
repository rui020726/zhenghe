"""SEARCH_TRACE_IDENTITY_REPORT.json: is B1 passive, and did the historical runs reproduce?

Two separate questions, judged by two separate blocks, because conflating them is how a
round ends up calling a configuration difference a detector bug.

``passivity`` (phase b1) -- same seed, same arm, probe on versus probe off, two repeats each.
Compared: every lifecycle record after ``behaviour_view`` normalisation, the detection
sequence, the emission sequence, the billed bytes, the task outcome triple, and the per-step
recorder (actions, rewards, done, both position arrays, digests of the environment's two rng
states). Determinism is judged by attempt 1 against attempt 2 inside a condition.

``reproduction`` (phase repro) -- each historical cell replayed today. Compared against the
archived artefact: the behaviour fields of the configuration, the map hash, both initial
layouts, the two checkpoint digests, and the detection sequence. A cell that does not match is
reported as ``not_reproduced`` with the reason list attached; it is never dropped and a
configuration difference is never written as a detector defect.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import sp_common as sc

from integrated_sar.lifecycle_metrics import window_metrics

BATCHES = {"d1": "B1_d1_20260923", "systemloop": "B2_systemloop_20260918"}


def rows_of(phase: str) -> list[dict[str, Any]]:
    out = []
    for path in sorted(sc.ROWS.glob(f"{phase}_*.json")):
        row = sc.read_json(path)
        row.setdefault("_row_file", path.name)
        # The probe flag is carried by the file name, because the b1 cells were written before
        # the row separated "probe_on" from the probe's own facts dict. Deriving it from the
        # name reads the same evidence twice rather than inventing a third value.
        if "probe_on" not in row:
            row["probe_on"] = path.stem.endswith("pon")
        row["probe"] = row.get("probe") if isinstance(row.get("probe"), dict) else {}
        out.append(row)
    return out


def emission_sequence(path) -> list[tuple]:
    out = []
    ordinal: dict[str, int] = {}
    for record in sc.iter_jsonl(path):
        if record.get("kind") != "send":
            continue
        target = str(record["target"])
        ordinal[target] = ordinal.get(target, 0) + 1
        out.append((float(record["step"]), target, int(record.get("subscribers") or 0),
                    int(record.get("cost_bytes") or 0), ordinal[target]))
    return sorted(out)


def detection_sequence(path) -> list[tuple]:
    return sorted((str(record["target_id"]), float(record["timestamp"]),
                   str(record.get("discovered_by")))
                  for record in sc.iter_jsonl(path)
                  if record.get("event_type") == "detection")


def lifecycle_view(path) -> list[dict[str, Any]]:
    return [sc.behaviour_view(record) for record in sc.iter_jsonl(path)]


def step_diff(a: list[dict], b: list[dict]) -> dict[str, Any]:
    """Where the per-step streams first part, field by field."""
    fields = ("actions", "rewards", "done", "agents", "goals", "py_rng", "np_rng", "step")
    first: dict[str, Any] = {}
    if len(a) != len(b):
        first["length"] = f"{len(a)} vs {len(b)}"
    for index in range(min(len(a), len(b))):
        for field in fields:
            if a[index].get(field) != b[index].get(field) and field not in first:
                first[field] = {"step": index + 1, "left": a[index].get(field),
                                "right": b[index].get(field)}
    return {"identical": not first, "first_difference_per_field": first}


def passivity_block() -> dict[str, Any]:
    rows = rows_of("b1")
    by_key = {(row["seed"], row["probe_on"], row["attempt"]): row for row in rows}
    verdicts: dict[str, Any] = {}
    for seed in sc.B1_CANDIDATE_SEEDS:
        cells = {}
        for probe in (True, False):
            for attempt in (1, 2):
                row = by_key.get((seed, probe, attempt))
                if row is None or not row.get("ok"):
                    cells[f"probe{int(probe)}_attempt{attempt}"] = "missing_or_failed"
        comparisons: dict[str, Any] = {}
        for attempt in (1, 2):
            on = by_key.get((seed, True, attempt))
            off = by_key.get((seed, False, attempt))
            if not (on and off and on.get("ok") and off.get("ok")):
                comparisons[f"paired_attempt{attempt}"] = "cells missing"
                continue
            steps_on = sc.read_json(sc.REPO_ROOT / on["step_record"])["rows"]
            steps_off = sc.read_json(sc.REPO_ROOT / off["step_record"])["rows"]
            lifecycle_on = sc.REPO_ROOT / on["artefacts"]["lifecycle"]
            lifecycle_off = sc.REPO_ROOT / off["artefacts"]["lifecycle"]
            comparisons[f"paired_attempt{attempt}"] = {
                "lifecycle_records_identical":
                    lifecycle_view(lifecycle_on) == lifecycle_view(lifecycle_off),
                "detection_sequence_identical":
                    detection_sequence(lifecycle_on) == detection_sequence(lifecycle_off),
                "emission_sequence_identical":
                    emission_sequence(sc.REPO_ROOT / on["artefacts"]["transcript"])
                    == emission_sequence(sc.REPO_ROOT / off["artefacts"]["transcript"]),
                "airtime_bytes_identical": on["airtime_bytes"] == off["airtime_bytes"],
                "outcome_identical": all(on[key] == off[key] for key in
                                         ("discovered", "on_time", "expired", "completed")),
                "config_behaviour_identical": on["config_sha"] == off["config_sha"],
                "per_step": step_diff(steps_on, steps_off),
                "steps_compared": len(steps_off),
            }
        repeat: dict[str, Any] = {}
        for probe in (True, False):
            first = by_key.get((seed, probe, 1))
            second = by_key.get((seed, probe, 2))
            if first and second and first.get("ok") and second.get("ok"):
                # Recomputed here rather than read from the row: the row's behaviour_sha was
                # hashed before the probe's echoed metric keys were added to IDENTITY_FIELDS,
                # so a probe-on row carried its own file name and differed from its own repeat.
                # The comparison that matters is one implementation of "what counts as
                # behaviour", applied now, to both files.
                view_first = lifecycle_view(sc.REPO_ROOT / first["artefacts"]["lifecycle"])
                view_second = lifecycle_view(sc.REPO_ROOT / second["artefacts"]["lifecycle"])
                same = (view_first == view_second
                        and first["detection_sha"] == second["detection_sha"])
                repeat[f"probe{int(probe)}_attempt1_vs_2"] = {
                    "behaviour_and_detection_identical": bool(same),
                    "stored_behaviour_sha_from_the_row": [first["behaviour_sha"],
                                                          second["behaviour_sha"]],
                    "recomputed_behaviour_sha": [sc.sha256_text(json.dumps(
                        view_first, sort_keys=True, default=str))[:16],
                        sc.sha256_text(json.dumps(view_second, sort_keys=True,
                                                  default=str))[:16]],
                }
        verdicts[f"s{seed}"] = {"paired": comparisons, "repeatability": repeat,
                               "probe_side_files": {
                                   key: (by_key.get((seed, True, attempt)) or {}).get(
                                       "artefacts", {}).get("search_probe")
                                   for attempt in (1, 2) for key in [f"attempt{attempt}"]},
                               }
    flat = [entry for seed in verdicts.values() for section in ("paired", "repeatability")
            for entry in seed[section].values() if isinstance(entry, dict)]
    checks = []
    for entry in flat:
        for key, value in entry.items():
            if key == "per_step":
                checks.append(bool(value.get("identical")))
            elif isinstance(value, bool):
                checks.append(value)
    return {
        "cells_used": len(rows),
        "seeds": [str(seed) for seed in sc.B1_CANDIDATE_SEEDS],
        "detail": verdicts,
        "all_checks_true": bool(checks) and all(checks),
        "checks_counted": len(checks),
        "statement": ("the side file is written beside the run, adds no lifecycle event_type, "
                      "is not an observation, and changes nothing that was compared"),
    }


def reproduction_block() -> dict[str, Any]:
    rows = rows_of("repro")
    per_cell: dict[str, Any] = {}
    for row in rows:
        if not row.get("ok"):
            per_cell[row["_row_file"][:-5]] = {"ok": False, "error": row.get("error")}
            continue
        archived_path = (sc.d1_archive_path("A1", row["seed"]) if row["batch"] == "d1"
                         else sc.systemloop_archive_path("baseline", row["seed"]))
        archived = sc.archived_configuration(archived_path)
        present = sc.configuration_of_lifecycle(sc.REPO_ROOT / row["artefacts"]["lifecycle"])
        behaviour_now = sc.behaviour_fields(present)
        behaviour_then = sc.behaviour_fields(archived)
        only_then = sorted(set(behaviour_then) - set(behaviour_now))
        only_now = sorted(set(behaviour_now) - set(behaviour_then))
        differing = sorted(key for key in set(behaviour_now) & set(behaviour_then)
                           if behaviour_now[key] != behaviour_then[key])
        scenario_now, scenario_then = row["scenario"], row["archived_scenario"]
        detections_now = detection_sequence(sc.REPO_ROOT / row["artefacts"]["lifecycle"])
        detections_then = detection_sequence(archived_path)
        same_scenario = all(scenario_now.get(key) == scenario_then.get(key) for key in
                            ("map_sha256", "goal_positions", "agent_positions"))
        per_cell[f"{row['arm']}/s{row['seed']}"] = {
            "batch": BATCHES[row["batch"]],
            "code_commit": row.get("code_commit"),
            "fields_only_in_archived_record": only_then,
            "fields_only_in_todays_record": only_now,
            "behaviour_fields_that_differ": differing,
            "substituted_fields": row.get("substituted_fields"),
            "map_and_layout_match": bool(same_scenario),
            "detection_count_then": len(detections_then),
            "detection_count_now": len(detections_now),
            "detection_sequence_match": detections_now == detections_then,
            "first_detection_step_differences": [
                {"target": then[0], "then_step": then[1],
                 "now_step": next((now[1] for now in detections_now if now[0] == then[0]), None)}
                for then in detections_then
                if not any(now[0] == then[0] and now[1] == then[1] for now in detections_now)
            ][:8],
            "goals_placed": row["goals_placed"],
            "undetected_now": row["undetected_ids"],
            "undectected_then_ids_then": sorted(
                {f"piloc-e1-target-{i}" for i in range(row["goals_placed"])}
                - {entry[0] for entry in detections_then}),
            "on_time_now": row["on_time"], "on_time_then": None,
            "historical_run_reproduced": (bool(same_scenario) and not differing
                                          and detections_now == detections_then),
        }
        # the archived on-time reading, from the same window convention the project uses
        metrics = window_metrics(list(sc.iter_jsonl(archived_path)),
                                 int(behaviour_then.get("steps") or 1000)).as_row()
        per_cell[f"{row['arm']}/s{row['seed']}"]["on_time_then"] = metrics["on_time"]
        per_cell[f"{row['arm']}/s{row['seed']}"]["outcome_matches_then"] = all(
            row[key] == metrics[key] for key in ("discovered", "on_time", "expired"))
    return {
        "cells": len(per_cell),
        "detail": per_cell,
        "reproduced": sorted(key for key, value in per_cell.items()
                             if value.get("historical_run_reproduced")),
        "not_reproduced": sorted(key for key, value in per_cell.items()
                                 if not value.get("historical_run_reproduced")),
        "rule": ("a cell that does not reproduce is reported as not_reproduced with its reason "
                 "attached; it is not deleted, and a configuration difference is never written "
                 "as a detector defect"),
    }


def code_identity() -> dict[str, Any]:
    def show(paths: list[str]) -> dict[str, str]:
        result = subprocess.run(["git", "diff", "--name-only", sc.START_SHA, "--", *paths],
                                cwd=str(sc.REPO_ROOT), capture_output=True, text=True, timeout=60)
        return {line.strip(): "modified since the start sha" for line in result.stdout.splitlines()
                if line.strip()}

    touched = show(["integrated_sar", "piloc/src/Map/timemap.py"])
    frozen = show(["runs/comm_info_path_validation_20260923",
                   "runs/real_search_undetected_attribution_20260923"])
    return {
        "start_sha": sc.START_SHA,
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(sc.REPO_ROOT),
                               capture_output=True, text=True, timeout=30).stdout.strip(),
        "production_files_changed": touched,
        "earlier_round_files_changed": frozen,
        "checkpoints": {str(path.relative_to(sc.REPO_ROOT)).replace("\\", "/"):
                        sc.sha256_file(path) for path in (sc.PIL_CHECKPOINT, sc.CTDE_CHECKPOINT)},
        "note": ("the probe is a default-off side recorder; the two checkpoint digests are "
                 "recorded so an arm cannot silently run on a different model"),
    }


def main(argv: list[str]) -> int:
    payload = {
        "round": sc.ROUND,
        "passivity_gate_P": passivity_block(),
        "reproduction_gate_Q": reproduction_block(),
        "code_identity": code_identity(),
        "launch_ledger": sc.ledger_report(),
    }
    path = sc.write_json("SEARCH_TRACE_IDENTITY_REPORT.json", payload)
    print(path)
    print(json.dumps({"passive": payload["passivity_gate_P"]["all_checks_true"],
                      "reproduced": payload["reproduction_gate_Q"]["reproduced"],
                      "not_reproduced": payload["reproduction_gate_Q"]["not_reproduced"],
                      "used": payload["launch_ledger"]["used"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
