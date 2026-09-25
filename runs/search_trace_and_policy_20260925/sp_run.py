"""Run one phase of the round-18 protocol: B1 passivity, historical reproduction, S1.

Every cell books its launch before the driver is touched, writes artefacts named by
phase/arm/seed/attempt, and records the hashes the identity report compares.

Configuration provenance is stated per cell rather than assumed: a seed that has an archived
record is replayed from that record, and a new seed inherits the delivered D1 candidate
configuration (read off ``data_A1_s1100000``'s own run_config) with only the seed changed.
Either way the list of substitutions is written into the row.

Usage:
    D:/project_env/python.exe runs/search_trace_and_policy_20260925/sp_run.py --plan b1
    D:/project_env/python.exe .../sp_run.py --plan data --worker 0 --workers 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import sp_common as sc

from integrated_sar.joint_driver import RunConfiguration, build_joint_run
from integrated_sar.lifecycle_metrics import window_metrics

#: The seed whose archived record carries the delivered candidate configuration.
CONFIGURATION_SOURCE_SEED = 1100000


def archived_record(batch: str, seed: int) -> dict[str, Any] | None:
    path = (sc.d1_archive_path("A1", seed) if batch == "d1"
            else sc.systemloop_archive_path("baseline", seed))
    return sc.archived_configuration(path) if path.exists() else None


def archived_scenario(batch: str, seed: int) -> dict[str, Any]:
    path = (sc.d1_archive_path("A1", seed) if batch == "d1"
            else sc.systemloop_archive_path("baseline", seed))
    return sc.archived_scenario(path) if path.exists() else {}


def base_fields(batch: str, seed: int) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """The behaviour fields this cell should run on, and what had to be substituted."""
    archived = archived_record(batch, seed) or {}
    source_seed = seed
    if not archived:
        source_seed = CONFIGURATION_SOURCE_SEED
        archived = (archived_record(batch, source_seed)
                    or archived_record("d1", source_seed))
    if not archived:
        raise FileNotFoundError(f"no archived configuration available for {batch}/s{seed}")
    fields, substituted = sc.configuration_from_archive(archived)
    if source_seed != seed:
        substituted.append(
            f"seed: no archived record for {seed}; the configuration is inherited from "
            f"s{source_seed} with only the seed changed")
    fields["seed"] = int(seed)
    return fields, substituted, archived


def build_configuration(phase: str, arm: str, seed: int, attempt: int, *, probe: bool,
                        batch: str = "d1",
                        extra: dict[str, Any] | None = None) -> tuple[RunConfiguration, dict[str, Any]]:
    fields, substituted, archived = base_fields(batch, seed)
    if extra:
        # A plumbing check shortens the episode; it is booked and never read as evidence.
        fields.update(extra)
        substituted.extend(f"override: {key}={value}" for key, value in extra.items())
    paths = sc.cell_paths(phase, arm, seed, attempt, probe=probe if phase == "b1" else None)
    fields.update({
        "output": str(paths["lifecycle"]),
        "comm_transcript": str(paths["transcript"]),
        "search_probe": str(paths["search_probe"]) if probe else None,
    })
    fields.update(sc.S1_ARM_OVERRIDES.get(arm, {}))
    if phase == "b1":
        # The decision trace is the artefact that carries the allocation decisions the
        # passivity gate compares. The archived A1 cell wrote none, so this is a registered
        # deviation, installed identically in both conditions, and never read back by a
        # decision -- see integrated_sar/allocation_trace.py.
        fields["decision_trace"] = str(paths["decision_trace"])
    return RunConfiguration(**fields), {
        "paths": paths,
        "substituted_fields": substituted,
        "archived_run_config": archived,
        "archived_scenario": (archived_scenario(batch, seed)
                              or archived_scenario(batch, CONFIGURATION_SOURCE_SEED)),
    }


def _scenario_of(path: Path) -> dict[str, Any]:
    """The scenario the run itself recorded: map hash plus both initial layouts."""
    for record in sc.iter_jsonl(path):
        if record.get("event_type") == "run_config":
            return dict((record.get("detail") or {}).get("scenario") or {})
    return {}


def _probe_facts(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"search_probe_present": False}
    rows = list(sc.iter_jsonl(path))
    steps = [row for row in rows if row.get("kind") == "step"]
    placed = [row for row in rows if row.get("kind") == "placed"]
    final = [row for row in rows if row.get("kind") == "final"]
    disagreements = [row for row in steps
                     if not (row.get("closure") or {}).get("all_views_agree", True)]
    pairs_without_event = [{"step": int(row.get("step") or 0),
                            "targets": sorted(set(row.get("pair_observed_target_ids") or ())
                                              - set(row.get("adapter_event_target_ids") or ()))}
                           for row in steps
                           if set(row.get("pair_observed_target_ids") or ())
                           - set(row.get("adapter_event_target_ids") or ())]
    deleted_ids = {str(target) for row in steps
                   for target in (row.get("env_deleted_target_ids") or ())}
    return {
        "search_probe_present": True,
        "probe_placed": len(placed),
        "probe_placed_ids": sorted(row["target_id"] for row in placed),
        "probe_steps": len(steps),
        "probe_disagreement_steps": len(disagreements),
        "probe_disagreement_examples": disagreements[:3],
        "probe_pairs_without_event": pairs_without_event[:8],
        "probe_deleted_ids": sorted(deleted_ids),
        "probe_never_deleted_ids": sorted({row["target_id"] for row in placed} - deleted_ids),
        "probe_final": final[-1] if final else None,
        "probe_anomalies": int((final[-1] if final else {}).get("anomaly_count") or 0),
    }


def run_cell(phase: str, arm: str, seed: int, attempt: int = 1, *, probe: bool = True,
             batch: str = "d1", extra: dict[str, Any] | None = None) -> dict[str, Any]:
    paths = sc.cell_paths(phase, arm, seed, attempt, probe=probe if phase == "b1" else None)
    if paths["row"].exists():
        return sc.read_json(paths["row"])
    sc.LOGS.mkdir(parents=True, exist_ok=True)
    sc.ROWS.mkdir(parents=True, exist_ok=True)
    booked = sc.ledger_reserve(phase, f"{phase}/{arm}/s{seed}/a{attempt}", detail=batch)
    configuration, context = build_configuration(phase, arm, seed, attempt,
                                                probe=probe, batch=batch, extra=extra)
    started = time.time()
    try:
        run = build_joint_run(configuration)
        recorder = sc.StepRecorder(run)
        summary = run.run()
        run.logger.close()
    except Exception as error:  # a technical failure is still a spent launch
        failure = {"phase": phase, "arm": arm, "seed": seed, "attempt": attempt,
                   "probe": probe, "batch": batch, "launch_index": booked, "ok": False,
                   "error": f"{type(error).__name__}: {error}"}
        paths["row"].write_text(json.dumps(failure, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        raise
    recorder.write(paths["step_record"])
    records = list(sc.iter_jsonl(paths["lifecycle"]))
    metrics = window_metrics(records, configuration.steps).as_row()
    probe_facts = _probe_facts(paths["search_probe"])
    detected_ids = {str(record["target_id"]) for record in records
                    if record.get("event_type") == "detection"}
    placed_ids = set(probe_facts.get("probe_placed_ids") or ()) or {
        f"piloc-e1-target-{index}" for index in range(int(configuration.num_goals))}
    communication = summary.get("communication") or {}
    archived = context["archived_run_config"]
    row = {
        "phase": phase, "arm": arm, "seed": seed, "attempt": attempt, "probe_on": bool(probe),
        "batch": batch, "launch_index": booked, "ok": True,
        "wall_seconds": round(time.time() - started, 2),
        "code_commit": (summary.get("run_signature") or {}).get("code_commit"),
        "search_backend": configuration.search_backend,
        "search_action_policy": configuration.search_action_policy,
        "steps": int(configuration.steps),
        "discovered": metrics["discovered"], "on_time": metrics["on_time"],
        "completed": metrics["completed"], "expired": metrics["expired"],
        "active": metrics["active"], "unassigned": metrics["unassigned"],
        "goals_placed": len(placed_ids),
        "detected_target_ids": sorted(detected_ids),
        "undetected_ids": sorted(placed_ids - detected_ids),
        "undetected_count": len(placed_ids - detected_ids),
        "probe": probe_facts,
        "airtime_bytes": int(communication.get("airtime_bytes") or 0),
        "emissions": int(communication.get("emissions") or 0),
        "packets_lost": int(communication.get("packets_lost") or 0),
        "config_sha": sc.sha256_text(json.dumps(sc.behaviour_fields(
            _configuration_of(paths["lifecycle"])), sort_keys=True, default=str))[:16],
        "archived_config_sha": sc.sha256_text(json.dumps(
            sc.behaviour_fields(archived), sort_keys=True, default=str))[:16],
        "behaviour_sha": sc.sha256_text(json.dumps(
            [sc.behaviour_view(record) for record in records], sort_keys=True,
            default=str))[:16],
        "detection_sha": sc.sha256_text(json.dumps(sorted(
            (str(record["target_id"]), float(record["timestamp"]),
             str(record.get("discovered_by")))
            for record in records if record.get("event_type") == "detection"
        )))[:16],
        "step_record": str(paths["step_record"].relative_to(sc.REPO_ROOT)).replace("\\", "/"),
        "step_record_sha": sc.sha256_text(json.dumps(
            sc.read_json(paths["step_record"]), sort_keys=True, default=str))[:16],
        "step_count": int(sc.read_json(paths["step_record"]).get("steps") or 0),
        "scenario": {key: _scenario_of(paths["lifecycle"]).get(key)
                     for key in ("map_sha256", "goal_positions", "agent_positions")},
        "archived_scenario": {key: context["archived_scenario"].get(key)
                              for key in ("map_sha256", "goal_positions", "agent_positions")},
        "substituted_fields": context["substituted_fields"],
        "artefacts": {key: str(value.relative_to(sc.REPO_ROOT)).replace("\\", "/")
                      for key, value in paths.items()},
        "status_counts": summary.get("status_counts"),
    }
    paths["row"].write_text(json.dumps(row, ensure_ascii=False, indent=1) + "\n",
                            encoding="utf-8")
    return row


def _configuration_of(path: Path) -> dict[str, Any]:
    for record in sc.iter_jsonl(path):
        if record.get("event_type") == "run_config":
            return dict((record.get("detail") or {}).get("configuration") or {})
    raise AssertionError(f"{path.name} has no run_config record")


# ---------------------------------------------------------------------------- plans ----------
def b1_plan() -> list[dict[str, Any]]:
    """B1 on/off x two candidate seeds x two repeats = 8 launches."""
    return [{"phase": "b1", "arm": "A1", "seed": seed, "attempt": attempt, "probe": probe,
             "batch": "d1"}
            for seed in sc.B1_CANDIDATE_SEEDS for probe in (True, False)
            for attempt in (1, 2)]


def repro_plan() -> list[dict[str, Any]]:
    """Ten historical miss seeds plus the four registered no-miss controls = 14 launches."""
    controls = sc.read_json(sc.ROUND_DIR / "CONTROL_SEEDS.json")["selected"]
    cells = [{"phase": "repro", "arm": "A1", "seed": seed, "attempt": 1, "probe": True,
              "batch": "d1"} for seed in sc.D1_MISSED_SEEDS]
    cells += [{"phase": "repro", "arm": "BASE", "seed": seed, "attempt": 1, "probe": True,
               "batch": "systemloop"} for seed in sc.SYSTEMLOOP_MISSED_SEEDS]
    cells += [{"phase": "repro", "arm": "CTRL", "seed": seed, "attempt": 1, "probe": True,
               "batch": "d1"} for seed in controls]
    return cells


def smoke_plan() -> list[dict[str, Any]]:
    """Interface, repeatability and configuration checks for both S1 arms = 8 launches."""
    cells = [{"phase": "smoke", "arm": arm, "seed": seed, "attempt": 1, "probe": True,
              "batch": "d1"} for seed in sc.S1_SEEDS[:2] for arm in sc.S1_ARMS]
    cells += [{"phase": "smoke", "arm": arm, "seed": seed, "attempt": 2, "probe": True,
               "batch": "d1"} for seed in sc.S1_SEEDS[:2] for arm in sc.S1_ARMS]
    return cells


def data_plan() -> list[dict[str, Any]]:
    return [{"phase": "data", "arm": arm, "seed": seed, "attempt": 1, "probe": True,
             "batch": "d1"} for seed in sc.S1_SEEDS for arm in sc.S1_ARMS]


PLANS = {"b1": b1_plan, "repro": repro_plan, "smoke": smoke_plan, "data": data_plan}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", choices=tuple(PLANS), required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args(argv)
    for index, cell in enumerate(PLANS[arguments.plan]()):
        if index % arguments.workers != arguments.worker:
            continue
        row = run_cell(cell["phase"], cell["arm"], cell["seed"], cell["attempt"],
                       probe=cell["probe"], batch=cell["batch"])
        facts = row.get("probe") or {}
        print(f"[{cell['phase']} {cell['arm']} s{cell['seed']} a{cell['attempt']}] "
              f"det={len(row.get('detected_target_ids') or [])} "
              f"undet={row.get('undetected_count')} onT={row.get('on_time')} "
              f"disagree={facts.get('probe_disagreement_steps')} "
              f"B={row.get('airtime_bytes')} {row.get('wall_seconds')}s", flush=True)
    print(json.dumps(sc.ledger_report()["by_phase"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
