"""Run D1 cells: four arms on live PILOC search and real CTDE allocation, plus the gates.

Every cell writes three artefacts under ``logs/`` — the lifecycle log, the transport
transcript and the B0 probe file — and one compact row under ``rows/``. Cell names carry
every parallel dimension (tag, arm, seed), because an earlier round lost data to two
dimensions sharing one filename.

Usage:
    D:/project_env/python.exe runs/comm_info_path_validation_20260923/d1_run.py --plan gate
    D:/project_env/python.exe .../d1_run.py --plan data --worker 0 --workers 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import d1_common as dc

from integrated_sar.joint_driver import RunConfiguration, build_joint_run
from integrated_sar.lifecycle_metrics import window_metrics
from integrated_sar.info_probe import behaviour_view


def build_configuration(tag: str, arm: str, seed: int, *, probe: bool,
                        extra: dict[str, Any] | None = None) -> RunConfiguration:
    fields = dc.base_configuration_fields()
    fields.update(dc.ARM_OVERRIDES[arm])
    fields.update(extra or {})
    paths = dc.cell_paths(tag, arm, seed)
    fields.update({"seed": seed, "output": str(paths["lifecycle"]),
                   "comm_transcript": str(paths["transcript"])})
    fields["info_probe"] = str(paths["probe"]) if probe else None
    return RunConfiguration(**fields)


def run_cell(tag: str, arm: str, seed: int, *, probe: bool = True,
             extra: dict[str, Any] | None = None, kind: str = "data") -> dict[str, Any]:
    paths = dc.cell_paths(tag, arm, seed)
    if paths["row"].exists():
        return json.loads(paths["row"].read_text(encoding="utf-8"))
    dc.LOGS.mkdir(parents=True, exist_ok=True)
    dc.ROWS.mkdir(parents=True, exist_ok=True)
    booked = dc.ledger_reserve(kind, f"{tag}/{arm}/s{seed}")
    configuration = build_configuration(tag, arm, seed, probe=probe, extra=extra)
    started = time.time()
    try:
        run = build_joint_run(configuration)
        summary = run.run()
        run.logger.close()
    except Exception as error:  # a technical failure is still a spent launch
        failure = {"tag": tag, "arm": arm, "seed": seed, "probe": probe, "kind": kind,
                   "launch_index": booked, "error": f"{type(error).__name__}: {error}",
                   "ok": False}
        paths["row"].write_text(json.dumps(failure, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        raise
    records = [json.loads(line) for line in
               paths["lifecycle"].read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = window_metrics(records, 1000).as_row()
    communication = summary.get("communication") or {}
    probe_summary: dict[str, Any] = {}
    if probe and paths["probe"].exists():
        for record in dc.iter_jsonl(paths["probe"]):
            if record.get("kind") == "probe_summary":
                probe_summary = record
    row = {
        "tag": tag, "arm": arm, "seed": seed, "probe": probe, "kind": kind,
        "launch_index": booked, "ok": True,
        "wall_seconds": round(time.time() - started, 2),
        "discovered": metrics["discovered"], "on_time": metrics["on_time"],
        "completed": metrics["completed"], "expired": metrics["expired"],
        "active": metrics["active"], "unassigned": metrics["unassigned"],
        "goals_placed": probe_summary.get("placed", configuration.num_goals),
        "targets_never_known": probe_summary.get("targets_never_known"),
        "probe_packets_arrived": probe_summary.get("packets_arrived"),
        "probe_packets_lost": probe_summary.get("packets_lost"),
        "knowledge_rows": probe_summary.get("knowledge_rows"),
        "arrival_rows": probe_summary.get("arrival_rows"),
        "probe_anomalies": probe_summary.get("anomaly_count"),
        "detection_sequence_sha": _sha_of(dc.detection_sequence(paths["lifecycle"])),
        "emission_sequence_sha": _sha_of(dc.emission_sequence(paths["transcript"]))
        if paths["transcript"].exists() else None,
        "packet_outcome_sha": _sha_of(dc.packet_outcomes(paths["probe"]))
        if paths["probe"].exists() else None,
        "airtime_bytes": int(communication.get("airtime_bytes") or 0),
        "emissions": int(communication.get("emissions") or 0),
        "first_emissions": int(communication.get("first_emissions") or 0),
        "refresh_emissions": int(communication.get("refresh_emissions") or 0),
        "transmission_attempts": int(communication.get("transmission_attempts") or 0),
        "packets_delivered": int(communication.get("packets_delivered") or 0),
        "packets_lost": int(communication.get("packets_lost") or 0),
        "delivery_refused": int(communication.get("delivery_refused") or 0),
        "budget_refused_sends": int(communication.get("budget_refused_sends") or 0),
        "capacity_refused_sends": int(communication.get("capacity_refused_sends") or 0),
        "events_accepted": int(communication.get("events_accepted") or 0),
        "transport_parameters": {
            key: communication.get(key) for key in
            ("delivery_probability", "delay_ticks", "timing_mode", "max_wait_steps",
             "bound_enforced_by_transport", "subscribers", "policy", "budget_enforced")
        },
        "config_sha": _sha_of(dc.behaviour_fields(
            dc.configuration_of(paths["lifecycle"]))),
        "behaviour_sha": _sha_of([behaviour_view(record) for record in records]),
        "status_counts": summary.get("status_counts"),
    }
    paths["row"].write_text(json.dumps(row, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
    return row


def _sha_of(rows: Any) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str)
    return __import__("hashlib").sha256(payload.encode("utf-8")).hexdigest()[:16]


def b0_plan() -> list[dict[str, Any]]:
    """Stage one: B0's paired replay and its determinism repeats, plus the timing-mode probe.

    Nothing about the four measurement arms runs until this stage has been judged, because
    a probe that changed behaviour would silently invalidate every number that follows.
    """
    cells: list[dict[str, Any]] = []
    for seed in (1100000, 1100001):
        for arm in ("A1", "A3"):
            cells.append({"tag": "b0off", "arm": arm, "seed": seed, "probe": False,
                          "kind": "b0"})
            cells.append({"tag": "b0on", "arm": arm, "seed": seed, "probe": True,
                          "kind": "b0"})
            cells.append({"tag": "b0offrep", "arm": arm, "seed": seed, "probe": False,
                          "kind": "b0"})
            cells.append({"tag": "b0onrep", "arm": arm, "seed": seed, "probe": True,
                          "kind": "b0"})
    for seed in (1100000, 1100001):
        cells.append({"tag": "modelegacy", "arm": "A1", "seed": seed, "probe": True,
                      "kind": "mode", "extra": {"comm_timing_mode": "legacy_shared_bound"}})
        cells.append({"tag": "modeowned", "arm": "A1", "seed": seed, "probe": True,
                      "kind": "mode", "extra": {"comm_timing_mode": "policy_owned_timing"}})
    return cells


def smoke_plan() -> list[dict[str, Any]]:
    """Stage two: one cell per arm on one seed, which the identity gates are judged on."""
    return [{"tag": "smoke", "arm": arm, "seed": 1100000, "probe": True, "kind": "smoke"}
            for arm in dc.DATA_ARMS]


def gate_plan() -> list[dict[str, Any]]:
    return b0_plan() + smoke_plan()


def data_plan() -> list[dict[str, Any]]:
    return [{"tag": "data", "arm": arm, "seed": seed, "probe": True, "kind": "data"}
            for seed in dc.SEEDS for arm in dc.DATA_ARMS]


PLANS = {"b0": b0_plan, "smoke": smoke_plan, "gate": gate_plan, "data": data_plan}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", choices=tuple(PLANS), required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args(argv)
    plan = PLANS[arguments.plan]()
    for index, cell in enumerate(plan):
        if index % arguments.workers != arguments.worker:
            continue
        row = run_cell(cell["tag"], cell["arm"], cell["seed"], probe=cell["probe"],
                       extra=cell.get("extra"), kind=cell["kind"])
        print(f"[{cell['kind']} {cell['arm']} s{cell['seed']}] disc={row['discovered']} "
              f"onT={row['on_time']} exp={row['expired']} neverKnown={row['targets_never_known']} "
              f"lost={row['packets_lost']} B={row['airtime_bytes']} {row['wall_seconds']}s",
              flush=True)
    print(json.dumps(dc.ledger_report(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
