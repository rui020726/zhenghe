"""Freeze what this round reads, before any of it is aggregated.

Two things are recorded for every input: its SHA256, and whether this round actually
opened it. The second matters because the unread families are the size of the
repository's history: an audit that says "no run logs arrival per goal" is only
credible if it also says which archives exist and were deliberately not scanned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import closure_common as cc

REPO = cc.REPO_ROOT

#: Read-only inputs: none of these may change during the round. The three governing
#: documents are deliberately absent — the round is instructed to update them, and their
#: pre-round state is taken from git instead (``closure_common.governing_state``).
EXPLICIT = [
    "AGENTS.md",
    "runs/comm_action_geometry_audit_20260923/COMM_ACTION_GEOMETRY_AUDIT.json",
    "runs/comm_action_geometry_audit_20260923/COMM_ACTION_GEOMETRY_AUDIT.md",
    "runs/comm_action_geometry_audit_20260923/COMM_ACTION_GEOMETRY_AUDIT_VERDICT.json",
    "runs/comm_action_geometry_audit_20260923/RECOMPUTE_L0_L4.json",
    "runs/comm_recipient_topology_feasibility_20260923/COMM_RECIPIENT_TOPOLOGY_RESULTS.json",
    "runs/comm_recipient_topology_feasibility_20260923/COMM_RECIPIENT_TOPOLOGY_RESULTS.md",
    "runs/comm_recipient_topology_feasibility_20260923/COMM_RECIPIENT_TOPOLOGY_DECISION.md",
    "runs/comm_recipient_topology_feasibility_20260923/COMM_RECIPIENT_TOPOLOGY_SOURCE_MAP.md",
    "runs/comm_task_pressure_learnability_20260922/GATES_g6.json",
    "runs/comm_task_pressure_learnability_20260922/GATES_g8.json",
    "runs/comm_task_pressure_learnability_20260922/GATES_g10.json",
    "runs/comm_task_pressure_learnability_20260922/GATES_g12.json",
    "runs/comm_task_pressure_learnability_20260922/COMM_TASK_PRESSURE_G12_EXTENSION_RESULTS.md",
    "runs/comm_rule_confirm_20260921/COMM_RULE_CONFIRMATION_RESULTS.json",
    "runs/comm_rule_confirm_20260921/COMMUNICATION_LEARNING_VERDICT.md",
    "runs/comm_control_20260919/INFORMATION_BOUNDARY.md",
    "runs/comm_control_20260919/TRAFFIC_PROBE.json",
    "runs/comm_protocol_v2_20260920/COMM_PROTOCOL_V2_RESULTS.md",
    "runs/comm_budget_masked_retrain_20260922/COMM_BUDGET_MASKED_RETRAIN_RESULTS.json",
    "runs/comm_budget_masked_retrain_20260922/COMM_BUDGET_MASKED_RETRAIN_RESULTS.md",
    "runs/comm_evidence_review_20260921/COMM_RULE_CONFIRMATION_RESULTS_REVISED.md",
    "runs/comm_stability_20260919/COMM_POLICY_ATTRIBUTION.md",
    "runs/joint_eval_20260916/expired_attribution.json",
    "runs/learnability_20260917/LEARNABILITY_VERDICT.md",
    "runs/duration_relevance_20260917/DURATION_RELEVANCE.md",
    "runs/coalition_formation_latency_20260918/COALITION_FORMATION_VERDICT.md",
    "runs/coalition_validation_20260918/SYSTEM_LOOP.json",
    "runs/coalition_validation_20260918/SYSTEM_CLOSED_LOOP_VERDICT.md",
    "runs/corrected_env_final_20260918/CORRECTED_ENV_SYSTEM_VERDICT.md",
    "runs/coordinator_baselines_20260917/RESERVATION_VERDICT.md",
    "runs/dispatch_trigger_audit_20260918/DISPATCH_TRIGGER_AUDIT.md",
    "integrated_sar/lifecycle_log.py",
    "integrated_sar/lifecycle_metrics.py",
    "integrated_sar/comm.py",
    "integrated_sar/comm_policy.py",
    "integrated_sar/joint_driver.py",
    "integrated_sar/allocation_trace.py",
    "integrated_sar/task_registry.py",
    "integrated_sar/adapters/task03.py",
    "task_allocation/old_task_env.py",
]

#: Families of per-episode logs. ``scanned`` says whether this round opened them.
FAMILIES = [
    {"name": "deployed live-search lifecycle",
     "dir": "runs/coalition_validation_20260918/systemloop", "pattern": "*_rp_s*.jsonl",
     "scanned": True,
     "note": "60 seeds x 2 arms; the only archived run with live PILOC search on the deployed "
             "information path"},
    {"name": "deployed replayed-search lifecycle",
     "dir": "runs/joint_eval_20260916", "pattern": "rp_*.jsonl",
     "scanned": True, "cells_aggregated": 90,
     "note": "3 arms x 30 seeds have lifecycle logs; the round's two further arms are logged "
             "only as ab_*.jsonl (allocation-behaviour traces) and are not read here"},
    {"name": "isolated budgeted comm lifecycle",
     "dir": "runs/comm_task_pressure_learnability_20260922/runs",
     "pattern": "rp_*_g*_s*.jsonl", "scanned": True, "note": "576 cells (4 layers x 9 arms x 16)"},
    {"name": "isolated budgeted comm transcript",
     "dir": "runs/comm_task_pressure_learnability_20260922/runs",
     "pattern": "ct_*_g*_s*.jsonl", "scanned": True, "note": "same 576 cells"},
    {"name": "lossy confirmation transcript",
     "dir": "runs/comm_rule_confirm_20260921/evaluation",
     "pattern": "ct_*_s*.jsonl", "scanned": "spot-check (4 seeds x 5 arms x 3 settings)",
     "note": "3 000 cells exist; only 60 are opened here, the rest of the quantities come from "
             "the round's published aggregate"},
    {"name": "lossy confirmation lifecycle",
     "dir": "runs/comm_rule_confirm_20260921/evaluation",
     "pattern": "rp_*_s*.jsonl", "scanned": False,
     "note": "1 650 MB of per-episode logs deliberately not re-derived; the published aggregate "
             "already carries the same outcome counts and detection-to-stage latencies"},
    {"name": "decision traces",
     "dir": "runs", "pattern": "tr_*.jsonl", "scanned": False,
     "note": "where true arrival times (on_site / group_arrived_at) live; ~4.9 GB across rounds, "
             "not opened this round, so arrival is treated as a proxy everywhere below"},
    {"name": "recorded detection streams",
     "dir": "runs", "pattern": "rec_s*.jsonl", "scanned": False,
     "note": "the search-side input of the replayed runs; establishes that those runs inherit, "
             "rather than measure, the discovery stage"},
]


def hash_explicit() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rel in EXPLICIT:
        path = REPO / rel
        if not path.exists():
            out[rel] = {"missing": True}
            continue
        out[rel] = {"bytes": path.stat().st_size, "sha256": cc.sha256_file(path)}
    return out


def scan_families() -> list[dict[str, Any]]:
    out = []
    for family in FAMILIES:
        base = REPO / family["dir"]
        paths = cc.family_paths(base, family["pattern"])
        entry = dict(family)
        entry.update(cc.describe_files(paths))
        entry["directory_exists"] = base.exists()
        out.append(entry)
    return out


def launch_ledger_snapshot() -> dict[str, Any]:
    """Re-read the pressure round's ledgers so 'this round started nothing' is checkable.

    ``starts`` is the number of episode launches and ``entries`` the number of
    registrations; a previous round documented that the two differ by re-registration
    rows, so both are reported and neither is called the launch count alone.
    """
    ledger_dir = REPO / "runs/comm_task_pressure_learnability_20260922"
    starts = 0
    entries = 0
    failures = 0
    by_kind: dict[str, int] = {}
    files = 0
    for path in sorted(ledger_dir.glob("LAUNCH_LEDGER*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        files += 1
        starts += int(document.get("starts") or 0)
        entries += len(document.get("entries") or [])
        failures += int(document.get("failures") or 0)
        for kind, count in (document.get("by_kind") or {}).items():
            by_kind[kind] = by_kind.get(kind, 0) + int(count)
    return {"ledger_files": files, "starts": starts, "entries": entries,
            "recorded_failures": failures, "by_kind": by_kind}


def sealed_block_on_disk() -> int:
    """Count artefacts naming a seed in the sealed 800000..800399 block.

    One pass over the file names: globbing per seed would walk the 35 000-file tree
    four hundred times for the same answer.
    """
    sealed = {str(seed) for seed in range(800000, 800400)}
    hits = 0
    for path in (REPO / "runs").rglob("*"):
        if not path.is_file():
            continue
        tokens = path.name.replace(".jsonl", "").replace(".json", "").split("_")
        if any(token in sealed for token in tokens):
            hits += 1
    return hits


def run() -> int:
    cc.install_env_tripwire()
    payload = {
        "round": cc.ROUND,
        "start_sha": cc.START_SHA,
        "explicit_inputs": hash_explicit(),
        "governing_docs_to_update_this_round": cc.governing_state(),
        "log_families": scan_families(),
        "sealed_block_800000_800399_artefacts_on_disk": sealed_block_on_disk(),
        "pressure_launch_ledger": launch_ledger_snapshot(),
        "episodes_generated_by_this_round": 0,
        "env_tripwire": cc.tripwire_report(),
    }
    cc.write_json("CLOSURE_INPUT_MANIFEST.json", payload)
    print(json.dumps({key: payload[key] for key in
                      ("sealed_block_800000_800399_artefacts_on_disk",
                       "pressure_launch_ledger", "env_tripwire")},
                     ensure_ascii=False, indent=1)[:900])
    for family in payload["log_families"]:
        print(f"- {family['name']}: files={family['files']} bytes={family['bytes']} "
              f"scanned={family['scanned']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
