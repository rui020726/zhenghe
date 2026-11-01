"""Read-only progress monitor.

It reads the ledger, the row tree, the training telemetry and the convergence certificates and
writes one exclusive snapshot inside execution/monitor. It never claims a slot, never touches a
world log and never repairs a missing number: anything it cannot read is reported as UNKNOWN,
not as zero.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import compact, constants as C


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe(label, function):
    try:
        return function()
    except Exception as error:
        return {"status": "UNKNOWN", "error": "%s: %s" % (type(error).__name__, error),
                "field": label}


def _count_files(directory):
    path = Path(directory)
    if not path.is_dir():
        return 0
    return sum(1 for item in path.glob("*") if item.is_file())


def _ledger(layout):
    ledger = Path(layout.ledger)
    claims = _count_files(ledger / "claims")
    results = _count_files(ledger / "results")
    statuses = {"ok": 0, "failed": 0, "terminated": 0}
    for item in sorted((ledger / "results").glob("*.json")):
        payload = json.loads(item.read_text(encoding="utf-8"))
        status = payload.get("status")
        if status in statuses:
            statuses[status] += 1
    return {"planned": C.WORLD_CAP_TOTAL, "claims": claims, "results": results,
            "by_status": statuses,
            "stop_marker": (ledger / "GLOBAL_STOP.json").is_file(),
            "gate_signal": (ledger / "GATE_FIRST_UPDATE.json").is_file()}


def _row_counts(layout):
    counts = {"train": _count_files(Path(layout.rows) / "train")}
    counts["total"] = counts["train"]
    counts["expected"] = C.TRAIN_WORLDS
    return counts


def _telemetry(layout):
    directory = Path(layout.telemetry)
    payload = {}
    for arm in C.ARMS_TRAIN:
        for seed in C.TRAIN_SEEDS:
            path = directory / ("%s_s%d.json" % (arm, seed))
            if not path.is_file():
                payload["%s_%d" % (arm, seed)] = {"status": "ABSENT"}
                continue
            record = compact.read_json(path)
            payload["%s_%d" % (arm, seed)] = {
                "episodes": len(record.get("episodes") or []),
                "updates": len(record.get("updates") or []),
                "optimizer_steps": sum(int(item.get("optimizer_steps", 0))
                                       for item in (record.get("updates") or [])),
                "terminal_episode": record.get("terminal_episode"),
                "converged": bool((record.get("convergence") or {}).get("converged")),
            }
    return payload


def _checkpoints(layout):
    directory = Path(layout.checkpoints)
    expected = len(C.ARMS_TRAIN) * len(C.TRAIN_SEEDS) * len(C.NEW_CHECKPOINT_EPISODES)
    if not directory.is_dir():
        return {"saved": 0, "expected": expected}
    saved = sum(1 for item in directory.rglob("*.pth") if item.is_file())
    return {"saved": saved, "expected": expected}


def _convergence(layout):
    directory = Path(layout.convergence)
    if not directory.is_dir():
        return {"certificates": 0, "checks": 0, "report_present": False}
    return {"certificates": len(list(directory.glob("CERTIFICATE_*.json"))),
            "checks": len(list(directory.glob("CHECK_*.json"))),
            "report_present": Path(layout.convergence_report).is_file()}


def snapshot(layout):
    """One exclusive snapshot; the query is a read of files only."""
    payload = {
        "schema": "stage5b-progress-snapshot/2",
        "experiment_id": C.RUN_ID,
        "stage": C.STAGE_INDEX,
        "root": str(Path(layout.root).resolve()),
        "recorded_utc": _utc(),
        "ledger": _safe("ledger", lambda: _ledger(layout)),
        "rows": _safe("rows", lambda: _row_counts(layout)),
        "telemetry": _safe("telemetry", lambda: _telemetry(layout)),
        "checkpoints": _safe("checkpoints", lambda: _checkpoints(layout)),
        "convergence": _safe("convergence", lambda: _convergence(layout)),
        "identity_present": Path(layout.identity).is_file(),
        "slot_plan_present": Path(layout.slot_plan).is_file(),
        "base_model_freeze_present": Path(layout.base_model_freeze).is_file(),
        "resume_preflight_present": Path(layout.resume_preflight_receipt).is_file(),
        "stage_verdict_if_no_certificate": C.STAGE_VERDICT_IF_NO_CERTIFICATE,
        "scope": "read-only; the snapshot is a monitoring aid and is not a scientific output",
        "unknown_rule": ("an unreadable input is reported as UNKNOWN; it is never replaced "
                         "by zero"),
    }
    path = Path(layout.monitor) / ("PROGRESS_%s_%d.json"
                                   % (_utc().replace(":", "").replace("-", ""),
                                      time.time_ns()))
    Path(layout.monitor).mkdir(parents=True, exist_ok=True)
    compact.write_json_exclusive(path, payload)
    return {"snapshot": str(path), "ledger": payload["ledger"],
            "rows": payload["rows"], "convergence": payload["convergence"]}
