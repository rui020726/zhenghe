"""Freeze the inputs of the recipient/topology audit.

Run:  python topology_inputs.py    # writes COMM_RECIPIENT_TOPOLOGY_INPUT_MANIFEST.json

Scope is stated rather than total: this round reads the frozen rows, the communication transcripts,
the production source files it cites in the source map, and the delivered round's judgement
artefacts. It does **not** read the allocation decision traces or lifecycle logs, so those are
listed as present-but-unread rather than hashed. Hashing everything would imply an evidentiary
relationship to everything.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import topology_common as T

CALLS = T.install_env_tripwire()
READ_SUBDIR = "runs"  # contains the ct_* transcripts this round opens


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=T.ROOT, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def hashed_inputs() -> dict[str, Any]:
    entries: dict[str, str] = {}
    total = 0

    def add(path: Path, key: str) -> None:
        nonlocal total
        entries[key] = T.sha256_file(path)
        total += path.stat().st_size

    for path in sorted((T.SOURCE_ROUND / "rows").rglob("*.jsonl")):
        add(path, str(path.relative_to(T.SOURCE_ROUND)).replace("\\", "/"))
    for path in sorted((T.SOURCE_ROUND / READ_SUBDIR).rglob("ct_*.jsonl")):
        add(path, str(path.relative_to(T.SOURCE_ROUND)).replace("\\", "/"))
    for name in sorted(
        ["GATES_g6.json", "GATES_g8.json", "GATES_g10.json", "GATES_g12.json",
         "RULE_MATRIX_g6.json", "RULE_MATRIX_g8.json", "RULE_MATRIX_g10.json",
         "RULE_MATRIX_g12.json", "PRESSURE_METRICS_g12.json",
         "ROW_PROVENANCE_g12.json", "G12_START_LEDGER.json"]
    ):
        path = T.SOURCE_ROUND / name
        if path.exists():
            add(path, name)
    for relative in (
        "integrated_sar/comm.py", "integrated_sar/comm_policy.py",
        "integrated_sar/comm_budget.py", "integrated_sar/joint_driver.py",
        "integrated_sar/adapters/task03.py", "integrated_sar/coordinator.py",
        "integrated_sar/assembly.py", "task_allocation/old_task_env.py",
        "task_allocation/old_driver.py",
    ):
        path = T.ROOT / relative
        if path.exists():
            add(path, relative)
    previous = {
        path.name: T.sha256_file(path)
        for path in sorted(p for p in T.PREVIOUS_ROUND.glob("*.json") if not p.name.startswith("_"))
        if not path.name.startswith("_")  # the previous round's scratch hash cache
    }
    for name, digest in previous.items():
        entries[f"previous_round/{name}"] = digest
    return {"file_count": len(entries), "total_bytes": total, "sha256": entries}


def grid() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for layer in T.LAYERS:
        rows = T.rows_of(layer)
        cells = {(str(row["arm"]), int(row["seed"])) for row in rows}
        transcripts = sum(
            1 for arm, seed in cells if T.transcript_path(layer, arm, seed).exists()
        )
        out[layer] = {
            "streams": len({seed for _arm, seed in cells}),
            "arms": len({arm for arm, _seed in cells}),
            "cells": len(cells),
            "transcripts_present": transcripts,
        }
    return out


def unread_but_present() -> dict[str, int]:
    counts: dict[str, int] = {}
    for layer in T.LAYERS:
        directory = T.SOURCE_ROUND / "runs" / layer
        counts[f"{layer}_tr_allocation_decision_traces"] = len(
            list(directory.glob("tr_*.jsonl"))
        )
        counts[f"{layer}_rp_lifecycle_logs"] = len(list(directory.glob("rp_*.jsonl")))
    return counts


def main(argv: list[str]) -> int:
    import pressure_common as common  # noqa: WPS433 - needs the tripwire path set first

    hashed = hashed_inputs()
    manifest = {
        "round": "comm_recipient_topology_feasibility",
        "date": "2026-09-23",
        "status": "source-and-stored-data feasibility audit; not a performance experiment",
        "question": "is subscriber_count a real per-message quantity in this system, or one "
                    "constant multiplied everywhere",
        "authorisation": {
            "new_episodes": 0, "new_streams": 0, "optimizer_steps": 0, "checkpoint_loads": 0,
            "production_code_changed": False, "reward_changed": False, "lambda_chosen": False,
            "budget_changed": False, "ack_or_loss_enabled": False, "topology_randomised": False,
            "payload_padding_used": False,
            "sealed_block_800000_800399_touched": False,
            "recipient_selection_merged_into_send_timing": False,
            "send_once_baseline_weakened": False,
        },
        "git": {
            "branch": git("branch", "--show-current"),
            "head": git("rev-parse", "HEAD"),
            "start_sha": "efa942857f5bce1af01b84c1739c8fc9583dcca7",
            "start_is_pushed_audit_head": True,
            "start_subject": git("log", "-1", "--format=%s", "efa942857f5bce1af01b84c1739c8fc9583dcca7"),
        },
        "budget_bytes": T.BUDGET_BYTES,
        "grid": grid(),
        "inputs_hashed": {k: v for k, v in hashed.items() if k != "sha256"},
        "input_sha256": hashed["sha256"],
        "inputs_present_but_not_read": unread_but_present(),
        "production_files_cited": sorted(
            {
                value["file"]
                for value in T.source_facts().values()
                if value.get("found")
            }
        ),
        "launch_ledger_at_freeze": {
            key: common.ledger_totals().get(key)
            for key in ("starts", "matrix_starts", "gate_starts", "failures")
        },
        "environment_tripwire": {
            "installed": True,
            "wrapped_callables": [f"{m}.{a}" for m, a in T.FORBIDDEN_CALLS],
            "calls_so_far": list(CALLS),
        },
    }
    T.write_json(T.HERE / "COMM_RECIPIENT_TOPOLOGY_INPUT_MANIFEST.json", manifest)
    print(json.dumps(
        {
            "files_hashed": manifest["inputs_hashed"]["file_count"],
            "mebibytes": round(manifest["inputs_hashed"]["total_bytes"] / 2**20, 1),
            "grid": manifest["grid"],
            "not_read": manifest["inputs_present_but_not_read"],
            "ledger": manifest["launch_ledger_at_freeze"],
            "tripwire_calls": manifest["environment_tripwire"]["calls_so_far"],
        },
        ensure_ascii=False,
        indent=1,
    ))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
