"""Process-safe ROUND41 launch ledger, isolated from every prior round."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

ROUND_DIR = Path(__file__).resolve().parent
NORMAL_CAP = 51
RESERVE_CAP = 5
HARD_CAP = 56


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    if not result:
        raise ValueError(f"empty launch claim key from {value!r}")
    return result


def _write_new(path: Path, payload: dict[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _update(path: Path, **values: Any) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(values)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _reserve_slot(category: str, *, logical_job: str, attempt: int) -> int:
    cap = NORMAL_CAP if category == "normal" else RESERVE_CAP
    slot_dir = ROUND_DIR / "launch_slots" / category
    slot_dir.mkdir(parents=True, exist_ok=True)
    for number in range(1, cap + 1):
        slot = slot_dir / f"{number:04d}.json"
        payload = {
            "category": category,
            "slot": number,
            "logical_job": logical_job,
            "attempt": int(attempt),
            "claimed_at_unix": time.time(),
        }
        if _write_new(slot, payload):
            return number
    raise RuntimeError(f"ROUND41 {category} launch quota exhausted for {logical_job}")


def claim_launch(logical_job: str, *, attempt: int,
                 metadata: dict[str, Any] | None = None) -> Path:
    category = "normal" if int(attempt) == 1 else "reserve"
    path = ROUND_DIR / "launch_claims" / _slug(logical_job) / f"attempt_{int(attempt):04d}.json"
    placeholder = {
        "round": 41,
        "logical_job": logical_job,
        "attempt": int(attempt),
        "category": category,
        "status": "claiming",
        "claimed_at_unix": time.time(),
        "environment_constructor_calls": 0,
        "simulation_started": False,
        "optimizer_step_calls": 0,
        "checkpoint_write_calls": 0,
    }
    if not _write_new(path, placeholder):
        raise RuntimeError(f"ROUND41 attempt already claimed: {logical_job} #{attempt}")
    try:
        slot = _reserve_slot(category, logical_job=logical_job, attempt=attempt)
    except Exception as error:
        _update(path, status="not_started_quota_exhausted", error=repr(error))
        raise
    head = _git("rev-parse", "HEAD")
    _update(path, status="claimed_before_environment", launch_slot=slot,
            metadata=metadata or {}, prelaunch_head=head)
    return path


def mark_environment_constructor(claim: Path) -> None:
    _update(claim, environment_constructor_calls=1,
            environment_constructor_called_at_unix=time.time())


def mark_simulation_started(claim: Path) -> None:
    _update(claim, simulation_started=True, simulation_started_at_unix=time.time())


def finish_launch(claim: Path, *, ok: bool, detail: str) -> None:
    _update(claim, status="complete" if ok else "failed_technical",
            finished_at_unix=time.time(), detail=str(detail)[:1400])


def next_attempt(logical_job: str) -> int:
    directory = ROUND_DIR / "launch_claims" / _slug(logical_job)
    attempts = []
    if directory.exists():
        for path in directory.glob("attempt_*.json"):
            try:
                attempts.append(int(path.stem.split("_")[-1]))
            except ValueError:
                continue
    return max(attempts, default=0) + 1


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROUND_DIR.parents[1]), *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def ledger_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {"normal": [], "reserve": [], "claims": []}
    for category in ("normal", "reserve"):
        path = ROUND_DIR / "launch_slots" / category
        if path.exists():
            result[category] = [
                json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(path.glob("*.json"))
            ]
    claim_dir = ROUND_DIR / "launch_claims"
    if claim_dir.exists():
        result["claims"] = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(claim_dir.glob("*/*.json"))
        ]
    result.update({
        "normal_used": len(result["normal"]),
        "reserve_used": len(result["reserve"]),
        "total_used": len(result["normal"]) + len(result["reserve"]),
        "environment_constructor_calls": sum(
            int(row.get("environment_constructor_calls", 0)) for row in result["claims"]
        ),
        "simulation_started": sum(bool(row.get("simulation_started"))
                                   for row in result["claims"]),
        "optimizer_step_calls": sum(int(row.get("optimizer_step_calls", 0))
                                     for row in result["claims"]),
        "checkpoint_write_calls": sum(int(row.get("checkpoint_write_calls", 0))
                                       for row in result["claims"]),
        "hard_cap": HARD_CAP,
    })
    if result["normal_used"] > NORMAL_CAP or result["reserve_used"] > RESERVE_CAP:
        raise RuntimeError("ROUND41 launch ledger exceeded the fixed 51+5 cap")
    return result


__all__ = [
    "HARD_CAP", "NORMAL_CAP", "RESERVE_CAP", "ROUND_DIR", "claim_launch",
    "finish_launch", "ledger_snapshot", "mark_environment_constructor",
    "mark_simulation_started", "next_attempt",
]
