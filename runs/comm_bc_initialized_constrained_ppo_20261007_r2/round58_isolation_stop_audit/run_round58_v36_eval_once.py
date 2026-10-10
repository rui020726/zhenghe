"""Single-instance, append-only supervisor for the 640 remaining eval jobs."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import r58_v26_independent_ledger_preflight as ledger

LOCK = HERE / "ROUND58_V36_EVAL_SINGLE_INSTANCE.lock"
STATE = HERE / "ROUND58_V36_EVAL_SUPERVISOR_STATE.json"
STDOUT = HERE / "ROUND58_V36_EVAL_STDOUT.log"
STDERR = HERE / "ROUND58_V36_EVAL_STDERR.log"


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".pending")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def run() -> int:
    if any(path.exists() for path in (LOCK, STATE, STDOUT, STDERR)):
        raise RuntimeError("V36 eval supervisor already started; inspect original state")
    gate = HERE / "ROUND58_V36C_ACTUAL_BOOKING_GATE.json"
    frozen = HERE / "ROUND58_RESUME_FREEZE_V36.json"
    preflight = json.loads(gate.read_text(encoding="utf-8"))
    freeze = json.loads(frozen.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed" or preflight.get("event", {}).get("matrix_rows") != 640:
        raise RuntimeError("actual CLI eval handoff gate did not pass")
    launcher = HERE / "r58_private_launcher_repaired.py"
    if hashlib.sha256(launcher.read_bytes()).hexdigest() != freeze["repair_identity"]["private_launcher_sha256"]:
        raise RuntimeError("private execution source differs from committed freeze")
    before = ledger.audit()
    if (before["complete_logical_jobs"] != 384 or before["remaining_plan_keys"] != 640
            or before["inflight_claims"] != 0):
        raise RuntimeError("V36 requires exactly 384 complete training and 640 absent eval keys")
    clean = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout
    if clean.strip():
        raise RuntimeError("source worktree is not clean before V36 eval")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()
    with LOCK.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "head": head, "started_unix": time.time()}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    state = {"kind": "round58-v36-eval-supervisor/1", "status": "running",
             "pid": os.getpid(), "head": head, "before": before,
             "started_unix": time.time()}
    atomic_json(STATE, state)
    args = [sys.executable, "-u", str(launcher),
            "--authorization", str(HERE / "ROUND58_RESUME_EXECUTION_AUTHORIZATION_V36.json"),
            "--freeze", str(frozen),
            "--cumulative-quota", str(HERE / "ROUND58_GEN3_CUMULATIVE_QUOTA.json"),
            "--scene-audit", str(HERE / "ROUND58_GEN3_RESUME_SCENE_OCCUPANCY_RECHECK.json"),
            "--output-root", str(REPO / "runs/comm_bc_initialized_constrained_ppo_20261007_r2/round58_gen3/raw"),
            "--mode", "recover_first_batch_gate"]
    try:
        with STDOUT.open("x", encoding="utf-8") as out, STDERR.open("x", encoding="utf-8") as err:
            child = subprocess.Popen(args, cwd=REPO, stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=err)
            state["child_pid"] = child.pid
            state["stdout_path"] = str(STDOUT)
            state["stderr_path"] = str(STDERR)
            atomic_json(STATE, state)
            code = child.wait()
        state["exit_code"] = code
        state["after"] = ledger.audit()
        state["ended_unix"] = time.time()
        state["terminal_stderr"] = STDERR.read_text(encoding="utf-8", errors="replace").rstrip().splitlines()[-1:] or []
        state["status"] = ("fixed_plan_cli_complete" if code == 0
                           and state["after"]["complete_logical_jobs"] == 1024
                           and state["after"]["remaining_plan_keys"] == 0
                           and state["after"]["inflight_claims"] == 0
                           else "stopped_or_incomplete")
        atomic_json(STATE, state)
        return 0 if state["status"] == "fixed_plan_cli_complete" else 1
    except BaseException as exc:
        state["status"] = "supervisor_error"
        state["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(STATE, state)
        raise
    finally:
        if "child_pid" not in state or "exit_code" in state:
            LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(run())
