"""Resume only the remaining frozen eval keys across exact safe cell boundaries."""
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
import r58_private_launcher_repaired as launcher

LOCK = HERE / "ROUND58_V37_EVAL_SINGLE_INSTANCE.lock"
STATE = HERE / "ROUND58_V37_CONTROLLED_STATE.json"
STOP = ("RuntimeError: ROUND58 graceful committed-batch/cell boundary reached before claim; "
        "all completed scientific operations are durable")


def save(state: dict) -> None:
    pending = STATE.with_name(STATE.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(STATE)


def run() -> int:
    if LOCK.exists() or STATE.exists():
        raise RuntimeError("V37 supervisor already started; inspect original process/state")
    freeze_path = HERE / "ROUND58_RESUME_FREEZE_V37.json"
    gate_path = HERE / "ROUND58_V37B_ACTUAL_EXISTING_CELLS_GATE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (gate.get("status") != "passed"
            or gate.get("existing_cell_reuse", {}).get("count") != 19
            or hashlib.sha256(Path(launcher.__file__).read_bytes()).hexdigest()
               != freeze["repair_identity"]["private_launcher_sha256"]):
        raise RuntimeError("V37 actual CLI existing-cell/source gate is not bound")
    before = ledger.audit()
    if (before["complete_logical_jobs"] != 403
            or before["remaining_plan_keys"] != 621
            or before["inflight_claims"] != 0):
        raise RuntimeError("V37 prelaunch must have exactly 403 complete and 621 missing")
    if subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                      capture_output=True, text=True, check=True).stdout.strip():
        raise RuntimeError("source worktree is not clean")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()
    with LOCK.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "head": head,
                                 "started_unix": time.time()}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    state = {"kind": "round58-v37-controlled-eval/1", "status": "running",
             "pid": os.getpid(), "head": head, "cycles": [], "initial": before}
    save(state)
    args = [sys.executable, "-u", str(HERE / "r58_private_launcher_repaired.py"),
            "--authorization", str(HERE / "ROUND58_RESUME_EXECUTION_AUTHORIZATION_V37.json"),
            "--freeze", str(freeze_path),
            "--cumulative-quota", str(HERE / "ROUND58_GEN3_CUMULATIVE_QUOTA.json"),
            "--scene-audit", str(HERE / "ROUND58_GEN3_RESUME_SCENE_OCCUPANCY_RECHECK.json"),
            "--output-root", str(REPO / "runs/comm_bc_initialized_constrained_ppo_20261007_r2/round58_gen3/raw"),
            "--mode", "recover_first_batch_gate"]
    try:
        for cycle in range(1, 101):
            before = ledger.audit()
            if before["inflight_claims"] or before["complete_logical_jobs"] < 403:
                raise RuntimeError("cannot resume with in-flight or regressed logical tasks")
            out_path = HERE / f"ROUND58_V37_EVAL_STDOUT_{cycle:03d}.log"
            err_path = HERE / f"ROUND58_V37_EVAL_STDERR_{cycle:03d}.log"
            if out_path.exists() or err_path.exists():
                raise RuntimeError("V37 cycle output already exists")
            with out_path.open("x", encoding="utf-8") as out, err_path.open("x", encoding="utf-8") as err:
                child = subprocess.Popen(args, cwd=REPO, stdin=subprocess.DEVNULL,
                                         stdout=out, stderr=err)
                state.update({"active_cycle": cycle, "active_child_pid": child.pid})
                save(state)
                code = child.wait()
            after = ledger.audit()
            stderr = err_path.read_text(encoding="utf-8", errors="replace")
            terminal = stderr.rstrip().splitlines()[-1] if stderr.strip() else ""
            row = {"cycle": cycle, "child_pid": child.pid, "exit_code": code,
                   "terminal_stderr": terminal, "before": before, "after": after,
                   "stdout_path": str(out_path), "stderr_path": str(err_path)}
            state["cycles"].append(row)
            state.pop("active_cycle", None)
            state.pop("active_child_pid", None)
            delta = after["complete_logical_jobs"] - before["complete_logical_jobs"]
            if (after["inflight_claims"] != 0 or delta < 0
                    or after["failed_technical_claims"] != before["failed_technical_claims"]
                    or after["cumulative_attempt_slots"] - before["cumulative_attempt_slots"] != delta
                    or after["cumulative_actual_constructor_calls"] - before["cumulative_actual_constructor_calls"] != delta):
                state["status"] = "stopped_accounting_error"
                save(state)
                raise RuntimeError("V37 cycle changed attempts/failures or left an in-flight claim")
            if code == 0:
                if after["complete_logical_jobs"] != 1024 or after["remaining_plan_keys"] != 0:
                    state["status"] = "stopped_incomplete_success"
                    save(state)
                    raise RuntimeError("V37 CLI success without complete fixed plan")
                state["status"] = "fixed_plan_cli_complete"
                save(state)
                return 0
            if (code != 1 or terminal != STOP or stderr.count("Traceback (most recent call last):") != 1
                    or delta <= 0):
                state["status"] = "stopped_non_graceful_error"
                save(state)
                raise RuntimeError(f"V37 cycle {cycle}: {terminal}")
            state["status"] = "safe_cell_boundary_relaunching"
            save(state)
        raise RuntimeError("V37 exhausted 100 safe cycles before closing 640 eval cells")
    except BaseException:
        state["status"] = "stopped_supervisor_or_child_error"
        save(state)
        raise
    finally:
        state["last_supervisor_event_unix"] = time.time()
        save(state)
        if "active_child_pid" not in state:
            LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(run())
