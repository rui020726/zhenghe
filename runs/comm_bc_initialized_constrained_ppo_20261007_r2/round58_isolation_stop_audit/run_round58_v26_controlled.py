"""Fail-closed supervisor for the already frozen ROUND58 V26 fixed plan.

It launches the real private CLI, waits for its terminal receipt, and relaunches
only after the exact registered safe-boundary stop plus independent raw ledger
and complete-batch checks.  It never creates an environment or updates a model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import r58_v26_independent_ledger_preflight as ledger

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
R2 = HERE.parent
RAW = R2 / "round58_gen3/raw"
RESUME_RAW = R2 / "round58_gen3_resume_v26/raw"
STATE = HERE / "ROUND58_V26_CONTROLLED_SUPERVISOR_STATE.json"
LOCK = RESUME_RAW / "runtime/ROUND58_V26_CONTROLLED_SUPERVISOR.lock"
STOP = ("RuntimeError: ROUND58 graceful committed-batch/cell boundary reached "
        "before claim; all completed scientific operations are durable")


def save_state(value: dict) -> None:
    temporary = STATE.with_name(STATE.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, STATE)


def complete_batch_boundary() -> dict[str, int]:
    """A graceful training stop must not strand a second prefix."""
    counts = {}
    for seed in (11, 22, 33):
        completed = []
        for path in (RAW / "launch_claims").glob(f"new_training_s{seed}_e*/attempt_*.json"):
            claim = json.loads(path.read_text(encoding="utf-8"))
            if claim.get("status") == "complete":
                completed.append(int(claim["metadata"]["episode_index"]))
        if sorted(completed) != list(range(len(completed))) or len(completed) % 4:
            raise RuntimeError(f"seed {seed}: graceful stop lacks contiguous complete batch episodes")
        counts[str(seed)] = len(completed)
        if len(completed) <= 4:
            continue
        batch = len(completed) // 4 - 1
        commit_path = RESUME_RAW / "checkpoints" / f"r40_batch_commit_s{seed}_b{batch}.json"
        if not commit_path.is_file():
            # The immutable V25 chain remains the source until V26 commits its
            # first new batch for seed11.
            old = R2 / "round58_gen3_resume_v22/raw/checkpoints" / commit_path.name
            if seed != 11 or batch != 14 or not old.is_file():
                raise RuntimeError(f"seed {seed}: latest complete batch lacks commit")
            commit_path = old
        receipt = json.loads(commit_path.read_text(encoding="utf-8"))
        if (int(receipt.get("training_seed", -1)) != seed
                or int(receipt.get("episodes_done", -1)) != len(completed)
                or int(receipt.get("updates", -1)) != len(completed) // 4):
            raise RuntimeError(f"seed {seed}: latest commit does not close complete prefix")
    return counts


def run() -> int:
    if not (HERE / "ROUND58_GEN3_REAL_RESTORE_GATE_V26B.json").is_file():
        raise RuntimeError("V26B real CLI gate evidence is missing")
    gate = json.loads((HERE / "ROUND58_GEN3_REAL_RESTORE_GATE_V26B.json").read_text(encoding="utf-8"))
    if (gate.get("status") != "passed" or gate.get("derived_ledger_spy_status")
            != "not_invoked_in_readonly_restore_gate"):
        raise RuntimeError("V26B gate status/ledger limitation differs")
    if subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                      capture_output=True, text=True, check=True).stdout.strip():
        raise RuntimeError("source worktree is not clean before fixed-plan execution")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"supervisor_pid": os.getpid(), "head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
            text=True, check=True).stdout.strip(), "started_unix": time.time()}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    state = {"kind": "round58-v26-controlled-supervisor/1", "status": "running",
             "pid": os.getpid(), "cycles": [], "head": json.loads(LOCK.read_text(encoding="utf-8"))["head"]}
    save_state(state)
    try:
        for cycle in range(1, 101):
            before = ledger.audit()
            if before["inflight_claims"]:
                raise RuntimeError("cannot launch with an in-flight claim")
            stdout_path = HERE / f"ROUND58_V26_EXECUTION_STDOUT_{cycle:03d}.log"
            stderr_path = HERE / f"ROUND58_V26_EXECUTION_STDERR_{cycle:03d}.log"
            if stdout_path.exists() or stderr_path.exists():
                raise RuntimeError("V26 cycle log already exists; refusing duplicate launch")
            args = [sys.executable, "-u", str(HERE / "r58_private_launcher_repaired.py"),
                    "--authorization", str(HERE / "ROUND58_RESUME_EXECUTION_AUTHORIZATION_V26.json"),
                    "--freeze", str(HERE / "ROUND58_RESUME_FREEZE_V26.json"),
                    "--cumulative-quota", str(HERE / "ROUND58_GEN3_CUMULATIVE_QUOTA.json"),
                    "--scene-audit", str(HERE / "ROUND58_GEN3_RESUME_SCENE_OCCUPANCY_RECHECK.json"),
                    "--output-root", str(RAW), "--mode", "recover_first_batch_gate"]
            with stdout_path.open("x", encoding="utf-8") as out, stderr_path.open("x", encoding="utf-8") as err:
                child = subprocess.Popen(args, cwd=REPO, stdout=out, stderr=err,
                                         stdin=subprocess.DEVNULL)
                state.update({"active_child_pid": child.pid, "active_cycle": cycle,
                              "status": "running"})
                save_state(state)
                exit_code = child.wait()
            after = ledger.audit()
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
            terminal = stderr.rstrip().splitlines()[-1] if stderr.strip() else ""
            row = {"cycle": cycle, "child_pid": child.pid, "exit_code": exit_code,
                   "terminal_stderr": terminal, "before": before, "after": after,
                   "stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}
            state["cycles"].append(row)
            state.pop("active_child_pid", None)
            state.pop("active_cycle", None)
            if exit_code == 0:
                state["status"] = "cli_success_pending_final_independent_validation"
                save_state(state)
                if after["complete_logical_jobs"] != 1024 or after["remaining_plan_keys"] != 0:
                    raise RuntimeError("CLI exited successfully without all 1024 logical tasks")
                state["status"] = "fixed_plan_cli_complete"
                save_state(state)
                return 0
            if (exit_code != 1 or terminal != STOP
                    or stderr.count("Traceback (most recent call last):") != 1):
                state["status"] = "stopped_non_graceful_error"
                save_state(state)
                raise RuntimeError(f"cycle {cycle}: non-graceful CLI stop: {terminal}")
            if after["inflight_claims"] or after["complete_logical_jobs"] <= before["complete_logical_jobs"]:
                state["status"] = "stopped_no_closed_progress"
                save_state(state)
                raise RuntimeError("graceful stop did not produce closed progress")
            row["complete_training_episodes_at_safe_stop"] = complete_batch_boundary()
            state["status"] = "safe_boundary_relaunching"
            save_state(state)
        raise RuntimeError("100 safe cycles exhausted before fixed plan closed")
    except BaseException:
        state["status"] = "stopped_supervisor_or_child_error"
        save_state(state)
        raise
    finally:
        state["last_supervisor_event_unix"] = time.time()
        save_state(state)
        # A still-owned child is never made launchable by deleting its lock.
        if "active_child_pid" not in state:
            LOCK.unlink()


if __name__ == "__main__":
    raise SystemExit(run())
