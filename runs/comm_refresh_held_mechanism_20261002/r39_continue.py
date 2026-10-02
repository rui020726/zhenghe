"""Continue the frozen ROUND39 matrix after the verified nine-cell gate.

This script does not alter the frozen runner or treatment. It binds the existing
v2 launch ledger to ROUND39, verifies the already-completed gate and claims, then
calls the frozen runner's one-cell entry point serially for the remaining cells.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PRELAUNCH_COMMIT = "e8f3e76f5555e1da271e6d8d560c32fadf151ba2"
EXPECTED_BRANCH = "exp/comm-refresh-held-mechanism"
sys.path[:0] = [
    str(ROOT), str(ROOT / "runs" / "comm_final_evaluation_20261002"),
    str(ROOT / "runs" / "comm_cost_v2_rerun_20260928"),
    str(ROOT / "runs" / "comm_cost_stage_b_20260927"),
    str(ROOT / "runs" / "comm_budget_action_mask_fix_20260922"),
    str(ROOT / "runs" / "comm_tight_budget_20260919"), str(HERE),
]

import r39_run  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout.strip()


def bind_round39_output() -> None:
    common = r39_run.v2_common
    common.OUTPUT_DIR = HERE
    common.LOGS = HERE / "logs"
    common.ROWS = HERE / "rows"
    common.REJECTED = HERE / "rejected"
    common.CHECKPOINTS = HERE / "checkpoints"
    common.LAUNCH_CAP = 153


def require_gate_and_resume_state() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("not on the dedicated ROUND39 branch")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked files changed after the gate validation commit")
    if git("rev-parse", f"{PRELAUNCH_COMMIT}^1") != r39_run.BASE_HEAD:
        raise RuntimeError("prelaunch commit ancestry no longer matches the frozen base")
    frozen_runner_blob = git(
        "rev-parse", f"{PRELAUNCH_COMMIT}:runs/comm_refresh_held_mechanism_20261002/r39_run.py")
    current_runner_blob = git("hash-object", str(HERE / "r39_run.py"))
    if current_runner_blob != frozen_runner_blob:
        raise RuntimeError("frozen r39_run.py differs from the prelaunch commit")

    gate = read_json(HERE / "FIRST_SCENE_GATE.json")
    if gate.get("schema") != "round39-first-scene-gate/1" or not gate.get("all_passed"):
        raise RuntimeError("corrected nine-cell gate has not passed")
    for seed in ("11", "33"):
        item = gate.get("high_refresh_seed_gates", {}).get(seed) or {}
        if (item.get("held_refresh_emissions") != 0
                or int(item.get("bytes_reduced", 0)) <= 0
                or not (item.get("effective_budget_order") or {}).get("nonnegative_and_sequential")):
            raise RuntimeError(f"high-refresh seed {seed} gate evidence is incomplete")
    if not gate.get("seed22_negative_control", {}).get("passed"):
        raise RuntimeError("seed22 negative control did not pass")
    if gate.get("optimizer_steps") != 0 or gate.get("production_modules_modified") != []:
        raise RuntimeError("gate report violates the frozen scope")

    book = read_json(HERE / "LAUNCH_BOOK.json")
    if book.get("prelaunch_commit") != PRELAUNCH_COMMIT:
        raise RuntimeError("launch book is not bound to the frozen prelaunch commit")
    if (book.get("starts"), book.get("completed"), book.get("technical_failures"),
            book.get("retries_used")) != (9, 9, 0, 0):
        raise RuntimeError("launch book must be exactly nine completed gate cells")
    if len(book.get("attempts", [])) != 9 or any(
            item.get("status") != "completed" or item.get("scene") != 1600000
            for item in book["attempts"]):
        raise RuntimeError("the first-scene attempt ledger is not nine completed cells")
    expected_gate_jobs = [job for job in r39_run.jobs() if job["scene"] == 1600000]
    if [(a["index"], a["arm"]) for a in book["attempts"]] != [
            (j["index"], j["arm"]) for j in expected_gate_jobs]:
        raise RuntimeError("gate attempts do not match the frozen nine-cell job order")
    if book.get("hard_cap") != 153 or book.get("planned_normal_starts") != 144:
        raise RuntimeError("normal launch plan or hard cap changed")

    # v2_common imports with its historical Stage C paths. Bind this fresh
    # process to ROUND39 before reading the already-populated shared ledger.
    bind_round39_output()
    claim_names = {path.name for path in (HERE / "claims").glob("*.claim")}
    expected_claims = {f"cell_{index:03d}_retry_0.claim" for index in range(9)}
    if claim_names != expected_claims:
        raise RuntimeError("claim directory must contain only the nine completed gate claims")
    row_paths = {Path(a["row_path"]).resolve() for a in book["attempts"]}
    if len(row_paths) != 9 or any(not path.is_file() for path in row_paths):
        raise RuntimeError("one or more first-scene rows are missing or duplicated")
    pending_rows = [job for job in r39_run.jobs() if job["scene"] != 1600000
                    and r39_run.row_path(job["arm"], job["scene"]).exists()]
    if pending_rows:
        raise RuntimeError("later-scene rows already exist; refusing duplicate runs")
    if r39_run.v2_common.ledger_total() != 9:
        raise RuntimeError("the real v2 launch ledger must be exactly nine before continuation")
    return book, expected_gate_jobs


def continue_matrix() -> None:
    book, _ = require_gate_and_resume_state()
    lock_path = HERE / ".runner.lock"
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("another ROUND39 process holds the exclusive launch lock") from error

    with os.fdopen(lock_fd, "w", encoding="utf-8") as sink:
        sink.write(json.dumps({"pid": os.getpid(), "prelaunch_commit": PRELAUNCH_COMMIT,
                               "continuation_commit": git("rev-parse", "HEAD"),
                               "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    try:
        bind_round39_output()
        common = r39_run.v2_common
        for directory in (common.OUTPUT_DIR, common.LOGS, common.ROWS,
                          common.REJECTED, common.CHECKPOINTS):
            directory.mkdir(parents=True, exist_ok=True)
        if common.ledger_total() != 9:
            raise RuntimeError("launch ledger changed while acquiring the continuation lock")

        book["status"] = "full_matrix_running_after_gate_recheck"
        book["continuation_commit"] = git("rev-parse", "HEAD")
        book["continuation_started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        atomic_json(HERE / "LAUNCH_BOOK.json", book)

        remaining = [job for job in r39_run.jobs() if job["scene"] != 1600000]
        if len(remaining) != 135:
            raise RuntimeError(f"frozen continuation must contain 135 jobs, got {len(remaining)}")
        for job in remaining:
            # run_one atomically creates the exclusive claim before any environment starts.
            r39_run.run_one(job)

        final_book = read_json(HERE / "LAUNCH_BOOK.json")
        if (final_book.get("starts"), final_book.get("completed"),
                final_book.get("technical_failures")) != (144, 144, 0):
            raise RuntimeError("normal 144-cell launch book did not close")
        if len(final_book.get("attempts", [])) != 144 or any(
                attempt.get("status") != "completed" for attempt in final_book["attempts"]):
            raise RuntimeError("attempt ledger is incomplete or has technical failures")
        if common.ledger_total() != 144 or common.ledger_total() > common.LAUNCH_CAP:
            raise RuntimeError("real Stage C launch ledger does not close at 144")
        final_book["status"] = "complete"
        final_book["normal_cells_completed"] = 144
        final_book["retries_used"] = 0
        final_book["optimizer_steps"] = 0
        final_book["training_launches"] = 0
        final_book["checkpoint_writes"] = 0
        final_book["real_start_ledger"] = common.ledger_report()
        final_book["continuation_finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        atomic_json(HERE / "LAUNCH_BOOK.json", final_book)
        print(json.dumps({"matrix": "COMPLETE", "starts": 144, "completed": 144,
                          "technical_failures": 0, "retries_used": 0,
                          "optimizer_steps": 0}, ensure_ascii=False), flush=True)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continue", dest="continue_matrix", action="store_true",
                        help="continue the remaining cells after the corrected gate")
    args = parser.parse_args()
    if not args.continue_matrix:
        parser.error("explicit --continue is required")
    continue_matrix()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
