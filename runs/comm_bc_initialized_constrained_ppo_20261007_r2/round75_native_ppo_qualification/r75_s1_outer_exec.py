#!/usr/bin/env python
"""ROUND75 S1 outer execution record. Pure stdlib; performs no neural work itself.

This wrapper does NOT modify the frozen runner. It re-verifies the pre-charge identities,
registers exactly ONE attempt with an O_EXCL marker, writes the pre-launch manifest, launches
the frozen runner once, captures the complete stdout/stderr bytes and the REAL exit code, and
writes a read-only post-run ledger snapshot. A wrapper-side failure is recorded as a failure and
is never rewritten as a success, and nothing is ever retried here.
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REQUEST_ID = "R75_S1_NATIVE_QUALIFICATION_20261015"
RUNNER_NAME = "r75_s1_runner.py"
RUNNER_ARGV = ["--run-s1"]
TIMEOUT_S = 7200

EXPECT_CODE = {
    "r75_s1_runner.py": "2a3586f17ba36ffbde9686d06c9278a0699c5c936fb94d12bede2b6a5f216b91",
    "r75_s1_precheck.py": "20a1b81fbe7bc8f9e3e6e6e546bcbb0316a96b48192e60f73b16c27a3dd071e9",
    "r75_frozen.py": "a4b556a687c3aff9650404d11149cb387879a4b4009c85502fc970d7bbe27996",
    "r75_guard.py": "4d730f71fb34ad411043c65141b6bcbb3f7d2f3c4668a6b53054e6400e35d55f",
    "r75_replay_adapter.py": "0034fa40fbb55362646c84e8ef987e1aa01d72c619bd1c82a244dc88a0224dc2",
    "ROUND75_S1_RUN_PLAN.json": "34024113349cdb2f774b9d2751ec0e1f46800ae2fb5720470254432b16ae26c6",
}
EXPECT_DATA = {
    "row": (r"runs/comm_bc_initialized_constrained_ppo_20261007_r2/round58_gen2/raw/rows/v2_r44_train_s11_e0_s1300027_a1.json",
            "882711788d8ccae4ec8092b4eb26df7da6771c9b352b6c7507624d3644abebd8"),
    "tape": (r"runs/comm_bc_initialized_constrained_ppo_20261007_r2/round58_gen2/raw/logs/v2_r44_train_s11_e0_s1300027_a1_tape.jsonl",
             "6d4c4363ff490fafe08b55bbf369dcfaf6284b741c2802cf9ef3374d9b792421"),
    "model": (r"runs/comm_rule_warmstart_cost_learning_20261006_r2/checkpoints/BC_POST_MODEL_s11.pt",
              "76ffc86c4aca26fb3b27380d5195a0cbd07822f7b85d1f1070a5550776dd458a"),
}
MARKER = "r75_s1_outer_attempt_marker.json"
PRELAUNCH = "r75_s1_outer_prelaunch_manifest.json"
OUTCOME = "r75_s1_outer_outcome.json"
WRAPPER_FAILURE = "r75_s1_outer_wrapper_failure.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return {"unix": time.time(),
            "iso": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}


def git(repo, *args):
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True)
    return proc.returncode, proc.stdout.decode("utf-8", "replace").strip(), proc.stderr.decode("utf-8", "replace").strip()


def ledger_snapshot(runtime_dir, frozen):
    files = {}
    for name in sorted(os.listdir(runtime_dir)):
        p = os.path.join(runtime_dir, name)
        if os.path.isfile(p):
            files[name] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    events, per_unit = [], {}
    total = os.path.join(runtime_dir, "r75_total_ledger.jsonl")
    if os.path.isfile(total):
        with open(total, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    events.append({"unparsed_line_bytes": len(line)})
                    continue
                events.append(rec)
                unit = rec.get("unit")
                if unit:
                    slot = per_unit.setdefault(unit, {"used_normal": 0, "used_reserve": 0,
                                                      "refusals": 0, "charged_events": 0})
                    for key in ("used_normal", "used_reserve", "refusals"):
                        if key in rec:
                            slot[key] = max(slot[key], int(rec[key]))
                    if rec.get("event") not in ("ledger-opened",):
                        slot["charged_events"] += 1
    return {"root": str(runtime_dir), "files": files, "total_ledger_events": len(events),
            "per_unit_last_known": per_unit,
            "frozen_snapshot_match": files == frozen if frozen else None,
            "note": ("read-only; used_normal/used_reserve/refusals are the maximum value seen in "
                     "any ledger event for that unit, charged_events counts non-open events")}


def write_json(path, payload):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def main():
    here = Path(__file__).resolve().parent
    repo = here
    while not (repo / ".git").exists():
        repo = repo.parent
    s1_dir = here / "raw" / "s1"
    runtime_dir = here / "raw" / "runtime"
    plan = json.loads((here / "ROUND75_S1_RUN_PLAN.json").read_text(encoding="utf-8"))
    frozen = plan["canonical_ledger_frozen_snapshot"]["files"]
    wrapper_sha = sha256_file(__file__)

    problems = []
    rc, head, err = git(repo, "rev-parse", "HEAD")
    if rc != 0:
        problems.append("git rev-parse HEAD failed: " + err)
    rc, parent, err = git(repo, "rev-parse", "HEAD^")
    if rc != 0:
        problems.append("git rev-parse HEAD^ failed: " + err)
    rc, branch, err = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if rc != 0:
        problems.append("git branch read failed: " + err)
    rc, dirty, err = git(repo, "status", "--porcelain=v1")
    if rc != 0:
        problems.append("git status failed: " + err)

    for name, want in sorted(EXPECT_CODE.items()):
        p = here / name
        if not p.is_file():
            problems.append("missing code file: " + name)
            continue
        got = sha256_file(p)
        if got != want:
            problems.append("code sha256 mismatch for %s: %s" % (name, got))
    for role, (rel, want) in sorted(EXPECT_DATA.items()):
        p = repo / rel
        if not p.is_file():
            problems.append("missing data file (%s): %s" % (role, rel))
            continue
        got = sha256_file(p)
        if got != want:
            problems.append("data sha256 mismatch for %s: %s" % (role, got))

    snap_before = ledger_snapshot(runtime_dir, frozen)
    if snap_before["frozen_snapshot_match"] is not True:
        problems.append("canonical ledger files do not reproduce the frozen zero-start snapshot")
    if snap_before["total_ledger_events"] != 12:
        problems.append("canonical ledger has %d events, 12 expected at the zero start"
                        % snap_before["total_ledger_events"])
    nonzero = {unit: v for unit, v in snap_before["per_unit_last_known"].items()
               if v.get("used_normal") or v.get("used_reserve") or v.get("refusals")
               or v.get("charged_events")}
    if nonzero:
        problems.append("canonical ledger is not at the zero start: %s" % json.dumps(nonzero))
    if (here / "raw" / "s1").exists() and any((here / "raw" / "s1").iterdir()):
        problems.append("raw/s1 already exists and is not empty; a prior attempt may have run")

    precheck = {"problems": problems, "head": head, "parent": parent, "branch": branch,
                "worktree_porcelain": dirty, "wrapper_sha256": wrapper_sha,
                "code_sha256": {}, "data_sha256": {}, "ledger_zero_start": snap_before,
                "verified_at": now()}
    for name in sorted(EXPECT_CODE):
        p = here / name
        if p.is_file():
            precheck["code_sha256"][name] = sha256_file(p)
    for role, (rel, _want) in sorted(EXPECT_DATA.items()):
        p = repo / rel
        if p.is_file():
            precheck["data_sha256"][role] = {"path": rel, "sha256": sha256_file(p)}

    s1_dir.mkdir(parents=True, exist_ok=True)
    write_json(s1_dir / PRELAUNCH, {"kind": "round75-s1-outer-prelaunch/1",
                                    "request_id": REQUEST_ID, **precheck})

    if problems:
        write_json(s1_dir / WRAPPER_FAILURE,
                   {"kind": "round75-s1-outer-refusal/1", "request_id": REQUEST_ID,
                    "refused_at": now(), "reason": "pre-charge verification failed",
                    "problems": problems, "head": head, "parent": parent,
                    "runner_launched": False, "note": ("nothing paid was started; no attempt "
                                                       "marker was created and no slot consumed")})
        sys.stderr.write("S1 OUTER REFUSED: pre-charge verification failed\n")
        for p in problems:
            sys.stderr.write("  - %s\n" % p)
        return 3

    # ONE attempt marker, exclusive creation. A second request must never start.
    marker_path = s1_dir / MARKER
    try:
        fd = os.open(str(marker_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        write_json(s1_dir / WRAPPER_FAILURE,
                   {"kind": "round75-s1-outer-refusal/1", "request_id": REQUEST_ID,
                    "refused_at": now(), "reason": "duplicate request: the attempt marker already exists",
                    "problems": ["an attempt marker for this request already exists"],
                    "runner_launched": False,
                    "note": "the frozen runner was NOT launched; no retry is permitted here"})
        sys.stderr.write("S1 OUTER REFUSED: duplicate request, marker already exists\n")
        return 4
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"kind": "round75-s1-outer-attempt-marker/1", "request_id": REQUEST_ID,
                   "created_at": now(), "wrapper_pid": os.getpid(), "head": head, "parent": parent,
                   "argv": [sys.executable, str(here / RUNNER_NAME)] + RUNNER_ARGV,
                   "note": ("created with O_EXCL before any paid Python started; its existence "
                            "forbids a second launch of this request")}, fh, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())

    out_bin = s1_dir / "r75_s1_outer_stdout.bin"
    err_bin = s1_dir / "r75_s1_outer_stderr.bin"
    started = now()
    proc = subprocess.Popen([sys.executable, str(here / RUNNER_NAME)] + RUNNER_ARGV,
                            cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timed_out = False
    try:
        out, err = proc.communicate(timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        out, err = proc.communicate()
    ended = now()
    returncode = proc.returncode

    with open(out_bin, "wb") as fh:
        fh.write(out)
    with open(err_bin, "wb") as fh:
        fh.write(err)
    out_text = out.decode("utf-8", "replace")
    err_text = err.decode("utf-8", "replace")
    with open(s1_dir / "r75_s1_outer_stdout.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(out_text)
    with open(s1_dir / "r75_s1_outer_stderr.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(err_text)

    snap_after = ledger_snapshot(runtime_dir, None)
    produced = sorted(p.name for p in s1_dir.iterdir() if p.is_file())
    outcome = {"kind": "round75-s1-outer-outcome/1", "request_id": REQUEST_ID,
               "runner_launched": True, "timed_out": timed_out, "timeout_s": TIMEOUT_S,
               "started_at": started, "ended_at": ended,
               "wall_seconds": round(ended["unix"] - started["unix"], 3),
               "child_pid": proc.pid, "returncode": returncode,
               "runner_exit_code": returncode,
               "stdout_bytes": len(out), "stderr_bytes": len(err),
               "stdout_sha256": hashlib.sha256(out).hexdigest(),
               "stderr_sha256": hashlib.sha256(err).hexdigest(),
               "stdout_utf8_decoded_losslessly": _lossless(out),
               "stderr_utf8_decoded_losslessly": _lossless(err),
               "stdout_tail": out_text[-4000:], "stderr_tail": err_text[-4000:],
               "ledger_after": snap_after, "s1_dir_files": produced,
               "head": head, "parent": parent, "branch": branch,
               "wrapper_sha256": wrapper_sha,
               "note": ("returncode is the REAL child exit code; a wrapper-side or outer failure "
                        "is never rewritten as 0. success means only that the frozen R58 replay "
                        "native PPO / real Adam / real serialized-restore qualification passed")}
    write_json(s1_dir / OUTCOME, outcome)
    return 0 if (returncode == 0 and not timed_out) else 5


def _lossless(raw):
    try:
        raw.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:                     # outer failure: recorded, never faked
        here = Path(__file__).resolve().parent
        s1_dir = here / "raw" / "s1"
        try:
            s1_dir.mkdir(parents=True, exist_ok=True)
            write_json(s1_dir / WRAPPER_FAILURE,
                       {"kind": "round75-s1-outer-wrapper-failure/1",
                        "request_id": REQUEST_ID, "failed_at": now(),
                        "error_type": type(exc).__name__, "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "note": ("the outer wrapper itself failed; this is NOT a successful run "
                                 "and NOT a runner exit code")})
        except Exception:
            pass
        sys.stderr.write("S1 OUTER WRAPPER FAILED: %s: %s\n" % (type(exc).__name__, exc))
        raise