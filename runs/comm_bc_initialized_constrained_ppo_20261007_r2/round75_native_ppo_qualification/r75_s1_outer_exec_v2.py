#!/usr/bin/env python
"""ROUND75 S1 outer execution record, REVISED. Pure stdlib; performs no neural work itself.

Revision of r75_s1_outer_exec.py, which is preserved byte-identical as the version that actually
executed request R75_S1_NATIVE_QUALIFICATION_20261015. Five things changed and nothing else:

1. Return-code attribution. The child return code, the wrapper return value and a wrapper-side
   exception are three different facts and are recorded separately. SystemExit is the wrapper's own
   exit transport, so the __main__ guard lets it through BEFORE the BaseException handler; a
   SystemExit can no longer be written down as a wrapper crash.
2. An explicit prepare/launch split. ``--prepare`` (the default) verifies the pre-charge identities
   and writes a prepared record. It creates NO attempt marker and starts NO child. ``--launch``
   additionally creates the O_EXCL attempt marker and runs the frozen runner exactly once.
   ``--launch`` requires an explicit --request-id, so the placeholder follow-up id below is never a
   run permission: renaming the request is not an authorization.
3. The execution identity and the private attempt directory are handed to the child as well,
   so the runner's own bundle, failure receipts, trial record and durable counters land in the
   SAME directory and under the SAME request id as this record. The runner refuses, before any
   paid import, a missing identity, the historical request id in any directory, the historical
   raw/s1 directory, a directory outside this output root, and a repeated request id.
4. An independent output location. This revision writes under raw/s1_revised/ and never touches
   raw/s1/, which holds the first attempt's originals.
5. A preserved first failure is registered, not feared. The nine originals under raw/s1/ are
   re-hashed on every run to prove they were not touched, and the request id that produced them
   is refused in ANY output directory. A different, explicitly authorized request id writing to
   its own directory is not refused merely because a preserved history exists; nothing under
   raw/s1/ is deleted, cleared, rewritten or counted as this revision's output.

The complete stdout/stderr BYTES are captured to .bin files, which are authoritative; the .txt
copies are decoded with errors="replace" and any loss is registered explicitly. A post-run ledger
snapshot is read-only. Nothing is ever retried here.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

FOLLOW_UP_REQUEST_ID = "R75_S1_NATIVE_QUALIFICATION_FOLLOWUP_UNASSIGNED"
OUTPUT_DIR_NAME = "raw/s1_revised"
RUNNER_NAME = "r75_s1_runner.py"
RUNNER_ARGV = ["--run-s1"]
TIMEOUT_S = 7200
PREPARED_RECORD = "r75_s1_outer_v2_prepared.json"
OUTCOME_RECORD = "r75_s1_outer_v2_outcome.json"
WRAPPER_FAILURE_RECORD = "r75_s1_outer_v2_wrapper_failure.json"

EXPECT_CODE = {
    "r75_s1_runner.py": "0d773a54b201f08bb02e075d763dc720e87f8e652b02a6e3bf2b9079f69b893f",
    "r75_s1_precheck.py": "f44ce536abd5e0368b8bb99ed0a09167ff9ba699dcb507c9c05d0e60472c9f0c",
    "r75_frozen.py": "83813d6e06971a1e1fc1e25d9e9f94b0fa72310e300e037c0fa73d539c540b80",
    "r75_guard.py": "4d730f71fb34ad411043c65141b6bcbb3f7d2f3c4668a6b53054e6400e35d55f",
    "r75_replay_adapter.py": "0034fa40fbb55362646c84e8ef987e1aa01d72c619bd1c82a244dc88a0224dc2",
    "ROUND75_S1_RUN_PLAN.json": "ab7120d5c469f2936ea7162023ad70b1d04dc3d50955c4e6535c46e6480ce735",
}
EXPECT_DATA = {
    "row": (r"runs/comm_bc_initialized_constrained_ppo_20261007_r2/round58_gen2/raw/rows/v2_r44_train_s11_e0_s1300027_a1.json",
            "882711788d8ccae4ec8092b4eb26df7da6771c9b352b6c7507624d3644abebd8"),
    "tape": (r"runs/comm_bc_initialized_constrained_ppo_20261007_r2/round58_gen2/raw/logs/v2_r44_train_s11_e0_s1300027_a1_tape.jsonl",
             "6d4c4363ff490fafe08b55bbf369dcfaf6284b741c2802cf9ef3374d9b792421"),
    "model": (r"runs/comm_rule_warmstart_cost_learning_20261006_r2/checkpoints/BC_POST_MODEL_s11.pt",
              "76ffc86c4aca26fb3b27380d5195a0cbd07822f7b85d1f1070a5550776dd458a"),
}
# The first attempt really ran and failed at the paid import. Its nine originals are preserved
# byte-for-byte under raw/s1/ and are registered here so that any later run can prove it did not
# touch them. Their existence is NOT a refusal: only a repeated request id is.
FIRST_ATTEMPT_DIR = "raw/s1"
FIRST_ATTEMPT_REQUEST_ID = "R75_S1_NATIVE_QUALIFICATION_20261015"
EXPECT_FIRST_ATTEMPT_FILES = {
    "r75_s1_outer_attempt_marker.json":
        "3723fb40b71bbee3fbf2ab56770c3c833d0278540b8ac542cba6188ddef0e496",
    "r75_s1_outer_outcome.json":
        "61475ef3b77e4401ec3048718abc9b19f7a811187978310e8e0b5403ea6e41fc",
    "r75_s1_outer_prelaunch_manifest.json":
        "fe93a5bda22d23a9de8255af1f959f57d28aee640cd91ba84500575738c4f09e",
    "r75_s1_outer_stderr.bin":
        "db569234b8624fbca27eea14ff001d0341f4261498bbe5e6355231ce0856c438",
    "r75_s1_outer_stderr.txt":
        "7c031d1e17b2a70c19ba3007eba95880e31e2d459045282ff6d3e1e1155eb639",
    "r75_s1_outer_stdout.bin":
        "37b5f7e364dcb0fbb6a435b309fff1ad2a284914a4da163de26dbc8aa1ddf22e",
    "r75_s1_outer_stdout.txt":
        "37b5f7e364dcb0fbb6a435b309fff1ad2a284914a4da163de26dbc8aa1ddf22e",
    "r75_s1_outer_wrapper_failure.json":
        "0810ac8e9a838010dbaa5b34ba50d635052e901e5e23e3939ebf69b0d867e7a6",
    "r75_s1_outer_wrapper_failure_clarification.json":
        "831390add015d4d60cb17db5164214bb966815f6bb11b1cfd02309f29f79c148",
}


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now():
    return {"unix": time.time(),
            "iso": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}


def git(repo, *args):
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True)
    return (proc.returncode, proc.stdout.decode("utf-8", "replace").strip(),
            proc.stderr.decode("utf-8", "replace").strip())


def ledger_snapshot(runtime_dir, frozen):
    files = {}
    for name in sorted(os.listdir(runtime_dir)):
        path = os.path.join(runtime_dir, name)
        if os.path.isfile(path):
            files[name] = {"sha256": sha256_file(path), "bytes": os.path.getsize(path)}
    events, per_unit = [], {}
    total = os.path.join(runtime_dir, "r75_total_ledger.jsonl")
    if os.path.isfile(total):
        with open(total, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    events.append({"unparsed_line_bytes": len(line)})
                    continue
                events.append(record)
                unit = record.get("unit")
                if unit:
                    slot = per_unit.setdefault(unit, {"used_normal": 0, "used_reserve": 0,
                                                      "refusals": 0, "charged_events": 0})
                    for key in ("used_normal", "used_reserve", "refusals"):
                        if key in record:
                            slot[key] = max(slot[key], int(record[key]))
                    if record.get("event") != "ledger-opened":
                        slot["charged_events"] += 1
    return {"root": str(runtime_dir), "files": files, "total_ledger_events": len(events),
            "per_unit_last_known": per_unit,
            "frozen_snapshot_match": (files == frozen) if frozen else None,
            "note": ("read-only; used_normal/used_reserve/refusals are the maximum value seen in "
                     "any ledger event for that unit, charged_events counts non-open events")}


def write_json(path, payload):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def first_attempt_state(here):
    """Register the preserved first failed attempt. Read-only; its existence is never a refusal."""
    first = here / FIRST_ATTEMPT_DIR
    if not first.exists():
        return {"present": False, "request_id": FIRST_ATTEMPT_REQUEST_ID, "files": {},
                "unregistered_files": [],
                "problems": ["raw/s1 is missing; the first attempt originals must be preserved"]}
    problems, files = [], {}
    for name, want in sorted(EXPECT_FIRST_ATTEMPT_FILES.items()):
        path = first / name
        if not path.is_file():
            problems.append("first attempt original missing: " + name)
            continue
        got = sha256_file(path)
        if got != want:
            problems.append("first attempt original changed: %s is %s, expected %s"
                            % (name, got, want))
        files[name] = {"sha256": got, "bytes": os.path.getsize(path)}
    unregistered = sorted(entry.name for entry in first.iterdir() if entry.is_file()
                          and entry.name not in EXPECT_FIRST_ATTEMPT_FILES)
    return {"present": True, "request_id": FIRST_ATTEMPT_REQUEST_ID, "files": files,
            "unregistered_files": unregistered, "problems": problems}


def duplicate_request_refusal(request_id, s1_dir, here):
    """A request id runs at most once, in any output directory. Read-only; returns a reason or None."""
    if request_id == FIRST_ATTEMPT_REQUEST_ID:
        return ("request id %s is the first failed attempt; its O_EXCL marker forbids a second "
                "launch in any directory" % request_id)
    for directory in (s1_dir, here / FIRST_ATTEMPT_DIR):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("r75_s1_outer_v2_attempt_marker_*.json")):
            try:
                recorded = json.loads(path.read_text(encoding="utf-8")).get("request_id")
            except (ValueError, OSError):
                continue
            if recorded == request_id:
                return ("an attempt marker for request id %s already exists at %s"
                        % (request_id, path.name))
    return None


def verify_pre_charge(repo, here, runtime_dir, plan):
    """Re-derive every identity the paid path depends on. Read-only; imports nothing paid."""
    problems = []
    for args in (("rev-parse", "HEAD"), ("rev-parse", "HEAD^")):
        code, out, err = git(repo, *args)
        if code != 0:
            problems.append("git %s failed: %s" % (" ".join(args), err))
    rc, head, _ = git(repo, "rev-parse", "HEAD")
    rc2, parent, _ = git(repo, "rev-parse", "HEAD^")
    rc3, branch, _ = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    rc4, dirty, _ = git(repo, "status", "--porcelain=v1")
    for name, want in sorted(EXPECT_CODE.items()):
        path = here / name
        if not path.is_file():
            problems.append("missing code file: " + name)
            continue
        got = sha256_file(path)
        if got != want:
            problems.append("code sha256 mismatch for %s: %s" % (name, got))
    for role, (rel, want) in sorted(EXPECT_DATA.items()):
        path = repo / rel
        if not path.is_file():
            problems.append("missing data file (%s): %s" % (role, rel))
            continue
        got = sha256_file(path)
        if got != want:
            problems.append("data sha256 mismatch for %s: %s" % (role, got))
    frozen = plan["canonical_ledger_frozen_snapshot"]["files"]
    snapshot = ledger_snapshot(runtime_dir, frozen)
    if snapshot["frozen_snapshot_match"] is not True:
        problems.append("canonical ledger files do not reproduce the frozen zero-start snapshot")
    if snapshot["total_ledger_events"] != 12:
        problems.append("canonical ledger has %d events, 12 expected at the zero start"
                        % snapshot["total_ledger_events"])
    nonzero = {unit: value for unit, value in snapshot["per_unit_last_known"].items()
               if value.get("used_normal") or value.get("used_reserve")
               or value.get("refusals") or value.get("charged_events")}
    if nonzero:
        problems.append("canonical ledger is not at the zero start: %s" % json.dumps(nonzero))
    first_attempt = first_attempt_state(here)
    problems.extend(first_attempt["problems"])
    record = {"problems": problems, "head": head, "parent": parent, "branch": branch,
              "worktree_porcelain": dirty, "code_sha256": {}, "data_sha256": {},
              "ledger_zero_start": snapshot, "first_attempt": first_attempt, "verified_at": now()}
    for name in sorted(EXPECT_CODE):
        path = here / name
        if path.is_file():
            record["code_sha256"][name] = sha256_file(path)
    for role, (rel, _want) in sorted(EXPECT_DATA.items()):
        path = repo / rel
        if path.is_file():
            record["data_sha256"][role] = {"path": rel, "sha256": sha256_file(path)}
    return record


def register_attempt(s1_dir, request_id, argv, head, parent, wrapper_sha):
    """ONE attempt marker per request id, exclusive creation. A duplicate never starts."""
    marker = s1_dir / ("r75_s1_outer_v2_attempt_marker_%s.json" % request_id)
    try:
        handle = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return None, marker
    with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"kind": "round75-s1-outer-v2-attempt-marker/1", "request_id": request_id,
                   "created_at": now(), "wrapper_pid": os.getpid(), "head": head,
                   "parent": parent, "argv": argv,
                   "note": ("created with O_EXCL before any paid Python started; its existence "
                            "forbids a second launch of this request id")}, fh, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    return marker, marker


def launch_once(repo, runner_path, argv, timeout_s):
    """Run the frozen runner exactly once and return the child facts. No retry, no fallback."""
    started = now()
    proc = subprocess.Popen([sys.executable, str(runner_path)] + argv, cwd=str(repo),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        out, err = proc.communicate()
    ended = now()
    return {"started_at": started, "ended_at": ended, "wall_seconds": round(ended["unix"] - started["unix"], 3),
            "child_pid": proc.pid, "child_returncode": proc.returncode, "timed_out": timed_out,
            "timeout_s": timeout_s, "stdout_bytes": out, "stderr_bytes": err}


def decode_report(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "utf8_decoded_losslessly": _lossless(raw)}


def _lossless(raw):
    try:
        raw.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def record_outcome(s1_dir, request_id, child, head, parent, branch, wrapper_sha, snapshot,
                   produced, launch_refusal=None):
    out, err = child["stdout_bytes"], child["stderr_bytes"]
    payload = {"kind": "round75-s1-outer-v2-outcome/1", "request_id": request_id,
               "runner_launched": True, "timed_out": child["timed_out"],
               "started_at": child["started_at"], "ended_at": child["ended_at"],
               "wall_seconds": child["wall_seconds"], "child_pid": child["child_pid"],
               "child_returncode": child["child_returncode"],
               "wrapper_return": 0 if (child["child_returncode"] == 0 and not child["timed_out"]) else 5,
               "wrapper_exception": None,
               "stdout": decode_report(out), "stderr": decode_report(err),
               "stdout_tail": out.decode("utf-8", "replace")[-4000:],
               "stderr_tail": err.decode("utf-8", "replace")[-4000:],
               "ledger_after": snapshot, "s1_dir_files": produced,
               "head": head, "parent": parent, "branch": branch,
               "wrapper_sha256": wrapper_sha,
               "note": ("child_returncode is the REAL child exit code, wrapper_return is this "
                        "process's own exit value and wrapper_exception is a wrapper-side crash; "
                        "the three are never merged. The .bin files are authoritative; the .txt "
                        "copies are lossy-decoded and any loss is registered above.")}
    if launch_refusal is not None:
        payload["launch_refusal"] = launch_refusal
    write_json(s1_dir / OUTCOME_RECORD, payload)
    return payload


def record_wrapper_failure(s1_dir, request_id, error):
    write_json(s1_dir / WRAPPER_FAILURE_RECORD,
               {"kind": "round75-s1-outer-v2-wrapper-failure/1", "request_id": request_id,
                "failed_at": now(), "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(),
                "note": ("a WRAPPER-side exception only; it is not a runner exit code, not a "
                         "second attempt and not an additional failure of the child")})


def prepare(repo, here, s1_dir, request_id, plan, runtime_dir):
    """Verify the pre-charge identities and write the prepared record. Starts nothing."""
    record = verify_pre_charge(repo, here, runtime_dir, plan)
    wrapper_sha = sha256_file(__file__)
    payload = {"kind": "round75-s1-outer-v2-prepared/1", "request_id": request_id,
               "prepared_at": now(), "wrapper_sha256": wrapper_sha,
               "launched": False, "attempt_marker_created": False, **record,
               "note": ("preparation only: the identities were re-verified and recorded, no "
                        "attempt marker was created and no child was started")}
    write_json(s1_dir / PREPARED_RECORD, payload)
    return payload, record["problems"]


def run(repo, here, s1_dir, request_id, plan, runtime_dir):
    """Verify, register the ONE attempt marker, launch once, record the outcome."""
    record = verify_pre_charge(repo, here, runtime_dir, plan)
    wrapper_sha = sha256_file(__file__)
    if record["problems"]:
        write_json(s1_dir / WRAPPER_FAILURE_RECORD,
                   {"kind": "round75-s1-outer-v2-refusal/1", "request_id": request_id,
                    "refused_at": now(), "reason": "pre-charge verification failed",
                    "problems": record["problems"], "runner_launched": False,
                    "note": "nothing paid was started and no attempt marker was created"})
        return 3, record["problems"]
    refusal = duplicate_request_refusal(request_id, s1_dir, here)
    if refusal is not None:
        write_json(s1_dir / WRAPPER_FAILURE_RECORD,
                   {"kind": "round75-s1-outer-v2-refusal/1", "request_id": request_id,
                    "refused_at": now(), "reason": "duplicate request id",
                    "problems": [refusal], "runner_launched": False,
                    "note": ("the frozen runner was NOT launched; a request id runs at most once "
                             "and the first attempt id can never run again")})
        return 4, [refusal]
    argv = ([sys.executable, str(here / RUNNER_NAME)] + RUNNER_ARGV
            + ["--request-id", request_id, "--attempt-dir", str(s1_dir)])
    marker, marker_path = register_attempt(s1_dir, request_id, argv, record["head"],
                                           record["parent"], wrapper_sha)
    if marker is None:
        write_json(s1_dir / WRAPPER_FAILURE_RECORD,
                   {"kind": "round75-s1-outer-v2-refusal/1", "request_id": request_id,
                    "refused_at": now(),
                    "reason": "duplicate request: the attempt marker %s already exists" % marker_path.name,
                    "problems": ["an attempt marker for this request id already exists"],
                    "runner_launched": False,
                    "note": "the frozen runner was NOT launched; no retry is permitted here"})
        return 4, ["duplicate attempt marker: %s" % marker_path.name]
    write_json(s1_dir / PREPARED_RECORD,
               {"kind": "round75-s1-outer-v2-prepared/1", "request_id": request_id,
                "prepared_at": now(), "wrapper_sha256": wrapper_sha, "launched": True,
                "attempt_marker_created": True, **record})
    child = launch_once(repo, here / RUNNER_NAME, argv, TIMEOUT_S)
    with open(s1_dir / "r75_s1_outer_v2_stdout.bin", "wb") as handle:
        handle.write(child["stdout_bytes"])
    with open(s1_dir / "r75_s1_outer_v2_stderr.bin", "wb") as handle:
        handle.write(child["stderr_bytes"])
    with open(s1_dir / "r75_s1_outer_v2_stdout.txt", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(child["stdout_bytes"].decode("utf-8", "replace"))
    with open(s1_dir / "r75_s1_outer_v2_stderr.txt", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(child["stderr_bytes"].decode("utf-8", "replace"))
    snapshot = ledger_snapshot(runtime_dir, None)
    produced = sorted(path.name for path in s1_dir.iterdir() if path.is_file())
    record_outcome(s1_dir, request_id, child, record["head"], record["parent"], record["branch"],
                   wrapper_sha, snapshot, produced)
    return (0 if (child["child_returncode"] == 0 and not child["timed_out"]) else 5), []


def _request_id_from_argv(argv):
    """The request id this process was invoked with, or the placeholder when none was given."""
    for index, item in enumerate(argv):
        if item == "--request-id" and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith("--request-id="):
            return item.split("=", 1)[1]
    return FOLLOW_UP_REQUEST_ID


def main(argv=None):
    parser = argparse.ArgumentParser(description="ROUND75 S1 outer execution record, revised")
    parser.add_argument("--launch", action="store_true",
                        help="register the ONE attempt marker and run the frozen runner once")
    parser.add_argument("--prepare", action="store_true",
                        help="verify and record only; creates no marker and starts no child")
    parser.add_argument("--request-id", default=FOLLOW_UP_REQUEST_ID,
                        help="the request id this attempt belongs to")
    args = parser.parse_args(argv)
    here = Path(__file__).resolve().parent
    repo = here
    while not (repo / ".git").exists():
        repo = repo.parent
    s1_dir = here / OUTPUT_DIR_NAME
    runtime_dir = here / "raw" / "runtime"
    plan = json.loads((here / "ROUND75_S1_RUN_PLAN.json").read_text(encoding="utf-8"))
    s1_dir.mkdir(parents=True, exist_ok=True)
    if args.launch:
        if args.request_id == FOLLOW_UP_REQUEST_ID:
            sys.stderr.write("S1 OUTER V2 REFUSED: --launch requires an explicit --request-id; "
                             "the placeholder follow-up id is not a run permission\n")
            return 2
        code, problems = run(repo, here, s1_dir, args.request_id, plan, runtime_dir)
        for problem in problems:
            sys.stderr.write("  - %s\n" % problem)
        return code
    payload, problems = prepare(repo, here, s1_dir, args.request_id, plan, runtime_dir)
    if problems:
        sys.stderr.write("S1 OUTER V2 REFUSED: pre-charge verification failed\n")
        for problem in problems:
            sys.stderr.write("  - %s\n" % problem)
        return 3
    print(json.dumps({"kind": "round75-s1-outer-v2-prepared-summary/1",
                      "request_id": args.request_id, "head": payload["head"],
                      "parent": payload["parent"], "launched": False,
                      "attempt_marker_created": False,
                      "code_sha256": payload["code_sha256"],
                      "ledger_zero_start_events": payload["ledger_zero_start"]["total_ledger_events"],
                      "note": ("preparation only; --launch was not given and is not authorized "
                               "by the repair request")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        _EXIT_CODE = main(sys.argv[1:])
    except SystemExit:
        raise                       # the wrapper's own exit transport, never a wrapper crash
    except BaseException as _error:
        try:
            _here = Path(__file__).resolve().parent
            record_wrapper_failure(_here / OUTPUT_DIR_NAME,
                                   _request_id_from_argv(sys.argv[1:]), _error)
        except Exception:
            pass
        sys.stderr.write("S1 OUTER V2 WRAPPER FAILED: %s: %s\n" % (type(_error).__name__, _error))
        raise
    raise SystemExit(_EXIT_CODE)