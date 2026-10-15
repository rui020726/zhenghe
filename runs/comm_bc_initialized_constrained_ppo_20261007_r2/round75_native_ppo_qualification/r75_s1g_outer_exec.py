#!/usr/bin/env python
"""ROUND75 S1G native retry outer execution record. Pure stdlib; performs no neural work itself.

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

FOLLOW_UP_REQUEST_ID = "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015"
OUTPUT_DIR_NAME = "raw/s1g_native_retry"
RUNNER_NAME = "r75_s1_runner.py"
RUNNER_ARGV = ["--run-s1g"]
TIMEOUT_S = 7200
PREPARED_RECORD = "r75_s1g_outer_v2_prepared.json"
OUTCOME_RECORD = "r75_s1g_outer_v2_outcome.json"
WRAPPER_FAILURE_RECORD = "r75_s1g_outer_v2_wrapper_failure.json"

EXPECT_CODE = {
    "r75_s1_runner.py": "4f84184a25d9058df97abf04fb00febe1d6b8c123944fcc9814e78651d3835d4",
    "r75_s1_precheck.py": "88b88c3854f25f81a1398d37066b5d62f4941a7529b349fe5a9d20361733625b",
    "r75_frozen.py": "bb4bae98518cc482f4b733794ee1c3f93097c56c5b3e85d6fed2c65f2a72198e",
    "r75_guard.py": "4d730f71fb34ad411043c65141b6bcbb3f7d2f3c4668a6b53054e6400e35d55f",
    "r75_replay_adapter.py": "0034fa40fbb55362646c84e8ef987e1aa01d72c619bd1c82a244dc88a0224dc2",
    "ROUND75_S1G_AR1_RUN_PLAN.json": "6303c093d0e19d71081e5c1c5d6393f21733d8f7c9c9fb87ec6fd21eecd7ebab",
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
    """Read the guard JSONs and total ledger, then reconcile their real event schemas.

    This is read-only. Charged rows are summed from each event n by category; used_after is a
    second reading, not a substitute for the sum. Guard top-level cumulative values, guard event
    lists and total-ledger event sequences must agree. Any missing or bad schema becomes unknown
    or mismatch and blocks permission; no history is cleared or corrected here.
    """
    runtime_dir = Path(runtime_dir)
    files = {}
    for name in sorted(os.listdir(runtime_dir)):
        path = runtime_dir / name
        if path.is_file():
            files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    total_path = runtime_dir / "r75_total_ledger.jsonl"
    events: list[dict] = []
    unparsed: list[int] = []
    if total_path.is_file():
        with total_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    events.append(json.loads(stripped))
                except ValueError:
                    unparsed.append(len(stripped.encode("utf-8")))
    guard_files = {"physical_optimizer_step": "r75_physical_optimizer_step_guard.json",
                   "offline_trunk_target_row": "r75_offline_trunk_target_row_guard.json",
                   "joint_record": "r75_joint_record_guard.json",
                   "environment_construction": "r75_environment_construction_guard.json"}
    per_unit: dict[str, dict] = {}
    mismatches: list[dict] = []
    for unit, guard_name in sorted(guard_files.items()):
        unit_ledger = [event for event in events
                       if event.get("unit") == unit
                       and event.get("event") in ("charged", "refused")]
        summary: dict = {"used_normal": 0, "used_reserve": 0, "refusals": 0,
                         "charged_events": 0, "guard_events": 0, "total_ledger_events": 0,
                         "used_after_normal": 0, "used_after_reserve": 0,
                         "schema_ok": False, "guard_ledger_match": False}
        valid = True
        problems: list[str] = []
        guard_path = runtime_dir / guard_name
        guard = None
        if not guard_path.is_file():
            valid = False
            problems.append("missing guard file: " + guard_name)
        else:
            try:
                guard = json.loads(guard_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as error:
                guard = None
                valid = False
                problems.append("unreadable guard " + guard_name + ": " + type(error).__name__)
        if not isinstance(guard, dict):
            valid = False
            problems.append("guard is not a JSON object: " + guard_name)
        else:
            required = ("kind", "unit", "used_normal", "used_reserve", "refusals", "events")
            missing = [key for key in required if key not in guard]
            if missing:
                valid = False
                problems.append("missing guard fields: " + ",".join(missing))
            if guard.get("unit") != unit:
                valid = False
                problems.append("guard unit mismatch")
            guard_events = guard.get("events")
            if not isinstance(guard_events, list):
                valid = False
                problems.append("guard events is not a list")
                guard_events = []
        if valid:
            for category, key in (("normal", "used_normal"), ("reserve", "used_reserve")):
                category_events = [event for event in guard_events
                                   if event.get("event") == "charged"
                                   and event.get("category") == category]
                n_sum = 0
                used_after = 0
                for index, event in enumerate(category_events):
                    value = event.get("n")
                    after = event.get("used_after")
                    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                        valid = False
                        problems.append("guard " + category + " event " + str(index) +
                                        " has invalid n")
                        break
                    if not isinstance(after, int) or isinstance(after, bool):
                        valid = False
                        problems.append("guard " + category + " event " + str(index) +
                                        " has invalid used_after")
                        break
                    n_sum += value
                    used_after = after
                if valid:
                    top_value = guard.get(key)
                    if not isinstance(top_value, int) or isinstance(top_value, bool):
                        valid = False
                        problems.append("guard top-level " + key + " is invalid")
                    elif n_sum != top_value or used_after != top_value:
                        valid = False
                        problems.append(
                            "guard " + key + " mismatch: top=" + str(top_value) +
                            ", n_sum=" + str(n_sum) + ", used_after=" + str(used_after))
            refusals = guard.get("refusals")
            if not isinstance(refusals, int) or isinstance(refusals, bool) or refusals < 0:
                valid = False
                problems.append("guard refusals is invalid")
            elif sum(event.get("event") == "refused" for event in guard_events) != refusals:
                valid = False
                problems.append("guard refusal count does not match its refused events")
        if valid:
            for event in unit_ledger:
                kind = event.get("event")
                category = event.get("category")
                if kind == "charged":
                    value = event.get("n")
                    after = event.get("used_after")
                    if category not in ("normal", "reserve"):
                        valid = False
                        problems.append("ledger charged event has invalid category")
                        break
                    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                        valid = False
                        problems.append("ledger charged event has invalid n")
                        break
                    if not isinstance(after, int) or isinstance(after, bool):
                        valid = False
                        problems.append("ledger charged event has invalid used_after")
                        break
                    if category == "normal":
                        summary["used_normal"] += value
                        summary["used_after_normal"] = after
                    else:
                        summary["used_reserve"] += value
                        summary["used_after_reserve"] = after
                    summary["charged_events"] += 1
                elif kind == "refused":
                    summary["refusals"] += 1
            if valid:
                summary["total_ledger_events"] = len(unit_ledger)
                summary["guard_events"] = len(guard_events)
                if summary["used_normal"] != guard.get("used_normal"):
                    valid = False
                    problems.append("normal n_sum does not match guard used_normal")
                if summary["used_reserve"] != guard.get("used_reserve"):
                    valid = False
                    problems.append("reserve n_sum does not match guard used_reserve")
                if summary["refusals"] != guard.get("refusals"):
                    valid = False
                    problems.append("ledger refusals do not match guard refusals")
                if summary["charged_events"] + summary["refusals"] != summary["guard_events"]:
                    valid = False
                    problems.append("ledger charged-event count does not match guard event count")
        if valid:
            summary["schema_ok"] = True
            summary["guard_ledger_match"] = True
        else:
            summary["used_normal"] = "unknown"
            summary["used_reserve"] = "unknown"
            summary["refusals"] = "unknown"
            summary["charged_events"] = "unknown"
        summary["problems"] = problems
        per_unit[unit] = summary
        if not valid:
            mismatches.append({"unit": unit, "problems": problems})
    return {"root": str(runtime_dir), "files": files,
            "total_ledger_events": len(events),
            "unparsed_total_ledger_lines": unparsed,
            "per_unit_last_known": per_unit,
            "guard_ledger_reconciliation": {"ok": not mismatches,
                                            "mismatches": mismatches,
                                            "units": sorted(per_unit)},
            "frozen_snapshot_match": (files == frozen) if frozen is not None else None,
            "note": ("read-only; n/category/used_after are reconciled with each guard top-level "
                     "cumulative and its event list; unknown or mismatch blocks use")}


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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
    """A request id runs at most once, in ANY attempt directory. Read-only; returns a reason or None.

    The search is global across the whole attempts base (every per-attempt subdirectory) plus the
    preserved raw/s1, so a request id cannot be re-launched by moving to a different directory. The
    first failed attempt id is refused by name in every directory."""
    if request_id == FIRST_ATTEMPT_REQUEST_ID:
        return ("request id %s is the first failed attempt; its O_EXCL marker forbids a second "
                "launch in any directory" % request_id)
    candidates = []
    attempts_base = here / OUTPUT_DIR_NAME
    if attempts_base.exists():
        candidates.extend(sorted(attempts_base.rglob("r75_s1_outer_v2_attempt_marker_*.json")))
    historical = here / FIRST_ATTEMPT_DIR
    if historical.exists():
        candidates.extend(sorted(historical.glob("r75_s1_outer_v2_attempt_marker_*.json")))
    for path in candidates:
        try:
            recorded = json.loads(path.read_text(encoding="utf-8")).get("request_id")
        except (ValueError, OSError):
            continue
        if recorded == request_id:
            return ("an attempt marker for request id %s already exists at %s"
                    % (request_id, path))
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
    baseline_contract = plan["s1g_execution_contract"]["ledger_baseline"]
    snapshot = ledger_snapshot(runtime_dir, None)
    if snapshot["total_ledger_events"] != baseline_contract["total_ledger_events"]:
        problems.append("canonical ledger has %d events, baseline %d expected"
                        % (snapshot["total_ledger_events"],
                           baseline_contract["total_ledger_events"]))
    if snapshot["total_ledger_sha256"] != baseline_contract["total_ledger_sha256"]:
        problems.append("canonical ledger SHA256 does not match the S1G baseline")
    for unit, want in sorted(baseline_contract["guards"].items()):
        got = snapshot["per_unit_last_known"].get(unit)
        if not isinstance(got, dict):
            problems.append("missing baseline counter for %s" % unit)
            continue
        for field in ("used_normal", "used_reserve", "refusals", "charged_events"):
            if got.get(field) != want.get(field):
                problems.append(
                    "baseline counter mismatch %s.%s: got %r, want %r"
                    % (unit, field, got.get(field), want.get(field)))
    if snapshot["guard_ledger_reconciliation"]["ok"] is not True:
        problems.append("guard/ledger reconciliation failed: "
                        + json.dumps(snapshot["guard_ledger_reconciliation"]["mismatches"],
                                     sort_keys=True))
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
                   "attempt_dir": str(s1_dir),
                   "created_at": now(), "wrapper_pid": os.getpid(), "head": head,
                   "parent": parent, "argv": list(argv),
                   "license_state": "issued", "consumed_at": None, "consumed_by_pid": None,
                   "note": ("created with O_EXCL before any paid Python started; it is the ONE "
                            "parent->child launch license. The child consumes it exactly once by "
                            "creating an O_EXCL consumption record; its existence also forbids a "
                            "second launch of this request id")}, fh, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    return marker, marker


def build_child_argv(runner_path, flags):
    """Assemble the ONE real child argument vector, exactly once.

    The interpreter and the script path appear a single time as the first two entries; the caller
    passes only the runner flags. launch_once() execs the returned vector verbatim and never
    re-prefixes it, so a real command can never carry python/runner twice."""
    return [sys.executable, str(runner_path)] + list(flags)


def launch_once(repo, runner_path, argv, timeout_s):
    """Run the frozen runner exactly once and return the child facts. No retry, no fallback.

    ``argv`` is the COMPLETE vector assembled once by build_child_argv(); its first two entries are
    already the interpreter and this runner path, so it is exec'd verbatim. ``runner_path`` is
    cross-checked against argv and is NEVER prepended a second time."""
    argv = list(argv)
    if argv[1:2] != [str(runner_path)]:
        raise ValueError("launch_once received an argv whose script entry is not the runner: %r"
                         % (argv,))
    started = now()
    proc = subprocess.Popen(argv, cwd=str(repo),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        out, err = proc.communicate()
    if not isinstance(proc.returncode, int) or isinstance(proc.returncode, bool):
        raise RuntimeError("launch_once failed closed: the child did not produce an integer "
                           "returncode; it is never recorded as success")
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
    child_returncode = child.get("child_returncode")
    timed_out = child.get("timed_out")
    child_returncode_valid = (isinstance(child_returncode, int)
                              and not isinstance(child_returncode, bool))
    timed_out_valid = isinstance(timed_out, bool)
    if not (child_returncode_valid and timed_out_valid):
        payload_returncode = "MISSING_OR_INVALID"
        wrapper_return = 5
    else:
        payload_returncode = child_returncode
        wrapper_return = 0 if (child_returncode == 0 and not timed_out) else 5
    payload = {"kind": "round75-s1g-outer-outcome/1", "request_id": request_id,
               "runner_launched": True, "timed_out": timed_out,
               "started_at": child["started_at"], "ended_at": child["ended_at"],
               "wall_seconds": child["wall_seconds"], "child_pid": child["child_pid"],
               "child_returncode": payload_returncode,
               "wrapper_return": wrapper_return,
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
               {"kind": "round75-s1g-outer-wrapper-failure/1", "request_id": request_id,
                "failed_at": now(), "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(),
                "note": ("a WRAPPER-side exception only; it is not a runner exit code, not a "
                         "second attempt and not an additional failure of the child")})


def prepare(repo, here, s1_dir, request_id, plan, runtime_dir):
    """Verify the pre-charge identities and write the prepared record. Starts nothing."""
    record = verify_pre_charge(repo, here, runtime_dir, plan)
    wrapper_sha = sha256_file(__file__)
    payload = {"kind": "round75-s1g-outer-prepared/1", "request_id": request_id,
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
                   {"kind": "round75-s1g-outer-refusal/1", "request_id": request_id,
                    "refused_at": now(), "reason": "pre-charge verification failed",
                    "problems": record["problems"], "runner_launched": False,
                    "note": "nothing paid was started and no attempt marker was created"})
        return 3, record["problems"]
    refusal = duplicate_request_refusal(request_id, s1_dir, here)
    if refusal is not None:
        write_json(s1_dir / WRAPPER_FAILURE_RECORD,
                   {"kind": "round75-s1g-outer-refusal/1", "request_id": request_id,
                    "refused_at": now(), "reason": "duplicate request id",
                    "problems": [refusal], "runner_launched": False,
                    "note": ("the frozen runner was NOT launched; a request id runs at most once "
                             "and the first attempt id can never run again")})
        return 4, [refusal]
    argv = build_child_argv(here / RUNNER_NAME,
                            RUNNER_ARGV + ["--request-id", request_id,
                                           "--attempt-dir", str(s1_dir)])
    marker, marker_path = register_attempt(s1_dir, request_id, argv, record["head"],
                                           record["parent"], wrapper_sha)
    if marker is None:
        write_json(s1_dir / WRAPPER_FAILURE_RECORD,
                   {"kind": "round75-s1g-outer-refusal/1", "request_id": request_id,
                    "refused_at": now(),
                    "reason": "duplicate request: the attempt marker %s already exists" % marker_path.name,
                    "problems": ["an attempt marker for this request id already exists"],
                    "runner_launched": False,
                    "note": "the frozen runner was NOT launched; no retry is permitted here"})
        return 4, ["duplicate attempt marker: %s" % marker_path.name]
    write_json(s1_dir / PREPARED_RECORD,
               {"kind": "round75-s1g-outer-prepared/1", "request_id": request_id,
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


def _resolve_attempt_dir(here, request_id, attempt_dir):
    """Resolve this attempt's private directory.

    Default: <attempts base>/<request-id> so distinct request ids never share a directory and never
    overwrite each other's originals. An explicit --attempt-dir is honoured only when it lives inside
    the output root and is not the historical raw/s1 or anything under it. Returns None to signal a
    refusal; the caller records it and returns 2."""
    if attempt_dir:
        target = Path(attempt_dir)
        if not target.is_absolute():
            target = here / target
        target = Path(os.path.normpath(str(target)))
        historical = Path(os.path.normpath(str(here / FIRST_ATTEMPT_DIR)))
        if target == historical or historical in target.parents:
            return None
        if target != here and here not in target.parents:
            return None
        return target
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in request_id)
    return here / OUTPUT_DIR_NAME / safe


def main(argv=None):
    parser = argparse.ArgumentParser(description="ROUND75 S1 outer execution record, revised")
    parser.add_argument("--launch", action="store_true",
                        help="register the ONE attempt marker and run the frozen runner once")
    parser.add_argument("--prepare", action="store_true",
                        help="verify and record only; creates no marker and starts no child")
    parser.add_argument("--request-id", default=None,
                        help="the request id this attempt belongs to")
    parser.add_argument("--attempt-dir", default=None,
                        help="optional explicit attempt directory inside the output root; defaults "
                             "to <attempts base>/<request-id> so each attempt is isolated")
    args = parser.parse_args(argv)
    if args.launch and not args.request_id:
        sys.stderr.write("S1 OUTER V2 REFUSED: --launch requires an explicit --request-id\n")
        return 2
    if not args.launch and args.request_id is None:
        args.request_id = FOLLOW_UP_REQUEST_ID
    here = Path(__file__).resolve().parent
    repo = here
    while not (repo / ".git").exists():
        repo = repo.parent
    runtime_dir = here / "raw" / "runtime"
    plan = json.loads((here / "ROUND75_S1G_AR1_RUN_PLAN.json").read_text(encoding="utf-8"))
    s1_dir = _resolve_attempt_dir(here, args.request_id, args.attempt_dir)
    if s1_dir is None:
        sys.stderr.write("S1 OUTER V2 REFUSED: --attempt-dir must live inside the output root and "
                         "must not be the historical raw/s1 or anything under it\n")
        return 2
    s1_dir.mkdir(parents=True, exist_ok=True)
    if args.launch:
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
    print(json.dumps({"kind": "round75-s1g-outer-prepared-summary/1",
                      "request_id": args.request_id, "head": payload["head"],
                      "parent": payload["parent"], "launched": False,
                      "attempt_marker_created": False,
                      "code_sha256": payload["code_sha256"],
                      "ledger_baseline_events": payload["ledger_zero_start"]["total_ledger_events"],
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
            _crash_id = _request_id_from_argv(sys.argv[1:])
            record_wrapper_failure(_here / OUTPUT_DIR_NAME / _crash_id, _crash_id, _error)
        except Exception:
            pass
        sys.stderr.write("S1 OUTER V2 WRAPPER FAILED: %s: %s\n" % (type(_error).__name__, _error))
        raise
    raise SystemExit(_EXIT_CODE)
