"""Safe recovery_05 cutover: consumed claims, donor restart plan, and root phase lock.

This module is pure-file/pure-data. It does not load a checkpoint payload or model, create a
World, draw random numbers, or start a training phase.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

R5_ROOT = Path("D:/code/整合/zhenghe-round42/runs/stage5b_pure_comm_convergence_20261027/recovery_05/revision_01")
CUTOVER_RELATIVE = "protocol/RECOVERY_05_CONSUMED_NOT_TRAINABLE.json"
CUTOVER_SCHEMA = "stage5b-recovery-05-consumed-not-trainable/1"
CUTOVER_STATUS = "CONSUMED_NOT_TRAINABLE_RETAINED"
BRANCH_MODE = "DONOR_RNG_RESTART_NEW_BRANCH_NOT_EXACT_CONTINUATION"
ROOT_PHASE_LOCK_RELATIVE = "execution/ledger/RECOVERY_ROOT_PHASE.lock"
CUTOVER_RUNS = {
    ("T1", 91001): (1800, 1804, 1924),
    ("T1", 91002): (1800, 1800, 1920),
    ("T1", 91003): (1440, 1440, 1560),
    ("T2", 91001): (1440, 1440, 1560),
    ("T2", 91002): (1440, 1440, 1560),
    ("T2", 91003): (1080, 1080, 1200),
}
CUTOVER_GAPS = {
    ("T1", 91001): {"episodes": (1800, 1804), "updates": (450, 451)},
}
CHECK_IMPACTS = [2040, 2520, 3000]
FIRST_EVALUABLE_AFTER_CUTOVER = 3480


class CutoverError(ValueError):
    pass


def need(ok, message):
    if not ok:
        raise CutoverError(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": sha256_bytes(raw)}


def _read_json(path):
    path = Path(path).resolve()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise CutoverError("invalid JSON %s: %s" % (path, error)) from error


def _execution_files(root):
    execution = root / "execution"
    need(execution.is_dir(), "recovery_05 execution evidence is missing")
    return sorted((p for p in execution.rglob("*") if p.is_file()), key=lambda p: str(p))


def _claim_records(root):
    claims_path = root / "execution/ledger/recovery_claims"
    paths = sorted(claims_path.glob("*.claim")) if claims_path.is_dir() else []
    need(len(paths) == 3, "recovery_05 must retain exactly three consumed claims")
    records = []
    for path in paths:
        body = _read_json(path)
        need(body.get("schema") == "stage5b-recovery-runtime-claim/1",
             "unknown recovery_05 claim schema")
        need(body.get("root") == str(root), "recovery_05 claim root mismatch")
        slot = body.get("slot")
        runtime = body.get("runtime_key")
        need(isinstance(slot, str) and isinstance(runtime, dict),
             "recovery_05 claim lacks slot/runtime_key")
        need(runtime.get("arm") == "T1" and runtime.get("training_seed") == 91001
             and runtime.get("episode_index") in (1800, 1801, 1802)
             and runtime.get("scene_seed") == 2201901 + runtime["episode_index"] - 1800,
             "recovery_05 consumed claim is outside the registered three-world prefix")
        records.append({"slot": slot, "runtime_key": runtime, "claim": pin(path)})
    need(sorted(x["runtime_key"]["episode_index"] for x in records) == [1800, 1801, 1802],
         "recovery_05 consumed episode set changed")
    return records


def _world_records(root):
    world = root / "execution/world_logs"
    dirs = sorted(p for p in world.iterdir() if p.is_dir()) if world.is_dir() else []
    expected = ["train__T1__91001__1800", "train__T1__91001__1801",
                "train__T1__91001__1802"]
    need([p.name for p in dirs] == expected,
         "recovery_05 world log set is not the consumed three-world prefix")
    records = []
    for directory in dirs:
        lifecycle = list(directory.rglob("lifecycle.jsonl.gz"))
        provenance = list(directory.rglob("log_provenance.json"))
        relay = list(directory.rglob("relay_transcript.jsonl.gz"))
        need(len(lifecycle) == len(provenance) == len(relay) == 1,
             "recovery_05 world log trio is incomplete")
        records.append({"run_directory": directory.name,
                        "lifecycle": pin(lifecycle[0]),
                        "provenance": pin(provenance[0]),
                        "relay": pin(relay[0])})
    return records


def _assert_empty_results(root):
    paths = {
        "results": root / "execution/ledger/recovery_results",
        "rows": root / "execution/rows/train",
        "journals": root / "execution/recovery/update_journal",
        "checkpoints": root / "execution/checkpoints",
        "certificates": root / "execution/recovery/certificates",
    }
    for label, path in paths.items():
        present = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else []
        need(not present, "recovery_05 unexpectedly contains trainable %s output" % label)


def build_consumed_manifest(root=R5_ROOT):
    """Build a deterministic pin set from the observed failed recovery_05 files."""
    root = Path(root).resolve()
    need(root == R5_ROOT.resolve(), "cutover admission only accepts the registered recovery_05 root")
    stop_path = root / "execution/ledger/GLOBAL_STOP.json"
    stop = _read_json(stop_path)
    need(stop.get("schema") == "stage5b-recovery-global-stop/1"
         and stop.get("status") == "TERMINAL_UNRECOVERABLE",
         "recovery_05 has no terminal unrecoverable stop marker")
    _assert_empty_results(root)
    claims = _claim_records(root)
    worlds = _world_records(root)
    files = [pin(path) for path in _execution_files(root)]
    return {
        "schema": CUTOVER_SCHEMA,
        "status": CUTOVER_STATUS,
        "source_root": str(root),
        "source_context_sha256": pin(root / "protocol/RECOVERY_RUNTIME_CONTEXT.json")["sha256"],
        "global_stop": pin(stop_path),
        "consumed_claims": claims,
        "world_logs": worlds,
        "files": files,
        "empty_trainable_outputs": {
            "results": 0, "rows": 0, "update_journals": 0,
            "checkpoints": 0, "certificates": 0,
        },
        "claim_disposition": "CONSUMED_NOT_TRAINABLE_NO_RETRY_NO_REPLACEMENT_SEED",
        "donor_episode": 1800,
        "resume_episode": 1804,
        "unavailable_history": {
            "episodes": [1800, 1804], "updates": [450, 451],
            "consequence": "NO_SYNTHETIC_ROWS_OR_UPDATE_DIAGNOSTICS",
        },
        "branch_mode": BRANCH_MODE,
        "checks_not_evaluable": CHECK_IMPACTS,
        "first_evaluable_check": FIRST_EVALUABLE_AFTER_CUTOVER,
        "note": ("All recovery_05 execution files and its three claims are retained read-only. "
                 "The new branch restores the 1800 donor RNG state and starts at 1804; it is "
                 "not exact continuation after the three failed Worlds."),
    }


def validate_consumed_manifest(body, root=R5_ROOT):
    root = Path(root).resolve()
    need(root == R5_ROOT.resolve(), "cutover evidence points at a noncanonical recovery_05 root")
    need(body.get("schema") == CUTOVER_SCHEMA and body.get("status") == CUTOVER_STATUS,
         "cutover evidence schema/status mismatch")
    need(body.get("source_root") == str(root), "cutover source root mismatch")
    need(body.get("claim_disposition") ==
         "CONSUMED_NOT_TRAINABLE_NO_RETRY_NO_REPLACEMENT_SEED",
         "cutover claim disposition changed")
    need(body.get("branch_mode") == BRANCH_MODE
         and body.get("donor_episode") == 1800 and body.get("resume_episode") == 1804,
         "cutover donor/restart branch semantics changed")
    need(body.get("unavailable_history", {}).get("episodes") == [1800, 1804]
         and body.get("unavailable_history", {}).get("updates") == [450, 451],
         "cutover history gap changed")
    need(body.get("checks_not_evaluable") == CHECK_IMPACTS
         and body.get("first_evaluable_check") == FIRST_EVALUABLE_AFTER_CUTOVER,
         "cutover check qualification changed")
    expected_files = [pin(path) for path in _execution_files(root)]
    files = body.get("files")
    need(isinstance(files, list) and files == expected_files,
         "recovery_05 execution evidence has additions/omissions/drift")
    need(body.get("consumed_claims") == _claim_records(root),
         "recovery_05 consumed claim pins changed")
    need(body.get("world_logs") == _world_records(root),
         "recovery_05 world-log pins changed")
    _assert_empty_results(root)
    return frozenset((x["runtime_key"]["arm"], x["runtime_key"]["training_seed"],
                      x["runtime_key"]["scene_seed"], x["runtime_key"]["action_rng_seed"])
                     for x in body["consumed_claims"])


def load_consumed_manifest(path):
    body = _read_json(path)
    return validate_consumed_manifest(body, R5_ROOT)


def apply_donor_restart_runs(runs):
    """Transform a completed-parent plan into the explicit recovery_06 donor restart plan."""
    need(type(runs) is list and len(runs) == 6, "cutover requires the exact six-run roster")
    result = []
    for item in runs:
        out = dict(item)
        key = (out.get("arm"), out.get("training_seed"))
        need(key in CUTOVER_RUNS, "cutover received an unregistered run")
        donor_episode, start, end = CUTOVER_RUNS[key]
        donor = out.get("donor")
        need(isinstance(donor, dict), "cutover run lacks its donor pin")
        observed = donor.get("observed_episode", donor.get("cursor_from_closed_journal"))
        need(observed == donor_episode,
             "completed-parent donor episode differs from the cutover table")
        need(donor.get("next_update") == donor_episode // 4,
             "donor next_update differs from its cursor")
        need(out.get("active") is True and out.get("new_world_cap") == 120,
             "cutover only accepts six active 120-world donor continuations")
        out.update({
            "start_episode": start,
            "end_episode_cap": end,
            "new_world_cap": 120,
            "next_update": start // 4,
            "donor_episode": donor_episode,
            "donor_next_update": donor_episode // 4,
            "branch_mode": BRANCH_MODE,
        })
        if key in CUTOVER_GAPS:
            source_gap = CUTOVER_GAPS[key]
            out["known_history_gap"] = {
                "episodes": list(source_gap["episodes"]),
                "updates": list(source_gap["updates"]),
            }
        result.append(out)
    need([(x["arm"], x["training_seed"]) for x in result] == list(CUTOVER_RUNS),
         "cutover changed the registered arm/seed order")
    need(sum(x["new_world_cap"] for x in result) == 720,
         "cutover six-run world cap is not exactly 720")
    return result


_HELD_LOCKS = set()

class RootPhaseLock:
    """Fail-fast one-owner lock for recovery G3/train under one root."""

    def __init__(self, root):
        self.path = Path(root).resolve() / ROOT_PHASE_LOCK_RELATIVE
        self.handle = None
        self.held = False

    def acquire(self):
        lock_key = str(self.path).casefold()
        if self.held or lock_key in _HELD_LOCKS:
            raise CutoverError("root phase lock is already held in this process")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise CutoverError("another recovery phase already holds the root lock: %s"
                               % self.path) from error
        self.handle, self.held = handle, True
        _HELD_LOCKS.add(lock_key)
        return self

    def release(self):
        if not self.held:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None
            self.held = False
            _HELD_LOCKS.discard(str(self.path).casefold())

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False
