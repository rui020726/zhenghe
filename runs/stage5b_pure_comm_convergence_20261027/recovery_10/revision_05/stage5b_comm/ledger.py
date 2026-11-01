"""Irreversible world-slot ledger.

Every real world must claim a slot before the environment is constructed. Claims and results
are exclusive creates, so a crashed or terminated phase leaves the slot consumed. The plan is
not trusted by length: every entry must equal constants.slot_plan() exactly in order, and a
duplicate claim or a second result is an error rather than a silent overwrite.

The pid/time/status fields recorded here are in-process facts about the worker that ran the
world. They are explicitly not a per-world operating-system exit code.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from . import constants as C

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")

STOP_NAME = "GLOBAL_STOP.json"
GATE_NAME = "GATE_FIRST_UPDATE.json"
LOCK_NAME = "LEDGER.lock"


class CrossProcessLock:
    """A real OS file lock shared by the parent and every spawned worker.

    Windows uses msvcrt byte-range locking on a one-byte lock file; other platforms fall back
    to fcntl.flock. Every process opens the same path, so the lock scope spans processes and
    threads, not just one interpreter.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"L")
            self.handle.flush()
        deadline = time.time() + 600.0
        while True:
            try:
                self.handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.time() > deadline:
                    self.handle.close()
                    self.handle = None
                    raise RuntimeError("timed out waiting for the ledger lock %s" % (self.path,))
                time.sleep(0.01)

    def __exit__(self, exc_type, exc, tb):
        if self.handle is None:
            return False
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
        return False


def _safe(slot):
    return _SAFE.sub("_", str(slot))


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_plan(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    slots = payload.get("slots")
    expected = C.slot_plan()
    if not isinstance(slots, list):
        raise RuntimeError("slot plan has no slots list")
    if int(payload.get("world_cap", -1)) != C.WORLD_CAP_TOTAL:
        raise RuntimeError("slot plan world_cap %r != %d"
                           % (payload.get("world_cap"), C.WORLD_CAP_TOTAL))
    if len(slots) != len(expected):
        raise RuntimeError("slot plan holds %d slots, expected %d"
                           % (len(slots), len(expected)))
    for position, (actual, wanted) in enumerate(zip(slots, expected)):
        if actual != wanted:
            raise RuntimeError("slot plan entry %d is %r, expected %r"
                               % (position, actual, wanted))
    ids = [item["slot"] for item in slots]
    if len(set(ids)) != len(ids):
        raise RuntimeError("slot plan contains duplicate slot ids")
    return slots


class SlotLedger:
    """One claim file and one result file per slot, both exclusive."""

    def __init__(self, layout):
        self.layout = layout
        self.claims = Path(layout.ledger) / "claims"
        self.results = Path(layout.ledger) / "results"
        self.plan_path = layout.slot_plan
        self._slots = None
        self._map = None

    def slots(self):
        if self._slots is None:
            self._slots = _load_plan(self.plan_path)
        return self._slots

    def ensure(self):
        self.claims.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)

    def entry(self, slot):
        if self._map is None:
            self._map = {item["slot"]: item for item in self.slots()}
        try:
            return self._map[slot]
        except KeyError:
            raise KeyError("slot %r is not in the frozen plan" % (slot,)) from None

    def claimed(self, slot):
        return (self.claims / (_safe(slot) + ".claim")).exists()

    def finished(self, slot):
        return (self.results / (_safe(slot) + ".json")).exists()

    def claim(self, slot, extra=None):
        """Consume one slot. Returns False when it was already consumed."""
        entry = self.entry(slot)
        self.ensure()
        path = self.claims / (_safe(slot) + ".claim")
        payload = {"schema": C.LEDGER_SCHEMA, "slot": slot, "entry": entry,
                   "claimed_utc": _utc(), "worker_pid": os.getpid(),
                   "worker_ppid": os.getppid(), "extra": extra or {},
                   "status_note": "in-process claim; this is not a per-world OS exit code"}
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        except FileExistsError:
            return False
        return True

    def finish(self, slot, status, payload):
        if status not in ("ok", "failed", "terminated"):
            raise ValueError("unknown slot status %r" % (status,))
        self.ensure()
        path = self.results / (_safe(slot) + ".json")
        body = {"schema": C.LEDGER_SCHEMA, "slot": slot, "status": status,
                "payload": payload, "worker_pid": os.getpid(), "ended_utc": _utc(),
                "status_note": ("in-process result; the phase-level OS exit code comes from "
                                "the runner's external receipt")}
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(body, ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def accounting(self, phase=None):
        """Close one phase over every registered slot, including unstarted slots.

        A slot with a result but no claim is a ledger error.  A slot that was submitted but
        never claimed has no on-disk entry in this design; it is reported as NOT_CLAIMED with
        submission_state UNKNOWN_NOT_RECORDED rather than being invented as cancelled.
        """
        entries = [item for item in self.slots()
                   if phase is None or item.get("phase") == phase]
        accepted = []
        failed = []
        claimed_incomplete = []
        not_claimed = []
        for item in entries:
            slot = item["slot"]
            claimed = self.claimed(slot)
            result_path = self.results / (_safe(slot) + ".json")
            if result_path.is_file():
                if not claimed:
                    raise RuntimeError("slot %s has a result but no claim" % (slot,))
                status = json.loads(result_path.read_text(encoding="utf-8")).get("status")
                if status == "ok":
                    accepted.append(slot)
                elif status in ("failed", "terminated"):
                    failed.append(slot)
                else:
                    raise RuntimeError("slot %s has unknown result status %r" % (slot, status))
            elif claimed:
                claimed_incomplete.append(slot)
            else:
                not_claimed.append(slot)
        closed = len(accepted) + len(failed) + len(claimed_incomplete) + len(not_claimed)
        if closed != len(entries):
            raise RuntimeError("slot accounting does not close: %d != %d"
                               % (closed, len(entries)))
        return {
            "schema": "stage5b-slot-accounting/1",
            "phase": phase,
            "registered": len(entries),
            "accepted": accepted,
            "failed": failed,
            "claimed_incomplete": claimed_incomplete,
            "not_claimed": not_claimed,
            "counts": {"accepted": len(accepted), "failed": len(failed),
                       "claimed_incomplete": len(claimed_incomplete),
                       "not_claimed": len(not_claimed), "registered": len(entries)},
            "submission_state": ("NOT_CLAIMED entries are UNKNOWN_NOT_RECORDED; no cancel "
                                 "event is fabricated for them"),
        }

    def counts(self):
        self.ensure()
        claims = len(list(self.claims.glob("*.claim")))
        results = len(list(self.results.glob("*.json")))
        by_status = {"ok": 0, "failed": 0, "terminated": 0}
        for path in sorted(self.results.glob("*.json")):
            status = json.loads(path.read_text(encoding="utf-8")).get("status")
            if status in by_status:
                by_status[status] += 1
        return {"planned": len(self.slots()), "claimed": claims, "results": results,
                "by_status": by_status,
                "remaining": int(len(self.slots())) - results}


class LedgerGate:
    """One in-process lock around the stop test and the exclusive claim.

    Claim files remain exclusive creates, so two workers can never own the same slot. This
    gate additionally makes "stop requested" and "claim this slot" one atomic step inside a
    worker process, closing the thread window where a stop could be observed as absent and a
    claim written afterwards. Stop markers and claims are still separate files across
    processes; a stop written by one process can race one in-flight claim by another, and the
    protocol treats that claim as a real consumed world rather than pretending it never ran.
    """

    def __init__(self, layout):
        self.layout = layout
        self.ledger = SlotLedger(layout)
        directory = Path(layout.ledger)
        self.stop_path = directory / STOP_NAME
        self.gate_path = directory / GATE_NAME
        self.lock_path = directory / LOCK_NAME
        self.lock = CrossProcessLock(self.lock_path)
        self._lock = self.lock

    def is_stopped(self):
        return self.stop_path.exists()

    def request_stop(self, reason, detail=None):
        with self._lock:
            if self.stop_path.exists():
                return json.loads(self.stop_path.read_text(encoding="utf-8"))
            self.stop_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": "stage5b-global-stop/1",
                "state": "TERMINAL_UNRECOVERABLE",
                "reason": str(reason),
                "requested_utc": _utc(),
                "pid": os.getpid(),
                "gate_signal_present": self.gate_path.exists(),
                "note": ("after this marker no new slot may be claimed; already-claimed worlds "
                         "finish and account; nothing is retried and there is no resume"),
            }
            if detail:
                # Evidence of the world that triggered the stop. Optional, so an ordinary stop
                # marker keeps exactly the keys earlier rounds recorded.
                payload["detail"] = detail
            try:
                with self.stop_path.open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False,
                                            sort_keys=True) + "\n")
            except FileExistsError:
                pass
            return payload

    def claim(self, slot, extra=None):
        """Atomic stop-check + exclusive claim: claimed / stopped / consumed."""
        with self._lock:
            if self.stop_path.exists():
                return "stopped"
            if not self.ledger.claim(slot, extra):
                return "consumed"
            return "claimed"

    def finish(self, slot, status, payload):
        return self.ledger.finish(slot, status, payload)
