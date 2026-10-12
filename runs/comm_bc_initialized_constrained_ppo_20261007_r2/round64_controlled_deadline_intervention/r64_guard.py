"""Small persistent ROUND64 attempt guard; slots are consumed before any builder call."""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class R64GuardError(RuntimeError):
    pass


class R64Guard:
    TASK_ID = "ROUND64"
    EXECUTION_VERSION = "round64-controlled-deadline-intervention/4"
    ACCEPTED_EXECUTION_VERSIONS = {
        "round64-controlled-deadline-intervention/2",
        "round64-controlled-deadline-intervention/3",
        "round64-controlled-deadline-intervention/4",
        "round64-controlled-deadline-intervention/5",
        "round64-controlled-deadline-intervention/6",
    }
    NORMAL_CAP = 10
    RESERVE_CAP = 2
    HARD_CAP = 12

    def __init__(self, root: Path, *, allow_build=True):
        self.root = Path(root).resolve()
        self.allow_build = bool(allow_build)
        self.slots = self.root / "guard_slots"
        self.ledger = self.root / "ROUND64_LEDGER.jsonl"
        self.slots.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._local = threading.local()

    def _slot_files(self):
        return sorted(self.slots.glob("*.slot"))

    def rows(self):
        if not self.ledger.exists():
            return []
        return [json.loads(line) for line in self.ledger.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def _slot_payloads(self):
        return [json.loads(path.read_text(encoding="utf-8")) for path in self._slot_files()]

    def snapshot(self):
        files = self._slot_payloads()
        rows = self.rows()
        consumed_rows = [row for row in rows if row["event"] == "slot_consumed"]
        if any(row.get("task") != self.TASK_ID or
               row.get("execution_version") not in self.ACCEPTED_EXECUTION_VERSIONS for row in rows):
            raise R64GuardError("R64 ledger contains a foreign task or execution version")
        if any(value.get("task") != self.TASK_ID or
               value.get("execution_version") not in self.ACCEPTED_EXECUTION_VERSIONS for value in files):
            raise R64GuardError("R64 slot contains a foreign task or execution version")
        if len(consumed_rows) != len(files) or {row["slot"] for row in consumed_rows} != {
                value["slot"] for value in files}:
            raise R64GuardError("slot files and append-only consumption ledger disagree; fail closed")
        if len(files) > self.HARD_CAP:
            raise R64GuardError("persisted R64 attempt count exceeds hard cap")
        count = lambda event: sum(row["event"] == event for row in rows)
        return {
            "slots_consumed": len(files), "normal": sum(x["category"] == "normal" for x in files),
            "reserve": sum(x["category"] == "reserve" for x in files),
            "hard_cap": self.HARD_CAP, "headroom": self.HARD_CAP - len(files),
            "builder_calls": count("builder_called"), "builder_returns": count("env_constructed"),
            "run_calls": count("run_called"), "run_returns": count("run_returned"),
            "run_failures": count("run_failed"),
            "steps_total": sum(int(x.get("steps", 0)) for x in rows if x["event"] == "run_returned"),
            "interrupted": count("interrupted"), "qualified": count("result_qualified"),
            "disqualified": count("result_disqualified"),
        }

    def _append(self, slot: str, logical: str, event: str, **details):
        record = {"task": self.TASK_ID, "execution_version": self.EXECUTION_VERSION,
                  "slot": slot, "logical": logical, "event": event,
                  "unix": time.time(), **details}
        with self.ledger.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _logical_events(self, logical):
        return [row for row in self.rows() if row.get("logical") == logical]

    def terminal(self, logical):
        terminal = ("result_qualified", "result_disqualified", "interrupted")
        return next((row["event"] for row in reversed(self._logical_events(logical))
                     if row["event"] in terminal), None)

    def _claim(self, logical, category, retry=False):
        if not logical or category not in ("normal", "reserve"):
            raise R64GuardError("invalid logical job or category")
        with self._lock:
            self.snapshot()
            current = self._slot_payloads()
            events = self._logical_events(logical)
            if category == "normal":
                if any(x["logical"] == logical for x in current):
                    raise R64GuardError(f"normal logical already consumed: {logical}")
                if sum(x["category"] == "normal" for x in current) >= self.NORMAL_CAP:
                    raise R64GuardError("normal cap reached")
            else:
                terminal = next((x["event"] for x in reversed(events)
                                 if x["event"] in ("result_qualified", "result_disqualified", "interrupted")), None)
                if not retry or terminal != "interrupted":
                    raise R64GuardError("reserve requires explicit retry of an interrupted attempt")
                if any(x["logical"] == logical and x["event"] == "result_qualified" for x in events):
                    raise R64GuardError("cannot retry a qualified logical")
                if sum(x["category"] == "reserve" for x in current) >= self.RESERVE_CAP:
                    raise R64GuardError("reserve cap reached")
            if len(current) >= self.HARD_CAP:
                raise R64GuardError("hard cap reached")

            limit = self.NORMAL_CAP if category == "normal" else self.RESERVE_CAP
            prefix = category + "-"
            for ordinal in range(1, limit + 1):
                slot = f"{prefix}{ordinal:02d}"
                path = self.slots / f"{slot}.slot"
                try:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue
                payload = {"task": self.TASK_ID, "execution_version": self.EXECUTION_VERSION,
                           "slot": slot, "logical": logical, "category": category,
                           "retry": bool(retry), "created_unix": time.time()}
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self._append(slot, logical, "slot_consumed", category=category, retry=bool(retry))
                return slot
            raise R64GuardError(f"no {category} slot available")

    @contextmanager
    def attempt(self, logical, *, category="normal", retry=False, delta=0, arm="baseline"):
        if getattr(self._local, "active", None) is not None:
            raise R64GuardError("nested R64 attempts are forbidden")
        slot = self._claim(logical, category, retry)
        attempt = R64Attempt(self, slot, logical, int(delta), str(arm))
        self._local.active = attempt
        try:
            yield attempt
        finally:
            self._local.active = None
            if not attempt.closed:
                attempt.event("interrupted")
                attempt.closed = True

    def active(self):
        return getattr(self._local, "active", None)

    def require_active(self):
        active = self.active()
        if active is None:
            raise R64GuardError("unregistered R64 environment/run path refused before builder")
        return active


class R64Attempt:
    def __init__(self, guard, slot, logical, delta, arm):
        self.guard = guard
        self.slot = slot
        self.logical = logical
        self.delta = delta
        self.arm = arm
        self.closed = False
        self.builder_started = False
        self.env_constructed = False
        self.run_started = False
        self.deadline_audit = {}
        self.profile_records = []
        self.registry_records = []
        self.task_env_records = []
        self.message_records = []
        self.live_run = None

    def event(self, name, **details):
        self.guard._append(self.slot, self.logical, name, delta=self.delta, arm=self.arm, **details)

    def builder_called(self):
        if self.builder_started:
            raise R64GuardError("builder called twice within one attempt")
        self.builder_started = True
        self.event("builder_called")

    def env_constructed_now(self):
        if not self.builder_started or self.env_constructed:
            raise R64GuardError("invalid builder-return accounting")
        self.env_constructed = True
        self.event("env_constructed")

    def run_called(self):
        if not self.env_constructed or self.run_started:
            raise R64GuardError("run call outside one constructed R64 episode")
        self.run_started = True
        self.event("run_called")

    def run_returned(self, steps):
        self.event("run_returned", steps=int(steps), ok=True)

    def run_failed(self, steps, exception):
        self.event("run_failed", steps=int(steps), exception=str(exception))

    def qualify(self, ok, detail):
        self.event("result_qualified" if ok else "result_disqualified", detail=detail)

    def finish(self):
        self.closed = True
        self.event("finished")
