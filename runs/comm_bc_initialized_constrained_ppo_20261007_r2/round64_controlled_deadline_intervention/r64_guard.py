"""Small persistent ROUND64 attempt guard; slots are consumed before any builder call."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from r64_identity import BINDING, HISTORICAL_EXECUTION_VERSIONS

HERE = Path(__file__).resolve().parent
RECOVERY_RECEIPT_PATH = HERE / "ROUND64_RECOVERY_RECEIPT.json"


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path):
    return _sha256_bytes(Path(path).read_bytes())


class R64GuardError(RuntimeError):
    pass


class R64Guard:
    TASK_ID = BINDING.task_id
    EXECUTION_VERSION = BINDING.execution_version
    ACCEPTED_EXECUTION_VERSIONS = HISTORICAL_EXECUTION_VERSIONS | {
        BINDING.execution_version
    }
    NORMAL_CAP = BINDING.normal_cap
    RESERVE_CAP = BINDING.reserve_cap
    HARD_CAP = BINDING.hard_cap

    def __init__(self, root: Path, *, allow_build=True):
        self.root = Path(root).resolve()
        self.allow_build = bool(allow_build)
        self.execution_version = BINDING.execution_version
        self.normal_cap = BINDING.normal_cap
        self.reserve_cap = BINDING.reserve_cap
        self.hard_cap = BINDING.hard_cap
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
        if len(files) > self.hard_cap:
            raise R64GuardError("persisted R64 attempt count exceeds hard cap")
        count = lambda event: sum(row["event"] == event for row in rows)
        return {
            "slots_consumed": len(files), "normal": sum(x["category"] == "normal" for x in files),
            "reserve": sum(x["category"] == "reserve" for x in files),
            "hard_cap": self.hard_cap, "headroom": self.hard_cap - len(files),
            "builder_calls": count("builder_called"), "builder_returns": count("env_constructed"),
            "run_calls": count("run_called"), "run_returns": count("run_returned"),
            "run_failures": count("run_failed"),
            "steps_total": sum(int(x.get("steps", 0)) for x in rows if x["event"] == "run_returned"),
            "interrupted": count("interrupted"), "qualified": count("result_qualified"),
            "disqualified": count("result_disqualified"),
        }

    def resume_plan_status(self, logicals):
        return resume_plan_status(
            self._slot_payloads(), self.rows(), logicals,
            normal_cap=self.normal_cap, reserve_cap=self.reserve_cap,
            hard_cap=self.hard_cap,
            recovery_receipts=self.verified_recovery_receipts(),
        )

    def verified_recovery_receipts(self):
        if self.root != (HERE / "raw").resolve():
            return {}
        if not RECOVERY_RECEIPT_PATH.is_file():
            return {}
        receipt = json.loads(RECOVERY_RECEIPT_PATH.read_text(encoding="utf-8"))
        if receipt.get("kind") != "round64-post-hoc-interruption-recovery-receipt/1":
            raise R64GuardError("unsupported R64 recovery receipt")
        if receipt.get("task") != self.TASK_ID:
            raise R64GuardError("R64 recovery receipt belongs to another task")
        authority = receipt.get("authority", {})
        if (authority.get("path") != BINDING.approval_path.name or
                authority.get("sha256") != BINDING.approval_sha256 or
                authority.get("qualification_review_path") != "QUALIFICATION_AND_RECOVERY_REVIEW.json"):
            raise R64GuardError("R64 recovery receipt is not bound to the approved recovery source")
        if _sha256_file(BINDING.approval_path) != BINDING.approval_sha256:
            raise R64GuardError("R64 cap approval changed after binding")
        review_path = HERE / authority["qualification_review_path"]
        if _sha256_file(review_path) != authority.get("qualification_review_sha256"):
            raise R64GuardError("R64 recovery qualification review hash mismatch")
        source = receipt.get("interrupted_attempt", {})
        logical = source.get("logical")
        if (logical != "d+30_FORCE_SEND" or source.get("slot") != "normal-07" or
                source.get("category") != "normal" or source.get("retry") is not False or
                receipt.get("classification") != "interrupted_before_observed_run_terminal"):
            raise R64GuardError("R64 recovery receipt attempts to reclassify another attempt")
        expected_sources = (
            (source.get("claim_path"), source.get("claim_sha256")),
            (source.get("slot_path"), source.get("slot_sha256")),
            (source.get("partial_lifecycle_path"), source.get("partial_lifecycle_sha256")),
        )
        for relative, expected in expected_sources:
            path = (self.root.parent / relative).resolve()
            if not path.is_relative_to(self.root.resolve().parent) or _sha256_file(path) != expected:
                raise R64GuardError(f"R64 recovery source hash mismatch: {relative}")
        ledger = self.ledger.read_bytes()
        prefix = source.get("ledger_prefix", {})
        count = int(prefix.get("line_count", -1))
        lines = ledger.splitlines(keepends=True)
        prefix_bytes = b"".join(lines[:count])
        if (len(lines) < count or len(prefix_bytes) != int(prefix.get("byte_count", -1)) or
                _sha256_bytes(prefix_bytes) != prefix.get("sha256")):
            raise R64GuardError("R64 recovery ledger-prefix identity mismatch")
        prefix_rows = [json.loads(line) for line in prefix_bytes.splitlines() if line.strip()]
        events = [row for row in prefix_rows if row.get("logical") == logical]
        if (not any(row.get("event") == "run_called" for row in events) or
                any(row.get("event") in ("run_returned", "run_failed", "interrupted",
                                           "result_qualified", "result_disqualified")
                    for row in events)):
            raise R64GuardError("R64 recovery receipt does not match the preserved nonterminal prefix")
        claim = json.loads((self.root.parent / source["claim_path"]).read_text(encoding="utf-8"))
        slot = json.loads((self.root.parent / source["slot_path"]).read_text(encoding="utf-8"))
        if (claim.get("logical") != logical or claim.get("slot") != "normal-07" or
                slot.get("logical") != logical or slot.get("slot") != "normal-07"):
            raise R64GuardError("R64 recovery receipt source identity mismatch")
        authorized = receipt.get("recovery_authorized", {})
        if (authorized.get("only_logical") != logical or
                authorized.get("only_category") != "reserve" or
                authorized.get("only_slot") != "reserve-03" or
                authorized.get("must_not_repeat_qualified_cells") is not True):
            raise R64GuardError("R64 recovery receipt grants an invalid retry scope")
        return {logical: {"receipt_sha256": _sha256_file(RECOVERY_RECEIPT_PATH),
                          "receipt": receipt}}

    def require_resumable_plan(self, logicals):
        status = self.resume_plan_status(logicals)
        if not status["can_resume"]:
            raise R64GuardError(
                "R64 fixed-plan recovery gate refused before any new slot: "
                f"unresolved={status['unresolved_logicals']}, "
                f"additional_attempts_required={status['additional_attempts_required']}, "
                f"normal_available={status['normal_available']}, "
                f"reserve_available={status['reserve_available']}, "
                f"nonterminal={status['nonterminal_logicals']}"
            )
        return status

    def _append(self, slot: str, logical: str, event: str, **details):
        record = {"task": self.TASK_ID, "execution_version": self.execution_version,
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
            receipt_retry = False
            if category == "normal":
                if any(x["logical"] == logical for x in current):
                    raise R64GuardError(f"normal logical already consumed: {logical}")
                if sum(x["category"] == "normal" for x in current) >= self.normal_cap:
                    raise R64GuardError("normal cap reached")
            else:
                terminal = next((x["event"] for x in reversed(events)
                                 if x["event"] in ("result_qualified", "result_disqualified", "interrupted")), None)
                receipt = self.verified_recovery_receipts().get(logical)
                receipt_retry = (terminal is None and receipt is not None and
                                 logical == "d+30_FORCE_SEND")
                if not retry or (terminal != "interrupted" and not receipt_retry):
                    raise R64GuardError("reserve requires explicit retry of an interrupted attempt")
                if any(x["logical"] == logical and x["event"] == "result_qualified" for x in events):
                    raise R64GuardError("cannot retry a qualified logical")
                if sum(x["category"] == "reserve" for x in current) >= self.reserve_cap:
                    raise R64GuardError("reserve cap reached")
            if len(current) >= self.hard_cap:
                raise R64GuardError("hard cap reached")

            limit = self.normal_cap if category == "normal" else self.reserve_cap
            prefix = category + "-"
            for ordinal in range(1, limit + 1):
                slot = f"{prefix}{ordinal:02d}"
                if receipt_retry and slot != "reserve-03":
                    continue
                path = self.slots / f"{slot}.slot"
                try:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue
                receipt_sha = receipt["receipt_sha256"] if category == "reserve" and receipt_retry else None
                payload = {"task": self.TASK_ID, "execution_version": self.execution_version,
                           "slot": slot, "logical": logical, "category": category,
                           "retry": bool(retry), "created_unix": time.time(),
                           "recovery_receipt_sha256": receipt_sha}
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self._append(slot, logical, "slot_consumed", category=category,
                             retry=bool(retry), recovery_receipt_sha256=receipt_sha)
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


def resume_plan_status(slot_payloads, rows, logicals, *, normal_cap, reserve_cap, hard_cap,
                       recovery_receipts=None):
    """Pure readiness calculation; it never creates a claim or consumes a slot."""
    by_logical = {}
    for row in rows:
        by_logical.setdefault(row.get("logical"), []).append(row)
    slots_by_logical = {}
    for slot in slot_payloads:
        slots_by_logical.setdefault(slot.get("logical"), []).append(slot)
    consumed_normal = sum(slot.get("category") == "normal" for slot in slot_payloads)
    consumed_reserve = sum(slot.get("category") == "reserve" for slot in slot_payloads)
    required_normal = required_reserve = 0
    nonterminal = []
    disqualified = []
    unresolved = []
    recovery_receipts = recovery_receipts or {}
    for logical in logicals:
        events = by_logical.get(logical, [])
        terminal = next((row.get("event") for row in reversed(events)
                         if row.get("event") in ("result_qualified", "result_disqualified", "interrupted")), None)
        if terminal == "result_qualified":
            continue
        unresolved.append(logical)
        if terminal == "result_disqualified":
            disqualified.append(logical)
        elif terminal == "interrupted":
            required_reserve += 1
        elif slots_by_logical.get(logical):
            if logical in recovery_receipts:
                required_reserve += 1
            else:
                nonterminal.append(logical)
        else:
            required_normal += 1
    normal_available = max(0, int(normal_cap) - consumed_normal)
    reserve_available = max(0, int(reserve_cap) - consumed_reserve)
    additional = required_normal + required_reserve + len(nonterminal)
    can_resume = (
        not disqualified and not nonterminal and
        required_normal <= normal_available and
        required_reserve <= reserve_available and
        additional <= max(0, int(hard_cap) - len(slot_payloads))
    )
    return {
        "can_resume": can_resume,
        "unresolved_logicals": unresolved,
        "nonterminal_logicals": nonterminal,
        "disqualified_logicals": disqualified,
        "normal_attempts_required": required_normal,
        "reserve_retries_required": required_reserve,
        "additional_attempts_required": additional,
        "normal_available": normal_available,
        "reserve_available": reserve_available,
        "hard_cap_headroom": max(0, int(hard_cap) - len(slot_payloads)),
    }
