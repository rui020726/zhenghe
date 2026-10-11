"""ROUND61 resource guard: the single launch entry that atomically claims and consumes a slot
BEFORE the real environment builder is called. No refund on construction failure, early
interruption, quarantine or retry. Records attempt/slot consumption, builder calls, actual
environment constructions, run calls/returns, actual steps and core-result qualification
separately, so the valid-result count is never mistaken for the actual-consumption count.

The guard is self-contained and persistent (on-disk), independent of any prior round's ledger.
Slot claiming uses O_CREAT|O_EXCL so concurrent contention for the last slot lets exactly one
winner. The real environment builder is only reached through guard.launch(); the frozen counted
builder additionally refuses construction when no atomic claim is active.
"""
from __future__ import annotations
import json, os, threading, time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class R61QuotaExhausted(RuntimeError):
    pass


class R61Guard:
    def __init__(self, root: Path, *, normal_cap: int, reserve_cap: int, hard_cap: int) -> None:
        self.root = Path(root)
        self.slots_dir = self.root / "guard_slots"
        self.ledger_path = self.root / "guard_ledger.jsonl"
        self.normal_cap = int(normal_cap)
        self.reserve_cap = int(reserve_cap)
        self.hard_cap = int(hard_cap)
        self.slots_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def consumed(self) -> int:
        return len(list(self.slots_dir.glob("*.slot")))

    def _claimed_logicals(self) -> set:
        out = set()
        for p in self.slots_dir.glob("*.slot"):
            try:
                out.add(json.loads(p.read_text(encoding="utf-8"))["logical"])
            except Exception:
                out.add(p.read_text(encoding="utf-8"))
        return out

    def claim(self, logical_job: str, *, category: str = "normal") -> int:
        """Atomically claim+consume one slot. Refuses duplicates, category overflow and hard cap.
        A claimed slot is never refunded."""
        if category not in ("normal", "reserve"):
            raise ValueError(f"unknown category {category!r}")
        with self._lock:
            if logical_job in self._claimed_logicals():
                raise R61QuotaExhausted(f"logical job already claimed (no re-execution/dedup): {logical_job}")
            if self.consumed() >= self.hard_cap:
                raise R61QuotaExhausted(
                    f"R61 hard cap {self.hard_cap} reached (consumed {self.consumed()}); refusing {logical_job}")
            for n in range(1, self.hard_cap + 1):
                p = self.slots_dir / f"{n:04d}.slot"
                try:
                    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue
                os.write(fd, json.dumps({"logical": logical_job, "category": category,
                                         "claimed_unix": time.time()}).encode("utf-8"))
                os.close(fd)
                self._record(n, "slot_consumed", logical=logical_job, category=category)
                return n
            raise R61QuotaExhausted("no free slot under hard cap")

    def _record(self, slot: int, event: str, **kw: Any) -> None:
        rec = {"slot": slot, "event": event, "unix": time.time(), **kw}
        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def terminal(self, logical_job: str):
        """Terminal state of a logical from the ledger: 'complete' (qualified or ran-but-
        disqualified), 'failed' (interrupted / technical failure), or None (not claimed)."""
        if not self.ledger_path.exists():
            return None
        term = None
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("logical") != logical_job:
                continue
            if e["event"] in ("result_qualified", "result_disqualified"):
                term = "complete"
            elif e["event"] == "interrupted":
                term = "failed"
        return term

    def _reserve_used(self) -> int:
        n = 0
        for p in self.slots_dir.glob("*.slot"):
            try:
                if json.loads(p.read_text(encoding="utf-8")).get("category") == "reserve":
                    n += 1
            except Exception:
                pass
        return n

    def claim_reserve(self, logical_job: str) -> int:
        """Reserve retry of a persisted technical-failed normal claim. Refuses unless the logical
        has a failed (interrupted) attempt and no complete attempt."""
        with self._lock:
            term = self.terminal(logical_job)
            if term != "failed":
                raise R61QuotaExhausted(f"reserve retry requires a technical-failed attempt: {logical_job} (terminal={term})")
            if self._reserve_used() >= self.reserve_cap:
                raise R61QuotaExhausted(f"reserve cap {self.reserve_cap} reached")
            if self.consumed() >= self.hard_cap:
                raise R61QuotaExhausted(f"hard cap {self.hard_cap} reached")
            for n in range(1, self.hard_cap + 1):
                p = self.slots_dir / f"{n:04d}.slot"
                try:
                    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue
                os.write(fd, json.dumps({"logical": logical_job, "category": "reserve",
                                         "claimed_unix": time.time()}).encode("utf-8"))
                os.close(fd)
                self._record(n, "slot_consumed", logical=logical_job, category="reserve", retry=True)
                return n
            raise R61QuotaExhausted("no free slot under hard cap")

    @contextmanager
    def launch(self, logical_job: str, *, category: str = "normal") -> Iterator["R61Launch"]:
        """Claim a slot, then run the body. The slot is consumed on entry and never refunded,
        whatever the body does (success, construction failure, exception, quarantine)."""
        slot = self.claim_reserve(logical_job) if category == "reserve" else self.claim(logical_job, category="normal")
        launch = R61Launch(self, slot, logical_job)
        try:
            yield launch
        finally:
            if not launch.closed:
                launch.mark_interrupted()

    def snapshot(self) -> dict:
        events = [json.loads(l) for l in self.ledger_path.read_text(encoding="utf-8").splitlines() if l.strip()] \
            if self.ledger_path.exists() else []
        def count(ev):
            return sum(1 for e in events if e["event"] == ev)
        return {"consumed_slots": self.consumed(), "hard_cap": self.hard_cap,
                "headroom": self.hard_cap - self.consumed(),
                "builder_calls": count("builder_called"), "env_constructions": count("env_constructed"),
                "run_calls": count("run_called"), "run_returns": count("run_returned"),
                "steps_total": sum(int(e.get("steps", 0)) for e in events if e["event"] == "run_returned"),
                "results_qualified": count("result_qualified"),
                "results_disqualified": count("result_disqualified"),
                "interrupted": count("interrupted")}


class R61Launch:
    def __init__(self, guard: R61Guard, slot: int, logical: str) -> None:
        self.guard = guard; self.slot = slot; self.logical = logical; self.closed = False

    def builder_called(self) -> None:
        self.guard._record(self.slot, "builder_called", logical=self.logical)

    def env_constructed(self) -> None:
        self.guard._record(self.slot, "env_constructed", logical=self.logical)

    def run_called(self) -> None:
        self.guard._record(self.slot, "run_called", logical=self.logical)

    def run_returned(self, *, steps: int, ok: bool) -> None:
        self.guard._record(self.slot, "run_returned", logical=self.logical, steps=int(steps), ok=bool(ok))

    def result(self, *, qualified: bool, detail: str = "") -> None:
        self.guard._record(self.slot, "result_qualified" if qualified else "result_disqualified",
                           logical=self.logical, detail=detail)

    def mark_interrupted(self) -> None:
        self.closed = True
        self.guard._record(self.slot, "interrupted", logical=self.logical)

    def finish(self) -> None:
        self.closed = True
        self.guard._record(self.slot, "finished", logical=self.logical)
