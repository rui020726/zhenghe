"""ROUND68 extra-computation budget guards (acceptance / synthetic tests only).

Independent of the 800 environment-construction budget (which the r40_common claim/slot ledger
enforces), ROUND68 caps the ACCEPTANCE/SYNTHETIC-TEST model forward rows and synthetic optimizer
steps. Real training forwards and optimizer steps are accounted separately under the 800 budget and
must NEVER be charged here.

Both guards persist an atomic counter that is charged BEFORE the work runs, never refunded on
failure or rerun, and never reset by changing the output subdirectory. A stale lock (>60s) is broken
so a crashed holder cannot deadlock the counter.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

_LOCK_STALE_SECONDS = 60.0


class _AtomicCounter:
    def __init__(self, ledger_path, cap: int, kind: str) -> None:
        self.path = Path(ledger_path)
        self.cap = int(cap)
        self.kind = kind
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.is_file():
            self._write({"kind": kind, "cap": self.cap, "used": 0, "events": []})

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, obj: dict) -> None:
        tmp = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def _acquire(self) -> Path:
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        while True:
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return lock
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > _LOCK_STALE_SECONDS:
                        lock.unlink()
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(0.05)

    def used(self) -> int:
        return int(self._read()["used"])

    def remaining(self) -> int:
        return self.cap - self.used()

    def charge(self, n: int, what: str) -> int:
        n = int(n)
        if n < 0:
            raise ValueError("charge amount must be >= 0")
        lock = self._acquire()
        try:
            obj = self._read()
            if obj["used"] + n > self.cap:
                raise RuntimeError(
                    f"{self.kind}: charging {n} ({what}) would take used to {obj['used'] + n} > cap {self.cap}")
            obj["used"] += n
            obj["events"].append({"what": str(what), "n": n, "used_after": obj["used"],
                                  "unix": time.time()})
            self._write(obj)
            return int(obj["used"])
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass

    def snapshot(self) -> dict:
        obj = self._read()
        return {"kind": self.kind, "cap": self.cap, "used": obj["used"],
                "remaining": self.cap - obj["used"], "events": obj["events"]}


class ForwardRowGuard(_AtomicCounter):
    """Caps acceptance/synthetic-test model forward rows ('per model per input row')."""

    def __init__(self, ledger_path, cap: int = 8192) -> None:
        super().__init__(ledger_path, cap, "r68-acceptance-forward-row-guard/1")


class SyntheticStepGuard(_AtomicCounter):
    """Caps synthetic optimizer.step physical calls (PPO-internal steps and failures included)."""

    def __init__(self, ledger_path, cap: int = 32) -> None:
        super().__init__(ledger_path, cap, "r68-synthetic-optimizer-step-guard/1")


class SyntheticPpoGuard(_AtomicCounter):
    """Counts synthetic PPO update calls separately (no hard cap asserted here; for the record)."""

    def __init__(self, ledger_path, cap: int = 32) -> None:
        super().__init__(ledger_path, cap, "r68-synthetic-ppo-call-guard/1")
