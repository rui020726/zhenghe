"""ROUND74 PHASE A resource wiring.

Single durable total ledger + atomic quota guards. Every guard charges BEFORE the physical work,
never refunds on failure/interruption/quarantine/retry, and persists on disk so consumption is not
reset by changing the output subdirectory, restarting, or copying a backup. Independent units are
kept separate and all charge to ONE durable round74 total ledger that is not bypassable by
redirecting the output path.

Units (R74 caps):
  * environment-construction slot : normal 579 + reserve 21, hard 600
  * physical optimizer.step       : normal 12 + reserve 4,  hard 16   (denominated in PHYSICAL
                                    torch optimizer.step calls, NOT PPO update calls -- this is the
                                    R73 defect fixed; every PPO-internal step, pristine and scratch
                                    included, is charged here before it runs)
  * offline trunk TARGET forward  : normal 3000 + reserve 3000, hard 6000, per-batch <= 1024,
    rows                             counted as ACTUAL input vectors whose last dim == 19 at the real
                                    forward entry (never n_records*epochs); JOINT records counted in a
                                    SEPARATE counter and never used to impersonate target rows
  * PPO update call               : record-only, separate unit (a PPO call != a physical step)

PHASE A does NOT load the task model, run forward/backward, run PPO, or build a real environment.
The only real physical consumption allowed in PHASE A is <= 2 toy Adam.step calls on a constant
single parameter with an explicitly-set gradient (no model / environment / forward), charged before
each step from the hard-16 synthetic budget, with an independent sub-limit of 2 proving the third
attempt is refused before any Adam call or state change.
"""
from __future__ import annotations
import json, os, threading, time
from pathlib import Path
from typing import Iterator

R74_CANONICAL_ROOT = Path(__file__).resolve().parent / "raw" / "runtime"


class R74QuotaExhausted(RuntimeError):
    pass


class R74TotalLedger:
    """The single durable append-only round74 total ledger. Every charge across every guard is
    appended here (fsync). Fixed canonical path; not bypassable by output-path redirection."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, **event) -> None:
        rec = {"unix": time.time(), **event}
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def events(self) -> list:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]


class _AtomicCounter:
    """Charge-before-work atomic counter with normal/reserve categories and a hard cap.
    Persists to its own file AND mirrors every charge to the single total ledger."""

    def __init__(self, path, *, normal_cap, reserve_cap, hard_cap, kind, ledger, unit) -> None:
        self.path = Path(path)
        self.normal_cap = int(normal_cap)
        self.reserve_cap = int(reserve_cap)
        self.hard_cap = int(hard_cap)
        self.kind = kind
        self.unit = unit
        self.ledger = ledger
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.is_file():
            self._write({"kind": kind, "unit": unit, "normal_cap": self.normal_cap,
                         "reserve_cap": self.reserve_cap, "hard_cap": self.hard_cap,
                         "used_normal": 0, "used_reserve": 0, "refusals": 0, "events": []})

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, obj: dict) -> None:
        tmp = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def used(self) -> int:
        o = self._read()
        return int(o["used_normal"]) + int(o["used_reserve"])

    def remaining(self) -> int:
        return self.hard_cap - self.used()

    def charge(self, n: int, what: str, *, category: str = "normal") -> int:
        n = int(n)
        if n < 0:
            raise ValueError("charge amount must be >= 0")
        if category not in ("normal", "reserve"):
            raise ValueError(f"unknown category {category!r}")
        with self._lock:
            o = self._read()
            cat_key = "used_normal" if category == "normal" else "used_reserve"
            cat_cap = self.normal_cap if category == "normal" else self.reserve_cap
            if o[cat_key] + n > cat_cap or self.used() + n > self.hard_cap:
                o["refusals"] = int(o.get("refusals", 0)) + 1
                self._write(o)
                self.ledger.append(guard=self.kind, event="charge_refused", unit=self.unit,
                                   category=category, n=n, what=str(what),
                                   used_after=self.used(), hard_cap=self.hard_cap)
                raise R74QuotaExhausted(
                    f"{self.kind}: charging {n} ({what}, {category}) refused; "
                    f"{cat_key}={o[cat_key]}/{cat_cap}, hard used={self.used()}/{self.hard_cap}")
            o[cat_key] += n
            o["events"].append({"what": str(what), "n": n, "category": category,
                                "used_after": self.used(), "unix": time.time()})
            self._write(o)
            self.ledger.append(guard=self.kind, event="charged", unit=self.unit,
                               category=category, n=n, what=str(what),
                               used_normal=o["used_normal"], used_reserve=o["used_reserve"],
                               used_after=self.used(), hard_cap=self.hard_cap)
            return self.used()

    def snapshot(self) -> dict:
        o = self._read()
        return {"kind": self.kind, "unit": self.unit,
                "normal_cap": self.normal_cap, "reserve_cap": self.reserve_cap,
                "hard_cap": self.hard_cap, "used_normal": o["used_normal"],
                "used_reserve": o["used_reserve"], "used": self.used(),
                "remaining": self.remaining(), "refusals": o.get("refusals", 0),
                "events": o["events"]}

class R74EnvSlotGuard(_AtomicCounter):
    """Real environment-construction slots: normal 579 + reserve 21, hard 600."""
    def __init__(self, path, ledger) -> None:
        super().__init__(path, normal_cap=579, reserve_cap=21, hard_cap=600,
                         kind="r74-env-construction-slot-guard/1", ledger=ledger,
                         unit="environment_construction")


class R74SyntheticStepGuard(_AtomicCounter):
    """Physical torch optimizer.step calls: normal 12 + reserve 4, hard 16.
    Denominominated in PHYSICAL optimizer.step (NOT PPO update calls)."""
    def __init__(self, path, ledger) -> None:
        super().__init__(path, normal_cap=12, reserve_cap=4, hard_cap=16,
                         kind="r74-synthetic-physical-optimizer-step-guard/1", ledger=ledger,
                         unit="physical_optimizer_step")


class R74OfflineForwardGuard(_AtomicCounter):
    """Offline trunk TARGET forward rows: normal 3000 + reserve 3000, hard 6000, per-batch <= 1024.
    Rows are ACTUAL input vectors whose last dim == 19, counted at the real forward entry."""
    def __init__(self, path, ledger) -> None:
        super().__init__(path, normal_cap=3000, reserve_cap=3000, hard_cap=6000,
                         kind="r74-offline-trunk-target-forward-guard/1", ledger=ledger,
                         unit="offline_trunk_target_row")
        self.per_batch_cap = 1024

    def charge_targets(self, n, what, *, category="normal", batch=False):
        n = int(n)
        if batch and n > self.per_batch_cap:
            self.ledger.append(guard=self.kind, event="per_batch_refused", n=n,
                               per_batch_cap=self.per_batch_cap, what=str(what))
            raise R74QuotaExhausted(
                f"{self.kind}: batch of {n} target rows exceeds per-batch cap {self.per_batch_cap}")
        return self.charge(n, what, category=category)


class R74JointRecordGuard(_AtomicCounter):
    """JOINT records counted SEPARATELY; never used to impersonate target rows (record-only)."""
    def __init__(self, path, ledger) -> None:
        super().__init__(path, normal_cap=10**9, reserve_cap=10**9, hard_cap=10**9,
                         kind="r74-joint-record-guard/1", ledger=ledger, unit="joint_record")


class R74PpoCallGuard(_AtomicCounter):
    """PPO update CALLS: a separate unit from physical step (record-only; a call != a step)."""
    def __init__(self, path, ledger) -> None:
        super().__init__(path, normal_cap=10**9, reserve_cap=10**9, hard_cap=10**9,
                         kind="r74-ppo-call-guard/1", ledger=ledger, unit="ppo_update_call")


def count_target_vectors(x) -> int:
    """Actual trunk TARGET input vectors = number of rows whose last dim == 19.
    Never an n_records*epochs estimate; a non-19 last dim contributes 0."""
    shape = getattr(x, "shape", None)
    if shape is None or len(shape) < 2:
        return 0
    if int(shape[-1]) != 19:
        return 0
    n = 1
    for d in shape[:-1]:
        n *= int(d)
    return int(n)


class R74Resources:
    """Wires every guard to the ONE canonical durable root + single total ledger. PHASE B must
    construct this with the default canonical root; redirecting the output path does not create a
    fresh ledger because the canonical root is fixed and every charge is fsync-appended there."""

    def __init__(self, root=None) -> None:
        self.root = Path(root) if root is not None else R74_CANONICAL_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger = R74TotalLedger(self.root / "r74_total_ledger.jsonl")
        self.env_slots = R74EnvSlotGuard(self.root / "r74_env_slot_counter.json", self.ledger)
        self.synth_steps = R74SyntheticStepGuard(self.root / "r74_synthetic_step_counter.json", self.ledger)
        self.offline_forward = R74OfflineForwardGuard(self.root / "r74_offline_forward_counter.json", self.ledger)
        self.joint_records = R74JointRecordGuard(self.root / "r74_joint_record_counter.json", self.ledger)
        self.ppo_calls = R74PpoCallGuard(self.root / "r74_ppo_call_counter.json", self.ledger)

    def snapshots(self) -> dict:
        return {
            "canonical_root": str(self.root),
            "total_ledger_path": str(self.ledger.path),
            "total_ledger_events": len(self.ledger.events()),
            "env_slots": self.env_slots.snapshot(),
            "synthetic_physical_steps": self.synth_steps.snapshot(),
            "offline_forward_target_rows": self.offline_forward.snapshot(),
            "joint_records": self.joint_records.snapshot(),
            "ppo_calls": self.ppo_calls.snapshot(),
        }
