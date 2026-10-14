"""ROUND75 central production resource ledger (the ONLY ledger real NN/Adam may attach to).

Stdlib-only (no torch): this module can be imported and exercised by pure spy tests with fake
callers. It reuses the R74-evidenced mechanisms (registered reuse, not a re-run of the R74 test
suites): charge-before-work pre-charge semantics, category caps (normal / technical / hard),
refusal logging that does not consume, durable JSON counters + append-only jsonl, a restart-safe
file read, a lock, single-target metering, and a consistency check.

R75 changes vs R74:
- production ONLY. The fixture execution mode is cancelled: ``simulated()`` raises, and the
  native wiring refuses any resource object without ``r75_central_production = True``.
- caps per the ROUND75 contract: synthetic physical optimizer steps normal 12 + technical 4 /
  hard 16; offline trunk target rows normal 5000 + technical 1000 / hard 6000 with a per-charge
  cap of 1024; joint records tracked separately (no numeric cap specified by the contract);
  environment constructions are FORBIDDEN entirely (0/0/0 - R75 runs no env, no real episode,
  no closed-loop simulation, and sampling may not construct an environment).
- technical reserve can only be spent on a PRE-REGISTERED failed attempt; ordinary check
  exhaustion never auto-upgrades.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

KIND = "r75-central-production-ledger/1"

# unit -> (normal_cap, technical_cap, hard_cap, per_charge_cap)
UNIT_CAPS = {
    "physical_optimizer_step": (12, 4, 16, None),
    "offline_trunk_target_row": (5000, 1000, 6000, 1024),
    "joint_record": (None, None, None, None),      # tracked separately; no numeric cap in contract
    "environment_construction": (0, 0, 0, None),   # forbidden entirely in R75
}


class R75LedgerError(RuntimeError):
    pass


def count_target_vectors(matrix) -> int:
    """Rows of a (n_candidates, 19) feature matrix; single target row accepted."""
    shape = getattr(matrix, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    if isinstance(matrix, (list, tuple)):
        return len(matrix)
    raise R75LedgerError("cannot count target rows of the given matrix")


class _Counter:
    def __init__(self, root: Path, unit: str) -> None:
        self.unit = unit
        self.normal_cap, self.technical_cap, self.hard_cap, self.per_charge_cap = UNIT_CAPS[unit]
        self.path = root / f"r75_{unit}.json"
        self.ledger_path = root / "r75_total_ledger.jsonl"
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.is_file():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"kind": f"r75-{self.unit}-guard/1", "unit": self.unit,
                "normal_cap": self.normal_cap, "technical_cap": self.technical_cap,
                "hard_cap": self.hard_cap, "per_charge_cap": self.per_charge_cap,
                "used_normal": 0, "used_reserve": 0, "refusals": 0,
                "registered_technical_attempts": [], "events": []}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def _append_ledger(self, event: dict[str, Any]) -> None:
        event = {"unix": time.time(), "guard": KIND, **event}
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def used(self) -> int:
        return int(self.state["used_normal"]) + int(self.state["used_reserve"])

    def pre_register_technical_attempt(self, attempt_id: str) -> None:
        if attempt_id not in self.state["registered_technical_attempts"]:
            self.state["registered_technical_attempts"].append(attempt_id)
            self._save()

    def charge(self, n: int, what: str, *, category: str = "normal") -> int:
        """Pre-charge: the durable count moves BEFORE the caller performs the work.

        A refused charge logs the refusal and consumes nothing. A successful charge is never
        refunded, including when the charged work subsequently fails.
        """
        n = int(n)
        if n < 0:
            raise R75LedgerError(f"{self.unit}: negative charge")
        if self.per_charge_cap is not None and n > self.per_charge_cap:
            self._refuse(n, what, f"per-charge cap {self.per_charge_cap}")
        if category not in ("normal", "technical"):
            raise R75LedgerError(f"{self.unit}: unknown category {category!r}")
        if category == "technical" and what not in self.state["registered_technical_attempts"] \
                and not any(what.startswith(a) for a in self.state["registered_technical_attempts"]):
            self._refuse(n, what, "technical reserve requires a pre-registered failed attempt")
        normal_room = None if self.normal_cap is None else self.normal_cap - self.state["used_normal"]
        if category == "normal" and normal_room is not None and n > normal_room:
            self._refuse(n, what, f"normal cap {self.normal_cap}")
        hard = self.hard_cap
        if hard is not None and self.used() + n > hard:
            self._refuse(n, what, f"hard cap {hard}")
        key = "used_normal" if category == "normal" else "used_reserve"
        self.state[key] += n
        self.state["events"].append({"event": "charged", "what": what, "n": n,
                                     "category": category, "used_after": self.used()})
        self._save()
        self._append_ledger({"event": "charged", "unit": self.unit, "what": what, "n": n,
                             "category": category, "used_after": self.used()})
        return self.used()

    def _refuse(self, n: int, what: str, reason: str) -> None:
        self.state["refusals"] += 1
        self.state["events"].append({"event": "charge_refused", "what": what, "n": n,
                                     "reason": reason, "used_after": self.used()})
        self._save()
        self._append_ledger({"event": "charge_refused", "unit": self.unit, "what": what,
                             "n": n, "reason": reason, "used_after": self.used()})
        raise R75LedgerError(f"{self.unit}: charge refused ({reason}): {what} n={n}")

    def snapshot(self) -> dict[str, Any]:
        return {"unit": self.unit, "used_normal": self.state["used_normal"],
                "used_reserve": self.state["used_reserve"], "used": self.used(),
                "refusals": self.state["refusals"], "normal_cap": self.normal_cap,
                "technical_cap": self.technical_cap, "hard_cap": self.hard_cap,
                "per_charge_cap": self.per_charge_cap}


class R75Resources:
    """The unique ROUND75 central production ledger. No fixture/simulated mode exists."""

    r75_central_production = True

    def __init__(self, root) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.root / "r75_ledger.lock"
        self._acquire_lock()
        self.synth_steps = _Counter(self.root, "physical_optimizer_step")
        self.offline_rows = _Counter(self.root, "offline_trunk_target_row")
        self.joint_records = _Counter(self.root, "joint_record")
        self.env_slots = _Counter(self.root, "environment_construction")

    # -- the ONLY construction path -------------------------------------------------
    @classmethod
    def production(cls, root) -> "R75Resources":
        return cls(root)

    @classmethod
    def simulated(cls, root) -> "R75Resources":  # pragma: no cover - refused by design
        raise R75LedgerError(
            "R75 fixture/simulated execution mode is CANCELLED: every real NN/Adam path must "
            "attach to the unique ROUND75 central production ledger")

    def _acquire_lock(self) -> None:
        try:
            self._lock_fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self._lock_fd, f"{os.getpid()} {time.time()}".encode("utf-8"))
        except FileExistsError:
            raise R75LedgerError(f"R75 ledger lock held: {self.lock_path}")

    def close(self) -> None:
        if getattr(self, "_lock_fd", None) is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
            try:
                self.lock_path.unlink()
            except OSError:
                pass

    def bootstrap(self) -> dict[str, Any]:
        """Persist the zero-value central ledger identity BEFORE any S1 charge exists.

        Pure file operation (no NN, no torch): writes each unit's zero counter and one
        append-only ``ledger-opened`` marker per unit. The marker is not a charge event, so
        ``consistency_check`` is unaffected. Refused once any consumption or refusal exists:
        the production identity can only be established at the true zero starting point.
        """
        for counter in (self.synth_steps, self.offline_rows, self.joint_records, self.env_slots):
            if counter.used() != 0 or int(counter.state["refusals"]):
                raise R75LedgerError(
                    f"R75 bootstrap refused: {counter.unit} is not at the zero starting point")
            counter._save()
            counter._append_ledger({"event": "ledger-opened", "unit": counter.unit,
                                    "used_normal": 0, "used_reserve": 0, "refusals": 0})
        return self.snapshots()

    # -- charge API used by the native wiring ---------------------------------------
    def charge_physical_step(self, what: str, *, category: str = "normal") -> int:
        return self.synth_steps.charge(1, what, category=category)

    def charge_offline_rows(self, n: int, what: str, *, category: str = "normal") -> int:
        return self.offline_rows.charge(int(n), what, category=category)

    def charge_trunk_forward(self, matrix, what: str, *, category: str = "normal") -> int:
        return self.offline_rows.charge(count_target_vectors(matrix), what, category=category)

    def charge_joint_record(self, n: int = 1, what: str = "joint_record") -> int:
        return self.joint_records.charge(n, what)

    def charge_env_construction(self, what: str) -> int:  # pragma: no cover - always refused
        return self.env_slots.charge(1, what)

    # -- bookkeeping -----------------------------------------------------------------
    def pre_register_technical_attempt(self, attempt_id: str) -> None:
        for counter in (self.synth_steps, self.offline_rows, self.joint_records, self.env_slots):
            counter.pre_register_technical_attempt(attempt_id)

    def snapshots(self) -> dict[str, Any]:
        return {"kind": KIND, "root": str(self.root),
                "physical_optimizer_step": self.synth_steps.snapshot(),
                "offline_trunk_target_row": self.offline_rows.snapshot(),
                "joint_record": self.joint_records.snapshot(),
                "environment_construction": self.env_slots.snapshot()}

    def consistency_check(self) -> dict[str, Any]:
        """Recompute every unit from the append-only ledger and compare with the counters."""
        totals: dict[str, int] = {}
        if self.root.joinpath("r75_total_ledger.jsonl").is_file():
            for line in self.root.joinpath("r75_total_ledger.jsonl").read_text(
                    encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("event") == "charged":
                    totals[event["unit"]] = totals.get(event["unit"], 0) + int(event["n"])
        counters = {"physical_optimizer_step": self.synth_steps,
                    "offline_trunk_target_row": self.offline_rows,
                    "joint_record": self.joint_records,
                    "environment_construction": self.env_slots}
        mismatches = {unit: {"ledger": totals.get(unit, 0), "counter": c.used()}
                      for unit, c in counters.items() if totals.get(unit, 0) != c.used()}
        return {"consistent": not mismatches, "mismatches": mismatches, "ledger_totals": totals}