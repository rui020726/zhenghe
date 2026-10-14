"""ROUND75 central production resource ledger (S0 acceptance hardening revision).

This revision closes the defects the decision-end review found in the S0 delivery
(commit 0f8574a). The S0 receipt is preserved; this file supersedes it and the change is
registered in ROUND75_S0_ACCEPTANCE_VERDICT.json.

Faithful reuse of the R74-verified guard machinery (registered, NOT re-written weaker):
the durable fail-closed counter (per-category caps, durable-state validation, cross-instance
/ cross-process O_EXCL lock with no recovery, fsync persist, ledger cross-check at open and
inside every charge, never refund), the verified 19-D row counter and the identity-preserving
entry wrapper are IMPORTED from the round74/r43 modules whose byte identity is pinned below.
R75 only adds: the R75 caps/kinds, the pinned canonical root, the production/fixture split,
the unified trunk-entry meter and the strict native-entry identity check.

What changed vs the S0 version (all decision-end items):
1. production() is PINNED to the canonical durable root (round75 raw/runtime); a non-canonical
   root or a redirected output can no longer mint fresh quota. A fixture is a SEPARATE
   non-production interface (explicit throwaway root; the canonical root and its descendants
   are refused; a fixture can never be handed to the real NN path).
2. _ProdCounter (r74) enforces the TECHNICAL category's own cap (4 steps / 1000 rows) in
   addition to the normal cap and the hard cap, validates the durable state (kind/unit/caps,
   non-negative ints, per-category and hard bounds), fails closed on a missing counter with
   prior charges, on a low backup, on a counter ahead of the ledger and on a missing ledger,
   and cross-checks the ledger INSIDE the lock on every charge (not only in a report).
3. count_target_vectors is the verified r74 rule: (19,) counts 1, (4,5,19) counts 20, a
   non-19 last dim raises at the trunk entry. A unified meter is installed at the REAL
   controller.inner.trunk.forward: the ACTUAL input shape is counted and pre-charged at the
   entry, covering the alignment pass, every epoch and any other legitimate check. Caller-side
   aggregate pre-charges are removed. Attempted / charged(licensed) / completed calls and rows
   are reported separately, so a mid-way failure never passes a pre-charge off as a forward.
4. freeze_trunk_only (r75_frozen) now explicitly sets the four trainable modules to
   requires_grad=True before freezing the trunk; it no longer relies on constructor defaults.
5. One native batch read is charged as ONE joint record per update and reported together with
   the unique recorded input records and the exposure count; the S0 plan's "joint records = 0"
   is superseded (see the acceptance verdict). Joint records are NOT environment episodes.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
R74_DIR = HERE.parent / "round74_constrained_early_head_adaptation"
for _text in (str(R74_DIR), str(HERE.parent)):
    if _text not in sys.path:
        sys.path.insert(0, _text)

# --- faithful reuse of the R74-verified primitives (byte identity pinned below) ----------
from r74_guard_prod import _ProdCounter, count_target_vectors          # noqa: E402
from r74_guard_prod import LedgerStateError as _R74LedgerStateError    # noqa: E402
from r74_guard import R74QuotaExhausted as _R74QuotaExhausted          # noqa: E402
from r74_guard import R74TotalLedger as _R74TotalLedger                # noqa: E402
from r43_call_counter_hooks import count_at_entry, COUNTER_HOOK_VERSION  # noqa: E402

# Single exception identity across the reused machinery (aliases, not new weak classes).
R75LedgerError = _R74LedgerStateError
R75QuotaExhausted = _R74QuotaExhausted

# --- pinned provenance of the reused verified sources ------------------------------------
REUSED_SOURCE_IDENTITY = {
    "r74_guard_prod.py": "623b9ec6cd243d27321e8c745a0247c97126c9028141cb0ed4760dd90c0796db",
    "r74_guard.py": "9a6b06374fc5b56b6d8077dc6aa332444533198aca0967cb36b867795af614d7",
    "r43_call_counter_hooks.py": "02f80072a14d7305b619edd0db9617a7b0a0f31315b3265aebf1e694ac7398f7",
    "counter_hook_version": COUNTER_HOOK_VERSION,
}

# THE canonical durable root. production() is pinned here; every other root is refused.
R75_CANONICAL_ROOT = HERE / "raw" / "runtime"

# The S0 (superseded) zero-value counter files, kept byte-identical as the S0 receipt.
LEGACY_S0_COUNTER_FILES = ("r75_physical_optimizer_step.json",
                           "r75_offline_trunk_target_row.json",
                           "r75_joint_record.json",
                           "r75_environment_construction.json")

KIND = "r75-central-production-ledger/1"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reused_source_identity() -> dict[str, Any]:
    """Fail closed if any reused R74/r43 source drifted from its pinned byte identity."""
    report = {}
    for name, expected in REUSED_SOURCE_IDENTITY.items():
        if name == "counter_hook_version":
            continue
        path = (R74_DIR if name.startswith("r74_") else HERE.parent) / name
        actual = sha256_file(path)
        if actual != expected:
            raise R75LedgerError(
                f"reused guard source {name} byte identity changed ({actual}); refusing R75")
        report[name] = actual
    return report


# ======================================================================================
# R75 counters: the R74-verified _ProdCounter with the R75 caps and kinds.
#   physical_optimizer_step : normal 12 + technical 4 / hard 16
#   offline_trunk_target_row: normal 5000 + technical 1000 / hard 6000, per-charge cap 1024
#   joint_record            : tracked separately (no numeric cap in the contract)
#   environment_construction: 0 / 0 / 0 - forbidden entirely in R75
# ======================================================================================
class R75PhysicalStepGuard(_ProdCounter):
    def __init__(self, path, ledger, *, mode, lock_timeout=15.0):
        super().__init__(path, mode=mode, ledger=ledger, normal_cap=12, reserve_cap=4,
                         hard_cap=16, kind="r75-physical-optimizer-step-guard/1",
                         unit="physical_optimizer_step", lock_timeout=lock_timeout)


class R75OfflineRowGuard(_ProdCounter):
    def __init__(self, path, ledger, *, mode, lock_timeout=15.0):
        super().__init__(path, mode=mode, ledger=ledger, normal_cap=5000, reserve_cap=1000,
                         hard_cap=6000, kind="r75-offline-trunk-target-row-guard/1",
                         unit="offline_trunk_target_row", lock_timeout=lock_timeout)
        self.per_charge_cap = 1024

    def charge_rows(self, n: int, what: str, *, category: str = "normal") -> int:
        """Pre-charge n offline target rows; the per-charge cap is ALWAYS enforced here."""
        n = int(n)
        if n > self.per_charge_cap:
            self.ledger.append(guard=self.kind, event="per_charge_refused", n=n,
                               per_charge_cap=self.per_charge_cap, what=str(what))
            raise R75QuotaExhausted(
                f"{self.kind}: {n} target rows exceeds the per-charge cap {self.per_charge_cap} "
                f"({what}); refused before any work, nothing consumed")
        return self.charge(n, what, category=category)

    def charge_trunk_forward(self, matrix, what: str, *, category: str = "normal") -> int:
        """REAL trunk-entry metering seam: count the ACTUAL 19-D rows, then pre-charge."""
        n = count_target_vectors(matrix)          # raises on a non-19 last dim
        return self.charge_rows(n, what, category=category)


class R75JointRecordGuard(_ProdCounter):
    def __init__(self, path, ledger, *, mode, lock_timeout=15.0):
        super().__init__(path, mode=mode, ledger=ledger, normal_cap=10 ** 9, reserve_cap=10 ** 9,
                         hard_cap=10 ** 9, kind="r75-joint-record-guard/1",
                         unit="joint_record", lock_timeout=lock_timeout)


class R75EnvSlotGuard(_ProdCounter):
    def __init__(self, path, ledger, *, mode, lock_timeout=15.0):
        super().__init__(path, mode=mode, ledger=ledger, normal_cap=0, reserve_cap=0,
                         hard_cap=0, kind="r75-environment-construction-guard/1",
                         unit="environment_construction", lock_timeout=lock_timeout)


# ======================================================================================
# unified trunk-entry meter (installed at the REAL controller.inner.trunk.forward)
# ======================================================================================
class TrunkEntryMeter:
    """Meter every REAL trunk forward: actual input shape counted and pre-charged at the entry.

    Covers the alignment pass, every training epoch and any other legitimate trunk call,
    because it wraps the module the controller itself calls. Attempted / charged(licensed) /
    completed calls and rows are separate counters: a refused call never runs the forward, a
    failed forward stays charged (no refund) but is never counted as completed, and a
    non-19 input raises before any charge or forward.
    """

    def __init__(self, module, guard: R75OfflineRowGuard, *, what_prefix: str):
        if getattr(module, "r75_trunk_meter", None) is not None:
            raise R75LedgerError(
                "a trunk entry meter is already installed on this module; double install refused")
        self.module = module
        self.guard = guard
        self.what_prefix = str(what_prefix)
        self._original_forward = module.forward
        self.calls_attempted = 0
        self.calls_charged = 0
        self.rows_charged = 0
        self.calls_completed = 0
        self.rows_completed = 0
        self.calls_refused = 0
        self.calls_failed = 0
        module.forward = self._forward
        module.r75_trunk_meter = self

    def _forward(self, *args, **kwargs):
        self.calls_attempted += 1
        batch = args[0] if args else kwargs.get("input", kwargs.get("batch"))
        rows = count_target_vectors(batch)        # a non-19 input raises before any charge
        try:
            self.guard.charge_trunk_forward(batch, f"{self.what_prefix}_call{self.calls_attempted}")
        except R75QuotaExhausted:
            self.calls_refused += 1
            raise
        self.calls_charged += 1
        self.rows_charged += rows
        try:
            output = self._original_forward(*args, **kwargs)
        except BaseException:
            self.calls_failed += 1                  # charged, NOT completed, never refunded
            raise
        self.calls_completed += 1
        self.rows_completed += rows
        return output

    def snapshot(self) -> dict[str, int]:
        return {"calls_attempted": self.calls_attempted, "calls_charged": self.calls_charged,
                "rows_charged": self.rows_charged, "calls_completed": self.calls_completed,
                "rows_completed": self.rows_completed, "calls_refused": self.calls_refused,
                "calls_failed": self.calls_failed}

    def uninstall(self) -> None:
        try:
            del self.module.forward                 # restore the class method
        except AttributeError:
            self.module.forward = self._original_forward
        try:
            del self.module.r75_trunk_meter
        except AttributeError:
            pass


# ======================================================================================
# the unique ROUND75 central production ledger (+ the separate non-production fixture)
# ======================================================================================
class R75Resources:
    """The ONLY resource object the real NN/Adam path may attach to.

    production() is pinned to the canonical durable root; fixture() is the separate
    non-production spy interface (throwaway root, canonical root and descendants refused,
    never handed to the real NN path). There is no simulated() quota path.
    """

    r75_central_production = True

    def __init__(self, *, mode: str, root=None):
        if mode not in ("production", "fixture"):
            raise ValueError("mode must be 'production' or 'fixture'")
        self.mode = mode
        canonical = R75_CANONICAL_ROOT
        if mode == "production":
            if root is not None and Path(root).resolve() != canonical.resolve():
                raise R75LedgerError(
                    "R75 production resources are PINNED to the canonical durable root "
                    f"{canonical}; a non-canonical root / redirected output is refused (it "
                    "cannot mint fresh quota)")
            self.root = canonical
        else:
            if root is None:
                raise ValueError("fixture resources require an explicit throwaway root")
            candidate = Path(root).resolve()
            if candidate == canonical.resolve() or canonical.resolve() in candidate.parents:
                raise R75LedgerError(
                    "a fixture may not use the canonical durable root or any path inside it; "
                    "the real ledger is only reachable through production()")
            self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.r75_fixture = (mode == "fixture")
        self.ledger = _R74TotalLedger(self.root / "r75_total_ledger.jsonl")
        self.physical_steps = R75PhysicalStepGuard(
            self.root / "r75_physical_optimizer_step_guard.json", self.ledger, mode=mode)
        self.offline_rows = R75OfflineRowGuard(
            self.root / "r75_offline_trunk_target_row_guard.json", self.ledger, mode=mode)
        self.joint_records = R75JointRecordGuard(
            self.root / "r75_joint_record_guard.json", self.ledger, mode=mode)
        self.env_slots = R75EnvSlotGuard(
            self.root / "r75_environment_construction_guard.json", self.ledger, mode=mode)
        self._registered_technical_attempts: list[str] = []
        self._trunk_meters: list[TrunkEntryMeter] = []

    # -- the ONLY construction paths ------------------------------------------------------
    @classmethod
    def production(cls, root=None) -> "R75Resources":
        return cls(mode="production", root=root)

    @classmethod
    def fixture(cls, root) -> "R75Resources":
        return cls(mode="fixture", root=root)

    @classmethod
    def simulated(cls, root) -> "R75Resources":  # pragma: no cover - refused by design
        raise R75LedgerError(
            "R75 has no simulated quota path; pure spies use the SEPARATE non-production "
            "fixture interface R75Resources.fixture(root), which the native NN path refuses")

    def close(self) -> None:
        """No instance-held lock exists: every counter lock is per-charge O_EXCL."""

    # -- technical-attempt pre-registration (R75 contract on top of the r74 counter) -------
    def pre_register_technical_attempt(self, attempt_id: str) -> None:
        if attempt_id not in self._registered_technical_attempts:
            self._registered_technical_attempts.append(attempt_id)
            self.ledger.append(guard=KIND, event="technical_attempt_registered",
                               attempt_id=str(attempt_id), mode=self.mode)

    def _require_registered(self, what: str, category: str) -> None:
        if category != "technical":
            return
        if not any(what == a or what.startswith(a)
                   for a in self._registered_technical_attempts):
            raise R75LedgerError(
                f"technical charge {what!r} is not a PRE-REGISTERED failed attempt; ordinary "
                f"exhaustion never auto-upgrades to the technical reserve")

    # -- charge API -----------------------------------------------------------------------
    def charge_physical_step(self, what: str, *, category: str = "normal") -> int:
        cat = "reserve" if category == "technical" else "normal"
        self._require_registered(what, category)
        return self.physical_steps.charge(1, what, category=cat)

    def charge_offline_rows(self, n: int, what: str, *, category: str = "normal") -> int:
        cat = "reserve" if category == "technical" else "normal"
        self._require_registered(what, category)
        return self.offline_rows.charge_rows(n, what, category=cat)

    def charge_joint_record(self, n: int = 1, what: str = "joint_record") -> int:
        return self.joint_records.charge(n, what)

    def charge_env_construction(self, what: str) -> int:  # pragma: no cover - always refused
        return self.env_slots.charge(1, what)

    # -- unified trunk-entry meter ----------------------------------------------------------
    def install_trunk_meter(self, controller, *, what_prefix: str) -> TrunkEntryMeter:
        meter = TrunkEntryMeter(controller.inner.trunk, self.offline_rows,
                                what_prefix=what_prefix)
        self._trunk_meters.append(meter)
        return meter

    def ensure_trunk_meter(self, controller, *, what_prefix: str) -> TrunkEntryMeter:
        """Idempotent: reuse an installed meter, otherwise install one.

        The meter is the UNIFIED real entry: once installed it keeps metering every real trunk
        forward of this controller (that is the requirement, not a leak); teardown is explicit
        through uninstall_trunk_meters().
        """
        existing = getattr(controller.inner.trunk, "r75_trunk_meter", None)
        if existing is not None:
            return existing
        return self.install_trunk_meter(controller, what_prefix=what_prefix)

    def uninstall_trunk_meters(self) -> None:
        for meter in self._trunk_meters:
            meter.uninstall()
        self._trunk_meters = []

    # -- reconciliation ---------------------------------------------------------------------
    def verify_or_fail(self) -> dict[str, Any]:
        """Force reconciliation NOW (open / before any work). Production fails closed."""
        maxima: dict[str, int] = {}
        for event in self.ledger.events():
            if event.get("event") == "charged":
                guard = event.get("guard")
                maxima[guard] = max(maxima.get(guard, 0), int(event.get("used_after", 0)))
        mismatches = {}
        for counter in (self.physical_steps, self.offline_rows, self.joint_records, self.env_slots):
            used = counter.used()
            ledger_max = maxima.get(counter.kind, 0)
            if self.mode == "production" and used != ledger_max:
                mismatches[counter.unit] = {"counter_used": used, "ledger_max": ledger_max}
        if mismatches:
            raise R75LedgerError(
                f"R75 production ledger reconciliation failed: {mismatches}; consumed permission "
                f"is preserved, nothing is refunded, recovered or rewritten")
        return {"mode": self.mode, "consistent": not mismatches, "ledger_maxima": maxima}

    def consistency_check(self) -> dict[str, Any]:
        """Report-only twin of verify_or_fail (never blocks by itself)."""
        try:
            return {**self.verify_or_fail(), "blocking": False}
        except R75LedgerError as error:
            return {"mode": self.mode, "consistent": False, "error": str(error),
                    "blocking": False}

    # -- zero-identity bootstrap -------------------------------------------------------------
    def bootstrap(self) -> dict[str, Any]:
        """Persist/verify the zero-value identity BEFORE any charge exists (pure file op)."""
        if self.mode != "production":
            raise R75LedgerError("bootstrap is a production-only identity action")
        for counter in (self.physical_steps, self.offline_rows, self.joint_records, self.env_slots):
            snap = counter.snapshot()
            if snap["used"] or snap["refusals"]:
                raise R75LedgerError(
                    f"bootstrap refused: {counter.unit} is not at the zero starting point "
                    f"(used={snap['used']}, refusals={snap['refusals']})")
        report = self.verify_or_fail()
        for counter in (self.physical_steps, self.offline_rows, self.joint_records, self.env_slots):
            self.ledger.append(guard=counter.kind, event="ledger-opened", unit=counter.unit,
                               used_normal=0, used_reserve=0, refusals=0)
        # Register the superseded S0 zero counters WITHOUT touching them (read-only hashes).
        legacy = {}
        for name in LEGACY_S0_COUNTER_FILES:
            path = self.root / name
            if not path.is_file():
                legacy[name] = "absent"
                continue
            obj = json.loads(path.read_text(encoding="utf-8"))
            if int(obj.get("used_normal", 0)) or int(obj.get("used_reserve", 0)) \
                    or int(obj.get("refusals", 0)):
                raise R75LedgerError(
                    f"bootstrap refused: legacy S0 counter {name} is not at zero; the S0 "
                    f"receipt may not be rewritten")
            legacy[name] = sha256_file(path)
            self.ledger.append(guard=KIND, event="legacy-s0-zero-superseded", file=name,
                               sha256=legacy[name], note="kept byte-identical; not cleared")
        return {**self.snapshots(), "verify": report, "legacy_s0_counters": legacy}

    # -- bookkeeping -------------------------------------------------------------------------
    def snapshots(self) -> dict[str, Any]:
        return {"kind": KIND, "mode": self.mode, "canonical_root": str(R75_CANONICAL_ROOT),
                "root": str(self.root),
                "pinned_to_canonical": self.root.resolve() == R75_CANONICAL_ROOT.resolve(),
                "physical_optimizer_step": self.physical_steps.snapshot(),
                "offline_trunk_target_row": self.offline_rows.snapshot(),
                "joint_record": self.joint_records.snapshot(),
                "environment_construction": self.env_slots.snapshot(),
                "total_ledger_events": len(self.ledger.events()),
                "registered_technical_attempts": list(self._registered_technical_attempts)}


# ======================================================================================
# the strict native-entry identity check (the only attachment the real NN path accepts)
# ======================================================================================
def assert_native_production_resources(resources, *,
                                       expected_guard_source_sha256: str | None = None) -> bool:
    """Fail closed unless resources IS the R75 central production ledger.

    Checks the actual class identity (a duck-typed object carrying a forgeable
    ``r75_central_production`` boolean is refused), the production mode, the canonical root,
    and - when the caller pins it - the byte identity of this very module's source, so a
    weakened guard cannot ride in on a renamed object.
    """
    if resources is None:
        raise R75LedgerError(
            "the R75 native PPO path requires the ROUND75 central production ledger; None is "
            "refused by source")
    if type(resources) is not R75Resources:
        raise R75LedgerError(
            f"the native path received {type(resources).__module__}.{type(resources).__name__}, "
            f"not r75_guard.R75Resources; a forged production object is refused by source")
    if resources.mode != "production" or getattr(resources, "r75_fixture", False):
        raise R75LedgerError(
            "the native path refuses a non-production (fixture) resource attachment; pure spies "
            "never reach the real NN path")
    if Path(resources.root).resolve() != R75_CANONICAL_ROOT.resolve():
        raise R75LedgerError(
            "the native path refuses a production object whose root is not the canonical "
            "durable root")
    if expected_guard_source_sha256 is not None:
        actual = sha256_file(Path(__file__).resolve())
        if actual != expected_guard_source_sha256:
            raise R75LedgerError(
                f"r75_guard.py source identity {actual} != pinned "
                f"{expected_guard_source_sha256}; a modified guard is refused by the native path "
                f"(re-register the pin only through an explicit acceptance revision)")
    return True