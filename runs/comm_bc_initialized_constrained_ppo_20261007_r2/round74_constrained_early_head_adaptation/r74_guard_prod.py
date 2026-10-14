"""ROUND74 PHASE A production resource seams (hardening). Phase B must use THIS path.

This module closes the gaps recorded in the PHASE A reconciliation without touching the original
PHASE A receipt (r74_guard.py / r74_phase_a.py / raw/runtime stay byte-identical). It reuses the
REAL production boundaries rather than re-manual-charge spies:

  * per-physical-step pre-charge goes through the ACTUAL r43 count_at_entry wrapper
    (runs/comm_bc_initialized_constrained_ppo_20261007_r2/r43_call_counter_hooks.py), the same
    identity-preserving wrapper installed around torch.optim.Adam.step by configure() in
    r46_execute_r2.py; a charge is therefore proven at the real Adam.step entry.
  * offline/online trunk TARGET forward metering uses the R71 ForwardLedger durable-base-in-lock
    pattern (analysis/round71-critic-calibration-readiness/r71_critic_calibration_seam.py) at the
    real trunk entry (controller.inner.trunk), counting ACTUAL 19-D input rows.

Production vs fixture are separated: R74Resources.production() pins the canonical durable root and
refuses any non-canonical root; R74Resources.fixture(root) records simulated consumption only and
can never act as production. Counters are durable-base-inside-lock (no lost update across two
instances), fail-closed on any corrupt / partial / foreign / lost / low-backup state, and never
refund. A cross-instance/cross-process O_EXCL lock serialises mutation; ANY residual lock fails
closed (no recovery), matching R71. This module does NOT claim physical power-loss atomicity.

PHASE A performs NO real Adam.step, NO model load, NO forward/backward, NO PPO, NO environment.
All verification is pure synthetic / spy over throwaway fixtures that never touch the canonical
ledger. The canonical ledger remains at its recorded 2 real toy steps.
"""
from __future__ import annotations
import contextlib, json, os, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                 # r74_guard (PHASE A receipt, reused for the exception)
sys.path.insert(0, str(HERE.parent))          # the REAL r43 call-counter hook lives one level up

from r74_guard import R74QuotaExhausted, R74_CANONICAL_ROOT
from r43_call_counter_hooks import count_at_entry, COUNTER_HOOK_VERSION  # the real boundary


class LedgerStateError(RuntimeError):
    """Fail-closed: a durable counter is corrupt / partial / foreign / lost / a low backup."""


def count_target_vectors(x) -> int:
    """Actual trunk TARGET input vectors = rows whose last dim == 19.

    FIX (R74 reconciliation item 3): a valid single-target 1-D input of shape (19,) counts as 1
    (the previous version returned 0). A non-19 last dim is REJECTED with LedgerStateError at the
    trunk entry rather than silently defaulting to 0, so a malformed matrix cannot sneak past the
    offline quota. Never an n_records*epochs estimate.
    """
    shape = getattr(x, "shape", None)
    if shape is None:
        raise LedgerStateError("trunk input has no .shape; cannot validate the 19-D feature dim")
    shape = tuple(int(d) for d in shape)
    if len(shape) == 0 or shape[-1] != 19:
        raise LedgerStateError(
            f"trunk input last dim {shape[-1] if shape else None} != 19; a non-19 matrix is "
            f"refused at the trunk entry, never defaulted to zero rows")
    n = 1
    for d in shape[:-1]:
        n *= int(d)
    return int(n)

class _ProdCounter:
    """Durable, fail-closed, cross-instance-atomic quota counter (normal/reserve + hard cap).

    Corrections vs the PHASE A _AtomicCounter:
      * cross-instance / cross-process O_EXCL lock (the PHASE A threading.Lock was per-instance and
        in-process only, so two instances could both pass a check against their own stale copy and
        lose an update). ANY residual lock fails closed; there is no recovery.
      * every charge reads the validated latest DURABLE state INSIDE the lock and publishes only
        after the new state is durable, so a refusal or persist failure never overwrites another
        instance's consumption.
      * production mode fails closed on a MISSING counter whose ledger shows prior charges, and on
        a counter whose used is below the ledger-recorded maximum (a stale / low backup). Fixtures
        may start fresh.
      * no refund: a charge is never undone on failure / interruption / quarantine / retry.
    """

    def __init__(self, path, *, mode, ledger, normal_cap, reserve_cap, hard_cap, kind, unit,
                 lock_timeout=15.0) -> None:
        if mode not in ("production", "fixture"):
            raise ValueError("mode must be 'production' or 'fixture'")
        self.path = Path(path)
        self.mode = mode
        self.ledger = ledger
        self.normal_cap = int(normal_cap)
        self.reserve_cap = int(reserve_cap)
        self.hard_cap = int(hard_cap)
        self.kind = kind
        self.unit = unit
        self.lock_timeout = float(lock_timeout)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_state()

    # ------------------------------------------------------------------ durable state
    def _read_persisted(self) -> dict:
        try:
            obj = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LedgerStateError(f"counter {self.path} unreadable/corrupt: {exc}") from exc
        if not isinstance(obj, dict):
            raise LedgerStateError(f"counter {self.path} is not a JSON object")
        for k in ("kind", "unit", "normal_cap", "reserve_cap", "hard_cap",
                  "used_normal", "used_reserve", "refusals"):
            if k not in obj:
                raise LedgerStateError(
                    f"counter {self.path} missing key {k!r}; a partial/foreign file is never zeroed")
        if obj["kind"] != self.kind or obj["unit"] != self.unit:
            raise LedgerStateError(
                f"counter identity {obj.get('kind')}/{obj.get('unit')} != {self.kind}/{self.unit}")
        for k in ("normal_cap", "reserve_cap", "hard_cap"):
            if obj[k] != getattr(self, k):
                raise LedgerStateError(f"counter {k} {obj[k]} != {getattr(self, k)}")
        for k in ("used_normal", "used_reserve", "refusals"):
            v = obj[k]
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise LedgerStateError(f"counter {k!r} must be a non-negative int")
        if obj["used_normal"] + obj["used_reserve"] > self.hard_cap:
            raise LedgerStateError(
                f"counter used {obj['used_normal'] + obj['used_reserve']} exceeds hard cap {self.hard_cap}")
        obj.setdefault("events", [])
        return obj

    def _cross_check_ledger(self, obj: dict) -> None:
        used = obj["used_normal"] + obj["used_reserve"]
        ledger_max = 0
        for e in self.ledger.events():
            if e.get("event") == "charged" and e.get("guard") == self.kind:
                try:
                    ledger_max = max(ledger_max, int(e.get("used_after", 0)))
                except (TypeError, ValueError):
                    raise LedgerStateError("ledger used_after is not an int")
        if used < ledger_max:
            raise LedgerStateError(
                f"production counter used {used} < ledger-recorded {ledger_max} for {self.kind}; "
                f"a stale/low backup is being adopted and fails closed")

    def _init_state(self) -> None:
        if self.path.is_file():
            obj = self._read_persisted()
            if self.mode == "production":
                self._cross_check_ledger(obj)
            return
        if self.mode == "production":
            prior = [e for e in self.ledger.events()
                     if e.get("event") == "charged" and e.get("guard") == self.kind]
            if prior:
                raise LedgerStateError(
                    f"production counter {self.path} is MISSING but the total ledger records "
                    f"{len(prior)} prior charge(s) for {self.kind}; a lost counter fails closed "
                    f"and is never reset to zero")
        self._persist({"kind": self.kind, "unit": self.unit, "normal_cap": self.normal_cap,
                       "reserve_cap": self.reserve_cap, "hard_cap": self.hard_cap,
                       "used_normal": 0, "used_reserve": 0, "refusals": 0, "events": []})

    def _persist(self, obj: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    @contextlib.contextmanager
    def _locked(self):
        lock = self.path.with_name(self.path.name + ".lock")
        deadline = time.time() + self.lock_timeout
        fd = None
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.time() >= deadline:
                    raise LedgerStateError(
                        f"could not acquire {lock} within {self.lock_timeout}s; fail closed, "
                        f"no lock recovery")
                time.sleep(0.005)
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(str(os.getpid()))
            yield
        finally:
            try:
                os.unlink(lock)
            except OSError:
                pass

    def charge(self, n: int, what: str, *, category: str = "normal") -> int:
        n = int(n)
        if n < 0:
            raise ValueError("charge amount must be >= 0")
        if category not in ("normal", "reserve"):
            raise ValueError(f"unknown category {category!r}")
        with self._locked():
            base = self._read_persisted()          # fresh validated durable base, inside the lock
            if self.mode == "production":
                self._cross_check_ledger(base)
            used = base["used_normal"] + base["used_reserve"]
            cat_key = "used_normal" if category == "normal" else "used_reserve"
            cat_cap = self.normal_cap if category == "normal" else self.reserve_cap
            reason = None
            if category == "normal" and base[cat_key] + n > cat_cap:
                reason = f"normal cap {cat_cap}"
            elif used + n > self.hard_cap:
                reason = f"hard cap {self.hard_cap}"
            if reason is not None:
                base["refusals"] += 1
                base["events"].append({"event": "refused", "what": str(what), "n": n,
                                       "category": category, "used_after": used, "unix": time.time()})
                self._persist(base)
                self.ledger.append(guard=self.kind, event="charge_refused", unit=self.unit,
                                   category=category, n=n, what=str(what), used_after=used,
                                   hard_cap=self.hard_cap, reason=reason)
                raise R74QuotaExhausted(
                    f"{self.kind}: refusing {n} ({what}, {category}); {reason}; used {used}/{self.hard_cap}")
            base[cat_key] += n
            new_used = base["used_normal"] + base["used_reserve"]
            base["events"].append({"event": "charged", "what": str(what), "n": n,
                                   "category": category, "used_after": new_used, "unix": time.time()})
            self._persist(base)
            self.ledger.append(guard=self.kind, event="charged", unit=self.unit, category=category,
                               n=n, what=str(what), used_after=new_used, hard_cap=self.hard_cap)
            return new_used

    def used(self) -> int:
        o = self._read_persisted()
        return o["used_normal"] + o["used_reserve"]

    def remaining(self) -> int:
        return self.hard_cap - self.used()

    def snapshot(self) -> dict:
        o = self._read_persisted()
        u = o["used_normal"] + o["used_reserve"]
        return {"kind": self.kind, "unit": self.unit, "mode": self.mode,
                "normal_cap": self.normal_cap, "reserve_cap": self.reserve_cap,
                "hard_cap": self.hard_cap, "used_normal": o["used_normal"],
                "used_reserve": o["used_reserve"], "used": u, "remaining": self.hard_cap - u,
                "refusals": o["refusals"]}


# ======================================================================================
# guards: same R74 caps AND same canonical kinds as the PHASE A receipt, so a PRODUCTION
# guard reads/continues the existing durable accounting (the 2 toy steps stay counted).
# ======================================================================================
class ProdEnvSlotGuard(_ProdCounter):
    def __init__(self, path, ledger, *, mode):
        super().__init__(path, mode=mode, ledger=ledger, normal_cap=579, reserve_cap=21,
                         hard_cap=600, kind="r74-env-construction-slot-guard/1",
                         unit="environment_construction")


class ProdSyntheticStepGuard(_ProdCounter):
    def __init__(self, path, ledger, *, mode):
        super().__init__(path, mode=mode, ledger=ledger, normal_cap=12, reserve_cap=4,
                         hard_cap=16, kind="r74-synthetic-physical-optimizer-step-guard/1",
                         unit="physical_optimizer_step")


class ProdOfflineForwardGuard(_ProdCounter):
    def __init__(self, path, ledger, *, mode):
        super().__init__(path, mode=mode, ledger=ledger, normal_cap=3000, reserve_cap=3000,
                         hard_cap=6000, kind="r74-offline-trunk-target-forward-guard/1",
                         unit="offline_trunk_target_row")
        self.per_batch_cap = 1024

    def charge_trunk_forward(self, matrix, what, *, category="normal"):
        """REAL trunk-entry metering seam (attach at controller.inner.trunk).

        Counts the ACTUAL 19-D rows handed to the trunk (a single (19,) input counts as 1). The
        per-batch 1024 cap is ALWAYS enforced here at the production entry and cannot be bypassed
        with a batch flag. A non-19 last dim raises via count_target_vectors and is never defaulted
        to zero, so a malformed matrix cannot evade the offline quota. The charge happens BEFORE the
        caller runs the real forward. Online/offline classification is by ESTABLISHED SOURCE: only
        this offline trunk-entry seam charges the offline budget; it is not a renameable flag.
        """
        n = count_target_vectors(matrix)          # raises on non-19 last dim
        if n > self.per_batch_cap:
            self.ledger.append(guard=self.kind, event="per_batch_refused", n=n,
                               per_batch_cap=self.per_batch_cap, what=str(what))
            raise R74QuotaExhausted(
                f"{self.kind}: batch of {n} target rows exceeds per-batch cap {self.per_batch_cap}")
        return self.charge(n, what, category=category)


class ProdJointRecordGuard(_ProdCounter):
    def __init__(self, path, ledger, *, mode):
        super().__init__(path, mode=mode, ledger=ledger, normal_cap=10**9, reserve_cap=10**9,
                         hard_cap=10**9, kind="r74-joint-record-guard/1", unit="joint_record")


class ProdPpoCallGuard(_ProdCounter):
    def __init__(self, path, ledger, *, mode):
        super().__init__(path, mode=mode, ledger=ledger, normal_cap=10**9, reserve_cap=10**9,
                         hard_cap=10**9, kind="r74-ppo-call-guard/1", unit="ppo_update_call")


# ======================================================================================
# unified per-physical-step pre-charge seam (real PPO / Adam / pristine / scratch steps)
# ======================================================================================
def make_charged_step(original_step, step_guard, *, what):
    """Wrap a REAL optimizer step so the physical-step guard is charged BEFORE it runs.

    Built on the ACTUAL r43 count_at_entry boundary (COUNTER_HOOK_VERSION), the same
    identity-preserving wrapper installed around torch.optim.Adam.step by configure(). The charge
    is inside the entry record, so:
      * a refused charge raises BEFORE the original step is entered (the step never runs);
      * the wrapper preserves args/results/exceptions, so an original step that raises has ALREADY
        been charged (no refund);
      * calling the wrapper N times (several steps inside one PPO update, or a pristine step
        restored around a gate) charges N times, one per physical step.

    Phase B installs this around the hooked torch.optim.Adam.step AND around the pristine
    _PRISTINE_ADAM_STEP temporarily restored around gate updates, so both the hooked and the
    hook-bypassing physical steps are metered by the same hard-16 budget.
    """
    def _record(*_args, **_kwargs):
        step_guard.charge(1, what)                # charge BEFORE the physical step; may refuse
    wrapped = count_at_entry(original_step, _record)
    return wrapped


from r74_guard import R74TotalLedger  # reuse the REAL append-only fsync total-ledger class


class R74ResourcesProd:
    """Production / fixture-separated R74 resource wiring (the path Phase B must use).

    production() is pinned to the canonical durable root and refuses any non-canonical / redirected
    root, so switching the output subdirectory can never mint fresh quota. It reads and CONTINUES
    the existing canonical accounting (the 2 real toy steps stay counted; 14 of 16 remain).
    fixture(root) uses a throwaway root + throwaway ledger, records SIMULATED consumption only, and
    can never act as production or reset production quota.
    """

    def __init__(self, *, mode, root=None):
        if mode not in ("production", "fixture"):
            raise ValueError("mode must be 'production' or 'fixture'")
        self.mode = mode
        if mode == "production":
            canonical = Path(R74_CANONICAL_ROOT)
            if root is not None and Path(root).resolve() != canonical.resolve():
                raise LedgerStateError(
                    "production resources are pinned to the canonical durable root; a non-canonical "
                    "root / redirected output is refused (it cannot mint fresh quota)")
            self.root = canonical
        else:
            if root is None:
                raise ValueError("fixture resources require an explicit throwaway root")
            self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger = R74TotalLedger(self.root / "r74_total_ledger.jsonl")
        self.env_slots = ProdEnvSlotGuard(self.root / "r74_env_slot_counter.json", self.ledger, mode=mode)
        self.synth_steps = ProdSyntheticStepGuard(self.root / "r74_synthetic_step_counter.json", self.ledger, mode=mode)
        self.offline_forward = ProdOfflineForwardGuard(self.root / "r74_offline_forward_counter.json", self.ledger, mode=mode)
        self.joint_records = ProdJointRecordGuard(self.root / "r74_joint_record_counter.json", self.ledger, mode=mode)
        self.ppo_calls = ProdPpoCallGuard(self.root / "r74_ppo_call_counter.json", self.ledger, mode=mode)

    @classmethod
    def production(cls):
        return cls(mode="production")

    @classmethod
    def fixture(cls, root):
        return cls(mode="fixture", root=root)

    def snapshots(self) -> dict:
        return {
            "mode": self.mode,
            "canonical_root": str(R74_CANONICAL_ROOT),
            "root": str(self.root),
            "pinned_to_canonical": self.root.resolve() == Path(R74_CANONICAL_ROOT).resolve(),
            "total_ledger_path": str(self.ledger.path),
            "total_ledger_events": len(self.ledger.events()),
            "env_slots": self.env_slots.snapshot(),
            "synthetic_physical_steps": self.synth_steps.snapshot(),
            "offline_forward_target_rows": self.offline_forward.snapshot(),
            "joint_records": self.joint_records.snapshot(),
            "ppo_calls": self.ppo_calls.snapshot(),
        }
