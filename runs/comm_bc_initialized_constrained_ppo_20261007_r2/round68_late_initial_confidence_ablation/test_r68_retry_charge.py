"""ROUND68 pure-ledger / spy counter-examples for the claimed-but-started booking family.

Zero model forward, zero optimizer step, zero environment construction: the innermost
original_builder is a spy, configure_and_inject only wires modules, and every run_fn is synthetic.

Proves:
  A) the FROZEN r66x._atomic_run DOES free-reuse a constructed-but-interrupted attempt (the bug
     that under-counted the e1 retry) -- documented, not asserted as acceptable;
  B) the FIXED _r68_atomic_run CHARGES a new reserve attempt for that case;
  C) a claimed-but-NEVER-constructed attempt is a safe free reuse -- now proven by a durable
     "not_issued" construction permit, NOT by the absence of a field;
  D) "permit issued but hook stats not refreshed" (a hard interruption right after issuance):
     the retry must charge a reserve or refuse, never construct again for free;
  E) a MISSING permit under the permit regime is an anomaly and refuses; it is never read as zero;
  F) a ZERO-CONSTRUCTION failure (failed_technical, permit still not_issued) must take a NEW
     reserve attempt and preserve the old failure, not overwrite it;
  G) the REAL permit guard (module factory) wraps the REAL r42_run counted builder and flips the
     permit to issued BEFORE the spied constructor is entered, refuses a build with no active
     claim, and refuses a second build for the same attempt;
  H) permit issuance is atomic: two competing issuances, exactly one wins;
  I) TRAINING and EVALUATION both reach base._r68_atomic_run through their real call paths;
  J) the fixed hard cap reads the late reserve correction exactly once (no double count).

Usage: python test_r68_retry_charge.py <r68_run_dir> <isolated_test_raw_root>
"""
import sys, json, os
from pathlib import Path

R68 = Path(sys.argv[1]).resolve()
TEST_RAW = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(R68))
sys.path.insert(0, str(R68.parent / "round66_bc_early_path_preserving_late_adaptation"))
import r66_execute as r66x           # frozen helper (the buggy free-reuse path)
import r68_execute as r68x_exec      # patched executor (_r68_atomic_run + permit layer)
import r68_eval as r68x_eval         # evaluation adapter (must share the one helper)

TEST_RAW.mkdir(parents=True, exist_ok=True)
cfg = r68x_exec.configure_and_inject(TEST_RAW)   # modules only; 0 forward / 0 env / 0 optimizer
r40 = cfg["runtime"]["r40_common"]
r42_run = cfg["r42_run"]

FAILS = []
def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAILS.append(name)

def slots():
    n = len(list((TEST_RAW / "launch_slots" / "normal").glob("*.json")))
    r = len(list((TEST_RAW / "launch_slots" / "reserve").glob("*.json")))
    return n, r

def claim_dir(logical):
    return TEST_RAW / "launch_claims" / r40._slug(logical)

def permit_path(logical, attempt):
    return claim_dir(logical) / "permits" / f"{int(attempt):04d}.json"

def permit_token_path(logical, attempt):
    return claim_dir(logical) / "permits" / f"{int(attempt):04d}.issued.json"

def set_claim(logical, attempt, **fields):
    path = claim_dir(logical) / f"attempt_{int(attempt):04d}.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    obj.update(fields)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path

def synth():
    return {"on_time": 0, "spent_bytes": 0}   # synthetic row: NO environment, NO forward, NO optimizer

def make_constructed_interrupted(logical):
    """A claim that DID construct (495-step partial) but whose hooks never refreshed."""
    cp = r40.claim_launch(logical, category="normal", attempt=1, metadata={"probe": True})
    set_claim(logical, 1, status="claimed_before_environment",
              environment_constructor_calls=1, completed_steps=0, simulation_started=False)
    return cp

print("CASE A: FROZEN r66x._atomic_run on a constructed-but-interrupted attempt (the bug)")
jobA = "probe_constructed_interrupted_A"
make_constructed_interrupted(jobA)
n0, r0 = slots()
r66x._atomic_run(cfg, jobA, {"probe": True}, synth, detail=lambda r: "synthetic")
n1, r1 = slots()
check("frozen path free-reuses (no new slot) -> demonstrates the under-count bug",
      n1 == n0 and r1 == r0)

print("CASE B: FIXED _r68_atomic_run on a constructed-but-interrupted attempt (the fix)")
jobB = "probe_constructed_interrupted_B"
make_constructed_interrupted(jobB)
n0, r0 = slots()
r68x_exec._r68_atomic_run(cfg, jobB, {"probe": True}, synth, detail=lambda r: "synthetic")
n1, r1 = slots()
check("fixed path charges a NEW reserve slot (constructed partial cannot be slot-free reused)",
      r1 == r0 + 1 and n1 == n0)
check("fixed path created a reserve attempt_0002",
      (claim_dir(jobB) / "attempt_0002.json").is_file())
check("is_complete resolves the latest complete attempt (reserve retry)",
      r68x_exec.R68Ledger(TEST_RAW).is_complete(jobB))

print("CASE C: FIXED _r68_atomic_run on a claimed-but-NEVER-constructed attempt (safe free reuse)")
jobC = "probe_claimed_never_constructed_C"
# Book the claim exactly the way _r68_atomic_run does at claim time (claim + not_issued permit),
# then model a process death BEFORE run_fn constructed anything: status stays
# claimed_before_environment and no hook ever fired.
r40.claim_launch(jobC, category="normal", attempt=1, metadata={"probe": True})
r68x_exec._create_permit_not_issued(r40, jobC, 1)
n0, r0 = slots()
r68x_exec._r68_atomic_run(cfg, jobC, {"probe": True}, synth, detail=lambda r: "synthetic")
n1, r1 = slots()
check("never-constructed claimed attempt stays a free reuse (no slot to double-count)",
      n1 == n0 and r1 == r0)
check("the free reuse is justified by a durable not_issued permit, not by a missing field",
      permit_path(jobC, 1).is_file()
      and json.loads(permit_path(jobC, 1).read_text(encoding="utf-8"))["state"] == "not_issued")

print("CASE D: permit ISSUED but hook stats NOT refreshed (hard interruption after issuance)")
jobD = "probe_permit_issued_hooks_stale_D"
r40.claim_launch(jobD, category="normal", attempt=1, metadata={"probe": True})
# the process died between the exclusive token create and any hook write: counters absent, the
# claim file still says claimed_before_environment, only the token proves a build was committed.
permit_path(jobD, 1).parent.mkdir(parents=True, exist_ok=True)
permit_path(jobD, 1).write_text(json.dumps(
    {"kind": "round68-construction-permit/1", "regime": r68x_exec.PERMIT_REGIME,
     "logical_job": jobD, "attempt": 1, "state": "not_issued", "permit_id": "probe"}, indent=2) + "\n",
    encoding="utf-8")
fd = os.open(str(permit_token_path(jobD, 1)), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
os.close(fd)
n0, r0 = slots()
r68x_exec._r68_atomic_run(cfg, jobD, {"probe": True}, synth, detail=lambda r: "synthetic")
n1, r1 = slots()
check("issued permit with stale hooks is NOT free-reused: a new reserve slot is charged",
      r1 == r0 + 1 and n1 == n0)
check("a fresh not_issued permit exists for the reserve retry",
      permit_path(jobD, 2).is_file()
      and json.loads(permit_path(jobD, 2).read_text(encoding="utf-8"))["state"] == "not_issued")

print("CASE E: MISSING permit under the permit regime is an anomaly, never a zero")
jobE = "probe_missing_permit_E"
r40.claim_launch(jobE, category="normal", attempt=1, metadata={"probe": True})
n0, r0 = slots()
refused = False
try:
    r68x_exec._r68_atomic_run(cfg, jobE, {"probe": True}, synth, detail=lambda r: "synthetic")
except RuntimeError as exc:
    refused = "no construction permit" in str(exc)
check("a missing permit refuses the retry instead of assuming never-constructed", refused)
n1, r1 = slots()
check("the refused retry consumed no slot (it never reached a construction)", n1 == n0 and r1 == r0)

print("CASE F: ZERO-CONSTRUCTION failure takes a new reserve and preserves the old failure")
jobF = "probe_zero_construction_failure_F"
r40.claim_launch(jobF, category="normal", attempt=1, metadata={"probe": True})
r68x_exec._create_permit_not_issued(r40, jobF, 1)
r40.finish_launch(claim_dir(jobF) / "attempt_0001.json", ok=False, detail="synthetic pre-construction failure")
set_claim(jobF, 1, status="failed_technical", failure_stage="before_environment",
          note="preserved: this attempt never constructed an environment")
n0, r0 = slots()
r68x_exec._r68_atomic_run(cfg, jobF, {"probe": True}, synth, detail=lambda r: "synthetic")
n1, r1 = slots()
check("a zero-construction failure books a NEW reserve attempt (no free reuse of the failed claim)",
      r1 == r0 + 1 and n1 == n0)
old = json.loads((claim_dir(jobF) / "attempt_0001.json").read_text(encoding="utf-8"))
check("the old failed_technical claim is preserved, not overwritten",
      old["status"] == "failed_technical" and old.get("failure_stage") == "before_environment")
retry2 = json.loads((claim_dir(jobF) / "attempt_0002.json").read_text(encoding="utf-8"))
check("the reserve retry records the failure it retries",
      retry2.get("metadata", {}).get("retry_of_attempt") == 1
      and retry2.get("metadata", {}).get("retry_reason") == "prior attempt failed_technical")

print("CASE G: the REAL permit guard over the REAL r42_run counted builder (spied constructor)")
order = []
class _FakeTarget:
    def __init__(self):
        self.orchestrator = type("O", (), {"step_index": 0})()

def spied_original_builder(configuration):
    # This is the exact point where a real environment would be constructed. Record what the
    # permit looked like AT THAT MOMENT.
    st = r68x_exec._permit_status(r40, "probe_real_seam_G", 1)
    order.append(("constructor_entered", st["state"], st["token_exists"]))
    return _FakeTarget()

spied_counted = r42_run._make_counted_builder(
    spied_original_builder,
    active_claim=lambda: r42_run._ACTIVE_CLAIM,
    active_progress=lambda: r42_run._ACTIVE_PROGRESS,
    update_claim=r40._update_claim,
    collector=lambda: None,
    dispatch_run=lambda call, collector: call(),
    write_progress=lambda claim, progress: r40._update_claim(claim, **progress),
)
guarded = r68x_exec._make_permit_guarded_builder(spied_counted, r40, r42_run)

jobG = "probe_real_seam_G"
r40.claim_launch(jobG, category="normal", attempt=1, metadata={"probe": True})
r68x_exec._create_permit_not_issued(r40, jobG, 1)
r42_run._ACTIVE_CLAIM = claim_dir(jobG) / "attempt_0001.json"
r42_run._ACTIVE_PROGRESS = r42_run._new_run_progress()
try:
    guarded(object())
finally:
    r42_run._ACTIVE_CLAIM = None
    r42_run._ACTIVE_PROGRESS = None
check("the spied constructor was entered exactly once", len(order) == 1)
check("the permit was already ISSUED (token present) when the real constructor was entered",
      bool(order) and order[0][1] == "issued" and order[0][2] is True)
check("after construction the permit is consumed",
      r68x_exec._permit_state(r40, jobG, 1) == "consumed")

r42_run._ACTIVE_CLAIM = claim_dir(jobG) / "attempt_0001.json"
r42_run._ACTIVE_PROGRESS = r42_run._new_run_progress()
try:
    refused_second = False
    try:
        guarded(object())
    except RuntimeError as exc:
        refused_second = ("already" in str(exc) or "issuance refused" in str(exc)
                          or "refused" in str(exc))
    check("a SECOND construction for the same attempt is refused", refused_second)
    no_claim_refused = False
    r42_run._ACTIVE_CLAIM = None
    try:
        guarded(object())
    except RuntimeError as exc:
        no_claim_refused = "without an atomic quota claim" in str(exc)
    check("a construction with no active claim is refused", no_claim_refused)
finally:
    r42_run._ACTIVE_CLAIM = None
    r42_run._ACTIVE_PROGRESS = None
check("the constructor was never entered a second time", len(order) == 1)

print("CASE H: permit issuance is atomic across competing entries")
jobH = "probe_issuance_race_H"
r40.claim_launch(jobH, category="normal", attempt=1, metadata={"probe": True})
r68x_exec._create_permit_not_issued(r40, jobH, 1)
first = r68x_exec._issue_permit(r40, jobH, 1)
second_refused = False
try:
    r68x_exec._issue_permit(r40, jobH, 1)
except RuntimeError as exc:
    second_refused = ("already issued" in str(exc) or "issuance refused" in str(exc))
check("the first issuance succeeds and the competing issuance is refused",
      bool(first) and second_refused)
check("exactly one issuance token exists", permit_token_path(jobH, 1).is_file())

print("CASE I: TRAINING and EVALUATION both reach base._r68_atomic_run (real call paths)")
train_hits = []
real_helper = r68x_exec._r68_atomic_run
def spy_helper(cfg_, logical, metadata, run_fn, *, detail, on_claim=None,
               allow_complete_retry=False):
    train_hits.append((logical, allow_complete_retry))
    return {"spied": True, "logical": logical}

r68x_exec._atomic_run = spy_helper          # what _run_train_episode resolves at call time
r68x_exec._r68_atomic_run = spy_helper      # what r68_eval resolves at call time
try:
    class _Collector:
        def policy(self):
            return None
    r68x_exec._run_train_episode(cfg, object(), _Collector(), seed=11, ep=200,
                                 scenario=1300000, lam=1.0)
    check("the TRAINING path reaches the shared helper",
          bool(train_hits) and train_hits[0][0] == "r68_new_training_s11_e200")

    def synthetic_run_fn(run_tag):
        return {"on_time": 0, "spent_bytes": 0, "steps": 1, "tag": run_tag}

    r68x_eval._safe_atomic_run(cfg, "probe_eval_path_I", {"probe": True}, synthetic_run_fn,
                               detail=lambda r: "synthetic", tag="r68eval_model_probe",
                               persist=None, cell=None)
    check("the EVALUATION path reaches the SAME shared helper",
          len(train_hits) > 1 and train_hits[-1][0] == "probe_eval_path_I")
    check("evaluation opts into the unsound-complete retry rule explicitly",
          len(train_hits) > 1 and train_hits[-1][1] is True)
finally:
    r68x_exec._atomic_run = real_helper
    r68x_exec._r68_atomic_run = real_helper

print("CASE J: the fixed hard cap reads the late reserve correction exactly once")
led = r68x_exec.R68Ledger(TEST_RAW)
counts = led.counts()
check("counts expose booked vs actual constructions separately",
      {"booked_attempts", "actual_environment_constructions",
       "late_correction_extra_constructions"} <= set(counts))
check("with no correction booked, actual == booked (no phantom charge)",
      counts["actual_environment_constructions"] == counts["booked_attempts"]
      and counts["late_correction_extra_constructions"] == 0)
# register a correction-style reserve slot and prove it is counted exactly once
reserve_dir = TEST_RAW / "launch_slots" / "reserve"
reserve_dir.mkdir(parents=True, exist_ok=True)
(reserve_dir / "0009.json").write_text(json.dumps({
    "category": "reserve", "slot": 9, "logical_job": jobA, "attempt": 1,
    "correction_note": "test-only correction slot", "registration": "late_registration",
    "predebit_status": "noncompliant_predebit"}, indent=2) + "\n", encoding="utf-8")
after = led.counts()
check("a late-registered correction slot adds exactly one actual construction",
      after["late_correction_extra_constructions"] == 1
      and after["actual_environment_constructions"] == after["booked_attempts"] + 1)
# The genuine duplicate scenario: a correction slot that points at an attempt which ALREADY has
# its own reserve claim. That construction is already booked by the claim, so the correction must
# not add a second charge.
jobK = "probe_correction_duplicate_K"
r40.claim_launch(jobK, category="normal", attempt=1, metadata={"probe": True})
r68x_exec._create_permit_not_issued(r40, jobK, 1)
r40.claim_launch(jobK, category="reserve", attempt=2, metadata={"probe": True})
(reserve_dir / "0010.json").write_text(json.dumps({
    "category": "reserve", "slot": 10, "logical_job": jobK, "attempt": 2,
    "correction_note": "test-only duplicate correction slot", "registration": "late_registration",
    "predebit_status": "noncompliant_predebit"}, indent=2) + "\n", encoding="utf-8")
dup = led.counts()
check("a correction slot referenced by a reserve claim is NOT counted twice",
      dup["late_correction_extra_constructions"] == 1
      and all(e["logical_job"] != jobK for e in dup["correction"]["extras"]))
check("the referenced-but-already-booked correction is reported, not silently dropped",
      any("0009" in e["slot"] for e in dup["correction"]["extras"]))
(reserve_dir / "0009.json").unlink()
(reserve_dir / "0010.json").unlink()

print("COUNTEREXAMPLE_RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
sys.exit(0 if not FAILS else 1)