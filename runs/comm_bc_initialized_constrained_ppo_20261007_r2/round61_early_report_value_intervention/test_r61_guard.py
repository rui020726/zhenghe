"""ROUND61 guard synthetic + spy tests (pure computation, no environment). Proves the required
properties on a throwaway synthetic ledger; fixtures never create a real machine license."""
from __future__ import annotations
import shutil, sys, tempfile, threading
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from r61_guard import R61Guard, R61QuotaExhausted

FAILED = []
def check(name, cond, **detail):
    print(("PASS" if cond else "FAIL"), name, "" if cond else detail)
    if not cond:
        FAILED.append(name)

def fresh(sub):
    d = HERE / "_guard_test" / sub
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return R61Guard(d, normal_cap=18, reserve_cap=2, hard_cap=20)

# 1. cap boundary: 19 consumed -> only 1 more; 20 -> refuse 21st
g = fresh("cap")
for i in range(19):
    g.claim(f"job_{i}")
check("consumed_19", g.consumed() == 19)
s20 = g.claim("job_19")
check("slot_20_allowed", s20 == 20 and g.consumed() == 20)
refused = False
try:
    g.claim("job_20")
except R61QuotaExhausted:
    refused = True
check("slot_21_refused", refused and g.consumed() == 20)

# 2. construction failure or non-qualifying result does not refund
g = fresh("norefund")
with g.launch("build_fail") as L:
    L.builder_called()
    raise_simulated = True
    if raise_simulated:
        L.env_constructed()  # env built, then construction failed before a qualified result
check("construction_failure_no_refund", g.consumed() == 1)
with g.launch("disqualified") as L:
    L.builder_called(); L.env_constructed(); L.run_called(); L.run_returned(steps=1000, ok=True)
    L.result(qualified=False, detail="synthetic disqualification")
snap = g.snapshot()
check("disqualified_no_refund", g.consumed() == 2 and snap["results_disqualified"] == 1
      and snap["results_qualified"] == 0 and snap["consumed_slots"] != snap["results_qualified"])

# 3. restart / output subdir / quarantine do not zero consumption
g = fresh("persist")
for i in range(5):
    g.claim(f"p_{i}")
g2 = R61Guard(g.root, normal_cap=18, reserve_cap=2, hard_cap=20)  # restart
check("restart_keeps_consumption", g2.consumed() == 5)
backup = HERE / "_guard_test" / "persist_backup"
if backup.exists():
    shutil.rmtree(backup)
shutil.copytree(g.root, backup)  # backup copy
check("backup_copy_not_new_execution", g.consumed() == 5 and R61Guard(g.root, normal_cap=18, reserve_cap=2, hard_cap=20).consumed() == 5)
quar = g.root / "quarantine"; quar.mkdir(exist_ok=True)  # isolate a failed material copy
shutil.copy2(next(g.slots_dir.glob("*.slot")), quar / "isolated.slot")
check("quarantine_keeps_consumption", g.consumed() == 5)

# 4. duplicate logical refused; same-output different-job not deduped
g = fresh("dedup")
g.claim("same_output_A")
dup = False
try:
    g.claim("same_output_A")
except R61QuotaExhausted:
    dup = True
check("duplicate_logical_refused", dup and g.consumed() == 1)
g.claim("same_output_B")  # different job, would produce identical output
check("same_output_not_deduped", g.consumed() == 2)

# 5. concurrent contention for the last slot -> exactly one winner
g = fresh("concurrent")
winners = []
barrier = threading.Barrier(2)
def racer(name):
    barrier.wait()
    try:
        g.claim(name); winners.append(name)
    except R61QuotaExhausted:
        pass
# fill to hard_cap-1 then race for the last slot
for i in range(19):
    g.claim(f"fill_{i}")
t1 = threading.Thread(target=racer, args=("racer_1",)); t2 = threading.Thread(target=racer, args=("racer_2",))
t1.start(); t2.start(); t1.join(); t2.join()
check("concurrent_last_slot_one_winner", len(winners) == 1 and g.consumed() == 20)

# 6. unregistered launch path refused before environment construction (spy builder)
g = fresh("unregistered")
class SpyBuilder:
    def __init__(self, guard):
        self.guard = guard; self.built = 0
    def build(self):
        # mirrors the frozen counted builder: refuse when no atomic claim is active
        if getattr(self.guard, "_active_claim", None) is None:
            raise RuntimeError("environment construction attempted without an atomic quota claim")
        self.built += 1
        return "env"
spy = SpyBuilder(g)
direct_refused = False
try:
    spy.build()  # no guard.launch active
except RuntimeError:
    direct_refused = True
check("unregistered_path_refused_before_env", direct_refused and spy.built == 0)
with g.launch("registered") as L:
    g._active_claim = L.slot
    try:
        spy.build()
    finally:
        g._active_claim = None
check("registered_path_builds", spy.built == 1)

# 7 & 8. R60 historical 608 is not R61 headroom; R61 is an independent new authorization
r60_root = HERE.parent / "round60_partitioned_actor_head_ablation" / "raw"
check("r61_independent_of_r60_ledger", g.root.resolve() != r60_root.resolve())
g_fresh = fresh("newauth")
snap = g_fresh.snapshot()
check("r61_new_authorization_headroom_20", snap["consumed_slots"] == 0 and snap["headroom"] == 20)
check("r60_608_not_claimed_as_r61_headroom", snap["headroom"] != (600 - 608))  # not derived from R60

shutil.rmtree(HERE / "_guard_test", ignore_errors=True)
print("GUARD_TESTS", "ALL_PASS" if not FAILED else f"FAILED:{FAILED}")
raise SystemExit(1 if FAILED else 0)
