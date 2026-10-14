"""ROUND74 PHASE A third-delegation counter-example tests (pure fixture, zero real consumption).

Independent of test_r74_guard.py (original 17) and test_r74_guard_prod.py (27). NOT a re-run of
either. Covers ONLY the three narrow fixes applied to r74_guard_prod.py after the decision-end
verified commit 6a3fbac:

  1. The reserve-category cap is enforced INDEPENDENTLY of the normal cap and the hard cap; the 5th
     synthetic RESERVE charge is refused while the total is still < hard 16. make_charged_step
     routes to reserve ONLY by an explicit category="reserve" bound to a registered attempt; a
     normal gate refuses at the normal cap and never auto-upgrades to reserve.
  2. _cross_check_ledger fails closed on ANY counter/ledger mismatch: counter-ahead (used>ledger,
     e.g. a persist interruption), a non-zero counter with a missing ledger, and the existing
     low-backup (used<ledger). Consumed permission is preserved; nothing is refunded/recovered.
  3. R74ResourcesProd.fixture refuses the canonical durable root and any path inside it; real
     PHASE B must use production(), never an injected fixture.

Zero real consumption: no Adam.step, no model, no forward/backward, no PPO, no environment. Every
fixture lives under a self-cleaned throwaway dir that is a SIBLING of the canonical root and never
touches the canonical ledger / marker / 2 real toy steps.
"""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r74_guard_prod as G
from r74_guard_prod import (R74ResourcesProd, R74QuotaExhausted, LedgerStateError,
                            make_charged_step, ProdSyntheticStepGuard)
from r74_guard import R74TotalLedger

SYNTH_KIND = "r74-synthetic-physical-optimizer-step-guard/1"
SYNTH_UNIT = "physical_optimizer_step"


def run_tests():
    results = []
    def check(name, cond, **detail):
        results.append({"name": name, "pass": bool(cond), **({} if cond else {"detail": detail})})

    base = HERE / "_guard_revision_test"

    def fresh(sub):
        d = base / sub
        if d.exists():
            shutil.rmtree(d)
        (d / "runtime").mkdir(parents=True)
        return R74ResourcesProd.fixture(d / "runtime")

    # ===== FIX 1: reserve cap independent; explicit reserve routing ==========================
    # 1a. reserve-first: 4 reserve OK, 5th refused while total still < hard 16; normal still works.
    r = fresh("reserve_first")
    for i in range(4):
        r.synth_steps.charge(1, "r%d" % i, category="reserve")
    snap4 = r.synth_steps.snapshot()
    refused5 = False
    try:
        r.synth_steps.charge(1, "r4", category="reserve")
    except R74QuotaExhausted:
        refused5 = True
    snap5 = r.synth_steps.snapshot()
    after_norm = r.synth_steps.charge(1, "n0", category="normal")
    check("reserve_first_4_ok",
          snap4["used_reserve"] == 4 and snap4["used_normal"] == 0 and snap4["used"] == 4)
    check("reserve_first_5th_refused_total_below_hard",
          refused5 and snap5["used_reserve"] == 4 and snap5["used"] == 4
          and snap5["used"] < snap5["hard_cap"] == 16)
    check("reserve_first_normal_still_independent",
          after_norm == 5 and r.synth_steps.snapshot()["used_normal"] == 1)

    # 1b. normal cap independent of reserve: fill normal 12, 13th normal refused, reserve still open.
    r2 = fresh("normal_first")
    for i in range(12):
        r2.synth_steps.charge(1, "n%d" % i, category="normal")
    refused_norm13 = False
    try:
        r2.synth_steps.charge(1, "n12", category="normal")
    except R74QuotaExhausted:
        refused_norm13 = True
    snap_n13 = r2.synth_steps.snapshot()
    res_after = r2.synth_steps.charge(1, "r0", category="reserve")
    check("normal_cap_12_enforced",
          refused_norm13 and snap_n13["used_normal"] == 12 and snap_n13["used_reserve"] == 0)
    check("reserve_still_open_after_normal_cap",
          res_after == 13 and r2.synth_steps.snapshot()["used_reserve"] == 1)

    # 1c. make_charged_step routes to reserve ONLY via explicit category="reserve".
    r3 = fresh("step_reserve")
    entered = {"n": 0}
    def fake_step(*a, **k):
        entered["n"] += 1
        return "ok"
    w = make_charged_step(fake_step, r3.synth_steps, what="reserve_step", category="reserve")
    w()
    s3 = r3.synth_steps.snapshot()
    check("make_charged_step_explicit_reserve",
          entered["n"] == 1 and s3["used_reserve"] == 1 and s3["used_normal"] == 0)

    # 1d. a NORMAL gate refuses at the normal cap and never auto-upgrades to reserve.
    r4 = fresh("normal_no_autoupgrade")
    wn = make_charged_step(fake_step, r4.synth_steps, what="norm_step")  # default category normal
    for _ in range(12):
        wn()
    refused13 = False
    try:
        wn()
    except R74QuotaExhausted:
        refused13 = True
    s4 = r4.synth_steps.snapshot()
    check("normal_gate_refuses_no_reserve_autoupgrade",
          refused13 and s4["used_normal"] == 12 and s4["used_reserve"] == 0)

    # ===== FIX 2: _cross_check_ledger fails closed on ANY mismatch ===========================
    def prod_counter_with(counter_obj, ledger_charged_usedafters, sub):
        d = base / sub
        if d.exists():
            shutil.rmtree(d)
        (d / "runtime").mkdir(parents=True)
        ledger = R74TotalLedger(d / "runtime" / "r74_total_ledger.jsonl")
        for ua in ledger_charged_usedafters:
            ledger.append(guard=SYNTH_KIND, event="charged", unit=SYNTH_UNIT,
                          category="normal", n=1, what="seed", used_after=ua, hard_cap=16)
        cpath = d / "runtime" / "r74_synthetic_step_counter.json"
        cpath.write_text(json.dumps(counter_obj), encoding="utf-8")
        return ProdSyntheticStepGuard(cpath, ledger, mode="production")

    def synth_counter(used_normal, used_reserve):
        return {"kind": SYNTH_KIND, "unit": SYNTH_UNIT, "normal_cap": 12, "reserve_cap": 4,
                "hard_cap": 16, "used_normal": used_normal, "used_reserve": used_reserve,
                "refusals": 0, "events": []}

    # 2a. counter-ahead: counter used 3, ledger max used_after 2 -> fail closed.
    ahead = False
    try:
        prod_counter_with(synth_counter(3, 0), [1, 2], "xc_ahead")
    except LedgerStateError:
        ahead = True
    check("cross_check_counter_ahead_fails_closed", ahead)

    # 2b. low-backup: counter used 1, ledger max used_after 2 -> fail closed (existing case kept).
    low = False
    try:
        prod_counter_with(synth_counter(1, 0), [1, 2], "xc_low")
    except LedgerStateError:
        low = True
    check("cross_check_low_backup_fails_closed", low)

    # 2c. non-zero counter with a missing ledger (no charged events for this kind) -> fail closed.
    miss = False
    try:
        prod_counter_with(synth_counter(2, 0), [], "xc_missing")
    except LedgerStateError:
        miss = True
    check("cross_check_nonzero_counter_missing_ledger_fails_closed", miss)

    # 2d. a consistent counter/ledger pair is ACCEPTED in production mode (no false positive).
    ok_accept = True
    try:
        c = prod_counter_with(synth_counter(2, 0), [1, 2], "xc_ok")
        if c.used() != 2:
            ok_accept = False
    except LedgerStateError:
        ok_accept = False
    check("cross_check_consistent_pair_accepted", ok_accept)

    # ===== FIX 3: fixture refuses the canonical root and any path inside it ==================
    canon = G.R74_CANONICAL_ROOT
    refused_canon = False
    try:
        R74ResourcesProd.fixture(canon)
    except LedgerStateError:
        refused_canon = True
    check("fixture_refuses_canonical_root", refused_canon)

    refused_sub = False
    try:
        R74ResourcesProd.fixture(canon / "some_subdir")
    except LedgerStateError:
        refused_sub = True
    check("fixture_refuses_path_inside_canonical", refused_sub)

    legit = fresh("legit_sibling")
    check("fixture_legit_sibling_works",
          legit.mode == "fixture" and legit.root.resolve() != Path(canon).resolve())

    shutil.rmtree(base, ignore_errors=True)
    n_pass = sum(1 for x in results if x["pass"])
    return {"total": len(results), "pass": n_pass, "fail": len(results) - n_pass,
            "all_pass": n_pass == len(results), "results": results}


if __name__ == "__main__":
    s = run_tests()
    for x in s["results"]:
        print(("PASS" if x["pass"] else "FAIL"), x["name"], "" if x["pass"] else x.get("detail"))
    print("R74_REVISION_TESTS", "ALL_PASS" if s["all_pass"] else "FAILED:%d" % s["fail"])
    raise SystemExit(0 if s["all_pass"] else 1)