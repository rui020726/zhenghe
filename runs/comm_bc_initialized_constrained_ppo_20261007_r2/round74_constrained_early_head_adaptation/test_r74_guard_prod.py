"""ROUND74 PHASE A pure-fixture spy tests for the PRODUCTION seams (r74_guard_prod.py).

Zero real consumption: no Adam.step, no model, no forward/backward, no PPO, no environment. Every
guard is a FIXTURE on a throwaway root under _guard_prod_test/ removed at the end; the canonical
ledger / marker / 2 real toy steps are never touched. SEPARATE from the original 17
(test_r74_guard.py) and NOT a re-run of them.
"""
from __future__ import annotations
import json, shutil, sys, threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r74_guard_prod as G
from r74_guard_prod import (R74ResourcesProd, R74QuotaExhausted, LedgerStateError,
                            count_target_vectors, make_charged_step, _ProdCounter,
                            ProdSyntheticStepGuard)
from r74_guard import R74TotalLedger


class Fake:
    def __init__(self, shape): self.shape = shape


def run_tests():
    results = []
    def check(name, cond, **detail):
        results.append({"name": name, "pass": bool(cond), **({} if cond else {"detail": detail})})

    def fresh(sub):
        d = HERE / "_guard_prod_test" / sub
        if d.exists():
            shutil.rmtree(d)
        (d / "runtime").mkdir(parents=True)
        return R74ResourcesProd.fixture(d / "runtime")

    # ---- item 3: count_target_vectors fix + non-19 rejection ------------------------------
    check("ctv_1d_19_is_one", count_target_vectors(Fake((19,))) == 1)
    check("ctv_2d_8x19", count_target_vectors(Fake((8, 19))) == 8)
    check("ctv_3d_4x5x19", count_target_vectors(Fake((4, 5, 19))) == 20)
    for bad in (Fake((8, 7)), Fake((19, 7)), Fake(()), Fake((3,))):
        try:
            count_target_vectors(bad); ok = False
        except LedgerStateError:
            ok = True
        check("ctv_non19_raises_%s" % (bad.shape,), ok)

    # ---- item 1: per-physical-step seam through the REAL count_at_entry -------------------
    res = fresh("step_seam")
    entered = {"n": 0}
    def fake_step(*a, **k):
        entered["n"] += 1
        return "ok"
    wrapped = make_charged_step(fake_step, res.synth_steps, what="ppo_internal_step")
    check("seam_is_real_count_at_entry_boundary", getattr(wrapped, "__wrapped__", None) is fake_step)
    for _ in range(3):
        wrapped()
    check("seam_multistep_each_charged", res.synth_steps.used() == 3 and entered["n"] == 3)

    res2 = fresh("step_refuse")
    hit = {"n": 0}
    def spy_step(*a, **k):
        hit["n"] += 1; return "ok"
    w2 = make_charged_step(spy_step, res2.synth_steps, what="s")
    for _ in range(12):
        w2()
    before = hit["n"]
    refused = False
    try:
        w2()
    except R74QuotaExhausted:
        refused = True
    check("seam_refusal_before_step", refused and hit["n"] == before and res2.synth_steps.used() == 12)

    res3 = fresh("step_fail")
    def failing_step(*a, **k):
        raise RuntimeError("boom")
    w3 = make_charged_step(failing_step, res3.synth_steps, what="s")
    before = res3.synth_steps.used()
    raised = False
    try:
        w3()
    except RuntimeError:
        raised = True
    check("seam_step_failure_no_refund", raised and res3.synth_steps.used() == before + 1)

    # ---- item 3: trunk-forward metering seam ----------------------------------------------
    res4 = fresh("trunk")
    res4.offline_forward.charge_trunk_forward(Fake((8, 19)), "b8")
    check("trunk_8x19_charges_8", res4.offline_forward.used() == 8)
    res4.offline_forward.charge_trunk_forward(Fake((19,)), "single")
    check("trunk_1d19_charges_1", res4.offline_forward.used() == 9)
    over = False
    try:
        res4.offline_forward.charge_trunk_forward(Fake((1025, 19)), "toobig")
    except R74QuotaExhausted:
        over = True
    check("trunk_per_batch_1024_enforced", over and res4.offline_forward.used() == 9)
    sneak = False
    try:
        res4.offline_forward.charge_trunk_forward(Fake((50, 7)), "non19")
    except LedgerStateError:
        sneak = True
    check("trunk_non19_refused", sneak and res4.offline_forward.used() == 9)
    check("trunk_only_offline_unit_moves",
          res4.env_slots.used() == 0 and res4.ppo_calls.used() == 0 and res4.joint_records.used() == 0)
    # ---- item 2: two instances, no lost update (durable-base-inside-lock) -----------------
    shared = HERE / "_guard_prod_test" / "shared"
    if shared.exists():
        shutil.rmtree(shared)
    (shared / "runtime").mkdir(parents=True)
    A = R74ResourcesProd.fixture(shared / "runtime")
    B = R74ResourcesProd.fixture(shared / "runtime")   # second instance, same durable files
    A.synth_steps.charge(5, "A")
    B.synth_steps.charge(4, "B")
    C = R74ResourcesProd.fixture(shared / "runtime")    # fresh reader sees the SUM, not max
    check("two_instances_no_lost_update", C.synth_steps.used() == 9)

    # ---- item 2: concurrent race for the last slot -> exactly one succeeds ----------------
    race = HERE / "_guard_prod_test" / "race" / "runtime"
    race.mkdir(parents=True)
    led = R74TotalLedger(race / "led.jsonl")
    cap1 = _ProdCounter(race / "cap1.json", mode="fixture", ledger=led, normal_cap=1,
                        reserve_cap=0, hard_cap=1, kind="race/1", unit="physical_optimizer_step")
    outcomes = {"ok": 0, "refused": 0}
    olock = threading.Lock()
    def racer():
        try:
            cap1.charge(1, "race"); 
            with olock: outcomes["ok"] += 1
        except R74QuotaExhausted:
            with olock: outcomes["refused"] += 1
    t1 = threading.Thread(target=racer); t2 = threading.Thread(target=racer)
    t1.start(); t2.start(); t1.join(); t2.join()
    check("concurrent_last_slot_only_one_succeeds",
          outcomes["ok"] == 1 and outcomes["refused"] == 1 and cap1.used() == 1)

    # ---- item 2: fail-closed on inconsistent durable state --------------------------------
    def write_json(p, obj):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj), encoding="utf-8")
    def mk_counter(dirname, **over):
        d = HERE / "_guard_prod_test" / dirname / "runtime"; d.mkdir(parents=True, exist_ok=True)
        led2 = R74TotalLedger(d / "led.jsonl")
        base = dict(path=d / "c.json", mode="fixture", ledger=led2, normal_cap=5, reserve_cap=2,
                    hard_cap=7, kind="k/1", unit="u")
        base.update(over)
        return _ProdCounter(**base)

    good = {"kind": "k/1", "unit": "u", "normal_cap": 5, "reserve_cap": 2, "hard_cap": 7,
            "used_normal": 1, "used_reserve": 0, "refusals": 0, "events": []}
    # corrupt
    d = HERE / "_guard_prod_test" / "corrupt" / "runtime"; d.mkdir(parents=True, exist_ok=True)
    (d / "c.json").write_text("{ not json", encoding="utf-8")
    try:
        _ProdCounter(path=d / "c.json", mode="fixture", ledger=R74TotalLedger(d / "l.jsonl"),
                     normal_cap=5, reserve_cap=2, hard_cap=7, kind="k/1", unit="u"); corrupt_ok = False
    except LedgerStateError:
        corrupt_ok = True
    check("failclosed_corrupt", corrupt_ok)
    # partial (missing key)
    part = dict(good); del part["used_normal"]
    try:
        p = HERE / "_guard_prod_test" / "partial" / "runtime" / "c.json"
        write_json(p, part)
        _ProdCounter(path=p, mode="fixture", ledger=R74TotalLedger(p.parent / "l.jsonl"),
                     normal_cap=5, reserve_cap=2, hard_cap=7, kind="k/1", unit="u"); part_ok = False
    except LedgerStateError:
        part_ok = True
    check("failclosed_partial_missing_key", part_ok)
    # foreign identity
    foreign = dict(good); foreign["kind"] = "other/9"
    try:
        p = HERE / "_guard_prod_test" / "foreign" / "runtime" / "c.json"
        write_json(p, foreign)
        _ProdCounter(path=p, mode="fixture", ledger=R74TotalLedger(p.parent / "l.jsonl"),
                     normal_cap=5, reserve_cap=2, hard_cap=7, kind="k/1", unit="u"); f_ok = False
    except LedgerStateError:
        f_ok = True
    check("failclosed_foreign_identity", f_ok)
    # used > hard cap
    over = dict(good); over["used_normal"] = 99
    try:
        p = HERE / "_guard_prod_test" / "over" / "runtime" / "c.json"
        write_json(p, over)
        _ProdCounter(path=p, mode="fixture", ledger=R74TotalLedger(p.parent / "l.jsonl"),
                     normal_cap=5, reserve_cap=2, hard_cap=7, kind="k/1", unit="u"); o_ok = False
    except LedgerStateError:
        o_ok = True
    check("failclosed_used_exceeds_hard", o_ok)
    # lost PRODUCTION counter (file missing, ledger shows prior charges) -> refuse, not reset
    d = HERE / "_guard_prod_test" / "lostprod" / "runtime"; d.mkdir(parents=True, exist_ok=True)
    ledp = R74TotalLedger(d / "led.jsonl")
    ledp.append(guard="k/1", event="charged", unit="u", category="normal", n=1, what="prior",
                used_after=1, hard_cap=7)
    try:
        _ProdCounter(path=d / "c.json", mode="production", ledger=ledp, normal_cap=5, reserve_cap=2,
                     hard_cap=7, kind="k/1", unit="u"); lost_ok = False
    except LedgerStateError:
        lost_ok = True
    check("failclosed_lost_production_counter", lost_ok and not (d / "c.json").exists())
    # low backup (counter used below ledger-recorded max) -> refuse
    d = HERE / "_guard_prod_test" / "lowbackup" / "runtime"; d.mkdir(parents=True, exist_ok=True)
    ledb = R74TotalLedger(d / "led.jsonl")
    ledb.append(guard="k/1", event="charged", unit="u", category="normal", n=2, what="prior",
                used_after=2, hard_cap=7)
    write_json(d / "c.json", dict(good, used_normal=1))    # stale/low backup claims only 1
    try:
        _ProdCounter(path=d / "c.json", mode="production", ledger=ledb, normal_cap=5, reserve_cap=2,
                     hard_cap=7, kind="k/1", unit="u"); lb_ok = False
    except LedgerStateError:
        lb_ok = True
    check("failclosed_low_backup", lb_ok)
    # residual lock fails closed (no recovery)
    d = HERE / "_guard_prod_test" / "reslock" / "runtime"; d.mkdir(parents=True, exist_ok=True)
    c = mk_counter("reslock")
    (d / "c.json.lock").write_text("999999", encoding="utf-8")   # simulate a residual lock
    rl = False
    try:
        c.charge(1, "x")
    except LedgerStateError:
        rl = True
    check("failclosed_residual_lock", rl)

    # ---- item 2: production pins canonical; a non-canonical root is refused ---------------
    noncanon = False
    try:
        R74ResourcesProd(mode="production", root=HERE / "_guard_prod_test" / "elsewhere")
    except LedgerStateError:
        noncanon = True
    check("production_refuses_noncanonical_root", noncanon)
    fx = fresh("fixture_only")
    check("fixture_is_not_canonical_and_independent",
          fx.mode == "fixture" and fx.root.resolve() != Path(G.R74_CANONICAL_ROOT).resolve()
          and fx.synth_steps.used() == 0)

    shutil.rmtree(HERE / "_guard_prod_test", ignore_errors=True)
    n_pass = sum(1 for r in results if r["pass"])
    return {"total": len(results), "pass": n_pass, "fail": len(results) - n_pass,
            "all_pass": n_pass == len(results), "results": results}


if __name__ == "__main__":
    s = run_tests()
    for r in s["results"]:
        print(("PASS" if r["pass"] else "FAIL"), r["name"], "" if r["pass"] else r.get("detail"))
    print("R74_PROD_SEAM_TESTS", "ALL_PASS" if s["all_pass"] else "FAILED:%d" % s["fail"])
    raise SystemExit(0 if s["all_pass"] else 1)
