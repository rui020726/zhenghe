"""ROUND74 PHASE A pure synthetic + spy quota tests. No task model, no environment, no real
Adam.step. Throwaway fixture ledgers only; never touches the canonical R74 ledger and never
creates a real machine licence. The <=2 real toy Adam.step proof lives in r74_phase_a.py and is
the ONLY real physical consumption of PHASE A."""
from __future__ import annotations
import shutil, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from r74_guard import (R74Resources, R74QuotaExhausted, R74SyntheticStepGuard, R74TotalLedger,
                       count_target_vectors)


def run_tests():
    results = []
    def check(name, cond, **detail):
        results.append({"name": name, "pass": bool(cond), **({} if cond else {"detail": detail})})

    def fresh(sub):
        d = HERE / "_guard_test" / sub
        if d.exists():
            shutil.rmtree(d)
        (d / "runtime").mkdir(parents=True)
        return R74Resources(d / "runtime")

    # 1. physical-step caps: normal 12 + reserve 4, hard 16 (denominated in physical steps)
    r = fresh("synthcaps").synth_steps
    for i in range(12):
        r.charge(1, f"n{i}")
    check("synth_normal_12_ok", r.used() == 12)
    refused = False
    try:
        r.charge(1, "n13")
    except R74QuotaExhausted:
        refused = True
    check("synth_normal_13_refused", refused and r.used() == 12)
    for i in range(4):
        r.charge(1, f"r{i}", category="reserve")
    check("synth_reserve_to_hard_16", r.used() == 16)
    refused = False
    try:
        r.charge(1, "over", category="reserve")
    except R74QuotaExhausted:
        refused = True
    check("synth_hard_16_refused", refused and r.used() == 16)
    check("synth_refusals_counted", r.snapshot()["refusals"] >= 2)

    # 2. charge-before-work, no refund on failure
    res = fresh("norefund")
    try:
        res.synth_steps.charge(1, "will_fail")
        raise RuntimeError("simulated failure after charge")
    except RuntimeError:
        pass
    check("charge_no_refund_after_failure", res.synth_steps.used() == 1)

    # 3. persistence: restart / same-root re-open keeps consumption (not reset by subdir)
    res = fresh("persist")
    for i in range(5):
        res.synth_steps.charge(1, f"p{i}")
    reopened = R74Resources(HERE / "_guard_test/persist/runtime")
    check("restart_keeps_consumption", reopened.synth_steps.used() == 5)

    # 4. two independent units: PPO call != physical step (the R73 defect fixed)
    res = fresh("units")
    res.synth_steps.charge(3, "steps")
    res.ppo_calls.charge(2, "calls")
    check("ppo_call_and_physical_step_independent",
          res.synth_steps.used() == 3 and res.ppo_calls.used() == 2)

    # 5. offline trunk TARGET metering: actual last-dim-19 rows, per-batch 1024, joint separate
    class FakeT:
        def __init__(self, shape):
            self.shape = shape
    check("count_targets_2d_19", count_target_vectors(FakeT((8, 19))) == 8)
    check("count_targets_3d_19", count_target_vectors(FakeT((4, 5, 19))) == 20)
    check("count_targets_non19_zero", count_target_vectors(FakeT((8, 7))) == 0)
    res = fresh("fwd")
    res.offline_forward.charge_targets(500, "b1", batch=True)
    check("fwd_batch_ok", res.offline_forward.used() == 500)
    refused = False
    try:
        res.offline_forward.charge_targets(1025, "big", batch=True)
    except R74QuotaExhausted:
        refused = True
    check("fwd_per_batch_1024_refused", refused and res.offline_forward.used() == 500)
    res.joint_records.charge(999, "joint")
    check("joint_records_separate_from_target_rows",
          res.offline_forward.used() == 500 and res.joint_records.used() == 999)

    # 6. single total ledger records every charge across guards
    res = fresh("ledger")
    res.synth_steps.charge(2, "a")
    res.offline_forward.charge_targets(10, "b")
    res.joint_records.charge(3, "c")
    charged = [e for e in res.ledger.events() if e.get("event") == "charged"]
    check("single_total_ledger_records_all_charges", len(charged) >= 3)

    # 7. env-construction slot caps wired
    res = fresh("env")
    check("env_slot_caps", res.env_slots.normal_cap == 579 and res.env_slots.reserve_cap == 21
          and res.env_slots.hard_cap == 600)
    res.env_slots.charge(1, "e0")
    check("env_slot_charge_ok", res.env_slots.used() == 1)

    shutil.rmtree(HERE / "_guard_test", ignore_errors=True)
    n_pass = sum(1 for r in results if r["pass"])
    return {"total": len(results), "pass": n_pass, "fail": len(results) - n_pass,
            "all_pass": n_pass == len(results), "results": results}


if __name__ == "__main__":
    s = run_tests()
    for r in s["results"]:
        print(("PASS" if r["pass"] else "FAIL"), r["name"], "" if r["pass"] else r.get("detail"))
    print("R74_GUARD_TESTS", "ALL_PASS" if s["all_pass"] else f"FAILED:{s['fail']}")
    raise SystemExit(0 if s["all_pass"] else 1)