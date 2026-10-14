"""ROUND74 PHASE A driver (resource wiring only).

This driver does NOT load the task model, run forward/backward, run PPO, or build a real
environment. The only real physical consumption is <= 2 toy Adam.step calls on a constant single
parameter with an explicitly-set gradient (no model / environment / forward). Each step is charged
BEFORE opt.step() from the canonical R74 hard-16 synthetic budget, and an independent sub-limit of
2 proves the 3rd attempt is refused before any Adam call or state change.

Run EXACTLY ONCE. A durable marker file prevents accidental re-charging (there is no refund).
"""
from __future__ import annotations
import hashlib, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch
from r74_guard import R74Resources, R74QuotaExhausted, _AtomicCounter, R74_CANONICAL_ROOT


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        out = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                             capture_output=True, text=True)
        return out.stdout.strip()
    except Exception as e:  # pragma: no cover
        return "error:%s" % e


def run_spy_tests() -> dict:
    from test_r74_guard import run_tests
    return run_tests()

def toy_adam_proof(res: R74Resources) -> dict:
    # Independent sub-limit of 2 (own counter; mirrored to the single total ledger with a distinct
    # kind so an auditor can separate the local proof from the real hard-16 budget).
    sublimit = _AtomicCounter(
        res.root / "r74_phase_a_toy_sublimit_counter.json",
        normal_cap=2, reserve_cap=0, hard_cap=2,
        kind="r74-phase-a-toy-sublimit/1", ledger=res.ledger,
        unit="phase_a_toy_step_sublimit")

    p = torch.tensor([0.0], requires_grad=True)
    opt = torch.optim.Adam([p], lr=0.1)

    calls = []
    for i in (1, 2):
        p.grad = torch.tensor([1.0])                       # explicit gradient, no forward
        sublimit.charge(1, "phase_a_toy_step_%d" % i)      # sub-limit FIRST (would refuse a 3rd)
        res.synth_steps.charge(1, "phase_a_toy_adam_step_%d" % i)  # real hard-16 consumption
        before = p.item()
        opt.step()                                         # the real physical optimizer.step
        after = p.item()
        opt.zero_grad()
        calls.append({"call": i, "before": before, "after": after,
                      "sublimit_used_after": sublimit.used(),
                      "global_used_after": res.synth_steps.used()})

    # 3rd attempt must be refused by the sub-limit BEFORE the global charge and BEFORE opt.step().
    p_before = p.item()
    state_rows_before = len(opt.state[p]["exp_avg"]) if p in opt.state else 0
    refused = False
    refusal_err = None
    try:
        sublimit.charge(1, "phase_a_toy_step_3_refused")   # raises at cap 2
        opt.step()                                         # must NOT be reached
    except R74QuotaExhausted as e:
        refused = True
        refusal_err = str(e)

    return {
        "permitted_calls": calls,
        "third_attempt_refused": refused,
        "third_attempt_refusal_error": refusal_err,
        "param_before_refusal": p_before,
        "param_after_refusal": p.item(),
        "param_unchanged_after_refusal": p_before == p.item(),
        "adam_state_rows_before_refusal": state_rows_before,
        "adam_state_rows_after_refusal": len(opt.state[p]["exp_avg"]) if p in opt.state else 0,
        "global_synth_steps_used": res.synth_steps.used(),
        "sublimit_used": sublimit.used(),
        "sublimit_refusals": sublimit.snapshot()["refusals"],
    }


def main() -> int:
    marker = R74_CANONICAL_ROOT / "r74_phase_a_toy_proof.done.json"
    res = R74Resources()  # canonical durable root ($R/raw/runtime)
    already = marker.exists()

    report = {
        "round": 74, "phase": "A",
        "canonical_root": str(res.root),
        "total_ledger_path": str(res.ledger.path),
        "marker_path": str(marker),
        "marker_preexisting": already,
    }

    # 1. spy suite: pure synthetic, throwaway fixtures, no real consumption, no canonical ledger.
    spy = run_spy_tests()
    report["spy_tests"] = {"total": spy["total"], "pass": spy["pass"],
                           "fail": spy["fail"], "all_pass": spy["all_pass"]}
    if not spy["all_pass"]:
        report["error"] = "spy suite failed"
        (HERE / "ROUND74_PHASE_A_ACCEPTANCE.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print("R74_PHASE_A SPY_FAIL", json.dumps(report["spy_tests"]))
        return 2

    # 2. real toy Adam proof (only if not already done -> no re-charge / no refund).
    if already:
        report["toy_adam_proof"] = {"skipped": True, "reason": "marker_present",
                                    "previous": json.loads(marker.read_text(encoding="utf-8"))}
        proof_ok = True
    else:
        proof = toy_adam_proof(res)
        proof_ok = (proof["global_synth_steps_used"] == 2
                    and proof["third_attempt_refused"]
                    and proof["param_unchanged_after_refusal"]
                    and proof["adam_state_rows_before_refusal"] == proof["adam_state_rows_after_refusal"]
                    and proof["sublimit_used"] == 2)
        report["toy_adam_proof"] = proof
        report["toy_adam_proof_ok"] = proof_ok
        if not proof_ok:
            report["error"] = "toy adam proof invariants failed"
            (HERE / "ROUND74_PHASE_A_ACCEPTANCE.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
            print("R74_PHASE_A PROOF_FAIL", json.dumps(proof))
            return 3
        marker.write_text(json.dumps({
            "global_synth_steps_used": res.synth_steps.used(),
            "sublimit_used": proof["sublimit_used"],
            "third_attempt_refused": proof["third_attempt_refused"],
            "unix": time.time(),
        }, indent=2), encoding="utf-8")

    # 3. actual consumption snapshot (authoritative, from the canonical durable counters).
    snap = res.snapshots()
    consumption = {
        "physical_optimizer_steps_real": snap["synthetic_physical_steps"]["used"],
        "environment_constructions": snap["env_slots"]["used"],
        "offline_forward_target_rows": snap["offline_forward_target_rows"]["used"],
        "ppo_update_calls": snap["ppo_calls"]["used"],
        "joint_records": snap["joint_records"]["used"],
    }
    report["consumption"] = consumption
    report["zero_side_effect"] = {
        "environment_constructions": consumption["environment_constructions"] == 0,
        "offline_forward_target_rows": consumption["offline_forward_target_rows"] == 0,
        "ppo_update_calls": consumption["ppo_update_calls"] == 0,
    }
    report["phase_b_status"] = ("NOT_STARTED - PHASE A is resource wiring only; "
                                "PHASE B (real CLI/controller/PPO/training/eval) awaits a separate "
                                "decision-end continuation and is NOT authorized here.")

    # 4. code freeze + git identity.
    files = {"r74_guard.py": sha256_file(HERE / "r74_guard.py"),
             "test_r74_guard.py": sha256_file(HERE / "test_r74_guard.py"),
             "r74_phase_a.py": sha256_file(HERE / "r74_phase_a.py")}
    head = git_head()
    (HERE / "ROUND74_FREEZE.json").write_text(json.dumps({
        "round": 74, "phase": "A", "git_head": head,
        "branch": "exp/round74-constrained-early-head-adaptation",
        "files": files, "canonical_root": str(res.root),
        "total_ledger_path": str(res.ledger.path),
    }, indent=2), encoding="utf-8")

    (HERE / "ROUND74_PHASE_A_ACCEPTANCE.json").write_text(json.dumps({
        "round": 74, "phase": "A", "git_head": head, "files": files,
        "spy_tests": report["spy_tests"], "toy_adam_proof": report["toy_adam_proof"],
        "consumption": consumption, "zero_side_effect": report["zero_side_effect"],
        "snapshots": snap, "phase_b_status": report["phase_b_status"],
        "note": ("PHASE A wiring only. Real physical consumption = 2 toy Adam.step on a constant "
                 "scalar with an explicit gradient (no model/env/forward). The 3rd attempt is "
                 "refused by the sub-limit-2 before the global charge and before opt.step; the "
                 "parameter and Adam state are unchanged and the global hard-16 stays at 2."),
    }, indent=2), encoding="utf-8")

    print("R74_PHASE_A", "OK" if proof_ok else "MARKER_SKIP")
    print(json.dumps({"git_head": head, "consumption": consumption,
                      "spy": report["spy_tests"], "marker_preexisting": already,
                      "phase_b": report["phase_b_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
