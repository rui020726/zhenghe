"""ROUND68 tail acceptance: training attempt-2 owns independent row/tape/log artefacts, and a
completed retry's reload/gate/slot resolution reads the QUALIFIED attempt, never a fixed a1.

Zero-environment by construction: the innermost environment constructor is never reached. The
fixture stubs only runner._run_episode (the seam that would call build_joint_run) and drives the
REAL r68_execute._run_train_episode -> base._r68_atomic_run -> r40_common claim/slot/permit chain.
A tripwire on the permit-guarded builder fails the test if any environment construction is attempted.

Run:  python test_r68_attempt2_paths.py <run_dir> <isolated_test_raw>
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r68_execute as X      # noqa: E402

CHECKS: list[tuple[str, bool, object]] = []


def check(name: str, ok: bool, detail: object = None) -> None:
    CHECKS.append((name, bool(ok), detail))


def main(argv) -> int:
    run_dir = Path(argv[1]).resolve() if len(argv) > 1 else HERE
    # Each run gets its own timestamped subdirectory, so a previous fixture run is never
    # overwritten and changing the output subdirectory cannot clear anything.
    base = Path(argv[2]).resolve() if len(argv) > 2 else (run_dir / "_r68_attempt2_test_raw")
    base.mkdir(parents=True, exist_ok=True)
    test_raw = base / f"run_{int(time.time())}_{os.getpid()}"
    test_raw.mkdir(parents=True, exist_ok=True)
    real_raw = run_dir / "raw"
    t0 = time.time()

    cfg = X.configure_and_inject(test_raw)
    common = cfg["common"]
    runner = cfg["runner"]
    r42_run = cfg["r42_run"]
    r40 = cfg["runtime"]["r40_common"]
    ck = Path(common.CHECKPOINTS)
    order = [int(x) for x in common.scenario_order()]

    # ---- tripwire: no environment construction may be attempted -----------------
    real_builder = runner.build_joint_run
    entered = {"n": 0}

    def tripwire(configuration):
        entered["n"] += 1
        raise RuntimeError("test tripwire: an environment construction was attempted")

    tripwire._r68_permit_guard = True
    runner.build_joint_run = tripwire

    # ---- spy on the episode seam -------------------------------------------------
    seen: list[dict] = []
    mode = {"fail": False}

    def spy(label, kind, scenario_seed, *, policy, attach=None, attempt=1, controller=None,
            collector=None, action_rng_seed=None, tag="", checkpoint_load_identity=None):
        seen.append({"label": label, "kind": kind, "scenario_seed": int(scenario_seed),
                     "policy": policy, "attempt": int(attempt), "tag": tag,
                     "action_rng_seed": action_rng_seed})
        if mode["fail"]:
            raise RuntimeError("synthetic technical failure after the episode seam was entered")
        paths = common.cell_paths(kind, int(scenario_seed), int(attempt), tag=tag)
        return {"on_time": 6, "spent_bytes": 2514, "artefacts": {k: str(v) for k, v in paths.items()},
                "scenario_seed": int(scenario_seed)}

    runner._run_episode = spy
    stub_collector = SimpleNamespace(policy=lambda: None)

    # ============================================================================
    # CASE A - attempt 1 keeps exactly the historical artefact name
    # ============================================================================
    X._run_train_episode(cfg, None, stub_collector, seed=11, ep=100, scenario=order[100], lam=1.0)
    a = seen[-1]
    check("A_attempt1_seen", a["attempt"] == 1, a["attempt"])
    check("A_tag_unchanged", a["tag"] == "r68_train_s11_e100", a["tag"])
    a_paths = common.cell_paths("train", order[100], 1, tag="r68_train_s11_e100")
    check("A_row_name_is_historical",
          a_paths["row"].name == f"v2_r68_train_s11_e100_s{order[100]}_a1.json", a_paths["row"].name)
    a_claim = json.loads((test_raw / "launch_claims" / "r68_new_training_s11_e100"
                          / "attempt_0001.json").read_text(encoding="utf-8"))
    check("A_claim_normal_attempt1", a_claim["category"] == "normal" and a_claim["attempt"] == 1)
    check("A_claim_complete", a_claim["status"] == "complete")
    check("A_permit_not_issued_no_builder",
          X._permit_state(r40, "r68_new_training_s11_e100", 1) == "not_issued")

    # ============================================================================
    # CASE B - a failed attempt-1 is retried on a reserve attempt-2 that owns
    #          its OWN row/tape/log/lifecycle/transcript artefacts and its OWN slot
    # ============================================================================
    mode["fail"] = True
    for ep in range(4):
        try:
            X._run_train_episode(cfg, None, stub_collector, seed=11, ep=ep,
                                 scenario=order[ep], lam=1.0)
            check(f"B_ep{ep}_failure_raised", False, "no exception")
        except RuntimeError as exc:
            check(f"B_ep{ep}_failure_raised", "synthetic technical failure" in str(exc))
    mode["fail"] = False
    fail_paths = {}
    for ep in range(4):
        rec = [s for s in seen if s["tag"] == f"r68_train_s11_e{ep}"][-1]
        fail_paths[ep] = common.cell_paths("train", order[ep], rec["attempt"],
                                           tag=f"r68_train_s11_e{ep}")
        check(f"B_ep{ep}_failed_on_attempt1", rec["attempt"] == 1, rec["attempt"])
    for ep in range(4):
        X._run_train_episode(cfg, None, stub_collector, seed=11, ep=ep,
                             scenario=order[ep], lam=1.0)
    ok_paths = {}
    for ep in range(4):
        recs = [s for s in seen if s["tag"] == f"r68_train_s11_e{ep}"]
        check(f"B_ep{ep}_retry_seen_attempt2", len(recs) == 2 and recs[-1]["attempt"] == 2,
              [r["attempt"] for r in recs])
        ok_paths[ep] = common.cell_paths("train", order[ep], 2, tag=f"r68_train_s11_e{ep}")
        for key in ("row", "tape", "lifecycle", "transcript"):
            check(f"B_ep{ep}_{key}_independent_of_attempt1",
                  ok_paths[ep][key] != fail_paths[ep][key],
                  (fail_paths[ep][key].name, ok_paths[ep][key].name))
        cdir = test_raw / "launch_claims" / f"r68_new_training_s11_e{ep}"
        c1 = json.loads((cdir / "attempt_0001.json").read_text(encoding="utf-8"))
        c2 = json.loads((cdir / "attempt_0002.json").read_text(encoding="utf-8"))
        check(f"B_ep{ep}_old_failure_preserved", c1["status"] == "failed_technical")
        check(f"B_ep{ep}_retry_is_reserve", c2["category"] == "reserve")
        check(f"B_ep{ep}_retry_of_attempt1", c2["metadata"].get("retry_of_attempt") == 1,
              c2["metadata"].get("retry_of_attempt"))
        check(f"B_ep{ep}_retry_complete", c2["status"] == "complete")
        check(f"B_ep{ep}_own_slot", c2.get("launch_slot") != c1.get("launch_slot"),
              (c1.get("launch_slot"), c2.get("launch_slot")))
        check(f"B_ep{ep}_permit_exists_both_attempts",
              X._permit_state(r40, f"r68_new_training_s11_e{ep}", 1) == "not_issued"
              and X._permit_state(r40, f"r68_new_training_s11_e{ep}", 2) == "not_issued")

    # ============================================================================
    # CASE C - reload / gate / slot resolution reads the QUALIFIED attempt 2
    # ============================================================================
    for ep in range(4):
        logical = f"r68_new_training_s11_e{ep}"
        check(f"C_ep{ep}_qualified_attempt_is_2", X._qualified_attempt(r40, logical) == 2,
              X._qualified_attempt(r40, logical))
        p = common.cell_paths("train", order[ep], X._qualified_attempt(r40, logical),
                              tag=f"r68_train_s11_e{ep}")
        check(f"C_ep{ep}_reload_path_is_attempt2_row", p["row"] == ok_paths[ep]["row"],
              p["row"].name)
    check("C_ep100_qualified_attempt_is_1",
          X._qualified_attempt(r40, "r68_new_training_s11_e100") == 1)

    # a logical job with no complete attempt resolves to 1 (nothing qualified yet)
    check("C_unqualified_job_defaults_to_1",
          X._qualified_attempt(r40, "r68_new_training_s11_e999") == 1)

    # ---- qualified batch-record resolution (decoy attempt01 must be ignored) ----
    init_sha = "b" * 64
    after_sha = "c" * 64
    (ck / "r68_init_identity_s11.json").write_text(json.dumps(
        {"seed": 11, "late_init_alpha": 0.10, "post_scale_parameter_sha256": init_sha,
         "pre_scale_equals_r60_ep0": True}), encoding="utf-8")
    (ck / "r40_batch_s11_b0_attempt01.json").write_text(json.dumps(
        {"batch": 0, "episodes_done_after": 999, "parameters_before_sha": init_sha,
         "parameters_after_sha": "0" * 64, "lambda_at_rollout": 1.0, "episodes": [0, 1, 2, 3]}),
        encoding="utf-8")
    (ck / "r40_batch_s11_b0_attempt02.json").write_text(json.dumps(
        {"batch": 0, "episodes_done_after": 4, "parameters_before_sha": init_sha,
         "parameters_after_sha": after_sha, "lambda_at_rollout": 1.0, "episodes": [0, 1, 2, 3]}),
        encoding="utf-8")
    stub_controller = SimpleNamespace(
        parameter_summary=lambda: {"parameter_sha256": after_sha, "parameter_object_count": 12})
    stub_state = {"training_seed": 11, "episodes_done": 4, "updates": 1}
    rec_path, rec = X._qualified_batch_record(cfg, 11, 0, episodes_done_after=4,
                                              before_sha=init_sha, controller=stub_controller,
                                              state=stub_state)
    check("C_batch_record_resolves_attempt02", rec_path.name == "r40_batch_s11_b0_attempt02.json",
          rec_path.name)
    check("C_batch_record_attempt_field",
          int(re.search(r"attempt(\d+)$", rec_path.stem).group(1)) == 2)
    try:
        X._qualified_batch_record(cfg, 11, 0, episodes_done_after=12345, before_sha=init_sha)
        check("C_batch_record_refuses_when_none_qualified", False, "no raise")
    except RuntimeError:
        check("C_batch_record_refuses_when_none_qualified", True)
    (ck / "r40_batch_s11_b0_attempt03.json").write_text(json.dumps(
        {"batch": 0, "episodes_done_after": 4, "parameters_before_sha": init_sha,
         "parameters_after_sha": after_sha, "lambda_at_rollout": 1.0, "episodes": [0, 1, 2, 3]}),
        encoding="utf-8")
    try:
        X._qualified_batch_record(cfg, 11, 0, episodes_done_after=4, before_sha=init_sha)
        check("C_batch_record_refuses_when_ambiguous", False, "no raise")
    except RuntimeError:
        check("C_batch_record_refuses_when_ambiguous", True)
    (ck / "r40_batch_s11_b0_attempt03.json").unlink()

    # ---- the gate-from-committed reload loop reads the attempt-2 rows -----------
    gate_path = test_raw / "runtime" / "r68_first_batch_gate_s11.json"
    check("C_gate_file_absent_before", not gate_path.exists())
    for ep in range(4):
        rp = ok_paths[ep]["row"]
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps({"scenario_seed": order[ep], "on_time": 6}), encoding="utf-8")
        ok_paths[ep]["tape"].parent.mkdir(parents=True, exist_ok=True)
        ok_paths[ep]["tape"].write_text("", encoding="utf-8")
    captured = {}

    def gate_spy(cfg_, controller_, state_, record_, episodes_, *, ep0_identity):
        captured["record"] = record_
        captured["episodes"] = episodes_
        captured["ep0_identity"] = ep0_identity
        return {"status": "passed", "checks": {}}

    reload_calls = []
    real_reload = X._reload_episode

    def reload_spy(row_path, tape_path, *, seed, ep, scenario, lam):
        reload_calls.append({"row": str(row_path), "tape": str(tape_path), "seed": int(seed),
                             "ep": int(ep), "scenario": int(scenario), "lam": float(lam)})
        return real_reload(row_path, tape_path, seed=seed, ep=ep, scenario=scenario, lam=lam)

    real_gate = X._first_batch_gate
    X._first_batch_gate = gate_spy
    X._reload_episode = reload_spy
    try:
        gate = X._gate_from_committed(cfg, 11, stub_controller, stub_state)
    finally:
        X._first_batch_gate = real_gate
        X._reload_episode = real_reload
    check("C_gate_returned", isinstance(gate, dict) and gate.get("status") == "passed")
    check("C_gate_used_attempt02_record",
          captured["record"].get("parameters_after_sha") == after_sha)
    check("C_gate_reloaded_4_episodes", len(captured["episodes"]) == 4)
    check("C_gate_reloaded_exactly_4_rows", len(reload_calls) == 4, len(reload_calls))
    for ep in range(4):
        check(f"C_gate_reload_ep{ep}_reads_attempt2_row",
              reload_calls[ep]["row"] == str(ok_paths[ep]["row"]), reload_calls[ep]["row"])
        check(f"C_gate_reload_ep{ep}_reads_attempt2_tape",
              reload_calls[ep]["tape"] == str(ok_paths[ep]["tape"]), reload_calls[ep]["tape"])
    g = json.loads(gate_path.read_text(encoding="utf-8"))
    check("C_gate_records_batch_attempt_2", g.get("batch_record_attempt") == 2,
          g.get("batch_record_attempt"))
    check("C_gate_records_resolution_mode",
          g.get("reload_attempt_resolution") == "qualified_attempt_from_complete_claim")
    check("C_gate_file_written", gate_path.is_file())

    # ---- slot resolution is attempt-independent BY DESIGN -----------------------
    plan = X.build_plan(order)
    slot_a = X._slot_for(plan, "r68_new_training_s11_e0")
    check("C_plan_slot_is_enumeration", slot_a == 1, slot_a)
    _c1 = json.loads((test_raw / "launch_claims" / "r68_new_training_s11_e0"
                      / "attempt_0001.json").read_text(encoding="utf-8"))
    _c2 = json.loads((test_raw / "launch_claims" / "r68_new_training_s11_e0"
                      / "attempt_0002.json").read_text(encoding="utf-8"))
    check("C_real_slot_binding_is_per_attempt",
          (_c1["launch_slot"], _c1["category"]) != (_c2["launch_slot"], _c2["category"]),
          (_c1.get("launch_slot"), _c1.get("category"), _c2.get("launch_slot"), _c2.get("category")))

    # ============================================================================
    # zero-consumption and isolation assertions
    # ============================================================================
    runner.build_joint_run = real_builder
    check("Z_no_environment_construction", entered["n"] == 0, entered["n"])
    check("Z_no_permit_issued", not any(
        (test_raw / "launch_claims").glob("*/permits/*.issued.json")))
    led = json.loads((real_raw / "runtime" / "ROUND68_LEDGER.json").read_text(encoding="utf-8"))
    check("Z_real_ledger_untouched",
          led["counts"]["actual_environment_constructions"] == 129
          and led["counts"]["normal_claims"] == 128)
    check("Z_real_rows_untouched", len(list((real_raw / "rows").glob("*.json"))) == 128)
    check("Z_real_checkpoints_untouched",
          len(list((real_raw / "checkpoints").glob("v2_train_s11_ep128.pt"))) == 1)

    failed = [n for n, ok, _ in CHECKS if not ok]
    out = {
        "kind": "round68-attempt2-path-acceptance/1",
        "task_id": X.TASK_ID, "branch": X.BRANCH,
        "status": "passed" if not failed else "failed",
        "checks_total": len(CHECKS),
        "checks_failed": failed,
        "checks": {n: {"passed": ok, "detail": d} for n, ok, d in CHECKS},
        "isolated_test_raw_root": str(test_raw),
        "consumption": {"environment_constructions": 0, "model_forward_rows": 0,
                        "synthetic_optimizer_steps": 0, "synthetic_ppo_calls": 0,
                        "checkpoint_writes": 0},
        "episode_seam_calls": len(seen),
        "note": ("The innermost environment constructor is never reached: runner._run_episode is "
                 "stubbed and a tripwire on the permit-guarded builder fails the test if any "
                 "construction is attempted. The claim/slot/permit/retry chain exercised is the "
                 "real r68_execute one."),
        "finished_unix": time.time(),
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (test_raw / "ROUND68_ATTEMPT2_ACCEPTANCE.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": out["status"], "checks_total": len(CHECKS), "failed": failed,
                      "elapsed": out["elapsed_seconds"]}, ensure_ascii=False))
    return 0 if not failed else 4


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))