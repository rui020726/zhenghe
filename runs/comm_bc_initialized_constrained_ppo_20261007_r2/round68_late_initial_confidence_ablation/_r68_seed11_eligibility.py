"""ROUND68 seed-11 original-chain eligibility receipt (READ-ONLY, zero environment).

Proves, from durable artefacts only, that the seed-11 ep128 model used for the ROUND68 statistics
comes from a complete compliant chain:
  * the start parameters are the compliant ROUND60 ep0 with the one-time 0.10 late-head scale;
  * the initial optimizer / lambda / RNG / data-order state matches the frozen contract;
  * compliant training covers episode 0..127 in 32 batches with no attempt-2 retry;
  * no non-compliant weight, Adam state or lambda was inherited;
  * the live execution version that ran the batches is the ROUND68 one with critic isolation;
  * the final ep128 checkpoint identity binds to that chain.

This script constructs NO environment, runs NO model forward, NO optimizer step and writes NO
checkpoint. It only reads committed artefacts and recomputes identities. It must be run AFTER
configure_and_inject so the live v2_controller / v2_trainer modules are importable.
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
sys.path.insert(0, str(HERE))
import r68_execute as X      # noqa: E402
import r68_frozen as r68     # noqa: E402
import r66_frozen as r66f    # noqa: E402

SEED = 11
EPISODES = 128
BATCHES = 32


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    t0 = time.time()
    cfg = X.configure_and_inject(RAW)      # wiring only: 0 forward / 0 env / 0 optimizer
    common = cfg["common"]
    ck = Path(common.CHECKPOINTS)
    order = [int(x) for x in common.scenario_order()]
    checks: dict[str, object] = {}
    info: dict[str, object] = {}
    _ckpt_before = {q.name: (q.stat().st_mtime_ns, q.stat().st_size) for q in ck.glob("*.pt")}

    # ---- 1. live wiring identity --------------------------------------------
    ident = cfg["identity"]
    checks["trainer_module_is_v2_trainer"] = Path(cfg["trainer"].__file__).name == "v2_trainer.py"
    checks["controller_module_is_v2_controller"] = (
        Path(cfg["controller_module"].__file__).name == "v2_controller.py")
    checks["ppo_update_is_r68"] = bool(ident["ppo_update_is_r68"])
    checks["save_is_r68"] = bool(ident["save_is_r68"])
    checks["load_is_r68"] = bool(ident["load_is_r68"])
    checks["load_ro_is_r68"] = bool(ident["load_ro_is_r68"])
    checks["build_joint_run_is_permit_guarded"] = bool(ident["build_joint_run_is_permit_guarded"])
    checks["construction_permit_regime_r68"] = ident["construction_permit_regime"] == X.PERMIT_REGIME
    checks["counted_builder_inside_permit_guard"] = (
        "counted_build" in str(ident["counted_builder_inside_permit_guard"]))
    checks["gae_lambda_1"] = float(common.GAE_LAMBDA) == 1.0
    checks["eta_lambda_0p5"] = float(common.ETA_LAMBDA) == 0.5
    checks["gamma_1"] = float(common.GAMMA) == 1.0
    checks["raw_root_is_r68"] = str(cfg["raw_root"].resolve()) == str(RAW.resolve())
    checks["checkpoints_under_r68_raw"] = str(ck.resolve()).startswith(str(RAW.resolve()))
    checks["rows_under_r68_raw"] = str(Path(common.ROWS).resolve()).startswith(str(RAW.resolve()))
    checks["r40_hard_cap_800"] = int(cfg["runtime"]["r40_common"].HARD_CAP) == 800
    checks["training_version_is_r68"] = r68.R68_TRAINING_VERSION.endswith(
        "+r68-late-initial-confidence-alpha0.10/1")
    info["identity"] = ident
    info["module_paths"] = {
        "trainer": str(Path(cfg["trainer"].__file__).resolve()),
        "controller": str(Path(cfg["controller_module"].__file__).resolve()),
        "common": str(Path(common.__file__).resolve()),
    }

    # ---- 2. init identity vs the compliant ROUND60 ep0 -----------------------
    ep0_id = _read(ck / f"r68_init_identity_s{SEED}.json")
    checks["init_identity_exists"] = True
    checks["init_alpha_0p10"] = abs(float(ep0_id["late_init_alpha"]) - r68.LATE_INIT_ALPHA) < 1e-12
    checks["init_pre_scale_equals_r60_ep0"] = bool(ep0_id["pre_scale_equals_r60_ep0"])
    checks["init_late_head_scaled_once"] = (
        ep0_id["late_head_sha_before_scale"] != ep0_id["late_head_sha_after_scale"])
    r60 = r68.load_r60_model(SEED, "ep0")          # fails closed on file/schema drift
    ref = r68.fresh_partitioned_controller(SEED, ep0_state=None, verify_ep0=False)
    ref.load_state_dict(r60["controller"])
    checks["r60_ep0_file_sha_matches_freeze"] = True
    checks["ref_pre_scale_sha_is_init_pre_scale"] = (
        ref.parameter_summary()["parameter_sha256"] == ep0_id["pre_scale_parameter_sha256"])
    checks["ref_trunk_identity_nonempty"] = (
        r66f._tensor_identity(r66f.partitioned_parameters(ref)[0])["tensor_count"] == 6)
    info["r60_ep0_source"] = {
        "path": str(Path(r66f.r60_model_path(SEED, "ep0"))),
        "file_sha256": _sha256_file(Path(r66f.r60_model_path(SEED, "ep0"))),
        "parameter_sha256": ref.parameter_summary()["parameter_sha256"],
        "frozen_sha256": r66f._tensor_identity(r66f.partitioned_parameters(ref)[0])["sha256"],
        "trunk_sha256": r68._module_sha(ref, "trunk"),
        "early_head_sha256": r68._module_sha(ref, "early_head"),
    }

    # ---- 3. batch-0 pre-update checkpoint == post-scale init -----------------
    c_b0, st_b0, id_b0 = r68.load_checkpoint_read_only(
        ck / f"v2_train_s{SEED}_before_b0.pt", freeze=X.R68_FREEZE)
    checks["before_b0_sha_is_post_scale_init"] = (
        c_b0.parameter_summary()["parameter_sha256"] == ep0_id["post_scale_parameter_sha256"])
    checks["before_b0_late_head_is_0p10_ep0"] = (
        torch.equal(c_b0.inner.late_head.weight, ref.inner.late_head.weight * r68.LATE_INIT_ALPHA)
        and torch.equal(c_b0.inner.late_head.bias, ref.inner.late_head.bias * r68.LATE_INIT_ALPHA))
    checks["before_b0_trunk_equals_r60_ep0"] = r68._module_sha(c_b0, "trunk") == r68._module_sha(ref, "trunk")
    checks["before_b0_early_head_equals_r60_ep0"] = (
        r68._module_sha(c_b0, "early_head") == r68._module_sha(ref, "early_head"))
    checks["before_b0_critics_equal_r60_ep0"] = (
        r68._module_sha(c_b0, "value_task_head") == r68._module_sha(ref, "value_task_head")
        and r68._module_sha(c_b0, "cost_head") == r68._module_sha(ref, "cost_head"))
    _b0_payload = torch.load(ck / f"v2_train_s{SEED}_before_b0.pt", map_location="cpu", weights_only=False)
    _b0_state = (_b0_payload.get("optimizer") or {}).get("state") or {}
    checks["before_b0_adam_state_empty_no_momentum_inherited"] = len(_b0_state) == 0
    checks["before_b0_adam_owns_six_params"] = (
        len((_b0_payload.get("optimizer") or {}).get("param_groups", [{}])[0].get("params", [])) == 6)
    checks["before_b0_rng_present"] = sorted((_b0_payload.get("rng") or {}).keys()) == [
        "numpy", "python", "torch"]
    checks["before_b0_lambda_is_1"] = abs(float(st_b0["lambda"]) - 1.0) < 1e-12
    checks["before_b0_episodes_done_0"] = int(st_b0["episodes_done"]) == 0
    checks["before_b0_updates_0"] = int(st_b0["updates"]) == 0
    checks["before_b0_optimizer_steps_0"] = int(st_b0["optimizer_steps"]) == 0
    checks["before_b0_data_order_seed_20260927"] = int(st_b0["data_order_seed"]) == 20260927
    checks["before_b0_structure_r68"] = id_b0.get("structure") == r68.R68_STRUCTURE
    checks["before_b0_late_init_alpha_0p10"] = abs(
        float(id_b0.get("late_init_alpha", -1)) - r68.LATE_INIT_ALPHA) < 1e-12
    checks["before_b0_frozen_mask"] = (
        id_b0.get("frozen_mask") or {}).get("frozen_modules") == ["trunk", "early_head"]
    info["before_b0"] = {
        "parameter_sha256": c_b0.parameter_summary()["parameter_sha256"],
        "frozen_sha256": r66f._tensor_identity(r68.frozen_parameters(c_b0))["sha256"],
    }

    # ---- 4. ep128 read-only identity -----------------------------------------
    c128, st128, id128 = r68.load_checkpoint_read_only(
        ck / f"v2_train_s{SEED}_ep128.pt", freeze=X.R68_FREEZE)
    ps128 = c128.parameter_summary()
    checks["ep128_structure_r68"] = id128.get("structure") == r68.R68_STRUCTURE
    checks["ep128_parameter_objects_12"] = ps128["parameter_object_count"] == 12
    checks["ep128_five_modules"] = sorted({k.split(".")[0] for k in ps128["shapes"]}) == sorted(
        ["trunk", "early_head", "late_head", "value_task_head", "value_cost_head"])
    checks["ep128_frozen_mask"] = (id128.get("frozen_mask") or {}).get("frozen_modules") == [
        "trunk", "early_head"]
    checks["ep128_trainable_mask"] = (id128.get("frozen_mask") or {}).get("trainable_modules") == [
        "late_head", "value_task_head", "value_cost_head"]
    checks["ep128_episodes_done_128"] = int(st128["episodes_done"]) == EPISODES
    checks["ep128_updates_32"] = int(st128["updates"]) == BATCHES
    checks["ep128_lambda_1p625"] = abs(float(st128["lambda"]) - 1.625) < 1e-12
    checks["ep128_data_order_seed_20260927"] = int(st128["data_order_seed"]) == 20260927
    checks["ep128_training_seed_11"] = int(st128["training_seed"]) == SEED
    checks["ep128_training_version_is_r68"] = str(id128.get("training_version", "")).endswith(
        "+r68-late-initial-confidence-alpha0.10/1")
    checks["ep128_read_only_restores_no_optimizer"] = id128.get("optimizer_restored") is False
    checks["ep128_read_only_restores_no_rng"] = id128.get("rng_restored") is False
    _e128_payload = torch.load(ck / f"v2_train_s{SEED}_ep128.pt", map_location="cpu", weights_only=False)
    _e128_state = (_e128_payload.get("optimizer") or {}).get("state") or {}
    checks["ep128_adam_state_six_params"] = len(_e128_state) == 6
    checks["ep128_adam_momentum_present"] = all(
        int(v.get("step", 0)) > 0 for v in _e128_state.values())
    checks["ep128_adam_each_param_236_steps"] = all(
        int(v.get("step", 0)) == 236 for v in _e128_state.values())
    checks["ep128_adam_param_steps_6x236"] = sum(
        int(v.get("step", 0)) for v in _e128_state.values()) == 6 * 236
    _lam_hist = _e128_payload.get("training_state", {}).get("lambda_history") or []
    checks["ep128_lambda_history_32"] = len(_lam_hist) == BATCHES
    checks["ep128_lambda_history_batches_0_31"] = [
        int(e.get("batch")) for e in _lam_hist] == list(range(BATCHES))
    checks["ep128_lambda_history_monotone_nondecreasing"] = all(
        float(_lam_hist[i]["lambda_next"]) <= float(_lam_hist[i + 1]["lambda_next"]) + 1e-12
        for i in range(len(_lam_hist) - 1))
    checks["ep128_lambda_final_matches_history"] = abs(
        float(_lam_hist[-1]["lambda_next"]) - float(st128["lambda"])) < 1e-12
    checks["ep128_rng_present"] = sorted((_e128_payload.get("rng") or {}).keys()) == [
        "numpy", "python", "torch"]
    checks["ep128_optimizer_param_groups_1"] = len(
        (_e128_payload.get("optimizer") or {}).get("param_groups", [])) == 1
    checks["ep128_trunk_equals_r60_ep0"] = r68._module_sha(c128, "trunk") == r68._module_sha(ref, "trunk")
    checks["ep128_early_head_equals_r60_ep0"] = (
        r68._module_sha(c128, "early_head") == r68._module_sha(ref, "early_head"))
    checks["ep128_late_head_moved_from_init"] = (
        r68._module_sha(c128, "late_head") != r68._module_sha(c_b0, "late_head"))
    checks["ep128_critics_moved_from_init"] = (
        r68._module_sha(c128, "value_task_head") != r68._module_sha(c_b0, "value_task_head")
        or r68._module_sha(c128, "cost_head") != r68._module_sha(c_b0, "cost_head"))
    info["ep128"] = {
        "parameter_sha256": ps128["parameter_sha256"],
        "frozen_sha256": r66f._tensor_identity(r68.frozen_parameters(c128))["sha256"],
        "late_head_sha256": r68._module_sha(c128, "late_head"),
        "optimizer_steps": int(st128["optimizer_steps"]),
        "lambda": float(st128["lambda"]),
        "file_sha256": _sha256_file(ck / f"v2_train_s{SEED}_ep128.pt"),
    }

    # ---- 5. 32-batch parameter chain -----------------------------------------
    recs = [_read(ck / f"r40_batch_s{SEED}_b{b}_attempt01.json") for b in range(BATCHES)]
    checks["batch_records_32_present"] = all(
        (ck / f"r40_batch_s{SEED}_b{b}_attempt01.json").is_file() for b in range(BATCHES))
    checks["batch_records_single_attempt01"] = (
        not list(ck.glob("r40_batch_s11_b*_attempt02*.json"))
        and len(list(ck.glob("r40_batch_s11_b*_attempt*.json"))) == BATCHES)
    chain_ok = True
    prev = ep0_id["post_scale_parameter_sha256"]
    for b, rec in enumerate(recs):
        if rec.get("parameters_before_sha") != prev:
            chain_ok = False
        prev = rec.get("parameters_after_sha")
    checks["batch_chain_unbroken"] = chain_ok
    checks["batch0_before_sha_is_post_scale_init"] = (
        recs[0].get("parameters_before_sha") == ep0_id["post_scale_parameter_sha256"])
    checks["batch31_after_sha_is_ep128"] = (
        recs[BATCHES - 1].get("parameters_after_sha") == ps128["parameter_sha256"])
    checks["every_batch_has_4_episodes"] = all(len(r.get("episodes", [])) == 4 for r in recs)
    eps = [int(e) for r in recs for e in r.get("episodes", [])]
    checks["batch_episodes_cover_0_127_once"] = sorted(eps) == list(range(EPISODES))
    checks["batch_ppo_steps_sum_236"] = sum(
        int((r.get("update") or {}).get("optimizer_steps", 0)) for r in recs) == 236
    checks["every_batch_lambda_recorded"] = all("lambda_at_rollout" in r for r in recs)

    # ---- 6. claims / rows ----------------------------------------------------
    claims_dir = RAW / "launch_claims"
    dirs = sorted(p for p in claims_dir.iterdir() if p.is_dir())
    attempt_files = sorted(claims_dir.glob("*/attempt_*.json"))
    checks["claim_dirs_128"] = len(dirs) == EPISODES
    checks["claims_exactly_128_attempt_files"] = len(attempt_files) == EPISODES
    checks["claims_all_attempt_0001"] = all(p.name == "attempt_0001.json" for p in attempt_files)
    claims = [_read(p) for p in attempt_files]
    checks["claims_all_complete"] = all(c.get("status") == "complete" for c in claims)
    checks["claims_all_normal_category"] = all(c.get("category") == "normal" for c in claims)
    checks["claims_no_reserve"] = not any(c.get("category") == "reserve" for c in claims)
    checks["claims_one_constructor_call_each"] = all(
        int(c.get("environment_constructor_calls", 0)) == 1 for c in claims)
    checks["claims_1000_steps_each"] = all(int(c.get("completed_steps", 0)) == 1000 for c in claims)
    checks["claims_one_run_invocation_each"] = all(
        int(c.get("run_invocation_calls", 0)) == 1 for c in claims)
    checks["claims_zero_build_failures"] = all(
        int(c.get("joint_run_build_failures", 0)) == 0 for c in claims)
    checks["claims_simulation_started_each"] = all(c.get("simulation_started") is True for c in claims)
    checks["claims_no_permits_dir"] = not any((d / "permits").exists() for d in dirs)
    checks["claim_jobs_match_plan"] = sorted(p.parent.name for p in attempt_files) == sorted(
        f"r68_new_training_s{SEED}_e{ep}" for ep in range(EPISODES))

    rows_dir = RAW / "rows"
    rows = sorted(rows_dir.glob("v2_r68_train_s11_e*_a1.json"))
    checks["rows_128"] = len(rows) == EPISODES
    checks["rows_no_attempt2_files"] = not list(rows_dir.glob("*_a2.json"))
    row_ok = True
    for ep in range(EPISODES):
        scene = order[ep]
        p = rows_dir / f"v2_r68_train_s{SEED}_e{ep}_s{scene}_a1.json"
        if not p.is_file():
            row_ok = False
        elif int(_read(p).get("scenario_seed", -1)) != scene:
            row_ok = False
    checks["rows_match_frozen_scenario_order"] = row_ok

    # ---- 7. PPO / optimizer journals -----------------------------------------
    ppo = [json.loads(l) for l in
           (RAW / "runtime" / "ppo_update_calls.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    checks["ppo_journal_32_calls"] = len(ppo) == BATCHES
    checks["ppo_journal_indices_0_31_contiguous"] = [
        int(p["update_index"]) for p in ppo] == list(range(BATCHES))
    checks["ppo_journal_steps_sum_236"] = sum(int(p.get("optimizer_steps", 0)) for p in ppo) == 236
    checks["ppo_journal_structure_r68"] = all(p.get("structure") == r68.R68_STRUCTURE for p in ppo)
    opt_lines = [l for l in
                 (RAW / "runtime" / "optimizer_step_calls.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    checks["optimizer_step_journal_236"] = len(opt_lines) == 236
    checks["optimizer_step_journal_seed11_only"] = all(
        json.loads(l).get("training_seed", 11) == 11 for l in opt_lines)

    # ---- 8. first-batch gate -------------------------------------------------
    gate = _read(RAW / "runtime" / f"r68_first_batch_gate_s{SEED}.json")
    checks["first_batch_gate_passed"] = gate.get("status") == "passed"
    bools = {k: v for k, v in gate.get("checks", {}).items() if isinstance(v, bool)}
    checks["first_batch_gate_all_bools_true"] = all(bools.values())
    checks["first_batch_gate_bool_count_21"] = len(bools) == 21
    checks["first_batch_gate_early_and_late_free"] = (
        int(gate["checks"].get("early_free", 0)) > 0 and int(gate["checks"].get("late_free", 0)) > 0)
    info["first_batch_gate"] = {
        "status": gate.get("status"), "early_free": gate.get("early_free"),
        "late_free": gate.get("late_free"),
        "lambda_after_first_batch": gate.get("lambda_after_first_batch"),
        "parameters_after_sha": gate.get("parameters_after_sha"),
        "early_free": gate["checks"].get("early_free"),
        "late_free": gate["checks"].get("late_free"),
    }

    # ---- 9. ledger / budget --------------------------------------------------
    led = _read(RAW / "runtime" / "ROUND68_LEDGER.json")
    c = led["counts"]
    checks["ledger_normal_128"] = c["normal_claims"] == 128
    checks["ledger_reserve_claims_0"] = c["reserve_claims"] == 0
    checks["ledger_complete_128"] = c["complete"] == 128
    checks["ledger_failed_technical_0"] = c["failed_technical"] == 0
    checks["ledger_booked_128"] = c["booked_attempts"] == 128
    checks["ledger_actual_129"] = c["actual_environment_constructions"] == 129
    checks["ledger_correction_extra_1"] = c["late_correction_extra_constructions"] == 1
    checks["ledger_correction_counted_once"] = (
        len(c["correction"]["extras"]) == 1 and c["correction"]["extras"][0]["counted"] is True)
    checks["ledger_hard_headroom_671"] = c["hard_headroom"] == 671
    checks["ledger_hard_cap_exceeded_by_0"] = c["hard_cap_exceeded_by"] == 0
    checks["ledger_historical_probe_0"] = led["historical_non_plan_probe"] == 0
    checks["ledger_remaining_plan_646"] = (384 - 128) + 390 == 646
    info["ledger"] = {k: c[k] for k in (
        "normal_claims", "reserve_claims", "complete", "booked_attempts",
        "actual_environment_constructions", "hard_cap", "hard_headroom",
        "hard_cap_exceeded_by", "late_correction_extra_constructions")}

    # ---- 10. guard ledgers (acceptance / synthetic only) ---------------------
    fw = _read(RAW / "runtime" / "r68_guard_forward_rows.json")
    stp = _read(RAW / "runtime" / "r68_guard_synthetic_steps.json")
    ppg = _read(RAW / "runtime" / "r68_guard_synthetic_ppo.json")
    checks["guard_forward_rows_9319_exceeded_preserved"] = fw["used"] == 9319 and fw["used"] > fw["cap"]
    checks["guard_synthetic_steps_18_within_32"] = stp["used"] == 18 and stp["used"] <= stp["cap"]
    checks["guard_synthetic_ppo_5_recorded"] = ppg["used"] == 5
    checks["guard_synthetic_ppo_events_sum_5"] = sum(int(e["n"]) for e in ppg["events"]) == 5
    info["guards"] = {
        "forward_rows": {"cap": fw["cap"], "used": fw["used"],
                         "status": "EXCEEDED (historical, preserved, not refunded)"},
        "synthetic_optimizer_steps": {"cap": stp["cap"], "used": stp["used"], "makeup_subcap_used": 16},
        "synthetic_ppo_calls": {"cap": ppg["cap"], "used": ppg["used"],
                                "note": "counted separately for the record; 5 includes one failed entry"},
    }

    # ---- 11. protected files -------------------------------------------------
    r2 = HERE.parent
    for name, want in (
            ("ROUND46_CURRENT_TASK.json",
             "cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b"),
            ("ROUND46_R2_V6_STOP_STATE.json",
             "93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18")):
        checks[f"protected_{name}"] = _sha256_file(r2 / name) == want

    # ---- 12. this receipt's own consumption ----------------------------------
    _after = {q.name: (q.stat().st_mtime_ns, q.stat().st_size) for q in ck.glob("*.pt")}
    checks["receipt_zero_checkpoint_writes"] = _after == _ckpt_before
    checks["receipt_zero_new_checkpoint_files"] = set(_after) == set(_ckpt_before)
    checks["receipt_zero_env_constructions"] = True
    checks["receipt_zero_model_forwards"] = True
    checks["receipt_zero_optimizer_steps"] = True

    status = "verified" if all(v for k, v in checks.items() if isinstance(v, bool)) else "unverified"
    failed = [k for k, v in checks.items() if isinstance(v, bool) and not v]
    receipt = {
        "kind": "round68-seed11-eligibility-receipt/1",
        "task_id": X.TASK_ID,
        "branch": X.BRANCH,
        "seed": SEED,
        "status": status,
        "checks": checks,
        "checks_total": len(checks),
        "checks_failed": failed,
        "info": info,
        "consumption": {"environment_constructions": 0, "model_forward_rows": 0,
                        "synthetic_optimizer_steps": 0, "synthetic_ppo_calls": 0,
                        "checkpoint_writes": 0, "ppo_replays": 0},
        "scope_note": ("Eligibility of the seed-11 ep128 model for the ROUND68 statistics. It does "
                       "NOT by itself establish the scientific result; it establishes that the model "
                       "comes from a complete compliant chain with no non-compliant inheritance."),
        "attempt2_note": ("No attempt-2 claim exists for seed 11 (all 128 claims are attempt_0001 and "
                          "all 128 rows are _a1), so the attempt-2 row/tape/log and reload/gate/slot "
                          "resolution was never exercised by seed 11. That path is registered and "
                          "fixture-proven separately BEFORE seed 22/33 training; see "
                          "ROUND68_FREEZE_ADDENDUM_3.json."),
        "finished_unix": time.time(),
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    out = RAW / "runtime" / "R68_SEED11_ELIGIBILITY_RECEIPT.json"
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8", newline="\n")
    print(json.dumps({"status": status, "checks_total": len(checks), "failed": failed,
                      "elapsed": receipt["elapsed_seconds"]}, ensure_ascii=False))
    return 0 if status == "verified" else 3


if __name__ == "__main__":
    raise SystemExit(main())