"""ROUND44 zero-environment, actual-code configuration and recovery gates."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
import tempfile
from pathlib import Path
from typing import Any

import torch

from r44_prepare import (ETA_NEW, RUN_DIR, RUNNER_VERSION, SEEDS, TRAINING_VERSION,
                         prepare, read, sha256, write_new)
from r44_runtime import configure


def _read_only_loader_gate(runtime: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    run = runtime["r42_run"]
    trainer = runtime["trainer"]
    results = {}
    for seed in SEEDS:
        old_controller, old_identity = run.load_eval_model("OLD", seed)
        expected = frozen["round42"]["old_models"][str(seed)]
        if (old_controller.parameter_summary()["parameter_sha256"]
                != expected["parameter_sha256"]
                or old_identity.get("optimizer_restored") is not False
                or old_identity.get("rng_restored") is not False):
            raise RuntimeError(f"seed {seed}: R43 OLD model was not loaded read-only")

        source = Path(expected["path"])
        ambient_before = trainer.rng_state()
        before_hash = runtime["r40_train"]._state_hash(ambient_before)
        adam = torch.optim.Adam
        calls = {"count": 0}

        def counted_adam(*args, **kwargs):
            calls["count"] += 1
            return adam(*args, **kwargs)

        torch.optim.Adam = counted_adam
        error = None
        try:
            try:
                trainer.load_checkpoint(source, freeze=frozen["round42"]["freeze_identity"])
            except RuntimeError as exc:
                error = str(exc)
        finally:
            torch.optim.Adam = adam
        after_hash = runtime["r40_train"]._state_hash(trainer.rng_state())
        if (not error or TRAINING_VERSION not in error and "training_version mismatch" not in error
                or calls["count"] != 0 or before_hash != after_hash):
            raise RuntimeError(
                f"seed {seed}: R43 trained checkpoint did not fail before optimizer/RNG restore; "
                f"error={error!r}, Adam_ctor_calls={calls['count']}, RNG_same={before_hash == after_hash}"
            )
        results[str(seed)] = {
            "readonly_eval_parameter_sha256": old_controller.parameter_summary()[
                "parameter_sha256"],
            "readonly_optimizer_restored": False,
            "readonly_rng_restored": False,
            "training_restore_rejected": True,
            "training_restore_rejection": error,
            "optimizer_constructor_calls_on_rejection": calls["count"],
            "ambient_rng_preserved_on_rejection": before_hash == after_hash,
        }
    return {"kind": "round44-version-boundary/1", "status": "passed",
            "r43_ep128_read_only_evaluation": "passed",
            "r43_trained_restore_into_round44": "rejected_before_optimizer_or_rng_restore",
            "by_seed": results}


def _same_version_restore_gate(runtime: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    trainer = runtime["trainer"]
    common = runtime["common"]
    controller_module = runtime["controller_module"]
    seed = 11
    ambient = trainer.rng_state()
    controller = controller_module.fresh_controller(seed)
    optimizer = torch.optim.Adam(controller.parameters, lr=float(common.LEARNING_RATE))
    state = trainer.fresh_training_state(seed)
    if (float(state.get("eta_lambda", float("nan"))) != ETA_NEW
            or state.get("training_version") != TRAINING_VERSION
            or float(state.get("lambda", float("nan"))) != 1.0):
        raise RuntimeError("ROUND44 fresh state did not bind eta/version/initial dual")
    payload = {
        "schema": trainer.STATE_SCHEMA,
        "trainer_tag": trainer.TRAINER_TAG,
        "training_version": TRAINING_VERSION,
        "reward_timing_version": trainer.REWARD_TIMING_VERSION,
        "comm_action_eligibility": "first-report-only/1",
        "comm_action_semantics": "one-shot-send-skip/1",
        "eta_lambda": ETA_NEW,
        "loss_version": trainer.LOSS_VERSION,
        "gae_schema": trainer.GAE_SCHEMA,
        "optimizer_schema": trainer.OPTIMIZER_SCHEMA,
        "reward_timing_schema": trainer.REWARD_TIMING_SCHEMA,
        "freeze": frozen["round42"]["freeze_identity"],
        "controller": copy.deepcopy(controller.state_dict()),
        "optimizer": optimizer.state_dict(),
        "training_state": copy.deepcopy(state),
        "rng": trainer.rng_state(),
    }
    identity_before = controller.parameter_summary()["parameter_sha256"]
    try:
        with tempfile.TemporaryDirectory(prefix="r44_version_restore_") as temporary:
            path = Path(temporary) / "exact_round44_state.pt"
            torch.save(payload, path)
            restored_controller, restored_optimizer, restored_state = trainer.load_checkpoint(
                path, freeze=frozen["round42"]["freeze_identity"])
            if (restored_controller.parameter_summary()["parameter_sha256"] != identity_before
                    or restored_state.get("training_version") != TRAINING_VERSION
                    or float(restored_state.get("eta_lambda", float("nan"))) != ETA_NEW
                    or len(restored_optimizer.state) != 0):
                raise RuntimeError("ROUND44 same-version exact state did not restore cleanly")
    finally:
        trainer.restore_rng(ambient)
    return {
        "kind": "round44-same-version-restore-gate/1", "status": "passed",
        "training_version": TRAINING_VERSION, "eta_lambda": ETA_NEW,
        "temporary_fixture_deleted": True, "optimizer_step_calls": 0,
        "restored_parameter_sha256": identity_before,
    }


def _initialization_gate(runtime: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    trainer = runtime["trainer"]
    common = runtime["common"]
    controller_module = runtime["controller_module"]
    records = {}
    for seed in SEEDS:
        ambient = trainer.rng_state()
        source = Path(frozen["round43"]["registered_ep0_inputs"][str(seed)]["path"])
        source_payload = torch.load(source, map_location="cpu", weights_only=False)
        source_rng = source_payload.get("rng")
        expected_rng_sha = frozen["round44"]["ep0_initialization_identity"][str(seed)][
            "rng_pickle_sha256"]
        if (not isinstance(source_rng, dict)
                or hashlib.sha256(pickle.dumps(source_rng, protocol=5)).hexdigest()
                != expected_rng_sha):
            raise RuntimeError(f"seed {seed}: registered Python/NumPy/Torch RNG identity changed")
        controller = controller_module.fresh_controller(seed)
        optimizer = torch.optim.Adam(controller.parameters, lr=float(common.LEARNING_RATE))
        state = trainer.fresh_training_state(seed)
        expected = frozen["round42"]["new_initial_models"][str(seed)]
        parameter_sha = controller.parameter_summary()["parameter_sha256"]
        if (parameter_sha != expected["parameter_sha256"]
                or len(optimizer.state) != 0
                or float(state.get("lambda", float("nan"))) != 1.0
                or float(state.get("eta_lambda", float("nan"))) != ETA_NEW
                or float(state.get("gae_lambda", float("nan"))) != 1.0
                or float(state.get("gamma", float("nan"))) != 1.0
                or state.get("training_version") != TRAINING_VERSION):
            trainer.restore_rng(ambient)
            raise RuntimeError(f"seed {seed}: fresh R44 initialization differs from R43 registered ep0")
        trainer.restore_rng(ambient)
        records[str(seed)] = {
            "source": frozen["round44"]["source_training_version_for_ep0"],
            "registered_ep0_path": frozen["round43"]["registered_ep0_inputs"][str(seed)]["path"],
            "registered_ep0_file_sha256": frozen["round43"]["registered_ep0_inputs"][str(seed)][
                "file_sha256"],
            "registered_and_fresh_parameter_sha256": parameter_sha,
            "empty_adam_state_entries": len(optimizer.state),
            "lambda_dual": float(state["lambda"]),
            "eta_lambda": float(state["eta_lambda"]),
            "gae_lambda": float(state["gae_lambda"]),
            "gamma": float(state["gamma"]),
            "data_order_seed": int(state["data_order_seed"]),
            "registered_rng_pickle_sha256": expected_rng_sha,
            "ambient_rng_restored_after_probe": True,
            "loaded_training_checkpoint": False,
        }
    return {"kind": "round44-ep0-initialization-gate/1", "status": "passed",
            "by_seed": records}


def _dual_update_gate(runtime: dict[str, Any]) -> dict[str, Any]:
    common = runtime["common"]
    trainer = runtime["trainer"]
    if float(common.ETA_LAMBDA) != ETA_NEW:
        raise RuntimeError("active common module does not carry eta_lambda=0.5")
    cases = ((1.0, 0.25, 1.125), (1.0, -0.25, 0.875),
             (1.0, 0.0, 1.0), (0.05, -0.25, 0.0))
    results = []
    for before, g_hat, expected in cases:
        actual = trainer.dual_update(before, g_hat)
        expected_next = max(0.0, before + ETA_NEW * g_hat)
        if (not math.isclose(float(actual["eta"]), ETA_NEW, abs_tol=1e-12)
                or not math.isclose(float(actual["lambda_next"]), expected_next,
                                    rel_tol=0.0, abs_tol=1e-12)
                or not math.isclose(expected_next, expected, rel_tol=0.0, abs_tol=1e-12)):
            raise RuntimeError(f"actual trainer dual_update mismatch: {actual}")
        results.append({"lambda_before": before, "g_hat": g_hat, "eta": actual["eta"],
                        "lambda_next": actual["lambda_next"], "independent_expected": expected})
    return {"kind": "round44-actual-dual-update-gate/1", "status": "passed",
            "cases": results, "actual_function": "v2_trainer.dual_update",
            "optimizer_step_calls": 0}


def run() -> dict[str, Any]:
    prepared = prepare()
    frozen = prepared["frozen"]
    runtime = configure(frozen)
    order = [int(value) for value in runtime["common"].scenario_order()]
    if order != [int(value) for value in frozen["round42"]["training_order"]]:
        raise RuntimeError("ROUND44 active trainer scenario order differs from the R43 registration")
    if len(order) != 128 or len(set(order)) != 48:
        raise RuntimeError("ROUND44 fixed seed order/scenario support is malformed")

    dual = _dual_update_gate(runtime)
    versions = _read_only_loader_gate(runtime, frozen)
    restore = _same_version_restore_gate(runtime, frozen)
    initialization = _initialization_gate(runtime, frozen)
    row_root = RUN_DIR / "rows"
    claim_root = RUN_DIR / "launch_claims"
    checkpoint_root = RUN_DIR / "checkpoints"
    unexpected_rows = list(row_root.glob("*.json")) if row_root.exists() else []
    unexpected_claims = list(claim_root.rglob("attempt_*.json")) if claim_root.exists() else []
    unexpected_checkpoints = list(checkpoint_root.glob("*.pt")) if checkpoint_root.exists() else []
    if unexpected_rows or unexpected_claims or unexpected_checkpoints:
        raise RuntimeError("ROUND44 prelaunch output directory contains experiment rows/claims/checkpoints")
    if runtime["r42_run"]._PPO_UPDATE_CALLS != 0 or runtime["r42_run"]._OPTIMIZER_STEP_CALLS != 0:
        raise RuntimeError("ROUND44 preflight unexpectedly counted a PPO/optimizer update")
    if runtime["r42_versions"].CHECKPOINT_WRITE_CALLS != 0:
        raise RuntimeError("ROUND44 preflight unexpectedly wrote an experimental checkpoint")

    init_path = RUN_DIR / "ROUND44_EP0_INITIALIZATION.json"
    version_path = RUN_DIR / "ROUND44_VERSION_BOUNDARY.json"
    dual_path = RUN_DIR / "ROUND44_DUAL_UPDATE_GATE.json"
    write_new(init_path, initialization)
    write_new(version_path, {**versions, "same_version_restore": restore})
    write_new(dual_path, dual)
    result = {
        "kind": "round44-zero-environment-prelaunch-gates/1",
        "status": "passed",
        "environment_constructor_calls": 0,
        "run_invocation_calls": 0,
        "completed_steps": 0,
        "ppo_update_calls": 0,
        "optimizer_step_calls": 0,
        "experimental_checkpoint_write_calls": 0,
        "synthetic_optimizer_step_calls": 0,
        "synthetic_checkpoint_fixture": "temporary version-boundary file; deleted",
        "source_commit_at_gate": "pre-freeze-commit; will be captured in prelaunch receipt",
        "runner_version": RUNNER_VERSION,
        "training_version": TRAINING_VERSION,
        "eta_lambda": ETA_NEW,
        "gae_lambda": 1.0,
        "gamma": 1.0,
        "training_order_sha256": frozen["round42"]["training_order_sha256"],
        "scenario_order_actual": order,
        "baseline_reuse_counts": prepared["reuse"]["counts"],
        "dual_update_gate": str(dual_path),
        "version_boundary_gate": str(version_path),
        "ep0_initialization_gate": str(init_path),
        "evaluation_index_tests": {"status": "run separately", "environment_starts": 0},
    }
    write_new(RUN_DIR / "ROUND44_ZERO_ENV_GATES.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

