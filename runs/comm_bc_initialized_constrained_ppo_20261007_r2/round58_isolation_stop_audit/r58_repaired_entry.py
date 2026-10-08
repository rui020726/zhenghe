"""Zero-launch runtime binder and fail-closed checkpoint policy for an R58 repair.

This module is deliberately not the fixed-plan executor. It installs the isolated
adapter on an already-created runtime object and separates model-only reads from
training resume eligibility. It never constructs an environment or saves a model.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

import r58_gate_prior_repaired as adapter


ENTRY_KIND = "round58-repaired-runtime-entry/1"
ROUTE_VERSION = "critic_representation_stop_gradient/1"
EXPECTED_TRAINER_SHA256 = "0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841"
EXPECTED_COMMON_SHA256 = "66d019a344d9cfef3c67bd8b722f8cc27144b9f71ddc8e5496451929d12f746a"
EXPECTED_COMMON = {
    "EPISODES_PER_UPDATE": 4,
    "MINIBATCH_STEPS": 256,
    "EPOCHS_PER_BATCH": 4,
    "GAMMA": 1.0,
    "GAE_LAMBDA": 0.95,
    "LEARNING_RATE": 0.0003,
    "CLIP": 0.2,
    "ENTROPY_BONUS": 0.01,
    "VALUE_COEFFICIENT": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "B0_BYTES": 10000.0,
    "ETA_LAMBDA": 0.05,
    "LAMBDA_INIT": 1.0,
    "DATA_ORDER_SEED": 20260927,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapter_identity() -> dict[str, str]:
    path = Path(adapter.__file__).resolve()
    return {"adapter_module": "r58_gate_prior_repaired",
            "adapter_path": str(path), "adapter_sha256": _sha256(path),
            "critic_route_version": ROUTE_VERSION,
            "trainer_sha256": EXPECTED_TRAINER_SHA256,
            "common_sha256": EXPECTED_COMMON_SHA256}


def _unwrap(function: Any) -> Any:
    seen: set[int] = set()
    while callable(function) and id(function) not in seen and hasattr(function, "__wrapped__"):
        seen.add(id(function))
        function = function.__wrapped__
    return function


def bind_repaired_update(runtime: dict[str, Any], raw_dir: Path,
                         critic_input_detached: Any, *, writer=None) -> dict[str, Any]:
    """Bind the repaired adapter to the exact trainer object the R40 adapter calls."""
    trainer = runtime.get("trainer")
    train_adapter = runtime.get("r40_train")
    if trainer is None or train_adapter is None:
        raise RuntimeError("runtime lacks the verified trainer or R40 training adapter")
    if runtime.get("common") is None:
        raise RuntimeError("runtime common module object is missing")
    body = _unwrap(trainer.ppo_update)
    source = Path(body.__code__.co_filename).resolve()
    source_sha = _sha256(source)
    if source_sha != EXPECTED_TRAINER_SHA256:
        raise RuntimeError(f"trainer source identity mismatch: {source_sha}")
    common = body.__globals__.get("common")
    if common is not runtime["common"]:
        raise RuntimeError("trainer globals.common is not the configured runtime common object")
    common_path = Path(common.__file__).resolve()
    common_sha = _sha256(common_path)
    if common_sha != EXPECTED_COMMON_SHA256:
        raise RuntimeError(f"common source identity mismatch: {common_sha}")
    live_values = {key: getattr(common, key, None) for key in EXPECTED_COMMON}
    mismatches = {key: {"expected": value, "actual": live_values[key]}
                  for key, value in EXPECTED_COMMON.items()
                  if live_values[key] != value}
    if mismatches:
        raise RuntimeError(f"live PPO configuration mismatch: {mismatches}")
    if runtime.get("critic_input_detached") is not critic_input_detached:
        raise RuntimeError("runtime critic isolation context identity mismatch")

    # This is the object assigned by the R40 executor immediately before train_seed.
    train_adapter.trainer = trainer
    if train_adapter.trainer is not trainer:
        raise RuntimeError("R40 trainer binding did not retain object identity")
    kwargs = {} if writer is None else {"writer": writer}
    receipt = adapter.install_gate_prior(
        trainer, critic_input_detached, Path(raw_dir), **kwargs)
    runtime["r58_repaired_adapter"] = receipt
    runtime["r58_repair_identity"] = adapter_identity()
    runtime["r58_live_configuration"] = live_values
    runtime["r58_source_objects"] = {
        "trainer_module": trainer.__name__, "trainer_path": str(source),
        "common_module": common.__name__, "common_path": str(common_path),
        "r40_train_module": getattr(train_adapter, "__name__", type(train_adapter).__name__),
        "trainer_object_bound_to_r40": train_adapter.trainer is trainer,
    }
    return {"kind": ENTRY_KIND, "adapter": adapter_identity(),
            "trainer_sha256": source_sha, "common_sha256": common_sha,
            "live_configuration": live_values,
            "trainer_object_bound_to_r40": train_adapter.trainer is trainer,
            "undo": receipt["undo"]}


def model_only_state(payload: dict[str, Any]) -> dict[str, Any]:
    """Return controller weights only; optimizer, RNG, and cursor are never restored."""
    state = payload.get("controller")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no controller state for model-only read")
    return {"kind": "r58-model-only-read/1", "training_version": payload.get("training_version"),
            "controller": state}


def require_repaired_resume(payload: dict[str, Any], expected_identity: dict[str, Any]) -> None:
    """Reject old R58 states even if their semantic version string mentions R53."""
    proof = payload.get("round58_repair_attestation")
    state = payload.get("training_state", {})
    freeze = payload.get("freeze", {})
    if not isinstance(proof, dict):
        raise RuntimeError("resume refused: checkpoint lacks per-update R58 repair attestation")
    if proof != expected_identity:
        raise RuntimeError("resume refused: repair attestation differs from current adapter identity")
    if (state.get("round58_repair_attestation") != expected_identity
            or freeze.get("round58_repair_attestation") != expected_identity):
        raise RuntimeError("resume refused: checkpoint, state, and freeze attestation do not close")
    if proof.get("critic_route_version") != ROUTE_VERSION:
        raise RuntimeError("resume refused: critic isolation route is not verified")


def read_model_only_checkpoint(path: Path) -> dict[str, Any]:
    """CPU deserialization for inference inspection; no optimizer/RNG restore occurs."""
    import torch
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return model_only_state(payload)
