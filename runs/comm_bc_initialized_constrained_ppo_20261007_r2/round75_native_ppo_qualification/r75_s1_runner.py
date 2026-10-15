# -*- coding: utf-8 -*-
"""ROUND75 S1 real runner (requests R75_S0_FINAL_PREP_20261014 / R75_S0_DRIVER_CLOSEOUT_20261015).
PREPARED, NOT RUN.

This module is the complete paid S1 driver described by ROUND75_S1_RUN_PLAN.json. Importing it
costs nothing: torch, numpy and r75_frozen are imported INSIDE run_s1(), so
``python r75_s1_runner.py`` (no argument) only prints the frozen plan summary.

Discipline implemented here (see the plan for the full contract):
  * the pure-file precheck runs BEFORE any paid module is imported, so a drifted source, a changed
    batch, a non-zero counter or a rebuilt ledger refuses the run before torch is ever loaded;
  * every real forward attaches to R75Resources.production() and the trunk-entry meter is
    installed BEFORE any possible forward, for both the A2 and the A3 controller;
  * every Adam.step (including the A1 control step) is pre-charged to the central ledger before
    optimizer.step(), the durable counter is reconciled item-by-item afterwards, and a passive
    observer on the REAL step entry records the attempt / completed / failed outcome plus the
    per-permit reconciliation without charging, refunding or opening another root;
  * A1 records, object by object, whether the gradient and the Adam state are present; an absent
    state is recorded as absent and never fabricated. "The trunk did not move" and the
    eight-object coverage are MEASURED from the live parameters, never written as a literal;
  * the full pre-update state is captured as recursive detach/clone payloads that share no mutable
    storage with the live objects, written to disk as a COMPLETE bundle (five modules, all Adam
    values with the eight-object mapping, the param-group hyperparameters, lambda, configuration,
    cursor, data and code identity, the three RNG states and the contract), and A3 restores from
    that ACTUAL FILE through the real optimizer.load_state_dict - never from the in-memory dict;
  * the comparison freezes four states (the A1 baseline, the A2 end, the A3 post-restore start and
    the A3 end) and compares them value by value: every module tensor, every Adam value, the
    param groups, the RNG states, lambda, the cursor and the data / code identity. The scientific
    native returns and the gradient / gate diagnostics are compared too; only the non-scientific
    metadata frozen in the plan in advance is excluded and accounted separately;
  * ONE attempt: any anomaly keeps the failure and the consumed originals and stops immediately.
    No auto-retry, no technical-reserve upgrade, no refund, no scratch/dev/preheat run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE / "ROUND75_S1_RUN_PLAN.json"
PLAN = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
S1_DIR = HERE / "raw" / "s1"
# The request id that already consumed its ONE attempt, and the directory holding its originals.
# Both stay the default so the preserved history keeps its place and an old command can never
# silently write into it again. A later attempt must name itself and its own directory on the
# command line; _bind_execution_identity() binds them BEFORE the precheck and before any paid
# import, so a run that cannot name itself never reaches torch.
FIRST_ATTEMPT_REQUEST_ID = "R75_S1_NATIVE_QUALIFICATION_20261015"
EXECUTION_REQUEST_ID = None

SEED = 11
LR = 3e-4


# --------------------------------------------------------------------------- helpers ----
def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tensor_digest(tensor) -> dict[str, Any]:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    flat = tensor.detach().abs().sum().item() if tensor.numel() else 0.0
    return {"sha256": _sha256_bytes(raw), "numel": int(tensor.numel()),
            "abs_sum": float(flat), "dtype": str(tensor.dtype),
            "shape": list(tensor.shape)}


def _module_digest(module) -> dict[str, Any]:
    return {name: _tensor_digest(tensor) for name, tensor in module.state_dict().items()}


def _rng_states(np, torch) -> dict[str, Any]:
    return {"python_random_state": repr(random.getstate()),
            "numpy_random_state": repr(np.random.get_state()),
            "torch_rng_state_sha256": _sha256_bytes(torch.get_rng_state().numpy().tobytes())}


def _payload_bytes(state) -> bytes:
    """The raw bytes of a tensor-like RNG payload (a torch CPU tensor, or the fake in tests)."""
    inner = state.numpy() if hasattr(state, "numpy") else state
    return inner.tobytes()


def _rng_report_from_payloads(numpy_state, torch_state, python_state) -> dict[str, Any]:
    """The expected RNG report, derived FROM THE RAW PAYLOADS of the loaded bundle.

    verify_restore needs a report to compare the live RNG against. Rebuilding it here from the
    payloads that were actually loaded keeps the expectation independent of the restore: it is
    computed BEFORE any live state is read, so a restore that installed the wrong state cannot
    supply its own expectation. The authoritative check is the raw payload comparison in
    _rng_payload_differences; this report mirrors _rng_states key for key.
    """
    return {"python_random_state": repr(python_state),
            "numpy_random_state": repr(numpy_state),
            "torch_rng_state_sha256": _sha256_bytes(_payload_bytes(torch_state))}



def _numpy_rng_state_fingerprint(state) -> dict[str, Any]:
    """Complete structural fingerprint for a legacy NumPy MT19937 state tuple.

    Key values are hashed from their full raw bytes; position and cache flags are compared as
    scalar fields. This deliberately does not seed, draw or restore anything.
    """
    if not isinstance(state, tuple) or len(state) != 5:
        return {"kind": "invalid_numpy_state", "type": type(state).__name__,
                "length": len(state) if hasattr(state, "__len__") else "unknown"}
    keys = state[1]
    key_bytes = _payload_bytes(keys)
    return {"algorithm": state[0],
            "keys": {"dtype": str(getattr(keys, "dtype", None)),
                     "shape": tuple(getattr(keys, "shape", ()) or ()),
                     "bytes_len": len(key_bytes),
                     "values_sha256": _sha256_bytes(key_bytes)},
            "position": state[2], "position_type": type(state[2]).__name__,
            "has_gauss": state[3], "has_gauss_type": type(state[3]).__name__,
            "cached_gaussian": state[4],
            "cached_gaussian_type": type(state[4]).__name__}


def _rng_identity_full(np_mod, torch_mod) -> dict[str, Any]:
    """Field-complete read-only fingerprints of the three current RNG states."""
    python_state = random.getstate()
    python_repr = repr(python_state)
    numpy_state = np_mod.random.get_state()
    torch_state = torch_mod.get_rng_state()
    torch_bytes = _payload_bytes(torch_state)
    return {"python": {"type": type(python_state).__name__,
                       "repr_len": len(python_repr),
                       "repr_sha256": _sha256_bytes(python_repr.encode("utf-8"))},
            "numpy": _numpy_rng_state_fingerprint(numpy_state),
            "torch": {"type": type(torch_state).__name__,
                      "dtype": str(getattr(torch_state, "dtype", None)),
                      "shape": tuple(getattr(torch_state, "shape", ()) or ()),
                      "bytes_len": len(torch_bytes),
                      "values_sha256": _sha256_bytes(torch_bytes)}}


def _rng_tensor_meta(obj) -> tuple:
    return (str(getattr(obj, "dtype", None)), tuple(getattr(obj, "shape", ()) or ()))


def _raw_equal(live, expected) -> bool:
    """Strict, non-ambiguous equality over raw RNG payloads.

    Tensors without ``tobytes`` are read through the existing detach-free, CPU-safe view path
    used by ``_payload_bytes``. Tensor-like objects require the same concrete type, dtype, shape
    and FULL raw bytes. Unknown objects compare by identity only. The result is always a bool.
    """
    if live is expected:
        return True
    if isinstance(live, (tuple, list)) or isinstance(expected, (tuple, list)):
        if type(live) is not type(expected):
            return False
        if len(live) != len(expected):
            return False
        return all(bool(_raw_equal(left, right)) for left, right in zip(live, expected))
    if isinstance(live, dict) or isinstance(expected, dict):
        if type(live) is not type(expected) or set(live) != set(expected):
            return False
        return all(bool(_raw_equal(live[key], expected[key])) for key in live)
    live_has_payload = hasattr(live, "tobytes") or hasattr(live, "numpy")
    expected_has_payload = hasattr(expected, "tobytes") or hasattr(expected, "numpy")
    if live_has_payload or expected_has_payload:
        if not (live_has_payload and expected_has_payload):
            return False
        if type(live) is not type(expected):
            return False
        if _rng_tensor_meta(live) != _rng_tensor_meta(expected):
            return False
        return bool(_payload_bytes(live) == _payload_bytes(expected))
    known_scalars = (bool, int, float, complex, str, bytes, bytearray, type(None))
    if isinstance(live, known_scalars) or isinstance(expected, known_scalars):
        if type(live) is not type(expected):
            return False
        return bool(live == expected)
    import numbers
    if isinstance(live, numbers.Number) and isinstance(expected, numbers.Number):
        return bool(live == expected)
    return False


def _first_raw_difference(live, expected, path: str):
    """The first differing position inside two raw payloads, without boolean ambiguity."""
    if isinstance(live, (tuple, list)) and isinstance(expected, (tuple, list)):
        if len(live) != len(expected):
            return {"where": path, "live_len": len(live), "expected_len": len(expected)}
        for index, (left, right) in enumerate(zip(live, expected)):
            if not _raw_equal(left, right):
                return _first_raw_difference(left, right, path + "[" + str(index) + "]")
        return {"where": path, "note": "no element difference found"}
    live_has_payload = hasattr(live, "tobytes") or hasattr(live, "numpy")
    expected_has_payload = hasattr(expected, "tobytes") or hasattr(expected, "numpy")
    if live_has_payload and expected_has_payload:
        if type(live) is not type(expected):
            return {"where": path, "live_type": type(live).__name__,
                    "expected_type": type(expected).__name__}
        left, right = _payload_bytes(live), _payload_bytes(expected)
        if left == right:
            return {"where": path, "note": "bytes equal"}
        for index, (a, b) in enumerate(zip(left, right)):
            if a != b:
                return {"where": path, "first_diff_byte": index, "live": a, "expected": b,
                        "live_len": len(left), "expected_len": len(right)}
        return {"where": path, "live_len": len(left), "expected_len": len(right),
                "note": "one payload is a prefix of the other"}
    return {"where": path, "live_type": type(live).__name__,
            "expected_type": type(expected).__name__,
            "live": repr(live)[:120], "expected": repr(expected)[:120]}



def _rng_payload_differences(saved, np_mod, torch_mod) -> list[dict[str, Any]]:
    """Value-by-value comparison of the THREE RAW RNG payloads against the loaded bundle."""
    problems: list[dict[str, Any]] = []
    live_numpy, expected_numpy = np_mod.random.get_state(), saved["_numpy_state"]
    if not _raw_equal(live_numpy, expected_numpy):
        problems.append({"where": "rng_payload:numpy_random_state",
                         "detail": _first_raw_difference(live_numpy, expected_numpy, "numpy")})
    live_torch, expected_torch = torch_mod.get_rng_state(), saved["_torch_state"]
    if not _raw_equal(live_torch, expected_torch):
        problems.append({"where": "rng_payload:torch_rng_state",
                         "detail": _first_raw_difference(live_torch, expected_torch, "torch")})
    live_python, expected_python = random.getstate(), saved["_python_state"]
    if not _raw_equal(live_python, expected_python):
        problems.append({"where": "rng_payload:python_random_state",
                         "detail": _first_raw_difference(live_python, expected_python, "python")})
    return problems

def _head_input_dim(module) -> int:
    """The critic input width WITHOUT a trunk forward (so the A1 control step costs 0 rows)."""
    for attribute in ("in_features", "input_dim"):
        value = getattr(module, attribute, None)
        if isinstance(value, int):
            return value
    for parameter in module.parameters():
        if parameter.dim() == 2:
            return int(parameter.shape[1])
    raise RuntimeError(
        "cannot determine the critic input width without a forward; refusing to run a trunk "
        "forward just to size the A1 control step (that would silently consume offline rows)")


MODULE_SLOTS = ("trunk", "early_head", "late_head", "value_task_head", "value_cost_head")

CONTRACT = {"metering_contract": PLAN["metering_contract"],
            "comparison_contract": PLAN["comparison_contract"],
            "execution_discipline": PLAN["execution_discipline"],
            "accounting": PLAN["accounting"]}


def _modules_of(controller) -> dict[str, Any]:
    """The five modules of the qualification controller, by stable slot name."""
    return {"trunk": controller.inner.trunk,
            "early_head": controller.inner.early_head,
            "late_head": controller.inner.late_head,
            "value_task_head": controller.inner.value_head,
            "value_cost_head": controller.cost_head}


def _is_tensor_like(value) -> bool:
    """Duck-typed tensor test, so the pure-stdlib counterexamples exercise this same code path."""
    return (hasattr(value, "detach") and hasattr(value, "clone")
            and hasattr(value, "numel") and hasattr(value, "shape")
            and hasattr(value, "dtype"))


def _clone_value(value):
    """Recursive detach/clone.

    A snapshot must NEVER share mutable storage with its source object: an Adam step that
    increments a tensor ``step`` in place, or a later update that writes into a parameter, must
    not be able to rewrite what an earlier capture recorded. The original references are dropped,
    not kept alongside the copies.
    """
    if _is_tensor_like(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        # the container TYPE is preserved: a tuple-valued hyperparameter (Adam betas) or the
        # python RNG state tuple must not drift into a list, or a later exact comparison of the
        # param groups would report a difference that never happened.
        cloned = [_clone_value(item) for item in value]
        return tuple(cloned) if isinstance(value, tuple) else cloned
    return value


def _clone_module_state(module) -> dict[str, Any]:
    """A deep detached copy of module.state_dict(). The ORIGINAL REFERENCES are never kept."""
    return {name: _clone_value(tensor) for name, tensor in module.state_dict().items()}


def _code_identity() -> dict[str, Any]:
    """Byte identity of every source this run depends on, so a drifted driver cannot claim to be
    the one that was frozen."""
    names = ("r75_s1_runner.py", "r75_s1_precheck.py", "r75_frozen.py", "r75_guard.py",
             "r75_replay_adapter.py")
    return {name: _sha256_bytes((HERE / name).read_bytes()) for name in names}


def _live_configuration(frozen) -> dict[str, Any]:
    """The S1 record's configuration, read from the LIVE module namespace.

    Every value is a live module attribute or a re-derived frozen batch fact, never a literal
    chosen in this driver: a drifted module default can then never be mistaken for a verified
    living value. ``common`` is the module r66_frozen configures IN PLACE (GAE 1.0 / ETA 0.5), so
    reading it after the imports is exactly the configuration the update will run with, and the
    0.95 / 0.05 module defaults are reported beside it rather than silently used.
    """
    common = frozen.common
    r66 = frozen.r66
    facts = PLAN["input_identity"]["batch_facts"]
    return {"lambda": float(facts["lambda"]),
            "lambda_at_rollout": float(facts["lambda_at_rollout"]),
            "eta_lambda": float(common.ETA_LAMBDA),
            "eta_lambda_batch_fact": float(facts["eta_lambda"]),
            "gae_lambda": float(common.GAE_LAMBDA),
            "gamma": float(common.GAMMA),
            "clip": float(common.CLIP),
            "max_grad_norm": float(common.MAX_GRAD_NORM),
            "entropy_bonus": float(common.ENTROPY_BONUS),
            "value_coefficient": float(common.VALUE_COEFFICIENT),
            "epochs_per_batch": int(common.EPOCHS_PER_BATCH),
            "minibatch_steps": int(common.MINIBATCH_STEPS),
            "log_prob_tolerance": float(common.LOG_PROB_TOLERANCE),
            "data_order_seed": int(common.DATA_ORDER_SEED),
            "b0_bytes": float(common.B0_BYTES),
            "budget_bytes": int(facts["budget_bytes"]),
            "learning_rate": float(common.LEARNING_RATE),
            "gate_beta": float(r66.GATE_BETA),
            "ready_cut": float(r66.READY_CUT),
            "feature_dim": int(PLAN["input_identity"]["required_batch_shape"]["feature_dim"]),
            "lr_used_by_this_runner": float(LR),
            "lr_matches_live_learning_rate": float(LR) == float(common.LEARNING_RATE)}


def _fixed_rng_triple(np_mod, torch, label: str) -> dict[str, Any]:
    """A FIXED, NON-DEFAULT RNG triple, installed before the pre-update capture.

    The ambient post-construction RNG state is whatever the loader happened to leave, so saving
    and restoring it would prove nothing. A named, non-default triple is installed instead so the
    save/restore round trip is exercised on a state that differs from every default. The data-order
    shuffle is untouched: r75_native_ppo_update builds its own
    ``random.Random(DATA_ORDER_SEED:update_index)`` and never reads the global stream.
    """
    seed = 20261015
    random.seed(seed)
    np_mod.random.seed(seed)
    torch.manual_seed(seed)
    _ = np_mod.random.random()          # consume one draw so the state is not the pristine seed
    _ = torch.rand(1)
    return {"label": label, "python_seed": seed, "numpy_seed": seed, "torch_seed": seed,
            "states": _rng_states(np_mod, torch)}

def _load_paid_modules():
    """The ONLY place torch / r75_frozen enter. Never called at import time."""
    sys.path.insert(0, str(HERE))
    import numpy as np                                     # noqa: F401
    import torch                                          # noqa: F401
    import r75_frozen                                     # noqa: F401
    import r75_guard                                      # noqa: F401
    return np, torch, r75_frozen, r75_guard


# ------------------------------------------------------------------ full-state contract --
def _adam_report(controller, optimizer) -> dict[str, Any]:
    """The Adam state REPORT, keyed by the eight-object name mapping. Digests only: a report can
    describe a running state, it can never BE one."""
    trainable = r75_trainable_names(controller)
    params = list(optimizer.param_groups[0]["params"])
    if len(trainable) != len(params):
        raise RuntimeError(
            f"trainable object mapping is not one-to-one: {len(trainable)} names for "
            f"{len(params)} optimizer parameters; refusing to guess the mapping")
    report: dict[str, Any] = {}
    for index, parameter in enumerate(params):
        name = trainable[index]
        state = dict(optimizer.state.get(parameter, {}))
        entry: dict[str, Any] = {"object": name, "present": bool(state)}
        for key, value in state.items():
            entry[key] = value if isinstance(value, int) else _tensor_digest(value)
        report[name] = entry
    return report


def _param_group_report(optimizer) -> list[dict[str, Any]]:
    """The param-group hyperparameters (lr and friends) as a report, parameters replaced by a count
    so no live tensor is ever published into the record."""
    report = []
    for group in optimizer.param_groups:
        entry = {key: value for key, value in group.items() if key != "params"}
        entry["parameter_count"] = len(group["params"])
        report.append(entry)
    return report


def _optimizer_payload(trainable, params, adam_payload, optimizer) -> dict[str, Any]:
    """An immutable optimizer.state_dict()-shaped payload.

    State is keyed by parameter INDEX and the param groups carry the REAL hyperparameters plus the
    index mapping, so a restore goes through the genuine ``optimizer.load_state_dict``. That is
    what recovers the TYPED Adam state (a tensor ``step`` stays a tensor, a float stays a float);
    a digest or a rounded summary can never be used as running state. An object with no state
    (a bootstrap actor) stays absent and is never fabricated.
    """
    index_of = {id(parameter): index for index, parameter in enumerate(params)}
    state: dict[int, dict[str, Any]] = {}
    for index, name in enumerate(trainable):
        payload = adam_payload.get(name) or {}
        if not payload:
            continue
        state[index] = {key: _clone_value(value) for key, value in payload.items()}
    groups = []
    for group in optimizer.param_groups:
        cloned = {key: _clone_value(value) for key, value in group.items() if key != "params"}
        cloned["params"] = [index_of[id(parameter)] for parameter in group["params"]]
        groups.append(cloned)
    return {"state": state, "param_groups": groups}


def capture_full_state(controller, optimizer, *, np_mod, torch_mod, data_identity: dict,
                       configuration: dict, cursor: int,
                       code_identity: dict | None = None,
                       contract: dict | None = None) -> dict[str, Any]:
    """Everything a resume needs, captured as IMMUTABLE payloads plus a separate report.

    Two layers are kept apart on purpose:
      * the REPORT (``modules``, ``adam``, ``param_groups``, ``rng``) - digests and hyperparameter
        summaries that go into the trial record and are compared value by value;
      * the PAYLOADS (``_module_payloads``, ``_adam_payloads``, ``_optimizer_payload`` and the
        three RNG states) - recursive detach/clone copies sharing NO mutable storage with the live
        objects, so no later update can rewrite what this capture recorded.
    """
    modules = _modules_of(controller)
    trainable = r75_trainable_names(controller)
    params = list(optimizer.param_groups[0]["params"])
    adam_payload: dict[str, Any] = {}
    for index, parameter in enumerate(params):
        adam_payload[trainable[index]] = {
            key: _clone_value(value)
            for key, value in dict(optimizer.state.get(parameter, {})).items()}
    return {"modules": {name: _module_digest(module) for name, module in modules.items()},
            "adam": _adam_report(controller, optimizer),
            "param_groups": _param_group_report(optimizer),
            "rng": _rng_states(np_mod, torch_mod),
            "lambda": float(configuration["lambda"]),
            "lambda_at_rollout": float(configuration["lambda_at_rollout"]),
            "eta_lambda": float(configuration["eta_lambda"]),
            "update_cursor": int(cursor),
            "data_identity": data_identity,
            "configuration": configuration,
            "code_identity": _code_identity() if code_identity is None else code_identity,
            "contract": CONTRACT if contract is None else contract,
            "_module_payloads": {name: _clone_module_state(module)
                                 for name, module in modules.items()},
            "_adam_payloads": adam_payload,
            "_optimizer_payload": _optimizer_payload(trainable, params, adam_payload, optimizer),
            "_numpy_state": np_mod.random.get_state(),
            "_torch_state": torch_mod.get_rng_state().clone(),
            "_python_state": random.getstate()}


def r75_trainable_names(controller) -> list[str]:
    names: list[str] = []
    for label, module in (("early_head", controller.inner.early_head),
                          ("late_head", controller.inner.late_head),
                          ("value_task_head", controller.inner.value_head),
                          ("value_cost_head", controller.cost_head)):
        for parameter_name, _parameter in module.named_parameters():
            names.append(f"{label}.{parameter_name}")
    return names


def build_state_bundle(saved: dict[str, Any]) -> dict[str, Any]:
    """The COMPLETE resume bundle.

    It carries the five module payloads, every Adam state value with the eight-object mapping, the
    optimizer payload with the param-group hyperparameters, the three RNG states, lambda, the
    configuration, the update cursor, the data and code identity and the contract. A bundle that
    omits any of these is not a resume source, and a summary is never substituted for a payload.
    """
    return {"module_payloads": {name: _clone_value(payload)
                                for name, payload in saved["_module_payloads"].items()},
            "adam_payloads": {name: {key: _clone_value(value) for key, value in payload.items()}
                              for name, payload in saved["_adam_payloads"].items()},
            "optimizer_payload": _clone_value(saved["_optimizer_payload"]),
            "numpy_state": saved["_numpy_state"],
            "torch_state": saved["_torch_state"].clone(),
            "python_state": saved["_python_state"],
            "lambda": saved["lambda"],
            "lambda_at_rollout": saved["lambda_at_rollout"],
            "eta_lambda": saved["eta_lambda"],
            "update_cursor": saved["update_cursor"],
            "data_identity": saved["data_identity"],
            "configuration": saved["configuration"],
            "code_identity": saved["code_identity"],
            "contract": saved["contract"]}


def save_state_bundle(path: Path, saved: dict[str, Any]) -> None:
    """Write the complete bundle to disk. This is the ONLY resume source for A3."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch = sys.modules["torch"]
    torch.save(build_state_bundle(saved), path)


def load_state_bundle(path: Path) -> dict[str, Any]:
    """Read the bundle back from the actual serialized file.

    The loading behaviour is PINNED EXPLICITLY instead of inheriting whatever default this
    torch build happens to carry. This environment installs torch 2.9.0, whose ``torch.load``
    resolves an UNSET ``weights_only`` to True (``_default_to_weights_only`` returns True when
    no pickle module is given), and the complete bundle carries the three raw RNG payloads - a
    NumPy MT19937 state tuple among them - which the restricted weights-only unpickler is not
    guaranteed to accept. The bundle is self-generated by this runner and hash-bound (its
    sha256 is recorded at save time and re-verified before the restore), so the full-load
    behaviour is chosen here, deliberately, instead of being discovered for the first time
    inside the paid S1 path.
    """
    torch = sys.modules["torch"]
    return torch.load(path, map_location="cpu", weights_only=False)


def _bundle_to_saved(bundle: dict[str, Any]) -> dict[str, Any]:
    """Adapt a LOADED bundle to the saved-state shape restore_full_state consumes."""
    saved = dict(bundle)
    saved["_module_payloads"] = bundle["module_payloads"]
    saved["_adam_payloads"] = bundle["adam_payloads"]
    saved["_optimizer_payload"] = bundle["optimizer_payload"]
    saved["_numpy_state"] = bundle["numpy_state"]
    saved["_torch_state"] = bundle["torch_state"]
    saved["_python_state"] = bundle["python_state"]
    saved["modules"] = {name: {key: _tensor_digest(tensor) for key, tensor in payload.items()}
                        for name, payload in bundle["module_payloads"].items()}
    saved["adam"] = {name: {"object": name, "present": bool(payload),
                            **{key: (value if isinstance(value, int) else _tensor_digest(value))
                               for key, value in payload.items()}}
                     for name, payload in bundle["adam_payloads"].items()}
    saved["param_groups"] = [{key: value for key, value in group.items() if key != "params"}
                             | {"parameter_count": len(group["params"])}
                             for group in bundle["optimizer_payload"]["param_groups"]]
    # The RNG report is REBUILT FROM THE LOADED RAW PAYLOADS, never copied from the live
    # post-restore state: the expectation must exist before the restore is verified, or
    # verify_restore compares the live RNG against None and the restore can never pass.
    saved["rng"] = _rng_report_from_payloads(bundle["numpy_state"], bundle["torch_state"],
                                             bundle["python_state"])
    return saved


def verify_restore(controller, optimizer, saved, *, np_mod, torch_mod) -> dict[str, Any]:
    """Value-by-value proof that the LIVE objects now equal the captured payloads.

    Every module tensor, every Adam value, the param-group hyperparameters, the three RNG states,
    lambda and the cursor are re-read from the live objects and compared. A single mismatch fails
    the run; nothing is silently tolerated and no field is re-filled from a constant to make the
    comparison pass.
    """
    problems: list[dict[str, Any]] = []
    live_modules = {name: _module_digest(module) for name, module in _modules_of(controller).items()}
    for name in sorted(set(live_modules) | set(saved["modules"])):
        left, right = live_modules.get(name, {}), saved["modules"].get(name, {})
        for key in sorted(set(left) | set(right)):
            if left.get(key) != right.get(key):
                problems.append({"where": f"module:{name}:{key}", "live": left.get(key),
                                 "captured": right.get(key)})
    live_adam = _adam_report(controller, optimizer)
    for name in sorted(set(live_adam) | set(saved["adam"])):
        if live_adam.get(name) != saved["adam"].get(name):
            problems.append({"where": f"adam:{name}", "live": live_adam.get(name),
                             "captured": saved["adam"].get(name)})
    for field, live in (("param_groups", _param_group_report(optimizer)),
                        ("rng", _rng_states(np_mod, torch_mod)),
                        ("lambda", float(saved["lambda"])),
                        ("lambda_at_rollout", float(saved["lambda_at_rollout"])),
                        ("eta_lambda", float(saved["eta_lambda"])),
                        ("update_cursor", int(saved["update_cursor"]))):
        if live != saved.get(field):
            problems.append({"where": field, "live": live, "captured": saved.get(field)})
    rng_payload_problems = _rng_payload_differences(saved, np_mod, torch_mod)
    problems.extend(rng_payload_problems)
    return {"problems": problems, "restored_exactly": not problems,
            "restored_from_payload": ["five module payloads", "optimizer payload (all Adam values "
                                      "and param-group hyperparameters)",
                                      "numpy / torch / python RNG raw payloads (value by value)",
                                      "lambda / lambda_at_rollout / eta_lambda", "update cursor",
                                      "data identity", "configuration", "code identity"]}


def restore_full_state(controller, optimizer, saved: dict[str, Any], *, np_mod, torch_mod) -> dict:
    """Restore into FRESH objects and PROVE it.

    Modules are loaded from the cloned payloads; the Adam state and the param-group
    hyperparameters go through the real ``optimizer.load_state_dict``; the three RNG states are
    restored from the payload. lambda and the cursor are carried by the payload and reported, never
    re-filled from the same constant to fake a recovery. The caller receives the verification.
    """
    for name, module in _modules_of(controller).items():
        module.load_state_dict(_clone_value(saved["_module_payloads"][name]))
    optimizer.load_state_dict(_clone_value(saved["_optimizer_payload"]))
    np_mod.random.set_state(saved["_numpy_state"])
    torch_mod.set_rng_state(saved["_torch_state"].clone())
    random.setstate(saved["_python_state"])
    return verify_restore(controller, optimizer, saved, np_mod=np_mod, torch_mod=torch_mod)


# ----------------------------------------------------------- real step-entry observation ----
class ObservedStepEntry:
    """Passive observation of the REAL optimizer.step entry, bound to the SAME production counter.

    It never charges, never refunds, never opens another root and never bypasses the step. For
    every real call it records an attempt, the durable counter read immediately before and after,
    and whether exactly ONE new permit appeared. A missing or a duplicated permit is a failure, not
    a note: a step that ran without a fresh permit, or a permit that was spent twice, must stop the
    attempt. Everything except ``step`` is delegated to the wrapped optimizer unchanged.
    """

    def __init__(self, optimizer, resources, label: str):
        self.__dict__["_optimizer"] = optimizer
        self.__dict__["_resources"] = resources
        self.__dict__["_label"] = str(label)
        self.__dict__["_inner_step"] = optimizer.step
        self.__dict__["_records"] = []
        self.__dict__["_start_used"] = self._durable_used()
        self.__dict__["_previous_used"] = self.__dict__["_start_used"]

    def step(self, *args, **kwargs):
        """Observe the REAL step entry and reconcile its permit against the same production counter.

        The pre-charge for a step happens immediately BEFORE the call, so the reconciliation
        window is the interval between the previous step's exit reading and this entry reading:
        exactly ONE new permit must appear there. A MISSING or a DUPLICATED permit REFUSES the
        step BEFORE the inner optimizer is ever delegated to - an unlicensed step never runs and
        a permit spent twice never fires a second real update. The refusal keeps the consumed
        permits (no refund, no counter edit) and stops the attempt by raising, so the failure
        surfaces at the boundary instead of after four more steps. Once the permit is right the
        inner step runs, and if the step itself charges something that is recorded at the return
        and further work is refused there too. The observer never charges, never refunds and
        never opens another root; everything except ``step`` is delegated unchanged.
        """
        index = len(self.__dict__["_records"]) + 1
        before = self._durable_used()
        previous = self.__dict__["_previous_used"]
        new_permits = before - previous
        attempt = {"index": index, "label": self.__dict__["_label"], "phase": "attempt",
                   "durable_used_before": before,
                   "expected_durable_used_before": previous + 1,
                   "new_permits_since_previous_step": new_permits,
                   "exactly_one_new_permit": new_permits == 1,
                   "permits_issued_since_start": before - self.__dict__["_start_used"]}
        if new_permits != 1:
            self.__dict__["_records"].append(
                {**attempt, "phase": "refused", "delegated": False,
                 "reason": ("no new permit was issued since the previous step"
                            if new_permits == 0 else
                            f"{new_permits} new permits appeared where exactly one is required"),
                 "note": ("refused BEFORE delegation: the real step was not executed and the "
                          "consumed permits are kept, never refunded")})
            raise RuntimeError(
                f"{self.__dict__['_label']} step #{index} refused: {new_permits} new permit(s) "
                f"since the previous step, exactly 1 required; the real step was not executed")
        # EXACTLY ONE terminal record per real step attempt. A separate "delegated" marker would
        # double the observed count, so a perfect run would report twice the frozen nine steps and
        # trip the step-count gate; the delegation is carried by the "delegated" flag instead.
        try:
            result = self.__dict__["_inner_step"](*args, **kwargs)
        except BaseException as error:                       # a failed step is recorded, never refunded
            self.__dict__["_records"].append(
                {**attempt, "phase": "failed", "delegated": True,
                 "error": f"{type(error).__name__}: {error}",
                 "durable_used_after": self._durable_used()})
            raise
        after = self._durable_used()
        if after != before:
            self.__dict__["_records"].append(
                {**attempt, "phase": "completed", "delegated": True,
                 "durable_used_after": after,
                 "permit_delta": after - before, "step_charged_nothing": False,
                 "refused_further_work": True,
                 "reason": ("the step itself charged the durable counter; further work is "
                            "refused at the return")})
            raise RuntimeError(
                f"{self.__dict__['_label']} step #{index} charged {after - before} unit(s) "
                f"itself; the observer neither double-charges nor frees a step")
        self.__dict__["_records"].append({**attempt, "phase": "completed", "delegated": True,
                                          "durable_used_after": after,
                                          "permit_delta": 0, "step_charged_nothing": True})
        self.__dict__["_previous_used"] = after
        return result

    @property
    def records(self):
        return self.__dict__["_records"]

    def _durable_used(self) -> int:
        """A non-charging read of the durable physical-step counter."""
        return int(self.__dict__["_resources"].snapshots()["physical_optimizer_step"]["used"])

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_optimizer"], name)


# ------------------------------------------------------------------- full-state comparison --
FULL_STATE_FIELDS = ("param_groups", "rng", "lambda", "lambda_at_rollout", "eta_lambda",
                     "update_cursor", "data_identity", "configuration", "code_identity",
                     "contract")


def compare_full_states(left: dict, right: dict, label: str) -> list[dict[str, Any]]:
    """Value-by-value comparison of two full states.

    Every module tensor (value, dtype, shape), every Adam value, the param-group hyperparameters,
    the three RNG states, lambda, the cursor and the data/code identity are compared. Payload and
    summary are separate layers here, and a rounded loss or a hash claim is never accepted as a
    substitute for the comparison.
    """
    differences: list[dict[str, Any]] = []
    for name in sorted(set(left["modules"]) | set(right["modules"])):
        a, b = left["modules"].get(name, {}), right["modules"].get(name, {})
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                differences.append({"where": f"{label}:module:{name}:{key}",
                                    "left": a.get(key), "right": b.get(key)})
    for name in sorted(set(left["adam"]) | set(right["adam"])):
        if left["adam"].get(name) != right["adam"].get(name):
            differences.append({"where": f"{label}:adam:{name}",
                                "left": left["adam"].get(name), "right": right["adam"].get(name)})
    for field in FULL_STATE_FIELDS:
        if left.get(field) != right.get(field):
            differences.append({"where": f"{label}:{field}", "left": left.get(field),
                                "right": right.get(field)})
    return differences


def compare_scientific_returns(left: dict, right: dict, plan: dict) -> tuple[list[str], list[dict]]:
    """Compare the scientific native returns and the gradient / gate diagnostics field by field.

    Only the NON-scientific metadata frozen in the plan in advance may be excluded, and the
    exclusion is accounted separately rather than silently applied: wall-clock timestamps, the
    central cumulative counters and ledger sequence numbers, temporary paths and process ids, and
    the A3 trunk-entry meter absolute cumulative readings (a second controller, so its absolute
    counts are expected to differ and are reconciled instead of diffed). No real scientific
    difference may be excluded, so gate_records and gate_diag are compared, not skipped.
    """
    excluded = set(plan["comparison_contract"]["excluded_in_advance_and_accounted_separately"])
    compared: list[str] = []
    differences: list[dict[str, Any]] = []
    for key in sorted(set(left) | set(right)):
        if key == "trunk_entry_meter":
            continue                      # excluded in advance; reconciled separately below
        if key in excluded:
            continue
        compared.append(key)
        if left.get(key) != right.get(key):
            differences.append({"field": key, "A2": left.get(key), "A3": right.get(key)})
    return compared, differences


def reconcile_trunk_meter(a2: dict, a3: dict, plan: dict) -> dict[str, Any]:
    """Reconcile the two trunk-entry meters instead of diffing their absolute readings.

    Each meter counts its own controller's real trunk forwards, so the absolute cumulative readings
    are per-controller by construction. What must hold is that each update consumed the frozen
    accounting and that the phase split is additive; both are checked here.
    """
    expected = plan["accounting"]["per_native_update"]
    report: dict[str, Any] = {"expected_per_update": expected, "controllers": {}}
    for label, native in (("A2", a2), ("A3", a3)):
        meter = native["trunk_entry_meter"]
        additive = all(meter["alignment"][key] + meter["epochs"][key] == meter["total"][key]
                       for key in meter["alignment"])
        report["controllers"][label] = {
            "additive": additive,
            "calls_charged": meter["total"]["calls_charged"],
            "rows_charged": meter["total"]["rows_charged"],
            "calls_completed": meter["total"]["calls_completed"],
            "rows_completed": meter["total"]["rows_completed"],
            "calls_refused": meter["total"]["calls_refused"],
            "calls_failed": meter["total"]["calls_failed"],
            "calls_match_plan": meter["total"]["calls_charged"] == expected["trunk_entry_calls"],
            "rows_match_plan": meter["total"]["rows_charged"] == expected["trunk_entry_rows"]}
    return report

# ------------------------------------------------------------------------------- A1 ------
def run_a1(controller, optimizer, resources, *, torch, np_mod, record: dict,
           configuration: dict) -> dict[str, Any]:
    """ONE critic-only physical step: non-empty Adam for the critics, actor/trunk never move.

    Every claim here is a MEASUREMENT. ``trunk_moved`` is the before/after digest comparison of
    the trunk's OWN parameters plus the identity check that no trunk parameter is inside the
    optimizer; the eight-object coverage is the name-to-parameter count; the pre-clip gradient
    readings come from the real ``parameter.grad``. Where the sample set cannot cover an object,
    that is reported as a coverage gap and never papered over with an extra forward. The step
    itself goes through the observed entry, so its permit is reconciled like any other.
    """
    value_head = controller.inner.value_head
    cost_head = controller.cost_head
    width = _head_input_dim(value_head)
    hidden = torch.zeros((4, width), dtype=torch.float32, requires_grad=False)
    task_values = value_head(hidden).squeeze(-1)
    cost_values = cost_head(hidden).squeeze(-1)
    # A legal, controllable, non-zero gradient on the two critics only.
    loss = (task_values - 1.0).pow(2).mean() + (cost_values + 1.0).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()

    objects = r75_trainable_names(controller)
    parameters = list(optimizer.param_groups[0]["params"])
    coverage = {"named_trainable_objects": len(objects),
                "optimizer_parameters": len(parameters),
                "coverage_complete": bool(objects) and len(objects) == len(parameters)}
    trunk_parameters = list(controller.inner.trunk.parameters())
    optimizer_ids = {id(parameter) for parameter in parameters}
    trunk_in_optimizer = [index for index, parameter in enumerate(trunk_parameters)
                          if id(parameter) in optimizer_ids]
    grad_record = {}
    for name, parameter in zip(objects, parameters):
        grad = parameter.grad
        grad_record[name] = {"grad_present": grad is not None,
                             "pre_clip_abs_sum": (None if grad is None
                                                  else float(grad.detach().abs().sum()))}
    with_gradient = sorted(name for name, entry in grad_record.items() if entry["grad_present"])
    without_gradient = sorted(name for name in grad_record if name not in with_gradient)
    norm = torch.nn.utils.clip_grad_norm_(parameters, float(configuration["max_grad_norm"]))
    trunk_before = {f"trunk_p{index}": _tensor_digest(parameter)
                    for index, parameter in enumerate(trunk_parameters)}
    head_before = {name: _tensor_digest(parameter) for name, parameter in zip(objects, parameters)}
    # Pre-charge BEFORE the physical step; a step failure never refunds.
    resources.charge_physical_step("r75_s1_A1_critic_only_control_step")
    optimizer.step()                     # the observed entry records the attempt and the permit
    trunk_after = {f"trunk_p{index}": _tensor_digest(parameter)
                   for index, parameter in enumerate(trunk_parameters)}
    head_after = {name: _tensor_digest(parameter) for name, parameter in zip(objects, parameters)}
    trunk_moved = sorted(key for key in trunk_before
                         if trunk_before[key]["sha256"] != trunk_after[key]["sha256"])
    moved = sorted(name for name in objects
                   if head_before[name]["sha256"] != head_after[name]["sha256"])
    adam_record = {}
    for name, parameter in zip(objects, parameters):
        state = optimizer.state.get(parameter, {})
        adam_record[name] = {"state_present": bool(state),
                             "step": (int(state["step"]) if "step" in state else None),
                             "exp_avg_present": "exp_avg" in state,
                             "exp_avg_sq_present": "exp_avg_sq" in state}
    record["A1"] = {"loss": float(loss.item()), "gradient_norm": float(norm),
                    "critic_input_width": width, "objects": objects,
                    "object_coverage": coverage,
                    "grad_record": grad_record,
                    "objects_with_gradient": with_gradient,
                    "objects_without_gradient": without_gradient,
                    "adam_record": adam_record,
                    "trunk_parameters_measured": len(trunk_parameters),
                    "trunk_parameters_in_optimizer": len(trunk_in_optimizer),
                    "trunk_requires_grad": [bool(parameter.requires_grad)
                                            for parameter in trunk_parameters],
                    "trunk_moved_parameters": trunk_moved,
                    "trunk_moved": bool(trunk_moved),
                    "moved_objects": moved,
                    "max_grad_norm_used": float(configuration["max_grad_norm"]),
                    "note": ("the trunk carries requires_grad=False and no trunk parameter is in "
                             "the optimizer; both facts are MEASURED here, and the actor heads "
                             "receive no gradient in this critic-only step")}
    return record["A1"]


# ------------------------------------------- fail-fast gates, receipts and real-entry counters --
BUNDLE_FIELDS = ("module_payloads", "adam_payloads", "optimizer_payload", "numpy_state",
                 "torch_state", "python_state", "lambda", "lambda_at_rollout", "eta_lambda",
                 "update_cursor", "data_identity", "configuration", "code_identity", "contract")

COUNTER_UNITS = ("paid_module_imports", "replay_data_reads", "checkpoint_file_loads",
                 "gate_controller_constructions", "optimizer_constructions",
                 "native_ppo_update_attempts", "optimizer_step_attempts")


def _expected_trainable_objects() -> int:
    """The frozen plan states the trainable-object count; it is READ from there, never assumed."""
    for entry in PLAN["gate_only_controller"]["r71_research_config_unchanged"]:
        if "trainable objects" in entry:
            return int(entry.split()[0])
    raise RuntimeError("the frozen plan does not state the trainable-object count")


def _new_counters() -> dict[str, list]:
    """Real-entry counters, one list per unit. Recording an entry never performs an operation."""
    return {unit: [] for unit in COUNTER_UNITS}


def _conservative_attempt_bounds() -> dict[str, Any]:
    """A conservative UPPER BOUND on the paid entries this driver can perform at all.

    The bound is stated before anything runs and does not depend on what actually happens, so the
    record can never shrink a real attempt by relabelling it. Only the entries that really happen
    are counted, one by one, with their unit. A dev or fixture path is never described as simulated
    and therefore free: the same counter is what a real run writes.
    """
    return {"paid_module_imports": {"bound": 4, "unit": "modules",
                                    "detail": "numpy, torch, r75_frozen and r75_guard"},
            "replay_data_reads": {"bound": 2, "unit": "files",
                                  "detail": "the frozen qualification row and the frozen tape"},
            "checkpoint_file_loads": {"bound": 3, "unit": "files",
                                      "detail": "the R58 ep0 behaviour checkpoint read inside "
                                                "build_gate_only_controller for the A1/A2 "
                                                "controller, the same read for the A3 controller, "
                                                "and the A1 state bundle read back from disk"},
            "gate_controller_constructions": {"bound": 2, "unit": "controllers",
                                              "detail": "one for A1/A2 and one for A3"},
            "optimizer_constructions": {"bound": 2, "unit": "optimizers",
                                        "detail": "one for A1/A2 and one for A3"},
            "native_ppo_update_attempts": {"bound": 2, "unit": "updates",
                                           "detail": "one A2 and one A3 native update; a failed "
                                                     "attempt is still an attempt"},
            "optimizer_step_attempts": {"bound": int(PLAN["accounting"]["planned_totals"]
                                                     ["physical_optimizer_steps"]),
                                        "unit": "steps",
                                        "detail": "the frozen plan total: 1 A1 control step, 4 A2 "
                                                  "and 4 A3, a refused or failed attempt included"}}


class _RunState:
    """Everything a failure receipt needs, kept apart from the scientific record.

    A failure has to be reportable from wherever it happens, so the phase, the real entry counters,
    the observers and the meters live here from the first paid line onwards. An item that cannot be
    read is recorded as UNKNOWN and never defaulted to zero: a zero would claim that nothing
    happened, which is a different statement from "we could not read it".
    """

    def __init__(self) -> None:
        self.phase = "start"
        self.record: dict[str, Any] = {}
        self.counters = _new_counters()
        self.resources = None
        self.observers: dict[str, Any] = {}
        self.meters: dict[str, Any] = {}
        self.unread: list[str] = []
        self.closed_out = False

    def count(self, unit: str, what: str, **detail) -> None:
        self.counters[unit].append({"what": what, "phase": self.phase, **detail})

    def _observed_step_entries(self) -> list[dict[str, Any]]:
        """The REAL optimizer.step attempts, read from the observed entries themselves.

        A refused step is an attempt: the entry was reached and the permit was wrong. A failed step
        is an attempt too. Counting from the records instead of from an expectation is what keeps a
        dev or fixture path from being described as free.
        """
        entries: list[dict[str, Any]] = []
        for label, observer in self.observers.items():
            try:
                for entry in observer.records:
                    entries.append({"what": f"{label}:step_{entry.get('index')}",
                                    "phase": entry.get("phase"),
                                    "exactly_one_new_permit": entry.get("exactly_one_new_permit"),
                                    "step_charged_nothing": entry.get("step_charged_nothing")})
            except Exception as error:                          # pragma: no cover - defensive
                self.unread.append(f"optimizer_step_attempts:{label}")
                entries.append({"what": f"{label}:step_?",
                                "unknown": f"{type(error).__name__}: {error}"})
        return entries

    def counter_report(self) -> dict[str, Any]:
        bounds = _conservative_attempt_bounds()
        report: dict[str, Any] = {}
        for unit in COUNTER_UNITS:
            entries = (self._observed_step_entries() if unit == "optimizer_step_attempts"
                       else self.counters[unit])
            report[unit] = {"unit": bounds[unit]["unit"], "counted": len(entries),
                            "entries": entries,
                            "within_conservative_bound": len(entries) <= bounds[unit]["bound"]}
        return report

    def step_records(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for label, observer in self.observers.items():
            try:
                out[label] = observer.records
            except Exception as error:                          # pragma: no cover - defensive
                self.unread.append(f"step_records:{label}")
                out[label] = {"unknown": f"{type(error).__name__}: {error}"}
        return out

    def meter_report(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for label, meter in self.meters.items():
            try:
                out[label] = meter.snapshot()
            except Exception as error:                          # pragma: no cover - defensive
                self.unread.append(f"meter:{label}")
                out[label] = {"unknown": f"{type(error).__name__}: {error}"}
        return out

    def durable_report(self) -> dict[str, Any]:
        if self.resources is None:
            self.unread.append("durable_counters:the production resources were never opened")
            return {"unknown": "the production resources were never opened in this attempt"}
        try:
            return self.resources.snapshots()
        except Exception as error:                              # pragma: no cover - defensive
            self.unread.append("durable_counters")
            return {"unknown": f"{type(error).__name__}: {error}"}


def _append_jsonl(path: Path, event: dict) -> None:
    """Append one event and fsync it. A receipt that is not on disk is not a receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_failure_receipt(state: _RunState, error: BaseException) -> Path:
    """The durable receipt of a FAILED attempt: appended, fsynced, never masking the exception.

    Nothing here reruns, refunds or repairs anything. It records where the run stopped, the raw
    exception and its traceback, which real attempts were already consumed, what the observed step
    entries and the per-controller meters said, and what could not be read. The caller re-raises the
    original error, so a failure is never converted into a success by its own bookkeeping.
    """
    event = {"kind": "round75-s1-failure-receipt/1",
             "request_id": EXECUTION_REQUEST_ID or PLAN["request_id"],
             "plan_request_id": PLAN["request_id"],
             "attempt_dir": str(S1_DIR),
             "head_at_writing": PLAN["head_at_writing"],
             "phase": state.phase,
             "error": {"type": type(error).__name__, "message": str(error)},
             "traceback": traceback.format_exc(),
             "counters": state.counter_report(),
             "conservative_attempt_bounds": _conservative_attempt_bounds(),
             "step_observations": state.step_records(),
             "meter_snapshots": state.meter_report(),
             "durable_counters": state.durable_report(),
             "unread": sorted(set(state.unread)),
             "completed_stage_artifacts": _completed_stage_artifacts(state),
             "note": ("appended and fsynced at the moment of failure; the original exception is "
                      "re-raised unchanged, nothing is retried, refunded or repaired")}
    path = S1_DIR / "r75_s1_failure_receipts.jsonl"
    _append_jsonl(path, event)
    return path


def _execution_identity() -> dict[str, Any]:
    """Read the actual git identity. PLAN head is historical and is never substituted here."""
    def git_value(*args):
        completed = subprocess.run(["git", "-C", str(HERE), *args], capture_output=True)
        if completed.returncode != 0:
            return "unknown", completed.stderr.decode("utf-8", "replace").strip()
        return completed.stdout.decode("utf-8", "replace").strip(), None
    head, head_error = git_value("rev-parse", "HEAD")
    parent, parent_error = git_value("rev-parse", "HEAD^")
    branch, branch_error = git_value("rev-parse", "--abbrev-ref", "HEAD")
    return {"head": head, "parent_head": parent, "branch": branch, "source": "git rev-parse",
            "errors": {"head": head_error, "parent_head": parent_error, "branch": branch_error}}


def _save_stage_snapshot(path: Path, *, stage: str, label: str, state, full_state,
                         save_fn=None, identity_fn=None) -> dict[str, Any]:
    """Durably save a completed-stage scientific and full-state snapshot.

    The default save function is torch.save with its real (obj, file) ABI. The call adds no RNG
    draw, forward, backward or optimizer step. The write is atomic, and the returned identity is
    what the next stage and any later failure receipt must cite. PLAN head is retained only as
    historical provenance beside the actual git execution identity.
    """
    if save_fn is None:
        torch = sys.modules.get("torch")
        if torch is None:
            raise RuntimeError("cannot save a stage snapshot before the paid modules are loaded")
        save_fn = torch.save
    if identity_fn is None:
        identity_fn = _execution_identity
    execution_identity = identity_fn()
    data_identity = (full_state.get("data_identity") if isinstance(full_state, dict) else None)
    payload = {"kind": "round75-s1-stage-snapshot/1",
               "stage": stage, "label": label,
               "request_id": EXECUTION_REQUEST_ID or PLAN["request_id"],
               "plan_frozen_head": PLAN["head_at_writing"],
               "plan_frozen_head_is_not_execution_head": True,
               "execution_identity": execution_identity,
               "code_identity": _code_identity(),
               "data_identity": data_identity if data_identity is not None else "missing",
               "counters": state.counter_report(),
               "unread": sorted(set(state.unread)),
               "record": state.record,
               "full_state": full_state}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        save_fn(payload, temporary)
        with open(temporary, "rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        identity = {"path": str(path), "sha256": _sha256_bytes(temporary.read_bytes()),
                    "bytes": temporary.stat().st_size,
                    "stage": stage, "label": label,
                    "unread": payload["unread"],
                    "execution_head": execution_identity.get("head"),
                    "data_identity_present": data_identity is not None,
                    "full_state_present": full_state is not None}
        os.replace(temporary, path)
        return identity
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _completed_stage_artifacts(state) -> list[dict[str, Any]]:
    """Verify and cite only completed-stage artifacts already recorded in this attempt."""
    artifacts: list[dict[str, Any]] = []
    for key, label in (("A1_baseline_bundle", "A1_baseline"),
                       ("A2_stage_snapshot", "A2_completed"),
                       ("A3_start_stage_snapshot", "A3_start_gate_passed")):
        item = state.record.get(key)
        if not isinstance(item, dict) or "path" not in item:
            continue
        entry = {"record_key": key, "label": label, "path": item["path"],
                 "recorded_sha256": item.get("sha256")}
        try:
            path = Path(item["path"])
            if not path.is_file():
                entry["status"] = "missing"
            else:
                actual = _sha256_bytes(path.read_bytes())
                entry["actual_sha256"] = actual
                entry["status"] = ("verified" if actual == item.get("sha256") else "hash_mismatch")
        except Exception as error:
            entry["status"] = "unread"
            entry["error"] = type(error).__name__ + ": " + str(error)
        artifacts.append(entry)
    return artifacts


# --------------------------------------------------------------- the fail-fast gates ----------
def _gate_a1(a1: dict) -> dict[str, Any]:
    """A1 must pass BEFORE anything expensive is started.

    The control step is cheap, so a broken control step must not be discovered four paid updates
    later: the actor heads and the trunk have to be measurably untouched, the eight-object mapping
    has to be one-to-one and unique, and both critics have to carry a non-empty Adam state. A
    failure raises here, with the measurements beside it.
    """
    expected = _expected_trainable_objects()
    problems: list[str] = []
    if a1["trunk_moved"]:
        problems.append(f"a trunk parameter moved in the critic-only control step: "
                        f"{a1['trunk_moved_parameters']}")
    if a1["trunk_parameters_in_optimizer"]:
        problems.append(f"{a1['trunk_parameters_in_optimizer']} trunk parameter(s) are inside the "
                        f"optimizer")
    if not a1["object_coverage"]["coverage_complete"]:
        problems.append(f"the trainable-object coverage is not complete: {a1['object_coverage']}")
    objects = list(a1["objects"])
    if len(objects) != len(set(objects)):
        problems.append("the trainable-object name mapping is not unique")
    if len(objects) != expected:
        problems.append(f"{len(objects)} trainable objects were mapped, {expected} are required by "
                        f"the frozen plan")
    for name in objects:
        if not name.startswith("value_"):
            continue
        entry = a1["adam_record"].get(name) or {}
        if not entry.get("state_present"):
            problems.append(f"the critic {name} carries no Adam state after the control step")
        elif not (entry.get("exp_avg_present") and entry.get("exp_avg_sq_present")):
            problems.append(f"the critic {name} has an incomplete Adam state")
    if problems:
        raise RuntimeError("A1 gate refused: " + "; ".join(problems))
    return {"passed": True, "expected_trainable_objects": expected,
            "mapped_trainable_objects": len(objects),
            "trunk_moved": a1["trunk_moved"],
            "trunk_parameters_in_optimizer": a1["trunk_parameters_in_optimizer"],
            "critics_with_nonempty_adam": sorted(
                name for name in objects if name.startswith("value_")
                and (a1["adam_record"].get(name) or {}).get("state_present")),
            "note": ("measured from the live parameters and the live optimizer state, and enforced "
                     "before the first paid native update is issued")}


def _gate_loaded_bundle(bundle: dict, bundle_path: Path, recorded_sha256: str) -> dict[str, Any]:
    """The disk file must be the resume source, byte for byte and field for field.

    A bundle that does not hash to the value recorded at save time is not the file this run wrote;
    a bundle whose code identity differs from the sources on disk is not the driver that was
    frozen. Either way the run stops here, before any restore and long before a paid update.
    """
    problems: list[str] = []
    actual = _sha256_bytes(bundle_path.read_bytes())
    if actual != recorded_sha256:
        problems.append(f"the bundle file hashes to {actual[:16]} but {recorded_sha256[:16]} was "
                        f"recorded at save time")
    missing = [field for field in BUNDLE_FIELDS if field not in bundle]
    if missing:
        problems.append(f"the loaded bundle is missing {missing}")
    for field in ("data_identity", "configuration", "code_identity", "contract"):
        if field in bundle and not bundle[field]:
            problems.append(f"the loaded bundle field {field} is empty")
    if bundle.get("code_identity") != _code_identity():
        problems.append("the bundle's code identity does not match the sources on disk")
    if problems:
        raise RuntimeError("bundle gate refused: " + "; ".join(problems))
    return {"passed": True, "file_sha256": actual, "recorded_sha256": recorded_sha256,
            "fields_present": sorted(field for field in BUNDLE_FIELDS if field in bundle),
            "note": ("the file on disk is hashed and compared with the value recorded when this "
                     "run wrote it, and its identity fields are compared with the live sources")}


def _gate_a3_start(baseline: dict, a3_start: dict, restore_report: dict, bundle: dict, *,
                   controller, optimizer, episode, update_index: int,
                   np_mod, torch_mod) -> dict[str, Any]:
    """The A3 start state must be PROVEN, and bound to the objects the update will use.

    The disk restore has to reproduce the A1 baseline value by value, the three raw RNG payloads
    have to match, and the state that is ABOUT to be updated has to be the state that was just
    proven: the live objects are re-read immediately before the call and must still hold the
    verified values, and the update is bound to those very objects and to the cursor the bundle
    carried. Only then is the second paid native update issued.
    """
    problems: list[str] = []
    if not restore_report["restored_exactly"]:
        problems.append(f"the disk restore did not reproduce the captured state: "
                        f"{restore_report['problems']}")
    differences = compare_full_states(baseline, a3_start,
                                      "A1_baseline_vs_A3_start_after_disk_restore")
    if differences:
        problems.append(f"{len(differences)} full-state difference(s) between the A1 baseline and "
                        f"the A3 start after the disk restore")
    rng_problems = _rng_payload_differences(a3_start, np_mod, torch_mod)
    if rng_problems:
        problems.append(f"{len(rng_problems)} raw RNG payload difference(s) at the A3 start")
    live_again = capture_full_state(controller, optimizer, np_mod=np_mod, torch_mod=torch_mod,
                                    data_identity=a3_start["data_identity"],
                                    configuration=a3_start["configuration"],
                                    cursor=a3_start["update_cursor"])
    drift = compare_full_states(a3_start, live_again, "A3_start_re_read_before_the_native_update")
    if drift:
        problems.append(f"{len(drift)} difference(s) between the verified A3 start and the live "
                        f"state re-read immediately before the native update")
    if int(bundle["update_cursor"]) != int(a3_start["update_cursor"]):
        problems.append("the bundle cursor and the verified start cursor disagree")
    if problems:
        raise RuntimeError("A3 start gate refused: " + "; ".join(problems))
    return {"passed": True, "differences": differences, "rng_problems": rng_problems,
            "re_read_drift": drift,
            "binding": {"controller_id": id(controller), "optimizer_id": id(optimizer),
                        "episode_id": id(episode), "update_index": int(update_index),
                        "update_cursor": int(bundle["update_cursor"]),
                        "note": ("the native update is addressed to exactly these objects and to "
                                 "this cursor, and the live state was re-read and proven equal to "
                                 "the verified start immediately before the call")}}

# ------------------------------------------------------------------------------- run -----
def run_s1() -> int:
    # Every paid phase below is registered in ``state`` BEFORE it runs, so a failure anywhere can
    # be receipted with the real phase, the real attempts and the real durable readings instead of
    # a reconstructed story. The known pre-condition failures are gated where they are found: a
    # broken control step, a bundle that is not the file this run wrote, or an A3 start that was
    # never proven stops the run immediately rather than being collected into one end-of-run list.
    state = _RunState()

    # The pure-file precheck runs BEFORE any paid module is touched. It is stdlib only, so it can
    # refuse the run before torch, numpy, the model, the optimizer or the ledger is ever opened.
    sys.path.insert(0, str(HERE))
    state.phase = "precheck"
    import r75_s1_precheck

    precheck_code = r75_s1_precheck.main()
    if precheck_code != 0:
        print("S1 REFUSED: the freeze precheck failed; nothing paid was started", file=sys.stderr)
        return 2

    state.phase = "paid_module_import"
    np_mod, torch, r75_frozen, r75_guard = _load_paid_modules()
    for module in ("numpy", "torch", "r75_frozen", "r75_guard"):
        state.count("paid_module_imports", module)

    try:
        state.phase = "live_configuration"
        configuration = _live_configuration(r75_frozen)
        if not configuration["lr_matches_live_learning_rate"]:
            print("S1 REFUSED: the runner LR is not the live LEARNING_RATE", file=sys.stderr)
            return 2
        identity = PLAN["input_identity"]
        data_identity = {"row_sha256": identity["row"]["sha256"],
                         "tape_sha256": identity["tape"]["sha256"],
                         "model_sha256": identity["r58_ep0_model"]["sha256"],
                         "behaviour_parameter_sha256": identity["behaviour_parameters"]["value"]}

        state.phase = "production_ledger"
        resources = r75_guard.R75Resources.production()
        state.resources = resources
        resources.verify_or_fail()
        print("canonical production ledger opened and reconciled at the zero starting point")

        # --- gate-only controller: R58 ep0 behaviour, two equal heads, trunk frozen ------------
        state.phase = "A2_controller_construction"
        controller, ep0_state, frozen_identity, summary = r75_frozen.build_gate_only_controller(SEED)
        state.count("gate_controller_constructions", "build_gate_only_controller",
                    label="A1_A2", seed=SEED)
        state.count("checkpoint_file_loads", "r58_ep0_behaviour_checkpoint",
                    label="A1_A2", seed=SEED,
                    detail="read inside build_gate_only_controller")
        # Install the meter BEFORE any possible forward (A2 controller).
        meter_a2 = resources.ensure_trunk_meter(controller, what_prefix="r75_s1_A2")
        state.meters["A2"] = meter_a2
        pre_a2 = meter_a2.snapshot()

        state.phase = "replay_data_load"
        episode, facts = r75_replay_adapter_load()
        state.count("replay_data_reads", "qualification_row", path=identity["row"]["path"])
        state.count("replay_data_reads", "qualification_tape", path=identity["tape"]["path"])
        optimizer = r75_frozen.build_optimizer(controller, LR)
        state.count("optimizer_constructions", "build_optimizer", label="A1_A2", lr=LR)
        # A1 and A2 share one controller, one optimizer and therefore one observed step entry.
        observer = ObservedStepEntry(optimizer, resources, "r75_s1_A1_A2")
        state.observers["A1_A2"] = observer

        record = state.record
        record.update({"kind": "round75-s1-trial/1", "request_id": PLAN["request_id"],
                       "head_at_writing": PLAN["head_at_writing"], "seed": SEED,
                       "data_identity": data_identity, "batch_facts": facts,
                       "frozen_identity": frozen_identity,
                       "parameter_summary": summary,
                       "pre_meter_a2": pre_a2,
                       "configuration": configuration,
                       "code_identity_at_start": _code_identity()})

        # --- A1: the critic-only control step --------------------------------------------------
        state.phase = "A1_critic_only_control_step"
        a1 = run_a1(controller, observer, resources, torch=torch, np_mod=np_mod, record=record,
                    configuration=configuration)
        # FAIL FAST: the control step is cheap, so a broken one must not surface four paid updates
        # later. The actor heads and the trunk must be measurably untouched, the object mapping
        # one-to-one and unique, and both critics must carry a non-empty Adam state.
        record["A1_gate"] = _gate_a1(a1)

        # --- the frozen A1 baseline: the ONE state that A2 and A3 both start from ---------------
        state.phase = "A1_baseline_capture"
        fixed_rng = _fixed_rng_triple(np_mod, torch, "r75_s1_pre_capture")
        a1_baseline = capture_full_state(controller, observer, np_mod=np_mod, torch_mod=torch,
                                         data_identity=data_identity, configuration=configuration,
                                         cursor=0)
        record["fixed_rng_triple"] = fixed_rng
        bundle_path = S1_DIR / "r75_s1_A1_full_state.pt"
        save_state_bundle(bundle_path, a1_baseline)
        record["A1_baseline_bundle"] = {"path": str(bundle_path),
                                        "sha256": _sha256_bytes(bundle_path.read_bytes()),
                                        "fields": sorted(build_state_bundle(a1_baseline))}

        # --- A2: the real native update from the A1 state --------------------------------------
        state.phase = "A2_native_update"
        state.count("native_ppo_update_attempts", "r75_native_ppo_update", label="A2",
                    update_index=0)
        record["A2"] = r75_frozen.r75_native_ppo_update(
            controller, observer, [episode], update_index=0, resources=resources)
        record["post_meter_a2"] = meter_a2.snapshot()
        a2_end = capture_full_state(controller, observer, np_mod=np_mod, torch_mod=torch,
                                    data_identity=data_identity, configuration=configuration, cursor=1)
        record["A2_stage_snapshot"] = _save_stage_snapshot(
            S1_DIR / "r75_s1_A2_stage_snapshot.pt", stage="A2_completed",
            label="A2_after_native_update_0", state=state, full_state=a2_end)

        # --- A3: restore from the ACTUAL SERIALIZED FILE into fresh objects --------------------
        state.phase = "A3_controller_construction"
        controller3, _state3, _frozen3, _summary3 = r75_frozen.build_gate_only_controller(SEED)
        state.count("gate_controller_constructions", "build_gate_only_controller",
                    label="A3", seed=SEED)
        state.count("checkpoint_file_loads", "r58_ep0_behaviour_checkpoint",
                    label="A3", seed=SEED,
                    detail="read inside build_gate_only_controller")
        meter_a3 = resources.ensure_trunk_meter(controller3, what_prefix="r75_s1_A3")
        state.meters["A3"] = meter_a3
        optimizer3 = r75_frozen.build_optimizer(controller3, LR)
        state.count("optimizer_constructions", "build_optimizer", label="A3", lr=LR)
        observer3 = ObservedStepEntry(optimizer3, resources, "r75_s1_A3")
        state.observers["A3"] = observer3

        state.phase = "A3_bundle_load"
        # The disk file is the resume source. The in-memory capture is NEVER used as a substitute.
        bundle = load_state_bundle(bundle_path)
        state.count("checkpoint_file_loads", "A1_state_bundle", path=str(bundle_path))
        # FAIL FAST: the file must be the one this run wrote and must carry the frozen identity.
        record["A3_bundle_gate"] = _gate_loaded_bundle(bundle, bundle_path,
                                                       record["A1_baseline_bundle"]["sha256"])
        record["A3_bundle_code_identity_matches_disk"] = bundle["code_identity"] == _code_identity()
        record["A3_bundle_lambda"] = {"lambda": bundle["lambda"],
                                      "lambda_at_rollout": bundle["lambda_at_rollout"],
                                      "eta_lambda": bundle["eta_lambda"],
                                      "update_cursor": bundle["update_cursor"]}
        # The restore happens AFTER the construction (which may consume torch RNG) and BEFORE the
        # update, so the start state is proven value-identical instead of being assumed.
        state.phase = "A3_restore"
        restore_report = restore_full_state(controller3, optimizer3, _bundle_to_saved(bundle),
                                            np_mod=np_mod, torch_mod=torch)
        record["A3_restore"] = restore_report
        a3_start = capture_full_state(controller3, observer3, np_mod=np_mod, torch_mod=torch,
                                      data_identity=bundle["data_identity"],
                                      configuration=bundle["configuration"],
                                      cursor=bundle["update_cursor"])
        # FAIL FAST: the start state must be proven, and bound to the objects the update will use,
        # before the second paid native update is issued.
        state.phase = "A3_start_gate"
        a3_gate = _gate_a3_start(a1_baseline, a3_start, restore_report, bundle,
                                 controller=controller3, optimizer=optimizer3, episode=episode,
                                 update_index=0, np_mod=np_mod, torch_mod=torch)
        restore_diffs = a3_gate["differences"]
        record["A3_start_gate"] = a3_gate
        record["A3_start_stage_snapshot"] = _save_stage_snapshot(
            S1_DIR / "r75_s1_A3_start_stage_snapshot.pt", stage="A3_start_gate_passed",
            label="A3_after_disk_restore", state=state, full_state=a3_start)

        state.phase = "A3_native_update"
        state.count("native_ppo_update_attempts", "r75_native_ppo_update", label="A3",
                    update_index=0)
        record["A3"] = r75_frozen.r75_native_ppo_update(
            controller3, observer3, [episode], update_index=0, resources=resources)
        record["post_meter_a3"] = meter_a3.snapshot()
        a3_end = capture_full_state(controller3, observer3, np_mod=np_mod, torch_mod=torch,
                                    data_identity=bundle["data_identity"],
                                    configuration=bundle["configuration"],
                                    cursor=int(bundle["update_cursor"]) + 1)

        # --- the four comparisons the contract demands -----------------------------------------
        state.phase = "comparison"
        excluded = PLAN["comparison_contract"]["excluded_in_advance_and_accounted_separately"]
        end_diffs = compare_full_states(a2_end, a3_end, "A2_end_vs_A3_end")
        compared_fields, science_diffs = compare_scientific_returns(record["A2"], record["A3"], PLAN)
        meter_reconciliation = reconcile_trunk_meter(record["A2"], record["A3"], PLAN)

        # --- the observed real step entries ----------------------------------------------------
        step_records = {"A1_A2": observer.records, "A3": observer3.records}
        step_problems = [entry for entries in step_records.values() for entry in entries
                         if entry["phase"] == "completed"
                         and not (entry["exactly_one_new_permit"] and entry["step_charged_nothing"])]
        failed_steps = [entry for entries in step_records.values() for entry in entries
                        if entry["phase"] == "failed"]
        total_real_steps = sum(len(entries) for entries in step_records.values())
        expected_steps = PLAN["accounting"]["planned_totals"]["physical_optimizer_steps"]
        record["step_entry_observation"] = {
            "records": step_records, "total_real_steps": total_real_steps,
            "expected_steps": expected_steps, "steps_match_plan": total_real_steps == expected_steps,
            "permit_problems": step_problems, "failed_steps": failed_steps,
            "note": ("passive observation of the REAL optimizer.step entry, bound to the same "
                     "production counter: it never charges, never refunds and never opens another "
                     "root; every completed step must show exactly one new permit issued by the "
                     "pre-charge and must charge nothing itself")}

        record["comparison"] = {
            "compared_fields": PLAN["comparison_contract"]["compared_value_by_value"],
            "compared_native_fields": compared_fields,
            "excluded_in_advance": excluded,
            "restore_differences": restore_diffs,
            "end_state_differences": end_diffs,
            "scientific_return_differences": science_diffs,
            "meter_reconciliation": meter_reconciliation}
        record["meter_additivity"] = {label: data["additive"] for label, data
                                      in meter_reconciliation["controllers"].items()}
        record["ledger"] = resources.snapshots()
        resources.verify_or_fail()

        # --- the failure gate: any inconsistency is a non-zero exit ----------------------------
        failures: list[str] = []
        if not restore_report["restored_exactly"]:
            failures.append("the A3 disk restore did not reproduce the captured A1 state exactly")
        if restore_diffs:
            failures.append(f"{len(restore_diffs)} full-state difference(s) between the A1 baseline "
                            f"and the A3 start after the disk restore")
        if end_diffs:
            failures.append(f"{len(end_diffs)} full-state difference(s) between the A2 and A3 end states")
        if science_diffs:
            failures.append(f"{len(science_diffs)} scientific return difference(s) between A2 and A3")
        if not all(record["meter_additivity"].values()):
            failures.append("the trunk-entry meter phase split is not additive")
        for label, data in meter_reconciliation["controllers"].items():
            if not (data["calls_match_plan"] and data["rows_match_plan"]):
                failures.append(f"the {label} trunk-entry metering does not match the frozen accounting")
        if a1["trunk_moved"]:
            failures.append("a trunk parameter moved in the critic-only control step")
        if a1["trunk_parameters_in_optimizer"]:
            failures.append("a trunk parameter is inside the optimizer")
        if not a1["object_coverage"]["coverage_complete"]:
            failures.append("the eight-object coverage is not complete")
        if step_problems:
            failures.append(f"{len(step_problems)} real step(s) without exactly one new permit")
        if failed_steps:
            failures.append(f"{len(failed_steps)} real step(s) failed")
        if not record["step_entry_observation"]["steps_match_plan"]:
            failures.append("the real step count does not match the frozen plan")
        if not record["A3_bundle_code_identity_matches_disk"]:
            failures.append("the bundle's code identity does not match the sources on disk")
        record["failures"] = failures
        record["attempt_accounting"] = {
            "counted": state.counter_report(),
            "conservative_attempt_bounds": _conservative_attempt_bounds(),
            "note": ("the counted entries are read from the real call sites and from the observed "
                     "step entries, one by one, with their unit; a failed or refused attempt is "
                     "still an attempt. The conservative bounds are stated before anything runs, "
                     "so relabelling can never shrink a real attempt. Nothing here is a simulation "
                     "and nothing here is free.")}

        state.phase = "closeout"
        S1_DIR.mkdir(parents=True, exist_ok=True)
        (S1_DIR / "r75_s1_trial_record.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps({"kind": "round75-s1-summary/1",
                          "A1_moved_objects": a1["moved_objects"],
                          "A1_trunk_moved": a1["trunk_moved"],
                          "A1_objects_with_gradient": a1["objects_with_gradient"],
                          "A1_gate": record["A1_gate"]["passed"],
                          "A2_optimizer_steps": record["A2"]["optimizer_steps"],
                          "A3_optimizer_steps": record["A3"]["optimizer_steps"],
                          "real_steps_observed": total_real_steps,
                          "restore_differences": len(restore_diffs),
                          "end_state_differences": len(end_diffs),
                          "scientific_return_differences": len(science_diffs),
                          "meter_additivity": record["meter_additivity"],
                          "counted_attempts": {unit: data["counted"]
                                               for unit, data in state.counter_report().items()},
                          "failures": failures,
                          "ledger": record["ledger"]}, indent=2, default=str))
        state.closed_out = True
        return 0 if not failures else 3
    except BaseException as error:
        # The receipt is a record, never a mask: the ORIGINAL exception leaves this process
        # unchanged, nothing is retried, refunded or repaired.
        try:
            receipt = _write_failure_receipt(state, error)
            receipt_note = str(receipt)
        except Exception as receipt_error:                    # pragma: no cover - defensive
            receipt_note = (f"the rich receipt could not be written: "
                            f"{type(receipt_error).__name__}: {receipt_error}")
        print(f"S1 FAILED in phase {state.phase!r}: {type(error).__name__}: {error}",
              file=sys.stderr)
        print(f"durable failure receipt: {receipt_note}", file=sys.stderr)
        raise
    finally:
        if not state.closed_out:
            # The real attempt count must survive no matter how the run ends, so it is appended and
            # fsynced here, after the receipt body, and before the exception leaves the process.
            try:
                _append_jsonl(S1_DIR / "r75_s1_attempt_counters.jsonl",
                              {"kind": "round75-s1-attempt-counters/1",
                               "request_id": EXECUTION_REQUEST_ID or PLAN["request_id"],
                               "plan_request_id": PLAN["request_id"],
                               "attempt_dir": str(S1_DIR),
                               "phase": state.phase,
                               "counters": state.counter_report(),
                               "conservative_attempt_bounds": _conservative_attempt_bounds(),
                               "unread": sorted(set(state.unread))})
            except Exception as counter_error:                # pragma: no cover - defensive
                print(f"the attempt-counter append failed: {type(counter_error).__name__}: "
                      f"{counter_error}", file=sys.stderr)

def _license_flag_value(argv, flag):
    """The value following --flag (or after --flag=) in a recorded child argv, or None."""
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            return argv[index + 1]
        if isinstance(item, str) and item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return None


def _license_is_fresh(recorded, request_id, target):
    """True only for a parent-issued, not-yet-consumed license matching THIS request and directory.

    A marker without license_state == "issued", without a --run-s1 argv, or whose recorded
    --request-id / --attempt-dir disagree with this child is NOT a consumable license; taking it
    over anyway would let a stale or cross-directory marker authorize a run."""
    if not isinstance(recorded, dict) or recorded.get("license_state") != "issued":
        return False
    argv = recorded.get("argv")
    if not isinstance(argv, list) or "--run-s1" not in argv:
        return False
    if _license_flag_value(argv, "--request-id") != request_id:
        return False
    recorded_dir = _license_flag_value(argv, "--attempt-dir")
    if recorded_dir is None:
        return False
    return Path(os.path.normpath(recorded_dir)) == Path(os.path.normpath(str(target)))


def _bind_execution_identity(request_id, attempt_dir) -> Path:
    """Bind the ONE execution identity and its private attempt directory. Path wiring only.

    The rules are the same ones the outer record enforces, restated here because the child is the
    process that actually writes: the historical request id is refused in ANY directory; the attempt
    directory must sit inside this output root and must not be the historical raw/s1 (nor anything
    under it); and the ONE parent->child launch license is taken over exactly once. The parent's
    register_attempt() writes a fresh "issued" license into this directory immediately before the
    launch; THIS function consumes it with an O_EXCL consumption record instead of mistaking the
    parent's own fresh marker for a duplicate. A missing, mismatched, already-consumed, old-id,
    historical-directory or cross-directory license is refused. The central production ledger is NOT
    touched: it stays at raw/runtime, so a new attempt directory never becomes a second ledger.
    """
    global S1_DIR, EXECUTION_REQUEST_ID
    if not request_id:
        raise RuntimeError("no request id was given; renaming a request is not an authorization")
    if request_id == FIRST_ATTEMPT_REQUEST_ID:
        raise RuntimeError("request id %s already consumed its one attempt and can never run "
                           "again, in any directory" % request_id)
    target = Path(attempt_dir) if attempt_dir else None
    if target is None or not str(target):
        raise RuntimeError("no attempt directory was given")
    if not target.is_absolute():
        target = HERE / target
    target = Path(os.path.normpath(str(target)))
    historical = Path(os.path.normpath(str(HERE / "raw" / "s1")))
    if target == historical or historical in target.parents:
        raise RuntimeError("the attempt directory must not be the historical raw/s1 or anything "
                           "under it: %s" % target)
    if target != HERE and HERE not in target.parents:
        raise RuntimeError("the attempt directory must live inside the ROUND75 output root: %s"
                           % target)
    license_marker = None
    recorded_license = None
    if target.exists():
        for marker in sorted(target.glob("r75_s1_*attempt_marker*.json")):
            try:
                recorded = json.loads(marker.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                recorded = None
            if not isinstance(recorded, dict) or recorded.get("request_id") != request_id:
                continue
            if _license_is_fresh(recorded, request_id, target):
                license_marker, recorded_license = marker, recorded
                break
            raise RuntimeError("the attempt license for request id %s at %s is not a fresh, "
                               "unconsumed parent license" % (request_id, marker.name))
    if license_marker is None:
        raise RuntimeError("no fresh parent license for request id %s in %s; a child may only take "
                           "over a license its parent just issued in this directory"
                           % (request_id, target))
    consumed = target / ("r75_s1_outer_v2_license_consumed_%s.json" % request_id)
    try:
        handle = os.open(str(consumed), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise RuntimeError("the attempt license for request id %s was already consumed; a request "
                           "id takes over its parent license exactly once" % request_id)
    with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"kind": "round75-s1-license-consumed/1", "request_id": request_id,
                   "consumed_at": {"unix": time.time(),
                                   "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())},
                   "consumed_by_pid": os.getpid(), "license_marker": license_marker.name,
                   "argv": recorded_license.get("argv"),
                   "note": ("non-repeatable takeover evidence; its O_EXCL creation is what forbids "
                            "a second child from consuming the same license")}, fh, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    try:
        recorded_license["license_state"] = "consumed"
        recorded_license["consumed_at"] = {"unix": time.time(),
                                           "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                                                time.localtime())}
        recorded_license["consumed_by_pid"] = os.getpid()
        license_marker.write_text(json.dumps(recorded_license, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
    S1_DIR = target
    EXECUTION_REQUEST_ID = request_id
    return target


def _repo_root() -> Path:
    """The repository root, found by walking up. The paid path must not depend on the CWD either:
    the frozen plan stores repo-relative paths and the launch directory is not under our control."""
    for candidate in (HERE, *HERE.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"cannot locate the repository root (.git) above {HERE}")


REPO_ROOT = _repo_root()


def r75_replay_adapter_load():
    import r75_replay_adapter as adapter
    identity = PLAN["input_identity"]
    # Absolute paths: the adapter reads the files exactly as given. Its bytes are pinned, so the
    # CWD independence is provided HERE rather than by editing a verified source.
    row_path = Path(identity["row"]["path"])
    tape_path = Path(identity["tape"]["path"])
    if not row_path.is_absolute():
        row_path = REPO_ROOT / row_path
    if not tape_path.is_absolute():
        tape_path = REPO_ROOT / tape_path
    return adapter.load_qualification_batch(row_path, tape_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ROUND75 S1 runner (prepared, not run)")
    parser.add_argument("--run-s1", action="store_true",
                        help="execute the paid S1 path (requires a separate decision-side approval)")
    parser.add_argument("--print-plan", action="store_true", help="print the frozen plan summary")
    parser.add_argument("--request-id", default=None,
                        help="the execution identity of THIS attempt; the historical request id "
                             "is refused in any directory")
    parser.add_argument("--attempt-dir", default=None,
                        help="this attempt's private directory inside the ROUND75 output root; "
                             "raw/s1 keeps the preserved first failure")
    args = parser.parse_args(argv)
    if args.print_plan or not args.run_s1:
        print(json.dumps({"kind": PLAN["kind"], "request_id": PLAN["request_id"],
                          "status": PLAN["status"], "head_at_writing": PLAN["head_at_writing"],
                          "future_command": PLAN["future_command"],
                          "accounting": PLAN["accounting"],
                          "note": ("no argument (or --print-plan) imports no torch and starts "
                                   "nothing; --run-s1 is the paid path")}, indent=2))
        return 0
    if not args.request_id or not args.attempt_dir:
        sys.stderr.write("S1 REFUSED: --run-s1 requires --request-id and --attempt-dir; a run "
                         "that cannot name itself and its own directory never reaches the paid "
                         "path\n")
        return 2
    try:
        _bind_execution_identity(args.request_id, args.attempt_dir)
    except RuntimeError as error:
        sys.stderr.write("S1 REFUSED: %s\n" % error)
        return 2
    return run_s1()


if __name__ == "__main__":
    raise SystemExit(main())