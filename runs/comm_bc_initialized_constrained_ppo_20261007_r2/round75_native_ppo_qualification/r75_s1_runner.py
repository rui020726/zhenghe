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
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE / "ROUND75_S1_RUN_PLAN.json"
PLAN = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
S1_DIR = HERE / "raw" / "s1"

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
    """Read the bundle back from the actual serialized file."""
    torch = sys.modules["torch"]
    return torch.load(path, map_location="cpu")


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
    return {"problems": problems, "restored_exactly": not problems,
            "restored_from_payload": ["five module payloads", "optimizer payload (all Adam values "
                                      "and param-group hyperparameters)",
                                      "numpy / torch / python RNG states",
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
        exactly ONE new permit must appear there - never zero (an unlicensed step) and never two
        (a permit spent twice). The step itself must then charge nothing, which is what proves
        the observer neither double-charges nor frees the step.
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
        try:
            result = self.__dict__["_inner_step"](*args, **kwargs)
        except BaseException as error:                       # a failed step is recorded, never refunded
            self.__dict__["_records"].append({**attempt, "phase": "failed",
                                              "error": f"{type(error).__name__}: {error}",
                                              "durable_used_after": self._durable_used()})
            raise
        after = self._durable_used()
        self.__dict__["_records"].append({**attempt, "phase": "completed",
                                          "durable_used_after": after,
                                          "permit_delta": after - before,
                                          "step_charged_nothing": after == before})
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


# ------------------------------------------------------------------------------- run -----
def run_s1() -> int:
    # The pure-file precheck runs BEFORE any paid module is touched. It is stdlib only, so it can
    # refuse the run before torch, numpy, the model, the optimizer or the ledger is ever opened.
    sys.path.insert(0, str(HERE))
    import r75_s1_precheck

    precheck_code = r75_s1_precheck.main()
    if precheck_code != 0:
        print("S1 REFUSED: the freeze precheck failed; nothing paid was started", file=sys.stderr)
        return 2

    np_mod, torch, r75_frozen, r75_guard = _load_paid_modules()

    configuration = _live_configuration(r75_frozen)
    if not configuration["lr_matches_live_learning_rate"]:
        print("S1 REFUSED: the runner LR is not the live LEARNING_RATE", file=sys.stderr)
        return 2
    identity = PLAN["input_identity"]
    data_identity = {"row_sha256": identity["row"]["sha256"],
                     "tape_sha256": identity["tape"]["sha256"],
                     "model_sha256": identity["r58_ep0_model"]["sha256"],
                     "behaviour_parameter_sha256": identity["behaviour_parameters"]["value"]}

    resources = r75_guard.R75Resources.production()
    resources.verify_or_fail()
    print("canonical production ledger opened and reconciled at the zero starting point")

    # --- gate-only controller: R58 ep0 behaviour, two equal heads, trunk frozen ------------
    controller, ep0_state, frozen_identity, summary = r75_frozen.build_gate_only_controller(SEED)
    # Install the meter BEFORE any possible forward (A2 controller).
    meter_a2 = resources.ensure_trunk_meter(controller, what_prefix="r75_s1_A2")
    pre_a2 = meter_a2.snapshot()

    episode, facts = r75_replay_adapter_load()
    optimizer = r75_frozen.build_optimizer(controller, LR)
    # A1 and A2 share one controller, one optimizer and therefore one observed step entry.
    observer = ObservedStepEntry(optimizer, resources, "r75_s1_A1_A2")

    record: dict[str, Any] = {"kind": "round75-s1-trial/1", "request_id": PLAN["request_id"],
                              "head_at_writing": PLAN["head_at_writing"], "seed": SEED,
                              "data_identity": data_identity, "batch_facts": facts,
                              "frozen_identity": frozen_identity,
                              "parameter_summary": summary,
                              "pre_meter_a2": pre_a2,
                              "configuration": configuration,
                              "code_identity_at_start": _code_identity()}

    # --- A1: the critic-only control step --------------------------------------------------
    a1 = run_a1(controller, observer, resources, torch=torch, np_mod=np_mod, record=record,
                configuration=configuration)

    # --- the frozen A1 baseline: the ONE state that A2 and A3 both start from ---------------
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
    record["A2"] = r75_frozen.r75_native_ppo_update(
        controller, observer, [episode], update_index=0, resources=resources)
    record["post_meter_a2"] = meter_a2.snapshot()
    a2_end = capture_full_state(controller, observer, np_mod=np_mod, torch_mod=torch,
                                data_identity=data_identity, configuration=configuration, cursor=1)

    # --- A3: restore from the ACTUAL SERIALIZED FILE into fresh objects --------------------
    controller3, _state3, _frozen3, _summary3 = r75_frozen.build_gate_only_controller(SEED)
    meter_a3 = resources.ensure_trunk_meter(controller3, what_prefix="r75_s1_A3")
    optimizer3 = r75_frozen.build_optimizer(controller3, LR)
    observer3 = ObservedStepEntry(optimizer3, resources, "r75_s1_A3")
    # The disk file is the resume source. The in-memory capture is NEVER used as a substitute.
    bundle = load_state_bundle(bundle_path)
    record["A3_bundle_code_identity_matches_disk"] = bundle["code_identity"] == _code_identity()
    record["A3_bundle_lambda"] = {"lambda": bundle["lambda"],
                                  "lambda_at_rollout": bundle["lambda_at_rollout"],
                                  "eta_lambda": bundle["eta_lambda"],
                                  "update_cursor": bundle["update_cursor"]}
    # The restore happens AFTER the construction (which may consume torch RNG) and BEFORE the
    # update, so the start state is proven value-identical instead of being assumed.
    restore_report = restore_full_state(controller3, optimizer3, _bundle_to_saved(bundle),
                                        np_mod=np_mod, torch_mod=torch)
    record["A3_restore"] = restore_report
    a3_start = capture_full_state(controller3, observer3, np_mod=np_mod, torch_mod=torch,
                                  data_identity=bundle["data_identity"],
                                  configuration=bundle["configuration"],
                                  cursor=bundle["update_cursor"])
    record["A3"] = r75_frozen.r75_native_ppo_update(
        controller3, observer3, [episode], update_index=0, resources=resources)
    record["post_meter_a3"] = meter_a3.snapshot()
    a3_end = capture_full_state(controller3, observer3, np_mod=np_mod, torch_mod=torch,
                                data_identity=bundle["data_identity"],
                                configuration=bundle["configuration"],
                                cursor=int(bundle["update_cursor"]) + 1)

    # --- the four comparisons the contract demands -----------------------------------------
    excluded = PLAN["comparison_contract"]["excluded_in_advance_and_accounted_separately"]
    restore_diffs = compare_full_states(a1_baseline, a3_start,
                                        "A1_baseline_vs_A3_start_after_disk_restore")
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

    S1_DIR.mkdir(parents=True, exist_ok=True)
    (S1_DIR / "r75_s1_trial_record.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"kind": "round75-s1-summary/1",
                      "A1_moved_objects": a1["moved_objects"],
                      "A1_trunk_moved": a1["trunk_moved"],
                      "A1_objects_with_gradient": a1["objects_with_gradient"],
                      "A2_optimizer_steps": record["A2"]["optimizer_steps"],
                      "A3_optimizer_steps": record["A3"]["optimizer_steps"],
                      "real_steps_observed": total_real_steps,
                      "restore_differences": len(restore_diffs),
                      "end_state_differences": len(end_diffs),
                      "scientific_return_differences": len(science_diffs),
                      "meter_additivity": record["meter_additivity"],
                      "failures": failures,
                      "ledger": record["ledger"]}, indent=2, default=str))
    return 0 if not failures else 3

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


def main() -> int:
    parser = argparse.ArgumentParser(description="ROUND75 S1 runner (prepared, not run)")
    parser.add_argument("--run-s1", action="store_true",
                        help="execute the paid S1 path (requires a separate decision-side approval)")
    parser.add_argument("--print-plan", action="store_true", help="print the frozen plan summary")
    args = parser.parse_args()
    if args.print_plan or not args.run_s1:
        print(json.dumps({"kind": PLAN["kind"], "request_id": PLAN["request_id"],
                          "status": PLAN["status"], "head_at_writing": PLAN["head_at_writing"],
                          "future_command": PLAN["future_command"],
                          "accounting": PLAN["accounting"],
                          "note": ("no argument (or --print-plan) imports no torch and starts "
                                   "nothing; --run-s1 is the paid path")}, indent=2))
        return 0
    return run_s1()


if __name__ == "__main__":
    raise SystemExit(main())