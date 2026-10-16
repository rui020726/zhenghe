"""R76 model-only INIT adaptation, runtime binding, and full-state resume contract."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import r76_contracts as contracts


class R76IdentityError(RuntimeError):
    """A source payload, mask, or checkpoint failed its role-bound identity check."""


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


R71_REQUIRED_PAYLOAD_FIELDS = (
    "schema", "trainer_tag", "training_version", "loss_version", "optimizer_schema",
    "structure", "controller", "optimizer", "frozen_mask", "training_state", "rng",
)
R71_PAYLOAD_FROZEN_MASK_FIELDS = (
    "frozen_modules", "trainable_modules", "frozen_sha256", "frozen_scalar_count",
    "frozen_parameter_object_count", "critic_heads_calibrated", "note",
)


def _validate_r71_init_source_file(source_file, *, seed: int) -> str:
    if source_file is None:
        raise R76IdentityError("the R71 INIT source file path is required; bytes are never guessed")
    try:
        filename, expected_bytes, expected_sha256 = contracts.R71_INIT_FILES[int(seed)]
    except KeyError as error:
        raise R76IdentityError(f"R76 has no registered R71 INIT file for seed {seed}") from error
    path = Path(source_file)
    if path.name != filename:
        raise R76IdentityError(
            f"seed {seed} must use the registered INIT filename {filename}, not {path.name}")
    if not path.is_file():
        raise R76IdentityError(f"registered R71 INIT file is missing: {path}")
    if path.stat().st_size != expected_bytes:
        raise R76IdentityError(
            f"registered R71 INIT byte count drifted: {path.stat().st_size} != {expected_bytes}")
    got = sha256_file(path)
    if got != expected_sha256:
        raise R76IdentityError(f"registered R71 INIT file SHA256 {got} != {expected_sha256}")
    return got


def validate_r71_init_source_file(source_file, *, seed: int) -> str:
    """Public byte-identity gate used before any checkpoint deserialisation."""
    return _validate_r71_init_source_file(source_file, seed=seed)


def validate_r71_init_identity(registered_identity: dict[str, Any], *, seed: int) -> None:
    if not isinstance(registered_identity, dict):
        raise R76IdentityError("the registered R71 INIT identity must be a JSON object")
    source = registered_identity.get("source")
    calibration = registered_identity.get("calibration")
    mask = registered_identity.get("frozen_mask")
    if not isinstance(source, dict) or not isinstance(calibration, dict) \
            or not isinstance(mask, dict):
        raise R76IdentityError("R71 INIT identity source/calibration/frozen_mask are required")
    if registered_identity.get("seed") != int(seed):
        raise R76IdentityError("the registered R71 INIT identity belongs to another seed")
    if registered_identity.get("kind") != "round71-init-identity/1":
        raise R76IdentityError("the registered INIT identity is not a round71-init-identity/1")
    if source.get("pre_scale_equals_r60_ep0") is not True:
        raise R76IdentityError("the INIT pre-scale parameters are not the R60 ep0 identity")
    if float(source.get("late_init_alpha")) != 0.10:
        raise R76IdentityError("the INIT late-head x0.10 identity is missing or drifted")
    if registered_identity.get("optimizer_fresh") is not True:
        raise R76IdentityError("the R71 INIT optimizer is not registered as fresh")
    if float(registered_identity.get("lambda_init")) != 1.0:
        raise R76IdentityError("the R71 INIT lambda identity is not fresh lambda=1.0")
    if registered_identity.get("rng_unchanged_by_calibration") is not True:
        raise R76IdentityError("the R71 calibration did not preserve its initial RNG identity")
    for module, key in contracts.R71_CALIBRATION_SHA_KEYS.items():
        if key not in calibration:
            raise R76IdentityError(f"R71 INIT calibration has no {key}")
    if mask != contracts.R71_REGISTERED_INIT_MASK:
        raise R76IdentityError(f"R71 INIT frozen mask drifted: {mask!r}")


def validate_r71_init_payload(payload: Any, registered_identity: dict[str, Any], *,
                              seed: int, source_file=None) -> tuple[dict[str, Any], str]:
    contracts.validate_exact_keys(payload, R71_REQUIRED_PAYLOAD_FIELDS, "R71 INIT payload")
    source_sha256 = _validate_r71_init_source_file(source_file, seed=seed)
    if payload.get("schema") != contracts.R71_STATE_SCHEMA:
        raise R76IdentityError(f"INIT schema {payload.get('schema')!r} is not R71")
    if payload.get("trainer_tag") != contracts.R71_TRAINER_TAG:
        raise R76IdentityError("INIT trainer tag is not the calibrated R71 tag")
    if payload.get("training_version") != contracts.R71_TRAINING_VERSION:
        raise R76IdentityError("INIT training version is not the exact R71 version")
    if payload.get("loss_version") != contracts.R71_LOSS_VERSION:
        raise R76IdentityError("INIT loss version is not the exact R71 loss version")
    if payload.get("optimizer_schema") != contracts.R71_OPTIMIZER_SCHEMA:
        raise R76IdentityError("INIT optimizer schema is not the exact R71 six-object schema")
    if payload.get("structure") != contracts.R71_STRUCTURE:
        raise R76IdentityError("INIT structure is not the exact R71 calibrated structure")
    if set(payload["controller"]) != set(contracts.R76_MODULES):
        raise R76IdentityError("INIT controller is not the five-module payload set")
    payload_mask = payload.get("frozen_mask")
    contracts.validate_exact_keys(payload_mask, R71_PAYLOAD_FROZEN_MASK_FIELDS,
                                  "R71 INIT payload frozen mask")
    if payload_mask.get("frozen_modules") != ["trunk", "early_head"] \
            or payload_mask.get("trainable_modules") != ["late_head", "value_task_head",
                                                         "value_cost_head"] \
            or payload_mask.get("frozen_parameter_object_count") != 6 \
            or payload_mask.get("frozen_scalar_count") != 5570 \
            or payload_mask.get("critic_heads_calibrated") is not True \
            or not isinstance(payload_mask.get("frozen_sha256"), str) \
            or len(payload_mask["frozen_sha256"]) != 64:
        raise R76IdentityError("payload frozen mask is not a valid R71 model-only mask")
    state = payload.get("training_state")
    contracts.validate_exact_keys(state, ("training_seed", "episodes_done", "updates",
                                          "optimizer_steps", "lambda"), "R71 INIT state")
    if int(state["training_seed"]) != int(seed) \
            or int(state["episodes_done"]) != 0 or int(state["updates"]) != 0 \
            or int(state["optimizer_steps"]) != 0 or float(state["lambda"]) != 1.0:
        raise R76IdentityError("the INIT state is not the untrained lambda=1 cursor 0/0")
    if not isinstance(payload.get("optimizer"), dict):
        raise R76IdentityError("the R71 INIT payload has no optimizer object; use read-only model "
                               "load, not absence of optimizer, as the model-only proof")
    validate_r71_init_identity(registered_identity, seed=seed)
    return payload, source_sha256


def build_r76_controller_from_r71_init(
        payload: Any, registered_identity: dict[str, Any], *, seed: int,
        controller_factory: Callable[[int], Any],
        module_sha: Callable[[Any, str], str],
        freeze_trunk_only: Callable[[Any], dict[str, Any]],
        source_file=None) -> tuple[Any, dict[str, Any]]:
    """Verify first, load values second, hash values third, then unfreeze early only."""
    _payload, source_sha256 = validate_r71_init_payload(
        payload, registered_identity, seed=seed, source_file=source_file)
    controller = controller_factory(int(seed))
    controller.load_state_dict(payload["controller"])
    module_hashes = {name: module_sha(controller, name)
                     for name in contracts.R76_MODULES}
    expected_hashes = {
        name: registered_identity["calibration"][contracts.R71_CALIBRATION_SHA_KEYS[name]]
        for name in contracts.R76_MODULES
    }
    if module_hashes != expected_hashes:
        raise R76IdentityError(f"R71 INIT module values drifted: {module_hashes!r}")
    mask = freeze_trunk_only(controller)
    contracts.validate_r76_mask_report(mask)
    after_hashes = {name: module_sha(controller, name) for name in contracts.R76_MODULES}
    if after_hashes != module_hashes:
        raise R76IdentityError("unfreezing early_head changed module values")
    report = {
        "source_file": None if source_file is None else str(Path(source_file).resolve()),
        "source_sha256": source_sha256,
        "registered_identity_sha256": contracts.sha256_text(registered_identity),
        "module_sha256": module_hashes,
        "mask": mask,
        "late_init_alpha": 0.10,
        "rescaled": False,
        "teacher_refit": False,
        "optimizer_restored": False,
        "lambda_restored": False,
        "rng_restored": False,
    }
    return controller, report


def _module_value(holder: Any, name: str):
    parts = name.split(".")
    value = holder
    for part in parts:
        value = getattr(value, part)
    return value


def parameter_object_mapping(controller: Any, optimizer: Any) -> list[str]:
    """Return the optimizer's parameter objects under stable names in optimizer order."""
    names_by_object: dict[int, str] = {}
    module_objects = {
        "trunk": controller.inner.trunk,
        "early_head": controller.inner.early_head,
        "late_head": controller.inner.late_head,
        "value_task_head": controller.inner.value_head,
        "value_cost_head": controller.cost_head,
    }
    frozen_parameter_ids = {
        id(parameter)
        for parameter in module_objects["trunk"].parameters()
    }
    for module_name in contracts.R76_TRAINABLE_MODULES:
        holder = module_objects[module_name]
        for parameter_name, parameter in holder.named_parameters():
            stable_name = f"{module_name}.{parameter_name}"
            parameter_id = id(parameter)
            if parameter_id in names_by_object:
                raise R76IdentityError(
                    f"R76 parameter object appears twice: {stable_name} and "
                    f"{names_by_object[parameter_id]}")
            names_by_object[parameter_id] = stable_name
    if len(names_by_object) != contracts.R76_TRAINABLE_OBJECT_COUNT:
        raise R76IdentityError(
            f"R76 trainable modules expose {len(names_by_object)} parameter objects, "
            f"not {contracts.R76_TRAINABLE_OBJECT_COUNT}")
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or [len(group.get("params", []))
                                        for group in groups] != [contracts.R76_TRAINABLE_OBJECT_COUNT]:
        raise R76IdentityError("R76 optimizer must have one parameter group with eight objects")
    optimizer_parameters = [parameter for group in groups
                            for parameter in group.get("params", [])]
    mapping = []
    for parameter in optimizer_parameters:
        stable_name = names_by_object.get(id(parameter))
        if stable_name is None:
            if id(parameter) in frozen_parameter_ids:
                raise R76IdentityError("R76 optimizer includes a frozen trunk parameter")
            raise R76IdentityError("R76 optimizer contains a parameter outside the trainable set")
        mapping.append(stable_name)
    if len(mapping) != len(set(mapping)):
        raise R76IdentityError("R76 optimizer maps one stable parameter name more than once")
    if any(name.startswith("trunk.") for name in mapping):
        raise R76IdentityError("R76 optimizer includes a frozen trunk parameter")
    return mapping


def validate_base_config(cfg: Any) -> dict[str, float]:
    """Read the configured living values; import defaults are never accepted."""
    if not isinstance(cfg, dict):
        raise R76IdentityError("R76 runtime cfg must be a dict")
    for key in ("trainer", "common", "runner", "runtime", "controller_module"):
        if key not in cfg:
            raise R76IdentityError(f"R76 runtime cfg is missing {key}")
    common = cfg["common"]
    trainer = cfg["trainer"]
    runner = cfg["runner"]
    for attr in ("batch_samples", "_unassigned_prefix_totals"):
        if not callable(getattr(trainer, attr, None)):
            raise R76IdentityError(f"the live trainer lacks {attr}")
    for attr in ("save_checkpoint", "load_checkpoint", "rng_state", "restore_rng"):
        if not callable(getattr(trainer, attr, None)):
            raise R76IdentityError(f"the live trainer lacks {attr}")
    if not callable(getattr(runner, "_run_episode", None)):
        raise R76IdentityError("the live runner lacks _run_episode")
    captured = {}
    for name, want in contracts.R76_LIVE_CONFIG.items():
        holder, attr = name.split(".", 1)
        if holder == "common":
            module = common
        else:
            module = sys.modules.get({"r66": "r66_frozen", "r68": "r68_frozen"}.get(holder, holder))
        if module is None:
            raise R76IdentityError(f"live {holder} module is not configured")
        got = float(_module_value(module, attr))
        if got != float(want):
            raise R76IdentityError(f"live config {name}={got!r} != {want!r}")
        captured[name] = got
    return captured


def require_resource_interface(resources: Any) -> None:
    if resources is None:
        raise R76IdentityError("R76 real binding requires an explicit resource object")
    for name in ("charge_joint_record", "ensure_trunk_meter", "charge_physical_step",
                 "charge_offline_rows", "verify_or_fail"):
        if not callable(getattr(resources, name, None)):
            raise R76IdentityError(f"R76 resources lack {name}")
    if getattr(resources, "mode", None) != "production":
        raise R76IdentityError("R76 real binding accepts only production resources")


def install_r76_runtime_bindings(cfg: dict[str, Any], resources: Any,
                                 native_module: Any, *, request_id: str) -> dict[str, Any]:
    """Bind native PPO, the eight-object optimizer, and save/load onto the live cfg."""
    captured = validate_base_config(cfg)
    require_resource_interface(resources)
    if request_id != contracts.REQUEST_ID:
        raise R76IdentityError("R76 runtime request id does not match the prepared request")

    def r76_native_ppo(controller, optimizer, episodes, *, update_index):
        return native_module.r75_native_ppo_update(
            controller, optimizer, episodes, update_index=update_index,
            resources=resources, physical_step_category="normal")

    def r76_save_checkpoint(controller, optimizer, state, path, *, freeze=None):
        mask = native_module.assert_mask(controller)
        return save_r76_checkpoint(
            path, controller, optimizer, state, cfg["trainer"].rng_state(),
            r76_source_identity(cfg, request_id=request_id),
            native_module.torch, mask=mask, module_sha=native_module.module_sha)

    def r76_load_checkpoint(path, *, freeze=None):
        for key in ("r76_controller_factory", "r76_module_sha"):
            if not callable(cfg.get(key)):
                raise R76IdentityError(f"R76 live cfg lacks a callable {key}")
        return load_r76_checkpoint(
            path, native_module.torch,
            controller_factory=cfg["r76_controller_factory"],
            module_sha=cfg["r76_module_sha"],
            freeze_trunk_only=native_module.freeze_trunk_only,
            build_optimizer=native_module.build_optimizer,
            restore_rng=cfg["trainer"].restore_rng,
            rng_state=cfg["trainer"].rng_state,
            expected_context=cfg.get("r76_resume_context"))

    old_ppo = cfg["trainer"].ppo_update
    cfg["trainer"].ppo_update = r76_native_ppo
    old_save = cfg["trainer"].save_checkpoint
    old_load = cfg["trainer"].load_checkpoint
    cfg["trainer"].save_checkpoint = r76_save_checkpoint
    cfg["trainer"].load_checkpoint = r76_load_checkpoint
    cfg["r76_module_sha"] = native_module.module_sha
    identity = {
        "request_id": request_id,
        "task_id": contracts.TASK_ID,
        "training_version": contracts.R76_TRAINING_VERSION,
        "loss_version": contracts.R76_LOSS_VERSION,
        "state_schema": contracts.R76_STATE_SCHEMA,
        "optimizer_schema": contracts.R76_OPTIMIZER_SCHEMA,
        "structure": contracts.R76_STRUCTURE,
        "frozen_modules": list(contracts.R76_FROZEN_MODULES),
        "trainable_modules": list(contracts.R76_TRAINABLE_MODULES),
        "live_config": captured,
        "ppo_replaced": cfg["trainer"].ppo_update is not old_ppo,
        "save_replaced": cfg["trainer"].save_checkpoint is not old_save,
        "load_replaced": cfg["trainer"].load_checkpoint is not old_load,
        "save_is_r76": cfg["trainer"].save_checkpoint is r76_save_checkpoint,
        "load_is_r76": cfg["trainer"].load_checkpoint is r76_load_checkpoint,
        "module_sha_is_r76": cfg["r76_module_sha"] is native_module.module_sha,
        "native_entry": f"{native_module.__name__}.r75_native_ppo_update",
        "resources_canonical_root": str(getattr(resources, "root", "")),
        "resources_mode": getattr(resources, "mode", None),
    }
    cfg["r76"] = {"identity": identity, "resources": resources}
    return identity


R76_REQUIRED_STATE_FIELDS = (
    "training_seed", "episodes_done", "updates", "optimizer_steps", "lambda",
    "lambda_at_rollout", "lambda_history", "data_order_seed", "b0_bytes", "gamma",
    "gae_lambda", "eta_lambda", "run_identity", "contract_version",
    "training_version", "loss_version", "state_schema", "optimizer_schema", "structure",
)
R76_REQUIRED_FULL_FIELDS = (
    "schema", "trainer_tag", "training_version", "loss_version", "optimizer_schema",
    "structure", "contract", "controller", "optimizer", "optimizer_param_groups",
    "frozen_mask", "training_state", "rng", "parameter_object_names", "source_identity",
)
R76_REQUIRED_RNG_FIELDS = ("python", "numpy", "torch")
R76_REQUIRED_SOURCE_IDENTITY_FIELDS = (
    "request_id", "task_id", "source_manifest", "module_sha256", "data_identity",
    "reference_identity", "config_identity", "resource_contract_sha256",
)
R76_REQUIRED_DATA_IDENTITY_FIELDS = (
    "data_order_seed", "b0_bytes", "episodes_per_seed", "scenario_order_sha256", "seeds",
)
R76_REQUIRED_REFERENCE_IDENTITY_FIELDS = ("kind", "reference_row_sha256")


def build_r76_state(seed: int, *, data_order_seed: int, b0_bytes: float) -> dict[str, Any]:
    return {
        "training_seed": int(seed),
        "episodes_done": 0,
        "updates": 0,
        "optimizer_steps": 0,
        "lambda": 1.0,
        "lambda_at_rollout": 1.0,
        "lambda_history": [],
        "data_order_seed": int(data_order_seed),
        "b0_bytes": float(b0_bytes),
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "eta_lambda": 0.5,
        "run_identity": {"request_id": contracts.REQUEST_ID, "task_id": contracts.TASK_ID},
        "contract_version": contracts.R76_FULL_CONTRACT["kind"],
        "training_version": contracts.R76_TRAINING_VERSION,
        "loss_version": contracts.R76_LOSS_VERSION,
        "state_schema": contracts.R76_STATE_SCHEMA,
        "optimizer_schema": contracts.R76_OPTIMIZER_SCHEMA,
        "structure": contracts.R76_STRUCTURE,
    }


def r76_source_identity(cfg: dict[str, Any], *, request_id: str) -> dict[str, Any]:
    """Build the exact non-module provenance carried by every R76 full checkpoint."""
    for key in ("r76_source_manifest", "r76_data_identity", "r76_reference_identity",
                "r76_runtime_identity", "r76_resource_contract_sha256"):
        if key not in cfg:
            raise R76IdentityError(f"R76 source identity is missing live cfg {key}")
    return {
        "request_id": request_id,
        "task_id": contracts.TASK_ID,
        "source_manifest": dict(cfg["r76_source_manifest"]),
        "data_identity": dict(cfg["r76_data_identity"]),
        "reference_identity": dict(cfg["r76_reference_identity"]),
        "config_identity": dict(cfg["r76_runtime_identity"]["live_config"]),
        "resource_contract_sha256": str(cfg["r76_resource_contract_sha256"]),
    }


def validate_r76_resume_context(payload: Any, expected: dict[str, Any] | None) -> None:
    """Bind a loaded checkpoint to this seed, data, reference, config, and contract."""
    if expected is None:
        return
    source_identity = payload["source_identity"]
    state = payload["training_state"]
    if int(expected.get("seed", -1)) != int(state["training_seed"]):
        raise R76IdentityError("R76 resume context seed does not match the checkpoint state")
    for field in ("data_identity", "reference_identity", "config_identity"):
        if expected.get(field) != source_identity.get(field):
            raise R76IdentityError(f"R76 resume context {field} does not match the checkpoint")
    expected_contract = str(expected.get("resource_contract_sha256", ""))
    if expected_contract != str(source_identity.get("resource_contract_sha256", "")):
        raise R76IdentityError("R76 resume context resource contract does not match")


def validate_r76_training_state(state: Any) -> dict[str, Any]:
    contracts.validate_exact_keys(state, R76_REQUIRED_STATE_FIELDS, "R76 training state")
    if int(state["episodes_done"]) < 0 or int(state["updates"]) < 0 \
            or int(state["optimizer_steps"]) < 0:
        raise R76IdentityError("R76 cursor fields must be non-negative")
    if int(state["episodes_done"]) != int(state["updates"]) * 4:
        raise R76IdentityError("R76 episode cursor is not four episodes per update")
    if not isinstance(state["run_identity"], dict) or \
            state["run_identity"].get("request_id") != contracts.REQUEST_ID or \
            state["run_identity"].get("task_id") != contracts.TASK_ID:
        raise R76IdentityError("R76 run identity is not the exact request/task identity")
    if state["contract_version"] != contracts.R76_FULL_CONTRACT["kind"]:
        raise R76IdentityError("R76 contract version is not the exact full-resume contract")
    for name, want in (("data_order_seed", 20260927), ("b0_bytes", 10000.0),
                       ("gamma", 1.0), ("gae_lambda", 1.0), ("eta_lambda", 0.5)):
        got = state[name]
        if float(got) != float(want):
            raise R76IdentityError(f"R76 state {name}={got!r} != {want!r}")
    for field in ("lambda", "lambda_at_rollout"):
        value = float(state[field])
        if not math.isfinite(value) or value < 0.0:
            raise R76IdentityError(f"R76 dual {field} is not finite and non-negative")
    if not isinstance(state["lambda_history"], list) or \
            len(state["lambda_history"]) != int(state["updates"]):
        raise R76IdentityError("R76 lambda history does not match the update cursor")
    for name, want in (("training_version", contracts.R76_TRAINING_VERSION),
                       ("loss_version", contracts.R76_LOSS_VERSION),
                       ("state_schema", contracts.R76_STATE_SCHEMA),
                       ("optimizer_schema", contracts.R76_OPTIMIZER_SCHEMA),
                       ("structure", contracts.R76_STRUCTURE)):
        if state[name] != want:
            raise R76IdentityError(f"R76 state {name}={state[name]!r} != {want!r}")
    return state


def validate_r76_parameter_names(names: Any) -> list[str]:
    if not isinstance(names, list) or len(names) != contracts.R76_TRAINABLE_OBJECT_COUNT:
        raise R76IdentityError("R76 checkpoint parameter mapping is not eight named objects")
    if len(names) != len(set(names)) or any(not isinstance(name, str) for name in names):
        raise R76IdentityError("R76 checkpoint parameter mapping is not unique and string-valued")
    expected_prefixes = tuple(f"{module}." for module in contracts.R76_TRAINABLE_MODULES)
    if any(not name.startswith(expected_prefixes) for name in names):
        raise R76IdentityError("R76 checkpoint parameter mapping includes a foreign object")
    if any(name.startswith("trunk.") for name in names):
        raise R76IdentityError("R76 checkpoint parameter mapping includes trunk")
    return names


def _optimizer_value_equal(left: Any, right: Any) -> bool:
    """Compare Adam payloads value-by-value, including tensor shape/dtype/content."""
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _optimizer_value_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _optimizer_value_equal(left[key], right[key]) for key in left)
    if hasattr(left, "shape") and hasattr(left, "dtype"):
        return bool(hasattr(right, "shape") and hasattr(right, "dtype")
                    and left.shape == right.shape and left.dtype == right.dtype
                    and (left == right).all())
    return left == right


def validate_r76_full_payload(payload: Any) -> dict[str, Any]:
    contracts.validate_exact_keys(payload, R76_REQUIRED_FULL_FIELDS, "R76 full payload")
    if payload.get("schema") in contracts.FOREIGN_STATE_SCHEMAS:
        raise R76IdentityError("a foreign checkpoint was offered as an R76 resume source")
    if payload.get("schema") != contracts.R76_STATE_SCHEMA:
        raise R76IdentityError("payload schema is not the exact R76 full-resume schema")
    if payload.get("trainer_tag") != contracts.R76_TRAINER_TAG:
        raise R76IdentityError("payload trainer tag is not R76")
    if payload.get("training_version") != contracts.R76_TRAINING_VERSION:
        raise R76IdentityError("payload training version is not R76")
    if payload.get("loss_version") != contracts.R76_LOSS_VERSION:
        raise R76IdentityError("payload loss version is not R76")
    if payload.get("optimizer_schema") != contracts.R76_OPTIMIZER_SCHEMA:
        raise R76IdentityError("payload optimizer schema is not the R76 eight-object schema")
    if payload.get("structure") != contracts.R76_STRUCTURE:
        raise R76IdentityError("payload structure is not R76")
    if payload.get("contract") != contracts.R76_FULL_CONTRACT:
        raise R76IdentityError("payload contract is not the exact R76 contract")
    if set(payload.get("controller", {})) != set(contracts.R76_MODULES):
        raise R76IdentityError("payload controller is not the five-module set")
    if payload.get("optimizer_param_groups") != [8]:
        raise R76IdentityError("payload does not declare exactly one 8-parameter optimizer group")
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, dict) or \
            [len(group.get("params", [])) for group in optimizer.get("param_groups", [])] != [8]:
        raise R76IdentityError("payload optimizer is not one 8-parameter optimizer group")
    contracts.validate_r76_mask_report(payload.get("frozen_mask", {}))
    validate_r76_training_state(payload.get("training_state", {}))
    contracts.validate_exact_keys(payload.get("rng", {}), R76_REQUIRED_RNG_FIELDS, "R76 RNG")
    validate_r76_parameter_names(payload.get("parameter_object_names", []))
    source_identity = payload.get("source_identity", {})
    contracts.validate_exact_keys(source_identity, R76_REQUIRED_SOURCE_IDENTITY_FIELDS,
                                  "R76 checkpoint source identity")
    if source_identity.get("request_id") != contracts.REQUEST_ID \
            or source_identity.get("task_id") != contracts.TASK_ID \
            or not isinstance(source_identity.get("module_sha256"), dict) \
            or set(source_identity["module_sha256"]) != set(contracts.R76_MODULES):
        raise R76IdentityError("R76 checkpoint source identity is incomplete")
    try:
        contracts.validate_source_manifest(source_identity.get("source_manifest", {}))
    except RuntimeError as error:
        raise R76IdentityError(f"R76 checkpoint source manifest is not pinned: {error}") from error
    data_identity = source_identity.get("data_identity", {})
    contracts.validate_exact_keys(data_identity, R76_REQUIRED_DATA_IDENTITY_FIELDS,
                                  "R76 checkpoint data identity")
    if int(data_identity.get("episodes_per_seed", -1)) != 128 \
            or data_identity.get("seeds") != [11, 22, 33]:
        raise R76IdentityError("R76 checkpoint data identity is not the fixed seed matrix")
    reference_identity = source_identity.get("reference_identity", {})
    contracts.validate_exact_keys(reference_identity,
                                  R76_REQUIRED_REFERENCE_IDENTITY_FIELDS,
                                  "R76 checkpoint reference identity")
    if reference_identity.get("kind") != "round68-reused-r66-immediate-references/1":
        raise R76IdentityError("R76 checkpoint reference identity is not the reused R66 set")
    if not isinstance(source_identity.get("config_identity"), dict) \
            or source_identity.get("config_identity") != contracts.R76_LIVE_CONFIG \
            or not isinstance(source_identity.get("resource_contract_sha256"), str) \
            or len(source_identity["resource_contract_sha256"]) != 64:
        raise R76IdentityError("R76 checkpoint reference/config/contract identity is incomplete")
    return payload


def build_r76_checkpoint_payload(controller: Any, optimizer: Any, state: dict[str, Any],
                                 rng_state: dict[str, Any], source_identity: dict[str, Any],
                                 *, mask: dict[str, Any],
                                 module_sha: Callable[[Any, str], str]) -> dict[str, Any]:
    validate_r76_training_state(state)
    parameter_object_names = parameter_object_mapping(controller, optimizer)
    source_identity = dict(source_identity)
    source_identity["module_sha256"] = {name: module_sha(controller, name)
                                        for name in contracts.R76_MODULES}
    payload = {
        "schema": contracts.R76_STATE_SCHEMA,
        "trainer_tag": contracts.R76_TRAINER_TAG,
        "training_version": contracts.R76_TRAINING_VERSION,
        "loss_version": contracts.R76_LOSS_VERSION,
        "optimizer_schema": contracts.R76_OPTIMIZER_SCHEMA,
        "structure": contracts.R76_STRUCTURE,
        "contract": contracts.R76_FULL_CONTRACT,
        "controller": {key: value for key, value in controller.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "optimizer_param_groups": [len(group["params"])
                                   for group in optimizer.state_dict()["param_groups"]],
        "frozen_mask": mask,
        "parameter_object_names": parameter_object_names,
        "training_state": state,
        "rng": rng_state,
        "source_identity": source_identity,
    }
    return validate_r76_full_payload(payload)


def save_r76_checkpoint(path, controller: Any, optimizer: Any, state: dict[str, Any],
                        rng_state: dict[str, Any], source_identity: dict[str, Any],
                        torch_module: Any, *, mask: dict[str, Any],
                        module_sha: Callable[[Any, str], str]) -> str:
    payload = build_r76_checkpoint_payload(controller, optimizer, state, rng_state,
                                           source_identity, mask=mask, module_sha=module_sha)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{hashlib.sha256(str(path).encode()).hexdigest()[:8]}.tmp")
    torch_module.save(payload, temporary)
    temporary.replace(path)
    return str(path)


def load_r76_checkpoint(path, torch_module: Any, *, controller_factory: Callable[[int], Any],
                        module_sha: Callable[[Any, str], str],
                        freeze_trunk_only: Callable[[Any], dict[str, Any]],
                        build_optimizer: Callable[[Any, float], Any],
                        restore_rng: Callable[[dict[str, Any]], None],
                        rng_state: Callable[[], dict[str, Any]] | None = None,
                        expected_context: dict[str, Any] | None = None
                        ) -> tuple[Any, Any, dict[str, Any]]:
    payload = validate_r76_full_payload(torch_module.load(
        Path(path), map_location="cpu", weights_only=False))
    validate_r76_resume_context(payload, expected_context)
    state = payload["training_state"]
    seed = int(state["training_seed"])
    controller = controller_factory(seed)
    controller.load_state_dict(payload["controller"])
    identity = {name: module_sha(controller, name) for name in contracts.R76_MODULES}
    if identity != payload["source_identity"].get("module_sha256"):
        raise R76IdentityError("restored module values do not match the saved R76 identity")
    mask = freeze_trunk_only(controller)
    contracts.validate_r76_mask_report(mask)
    if [len(group["params"]) for group in payload["optimizer"]["param_groups"]] != [8]:
        raise R76IdentityError("refusing to restore an optimizer that is not eight objects")
    optimizer = build_optimizer(controller, 3e-4)
    if parameter_object_mapping(controller, optimizer) != payload["parameter_object_names"]:
        raise R76IdentityError("restored optimizer parameter-name mapping does not match R76")
    optimizer.load_state_dict(payload["optimizer"])
    restored_optimizer = optimizer.state_dict()
    if [len(group["params"]) for group in restored_optimizer["param_groups"]] != [8] \
            or set(restored_optimizer["state"]) != set(payload["optimizer"]["state"]):
        raise R76IdentityError(
            "restored optimizer parameter groups/slots do not match the saved R76 payload")
    for saved_group, restored_group in zip(payload["optimizer"]["param_groups"],
                                           restored_optimizer["param_groups"]):
        if set(saved_group) != set(restored_group) \
                or any(key != "params" and not _optimizer_value_equal(
                    saved_group.get(key), restored_group.get(key))
                    for key in set(saved_group) | set(restored_group)):
            raise R76IdentityError("restored Adam param-group values do not match the payload")
    for index in sorted(set(payload["optimizer"]["state"])):
        saved_state = payload["optimizer"]["state"][index]
        restored_state = restored_optimizer["state"][index]
        if set(saved_state) != set(restored_state) or any(
                not _optimizer_value_equal(saved_state[key], restored_state[key])
                for key in saved_state):
            raise R76IdentityError(
                f"restored Adam slot {index} value/shape/dtype does not match the payload")
    restore_rng(payload["rng"])
    if rng_state is not None and not _optimizer_value_equal(
            rng_state(), payload["rng"]):
        raise R76IdentityError("restored RNG state does not equal the saved RNG payload")
    return controller, optimizer, state
