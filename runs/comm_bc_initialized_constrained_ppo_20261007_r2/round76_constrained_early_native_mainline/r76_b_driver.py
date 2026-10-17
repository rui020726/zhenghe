"""ROUND76 B qualification driver: R71 model-only init and the 17-step save/restore path."""

from __future__ import annotations

import argparse
import hashlib
from importlib.machinery import PathFinder
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import r76_b_contracts as b_contracts
import r76_execute
import r76_runtime


HERE = Path(__file__).resolve().parent
SEED = 11
BRANCH_PHYSICAL_STEPS = 9
SCIENTIFIC_CHECKPOINT_COUNT = 3
SCIENTIFIC_CHECKPOINT_LOAD_COUNT = 2

R2_CANONICAL_R66_DIR = HERE.parent / "round66_bc_early_path_preserving_late_adaptation"
R2_CANONICAL_R68_DIR = HERE.parent / "round68_late_initial_confidence_ablation"
R2_COLD_START_SOURCES = {
    "r66_execute": {
        "path": R2_CANONICAL_R66_DIR / "r66_execute.py",
        "bytes": 53131,
        "sha256": "9164f0c989847e78d1018512bcdd26cb1d234971f61f05f03a6d58927cf14670",
    },
    "r68_execute": {
        "path": R2_CANONICAL_R68_DIR / "r68_execute.py",
        "bytes": 61201,
        "sha256": "c123dfe753effc5b6e35b8ce680e6bb1092a80a0d0deb40a0971df09e1c2cd3e",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_r2_cold_start_paths() -> dict[str, Any]:
    """Pin canonical R66/R68 cold-start origins without importing either module."""
    r66_dir = R2_CANONICAL_R66_DIR.resolve()
    r68_dir = R2_CANONICAL_R68_DIR.resolve()
    if not r66_dir.is_dir() or not r68_dir.is_dir():
        raise RuntimeError("R2 canonical R66/R68 runtime directories are missing")
    if Path(r76_execute.R66_DIR).resolve() != r66_dir:
        raise RuntimeError("R2 pinned R66 path differs from r76_execute.R66_DIR")
    resolved_paths: dict[str, Path] = {}
    for name, expected in R2_COLD_START_SOURCES.items():
        path = expected["path"].resolve()
        resolved_paths[name] = path
        module = sys.modules.get(name)
        if module is not None:
            spec = getattr(module, "__spec__", None)
            origin = getattr(spec, "origin", None) if spec is not None else None
            module_file = getattr(module, "__file__", None)
            if not isinstance(origin, str) or Path(origin).resolve() != path:
                raise RuntimeError(
                    f"R2 cold start found a foreign or opaque cached {name} module")
            if module_file is not None and Path(module_file).resolve() != path:
                raise RuntimeError(
                    f"R2 cold start cached {name} __file__ does not match its origin")

        if not path.is_file():
            raise RuntimeError(f"R2 pinned {name} source is missing: {path}")
        if path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"R2 pinned {name} source byte size drifted")
        if sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"R2 pinned {name} source SHA256 drifted")

    for directory in sorted({r66_dir, r68_dir}, key=lambda value: str(value)):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))

    return {
        "kind": "round76-r2-cold-start-path-identity/1",
        "r66_dir": str(r66_dir),
        "r68_dir": str(r68_dir),
        "sources": {
            name: str(_resolve_final_search_origin(name, path))
            for name, path in resolved_paths.items()
        },
    }


def _resolve_final_search_origin(name: str, expected_path: Path) -> Path:
    """Resolve the actual final sys.path search result without executing the module."""
    spec = PathFinder.find_spec(name, sys.path)
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not isinstance(origin, str) or Path(origin).resolve() != expected_path:
        got = str(Path(origin).resolve()) if origin else "<none>"
        raise RuntimeError(
            f"R2 {name} final search origin mismatch: {got} != {expected_path}")
    return Path(origin).resolve()


def _write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_r2_input_counts(counts: dict[str, Any], contract: dict[str, Any], *,
                           capture_root: Path | None = None) -> dict[str, Any]:
    """Write the pre-model counts once, durably, and never overwrite a partial artifact."""
    root = (Path(capture_root) if capture_root is not None else
            HERE / "raw" / "qualification" / "R76_B2_R2_PARENT_CAPTURE_20261017")
    if not root.is_dir():
        raise RuntimeError("R2 parent capture root must already exist before counts write")
    path = root / "input_counts.json"
    if path.exists():
        raise RuntimeError("R2 input counts artifact already exists")
    payload = {
        "kind": "round76-r2-pre-model-input-counts/1",
        "identity": {
            "request_id": contract["request_id"],
            "task_id": contract["task_id"],
            "branch": contract["branch"],
            "scope": contract["scope"],
            "source_head": contract["source_head"],
        },
        "input_counts": dict(counts),
    }
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_b2_release(gates: dict[str, Any]) -> None:
    if not all(gates.get(name) is True for name in (
            "b0_source_review_passed", "b1_pure_tests_released",
            "b2_cpu_qualification_released")):
        raise RuntimeError("R76 B2 is not released by the current decision-chat gate state")


def validate_b2_bundle(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], str]:
    contract_path = Path(args.contract_path).resolve()
    permit_path = Path(args.permit_path).resolve()
    contract_sha256 = sha256_file(contract_path)
    permit = _read_json(permit_path)
    b_contracts.validate_permit(permit, source_head=args.source_head,
                                contract_sha256=contract_sha256)
    contract = _read_json(contract_path)
    b_contracts.validate_contract(contract, source_head=args.source_head,
                                  driver_sha256=sha256_file(HERE / "r76_b_driver.py"))
    return contract, permit, contract_sha256


def pre_model_input_counts(input_manifest: dict[str, Any]
                           ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Hash and materialize four episodes once; count actual ``batch_samples`` N."""
    import r76_episode_protocol as protocol

    total_records = 0
    samples_n = 0
    effective_feature_rows = 0
    max_targets_per_record = 0
    episodes: list[dict[str, Any]] = []
    for frozen in input_manifest["episodes"]:
        row_path = b_contracts.RUN_ROOT / frozen["row_path"]
        tape_path = b_contracts.RUN_ROOT / frozen["tape_path"]
        if row_path.stat().st_size != int(frozen["row_bytes"]) \
                or sha256_file(row_path) != frozen["row_sha256"]:
            raise RuntimeError(f"R76 B frozen row drifted: {row_path}")
        if tape_path.stat().st_size != int(frozen["tape_bytes"]) \
                or sha256_file(tape_path) != frozen["tape_sha256"]:
            raise RuntimeError(f"R76 B frozen tape drifted: {tape_path}")
        row = protocol.read_row(row_path)
        tape = protocol.read_tape(tape_path)
        if int(row["training_seed"]) != SEED \
                or int(row["episode_index"]) != int(frozen["episode"]) \
                or int(row["scenario_seed"]) != int(frozen["scenario"]) \
                or float(row["lambda"]) != 1.0:
            raise RuntimeError(f"R76 B frozen episode identity mismatch: {row_path}")
        episode = protocol.rebuild_episode(row, tape)
        records = episode["collector"].records
        if not records:
            raise RuntimeError("R76 B frozen episode has no recorded decision steps")
        for record in records:
            targets = record["targets"]
            features = record["features"]
            if len(features) != len(targets) or len(targets) > 6:
                raise RuntimeError("R76 B frozen record exceeds six routed targets")
            if any(len(feature) != 19 for feature in features):
                raise RuntimeError("R76 B frozen feature is not a 19-vector")
            effective_feature_rows += len(features)
            # v2_trainer.batch_samples emits exactly one sample per recorded tape row.
            # Count that N here without calling batch_samples or parsing the real tape.
            samples_n += 1
            max_targets_per_record = max(max_targets_per_record, len(targets))
        total_records += len(records)
        episodes.append(episode)
    if len(episodes) != 4:
        raise RuntimeError("R76 B requires exactly four materialized episodes")
    if samples_n != b_contracts.SAMPLES_PER_BATCH:
        raise RuntimeError(
            "R76 B batch sample count mismatch: "
            f"N={samples_n} != {b_contracts.SAMPLES_PER_BATCH}")
    if total_records > b_contracts.MAX_UNIQUE_RECORDS \
            or effective_feature_rows > b_contracts.MAX_SINGLE_TRUNK_ROWS:
        raise RuntimeError("R76 B frozen input field-count limit exceeded")
    counts = {
        "M": total_records,
        "N": samples_n,
        "valid_samples": samples_n,
        "T": effective_feature_rows,
        "max_targets_per_record": max_targets_per_record,
        "episodes": len(episodes),
    }
    return counts, episodes


def configure_qualification_runtime(contract: dict[str, Any],
                                   resource_contract_sha256: str) -> dict[str, Any]:
    cold_start = validate_r2_cold_start_paths()
    if str(r76_execute.R66_DIR) not in sys.path:
        sys.path.insert(0, str(r76_execute.R66_DIR))
    import r66_execute
    import r68_execute
    import r76_guard
    import r76_native

    cfg = r66_execute.configure_and_inject(HERE / "raw")
    r40 = cfg["runtime"]["r40_common"]
    common = cfg["common"]
    environment_caps = contract["resource_caps"]["environment_construction"]
    for holder, names in ((r40, ("NORMAL_CAP", "RESERVE_CAP", "HARD_CAP")),
                          (common, ("NORMAL_LAUNCHES", "RESERVE_LAUNCHES", "LAUNCH_CAP"))):
        for name, value in zip(names, (environment_caps["normal"],
                                       environment_caps["reserve"],
                                       environment_caps["hard"])):
            setattr(holder, name, value)
    r40.plan = lambda: {"normal_total": environment_caps["normal"],
                        "technical_reserve_max": environment_caps["reserve"],
                        "hard_cap": environment_caps["hard"],
                        "task_id": b_contracts.TASK_ID}
    resources = r76_guard.R75Resources.qualification_from_contract(contract)
    resources.verify_or_fail()
    cfg["r76_source_manifest"] = r76_execute._validated_source_manifest(
        r76_native, r68_execute, cfg)
    cfg["r76_controller_factory"] = lambda seed: r76_execute._fresh_architecture_controller(
        r76_native, seed)
    identity = r76_runtime.install_r76_runtime_bindings(
        cfg, resources, r76_native, request_id=b_contracts.REQUEST_ID,
        task_id=b_contracts.TASK_ID)
    cfg["r76_resource_contract_sha256"] = resource_contract_sha256
    cfg["r76_resource_contract"] = contract
    cfg["r76_runtime_identity"] = identity
    cfg["r76_cold_start_identity"] = cold_start
    return cfg


def _build_identity_context(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    import r68_execute

    common = cfg["common"]
    order = [int(value) for value in common.scenario_order()]
    data_identity = {
        "data_order_seed": int(common.DATA_ORDER_SEED),
        "b0_bytes": float(common.B0_BYTES),
        "episodes_per_seed": int(common.EPISODES_PER_SEED),
        "scenario_order_sha256": r76_contracts_sha256_text(order),
        "seeds": [11, 22, 33],
    }
    references = r68_execute.reuse_r66_references(cfg, order)
    reference_identity = r76_execute._reference_identity(cfg, order)
    cfg["r76_reference_identity"] = reference_identity
    cfg["r76_references"] = references
    cfg["r76_resume_context"] = {
        "seed": SEED,
        "data_identity": data_identity,
        "reference_identity": reference_identity,
        "config_identity": dict(cfg["r76_runtime_identity"]["live_config"]),
        "resource_contract_sha256": str(cfg["r76_resource_contract_sha256"]),
    }
    cfg["r76_data_identity"] = data_identity
    return data_identity, reference_identity


def r76_contracts_sha256_text(value: Any) -> str:
    import r76_contracts as contracts

    return contracts.sha256_text(value)


def _clone_value(value: Any):
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().cpu().clone()
    if hasattr(value, "copy") and hasattr(value, "dtype"):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        cloned = [_clone_value(item) for item in value]
        return tuple(cloned) if isinstance(value, tuple) else cloned
    return value


def _capture_full_state(cfg: dict[str, Any], controller: Any, optimizer: Any,
                        state: dict[str, Any]) -> dict[str, Any]:
    import r76_native

    source_identity = r76_runtime.r76_source_identity(
        cfg, request_id=b_contracts.REQUEST_ID, task_id=b_contracts.TASK_ID)
    payload = r76_runtime.build_r76_checkpoint_payload(
        controller, optimizer, state, cfg["trainer"].rng_state(), source_identity,
        mask=r76_native.assert_mask(controller), module_sha=r76_native.module_sha,
        expected_identity=cfg["r76_expected_identity"])
    return _clone_value(payload)


MISSING = object()


def _dict_value(value: Any, key: Any) -> Any:
    return value[key] if isinstance(value, dict) and key in value else MISSING


def _equal_value(left: Any, right: Any) -> bool:
    if hasattr(left, "shape") and hasattr(left, "dtype"):
        if not hasattr(right, "shape") or not hasattr(right, "dtype"):
            return False
        return bool(left.shape == right.shape and left.dtype == right.dtype
                    and (left == right).all())
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_equal_value(left[key], right[key])
                                               for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_equal_value(a, b)
                                               for a, b in zip(left, right))
    return left == right


def _differences(left: Any, right: Any, path: str = "") -> list[str]:
    if _equal_value(left, right):
        return []
    if isinstance(left, dict) and isinstance(right, dict):
        found: list[str] = []
        for key in sorted(set(left) | set(right)):
            left_value = _dict_value(left, key)
            right_value = _dict_value(right, key)
            key_path = f"{path}.{key}" if path else str(key)
            if left_value is MISSING or right_value is MISSING:
                if left_value is MISSING and right_value is MISSING:
                    continue
                side = "missing_left" if left_value is MISSING else "missing_right"
                found.append(f"{key_path}:{side}")
            else:
                found.extend(_differences(left_value, right_value, key_path))
        return found
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) \
            and len(left) == len(right):
        found: list[str] = []
        for index, (a, b) in enumerate(zip(left, right)):
            found.extend(_differences(a, b, f"{path}[{index}]"))
        return found
    return [path or "<root>"]


def _a1_input_width(head: Any) -> int:
    for attribute in ("in_features", "input_dim"):
        value = getattr(head, attribute, None)
        if isinstance(value, int):
            return value
    for parameter in head.parameters():
        if parameter.dim() == 2:
            return int(parameter.shape[1])
    raise RuntimeError("cannot determine critic input width without a forward")


def a1_critic_control_step(cfg: dict[str, Any], controller: Any, optimizer: Any,
                           state: dict[str, Any]) -> dict[str, Any]:
    import r76_native

    torch = r76_native.torch
    resources = cfg["r76"]["resources"]
    objects = r76_runtime.parameter_object_mapping(controller, optimizer)
    parameters = list(optimizer.param_groups[0]["params"])
    if len(objects) != 8 or len(objects) != len(set(objects)):
        raise RuntimeError("A1 requires the exact eight-object optimizer mapping")
    trunk_parameters = list(controller.inner.trunk.parameters())
    optimizer_ids = {id(parameter) for parameter in parameters}
    trunk_in_optimizer = [index for index, parameter in enumerate(trunk_parameters)
                          if id(parameter) in optimizer_ids]
    all_modules = ("trunk", "early_head", "late_head", "value_task_head", "value_cost_head")
    before_module_sha = {name: r76_native.module_sha(controller, name)
                         for name in all_modules}
    before_parameter_sha = controller.parameter_summary()["parameter_sha256"]
    width = _a1_input_width(controller.inner.value_head)
    hidden = torch.zeros((4, width), dtype=torch.float32, requires_grad=False)
    task_values = controller.inner.value_head(hidden).squeeze(-1)
    cost_values = controller.cost_head(hidden).squeeze(-1)
    loss = (task_values - 1.0).pow(2).mean() + (cost_values + 1.0).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters, float(cfg["common"].MAX_GRAD_NORM))
    resources.charge_physical_step("r76_b_A1_critic_only_control_step", category="normal")
    optimizer.step()
    after_module_sha = {name: r76_native.module_sha(controller, name)
                        for name in all_modules}
    after_parameter_sha = controller.parameter_summary()["parameter_sha256"]
    moved_modules = sorted(name for name, value in before_module_sha.items()
                           if value != after_module_sha[name])
    adam_record = {}
    for name, parameter in zip(objects, parameters):
        adam_state = optimizer.state.get(parameter, {})
        adam_record[name] = {
            "state_present": bool(adam_state),
            "exp_avg_present": "exp_avg" in adam_state,
            "exp_avg_sq_present": "exp_avg_sq" in adam_state,
        }
    report = {
        "kind": "round76-b-a1-critic-only-control-step/1",
        "loss": float(loss.item()),
        "gradient_norm": float(gradient_norm),
        "critic_input_width": width,
        "optimizer_steps": 1,
        "trunk_calls": 0,
        "joint_records": 0,
        "objects": objects,
        "trunk_parameters_measured": len(trunk_parameters),
        "trunk_parameters_in_optimizer": len(trunk_in_optimizer),
        "trunk_requires_grad": [bool(parameter.requires_grad)
                                for parameter in trunk_parameters],
        "moved_modules": moved_modules,
        "moved_objects": moved_modules,
        "adam_record": adam_record,
        "parameters_before_sha256": before_parameter_sha,
        "parameters_after_sha256": after_parameter_sha,
    }
    problems: list[str] = []
    if trunk_in_optimizer or any(name in moved_modules for name in
                                 ("trunk", "early_head", "late_head")):
        problems.append("A1 moved or included trunk/actor parameters")
    if len(objects) != 8 or len(objects) != len(set(objects)):
        problems.append("A1 did not map exactly eight unique trainable objects")
    for name in objects:
        if name.startswith(("value_task_head", "value_cost_head")):
            entry = adam_record[name]
            if not entry["state_present"] or not entry["exp_avg_present"] \
                    or not entry["exp_avg_sq_present"]:
                problems.append(f"critic {name} has incomplete Adam state")
    if problems:
        raise RuntimeError("A1 gate refused: " + "; ".join(problems))
    state["optimizer_steps"] += 1
    _write_json(HERE / "raw" / "qualification" / "r76_b_a1_control.json", report)
    return report


def _meter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    keys = ("calls_attempted", "calls_charged", "rows_charged", "calls_completed",
            "rows_completed", "calls_refused", "calls_failed")
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in keys}


def native_update(cfg: dict[str, Any], controller: Any, optimizer: Any,
                  state: dict[str, Any], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    import r76_native

    resources = cfg["r76"]["resources"]
    state["lambda_at_rollout"] = float(state["lambda"])
    meter = resources.ensure_trunk_meter(controller, what_prefix="r76_b_native_update_0")
    pre_meter = meter.snapshot()
    report = r76_native.r75_native_ppo_update(
        controller, optimizer, episodes, update_index=0,
        resources=resources, physical_step_category="normal")
    if int(report["optimizer_steps"]) != 8:
        raise RuntimeError("each qualification native branch must perform exactly eight steps")
    alignment = report["likelihood_alignment"]
    if float(alignment["max_abs_gap"]) > 1e-6 or not alignment["passed"]:
        raise RuntimeError("the qualification native branch failed likelihood alignment")
    report["trunk_entry_meter_delta"] = _meter_delta(pre_meter, meter.snapshot())
    trainer = cfg["trainer"]
    g_record = trainer.batch_g_hat(episodes, cfg["r76_references"])
    dual_record = trainer.dual_update(state["lambda"], g_record["g_hat"])
    state["lambda"] = dual_record["lambda_next"]
    state["lambda_history"].append({**dual_record, "batch": int(state["updates"])})
    state["updates"] += 1
    state["optimizer_steps"] += int(report["optimizer_steps"])
    state["episodes_done"] += 4
    return report


B2_PHASES = ("init", "a1", "a2", "a3_restore", "a3", "comparison", "success_gate")


class B2OperationMeter:
    """Attempt/completed accounting at real operation call sites; failures never refund."""

    def __init__(self, caps: dict[str, int]):
        self.limits = dict(caps)
        self.counts = {name: {"attempted": 0, "completed": 0, "failed": 0}
                       for name in caps}

    def _begin(self, name: str) -> None:
        if name not in self.counts:
            raise RuntimeError(f"unknown R76 B operation: {name}")
        entry = self.counts[name]
        if entry["attempted"] >= int(self.limits[name]):
            raise RuntimeError(
                f"R76 B hard operation cap reached: {name} "
                f"{entry['attempted']}/{self.limits[name]}")
        entry["attempted"] += 1

    @contextmanager
    def operation(self, *names: str):
        begun: list[str] = []
        try:
            for name in names:
                self._begin(name)
                begun.append(name)
        except BaseException:
            for name in begun:
                self.counts[name]["failed"] += 1
            raise
        try:
            yield
        except BaseException:
            for name in begun:
                self.counts[name]["failed"] += 1
            raise
        else:
            for name in begun:
                self.counts[name]["completed"] += 1

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {name: dict(entry) for name, entry in self.counts.items()}


class RealB2Ops:
    """Thin adapter around the already accepted R76 real functions and live cfg."""

    def __init__(self, cfg: dict[str, Any], output_root: Path,
                 input_counts: dict[str, Any], episodes: list[dict[str, Any]]):
        b2_caps = b_contracts.B_STAGE_CAPS["b2"]
        self.cfg = cfg
        self.output_root = Path(output_root)
        self.input_counts = input_counts
        self.episodes = episodes
        self.a1_baseline: dict[str, Any] | None = None
        self.scientific_files: dict[str, dict[str, Any]] = {}
        self.meter = B2OperationMeter({
            "controller_factory": int(b2_caps["factory_constructions"]),
            "adam_construction": int(b2_caps["adam_constructions"]),
            "init_load": int(b2_caps["init_loads"]),
            "own_a1_load": int(b2_caps["own_a1_loads"]),
            "checkpoint_save": int(b2_caps["checkpoint_saves"]),
            "checkpoint_load": int(b2_caps["checkpoint_loads"]),
        })
        self._base_controller_factory = cfg["r76_controller_factory"]
        cfg["r76_controller_factory"] = self.build_controller_factory
        cfg["r76_build_optimizer"] = self.build_optimizer

    def build_controller_factory(self, seed: int) -> Any:
        with self.meter.operation("controller_factory"):
            return self._base_controller_factory(int(seed))

    def build_controller(self) -> tuple[Any, dict[str, Any]]:
        with self.meter.operation("init_load", "checkpoint_load"):
            return r76_execute.r76_fresh_controller(self.cfg, SEED)

    def build_optimizer(self, controller: Any,
                        lr: float | None = None) -> Any:
        with self.meter.operation("adam_construction"):
            import r76_native

            optimizer_lr = float(
                self.cfg["common"].LEARNING_RATE if lr is None else lr)
            return r76_native.build_optimizer(
                controller, optimizer_lr)

    def build_state(self) -> dict[str, Any]:
        common = self.cfg["common"]
        return r76_runtime.build_r76_state(
            SEED, data_order_seed=int(common.DATA_ORDER_SEED),
            b0_bytes=float(common.B0_BYTES),
            identity={"request_id": b_contracts.REQUEST_ID,
                      "task_id": b_contracts.TASK_ID})

    def a1(self, controller: Any, optimizer: Any,
           state: dict[str, Any]) -> dict[str, Any]:
        return a1_critic_control_step(self.cfg, controller, optimizer, state)

    def capture(self, controller: Any, optimizer: Any,
                state: dict[str, Any]) -> dict[str, Any]:
        return _capture_full_state(self.cfg, controller, optimizer, state)

    def save_checkpoint(self, controller: Any, optimizer: Any,
                        state: dict[str, Any], path: Path) -> dict[str, Any]:
        with self.meter.operation("checkpoint_save"):
            return _save_full_state(self.cfg, controller, optimizer, state, Path(path))

    def load_checkpoint(self, path: Path) -> tuple[Any, Any, dict[str, Any]]:
        with self.meter.operation("own_a1_load", "checkpoint_load"):
            return self.cfg["trainer"].load_checkpoint(Path(path))

    def native_update(self, controller: Any, optimizer: Any,
                      state: dict[str, Any], episodes: list[dict[str, Any]]) -> dict[str, Any]:
        return native_update(self.cfg, controller, optimizer, state, episodes)

    def verify_disk(self, path: Path, expected_sha256: str) -> None:
        if sha256_file(Path(path)) != expected_sha256:
            raise RuntimeError(f"R76 B disk checkpoint drifted: {path}")

    def resource_snapshot(self) -> dict[str, Any]:
        return self.cfg["r76"]["resources"].snapshots()

    def verify_resources(self) -> dict[str, Any]:
        return self.cfg["r76"]["resources"].verify_or_fail()

    def operation_counts(self) -> dict[str, dict[str, int]]:
        return self.meter.snapshot()

    def record_stage(self, value: dict[str, Any]) -> str:
        path = self.output_root / "r76_b_execution_stage.json"
        if path.exists():
            raise RuntimeError("R76 B private execution-stage record already exists")
        _write_json(path, value)
        return str(path)

    def write_result(self, value: dict[str, Any]) -> str:
        path = self.output_root / "r76_b_results.json"
        if path.exists():
            raise RuntimeError("R76 B result record already exists")
        _write_json(path, value)
        return str(path)


def _counter_used(snapshot: dict[str, Any], unit: str) -> int:
    return int(snapshot[unit]["used"])


def orchestrate(ops: Any) -> int:
    """Pure fixed B2 sequence; all environment/NN/checkpoint side effects come through ops."""
    phases = {name: {"status": "not_started"} for name in B2_PHASES}
    snapshots: dict[str, dict[str, Any]] = {}
    reports: dict[str, dict[str, Any]] = {}
    current: str | None = None

    def begin(name: str) -> None:
        nonlocal current
        current = name
        phases[name] = {"status": "running"}

    def complete(name: str) -> None:
        phases[name] = {"status": "completed"}

    try:
        snapshots["before"] = ops.resource_snapshot()
        begin("init")
        controller, init_report = ops.build_controller()
        optimizer = ops.build_optimizer(controller)
        state = ops.build_state()
        reports["init"] = init_report
        complete("init")

        begin("a1")
        reports["a1"] = ops.a1(controller, optimizer, state)
        a1_baseline = ops.capture(controller, optimizer, state)
        ops.a1_baseline = a1_baseline
        a1_path = ops.output_root / "r76_b_a1_state.pt"
        a1_file = ops.save_checkpoint(controller, optimizer, state, a1_path)
        ops.scientific_files["a1"] = a1_file
        snapshots["after_a1"] = ops.resource_snapshot()
        complete("a1")

        begin("a2")
        reports["a2"] = ops.native_update(controller, optimizer, state, ops.episodes)
        a2_payload = ops.capture(controller, optimizer, state)
        a2_path = ops.output_root / "r76_b_a2_state.pt"
        a2_file = ops.save_checkpoint(controller, optimizer, state, a2_path)
        snapshots["after_a2"] = ops.resource_snapshot()
        complete("a2")

        begin("a3_restore")
        a1_file = ops.scientific_files["a1"]
        ops.verify_disk(a1_path, a1_file["sha256"])
        restored_controller, restored_optimizer, restored_state = ops.load_checkpoint(a1_path)
        a3_start = ops.capture(restored_controller, restored_optimizer, restored_state)
        a3_start_differences = _differences(ops.a1_baseline, a3_start)
        if a3_start_differences:
            raise RuntimeError("A3 start differs from the A1 baseline: "
                               + "; ".join(a3_start_differences))
        complete("a3_restore")

        begin("a3")
        reports["a3"] = ops.native_update(restored_controller, restored_optimizer,
                                          restored_state, ops.episodes)
        a3_payload = ops.capture(restored_controller, restored_optimizer, restored_state)
        a3_path = ops.output_root / "r76_b_a3_state.pt"
        a3_file = ops.save_checkpoint(restored_controller, restored_optimizer,
                                      restored_state, a3_path)
        snapshots["after_a3"] = ops.resource_snapshot()
        complete("a3")

        begin("comparison")
        payload_differences = _differences(a2_payload, a3_payload)
        scientific_differences = _differences(reports["a2"], reports["a3"])
        if payload_differences or scientific_differences:
            raise RuntimeError("A2/A3 full-state comparison failed: "
                               + "; ".join(payload_differences + scientific_differences))
        complete("comparison")

        begin("success_gate")
        verification = ops.verify_resources()
        if not bool(verification.get("consistent")):
            raise RuntimeError("R76 B success gate failed resource reconciliation")
        step_entries = sum(int(reports[name]["optimizer_steps"])
                           for name in ("a1", "a2", "a3"))
        if step_entries != _counter_used(snapshots["after_a3"],
                                         "physical_optimizer_step"):
            raise RuntimeError("R76 B optimizer step entries do not match the central guard")
        expected_phase_steps = {"before": 0, "after_a1": 1,
                                "after_a2": 9, "after_a3": 17}
        phase_steps = {name: _counter_used(snapshots[name], "physical_optimizer_step")
                       for name in expected_phase_steps}
        if phase_steps != expected_phase_steps:
            raise RuntimeError(f"R76 B phase step additivity failed: {phase_steps}")
        expected_counts = {
            "controller_factory": (2, 2),
            "adam_construction": (2, 2),
            "init_load": (1, 1),
            "own_a1_load": (1, 1),
            "checkpoint_save": (3, 3),
            "checkpoint_load": (2, 2),
        }
        actual_counts = ops.operation_counts()
        for name, (attempted, completed) in expected_counts.items():
            entry = actual_counts[name]
            if entry["attempted"] != attempted or entry["completed"] != completed:
                raise RuntimeError(
                    f"R76 B operation count mismatch for {name}: {entry}")
        complete("success_gate")

        result = {
            "kind": "round76-b-qualification-results/1",
            "task_id": b_contracts.TASK_ID,
            "input_counts": ops.input_counts,
            "init_report": reports["init"],
            "a1": reports["a1"],
            "a2": reports["a2"],
            "a3": reports["a3"],
            "a2_a3_full_state_equal": True,
            "a2_a3_scientific_report_equal": True,
            "a2_file": a2_file,
            "a3_file": a3_file,
            "resource_snapshot": snapshots["after_a3"],
            "branch_path_physical_steps": {
                "a1_a2": _counter_used(snapshots["after_a2"],
                                        "physical_optimizer_step"),
                "a1_a3": _counter_used(snapshots["after_a2"],
                                        "physical_optimizer_step"),
            },
            "branch_path_optimizer_step_entries": {
                "a1_a2": int(reports["a1"]["optimizer_steps"])
                         + int(reports["a2"]["optimizer_steps"]),
                "a1_a3": int(reports["a1"]["optimizer_steps"])
                         + int(reports["a3"]["optimizer_steps"]),
            },
            "central_optimizer_steps": _counter_used(snapshots["after_a3"],
                                                      "physical_optimizer_step"),
            "optimizer_step_entry_count": step_entries,
            "operation_counts": actual_counts,
        }
        ops.write_result(result)
        return 0
    except BaseException as error:
        if current is not None:
            phases[current] = {"status": "failed", "error": repr(error)}
        try:
            snapshots["failure"] = ops.resource_snapshot()
        except BaseException as snapshot_error:
            snapshots["failure"] = {"capture_error": repr(snapshot_error)}
        for name in B2_PHASES:
            if name not in phases or phases[name].get("status") == "not_started":
                phases[name] = {"status": "not_executed"}
        stage = {
            "kind": "round76-b-private-execution-stage/1",
            "status": "failed",
            "phases": phases,
            "operation_counts": ops.operation_counts(),
            "resource_snapshots": snapshots,
            "error": repr(error),
        }
        ops.record_stage(stage)
        raise


def _save_full_state(cfg: dict[str, Any], controller: Any, optimizer: Any,
                     state: dict[str, Any], path: Path) -> dict[str, Any]:
    cfg["trainer"].save_checkpoint(controller, optimizer, state, path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": Path(path).stat().st_size}


def run_b2(args: argparse.Namespace) -> int:
    validate_b2_release(_read_json(HERE / "ROUND76_B_TASK_STATE.json")["gates"])
    contract, permit, contract_sha256 = validate_b2_bundle(args)
    input_counts, episodes = pre_model_input_counts(contract["input_manifest"])
    _write_r2_input_counts(input_counts, contract)
    cfg = configure_qualification_runtime(contract, contract_sha256)
    _build_identity_context(cfg)
    ops = RealB2Ops(cfg, b_contracts.RESOURCE_ROOT, input_counts, episodes)
    return orchestrate(ops)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROUND76 B2 native-restore qualification")
    parser.add_argument("--show-b0-state", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--contract-path")
    parser.add_argument("--permit-path")
    parser.add_argument("--source-head")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.show_b0_state:
        print(json.dumps(b_contracts.build_b0_state(), ensure_ascii=False, indent=2,
                         sort_keys=True))
        return 0
    if args.execute:
        missing = [name for name in ("contract_path", "permit_path", "source_head")
                   if not getattr(args, name)]
        if missing:
            raise SystemExit(f"B2 requires --{', --'.join(missing)}")
        return run_b2(args)
    raise SystemExit("B2 requires --execute or --show-b0-state")
