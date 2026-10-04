"""Read-only audit of the committed seed-11 batch and exact ROUND43 resume bridge."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

import r42_run
import r43_call_counter_hooks


FREEZE_PATH = RUN_DIR / "ROUND43_FROZEN_CONFIG.json"
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
RUNTIME_DIR = RUN_DIR / "runtime"
OLD_LEDGER_COPY = RUNTIME_DIR / "ROUND43_EXECUTION_LEDGER_FREEZE1_AT_STOP.json"
OLD_PROGRESS_COPY = RUNTIME_DIR / "ROUND43_EXECUTION_PROGRESS_FREEZE1.json"
OLD_RECEIPT_COPY = RUNTIME_DIR / "ROUND43_PRELAUNCH_IDENTITY_FREEZE1.json"
OLD_PREFLIGHT_COPY = RUNTIME_DIR / "ROUND43_PREFLIGHT_FREEZE1.json"
FAILURE_PATH = RUN_DIR / "failures" / "ROUND43_RUNCALL_RESUME_FAILURE_FREEZE1.json"
STATE_AUDIT_PATH = RUN_DIR / "ROUND43_SEED11_BATCH0_STATE_RECONCILIATION.json"
LEDGER_AUDIT_PATH = RUN_DIR / "ROUND43_LEDGER_RECONCILIATION.json"
GATE_PATH = RUN_DIR / "ROUND43_RESUME_RECOVERY_GATE.json"

ARTIFACT_DIRS = (
    "launch_claims", "launch_slots", "logs", "rows", "checkpoints",
    "training_progress", "runtime", "failures", "rejected",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_immutable(path: Path, value: Any) -> None:
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"recovery evidence already exists and differs: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()


def _deep_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def feed(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"torch")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b"numpy")
            digest.update(str(item.dtype).encode("ascii"))
            digest.update(json.dumps(list(item.shape)).encode("ascii"))
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b"{")
            for key in sorted(item, key=lambda x: repr(x)):
                feed(key)
                feed(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"[")
            for child in item:
                feed(child)
            digest.update(b"]")
        elif isinstance(item, np.generic):
            feed(item.item())
        else:
            digest.update(type(item).__name__.encode("utf-8"))
            digest.update(repr(item).encode("utf-8"))
            digest.update(b";")

    feed(value)
    return digest.hexdigest()


def _snapshot_artifacts() -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for directory in ARTIFACT_DIRS:
        root = RUN_DIR / directory
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                result[path.relative_to(RUN_DIR).as_posix()] = (
                    path.stat().st_size, _sha256(path))
    return result


def _run_tests(module_name: str) -> dict[str, Any]:
    module = __import__(module_name)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError(f"{module_name} failed:\n{stream.getvalue()}")
    return {
        "module": module_name,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "stdout": stream.getvalue(),
    }


def _module_parameter_manifest(controller: Any) -> list[tuple[str, Any]]:
    pairs: list[tuple[str, Any]] = []
    for prefix, module in (
        ("trunk", controller.inner.trunk),
        ("action_head", controller.inner.action_head),
        ("task_value_head", controller.inner.value_head),
        ("cost_head", controller.cost_head),
    ):
        pairs.extend((f"{prefix}.{name}", parameter)
                     for name, parameter in module.named_parameters())
    if [tuple(parameter.shape) for _name, parameter in pairs] != [
            tuple(parameter.shape) for parameter in controller.parameters]:
        raise AssertionError("actual optimizer parameter order differs from module order")
    return pairs


def _audit_state(runtime: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    commit_path = CHECKPOINT_DIR / "r40_batch_commit_s11_b0.json"
    record_path = CHECKPOINT_DIR / "r40_batch_s11_b0_attempt01.json"
    after_path = CHECKPOINT_DIR / "v2_train_s11_after_b0.pt"
    rolling_path = CHECKPOINT_DIR / "v2_train_s11.pt"
    preupdate_path = CHECKPOINT_DIR / "v2_train_s11_before_b0.pt"
    commit = _read(commit_path)
    record = _read(record_path)
    if _sha256(record_path) != commit["batch_record_sha256"]:
        raise AssertionError("batch record SHA differs from the immutable commit")
    for path, field in ((rolling_path, "rolling_checkpoint_sha256"),
                        (after_path, "postupdate_checkpoint_sha256")):
        if _sha256(path) != commit[field]:
            raise AssertionError(f"checkpoint SHA differs from the batch commit: {path.name}")
    if _sha256(preupdate_path) != record["preupdate_checkpoint_sha256"]:
        raise AssertionError("preupdate checkpoint SHA differs from its batch record")

    after_payload = torch.load(after_path, map_location="cpu", weights_only=False)
    rolling_payload = torch.load(rolling_path, map_location="cpu", weights_only=False)
    equal_sections = {}
    for name in ("controller", "optimizer", "training_state", "rng", "freeze"):
        left, right = after_payload[name], rolling_payload[name]
        left_hash, right_hash = _deep_hash(left), _deep_hash(right)
        equal_sections[name] = {"after_b0_sha256": left_hash,
                                "rolling_sha256": right_hash,
                                "identical": left_hash == right_hash}
        if left_hash != right_hash:
            raise AssertionError(f"after_b0 and rolling checkpoint differ in {name}")

    state = after_payload["training_state"]
    expected_fields = {
        "training_seed": 11,
        "episodes_done": 4,
        "updates": 1,
        "optimizer_steps": 12,
        "lambda": 1.0125,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "training_version": frozen["round42"]["training_version"],
        "reward_timing_version": frozen["round42"]["reward_timing_version"],
        "b0_bytes": 10000.0,
        "delta": 0.0,
        "eta_lambda": 0.05,
        "data_order_seed": 20260927,
    }
    for key, expected in expected_fields.items():
        if state.get(key) != expected:
            raise AssertionError(f"batch-0 checkpoint has wrong {key}: {state.get(key)!r}")
    expected_lambda_history = [{
        "batch": 0,
        "g_hat": record["dual"]["g_hat"],
        "lambda_before": record["dual"]["lambda_before"],
        "lambda_next": record["dual"]["lambda_next"],
    }]
    if state.get("lambda_history") != expected_lambda_history:
        raise AssertionError("batch-0 checkpoint lambda history does not close against the record")
    if commit["episodes_done"] != 4 or commit["updates"] != 1 \
            or commit["optimizer_steps"] != 12:
        raise AssertionError("batch-0 commit cursor or update count is inconsistent")

    rng_before = runtime["trainer"].rng_state()
    controller, loaded_state, read_only_identity = runtime["trainer"].load_checkpoint_read_only(
        after_path, freeze=frozen["round42"]["freeze_identity"])
    rng_after = runtime["trainer"].rng_state()
    if _deep_hash(rng_before) != _deep_hash(rng_after):
        raise AssertionError("read-only state audit changed ambient Python/NumPy/Torch RNG")
    if loaded_state != state:
        raise AssertionError("read-only loader state differs from the serialized recovery state")
    controller_identity = json.loads(json.dumps(controller.parameter_summary()))
    if controller_identity != record["controller_parameters_after"]:
        raise AssertionError("read-only loaded controller differs from the committed batch identity")
    module_digests = runtime["r40_train"]._module_digests(controller)
    if module_digests != record["controller_module_digests_after"]:
        raise AssertionError("module parameter hashes differ from the committed batch record")

    optimizer = after_payload["optimizer"]
    if len(optimizer["param_groups"]) != 1:
        raise AssertionError("batch-0 optimizer parameter-group count changed")
    group = optimizer["param_groups"][0]
    parameters = _module_parameter_manifest(controller)
    parameter_ids = list(group["params"])
    if len(parameters) != 10 or len(parameter_ids) != 10 \
            or set(optimizer["state"]) != set(parameter_ids):
        raise AssertionError("full dual-critic Adam parameter set is incomplete")
    moment_rows = []
    step_values = []
    for (name, parameter), parameter_id in zip(parameters, parameter_ids):
        entry = optimizer["state"][parameter_id]
        step = int(entry["step"].item()) if isinstance(entry["step"], torch.Tensor) else int(entry["step"])
        step_values.append(step)
        if step != 12:
            raise AssertionError(f"Adam state step for {name} is {step}, expected 12")
        if tuple(entry["exp_avg"].shape) != tuple(parameter.shape) \
                or tuple(entry["exp_avg_sq"].shape) != tuple(parameter.shape):
            raise AssertionError(f"Adam moment shape mismatch for {name}")
        if not torch.isfinite(entry["exp_avg"]).all() \
                or not torch.isfinite(entry["exp_avg_sq"]).all():
            raise AssertionError(f"Adam moments are non-finite for {name}")
        moment_rows.append({
            "parameter_id": int(parameter_id), "name": name,
            "shape": list(parameter.shape), "step": step,
            "exp_avg_sha256": _deep_hash(entry["exp_avg"]),
            "exp_avg_sq_sha256": _deep_hash(entry["exp_avg_sq"]),
            "exp_avg_l1": float(entry["exp_avg"].abs().sum()),
            "exp_avg_sq_l1": float(entry["exp_avg_sq"].abs().sum()),
        })
    if group["lr"] != 0.0003 or group["betas"] != (0.9, 0.999) \
            or group["eps"] != 1e-8 or group["weight_decay"] != 0:
        raise AssertionError("Adam hyperparameters differ from the frozen optimizer")

    common = runtime["common"]
    order = [int(x) for x in common.scenario_order()]
    batch_scenarios = [int(x) for x in record["scenarios"]]
    if order[:4] != batch_scenarios:
        raise AssertionError("saved batch scenarios are not the registered first four")
    remaining = order[4:]
    if len(remaining) != 124 or remaining != [int(x) for x in frozen["round42"]["training_order"][4:]]:
        raise AssertionError("remaining seed-11 scenario order changed")
    rng_parts = {key: _deep_hash(value) for key, value in after_payload["rng"].items()}
    if set(rng_parts) != {"python", "numpy", "torch"}:
        raise AssertionError(f"unexpected/incomplete registered RNG payload: {list(rng_parts)}")

    return {
        "kind": "round43-seed11-batch0-full-state-reconciliation/1",
        "status": "passed",
        "source_of_truth": "immutable after_b0 checkpoint plus matching batch commit and batch record",
        "paths": {name: {"path": str(path), "sha256": _sha256(path),
                         "size_bytes": path.stat().st_size}
                  for name, path in (("batch_commit", commit_path),
                                     ("batch_record", record_path),
                                     ("immutable_after_b0", after_path),
                                     ("rolling", rolling_path),
                                     ("preupdate", preupdate_path))},
        "checkpoint_sections": equal_sections,
        "state_cursor": {key: state[key] for key in (
            "training_seed", "episodes_done", "updates", "optimizer_steps", "lambda",
            "lambda_history", "training_version", "gae_lambda", "gamma", "data_order_seed")},
        "controller": {
            "parameter_identity": controller_identity,
            "module_parameter_sha256": module_digests,
            "read_only_load_identity": read_only_identity,
        },
        "optimizer": {
            "class": "torch.optim.Adam",
            "parameter_group_count": len(optimizer["param_groups"]),
            "parameter_count": len(parameters),
            "parameter_ids": [int(x) for x in parameter_ids],
            "optimizer_state_count": len(optimizer["state"]),
            "step_values_by_parameter": step_values,
            "all_steps_equal_committed_optimizer_steps": True,
            "param_group_hyperparameters": {k: v for k, v in group.items() if k != "params"},
            "moments_by_parameter": moment_rows,
            "full_optimizer_state_sha256": _deep_hash(optimizer),
        },
        "rng": {
            "components": rng_parts,
            "serialized_rng_sha256": _deep_hash(after_payload["rng"]),
            "ambient_rng_unchanged_by_read_only_load": True,
        },
        "scenario_cursor": {
            "batch0_scenarios": batch_scenarios,
            "remaining_episode_indices": [4, 127],
            "remaining_scenarios": remaining,
            "remaining_scenarios_sha256": hashlib.sha256(
                json.dumps(remaining).encode("utf-8")).hexdigest(),
        },
    }


def _copy_isolated_state(target: Path) -> None:
    names = (
        "v2_train_s11.pt", "v2_train_s11_after_b0.pt",
        "r40_batch_commit_s11_b0.json", "r40_batch_s11_b0_attempt01.json",
    )
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(CHECKPOINT_DIR / name, target / name)


def _probe_old_and_fixed_resume(runtime: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    round40_root = Path(frozen["execution_identity"]["round40_worktree"])
    original_path = (round40_root / "runs" / "comm_first_report_training_ablation_20261003"
                     / "r40_train.py")
    spec = importlib.util.spec_from_file_location("r43_original_r40_train_probe", original_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import the exact frozen ROUND40 training entry")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    common = runtime["common"]
    trainer = runtime["trainer"]
    adapter = runtime["r40_train"]
    r40_run = runtime["r40_run"]
    old_adapter_common, old_adapter_runner, old_adapter_trainer = (
        adapter.common, adapter.runner, adapter.trainer)
    adapter.common, adapter.runner, adapter.trainer = (
        common, runtime["runner"], trainer)
    old_common_checkpoints, old_common_output = common.CHECKPOINTS, common.OUTPUT_DIR
    old_common_logs = common.LOGS
    old_run_entry = r40_run.run_training_episode
    old_trainer_save = trainer.save_checkpoint
    old_trainer_ppo = trainer.ppo_update
    old_adam_step = torch.optim.Adam.step
    old_module_save = {name: getattr(adapter, name) for name in (
        "_save_preupdate_checkpoint", "_write_immutable_snapshot", "_write_batch_commit",
        "_save_milestone", "_save_progress")}
    original.common = common
    original.runner = runtime["runner"]
    original.trainer = trainer
    saved_rng = trainer.rng_state()
    saved_rng_hash = _deep_hash(saved_rng)
    optimizer_calls = 0
    ppo_calls = 0
    save_calls = 0
    environment_entries = 0
    scratch_checkpoint_writes = 0
    old_failure = None
    fixed_guard_error = None
    fixed_cursor = None

    def blocked_environment(*_args, **_kwargs):
        nonlocal environment_entries
        environment_entries += 1
        raise RuntimeError("ROUND43 recovery test blocked before environment construction")

    def blocked_save(*_args, **_kwargs):
        nonlocal save_calls
        save_calls += 1
        raise RuntimeError("ROUND43 recovery test blocked checkpoint save")

    def blocked_ppo(*_args, **_kwargs):
        nonlocal ppo_calls
        ppo_calls += 1
        raise RuntimeError("ROUND43 recovery test blocked PPO update")

    def blocked_step(*_args, **_kwargs):
        nonlocal optimizer_calls
        optimizer_calls += 1
        raise RuntimeError("ROUND43 recovery test blocked Adam.step")

    def no_checkpoint_file(*_args, **_kwargs):
        nonlocal scratch_checkpoint_writes
        scratch_checkpoint_writes += 1
        raise RuntimeError("isolated resume regression forbids checkpoint writes")

    def no_preupdate_file(_controller, _optimizer, _state, training_seed,
                          batch_index, _freeze):
        if int(training_seed) != 11 or int(batch_index) != 1:
            raise AssertionError("unexpected isolated resume checkpoint boundary")
        return common.CHECKPOINTS / "BLOCKED_BEFORE_B1.pt", "blocked-without-write"

    try:
        with tempfile.TemporaryDirectory(prefix="round43_resume_probe_") as temp_name:
            temp_root = Path(temp_name)
            temp_checkpoints = temp_root / "checkpoints"
            _copy_isolated_state(temp_checkpoints)
            common.CHECKPOINTS = temp_checkpoints
            common.OUTPUT_DIR = temp_root
            common.LOGS = temp_root / "logs"
            original._save_progress = lambda *_args, **_kwargs: None
            original_snapshot = {name: getattr(original, name, None) for name in (
                "_write_immutable_snapshot", "_write_batch_commit", "_save_preupdate_checkpoint",
                "_save_milestone", "_save_progress")}
            for name in original_snapshot:
                if original_snapshot[name] is not None:
                    setattr(original, name, no_checkpoint_file)
            trainer.save_checkpoint = blocked_save
            trainer.ppo_update = blocked_ppo
            torch.optim.Adam.step = blocked_step
            r40_run.run_training_episode = blocked_environment

            ref_rows = r42_run.reference_cache()
            reference_cache = {scene: row for scene, row in ref_rows.items()}
            try:
                original.train_seed(11, episodes=8,
                                    freeze=frozen["round42"]["freeze_identity"],
                                    reference_cache=reference_cache)
            except UnboundLocalError as error:
                old_failure = {
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "contains_parameter_identity": "parameter_identity" in str(error),
                }
            else:
                raise AssertionError("frozen ROUND40 resume entry did not reproduce its failure")
            if old_failure is None or not old_failure["contains_parameter_identity"]:
                raise AssertionError("old resume entry failed for a different reason")

            adapter._save_progress = lambda *_args, **_kwargs: None
            adapter._save_preupdate_checkpoint = no_preupdate_file
            adapter._write_immutable_snapshot = no_checkpoint_file
            adapter._write_batch_commit = no_checkpoint_file
            adapter._save_milestone = no_checkpoint_file
            try:
                adapter.train_seed(11, episodes=8,
                                   freeze=frozen["round42"]["freeze_identity"],
                                   reference_cache=reference_cache)
            except RuntimeError as error:
                fixed_guard_error = str(error)
            else:
                raise AssertionError("patched resume did not reach the guarded next episode")
            if fixed_guard_error != (
                    "ROUND43 recovery test blocked before environment construction"):
                raise AssertionError(f"patched resume stopped before the intended guard: "
                                     f"{fixed_guard_error}")
            if environment_entries != 1:
                raise AssertionError("patched resume did not reach exactly the next episode entry")
            if list(temp_checkpoints.glob("*.pt")) != sorted(
                    [temp_checkpoints / "v2_train_s11.pt",
                     temp_checkpoints / "v2_train_s11_after_b0.pt"]):
                raise AssertionError("isolated resume probe wrote a checkpoint file")
            fixed_cursor = {
                "resume_source": "isolated byte-for-byte copy of immutable after_b0/rolling state",
                "state_cursor": "episode 4, update 1, Adam state step 12, lambda 1.0125",
                "next_episode_index": 4,
                "entry_reached": "run_training_episode guard before any environment constructor",
            }
    finally:
        trainer.restore_rng(saved_rng)
        common.CHECKPOINTS, common.OUTPUT_DIR, common.LOGS = (
            old_common_checkpoints, old_common_output, old_common_logs)
        r40_run.run_training_episode = old_run_entry
        trainer.save_checkpoint = old_trainer_save
        trainer.ppo_update = old_trainer_ppo
        torch.optim.Adam.step = old_adam_step
        for name, value in old_module_save.items():
            setattr(adapter, name, value)
        adapter.common, adapter.runner, adapter.trainer = (
            old_adapter_common, old_adapter_runner, old_adapter_trainer)
        if 'original_snapshot' in locals():
            for name, value in original_snapshot.items():
                if value is not None:
                    setattr(original, name, value)

    if _deep_hash(trainer.rng_state()) != saved_rng_hash:
        raise AssertionError("isolated resume regression did not restore ambient RNG")
    if (optimizer_calls, ppo_calls, save_calls, scratch_checkpoint_writes) != (0, 0, 0, 0):
        raise AssertionError("isolated resume regression crossed a forbidden update/write guard")
    return {
        "kind": "round43-actual-resume-entry-regression/1",
        "status": "passed",
        "frozen_round40_source_path": str(original_path),
        "frozen_round40_source_sha256": _sha256(original_path),
        "old_entry_actual_resume_failure": old_failure,
        "patched_adapter_path": str(Path(adapter.__file__)),
        "patched_adapter_sha256": _sha256(Path(adapter.__file__)),
        "patched_entry": fixed_cursor,
        "guard_calls": {
            "next_episode_entry_before_environment": environment_entries,
            "real_or_synthetic_optimizer_updates": optimizer_calls,
            "ppo_updates": ppo_calls,
            "actual_checkpoint_saves": save_calls,
            "isolated_checkpoint_writes": scratch_checkpoint_writes,
        },
        "temporary_isolation_inputs": {
            "checkpoint_state_files_copied_byte_for_byte": 2,
            "batch_commit_and_record_files_copied_byte_for_byte": 2,
            "checkpoint_serialization_writes": 0,
        },
        "ambient_rng_restored": True,
    }


def _failure_record(frozen: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    previous = _read(OLD_RECEIPT_COPY)
    return {
        "kind": "round43-runner-resume-failure/1",
        "status": "preserved_harness_failure_before_next_environment_claim",
        "execution_head": previous["head"],
        "branch": previous["branch"],
        "source_code_version": "ROUND43 initial source freeze; full-trace semantics unchanged",
        "call_path": [
            "r43_execute.run seed11 ep128 continuation",
            "r42_run.train_seed resume=True",
            "ROUND40 r40_train.train_seed resume_source branch",
            "before = parameter_identity(controller)",
        ],
        "exception": {
            "type": "UnboundLocalError",
            "message": "local variable 'parameter_identity' referenced before assignment",
            "source_location": {
                "path": frozen["execution_identity"]["round40_worktree"]
                        + "\\runs\\comm_first_report_training_ablation_20261003\\r40_train.py",
                "line": 386,
            },
            "reason": "parameter_identity is imported only inside the fresh-start branch, while the resume branch reaches its use without executing that local import",
        },
        "pre_failure_cursor": {
            "seed": 11, "episodes_done": 4, "updates": 1,
            "optimizer_steps_committed": 12, "lambda_dual": 1.0125,
        },
        "accounting_at_stop": {
            "environment_claims_delta_during_failed_resume": 0,
            "environment_constructors_delta": 0,
            "environment_run_calls_delta": 0,
            "PPO_or_optimizer_call_delta": 0,
            "normal_slots_used_before_failure": ledger["counts"]["normal_used"],
            "technical_reserve_used_before_failure": ledger["counts"]["reserve_used"],
        },
        "preservation": {
            "original_ledger_copy": str(OLD_LEDGER_COPY),
            "original_progress_copy": str(OLD_PROGRESS_COPY),
            "original_prelaunch_identity_copy": str(OLD_RECEIPT_COPY),
            "original_preflight_copy": str(OLD_PREFLIGHT_COPY),
            "batch0_commit_and_checkpoint_unchanged": True,
        },
    }


def main() -> int:
    frozen = _read(FREEZE_PATH)
    round40_freeze_path = (Path(frozen["execution_identity"]["round40_worktree"]) / "runs"
                           / "comm_first_report_training_ablation_20261003"
                           / "ROUND40_FROZEN_CONFIG.json")
    round40_freeze = _read(round40_freeze_path)["round40"]["freeze_identity"]
    runtime = r42_run.configure(frozen, round40_freeze)
    before_artifacts = _snapshot_artifacts()
    ambient = runtime["trainer"].rng_state()
    ambient_hash = _deep_hash(ambient)

    state_audit = _audit_state(runtime, frozen)
    old_ledger = _read(OLD_LEDGER_COPY)
    actual_ledger = r42_run.ledger_snapshot(strict=True)
    journal_dir = RUNTIME_DIR
    ppo_entries = [json.loads(line) for line in (
        journal_dir / "ppo_update_calls.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    optimizer_entries = [json.loads(line) for line in (
        journal_dir / "optimizer_step_calls.jsonl").read_text(encoding="utf-8").splitlines()
                         if line.strip()]
    checkpoint_entries = [json.loads(line) for line in (
        journal_dir / "checkpoint_writes.jsonl").read_text(encoding="utf-8").splitlines()
                          if line.strip()]
    batch_files = sorted(CHECKPOINT_DIR.glob("r40_batch_s*_b*_attempt*.json"))
    if len(batch_files) != 1:
        raise AssertionError(f"expected only committed seed11 batch0; found {len(batch_files)}")
    if len(ppo_entries) != 1 or ppo_entries[0].get("update_index") != 0:
        raise AssertionError("PPO journal does not close against the one committed batch")
    if len(optimizer_entries) != 12 \
            or [entry["call_index"] for entry in optimizer_entries] != list(range(1, 13)):
        raise AssertionError("Adam invocation journal does not show the 12 batch-0 calls")
    if len(checkpoint_entries) != 4:
        raise AssertionError("checkpoint save-call journal does not contain four completed calls")
    if actual_ledger["counts"]["optimizer_step_calls"] != 12 \
            or actual_ledger["counts"]["ppo_update_calls"] != 1 \
            or actual_ledger["counts"]["checkpoint_write_calls"] != 4:
        raise AssertionError("reconciled invocation counters differ from durable journals")
    if old_ledger["counts"]["optimizer_step_calls"] != 0 \
            or old_ledger["counts"]["ppo_update_calls"] != 0 \
            or old_ledger["counts"]["checkpoint_write_calls"] != 2:
        raise AssertionError("preserved pre-update ledger does not match the observed stale snapshot")

    counter_regression = _run_tests("test_r43_call_counter_hooks")
    adapter_ast_regression = _run_tests("test_r43_resume_adapter")
    resume_regression = _probe_old_and_fixed_resume(runtime, frozen)
    if _deep_hash(runtime["trainer"].rng_state()) != ambient_hash:
        raise AssertionError("recovery gate changed process RNG state")
    after_artifacts = _snapshot_artifacts()
    if after_artifacts != before_artifacts:
        changed = sorted(set(after_artifacts) ^ set(before_artifacts))
        changed.extend(name for name in before_artifacts.keys() & after_artifacts.keys()
                       if before_artifacts[name] != after_artifacts[name])
        raise AssertionError(f"recovery gate changed existing raw artifacts: {changed}")

    _write_immutable(STATE_AUDIT_PATH, state_audit)
    failure = _failure_record(frozen, actual_ledger)
    _write_immutable(FAILURE_PATH, failure)
    state_audit_identity = {"path": STATE_AUDIT_PATH.name,
                            "sha256": _sha256(STATE_AUDIT_PATH)}
    ledger_audit = {
        "kind": "round43-ledger-reconciliation/1",
        "status": "passed",
        "original_persisted_ledger": {
            "path": str(OLD_LEDGER_COPY),
            "sha256": _sha256(OLD_LEDGER_COPY),
            "counts": old_ledger["counts"],
            "interpretation": "pre-batch-update snapshot; retains zero PPO/Adam counts as originally written",
        },
        "corrected_cumulative_snapshot": actual_ledger["counts"],
        "evidence": {
            "ppo_update_call_entries": ppo_entries,
            "optimizer_step_call_entry_count": len(optimizer_entries),
            "optimizer_step_call_indices": [entry["call_index"] for entry in optimizer_entries],
            "checkpoint_save_call_entries": checkpoint_entries,
            "committed_batch_records": [str(path) for path in batch_files],
            "batch_record_optimizer_steps": 12,
            "after_b0_optimizer_state_steps": state_audit["optimizer"][
                "step_values_by_parameter"],
        },
        "scientific_vs_accounting": {
            "scientific_configuration_changed": False,
            "training_update_replayed": False,
            "main_ledger_reconciled_from_durable_call_journals_and_committed_batch": True,
            "checkpoint_save_calls_source": "checkpoint_writes.jsonl invocation wrapper records, not checkpoint file count",
        },
    }
    _write_immutable(LEDGER_AUDIT_PATH, ledger_audit)

    frozen_source_files = {
        name: _sha256(RUN_DIR / name)
        for name in (
            "r42_run.py", "r42_versions.py", "r42_one_shot.py", "r42_analysis.py",
            "r43_preflight.py", "r43_analysis.py", "r43_execute.py", "r43_report.py",
            "r43_freeze.py", "r40_train_resume_adapter.py", "r43_call_counter_hooks.py",
            "test_r43_call_counter_hooks.py", "test_r43_resume_adapter.py",
            "r43_recovery_gate.py",
        )
    }
    gate = {
        "kind": "round43-resume-recovery-gate/1",
        "status": "passed",
        "prior_execution_head": _read(OLD_RECEIPT_COPY)["head"],
        "source_code_sha256_for_next_freeze": frozen_source_files,
        "training_semantics_version": frozen["round42"]["training_version"],
        "mechanical_resume_adapter": "ROUND42-validated import-scope adapter; only parameter_identity import moved to function entry",
        "batch0_state_audit": state_audit_identity,
        "ledger_reconciliation": {"path": LEDGER_AUDIT_PATH.name,
                                   "sha256": _sha256(LEDGER_AUDIT_PATH)},
        "resume_failure_record": {"path": FAILURE_PATH.relative_to(RUN_DIR).as_posix(),
                                  "sha256": _sha256(FAILURE_PATH)},
        "actual_entry_regression": resume_regression,
        "adapter_ast_regression": adapter_ast_regression,
        "training_call_counter_boundary_regression": counter_regression,
        "state_sources_unchanged_during_gate": True,
        "ambient_rng_restored": True,
        "environment_constructors": 0,
        "joint_run_constructors": 0,
        "episode_claims_created": 0,
        "real_optimizer_updates": 0,
        "synthetic_optimizer_updates": 0,
        "checkpoint_serialization_writes": 0,
    }
    _write_immutable(GATE_PATH, gate)
    print(json.dumps({
        "status": "passed",
        "old_resume": resume_regression["old_entry_actual_resume_failure"],
        "fixed_resume_guard": resume_regression["patched_entry"],
        "state_cursor": state_audit["state_cursor"],
        "adam_state_steps": state_audit["optimizer"]["step_values_by_parameter"],
        "pre-update_ledger_counts": {
            "ppo": old_ledger["counts"]["ppo_update_calls"],
            "adam": old_ledger["counts"]["optimizer_step_calls"],
        },
        "reconciled_counts": {
            "ppo": actual_ledger["counts"]["ppo_update_calls"],
            "adam": actual_ledger["counts"]["optimizer_step_calls"],
            "save_calls": actual_ledger["counts"]["checkpoint_write_calls"],
        },
        "environment_constructors": 0,
        "optimizer_updates": 0,
        "gate_path": str(GATE_PATH),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
