"""Zero-environment comparator and exact-resume gates for the fixed ROUND44 continuation."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from r44_prepare import RUN_DIR, TRAINING_VERSION, read, sha256, write_new
from r44_resume_bridge import (
    COMPARATOR_TEST_PATH, EXPECTED_COMPARATOR_REJECTIONS, GATE_PATH, IDENTITY_PATH,
    MECHANICAL_VERSION, OLD_PROJECTION_PATH, R1_IDENTITY_PATH, R2_FAILURE_PATH,
    R2_IDENTITY_PATH, READY_PATH, SOURCE_CONFIG_RECONCILIATION_PATH,
    STATE_BRIDGE_VERSION, WHITELIST_PATH, build_state_whitelist,
    seal_resume_execution_identity, verify_pre_resume_state, verify_resume_execution_identity,
    verify_state_whitelist,
)


def _hash(value: Any, state_hash: Callable[[Any], str]) -> str:
    return state_hash(value)


def _counter_snapshot(runtime: dict[str, Any]) -> dict[str, int]:
    run = runtime["r42_run"]
    versions = runtime["r42_versions"]
    return {
        "optimizer_step_calls": int(run._OPTIMIZER_STEP_CALLS),
        "ppo_update_calls": int(run._PPO_UPDATE_CALLS),
        "checkpoint_write_calls": int(versions.CHECKPOINT_WRITE_CALLS),
    }


def _ambient_rng(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }


def _ambient_rng_hash(runtime: dict[str, Any]) -> str:
    return runtime["r40_train"]._state_hash(runtime["trainer"].rng_state())


def _restore_ambient_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def _expect_comparator_rejection(label: str, call: Callable[[], Any], expected_fragment: str) -> str:
    try:
        call()
    except RuntimeError as exc:
        message = str(exc)
        if expected_fragment not in message:
            raise RuntimeError(f"{label}: rejected for an unexpected reason: {message}") from exc
        return message
    raise RuntimeError(f"{label}: comparator accepted a deliberately altered in-memory record")


def _is_expected_old_training_checkpoint_rejection(message: str) -> bool:
    """Accept only the registered version/freeze rejection spellings."""
    return any(token in message for token in (
        "training_version mismatch", "training version mismatch", "freeze identity mismatch",
    ))


def _run_comparator_regression(runtime: dict[str, Any], gate: dict[str, Any],
                               real_comparison: dict[str, Any]) -> dict[str, Any]:
    import r44_execute

    seed = 11
    run = runtime["r42_run"]
    state_hash = runtime["r40_train"]._state_hash
    commit = read(runtime["common"].CHECKPOINTS / "r40_batch_commit_s11_b0.json")
    record_path = runtime["common"].CHECKPOINTS / str(commit["batch_record"])
    post_path = runtime["common"].CHECKPOINTS / str(commit["postupdate_checkpoint"])
    artifacts = read(record_path)["episode_artifacts"]
    protected_artifacts: dict[str, str] = {}
    for item in artifacts:
        old_row = read(Path(item["row_path"]))
        new_row_path = RUN_DIR / "rows" / Path(item["row_path"]).name.replace("v2_r43_", "v2_r44_")
        old_paths = old_row["artefacts"]
        new_row = read(new_row_path)
        for row in (old_row, new_row):
            for artifact_path in row["artefacts"].values():
                protected_artifacts[str(artifact_path)] = sha256(Path(artifact_path))
        for path in (Path(item["row_path"]), new_row_path,
                     Path(item["tape_path"]), RUN_DIR / "logs" / Path(item["tape_path"]).name.replace("v2_r43_", "v2_r44_")):
            protected_artifacts[str(path)] = sha256(path)
    checkpoint_paths = (
        runtime["common"].CHECKPOINTS / "v2_train_s11_after_b0.pt",
        runtime["common"].CHECKPOINTS / "v2_train_s11.pt",
    )
    checkpoint_hashes_before = {str(path): sha256(path) for path in checkpoint_paths}
    protected_runtime = (
        RUN_DIR / "runtime" / "ROUND44_EXECUTION_LEDGER.json",
        RUN_DIR / "runtime" / "ROUND44_EXECUTION_PROGRESS.json",
        RUN_DIR / "runtime" / "ROUND44_PRELAUNCH_IDENTITY.json",
        RUN_DIR / "runtime" / "ROUND44_RESUME_EXECUTION_IDENTITY_R1.json",
        RUN_DIR / "runtime" / "ROUND44_OLD_COMPARATOR_REPRODUCTION.json",
        RUN_DIR / "ROUND44_STATUS.json",
        RUN_DIR / "ROUND44_STOP_REPORT.md",
        RUN_DIR / "ROUND44_STOP_STATE.json",
        RUN_DIR / "ROUND44_STOP_ARTIFACT_INVENTORY.json",
        RUN_DIR / "ROUND44_EXECUTION_FAILURE_CAPTURE_SEED11_B0.txt",
    )
    protected_runtime_hashes = {str(path): sha256(path) for path in protected_runtime}
    original_ledger_hash = sha256(RUN_DIR / "runtime" / "ROUND44_EXECUTION_LEDGER.json")
    original_progress_hash = sha256(RUN_DIR / "runtime" / "ROUND44_EXECUTION_PROGRESS.json")
    counters_before = _counter_snapshot(runtime)
    ambient_before = _ambient_rng_hash(runtime)
    original_read = r44_execute.read
    original_loader = run.torch_load_cpu
    original_jsonl = r44_execute._read_jsonl
    original_writer = r44_execute._write_new_or_match
    original_sha256 = r44_execute.sha256
    rejections: dict[str, str] = {}
    new_episode = artifacts[0]
    r43_row_path = Path(new_episode["row_path"]).resolve()
    r44_row_path = (RUN_DIR / "rows" / r43_row_path.name.replace("v2_r43_", "v2_r44_")).resolve()
    r43_tape_path = Path(new_episode["tape_path"]).resolve()
    r44_tape_path = (RUN_DIR / "logs" / r43_tape_path.name.replace("v2_r43_", "v2_r44_")).resolve()
    old_row_data = original_read(r43_row_path)
    new_row_data = original_read(r44_row_path)
    r44_lifecycle_path = Path(new_row_data["artefacts"]["lifecycle"]).resolve()

    def reset_hooks() -> None:
        r44_execute.read = original_read
        r44_execute._read_jsonl = original_jsonl
        r44_execute._write_new_or_match = original_writer
        r44_execute.sha256 = original_sha256
        run.torch_load_cpu = original_loader

    def expect_rejection(label: str, expected_fragment: str, *,
                         read_hook: Callable[[Path], Any] | None = None,
                         jsonl_hook: Callable[[Path], Any] | None = None,
                         loader_hook: Callable[[Path], Any] | None = None,
                         sha_hook: Callable[[Path], str] | None = None) -> None:
        r44_execute.read = read_hook or original_read
        r44_execute._read_jsonl = jsonl_hook or original_jsonl
        run.torch_load_cpu = loader_hook or original_loader
        r44_execute.sha256 = sha_hook or original_sha256
        try:
            rejections[label] = _expect_comparator_rejection(
                label, lambda: r44_execute._compare_first_batch(seed, runtime, gate),
                expected_fragment)
        finally:
            reset_hooks()

    def altered_record(mutator: Callable[[dict[str, Any]], None]) -> Callable[[Path], Any]:
        def patched(path: Path) -> Any:
            value = original_read(path)
            if Path(path).resolve() == record_path.resolve():
                value = copy.deepcopy(value)
                mutator(value)
            return value
        return patched

    try:
        expect_rejection(
            "parameter", "actor/optimizer parameter path differs",
            read_hook=altered_record(
                lambda item: item["controller_parameters_after"].__setitem__(
                    "parameter_sha256", "0" * 64)))
        expect_rejection(
            "dual", "dual update did not isolate eta",
            read_hook=altered_record(
                lambda item: item["dual"].__setitem__("lambda_next", 9.0)))

        def altered_optimizer(path: Path) -> Any:
            payload = original_loader(path)
            if Path(path).resolve() == post_path.resolve():
                payload = copy.deepcopy(payload)
                payload["optimizer"]["param_groups"][0]["lr"] += 1e-7
            return payload

        expect_rejection("optimizer", "batch-zero optimizer state differs",
                         loader_hook=altered_optimizer)

        signature_module = importlib.import_module("integrated_sar.run_signature")
        lifecycle_records = original_jsonl(r44_lifecycle_path)
        modified_config = copy.deepcopy(lifecycle_records[0]["detail"]["configuration"])
        modified_config["comm_range"] = float(modified_config["comm_range"]) + 0.01
        modified_hash = signature_module.config_hash(modified_config)

        def changed_config_row(path: Path) -> Any:
            value = original_read(path)
            if Path(path).resolve() == r44_row_path:
                value = copy.deepcopy(value)
                value["config_hash"] = modified_hash
            return value

        def changed_config_lifecycle(path: Path) -> Any:
            value = original_jsonl(path)
            if Path(path).resolve() == r44_lifecycle_path:
                value = copy.deepcopy(value)
                value[0]["detail"]["configuration"]["comm_range"] = modified_config["comm_range"]
                value[0]["detail"]["run_signature"]["config_hash"] = modified_hash
            return value

        expect_rejection(
            "configuration", "run configuration has unregistered differences",
            read_hook=changed_config_row, jsonl_hook=changed_config_lifecycle)

        def changed_artifact_path(path: Path) -> Any:
            value = original_read(path)
            if Path(path).resolve() == r44_row_path:
                value = copy.deepcopy(value)
                value["artefacts"]["transcript"] = value["artefacts"]["lifecycle"]
            return value

        expect_rejection(
            "artifact_path", "registered R43/R44 label pair", read_hook=changed_artifact_path)

        r44_transcript_path = Path(new_row_data["artefacts"]["transcript"]).resolve()

        def changed_transcript_hash(path: Path) -> str:
            if Path(path).resolve() == r44_transcript_path:
                return "0" * 64
            return original_sha256(path)

        expect_rejection(
            "transcript", "R43/R44 transcript content differs",
            sha_hook=changed_transcript_hash)

        # A timing-only in-memory perturbation must pass through the actual comparator entry.
        def changed_wallclock(path: Path) -> Any:
            value = original_read(path)
            if Path(path).resolve() == r44_row_path:
                value = copy.deepcopy(value)
                value["seconds"] += 100.0
                value["collector_summary"]["timing"]["decide_milliseconds_per_call"] += 100.0
            return value

        r44_execute.read = changed_wallclock
        r44_execute._write_new_or_match = lambda _path, _value: None
        positive_timing = r44_execute._compare_first_batch(seed, runtime, gate)
        if positive_timing.get("status") != "passed":
            raise RuntimeError("timing-only row perturbation was not accepted by the new comparator")
        reset_hooks()

        def changed_row(mutator: Callable[[dict[str, Any]], None]) -> Callable[[Path], Any]:
            def patched(path: Path) -> Any:
                value = original_read(path)
                if Path(path).resolve() == r44_row_path:
                    value = copy.deepcopy(value)
                    mutator(value)
                return value
            return patched

        row_negatives = (
            ("on_time", lambda row: row.__setitem__("on_time", int(row["on_time"]) + 1)),
            ("task_set", lambda row: row["task_reward_events"][0].__setitem__("task_id", "tampered-task")),
            ("cost", lambda row: row.__setitem__("spent_bytes", int(row["spent_bytes"]) + 1)),
            ("task_reward", lambda row: row["task_stream"].__setitem__(
                next(iter(row["task_stream"])), float(row["task_stream"][next(iter(row["task_stream"]))]) + 1.0)),
            ("budget_mask", lambda row: row["collector_summary"]["timing"].__setitem__(
                "masked_by_credit", int(row["collector_summary"]["timing"]["masked_by_credit"]) + 1)),
            ("candidate_counter", lambda row: row["collector_summary"]["timing"][
                "first_report_only_counts"].__setitem__(
                    "first_report_candidates",
                    int(row["collector_summary"]["timing"]["first_report_only_counts"][
                        "first_report_candidates"]) + 1)),
            ("decision_counter", lambda row: row["collector_summary"]["timing"].__setitem__(
                "decisions", int(row["collector_summary"]["timing"]["decisions"]) + 1)),
        )
        for label, mutator in row_negatives:
            expect_rejection(label, "pre-update rollout outcomes changed",
                             read_hook=changed_row(mutator))

        def changed_tape(mutator: Callable[[dict[str, Any]], None]) -> Callable[[Path], Any]:
            def patched(path: Path) -> Any:
                value = original_jsonl(path)
                if Path(path).resolve() == r44_tape_path:
                    value = copy.deepcopy(value)
                    mutator(value[0])
                return value
            return patched

        def change_action(tape: dict[str, Any]) -> None:
            target = next(iter(tape["actions"]))
            tape["actions"][target] = 0 if int(tape["actions"][target]) else 1

        def change_probability(tape: dict[str, Any]) -> None:
            target = next(iter(tape["send_probabilities"]))
            before = float(tape["send_probabilities"][target])
            tape["send_probabilities"][target] = before + 0.11 if before < 0.88 else before - 0.11

        expect_rejection("action", "action/features/log-prob tape differs",
                         jsonl_hook=changed_tape(change_action))
        expect_rejection("probability", "action/features/log-prob tape differs",
                         jsonl_hook=changed_tape(change_probability))
        expect_rejection("likelihood", "action/features/log-prob tape differs",
                         jsonl_hook=changed_tape(
                             lambda tape: tape.__setitem__("log_prob", float(tape["log_prob"]) + 0.1)))
    finally:
        reset_hooks()

    if _counter_snapshot(runtime) != counters_before:
        raise RuntimeError("ROUND44 comparator regression changed cumulative update/write counters")
    if _ambient_rng_hash(runtime) != ambient_before:
        raise RuntimeError("ROUND44 comparator regression changed ambient RNG")
    for path, expected in checkpoint_hashes_before.items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"ROUND44 comparator regression changed a checkpoint: {path}")
    for path, expected in protected_artifacts.items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"ROUND44 comparator regression changed a saved run artifact: {path}")
    for path, expected in protected_runtime_hashes.items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"ROUND44 comparator regression changed an original stop/R1 file: {path}")
    if (sha256(RUN_DIR / "runtime" / "ROUND44_EXECUTION_LEDGER.json") != original_ledger_hash
            or sha256(RUN_DIR / "runtime" / "ROUND44_EXECUTION_PROGRESS.json")
            != original_progress_hash):
        raise RuntimeError("ROUND44 comparator regression changed original ledger/progress")
    actual_episode_metadata = real_comparison.get("episodes", [])
    if (len(actual_episode_metadata) != 4
            or any(item.get("metadata_reconciliation", {}).get(
                "run_configuration_differing_fields") != ["comm_transcript", "output"]
                   for item in actual_episode_metadata)
            or any(item.get("metadata_reconciliation", {}).get(
                "registered_row_label_change", [None, None])[0]
                   == item.get("metadata_reconciliation", {}).get(
                       "registered_row_label_change", [None, None])[1]
                   for item in actual_episode_metadata)):
        raise RuntimeError("ROUND44 real comparator did not accept only the registered label/output changes")
    if positive_timing.get("projection_version") != "round44-row-projection/2":
        raise RuntimeError("ROUND44 wall-clock positive did not use the R2 projection")
    result = {
        "kind": "round44-comparator-regression/2", "status": "passed",
        "tested_entry": "r44_execute._compare_first_batch(seed=11) with committed batch0",
        "projection_version": "round44-row-projection/2",
        "expected_rejections": [
            "parameter", "dual", "optimizer", "configuration", "artifact_path", "transcript",
            "on_time", "task_set", "cost", "task_reward", "budget_mask",
            "candidate_counter", "decision_counter", "action", "probability", "likelihood",
        ],
        "rejection_messages": rejections,
        "accepted_wallclock_timing_perturbation": {
            "status": "passed", "seconds_delta": 100.0,
            "decide_milliseconds_per_call_delta": 100.0,
            "writer_suppressed_only_for_synthetic_in_memory_call": True,
        },
        "accepted_real_registered_row_labels_and_output_paths": [
            item["metadata_reconciliation"] for item in actual_episode_metadata
        ],
        "counters_before": counters_before, "counters_after": _counter_snapshot(runtime),
        "ambient_rng_hash_before": ambient_before,
        "ambient_rng_hash_after": _ambient_rng_hash(runtime),
        "immutable_row_tape_lifecycle_transcript_checkpoint_stop_r1_hashes_unchanged": True,
        "environment_starts": 0, "optimizer_steps": 0, "checkpoint_writes": 0,
    }
    write_new(COMPARATOR_TEST_PATH, result)
    return result


def _next_values_from_saved_rng(rng: dict[str, Any]) -> dict[str, Any]:
    python_state = rng["python"]
    py_generator = random.Random()
    py_generator.setstate((int(python_state[0]), tuple(python_state[1]), python_state[2]))

    numpy_state = rng["numpy"]
    np_generator = np.random.RandomState()
    np_generator.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                            int(numpy_state[2]), int(numpy_state[3]),
                            float(numpy_state[4])))

    torch_generator = torch.Generator(device="cpu")
    torch_generator.set_state(torch.tensor(rng["torch"], dtype=torch.uint8))
    return {
        "python_random": py_generator.random(),
        "numpy_random": float(np_generator.random_sample()),
        "torch_rand_cpu": torch.rand(3, generator=torch_generator).tolist(),
    }


def _exact_restore_gate(runtime: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    run = runtime["r42_run"]
    train = runtime["r40_train"]
    trainer = runtime["trainer"]
    whitelist = verify_state_whitelist(check_live_mutables=True)
    permitted = whitelist["permitted_resume"]
    checkpoint = RUN_DIR / "checkpoints" / "v2_train_s11_after_b0.pt"
    if sha256(checkpoint) != permitted["postupdate_checkpoint_sha256"]:
        raise RuntimeError("ROUND44 after-b0 checkpoint no longer matches the exact resume whitelist")

    payload = run.torch_load_cpu(checkpoint)
    expected_state = payload.get("training_state")
    if not isinstance(expected_state, dict):
        raise RuntimeError("ROUND44 after-b0 checkpoint lacks complete training state")
    if (int(expected_state.get("training_seed", -1)) != 11
            or int(expected_state.get("episodes_done", -1)) != 4
            or int(expected_state.get("updates", -1)) != 1
            or int(expected_state.get("optimizer_steps", -1)) != 12
            or float(expected_state.get("lambda", float("nan"))) != 1.125
            or float(expected_state.get("eta_lambda", float("nan"))) != 0.5
            or expected_state.get("training_version") != TRAINING_VERSION
            or expected_state.get("comm_action_semantics") != "one-shot-send-skip/1"
            or not isinstance(expected_state.get("lambda_history"), list)):
        raise RuntimeError("ROUND44 checkpoint cursor/version/dual state is not the exact ep4 state")
    saved_rng = payload.get("rng")
    if not isinstance(saved_rng, dict) or set(saved_rng) != {"python", "numpy", "torch"}:
        raise RuntimeError("ROUND44 checkpoint RNG sources are incomplete")

    ambient = _ambient_rng(runtime)
    ambient_hash_before = _ambient_rng_hash(runtime)
    counters_before = _counter_snapshot(runtime)
    restore_identity: dict[str, Any]
    module_before: dict[str, str] = {}
    try:
        read_only_controller, _read_only_state, read_only_identity = \
            trainer.load_checkpoint_read_only(checkpoint, freeze=runtime["round42_freeze"])
        if (read_only_identity.get("optimizer_restored") is not False
                or read_only_identity.get("rng_restored") is not False):
            raise RuntimeError("ROUND44 state precheck was not a read-only checkpoint load")
        module_before = train._module_digests(read_only_controller)
        controller, optimizer, state = trainer.load_checkpoint(
            checkpoint, freeze=runtime["round42_freeze"])
        parameter_identity = controller.parameter_summary()["parameter_sha256"]
        module_after = train._module_digests(controller)
        record = read(runtime["common"].CHECKPOINTS / "r40_batch_s11_b0_attempt01.json")
        if (parameter_identity != permitted["parameter_sha256"]
                or module_after != record["controller_module_digests_after"]):
            raise RuntimeError("ROUND44 restored parameter/module digest differs from committed batch0")
        if controller.parameters[0].device.type != "cpu":
            raise RuntimeError("ROUND44 exact resume gate constructed a non-CPU controller")
        module_state_key_map = {
            "trunk": controller.inner.trunk,
            "action_head": controller.inner.action_head,
            "value_task_head": controller.inner.value_head,
            "value_cost_head": controller.cost_head,
        }
        module_state_equal = {
            saved_name: train._state_hash(module.state_dict())
            == train._state_hash(payload["controller"][saved_name])
            for saved_name, module in module_state_key_map.items()
        }
        if not all(module_state_equal.values()):
            raise RuntimeError(f"ROUND44 module restoration mismatch: {module_state_equal}")
        optimizer_state = optimizer.state_dict()
        optimizer_hash = train._state_hash(optimizer_state)
        expected_optimizer_hash = train._state_hash(payload["optimizer"])
        optimizer_parameter_ids = [id(value) for group in optimizer.param_groups
                                   for value in group["params"]]
        controller_parameter_ids = [id(value) for value in controller.parameters]
        module_parameter_ids = []
        for module in module_state_key_map.values():
            module_parameter_ids.extend(id(value) for value in module.parameters())
        if (optimizer_parameter_ids != controller_parameter_ids
                or module_parameter_ids != controller_parameter_ids):
            raise RuntimeError("ROUND44 restored Adam is not bound to exactly the four controller modules")
        if (optimizer_hash != expected_optimizer_hash
                or len(optimizer_state.get("state", {})) != len(controller.parameters)):
            raise RuntimeError("ROUND44 restored Adam moments/state differ from the saved checkpoint")
        step_values = []
        for entry in optimizer_state["state"].values():
            step = entry.get("step")
            step_values.append(float(step.item()) if isinstance(step, torch.Tensor) else float(step))
            if not all(key in entry for key in ("step", "exp_avg", "exp_avg_sq")):
                raise RuntimeError("ROUND44 restored Adam state is missing moments or step")
        if _hash(state, train._state_hash) != _hash(expected_state, train._state_hash):
            raise RuntimeError("ROUND44 restored training cursor or lambda history differs")
        restored_rng = trainer.rng_state()
        if _hash(restored_rng, train._state_hash) != _hash(saved_rng, train._state_hash):
            raise RuntimeError("ROUND44 actual Python/NumPy/Torch RNG restore differs from checkpoint")

        predicted_next = _next_values_from_saved_rng(saved_rng)
        actual_next = {
            "python_random": random.random(),
            "numpy_random": float(np.random.random_sample()),
            "torch_rand_cpu": torch.rand(3).tolist(),
        }
        if predicted_next != actual_next:
            raise RuntimeError("ROUND44 restored RNG next values do not match the saved streams")
        order = [int(value) for value in runtime["common"].scenario_order()]
        expected_order = [int(value) for value in frozen["round42"]["training_order"]]
        if len(order) != 128 or order != expected_order:
            raise RuntimeError("ROUND44 resumed training scene schedule differs from frozen 128 order")
        if int(order[4]) != 1300014:
            raise RuntimeError("ROUND44 next scene at episode4 differs from the registered schedule")
        restore_identity = {
            "training_seed": int(state["training_seed"]),
            "episodes_done": int(state["episodes_done"]),
            "updates": int(state["updates"]),
            "optimizer_steps": int(state["optimizer_steps"]),
            "lambda": float(state["lambda"]),
            "lambda_history": state["lambda_history"],
            "parameter_sha256": parameter_identity,
            "module_parameter_sha256": module_after,
            "module_state_equal_to_checkpoint": module_state_equal,
            "optimizer_state_sha256": optimizer_hash,
            "optimizer_parameter_count": len(controller.parameters),
            "optimizer_state_entries": len(optimizer_state["state"]),
            "adam_step_values": sorted(set(step_values)),
            "python_numpy_torch_rng_state_sha256": train._state_hash(restored_rng),
            "rng_next_values_predicted_from_checkpoint": predicted_next,
            "rng_next_values_after_actual_restore": actual_next,
            "scenario_order_sha256": hashlib.sha256(
                json.dumps(order, separators=(",", ":")).encode("ascii")).hexdigest(),
            "next_episode_index": 4,
            "next_scenario_seed": int(order[4]),
        }
    finally:
        _restore_ambient_rng(ambient)

    counters_after = _counter_snapshot(runtime)
    ambient_hash_after = _ambient_rng_hash(runtime)
    if counters_after != counters_before:
        raise RuntimeError("ROUND44 exact restore gate changed optimizer/PPO/checkpoint counters")
    if ambient_hash_after != ambient_hash_before:
        raise RuntimeError("ROUND44 exact restore gate failed to restore its own ambient RNG")
    verify_state_whitelist(check_live_mutables=True)
    if module_before != module_after:
        raise RuntimeError("ROUND44 read-only precheck module hashes disagree with actual restore")

    return {
        "kind": "round44-exact-resume-gate/3", "status": "passed",
        "training_seed": 11, "episodes_done": 4, "updates": 1,
        "optimizer_steps_saved": 12, "lambda": 1.125,
        "eta_lambda": 0.5, "training_version": TRAINING_VERSION,
        "state_bridge_version": STATE_BRIDGE_VERSION,
        "mechanical_version": MECHANICAL_VERSION,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "registered_parameter_sha256": permitted["parameter_sha256"],
        "resume_state": restore_identity,
        "counters_before": counters_before, "counters_after": counters_after,
        "ambient_rng_hash_before": ambient_hash_before,
        "ambient_rng_hash_after": ambient_hash_after,
        "environment_starts": 0, "run_calls": 0,
        "optimizer_steps": 0, "checkpoint_writes": 0,
        "batch0_replayed": False,
        "checkpoint_written_back": False,
    }


def run_read_only_gates() -> dict[str, Any]:
    import r44_execute
    from r44_runtime import configure

    frozen = read(RUN_DIR / "ROUND44_FROZEN_CONFIG.json")
    head = subprocess.run(["git", "-C", str(RUN_DIR.parents[1]), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    verify_resume_execution_identity(frozen, head)
    verify_state_whitelist(check_live_mutables=True)
    runtime = configure(frozen)
    ambient = _ambient_rng(runtime)
    ambient_hash_before = _ambient_rng_hash(runtime)
    counters_before = _counter_snapshot(runtime)
    ledger_hash_before = sha256(RUN_DIR / "runtime" / "ROUND44_EXECUTION_LEDGER.json")
    progress_hash_before = sha256(RUN_DIR / "runtime" / "ROUND44_EXECUTION_PROGRESS.json")
    immutable_state_before = verify_state_whitelist(check_live_mutables=True)

    # This is the real R43 training gate wired to saved rows/tapes and actual batch_samples.
    # Controller reconstruction can touch process RNG, so restore the ambient stream after it.
    try:
        gate = runtime["r42_analysis"].first_batch_gate(11, runtime=runtime, frozen=frozen)
    finally:
        _restore_ambient_rng(ambient)
    if gate.get("status") != "passed":
        raise RuntimeError("ROUND44 saved batch0 did not pass the actual first-batch gate")
    comparison = r44_execute._compare_first_batch(11, runtime, gate)
    if (comparison.get("status") != "passed"
            or comparison.get("projection_version") != "round44-row-projection/2"
            or comparison.get("actor_lambda_before") != 1.0
            or comparison.get("r44_eta_lambda") != 0.5
            or comparison.get("r43_eta_lambda") != 0.05
            or comparison.get("g_hat_equal") is not True
            or comparison.get("r43_lambda_next") != 1.0125
            or comparison.get("r44_lambda_next") != 1.125
            or comparison.get("postupdate_optimizer_state_equal") is not True
            or comparison.get("postupdate_parameters_equal") is not True):
        raise RuntimeError("ROUND44 first-batch comparison did not isolate eta at batch end")
    comparator_regression = _run_comparator_regression(runtime, gate, comparison)
    if _ambient_rng_hash(runtime) != ambient_hash_before:
        raise RuntimeError("ROUND44 comparator/gate changed ambient RNG before exact restore")

    # Prove the old trained R43 checkpoint is rejected by the actual training restore path.
    old_checkpoint = Path(frozen["round42"]["old_models"]["11"]["path"])
    old_checkpoint_hash = str(frozen["round42"]["old_models"]["11"]["file_sha256"])
    if not old_checkpoint.is_file() or sha256(old_checkpoint) != old_checkpoint_hash:
        raise RuntimeError("ROUND44 read-only OLD R43 checkpoint identity changed")
    old_rng_hash = _ambient_rng_hash(runtime)
    old_counters = _counter_snapshot(runtime)
    try:
        runtime["trainer"].load_checkpoint(
            old_checkpoint, freeze=runtime["round42_freeze"])
    except RuntimeError as exc:
        old_rejection = str(exc)
        if not _is_expected_old_training_checkpoint_rejection(old_rejection):
            raise RuntimeError(f"R43 checkpoint rejected for an unexpected reason: {exc}") from exc
    else:
        raise RuntimeError("ROUND44 actual resume loader accepted the trained R43 checkpoint")
    if (_ambient_rng_hash(runtime) != old_rng_hash
            or _counter_snapshot(runtime) != old_counters):
        raise RuntimeError("rejected R43 checkpoint load changed RNG or execution counters")

    restore_gate = _exact_restore_gate(runtime, frozen)
    restore_gate["old_r43_training_checkpoint_rejected"] = {
        "path": str(old_checkpoint), "file_sha256": old_checkpoint_hash,
        "rejection": old_rejection,
    }
    comparison_path = RUN_DIR / "ROUND44_FIRST_BATCH_COMPARISON_SEED11.json"
    gate_path = RUN_DIR / "ROUND44_FIRST_BATCH_GATE_SEED11.json"
    gate_payload = {
        **gate,
        "round44_eta_lambda": 0.5,
        "r43_batch0_comparison_sha256": sha256(comparison_path),
        "r43_batch0_comparison": comparison,
    }
    write_new(gate_path, gate_payload)
    write_new(GATE_PATH, restore_gate)
    if sha256(RUN_DIR / "runtime" / "ROUND44_EXECUTION_LEDGER.json") != ledger_hash_before:
        raise RuntimeError("ROUND44 read-only gates changed the original execution ledger")
    if sha256(RUN_DIR / "runtime" / "ROUND44_EXECUTION_PROGRESS.json") != progress_hash_before:
        raise RuntimeError("ROUND44 read-only gates changed the original execution progress")
    if _counter_snapshot(runtime) != counters_before:
        raise RuntimeError("ROUND44 read-only gate suite changed cumulative counters")
    _restore_ambient_rng(ambient)
    if _ambient_rng_hash(runtime) != ambient_hash_before:
        raise RuntimeError("ROUND44 read-only gate suite failed to restore ambient RNG")
    verify_state_whitelist(check_live_mutables=True)

    original_identity = read(RUN_DIR / "runtime" / "ROUND44_PRELAUNCH_IDENTITY.json")
    source_regression_path = RUN_DIR / "runtime" / "ROUND44_SOURCE_IDENTITY_REGRESSION.json"
    source_regression = read(source_regression_path)
    if source_regression.get("status") != "passed":
        raise RuntimeError("previous R43 report/source identity positive-negative gate is missing")
    ready = {
        "kind": "round44-resume-ready/3", "status": "passed",
        "mechanical_version": MECHANICAL_VERSION,
        "state_bridge_version": STATE_BRIDGE_VERSION,
        "execution_head": head,
        "execution_identity_sha256": sha256(IDENTITY_PATH),
        "original_prelaunch_identity_sha256": sha256(
            RUN_DIR / "runtime" / "ROUND44_PRELAUNCH_IDENTITY.json"),
        "original_prelaunch_head": original_identity["head"],
        "state_whitelist_sha256": sha256(WHITELIST_PATH),
        "r1_identity_sha256": sha256(R1_IDENTITY_PATH),
        "r2_identity_sha256": sha256(R2_IDENTITY_PATH),
        "r2_failure_sha256": sha256(R2_FAILURE_PATH),
        "old_projection_evidence_sha256": sha256(OLD_PROJECTION_PATH),
        "source_config_reconciliation_sha256": sha256(SOURCE_CONFIG_RECONCILIATION_PATH),
        "state_whitelist": immutable_state_before["permitted_resume"],
        "exact_resume_gate_sha256": sha256(GATE_PATH),
        "comparator_regression_sha256": sha256(COMPARATOR_TEST_PATH),
        "first_batch_comparison_sha256": sha256(comparison_path),
        "first_batch_gate_sha256": sha256(gate_path),
        "r43_report_source_identity_regression_sha256": sha256(source_regression_path),
        "environment_starts": 0, "optimizer_steps": 0, "checkpoint_writes": 0,
    }
    write_new(READY_PATH, ready)
    return {
        "status": "passed", "head": head, "first_batch_gate": gate_payload,
        "comparison": comparison, "comparator_regression": comparator_regression,
        "exact_resume_gate": restore_gate, "ready_receipt": ready,
        "source_identity_regression": source_regression,
        "environment_starts": 0, "optimizer_steps": 0, "checkpoint_writes": 0,
        "original_ledger_hash_preserved": ledger_hash_before,
        "original_progress_hash_preserved": progress_hash_before,
    }


def main() -> int:
    if "--seal-state-whitelist" in sys.argv:
        result = build_state_whitelist()
        print(json.dumps({"status": result["status"], "path": str(WHITELIST_PATH),
                          "permitted_resume": result["permitted_resume"]}, indent=2))
    elif "--seal-execution-identity" in sys.argv:
        result = seal_resume_execution_identity()
        print(json.dumps({"status": result["status"], "head": result["head"],
                          "bridge": result["state_bridge_version"]}, indent=2))
    else:
        result = run_read_only_gates()
        print(json.dumps({
            "status": result["status"], "head": result["head"],
            "first_batch_comparison": result["comparison"]["status"],
            "exact_resume_gate": result["exact_resume_gate"]["status"],
            "comparator_negative_cases": result["comparator_regression"]["expected_rejections"],
            "environment_starts": 0, "optimizer_steps": 0, "checkpoint_writes": 0,
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
