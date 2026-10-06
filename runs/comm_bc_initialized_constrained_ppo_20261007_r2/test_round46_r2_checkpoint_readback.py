"""Zero-environment regression through the frozen ROUND40/R42 checkpoint paths."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable


FIXTURE_KIND = "round46-r2-synthetic-untrained-checkpoint-fixtures/1"
RESULT_KIND = "round46-r2-checkpoint-readback-regression/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_actual_r40_adapter(runner: Any):
    source = runner.R46_DIR / "r40_train_resume_adapter.py"
    spec = importlib.util.spec_from_file_location("round46_r2_actual_r40_train", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen R40 train adapter: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_untrained_state(state: dict[str, Any], seed: int, runner: Any) -> None:
    expected = {
        "training_seed": seed,
        "episodes_done": 0,
        "updates": 0,
        "optimizer_steps": 0,
        "lambda": 1.0,
        "training_version": runner.TRAINING_VERSION,
        "comm_action_semantics": "one-shot-send-skip/1",
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise AssertionError(f"seed {seed}: synthetic fixture state {key}={state.get(key)!r}")


def _fresh_bc_controller(runner: Any, runtime: dict[str, Any], seed: int):
    trainer = runtime["trainer"]
    controller = runtime["round46_r2_base_fresh_controller"](seed)
    r45_bc = sys.modules["r45_bc"]
    r45 = runtime["module"]
    entry = r45._bc_entry_for_seed(seed)
    payload = r45_bc.read_bc_checkpoint(Path(entry["checkpoint_path"]))
    before_load = trainer.rng_state()
    r45_bc.load_bc_controller_state(controller, payload)
    after_load = trainer.rng_state()
    if not r45._same_value(before_load, after_load):
        raise AssertionError(f"seed {seed}: BC model-only fixture load changed ambient RNG")
    if (controller.parameter_summary().get("parameter_sha256") != entry["parameter_sha256"]
            or runner._module_digest(controller, r45_bc)
            != entry["module_parameter_sha256"]):
        raise AssertionError(f"seed {seed}: synthetic fixture did not start from the registered BC model")
    return controller, entry


def _make_fixtures(runner: Any, r45: Any, runtime: dict[str, Any],
                   r40_test: Any, fixture_root: Path) -> dict[str, Any]:
    import torch

    manifest_path = fixture_root / "ROUND46_R2_FIXTURE_MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") != FIXTURE_KIND or manifest.get("status") != "passed":
            raise RuntimeError("existing R2 synthetic fixture manifest has an unexpected identity")
        for item in manifest["fixtures"]:
            path = Path(item["path"])
            if not path.is_file() or _sha256(path) != item["sha256"]:
                raise RuntimeError(f"R2 synthetic checkpoint fixture changed: {path}")
        return manifest

    trainer = runtime["trainer"]
    common = runtime["common"]
    old_checkpoint_dir = common.CHECKPOINTS
    old_output_dir = common.OUTPUT_DIR
    old_r40_train = runtime["r40_train"]
    runtime["r40_train"] = r40_test
    r40_test.common = common
    r40_test.runner = runtime["runner"]
    r40_test.trainer = trainer
    fixture_rows: list[dict[str, Any]] = []
    save_calls_before = int(runtime["r42_versions"].CHECKPOINT_WRITE_CALLS)
    try:
        for seed in runner.SEEDS:
            seed_root = fixture_root / f"seed_{seed}"
            fresh_dir = seed_root / "fresh" / "checkpoints"
            resume_dir = seed_root / "resume" / "checkpoints"
            fresh_dir.mkdir(parents=True, exist_ok=True)
            resume_dir.mkdir(parents=True, exist_ok=True)

            controller, entry = _fresh_bc_controller(runner, runtime, seed)
            optimizer = torch.optim.Adam(
                controller.parameters, lr=float(common.LEARNING_RATE))
            parameter_ids = [id(parameter) for group in optimizer.param_groups
                             for parameter in group["params"]]
            if len(parameter_ids) != 10 or len(set(parameter_ids)) != 10:
                raise AssertionError(f"seed {seed}: fixture Adam does not own ten unique parameters")
            if optimizer.state_dict().get("state") != {}:
                raise AssertionError(f"seed {seed}: fixture Adam is not fresh")
            state = trainer.fresh_training_state(seed)
            _assert_untrained_state(state, seed, runner)

            common.CHECKPOINTS = fresh_dir
            common.OUTPUT_DIR = fresh_dir.parent
            fresh_path = r40_test._save_milestone(
                controller, optimizer, state, seed, 0, runtime["round42_freeze"])
            if not fresh_path.is_file():
                raise AssertionError(f"seed {seed}: actual R40 ep0 save did not create its fixture")

            # Create a valid but changed payload. The checkpoint loader must restore this
            # payload and must never reapply the BC initialization during restoration.
            changed = copy.deepcopy(controller)
            with torch.no_grad():
                changed.parameters[0].add_(0.03125)
            changed_sha = changed.parameter_summary()["parameter_sha256"]
            if changed_sha == entry["parameter_sha256"]:
                raise AssertionError(f"seed {seed}: changed payload did not change parameters")
            changed_optimizer = torch.optim.Adam(
                changed.parameters, lr=float(common.LEARNING_RATE))
            changed_state = trainer.fresh_training_state(seed)
            _assert_untrained_state(changed_state, seed, runner)
            common.CHECKPOINTS = resume_dir
            common.OUTPUT_DIR = resume_dir.parent
            changed_ep0 = r40_test._save_milestone(
                changed, changed_optimizer, changed_state, seed, 0,
                runtime["round42_freeze"])
            rolling = resume_dir / f"v2_train_s{seed}.pt"
            shutil.copyfile(changed_ep0, rolling)
            if _sha256(changed_ep0) != _sha256(rolling):
                raise AssertionError(f"seed {seed}: rolling fixture differs from its ep0 payload")

            fixture_rows.extend([
                {"seed": seed, "role": "fresh_bc_ep0",
                 "path": str(fresh_path.resolve()), "sha256": _sha256(fresh_path),
                 "parameter_sha256": entry["parameter_sha256"],
                 "episodes_done": 0, "updates": 0, "optimizer_steps": 0},
                {"seed": seed, "role": "changed_payload_ep0",
                 "path": str(changed_ep0.resolve()), "sha256": _sha256(changed_ep0),
                 "parameter_sha256": changed_sha,
                 "episodes_done": 0, "updates": 0, "optimizer_steps": 0},
                {"seed": seed, "role": "changed_payload_rolling_copy",
                 "path": str(rolling.resolve()), "sha256": _sha256(rolling),
                 "byte_identical_to_ep0": True,
                 "episodes_done": 0, "updates": 0, "optimizer_steps": 0},
            ])
            del controller, optimizer, changed, changed_optimizer
    finally:
        common.CHECKPOINTS = old_checkpoint_dir
        common.OUTPUT_DIR = old_output_dir
        runtime["r40_train"] = old_r40_train

    save_calls_after = int(runtime["r42_versions"].CHECKPOINT_WRITE_CALLS)
    trainer_save_calls = save_calls_after - save_calls_before
    if trainer_save_calls != 2 * len(runner.SEEDS):
        raise AssertionError(
            f"synthetic fixture save count={trainer_save_calls}, expected={2 * len(runner.SEEDS)}")
    manifest = {
        "kind": FIXTURE_KIND,
        "status": "passed",
        "source": "actual frozen trainer.save_checkpoint and R40 _save_milestone",
        "fixture_count": len(fixture_rows),
        "trainer_save_checkpoint_calls": trainer_save_calls,
        "byte_copy_fixture_writes": len(runner.SEEDS),
        "environment_starts": 0,
        "claims": 0,
        "ppo_updates": 0,
        "optimizer_step_calls": 0,
        "fixtures": fixture_rows,
    }
    fixture_root.mkdir(parents=True, exist_ok=True)
    runner._write_json(manifest_path, manifest, immutable=True)
    return manifest


def _instrument_restore_calls(runtime: dict[str, Any], call: Callable[[], Any]):
    trainer = runtime["trainer"]
    original_restore = trainer._restore_controller
    original_read_only = trainer.load_checkpoint_read_only
    calls = {"restore_controller": 0, "load_checkpoint_read_only": 0}

    def counted_restore(payload: dict[str, Any], state: dict[str, Any]):
        calls["restore_controller"] += 1
        return original_restore(payload, state)

    def counted_read_only(*args: Any, **kwargs: Any):
        calls["load_checkpoint_read_only"] += 1
        return original_read_only(*args, **kwargs)

    trainer._restore_controller = counted_restore
    trainer.load_checkpoint_read_only = counted_read_only
    try:
        result = call()
        error = None
    except Exception as caught:
        result = None
        error = caught
    finally:
        trainer._restore_controller = original_restore
        trainer.load_checkpoint_read_only = original_read_only
    return result, calls, error


def _tracked_adam(torch: Any, call: Callable[[], Any]):
    original = torch.optim.Adam
    records: list[dict[str, Any]] = []

    def factory(parameters: Any, *args: Any, **kwargs: Any):
        values = list(parameters)
        optimizer = original(values, *args, **kwargs)
        ids = [id(parameter) for group in optimizer.param_groups
               for parameter in group["params"]]
        records.append({
            "parameter_count": len(ids),
            "unique_parameter_count": len(set(ids)),
            "state_empty": optimizer.state_dict().get("state") == {},
        })
        return optimizer

    torch.optim.Adam = factory
    try:
        result = call()
    finally:
        torch.optim.Adam = original
    return result, records


def _runtime_counts(runtime: dict[str, Any]) -> dict[str, int]:
    ledger = runtime["r42_run"].ledger_snapshot(strict=True)
    counts = ledger["counts"]
    return {
        "normal_used": int(counts["normal_used"]),
        "reserve_used": int(counts["reserve_used"]),
        "total_used": int(counts["total_used"]),
        "environment_constructor_calls": int(counts["environment_constructor_calls"]),
        "run_invocation_calls": int(counts["run_invocation_calls"]),
        "completed_steps": int(counts["completed_steps"]),
        "ppo_update_calls": int(counts["ppo_update_calls"]),
        "optimizer_step_calls": int(counts["optimizer_step_calls"]),
        "checkpoint_write_calls": int(counts["checkpoint_write_calls"]),
        "checkpoint_write_journal_entries": int(counts["checkpoint_write_journal_entries"]),
    }


def run_preflight_regression(runner: Any, r45: Any,
                             runtime: dict[str, Any], *,
                             write_result: bool) -> dict[str, Any]:
    import torch
    import r45_bc

    if r45 is not runtime["module"]:
        raise RuntimeError("R2 regression runtime/module identity mismatch")
    original_call = runtime["round46_r2_legacy_call_training_seed"]
    fixed_call = r45._call_training_seed
    if original_call is fixed_call:
        raise RuntimeError("R2 legacy failure path was not retained for the independent counterexample")

    r40_test = _load_actual_r40_adapter(runner)
    fixture_root = runner.R46_DIR / "test_fixtures"
    before_all = _runtime_counts(runtime)
    fixture_manifest = _make_fixtures(runner, r45, runtime, r40_test, fixture_root)
    after_fixtures = _runtime_counts(runtime)
    fixture_write_calls = (after_fixtures["checkpoint_write_calls"]
                           - before_all["checkpoint_write_calls"])
    if fixture_write_calls not in (0, 2 * len(runner.SEEDS)):
        raise AssertionError(f"unexpected synthetic fixture write delta: {fixture_write_calls}")
    if (after_fixtures["normal_used"] != 0
            or after_fixtures["reserve_used"] != 0
            or after_fixtures["total_used"] != 0
            or after_fixtures["environment_constructor_calls"] != 0
            or after_fixtures["run_invocation_calls"] != 0
            or after_fixtures["completed_steps"] != 0
            or after_fixtures["ppo_update_calls"] != 0
            or after_fixtures["optimizer_step_calls"] != 0):
        raise AssertionError(f"fixture construction changed an environment/training count: {after_fixtures}")

    trainer = runtime["trainer"]
    common = runtime["common"]
    original_checkpoint_dir = common.CHECKPOINTS
    original_output_dir = common.OUTPUT_DIR
    original_r40_train = runtime["r40_train"]
    runtime["r40_train"] = r40_test
    r40_test.common = common
    r40_test.runner = runtime["runner"]
    r40_test.trainer = trainer
    per_seed: dict[str, Any] = {}
    old_fresh_failure_count = 0
    old_resume_failure_count = 0
    read_only_direct_calls = 0
    resume_direct_calls = 0
    try:
        for seed in runner.SEEDS:
            fixture_rows = {item["role"]: item for item in fixture_manifest["fixtures"]
                            if int(item["seed"]) == seed}
            fresh_path = Path(fixture_rows["fresh_bc_ep0"]["path"])
            changed_ep0 = Path(fixture_rows["changed_payload_ep0"]["path"])
            rolling = Path(fixture_rows["changed_payload_rolling_copy"]["path"])
            bc_entry = r45._bc_entry_for_seed(seed)

            # Original R45 wiring must fail at the same actual consumer path.
            common.CHECKPOINTS = fresh_path.parent
            common.OUTPUT_DIR = (fixture_root / "probe_runs" / f"pid_{os.getpid()}"
                                 / f"seed_{seed}" / "legacy_fresh")
            legacy_report, legacy_calls, legacy_error = _instrument_restore_calls(
                runtime,
                lambda: original_call(runtime, seed, episodes=0),
            )
            if not isinstance(legacy_error, RuntimeError):
                if legacy_error is not None:
                    raise legacy_error
                raise AssertionError(f"seed {seed}: old readback wiring unexpectedly returned {legacy_report}")
            if str(legacy_error) != f"seed {seed}: fresh BC controller injection was invoked twice":
                raise legacy_error
            if (legacy_calls["load_checkpoint_read_only"] != 1
                    or legacy_calls["restore_controller"] != 1):
                raise AssertionError(
                    f"seed {seed}: old exception did not cross the actual read-only restore: {legacy_calls}")
            old_fresh_failure_count += 1

            # The R2 path initializes BC once, then reads the saved payload repeatedly
            # through the ordinary factory and registered R44 RNG path.
            common.CHECKPOINTS = fresh_path.parent
            common.OUTPUT_DIR = (fixture_root / "probe_runs" / f"pid_{os.getpid()}"
                                 / f"seed_{seed}" / "fresh")
            before_fixed_fresh = _runtime_counts(runtime)
            fixed_fresh, fresh_adam = _tracked_adam(
                torch,
                lambda: _instrument_restore_calls(
                    runtime,
                    lambda: fixed_call(runtime, seed, episodes=0),
                ),
            )
            fresh_report, fresh_calls, fresh_error = fixed_fresh
            if fresh_error is not None:
                raise RuntimeError(f"seed {seed}: fixed fresh checkpoint readback failed") from fresh_error
            after_fixed_fresh = _runtime_counts(runtime)
            initialization = fresh_report.get("round45_bc_initialization", {})
            rng_priming = fresh_report.get("rng_priming", {})
            if (int(fresh_report.get("episodes_done", -1)) != 0
                    or int(fresh_report.get("updates", -1)) != 0
                    or int(fresh_report.get("optimizer_steps", -1)) != 0
                    or initialization.get("applied") is not True
                    or initialization.get("bc_parameter_sha256") != bc_entry["parameter_sha256"]
                    or initialization.get("module_parameter_sha256")
                    != bc_entry["module_parameter_sha256"]
                    or initialization.get("bc_load_changed_rng") is not False
                    or initialization.get("bc_optimizer_loaded") is not False
                    or initialization.get("registered_ep0_rollout_rng_applied_by_r42_entry") is not True
                    or rng_priming.get("applied_after_fresh_controller") is not True
                    or fresh_calls["load_checkpoint_read_only"] < 2
                    or fresh_calls["restore_controller"] < 2
                    or len(fresh_adam) != 1
                    or fresh_adam[0] != {"parameter_count": 10,
                                        "unique_parameter_count": 10,
                                        "state_empty": True}):
                raise AssertionError(
                    f"seed {seed}: fixed fresh init/readback/Adam/RNG gate failed: "
                    f"init={initialization}, rng={rng_priming}, calls={fresh_calls}, adam={fresh_adam}")
            if any(after_fixed_fresh[key] != before_fixed_fresh[key] for key in (
                    "normal_used", "reserve_used", "total_used", "environment_constructor_calls",
                    "run_invocation_calls", "completed_steps", "ppo_update_calls",
                    "optimizer_step_calls", "checkpoint_write_calls")):
                raise AssertionError(f"seed {seed}: fixed fresh readback changed execution counters")

            # Original resume factory path must also fail on the second payload restore.
            common.CHECKPOINTS = rolling.parent
            common.OUTPUT_DIR = (fixture_root / "probe_runs" / f"pid_{os.getpid()}"
                                 / f"seed_{seed}" / "legacy_resume")
            legacy_resume, legacy_resume_calls, legacy_resume_error = _instrument_restore_calls(
                runtime,
                lambda: original_call(runtime, seed, episodes=0),
            )
            if not isinstance(legacy_resume_error, RuntimeError):
                if legacy_resume_error is not None:
                    raise legacy_resume_error
                resume_report = legacy_resume
                raise AssertionError(f"seed {seed}: old resume wiring unexpectedly returned {resume_report}")
            if str(legacy_resume_error) != f"seed {seed}: fresh BC controller injection was invoked twice":
                raise legacy_resume_error
            if legacy_resume_calls["restore_controller"] < 2:
                raise AssertionError(
                    f"seed {seed}: old resume exception did not cross both real restores: "
                    f"{legacy_resume_calls}")
            old_resume_failure_count += 1

            # R2 resume construction validates the rolling state and restores the saved
            # changed payload; the BC initializer is not invoked at all.
            common.CHECKPOINTS = rolling.parent
            common.OUTPUT_DIR = (fixture_root / "probe_runs" / f"pid_{os.getpid()}"
                                 / f"seed_{seed}" / "resume")
            before_fixed_resume = _runtime_counts(runtime)
            fixed_resume, resume_adam = _tracked_adam(
                torch,
                lambda: _instrument_restore_calls(
                    runtime,
                    lambda: fixed_call(runtime, seed, episodes=0),
                ),
            )
            resume_report, resume_calls, resume_error = fixed_resume
            if resume_error is not None:
                raise RuntimeError(f"seed {seed}: fixed resume checkpoint restore failed") from resume_error
            after_fixed_resume = _runtime_counts(runtime)
            resume_initialization = resume_report.get("round45_bc_initialization", {})
            changed_sha = fixture_rows["changed_payload_ep0"]["parameter_sha256"]
            resumed_sha = resume_report.get("controller_parameters", {}).get("parameter_sha256")
            if (int(resume_report.get("episodes_done", -1)) != 0
                    or int(resume_report.get("updates", -1)) != 0
                    or int(resume_report.get("optimizer_steps", -1)) != 0
                    or resume_initialization.get("applied") is not False
                    or resumed_sha != changed_sha
                    or resumed_sha == bc_entry["parameter_sha256"]
                    or resume_calls["restore_controller"] < 3
                    or resume_calls["load_checkpoint_read_only"] < 1
                    or len(resume_adam) != 1
                    or resume_adam[0] != {"parameter_count": 10,
                                          "unique_parameter_count": 10,
                                          "state_empty": True}):
                raise AssertionError(
                    f"seed {seed}: fixed resume restored BC or failed state gate: "
                    f"init={resume_initialization}, digest={resumed_sha}, calls={resume_calls}, "
                    f"adam={resume_adam}")
            if any(after_fixed_resume[key] != before_fixed_resume[key] for key in (
                    "normal_used", "reserve_used", "total_used", "environment_constructor_calls",
                    "run_invocation_calls", "completed_steps", "ppo_update_calls",
                    "optimizer_step_calls", "checkpoint_write_calls")):
                raise AssertionError(f"seed {seed}: fixed resume readback changed execution counters")

            # Exercise repeated real read-only and full resume loaders with a factory that
            # would fail immediately if BC or the ordinary module binding leaked in.
            controller_module = runtime["controller_module"]
            base_factory = runtime["round46_r2_base_fresh_controller"]
            original_factory = controller_module.fresh_controller
            forbidden_calls = {"count": 0}

            def forbidden_factory(model_seed: int):
                forbidden_calls["count"] += 1
                raise AssertionError(f"seed {seed}: readback called a non-base factory for {model_seed}")

            controller_module.fresh_controller = forbidden_factory
            ambient_rng = trainer.rng_state()
            readonly_digests = []
            try:
                for _ in range(2):
                    with runner._checkpoint_restore_factory_scope(runtime):
                        restored, state, identity = trainer.load_checkpoint_read_only(
                            changed_ep0, freeze=runtime["round42_freeze"])
                    readonly_digests.append(restored.parameter_summary()["parameter_sha256"])
                    if (state.get("episodes_done") != 0
                            or identity.get("read_only_for_round42") is not True
                            or identity.get("rng_restored") is not False):
                        raise AssertionError(f"seed {seed}: read-only loader identity failed: {identity}")
                if not r45._same_value(ambient_rng, trainer.rng_state()):
                    raise AssertionError(f"seed {seed}: repeated read-only load changed training RNG")
                if readonly_digests != [changed_sha, changed_sha]:
                    raise AssertionError(f"seed {seed}: repeated read-only load did not restore payload")

                ambient_before_resume = trainer.rng_state()
                with runner._checkpoint_restore_factory_scope(runtime):
                    resumed_controller, resumed_optimizer, resumed_state = trainer.load_checkpoint(
                        rolling, freeze=runtime["round42_freeze"])
                if (resumed_controller.parameter_summary()["parameter_sha256"] != changed_sha
                        or resumed_state.get("episodes_done") != 0
                        or resumed_optimizer.state_dict().get("state") != {}):
                    raise AssertionError(f"seed {seed}: full resume loader did not honor changed payload")
                trainer.restore_rng(ambient_before_resume)
            finally:
                controller_module.fresh_controller = original_factory
                trainer.restore_rng(ambient_rng)
            if forbidden_calls["count"] != 0:
                raise AssertionError(f"seed {seed}: forbidden restore factory was called")
            read_only_direct_calls += 2
            resume_direct_calls += 1

            per_seed[str(seed)] = {
                "bc_four_module_initialization": {
                    "checkpoint_sha256": bc_entry["checkpoint_sha256"],
                    "parameter_sha256": initialization.get("bc_parameter_sha256"),
                    "module_parameter_sha256": initialization.get("module_parameter_sha256"),
                    "bc_optimizer_loaded": False,
                    "bc_rng_loaded": False,
                    "r44_ep0_rng_applied_after_initialization": True,
                },
                "fresh_adam": fresh_adam[0],
                "fresh_real_save_readback": {
                    "status": "passed",
                    "old_wiring_failure_reproduced": True,
                    "old_exception": "fresh BC controller injection was invoked twice",
                    "fixed_read_only_calls_during_train_seed": fresh_calls[
                        "load_checkpoint_read_only"],
                    "fixed_restore_calls_during_train_seed": fresh_calls["restore_controller"],
                    "fixed_adam_state_empty": True,
                },
                "resume_payload_restore": {
                    "status": "passed",
                    "old_wiring_failure_reproduced": True,
                    "bc_injection_applied_on_resume": False,
                    "payload_parameter_sha256": changed_sha,
                    "restored_parameter_sha256": resumed_sha,
                    "read_only_calls_during_resume": resume_calls[
                        "load_checkpoint_read_only"],
                    "restore_calls_during_resume": resume_calls["restore_controller"],
                    "fresh_resume_adam_state_empty": True,
                },
                "repeated_read_only": {
                    "calls": len(readonly_digests),
                    "payload_parameter_sha256": changed_sha,
                    "restored_parameter_sha256": readonly_digests,
                    "ambient_training_rng_preserved": True,
                    "bc_or_forbidden_factory_calls": forbidden_calls["count"],
                },
            }
    finally:
        common.CHECKPOINTS = original_checkpoint_dir
        common.OUTPUT_DIR = original_output_dir
        runtime["r40_train"] = original_r40_train

    final_counts = _runtime_counts(runtime)
    if (old_fresh_failure_count != len(runner.SEEDS)
            or old_resume_failure_count != len(runner.SEEDS)
            or any(final_counts[key] != 0 for key in (
                "normal_used", "reserve_used", "total_used", "environment_constructor_calls",
                "run_invocation_calls", "completed_steps", "ppo_update_calls",
                "optimizer_step_calls"))
            or final_counts["checkpoint_write_calls"] != after_fixtures[
                "checkpoint_write_calls"]):
        raise AssertionError(
            f"R2 regression did not close at zero execution counts: {final_counts}")
    if final_counts["checkpoint_write_calls"] != final_counts["checkpoint_write_journal_entries"]:
        raise AssertionError("synthetic fixture checkpoint writes do not close against their call journal")

    result = {
        "kind": RESULT_KIND,
        "status": "passed",
        "runner_version": runner.ROUND46_RUNNER_VERSION,
        "training_version": runner.TRAINING_VERSION,
        "test_source_path": str(Path(__file__).resolve()),
        "test_source_sha256": _sha256(Path(__file__)),
        "prior_attempt_stop_sha256": runner._prior_failed_attempt_identity()["files"][
            "stop_state"]["sha256"],
        "real_legacy_path": "R45 _call_training_seed -> R42 train_seed -> R40 _save_milestone -> R42 load_checkpoint_read_only -> trainer._restore_controller",
        "legacy_fresh_failures_reproduced": old_fresh_failure_count,
        "legacy_resume_failures_reproduced": old_resume_failure_count,
        "fixed_fresh_seeds_passed": list(runner.SEEDS),
        "fixed_resume_seeds_passed": list(runner.SEEDS),
        "synthetic_fixture_manifest_path": str(
            (fixture_root / "ROUND46_R2_FIXTURE_MANIFEST.json").resolve()),
        "synthetic_fixture_manifest_sha256": _sha256(
            fixture_root / "ROUND46_R2_FIXTURE_MANIFEST.json"),
        "synthetic_fixture_trainer_save_calls_total": int(
            fixture_manifest["trainer_save_checkpoint_calls"]),
        "synthetic_fixture_trainer_save_calls_this_preflight": fixture_write_calls,
        "synthetic_fixture_byte_copy_writes_total": int(
            fixture_manifest["byte_copy_fixture_writes"]),
        "repeated_direct_read_only_calls": read_only_direct_calls,
        "direct_resume_loader_calls": resume_direct_calls,
        "per_seed": per_seed,
        "counts_before_fixture_creation": before_all,
        "counts_after_fixture_creation": after_fixtures,
        "counts_after_all_regression_paths": final_counts,
        "current_round_normal_claims": 0,
        "current_round_reserve_claims": 0,
        "environment_constructor_calls": 0,
        "run_invocation_calls": 0,
        "completed_steps": 0,
        "ppo_update_calls": 0,
        "optimizer_step_calls": 0,
    }
    if write_result:
        result_path = runner.READBACK_REGRESSION_RESULT_PATH
        runner._write_json(result_path, result, immutable=True)
    return result


def _load_runner():
    runner_path = Path(__file__).resolve().with_name("r46_execute_r2.py")
    spec = importlib.util.spec_from_file_location("round46_r2_dev_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load R2 runner at {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ROUND46-R2 zero-environment seam regression")
    parser.add_argument("--development-probe", action="store_true", required=True,
                        help="run real save/readback/resume seam tests; never starts an environment")
    parser.add_argument("--write-result", action="store_true",
                        help="write the formal regression receipt")
    args = parser.parse_args()
    runner = _load_runner()
    if not runner.KEY_PLAN_PATH.is_file():
        identity = runner._input_identity()
        plan = runner._logical_key_plan(
            list(identity["scenario_order"]), identity["scenario_order_verification"])
        runner._write_json(runner.KEY_PLAN_PATH, plan)
    r45 = runner._configure_runtime_module()
    runtime = runner._RUNTIME
    if runtime is None:
        raise RuntimeError("R2 runtime was not configured")
    runtime["module"] = r45
    result = run_preflight_regression(
        runner, r45, runtime, write_result=args.write_result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
