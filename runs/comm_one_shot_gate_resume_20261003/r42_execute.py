"""Execute the frozen ROUND42 672-cell plan, stopping at any gate or runtime failure."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from r42_analysis import analyze_all, first_batch_gate
from r42_report import write_final_deliverables
from r42_run import (DEV_SCENARIOS, RUN_DIR, SEEDS, configure, restore_runtime_hooks,
                     run_model_eval, run_reference, run_rule, train_seed,
                     verify_frozen_source, write_execution_ledger)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _progress(*, phase: str, done: int, total: int, last: str,
              detail: dict[str, Any] | None = None) -> None:
    _write_atomic(RUN_DIR / "runtime" / "ROUND42_PROGRESS.json", {
        "kind": "round42-progress/1",
        "phase": phase,
        "completed_cells_in_phase": int(done),
        "planned_cells_in_phase": int(total),
        "last_completed_or_attempted_cell": last,
        "updated_at_unix": time.time(),
        "detail": detail or {},
    })


def _ensure_clean_start() -> None:
    frozen_path = RUN_DIR / "ROUND42_FROZEN_CONFIG.json"
    preflight_path = RUN_DIR / "ROUND42_PREFLIGHT.json"
    if not frozen_path.is_file() or not preflight_path.is_file():
        raise RuntimeError("ROUND42 prelaunch freeze or preflight artifact is missing")
    frozen = _read(frozen_path)
    preflight = _read(preflight_path)
    if frozen.get("status") != "passed_before_environment_start" or preflight.get("status") != "passed":
        raise RuntimeError("ROUND42 prelaunch gate has not passed")
    for directory in (RUN_DIR / "logs", RUN_DIR / "rows", RUN_DIR / "checkpoints",
                      RUN_DIR / "launch_claims", RUN_DIR / "launch_slots"):
        if directory.exists() and any(directory.rglob("*")):
            raise RuntimeError(
                f"ROUND42 refuses implicit resume; prior run artifacts need explicit review: "
                f"{directory}"
            )
    progress = RUN_DIR / "runtime" / "ROUND42_PROGRESS.json"
    if progress.exists():
        raise RuntimeError("ROUND42 progress exists; this runner never auto-retries or resumes")


def _seed_carryover(frozen: dict[str, Any]) -> dict[str, Any]:
    """Import the immutable failed attempt into the isolated quota ledger byte-for-byte."""
    manifest_path = RUN_DIR / "ROUND42R_CARRYOVER.json"
    manifest = _read(manifest_path)
    if manifest.get("status") != "verified_before_new_execution":
        raise RuntimeError("ROUND42-R carryover manifest is not verified")
    expected_sha = frozen["execution_identity"]["carryover_manifest_sha256"]
    import hashlib
    actual_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError("ROUND42-R carryover manifest hash differs from the freeze")
    root = Path(frozen["execution_identity"]["round42_worktree"])
    imports = (
        ("old_claim", RUN_DIR / "launch_claims" / "immediate_reference_scene1300000"
         / "attempt_0001.json"),
        ("old_normal_slot", RUN_DIR / "launch_slots" / "normal" / "0001.json"),
    )
    for source_key, target in imports:
        record = manifest["sources"][source_key]
        source = root / record["repo_relative_path"]
        source_bytes = source.read_bytes()
        if len(source_bytes) != int(record["size_bytes"]):
            raise RuntimeError(f"ROUND42-R carryover source size changed: {source}")
        if hashlib.sha256(source_bytes).hexdigest() != record["sha256"]:
            raise RuntimeError(f"ROUND42-R carryover source hash changed: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(source_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        if target.read_bytes() != source_bytes:
            raise RuntimeError(f"ROUND42-R failed byte-for-byte carryover import: {target}")

    job_ledger = {
        "kind": "round42r-logical-job-ledger/1",
        "scope": "explicit old-failure-to-reserve-retry mapping; all other logical jobs are recorded by their atomic attempt claims",
        "carryover_manifest_sha256": expected_sha,
        "jobs": {
            "immediate_reference_scene1300000": {
                "status": "pending_reserve_retry",
                "attempts": [
                    {"attempt": 1, "category": "normal", "status": "failed_technical",
                     "source": manifest["sources"]["old_claim"]["repo_relative_path"]},
                    {"attempt": 2, "category": "reserve", "status": "claimed_on_execution",
                     "retry_of_attempt": 1},
                ],
            },
        },
        "normal_slots_imported": 1,
        "reserve_slots_imported": 0,
        "new_normal_jobs_remaining": 671,
        "logical_job_target": 672,
        "hard_cap": 700,
    }
    _write_atomic(RUN_DIR / "runtime" / "ROUND42R_LOGICAL_JOB_LEDGER.json", job_ledger)
    return manifest


def _update_retry_job(status: str, *, row: dict[str, Any] | None = None,
                      error: BaseException | None = None) -> None:
    path = RUN_DIR / "runtime" / "ROUND42R_LOGICAL_JOB_LEDGER.json"
    ledger = _read(path)
    job = ledger["jobs"]["immediate_reference_scene1300000"]
    job["status"] = status
    retry = job["attempts"][1]
    retry["status"] = status
    if row is not None:
        retry["row_path"] = row["artefacts"]["row"]
        retry["steps"] = int(row["steps"])
        retry["on_time"] = int(row["on_time"])
        retry["spent_bytes"] = int(row["spent_bytes"])
    if error is not None:
        retry["error"] = f"{type(error).__name__}: {error}"
    _write_atomic(path, ledger)


def _validate_retry_reference(runtime: dict[str, Any], frozen: dict[str, Any],
                              row: dict[str, Any]) -> dict[str, Any]:
    import hashlib
    import r42_run

    if (int(row.get("scenario_seed", -1)) != 1_300_000
            or int(row.get("attempt", -1)) != 2
            or row.get("reference_arm") != "immediate"
            or row.get("policy") != "immediate"
            or int(row.get("steps", -1)) != int(frozen["round42"]["steps_per_episode"])
            or int(row.get("on_time", -1)) != int(row.get("task_reward_total", -2))
            or int(row.get("spent_bytes", -1)) != int(row.get("airtime_tap_total", -2))
            or int(row.get("spent_bytes", -1)) != sum(
                int(item.get("airtime_bytes", 0)) for item in row["cost_stream"].values())):
        raise RuntimeError(f"ROUND42-R reference retry row did not close: {row.get('label')}")
    claim_path = (RUN_DIR / "launch_claims" / "immediate_reference_scene1300000"
                  / "attempt_0002.json")
    claim = _read(claim_path)
    if (claim.get("status") != "complete" or claim.get("category") != "reserve"
            or int(claim.get("attempt", -1)) != 2
            or claim.get("metadata", {}).get("retry_of") != (
                "runs/comm_one_shot_report_selection_20261003/launch_claims/"
                "immediate_reference_scene1300000/attempt_0001.json")
            or int(claim.get("environment_constructor_calls", 0)) != 1
            or int(claim.get("joint_run_constructed_calls", 0)) != 1
            or int(claim.get("run_invocation_calls", 0)) != 1
            or int(claim.get("joint_run_method_calls", 0)) != 1
            or int(claim.get("run_return_calls", 0)) != 1
            or int(claim.get("completed_steps", -1)) != int(row["steps"])):
        raise RuntimeError(f"ROUND42-R reserve retry claim failed its runner-count gate: {claim}")

    lifecycle = Path(row["artefacts"]["lifecycle"])
    config_records = [
        json.loads(line) for line in lifecycle.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    config_record = next((item for item in config_records
                          if item.get("event_type") == "run_config"), None)
    if config_record is None:
        raise RuntimeError("ROUND42-R retry lifecycle has no run_config record")
    detail = config_record.get("detail", {})
    signature = detail.get("run_signature", {})
    if signature.get("config_hash") != row.get("config_hash"):
        raise RuntimeError("ROUND42-R retry row/lifecycle config hashes differ")
    expected_fields, _substituted = runtime["common"].stage_b_inherited_config(1_300_000)
    expected_paths = runtime["common"].cell_paths("reference", 1_300_000, 2, tag="")
    expected_fields["comm_policy"] = "immediate"
    expected_fields["output"] = str(expected_paths["lifecycle"])
    expected_fields["comm_transcript"] = str(expected_paths["transcript"])
    expected_configuration = runtime["runner"].RunConfiguration(**expected_fields)
    from integrated_sar.run_signature import config_hash
    expected_config_hash = config_hash(expected_configuration)
    if (signature.get("config_hash") != expected_config_hash
            or detail.get("configuration") != expected_configuration.as_dict()):
        raise RuntimeError("ROUND42-R retry configuration differs from the frozen inherited recipe")
    if (signature.get("code_commit") != frozen["execution_identity"][
            "canonical_read_only_input_head"]
            or int(signature.get("seed", -1)) != 1_300_000):
        raise RuntimeError("ROUND42-R retry run signature source or scenario seed differs")
    expected_checkpoints = frozen["round42"]["frozen_search_allocation_checkpoints"]
    expected_hashes = {
        "piloc": expected_checkpoints["shared_mappo_model_21_3100.pth"]["sha256"],
        "ctde": expected_checkpoints["ctde_final_7_1.pth"]["sha256"],
    }
    if signature.get("checkpoints") != expected_hashes:
        raise RuntimeError(
            f"ROUND42-R retry search/allocation checkpoint identity differs: "
            f"{signature.get('checkpoints')} != {expected_hashes}")
    if detail.get("configuration", {}).get("comm_policy") != "immediate":
        raise RuntimeError("ROUND42-R retry lifecycle does not record immediate comm policy")
    for entry in expected_checkpoints.values():
        path = Path(entry["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"ROUND42-R frozen search/allocation weight changed: {path}")
    if (row.get("training_version") != frozen["round42"]["training_version"]
            or row.get("reward_timing_version") != frozen["round42"]["reward_timing_version"]
            or row.get("runner_version") != frozen["execution_identity"]["runner_version"]):
        raise RuntimeError("ROUND42-R retry row execution identity differs from freeze")
    counts = r42_run.ledger_snapshot(strict=True)["counts"]
    if (counts["environment_constructor_calls"] != 2
            or counts["joint_run_constructed_calls"] != 2
            or counts["run_invocation_calls"] != 1
            or counts["joint_run_method_calls"] != 1
            or counts["run_return_calls"] != 1
            or counts["completed_steps"] != int(row["steps"])
            or counts["normal_used"] != 1 or counts["reserve_used"] != 1
            or counts["total_used"] != 2):
        raise RuntimeError(f"ROUND42-R retry cumulative ledger failed: {counts}")
    gate = {
        "kind": "round42r-reference-retry-gate/1",
        "status": "passed",
        "logical_job": "immediate_reference_scene1300000",
        "old_failed_attempt": {
            "attempt": 1, "category": "normal", "status": "failed_technical",
            "claim_path": "runs/comm_one_shot_report_selection_20261003/launch_claims/immediate_reference_scene1300000/attempt_0001.json",
            "claim_sha256": _read(RUN_DIR / "ROUND42R_CARRYOVER.json")[
                "sources"]["old_claim"]["sha256"],
        },
        "successful_attempt": {
            "attempt": 2, "category": "reserve", "status": "complete",
            "row_path": row["artefacts"]["row"],
            "row_sha256": hashlib.sha256(Path(row["artefacts"]["row"]).read_bytes()).hexdigest(),
            "lifecycle_sha256": hashlib.sha256(lifecycle.read_bytes()).hexdigest(),
            "run_signature": signature,
            "counts": {
                "constructors_cumulative": counts["environment_constructor_calls"],
                "joint_runs_constructed_cumulative": counts["joint_run_constructed_calls"],
                "run_calls": counts["run_invocation_calls"],
                "completed_steps": counts["completed_steps"],
            },
            "outcome": {
                "steps": int(row["steps"]), "on_time": int(row["on_time"]),
                "task_reward_total": int(row["task_reward_total"]),
                "spent_bytes": int(row["spent_bytes"]),
                "airtime_tap_total": int(row["airtime_tap_total"]),
            },
        },
        "old_failure_preserved": True,
        "reserve_linkage_valid": True,
        "scientific_configuration_and_search_allocation_identity_valid": True,
        "next_action": "continue the remaining 63 immediate references, then the frozen training and evaluation plan",
    }
    _write_atomic(RUN_DIR / "runtime" / "ROUND42R_REFERENCE_RETRY_GATE.json", gate)
    _update_retry_job("complete", row=row)
    return gate


def _validate_reference_block(runtime: dict[str, Any]) -> None:
    rows = list(runtime["common"].ROWS.glob("v2_reference_s*_a*.json"))
    references = [_read(path) for path in rows]
    scenes = [int(row["scenario_seed"]) for row in references]
    expected = set(range(1_300_000, 1_300_048)) | set(DEV_SCENARIOS)
    if len(references) != 64 or set(scenes) != expected:
        raise RuntimeError(f"immediate reference block is incomplete or duplicated: {scenes}")
    for row in references:
        if (int(row["on_time"]) != int(row["task_reward_total"])
                or int(row["spent_bytes"]) != int(row["airtime_tap_total"])
                or int(row["steps"]) != 1000
                or row.get("reference_arm") != "immediate"):
            raise RuntimeError(f"immediate task/cost closure failed: {row['label']}")


def _write_gate(seed: int, gate: dict[str, Any]) -> None:
    path = RUN_DIR / "runtime" / "gates" / f"seed_{seed}_first_batch.json"
    _write_atomic(path, gate)


def _execute() -> dict[str, Any]:
    _ensure_clean_start()
    frozen = _read(RUN_DIR / "ROUND42_FROZEN_CONFIG.json")
    round40_frozen = _read(
        Path(frozen["execution_identity"]["round40_worktree"])
        / "runs" / "comm_first_report_training_ablation_20261003"
        / "ROUND40_FROZEN_CONFIG.json"
    )
    prelaunch_head = verify_frozen_source(frozen)
    runtime = configure(frozen, round40_frozen["round40"]["freeze_identity"])
    prelaunch_head = verify_frozen_source(frozen)
    _write_atomic(RUN_DIR / "runtime" / "ROUND42_PRELAUNCH_IDENTITY.json", {
        "kind": "round42-committed-prelaunch-identity/1",
        "branch": frozen["execution_identity"]["branch"],
        "head": prelaunch_head,
        "basis_commit": frozen["execution_identity"]["round42_basis_commit"],
        "frozen_config_sha256": frozen["round42"]["freeze_identity"]["freeze_sha256"],
        "round40_head": frozen["execution_identity"]["round40_delivery_head"],
        "round41_head": frozen["execution_identity"]["round41_delivery_head"],
        "environment_starts_before_this_record": 0,
    })
    _seed_carryover(frozen)
    plan = frozen["round42"]["plan"]
    started = time.time()

    retry_scene = 1_300_000
    try:
        retry_row = run_reference(retry_scene)
        retry_gate = _validate_retry_reference(runtime, frozen, retry_row)
    except BaseException as error:
        _update_retry_job("failed_gate", error=error)
        raise
    _progress(phase="immediate_reference", done=1, total=64,
              last=str(retry_row["label"]), detail={
                  "scenario": retry_scene, "attempt": int(retry_row["attempt"]),
                  "retry_gate": "passed",
              })

    remaining_references = [*range(1_300_001, 1_300_048), *DEV_SCENARIOS]
    for index, scene in enumerate(remaining_references, start=2):
        verify_frozen_source(frozen)
        row = run_reference(int(scene))
        _progress(phase="immediate_reference", done=index, total=64,
                  last=str(row["label"]), detail={"scenario": int(scene)})
    _validate_reference_block(runtime)
    write_execution_ledger()

    gates: dict[str, Any] = {}
    training_reports: dict[str, Any] = {}
    for seed in SEEDS:
        verify_frozen_source(frozen)
        first = train_seed(seed, episodes=4, frozen=frozen)
        _progress(phase=f"new_training_seed_{seed}_first_four", done=4, total=4,
                  last=f"seed={seed} batch=0", detail={
                      "episodes_done": first["episodes_done"],
                      "optimizer_steps": first["optimizer_steps"],
                  })
        gate = first_batch_gate(seed, runtime=runtime, frozen=frozen)
        gates[str(seed)] = gate
        _write_gate(seed, gate)
        _progress(phase=f"new_training_seed_{seed}_gate", done=1, total=1,
                  last=f"seed={seed} first-batch-gate-passed", detail=gate[
                      "independent_likelihood"])

        verify_frozen_source(frozen)
        completed = train_seed(seed, episodes=128, frozen=frozen)
        if (int(completed["episodes_done"]) != 128
                or int(completed["updates"]) != 32
                or int(completed["optimizer_steps"]) <= 0):
            raise RuntimeError(f"seed {seed}: completed training checkpoint cursor mismatch")
        training_reports[str(seed)] = {
            "first_four_gate": gate,
            "completion": {key: value for key, value in completed.items()
                           if key not in ("history", "lambda_history")},
            "lambda_history": completed["lambda_history"],
        }
        _write_atomic(RUN_DIR / "runtime" / "training_reports.json", training_reports)
        write_execution_ledger()
        _progress(phase=f"new_training_seed_{seed}_complete", done=128, total=128,
                  last=f"seed={seed} ep128", detail=training_reports[str(seed)]["completion"])

    # Models are loaded once per (arm, seed), then used read-only in both decoder modes.
    from r42_run import load_eval_model

    evaluation_cells = 0
    evaluations_total = int(plan["old_new_dev"]["total"])
    for arm in ("OLD", "NEW"):
        for seed in SEEDS:
            controller, identity = load_eval_model(arm, seed)
            original_parameter_sha = identity["parameter_identity"]["parameter_sha256"]
            for mode in ("greedy", "sampled"):
                for scene in DEV_SCENARIOS:
                    verify_frozen_source(frozen)
                    row = run_model_eval(
                        arm=arm, training_seed=seed, mode=mode, scenario=int(scene),
                        controller=controller, identity=identity)
                    if (controller.parameter_summary()["parameter_sha256"]
                            != original_parameter_sha):
                        raise RuntimeError(f"read-only {arm} model changed during evaluation")
                    evaluation_cells += 1
                    _progress(phase="old_new_development", done=evaluation_cells,
                              total=evaluations_total, last=row["label"], detail={
                                  "arm": arm, "seed": seed, "mode": mode,
                                  "scenario": int(scene),
                              })

    rule_cells = 0
    for arm in ("late_discovery_gate", "always_hold"):
        for scene in DEV_SCENARIOS:
            verify_frozen_source(frozen)
            row = run_rule(arm, int(scene))
            rule_cells += 1
            _progress(phase="rule_development", done=rule_cells, total=32,
                      last=row["label"], detail={"arm": arm, "scenario": int(scene)})

    ledger = write_execution_ledger()
    counts = ledger["counts"]
    if (counts["normal_used"] != 672 or counts["reserve_used"] != 1
            or counts["total_used"] != 673
            or counts["environment_constructor_calls"] != 673
            or counts["joint_run_constructed_calls"] != 673
            or counts["run_invocation_calls"] != 672
            or counts["joint_run_method_calls"] != 672
            or counts["run_return_calls"] != 672
            or counts["completed_steps"] != 672_000
            or counts["completed_logical_jobs"] != 672
            or counts["ppo_update_calls"] != 96
            or not counts["ppo_update_count_matches_committed_batches"]
            or counts["simulation_started"] != 672
            or not counts["optimizer_step_count_matches_committed_batches"]):
        raise RuntimeError(f"ROUND42 final launch/update ledger failed its exact count gate: {counts}")

    paired = analyze_all(runtime=runtime, frozen=frozen)
    result = {
        "kind": "round42-execution-result/1",
        "status": "complete",
        "duration_seconds": round(time.time() - started, 2),
        "prelaunch_head": prelaunch_head,
        "training_version": frozen["round42"]["training_version"],
        "runner_version": frozen["execution_identity"]["runner_version"],
        "reference_retry_gate": retry_gate,
        "carryover_manifest_sha256": frozen["execution_identity"][
            "carryover_manifest_sha256"],
        "first_batch_gates": gates,
        "training_reports": training_reports,
        "paired_results": paired,
        "execution_ledger": ledger,
    }
    _write_atomic(RUN_DIR / "runtime" / "ROUND42_EXECUTION_RESULT.json", result)
    write_final_deliverables(runtime=runtime, frozen=frozen, result=result)
    _write_atomic(RUN_DIR / "ROUND42R_REFERENCE_RETRY_GATE.json", retry_gate)
    _write_atomic(RUN_DIR / "ROUND42R_FINAL_LEDGER.json", ledger)
    _write_atomic(RUN_DIR / "ROUND42R_LOGICAL_JOB_LEDGER.json",
                  _read(RUN_DIR / "runtime" / "ROUND42R_LOGICAL_JOB_LEDGER.json"))
    _progress(phase="complete", done=672, total=672, last="all registered cells complete",
              detail=counts)
    return result


def _record_stopped_state(error: BaseException) -> None:
    """Create a reviewable partial delivery after a gate/runtime stop, without retrying."""
    claims = []
    for path in sorted((RUN_DIR / "launch_claims").glob("*/*.json")):
        try:
            claims.append(_read(path))
        except Exception:
            continue
    normal_used_local = len(list((RUN_DIR / "launch_slots" / "normal").glob("*.json")))
    reserve_used = len(list((RUN_DIR / "launch_slots" / "reserve").glob("*.json")))
    carryover_path = RUN_DIR / "ROUND42R_CARRYOVER.json"
    carryover = _read(carryover_path) if carryover_path.is_file() else None
    old_job = "immediate_reference_scene1300000"
    old_imported = any(row.get("logical_job") == old_job and int(row.get("attempt", 0)) == 1
                       for row in claims)
    if carryover is not None and not old_imported:
        source = Path(_read(RUN_DIR / "ROUND42_FROZEN_CONFIG.json")[
            "execution_identity"]["round42_worktree"]
        ) / carryover["sources"]["old_claim"]["repo_relative_path"]
        if source.is_file():
            claims.append(_read(source))
    normal_used = normal_used_local + int(carryover is not None and not old_imported)
    constructor_calls = sum(int(row.get("environment_constructor_calls", 0)) for row in claims)
    run_calls = sum(int(row.get("run_invocation_calls", 0)) for row in claims)
    joint_run_calls = sum(int(row.get("joint_run_method_calls", 0)) for row in claims)
    constructed_returns = sum(int(row.get("joint_run_constructed_calls", 0)) for row in claims)
    if carryover is not None:
        constructed_returns += int(carryover.get("old_failure", {}).get(
            "joint_run_constructed_calls", 0))
    completed_steps = sum(int(row.get("completed_steps", 0)) for row in claims)
    simulation_started = sum(bool(row.get("simulation_started")) for row in claims)
    failed = [row for row in claims if row.get("status") == "failed_technical"]
    old_failed = [row for row in failed if row.get("logical_job") == old_job
                  and int(row.get("attempt", 0)) == 1]
    new_failed = [row for row in failed if row not in old_failed]
    completed_jobs = {str(row.get("logical_job")) for row in claims
                      if row.get("status") == "complete"}
    state = {
        "kind": "round42-stopped-execution-state/1",
        "status": "stopped_without_automatic_retry",
        "exception": f"{type(error).__name__}: {error}",
        "normal_launch_claims": normal_used,
        "reserve_launch_claims": reserve_used,
        "total_launch_claims": normal_used + reserve_used,
        "old_failed_claims_preserved": len(old_failed),
        "new_failed_claims": new_failed,
        "cumulative_failed_claims": failed,
        "environment_constructor_calls": constructor_calls,
        "joint_run_constructed_calls": constructed_returns,
        "run_invocation_calls": run_calls,
        "joint_run_method_calls": joint_run_calls,
        "completed_steps": completed_steps,
        "simulation_started_calls": simulation_started,
        "completed_logical_jobs": len(completed_jobs),
        "failed_claims": failed,
        "claim_count": len(claims),
        "no_automatic_retry_or_resume": True,
        "traceback": __import__("traceback").format_exc(),
    }
    try:
        import r42_run
        state["optimizer_step_hook_calls_in_process"] = int(r42_run._OPTIMIZER_STEP_CALLS)
        state["ppo_update_calls_in_process"] = int(r42_run._PPO_UPDATE_CALLS)
        if r42_run._RUNTIME is not None:
            state["execution_ledger"] = r42_run.ledger_snapshot(strict=False)
    except BaseException as ledger_error:
        state["ledger_snapshot_error"] = repr(ledger_error)
    _write_atomic(RUN_DIR / "ROUND42_STOP_STATE.json", state)
    report = [
        "# ROUND42-R stopped execution report",
        "",
        f"- Stop reason: `{state['exception']}`",
        f"- Old ROUND42 constructor failure preserved: {len(old_failed)}",
        f"- New technical failures: {len(new_failed)}",
        f"- Normal claims including carryover: {normal_used} / 672",
        f"- Technical reserve claims: {reserve_used} / 28",
        f"- Cumulative constructor calls: {constructor_calls}",
        f"- Cumulative JointRun constructions: {constructed_returns}",
        f"- `run()` invocations / underlying calls: {run_calls} / {joint_run_calls}",
        f"- Completed steps / successful logical jobs: {completed_steps} / {len(completed_jobs)}",
        f"- Simulation starts with at least one completed step: {simulation_started}",
        f"- PPO updates / Adam optimizer steps: {state.get('ppo_update_calls_in_process', 0)} / {state.get('optimizer_step_hook_calls_in_process', 0)}",
        "- Failed claims remain failed; no automatic retry or resume was attempted.",
        "- Raw rows, lifecycle files and checkpoints remain in this isolated run directory.",
        "",
    ]
    (RUN_DIR / "ROUND42_STOP_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8", newline="\n")
    review_ledger = state.get("execution_ledger")
    if review_ledger is None:
        ledger_path = RUN_DIR / "runtime" / "ROUND42_EXECUTION_LEDGER.json"
        if ledger_path.is_file():
            review_ledger = _read(ledger_path)
    if review_ledger is not None:
        _write_atomic(RUN_DIR / "ROUND42R_FINAL_LEDGER.json", review_ledger)
    logical_jobs_path = RUN_DIR / "runtime" / "ROUND42R_LOGICAL_JOB_LEDGER.json"
    if logical_jobs_path.is_file():
        _write_atomic(RUN_DIR / "ROUND42R_LOGICAL_JOB_LEDGER.json",
                      _read(logical_jobs_path))
    retry_gate_path = RUN_DIR / "runtime" / "ROUND42R_REFERENCE_RETRY_GATE.json"
    if retry_gate_path.is_file():
        _write_atomic(RUN_DIR / "ROUND42R_REFERENCE_RETRY_GATE.json", _read(retry_gate_path))


def main() -> int:
    result: dict[str, Any] | None = None
    try:
        result = _execute()
    except BaseException as error:
        failure = {
            "kind": "round42-execution-stop/1",
            "status": "stopped_without_automatic_retry",
            "exception": f"{type(error).__name__}: {error}",
            "time_unix": time.time(),
            "traceback": __import__("traceback").format_exc(),
        }
        try:
            _write_atomic(RUN_DIR / "runtime" / "ROUND42_EXECUTION_STOP.json", failure)
        except BaseException:
            pass
        try:
            write_execution_ledger(strict=False)
        except BaseException:
            pass
        try:
            _record_stopped_state(error)
        except BaseException:
            pass
        raise
    finally:
        restore_runtime_hooks()
    if result is None:
        raise RuntimeError("ROUND42 executor returned no result")
    print(json.dumps({
        "status": result["status"],
        "prelaunch_head": result["prelaunch_head"],
        "launch_counts": result["execution_ledger"]["counts"],
        "report": str(RUN_DIR / "ROUND42_REPORT.md"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
