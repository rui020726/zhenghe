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
    plan = frozen["round42"]["plan"]
    started = time.time()

    references = [*range(1_300_000, 1_300_048), *DEV_SCENARIOS]
    for index, scene in enumerate(references, start=1):
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
    if (counts["normal_used"] != 672 or counts["reserve_used"] != 0
            or counts["total_used"] != 672
            or counts["environment_constructor_calls"] != 672
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
        "first_batch_gates": gates,
        "training_reports": training_reports,
        "paired_results": paired,
        "execution_ledger": ledger,
    }
    _write_atomic(RUN_DIR / "runtime" / "ROUND42_EXECUTION_RESULT.json", result)
    write_final_deliverables(runtime=runtime, frozen=frozen, result=result)
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
    normal_used = len(list((RUN_DIR / "launch_slots" / "normal").glob("*.json")))
    reserve_used = len(list((RUN_DIR / "launch_slots" / "reserve").glob("*.json")))
    constructor_calls = sum(int(row.get("environment_constructor_calls", 0)) for row in claims)
    simulation_started = sum(bool(row.get("simulation_started")) for row in claims)
    failed = [row for row in claims if row.get("status") == "failed_technical"]
    state = {
        "kind": "round42-stopped-execution-state/1",
        "status": "stopped_without_automatic_retry",
        "exception": f"{type(error).__name__}: {error}",
        "normal_launch_claims": normal_used,
        "reserve_launch_claims": reserve_used,
        "total_launch_claims": normal_used + reserve_used,
        "environment_constructor_calls": constructor_calls,
        "simulation_started_calls": simulation_started,
        "failed_claims": failed,
        "claim_count": len(claims),
        "no_automatic_retry_or_resume": True,
        "traceback": __import__("traceback").format_exc(),
    }
    try:
        import r42_run
        state["optimizer_step_hook_calls_in_process"] = int(r42_run._OPTIMIZER_STEP_CALLS)
        if r42_run._RUNTIME is not None:
            state["execution_ledger"] = r42_run.ledger_snapshot(strict=False)
    except BaseException as ledger_error:
        state["ledger_snapshot_error"] = repr(ledger_error)
    _write_atomic(RUN_DIR / "ROUND42_STOP_STATE.json", state)
    report = [
        "# ROUND42 stopped execution report",
        "",
        f"- Stop reason: `{state['exception']}`",
        f"- Environment constructor calls: {constructor_calls}",
        f"- Simulation starts: {simulation_started}",
        f"- Normal claims used: {normal_used} / 672",
        f"- Technical reserve claims used: {reserve_used} / 28",
        f"- Technical failed claims: {len(failed)}",
        "- Automatic retry/resume: none",
        "- Environment outcomes, if any, remain in the Git-ignored raw artifact directories; their paths and hashes must be inventoried before any later recovery decision.",
        "- The failed gate/runtime condition must be reviewed before authorizing another run.",
        "",
    ]
    (RUN_DIR / "ROUND42_STOP_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8", newline="\n")
    status_path = RUN_DIR.parents[1] / "PROJECT_STATUS.md"
    marker = "## ROUND42 — execution stopped"
    if status_path.is_file():
        existing = status_path.read_text(encoding="utf-8")
        if marker not in existing and "## ROUND42 — one-shot SEND/SKIP first-report selection" not in existing:
            status_path.write_text(
                existing.rstrip() + "\n\n" + marker + "\n\n"
                + f"- Local branch: `{_read(RUN_DIR / 'ROUND42_FROZEN_CONFIG.json')['execution_identity']['branch']}`.\n"
                + f"- Execution halted before automatic retry: {state['exception']}.\n"
                + f"- Claims: {normal_used} normal + {reserve_used} reserve; environment constructors={constructor_calls}; simulations started={simulation_started}.\n"
                + "- Review `runs/comm_one_shot_report_selection_20261003/ROUND42_STOP_REPORT.md`; no continuation was launched.\n",
                encoding="utf-8", newline="\n")


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
