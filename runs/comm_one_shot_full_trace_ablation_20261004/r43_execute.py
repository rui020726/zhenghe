"""Bounded ROUND43 execution: three 128-episode seeds and fixed ep128 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

import r42_analysis
import r42_run
import r43_analysis
import r43_report


FROZEN_PATH = RUN_DIR / "ROUND43_FROZEN_CONFIG.json"
PREFLIGHT_PATH = RUN_DIR / "ROUND43_PREFLIGHT.json"
PRELAUNCH_PATH = RUN_DIR / "runtime" / "ROUND43_PRELAUNCH_IDENTITY.json"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_new_or_match(path: Path, value: Any) -> None:
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"ROUND43 stage receipt conflicts with existing data: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def _stage_summary(report: dict[str, Any], *, stage: str) -> dict[str, Any]:
    return {
        "kind": "round43-training-stage-receipt/1",
        "stage": stage,
        "training_seed": int(report["training_seed"]),
        "episodes_done": int(report["episodes_done"]),
        "updates": int(report["updates"]),
        "optimizer_steps": int(report["optimizer_steps"]),
        "lambda": float(report["lambda"]),
        "training_version": report["training_version"],
        "comm_action_eligibility": report["comm_action_eligibility"],
        "checkpoint": report["checkpoint"],
        "controller_parameters": report["controller_parameters"],
        "history": report["history"],
        "rng_priming": report.get("rng_priming"),
    }


def _claim_status(run: dict[str, Any], logical_job: str, attempt: int) -> str | None:
    slug = run["r40_common"]._slug(logical_job)
    path = RUN_DIR / "launch_claims" / slug / f"attempt_{attempt:04d}.json"
    return _read(path).get("status") if path.is_file() else None


def _completed_eval_row(runtime: dict[str, Any], seed: int, mode: str,
                        scene: int) -> dict[str, Any] | None:
    matches = []
    logical = f"model_eval_new_s{seed}_{mode}_scene{scene}"
    claim_dir = RUN_DIR / "launch_claims" / runtime["r40_common"]._slug(logical)
    completed_attempts = set()
    if claim_dir.is_dir():
        for path in claim_dir.glob("attempt_*.json"):
            try:
                attempt = int(path.stem.split("_")[-1])
            except ValueError:
                continue
            if _read(path).get("status") == "complete":
                completed_attempts.add(attempt)
    for path in runtime["common"].ROWS.glob("*.json"):
        row = _read(path)
        if (row.get("kind") == "eval" and row.get("model_arm") == "NEW"
                and int(row.get("training_seed", -1)) == seed
                and row.get("eval_mode") == mode
                and int(row.get("scenario_seed", -1)) == scene
                and int(row.get("attempt", -1)) in completed_attempts):
            matches.append(row)
    if len(matches) > 1:
        raise RuntimeError(f"duplicate successful ROUND43 evaluation cells: {seed}/{mode}/{scene}")
    return matches[0] if matches else None


def _committed_training_rows(runtime: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Return only training rows referenced by the 32 immutable committed PPO batches."""
    common = runtime["common"]
    selected: dict[int, dict[str, Any]] = {}
    for batch in range(32):
        commit_path = common.CHECKPOINTS / f"r40_batch_commit_s{seed}_b{batch}.json"
        if not commit_path.is_file():
            raise RuntimeError(f"seed {seed} batch {batch}: immutable commit receipt missing")
        commit = _read(commit_path)
        record_path = common.CHECKPOINTS / commit["batch_record"]
        if not record_path.is_file() or _sha256(record_path) != commit["batch_record_sha256"]:
            raise RuntimeError(f"seed {seed} batch {batch}: committed batch record hash mismatch")
        record = _read(record_path)
        if int(record.get("batch", -1)) != batch or int(record.get("training_seed", -1)) != seed:
            raise RuntimeError(f"seed {seed} batch {batch}: committed record identity mismatch")
        for artifact in record["episode_artifacts"]:
            episode = int(artifact["episode_index"])
            row_path = Path(artifact["row_path"])
            if not row_path.is_file() or _sha256(row_path) != artifact["row_sha256"]:
                raise RuntimeError(f"seed {seed} episode {episode}: committed row hash mismatch")
            row = _read(row_path)
            logical_job = f"new_training_s{seed}_e{episode}"
            claim_path = (RUN_DIR / "launch_claims"
                          / runtime["r40_common"]._slug(logical_job)
                          / f"attempt_{int(row['attempt']):04d}.json")
            if not claim_path.is_file() or _read(claim_path).get("status") != "complete":
                raise RuntimeError(f"seed {seed} episode {episode}: committed training row lacks a complete claim")
            if episode in selected:
                raise RuntimeError(f"seed {seed} episode {episode}: multiple committed training rows")
            selected[episode] = row
    if set(selected) != set(range(128)):
        raise RuntimeError(f"seed {seed}: committed batch artifacts do not cover episodes 0..127")
    return [selected[episode] for episode in range(128)]


def _validate_training_completion(runtime: dict[str, Any], frozen: dict[str, Any],
                                   seed: int) -> dict[str, Any]:
    checkpoints = runtime["common"].CHECKPOINTS
    rolling = checkpoints / f"v2_train_s{seed}.pt"
    if not rolling.is_file():
        raise RuntimeError(f"seed {seed}: rolling ep128 checkpoint is missing")
    payload = r42_run.torch_load_cpu(rolling)
    state = payload["training_state"]
    if (payload.get("training_version") != frozen["round42"]["training_version"]
            or int(state.get("training_seed", -1)) != seed
            or int(state.get("episodes_done", -1)) != 128
            or int(state.get("updates", -1)) != 32
            or float(state.get("gae_lambda", -1)) != 1.0
            or float(state.get("gamma", -1)) != 1.0):
        raise RuntimeError(f"seed {seed}: rolling checkpoint is not the frozen ROUND43 ep128 state")
    seed_rows = _committed_training_rows(runtime, seed)
    episodes = {int(row["episode_index"]): row for row in seed_rows}
    order = list(runtime["common"].scenario_order())
    if len(seed_rows) != 128 or set(episodes) != set(range(128)):
        raise RuntimeError(f"seed {seed}: training rollout rows are incomplete or duplicated")
    for episode, row in episodes.items():
        if (int(row["scenario_seed"]) != int(order[episode])
                or row.get("training_version") != frozen["round42"]["training_version"]
                or row.get("comm_action_semantics") != "one-shot-send-skip/1"):
            raise RuntimeError(f"seed {seed} episode {episode}: row identity/order mismatch")
    milestones = {}
    for episode in (0, 64, 128):
        candidates = list(checkpoints.glob(f"v2_train_s{seed}_ep{episode}*.pt"))
        if len(candidates) != 1:
            raise RuntimeError(f"seed {seed} ep{episode}: milestone count is {len(candidates)}")
        milestones[str(episode)] = {"path": str(candidates[0]),
                                    "sha256": _sha256(candidates[0])}
    return {
        "training_seed": seed,
        "episodes_done": int(state["episodes_done"]),
        "updates": int(state["updates"]),
        "optimizer_steps": int(state["optimizer_steps"]),
        "lambda": float(state["lambda"]),
        "training_version": payload["training_version"],
        "rolling_checkpoint_sha256": _sha256(rolling),
        "milestones": milestones,
        "training_rows": len(seed_rows),
    }


def run(*, resume: bool = True) -> dict[str, Any]:
    frozen = _read(FROZEN_PATH)
    if frozen.get("status") != "prelaunch_frozen":
        raise RuntimeError("ROUND43 freeze is not marked prelaunch_frozen")
    preflight = _read(PREFLIGHT_PATH)
    if preflight.get("status") != "passed" or preflight.get("draft_mode") is not False:
        raise RuntimeError("ROUND43 actual-code prelaunch correctness gate has not passed")
    current_head = r42_run.verify_frozen_source(frozen)
    if not PRELAUNCH_PATH.is_file():
        raise RuntimeError("ROUND43 exact prelaunch identity receipt is missing")
    prelaunch = _read(PRELAUNCH_PATH)
    if prelaunch.get("head") != current_head:
        raise RuntimeError("ROUND43 current HEAD differs from prelaunch identity receipt")
    runtime = r42_run.configure(
        frozen,
        _read(Path(frozen["execution_identity"]["round40_worktree"]) / "runs" /
              "comm_first_report_training_ablation_20261003" /
              "ROUND40_FROZEN_CONFIG.json")["round40"]["freeze_identity"],
    )
    order = list(runtime["common"].scenario_order())
    if order != [int(x) for x in frozen["round42"]["training_order"]]:
        raise RuntimeError("ROUND43 scenario order changed after prelaunch gate")

    progress_path = RUN_DIR / "runtime" / "ROUND43_EXECUTION_PROGRESS.json"
    progress = {
        "kind": "round43-execution-progress/1", "status": "running",
        "prelaunch_head": current_head, "started_at_unix": time.time(),
        "training": {}, "evaluations_completed": 0,
    }
    if progress_path.is_file():
        old_progress = _read(progress_path)
        if old_progress.get("prelaunch_head") != current_head:
            raise RuntimeError("existing ROUND43 progress belongs to a different source HEAD")
        progress.update(old_progress)
        if progress.get("status") == "complete":
            raise RuntimeError("ROUND43 progress already says complete; inspect final artifacts")
    _write_atomic(progress_path, progress)

    round40_freeze_path = (Path(frozen["execution_identity"]["round40_worktree"])
                           / "runs" / "comm_first_report_training_ablation_20261003"
                           / "ROUND40_FROZEN_CONFIG.json")
    # Training proceeds seed by seed: each actual first four-episode PPO batch is gated before
    # the same frozen seed is resumed for its remaining 124 episodes.
    import r42_analysis as analysis

    training_reports = RUN_DIR / "training_reports"
    training_reports.mkdir(parents=True, exist_ok=True)
    for seed in r42_run.SEEDS:
        first4_receipt = training_reports / f"ROUND43_SEED{seed}_FIRST4.json"
        final_receipt = training_reports / f"ROUND43_SEED{seed}_EP128.json"
        batch0_commit = runtime["common"].CHECKPOINTS / f"r40_batch_commit_s{seed}_b0.json"
        if not batch0_commit.is_file():
            print(f"ROUND43 seed {seed}: executing first four episodes and batch-0 gate", flush=True)
            first4 = r42_run.train_seed(seed, episodes=4, frozen=frozen)
            if int(first4["episodes_done"]) != 4 or int(first4["updates"]) != 1:
                raise RuntimeError(f"seed {seed}: first four episodes did not commit batch 0")
            _write_new_or_match(first4_receipt, _stage_summary(first4, stage="first_four"))
        elif not first4_receipt.is_file():
            print(f"ROUND43 seed {seed}: batch 0 is already committed; writing its gate receipt", flush=True)
        gate_path = RUN_DIR / f"ROUND43_FIRST_BATCH_GATE_SEED{seed}.json"
        if gate_path.is_file():
            gate = _read(gate_path)
            if gate.get("status") != "passed" or int(gate.get("training_seed", -1)) != seed:
                raise RuntimeError(f"seed {seed}: existing first-batch gate is not a pass")
        else:
            gate = analysis.first_batch_gate(
                seed, runtime=runtime, frozen=frozen)
            _write_new_or_match(gate_path, gate)
        if not first4_receipt.is_file():
            # A committed batch record is sufficient to reconstruct the receipt without an
            # extra environment. Use its immutable state and parameter identities.
            record = _read(runtime["common"].CHECKPOINTS /
                           _read(batch0_commit)["batch_record"])
            _write_new_or_match(first4_receipt, {
                "kind": "round43-training-stage-receipt/1", "stage": "first_four",
                "training_seed": seed, "episodes_done": 4, "updates": 1,
                "optimizer_steps": int(record["update"]["optimizer_steps"]),
                "lambda": float(record["dual"]["lambda_next"]),
                "training_version": frozen["round42"]["training_version"],
                "comm_action_eligibility": "first-report-only/1",
                "checkpoint": str(runtime["common"].CHECKPOINTS /
                                   f"v2_train_s{seed}.pt"),
                "controller_parameters": record["controller_parameters_after"],
                "history": [], "rng_priming": {
                    "source": "ROUND43 same-version checkpoint resume",
                    "loaded_model": True, "loaded_optimizer": True,
                },
            })
        print(f"ROUND43 seed {seed}: first-batch gate passed; continuing to ep128", flush=True)
        if not final_receipt.is_file():
            report = r42_run.train_seed(seed, episodes=128, frozen=frozen)
            if int(report["episodes_done"]) != 128 or int(report["updates"]) != 32:
                raise RuntimeError(f"seed {seed}: training did not reach ep128/32 batches")
            _write_new_or_match(final_receipt, _stage_summary(report, stage="ep128"))
        seed_final = _validate_training_completion(runtime, frozen, seed)
        progress["training"][str(seed)] = {
            "status": "complete", **seed_final,
            "first_batch_gate": str(gate_path),
            "first_batch_gate_sha256": _sha256(gate_path),
            "first4_receipt": str(first4_receipt),
            "ep128_receipt": str(final_receipt),
        }
        progress["updated_at_unix"] = time.time()
        _write_atomic(progress_path, progress)
        r42_run.write_execution_ledger(strict=True)
        print(f"ROUND43 seed {seed}: ep128 complete, updates=32, optimizer_steps="
              f"{seed_final['optimizer_steps']}, lambda={seed_final['lambda']:.6f}", flush=True)

    # Evaluate only each immutable ROUND43 ep128 checkpoint on the fixed 16-scene block.
    # OLD is the read-only ROUND42 NEW ep128 matrix; it is reused without new launches.
    for seed in r42_run.SEEDS:
        controller, identity = r42_run.load_eval_model("NEW", seed)
        for mode in ("greedy", "sampled"):
            for scene in r42_run.DEV_SCENARIOS:
                row = _completed_eval_row(runtime, seed, mode, scene)
                if row is None:
                    print(f"ROUND43 eval NEW s{seed}/{mode} scene={scene}", flush=True)
                    row = r42_run.run_model_eval(
                        arm="NEW", training_seed=seed, mode=mode, scenario=scene,
                        controller=controller, identity=identity)
                if (row.get("training_version") != frozen["round42"]["training_version"]
                        or row.get("checkpoint_parameter_identity", {}).get("parameter_sha256")
                        != identity["parameter_identity"]["parameter_sha256"]):
                    raise RuntimeError(f"ROUND43 eval row/checkpoint identity mismatch: {seed}/{mode}/{scene}")
                progress["evaluations_completed"] = sum(
                    _completed_eval_row(runtime, s, m, sc) is not None
                    for s in r42_run.SEEDS for m in ("greedy", "sampled")
                    for sc in r42_run.DEV_SCENARIOS)
                progress["updated_at_unix"] = time.time()
                _write_atomic(progress_path, progress)
        print(f"ROUND43 seed {seed}: fixed greedy/sample evaluation complete", flush=True)

    ledger = r42_run.write_execution_ledger(strict=True)
    counts = ledger["counts"]
    completed_claims = [claim for claim in ledger["claims"]
                        if claim.get("status") == "complete"]
    if (int(counts["normal_used"]) != 480
            or int(counts["reserve_used"]) > 24
            or int(counts["total_used"]) > 600
            or int(counts["completed_logical_jobs"]) != 480
            or int(counts["ppo_update_calls"]) != 96
            or not counts["optimizer_step_count_matches_committed_batches"]
            or len(completed_claims) < 480):
        raise RuntimeError(f"ROUND43 final execution ledger does not close: {counts}")
    for claim in completed_claims:
        if (int(claim.get("environment_constructor_calls", 0)) != 1
                or int(claim.get("joint_run_constructed_calls", 0)) != 1
                or int(claim.get("run_invocation_calls", 0)) != 1
                or int(claim.get("joint_run_method_calls", 0)) != 1
                or int(claim.get("run_return_calls", 0)) != 1
                or int(claim.get("completed_steps", 0)) != 1000):
            raise RuntimeError(f"successful environment claim has an incomplete run count: {claim}")

    analysis_result = r43_analysis.write_analysis(runtime, frozen)
    report_result = r43_report.write_final_report(
        frozen=frozen, analysis=analysis_result, ledger=ledger, progress=progress)
    progress["status"] = "complete"
    progress["finished_at_unix"] = time.time()
    progress["evaluations_completed"] = 96
    progress["final_ledger_counts"] = counts
    _write_atomic(progress_path, progress)
    return {"ledger": ledger, "analysis": analysis_result, "progress": progress,
            "report": report_result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true",
                        help="refuse an existing progress receipt")
    args = parser.parse_args()
    if args.no_resume and (RUN_DIR / "runtime" / "ROUND43_EXECUTION_PROGRESS.json").exists():
        raise RuntimeError("--no-resume requested but ROUND43 progress already exists")
    result = run(resume=not args.no_resume)
    print(json.dumps({
        "status": result["progress"]["status"],
        "ledger_counts": result["ledger"]["counts"],
        "scientific_result": result["analysis"]["machine_verdict"]["scientific_result"],
        "greedy_joint_improvement": result["analysis"]["machine_verdict"][
            "greedy_primary"]["joint_improvement_by_registered_development_rule"],
        "report_path": result["report"]["report_path"],
        "report_sha256": result["report"]["report_sha256"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
