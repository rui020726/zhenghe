"""Thin ROUND45-R2 adapter over the frozen R44/R43 execution and training seams.

This module owns R2 identities, durable key consumption, BC model injection, and
orchestration. Environment transitions, PPO/GAE math, costs, one-shot collection,
and checkpoint serialization remain in the verified R44/R43/R40 modules.
"""

from __future__ import annotations

import contextvars
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator


R2_DIR = Path(__file__).resolve().parent
REPO = R2_DIR.parents[1]
RUN_DIR = R2_DIR
R44_DIR = REPO / "runs" / "comm_dual_response_ablation_20261004"
R43_DIR = REPO / "runs" / "comm_one_shot_full_trace_ablation_20261004"
ROUND42_OUTPUT = REPO / "runs" / "comm_one_shot_gate_resume_20261003_v2"
ROUND40_ROOT = REPO.parent / "zhenghe-round40"
ROUND40_OUTPUT = ROUND40_ROOT / "runs" / "comm_first_report_training_ablation_20261003"
INPUT_ROOT = REPO.parent / "zhenghe"
R44_FINAL_HEAD = "cbd437afad6ddc3a7a68c8b7e10c0cd572a778dd"
R44_MATRIX_AUDIT_SHA256 = "16e8c9d628ca2bd8351b21a6e9fcccac335ecd6404136e41761cbbf51dda45bc"
RUNNER_VERSION = (
    "counted-jointrun-proxy/1+eval-key-index/1+round45-bc-warmstart/1+"
    "round45-bc-nested-controller-state/2+round45-postbc-readback/1+"
    "round45-attempt/2+r2-durable-key-consumer/1"
)

CURRENT_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "round45_r2_key_context", default=None)
RUNTIME: dict[str, Any] | None = None
R44_RUNTIME_MODULE: Any = None
R44_INDEX_MODULE: Any = None
R2_FREEZE: dict[str, Any] | None = None
SEAM_TEST_MODE = False
ENFORCE_PRELAUNCH = False


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_atomic(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _same_value(left: Any, right: Any) -> bool:
    """Compare nested Python RNG state, including NumPy arrays and Torch tensors."""
    try:
        import numpy as np
        import torch
    except ImportError:
        np = None
        torch = None
    if torch is not None and isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if np is not None and isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and bool(np.array_equal(left, right))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _same_value(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return (isinstance(right, type(left)) and len(left) == len(right)
                and all(_same_value(a, b) for a, b in zip(left, right)))
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _source_hashes() -> dict[str, str]:
    paths = {
        "r45_r2_execute.py": R2_DIR / "r45_r2_execute.py",
        "r45_bc.py": R2_DIR / "r45_bc.py",
        "r45_data.py": R2_DIR / "r45_data.py",
        "r45_preflight.py": R2_DIR / "r45_preflight.py",
        "r40_train_resume_adapter.py": R2_DIR / "r40_train_resume_adapter.py",
        "r42_one_shot.py": R2_DIR / "r42_one_shot.py",
        "r42_versions.py": R2_DIR / "r42_versions.py",
        "r43_call_counter_hooks.py": R2_DIR / "r43_call_counter_hooks.py",
        "r44_runtime.py": R44_DIR / "r44_runtime.py",
        "r44_prepare.py": R44_DIR / "r44_prepare.py",
        "r44_eval_index.py": R44_DIR / "r44_eval_index.py",
        "r44_resume_bridge.py": R44_DIR / "r44_resume_bridge.py",
        "r43_r42_run.py": R43_DIR / "r42_run.py",
        "r43_r42_analysis.py": R43_DIR / "r42_analysis.py",
        "r40_common.py": ROUND40_OUTPUT / "r40_common.py",
        "r40_run.py": ROUND40_OUTPUT / "r40_run.py",
        "r40_train.py": ROUND40_OUTPUT / "r40_train.py",
        "r40_versions.py": ROUND40_OUTPUT / "r40_versions.py",
        "canonical_v2_run.py": INPUT_ROOT / "runs" / "comm_cost_v2_rerun_20260928" / "v2_run.py",
        "canonical_v2_trainer.py": INPUT_ROOT / "runs" / "comm_cost_v2_rerun_20260928" / "v2_trainer.py",
        "canonical_v2_controller.py": INPUT_ROOT / "runs" / "comm_cost_v2_rerun_20260928" / "v2_controller.py",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"R2 runtime source file(s) missing: {missing}")
    return {name: _sha256(path) for name, path in paths.items()}


def verify_prelaunch_identity(_frozen: dict[str, Any] | None = None) -> str:
    """Fail closed on source, baseline, BC, or prelaunch identity drift."""
    receipt_path = RUN_DIR / "ROUND45_R2_PRELAUNCH_IDENTITY.json"
    freeze_path = RUN_DIR / "ROUND45_R2_POST_BC_ENV_FREEZE.json"
    if not receipt_path.is_file() or not freeze_path.is_file():
        raise RuntimeError("R2 prelaunch identity or post-BC freeze is missing")
    receipt = _json(receipt_path)
    frozen = _json(freeze_path)
    if receipt.get("status") != "passed" or frozen.get("status") != "frozen_before_environment":
        raise RuntimeError("R2 prelaunch/freeze status is not passed")
    if _sha256(freeze_path) != receipt.get("post_bc_freeze_sha256"):
        raise RuntimeError("R2 post-BC freeze SHA256 changed")
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != frozen.get("branch") or branch != "exp/comm-rule-warmstart-cost-learning-r2":
        raise RuntimeError(f"R2 branch identity mismatch: {branch}")
    if _git("rev-parse", "HEAD^") != receipt.get("freeze_commit"):
        raise RuntimeError("R2 HEAD is not the frozen prelaunch receipt child")
    allowed_dirty = {str((RUN_DIR / "ROUND45_CURRENT_TASK.json").resolve()).lower()}
    dirty = []
    for line in _git("status", "--porcelain", "--untracked-files=no").splitlines():
        raw = line[3:].strip().strip('"')
        path = (REPO / raw).resolve()
        if str(path).lower() not in allowed_dirty:
            dirty.append(line)
    if dirty:
        raise RuntimeError(f"R2 frozen tracked source changed: {dirty[:4]}")
    actual_sources = _source_hashes()
    if actual_sources != frozen.get("source_files_sha256"):
        changed = sorted(key for key in set(actual_sources) | set(
            frozen.get("source_files_sha256", {}))
                         if actual_sources.get(key) != frozen.get("source_files_sha256", {}).get(key))
        raise RuntimeError(f"R2 frozen source SHA256 mismatch: {changed}")
    audit_path = R44_DIR / "ROUND44_EVAL_MATRIX_AUDIT.json"
    if _sha256(audit_path) != R44_MATRIX_AUDIT_SHA256:
        raise RuntimeError("R44 final 96-cell audit SHA256 changed")
    baseline_path = R44_DIR / "ROUND44_BASELINE_REUSE.json"
    if _sha256(baseline_path) != frozen.get("round44_baseline_reuse_sha256"):
        raise RuntimeError("R44 fixed baseline reuse manifest SHA256 changed")
    reference_receipt = RUN_DIR / "runtime" / "ROUND45_R2_REFERENCE_CACHE_REUSE.json"
    if (not reference_receipt.is_file()
            or _sha256(reference_receipt)
            != frozen.get("r2_reference_cache_reuse_receipt_sha256")):
        raise RuntimeError("R2 immediate-reference reuse receipt changed")
    if _git("rev-parse", "exp/comm-dual-response-ablation") != R44_FINAL_HEAD:
        raise RuntimeError("R44 final delivery ref changed")
    task = _json(RUN_DIR / "ROUND45_CURRENT_TASK.json")
    if (task.get("branch") != branch or task.get("attempt") != "ROUND45-R/fresh-seed-attempt/2"
            or task.get("this_attempt_bc_update_cap") != 3000
            or task.get("family_cumulative_bc_update_hard_cap") != 4000):
        raise RuntimeError("R2 mutable task state changed a frozen attempt/cap identity")
    pre_bc = _json(RUN_DIR / "ROUND45_PRE_BC_FREEZE.json")
    expected_python = Path(pre_bc["python"]["executable"]).resolve()
    if (Path(sys.executable).resolve() != expected_python
            or RUNTIME is None
            or str(RUNTIME["common"].DEVICE) != pre_bc["python"]["device"]
            or str(RUNTIME["common"].DEVICE) != "cpu"):
        raise RuntimeError("R2 execution interpreter/device differs from its pre-BC freeze")
    for item in frozen["bc_models"]:
        checkpoint = Path(item["checkpoint_path"])
        if not checkpoint.is_file() or _sha256(checkpoint) != item["checkpoint_sha256"]:
            raise RuntimeError(f"post-BC seed {item['seed']} checkpoint hash changed")
    return head


def _set_context(context: dict[str, Any]):
    return CURRENT_CONTEXT.set(context)


def _attempts(logical_job: str) -> list[tuple[Path, dict[str, Any]]]:
    assert RUNTIME is not None
    common = RUNTIME["r40_common"]
    directory = Path(common.ROUND_DIR) / "launch_claims" / common._slug(logical_job)
    if not directory.is_dir():
        return []
    result = []
    for path in sorted(directory.glob("attempt_*.json")):
        result.append((path, _json(path)))
    return result


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"R2 completed row artifact is missing: {path}")
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.append(json.loads(line))
    return result


def _verify_receipt_artifacts(claim: dict[str, Any], row: dict[str, Any]) -> None:
    assert R44_INDEX_MODULE is not None
    receipt = claim.get("round45_artifact_receipt")
    if not isinstance(receipt, dict) or receipt.get("kind") != "round45-r2-artifact-receipt/1":
        raise RuntimeError("completed R2 cell lacks its durable artifact receipt")
    if receipt.get("attempt") != claim.get("attempt"):
        raise RuntimeError("R2 artifact receipt attempt mismatch")
    if receipt.get("training_version") != RUNTIME["r42_versions"].TRAINING_VERSION:
        raise RuntimeError("R2 artifact receipt training version mismatch")
    if receipt.get("runner_version") != RUNNER_VERSION:
        raise RuntimeError("R2 artifact receipt runner version mismatch")
    artifacts = row.get("artefacts")
    if not isinstance(artifacts, dict) or set(receipt.get("artifacts", {})) != {
            "row", "lifecycle", "transcript", "tape"}:
        raise RuntimeError("R2 artifact map/receipt is incomplete")
    for name in ("row", "lifecycle", "transcript", "tape"):
        path = Path(artifacts[name]).resolve()
        actual = {"path": str(path), "sha256": R44_INDEX_MODULE._sha256(path),
                  "size_bytes": path.stat().st_size}
        if receipt["artifacts"].get(name) != actual:
            raise RuntimeError(f"R2 {name} artifact no longer matches its durable receipt")
        claimed = claim.get("round45_artifact_paths", {}).get(name)
        if claimed is None or Path(claimed).resolve() != path:
            raise RuntimeError(f"R2 claim {name} path differs from completed row")


def _verify_eval_receipt(claim: dict[str, Any], row: dict[str, Any]) -> None:
    if row.get("kind") != "eval":
        return
    receipt = claim.get("round44_eval_artifact_receipt")
    key = [int(row["training_seed"]), str(row["eval_mode"]), int(row["scenario_seed"])]
    if (not isinstance(receipt, dict)
            or receipt.get("kind") != "round44-eval-artifact-receipt/2"
            or receipt.get("eval_key") != key
            or receipt.get("attempt") != int(row["attempt"])
            or receipt.get("training_version") != RUNTIME["r42_versions"].TRAINING_VERSION
            or receipt.get("runner_version") != RUNNER_VERSION
            or receipt.get("checkpoint_parameter_sha256")
            != row.get("checkpoint_parameter_identity", {}).get("parameter_sha256")):
        raise RuntimeError("R2 evaluation lacks a valid R44 receipt identity")
    if set(receipt.get("artifacts", {})) != {"row", "lifecycle", "transcript", "tape"}:
        raise RuntimeError("R2 evaluation R44 receipt artifact set is incomplete")


def _completed_row(context: dict[str, Any], attempts: list[tuple[Path, dict[str, Any]]],
                   *, collector: Any = None, runner_kwargs: dict[str, Any] | None = None,
                   identity: dict[str, Any] | None = None) -> dict[str, Any] | None:
    complete = [(path, claim) for path, claim in attempts if claim.get("status") == "complete"]
    if not complete:
        return None
    if len(complete) != 1:
        raise RuntimeError(f"R2 logical key has multiple complete claims: {context['logical_job']}")
    claim_path, claim = complete[0]
    if claim is not attempts[-1][1] and claim_path != attempts[-1][0]:
        raise RuntimeError(f"R2 key has an attempt after its completed attempt: {context['logical_job']}")
    if (claim.get("round") != 45
            or claim.get("training_version") != RUNTIME["r42_versions"].TRAINING_VERSION
            or claim.get("runner_version") != RUNNER_VERSION
            or claim.get("comm_action_semantics") != "one-shot-send-skip/1"
            or claim.get("round45_key") != _serializable_key(context)):
        raise RuntimeError(f"R2 completed claim identity mismatch: {claim_path}")
    counts = claim
    if context.get("scope") != "seam_regression":
        expected_counts = {
            "environment_constructor_calls": 1, "run_invocation_calls": 1,
            "joint_run_method_calls": 1, "run_return_calls": 1,
            "run_failure_calls": 0, "completed_steps": 1000,
        }
        if any(int(counts.get(field, -1)) != value
               for field, value in expected_counts.items()):
            raise RuntimeError(f"R2 completed claim has incomplete real-run counts: {claim_path}")
    artifact_paths = claim.get("round45_artifact_paths")
    if not isinstance(artifact_paths, dict) or "row" not in artifact_paths:
        raise RuntimeError(f"R2 completed claim lacks its row path: {claim_path}")
    row_path = Path(artifact_paths["row"])
    row = _json(row_path)
    if int(row.get("attempt", -1)) != int(claim.get("attempt", -2)):
        raise RuntimeError("R2 completed row/claim attempt mismatch")
    if (row.get("training_version") != RUNTIME["r42_versions"].TRAINING_VERSION
            or row.get("runner_version") != RUNNER_VERSION
            or row.get("comm_action_semantics") != "one-shot-send-skip/1"):
        raise RuntimeError(f"R2 completed row version mismatch: {row_path}")
    if int(row.get("scenario_seed", -1)) != int(context["scenario_seed"]):
        raise RuntimeError(f"R2 completed row scene mismatch: {row_path}")
    if context.get("scope") == "ppo_training":
        if (int(row.get("training_seed", -1)) != int(context["seed"])
                or int(row.get("episode_index", -1)) != int(context["episode_index"])):
            raise RuntimeError(f"R2 recovered training row key mismatch: {row_path}")
        expected_lambda = context.get("lambda_at_rollout")
        if expected_lambda is not None and float(row.get("lambda_at_rollout", -1)) != float(expected_lambda):
            raise RuntimeError(f"R2 recovered training row lambda mismatch: {row_path}")
        controller = (runner_kwargs or {}).get("controller")
        if controller is not None:
            expected = controller.parameter_summary()["parameter_sha256"]
            saved = row.get("controller_parameters_after", {}).get("parameter_sha256")
            if saved != expected:
                raise RuntimeError("completed rollout was collected by a different controller state")
    elif row.get("kind") == "eval":
        if (int(row.get("training_seed", -1)) != int(context["seed"])
                or str(row.get("eval_mode")) != str(context["mode"])):
            raise RuntimeError(f"R2 recovered evaluation row key mismatch: {row_path}")
        expected_param = None
        if identity is not None:
            expected_param = identity.get("parameter_identity", {}).get("parameter_sha256")
        elif runner_kwargs and runner_kwargs.get("checkpoint_load_identity"):
            expected_param = runner_kwargs["checkpoint_load_identity"].get(
                "parameter_identity", {}).get("parameter_sha256")
        if expected_param and row.get("checkpoint_parameter_identity", {}).get(
                "parameter_sha256") != expected_param:
            raise RuntimeError("completed evaluation belongs to another checkpoint")
    _verify_receipt_artifacts(claim, row)
    _verify_eval_receipt(claim, row)
    if collector is not None:
        collector.records = _load_jsonl(Path(row["artefacts"]["tape"]))
    return row


def _serializable_key(context: dict[str, Any]) -> dict[str, Any]:
    fields = ("scope", "logical_job", "seed", "episode_index", "scenario_seed", "mode",
              "tag", "kind")
    return {key: context[key] for key in fields if key in context}


def _planned_key(context: dict[str, Any]) -> None:
    if context.get("scope") == "seam_regression" and SEAM_TEST_MODE:
        return
    if R2_FREEZE is None:
        raise RuntimeError("R2 immutable environment key freeze is not loaded")
    logical_jobs = R2_FREEZE["logical_key_plan"]["logical_jobs"]
    if context.get("logical_job") not in set(logical_jobs):
        raise RuntimeError(f"logical key is outside the frozen 528-key plan: {context}")


def _context_for_cell(kind: str, logical_job: str, scenario_seed: int,
                      metadata: dict[str, Any], runner_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    if kind == "train":
        return {
            "scope": "ppo_training", "logical_job": logical_job,
            "seed": int(metadata["training_seed"]),
            "episode_index": int(metadata["episode_index"]),
            "scenario_seed": int(scenario_seed), "kind": kind,
            "tag": str((runner_kwargs or {}).get("tag", "")),
            "allow_technical_retry": True,
        }
    if kind == "eval":
        return {
            "scope": "final_ep128_eval", "logical_job": logical_job,
            "seed": int(metadata["training_seed"]),
            "mode": str(metadata["mode"]), "scenario_seed": int(scenario_seed),
            "kind": kind, "tag": str((runner_kwargs or {}).get("tag", "")),
            "allow_technical_retry": True,
        }
    raise RuntimeError(f"R2 runner may not consume an unregistered environment kind: {kind}")


def _claim_paths(context: dict[str, Any], attempt: int) -> dict[str, str]:
    assert RUNTIME is not None
    paths = RUNTIME["common"].cell_paths(
        str(context["kind"]), int(context["scenario_seed"]), int(attempt),
        tag=str(context["tag"]))
    return {name: str(path.resolve()) for name, path in paths.items()}


def _record_claim(context: dict[str, Any], logical_job: str, attempt: int,
                  metadata: dict[str, Any], base_claim: Any) -> Path:
    path = base_claim(logical_job, attempt=attempt, metadata=metadata)
    assert RUNTIME is not None
    row_paths = _claim_paths(context, attempt)
    RUNTIME["r40_common"]._update_claim(
        path,
        round=45,
        training_version=RUNTIME["r42_versions"].TRAINING_VERSION,
        runner_version=RUNNER_VERSION,
        round45_key=_serializable_key(context),
        round45_artifact_paths=row_paths,
        round45_attempt_kind="normal" if attempt == 1 else "technical_reserve",
    )
    return path


def _record_artifact_receipt(kind: str, row: dict[str, Any], active_claim: Path,
                             base_booked: Any, label: str, ledger: Any) -> None:
    base_booked(label, kind, row, ledger)
    assert RUNTIME is not None and R44_INDEX_MODULE is not None
    paths = row.get("artefacts")
    if not isinstance(paths, dict):
        raise RuntimeError("completed R2 row lacks its artifact map")
    actual = {}
    for name in ("row", "lifecycle", "transcript", "tape"):
        path = Path(paths[name]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"R2 row artifact was not flushed: {path}")
        actual[name] = {"path": str(path), "sha256": R44_INDEX_MODULE._sha256(path),
                        "size_bytes": path.stat().st_size}
    receipt = {
        "kind": "round45-r2-artifact-receipt/1",
        "attempt": int(row["attempt"]),
        "training_version": RUNTIME["r42_versions"].TRAINING_VERSION,
        "runner_version": RUNNER_VERSION,
        "round45_key": _serializable_key(CURRENT_CONTEXT.get() or {}),
        "artifacts": actual,
    }
    RUNTIME["r40_common"]._update_claim(active_claim,
                                         round45_artifact_receipt=receipt)


def _install_runtime_adapters(runtime: dict[str, Any], *, enforcement: bool) -> None:
    global RUNTIME, R44_INDEX_MODULE, ENFORCE_PRELAUNCH
    RUNTIME = runtime
    ENFORCE_PRELAUNCH = bool(enforcement)
    run_module = runtime["r42_run"]
    r40_common = runtime["r40_common"]
    r44 = R44_RUNTIME_MODULE
    assert r44 is not None
    import r44_eval_index
    R44_INDEX_MODULE = r44_eval_index

    base_claim = run_module._claim

    def claim(logical_job: str, *, attempt: int, metadata: dict[str, Any]):
        context = CURRENT_CONTEXT.get()
        if context is None or context.get("logical_job") != logical_job:
            raise RuntimeError(f"R2 claim has no matching frozen key context: {logical_job}")
        _planned_key(context)
        if ENFORCE_PRELAUNCH:
            verify_prelaunch_identity(runtime["frozen"])
        return _record_claim(context, logical_job, attempt, metadata, base_claim)

    run_module._claim = claim

    base_write_ledger = run_module.write_execution_ledger

    def write_ledger(*, strict: bool = True) -> dict[str, Any]:
        ledger = run_module.ledger_snapshot(strict=strict)
        ledger["kind"] = "round45-r2-execution-ledger/1"
        ledger["round"] = 45
        ledger["runner_version"] = RUNNER_VERSION
        ledger["training_version"] = runtime["r42_versions"].TRAINING_VERSION
        path = RUN_DIR / "runtime" / "ROUND45_R2_EXECUTION_LEDGER.json"
        _write_atomic(path, ledger)
        return ledger

    run_module.write_execution_ledger = write_ledger

    base_execute = run_module._execute_cell

    def guarded_execute(label: str, logical_job: str, kind: str, scenario: int,
                        metadata: dict[str, Any], *, collector: Any = None,
                        runner_kwargs: dict[str, Any] | None = None,
                        row_updates: dict[str, Any] | None = None,
                        identity: dict[str, Any] | None = None) -> dict[str, Any]:
        context = CURRENT_CONTEXT.get()
        local_kwargs = dict(runner_kwargs or {})
        local_updates = dict(row_updates or {})
        if context is None:
            context = _context_for_cell(kind, logical_job, int(scenario), metadata,
                                        local_kwargs)
        elif context.get("scope") == "bc_only_greedy_control":
            logical_job = str(context["logical_job"])
            local_kwargs["tag"] = str(context["tag"])
            local_updates["r45_evaluation_role"] = "BC_ONLY_GREEDY_CONTROL"
            local_updates["bc_checkpoint_sha256"] = context["bc_checkpoint_sha256"]
            local_updates["bc_parameter_sha256"] = context["bc_parameter_sha256"]
        else:
            logical_job = str(context["logical_job"])
        context = dict(context)
        context.update({"logical_job": logical_job, "kind": kind,
                        "scenario_seed": int(scenario),
                        "tag": str(local_kwargs.get("tag", context.get("tag", "")))})
        _planned_key(context)
        prior = _attempts(logical_job)
        existing = _completed_row(context, prior, collector=collector,
                                  runner_kwargs=local_kwargs, identity=identity)
        if existing is not None:
            return existing
        if any(claim.get("status") not in ("failed_technical", "not_started_quota_exhausted")
               for _, claim in prior):
            raise RuntimeError(f"R2 key has a nonterminal claim; refusing replay: {logical_job}")
        failed = [(path, claim) for path, claim in prior
                  if claim.get("status") == "failed_technical"]
        if any(claim.get("status") == "not_started_quota_exhausted" for _, claim in prior):
            raise RuntimeError(f"R2 key previously failed before quota claim: {logical_job}")
        next_attempt = runtime["r40_common"].next_attempt(logical_job)
        if failed:
            if (len(failed) != 1 or int(failed[0][1].get("attempt", -1)) != 1
                    or next_attempt != 2 or not context.get("allow_technical_retry", False)):
                raise RuntimeError(f"R2 technical failure is not reserve-retry eligible: {logical_job}")
            claim = failed[0][1]
            if (claim.get("round45_artifact_receipt") is not None
                    or int(claim.get("run_return_calls", 0)) > 0
                    or int(claim.get("completed_steps", 0)) >= 1000):
                raise RuntimeError(f"R2 failed key already returned a completed episode: {logical_job}")
        elif prior or next_attempt != 1:
            raise RuntimeError(f"R2 logical key has unexplained prior attempts: {logical_job}")
        token = CURRENT_CONTEXT.set(context)
        try:
            try:
                return base_execute(
                    label, logical_job, kind, int(scenario), metadata,
                    collector=collector, runner_kwargs=local_kwargs,
                    row_updates=local_updates, identity=identity)
            except BaseException:
                failed = _attempts(logical_job)
                if (not context.get("allow_technical_retry", False)
                        or context.get("reserve_retry_used", False)
                        or len(failed) != 1
                        or int(failed[0][1].get("attempt", -1)) != 1
                        or failed[0][1].get("status") != "failed_technical"
                        or failed[0][1].get("round45_artifact_receipt") is not None
                        or int(failed[0][1].get("run_return_calls", 0)) > 0
                        or int(failed[0][1].get("completed_steps", 0)) >= 1000):
                    raise
                context["reserve_retry_used"] = True
                context["reserve_retry_reason"] = "single failed attempt before run return/completion"
                return base_execute(
                    label, logical_job, kind, int(scenario), metadata,
                    collector=collector, runner_kwargs=local_kwargs,
                    row_updates=local_updates, identity=identity)
        finally:
            CURRENT_CONTEXT.reset(token)

    run_module._execute_cell = guarded_execute

    base_booked = runtime["runner"]._booked

    def booked(label: str, kind: str, row: dict[str, Any], ledger: Any) -> None:
        active = run_module._ACTIVE_CLAIM
        if active is None:
            raise RuntimeError("R2 row receipt has no active atomic claim")
        _record_artifact_receipt(kind, row, active, base_booked, label, ledger)

    runtime["runner"]._booked = booked

    base_training_episode = runtime["r40_run"].run_training_episode

    def training_episode(common: Any, runner: Any, controller: Any, *,
                         training_seed: int, episode_index: int,
                         scenario_seed: int, lam: float) -> dict[str, Any]:
        context = {
            "scope": "ppo_training",
            "logical_job": f"new_training_s{int(training_seed)}_e{int(episode_index)}",
            "seed": int(training_seed), "episode_index": int(episode_index),
            "scenario_seed": int(scenario_seed), "kind": "train",
            "tag": f"r44_train_s{int(training_seed)}_e{int(episode_index)}",
            "lambda_at_rollout": float(lam), "allow_technical_retry": True,
        }
        token = CURRENT_CONTEXT.set(context)
        try:
            result = base_training_episode(
                common, runner, controller, training_seed=training_seed,
                episode_index=episode_index, scenario_seed=scenario_seed, lam=lam)
        finally:
            CURRENT_CONTEXT.reset(token)
        return result

    runtime["r40_run"].run_training_episode = training_episode


def load_runtime(*, enforce_prelaunch: bool = True) -> dict[str, Any]:
    """Configure the exact R44 runtime and attach only R2 path/version/claim adapters."""
    global RUNTIME, R44_RUNTIME_MODULE, R2_FREEZE, ENFORCE_PRELAUNCH
    if RUNTIME is not None:
        ENFORCE_PRELAUNCH = bool(enforce_prelaunch)
        if enforce_prelaunch:
            verify_prelaunch_identity(RUNTIME["frozen"])
            R2_FREEZE = _json(RUN_DIR / "ROUND45_R2_POST_BC_ENV_FREEZE.json")
        return RUNTIME
    if str(R2_DIR) not in sys.path:
        sys.path.insert(0, str(R2_DIR))
    if str(R44_DIR) not in sys.path:
        sys.path.insert(0, str(R44_DIR))
    import r44_prepare
    import r44_runtime

    configuration = _json(RUN_DIR / "ROUND45_CONFIGURATION_DIFF.json")
    training_version = str(configuration["training_version"])
    r44_prepare.ROOT = REPO
    r44_prepare.RUN_DIR = RUN_DIR
    r44_prepare.R43_DIR = R43_DIR
    r44_prepare.R42_DIR = ROUND42_OUTPUT
    r44_prepare.TRAINING_VERSION = training_version
    r44_prepare.RUNNER_VERSION = RUNNER_VERSION
    r44_runtime.RUN_DIR = RUN_DIR
    r44_runtime.ROOT = REPO
    r44_runtime.INPUT_ROOT = INPUT_ROOT
    r44_runtime.ROUND40_ROOT = ROUND40_ROOT
    r44_runtime.ROUND40_OUTPUT = ROUND40_OUTPUT
    r44_runtime.ROUND42_OUTPUT = ROUND42_OUTPUT
    r44_runtime.R43_FREEZE_PATH = R43_DIR / "ROUND43_FROZEN_CONFIG.json"
    r44_runtime.R44_FREEZE_PATH = R44_DIR / "ROUND44_FROZEN_CONFIG.json"
    r44_runtime.TRAINING_VERSION = training_version
    r44_runtime.RUNNER_VERSION = RUNNER_VERSION
    r44_runtime.verify_frozen_source = verify_prelaunch_identity
    R44_RUNTIME_MODULE = r44_runtime

    source_frozen = _json(R44_DIR / "ROUND44_FROZEN_CONFIG.json")
    runtime = r44_runtime.configure(source_frozen)
    # R44 configure verifies its original frozen version before this R2 version boundary.
    versions = runtime["r42_versions"]
    versions.TRAINING_VERSION = training_version
    runtime["trainer"].TRAINING_VERSION = training_version
    runtime["runner"].TRAINING_VERSION = training_version
    runtime["r42_run"].TRAINING_VERSION = training_version
    runtime["r42_run"].RUNNER_VERSION = RUNNER_VERSION
    frozen = copy.deepcopy(source_frozen)
    frozen["round42"]["training_version"] = training_version
    frozen["round42"]["runner_version"] = RUNNER_VERSION
    freeze_identity = frozen["round42"]["freeze_identity"]
    freeze_identity["training_version"] = training_version
    freeze_identity["runner_version"] = RUNNER_VERSION
    freeze_identity["execution_runner_version"] = RUNNER_VERSION
    freeze_body = dict(freeze_identity)
    freeze_body.pop("freeze_sha256", None)
    freeze_identity["freeze_sha256"] = hashlib.sha256(json.dumps(
        freeze_body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    bc_summary = _json(RUN_DIR / "ROUND45_R2_BC_TRAINING_SUMMARY.json")
    for model in _bc_model_manifest(bc_summary):
        seed_key = str(int(model["seed"]))
        initial = frozen["round42"]["new_initial_models"][seed_key]
        initial["parameter_sha256"] = model["parameter_sha256"]
        initial["module_parameter_sha256"] = model["module_parameter_sha256"]
        initial["round45_bc_checkpoint_sha256"] = model["checkpoint_sha256"]
    runtime["frozen"] = frozen
    runtime["round42_freeze"] = freeze_identity
    runtime["common"].OUTPUT_DIR = RUN_DIR
    runtime["common"].LOGS = RUN_DIR / "logs"
    runtime["common"].ROWS = RUN_DIR / "rows"
    runtime["common"].REJECTED = RUN_DIR / "rejected"
    runtime["common"].CHECKPOINTS = RUN_DIR / "checkpoints"
    runtime["common"].DEVICE = runtime["common"].torch.device("cpu") if hasattr(
        runtime["common"], "torch") else __import__("torch").device("cpu")
    runtime["common"].NORMAL_LAUNCHES = 528
    runtime["common"].RESERVE_LAUNCHES = 24
    runtime["common"].LAUNCH_CAP = 552
    common = runtime["r40_common"]
    common.ROUND_DIR = RUN_DIR
    common.NORMAL_CAP = 528
    common.RESERVE_CAP = 24
    common.HARD_CAP = 552
    common.plan = lambda: {
        "normal": {"bc_only_greedy_controls": 48, "new_training": 384,
                   "new_final_evaluation": 96},
        "normal_total": 528, "technical_reserve_max": 24, "hard_cap": 552,
    }
    if (common.ROUND_DIR.resolve() != RUN_DIR.resolve()
            or (int(common.NORMAL_CAP), int(common.RESERVE_CAP), int(common.HARD_CAP))
            != (528, 24, 552)
            or (int(runtime["common"].NORMAL_LAUNCHES),
                int(runtime["common"].RESERVE_LAUNCHES),
                int(runtime["common"].LAUNCH_CAP)) != (528, 24, 552)
            or float(runtime["common"].GAE_LAMBDA) != 1.0
            or float(runtime["common"].GAMMA) != 1.0
            or float(runtime["common"].ETA_LAMBDA) != 0.5
            or float(runtime["common"].B0_BYTES) != 10000.0
            or float(runtime["common"].DELTA) != 0.0
            or float(runtime["common"].LAMBDA_INIT) != 1.0
            or str(runtime["common"].DEVICE) != "cpu"):
        raise RuntimeError("R2 quota or frozen R44 math did not reach the real runtime modules")
    _install_runtime_adapters(runtime, enforcement=enforce_prelaunch)
    _install_bc_training_seams(runtime)
    _install_reference_cache_adapter(runtime)
    if enforce_prelaunch:
        verify_prelaunch_identity(frozen)
        R2_FREEZE = _json(RUN_DIR / "ROUND45_R2_POST_BC_ENV_FREEZE.json")
    return runtime


def _bc_entry_for_seed(seed: int) -> dict[str, Any]:
    if R2_FREEZE is None:
        freeze_path = RUN_DIR / "ROUND45_R2_POST_BC_ENV_FREEZE.json"
        if not freeze_path.is_file():
            raise RuntimeError("R2 post-BC freeze is required before using BC models")
        freeze = _json(freeze_path)
    else:
        freeze = R2_FREEZE
    matches = [item for item in freeze["bc_models"] if int(item["seed"]) == int(seed)]
    if len(matches) != 1:
        raise RuntimeError(f"frozen BC identity for seed {seed} is missing or duplicated")
    item = matches[0]
    path = Path(item["checkpoint_path"])
    if not path.is_file() or _sha256(path) != item["checkpoint_sha256"]:
        raise RuntimeError(f"frozen BC checkpoint changed for seed {seed}")
    return item


def _module_parameter_digests(controller: Any, bc_module: Any) -> dict[str, str]:
    return {
        "trunk": bc_module._module_hash(controller.inner.trunk),
        "action_head": bc_module._module_hash(controller.inner.action_head),
        "task_value_head": bc_module._module_hash(controller.inner.value_head),
        "cost_value_head": bc_module._module_hash(controller.cost_head),
    }


def _promote_ep0_to_rolling(runtime: dict[str, Any], controller: Any,
                            optimizer: Any, state: dict[str, Any], seed: int,
                            source: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    """Make the production ep0 milestone the byte-identical rolling recovery point."""
    import torch
    trainer = runtime["trainer"]
    common = runtime["common"]
    bc = sys.modules["r45_bc"]
    identity = _bc_entry_for_seed(seed)
    if (int(state.get("training_seed", -1)) != seed
            or int(state.get("episodes_done", -1)) != 0
            or int(state.get("updates", -1)) != 0
            or int(state.get("optimizer_steps", -1)) != 0
            or float(state.get("lambda", float("nan"))) != 1.0
            or state.get("training_version") != runtime["r42_versions"].TRAINING_VERSION
            or controller.parameter_summary().get("parameter_sha256") != identity["parameter_sha256"]
            or optimizer.state_dict().get("state") != {}
            or len([parameter for group in optimizer.param_groups
                    for parameter in group["params"]]) != len(controller.parameters)
            or [id(parameter) for group in optimizer.param_groups
                for parameter in group["params"]] != [id(parameter) for parameter in controller.parameters]):
        raise RuntimeError(f"seed {seed}: ep0 checkpoint is not BC-initialized with fresh Adam")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    saved_state = payload.get("training_state", {})
    if (payload.get("training_version") != runtime["r42_versions"].TRAINING_VERSION
            or payload.get("freeze") != freeze
            or saved_state != state
            or payload.get("optimizer", {}).get("state") != {}
            or payload.get("controller") is None):
        raise RuntimeError(f"seed {seed}: production ep0 checkpoint identity/state mismatch")
    registered = runtime["frozen"]["round43"]["registered_ep0_inputs"][str(seed)]
    registered_path = Path(registered["path"])
    if not registered_path.is_file() or _sha256(registered_path) != registered["file_sha256"]:
        raise RuntimeError(f"seed {seed}: registered R44 rollout RNG source changed")
    registered_payload = torch.load(registered_path, map_location="cpu", weights_only=False)
    if not _same_value(payload.get("rng"), registered_payload.get("rng")):
        raise RuntimeError(f"seed {seed}: BC-init ep0 checkpoint did not retain registered R44 RNG")
    actual_modules = _module_parameter_digests(controller, bc)
    if actual_modules != identity["module_parameter_sha256"]:
        raise RuntimeError(f"seed {seed}: BC-init module parameter digests mismatch")

    rolling = Path(common.CHECKPOINTS) / f"v2_train_s{seed}.pt"
    source_hash = _sha256(source)
    if rolling.exists():
        if _sha256(rolling) != source_hash:
            raise RuntimeError(f"seed {seed}: rolling ep0 recovery file already differs")
    else:
        temporary = rolling.with_name(rolling.name + f".{os.getpid()}.ep0-copy")
        if temporary.exists():
            raise RuntimeError(f"seed {seed}: stale ep0 promotion temporary requires inspection")
        with source.open("rb") as src, temporary.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
            dst.flush()
            os.fsync(dst.fileno())
        if _sha256(temporary) != source_hash:
            raise RuntimeError(f"seed {seed}: ep0 rolling copy is not byte-identical")
        os.replace(temporary, rolling)
    receipt = {
        "kind": "round45-r2-ep0-rolling-copy-receipt/1",
        "status": "passed", "seed": seed,
        "source_path": str(source.resolve()), "source_sha256": source_hash,
        "rolling_path": str(rolling.resolve()), "rolling_sha256": _sha256(rolling),
        "byte_identical": _sha256(rolling) == source_hash,
        "training_version": runtime["r42_versions"].TRAINING_VERSION,
        "bc_checkpoint_sha256": identity["checkpoint_sha256"],
        "bc_parameter_sha256": identity["parameter_sha256"],
        "registered_rng_source_sha256": registered["file_sha256"],
        "episodes_done": 0, "updates": 0, "optimizer_steps": 0,
        "lambda": 1.0, "fresh_adam_state_empty": True,
    }
    receipt_path = RUN_DIR / "runtime" / f"ROUND45_R2_EP0_RECOVERY_s{seed}.json"
    if receipt_path.exists():
        if _json(receipt_path) != receipt:
            raise RuntimeError(f"seed {seed}: prior ep0 recovery receipt conflicts")
    else:
        _write_atomic(receipt_path, receipt)
    return receipt


def _install_bc_training_seams(runtime: dict[str, Any]) -> None:
    r40_train = runtime["r40_train"]
    base_save_milestone = r40_train._save_milestone

    def save_milestone(controller: Any, optimizer: Any, state: dict[str, Any],
                       training_seed: int, episodes: int,
                       freeze: dict[str, Any]) -> Path:
        path = base_save_milestone(controller, optimizer, state,
                                   training_seed, episodes, freeze)
        if int(episodes) == 0:
            _promote_ep0_to_rolling(runtime, controller, optimizer, state,
                                    int(training_seed), Path(path), freeze)
        return path

    r40_train._save_milestone = save_milestone


def _install_reference_cache_adapter(runtime: dict[str, Any]) -> None:
    """Call the real R43 cache once, then reuse its verified fixed 64 rows."""
    run_module = runtime["r42_run"]
    original = run_module.reference_cache
    cache = original()
    manifest = _json(R44_DIR / "ROUND44_BASELINE_REUSE.json")
    entries = manifest["round42_immediate_references"]
    expected = {int(item["scene"]): item for item in entries}
    if len(expected) != 64 or set(cache) != set(expected):
        raise RuntimeError("R2 training reference cache differs from the registered 64 immediate rows")
    for scene, item in expected.items():
        path = Path(item["path"])
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"R2 immediate training reference changed for scene {scene}")
        row = _json(path)
        if (int(row.get("scenario_seed", -1)) != scene
                or row.get("reference_arm") != "immediate"
                or cache[scene].get("behaviour_sha256") != row.get("behaviour_sha256")):
            raise RuntimeError(f"R2 immediate training reference identity mismatch for scene {scene}")
    runtime["r2_reference_cache"] = cache
    runtime["r2_reference_cache_sha256"] = hashlib.sha256(json.dumps(
        {str(scene): item["sha256"] for scene, item in sorted(expected.items())},
        sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    run_module.reference_cache = lambda: dict(cache)
    receipt = {
        "kind": "round45-r2-reference-cache-reuse/1", "status": "passed",
        "source_manifest_path": str((R44_DIR / "ROUND44_BASELINE_REUSE.json").resolve()),
        "source_manifest_sha256": _sha256(R44_DIR / "ROUND44_BASELINE_REUSE.json"),
        "source_round42_head": manifest["round42_reference_source_head"],
        "reference_count": len(cache),
        "reference_scene_ids": sorted(cache),
        "reference_row_sha256": {str(scene): expected[scene]["sha256"]
                                  for scene in sorted(expected)},
        "reference_set_identity_sha256": runtime["r2_reference_cache_sha256"],
        "loaded_via_actual_r43_reference_cache": True,
        "reference_cache_reloads_after_install": 0,
    }
    receipt_path = RUN_DIR / "runtime" / "ROUND45_R2_REFERENCE_CACHE_REUSE.json"
    if receipt_path.exists():
        if _json(receipt_path) != receipt:
            raise RuntimeError("R2 immediate reference reuse receipt changed")
    else:
        _write_atomic(receipt_path, receipt)


def _call_training_seed(runtime: dict[str, Any], seed: int, *, episodes: int) -> dict[str, Any]:
    """Call the R43 entry and R40 trainer with BC model-only initialization."""
    import r45_bc
    model = _bc_entry_for_seed(seed)
    module = runtime["controller_module"]
    base_fresh_controller = module.fresh_controller
    injection: dict[str, Any] = {"seed": int(seed), "applied": False}

    def fresh_bc_controller(model_seed: int):
        controller = base_fresh_controller(model_seed)
        if int(model_seed) != int(seed):
            return controller
        if injection["applied"]:
            raise RuntimeError(f"seed {seed}: fresh BC controller injection was invoked twice")
        if _sha256(Path(model["checkpoint_path"])) != model["checkpoint_sha256"]:
            raise RuntimeError(f"seed {seed}: BC checkpoint changed before fresh controller load")
        payload = r45_bc.read_bc_checkpoint(Path(model["checkpoint_path"]))
        before_rng = runtime["trainer"].rng_state()
        r45_bc.load_bc_controller_state(controller, payload)
        after_rng = runtime["trainer"].rng_state()
        parameter = controller.parameter_summary()["parameter_sha256"]
        modules = _module_parameter_digests(controller, r45_bc)
        if (parameter != model["parameter_sha256"]
                or modules != model["module_parameter_sha256"]
                or not _same_value(before_rng, after_rng)
                or int(payload.get("seed", -1)) != int(seed)
                or payload.get("post_bc_parameter_sha256") != model["parameter_sha256"]):
            raise RuntimeError(f"seed {seed}: model-only BC initialization identity failed")
        injection.update({
            "applied": True,
            "bc_checkpoint_path": model["checkpoint_path"],
            "bc_checkpoint_sha256": model["checkpoint_sha256"],
            "bc_parameter_sha256": parameter,
            "module_parameter_sha256": modules,
            "bc_load_changed_rng": False,
            "bc_optimizer_loaded": False,
            "registered_ep0_rollout_rng_applied_by_r42_entry": False,
        })
        return controller

    module.fresh_controller = fresh_bc_controller
    try:
        report = runtime["r42_run"].train_seed(
            int(seed), episodes=int(episodes), frozen=runtime["frozen"])
    finally:
        module.fresh_controller = base_fresh_controller
    if injection["applied"]:
        priming = report.get("rng_priming", {})
        if (priming.get("applied_after_fresh_controller") is not True
                or priming.get("loaded_model") is not False
                or priming.get("loaded_optimizer") is not False):
            raise RuntimeError(f"seed {seed}: R42 did not apply only its registered ep0 RNG")
        injection["registered_ep0_rollout_rng_applied_by_r42_entry"] = True
    elif not (Path(runtime["common"].CHECKPOINTS)
              / f"v2_train_s{seed}.pt").is_file():
        raise RuntimeError(f"seed {seed}: resumed train call has neither BC init nor R2 rolling state")
    report["round45_bc_initialization"] = injection
    report["training_version"] = runtime["r42_versions"].TRAINING_VERSION
    report["runner_version"] = RUNNER_VERSION
    return report


def _fixed_gate_baseline() -> dict[str, Any]:
    """Verify only the fixed development references and late-rule rows used by R2 gates."""
    manifest_path = R44_DIR / "ROUND44_BASELINE_REUSE.json"
    manifest = _json(manifest_path)
    references = [item for item in manifest["round42_immediate_references"]
                  if 1_301_000 <= int(item["scene"]) < 1_301_016]
    fixed_rules = {
        arm: [item for item in manifest["round42_fixed_rules"] if item.get("arm") == arm]
        for arm in ("late_discovery_gate", "always_hold")
    }
    if len(references) != 16 or any(len(items) != 16 for items in fixed_rules.values()):
        raise RuntimeError("fixed R44 baseline lacks the registered 16-scene comparison sets")
    reference_rows: dict[int, dict[str, Any]] = {}
    rule_rows: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in fixed_rules}
    for item in references:
        scene = int(item["scene"])
        path = Path(item["path"])
        if (scene in reference_rows or not path.is_file()
                or _sha256(path) != item["sha256"]):
            raise RuntimeError(f"fixed immediate reference identity failed for scene {scene}")
        row = _json(path)
        if (int(row.get("scenario_seed", -1)) != scene
                or row.get("reference_arm") != "immediate"):
            raise RuntimeError(f"fixed immediate reference row identity failed: {path}")
        reference_rows[scene] = row
    for arm, items in fixed_rules.items():
        for item in items:
            scene = int(item["scene"])
            path = Path(item["path"])
            if (scene in rule_rows[arm] or not path.is_file()
                    or _sha256(path) != item["sha256"]):
                raise RuntimeError(f"fixed {arm} rule identity failed for scene {scene}")
            row = _json(path)
            if (int(row.get("scenario_seed", -1)) != scene
                    or row.get("model_arm") != arm):
                raise RuntimeError(f"fixed {arm} rule row identity failed: {path}")
            rule_rows[arm][scene] = row
    scenes = set(range(1_301_000, 1_301_016))
    if (set(reference_rows) != scenes
            or any(set(rows) != scenes for rows in rule_rows.values())):
        raise RuntimeError("fixed baseline scene identities are not the R44 development block")
    late_bytes = [int(row["spent_bytes"])
                  for row in rule_rows["late_discovery_gate"].values()]
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "round42_source_head": manifest["round42_reference_source_head"],
        "reference_rows": reference_rows,
        "reference_row_sha256": {int(item["scene"]): item["sha256"] for item in references},
        "rule_rows": rule_rows,
        "rule_row_sha256": {
            arm: {int(item["scene"]): item["sha256"] for item in items}
            for arm, items in fixed_rules.items()},
        "late_mean_spent_bytes": sum(late_bytes) / len(late_bytes),
        "development_scenes": sorted(scenes),
    }


def _on_time_ids(row: dict[str, Any]) -> set[str]:
    return {str(event["task_id"]) for event in row.get("task_reward_events", [])
            if event.get("credited") and event.get("on_time")}


def _load_bc_eval_controller(runtime: dict[str, Any], seed: int):
    import r45_bc
    model = _bc_entry_for_seed(seed)
    trainer = runtime["trainer"]
    ambient_rng = trainer.rng_state()
    try:
        controller = runtime["controller_module"].fresh_controller(int(seed))
        payload = r45_bc.read_bc_checkpoint(Path(model["checkpoint_path"]))
        r45_bc.load_bc_controller_state(controller, payload)
    finally:
        trainer.restore_rng(ambient_rng)
    actual = controller.parameter_summary()
    modules = _module_parameter_digests(controller, r45_bc)
    if actual.get("parameter_sha256") != model["parameter_sha256"]:
        raise RuntimeError(f"seed {seed}: read-only BC evaluation parameter digest mismatch")
    if modules != model["module_parameter_sha256"]:
        raise RuntimeError(f"seed {seed}: read-only BC evaluation module digest mismatch")
    controller.eval()
    identity = {
        "arm": "BC_ONLY_R2", "path": model["checkpoint_path"],
        "file_sha256": model["checkpoint_sha256"], "training_seed": int(seed),
        "episodes_done": 0, "updates": int(model["updates"]),
        "optimizer_steps": int(model["updates"]), "lambda": 1.0,
        "parameter_identity": actual,
        "module_parameter_sha256": modules,
        "evaluation_semantics": "one-shot-send-skip/1",
        "optimizer_restored": False, "rng_restored": False,
        "read_only_bc_model": True,
    }
    return controller, identity, model


def _run_control_cell(runtime: dict[str, Any], seed: int, scene: int,
                      controller: Any, identity: dict[str, Any],
                      model: dict[str, Any]) -> dict[str, Any]:
    from r42_one_shot import OneShotSendSkipCollector
    collector = OneShotSendSkipCollector(controller, greedy=True)
    logical = f"r45_bc_control_s{seed}_scene{scene}"
    tag = f"r45_bc_greedy_s{seed}_scene{scene}"
    context = {
        "scope": "bc_only_greedy_control", "logical_job": logical,
        "seed": int(seed), "mode": "greedy", "scenario_seed": int(scene),
        "kind": "eval", "tag": tag, "bc_checkpoint_sha256": model["checkpoint_sha256"],
        "bc_parameter_sha256": model["parameter_sha256"],
        "allow_technical_retry": True,
    }
    token = CURRENT_CONTEXT.set(context)
    try:
        return runtime["r42_run"]._execute_cell(
            f"bc_only_greedy_s{seed}_scene{scene}_r45", logical, "eval", int(scene),
            {"arm": "BC_ONLY_R2", "training_seed": int(seed), "mode": "greedy",
             "scenario_seed": int(scene)},
            collector=collector,
            runner_kwargs={
                "policy": "learned", "attach": collector.policy(),
                "controller": controller, "collector": collector,
                "action_rng_seed": runtime["runner"].eval_action_seed(seed, scene, "greedy"),
                "tag": tag, "checkpoint_load_identity": identity,
            },
            row_updates={
                "model_arm": "BC_ONLY_R2", "training_seed": int(seed),
                "eval_mode": "greedy", "eval_block": "round45_bc_only_greedy_control",
                "action_rng_seed": runtime["runner"].eval_action_seed(seed, scene, "greedy"),
                "checkpoint_parameter_identity": identity["parameter_identity"],
                "r45_evaluation_role": "BC_ONLY_GREEDY_CONTROL",
                "bc_checkpoint_sha256": model["checkpoint_sha256"],
                "bc_parameter_sha256": model["parameter_sha256"],
                "teacher_inference_used": False,
            },
            identity=identity,
        )
    finally:
        CURRENT_CONTEXT.reset(token)


def _control_technical_gate(runtime: dict[str, Any], seed: int, row: dict[str, Any],
                            model: dict[str, Any], before: tuple[int, int]) -> dict[str, Any]:
    # Row files do not repeat logical_job, so resolve the registered key directly.
    claims = _attempts(f"r45_bc_control_s{seed}_scene{int(row['scenario_seed'])}")
    if len(claims) != 1 or claims[0][1].get("status") != "complete":
        raise RuntimeError(f"seed {seed}: first BC control has no unique complete claim")
    claim = claims[0][1]
    run_counts = {
        key: int(claim.get(key, -1)) for key in (
            "environment_constructor_calls", "run_invocation_calls", "joint_run_method_calls",
            "run_return_calls", "run_failure_calls", "completed_steps")
    }
    expected = {"environment_constructor_calls": 1, "run_invocation_calls": 1,
                "joint_run_method_calls": 1, "run_return_calls": 1,
                "run_failure_calls": 0, "completed_steps": 1000}
    if run_counts != expected:
        raise RuntimeError(f"seed {seed}: first BC control run accounting mismatch: {run_counts}")
    param = row.get("controller_parameters_after", {}).get("parameter_sha256")
    if param != model["parameter_sha256"]:
        raise RuntimeError(f"seed {seed}: first BC control changed or misidentified its model")
    if (row.get("teacher_inference_used") is not False
            or row.get("r45_evaluation_role") != "BC_ONLY_GREEDY_CONTROL"
            or row.get("comm_action_semantics") != "one-shot-send-skip/1"
            or int(row.get("scenario_seed", -1)) < 1_301_000):
        raise RuntimeError(f"seed {seed}: first BC control protocol identity failed")
    steps_after = int(runtime["r42_run"]._OPTIMIZER_STEP_CALLS)
    ppo_after = int(runtime["r42_run"]._PPO_UPDATE_CALLS)
    if (steps_after, ppo_after) != before:
        raise RuntimeError(f"seed {seed}: BC control changed optimizer/PPO counters")
    return {"seed": int(seed), "scene": int(row["scenario_seed"]),
            "claim_run_counts": run_counts, "parameter_sha256": param,
            "optimizer_step_calls_before_after": [before[0], steps_after],
            "ppo_update_calls_before_after": [before[1], ppo_after],
            "teacher_inference_used": False}


def _run_bc_controls(runtime: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline = _fixed_gate_baseline()
    optimizer_before = int(runtime["r42_run"]._OPTIMIZER_STEP_CALLS)
    ppo_before = int(runtime["r42_run"]._PPO_UPDATE_CALLS)
    technical_gates = []
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    models = {}
    # First complete and inspect one fixed scene for each seed, then execute the rest.
    first_scene = 1_301_000
    for seed in (11, 22, 33):
        controller, identity, model = _load_bc_eval_controller(runtime, seed)
        models[seed] = model
        row = _run_control_cell(runtime, seed, first_scene, controller, identity, model)
        rows[(seed, first_scene)] = row
        technical_gates.append(_control_technical_gate(
            runtime, seed, row, model, (optimizer_before, ppo_before)))
        del controller
    for seed in (11, 22, 33):
        controller, identity, model = _load_bc_eval_controller(runtime, seed)
        for scene in range(1_301_001, 1_301_016):
            rows[(seed, scene)] = _run_control_cell(
                runtime, seed, scene, controller, identity, model)
        del controller

    per_seed: dict[str, Any] = {}
    all_pass = True
    for seed in (11, 22, 33):
        seed_rows = [rows[(seed, scene)] for scene in range(1_301_000, 1_301_016)]
        scene_results = []
        exact = True
        for row in seed_rows:
            scene = int(row["scenario_seed"])
            target = _on_time_ids(baseline["reference_rows"][scene])
            actual = _on_time_ids(row)
            equal = actual == target
            exact = exact and equal
            scene_results.append({
                "scene": scene, "immediate_on_time_task_ids": sorted(target),
                "bc_on_time_task_ids": sorted(actual), "exact_identity_set_match": equal,
                "on_time_count": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]),
            })
        bytes_mean = sum(item["spent_bytes"] for item in scene_results) / len(scene_results)
        bytes_pass = bytes_mean <= baseline["late_mean_spent_bytes"]
        seed_pass = exact and bytes_pass
        all_pass = all_pass and seed_pass
        per_seed[str(seed)] = {
            "scene_count": len(seed_rows), "scene_identity_set_match_all": exact,
            "mean_spent_bytes": bytes_mean,
            "fixed_late_discovery_gate_mean_spent_bytes": baseline["late_mean_spent_bytes"],
            "mean_bytes_at_or_below_late_rule": bytes_pass,
            "passes_training_gate": seed_pass,
            "scenes": scene_results,
        }
    result = {
        "kind": "round45-r2-bc-only-greedy-control-gate/1",
        "status": "passed" if all_pass else "failed",
        "training_authorized_by_gate": all_pass,
        "technical_first_scene_gates": technical_gates,
        "baseline": {
            "manifest_path": baseline["manifest_path"],
            "manifest_sha256": baseline["manifest_sha256"],
            "immediate_reference_row_sha256": baseline["reference_row_sha256"],
            "fixed_rule_row_sha256": baseline["rule_row_sha256"],
            "fixed_late_mean_spent_bytes": baseline["late_mean_spent_bytes"],
            "development_scenes": baseline["development_scenes"],
        },
        "per_seed": per_seed,
        "optimizer_step_calls": int(runtime["r42_run"]._OPTIMIZER_STEP_CALLS) - optimizer_before,
        "ppo_update_calls": int(runtime["r42_run"]._PPO_UPDATE_CALLS) - ppo_before,
        "environment_cells": 48,
    }
    _write_atomic(RUN_DIR / "ROUND45_R2_BC_CONTROL_GATE.json", result)
    ordered_rows = [rows[(seed, scene)] for seed in (11, 22, 33)
                    for scene in range(1_301_000, 1_301_016)]
    return ordered_rows, result


def _task_state(stage: str, **updates: Any) -> None:
    path = RUN_DIR / "ROUND45_CURRENT_TASK.json"
    task = _json(path)
    task.update({"stage": stage, "updated_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "real_bc_updates": 3000, "family_cumulative_real_bc_updates": 4000,
                 "environment_constructed": int(task.get("environment_constructed", 0)),
                 "ppo_updates": int(task.get("ppo_updates", 0)),
                 "remote_push": False})
    task.update(updates)
    _write_atomic(path, task)


def _run_first_batch_gate(runtime: dict[str, Any], seed: int) -> dict[str, Any]:
    gate_path = RUN_DIR / f"ROUND45_R2_FIRST_BATCH_GATE_s{seed}.json"
    batch_path = RUN_DIR / "checkpoints" / f"r40_batch_s{seed}_b0_attempt01.json"
    if not batch_path.is_file():
        records = sorted((RUN_DIR / "checkpoints").glob(f"r40_batch_s{seed}_b0_attempt*.json"))
        if len(records) != 1:
            raise RuntimeError(f"seed {seed}: first PPO batch record is missing or ambiguous")
        batch_path = records[0]
    batch_sha = _sha256(batch_path)
    model = _bc_entry_for_seed(seed)
    if gate_path.is_file():
        gate = _json(gate_path)
        if (gate.get("status") != "passed" or int(gate.get("seed", -1)) != seed
                or gate.get("first_batch_record_sha256") != batch_sha
                or gate.get("bc_checkpoint_sha256") != model["checkpoint_sha256"]
                or gate.get("bc_parameter_sha256") != model["parameter_sha256"]):
            raise RuntimeError(f"seed {seed}: saved first-batch gate conflicts with current batch")
        return gate

    import torch
    raw = json.loads(batch_path.read_text(encoding="utf-8"))
    checkpoint_path = RUN_DIR / "checkpoints" / str(raw["preupdate_checkpoint"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("training_state", {})
    optimizer = payload.get("optimizer", {})
    param_groups = optimizer.get("param_groups", [])
    param_ids = [identifier for group in param_groups for identifier in group.get("params", [])]
    registered = runtime["frozen"]["round43"]["registered_ep0_inputs"][str(seed)]
    registered_payload = torch.load(Path(registered["path"]), map_location="cpu",
                                    weights_only=False)
    pre_checks = {
        "preupdate_episode_cursor": int(state.get("episodes_done", -1)),
        "preupdate_update_cursor": int(state.get("updates", -1)),
        "preupdate_optimizer_steps": int(state.get("optimizer_steps", -1)),
        "preupdate_lambda": float(state.get("lambda", float("nan"))),
        "empty_fresh_adam_state": optimizer.get("state") == {},
        "full_parameter_adam_count": len(param_ids),
        "ten_unique_optimizer_parameter_ids": len(param_ids) == 10 and len(set(param_ids)) == 10,
        "bc_parameter_sha256": raw.get("controller_parameters_before", {}).get(
            "parameter_sha256"),
        "registered_r44_ep0_rng_exact": _same_value(payload.get("rng"),
                                                     registered_payload.get("rng")),
        "first_batch_lambda": float(raw.get("lambda_at_rollout", float("nan"))),
    }
    if (pre_checks["preupdate_episode_cursor"] != 0
            or pre_checks["preupdate_update_cursor"] != 0
            or pre_checks["preupdate_optimizer_steps"] != 0
            or pre_checks["preupdate_lambda"] != 1.0
            or not pre_checks["empty_fresh_adam_state"]
            or not pre_checks["ten_unique_optimizer_parameter_ids"]
            or pre_checks["bc_parameter_sha256"] != model["parameter_sha256"]
            or not pre_checks["registered_r44_ep0_rng_exact"]
            or pre_checks["first_batch_lambda"] != 1.0):
        raise RuntimeError(f"seed {seed}: BC/PPO fresh-start boundary checks failed: {pre_checks}")

    try:
        gate_body = runtime["r42_analysis"].first_batch_gate(
            seed, runtime=runtime, frozen=runtime["frozen"])
        status = "passed"
        error = None
    except BaseException as exc:
        gate_body = {}
        status = "failed"
        error = {"type": type(exc).__name__, "message": str(exc)}
    result = {
        "kind": "round45-r2-first-batch-real-ppo-gate/1", "status": status,
        "seed": int(seed), "first_batch_record_path": str(batch_path.resolve()),
        "first_batch_record_sha256": batch_sha,
        "preupdate_checkpoint_path": str(checkpoint_path.resolve()),
        "preupdate_checkpoint_sha256": _sha256(checkpoint_path),
        "bc_checkpoint_sha256": model["checkpoint_sha256"],
        "bc_parameter_sha256": model["parameter_sha256"],
        "preupdate_boundary_checks": pre_checks,
        "actual_r43_first_batch_gate": gate_body,
        "exception": error,
        "environment_constructed": 0,
        "additional_optimizer_steps": 0,
        "synthetic_updates": 0,
    }
    _write_atomic(gate_path, result)
    if status != "passed":
        raise RuntimeError(f"seed {seed}: actual first-batch gate failed: {error}")
    return result


def _run_training_plan(runtime: dict[str, Any]) -> dict[int, dict[str, Any]]:
    reports: dict[int, dict[str, Any]] = {}
    for seed in (11, 22, 33):
        _task_state(f"seed {seed}: running authorized first four PPO episodes",
                    active_training_seed=seed, episodes_requested=4)
        first_four = _call_training_seed(runtime, seed, episodes=4)
        gate = _run_first_batch_gate(runtime, seed)
        if gate.get("status") != "passed":
            raise RuntimeError(f"seed {seed}: first four episode gate did not pass")
        _task_state(f"seed {seed}: first-batch gate passed; continuing to fixed ep128",
                    active_training_seed=seed, first_batch_gate="passed",
                    seed_episodes_done=int(first_four.get("episodes_done", 4)))
        full = _call_training_seed(runtime, seed, episodes=128)
        if (int(full.get("episodes_done", -1)) != 128
                or int(full.get("updates", -1)) != 32
                or int(full.get("optimizer_steps", -1)) <= 0
                or full.get("training_version") != runtime["r42_versions"].TRAINING_VERSION
                or full.get("round45_bc_initialization") is None):
            raise RuntimeError(f"seed {seed}: fixed 128-episode training identity did not close")
        training_path = RUN_DIR / f"ROUND45_R2_TRAINING_s{seed}.json"
        if training_path.is_file():
            prior = _json(training_path)
            checkpoint = RUN_DIR / "checkpoints" / f"v2_train_s{seed}_ep128.pt"
            if (prior.get("status") != "complete"
                    or _sha256(checkpoint) != prior.get("ep128_checkpoint_sha256")
                    or prior.get("bc_checkpoint", {}).get("checkpoint_sha256")
                    != _bc_entry_for_seed(seed)["checkpoint_sha256"]):
                raise RuntimeError(f"seed {seed}: existing training result conflicts with ep128 state")
            reports[seed] = prior
            _task_state(f"seed {seed}: fixed ep128 complete (resume verified)",
                        active_training_seed=seed, seed_episodes_done=128,
                        seed_ppo_updates=32)
            continue
        result = {
            "kind": "round45-r2-seed-training-result/1", "seed": seed,
            "status": "complete", "training_version": runtime["r42_versions"].TRAINING_VERSION,
            "runner_version": RUNNER_VERSION, "bc_checkpoint": _bc_entry_for_seed(seed),
            "first_four_report": first_four, "first_batch_gate": gate,
            "fixed_ep128_report": full,
            "ep128_checkpoint_path": str((RUN_DIR / "checkpoints"
                / f"v2_train_s{seed}_ep128.pt").resolve()),
            "ep128_checkpoint_sha256": _sha256(RUN_DIR / "checkpoints"
                / f"v2_train_s{seed}_ep128.pt"),
        }
        reports[seed] = result
        _write_atomic(training_path, result)
        _task_state(f"seed {seed}: fixed ep128 complete",
                    active_training_seed=seed, seed_episodes_done=128,
                    seed_ppo_updates=32,
                    cumulative_ppo_updates=sum(int(item["fixed_ep128_report"]["updates"])
                                              for item in reports.values()))
    return reports


def _run_final_eval_matrix(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for seed in (11, 22, 33):
        controller, identity = runtime["r42_run"].load_eval_model("NEW", seed)
        if (int(identity.get("episodes_done", -1)) != 128
                or int(identity.get("updates", -1)) != 32
                or identity.get("optimizer_restored") is not False
                or identity.get("rng_restored") is not False):
            raise RuntimeError(f"seed {seed}: final evaluation model is not read-only ep128")
        for mode in ("greedy", "sampled"):
            for scene in range(1_301_000, 1_301_016):
                row = runtime["r42_run"].run_model_eval(
                    arm="NEW", training_seed=seed, mode=mode, scenario=scene,
                    controller=controller, identity=identity)
                rows.append(row)
                if len(rows) % 16 == 0:
                    _task_state("fixed ep128 final evaluation in progress",
                                final_eval_rows=len(rows), active_eval_seed=seed,
                                active_eval_mode=mode)
        del controller
    if len(rows) != 96:
        raise RuntimeError(f"R2 final evaluation produced {len(rows)}/96 cells")
    keys = {(int(row["training_seed"]), str(row["eval_mode"]),
             int(row["scenario_seed"])) for row in rows}
    expected = {(seed, mode, scene) for seed in (11, 22, 33)
                for mode in ("greedy", "sampled")
                for scene in range(1_301_000, 1_301_016)}
    if keys != expected:
        raise RuntimeError("R2 final ep128 evaluation matrix keys do not match the fixed 96 cells")
    return rows


def _load_r44_primary_eval_matrix() -> dict[tuple[int, str, int], dict[str, Any]]:
    audit_path = R44_DIR / "ROUND44_EVAL_MATRIX_AUDIT.json"
    if _sha256(audit_path) != R44_MATRIX_AUDIT_SHA256:
        raise RuntimeError("fixed R44 final 96-cell matrix audit changed")
    audit = _json(audit_path)
    if audit.get("status") != "passed" or int(audit.get("cell_count", -1)) != 96:
        raise RuntimeError("fixed R44 final matrix audit is not a 96-cell pass")
    result: dict[tuple[int, str, int], dict[str, Any]] = {}
    for entry in audit["entries"]:
        key = (int(entry["eval_key"][0]), str(entry["eval_key"][1]),
               int(entry["eval_key"][2]))
        if key in result:
            raise RuntimeError(f"duplicate R44 audited matrix key: {key}")
        path = Path(entry["row_path"])
        claim_path = Path(entry["claim_path"])
        if (not path.is_file() or _sha256(path) != entry["row_sha256"]
                or not claim_path.is_file() or _sha256(claim_path) != entry["claim_sha256"]):
            raise RuntimeError(f"R44 audited final cell or claim changed: {key}")
        row = _json(path)
        if (int(row.get("training_seed", -1)) != key[0]
                or str(row.get("eval_mode")) != key[1]
                or int(row.get("scenario_seed", -1)) != key[2]
                or row.get("model_arm") != "NEW"
                or row.get("checkpoint_parameter_identity", {}).get("parameter_sha256")
                != entry["checkpoint_parameter_sha256"]):
            raise RuntimeError(f"R44 audited final row identity mismatch: {key}")
        result[key] = row
    if len(result) != 96:
        raise RuntimeError(f"R44 fixed matrix recovered {len(result)}/96 rows")
    return result


def _summary_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "on_time_total": sum(int(row["on_time"]) for row in rows),
        "spent_bytes_total": sum(int(row["spent_bytes"]) for row in rows),
        "mean_on_time": (sum(int(row["on_time"]) for row in rows) / len(rows)
                         if rows else 0.0),
        "mean_spent_bytes": (sum(int(row["spent_bytes"]) for row in rows) / len(rows)
                             if rows else 0.0),
        "scenes": [{
            "scene": int(row["scenario_seed"]), "on_time": int(row["on_time"]),
            "on_time_task_ids": sorted(_on_time_ids(row)),
            "spent_bytes": int(row["spent_bytes"]),
        } for row in sorted(rows, key=lambda item: int(item["scenario_seed"]))],
    }


def _paired_result(left: dict[int, dict[str, Any]], right: dict[int, dict[str, Any]],
                   left_name: str, right_name: str) -> dict[str, Any]:
    if set(left) != set(right):
        raise RuntimeError(f"paired scene keys differ: {left_name}/{right_name}")
    scenes = []
    for scene in sorted(left):
        left_ids = _on_time_ids(left[scene])
        right_ids = _on_time_ids(right[scene])
        scenes.append({
            "scene": scene,
            f"{left_name}_on_time": int(left[scene]["on_time"]),
            f"{right_name}_on_time": int(right[scene]["on_time"]),
            "on_time_delta_right_minus_left": int(right[scene]["on_time"])
                                                - int(left[scene]["on_time"]),
            f"{left_name}_spent_bytes": int(left[scene]["spent_bytes"]),
            f"{right_name}_spent_bytes": int(right[scene]["spent_bytes"]),
            "spent_bytes_delta_right_minus_left": int(right[scene]["spent_bytes"])
                                                   - int(left[scene]["spent_bytes"]),
            "recovered_task_ids": sorted(right_ids - left_ids),
            "lost_task_ids": sorted(left_ids - right_ids),
            "exact_on_time_id_set_match": left_ids == right_ids,
        })
    on_delta = [item["on_time_delta_right_minus_left"] for item in scenes]
    byte_delta = [item["spent_bytes_delta_right_minus_left"] for item in scenes]
    return {
        "left": left_name, "right": right_name, "scene_count": len(scenes),
        "on_time_delta_total": sum(on_delta),
        "spent_bytes_delta_total": sum(byte_delta),
        "scenes_exactly_matched": sum(item["exact_on_time_id_set_match"] for item in scenes),
        "scenes": scenes,
    }


def _analyze_completed_run(runtime: dict[str, Any],
                           controls_gate: dict[str, Any]) -> dict[str, Any]:
    baseline = _fixed_gate_baseline()
    rows = [_json(path) for path in sorted(Path(runtime["common"].ROWS).glob("*.json"))]
    version = runtime["r42_versions"].TRAINING_VERSION
    rows = [row for row in rows if row.get("training_version") == version]
    training = [row for row in rows if row.get("training_arm") == "NEW"]
    controls = [row for row in rows
                if row.get("r45_evaluation_role") == "BC_ONLY_GREEDY_CONTROL"]
    evaluations = [row for row in rows if row.get("kind") == "eval"
                   and row.get("r45_evaluation_role") != "BC_ONLY_GREEDY_CONTROL"]
    if (len(training) != 384 or len(controls) != 48 or len(evaluations) != 96):
        raise RuntimeError(
            f"R2 completed row counts do not close: train={len(training)}, "
            f"control={len(controls)}, eval={len(evaluations)}")
    for row in training + controls + evaluations:
        runtime["r42_analysis"]._validate_reward_timing(row)
        if row.get("comm_action_semantics") != "one-shot-send-skip/1":
            raise RuntimeError(f"R2 row has a foreign one-shot action identity: {row.get('label')}")

    r44 = _load_r44_primary_eval_matrix()
    immediate = baseline["reference_rows"]
    fixed_rules = baseline["rule_rows"]
    per_seed = {}
    comparisons = {"R2_minus_R44_NEW": {}, "R2_minus_BC_only_greedy": {},
                  "R2_minus_immediate": {}, "R2_minus_fixed_rules": {}}
    for seed in (11, 22, 33):
        bc_rows = {int(row["scenario_seed"]): row for row in controls
                   if int(row["training_seed"]) == seed}
        greedy_rows = {int(row["scenario_seed"]): row for row in evaluations
                       if int(row["training_seed"]) == seed and row.get("eval_mode") == "greedy"}
        sampled_rows = {int(row["scenario_seed"]): row for row in evaluations
                        if int(row["training_seed"]) == seed and row.get("eval_mode") == "sampled"}
        train_rows = [row for row in training if int(row["training_seed"]) == seed]
        if (set(bc_rows) != set(baseline["development_scenes"])
                or set(greedy_rows) != set(baseline["development_scenes"])
                or set(sampled_rows) != set(baseline["development_scenes"])
                or len(train_rows) != 128):
            raise RuntimeError(f"R2 seed {seed} has incomplete keyed training/evaluation rows")
        on_time_equal = all(_on_time_ids(greedy_rows[scene])
                            == _on_time_ids(immediate[scene])
                            for scene in baseline["development_scenes"])
        mean_bytes = sum(int(row["spent_bytes"]) for row in greedy_rows.values()) / 16
        task_preserved = on_time_equal
        cost_below_rule = mean_bytes < baseline["late_mean_spent_bytes"]
        r2_r44_by_mode = {}
        for mode, new_rows in (("greedy", greedy_rows), ("sampled", sampled_rows)):
            r44_rows = {scene: r44[(seed, mode, scene)]
                        for scene in baseline["development_scenes"]}
            r2_r44_by_mode[mode] = _paired_result(
                r44_rows, new_rows, "R44_NEW", "R2_NEW")
            comparisons["R2_minus_R44_NEW"][f"s{seed}_{mode}"] = r2_r44_by_mode[mode]
            comparisons["R2_minus_immediate"][f"s{seed}_{mode}"] = _paired_result(
                immediate, new_rows, "immediate", "R2_NEW")
        comparisons["R2_minus_BC_only_greedy"][f"s{seed}_greedy"] = _paired_result(
            bc_rows, greedy_rows, "BC_only", "R2_NEW")
        comparisons["R2_minus_fixed_rules"][f"s{seed}_greedy_vs_late"] = _paired_result(
            fixed_rules["late_discovery_gate"], greedy_rows, "late_discovery_gate", "R2_NEW")
        comparisons["R2_minus_fixed_rules"][f"s{seed}_greedy_vs_always_hold"] = _paired_result(
            fixed_rules["always_hold"], greedy_rows, "always_hold", "R2_NEW")
        per_seed[str(seed)] = {
            "training_rollouts": _summary_group(train_rows),
            "bc_only_greedy": _summary_group(list(bc_rows.values())),
            "r2_ep128_greedy": _summary_group(list(greedy_rows.values())),
            "r2_ep128_sampled": _summary_group(list(sampled_rows.values())),
            "final_greedy_task_ids_equal_immediate_every_scene": task_preserved,
            "final_greedy_mean_bytes": mean_bytes,
            "fixed_late_rule_mean_bytes": baseline["late_mean_spent_bytes"],
            "final_greedy_mean_bytes_strictly_below_late_rule": cost_below_rule,
            "meets_primary_per_seed_success_criterion": task_preserved and cost_below_rule,
            "final_parameter_identity": runtime["r42_run"].load_eval_model("NEW", seed)[1][
                "parameter_identity"],
            "r2_minus_r44_by_mode": r2_r44_by_mode,
        }
    all_success = all(item["meets_primary_per_seed_success_criterion"]
                      for item in per_seed.values())
    training_reports = {}
    for seed in (11, 22, 33):
        training_path = RUN_DIR / f"ROUND45_R2_TRAINING_s{seed}.json"
        training_reports[str(seed)] = {
            "path": str(training_path.resolve()), "sha256": _sha256(training_path),
            "checkpoint_path": str((RUN_DIR / "checkpoints"
                / f"v2_train_s{seed}_ep128.pt").resolve()),
            "checkpoint_sha256": _sha256(RUN_DIR / "checkpoints"
                / f"v2_train_s{seed}_ep128.pt"),
        }
    ledger = runtime["r42_run"].ledger_snapshot(strict=True)
    result = {
        "kind": "round45-r2-final-results/1",
        "status": "complete_research_success" if all_success else "complete_research_goal_not_met",
        "training_version": version, "runner_version": RUNNER_VERSION,
        "round44_primary_eval_matrix_audit_sha256": _sha256(
            R44_DIR / "ROUND44_EVAL_MATRIX_AUDIT.json"),
        "round44_baseline_reuse_sha256": _sha256(R44_DIR / "ROUND44_BASELINE_REUSE.json"),
        "bc_only_control_gate": controls_gate,
        "fixed_baseline": {
            "late_discovery_gate": _summary_group(
                list(fixed_rules["late_discovery_gate"].values())),
            "always_hold": _summary_group(list(fixed_rules["always_hold"].values())),
            "immediate": _summary_group(list(immediate.values())),
            "fixed_late_rule_mean_spent_bytes": baseline["late_mean_spent_bytes"],
        },
        "per_seed": per_seed, "paired_comparisons": comparisons,
        "training_checkpoint_receipts": training_reports,
        "counts": ledger["counts"],
        "budget": {"planned_normal_keys": 528, "normal_used": ledger["normal_used"],
                   "technical_reserve_used": ledger["reserve_used"],
                   "total_claims": ledger["total_used"], "hard_cap": 552,
                   "remaining_hard_cap": 552 - int(ledger["total_used"])},
        "interpretation": {
            "primary_success": "each seed's final greedy model preserves the exact immediate on-time target-ID set in every development scene and has mean measured bytes strictly below the fixed late_discovery_gate baseline",
            "bc_gate_is_not_rl_evidence": True,
            "training_improvement_reported_separately_from_rule_initialization": True,
            "development_data_is_not_independent_confirmation": True,
            "R43_rows_historical_only": True,
        },
    }
    _write_atomic(RUN_DIR / "ROUND45_R2_FINAL_RESULTS.json", result)
    return result


def _write_final_report(result: dict[str, Any] | None, *, gate_failed: bool = False,
                        error: dict[str, Any] | None = None) -> Path:
    ledger_path = RUN_DIR / "runtime" / "ROUND45_R2_EXECUTION_LEDGER.json"
    if ledger_path.is_file():
        ledger = _json(ledger_path)
        counts = ledger.get("counts", {})
    else:
        counts = {}
    task = _json(RUN_DIR / "ROUND45_CURRENT_TASK.json")
    bc_summary = _json(RUN_DIR / "ROUND45_R2_BC_TRAINING_SUMMARY.json")
    lines = [
        "# ROUND45-R2 执行报告", "",
        f"- 分支：`{_git('branch', '--show-current')}`",
        f"- 当前 HEAD：`{_git('rev-parse', 'HEAD')}`",
        f"- 执行阶段：{task.get('stage')}",
        f"- 运行器版本：`{RUNNER_VERSION}`",
        f"- 训练语义版本：`{_json(RUN_DIR / 'ROUND45_CONFIGURATION_DIFF.json')['training_version']}`",
        "", "## 计数", "",
        f"- BC：本次 {bc_summary['this_attempt_real_bc_optimizer_updates']} 次真实更新；含旧失败共 {bc_summary['family_cumulative_real_bc_optimizer_updates']} 次。没有重跑 BC。",
        f"- 环境构造：{counts.get('environment_constructor_calls', 0)} 次；run 调用：{counts.get('run_invocation_calls', 0)} 次；成功返回：{counts.get('run_return_calls', 0)} 次。",
        f"- 正常 claim：{counts.get('normal_used', 0)}/528；技术 reserve：{counts.get('reserve_used', 0)}/24；总 claim：{counts.get('total_used', 0)}/552。",
        f"- PPO 更新：{counts.get('ppo_update_calls', 0)}；Adam step：{counts.get('optimizer_step_calls', 0)}。",
        "", "## 门禁与结果", "",
    ]
    if gate_failed:
        lines += ["- BC-only 控制门禁未通过；未启动 PPO 训练或最终模型评估。",
                  f"- 控制门禁文件：`{(RUN_DIR / 'ROUND45_R2_BC_CONTROL_GATE.json').name}`。"]
    elif error is not None:
        lines += ["- 执行在门禁/运行错误后停止，保留所有 claim、row、日志和检查点。",
                  f"- 错误：`{error.get('type')}: {error.get('message')}`"]
    elif result is not None:
        lines += [f"- 研究判定：`{result['status']}`。",
                  f"- BC-only greedy 控制门禁：`{result['bc_only_control_gate']['status']}`。",
                  f"- 任务数与成本主目标：{result['interpretation']['primary_success']}。",
                  "- 每 seed 详细指标、逐场景目标 ID、与 BC / R44 NEW / immediate / 两条固定规则的配对差异见 `ROUND45_R2_FINAL_RESULTS.json`。"]
        for seed, item in result["per_seed"].items():
            lines.append(
                f"- seed {seed}：精确保留 immediate 目标集合={item['final_greedy_task_ids_equal_immediate_every_scene']}；"
                f"greedy mean bytes={item['final_greedy_mean_bytes']:.2f}；"
                f"late rule mean bytes={item['fixed_late_rule_mean_bytes']:.2f}；"
                f"主目标通过={item['meets_primary_per_seed_success_criterion']}。")
    lines += ["", "## 限制", "",
              "- BC 控制通过只代表固定规则初始化满足进入训练的预设门槛；不代表 RL 增量价值。",
              "- R43 结果仅作历史记录；主要模型对照是固定 R44 NEW 矩阵。",
              "- 16 个开发场景不是独立确认集。未调参、未选 seed、未追加场景或训练。",
              "- 未推送、合并或部署。"]
    report_path = RUN_DIR / "ROUND45_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_fixed_plan() -> dict[str, Any] | None:
    runtime = load_runtime(enforce_prelaunch=True)
    source_receipt = _real_runtime_source_receipt(runtime)
    source_receipt_path = RUN_DIR / "ROUND45_R2_RUNTIME_SOURCE_RECEIPT.json"
    _write_atomic(source_receipt_path, source_receipt)
    controls, control_gate = _run_bc_controls(runtime)
    ledger = runtime["r42_run"].ledger_snapshot(strict=True)
    _task_state("48 BC-only greedy controls complete",
                control_gate=control_gate["status"], control_rows=len(controls),
                environment_constructed=ledger["counts"]["environment_constructor_calls"],
                normal_claims=ledger["normal_used"], reserve_claims=ledger["reserve_used"],
                ppo_updates=ledger["counts"]["ppo_update_calls"])
    if control_gate.get("training_authorized_by_gate") is not True:
        _task_state("stopped after BC-only controls: fixed technical pass, scientific gate failed",
                    control_gate="failed", ppo_updates=0,
                    final_evaluations=0)
        _write_final_report(None, gate_failed=True)
        return None

    _task_state("BC control gate passed; authorized fixed three-seed PPO plan",
                control_gate="passed", ppo_updates=ledger["counts"]["ppo_update_calls"])
    training_reports = _run_training_plan(runtime)
    _task_state("fixed ep128 training complete; starting 96-cell final evaluation",
                training_seeds_complete=sorted(training_reports),
                ppo_updates=int(runtime["r42_run"]._PPO_UPDATE_CALLS))
    eval_rows = _run_final_eval_matrix(runtime)
    runtime["r42_run"].write_execution_ledger(strict=True)
    result = _analyze_completed_run(runtime, control_gate)
    task_stage = ("fixed plan complete: per-seed primary research criterion passed"
                  if result["status"] == "complete_research_success"
                  else "fixed plan complete: primary research criterion not met")
    _task_state(task_stage, control_gate="passed", final_evaluations=len(eval_rows),
                ppo_updates=result["counts"]["ppo_update_calls"],
                environment_constructed=result["counts"]["environment_constructor_calls"],
                normal_claims=result["budget"]["normal_used"],
                reserve_claims=result["budget"]["technical_reserve_used"],
                final_research_status=result["status"])
    _write_final_report(result)
    return result


def _write_stop_state(error: BaseException) -> None:
    error_record = {"type": type(error).__name__, "message": str(error)}
    counts: dict[str, Any] = {}
    try:
        if RUNTIME is not None:
            ledger = RUNTIME["r42_run"].ledger_snapshot(strict=False)
            RUNTIME["r42_run"].write_execution_ledger(strict=False)
            counts = ledger.get("counts", {})
    except BaseException as ledger_error:
        error_record["ledger_error"] = f"{type(ledger_error).__name__}: {ledger_error}"
    _task_state("stopped with preserved R2 evidence", stop_reason=error_record,
                environment_constructed=counts.get("environment_constructor_calls", 0),
                normal_claims=counts.get("normal_used", 0),
                reserve_claims=counts.get("reserve_used", 0),
                ppo_updates=counts.get("ppo_update_calls", 0))
    _write_final_report(None, error=error_record)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Execute the frozen ROUND45-R2 plan")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zero-environment-seam-regression", action="store_true")
    group.add_argument("--prepare-post-bc-freeze", action="store_true")
    group.add_argument("--seal-prelaunch-identity", action="store_true")
    group.add_argument("--execute-fixed-plan", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        if arguments.zero_environment_seam_regression:
            result = run_zero_environment_seam_regression()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return 0 if result.get("status") == "passed" else 1
        if arguments.prepare_post_bc_freeze:
            result = prepare_post_bc_freeze()
            print(json.dumps({"status": result.get("status"),
                              "post_bc_freeze_sha256": _sha256(
                                  RUN_DIR / "ROUND45_R2_POST_BC_ENV_FREEZE.json"),
                              "normal_environment_keys": result["normal_environment_keys"],
                              "technical_reserve_max": result["technical_reserve_max"],
                              "hard_environment_cap": result["hard_environment_cap"]},
                             ensure_ascii=False), flush=True)
            return 0
        if arguments.seal_prelaunch_identity:
            result = seal_prelaunch_identity()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return 0
        if arguments.execute_fixed_plan:
            try:
                result = run_fixed_plan()
            except BaseException as error:
                _write_stop_state(error)
                raise
            if result is None:
                return 2
            print(json.dumps({"status": result["status"],
                              "environment_claims": result["counts"]["total_used"],
                              "ppo_updates": result["counts"]["ppo_update_calls"],
                              "report": str((RUN_DIR / "ROUND45_REPORT.md").resolve())},
                             ensure_ascii=False), flush=True)
            return 0
    except BaseException as error:
        print(f"ROUND45-R2 STOP: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
    return 2


def _key_plan(runtime: dict[str, Any]) -> dict[str, Any]:
    frozen = runtime["frozen"]
    order = [int(value) for value in runtime["common"].scenario_order()]
    expected_order = [int(value) for value in frozen["round42"]["training_order"]]
    if order != expected_order:
        raise RuntimeError("actual R44 scenario_order differs from its frozen 128-episode order")
    controls = []
    training = []
    evaluation = []
    for seed in (11, 22, 33):
        for scene in range(1_301_000, 1_301_016):
            controls.append({
                "scope": "bc_only_greedy_control", "logical_job":
                    f"r45_bc_control_s{seed}_scene{scene}",
                "seed": seed, "mode": "greedy", "scene": scene,
                "normal": True,
            })
    for seed in (11, 22, 33):
        for episode_index, scene in enumerate(order):
            training.append({
                "scope": "ppo_training", "logical_job":
                    f"new_training_s{seed}_e{episode_index}",
                "seed": seed, "episode_index": episode_index,
                "scene": scene, "normal": True,
            })
    for seed in (11, 22, 33):
        for mode in ("greedy", "sampled"):
            for scene in range(1_301_000, 1_301_016):
                evaluation.append({
                    "scope": "final_ep128_eval", "logical_job":
                        f"model_eval_new_s{seed}_{mode}_scene{scene}",
                    "seed": seed, "mode": mode, "scene": scene,
                    "normal": True,
                })
    keys = controls + training + evaluation
    logical_jobs = [str(item["logical_job"]) for item in keys]
    if len(keys) != 528 or len(set(logical_jobs)) != 528:
        raise RuntimeError("R2 frozen environment plan is not 528 unique logical keys")
    if (len(controls), len(training), len(evaluation)) != (48, 384, 96):
        raise RuntimeError("R2 environment plan counts differ from the authorization")
    return {
        "kind": "round45-r2-logical-key-plan/1",
        "training_scenario_order": order,
        "training_scenario_order_sha256": hashlib.sha256(json.dumps(
            order, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "bc_control_keys": controls,
        "ppo_training_keys": training,
        "final_evaluation_keys": evaluation,
        "logical_jobs": logical_jobs,
        "normal_key_count": 528,
        "technical_reserve_claims_max": 24,
        "hard_environment_claim_cap": 552,
        "r43_evaluations_primary": False,
        "r44_new_ep128_matrix_primary": True,
    }


def _bc_model_manifest(summary: dict[str, Any]) -> list[dict[str, Any]]:
    models = []
    for item in summary["per_seed"]:
        seed = int(item["seed"])
        checkpoint = RUN_DIR / "checkpoints" / f"bc_seed{seed}_ep1000.pt"
        # The fixed R2 saver names model files by the receipt's registered path.
        receipt = _json(RUN_DIR / f"BC_CHECKPOINT_RELOAD_VERIFICATION_s{seed}.json")
        checkpoint = Path(receipt["checkpoint_path"])
        metrics_path = RUN_DIR / f"BC_METRICS_s{seed}.json"
        journal_path = RUN_DIR / "bc_journals" / f"BC_OPTIMIZER_STEPS_s{seed}.jsonl"
        expected = {
            "checkpoint_sha256": item["checkpoint_sha256"],
            "readback_receipt_sha256": item["readback_receipt_sha256"],
        }
        actual_receipt_hash = _sha256(RUN_DIR / f"BC_CHECKPOINT_RELOAD_VERIFICATION_s{seed}.json")
        if (not checkpoint.is_file() or _sha256(checkpoint) != expected["checkpoint_sha256"]
                or actual_receipt_hash != expected["readback_receipt_sha256"]
                or receipt.get("status") != "passed"
                or receipt.get("environment_constructed") != 0
                or receipt.get("bc_updates_invoked_and_returned") != 1000
                or receipt.get("post_bc_parameter_sha256") is None
                or receipt.get("metrics_sha256") != _sha256(metrics_path)
                or receipt.get("journal_sha256") != _sha256(journal_path)
                or not metrics_path.is_file() or not journal_path.is_file()):
            raise RuntimeError(f"R2 post-BC seed {seed} source identity is incomplete")
        models.append({
            "seed": seed, "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "readback_receipt_path": str((RUN_DIR / f"BC_CHECKPOINT_RELOAD_VERIFICATION_s{seed}.json").resolve()),
            "readback_receipt_sha256": actual_receipt_hash,
            "metrics_path": str(metrics_path.resolve()),
            "metrics_sha256": _sha256(metrics_path),
            "journal_path": str(journal_path.resolve()),
            "journal_sha256": _sha256(journal_path),
            "parameter_sha256": receipt["post_bc_parameter_sha256"],
            "module_parameter_sha256": receipt["module_parameter_sha256"],
            "optimizer_state_sha256": receipt["optimizer_state_sha256"],
            "updates": int(receipt["bc_updates_invoked_and_returned"]),
            "family_cumulative_updates_including_old_failure": 1000 + int(item["optimizer_steps"]),
        })
    if sum(item["updates"] for item in models) != 3000:
        raise RuntimeError("R2 BC continuation count is not 3,000 real optimizer updates")
    return models


def prepare_post_bc_freeze() -> dict[str, Any]:
    """Create the immutable model/config/528-key freeze before any environment starts."""
    task = _json(RUN_DIR / "ROUND45_CURRENT_TASK.json")
    summary = _json(RUN_DIR / "ROUND45_R2_BC_TRAINING_SUMMARY.json")
    if (summary.get("status") != "complete"
            or summary.get("this_attempt_real_bc_optimizer_updates") != 3000
            or summary.get("family_cumulative_real_bc_optimizer_updates") != 4000
            or summary.get("environment_constructed") != 0
            or task.get("branch") != "exp/comm-rule-warmstart-cost-learning-r2"):
        raise RuntimeError("R2 BC completion/task identity is not eligible for environment freeze")
    if _git("branch", "--show-current") != task["branch"]:
        raise RuntimeError("R2 branch changed before the environment freeze")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked R2 source is dirty before environment freeze")
    runtime = load_runtime(enforce_prelaunch=False)
    plan = _key_plan(runtime)
    model_manifest = _bc_model_manifest(summary)
    config_diff = _json(RUN_DIR / "ROUND45_CONFIGURATION_DIFF.json")
    reuse = _json(RUN_DIR / "ROUND45_R2_REUSE_MANIFEST.json")
    audit_path = R44_DIR / "ROUND44_EVAL_MATRIX_AUDIT.json"
    baseline_path = R44_DIR / "ROUND44_BASELINE_REUSE.json"
    if (_sha256(audit_path) != R44_MATRIX_AUDIT_SHA256
            or reuse["reused_unchanged_inputs"]["round44_primary_eval_matrix_audit_sha256"]
            != R44_MATRIX_AUDIT_SHA256):
        raise RuntimeError("R44 primary matrix audit identity differs from R2 reuse receipt")
    baseline = _fixed_gate_baseline()
    reference_reuse_receipt = RUN_DIR / "runtime" / "ROUND45_R2_REFERENCE_CACHE_REUSE.json"
    if not reference_reuse_receipt.is_file():
        raise RuntimeError("R2 actual immediate-reference cache reuse receipt is missing")
    source_hashes = _source_hashes()
    pre_bc = _json(RUN_DIR / "ROUND45_PRE_BC_FREEZE.json")
    freeze = {
        "kind": "round45-r2-post-bc-environment-freeze/1",
        "status": "frozen_before_environment",
        "branch": task["branch"],
        "source_commit": _git("rev-parse", "HEAD"),
        "pre_bc_source_commit": pre_bc["committed_source_head"],
        "pre_bc_freeze_sha256": _sha256(RUN_DIR / "ROUND45_PRE_BC_FREEZE.json"),
        "bc_training_summary_sha256": _sha256(RUN_DIR / "ROUND45_R2_BC_TRAINING_SUMMARY.json"),
        "bc_real_updates_this_attempt": 3000,
        "bc_real_updates_prior_failed_attempt": 1000,
        "bc_family_cumulative_real_updates": 4000,
        "bc_synthetic_optimizer_updates_previously_used": 2,
        "bc_models": model_manifest,
        "training_version": config_diff["training_version"],
        "runner_version": RUNNER_VERSION,
        "preregistered_runner_version": config_diff["runner_version"],
        "runner_identity_extension": "R2 durable unique-key consumer, receipt replay guard, and reserve attempt binding",
        "reward_timing_version": config_diff["reward_timing_version"],
        "sole_scientific_method_change": "fixed late_discovery_gate_147 offline BC initialization; unchanged R44 constrained PPO",
        "scientific_config_reuse": {
            "r44_final_delivery_head": R44_FINAL_HEAD,
            "r44_primary_matrix_audit_sha256": R44_MATRIX_AUDIT_SHA256,
            "r44_frozen_config_sha256": _sha256(R44_DIR / "ROUND44_FROZEN_CONFIG.json"),
            "gae_lambda": 1.0, "gamma": 1.0, "eta_lambda": 0.5,
            "b0_bytes": 10000.0, "delta": 0.0, "lambda_init": 1.0,
            "observation_width": int(runtime["frozen"]["round42"]["actor_feature_width"]),
            "all_rows": runtime["frozen"]["round42"]["row_filtering"]["policy"],
            "ppo_hyperparameters": copy.deepcopy(
                runtime["frozen"]["round42"]["ppo_hyperparameters"]),
            "action_semantics": "one-shot-send-skip/1",
            "ppo_math": "R44 actual r40_train_resume_adapter and r42 trainer; no copied training math",
            "teacher_inference": False,
        },
        "logical_key_plan": plan,
        "normal_environment_keys": 528,
        "technical_reserve_max": 24,
        "hard_environment_cap": 552,
        "baseline_reuse_manifest_sha256": _sha256(RUN_DIR / "ROUND45_R2_REUSE_MANIFEST.json"),
        "round44_baseline_reuse_sha256": _sha256(baseline_path),
        "r2_reference_cache_reuse_receipt_sha256": _sha256(reference_reuse_receipt),
        "bc_control_fixed_baseline": {
            "late_discovery_gate_mean_spent_bytes": baseline["late_mean_spent_bytes"],
            "immediate_development_scene_row_sha256": {
                str(scene): digest for scene, digest in baseline["reference_row_sha256"].items()},
            "fixed_rule_development_scene_row_sha256": {
                arm: {str(scene): digest for scene, digest in hashes.items()}
                for arm, hashes in baseline["rule_row_sha256"].items()},
        },
        "baseline_policy": "reuse registered 64 immediate + 32 fixed rules + R44 NEW ep128 96; R43 rows historical only",
        "source_files_sha256": source_hashes,
        "activity_at_freeze": {
            "environment_constructed": 0, "ppo_updates": 0,
            "real_bc_optimizer_updates_this_attempt": 3000,
            "bc_family_cumulative_real_updates": 4000,
            "synthetic_optimizer_updates_previously_used": 2,
            "old_bc_attempt_modified": False,
        },
    }
    plan_path = RUN_DIR / "ROUND45_R2_ENVIRONMENT_KEY_PLAN.json"
    _write_atomic(plan_path, plan)
    freeze["logical_key_plan_sha256"] = _sha256(plan_path)
    freeze_path = RUN_DIR / "ROUND45_R2_POST_BC_ENV_FREEZE.json"
    _write_atomic(freeze_path, freeze)
    task.update({
        "stage": "post-BC freeze generated; zero-environment seam gate and prelaunch receipt pending",
        "real_bc_updates": 3000,
        "family_cumulative_real_bc_updates": 4000,
        "synthetic_optimizer_updates": 2,
        "environment_constructed": 0,
        "ppo_updates": 0,
        "normal_environment_keys": 528,
        "technical_reserve_max": 24,
        "hard_environment_cap": 552,
        "post_bc_freeze_sha256": _sha256(freeze_path),
        "remote_push": False,
    })
    _write_atomic(RUN_DIR / "ROUND45_CURRENT_TASK.json", task)
    return freeze


def seal_prelaunch_identity() -> dict[str, Any]:
    freeze_path = RUN_DIR / "ROUND45_R2_POST_BC_ENV_FREEZE.json"
    freeze = _json(freeze_path)
    regression_path = RUN_DIR / "ROUND45_R2_ZERO_ENV_SEAM_REGRESSION.json"
    regression = _json(regression_path)
    if (freeze.get("status") != "frozen_before_environment"
            or regression.get("status") != "passed"
            or regression.get("environment_constructed") != 0
            or regression.get("ppo_updates") != 0
            or regression.get("optimizer_step_calls") != 0):
        raise RuntimeError("R2 prelaunch freeze/seam regression is not clean")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be committed before the R2 prelaunch seal")
    head = _git("rev-parse", "HEAD")
    receipt = {
        "kind": "round45-r2-prelaunch-identity/1",
        "status": "passed",
        "branch": _git("branch", "--show-current"),
        "freeze_commit": head,
        "post_bc_freeze_sha256": _sha256(freeze_path),
        "logical_key_plan_sha256": _sha256(RUN_DIR / "ROUND45_R2_ENVIRONMENT_KEY_PLAN.json"),
        "zero_environment_seam_regression_sha256": _sha256(regression_path),
        "source_files_sha256": _source_hashes(),
        "r44_final_delivery_head": R44_FINAL_HEAD,
        "r44_primary_matrix_audit_sha256": R44_MATRIX_AUDIT_SHA256,
        "normal_keys": 528, "technical_reserve": 24, "hard_cap": 552,
        "environment_constructed_before_seal": 0,
        "ppo_updates_before_seal": 0,
        "real_bc_updates_this_attempt": 3000,
        "family_cumulative_real_bc_updates": 4000,
    }
    _write_atomic(RUN_DIR / "ROUND45_R2_PRELAUNCH_IDENTITY.json", receipt)
    task = _json(RUN_DIR / "ROUND45_CURRENT_TASK.json")
    task["stage"] = "R2 source, post-BC models, 528 keys, and zero-environment seam frozen; ready for 48 controls"
    task["prelaunch_receipt_sha256"] = _sha256(RUN_DIR / "ROUND45_R2_PRELAUNCH_IDENTITY.json")
    task["environment_constructed"] = 0
    task["ppo_updates"] = 0
    _write_atomic(RUN_DIR / "ROUND45_CURRENT_TASK.json", task)
    return receipt


def _real_runtime_source_receipt(runtime: dict[str, Any]) -> dict[str, Any]:
    files = _source_hashes()
    return {
        "kind": "round45-r2-runtime-source-check/1",
        "source_hashes": files,
        "runner_module": str(Path(runtime["r42_run"].__file__).resolve()),
        "counted_environment_entry": "r42_run._execute_cell -> v2_run.build_joint_run counted proxy",
        "training_entry": "r42_run.train_seed -> R40 r40_train_resume_adapter.train_seed",
        "r44_runtime": str(Path(R44_DIR / "r44_runtime.py").resolve()),
        "r44_runtime_hash": files["r44_runtime.py"],
        "actual_atomic_claim_module": str(Path(runtime["r40_common"].__file__).resolve()),
        "normal_cap": int(runtime["r40_common"].NORMAL_CAP),
        "reserve_cap": int(runtime["r40_common"].RESERVE_CAP),
        "hard_cap": int(runtime["r40_common"].HARD_CAP),
    }


def _copy_tree_state(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": RUN_DIR,
        "r44_run_dir": R44_RUNTIME_MODULE.RUN_DIR,
        "common_output": runtime["common"].OUTPUT_DIR,
        "common_logs": runtime["common"].LOGS,
        "common_rows": runtime["common"].ROWS,
        "common_rejected": runtime["common"].REJECTED,
        "common_checkpoints": runtime["common"].CHECKPOINTS,
        "claim_round_dir": runtime["r40_common"].ROUND_DIR,
        "normal_cap": runtime["r40_common"].NORMAL_CAP,
        "reserve_cap": runtime["r40_common"].RESERVE_CAP,
        "hard_cap": runtime["r40_common"].HARD_CAP,
        "normal_launches": runtime["common"].NORMAL_LAUNCHES,
        "reserve_launches": runtime["common"].RESERVE_LAUNCHES,
        "launch_cap": runtime["common"].LAUNCH_CAP,
    }


def run_zero_environment_seam_regression() -> dict[str, Any]:
    """Exercise the actual R2 claim/row/receipt/skip seam without constructing an env."""
    global SEAM_TEST_MODE
    runtime = load_runtime(enforce_prelaunch=False)
    run_module = runtime["r42_run"]
    common = runtime["common"]
    r40_common = runtime["r40_common"]
    runner = runtime["runner"]
    before_steps = int(run_module._OPTIMIZER_STEP_CALLS)
    before_updates = int(run_module._PPO_UPDATE_CALLS)
    before_ckpt = int(runtime["r42_versions"].CHECKPOINT_WRITE_CALLS)
    original_run_episode = runner._run_episode
    original_frozen_verifier = R44_RUNTIME_MODULE.verify_frozen_source
    original_state = _copy_tree_state(runtime)
    calls: dict[str, int] = {}
    failures: dict[str, int] = {}
    SEAM_TEST_MODE = True
    try:
        # The real R44 claim seam invokes the source prelaunch verifier. During this
        # synthetic-only temp-root test, satisfy that precondition with the current HEAD;
        # no production key or environment can be reached from this fixture.
        R44_RUNTIME_MODULE.verify_frozen_source = lambda _frozen: _git("rev-parse", "HEAD")
        with tempfile.TemporaryDirectory(prefix="round45_r2_seam_") as temp_name:
            root = Path(temp_name).resolve()
            test_logs, test_rows, test_rejected, test_checkpoints = (
                root / "logs", root / "rows", root / "rejected", root / "checkpoints")
            for path in (root, test_logs, test_rows, test_rejected, test_checkpoints):
                path.mkdir(parents=True, exist_ok=True)
            runtime["r42_run"].RUN_DIR = root
            R44_RUNTIME_MODULE.RUN_DIR = root
            r40_common.ROUND_DIR = root
            common.OUTPUT_DIR = root
            common.LOGS = test_logs
            common.ROWS = test_rows
            common.REJECTED = test_rejected
            common.CHECKPOINTS = test_checkpoints
            r40_common.NORMAL_CAP = 3
            r40_common.RESERVE_CAP = 1
            r40_common.HARD_CAP = 4

            def fake_episode(label: str, kind: str, scenario_seed: int, *, policy: str,
                             attach: Any = None, attempt: int = 1,
                             controller: Any = None, collector: Any = None,
                             action_rng_seed: int | None = None, tag: str = "",
                             checkpoint_load_identity: dict[str, Any] | None = None) -> dict[str, Any]:
                context = CURRENT_CONTEXT.get()
                if context is None:
                    raise RuntimeError("seam regression fake did not pass through key context")
                logical = str(context["logical_job"])
                calls[logical] = calls.get(logical, 0) + 1
                if logical == "r45_r2_seam_failure" and failures.get(logical, 0) == 0:
                    failures[logical] = 1
                    raise ValueError("synthetic seam failure before environment construction")
                paths = common.cell_paths(kind, int(scenario_seed), int(attempt), tag=tag)
                for name in ("lifecycle", "transcript", "tape"):
                    Path(paths[name]).parent.mkdir(parents=True, exist_ok=True)
                    Path(paths[name]).write_text("\n", encoding="utf-8")
                return {
                    "kind": kind, "label": label, "scenario_seed": int(scenario_seed),
                    "attempt": int(attempt), "policy": policy, "spent_bytes": 0,
                    "on_time": 0, "steps": 0, "artefacts":
                        {name: str(path) for name, path in paths.items()},
                }

            runner._run_episode = fake_episode
            successful_keys = [
                ("r45_r2_seam_success_1", 1301000),
                ("r45_r2_seam_success_2", 1301001),
            ]

            def consume_synthetic(logical: str, scene: int, *, retry: bool = False) -> dict[str, Any]:
                context = {
                    "scope": "seam_regression", "logical_job": logical,
                    "scenario_seed": int(scene), "kind": "train",
                    "tag": logical, "allow_technical_retry": retry,
                }
                token = CURRENT_CONTEXT.set(context)
                try:
                    return run_module._execute_cell(
                        logical, logical, "train", scene,
                        {"arm": "NEW", "scenario_seed": scene,
                         "training_seed": 11, "episode_index": 0},
                        runner_kwargs={"policy": "learned", "tag": logical},
                    )
                finally:
                    CURRENT_CONTEXT.reset(token)

            for logical, scene in successful_keys:
                consume_synthetic(logical, scene)
            claims_before_duplicate = len(list((root / "launch_claims").glob("*/attempt_*.json")))
            duplicate = consume_synthetic(*successful_keys[0])
            claims_after_duplicate = len(list((root / "launch_claims").glob("*/attempt_*.json")))
            if (calls.get(successful_keys[0][0]) != 1 or int(duplicate["attempt"]) != 1
                    or claims_after_duplicate != claims_before_duplicate):
                raise RuntimeError("completed R2 key was launched twice")

            failure_key = ("r45_r2_seam_failure", 1301002)
            try:
                consume_synthetic(*failure_key)
            except ValueError as exc:
                if "synthetic seam failure" not in str(exc):
                    raise
            else:
                raise RuntimeError("synthetic failed key did not preserve its original exception")
            try:
                consume_synthetic(*failure_key, retry=False)
            except RuntimeError as exc:
                if "reserve-retry eligible" not in str(exc):
                    raise
            else:
                raise RuntimeError("failed key was rerun without an explicit reserve retry")
            retried = consume_synthetic(*failure_key, retry=True)
            if (int(retried["attempt"]) != 2
                    or calls.get(failure_key[0]) != 2):
                raise RuntimeError("explicit technical retry did not use one distinct reserve attempt")

            ledger = run_module.ledger_snapshot(strict=True)
            counts = ledger["counts"]
            if (counts["normal_used"] != 3 or counts["reserve_used"] != 1
                    or counts["environment_constructor_calls"] != 0
                    or counts["run_invocation_calls"] != 0
                    or counts["completed_steps"] != 0):
                raise RuntimeError(f"R2 seam regression used or miscounted a real environment: {counts}")
            claim_files = list((root / "launch_claims").glob("*/attempt_*.json"))
            receipts = [(_json(path).get("round45_artifact_receipt") or {})
                        for path in claim_files if _json(path).get("status") == "complete"]
            if len(receipts) != 3 or any(
                    receipt.get("kind") != "round45-r2-artifact-receipt/1"
                    for receipt in receipts):
                raise RuntimeError("R2 seam did not create/verify durable row artifact receipts")
            result = {
                "kind": "round45-r2-zero-environment-consumption-regression/1",
                "status": "passed",
                "actual_claim_consumer": "r40_common.claim_launch",
                "actual_row_executor_seam": "R44 r44_runtime.configure + R43 r42_run._execute_cell",
                "actual_row_writer": "canonical v2_run._booked through R44/R2 receipt wrappers",
                "prelaunch_verifier_scope": "temporary current-head stub inside temp-only synthetic claim fixture",
                "sequential_unique_keys": 3,
                "successful_keys_before_failure": 2,
                "duplicate_completed_key_added_claim": False,
                "completed_key_replay": "returned verified row without another _execute_cell body/claim",
                "synthetic_failed_key": {
                    "attempt_1_exception": "ValueError preserved",
                    "implicit_retry": "refused without creating attempt_2",
                    "explicit_reserve_retry": "one attempt_2 only",
                },
                "claim_receipts": len(receipts),
                "python_executable": sys.executable,
                "actual_device": str(common.DEVICE),
                "production_quota_before_temp_fixture": {
                    "normal": original_state["normal_cap"],
                    "reserve": original_state["reserve_cap"],
                    "hard_cap": original_state["hard_cap"],
                    "normal_launches": original_state["normal_launches"],
                    "reserve_launches": original_state["reserve_launches"],
                    "launch_cap": original_state["launch_cap"],
                },
                "actual_scientific_values": {
                    "observation_width": runtime["frozen"]["round42"]["actor_feature_width"],
                    "gamma": float(common.GAMMA), "gae_lambda": float(common.GAE_LAMBDA),
                    "eta_lambda": float(common.ETA_LAMBDA),
                    "b0_bytes": float(common.B0_BYTES), "delta": float(common.DELTA),
                    "lambda_init": float(common.LAMBDA_INIT),
                    "all_rows": runtime["frozen"]["round42"]["row_filtering"]["policy"],
                },
                "environment_constructed": counts["environment_constructor_calls"],
                "ppo_updates": before_updates - before_updates,
                "optimizer_step_calls": before_steps - before_steps,
                "checkpoint_write_calls": before_ckpt - before_ckpt,
                "normal_slots_in_temp_fixture": counts["normal_used"],
                "reserve_slots_in_temp_fixture": counts["reserve_used"],
                "temporary_fixture_only": True,
            }
    finally:
        runner._run_episode = original_run_episode
        R44_RUNTIME_MODULE.verify_frozen_source = original_frozen_verifier
        runtime["r42_run"].RUN_DIR = original_state["run_dir"]
        R44_RUNTIME_MODULE.RUN_DIR = original_state["r44_run_dir"]
        common.OUTPUT_DIR = original_state["common_output"]
        common.LOGS = original_state["common_logs"]
        common.ROWS = original_state["common_rows"]
        common.REJECTED = original_state["common_rejected"]
        common.CHECKPOINTS = original_state["common_checkpoints"]
        r40_common.ROUND_DIR = original_state["claim_round_dir"]
        r40_common.NORMAL_CAP = original_state["normal_cap"]
        r40_common.RESERVE_CAP = original_state["reserve_cap"]
        r40_common.HARD_CAP = original_state["hard_cap"]
        common.NORMAL_LAUNCHES = original_state["normal_launches"]
        common.RESERVE_LAUNCHES = original_state["reserve_launches"]
        common.LAUNCH_CAP = original_state["launch_cap"]
        SEAM_TEST_MODE = False
    if (int(run_module._OPTIMIZER_STEP_CALLS) != before_steps
            or int(run_module._PPO_UPDATE_CALLS) != before_updates
            or int(runtime["r42_versions"].CHECKPOINT_WRITE_CALLS) != before_ckpt):
        raise RuntimeError("R2 zero-environment seam test changed PPO/optimizer/checkpoint counters")
    result_path = RUN_DIR / "ROUND45_R2_ZERO_ENV_SEAM_REGRESSION.json"
    _write_atomic(result_path, result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
