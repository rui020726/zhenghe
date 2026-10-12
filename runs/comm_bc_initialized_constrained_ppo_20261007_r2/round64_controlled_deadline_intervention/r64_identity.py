"""Single immutable candidate binding shared by the R64 executor and guard."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
CANDIDATE_PATH = HERE / "ROUND64_RESUME_SEAM_CANDIDATE.json"
HISTORICAL_EXECUTION_VERSIONS = frozenset(
    f"round64-controlled-deadline-intervention/{n}" for n in (2, 3, 4, 5, 6, 7)
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R64ExecutionBinding:
    task_id: str
    branch: str
    execution_version: str
    candidate_sha256: str
    approval_path: Path
    approval_sha256: str
    source_freeze_path: Path
    source_freeze_sha256: str
    scene: int
    focus_target: str
    seed: int
    mode: str
    deadline_deltas: tuple[int, ...]
    horizon_steps: int
    goals: int
    budget_bytes: int
    ack: bool
    observation_dimensions: int
    model_path_relative_to_r2: str
    model_file_sha256: str
    model_parameter_sha256: str
    normal_cap: int
    reserve_cap: int
    hard_cap: int


def load_binding(path: Path = CANDIDATE_PATH) -> R64ExecutionBinding:
    path = Path(path).resolve()
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("kind") != "round64-approved-resume-freeze/1":
        raise RuntimeError("unsupported R64 candidate freeze")
    experiment = doc["fixed_experiment"]
    limits = doc["attempt_limits"]
    source = doc["source_freeze"]
    approval = doc["approval"]
    approval_path = (path.parent / approval["path"]).resolve()
    if _sha256(approval_path) != approval["sha256"]:
        raise RuntimeError("R64 cap amendment approval hash mismatch")
    approval_doc = json.loads(approval_path.read_text(encoding="utf-8"))
    approved = approval_doc.get("approval_scope", {})
    if (approval_doc.get("kind") != "round64-explicit-cap-amendment/1" or
            approval_doc.get("human_answer") != "批准累计最多13次，完成原矩阵" or
            approved.get("cumulative_hard_cap") != int(limits["hard_cap"]) or
            approved.get("normal_cap") != int(limits["normal"]) or
            approved.get("reserve_cap") != int(limits["reserve"]) or
            approved.get("new_attempts_authorized") != 4 or
            approved.get("failed_attempts_refund") is not False):
        raise RuntimeError("R64 candidate limits are not covered by the explicit approval")
    return R64ExecutionBinding(
        task_id=doc["task"], branch=doc["branch"],
        execution_version=doc["candidate_execution_version"],
        candidate_sha256=_sha256(path),
        approval_path=approval_path,
        approval_sha256=approval["sha256"],
        source_freeze_path=(path.parent / source["path"]).resolve(),
        source_freeze_sha256=source["sha256"],
        scene=int(experiment["scene"]), focus_target=experiment["focus_target"],
        seed=int(experiment["seed"]), mode=experiment["mode"],
        deadline_deltas=tuple(int(x) for x in experiment["deadline_deltas"]),
        horizon_steps=int(experiment["horizon_steps"]), goals=int(experiment["goals"]),
        budget_bytes=int(experiment["budget_bytes"]), ack=bool(experiment["ack"]),
        observation_dimensions=int(experiment["observation_dimensions"]),
        model_path_relative_to_r2=experiment["model_path_relative_to_r2"],
        model_file_sha256=experiment["model_file_sha256"],
        model_parameter_sha256=experiment["model_parameter_sha256"],
        normal_cap=int(limits["normal"]), reserve_cap=int(limits["reserve"]),
        hard_cap=int(limits["hard_cap"]),
    )


BINDING = load_binding()


def validate_live_binding(
    *,
    executor_version: str,
    guard_version: str,
    model_identity: Mapping[str, Any],
    guard_limits: Mapping[str, int],
    raw_root: Path,
    candidate_path: Path = CANDIDATE_PATH,
) -> dict[str, Any]:
    """Fail before a slot is claimed unless all live values match this candidate."""
    binding = BINDING
    candidate_path = Path(candidate_path).resolve()
    if _sha256(candidate_path) != binding.candidate_sha256:
        raise RuntimeError("R64 candidate freeze changed after module binding")
    doc = json.loads(candidate_path.read_text(encoding="utf-8"))
    if doc.get("candidate_execution_version") != binding.execution_version:
        raise RuntimeError("R64 candidate execution version differs from the shared binding")
    if _sha256(binding.source_freeze_path) != binding.source_freeze_sha256:
        raise RuntimeError("historical ROUND64_FREEZE changed; candidate identity is stale")
    approval_ref = doc.get("approval", {})
    if (approval_ref.get("path") != binding.approval_path.name or
            approval_ref.get("sha256") != binding.approval_sha256 or
            _sha256(binding.approval_path) != binding.approval_sha256):
        raise RuntimeError("R64 cap amendment approval changed after binding")
    source_freeze = json.loads(binding.source_freeze_path.read_text(encoding="utf-8"))
    source_model = source_freeze["model"]
    source_world = source_freeze["world"]
    source_intervention = source_freeze["intervention"]
    frozen_values = (
        source_freeze.get("task") == binding.task_id,
        source_freeze.get("branch") == binding.branch,
        source_model.get("path_relative_to_r2") == binding.model_path_relative_to_r2,
        source_model.get("file_sha256") == binding.model_file_sha256,
        source_model.get("parameter_sha256") == binding.model_parameter_sha256,
        source_model.get("seed") == binding.seed,
        source_model.get("mode") == binding.mode,
        source_world.get("scene") == binding.scene,
        source_world.get("focus_target") == binding.focus_target,
        source_world.get("steps") == binding.horizon_steps,
        source_world.get("num_goals") == binding.goals,
        source_world.get("comm_budget_bytes") == binding.budget_bytes,
        source_world.get("comm_ack") is binding.ack,
        source_intervention.get("deltas") == list(binding.deadline_deltas),
        source_intervention.get("feature_schema_dimensions") == binding.observation_dimensions,
    )
    if not all(frozen_values):
        raise RuntimeError("R64 candidate values differ from the immutable historical freeze")
    original_limits = source_freeze.get("authorization", {})
    approval_doc = json.loads(binding.approval_path.read_text(encoding="utf-8"))
    approved_limits = approval_doc.get("approval_scope", {})
    if (original_limits.get("hard_attempt_cap") != 12 or
            approved_limits.get("cumulative_hard_cap") != binding.hard_cap or
            approved_limits.get("normal_cap") != binding.normal_cap or
            approved_limits.get("reserve_cap") != binding.reserve_cap or
            approved_limits.get("new_attempts_authorized") != 4):
        raise RuntimeError("R64 cap amendment does not preserve and explicitly supersede the old cap")
    if executor_version != guard_version:
        raise RuntimeError(
            f"R64 executor/guard version mismatch: {executor_version!r} != {guard_version!r}"
        )
    if executor_version != binding.execution_version:
        raise RuntimeError(
            f"R64 executor version {executor_version!r} differs from candidate "
            f"{binding.execution_version!r}"
        )
    if guard_version != binding.execution_version:
        raise RuntimeError(
            f"R64 counter guard version {guard_version!r} differs from candidate "
            f"{binding.execution_version!r}"
        )
    expected_model = {
        "file_sha256": binding.model_file_sha256,
        "parameter_sha256": binding.model_parameter_sha256,
        "execution_version": binding.execution_version,
        "load_mode": "read_only",
        "load_scope": "model_only",
        "optimizer_restored": False,
        "rng_restored": False,
        "training_cursor_restored": False,
    }
    mismatched = {key: (model_identity.get(key), value)
                  for key, value in expected_model.items()
                  if model_identity.get(key) != value}
    model_path = str(model_identity.get("model_path", "")).replace("\\", "/")
    if not model_path.endswith(binding.model_path_relative_to_r2):
        mismatched["model_path"] = (model_identity.get("model_path"),
                                    binding.model_path_relative_to_r2)
    if mismatched:
        raise RuntimeError(f"R64 model row differs from candidate binding: {mismatched}")
    expected_limits = {
        "normal_cap": binding.normal_cap,
        "reserve_cap": binding.reserve_cap,
        "hard_cap": binding.hard_cap,
    }
    if dict(guard_limits) != expected_limits:
        raise RuntimeError(
            f"R64 guard limits differ from candidate binding: {dict(guard_limits)}"
        )
    if Path(raw_root).resolve() != (HERE / "raw").resolve():
        raise RuntimeError("R64 live raw root differs from the candidate-bound output root")
    return {
        "candidate_sha256": binding.candidate_sha256,
        "source_freeze_sha256": binding.source_freeze_sha256,
        "execution_version": binding.execution_version,
        "model_file_sha256": binding.model_file_sha256,
        "model_parameter_sha256": binding.model_parameter_sha256,
        "limits": expected_limits,
        "validated": True,
    }
