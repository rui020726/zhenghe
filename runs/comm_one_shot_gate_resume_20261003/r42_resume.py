"""Explicit ROUND42-R resume entry point for the fixed, already-authorized plan.

The original executor remains untouched. This entry point consumes the registered gate and
episode-4 full-state checkpoint in an isolated output directory, then runs only the remaining
fixed cells. Training semantics stay at one-shot-send-skip/1 and the existing runner remains
counted-jointrun-proxy/1.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


RUN_DIR = Path(__file__).resolve().parent
ROUND42_ROOT = RUN_DIR.parents[1]
WORKSPACE_ROOT = ROUND42_ROOT.parent
ORIGINAL_RUN_DIR = ROUND42_ROOT / "runs" / "comm_one_shot_report_selection_20261003_r"
RESUME_VERSION = "round42-gate-and-exact-resume/1"
GATE_PATH_VERSION = "round42-frozen-checkpoint-path-map/2"
TRAINING_VERSION = (
    "constrained-cost-v2/one-shot-send-skip/1/"
    "published-completion-event-at-relay-tick/2"
)
ACTION_SEMANTICS = "one-shot-send-skip/1"
RUNNER_VERSION = "counted-jointrun-proxy/1"
PRELAUNCH_HEAD = "f46fe4703aa03ce6400f63c601c3203ef741dd9d"
REVIEW_HEAD = "2825069eaf9acf0fff87a9a02e69ae2f8e88c785"
LATEST_SOURCE_DELIVERY_HEAD = "5cf41b5edd1b5e13f41b15abc08ae5876da4cd11"
RECOVERY_SHA256 = "a85722e0864456ea3eb735195c7ffcde4be9d42301bdf2bc0781db6747a851ef"
PLAN = {
    "previously_complete": 68,
    "remaining": 604,
    "normal_total": 672,
    "reserve_used": 1,
    "hard_cap": 700,
    "remaining_cells": {
        "seed11_episodes4_127": 124,
        "seed22_training": 128,
        "seed33_training": 128,
        "old_new_greedy_sampled_development": 192,
        "late_and_hold_rule_development": 32,
    },
}
_FULL_REPOSITORY_IDENTITY_HEAD: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


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


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROUND42_ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _git_at(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _git_blob_sha(head: str, path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROUND42_ROOT), "show", f"{head}:{path}"],
        check=True, capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _dependency_path(root: Path, relative: str) -> Path:
    direct = root / relative
    if direct.exists():
        return direct
    return root / "runs" / "comm_cost_v2_rerun_20260928" / relative


def _check_git_ancestor(ancestor: str, head: str) -> None:
    subprocess.run(["git", "-C", str(ROUND42_ROOT), "merge-base", "--is-ancestor",
                    ancestor, head], check=True, stdout=subprocess.DEVNULL)


def verify_resume_source(frozen: dict[str, Any]) -> str:
    """Fail closed on source identity, freeze drift, unregistered code, or dirty inputs."""
    global _FULL_REPOSITORY_IDENTITY_HEAD
    manifest_path = RUN_DIR / "ROUND42_GATE_RESUME_VERSION.json"
    manifest = _read(manifest_path)
    identity = frozen["execution_identity"]
    if manifest.get("resume_execution_version") != RESUME_VERSION:
        raise RuntimeError("explicit resume execution version is not registered")
    if manifest.get("gate_path_identity_version") != GATE_PATH_VERSION:
        raise RuntimeError("gate path identity version is not registered")
    if (frozen["round42"].get("training_version") != TRAINING_VERSION
            or frozen["round42"].get("comm_action_semantics") != ACTION_SEMANTICS
            or identity.get("runner_version") != RUNNER_VERSION
            or frozen["round42"].get("runner_version") != RUNNER_VERSION):
        raise RuntimeError("ROUND42 training/action/runner identity differs from its freeze")

    branch = _git("branch", "--show-current")
    if branch != manifest["repository"]["branch"]:
        raise RuntimeError(f"resume branch changed: {branch}")
    status = _git("status", "--porcelain", "--untracked-files=normal")
    if status:
        raise RuntimeError("source worktree is dirty during the frozen resume: " + status)
    head = _git("rev-parse", "HEAD")
    manifest_relative = manifest_path.relative_to(ROUND42_ROOT).as_posix()
    if _git_blob_sha(head, manifest_relative) != _sha256(manifest_path):
        raise RuntimeError("resume version manifest differs from the committed source")

    full_repo_check = head != _FULL_REPOSITORY_IDENTITY_HEAD
    if full_repo_check:
        expected_ancestors = [PRELAUNCH_HEAD, REVIEW_HEAD, LATEST_SOURCE_DELIVERY_HEAD]
        expected_ancestors.extend(manifest["repository"].get("additional_ancestor_heads", []))
        for ancestor in expected_ancestors:
            _check_git_ancestor(ancestor, head)
        changed = set(filter(None, _git("diff", "--name-only", REVIEW_HEAD, head).splitlines()))
        registered = set(manifest["repository"]["registered_paths_since_review_head"])
        if changed != registered:
            raise RuntimeError(
                f"unregistered repository differences since review head: "
                f"unregistered={sorted(changed - registered)}, "
                f"missing={sorted(registered - changed)}")
        for relative, expected in manifest["repository"]["registered_file_sha256"].items():
            worktree_sha = expected["worktree_sha256"]
            committed_sha = expected["committed_blob_sha256"]
            path = ROUND42_ROOT / Path(relative)
            if not path.is_file() or _sha256(path) != worktree_sha:
                raise RuntimeError(f"registered resume source file changed: {relative}")
            if _git_blob_sha(head, relative) != committed_sha:
                raise RuntimeError(f"committed resume source file changed: {relative}")
        for relative in registered:
            subprocess.run(["git", "-C", str(ROUND42_ROOT), "cat-file", "-e",
                            f"{head}:{relative}"], check=True, stdout=subprocess.DEVNULL)

    declared_config_sha = manifest["original_execution"]["frozen_config_file_sha256"]
    freeze_path = RUN_DIR / "ROUND42_FROZEN_CONFIG.json"
    if _sha256(freeze_path) != declared_config_sha:
        raise RuntimeError("original frozen config file changed")
    original_prelaunch_path = ORIGINAL_RUN_DIR / "runtime" / "ROUND42_PRELAUNCH_IDENTITY.json"
    prelaunch = _read(original_prelaunch_path)
    if (prelaunch.get("head") != PRELAUNCH_HEAD
            or prelaunch.get("frozen_config_sha256")
            != frozen["round42"]["freeze_identity"]["freeze_sha256"]):
        raise RuntimeError("original ROUND42 prelaunch identity changed")
    if _sha256(freeze_path) != _sha256(ORIGINAL_RUN_DIR / "ROUND42_FROZEN_CONFIG.json"):
        raise RuntimeError("copied freeze differs byte-for-byte from the original execution")

    freeze_identity = dict(frozen["round42"]["freeze_identity"])
    declared_freeze_sha = freeze_identity.pop("freeze_sha256", None)
    if _canonical_sha256(freeze_identity) != declared_freeze_sha:
        raise RuntimeError("ROUND42 freeze identity content hash mismatch")
    diff_path = RUN_DIR / "ROUND42_CONFIGURATION_DIFF.json"
    if _canonical_sha256(_read(diff_path)) != frozen["round42"]["freeze_identity"].get(
            "configuration_diff_sha256"):
        raise RuntimeError("ROUND42 configuration diff differs from its freeze")
    if _git_at(Path(identity["canonical_read_only_input_root"]), "rev-parse", "HEAD") != identity[
            "canonical_read_only_input_head"]:
        raise RuntimeError("canonical read-only input checkout HEAD changed")
    if _git_at(Path(identity["round40_worktree"]), "rev-parse", "HEAD") != identity[
            "round40_delivery_head"]:
        raise RuntimeError("ROUND40 delivery checkout HEAD changed")
    if _git_at(Path(identity["round41_worktree"]), "rev-parse", "HEAD") != identity[
            "round41_delivery_head"]:
        raise RuntimeError("ROUND41 delivery checkout HEAD changed")

    old_code = identity["round42_code_sha256"]
    old_run = ORIGINAL_RUN_DIR
    for name, expected in old_code.items():
        old_path = old_run / name
        if not old_path.is_file() or _sha256(old_path) != expected:
            raise RuntimeError(f"original ROUND42 frozen code changed: {name}")
        if name != "r42_analysis.py":
            copied = RUN_DIR / name
            if not copied.is_file() or _sha256(copied) != expected:
                raise RuntimeError(f"copied scientific/runner code changed: {name}")
    new_code = manifest["new_execution_code_sha256"]
    for name, expected in new_code.items():
        path = RUN_DIR / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"new gate/resume code changed: {name}")
    if new_code.get("r42_analysis.py") != _sha256(RUN_DIR / "r42_analysis.py"):
        raise RuntimeError("versioned gate path implementation changed")

    input_root = Path(identity["canonical_read_only_input_root"])
    for relative, expected in identity["frozen_dependency_sha256"].items():
        path = _dependency_path(input_root, relative)
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"frozen production/training dependency changed: {relative}")
    for relative, item in identity["round42_checkout_dependency_sha256"].items():
        path = _dependency_path(ROUND42_ROOT, relative)
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"ROUND42 checkout dependency changed: {relative}")
    round40_dir = Path(identity["round40_worktree"]) / "runs" / (
        "comm_first_report_training_ablation_20261003")
    for name, expected in identity["round40_code_sha256"].items():
        path = round40_dir / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"ROUND40 frozen training code changed: {name}")
    round41_dir = ROUND42_ROOT / "runs" / "comm_temporal_decoder_diagnostic_20261003"
    for name, expected in identity["round41_code_sha256"].items():
        path = round41_dir / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"ROUND41 frozen imported code changed: {name}")
    round40_frozen_path = round40_dir / "ROUND40_FROZEN_CONFIG.json"
    if _canonical_sha256(_read(round40_frozen_path)["round40"]["freeze_identity"]) != frozen[
            "round42"]["round40_freeze_sha256"]:
        raise RuntimeError("ROUND40 freeze identity changed")
    for name, expected in identity["round40_code_sha256"].items():
        if _sha256(round40_dir / name) != expected:
            raise RuntimeError(f"ROUND40 checkpoint/training source drift: {name}")
    if full_repo_check:
        _FULL_REPOSITORY_IDENTITY_HEAD = head
    for item in frozen["round42"]["frozen_search_allocation_checkpoints"].values():
        path = Path(item["path"]).resolve(strict=True)
        if _sha256(path) != item["sha256"]:
            raise RuntimeError(f"frozen PILOC/CTDE checkpoint changed: {path}")
    return head


def _inventory(paths: list[Path], *, relative_to: Path) -> list[dict[str, Any]]:
    found: dict[str, Path] = {}
    for path in paths:
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = [candidate for candidate in path.rglob("*")
                          if candidate.is_file() and "__pycache__" not in candidate.parts]
        else:
            continue
        for candidate in candidates:
            found[candidate.relative_to(relative_to).as_posix()] = candidate
    return [
        {"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for relative, path in sorted(found.items())
    ]


def _original_artifact_paths() -> list[Path]:
    names = (
        "ROUND42_STOP_REPORT.md", "ROUND42_STOP_STATE.json", "ROUND42R_FINAL_LEDGER.json",
        "ROUND42R_FIXED_PLAN_CONTINUATION_REVIEW.md",
        "ROUND42R_FIXED_PLAN_CONTINUATION_STATUS.json", "ROUND42R_LOGICAL_JOB_LEDGER.json",
        "ROUND42R_REFERENCE_RETRY_GATE.json", "ROUND42R_STOP_ARTIFACT_INVENTORY.json",
        "ROUND42R_STOP_DIAGNOSIS.md", "runtime", "launch_claims", "launch_slots", "rows",
        "checkpoints", "training_progress", "logs",
    )
    return [ORIGINAL_RUN_DIR / name for name in names]


def _assert_copied_inputs_match_source() -> dict[str, Any]:
    directory_names = ("launch_claims", "launch_slots", "rows", "checkpoints",
                       "training_progress")
    comparisons = {}
    for name in directory_names:
        old = _inventory([ORIGINAL_RUN_DIR / name], relative_to=ORIGINAL_RUN_DIR)
        new = _inventory([RUN_DIR / name], relative_to=RUN_DIR)
        if old != new:
            raise RuntimeError(f"isolated resume copy differs from existing {name} artifacts")
        comparisons[name] = {"file_count": len(old), "same_paths_sizes_hashes": True}
    old_journal = ORIGINAL_RUN_DIR / "runtime" / "checkpoint_writes.jsonl"
    new_journal = RUN_DIR / "runtime" / "checkpoint_writes.jsonl"
    if _sha256(old_journal) != _sha256(new_journal):
        raise RuntimeError("checkpoint-write history was not copied byte-for-byte")
    comparisons["runtime/checkpoint_writes.jsonl"] = {
        "same_sha256": _sha256(new_journal),
    }
    return comparisons


def _real_rng_probe(payload: dict[str, Any], trainer: Any) -> dict[str, Any]:
    import onpolicy

    prior = trainer.rng_state()
    if not onpolicy.restore_rng(payload):
        raise RuntimeError("resume checkpoint has no restorable Python/NumPy/Torch RNG state")
    expected = {
        "python": [random.random() for _ in range(4)],
        "numpy": np.random.random(4).tolist(),
        "torch_cpu": torch.rand(4, dtype=torch.float32).tolist(),
    }
    onpolicy.restore_rng(prior)
    return expected


def _tree_equal(left: Any, right: Any, *, path: str = "root") -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)
                and tuple(left.shape) == tuple(right.shape)
                and left.dtype == right.dtype
                and torch.equal(left.detach().cpu(), right.detach().cpu())):
            raise RuntimeError(f"resume state tensor mismatch at {path}")
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)
                and set(left) == set(right)):
            raise RuntimeError(f"resume state mapping keys mismatch at {path}")
        for key in left:
            _tree_equal(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
                and len(left) == len(right)):
            raise RuntimeError(f"resume state sequence mismatch at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            _tree_equal(a, b, path=f"{path}[{index}]")
        return
    if isinstance(left, float) or isinstance(right, float):
        if not (isinstance(left, (int, float)) and isinstance(right, (int, float))
                and math.isfinite(float(left)) and math.isfinite(float(right))
                and float(left) == float(right)):
            raise RuntimeError(f"resume state scalar mismatch at {path}: {left!r} != {right!r}")
        return
    if left != right:
        raise RuntimeError(f"resume state mismatch at {path}: {left!r} != {right!r}")


def _module_state(controller: Any) -> dict[str, Any]:
    return {
        "trunk": controller.inner.trunk.state_dict(),
        "action_head": controller.inner.action_head.state_dict(),
        "value_task_head": controller.inner.value_head.state_dict(),
        "value_cost_head": controller.cost_head.state_dict(),
    }


def verify_exact_resume(runtime: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    """Load the approved full state in memory and prove exact restoration, without stepping."""
    trainer = runtime["trainer"]
    r42_run = runtime["r42_run"]
    versions = runtime["r42_versions"]
    checkpoint_dir = RUN_DIR / "checkpoints"
    snapshot = checkpoint_dir / "v2_train_s11_after_b0.pt"
    rolling = checkpoint_dir / "v2_train_s11.pt"
    batch_record_path = checkpoint_dir / "r40_batch_s11_b0_attempt01.json"
    commit_path = checkpoint_dir / "r40_batch_commit_s11_b0.json"
    for path in (snapshot, rolling, batch_record_path, commit_path):
        if not path.is_file():
            raise RuntimeError(f"approved seed 11 resume evidence is missing: {path}")
    if _sha256(snapshot) != RECOVERY_SHA256 or _sha256(rolling) != RECOVERY_SHA256:
        raise RuntimeError("approved after-batch and rolling checkpoints are not byte-identical")
    before_hashes = {path.name: _sha256(path) for path in checkpoint_dir.iterdir()
                     if path.is_file()}
    journal = RUN_DIR / "runtime" / "checkpoint_writes.jsonl"
    journal_before = _sha256(journal)
    step_calls_before = int(r42_run._OPTIMIZER_STEP_CALLS)
    update_calls_before = int(r42_run._PPO_UPDATE_CALLS)
    write_calls_before = int(versions.CHECKPOINT_WRITE_CALLS)

    payload = torch.load(snapshot, map_location="cpu", weights_only=False)
    state_source = payload.get("training_state")
    expected_cursor = {"training_seed": 11, "episodes_done": 4, "updates": 1,
                       "optimizer_steps": 12, "lambda": 1.0125}
    if not isinstance(state_source, dict):
        raise RuntimeError("approved resume checkpoint has no training state")
    for key, expected in expected_cursor.items():
        if state_source.get(key) != expected:
            raise RuntimeError(f"approved resume cursor mismatch for {key}")
    if (payload.get("training_version") != TRAINING_VERSION
            or state_source.get("training_version") != TRAINING_VERSION
            or payload.get("comm_action_semantics") != ACTION_SEMANTICS
            or state_source.get("comm_action_semantics") != ACTION_SEMANTICS
            or payload.get("freeze") != runtime["round42_freeze"]):
        raise RuntimeError("approved resume checkpoint version/freeze identity mismatch")
    order = list(runtime["common"].scenario_order())
    order_sha = hashlib.sha256(json.dumps(order).encode("utf-8")).hexdigest()
    if order_sha != frozen["round42"]["training_order_sha256"]:
        raise RuntimeError("scenario order differs from the frozen ROUND42 training order")
    record = _read(batch_record_path)
    commit = _read(commit_path)
    if (record.get("episodes_done_after") != 4 or record.get("updates_after") != 1
            or record.get("optimizer_steps_after") != 12
            or float(record.get("dual", {}).get("lambda_next", float("nan"))) != 1.0125
            or commit.get("batch_record_sha256") != _sha256(batch_record_path)
            or commit.get("rolling_checkpoint_sha256") != RECOVERY_SHA256
            or commit.get("postupdate_checkpoint_sha256") != RECOVERY_SHA256):
        raise RuntimeError("batch-0 commit receipt does not reconcile with recovery source")
    existing_rows = [_read(path) for path in (RUN_DIR / "rows").glob("*.json")]
    for episode in range(4):
        matches = [row for row in existing_rows
                   if row.get("training_arm") == "NEW"
                   and int(row.get("training_seed", -1)) == 11
                   and int(row.get("episode_index", -1)) == episode]
        if len(matches) != 1:
            raise RuntimeError(f"seed 11 completed episode {episode} row is missing/duplicated")
        row = matches[0]
        if int(row.get("episode_index", -1)) != episode or int(
                row.get("scenario_seed", -1)) != int(order[episode]):
            raise RuntimeError(f"seed 11 episode {episode} does not match frozen order")

    expected_rng = _real_rng_probe(payload.get("rng") or {}, trainer)
    controller, optimizer, restored_state = trainer.load_checkpoint(
        snapshot, freeze=runtime["round42_freeze"])
    _tree_equal(state_source, restored_state, path="training_state")
    expected_modules = payload.get("controller")
    if not isinstance(expected_modules, dict):
        raise RuntimeError("approved resume checkpoint has no controller module state")
    restored_modules = _module_state(controller)
    _tree_equal(expected_modules, restored_modules, path="controller")
    optimizer_state = optimizer.state_dict()
    _tree_equal(payload.get("optimizer"), optimizer_state, path="optimizer")
    expected_parameters = list(controller.parameters)
    actual_parameters = [parameter for group in optimizer.param_groups
                         for parameter in group["params"]]
    if ([id(parameter) for parameter in actual_parameters]
            != [id(parameter) for parameter in expected_parameters]):
        raise RuntimeError("restored Adam parameter order differs from controller parameter set")
    if len(optimizer.state) != len(expected_parameters):
        raise RuntimeError("restored Adam state is incomplete for the registered parameter set")
    for index, parameter in enumerate(expected_parameters):
        state = optimizer.state.get(parameter)
        if not isinstance(state, dict) or not {"step", "exp_avg", "exp_avg_sq"} <= set(state):
            raise RuntimeError(f"restored Adam moments are incomplete at parameter {index}")
        if int(state["step"].item()) != 12:
            raise RuntimeError(f"restored Adam step mismatch at parameter {index}")
    batch_identity = record["controller_parameters_after"]["parameter_sha256"]
    if controller.parameter_summary()["parameter_sha256"] != batch_identity:
        raise RuntimeError("restored model parameter digest differs from committed batch-0 state")
    restored_rng = trainer.rng_state()
    _tree_equal(payload["rng"], restored_rng, path="rng_after_restore")
    actual_rng = {
        "python": [random.random() for _ in range(4)],
        "numpy": np.random.random(4).tolist(),
        "torch_cpu": torch.rand(4, dtype=torch.float32).tolist(),
    }
    _tree_equal(expected_rng, actual_rng, path="next_rng_outputs")
    # Forward and diagnostics above may consume RNG; put the real process back at the registered
    # post-batch state before the continuation code is allowed to run.
    import onpolicy
    if not onpolicy.restore_rng(payload["rng"]):
        raise RuntimeError("could not reapply registered RNG state after resume diagnostics")
    _tree_equal(payload["rng"], trainer.rng_state(), path="rng_reapplied")

    after_hashes = {path.name: _sha256(path) for path in checkpoint_dir.iterdir()
                    if path.is_file()}
    if (before_hashes != after_hashes or journal_before != _sha256(journal)
            or step_calls_before != int(r42_run._OPTIMIZER_STEP_CALLS)
            or update_calls_before != int(r42_run._PPO_UPDATE_CALLS)
            or write_calls_before != int(versions.CHECKPOINT_WRITE_CALLS)):
        raise RuntimeError("read-only recovery check changed counters or checkpoint bytes")
    parameter_names = ("trunk", "action_head", "value_task_head", "value_cost_head")
    return {
        "kind": "round42-exact-resume-verification/1",
        "status": "passed",
        "source": {"path": str(snapshot), "sha256": _sha256(snapshot),
                   "rolling_copy_path": str(rolling), "rolling_copy_sha256": _sha256(rolling),
                   "byte_identical": True},
        "cursor": expected_cursor,
        "training_version": TRAINING_VERSION,
        "action_semantics": ACTION_SEMANTICS,
        "runner_version": RUNNER_VERSION,
        "scenario_order_sha256": order_sha,
        "model": {
            "parameter_sha256": batch_identity,
            "modules_exact": list(parameter_names),
            "controller_tensors_exact": True,
        },
        "optimizer": {
            "parameter_order_matches_controller": True,
            "state_entry_count": len(optimizer.state),
            "adam_step_values": sorted({int(state["step"].item())
                                         for state in optimizer.state.values()}),
            "all_exp_avg_and_exp_avg_sq_exact": True,
        },
        "state_history_and_cursor_exact": True,
        "rng": {
            "streams": ["python_global", "numpy_global", "torch_cpu_global"],
            "next_random_outputs_match": True,
            "restored_again_after_diagnostics": True,
            "deterministic_local_generators": [
                "scenario order uses random.Random(DATA_ORDER_SEED)",
                "PPO minibatch shuffle uses random.Random(batch-derived seed)",
            ],
            "per_episode_action_sampling": "torch CPU generator state; action seed is fixed by episode",
            "cuda_generator_used": False,
        },
        "read_only_counts": {
            "environment_constructions": 0,
            "environment_starts": 0,
            "optimizer_step_calls": 0,
            "ppo_update_calls": 0,
            "checkpoint_writes": 0,
            "checkpoint_files_unchanged": True,
            "checkpoint_write_journal_unchanged": True,
        },
    }


def _claim_records() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((RUN_DIR / "launch_claims").glob("*/*.json")):
        rows.append(_read(path))
    return rows


def _validate_seed_cursor(seed: int, cursor: int, *, allow_no_training_claims: bool = False) -> None:
    claims = _claim_records()
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for claim in claims:
        logical = str(claim.get("logical_job", ""))
        match = re.fullmatch(rf"new_training_s{int(seed)}_e(\d+)", logical)
        if match:
            by_episode.setdefault(int(match.group(1)), []).append(claim)
    if allow_no_training_claims and cursor == 0 and not by_episode:
        return
    for episode in range(cursor):
        attempts = by_episode.get(episode, [])
        if len(attempts) != 1 or attempts[0].get("status") != "complete":
            raise RuntimeError(
                f"seed {seed} checkpoint cursor includes incomplete/duplicate episode {episode}")
    extras = sorted(set(by_episode) - set(range(cursor)))
    if extras:
        raise RuntimeError(
            f"seed {seed} has episode claims beyond its committed checkpoint cursor: {extras}")
    batch_records = []
    for path in (RUN_DIR / "checkpoints").glob(f"r40_batch_s{int(seed)}_b*_attempt*.json"):
        record = _read(path)
        batch_records.append((int(record["batch"]), record))
    batch_records.sort(key=lambda item: item[0])
    expected_batches = cursor // 4
    if [batch for batch, _ in batch_records] != list(range(expected_batches)):
        raise RuntimeError(f"seed {seed} batch records do not match cursor {cursor}")
    for batch, record in batch_records:
        commit_path = RUN_DIR / "checkpoints" / f"r40_batch_commit_s{seed}_b{batch}.json"
        if not commit_path.is_file():
            raise RuntimeError(f"seed {seed} batch {batch} lacks its commit receipt")
        commit = _read(commit_path)
        record_path = next(path for path in (RUN_DIR / "checkpoints").glob(
            f"r40_batch_s{seed}_b{batch}_attempt*.json") if _read(path) == record)
        if commit.get("batch_record_sha256") != _sha256(record_path):
            raise RuntimeError(f"seed {seed} batch {batch} record/commit hash mismatch")
        post = RUN_DIR / "checkpoints" / str(commit["postupdate_checkpoint"])
        if not post.is_file() or _sha256(post) != commit.get("postupdate_checkpoint_sha256"):
            raise RuntimeError(f"seed {seed} batch {batch} postupdate state is not committed")


def reconcile_completed_inputs(runtime: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    common = runtime["common"]
    rows = [_read(path) for path in sorted(common.ROWS.glob("*.json"))]
    refs = [row for row in rows if row.get("reference_arm") == "immediate"]
    s11 = [row for row in rows if row.get("training_arm") == "NEW"
           and int(row.get("training_seed", -1)) == 11]
    if len(refs) != 64 or len({int(row["scenario_seed"]) for row in refs}) != 64:
        raise RuntimeError("the 64 completed immediate references are incomplete or duplicated")
    if len(s11) != 4 or {int(row["episode_index"]) for row in s11} != set(range(4)):
        raise RuntimeError("seed 11 episode 0-3 source rows are incomplete or duplicated")
    if len(rows) != 68:
        raise RuntimeError(f"unexpected pre-resume row count: {len(rows)}")
    _validate_seed_cursor(11, 4)
    _validate_seed_cursor(22, 0, allow_no_training_claims=True)
    _validate_seed_cursor(33, 0, allow_no_training_claims=True)
    if _sha256(RUN_DIR / "ROUND42R_CARRYOVER.json") != frozen["execution_identity"][
            "carryover_manifest_sha256"]:
        raise RuntimeError("ROUND42 failed-claim carryover evidence changed")
    reference_gate = _read(ORIGINAL_RUN_DIR / "ROUND42R_REFERENCE_RETRY_GATE.json")
    if reference_gate.get("status") != "passed" or reference_gate.get(
            "scientific_configuration_and_search_allocation_identity_valid") is not True:
        raise RuntimeError("previously completed reserve reference is not reusable")
    ledger = runtime["r42_run"].ledger_snapshot(strict=True)
    counts = ledger["counts"]
    if (counts["normal_used"] != 68 or counts["reserve_used"] != 1
            or counts["total_used"] != 69 or counts["completed_logical_jobs"] != 68
            or counts["environment_constructor_calls"] != 69
            or counts["completed_steps"] != 68_000
            or counts["ppo_update_calls"] != 1 or counts["optimizer_step_calls"] != 12):
        raise RuntimeError(f"copied ROUND42 cumulative ledger did not reconcile: {counts}")
    return {
        "status": "passed",
        "reused_immediate_reference_rows": len(refs),
        "reused_seed11_training_rows": len(s11),
        "copied_claim_slot_row_checkpoint_files_match_source": True,
        "old_gate_failure_preserved": True,
        "old_stop_state_preserved": True,
        "prior_reference_retry_reused_without_rerun": True,
        "historical_ledger_counts": counts,
    }


def _progress(phase: str, done: int, total: int, last: str,
              detail: dict[str, Any] | None = None) -> None:
    _write_atomic(RUN_DIR / "runtime" / "ROUND42_RESUME_PROGRESS.json", {
        "kind": "round42-explicit-resume-progress/1",
        "resume_execution_version": RESUME_VERSION,
        "phase": phase,
        "completed_cells_in_phase": int(done),
        "planned_cells_in_phase": int(total),
        "last_completed_or_attempted_cell": last,
        "updated_at_unix": time.time(),
        "detail": detail or {},
    })


def _install_execution_identity(runtime: dict[str, Any], version_manifest: dict[str, Any],
                                execution_head: str) -> None:
    """Attach the new execution identity to fresh rows/claims and batch commit evidence."""
    r42_run = runtime["r42_run"]
    original_claim = r42_run._claim

    def versioned_claim(logical_job: str, *, attempt: int, metadata: dict[str, Any]):
        metadata = dict(metadata)
        metadata["resume_execution_version"] = RESUME_VERSION
        metadata["resume_version_manifest_sha256"] = _sha256(
            RUN_DIR / "ROUND42_GATE_RESUME_VERSION.json")
        metadata["resume_execution_head"] = execution_head
        return original_claim(logical_job, attempt=attempt, metadata=metadata)

    r42_run._claim = versioned_claim
    runner = runtime["runner"]
    original_booked = runner._booked

    def versioned_booked(label: str, kind: str, row: dict[str, Any], ledger: Any) -> Any:
        row["execution_identity"] = {
            "resume_execution_version": RESUME_VERSION,
            "gate_path_identity_version": GATE_PATH_VERSION,
            "version_manifest_sha256": _sha256(RUN_DIR / "ROUND42_GATE_RESUME_VERSION.json"),
            "execution_head": execution_head,
        }
        return original_booked(label, kind, row, ledger)

    runner._booked = versioned_booked
    r40_train = runtime["r40_train"]
    original_commit = r40_train._write_batch_commit

    def versioned_commit(training_seed: int, batch_index: int, record_path: Path,
                         checkpoint: Path, postupdate_checkpoint: Path,
                         state: dict[str, Any], parameter_identity: dict[str, Any]) -> None:
        original_commit(training_seed, batch_index, record_path, checkpoint,
                        postupdate_checkpoint, state, parameter_identity)
        receipt_path = RUN_DIR / "checkpoints" / (
            f"r40_batch_commit_s{int(training_seed)}_b{int(batch_index)}.json")
        receipt = _read(receipt_path)
        chain_path = RUN_DIR / "runtime" / "ROUND42_RESUME_STATE_CHAIN.jsonl"
        entry = {
            "kind": "round42-resume-checkpoint-commit/1",
            "resume_execution_version": RESUME_VERSION,
            "execution_head": execution_head,
            "version_manifest_sha256": _sha256(
                RUN_DIR / "ROUND42_GATE_RESUME_VERSION.json"),
            "training_seed": int(training_seed),
            "batch": int(batch_index),
            "episodes_done": int(state["episodes_done"]),
            "updates": int(state["updates"]),
            "optimizer_steps": int(state["optimizer_steps"]),
            "lambda": float(state["lambda"]),
            "training_version": state["training_version"],
            "batch_record": receipt["batch_record"],
            "batch_record_sha256": receipt["batch_record_sha256"],
            "rolling_checkpoint": receipt["rolling_checkpoint"],
            "rolling_checkpoint_sha256": receipt["rolling_checkpoint_sha256"],
            "postupdate_checkpoint": receipt["postupdate_checkpoint"],
            "postupdate_checkpoint_sha256": receipt["postupdate_checkpoint_sha256"],
        }
        with chain_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    r40_train._write_batch_commit = versioned_commit
    runtime["execution_identity_hooks"] = {
        "restore_claim": (r42_run, original_claim),
        "restore_booked": (runner, original_booked),
        "restore_batch_commit": (r40_train, original_commit),
    }


def _restore_execution_identity(runtime: dict[str, Any]) -> None:
    hooks = runtime.get("execution_identity_hooks") or {}
    for key, attribute in (("restore_claim", "_claim"),
                           ("restore_booked", "_booked"),
                           ("restore_batch_commit", "_write_batch_commit")):
        if key in hooks:
            obj, original = hooks[key]
            setattr(obj, attribute, original)


def _load_historical_update_counters(runtime: dict[str, Any]) -> None:
    records = sorted((RUN_DIR / "checkpoints").glob("r40_batch_s*_b*_attempt*.json"))
    steps = sum(int(_read(path).get("update", {}).get("optimizer_steps", 0))
                for path in records)
    runtime["r42_run"]._OPTIMIZER_STEP_CALLS = steps
    runtime["r42_run"]._PPO_UPDATE_CALLS = len(records)
    if steps != 12 or len(records) != 1:
        raise RuntimeError("historical PPO/Adam counters do not match the committed batch-0")


def _verify_final_seed(seed: int, runtime: dict[str, Any], frozen: dict[str, Any],
                       report: dict[str, Any]) -> dict[str, Any]:
    checkpoint = RUN_DIR / "checkpoints" / f"v2_train_s{seed}.pt"
    milestone = RUN_DIR / "checkpoints" / f"v2_train_s{seed}_ep128.pt"
    if not checkpoint.is_file() or not milestone.is_file():
        raise RuntimeError(f"seed {seed} ep128 recoverable checkpoint is missing")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("training_state", {})
    if (int(state.get("episodes_done", -1)) != 128
            or int(state.get("updates", -1)) != 32
            or int(state.get("training_seed", -1)) != seed
            or payload.get("training_version") != TRAINING_VERSION
            or payload.get("comm_action_semantics") != ACTION_SEMANTICS
            or payload.get("freeze") != runtime["round42_freeze"]):
        raise RuntimeError(f"seed {seed}: final checkpoint identity/cursor mismatch")
    if (int(report.get("episodes_done", -1)) != 128
            or int(report.get("updates", -1)) != 32
            or int(report.get("optimizer_steps", -1)) != int(state.get("optimizer_steps", -2))):
        raise RuntimeError(f"seed {seed}: final training report/checkpoint mismatch")
    order = list(runtime["common"].scenario_order())
    rows = [_read(path) for path in (RUN_DIR / "rows").glob("*.json")]
    episodes = [row for row in rows if row.get("training_arm") == "NEW"
                and int(row.get("training_seed", -1)) == seed]
    if (len(episodes) != 128
            or {int(row["episode_index"]) for row in episodes} != set(range(128))):
        raise RuntimeError(f"seed {seed}: final training rows are incomplete/duplicated")
    for row in episodes:
        episode = int(row["episode_index"])
        if int(row["scenario_seed"]) != int(order[episode]):
            raise RuntimeError(f"seed {seed} episode order differs at {episode}")
        if row.get("training_version") != TRAINING_VERSION or row.get(
                "comm_action_semantics") != ACTION_SEMANTICS:
            raise RuntimeError(f"seed {seed} episode {episode} has wrong training identity")
    _validate_seed_cursor(seed, 128)
    return {
        "training_seed": seed,
        "episodes_done": int(state["episodes_done"]),
        "updates": int(state["updates"]),
        "optimizer_steps": int(state["optimizer_steps"]),
        "lambda": float(state["lambda"]),
        "rolling_checkpoint_sha256": _sha256(checkpoint),
        "ep128_checkpoint_sha256": _sha256(milestone),
        "training_rows": len(episodes),
        "scenario_order_sha256": frozen["round42"]["training_order_sha256"],
    }


def _validate_final_ledger(runtime: dict[str, Any]) -> dict[str, Any]:
    ledger = runtime["r42_run"].write_execution_ledger(strict=True)
    counts = ledger["counts"]
    expected = {
        "normal_used": 672,
        "reserve_used": 1,
        "total_used": 673,
        "environment_constructor_calls": 673,
        "joint_run_constructed_calls": 673,
        "run_invocation_calls": 672,
        "joint_run_method_calls": 672,
        "run_return_calls": 672,
        "completed_steps": 672_000,
        "completed_logical_jobs": 672,
        "technical_failed_claims": 1,
        "ppo_update_calls": 96,
        "checkpoint_write_calls": counts["checkpoint_write_journal_entries"],
        "hard_cap": 700,
    }
    for name, value in expected.items():
        if counts.get(name) != value:
            raise RuntimeError(f"fixed-plan cumulative ledger mismatch for {name}: "
                               f"{counts.get(name)} != {value}")
    if counts.get("run_failure_calls") != 0 or counts.get("joint_run_build_failures") != 0:
        raise RuntimeError("new fixed-plan execution contains a technical run/build failure")
    if (counts.get("simulation_started") != 672
            or not counts.get("optimizer_step_count_matches_committed_batches")
            or not counts.get("ppo_update_count_matches_committed_batches")):
        raise RuntimeError("fixed-plan simulation/training counters do not close")
    failed = [claim for claim in ledger["claims"] if claim.get("status") == "failed_technical"]
    if (len(failed) != 1 or int(failed[0].get("attempt", -1)) != 1
            or failed[0].get("logical_job") != "immediate_reference_scene1300000"):
        raise RuntimeError("failure ledger no longer contains only the preserved original failure")
    return ledger


def _write_resume_report(result: dict[str, Any], preservation: dict[str, Any]) -> Path:
    counts = result["execution_ledger"]["counts"]
    paired = result["paired_results"]
    lines = [
        "# ROUND42-R gate repair, exact resume, and fixed-plan completion",
        "",
        f"- Status: **{result['status']}**",
        f"- Resume execution version: `{RESUME_VERSION}`",
        f"- Gate path identity version: `{GATE_PATH_VERSION}`",
        f"- Training version: `{TRAINING_VERSION}`",
        f"- Action semantics / runner: `{ACTION_SEMANTICS}` / `{RUNNER_VERSION}`",
        f"- Execution source HEAD: `{result['execution_head']}`",
        f"- Previous reviewed HEAD: `{REVIEW_HEAD}`; latest preceding delivery HEAD: `"
        f"{LATEST_SOURCE_DELIVERY_HEAD}`; original prelaunch HEAD: `{PRELAUNCH_HEAD}`.",
        "",
        "## Gate and exact recovery",
        "",
        "The seed 11 batch-0 gate was rerun through its complete original audit path. The real "
        "batch record and frozen checkpoint entries differed only in slash spelling; the repaired "
        "comparison retained the canonical root and full directories, rejected collisions, and "
        "rehashed both frozen files. All action, likelihood, reward tick, cost, GAE, dual, "
        "optimizer-scope, parameter-chain, version, and count assertions passed.",
        "",
        f"The approved seed 11 after-batch checkpoint was reloaded from its registered SHA256 `"
        f"{result['exact_resume_verification']['source']['sha256']}`. Model modules, all Adam "
        "steps/moments, lambda/history/cursor, training order, Python/NumPy/Torch CPU RNG state, "
        "and next random outputs matched exactly. Diagnostics reapplied the registered RNG state "
        "before continuation.",
        "",
        "Both checks were read-only: 0 environment constructions, 0 environment starts, 0 "
        "optimizer.step calls, and 0 checkpoint writes. The old failed gate and old stop state "
        "remain intact.",
        "",
        "## Fixed plan and cumulative accounting",
        "",
        f"- Completed before this continuation: 68 successful logical jobs.",
        f"- Completed in this continuation: 604 logical jobs.",
        f"- Final total: {counts['completed_logical_jobs']} / 672 successful logical jobs.",
        f"- Claims: {counts['normal_used']} normal + {counts['reserve_used']} reserve = "
        f"{counts['total_used']} / {counts['hard_cap']} total claims.",
        f"- Environment constructions: {counts['environment_constructor_calls']}; run calls: "
        f"{counts['run_invocation_calls']}; completed steps: {counts['completed_steps']:,}.",
        f"- Preserved original technical failure: {counts['technical_failed_claims']}; new "
        f"technical failures: {counts['run_failure_calls'] + counts['joint_run_build_failures']}.",
        f"- PPO updates: {counts['ppo_update_calls']}; Adam steps: "
        f"{counts['optimizer_step_calls']}; checkpoint writes: {counts['checkpoint_write_calls']}.",
        f"- Technical reserve remaining: {28 - counts['reserve_used']} / 28; hard cap remains 700.",
        "- 64 immediate references and seed 11 episodes 0–3 were reused from the original "
        "records; batch 0 was not replayed.",
        "",
        "## Registered development results",
        "",
        "Greedy and sampled comparisons are separate, on the previously used 16-scene development "
        "block. These results do not establish generalization or an overall zero-loss guarantee.",
        "",
        "| Seed / mode | OLD tasks / bytes | NEW tasks / bytes | NEW−OLD tasks / bytes |",
        "|---|---:|---:|---:|",
    ]
    summaries = paired["group_summaries"]
    for seed in (11, 22, 33):
        for mode in ("greedy", "sampled"):
            key = f"s{seed}_{mode}"
            old = summaries[f"{key}_OLD"]
            new = summaries[f"{key}_NEW"]
            delta = paired["primary_new_minus_old_paired_by_seed_and_mode"][key]
            byte_delta = int(delta["spent_bytes_total_right"]
                              - delta["spent_bytes_total_left"])
            lines.append(
                f"| {seed} / {mode} | {old['on_time_total']} / {old['spent_bytes_total']} | "
                f"{new['on_time_total']} / {new['spent_bytes_total']} | "
                f"{int(delta['on_time_total_delta']):+d} / {byte_delta:+d} |"
            )
    verdict = _read(RUN_DIR / "ROUND42_VERDICT.json")
    lines.extend([
        "",
        f"Task-count and communication-cost verdict indicator: **{verdict.get('new_preserves_task_count_and_reduces_first_reports_and_cost_in_all_seed_mode_pairs')}**.",
        "This is a descriptive check of the frozen development block. The OLD and NEW policies "
        "were evaluated under the same one-shot SEND/SKIP protocol. A full-SEND policy is not "
        "additional savings over immediate; an all-SKIP policy that loses tasks is not a success.",
        "",
        "## Preservation and deliverables",
        "",
        f"- Original run inventory: {preservation['old_source_file_count']} files; all hashes "
        f"unchanged (`{preservation['inventory_sha256']}`).",
        "- The original prelaunch identity, failed gate, stop state, claims, ledgers, batch-0 "
        "rows/tapes/logs, and checkpoints were not rewritten.",
        "- Machine-readable outputs: `ROUND42_FIRST_BATCH_GATES.json`, "
        "`ROUND42_EXACT_RESUME_VERIFICATION.json` (in runtime), `ROUND42R_EXECUTION_LEDGER.json`, "
        "`ROUND42_RESUME_FINAL_RESULT.json`, `ROUND42_CHECKPOINT_INDEX.json`, "
        "`ROUND42_RAW_ARTIFACT_MANIFEST.json`, and `ROUND42_VERDICT.json`.",
        "- No push, deployment, extra scene, or confirmation experiment was performed.",
        "",
    ])
    path = RUN_DIR / "ROUND42_RESUME_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def _run_fixed_plan(runtime: dict[str, Any], frozen: dict[str, Any],
                    gate_results: dict[str, Any], resume_check: dict[str, Any],
                    initial_reconcile: dict[str, Any], execution_head: str) -> dict[str, Any]:
    import r42_analysis
    import r42_run
    import r42_report
    from r42_execute import _validate_reference_block

    start_marker = RUN_DIR / "runtime" / "ROUND42_RESUME_START.json"
    _write_exclusive(start_marker, {
        "kind": "round42-explicit-resume-start/1",
        "status": "running",
        "resume_execution_version": RESUME_VERSION,
        "execution_head": execution_head,
        "version_manifest_sha256": _sha256(RUN_DIR / "ROUND42_GATE_RESUME_VERSION.json"),
        "source_head": LATEST_SOURCE_DELIVERY_HEAD,
        "original_prelaunch_head": PRELAUNCH_HEAD,
        "completed_cells_before_resume": 68,
        "remaining_logical_cells": PLAN["remaining"],
        "environment_starts_before_this_record": 0,
        "exact_resume_check": "passed",
        "first_batch_gate_seed11": "passed",
    })

    started = time.time()
    training_reports: dict[str, Any] = {}
    for seed in (11, 22, 33):
        if seed == 11:
            _validate_seed_cursor(11, 4)
            _progress("new_training_seed_11_resume", 0, 124, "seed=11 next_episode=4",
                      {"resume_cursor": 4, "batch0_replayed": False})
            completed = r42_run.train_seed(11, episodes=128, frozen=frozen)
            if completed.get("history", [{}])[0].get("episode_index") != 4:
                raise RuntimeError("seed 11 continuation did not begin at episode 4")
            final = _verify_final_seed(11, runtime, frozen, completed)
            training_reports["11"] = {
                "first_four_gate": gate_results["11"],
                "completion": {key: value for key, value in completed.items()
                               if key not in ("history", "lambda_history")},
                "lambda_history": completed["lambda_history"],
                "resume_verification": resume_check,
                "final_checkpoint_reconciliation": final,
                "previous_episodes_reused": [0, 1, 2, 3],
            }
            r42_run.write_execution_ledger()
            _write_atomic(RUN_DIR / "runtime" / "ROUND42_RESUME_TRAINING_REPORTS.json",
                          training_reports)
            _progress("new_training_seed_11_complete", 124, 124, "seed=11 ep128",
                      final)
            continue

        _validate_seed_cursor(seed, 0, allow_no_training_claims=True)
        first = r42_run.train_seed(seed, episodes=4, frozen=frozen)
        if (int(first.get("episodes_done", -1)) != 4
                or int(first.get("updates", -1)) != 1):
            raise RuntimeError(f"seed {seed}: first four did not commit exactly batch 0")
        gate = r42_analysis.first_batch_gate(seed, runtime=runtime, frozen=frozen)
        gate_results[str(seed)] = gate
        _write_atomic(RUN_DIR / "runtime" / "gates" / f"seed_{seed}_first_batch.json", gate)
        _progress(f"new_training_seed_{seed}_gate", 1, 1,
                  f"seed={seed} first-batch gate passed", gate[
                      "independent_likelihood"])
        _validate_seed_cursor(seed, 4)
        completed = r42_run.train_seed(seed, episodes=128, frozen=frozen)
        if (completed.get("history", [{}])[0].get("episode_index") != 4
                or int(completed.get("episodes_done", -1)) != 128
                or int(completed.get("updates", -1)) != 32):
            raise RuntimeError(f"seed {seed}: training continuation cursor mismatch")
        final = _verify_final_seed(seed, runtime, frozen, completed)
        training_reports[str(seed)] = {
            "first_four_gate": gate,
            "completion": {key: value for key, value in completed.items()
                           if key not in ("history", "lambda_history")},
            "lambda_history": completed["lambda_history"],
            "final_checkpoint_reconciliation": final,
            "previous_episodes_reused": [],
        }
        r42_run.write_execution_ledger()
        _write_atomic(RUN_DIR / "runtime" / "ROUND42_RESUME_TRAINING_REPORTS.json",
                      training_reports)
        _progress(f"new_training_seed_{seed}_complete", 128, 128,
                  f"seed={seed} ep128", final)

    _validate_reference_block(runtime)
    # The OLD arm is the registered ROUND40 NEW ep128 model; both arms use the frozen one-shot
    # decoder. Each model is loaded once per seed, then evaluated greedily and sampled.
    evaluation_total = 192
    evaluation_done = 0
    for arm in ("OLD", "NEW"):
        for seed in (11, 22, 33):
            controller, identity = r42_run.load_eval_model(arm, seed)
            parameter_sha = identity["parameter_identity"]["parameter_sha256"]
            for mode in ("greedy", "sampled"):
                for scene in r42_run.DEV_SCENARIOS:
                    verify_resume_source(frozen)
                    row = r42_run.run_model_eval(
                        arm=arm, training_seed=seed, mode=mode, scenario=int(scene),
                        controller=controller, identity=identity)
                    if controller.parameter_summary()["parameter_sha256"] != parameter_sha:
                        raise RuntimeError(f"read-only {arm} evaluation changed seed {seed} model")
                    evaluation_done += 1
                    _progress("old_new_development", evaluation_done, evaluation_total,
                              row["label"], {"arm": arm, "seed": seed, "mode": mode,
                                             "scenario": int(scene)})

    rule_done = 0
    for arm in ("late_discovery_gate", "always_hold"):
        for scene in r42_run.DEV_SCENARIOS:
            verify_resume_source(frozen)
            row = r42_run.run_rule(arm, int(scene))
            rule_done += 1
            _progress("rule_development", rule_done, 32, row["label"],
                      {"arm": arm, "scenario": int(scene)})

    ledger = _validate_final_ledger(runtime)
    paired = r42_analysis.analyze_all(runtime=runtime, frozen=frozen)
    result = {
        "kind": "round42r-fixed-plan-resume-result/1",
        "status": "complete",
        "resume_execution_version": RESUME_VERSION,
        "gate_path_identity_version": GATE_PATH_VERSION,
        "execution_head": execution_head,
        "original_prelaunch_head": PRELAUNCH_HEAD,
        "source_review_head": REVIEW_HEAD,
        "latest_source_delivery_head": LATEST_SOURCE_DELIVERY_HEAD,
        "duration_seconds": round(time.time() - started, 2),
        "training_version": TRAINING_VERSION,
        "comm_action_semantics": ACTION_SEMANTICS,
        "runner_version": RUNNER_VERSION,
        "reference_retry_gate_reused": _read(ORIGINAL_RUN_DIR / "ROUND42R_REFERENCE_RETRY_GATE.json"),
        "initial_state_reconciliation": initial_reconcile,
        "seed11_gate": gate_results["11"],
        "exact_resume_verification": resume_check,
        "first_batch_gates": gate_results,
        "training_reports": training_reports,
        "paired_results": paired,
        "execution_ledger": ledger,
    }
    _write_atomic(RUN_DIR / "runtime" / "ROUND42_RESUME_EXECUTION_RESULT.json", result)
    report_outputs = r42_report.write_final_deliverables(
        runtime=runtime, frozen=frozen, result={
            "status": "complete",
            "duration_seconds": result["duration_seconds"],
            "prelaunch_head": execution_head,
            "training_version": TRAINING_VERSION,
            "runner_version": RUNNER_VERSION,
            "reference_retry_gate": result["reference_retry_gate_reused"],
            "first_batch_gates": gate_results,
            "training_reports": training_reports,
            "paired_results": paired,
            "execution_ledger": ledger,
        })
    _write_atomic(RUN_DIR / "ROUND42R_EXECUTION_LEDGER.json", ledger)
    _write_atomic(RUN_DIR / "ROUND42R_LOGICAL_JOB_LEDGER.json", {
        "kind": "round42r-logical-job-ledger/2",
        "status": "complete",
        "plan_total": 672,
        "successfully_completed": 672,
        "completed_before_resume": 68,
        "completed_after_resume": 604,
        "reused_immediate_reference_scenes": sorted(int(row["scenario_seed"])
                                                       for row in [_read(path) for path in
                                                           (RUN_DIR / "rows").glob(
                                                               "v2_reference_s*_a*.json")]),
        "original_failed_attempt_preserved": True,
        "reserve_retry_attempt_reused": True,
        "normal_claims_including_original_failure": 672,
        "reserve_claims": 1,
        "technical_reserve_remaining": 27,
        "hard_cap": 700,
    })
    _write_atomic(RUN_DIR / "ROUND42_RESUME_FINAL_RESULT.json", result)
    _progress("complete", 604, 604, "all remaining fixed-plan cells complete",
              ledger["counts"])
    return {"result": result, "report_outputs": report_outputs}


def run() -> int:
    start_marker = RUN_DIR / "runtime" / "ROUND42_RESUME_START.json"
    if start_marker.exists():
        raise RuntimeError(
            "a resume attempt already exists in this isolated directory; refusing duplicate execution")
    if (RUN_DIR / "runtime" / "ROUND42_RESUME_EXECUTION_RESULT.json").exists():
        raise RuntimeError("resume result already exists; refusing duplicate execution")
    frozen = _read(RUN_DIR / "ROUND42_FROZEN_CONFIG.json")
    version_manifest = _read(RUN_DIR / "ROUND42_GATE_RESUME_VERSION.json")
    head = verify_resume_source(frozen)
    if head != _git("rev-parse", "HEAD"):
        raise RuntimeError("source HEAD changed during the initial resume verification")
    original_run_hash = _inventory(_original_artifact_paths(), relative_to=ORIGINAL_RUN_DIR)
    input_copy_comparison = _assert_copied_inputs_match_source()
    _write_atomic(RUN_DIR / "runtime" / "ROUND42_OLD_SOURCE_INVENTORY.json", {
        "kind": "round42-old-source-preservation-inventory/1",
        "source_run_directory": str(ORIGINAL_RUN_DIR),
        "file_count": len(original_run_hash),
        "files": original_run_hash,
    })

    round40_path = Path(frozen["execution_identity"]["round40_worktree"]) / "runs" / (
        "comm_first_report_training_ablation_20261003") / "ROUND40_FROZEN_CONFIG.json"
    round40_frozen = _read(round40_path)
    import r42_analysis
    import r42_run

    runtime = r42_run.configure(frozen, round40_frozen["round40"]["freeze_identity"])
    runtime["r42_run"] = r42_run
    r42_run.verify_frozen_source = verify_resume_source
    _load_historical_update_counters(runtime)
    input_reconcile = reconcile_completed_inputs(runtime, frozen)
    identity_before = {
        "execution_version": RESUME_VERSION,
        "gate_path_identity_version": GATE_PATH_VERSION,
        "execution_head": head,
        "version_manifest_sha256": _sha256(RUN_DIR / "ROUND42_GATE_RESUME_VERSION.json"),
        "training_version": TRAINING_VERSION,
        "action_semantics": ACTION_SEMANTICS,
        "runner_version": RUNNER_VERSION,
        "prelaunch_head": PRELAUNCH_HEAD,
        "known_review_head": REVIEW_HEAD,
        "latest_source_delivery_head": LATEST_SOURCE_DELIVERY_HEAD,
        "old_checkpoint_resume_source_sha256": RECOVERY_SHA256,
        "new_gate_code_sha256": version_manifest["new_execution_code_sha256"],
        "scientific_dependencies_unchanged": True,
    }
    _write_exclusive(RUN_DIR / "runtime" / "ROUND42_RESUME_EXECUTION_IDENTITY.json",
                     identity_before)
    _write_exclusive(RUN_DIR / "runtime" / "ROUND42_RESUME_INPUT_RECONCILIATION.json",
                     {"copied_inputs": input_copy_comparison,
                      "completed_state": input_reconcile})

    # The old full gate is called unchanged here except for its now-versioned path helper. Its
    # checks include every assertion after the old failure point, all from existing rows/tapes.
    gate = r42_analysis.first_batch_gate(11, runtime=runtime, frozen=frozen)
    if gate.get("status") != "passed" or len(gate.get(
            "frozen_checkpoint_path_identity", [])) != 2:
        raise RuntimeError("repaired real seed-11 first-batch gate did not pass completely")
    _write_atomic(RUN_DIR / "runtime" / "gates" / "seed_11_first_batch.json", gate)
    if (int(r42_run._OPTIMIZER_STEP_CALLS) != 12
            or int(r42_run._PPO_UPDATE_CALLS) != 1
            or int(runtime["r42_versions"].CHECKPOINT_WRITE_CALLS) != 4):
        raise RuntimeError("real gate changed the committed training counters")
    _write_atomic(RUN_DIR / "runtime" / "ROUND42_REAL_GATE_RECHECK.json", {
        "kind": "round42-real-seed11-first-batch-gate-recheck/1",
        "status": "passed",
        "gate_result": gate,
        "optimizer_steps_before_and_after": 12,
        "ppo_updates_before_and_after": 1,
        "checkpoint_writes_before_and_after": 4,
        "environment_constructions": 0,
        "environment_starts": 0,
        "optimizer_steps_during_gate": 0,
        "checkpoint_writes_during_gate": 0,
    })

    resume_check = verify_exact_resume(runtime, frozen)
    _write_atomic(RUN_DIR / "runtime" / "ROUND42_EXACT_RESUME_VERIFICATION.json", resume_check)
    if resume_check.get("status") != "passed":
        raise RuntimeError("exact resume verification failed")
    _install_execution_identity(runtime, version_manifest, head)
    try:
        completed = _run_fixed_plan(runtime, frozen, {"11": gate}, resume_check,
                                    input_reconcile, head)
    except BaseException as error:
        failure = {
            "kind": "round42-explicit-resume-stop/1",
            "status": "stopped_without_automatic_retry",
            "exception": f"{type(error).__name__}: {error}",
            "time_unix": time.time(),
            "traceback": __import__("traceback").format_exc(),
        }
        _write_atomic(RUN_DIR / "runtime" / "ROUND42_RESUME_STOP.json", failure)
        try:
            r42_run.write_execution_ledger(strict=False)
        except BaseException:
            pass
        raise
    finally:
        _restore_execution_identity(runtime)
        r42_run.restore_runtime_hooks()

    end_inventory = _inventory(_original_artifact_paths(), relative_to=ORIGINAL_RUN_DIR)
    if end_inventory != original_run_hash:
        raise RuntimeError("one or more ROUND42 stopped-run source artifacts changed")
    preservation = {
        "kind": "round42-old-source-preservation-result/1",
        "status": "passed",
        "old_source_file_count": len(end_inventory),
        "old_source_sha256_inventory_unchanged": True,
        "inventory_sha256": _canonical_sha256(end_inventory),
        "old_gate_failure_preserved": True,
        "old_stop_state_preserved": True,
        "old_claims_ledgers_rows_tapes_checkpoints_unchanged": True,
    }
    _write_atomic(RUN_DIR / "ROUND42_OLD_SOURCE_PRESERVATION.json", preservation)
    completed["result"]["old_source_preservation"] = preservation
    _write_atomic(RUN_DIR / "ROUND42_RESUME_FINAL_RESULT.json", completed["result"])
    _write_atomic(RUN_DIR / "runtime" / "ROUND42_RESUME_EXECUTION_RESULT.json",
                  completed["result"])
    report_path = _write_resume_report(completed["result"], preservation)
    start_state = _read(start_marker)
    start_state["status"] = "complete"
    start_state["finished_at_unix"] = time.time()
    start_state["final_counts"] = completed["result"]["execution_ledger"]["counts"]
    _write_atomic(start_marker, start_state)
    print(json.dumps({
        "status": "complete",
        "execution_head": head,
        "report": str(RUN_DIR / "ROUND42_REPORT.md"),
        "resume_report": str(report_path),
        "counts": completed["result"]["execution_ledger"]["counts"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
