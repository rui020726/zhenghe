"""Narrow, versioned identity bridge for the authorized ROUND44 ep4 continuation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from r44_prepare import RUN_DIR, TRAINING_VERSION


BASE_EXECUTION_HEAD = "2353c4616746cf0a7c9fcb3a519a7042a28ba176"
TASK_START_HEAD = "2af9f7e66982eb046725057e026c05512b788169"
R1_EXECUTION_HEAD = "43399eb9b74031e1ee88752618c04e84082bb811"
R1_IDENTITY_SHA256 = "ffc75ffeca743134a9f0df0bff173e1b0b4e6f544bf88ad4d3ef31119336c860"
R2_EXECUTION_HEAD = "453924cf4abb0dcc649c1d877236fe12c73e12e5"
R2_IDENTITY_SHA256 = "15aa3cbb34f6b52574d4852eb42503fd7c491a6119302e2da7945dfdde28252b"
R2_FAILURE_SHA256 = "7d2a4e8b7a8f73aa28efcd2936d0afdc30016ab6a11ebd585f66afec384c52e2"
BRANCH = "exp/comm-dual-response-ablation"
STATE_BRIDGE_VERSION = "round44-exact-resume-state-bridge/1"
MECHANICAL_VERSION = "round44-resume-comparator-and-restore/3"
RESUME_PROGRESS_KIND = "round44-execution-progress-resume-r3/3"
WHITELIST_PATH = RUN_DIR / "ROUND44_RESUME_STATE_WHITELIST_R1.json"
SNAPSHOT_MANIFEST_PATH = RUN_DIR / "runtime" / "ROUND44_PRE_RESUME_R1_SNAPSHOT_MANIFEST.json"
R1_IDENTITY_PATH = RUN_DIR / "runtime" / "ROUND44_RESUME_EXECUTION_IDENTITY_R1.json"
R1_GATE_PATH = RUN_DIR / "runtime" / "ROUND44_EXACT_RESUME_GATE_R1.json"
R1_COMPARATOR_TEST_PATH = RUN_DIR / "runtime" / "ROUND44_COMPARATOR_REGRESSION_R1.json"
R1_READY_PATH = RUN_DIR / "runtime" / "ROUND44_RESUME_READY_R1.json"
R2_IDENTITY_PATH = RUN_DIR / "runtime" / "ROUND44_RESUME_EXECUTION_IDENTITY_R2.json"
R2_FAILURE_PATH = RUN_DIR / "runtime" / "ROUND44_R2_GATE_ATTEMPT_FAILURE.json"
R2_COMPARATOR_TEST_PATH = RUN_DIR / "runtime" / "ROUND44_COMPARATOR_REGRESSION_R2.json"
R2_GATE_PATH = RUN_DIR / "runtime" / "ROUND44_EXACT_RESUME_GATE_R2.json"
R2_READY_PATH = RUN_DIR / "runtime" / "ROUND44_RESUME_READY_R2.json"
IDENTITY_PATH = RUN_DIR / "runtime" / "ROUND44_RESUME_EXECUTION_IDENTITY_R3.json"
RESUME_PROGRESS_PATH = RUN_DIR / "runtime" / "ROUND44_EXECUTION_PROGRESS_RESUME_R3.json"
GATE_PATH = RUN_DIR / "runtime" / "ROUND44_EXACT_RESUME_GATE_R3.json"
COMPARATOR_TEST_PATH = RUN_DIR / "runtime" / "ROUND44_COMPARATOR_REGRESSION_R3.json"
READY_PATH = RUN_DIR / "runtime" / "ROUND44_RESUME_READY_R3.json"
OLD_PROJECTION_PATH = RUN_DIR / "runtime" / "ROUND44_OLD_PROJECTION_FOUR_EPISODES_R2.json"
SOURCE_CONFIG_RECONCILIATION_PATH = (
    RUN_DIR / "runtime" / "ROUND44_SOURCE_CONFIG_RECONCILIATION_R2.json"
)
R1_FAILURE_REPRODUCTION_PATH = RUN_DIR / "runtime" / "ROUND44_OLD_COMPARATOR_REPRODUCTION.json"
SNAPSHOT_ROOT = RUN_DIR / "runtime" / "ROUND44_PRE_RESUME_R1_SNAPSHOT"
STOP_FILES = (
    "ROUND44_STATUS.json",
    "ROUND44_STOP_REPORT.md",
    "ROUND44_STOP_STATE.json",
    "ROUND44_STOP_ARTIFACT_INVENTORY.json",
    "ROUND44_EXECUTION_FAILURE_CAPTURE_SEED11_B0.txt",
)
MUTABLE_PRE_RESUME_PATHS = (
    "runtime/ROUND44_EXECUTION_LEDGER.json",
    "runtime/ROUND44_EXECUTION_PROGRESS.json",
    "runtime/optimizer_step_calls.jsonl",
    "runtime/ppo_update_calls.jsonl",
    "runtime/checkpoint_writes.jsonl",
    "checkpoints/v2_train_s11.pt",
)
SOURCE_FILES = (
    "r44_execute.py", "r44_runtime.py", "r44_resume_bridge.py",
    "r44_resume_gate.py", "test_r44_resume_gate.py",
    "RUNNER_FIX.md", "ROUND44_RESUME_STATE_WHITELIST_R1.json",
    "ROUND44_CURRENT_CONTINUATION_BASIS.md", "ROUND44_CURRENT_TASK.json",
)
ALLOWED_TASK_DELTA = {
    "PROJECT_STATUS.md",
    f"runs/{RUN_DIR.name}/ROUND44_CURRENT_CONTINUATION_BASIS.md",
    f"runs/{RUN_DIR.name}/ROUND44_CURRENT_TASK.json",
    f"runs/{RUN_DIR.name}/r44_execute.py",
    f"runs/{RUN_DIR.name}/r44_runtime.py",
    f"runs/{RUN_DIR.name}/r44_resume_bridge.py",
    f"runs/{RUN_DIR.name}/r44_resume_gate.py",
    f"runs/{RUN_DIR.name}/test_r44_resume_gate.py",
    f"runs/{RUN_DIR.name}/RUNNER_FIX.md",
    f"runs/{RUN_DIR.name}/ROUND44_RESUME_STATE_WHITELIST_R1.json",
}
R2_ALLOWED_TASK_DELTA = {
    f"runs/{RUN_DIR.name}/ROUND44_CURRENT_CONTINUATION_BASIS.md",
    f"runs/{RUN_DIR.name}/ROUND44_CURRENT_TASK.json",
    f"runs/{RUN_DIR.name}/r44_execute.py",
    f"runs/{RUN_DIR.name}/r44_resume_bridge.py",
    f"runs/{RUN_DIR.name}/r44_resume_gate.py",
    f"runs/{RUN_DIR.name}/test_r44_resume_gate.py",
    f"runs/{RUN_DIR.name}/RUNNER_FIX.md",
}
R3_ALLOWED_TASK_DELTA = {
    f"runs/{RUN_DIR.name}/ROUND44_CURRENT_TASK.json",
    f"runs/{RUN_DIR.name}/RUNNER_FIX.md",
    f"runs/{RUN_DIR.name}/r44_execute.py",
    f"runs/{RUN_DIR.name}/r44_resume_bridge.py",
    f"runs/{RUN_DIR.name}/r44_resume_gate.py",
    f"runs/{RUN_DIR.name}/test_r44_resume_gate.py",
}
EXPECTED_COMPARATOR_REJECTIONS = [
    "parameter", "dual", "optimizer", "configuration", "artifact_path",
    "transcript", "on_time", "task_set", "cost", "task_reward", "budget_mask",
    "candidate_counter", "decision_counter", "action", "probability", "likelihood",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_exclusive_or_match(path: Path, value: Any) -> None:
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"immutable ROUND44 resume identity conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(RUN_DIR.parents[1]), *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def _repo_root() -> Path:
    return RUN_DIR.parents[1]


def _relative_file_map(paths: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative in paths:
        path = RUN_DIR / Path(relative)
        if not path.is_file():
            raise RuntimeError(f"ROUND44 whitelisted source artifact is missing: {relative}")
        result[relative.replace("\\", "/")] = {
            "sha256": _sha256(path), "size_bytes": path.stat().st_size,
        }
    return result


def _load_original_receipt() -> dict[str, Any]:
    path = RUN_DIR / "runtime" / "ROUND44_PRELAUNCH_IDENTITY.json"
    receipt = _read(path)
    if (receipt.get("status") != "passed"
            or receipt.get("head") != BASE_EXECUTION_HEAD
            or receipt.get("branch") != BRANCH):
        raise RuntimeError("original ROUND44 prelaunch identity no longer matches source head")
    return receipt


def _verify_snapshot_manifest() -> tuple[dict[str, Any], str]:
    if not SNAPSHOT_MANIFEST_PATH.is_file():
        raise RuntimeError("pre-resume byte snapshot manifest is missing")
    manifest = _read(SNAPSHOT_MANIFEST_PATH)
    if manifest.get("kind") != "round44-pre-resume-byte-snapshot/1":
        raise RuntimeError("pre-resume snapshot manifest has an unknown version")
    for entry in manifest.get("files", []):
        relative = str(entry["path"])
        snapshot = SNAPSHOT_ROOT / Path(relative)
        if (not snapshot.is_file()
                or _sha256(snapshot) != entry.get("sha256")
                or snapshot.stat().st_size != int(entry.get("size_bytes", -1))):
            raise RuntimeError(f"pre-resume snapshot byte identity mismatch: {relative}")
    return manifest, _sha256(SNAPSHOT_MANIFEST_PATH)


def build_state_whitelist() -> dict[str, Any]:
    """Seal only the already-committed seed11/batch0 state; never derives a new state."""
    repo_head = _git("rev-parse", "HEAD")
    if repo_head != TASK_START_HEAD or _git("branch", "--show-current") != BRANCH:
        raise RuntimeError("ROUND44 state whitelist must be built at the verified stop head")
    frozen = _read(RUN_DIR / "ROUND44_FROZEN_CONFIG.json")
    status = _read(RUN_DIR / "ROUND44_STATUS.json")
    ledger = _read(RUN_DIR / "runtime" / "ROUND44_EXECUTION_LEDGER.json")
    progress = _read(RUN_DIR / "runtime" / "ROUND44_EXECUTION_PROGRESS.json")
    if (frozen.get("status") != "prelaunch_frozen"
            or frozen["round42"].get("training_version") != TRAINING_VERSION
            or float(frozen["round44"].get("eta_lambda", -1)) != 0.5):
        raise RuntimeError("ROUND44 frozen science identity differs from the authorized eta=0.5 run")
    if (status.get("status") != "stopped_after_seed11_batch0_runner_failure"
            or status.get("failure") != "NameError: name 'run_module' is not defined in _compare_first_batch"):
        raise RuntimeError("ROUND44 immutable stop status differs from the recorded comparator failure")
    if (ledger.get("counts", {}).get("normal_used") != 4
            or ledger.get("counts", {}).get("reserve_used") != 0
            or ledger.get("counts", {}).get("completed_logical_jobs") != 4
            or ledger.get("counts", {}).get("optimizer_step_calls") != 12
            or ledger.get("counts", {}).get("ppo_update_calls") != 1
            or ledger.get("counts", {}).get("checkpoint_write_calls") != 4
            or len(ledger.get("claims", [])) != 4):
        raise RuntimeError("ROUND44 ledger does not describe exactly the four completed seed11 episodes")
    if (progress.get("status") != "running"
            or progress.get("prelaunch_head") != BASE_EXECUTION_HEAD
            or progress.get("training") != {}
            or int(progress.get("evaluations_completed", -1)) != 0):
        raise RuntimeError("original ROUND44 progress changed before exact recovery")
    expected_jobs = [f"new_training_s11_e{episode}" for episode in range(4)]
    ledger_jobs = [claim.get("logical_job") for claim in ledger["claims"]]
    if ledger_jobs != expected_jobs or any(claim.get("status") != "complete"
                                           for claim in ledger["claims"]):
        raise RuntimeError("ROUND44 claim list is not exactly the four successful seed11 episodes")

    commit_path = RUN_DIR / "checkpoints" / "r40_batch_commit_s11_b0.json"
    record = _read(RUN_DIR / "checkpoints" / "r40_batch_s11_b0_attempt01.json")
    commit = _read(commit_path)
    if (commit.get("batch_record") != "r40_batch_s11_b0_attempt01.json"
            or commit.get("batch_record_sha256") != _sha256(
                RUN_DIR / "checkpoints" / str(commit["batch_record"]))
            or record.get("training_seed") != 11 or record.get("batch") != 0
            or record.get("episodes") != [0, 1, 2, 3]
            or record.get("scenarios") != [1300027, 1300014, 1300028, 1300044]
            or float(record.get("lambda_at_rollout", -1)) != 1.0
            or float(record.get("dual", {}).get("lambda_before", -1)) != 1.0
            or float(record.get("dual", {}).get("g_hat", -1)) != 0.25
            or float(record.get("dual", {}).get("eta", -1)) != 0.5
            or float(record.get("dual", {}).get("lambda_next", -1)) != 1.125
            or record.get("training_version") != TRAINING_VERSION):
        raise RuntimeError("ROUND44 seed11 batch-zero record/commit is not the authorized state")

    immutable_paths = [
        "ROUND44_STATUS.json", "ROUND44_STOP_REPORT.md", "ROUND44_STOP_STATE.json",
        "ROUND44_STOP_ARTIFACT_INVENTORY.json",
        "ROUND44_EXECUTION_FAILURE_CAPTURE_SEED11_B0.txt",
        "ROUND44_FROZEN_CONFIG.json", "runtime/ROUND44_PRELAUNCH_IDENTITY.json",
        "checkpoints/r40_batch_commit_s11_b0.json",
        "checkpoints/r40_batch_s11_b0_attempt01.json",
        "checkpoints/v2_train_s11_before_b0.pt",
        "checkpoints/v2_train_s11_after_b0.pt", "checkpoints/v2_train_s11_ep0.pt",
        "training_reports/ROUND44_SEED11_FIRST4.json",
    ]
    for index in range(1, 5):
        immutable_paths.extend((
            f"launch_slots/normal/{index:04d}.json",
            f"launch_claims/new_training_s11_e{index - 1}/attempt_0001.json",
        ))
    for artifact in record["episode_artifacts"]:
        immutable_paths.extend((
            str(Path(artifact["row_path"]).resolve().relative_to(RUN_DIR.resolve())).replace("\\", "/"),
            str(Path(artifact["tape_path"]).resolve().relative_to(RUN_DIR.resolve())).replace("\\", "/"),
        ))
    immutable_files = _relative_file_map(immutable_paths)
    for artifact in record["episode_artifacts"]:
        for name in ("row_path", "tape_path"):
            relative = str(Path(artifact[name]).resolve().relative_to(RUN_DIR.resolve())).replace("\\", "/")
            if immutable_files[relative]["sha256"] != artifact[name.replace("_path", "_sha256")]:
                raise RuntimeError(f"ROUND44 batch-zero artifact hash differs: {relative}")

    snapshot_manifest, snapshot_manifest_sha = _verify_snapshot_manifest()
    mutable_snapshot_map = {
        entry["path"]: {"sha256": entry["sha256"], "size_bytes": entry["size_bytes"]}
        for entry in snapshot_manifest["files"]
        if entry["path"] in MUTABLE_PRE_RESUME_PATHS
    }
    if set(mutable_snapshot_map) != set(MUTABLE_PRE_RESUME_PATHS):
        raise RuntimeError("pre-resume snapshots do not cover all mutable continuation state")

    original_receipt = _load_original_receipt()
    raw_counts = ledger["counts"]
    whitelist = {
        "kind": STATE_BRIDGE_VERSION,
        "status": "sealed_for_ep4_only",
        "base_execution_head": BASE_EXECUTION_HEAD,
        "stop_head_before_mechanical_fix": TASK_START_HEAD,
        "branch": BRANCH,
        "training_version": TRAINING_VERSION,
        "runner_version": status.get("runner_version"),
        "permitted_resume": {
            "training_seed": 11, "batch": 0, "episodes_done": 4,
            "updates": 1, "optimizer_steps": 12, "lambda": 1.125,
            "eta_lambda": 0.5, "parameter_sha256": record[
                "controller_parameters_after"]["parameter_sha256"],
            "postupdate_checkpoint_sha256": commit["postupdate_checkpoint_sha256"],
        },
        "dual_history": {
            "lambda_before": 1.0, "g_hat": 0.25, "eta": 0.5,
            "lambda_next": 1.125, "delta": 0.0,
        },
        "accounting_before_resume": {
            "normal_used": int(raw_counts["normal_used"]),
            "reserve_used": int(raw_counts["reserve_used"]),
            "completed_steps": int(raw_counts["completed_steps"]),
            "ppo_update_calls": int(raw_counts["ppo_update_calls"]),
            "optimizer_step_calls": int(raw_counts["optimizer_step_calls"]),
            "checkpoint_write_calls": int(raw_counts["checkpoint_write_calls"]),
            "hard_cap": int(raw_counts["hard_cap"]),
        },
        "immutable_artifacts": immutable_files,
        "preserved_mutable_state_snapshots": {
            "manifest_path": "runtime/ROUND44_PRE_RESUME_R1_SNAPSHOT_MANIFEST.json",
            "manifest_sha256": snapshot_manifest_sha,
            "snapshot_files": mutable_snapshot_map,
        },
        "original_prelaunch_receipt_sha256": _sha256(
            RUN_DIR / "runtime" / "ROUND44_PRELAUNCH_IDENTITY.json"),
        "original_frozen_config_sha256": _sha256(RUN_DIR / "ROUND44_FROZEN_CONFIG.json"),
        "original_prelaunch_source_code_sha256": original_receipt["source_code_sha256"],
        "not_permitted": [
            "ROUND43 or other seed checkpoints as resume input",
            "any checkpoint other than the exact seed11 ep4/update1 state",
            "replay of seed11 batch zero",
            "fresh seed11 training state",
            "change to scientific training version or eta_lambda",
        ],
    }
    _write_exclusive_or_match(WHITELIST_PATH, whitelist)
    return whitelist


def verify_state_whitelist(*, check_live_mutables: bool) -> dict[str, Any]:
    if not WHITELIST_PATH.is_file():
        raise RuntimeError("versioned ROUND44 exact-state whitelist is missing")
    whitelist = _read(WHITELIST_PATH)
    if (whitelist.get("kind") != STATE_BRIDGE_VERSION
            or whitelist.get("status") != "sealed_for_ep4_only"
            or whitelist.get("base_execution_head") != BASE_EXECUTION_HEAD
            or whitelist.get("stop_head_before_mechanical_fix") != TASK_START_HEAD
            or whitelist.get("branch") != BRANCH
            or whitelist.get("training_version") != TRAINING_VERSION):
        raise RuntimeError("ROUND44 resume whitelist does not match the authorized version/state")
    snapshot_manifest, actual_manifest_sha = _verify_snapshot_manifest()
    snapshot_identity = whitelist["preserved_mutable_state_snapshots"]
    if actual_manifest_sha != snapshot_identity.get("manifest_sha256"):
        raise RuntimeError("ROUND44 pre-resume snapshot manifest differs from the sealed whitelist")
    manifest_by_path = {entry["path"]: entry for entry in snapshot_manifest["files"]}
    for relative in MUTABLE_PRE_RESUME_PATHS:
        if relative not in manifest_by_path:
            raise RuntimeError(f"ROUND44 pre-resume snapshot is missing {relative}")
        entry = manifest_by_path[relative]
        if check_live_mutables:
            live = RUN_DIR / Path(relative)
            if (not live.is_file() or _sha256(live) != entry["sha256"]
                    or live.stat().st_size != int(entry["size_bytes"])):
                raise RuntimeError(f"live pre-resume state changed before continuation: {relative}")
    if whitelist.get("preserved_mutable_state_snapshots", {}).get("snapshot_files") != {
            relative: {"sha256": manifest_by_path[relative]["sha256"],
                       "size_bytes": manifest_by_path[relative]["size_bytes"]}
            for relative in MUTABLE_PRE_RESUME_PATHS}:
        raise RuntimeError("ROUND44 snapshot file inventory differs from whitelist")
    for relative, entry in whitelist.get("immutable_artifacts", {}).items():
        path = RUN_DIR / Path(relative)
        if (not path.is_file() or _sha256(path) != entry["sha256"]
                or path.stat().st_size != int(entry["size_bytes"])):
            raise RuntimeError(f"ROUND44 immutable ep4 artifact differs from whitelist: {relative}")
    if (_sha256(RUN_DIR / "runtime" / "ROUND44_PRELAUNCH_IDENTITY.json")
            != whitelist.get("original_prelaunch_receipt_sha256")
            or _sha256(RUN_DIR / "ROUND44_FROZEN_CONFIG.json")
            != whitelist.get("original_frozen_config_sha256")):
        raise RuntimeError("ROUND44 original scientific freeze or prelaunch receipt changed")
    return whitelist


def _verify_r1_identity_unchanged() -> dict[str, Any]:
    if not R1_IDENTITY_PATH.is_file() or _sha256(R1_IDENTITY_PATH) != R1_IDENTITY_SHA256:
        raise RuntimeError("ROUND44 R1 resume identity is missing or was modified")
    identity = _read(R1_IDENTITY_PATH)
    if (identity.get("status") != "passed"
            or identity.get("head") != R1_EXECUTION_HEAD
            or identity.get("branch") != BRANCH
            or identity.get("mechanical_version") != "round44-resume-comparator-and-restore/1"
            or identity.get("state_bridge_version") != STATE_BRIDGE_VERSION):
        raise RuntimeError("ROUND44 R1 identity is not the preserved 43399eb parent receipt")
    for relative, expected in identity.get("source_code_sha256", {}).items():
        object_path = f"runs/{RUN_DIR.name}/{relative}"
        result = subprocess.run(
            ["git", "-C", str(_repo_root()), "show", f"{R1_EXECUTION_HEAD}:{object_path}"],
            check=True, capture_output=True,
        )
        if hashlib.sha256(result.stdout).hexdigest() != expected:
            raise RuntimeError(f"ROUND44 R1 committed source identity differs: {relative}")
    return identity


def _verify_r2_source_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    if not OLD_PROJECTION_PATH.is_file() or not SOURCE_CONFIG_RECONCILIATION_PATH.is_file():
        raise RuntimeError("ROUND44 R2 four-episode source evidence is missing")
    old_projection = _read(OLD_PROJECTION_PATH)
    source_config = _read(SOURCE_CONFIG_RECONCILIATION_PATH)
    if (old_projection.get("status") != "captured_before_comparator_revision"
            or len(old_projection.get("episodes", [])) != 4
            or any(len(item.get("whole_row_diff", [])) != 11
                   or len(item.get("old_science_projection_diff", [])) != 3
                   or item.get("tape_bytes_equal") is not True
                   for item in old_projection["episodes"])):
        raise RuntimeError("ROUND44 old projection evidence does not cover the real four episodes")
    modules = source_config.get("runtime_module_sources", {})
    if (source_config.get("status") != "passed_source_and_config_only"
            or len(source_config.get("episodes", [])) != 4
            or not modules or any(item.get("matches_frozen_source") is not True
                                  for item in modules.values())
            or any(item.get("configuration_differing_fields")
                   != ["comm_transcript", "output"]
                   or item.get("normalized_hash_equal") is not True
                   for item in source_config["episodes"])):
        raise RuntimeError("ROUND44 source/config evidence is incomplete or found extra drift")
    return old_projection, source_config


def _r2_preserved_originals() -> dict[str, str]:
    result = {relative: _sha256(RUN_DIR / relative) for relative in STOP_FILES}
    result["runtime/ROUND44_OLD_COMPARATOR_REPRODUCTION.json"] = _sha256(
        R1_FAILURE_REPRODUCTION_PATH)
    result["runtime/ROUND44_RESUME_EXECUTION_IDENTITY_R1.json"] = _sha256(R1_IDENTITY_PATH)
    return result


def _assert_r1_gate_outputs_absent() -> None:
    for path in (R1_GATE_PATH, R1_COMPARATOR_TEST_PATH, R1_READY_PATH):
        if path.exists():
            raise RuntimeError(f"ROUND44 R1 failed-gate namespace must remain untouched: {path}")


def _verify_r2_identity_unchanged() -> dict[str, Any]:
    if not R2_IDENTITY_PATH.is_file() or _sha256(R2_IDENTITY_PATH) != R2_IDENTITY_SHA256:
        raise RuntimeError("ROUND44 R2 execution identity is missing or was modified")
    identity = _read(R2_IDENTITY_PATH)
    if (identity.get("status") != "passed"
            or identity.get("kind") != "round44-resume-execution-identity/2"
            or identity.get("head") != R2_EXECUTION_HEAD
            or identity.get("branch") != BRANCH
            or identity.get("mechanical_version") != "round44-resume-comparator-and-restore/2"
            or identity.get("state_bridge_version") != STATE_BRIDGE_VERSION):
        raise RuntimeError("ROUND44 R2 is not the preserved 453924c execution identity")
    for relative, expected in identity.get("source_code_sha256", {}).items():
        object_path = f"runs/{RUN_DIR.name}/{relative}"
        result = subprocess.run(
            ["git", "-C", str(_repo_root()), "show", f"{R2_EXECUTION_HEAD}:{object_path}"],
            check=True, capture_output=True,
        )
        if hashlib.sha256(result.stdout).hexdigest() != expected:
            raise RuntimeError(f"ROUND44 R2 committed source identity differs: {relative}")
    return identity


def _verify_r2_gate_attempt_failure(*, require_shared_gate_absent: bool = False) -> dict[str, Any]:
    if not R2_FAILURE_PATH.is_file() or _sha256(R2_FAILURE_PATH) != R2_FAILURE_SHA256:
        raise RuntimeError("ROUND44 R2 gate-attempt failure evidence is missing or changed")
    failure = _read(R2_FAILURE_PATH)
    if (failure.get("kind") != "round44-r2-gate-attempt-failure/1"
            or failure.get("status") != "failed_before_exact_restore_receipt"
            or failure.get("execution_head") != R2_EXECUTION_HEAD
            or "training_version mismatch" not in failure.get("actual_exception_message", "")
            or failure.get("environment_starts") != 0
            or failure.get("run_calls") != 0
            or failure.get("new_claims") != 0
            or failure.get("optimizer_steps") != 0
            or failure.get("checkpoint_writes") != 0):
        raise RuntimeError("ROUND44 R2 gate failure evidence does not match the isolated assertion defect")
    shared_gate_path = RUN_DIR / "ROUND44_FIRST_BATCH_GATE_SEED11.json"
    if (R2_GATE_PATH.exists() or R2_READY_PATH.exists()
            or (require_shared_gate_absent and shared_gate_path.exists())):
        raise RuntimeError("ROUND44 R2 partial pass receipt exists and cannot be silently replaced")
    regression = _read(R2_COMPARATOR_TEST_PATH)
    comparison_path = RUN_DIR / "ROUND44_FIRST_BATCH_COMPARISON_SEED11.json"
    comparison = _read(comparison_path)
    episodes = comparison.get("episodes", [])
    if (regression.get("kind") != "round44-comparator-regression/2"
            or regression.get("status") != "passed"
            or regression.get("expected_rejections") != EXPECTED_COMPARATOR_REJECTIONS
            or _sha256(R2_COMPARATOR_TEST_PATH)
            != failure.get("passed_before_failure", {}).get("r2_comparator_regression_sha256")
            or comparison.get("kind") != "round44-first-batch-vs-r43-comparison/2"
            or comparison.get("status") != "passed"
            or len(episodes) != 4
            or any(item.get("metadata_reconciliation", {}).get(
                "transcript_content_equal") is not True for item in episodes)
            or _sha256(comparison_path)
            != failure.get("passed_before_failure", {}).get("first_batch_comparison_sha256")):
        raise RuntimeError("ROUND44 R2 comparator result does not match its preserved failure record")
    return failure


def _r3_preserved_material() -> dict[str, str]:
    result = _r2_preserved_originals()
    result["runtime/ROUND44_RESUME_EXECUTION_IDENTITY_R2.json"] = _sha256(R2_IDENTITY_PATH)
    result["runtime/ROUND44_R2_GATE_ATTEMPT_FAILURE.json"] = _sha256(R2_FAILURE_PATH)
    result["runtime/ROUND44_COMPARATOR_REGRESSION_R2.json"] = _sha256(
        R2_COMPARATOR_TEST_PATH)
    result["ROUND44_FIRST_BATCH_COMPARISON_SEED11.json"] = _sha256(
        RUN_DIR / "ROUND44_FIRST_BATCH_COMPARISON_SEED11.json")
    return result


def seal_resume_execution_identity() -> dict[str, Any]:
    """Seal the R3 rejection-message check while preserving R1, R2, and both failures."""
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != BRANCH or _git("status", "--porcelain"):
        raise RuntimeError("ROUND44 R3 resume source must be committed and the worktree clean")
    if head == R2_EXECUTION_HEAD:
        raise RuntimeError("ROUND44 R3 must descend from, not replace, the R2 mechanical identity")
    for ancestor in (BASE_EXECUTION_HEAD, TASK_START_HEAD, R1_EXECUTION_HEAD, R2_EXECUTION_HEAD):
        subprocess.run(["git", "-C", str(_repo_root()), "merge-base", "--is-ancestor",
                        ancestor, head], check=True, stdout=subprocess.DEVNULL)
    r1_identity = _verify_r1_identity_unchanged()
    r2_identity = _verify_r2_identity_unchanged()
    r2_failure = _verify_r2_gate_attempt_failure(require_shared_gate_absent=True)
    _assert_r1_gate_outputs_absent()
    whitelist = verify_state_whitelist(check_live_mutables=True)
    changed_from_r2 = sorted(path.replace("\\", "/") for path in _git(
        "diff", "--name-only", f"{R2_EXECUTION_HEAD}..{head}").splitlines())
    if not set(changed_from_r2).issubset(R3_ALLOWED_TASK_DELTA):
        raise RuntimeError(f"ROUND44 R3 commit changed a path outside its narrow delta: {changed_from_r2}")
    changed_from_r1 = sorted(path.replace("\\", "/") for path in _git(
        "diff", "--name-only", f"{R1_EXECUTION_HEAD}..{head}").splitlines())
    if not set(changed_from_r1).issubset(R2_ALLOWED_TASK_DELTA):
        raise RuntimeError(f"ROUND44 R1-to-R3 source delta is unauthorized: {changed_from_r1}")
    changed_from_start = sorted(path.replace("\\", "/") for path in _git(
        "diff", "--name-only", f"{TASK_START_HEAD}..{head}").splitlines())
    if not set(changed_from_start).issubset(ALLOWED_TASK_DELTA):
        raise RuntimeError(f"ROUND44 R3 cumulative source delta is unauthorized: {changed_from_start}")

    old_projection, source_config = _verify_r2_source_evidence()
    original = _load_original_receipt()
    original_path = RUN_DIR / "runtime" / "ROUND44_PRELAUNCH_IDENTITY.json"
    source_hashes: dict[str, str] = {}
    for relative, expected in original["source_code_sha256"].items():
        path = RUN_DIR / relative
        if not path.is_file():
            raise RuntimeError(f"ROUND44 frozen code source is missing: {relative}")
        source_hashes[relative] = _sha256(path)
        if relative not in ("r44_execute.py", "r44_runtime.py") \
                and source_hashes[relative] != expected:
            raise RuntimeError(f"ROUND44 unrelated frozen executor code changed: {relative}")
    for relative in SOURCE_FILES:
        path = RUN_DIR / relative
        if not path.is_file():
            raise RuntimeError(f"ROUND44 R3 source artifact is missing: {relative}")
        source_hashes[relative] = _sha256(path)

    identity = {
        "kind": "round44-resume-execution-identity/3",
        "status": "passed",
        "mechanical_version": MECHANICAL_VERSION,
        "state_bridge_version": STATE_BRIDGE_VERSION,
        "head": head,
        "branch": branch,
        "authorized_base_execution_head": BASE_EXECUTION_HEAD,
        "r1_parent_head": R1_EXECUTION_HEAD,
        "r1_identity_path": R1_IDENTITY_PATH.name,
        "r1_identity_sha256": _sha256(R1_IDENTITY_PATH),
        "r2_parent_head": R2_EXECUTION_HEAD,
        "r2_identity_path": R2_IDENTITY_PATH.name,
        "r2_identity_sha256": _sha256(R2_IDENTITY_PATH),
        "r2_failure_path": R2_FAILURE_PATH.name,
        "r2_failure_sha256": _sha256(R2_FAILURE_PATH),
        "r2_comparator_regression_sha256": _sha256(R2_COMPARATOR_TEST_PATH),
        "r2_first_batch_comparison_sha256": _sha256(
            RUN_DIR / "ROUND44_FIRST_BATCH_COMPARISON_SEED11.json"),
        "r2_shared_first_batch_gate_absent_at_seal": not (
            RUN_DIR / "ROUND44_FIRST_BATCH_GATE_SEED11.json").exists(),
        "r2_gate_attempt_interpretation": r2_failure["interpretation"],
        "task_start_head": TASK_START_HEAD,
        "r3_allowed_task_delta_paths": sorted(R3_ALLOWED_TASK_DELTA),
        "observed_r3_task_delta_paths": changed_from_r2,
        "observed_r1_through_r3_paths": changed_from_r1,
        "cumulative_allowed_task_delta_paths": sorted(ALLOWED_TASK_DELTA),
        "observed_cumulative_task_delta_paths": changed_from_start,
        "training_version": TRAINING_VERSION,
        "runner_version": original["runner_version"],
        "eta_lambda": 0.5,
        "source_code_sha256": source_hashes,
        "state_whitelist_path": WHITELIST_PATH.name,
        "state_whitelist_sha256": _sha256(WHITELIST_PATH),
        "snapshot_manifest_sha256": whitelist[
            "preserved_mutable_state_snapshots"]["manifest_sha256"],
        "original_prelaunch_receipt_sha256": _sha256(original_path),
        "original_prelaunch_head": original["head"],
        "original_frozen_config_sha256": _sha256(RUN_DIR / "ROUND44_FROZEN_CONFIG.json"),
        "seed11_resume": whitelist["permitted_resume"],
        "old_projection_failure_evidence": {
            "path": str(OLD_PROJECTION_PATH.relative_to(RUN_DIR)).replace("\\", "/"),
            "sha256": _sha256(OLD_PROJECTION_PATH),
            "episodes": len(old_projection["episodes"]),
        },
        "source_config_reconciliation": {
            "path": str(SOURCE_CONFIG_RECONCILIATION_PATH.relative_to(RUN_DIR)).replace("\\", "/"),
            "sha256": _sha256(SOURCE_CONFIG_RECONCILIATION_PATH),
            "episodes": len(source_config["episodes"]),
        },
        "preserved_r1_r2_failure_and_stop_hashes": _r3_preserved_material(),
        "r1_gate_outputs_remain_absent": True,
        "r2_gate_failure_remains_unmodified": True,
        "no_scientific_configuration_change": True,
    }
    _write_exclusive_or_match(IDENTITY_PATH, identity)
    return identity


def verify_resume_execution_identity(frozen: dict[str, Any], current_head: str) -> dict[str, Any]:
    if not IDENTITY_PATH.is_file():
        raise RuntimeError("ROUND44 R3 versioned resume execution identity is missing")
    identity = _read(IDENTITY_PATH)
    branch = _git("branch", "--show-current")
    if (identity.get("status") != "passed"
            or identity.get("kind") != "round44-resume-execution-identity/3"
            or identity.get("mechanical_version") != MECHANICAL_VERSION
            or identity.get("state_bridge_version") != STATE_BRIDGE_VERSION
            or identity.get("head") != current_head
            or current_head != _git("rev-parse", "HEAD")
            or branch != BRANCH or identity.get("branch") != BRANCH
            or _git("status", "--porcelain")):
        raise RuntimeError("ROUND44 current branch/HEAD/worktree differs from R3 resume identity")
    for ancestor in (BASE_EXECUTION_HEAD, TASK_START_HEAD, R1_EXECUTION_HEAD, R2_EXECUTION_HEAD):
        subprocess.run(["git", "-C", str(_repo_root()), "merge-base", "--is-ancestor",
                        ancestor, current_head], check=True, stdout=subprocess.DEVNULL)
    r1_identity = _verify_r1_identity_unchanged()
    if (identity.get("r1_identity_sha256") != _sha256(R1_IDENTITY_PATH)
            or r1_identity.get("head") != identity.get("r1_parent_head")):
        raise RuntimeError("ROUND44 R1 identity changed after R2/R3 were sealed")
    r2_identity = _verify_r2_identity_unchanged()
    r2_failure = _verify_r2_gate_attempt_failure()
    if (identity.get("r2_identity_sha256") != _sha256(R2_IDENTITY_PATH)
            or r2_identity.get("head") != identity.get("r2_parent_head")
            or identity.get("r2_failure_sha256") != _sha256(R2_FAILURE_PATH)
            or identity.get("r2_failure_sha256") != R2_FAILURE_SHA256
            or identity.get("r2_comparator_regression_sha256")
            != _sha256(R2_COMPARATOR_TEST_PATH)
            or identity.get("r2_first_batch_comparison_sha256")
            != _sha256(RUN_DIR / "ROUND44_FIRST_BATCH_COMPARISON_SEED11.json")
            or identity.get("r2_shared_first_batch_gate_absent_at_seal") is not True
            or r2_failure.get("status") != "failed_before_exact_restore_receipt"):
        raise RuntimeError("ROUND44 R2 failure evidence changed after R3 was sealed")
    _assert_r1_gate_outputs_absent()
    changed_from_r2 = sorted(path.replace("\\", "/") for path in _git(
        "diff", "--name-only", f"{R2_EXECUTION_HEAD}..{current_head}").splitlines())
    changed_from_r1 = sorted(path.replace("\\", "/") for path in _git(
        "diff", "--name-only", f"{R1_EXECUTION_HEAD}..{current_head}").splitlines())
    changed_from_start = sorted(path.replace("\\", "/") for path in _git(
        "diff", "--name-only", f"{TASK_START_HEAD}..{current_head}").splitlines())
    if (changed_from_r2 != identity.get("observed_r3_task_delta_paths")
            or not set(changed_from_r2).issubset(R3_ALLOWED_TASK_DELTA)
            or changed_from_r1 != identity.get("observed_r1_through_r3_paths")
            or not set(changed_from_r1).issubset(R2_ALLOWED_TASK_DELTA)
            or changed_from_start != identity.get("observed_cumulative_task_delta_paths")
            or not set(changed_from_start).issubset(ALLOWED_TASK_DELTA)):
        raise RuntimeError("ROUND44 R3 source delta differs from its exact committed whitelist")
    if (identity.get("training_version") != TRAINING_VERSION
            or frozen.get("round42", {}).get("training_version") != TRAINING_VERSION
            or identity.get("eta_lambda") != 0.5
            or frozen.get("round42", {}).get("objective", {}).get("eta_lambda") != 0.5):
        raise RuntimeError("ROUND44 R3 identity changed the scientific training version/eta")
    whitelist = verify_state_whitelist(check_live_mutables=False)
    original_path = RUN_DIR / "runtime" / "ROUND44_PRELAUNCH_IDENTITY.json"
    if (_sha256(WHITELIST_PATH) != identity.get("state_whitelist_sha256")
            or _sha256(original_path) != identity.get("original_prelaunch_receipt_sha256")
            or _sha256(RUN_DIR / "ROUND44_FROZEN_CONFIG.json")
            != identity.get("original_frozen_config_sha256")
            or whitelist["preserved_mutable_state_snapshots"]["manifest_sha256"]
            != identity.get("snapshot_manifest_sha256")):
        raise RuntimeError("ROUND44 R3 identity no longer matches the original freeze/whitelist")
    original = _load_original_receipt()
    for relative, expected in identity.get("source_code_sha256", {}).items():
        path = RUN_DIR / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"ROUND44 R3 resume source code hash mismatch: {relative}")
        if relative in original["source_code_sha256"] \
                and relative not in ("r44_execute.py", "r44_runtime.py") \
                and expected != original["source_code_sha256"][relative]:
            raise RuntimeError(f"ROUND44 original executor source drifted: {relative}")
    old_projection, source_config = _verify_r2_source_evidence()
    if (_sha256(OLD_PROJECTION_PATH) != identity["old_projection_failure_evidence"]["sha256"]
            or len(old_projection["episodes"]) != identity["old_projection_failure_evidence"]["episodes"]
            or _sha256(SOURCE_CONFIG_RECONCILIATION_PATH)
            != identity["source_config_reconciliation"]["sha256"]
            or len(source_config["episodes"])
            != identity["source_config_reconciliation"]["episodes"]
            or _r3_preserved_material() != identity.get("preserved_r1_r2_failure_and_stop_hashes")):
        raise RuntimeError("ROUND44 R3 source evidence or preserved R1/R2 failure material changed")
    return identity


def verify_resume_ready(frozen: dict[str, Any], current_head: str) -> dict[str, Any]:
    identity = verify_resume_execution_identity(frozen, current_head)
    if not READY_PATH.is_file():
        raise RuntimeError("ROUND44 R3 exact-resume/comparator gates are not ready for a claim")
    ready = _read(READY_PATH)
    if (ready.get("kind") != "round44-resume-ready/3"
            or ready.get("status") != "passed"
            or ready.get("execution_head") != current_head
            or ready.get("mechanical_version") != MECHANICAL_VERSION
            or ready.get("state_bridge_version") != STATE_BRIDGE_VERSION
            or ready.get("execution_identity_sha256") != _sha256(IDENTITY_PATH)
            or ready.get("state_whitelist_sha256") != _sha256(WHITELIST_PATH)
            or ready.get("r1_identity_sha256") != _sha256(R1_IDENTITY_PATH)
            or ready.get("r2_identity_sha256") != _sha256(R2_IDENTITY_PATH)
            or ready.get("r2_failure_sha256") != _sha256(R2_FAILURE_PATH)
            or ready.get("old_projection_evidence_sha256") != _sha256(OLD_PROJECTION_PATH)
            or ready.get("source_config_reconciliation_sha256")
            != _sha256(SOURCE_CONFIG_RECONCILIATION_PATH)):
        raise RuntimeError("ROUND44 R3 resume-ready receipt does not match its source identity")
    for key, path in (("exact_resume_gate_sha256", GATE_PATH),
                      ("comparator_regression_sha256", COMPARATOR_TEST_PATH),
                      ("first_batch_comparison_sha256",
                       RUN_DIR / "ROUND44_FIRST_BATCH_COMPARISON_SEED11.json"),
                      ("first_batch_gate_sha256", RUN_DIR / "ROUND44_FIRST_BATCH_GATE_SEED11.json")):
        if not path.is_file() or ready.get(key) != _sha256(path):
            raise RuntimeError(f"ROUND44 R3 ready receipt lacks its sealed gate artifact: {key}")
    gate = _read(GATE_PATH)
    regression = _read(COMPARATOR_TEST_PATH)
    comparison = _read(RUN_DIR / "ROUND44_FIRST_BATCH_COMPARISON_SEED11.json")
    first_batch = _read(RUN_DIR / "ROUND44_FIRST_BATCH_GATE_SEED11.json")
    episode_gates = comparison.get("episodes", [])
    if (gate.get("kind") != "round44-exact-resume-gate/3"
            or gate.get("status") != "passed" or gate.get("environment_starts") != 0
            or gate.get("optimizer_steps") != 0 or gate.get("checkpoint_writes") != 0
            or gate.get("training_seed") != 11 or gate.get("episodes_done") != 4
            or gate.get("updates") != 1 or gate.get("optimizer_steps_saved") != 12
            or gate.get("lambda") != 1.125
            or regression.get("kind") != "round44-comparator-regression/2"
            or regression.get("status") != "passed"
            or regression.get("expected_rejections") != EXPECTED_COMPARATOR_REJECTIONS
            or regression.get("accepted_wallclock_timing_perturbation", {}).get("status") != "passed"
            or len(regression.get("accepted_real_registered_row_labels_and_output_paths", [])) != 4
            or comparison.get("kind") != "round44-first-batch-vs-r43-comparison/2"
            or comparison.get("projection_version") != "round44-row-projection/2"
            or comparison.get("status") != "passed" or len(episode_gates) != 4
            or any(item.get("science_projection_equal") is not True
                   or item.get("tape_equal") is not True
                   or item.get("metadata_reconciliation", {}).get(
                       "transcript_content_equal") is not True
                   or item.get("metadata_reconciliation", {}).get(
                       "r43_config_hash_matches_row_and_signature") is not True
                   or item.get("metadata_reconciliation", {}).get(
                       "r44_config_hash_matches_row_and_signature") is not True
                   or item.get("metadata_reconciliation", {}).get(
                       "path_normalized_config_equal") is not True
                   for item in episode_gates)
            or first_batch.get("status") != "passed"
            or first_batch.get("training_seed") != 11
            or first_batch.get("r43_batch0_comparison", {}).get("projection_version")
            != "round44-row-projection/2"):
        raise RuntimeError("ROUND44 R3 resume-ready gate contents are incomplete or failed")
    return identity


def verify_pre_resume_state(frozen: dict[str, Any], current_head: str) -> dict[str, Any]:
    """Last exact live-state check before the runner mutates cumulative ledgers/progress."""
    verify_resume_ready(frozen, current_head)
    return verify_state_whitelist(check_live_mutables=True)


__all__ = [
    "BASE_EXECUTION_HEAD", "BRANCH", "COMPARATOR_TEST_PATH",
    "EXPECTED_COMPARATOR_REJECTIONS", "GATE_PATH", "IDENTITY_PATH",
    "MECHANICAL_VERSION", "OLD_PROJECTION_PATH", "R1_EXECUTION_HEAD",
    "R1_IDENTITY_PATH", "READY_PATH", "RESUME_PROGRESS_PATH",
    "RESUME_PROGRESS_KIND", "R2_EXECUTION_HEAD", "R2_FAILURE_PATH", "R2_IDENTITY_PATH",
    "R3_ALLOWED_TASK_DELTA", "SOURCE_CONFIG_RECONCILIATION_PATH", "STATE_BRIDGE_VERSION",
    "TASK_START_HEAD", "WHITELIST_PATH", "build_state_whitelist",
    "seal_resume_execution_identity", "verify_pre_resume_state",
    "verify_resume_execution_identity", "verify_resume_ready", "verify_state_whitelist",
]
