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
BRANCH = "exp/comm-dual-response-ablation"
STATE_BRIDGE_VERSION = "round44-exact-resume-state-bridge/1"
MECHANICAL_VERSION = "round44-resume-comparator-and-restore/1"
WHITELIST_PATH = RUN_DIR / "ROUND44_RESUME_STATE_WHITELIST_R1.json"
SNAPSHOT_MANIFEST_PATH = RUN_DIR / "runtime" / "ROUND44_PRE_RESUME_R1_SNAPSHOT_MANIFEST.json"
IDENTITY_PATH = RUN_DIR / "runtime" / "ROUND44_RESUME_EXECUTION_IDENTITY_R1.json"
RESUME_PROGRESS_PATH = RUN_DIR / "runtime" / "ROUND44_EXECUTION_PROGRESS_RESUME_R1.json"
GATE_PATH = RUN_DIR / "runtime" / "ROUND44_EXACT_RESUME_GATE_R1.json"
COMPARATOR_TEST_PATH = RUN_DIR / "runtime" / "ROUND44_COMPARATOR_REGRESSION_R1.json"
READY_PATH = RUN_DIR / "runtime" / "ROUND44_RESUME_READY_R1.json"
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


def seal_resume_execution_identity() -> dict[str, Any]:
    """Bind the committed mechanical repair to one HEAD and one exact ep4 state."""
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != BRANCH or _git("status", "--porcelain"):
        raise RuntimeError("ROUND44 resume source must be committed and the worktree clean")
    for ancestor in (BASE_EXECUTION_HEAD, TASK_START_HEAD):
        subprocess.run(["git", "-C", str(_repo_root()), "merge-base", "--is-ancestor",
                        ancestor, head], check=True, stdout=subprocess.DEVNULL)
    whitelist = verify_state_whitelist(check_live_mutables=True)
    allowed_delta = sorted(ALLOWED_TASK_DELTA)
    changed = sorted(path.replace("\\", "/") for path in _git(
        "diff", "--name-only", f"{TASK_START_HEAD}..{head}").splitlines())
    if not set(changed).issubset(ALLOWED_TASK_DELTA):
        raise RuntimeError(f"ROUND44 mechanical resume commit changed an unauthorized path: {changed}")

    original = _load_original_receipt()
    original_path = RUN_DIR / "runtime" / "ROUND44_PRELAUNCH_IDENTITY.json"
    source_hashes = {}
    for relative in original["source_code_sha256"]:
        path = RUN_DIR / relative
        if not path.is_file():
            raise RuntimeError(f"ROUND44 frozen code source is missing: {relative}")
        source_hashes[relative] = _sha256(path)
        if relative not in ("r44_execute.py", "r44_runtime.py") \
                and source_hashes[relative] != original["source_code_sha256"][relative]:
            raise RuntimeError(f"ROUND44 unrelated frozen executor code changed: {relative}")
    for relative in SOURCE_FILES:
        path = RUN_DIR / relative
        source_hashes[relative] = _sha256(path)

    identity = {
        "kind": "round44-resume-execution-identity/1",
        "status": "passed",
        "mechanical_version": MECHANICAL_VERSION,
        "state_bridge_version": STATE_BRIDGE_VERSION,
        "head": head,
        "branch": branch,
        "authorized_base_execution_head": BASE_EXECUTION_HEAD,
        "task_start_head": TASK_START_HEAD,
        "allowed_task_delta_paths": allowed_delta,
        "observed_task_delta_paths": changed,
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
        "no_scientific_configuration_change": True,
    }
    _write_exclusive_or_match(IDENTITY_PATH, identity)
    return identity


def verify_resume_execution_identity(frozen: dict[str, Any], current_head: str) -> dict[str, Any]:
    if not IDENTITY_PATH.is_file():
        raise RuntimeError("ROUND44 versioned resume execution identity is missing")
    identity = _read(IDENTITY_PATH)
    branch = _git("branch", "--show-current")
    if (identity.get("status") != "passed"
            or identity.get("mechanical_version") != MECHANICAL_VERSION
            or identity.get("state_bridge_version") != STATE_BRIDGE_VERSION
            or identity.get("head") != current_head
            or current_head != _git("rev-parse", "HEAD")
            or branch != BRANCH or identity.get("branch") != BRANCH
            or _git("status", "--porcelain")):
        raise RuntimeError("ROUND44 current branch/HEAD/worktree differs from resume identity")
    for ancestor in (BASE_EXECUTION_HEAD, TASK_START_HEAD):
        subprocess.run(["git", "-C", str(_repo_root()), "merge-base", "--is-ancestor",
                        ancestor, current_head], check=True, stdout=subprocess.DEVNULL)
    changed = sorted(path.replace("\\", "/") for path in _git(
        "diff", "--name-only", f"{TASK_START_HEAD}..{current_head}").splitlines())
    if changed != identity.get("observed_task_delta_paths") \
            or not set(changed).issubset(ALLOWED_TASK_DELTA):
        raise RuntimeError("ROUND44 resume source delta differs from its committed whitelist")
    if (identity.get("training_version") != TRAINING_VERSION
            or frozen.get("round42", {}).get("training_version") != TRAINING_VERSION
            or identity.get("eta_lambda") != 0.5
            or frozen.get("round42", {}).get("objective", {}).get("eta_lambda") != 0.5):
        raise RuntimeError("ROUND44 resume identity changed the scientific training version/eta")
    whitelist = verify_state_whitelist(check_live_mutables=False)
    if (_sha256(WHITELIST_PATH) != identity.get("state_whitelist_sha256")
            or _sha256(RUN_DIR / "runtime" / "ROUND44_PRELAUNCH_IDENTITY.json")
            != identity.get("original_prelaunch_receipt_sha256")
            or _sha256(RUN_DIR / "ROUND44_FROZEN_CONFIG.json")
            != identity.get("original_frozen_config_sha256")):
        raise RuntimeError("ROUND44 resume identity no longer matches its original freeze/whitelist")
    original = _load_original_receipt()
    for relative, expected in identity.get("source_code_sha256", {}).items():
        path = RUN_DIR / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"ROUND44 resume source code hash mismatch: {relative}")
        if relative in original["source_code_sha256"] \
                and relative not in ("r44_execute.py", "r44_runtime.py") \
                and expected != original["source_code_sha256"][relative]:
            raise RuntimeError(f"ROUND44 original executor source drifted: {relative}")
    return identity


def verify_resume_ready(frozen: dict[str, Any], current_head: str) -> dict[str, Any]:
    identity = verify_resume_execution_identity(frozen, current_head)
    if not READY_PATH.is_file():
        raise RuntimeError("ROUND44 exact-resume and comparator gates are not ready for a claim")
    ready = _read(READY_PATH)
    if (ready.get("kind") != "round44-resume-ready/1"
            or ready.get("status") != "passed"
            or ready.get("execution_head") != current_head
            or ready.get("mechanical_version") != MECHANICAL_VERSION
            or ready.get("state_bridge_version") != STATE_BRIDGE_VERSION
            or ready.get("execution_identity_sha256") != _sha256(IDENTITY_PATH)
            or ready.get("state_whitelist_sha256") != _sha256(WHITELIST_PATH)):
        raise RuntimeError("ROUND44 resume-ready receipt does not match this frozen execution")
    for key, path in (("exact_resume_gate_sha256", GATE_PATH),
                      ("comparator_regression_sha256", COMPARATOR_TEST_PATH),
                      ("first_batch_comparison_sha256",
                       RUN_DIR / "ROUND44_FIRST_BATCH_COMPARISON_SEED11.json"),
                      ("first_batch_gate_sha256", RUN_DIR / "ROUND44_FIRST_BATCH_GATE_SEED11.json")):
        if not path.is_file() or ready.get(key) != _sha256(path):
            raise RuntimeError(f"ROUND44 ready receipt lacks the sealed gate artifact: {key}")
    gate = _read(GATE_PATH)
    regression = _read(COMPARATOR_TEST_PATH)
    comparison = _read(RUN_DIR / "ROUND44_FIRST_BATCH_COMPARISON_SEED11.json")
    first_batch = _read(RUN_DIR / "ROUND44_FIRST_BATCH_GATE_SEED11.json")
    if (gate.get("status") != "passed" or gate.get("environment_starts") != 0
            or gate.get("optimizer_steps") != 0 or gate.get("checkpoint_writes") != 0
            or gate.get("training_seed") != 11 or gate.get("episodes_done") != 4
            or gate.get("updates") != 1 or gate.get("optimizer_steps_saved") != 12
            or gate.get("lambda") != 1.125
            or regression.get("status") != "passed"
            or regression.get("expected_rejections") != ["parameter", "dual", "optimizer"]
            or comparison.get("status") != "passed"
            or first_batch.get("status") != "passed"
            or first_batch.get("training_seed") != 11):
        raise RuntimeError("ROUND44 resume-ready gate contents are incomplete or failed")
    return identity


def verify_pre_resume_state(frozen: dict[str, Any], current_head: str) -> dict[str, Any]:
    """Last exact live-state check before the runner mutates cumulative ledgers/progress."""
    verify_resume_ready(frozen, current_head)
    return verify_state_whitelist(check_live_mutables=True)


__all__ = [
    "BASE_EXECUTION_HEAD", "BRANCH", "GATE_PATH", "IDENTITY_PATH",
    "MECHANICAL_VERSION", "READY_PATH", "RESUME_PROGRESS_PATH", "STATE_BRIDGE_VERSION",
    "TASK_START_HEAD", "WHITELIST_PATH", "build_state_whitelist",
    "seal_resume_execution_identity", "verify_pre_resume_state",
    "verify_resume_execution_identity", "verify_resume_ready", "verify_state_whitelist",
]
