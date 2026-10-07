"""ROUND46 adapter: BC-initialized constrained PPO with the frozen R44 runtime.

This entry point deliberately has no default action. It reuses the real R45 BC
model-only loader, R44 runtime configuration, R42 training entry, R43 first-batch
gate, and R44 final evaluation machinery. ROUND46 owns only its identities,
quota, fixed key sequence, execution order, and reports.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


R46_DIR = Path(__file__).resolve().parent
REPO = R46_DIR.parents[1]
ROUND45_DIR = REPO / "runs" / "comm_rule_warmstart_cost_learning_20261006_r2"
R44_DIR = REPO / "runs" / "comm_dual_response_ablation_20261004"
R42_DIR = REPO / "runs" / "comm_one_shot_gate_resume_20261003_v2"
INPUT_ROOT = REPO.parent / "zhenghe"
ROUND40_ROOT = REPO.parent / "zhenghe-round40"
ROUND40_OUTPUT = ROUND40_ROOT / "runs" / "comm_first_report_training_ablation_20261003"

ROUND46_BRANCH = "fix/comm-bc-initialization-readback"
ROUND46_START_HEAD = "f27bed1904ed3d8b08ead2d821c325bdaab6d04f"
ROUND46_RUNNER_VERSION = (
    "round46-bc-initialized-constrained-ppo/5+"
    "round45-model-only-bc-loader/1+"
    "round44-constrained-ppo-runtime/1+"
    "cross-seed-batch0-gate/1+"
    "checkpoint-readback-factory-scope/1+"
    "mutable-task-state-guard/1+"
    "inner-claim-frozen-source-binding/2+"
    "reused-zero-environment-gates/2+"
    "git-worktree-text-identity-namespace/1"
)
R2_BASIS_FIRST_LINE = (
    "当前任务=ROUND46-R2；分支=fix/comm-bc-initialization-readback；"
    "隔离目录=runs/comm_bc_initialized_constrained_ppo_20261007_r2；"
    "480正常+24储备，hard504。"
)
TRAINING_VERSION = (
    "constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/"
    "dual-eta-lambda-0p5/1/rule-bc-late147-init/1/"
    "published-completion-event-at-relay-tick/2"
)
SEEDS = (11, 22, 33)
TRAIN_EPISODES_PER_SEED = 128
EPISODES_PER_BATCH = 4
TRAINING_KEYS = 384
EVALUATION_KEYS = 96
NORMAL_CAP = 480
RESERVE_CAP = 24
HARD_CAP = 504
DEV_SCENES = tuple(range(1_301_000, 1_301_016))
BOOTSTRAP_SEED = 20261007
BOOTSTRAP_REPLICATES = 10_000

R45_FREEZE_PATH = ROUND45_DIR / "ROUND45_R2_POST_BC_ENV_FREEZE_R3_FINAL3.json"
R45_CONTROL_GATE_PATH = ROUND45_DIR / "ROUND45_R2_BC_CONTROL_GATE_R3.json"
R45_CONTROL_RECOMPUTATION_PATH = (
    ROUND45_DIR / "ROUND45_R2_CONTROL_RAW_ROW_RECOMPUTATION_R3_FINAL3.json"
)
R45_KEY_PLAN_PATH = ROUND45_DIR / "ROUND45_R2_ENVIRONMENT_KEY_PLAN_R3_FINAL3.json"
R44_FREEZE_PATH = R44_DIR / "ROUND44_FROZEN_CONFIG.json"
R44_BASELINE_PATH = R44_DIR / "ROUND44_BASELINE_REUSE.json"
R44_MATRIX_AUDIT_PATH = R44_DIR / "ROUND44_EVAL_MATRIX_AUDIT.json"

FROZEN_CONFIG_PATH = R46_DIR / "ROUND46_R2_V5_FROZEN_CONFIG.json"
CONFIG_DIFF_PATH = R46_DIR / "ROUND46_R2_V5_CONFIGURATION_DIFF.json"
KEY_PLAN_PATH = R46_DIR / "ROUND46_R2_V5_KEY_PLAN.json"
TRAINING_VERSION_PATH = R46_DIR / "ROUND46_R2_V5_TRAINING_VERSION_MANIFEST.json"
INPUT_IDENTITY_PATH = R46_DIR / "ROUND46_R2_V5_INPUT_IDENTITY.json"
RUNTIME_FREEZE_PATH = R46_DIR / "ROUND46_R2_V5_RUNTIME_FREEZE.json"
PRELAUNCH_GATE_PATH = R46_DIR / "ROUND46_R2_V5_PRELAUNCH_GATE.json"
PRELAUNCH_SHA_PATH = R46_DIR / "ROUND46_R2_V5_PRELAUNCH_GATE.sha256"
CURRENT_TASK_PATH = R46_DIR / "ROUND46_CURRENT_TASK.json"
LEDGER_PATH = R46_DIR / "runtime" / "ROUND46_EXECUTION_LEDGER.json"
REPORT_PATH = R46_DIR / "ROUND46_REPORT.md"
FINAL_RESULTS_PATH = R46_DIR / "ROUND46_FINAL_RESULTS.json"
PAIRED_RESULTS_PATH = R46_DIR / "ROUND46_PAIRED_RESULTS.json"
TARGET_DIFFS_PATH = R46_DIR / "ROUND46_TARGET_ID_DIFFS.json"
CURVES_PATH = R46_DIR / "ROUND46_TRAINING_CURVES.json"
CHECKPOINT_INDEX_PATH = R46_DIR / "ROUND46_CHECKPOINT_INDEX.json"
ARTIFACT_MANIFEST_PATH = R46_DIR / "ROUND46_ARTIFACT_MANIFEST.json"
STOP_STATE_PATH = R46_DIR / "ROUND46_R2_V5_STOP_STATE.json"
PRIOR_R46_DIR = REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007"
READBACK_REGRESSION_PATH = R46_DIR / "test_round46_r2_checkpoint_readback.py"
READBACK_REGRESSION_RESULT_PATH = R46_DIR / "ROUND46_R2_CHECKPOINT_READBACK_REGRESSION_R3.json"
STATUS_GUARD_REGRESSION_PATH = R46_DIR / "ROUND46_R2_STATUS_GUARD_REGRESSION.json"
V3_FROZEN_CONFIG_PATH = R46_DIR / "ROUND46_FROZEN_CONFIG.json"
V3_CONFIG_DIFF_PATH = R46_DIR / "ROUND46_CONFIGURATION_DIFF.json"
V3_INPUT_IDENTITY_PATH = R46_DIR / "ROUND46_INPUT_IDENTITY.json"
V3_KEY_PLAN_PATH = R46_DIR / "ROUND46_KEY_PLAN.json"
V3_TRAINING_VERSION_PATH = R46_DIR / "TRAINING_VERSION_MANIFEST.json"
V3_PRELAUNCH_GATE_PATH = R46_DIR / "ROUND46_PRELAUNCH_GATE.json"
V3_PRELAUNCH_SHA_PATH = R46_DIR / "ROUND46_PRELAUNCH_GATE.sha256"
V3_RUNTIME_FREEZE_PATH = R46_DIR / "ROUND46_RUNTIME_FREEZE.json"
V3_STOP_ARCHIVE_PATH = R46_DIR / "ROUND46_R2_V3_STOP_ARCHIVE.json"
V4_BINDING_AUDIT_PATH = R46_DIR / "ROUND46_R2V4_BINDING_AUDIT.json"
V4_FROZEN_CONFIG_PATH = R46_DIR / "ROUND46_R2_V4_FROZEN_CONFIG.json"
V4_CONFIG_DIFF_PATH = R46_DIR / "ROUND46_R2_V4_CONFIGURATION_DIFF.json"
V4_KEY_PLAN_PATH = R46_DIR / "ROUND46_R2_V4_KEY_PLAN.json"
V4_TRAINING_VERSION_PATH = R46_DIR / "ROUND46_R2_V4_TRAINING_VERSION_MANIFEST.json"
V4_INPUT_IDENTITY_PATH = R46_DIR / "ROUND46_R2_V4_INPUT_IDENTITY.json"
V4_RUNTIME_FREEZE_PATH = R46_DIR / "ROUND46_R2_V4_RUNTIME_FREEZE.json"
V4_PREFLIGHT_FAILURE_PATH = R46_DIR / "ROUND46_R2_V4_PREFLIGHT_FAILURE.json"
V4_PREFLIGHT_TRACEBACK_PATH = R46_DIR / "ROUND46_R2_V4_PREFLIGHT_FAILURE_TRACEBACK.txt"
V4_STOP_REPORT_PATH = R46_DIR / "ROUND46_R2_V4_PREFLIGHT_STOP_REPORT.md"
V4_PRELAUNCH_GATE_PATH = R46_DIR / "ROUND46_R2_V4_PRELAUNCH_GATE.json"
V4_BINDING_REGRESSION_PATH = R46_DIR / "ROUND46_R2_V4_BINDING_REGRESSION.json"
V4_STATUS_GUARD_REUSE_PATH = R46_DIR / "ROUND46_R2_V4_STATUS_GUARD_REUSE.json"
V5_BINDING_REGRESSION_PATH = R46_DIR / "ROUND46_R2_V5_BINDING_REGRESSION.json"
V5_STATUS_GUARD_REUSE_PATH = R46_DIR / "ROUND46_R2_V5_STATUS_GUARD_REUSE.json"
V5_TEXT_IDENTITY_REGRESSION_PATH = R46_DIR / "ROUND46_R2_V5_TEXT_IDENTITY_REGRESSION.json"
# --- ROUND46 /6 private execution output namespace --------------------------
# Every /5 execution output (recovery receipt, checkpoints, progress, rows,
# logs, claims, journal) landed in the ROUND46 root. The frozen pre-claim guard
# `_prior_r2_preclaim_attempt_identity` refuses to start while any
# `v2_train_s*.pt` exists in `R46_DIR/checkpoints`, and training itself writes
# exactly those files there, so a run in the shared root can never take its
# first atomic claim. /6 therefore moves the whole output root to a private
# directory and keeps every /5 file in place. The family execution ledger stays
# the single cumulative ledger; the /6 journal is seeded byte-for-byte from the
# family journal so the checkpoint-save count continues and is never reset.
V6_DIR = R46_DIR / "v6"
V6_RUNTIME_DIR = V6_DIR / "runtime"
V6_CHECKPOINTS_DIR = V6_DIR / "checkpoints"
V6_PROGRESS_DIR = V6_DIR / "training_progress"
V6_LOGS_DIR = V6_DIR / "logs"
V6_ROWS_DIR = V6_DIR / "rows"
V6_REJECTED_DIR = V6_DIR / "rejected"
FAMILY_LEDGER_PATH = R46_DIR / "runtime" / "ROUND46_EXECUTION_LEDGER.json"
FAMILY_JOURNAL_PATH = R46_DIR / "runtime" / "checkpoint_writes.jsonl"

# /6 artifacts. The /5 files they replace are preserved on disk unchanged; the
# names below are the active identities the /6 run writes and verifies.
V6_FROZEN_CONFIG_PATH = R46_DIR / "ROUND46_R2_V6_FROZEN_CONFIG.json"
V6_CONFIG_DIFF_PATH = R46_DIR / "ROUND46_R2_V6_CONFIGURATION_DIFF.json"
V6_KEY_PLAN_PATH = R46_DIR / "ROUND46_R2_V6_KEY_PLAN.json"
V6_TRAINING_VERSION_PATH = R46_DIR / "ROUND46_R2_V6_TRAINING_VERSION_MANIFEST.json"
V6_INPUT_IDENTITY_PATH = R46_DIR / "ROUND46_R2_V6_INPUT_IDENTITY.json"
V6_RUNTIME_FREEZE_PATH = R46_DIR / "ROUND46_R2_V6_RUNTIME_FREEZE.json"
V6_PRELAUNCH_GATE_PATH = R46_DIR / "ROUND46_R2_V6_PRELAUNCH_GATE.json"
V6_PRELAUNCH_SHA_PATH = R46_DIR / "ROUND46_R2_V6_PRELAUNCH_GATE.sha256"
V6_TEXT_IDENTITY_REGRESSION_PATH = (
    R46_DIR / "ROUND46_R2_V6_TEXT_IDENTITY_REGRESSION.json"
)
V6_STATUS_GUARD_REUSE_PATH = R46_DIR / "ROUND46_R2_V6_STATUS_GUARD_REUSE.json"
V6_BINDING_REGRESSION_PATH = R46_DIR / "ROUND46_R2_V6_BINDING_REGRESSION.json"
V6_CHECKPOINT_JOURNAL_ROUTE_AUDIT_PATH = (
    R46_DIR / "ROUND46_R2_V6_CHECKPOINT_JOURNAL_ROUTE_AUDIT.json"
)
V6_STOP_STATE_PATH = R46_DIR / "ROUND46_R2_V6_STOP_STATE.json"
V6_PREFLIGHT_NOTE_PATH = R46_DIR / "ROUND46_R2_V6_PREFLIGHT_NOTE.json"

FROZEN_CONFIG_PATH = V6_FROZEN_CONFIG_PATH
CONFIG_DIFF_PATH = V6_CONFIG_DIFF_PATH
KEY_PLAN_PATH = V6_KEY_PLAN_PATH
TRAINING_VERSION_PATH = V6_TRAINING_VERSION_PATH
INPUT_IDENTITY_PATH = V6_INPUT_IDENTITY_PATH
RUNTIME_FREEZE_PATH = V6_RUNTIME_FREEZE_PATH
PRELAUNCH_GATE_PATH = V6_PRELAUNCH_GATE_PATH
PRELAUNCH_SHA_PATH = V6_PRELAUNCH_SHA_PATH
V5_TEXT_IDENTITY_REGRESSION_PATH = V6_TEXT_IDENTITY_REGRESSION_PATH
V5_STATUS_GUARD_REUSE_PATH = V6_STATUS_GUARD_REUSE_PATH
V5_BINDING_REGRESSION_PATH = V6_BINDING_REGRESSION_PATH
CHECKPOINT_JOURNAL_ROUTE_AUDIT_PATH = V6_CHECKPOINT_JOURNAL_ROUTE_AUDIT_PATH
STOP_STATE_PATH = V6_STOP_STATE_PATH

CHECKPOINT_JOURNAL_PATH = V6_RUNTIME_DIR / "checkpoint_writes.jsonl"
CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH = (
    R46_DIR / "ROUND46_R2_CHECKPOINT_JOURNAL_ROUTE_REGRESSION.json"
)
LEGACY_CHECKPOINT_JOURNAL_PATH = (
    ROUND45_DIR / "runtime" / "checkpoint_writes.jsonl"
)
R2_V2_PRECLAIM_ARCHIVE_PATH = (
    R46_DIR / "attempts" / "runner_v2_preclaim_guard" / "ATTEMPT_ARCHIVE.json"
)

_R45_MODULE: Any = None
_RUNTIME: dict[str, Any] | None = None
_R2_V5_REQUIRE_EXECUTION_EVIDENCE = False


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any, *, sort_keys: bool = False) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=sort_keys, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    if immutable and path.exists():
        if path.read_bytes() != payload.encode("utf-8"):
            raise RuntimeError(f"immutable ROUND46 file conflicts with existing bytes: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    os.replace(temporary, path)


def _pretty_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_text_lf(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def _run_git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=check, capture_output=True, text=True, encoding="utf-8",
    )
    # Porcelain status starts each worktree-only modification with a leading
    # space. Preserve it for the source-cleanliness parser; retain the prior
    # trimmed contract for branch, HEAD, and other Git queries.
    is_porcelain_status = args[:1] == ("status",) and "--porcelain" in args
    stdout = result.stdout.rstrip("\r\n") if is_porcelain_status else result.stdout.strip()
    if check:
        return stdout
    return stdout + ("\n" + result.stderr.strip() if result.stderr else "")


def _assert_branch_head(*, exact_start: bool = False) -> None:
    branch = _run_git("branch", "--show-current")
    head = _run_git("rev-parse", "HEAD")
    if branch != ROUND46_BRANCH:
        raise RuntimeError(f"ROUND46 branch mismatch: expected {ROUND46_BRANCH}, found {branch}")
    if exact_start and head != ROUND46_START_HEAD:
        raise RuntimeError(f"ROUND46 start HEAD mismatch: {head}")
    if not exact_start:
        ancestor = subprocess.run(
            ["git", "-C", str(REPO), "merge-base", "--is-ancestor",
             ROUND46_START_HEAD, "HEAD"],
            capture_output=True,
        )
        if ancestor.returncode != 0:
            raise RuntimeError("ROUND46 start commit is no longer an ancestor of HEAD")


def _assert_source_worktree_clean() -> None:
    status = _run_git("status", "--porcelain", "--untracked-files=no")
    unexpected = _unexpected_source_status_lines(status)
    if unexpected:
        raise RuntimeError(
            "tracked source worktree is dirty; refusing ROUND46 environment entry: "
            + "\n".join(unexpected))


def _unexpected_source_status_lines(status: str) -> list[str]:
    mutable_task_paths = {
        CURRENT_TASK_PATH.resolve().relative_to(REPO).as_posix(),
        (R46_DIR / "CURRENT_TASK.json").resolve().relative_to(REPO).as_posix(),
    }
    # The family execution ledger lives under the round's runtime directory and
    # is rewritten by every claim/ledger snapshot during a run, exactly like the
    # mutable current-task aliases. Allow only the round-runtime mutable state.
    mutable_runtime_prefix = (
        (R46_DIR / "runtime").resolve().relative_to(REPO).as_posix() + "/")
    unexpected = []
    for line in status.splitlines():
        path = line[3:].replace("\\", "/") if len(line) >= 4 else ""
        if line[:2] == " M" and (path in mutable_task_paths
                                 or path.startswith(mutable_runtime_prefix)):
            continue
        unexpected.append(line)
    return unexpected


def _prior_failed_attempt_identity() -> dict[str, Any]:
    """Bind this continuation to the untouched, zero-claim ROUND46 v1 stop."""
    files = {
        "execution_basis": PRIOR_R46_DIR / "EXECUTION_BASIS.md",
        "frozen_config": PRIOR_R46_DIR / "ROUND46_FROZEN_CONFIG.json",
        "prelaunch_gate": PRIOR_R46_DIR / "ROUND46_PRELAUNCH_GATE.json",
        "key_plan": PRIOR_R46_DIR / "ROUND46_KEY_PLAN.json",
        "input_identity": PRIOR_R46_DIR / "ROUND46_INPUT_IDENTITY.json",
        "stop_state": PRIOR_R46_DIR / "ROUND46_STOP_STATE.json",
        "ledger": PRIOR_R46_DIR / "runtime" / "ROUND46_EXECUTION_LEDGER.json",
        "checkpoint_write_journal": PRIOR_R46_DIR / "runtime" / "checkpoint_writes.jsonl",
        "seed11_ep0": PRIOR_R46_DIR / "checkpoints" / "v2_train_s11_ep0.pt",
        "seed11_rolling": PRIOR_R46_DIR / "checkpoints" / "v2_train_s11.pt",
    }
    expected = {
        "execution_basis": "2d6aac2fb99728cf717633c00db83d0614edca94947824d8df9f4ffc480851ce",
        "frozen_config": "f222622b28c67397451695f42598838d6e43a1a90b4f1fbd46d86addb57f0c42",
        "prelaunch_gate": "f4dbeffa293bb12e9fb6c97f2723f8ca56cbae508da5eb15f6746b1469138faa",
        "key_plan": "bd058fdc692f77a0cd7b6aa8bb3caa16078c1b95fd708cbc652ce4b94dbb34f7",
        "input_identity": "aa1a886f1cf7787dfb74ff3fb0ba3ae4019593c219218424f670d51c91c8d98e",
        "stop_state": "cb52d590e714b9ede9e77ca1ed8ecc6f25fbcbe41c85236987891acd99cf9706",
        "ledger": "0650276608d2baafb9e13de88a660aba5f44b1c6a73b610712ad0a1e603357b9",
        "checkpoint_write_journal": "40bd7397f5cfd08b9dd2ae7afcfa53540aff1e1f1826b4f3537db32e6ba849d3",
        "seed11_ep0": "712b1b619be5fed6822bf45dbd97bfb400f4d577874f4af07f4c74d2851b3835",
        "seed11_rolling": "712b1b619be5fed6822bf45dbd97bfb400f4d577874f4af07f4c74d2851b3835",
    }
    actual = {}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"preserved ROUND46 v1 evidence is missing: {path}")
        digest = _sha256(path)
        if digest != expected[name]:
            raise RuntimeError(f"preserved ROUND46 v1 evidence changed: {name} {digest}")
        actual[name] = {"path": str(path.resolve()), "sha256": digest}
    stop = _json(files["stop_state"])
    ledger = _json(files["ledger"])
    counts = ledger.get("counts", {})
    expected_zero = (
        "normal_used", "reserve_used", "total_used", "environment_constructor_calls",
        "joint_run_constructed_calls", "run_invocation_calls", "completed_steps",
        "completed_logical_jobs", "ppo_update_calls", "optimizer_step_calls",
    )
    if any(int(counts.get(key, -1)) != 0 for key in expected_zero):
        raise RuntimeError("ROUND46 v1 ledger no longer has the registered zero-start state")
    if int(counts.get("checkpoint_write_calls", -1)) != 1:
        raise RuntimeError("ROUND46 v1 checkpoint write count differs from the preserved stop")
    if (stop.get("status") != "stopped"
            or stop.get("error", {}).get("type") != "RuntimeError"
            or stop.get("error", {}).get("message")
            != "seed 11: fresh BC controller injection was invoked twice"
            or stop.get("ledger", {}).get("counts", {}).get("checkpoint_write_calls") != 1):
        raise RuntimeError("ROUND46 v1 stop state no longer matches its readback failure")
    if _json(files["prelaunch_gate"]).get("status") != "passed":
        raise RuntimeError("ROUND46 v1 zero-environment prelaunch receipt is not passed")
    return {
        "kind": "round46-r2-prior-attempt-identity/1",
        "prior_runner_version": stop.get("runner_version"),
        "prior_status": stop.get("status"),
        "prior_error": stop.get("error"),
        "prior_counts": counts,
        "prior_claims": len(ledger.get("claims", [])),
        "prior_key_cursor": {"normal": 0, "reserve": 0, "training_episode": 0,
                              "ppo_batch": 0, "final_evaluation": 0},
        "files": actual,
        "ep0_and_rolling_byte_identical": (
            actual["seed11_ep0"]["sha256"] == actual["seed11_rolling"]["sha256"]
        ),
        "new_training_may_load_prior_checkpoint": False,
    }


def _prior_r2_preclaim_attempt_identity() -> dict[str, Any]:
    """Bind the pre-claim v2 stop and prove its untrained checkpoints are archived."""
    if not R2_V2_PRECLAIM_ARCHIVE_PATH.is_file():
        raise FileNotFoundError(f"ROUND46-R2 v2 pre-claim archive is missing: {R2_V2_PRECLAIM_ARCHIVE_PATH}")
    manifest = _json(R2_V2_PRECLAIM_ARCHIVE_PATH)
    if (manifest.get("kind") != "round46-r2-runner-v2-preclaim-stop-archive/1"
            or manifest.get("status") != "preserved; not resumed"
            or manifest.get("training_version") != TRAINING_VERSION
            or manifest.get("counts") != {
                "claims": 0, "environment_constructor_calls": 0,
                "run_invocation_calls": 0, "completed_steps": 0,
                "ppo_updates": 0, "optimizer_step_calls": 0,
                "checkpoint_write_calls": 8,
                "checkpoint_write_journal_entries": 8,
                "synthetic_fixture_writes": 6,
                "preclaim_training_checkpoint_write_calls": 2,
            }):
        raise RuntimeError("ROUND46-R2 v2 pre-claim archive identity or counters changed")

    archived_files = {}
    for item in manifest.get("files", []):
        relative = item.get("archived_relative_path")
        path = (R46_DIR / str(relative)).resolve()
        try:
            path.relative_to(R46_DIR.resolve())
        except ValueError as error:
            raise RuntimeError(f"archived v2 artifact escapes R2 output: {path}") from error
        if not path.is_file() or _sha256(path) != item.get("sha256"):
            raise RuntimeError(f"archived v2 artifact identity changed: {path}")
        archived_files[item.get("original_relative_path")] = item

    checkpoint_entries = [item for key, item in archived_files.items()
                          if str(key).startswith("checkpoints/") and str(key).endswith(".pt")]
    checkpoint_hashes = {item["sha256"] for item in checkpoint_entries}
    payload = manifest.get("checkpoint_payloads", {})
    if (len(checkpoint_entries) != 3 or len(checkpoint_hashes) != 1
            or checkpoint_hashes != {payload.get("sha256")}
            or payload.get("all_three_files_byte_identical") is not True
            or payload.get("seed") != 11 or payload.get("episodes_done") != 0
            or payload.get("updates") != 0 or payload.get("optimizer_steps") != 0
            or payload.get("lambda") != 1.0):
        raise RuntimeError("archived v2 training checkpoints are not the verified untrained ep0 payload")
    if list((R46_DIR / "checkpoints").glob("v2_train_s*.pt")):
        raise RuntimeError("an archived ROUND46-R2 v2 checkpoint remains active for recovery")

    stop_item = archived_files.get("ROUND46_STOP_STATE.json")
    recovery_item = archived_files.get("runtime/ROUND45_R2_EP0_RECOVERY_s11.json")
    if stop_item is None or recovery_item is None:
        raise RuntimeError("ROUND46-R2 v2 stop or ep0 recovery receipt was not archived")
    stop_path = R46_DIR / stop_item["archived_relative_path"]
    stop = _json(stop_path)
    stop_counts = stop.get("ledger", {}).get("counts", {})
    if (stop.get("status") != "stopped"
            or stop.get("error", {}).get("message")
            != "tracked worktree is dirty; refusing ROUND46 environment entry: M runs/comm_bc_initialized_constrained_ppo_20261007_r2/ROUND46_CURRENT_TASK.json"
            or any(int(stop_counts.get(key, -1)) != 0 for key in (
                "environment_constructor_calls", "run_invocation_calls", "completed_steps",
                "ppo_update_calls", "optimizer_step_calls"))
            or int(stop_counts.get("checkpoint_write_calls", -1)) != 8):
        raise RuntimeError("ROUND46-R2 v2 archived stop state no longer matches its pre-claim failure")
    recovery_path = R46_DIR / recovery_item["archived_relative_path"]
    recovery = _json(recovery_path)
    if (recovery.get("status") != "passed" or recovery.get("seed") != 11
            or recovery.get("episodes_done") != 0 or recovery.get("updates") != 0
            or recovery.get("optimizer_steps") != 0
            or recovery.get("source_sha256") != next(iter(checkpoint_hashes))
            or recovery.get("byte_identical") is not True):
        raise RuntimeError("ROUND46-R2 v2 ep0 recovery receipt does not match archived checkpoint bytes")

    ledger_record = manifest.get("ledger_snapshot", {})
    ledger_path = (R46_DIR / str(ledger_record.get("path", ""))).resolve()
    if (not ledger_path.is_file()
            or _sha256(ledger_path) != ledger_record.get("sha256")):
        raise RuntimeError("ROUND46-R2 v2 stop ledger snapshot identity changed")
    return {
        "kind": manifest["kind"],
        "archive_manifest_path": R2_V2_PRECLAIM_ARCHIVE_PATH.relative_to(REPO).as_posix(),
        "archive_manifest_sha256": _sha256(R2_V2_PRECLAIM_ARCHIVE_PATH),
        "runner_version": manifest["runner_version"],
        "counts": manifest["counts"],
        "archived_untrained_checkpoint_count": len(checkpoint_entries),
        "archived_checkpoint_sha256": next(iter(checkpoint_hashes)),
        "archived_stop_state_sha256": stop_item["sha256"],
        "archived_recovery_receipt_sha256": recovery_item["sha256"],
        "archived_ledger_snapshot_sha256": ledger_record["sha256"],
        "active_prior_checkpoint_count": 0,
        "new_training_may_load_prior_checkpoint": False,
    }


def _checkpoint_journal_route_audit(*, create: bool) -> dict[str, Any]:
    """Preserve and classify R2 writes that the R45 loader misrouted to its output."""
    if not LEGACY_CHECKPOINT_JOURNAL_PATH.is_file():
        raise FileNotFoundError(
            f"misrouted ROUND46 checkpoint journal is missing: {LEGACY_CHECKPOINT_JOURNAL_PATH}"
        )
    source_bytes = LEGACY_CHECKPOINT_JOURNAL_PATH.read_bytes()
    if not create and CHECKPOINT_JOURNAL_ROUTE_AUDIT_PATH.is_file():
        audit = _json(CHECKPOINT_JOURNAL_ROUTE_AUDIT_PATH)
        local_bytes = CHECKPOINT_JOURNAL_PATH.read_bytes() if CHECKPOINT_JOURNAL_PATH.is_file() else b""
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        if (audit.get("kind") != "round46-r2-checkpoint-journal-route-audit/2"
                or audit.get("legacy_path") != str(LEGACY_CHECKPOINT_JOURNAL_PATH.resolve())
                or audit.get("legacy_sha256") != source_hash
                or audit.get("legacy_entry_count") != len(source_bytes.splitlines())
                or audit.get("r2_local_copy_path") != str(CHECKPOINT_JOURNAL_PATH.resolve())
                or audit.get("r2_local_copy_initial_sha256") != source_hash
                or not local_bytes.startswith(source_bytes)):
            raise RuntimeError("preserved checkpoint-journal route identity or local prefix changed")
        return audit
    raw_lines = source_bytes.splitlines()
    entries = [json.loads(line.decode("utf-8")) for line in raw_lines if line.strip()]
    if len(entries) != 15 or len(raw_lines) != 15:
        raise RuntimeError(
            f"misrouted ROUND46 checkpoint journal expected 15 historical entries; got {len(entries)}"
        )
    normalized_paths = [str(item.get("path_at_write", "")).replace("/", "\\").casefold()
                        for item in entries]
    fixture_root = str(R46_DIR / "test_fixtures").replace("/", "\\").casefold() + "\\"
    checkpoint_root = str(R46_DIR / "checkpoints").replace("/", "\\").casefold() + "\\"
    if any(not path.startswith(fixture_root) and not path.startswith(checkpoint_root)
           for path in normalized_paths):
        raise RuntimeError("misrouted checkpoint journal contains a path outside ROUND46-R2")
    if any(item.get("training_version") != TRAINING_VERSION
           or int(item.get("episodes_done", -1)) != 0
           or int(item.get("updates", -1)) != 0 for item in entries):
        raise RuntimeError("misrouted checkpoint journal contains a non-ep0 or foreign-version write")

    categories = ["v2_synthetic_fixture"] * 6 + ["v2_preclaim_untrained_checkpoint"] * 2
    categories.append("partial_v3_development_probe_fixture")
    categories.extend(["v3_synthetic_fixture"] * 6)
    for index, (path, category) in enumerate(zip(normalized_paths, categories, strict=True)):
        if index < 6 and "\\test_fixtures\\seed_" not in path:
            raise RuntimeError("historical v2 fixture journal entries do not match their category")
        if 6 <= index < 8 and "\\checkpoints\\v2_train_s11_" not in path:
            raise RuntimeError("historical v2 pre-claim entries do not match their category")
        if index == 8 and "\\test_fixtures\\seed_11\\fresh\\checkpoints\\" not in path:
            raise RuntimeError("partial v3 probe journal entry does not match its saved seed-11 fixture")
        if index >= 9 and "\\test_fixtures\\runner_3\\" not in path:
            raise RuntimeError("historical v3 fixture journal entries do not match their category")

    archived_v2 = _json(R2_V2_PRECLAIM_ARCHIVE_PATH)
    v2_fixture_manifest = _json(R46_DIR / "test_fixtures" / "ROUND46_R2_FIXTURE_MANIFEST.json")
    v3_manifest_path = R46_DIR / "test_fixtures" / "runner_3" / "ROUND46_R2_FIXTURE_MANIFEST.json"
    v3_manifest = _json(v3_manifest_path)
    round45_stop_path = ROUND45_DIR / "ROUND45_R2_STOP_STATE_R3_FINAL3.json"
    round45_stop = _json(round45_stop_path)
    round45_checkpoint_writes = int(
        round45_stop.get("runtime_ledger", {}).get("counts", {}).get(
            "checkpoint_write_calls", -1))
    if (archived_v2.get("counts", {}).get("checkpoint_write_journal_entries") != 8
            or archived_v2.get("counts", {}).get("synthetic_fixture_writes") != 6
            or archived_v2.get("counts", {}).get("preclaim_training_checkpoint_write_calls") != 2
            or v3_manifest.get("kind") != "round46-r2-synthetic-untrained-checkpoint-fixtures/1"
            or v3_manifest.get("trainer_save_checkpoint_calls") != 6
            or v3_manifest.get("environment_starts") != 0
            or v3_manifest.get("optimizer_step_calls") != 0
            or round45_checkpoint_writes != 0):
        raise RuntimeError("historical checkpoint journal categories lack matching preserved receipts")

    def final_checkpoint_path(entry: dict[str, Any]) -> Path:
        written_path = str(entry.get("path_at_write", ""))
        match = re.fullmatch(r"(.*)\.\d+\.(?:milestone-tmp|snapshot-tmp)", written_path)
        return Path(match.group(1) if match else written_path).resolve()

    v2_fixture_saves = [item for item in v2_fixture_manifest.get("fixtures", [])
                        if item.get("role") in {"fresh_bc_ep0", "changed_payload_ep0"}]
    v3_fixture_saves = [item for item in v3_manifest.get("fixtures", [])
                        if item.get("role") in {"fresh_bc_ep0", "changed_payload_ep0"}]
    if len(v2_fixture_saves) != 6 or len(v3_fixture_saves) != 6:
        raise RuntimeError("v2/v3 fixture receipts do not enumerate six actual trainer saves each")
    for index, fixture in enumerate(v2_fixture_saves):
        entry = entries[index]
        if (final_checkpoint_path(entry) != Path(fixture["path"]).resolve()
                or entry.get("sha256_at_write") != fixture.get("sha256")):
            raise RuntimeError(f"v2 journal entry {index + 1} differs from its fixture receipt")
    archived_checkpoint_hash = archived_v2.get("checkpoint_payloads", {}).get("sha256")
    for index in (6, 7):
        if entries[index].get("sha256_at_write") != archived_checkpoint_hash:
            raise RuntimeError(f"v2 pre-claim journal entry {index + 1} differs from archived checkpoint bytes")
    partial_probe = entries[8]
    partial_path = final_checkpoint_path(partial_probe)
    if not partial_path.is_file() or _sha256(partial_path) != partial_probe.get("sha256_at_write"):
        raise RuntimeError("v3 failed-probe journal entry has no matching preserved synthetic checkpoint")
    for index, fixture in enumerate(v3_fixture_saves, start=9):
        entry = entries[index]
        if (final_checkpoint_path(entry) != Path(fixture["path"]).resolve()
                or entry.get("sha256_at_write") != fixture.get("sha256")):
            raise RuntimeError(f"v3 journal entry {index + 1} differs from its fixture receipt")

    if not LEGACY_CHECKPOINT_JOURNAL_PATH.is_file():
        raise RuntimeError("legacy journal disappeared during route audit")
    target_bytes = CHECKPOINT_JOURNAL_PATH.read_bytes() if CHECKPOINT_JOURNAL_PATH.exists() else None
    if create:
        if target_bytes is None:
            CHECKPOINT_JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with CHECKPOINT_JOURNAL_PATH.open("xb") as stream:
                stream.write(source_bytes)
            target_bytes = source_bytes
        elif target_bytes != source_bytes:
            raise RuntimeError("ROUND46-local checkpoint journal already exists with different bytes")
    if target_bytes is None or not target_bytes.startswith(source_bytes):
        raise RuntimeError("ROUND46-local checkpoint journal does not preserve the misrouted history prefix")

    audit = {
        "kind": "round46-r2-checkpoint-journal-route-audit/2",
        "status": "preserved; future writes scoped to ROUND46-R2",
        "legacy_path": str(LEGACY_CHECKPOINT_JOURNAL_PATH.resolve()),
        "legacy_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "legacy_entry_count": len(entries),
        "legacy_entries": [
            {"ordinal": index + 1, "category": categories[index],
             "runner_version_attributed": (
                 archived_v2["runner_version"] if index < 8 else ROUND46_RUNNER_VERSION
             ), **item}
            for index, item in enumerate(entries)
        ],
        "write_order_basis": "append-only JSONL order; each record's destination and payload SHA were matched to the contemporaneous fixture/archive receipt",
        "counts_by_category": {
            "v2_synthetic_fixture": 6,
            "v2_preclaim_untrained_checkpoint": 2,
            "partial_v3_development_probe_fixture": 1,
            "v3_synthetic_fixture": 6,
            "unclassified": 0,
        },
        "r2_local_copy_path": str(CHECKPOINT_JOURNAL_PATH.resolve()),
        "r2_local_copy_initial_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "round45_stop_state_path": str(round45_stop_path.resolve()),
        "round45_stop_state_sha256": _sha256(round45_stop_path),
        "round45_stop_state_checkpoint_write_calls": round45_checkpoint_writes,
        "legacy_file_modified_by_this_audit": False,
        "future_checkpoint_journal_path": str(CHECKPOINT_JOURNAL_PATH.resolve()),
    }
    if CHECKPOINT_JOURNAL_ROUTE_AUDIT_PATH.exists():
        if _json(CHECKPOINT_JOURNAL_ROUTE_AUDIT_PATH) != audit:
            raise RuntimeError("checkpoint journal route audit conflicts with preserved identity")
    elif create:
        _write_json(CHECKPOINT_JOURNAL_ROUTE_AUDIT_PATH, audit, immutable=True)
    else:
        raise FileNotFoundError("ROUND46 checkpoint journal route audit was not frozen")
    return audit


def _rebind_checkpoint_journal(runtime: dict[str, Any]) -> dict[str, Any]:
    """Point the real R42 save hook and ledger at this ROUND46 output directory."""
    run_module = runtime["r42_run"]
    trainer = runtime["trainer"]
    wrapper = trainer.save_checkpoint
    closure = wrapper.__closure__ or ()
    cells = dict(zip(wrapper.__code__.co_freevars, closure, strict=True))
    journal_cell = cells.get("checkpoint_journal")
    if journal_cell is None:
        raise RuntimeError("actual R42 checkpoint writer has no auditable journal-path closure")
    captured_path = Path(journal_cell.cell_contents).resolve()
    if captured_path != LEGACY_CHECKPOINT_JOURNAL_PATH.resolve():
        raise RuntimeError(f"unexpected captured R42 checkpoint journal path: {captured_path}")
    audit = _checkpoint_journal_route_audit(create=False)
    try:
        journal_cell.cell_contents = CHECKPOINT_JOURNAL_PATH
    except (AttributeError, TypeError) as error:
        raise RuntimeError("Python runtime cannot safely rebind the R42 journal-path cell") from error
    if Path(journal_cell.cell_contents).resolve() != CHECKPOINT_JOURNAL_PATH.resolve():
        raise RuntimeError("actual R42 checkpoint writer did not accept the ROUND46-local journal path")

    run_module.RUN_DIR = V6_DIR
    analysis_module = runtime.get("r42_analysis")
    if analysis_module is not None and Path(analysis_module.RUN_DIR).resolve() == ROUND45_DIR.resolve():
        analysis_module.RUN_DIR = V6_DIR
    line_count = sum(1 for line in CHECKPOINT_JOURNAL_PATH.open("r", encoding="utf-8") if line.strip())
    runtime["r42_versions"].CHECKPOINT_WRITE_CALLS = line_count
    if (Path(run_module.RUN_DIR).resolve() != V6_DIR.resolve()
            or line_count < int(audit["legacy_entry_count"])):
        raise RuntimeError("ROUND46 checkpoint writer, ledger, and journal output scopes disagree")
    return {
        "r42_run_output_dir": str(Path(run_module.RUN_DIR).resolve()),
        "journal_closure_before": str(captured_path),
        "journal_closure_after": str(CHECKPOINT_JOURNAL_PATH.resolve()),
        "journal_entry_count": line_count,
        "legacy_journal_sha256_preserved": audit["legacy_sha256"],
        "route_audit_sha256": _sha256(CHECKPOINT_JOURNAL_ROUTE_AUDIT_PATH),
    }


def _python_default_json_list_sha256(values: list[int]) -> str:
    # Matches hashlib.sha256(json.dumps(values).encode("utf-8")) for Python ints.
    return hashlib.sha256(json.dumps(values).encode("utf-8")).hexdigest()


def _source_scenario_order() -> tuple[list[int], dict[str, Any]]:
    frozen = _json(R44_FREEZE_PATH)
    order = [int(value) for value in frozen["round42"]["training_order"]]
    if len(order) != 128 or len(set(order)) != 48 or set(order) != set(range(1_300_000, 1_300_048)):
        raise RuntimeError("ROUND44 frozen training order is not the exact 128-item fixed sequence")
    expected_sha = frozen["round42"]["training_order_sha256"]
    actual_sha = _python_default_json_list_sha256(order)
    if actual_sha != expected_sha:
        raise RuntimeError("ROUND44 training order hash does not match Python default JSON serialization")
    r45_plan = _json(R45_KEY_PLAN_PATH)
    flat = sorted(r45_plan["ppo_training_keys"],
                  key=lambda item: (int(item["seed"]), int(item["episode_index"])))
    if len(flat) != TRAINING_KEYS:
        raise RuntimeError(f"ROUND45 training key plan has {len(flat)} entries, expected 384")
    comparisons = []
    for seed in SEEDS:
        actual = [int(item["scene"]) for item in flat if int(item["seed"]) == seed]
        if len(actual) != 128:
            raise RuntimeError(f"ROUND45 seed {seed} has {len(actual)} training keys, expected 128")
        diffs = [
            {"episode_index": index, "round44_scene": order[index],
             "round45_scene": actual[index]}
            for index in range(128) if actual[index] != order[index]
        ]
        comparisons.append({
            "seed": seed, "count": len(actual), "exact_elementwise_match": not diffs,
            "difference_count": len(diffs), "differences": diffs,
            "round45_episode_order_sha256_python_default_json":
                _python_default_json_list_sha256(actual),
        })
        if diffs:
            raise RuntimeError(f"ROUND45 seed {seed} scenario sequence differs from ROUND44")
    r45_sequence = [int(value) for value in r45_plan["training_scenario_order"]]
    if r45_sequence != order:
        raise RuntimeError("ROUND45 stored sequence differs from ROUND44 despite per-seed checks")
    return order, {
        "source": str(R44_FREEZE_PATH),
        "round44_stored_sha256": expected_sha,
        "round44_python_default_json_sha256": actual_sha,
        "round45_stored_sha256": r45_plan["training_scenario_order_sha256"],
        "round45_python_default_json_sha256": _python_default_json_list_sha256(r45_sequence),
        "round45_stored_serialization": "compact JSON separators, no comma-space",
        "round44_hash_serialization": "Python json.dumps(list_of_ints).encode('utf-8'), default separators",
        "hash_note": "Different stored hashes are JSON serialization differences; all 128 entries match element-by-element for each seed.",
        "per_seed_comparisons": comparisons,
    }


def _logical_key_plan(order: list[int], sequence_record: dict[str, Any]) -> dict[str, Any]:
    training = []
    for seed in SEEDS:
        for episode_index, scene in enumerate(order):
            training.append({
                "scope": "ppo_training",
                "logical_job": f"new_training_s{seed}_e{episode_index}",
                "seed": seed, "episode_index": episode_index, "scene": scene,
                "normal": True,
            })
    evaluations = []
    for seed in SEEDS:
        for mode in ("greedy", "sampled"):
            for scene in DEV_SCENES:
                evaluations.append({
                    "scope": "final_ep128_eval",
                    "logical_job": f"model_eval_new_s{seed}_{mode}_scene{scene}",
                    "seed": seed, "mode": mode, "scene": scene, "normal": True,
                })
    keys = [item["logical_job"] for item in training + evaluations]
    if (len(training) != TRAINING_KEYS or len(evaluations) != EVALUATION_KEYS
            or len(keys) != NORMAL_CAP or len(set(keys)) != NORMAL_CAP):
        raise RuntimeError("ROUND46 fixed 480-key plan failed count or uniqueness checks")
    return {
        "kind": "round46-logical-key-plan/1",
        "training_scenario_order": order,
        "training_scenario_order_sha256": _python_default_json_list_sha256(order),
        "training_scenario_order_hash_method":
            "sha256(json.dumps(order).encode('utf-8')); Python default separators",
        "scenario_sequence_verification": sequence_record,
        "training_keys": training, "final_evaluation_keys": evaluations,
        "logical_jobs": keys,
        "training_key_count": len(training), "final_evaluation_key_count": len(evaluations),
        "normal_key_count": len(keys), "technical_reserve_max": RESERVE_CAP,
        "hard_environment_claim_cap": HARD_CAP,
    }


def _verify_r45_control_inputs() -> dict[str, Any]:
    freeze = _json(R45_FREEZE_PATH)
    control_gate = _json(R45_CONTROL_GATE_PATH)
    recomputation = _json(R45_CONTROL_RECOMPUTATION_PATH)
    if control_gate.get("status") != "failed" or control_gate.get("training_authorized_by_gate") is not False:
        raise RuntimeError("ROUND45 BC-only gate no longer has its original failed status")
    gate_sha = _sha256(R45_CONTROL_GATE_PATH)
    if gate_sha.lower() != "ccad7597d9b75c46683cd1b49e252deb928b1ee3b183fa9c1c4580d106e5af05":
        raise RuntimeError("ROUND45 failed BC-only gate bytes changed")
    if _sha256(R45_FREEZE_PATH) != "1deba66aa01aeb96782ecfffaf05f4c24a02b05fe8fcde9bb747d47ef9aaec67":
        raise RuntimeError("ROUND45 post-BC freeze bytes changed")
    if recomputation.get("status") != "passed":
        raise RuntimeError("registered ROUND45 BC control row recomputation is not passed")
    files = recomputation["control_rows"]["files"]
    if len(files) != 48 or not recomputation["control_rows"]["all_keys_unique_and_complete"]:
        raise RuntimeError("ROUND45 BC control raw-row registry is not exactly the frozen 48 rows")
    checked = []
    for item in files:
        row_path = Path(item.get("row_path", item.get("path", "")))
        if not row_path.is_absolute():
            row_path = ROUND45_DIR / row_path
        expected = item.get("row_sha256", item.get("sha256"))
        if not row_path.is_file() or not expected or _sha256(row_path) != expected:
            raise RuntimeError(f"ROUND45 registered BC control row identity failed: {row_path}")
        checked.append({"path": str(row_path), "sha256": expected})
    return {
        "round45_freeze_sha256": _sha256(R45_FREEZE_PATH),
        "round45_failed_gate_sha256": gate_sha,
        "round45_failed_gate_status": control_gate["status"],
        "round45_control_recomputation_sha256": _sha256(R45_CONTROL_RECOMPUTATION_PATH),
        "round45_control_recomputation_status": recomputation["status"],
        "registered_bc_control_rows_count": len(checked),
        "registered_bc_control_rows": checked,
        "round45_freeze": freeze,
    }


def _copy_runtime_aliases(source_identity: dict[str, Any]) -> None:
    # Copy only code and small identity/configuration receipts. BC metrics,
    # checkpoints, readback records, and optimizer journals remain at their
    # original ROUND45 paths and are referenced by hash, never copied here.
    copy_pairs = [
        (ROUND45_DIR / "ROUND45_CONFIGURATION_DIFF.json",
         R46_DIR / "ROUND45_CONFIGURATION_DIFF.json"),
        (ROUND45_DIR / "ROUND45_R2_BC_TRAINING_SUMMARY.json",
         R46_DIR / "ROUND45_R2_BC_TRAINING_SUMMARY.json"),
    ]
    for name in (
        "r40_train_resume_adapter.py", "r42_one_shot.py",
        "r42_versions.py", "r43_call_counter_hooks.py",
    ):
        copy_pairs.append((ROUND45_DIR / name, R46_DIR / name))
    for source, destination in copy_pairs:
        if not source.is_file():
            raise FileNotFoundError(source)
        expected = _sha256(source)
        if destination.exists():
            if _sha256(destination) != expected:
                raise RuntimeError(f"ROUND46 runtime provenance alias conflicts: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        if _sha256(destination) != expected:
            raise RuntimeError(f"ROUND46 runtime provenance copy hash mismatch: {destination}")

    legacy_task = R46_DIR / "ROUND45_CURRENT_TASK.json"
    if not legacy_task.exists():
        _write_json(legacy_task, {
            "kind": "round46-r45-loader-task-alias/1", "stage": "prepared",
            "branch": ROUND46_BRANCH, "runner_version": ROUND46_RUNNER_VERSION,
            "training_version": TRAINING_VERSION, "environment_constructed": 0,
            "ppo_updates": 0, "normal_environment_keys": NORMAL_CAP,
            "technical_reserve_max": RESERVE_CAP, "hard_environment_cap": HARD_CAP,
        })


@contextlib.contextmanager
def _checkpoint_restore_factory_scope(runtime: dict[str, Any]):
    """Keep BC injection out of every actual payload-to-controller restore."""
    trainer = runtime["trainer"]
    controller_module = runtime["controller_module"]
    original_restore = trainer._restore_controller
    base_factory = runtime["round46_r2_base_fresh_controller"]
    audit = runtime["round46_r2_restore_scope_audit"]

    def restore_from_payload(payload: dict[str, Any], state: dict[str, Any]):
        active_factory = controller_module.fresh_controller
        controller_module.fresh_controller = base_factory
        audit["payload_restore_calls"] += 1
        try:
            return original_restore(payload, state)
        finally:
            controller_module.fresh_controller = active_factory

    trainer._restore_controller = restore_from_payload
    audit["scope_entries"] += 1
    try:
        yield
    finally:
        trainer._restore_controller = original_restore


def _install_round46_r2_training_factory_scope(module: Any,
                                               runtime: dict[str, Any]) -> None:
    """Scope ordinary payload restores to the base factory, leaving BC for fresh init only."""
    original_call = module._call_training_seed
    runtime["round46_r2_legacy_call_training_seed"] = original_call
    runtime["round46_r2_base_fresh_controller"] = runtime[
        "controller_module"].fresh_controller
    runtime["round46_r2_restore_scope_audit"] = {
        "scope_entries": 0, "payload_restore_calls": 0,
        "base_factory_id": id(runtime["controller_module"].fresh_controller),
        "training_factory_id_before_patch": id(runtime["controller_module"].fresh_controller),
    }

    def call_training_seed(runtime_arg: dict[str, Any], seed: int, *,
                           episodes: int) -> dict[str, Any]:
        if runtime_arg is not runtime:
            raise RuntimeError("ROUND46-R2 training wrapper received an unknown runtime")
        with _checkpoint_restore_factory_scope(runtime):
            return original_call(runtime_arg, seed, episodes=episodes)

    module._call_training_seed = call_training_seed


def _seed_v6_output_namespace() -> dict[str, Any]:
    """Create the /6 private output root and continue the family checkpoint journal.

    The /6 journal starts as a byte-for-byte continuation of the family journal
    (never zero), so the cumulative checkpoint-save count is preserved. The
    preserved route audit is re-scoped to the /6 journal path with all of its
    historical fields unchanged.
    """
    for directory in (V6_DIR, V6_RUNTIME_DIR, V6_CHECKPOINTS_DIR, V6_PROGRESS_DIR,
                      V6_LOGS_DIR, V6_ROWS_DIR, V6_REJECTED_DIR,
                      V6_DIR / "launch_claims", V6_DIR / "launch_slots",
                      V6_DIR / "training_reports", V6_DIR / "failures"):
        directory.mkdir(parents=True, exist_ok=True)
    alias = V6_DIR / "ROUND45_CURRENT_TASK.json"
    if not alias.exists():
        source_alias = R46_DIR / "ROUND45_CURRENT_TASK.json"
        if source_alias.is_file():
            shutil.copyfile(source_alias, alias)
        else:
            _write_json(alias, {
                "kind": "round46-r45-loader-task-alias/1", "stage": "prepared",
                "branch": ROUND46_BRANCH, "runner_version": ROUND46_RUNNER_VERSION,
                "training_version": TRAINING_VERSION, "environment_constructed": 0,
                "ppo_updates": 0, "normal_environment_keys": NORMAL_CAP,
                "technical_reserve_max": RESERVE_CAP, "hard_environment_cap": HARD_CAP,
            })
    if not FAMILY_JOURNAL_PATH.is_file():
        raise FileNotFoundError(FAMILY_JOURNAL_PATH)
    family_bytes = FAMILY_JOURNAL_PATH.read_bytes()
    if CHECKPOINT_JOURNAL_PATH.exists():
        existing = CHECKPOINT_JOURNAL_PATH.read_bytes()
        if not existing.startswith(family_bytes):
            raise RuntimeError("/6 checkpoint journal no longer continues the family journal")
    else:
        with CHECKPOINT_JOURNAL_PATH.open("xb") as stream:
            stream.write(family_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    seed_sha = hashlib.sha256(CHECKPOINT_JOURNAL_PATH.read_bytes()).hexdigest()
    v5_audit_path = R46_DIR / "ROUND46_R2_CHECKPOINT_JOURNAL_ROUTE_AUDIT_R2.json"
    if not v5_audit_path.is_file():
        raise FileNotFoundError(v5_audit_path)
    scoped = dict(_json(v5_audit_path))
    scoped["r2_local_copy_path"] = str(CHECKPOINT_JOURNAL_PATH.resolve())
    scoped["future_checkpoint_journal_path"] = str(CHECKPOINT_JOURNAL_PATH.resolve())
    scoped["v6_family_journal_seed_sha256"] = seed_sha
    scoped["v6_family_journal_entries"] = len(family_bytes.splitlines())
    if CHECKPOINT_JOURNAL_ROUTE_AUDIT_PATH.exists():
        current = _json(CHECKPOINT_JOURNAL_ROUTE_AUDIT_PATH)
        if current != scoped and current.get("v6_family_journal_seed_sha256") != seed_sha:
            raise RuntimeError("/6 checkpoint-journal route audit conflicts with preserved identity")
    else:
        _write_json(CHECKPOINT_JOURNAL_ROUTE_AUDIT_PATH, scoped, immutable=True)
    return {"family_journal_entries": scoped["v6_family_journal_entries"], "seed_sha256": seed_sha}


def _configure_runtime_module() -> Any:
    global _R45_MODULE, _RUNTIME
    if _R45_MODULE is not None and _RUNTIME is not None:
        return _R45_MODULE
    _seed_v6_output_namespace()
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    _copy_runtime_aliases(_input_identity())
    for path in (ROUND45_DIR, R44_DIR, R42_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    source = ROUND45_DIR / "r45_r2_execute_r3.py"
    spec = importlib.util.spec_from_file_location("round46_r45_runtime_adapter", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import verified R45 helper source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    expected_source_hashes = _json(R45_FREEZE_PATH)["source_files_sha256"]
    actual_source_hashes = module._source_hashes()
    if actual_source_hashes != expected_source_hashes:
        differing = sorted(
            key for key in set(actual_source_hashes) | set(expected_source_hashes)
            if actual_source_hashes.get(key) != expected_source_hashes.get(key)
        )
        raise RuntimeError(f"ROUND45 frozen runtime source identity changed: {differing}")
    module.RUN_DIR = V6_DIR
    module.RUNNER_VERSION = ROUND46_RUNNER_VERSION
    module.R3_FREEZE_PATH = RUNTIME_FREEZE_PATH
    module.R3_KEY_PLAN_PATH = KEY_PLAN_PATH
    module.R3_PRELAUNCH_PATH = PRELAUNCH_GATE_PATH
    module.R3_EXECUTION_LEDGER_PATH = LEDGER_PATH
    module.R3_REPORT_PATH = REPORT_PATH
    module.R3_FINAL_RESULTS_PATH = FINAL_RESULTS_PATH
    module.R3_STATUS_PARSER_REGRESSION_PATH = R46_DIR / "ROUND46_STATUS_PATH_CHECK.json"
    module.ADOPTION_PATH = R46_DIR / "ROUND46_NO_ADOPTION_RECEIPT.json"
    module.RUNTIME = None
    module.R2_FREEZE = None
    module.ENFORCE_PRELAUNCH = False
    # R45's read-only BC manifest resolves its verification receipt, metrics,
    # and optimizer journal through RUN_DIR. Read those immutable originals in
    # place; do not copy the BC journals into this ROUND46 output directory.
    module.RUN_DIR = ROUND45_DIR
    try:
        runtime = module.load_runtime(enforce_prelaunch=False)
    finally:
        module.RUN_DIR = V6_DIR
    if runtime["r42_versions"].TRAINING_VERSION != TRAINING_VERSION:
        raise RuntimeError("ROUND46 actual R42 trainer loaded a different training semantic version")
    common = runtime["common"]
    r40_common = runtime["r40_common"]
    common.OUTPUT_DIR = V6_DIR
    common.LOGS = V6_LOGS_DIR
    common.ROWS = V6_ROWS_DIR
    common.REJECTED = V6_REJECTED_DIR
    common.CHECKPOINTS = V6_CHECKPOINTS_DIR
    r40_common.ROUND_DIR = V6_DIR
    common.NORMAL_LAUNCHES = NORMAL_CAP
    common.RESERVE_LAUNCHES = RESERVE_CAP
    common.LAUNCH_CAP = HARD_CAP
    r40_common.NORMAL_CAP = NORMAL_CAP
    r40_common.RESERVE_CAP = RESERVE_CAP
    r40_common.HARD_CAP = HARD_CAP
    r40_common.ROUND_DIR = V6_DIR
    r40_common.plan = lambda: {
        "normal": {"new_training": TRAINING_KEYS, "new_final_evaluation": EVALUATION_KEYS},
        "normal_total": NORMAL_CAP, "technical_reserve_max": RESERVE_CAP,
        "hard_cap": HARD_CAP,
    }
    common.DEVICE = torch.device("cpu")
    if ((int(common.NORMAL_LAUNCHES), int(common.RESERVE_LAUNCHES), int(common.LAUNCH_CAP))
            != (NORMAL_CAP, RESERVE_CAP, HARD_CAP)):
        raise RuntimeError("ROUND46 quota did not reach the actual R42/common runtime")
    if ((int(r40_common.NORMAL_CAP), int(r40_common.RESERVE_CAP), int(r40_common.HARD_CAP))
            != (NORMAL_CAP, RESERVE_CAP, HARD_CAP)):
        raise RuntimeError("ROUND46 quota did not reach the atomic claim ledger")
    if (float(common.GAMMA) != 1.0 or float(common.GAE_LAMBDA) != 1.0
            or float(common.ETA_LAMBDA) != 0.5 or float(common.B0_BYTES) != 10000.0
            or float(common.DELTA) != 0.0 or float(common.LAMBDA_INIT) != 1.0
            or str(common.DEVICE) != "cpu"):
        raise RuntimeError("ROUND46 constrained-PPO math or CPU device differs from freeze")
    expected_order = [int(value) for value in _json(KEY_PLAN_PATH)["training_scenario_order"]]
    if [int(value) for value in common.scenario_order()] != expected_order:
        raise RuntimeError("ROUND46 actual training runtime scenario order differs from its freeze")
    if tuple(int(seed) for seed in common.TRAIN_SEEDS) != SEEDS:
        raise RuntimeError("ROUND46 actual training runtime seed set differs from its freeze")

    r45_freeze = _json(R45_FREEZE_PATH)
    order = [int(value) for value in _json(KEY_PLAN_PATH)["training_scenario_order"]]
    plan = _json(KEY_PLAN_PATH)
    module.R2_FREEZE = {
        "kind": "round46-runtime-key-freeze/1",
        "runner_version": ROUND46_RUNNER_VERSION,
        "training_version": TRAINING_VERSION,
        "bc_models": copy.deepcopy(r45_freeze["bc_models"]),
        "logical_key_plan": {"logical_jobs": list(plan["logical_jobs"])},
        "scenario_order": order,
    }

    original_record_claim = module._record_claim

    def record_claim(context: dict[str, Any], logical_job: str, attempt: int,
                     metadata: dict[str, Any], base_claim: Any):
        path = original_record_claim(context, logical_job, attempt, metadata, base_claim)
        r40_common._update_claim(
            path, round=46, runner_version=ROUND46_RUNNER_VERSION,
            training_version=TRAINING_VERSION,
            round46_key=dict(context),
            round46_attempt_kind="normal" if attempt == 1 else "technical_reserve",
            round46_artifact_paths=module._claim_paths(context, attempt),
        )
        return path

    module._record_claim = record_claim

    original_receipt = module._create_round45_artifact_receipt

    def create_receipt(row: dict[str, Any], claim: dict[str, Any]) -> dict[str, Any]:
        receipt = original_receipt(row, claim)
        receipt["round46_execution_identity"] = {
            "round": 46, "runner_version": ROUND46_RUNNER_VERSION,
            "training_version": TRAINING_VERSION,
        }
        return receipt

    module._create_round45_artifact_receipt = create_receipt

    run_module = runtime["r42_run"]
    runtime["round46_checkpoint_journal_route"] = _rebind_checkpoint_journal(runtime)

    def write_ledger(*, strict: bool = True) -> dict[str, Any]:
        ledger = run_module.ledger_snapshot(strict=strict)
        ledger.update({
            "kind": "round46-execution-ledger/1", "round": 46,
            "runner_version": ROUND46_RUNNER_VERSION,
            "training_version": TRAINING_VERSION,
            "normal_cap": NORMAL_CAP, "technical_reserve_max": RESERVE_CAP,
            "hard_cap": HARD_CAP,
            "logical_key_plan_sha256": _sha256(KEY_PLAN_PATH),
        })
        _write_json(LEDGER_PATH, ledger)
        return ledger

    run_module.write_execution_ledger = write_ledger
    original_task_state = module._task_state

    def task_state(stage: str, **updates: Any) -> None:
        original_task_state(stage, **updates)
        state = _json(CURRENT_TASK_PATH) if CURRENT_TASK_PATH.exists() else {}
        state.update({
            "kind": "round46-current-task/1", "round": 46,
            "stage": stage, "branch": ROUND46_BRANCH,
            "head": _run_git("rev-parse", "HEAD"),
            "runner_version": ROUND46_RUNNER_VERSION,
            "training_version": TRAINING_VERSION,
            "updated_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        state.update(updates)
        _write_json(CURRENT_TASK_PATH, state)

    module._task_state = task_state
    runtime["module"] = module
    _install_round46_r2_training_factory_scope(module, runtime)
    _bind_r44_claim_verifier(module, runtime)
    _R45_MODULE = module
    _RUNTIME = runtime
    return module


def _input_identity() -> dict[str, Any]:
    journal_route = _checkpoint_journal_route_audit(create=False)
    source_freeze = _json(R45_FREEZE_PATH)
    control = _verify_r45_control_inputs()
    bc_models = {}
    for item in source_freeze["bc_models"]:
        seed = int(item["seed"])
        required = (
            ("checkpoint_path", "checkpoint_sha256"),
            ("readback_receipt_path", "readback_receipt_sha256"),
            ("metrics_path", "metrics_sha256"),
            ("journal_path", "journal_sha256"),
        )
        model = dict(item)
        for path_key, sha_key in required:
            path = Path(item[path_key])
            if not path.is_file() or _sha256(path) != item[sha_key]:
                raise RuntimeError(f"ROUND45 BC source identity mismatch for seed {seed}: {path}")
        bc_models[str(seed)] = model
    r44 = _json(R44_FREEZE_PATH)
    registered_rng = {}
    for seed in SEEDS:
        item = r44["round43"]["registered_ep0_inputs"][str(seed)]
        path = Path(item["path"])
        actual = _sha256(path)
        if actual != item["file_sha256"]:
            raise RuntimeError(f"ROUND44 ep0 RNG source hash mismatch for seed {seed}")
        registered_rng[str(seed)] = {
            "path": str(path), "sha256": actual,
            "parameter_sha256": item.get("parameter_sha256"),
        }
    order, order_check = _source_scenario_order()
    return {
        "round45_freeze_sha256": _sha256(R45_FREEZE_PATH),
        "round45_bc_summary_sha256": _sha256(
            ROUND45_DIR / "ROUND45_R2_BC_TRAINING_SUMMARY.json"),
        "round45_control_gate_sha256": control["round45_failed_gate_sha256"],
        "round45_control_recomputation_sha256":
            control["round45_control_recomputation_sha256"],
        "round45_control_rows": control["registered_bc_control_rows"],
        "round44_frozen_config_sha256": _sha256(R44_FREEZE_PATH),
        "round44_baseline_reuse_sha256": _sha256(R44_BASELINE_PATH),
        "round44_eval_matrix_audit_sha256": _sha256(R44_MATRIX_AUDIT_PATH),
        "round44_registered_ep0_rng": registered_rng,
        "bc_models_by_seed": bc_models,
        "scenario_order": order, "scenario_order_verification": order_check,
        "round45_source_files_sha256": source_freeze["source_files_sha256"],
        "round45_freeze_runner_version": source_freeze["runner_version"],
        "round45_gate_status_preserved": "failed",
        "round45_gate_training_authorized": False,
        "prior_round46_failed_attempt": _prior_failed_attempt_identity(),
        "prior_round46_r2_preclaim_attempt": _prior_r2_preclaim_attempt_identity(),
        "checkpoint_journal_route_audit_sha256": _sha256(CHECKPOINT_JOURNAL_ROUTE_AUDIT_PATH),
        "checkpoint_journal_route_legacy_sha256": journal_route["legacy_sha256"],
        "checkpoint_journal_route_initial_entry_count": journal_route["legacy_entry_count"],
    }


def _closure_values(function: Any) -> dict[str, Any]:
    code = getattr(function, "__code__", None)
    cells = getattr(function, "__closure__", None)
    if code is None or cells is None or len(code.co_freevars) != len(cells):
        return {}
    values = {}
    for name, cell in zip(code.co_freevars, cells, strict=True):
        try:
            values[name] = cell.cell_contents
        except ValueError:
            continue
    return values


def _actual_r44_claim(runtime: dict[str, Any], module: Any) -> tuple[Any, Any]:
    r44_module = getattr(module, "R44_RUNTIME_MODULE", None)
    run_module = runtime.get("r42_run")
    if r44_module is None or run_module is None:
        raise RuntimeError("ROUND46 could not locate the configured R44/R42 claim modules")
    wrapped_claim = getattr(run_module, "_claim", None)
    base_claim = _closure_values(wrapped_claim).get("base_claim")
    base_code = getattr(base_claim, "__code__", None)
    if (not callable(base_claim) or base_code is None
            or Path(base_code.co_filename).resolve()
            != (R44_DIR / "r44_runtime.py").resolve()
            or getattr(base_claim, "__globals__", None) is not r44_module.__dict__):
        raise RuntimeError("ROUND46 claim wrapper no longer reaches the configured R44 claim function")
    if base_claim.__globals__.get("verify_frozen_source") is not getattr(
            r44_module, "verify_frozen_source", None):
        raise RuntimeError("R44 claim global verifier pointer differs from the R44 module attribute")
    return r44_module, base_claim


def _bind_r44_claim_verifier(module: Any, runtime: dict[str, Any]) -> dict[str, Any]:
    """Bind only the R46 adapter's strict verifier into the real R44 module global."""
    r44_module, claim_function = _actual_r44_claim(runtime, module)
    stale_verifier = r44_module.verify_frozen_source
    if stale_verifier is not getattr(module, "verify_prelaunch_identity", None):
        raise RuntimeError("ROUND46 expected the audited R45 verifier pointer before rebinding R44")

    def strict_claim_verifier(frozen: dict[str, Any]) -> str:
        expected_frozen = _json(R44_FREEZE_PATH)
        if _json_sha256(frozen, sort_keys=True) != _json_sha256(
                expected_frozen, sort_keys=True):
            raise RuntimeError("R44 claim supplied a frozen object outside the registered R44 source")
        _verify_prelaunch_identity(
            runtime, require_r2_v5_evidence=_R2_V5_REQUIRE_EXECUTION_EVIDENCE)
        return _run_git("rev-parse", "HEAD")

    strict_claim_verifier.__name__ = "round46_r2_v5_claim_source_verifier"
    strict_claim_verifier.__qualname__ = strict_claim_verifier.__name__
    r44_module.verify_frozen_source = strict_claim_verifier
    bound_r44_module, bound_claim = _actual_r44_claim(runtime, module)
    if (bound_r44_module is not r44_module or bound_claim is not claim_function
            or bound_claim.__globals__.get("verify_frozen_source") is not strict_claim_verifier):
        raise RuntimeError("ROUND46 inner R44 claim verifier binding did not take effect")
    binding = {
        "r44_module": r44_module,
        "claim_function": claim_function,
        "stale_verifier": stale_verifier,
        "strict_verifier": strict_claim_verifier,
    }
    runtime["round46_r44_claim_binding"] = binding
    module._round46_r44_claim_binding = binding
    return binding


def _git_blob_bytes(relative_path: str) -> bytes:
    object_id = _run_git("rev-parse", f"HEAD:{relative_path}")
    return subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "blob", object_id],
        check=True, capture_output=True,
    ).stdout


def _compare_text_file_identity(
        raw_bytes: bytes, expected_raw_sha256: str, git_blob_bytes: bytes,
        expected_git_blob_content_sha256: str | None = None) -> dict[str, str]:
    """Compare committed UTF-8 text while keeping raw and Git-blob hashes distinct."""
    if not isinstance(raw_bytes, bytes) or not isinstance(git_blob_bytes, bytes):
        raise TypeError("ROUND46 text identity inputs must be bytes")
    if b"\x00" in raw_bytes or b"\x00" in git_blob_bytes:
        raise RuntimeError("ROUND46 text identity check rejected a NUL-containing file")
    try:
        raw_bytes.decode("utf-8")
        git_blob_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("ROUND46 text identity check rejected non-UTF-8 bytes") from error

    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    blob_sha256 = hashlib.sha256(git_blob_bytes).hexdigest()
    if raw_sha256 != expected_raw_sha256:
        raise RuntimeError(
            "ROUND46 raw worktree file SHA256 differs from its frozen raw-file identity")
    if (expected_git_blob_content_sha256 is not None
            and blob_sha256 != expected_git_blob_content_sha256):
        raise RuntimeError(
            "ROUND46 Git blob content SHA256 differs from its frozen blob-content identity")
    if raw_bytes == git_blob_bytes:
        comparison_rule = "byte_identical"
    elif raw_bytes.replace(b"\r\n", b"\n") == git_blob_bytes:
        comparison_rule = "crlf_to_lf_only"
    else:
        raise RuntimeError(
            "ROUND46 text differs beyond byte identity or CRLF-to-LF normalization")
    return {
        "raw_file_sha256": raw_sha256,
        "git_blob_content_sha256": blob_sha256,
        "comparison_rule": comparison_rule,
    }


def _require_committed_file(
        path: Path, expected_raw_sha256: str,
        expected_git_blob_content_sha256: str | None = None) -> dict[str, str]:
    relative = path.resolve().relative_to(REPO).as_posix()
    raw_bytes = path.read_bytes()
    blob_bytes = _git_blob_bytes(relative)
    try:
        identity = _compare_text_file_identity(
            raw_bytes, expected_raw_sha256, blob_bytes,
            expected_git_blob_content_sha256)
    except (RuntimeError, TypeError) as error:
        raise RuntimeError(
            f"ROUND46 required committed text identity failed: {relative}: {error}") from error
    return {"path": relative, **identity}


def _capture_committed_text_identity(
        path: Path, expected_raw_sha256: str | None = None,
        expected_git_blob_content_sha256: str | None = None) -> dict[str, str]:
    raw_bytes = path.read_bytes()
    raw_sha256 = expected_raw_sha256 or hashlib.sha256(raw_bytes).hexdigest()
    relative = path.resolve().relative_to(REPO).as_posix()
    identity = _compare_text_file_identity(
        raw_bytes, raw_sha256, _git_blob_bytes(relative),
        expected_git_blob_content_sha256)
    return {"path": relative, **identity}


def _frozen_text_identity(path: Path) -> dict[str, str]:
    relative = path.resolve().relative_to(REPO).as_posix()
    frozen = _json(FROZEN_CONFIG_PATH)
    record = frozen.get("text_file_identity_manifest", {}).get(relative)
    if not isinstance(record, dict):
        raise RuntimeError(f"ROUND46 frozen text identity is missing: {relative}")
    return _require_committed_file(
        path, record["raw_file_sha256"], record["git_blob_content_sha256"])


def run_r2_v5_text_identity_regression() -> dict[str, Any]:
    """Exercise _require_committed_file against exact and newline-only text cases."""
    if V5_TEXT_IDENTITY_REGRESSION_PATH.exists():
        raise RuntimeError("ROUND46 V5 text identity regression output already exists")
    real_blob_reader = globals()["_git_blob_bytes"]
    before_runtime = _RUNTIME
    if before_runtime is not None:
        raise RuntimeError("ROUND46 V5 text identity regression must run before runtime initialization")
    cases: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix=".round46-r2-v5-text-", dir=REPO) as folder:
        probe = Path(folder) / "identity.txt"

        def exercise(label: str, raw: bytes, blob: bytes,
                     expected_raw: str, expected_blob: str,
                     expected_rule: str | None = None,
                     expected_error: str | None = None) -> None:
            probe.write_bytes(raw)
            globals()["_git_blob_bytes"] = lambda _relative: blob
            try:
                identity = _require_committed_file(
                    probe, expected_raw, expected_blob)
            except RuntimeError as error:
                if expected_error is None or expected_error not in str(error):
                    raise AssertionError(
                        f"ROUND46 V5 {label} rejected for the wrong reason: {error}") from error
                cases[label] = {"passed": True, "rejected": True, "reason": str(error)}
                return
            if expected_error is not None:
                raise AssertionError(f"ROUND46 V5 {label} unexpectedly passed")
            if identity["comparison_rule"] != expected_rule:
                raise AssertionError(
                    f"ROUND46 V5 {label} returned unexpected comparison rule: {identity}")
            cases[label] = {"passed": True, "rejected": False, **identity}

        try:
            lf = b"identity=stable\n"
            crlf = b"identity=stable\r\n"
            exercise(
                "lf_byte_identical", lf, lf,
                hashlib.sha256(lf).hexdigest(), hashlib.sha256(lf).hexdigest(),
                expected_rule="byte_identical")
            exercise(
                "frozen_crlf_to_matching_lf_blob", crlf, lf,
                hashlib.sha256(crlf).hexdigest(), hashlib.sha256(lf).hexdigest(),
                expected_rule="crlf_to_lf_only")
            altered = b"identity=altered\r\n"
            original_blob = b"identity=stable\n"
            exercise(
                "non_newline_content_change", altered, original_blob,
                hashlib.sha256(altered).hexdigest(), hashlib.sha256(original_blob).hexdigest(),
                expected_error="differs beyond byte identity")
            exercise(
                "raw_hash_mismatch", crlf, lf,
                hashlib.sha256(lf).hexdigest(), hashlib.sha256(lf).hexdigest(),
                expected_error="raw worktree file SHA256 differs")
            exercise(
                "blob_content_hash_mismatch", lf, lf,
                hashlib.sha256(lf).hexdigest(), hashlib.sha256(b"other\n").hexdigest(),
                expected_error="Git blob content SHA256 differs")
            reverse_normalization = b"identity=stable\r\n"
            exercise(
                "lf_to_crlf_blob_rejected", lf, reverse_normalization,
                hashlib.sha256(lf).hexdigest(), hashlib.sha256(reverse_normalization).hexdigest(),
                expected_error="differs beyond byte identity")
        finally:
            globals()["_git_blob_bytes"] = real_blob_reader

    if _RUNTIME is not before_runtime:
        raise RuntimeError("ROUND46 V5 text identity regression initialized or replaced runtime")
    if set(cases) != {
            "lf_byte_identical", "frozen_crlf_to_matching_lf_blob",
            "non_newline_content_change", "raw_hash_mismatch",
            "blob_content_hash_mismatch",
            "lf_to_crlf_blob_rejected"}:
        raise RuntimeError("ROUND46 V5 text identity regression omitted a required case")
    result = {
        "kind": "round46-r2-v5-text-identity-regression/1",
        "status": "passed",
        "runner_version": ROUND46_RUNNER_VERSION,
        "runner_source_sha256": _sha256(Path(__file__)),
        "helper_under_test": "_require_committed_file",
        "comparison_rule": "raw hash is frozen independently; Git blob content may be byte-identical or raw CRLF normalized to LF only",
        "cases": cases,
        "environment_constructor_calls": 0,
        "run_invocation_calls": 0,
        "normal_claims": 0,
        "reserve_claims": 0,
        "completed_steps": 0,
        "ppo_updates": 0,
        "optimizer_step_calls": 0,
        "checkpoint_write_calls_this_regression": 0,
        "runtime_initialized": False,
    }
    _write_json(V5_TEXT_IDENTITY_REGRESSION_PATH, result, immutable=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _function_source_hashes(source_bytes: bytes, names: tuple[str, ...]) -> dict[str, str]:
    source = source_bytes.decode("utf-8").replace("\r\n", "\n")
    tree = ast.parse(source)
    definitions = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    result = {}
    for name in names:
        node = definitions.get(name)
        segment = ast.get_source_segment(source, node) if node is not None else None
        if not segment:
            raise RuntimeError(f"ROUND46 status guard source function is unavailable: {name}")
        result[name] = hashlib.sha256(segment.encode("utf-8")).hexdigest()
    return result


def _status_guard_function_hashes(source_bytes: bytes) -> dict[str, str]:
    return _function_source_hashes(source_bytes, (
        "_assert_source_worktree_clean",
        "_unexpected_source_status_lines",
        "run_status_guard_regression",
    ))


def _create_status_guard_reuse_receipt() -> dict[str, Any]:
    if not PRELAUNCH_GATE_PATH.is_file() or not STATUS_GUARD_REGRESSION_PATH.is_file():
        raise RuntimeError("ROUND46 V5 cannot reuse the prior status guard before its gate exists")
    prior = _json(STATUS_GUARD_REGRESSION_PATH)
    prior_gate = _json(V3_PRELAUNCH_GATE_PATH)
    if (prior.get("kind") != "round46-r2-status-guard-regression/1"
            or prior.get("status") != "passed"
            or prior.get("prelaunch_gate_sha256") != _sha256(V3_PRELAUNCH_GATE_PATH)
            or prior_gate.get("status") != "passed"
            or prior_gate.get("runner_source_sha256")
            != _json(V3_FROZEN_CONFIG_PATH).get("runner_source_sha256")):
        raise RuntimeError("ROUND46 prior status guard evidence no longer closes on the V3 gate")
    old_source = subprocess.run(
        ["git", "-C", str(REPO), "show",
         "02fd982354de2cf20e5363c8c68f50d9899eb1e9:"
         + Path(__file__).resolve().relative_to(REPO).as_posix()],
        check=True, capture_output=True,
    ).stdout
    old_hashes = _status_guard_function_hashes(old_source)
    current_hashes = _status_guard_function_hashes(Path(__file__).read_bytes())
    if old_hashes != current_hashes:
        raise RuntimeError("ROUND46 status guard implementation changed; prior receipt cannot be reused")
    result = {
        "kind": "round46-r2-v5-status-guard-reuse/1",
        "status": "passed_reused_exact_implementation",
        "runner_version": ROUND46_RUNNER_VERSION,
        "training_version": TRAINING_VERSION,
        "prior_runner_version": prior.get("runner_version"),
        "prior_status_guard_receipt_sha256": _sha256(STATUS_GUARD_REGRESSION_PATH),
        "prior_prelaunch_gate_sha256": _sha256(V3_PRELAUNCH_GATE_PATH),
        "v5_prelaunch_gate_sha256": _sha256(PRELAUNCH_GATE_PATH),
        "v3_source_commit": "02fd982354de2cf20e5363c8c68f50d9899eb1e9",
        "v3_status_guard_source_function_sha256": old_hashes,
        "v5_status_guard_source_function_sha256": current_hashes,
        "actual_status_guard_regression_rerun": False,
        "old_receipt_status_preserved": prior["status"],
        "old_gate_status_preserved": prior_gate["status"],
        "environment_constructor_calls": 0,
        "run_invocation_calls": 0,
        "normal_claims": 0,
        "reserve_claims": 0,
        "ppo_updates": 0,
        "optimizer_step_calls": 0,
    }
    _write_json(V5_STATUS_GUARD_REUSE_PATH, result, immutable=True)
    return result


def prepare() -> None:
    # A pre-claim runner-gate failure can require a new execution freeze. Keep
    # all ROUND46 commits on the named branch and preserve the original start
    # commit as an ancestor, while allowing that explicitly versioned refreeze.
    _assert_branch_head()
    _assert_source_worktree_clean()
    if not (R46_DIR / "EXECUTION_BASIS.md").is_file():
        raise FileNotFoundError("EXECUTION_BASIS.md must be saved before ROUND46 prepare")
    basis_first = (R46_DIR / "EXECUTION_BASIS.md").read_text(encoding="utf-8").splitlines()[0]
    if basis_first != R2_BASIS_FIRST_LINE:
        raise RuntimeError("ROUND46-R2 execution basis first-line identity mismatch")
    if not READBACK_REGRESSION_PATH.is_file():
        raise FileNotFoundError("ROUND46-R2 real checkpoint-readback regression is missing")
    if not V5_TEXT_IDENTITY_REGRESSION_PATH.is_file():
        raise FileNotFoundError("ROUND46 V5 text identity regression is missing")
    v5_text_regression = _json(V5_TEXT_IDENTITY_REGRESSION_PATH)
    if (v5_text_regression.get("kind") != "round46-r2-v5-text-identity-regression/1"
            or v5_text_regression.get("status") != "passed"
            or v5_text_regression.get("runner_version") != ROUND46_RUNNER_VERSION
            or v5_text_regression.get("runner_source_sha256") != _sha256(Path(__file__))
            or any(v5_text_regression.get(key) != 0 for key in (
                "environment_constructor_calls", "run_invocation_calls", "normal_claims",
                "reserve_claims", "completed_steps", "ppo_updates",
                "optimizer_step_calls", "checkpoint_write_calls_this_regression"))):
        raise RuntimeError("ROUND46 V5 text identity regression receipt failed its source/zero-work gate")
    v5_artifacts = (
        FROZEN_CONFIG_PATH, CONFIG_DIFF_PATH, KEY_PLAN_PATH, TRAINING_VERSION_PATH,
        INPUT_IDENTITY_PATH, RUNTIME_FREEZE_PATH, PRELAUNCH_GATE_PATH,
        PRELAUNCH_SHA_PATH, V5_BINDING_REGRESSION_PATH, V5_STATUS_GUARD_REUSE_PATH,
        STOP_STATE_PATH,
    )
    if any(path.exists() for path in v5_artifacts):
        raise RuntimeError("ROUND46 V5 freeze/evidence already exists; refusing to overwrite it")
    if any(path.exists() for path in (
            V4_PRELAUNCH_GATE_PATH, V4_BINDING_REGRESSION_PATH,
            V4_STATUS_GUARD_REUSE_PATH)):
        raise RuntimeError("ROUND46 V4 unexpectedly has a gate or post-gate receipt; preserve and audit first")
    if list((V6_CHECKPOINTS_DIR).glob("v2_train_s*.pt")):
        raise RuntimeError("ROUND46-R2 will not freeze while a prior untrained checkpoint is active")

    if not all(path.is_file() for path in (
            V3_FROZEN_CONFIG_PATH, V3_INPUT_IDENTITY_PATH, V3_KEY_PLAN_PATH,
            V3_PRELAUNCH_GATE_PATH, V3_PRELAUNCH_SHA_PATH, V3_RUNTIME_FREEZE_PATH,
            STATUS_GUARD_REGRESSION_PATH, V3_STOP_ARCHIVE_PATH, V4_BINDING_AUDIT_PATH)):
        raise RuntimeError("ROUND46 V5 requires the complete committed V3 stop/audit lineage")
    v4_required_files = (
        V4_FROZEN_CONFIG_PATH, V4_CONFIG_DIFF_PATH, V4_KEY_PLAN_PATH,
        V4_TRAINING_VERSION_PATH, V4_INPUT_IDENTITY_PATH, V4_RUNTIME_FREEZE_PATH,
        V4_PREFLIGHT_FAILURE_PATH, V4_PREFLIGHT_TRACEBACK_PATH, V4_STOP_REPORT_PATH,
    )
    if not all(path.is_file() for path in v4_required_files):
        raise RuntimeError("ROUND46 V5 requires the complete preserved V4 stop/freeze evidence")
    v3_freeze = _json(V3_FROZEN_CONFIG_PATH)
    v3_gate = _json(V3_PRELAUNCH_GATE_PATH)
    v3_archive = _json(V3_STOP_ARCHIVE_PATH)
    binding_audit = _json(V4_BINDING_AUDIT_PATH)
    v4_freeze = _json(V4_FROZEN_CONFIG_PATH)
    v4_failure = _json(V4_PREFLIGHT_FAILURE_PATH)
    v4_hash_namespaces = v4_failure.get("hash_namespaces_at_failure", {})
    if (v3_freeze.get("runner_version", "").split("+")[0]
            != "round46-bc-initialized-constrained-ppo/3"
            or v3_gate.get("status") != "passed"
            or V3_PRELAUNCH_SHA_PATH.read_text(encoding="ascii").strip()
            != _sha256(V3_PRELAUNCH_GATE_PATH)
            or v3_gate.get("runner_source_sha256") != v3_freeze.get("runner_source_sha256")
            or v3_archive.get("status") != "hash_closed"
            or v3_archive.get("normal_claims") != 0
            or v3_archive.get("reserve_claims") != 0
            or v3_archive.get("active_checkpoint_files_remaining") != 0
            or v3_archive.get("active_progress_files_remaining") != 0
            or binding_audit.get("status") != "confirmed_misbound"
            or binding_audit.get("head") != "02fd982354de2cf20e5363c8c68f50d9899eb1e9"
            or v4_failure.get("kind") != "round46-r2-v4-preflight-stop/1"
            or v4_failure.get("status") != "stopped_before_claim"
            or v4_failure.get("v4_gate_created") is not False
            or v4_failure.get("v4_binding_regression_run") is not False
            or v4_failure.get("execute_invoked") is not False
            or v4_failure.get("counts_at_stop", {}).get("normal_claims") != 0
            or v4_failure.get("counts_at_stop", {}).get("reserve_claims") != 0
            or v4_failure.get("counts_at_stop", {}).get("environment_constructor_calls") != 0
            or v4_failure.get("counts_at_stop", {}).get("optimizer_step_calls") != 0
            or v4_hash_namespaces.get("v3_stop_archive_worktree_file_sha256")
            != "c66fb42fa8fc1e4513eb6a768b14ea24e231091da55c14e6c47b7ccadc0d327a"
            or v4_hash_namespaces.get("v3_stop_archive_git_blob_sha256")
            != "80438b581c28ccd13dd6f535af4a551f242da0f08c0356e6d76ce77e538b19aa"
            or v4_freeze.get("runner_source_sha256")
            != v4_failure.get("runner_source_sha256")):
        raise RuntimeError("ROUND46 V3 freeze/gate/archive/audit lineage failed")

    _checkpoint_journal_route_audit(create=False)
    input_identity = _input_identity()
    if input_identity != _json(V3_INPUT_IDENTITY_PATH):
        raise RuntimeError("ROUND46 V5 source inputs differ from the stopped V3 frozen inputs")
    _copy_runtime_aliases(input_identity)
    order = list(input_identity["scenario_order"])
    sequence_check = input_identity["scenario_order_verification"]
    key_plan = _logical_key_plan(order, sequence_check)
    _write_json(KEY_PLAN_PATH, key_plan)

    module = _configure_runtime_module()
    runtime = _RUNTIME
    assert runtime is not None
    if (len(runtime.get("r2_reference_cache", {})) != 64
            or runtime.get("r2_reference_cache_sha256") is None):
        raise RuntimeError("ROUND46 requires the actual fixed 64-row immediate reference cache")
    runtime_frozen = copy.deepcopy(runtime["frozen"])
    round42 = runtime_frozen["round42"]
    round42["training_version"] = TRAINING_VERSION
    round42["runner_version"] = ROUND46_RUNNER_VERSION
    freeze_identity = round42["freeze_identity"]
    freeze_identity["training_version"] = TRAINING_VERSION
    freeze_identity["runner_version"] = ROUND46_RUNNER_VERSION
    freeze_identity["execution_runner_version"] = ROUND46_RUNNER_VERSION
    freeze_body = dict(freeze_identity)
    freeze_body.pop("freeze_sha256", None)
    freeze_identity["freeze_sha256"] = _json_sha256(freeze_body, sort_keys=True)
    _write_json(RUNTIME_FREEZE_PATH, runtime_frozen)

    training_manifest = {
        "kind": "round46-r2-v5-training-version-manifest/1",
        "round": 46,
        "training_version": TRAINING_VERSION,
        "runner_version": ROUND46_RUNNER_VERSION,
        "semantic_version_reused_without_change": True,
        "initializer": "ROUND45 BC_POST_MODEL four modules, model-only, same seed",
        "optimizer": "fresh Adam, ten unique controller parameters, empty state",
        "dual_lambda_initial": 1.0,
        "r44_ep0_rng_only": True,
        "bc_optimizer_or_rng_restored": False,
        "round44_trained_weights_or_optimizer_restored": False,
        "gamma": float(runtime["common"].GAMMA),
        "gae_lambda": float(runtime["common"].GAE_LAMBDA),
        "eta_lambda": float(runtime["common"].ETA_LAMBDA),
        "b0_bytes": float(runtime["common"].B0_BYTES),
        "delta": float(runtime["common"].DELTA),
        "actor_advantage": "lambda_dual*A_task-A_positive_cost",
        "observation_width": int(runtime_frozen["round42"]["actor_feature_width"]),
        "row_filtering": runtime_frozen["round42"]["row_filtering"]["policy"],
        "action_semantics": "one-shot-send-skip/1",
        "reward_timing_version": "published-completion-event-at-relay-tick/2",
        "changed_scientific_factor": "model initialization only: rule-BC late147 to constrained PPO",
        "training_episodes": 384, "updates": 96,
    }
    _write_json(TRAINING_VERSION_PATH, training_manifest)
    _write_json(INPUT_IDENTITY_PATH, input_identity)
    runner_path = Path(__file__).resolve()
    runner_sha = _sha256(runner_path)
    text_identity_paths = (
        runner_path, READBACK_REGRESSION_PATH, V5_TEXT_IDENTITY_REGRESSION_PATH,
        REPO / ".gitattributes",
        V3_FROZEN_CONFIG_PATH, V3_CONFIG_DIFF_PATH, V3_INPUT_IDENTITY_PATH,
        V3_KEY_PLAN_PATH, V3_TRAINING_VERSION_PATH, V3_RUNTIME_FREEZE_PATH,
        V3_PRELAUNCH_GATE_PATH, V3_PRELAUNCH_SHA_PATH,
        READBACK_REGRESSION_RESULT_PATH, CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH,
        STATUS_GUARD_REGRESSION_PATH, V3_STOP_ARCHIVE_PATH,
        V4_BINDING_AUDIT_PATH, V4_FROZEN_CONFIG_PATH, V4_CONFIG_DIFF_PATH,
        V4_KEY_PLAN_PATH, V4_TRAINING_VERSION_PATH, V4_INPUT_IDENTITY_PATH,
        V4_RUNTIME_FREEZE_PATH, V4_PREFLIGHT_FAILURE_PATH,
        V4_PREFLIGHT_TRACEBACK_PATH, V4_STOP_REPORT_PATH,
    )
    v4_archived_raw_sha = v4_hash_namespaces[
        "v3_stop_archive_worktree_file_sha256"]
    v4_archived_blob_sha = v4_hash_namespaces[
        "v3_stop_archive_git_blob_sha256"]
    text_identity_manifest = {}
    for path in text_identity_paths:
        expected_raw = None
        expected_blob = None
        if path.resolve() == V3_STOP_ARCHIVE_PATH.resolve():
            expected_raw = v4_archived_raw_sha
            expected_blob = v4_archived_blob_sha
        elif path.resolve() == V4_FROZEN_CONFIG_PATH.resolve():
            expected_raw = v4_failure.get("freeze_config_sha256")
        elif path.resolve() == runner_path.resolve():
            expected_raw = runner_sha
        elif path.resolve() == READBACK_REGRESSION_PATH.resolve():
            expected_raw = v3_freeze.get("readback_regression_sha256")
        record = _capture_committed_text_identity(
            path, expected_raw, expected_blob)
        text_identity_manifest[record["path"]] = record
    runner_identity = text_identity_manifest[
        runner_path.relative_to(REPO).as_posix()]
    readback_identity = text_identity_manifest[
        READBACK_REGRESSION_PATH.relative_to(REPO).as_posix()]
    text_regression_identity = text_identity_manifest[
        V5_TEXT_IDENTITY_REGRESSION_PATH.relative_to(REPO).as_posix()]
    if (runner_identity["raw_file_sha256"] != runner_sha
            or readback_identity["raw_file_sha256"] != v3_freeze.get(
                "readback_regression_sha256")
            or v5_text_regression.get("runner_source_sha256") != runner_sha
            or text_regression_identity["raw_file_sha256"]
            != _sha256(V5_TEXT_IDENTITY_REGRESSION_PATH)):
        raise RuntimeError("ROUND46 V5 committed text identity manifest failed source binding")
    config_diff = {
        "kind": "round46-r2-v5-configuration-diff/1",
        "training_version": TRAINING_VERSION,
        "runner_version": ROUND46_RUNNER_VERSION,
        "source_round44_frozen_config_sha256": input_identity["round44_frozen_config_sha256"],
        "source_round45_post_bc_freeze_sha256": input_identity["round45_freeze_sha256"],
        "scientific_changes": ["initialization model: same-seed ROUND45 BC four-module weights"],
        "unchanged": {
            "training_scenes": "ROUND44 frozen 128-item order, elementwise verified for all seeds",
            "development_scenes": list(DEV_SCENES),
            "gamma": 1.0, "gae_lambda": 1.0, "eta_lambda": 0.5,
            "b0_bytes": 10000.0, "delta": 0.0, "lambda_initial": 1.0,
            "one_shot_action_semantics": "one-shot-send-skip/1",
            "dual_critics": True, "all_rows": True, "ack": False,
            "max_steps": 1000, "goal_count": 6, "communication_budget_bytes": 29300,
            "device": "cpu", "torch_threads": 1, "interop_threads": 1,
        },
        "counts": {
            "training_episodes": TRAINING_KEYS,
            "final_evaluation_cells": EVALUATION_KEYS,
            "normal_claims": NORMAL_CAP, "technical_reserve": RESERVE_CAP,
            "hard_cap": HARD_CAP,
        },
    }
    _write_json(CONFIG_DIFF_PATH, config_diff)
    frozen = {
        "kind": "round46-r2-v5-execution-freeze/1",
        "status": "prepared_before_environment",
        "round": 46, "branch": ROUND46_BRANCH,
        "start_head": ROUND46_START_HEAD, "prepare_head": _run_git("rev-parse", "HEAD"),
        "runner_version": ROUND46_RUNNER_VERSION,
        "runner_path": runner_path.relative_to(REPO).as_posix(),
        "runner_source_sha256": runner_sha,
        "readback_regression_path": READBACK_REGRESSION_PATH.relative_to(REPO).as_posix(),
        "readback_regression_sha256": _sha256(READBACK_REGRESSION_PATH),
        "text_identity_regression_path": V5_TEXT_IDENTITY_REGRESSION_PATH.relative_to(REPO).as_posix(),
        "text_identity_regression_sha256": _sha256(V5_TEXT_IDENTITY_REGRESSION_PATH),
        "text_file_identity_manifest": text_identity_manifest,
        "git_text_identity_policy": {
            "core_autocrlf": _run_git("config", "--get", "core.autocrlf"),
            "gitattributes_identity": text_identity_manifest[(REPO / ".gitattributes").relative_to(REPO).as_posix()],
            "allowed_worktree_to_blob_normalization": "CRLF to LF only",
            "raw_worktree_sha256_and_git_blob_content_sha256_are_separate_namespaces": True,
        },
        "training_version": TRAINING_VERSION,
        "predecessor_runner_version": v4_freeze["runner_version"],
        "predecessor_stop_commit": v4_failure["head_at_failure"],
        "v4_failure_path": V4_PREFLIGHT_FAILURE_PATH.relative_to(REPO).as_posix(),
        "v4_failure_sha256": _sha256(V4_PREFLIGHT_FAILURE_PATH),
        "v4_stop_report_sha256": _sha256(V4_STOP_REPORT_PATH),
        "v4_prelaunch_gate_absent": not V4_PRELAUNCH_GATE_PATH.exists(),
        "v4_binding_regression_absent": not V4_BINDING_REGRESSION_PATH.exists(),
        "v4_status_guard_reuse_absent": not V4_STATUS_GUARD_REUSE_PATH.exists(),
        "predecessor_stop_archive_sha256": _sha256(V3_STOP_ARCHIVE_PATH),
        "binding_audit_sha256": _sha256(V4_BINDING_AUDIT_PATH),
        "reused_preflight_lineage": {
            "v3_frozen_config_sha256": _sha256(V3_FROZEN_CONFIG_PATH),
            "v3_configuration_diff_sha256": _sha256(V3_CONFIG_DIFF_PATH),
            "v3_input_identity_sha256": _sha256(V3_INPUT_IDENTITY_PATH),
            "v3_key_plan_sha256": _sha256(V3_KEY_PLAN_PATH),
            "v3_training_version_manifest_sha256": _sha256(V3_TRAINING_VERSION_PATH),
            "v3_runtime_freeze_sha256": _sha256(V3_RUNTIME_FREEZE_PATH),
            "v3_prelaunch_gate_sha256": _sha256(V3_PRELAUNCH_GATE_PATH),
            "v3_prelaunch_sha_file_sha256": _sha256(V3_PRELAUNCH_SHA_PATH),
            "v3_readback_regression_sha256": _sha256(READBACK_REGRESSION_RESULT_PATH),
            "v3_checkpoint_journal_route_regression_sha256":
                _sha256(CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH),
            "v3_status_guard_regression_sha256":
                _sha256(STATUS_GUARD_REGRESSION_PATH),
            "v3_stop_archive_raw_file_sha256": v4_archived_raw_sha,
            "v3_stop_archive_git_blob_content_sha256": v4_archived_blob_sha,
            "v4_binding_audit_raw_file_sha256": _sha256(V4_BINDING_AUDIT_PATH),
            "v4_frozen_config_sha256": _sha256(V4_FROZEN_CONFIG_PATH),
            "v4_failure_sha256": _sha256(V4_PREFLIGHT_FAILURE_PATH),
            "v4_failure_traceback_sha256": _sha256(V4_PREFLIGHT_TRACEBACK_PATH),
            "v4_stop_report_sha256": _sha256(V4_STOP_REPORT_PATH),
            "text_identity_regression_sha256": _sha256(V5_TEXT_IDENTITY_REGRESSION_PATH),
            "old_receipts_reexecuted": False,
        },
        "training_version_manifest_sha256": _sha256(TRAINING_VERSION_PATH),
        "configuration_diff_sha256": _sha256(CONFIG_DIFF_PATH),
        "key_plan_sha256": _sha256(KEY_PLAN_PATH),
        "input_identity_sha256": _sha256(INPUT_IDENTITY_PATH),
        "runtime_freeze_sha256": _sha256(RUNTIME_FREEZE_PATH),
        "logical_key_plan": key_plan,
        "inputs": {
            key: value for key, value in input_identity.items()
            if key != "bc_models_by_seed" and key != "scenario_order"
        },
        "bc_models": {
            seed: {
                "checkpoint_path": item["checkpoint_path"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "parameter_sha256": item["parameter_sha256"],
                "module_parameter_sha256": item["module_parameter_sha256"],
                "readback_receipt_sha256": item["readback_receipt_sha256"],
                "metrics_sha256": item["metrics_sha256"],
                "journal_sha256": item["journal_sha256"],
            } for seed, item in input_identity["bc_models_by_seed"].items()
        },
        "registered_r44_ep0_rng": input_identity["round44_registered_ep0_rng"],
        "training_order": order,
        "sequence_order_check": sequence_check,
        "quota": {
            "normal": NORMAL_CAP, "technical_reserve": RESERVE_CAP,
            "hard_cap": HARD_CAP, "round45_claims_excluded": 48,
            "combined_family_cap": 552,
        },
        "execution_order": [
            "first 4 episodes for seeds 11, 22, 33",
            "run actual first-batch gate for seeds 11, 22, 33",
            "continue seeds 11, 22, 33 to fixed ep128",
            "run exactly 96 NEW ep128 greedy/sample evaluation cells",
        ],
        "prohibited": [
            "ROUND45 fixed-plan entrypoint", "BC optimizer or RNG restore",
            "R44 trained model or optimizer restore", "additional environment starts",
            "additional training", "tuning", "network push", "deployment",
        ],
    }
    _write_json(FROZEN_CONFIG_PATH, frozen)
    _write_json(CURRENT_TASK_PATH, {
        "kind": "round46-current-task/1", "stage": "prepared; zero environments",
        "branch": ROUND46_BRANCH, "head": _run_git("rev-parse", "HEAD"),
        "runner_version": ROUND46_RUNNER_VERSION, "training_version": TRAINING_VERSION,
        "normal_environment_keys": NORMAL_CAP,
        "technical_reserve_max": RESERVE_CAP, "hard_environment_cap": HARD_CAP,
        "environment_constructed": 0, "run_invocation_calls": 0,
        "completed_episodes": 0, "ppo_updates": 0, "optimizer_step_calls": 0,
    })
    ledger = runtime["r42_run"].ledger_snapshot(strict=True)
    if (int(ledger["total_used"]) != 0
            or int(ledger["counts"]["environment_constructor_calls"]) != 0
            or int(ledger["counts"]["ppo_update_calls"]) != 0
            or int(ledger["counts"]["optimizer_step_calls"]) != 0):
        raise RuntimeError(f"ROUND46 prepare unexpectedly consumed runtime work: {ledger['counts']}")
    print(json.dumps({
        "status": "prepared", "runner_sha256": runner_sha,
        "scenario_sequence": sequence_check,
        "normal_keys": NORMAL_CAP, "reserve": RESERVE_CAP, "hard_cap": HARD_CAP,
        "environment_constructed": 0, "ppo_updates": 0, "optimizer_step_calls": 0,
    }, ensure_ascii=False, indent=2))


def _module_digest(controller: Any, r45_bc: Any) -> dict[str, str]:
    result = {}
    for name, module in (
        ("trunk", controller.inner.trunk),
        ("action_head", controller.inner.action_head),
        ("task_value_head", controller.inner.value_head),
        ("cost_value_head", controller.cost_head),
    ):
        result[name] = r45_bc._module_hash(module)
    return result


def _load_readback_regression_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "round46_r2_checkpoint_readback_regression", READBACK_REGRESSION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load ROUND46-R2 checkpoint regression: {READBACK_REGRESSION_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_checkpoint_journal_route_regression(runtime: dict[str, Any]) -> dict[str, Any]:
    """Exercise one real trainer save and prove it appends only to ROUND46-R2."""
    path = R46_DIR / "test_fixtures" / "checkpoint_route" / "round46_route_probe_s11.pt"
    if CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH.is_file():
        result = _json(CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH)
        current_entries = [json.loads(line) for line in
                           CHECKPOINT_JOURNAL_PATH.read_text(encoding="utf-8").splitlines()
                           if line.strip()]
        matching_entries = [item for item in current_entries
                            if Path(item.get("path_at_write", "")).resolve() == path.resolve()]
        if (result.get("kind") != "round46-r2-checkpoint-journal-route-regression/1"
                or result.get("status") != "passed"
                or result.get("runner_version") != ROUND46_RUNNER_VERSION
                or result.get("training_version") != TRAINING_VERSION
                or not path.is_file()
                or _sha256(path) != result.get("checkpoint_sha256")
                or len(matching_entries) != 1
                or matching_entries[0].get("sha256_at_write") != result.get("checkpoint_sha256")
                or Path(runtime["r42_run"].RUN_DIR).resolve() != R46_DIR.resolve()):
            raise RuntimeError("existing checkpoint-journal route regression conflicts with this freeze")
        return result
    if path.exists():
        raise RuntimeError("route-probe checkpoint exists without its immutable regression receipt")

    trainer = runtime["trainer"]
    run_module = runtime["r42_run"]
    versions = runtime["r42_versions"]
    journal_path = CHECKPOINT_JOURNAL_PATH
    legacy_sha_before = _sha256(LEGACY_CHECKPOINT_JOURNAL_PATH)
    before = run_module.ledger_snapshot(strict=True)
    journal_before = journal_path.read_bytes()
    ambient_rng = trainer.rng_state()
    import torch

    controller = None
    optimizer = None
    try:
        controller = runtime["controller_module"].fresh_controller(SEEDS[0])
        optimizer = torch.optim.Adam(
            controller.parameters, lr=float(runtime["common"].LEARNING_RATE))
        state = trainer.fresh_training_state(SEEDS[0])
        expected_state = {
            "training_seed": SEEDS[0], "episodes_done": 0, "updates": 0,
            "optimizer_steps": 0, "training_version": TRAINING_VERSION,
            "comm_action_semantics": "one-shot-send-skip/1",
        }
        if any(state.get(key) != value for key, value in expected_state.items()):
            raise RuntimeError("route-probe state is not a fresh ep0 state")
        if optimizer.state_dict().get("state") != {}:
            raise RuntimeError("route-probe Adam must be fresh and empty")
        saved_path = trainer.save_checkpoint(
            controller, optimizer, state, path, freeze=runtime["round42_freeze"])
    finally:
        trainer.restore_rng(ambient_rng)
    assert controller is not None and optimizer is not None
    rng_preserved = runtime["module"]._same_value(ambient_rng, trainer.rng_state())
    if Path(saved_path).resolve() != path.resolve() or not path.is_file():
        raise RuntimeError("actual trainer save did not return the route-probe checkpoint path")
    after = run_module.ledger_snapshot(strict=True)
    journal_after = journal_path.read_bytes()
    new_entries = [json.loads(line.decode("utf-8")) for line in journal_after[len(journal_before):].splitlines()
                   if line.strip()]
    checkpoint_sha = _sha256(path)
    if (len(new_entries) != 1
            or Path(new_entries[0].get("path_at_write", "")).resolve() != path.resolve()
            or new_entries[0].get("sha256_at_write") != checkpoint_sha
            or after["counts"]["checkpoint_write_calls"]
            - before["counts"]["checkpoint_write_calls"] != 1
            or after["counts"]["checkpoint_write_journal_entries"]
            - before["counts"]["checkpoint_write_journal_entries"] != 1
            or not journal_after.startswith(journal_before)
            or not rng_preserved
            or optimizer.state_dict().get("state") != {}
            or _sha256(LEGACY_CHECKPOINT_JOURNAL_PATH) != legacy_sha_before
            or Path(run_module.RUN_DIR).resolve() != R46_DIR.resolve()
            or not Path(path).resolve().is_relative_to(R46_DIR.resolve())):
        raise RuntimeError("actual checkpoint writer escaped ROUND46-R2 or changed runtime state")
    for key in ("normal_used", "reserve_used", "total_used", "environment_constructor_calls",
                "run_invocation_calls", "completed_steps", "ppo_update_calls",
                "optimizer_step_calls"):
        before_value = before.get(key, before.get("counts", {}).get(key))
        after_value = after.get(key, after.get("counts", {}).get(key))
        if before_value != after_value:
            raise RuntimeError(f"route-probe checkpoint save changed execution count {key}")

    result = {
        "kind": "round46-r2-checkpoint-journal-route-regression/1",
        "status": "passed",
        "runner_version": ROUND46_RUNNER_VERSION,
        "training_version": TRAINING_VERSION,
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "r42_run_output_dir": str(Path(run_module.RUN_DIR).resolve()),
        "journal_path": str(journal_path.resolve()),
        "journal_entry_count_before": before["counts"]["checkpoint_write_journal_entries"],
        "journal_entry_count_after": after["counts"]["checkpoint_write_journal_entries"],
        "trainer_save_calls_delta": 1,
        "journal_entries_delta": 1,
        "journal_last_entry": new_entries[0],
        "legacy_round45_journal_path": str(LEGACY_CHECKPOINT_JOURNAL_PATH.resolve()),
        "legacy_round45_journal_sha256_unchanged": _sha256(LEGACY_CHECKPOINT_JOURNAL_PATH),
        "ambient_rng_preserved": rng_preserved,
        "optimizer_state_empty": optimizer.state_dict().get("state") == {},
        "claims": after["normal_used"] + after["reserve_used"],
        "environment_constructor_calls": after["counts"]["environment_constructor_calls"],
        "run_invocation_calls": after["counts"]["run_invocation_calls"],
        "completed_steps": after["counts"]["completed_steps"],
        "ppo_updates": after["counts"]["ppo_update_calls"],
        "optimizer_step_calls": after["counts"]["optimizer_step_calls"],
    }
    _write_json(CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH, result, immutable=True)
    return result


def _preflight_reuse_v3_zero_environment_evidence() -> dict[str, Any]:
    """Freeze V5 using exact prior receipts without replaying old fixtures."""
    _assert_branch_head()
    _assert_source_worktree_clean()
    if PRELAUNCH_GATE_PATH.exists() or PRELAUNCH_SHA_PATH.exists():
        raise RuntimeError("ROUND46 V5 preflight receipt already exists; refusing to rewrite it")
    if not FROZEN_CONFIG_PATH.is_file() or not KEY_PLAN_PATH.is_file():
        raise RuntimeError("ROUND46 V5 prepare/freeze files are missing")
    frozen = _json(FROZEN_CONFIG_PATH)
    expected_runner_sha = _sha256(Path(__file__))
    if (frozen.get("status") != "prepared_before_environment"
            or frozen.get("runner_version") != ROUND46_RUNNER_VERSION
            or frozen.get("training_version") != TRAINING_VERSION
            or frozen.get("runner_source_sha256") != expected_runner_sha
            or frozen.get("readback_regression_sha256") != _sha256(READBACK_REGRESSION_PATH)
            or frozen.get("text_identity_regression_sha256")
            != _sha256(V5_TEXT_IDENTITY_REGRESSION_PATH)
            or _sha256(KEY_PLAN_PATH) != frozen.get("key_plan_sha256")
            or _sha256(CONFIG_DIFF_PATH) != frozen.get("configuration_diff_sha256")
            or _sha256(TRAINING_VERSION_PATH) != frozen.get("training_version_manifest_sha256")
            or _sha256(INPUT_IDENTITY_PATH) != frozen.get("input_identity_sha256")
            or _sha256(RUNTIME_FREEZE_PATH) != frozen.get("runtime_freeze_sha256")):
        raise RuntimeError("ROUND46 V5 frozen runner/config/key/input identities failed")
    if (_run_git("rev-parse", "HEAD") == frozen.get("prepare_head")
            or _run_git("rev-parse", "HEAD^") != frozen.get("prepare_head")):
        raise RuntimeError("ROUND46 V5 source/config freeze must be one local commit before preflight")
    runner_relative = Path(__file__).resolve().relative_to(REPO).as_posix()
    text_manifest = frozen.get("text_file_identity_manifest", {})
    for relative, record in text_manifest.items():
        if not isinstance(record, dict) or record.get("path") != relative:
            raise RuntimeError(f"ROUND46 V5 frozen text identity record is malformed: {relative}")
        identity = _require_committed_file(
            REPO / Path(relative), record.get("raw_file_sha256", ""),
            record.get("git_blob_content_sha256", ""))
        if (identity.get("comparison_rule") != record.get("comparison_rule")
                or identity.get("raw_file_sha256") != record.get("raw_file_sha256")
                or identity.get("git_blob_content_sha256")
                != record.get("git_blob_content_sha256")):
            raise RuntimeError(f"ROUND46 V5 frozen text namespace identity changed: {relative}")
    if runner_relative not in text_manifest:
        raise RuntimeError("ROUND46 V5 runner is absent from its frozen text identity manifest")
    if V4_PRELAUNCH_GATE_PATH.exists() or V4_BINDING_REGRESSION_PATH.exists() \
            or V4_STATUS_GUARD_REUSE_PATH.exists():
        raise RuntimeError("ROUND46 V4 stop acquired a gate or post-gate receipt after its archive")

    lineage = frozen.get("reused_preflight_lineage", {})
    lineage_paths = (
        (V3_FROZEN_CONFIG_PATH, "v3_frozen_config_sha256"),
        (V3_CONFIG_DIFF_PATH, "v3_configuration_diff_sha256"),
        (V3_INPUT_IDENTITY_PATH, "v3_input_identity_sha256"),
        (V3_KEY_PLAN_PATH, "v3_key_plan_sha256"),
        (V3_TRAINING_VERSION_PATH, "v3_training_version_manifest_sha256"),
        (V3_RUNTIME_FREEZE_PATH, "v3_runtime_freeze_sha256"),
        (V3_PRELAUNCH_GATE_PATH, "v3_prelaunch_gate_sha256"),
        (V3_PRELAUNCH_SHA_PATH, "v3_prelaunch_sha_file_sha256"),
        (READBACK_REGRESSION_RESULT_PATH, "v3_readback_regression_sha256"),
        (CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH,
         "v3_checkpoint_journal_route_regression_sha256"),
        (STATUS_GUARD_REGRESSION_PATH, "v3_status_guard_regression_sha256"),
        (V3_STOP_ARCHIVE_PATH, "v3_stop_archive_raw_file_sha256"),
        (V4_BINDING_AUDIT_PATH, "v4_binding_audit_raw_file_sha256"),
    )
    for path, key in lineage_paths:
        expected = lineage.get(key)
        if expected is None:
            raise RuntimeError(f"ROUND46 V5 prior evidence hash is missing: {path.name}")
        identity = _frozen_text_identity(path)
        if identity.get("raw_file_sha256") != expected:
            raise RuntimeError(f"ROUND46 V5 prior evidence raw-file identity changed: {path.name}")
    v3_freeze = _json(V3_FROZEN_CONFIG_PATH)
    v3_gate = _json(V3_PRELAUNCH_GATE_PATH)
    v3_identity = _json(V3_INPUT_IDENTITY_PATH)
    v3_archive = _json(V3_STOP_ARCHIVE_PATH)
    binding_audit = _json(V4_BINDING_AUDIT_PATH)
    v3_gate_sha = _sha256(V3_PRELAUNCH_GATE_PATH)
    if (V3_PRELAUNCH_SHA_PATH.read_text(encoding="ascii").strip() != v3_gate_sha
            or v3_gate.get("kind") != "round46-zero-environment-preflight/1"
            or v3_gate.get("status") != "passed"
            or v3_gate.get("runner_source_sha256") != v3_freeze.get("runner_source_sha256")
            or v3_gate.get("training_version") != TRAINING_VERSION
            or v3_archive.get("status") != "hash_closed"
            or v3_archive.get("normal_claims") != 0
            or v3_archive.get("reserve_claims") != 0
            or v3_archive.get("environment_constructor_calls") != 0
            or v3_archive.get("run_invocation_calls") != 0
            or v3_archive.get("completed_steps") != 0
            or v3_archive.get("ppo_updates") != 0
            or v3_archive.get("optimizer_steps") != 0
            or v3_archive.get("checkpoint_write_calls") != 18
            or binding_audit.get("status") != "confirmed_misbound"):
        raise RuntimeError("ROUND46 V3 prior preflight and stopped-attempt archive failed")
    if _sha256(V3_STOP_ARCHIVE_PATH) != frozen.get("predecessor_stop_archive_sha256"):
        raise RuntimeError("ROUND46 V5 freeze no longer identifies the preserved V3 stop archive")
    if _sha256(V4_BINDING_AUDIT_PATH) != frozen.get("binding_audit_sha256"):
        raise RuntimeError("ROUND46 V5 freeze no longer identifies its binding audit")
    if lineage.get("old_receipts_reexecuted") is not False:
        raise RuntimeError("ROUND46 V5 freeze does not preserve the reuse-only gate decision")

    actual_inputs = _input_identity()
    if actual_inputs != _json(INPUT_IDENTITY_PATH) or actual_inputs != v3_identity:
        raise RuntimeError("ROUND46 V5 live source inputs differ from V3 or the V4 freeze")
    if _sha256(R45_CONTROL_GATE_PATH) != actual_inputs["round45_control_gate_sha256"]:
        raise RuntimeError("ROUND45 original failed gate changed before ROUND46 V5 preflight")
    v3_plan = _json(V3_KEY_PLAN_PATH)
    v5_plan = _json(KEY_PLAN_PATH)
    if (v5_plan.get("logical_jobs") != v3_plan.get("logical_jobs")
            or v5_plan.get("training_scenario_order") != v3_plan.get("training_scenario_order")
            or v5_plan.get("final_evaluation_keys") != v3_plan.get("final_evaluation_keys")):
        raise RuntimeError("ROUND46 V5 key plan differs from the fixed stopped V3 plan")

    old_readback = v3_gate.get("checkpoint_readback_regression", {})
    readback = _json(READBACK_REGRESSION_RESULT_PATH)
    old_route = v3_gate.get("checkpoint_journal_route_regression", {})
    route = _json(CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH)
    if (old_readback.get("status") != "passed"
            or old_readback.get("source_sha256") != _sha256(READBACK_REGRESSION_PATH)
            or old_readback.get("receipt_sha256") != _sha256(READBACK_REGRESSION_RESULT_PATH)
            or readback.get("kind") != "round46-r2-checkpoint-readback-regression/1"
            or readback.get("status") != "passed"
            or readback.get("test_source_sha256") != _sha256(READBACK_REGRESSION_PATH)
            or int(readback.get("synthetic_fixture_trainer_save_calls_this_preflight", -1)) != 0
            or int(readback.get("environment_constructor_calls", -1)) != 0
            or int(readback.get("ppo_update_calls", -1)) != 0
            or int(readback.get("optimizer_step_calls", -1)) != 0):
        raise RuntimeError("ROUND46 V3 checkpoint-readback regression is not reusable unchanged")
    if (old_route.get("status") != "passed"
            or old_route.get("receipt_sha256") != _sha256(CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH)
            or route.get("kind") != "round46-r2-checkpoint-journal-route-regression/1"
            or route.get("status") != "passed"
            or route.get("trainer_save_calls_delta") != 1
            or route.get("claims") != 0
            or route.get("environment_constructor_calls") != 0
            or route.get("run_invocation_calls") != 0
            or route.get("optimizer_step_calls") != 0
            or route.get("ambient_rng_preserved") is not True):
        raise RuntimeError("ROUND46 V3 checkpoint-journal routing regression is not reusable unchanged")
    if (_sha256(READBACK_REGRESSION_PATH) != v3_freeze.get("readback_regression_sha256")
            or _sha256(READBACK_REGRESSION_RESULT_PATH) != old_readback.get("receipt_sha256")):
        raise RuntimeError("ROUND46 V3 readback source/result do not match their committed freeze")

    module = _configure_runtime_module()
    runtime = _RUNTIME
    assert runtime is not None
    if (len(runtime.get("r2_reference_cache", {})) != 64
            or runtime.get("r2_reference_cache_sha256") is None
            or runtime["r42_versions"].TRAINING_VERSION != TRAINING_VERSION):
        raise RuntimeError("ROUND46 V5 actual runtime/reference cache does not match its freeze")
    if list((V6_CHECKPOINTS_DIR).glob("v2_train_s*.pt")):
        raise RuntimeError("ROUND46 V5 found an active V3 checkpoint; refusing to resume it")
    if list((V6_PROGRESS_DIR).glob("*.json")):
        raise RuntimeError("ROUND46 V5 found active V3 progress; refusing to resume it")
    before = runtime["r42_run"].ledger_snapshot(strict=True)
    counts = before["counts"]
    journal_entries = len(CHECKPOINT_JOURNAL_PATH.read_text(encoding="utf-8").splitlines())
    archive_journal = _json(V3_STOP_ARCHIVE_PATH)
    journal_snapshots = [item for item in archive_journal.get("archived_files", [])
                         if item.get("role") == "checkpoint_write_journal_snapshot"]
    if len(journal_snapshots) != 1:
        raise RuntimeError("ROUND46 V3 stop archive does not identify exactly one journal snapshot")
    journal_snapshot = journal_snapshots[0]
    archived_journal_path = Path(journal_snapshot["archive"])
    if (int(before.get("normal_used", -1)) != 0
            or int(before.get("reserve_used", -1)) != 0
            or int(before.get("total_used", -1)) != 0
            or int(counts.get("environment_constructor_calls", -1)) != 0
            or int(counts.get("run_invocation_calls", -1)) != 0
            or int(counts.get("completed_steps", -1)) != 0
            or int(counts.get("ppo_update_calls", -1)) != 0
            or int(counts.get("optimizer_step_calls", -1)) != 0
            or int(counts.get("checkpoint_write_calls", -1)) != 18
            or int(counts.get("checkpoint_write_journal_entries", -1)) != 18
            or journal_entries != 18
            or archive_journal.get("checkpoint_write_journal_entries") != 18
            or not archived_journal_path.is_file()
            or _sha256(archived_journal_path) != journal_snapshot.get("sha256")
            or _sha256(CHECKPOINT_JOURNAL_PATH) != journal_snapshot.get("sha256")):
        raise RuntimeError("ROUND46 V5 actual ledger no longer matches the preserved zero-claim stop")
    committed_text_identity_checks: dict[str, dict[str, str]] = {}
    for relative, expected in (
        (frozen["runner_path"], frozen["runner_source_sha256"]),
        (frozen["readback_regression_path"], frozen["readback_regression_sha256"]),
        (V3_PRELAUNCH_GATE_PATH.relative_to(REPO).as_posix(), v3_gate_sha),
    ):
        identity = _frozen_text_identity(REPO / Path(relative))
        if identity["raw_file_sha256"] != expected:
            raise RuntimeError(f"ROUND46 V5 inherited committed source identity differs: {relative}")
        committed_text_identity_checks[relative] = identity
    for path, expected in (
        (FROZEN_CONFIG_PATH, _sha256(FROZEN_CONFIG_PATH)),
        (CONFIG_DIFF_PATH, frozen["configuration_diff_sha256"]),
        (KEY_PLAN_PATH, frozen["key_plan_sha256"]),
        (TRAINING_VERSION_PATH, frozen["training_version_manifest_sha256"]),
        (INPUT_IDENTITY_PATH, frozen["input_identity_sha256"]),
        (RUNTIME_FREEZE_PATH, frozen["runtime_freeze_sha256"]),
    ):
        identity = _require_committed_file(path, expected)
        if identity["raw_file_sha256"] != expected:
            raise RuntimeError(f"ROUND46 V5 committed freeze artifact differs: {path.name}")
        committed_text_identity_checks[path.resolve().relative_to(REPO).as_posix()] = identity

    import torch
    if (str(runtime["common"].DEVICE) != "cpu"
            or torch.get_num_threads() != 1
            or torch.get_num_interop_threads() != 1):
        raise RuntimeError("ROUND46 V5 runtime device/thread boundary changed")
    old_models = v3_gate.get("bc_model_only_checks", [])
    if (len(old_models) != 3
            or {int(item.get("seed", -1)) for item in old_models} != set(SEEDS)
            or any(item.get("fresh_adam_state_empty") is not True
                   or item.get("optimizer_parameter_ids_unique") is not True
                   or item.get("bc_load_preserved_ambient_rng") is not True
                   for item in old_models)):
        raise RuntimeError("ROUND46 V3 four-module BC/fresh-Adam checks no longer close")

    after = runtime["r42_run"].ledger_snapshot(strict=True)
    if before != after:
        raise RuntimeError("ROUND46 V5 receipt construction changed the real ledger")
    receipt = {
        "kind": "round46-r2-v5-zero-environment-preflight/1",
        "status": "passed", "round": 46, "branch": ROUND46_BRANCH,
        "head_at_preflight": _run_git("rev-parse", "HEAD"),
        "runner_version": ROUND46_RUNNER_VERSION,
        "runner_path": runner_relative,
        "runner_source_sha256": expected_runner_sha,
        "text_identity_regression_sha256": _sha256(V5_TEXT_IDENTITY_REGRESSION_PATH),
        "text_identity_regression_status": _json(V5_TEXT_IDENTITY_REGRESSION_PATH)["status"],
        "text_file_identity_manifest": text_manifest,
        "committed_text_identity_checks": committed_text_identity_checks,
        "v4_stop_preserved": {
            "failure_sha256": _sha256(V4_PREFLIGHT_FAILURE_PATH),
            "stop_report_sha256": _sha256(V4_STOP_REPORT_PATH),
            "status": _json(V4_PREFLIGHT_FAILURE_PATH)["status"],
            "claims": 0,
            "environment_constructor_calls": 0,
            "optimizer_step_calls": 0,
            "v4_prelaunch_gate_absent": not V4_PRELAUNCH_GATE_PATH.exists(),
            "v4_binding_regression_absent": not V4_BINDING_REGRESSION_PATH.exists(),
        },
        "frozen_config_sha256": _sha256(FROZEN_CONFIG_PATH),
        "key_plan_sha256": _sha256(KEY_PLAN_PATH),
        "input_identity_sha256": _sha256(INPUT_IDENTITY_PATH),
        "training_version": TRAINING_VERSION,
        "normal_cap": NORMAL_CAP, "technical_reserve_max": RESERVE_CAP,
        "hard_cap": HARD_CAP,
        "round45_failed_gate_sha256_unchanged": _sha256(R45_CONTROL_GATE_PATH),
        "round45_failed_gate_status_preserved": True,
        "round44_scenario_sequence_all_seeds_exact": True,
        "scenario_order_check": v5_plan["scenario_sequence_verification"],
        "bc_model_only_checks_reused_from_v3": old_models,
        "registered_r44_ep0_rng_reused_from_v3": v3_gate.get("registered_r44_ep0_rng"),
        "checkpoint_readback_regression": {
            "kind": readback["kind"], "status": readback["status"],
            "source_sha256": readback["test_source_sha256"],
            "receipt_sha256": _sha256(READBACK_REGRESSION_RESULT_PATH),
            "legacy_fresh_failures_reproduced": readback["legacy_fresh_failures_reproduced"],
            "legacy_resume_failures_reproduced": readback["legacy_resume_failures_reproduced"],
            "fixed_fresh_seeds_passed": readback["fixed_fresh_seeds_passed"],
            "fixed_resume_seeds_passed": readback["fixed_resume_seeds_passed"],
            "synthetic_fixture_trainer_save_calls_this_preflight": 0,
            "environment_constructor_calls": 0,
            "ppo_update_calls": 0, "optimizer_step_calls": 0,
            "reused_prior_receipt": True,
        },
        "checkpoint_journal_route_regression": {
            "kind": route["kind"], "status": route["status"],
            "receipt_path": CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH.relative_to(REPO).as_posix(),
            "receipt_sha256": _sha256(CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH),
            "checkpoint_sha256": route["checkpoint_sha256"],
            "trainer_save_calls_delta": route["trainer_save_calls_delta"],
            "claims": 0, "environment_constructor_calls": 0,
            "optimizer_step_calls": 0, "reused_prior_receipt": True,
        },
        "reused_prelaunch_lineage": lineage,
        "actual_input_identity_rechecked": True,
        "active_prior_checkpoint_count": 0,
        "active_prior_progress_count": 0,
        "checkpoint_write_accounting": {
            "cumulative_checkpoint_write_calls": int(counts["checkpoint_write_calls"]),
            "v3_preflight_gate_checkpoint_write_calls": int(
                v3_gate.get("checkpoint_write_calls", v3_gate.get("after_counts", {}).get(
                    "checkpoint_write_calls", -1))),
            "v3_failed_execute_untrained_ep0_saves": 2,
            "new_v5_preflight_checkpoint_writes": 0,
            "legacy_preflight_and_fixture_tests_reexecuted": False,
        },
        "frozen_runtime_source_sha256": module._source_hashes(),
        "device": str(runtime["common"].DEVICE),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "before_counts": before["counts"], "after_counts": after["counts"],
        "environment_constructed": 0, "run_invocation_calls": 0,
        "completed_episodes": 0, "ppo_updates": 0,
        "optimizer_step_calls": 0,
        "checkpoint_write_calls": int(counts["checkpoint_write_calls"]),
        "claims": 0, "network_or_push": False,
    }
    _write_json(PRELAUNCH_GATE_PATH, receipt, immutable=True)
    _write_text_lf(PRELAUNCH_SHA_PATH, _sha256(PRELAUNCH_GATE_PATH) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def preflight() -> None:
    _preflight_reuse_v3_zero_environment_evidence()
    return
    _assert_branch_head()
    _assert_source_worktree_clean()
    if not FROZEN_CONFIG_PATH.is_file() or not KEY_PLAN_PATH.is_file():
        raise RuntimeError("ROUND46 prepare/freeze files are missing")
    frozen = _json(FROZEN_CONFIG_PATH)
    if (frozen.get("status") != "prepared_before_environment"
            or frozen.get("runner_source_sha256") != _sha256(Path(__file__))
            or frozen.get("readback_regression_sha256") != _sha256(READBACK_REGRESSION_PATH)
            or _sha256(KEY_PLAN_PATH) != frozen.get("key_plan_sha256")
            or _sha256(CONFIG_DIFF_PATH) != frozen.get("configuration_diff_sha256")
            or _sha256(TRAINING_VERSION_PATH) != frozen.get("training_version_manifest_sha256")
            or _sha256(INPUT_IDENTITY_PATH) != frozen.get("input_identity_sha256")):
        raise RuntimeError("ROUND46 frozen runner or logical key plan identity failed")
    current_head = _run_git("rev-parse", "HEAD")
    if (current_head == frozen.get("prepare_head")
            or _run_git("rev-parse", "HEAD^") != frozen.get("prepare_head")):
        raise RuntimeError("ROUND46 code/config freeze must be a single local commit before preflight")
    _require_committed_file(Path(__file__), frozen["runner_source_sha256"])
    _require_committed_file(READBACK_REGRESSION_PATH, frozen["readback_regression_sha256"])

    module = _configure_runtime_module()
    runtime = _RUNTIME
    assert runtime is not None
    runtime["module"] = module
    checkpoint_journal_route_regression = _run_checkpoint_journal_route_regression(runtime)
    actual_inputs = _input_identity()
    if actual_inputs != _json(INPUT_IDENTITY_PATH):
        raise RuntimeError("ROUND46 source inputs changed after the preparation freeze")
    import torch
    import r45_bc
    before_readback_regression = runtime["r42_run"].ledger_snapshot(strict=True)
    regression_module = _load_readback_regression_module()
    readback_regression = regression_module.run_preflight_regression(
        sys.modules[__name__], module, runtime, write_result=True)
    after_readback_regression = runtime["r42_run"].ledger_snapshot(strict=True)
    readback_result_sha256 = _sha256(READBACK_REGRESSION_RESULT_PATH)
    if (readback_regression.get("kind") != "round46-r2-checkpoint-readback-regression/1"
            or readback_regression.get("status") != "passed"
            or readback_regression.get("test_source_sha256") != frozen["readback_regression_sha256"]
            or after_readback_regression["total_used"] != before_readback_regression["total_used"]
            or any(after_readback_regression["counts"].get(key)
                   != before_readback_regression["counts"].get(key)
                   for key in ("environment_constructor_calls", "run_invocation_calls",
                               "completed_steps", "ppo_update_calls", "optimizer_step_calls"))
            or int(after_readback_regression["counts"]["checkpoint_write_calls"])
            - int(before_readback_regression["counts"]["checkpoint_write_calls"])
            != int(readback_regression["synthetic_fixture_trainer_save_calls_this_preflight"])):
        raise RuntimeError("ROUND46-R2 checkpoint readback regression identity or zero-side-effect check failed")
    prior_r2_attempt = actual_inputs["prior_round46_r2_preclaim_attempt"]
    prior_preclaim_writes = int(
        prior_r2_attempt["counts"]["preclaim_training_checkpoint_write_calls"])
    prior_synthetic_fixture_writes = int(
        prior_r2_attempt["counts"]["synthetic_fixture_writes"])
    synthetic_fixture_writes = int(readback_regression[
        "synthetic_fixture_trainer_save_calls_total"])
    journal_route = _checkpoint_journal_route_audit(create=False)
    cumulative_checkpoint_writes = int(after_readback_regression["counts"][
        "checkpoint_write_calls"])
    expected_historical_checkpoint_writes = (
        prior_preclaim_writes + prior_synthetic_fixture_writes
        + int(journal_route["counts_by_category"]["partial_v3_development_probe_fixture"])
        + int(journal_route["counts_by_category"]["v3_synthetic_fixture"])
    )
    new_preflight_checkpoint_writes = int(readback_regression[
        "synthetic_fixture_trainer_save_calls_this_preflight"])
    route_probe_writes = int(checkpoint_journal_route_regression["trainer_save_calls_delta"])
    if (int(journal_route["legacy_entry_count"]) != expected_historical_checkpoint_writes
            or cumulative_checkpoint_writes != int(journal_route["legacy_entry_count"])
            + route_probe_writes
            + new_preflight_checkpoint_writes):
        raise RuntimeError(
            "ROUND46 checkpoint write ledger does not close across the preserved v2/v3 attempts "
            "and current preflight")
    before_ledger = runtime["r42_run"].ledger_snapshot(strict=True)
    optimizer_steps_before = int(runtime["r42_run"]._OPTIMIZER_STEP_CALLS)
    ppo_updates_before = int(runtime["r42_run"]._PPO_UPDATE_CALLS)
    source_freeze = _json(R45_FREEZE_PATH)
    model_checks = []
    for seed in SEEDS:
        model = next(item for item in source_freeze["bc_models"]
                     if int(item["seed"]) == seed)
        payload = r45_bc.read_bc_checkpoint(Path(model["checkpoint_path"]))
        controller = runtime["controller_module"].fresh_controller(seed)
        rng_before = runtime["trainer"].rng_state()
        r45_bc.load_bc_controller_state(controller, payload)
        rng_after = runtime["trainer"].rng_state()
        module_hashes = _module_digest(controller, r45_bc)
        parameter_sha = controller.parameter_summary()["parameter_sha256"]
        ids = [id(parameter) for parameter in controller.parameters]
        optimizer = torch.optim.Adam(controller.parameters,
                                     lr=runtime["common"].LEARNING_RATE)
        item = {
            "seed": seed,
            "checkpoint_sha256": _sha256(Path(model["checkpoint_path"])),
            "expected_checkpoint_sha256": model["checkpoint_sha256"],
            "parameter_sha256": parameter_sha,
            "expected_parameter_sha256": model["parameter_sha256"],
            "module_parameter_sha256": module_hashes,
            "expected_module_parameter_sha256": model["module_parameter_sha256"],
            "bc_load_preserved_ambient_rng": module._same_value(rng_before, rng_after),
            "bc_optimizer_or_rng_restored": False,
            "round44_trained_weights_loaded": False,
            "fresh_adam_state_empty": optimizer.state_dict()["state"] == {},
            "optimizer_parameter_count": len(ids),
            "optimizer_parameter_ids_unique": len(ids) == len(set(ids)) == 10,
            "optimizer_step_calls_delta": int(runtime["r42_run"]._OPTIMIZER_STEP_CALLS)
            - optimizer_steps_before,
            "ppo_update_calls_delta": int(runtime["r42_run"]._PPO_UPDATE_CALLS)
            - ppo_updates_before,
        }
        expected_modules = model["module_parameter_sha256"]
        if (item["checkpoint_sha256"] != item["expected_checkpoint_sha256"]
                or parameter_sha != model["parameter_sha256"]
                or module_hashes != expected_modules
                or not item["bc_load_preserved_ambient_rng"]
                or not item["fresh_adam_state_empty"]
                or not item["optimizer_parameter_ids_unique"]):
            raise RuntimeError(f"ROUND46 BC model-only / fresh Adam preflight failed: {item}")
        model_checks.append(item)
        del optimizer, controller, payload

    registered_rng = {}
    for seed in SEEDS:
        source = runtime["frozen"]["round43"]["registered_ep0_inputs"][str(seed)]
        path = Path(source["path"])
        if _sha256(path) != source["file_sha256"]:
            raise RuntimeError(f"ROUND46 registered R44 ep0 RNG identity failed for seed {seed}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload.get("rng"), dict):
            raise RuntimeError(f"ROUND46 registered ep0 checkpoint lacks RNG bundle for seed {seed}")
        registered_rng[str(seed)] = {
            "path": str(path), "sha256": _sha256(path),
            "rng_keys": sorted(payload["rng"]),
            "registered_parameter_sha256": source.get("parameter_sha256"),
        }
        del payload

    if (int(runtime["r42_run"]._OPTIMIZER_STEP_CALLS) != optimizer_steps_before
            or int(runtime["r42_run"]._PPO_UPDATE_CALLS) != ppo_updates_before):
        raise RuntimeError("ROUND46 zero-environment preflight changed optimizer/PPO counters")
    after_ledger = runtime["r42_run"].ledger_snapshot(strict=True)
    if (after_ledger["counts"]["environment_constructor_calls"] != 0
            or after_ledger["counts"]["run_invocation_calls"] != 0
            or after_ledger["counts"]["ppo_update_calls"] != 0
            or after_ledger["counts"]["optimizer_step_calls"] != 0
            or after_ledger["total_used"] != 0):
        raise RuntimeError("ROUND46 zero-environment preflight changed the execution ledger")
    old_gate_sha = _sha256(R45_CONTROL_GATE_PATH)
    if old_gate_sha != frozen["inputs"]["round45_control_gate_sha256"]:
        raise RuntimeError("ROUND45 failed BC-only gate changed during ROUND46 preflight")

    checkpoint_write_accounting = {
        "cumulative_checkpoint_write_calls": cumulative_checkpoint_writes,
        "v2_synthetic_readback_fixture_writes": prior_synthetic_fixture_writes,
        "archived_v2_preclaim_training_writes": prior_preclaim_writes,
        "partial_v3_development_probe_fixture_writes": int(
            journal_route["counts_by_category"]["partial_v3_development_probe_fixture"]),
        "v3_synthetic_readback_fixture_writes": synthetic_fixture_writes,
        "checkpoint_journal_route_probe_writes": route_probe_writes,
        "new_preflight_checkpoint_writes": new_preflight_checkpoint_writes,
        "unclassified_checkpoint_writes": 0,
    }
    receipt = {
        "kind": "round46-zero-environment-preflight/1",
        "status": "passed", "round": 46, "branch": ROUND46_BRANCH,
        "head_at_preflight": _run_git("rev-parse", "HEAD"),
        "runner_version": ROUND46_RUNNER_VERSION,
        "runner_path": Path(__file__).resolve().relative_to(REPO).as_posix(),
        "runner_source_sha256": _sha256(Path(__file__)),
        "frozen_config_sha256": _sha256(FROZEN_CONFIG_PATH),
        "key_plan_sha256": _sha256(KEY_PLAN_PATH),
        "input_identity_sha256": _sha256(INPUT_IDENTITY_PATH),
        "training_version": TRAINING_VERSION,
        "normal_cap": NORMAL_CAP, "technical_reserve_max": RESERVE_CAP,
        "hard_cap": HARD_CAP,
        "round45_failed_gate_sha256_unchanged": old_gate_sha,
        "round45_failed_gate_status_preserved": True,
        "round44_scenario_sequence_all_seeds_exact": True,
        "scenario_order_check": _json(KEY_PLAN_PATH)["scenario_sequence_verification"],
        "bc_model_only_checks": model_checks,
        "registered_r44_ep0_rng": registered_rng,
        "checkpoint_readback_regression": {
            "kind": readback_regression["kind"],
            "status": readback_regression["status"],
            "source_sha256": readback_regression["test_source_sha256"],
            "receipt_sha256": readback_result_sha256,
            "legacy_fresh_failures_reproduced": readback_regression[
                "legacy_fresh_failures_reproduced"],
            "legacy_resume_failures_reproduced": readback_regression[
                "legacy_resume_failures_reproduced"],
            "fixed_fresh_seeds_passed": readback_regression["fixed_fresh_seeds_passed"],
            "fixed_resume_seeds_passed": readback_regression["fixed_resume_seeds_passed"],
            "synthetic_fixture_trainer_save_calls_this_preflight": readback_regression[
                "synthetic_fixture_trainer_save_calls_this_preflight"],
            "environment_constructor_calls": readback_regression[
                "environment_constructor_calls"],
            "ppo_update_calls": readback_regression["ppo_update_calls"],
            "optimizer_step_calls": readback_regression["optimizer_step_calls"],
        },
        "prior_round46_r2_preclaim_attempt": prior_r2_attempt,
        "checkpoint_journal_route": {
            "audit_sha256": _sha256(CHECKPOINT_JOURNAL_ROUTE_AUDIT_PATH),
            "legacy_path": journal_route["legacy_path"],
            "legacy_sha256": journal_route["legacy_sha256"],
            "legacy_entry_count": journal_route["legacy_entry_count"],
            "round46_local_copy_path": journal_route["r2_local_copy_path"],
            "round46_local_copy_sha256": _sha256(CHECKPOINT_JOURNAL_PATH),
            "future_writer_bound_to_round46_local_copy": True,
        },
        "checkpoint_write_accounting": checkpoint_write_accounting,
        "counts_before_checkpoint_readback_regression": before_readback_regression["counts"],
        "counts_after_checkpoint_readback_regression": after_readback_regression["counts"],
        "reference_cache": {
            "row_count": len(runtime.get("r2_reference_cache", {})),
            "scene_ids": sorted(int(scene) for scene in runtime.get("r2_reference_cache", {})),
            "identity_sha256": runtime.get("r2_reference_cache_sha256"),
            "loaded_through_actual_r42_r43_reference_cache": True,
        },
        "frozen_runtime_source_sha256": module._source_hashes(),
        "device": str(runtime["common"].DEVICE),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "before_counts": before_ledger["counts"],
        "after_counts": after_ledger["counts"],
        "environment_constructed": 0, "run_invocation_calls": 0,
        "completed_episodes": 0, "ppo_updates": 0,
        "optimizer_step_calls": 0,
        "checkpoint_write_calls": int(after_ledger["counts"]["checkpoint_write_calls"]),
        "checkpoint_write_accounting": checkpoint_write_accounting,
        "checkpoint_journal_output_scope": runtime.get("round46_checkpoint_journal_route"),
        "checkpoint_journal_route_regression": {
            "kind": checkpoint_journal_route_regression["kind"],
            "status": checkpoint_journal_route_regression["status"],
            "receipt_path": CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH.relative_to(REPO).as_posix(),
            "receipt_sha256": _sha256(CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH),
            "checkpoint_sha256": checkpoint_journal_route_regression["checkpoint_sha256"],
            "trainer_save_calls_delta": route_probe_writes,
            "claims": checkpoint_journal_route_regression["claims"],
            "environment_constructor_calls": checkpoint_journal_route_regression[
                "environment_constructor_calls"],
            "optimizer_step_calls": checkpoint_journal_route_regression["optimizer_step_calls"],
        },
        "network_or_push": False,
    }
    _write_json(PRELAUNCH_GATE_PATH, receipt)
    _write_text_lf(PRELAUNCH_SHA_PATH, _sha256(PRELAUNCH_GATE_PATH) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def _verify_prelaunch_identity(runtime: dict[str, Any], *,
                               require_r2_v5_evidence: bool = False) -> None:
    _assert_branch_head()
    _assert_source_worktree_clean()
    journal_route = _checkpoint_journal_route_audit(create=False)
    bound_route = runtime.get("round46_checkpoint_journal_route", {})
    if (Path(runtime["r42_run"].RUN_DIR).resolve() != V6_DIR.resolve()
            or Path(bound_route.get("journal_closure_after", "")).resolve()
            != CHECKPOINT_JOURNAL_PATH.resolve()
            or _sha256(CHECKPOINT_JOURNAL_ROUTE_AUDIT_PATH)
            != _json(INPUT_IDENTITY_PATH).get("checkpoint_journal_route_audit_sha256")
            or journal_route.get("legacy_sha256")
            != _json(INPUT_IDENTITY_PATH).get("checkpoint_journal_route_legacy_sha256")):
        raise RuntimeError("ROUND46 checkpoint journal output scope or historical identity failed")
    frozen = _json(FROZEN_CONFIG_PATH)
    receipt = _json(PRELAUNCH_GATE_PATH)
    pointer = PRELAUNCH_SHA_PATH.read_text(encoding="ascii").strip()
    actual_inputs = _input_identity()
    if actual_inputs != _json(INPUT_IDENTITY_PATH):
        raise RuntimeError("ROUND46 V5 frozen inputs changed after prelaunch")
    if (frozen.get("kind") != "round46-r2-v5-execution-freeze/1"
            or frozen.get("status") != "prepared_before_environment"
            or frozen.get("runner_version") != ROUND46_RUNNER_VERSION
            or frozen.get("training_version") != TRAINING_VERSION
            or frozen.get("branch") != ROUND46_BRANCH
            or frozen.get("runtime_freeze_sha256") != _sha256(RUNTIME_FREEZE_PATH)
            or receipt.get("branch") != ROUND46_BRANCH
            or receipt.get("runner_version") != ROUND46_RUNNER_VERSION
            or receipt.get("kind") != "round46-r2-v5-zero-environment-preflight/1"
            or receipt.get("text_identity_regression_status") != "passed"
            or receipt.get("text_identity_regression_sha256")
            != frozen.get("text_identity_regression_sha256")):
        raise RuntimeError("ROUND46 V5 execution freeze/runner identity is not active")
    for relative, record in receipt.get("committed_text_identity_checks", {}).items():
        current = _require_committed_file(
            REPO / Path(relative), record.get("raw_file_sha256", ""),
            record.get("git_blob_content_sha256", ""))
        if current.get("comparison_rule") != record.get("comparison_rule"):
            raise RuntimeError(f"ROUND46 V5 committed text identity rule changed: {relative}")
    if not receipt.get("committed_text_identity_checks"):
        raise RuntimeError("ROUND46 V5 preflight lacks committed text namespace receipts")
    reused = frozen.get("reused_preflight_lineage", {})
    for path, key in (
        (V3_FROZEN_CONFIG_PATH, "v3_frozen_config_sha256"),
        (V3_CONFIG_DIFF_PATH, "v3_configuration_diff_sha256"),
        (V3_INPUT_IDENTITY_PATH, "v3_input_identity_sha256"),
        (V3_KEY_PLAN_PATH, "v3_key_plan_sha256"),
        (V3_TRAINING_VERSION_PATH, "v3_training_version_manifest_sha256"),
        (V3_RUNTIME_FREEZE_PATH, "v3_runtime_freeze_sha256"),
        (V3_PRELAUNCH_GATE_PATH, "v3_prelaunch_gate_sha256"),
        (V3_PRELAUNCH_SHA_PATH, "v3_prelaunch_sha_file_sha256"),
        (READBACK_REGRESSION_RESULT_PATH, "v3_readback_regression_sha256"),
        (CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH,
         "v3_checkpoint_journal_route_regression_sha256"),
        (STATUS_GUARD_REGRESSION_PATH, "v3_status_guard_regression_sha256"),
    ):
        identity = _frozen_text_identity(path)
        if identity.get("raw_file_sha256") != reused.get(key):
            raise RuntimeError(f"ROUND46 V3 reused evidence identity changed: {path.name}")
    if (_frozen_text_identity(V3_STOP_ARCHIVE_PATH).get("raw_file_sha256")
            != frozen.get("predecessor_stop_archive_sha256")
            or _frozen_text_identity(V4_BINDING_AUDIT_PATH).get("raw_file_sha256")
            != frozen.get("binding_audit_sha256")):
        raise RuntimeError("ROUND46 V5 stopped-attempt lineage or source inputs changed")
    v4_preserved = receipt.get("v4_stop_preserved", {})
    v4_failure = _json(V4_PREFLIGHT_FAILURE_PATH)
    if (v4_preserved.get("failure_sha256") != _sha256(V4_PREFLIGHT_FAILURE_PATH)
            or v4_preserved.get("stop_report_sha256") != _sha256(V4_STOP_REPORT_PATH)
            or v4_preserved.get("status") != "stopped_before_claim"
            or v4_preserved.get("claims") != 0
            or v4_preserved.get("environment_constructor_calls") != 0
            or v4_preserved.get("optimizer_step_calls") != 0
            or v4_preserved.get("v4_prelaunch_gate_absent") is not True
            or v4_preserved.get("v4_binding_regression_absent") is not True
            or v4_failure.get("status") != "stopped_before_claim"
            or V4_PRELAUNCH_GATE_PATH.exists()
            or V4_BINDING_REGRESSION_PATH.exists()
            or V4_STATUS_GUARD_REUSE_PATH.exists()):
        raise RuntimeError("ROUND46 V5 did not preserve the stopped V4 lineage exactly")
    route_regression_receipt = receipt.get("checkpoint_journal_route_regression", {})
    if (route_regression_receipt.get("kind")
            != "round46-r2-checkpoint-journal-route-regression/1"
            or route_regression_receipt.get("status") != "passed"
            or not CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH.is_file()
            or route_regression_receipt.get("receipt_sha256")
            != _sha256(CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH)):
        raise RuntimeError("ROUND46 checkpoint-journal route regression receipt failed")
    route_regression = _json(CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH)
    if (route_regression.get("status") != "passed"
            or route_regression.get("trainer_save_calls_delta") != 1
            or route_regression.get("claims") != 0
            or route_regression.get("environment_constructor_calls") != 0
            or route_regression.get("run_invocation_calls") != 0
            or route_regression.get("ppo_updates") != 0
            or route_regression.get("optimizer_step_calls") != 0
            or route_regression.get("ambient_rng_preserved") is not True):
        raise RuntimeError("ROUND46 checkpoint-journal route regression side-effect gate failed")
    if (_sha256(PRELAUNCH_GATE_PATH) != pointer
            or receipt.get("kind") != "round46-r2-v5-zero-environment-preflight/1"
            or receipt.get("status") != "passed"
            or receipt.get("frozen_config_sha256") != _sha256(FROZEN_CONFIG_PATH)
            or receipt.get("key_plan_sha256") != _sha256(KEY_PLAN_PATH)
            or receipt.get("input_identity_sha256") != _sha256(INPUT_IDENTITY_PATH)
            or receipt.get("runner_source_sha256") != frozen.get("runner_source_sha256")
            or receipt.get("training_version") != TRAINING_VERSION
            or receipt.get("normal_cap") != NORMAL_CAP
            or receipt.get("technical_reserve_max") != RESERVE_CAP
            or receipt.get("hard_cap") != HARD_CAP
            or receipt.get("environment_constructed") != 0
            or receipt.get("optimizer_step_calls") != 0):
        raise RuntimeError("ROUND46 prelaunch identity receipt failed")
    regression = receipt.get("checkpoint_readback_regression", {})
    regression_path = READBACK_REGRESSION_RESULT_PATH
    if (regression.get("status") != "passed"
            or regression.get("source_sha256") != frozen.get("readback_regression_sha256")
            or not regression_path.is_file()
            or regression.get("receipt_sha256") != _sha256(regression_path)):
        raise RuntimeError("ROUND46 checkpoint readback regression receipt identity failed")
    prior_r2_attempt = frozen.get("inputs", {}).get("prior_round46_r2_preclaim_attempt", {})
    if (_sha256(R2_V2_PRECLAIM_ARCHIVE_PATH)
            != prior_r2_attempt.get("archive_manifest_sha256")):
        raise RuntimeError("ROUND46-R2 prior pre-claim attempt archive identity changed")
    ancestor = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor",
         receipt["head_at_preflight"], "HEAD"], capture_output=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("ROUND46 preflight commit is not an ancestor of current HEAD")
    for relative, expected in (
        (receipt["runner_path"], receipt["runner_source_sha256"]),
        (READBACK_REGRESSION_PATH.relative_to(REPO).as_posix(),
         frozen["readback_regression_sha256"]),
        (CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH.relative_to(REPO).as_posix(),
         route_regression_receipt["receipt_sha256"]),
        (regression_path.relative_to(REPO).as_posix(), regression["receipt_sha256"]),
        (PRELAUNCH_GATE_PATH.relative_to(REPO).as_posix(), pointer),
        (PRELAUNCH_SHA_PATH.relative_to(REPO).as_posix(),
         hashlib.sha256((pointer + "\n").encode("ascii")).hexdigest()),
    ):
        identity = _require_committed_file(REPO / Path(relative), expected)
        if identity.get("raw_file_sha256") != expected:
            raise RuntimeError(f"ROUND46 committed prelaunch identity differs: {relative}")
    if _sha256(R45_CONTROL_GATE_PATH) != frozen["inputs"]["round45_control_gate_sha256"]:
        raise RuntimeError("ROUND45 old failed gate was modified")
    if ((int(runtime["common"].NORMAL_LAUNCHES), int(runtime["common"].RESERVE_LAUNCHES),
            int(runtime["common"].LAUNCH_CAP)) != (NORMAL_CAP, RESERVE_CAP, HARD_CAP)):
        raise RuntimeError("ROUND46 runtime quota drifted after preflight")
    if require_r2_v5_evidence:
        _verify_status_guard_regression_receipt()
        _verify_v5_binding_regression_receipt()


def run_status_guard_regression() -> dict[str, Any]:
    """Exercise the actual task-state -> pre-claim identity seam with zero claims."""
    _assert_branch_head()
    _assert_source_worktree_clean()
    porcelain_args = ("status", "--porcelain", "--untracked-files=no")
    baseline_status = _run_git(*porcelain_args)
    if _unexpected_source_status_lines(baseline_status):
        raise RuntimeError("status-guard regression did not start from an allowed worktree")
    module = _configure_runtime_module()
    runtime = _RUNTIME
    assert runtime is not None
    module.verify_prelaunch_identity = lambda _frozen=None: _verify_prelaunch_identity(runtime)
    module.ENFORCE_PRELAUNCH = True
    module.verify_prelaunch_identity(runtime["frozen"])

    task_paths = (CURRENT_TASK_PATH, R46_DIR / "CURRENT_TASK.json")
    originals = {path: path.read_bytes() for path in task_paths}
    guarded_paths = (Path(__file__).resolve(), FROZEN_CONFIG_PATH.resolve())
    guarded_originals = {path: path.read_bytes() for path in guarded_paths}
    before = runtime["r42_run"].ledger_snapshot(strict=True)
    status_update_passed = False
    rejected_changes = {}
    actual_status_repr = None
    actual_status_paths = []
    archived_v2_identity = _prior_r2_preclaim_attempt_identity()
    try:
        module._task_state(
            "ROUND46-R2 zero-environment mutable task-state guard regression",
            phase="status_guard_regression", training_started=False,
        )
        current_task = _json(R46_DIR / "CURRENT_TASK.json")
        current_task["status_guard_probe"] = "temporary state update before pre-claim identity check"
        _write_json(R46_DIR / "CURRENT_TASK.json", current_task)
        status_after_update = _run_git(*porcelain_args)
        actual_status_repr = repr(status_after_update)
        actual_status_paths = [
            line[3:].replace("\\", "/") for line in status_after_update.splitlines()
            if len(line) >= 4
        ]
        expected_status_paths = {
            path.resolve().relative_to(REPO).as_posix() for path in task_paths
        }
        if (not status_after_update.startswith(" M ")
                or set(actual_status_paths) != expected_status_paths
                or _unexpected_source_status_lines(status_after_update)):
            raise RuntimeError(
                "real Git porcelain status did not preserve/allow exactly the two task-state paths: "
                + repr(status_after_update))
        module.verify_prelaunch_identity(runtime["frozen"])
        status_update_passed = True

        for label, changed_path in (
                ("runner_source", Path(__file__).resolve()),
                ("frozen_configuration", FROZEN_CONFIG_PATH.resolve())):
            original_bytes = guarded_originals[changed_path]
            try:
                changed_path.write_bytes(original_bytes + b"\n")
                actual_dirty_status = _run_git(*porcelain_args)
                changed_rel = changed_path.relative_to(REPO).as_posix()
                try:
                    module.verify_prelaunch_identity(runtime["frozen"])
                except RuntimeError as error:
                    if ("tracked source worktree is dirty" not in str(error)
                            or changed_rel not in str(error)):
                        raise
                    rejected_changes[label] = {
                        "status": "rejected by actual Git status",
                        "path": changed_rel,
                        "git_status_repr": repr(actual_dirty_status),
                    }
                else:
                    raise AssertionError(f"mutable task-state guard accepted changed {label}")
            finally:
                changed_path.write_bytes(original_bytes)
    finally:
        for path, contents in guarded_originals.items():
            path.write_bytes(contents)
        for path, contents in originals.items():
            path.write_bytes(contents)

    guarded_files_restored = all(path.read_bytes() == contents
                                 for path, contents in guarded_originals.items())
    if not guarded_files_restored:
        raise RuntimeError("status-guard regression did not restore runner/config bytes exactly")

    after = runtime["r42_run"].ledger_snapshot(strict=True)
    for key in ("normal_used", "reserve_used", "total_used"):
        if int(before[key]) != int(after[key]):
            raise RuntimeError(f"status guard regression changed quota field {key}")
    for key in ("environment_constructor_calls", "run_invocation_calls", "completed_steps",
                "ppo_update_calls", "optimizer_step_calls", "checkpoint_write_calls"):
        if int(before["counts"][key]) != int(after["counts"][key]):
            raise RuntimeError(f"status guard regression changed execution count {key}")
    final_status = _run_git(*porcelain_args)
    if (not status_update_passed or set(rejected_changes) != {"runner_source", "frozen_configuration"}
            or final_status != baseline_status):
        raise RuntimeError("status guard regression did not restore its temporary status-only changes")

    result = {
        "kind": "round46-r2-status-guard-regression/1",
        "status": "passed",
        "runner_version": ROUND46_RUNNER_VERSION,
        "training_version": TRAINING_VERSION,
        "head_at_regression": _run_git("rev-parse", "HEAD"),
        "prelaunch_gate_sha256": _sha256(PRELAUNCH_GATE_PATH),
        "old_v2_failure_reproduced_from_actual_archived_stop": {
            "status": "passed",
            "error": "tracked worktree is dirty; refusing ROUND46 environment entry: M runs/comm_bc_initialized_constrained_ppo_20261007_r2/ROUND46_CURRENT_TASK.json",
            "archived_stop_state_sha256": archived_v2_identity["archived_stop_state_sha256"],
        },
        "actual_task_state_then_preclaim_identity_passed": status_update_passed,
        "actual_git_porcelain_status_repr_after_task_state_write": actual_status_repr,
        "actual_git_porcelain_status_paths_after_task_state_write": actual_status_paths,
        "git_status_leading_space_preserved": bool(actual_status_repr and "' M " in actual_status_repr),
        "git_status_baseline_repr": repr(baseline_status),
        "git_status_restored_repr": repr(final_status),
        "mutable_status_paths_allowed": [
            CURRENT_TASK_PATH.resolve().relative_to(REPO).as_posix(),
            (R46_DIR / "CURRENT_TASK.json").resolve().relative_to(REPO).as_posix(),
        ],
        "scientific_source_refusals": rejected_changes,
        "temporary_status_files_restored_byte_identically": True,
        "source_and_frozen_config_changes_rejected_with_real_git_status": True,
        "source_and_frozen_config_restored_byte_identically": guarded_files_restored,
        "source_and_frozen_config_sha256_before_after": {
            str(path.resolve()): {
                "before": hashlib.sha256(contents).hexdigest(),
                "after": _sha256(path),
            }
            for path, contents in guarded_originals.items()
        },
        "counts_before": before["counts"],
        "counts_after": after["counts"],
        "normal_claims": after["normal_used"],
        "reserve_claims": after["reserve_used"],
        "environment_constructor_calls": after["counts"]["environment_constructor_calls"],
        "run_invocation_calls": after["counts"]["run_invocation_calls"],
        "completed_steps": after["counts"]["completed_steps"],
        "ppo_updates": after["counts"]["ppo_update_calls"],
        "optimizer_step_calls": after["counts"]["optimizer_step_calls"],
        "checkpoint_write_calls": after["counts"]["checkpoint_write_calls"],
        "checkpoint_write_delta": 0,
        "synthetic_environment_files_written": 0,
    }
    _write_json(STATUS_GUARD_REGRESSION_PATH, result, immutable=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _verify_status_guard_regression_receipt() -> None:
    if not STATUS_GUARD_REGRESSION_PATH.is_file():
        raise RuntimeError("ROUND46-R2 zero-environment status-guard regression is missing")
    result = _json(STATUS_GUARD_REGRESSION_PATH)
    if (result.get("kind") != "round46-r2-status-guard-regression/1"
            or result.get("status") != "passed"
            or result.get("training_version") != TRAINING_VERSION
            or result.get("actual_task_state_then_preclaim_identity_passed") is not True
            or result.get("temporary_status_files_restored_byte_identically") is not True
            or result.get("git_status_leading_space_preserved") is not True
            or result.get("source_and_frozen_config_changes_rejected_with_real_git_status") is not True
            or result.get("source_and_frozen_config_restored_byte_identically") is not True
            or result.get("checkpoint_write_delta") != 0
            or any(result.get(key) != 0 for key in (
                "normal_claims", "reserve_claims", "environment_constructor_calls",
                "run_invocation_calls", "completed_steps", "ppo_updates",
                "optimizer_step_calls"))):
        raise RuntimeError("ROUND46-R2 zero-environment status-guard regression receipt failed")
    if result.get("runner_version", "").split("+")[0] != \
            "round46-bc-initialized-constrained-ppo/3":
        raise RuntimeError("ROUND46 V5 may reuse only the preserved V3 status-guard receipt")
    if result.get("prelaunch_gate_sha256") != _sha256(V3_PRELAUNCH_GATE_PATH):
        raise RuntimeError("ROUND46 V3 status-guard receipt no longer matches its original gate")
    _frozen_text_identity(STATUS_GUARD_REGRESSION_PATH)
    if not V5_STATUS_GUARD_REUSE_PATH.is_file():
        raise RuntimeError("ROUND46 V5 status-guard reuse receipt is missing")
    reused = _json(V5_STATUS_GUARD_REUSE_PATH)
    current_hashes = _status_guard_function_hashes(Path(__file__).read_bytes())
    if (reused.get("kind") != "round46-r2-v5-status-guard-reuse/1"
            or reused.get("status") != "passed_reused_exact_implementation"
            or reused.get("runner_version") != ROUND46_RUNNER_VERSION
            or reused.get("training_version") != TRAINING_VERSION
            or reused.get("prior_status_guard_receipt_sha256")
            != _sha256(STATUS_GUARD_REGRESSION_PATH)
            or reused.get("prior_prelaunch_gate_sha256") != _sha256(V3_PRELAUNCH_GATE_PATH)
            or reused.get("v5_prelaunch_gate_sha256") != _sha256(PRELAUNCH_GATE_PATH)
            or reused.get("v3_status_guard_source_function_sha256") != current_hashes
            or reused.get("v5_status_guard_source_function_sha256") != current_hashes
            or reused.get("actual_status_guard_regression_rerun") is not False
            or any(reused.get(key) != 0 for key in (
                "environment_constructor_calls", "run_invocation_calls", "normal_claims",
                "reserve_claims", "ppo_updates", "optimizer_step_calls"))):
        raise RuntimeError("ROUND46 V5 status-guard reuse identity or scope failed")
    _require_committed_file(V5_STATUS_GUARD_REUSE_PATH, _sha256(V5_STATUS_GUARD_REUSE_PATH))


def _verify_v5_binding_regression_receipt() -> None:
    if not V5_BINDING_REGRESSION_PATH.is_file():
        raise RuntimeError("ROUND46 V5 actual inner-claim binding regression is missing")
    result = _json(V5_BINDING_REGRESSION_PATH)
    expected = {
        "kind": "round46-r2-v5-inner-claim-binding-regression/1",
        "status": "passed",
        "runner_version": ROUND46_RUNNER_VERSION,
        "training_version": TRAINING_VERSION,
        "runner_source_sha256": _sha256(Path(__file__)),
        "frozen_config_sha256": _sha256(FROZEN_CONFIG_PATH),
        "prelaunch_gate_sha256": _sha256(PRELAUNCH_GATE_PATH),
        "input_identity_sha256": _sha256(INPUT_IDENTITY_PATH),
        "binding_audit_sha256": _sha256(V4_BINDING_AUDIT_PATH),
        "v3_stop_archive_sha256": _sha256(V3_STOP_ARCHIVE_PATH),
        "status_guard_reuse_receipt_sha256": _sha256(V5_STATUS_GUARD_REUSE_PATH),
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError("ROUND46 V5 inner-claim binding regression identity differs")
    required_true = (
        "actual_r42_to_r45_to_r44_claim_chain_exercised",
        "stale_r45_binding_reproduced_before_atomic_claim",
        "v5_binding_reached_atomic_claim_probe_once",
        "source_mismatch_rejected_before_atomic_claim",
        "config_mismatch_rejected_before_atomic_claim",
        "key_plan_mismatch_rejected_before_atomic_claim",
        "model_identity_mismatch_rejected_before_atomic_claim",
        "all_temporary_mutations_restored_byte_identically",
        "claim_ledger_checkpoint_and_output_snapshots_unchanged",
        "trainer_rng_preserved",
        "actual_r44_global_points_to_v5_verifier_after_regression",
    )
    if any(result.get(key) is not True for key in required_true):
        raise RuntimeError("ROUND46 V5 inner-claim binding regression did not pass every case")
    if (result.get("stale_binding_atomic_claim_probe_calls") != 0
            or result.get("v5_binding_atomic_claim_probe_calls") != 1
            or result.get("normal_claim_delta") != 0
            or result.get("reserve_claim_delta") != 0
            or result.get("environment_constructor_calls_delta") != 0
            or result.get("run_invocation_calls_delta") != 0
            or result.get("completed_steps_delta") != 0
            or result.get("ppo_update_calls_delta") != 0
            or result.get("optimizer_step_calls_delta") != 0
            or result.get("checkpoint_write_calls_delta") != 0):
        raise RuntimeError("ROUND46 V5 binding regression side-effect counters are not zero")
    _require_committed_file(V5_BINDING_REGRESSION_PATH, _sha256(V5_BINDING_REGRESSION_PATH))


def run_r2_v5_binding_regression() -> dict[str, Any]:
    """Exercise the actual wrapped R42 -> R45 -> R44 claim with an atomic-boundary probe."""
    global _R2_V5_REQUIRE_EXECUTION_EVIDENCE
    _assert_branch_head()
    _assert_source_worktree_clean()
    if (V5_BINDING_REGRESSION_PATH.exists()
            or V5_STATUS_GUARD_REUSE_PATH.exists()):
        raise RuntimeError("ROUND46 V5 binding regression outputs already exist; refusing to overwrite")
    module = _configure_runtime_module()
    runtime = _RUNTIME
    assert runtime is not None
    if not PRELAUNCH_GATE_PATH.is_file() or not PRELAUNCH_SHA_PATH.is_file():
        raise RuntimeError("ROUND46 V5 prelaunch gate must be committed before binding regression")
    module.verify_prelaunch_identity = lambda _frozen=None: _verify_prelaunch_identity(
        runtime, require_r2_v5_evidence=False)
    module.ENFORCE_PRELAUNCH = True
    _R2_V5_REQUIRE_EXECUTION_EVIDENCE = False
    _verify_prelaunch_identity(runtime, require_r2_v5_evidence=False)

    binding = runtime.get("round46_r44_claim_binding", {})
    r44_module, actual_claim = _actual_r44_claim(runtime, module)
    stale_verifier = binding.get("stale_verifier")
    strict_verifier = binding.get("strict_verifier")
    if (not callable(stale_verifier) or not callable(strict_verifier)
            or actual_claim is not binding.get("claim_function")
            or r44_module.verify_frozen_source is not strict_verifier):
        raise RuntimeError("ROUND46 V5 regression did not start with the expected actual R44 binding")

    run_module = runtime["r42_run"]
    r40_common = runtime["r40_common"]
    if run_module._ACTIVE_CLAIM is not None:
        raise RuntimeError("ROUND46 V5 claim binding regression found an active claim")
    plan = _json(KEY_PLAN_PATH)
    scenario = int(plan["training_scenario_order"][0])
    logical_job = "new_training_s11_e0"
    metadata = {"training_seed": 11, "episode_index": 0}
    context = module._context_for_cell(
        "train", logical_job, scenario, metadata, {"tag": "train_s11_e0_r44"})
    if logical_job not in set(plan["logical_jobs"]):
        raise RuntimeError("ROUND46 V5 probe key is outside its committed logical-key plan")

    class ClaimLaunchBoundaryProbe(RuntimeError):
        pass

    probe_calls = 0

    def probe_claim_launch(*_args: Any, **_kwargs: Any):
        nonlocal probe_calls
        probe_calls += 1
        raise ClaimLaunchBoundaryProbe("ROUND46 V5 stopped at the atomic claim boundary")

    original_claim_launch = r40_common.claim_launch
    original_task_state = {
        path: path.read_bytes() for path in (CURRENT_TASK_PATH, R46_DIR / "CURRENT_TASK.json")
        if path.is_file()
    }
    source_path = Path(__file__).resolve()
    guarded_paths = (source_path, FROZEN_CONFIG_PATH, KEY_PLAN_PATH)
    guarded_originals = {path: path.read_bytes() for path in guarded_paths}
    ledger_before = run_module.ledger_snapshot(strict=True)
    journal_before = CHECKPOINT_JOURNAL_PATH.read_bytes()
    ledger_file_before = LEDGER_PATH.read_bytes() if LEDGER_PATH.is_file() else None
    output_roots = (
        V6_ROWS_DIR, V6_LOGS_DIR, V6_DIR / "launch_claims",
        V6_CHECKPOINTS_DIR, V6_PROGRESS_DIR,
        V6_DIR / "training_reports", V6_DIR / "failures", V6_REJECTED_DIR,
    )

    def snapshot_outputs() -> dict[str, tuple[int, str]]:
        files = {}
        for root in output_roots:
            if root.exists():
                for path in sorted(item for item in root.rglob("*") if item.is_file()):
                    files[path.relative_to(R46_DIR).as_posix()] = (path.stat().st_size, _sha256(path))
        return files

    outputs_before = snapshot_outputs()
    rng_before = runtime["trainer"].rng_state()
    original_input_identity = globals()["_input_identity"]

    def call_actual_claim() -> None:
        token = module.CURRENT_CONTEXT.set(dict(context))
        try:
            run_module._claim(logical_job, attempt=1, metadata=dict(metadata))
        finally:
            module.CURRENT_CONTEXT.reset(token)

    stale_reproduced = False
    fixed_binding_reached_probe = False
    mismatch_results: dict[str, dict[str, Any]] = {}
    r40_common.claim_launch = probe_claim_launch
    try:
        r44_module.verify_frozen_source = stale_verifier
        try:
            call_actual_claim()
        except ClaimLaunchBoundaryProbe:
            raise RuntimeError("stale R45 verifier unexpectedly passed the actual R44 claim")
        except RuntimeError as error:
            stale_reproduced = "R2 prelaunch/freeze status is not passed" in str(error)
        if not stale_reproduced or probe_calls != 0:
            raise RuntimeError("ROUND46 V5 did not reproduce the prior stale-binding rejection")

        r44_module.verify_frozen_source = strict_verifier
        try:
            call_actual_claim()
        except ClaimLaunchBoundaryProbe:
            fixed_binding_reached_probe = True
        if not fixed_binding_reached_probe or probe_calls != 1:
            raise RuntimeError("ROUND46 V5 binding failed to reach the actual atomic claim probe")

        def expect_rejection(label: str) -> None:
            calls_before = probe_calls
            try:
                call_actual_claim()
            except ClaimLaunchBoundaryProbe:
                raise RuntimeError(f"ROUND46 V5 {label} mismatch reached the atomic claim probe")
            except (RuntimeError, ValueError, OSError) as error:
                mismatch_results[label] = {
                    "rejected": True,
                    "rejection": f"{type(error).__name__}: {error}",
                    "atomic_probe_calls_delta": probe_calls - calls_before,
                }
            else:
                raise RuntimeError(f"ROUND46 V5 {label} mismatch was not rejected")
            if probe_calls != calls_before:
                raise RuntimeError(f"ROUND46 V5 {label} mismatch reached claim_launch")

        for label, path in (("source", source_path), ("config", FROZEN_CONFIG_PATH),
                            ("key_plan", KEY_PLAN_PATH)):
            original = guarded_originals[path]
            try:
                path.write_bytes(original + b"\n")
                expect_rejection(label)
            finally:
                path.write_bytes(original)
                if path.read_bytes() != original:
                    raise RuntimeError(f"ROUND46 V5 {label} probe bytes were not restored")

        original_input_identity_function = globals()["_input_identity"]
        try:
            def mismatched_input_identity() -> dict[str, Any]:
                value = copy.deepcopy(original_input_identity_function())
                value["bc_models_by_seed"]["11"]["parameter_sha256"] = "0" * 64
                return value
            globals()["_input_identity"] = mismatched_input_identity
            expect_rejection("model_identity")
        finally:
            globals()["_input_identity"] = original_input_identity_function
    finally:
        globals()["_input_identity"] = original_input_identity
        r44_module.verify_frozen_source = strict_verifier
        r40_common.claim_launch = original_claim_launch

    ledger_after = run_module.ledger_snapshot(strict=True)
    rng_after = runtime["trainer"].rng_state()
    outputs_after = snapshot_outputs()
    task_restored = all(path.read_bytes() == contents
                        for path, contents in original_task_state.items())
    guarded_restored = all(path.read_bytes() == contents
                           for path, contents in guarded_originals.items())
    counters_before = ledger_before["counts"]
    counters_after = ledger_after["counts"]
    if (ledger_before != ledger_after
            or outputs_before != outputs_after
            or journal_before != CHECKPOINT_JOURNAL_PATH.read_bytes()
            or ledger_file_before != (LEDGER_PATH.read_bytes() if LEDGER_PATH.is_file() else None)
            or not task_restored or not guarded_restored
            or run_module._ACTIVE_CLAIM is not None
            or r44_module.verify_frozen_source is not strict_verifier
            or module._same_value(rng_before, rng_after) is not True
            or set(mismatch_results) != {"source", "config", "key_plan", "model_identity"}):
        raise RuntimeError("ROUND46 V5 binding regression changed real execution state or missed a case")

    status_reuse = _create_status_guard_reuse_receipt()
    result = {
        "kind": "round46-r2-v5-inner-claim-binding-regression/1",
        "status": "passed",
        "runner_version": ROUND46_RUNNER_VERSION,
        "training_version": TRAINING_VERSION,
        "head_at_regression": _run_git("rev-parse", "HEAD"),
        "runner_source_sha256": _sha256(Path(__file__)),
        "frozen_config_sha256": _sha256(FROZEN_CONFIG_PATH),
        "prelaunch_gate_sha256": _sha256(PRELAUNCH_GATE_PATH),
        "input_identity_sha256": _sha256(INPUT_IDENTITY_PATH),
        "binding_audit_sha256": _sha256(V4_BINDING_AUDIT_PATH),
        "v3_stop_archive_sha256": _sha256(V3_STOP_ARCHIVE_PATH),
        "status_guard_reuse_receipt_sha256": _sha256(V5_STATUS_GUARD_REUSE_PATH),
        "actual_r42_to_r45_to_r44_claim_chain_exercised": True,
        "actual_claim_function_source": str(Path(actual_claim.__code__.co_filename).resolve()),
        "actual_r44_global_points_to_v5_verifier_after_regression": True,
        "stale_r45_binding_reproduced_before_atomic_claim": stale_reproduced,
        "stale_binding_error": "R2 prelaunch/freeze status is not passed",
        "stale_binding_atomic_claim_probe_calls": 0,
        "v5_binding_reached_atomic_claim_probe_once": fixed_binding_reached_probe,
        "v5_binding_atomic_claim_probe_calls": probe_calls,
        "mismatch_rejections": mismatch_results,
        "source_mismatch_rejected_before_atomic_claim": mismatch_results["source"]["rejected"],
        "config_mismatch_rejected_before_atomic_claim": mismatch_results["config"]["rejected"],
        "key_plan_mismatch_rejected_before_atomic_claim": mismatch_results["key_plan"]["rejected"],
        "model_identity_mismatch_rejected_before_atomic_claim": mismatch_results["model_identity"]["rejected"],
        "all_temporary_mutations_restored_byte_identically": guarded_restored,
        "claim_ledger_checkpoint_and_output_snapshots_unchanged": (
            ledger_before == ledger_after and outputs_before == outputs_after
            and journal_before == CHECKPOINT_JOURNAL_PATH.read_bytes()),
        "trainer_rng_preserved": module._same_value(rng_before, rng_after),
        "task_state_restored_byte_identically": task_restored,
        "status_guard_reuse_receipt_status": status_reuse["status"],
        "counts_before": ledger_before["counts"],
        "counts_after": ledger_after["counts"],
        "normal_claim_delta": ledger_after["normal_used"] - ledger_before["normal_used"],
        "reserve_claim_delta": ledger_after["reserve_used"] - ledger_before["reserve_used"],
        "environment_constructor_calls_delta": counters_after["environment_constructor_calls"]
            - counters_before["environment_constructor_calls"],
        "run_invocation_calls_delta": counters_after["run_invocation_calls"]
            - counters_before["run_invocation_calls"],
        "completed_steps_delta": counters_after["completed_steps"] - counters_before["completed_steps"],
        "ppo_update_calls_delta": counters_after["ppo_update_calls"]
            - counters_before["ppo_update_calls"],
        "optimizer_step_calls_delta": counters_after["optimizer_step_calls"]
            - counters_before["optimizer_step_calls"],
        "checkpoint_write_calls_delta": counters_after["checkpoint_write_calls"]
            - counters_before["checkpoint_write_calls"],
        "environment_constructed": 0,
        "run_invocation_calls": 0,
        "ppo_updates": 0,
        "optimizer_step_calls": 0,
    }
    _write_json(V5_BINDING_REGRESSION_PATH, result, immutable=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _check_first_four_report(seed: int, report: dict[str, Any]) -> None:
    if (int(report.get("episodes_done", -1)) < EPISODES_PER_BATCH
            or int(report.get("updates", -1)) < 1
            or report.get("training_version") != TRAINING_VERSION):
        raise RuntimeError(f"ROUND46 seed {seed} did not close its first real PPO batch")


def _run_first_four_and_all_gates(module: Any, runtime: dict[str, Any]) -> dict[int, dict[str, Any]]:
    first_four: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        module._task_state(
            f"ROUND46 seed {seed}: fixed first four training episodes",
            active_training_seed=seed, phase="first_four",
        )
        report = module._call_training_seed(runtime, seed, episodes=4)
        _check_first_four_report(seed, report)
        first_four_path = R46_DIR / f"ROUND46_FIRST_FOUR_s{seed}.json"
        if first_four_path.exists():
            saved = _json(first_four_path)
            if (saved.get("kind") != "round46-first-four-report/1"
                    or int(saved.get("seed", -1)) != seed
                    or saved.get("training_version") != TRAINING_VERSION):
                raise RuntimeError(f"ROUND46 first-four receipt identity changed for seed {seed}")
            first_four[seed] = saved["report"]
        else:
            first_four[seed] = report
            _write_json(first_four_path, {
                "kind": "round46-first-four-report/1", "seed": seed,
                "runner_version": ROUND46_RUNNER_VERSION,
                "training_version": TRAINING_VERSION, "report": report,
            })

    gates = {}
    for seed in SEEDS:
        module._task_state(
            f"ROUND46 seed {seed}: actual R43/R44 first-batch correctness gate",
            active_training_seed=seed, phase="first_batch_gate",
        )
        gate = module._run_first_batch_gate(runtime, seed)
        if gate.get("status") != "passed":
            raise RuntimeError(f"ROUND46 seed {seed} actual batch0 gate failed")
        gate_round46 = dict(gate)
        gate_round46.update({
            "kind": "round46-actual-first-batch-gate/1",
            "round": 46, "runner_version": ROUND46_RUNNER_VERSION,
            "training_version": TRAINING_VERSION,
            "actual_gate_implementation": "R43 first_batch_gate via R44/R42 runtime",
        })
        _write_json(R46_DIR / f"ROUND46_FIRST_BATCH_GATE_s{seed}.json", gate_round46)
        gates[seed] = gate_round46
    if set(gates) != set(SEEDS) or any(item["status"] != "passed" for item in gates.values()):
        raise RuntimeError("ROUND46 cross-seed gate did not pass for all three seeds")
    module._task_state(
        "ROUND46 all three first-batch gates passed; fixed continuation authorized",
        all_seed_first_batch_gates="passed", gates_completed=3,
    )
    return first_four


def _continue_fixed_training(module: Any, runtime: dict[str, Any],
                             first_four: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    summaries: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        module._task_state(
            f"ROUND46 seed {seed}: fixed continuation to ep128",
            active_training_seed=seed, phase="continuation",
        )
        full = module._call_training_seed(runtime, seed, episodes=128)
        if (int(full.get("episodes_done", -1)) != 128
                or int(full.get("updates", -1)) != 32
                or int(full.get("optimizer_steps", -1)) <= 0
                or full.get("training_version") != TRAINING_VERSION):
            raise RuntimeError(f"ROUND46 seed {seed} fixed ep128 training failed closure")
        checkpoint = V6_CHECKPOINTS_DIR / f"v2_train_s{seed}_ep128.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        summary = {
            "kind": "round46-seed-training-result/1", "status": "complete",
            "seed": seed, "training_version": TRAINING_VERSION,
            "runner_version": ROUND46_RUNNER_VERSION,
            "first_four_report": first_four[seed],
            "fixed_ep128_report": full,
            "ep128_checkpoint_path": str(checkpoint.resolve()),
            "ep128_checkpoint_sha256": _sha256(checkpoint),
            "bc_checkpoint": module._bc_entry_for_seed(seed),
        }
        summary_path = R46_DIR / f"ROUND46_TRAINING_s{seed}.json"
        if summary_path.exists():
            saved = _json(summary_path)
            if (saved.get("status") != "complete"
                    or saved.get("ep128_checkpoint_sha256") != summary["ep128_checkpoint_sha256"]):
                raise RuntimeError(f"ROUND46 seed {seed} ep128 summary conflicts with checkpoint")
            summary = saved
        else:
            _write_json(summary_path, summary)
        summaries[seed] = summary
        module._task_state(
            f"ROUND46 seed {seed}: ep128 complete",
            seed_episodes_done=128, seed_ppo_updates=32,
            completed_training_seeds=sorted(summaries),
        )
    return summaries


def _load_round45_bc_comparison() -> dict[int, dict[int, dict[str, Any]]]:
    summary = _json(R45_CONTROL_RECOMPUTATION_PATH)
    result = {}
    for seed in SEEDS:
        rows = summary["per_seed"][str(seed)]["scenes"]
        by_scene = {int(item["scene"]): item for item in rows}
        if set(by_scene) != set(DEV_SCENES):
            raise RuntimeError(f"ROUND45 BC comparison seed {seed} has a different scene set")
        result[seed] = by_scene
    return result


def _on_time_ids(r45: Any, row: dict[str, Any]) -> set[str]:
    return {str(value) for value in r45._on_time_ids(row)}


def _bootstrap_index_matrix() -> tuple[Any, str]:
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indexes = rng.integers(0, 16, size=(BOOTSTRAP_REPLICATES, 16), dtype=np.int64)
    digest = hashlib.sha256(indexes.tobytes(order="C")).hexdigest()
    return indexes, digest


def _paired_metric(base: list[float], new: list[float], indexes: Any,
                   index_sha256: str) -> dict[str, Any]:
    import numpy as np
    base_values = np.asarray(base, dtype=np.float64)
    new_values = np.asarray(new, dtype=np.float64)
    differences = new_values - base_values
    draws = differences[indexes].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975], method="linear")
    return {
        "n_scenes": len(base), "baseline_mean": float(base_values.mean()),
        "new_mean": float(new_values.mean()),
        "mean_new_minus_baseline": float(differences.mean()),
        "shared_scene_bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap": {
            "rng": "numpy.random.Generator(PCG64(20261007))",
            "index_shape": [BOOTSTRAP_REPLICATES, 16],
            "index_matrix_sha256": index_sha256,
            "aggregation": "values[index].mean(axis=1)",
            "quantile_method": "linear",
        },
    }


def _build_analysis(module: Any, runtime: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    row_dir = Path(runtime["common"].ROWS)
    for path in sorted(row_dir.glob("*.json")):
        row = _json(path)
        if row.get("training_version") == TRAINING_VERSION:
            rows.append(row)
    training_rows = [row for row in rows if row.get("training_arm") == "NEW"]
    eval_rows = [row for row in rows if row.get("kind") == "eval"
                 and row.get("model_arm") == "NEW"]
    expected_train = {(seed, episode) for seed in SEEDS for episode in range(128)}
    train_keys = {(int(row["training_seed"]), int(row["episode_index"]))
                  for row in training_rows}
    expected_eval = {
        (seed, mode, scene) for seed in SEEDS
        for mode in ("greedy", "sampled") for scene in DEV_SCENES
    }
    eval_keys = {
        (int(row["training_seed"]), str(row["eval_mode"]), int(row["scenario_seed"]))
        for row in eval_rows
    }
    if len(training_rows) != TRAINING_KEYS or train_keys != expected_train:
        raise RuntimeError(f"ROUND46 training row closure failed: {len(training_rows)}/384")
    if len(eval_rows) != EVALUATION_KEYS or eval_keys != expected_eval:
        raise RuntimeError(f"ROUND46 final evaluation key closure failed: {len(eval_rows)}/96")
    for row in training_rows + eval_rows:
        runtime["r42_analysis"]._validate_reward_timing(row)
        if row.get("comm_action_semantics") != "one-shot-send-skip/1":
            raise RuntimeError(f"ROUND46 row has foreign one-shot identity: {row.get('label')}")

    baseline = module._fixed_gate_baseline()
    bc_rows = _load_round45_bc_comparison()
    immediate = baseline["reference_rows"]
    rules = baseline["rule_rows"]
    indexes, index_sha = _bootstrap_index_matrix()
    paired: dict[str, Any] = {}
    per_seed: dict[str, Any] = {}
    target_diffs: dict[str, Any] = {}
    for seed in SEEDS:
        seed_train = sorted(
            [row for row in training_rows if int(row["training_seed"]) == seed],
            key=lambda row: int(row["episode_index"]),
        )
        if len(seed_train) != 128:
            raise RuntimeError(f"ROUND46 seed {seed} does not have exactly 128 training rows")
        mode_rows = {}
        mode_diffs = {}
        for mode in ("greedy", "sampled"):
            by_scene = {
                int(row["scenario_seed"]): row for row in eval_rows
                if int(row["training_seed"]) == seed and row.get("eval_mode") == mode
            }
            if set(by_scene) != set(DEV_SCENES):
                raise RuntimeError(f"ROUND46 seed {seed} {mode} eval has wrong scene set")
            mode_rows[mode] = by_scene
            scene_diffs = []
            for scene in DEV_SCENES:
                row = by_scene[scene]
                ids = sorted(_on_time_ids(module, row))
                immediate_ids = sorted(_on_time_ids(module, immediate[scene]))
                bc_ids = sorted(str(value) for value in bc_rows[seed][scene]["bc_on_time_task_ids"])
                scene_diffs.append({
                    "scene": scene,
                    "new_on_time_task_ids": ids,
                    "immediate_on_time_task_ids": immediate_ids,
                    "bc_greedy_on_time_task_ids": bc_ids,
                    "new_minus_immediate_missing": sorted(set(immediate_ids) - set(ids)),
                    "new_minus_immediate_extra": sorted(set(ids) - set(immediate_ids)),
                    "new_minus_bc_missing": sorted(set(bc_ids) - set(ids)),
                    "new_minus_bc_extra": sorted(set(ids) - set(bc_ids)),
                    "exact_match_immediate": ids == immediate_ids,
                    "exact_match_bc_greedy": ids == bc_ids,
                })
            mode_diffs[mode] = scene_diffs

            base_sets = {
                "BC_greedy": [bc_rows[seed][scene]["bc_on_time_task_ids"] for scene in DEV_SCENES],
                "immediate": [_on_time_ids(module, immediate[scene]) for scene in DEV_SCENES],
                "late_discovery_gate": [
                    _on_time_ids(module, rules["late_discovery_gate"][scene])
                    for scene in DEV_SCENES],
                "always_hold": [
                    _on_time_ids(module, rules["always_hold"][scene])
                    for scene in DEV_SCENES],
            }
            base_cost = {
                "BC_greedy": [float(bc_rows[seed][scene]["spent_bytes"]) for scene in DEV_SCENES],
                "immediate": [float(immediate[scene]["spent_bytes"]) for scene in DEV_SCENES],
                "late_discovery_gate": [
                    float(rules["late_discovery_gate"][scene]["spent_bytes"])
                    for scene in DEV_SCENES],
                "always_hold": [
                    float(rules["always_hold"][scene]["spent_bytes"])
                    for scene in DEV_SCENES],
            }
            new_cost = [float(by_scene[scene]["spent_bytes"]) for scene in DEV_SCENES]
            new_tasks = [float(len(_on_time_ids(module, by_scene[scene])))
                         for scene in DEV_SCENES]
            metrics = {}
            for benchmark in ("BC_greedy", "immediate", "late_discovery_gate", "always_hold"):
                benchmark_tasks = [float(len(values)) for values in base_sets[benchmark]]
                metrics[benchmark] = {
                    "cost_bytes": _paired_metric(base_cost[benchmark], new_cost,
                                                 indexes, index_sha),
                    "on_time_task_count": _paired_metric(benchmark_tasks, new_tasks,
                                                         indexes, index_sha),
                    "exact_task_id_set_equal_every_scene": all(
                        set(base_sets[benchmark][index]) ==
                        _on_time_ids(module, by_scene[scene])
                        for index, scene in enumerate(DEV_SCENES)),
                    "baseline_task_id_set_counts": [
                        len(set(values)) for values in base_sets[benchmark]],
                    "new_task_id_set_counts": [int(value) for value in new_tasks],
                }
            paired[f"s{seed}_{mode}"] = metrics

        greedy = mode_rows["greedy"]
        immediate_equal = all(
            _on_time_ids(module, greedy[scene]) == _on_time_ids(module, immediate[scene])
            for scene in DEV_SCENES)
        mean_greedy_bytes = sum(int(greedy[scene]["spent_bytes"])
                                for scene in DEV_SCENES) / 16.0
        late_mean = baseline["late_mean_spent_bytes"]
        seed_train_report = _json(R46_DIR / f"ROUND46_TRAINING_s{seed}.json")
        per_seed[str(seed)] = {
            "training_episode_count": len(seed_train),
            "ppo_update_count": int(seed_train_report["fixed_ep128_report"]["updates"]),
            "optimizer_step_count": int(seed_train_report["fixed_ep128_report"]["optimizer_steps"]),
            "bc_initial_parameter_sha256": module._bc_entry_for_seed(seed)["parameter_sha256"],
            "final_parameter_identity": runtime["r42_run"].load_eval_model("NEW", seed)[1][
                "parameter_identity"],
            "greedy_mean_on_time_task_count": sum(
                len(_on_time_ids(module, greedy[scene])) for scene in DEV_SCENES) / 16.0,
            "greedy_mean_spent_bytes": mean_greedy_bytes,
            "sampled_mean_on_time_task_count": sum(
                len(_on_time_ids(module, mode_rows["sampled"][scene])) for scene in DEV_SCENES) / 16.0,
            "sampled_mean_spent_bytes": sum(
                int(mode_rows["sampled"][scene]["spent_bytes"]) for scene in DEV_SCENES) / 16.0,
            "immediate_exact_task_id_sets_preserved_greedy": immediate_equal,
            "greedy_cost_strictly_below_late_rule": mean_greedy_bytes < late_mean,
            "primary_seed_success": immediate_equal and mean_greedy_bytes < late_mean,
            "late_rule_mean_bytes": late_mean,
            "bc_greedy_mean_bytes": sum(
                int(bc_rows[seed][scene]["spent_bytes"]) for scene in DEV_SCENES) / 16.0,
            "bc_sampled_paired_baseline": "unavailable; ROUND45 BC controls were greedy-only",
            "training_rollout_means": {
                "on_time_task_count": sum(int(row["on_time"]) for row in seed_train) / 128.0,
                "spent_bytes": sum(int(row["spent_bytes"]) for row in seed_train) / 128.0,
            },
            "first_choice_counts": _choice_counts(seed_train + list(greedy.values())),
        }
        target_diffs[str(seed)] = mode_diffs

    all_primary = all(item["primary_seed_success"] for item in per_seed.values())
    ledger = runtime["r42_run"].ledger_snapshot(strict=True)
    results = {
        "kind": "round46-final-results/1",
        "status": "complete_development_block_joint_improvement" if all_primary
        else "complete_primary_criterion_not_met",
        "training_version": TRAINING_VERSION,
        "runner_version": ROUND46_RUNNER_VERSION,
        "primary_readout": "greedy",
        "secondary_readout": "sampled",
        "seed_results": per_seed,
        "baseline_summary": {
            "immediate": _summary_group(list(immediate.values())),
            "late_discovery_gate": _summary_group(list(rules["late_discovery_gate"].values())),
            "always_hold": _summary_group(list(rules["always_hold"].values())),
            "round45_bc_greedy": {
                str(seed): {
                    "mean_on_time_task_count": sum(
                        int(bc_rows[seed][scene]["on_time_count"]) for scene in DEV_SCENES) / 16.0,
                    "mean_spent_bytes": sum(
                        int(bc_rows[seed][scene]["spent_bytes"]) for scene in DEV_SCENES) / 16.0,
                } for seed in SEEDS
            },
            "round45_bc_sampled": "unavailable",
        },
        "paired_comparisons": paired,
        "bootstrap": {
            "rng": "numpy.random.Generator(PCG64(20261007))",
            "replicates": BOOTSTRAP_REPLICATES, "scene_count": 16,
            "index_matrix_sha256": index_sha,
            "aggregation": "values[index].mean(axis=1)",
            "quantile_method": "linear",
        },
        "counts": ledger["counts"],
        "budget": {
            "normal_planned": NORMAL_CAP, "normal_used": ledger["normal_used"],
            "reserve_used": ledger["reserve_used"], "total_used": ledger["total_used"],
            "hard_cap": HARD_CAP, "remaining_hard_cap": HARD_CAP - int(ledger["total_used"]),
        },
        "interpretation": {
            "development_block_joint_improvement": all_primary,
            "criterion": "each seed's greedy exact immediate on-time task-ID sets in every development scene and mean spent bytes strictly below the fixed late_discovery_gate block mean",
            "development_data_is_not_independent_confirmation": True,
            "not_formal_noninferiority_or_generalization": True,
            "round45_failed_gate_remains_failed": True,
            "round45_gate_not_used_as_round46_training_authorization": True,
            "tick_timing_is_not_claimed_to_explain_all_learning_failures": True,
        },
    }
    _write_json(PAIRED_RESULTS_PATH, {
        "kind": "round46-paired-results/1",
        "bootstrap": results["bootstrap"],
        "paired_comparisons": paired,
    })
    _write_json(TARGET_DIFFS_PATH, {
        "kind": "round46-target-id-diffs/1",
        "scenes": list(DEV_SCENES), "per_seed": target_diffs,
    })
    _write_json(FINAL_RESULTS_PATH, results)
    return results, {"training_rows": training_rows, "eval_rows": eval_rows,
                     "ledger": ledger, "baseline": baseline}


def _choice_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "free_choices": 0, "SEND_NOW": 0, "SKIP_EPISODE": 0,
        "fixed_HOLD": 0, "rows_without_action_counts": 0,
    }
    for row in rows:
        summary = row.get("collector_summary") or {}
        actions = summary.get("one_shot_counts") or {}
        if not isinstance(actions, dict) or not actions:
            counts["rows_without_action_counts"] += 1
            continue
        send = int(actions.get("send_now_choices", 0) or 0)
        skip = int(actions.get("skip_episode_choices", 0) or 0)
        counts["SEND_NOW"] += send
        counts["SKIP_EPISODE"] += skip
        counts["free_choices"] += int(actions.get("free_choices", send + skip) or 0)
        counts["fixed_HOLD"] += int(actions.get("fixed_holds_after_choice", 0) or 0)
    return counts


def _summary_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "on_time_total": sum(int(row["on_time"]) for row in rows),
        "spent_bytes_total": sum(int(row["spent_bytes"]) for row in rows),
        "mean_on_time": (sum(int(row["on_time"]) for row in rows) / len(rows)
                         if rows else 0.0),
        "mean_spent_bytes": (sum(int(row["spent_bytes"]) for row in rows) / len(rows)
                             if rows else 0.0),
    }


def _write_training_curves(runtime: dict[str, Any]) -> None:
    episodes = []
    batches = []
    for seed in SEEDS:
        seed_rows = []
        for path in sorted((V6_ROWS_DIR).glob("*.json")):
            row = _json(path)
            if (row.get("training_version") == TRAINING_VERSION
                    and row.get("training_arm") == "NEW"
                    and int(row.get("training_seed", -1)) == seed):
                seed_rows.append(row)
        seed_rows.sort(key=lambda row: int(row["episode_index"]))
        if len(seed_rows) != 128:
            raise RuntimeError(f"ROUND46 training curves require 128 rows for seed {seed}")
        for row in seed_rows:
            episodes.append({
                "seed": seed, "episode_index": int(row["episode_index"]),
                "scene": int(row["scenario_seed"]), "on_time": int(row["on_time"]),
                "spent_bytes": int(row["spent_bytes"]),
                "lambda_at_rollout": row.get("lambda_at_rollout"),
            })
        for batch_index in range(32):
            candidates = sorted((V6_CHECKPOINTS_DIR).glob(
                f"r40_batch_s{seed}_b{batch_index}_attempt*.json"))
            if len(candidates) != 1:
                raise RuntimeError(f"ROUND46 seed {seed} batch {batch_index} missing/ambiguous")
            raw = _json(candidates[0])
            update = raw.get("update", {})
            batches.append({
                "seed": seed, "batch_index": batch_index,
                "batch_record": candidates[0].name,
                "batch_record_sha256": _sha256(candidates[0]),
                "g_hat": update.get("g_hat", raw.get("g_hat")),
                "lambda_before": raw.get("lambda_at_rollout"),
                "lambda_after": update.get("lambda_next", update.get("lambda")),
                "optimizer_steps": update.get("optimizer_steps"),
                "preupdate_checkpoint": raw.get("preupdate_checkpoint"),
                "preupdate_checkpoint_sha256": raw.get("preupdate_checkpoint_sha256"),
            })
    _write_json(CURVES_PATH, {
        "kind": "round46-training-curves/1",
        "episode_count": len(episodes), "batch_count": len(batches),
        "episodes": episodes, "batches": batches,
    })


def _write_checkpoint_index(runtime: dict[str, Any]) -> None:
    import torch
    import r45_bc
    entries = []
    for seed in SEEDS:
        for episode in (0, 64, 128):
            path = V6_CHECKPOINTS_DIR / f"v2_train_s{seed}_ep{episode}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            controller, _, identity = runtime["trainer"].load_checkpoint_read_only(
                path, freeze=runtime["round42_freeze"])
            parameter_sha = identity.get("parameter_identity", {}).get(
                "parameter_sha256", controller.parameter_summary()["parameter_sha256"])
            entry = {
                "seed": seed, "episode": episode, "path": str(path.resolve()),
                "file_sha256": _sha256(path),
                "parameter_sha256": parameter_sha,
                "module_parameter_sha256": _module_digest(controller, r45_bc),
                "training_version": payload.get("training_version"),
                "episodes_done": payload.get("training_state", {}).get("episodes_done"),
                "updates": payload.get("training_state", {}).get("updates"),
                "optimizer_steps": payload.get("training_state", {}).get("optimizer_steps"),
                "lambda_dual": payload.get("training_state", {}).get("lambda"),
                "optimizer_loaded_for_index": False,
                "rng_loaded_for_index": False,
            }
            if entry["training_version"] != TRAINING_VERSION:
                raise RuntimeError(f"ROUND46 checkpoint training version mismatch: {path}")
            entries.append(entry)
            del controller, payload
        for batch_index in range(32):
            for prefix in ("r40_batch", "r40_preupdate"):
                candidates = list((V6_CHECKPOINTS_DIR).glob(
                    f"{prefix}_s{seed}_b{batch_index}_attempt*.json"))
                if prefix == "r40_batch" and len(candidates) != 1:
                    raise RuntimeError(f"ROUND46 batch state missing for seed {seed}, batch {batch_index}")
                for path in candidates:
                    entries.append({
                        "seed": seed, "batch_index": batch_index,
                        "record_kind": prefix, "path": str(path.resolve()),
                        "file_sha256": _sha256(path),
                    })
    _write_json(CHECKPOINT_INDEX_PATH, {
        "kind": "round46-checkpoint-index/1",
        "checkpoints_and_batch_records": entries,
        "milestone_checkpoint_count": 9,
        "batch_count": 96,
    })


def _write_artifact_manifest(runtime: dict[str, Any]) -> None:
    files = []
    include_roots = (
        V6_ROWS_DIR, V6_LOGS_DIR, V6_DIR / "launch_claims",
        V6_CHECKPOINTS_DIR, V6_PROGRESS_DIR,
        V6_DIR / "training_reports", V6_DIR / "failures", V6_REJECTED_DIR,
    )
    for root in include_roots:
        if root.exists():
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                files.append({
                    "path": str(path.resolve()),
                    "relative_path": path.relative_to(R46_DIR).as_posix(),
                    "size_bytes": path.stat().st_size, "sha256": _sha256(path),
                })
    _write_json(ARTIFACT_MANIFEST_PATH, {
        "kind": "round46-raw-artifact-manifest/1",
        "raw_model_tape_logs_and_ledger_are_git_ignored": True,
        "file_count": len(files), "files": files,
        "ledger_sha256": _sha256(LEDGER_PATH) if LEDGER_PATH.is_file() else None,
    })


def _write_report(results: dict[str, Any], detail: dict[str, Any]) -> None:
    ledger = detail["ledger"]
    counts = ledger["counts"]
    lines = [
        "# ROUND46 执行报告", "",
        f"- 状态：{results['status']}",
        f"- 分支：{ROUND46_BRANCH}",
        f"- 运行器：{ROUND46_RUNNER_VERSION}",
        f"- 训练语义版本：{TRAINING_VERSION}",
        f"- 配置冻结 SHA256：{_sha256(FROZEN_CONFIG_PATH)}",
        "",
        "## 结果",
        "",
        "每 seed greedy 保持 immediate 目标身份集合且开发块平均字节严格低于 late 规则时，才记为开发块联合改善；sampled 为辅助读数。",
    ]
    for seed in SEEDS:
        item = results["seed_results"][str(seed)]
        lines.append(
            f"- seed {seed}: greedy tasks={item['greedy_mean_on_time_task_count']:.3f}/scene, "
            f"bytes={item['greedy_mean_spent_bytes']:.3f}; sampled tasks="
            f"{item['sampled_mean_on_time_task_count']:.3f}/scene, "
            f"bytes={item['sampled_mean_spent_bytes']:.3f}; "
            f"preserves immediate IDs={item['immediate_exact_task_id_sets_preserved_greedy']}; "
            f"greedy below late={item['greedy_cost_strictly_below_late_rule']}.")
    lines.extend([
        "",
        "## 配额与执行计数",
        "",
        f"- 正常 claim：{ledger.get('normal_used', 0)}/{NORMAL_CAP}；技术储备：{ledger.get('reserve_used', 0)}/{RESERVE_CAP}；硬上限：{HARD_CAP}。",
        f"- 环境构造：{counts.get('environment_constructor_calls', 0)}；run 调用：{counts.get('run_invocation_calls', 0)}；成功返回：{counts.get('run_return_calls', 0)}；完成步数：{counts.get('completed_steps', 0)}。",
        f"- 完成逻辑任务：{counts.get('completed_logical_jobs', 0)}；PPO updates：{counts.get('ppo_update_calls', 0)}；Adam step：{counts.get('optimizer_step_calls', 0)}。",
        "",
        "## 输入与限制",
        "",
        f"- 128项训练场景顺序对三个 seed 均逐项等于 ROUND44；ROUND44 默认 JSON 序列化哈希为 {_json(KEY_PLAN_PATH)['training_scenario_order_sha256']}。ROUND45 文件中的紧凑 JSON 哈希不同，但内容顺序相同。",
        f"- ROUND45 BC-only 门禁仍为 failed，原 SHA256 {_json(INPUT_IDENTITY_PATH)['round45_control_gate_sha256']}；ROUND46 是单独冻结授权。",
        "- BC checkpoint 仅加载四个模型模块；不加载 BC optimizer/RNG；仅恢复对应 ROUND44 ep0 rollout RNG。",
        "- 训练保持冻结的奖励、时序、成本、mask、GAE、PPO、对偶和评估解码语义。",
        "- 16个开发场景已用于开发比较，不构成独立确认、正式非劣或泛化证据。结果不证明时间偏移是全部学习失败的唯一原因。",
        "- BC sampled 对照不存在（ROUND45 仅有 greedy 控制）；不补做授权外评估。",
        "- 未推送、部署或追加训练。原始模型、tape、日志和账本由本地清单按哈希登记。",
        "",
        "## 详细产物",
        "",
        f"- 最终结果：{FINAL_RESULTS_PATH.name}",
        f"- 配对比较与 bootstrap：{PAIRED_RESULTS_PATH.name}",
        f"- 逐场景目标 ID 差异：{TARGET_DIFFS_PATH.name}",
        f"- 训练曲线：{CURVES_PATH.name}",
        f"- checkpoint 与批状态索引：{CHECKPOINT_INDEX_PATH.name}",
        f"- 原始产物清单：{ARTIFACT_MANIFEST_PATH.name}",
    ])
    _write_text_lf(REPORT_PATH, "\n".join(lines) + "\n")


def execute() -> None:
    global _R2_V5_REQUIRE_EXECUTION_EVIDENCE
    if not PRELAUNCH_GATE_PATH.is_file() or not PRELAUNCH_SHA_PATH.is_file():
        raise RuntimeError("ROUND46 zero-environment preflight receipt must be committed first")
    module = _configure_runtime_module()
    runtime = _RUNTIME
    assert runtime is not None
    _R2_V5_REQUIRE_EXECUTION_EVIDENCE = True
    module.verify_prelaunch_identity = lambda _frozen=None: _verify_prelaunch_identity(
        runtime, require_r2_v5_evidence=True)
    module.ENFORCE_PRELAUNCH = True
    _verify_prelaunch_identity(runtime, require_r2_v5_evidence=True)
    _verify_status_guard_regression_receipt()
    _assert_source_worktree_clean()

    try:
        first_four = _run_first_four_and_all_gates(module, runtime)
        _continue_fixed_training(module, runtime, first_four)
        module._task_state(
            "ROUND46 three seed ep128 complete; run exactly 96 fixed final evaluation cells",
            phase="final_evaluation",
        )
        eval_rows = module._run_final_eval_matrix(runtime)
        if len(eval_rows) != EVALUATION_KEYS:
            raise RuntimeError(f"ROUND46 final evaluation returned {len(eval_rows)}/96 cells")
        runtime["r42_run"].write_execution_ledger(strict=True)
        results, detail = _build_analysis(module, runtime)
        _write_training_curves(runtime)
        _write_checkpoint_index(runtime)
        _write_artifact_manifest(runtime)
        _write_report(results, detail)
        module._task_state(
            "ROUND46 fixed plan complete",
            final_research_status=results["status"],
            final_evaluations=len(eval_rows),
            environment_constructed=detail["ledger"]["counts"]["environment_constructor_calls"],
            ppo_updates=detail["ledger"]["counts"]["ppo_update_calls"],
            optimizer_step_calls=detail["ledger"]["counts"]["optimizer_step_calls"],
        )
        print(json.dumps({
            "status": results["status"], "report": str(REPORT_PATH),
            "results": str(FINAL_RESULTS_PATH),
            "counts": detail["ledger"]["counts"],
            "budget": results["budget"],
        }, ensure_ascii=False, indent=2))
    except BaseException as error:
        _write_stop_state(error, module, runtime)
        raise


def _write_stop_state(error: BaseException, module: Any,
                      runtime: dict[str, Any]) -> None:
    ledger = {}
    source = "unavailable"
    try:
        ledger = runtime["r42_run"].ledger_snapshot(strict=False)
        source = "actual ROUND42/R40 ledger snapshot"
        runtime["r42_run"].write_execution_ledger(strict=False)
    except BaseException as ledger_error:
        ledger = {"ledger_error": f"{type(ledger_error).__name__}: {ledger_error}"}
    counts = ledger.get("counts", {})
    old_gate_preserved = True
    if INPUT_IDENTITY_PATH.exists():
        old_gate_preserved = (
            _sha256(R45_CONTROL_GATE_PATH)
            == _json(INPUT_IDENTITY_PATH).get("round45_control_gate_sha256")
        )
    state = {
        "kind": "round46-stop-state/1", "status": "stopped",
        "runner_version": ROUND46_RUNNER_VERSION,
        "training_version": TRAINING_VERSION,
        "error": {"type": type(error).__name__, "message": str(error)},
        "ledger_source": source, "ledger": ledger,
        "old_round45_gate_preserved": old_gate_preserved,
        "environment_constructed": int(counts.get("environment_constructor_calls", 0)),
        "run_invocation_calls": int(counts.get("run_invocation_calls", 0)),
        "completed_steps": int(counts.get("completed_steps", 0)),
        "ppo_updates": int(counts.get("ppo_update_calls", 0)),
        "optimizer_step_calls": int(counts.get("optimizer_step_calls", 0)),
    }
    _write_json(STOP_STATE_PATH, state)
    _write_json(CURRENT_TASK_PATH, {
        "kind": "round46-current-task/1", "stage": "stopped with evidence preserved",
        "branch": ROUND46_BRANCH, "runner_version": ROUND46_RUNNER_VERSION,
        "training_version": TRAINING_VERSION, **state,
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROUND46 frozen execution adapter; select exactly one explicit phase."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true",
                       help="verify inputs and write the code/config/key freeze; zero environments")
    group.add_argument("--preflight", action="store_true",
                       help="run the zero-environment BC/RNG/Adam/runtime identity gate")
    group.add_argument("--status-guard-regression", action="store_true",
                       help="verify mutable task-state writes before pre-claim identity checks; zero environments")
    group.add_argument("--r2-v5-text-identity-regression", action="store_true",
                       help="test raw-file and Git blob text identity namespaces; zero runtime work")
    group.add_argument("--r2-v5-binding-regression", action="store_true",
                       help="exercise the actual R42/R45/R44 claim verifier binding; zero claims/environments")
    group.add_argument("--execute", action="store_true",
                       help="run only the frozen 384 training + 96 final evaluation plan")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.preflight:
        preflight()
    elif args.status_guard_regression:
        run_status_guard_regression()
    elif args.r2_v5_text_identity_regression:
        run_r2_v5_text_identity_regression()
    elif args.r2_v5_binding_regression:
        run_r2_v5_binding_regression()
    elif args.execute:
        execute()
    else:
        raise RuntimeError("ROUND46 requires exactly one explicit mode")


if __name__ == "__main__":
    main()
