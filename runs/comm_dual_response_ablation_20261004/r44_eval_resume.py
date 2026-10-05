"""ROUND44 receipt repair and fixed evaluation continuation.

This entry point is intentionally evaluation-only.  It validates the exact
source and execution identity chain, configures the frozen R44 evaluation runtime, reads the
three already-completed ep128 models, then runs only the registered 96 cells.
It never enters the training or exact-resume runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = Path(__file__).resolve().parent
RUN_REL = f"runs/{RUN_DIR.name}"
BASE_STOP_HEAD = "12c3bdde0c85d9cf31eb1f1fc264f397701babac"
INITIAL_SOURCE_HEAD = "347da3b8b559e669c57334f51ed17bcb3e90e3b2"
SOURCE_IDENTITY_VALIDATOR_HEAD = "3fbd702c398b9cc45d3495183b0ce73ac3ded178"
SOURCE_AMBIENT_GUARD_HEAD = "b7c27b809cb7ac1fd3f6282a6c03265dfc44ea60"
PRIOR_EXECUTION_COMMIT = "1f43d869df6200f9de3e6bd7ae139e8c6e7f7c10"
RECEIPT_RESUME_SOURCE_HEAD = "c18e7d6564aac132104649dc18c3ab52acb76023"
R2_EXECUTION_IDENTITY_HEAD = "a48261e5aa1def4ddf130f1377dbdff0e4cb51d1"
SCIENTIFIC_FREEZE_HEAD = "2353c4616746cf0a7c9fcb3a519a7042a28ba176"
TRAINING_EXECUTION_HEAD = "26b22680b8c3c5cb9bb91bc44799c76cec6f874a"
BRANCH = "exp/comm-dual-response-ablation"
SEEDS = (11, 22, 33)
MODES = ("greedy", "sampled")
SCENES = tuple(range(1_301_000, 1_301_016))
RETRY_KEY = (11, "greedy", 1_301_000)
TRAINING_VERSION = (
    "constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/"
    "dual-eta-lambda-0p5/1/published-completion-event-at-relay-tick/2"
)
REPRESENTATION_KIND = "round44-strict-json-value/1"
INITIAL_SOURCE_ALLOWED_PATHS = {
    f"{RUN_REL}/ROUND44_CURRENT_TASK.json",
    f"{RUN_REL}/ROUND44_EVAL_MECHANICAL_DIFF_R1.json",
    f"{RUN_REL}/ROUND44_EVAL_RECEIPT_REPAIR_REGRESSION_R1.json",
    f"{RUN_REL}/r44_eval_index.py",
    f"{RUN_REL}/test_r44_eval_index.py",
    f"{RUN_REL}/r44_eval_resume.py",
    f"{RUN_REL}/test_r44_eval_resume.py",
}
SOURCE_REVISION_ALLOWED_PATHS = {
    f"{RUN_REL}/ROUND44_CURRENT_TASK.json",
    f"{RUN_REL}/ROUND44_EVAL_MECHANICAL_DIFF_R1.json",
    f"{RUN_REL}/ROUND44_EVAL_RECEIPT_REPAIR_REGRESSION_R1.json",
    f"{RUN_REL}/r44_eval_resume.py",
    f"{RUN_REL}/test_r44_eval_resume.py",
}
SOURCE_AMBIENT_GUARD_REVISION_ALLOWED_PATHS = set(SOURCE_REVISION_ALLOWED_PATHS)
SOURCE_RECEIPT_RESUME_REVISION_ALLOWED_PATHS = set(SOURCE_REVISION_ALLOWED_PATHS)
SOURCE_CLAIM_QUEUE_REVISION_ALLOWED_PATHS = set(SOURCE_REVISION_ALLOWED_PATHS)
SOURCE_ALLOWED_PATHS = (INITIAL_SOURCE_ALLOWED_PATHS | SOURCE_REVISION_ALLOWED_PATHS
                        | SOURCE_AMBIENT_GUARD_REVISION_ALLOWED_PATHS
                        | SOURCE_RECEIPT_RESUME_REVISION_ALLOWED_PATHS
                        | SOURCE_CLAIM_QUEUE_REVISION_ALLOWED_PATHS)
IDENTITY_ALLOWED_PATHS = {
    f"{RUN_REL}/runtime/ROUND44_EVAL_ONLY_GATE_R3.json",
    f"{RUN_REL}/runtime/ROUND44_EVAL_ONLY_EXECUTION_IDENTITY_R3.json",
}
IDENTITY_ALLOWED_PATHS_R1 = {
    f"{RUN_REL}/runtime/ROUND44_EVAL_ONLY_GATE_R1.json",
    f"{RUN_REL}/runtime/ROUND44_EVAL_ONLY_EXECUTION_IDENTITY_R1.json",
}
IDENTITY_ALLOWED_PATHS_R2 = {
    f"{RUN_REL}/runtime/ROUND44_EVAL_ONLY_GATE_R2.json",
    f"{RUN_REL}/runtime/ROUND44_EVAL_ONLY_EXECUTION_IDENTITY_R2.json",
}
IMMUTABLE_MANIFEST = RUN_DIR / "runtime" / "ROUND44_EVAL_INPUT_IMMUTABLE_MANIFEST_R1.json"
PREPATCH_INDEX = RUN_DIR / "runtime" / "ROUND44_R44_EVAL_INDEX_PREPATCH_R1.py"
OLD_FAILURE_REPRO = RUN_DIR / "runtime" / "ROUND44_OLD_RECEIPT_FAILURE_REPRODUCTION_R1.json"
READONLY_GATE = RUN_DIR / "runtime" / "ROUND44_EVAL_READONLY_MODEL_GATE_R1.json"
EVAL_GATE_R1 = RUN_DIR / "runtime" / "ROUND44_EVAL_ONLY_GATE_R1.json"
EVAL_IDENTITY_R1 = RUN_DIR / "runtime" / "ROUND44_EVAL_ONLY_EXECUTION_IDENTITY_R1.json"
EVAL_GATE = RUN_DIR / "runtime" / "ROUND44_EVAL_ONLY_GATE_R3.json"
EVAL_IDENTITY = RUN_DIR / "runtime" / "ROUND44_EVAL_ONLY_EXECUTION_IDENTITY_R3.json"
POST_RETRY_AUDIT = RUN_DIR / "runtime" / "ROUND44_EVAL_POST_RETRY_RECEIPT_AUDIT_R1.json"
RETRY_GATE = RUN_DIR / "runtime" / "ROUND44_EVAL_RETRY_GATE_R1.json"
EVAL_GATE_R2 = RUN_DIR / "runtime" / "ROUND44_EVAL_ONLY_GATE_R2.json"
EVAL_IDENTITY_R2 = RUN_DIR / "runtime" / "ROUND44_EVAL_ONLY_EXECUTION_IDENTITY_R2.json"
PRIOR_PROGRESS_R1 = RUN_DIR / "runtime" / "ROUND44_EXECUTION_PROGRESS_EVAL_R1.json"
PRIOR_PROGRESS_R2 = RUN_DIR / "runtime" / "ROUND44_EXECUTION_PROGRESS_EVAL_R2.json"
PROGRESS = RUN_DIR / "runtime" / "ROUND44_EXECUTION_PROGRESS_EVAL_R3.json"
LOCK_PATH = RUN_DIR / "runtime" / "ROUND44_EXECUTOR_LOCK.json"
TASK_PATH = RUN_DIR / "ROUND44_CURRENT_TASK.json"
PARTIAL_LEDGER = RUN_DIR / "ROUND44_PARTIAL_EXECUTION_LEDGER.json"
LIVE_LEDGER = RUN_DIR / "runtime" / "ROUND44_EXECUTION_LEDGER.json"
PRELAUNCH_RECEIPT = RUN_DIR / "runtime" / "ROUND44_PRELAUNCH_IDENTITY.json"
R4_IDENTITY = RUN_DIR / "runtime" / "ROUND44_RESUME_EXECUTION_IDENTITY_R4.json"
R4_READY = RUN_DIR / "runtime" / "ROUND44_RESUME_READY_R4.json"
R4_GATE = RUN_DIR / "runtime" / "ROUND44_EXACT_RESUME_GATE_R4.json"
R4_COMPARATOR = RUN_DIR / "runtime" / "ROUND44_COMPARATOR_REGRESSION_R4.json"
PARAMETER_SHA = {
    11: "a0e9e60a5f2ed040b9f40963a9c9129c84afcf71b5c39d3cad91ff78a6d2323c",
    22: "ce45ad13794d6d39691dfe872f29fcec83a20b2fd5bfb032eac30aa22b6993fd",
    33: "3af3f52dd0c293b2d71f9d5191d4b9d3f4641a8eb90cba4ee19b93744908c5a2",
}
EP128_FILE_SHA = {
    11: "7b08115cc286df69fdf16d7fc8e4b170d61f94e343f1caecd5d4b4191b727205",
    22: "22a7df79697dd18d465cc0c9e9d6a292f2251bed980a0883ec18c02bb5ab0268",
    33: "03dbdb0c6e021b97d1d67824271eda6703f9389aad02f0f3f3bdb8b9892b7b63",
}


class EvalResumeError(RuntimeError):
    """A frozen identity, receipt, or eval-only condition failed closed."""


class EvalOnlyViolation(EvalResumeError):
    """A training, optimizer, checkpoint-write, or training-restore hook was called."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvalResumeError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise EvalResumeError(f"non-standard JSON number is not allowed: {value}")


def _validate_json_finite(value: Any, path: str = "$") -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise EvalResumeError(f"non-finite JSON number at {path}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_finite(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise EvalResumeError(f"JSON key at {path} must be a string")
            _validate_json_finite(item, f"{path}.{key}")
        return
    raise EvalResumeError(f"unsupported persisted JSON value at {path}: {type(value).__name__}")


def read_json(path: Path) -> Any:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"),
                           object_pairs_hook=_strict_object,
                           parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvalResumeError(f"cannot read strict JSON: {path}") from error
    _validate_json_finite(value, str(path))
    return value


def _git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                            capture_output=True, text=not binary)
    return result.stdout if binary else result.stdout.strip()


def _parent(commit: str) -> str:
    line = str(_git("rev-list", "--parents", "-n", "1", commit)).split()
    if len(line) != 2:
        raise EvalResumeError(f"commit must have exactly one parent: {commit}")
    return line[1]


def _changed_paths(left: str, right: str) -> set[str]:
    raw = str(_git("diff", "--name-only", left, right))
    return {line.replace("\\", "/") for line in raw.splitlines() if line}


def _blob_sha(commit: str, repo_relative: str) -> str:
    value = _git("show", f"{commit}:{repo_relative}", binary=True)
    assert isinstance(value, bytes)
    return sha256_bytes(value)


def _assert_ancestor(parent: str, child: str) -> None:
    result = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                             parent, child], capture_output=True)
    if result.returncode != 0:
        raise EvalResumeError(f"required commit is not an ancestor: {parent} -> {child}")


def evaluation_schedule() -> tuple[tuple[int, str, int], ...]:
    return tuple((seed, mode, scene) for seed in SEEDS for mode in MODES for scene in SCENES)


def validate_eval_claim(logical_job: str, attempt: int,
                        metadata: dict[str, Any]) -> tuple[tuple[int, str, int], str]:
    match = re.fullmatch(
        r"model_eval_new_s(11|22|33)_(greedy|sampled)_scene(13010\d{2})", logical_job)
    if match is None:
        raise EvalResumeError(f"eval-only runner rejected non-matrix logical job: {logical_job}")
    key = (int(match.group(1)), match.group(2), int(match.group(3)))
    if key not in evaluation_schedule():
        raise EvalResumeError(f"eval-only runner rejected foreign key: {key}")
    expected_metadata = {"arm": "NEW", "training_seed": key[0],
                         "mode": key[1], "scenario_seed": key[2]}
    if metadata != expected_metadata:
        raise EvalResumeError(f"eval-only runner rejected claim metadata mismatch: {key}")
    expected_attempt = 2 if key == RETRY_KEY else 1
    expected_category = "reserve" if key == RETRY_KEY else "normal"
    if type(attempt) is not int or attempt != expected_attempt:
        raise EvalResumeError(
            f"eval-only runner rejected attempt {attempt} for {key}; expected {expected_attempt}")
    return key, expected_category


def _forbid(action: str) -> Callable[..., Any]:
    def rejected(*_args: Any, **_kwargs: Any) -> Any:
        raise EvalOnlyViolation(f"ROUND44 eval-only guard rejected {action}")
    return rejected


def _rng_state_sha256(value: Any) -> str:
    if type(value) is not dict or set(value) != {"python", "numpy", "torch"}:
        raise EvalResumeError("ambient RNG snapshot has an unexpected shape")
    _validate_json_finite(value)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return sha256_bytes(encoded)


def _eval_rows_equivalent(in_memory: dict[str, Any], durable: dict[str, Any]) -> bool:
    """Compare a runner row to its JSON receipt under the strict persisted-value contract."""
    from r44_eval_index import _json_value_equal

    return _json_value_equal(in_memory, durable)


def install_read_only_guards(run_module: Any, runtime: dict[str, Any],
                             torch_module: Any) -> list[str]:
    """Make unauthorized training/update/restore/write operations fail immediately."""
    guarded: list[str] = []

    def patch(owner: Any, name: str, action: str) -> None:
        if hasattr(owner, name):
            setattr(owner, name, _forbid(action))
            guarded.append(action)

    for name in ("train_seed", "run_training_episode", "run_reference", "run_rule"):
        patch(run_module, name, f"r42_run.{name}")
    trainer = runtime["trainer"]
    for name, action in (
        ("ppo_update", "trainer.ppo_update"),
        ("save_checkpoint", "trainer.save_checkpoint"),
        ("load_checkpoint", "trainer.load_checkpoint_training_state"),
    ):
        patch(trainer, name, action)
    for name in ("restore_rng", "rng_state", "load_checkpoint_read_only"):
        if not callable(getattr(trainer, name, None)):
            raise EvalResumeError(f"required read-only trainer boundary is missing: {name}")
    original_restore_rng = trainer.restore_rng
    original_read_only_loader = trainer.load_checkpoint_read_only
    rng_guard_state: dict[str, Any] = {
        "active": False, "expected_ambient_sha256": None,
        "restore_calls_in_loader": 0, "read_only_load_calls": 0,
        "ambient_restore_calls": 0,
    }

    def guarded_restore_rng(state: Any) -> Any:
        if not rng_guard_state["active"]:
            raise EvalOnlyViolation(
                "ROUND44 eval-only guard rejected trainer.restore_rng outside the read-only loader")
        if int(rng_guard_state["restore_calls_in_loader"]) != 0:
            raise EvalOnlyViolation("ROUND44 read-only loader attempted multiple RNG restores")
        if _rng_state_sha256(state) != rng_guard_state["expected_ambient_sha256"]:
            raise EvalOnlyViolation(
                "ROUND44 read-only loader attempted to restore non-ambient/checkpoint RNG state")
        result = original_restore_rng(state)
        if result is not True:
            raise EvalOnlyViolation("ROUND44 ambient RNG preservation did not restore a valid snapshot")
        rng_guard_state["restore_calls_in_loader"] = 1
        rng_guard_state["ambient_restore_calls"] += 1
        return result

    def guarded_read_only_loader(*args: Any, **kwargs: Any) -> Any:
        if rng_guard_state["active"]:
            raise EvalOnlyViolation("nested read-only checkpoint loads are forbidden")
        ambient_sha = _rng_state_sha256(trainer.rng_state())
        rng_guard_state["active"] = True
        rng_guard_state["expected_ambient_sha256"] = ambient_sha
        rng_guard_state["restore_calls_in_loader"] = 0
        try:
            result = original_read_only_loader(*args, **kwargs)
            if int(rng_guard_state["restore_calls_in_loader"]) != 1:
                raise EvalOnlyViolation(
                    "read-only checkpoint loader did not restore its exact ambient RNG snapshot")
            if _rng_state_sha256(trainer.rng_state()) != ambient_sha:
                raise EvalOnlyViolation("read-only checkpoint load changed ambient RNG state")
            rng_guard_state["read_only_load_calls"] += 1
            return result
        finally:
            rng_guard_state["active"] = False
            rng_guard_state["expected_ambient_sha256"] = None
            rng_guard_state["restore_calls_in_loader"] = 0

    trainer.restore_rng = guarded_restore_rng
    trainer.load_checkpoint_read_only = guarded_read_only_loader
    guarded.extend(("trainer.restore_rng_checkpoint_state_rejected",
                    "trainer.load_checkpoint_read_only_ambient_preservation_guard"))
    runtime["eval_only_rng_guard_state"] = rng_guard_state
    r40_train = runtime.get("r40_train")
    if r40_train is not None:
        for name in ("train_seed", "_save_milestone", "_write_immutable_snapshot",
                     "_save_preupdate_checkpoint"):
            patch(r40_train, name, f"r40_train.{name}")
    patch(torch_module, "save", "torch.save")
    optim = getattr(torch_module, "optim", None)
    if optim is not None:
        patch(getattr(optim, "Optimizer", object()), "step", "Optimizer.step")
        patch(getattr(optim, "Adam", object()), "step", "Adam.step")
    if not all(name in guarded for name in (
            "r42_run.train_seed", "r42_run.run_training_episode", "trainer.ppo_update",
            "trainer.save_checkpoint", "trainer.load_checkpoint_training_state",
            "trainer.restore_rng_checkpoint_state_rejected",
            "trainer.load_checkpoint_read_only_ambient_preservation_guard",
            "torch.save", "Adam.step")):
        raise EvalResumeError(f"required eval-only rejection hook was not installed: {guarded}")
    return guarded


def _write_new_or_match(path: Path, value: Any) -> None:
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise EvalResumeError(f"immutable eval-only evidence already exists and differs: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def _write_atomic(path: Path, value: Any) -> None:
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _acquire_lock(head: str) -> Path:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"kind": "round44-eval-only-lock/1", "pid": os.getpid(),
                       "head": head, "started_at_unix": time.time()},
                      ensure_ascii=False, indent=2) + "\n"
    descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    return LOCK_PATH


def _release_lock(path: Path) -> None:
    try:
        lock = read_json(path)
        if int(lock.get("pid", -1)) == os.getpid():
            path.unlink()
    except FileNotFoundError:
        pass


def _verify_r4_history(source_or_identity_head: str) -> dict[str, Any]:
    identity = read_json(R4_IDENTITY)
    ready = read_json(R4_READY)
    gate = read_json(R4_GATE)
    comparator = read_json(R4_COMPARATOR)
    hashes = {path.name: sha256_file(path) for path in
              (R4_IDENTITY, R4_READY, R4_GATE, R4_COMPARATOR)}
    if identity.get("status") != "passed" or identity.get("head") != TRAINING_EXECUTION_HEAD:
        raise EvalResumeError("historical R4 training identity is not the registered passed state")
    if ready.get("status") != "passed" or ready.get("execution_head") != TRAINING_EXECUTION_HEAD:
        raise EvalResumeError("historical R4 ready receipt is not the registered passed state")
    if ready.get("execution_identity_sha256") != hashes[R4_IDENTITY.name]:
        raise EvalResumeError("R4 ready receipt no longer binds its original identity file")
    if gate.get("status") != "passed" or comparator.get("status") != "passed":
        raise EvalResumeError("historical R4 exact-resume/comparator gate evidence is not passed")
    if (_parent(TRAINING_EXECUTION_HEAD) != identity.get("r3_parent_head")
            or identity.get("no_scientific_configuration_change") is not True):
        raise EvalResumeError("R4 execution head/registered mechanical continuation identity changed")
    _assert_ancestor(SCIENTIFIC_FREEZE_HEAD, TRAINING_EXECUTION_HEAD)
    _assert_ancestor(TRAINING_EXECUTION_HEAD, source_or_identity_head)
    prelaunch = read_json(PRELAUNCH_RECEIPT)
    if (prelaunch.get("status") != "passed"
            or prelaunch.get("head") != SCIENTIFIC_FREEZE_HEAD
            or prelaunch.get("branch") != BRANCH):
        raise EvalResumeError("original ROUND44 scientific prelaunch receipt changed")
    r4_source_hashes = identity.get("source_code_sha256", {})
    if not isinstance(r4_source_hashes, dict):
        raise EvalResumeError("historical R4 identity lacks its source hash map")
    checked_r4_source_hashes: dict[str, str] = {}
    r4_source_differences: set[str] = set()
    for relative, expected in prelaunch.get("source_code_sha256", {}).items():
        repo_path = f"{RUN_REL}/{relative}"
        frozen_blob = _blob_sha(SCIENTIFIC_FREEZE_HEAD, repo_path)
        if frozen_blob != expected:
            raise EvalResumeError(
                f"original ROUND44 freeze blob does not match its prelaunch receipt: {relative}")
        r4_expected = r4_source_hashes.get(relative)
        if not isinstance(r4_expected, str):
            raise EvalResumeError(f"R4 source identity omits a prelaunch source: {relative}")
        r4_blob = _blob_sha(TRAINING_EXECUTION_HEAD, repo_path)
        if r4_blob != r4_expected:
            raise EvalResumeError(f"R4 committed source blob differs from its identity: {relative}")
        if expected != r4_expected:
            r4_source_differences.add(relative)
        if relative in {"r44_eval_index.py", "test_r44_eval_index.py"}:
            # These two files are intentionally versioned in the new receipt repair.
            if r4_expected != expected:
                raise EvalResumeError(f"historical R4 index baseline changed: {relative}")
            if relative == "r44_eval_index.py" and sha256_file(PREPATCH_INDEX) != expected:
                raise EvalResumeError("preserved prepatch receipt/index source hash changed")
            checked_r4_source_hashes[relative] = r4_expected
            continue
        path = RUN_DIR / relative
        if not path.is_file() or sha256_file(path) != r4_expected:
            raise EvalResumeError(f"R4 execution source differs from its committed identity: {relative}")
        checked_r4_source_hashes[relative] = r4_expected
    expected_r4_mechanical_paths = {"r44_execute.py", "r44_runtime.py"}
    if r4_source_differences != expected_r4_mechanical_paths:
        raise EvalResumeError(
            "R4 source delta from the original freeze is not the registered mechanical pair: "
            f"{sorted(r4_source_differences)}")
    diff_value = _git(
        "diff", "--binary", "--no-ext-diff", SCIENTIFIC_FREEZE_HEAD,
        TRAINING_EXECUTION_HEAD, "--", f"{RUN_REL}/r44_execute.py",
        f"{RUN_REL}/r44_runtime.py", binary=True)
    assert isinstance(diff_value, bytes)
    return {"r4_identity_sha256": hashes[R4_IDENTITY.name],
            "r4_ready_sha256": hashes[R4_READY.name],
            "r4_exact_gate_sha256": hashes[R4_GATE.name],
            "r4_comparator_sha256": hashes[R4_COMPARATOR.name],
            "original_prelaunch_identity_sha256": sha256_file(PRELAUNCH_RECEIPT),
            "original_freeze_source_count": len(prelaunch.get("source_code_sha256", {})),
            "r4_execution_source_sha256": checked_r4_source_hashes,
            "r4_registered_mechanical_source_delta": {
                "paths": sorted(r4_source_differences),
                "git_diff_sha256": sha256_bytes(diff_value),
                "training_execution_head": TRAINING_EXECUTION_HEAD,
                "head_parent": _parent(TRAINING_EXECUTION_HEAD)}}


def _verify_receipt_resume_source_chain() -> dict[str, Any]:
    """Revalidate the preserved R1 -> receipt-resume source segment exactly."""
    if _parent(INITIAL_SOURCE_HEAD) != BASE_STOP_HEAD:
        raise EvalResumeError("initial eval-only source commit no longer directly follows the stop head")
    if _changed_paths(BASE_STOP_HEAD, INITIAL_SOURCE_HEAD) != INITIAL_SOURCE_ALLOWED_PATHS:
        raise EvalResumeError("initial eval-only source commit differs from its exact path allowlist")
    if _parent(SOURCE_IDENTITY_VALIDATOR_HEAD) != INITIAL_SOURCE_HEAD:
        raise EvalResumeError("source identity-validator revision no longer directly follows its preserved parent")
    if _changed_paths(INITIAL_SOURCE_HEAD, SOURCE_IDENTITY_VALIDATOR_HEAD) != SOURCE_REVISION_ALLOWED_PATHS:
        raise EvalResumeError("source identity-validator revision differs from its exact path allowlist")
    if _parent(SOURCE_AMBIENT_GUARD_HEAD) != SOURCE_IDENTITY_VALIDATOR_HEAD:
        raise EvalResumeError("ambient-guard source no longer directly follows the identity-validator commit")
    if _changed_paths(SOURCE_IDENTITY_VALIDATOR_HEAD, SOURCE_AMBIENT_GUARD_HEAD) != SOURCE_AMBIENT_GUARD_REVISION_ALLOWED_PATHS:
        raise EvalResumeError("ambient-guard source differs from its exact path allowlist")
    if _parent(PRIOR_EXECUTION_COMMIT) != SOURCE_AMBIENT_GUARD_HEAD:
        raise EvalResumeError("prior eval identity no longer directly follows the ambient-guard source")
    if _changed_paths(SOURCE_AMBIENT_GUARD_HEAD, PRIOR_EXECUTION_COMMIT) != IDENTITY_ALLOWED_PATHS_R1:
        raise EvalResumeError("prior eval identity commit differs from its exact R1 path allowlist")
    if _parent(RECEIPT_RESUME_SOURCE_HEAD) != PRIOR_EXECUTION_COMMIT:
        raise EvalResumeError("receipt-resume revision no longer directly follows the preserved R1 identity")
    if _changed_paths(PRIOR_EXECUTION_COMMIT, RECEIPT_RESUME_SOURCE_HEAD) != SOURCE_RECEIPT_RESUME_REVISION_ALLOWED_PATHS:
        raise EvalResumeError("receipt-resume revision differs from its exact path allowlist")
    _assert_ancestor(SCIENTIFIC_FREEZE_HEAD, RECEIPT_RESUME_SOURCE_HEAD)
    _assert_ancestor(TRAINING_EXECUTION_HEAD, RECEIPT_RESUME_SOURCE_HEAD)
    hashes = {}
    for relative in SOURCE_ALLOWED_PATHS:
        hashes[relative] = _blob_sha(RECEIPT_RESUME_SOURCE_HEAD, relative)
    return {
        "base_stop_head": BASE_STOP_HEAD,
        "initial_source_head": INITIAL_SOURCE_HEAD,
        "source_identity_validator_head": SOURCE_IDENTITY_VALIDATOR_HEAD,
        "ambient_guard_head": SOURCE_AMBIENT_GUARD_HEAD,
        "prior_execution_commit": PRIOR_EXECUTION_COMMIT,
        "revised_source_head": RECEIPT_RESUME_SOURCE_HEAD,
        "initial_source_allowed_paths": sorted(INITIAL_SOURCE_ALLOWED_PATHS),
        "source_identity_validator_revision_allowed_paths": sorted(SOURCE_REVISION_ALLOWED_PATHS),
        "ambient_guard_revision_allowed_paths": sorted(SOURCE_AMBIENT_GUARD_REVISION_ALLOWED_PATHS),
        "receipt_resume_revision_allowed_paths": sorted(SOURCE_RECEIPT_RESUME_REVISION_ALLOWED_PATHS),
        "source_file_sha256": hashes,
    }


def _verify_source_commit_chain(source_head: str) -> dict[str, Any]:
    receipt_evidence = _verify_receipt_resume_source_chain()
    if _parent(R2_EXECUTION_IDENTITY_HEAD) != RECEIPT_RESUME_SOURCE_HEAD:
        raise EvalResumeError("preserved R2 identity no longer directly follows receipt-resume source")
    if _changed_paths(RECEIPT_RESUME_SOURCE_HEAD, R2_EXECUTION_IDENTITY_HEAD) != IDENTITY_ALLOWED_PATHS_R2:
        raise EvalResumeError("preserved R2 identity commit differs from its exact two-path allowlist")
    if _parent(source_head) != R2_EXECUTION_IDENTITY_HEAD:
        raise EvalResumeError("claim-queue source revision must directly follow the preserved R2 identity")
    if _changed_paths(R2_EXECUTION_IDENTITY_HEAD, source_head) != SOURCE_CLAIM_QUEUE_REVISION_ALLOWED_PATHS:
        raise EvalResumeError("claim-queue source revision differs from its exact path allowlist")
    _assert_ancestor(SCIENTIFIC_FREEZE_HEAD, source_head)
    _assert_ancestor(TRAINING_EXECUTION_HEAD, source_head)
    hashes = {relative: _blob_sha(source_head, relative) for relative in SOURCE_ALLOWED_PATHS}
    return {
        **receipt_evidence,
        "receipt_resume_source_head": RECEIPT_RESUME_SOURCE_HEAD,
        "r2_identity_head": R2_EXECUTION_IDENTITY_HEAD,
        "r2_identity_allowed_paths": sorted(IDENTITY_ALLOWED_PATHS_R2),
        "claim_queue_source_parent": R2_EXECUTION_IDENTITY_HEAD,
        "claim_queue_revision_allowed_paths": sorted(SOURCE_CLAIM_QUEUE_REVISION_ALLOWED_PATHS),
        "revised_source_head": source_head,
        "source_file_sha256": hashes,
    }


def _assert_source_commit(*, expected_head: str | None = None) -> str:
    branch = str(_git("branch", "--show-current"))
    if branch != BRANCH:
        raise EvalResumeError(f"eval-only identity branch mismatch: {branch}")
    if str(_git("status", "--porcelain", "--untracked-files=all")):
        raise EvalResumeError("tracked/untracked worktree is dirty at an eval-only freeze boundary")
    head = str(_git("rev-parse", "HEAD"))
    if expected_head is not None and head != expected_head:
        raise EvalResumeError(f"eval-only source head changed: expected {expected_head}, got {head}")
    _verify_source_commit_chain(head)
    _verify_r4_history(head)
    return head


def _identity_paths() -> tuple[str, str]:
    return (f"{RUN_REL}/runtime/ROUND44_EVAL_ONLY_GATE_R3.json",
            f"{RUN_REL}/runtime/ROUND44_EVAL_ONLY_EXECUTION_IDENTITY_R3.json")


def _verify_preserved_r2_identity() -> dict[str, Any]:
    """Verify the committed R2 gate/identity and record its untouched progress state."""
    gate_sha = sha256_file(EVAL_GATE_R2)
    identity_sha = sha256_file(EVAL_IDENTITY_R2)
    progress_sha = sha256_file(PRIOR_PROGRESS_R2) if PRIOR_PROGRESS_R2.is_file() else None
    for relative, working_sha in (
            (f"{RUN_REL}/runtime/ROUND44_EVAL_ONLY_GATE_R2.json", gate_sha),
            (f"{RUN_REL}/runtime/ROUND44_EVAL_ONLY_EXECUTION_IDENTITY_R2.json", identity_sha)):
        if working_sha != _blob_sha(R2_EXECUTION_IDENTITY_HEAD, relative):
            raise EvalResumeError(f"preserved R2 identity artifact differs from its commit: {relative}")

    r2_gate = read_json(EVAL_GATE_R2)
    r2_identity = read_json(EVAL_IDENTITY_R2)
    receipt_evidence = _verify_receipt_resume_source_chain()
    expected_chain = [BASE_STOP_HEAD, INITIAL_SOURCE_HEAD,
                      SOURCE_IDENTITY_VALIDATOR_HEAD, SOURCE_AMBIENT_GUARD_HEAD,
                      PRIOR_EXECUTION_COMMIT, RECEIPT_RESUME_SOURCE_HEAD]
    if (r2_identity.get("kind") != "round44-eval-only-execution-identity/1"
            or r2_identity.get("status") != "passed"
            or r2_identity.get("branch") != BRANCH
            or r2_identity.get("source_commit") != RECEIPT_RESUME_SOURCE_HEAD
            or r2_identity.get("source_commit_chain") != expected_chain
            or r2_identity.get("source_commit_chain_evidence") != receipt_evidence
            or r2_identity.get("source_file_sha256") != receipt_evidence["source_file_sha256"]
            or r2_identity.get("gate_sha256") != gate_sha
            or r2_identity.get("readonly_model_gate_sha256") != sha256_file(READONLY_GATE)
            or r2_identity.get("fixed_eval_keys_sha256")
            != "cf7850cd3ee31905de75e80af5c80b7f64b011e20e3485897555f38e93a52053"
            or r2_identity.get("post_retry_receipt_audit_sha256") != sha256_file(POST_RETRY_AUDIT)
            or r2_identity.get("attempt2_claim_sha256")
            != "9acc52b91428fce0c9ccc7478b7df43a324b6664cac11918e14caf23ccb70eda"
            or r2_identity.get("attempt2_row_sha256")
            != "76d8e3fb4896ed9464cbc4d3b5608c1b25294ce3305c22b0cbbf4a481a04164c"):
        raise EvalResumeError("preserved R2 identity fields or exact receipt bindings changed")
    if (r2_gate.get("kind") != "round44-eval-only-prelaunch-gate/1"
            or r2_gate.get("status") != "passed"
            or r2_gate.get("source_commit") != RECEIPT_RESUME_SOURCE_HEAD
            or r2_gate.get("readonly_model_gate_sha256") != r2_identity.get(
                "readonly_model_gate_sha256")
            or r2_gate.get("fixed_eval_keys_sha256") != r2_identity.get(
                "fixed_eval_keys_sha256")
            or r2_gate.get("quota") != {"normal": 480, "reserve": 24, "hard_cap": 504}
            or r2_gate.get("post_retry_receipt_audit_sha256")
            != r2_identity.get("post_retry_receipt_audit_sha256")
            or r2_gate.get("prior_execution_commit") != PRIOR_EXECUTION_COMMIT
            or r2_gate.get("prior_execution_identity_sha256") != sha256_file(EVAL_IDENTITY_R1)
            or r2_gate.get("prior_execution_gate_sha256") != sha256_file(EVAL_GATE_R1)
            or r2_gate.get("prior_progress_sha256") != sha256_file(PRIOR_PROGRESS_R1)):
        raise EvalResumeError("preserved R2 gate no longer matches its identity and R1 evidence")
    return {
        "commit": R2_EXECUTION_IDENTITY_HEAD,
        "identity_sha256": identity_sha,
        "gate_sha256": gate_sha,
        "progress_sha256": progress_sha,
        "progress_exists": progress_sha is not None,
    }


def verify_eval_identity() -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Enforce exact eval-only child commit and all bound source/input receipts."""
    branch = str(_git("branch", "--show-current"))
    if branch != BRANCH:
        raise EvalResumeError(f"eval-only identity branch mismatch: {branch}")
    if str(_git("status", "--porcelain", "--untracked-files=all")):
        raise EvalResumeError("worktree is dirty after eval-only freeze; refusing claims")
    head = str(_git("rev-parse", "HEAD"))
    identity = read_json(EVAL_IDENTITY)
    gate = read_json(EVAL_GATE)
    source_head = identity.get("source_commit")
    source_chain_evidence = _verify_source_commit_chain(source_head)
    r2_evidence = _verify_preserved_r2_identity()
    if identity.get("kind") != "round44-eval-only-execution-identity/1" \
            or identity.get("status") != "passed" or identity.get("branch") != BRANCH:
        raise EvalResumeError("eval-only execution identity is malformed or not passed")
    if identity.get("gate_sha256") != sha256_file(EVAL_GATE):
        raise EvalResumeError("eval-only identity does not bind this exact prelaunch gate")
    if (identity.get("prior_r2_identity_commit") != r2_evidence["commit"]
            or identity.get("prior_r2_identity_sha256") != r2_evidence["identity_sha256"]
            or identity.get("prior_r2_gate_sha256") != r2_evidence["gate_sha256"]
            or identity.get("prior_r2_progress_sha256") != r2_evidence["progress_sha256"]
            or gate.get("prior_r2_identity_commit") != r2_evidence["commit"]
            or gate.get("prior_r2_identity_sha256") != r2_evidence["identity_sha256"]
            or gate.get("prior_r2_gate_sha256") != r2_evidence["gate_sha256"]
            or gate.get("prior_r2_progress_sha256") != r2_evidence["progress_sha256"]):
        raise EvalResumeError("R3 gate/identity does not bind the preserved R2 identity and progress state")
    expected_chain = [BASE_STOP_HEAD, INITIAL_SOURCE_HEAD,
                      SOURCE_IDENTITY_VALIDATOR_HEAD, SOURCE_AMBIENT_GUARD_HEAD,
                      PRIOR_EXECUTION_COMMIT, RECEIPT_RESUME_SOURCE_HEAD,
                      R2_EXECUTION_IDENTITY_HEAD, source_head]
    if identity.get("source_commit_chain") != expected_chain:
        raise EvalResumeError("eval-only identity does not bind the exact mechanical source commit chain")
    if _parent(head) != source_head or _changed_paths(source_head, head) != IDENTITY_ALLOWED_PATHS:
        raise EvalResumeError("eval-only identity commit parent/path whitelist mismatch")
    if gate.get("kind") != "round44-eval-only-prelaunch-gate/1" \
            or gate.get("status") != "passed" or gate.get("source_commit") != source_head:
        raise EvalResumeError("eval-only prelaunch gate does not match its source commit")
    if (gate.get("readonly_model_gate_sha256") != identity.get("readonly_model_gate_sha256")
            or gate.get("fixed_eval_keys_sha256") != identity.get("fixed_eval_keys_sha256")
            or gate.get("quota") != {"normal": 480, "reserve": 24, "hard_cap": 504}):
        raise EvalResumeError("prelaunch gate does not match model/plan/quota identity")
    if (gate.get("post_retry_receipt_audit_sha256")
            != identity.get("post_retry_receipt_audit_sha256")
            or gate.get("prior_execution_commit") != identity.get("prior_execution_commit")
            or gate.get("prior_execution_identity_sha256")
            != identity.get("prior_execution_identity_sha256")
            or gate.get("prior_execution_gate_sha256")
            != identity.get("prior_execution_gate_sha256")
            or gate.get("prior_progress_sha256") != identity.get("prior_progress_sha256")
            or gate.get("read_only_model_gate_reused_from_source_commit")
            != SOURCE_AMBIENT_GUARD_HEAD
            or identity.get("post_retry_receipt_audit_sha256") != sha256_file(POST_RETRY_AUDIT)):
        raise EvalResumeError("eval-only identity does not bind the completed retry receipt audit")
    if (identity.get("prior_execution_identity_sha256") != sha256_file(EVAL_IDENTITY_R1)
            or identity.get("prior_execution_gate_sha256") != sha256_file(EVAL_GATE_R1)
            or identity.get("prior_progress_sha256") != sha256_file(PRIOR_PROGRESS_R1)):
        raise EvalResumeError("retry-resume identity does not preserve the prior execution attempt")
    prior_identity = read_json(EVAL_IDENTITY_R1)
    prior_gate = read_json(EVAL_GATE_R1)
    prior_progress = read_json(PRIOR_PROGRESS_R1)
    post_retry_audit = read_json(POST_RETRY_AUDIT)
    readonly_model_gate = read_json(READONLY_GATE)
    for relative in IDENTITY_ALLOWED_PATHS_R1:
        if (sha256_file(ROOT / Path(relative))
                != _blob_sha(PRIOR_EXECUTION_COMMIT, relative)):
            raise EvalResumeError(f"preserved R1 identity artifact differs from its commit: {relative}")
    if (prior_identity.get("source_commit") != SOURCE_AMBIENT_GUARD_HEAD
            or prior_identity.get("status") != "passed"
            or prior_gate.get("status") != "passed"
            or prior_gate.get("source_commit") != SOURCE_AMBIENT_GUARD_HEAD
            or prior_identity.get("source_commit_chain") != [
                BASE_STOP_HEAD, INITIAL_SOURCE_HEAD, SOURCE_IDENTITY_VALIDATOR_HEAD,
                SOURCE_AMBIENT_GUARD_HEAD]
            or prior_progress.get("prelaunch_head") != identity.get("prior_execution_commit")
            or prior_progress.get("status") != "running_retry_gate"
            or prior_progress.get("retry_gate_passed") is not False
            or prior_progress.get("completed_eval_keys") != []
            or prior_progress.get("evaluations_completed") != 0
            or identity.get("prior_execution_commit") != PRIOR_EXECUTION_COMMIT
            or readonly_model_gate.get("status") != "passed"
            or readonly_model_gate.get("source_or_execution_head") != SOURCE_AMBIENT_GUARD_HEAD
            or post_retry_audit.get("status") != "passed"
            or post_retry_audit.get("attempt2", {}).get("claim_status") != "complete"
            or post_retry_audit.get("attempt2", {}).get("eval_index_completed_count") != 1
            or post_retry_audit.get("old_attempt1", {}).get("unchanged") is not True
            or post_retry_audit.get("training_immutability", {}).get(
                "all_204_checkpoint_hashes_unchanged") is not True
            or post_retry_audit.get("training_immutability", {}).get(
                "all_training_journal_hashes_unchanged") is not True
            or identity.get("attempt2_claim_sha256")
            != post_retry_audit.get("failure", {}).get("attempt2_claim_sha256")
            or identity.get("attempt2_row_sha256")
            != post_retry_audit.get("failure", {}).get("attempt2_row_sha256")
            or post_retry_audit.get("failure", {}).get("execution_identity_head")
            != identity.get("prior_execution_commit")):
        raise EvalResumeError("preserved attempt2 evidence does not match the retry-resume identity")
    for relative in IDENTITY_ALLOWED_PATHS:
        working_sha = sha256_file(ROOT / Path(relative))
        if working_sha != _blob_sha(head, relative):
            raise EvalResumeError(f"committed eval identity artifact changed: {relative}")
    if identity.get("source_commit_chain_evidence") != source_chain_evidence:
        raise EvalResumeError("eval-only identity source chain blob evidence changed")
    for relative, expected in identity.get("source_file_sha256", {}).items():
        if sha256_file(ROOT / Path(relative)) != expected \
                or _blob_sha(source_head, relative) != expected:
            raise EvalResumeError(f"eval-only source file does not match frozen blob: {relative}")
    manifest_sha = sha256_file(IMMUTABLE_MANIFEST)
    regression_path = RUN_DIR / "ROUND44_EVAL_RECEIPT_REPAIR_REGRESSION_R1.json"
    diff_path = RUN_DIR / "ROUND44_EVAL_MECHANICAL_DIFF_R1.json"
    if (manifest_sha != identity.get("immutable_input_manifest_sha256")
            or sha256_file(regression_path) != identity.get("receipt_regression_sha256")
            or sha256_file(diff_path) != identity.get("mechanical_diff_sha256")
            or sha256_file(READONLY_GATE) != identity.get("readonly_model_gate_sha256")):
        raise EvalResumeError("eval-only identity's immutable input or prelaunch evidence changed")
    history = _verify_r4_history(source_head)
    if history != identity.get("preserved_r4_history_hashes"):
        raise EvalResumeError("historical R4 identity evidence changed after eval-only freeze")
    if identity.get("retry_key") != [11, "greedy", 1_301_000] \
            or identity.get("schedule") != [[s, m, scene] for s, m, scene in evaluation_schedule()]:
        raise EvalResumeError("frozen evaluation key list or retry key changed")
    if identity.get("counts_before_r3_resume") != {
            "normal_used": 385, "reserve_used": 1, "completed_training_jobs": 384,
            "valid_new_eval_cells": 1, "optimizer_step_calls": 1016,
            "ppo_update_calls": 96, "checkpoint_write_calls": 297}:
        raise EvalResumeError("R3 resume count snapshot differs from the authorized post-retry state")
    return head, identity, gate


def verify_prelaunch_source_gate() -> tuple[str, dict[str, Any]]:
    """Validate the first source commit before the no-environment read-only gate."""
    head = _assert_source_commit()
    task = read_json(TASK_PATH)
    if (task.get("current_state", {}).get("counts", {}).get("normal_used") != 385
            or task.get("current_quota", {}).get("normal_remaining") != 95
            or task.get("current_quota", {}).get("reserve_remaining") != 24):
        raise EvalResumeError("current task entry does not contain the live 385/95/24 quota")
    manifest = read_json(IMMUTABLE_MANIFEST)
    manifest_sha = sha256_file(IMMUTABLE_MANIFEST)
    if (manifest_sha != "b3bb6f7c2488b2acb03f467c799b6eb74554795cf9041eb30529031774a3ebeb"
            or manifest.get("training_rows", {}).get("count") != 384
            or manifest.get("batch_commits", {}).get("count") != 96
            or manifest.get("checkpoint_pt_files", {}).get("count") != 204
            or manifest.get("verified_reuse_rows", {}).get("count") != 192
            or len(manifest.get("verified_r43_ep128_old_checkpoints", {})) != 3):
        raise EvalResumeError("immutable input manifest hash/counts differ from the verified snapshot")
    return head, manifest


def _counter_snapshot(ledger: dict[str, Any]) -> dict[str, int]:
    counts = ledger["counts"]
    return {key: int(counts[key]) for key in (
        "normal_used", "reserve_used", "total_used", "environment_constructor_calls",
        "run_invocation_calls", "completed_steps", "completed_logical_jobs",
        "technical_failed_claims", "retry_claims", "optimizer_step_calls",
        "ppo_update_calls", "checkpoint_write_calls")}


def _verify_snapshot_inputs(manifest: dict[str, Any], *,
                           allow_completed_eval_progress: bool = False) -> dict[str, Any]:
    partial = read_json(PARTIAL_LEDGER)
    live = read_json(LIVE_LEDGER)
    current_counts = _counter_snapshot(live)
    expected_counts = manifest["counts_at_snapshot"]
    if allow_completed_eval_progress:
        expected_live_counts = _counts_after_verified_eval_claims(manifest, expected_counts)
        if current_counts != expected_live_counts:
            changed = sorted(key for key in current_counts
                             if current_counts[key] != expected_live_counts[key])
            raise EvalResumeError(
                f"live ledger does not reconcile to complete fixed-evaluation claims: {changed}")
    else:
        for field in ("normal_used", "reserve_used", "total_used", "environment_constructor_calls",
                      "run_invocation_calls", "completed_steps", "completed_logical_jobs",
                      "technical_failed_claims", "retry_claims", "optimizer_step_calls",
                      "ppo_update_calls", "checkpoint_write_calls"):
            if current_counts[field] != int(expected_counts[field]):
                raise EvalResumeError(f"live ledger differs from immutable snapshot at {field}")
    if partial.get("execution_counts") != expected_counts:
        # The partial closeout is a historical stop copy and is exact for these fields.
        for field, value in partial.get("execution_counts", {}).items():
            if field in expected_counts and expected_counts[field] != value:
                raise EvalResumeError(f"partial ledger differs from snapshot at {field}")
    if (partial.get("failed_eval_attempt", {}).get("claim_sha256")
            != manifest.get("failed_attempt1", {}).get("artifacts", {}).get("claim", {}).get("sha256")):
        raise EvalResumeError("failed attempt1 identity differs from immutable manifest")
    checkpoint_rows = manifest["checkpoint_pt_files"]["files"]
    if len(checkpoint_rows) != 204:
        raise EvalResumeError("immutable checkpoint manifest no longer lists all 204 files")
    checkpoint_hashes = {}
    for item in checkpoint_rows:
        path = ROOT / Path(item["path"])
        if not path.is_file() or path.stat().st_size != int(item["size_bytes"]):
            raise EvalResumeError(f"checkpoint file/size differs from immutable snapshot: {path}")
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise EvalResumeError(f"checkpoint content differs from immutable snapshot: {path}")
        checkpoint_hashes[item["path"]] = actual
    journal_hashes = {}
    for name, item in manifest["training_counter_journals"].items():
        path = ROOT / Path(item["path"])
        if (not path.is_file() or path.stat().st_size != int(item["size_bytes"])
                or sha256_file(path) != item["sha256"]):
            raise EvalResumeError(f"training counter journal changed: {path}")
        with path.open("r", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        if count != int(item["record_count"]):
            raise EvalResumeError(f"training counter journal count changed: {path}")
        journal_hashes[name] = item["sha256"]
    failed = manifest["failed_attempt1"]["artifacts"]
    failed_hashes = {}
    for name, item in failed.items():
        path = ROOT / Path(item["path"])
        if (not path.is_file() or path.stat().st_size != int(item["size_bytes"])
                or sha256_file(path) != item["sha256"]):
            raise EvalResumeError(f"original failed attempt1 artifact changed: {path}")
        failed_hashes[name] = item["sha256"]
    return {"counts": current_counts, "checkpoint_count": len(checkpoint_hashes),
            "checkpoint_sha256_by_path": checkpoint_hashes,
            "training_counter_journal_sha256": journal_hashes,
            "failed_attempt1_artifacts_sha256": failed_hashes,
            "partial_ledger_status": partial.get("status"),
            "live_ledger_sha256": sha256_file(LIVE_LEDGER)}


def _install_identity_validator(runtime_module: Any, run_module: Any,
                                validator: Callable[..., Any]) -> None:
    runtime_module.verify_frozen_source = validator
    if run_module is not None:
        run_module.verify_frozen_source = validator


def _install_claim_order_guard(run_module: Any, runtime: dict[str, Any], *,
                               frozen: dict[str, Any],
                               identity_validator: Callable[..., Any],
                               expected_keys: list[tuple[int, str, int]]) -> dict[str, Any]:
    """Install the production claim guard with one fixed key list and one cursor."""
    original_claim = run_module._claim
    cursor = {"index": 0, "keys": list(expected_keys)}

    def guarded_claim(logical_job: str, *, attempt: int,
                      metadata: dict[str, Any]) -> Path:
        identity_validator(frozen)
        if cursor["index"] >= len(cursor["keys"]):
            raise EvalResumeError("eval-only claim attempted after the fixed schedule ended")
        key, category = validate_eval_claim(logical_job, attempt, metadata)
        expected = cursor["keys"][cursor["index"]]
        if key != expected:
            raise EvalResumeError(
                f"eval-only claim order mismatch: got {key}, expected {expected}")
        claim_path = original_claim(logical_job, attempt=attempt, metadata=metadata)
        claim = read_json(claim_path)
        if claim.get("category") != category:
            raise EvalResumeError(f"atomic claim category mismatch for {key}: {claim_path}")
        return claim_path

    run_module._claim = guarded_claim
    runtime["eval_only_claim_cursor"] = cursor
    runtime["eval_only_base_claim"] = original_claim
    return cursor


def _configure_eval_runtime(*, source_head: str, frozen: dict[str, Any],
                            identity_validator: Callable[..., Any],
                            expected_keys: list[tuple[int, str, int]]) -> tuple[dict[str, Any], list[str]]:
    import torch
    import r44_runtime

    _install_identity_validator(r44_runtime, None, identity_validator)
    runtime = r44_runtime.configure(frozen)
    run_module = runtime["r42_run"]
    _install_identity_validator(r44_runtime, run_module, identity_validator)
    guards = install_read_only_guards(run_module, runtime, torch)
    _install_claim_order_guard(
        run_module, runtime, frozen=frozen,
        identity_validator=identity_validator, expected_keys=expected_keys)
    return runtime, guards


def _claim_attempt_paths(key: tuple[int, str, int]) -> list[Path]:
    logical = f"model_eval_new_s{key[0]}_{key[1]}_scene{key[2]}"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", logical).strip("_.")
    return sorted((RUN_DIR / "launch_claims" / slug).glob("attempt_*.json"))


def _counts_after_verified_eval_claims(
        manifest: dict[str, Any], baseline: dict[str, Any]) -> dict[str, int]:
    """Reconcile only terminal fixed-matrix claim deltas over the immutable training snapshot."""
    failed_claim_sha = manifest.get("failed_attempt1", {}).get("artifacts", {}).get(
        "claim", {}).get("sha256")
    if not isinstance(failed_claim_sha, str):
        raise EvalResumeError("immutable manifest lacks the preserved failed-attempt claim hash")
    baseline_claim_seen = False
    new_claims: list[tuple[tuple[int, str, int], str, dict[str, Any]]] = []
    for key in evaluation_schedule():
        logical_job = f"model_eval_new_s{key[0]}_{key[1]}_scene{key[2]}"
        paths = _claim_attempt_paths(key)
        attempts = []
        for path in paths:
            digest = sha256_file(path)
            claim = read_json(path)
            if digest == failed_claim_sha:
                if baseline_claim_seen or key != RETRY_KEY:
                    raise EvalResumeError("preserved failed attempt claim appears more than once or at a foreign key")
                baseline_claim_seen = True
                continue
            if type(claim.get("attempt")) is not int:
                raise EvalResumeError(f"evaluation claim attempt must be an integer: {path}")
            attempt = claim["attempt"]
            resolved_key, category = validate_eval_claim(
                logical_job, attempt, claim.get("metadata"))
            if resolved_key != key or claim.get("category") != category:
                raise EvalResumeError(f"evaluation claim identity/category mismatch: {path}")
            if (claim.get("round") != 44
                    or claim.get("training_version") != TRAINING_VERSION
                    or claim.get("comm_action_semantics") != "one-shot-send-skip/1"):
                raise EvalResumeError(f"evaluation claim version identity mismatch: {path}")
            if claim.get("status") not in {"complete", "failed_technical"}:
                raise EvalResumeError(f"cannot resume with a nonterminal evaluation claim: {path}")
            if claim.get("status") == "complete":
                for field, expected in (
                    ("environment_constructor_calls", 1), ("run_invocation_calls", 1),
                    ("joint_run_constructed_calls", 1), ("joint_run_method_calls", 1),
                    ("run_return_calls", 1), ("run_failure_calls", 0),
                    ("completed_steps", 1_000), ("optimizer_step_calls", 0),
                    ("checkpoint_write_calls", 0),
                ):
                    if type(claim.get(field)) is not int or claim[field] != expected:
                        raise EvalResumeError(f"completed evaluation claim has bad {field}: {path}")
            attempts.append(attempt)
            new_claims.append((key, category, claim))
        if key == RETRY_KEY:
            if attempts not in ([], [2]):
                raise EvalResumeError(f"reserve retry claims are not the preserved 1 then new 2: {attempts}")
        elif attempts not in ([], [1]):
            raise EvalResumeError(f"ordinary evaluation attempts differ from the frozen single attempt: {key}")
    if not baseline_claim_seen:
        raise EvalResumeError("the original failed attempt claim is missing from the fixed retry key")

    result = {key: int(baseline[key]) for key in _counter_snapshot_keys()}
    for _key, category, claim in new_claims:
        result["normal_used"] += int(category == "normal")
        result["reserve_used"] += int(category == "reserve")
        result["total_used"] += 1
        result["environment_constructor_calls"] += int(claim.get("environment_constructor_calls", 0))
        result["run_invocation_calls"] += int(claim.get("run_invocation_calls", 0))
        result["completed_steps"] += int(claim.get("completed_steps", 0))
        result["completed_logical_jobs"] += int(claim.get("status") == "complete")
        result["technical_failed_claims"] += int(claim.get("status") == "failed_technical")
        result["retry_claims"] += int(category == "reserve")
    if (result["normal_used"] > 480 or result["reserve_used"] > 24
            or result["total_used"] > 504):
        raise EvalResumeError("observed eval claim deltas exceed the frozen quota")
    return result


def _counter_snapshot_keys() -> tuple[str, ...]:
    return ("normal_used", "reserve_used", "total_used", "environment_constructor_calls",
            "run_invocation_calls", "completed_steps", "completed_logical_jobs",
            "technical_failed_claims", "retry_claims", "optimizer_step_calls",
            "ppo_update_calls", "checkpoint_write_calls")


def _model_gate(*, runtime: dict[str, Any], identity_head: str,
                snapshot: dict[str, Any], guards: list[str]) -> dict[str, Any]:
    import torch
    from r44_eval_index import build_eval_index
    from r44_prepare import RUNNER_VERSION, TRAINING_VERSION

    run_module = runtime["r42_run"]
    frozen = runtime["frozen"]
    frozen_round42 = runtime["frozen"]["round42"]
    if (float(runtime["common"].ETA_LAMBDA) != 0.5
            or float(runtime["common"].GAE_LAMBDA) != 1.0
            or float(runtime["common"].GAMMA) != 1.0
            or int(frozen_round42["steps_per_episode"]) != 1_000
            or int(frozen_round42["actor_feature_width"]) != 19):
        raise EvalResumeError("actual configured evaluation constants differ from freeze")
    if int(runtime["common"].NORMAL_LAUNCHES) != 480 \
            or int(runtime["common"].RESERVE_LAUNCHES) != 24 \
            or int(runtime["common"].LAUNCH_CAP) != 504:
        raise EvalResumeError("actual atomic launch quota differs from 480+24=504")
    partial = read_json(PARTIAL_LEDGER)
    expected_parameters = {int(seed): partial["training"][seed]["parameter_sha256"]
                           for seed in ("11", "22", "33")}
    if expected_parameters != PARAMETER_SHA:
        raise EvalResumeError("ep128 parameter digest table differs from partial ledger")
    checkpoint_parameters: dict[int, str] = {}
    loaded: dict[int, tuple[Any, dict[str, Any], dict[str, Any]]] = {}
    model_records = []
    trainer = runtime["trainer"]
    rng_guard_state = runtime.get("eval_only_rng_guard_state")
    if type(rng_guard_state) is not dict:
        raise EvalResumeError("read-only model gate lacks its installed ambient RNG guard state")
    tape_path = RUN_DIR / "logs" / "v2_r44_eval_new_s11_greedy_s1301000_a1_tape.jsonl"
    manifest = read_json(IMMUTABLE_MANIFEST)
    failed_tape = manifest["failed_attempt1"]["artifacts"]["tape"]
    if sha256_file(tape_path) != failed_tape["sha256"]:
        raise EvalResumeError("failed attempt1 source tape changed before read-only model forward")
    with tape_path.open("r", encoding="utf-8") as handle:
        first_tape = json.loads(next(handle), object_pairs_hook=_strict_object,
                                parse_constant=_reject_constant)
    _validate_json_finite(first_tape)
    features = first_tape.get("features")
    if (type(features) is not list or len(features) < 1
            or type(features[0]) is not list or len(features[0]) != 19):
        raise EvalResumeError("preserved attempt1 tape lacks a valid 19D forward input")
    observation = torch.tensor(features, dtype=torch.float32, device="cpu")
    if not bool(torch.isfinite(observation).all()):
        raise EvalResumeError("preserved tape observation is non-finite")
    before_ledger = run_module.ledger_snapshot(strict=True)
    before_counters = _counter_snapshot(before_ledger)
    for seed in SEEDS:
        expected = partial["training"][str(seed)]
        path = ROOT / Path(expected["checkpoint_path"])
        if (not path.is_file() or path.stat().st_size != 114_027
                or sha256_file(path) != EP128_FILE_SHA[seed]
                or EP128_FILE_SHA[seed] != expected["checkpoint_file_sha256"]
                or PARAMETER_SHA[seed] != expected["parameter_sha256"]):
            raise EvalResumeError(f"ep128 checkpoint byte identity mismatch for seed {seed}")
        rng_before_load = _rng_state_sha256(trainer.rng_state())
        loads_before = int(rng_guard_state["read_only_load_calls"])
        restores_before = int(rng_guard_state["ambient_restore_calls"])
        controller, loaded_identity = run_module.load_eval_model("NEW", seed)
        rng_after_load = _rng_state_sha256(trainer.rng_state())
        if (loaded_identity.get("optimizer_restored") is not False
                or loaded_identity.get("rng_restored") is not False
                or loaded_identity.get("ambient_rng_preserved") is not True
                or loaded_identity.get("episodes_done") != 128
                or loaded_identity.get("parameter_identity", {}).get("parameter_sha256")
                != PARAMETER_SHA[seed]
                or sha256_file(path) != EP128_FILE_SHA[seed]):
            raise EvalResumeError(f"ep128 read-only load identity mismatch for seed {seed}")
        if rng_after_load != rng_before_load:
            raise EvalResumeError(f"ep128 read-only load changed ambient RNG state for s{seed}")
        if (int(rng_guard_state["read_only_load_calls"]) != loads_before + 1
                or int(rng_guard_state["ambient_restore_calls"]) != restores_before + 1
                or bool(rng_guard_state["active"])):
            raise EvalResumeError(f"ep128 read-only load bypassed or repeated its RNG guard for s{seed}")
        rng_before_forward = _rng_state_sha256(trainer.rng_state())
        with torch.inference_mode():
            logits, task_values = controller.forward(observation)
            cost_values = controller.cost_values(observation)
        rng_after_forward = _rng_state_sha256(trainer.rng_state())
        if rng_after_forward != rng_before_forward:
            raise EvalResumeError(f"ep128 CPU forward changed ambient RNG state for s{seed}")
        for label, tensor, expected_shape in (
                ("logits", logits, (1, 2)), ("task_values", task_values, (1,)),
                ("cost_values", cost_values, (1,))):
            if tuple(tensor.shape) != expected_shape or not bool(torch.isfinite(tensor).all()):
                raise EvalResumeError(f"ep128 CPU forward failed {label} shape/finite check for s{seed}")
        after_summary = controller.parameter_summary()
        if after_summary["parameter_sha256"] != PARAMETER_SHA[seed]:
            raise EvalResumeError(f"ep128 parameters changed during read-only forward for s{seed}")
        checkpoint_parameters[seed] = PARAMETER_SHA[seed]
        loaded[seed] = (controller, loaded_identity, after_summary)
        model_records.append({
            "seed": seed, "checkpoint_path": str(path.resolve()),
            "checkpoint_file_sha256": EP128_FILE_SHA[seed],
            "parameter_sha256": PARAMETER_SHA[seed],
            "training_version": loaded_identity.get("training_version"),
            "episodes_done": loaded_identity["episodes_done"],
            "optimizer_restored": False, "rng_restored": False,
            "ambient_rng_preserved": True,
            "ambient_rng_sha256": {
                "before_read_only_load": rng_before_load,
                "after_read_only_load": rng_after_load,
                "before_cpu_forward": rng_before_forward,
                "after_cpu_forward": rng_after_forward,
            },
            "read_only_loader": "trainer.load_checkpoint_read_only",
            "forward_input": {"source": str(tape_path.resolve()),
                              "attempt": 1, "step": int(first_tape["step"]),
                              "dimension": 19},
            "forward_outputs": {"logits": [float(v) for v in logits.detach().cpu().reshape(-1)],
                                "task_values": [float(v) for v in task_values.detach().cpu().reshape(-1)],
                                "cost_values": [float(v) for v in cost_values.detach().cpu().reshape(-1)]},
            "parameter_identity_unchanged_after_forward": True,
        })
    if (int(rng_guard_state["read_only_load_calls"]) != len(SEEDS)
            or int(rng_guard_state["ambient_restore_calls"]) != len(SEEDS)
            or bool(rng_guard_state["active"])):
        raise EvalResumeError("three-seed model gate did not use exactly one guarded load per seed")
    after_ledger = run_module.ledger_snapshot(strict=True)
    after_counters = _counter_snapshot(after_ledger)
    if before_counters != after_counters:
        raise EvalResumeError("read-only checkpoint load/forward changed ledger or update counters")
    if before_counters != snapshot["counts"]:
        raise EvalResumeError("prelaunch runtime ledger differs from the preserved live snapshot")
    index = build_eval_index(
        RUN_DIR, training_version=TRAINING_VERSION,
        reward_timing_version=runtime["r42_versions"].REWARD_TIMING_VERSION,
        runner_version=RUNNER_VERSION,
        checkpoint_parameter_sha256_by_seed=checkpoint_parameters,
    )
    if index.completed_count != 0 or index.get(*RETRY_KEY) is not None:
        raise EvalResumeError("expected the preserved failed attempt1 to yield an empty science index")
    retry_claims = _claim_attempt_paths(RETRY_KEY)
    if len(retry_claims) != 1 or read_json(retry_claims[0]).get("status") != "failed_technical":
        raise EvalResumeError("original failed attempt1 is not the sole pre-retry claim")
    source_head = identity_head
    return {
        "kind": "round44-eval-readonly-model-gate/1", "status": "passed",
        "source_or_execution_head": source_head,
        "training_version": TRAINING_VERSION, "runner_version": RUNNER_VERSION,
        "scientific_values": {"eta_lambda": 0.5, "gae_lambda": 1.0, "gamma": 1.0,
                              "normal_cap": 480, "reserve_cap": 24, "hard_cap": 504,
                              "horizon": int(frozen_round42["steps_per_episode"]),
                              "actor_feature_width": int(frozen_round42["actor_feature_width"]),
                              "budget_bytes": 29300},
        "current_counts_before_after": {"before": before_counters, "after": after_counters},
        "model_loads_and_forward": model_records,
        "ambient_rng_audit": {
            "snapshot_fields": ["python", "numpy", "torch"],
            "read_only_load_calls": int(rng_guard_state["read_only_load_calls"]),
            "allowed_ambient_restore_calls": int(rng_guard_state["ambient_restore_calls"]),
            "all_loads_preserved_ambient_rng": True,
            "all_cpu_forwards_preserved_ambient_rng": True,
        },
        "initial_valid_eval_index_count": index.completed_count,
        "failed_attempt1_claim_sha256": sha256_file(retry_claims[0]),
        "failed_attempt1_row_sha256": sha256_file(RUN_DIR / "rows" /
            "v2_r44_eval_new_s11_greedy_s1301000_a1.json"),
        "training_counter_journal_sha256_unchanged": snapshot["training_counter_journal_sha256"],
        "guarded_operations": guards,
        "environment_constructor_calls": 0, "run_invocation_calls": 0,
        "optimizer_step_calls_delta": 0, "ppo_update_calls_delta": 0,
        "checkpoint_write_calls_delta": 0,
        "checkpoint_file_set_count": snapshot["checkpoint_count"],
        "checkpoint_file_hashes_unchanged": True,
    }, loaded


def prelaunch_audit() -> dict[str, Any]:
    source_head, manifest = verify_prelaunch_source_gate()
    lock = _acquire_lock(source_head)
    try:
        snapshot = _verify_snapshot_inputs(manifest)
        sys.path.insert(0, str(RUN_DIR))
        import r44_prepare
        from r44_runtime import configure

        frozen = r44_prepare.read(RUN_DIR / "ROUND44_FROZEN_CONFIG.json")
        runtime, guards = _configure_eval_runtime(
            source_head=source_head, frozen=frozen,
            identity_validator=lambda _frozen=None: _assert_source_commit(expected_head=source_head),
            expected_keys=[RETRY_KEY],
        )
        gate, loaded = _model_gate(runtime=runtime, identity_head=source_head,
                                   snapshot=snapshot, guards=guards)
        gate.update({
            "immutable_input_manifest_sha256": sha256_file(IMMUTABLE_MANIFEST),
            "receipt_regression_sha256": sha256_file(
                RUN_DIR / "ROUND44_EVAL_RECEIPT_REPAIR_REGRESSION_R1.json"),
            "mechanical_diff_sha256": sha256_file(RUN_DIR / "ROUND44_EVAL_MECHANICAL_DIFF_R1.json"),
            "prepatch_index_sha256": sha256_file(PREPATCH_INDEX),
            "old_failure_reproduction_sha256": sha256_file(OLD_FAILURE_REPRO),
            "r4_history_hashes": _verify_r4_history(source_head),
            "input_snapshot_check": snapshot,
            "read_only_load_count": len(loaded),
            "attempts_before_retry": {"claim_count": 1, "retry_attempt_number": 2,
                                      "reserve_used": 0, "valid_eval_cells": 0},
        })
        _write_new_or_match(READONLY_GATE, gate)
        return gate
    finally:
        _release_lock(lock)


def _check_final_identity_source() -> tuple[str, dict[str, Any], dict[str, Any]]:
    head, identity, gate = verify_eval_identity()
    if _parent(head) != identity["source_commit"]:
        raise EvalResumeError("execution identity no longer directly follows the eval code commit")
    return head, identity, gate


def _update_progress(head: str, identity: dict[str, Any], index: Any,
                     status: str, *, retry_gate_passed: bool) -> dict[str, Any]:
    existing = read_json(PROGRESS) if PROGRESS.is_file() else None
    if existing is None:
        progress = {
            "kind": "round44-execution-progress-eval-only-r3/1",
            "status": status, "prelaunch_head": head,
            "source_commit": identity["source_commit"],
            "execution_identity_sha256": sha256_file(EVAL_IDENTITY),
            "completed_eval_keys": index.completed_keys,
            "evaluations_completed": index.completed_count,
            "retry_gate_passed": retry_gate_passed,
            "started_at_unix": time.time(),
        }
    else:
        expected_identity_sha = sha256_file(EVAL_IDENTITY)
        if (existing.get("kind") != "round44-execution-progress-eval-only-r3/1"
                or existing.get("prelaunch_head") != head
                or existing.get("execution_identity_sha256") != expected_identity_sha
                or existing.get("status") == "complete"):
            raise EvalResumeError("existing eval-only progress belongs to a different identity/state")
        progress = existing
        progress["status"] = status
        progress["completed_eval_keys"] = index.completed_keys
        progress["evaluations_completed"] = index.completed_count
        progress["retry_gate_passed"] = retry_gate_passed
        progress["updated_at_unix"] = time.time()
    _write_atomic(PROGRESS, progress)
    return progress


def _retry_gate(runtime: dict[str, Any], index: Any, parameter_sha: dict[int, str],
                controller: Any, identity: dict[str, Any], before_parameters: dict[str, str],
                checkpoint_hashes_before: dict[str, str], journals_before: dict[str, str]) -> dict[str, Any]:
    from r44_eval_index import verify_new_cell
    from r44_prepare import RUNNER_VERSION, TRAINING_VERSION

    if index.get(*RETRY_KEY) is None:
        raise EvalResumeError("the authorized retry is not in the verified evaluation index")
    row, entry = verify_new_cell(
        RUN_DIR, seed=RETRY_KEY[0], mode=RETRY_KEY[1], scene=RETRY_KEY[2],
        training_version=TRAINING_VERSION,
        reward_timing_version=runtime["r42_versions"].REWARD_TIMING_VERSION,
        runner_version=RUNNER_VERSION,
        checkpoint_parameter_sha256=parameter_sha[RETRY_KEY[0]],
    )
    if row.get("attempt") != 2 or entry.get("attempt") != 2:
        raise EvalResumeError("retry receipt/index does not reference reserve attempt 2")
    retry_claim = read_json(Path(entry["claim_path"]))
    if retry_claim.get("category") != "reserve" or retry_claim.get("status") != "complete":
        raise EvalResumeError("reserve retry claim is not complete under attempt 2")
    if int(row.get("steps", -1)) != 1000 or int(retry_claim.get("completed_steps", -1)) != 1000:
        raise EvalResumeError("reserve retry did not complete the frozen 1000 steps")
    runtime["r42_analysis"]._validate_reward_timing(row)
    aggregate = runtime["r42_analysis"]._aggregate([row])
    if (int(aggregate["on_time_total"]) != int(row["on_time"])
            or int(aggregate["spent_bytes_total"]) != int(row["spent_bytes"])
            or float(row["task_reward_total"]) != float(row["on_time"])):
        raise EvalResumeError("retry task/reward/cost aggregate did not close")
    after_parameters = controller.parameter_summary()["parameter_sha256"]
    if after_parameters != before_parameters[str(RETRY_KEY[0])]:
        raise EvalResumeError("evaluation changed the read-only ep128 controller parameters")
    checkpoints_after = _checkpoint_hash_map()
    journals_after = _journal_hash_map()
    if checkpoints_after != checkpoint_hashes_before:
        raise EvalResumeError("checkpoint files changed during the authorized retry")
    if journals_after != journals_before:
        raise EvalResumeError("training counter journals changed during the authorized retry")
    return {
        "kind": "round44-eval-retry-gate/1", "status": "passed",
        "eval_key": list(RETRY_KEY), "attempt": 2, "category": "reserve",
        "failed_attempt1_preserved": True,
        "failed_attempt1_claim_sha256": sha256_file(_claim_attempt_paths(RETRY_KEY)[0]),
        "new_claim_path": entry["claim_path"], "new_claim_sha256": entry["claim_sha256"],
        "new_row_path": entry["row_path"], "new_row_sha256": entry["row_sha256"],
        "row_steps": int(row["steps"]), "on_time": int(row["on_time"]),
        "spent_bytes": int(row["spent_bytes"]),
        "task_reward_total": float(row["task_reward_total"]),
        "receipt_kind": entry["kind"],
        "representation_version": entry["json_representation"],
        "artifact_sha256": {name: value["sha256"]
                             for name, value in entry["artifacts"].items()},
        "timing_validation": "passed",
        "aggregate_closure": aggregate,
        "parameter_sha256_before_after": after_parameters,
        "checkpoint_hashes_unchanged": True,
        "training_journals_unchanged": True,
        "optimizer_step_delta": 0, "ppo_update_delta": 0,
        "checkpoint_write_delta": 0,
        "valid_eval_index_count_after_retry": index.completed_count,
    }


def _checkpoint_hash_map() -> dict[str, str]:
    manifest = read_json(IMMUTABLE_MANIFEST)
    return {item["path"]: sha256_file(ROOT / Path(item["path"]))
            for item in manifest["checkpoint_pt_files"]["files"]}


def _journal_hash_map() -> dict[str, str]:
    manifest = read_json(IMMUTABLE_MANIFEST)
    return {name: sha256_file(ROOT / Path(item["path"]))
            for name, item in manifest["training_counter_journals"].items()}


def _make_claim_guard_queue(index: Any) -> list[tuple[int, str, int]]:
    missing = [key for key in evaluation_schedule() if index.get(*key) is None]
    if RETRY_KEY in missing and missing[0] != RETRY_KEY:
        raise EvalResumeError("fixed evaluation schedule no longer places the reserve retry first")
    return missing


def _consume_claim_queue_key(cursor: dict[str, Any],
                             expected: tuple[int, str, int]) -> None:
    keys = cursor["keys"]
    index = cursor["index"]
    if index >= len(keys) or keys[index] != expected:
        raise EvalResumeError(
            f"completed evaluation did not match the next queued key: {expected}")
    cursor["index"] += 1


def _check_retry_attempt_state(index: Any) -> None:
    paths = _claim_attempt_paths(RETRY_KEY)
    claims = [read_json(path) for path in paths]
    attempts = [int(item.get("attempt", -1)) for item in claims]
    if attempts != [1] and attempts != [1, 2]:
        raise EvalResumeError(f"failed key has unexpected claim attempts: {attempts}")
    if claims[0].get("status") != "failed_technical":
        raise EvalResumeError("normal attempt1 is not preserved as technical failure")
    if len(claims) == 2:
        if claims[1].get("status") != "complete" or index.get(*RETRY_KEY) is None:
            raise EvalResumeError("reserve attempt2 exists but is not a verified complete retry")


def run_evaluation() -> dict[str, Any]:
    head, identity, identity_gate = _check_final_identity_source()
    lock = _acquire_lock(head)
    try:
        manifest = read_json(IMMUTABLE_MANIFEST)
        if sha256_file(IMMUTABLE_MANIFEST) != identity["immutable_input_manifest_sha256"]:
            raise EvalResumeError("immutable input manifest changed after identity freeze")
        snapshot = _verify_snapshot_inputs(manifest, allow_completed_eval_progress=True)
        sys.path.insert(0, str(RUN_DIR))
        import torch
        import r44_prepare
        import r44_runtime
        import r44_analysis
        import r44_report
        from r44_eval_index import audit_fixed_matrix, build_eval_index, verify_new_cell
        from r44_prepare import RUNNER_VERSION, TRAINING_VERSION

        frozen = r44_prepare.read(RUN_DIR / "ROUND44_FROZEN_CONFIG.json")
        runtime, guards = _configure_eval_runtime(
            source_head=identity["source_commit"], frozen=frozen,
            identity_validator=lambda _frozen=None: verify_eval_identity()[0],
            expected_keys=[],
        )
        run_module = runtime["r42_run"]
        before_ledger = run_module.ledger_snapshot(strict=True)
        before_counts = _counter_snapshot(before_ledger)
        if before_counts != snapshot["counts"]:
            raise EvalResumeError("live execution ledger changed before evaluation entry")
        if (float(runtime["common"].ETA_LAMBDA) != 0.5
                or float(runtime["common"].GAE_LAMBDA) != 1.0
                or float(runtime["common"].GAMMA) != 1.0):
            raise EvalResumeError("actual eval runtime changed the frozen scientific values")
        partial = read_json(PARTIAL_LEDGER)
        parameter_sha = {int(seed): partial["training"][seed]["parameter_sha256"]
                         for seed in ("11", "22", "33")}
        checkpoint_parameters = {}
        loaded_models = {}
        before_parameter_sha = {}
        for seed in SEEDS:
            controller, model_identity = run_module.load_eval_model("NEW", seed)
            if (model_identity.get("optimizer_restored") is not False
                    or model_identity.get("rng_restored") is not False
                    or model_identity.get("parameter_identity", {}).get("parameter_sha256")
                    != parameter_sha[seed]
                    or model_identity.get("file_sha256") != EP128_FILE_SHA[seed]):
                raise EvalResumeError(f"actual eval model is not the read-only ep128 state: {seed}")
            checkpoint_parameters[seed] = parameter_sha[seed]
            before_parameter_sha[str(seed)] = controller.parameter_summary()["parameter_sha256"]
            loaded_models[seed] = (controller, model_identity)

        index = build_eval_index(
            RUN_DIR, training_version=TRAINING_VERSION,
            reward_timing_version=runtime["r42_versions"].REWARD_TIMING_VERSION,
            runner_version=RUNNER_VERSION,
            checkpoint_parameter_sha256_by_seed=checkpoint_parameters,
        )
        _check_retry_attempt_state(index)
        missing = _make_claim_guard_queue(index)
        runtime["eval_only_claim_cursor"]["keys"] = missing
        runtime["eval_only_claim_cursor"]["index"] = 0
        progress = _update_progress(head, identity, index, "running_retry_gate",
                                    retry_gate_passed=(RETRY_KEY not in missing
                                                       and RETRY_GATE.is_file()))
        checkpoints_before = _checkpoint_hash_map()
        journals_before = _journal_hash_map()
        if len(checkpoints_before) != 204:
            raise EvalResumeError("expected all 204 registered checkpoint files before evaluation")
        if RETRY_KEY in missing:
            if (runtime["r40_common"].next_attempt(
                    f"model_eval_new_s{RETRY_KEY[0]}_{RETRY_KEY[1]}_scene{RETRY_KEY[2]}") != 2
                    or int(before_counts["reserve_used"]) != 0):
                raise EvalResumeError("reserve slot1/attempt2 is not the exact next claim")
            print("ROUND44: reserve attempt2 for (11, greedy, 1301000)", flush=True)
            controller, model_identity = loaded_models[RETRY_KEY[0]]
            row = run_module.run_model_eval(
                arm="NEW", training_seed=RETRY_KEY[0], mode=RETRY_KEY[1],
                scenario=RETRY_KEY[2], controller=controller, identity=model_identity)
            verified_row, entry = verify_new_cell(
                RUN_DIR, seed=RETRY_KEY[0], mode=RETRY_KEY[1], scene=RETRY_KEY[2],
                training_version=TRAINING_VERSION,
                reward_timing_version=runtime["r42_versions"].REWARD_TIMING_VERSION,
                runner_version=RUNNER_VERSION,
                checkpoint_parameter_sha256=checkpoint_parameters[RETRY_KEY[0]],
            )
            if not _eval_rows_equivalent(row, verified_row) or int(row.get("attempt", -1)) != 2:
                raise EvalResumeError("reserve retry in-memory row differs from its durable receipt")
            index.add_verified(entry, verified_row)
            retry_cursor = runtime["eval_only_claim_cursor"]
            if (retry_cursor["index"] >= len(retry_cursor["keys"])
                    or retry_cursor["keys"][retry_cursor["index"]] != RETRY_KEY):
                raise EvalResumeError("retry key was not the first atomic evaluation claim")
            retry_result = _retry_gate(
                runtime, index, checkpoint_parameters, controller,
                model_identity, before_parameter_sha, checkpoints_before, journals_before)
            _write_new_or_match(RETRY_GATE, retry_result)
            _consume_claim_queue_key(retry_cursor, RETRY_KEY)
            progress = _update_progress(head, identity, index, "running_fixed_95",
                                        retry_gate_passed=True)
            print("ROUND44: retry gate passed; continuing original fixed schedule", flush=True)
        else:
            if RETRY_GATE.is_file():
                retry_result = read_json(RETRY_GATE)
                if (retry_result.get("status") != "passed"
                        or sha256_file(READONLY_GATE) != identity_gate.get(
                            "readonly_model_gate_sha256")):
                    raise EvalResumeError("previous retry gate does not match the execution identity")
            else:
                if index.get(*RETRY_KEY) is None:
                    raise EvalResumeError("retry gate is missing and the durable retry row is unavailable")
                print("ROUND44: rebuilding retry gate from the already verified attempt2 receipt; no new claim",
                      flush=True)
                controller, model_identity = loaded_models[RETRY_KEY[0]]
                retry_result = _retry_gate(
                    runtime, index, checkpoint_parameters, controller,
                    model_identity, before_parameter_sha, checkpoints_before, journals_before)
                _write_new_or_match(RETRY_GATE, retry_result)
            progress = _update_progress(head, identity, index, "running_fixed_95",
                                        retry_gate_passed=True)

        # The cursor begins after a just-completed retry; existing verified cells are reused.
        claim_cursor = runtime["eval_only_claim_cursor"]
        for key in evaluation_schedule():
            seed, mode, scene = key
            row = index.get(seed, mode, scene)
            if row is not None:
                continue
            if not progress.get("retry_gate_passed"):
                raise EvalResumeError("remaining fixed cells are blocked until the retry gate passes")
            if (claim_cursor["index"] >= len(claim_cursor["keys"])
                    or claim_cursor["keys"][claim_cursor["index"]] != key):
                raise EvalResumeError(f"remaining claim is out of the fixed schedule: {key}")
            expected_attempt = 1
            if key == RETRY_KEY:
                raise EvalResumeError("reserve retry was not indexed before remaining fixed cells")
            print(f"ROUND44: NEW s{seed}/{mode} scene={scene}", flush=True)
            controller, model_identity = loaded_models[seed]
            row = run_module.run_model_eval(
                arm="NEW", training_seed=seed, mode=mode, scenario=scene,
                controller=controller, identity=model_identity)
            verified_row, entry = verify_new_cell(
                RUN_DIR, seed=seed, mode=mode, scene=scene,
                training_version=TRAINING_VERSION,
                reward_timing_version=runtime["r42_versions"].REWARD_TIMING_VERSION,
                runner_version=RUNNER_VERSION,
                checkpoint_parameter_sha256=checkpoint_parameters[seed],
            )
            if not _eval_rows_equivalent(row, verified_row) or int(row.get("attempt", -1)) != expected_attempt:
                raise EvalResumeError(f"normal evaluation row/attempt mismatch for {key}")
            if controller.parameter_summary()["parameter_sha256"] != before_parameter_sha[str(seed)]:
                raise EvalResumeError(f"read-only model parameters changed during evaluation: seed {seed}")
            index.add_verified(entry, verified_row)
            _consume_claim_queue_key(claim_cursor, key)
            progress = _update_progress(head, identity, index, "running_fixed_95",
                                        retry_gate_passed=True)
            # A strict cumulative ledger snapshot after every completed logical cell.
            run_module.write_execution_ledger(strict=True)
        if index.completed_count != 96:
            raise EvalResumeError(f"fixed evaluation index is incomplete: {index.completed_count}/96")
        if claim_cursor["index"] != len(claim_cursor["keys"]):
            remaining = claim_cursor["keys"][claim_cursor["index"]:]
            raise EvalResumeError(f"fixed evaluation claims remain after the schedule: {remaining[:1]}")

        eval_audit = audit_fixed_matrix(
            RUN_DIR, training_version=TRAINING_VERSION,
            reward_timing_version=runtime["r42_versions"].REWARD_TIMING_VERSION,
            runner_version=RUNNER_VERSION,
            checkpoint_parameter_sha256_by_seed=checkpoint_parameters,
        )
        _write_new_or_match(RUN_DIR / "ROUND44_EVAL_MATRIX_AUDIT.json", eval_audit)
        ledger = run_module.write_execution_ledger(strict=True)
        _write_new_or_match(RUN_DIR / "ROUND44_EXECUTION_LEDGER_FINAL.json", ledger)
        from r44_execute import _validate_claim_closure
        claim_closure = _validate_claim_closure(ledger)
        counts = ledger["counts"]
        expected_final = {
            "normal_used": 480, "reserve_used": 1, "total_used": 481,
            "environment_constructor_calls": 481, "joint_run_constructed_calls": 481,
            "run_invocation_calls": 481, "completed_steps": 481_000,
            "completed_logical_jobs": 480, "technical_failed_claims": 1,
            "retry_claims": 1, "optimizer_step_calls": 1_016,
            "ppo_update_calls": 96, "checkpoint_write_calls": 297,
            "hard_cap": 504,
        }
        for name, expected in expected_final.items():
            if int(counts.get(name, -1)) != expected:
                raise EvalResumeError(f"final execution ledger count {name}={counts.get(name)}; expected {expected}")
        if _checkpoint_hash_map() != checkpoints_before or _journal_hash_map() != journals_before:
            raise EvalResumeError("checkpoint or training-update journal changed during fixed evaluation")

        # Analysis is read-only; it verifies all 384 training rows, 96 batches, baselines, and
        # checkpoint milestone payloads before producing the final paired matrix.
        analysis = r44_analysis.write_analysis(runtime=runtime, frozen=frozen,
                                               eval_audit=eval_audit)
        first_batch_comparisons = [read_json(
            RUN_DIR / f"ROUND44_FIRST_BATCH_COMPARISON_SEED{seed}.json") for seed in SEEDS]
        all_comparison = {
            "kind": "round44-first-batch-comparison-all-seeds/1", "status": "passed",
            "by_seed": {str(item["training_seed"]): item for item in first_batch_comparisons},
            "parameter_identical_seed_count": sum(bool(item["postupdate_parameters_equal"])
                                                   for item in first_batch_comparisons),
            "actor_and_optimizer_updates_identical": all(
                item["ppo_update_report_equal"] and item["postupdate_parameters_equal"]
                and item["postupdate_optimizer_state_equal"] for item in first_batch_comparisons),
            "only_batch_end_dual_response_differs": all(
                item["g_hat_equal"] and item["r44_eta_lambda"] == 0.5
                for item in first_batch_comparisons),
        }
        _write_new_or_match(RUN_DIR / "ROUND44_FIRST_BATCH_COMPARISON.json", all_comparison)
        from r44_execute import _raw_artifact_manifest
        raw_manifest = _raw_artifact_manifest()
        lock_rel = f"runtime/{LOCK_PATH.name}"
        raw_manifest["files"] = [item for item in raw_manifest["files"]
                                  if item["relative_path"] != lock_rel]
        raw_manifest["raw_files_count"] = len(raw_manifest["files"])
        _write_new_or_match(RUN_DIR / "ROUND44_RAW_ARTIFACT_MANIFEST.json", raw_manifest)
        zero_tests_path = RUN_DIR / "ROUND44_ZERO_ENV_TESTS.json"
        if not zero_tests_path.is_file():
            zero_tests_path = RUN_DIR / "ROUND44_EVAL_RECEIPT_REPAIR_REGRESSION_R1.json"
        zero_tests = read_json(zero_tests_path)
        report = r44_report.write_report(
            result=analysis, ledger=ledger, progress=progress, eval_audit=eval_audit,
            execution_head=head, zero_tests=zero_tests,
            final_artifact_count=raw_manifest["raw_files_count"],
        )
        progress.update({
            "status": "complete", "finished_at_unix": time.time(),
            "evaluations_completed": 96, "completed_eval_keys": index.completed_keys,
            "final_ledger_counts": counts, "claim_closure": claim_closure,
            "eval_matrix_audit_sha256": sha256_file(RUN_DIR / "ROUND44_EVAL_MATRIX_AUDIT.json"),
            "report_sha256": report["sha256"],
        })
        _write_atomic(PROGRESS, progress)
        status = {
            "kind": "round44-project-status/5", "round": 44,
            "status": "complete_local_delivery_pending_commits",
            "execution_matrix_status": "complete", "branch": BRANCH,
            "scientific_execution_head": head,
            "execution_identity_path": str(EVAL_IDENTITY),
            "execution_identity_sha256": sha256_file(EVAL_IDENTITY),
            "training_version": TRAINING_VERSION, "runner_version": RUNNER_VERSION,
            "receipt_representation_version": REPRESENTATION_KIND,
            "eta_lambda_old": 0.05, "eta_lambda_new": 0.5,
            "result": analysis["machine_verdict"]["scientific_result"],
            "normal_claims": counts["normal_used"], "reserve_claims": counts["reserve_used"],
            "total_claims": counts["total_used"], "hard_cap": 504,
            "environment_constructor_calls": counts["environment_constructor_calls"],
            "run_invocation_calls": counts["run_invocation_calls"],
            "completed_steps": counts["completed_steps"],
            "ppo_update_calls": counts["ppo_update_calls"],
            "optimizer_step_calls": counts["optimizer_step_calls"],
            "checkpoint_write_calls": counts["checkpoint_write_calls"],
            "training_episodes": 384, "new_final_evaluations": 96,
            "reserve_retry_key": list(RETRY_KEY), "reserve_retry_attempt": 2,
            "retry_gate_sha256": sha256_file(RETRY_GATE),
            "reused_old_r43_ep128_evaluations": 96,
            "independent_confirmation": False, "remote_push": False,
            "report_path": report["path"], "report_sha256": report["sha256"],
            "raw_artifact_count": raw_manifest["raw_files_count"],
            "preserved_historical_stop_report": str(RUN_DIR / "ROUND44_STOP_REPORT_EVAL_RECEIPT_MISMATCH.md"),
        }
        _write_new_or_match(RUN_DIR / "ROUND44_STATUS_COMPLETE_EVAL_R1.json", status)
        task = read_json(TASK_PATH)
        task["status"] = "complete_local_delivery_pending_commits"
        task["current_head_at_persist"] = head
        task["current_stage"] = "ROUND44 fixed evaluation complete; awaiting local thematic commits, no push."
        task["next_action"] = "Commit the completed ROUND44 evaluation report, audit, machine verdict and source identity locally; then stop."
        task["current_state"]["counts"] = {
            **_counter_snapshot(ledger), "valid_new_eval_cells": 96,
            "planned_new_eval_cells": 96, "training_complete": 384,
        }
        task["current_quota"].update({
            "normal_used": counts["normal_used"], "normal_remaining": 0,
            "reserve_used": counts["reserve_used"], "reserve_remaining": 23,
            "total_used": counts["total_used"], "hard_cap_remaining": 23,
            "valid_new_eval_cells": 96,
        })
        task["quota"] = dict(task["current_quota"])
        task["active_execution_quota_snapshot"] = dict(task["current_quota"],
            training_complete=384, training_authorized_again=False,
            failed_key=list(RETRY_KEY))
        task["final_report"] = {"path": report["path"], "sha256": report["sha256"]}
        task["eval_execution_identity"] = {
            "path": str(EVAL_IDENTITY), "sha256": sha256_file(EVAL_IDENTITY),
            "source_commit": identity["source_commit"], "execution_commit": head,
        }
        _write_atomic(TASK_PATH, task)
        return {"progress": progress, "ledger": ledger, "analysis": analysis,
                "eval_audit": eval_audit, "report": report, "status": status,
                "raw_manifest": raw_manifest}
    finally:
        _release_lock(lock)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prelaunch-audit-only", action="store_true",
                        help="verify and read-only forward ep128 models; launch no environments")
    args = parser.parse_args(argv)
    if args.prelaunch_audit_only:
        result = prelaunch_audit()
        print(json.dumps({"status": result["status"],
                          "source_commit": result["source_or_execution_head"],
                          "models": len(result["model_loads_and_forward"]),
                          "environment_constructor_calls": 0,
                          "optimizer_step_calls_delta": 0},
                         ensure_ascii=False, indent=2), flush=True)
        return 0
    result = run_evaluation()
    print(json.dumps({"status": result["progress"]["status"],
                      "counts": result["ledger"]["counts"],
                      "scientific_result": result["analysis"]["machine_verdict"][
                          "scientific_result"],
                      "report": result["report"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
