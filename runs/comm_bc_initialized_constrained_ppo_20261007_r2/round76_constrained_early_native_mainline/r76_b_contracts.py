"""Pure ROUND76 B qualification contracts (stdlib only, no real execution)."""

from __future__ import annotations

import json
import hashlib
import string
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent

REQUEST_ID = "R76_B0_SOURCE_PREPARATION_20261016"
TASK_ID = "R76_B_R71_NATIVE_RESTORE_QUALIFICATION_20261016"
BRANCH = "exp/round76-constrained-early-native-mainline"
BASE_HEAD = "f509ea35a0d0b65769079fa8e5e027565d8672f3"
PARENT_HEAD = "dc363721cc2954eabce04e34e2ccc578866ee468"

RESOURCE_ROOT = HERE / "raw" / "qualification"
CONTRACT_KIND = "round76-b-qualification-contract/1"
PERMIT_KIND = "round76-b-qualification-permission/1"

SAMPLES_PER_BATCH = 265
EPOCHS_PER_BATCH = 4
MINIBATCH_STEPS = 256
OPTIMIZER_STEPS_PER_NATIVE = 8
PHYSICAL_STEPS_PER_BRANCH = 9
TOTAL_PHYSICAL_STEPS = 17
MAX_UNIQUE_RECORDS = 4000
MAX_SINGLE_TRUNK_ROWS = 1024
MAX_TRUNK_ROWS = 15900
MAX_TRUNK_CALLS = 546
ENVIRONMENT_CONSTRUCTIONS = 0

QUALIFICATION_LIMITS = {
    "samples_per_batch": SAMPLES_PER_BATCH,
    "epochs_per_batch": EPOCHS_PER_BATCH,
    "minibatch_steps": MINIBATCH_STEPS,
    "optimizer_steps_per_native": OPTIMIZER_STEPS_PER_NATIVE,
    "physical_steps_per_branch": PHYSICAL_STEPS_PER_BRANCH,
    "total_physical_steps": TOTAL_PHYSICAL_STEPS,
    "max_unique_records": MAX_UNIQUE_RECORDS,
    "max_single_trunk_rows": MAX_SINGLE_TRUNK_ROWS,
    "max_trunk_rows": MAX_TRUNK_ROWS,
    "max_trunk_calls": MAX_TRUNK_CALLS,
    "environment_constructions": ENVIRONMENT_CONSTRUCTIONS,
}

RESOURCE_CAPS = {
    "physical_optimizer_step": {"normal": 17, "reserve": 0, "hard": 17},
    "offline_trunk_target_row": {"normal": 15900, "reserve": 0, "hard": 15900},
    "trunk_forward_call": {"normal": 546, "reserve": 0, "hard": 546},
    "joint_record": {"normal": 2, "reserve": 0, "hard": 2},
    "environment_construction": {"normal": 0, "reserve": 0, "hard": 0},
}

SOURCE_MANIFEST_FILES = (
    "r76_b_contracts.py",
    "r76_b_driver.py",
    "r76_b_entry.py",
    "r76_cli.py",
    "r76_contracts.py",
    "r76_episode_protocol.py",
    "r76_execute.py",
    "r76_guard.py",
    "r76_runtime.py",
    "r76_native.py",
    "r76_b_parent.ps1",
    "r76_b_parent_r1.ps1",
    "test_r76_b0_source_review.py",
    "test_r76_protocol_history_r1.py",
)

B_STAGE_CAPS = {
    "kind": "round76-b-stage-caps/1",
    "b0": {
        "python_processes": 0,
        "fixed_case_executions": 0,
    },
    "b1": {
        "python_processes": 2,
        "distinct_cases": 3,
        "fixed_case_executions": 6,
    },
    "b2": {
        "cpu_nn_python_processes": 1,
        "normal_optimizer_steps": 17,
        "reserve_optimizer_steps": 0,
        "hard_optimizer_steps": 17,
        "offline_trunk_target_rows": 15900,
        "single_trunk_input_rows": 1024,
        "max_trunk_calls": 546,
        "batch_reads": 2,
        "factory_constructions": 2,
        "adam_constructions": 2,
        "init_loads": 1,
        "own_a1_loads": 1,
        "checkpoint_saves": 3,
        "checkpoint_loads": 2,
        "environment_constructions": 0,
    },
    "old_r75_hard_optimizer_steps_preserved": 16,
}

B0_TEST_CASES = (
    "test_contract_identity_permit_and_resource_role_refusals",
    "test_b0_state_entry_and_pure_orchestrate_close_17_steps",
    "test_native_scientific_ast_and_guard_fixture_isolation",
)

B0_NEW_SOURCE_FILES = (
    "r76_b_contracts.py",
    "r76_b_driver.py",
    "r76_b_entry.py",
    "r76_b_parent.ps1",
    "test_r76_b0_source_review.py",
)

OLD_REPORT = {
    "path": "round71_critic_initialization_calibration/raw/checkpoints/"
            "r40_batch_s11_b0_attempt01.json",
    "bytes": 5736,
    "sha256": "19743d595783bb85a640ab78ca6790f0fabc90ac02ad36a8a10ee8785c44ca8e",
}

def sha256_contract_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_manifest(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(SOURCE_MANIFEST_FILES):
        raise BContractError("qualification source manifest is missing or contains foreign files")
    allowed = set(string.hexdigits.lower())
    for name, want in value.items():
        if not isinstance(want, str) or len(want) != 64 \
                or any(char not in allowed for char in want.lower()):
            raise BContractError(f"qualification source {name} SHA256 is malformed")
        source_path = HERE / name
        if not source_path.is_file():
            raise BContractError(f"qualification source file is missing: {name}")
        got = sha256_contract_file(source_path)
        if got != want.lower():
            raise BContractError(f"qualification source {name} drifted: {got} != {want}")
    return value


def zero_b0_consumption() -> dict[str, Any]:
    return {
        "python_process": 0,
        "fixed_case_execution": 0,
        "model_construction_or_load": 0,
        "forward_backward": 0,
        "optimizer_step": 0,
        "environment": 0,
        "real_rng_restore": 0,
        "serializer": 0,
        "production_ledger": 0,
    }


def validate_stage_caps(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value != B_STAGE_CAPS:
        raise BContractError("B stage caps do not match the approved envelope")
    return value


def build_b0_state() -> dict[str, Any]:
    return {
        "kind": "round76-b-task-state/1",
        "request_id": REQUEST_ID,
        "task_id": TASK_ID,
        "branch": BRANCH,
        "base_head": BASE_HEAD,
        "parent_head": PARENT_HEAD,
        "stage": "B0_SOURCE_PREPARATION",
        "status": "SOURCE_REVIEW_NOT_PASSED",
        "caps": B_STAGE_CAPS,
        "b0_new_source_files": list(B0_NEW_SOURCE_FILES),
        "b0_test_cases": list(B0_TEST_CASES),
        "gates": {
            "b0_source_review_passed": False,
            "b1_pure_tests_released": False,
            "b2_cpu_qualification_released": False,
        },
        "consumption": {
            "b0": zero_b0_consumption(),
            "b1": {"status": "not_released", "consumption": None},
            "b2": {"status": "not_released", "consumption": None},
        },
        "scope_note": "B0 is source preparation only; it does not run tests or release B1/B2.",
    }

R71_INIT = {
    "path": "round71_critic_initialization_calibration/raw/checkpoints/"
            "v2_train_s11_init.pt",
    "bytes": 50111,
    "sha256": "fc25ead19bce38d00ae3bf619971ab717084711bdd2b5eb0431d1dda720a952f",
}

EPISODES = (
    {
        "episode": 0, "scenario": 1300027, "attempt": 2,
        "row_path": "round71_critic_initialization_calibration/raw/rows/"
                    "v2_r71_train_s11_e0_s1300027_a2.json",
        "row_bytes": 341408,
        "row_sha256": "727c4faa8b391c7471524c291b630850ba6a63d3188d7b6d0f75bb904a04cf71",
        "tape_path": "round71_critic_initialization_calibration/raw/logs/"
                     "v2_r71_train_s11_e0_s1300027_a2_tape.jsonl",
        "tape_bytes": 215634,
        "tape_sha256": "7d2fa6b70fbcc3e66d7c013f0e96a163fadc58f1b0e3279cd96f2a5172c253ae",
    },
    {
        "episode": 1, "scenario": 1300014, "attempt": 1,
        "row_path": "round71_critic_initialization_calibration/raw/rows/"
                    "v2_r71_train_s11_e1_s1300014_a1.json",
        "row_bytes": 341491,
        "row_sha256": "a671508751a9ba0b3077899ef13a879114a751fb992a7c4255191dfaa197854a",
        "tape_path": "round71_critic_initialization_calibration/raw/logs/"
                     "v2_r71_train_s11_e1_s1300014_a1_tape.jsonl",
        "tape_bytes": 190843,
        "tape_sha256": "18a2c867f3e5d4aaa689365fb1d6ccca213408fab823ad78c5792803448c2adc",
    },
    {
        "episode": 2, "scenario": 1300028, "attempt": 1,
        "row_path": "round71_critic_initialization_calibration/raw/rows/"
                    "v2_r71_train_s11_e2_s1300028_a1.json",
        "row_bytes": 340948,
        "row_sha256": "de3b2ba09e2462f8c92634d5803c47d29fa688fe0bec52cd617f397593d8eab0",
        "tape_path": "round71_critic_initialization_calibration/raw/logs/"
                     "v2_r71_train_s11_e2_s1300028_a1_tape.jsonl",
        "tape_bytes": 188184,
        "tape_sha256": "ef33a3f9fe84ec8a06ad73cbb8aff195acab984b5517f1d2ce507a68a7f1e8b0",
    },
    {
        "episode": 3, "scenario": 1300044, "attempt": 1,
        "row_path": "round71_critic_initialization_calibration/raw/rows/"
                    "v2_r71_train_s11_e3_s1300044_a1.json",
        "row_bytes": 341667,
        "row_sha256": "c6868c0bd0479dbf9be0f49aa9c05fd9b74a59c0598f4c245e900831ffad35cd",
        "tape_path": "round71_critic_initialization_calibration/raw/logs/"
                     "v2_r71_train_s11_e3_s1300044_a1_tape.jsonl",
        "tape_bytes": 170262,
        "tape_sha256": "4e777d09c1038e51b7fc8aecf43ca47194a2d96ae231bfb573165a1d916bab1d",
    },
)

CONTRACT_FIELDS = (
    "kind", "status", "request_id", "task_id", "branch", "source_head", "scope",
    "resource_root", "driver_sha256", "source_manifest", "input_manifest",
    "resource_caps", "qualification_limits",
)
PERMIT_FIELDS = (
    "kind", "status", "request_id", "task_id", "branch", "source_head", "scope",
    "contract_sha256", "resource_root", "contract_path",
)


class BContractError(RuntimeError):
    pass


def validate_caps(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value != RESOURCE_CAPS:
        raise BContractError("qualification resource caps do not match the approved envelope")
    return value


def validate_limits(value: Any) -> dict[str, Any]:
    expected = QUALIFICATION_LIMITS
    if not isinstance(value, dict) or value != expected:
        raise BContractError("qualification limits do not match the approved envelope")
    return value


def validate_input_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BContractError("input manifest must be a dict")
    if value.get("old_report") != OLD_REPORT or value.get("r71_init") != R71_INIT:
        raise BContractError("old-report or R71-INIT identity drifted")
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(EPISODES) \
            or any(episode != expected for episode, expected in zip(episodes, EPISODES)):
        raise BContractError("fixed episode identities drifted")
    return value


def validate_contract(value: Any, *, source_head: str, driver_sha256: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(CONTRACT_FIELDS):
        raise BContractError("qualification contract fields do not match the contract")
    if value.get("kind") != CONTRACT_KIND or value.get("status") != "issued":
        raise BContractError("qualification contract is not an issued B2 contract")
    if value.get("request_id") != REQUEST_ID or value.get("task_id") != TASK_ID:
        raise BContractError("qualification contract identity does not match B")
    if value.get("branch") != BRANCH or value.get("source_head") != source_head:
        raise BContractError("qualification contract branch/source identity does not match")
    if value.get("scope") != "r71_native_restore_qualification":
        raise BContractError("qualification contract scope is not B2")
    if Path(value.get("resource_root", "")).resolve() != RESOURCE_ROOT.resolve():
        raise BContractError("qualification resource root is not the new canonical root")
    if value.get("driver_sha256") != driver_sha256:
        raise BContractError("qualification driver hash drifted")
    validate_source_manifest(value.get("source_manifest"))
    validate_caps(value.get("resource_caps"))
    validate_limits(value.get("qualification_limits"))
    validate_input_manifest(value.get("input_manifest"))
    return value


def validate_permit(value: Any, *, source_head: str,
                    contract_sha256: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(PERMIT_FIELDS):
        raise BContractError("qualification permit fields do not match the permit")
    if value.get("kind") != PERMIT_KIND or value.get("status") != "issued":
        raise BContractError("qualification permit is not issued")
    if value.get("request_id") != REQUEST_ID or value.get("task_id") != TASK_ID:
        raise BContractError("qualification permit identity does not match B")
    if value.get("branch") != BRANCH or value.get("source_head") != source_head:
        raise BContractError("qualification permit branch/source identity does not match")
    if value.get("scope") != "r71_native_restore_qualification":
        raise BContractError("qualification permit scope is not B2")
    if value.get("contract_sha256") != contract_sha256:
        raise BContractError("qualification permit does not bind the contract file")
    if Path(value.get("resource_root", "")).resolve() != RESOURCE_ROOT.resolve():
        raise BContractError("qualification permit resource root is not canonical")
    return value
