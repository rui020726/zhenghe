"""ROUND46 adapter: BC-initialized constrained PPO with the frozen R44 runtime.

This entry point deliberately has no default action. It reuses the real R45 BC
model-only loader, R44 runtime configuration, R42 training entry, R43 first-batch
gate, and R44 final evaluation machinery. ROUND46 owns only its identities,
quota, fixed key sequence, execution order, and reports.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
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

ROUND46_BRANCH = "exp/comm-bc-initialized-constrained-ppo"
ROUND46_START_HEAD = "384377ca1fee530c9a43344e9bc392b5d31594b3"
ROUND46_RUNNER_VERSION = (
    "round46-bc-initialized-constrained-ppo/1+"
    "round45-model-only-bc-loader/1+"
    "round44-constrained-ppo-runtime/1+"
    "cross-seed-batch0-gate/1"
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

FROZEN_CONFIG_PATH = R46_DIR / "ROUND46_FROZEN_CONFIG.json"
CONFIG_DIFF_PATH = R46_DIR / "ROUND46_CONFIGURATION_DIFF.json"
KEY_PLAN_PATH = R46_DIR / "ROUND46_KEY_PLAN.json"
TRAINING_VERSION_PATH = R46_DIR / "TRAINING_VERSION_MANIFEST.json"
INPUT_IDENTITY_PATH = R46_DIR / "ROUND46_INPUT_IDENTITY.json"
RUNTIME_FREEZE_PATH = R46_DIR / "ROUND46_RUNTIME_FREEZE.json"
PRELAUNCH_GATE_PATH = R46_DIR / "ROUND46_PRELAUNCH_GATE.json"
PRELAUNCH_SHA_PATH = R46_DIR / "ROUND46_PRELAUNCH_GATE.sha256"
CURRENT_TASK_PATH = R46_DIR / "ROUND46_CURRENT_TASK.json"
LEDGER_PATH = R46_DIR / "runtime" / "ROUND46_EXECUTION_LEDGER.json"
REPORT_PATH = R46_DIR / "ROUND46_REPORT.md"
FINAL_RESULTS_PATH = R46_DIR / "ROUND46_FINAL_RESULTS.json"
PAIRED_RESULTS_PATH = R46_DIR / "ROUND46_PAIRED_RESULTS.json"
TARGET_DIFFS_PATH = R46_DIR / "ROUND46_TARGET_ID_DIFFS.json"
CURVES_PATH = R46_DIR / "ROUND46_TRAINING_CURVES.json"
CHECKPOINT_INDEX_PATH = R46_DIR / "ROUND46_CHECKPOINT_INDEX.json"
ARTIFACT_MANIFEST_PATH = R46_DIR / "ROUND46_ARTIFACT_MANIFEST.json"
STOP_STATE_PATH = R46_DIR / "ROUND46_STOP_STATE.json"

_R45_MODULE: Any = None
_RUNTIME: dict[str, Any] | None = None


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
    if check:
        return result.stdout.strip()
    return result.stdout.strip() + ("\n" + result.stderr.strip() if result.stderr else "")


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
    if status:
        raise RuntimeError(f"tracked worktree is dirty; refusing ROUND46 environment entry: {status}")


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
    # R45's verified loaders use fixed filenames. These are byte-identical
    # provenance aliases inside the ignored ROUND46 runtime directory.
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
    for seed, model in source_identity["bc_models_by_seed"].items():
        for key, destination_name in (
            ("readback_receipt_path", f"BC_CHECKPOINT_RELOAD_VERIFICATION_s{seed}.json"),
            ("metrics_path", f"BC_METRICS_s{seed}.json"),
            ("journal_path", f"bc_journals/BC_OPTIMIZER_STEPS_s{seed}.jsonl"),
        ):
            copy_pairs.append((Path(model[key]), R46_DIR / destination_name))
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


def _configure_runtime_module() -> Any:
    global _R45_MODULE, _RUNTIME
    if _R45_MODULE is not None and _RUNTIME is not None:
        return _R45_MODULE
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
    module.RUN_DIR = R46_DIR
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

    runtime = module.load_runtime(enforce_prelaunch=False)
    if runtime["r42_versions"].TRAINING_VERSION != TRAINING_VERSION:
        raise RuntimeError("ROUND46 actual R42 trainer loaded a different training semantic version")
    common = runtime["common"]
    r40_common = runtime["r40_common"]
    common.NORMAL_LAUNCHES = NORMAL_CAP
    common.RESERVE_LAUNCHES = RESERVE_CAP
    common.LAUNCH_CAP = HARD_CAP
    r40_common.NORMAL_CAP = NORMAL_CAP
    r40_common.RESERVE_CAP = RESERVE_CAP
    r40_common.HARD_CAP = HARD_CAP
    r40_common.ROUND_DIR = R46_DIR
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
    _R45_MODULE = module
    _RUNTIME = runtime
    return module


def _input_identity() -> dict[str, Any]:
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
    }


def prepare() -> None:
    _assert_branch_head(exact_start=True)
    _assert_source_worktree_clean()
    if not (R46_DIR / "EXECUTION_BASIS.md").is_file():
        raise FileNotFoundError("EXECUTION_BASIS.md must be saved before ROUND46 prepare")
    basis_first = (R46_DIR / "EXECUTION_BASIS.md").read_text(encoding="utf-8").splitlines()[0]
    if basis_first != "当前任务=ROUND46；不是ROUND43，不恢复失败的ROUND45；384训练+96评估，储备24，hard504。":
        raise RuntimeError("ROUND46 execution basis first-line identity mismatch")

    input_identity = _input_identity()
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
        "kind": "round46-training-version-manifest/1",
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
    config_diff = {
        "kind": "round46-configuration-diff/1",
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
        "kind": "round46-execution-freeze/1",
        "status": "prepared_before_environment",
        "round": 46, "branch": ROUND46_BRANCH,
        "start_head": ROUND46_START_HEAD, "prepare_head": _run_git("rev-parse", "HEAD"),
        "runner_version": ROUND46_RUNNER_VERSION,
        "runner_path": runner_path.relative_to(REPO).as_posix(),
        "runner_source_sha256": runner_sha,
        "training_version": TRAINING_VERSION,
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


def preflight() -> None:
    _assert_branch_head()
    _assert_source_worktree_clean()
    if not FROZEN_CONFIG_PATH.is_file() or not KEY_PLAN_PATH.is_file():
        raise RuntimeError("ROUND46 prepare/freeze files are missing")
    frozen = _json(FROZEN_CONFIG_PATH)
    if (frozen.get("status") != "prepared_before_environment"
            or frozen.get("runner_source_sha256") != _sha256(Path(__file__))
            or _sha256(KEY_PLAN_PATH) != frozen.get("key_plan_sha256")
            or _sha256(CONFIG_DIFF_PATH) != frozen.get("configuration_diff_sha256")
            or _sha256(TRAINING_VERSION_PATH) != frozen.get("training_version_manifest_sha256")
            or _sha256(INPUT_IDENTITY_PATH) != frozen.get("input_identity_sha256")):
        raise RuntimeError("ROUND46 frozen runner or logical key plan identity failed")
    current_head = _run_git("rev-parse", "HEAD")
    if (current_head == frozen.get("prepare_head")
            or _run_git("rev-parse", "HEAD^") != frozen.get("prepare_head")):
        raise RuntimeError("ROUND46 code/config freeze must be a single local commit before preflight")
    runner_object = _run_git("rev-parse", f"HEAD:{Path(__file__).resolve().relative_to(REPO).as_posix()}")
    runner_blob = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "blob", runner_object],
        check=True, capture_output=True,
    ).stdout
    if hashlib.sha256(runner_blob).hexdigest() != frozen["runner_source_sha256"]:
        raise RuntimeError("ROUND46 committed runner bytes differ from executed source")

    module = _configure_runtime_module()
    runtime = _RUNTIME
    assert runtime is not None
    actual_inputs = _input_identity()
    if actual_inputs != _json(INPUT_IDENTITY_PATH):
        raise RuntimeError("ROUND46 source inputs changed after the preparation freeze")
    import torch
    import r45_bc
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
        "optimizer_step_calls": 0, "checkpoint_write_calls": 0,
        "network_or_push": False,
    }
    _write_json(PRELAUNCH_GATE_PATH, receipt)
    _write_text_lf(PRELAUNCH_SHA_PATH, _sha256(PRELAUNCH_GATE_PATH) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def _verify_prelaunch_identity(runtime: dict[str, Any]) -> None:
    _assert_branch_head()
    _assert_source_worktree_clean()
    frozen = _json(FROZEN_CONFIG_PATH)
    receipt = _json(PRELAUNCH_GATE_PATH)
    pointer = PRELAUNCH_SHA_PATH.read_text(encoding="ascii").strip()
    if (_sha256(PRELAUNCH_GATE_PATH) != pointer
            or receipt.get("kind") != "round46-zero-environment-preflight/1"
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
    ancestor = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor",
         receipt["head_at_preflight"], "HEAD"], capture_output=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("ROUND46 preflight commit is not an ancestor of current HEAD")
    for relative, expected in (
        (receipt["runner_path"], receipt["runner_source_sha256"]),
        (PRELAUNCH_GATE_PATH.relative_to(REPO).as_posix(), pointer),
        (PRELAUNCH_SHA_PATH.relative_to(REPO).as_posix(),
         hashlib.sha256((pointer + "\n").encode("ascii")).hexdigest()),
    ):
        blob_id = _run_git("rev-parse", f"HEAD:{relative}")
        blob = subprocess.run(
            ["git", "-C", str(REPO), "cat-file", "blob", blob_id],
            check=True, capture_output=True,
        ).stdout
        actual = hashlib.sha256(blob).hexdigest()
        if actual != expected:
            raise RuntimeError(f"ROUND46 committed prelaunch identity differs: {relative}")
    if _sha256(R45_CONTROL_GATE_PATH) != frozen["inputs"]["round45_control_gate_sha256"]:
        raise RuntimeError("ROUND45 old failed gate was modified")
    if ((int(runtime["common"].NORMAL_LAUNCHES), int(runtime["common"].RESERVE_LAUNCHES),
            int(runtime["common"].LAUNCH_CAP)) != (NORMAL_CAP, RESERVE_CAP, HARD_CAP)):
        raise RuntimeError("ROUND46 runtime quota drifted after preflight")


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
        checkpoint = R46_DIR / "checkpoints" / f"v2_train_s{seed}_ep128.pt"
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
        for path in sorted((R46_DIR / "rows").glob("*.json")):
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
            candidates = sorted((R46_DIR / "checkpoints").glob(
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
            path = R46_DIR / "checkpoints" / f"v2_train_s{seed}_ep{episode}.pt"
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
                candidates = list((R46_DIR / "checkpoints").glob(
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
        R46_DIR / "rows", R46_DIR / "logs", R46_DIR / "launch_claims",
        R46_DIR / "checkpoints", R46_DIR / "training_progress",
        R46_DIR / "training_reports", R46_DIR / "failures", R46_DIR / "rejected",
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
    if not PRELAUNCH_GATE_PATH.is_file() or not PRELAUNCH_SHA_PATH.is_file():
        raise RuntimeError("ROUND46 zero-environment preflight receipt must be committed first")
    module = _configure_runtime_module()
    runtime = _RUNTIME
    assert runtime is not None
    module.verify_prelaunch_identity = lambda _frozen=None: _verify_prelaunch_identity(runtime)
    module.ENFORCE_PRELAUNCH = True
    _verify_prelaunch_identity(runtime)
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
    group.add_argument("--execute", action="store_true",
                       help="run only the frozen 384 training + 96 final evaluation plan")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.preflight:
        preflight()
    elif args.execute:
        execute()
    else:
        raise RuntimeError("ROUND46 requires exactly one explicit mode")


if __name__ == "__main__":
    main()
