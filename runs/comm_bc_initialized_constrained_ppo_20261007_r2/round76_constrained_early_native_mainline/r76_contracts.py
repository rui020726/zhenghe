"""Pure ROUND76 A0 contracts: identities, masks, configuration, and field sets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

TASK_ID = "R76_A_MAINLINE_WIRING_PREPARATION_20261016"
BRANCH = "exp/round76-constrained-early-native-mainline"
REQUEST_ID = TASK_ID

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent

# The R71 calibration INIT is a model-only source.  Its optimizer, lambda and RNG are never
# restored into R76.  Only the five module values are transferred after payload validation.
R71_INIT_ROOT = RUN_ROOT / "round71_critic_initialization_calibration" / "raw" / "checkpoints"
R71_INIT_FILES = {
    11: ("v2_train_s11_init.pt", 50111,
         "fc25ead19bce38d00ae3bf619971ab717084711bdd2b5eb0431d1dda720a952f"),
    22: ("v2_train_s22_init.pt", 50175,
         "d95b9a27d5c28c8cd0b80c63b3539c78763bbaf53b79734a522e8e5a59cd8f81"),
    33: ("v2_train_s33_init.pt", 50175,
         "27baa262443a5ca6c6fb15bd60e0b294f6313324fe84cf386a2ac24377a874a9"),
}
R71_CHECKPOINT_INDEX_SHA256 = (
    "0c027d4817325bf4e02186d658ac52b0ccf6d9917a458380f72b8de8ea2a5009")

R71_STATE_SCHEMA = ("constrained-cost-state/2+partitioned-actor-head/1+frozen-early-path/1"
                    "+r68-late-init-alpha/1+r71-critic-calibration-init/1")
R71_TRAINER_TAG = "constrained-cost-v2-frozen-early-path-late-init-alpha-critic-calibration"
R71_TRAINING_VERSION = (
    "constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/"
    "dual-eta-lambda-0p5/1/rule-bc-late147-init/1/published-completion-event-at-relay-tick/2"
    "+r53-critic-representation-stop-gradient/1+r56-soft-early-report-prior-beta0p10/1"
    "+r58-prior-strength-beta1p00/1+r60-partitioned-early-late-actor-head/1"
    "+r66-frozen-early-path-late-adaptation/1+r68-late-initial-confidence-alpha0.10/1"
    "+r71-critic-calibration-init/1")
R71_LOSS_VERSION = "constrained-lagrangian/2+partitioned-actor-head/1+frozen-early-path/1"
R71_OPTIMIZER_SCHEMA = ("stage-c-frozen-early-path-six-module-parameter-set/1"
                        "+r68-late-init-alpha/1+r71-critic-calibration-init/1")
R71_STRUCTURE = ("partitioned-early-late-actor-head/1+frozen-early-path-late-adaptation/1"
                 "+late-initial-confidence-alpha0.10/1+r71-critic-calibration-heads/1")

R76_TRAINING_VERSION = "round76-constrained-early-native-mainline/1"
R76_LOSS_VERSION = "constrained-lagrangian/2+partitioned-actor-head/1+r76-native-early-mainline/1"
R76_STATE_SCHEMA = "constrained-cost-state/2+partitioned-actor-head/1+r76-native-early-mainline/1"
R76_OPTIMIZER_SCHEMA = "stage-c-native-early-mainline-eight-object-parameter-set/1"
R76_STRUCTURE = ("partitioned-early-late-actor-head/1+r76-native-early-mainline-trunk-frozen/1")
R76_TRAINER_TAG = "constrained-cost-v2-r76-native-early-mainline"

R76_MODULES = ("trunk", "early_head", "late_head", "value_task_head", "value_cost_head")
R76_FROZEN_MODULES = ("trunk",)
R76_TRAINABLE_MODULES = ("early_head", "late_head", "value_task_head", "value_cost_head")
R76_FROZEN_OBJECT_COUNT = 4
R76_TRAINABLE_OBJECT_COUNT = 8
R76_FROZEN_SCALAR_COUNT = 5440
R76_TRAINABLE_SCALAR_COUNT = 390
R76_TOTAL_SCALAR_COUNT = 5830
R76_SOURCE_MANIFEST_NAMES = (
    "r71_trial_execute.py",
    "r71_runtime_binding.py",
    "r68_execute.py",
    "r75_frozen.py",
    "v2_trainer.py",
    "r42_one_shot.py",
)

R76_PINNED_SOURCE_MANIFEST = {
    "r71_trial_execute.py": "66417acdfd72b15d9baf5f1b2a1524dcf09cc754e9e48af699c6a46159b18721",
    "r71_runtime_binding.py": "b16396e01c5aa3f5d91ab99b026aa10c3646a562897e632d24579426aa3e9812",
    "r68_execute.py": "c123dfe753effc5b6e35b8ce680e6bb1092a80a0d0deb40a0971df09e1c2cd3e",
    "r75_frozen.py": "bb4bae98518cc482f4b733794ee1c3f93097c56c5b3e85d6fed2c65f2a72198e",
    "v2_trainer.py": "0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841",
    "r42_one_shot.py": "3c342108148de42f395faeec25544976eed7efdfdd32a99685d73bd59101ab0d",
}

R76_LIVE_CONFIG = {
    "common.GAMMA": 1.0,
    "common.GAE_LAMBDA": 1.0,
    "common.ETA_LAMBDA": 0.5,
    "common.B0_BYTES": 10000.0,
    "common.DATA_ORDER_SEED": 20260927,
    "common.EPOCHS_PER_BATCH": 4,
    "common.MINIBATCH_STEPS": 256,
    "r66.READY_CUT": 147.0,
    "r66.HORIZON": 1000.0,
    "r66.GATE_BETA": 1.0,
    "r68.LATE_INIT_ALPHA": 0.10,
}

R76_FULL_CONTRACT = {
    "kind": "round76-native-early-mainline-full-resume/1",
    "task_id": TASK_ID,
    "branch": BRANCH,
    "training_version": R76_TRAINING_VERSION,
    "loss_version": R76_LOSS_VERSION,
    "state_schema": R76_STATE_SCHEMA,
    "optimizer_schema": R76_OPTIMIZER_SCHEMA,
    "structure": R76_STRUCTURE,
    "frozen_modules": list(R76_FROZEN_MODULES),
    "trainable_modules": list(R76_TRAINABLE_MODULES),
    "optimizer_param_groups_lengths": [8],
    "init_policy": ("R71 calibrated model-only five modules; late x0.10 is already baked in "
                    "and is never reapplied"),
    "science_delta_vs_r71": "early_head trainable; trunk remains frozen",
}

R76_REQUIRED_TAPE_FIELDS = (
    "schema", "comm_action_semantics", "comm_action_eligibility", "base_action_mask_schema",
    "step", "targets", "features", "actions", "effective_actions", "requested",
    "decision_class", "free_one_shot_choices", "free_first_reports", "free", "actionable",
    "masked", "masked_reasons", "fixed_hold_one_shot_closed", "fixed_hold_refresh",
    "fixed_hold_reasons", "forced", "attempt_charge_bytes", "send_executable_initial",
    "send_block_reason_initial", "candidate_audit", "relay_observation_remaining_bytes",
    "shadow_budget_before", "shadow_budget_after", "shadow_capacity_before",
    "shadow_capacity_after", "log_prob", "value", "cost_value", "send_probabilities",
    "one_shot_state_after_step",
)
R76_REQUIRED_ROW_FIELDS = (
    "kind", "label", "scenario_seed", "attempt", "policy", "spent_bytes", "budget_bytes",
    "airtime_tap_total", "emissions", "on_time", "completed", "expired", "discovered",
    "active", "unassigned", "task_reward_total", "training_version",
    "reward_timing_version", "b0_bytes", "normalised_cost_total", "emission_sequence",
    "refusal_sequence", "cost_stream", "task_stream", "steps", "artefacts",
    "training_seed", "episode_index", "lambda",
)
FOREIGN_STATE_SCHEMAS = {
    R71_STATE_SCHEMA,
    "constrained-cost-state/2+partitioned-actor-head/1+native-ppo-qualification/1",
}

R76_REAL_EXECUTION_CONTRACT_KIND = "round76-real-training-contract/1"
R76_REAL_EXECUTION_CONTRACT_FIELDS = (
    "kind", "status", "request_id", "task_id", "branch", "source_head", "scope",
    "resource_root", "driver_sha256", "source_manifest", "training_matrix",
    "resource_caps",
)
REGISTERED_REAL_CONTRACT_SHA256: str | None = None
R76_RESOURCE_UNITS = (
    "physical_optimizer_step", "offline_trunk_target_row", "joint_record",
    "environment_construction",
)


def sha256_text(value: Any, *, encoding: str = "utf-8") -> str:
    if isinstance(value, bytes):
        data = value
    else:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode(encoding)
    return hashlib.sha256(data).hexdigest()


def validate_exact_keys(value: Any, required: tuple[str, ...], label: str = "value") -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a dict")
    missing = [name for name in required if name not in value]
    if missing:
        raise RuntimeError(f"{label} is missing required fields: {missing}")
    if len(value) != len(set(value)):
        raise RuntimeError(f"{label} has duplicate-like non-string keys")


def validate_r76_mask_report(mask: dict[str, Any]) -> dict[str, Any]:
    required = {
        "frozen_modules": list(R76_FROZEN_MODULES),
        "trainable_modules": list(R76_TRAINABLE_MODULES),
        "frozen_parameter_object_count": R76_FROZEN_OBJECT_COUNT,
        "trainable_parameter_object_count": R76_TRAINABLE_OBJECT_COUNT,
        "frozen_scalar_count": R76_FROZEN_SCALAR_COUNT,
        "trainable_scalar_count": R76_TRAINABLE_SCALAR_COUNT,
    }
    for key, want in required.items():
        if mask.get(key) != want:
            raise RuntimeError(f"R76 mask {key}={mask.get(key)!r} != {want!r}")
    if R76_FROZEN_SCALAR_COUNT + R76_TRAINABLE_SCALAR_COUNT != R76_TOTAL_SCALAR_COUNT:
        raise RuntimeError("R76 mask scalar counts do not sum to the five-module total")
    return mask


R71_CALIBRATION_SHA_KEYS = {
    "trunk": "trunk_sha256",
    "early_head": "early_head_sha256",
    "late_head": "late_head_sha256",
    "value_task_head": "value_task_head_sha256",
    "value_cost_head": "value_cost_head_sha256",
}
R71_REGISTERED_INIT_MASK = {
    "frozen_parameter_object_count": 6,
    "trainable_parameter_object_count": 6,
    "frozen_scalar_count": 5570,
    "trainable_scalar_count": 260,
    "frozen_modules": ["trunk", "early_head"],
    "trainable_modules": ["late_head", "value_task_head", "value_cost_head"],
}
R71_PAYLOAD_FROZEN_MASK_REQUIRED_FIELDS = (
    "frozen_modules", "trainable_modules", "frozen_sha256", "frozen_scalar_count",
    "frozen_parameter_object_count", "critic_heads_calibrated", "note",
)


def validate_source_manifest(source_manifest: dict[str, str], *,
                            sha256_file=None) -> dict[str, str]:
    expected = R76_PINNED_SOURCE_MANIFEST
    if set(source_manifest) != set(expected):
        raise RuntimeError("R76 source manifest keys do not match the pinned source set")
    result = {}
    for name, want in expected.items():
        got = source_manifest[name]
        if sha256_file is not None:
            got = sha256_file(source_manifest[name])
        if got != want:
            raise RuntimeError(f"R76 source {name} SHA256 {got} != {want}")
        result[name] = got
    return result


def sha256_contract_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_real_execution_contract(path, *, source_head: str,
                                     driver_path: Path | None = None) -> dict[str, Any]:
    """Parse and fully bind an externally frozen real-training contract.

    A CLI permit only names the contract and carries its file hash.  It is never treated
    as the scientific contract itself.
    """
    contract_path = Path(path)
    if REGISTERED_REAL_CONTRACT_SHA256 is None:
        raise RuntimeError(
            "no separately frozen real-training contract is registered for this R76 A")
    if REGISTERED_REAL_CONTRACT_SHA256 != sha256_contract_file(contract_path):
        raise RuntimeError(
            "the supplied real-training contract is not the registered frozen contract")
    if not contract_path.is_file():
        raise RuntimeError(
            "the issued permit names no existing separately frozen real-training contract")
    try:
        document = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError("the real-training contract is not valid JSON") from error
    validate_exact_keys(document, R76_REAL_EXECUTION_CONTRACT_FIELDS,
                        "real execution contract")
    if document["kind"] != R76_REAL_EXECUTION_CONTRACT_KIND:
        raise RuntimeError("the real-training contract kind is not R76")
    if document["status"] != "issued":
        raise RuntimeError("the real-training contract is not issued")
    if document["request_id"] != REQUEST_ID or document["task_id"] != TASK_ID \
            or document["branch"] != BRANCH or document["source_head"] != source_head:
        raise RuntimeError("the real-training contract identity does not match R76 A")
    if document["scope"] != "real_training_frozen_contract":
        raise RuntimeError("the real-training contract scope is not real training")
    canonical_root = (HERE / "raw" / "runtime").resolve()
    if Path(document["resource_root"]).resolve() != canonical_root:
        raise RuntimeError("the real-training contract resource root is not canonical")
    driver_path = Path(driver_path) if driver_path is not None else HERE / "r76_execute.py"
    actual_driver_sha = sha256_contract_file(driver_path)
    if document["driver_sha256"] != actual_driver_sha:
        raise RuntimeError(
            f"the real-training contract driver hash drifted: {actual_driver_sha}")
    source_manifest = document["source_manifest"]
    if not isinstance(source_manifest, dict) or source_manifest != R76_PINNED_SOURCE_MANIFEST:
        raise RuntimeError("the real-training contract source manifest is not the pinned set")
    matrix = document["training_matrix"]
    matrix_fields = ("kind", "seeds", "episodes_per_seed", "episodes_per_update",
                     "data_order_seed", "b0_bytes", "scenario_order_sha256")
    validate_exact_keys(matrix, matrix_fields, "real-training training matrix")
    if matrix["kind"] != "round76-training-matrix/1" \
            or matrix["seeds"] != [11, 22, 33] \
            or matrix["episodes_per_seed"] != 128 \
            or matrix["episodes_per_update"] != 4 \
            or float(matrix["data_order_seed"]) != 20260927.0 \
            or float(matrix["b0_bytes"]) != 10000.0 \
            or not isinstance(matrix["scenario_order_sha256"], str) \
            or len(matrix["scenario_order_sha256"]) != 64:
        raise RuntimeError("the real-training contract training matrix does not match R76")
    caps = document["resource_caps"]
    validate_exact_keys(caps, R76_RESOURCE_UNITS, "real-training resource caps")
    for unit in R76_RESOURCE_UNITS:
        unit_caps = caps[unit]
        validate_exact_keys(unit_caps, ("normal", "reserve", "hard"),
                            f"real-training {unit} caps")
        for category in ("normal", "reserve", "hard"):
            value = unit_caps[category]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"real-training {unit}.{category} cap is not a non-negative integer")
    return document
