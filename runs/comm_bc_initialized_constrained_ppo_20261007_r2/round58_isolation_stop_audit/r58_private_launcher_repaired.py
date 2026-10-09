"""Private, execution-disabled ROUND58 adapter using the real R58/R40 route.

The historic launcher remains byte-for-byte untouched. This audit launcher can
configure and bind the exact live modules in a private output root, but cannot
start work: a new authorization, freeze identity, and cumulative quota receipt
are not present in this checkout.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import hashlib
import argparse
import copy
import types
import builtins
import hashlib
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parents[0]
REPO_ROOT = HERE.parents[2]
GEN2 = HERE.parent / "round58_gen2"
R53 = HERE.parent / "round53_critic_isolation_trial_readiness"
R45 = HERE.parents[2] / "runs" / "comm_rule_warmstart_cost_learning_20261006_r2"
CANON = HERE.parents[2].parent / "zhenghe" / "runs" / "comm_cost_v2_rerun_20260928"
FROZEN_SP_COMMON_SHA256 = "b119e430c95db23e39f8214d4dab2eeab0853f23a8bf7d80f8758a898fc9f0dc"
FROZEN_SP_COMMON_BINDING_VERSION = "round58-exact-stage-b-sp-common-source/1"

for _path in (GEN2, R53, R45, CANON):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import r58_gate_prior_repaired as repaired_adapter  # noqa: E402
import r58_repaired_entry as resume_guard  # noqa: E402
import r58_checkpoint_contract_repaired as checkpoint_contract  # noqa: E402
import r58_identity_ledger_close_v2 as identity_ledger  # noqa: E402
import r58_resume_seam as resume_seam  # noqa: E402

PRIVATE_GENERATION = RUN_ROOT / "round58_gen3"
PRIVATE_RAW = PRIVATE_GENERATION / "raw"
REPAIRED_TRAINING_VERSION = (
    "constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/"
    "dual-eta-lambda-0p5/1/rule-bc-late147-init/1/"
    "published-completion-event-at-relay-tick/2+r53-critic-representation-stop-gradient/1+"
    "r56-soft-early-report-prior-beta0p10/1+r58-prior-strength-beta1p00/1+"
    "r58-repaired-private-generation3/1+r58-cumulative-quota-history16-normal1024-reserve24-hard1064/2"
)
RESUME_RUNTIME_ALIAS_HASHES = {
    "r40_train_resume_adapter.py": "688dee0c60a5ab7893991d10948a7c5cff12ec6ba6f91fce07c3439a0ee7f2ca",
    "r42_one_shot.py": "3c342108148de42f395faeec25544976eed7efdfdd32a99685d73bd59101ab0d",
    "r42_versions.py": "26b9e1a3d4f3b9a5d39ca020a5ebd3653615c09a76a77ae4138f4ab98420a3d8",
    "r43_call_counter_hooks.py": "02f80072a14d7305b619edd0db9617a7b0a0f31315b3265aebf1e694ac7398f7",
}
RESUME_APPROVAL_SOURCE_PATH = HERE / "ROUND58_GEN3_RESUME_APPROVAL_SOURCE.json"
RESUME_APPROVAL_THREAD_ID = "01a0a044-592f-73e3-838c-19c1d79465d3"
RESUME_APPROVAL_QUOTE = "核验通过后批准完成剩余固定计划"
_GRANT_TOKEN = object()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verified_sp_common_module(common_module: Any):
    """Load only the frozen Stage-B sp_common selected by the live v2_common."""
    expected_root = CANON.parents[1] / "runs" / "search_trace_and_policy_20260925"
    expected_source = expected_root / "sp_common.py"
    actual_v2 = Path(getattr(common_module, "__file__", "")).resolve()
    declared_root = Path(getattr(common_module, "ROUND18_DIR", "")).resolve()
    if actual_v2 != (CANON / "v2_common.py").resolve():
        raise PermissionError("refusing sp_common binding: live v2_common source mismatch")
    if declared_root != expected_root.resolve():
        raise PermissionError("refusing sp_common binding: live ROUND18_DIR differs from frozen source root")
    if (not expected_source.is_file()
            or _sha256_file(expected_source) != FROZEN_SP_COMMON_SHA256):
        raise PermissionError("frozen Stage-B sp_common source is missing or hash-mismatched")
    cached = sys.modules.get("sp_common")
    if cached is not None:
        cached_path = Path(getattr(cached, "__file__", "")).resolve()
        if (cached_path != expected_source.resolve()
                or _sha256_file(cached_path) != FROZEN_SP_COMMON_SHA256):
            raise PermissionError("refusing cached sp_common with unapproved provenance")
        return cached
    spec = importlib.util.spec_from_file_location("sp_common", expected_source)
    if spec is None or spec.loader is None:
        raise ImportError("cannot create exact frozen sp_common import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sp_common"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("sp_common", None)
        raise
    if (Path(getattr(module, "__file__", "")).resolve() != expected_source.resolve()
            or _sha256_file(module.__file__) != FROZEN_SP_COMMON_SHA256):
        sys.modules.pop("sp_common", None)
        raise PermissionError("loaded sp_common failed post-import provenance check")
    return module


class _ValidatedGrant:
    """In-process result of validating external human receipts; never a CLI input type."""
    def __init__(self, token: object, values: dict[str, Any]):
        if token is not _GRANT_TOKEN:
            raise PermissionError("validated grant objects are not user-constructible")
        self.values = values


def execute_authorized(*, authorization: dict[str, Any] | None = None,
                       freeze_identity: dict[str, Any] | None = None,
                       cumulative_quota: dict[str, Any] | None = None) -> None:
    """Unconditionally deny dictionary grants: CLI cannot turn test objects into execution.

    Actual execution is reachable only through ``main`` with externally stored receipts,
    and remains blocked unless all independent identities and cumulative accounting close.
    """
    raise PermissionError("ROUND58 execution requires external approval/freeze/quota/scene receipt files")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _read_external_json(path: Path, label: str) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise PermissionError(f"ROUND58 {label} must be an existing absolute receipt file")
    resolved = path.resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PermissionError(f"ROUND58 {label} must contain a JSON object")
    if value.get("fixture") is True or value.get("environment") == "test":
        raise PermissionError(f"ROUND58 {label}: test fixtures cannot authorize CLI execution")
    return value


def _validate_resume_approval_source_payload(value: dict[str, Any]) -> None:
    scope = value.get("approval_scope_as_relayed_by_owner", {})
    expected = {
        "resume_only_from_each_seed_episode_4": True,
        "remaining_training_episodes": 372,
        "remaining_evaluation_cells": 640,
        "replay_first_12_episodes": False,
        "replay_original_three_batch_updates": False,
        "new_training_or_evaluation_quota": False,
        "hyperparameter_change": False,
        "push_merge_deploy": False,
    }
    if (value.get("kind") != "round58-gen3-resume-owner-approval-source/1"
            or value.get("status") != "owner_approved_continue_after_exact_restore_gate"
            or value.get("source_thread_id") != RESUME_APPROVAL_THREAD_ID
            or value.get("approval_quote") != RESUME_APPROVAL_QUOTE
            or scope != expected or value.get("authorized_cumulative_limit") != 1064):
        raise PermissionError("ROUND58 resume owner approval source is missing or scope-mismatched")


def _resume_approval_source_identity() -> dict[str, str]:
    if not RESUME_APPROVAL_SOURCE_PATH.is_file():
        raise PermissionError("ROUND58 resume owner approval source file is missing")
    raw = RESUME_APPROVAL_SOURCE_PATH.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    _validate_resume_approval_source_payload(value)
    return {"path": str(RESUME_APPROVAL_SOURCE_PATH.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _bc_model_identity() -> tuple[dict[str, Any], str]:
    identity_path = RUN_ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    models = identity.get("bc_models_by_seed")
    if set(models or {}) != {"11", "22", "33"}:
        raise PermissionError("ROUND58 registered BC model identities are incomplete")
    body = {seed: {key: models[seed].get(key) for key in
                   ("checkpoint_path", "checkpoint_sha256", "parameter_sha256")}
            for seed in ("11", "22", "33")}
    if any(not all(isinstance(v, str) and v for v in item.values()) for item in body.values()):
        raise PermissionError("ROUND58 registered BC checkpoint/parameter identity is incomplete")
    return body, _canonical_sha(body)


def _execution_basis_identity() -> str:
    path = HERE / "ROUND58_GEN3_EXECUTION_BASIS.json"
    if not path.is_file():
        raise PermissionError("ROUND58 gen3 owner-approved execution basis is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "kind": "round58-gen3-execution-basis/1",
        "status": "owner_approved_execute",
        "task_id": "ROUND58-GEN3-FIXED-PLAN-EXECUTION",
        "source_channel": "direct current user instruction",
        "approved_historical_consumption": 16,
        "new_normal_cap": 1024,
        "new_reserve_cap": 24,
        "hard_cap": 1064,
        "execution_requested": True,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise PermissionError("ROUND58 execution basis is not the direct approved gen3 task")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repair_identity() -> dict[str, Any]:
    candidate_path = HERE / "ROUND58_PRELAUNCH_CANDIDATE.json"
    input_identity_path = HERE / "ROUND58_CANDIDATE_INPUT_IDENTITY.json"
    if not candidate_path.is_file() or not input_identity_path.is_file():
        raise PermissionError("ROUND58 candidate science/config/model identity inputs are missing")
    return {"critic_route_version": resume_guard.ROUTE_VERSION,
            **resume_guard.adapter_identity(),
            "train_pilot_source": checkpoint_contract.train_pilot_source_identity(REPO_ROOT),
            "stage_b_sp_common_source": {
                "path": str((CANON.parents[1] / "runs" / "search_trace_and_policy_20260925" / "sp_common.py").resolve()),
                "sha256": FROZEN_SP_COMMON_SHA256,
                "binding_version": FROZEN_SP_COMMON_BINDING_VERSION,
            },
            "resume_runtime_aliases": dict(RESUME_RUNTIME_ALIAS_HASHES),
            "ppo_counter_bridge_version": "round58-ppo-counter-bridge/2",
            "identity_ledger_schema": identity_ledger.IDENTITY_VERSION,
            "identity_ledger_sha256": hashlib.sha256(
                Path(identity_ledger.__file__).read_bytes()).hexdigest(),
            "private_launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checkpoint_contract_sha256": hashlib.sha256(
                Path(checkpoint_contract.__file__).read_bytes()).hexdigest(),
            "resume_seam_sha256": hashlib.sha256(
                Path(resume_seam.__file__).read_bytes()).hexdigest(),
            "approved_resume_map_sha256": resume_seam.MAP_SHA256,
            "historical_checkpoint_journal_sha256": resume_seam.validate_historical_checkpoint_journal(
                REPO_ROOT)["sha256"],
            "candidate_config_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            "candidate_input_identity_sha256": hashlib.sha256(
                input_identity_path.read_bytes()).hexdigest(),
            "execution_basis_sha256": _execution_basis_identity(),
            "training_version": REPAIRED_TRAINING_VERSION}


def _validate_cumulative_quota_v2(quota: dict[str, Any], history: dict[str, Any]) -> dict[str, int]:
    """Validate the append-only ROUND58-v2 quota without relabeling old claims.

    Historical category disagreement is deliberately carried as evidence metadata;
    the cumulative hard cap counts actual launches, while the normal/reserve caps
    apply only to this new generation's fixed plan and technical retries.
    """
    env = history.get("environment_consumption", {})
    actual_history = env.get("known_constructor_and_run_calls")
    claim_total = env.get("gen1_complete_claims", 0) + env.get("gen2_complete_claims", 0)
    c1, c2 = env.get("gen1_categories", {}), env.get("gen2_claim_categories", {})
    categorized = sum(c1.get(k, 0) for k in ("normal", "reserve")) + sum(
        c2.get(k, 0) for k in ("normal", "reserve"))
    if (quota.get("kind") != "round58-cumulative-quota-receipt/2"
            or quota.get("status") != "reconciled_with_preserved_category_conflict"
            or type(actual_history) is not int or actual_history != 16
            or type(claim_total) is not int or claim_total != actual_history
            or categorized != actual_history
            or env.get("gen2_registration_says_reserve_from_gen1_failure") != 4
            or not str(env.get("reserve_classification", "")).startswith("unreconciled")
            or quota.get("historical_category_disposition") != "unresolved_preserved_no_relabel"):
        raise PermissionError("ROUND58 v2 historical total/category evidence is missing or changed")
    fields = {
        "historical_consumed": actual_history,
        "planned_normal": 1024,
        "new_normal_cap": 1024,
        "new_reserve_cap": 24,
        "hard_cap": 1064,
        "normal_remaining": 1024,
        "reserve_remaining": 24,
        "hard_remaining": 1048,
    }
    for key, expected in fields.items():
        if type(quota.get(key)) is not int or quota.get(key) != expected:
            raise PermissionError(f"ROUND58 v2 cumulative quota mismatch: {key}")
    if quota.get("historical_category_disposition") != "unresolved_preserved_no_relabel":
        raise PermissionError("ROUND58 v2 may not relabel historical claim categories")
    total = actual_history + fields["planned_normal"] + fields["new_reserve_cap"]
    if total != fields["hard_cap"] or total > 1064:
        raise PermissionError("ROUND58 v2 cumulative hard-cap arithmetic does not close")
    return {"historical_consumed": actual_history,
            "normal_remaining": 1024,
            "reserve_remaining": 24,
            "hard_remaining": 1064 - actual_history,
            "planned_max_total": total}


def _validate_external_grant(*, authorization_path: Path, freeze_path: Path,
                             quota_path: Path, scene_audit_path: Path,
                             output_root: Path, mode: str) -> _ValidatedGrant:
    # This entire function is read-only and precedes legacy plan execution/configure.
    if mode not in ("fresh", "recover_first_batch_gate"):
        raise PermissionError("only frozen fresh start or attested first-batch continuation is admitted")
    resume_approval_source = (_resume_approval_source_identity()
                              if mode == "recover_first_batch_gate" else None)
    approval = _read_external_json(authorization_path, "human approval")
    frozen = _read_external_json(freeze_path, "freeze identity")
    quota = _read_external_json(quota_path, "cumulative quota receipt")
    scene_audit = _read_external_json(scene_audit_path, "scene occupancy audit")
    migration_proof_path = HERE / "ROUND58_GEN3_PLAN_IDENTITY_MIGRATION_PROOF_V18.json"
    if mode == "recover_first_batch_gate" and not migration_proof_path.is_file():
        raise PermissionError("ROUND58 recovery plan identity migration proof is missing")
    migration_proof_sha256 = (hashlib.sha256(migration_proof_path.read_bytes()).hexdigest()
                              if mode == "recover_first_batch_gate" else None)
    sys_path_before_legacy_load = list(sys.path)
    legacy = load_legacy_launcher()
    if (not isinstance(quota.get("historical_manifest_sha256"), str)
            or len(quota["historical_manifest_sha256"]) != 64):
        raise PermissionError("ROUND58 v2 quota requires a hash-closed historical manifest")
    history_path = Path(str(quota.get("historical_manifest_path", ""))).expanduser()
    if (not history_path.is_absolute() or not history_path.is_file()
            or hashlib.sha256(history_path.read_bytes()).hexdigest()
               != quota["historical_manifest_sha256"]):
        raise PermissionError("ROUND58 v2 history manifest is missing or hash-mismatched")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    quota_counts = _validate_cumulative_quota_v2(quota, history)
    historical_total = quota_counts["historical_consumed"]
    remaining_reserve = quota_counts["reserve_remaining"]
    contracts = sys.modules.get("r58_contracts")
    if contracts is None:
        raise RuntimeError("ROUND58 plan contract module was not imported")
    old_contract_cap, old_legacy_cap = contracts.RESERVE_CAP, legacy.RESERVE_CAP
    contracts.RESERVE_CAP = remaining_reserve
    legacy.RESERVE_CAP = remaining_reserve
    try:
        plan_base = legacy.plan_from_frozen_sources()
    finally:
        contracts.RESERVE_CAP = old_contract_cap
        legacy.RESERVE_CAP = old_legacy_cap
        sys.modules.pop("round58_legacy_launcher_readonly", None)
        sys.path[:] = sys_path_before_legacy_load
    repair = _repair_identity()
    resume_sources = resume_seam.validate_resume_sources(REPO_ROOT)
    plan = copy.deepcopy(plan_base)
    plan["training_version"] = REPAIRED_TRAINING_VERSION
    plan["r58_repair_identity"] = repair
    plan["generation_id"] = "round58_gen3"
    plan["status"] = "frozen_approved_gen3"
    plan["authorization"] = "direct_user_owner_approval_identity_bound"
    plan_sha = legacy.canonical_sha(plan)
    # Reuse the canonical historical approval contract; extra fields bind the new generation.
    from r58_contracts import authorize_receipt
    old_cap = contracts.RESERVE_CAP
    contracts.RESERVE_CAP = remaining_reserve
    try:
        authorize_receipt(approval, plan_sha256=plan_sha)
    finally:
        contracts.RESERVE_CAP = old_cap
    bc_models, bc_sha = _bc_model_identity()
    expected_output = PRIVATE_RAW.resolve()
    expected_checkpoint_root = resume_sources["target_checkpoint_root"]
    requested_output = Path(output_root).expanduser()
    if not requested_output.is_absolute() or requested_output.resolve() != expected_output:
        raise PermissionError("ROUND58 output root must be the dedicated round58_gen3/raw path")
    if mode == "fresh" and expected_output.exists():
        raise PermissionError("ROUND58 fresh generation output already exists; refusing overwrite")
    if mode == "recover_first_batch_gate" and not expected_output.is_dir():
        raise PermissionError("ROUND58 continuation requires the existing generation root")
    expected_freeze = {
        "kind": "round58-repaired-execution-freeze/1", "status": "frozen",
        "plan_sha256": plan_sha, "training_version": REPAIRED_TRAINING_VERSION,
        "repair_identity": repair, "bc_model_identity_sha256": bc_sha,
        "execution_basis_sha256": _execution_basis_identity(),
        "output_root": str(expected_output), "generation_id": "round58_gen3",
        "checkpoint_root": str(expected_checkpoint_root),
        "mode": mode, "cumulative_quota_sha256": hashlib.sha256(
            Path(quota_path).read_bytes()).hexdigest(), "reserve_cap": remaining_reserve,
        "scene_audit_sha256": hashlib.sha256(Path(scene_audit_path).read_bytes()).hexdigest(),
        **({"resume_approval_source_sha256": resume_approval_source["sha256"]}
           if resume_approval_source is not None else {}),
        **({"plan_identity_migration_proof_sha256": migration_proof_sha256}
           if migration_proof_sha256 is not None else {}),
        "bc_models": bc_models,
    }
    for key, value in expected_freeze.items():
        if frozen.get(key) != value:
            raise PermissionError(f"ROUND58 freeze mismatch: {key}")
    sources = scene_audit.get("checked_sources", {})
    if (scene_audit.get("kind") != "round58-scene-occupancy-audit/1"
            or scene_audit.get("status") != "clear"
            or scene_audit.get("scenes") != list(range(1_700_000, 1_700_032))
            or scene_audit.get("conflicts") != []
            or scene_audit.get("result_blind") is not True
            or any(sources.get(name) is not True
                   for name in ("registrations", "reservations", "claims", "artifacts"))
            or not isinstance(scene_audit.get("audited_at_utc"), str)
            or len(scene_audit["audited_at_utc"]) < 20):
        raise PermissionError("ROUND58 frozen evaluation scene block is not currently cleared")
    if approval.get("plan_sha256") != plan_sha or approval.get("generation_id") != "round58_gen3" \
            or approval.get("repair_identity") != repair or approval.get("output_root") != str(expected_output) \
            or approval.get("checkpoint_root") != str(expected_checkpoint_root) \
            or approval.get("bc_model_identity_sha256") != bc_sha \
            or approval.get("execution_basis_sha256") != _execution_basis_identity() \
            or approval.get("freeze_sha256") != hashlib.sha256(Path(freeze_path).read_bytes()).hexdigest() \
            or approval.get("quota_sha256") != hashlib.sha256(Path(quota_path).read_bytes()).hexdigest() \
            or approval.get("scene_audit_sha256") != hashlib.sha256(Path(scene_audit_path).read_bytes()).hexdigest() \
            or (resume_approval_source is not None
                and (approval.get("resume_approval_source_path") != resume_approval_source["path"]
                     or approval.get("resume_approval_source_sha256") != resume_approval_source["sha256"])) \
            or (migration_proof_sha256 is not None
                and approval.get("plan_identity_migration_proof_sha256") != migration_proof_sha256):
        raise PermissionError("ROUND58 human approval does not bind the repaired generation/model/output")
    return _ValidatedGrant(_GRANT_TOKEN, {
        "authorization_path": Path(authorization_path).resolve(),
        "freeze_path": Path(freeze_path).resolve(), "quota_path": Path(quota_path).resolve(),
        "scene_audit_path": Path(scene_audit_path).resolve(), "output_root": expected_output,
        "checkpoint_root": expected_checkpoint_root,
        "resume_sources": resume_sources,
        "mode": mode, "plan": plan, "plan_sha256": plan_sha,
        "repair_identity": repair, "bc_models": bc_models, "bc_model_identity_sha256": bc_sha,
        "reserve_cap": remaining_reserve, "quota_sha256": hashlib.sha256(
            Path(quota_path).read_bytes()).hexdigest(),
        "historical_manifest_path": history_path.resolve(),
        "approval_receipt": copy.deepcopy(approval),
        "input_file_sha256": {
            "authorization": hashlib.sha256(Path(authorization_path).read_bytes()).hexdigest(),
            "freeze": hashlib.sha256(Path(freeze_path).read_bytes()).hexdigest(),
            "quota": hashlib.sha256(Path(quota_path).read_bytes()).hexdigest(),
            "scene_audit": hashlib.sha256(Path(scene_audit_path).read_bytes()).hexdigest(),
            **({"resume_approval_source": resume_approval_source["sha256"]}
               if resume_approval_source is not None else {}),
            **({"plan_identity_migration_proof": migration_proof_sha256}
               if migration_proof_sha256 is not None else {}),
        },
    })


def _install_repaired_runtime_route(legacy: Any, grant: _ValidatedGrant) -> dict[str, Any]:
    values = grant.values
    contract_module = sys.modules.get("r58_contracts")
    if contract_module is None:
        raise RuntimeError("R58 execution contract module is missing")
    old = {"RAW": legacy.RAW, "TRAINING_VERSION": legacy.TRAINING_VERSION,
           "RESERVE_CAP": legacy.RESERVE_CAP,
           "HERE": legacy.HERE,
           "plan_from_frozen_sources": legacy.plan_from_frozen_sources,
           "install_gate_prior": legacy.install_gate_prior,
           "bind_live_runtime": legacy.bind_live_runtime,
           "bind_runner_paths": legacy.bind_runner_paths,
           "recover_first_batch_gate_then_continue": legacy.recover_first_batch_gate_then_continue,
           "quarantine_failed_first_batch_gate_receipts": legacy.quarantine_failed_first_batch_gate_receipts,
           "verify_persisted_first_batch_state": legacy.verify_persisted_first_batch_state,
           "read_json": legacy.read_json,
           "install_r53_claim_verifier": legacy.install_r53_claim_verifier}
    plan_base_fn = old["plan_from_frozen_sources"]
    bind_base_fn = old["bind_live_runtime"]
    installed_adapters = []
    installed_runtimes = []
    original_import = builtins.__import__
    resume_root_for_configure = (Path(values["checkpoint_root"]).resolve().parent
                                 if values.get("mode") == "recover_first_batch_gate" else None)
    configure_bridge: dict[str, Any] = {"installed": False, "run_module": None,
                                        "original_run_dir": None, "configure_input_run_dir": None}

    def bind_r42_configure_root(r44_module: Any) -> None:
        if resume_root_for_configure is None or getattr(r44_module, "_r58_configure_root_bridge", False):
            return
        original_loader = getattr(r44_module, "_load_r43_runner", None)
        if not callable(original_loader):
            raise RuntimeError("ROUND58 resume cannot locate actual R42 loader before configure")

        def load_r42_with_configure_root():
            run_module = original_loader()
            previous = configure_bridge.get("run_module")
            if previous is run_module and getattr(run_module, "_r58_resume_configure_bound", False):
                return run_module
            if previous is not None and previous is not run_module:
                raise RuntimeError("ROUND58 resume loader returned conflicting R42 module objects")
            if getattr(run_module, "_r58_resume_configure_bound", False):
                raise RuntimeError("ROUND58 resume configure bridge has ambiguous ownership")
            original_configure = run_module.configure

            def provision_frozen_runtime_aliases(target_root: Path) -> dict[str, str]:
                """Create only the four hash-frozen R42 aliases in the private resume root."""
                target_root = Path(target_root).resolve()
                target_root.mkdir(parents=True, exist_ok=True)
                materialized = {}
                for name, expected_sha in RESUME_RUNTIME_ALIAS_HASHES.items():
                    source = PRIVATE_RAW / name
                    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha:
                        raise PermissionError(f"frozen runtime alias source mismatch: {source}")
                    target = target_root / name
                    if target.exists():
                        actual = hashlib.sha256(target.read_bytes()).hexdigest()
                        if actual != expected_sha:
                            raise PermissionError(f"refusing to replace non-matching resume alias: {target}")
                    else:
                        payload = source.read_bytes()
                        with target.open("xb") as handle:
                            handle.write(payload)
                            handle.flush()
                            os.fsync(handle.fileno())
                        if hashlib.sha256(target.read_bytes()).hexdigest() != expected_sha:
                            raise RuntimeError(f"resume runtime alias copy failed hash check: {target}")
                    materialized[name] = hashlib.sha256(target.read_bytes()).hexdigest()
                return materialized

            def configure_with_resume_root(frozen, round40_freeze):
                previous_run_dir = Path(run_module.RUN_DIR).resolve()
                previous_sys_path = list(sys.path)
                configure_bridge["run_module"] = run_module
                configure_bridge["original_run_dir"] = str(previous_run_dir)
                configure_bridge["configure_input_run_dir"] = str(resume_root_for_configure)
                # Keep R44's historical RUN_DIR for its frozen config, but bind
                # the actual R42 configure input, call journals, and module
                # provenance to the fresh sibling generation. Its exact runtime
                # source aliases are separately hash-checked at that root.
                run_module.RUN_DIR = resume_root_for_configure
                configure_bridge["runtime_aliases"] = provision_frozen_runtime_aliases(
                    resume_root_for_configure)
                if str(resume_root_for_configure) in sys.path:
                    sys.path.remove(str(resume_root_for_configure))
                sys.path.insert(0, str(resume_root_for_configure))
                for name, expected_sha in RESUME_RUNTIME_ALIAS_HASHES.items():
                    alias = Path(name).stem
                    loaded = sys.modules.get(alias)
                    if loaded is None:
                        continue
                    loaded_path = Path(getattr(loaded, "__file__", "")).resolve()
                    loaded_sha = (hashlib.sha256(loaded_path.read_bytes()).hexdigest()
                                  if loaded_path.is_file() else None)
                    if loaded_path == (PRIVATE_RAW / name).resolve() and loaded_sha == expected_sha:
                        sys.modules.pop(alias, None)
                    elif (loaded_path != (resume_root_for_configure / name).resolve()
                          or loaded_sha != expected_sha):
                        raise PermissionError(f"refusing cached runtime alias with unapproved provenance: {name}")
                try:
                    return original_configure(frozen, round40_freeze)
                finally:
                    run_module.RUN_DIR = previous_run_dir
                    sys.path[:] = previous_sys_path

            run_module.configure = configure_with_resume_root
            run_module._r58_resume_configure_bound = True
            configure_bridge["run_module"] = run_module
            configure_bridge["original_configure"] = original_configure
            configure_bridge["installed"] = True
            return run_module

        r44_module._load_r43_runner = load_r42_with_configure_root
        r44_module._r58_configure_root_bridge = True
        configure_bridge["r44_module"] = r44_module
        configure_bridge["original_loader"] = original_loader

    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sp_common" and level == 0:
            common_module = sys.modules.get("v2_common")
            if common_module is None:
                raise PermissionError("cannot bind sp_common without the configured frozen v2_common")
            return _verified_sp_common_module(common_module)
        imported = original_import(name, globals, locals, fromlist, level)
        if name == "r44_runtime":
            bind_r42_configure_root(sys.modules.get("r44_runtime", imported))
        return imported

    if resume_root_for_configure is not None:
        existing_r44 = sys.modules.get("r44_runtime")
        if existing_r44 is not None:
            bind_r42_configure_root(existing_r44)
        builtins.__import__ = private_import
    def bound_plan():
        plan = plan_base_fn()
        plan["training_version"] = REPAIRED_TRAINING_VERSION
        plan["r58_repair_identity"] = values["repair_identity"]
        plan["generation_id"] = "round58_gen3"
        plan["status"] = "frozen_approved_gen3"
        plan["authorization"] = "direct_user_owner_approval_identity_bound"
        return plan
    def install_repaired(trainer, detach_context, output_root):
        receipt = repaired_adapter.install_gate_prior(
            trainer, detach_context, Path(output_root))
        installed_adapters.append(receipt)
        return receipt
    installed_contracts = []
    bind_paths_base = old["bind_runner_paths"]
    def bind_gen3_paths(base, raw_root):
        # The canonical guard requires raw_root to be below its declared root.
        # Temporarily scope that root to the one frozen gen3 generation, then
        # restore the preserved gen2 module identity immediately.
        legacy.HERE = PRIVATE_GENERATION
        try:
            result = bind_paths_base(base, raw_root)
            if values.get("mode") == "recover_first_batch_gate":
                checkpoint_root = Path(values["checkpoint_root"]).resolve()
                resume_root = checkpoint_root.parent
                if (checkpoint_root != values["resume_sources"]["target_checkpoint_root"]
                        or resume_root != checkpoint_root.parents[0]
                        or resume_root == values["output_root"]):
                    raise PermissionError("resume output route differs from the approved source map")
                base.CHECKPOINT_JOURNAL_PATH = resume_root / "runtime" / "checkpoint_writes.jsonl"
                base.FAMILY_JOURNAL_PATH = base.CHECKPOINT_JOURNAL_PATH
                base.FAMILY_LEDGER_PATH = resume_root / "runtime" / "ROUND53_EXECUTION_LEDGER.json"
                base.LEDGER_PATH = base.FAMILY_LEDGER_PATH
                result.update({"resume_checkpoint_root": str(checkpoint_root),
                               "resume_runtime_root": str(resume_root / "runtime"),
                               "resume_claim_root_preserved": str(Path(raw_root).resolve())})
            return result
        finally:
            legacy.HERE = old["HERE"]
    def bind_repaired(base, runtime, plan, *, checkpoint_mode="forbid_existing"):
        plan = copy.deepcopy(plan)
        plan["training_version"] = REPAIRED_TRAINING_VERSION
        plan["r58_repair_identity"] = values["repair_identity"]
        # Recovery keeps historical gen3 claims/rows under raw, while the new
        # checkpoint-write journal is intentionally rooted in the approved
        # sibling resume generation. Permit only that one journal path through
        # the legacy raw-root containment assertion; every other path still
        # goes through the original guard unchanged.
        if values.get("mode") == "recover_first_batch_gate":
            raw_root = Path(values["output_root"]).resolve()
            resume_journal = (Path(values["checkpoint_root"]).resolve().parent
                              / "runtime" / "checkpoint_writes.jsonl").resolve()
            original_path_guard = legacy.require_private_output_root
            journal_guard_calls = {"count": 0}

            def require_approved_resume_journal(root, paths):
                root_path = Path(root).resolve()
                filtered = []
                for path in paths:
                    candidate = Path(path).resolve()
                    if candidate == resume_journal:
                        if root_path != raw_root:
                            raise PermissionError("resume journal exception used with an unapproved raw root")
                        journal_guard_calls["count"] += 1
                        if journal_guard_calls["count"] != 1:
                            raise PermissionError("resume journal path was admitted more than once")
                        continue
                    filtered.append(path)
                return original_path_guard(root, filtered)

            legacy.require_private_output_root = require_approved_resume_journal
            try:
                live = bind_base_fn(base, runtime, plan, checkpoint_mode=checkpoint_mode)
            finally:
                legacy.require_private_output_root = original_path_guard
            if journal_guard_calls["count"] != 1:
                raise PermissionError("actual binder did not validate the unique approved resume journal path")
        else:
            live = bind_base_fn(base, runtime, plan, checkpoint_mode=checkpoint_mode)
        if len(installed_adapters) != 1:
            raise RuntimeError("repaired prior adapter was not installed exactly once")
        runtime["r56_gate_prior"] = installed_adapters[0]
        runtime["r58_repaired_adapter"] = installed_adapters[0]
        resume_runtime = None
        if values.get("mode") == "recover_first_batch_gate":
            checkpoint_root = Path(values["checkpoint_root"]).resolve()
            resume_root = checkpoint_root.parent
            resume_runtime = resume_root / "runtime"
            if checkpoint_root != values["resume_sources"]["target_checkpoint_root"]:
                raise PermissionError("resume checkpoint root changed after receipt validation")
            # The old gen3 raw root remains authoritative for claims/slots and frozen inputs.
            # All future rows, logs, progress, checkpoints, and call journals route to this
            # explicit sibling resume generation; no historical checkpoint or journal is copied.
            for path in (resume_root, resume_runtime, checkpoint_root,
                         resume_root / "rows", resume_root / "logs",
                         resume_root / "rejected", resume_root / "training_progress"):
                if path.exists():
                    if path == resume_root and not path.is_dir():
                        raise RuntimeError("resume output root exists as a non-directory")
                else:
                    path.mkdir(parents=True, exist_ok=True)
            common = runtime["common"]
            common.OUTPUT_DIR = resume_root
            common.LOGS = resume_root / "logs"
            common.ROWS = resume_root / "rows"
            common.REJECTED = resume_root / "rejected"
            common.CHECKPOINTS = checkpoint_root
            runtime["r40_common"].ROUND_DIR = values["output_root"]
            runtime["r42_run"].RUN_DIR = values["output_root"]
            base.CHECKPOINT_JOURNAL_PATH = resume_runtime / "checkpoint_writes.jsonl"
            base.FAMILY_JOURNAL_PATH = base.CHECKPOINT_JOURNAL_PATH
            base.FAMILY_LEDGER_PATH = resume_runtime / "ROUND53_EXECUTION_LEDGER.json"
            base.LEDGER_PATH = base.FAMILY_LEDGER_PATH
        if checkpoint_mode == "forbid_existing" and Path(runtime["common"].CHECKPOINTS).resolve() != values["output_root"] / "checkpoints":
            if values.get("mode") != "recover_first_batch_gate":
                raise RuntimeError("live Round58 checkpoint root differs from the approved generation root")
        contract = checkpoint_contract.install(
            runtime["trainer"], runtime["r40_train"],
            identity=values["repair_identity"], training_version=REPAIRED_TRAINING_VERSION,
            expected_checkpoint_root=runtime["common"].CHECKPOINTS,
            versions=runtime["r42_versions"], r42_run=runtime["r42_run"],
            approved_sources=(None if resume_runtime is None else values["resume_sources"]["sources"]),
            resume_map_sha256=(None if resume_runtime is None
                               else values["resume_sources"]["manifest_sha256"]))
        runtime["r58_checkpoint_contract"] = contract
        if resume_runtime is None:
            counter_bridge = identity_ledger.bridge_ppo_update_counter(
                runtime["trainer"], runtime["r42_run"])
            runtime["r58_ppo_counter_bridge"] = counter_bridge
        else:
            configure_bridge = resume_seam.install_r42_configure_compat(
                runtime["r42_run"], REPO_ROOT,
                extra_batch_root=runtime["common"].CHECKPOINTS,
                proof=values["resume_sources"], resume_runtime_dir=resume_runtime)
            runtime["r58_resume_configure_bridge"] = configure_bridge
            runtime["r58_resume_source_proof"] = values["resume_sources"]
            call_journals = resume_seam.install_persistent_call_journals(
                runtime, resume_runtime_dir=resume_runtime)
            runtime["r58_ppo_counter_bridge"] = call_journals
            runtime["r58_call_journals"] = call_journals
            runtime["r58_resume_source_map_sha256"] = values["resume_sources"]["manifest_sha256"]
        runtime["r58_repair_identity"] = values["repair_identity"]
        if runtime["r40_train"].trainer is not runtime["trainer"]:
            contract["undo"]()
            raise RuntimeError("repaired checkpoint route is not the trainer used by R40")
        installed_contracts.append(contract)
        installed_runtimes.append((runtime, live))
        return live

    if values.get("mode") == "recover_first_batch_gate":
        raw_root = Path(values["output_root"]).resolve()
        resume_journal = (Path(values["checkpoint_root"]).resolve().parent
                          / "runtime" / "checkpoint_writes.jsonl").resolve()
        def install_resume_claim_verifier(base, runtime):
            # Preserve the frozen R53 source/branch/worktree assertions while binding
            # claims to the exact sibling resume journal, not the historical gen3 root.
            r45_module = getattr(base, "_R45_MODULE", None)
            r44_module = getattr(r45_module, "R44_RUNTIME_MODULE", None) if r45_module else None
            if r44_module is None:
                raise RuntimeError("R58 resume claim verifier cannot locate live R44 module")
            stale = getattr(r44_module, "verify_frozen_source", None)
            stale_code = getattr(stale, "__code__", None)
            if (stale_code is None
                    or getattr(stale, "__name__", "") != "round46_r2_v5_claim_source_verifier"
                    or Path(stale_code.co_filename).resolve() != legacy.SOURCE_RUNNER.resolve()):
                raise RuntimeError("R58 resume expected the original R46 strict verifier")
            registered_frozen = legacy.read_json(legacy.R44_DIR / "ROUND44_FROZEN_CONFIG.json")
            raw = Path(base.V6_DIR).resolve()
            if (Path(runtime["r42_run"].RUN_DIR).resolve() != raw
                    or raw != raw_root):
                raise RuntimeError("R58 resume claim namespace moved from historical gen3 raw")
            journal = Path(base.CHECKPOINT_JOURNAL_PATH).resolve()
            if journal != resume_journal:
                raise RuntimeError("R58 resume claim journal differs from the approved sibling root")
            def resume_claim_verifier(frozen):
                if frozen != registered_frozen:
                    raise RuntimeError("R44 claim supplied a frozen object outside registered source")
                base._assert_branch_head()
                base._assert_source_worktree_clean()
                if Path(runtime["r42_run"].RUN_DIR).resolve() != raw:
                    raise RuntimeError("R58 resume claim root moved before claim")
                if Path(base.CHECKPOINT_JOURNAL_PATH).resolve() != journal:
                    raise RuntimeError("R58 resume claim journal route changed")
                audit = base._checkpoint_journal_route_audit(create=False)
                if (audit.get("status") != "fresh_private_route"
                        or audit.get("historical_family_journal_copied") is not False
                        or Path(audit.get("journal_path", "")).resolve() != journal):
                    raise RuntimeError("R58 resume private journal route audit did not close")
                return base._run_git("rev-parse", "HEAD")
            resume_claim_verifier.__name__ = "round58_resume_private_claim_source_verifier"
            resume_claim_verifier.__qualname__ = resume_claim_verifier.__name__
            r44_module.verify_frozen_source = resume_claim_verifier
            if r44_module.verify_frozen_source is not resume_claim_verifier:
                raise RuntimeError("R58 resume claim verifier binding failed")
            return {"kind": "round58-resume-claim-verifier/1", "status": "installed",
                    "historical_claim_root": str(raw), "resume_checkpoint_journal": str(journal),
                    "source_verifier": str(Path(stale_code.co_filename).resolve())}

        legacy.install_r53_claim_verifier = install_resume_claim_verifier
        source_map = values["resume_sources"]
        qualification_path = REPO_ROOT / resume_seam.MAP_REL
        audit_ref = source_map["manifest"]["qualification_audit"]
        verdict_ref = source_map["manifest"]["qualification_verdict"]
        qualification_audit = json.loads((REPO_ROOT / audit_ref["path"]).read_text(encoding="utf-8"))
        qualification_verdict = json.loads((REPO_ROOT / verdict_ref["path"]).read_text(encoding="utf-8"))
        original_read_json = old["read_json"]
        gate_path_by_seed = {seed: raw_root / f"ROUND45_R2_FIRST_BATCH_GATE_R3_s{seed}.json"
                             for seed in (11, 22, 33)}

        def resume_read_json(path):
            resolved = Path(path).resolve()
            for seed, source_path in gate_path_by_seed.items():
                if resolved == source_path.resolve():
                    row = next((item for item in qualification_audit.get("seeds", [])
                                if int(item.get("seed", -1)) == seed), None)
                    if row is None or row.get("status") != "passed":
                        raise PermissionError(f"seed {seed}: prior actual read-only gate qualification missing")
                    first_four_path = raw_root / f"ROUND46_FIRST_FOUR_s{seed}.json"
                    first_four = original_read_json(first_four_path)
                    bc_hash = (first_four.get("report", {})
                               .get("round45_bc_initialization", {})
                               .get("bc_parameter_sha256"))
                    approved_bc = values["bc_models"][str(seed)]["parameter_sha256"]
                    if bc_hash != approved_bc:
                        raise PermissionError(f"seed {seed}: approved gate view has a BC identity mismatch")
                    qualified = next((item for item in qualification_verdict.get("per_seed", [])
                                      if int(item.get("seed", -1)) == seed), None)
                    if qualified is None or qualified.get("qualification") != resume_seam.QUALIFIED:
                        raise PermissionError(f"seed {seed}: prior qualification verdict is not accepted")
                    return {"kind": "round58-in-memory-qualified-gate-view/1",
                            "status": "passed", "seed": seed,
                            "bc_parameter_sha256": bc_hash,
                            "qualification_source_sha256": verdict_ref["sha256"],
                            "original_failed_gate_preserved": source_path.is_file()}
            return original_read_json(path)

        verify_base = legacy.verify_persisted_first_batch_state
        def verify_preserved_first_batch(base, raw, plan):
            # The existing verifier must see the original failed receipt, never the derived view.
            active = legacy.read_json
            legacy.read_json = original_read_json
            try:
                result = verify_base(base, raw, plan)
            finally:
                legacy.read_json = active
            if result.get("status") != "passed":
                raise RuntimeError("preserved first-batch core state verification did not pass")
            return result

        def preserve_failed_gate_receipts(raw):
            rows = []
            for seed, path in gate_path_by_seed.items():
                if not path.is_file():
                    continue
                item = original_read_json(path)
                if item.get("status") != "failed":
                    raise RuntimeError(f"seed {seed}: existing failed gate receipt changed status")
                rows.append({"seed": seed, "path": str(path), "sha256": _sha256_file(path),
                             "status": "failed_preserved_no_quarantine"})
            if not rows:
                raise RuntimeError("resume requires the original failed gate receipt to remain present")
            return {"kind": "round58-preserved-original-gate-receipts/1",
                    "status": "preserved", "receipts": rows}

        def resume_from_qualified_commits(base, module, runtime):
            proof = resume_seam.validate_resume_sources(REPO_ROOT)
            first_four = {}
            for seed in (11, 22, 33):
                path = raw_root / f"ROUND46_FIRST_FOUR_s{seed}.json"
                row = original_read_json(path)
                if (row.get("kind") != "round46-first-four-report/1"
                        or int(row.get("seed", -1)) != seed
                        or row.get("training_version") != REPAIRED_TRAINING_VERSION
                        or int(row.get("report", {}).get("episodes_done", -1)) != 4):
                    raise RuntimeError(f"seed {seed}: historical first-four receipt identity/cursor mismatch")
                first_four[seed] = row["report"]
            # Resume is now allowed only from the exact mapped history, never from gen2 or a glob.
            base.R46_DIR = Path(values["checkpoint_root"]).parent
            base.V6_CHECKPOINTS_DIR = Path(values["checkpoint_root"])
            if Path(runtime["common"].CHECKPOINTS).resolve() != base.V6_CHECKPOINTS_DIR.resolve():
                raise RuntimeError("live R40 checkpoint selector is not bound to the private resume root")
            selected_sources = {}
            for seed in (11, 22, 33):
                selected = Path(runtime["r40_train"]._latest_resume_source(
                    seed, runtime["frozen"])).resolve()
                expected = Path(proof["sources"][seed]["path"]).resolve()
                if selected != expected:
                    raise PermissionError(f"seed {seed}: live resume selector did not choose its exact qualified source")
                selected_sources[seed] = str(selected)
            restore_gate_hook = globals().get("_ROUND58_REAL_RESTORE_GATE_HOOK")
            if restore_gate_hook is not None:
                if not callable(restore_gate_hook):
                    raise TypeError("ROUND58 real-restore gate hook must be callable")
                restore_gate_hook({"kind": "round58-real-checkpoint-restore-gate/1",
                                   "runtime": runtime, "proof": proof,
                                   # train_seed receives round42_freeze, not the umbrella R43/R44 freeze.
                                   "freeze": runtime["round42_freeze"],
                                   "selected_sources": selected_sources})
                raise RuntimeError("ROUND58 real-restore gate hook returned without controlled stop")
            preflight_hook = globals().get("_ROUND58_PREFLIGHT_STOP_HOOK")
            if preflight_hook is not None:
                if not callable(preflight_hook):
                    raise TypeError("ROUND58 preflight stop hook must be callable")
                preflight_hook({"kind": "round58-full-cli-resume-preflight-handoff/1",
                                "status": "all_gates_passed_before_fixed_continuation",
                                "migration_proof_sha256": values["input_file_sha256"]["plan_identity_migration_proof"],
                                "source_map_sha256": proof["manifest_sha256"],
                                "selected_sources": selected_sources,
                                "configure_bridge": {
                                    "installed_before_actual_r42_configure": configure_bridge["installed"],
                                    "historical_run_dir": configure_bridge["original_run_dir"],
                                    "configure_input_run_dir": configure_bridge["configure_input_run_dir"],
                                },
                                "runtime_roots": {
                                    "historical_claim_root": str(Path(runtime["r40_common"].ROUND_DIR).resolve()),
                                    "historical_batch_input_root": str(Path(runtime["r42_run"].RUN_DIR).resolve()),
                                    "resume_output_root": str(Path(runtime["common"].OUTPUT_DIR).resolve()),
                                    "resume_checkpoint_root": str(Path(runtime["common"].CHECKPOINTS).resolve()),
                                },
                                "episode4_scenarios": {seed: 1300003 for seed in (11, 22, 33)},
                                "environment_constructed": 0, "claims_created": 0,
                                "optimizer_steps": 0, "ppo_updates": 0, "checkpoint_writes": 0})
                raise RuntimeError("ROUND58 preflight stop hook returned without stopping execution")
            result = base._continue_fixed_training(module, runtime, first_four)
            return {"kind": "round58-qualified-episode4-continuation/1",
                    "status": "complete", "source_map_sha256": proof["manifest_sha256"],
                    "first_four_reused": True, "gates_reused_from_actual_readonly_audit": True}, result

        legacy.read_json = resume_read_json
        legacy.verify_persisted_first_batch_state = verify_preserved_first_batch
        legacy.quarantine_failed_first_batch_gate_receipts = preserve_failed_gate_receipts
        legacy.recover_first_batch_gate_then_continue = resume_from_qualified_commits
    legacy.RAW = values["output_root"]
    legacy.TRAINING_VERSION = REPAIRED_TRAINING_VERSION
    legacy.RESERVE_CAP = values["reserve_cap"]
    old_contract_reserve = getattr(contract_module, "RESERVE_CAP", None)
    contract_module.RESERVE_CAP = values["reserve_cap"]
    legacy.plan_from_frozen_sources = bound_plan
    legacy.install_gate_prior = install_repaired
    legacy.bind_live_runtime = bind_repaired
    legacy.bind_runner_paths = bind_gen3_paths
    def undo():
        if resume_root_for_configure is not None:
            if builtins.__import__ is not private_import:
                raise RuntimeError("ROUND58 resume import bridge changed before teardown")
            builtins.__import__ = original_import
            r44_module = configure_bridge.get("r44_module")
            if r44_module is not None:
                r44_module._load_r43_runner = configure_bridge["original_loader"]
                delattr(r44_module, "_r58_configure_root_bridge")
            run_module = configure_bridge.get("run_module")
            if run_module is not None and getattr(run_module, "_r58_resume_configure_bound", False):
                run_module.configure = configure_bridge["original_configure"]
                delattr(run_module, "_r58_resume_configure_bound")
        for runtime, live in reversed(installed_runtimes):
            call_journals = runtime.get("r58_call_journals")
            if call_journals and callable(call_journals.get("undo")):
                call_journals["undo"]()
            counter = runtime.get("r58_ppo_counter_bridge")
            if counter and callable(counter.get("undo")):
                counter["undo"]()
            adapter = runtime.get("r58_repaired_adapter")
            if adapter and callable(adapter.get("undo")):
                adapter["undo"]()
            undo_live = live.get("route_undo")
            if callable(undo_live):
                undo_live()
        for contract in reversed(installed_contracts):
            contract["undo"]()
        for key, value in old.items():
            setattr(legacy, key, value)
        contract_module.RESERVE_CAP = old_contract_reserve
    return {"undo": undo, "installed_contracts": installed_contracts,
            "resume_configure_bridge": configure_bridge}


def _dispatch_validated(grant: _ValidatedGrant, *, legacy_loader=None):
    if not isinstance(grant, _ValidatedGrant) or not grant.values:
        raise PermissionError("ROUND58 execution requires a validated external receipt set")
    values = grant.values
    if not values.get("test_only"):
        paths = {"authorization": values["authorization_path"],
                 "freeze": values["freeze_path"], "quota": values["quota_path"],
                 "scene_audit": values["scene_audit_path"]}
        if values["mode"] == "recover_first_batch_gate":
            paths["plan_identity_migration_proof"] = HERE / "ROUND58_GEN3_PLAN_IDENTITY_MIGRATION_PROOF_V18.json"
        for key, path in paths.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != values["input_file_sha256"][key]:
                raise PermissionError(f"ROUND58 {key} receipt changed after preflight")
        if _repair_identity() != values["repair_identity"]:
            raise PermissionError("ROUND58 repaired source identity changed after preflight")
        current_models, current_model_sha = _bc_model_identity()
        if current_models != values["bc_models"] or current_model_sha != values["bc_model_identity_sha256"]:
            raise PermissionError("ROUND58 BC model identity changed after preflight")
        if values["mode"] == "fresh" and Path(values["output_root"]).exists():
            raise PermissionError("ROUND58 output root appeared after preflight")
    snapshot = _snapshot_runtime_bindings()
    legacy = (load_legacy_launcher if legacy_loader is None else legacy_loader)()
    route = None
    migration_route = None
    try:
        route = _install_repaired_runtime_route(legacy, grant)
        if legacy.canonical_sha(legacy.plan_from_frozen_sources()) != values["plan_sha256"]:
            raise PermissionError("ROUND58 frozen plan identity changed before execution")
        if values.get("approval_receipt") is not None:
            original_read_json = legacy.read_json
            auth_path = Path(values["authorization_path"]).resolve()
            approval_receipt = copy.deepcopy(values["approval_receipt"])
            def frozen_read_json(path):
                if Path(path).resolve() == auth_path:
                    if not values.get("test_only") and hashlib.sha256(
                            auth_path.read_bytes()).hexdigest() != values["input_file_sha256"]["authorization"]:
                        raise PermissionError("ROUND58 human approval changed before legacy preflight")
                    return copy.deepcopy(approval_receipt)
                return original_read_json(path)
            legacy.read_json = frozen_read_json
        if values["mode"] == "recover_first_batch_gate":
            current_live_plan = legacy.plan_from_frozen_sources()
            migration_route = resume_seam.install_historical_plan_comparison_view(
                legacy, repo_root=REPO_ROOT, current_plan=current_live_plan,
                current_repair_identity=values["repair_identity"],
                proof_path=HERE / "ROUND58_GEN3_PLAN_IDENTITY_MIGRATION_PROOF_V18.json",
                expected_proof_sha256=values["input_file_sha256"]["plan_identity_migration_proof"])
            legacy.round58_plan_identity_migration = {
                key: value for key, value in migration_route.items() if key != "undo"}
        # This is the preserved ROUND58 fixed-plan orchestration, not a parallel trainer.
        return legacy.execute_authorized(
            auth_path=grant.values["authorization_path"], mode=grant.values["mode"])
    finally:
        if migration_route is not None:
            migration_route["undo"]()
        if route is not None:
            route["undo"]()
        _restore_runtime_bindings(snapshot)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute only an externally frozen ROUND58 generation")
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--cumulative-quota", type=Path, required=True)
    parser.add_argument("--scene-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("fresh", "recover_first_batch_gate"), default="fresh")
    args = parser.parse_args(argv)
    grant = _validate_external_grant(
        authorization_path=args.authorization, freeze_path=args.freeze,
        quota_path=args.cumulative_quota, scene_audit_path=args.scene_audit,
        output_root=args.output_root, mode=args.mode)
    result = _dispatch_validated(grant)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def load_legacy_launcher():
    """Load the preserved launcher module without invoking its main entry."""
    path = GEN2 / "r58_execute.py"
    spec = importlib.util.spec_from_file_location("round58_legacy_launcher_readonly", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load preserved ROUND58 launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _snapshot_module_value(value):
    if isinstance(value, (dict, list, set, tuple)):
        try:
            return copy.deepcopy(value)
        except Exception:
            return value
    return value


def _snapshot_runtime_bindings() -> dict[str, Any]:
    modules = dict(sys.modules)
    tracked_roots = tuple(path.resolve() for path in (GEN2, R53, R45, CANON))
    states = {}
    for name, module in modules.items():
        if not isinstance(module, types.ModuleType):
            continue
        source = getattr(module, "__file__", None)
        tracked = name.startswith(("v2_", "r40_", "r42_", "r43_", "r44_", "r45_", "r58_"))
        if source:
            try:
                resolved = Path(source).resolve()
                tracked = tracked or any(resolved == root or root in resolved.parents
                                         for root in tracked_roots)
            except (OSError, RuntimeError):
                pass
        if tracked:
            states[name] = {key: _snapshot_module_value(value)
                            for key, value in vars(module).items()}
    return {"modules": modules, "module_states": states, "sys_path": list(sys.path),
            "tracked_roots": tracked_roots}


def _restore_runtime_bindings(snapshot: dict[str, Any]) -> None:
    before = snapshot["modules"]
    roots = snapshot["tracked_roots"]
    prefixes = ("v2_", "r40_", "r42_", "r43_", "r44_", "r45_", "r58_")
    for name, module in list(sys.modules.items()):
        if name in before:
            continue
        source = getattr(module, "__file__", None)
        under_root = False
        if source:
            try:
                resolved = Path(source).resolve()
                under_root = any(resolved == root or root in resolved.parents for root in roots)
            except (OSError, RuntimeError):
                pass
        if name.startswith(prefixes) or under_root:
            sys.modules.pop(name, None)
    for name, state in snapshot["module_states"].items():
        module = before.get(name)
        if isinstance(module, types.ModuleType):
            module.__dict__.clear()
            module.__dict__.update(state)
            sys.modules[name] = module
    sys.path[:] = snapshot["sys_path"]


def prepare_actual_runtime_for_audit(raw_root: Path, *, writer=None) -> dict[str, Any]:
    """Configure live modules in-process, with reversible module/global bindings."""
    snapshot = _snapshot_runtime_bindings()
    try:
        binding = _prepare_actual_runtime_for_audit_impl(raw_root, writer=writer)
        binding["mutation_snapshot"] = snapshot
        return binding
    except BaseException:
        _restore_runtime_bindings(snapshot)
        raise


def _prepare_actual_runtime_for_audit_impl(raw_root: Path, *, writer=None) -> dict[str, Any]:
    """Configure actual R46/R45/R40 objects and bind the repaired R58 adapter.

    This configures module objects and private paths only. It does not call the
    episode provider, optimizer update, checkpoint restore/save, or task runner.
    """
    import torch
    legacy = load_legacy_launcher()
    raw_root = Path(raw_root).resolve()
    if raw_root == GEN2.resolve() or GEN2.resolve() not in raw_root.parents:
        raise RuntimeError("audit runtime output must be a fresh child of, not the root of, gen2")
    # bind_runner_paths enforces containment under the launcher's immutable HERE.
    legacy.RAW = raw_root
    base = legacy.load_base_runner()
    legacy.bind_runner_paths(base, raw_root)
    legacy.install_private_configure_seams(base, allow_existing_journal=False)
    legacy.install_protected_dirty_file_guard(base)
    base._R45_MODULE = None
    base._RUNTIME = None
    # The legacy runtime configure mutates GAE_LAMBDA to the frozen full-trace
    # value. A prior in-process audit bind must not be mistaken for source input.
    import r45_bc  # noqa: PLC0415
    r45_bc.configure_source_imports()
    import v2_common  # noqa: PLC0415
    import r42_versions  # noqa: PLC0415
    source_frozen = legacy.read_json(legacy.R44_DIR / "ROUND44_FROZEN_CONFIG.json")
    v2_common.GAE_LAMBDA = float(source_frozen["round43"]["gae_lambda_old"])
    r42_versions.TRAINING_VERSION = source_frozen["round42"]["training_version"]
    for alias in legacy.RUNTIME_CODE_ALIASES:
        sys.modules.pop(Path(alias).stem, None)
    if str(raw_root) in sys.path:
        sys.path.remove(str(raw_root))
    sys.path.insert(0, str(raw_root))
    original_installer = legacy.install_gate_prior
    installed_receipts = []
    def install_repaired(trainer, detach_context, output_root):
        kwargs = {} if writer is None else {"writer": writer}
        receipt = repaired_adapter.install_gate_prior(
            trainer, detach_context, Path(output_root), **kwargs)
        installed_receipts.append(receipt)
        return receipt
    legacy.install_gate_prior = install_repaired
    original_write_bytes = Path.write_bytes
    original_open = Path.open
    original_touch = Path.touch
    journal_target = (raw_root / "runtime" / "checkpoint_writes.jsonl").resolve()
    journal_route_spy = []
    def intercept_empty_journal(path, data):
        if Path(path).resolve() == journal_target:
            journal_route_spy.append({"path": str(journal_target), "bytes": len(data),
                                      "action": "in-memory-only"})
            return len(data)
        return original_write_bytes(path, data)
    def intercept_journal_open(path, mode="r", *args, **kwargs):
        if Path(path).resolve() == journal_target and any(flag in mode for flag in ("w", "a", "x", "+")):
            journal_route_spy.append({"path": str(journal_target), "mode": mode,
                                      "action": "in-memory-only"})
            return io.BytesIO() if "b" in mode else io.StringIO()
        return original_open(path, mode, *args, **kwargs)
    def intercept_journal_touch(path, *args, **kwargs):
        if Path(path).resolve() == journal_target:
            journal_route_spy.append({"path": str(journal_target), "action": "in-memory-only-touch"})
            return path
        return original_touch(path, *args, **kwargs)
    Path.write_bytes = intercept_empty_journal
    Path.open = intercept_journal_open
    Path.touch = intercept_journal_touch
    try:
        base._configure_runtime_module()
        runtime = base._RUNTIME
        if not isinstance(runtime, dict):
            raise RuntimeError("actual Round46 configure did not produce runtime object map")
        plan = legacy.plan_from_frozen_sources()
        live = legacy.bind_live_runtime(base, runtime, plan, checkpoint_mode="forbid_existing")
        # The audit-only configure path does not pass through the CLI grant
        # dispatcher; bind the same real outer counter bridge before returning
        # its live trainer so the call chain under audit is identical.
        if not runtime.get("r58_ppo_counter_bridge"):
            runtime["r58_ppo_counter_bridge"] = identity_ledger.bridge_ppo_update_counter(
                runtime["trainer"], runtime["r42_run"])
    finally:
        Path.write_bytes = original_write_bytes
        Path.open = original_open
        Path.touch = original_touch
        legacy.install_gate_prior = original_installer
    if len(installed_receipts) != 1:
        raise RuntimeError(f"expected one repaired prior installation, found {len(installed_receipts)}")
    runtime["r58_repaired_adapter"] = installed_receipts[0]

    trainer = runtime["trainer"]
    r40 = runtime["r40_train"]
    if r40.trainer is not trainer:
        raise RuntimeError("actual R40 training module is not bound to the configured trainer object")
    closure_values = []
    wrapper_chain = []
    cursor = trainer.ppo_update
    seen = set()
    while callable(cursor) and id(cursor) not in seen:
        seen.add(id(cursor))
        wrapper_chain.append(cursor)
        closure_values.extend(cell.cell_contents for cell in
                              (getattr(cursor, "__closure__", None) or ()))
        cursor = getattr(cursor, "__wrapped__", None)
    compiled_updates = [value for value in closure_values
                        if callable(value) and getattr(value, "__code__", None)
                        and value.__code__.co_filename.startswith("<r58-gate-ppo-update-repaired")]
    if len(compiled_updates) != 1 or compiled_updates[0].__globals__.get("common") is not runtime["common"]:
        raise RuntimeError("actual trainer ppo_update globals do not point at configured common")
    if runtime.get("r58_repaired_adapter", {}).get("kind") != "r58-gate-prior-repaired/1":
        raise RuntimeError("actual R58 runtime did not install the repaired prior adapter")
    checkpoint = checkpoint_contract.install(
        trainer, r40, identity=_repair_identity(),
        training_version=REPAIRED_TRAINING_VERSION,
        expected_checkpoint_root=runtime["common"].CHECKPOINTS,
        versions=runtime["r42_versions"], r42_run=runtime["r42_run"])
    runtime["r58_checkpoint_contract"] = checkpoint
    return {"legacy": legacy, "base": base, "runtime": runtime,
            "live": live, "torch": torch, "raw_root": raw_root,
            "trainer_object_id": id(trainer), "r40_trainer_object_id": id(r40.trainer),
            "compiled_ppo_object_id": id(compiled_updates[0]),
            "checkpoint_journal_io_spy": journal_route_spy,
            "checkpoint_contract": checkpoint,
            "trainer_source": str(Path(trainer.ppo_update.__wrapped__.__code__.co_filename).resolve())
            if hasattr(trainer.ppo_update, "__wrapped__") else str(Path(trainer.ppo_update.__code__.co_filename).resolve()),
            "ppo_wrapper_chain": [getattr(value, "__name__", type(value).__name__)
                                  for value in wrapper_chain],
            "compiled_ppo_identity": id(compiled_updates[0])}


def resume_from_payload_before_any_restore(payload: dict[str, Any],
                                           expected_repair_identity: dict[str, Any],
                                           *, write_state, restore_model, restore_optimizer,
                                           restore_rng, construct_environment,
                                           optimizer_step):
    """The real repaired resume branch; legacy attestation is checked first."""
    resume_guard.require_repaired_resume(payload, expected_repair_identity)
    write_state(payload)
    model = restore_model(payload)
    optimizer = restore_optimizer(payload)
    restore_rng(payload)
    env = construct_environment()
    optimizer_step()
    return model, optimizer, env


def close_audit_runtime(binding: dict[str, Any]) -> None:
    snapshot = binding.get("mutation_snapshot")
    try:
        runtime = binding["runtime"]
        counter = runtime.get("r58_ppo_counter_bridge")
        if counter and callable(counter.get("undo")):
            counter["undo"]()
        checkpoint = runtime.get("r58_checkpoint_contract")
        if checkpoint and callable(checkpoint.get("undo")):
            checkpoint["undo"]()
        adapter = runtime.get("r58_repaired_adapter")
        if adapter and callable(adapter.get("undo")):
            adapter["undo"]()
        undo = binding.get("live", {}).get("route_undo")
        if callable(undo):
            undo()
    finally:
        if snapshot:
            _restore_runtime_bindings(snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
