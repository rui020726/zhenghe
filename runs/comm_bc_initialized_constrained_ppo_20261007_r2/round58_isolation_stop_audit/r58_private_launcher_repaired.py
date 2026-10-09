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
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parents[0]
GEN2 = HERE.parent / "round58_gen2"
R53 = HERE.parent / "round53_critic_isolation_trial_readiness"
R45 = HERE.parents[2] / "runs" / "comm_rule_warmstart_cost_learning_20261006_r2"
CANON = HERE.parents[2].parent / "zhenghe" / "runs" / "comm_cost_v2_rerun_20260928"

for _path in (GEN2, R53, R45, CANON):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import r58_gate_prior_repaired as repaired_adapter  # noqa: E402
import r58_repaired_entry as resume_guard  # noqa: E402
import r58_checkpoint_contract_repaired as checkpoint_contract  # noqa: E402
import r58_identity_ledger_close_v2 as identity_ledger  # noqa: E402

PRIVATE_GENERATION = RUN_ROOT / "round58_gen3"
PRIVATE_RAW = PRIVATE_GENERATION / "raw"
REPAIRED_TRAINING_VERSION = (
    "constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/"
    "dual-eta-lambda-0p5/1/rule-bc-late147-init/1/"
    "published-completion-event-at-relay-tick/2+r53-critic-representation-stop-gradient/1+"
    "r56-soft-early-report-prior-beta0p10/1+r58-prior-strength-beta1p00/1+"
    "r58-repaired-private-generation3/1+r58-cumulative-quota-history16-normal1024-reserve24-hard1064/2"
)
_GRANT_TOKEN = object()


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
            "ppo_counter_bridge_version": "round58-ppo-counter-bridge/2",
            "identity_ledger_schema": identity_ledger.IDENTITY_VERSION,
            "identity_ledger_sha256": hashlib.sha256(
                Path(identity_ledger.__file__).read_bytes()).hexdigest(),
            "private_launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checkpoint_contract_sha256": hashlib.sha256(
                Path(checkpoint_contract.__file__).read_bytes()).hexdigest(),
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
    approval = _read_external_json(authorization_path, "human approval")
    frozen = _read_external_json(freeze_path, "freeze identity")
    quota = _read_external_json(quota_path, "cumulative quota receipt")
    scene_audit = _read_external_json(scene_audit_path, "scene occupancy audit")
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
        "mode": mode, "cumulative_quota_sha256": hashlib.sha256(
            Path(quota_path).read_bytes()).hexdigest(), "reserve_cap": remaining_reserve,
        "scene_audit_sha256": hashlib.sha256(Path(scene_audit_path).read_bytes()).hexdigest(),
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
            or approval.get("bc_model_identity_sha256") != bc_sha \
            or approval.get("execution_basis_sha256") != _execution_basis_identity() \
            or approval.get("freeze_sha256") != hashlib.sha256(Path(freeze_path).read_bytes()).hexdigest() \
            or approval.get("quota_sha256") != hashlib.sha256(Path(quota_path).read_bytes()).hexdigest() \
            or approval.get("scene_audit_sha256") != hashlib.sha256(Path(scene_audit_path).read_bytes()).hexdigest():
        raise PermissionError("ROUND58 human approval does not bind the repaired generation/model/output")
    return _ValidatedGrant(_GRANT_TOKEN, {
        "authorization_path": Path(authorization_path).resolve(),
        "freeze_path": Path(freeze_path).resolve(), "quota_path": Path(quota_path).resolve(),
        "scene_audit_path": Path(scene_audit_path).resolve(), "output_root": expected_output,
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
           "bind_runner_paths": legacy.bind_runner_paths}
    plan_base_fn = old["plan_from_frozen_sources"]
    bind_base_fn = old["bind_live_runtime"]
    installed_adapters = []
    installed_runtimes = []
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
            return bind_paths_base(base, raw_root)
        finally:
            legacy.HERE = old["HERE"]
    def bind_repaired(base, runtime, plan, *, checkpoint_mode="forbid_existing"):
        plan = copy.deepcopy(plan)
        plan["training_version"] = REPAIRED_TRAINING_VERSION
        plan["r58_repair_identity"] = values["repair_identity"]
        live = bind_base_fn(base, runtime, plan, checkpoint_mode=checkpoint_mode)
        if len(installed_adapters) != 1:
            raise RuntimeError("repaired prior adapter was not installed exactly once")
        runtime["r56_gate_prior"] = installed_adapters[0]
        runtime["r58_repaired_adapter"] = installed_adapters[0]
        if checkpoint_mode == "forbid_existing" and Path(runtime["common"].CHECKPOINTS).resolve() != values["output_root"] / "checkpoints":
            raise RuntimeError("live Round58 checkpoint root differs from the approved generation root")
        contract = checkpoint_contract.install(
            runtime["trainer"], runtime["r40_train"],
            identity=values["repair_identity"], training_version=REPAIRED_TRAINING_VERSION,
            expected_checkpoint_root=runtime["common"].CHECKPOINTS,
            versions=runtime["r42_versions"], r42_run=runtime["r42_run"])
        runtime["r58_checkpoint_contract"] = contract
        counter_bridge = identity_ledger.bridge_ppo_update_counter(
            runtime["trainer"], runtime["r42_run"])
        runtime["r58_ppo_counter_bridge"] = counter_bridge
        runtime["r58_repair_identity"] = values["repair_identity"]
        if runtime["r40_train"].trainer is not runtime["trainer"]:
            contract["undo"]()
            raise RuntimeError("repaired checkpoint route is not the trainer used by R40")
        installed_contracts.append(contract)
        installed_runtimes.append((runtime, live))
        return live
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
        for runtime, live in reversed(installed_runtimes):
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
    return {"undo": undo, "installed_contracts": installed_contracts}


def _dispatch_validated(grant: _ValidatedGrant, *, legacy_loader=None):
    if not isinstance(grant, _ValidatedGrant) or not grant.values:
        raise PermissionError("ROUND58 execution requires a validated external receipt set")
    values = grant.values
    if not values.get("test_only"):
        paths = {"authorization": values["authorization_path"],
                 "freeze": values["freeze_path"], "quota": values["quota_path"],
                 "scene_audit": values["scene_audit_path"]}
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
        # This is the preserved ROUND58 fixed-plan orchestration, not a parallel trainer.
        return legacy.execute_authorized(
            auth_path=grant.values["authorization_path"], mode=grant.values["mode"])
    finally:
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
