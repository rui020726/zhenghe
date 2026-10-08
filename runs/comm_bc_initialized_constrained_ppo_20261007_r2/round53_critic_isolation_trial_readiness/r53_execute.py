"""ROUND53 private, authorization-gated execution harness.

Importing this file is inert.  ``execute_authorized`` requires a separate
human-issued quota receipt before runtime configuration, environment creation,
claim creation, optimizer recovery/update, or checkpoint output is possible.
This file is not invoked by the readiness/test commands.
"""
from __future__ import annotations
import hashlib, importlib.util, json, shutil, sys
from pathlib import Path
from typing import Any

from r53_contracts import (BRANCH, HARD_CAP, NORMAL_CAP, RESERVE_CAP, SCENES,
    SEEDS, TASK_ID, TRAINING_VERSION, authorize_receipt, build_fixed_plan,
    sha256_file, validate_claims, validate_slot_records)
from r53_contracts import nonfatal_derived_writer
from round53_trial_adapter import (install_on_verified_trainer,
    require_private_output_root, validate_live_binding)

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent
REPO = RUN_ROOT.parents[1]
CANON = REPO.parent / "zhenghe" / "runs" / "comm_cost_v2_rerun_20260928"
SOURCE_RUNNER = RUN_ROOT / "r46_execute_r2.py"
R45_DIR = REPO / "runs" / "comm_rule_warmstart_cost_learning_20261006_r2"
R43_DIR = REPO / "runs" / "comm_one_shot_full_trace_ablation_20261004"
R44_DIR = REPO / "runs" / "comm_dual_response_ablation_20261004"
R42_DIR = REPO / "runs" / "comm_one_shot_gate_resume_20261003_v2"
RAW = HERE / "raw"
PLAN_PATH = HERE / "TRIAL_PLAN.json"
AUTH_PATH = HERE / "EXECUTION_AUTHORIZATION.json"
RUNTIME_CODE_ALIASES = ("r40_train_resume_adapter.py", "r42_one_shot.py",
                        "r42_versions.py", "r43_call_counter_hooks.py")
R53_RUNNER_VERSION = "round53-private-runner/1+" + TRAINING_VERSION


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def plan_from_frozen_sources() -> dict[str, Any]:
    frozen = read_json(RUN_ROOT / "ROUND46_R2_V6_FROZEN_CONFIG.json")
    key_plan = read_json(RUN_ROOT / "ROUND46_R2_V6_KEY_PLAN.json")
    training_manifest = read_json(RUN_ROOT / "ROUND46_R2_V6_TRAINING_VERSION_MANIFEST.json")
    r44_freeze = read_json(R44_DIR / "ROUND44_FROZEN_CONFIG.json")
    r42 = r44_freeze["round42"]
    objective = r42["objective"]
    frozen_checks = {
        "steps_per_episode": r42.get("steps_per_episode") == 1000,
        "observation_width": r42.get("actor_feature_width") == 19,
        "b0_bytes": objective.get("b0_bytes") == 10000.0,
        "delta": objective.get("delta") == 0.0,
        "gamma": objective.get("gamma") == 1.0,
        "gae_lambda": objective.get("gae_lambda") == 1.0,
        "eta_lambda": objective.get("eta_lambda") == 0.5,
        "lambda_init": objective.get("lambda_init") == 1.0,
        "ALL_ROWS": "ALL_ROWS" in training_manifest.get("row_filtering", ""),
        "one_shot": training_manifest.get("action_semantics") == "one-shot-send-skip/1",
        "training_episodes": training_manifest.get("training_episodes") == 384,
        "updates": training_manifest.get("updates") == 96,
    }
    if not all(frozen_checks.values()):
        raise RuntimeError(f"frozen scientific configuration mismatch: {frozen_checks}")
    if sha256_file(SOURCE_RUNNER) != frozen.get("runner_source_sha256"):
        raise RuntimeError("actual ROUND46 runner source bytes differ from its frozen identity")
    order = [int(v) for v in key_plan["training_scenario_order"]]
    r44_order = [int(v) for v in read_json(R44_DIR / "ROUND44_FROZEN_CONFIG.json")["round42"]["training_order"]]
    if order != r44_order:
        raise RuntimeError("ROUND53 frozen training sequence differs from ROUND44")
    plan = build_fixed_plan(order)
    plan["training_order"] = order
    plan["training_order_sha256"] = hashlib.sha256(json.dumps(order).encode("utf-8")).hexdigest()
    plan["source_freeze_sha256"] = sha256_file(RUN_ROOT / "ROUND46_R2_V6_FROZEN_CONFIG.json")
    plan["source_key_plan_sha256"] = sha256_file(RUN_ROOT / "ROUND46_R2_V6_KEY_PLAN.json")
    plan["authorization"] = "PENDING_USER_APPROVAL"
    input_identity = read_json(RUN_ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json")
    registered = input_identity.get("round44_registered_ep0_rng", {})
    frozen_registered = r44_freeze.get("round43", {}).get("registered_ep0_inputs", {})
    if set(registered) != {str(seed) for seed in SEEDS} or set(frozen_registered) != set(registered):
        raise RuntimeError("registered R44 ep0 RNG identities are not exactly the three frozen seeds")
    rng_sources = {}
    for seed in SEEDS:
        entry = registered[str(seed)]
        frozen_entry = frozen_registered[str(seed)]
        path = Path(entry["path"])
        expected_sha = entry["sha256"]
        if (Path(frozen_entry["path"]).resolve() != path.resolve()
                or frozen_entry.get("file_sha256") != expected_sha
                or path.name != f"v2_train_s{seed}_ep0.pt"
                or not path.is_file() or sha256_file(path) != expected_sha):
            raise RuntimeError(f"seed {seed}: registered R44 ep0 RNG source path/hash mismatch")
        rng_sources[str(seed)] = {"path": str(path.resolve()), "sha256": expected_sha,
                                  "rng_components": ["python", "numpy", "torch"],
                                  "load_role": "rng_only_after_same_seed_BC_POST_model_init",
                                  "model_or_optimizer_restored": False}
    plan["registered_r44_ep0_rng_only"] = rng_sources
    plan["source_content_identities"] = {
        "round46_runner_sha256": sha256_file(SOURCE_RUNNER),
        "canonical_trainer_sha256": sha256_file(CANON / "v2_trainer.py"),
        "round44_freeze_sha256": sha256_file(R44_DIR / "ROUND44_FROZEN_CONFIG.json"),
        "round44_reuse_manifest_sha256": sha256_file(R44_DIR / "ROUND44_BASELINE_REUSE.json"),
        "round46_input_identity_sha256": sha256_file(RUN_ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json"),
        "round46_v10_manifest_sha256": sha256_file(RUN_ROOT / "v10_eval_final_close" / "ROUND46_R2_V10_ANALYSIS_INPUT_MANIFEST.json"),
        "round52_prototype_contract_sha256": sha256_file(RUN_ROOT / "round52_critic_gradient_isolation_prototype" / "PROTOTYPE_CONTRACT.json"),
        "round52_gradient_evidence_sha256": sha256_file(RUN_ROOT / "round52_critic_gradient_isolation_prototype" / "FORWARD_AND_GRADIENT_COMPARISON.json"),
        "r53_executor_sha256": sha256_file(HERE / "r53_execute.py"),
        "r53_contracts_sha256": sha256_file(HERE / "r53_contracts.py"),
        "r53_policy_adapter_sha256": sha256_file(HERE / "round53_trial_adapter.py"),
    }
    plan["scientific_config_identity"] = {
        "objective": "min E[c] s.t. g=E[Yref]-E[Ypi]-delta<=0",
        "actor_advantage": "lambda_before*A_task-A_positive_cost",
        "frozen_round44_objective": objective,
        "frozen_checks": frozen_checks,
        "search_and_allocation": "reuse exact ROUND44 frozen runtime; no edits",
        "ack": "disabled per frozen immediate/training protocol",
        "timing": "policy_owned_timing",
        "actions": "one-shot SEND/SKIP; original masks and greedy tie rule",
        "changed_training_factor_only": "critic representation stop-gradient at task/cost critic-head inputs",
        "fresh_rng_scheme": "restore only the exact registered same-seed ROUND44 ep0 Python/NumPy/Torch RNG payload after BC_POST model-only initialization via frozen R42 train_seed; do not restore its model or optimizer",
    }
    return plan


def verify_baseline_reuse() -> dict[str, Any]:
    """Hash-close exactly the registered rows needed for 48+96+32 reuse."""
    baseline = read_json(R44_DIR / "ROUND44_BASELINE_REUSE.json")
    v6_identity = read_json(RUN_ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json")
    v10 = RUN_ROOT / "v10_eval_final_close"
    v10_manifest = read_json(v10 / "ROUND46_R2_V10_ANALYSIS_INPUT_MANIFEST.json")
    verdict = read_json(v10 / "ROUND46_R2_V10_MACHINE_VERDICT.json")
    final_results = read_json(v10 / "ROUND46_R2_V10_FINAL_RESULTS.json")
    expected_training_version = read_json(RUN_ROOT / "ROUND46_R2_V6_TRAINING_VERSION_MANIFEST.json")["training_version"]
    ep128_parameters = {str(seed): final_results["seed_results"][str(seed)]["final_parameter_identity"]["parameter_sha256"]
                        for seed in SEEDS}
    ep128_models = {}
    for seed in SEEDS:
        summary_path = RUN_ROOT / f"ROUND46_TRAINING_s{seed}.json"
        summary = read_json(summary_path)
        model_path = Path(summary["ep128_checkpoint_path"])
        if (summary.get("status") != "complete" or summary.get("training_version") != expected_training_version
                or int(summary.get("fixed_ep128_report", {}).get("episodes_done", -1)) != 128
                or not model_path.is_file()
                or sha256_file(model_path) != summary.get("ep128_checkpoint_sha256")):
            raise RuntimeError(f"PPO seed {seed} actual ep128 checkpoint identity is not closed")
        ep128_models[str(seed)] = {"summary_path": str(summary_path),
                                   "summary_sha256": sha256_file(summary_path),
                                   "checkpoint_path": str(model_path),
                                   "checkpoint_sha256": summary["ep128_checkpoint_sha256"],
                                   "parameter_sha256": ep128_parameters[str(seed)]}
    order = [int(x) for x in read_json(RUN_ROOT / "ROUND46_R2_V6_KEY_PLAN.json")["training_scenario_order"]]
    train_scenes = set(order)
    immediate_registry = {int(item["scene"]): item for item in baseline["round42_immediate_references"]}
    if not train_scenes.issubset(immediate_registry):
        raise RuntimeError(f"immediate training cache misses frozen scenes: {sorted(train_scenes-set(immediate_registry))}")
    immediate_rows = []
    common_identity = None
    for scene in sorted(train_scenes):
        item = immediate_registry[scene]; path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"immediate cache hash missing/mismatch for scene {scene}: {path}")
        if int(item.get("size_bytes", path.stat().st_size)) != path.stat().st_size:
            raise RuntimeError(f"immediate cache byte length mismatch for scene {scene}: {path}")
        row = read_json(path)
        expected_fields = {"scenario_seed": scene, "reference_arm": "immediate",
                           "policy": "immediate", "steps": 1000,
                           "budget_bytes": 29300, "comm_action_semantics": "one-shot-send-skip/1",
                           "comm_action_eligibility": "first-report-only/1",
                           "reward_timing_version": "published-completion-event-at-relay-tick/2",
                           "ack_enabled": False}
        # ACK is nested under transport_parameters in archived rows.
        for key, expected in expected_fields.items():
            if key == "ack_enabled":
                actual = row.get("transport_parameters", {}).get(key)
            else: actual = row.get(key)
            if actual != expected:
                raise RuntimeError(f"immediate scene {scene} {key}: {actual!r} != {expected!r}")
        if not isinstance(row.get("on_time"), int) or not isinstance(row.get("spent_bytes"), int):
            raise RuntimeError(f"immediate scene {scene} lacks integer on_time/spent_bytes")
        transport = row.get("transport_parameters", {})
        if (transport.get("timing_mode") != "policy_owned_timing"
                or transport.get("ack_enabled") is not False
                or transport.get("budget", {}).get("bytes_per_episode") != 29300):
            raise RuntimeError(f"immediate scene {scene} timing/ACK/budget protocol mismatch")
        # World/config and behaviour fingerprints legitimately vary with scene;
        # compare only protocol/transport invariants across scenes, while the
        # per-scene bytes and hashes remain independently closed above.
        identity = (row.get("training_version"), row.get("reward_timing_version"),
                    row.get("comm_action_semantics"), row.get("comm_action_eligibility"),
                    json.dumps(row.get("transport_parameters"), sort_keys=True))
        if common_identity is None: common_identity = identity
        elif identity != common_identity:
            raise RuntimeError("immediate cache rows differ in frozen config/behavior/transport identity")
        immediate_rows.append({"scene": scene, "path": str(path), "sha256": item["sha256"],
                               "on_time": row["on_time"], "spent_bytes": row["spent_bytes"],
                               "config_hash": row["config_hash"],
                               "behaviour_sha256": row["behaviour_sha256"],
                               "protocol_identity": list(identity[:4])})

    bc_rows = []
    bc_model_sources = []
    for seed_text, model in v6_identity.get("bc_models_by_seed", {}).items():
        seed = int(seed_text)
        for path_key, sha_key in (("checkpoint_path", "checkpoint_sha256"),
                                  ("readback_receipt_path", "readback_receipt_sha256"),
                                  ("metrics_path", "metrics_sha256"),
                                  ("journal_path", "journal_sha256")):
            path = Path(model[path_key])
            if not path.is_file() or sha256_file(path) != model[sha_key]:
                raise RuntimeError(f"BC_POST seed {seed} source hash mismatch: {path}")
        bc_model_sources.append({"seed": seed,
                                 "checkpoint_path": model["checkpoint_path"],
                                 "checkpoint_sha256": model["checkpoint_sha256"],
                                 "parameter_sha256": model["parameter_sha256"],
                                 "readback_receipt_sha256": model["readback_receipt_sha256"],
                                 "metrics_sha256": model["metrics_sha256"],
                                 "journal_sha256": model["journal_sha256"]})
    if {m["seed"] for m in bc_model_sources} != set(SEEDS):
        raise RuntimeError("BC_POST model-only initialization identities are not exactly seed matched")
    if len(v6_identity.get("round45_control_rows", [])) != 48:
        raise RuntimeError("BC greedy control registry is not exactly 48 rows")
    for item in v6_identity["round45_control_rows"]:
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"BC greedy row hash mismatch: {path}")
        row = read_json(path)
        seed = int(row.get("training_seed", -1))
        scene = int(item.get("scene", row.get("scenario_seed", -1)))
        transport = row.get("transport_parameters", {})
        if (row.get("model_arm") != "BC_ONLY_R2" or row.get("eval_mode") != "greedy"
                or seed not in SEEDS
                or int(row.get("scenario_seed", -1)) != scene or scene not in SCENES
                or int(row.get("steps", -1)) != 1000
                or row.get("training_version") != expected_training_version
                or row.get("reward_timing_version") != "published-completion-event-at-relay-tick/2"
                or row.get("comm_action_semantics") != "one-shot-send-skip/1"
                or row.get("comm_action_eligibility") != "first-report-only/1"
                or transport.get("timing_mode") != "policy_owned_timing"
                or transport.get("ack_enabled") is not False
                or transport.get("budget", {}).get("bytes_per_episode") != 29300
                or not isinstance(row.get("config_hash"), str) or not row["config_hash"]
                or not isinstance(row.get("behaviour_sha256"), str) or not row["behaviour_sha256"]
                or row.get("budget_bytes") != 29300 or row.get("b0_bytes") != 10000.0
                or not isinstance(row.get("on_time"), int)
                or not isinstance(row.get("spent_bytes"), int)
                or not 0 <= row["spent_bytes"] <= row["budget_bytes"]):
            raise RuntimeError(f"BC greedy row schema/protocol mismatch: {path}")
        expected_parameter = v6_identity["bc_models_by_seed"][str(seed)]["parameter_sha256"]
        actual_parameter = row.get("checkpoint_parameter_identity", {}).get("parameter_sha256")
        if (actual_parameter != expected_parameter
                or row.get("bc_parameter_sha256") != expected_parameter
                or row.get("bc_checkpoint_sha256") != v6_identity["bc_models_by_seed"][str(seed)]["checkpoint_sha256"]):
            raise RuntimeError(f"BC greedy row parameter role mismatch: {path}")
        bc_rows.append({"path": str(path), "sha256": item["sha256"],
                        "seed": seed, "scene": scene,
                        "bc_parameter_sha256": row["bc_parameter_sha256"],
                        "bc_checkpoint_sha256": row["bc_checkpoint_sha256"],
                        "config_hash": row["config_hash"],
                        "behaviour_sha256": row["behaviour_sha256"],
                        "spent_bytes": row["spent_bytes"], "budget_bytes": row["budget_bytes"]})
    if len({(r["seed"], r["scene"]) for r in bc_rows}) != 48:
        raise RuntimeError("BC greedy rows do not close the 3x16 key matrix")

    ppo_rows = [row for row in v10_manifest["rows"]
                if row.get("role") in ("new_eval_row", "adopted_row")]
    # The input manifest records all staged source bytes; accepted matrix is authoritative.
    accepted = verdict.get("effective_matrix", {}).get("effective_rows")
    if accepted != 96 or len(ppo_rows) < 96:
        raise RuntimeError(f"shared PPO accepted matrix/manifest mismatch: {accepted}/{len(ppo_rows)}")
    ppo_checked = []
    for item in ppo_rows:
        path = Path(item["source"])
        if not path.is_file() or sha256_file(path) != item["source_sha256"]:
            raise RuntimeError(f"PPO eval source row hash mismatch: {path}")
        row = read_json(path)
        if row.get("eval_mode") not in ("greedy", "sampled"):
            continue
        seed = int(row.get("training_seed", -1)); scene = int(row.get("scenario_seed", -1))
        mode = row["eval_mode"]
        transport = row.get("transport_parameters", {})
        actual_parameter = row.get("checkpoint_parameter_identity", {}).get("parameter_sha256")
        if (row.get("model_arm") != "NEW"
                or int(row.get("steps", -1)) != 1000
                or seed not in SEEDS
                or scene not in SCENES
                or row.get("training_version") != expected_training_version
                or row.get("reward_timing_version") != "published-completion-event-at-relay-tick/2"
                or row.get("comm_action_semantics") != "one-shot-send-skip/1"
                or row.get("comm_action_eligibility") != "first-report-only/1"
                or transport.get("timing_mode") != "policy_owned_timing"
                or transport.get("ack_enabled") is not False
                or transport.get("budget", {}).get("bytes_per_episode") != 29300
                or not isinstance(row.get("config_hash"), str) or not row["config_hash"]
                or not isinstance(row.get("behaviour_sha256"), str) or not row["behaviour_sha256"]
                or row.get("budget_bytes") != 29300 or row.get("b0_bytes") != 10000.0
                or not isinstance(row.get("on_time"), int)
                or not isinstance(row.get("spent_bytes"), int)
                or not 0 <= row["spent_bytes"] <= row["budget_bytes"]
                or actual_parameter != ep128_parameters.get(str(seed))):
            raise RuntimeError(f"PPO final row schema/configuration mismatch: {path}")
        ppo_checked.append({"path": str(path), "sha256": item["source_sha256"],
                            "seed": seed, "scene": scene, "mode": mode,
                            "ep128_parameter_sha256": actual_parameter,
                            "config_hash": row["config_hash"],
                            "behaviour_sha256": row["behaviour_sha256"],
                            "spent_bytes": row["spent_bytes"], "budget_bytes": row["budget_bytes"]})
    if (len(ppo_checked) != 96
            or len({(r["seed"], r["scene"], r["mode"]) for r in ppo_checked}) != 96):
        raise RuntimeError("shared PPO source rows do not close 3x2x16")

    refs = {}
    for label, entries in (("immediate_dev", baseline["round42_immediate_references"]),
                           ("late_dev", [r for r in baseline["round42_fixed_rules"]
                                          if r.get("arm") == "late_discovery_gate"])):
        selected = [r for r in entries if int(r["scene"]) in SCENES]
        if len(selected) != 16: raise RuntimeError(f"{label} registry is not 16 scenes")
        checked = []
        for item in selected:
            path = Path(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"{label} row hash mismatch: {path}")
            row = read_json(path)
            if not isinstance(row.get("on_time"), int) or not isinstance(row.get("spent_bytes"), int):
                raise RuntimeError(f"{label} row lacks integer source fields: {path}")
            checked.append({"scene": int(item["scene"]), "path": str(path), "sha256": item["sha256"]})
        refs[label] = checked
    return {"kind": "round53-baseline-reuse-verification/2", "status": "passed",
            "training_immediate_cache": {"rows": len(immediate_rows), "scene_ids": sorted(train_scenes),
                                           "rows_by_scene": immediate_rows,
            "protocol_transport_identity": list(common_identity or ())},
            "bc_greedy": {"rows": len(bc_rows), "keys": bc_rows},
            "bc_post_model_sources": bc_model_sources,
            "shared_ppo": {"accepted_rows": accepted, "verified_source_rows": len(ppo_checked),
                           "keys": ppo_checked,
                           "ep128_models": ep128_models,
                           "manifest_sha256": sha256_file(v10 / "ROUND46_R2_V10_ANALYSIS_INPUT_MANIFEST.json")},
            "development_references": refs,
            "bc_sampled": {"required_new_evaluation_rows": 48,
                           "existing_input_required": False,
                           "not_generated_in_readiness": True},
            "historical_claims_rescanned": False}


def load_base_runner() -> Any:
    """Pure import of the frozen Round46 module; does not configure runtime."""
    spec = importlib.util.spec_from_file_location("round53_frozen_r46_runner", SOURCE_RUNNER)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load frozen runner source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bind_runner_paths(base: Any, raw_root: Path = RAW) -> dict[str, str]:
    """Rebind every path global physically rooted under ROUND46, preserving source roots."""
    old_root = Path(base.R46_DIR).resolve(); raw_root = Path(raw_root).resolve()
    require_private_output_root(HERE, [raw_root])
    old_v6 = (old_root / "v6").resolve()
    if raw_root == old_root or raw_root == old_v6 or old_v6 in raw_root.parents:
        raise RuntimeError("private raw root overlaps the historical ROUND46/V6 output namespace")
    bound = {}
    for name, value in list(vars(base).items()):
        if not isinstance(value, Path) or name in ("PRIOR_R46_DIR",): continue
        resolved = value.resolve()
        if resolved == old_root or old_root in resolved.parents:
            mapped = raw_root / resolved.relative_to(old_root)
            setattr(base, name, mapped)
            bound[name] = str(mapped)
    base.R46_DIR = raw_root
    base.V6_DIR = raw_root
    base.V6_RUNTIME_DIR = raw_root / "runtime"
    base.V6_CHECKPOINTS_DIR = raw_root / "checkpoints"
    base.V6_PROGRESS_DIR = raw_root / "training_progress"
    base.V6_LOGS_DIR = raw_root / "logs"
    base.V6_ROWS_DIR = raw_root / "rows"
    base.V6_REJECTED_DIR = raw_root / "rejected"
    base.FAMILY_LEDGER_PATH = raw_root / "runtime" / "ROUND53_EXECUTION_LEDGER.json"
    base.LEDGER_PATH = base.FAMILY_LEDGER_PATH
    base.FAMILY_JOURNAL_PATH = raw_root / "runtime" / "checkpoint_writes.jsonl"
    base.CHECKPOINT_JOURNAL_PATH = base.FAMILY_JOURNAL_PATH
    base.ROUND46_BRANCH = BRANCH
    base.ROUND46_RUNNER_VERSION = R53_RUNNER_VERSION
    # Keep the source runtime's version only during configuration; the explicit
    # R53 resume/version boundary is installed immediately afterward, before any
    # training API is reachable.
    base.TRAINING_KEYS, base.EVALUATION_KEYS = 384, 144
    base.NORMAL_CAP, base.RESERVE_CAP, base.HARD_CAP = NORMAL_CAP, RESERVE_CAP, HARD_CAP
    # The frozen training order is read-only input, not an R53 output artifact.
    base.KEY_PLAN_PATH = RUN_ROOT / "ROUND46_R2_V6_KEY_PLAN.json"
    bound.update({"R46_DIR": str(raw_root), "V6_DIR": str(raw_root),
                  "LEDGER_PATH": str(base.LEDGER_PATH),
                  "CHECKPOINT_JOURNAL_PATH": str(base.CHECKPOINT_JOURNAL_PATH)})
    require_private_output_root(raw_root, [Path(v) for v in bound.values()
                                           if Path(v).is_absolute()
                                           and Path(v).resolve() != raw_root])
    return bound


def install_private_configure_seams(base: Any) -> None:
    """Replace V6/family-journal seams; never call the historical V6 seeder."""
    def prepare_r53_output_namespace():
        for path in (base.V6_DIR, base.V6_RUNTIME_DIR, base.V6_CHECKPOINTS_DIR,
                     base.V6_PROGRESS_DIR, base.V6_LOGS_DIR, base.V6_ROWS_DIR,
                     base.V6_REJECTED_DIR):
            path.mkdir(parents=True, exist_ok=True)
        for name in RUNTIME_CODE_ALIASES:
            source = R45_DIR / name
            destination = base.V6_DIR / name
            if not source.is_file(): raise FileNotFoundError(source)
            if destination.exists() and sha256_file(destination) != sha256_file(source):
                raise RuntimeError(f"private runtime alias conflict: {destination}")
            if not destination.exists(): shutil.copyfile(source, destination)
            if sha256_file(destination) != sha256_file(source):
                raise RuntimeError(f"private runtime alias copy mismatch: {name}")

    base._seed_v6_output_namespace = prepare_r53_output_namespace
    base._copy_runtime_aliases = lambda _identity: prepare_r53_output_namespace()
    base._input_identity = lambda: read_json(RUN_ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json")
    base._checkpoint_journal_route_audit = lambda create=False: {
        "kind": "round53-private-journal-route/1", "status": "fresh_private_route",
        "journal_path": str(base.CHECKPOINT_JOURNAL_PATH),
        "historical_family_journal_copied": False}

    def rebind_private_checkpoint_journal(runtime: dict[str, Any]) -> dict[str, Any]:
        trainer = runtime["trainer"]
        run = runtime["r42_run"]
        save = trainer.save_checkpoint
        cells = dict(zip(save.__code__.co_freevars, save.__closure__ or (), strict=True))
        cell = cells.get("checkpoint_journal")
        if cell is None: raise RuntimeError("actual trainer save hook lacks checkpoint journal closure")
        before = Path(cell.cell_contents).resolve()
        allowed = {Path(base.LEGACY_CHECKPOINT_JOURNAL_PATH).resolve(),
                   Path(base.CHECKPOINT_JOURNAL_PATH).resolve()}
        if before not in allowed: raise RuntimeError(f"unexpected actual checkpoint journal closure: {before}")
        target = Path(base.CHECKPOINT_JOURNAL_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): target.write_bytes(b"")
        elif target.stat().st_size:
            raise RuntimeError("new private checkpoint journal is not empty before training")
        cell.cell_contents = target
        run.RUN_DIR = base.V6_DIR
        return {"save_hook": "trainer.save_checkpoint", "closure_before": str(before),
                "closure_after": str(target.resolve()), "run_root": str(Path(run.RUN_DIR).resolve()),
                "journal_is_private": True, "family_journal_copied": False}
    base._rebind_checkpoint_journal = rebind_private_checkpoint_journal


PROTECTED_PREEXISTING_DIRTY_FILES = {
    "runs/comm_bc_initialized_constrained_ppo_20261007_r2/ROUND46_CURRENT_TASK.json",
    "runs/comm_bc_initialized_constrained_ppo_20261007_r2/ROUND46_R2_V6_STOP_STATE.json",
}


def install_protected_dirty_file_guard(base: Any) -> dict[str, Any]:
    """Allow only the two byte-unchanged user-protected prelaunch dirty files."""
    original = base._assert_source_worktree_clean

    def guarded_assert_source_worktree_clean() -> None:
        status = base._run_git("status", "--porcelain", "--untracked-files=no")
        unexpected = base._unexpected_source_status_lines(status)
        remaining = []
        allowed = []
        for line in unexpected:
            relative = line[3:].replace("\\", "/") if len(line) >= 4 else ""
            if relative in PROTECTED_PREEXISTING_DIRTY_FILES:
                allowed.append(relative)
            else:
                remaining.append(line)
        if remaining:
            raise RuntimeError(
                "tracked source worktree is dirty; ROUND53 allows only protected files: "
                + "\n".join(remaining))
        return None

    guarded_assert_source_worktree_clean.__name__ = "guarded_assert_source_worktree_clean"
    base._assert_source_worktree_clean = guarded_assert_source_worktree_clean
    return {"original_function": original, "allowed_relative_paths":
            sorted(PROTECTED_PREEXISTING_DIRTY_FILES)}


def install_r53_claim_verifier(base: Any, runtime: dict[str, Any]) -> dict[str, Any]:
    """Replace the R46 historical-identity claim verifier for the private R53 route.

    Keeps the frozen-source identity, branch/start-ancestor, and guarded
    worktree-clean checks; validates the fresh private journal route instead of
    the R46 prelaunch legacy-journal identity.  No science/config/trainer change.
    """
    r45_module = getattr(base, "_R45_MODULE", None)
    r44_module = getattr(r45_module, "R44_RUNTIME_MODULE", None) if r45_module else None
    if r44_module is None:
        raise RuntimeError("R53 claim verifier could not locate the live R44 module")
    stale = getattr(r44_module, "verify_frozen_source", None)
    stale_code = getattr(stale, "__code__", None)
    if (stale_code is None
            or getattr(stale, "__name__", "") != "round46_r2_v5_claim_source_verifier"
            or Path(stale_code.co_filename).resolve() != SOURCE_RUNNER.resolve()):
        raise RuntimeError("R53 expected the R46 strict claim verifier before rebinding the R44 global")
    registered_frozen = read_json(R44_DIR / "ROUND44_FROZEN_CONFIG.json")
    raw = Path(base.V6_DIR).resolve()
    journal = Path(base.CHECKPOINT_JOURNAL_PATH).resolve()

    def r53_claim_verifier(frozen: dict[str, Any]) -> str:
        if frozen != registered_frozen:
            raise RuntimeError("R44 claim supplied a frozen object outside the registered R44 source")
        base._assert_branch_head()
        base._assert_source_worktree_clean()
        if Path(runtime["r42_run"].RUN_DIR).resolve() != raw:
            raise RuntimeError("R53 private run root moved before claim")
        if journal.parent != (raw / "runtime").resolve():
            raise RuntimeError("R53 claim journal is not the fresh private journal")
        audit = base._checkpoint_journal_route_audit(create=False)
        if (audit.get("status") != "fresh_private_route"
                or audit.get("historical_family_journal_copied") is not False
                or Path(audit.get("journal_path", "")).resolve() != journal):
            raise RuntimeError("R53 checkpoint journal route audit does not close the fresh private journal")
        return base._run_git("rev-parse", "HEAD")

    r53_claim_verifier.__name__ = "round53_private_claim_source_verifier"
    r53_claim_verifier.__qualname__ = r53_claim_verifier.__name__
    r44_module.verify_frozen_source = r53_claim_verifier
    if r44_module.verify_frozen_source is not r53_claim_verifier:
        raise RuntimeError("R53 private claim verifier rebinding failed")
    return {"kind": "round53-private-claim-verifier/1", "status": "installed",
            "rebound_from": {"name": stale.__name__,
                             "source": str(Path(stale_code.co_filename).resolve())},
            "r44_module_file": (str(Path(r44_module.__file__).resolve())
                                if getattr(r44_module, "__file__", None) else None),
            "private_journal": str(journal)}


def seed_private_task_state(raw: Path) -> None:
    """Seed the live R45 task-state owner without touching historical task files."""
    path = raw / "ROUND45_CURRENT_TASK.json"
    if path.exists():
        state = read_json(path)
        if state.get("kind") != "round53-private-task-state/1":
            raise RuntimeError("private R53 task state already exists with an unexpected identity")
        return
    state = {
        "kind": "round53-private-task-state/1", "round": 53, "task_id": TASK_ID,
        "branch": BRANCH, "training_version": TRAINING_VERSION,
        "attempt": "ROUND53-private-critic-isolation/1",
        "output_root": str(Path(raw).resolve()),
        "environment_constructed": 0, "normal_claims": 0,
        "reserve_claims": 0, "ppo_updates": 0, "optimizer_steps": 0,
        "checkpoint_writes": 0, "remote_push": False,
        "seed_reason": "R45 task_state expects an existing mutable current-task file in its RUN_DIR",
    }
    raw.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bind_live_runtime(base: Any, runtime: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:

    """Explicitly bind all output-owning live modules and actual trainer call path."""
    raw = Path(base.V6_DIR).resolve()
    required = ("common", "r40_common", "r42_run", "r40_train", "trainer",
                "r42_versions", "controller_module", "runner")
    missing = [key for key in required if key not in runtime]
    if missing: raise RuntimeError(f"runtime missing required live objects: {missing}")
    common, r40 = runtime["common"], runtime["r40_common"]
    output_fields = {
        common: {"OUTPUT_DIR": raw, "LOGS": raw / "logs", "ROWS": raw / "rows",
                 "REJECTED": raw / "rejected", "CHECKPOINTS": raw / "checkpoints",
                 "NORMAL_LAUNCHES": NORMAL_CAP, "RESERVE_LAUNCHES": RESERVE_CAP,
                 "LAUNCH_CAP": HARD_CAP},
        r40: {"ROUND_DIR": raw, "NORMAL_CAP": NORMAL_CAP,
              "RESERVE_CAP": RESERVE_CAP, "HARD_CAP": HARD_CAP},
    }
    for obj, fields in output_fields.items():
        for name, value in fields.items(): setattr(obj, name, value)
    r40.plan = lambda: {"normal_total": NORMAL_CAP,
                        "technical_reserve_max": RESERVE_CAP, "hard_cap": HARD_CAP,
                        "training_keys": 384, "evaluation_keys": 144,
                        "task_id": TASK_ID}
    run = runtime["r42_run"]
    run.RUN_DIR = raw
    seed_private_task_state(raw)
    # These are the modules actually imported by R45/R44/R42; bind each named
    # owner explicitly, including r44_runtime (the prior leak point).
    r45_module = getattr(base, "_R45_MODULE", None)
    r44_runtime = getattr(r45_module, "R44_RUNTIME_MODULE", None) if r45_module else None
    analysis_module = runtime.get("r42_analysis") or sys.modules.get("r42_analysis")
    output_modules = {
        "r42_run": run,
        "r44_runtime": r44_runtime,
        "r44_prepare": sys.modules.get("r44_prepare"),
        "r42_analysis": analysis_module,
    }
    for label, module in output_modules.items():
        if module is None or not hasattr(module, "RUN_DIR"):
            raise RuntimeError(f"live output-owning module lacks explicit RUN_DIR binding: {label}")
        module.RUN_DIR = raw
    # R42 obtains cache rows through this function at train_seed call time.
    immediate = {int(row["scene"]): read_json(Path(row["path"]))
                 for row in verify_baseline_reuse()["training_immediate_cache"]["rows_by_scene"]}
    run.reference_cache = lambda: immediate
    r40_train = runtime["r40_train"]
    r40_train.common = common
    r40_train.runner = runtime["runner"]
    r40_train.trainer = runtime["trainer"]
    original_r40_train_seed = r40_train.train_seed
    def verified_r40_train_seed(*args, **kwargs):
        if (r40_train.trainer is not runtime["trainer"]
                or r40_train.common is not common
                or r40_train.runner is not runtime["runner"]):
            raise RuntimeError("actual r42_run -> r40_train binding does not use the verified live objects")
        return original_r40_train_seed(*args, **kwargs)
    r40_train.train_seed = verified_r40_train_seed

    # Preserve the frozen R46/R45/R42 fresh-start route.  R45 injects the
    # same-seed BC_POST model-only state; the actual R42 train_seed then loads
    # only the registered R44 ep0 RNG and restores it after fresh_controller.
    # Same-R53 continuation is handled by the unchanged R40 resume adapter.
    frozen_rng = runtime["frozen"]["round43"]["registered_ep0_inputs"]
    expected_rng = plan["registered_r44_ep0_rng_only"]
    for seed in SEEDS:
        actual = frozen_rng.get(str(seed), {})
        expected = expected_rng[str(seed)]
        if (Path(actual.get("path", "")).resolve() != Path(expected["path"]).resolve()
                or actual.get("file_sha256") != expected["sha256"]
                or sha256_file(Path(expected["path"])) != expected["sha256"]):
            raise RuntimeError(f"runtime frozen R44 ep0 RNG binding mismatch for seed {seed}")
    legacy_train = runtime["round46_r2_legacy_call_training_seed"]
    train_seed = run.train_seed
    if (not callable(legacy_train) or legacy_train.__name__ != "_call_training_seed"
            or Path(legacy_train.__code__.co_filename).resolve()
               != (R45_DIR / "r45_r2_execute_r3.py").resolve()
            or not callable(train_seed) or train_seed.__name__ != "train_seed"
            or Path(train_seed.__code__.co_filename).resolve()
               != (R43_DIR / "r42_run.py").resolve()):
        raise RuntimeError("frozen R46->R45 BC initializer->R42 registered-RNG training route is not live")
    r46_training_call = runtime["module"]._call_training_seed
    if (not callable(r46_training_call)
            or Path(r46_training_call.__code__.co_filename).resolve() != SOURCE_RUNNER.resolve()):
        raise RuntimeError("ROUND46 factory-scope wrapper is not installed around the R45 training call")
    rng_route_identity = {"R46_training_wrapper": str(Path(r46_training_call.__code__.co_filename).resolve()),
                          "R45_model_only_initializer": str(Path(legacy_train.__code__.co_filename).resolve()),
                          "R42_train_seed": str(Path(train_seed.__code__.co_filename).resolve()),
                          "registered_sources": expected_rng,
                          "restore_order": "BC_POST fresh_controller -> exact registered R44 ep0 RNG restore -> R40 first episode",
                          "restores_old_model_optimizer": False}

    # A pre-claim guard validates every logical key before the actual durable claim.
    jobs = {item["logical_job"]: item for item in plan["jobs"]}
    original_claim = run._claim
    def planned_claim(logical_job, *args, **kwargs):
        if logical_job not in jobs:
            raise RuntimeError(f"unplanned ROUND53 claim refused: {logical_job}")
        attempt = int(kwargs.get("attempt", args[0] if args else 1))
        job = jobs[logical_job]
        slug = runtime["r40_common"]._slug(str(logical_job))
        attempt_one = raw / "launch_claims" / slug / "attempt_0001.json"
        if attempt == 1:
            if attempt_one.exists(): raise RuntimeError(f"duplicate normal claim refused: {logical_job}")
        elif attempt == 2:
            original = read_json(attempt_one) if attempt_one.is_file() else {}
            if (original.get("logical_job") != logical_job
                    or int(original.get("attempt", -1)) != 1
                    or original.get("category") != "normal"
                    or int(original.get("launch_slot", -1)) != int(job["normal_slot"])
                    or original.get("status") != "failed_technical"):
                raise RuntimeError("technical reserve retry requires an actual failed_technical attempt 1")
        else:
            raise RuntimeError(f"attempt number outside frozen one-retry contract: {attempt}")
        claim_path = original_claim(logical_job, *args, **kwargs)
        claim = read_json(claim_path)
        expected_category = "normal" if attempt == 1 else "reserve"
        slot = int(claim.get("launch_slot", 0))
        expected_slot = int(job["normal_slot"]) if attempt == 1 else slot
        slot_path = raw / "launch_slots" / expected_category / f"{slot:04d}.json"
        slot_record = read_json(slot_path) if slot_path.is_file() else {}
        if (claim.get("logical_job") != logical_job or int(claim.get("attempt", -1)) != attempt
                or claim.get("category") != expected_category
                or (attempt == 1 and slot != expected_slot)
                or slot_record.get("category") != expected_category
                or int(slot_record.get("slot", 0)) != slot
                or slot_record.get("logical_job") != logical_job
                or int(slot_record.get("attempt", 0)) != attempt):
            raise RuntimeError("persisted claim/slot identity differs from the frozen ROUND53 plan")
        return claim_path
    run._claim = planned_claim

    # The atomic claim is authoritative. A derived snapshot/write failure is
    # recorded separately and must not escape into _execute_cell's broad except
    # (which would otherwise flip an already persisted complete claim to failed).
    snapshot_sequence = {"n": 0}
    def save_snapshot_warning(warning):
        snapshot_sequence["n"] += 1
        warning_dir = raw / "runtime" / "derived_snapshot_warnings"
        warning_dir.mkdir(parents=True, exist_ok=True)
        path = warning_dir / f"snapshot_{snapshot_sequence['n']:06d}.json"
        path.write_text(json.dumps(warning, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    run.write_execution_ledger = nonfatal_derived_writer(
        run.write_execution_ledger, save_snapshot_warning)

    # Route the exact physical trainer object; the r42_run -> r40_train adapter
    # assigns this same object immediately before entering train_seed.
    undo_update = install_on_verified_trainer(
        runtime["trainer"], source_sha256=sha256_file(CANON / "v2_trainer.py"),
        source_path=CANON / "v2_trainer.py")
    runtime["r40_train"].trainer = runtime["trainer"]
    validate_live_binding(runtime, runtime["trainer"])

    # New freeze identity and semantic version make old PPO payloads incompatible.
    freeze = runtime["round42_freeze"]
    freeze["training_version"] = TRAINING_VERSION
    freeze["critic_gradient_route_version"] = "critic_representation_stop_gradient/1"
    versions = runtime["r42_versions"]
    versions.TRAINING_VERSION = TRAINING_VERSION
    base.TRAINING_VERSION = TRAINING_VERSION
    base.ROUND46_RUNNER_VERSION = R53_RUNNER_VERSION
    runtime["trainer"].TRAINING_VERSION = TRAINING_VERSION
    run.TRAINING_VERSION = TRAINING_VERSION
    runtime["runner"].TRAINING_VERSION = TRAINING_VERSION
    runtime["round42_freeze"] = freeze
    # The R40 adapter scans only its private checkpoint root. It must be empty.
    checkpoint_dir = Path(common.CHECKPOINTS)
    stale = list(checkpoint_dir.glob("*.pt"))
    if stale: raise RuntimeError(f"private fresh-training checkpoint root is not empty: {stale}")
    bindings = {label: str(Path(module.RUN_DIR).resolve())
                for label, module in output_modules.items()}
    expected = str(raw)
    if any(value != expected for value in bindings.values()):
        raise RuntimeError(f"not every live output module is bound to R53: {bindings}")
    save = runtime["trainer"].save_checkpoint
    save_cells = dict(zip(save.__code__.co_freevars, save.__closure__ or (), strict=True))
    journal_cell = save_cells.get("checkpoint_journal")
    if (journal_cell is None or Path(journal_cell.cell_contents).resolve()
            != Path(base.CHECKPOINT_JOURNAL_PATH).resolve()):
        raise RuntimeError("actual save_checkpoint closure is not routed to the R53 journal")
    bc_module = getattr(base, "_R45_MODULE", None)
    bc_entry = getattr(bc_module, "_bc_entry_for_seed", None)
    if not callable(bc_entry): raise RuntimeError("live R45 BC_POST model-only identity accessor is missing")
    bc_roles = {}
    expected_bc = read_json(RUN_ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json")["bc_models_by_seed"]
    for seed in SEEDS:
        role = bc_entry(seed); expected_role = expected_bc[str(seed)]
        if (Path(role.get("checkpoint_path", "")).resolve() != Path(expected_role["checkpoint_path"]).resolve()
                or role.get("checkpoint_sha256") != expected_role["checkpoint_sha256"]
                or role.get("parameter_sha256") != expected_role["parameter_sha256"]):
            raise RuntimeError(f"live R45 BC_POST role identity mismatch for seed {seed}")
        bc_roles[str(seed)] = {"path": role["checkpoint_path"],
                               "checkpoint_sha256": role["checkpoint_sha256"],
                               "parameter_sha256": role["parameter_sha256"]}
    for path in (common.OUTPUT_DIR, common.LOGS, common.ROWS, common.REJECTED, common.CHECKPOINTS,
                 r40.ROUND_DIR, run.RUN_DIR, base.CHECKPOINT_JOURNAL_PATH,
                 raw / "launch_claims", raw / "launch_slots"):
        if Path(path).resolve() != raw:
            require_private_output_root(raw, [Path(path)])
    return {"bindings": bindings, "common_output": str(common.OUTPUT_DIR),
            "common_rows": str(common.ROWS), "common_logs": str(common.LOGS),
            "claim_root": str(r40.ROUND_DIR), "booking_root": str(run.RUN_DIR),
            "checkpoint_root": str(common.CHECKPOINTS),
            "checkpoint_journal": str(base.CHECKPOINT_JOURNAL_PATH),
            "launch_claims_root": str(raw / "launch_claims"),
            "launch_slots_root": str(raw / "launch_slots"),
            "bc_post_roles": bc_roles,
            "live_object_ids": {"trainer": id(runtime["trainer"]),
                                "r40_train_trainer": id(r40_train.trainer),
                                "runner": id(runtime["runner"]),
                                "r40_run": id(runtime["r40_run"]),
                                "r42_run": id(run), "r44_runtime": id(r44_runtime)},
            "r40_trainer_is_runtime_trainer": r40_train.trainer is runtime["trainer"],
            "ppo_update_wrapped": runtime["trainer"].ppo_update is not None,
            "checkpoint_version": TRAINING_VERSION,
            "fresh_rng_contract": "registered R44 ep0 Python/NumPy/Torch state is restored by frozen R42 train_seed after BC-only controller construction",
            "registered_ep0_rng_route": rng_route_identity,
            "registered_ep0_rng_route_identity": rng_route_identity,
            "route_undo": undo_update}


def execute_authorized(auth_path: Path = AUTH_PATH) -> dict[str, Any]:
    """After authorization and source preflight, run counted first-four gates before continuation."""
    plan = plan_from_frozen_sources()
    plan_bytes_sha = canonical_sha(plan)
    receipt = read_json(auth_path)  # Missing file is a hard pre-runtime stop.
    authorize_receipt(receipt, plan_sha256=plan_bytes_sha)
    baseline = verify_baseline_reuse()
    # Verify code/source identity, independently of documentation commit HEAD.
    expected_trainer = "0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841"
    if sha256_file(CANON / "v2_trainer.py") != expected_trainer:
        raise RuntimeError("canonical trainer byte identity changed")
    RAW.mkdir(parents=True, exist_ok=False)
    plan_path = RAW / "ROUND53_FIXED_PLAN.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    base = load_base_runner()
    bind_runner_paths(base, RAW)
    install_private_configure_seams(base)
    # Configure the exact frozen runtime, but never call the monolithic R46 execute.
    source_guard = install_protected_dirty_file_guard(base)
    base._R45_MODULE = None; base._RUNTIME = None
    base._configure_runtime_module()
    runtime = base._RUNTIME
    if runtime is None: raise RuntimeError("R53 private runtime did not configure")
    claim_verifier = install_r53_claim_verifier(base, runtime)
    live = bind_live_runtime(base, runtime, plan)
    runtime["round53_protected_dirty_file_guard"] = source_guard
    module = runtime["module"]
    first4, trained = run_first_four_gate_then_continue(base, module, runtime)
    rows = []
    run = runtime["r42_run"]
    for seed in SEEDS:
        controller, identity = run.load_eval_model("NEW", seed)
        for mode in ("greedy", "sampled"):
            for scene in SCENES:
                rows.append(run.run_model_eval(arm="NEW", training_seed=seed, mode=mode,
                                               scenario=scene, controller=controller,
                                               identity=identity))
    # BC_POST model-only fresh factory; no optimizer/RNG state is restored.
    bc_entries = read_json(RUN_ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json")["bc_models_by_seed"]
    import torch
    for seed in SEEDS:
        controller = runtime["controller_module"].fresh_controller(seed)
        entry = bc_entries[str(seed)]
        if controller.parameter_summary().get("parameter_sha256") != entry["parameter_sha256"]:
            raise RuntimeError(f"BC_POST model-only initialization mismatch for seed {seed}")
        identity = {"arm": "BC", "training_seed": seed, "episodes_done": 0,
                    "load_mode": "bc_model_only_four_modules", "optimizer_restored": False,
                    "rng_restored": False, "file_sha256": entry["checkpoint_sha256"],
                    "parameter_identity": {"parameter_sha256": entry["parameter_sha256"]}}
        for scene in SCENES:
            rows.append(run.run_model_eval(arm="BC", training_seed=seed, mode="sampled",
                                           scenario=scene, controller=controller,
                                           identity=identity))
    if len(rows) != 144: raise RuntimeError(f"ROUND53 eval closure {len(rows)}/144")
    claim_paths = sorted((RAW / "launch_claims").glob("*/*.json"))
    claims = [read_json(path) for path in claim_paths]
    for path, claim in zip(claim_paths, claims, strict=True):
        expected_slug = runtime["r40_common"]._slug(str(claim.get("logical_job", "")))
        expected_name = f"attempt_{int(claim.get('attempt', 0)):04d}.json"
        if path.parent.name != expected_slug or path.name != expected_name:
            raise RuntimeError(f"claim file path/core identity mismatch: {path}")
    core_closure = validate_claims(plan, claims, strict=True)
    normal_slots = list((RAW / "launch_slots" / "normal").glob("*.json"))
    reserve_slots = list((RAW / "launch_slots" / "reserve").glob("*.json"))
    normal_records = [read_json(p) for p in normal_slots]
    reserve_records = [read_json(p) for p in reserve_slots]
    for category, paths, records in (("normal", normal_slots, normal_records),
                                     ("reserve", reserve_slots, reserve_records)):
        for path, record in zip(paths, records, strict=True):
            if path.name != f"{int(record.get('slot', 0)):04d}.json":
                raise RuntimeError(f"slot file path/core identity mismatch: {path}")
    slot_closure = validate_slot_records(
        plan, claims, normal_records, reserve_records, strict=True)
    ledger = run.write_execution_ledger(strict=True)
    if ledger.get("status") == "deferred":
        raise RuntimeError("final derived ledger did not close; authoritative claims are preserved")
    result = {"kind": "round53-execution-result/1", "status": "complete",
              "task_id": TASK_ID, "runner_version": R53_RUNNER_VERSION,
              "training_version": TRAINING_VERSION, "training": trained,
              "first_four_and_gates": first4, "evaluation_rows": len(rows),
              "ledger": ledger, "claim_closure": core_closure,
              "slot_closure": slot_closure,
              "normal_slots": len(normal_slots), "reserve_slots": len(reserve_slots),
              "baseline_reuse": baseline,
              "live_bindings": {k: v for k, v in live.items() if k != "route_undo"},
              "protected_dirty_file_guard": source_guard,
              "r53_claim_verifier": claim_verifier}
    (RAW / "ROUND53_EXECUTION_RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def run_first_four_gate_then_continue(base: Any, module: Any,
                                      runtime: dict[str, Any]) -> tuple[Any, Any]:
    """Enter the plan's counted first-four gate; continuation is impossible unless it passes."""
    first_four = base._run_first_four_and_all_gates(module, runtime)
    continued = base._continue_fixed_training(module, runtime, first_four)
    return first_four, continued


if __name__ == "__main__":
    print(json.dumps(execute_authorized(), ensure_ascii=False, indent=2))
