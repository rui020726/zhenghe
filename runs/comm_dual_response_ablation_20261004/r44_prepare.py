"""Build and verify ROUND44's fixed single-variable freeze and read-only baselines."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = Path(__file__).resolve().parent
R43_DIR = ROOT / "runs" / "comm_one_shot_full_trace_ablation_20261004"
R42_DIR = ROOT / "runs" / "comm_one_shot_gate_resume_20261003_v2"
R43_FREEZE = R43_DIR / "ROUND43_FROZEN_CONFIG.json"
R43_MANIFEST = R43_DIR / "ROUND43_RAW_ARTIFACT_MANIFEST.json"
R43_BASELINE = R43_DIR / "ROUND43_BASELINE_REUSE.json"
R43_CHECKPOINT_INDEX = R43_DIR / "ROUND43_CHECKPOINT_INDEX.json"
R43_PAIRED = R43_DIR / "ROUND43_PAIRED_RESULTS.json"
R43_STATUS = R43_DIR / "ROUND43_STATUS.json"
R43_REPORT_CORRECTION = R43_DIR / "ROUND43_REPORT_FAILURE_CORRECTION.json"
R42_MANIFEST = R42_DIR / "ROUND42_RAW_ARTIFACT_MANIFEST.json"
R42_CHECKPOINT_INDEX = R42_DIR / "ROUND42_CHECKPOINT_INDEX.json"
R42_PAIRED = R42_DIR / "ROUND42_PAIRED_RESULTS.json"

TRAINING_VERSION = (
    "constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/"
    "dual-eta-lambda-0p5/1/published-completion-event-at-relay-tick/2"
)
RUNNER_VERSION = "counted-jointrun-proxy/1+eval-key-index/1"
ETA_OLD = 0.05
ETA_NEW = 0.5
SEEDS = (11, 22, 33)
MODES = ("greedy", "sampled")
SCENES = tuple(range(1_301_000, 1_301_016))
TRAINING_SOURCE_HEAD = "9f2b6b6411c50849ce1a9eedbaedffaf0aededa5"
R43_EXECUTION_HEAD = "7ed1a03c445fa31fe90fd70a224a97b8366fa182"
INITIAL_SOURCE_VERSION = (
    "constrained-cost-v2/one-shot-send-skip/1/"
    "published-completion-event-at-relay-tick/2"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"ROUND44 freeze artifact already exists and differs: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()


def verify_r43_source_identity() -> dict[str, Any]:
    """Reconcile the execution freeze with R43's separately versioned report finalizer."""
    frozen = read(R43_FREEZE)
    status = read(R43_STATUS)
    correction = read(R43_REPORT_CORRECTION)
    identity = frozen["execution_identity"]
    expected_sources = identity["source_code_sha256"]
    if (status.get("status") != "complete_local_delivery"
            or status.get("branch") != "exp/comm-one-shot-full-trace-ablation"
            or status.get("source_freeze_head") != R43_EXECUTION_HEAD
            or correction.get("status") != "report_only_recovery_complete"):
        raise RuntimeError("ROUND43 final delivery/report-only recovery identity is incomplete")

    delivery_head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "exp/comm-one-shot-full-trace-ablation"],
        check=True, capture_output=True, text=True).stdout.strip()
    if delivery_head != TRAINING_SOURCE_HEAD:
        raise RuntimeError(f"ROUND43 final delivery branch moved: {delivery_head}")

    report_path = "r43_report.py"
    report_code = correction.get("report_code_identity", {})
    if (report_code.get("head") != status.get("report_code_head")
            or report_code.get("renderer_sha256") != status.get("report_code_sha256")
            or report_code.get("version") != status.get("report_version")):
        raise RuntimeError("ROUND43 report-only finalizer identity disagrees across records")
    finalizer_sha = sha256(R43_DIR / report_path)
    if finalizer_sha != status.get("report_code_sha256"):
        raise RuntimeError("ROUND43 final report renderer differs from its registered hash")
    finalizer_repo_path = f"runs/{R43_DIR.name}/{report_path}"
    committed_finalizer = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{report_code['head']}:{finalizer_repo_path}"],
        check=True, capture_output=True).stdout
    committed_finalizer_sha = hashlib.sha256(committed_finalizer).hexdigest()
    if committed_finalizer_sha != finalizer_sha:
        raise RuntimeError("ROUND43 report renderer does not match its report-code commit")
    report_file_sha = sha256(R43_DIR / "ROUND43_REPORT.md")
    if report_file_sha != status.get("report_sha256"):
        raise RuntimeError("ROUND43 finalized report differs from its registered hash")
    if not subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
             str(report_code["head"]), delivery_head],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError("ROUND43 report-only code head is not in its final delivery history")

    finalization_counts = correction.get("finalization_accounting", {})
    report_only_zero_fields = (
        "environment_constructor_calls", "run_invocation_calls", "ppo_update_calls",
        "optimizer_step_calls", "checkpoint_write_calls", "checkpoint_deserializations",
        "forward_calls",
    )
    if any(int(finalization_counts.get(name, -1)) != 0 for name in report_only_zero_fields):
        raise RuntimeError("ROUND43 report finalization records nonzero experiment activity")

    current_sources = {}
    for relative, expected in expected_sources.items():
        path = R43_DIR / relative
        if not path.is_file():
            raise RuntimeError(f"ROUND43 frozen source file is missing: {relative}")
        actual = sha256(path)
        current_sources[relative] = actual
        if actual != expected and relative != report_path:
            raise RuntimeError(f"ROUND43 inherited execution source changed: {relative}")

    finalization_identity = {
        "kind": "round43-report-only-finalization-identity/1",
        "status": "verified",
        "path": report_path,
        "role": "report-only finalizer; not loaded by ROUND44 runner",
        "execution_freeze_head": R43_EXECUTION_HEAD,
        "execution_freeze_sha256": expected_sources[report_path],
        "report_code_head": report_code["head"],
        "report_code_sha256": finalizer_sha,
        "final_delivery_head": delivery_head,
        "final_report_sha256": report_file_sha,
        "finalization_activity": {name: 0 for name in report_only_zero_fields},
    }
    return {
        "frozen_source_code_sha256": expected_sources,
        "current_source_code_sha256": current_sources,
        "report_only_finalization_identity": finalization_identity,
    }


def canonical_sha(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _manifest_map(path: Path) -> dict[str, dict[str, Any]]:
    obj = read(path)
    entries = obj.get("files")
    if not isinstance(entries, list):
        raise RuntimeError(f"artifact manifest is malformed: {path}")
    result = {}
    for item in entries:
        relative = str(item.get("relative_path", item.get("round_relative_path", ""))).replace(
            "\\", "/")
        if not relative or relative in result:
            raise RuntimeError(f"manifest has empty/duplicate path: {path}: {relative}")
        result[relative] = item
    return result


def _check_manifest_file(root: Path, manifest: dict[str, dict[str, Any]], relative: str,
                         expected_sha: str | None = None) -> dict[str, Any]:
    item = manifest.get(relative.replace("\\", "/"))
    path = root / Path(relative)
    if item is None or not path.is_file():
        raise RuntimeError(f"manifest item or source file is missing: {relative}")
    actual = sha256(path)
    if actual != item.get("sha256") or path.stat().st_size != item.get("size_bytes"):
        raise RuntimeError(f"source file does not match raw manifest: {relative}")
    if expected_sha is not None and actual != expected_sha:
        raise RuntimeError(f"source file differs from reuse audit: {relative}")
    return {"relative_path": relative.replace("\\", "/"), "path": str(path.resolve()),
            "size_bytes": path.stat().st_size, "sha256": actual}


def _validate_row(row: dict[str, Any], *, kind: str, scene: int,
                  expected_arm: str | None = None) -> None:
    if row.get("kind") != kind or int(row.get("scenario_seed", -1)) != scene:
        raise RuntimeError(f"baseline row identity mismatch: {kind} scene={scene}")
    if (int(row.get("steps", -1)) != 1000
            or int(row.get("budget_bytes", -1)) != 29300
            or row.get("comm_action_semantics") != "one-shot-send-skip/1"):
        raise RuntimeError(f"baseline row protocol or horizon mismatch: {row.get('label')}")
    if expected_arm is not None and row.get("model_arm") != expected_arm:
        raise RuntimeError(f"baseline row arm mismatch: expected {expected_arm}")
    if row.get("cost_unit") not in (None, "payload_attempt_bytes"):
        raise RuntimeError(f"baseline cost unit mismatch: {row.get('label')}")


def verify_reuse() -> dict[str, Any]:
    r43_frozen = read(R43_FREEZE)
    r43_status = read(R43_STATUS)
    if (r43_status.get("status") != "complete_local_delivery"
            or r43_status.get("execution_matrix_status") != "complete"
            or r43_status.get("source_freeze_head") != R43_EXECUTION_HEAD
            or r43_status.get("branch") != "exp/comm-one-shot-full-trace-ablation"):
        raise RuntimeError("ROUND43 is not the registered complete fixed-plan delivery")
    r43_branch_head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "exp/comm-one-shot-full-trace-ablation"],
        check=True, capture_output=True, text=True).stdout.strip()
    if r43_branch_head != TRAINING_SOURCE_HEAD:
        raise RuntimeError(f"ROUND43 delivery branch moved from its registered head: {r43_branch_head}")
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                       R43_EXECUTION_HEAD, TRAINING_SOURCE_HEAD],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("ROUND43 complete delivery no longer descends from its execution head")
    r43_manifest = _manifest_map(R43_MANIFEST)
    r42_manifest = _manifest_map(R42_MANIFEST)
    reuse = read(R43_BASELINE)
    if reuse.get("status") != "passed":
        raise RuntimeError("ROUND43's prior reuse audit did not pass")
    if sha256(R42_MANIFEST) != reuse.get("raw_manifest_sha256"):
        raise RuntimeError("ROUND43 reuse audit's ROUND42 raw-manifest identity changed")
    if sha256(R42_PAIRED) != reuse.get("paired_results_sha256"):
        raise RuntimeError("ROUND43 reuse audit's ROUND42 paired-result identity changed")
    if sha256(R42_CHECKPOINT_INDEX) != reuse.get("checkpoint_index_sha256"):
        raise RuntimeError("ROUND43 reuse audit's ROUND42 checkpoint-index identity changed")

    r42_reuse = reuse["reused"]
    references = r42_reuse["immediate_references"]["rows"]
    rules = r42_reuse["fixed_rules"]["rows"]
    if len(references) != 64 or len(rules) != 32:
        raise RuntimeError("ROUND44 requires exactly the prior 64 references and 32 rule rows")
    reference_records = []
    reference_scenes = set()
    for item in references:
        relative = str(item["relative_path"]).replace("\\", "/")
        record = _check_manifest_file(R42_DIR, r42_manifest, relative, item["sha256"])
        row = read(Path(record["path"]))
        scene = int(row["scenario_seed"])
        _validate_row(row, kind="reference", scene=scene)
        if row.get("reference_arm") != "immediate" or scene in reference_scenes:
            raise RuntimeError(f"immediate baseline duplicated or mislabeled: {relative}")
        reference_scenes.add(scene)
        reference_records.append({**record, "scene": scene, "arm": "immediate"})
    if len(reference_scenes) != 64:
        raise RuntimeError("ROUND44 immediate-reference scene identities do not close")

    rule_records = []
    rule_keys = set()
    for item in rules:
        relative = str(item["relative_path"]).replace("\\", "/")
        record = _check_manifest_file(R42_DIR, r42_manifest, relative, item["sha256"])
        row = read(Path(record["path"]))
        arm = str(row.get("model_arm"))
        scene = int(row["scenario_seed"])
        if arm not in ("late_discovery_gate", "always_hold"):
            raise RuntimeError(f"unexpected rule arm in fixed reuse: {arm}")
        _validate_row(row, kind="rule", scene=scene, expected_arm=arm)
        key = (arm, scene)
        if key in rule_keys:
            raise RuntimeError(f"duplicate fixed-rule row: {key}")
        rule_keys.add(key)
        rule_records.append({**record, "scene": scene, "arm": arm})
    expected_rule_keys = {(arm, scene) for arm in ("late_discovery_gate", "always_hold")
                          for scene in SCENES}
    if rule_keys != expected_rule_keys:
        raise RuntimeError("fixed late/hold rule reuse does not match the 16-scene block")

    checkpoint_index = read(R43_CHECKPOINT_INDEX)
    r43_version = str(checkpoint_index["training_version"])
    if r43_version != str(r43_frozen["round42"]["training_version"]):
        raise RuntimeError("ROUND43 checkpoint index training version differs from its freeze")
    old_checkpoints = {}
    for seed in SEEDS:
        info = checkpoint_index["by_seed"][str(seed)]["128"]
        path = Path(info["path"])
        if not path.is_file() or sha256(path) != info["file_sha256"]:
            raise RuntimeError(f"R43 OLD ep128 checkpoint bytes mismatch for seed {seed}")
        if (int(info["training_state"]["episodes_done"]) != 128
                or int(info["training_state"]["updates"]) != 32
                or info["training_state"]["training_version"] != r43_version
                or info["read_only_verification"].get("optimizer_restored") is not False
                or info["read_only_verification"].get("rng_restored") is not False):
            raise RuntimeError(f"R43 OLD ep128 checkpoint metadata mismatch for seed {seed}")
        old_checkpoints[str(seed)] = {
            "path": str(path.resolve()), "file_sha256": info["file_sha256"],
            "file_size_bytes": int(info["file_size_bytes"]),
            "parameter_sha256": info["parameter_sha256"],
            "module_parameter_sha256": info["module_parameter_sha256"],
            "training_version": r43_version,
            "episodes_done": 128, "updates": 32,
            "optimizer_steps": info["training_state"]["optimizer_steps"],
            "lambda_dual": info["training_state"]["lambda"],
            "read_only": True,
        }

    old_eval_records = []
    old_eval_keys = set()
    for item in r43_manifest.values():
        relative = str(item["relative_path"]).replace("\\", "/")
        if not relative.startswith("rows/v2_r43_eval_new_") or not relative.endswith(".json"):
            continue
        record = _check_manifest_file(R43_DIR, r43_manifest, relative)
        row = read(Path(record["path"]))
        seed, mode, scene = (int(row["training_seed"]), str(row["eval_mode"]),
                             int(row["scenario_seed"]))
        if (seed not in SEEDS or mode not in MODES or scene not in SCENES
                or row.get("training_version") != r43_version
                or row.get("reward_timing_version") != r43_frozen["round42"]["reward_timing_version"]
                or row.get("comm_action_semantics") != "one-shot-send-skip/1"
                or row.get("runner_version") != r43_frozen["round43"]["runner_version"]
                or row.get("eval_block") != f"one_shot_{mode}"):
            raise RuntimeError(f"R43 OLD eval row identity/protocol mismatch: {relative}")
        _validate_row(row, kind="eval", scene=scene, expected_arm="NEW")
        key = (seed, mode, scene)
        if key in old_eval_keys:
            raise RuntimeError(f"duplicate R43 OLD evaluation cell: {key}")
        identity = row.get("checkpoint_parameter_identity", {})
        if identity.get("parameter_sha256") != old_checkpoints[str(seed)]["parameter_sha256"]:
            raise RuntimeError(f"R43 OLD eval does not point to its ep128 checkpoint: {key}")
        old_eval_keys.add(key)
        old_eval_records.append({**record, "seed": seed, "mode": mode, "scene": scene,
                                 "parameter_sha256": identity["parameter_sha256"]})
    expected_eval_keys = {(seed, mode, scene) for seed in SEEDS for mode in MODES
                          for scene in SCENES}
    if old_eval_keys != expected_eval_keys:
        raise RuntimeError(f"R43 OLD evaluation matrix incomplete: {len(old_eval_keys)}/96")

    analysis_spec = importlib.util.spec_from_file_location(
        "round44_readonly_r42_analysis", R43_DIR / "r42_analysis.py")
    if analysis_spec is None or analysis_spec.loader is None:
        raise RuntimeError("cannot load the frozen pure reward-timing validator")
    analysis_module = importlib.util.module_from_spec(analysis_spec)
    analysis_spec.loader.exec_module(analysis_module)
    timing_rows = [read(Path(item["path"])) for item in references + rules]
    timing_rows.extend(read(Path(item["path"])) for item in old_eval_records)
    timing_closure = [analysis_module._validate_reward_timing(row) for row in timing_rows]

    ep0_inputs = copy.deepcopy(r43_frozen["round43"]["registered_ep0_inputs"])
    initial_models = copy.deepcopy(r43_frozen["round42"]["new_initial_models"])
    init_records = {}
    for seed in SEEDS:
        item = ep0_inputs[str(seed)]
        path = Path(item["path"])
        if not path.is_file() or sha256(path) != item["file_sha256"]:
            raise RuntimeError(f"registered ep0 RNG source changed for seed {seed}")
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("training_state", {})
        optimizer = payload.get("optimizer", {})
        rng = payload.get("rng", {})
        initial = initial_models[str(seed)]
        if (payload.get("training_version") != INITIAL_SOURCE_VERSION
                or state.get("training_version") != INITIAL_SOURCE_VERSION
                or int(state.get("episodes_done", -1)) != 0
                or int(state.get("updates", -1)) != 0
                or int(state.get("optimizer_steps", -1)) != 0
                or float(state.get("lambda", float("nan"))) != 1.0
                or float(state.get("gamma", float("nan"))) != 1.0
                or float(state.get("gae_lambda", float("nan"))) != 0.95
                or int(state.get("training_seed", -1)) != seed
                or len(optimizer.get("state", {})) != 0
                or not {"python", "numpy", "torch"}.issubset(rng)
                or item.get("parameter_sha256") != initial.get("parameter_sha256")):
            raise RuntimeError(f"registered ep0 parameter/empty-Adam/RNG identity mismatch: seed {seed}")
        init_records[str(seed)] = {
            "path": str(path.resolve()), "file_sha256": item["file_sha256"],
            "parameter_sha256": initial["parameter_sha256"],
            "episodes_done": 0, "updates": 0, "optimizer_steps": 0,
            "lambda_dual": float(state["lambda"]), "optimizer_state_entries": 0,
            "rng_sources": sorted(rng), "rng_pickle_sha256": hashlib.sha256(
                pickle.dumps(rng, protocol=5)).hexdigest(),
            "loaded_model": False, "loaded_optimizer": False,
            "rng_restored_before_first_rollout": True,
        }

    return {
        "r43_training_version": r43_version,
        "r43_execution_head": R43_EXECUTION_HEAD,
        "r43_final_delivery_head": TRAINING_SOURCE_HEAD,
        "r43_raw_manifest_sha256": sha256(R43_MANIFEST),
        "r43_paired_results_sha256": sha256(R43_PAIRED),
        "r43_checkpoint_index_sha256": sha256(R43_CHECKPOINT_INDEX),
        "round42_reference_source_head": r43_frozen["execution_identity"]["source_delivery_head"],
        "round42_immediate_references": reference_records,
        "round42_fixed_rules": rule_records,
        "r43_new_ep128_eval_rows_as_old": sorted(
            old_eval_records, key=lambda x: (x["seed"], x["mode"], x["scene"])),
        "r43_new_ep128_checkpoints_as_old": old_checkpoints,
        "registered_ep0_rng_inputs": ep0_inputs,
        "registered_ep0_initial_models": initial_models,
        "initialization_gate": init_records,
        "counts": {"immediate": 64, "late_and_hold": 32, "old_r43_ep128_evaluations": 96,
                   "old_r43_ep128_checkpoints": 3, "ep0_identity_inputs": 3},
        "verification": {
            "r43_reuse_audit": "passed",
            "round42_raw_manifest_join": "passed",
            "r43_raw_manifest_join": "passed",
            "old_source_is_r43_new_ep128": True,
            "read_only_checkpoints": True,
            "no_recreated_baselines": True,
            "reward_timing_cost_and_task_event_closure": {
                "status": "passed", "rows_checked": len(timing_closure),
                "credited_tasks": sum(item["credited_count"] for item in timing_closure),
            },
        },
    }


def build_freeze(reuse: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = read(R43_FREEZE)
    frozen = copy.deepcopy(source)
    previous_version = str(source["round42"]["training_version"])
    if source["round42"]["objective"].get("eta_lambda") != ETA_OLD:
        raise RuntimeError("R43 baseline eta_lambda is not the frozen 0.05 value")
    if source["round42"]["objective"].get("gae_lambda") != 1.0 \
            or source["round42"]["objective"].get("gamma") != 1.0 \
            or source["round42"]["objective"].get("lambda_init") != 1.0:
        raise RuntimeError("R43 shared GAE/gamma/dual initialization identity changed")

    frozen["round42"]["source_training_version"] = INITIAL_SOURCE_VERSION
    frozen["round42"]["old_training_version"] = previous_version
    frozen["round42"]["training_version"] = TRAINING_VERSION
    frozen["round42"]["objective"]["eta_lambda"] = ETA_NEW
    frozen["round42"]["old_models"] = {
        seed: {**reuse["r43_new_ep128_checkpoints_as_old"][seed],
               "round43_arm": "NEW", "round44_role": "OLD",
               "protocol_adaptation": "none; same one-shot SEND/SKIP protocol"}
        for seed in map(str, SEEDS)
    }
    frozen["round42"]["new_initial_models"] = copy.deepcopy(
        reuse["registered_ep0_initial_models"])
    frozen["round42"]["freeze_identity"]["kind"] = "round44-dual-response-eta-ablation-freeze/1"
    frozen["round42"]["freeze_identity"]["training_version"] = TRAINING_VERSION
    frozen["round42"]["freeze_identity"]["eta_lambda"] = ETA_NEW
    frozen["round42"]["freeze_identity"]["old_training_version"] = previous_version
    frozen["round42"]["freeze_identity"]["execution_runner_version"] = RUNNER_VERSION
    frozen["round42"]["freeze_identity"]["r43_delivery_head"] = TRAINING_SOURCE_HEAD
    frozen["round42"]["freeze_identity"].pop("freeze_sha256", None)

    plan = {
        "new_training": {"seeds": list(SEEDS), "episodes_each": 128, "total": 384},
        "new_final_evaluation": {"seeds": list(SEEDS), "modes": list(MODES),
                                 "scenes": list(SCENES), "total": 96},
        "reused_immediate_reference": {"cells": 64, "scheduled_new_launches": 0},
        "reused_late_and_always_hold_rules": {"cells": 32, "scheduled_new_launches": 0},
        "r43_new_ep128_reused_as_old": {"cells": 96, "scheduled_new_launches": 0},
        "baseline_repair": {"maximum": 0, "scheduled": 0},
        "normal_total": 480, "technical_reserve_max": 24, "hard_cap": 504,
        "new_streams": 0, "confirmation_scenes": 0,
    }
    frozen["round42"]["plan"] = plan
    frozen["round43"]["registered_ep0_inputs"] = copy.deepcopy(
        reuse["registered_ep0_rng_inputs"])
    frozen["round44"] = {
        "runner_version": RUNNER_VERSION,
        "training_version": TRAINING_VERSION,
        "source_training_version_for_ep0": INITIAL_SOURCE_VERSION,
        "old_training_version": previous_version,
        "eta_lambda_old": ETA_OLD,
        "eta_lambda": ETA_NEW,
        "lambda_init": 1.0,
        "gae_lambda": 1.0,
        "gamma": 1.0,
        "delta": 0.0,
        "b0_bytes": 10000.0,
        "normal_launch_plan": {"training": 384, "new_final_evaluation": 96, "total": 480},
        "baseline_repair_plan": {"maximum": 0, "scheduled": 0},
        "technical_reserve": 24,
        "hard_launch_cap": 504,
        "eval_index_version": "round44-eval-artifact-receipt/1",
        "completion_index_strategy": "one-time-validated-index-plus-verified-incremental-add/1",
        "first_batch_comparison": "exact R43 batch0 actor/model/optimizer equality; lambda_next uses eta=0.5",
        "r43_old_evaluation_matrix_count": 96,
        "r43_final_delivery_head": TRAINING_SOURCE_HEAD,
        "ep0_initialization_identity": copy.deepcopy(reuse["initialization_gate"]),
        "unmodified_round43_execution_source_head": R43_EXECUTION_HEAD,
    }
    expected_objective = copy.deepcopy(source["round42"]["objective"])
    expected_objective["eta_lambda"] = ETA_NEW
    if frozen["round42"]["objective"] != expected_objective:
        raise RuntimeError("ROUND44 scientific objective contains a change beyond eta_lambda")
    if frozen["round43"] != source["round43"]:
        raise RuntimeError("ROUND44 changed the inherited GAE/RNG/protocol execution contract")
    if frozen["round42"]["new_initial_models"] != source["round42"]["new_initial_models"]:
        raise RuntimeError("ROUND44 changed registered seed initialization identities")
    for key in ("training_order", "training_order_sha256", "training_scenes",
                "development_scenes", "ppo_hyperparameters", "steps_per_episode",
                "world_config_source", "candidate_order", "actor_feature_width",
                "row_filtering", "frozen_search_allocation_checkpoints",
                "comm_action_eligibility", "comm_action_semantics",
                "reward_timing_version", "episodes_per_seed", "episodes_per_update",
                "batches_per_seed"):
        if frozen["round42"].get(key) != source["round42"].get(key):
            raise RuntimeError(f"ROUND44 changed frozen scientific contract field {key}")
    frozen["round42"]["freeze_identity"]["freeze_sha256"] = canonical_sha(
        frozen["round42"]["freeze_identity"])

    diff = {
        "kind": "round44-scientific-configuration-diff/1",
        "source_round": 43,
        "source_training_version": previous_version,
        "new_training_version": TRAINING_VERSION,
        "scientific_variable_count": 1,
        "scientific_differences": [{
            "path": "round42.objective.eta_lambda",
            "from": ETA_OLD, "to": ETA_NEW,
            "interpretation": "dual response step size per completed four-episode PPO batch",
        }],
        "verified_unchanged": {
            "gae_lambda": 1.0, "gamma": 1.0, "lambda_init": 1.0,
            "delta": 0.0, "b0_bytes": 10000.0,
            "training_seeds": list(SEEDS), "episodes_per_seed": 128,
            "batches_per_seed": 32, "training_order_sha256": source[
                "round42"]["training_order_sha256"],
            "training_scenes": source["round42"]["training_scenes"],
            "development_scenes": list(SCENES),
            "action_semantics": "one-shot-send-skip/1",
            "actor_dual_timing": "pre-update lambda_dual is used throughout a batch",
        },
        "execution_only_differences": {
            "hard_launch_cap": {"round43": 600, "round44": 504},
            "runner_version": {"round43": "counted-jointrun-proxy/1",
                               "round44": RUNNER_VERSION},
            "evaluation_completion_index": "one-time scan then verified incremental updates",
            "old_arm": {"round43": "ROUND42 NEW ep128", "round44": "ROUND43 NEW ep128"},
        },
        "scientific_configuration_zero_drift": True,
        "verification_checks": {
            "objective_equals_r43_except_eta_lambda": True,
            "round43_gaE_rng_and_protocol_block_byte_semantics_unchanged": True,
            "registered_ep0_parameter_identities_unchanged": True,
            "training_order_scenes_ppo_world_observation_mask_unchanged": True,
        },
    }
    diff["canonical_sha256"] = canonical_sha(diff)
    frozen["round42"]["freeze_identity"]["configuration_diff_sha256"] = diff[
        "canonical_sha256"]
    # The embedded freeze hash includes the complete scientific diff identity.
    freeze_identity = dict(frozen["round42"]["freeze_identity"])
    freeze_identity.pop("freeze_sha256", None)
    frozen["round42"]["freeze_identity"]["freeze_sha256"] = canonical_sha(freeze_identity)
    frozen["kind"] = "round44-dual-response-frozen-config/1"
    frozen["status"] = "prelaunch_frozen"
    return frozen, diff


def prepare() -> dict[str, Any]:
    reuse = verify_reuse()
    frozen, diff = build_freeze(reuse)
    write_new(RUN_DIR / "ROUND44_BASELINE_REUSE.json", reuse)
    write_new(RUN_DIR / "ROUND44_CONFIGURATION_DIFF.json", diff)
    write_new(RUN_DIR / "ROUND44_FROZEN_CONFIG.json", frozen)
    return {"reuse": reuse, "frozen": frozen, "diff": diff}


if __name__ == "__main__":
    result = prepare()
    print(json.dumps({
        "status": "passed", "reused": result["reuse"]["counts"],
        "training_version": result["frozen"]["round42"]["training_version"],
        "eta_lambda": result["frozen"]["round44"]["eta_lambda"],
        "scientific_difference_count": len(result["diff"]["scientific_differences"]),
        "hard_cap": result["frozen"]["round44"]["hard_launch_cap"],
    }, ensure_ascii=False, indent=2))

