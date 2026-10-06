"""Zero-environment ROUND45 source/input and BC-plan freeze gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

import r45_bc


RUN_DIR = r45_bc.RUN_DIR
ROOT = r45_bc.ROOT
EXPECTED_BRANCH = "exp/comm-rule-warmstart-cost-learning"
EXPECTED_BASE = "cbd437afad6ddc3a7a68c8b7e10c0cd572a778dd"


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _source_hash(path: Path) -> str:
    return r45_bc.sha256_file(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_manifest_entry(root: Path, manifest: dict[str, Any], relative: str) -> dict[str, Any]:
    item = manifest.get(relative.replace("\\", "/"))
    path = root / Path(relative)
    if item is None or not path.is_file():
        raise RuntimeError(f"R44 manifest/source missing: {relative}")
    digest = _source_hash(path)
    if digest != item.get("sha256") or path.stat().st_size != int(item.get("size_bytes", -1)):
        raise RuntimeError(f"R44 manifest identity mismatch: {relative}")
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": digest}


def _verify_round44_primary_ep128_matrix(frozen: dict[str, Any]) -> dict[str, Any]:
    """Read-only validation of ROUND44 NEW ep128 rows, receipts, and source checkpoints."""
    r44_dir = r45_bc.R44_DIR
    audit_path = r44_dir / "ROUND44_EVAL_MATRIX_AUDIT.json"
    status_path = r44_dir / "ROUND44_STATUS_COMPLETE_EVAL_R1.json"
    checkpoint_index_path = r44_dir / "ROUND44_CHECKPOINT_INDEX.json"
    raw_manifest_path = r44_dir / "ROUND44_RAW_ARTIFACT_MANIFEST.json"
    persisted_audit = _read_json(audit_path)
    status = _read_json(status_path)
    checkpoint_index = _read_json(checkpoint_index_path)
    raw_manifest = _read_json(raw_manifest_path)
    raw_manifest_files = {
        str(item.get("relative_path", item.get("round_relative_path", ""))).replace("\\", "/"):
        item for item in raw_manifest.get("files", [])
    }

    training_version = frozen["round44"]["training_version"]
    runner_version = frozen["round44"]["runner_version"]
    reward_timing_version = "published-completion-event-at-relay-tick/2"
    if not training_version.endswith("/" + reward_timing_version):
        raise RuntimeError("ROUND44 frozen training version does not contain the registered reward timing")
    if (status.get("status") != "complete_local_delivery"
            or status.get("execution_matrix_status") != "complete"
            or status.get("training_version") != training_version
            or status.get("runner_version") != runner_version
            or status.get("new_final_evaluations") != 96
            or status.get("eval_matrix_unique_keys") != 96
            or status.get("eval_matrix_audit_sha256") != _source_hash(audit_path)
            or checkpoint_index.get("training_version") != training_version):
        raise RuntimeError("ROUND44 complete status, audit, and checkpoint index identities disagree")
    if (persisted_audit.get("kind") != "round44-independent-eval-matrix-audit/1"
            or persisted_audit.get("status") != "passed"
            or persisted_audit.get("cell_count") != 96):
        raise RuntimeError("ROUND44 persisted eval matrix audit is not a complete 96-cell pass")

    # Execute the frozen R44 disk auditor against the actual run directory. This re-reads
    # rows, claims, and artifacts only; it does not construct a JointRun or invoke simulation.
    source = r44_dir / "r44_eval_index.py"
    spec = importlib.util.spec_from_file_location("r45_readonly_r44_eval_index", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the registered ROUND44 eval-index auditor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    expected_checkpoints: dict[int, str] = {}
    checkpoint_results: dict[str, Any] = {}
    v2_controller = r45_bc.configure_source_imports()
    ambient = r45_bc._rng_snapshot()
    try:
        for seed in r45_bc.SEEDS:
            indexed = checkpoint_index["by_seed"][str(seed)]["128"]
            path = Path(indexed["path"]).resolve()
            expected_path = (r44_dir / "checkpoints" / f"v2_train_s{seed}_ep128.pt").resolve()
            if path != expected_path:
                raise RuntimeError(f"ROUND44 seed {seed} ep128 path is not the fixed checkpoint")
            raw_identity = _verify_manifest_entry(
                r44_dir, raw_manifest_files,
                f"checkpoints/v2_train_s{seed}_ep128.pt")
            if (not path.is_file()
                    or _source_hash(path) != indexed.get("file_sha256")
                    or path.stat().st_size != int(indexed.get("file_size_bytes", -1))
                    or raw_identity["sha256"] != indexed.get("file_sha256")
                    or raw_identity["size_bytes"] != int(indexed.get("file_size_bytes", -1))):
                raise RuntimeError(f"ROUND44 seed {seed} ep128 checkpoint bytes differ from index/manifest")

            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload.get("training_state", {})
            expected_state = {
                "training_seed": seed, "episodes_done": 128, "updates": 32,
                "training_version": training_version,
                "reward_timing_version": reward_timing_version,
                "eta_lambda": float(frozen["round44"]["eta_lambda"]),
                "gamma": float(frozen["round44"]["gamma"]),
                "gae_lambda": float(frozen["round44"]["gae_lambda"]),
            }
            if (payload.get("training_version") != training_version
                    or payload.get("reward_timing_version") != reward_timing_version
                    or not isinstance(payload.get("controller"), dict)
                    or not isinstance(payload.get("optimizer"), dict)
                    or not isinstance(state, dict)
                    or any(state.get(key) != value for key, value in expected_state.items())):
                raise RuntimeError(f"ROUND44 seed {seed} ep128 checkpoint metadata mismatch")

            controller = v2_controller.fresh_controller(seed)
            state_dict = payload["controller"]
            controller.inner.trunk.load_state_dict(state_dict["trunk"])
            controller.inner.action_head.load_state_dict(state_dict["action_head"])
            controller.inner.value_head.load_state_dict(state_dict["value_task_head"])
            controller.cost_head.load_state_dict(state_dict["value_cost_head"])
            parameter_identity = controller.parameter_summary()
            module_hashes = {
                "trunk": r45_bc._module_hash(controller.inner.trunk),
                "action_head": r45_bc._module_hash(controller.inner.action_head),
                "task_value_head": r45_bc._module_hash(controller.inner.value_head),
                "cost_value_head": r45_bc._module_hash(controller.cost_head),
            }
            if (parameter_identity["parameter_sha256"] != indexed.get("parameter_sha256")
                    or parameter_identity["parameter_sha256"]
                    != checkpoint_index["parameter_chain"][str(seed)]["ep128_parameter_sha256"]
                    or module_hashes != indexed.get("module_parameter_sha256")):
                raise RuntimeError(f"ROUND44 seed {seed} ep128 parameter contents mismatch")
            checkpoint_results[str(seed)] = {
                "path": str(path), "file_sha256": indexed["file_sha256"],
                "parameter_sha256": parameter_identity["parameter_sha256"],
                "module_parameter_sha256": module_hashes,
                "episodes_done": int(state["episodes_done"]),
                "updates": int(state["updates"]),
                "optimizer_steps": int(state["optimizer_steps"]),
                "lambda": float(state["lambda"]),
                "training_version": state["training_version"],
                "read_only": True,
            }
            expected_checkpoints[seed] = parameter_identity["parameter_sha256"]
    finally:
        r45_bc._restore_rng(ambient)

    fresh_audit = module.audit_fixed_matrix(
        r44_dir,
        training_version=training_version,
        reward_timing_version=reward_timing_version,
        runner_version=runner_version,
        checkpoint_parameter_sha256_by_seed=expected_checkpoints,
    )
    if fresh_audit != persisted_audit:
        raise RuntimeError("ROUND44 persisted matrix differs from a fresh R44 strict-index audit")
    entries = fresh_audit["entries"]
    expected_retry = (11, "greedy", 1_301_000)
    retry_rows = [entry for entry in entries
                  if tuple(entry.get("eval_key", ())) == expected_retry]
    if len(retry_rows) != 1 or retry_rows[0].get("attempt") != 2:
        raise RuntimeError("ROUND44 s11/greedy/1301000 does not select its valid attempt 2")

    return {
        "kind": "round44-primary-new-ep128-eval-matrix-readonly-validation/1",
        "status": "passed", "cell_count": len(entries),
        "source": "ROUND44 NEW ep128, fixed final strict index",
        "training_version": training_version, "runner_version": runner_version,
        "reward_timing_version": reward_timing_version,
        "eval_matrix_audit_sha256": _source_hash(audit_path),
        "checkpoint_index_sha256": _source_hash(checkpoint_index_path),
        "fresh_audit_matches_persisted": True,
        "group_counts": fresh_audit["group_counts"],
        "selected_retry": {
            "eval_key": list(expected_retry), "attempt": 2,
            "row_sha256": retry_rows[0]["row_sha256"],
            "claim_sha256": retry_rows[0]["claim_sha256"],
        },
        "ep128_checkpoints_read_only": checkpoint_results,
        "entries": entries,
    }


def verify_r44_inputs() -> dict[str, Any]:
    frozen = _read_json(r45_bc.R44_FREEZE_PATH)
    prelaunch = _read_json(r45_bc.R44_DIR / "runtime" / "ROUND44_PRELAUNCH_IDENTITY.json")
    report_path = r45_bc.R44_DIR / "ROUND44_REPORT.md"
    report_hash = _source_hash(report_path)
    if report_hash != "2c078d055c73e7f270f74015be991491e0691d7593af32870e6eee895ab9912a":
        raise RuntimeError("R44 final report bytes differ from the verified continuation note")
    if (prelaunch.get("status") != "passed"
            or prelaunch.get("frozen_config_sha256") != _source_hash(r45_bc.R44_FREEZE_PATH)
            or prelaunch.get("baseline_reuse_sha256")
            != _source_hash(r45_bc.R44_DIR / "ROUND44_BASELINE_REUSE.json")):
        raise RuntimeError("R44 prelaunch receipt or baseline reuse identity differs")
    if frozen.get("status") != "prelaunch_frozen":
        raise RuntimeError("R44 science freeze is not the registered prelaunch freeze")

    raw_manifest_path = r45_bc.R44_DIR / "ROUND44_RAW_ARTIFACT_MANIFEST.json"
    raw_manifest = _read_json(raw_manifest_path)
    files = {}
    for item in raw_manifest.get("files", []):
        relative = str(item.get("relative_path", item.get("round_relative_path", ""))).replace(
            "\\", "/")
        if relative:
            files[relative] = item
    if len(files) != len(raw_manifest.get("files", [])):
        raise RuntimeError("R44 raw artifact manifest has duplicate/empty paths")
    reuse_audit_path = r45_bc.R44_DIR / "ROUND44_BASELINE_REUSE.json"
    reuse_audit = _read_json(reuse_audit_path)
    reuse_counts = reuse_audit.get("counts", {})
    for name, expected in (("immediate", 64), ("late_and_hold", 32),
                           ("old_r43_ep128_evaluations", 96)):
        if int(reuse_counts.get(name, -1)) != expected:
            raise RuntimeError(f"R44 baseline reuse count differs for {name}")
    verified_reuse_rows = {"immediate": 0, "fixed_rules": 0, "r43_ep128_eval": 0}
    for group, expected_kind in (("round42_immediate_references", "reference"),
                                 ("round42_fixed_rules", "rule"),
                                 ("r43_new_ep128_eval_rows_as_old", "eval")):
        for item in reuse_audit.get(group, []):
            path = Path(item["path"])
            if (not path.is_file() or _source_hash(path) != item["sha256"]
                    or path.stat().st_size != int(item["size_bytes"])):
                raise RuntimeError(f"R44 reused baseline artifact changed: {path}")
            row = _read_json(path)
            if (row.get("kind") != expected_kind
                    or int(row.get("scenario_seed", -1)) != int(item["scene"])
                    or int(row.get("steps", -1)) != 1000
                    or int(row.get("budget_bytes", -1)) != 29300
                    or row.get("comm_action_semantics") != "one-shot-send-skip/1"
                    or row.get("cost_unit") not in (None, "payload_attempt_bytes")):
                raise RuntimeError(f"R44 reused baseline row key/protocol mismatch: {path}")
            if group == "round42_immediate_references":
                if row.get("reference_arm") != "immediate":
                    raise RuntimeError(f"wrong R44 immediate reference arm: {path}")
                verified_reuse_rows["immediate"] += 1
            elif group == "round42_fixed_rules":
                identity = row.get("rule_identity", {})
                if (identity.get("arm") != item.get("arm")
                        or row.get("model_arm") != item.get("arm")):
                    raise RuntimeError(f"R44 fixed-rule identity mismatch: {path}")
                if item.get("arm") == "late_discovery_gate":
                    parameters = identity.get("built_parameters", {})
                    if float(parameters.get("cut_steps", float("nan"))) != 147.0:
                        raise RuntimeError(f"R44 late rule is not cut_steps=147: {path}")
                verified_reuse_rows["fixed_rules"] += 1
            else:
                checkpoint_identity = row.get("checkpoint_parameter_identity", {})
                if (int(row.get("training_seed", -1)) != int(item["seed"])
                        or row.get("eval_mode") != item["mode"]
                        or checkpoint_identity.get("parameter_sha256")
                        != item["parameter_sha256"]
                        or row.get("training_version") != reuse_audit["r43_training_version"]):
                    raise RuntimeError(f"R44 reused ep128 evaluation identity mismatch: {path}")
                verified_reuse_rows["r43_ep128_eval"] += 1
    if verified_reuse_rows != {"immediate": 64, "fixed_rules": 32, "r43_ep128_eval": 96}:
        raise RuntimeError(f"R44 reused baseline rows do not close: {verified_reuse_rows}")
    verified_old_checkpoints = {}
    for seed, item in reuse_audit.get("r43_new_ep128_checkpoints_as_old", {}).items():
        path = Path(item["path"])
        if (not path.is_file() or _source_hash(path) != item["file_sha256"]
                or path.stat().st_size != int(item["file_size_bytes"])):
            raise RuntimeError(f"R44 read-only ep128 checkpoint changed: {path}")
        model = frozen["round42"]["old_models"][str(seed)]
        if (item.get("parameter_sha256") != model.get("parameter_sha256")
                or item.get("training_version") != frozen["round44"]["old_training_version"]
                or int(item.get("episodes_done", -1)) != 128
                or int(item.get("updates", -1)) != 32
                or item.get("read_only") is not True):
            raise RuntimeError(f"R44 read-only ep128 checkpoint identity mismatch: seed={seed}")
        verified_old_checkpoints[str(seed)] = {
            "path": str(path.resolve()), "file_sha256": item["file_sha256"],
            "parameter_sha256": item["parameter_sha256"],
            "episodes_done": item["episodes_done"], "updates": item["updates"],
            "read_only": True,
        }
    if len(verified_old_checkpoints) != 3:
        raise RuntimeError("R44 read-only ep128 checkpoint set is incomplete")
    primary_r44_eval = _verify_round44_primary_ep128_matrix(frozen)

    # Recheck the versioned R43/R44 source chain and the model/ep0 identities. This is read-only;
    # it reuses the committed R44 baseline and does not start or replay any environment.
    source_identity = frozen["execution_identity"]
    verified_r43_source_code = {}
    for relative, expected in prelaunch.get("r43_final_delivery_source_code_sha256", {}).items():
        path = r45_bc.R43_DIR / relative
        actual = _source_hash(path)
        if actual != expected:
            raise RuntimeError(f"R44-reused ROUND43 source changed: {relative}")
        verified_r43_source_code[relative] = actual

    verified_r44_delivery_sources = {}
    r44_sources_changed_after_prelaunch = []
    delivery_tree = _git("rev-parse", EXPECTED_BASE)
    if delivery_tree != EXPECTED_BASE:
        raise RuntimeError("registered ROUND44 final delivery commit is unavailable")
    for relative, prelaunch_hash in prelaunch.get("source_code_sha256", {}).items():
        path = r45_bc.R44_DIR / relative
        actual = _source_hash(path)
        object_path = f"runs/{r45_bc.R44_DIR.name}/{relative}"
        committed = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{EXPECTED_BASE}:{object_path}"],
            check=True, capture_output=True).stdout
        committed_hash = hashlib.sha256(committed).hexdigest()
        if actual != committed_hash:
            raise RuntimeError(f"ROUND44 source differs from its final committed object: {relative}")
        verified_r44_delivery_sources[relative] = committed_hash
        if actual != prelaunch_hash:
            r44_sources_changed_after_prelaunch.append({
                "path": relative, "prelaunch_sha256": prelaunch_hash,
                "final_delivery_sha256": committed_hash,
                "binding_commit": EXPECTED_BASE,
            })

    verified_inherited_dependencies = {}
    for relative, expected in source_identity.get("inherited_dependency_sha256", {}).items():
        path = r45_bc.CANONICAL_ROOT / relative
        if not path.is_file():
            path = r45_bc.CANONICAL_ROOT / "runs" / "comm_cost_v2_rerun_20260928" / relative
        actual = _source_hash(path)
        if actual != expected:
            raise RuntimeError(f"R44 canonical dependency changed: {relative}")
        verified_inherited_dependencies[relative] = actual
    verified_round40_sources = {}
    for relative, expected in source_identity.get("round40_code_sha256", {}).items():
        path = r45_bc.ROUND40_DIR / relative
        actual = _source_hash(path)
        if actual != expected:
            raise RuntimeError(f"R44 ROUND40 code dependency changed: {relative}")
        verified_round40_sources[relative] = actual

    roots = {
        "canonical_read_only_input_root": r45_bc.CANONICAL_ROOT,
        "round40_delivery_root": r45_bc.ROUND40_DIR.parents[1],
        "round41_delivery_root": Path(source_identity["round41_worktree"]),
    }
    heads = {}
    for key, root in roots.items():
        actual = _git("rev-parse", "HEAD", cwd=root)
        expected_key = {
            "canonical_read_only_input_root": "canonical_read_only_input_head",
            "round40_delivery_root": "round40_delivery_head",
            "round41_delivery_root": "round41_delivery_head",
        }[key]
        if actual != source_identity[expected_key]:
            raise RuntimeError(f"R44 readonly dependency checkout moved: {key}={actual}")
        heads[key] = actual

    r44_manifest = files
    r44_ep0 = r45_bc._expected_ep0_identities()
    ep0_results = {}
    for seed in r45_bc.SEEDS:
        item = r44_ep0[seed]
        checkpoint = Path(item["registered_ep0_path"])
        if (not checkpoint.is_file()
                or _source_hash(checkpoint) != item["registered_ep0_file_sha256"]):
            raise RuntimeError(f"seed {seed}: registered R44 ep0 bytes changed")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("training_state", {})
        if (int(state.get("episodes_done", -1)) != 0
                or int(state.get("updates", -1)) != 0
                or int(state.get("optimizer_steps", -1)) != 0
                or float(state.get("lambda", float("nan"))) != 1.0
                or payload.get("controller") is None
                or not isinstance(payload.get("rng"), dict)
                or payload.get("optimizer", {}).get("state") != {}):
            raise RuntimeError(f"seed {seed}: ep0 source is not a read-only empty PPO state")
        ep0_results[str(seed)] = {
            "checkpoint_path": str(checkpoint.resolve()),
            "file_sha256": item["registered_ep0_file_sha256"],
            "parameter_sha256": item["registered_and_fresh_parameter_sha256"],
            "registered_rng_pickle_sha256": item["registered_rng_pickle_sha256"],
            "episodes_done": int(state["episodes_done"]), "updates": int(state["updates"]),
            "optimizer_steps": int(state["optimizer_steps"]), "lambda": float(state["lambda"]),
            "model_used_as_bc_initializer": False,
            "only_registered_rng_is_authorized_for_later_rollout": True,
        }
    return {
        "frozen_config_sha256": _source_hash(r45_bc.R44_FREEZE_PATH),
        "prelaunch_receipt_sha256": _source_hash(
            r45_bc.R44_DIR / "runtime" / "ROUND44_PRELAUNCH_IDENTITY.json"),
        "baseline_reuse_sha256": _source_hash(reuse_audit_path),
        "raw_manifest_sha256": _source_hash(raw_manifest_path),
        "raw_manifest_file_count": len(r44_manifest),
        "verified_reused_r43_source_file_count": len(verified_r43_source_code),
        "verified_reused_r43_source_sha256": verified_r43_source_code,
        "verified_r44_final_delivery_source_file_count": len(verified_r44_delivery_sources),
        "verified_r44_final_delivery_source_sha256": verified_r44_delivery_sources,
        "r44_sources_changed_after_prelaunch_but_bound_to_final_delivery":
            r44_sources_changed_after_prelaunch,
        "verified_inherited_dependency_file_count": len(verified_inherited_dependencies),
        "verified_round40_source_file_count": len(verified_round40_sources),
        "round44_report_sha256": report_hash,
        "readonly_dependency_heads": heads,
        "ep0_readonly_identities": ep0_results,
        "baseline_reuse_counts": reuse_counts,
        "baseline_reuse_files": verified_reuse_rows,
        "old_ep128_readonly_checkpoints": verified_old_checkpoints,
        "r44_primary_new_ep128_eval_matrix": primary_r44_eval,
        "r43_historical_old_eval_matrix": {
            "source": "R44 baseline-reuse archive of ROUND43 NEW ep128 rows",
            "cell_count": verified_reuse_rows["r43_ep128_eval"],
            "used_as_round45_primary_old_comparison": False,
            "baseline_reuse_sha256": _source_hash(reuse_audit_path),
        },
    }


def _actual_adam_defaults(controller: Any) -> dict[str, Any]:
    optimizer = r45_bc.make_bc_optimizer(controller)
    defaults = dict(optimizer.defaults)
    return {key: list(value) if isinstance(value, tuple) else value
            for key, value in sorted(defaults.items())}


def freeze() -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"ROUND45 branch changed: {branch}")
    if _git("merge-base", "--is-ancestor", EXPECTED_BASE, head) != "":
        raise RuntimeError("ROUND44 delivery head is not an ancestor")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked files are dirty before the ROUND45 BC freeze")
    if Path(sys.executable).resolve() != Path(r"D:\project_env\python.exe").resolve():
        raise RuntimeError(f"not running in the registered experiment interpreter: {sys.executable}")
    if torch.__version__ != "2.9.0+cu128":
        raise RuntimeError(f"Torch version changed: {torch.__version__}")

    records, audit, dataset_sha = r45_bc.load_verified_dataset()
    expected_ep0 = r45_bc._expected_ep0_identities()
    v2_controller = r45_bc.configure_source_imports()
    ambient = r45_bc._rng_snapshot()
    fresh = {}
    adam_defaults = None
    controller_sources = {}
    try:
        import sb_controller
        import train_pilot
        for module in (v2_controller, sb_controller, train_pilot):
            controller_sources[Path(module.__file__).name] = _source_hash(Path(module.__file__))
        inherited = _read_json(r45_bc.R44_FREEZE_PATH)["execution_identity"][
            "inherited_dependency_sha256"]
        for key, actual in controller_sources.items():
            expected_key = {
                "v2_controller.py": "v2_controller.py",
                "sb_controller.py": "runs/comm_cost_stage_b_20260927/sb_controller.py",
                "train_pilot.py": "runs/comm_budget_pilot_20260919/train_pilot.py",
            }[key]
            if actual != inherited[expected_key]:
                raise RuntimeError(f"canonical controller source hash changed: {key}")
        for seed in r45_bc.SEEDS:
            controller = v2_controller.fresh_controller(seed)
            actual = controller.parameter_summary()["parameter_sha256"]
            expected = expected_ep0[seed]["registered_and_fresh_parameter_sha256"]
            if actual != expected or actual != r45_bc.EXPECTED_EP0_PARAMETER_SHA256[seed]:
                raise RuntimeError(f"seed {seed}: fresh controller differs from R44 ep0")
            fresh[str(seed)] = {
                "fresh_parameter_sha256": actual,
                "registered_r44_ep0_sha256": expected,
                "match": True,
                "bc_model_source": "fresh_controller(seed); no checkpoint weights loaded",
            }
            if adam_defaults is None:
                adam_defaults = _actual_adam_defaults(controller)
    finally:
        r45_bc._restore_rng(ambient)

    synthetic = _read_json(RUN_DIR / "BC_SYNTHETIC_REGRESSION.json")
    if (synthetic.get("status") != "passed"
            or int(synthetic.get("synthetic_optimizer_steps", 65)) > 64):
        raise RuntimeError("BC actual-update synthetic regression is not a pass or exceeds cap")
    tie_regression = _read_json(RUN_DIR / "BC_TIE_BREAK_REGRESSION.json")
    if (tie_regression.get("status") != "passed"
            or tie_regression.get("diagnostic_action") != "SEND_NOW"
            or int(tie_regression.get("optimizer_steps", -1)) != 0
            or int(tie_regression.get("environment_constructed", -1)) != 0):
        raise RuntimeError("greedy tie-break diagnostic regression is not a zero-cost pass")
    teacher_regression = _read_json(RUN_DIR / "TEACHER_REGRESSION.json")
    if (teacher_regression.get("status") != "passed"
            or int(teacher_regression.get("tests_run", -1)) != 4
            or int(teacher_regression.get("optimizer_step_calls", -1)) != 0
            or int(teacher_regression.get("environment_constructed", -1)) != 0
            or teacher_regression.get("test_source_sha256")
            != _source_hash(RUN_DIR / "test_r45_teacher.py")):
        raise RuntimeError("teacher time-basis regression is stale or did not pass")
    configuration_diff = _read_json(RUN_DIR / "ROUND45_CONFIGURATION_DIFF.json")
    r44_frozen_config = _read_json(r45_bc.R44_FREEZE_PATH)
    if (configuration_diff.get("status") != "preregistered"
            or configuration_diff.get("training_version") != r45_bc.PPO_TRAINING_VERSION
            or configuration_diff.get("parent_training_version")
            != r44_frozen_config["round44"]["training_version"]
            or float(configuration_diff["inherited_without_change"]["gamma"])
            != float(r44_frozen_config["round44"]["gamma"])
            or float(configuration_diff["inherited_without_change"]["gae_lambda"])
            != float(r44_frozen_config["round44"]["gae_lambda"])
            or float(configuration_diff["inherited_without_change"]["eta_lambda"])
            != float(r44_frozen_config["round44"]["eta_lambda"])):
        raise RuntimeError("ROUND45 preregistered config diff changed an inherited value")
    input_identity = verify_r44_inputs()
    source_paths = (
        "EXECUTION_BASIS_FULL.md", "EXECUTION_BASIS.md", "ROUND45_CONTINUATION_NOTE.md",
        "BC_DATA_SPLIT_AND_SELECTION.md", "PRE_LABEL_SEARCH_SCOPE_NOTE.md",
        "BC_TRAINING_PLAN.md", "ROUND45_CONFIGURATION_DIFF.json", "BC_DATASET.jsonl",
        "BC_DATA_AUDIT_R2.json", "r45_data.py",
        "test_r45_teacher.py", "r45_bc.py",
        "test_r45_bc.py", "r45_preflight.py", "ROUND45_CURRENT_TASK.json",
        "BC_SYNTHETIC_REGRESSION.json", "BC_SYNTHETIC_REGRESSION_PRE_GREEDY_TIE_FIX.json",
        "BC_TIE_BREAK_REGRESSION.json", "TEACHER_REGRESSION.json",
    )
    source_hashes = {relative: _source_hash(RUN_DIR / relative) for relative in source_paths}
    if not audit.get("input_artifact_manifest", {}).get("training_tapes"):
        raise RuntimeError("BC audit does not contain the already-verified R44 tape identity set")

    result = {
        "kind": "round45-pre-bc-freeze/1", "status": "passed",
        "branch": branch, "committed_source_head": head,
        "execution_basis_sha256": _source_hash(RUN_DIR / "EXECUTION_BASIS_FULL.md"),
        "continuation_note_sha256": _source_hash(RUN_DIR / "ROUND45_CONTINUATION_NOTE.md"),
        "source_files_sha256": source_hashes,
        "python": {"executable": str(Path(sys.executable).resolve()),
                   "version": sys.version, "torch": torch.__version__,
                   "cuda_available": bool(torch.cuda.is_available()), "device": "cpu"},
        "bc": {
            "version": r45_bc.BC_VERSION,
            "downstream_training_version": r45_bc.PPO_TRAINING_VERSION,
            "seeds": list(r45_bc.SEEDS), "updates_per_seed": r45_bc.BC_UPDATES,
            "optimizer_steps_planned": 3 * r45_bc.BC_UPDATES,
            "synthetic_optimizer_steps_already_used": int(
                synthetic["synthetic_optimizer_steps"]),
            "dataset_sha256": dataset_sha,
            "dataset_audit_sha256": _source_hash(r45_bc.AUDIT_PATH),
            "fit_samples": sum(row["split"] == "fit" for row in records),
            "validation_samples": sum(row["split"] == "validation" for row in records),
            "inverse_frequency_weights": r45_bc.inverse_frequency_weights(records),
            "loss": "weighted cross_entropy(mean)", "global_grad_clip": 0.5,
            "adam_defaults": adam_defaults,
            "fresh_controller_identities": fresh,
            "controller_source_files_sha256": controller_sources,
            "critic_heads_excluded_from_bc_optimizer": True,
            "r44_bc_input_identity": input_identity,
            "primary_old_comparison_source": "ROUND44 NEW ep128 final strict-index matrix",
            "historical_r43_matrix_used_as_primary": False,
            "test_result_sha256": _source_hash(RUN_DIR / "BC_SYNTHETIC_REGRESSION.json"),
            "teacher_test_result_sha256": _source_hash(RUN_DIR / "TEACHER_REGRESSION.json"),
            "greedy_tie_break_result_sha256": _source_hash(
                RUN_DIR / "BC_TIE_BREAK_REGRESSION.json"),
            "teacher_test_source_sha256": source_hashes["test_r45_teacher.py"],
        },
        "environment_plan": {
            "bc_control": 48, "ppo_training": 384, "final_eval": 96,
            "normal_total": 528, "technical_reserve": 24, "hard_cap": 552,
            "claims_consumed": 0, "environments_constructed": 0,
        },
        "activity_before_freeze": {
            "real_bc_optimizer_steps": 0, "ppo_updates": 0,
            "optimizer_step_calls": 0, "environment_constructor_calls": 0,
            "run_invocation_calls": 0, "checkpoint_writes": 0,
        },
    }
    target = RUN_DIR / "ROUND45_PRE_BC_FREEZE.json"
    r45_bc._write_exclusive(target, result)
    return result


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2, allow_nan=False))
