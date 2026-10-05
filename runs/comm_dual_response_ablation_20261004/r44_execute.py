"""Frozen ROUND44 execution: fixed training, indexed evaluation, reconciliation, report."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from r44_eval_index import (SCENES, SEEDS, MODES, audit_fixed_matrix,
                            build_eval_index, verify_new_cell)
from r44_prepare import (RUN_DIR, ROOT, R43_DIR, TRAINING_VERSION, RUNNER_VERSION,
                         prepare, read, sha256, verify_r43_source_identity, write_new)
from r44_runtime import PRELAUNCH_PATH, configure, verify_frozen_source
from r44_resume_bridge import (
    IDENTITY_PATH as RESUME_IDENTITY_PATH, MECHANICAL_VERSION,
    RESUME_PROGRESS_PATH, STATE_BRIDGE_VERSION,
    verify_pre_resume_state, verify_resume_ready,
)


CODE_FILES = (
    "EXECUTION_BASIS.md", "r42_versions.py", "r42_one_shot.py",
    "r40_train_resume_adapter.py", "r43_call_counter_hooks.py",
    "r44_eval_index.py", "test_r44_eval_index.py", "r44_prepare.py",
    "r44_runtime.py", "r44_preflight.py", "r44_analysis.py",
    "r44_report.py", "r44_execute.py",
)


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary.write_text(body, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_new_or_match(path: Path, value: Any) -> None:
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"ROUND44 immutable receipt exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def write_prelaunch_receipt() -> dict[str, Any]:
    """Seal the exact committed implementation/config before the first environment claim."""
    frozen = read(RUN_DIR / "ROUND44_FROZEN_CONFIG.json")
    zero = read(RUN_DIR / "ROUND44_ZERO_ENV_GATES.json")
    test_receipt = read(RUN_DIR / "ROUND44_ZERO_ENV_TESTS.json")
    if frozen.get("status") != "prelaunch_frozen" or zero.get("status") != "passed":
        raise RuntimeError("ROUND44 freeze or zero-environment gate is missing")
    if test_receipt.get("status") != "passed" or test_receipt.get("environment_starts") != 0:
        raise RuntimeError("ROUND44 indexed-evaluation pure tests have not passed")
    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    if branch != "exp/comm-dual-response-ablation":
        raise RuntimeError(f"ROUND44 execution is on the wrong branch: {branch}")
    if _git("status", "--porcelain"):
        raise RuntimeError("ROUND44 prelaunch source tree must be clean after the freeze commit")
    if not subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                           "c91ccf91c6104b4bd27a8d8828c65dd3953be27d", head],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError("ROUND44 freeze commit does not descend from its execution basis")
    for folder in ("launch_claims", "rows", "checkpoints", "logs"):
        base = RUN_DIR / folder
        if base.exists() and list(base.rglob("*.json")) + list(base.rglob("*.pt")) \
                + list(base.rglob("*.jsonl")):
            raise RuntimeError(f"ROUND44 has prelaunch experiment artifacts in {folder}")
    source_hashes = {}
    for relative in CODE_FILES:
        path = RUN_DIR / relative
        if not path.is_file():
            raise RuntimeError(f"ROUND44 frozen source file is missing: {relative}")
        source_hashes[relative] = sha256(path)
        subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"HEAD:runs/{RUN_DIR.name}/{relative}"],
                       check=True, stdout=subprocess.DEVNULL)

    r43_source_identity = verify_r43_source_identity()
    r43_hashes = r43_source_identity["current_source_code_sha256"]
    copied = {name: {"sha256": source_hashes[name],
                     "source_sha256": r43_hashes[name]}
              for name in ("r42_one_shot.py", "r40_train_resume_adapter.py",
                           "r43_call_counter_hooks.py")}
    if any(value["sha256"] != value["source_sha256"] for value in copied.values()):
        raise RuntimeError("ROUND44 helper copy differs from the frozen R43 helper bytes")

    receipt = {
        "kind": "round44-prelaunch-identity/1", "status": "passed",
        "head": head, "branch": branch,
        "training_version": TRAINING_VERSION, "runner_version": RUNNER_VERSION,
        "eta_lambda": 0.5, "gae_lambda": 1.0, "gamma": 1.0,
        "source_code_sha256": source_hashes,
        "copied_r43_helpers": copied,
        "r43_frozen_source_code_sha256": r43_source_identity[
            "frozen_source_code_sha256"],
        "r43_final_delivery_source_code_sha256": r43_hashes,
        "r43_report_only_finalization_identity": r43_source_identity[
            "report_only_finalization_identity"],
        "frozen_config_sha256": sha256(RUN_DIR / "ROUND44_FROZEN_CONFIG.json"),
        "configuration_diff_sha256": sha256(RUN_DIR / "ROUND44_CONFIGURATION_DIFF.json"),
        "baseline_reuse_sha256": sha256(RUN_DIR / "ROUND44_BASELINE_REUSE.json"),
        "zero_environment_gate_sha256": sha256(RUN_DIR / "ROUND44_ZERO_ENV_GATES.json"),
        "zero_environment_test_sha256": sha256(RUN_DIR / "ROUND44_ZERO_ENV_TESTS.json"),
        "source_delivery_head": frozen["execution_identity"]["source_delivery_head"],
        "source_execution_head": frozen["execution_identity"]["source_execution_head"],
        "reused_round43_delivery_head": frozen["round44"]["r43_final_delivery_head"],
        "prelaunch_environment_constructor_calls": 0,
        "prelaunch_optimizer_step_calls": 0,
        "prelaunch_experimental_checkpoint_write_calls": 0,
        "quota": {"normal": 480, "reserve": 24, "hard_cap": 504},
        "fixed_plan": {"training_episodes": 384, "new_evaluations": 96,
                       "baseline_repair": 0},
    }
    if PRELAUNCH_PATH.exists():
        existing = read(PRELAUNCH_PATH)
        if existing != receipt:
            raise RuntimeError("ROUND44 prelaunch identity receipt already exists with different identity")
    else:
        _write_new_or_match(PRELAUNCH_PATH, receipt)
    return receipt


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _load_r43_batch0(seed: int) -> tuple[dict[str, Any], str]:
    curves = read(R43_DIR / "ROUND43_TRAINING_CURVES.json")
    batches = curves["by_seed"][str(seed)]["batches"]
    if len(batches) != 32 or int(batches[0]["batch"]) != 0:
        raise RuntimeError(f"R43 seed {seed} batch-zero source record is missing")
    item = batches[0]
    path = Path(item["record_path"])
    if not path.is_file() or sha256(path) != item["record_sha256"]:
        raise RuntimeError(f"R43 seed {seed} batch-zero source record hash mismatch")
    return read(path), item["record_sha256"]


def _row_science_projection(row: dict[str, Any]) -> dict[str, Any]:
    """Compare every saved row value except the two source-proven wall-clock readings.

    Round label/version/config identity and artifact paths are reconciled separately by
    :func:`_reconcile_round_row_metadata`; task, cost, event, policy, and all remaining
    collector timing fields stay in this projection.
    """
    projected = {key: value for key, value in row.items()
                 if key not in {"seconds", "config_hash", "label", "runner_version",
                                "training_version", "eta_lambda", "artefacts"}}
    summary = projected.get("collector_summary")
    timing = summary.get("timing") if isinstance(summary, dict) else None
    if not isinstance(timing, dict) or "decide_milliseconds_per_call" not in timing:
        raise RuntimeError("row is missing the registered collector wall-clock timing value")
    timing = dict(timing)
    timing.pop("decide_milliseconds_per_call")
    summary = dict(summary)
    summary["timing"] = timing
    projected["collector_summary"] = summary
    return projected


def _validate_artifact_path_pair(old_row: dict[str, Any], new_row: dict[str, Any],
                                 episode_index: int) -> dict[str, str]:
    old_artifacts = old_row.get("artefacts")
    new_artifacts = new_row.get("artefacts")
    if (not isinstance(old_artifacts, dict) or not isinstance(new_artifacts, dict)
            or set(old_artifacts) != set(new_artifacts) or not old_artifacts):
        raise RuntimeError(f"episode {episode_index}: artifact roles differ or are missing")
    resolved: dict[str, str] = {}
    for role in sorted(old_artifacts):
        old_path, new_path = Path(str(old_artifacts[role])).resolve(), Path(
            str(new_artifacts[role])).resolve()
        if not old_path.is_file() or not new_path.is_file():
            raise RuntimeError(f"episode {episode_index}: {role} artifact path is missing")
        try:
            old_relative = old_path.relative_to(R43_DIR.resolve()).as_posix()
            new_relative = new_path.relative_to(RUN_DIR.resolve()).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"episode {episode_index}: {role} artifact escaped its registered run directory"
            ) from exc
        if old_relative.replace("v2_r43_", "v2_r44_") != new_relative:
            raise RuntimeError(
                f"episode {episode_index}: {role} path is not the registered R43/R44 label pair"
            )
        resolved[role] = str(new_path)
    return resolved


def _reconcile_round_row_metadata(old_row: dict[str, Any], new_row: dict[str, Any],
                                  training_seed: int, episode_index: int,
                                  runtime: dict[str, Any]) -> dict[str, Any]:
    """Validate the registered R43/R44 identity and explain each permitted row difference."""
    r43_freeze = read(R43_DIR / "ROUND43_FROZEN_CONFIG.json")
    r44_freeze = read(RUN_DIR / "ROUND44_FROZEN_CONFIG.json")
    config_diff = read(RUN_DIR / "ROUND44_CONFIGURATION_DIFF.json")
    source_receipt_path = RUN_DIR / "runtime" / "ROUND44_SOURCE_CONFIG_RECONCILIATION_R2.json"
    if not source_receipt_path.is_file():
        raise RuntimeError("source-proven R2 config_hash reconciliation receipt is missing")
    source_receipt = read(source_receipt_path)
    if source_receipt.get("status") != "passed_source_and_config_only":
        raise RuntimeError("R2 runtime-source/config reconciliation did not pass")

    # These are the only row identity changes registered by the frozen ROUND44 single-variable diff.
    differences = config_diff.get("scientific_differences", [])
    if (config_diff.get("scientific_variable_count") != 1 or len(differences) != 1
            or differences[0].get("path") != "round42.objective.eta_lambda"
            or float(differences[0].get("from", float("nan"))) != 0.05
            or float(differences[0].get("to", float("nan"))) != 0.5):
        raise RuntimeError("ROUND44 row metadata no longer matches the one-variable eta freeze")
    old_version = str(r43_freeze["round42"]["training_version"])
    new_version = str(r44_freeze["round42"]["training_version"])
    old_runner = str(r43_freeze["round42"]["runner_version"])
    registered_runners = config_diff["execution_only_differences"]["runner_version"]
    new_runner = str(registered_runners["round44"])
    if (old_version != str(config_diff["source_training_version"])
            or new_version != TRAINING_VERSION
            or new_version != str(config_diff["new_training_version"])
            or old_runner != str(registered_runners["round43"])
            or new_runner != RUNNER_VERSION):
        raise RuntimeError("R43/R44 row version metadata differs from the registered freezes")

    expected_episode = int(old_row.get("episode_index", -1))
    if expected_episode != episode_index or int(new_row.get("episode_index", -2)) != episode_index:
        raise RuntimeError(f"episode {episode_index}: row episode identity differs")
    if old_row.get("label") != f"train_s{training_seed}_e{episode_index}_r43" \
            or new_row.get("label") != f"train_s{training_seed}_e{episode_index}_r44":
        raise RuntimeError(f"episode {episode_index}: row labels are not the registered run labels")
    if old_row.get("runner_version") != old_runner or new_row.get("runner_version") != new_runner:
        raise RuntimeError(f"episode {episode_index}: row runner versions are not registered")
    if old_row.get("training_version") != old_version or new_row.get("training_version") != new_version:
        raise RuntimeError(f"episode {episode_index}: row training versions are not registered")
    if "eta_lambda" in old_row or float(new_row.get("eta_lambda", float("nan"))) != 0.5:
        raise RuntimeError(f"episode {episode_index}: row eta fields do not encode the registered change")
    if (int(old_row.get("attempt", -1)) != 1
            or int(new_row.get("attempt", -1)) != 1
            or old_row.get("attempt") != new_row.get("attempt")):
        raise RuntimeError(f"episode {episode_index}: attempt identity differs")

    old_seconds, new_seconds = old_row.get("seconds"), new_row.get("seconds")
    old_timing = (old_row.get("collector_summary") or {}).get("timing")
    new_timing = (new_row.get("collector_summary") or {}).get("timing")
    old_decide_ms = old_timing.get("decide_milliseconds_per_call") if isinstance(old_timing, dict) else None
    new_decide_ms = new_timing.get("decide_milliseconds_per_call") if isinstance(new_timing, dict) else None
    for label, value in (("R43 seconds", old_seconds), ("R44 seconds", new_seconds),
                         ("R43 decide milliseconds", old_decide_ms),
                         ("R44 decide milliseconds", new_decide_ms)):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"episode {episode_index}: {label} is not a finite wall-clock value")

    resolved_artifacts = _validate_artifact_path_pair(old_row, new_row, episode_index)
    old_transcript = Path(old_row["artefacts"]["transcript"]).resolve()
    new_transcript = Path(new_row["artefacts"]["transcript"]).resolve()
    old_transcript_sha = sha256(old_transcript)
    new_transcript_sha = sha256(new_transcript)
    if old_transcript_sha != new_transcript_sha:
        raise RuntimeError(f"episode {episode_index}: R43/R44 transcript content differs")
    old_lifecycle = Path(old_row["artefacts"]["lifecycle"]).resolve()
    new_lifecycle = Path(new_row["artefacts"]["lifecycle"]).resolve()
    old_records, new_records = _read_jsonl(old_lifecycle), _read_jsonl(new_lifecycle)
    if (not old_records or not new_records
            or old_records[0].get("event_type") != "run_config"
            or new_records[0].get("event_type") != "run_config"):
        raise RuntimeError(f"episode {episode_index}: lifecycle run_config identity is missing")
    for records, label in ((old_records, "R43"), (new_records, "R44")):
        final = records[-1]
        if not (final.get("event_type") == "final" and final.get("status") == "run_finished"
                and str(final.get("target_id")) == "__run__"):
            raise RuntimeError(f"episode {episode_index}: {label} lifecycle log is incomplete")
    old_detail, new_detail = old_records[0].get("detail", {}), new_records[0].get("detail", {})
    old_config, new_config = old_detail.get("configuration"), new_detail.get("configuration")
    old_signature, new_signature = old_detail.get("run_signature"), new_detail.get("run_signature")
    if not all(isinstance(item, dict) for item in
               (old_config, new_config, old_signature, new_signature)):
        raise RuntimeError(f"episode {episode_index}: run configuration/signature is malformed")
    if old_detail.get("scenario") != new_detail.get("scenario"):
        raise RuntimeError(f"episode {episode_index}: saved environment scenario differs")
    config_differences = sorted(
        key for key in set(old_config) | set(new_config)
        if key not in old_config or key not in new_config or old_config[key] != new_config[key]
    )
    if config_differences != ["comm_transcript", "output"]:
        raise RuntimeError(
            f"episode {episode_index}: run configuration has unregistered differences: "
            f"{config_differences}"
        )
    if (Path(str(old_config.get("output"))).resolve() != old_lifecycle
            or Path(str(new_config.get("output"))).resolve() != new_lifecycle
            or Path(str(old_config.get("comm_transcript"))).resolve()
            != Path(old_row["artefacts"]["transcript"]).resolve()
            or Path(str(new_config.get("comm_transcript"))).resolve()
            != Path(new_row["artefacts"]["transcript"]).resolve()):
        raise RuntimeError(f"episode {episode_index}: run-config output path does not match its artifact")

    signature_module = importlib.import_module("integrated_sar.run_signature")
    source_identity = source_receipt["runtime_module_sources"].get("integrated_sar.run_signature")
    expected_signature_path = (Path(runtime["r42_run"].INPUT_ROOT).resolve()
                               / "integrated_sar" / "run_signature.py").resolve()
    actual_signature_path = Path(signature_module.__file__).resolve()
    actual_signature_sha = sha256(actual_signature_path)
    if (not isinstance(source_identity, dict)
            or actual_signature_path != expected_signature_path
            or str(actual_signature_path) != source_identity.get("file")
            or actual_signature_sha != source_identity.get("sha256")
            or source_identity.get("matches_frozen_source") is not True):
        raise RuntimeError("actual run_signature.config_hash source differs from its frozen provenance")
    old_hash = signature_module.config_hash(old_config)
    new_hash = signature_module.config_hash(new_config)
    if (old_hash != old_signature.get("config_hash")
            or new_hash != new_signature.get("config_hash")
            or old_hash != old_row.get("config_hash")
            or new_hash != new_row.get("config_hash")):
        raise RuntimeError(f"episode {episode_index}: config_hash does not match its own full config")
    signature_differences = sorted(
        key for key in set(old_signature) | set(new_signature)
        if key not in old_signature or key not in new_signature
        or old_signature[key] != new_signature[key]
    )
    if signature_differences != ["config_hash"]:
        raise RuntimeError(f"episode {episode_index}: run signature differs beyond config_hash")
    old_normalized, new_normalized = dict(old_config), dict(new_config)
    for config in (old_normalized, new_normalized):
        config["output"] = "<verified per-run lifecycle output>"
        config["comm_transcript"] = "<verified per-run transcript output>"
    if old_normalized != new_normalized:
        raise RuntimeError(f"episode {episode_index}: path-normalized run configs differ")
    old_normalized_hash = signature_module.config_hash(old_normalized)
    new_normalized_hash = signature_module.config_hash(new_normalized)
    if old_normalized_hash != new_normalized_hash:
        raise RuntimeError(f"episode {episode_index}: path-normalized config hashes differ")

    return {
        "registered_row_label_change": [old_row["label"], new_row["label"]],
        "registered_runner_version_change": [old_runner, new_runner],
        "registered_training_version_change": [old_version, new_version],
        "registered_eta_change": {"r43": 0.05, "r44": float(new_row["eta_lambda"])},
        "row_attempt_equal": True,
        "verified_artifact_paths": resolved_artifacts,
        "transcript_sha256": old_transcript_sha,
        "transcript_content_equal": True,
        "run_configuration_differing_fields": config_differences,
        "run_config_scenario_equal": True,
        "run_signature_differing_fields": signature_differences,
        "r43_config_hash_recomputed": old_hash,
        "r44_config_hash_recomputed": new_hash,
        "r43_config_hash_matches_row_and_signature": True,
        "r44_config_hash_matches_row_and_signature": True,
        "path_normalized_config_hash": old_normalized_hash,
        "path_normalized_config_equal": True,
        "wall_clock_measurements": {
            "r43_seconds": float(old_seconds), "r44_seconds": float(new_seconds),
            "r43_decide_milliseconds_per_call": float(old_decide_ms),
            "r44_decide_milliseconds_per_call": float(new_decide_ms),
        },
    }


def _compare_first_batch(seed: int, runtime: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    run = runtime["r42_run"]
    checkpoint_dir = runtime["common"].CHECKPOINTS
    commit_path = checkpoint_dir / f"r40_batch_commit_s{seed}_b0.json"
    commit = read(commit_path)
    r44_record_path = checkpoint_dir / str(commit["batch_record"])
    if not r44_record_path.is_file() or sha256(r44_record_path) != commit["batch_record_sha256"]:
        raise RuntimeError(f"ROUND44 seed {seed}: batch-zero record does not match its commit")
    new_record = read(r44_record_path)
    old_record, old_sha = _load_r43_batch0(seed)
    old_commit = read(R43_DIR / "checkpoints" / f"r40_batch_commit_s{seed}_b0.json")
    for field in ("batch", "training_seed", "episodes", "scenarios", "lambda_at_rollout",
                  "optimizer_state_sha256_before"):
        if new_record.get(field) != old_record.get(field):
            raise RuntimeError(f"seed {seed}: ROUND44 batch-zero {field} differs from R43")
    if (new_record["controller_parameters_before"]["parameter_sha256"]
            != old_record["controller_parameters_before"]["parameter_sha256"]
            or new_record["controller_parameters_after"]["parameter_sha256"]
            != old_record["controller_parameters_after"]["parameter_sha256"]):
        raise RuntimeError(f"seed {seed}: first-batch actor/optimizer parameter path differs from R43")
    if new_record.get("constraint") != old_record.get("constraint"):
        raise RuntimeError(f"seed {seed}: same-seed batch-zero paired constraint differs from R43")
    if new_record.get("update") != old_record.get("update"):
        raise RuntimeError(f"seed {seed}: first-batch PPO update differs from R43 under pre-batch lambda=1")
    old_postupdate = R43_DIR / "checkpoints" / old_commit["postupdate_checkpoint"]
    new_postupdate = checkpoint_dir / commit["postupdate_checkpoint"]
    if (not old_postupdate.is_file()
            or sha256(old_postupdate) != old_commit["postupdate_checkpoint_sha256"]
            or not new_postupdate.is_file()
            or sha256(new_postupdate) != commit["postupdate_checkpoint_sha256"]):
        raise RuntimeError(f"seed {seed}: postupdate checkpoint hash did not match its commit")
    old_optimizer_hash = runtime["r40_train"]._state_hash(
        run.torch_load_cpu(old_postupdate)["optimizer"])
    new_optimizer_hash = runtime["r40_train"]._state_hash(
        run.torch_load_cpu(new_postupdate)["optimizer"])
    if old_optimizer_hash != new_optimizer_hash:
        raise RuntimeError(f"seed {seed}: batch-zero optimizer state differs from R43")
    old_dual, new_dual = old_record["dual"], new_record["dual"]
    g_hat = float(old_dual["g_hat"])
    expected_old = max(0.0, float(old_dual["lambda_before"]) + 0.05 * g_hat)
    expected_new = max(0.0, float(new_dual["lambda_before"]) + 0.5 * g_hat)
    if (float(old_dual["eta"]) != 0.05 or float(new_dual["eta"]) != 0.5
            or float(new_dual["lambda_before"]) != 1.0
            or float(old_dual["lambda_before"]) != 1.0
            or abs(float(old_dual["lambda_next"]) - expected_old) > 1e-12
            or abs(float(new_dual["lambda_next"]) - expected_new) > 1e-12
            or float(new_dual["g_hat"]) != g_hat):
        raise RuntimeError(f"seed {seed}: first-batch dual update did not isolate eta")

    compared_episodes = []
    for old_artifact, new_artifact in zip(old_record["episode_artifacts"],
                                          new_record["episode_artifacts"]):
        if (int(old_artifact["episode_index"]) != int(new_artifact["episode_index"])
                or old_artifact["scenario_seed"] != new_artifact["scenario_seed"]):
            raise RuntimeError(f"seed {seed}: batch-zero episode order changed")
        old_row_path, new_row_path = Path(old_artifact["row_path"]), Path(new_artifact["row_path"])
        old_tape_path, new_tape_path = Path(old_artifact["tape_path"]), Path(new_artifact["tape_path"])
        if (sha256(old_row_path) != old_artifact["row_sha256"]
                or sha256(new_row_path) != new_artifact["row_sha256"]
                or sha256(old_tape_path) != old_artifact["tape_sha256"]
                or sha256(new_tape_path) != new_artifact["tape_sha256"]):
            raise RuntimeError(f"seed {seed}: batch-zero row/tape source hash mismatch")
        old_row, new_row = read(old_row_path), read(new_row_path)
        metadata_reconciliation = _reconcile_round_row_metadata(
            old_row, new_row, seed, int(old_artifact["episode_index"]), runtime)
        if _row_science_projection(old_row) != _row_science_projection(new_row):
            raise RuntimeError(
                f"seed {seed} episode {old_artifact['episode_index']}: pre-update rollout outcomes changed")
        if _read_jsonl(old_tape_path) != _read_jsonl(new_tape_path):
            raise RuntimeError(
                f"seed {seed} episode {old_artifact['episode_index']}: action/features/log-prob tape differs")
        compared_episodes.append({
            "episode_index": int(old_artifact["episode_index"]),
            "scenario_seed": int(old_artifact["scenario_seed"]),
            "old_row_sha256": old_artifact["row_sha256"],
            "new_row_sha256": new_artifact["row_sha256"],
            "old_tape_sha256": old_artifact["tape_sha256"],
            "new_tape_sha256": new_artifact["tape_sha256"],
            "metadata_reconciliation": metadata_reconciliation,
            "science_projection_equal": True, "tape_equal": True,
        })
    if len(compared_episodes) != 4:
        raise RuntimeError(f"seed {seed}: expected exactly four batch-zero episodes")
    result = {
        "kind": "round44-first-batch-vs-r43-comparison/2", "status": "passed",
        "projection_version": "round44-row-projection/2",
        "training_seed": seed,
        "r43_batch_record_sha256": old_sha,
        "r44_batch_record_sha256": sha256(r44_record_path),
        "actor_lambda_before": float(new_dual["lambda_before"]),
        "r43_eta_lambda": float(old_dual["eta"]),
        "r44_eta_lambda": float(new_dual["eta"]),
        "g_hat_equal": True,
        "r43_lambda_next": float(old_dual["lambda_next"]),
        "r44_lambda_next": float(new_dual["lambda_next"]),
        "preupdate_parameters_equal": True,
        "postupdate_parameters_equal": True,
        "postupdate_optimizer_state_equal": True,
        "postupdate_optimizer_state_sha256": new_optimizer_hash,
        "ppo_update_report_equal": True,
        "paired_constraint_equal": True,
        "action_features_rewards_likelihood_tapes_equal": True,
        "episodes": compared_episodes,
        "first_batch_gate": gate,
    }
    _write_new_or_match(RUN_DIR / f"ROUND44_FIRST_BATCH_COMPARISON_SEED{seed}.json", result)
    return result


def _stage_summary(report: dict[str, Any], stage: str) -> dict[str, Any]:
    return {
        "kind": "round44-training-stage-receipt/1", "stage": stage,
        "training_seed": int(report["training_seed"]),
        "episodes_done": int(report["episodes_done"]),
        "updates": int(report["updates"]),
        "optimizer_steps": int(report["optimizer_steps"]),
        "lambda": float(report["lambda"]),
        "training_version": TRAINING_VERSION,
        "eta_lambda": 0.5,
        "comm_action_semantics": "one-shot-send-skip/1",
        "milestone_paths": report.get("milestone_paths", {}),
        "rng_priming": report.get("rng_priming", {}),
    }


def _validate_claim_closure(ledger: dict[str, Any]) -> dict[str, Any]:
    counts = ledger["counts"]
    claims = ledger["claims"]
    complete = [claim for claim in claims if claim.get("status") == "complete"]
    if (int(counts["normal_used"]) != 480 or int(counts["reserve_used"]) > 24
            or int(counts["total_used"]) > 504
            or int(counts["completed_logical_jobs"]) != 480
            or len(complete) != 480 or int(counts["ppo_update_calls"]) != 96
            or not counts["optimizer_step_count_matches_committed_batches"]
            or not counts["ppo_update_count_matches_committed_batches"]):
        raise RuntimeError(f"ROUND44 final ledger does not close: {counts}")
    for claim in complete:
        if (claim.get("round") != 44
                or claim.get("training_version") != TRAINING_VERSION
                or claim.get("runner_version") != RUNNER_VERSION
                or int(claim.get("environment_constructor_calls", 0)) != 1
                or int(claim.get("run_invocation_calls", 0)) != 1
                or int(claim.get("joint_run_method_calls", 0)) != 1
                or int(claim.get("run_return_calls", 0)) != 1
                or int(claim.get("run_failure_calls", 0)) != 0
                or int(claim.get("completed_steps", 0)) != 1000):
            raise RuntimeError(f"successful claim identity/run counters failed: {claim}")
    return {"successful_logical_jobs": len(complete),
            "normal_used": int(counts["normal_used"]),
            "reserve_used": int(counts["reserve_used"]),
            "total_used": int(counts["total_used"]),
            "hard_cap": int(counts["hard_cap"]),
            "technical_failed_claims": int(counts["technical_failed_claims"]),
            "retry_claims": int(counts["retry_claims"])}


def _raw_artifact_manifest() -> dict[str, Any]:
    roots = ("rows", "logs", "checkpoints", "launch_claims", "launch_slots",
             "failures", "training_progress", "training_reports", "runtime")
    files = []
    for dirname in roots:
        root = RUN_DIR / dirname
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            files.append({
                "relative_path": path.relative_to(RUN_DIR).as_posix(),
                "path": str(path.resolve()), "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    return {"kind": "round44-raw-artifact-manifest/1",
            "raw_artifacts_remain_in_local_run_directory": True,
            "raw_files_count": len(files), "files": files}


def _lock_path() -> Path:
    return RUN_DIR / "runtime" / "ROUND44_EXECUTOR_LOCK.json"


def _acquire_lock(head: str) -> Path:
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "head": head, "started_at_unix": time.time()}
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _release_lock(path: Path) -> None:
    try:
        payload = read(path)
        if int(payload.get("pid", -1)) == os.getpid():
            path.unlink()
    except FileNotFoundError:
        pass


def run() -> dict[str, Any]:
    frozen = read(RUN_DIR / "ROUND44_FROZEN_CONFIG.json")
    current_head = verify_frozen_source(frozen)
    verify_resume_ready(frozen, current_head)
    lock = _acquire_lock(current_head)
    try:
        runtime = configure(frozen)
        import r44_analysis
        runtime["r44_analysis_module"] = r44_analysis
        run_module = runtime["r42_run"]
        common = runtime["common"]
        order = [int(value) for value in common.scenario_order()]
        if order != [int(value) for value in frozen["round42"]["training_order"]]:
            raise RuntimeError("ROUND44 actual scenario order differs from the committed freeze")
        original_progress_path = RUN_DIR / "runtime" / "ROUND44_EXECUTION_PROGRESS.json"
        resuming = RESUME_IDENTITY_PATH.is_file()
        progress_path = RESUME_PROGRESS_PATH if resuming else original_progress_path
        if progress_path.is_file():
            progress = read(progress_path)
            if progress.get("prelaunch_head") != current_head:
                raise RuntimeError("existing ROUND44 progress belongs to a different execution head")
            if progress.get("status") == "complete":
                raise RuntimeError("ROUND44 progress is already complete; inspect report-only outputs")
        else:
            if resuming:
                verify_pre_resume_state(frozen, current_head)
            progress = {
                "kind": "round44-execution-progress-resume-r2/2",
                "status": "running", "prelaunch_head": current_head,
                "resume_bridge": STATE_BRIDGE_VERSION,
                "mechanical_version": MECHANICAL_VERSION,
                "started_at_unix": time.time(),
                "training": {}, "evaluations_completed": 0, "completed_eval_keys": [],
            }
        run_module.write_execution_ledger(strict=True)
        _write_atomic(progress_path, progress)

        analysis_gate = runtime["r42_analysis"]
        report_dir = RUN_DIR / "training_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            first4_path = report_dir / f"ROUND44_SEED{seed}_FIRST4.json"
            final_path = report_dir / f"ROUND44_SEED{seed}_EP128.json"
            gate_path = RUN_DIR / f"ROUND44_FIRST_BATCH_GATE_SEED{seed}.json"
            commit0 = common.CHECKPOINTS / f"r40_batch_commit_s{seed}_b0.json"
            if not commit0.is_file():
                print(f"ROUND44 seed {seed}: run first 4 episodes for the real batch gate", flush=True)
                report4 = run_module.train_seed(seed, episodes=4, frozen=frozen)
                if int(report4["episodes_done"]) != 4 or int(report4["updates"]) != 1:
                    raise RuntimeError(f"seed {seed}: first four episodes did not close batch zero")
                _write_new_or_match(first4_path, _stage_summary(report4, "first_four"))
            if gate_path.is_file():
                gate = read(gate_path)
                if gate.get("status") != "passed" or int(gate.get("training_seed", -1)) != seed:
                    raise RuntimeError(f"seed {seed}: existing first-batch gate is not a pass")
            else:
                gate = analysis_gate.first_batch_gate(seed, runtime=runtime, frozen=frozen)
                if gate.get("status") != "passed":
                    raise RuntimeError(f"seed {seed}: actual first-batch correctness gate failed")
                comparison = _compare_first_batch(seed, runtime, gate)
                gate = {**gate, "round44_eta_lambda": 0.5,
                        "r43_batch0_comparison_sha256": sha256(
                            RUN_DIR / f"ROUND44_FIRST_BATCH_COMPARISON_SEED{seed}.json"),
                        "r43_batch0_comparison": comparison}
                _write_new_or_match(gate_path, gate)
            compare_path = RUN_DIR / f"ROUND44_FIRST_BATCH_COMPARISON_SEED{seed}.json"
            if not compare_path.is_file():
                comparison = _compare_first_batch(seed, runtime, gate)
            else:
                comparison = read(compare_path)
                if comparison.get("status") != "passed":
                    raise RuntimeError(f"seed {seed}: R43/R44 batch-zero comparison is not a pass")
            if not first4_path.is_file():
                commit = read(commit0)
                record = read(common.CHECKPOINTS / commit["batch_record"])
                _write_new_or_match(first4_path, {
                    "kind": "round44-training-stage-receipt/1", "stage": "first_four",
                    "training_seed": seed, "episodes_done": 4, "updates": 1,
                    "optimizer_steps": int(record["update"]["optimizer_steps"]),
                    "lambda": float(record["dual"]["lambda_next"]),
                    "eta_lambda": 0.5, "training_version": TRAINING_VERSION,
                    "checkpoint": str(common.CHECKPOINTS / f"v2_train_s{seed}.pt"),
                    "rng_priming": {"source": "registered R43 ep0 RNG after fresh controller",
                                    "loaded_model": False, "loaded_optimizer": False},
                })
            print(f"ROUND44 seed {seed}: actual first-batch and R43 equality gates passed", flush=True)

            final_checkpoint = common.CHECKPOINTS / f"v2_train_s{seed}_ep128.pt"
            if not final_checkpoint.is_file():
                report128 = run_module.train_seed(seed, episodes=128, frozen=frozen)
                if int(report128["episodes_done"]) != 128 or int(report128["updates"]) != 32:
                    raise RuntimeError(f"seed {seed}: training did not close at ep128/32 batches")
                _write_new_or_match(final_path, _stage_summary(report128, "ep128"))
            if not final_path.is_file():
                payload = run_module.torch_load_cpu(final_checkpoint)
                state = payload["training_state"]
                _write_new_or_match(final_path, {
                    "kind": "round44-training-stage-receipt/1", "stage": "ep128",
                    "training_seed": seed, "episodes_done": int(state["episodes_done"]),
                    "updates": int(state["updates"]),
                    "optimizer_steps": int(state["optimizer_steps"]),
                    "lambda": float(state["lambda"]), "eta_lambda": float(state["eta_lambda"]),
                    "training_version": TRAINING_VERSION,
                    "checkpoint": str(final_checkpoint),
                    "recovered_from_same_version_checkpoint": True,
                })
            rows, batches = runtime["r44_analysis_module"]._committed_training(runtime, seed)
            if len(rows) != 128 or len(batches) != 32:
                raise RuntimeError(f"seed {seed}: post-training committed rows/batches do not close")
            progress["training"][str(seed)] = {
                "status": "complete", "episodes": 128, "batches": 32,
                "first_batch_gate": str(gate_path),
                "first_batch_comparison": str(compare_path),
                "final_receipt": str(final_path),
                "ep128_parameter_sha256": batches[-1]["record"][
                    "controller_parameters_after"]["parameter_sha256"],
                "optimizer_steps": sum(int(item["record"]["update"]["optimizer_steps"])
                                        for item in batches),
            }
            progress["updated_at_unix"] = time.time()
            _write_atomic(progress_path, progress)
            run_module.write_execution_ledger(strict=True)
            print(f"ROUND44 seed {seed}: ep128 complete", flush=True)

        # Build the validated fixed schedule index exactly once before evaluation.
        checkpoint_parameter_sha = {}
        loaded_models = {}
        for seed in SEEDS:
            controller, identity = run_module.load_eval_model("NEW", seed)
            checkpoint_parameter_sha[seed] = identity["parameter_identity"]["parameter_sha256"]
            loaded_models[seed] = (controller, identity)
        index = build_eval_index(
            RUN_DIR, training_version=TRAINING_VERSION,
            reward_timing_version=runtime["r42_versions"].REWARD_TIMING_VERSION,
            runner_version=RUNNER_VERSION,
            checkpoint_parameter_sha256_by_seed=checkpoint_parameter_sha,
        )
        for key in ((seed, mode, scene) for seed in SEEDS for mode in MODES for scene in SCENES):
            seed, mode, scene = key
            row = index.get(seed, mode, scene)
            if row is None:
                print(f"ROUND44 eval NEW s{seed}/{mode} scene={scene}", flush=True)
                controller, identity = loaded_models[seed]
                row = run_module.run_model_eval(
                    arm="NEW", training_seed=seed, mode=mode, scenario=scene,
                    controller=controller, identity=identity)
                verified_row, entry = verify_new_cell(
                    RUN_DIR, seed=seed, mode=mode, scene=scene,
                    training_version=TRAINING_VERSION,
                    reward_timing_version=runtime["r42_versions"].REWARD_TIMING_VERSION,
                    runner_version=RUNNER_VERSION,
                    checkpoint_parameter_sha256=checkpoint_parameter_sha[seed],
                )
                if verified_row != row:
                    raise RuntimeError(f"new evaluation in-memory/durable row mismatch: {key}")
                index.add_verified(entry, verified_row)
                row = verified_row
            if (row.get("training_version") != TRAINING_VERSION
                    or row.get("runner_version") != RUNNER_VERSION
                    or row.get("checkpoint_parameter_identity", {}).get("parameter_sha256")
                    != checkpoint_parameter_sha[seed]):
                raise RuntimeError(f"ROUND44 evaluation cell identity mismatch: {key}")
            progress["evaluations_completed"] = index.completed_count
            progress["completed_eval_keys"] = index.completed_keys
            progress["updated_at_unix"] = time.time()
            _write_atomic(progress_path, progress)
        if index.completed_count != 96:
            raise RuntimeError(f"ROUND44 incremental eval index has {index.completed_count}/96 cells")

        eval_audit = audit_fixed_matrix(
            RUN_DIR, training_version=TRAINING_VERSION,
            reward_timing_version=runtime["r42_versions"].REWARD_TIMING_VERSION,
            runner_version=RUNNER_VERSION,
            checkpoint_parameter_sha256_by_seed=checkpoint_parameter_sha,
        )
        _write_new_or_match(RUN_DIR / "ROUND44_EVAL_MATRIX_AUDIT.json", eval_audit)
        ledger = run_module.write_execution_ledger(strict=True)
        _write_new_or_match(RUN_DIR / "ROUND44_EXECUTION_LEDGER_FINAL.json", ledger)
        claim_summary = _validate_claim_closure(ledger)

        analysis = r44_analysis.write_analysis(runtime=runtime, frozen=frozen,
                                               eval_audit=eval_audit)
        comparisons = [read(RUN_DIR / f"ROUND44_FIRST_BATCH_COMPARISON_SEED{s}.json")
                       for s in SEEDS]
        first_batch_comparison = {
            "kind": "round44-first-batch-comparison-all-seeds/1", "status": "passed",
            "by_seed": {str(item["training_seed"]): item for item in comparisons},
            "parameter_identical_seed_count": sum(bool(item["postupdate_parameters_equal"])
                                                   for item in comparisons),
            "actor_and_optimizer_updates_identical": all(
                item["ppo_update_report_equal"] and item["postupdate_parameters_equal"]
                and item["postupdate_optimizer_state_equal"]
                for item in comparisons),
            "only_batch_end_dual_response_differs": all(
                item["g_hat_equal"] and item["r44_eta_lambda"] == 0.5
                for item in comparisons),
        }
        _write_new_or_match(RUN_DIR / "ROUND44_FIRST_BATCH_COMPARISON.json",
                            first_batch_comparison)
        raw_manifest = _raw_artifact_manifest()
        _write_new_or_match(RUN_DIR / "ROUND44_RAW_ARTIFACT_MANIFEST.json", raw_manifest)
        zero_tests = read(RUN_DIR / "ROUND44_ZERO_ENV_TESTS.json")
        report = __import__("r44_report").write_report(
            result=analysis, ledger=ledger, progress=progress, eval_audit=eval_audit,
            execution_head=current_head, zero_tests=zero_tests,
            final_artifact_count=raw_manifest["raw_files_count"],
        )
        progress.update({
            "status": "complete", "finished_at_unix": time.time(),
            "evaluations_completed": 96, "final_ledger_counts": ledger["counts"],
            "claim_closure": claim_summary,
            "eval_matrix_audit_sha256": sha256(RUN_DIR / "ROUND44_EVAL_MATRIX_AUDIT.json"),
            "report_sha256": report["sha256"],
        })
        _write_atomic(progress_path, progress)
        status = {
            "kind": "round44-project-status/2", "round": 44,
            "status": "complete_local_delivery_pending_commits",
            "execution_matrix_status": "complete", "branch": _git("branch", "--show-current"),
            "scientific_execution_head": current_head,
            "training_version": TRAINING_VERSION, "runner_version": RUNNER_VERSION,
            "eta_lambda_old": 0.05, "eta_lambda_new": 0.5,
            "result": analysis["machine_verdict"]["scientific_result"],
            "normal_claims": ledger["counts"]["normal_used"],
            "reserve_claims": ledger["counts"]["reserve_used"],
            "total_claims": ledger["counts"]["total_used"], "hard_cap": 504,
            "environment_constructor_calls": ledger["counts"]["environment_constructor_calls"],
            "run_invocation_calls": ledger["counts"]["run_invocation_calls"],
            "completed_steps": ledger["counts"]["completed_steps"],
            "ppo_update_calls": ledger["counts"]["ppo_update_calls"],
            "optimizer_step_calls": ledger["counts"]["optimizer_step_calls"],
            "checkpoint_write_calls": ledger["counts"]["checkpoint_write_calls"],
            "training_episodes": 384, "new_final_evaluations": 96,
            "reused_old_r43_ep128_evaluations": 96,
            "independent_confirmation": False, "remote_push": False,
            "report_path": report["path"], "report_sha256": report["sha256"],
            "raw_artifact_count": raw_manifest["raw_files_count"],
        }
        status["stop_record_preserved"] = str(RUN_DIR / "ROUND44_STOP_REPORT.md")
        status["resume_bridge"] = "round44-exact-resume-state-bridge/1"
        status["mechanical_version"] = "round44-resume-comparator-and-restore/2"
        _write_new_or_match(RUN_DIR / "ROUND44_STATUS_COMPLETE_R2.json", status)
        return {"progress": progress, "ledger": ledger, "analysis": analysis,
                "report": report, "eval_audit": eval_audit, "status": status}
    finally:
        _release_lock(lock)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-prelaunch-receipt", action="store_true")
    args = parser.parse_args()
    if args.write_prelaunch_receipt:
        result = write_prelaunch_receipt()
        print(json.dumps({"status": result["status"], "head": result["head"],
                          "normal": 480, "reserve": 24, "hard_cap": 504},
                         ensure_ascii=False, indent=2), flush=True)
        return 0
    result = run()
    print(json.dumps({
        "status": result["progress"]["status"],
        "counts": result["ledger"]["counts"],
        "scientific_result": result["analysis"]["machine_verdict"]["scientific_result"],
        "report": result["report"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

