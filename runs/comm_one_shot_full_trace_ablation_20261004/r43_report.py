"""Build ROUND43 report, correction note, status, and source/raw inventories."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from r42_run import ROUND42_OUTPUT, ROUND40_ROOT, ROUND41_OUTPUT, ROUND42_ROOT, RUN_DIR


REPORT_VERSION = "round43-report-only-finalization/2"
PREFLIGHT_FAILURE_KIND = "round43-preflight-gate-failure/1"
PREFLIGHT_FAILURE_STATUS = "preserved_superseded_assertion_failure"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()
    temporary.replace(path)


def _fmt_bool(value: bool) -> str:
    return "是" if value else "否"


def _fmt_group(group: dict[str, Any]) -> str:
    return (f"按时任务 {group['on_time_total']}，计费 {group['spent_bytes_total']:,} B，"
            f"首报 SEND {group['send_now_choices_total']} / SKIP "
            f"{group['skip_episode_choices_total']}（自由选择 "
            f"{group['free_choices_total']}）")


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"preflight failure field {field!r} must be nonempty text")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"preflight failure field {field!r} must be a nonnegative integer")
    return value


def render_preflight_failure_correction(record: Any, *, source_path: str,
                                        source_sha256: str) -> dict[str, Any]:
    """Normalize only the two known ROUND43 preflight failure record layouts.

    The current persisted record stores ``failure`` and nests the preservation flag under
    ``post_failure_read_only_audit``.  The earlier report contract used top-level
    ``exception`` and ``preserved_artifacts_unchanged`` fields.  No unknown or incomplete
    record is guessed into a reportable failure.
    """
    if not isinstance(record, dict) or record.get("kind") != PREFLIGHT_FAILURE_KIND:
        raise ValueError("unsupported ROUND43 preflight failure record kind")
    if record.get("status") != PREFLIGHT_FAILURE_STATUS:
        raise ValueError("unsupported ROUND43 preflight failure record status")
    if "failure" in record and "exception" in record:
        raise ValueError("preflight failure record has ambiguous failure and exception fields")
    if "failure" in record:
        failure = record["failure"]
        if not isinstance(failure, dict):
            raise ValueError("preflight failure field 'failure' must be an object")
        if set(failure) - {"type", "message", "source", "cause"}:
            raise ValueError("preflight failure object has unknown fields")
        exception: Any = {
            "type": _nonempty_text(failure.get("type"), "failure.type"),
            "message": _nonempty_text(failure.get("message"), "failure.message"),
        }
        if "source" in failure:
            exception["source"] = _nonempty_text(failure["source"], "failure.source")
        if "cause" in failure:
            exception["cause"] = _nonempty_text(failure["cause"], "failure.cause")
        schema = "nested_failure_and_post_failure_audit"
    elif "exception" in record:
        legacy = record["exception"]
        if isinstance(legacy, str):
            exception = _nonempty_text(legacy, "exception")
        elif isinstance(legacy, dict):
            exception = {
                "type": _nonempty_text(legacy.get("type"), "exception.type"),
                "message": _nonempty_text(legacy.get("message"), "exception.message"),
            }
        else:
            raise ValueError("legacy preflight exception must be text or a typed object")
        schema = "legacy_exception_and_flat_preservation_flag"
    else:
        raise ValueError("preflight failure record has neither supported failure field")

    observed = record.get("observed_at_failure")
    if not isinstance(observed, dict):
        raise ValueError("preflight failure field 'observed_at_failure' must be an object")
    for field in ("normal_claim_delta", "environment_constructor_delta",
                  "real_optimizer_updates"):
        _nonnegative_int(observed.get(field), f"observed_at_failure.{field}")

    if "post_failure_read_only_audit" in record:
        audit = record["post_failure_read_only_audit"]
        if not isinstance(audit, dict):
            raise ValueError("post_failure_read_only_audit must be an object")
        preserved = audit.get("preserved_artifacts_unchanged")
        preservation_audit = audit
    elif "preserved_artifacts_unchanged" in record:
        preserved = record["preserved_artifacts_unchanged"]
        preservation_audit = {"preserved_artifacts_unchanged": preserved}
    else:
        raise ValueError("preflight failure record lacks its preservation audit")
    if type(preserved) is not bool:
        raise ValueError("preflight preservation audit must be a boolean")

    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("preflight failure source SHA256 is malformed")
    return {
        "path": source_path,
        "sha256": source_sha256,
        "schema_layout": schema,
        "exception": exception,
        "record_status": record["status"],
        "execution_claims_delta": observed["normal_claim_delta"],
        "environment_constructors_delta": observed["environment_constructor_delta"],
        "real_optimizer_updates": observed["real_optimizer_updates"],
        "observed_at_failure": observed,
        "preservation_audit": preservation_audit,
        "preserved_artifacts_unchanged": preserved,
        "classification": (
            "preflight assertion assumed zero historical Adam calls; fixed to assert zero delta"
        ),
    }


def _format_ids(items: Any) -> str:
    if not isinstance(items, list):
        raise ValueError("paired task-id differences must be lists")
    if not items:
        return "∅"
    rendered = []
    for item in items:
        if isinstance(item, dict):
            scene = item.get("scene")
            task_id = item.get("task_id")
            if type(scene) is not int or not isinstance(task_id, str) or not task_id:
                raise ValueError("paired task-id entry is malformed")
            rendered.append(f"{scene}:{task_id}")
        elif isinstance(item, str) and item:
            rendered.append(item)
        else:
            raise ValueError("paired task-id entry has an unsupported type")
    return ", ".join(rendered)


def _absolute_row(label: str, group: dict[str, Any]) -> str:
    n = group.get("n")
    if type(n) is not int or n != 16:
        raise ValueError(f"{label}: expected 16 fixed development scenes")
    y = _nonnegative_int(group.get("on_time_total"), f"{label}.on_time_total")
    cost = _nonnegative_int(group.get("spent_bytes_total"), f"{label}.spent_bytes_total")
    sends = _nonnegative_int(group.get("send_now_choices_total"), f"{label}.send_now_choices_total")
    skips = _nonnegative_int(group.get("skip_episode_choices_total"), f"{label}.skip_episode_choices_total")
    emissions = _nonnegative_int(group.get("first_report_emissions_total"), f"{label}.first_report_emissions_total")
    first_bytes = _nonnegative_int(group.get("first_report_bytes_total"), f"{label}.first_report_bytes_total")
    packet_lost = _nonnegative_int(group.get("packet_lost_total"), f"{label}.packet_lost_total")
    return (f"| {label} | {y} / {y / n:.4f} | {cost:,} / {cost / n:,.4f} B | "
            f"{emissions} / {first_bytes:,} B | {sends} / {skips} | {packet_lost} |")


def _paired_row(seed: int, mode: str, comparator: str,
                comparison: dict[str, Any]) -> str:
    signed_dy = comparison.get("on_time_total_delta")
    if type(signed_dy) is not int:
        raise ValueError("paired.on_time_total_delta must be an integer")
    left_cost = _nonnegative_int(comparison.get("spent_bytes_total_left"),
                                 "paired.spent_bytes_total_left")
    right_cost = _nonnegative_int(comparison.get("spent_bytes_total_right"),
                                  "paired.spent_bytes_total_right")
    dc = right_cost - left_cost
    scene_count = comparison.get("scene_count")
    if type(scene_count) is not int or scene_count != 16:
        raise ValueError(f"seed {seed} {mode} vs {comparator}: paired scene count is not 16")
    return (f"| {seed} | {mode} | {comparator} | {signed_dy:+d} / {signed_dy / 16:+.4f} | "
            f"{dc:+,} / {dc / 16:+,.4f} B | "
            f"{_format_ids(comparison.get('recovered_task_ids'))} | "
            f"{_format_ids(comparison.get('lost_task_ids'))} |")


def _result_tables(paired: dict[str, Any], curves: dict[str, Any]) -> tuple[str, str, str]:
    summaries = paired["group_summaries"]
    absolute = [
        "| 组 | 按时 Y 总数 / 每集均值 | 计费 C 总数 / 每集均值 | 首报发射 / 首报字节 | 自由 SEND / SKIP | 丢包 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("immediate_reference_development", "late_discovery_gate", "always_hold"):
        absolute.append(_absolute_row(name, summaries[name]))
    for seed in (11, 22, 33):
        for mode in ("greedy", "sampled"):
            for arm in ("OLD_round42_new_ep128", "NEW_round43_ep128"):
                key = f"s{seed}_{mode}_{arm}"
                absolute.append(_absolute_row(key, summaries[key]))

    paired_lines = [
        "| Seed | 模式 | 对照（NEW−对照） | ΔY 总数 / 每集 | ΔC 总数 / 每集 | NEW救回目标 | NEW丢失目标 |",
        "|---:|---|---|---:|---:|---|---|",
    ]
    for seed in (11, 22, 33):
        for mode in ("greedy", "sampled"):
            key = f"s{seed}_{mode}"
            comparisons = [
                ("OLD_round42_new_ep128", paired["primary_new_minus_old_by_seed_and_mode"][key]),
                ("immediate", paired["secondary_new_vs_immediate_by_seed_and_mode"][key]),
                ("late_discovery_gate", paired["new_vs_fixed_rules_by_seed_and_mode"][key]["late_discovery_gate"]),
                ("always_hold", paired["new_vs_fixed_rules_by_seed_and_mode"][key]["always_hold"]),
            ]
            paired_lines.extend(_paired_row(seed, mode, name, comparison)
                                for name, comparison in comparisons)

    lambda_lines = ["| Seed | Batch λ_next（按 batch 0→31；共32次） |", "|---:|---|"]
    for seed in (11, 22, 33):
        batches = curves["by_seed"][str(seed)]["batches"]
        if len(batches) != 32:
            raise ValueError(f"seed {seed}: expected 32 λ updates")
        values = []
        for expected_batch, batch in enumerate(batches):
            if batch.get("batch") != expected_batch:
                raise ValueError(f"seed {seed}: λ trajectory batch index mismatch")
            values.append(f"{float(batch['dual']['lambda_next']):.4f}")
        lambda_lines.append(f"| {seed} | `{' → '.join(values)}` |")
    return "\n".join(absolute), "\n".join(paired_lines), "\n".join(lambda_lines)


def _relative_to_run(path: Path) -> str:
    try:
        return path.resolve().relative_to(RUN_DIR.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact path escapes the ROUND43 run directory: {path}") from error


def _verify_manifested_file(path: Path, manifest_entries: dict[str, Any]) -> None:
    relative = _relative_to_run(path)
    entry = manifest_entries.get(relative)
    if not path.is_file() or not isinstance(entry, dict):
        raise RuntimeError(f"ROUND43 raw manifest lacks fixed evaluation artifact: {relative}")
    if (int(entry.get("size_bytes", -1)) != path.stat().st_size
            or entry.get("sha256") != _sha256(path)):
        raise RuntimeError(f"ROUND43 raw artifact changed after execution: {relative}")


def _audit_new_eval_matrix(*, frozen: dict[str, Any], paired: dict[str, Any],
                           checkpoint_index: dict[str, Any],
                           raw_manifest: dict[str, Any]) -> dict[str, Any]:
    if raw_manifest.get("kind") != "round43-raw-artifact-manifest/1":
        raise ValueError("ROUND43 raw artifact manifest schema is unsupported")
    manifest_files = raw_manifest.get("files")
    if not isinstance(manifest_files, list):
        raise ValueError("ROUND43 raw artifact manifest has no files list")
    manifest_entries: dict[str, Any] = {}
    for entry in manifest_files:
        if not isinstance(entry, dict) or not isinstance(entry.get("relative_path"), str):
            raise ValueError("ROUND43 raw artifact manifest contains a malformed entry")
        key = entry["relative_path"].replace("\\", "/")
        if key in manifest_entries:
            raise ValueError(f"ROUND43 raw artifact manifest contains duplicate path: {key}")
        manifest_entries[key] = entry
    if int(raw_manifest.get("raw_files_count", -1)) != len(manifest_entries):
        raise ValueError("ROUND43 raw artifact manifest file count does not close")

    expected_version = frozen["round42"]["training_version"]
    expected_runner = frozen["round43"]["runner_version"]
    expected_keys = {
        (seed, mode, scene)
        for seed in (11, 22, 33)
        for mode in ("greedy", "sampled")
        for scene in range(1_301_000, 1_301_016)
    }
    observed_keys: set[tuple[int, str, int]] = set()
    checked_files = 0
    for seed, mode, scene in sorted(expected_keys):
        logical = f"model_eval_new_s{seed}_{mode}_scene{scene}"
        claim_dir = RUN_DIR / "launch_claims" / logical
        claim_paths = sorted(claim_dir.glob("attempt_*.json"))
        if len(claim_paths) != 1:
            raise RuntimeError(f"{logical}: expected exactly one attempt claim, found {len(claim_paths)}")
        claim_path = claim_paths[0]
        claim = _read(claim_path)
        if (claim.get("logical_job") != logical or claim.get("status") != "complete"
                or claim.get("category") != "normal"
                or int(claim.get("attempt", -1)) != 1
                or claim.get("training_version") != expected_version
                or claim.get("runner_version") != expected_runner
                or int(claim.get("completed_steps", -1)) != 1000
                or int(claim.get("environment_constructor_calls", -1)) != 1
                or int(claim.get("joint_run_constructed_calls", -1)) != 1
                or int(claim.get("run_invocation_calls", -1)) != 1
                or int(claim.get("run_return_calls", -1)) != 1):
            raise RuntimeError(f"{logical}: claim is incomplete or has the wrong identity")
        _verify_manifested_file(claim_path, manifest_entries)
        checked_files += 1

        row_glob = f"v2_r43_eval_new_s{seed}_{mode}_s{scene}_a*.json"
        row_paths = sorted((RUN_DIR / "rows").glob(row_glob))
        if len(row_paths) != 1:
            raise RuntimeError(f"{logical}: expected exactly one row, found {len(row_paths)}")
        row_path = row_paths[0]
        row = _read(row_path)
        key = (int(row.get("training_seed", -1)), str(row.get("eval_mode")),
               int(row.get("scenario_seed", -1)))
        if key in observed_keys:
            raise RuntimeError(f"ROUND43 duplicate NEW evaluation key: {key}")
        observed_keys.add(key)
        expected_parameter = checkpoint_index["by_seed"][str(seed)]["128"][
            "parameter_sha256"]
        if (key != (seed, mode, scene)
                or row.get("kind") != "eval"
                or row.get("model_arm") != "NEW"
                or int(row.get("attempt", -1)) != 1
                or row.get("training_version") != expected_version
                or row.get("runner_version") != expected_runner
                or row.get("comm_action_semantics") != "one-shot-send-skip/1"
                or row.get("eval_block") != f"one_shot_{mode}"
                or int(row.get("steps", -1)) != 1000
                or row.get("checkpoint_parameter_identity", {}).get("parameter_sha256")
                != expected_parameter):
            raise RuntimeError(f"{logical}: row is incomplete or has the wrong identity")
        artefacts = row.get("artefacts")
        if not isinstance(artefacts, dict) or set(("row", "tape", "lifecycle", "transcript")) - set(artefacts):
            raise RuntimeError(f"{logical}: row lacks the four linked artifacts")
        if Path(artefacts["row"]).resolve() != row_path.resolve():
            raise RuntimeError(f"{logical}: linked row path differs from selected row")
        for artifact_name in ("row", "tape", "lifecycle", "transcript"):
            artifact_path = Path(artefacts[artifact_name])
            _verify_manifested_file(artifact_path, manifest_entries)
            checked_files += 1
        if int(claim.get("metadata", {}).get("training_seed", -1)) != seed \
                or claim.get("metadata", {}).get("mode") != mode \
                or int(claim.get("metadata", {}).get("scenario_seed", -1)) != scene:
            raise RuntimeError(f"{logical}: claim metadata does not match its row key")
        if claim_path.stem != "attempt_0001":
            raise RuntimeError(f"{logical}: completed evaluation is not attempt 1")

    if observed_keys != expected_keys or len(observed_keys) != 96:
        raise RuntimeError("ROUND43 NEW evaluation matrix does not contain the exact 96 fixed keys")
    if checked_files != 480:
        raise RuntimeError(f"ROUND43 evaluation identity audit checked {checked_files}, not 480 files")
    scope = paired.get("data_scope", {})
    expected_scope = {
        "immediate_references_reused": 64,
        "late_and_always_hold_rows_reused": 32,
        "old_round42_new_ep128_evaluations_reused": 96,
        "new_training_episodes": 384,
        "new_ep128_final_evaluations": 96,
    }
    if any(scope.get(key) != value for key, value in expected_scope.items()):
        raise RuntimeError(f"ROUND43 paired result data scope changed: {scope}")
    return {
        "status": "passed",
        "expected_keys": 96,
        "observed_keys": 96,
        "groups": {f"s{seed}_{mode}": 16 for seed in (11, 22, 33)
                   for mode in ("greedy", "sampled")},
        "complete_normal_claims": 96,
        "claim_row_and_linked_artifact_files_sha256_checked": checked_files,
        "training_version": expected_version,
        "runner_version": expected_runner,
        "checkpoint_parameter_identity_checked_against_ep128_index": True,
        "scope": expected_scope,
    }


def _validate_report_inputs(*, frozen: dict[str, Any], analysis: dict[str, Any],
                            ledger: dict[str, Any], progress: dict[str, Any],
                            raw_manifest: dict[str, Any]) -> dict[str, Any]:
    paired = analysis.get("paired_results")
    curves = analysis.get("training_curves")
    index = analysis.get("checkpoint_index")
    verdict = analysis.get("machine_verdict")
    if not all(isinstance(value, dict) for value in (paired, curves, index, verdict)):
        raise ValueError("report inputs are missing a persisted ROUND43 analysis artifact")
    if paired.get("kind") != "round43-paired-results/1" \
            or curves.get("kind") != "round43-training-curves/1" \
            or index.get("kind") != "round43-checkpoint-index/1" \
            or verdict.get("kind") != "round43-machine-verdict/1":
        raise ValueError("unsupported persisted ROUND43 analysis schema")
    if progress.get("kind") != "round43-execution-progress/1" \
            or progress.get("status") != "running" \
            or int(progress.get("evaluations_completed", -1)) != 96:
        raise RuntimeError("ROUND43 progress is not the expected post-matrix/pre-report state")
    source_head = progress.get("prelaunch_head")
    if not isinstance(source_head, str) or not re.fullmatch(r"[0-9a-f]{40}", source_head):
        raise RuntimeError("ROUND43 original execution freeze head is missing or malformed")
    if verdict.get("completion") != "complete":
        raise RuntimeError("ROUND43 machine verdict does not mark the matrix analysis complete")

    counts = ledger.get("counts")
    claims = ledger.get("claims")
    if not isinstance(counts, dict) or not isinstance(claims, list):
        raise ValueError("ROUND43 ledger is malformed")
    required_counts = {
        "normal_used": 480, "reserve_used": 0, "total_used": 480,
        "environment_constructor_calls": 480, "joint_run_constructed_calls": 480,
        "run_invocation_calls": 480, "joint_run_method_calls": 480,
        "run_return_calls": 480, "run_failure_calls": 0,
        "completed_steps": 480000, "completed_logical_jobs": 480,
        "technical_failed_claims": 0, "retry_claims": 0,
        "optimizer_step_calls": 1016, "optimizer_step_journal_entries": 1016,
        "ppo_update_calls": 96, "ppo_update_journal_entries": 96,
        "checkpoint_write_calls": 297, "checkpoint_write_journal_entries": 297,
        "hard_cap": 600,
    }
    if any(counts.get(key) != value for key, value in required_counts.items()):
        raise RuntimeError(f"ROUND43 final ledger does not match the completed matrix: {counts}")
    if len(claims) != 480 or any(claim.get("status") != "complete" for claim in claims):
        raise RuntimeError("ROUND43 ledger contains a missing or incomplete logical claim")
    if (not counts.get("optimizer_step_count_matches_committed_batches")
            or not counts.get("ppo_update_count_matches_committed_batches")):
        raise RuntimeError("ROUND43 call journals do not close against committed training batches")

    expected_training_jobs = {
        f"new_training_s{seed}_e{episode}"
        for seed in (11, 22, 33) for episode in range(128)
    }
    expected_eval_jobs = {
        f"model_eval_new_s{seed}_{mode}_scene{scene}"
        for seed in (11, 22, 33) for mode in ("greedy", "sampled")
        for scene in range(1_301_000, 1_301_016)
    }
    claim_keys = [claim.get("logical_job") for claim in claims]
    if len(set(claim_keys)) != 480 or set(claim_keys) != expected_training_jobs | expected_eval_jobs:
        raise RuntimeError("ROUND43 ledger logical jobs do not equal 384 training + 96 NEW evaluation")

    training_progress = progress.get("training")
    if not isinstance(training_progress, dict):
        raise ValueError("ROUND43 progress lacks seed training state")
    optimizer_steps = 0
    for seed in (11, 22, 33):
        item = training_progress.get(str(seed))
        batches = curves.get("by_seed", {}).get(str(seed), {}).get("batches")
        checkpoints = index.get("by_seed", {}).get(str(seed), {})
        if (not isinstance(item, dict) or item.get("status") != "complete"
                or item.get("episodes_done") != 128 or item.get("updates") != 32
                or item.get("training_version") != frozen["round42"]["training_version"]
                or not isinstance(batches, list) or len(batches) != 32
                or not isinstance(checkpoints, dict)):
            raise RuntimeError(f"ROUND43 seed {seed} training receipts are incomplete")
        optimizer_steps += sum(int(batch.get("update", {}).get("optimizer_steps", 0))
                               for batch in batches)
        chain = index.get("parameter_chain", {}).get(str(seed), {})
        if not chain.get("chain_contiguous") or len(chain.get("batches", [])) != 32:
            raise RuntimeError(f"ROUND43 seed {seed} parameter chain is incomplete")
        for episode in (0, 64, 128):
            checkpoint = checkpoints.get(str(episode))
            if not isinstance(checkpoint, dict):
                raise RuntimeError(f"ROUND43 seed {seed} ep{episode} checkpoint index is missing")
            training_state = checkpoint.get("training_state", {})
            expected_updates = episode // 4
            if (training_state.get("training_seed") != seed
                    or training_state.get("episodes_done") != episode
                    or training_state.get("updates") != expected_updates
                    or training_state.get("gae_lambda") != 1.0
                    or training_state.get("gamma") != 1.0
                    or training_state.get("training_version") != frozen["round42"]["training_version"]):
                raise RuntimeError(f"ROUND43 seed {seed} ep{episode} checkpoint state mismatch")
            checkpoint_path = Path(checkpoint["path"])
            if not checkpoint_path.is_file() or _sha256(checkpoint_path) != checkpoint.get("file_sha256"):
                raise RuntimeError(f"ROUND43 seed {seed} ep{episode} checkpoint hash mismatch")
        for suffix in ("first_batch_gate", "first4_receipt", "ep128_receipt"):
            receipt_path = Path(item.get(suffix, ""))
            if not receipt_path.is_file():
                raise RuntimeError(f"ROUND43 seed {seed} {suffix} file is missing")
            if suffix == "first_batch_gate":
                gate = _read(receipt_path)
                if gate.get("status") != "passed":
                    raise RuntimeError(f"ROUND43 seed {seed} first-batch gate did not pass")
                if _sha256(receipt_path) != item.get("first_batch_gate_sha256"):
                    raise RuntimeError(f"ROUND43 seed {seed} first-batch gate hash mismatch")
    if optimizer_steps != counts["optimizer_step_calls"]:
        raise RuntimeError("ROUND43 optimizer journal total does not equal recorded batch updates")

    eval_audit = _audit_new_eval_matrix(
        frozen=frozen, paired=paired, checkpoint_index=index, raw_manifest=raw_manifest)
    return {
        "execution_source_freeze_head": source_head,
        "normal_claims": len(claims),
        "new_training_episodes": 384,
        "new_evaluations": eval_audit,
        "seed_training": {
            str(seed): {
                "episodes": training_progress[str(seed)]["episodes_done"],
                "batches": training_progress[str(seed)]["updates"],
                "optimizer_steps": training_progress[str(seed)]["optimizer_steps"],
                "lambda_final": training_progress[str(seed)]["lambda"],
                "first_batch_gate": "passed",
            } for seed in (11, 22, 33)
        },
        "execution_ledger_counts": counts,
        "original_progress_status_preserved": progress["status"],
    }


def _snapshot_file_hashes(names: tuple[str, ...]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        path = RUN_DIR / name
        result[name] = _sha256(path) if path.is_file() else None
    return result


def write_final_report(*, frozen: dict[str, Any], analysis: dict[str, Any],
                       ledger: dict[str, Any], progress: dict[str, Any]) -> dict[str, Any]:
    paired = analysis["paired_results"]
    verdict = analysis["machine_verdict"]
    curves = analysis["training_curves"]
    index = analysis["checkpoint_index"]
    output_names = (
        "ROUND43_SOURCE_INVENTORY.json", "ROUND43_RAW_ARTIFACT_MANIFEST.json",
        "TRAINING_VERSION_MANIFEST.json", "ROUND43_RESUME_RECONCILIATION.json",
        "ROUND42_REPORT_CORRECTION.json", "ROUND43_REPORT.md", "ROUND43_STATUS.json",
        "ROUND43_REPORT_FAILURE_CORRECTION.json",
    )
    old_output_hashes = _snapshot_file_hashes(output_names)
    analysis_input_names = (
        "ROUND43_PAIRED_RESULTS.json", "ROUND43_TRAINING_CURVES.json",
        "ROUND43_CHECKPOINT_INDEX.json", "MACHINE_VERDICT.json",
    )
    analysis_input_hashes = _snapshot_file_hashes(analysis_input_names)
    if any(value is None for value in analysis_input_hashes.values()):
        raise RuntimeError("ROUND43 report-only finalization lacks persisted analysis inputs")
    ledger_path = RUN_DIR / "runtime" / "ROUND43_EXECUTION_LEDGER.json"
    progress_path = RUN_DIR / "runtime" / "ROUND43_EXECUTION_PROGRESS.json"
    ledger_sha_before = _sha256(ledger_path)
    progress_sha_before = _sha256(progress_path)
    preflight = _read(RUN_DIR / "ROUND43_PREFLIGHT.json")
    baseline = _read(RUN_DIR / "ROUND43_BASELINE_REUSE.json")
    ep0 = _read(RUN_DIR / "ROUND43_EP0_INITIALIZATION.json")
    resume_bridge = _read(RUN_DIR / "runtime" / "ROUND43_RESUME_BRIDGE.json")
    recovery_gate = _read(RUN_DIR / "ROUND43_RESUME_RECOVERY_GATE.json")
    state_reconciliation = _read(
        RUN_DIR / "ROUND43_SEED11_BATCH0_STATE_RECONCILIATION.json")
    ledger_reconciliation = _read(RUN_DIR / "ROUND43_LEDGER_RECONCILIATION.json")
    failed_resume = _read(
        RUN_DIR / "failures" / "ROUND43_RUNCALL_RESUME_FAILURE_FREEZE1.json")
    preflight_failure_path = RUN_DIR / "failures" / "ROUND43_PREFLIGHT_COUNTER_ASSERTION_C8.json"
    preflight_failure_raw = _read(preflight_failure_path)
    preflight_failure = render_preflight_failure_correction(
        preflight_failure_raw, source_path=str(preflight_failure_path),
        source_sha256=_sha256(preflight_failure_path))
    raw_manifest_path = RUN_DIR / "ROUND43_RAW_ARTIFACT_MANIFEST.json"
    existing_raw_manifest = _read(raw_manifest_path)
    finalization_audit = _validate_report_inputs(
        frozen=frozen, analysis=analysis, ledger=ledger, progress=progress,
        raw_manifest=existing_raw_manifest)
    prior_source_inventory = _read(RUN_DIR / "ROUND43_SOURCE_INVENTORY.json")
    source_freeze_head = finalization_audit["execution_source_freeze_head"]
    if (prior_source_inventory.get("source_freeze_head") != source_freeze_head
            or prior_source_inventory.get("source_code_sha256")
            != frozen["execution_identity"]["source_code_sha256"]):
        raise RuntimeError("ROUND43 persisted execution identity differs from its prelaunch receipts")
    counters = ledger["counts"]
    branch = _git(ROUND42_ROOT, "branch", "--show-current")
    report_code_head = _git(ROUND42_ROOT, "rev-parse", "HEAD")
    report_code_sha256 = _sha256(Path(__file__).resolve())
    if branch != frozen["execution_identity"]["branch"]:
        raise RuntimeError("ROUND43 report branch differs from the frozen execution branch")

    source_inventory = {
        "kind": "round43-source-and-baseline-inventory/2",
        "branch": branch,
        "source_freeze_head": source_freeze_head,
        "report_renderer_identity": {
            "version": REPORT_VERSION,
            "head": report_code_head,
            "path": "r43_report.py",
            "sha256": report_code_sha256,
        },
        "execution_freeze": frozen["execution_identity"],
        "training_version": frozen["round42"]["training_version"],
        "source_training_version": frozen["round42"]["source_training_version"],
        "execution_basis": {
            "path": str(RUN_DIR / "EXECUTION_BASIS.md"),
            "sha256": _sha256(RUN_DIR / "EXECUTION_BASIS.md"),
        },
        "configuration_diff": {
            "path": str(RUN_DIR / "ROUND43_CONFIGURATION_DIFF.json"),
            "sha256": _sha256(RUN_DIR / "ROUND43_CONFIGURATION_DIFF.json"),
        },
        "prelaunch_gates": {
            "path": str(RUN_DIR / "ROUND43_PREFLIGHT.json"),
            "sha256": _sha256(RUN_DIR / "ROUND43_PREFLIGHT.json"),
            "runtime_environment": preflight["runtime_environment"],
        },
        "execution_recovery": {
            "initial_freeze_head": resume_bridge["prior_prelaunch_head"],
            "recovery_freeze_head": source_freeze_head,
            "execution_code_version": frozen["execution_identity"][
                "execution_code_version"],
            "training_semantics_version_unchanged": (
                resume_bridge["training_version"] == frozen["round42"]["training_version"]),
            "resume_bridge": {
                "path": str(RUN_DIR / "runtime" / "ROUND43_RESUME_BRIDGE.json"),
                "sha256": _sha256(RUN_DIR / "runtime" / "ROUND43_RESUME_BRIDGE.json"),
            },
            "recovery_gate": {
                "path": str(RUN_DIR / "ROUND43_RESUME_RECOVERY_GATE.json"),
                "sha256": _sha256(RUN_DIR / "ROUND43_RESUME_RECOVERY_GATE.json"),
            },
        },
        "frozen_search_allocation_inputs": preflight[
            "frozen_search_allocation_inputs"],
        "registered_ep0_inputs": ep0["by_seed"],
        "baseline_reuse_manifest": {
            "path": str(RUN_DIR / "ROUND43_BASELINE_REUSE.json"),
            "sha256": _sha256(RUN_DIR / "ROUND43_BASELINE_REUSE.json"),
            "reused_counts": {key: value["count"] for key, value in baseline["reused"].items()},
        },
        "source_code_sha256": frozen["execution_identity"]["source_code_sha256"],
        "inherited_dependency_sha256": frozen["execution_identity"][
            "inherited_dependency_sha256"],
        "round40_code_sha256": frozen["execution_identity"]["round40_code_sha256"],
        "round40_delivery_head": frozen["execution_identity"]["round40_delivery_head"],
        "round41_delivery_head": frozen["execution_identity"]["round41_delivery_head"],
        "canonical_read_only_input_head": frozen["execution_identity"][
            "canonical_read_only_input_head"],
        "round42_originals_unchanged": True,
    }
    _atomic_json(RUN_DIR / "ROUND43_SOURCE_INVENTORY.json", source_inventory)

    raw_roots = ("rows", "logs", "checkpoints", "launch_claims", "launch_slots",
                 "training_progress", "runtime", "failures", "rejected")
    raw_files: list[dict[str, Any]] = []
    for name in raw_roots:
        folder = RUN_DIR / name
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            if path.relative_to(RUN_DIR).as_posix() == "runtime/ROUND43_EXECUTION_PROGRESS.json":
                continue
            raw_files.append({
                "relative_path": path.relative_to(RUN_DIR).as_posix(),
                "path": str(path), "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    raw_manifest = {
        "kind": "round43-raw-artifact-manifest/1",
        "raw_artifacts_remain_in_local_run_directory": True,
        "raw_files_count": len(raw_files),
        "files": raw_files,
    }
    prior_raw_fingerprints = sorted(
        (str(item["relative_path"]).replace("\\", "/"), int(item["size_bytes"]),
         str(item["sha256"])) for item in existing_raw_manifest["files"])
    new_raw_fingerprints = sorted(
        (item["relative_path"], int(item["size_bytes"]), item["sha256"])
        for item in raw_files)
    if prior_raw_fingerprints != new_raw_fingerprints:
        raise RuntimeError("report-only finalization found raw artifacts changed since the matrix audit")
    _atomic_json(RUN_DIR / "ROUND43_RAW_ARTIFACT_MANIFEST.json", raw_manifest)

    version_manifest = {
        "kind": "training-version-manifest/1",
        "previous_version": frozen["round42"]["source_training_version"],
        "previous_checkpoint_policy": {
            "resume": "rejected by the actual ROUND43 restore entry before controller, optimizer, or RNG restore",
            "read_only": "allowed with explicit ROUND42 identity",
        },
        "new_version": frozen["round42"]["training_version"],
        "new_checkpoint_policy": {
            "resume": "accepted only with the exact ROUND43 version and freeze identity",
            "initialization": "fresh same-seed controller and empty Adam; ROUND42 ep0 supplies only the registered RNG state",
        },
        "only_research_variable": {"gae_lambda": {"old": 0.95, "new": 1.0}},
        "unchanged": {"gamma": 1.0, "lambda_dual_initial": 1.0,
                      "eta_lambda": 0.05, "delta": 0.0, "b0_bytes": 10000,
                      "comm_action_semantics": "one-shot-send-skip/1",
                      "reward_timing_version": frozen["round42"]["reward_timing_version"]},
        "round42_report_correction": {
            "original_text_claimed": "two greedy NEW seeds were all-SKIP",
            "corrected_from_original_table_and_rows": "all three greedy NEW seeds were all-SKIP",
            "original_round42_report_edited": False,
        },
    }
    _atomic_json(RUN_DIR / "TRAINING_VERSION_MANIFEST.json", version_manifest)

    resume_reconciliation = {
        "kind": "round43-exact-resume-reconciliation/1",
        "status": "passed",
        "initial_freeze_head": resume_bridge["prior_prelaunch_head"],
        "recovery_freeze_head": source_freeze_head,
        "report_code_head": report_code_head,
        "report_code_sha256": report_code_sha256,
        "execution_code_version": frozen["execution_identity"][
            "execution_code_version"],
        "training_semantics_version": frozen["round42"]["training_version"],
        "resume_cursor": resume_bridge["resume_cursor"],
        "previously_completed_normal_claims": resume_bridge["normal_plan"][
            "used_before_resume"],
        "remaining_normal_claims_at_resume": resume_bridge["normal_plan"]["remaining"],
        "technical_reserve_max": resume_bridge["normal_plan"]["technical_reserve"],
        "hard_cap": resume_bridge["normal_plan"]["hard_cap"],
        "initial_runner_failure": {
            "kind": failed_resume["kind"],
            "status": failed_resume["status"],
            "exception": failed_resume["exception"],
            "claim_delta": failed_resume["accounting_at_stop"][
                "environment_claims_delta_during_failed_resume"],
            "environment_constructor_delta": failed_resume["accounting_at_stop"][
                "environment_constructors_delta"],
            "run_call_delta": failed_resume["accounting_at_stop"][
                "environment_run_calls_delta"],
            "optimizer_update_delta": failed_resume["accounting_at_stop"][
                "PPO_or_optimizer_call_delta"],
            "source_path": str(
                RUN_DIR / "failures" / "ROUND43_RUNCALL_RESUME_FAILURE_FREEZE1.json"),
            "source_sha256": _sha256(
                RUN_DIR / "failures" / "ROUND43_RUNCALL_RESUME_FAILURE_FREEZE1.json"),
            "classification": "runner defect before a new launch claim; not a zero-result episode",
        },
        "superseded_preflight_counter_assertion": preflight_failure,
        "old_ledger_snapshot": {
            "normal_used": ledger_reconciliation["original_persisted_ledger"][
                "counts"]["normal_used"],
            "ppo_update_calls": ledger_reconciliation[
                "original_persisted_ledger"]["counts"]["ppo_update_calls"],
            "optimizer_step_calls": ledger_reconciliation[
                "original_persisted_ledger"]["counts"]["optimizer_step_calls"],
            "checkpoint_save_calls": ledger_reconciliation[
                "original_persisted_ledger"]["counts"]["checkpoint_write_calls"],
            "preserved_without_rewrite": True,
        },
        "reconciled_at_pause": ledger_reconciliation[
            "corrected_cumulative_snapshot"],
        "evidence": {
            "seed11_batch0_state_reconciliation": {
                "path": str(RUN_DIR / "ROUND43_SEED11_BATCH0_STATE_RECONCILIATION.json"),
                "sha256": _sha256(
                    RUN_DIR / "ROUND43_SEED11_BATCH0_STATE_RECONCILIATION.json"),
                "status": state_reconciliation["status"],
            },
            "ledger_reconciliation": {
                "path": str(RUN_DIR / "ROUND43_LEDGER_RECONCILIATION.json"),
                "sha256": _sha256(RUN_DIR / "ROUND43_LEDGER_RECONCILIATION.json"),
                "status": ledger_reconciliation["status"],
            },
            "resume_recovery_gate": {
                "path": str(RUN_DIR / "ROUND43_RESUME_RECOVERY_GATE.json"),
                "sha256": _sha256(RUN_DIR / "ROUND43_RESUME_RECOVERY_GATE.json"),
                "status": recovery_gate["status"],
            },
        },
        "first_batch_replayed": False,
        "optimizer_step_replayed": False,
        "round42_trained_checkpoint_used_as_resume_state": False,
    }
    if (resume_reconciliation["initial_runner_failure"]["claim_delta"] != 0
            or resume_reconciliation["initial_runner_failure"][
                "environment_constructor_delta"] != 0
            or resume_reconciliation["initial_runner_failure"]["run_call_delta"] != 0
            or resume_reconciliation["initial_runner_failure"][
                "optimizer_update_delta"] != 0
            or resume_reconciliation["remaining_normal_claims_at_resume"] != 476
            or not recovery_gate["status"] == "passed"
            or not state_reconciliation["status"] == "passed"
            or not ledger_reconciliation["status"] == "passed"
            or not preflight_failure["preserved_artifacts_unchanged"]):
        raise RuntimeError("ROUND43 resume reconciliation did not close")
    _atomic_json(RUN_DIR / "ROUND43_RESUME_RECONCILIATION.json",
                 resume_reconciliation)

    old_eval = baseline["reused"]["old_round42_new_ep128_evaluations"]
    correction = {
        "kind": "round42-report-correction/1",
        "source_round42_report": str(ROUND42_OUTPUT / "ROUND42_RESUME_FINAL_RESULT.json"),
        "source_round42_paired_results": str(ROUND42_OUTPUT / "ROUND42_PAIRED_RESULTS.json"),
        "registered_correction": version_manifest["round42_report_correction"],
        "supporting_new_ep128_greedy_rows": {},
    }
    for seed in (11, 22, 33):
        selected = [item for item in old_eval["rows"]
                    if int(item["row"]["training_seed"]) == seed
                    and item["row"]["eval_mode"] == "greedy"]
        correction["supporting_new_ep128_greedy_rows"][str(seed)] = {
            "count": len(selected),
            "sum_send_now_choices": sum(int(item["row"]["collector_summary"][
                "one_shot_counts"]["send_now_choices"]) for item in selected),
            "sum_skip_episode_choices": sum(int(item["row"]["collector_summary"][
                "one_shot_counts"]["skip_episode_choices"]) for item in selected),
            "row_path_sha256": [{"path": item["path"], "sha256": item["sha256"]}
                                for item in selected],
        }
        if (len(selected) != 16
                or correction["supporting_new_ep128_greedy_rows"][str(seed)][
                    "sum_send_now_choices"] != 0):
            raise RuntimeError(f"ROUND42 all-SKIP correction not proven for seed {seed}")
    _atomic_json(RUN_DIR / "ROUND42_REPORT_CORRECTION.json", correction)

    # Include stable, no-self-reference manifest identity in the source/report inventory.
    raw_manifest_sha = _sha256(RUN_DIR / "ROUND43_RAW_ARTIFACT_MANIFEST.json")
    source_inventory["raw_artifact_manifest"] = {
        "path": str(RUN_DIR / "ROUND43_RAW_ARTIFACT_MANIFEST.json"),
        "sha256": raw_manifest_sha, "raw_files_count": len(raw_files),
    }
    _atomic_json(RUN_DIR / "ROUND43_SOURCE_INVENTORY.json", source_inventory)

    greedy_lines = []
    for seed in (11, 22, 33):
        item = verdict["greedy_primary"]["by_seed"][str(seed)]
        greedy_lines.append(
            f"- seed {seed}：目标集合与 immediate 完全一致={_fmt_bool(item['exact_immediate_on_time_task_set_by_scene'])}；"
            f"NEW {item['new_on_time_total']} 个按时任务 / {item['new_spent_bytes_total']:,} B，"
            f"late {item['late_on_time_total']} 个 / {item['late_spent_bytes_total']:,} B；"
            f"低于 late={_fmt_bool(item['cost_below_late_discovery_gate'])}。"
        )
    sampled_lines = []
    for seed in (11, 22, 33):
        newg = paired["group_summaries"][f"s{seed}_sampled_NEW_round43_ep128"]
        oldg = paired["group_summaries"][f"s{seed}_sampled_OLD_round42_new_ep128"]
        sampled_lines.append(
            f"- seed {seed}：NEW {_fmt_group(newg)}；OLD {_fmt_group(oldg)}。"
        )

    ledger_summary = {
        key: counters.get(key) for key in (
            "normal_used", "reserve_used", "total_used", "environment_constructor_calls",
            "joint_run_constructed_calls", "run_invocation_calls", "joint_run_method_calls",
            "run_return_calls", "run_failure_calls", "completed_steps",
            "completed_logical_jobs", "technical_failed_claims", "retry_claims",
            "ppo_update_calls", "ppo_update_journal_entries", "optimizer_step_calls",
            "optimizer_step_journal_entries", "checkpoint_write_calls", "hard_cap")
    }
    training_state_lines = []
    for seed in (11, 22, 33):
        values = index["by_seed"][str(seed)]["128"]["training_state"]
        training_state_lines.append(
            f"- seed {seed}：episodes={values['episodes_done']}、batches={values['updates']}、"
            f"optimizer steps={values['optimizer_steps']}、λ_dual={values['lambda']:.6f}、"
            f"ep128 参数 SHA256=`{index['by_seed'][str(seed)]['128']['parameter_sha256']}`。"
        )
    absolute_table, paired_table, lambda_table = _result_tables(paired, curves)

    report = f"""# ROUND43 报告：一次性首报选择的完整 GAE trace 消融

## 结论

是否完成：是。三 seed 固定训练和固定评估均完成；总计 {counters['normal_used']} 次正常启动、{counters['reserve_used']} 次技术储备，硬上限 {counters['hard_cap']}。

任务保持是否改善：{"是" if verdict['greedy_primary']['all_three_seeds_exactly_preserve_immediate_task_set'] else "否"}。三个 greedy seed 均保持 immediate 的开发块按时任务身份集合：{_fmt_bool(verdict['greedy_primary']['all_three_seeds_exactly_preserve_immediate_task_set'])}。

是否优于强规则：{"开发块观察到联合改善" if verdict['greedy_primary']['joint_improvement_by_registered_development_rule'] else "未观察到满足预登记联合改善条件的结果"}。条件为三个 greedy seed 各自保持 immediate 按时任务集合，且计费低于 late_discovery_gate。

跨 seed 是否一致：{_fmt_bool(verdict['cross_seed_consistency']['greedy_joint_criterion_all_seeds'])}。判定只针对已用于开发的16个场景，不代表正式非劣、总体零损失或泛化。

训练、评估和分析矩阵闭合后，原执行命令以退出码1停止在报告渲染阶段：报告器读取了不存在的顶层 `exception`，而保留的 preflight 记录把异常放在 `failure`，保留审计放在 `post_failure_read_only_audit`。这是报告接缝故障；没有失败科学格。以下收尾只读已有结果，不重进 `r43_execute.py`、不重建环境、不更新模型。原执行 progress 继续保留 `running`，本地交付完成态另记在 `ROUND43_STATUS.json`。

## 研究变量与边界

唯一研究变量为共享 `gae_lambda` 从 0.95 改为 1.0；`gamma=1.0`、`lambda_dual` 初值 1.0、`eta_lambda=0.05`、`delta=0`、`B0=10000 B` 均保持冻结。task 与 positive-cost GAE 共用新 trace 系数，组合仍为 `lambda_dual*A_task - A_positive_cost`。这是一项方法消融，不是修复旧 GAE 数学错误；GAE=1 同时改变 `A+V` critic target，且可能增大方差。本结果不能证明 ROUND42 的学习失败只由信用衰减导致。

## Greedy 主结果

{chr(10).join(greedy_lines)}

联合判定：`{verdict['scientific_result']}`。

## Sampled 辅助结果

{chr(10).join(sampled_lines)}

完整逐 seed/mode 的 NEW−OLD、NEW−immediate、NEW−late、NEW−always_hold 配对结果和逐目标恢复/丢失列表见 `ROUND43_PAIRED_RESULTS.json`。绝对任务、载荷尝试字节、首报 SEND/SKIP、交付和丢包统计也在其中。

## 所有固定组的绝对结果

`Y` 是按时完成任务数，表中先给16场景总数，再给每集均值；`C` 是载荷发送尝试的计费字节（含丢包），先给总数，再给每集均值。首报列为发送次数/相应首报字节；SEND/SKIP 列为自由首报选择数。丢包仍按尝试字节计入 C。

{absolute_table}

## 按场景配对差和目标身份变化

方向统一为 `NEW − 对照`；ΔY、ΔC 同时列16场景总差与每集均值。救回/丢失集合按 `scene:task_id` 列出；`∅` 表示集合为空，不用净差替代目标身份核对。

{paired_table}

## λ_dual 完整轨迹

每行按32个 batch 的 `lambda_next` 顺序列出；起点均为 λ=1.0000，batch 更新规则为冻结的 `eta_lambda=0.05`。

{lambda_table}

## 训练和路径完整性

{chr(10).join(training_state_lines)}

- 训练曲线和96个批记录：`ROUND43_TRAINING_CURVES.json`；每批含 `g_hat`、`lambda_dual` 更新、PPO指标、梯度摘要、场景和episode行身份。
- ep0/64/128 checkpoint 字节哈希、参数及模块摘要、optimizer/RNG摘要：`ROUND43_CHECKPOINT_INDEX.json`。
- 三个 seed 各32个批的参数链连续；每个 ep128 参数摘要与最后一批更新后状态核对。
- 首批真实 PPO 接线门禁：`ROUND43_FIRST_BATCH_GATE_SEED11.json`、`ROUND43_FIRST_BATCH_GATE_SEED22.json`、`ROUND43_FIRST_BATCH_GATE_SEED33.json`。
- 执行账本和逐 claim 计数：`runtime/ROUND43_EXECUTION_LEDGER.json`；完成逻辑任务={counters['completed_logical_jobs']}，真实完成步={counters['completed_steps']}。

启动与更新对账：{json.dumps(ledger_summary, ensure_ascii=False)}。

## 恢复与计数勘误

首次冻结 `{resume_bridge['prior_prelaunch_head']}` 下，seed11 已完成 episode0–3 并提交 batch0。继续运行时在下一次 episode claim 前，冻结 ROUND40 训练入口因 `parameter_identity` 局部导入作用域错误抛出 `UnboundLocalError`。这次是运行器缺陷：新增 claim、环境构造、run 调用和优化更新均为0，不作为零任务/零成本 episode，也不计入480个正常任务。

恢复门禁复核了 after_b0 与 rolling 状态、完整模型与 Adam 状态、12个 Adam step、λ与历史、训练版本、游标、RNG 和剩余场景顺序。ROUND42 验证过的适配器只把 `parameter_identity` 导入移到训练入口作用域；训练语义版本保持 `{resume_bridge['training_version']}`。恢复从 episode4、update1、Adam step12、λ={resume_bridge['resume_cursor']['lambda_dual']:.4f} 开始，首批没有重放。恢复冻结 `{source_freeze_head}` 使用执行代码身份 `{frozen['execution_identity']['execution_code_version']}`。

第一次 recovery preflight 因旧累计计数断言而停止：它把历史 Adam 12次调用误判为本次门禁调用。该次无 claim、无环境构造；测试内的伪优化器只到达立即抛错的更新边界，真实模型、优化器与 checkpoint 未变。断言现改为比较门禁前后的累计计数；失败记录与产物哈希核验见 `failures/ROUND43_PREFLIGHT_COUNTER_ASSERTION_C8.json`。

停机时原始 ledger 的快照把 PPO/Adam 计数记为0、checkpoint 保存调用记为2。该旧快照作为原件保留；批提交、逐调用日志及 checkpoint 状态核验出的累计值分别为 PPO 1、Adam 12、checkpoint 保存调用4。checkpoint 保存调用依据调用日志，不由文件数量推算。精确证据和哈希见 `ROUND43_RESUME_RECONCILIATION.json`、`ROUND43_LEDGER_RECONCILIATION.json` 与 `ROUND43_SEED11_BATCH0_STATE_RECONCILIATION.json`。

一次 recovery preflight 因旧累计计数断言产生的历史失败记录也保留。真实 optimizer 更新=0，合成 optimizer 更新=0；测试曾记录1次立即失败的合成 step 尝试。当前报告按已保存的嵌套 `failure` 与 `post_failure_read_only_audit` 字段读取它，完整文案、观察计数和原件 SHA 见 `ROUND43_RESUME_RECONCILIATION.json` 与 `ROUND43_REPORT_FAILURE_CORRECTION.json`。字段兼容没有改写失败记录，也没有把它改成通过。

会话/usage 中断提示与实验进程故障分开记录。当前可核实的执行命令退出码1来自报告器 `KeyError: 'exception'`；不是仿真、训练或评估失败。报告终结期间新增 claim、环境构造、run、PPO、optimizer 更新和 checkpoint 写入均为0；execution ledger 与原 `running` progress 哈希前后保持不变。

## 初值与随机状态

NEW 均由同 seed fresh controller、空 Adam 和 `lambda_dual=1.0` 开始。每 seed 的模型参数哈希与 ROUND42 对应 ep0 相同；optimizer state、cursor=0、updates=0、optimizer_steps=0、lambda 初值相符。ROUND42 ep0 checkpoint 只读提供登记的 RNG，不提供模型权重或 optimizer 状态。

门禁发现当前进程 fresh-controller 构造后的 Python/NumPy 全局状态在部分 seed 上与旧进程捕获值不同；runner 在 controller 构造之后、首次 rollout 之前，通过实际 fresh-controller 接线恢复登记的 ep0 RNG。恢复后的 Python、NumPy、Torch 状态均逐项与相应 ep0 checkpoint 一致。环境每 episode 按原 `scenario_seed` 构造并 reseed 其自身状态；通信动作仍使用原逐集 SHA256 action seed；PPO minibatch 顺序仍用 `DATA_ORDER_SEED:update_index` 的独立 `random.Random`；场景顺序 SHA256 与 ROUND42 一致。这个核验支持相同的起始 RNG 状态，不代表三 seed 轨迹在参数更新后仍逐位配对。

细项见 `ROUND43_EP0_INITIALIZATION.json`、`ROUND43_PREFLIGHT.json` 和 `ROUND43_CONFIGURATION_DIFF.json`。

## 版本与旧结果勘误

ROUND43 登记新训练版本 `{frozen['round42']['training_version']}`。实际恢复入口拒绝旧 ROUND42 已训练 checkpoint 作为 ROUND43 训练恢复源；旧模型仍允许带身份信息只读加载。ROUND42 的原始报告正文把 greedy NEW 全 SKIP 写成“两 seed”；逐行原件和配对表确认是三个 seed。勘误保存在 `ROUND42_REPORT_CORRECTION.json`，未改写 ROUND42 原报告、模型、tape 或日志。

## 基线复用与数据范围

本轮复用 64 个 immediate 行、32 个 late/always_hold 规则行，以及 ROUND42 NEW ep128 的96个 OLD评估行；三类均已校验原 manifest 哈希和对应身份。ROUND43 新增384个训练episode、96个 ep128 NEW评估。参考源、checkpoint及文件哈希见 `ROUND43_BASELINE_REUSE.json`、`ROUND43_SOURCE_INVENTORY.json` 与 `ROUND43_RAW_ARTIFACT_MANIFEST.json`。原始权重、tape、环境日志保留在本地 run 目录，没有写入报告仓库内容之外的文件。

## 预启动与复现身份

- Branch：`{branch}`
- 预启动执行源码 HEAD：`{source_freeze_head}`
- 报告终结器 HEAD：`{report_code_head}`；`r43_report.py` SHA256：`{report_code_sha256}`（与执行源码身份分列）
- Python：`{frozen['execution_identity']['python_executable']}`；实验设备：CPU
- Runtime：Python {preflight['runtime_environment']['python_version']}、NumPy {preflight['runtime_environment']['numpy_version']}、PyTorch {preflight['runtime_environment']['torch_version']}（CUDA 可用性={preflight['runtime_environment']['cuda_available_but_unused']}，本轮运行固定 CPU）
- 执行依据 SHA256：`{source_inventory['execution_basis']['sha256']}`
- 配置差异 SHA256：`{source_inventory['configuration_diff']['sha256']}`
- 原始产物清单：`ROUND43_RAW_ARTIFACT_MANIFEST.json`（{len(raw_manifest['files'])}个文件，SHA256 `{raw_manifest_sha}`）
- 无环境正确性门禁：通过，`draft_mode=false`；记录在 `ROUND43_PREFLIGHT.json`。
- 每个 seed 首四集使用真实 batch_samples/GAE/actor/恢复路径；通过后完成其余124集。
- 固定命令：`D:\\project_env\\python.exe runs/comm_one_shot_full_trace_ablation_20261004/r43_execute.py`。

## 终结阶段门禁

- 96 个 NEW 格按精确 `(seed, mode, scene)` 清单复核：每格1个 complete normal claim、1行、1条 tape、1份 lifecycle、1份 transcript；共480个 claim/row/linked-artifact 文件匹配原始 SHA 清单，且 ep128 参数身份逐 seed 闭合。
- 训练项384集、96批、三 seed 各128集/32批；ep0/64/128 checkpoint 身份与本地文件哈希复核，三份首批门禁均通过。
- 本次 report-only 阶段 checkpoint deserialization/forward=0。此前 analysis 已生成 `ROUND43_CHECKPOINT_INDEX.json`，含9个里程碑的只读身份核验。
- 账本累计：480正常启动、0 reserve、480构造、480 run、480000步、0运行失败；PPO 96、Adam step 1016、checkpoint 写入297。基线复用的64 immediate、32规则和96 OLD评估不计入这480次新启动。
- 完成态和报告故障更正见 `ROUND43_STATUS.json`、`ROUND43_REPORT_FAILURE_CORRECTION.json`。历史 `runtime/ROUND43_EXECUTION_PROGRESS.json` 保留 `running` 原值。

## 限制

结果来自已用于开发的固定16场景。没有独立确认实验、额外训练、调参、hazard decoder 或补充场景。通信成本仅指载荷尝试字节，丢包仍计费，不代表全系统通信或能耗。ROUND42 原始模型、tape、日志和报告均未覆盖。
"""
    report_path = RUN_DIR / "ROUND43_REPORT.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    report_sha256 = _sha256(report_path)
    status = {
        "kind": "round43-project-status/2",
        "round": 43,
        "status": "complete_local_delivery",
        "execution_matrix_status": "complete",
        "execution_progress_status_preserved": progress["status"],
        "source_freeze_head": source_freeze_head,
        "report_code_head": report_code_head,
        "report_code_sha256": report_code_sha256,
        "report_version": REPORT_VERSION,
        "branch": branch,
        "training_version": frozen["round42"]["training_version"],
        "result": verdict["scientific_result"],
        "report_only_recovery_after_executor_exit_1": "fixed KeyError in failure-record rendering",
        "prelaunch_gate": "passed",
        "training_episodes": 384,
        "final_evaluations": 96,
        "normal_launches": counters["normal_used"],
        "reserve_launches": counters["reserve_used"],
        "hard_cap": counters["hard_cap"],
        "environment_constructor_calls": counters["environment_constructor_calls"],
        "completed_steps": counters["completed_steps"],
        "ppo_update_calls": counters["ppo_update_calls"],
        "optimizer_step_calls": counters["optimizer_step_calls"],
        "checkpoint_write_calls": counters["checkpoint_write_calls"],
        "report_sha256": report_sha256,
        "report_failure_correction_path": "ROUND43_REPORT_FAILURE_CORRECTION.json",
        "new_environment_starts_during_finalization": 0,
        "new_optimizer_updates_during_finalization": 0,
        "checkpoint_write_calls_during_finalization": 0,
        "independent_confirmation": False,
        "remote_push": False,
    }
    _atomic_json(RUN_DIR / "ROUND43_STATUS.json", status)
    if _sha256(ledger_path) != ledger_sha_before or _sha256(progress_path) != progress_sha_before:
        raise RuntimeError("report-only finalization changed the frozen execution ledger/progress")
    correction = {
        "kind": "round43-report-failure-correction/1",
        "status": "report_only_recovery_complete",
        "original_executor": {
            "command": "D:\\project_env\\python.exe r43_execute.py",
            "exit_code": 1,
            "failure": {
                "type": "KeyError",
                "message": "'exception'",
                "source_location": "r43_report.py:217 before report compatibility fix",
                "evidence_source": "original r43_execute command session output; recorded exit code 1",
                "classification": "report generator failure after training, evaluation, ledger closure, and analysis; no failed scientific cell",
            },
            "new_claims_after_report_failure": 0,
            "new_environment_starts_after_report_failure": 0,
        },
        "reported_session_or_usage_interruption": {
            "classification": "separate interaction/session notice; not an experiment claim or runner result",
            "counted_in_execution_ledger": False,
        },
        "matrix_audit": finalization_audit,
        "execution_source_identity": {
            "prelaunch_head": source_freeze_head,
            "execution_code_version": frozen["execution_identity"]["execution_code_version"],
            "frozen_source_code_sha256": frozen["execution_identity"]["source_code_sha256"],
        },
        "report_code_identity": {
            "head": report_code_head,
            "renderer_sha256": report_code_sha256,
            "version": REPORT_VERSION,
        },
        "preservation": {
            "execution_progress_path": str(progress_path),
            "execution_progress_status": progress["status"],
            "execution_progress_sha256_before": progress_sha_before,
            "execution_progress_sha256_after": _sha256(progress_path),
            "execution_progress_unchanged": _sha256(progress_path) == progress_sha_before,
            "execution_ledger_sha256_before": ledger_sha_before,
            "execution_ledger_sha256_after": _sha256(ledger_path),
            "execution_ledger_unchanged": _sha256(ledger_path) == ledger_sha_before,
            "prior_output_hashes": old_output_hashes,
            "persisted_analysis_input_hashes": analysis_input_hashes,
        },
        "finalization_accounting": {
            "environment_constructor_calls": 0,
            "run_invocation_calls": 0,
            "ppo_update_calls": 0,
            "optimizer_step_calls": 0,
            "checkpoint_write_calls": 0,
            "checkpoint_deserializations": 0,
            "forward_calls": 0,
            "execution_ledger_counts": counters,
        },
        "new_output_hashes": {
            name: (_sha256(RUN_DIR / name) if (RUN_DIR / name).is_file() else None)
            for name in output_names if name != "ROUND43_REPORT_FAILURE_CORRECTION.json"
        },
        "scientific_interpretation": verdict["scientific_result"],
        "original_failure_records_edited": False,
        "raw_rows_claims_tapes_checkpoints_or_logs_overwritten": False,
    }
    correction_path = RUN_DIR / "ROUND43_REPORT_FAILURE_CORRECTION.json"
    _atomic_json(correction_path, correction)
    return {"report_path": str(report_path), "report_sha256": report_sha256,
            "status": status, "status_sha256": _sha256(RUN_DIR / "ROUND43_STATUS.json"),
            "correction_path": str(correction_path),
            "correction_sha256": _sha256(correction_path),
            "raw_manifest_sha256": raw_manifest_sha}


def finalize_recorded_results() -> dict[str, Any]:
    """Render persisted ROUND43 results without importing or calling the executor."""
    def read(name: str) -> Any:
        return _read(RUN_DIR / name)

    analysis = {
        "paired_results": read("ROUND43_PAIRED_RESULTS.json"),
        "training_curves": read("ROUND43_TRAINING_CURVES.json"),
        "checkpoint_index": read("ROUND43_CHECKPOINT_INDEX.json"),
        "machine_verdict": read("MACHINE_VERDICT.json"),
    }
    return write_final_report(
        frozen=read("ROUND43_FROZEN_CONFIG.json"),
        analysis=analysis,
        ledger=read("runtime/ROUND43_EXECUTION_LEDGER.json"),
        progress=read("runtime/ROUND43_EXECUTION_PROGRESS.json"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="ROUND43 report-only finalization")
    parser.add_argument("--finalize-recorded-results", action="store_true",
                        help="render saved results without starting the ROUND43 executor")
    args = parser.parse_args()
    if not args.finalize_recorded_results:
        parser.error("pass --finalize-recorded-results to run the report-only path")
    # Keep the console summary ASCII-safe on Windows code pages; report files remain UTF-8.
    print(json.dumps(finalize_recorded_results(), ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["finalize_recorded_results", "main", "render_preflight_failure_correction",
           "write_final_report"]
