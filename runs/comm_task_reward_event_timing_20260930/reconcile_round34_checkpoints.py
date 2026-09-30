"""Read-only ROUND34 checkpoint and source-identity reconciliation (stdlib only).

This script hashes checkpoint bytes and joins the archived JSON index, batch records,
mechanism diagnostics, episode rows, and tapes by explicit batch/episode identity. It never
opens a .pt payload and therefore cannot verify tensor contents or run a forward pass.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
R32 = ROOT / "runs" / "comm_actor_reduction_ablation_20261001"
AUDIT33 = ROOT / "runs" / "comm_temporal_credit_audit_20260930"
OUTPUT = ROOT / "runs" / "comm_task_reward_event_timing_20260930" / \
    "CHECKPOINT_IDENTITY_RECONCILIATION.json"
FIXED_BATCHES = (0, 15, 31)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def resolve_repo_path(value: str) -> Path:
    return ROOT / Path(value)


def sorted_batch_records(group: dict[str, Any]) -> dict[int, tuple[Path, dict[str, Any]]]:
    """Join the list by its JSON `batch` field, never by list position."""
    records: dict[int, tuple[Path, dict[str, Any]]] = {}
    for value in group["batch_records"]:
        path = resolve_repo_path(value)
        record = read_json(path)
        batch = int(record["batch"])
        if batch in records:
            raise ValueError(f"duplicate batch {batch}: {path}")
        if path.stem.rsplit("_b", 1)[-1] != str(batch):
            raise ValueError(f"filename/record batch mismatch: {path}")
        records[batch] = (path, record)
    return records


def row_and_tape_index() -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[str, Any]]:
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    row_inventory = hashlib.sha256()
    row_paths = sorted((R32 / "rows").glob("*.json"), key=lambda p: p.name)
    for path in row_paths:
        raw = path.read_bytes()
        row_inventory.update(path.name.encode("utf-8"))
        row_inventory.update(b"\t")
        row_inventory.update(sha256_bytes(raw).encode("ascii"))
        row_inventory.update(b"\n")
        row = json.loads(raw.decode("utf-8"))
        if row.get("kind") == "train":
            key = (str(row["reduction"]), int(row["training_seed"]),
                   int(row["episode_index"]))
            rows[key] = row

    # Match ROUND33's explicit row order: reduction name, numeric seed, numeric episode.
    train_rows = sorted(rows.values(), key=lambda row: (
        str(row["reduction"]), int(row["training_seed"]), int(row["episode_index"])))
    tape_inventory = hashlib.sha256()
    for row in train_rows:
        tape_path = resolve_repo_path(row["artefacts"]["tape"])
        tape_inventory.update(tape_path.name.encode("utf-8"))
        tape_inventory.update(b"\t")
        tape_inventory.update(sha256_file(tape_path).encode("ascii"))
        tape_inventory.update(b"\n")

    return rows, {
        "row_json_file_count": len(row_paths),
        "row_json_inventory_sha256": row_inventory.hexdigest(),
        "training_row_count": len(train_rows),
        "training_tape_count": len(train_rows),
        "training_tape_inventory_sha256": tape_inventory.hexdigest(),
    }


def tape_rows_for_batch(
    arm: str,
    seed: int,
    record: dict[str, Any],
    rows: dict[tuple[str, int, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, bool]:
    tape_refs: list[dict[str, Any]] = []
    total_lines = 0
    rows_match = True
    for episode, scenario in zip(record["episodes"], record["scenarios"]):
        key = (arm, seed, int(episode))
        row = rows.get(key)
        if row is None:
            rows_match = False
            tape_refs.append({"episode_index": int(episode), "missing_row": True})
            continue
        tape_path = resolve_repo_path(row["artefacts"]["tape"])
        if int(row["scenario_seed"]) != int(scenario):
            rows_match = False
        raw = tape_path.read_bytes()
        line_count = sum(1 for line in raw.splitlines() if line.strip())
        total_lines += line_count
        tape_refs.append({
            "episode_index": int(episode),
            "scenario_seed": int(scenario),
            "episode_row": str(Path(row["artefacts"]["row"]).as_posix()),
            "tape_path": str(Path(row["artefacts"]["tape"]).as_posix()),
            "tape_sha256": sha256_bytes(raw),
            "tape_rows": line_count,
        })
    if len(record["episodes"]) != len(record["scenarios"]):
        rows_match = False
    return tape_refs, total_lines, rows_match


def checkpoint_row(
    run_key: str,
    group: dict[str, Any],
    batch_index: int,
    records: dict[int, tuple[Path, dict[str, Any]]],
    mechanism: dict[str, Any],
    rows: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    arm, seed_text = run_key.rsplit("_s", 1)
    seed = int(seed_text)
    cp_meta = group["pre_update_checkpoints"][str(batch_index)]
    cp_path = resolve_repo_path(cp_meta["path"])
    cp_hash = sha256_file(cp_path)
    batch_path, batch = records[batch_index]
    point = mechanism["runs"][run_key][f"b{batch_index}"]
    tape_refs, tape_line_count, tape_row_map_ok = tape_rows_for_batch(
        arm, seed, batch, rows)

    before = batch.get("controller_parameters_before", {})
    mechanism_counts = point.get("recorded_during_training", {}).get(
        "samples_recorded_vs_recomputed", {})
    update_samples = int(batch.get("update", {}).get("samples", -1))
    mechanism_sample_counts = {
        "advantages_A_task_n": point.get("advantages", {}).get("A_task", {}).get("n"),
        "recorded_samples": mechanism_counts.get("recorded"),
        "recomputed_samples": mechanism_counts.get("recomputed"),
        "stored_match_flag": mechanism_counts.get("match"),
        "batch_samples": update_samples,
        "current_tape_rows": tape_line_count,
    }
    parameter_strings_align = (
        cp_meta.get("parameter_sha256") == before.get("parameter_sha256")
        == point.get("pre_update_parameter_sha256")
    )
    lambda_values = [
        float(cp_meta["lambda"]),
        float(batch["lambda_at_rollout"]),
        float(point["state_at_start"]["lambda"]),
    ]
    lambda_align = max(lambda_values) - min(lambda_values) <= 1e-12
    expected_episodes_done = 4 * (batch_index + 1)
    state_metadata_align = (
        int(cp_meta["episodes_done"]) == int(point["state_at_start"]["episodes_done"])
        == expected_episodes_done
        and int(cp_meta["updates"]) == int(point["state_at_start"]["updates"]) == batch_index
        and batch.get("episodes") == list(range(batch_index * 4, batch_index * 4 + 4))
    )
    prior_optimizer_steps = sum(
        int(records[prior][1]["update"]["optimizer_steps"])
        for prior in range(batch_index)
    )
    optimizer_claim_align = (
        prior_optimizer_steps == int(point["state_at_start"]["optimizer_steps"])
    )
    scenario_map_align = (
        cp_meta.get("scenarios_of_batch") == batch.get("scenarios")
        and tape_row_map_ok
    )
    tape_sample_counts_align = (
        tape_row_map_ok
        and len({
            tape_line_count,
            update_samples,
            mechanism_sample_counts["advantages_A_task_n"],
            mechanism_sample_counts["recorded_samples"],
            mechanism_sample_counts["recomputed_samples"],
        }) == 1
        and mechanism_sample_counts["stored_match_flag"] is True
    )
    byte_hash_match = cp_hash == cp_meta.get("sha256")
    batch_file_checkpoint_link = (
        Path(batch["pre_update_checkpoint"]).name == cp_path.name
        and batch_index in batch.get("pre_update_checkpoint_saved_for_batches", [])
    )

    records_list = group["batch_records"]
    positional_path = resolve_repo_path(records_list[batch_index])
    positional_batch = read_json(positional_path)
    positional_batch_number = int(positional_batch["batch"])

    return {
        "checkpoint_path": str(Path(cp_meta["path"]).as_posix()),
        "file_sha256": {
            "observed": cp_hash,
            "index_claim": cp_meta.get("sha256"),
            "size_bytes_observed": cp_path.stat().st_size,
            "size_bytes_index_claim": cp_meta.get("size_bytes"),
            "status": "pass" if byte_hash_match else "fail",
        },
        "arm": arm,
        "training_seed": seed,
        "batch_index": batch_index,
        "phase": "pre_update_checkpoint_for_this_batch",
        "state_claims_from_index_and_mechanism": {
            "episodes_done": int(cp_meta["episodes_done"]),
            "updates": int(cp_meta["updates"]),
            "optimizer_steps_before_this_batch": int(point["state_at_start"]["optimizer_steps"]),
            "optimizer_steps_cross_checked_against_prior_batch_records": prior_optimizer_steps,
            "lambda": float(cp_meta["lambda"]),
            "training_version_claim_from_round32_group": group.get("trainer_tag"),
            "checkpoint_payload_training_version": "unavailable_without_pytorch",
            "reward_timing_version_in_checkpoint_payload": "unavailable_without_pytorch",
        },
        "parameter_identity_claims_not_recomputed_from_pt": {
            "checkpoint_index_parameter_sha256_claim": cp_meta.get("parameter_sha256"),
            "batch_record_controller_parameters_before": before,
            "module_digest_claims_before": batch.get("controller_module_digests_before"),
            "mechanism_pre_update_parameter_sha256_claim": point.get("pre_update_parameter_sha256"),
            "digest_recipe": {
                "controller_parameter_sha256": (
                    "StageBController.parameter_summary / r32_manifest.parameter_sha: "
                    "SHA256 over sorted module-prefixed parameter names and contiguous CPU tensor bytes"
                ),
                "module_digest_sha256": (
                    "r32_train.module_digests: module-local parameter names and contiguous CPU tensor bytes; "
                    "truncated to 16 hex characters"
                ),
                "file_sha256": "SHA256 of serialized checkpoint file bytes",
            },
            "index_batch_mechanism_digest_strings_align": parameter_strings_align,
            "tensor_content_status": "unavailable_without_pytorch",
            "module_tensor_content_status": "unavailable_without_pytorch",
        },
        "corrected_batch_record_source": {
            "path": str(batch_path.relative_to(ROOT).as_posix()),
            "batch_field": int(batch["batch"]),
            "episodes": [int(value) for value in batch["episodes"]],
            "scenarios": [int(value) for value in batch["scenarios"]],
            "lambda_at_rollout": float(batch["lambda_at_rollout"]),
            "update_samples": update_samples,
            "optimizer_steps_in_this_batch": int(batch["update"]["optimizer_steps"]),
            "pre_update_checkpoint_link_matches": batch_file_checkpoint_link,
            "likelihood_alignment_saved_gate": batch["update"].get("likelihood_alignment"),
        },
        "mechanism_diagnosis_source": {
            "path": "runs/comm_actor_reduction_ablation_20261001/UPDATE_MECHANISM.json",
            "point": f"{run_key}/b{batch_index}",
            "state_at_start": point.get("state_at_start"),
            "samples": mechanism_sample_counts,
        },
        "tape_sources": tape_refs,
        "corrected_metadata_alignment": {
            "checkpoint_file_sha256_matches_index": byte_hash_match,
            "checkpoint_index_batch_record_parameter_claims_match": parameter_strings_align,
            "lambda_claims_match": lambda_align,
            "episodes_and_update_claims_match": state_metadata_align,
            "optimizer_step_claim_matches_prior_batch_record_sum": optimizer_claim_align,
            "scenario_and_episode_tape_mapping_matches": scenario_map_align,
            "tape_and_saved_sample_counts_match": tape_sample_counts_align,
            "batch_record_points_to_this_checkpoint": batch_file_checkpoint_link,
            "metadata_claim_status": (
                "aligned_across_index_batch_record_mechanism_and_tapes" if all((
                    byte_hash_match, parameter_strings_align, lambda_align, state_metadata_align,
                    optimizer_claim_align, scenario_map_align, tape_sample_counts_align,
                    batch_file_checkpoint_link)) else "conflict_or_missing_source"
            ),
        },
        "round33_positional_array_mapping": {
            "array_position_used_as_batch_index": batch_index,
            "selected_record_path": str(positional_path.relative_to(ROOT).as_posix()),
            "selected_record_batch_field": positional_batch_number,
            "wrong_record_selected": positional_batch_number != batch_index,
            "corrected_lookup": "select batch record by its explicit `batch` field / filename suffix",
        },
        "verification_layers": {
            "checkpoint_file_sha256": "pass" if byte_hash_match else "fail",
            "index_batch_mechanism_identity_claims": (
                "pass" if all((parameter_strings_align, lambda_align, state_metadata_align,
                               optimizer_claim_align, scenario_map_align, tape_sample_counts_align,
                               batch_file_checkpoint_link)) else "fail"
            ),
            "checkpoint_payload_metadata": "unavailable_without_pytorch",
            "parameter_content": "unavailable_without_pytorch",
            "module_parameter_content": "unavailable_without_pytorch",
            "native_forward": "not_run_without_pytorch",
            "native_likelihood_recomputation": "not_run_without_pytorch",
            "saved_likelihood_gate": "recorded_only; not a native checkpoint forward",
        },
    }


def main() -> None:
    index = read_json(R32 / "CHECKPOINT_INDEX.json")
    mechanism = read_json(R32 / "UPDATE_MECHANISM.json")
    old_audit = read_json(AUDIT33 / "TEMPORAL_CREDIT_AUDIT.json")
    old_critic = read_json(AUDIT33 / "CRITIC_DIAGNOSTIC.json")
    rows, current_inventory = row_and_tape_index()

    prior_inventory = old_audit["source_lineage"]
    inventory_match = (
        current_inventory["row_json_inventory_sha256"]
        == prior_inventory["row_json_inventory_sha256"]
        and current_inventory["training_tape_inventory_sha256"]
        == prior_inventory["training_tape_inventory_sha256"]
    )

    old_conflicts = {item["point"]: item for item in old_critic["provenance_conflicts"]}
    old_validation = {item["point"]: item for item in old_critic["checkpoint_validation"]}
    checkpoints: list[dict[str, Any]] = []
    # Check all six b0 initialization points before inspecting the later b15/b31 states.
    for batch_index in FIXED_BATCHES:
        for run_key in sorted(index):
            group = index[run_key]
            if "pre_update_checkpoints" not in group:
                continue
            records = sorted_batch_records(group)
            row = checkpoint_row(run_key, group, batch_index, records, mechanism, rows)
            point_key = f"{run_key}/b{batch_index}"
            old = old_conflicts.get(point_key)
            old_validation_row = old_validation.get(point_key)
            row["round33_reported_conflict"] = old
            row["round33_reported_checkpoint_validation"] = old_validation_row
            row["round33_mapping_reconstruction_matches_reported_episode_list"] = (
                old_validation_row is not None
                and old_validation_row.get("episode_list") == read_json(
                    resolve_repo_path(group["batch_records"][batch_index]))["episodes"]
            )
            checkpoints.append(row)

    wrong_rows = [row for row in checkpoints
                  if row["round33_positional_array_mapping"]["wrong_record_selected"]]
    corrected_metadata_pass = [row for row in checkpoints if
        row["corrected_metadata_alignment"]["metadata_claim_status"]
        == "aligned_across_index_batch_record_mechanism_and_tapes"]
    byte_pass = [row for row in checkpoints if row["file_sha256"]["status"] == "pass"]
    report = {
        "round": 34,
        "artifact_type": "read-only checkpoint identity reconciliation",
        "inputs": {
            "round32_index": "runs/comm_actor_reduction_ablation_20261001/CHECKPOINT_INDEX.json",
            "round32_batch_records": "runs/comm_actor_reduction_ablation_20261001/checkpoints/r32_batch_<arm>_s<seed>_b<batch>.json",
            "round32_mechanism": "runs/comm_actor_reduction_ablation_20261001/UPDATE_MECHANISM.json",
            "round32_rows_and_tapes": "runs/comm_actor_reduction_ablation_20261001/{rows,logs}",
            "round33_temporal_audit": "runs/comm_temporal_credit_audit_20260930/TEMPORAL_CREDIT_AUDIT.json",
            "round33_critic_diagnostic": "runs/comm_temporal_credit_audit_20260930/CRITIC_DIAGNOSTIC.json",
        },
        "mapping_correction": {
            "checkpoint_review_batch_order": list(FIXED_BATCHES),
            "finding": (
                "ROUND33 indexed the lexically sorted `group.batch_records` list with the numeric "
                "batch number. At b15 this selects list position 15 (record b22); at b31 it selects "
                "position 31 (record b9). This created the 12 reported b15/b31 conflicts."
            ),
            "correction": "join by each sidecar's explicit `batch` field and verify its filename suffix",
            "wrong_positional_mappings": len(wrong_rows),
            "wrong_points": [
                {"point": f"{row['arm']}_s{row['training_seed']}/b{row['batch_index']}",
                 "selected_record_batch": row["round33_positional_array_mapping"]["selected_record_batch_field"],
                 "selected_record_path": row["round33_positional_array_mapping"]["selected_record_path"]}
                for row in wrong_rows
            ],
            "round33_reported_episode_lists_reproduced_from_positional_mappings": sum(
                row["round33_mapping_reconstruction_matches_reported_episode_list"]
                for row in checkpoints
            ),
            "preserved_round33_outputs": True,
        },
        "round33_raw_data_identity": {
            **current_inventory,
            "row_inventory_sha256_in_round33": prior_inventory["row_json_inventory_sha256"],
            "training_tape_inventory_sha256_in_round33": prior_inventory["training_tape_inventory_sha256"],
            "round33_row_and_training_tape_inventories_match": inventory_match,
        },
        "summary": {
            "checkpoint_files": len(checkpoints),
            "file_sha256_pass": len(byte_pass),
            "corrected_index_batch_mechanism_tape_metadata_aligned": len(corrected_metadata_pass),
            "checkpoint_payload_metadata_available": 0,
            "parameter_content_available": 0,
            "module_parameter_content_available": 0,
            "native_forward_runs": 0,
            "native_likelihood_recomputations": 0,
            "stored_likelihood_gates_recorded_pass": sum(
                row["corrected_batch_record_source"]["likelihood_alignment_saved_gate"].get("passed") is True
                for row in checkpoints
            ),
        },
        "hash_namespace_rules": {
            "file_sha256": "hash of serialized .pt file bytes; compare only with file/index SHA256",
            "controller_parameter_sha256": "parameter-content digest claim using sorted module-prefixed names and contiguous tensor bytes; not recomputed here",
            "module_digests": "four independent truncated module-level digests; do not compare with whole-controller or file SHA256",
            "batch_record_identity": "episodes, scenarios, lambda, batch index, update counts, and saved hash claims; not checkpoint tensor content",
        },
        "checkpoint_verification_limit": (
            "PyTorch is unavailable. Matching file SHA256 proves bytes match the index entry only. "
            "Corrected metadata alignment proves archived JSON claims and tape counts are internally "
            "consistent, but does not prove the serialized .pt payload contains those parameters."
        ),
        "checkpoints": checkpoints,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)),
        "checkpoints": len(checkpoints),
        "byte_sha_pass": len(byte_pass),
        "corrected_metadata_aligned": len(corrected_metadata_pass),
        "round33_wrong_batch_record_offsets": len(wrong_rows),
        "round33_raw_row_tape_inventory_match": inventory_match,
        "payload_content_status": "unavailable_without_pytorch",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
