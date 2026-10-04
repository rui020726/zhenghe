"""ROUND43 fixed-block analysis using the registered ROUND42 paired statistics."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

import r42_analysis
from r42_run import ROUND42_OUTPUT, RUN_DIR, SEEDS, configure


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_or_match(path: Path, value: Any) -> None:
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"ROUND43 result already exists with different content: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()


def _claim_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")


def _claim_is_complete(logical_job: str, attempt: int) -> bool:
    claim = RUN_DIR / "launch_claims" / _claim_slug(logical_job) / f"attempt_{attempt:04d}.json"
    return claim.is_file() and _read(claim).get("status") == "complete"


def _committed_training_rows(seed: int) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    checkpoint_dir = RUN_DIR / "checkpoints"
    for batch in range(32):
        commit_path = checkpoint_dir / f"r40_batch_commit_s{seed}_b{batch}.json"
        if not commit_path.is_file():
            raise RuntimeError(f"seed {seed} batch {batch}: immutable commit receipt missing")
        commit = _read(commit_path)
        record_path = checkpoint_dir / commit["batch_record"]
        if not record_path.is_file() or _sha256(record_path) != commit["batch_record_sha256"]:
            raise RuntimeError(f"seed {seed} batch {batch}: committed record hash mismatch")
        record = _read(record_path)
        if int(record.get("batch", -1)) != batch or int(record.get("training_seed", -1)) != seed:
            raise RuntimeError(f"seed {seed} batch {batch}: committed record identity mismatch")
        for artifact in record["episode_artifacts"]:
            episode = int(artifact["episode_index"])
            row_path = Path(artifact["row_path"])
            if not row_path.is_file() or _sha256(row_path) != artifact["row_sha256"]:
                raise RuntimeError(f"seed {seed} episode {episode}: committed row hash mismatch")
            row = _read(row_path)
            logical_job = f"new_training_s{seed}_e{episode}"
            if not _claim_is_complete(logical_job, int(row["attempt"])):
                raise RuntimeError(f"seed {seed} episode {episode}: committed row lacks a complete claim")
            if episode in selected:
                raise RuntimeError(f"seed {seed} episode {episode}: multiple committed rows")
            selected[episode] = row
    if set(selected) != set(range(128)):
        raise RuntimeError(f"seed {seed}: committed batches do not cover episodes 0..127")
    return [selected[episode] for episode in range(128)]


def analyze(*, runtime: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    """Build paired results, absolute metrics, batch curve, and read-only checkpoint index."""
    common = runtime["common"]
    new_version = frozen["round42"]["training_version"]
    source_version = frozen["round42"]["source_training_version"]
    manifest = _read(ROUND42_OUTPUT / "ROUND42_RAW_ARTIFACT_MANIFEST.json")
    manifest_entries = {item["round_relative_path"].replace("\\", "/"): item
                        for item in manifest["files"]}

    def baseline_rows(pattern: str) -> list[dict[str, Any]]:
        result = []
        for path in sorted((ROUND42_OUTPUT / "rows").glob(pattern)):
            relative = path.relative_to(ROUND42_OUTPUT).as_posix()
            entry = manifest_entries.get(relative)
            if entry is None or _sha256(path) != entry["sha256"]:
                raise RuntimeError(f"ROUND42 baseline manifest mismatch: {relative}")
            result.append(_read(path))
        return result

    refs = baseline_rows("v2_reference_s*_a*.json")
    rules = baseline_rows("v2_r42_rule_*.json")
    old_evals = baseline_rows("v2_r42_eval_new_*.json")
    all_current_rows = [_read(path) for path in sorted(common.ROWS.glob("*.json"))]
    current_rows = [row for row in all_current_rows if row.get("training_version") == new_version]
    training = [row for seed in SEEDS for row in _committed_training_rows(seed)]
    new_evals = []
    for row in current_rows:
        if row.get("kind") != "eval" or row.get("model_arm") != "NEW":
            continue
        seed, mode, scene, attempt = (int(row["training_seed"]), str(row["eval_mode"]),
                                      int(row["scenario_seed"]), int(row["attempt"]))
        logical = f"model_eval_new_s{seed}_{mode}_scene{scene}"
        if _claim_is_complete(logical, attempt):
            new_evals.append(row)
    eval_keys = [(int(row["training_seed"]), str(row["eval_mode"]),
                  int(row["scenario_seed"])) for row in new_evals]
    if len(eval_keys) != len(set(eval_keys)):
        raise RuntimeError("ROUND43 contains multiple successful evaluation attempts for one fixed cell")
    if (len(refs), len(rules), len(old_evals), len(training), len(new_evals)) != (
            64, 32, 96, 384, 96):
        raise RuntimeError(
            "ROUND43 fixed cells incomplete: "
            f"refs/rules/OLD/train/NEW={len(refs)}/{len(rules)}/{len(old_evals)}/"
            f"{len(training)}/{len(new_evals)}")

    for row in refs + rules + old_evals:
        r42_analysis._validate_reward_timing(row)
    for row in current_rows:
        r42_analysis._validate_reward_timing(row)
        if row.get("comm_action_semantics") != "one-shot-send-skip/1":
            raise RuntimeError(f"ROUND43 row has wrong action protocol: {row.get('label')}")

    order = list(common.scenario_order())
    if len(order) != 128 or order != [int(x) for x in frozen["round42"]["training_order"]]:
        raise RuntimeError("ROUND43 active training order differs from the frozen ROUND42 order")
    train_groups: dict[str, Any] = {}
    for seed in SEEDS:
        seed_rows = [row for row in training if int(row["training_seed"]) == seed]
        by_episode = {int(row["episode_index"]): row for row in seed_rows}
        if len(seed_rows) != 128 or set(by_episode) != set(range(128)):
            raise RuntimeError(f"seed {seed}: ROUND43 training rows are incomplete/duplicated")
        for episode, row in by_episode.items():
            if (int(row["scenario_seed"]) != int(order[episode])
                    or row.get("training_version") != new_version
                    or float(row.get("lambda_at_rollout", -1)) < 0):
                raise RuntimeError(f"seed {seed} episode {episode}: training identity/order mismatch")
        train_groups[str(seed)] = r42_analysis._aggregate(seed_rows)

    old_groups = {(int(row["training_seed"]), str(row["eval_mode"])): []
                  for row in old_evals}
    new_groups = {(int(row["training_seed"]), str(row["eval_mode"])): []
                  for row in new_evals}
    for row in old_evals:
        if row.get("training_version") != source_version:
            raise RuntimeError(f"ROUND42 OLD row has unexpected source version: {row.get('label')}")
        old_groups[(int(row["training_seed"]), str(row["eval_mode"]))].append(row)
    for row in new_evals:
        new_groups[(int(row["training_seed"]), str(row["eval_mode"]))].append(row)
        if (row.get("eval_block") != f"one_shot_{row.get('eval_mode')}"
                or row.get("runner_version") != frozen["round43"]["runner_version"]):
            raise RuntimeError(f"ROUND43 eval mode/runner identity mismatch: {row.get('label')}")
    for key, rows in old_groups.items():
        if len(rows) != 16:
            raise RuntimeError(f"ROUND42 OLD evaluation group {key} has {len(rows)} rows")
    for key, rows in new_groups.items():
        if len(rows) != 16:
            raise RuntimeError(f"ROUND43 NEW evaluation group {key} has {len(rows)} rows")

    ref_dev = {int(row["scenario_seed"]): row for row in refs
               if int(row["scenario_seed"]) in set(range(1_301_000, 1_301_016))}
    if len(ref_dev) != 16:
        raise RuntimeError("ROUND43 immediate development reference set is incomplete")
    rules_by_arm: dict[str, list[dict[str, Any]]] = {
        arm: [row for row in rules if row.get("model_arm") == arm]
        for arm in ("late_discovery_gate", "always_hold")
    }
    if any(len(rows) != 16 for rows in rules_by_arm.values()):
        raise RuntimeError("ROUND43 fixed rule development rows are incomplete")

    paired_new_old: dict[str, Any] = {}
    paired_immediate: dict[str, Any] = {}
    paired_rules: dict[str, Any] = {}
    group_summaries: dict[str, Any] = {
        "immediate_reference_development": r42_analysis._aggregate(list(ref_dev.values())),
        "late_discovery_gate": r42_analysis._aggregate(rules_by_arm["late_discovery_gate"]),
        "always_hold": r42_analysis._aggregate(rules_by_arm["always_hold"]),
        "new_training_rollouts": train_groups,
    }
    for seed in SEEDS:
        for mode in ("greedy", "sampled"):
            key = f"s{seed}_{mode}"
            old_rows = old_groups[(seed, mode)]
            new_rows = new_groups[(seed, mode)]
            paired_new_old[key] = r42_analysis._paired_rows(
                old_rows, new_rows, left_name="OLD_round42_new_ep128",
                right_name="NEW_round43_ep128")
            paired_immediate[key] = r42_analysis._paired_rows(
                list(ref_dev.values()), new_rows,
                left_name="immediate", right_name="NEW_round43")
            paired_rules[key] = {
                arm: r42_analysis._paired_rows(
                    rows, new_rows, left_name=arm, right_name="NEW_round43")
                for arm, rows in rules_by_arm.items()
            }
            group_summaries[f"{key}_OLD_round42_new_ep128"] = r42_analysis._aggregate(old_rows)
            group_summaries[f"{key}_NEW_round43_ep128"] = r42_analysis._aggregate(new_rows)

    greedy_seed_verdicts: dict[str, Any] = {}
    for seed in SEEDS:
        key = f"s{seed}_greedy"
        comparison = paired_immediate[key]
        immediate_exact = all(not scene["recovered_task_ids"] and not scene["lost_task_ids"]
                              for scene in comparison["scenes"])
        new_group = group_summaries[f"{key}_NEW_round43_ep128"]
        late_group = group_summaries["late_discovery_gate"]
        greedy_seed_verdicts[str(seed)] = {
            "exact_immediate_on_time_task_set_by_scene": immediate_exact,
            "new_on_time_total": new_group["on_time_total"],
            "immediate_on_time_total": group_summaries[
                "immediate_reference_development"]["on_time_total"],
            "late_on_time_total": late_group["on_time_total"],
            "new_spent_bytes_total": new_group["spent_bytes_total"],
            "late_spent_bytes_total": late_group["spent_bytes_total"],
            "new_mean_spent_bytes_per_scene": new_group["spent_bytes_total"] / 16.0,
            "late_mean_spent_bytes_per_scene": late_group["spent_bytes_total"] / 16.0,
            "cost_below_late_discovery_gate": (
                new_group["spent_bytes_total"] < late_group["spent_bytes_total"]),
            "free_choices": new_group["free_choices_total"],
            "send_now_choices": new_group["send_now_choices_total"],
            "skip_episode_choices": new_group["skip_episode_choices_total"],
        }

    joint_improvement = all(
        item["exact_immediate_on_time_task_set_by_scene"]
        and item["cost_below_late_discovery_gate"]
        for item in greedy_seed_verdicts.values())

    paired_result = {
        "kind": "round43-paired-results/1",
        "training_version": new_version,
        "source_training_version": source_version,
        "comm_action_semantics": "one-shot-send-skip/1",
        "data_scope": {
            "immediate_references_reused": len(refs),
            "late_and_always_hold_rows_reused": len(rules),
            "old_round42_new_ep128_evaluations_reused": len(old_evals),
            "new_training_episodes": len(training),
            "new_ep128_final_evaluations": len(new_evals),
            "development_scene_ids": sorted(ref_dev),
            "independent_test": False,
        },
        "primary_new_minus_old_by_seed_and_mode": paired_new_old,
        "secondary_new_vs_immediate_by_seed_and_mode": paired_immediate,
        "new_vs_fixed_rules_by_seed_and_mode": paired_rules,
        "fixed_rule_absolute_performance": {
            arm: r42_analysis._aggregate(rows) for arm, rows in rules_by_arm.items()},
        "group_summaries": group_summaries,
        "interpretation_limits": [
            "The same 16-scene development block is reused and is not an independent test.",
            "Greedy is primary; sampled results are secondary and remain separate.",
            "No pooled population, significance, formal noninferiority, or generalization claim is made.",
            "A one-shot target-level decision is not a general solution to communication timing.",
            "A GAE change may alter actor signals and both critic targets; it does not prove the sole cause of ROUND42 learning failure.",
        ],
    }

    training_rows_by_key = {(int(row["training_seed"]), int(row["episode_index"])): row
                           for row in training}
    training_curves: dict[str, Any] = {}
    parameter_chain: dict[str, Any] = {}
    for seed in SEEDS:
        chain = []
        batches = []
        previous_parameter = None
        for batch in range(32):
            commit_path = RUN_DIR / "checkpoints" / f"r40_batch_commit_s{seed}_b{batch}.json"
            if not commit_path.is_file():
                raise RuntimeError(f"seed {seed} batch {batch}: batch commit receipt missing")
            commit = _read(commit_path)
            record_path = RUN_DIR / "checkpoints" / commit["batch_record"]
            if _sha256(record_path) != commit["batch_record_sha256"]:
                raise RuntimeError(f"seed {seed} batch {batch}: batch record hash mismatch")
            record = _read(record_path)
            if int(record["batch"]) != batch or int(record["training_seed"]) != seed:
                raise RuntimeError(f"seed {seed} batch {batch}: batch record identity mismatch")
            before = record["controller_parameters_before"]["parameter_sha256"]
            after = record["controller_parameters_after"]["parameter_sha256"]
            if previous_parameter is not None and before != previous_parameter:
                raise RuntimeError(f"seed {seed} batch {batch}: parameter chain is discontinuous")
            previous_parameter = after
            episode_metrics = []
            for episode in record["episodes"]:
                row = training_rows_by_key[(seed, int(episode))]
                episode_metrics.append({
                    "episode_index": int(episode),
                    "scenario_seed": int(row["scenario_seed"]),
                    "on_time": int(row["on_time"]),
                    "spent_bytes": int(row["spent_bytes"]),
                    "lambda_at_rollout": float(row["lambda_at_rollout"]),
                    "action_rng_seed": int(row["action_rng_seed"]),
                })
            batches.append({
                "batch": batch,
                "record_path": str(record_path),
                "record_sha256": _sha256(record_path),
                "episodes": episode_metrics,
                "update": record["update"],
                "constraint": record["constraint"],
                "dual": record["dual"],
                "parameters_before": record["controller_parameters_before"],
                "parameters_after": record["controller_parameters_after"],
                "postupdate_checkpoint": record["postupdate_checkpoint"],
                "postupdate_checkpoint_sha256": commit["postupdate_checkpoint_sha256"],
            })
            chain.append({"batch": batch, "before": before, "after": after,
                          "record_sha256": _sha256(record_path)})
        training_curves[str(seed)] = {"episodes": 128, "batches": batches}
        parameter_chain[str(seed)] = {
            "registered_ep0_parameter_sha256": frozen["round42"]["new_initial_models"][
                str(seed)]["parameter_sha256"],
            "batches": chain,
            "ep128_parameter_sha256": previous_parameter,
            "chain_contiguous": True,
        }

    checkpoint_index: dict[str, Any] = {
        "kind": "round43-checkpoint-index/1",
        "training_version": new_version,
        "read_only_loaded_old_ep128_identity": _read(
            RUN_DIR / "ROUND43_BASELINE_REUSE.json")["checkpoint_content_verification"],
        "registered_ep0_rng_inputs": frozen["round43"]["registered_ep0_inputs"],
        "by_seed": {},
    }
    for seed in SEEDS:
        checkpoint_index["by_seed"][str(seed)] = {}
        for episode in (0, 64, 128):
            candidates = sorted((RUN_DIR / "checkpoints").glob(
                f"v2_train_s{seed}_ep{episode}*.pt"))
            if len(candidates) != 1:
                raise RuntimeError(
                    f"seed {seed} ep{episode}: expected one immutable milestone, found {candidates}")
            path = candidates[0]
            file_sha = _sha256(path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload.get("training_state", {})
            controller, loaded_state, identity = runtime["trainer"].load_checkpoint_read_only(
                path, freeze=runtime["round42_freeze"])
            summary = controller.parameter_summary()
            expected_cursor = episode
            expected_updates = episode // 4
            if (payload.get("training_version") != new_version
                    or int(state.get("episodes_done", -1)) != expected_cursor
                    or int(state.get("updates", -1)) != expected_updates
                    or float(state.get("gae_lambda", -1)) != 1.0
                    or float(state.get("gamma", -1)) != 1.0
                    or int(state.get("training_seed", -1)) != seed
                    or identity.get("optimizer_restored") is not False
                    or identity.get("rng_restored") is not False):
                raise RuntimeError(f"seed {seed} ep{episode}: milestone checkpoint identity mismatch")
            batch_parameter = parameter_chain[str(seed)]["ep128_parameter_sha256"]
            if episode == 128 and summary["parameter_sha256"] != batch_parameter:
                raise RuntimeError(f"seed {seed}: ep128 parameter does not close the batch chain")
            optimizer = payload.get("optimizer", {})
            training_state = dict(state)
            rng = payload.get("rng")
            if not isinstance(rng, dict) or not {"python", "numpy", "torch"}.issubset(rng):
                raise RuntimeError(f"seed {seed} ep{episode}: checkpoint lacks required RNG states")
            checkpoint_index["by_seed"][str(seed)][str(episode)] = {
                "path": str(path), "file_sha256": file_sha,
                "file_size_bytes": path.stat().st_size,
                "parameter_sha256": summary["parameter_sha256"],
                "module_parameter_sha256": runtime["r40_train"]._module_digests(controller),
                "optimizer_state_sha256": runtime["r40_train"]._state_hash(optimizer),
                "optimizer_state_entries": len(optimizer.get("state", {})),
                "rng_state_sources": sorted(rng),
                "rng_state_sha256": runtime["r40_train"]._state_hash(rng),
                "training_state": {key: training_state.get(key) for key in (
                    "training_seed", "episodes_done", "updates", "optimizer_steps",
                    "lambda", "gamma", "gae_lambda", "training_version",
                    "reward_timing_version", "comm_action_semantics", "data_order_seed")},
                "read_only_verification": identity,
                "loaded_state_episode_cursor": int(loaded_state["episodes_done"]),
            }
    checkpoint_index["parameter_chain"] = parameter_chain

    greedy_joint = all(
        item["exact_immediate_on_time_task_set_by_scene"]
        and item["cost_below_late_discovery_gate"]
        for item in greedy_seed_verdicts.values())
    machine_verdict = {
        "kind": "round43-machine-verdict/1",
        "completion": "complete",
        "scientific_result": (
            "development_block_joint_improvement_observed" if greedy_joint
            else "development_block_joint_improvement_not_observed"),
        "greedy_primary": {
            "all_three_seeds_exactly_preserve_immediate_task_set": all(
                item["exact_immediate_on_time_task_set_by_scene"]
                for item in greedy_seed_verdicts.values()),
            "all_three_seeds_below_late_discovery_gate_mean_cost": all(
                item["cost_below_late_discovery_gate"]
                for item in greedy_seed_verdicts.values()),
            "joint_improvement_by_registered_development_rule": greedy_joint,
            "by_seed": greedy_seed_verdicts,
        },
        "sampled_secondary": {
            "reported_separately": True,
            "by_seed": {str(seed): {
                "paired_vs_old": paired_new_old[f"s{seed}_sampled"],
                "vs_immediate": paired_immediate[f"s{seed}_sampled"],
            } for seed in SEEDS},
        },
        "cross_seed_consistency": {
            "greedy_joint_criterion_all_seeds": greedy_joint,
            "seed_count": len(SEEDS),
            "outcome_classes": greedy_seed_verdicts,
        },
        "limits": [
            "This is an exploratory comparison on the already-used 16-scene development block.",
            "No independent confirmation, population no-loss, formal noninferiority, or generalization claim.",
            "The GAE trace change can affect actor credit and critic targets; observed changes do not prove that credit decay was the sole previous cause.",
        ],
    }

    return {
        "paired_results": paired_result,
        "training_curves": {
            "kind": "round43-training-curves/1",
            "training_version": new_version,
            "batch_record_count": sum(len(value["batches"]) for value in training_curves.values()),
            "optimizer_steps_from_records": sum(
                int(batch["update"].get("optimizer_steps", 0))
                for value in training_curves.values() for batch in value["batches"]),
            "by_seed": training_curves,
        },
        "checkpoint_index": checkpoint_index,
        "machine_verdict": machine_verdict,
    }


def write_analysis(runtime: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    result = analyze(runtime=runtime, frozen=frozen)
    _write_new_or_match(RUN_DIR / "ROUND43_PAIRED_RESULTS.json", result["paired_results"])
    _write_new_or_match(RUN_DIR / "ROUND43_TRAINING_CURVES.json", result["training_curves"])
    _write_new_or_match(RUN_DIR / "ROUND43_CHECKPOINT_INDEX.json", result["checkpoint_index"])
    _write_new_or_match(RUN_DIR / "MACHINE_VERDICT.json", result["machine_verdict"])
    return result


__all__ = ["analyze", "write_analysis"]
