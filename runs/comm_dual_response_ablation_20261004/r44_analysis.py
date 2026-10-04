"""ROUND44 paired development analysis from immutable R44 and verified reused rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from r44_eval_index import SCENES, SEEDS, MODES
from r44_prepare import ETA_NEW, RUN_DIR, RUNNER_VERSION, TRAINING_VERSION, read, sha256


def _write_new(path: Path, value: Any) -> None:
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"ROUND44 result already exists with different content: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()


def _claim_path(logical_job: str, attempt: int) -> Path:
    import re
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", logical_job).strip("_.")
    return RUN_DIR / "launch_claims" / slug / f"attempt_{attempt:04d}.json"


def _claim_complete(logical_job: str, attempt: int) -> dict[str, Any]:
    path = _claim_path(logical_job, attempt)
    if not path.is_file():
        raise RuntimeError(f"ROUND44 row lacks its durable claim: {logical_job} attempt {attempt}")
    claim = read(path)
    if (claim.get("status") != "complete" or claim.get("round") != 44
            or claim.get("training_version") != TRAINING_VERSION
            or claim.get("runner_version") != RUNNER_VERSION
            or claim.get("comm_action_semantics") != "one-shot-send-skip/1"):
        raise RuntimeError(f"ROUND44 completed claim identity mismatch: {path}")
    return claim


def _committed_training(runtime: dict[str, Any], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_episode: dict[int, dict[str, Any]] = {}
    batch_records: list[dict[str, Any]] = []
    checkpoint_dir = RUN_DIR / "checkpoints"
    for batch in range(32):
        commit_path = checkpoint_dir / f"r40_batch_commit_s{seed}_b{batch}.json"
        if not commit_path.is_file():
            raise RuntimeError(f"seed {seed} batch {batch}: committed batch receipt missing")
        commit = read(commit_path)
        record_path = checkpoint_dir / str(commit["batch_record"])
        if not record_path.is_file() or sha256(record_path) != commit["batch_record_sha256"]:
            raise RuntimeError(f"seed {seed} batch {batch}: immutable batch record hash mismatch")
        postupdate_path = checkpoint_dir / str(commit["postupdate_checkpoint"])
        if (not postupdate_path.is_file()
                or sha256(postupdate_path) != commit["postupdate_checkpoint_sha256"]):
            raise RuntimeError(f"seed {seed} batch {batch}: postupdate checkpoint hash mismatch")
        record = read(record_path)
        dual = record.get("dual", {})
        if (int(record.get("training_seed", -1)) != seed
                or int(record.get("batch", -1)) != batch
                or record.get("training_version") != TRAINING_VERSION
                or not abs(float(dual.get("eta", float("nan"))) - ETA_NEW) < 1e-12
                or record.get("episodes_done_after") != (batch + 1) * 4
                or record.get("updates_after") != batch + 1):
            raise RuntimeError(f"seed {seed} batch {batch}: version/eta/cursor mismatch")
        if not abs(float(dual["lambda_next"])
                   - max(0.0, float(dual["lambda_before"]) + ETA_NEW * float(dual["g_hat"]))) < 1e-10:
            raise RuntimeError(f"seed {seed} batch {batch}: recorded dual update is not eta=0.5")
        before = record["controller_parameters_before"]["parameter_sha256"]
        after = record["controller_parameters_after"]["parameter_sha256"]
        if commit.get("controller_parameters", {}).get("parameter_sha256") != after:
            raise RuntimeError(f"seed {seed} batch {batch}: commit receipt parameter mismatch")
        batch_records.append({
            "batch": batch, "path": str(record_path),
            "sha256": sha256(record_path), "record": record,
        })
        for artifact in record["episode_artifacts"]:
            episode = int(artifact["episode_index"])
            row_path = Path(artifact["row_path"])
            tape_path = Path(artifact["tape_path"])
            if (not row_path.is_file() or sha256(row_path) != artifact["row_sha256"]
                    or not tape_path.is_file() or sha256(tape_path) != artifact["tape_sha256"]):
                raise RuntimeError(f"seed {seed} episode {episode}: committed row/tape hash mismatch")
            row = read(row_path)
            logical = f"new_training_s{seed}_e{episode}"
            claim = _claim_complete(logical, int(row["attempt"]))
            if (episode in rows_by_episode or row.get("kind") != "train"
                    or row.get("training_version") != TRAINING_VERSION
                    or row.get("runner_version") != RUNNER_VERSION
                    or float(row.get("eta_lambda", float("nan"))) != ETA_NEW
                    or row.get("comm_action_semantics") != "one-shot-send-skip/1"
                    or claim.get("logical_job") != logical
                    or claim.get("completed_steps") != 1000):
                raise RuntimeError(f"seed {seed} episode {episode}: training row/claim identity mismatch")
            rows_by_episode[episode] = row
    if set(rows_by_episode) != set(range(128)):
        raise RuntimeError(f"seed {seed}: committed training episodes do not cover 0..127")
    order = list(runtime["common"].scenario_order())
    if len(order) != 128:
        raise RuntimeError("ROUND44 scenario schedule does not contain exactly 128 entries")
    for episode, row in rows_by_episode.items():
        if int(row["scenario_seed"]) != int(order[episode]):
            raise RuntimeError(f"seed {seed} episode {episode}: scenario-order mismatch")
    return [rows_by_episode[episode] for episode in range(128)], batch_records


def _reuse_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    reuse = read(RUN_DIR / "ROUND44_BASELINE_REUSE.json")
    refs = [read(Path(item["path"])) for item in reuse["round42_immediate_references"]]
    rules = [read(Path(item["path"])) for item in reuse["round42_fixed_rules"]]
    old = [read(Path(item["path"])) for item in reuse["r43_new_ep128_eval_rows_as_old"]]
    for item, row in zip(reuse["round42_immediate_references"], refs):
        if sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("reused immediate reference changed after the prelaunch audit")
    for item, row in zip(reuse["round42_fixed_rules"], rules):
        if sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("reused fixed rule changed after the prelaunch audit")
    for item, row in zip(reuse["r43_new_ep128_eval_rows_as_old"], old):
        if sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("reused ROUND43 OLD evaluation row changed after preflight")
    return refs, rules, old


def analyze(*, runtime: dict[str, Any], frozen: dict[str, Any],
            eval_audit: dict[str, Any]) -> dict[str, Any]:
    import r42_analysis

    refs, rules, old_evals = _reuse_rows()
    training: list[dict[str, Any]] = []
    batch_records_by_seed: dict[str, list[dict[str, Any]]] = {}
    parameter_chain: dict[str, Any] = {}
    for seed in SEEDS:
        rows, records = _committed_training(runtime, seed)
        training.extend(rows)
        batch_records_by_seed[str(seed)] = records
        previous = frozen["round42"]["new_initial_models"][str(seed)]["parameter_sha256"]
        chain = []
        for item in records:
            record = item["record"]
            before = record["controller_parameters_before"]["parameter_sha256"]
            after = record["controller_parameters_after"]["parameter_sha256"]
            if before != previous:
                raise RuntimeError(f"seed {seed} batch {item['batch']}: parameter chain is discontinuous")
            previous = after
            chain.append({"batch": item["batch"], "before": before, "after": after,
                          "record_sha256": item["sha256"]})
        parameter_chain[str(seed)] = {
            "registered_ep0_parameter_sha256": frozen["round42"]["new_initial_models"][
                str(seed)]["parameter_sha256"],
            "batches": chain, "ep128_parameter_sha256": previous,
            "chain_contiguous": True,
        }

    new_evals = [read(Path(entry["row_path"])) for entry in eval_audit["entries"]]
    for row in refs + rules + old_evals + training + new_evals:
        r42_analysis._validate_reward_timing(row)
    for row in old_evals:
        if (row.get("training_version") != frozen["round42"]["old_training_version"]
                or row.get("comm_action_semantics") != "one-shot-send-skip/1"):
            raise RuntimeError(f"ROUND44 OLD row identity mismatch: {row.get('label')}")
    for row in training:
        if float(row.get("eta_lambda", float("nan"))) != ETA_NEW:
            raise RuntimeError(f"ROUND44 training row has wrong eta: {row.get('label')}")
    for row in new_evals:
        if (row.get("training_version") != TRAINING_VERSION
                or float(row.get("eta_lambda", float("nan"))) != ETA_NEW
                or row.get("runner_version") != RUNNER_VERSION):
            raise RuntimeError(f"ROUND44 NEW eval version/eta mismatch: {row.get('label')}")

    if (len(refs), len(rules), len(old_evals), len(training), len(new_evals)) != (
            64, 32, 96, 384, 96):
        raise RuntimeError("ROUND44 paired matrix has a missing or extra fixed group")
    order = [int(v) for v in runtime["common"].scenario_order()]
    if order != [int(v) for v in frozen["round42"]["training_order"]]:
        raise RuntimeError("ROUND44 actual training order differs from frozen R43 order")

    dev_set = set(SCENES)
    immediate_by_scene = {int(row["scenario_seed"]): row for row in refs
                          if int(row["scenario_seed"]) in dev_set}
    if set(immediate_by_scene) != dev_set:
        raise RuntimeError("ROUND44 reused immediate development set is incomplete")
    rules_by_arm = {arm: [row for row in rules if row.get("model_arm") == arm]
                    for arm in ("late_discovery_gate", "always_hold")}
    if any(len(rows) != 16 for rows in rules_by_arm.values()):
        raise RuntimeError("ROUND44 reused late/hold rules do not cover 16 scenes each")

    old_groups = {(seed, mode): [] for seed in SEEDS for mode in MODES}
    new_groups = {(seed, mode): [] for seed in SEEDS for mode in MODES}
    for row in old_evals:
        old_groups[(int(row["training_seed"]), str(row["eval_mode"]))].append(row)
    for row in new_evals:
        new_groups[(int(row["training_seed"]), str(row["eval_mode"]))].append(row)
    if any(len(group) != 16 for group in (*old_groups.values(), *new_groups.values())):
        raise RuntimeError("ROUND44 OLD/NEW eval group does not contain exactly 16 scenes")

    refs_dev = list(immediate_by_scene.values())
    group_summaries: dict[str, Any] = {
        "immediate_reference_development": r42_analysis._aggregate(refs_dev),
        "late_discovery_gate": r42_analysis._aggregate(rules_by_arm["late_discovery_gate"]),
        "always_hold": r42_analysis._aggregate(rules_by_arm["always_hold"]),
        "new_training_rollouts": {
            str(seed): r42_analysis._aggregate([row for row in training
                                                if int(row["training_seed"]) == seed])
            for seed in SEEDS
        },
    }
    paired_old: dict[str, Any] = {}
    paired_immediate: dict[str, Any] = {}
    paired_rules: dict[str, Any] = {}
    for seed in SEEDS:
        for mode in MODES:
            key = f"s{seed}_{mode}"
            left, right = old_groups[(seed, mode)], new_groups[(seed, mode)]
            paired_old[key] = r42_analysis._paired_rows(
                left, right, left_name="OLD_R43_NEW_EP128", right_name="NEW_R44_EP128")
            paired_immediate[key] = r42_analysis._paired_rows(
                refs_dev, right, left_name="immediate", right_name="NEW_R44")
            paired_rules[key] = {
                arm: r42_analysis._paired_rows(
                    rows, right, left_name=arm, right_name="NEW_R44")
                for arm, rows in rules_by_arm.items()
            }
            group_summaries[f"{key}_OLD_R43_NEW_EP128"] = r42_analysis._aggregate(left)
            group_summaries[f"{key}_NEW_R44_EP128"] = r42_analysis._aggregate(right)

    greedy_by_seed = {}
    for seed in SEEDS:
        key = f"s{seed}_greedy"
        exact = all(not scene["recovered_task_ids"] and not scene["lost_task_ids"]
                    for scene in paired_immediate[key]["scenes"])
        new_summary = group_summaries[f"{key}_NEW_R44_EP128"]
        old_summary = group_summaries[f"{key}_OLD_R43_NEW_EP128"]
        late = group_summaries["late_discovery_gate"]
        greedy_by_seed[str(seed)] = {
            "exact_immediate_on_time_task_set_by_scene": exact,
            "new_on_time_total": new_summary["on_time_total"],
            "old_r43_on_time_total": old_summary["on_time_total"],
            "immediate_on_time_total": group_summaries[
                "immediate_reference_development"]["on_time_total"],
            "late_on_time_total": late["on_time_total"],
            "new_spent_bytes_total": new_summary["spent_bytes_total"],
            "old_r43_spent_bytes_total": old_summary["spent_bytes_total"],
            "late_spent_bytes_total": late["spent_bytes_total"],
            "new_mean_spent_bytes_per_scene": new_summary["spent_bytes_total"] / 16.0,
            "late_mean_spent_bytes_per_scene": late["spent_bytes_total"] / 16.0,
            "cost_below_late_discovery_gate": (
                new_summary["spent_bytes_total"] < late["spent_bytes_total"]),
            "free_choices": new_summary["free_choices_total"],
            "send_now_choices": new_summary["send_now_choices_total"],
            "skip_episode_choices": new_summary["skip_episode_choices_total"],
        }
    joint = all(row["exact_immediate_on_time_task_set_by_scene"]
                and row["cost_below_late_discovery_gate"]
                for row in greedy_by_seed.values())

    training_curves_by_seed = {}
    for seed in SEEDS:
        episode_by_index = {int(row["episode_index"]): row for row in training
                            if int(row["training_seed"]) == seed}
        batches = []
        for item in batch_records_by_seed[str(seed)]:
            record = item["record"]
            episodes = []
            for episode in record["episodes"]:
                row = episode_by_index[int(episode)]
                episodes.append({
                    "episode_index": int(episode),
                    "scenario_seed": int(row["scenario_seed"]),
                    "on_time": int(row["on_time"]),
                    "spent_bytes": int(row["spent_bytes"]),
                    "lambda_at_rollout": float(row["lambda_at_rollout"]),
                    "action_rng_seed": int(row["action_rng_seed"]),
                })
            batches.append({
                "batch": item["batch"], "record_path": item["path"],
                "record_sha256": item["sha256"], "episodes": episodes,
                "update": record["update"], "constraint": record["constraint"],
                "dual": record["dual"],
                "parameters_before": record["controller_parameters_before"],
                "parameters_after": record["controller_parameters_after"],
                "postupdate_checkpoint": record["postupdate_checkpoint"],
                "postupdate_checkpoint_sha256": record.get("postupdate_checkpoint_sha256"),
            })
        training_curves_by_seed[str(seed)] = {"episodes": 128, "batches": batches}

    checkpoint_index = {
        "kind": "round44-checkpoint-index/1",
        "training_version": TRAINING_VERSION,
        "source_ep0_rng_inputs": frozen["round43"]["registered_ep0_inputs"],
        "read_only_loaded_old_ep128": frozen["round42"]["old_models"],
        "by_seed": {}, "parameter_chain": parameter_chain,
    }
    for seed in SEEDS:
        checkpoint_index["by_seed"][str(seed)] = {}
        for episode in (0, 64, 128):
            matches = sorted((RUN_DIR / "checkpoints").glob(
                f"v2_train_s{seed}_ep{episode}*.pt"))
            if len(matches) != 1:
                raise RuntimeError(f"seed {seed} ep{episode}: expected one milestone checkpoint")
            path = matches[0]
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload.get("training_state", {})
            controller, loaded_state, identity = runtime["trainer"].load_checkpoint_read_only(
                path, freeze=frozen["round42"]["freeze_identity"])
            summary = controller.parameter_summary()
            if (payload.get("training_version") != TRAINING_VERSION
                    or float(payload.get("eta_lambda", float("nan"))) != ETA_NEW
                    or state.get("training_version") != TRAINING_VERSION
                    or float(state.get("eta_lambda", float("nan"))) != ETA_NEW
                    or int(state.get("episodes_done", -1)) != episode
                    or int(state.get("updates", -1)) != episode // 4
                    or int(state.get("training_seed", -1)) != seed
                    or identity.get("optimizer_restored") is not False
                    or identity.get("rng_restored") is not False):
                raise RuntimeError(f"seed {seed} ep{episode}: checkpoint identity mismatch")
            if episode == 0 and summary["parameter_sha256"] != frozen["round42"][
                    "new_initial_models"][str(seed)]["parameter_sha256"]:
                raise RuntimeError(f"seed {seed}: ep0 parameter differs from registered initial model")
            if episode == 128 and summary["parameter_sha256"] != parameter_chain[str(seed)][
                    "ep128_parameter_sha256"]:
                raise RuntimeError(f"seed {seed}: ep128 checkpoint does not close parameter chain")
            optimizer_state = payload.get("optimizer", {})
            rng = payload.get("rng", {})
            checkpoint_index["by_seed"][str(seed)][str(episode)] = {
                "path": str(path.resolve()), "file_sha256": sha256(path),
                "file_size_bytes": path.stat().st_size,
                "parameter_sha256": summary["parameter_sha256"],
                "module_parameter_sha256": runtime["r40_train"]._module_digests(controller),
                "optimizer_state_sha256": runtime["r40_train"]._state_hash(optimizer_state),
                "optimizer_state_entries": len(optimizer_state.get("state", {})),
                "rng_sources": sorted(rng),
                "rng_state_sha256": runtime["r40_train"]._state_hash(rng),
                "training_state": {key: state.get(key) for key in (
                    "training_seed", "episodes_done", "updates", "optimizer_steps", "lambda",
                    "eta_lambda", "gamma", "gae_lambda", "training_version",
                    "reward_timing_version", "comm_action_semantics", "data_order_seed")},
                "read_only_verification": identity,
                "loaded_state_episode_cursor": int(loaded_state["episodes_done"]),
            }

    paired = {
        "kind": "round44-paired-results/1",
        "training_version": TRAINING_VERSION,
        "source_training_version": frozen["round42"]["old_training_version"],
        "comm_action_semantics": "one-shot-send-skip/1",
        "scientific_variable": {"eta_lambda_old": 0.05, "eta_lambda_new": 0.5,
                                "gae_lambda": 1.0, "gamma": 1.0},
        "data_scope": {"reused_immediate_references": 64,
                       "reused_late_and_hold_rule_rows": 32,
                       "old_r43_new_ep128_evaluations": 96,
                       "new_training_episodes": len(training),
                       "new_ep128_final_evaluations": len(new_evals),
                       "development_scene_ids": list(SCENES),
                       "independent_test": False},
        "new_minus_old_r43_by_seed_and_mode": paired_old,
        "new_vs_immediate_by_seed_and_mode": paired_immediate,
        "new_vs_fixed_rules_by_seed_and_mode": paired_rules,
        "fixed_rule_absolute_performance": {
            arm: r42_analysis._aggregate(rows) for arm, rows in rules_by_arm.items()},
        "group_summaries": group_summaries,
        "greedy_primary_verdict_by_seed": greedy_by_seed,
        "registered_joint_improvement_observed": joint,
        "interpretation_limits": [
            "Results use the already-used 16-scene development block.",
            "Greedy is primary and sampled is secondary; seeds and modes are not pooled.",
            "No formal noninferiority, population zero-loss, or generalization claim is made.",
            "Changing eta_lambda tests dual response speed; an outcome does not identify a sole cause of prior learning failure.",
        ],
    }
    curves = {
        "kind": "round44-training-curves/1", "training_version": TRAINING_VERSION,
        "eta_lambda": ETA_NEW, "batch_record_count": 96,
        "optimizer_steps_from_records": sum(
            int(batch["update"].get("optimizer_steps", 0))
            for value in training_curves_by_seed.values() for batch in value["batches"]),
        "by_seed": training_curves_by_seed,
    }
    verdict = {
        "kind": "round44-machine-verdict/1", "completion": "complete",
        "scientific_result": (
            "development_block_joint_improvement_observed" if joint
            else "development_block_joint_improvement_not_observed"),
        "registered_primary_rule": {
            "all_three_greedy_seeds_exactly_preserve_immediate_task_identity_set": all(
                item["exact_immediate_on_time_task_set_by_scene"]
                for item in greedy_by_seed.values()),
            "all_three_greedy_seeds_cost_below_late": all(
                item["cost_below_late_discovery_gate"] for item in greedy_by_seed.values()),
            "joint_improvement": joint,
            "by_seed": greedy_by_seed,
        },
        "sampled_secondary": {"reported_separately": True,
                               "by_seed_and_mode": {
                                   key: value for key, value in paired_old.items()
                                   if key.endswith("_sampled")}},
        "cross_seed_consistency": {"greedy_joint_rule_all_seeds": joint,
                                   "seed_count": 3, "per_seed": greedy_by_seed},
        "limits": paired["interpretation_limits"],
    }
    return {"paired_results": paired, "training_curves": curves,
            "checkpoint_index": checkpoint_index, "machine_verdict": verdict}


def write_analysis(*, runtime: dict[str, Any], frozen: dict[str, Any],
                   eval_audit: dict[str, Any]) -> dict[str, Any]:
    result = analyze(runtime=runtime, frozen=frozen, eval_audit=eval_audit)
    _write_new(RUN_DIR / "ROUND44_PAIRED_RESULTS.json", result["paired_results"])
    _write_new(RUN_DIR / "ROUND44_TRAINING_CURVES.json", result["training_curves"])
    _write_new(RUN_DIR / "ROUND44_CHECKPOINT_INDEX.json", result["checkpoint_index"])
    _write_new(RUN_DIR / "MACHINE_VERDICT.json", result["machine_verdict"])
    return result


__all__ = ["analyze", "write_analysis"]

