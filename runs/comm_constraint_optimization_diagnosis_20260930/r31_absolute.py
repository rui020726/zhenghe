"""ROUND31 §5 绝对表现表 + §8 对偶尺度描述性检查（0 环境启动，纯读取既有落盘件）。

绝对表现覆盖执行依据 §5 列出的全部记录：OLD/NEW × seed11/22/33 × greedy/sampled、
immediate reference、late_discovery_gate_147。每一格都从**行内字段**取数（`on_time`、
`spent_bytes`、`scenario_seed`、`arm`、`training_seed`、`eval_mode`），并用 ep128
milestone 重算的参数哈希与行内 `controller_parameters_after`、batch-31 记录、
CHECKPOINT_INDEX 四方交叉核对——文件名只当候选键，不作结论依据。

§8 只做描述性检查：`J_lambda = -mean(c) + lambda * mean(Y)` 的排序，以及两个已有策略在
任务均值不等时的经验交叉点 `lambda_cross = (c_A - c_B) / (Y_A - Y_B)`。c 一律是
`spent_bytes / B0`（同一计量口径），Y 一律是每 episode 按时任务数。greedy 与 sampled 分表，
任务差为零时不计算有限交叉点。不调参、不反向选 lambda、不称最优。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import r31_common as k

#: ROUND30_REPORT.md 与 PAIRED_COMPARISON.json 里待核对的汇报数字（不是本轮的输入结论）。
REPORTED_BY_ROUND30 = {
    "delta_y_greedy": {"11": 0.0, "22": 0.0, "33": 0.0},
    "delta_c_greedy": {"11": -9.0, "22": 0.0, "33": -334.875},
    "delta_y_sampled": {"11": 0.0, "22": 0.0625, "33": 0.1875},
    "delta_c_sampled": {"11": -57.0, "22": -4197.75, "33": 58.5},
    "absolute_dev": {
        "reference_immediate": {"on_time": 6.0, "bytes": 14939.25},
        "late_discovery_gate_147": {"on_time": 6.0, "bytes": 4838.25},
        "old_s11_greedy": {"on_time": 5.5, "bytes": 28207.875},
        "new_s11_greedy": {"on_time": 5.5, "bytes": 28198.875},
        "old_s22_greedy": {"on_time": 5.5, "bytes": 0.0},
        "new_s22_greedy": {"on_time": 5.5, "bytes": 0.0},
        "old_s33_greedy": {"on_time": 6.0, "bytes": 2398.5},
        "new_s33_greedy": {"on_time": 6.0, "bytes": 2063.625},
    },
}


def census(index: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in index.values():
        key = record["key"]
        if key["kind"] == "eval":
            name = f"eval_{key['arm']}_s{key['seed']}_{key['mode']}"
        elif key["kind"] == "train":
            name = f"train_{key['arm']}_s{key['seed']}"
        elif key["kind"] == "reference":
            name = "reference"
        else:
            name = f"rule_{key['rule']}"
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def expected_census() -> dict[str, int]:
    expected = {"reference": len(k.TRAINING_SCENARIOS) + len(k.DEVELOPMENT_SCENARIOS)}
    expected[f"rule_{k.LATE_RULE_ARM}"] = len(k.DEVELOPMENT_SCENARIOS)
    for arm in k.ARMS:
        for seed in k.TRAIN_SEEDS:
            expected[f"train_{arm}_s{seed}"] = k.EPISODES_PER_SEED
            for mode in k.MODES:
                expected[f"eval_{arm}_s{seed}_{mode}"] = len(k.DEVELOPMENT_SCENARIOS)
    return dict(sorted(expected.items()))


# ------------------------------------------------------------------ 身份核验 -------------
def checkpoint_identity(index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """ep128/ep64/ep0 payload <-> batch records <-> eval rows <-> CHECKPOINT_INDEX."""
    out_index = k.read_json(k.R30_DIR / "CHECKPOINT_INDEX.json")
    cells: dict[str, Any] = {}
    for arm in k.ARMS:
        for seed in k.TRAIN_SEEDS:
            cell = f"{arm}_s{seed}"
            ep128 = k.milestone_path(arm, seed, k.EPISODES_PER_SEED)
            payload = k.load_payload(ep128)
            recomputed = k.parameter_sha256_from_payload(payload)
            state = payload["training_state"]
            batch31 = k.read_json(k.batch_path(arm, seed, 31))
            batch0 = k.read_json(k.batch_path(arm, seed, 0))
            ep0_payload = k.load_payload(k.milestone_path(arm, seed, 0))
            ep0_sha = k.parameter_sha256_from_payload(ep0_payload)
            ep64_sha = k.parameter_sha256_from_payload(k.load_payload(
                k.milestone_path(arm, seed, 64)))

            eval_seen: dict[str, int] = {}
            eval_optimizer_steps: set[int] = set()
            eval_rows = 0
            for record in index.values():
                key = record["key"]
                if key["kind"] != "eval" or key["arm"] != arm or int(key["seed"]) != seed:
                    continue
                eval_rows += 1
                after = (record["row"].get("controller_parameters_after") or {}).get(
                    "parameter_sha256")
                eval_seen[after] = eval_seen.get(after, 0) + 1
                eval_optimizer_steps.add(int(record["row"].get("optimizer_steps", 0)))

            index_entry = out_index.get(cell, {})
            milestone_index = (index_entry.get("milestones") or {}).get("128", {})
            entry: dict[str, Any] = {
                "arm": arm, "training_seed": seed,
                "ep128_path": str(ep128.relative_to(k.ROOT)),
                "ep128_file_sha256": k.sha256_file(ep128),
                "ep128_parameter_sha256_recomputed": recomputed,
                "ep64_parameter_sha256_recomputed": ep64_sha,
                "ep0_parameter_sha256_recomputed": ep0_sha,
                "ep128_training_state": {
                    "episodes_done": int(state["episodes_done"]),
                    "updates": int(state["updates"]),
                    "optimizer_steps": int(state["optimizer_steps"]),
                    "lambda": float(state["lambda"]),
                    "training_seed": int(state["training_seed"]),
                    "trainer_tag": payload["trainer_tag"],
                    "loss_version": state["loss_version"],
                    "state_schema": state["state_schema"],
                    "gae_schema": state["gae_schema"],
                    "optimizer_schema": state["optimizer_schema"],
                    "reward_timing_schema": state["reward_timing_schema"],
                    "b0_bytes": float(state["b0_bytes"]), "delta": float(state["delta"]),
                    "eta_lambda": float(state["eta_lambda"]), "gamma": float(state["gamma"]),
                    "gae_lambda": float(state["gae_lambda"]),
                    "data_order_seed": int(state["data_order_seed"]),
                },
                "freeze_arm_in_payload": payload["freeze"].get("arm"),
                "freeze_sha256_of_this_round": k.sha256_file(
                    k.R30_DIR / "CONFIG_AND_MODEL_MANIFEST.json"),
                "batch31_recorded_parameters_after": batch31["controller_parameters_after"]
                ["parameter_sha256"],
                "batch0_recorded_parameters_before": batch0["controller_parameters_before"]
                ["parameter_sha256"],
                "batch0_recorded_parameters_after": batch0["controller_parameters_after"]
                ["parameter_sha256"],
                "index_ep128_file_sha256": milestone_index.get("sha256"),
                "index_lambda": index_entry.get("lambda"),
                "index_updates": index_entry.get("updates"),
                "index_optimizer_steps": index_entry.get("optimizer_steps"),
                "index_rolling_sha256": index_entry.get("rolling_sha256"),
                "eval_rows_seen": eval_rows,
                "eval_row_parameter_sha256_counts": eval_seen,
                "eval_row_optimizer_steps": sorted(eval_optimizer_steps),
            }
            entry["checks"] = {
                "index_ep128_file_sha_matches_disk": (
                    milestone_index.get("sha256") == entry["ep128_file_sha256"]),
                "recomputed_ep128_sha_matches_batch31_after": (
                    recomputed == entry["batch31_recorded_parameters_after"]),
                "recomputed_ep0_sha_matches_batch0_before": (
                    ep0_sha == entry["batch0_recorded_parameters_before"]),
                "batch0_after_sha_is_not_batch0_before": (
                    entry["batch0_recorded_parameters_after"]
                    != entry["batch0_recorded_parameters_before"]),
                "all_eval_rows_share_ep128_sha": (
                    list(eval_seen.keys()) == [recomputed]
                    and eval_rows == 2 * len(k.DEVELOPMENT_SCENARIOS)),
                "eval_rows_carried_zero_optimizer_steps": eval_optimizer_steps == {0},
                "ep128_state_episodes_updates_lambda_closed": (
                    entry["ep128_training_state"]["episodes_done"] == k.EPISODES_PER_SEED
                    and entry["ep128_training_state"]["updates"] == 32
                    and abs(entry["ep128_training_state"]["lambda"]
                            - float(batch31["dual"]["lambda_next"])) < 1e-12),
                "index_lambda_matches_batch31": (
                    abs(float(index_entry["lambda"]) - float(batch31["dual"]["lambda_next"]))
                    < 1e-12),
                "index_rolling_sha_equals_ep128_sha": (
                    index_entry.get("rolling_sha256") == entry["ep128_file_sha256"]),
                "freeze_arm_matches_cell": entry["freeze_arm_in_payload"] == arm,
                "trainer_contract_is_correctness_v2": (
                    payload["trainer_tag"] == "constrained-cost-v2"
                    and state["loss_version"] == "constrained-lagrangian/2"),
            }
            cells[cell] = entry
    return cells


# ------------------------------------------------------------------ 绝对表现 -------------
def scene_block(rows_by_index: dict[int, dict[str, Any]]) -> dict[str, Any]:
    on_time = [float(row["on_time"]) for row in rows_by_index.values()]
    bytes_ = [float(row["spent_bytes"]) for row in rows_by_index.values()]
    return {
        "records": len(rows_by_index),
        "mean_on_time": round(k.mean(on_time), 9),
        "mean_spent_bytes": round(k.mean(bytes_), 9),
        "mean_normalised_cost_c": round(k.mean(bytes_) / k.B0_BYTES, 9),
        "total_spent_bytes": round(sum(bytes_), 6),
        "on_time_distribution": {str(value): on_time.count(value)
                                 for value in sorted(set(on_time))},
        "per_scene": {str(scenario): {"on_time": int(row["on_time"]),
                                      "spent_bytes": int(row["spent_bytes"]),
                                      "emissions": int(row["emissions"]),
                                      "expired": int(row["expired"]),
                                      "completed": int(row["completed"]),
                                      "discovered": int(row["discovered"]),
                                      "row": str(Path(row["artefacts"]["row"]).name),
                                      "behaviour_sha256": row["behaviour_sha256"]}
                      for scenario, row in sorted(rows_by_index.items())},
    }


def paired_vs_reference(policy: dict[int, dict[str, Any]],
                        reference: dict[int, dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(policy)
    losses = [float(reference[s]["on_time"]) - float(policy[s]["on_time"]) for s in keys]
    return {
        "mean_paired_task_loss_vs_immediate": round(k.mean(losses), 9),
        "scene_level_paired_task_loss": {str(s): round(loss, 9) for s, loss in zip(keys, losses)},
        "counts_vs_immediate_task": {
            "policy_above_reference": sum(1 for v in losses if v < 0),
            "policy_equal_reference": sum(1 for v in losses if v == 0),
            "policy_below_reference": sum(1 for v in losses if v > 0)},
        "scenes_with_task_loss": [str(s) for s, loss in zip(keys, losses) if loss > 0],
        "max_scene_task_loss": max(losses),
    }


def paired_vs_late(policy: dict[int, dict[str, Any]],
                   late: dict[int, dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(policy)
    delta_y = [float(policy[s]["on_time"]) - float(late[s]["on_time"]) for s in keys]
    delta_c = [float(policy[s]["spent_bytes"]) - float(late[s]["spent_bytes"]) for s in keys]
    return {
        "mean_delta_y_vs_late": round(k.mean(delta_y), 9),
        "mean_delta_c_bytes_vs_late": round(k.mean(delta_c), 9),
        "counts_vs_late": {
            "task_better": sum(1 for v in delta_y if v > 0),
            "task_equal": sum(1 for v in delta_y if v == 0),
            "task_worse": sum(1 for v in delta_y if v < 0),
            "cheaper": sum(1 for v in delta_c if v < 0),
            "cost_equal": sum(1 for v in delta_c if v == 0),
            "costlier": sum(1 for v in delta_c if v > 0)},
        "cheaper_but_lost_task_vs_late": [str(s) for s, dy, dc in zip(keys, delta_y, delta_c)
                                          if dc < 0 and dy < 0],
        "scene_level_vs_late": {str(s): {"delta_y": round(dy, 9), "delta_c_bytes": round(dc, 9)}
                               for s, dy, dc in zip(keys, delta_y, delta_c)},
    }


def new_minus_old(old: dict[int, dict[str, Any]], new: dict[int, dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(new)
    delta_y = [float(new[s]["on_time"]) - float(old[s]["on_time"]) for s in keys]
    delta_c = [float(new[s]["spent_bytes"]) - float(old[s]["spent_bytes"]) for s in keys]
    return {
        "delta_y_mean": round(k.mean(delta_y), 9),
        "delta_c_mean_bytes": round(k.mean(delta_c), 9),
        "delta_c_mean_normalised": round(k.mean(delta_c) / k.B0_BYTES, 9),
        "counts_delta_y": {"positive": sum(1 for v in delta_y if v > 0),
                           "zero": sum(1 for v in delta_y if v == 0),
                           "negative": sum(1 for v in delta_y if v < 0)},
        "counts_delta_c": {"cheaper": sum(1 for v in delta_c if v < 0),
                           "equal": sum(1 for v in delta_c if v == 0),
                           "costlier": sum(1 for v in delta_c if v > 0)},
        "scene_level": {str(s): {"delta_y": round(dy, 9), "delta_c_bytes": round(dc, 9)}
                        for s, dy, dc in zip(keys, delta_y, delta_c)},
    }


# ------------------------------------------------------------------ §8 对偶尺度 -------------
def dual_scale(cells: dict[str, dict[str, Any]], label: str) -> dict[str, Any]:
    """J_lambda ordering plus empirical crossover points over an already-observed set."""
    ordered = []
    for name, cell in sorted(cells.items()):
        average_y = float(cell["mean_on_time"])
        average_c = float(cell["mean_normalised_cost_c"])
        entry = {"strategy": name, "mean_on_time_Y": round(average_y, 9),
                 "mean_normalised_cost_c": round(average_c, 9), "source": label}
        for lam in (0.0, 1.0, 1.2375, 1.5, 2.0):
            entry[f"J_lambda_at_{lam:g}"] = round(-average_c + lam * average_y, 9)
        ordered.append(entry)
    crossovers = []
    for i, left in enumerate(ordered):
        for right in ordered[i + 1:]:
            dy = left["mean_on_time_Y"] - right["mean_on_time_Y"]
            dc = left["mean_normalised_cost_c"] - right["mean_normalised_cost_c"]
            if dy == 0:
                crossovers.append({"pair": [left["strategy"], right["strategy"]],
                                   "delta_Y": 0.0, "delta_c": round(dc, 9), "lambda_cross": None,
                                   "note": "task means equal: no finite crossover point, the "
                                           "ordering is decided by cost alone"})
                continue
            crossovers.append({"pair": [left["strategy"], right["strategy"]],
                               "delta_Y": round(dy, 9), "delta_c": round(dc, 9),
                               "lambda_cross": round(dc / dy, 9),
                               "note": "empirical crossover only: at lambda above/below this "
                                       "value the scalar objective flips the pair's order"})
    return {"strategies": ordered, "empirical_crossovers": crossovers,
            "scope": "descriptive ordering of the registered scalar objective over strategies that "
                     "already exist in these artefacts; not an optimal lambda, not a sufficiency "
                     "claim, not a deployment parameter, and no claim that the network can reach "
                     "any listed strategy"}


def main() -> int:
    built = k.build_row_index()
    index = built["index"]

    reference_dev: dict[int, dict[str, Any]] = {}
    reference_train: dict[int, dict[str, Any]] = {}
    late: dict[int, dict[str, Any]] = {}
    for record in index.values():
        key = record["key"]
        if key["kind"] == "reference":
            scenario = int(key["scenario"])
            target = reference_dev if scenario in k.DEVELOPMENT_SCENARIOS else reference_train
            target[scenario] = record["row"]
        elif key["kind"] == "rule" and key["rule"] == k.LATE_RULE_ARM:
            late[int(key["scenario"])] = record["row"]

    eval_by_cell: dict[str, dict[int, dict[str, Any]]] = {}
    train_by_cell: dict[str, dict[int, dict[str, Any]]] = {}
    for record in index.values():
        key = record["key"]
        if key["kind"] == "eval":
            eval_by_cell.setdefault(f"{key['arm']}_s{key['seed']}_{key['mode']}", {})[
                int(key["scenario"])] = record["row"]
        elif key["kind"] == "train":
            train_by_cell.setdefault(f"{key['arm']}_s{key['seed']}", {})[
                int(key["episode"])] = record["row"]

    cells: dict[str, Any] = {}
    for cell, rows in sorted(eval_by_cell.items()):
        entry = scene_block(rows)
        entry["paired_vs_immediate"] = paired_vs_reference(rows, reference_dev)
        entry["vs_late"] = paired_vs_late(rows, late)
        entry["block"] = "development 1301000..1301015, ep128 final checkpoint"
        cells[cell] = entry
    cells["reference_immediate"] = scene_block(reference_dev)
    cells["reference_immediate"]["block"] = "development 1301000..1301015, immediate rule"
    cells["late_discovery_gate_147"] = scene_block(late)
    cells["late_discovery_gate_147"]["block"] = "development 1301000..1301015, late rule"

    new_old_blocks: dict[tuple[str, str], dict[str, Any]] = {}
    for seed in k.TRAIN_SEEDS:
        for mode in k.MODES:
            new_old_blocks[(str(seed), mode)] = new_minus_old(
                eval_by_cell[f"old_s{seed}_{mode}"], eval_by_cell[f"new_s{seed}_{mode}"])
    new_old = {f"{seed}_{mode}": block for (seed, mode), block in sorted(new_old_blocks.items())}

    training_cells: dict[str, Any] = {}
    for cell, rows in sorted(train_by_cell.items()):
        ordered = [rows[episode] for episode in sorted(rows)]
        scenes = [int(row["scenario_seed"]) for row in ordered]
        by_episode = {int(row["episode_index"]): row for row in ordered}
        entry = scene_block(by_episode)
        entry["block"] = "training 1300000..1300047, all 128 episodes in the frozen order"
        entry["distinct_scenes"] = len(set(scenes))
        entry["scene_visit_counts"] = {str(s): scenes.count(s) for s in sorted(set(scenes))}
        entry["paired_task_loss_vs_immediate_all_episodes"] = paired_vs_reference(
            by_episode, {episode: reference_train[scenario]
                         for episode, scenario in enumerate(scenes)})
        training_cells[cell] = entry

    reported: dict[str, Any] = {}
    for (seed, mode), block in sorted(new_old_blocks.items()):
        for quantity, table, tolerance in (("delta_y", "delta_y", 1e-9),
                                           ("delta_c", "delta_c", 1e-6)):
            expected = REPORTED_BY_ROUND30[f"{table}_{mode}"][seed]
            recomputed = (block["delta_y_mean"] if quantity == "delta_y"
                          else block["delta_c_mean_bytes"])
            reported[f"{quantity}_s{seed}_{mode}"] = {
                "reported_by_round30": expected, "recomputed": recomputed,
                "match": abs(expected - recomputed) < tolerance}
    for name, expected in REPORTED_BY_ROUND30["absolute_dev"].items():
        source = cells[name]
        reported[f"absolute_{name}"] = {
            "reported_by_round30": expected,
            "recomputed": {"on_time": source["mean_on_time"],
                           "bytes": source["mean_spent_bytes"]},
            "match": (abs(expected["on_time"] - source["mean_on_time"]) < 1e-9
                      and abs(expected["bytes"] - source["mean_spent_bytes"]) < 1e-6)}

    #: 与 ROUND30 自己存档的 PAIRED_COMPARISON.json 逐格核对（复用既有统计件，不换方法）。
    paired = k.read_json(k.R30_DIR / "PAIRED_COMPARISON.json")
    for name, comparison in sorted(paired["comparisons"].items()):
        mode = comparison["mode"]
        seed = str(comparison["training_seed"])
        block = new_old_blocks[(seed, mode)]
        vs = comparison["vs_immediate"]
        checks = {
            "delta_y": (comparison["delta_y"]["mean"], block["delta_y_mean"], 1e-9),
            "delta_c": (comparison["delta_c"]["mean"], block["delta_c_mean_bytes"], 1e-6),
            "reference_on_time": (vs["reference_mean_on_time"],
                                  cells["reference_immediate"]["mean_on_time"], 1e-9),
            "reference_bytes": (vs["reference_mean_bytes"],
                                cells["reference_immediate"]["mean_spent_bytes"], 1e-6),
            "late_on_time": (vs["late_mean_on_time"],
                             cells["late_discovery_gate_147"]["mean_on_time"], 1e-9),
            "late_bytes": (vs["late_mean_bytes"],
                           cells["late_discovery_gate_147"]["mean_spent_bytes"], 1e-6),
        }
        for arm in ("old", "new"):
            cell = f"{arm}_s{seed}_{mode}"
            checks[f"{arm}_on_time"] = (vs[f"{arm}_mean_on_time"],
                                        cells[cell]["mean_on_time"], 1e-9)
            checks[f"{arm}_bytes"] = (vs[f"{arm}_mean_bytes"],
                                      cells[cell]["mean_spent_bytes"], 1e-6)
        for field, (expected, recomputed, tolerance) in checks.items():
            reported[f"paired_json_{name}_{field}"] = {
                "reported_by_round30": expected, "recomputed": recomputed,
                "match": abs(expected - recomputed) < tolerance}
        dy_counts = comparison["delta_y"]["counts"]
        reported[f"paired_json_{name}_delta_y_counts"] = {
            "reported_by_round30": dy_counts, "recomputed": block["counts_delta_y"],
            "match": dy_counts == block["counts_delta_y"]}
        dc_counts = comparison["delta_c"]["counts"]
        reported[f"paired_json_{name}_delta_c_counts"] = {
            "reported_by_round30": dc_counts, "recomputed": block["counts_delta_c"],
            "match": dc_counts == block["counts_delta_c"]}

    dev_dual: dict[str, Any] = {}
    for mode in k.MODES:
        subset = {name: value for name, value in cells.items() if name.endswith(f"_{mode}")}
        subset["rule_immediate"] = cells["reference_immediate"]
        subset["rule_late_discovery_gate_147"] = cells["late_discovery_gate_147"]
        dev_dual[mode] = dual_scale(
            {name: {"mean_on_time": value["mean_on_time"],
                    "mean_normalised_cost_c": value["mean_normalised_cost_c"]}
             for name, value in subset.items()},
            "development block 1301000..1301015, ep128 final records")
    training_subset = {name: {"mean_on_time": value["mean_on_time"],
                              "mean_normalised_cost_c": value["mean_normalised_cost_c"]}
                       for name, value in training_cells.items()}
    training_subset["rule_immediate"] = {
        "mean_on_time": round(k.mean([float(row["on_time"])
                                      for row in reference_train.values()]), 9),
        "mean_normalised_cost_c": round(k.mean([float(row["spent_bytes"])
                                                for row in reference_train.values()]) / k.B0_BYTES,
                                        9)}

    payload = {
        "round": "31",
        "section": "absolute performance rebuilt from ROUND30 final records (0 launches)",
        "source": {
            "branch_of_round30": "exp/comm-deadline-encoding-ablation",
            "round30_report_commit": "915868297c08d4b01b7dfeff0ea810d5394057a3",
            "round30_dir": str(k.R30_DIR.relative_to(k.ROOT)),
            "this_branch": "analysis/comm-constraint-optimization-diagnosis",
            "b0_bytes": k.B0_BYTES, "delta": k.DELTA,
        },
        "identity": {
            "row_census": {"files": built["files"], "parsed": built["parsed"],
                           "unparsed": built["unparsed"], "counts": census(index),
                           "expected_counts": expected_census(),
                           "counts_match": census(index) == expected_census()},
            "row_identity_problems": built["identity_problems"],
            "row_identity_problems_count": len(built["identity_problems"]),
            "checkpoint_identity": checkpoint_identity(index),
        },
        "development_block": {"scenes": sorted(reference_dev), "cells": cells},
        "delta_new_minus_old_recomputed": new_old,
        "reconciliation_with_round30_report": reported,
        "training_block_cells": training_cells,
        "dual_scale_descriptive": {
            "definition": {"c": "mean(spent_bytes)/B0 with B0=10000 (same metering convention)",
                           "Y": "mean on-time tasks per episode",
                           "J_lambda": "-c + lambda*Y",
                           "lambda_cross": "(c_A - c_B)/(Y_A - Y_B), only when Y_A != Y_B",
                           "lambda_values_used_for_ordering": [0.0, 1.0, 1.2375, 1.5, 2.0]},
            "greedy": dev_dual["greedy"],
            "sampled": dev_dual["sampled"],
            "training_block": dual_scale(training_subset,
                                         "training block 1300000..1300047, all 128 episodes"),
            "shared_comparators": "immediate and late carry the same reference and delta for "
                                  "every candidate on one scene set, so they are common constants "
                                  "in the Lagrangian; they are also listed as observed strategies",
            "warning": "this orders strategies that already exist; it is not an optimal lambda, "
                       "not a sufficiency claim, not a deployment parameter, and no claim that "
                       "the network can reach any listed strategy",
        },
        "limits": [
            "the 16 development scenes were reused by earlier rounds: descriptive only, no "
            "confirmatory significance, no delta=0 deployment guarantee",
            "one rollout per scene per mode: the sampled column carries its own sampling variance; "
            "ROUND30's frozen intervals are cited as-is and no new statistical method is fitted",
            "the three training seeds are reported separately and never pooled",
            "training-block means are on 128 episodes with scenario repeats from the frozen order; "
            "they are not independent of the development numbers",
        ],
    }
    out = k.write_json("ABSOLUTE_PERFORMANCE.json", payload)
    print(json.dumps({
        "rows": built["files"], "identity_problems": len(built["identity_problems"]),
        "counts_match": census(index) == expected_census(),
        "reference_immediate": [cells["reference_immediate"]["mean_on_time"],
                                cells["reference_immediate"]["mean_spent_bytes"]],
        "late": [cells["late_discovery_gate_147"]["mean_on_time"],
                 cells["late_discovery_gate_147"]["mean_spent_bytes"]],
        "cells": {name: [value["mean_on_time"], value["mean_spent_bytes"],
                         value["paired_vs_immediate"]["mean_paired_task_loss_vs_immediate"]]
                  for name, value in sorted(cells.items())
                  if name not in ("reference_immediate", "late_discovery_gate_147")},
        "reconciliation_all_match": all(v["match"] for v in reported.values()),
        "reconciliation_failures": [name for name, value in reported.items()
                                    if not value["match"]],
        "checkpoint_check_failures": {cell: [name for name, ok in checks.items() if not ok]
                                      for cell, checks in
                                      ((c, payload["identity"]["checkpoint_identity"][c]["checks"])
                                       for c in sorted(payload["identity"]["checkpoint_identity"]))
                                      if [name for name, ok in checks.items() if not ok]},
    }, ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
