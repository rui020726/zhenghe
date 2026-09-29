"""ROUND31 §6 六条运行的约束与 λ 轨迹重建（0 环境启动，只读既有落盘件）。

覆盖 OLD/NEW × seed11/22/33 的全部 32 批。每批既从 ROUND30 的 batch 记录取登记值，也从
**独立来源重算**：

- reference 的按时任务数取自该场景的 immediate reference 行（`r30_reference_s*_a1.json`）；
- policy 的按时任务数与已计量字节取自该 episode 的训练行（`r30_train_*_e*_s*_a*.json`）；
- actor 样本数、masked/forced/actionable 候选数从该 episode 的 tape 逐行重数；
- `g_hat`、`lambda_next` 按登记公式重算（`constraint_statistic` / `dual_step` 的同一数学，
  这里独立实现后与记录逐批比对）。

时序以冻结契约为准：`r30_train.train_arm_seed` 在一批内先以 `lam = state["lambda"]` 跑完 4 集
rollout，再 `ppo_update`，最后才 `dual_update` —— 所以**该批 actor 用的是更新前的 λ**，
`lambda_next` 只从下一批的 rollout 起生效。本文件核对该关系，不自行换时序。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import r31_common as k

FIXED_BATCHES = (0, 15, 31)


def load_train_rows() -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[str, Any]]:
    """Every ROUND30 training row keyed by (arm, training_seed, episode_index)."""
    built = k.build_row_index()
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for record in built["index"].values():
        key = record["key"]
        if key["kind"] == "train":
            rows[(key["arm"], int(key["seed"]), int(key["episode"]))] = record["row"]
    return rows, built


def reference_on_time() -> dict[int, float]:
    out: dict[int, float] = {}
    for path in sorted(k.R30_ROWS.glob("r30_reference_s*_a1.json")):
        row = k.read_json(path)
        scenario = int(row["scenario_seed"])
        if scenario in out and out[scenario] != float(row["on_time"]):
            raise RuntimeError(f"scenario {scenario} has two different cached references")
        out[scenario] = float(row["on_time"])
    return out


def recount_tapes(tape_file: Path) -> dict[str, Any]:
    decision_steps = 0
    samples_with_actionable = 0
    samples_without_actionable = 0
    actionable_candidates = 0
    masked_candidates = 0
    forced_candidates = 0
    total_candidates = 0
    wanted_send = 0
    held = 0
    steps: list[int] = []
    for tape in k.iter_jsonl(tape_file):
        decision_steps += 1
        targets = list(tape["targets"])
        free = set(tape["actionable"])
        total_candidates += len(targets)
        actionable_candidates += len(free)
        masked_candidates += len(tape["masked"])
        forced_candidates += len(tape["forced"])
        if free:
            samples_with_actionable += 1
        else:
            samples_without_actionable += 1
        for target in targets:
            if target in tape["requested"] and tape["requested"][target]:
                wanted_send += 1
        held += sum(1 for target in targets
                    if int(tape["actions"].get(target, 0)) == 0)
        steps.append(int(tape["step"]))
    return {
        "decision_steps": decision_steps,
        "actor_samples_with_actionable": samples_with_actionable,
        "actor_samples_without_actionable": samples_without_actionable,
        "candidates_total": total_candidates,
        "actionable_candidates": actionable_candidates,
        "masked_candidates": masked_candidates,
        "forced_candidates": forced_candidates,
        "requested_sends": wanted_send,
        "hold_actions": held,
        "first_decision_step": min(steps) if steps else None,
        "last_decision_step": max(steps) if steps else None,
    }


def is_finite(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def scan_non_finite(obj: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(scan_non_finite(value, f"{path}.{key}"))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(scan_non_finite(value, f"{path}[{index}]"))
    elif isinstance(obj, float) and not math.isfinite(obj):
        found.append(path)
    return found


def constraint_statistic(y_reference: list[float], y_policy: list[float],
                         delta: float) -> float:
    """Independent implementation of `cost_math.constraint_statistic`."""
    return (sum(y_reference) / len(y_reference)) - (sum(y_policy) / len(y_policy)) - delta


def dual_step(lam: float, g_hat: float, eta: float) -> float:
    """Independent implementation of `cost_math.dual_step`."""
    return max(0.0, lam + eta * g_hat)


def main() -> int:
    rows, built = load_train_rows()
    references = reference_on_time()
    order = k.c30.scenario_order()

    runs: dict[str, Any] = {}
    for arm in k.ARMS:
        for seed in k.TRAIN_SEEDS:
            cell = f"{arm}_s{seed}"
            batches: list[dict[str, Any]] = []
            problems: list[str] = []
            tape_cache: dict[int, dict[str, Any]] = {}
            previous_next = None
            for index in range(32):
                record = k.read_json(k.batch_path(arm, seed, index))
                episodes = [int(e) for e in record["episodes"]]
                scenarios = [int(s) for s in record["scenarios"]]
                cell_rows = []
                for episode in episodes:
                    row = rows.get((arm, seed, episode))
                    if row is None:
                        problems.append(f"batch {index}: training row e{episode} is missing")
                        continue
                    cell_rows.append(row)
                recomputed_scenarios = [int(row["scenario_seed"]) for row in cell_rows]
                order_scenarios = [order[episode] for episode in episodes]
                if scenarios != recomputed_scenarios:
                    problems.append(f"batch {index}: scenarios {scenarios} != rows "
                                    f"{recomputed_scenarios}")
                if scenarios != order_scenarios:
                    problems.append(f"batch {index}: scenarios {scenarios} != frozen order "
                                    f"{order_scenarios}")

                y_reference = [float(references[s]) for s in scenarios]
                y_policy = [float(row["on_time"]) for row in cell_rows]
                g_recomputed = constraint_statistic(y_reference, y_policy, k.DELTA)
                g_recorded = float(record["constraint"]["g_hat"])
                lam_before = float(record["dual"]["lambda_before"])
                lam_next_recorded = float(record["dual"]["lambda_next"])
                lam_next_recomputed = dual_step(lam_before, g_recorded, k.ETA_LAMBDA)
                lam_rollout = float(record["lambda_at_rollout"])
                if abs(g_recomputed - g_recorded) > 1e-12:
                    problems.append(f"batch {index}: g_hat recorded {g_recorded} != recomputed "
                                    f"{g_recomputed}")
                if abs(lam_next_recorded - lam_next_recomputed) > 1e-12:
                    problems.append(f"batch {index}: lambda_next recorded {lam_next_recorded} != "
                                    f"max(0, lambda_before + eta*g_hat) = "
                                    f"{lam_next_recomputed}")
                if abs(lam_rollout - lam_before) > 1e-12:
                    problems.append(f"batch {index}: lambda_at_rollout {lam_rollout} != "
                                    f"lambda_before {lam_before} (the batch's actor would then "
                                    f"not be using the pre-update lambda)")
                if previous_next is not None and abs(lam_before - previous_next) > 1e-12:
                    problems.append(f"batch {index}: lambda_before {lam_before} != previous "
                                    f"lambda_next {previous_next}")
                if index == 0 and abs(lam_before - k.LAMBDA_INIT) > 1e-12:
                    problems.append(f"batch 0: lambda_before {lam_before} != registered "
                                    f"lambda_initial {k.LAMBDA_INIT}")
                previous_next = lam_next_recorded

                # pairing recorded inside the constraint block must match the rows it names
                pairs = record["constraint"]["pairs"]
                for pair, scenario in zip(pairs, scenarios):
                    if int(pair["scenario"]) != scenario:
                        problems.append(f"batch {index}: pair scenario {pair['scenario']} != "
                                        f"{scenario}")
                    if abs(float(pair["y_reference"]) - float(references[scenario])) > 1e-12:
                        problems.append(f"batch {index}: pair y_reference for {scenario} != the "
                                        f"cached immediate reference row")

                spent = [float(row["spent_bytes"]) for row in cell_rows]
                normalised = [value / k.B0_BYTES for value in spent]
                tape_counts = {"decision_steps": 0, "actor_samples_with_actionable": 0,
                               "actor_samples_without_actionable": 0, "candidates_total": 0,
                               "actionable_candidates": 0, "masked_candidates": 0,
                               "forced_candidates": 0, "requested_sends": 0, "hold_actions": 0}
                for episode in episodes:
                    row = rows.get((arm, seed, episode))
                    if row is None:
                        continue
                    tape_file = Path(row["artefacts"]["tape"])
                    if episode not in tape_cache:
                        tape_cache[episode] = recount_tapes(tape_file)
                    counts = tape_cache[episode]
                    for field, value in counts.items():
                        if isinstance(value, int) and field in tape_counts:
                            tape_counts[field] += value
                    summary = row.get("collector_summary") or {}
                    for ours, theirs in (("decision_steps", "decision_steps"),
                                         ("actionable_candidates", "actionable_total"),
                                         ("masked_candidates", "masked_total"),
                                         ("forced_candidates", "forced_total"),
                                         ("candidates_total", "candidates_total")):
                        if theirs in summary and int(summary[theirs]) != int(counts[ours]):
                            problems.append(f"batch {index} e{episode}: tape recount {ours}"
                                            f"={counts[ours]} != row collector_summary "
                                            f"{theirs}={summary[theirs]}")
                    if abs(float(row["lambda"]) - lam_rollout) > 1e-12:
                        problems.append(f"batch {index} e{episode}: row lambda {row['lambda']} "
                                        f"!= batch lambda_at_rollout {lam_rollout}")
                update = record.get("update") or {}
                samples_recorded = int(update.get("samples", -1))
                if samples_recorded != tape_counts["decision_steps"]:
                    problems.append(f"batch {index}: update.samples {samples_recorded} != tape "
                                    f"decision steps {tape_counts['decision_steps']}")
                alignment = update.get("likelihood_alignment") or {}
                if not alignment.get("passed", False):
                    problems.append(f"batch {index}: likelihood alignment did not pass")
                shapes_update = record.get("optimizer_parameter_shapes")
                shapes_controller = record.get("controller_parameter_shapes")
                if shapes_update != shapes_controller:
                    problems.append(f"batch {index}: optimizer parameter set differs from the "
                                    f"controller parameter set")
                non_finite = scan_non_finite({
                    "lambda_at_rollout": lam_rollout, "dual": record["dual"],
                    "constraint": record["constraint"], "update": update})
                if non_finite:
                    problems.append(f"batch {index}: non-finite values at {non_finite}")

                batches.append({
                    "batch": index,
                    "episodes": episodes,
                    "scenarios": scenarios,
                    "y_reference": y_reference,
                    "y_policy": y_policy,
                    "paired_task_loss_by_scene": [round(r - p, 9)
                                                  for r, p in zip(y_reference, y_policy)],
                    "g_hat_recorded": g_recorded,
                    "g_hat_recomputed": round(g_recomputed, 12),
                    "lambda_before": lam_before,
                    "lambda_after_recorded": lam_next_recorded,
                    "lambda_after_recomputed": round(lam_next_recomputed, 12),
                    "lambda_at_rollout": lam_rollout,
                    "actor_used_pre_update_lambda": abs(lam_rollout - lam_before) < 1e-12,
                    "lambda_step_sign": record["dual"]["direction"],
                    "spent_bytes_by_episode": spent,
                    "cost_c_by_episode": [round(v, 9) for v in normalised],
                    "cost_c_batch_mean": round(k.mean(normalised), 9),
                    "cost_c_batch_sum": round(sum(normalised), 9),
                    "task_reward_total_by_episode": [int(row["task_reward_total"])
                                                     for row in cell_rows],
                    "emissions_by_episode": [int(row["emissions"]) for row in cell_rows],
                    "tape_counts": tape_counts,
                    "update_samples_recorded": samples_recorded,
                    "update_minibatches_recorded": int(update.get("minibatches", -1)),
                    "optimizer_steps_recorded": int(update.get("optimizer_steps", -1)),
                    "epochs_recorded": int(update.get("epochs", -1)),
                    "policy_surrogate": update.get("policy_surrogate"),
                    "value_loss_task": update.get("value_loss_task"),
                    "value_loss_cost": update.get("value_loss_cost"),
                    "entropy": update.get("entropy"),
                    "approx_kl": update.get("approx_kl"),
                    "clip_fraction": update.get("clip_fraction"),
                    "gradient_norm_mean": update.get("gradient_norm_mean"),
                    "module_gradient_l1_mean": update.get("module_gradient_l1_mean"),
                    "likelihood_max_abs_gap": alignment.get("max_abs_gap"),
                    "unassigned_prefix_task_reward": update.get("unassigned_prefix_task_reward"),
                    "unassigned_prefix_cost_bytes": update.get("unassigned_prefix_cost_bytes"),
                    "parameters_before_sha256": record["controller_parameters_before"]
                    ["parameter_sha256"],
                    "parameters_after_sha256": record["controller_parameters_after"]
                    ["parameter_sha256"],
                    "non_finite_paths": non_finite,
                })

            lam_values = [b["lambda_after_recorded"] for b in batches]
            g_values = [b["g_hat_recomputed"] for b in batches]
            all_rows = [rows[(arm, seed, e)] for e in range(k.EPISODES_PER_SEED)
                        if (arm, seed, e) in rows]
            losses_all = [float(references[int(row["scenario_seed"])]) - float(row["on_time"])
                          for row in all_rows]
            first8 = batches[:8]
            last8 = batches[-8:]

            chain_closed = all(batches[i + 1]["lambda_before"] == batches[i]
                               ["lambda_after_recorded"] for i in range(31))
            weight_chain = all(batches[i]["parameters_after_sha256"]
                               == batches[i + 1]["parameters_before_sha256"] for i in range(31))
            summary = {
                "lambda_initial": batches[0]["lambda_before"],
                "lambda_final": batches[-1]["lambda_after_recorded"],
                "lambda_min": min(lam_values),
                "lambda_max": max(lam_values),
                "lambda_range": round(max(lam_values) - min(lam_values), 12),
                "lambda_history_in_checkpoint_matches_batches": None,
                "batches_with_positive_g_hat": sum(1 for v in g_values if v > 0),
                "batches_with_zero_g_hat": sum(1 for v in g_values if v == 0),
                "batches_with_negative_g_hat": sum(1 for v in g_values if v < 0),
                "g_hat_mean": round(k.mean(g_values), 12),
                "g_hat_first8_mean": round(k.mean([b["g_hat_recomputed"] for b in first8]), 12),
                "g_hat_last8_mean": round(k.mean([b["g_hat_recomputed"] for b in last8]), 12),
                "mean_paired_task_loss_all_training_episodes": round(k.mean(losses_all), 9),
                "training_episodes": len(all_rows),
                "first8_mean_paired_task_loss": round(k.mean(
                    [loss for b in first8 for loss in b["paired_task_loss_by_scene"]]), 9),
                "last8_mean_paired_task_loss": round(k.mean(
                    [loss for b in last8 for loss in b["paired_task_loss_by_scene"]]), 9),
                "first8_mean_cost_c_per_episode": round(k.mean(
                    [v for b in first8 for v in b["cost_c_by_episode"]]), 9),
                "last8_mean_cost_c_per_episode": round(k.mean(
                    [v for b in last8 for v in b["cost_c_by_episode"]]), 9),
                "first8_total_spent_bytes": round(
                    sum(v for b in first8 for v in b["spent_bytes_by_episode"]), 6),
                "last8_total_spent_bytes": round(
                    sum(v for b in last8 for v in b["spent_bytes_by_episode"]), 6),
                "optimizer_steps_total": sum(b["optimizer_steps_recorded"] for b in batches),
                "actor_samples_total": sum(b["tape_counts"]["decision_steps"] for b in batches),
                "actor_samples_without_actionable_total": sum(
                    b["tape_counts"]["actor_samples_without_actionable"] for b in batches),
                "masked_candidates_total": sum(b["tape_counts"]["masked_candidates"]
                                               for b in batches),
                "actionable_candidates_total": sum(b["tape_counts"]["actionable_candidates"]
                                                   for b in batches),
                "candidates_total": sum(b["tape_counts"]["candidates_total"] for b in batches),
                "dual_updates_performed": len(batches),
                "lambda_chain_closed": chain_closed,
                "weight_chain_closed": weight_chain,
                "non_finite_found": [b["non_finite_paths"] for b in batches
                                     if b["non_finite_paths"]],
                "batches_with_task_loss": sum(1 for b in batches
                                              if any(v > 0 for v in b["paired_task_loss_by_scene"])),
                "batches_with_g_hat_zero": sum(1 for b in batches if b["g_hat_recomputed"] == 0),
            }
            # the checkpoint's own lambda_history must describe the same 32 batches
            state = k.load_payload(k.milestone_path(arm, seed, k.EPISODES_PER_SEED)
                                   )["training_state"]
            history = state["lambda_history"]
            summary["lambda_history_in_checkpoint_matches_batches"] = (
                len(history) == 32 and all(
                    int(entry["batch"]) == batches[n]["batch"]
                    and abs(float(entry["g_hat"]) - batches[n]["g_hat_recomputed"]) < 1e-12
                    and abs(float(entry["lambda_before"]) - batches[n]["lambda_before"]) < 1e-12
                    and abs(float(entry["lambda_next"]) - batches[n]["lambda_after_recorded"])
                    < 1e-12
                    for n, entry in enumerate(history)))
            runs[cell] = {"arm": arm, "training_seed": seed, "batches": batches,
                          "summary": summary, "problems": problems}

    payload = {
        "round": "31",
        "section": "constraint and dual-variable trajectories rebuilt from ROUND30 batch "
                   "records, training rows, reference rows and tapes (0 launches, 0 optimizer "
                   ".step)",
        "contract": {
            "objective": "c = C/B0; g = E[Y_ref] - E[Y_policy] - delta; L_max = -E[c] - lambda*g; "
                         "A_actor = lambda*A_task - A_cost",
            "dual": "lambda_next = max(0, lambda_before + eta_lambda * g_hat)",
            "parameters": {"B0_bytes": k.B0_BYTES, "delta": k.DELTA,
                           "lambda_initial": k.LAMBDA_INIT, "eta_lambda": k.ETA_LAMBDA,
                           "episodes_per_batch": k.EPISODES_PER_UPDATE},
            "lambda_timing": "r30_train.train_arm_seed freezes lam = state['lambda'] for the "
                             "whole batch, runs the four rollouts, calls ppo_update, and only "
                             "then dual_update; so every batch's actor is trained with the "
                             "PRE-update lambda and lambda_next takes effect from the next "
                             "batch's rollout. Verified field-by-field, not assumed.",
            "reference_is_a_scalar": "batch_g_hat reads the cached immediate on-time count as a "
                                     "plain float, so it carries no gradient into the actor",
            "source_of_truth": "v2_trainer (correctness-v2) + cost_math (Stage A), as frozen and "
                               "referenced by ROUND30; nothing here redefines the training math",
        },
        "inputs": {
            "batch_records": 6 * 32,
            "training_rows_used": len(rows),
            "reference_rows_used": len(references),
            "row_identity_problems_from_census": len(built["identity_problems"]),
            "tape_files_read": sum(len(r["batches"]) * k.EPISODES_PER_UPDATE for r in runs.values()),
            "non_finite_checks": "every lambda/g_hat/update field of all 192 batch records",
        },
        "runs": runs,
        "cross_run": {},
        "reading_notes": [
            "the training trajectory is the SAMPLED policy's constraint trajectory; the greedy "
            "decode has its own numbers only in ABSOLUTE_PERFORMANCE.json and may not be read off "
            "these g_hat values",
            "g_hat here is a 4-episode paired point estimate per batch; a batch mean of 0 can "
            "still contain individual scenes with loss and others with gain",
            "a curve that is still moving at batch 31 only shows the run had not stabilised; it "
            "does not establish that more training would help",
        ],
    }

    for seed in k.TRAIN_SEEDS:
        old_run, new_run = runs[f"old_s{seed}"], runs[f"new_s{seed}"]
        payload["cross_run"][f"s{seed}"] = {
            "identical_g_hat_sequence": [b["g_hat_recomputed"] for b in old_run["batches"]]
                                         == [b["g_hat_recomputed"] for b in new_run["batches"]],
            "identical_lambda_sequence": [b["lambda_after_recorded"] for b in old_run["batches"]]
                                         == [b["lambda_after_recorded"]
                                             for b in new_run["batches"]],
            "first_batch_index_where_lambda_differs": next(
                (i for i in range(32)
                 if abs(old_run["batches"][i]["lambda_after_recorded"]
                        - new_run["batches"][i]["lambda_after_recorded"]) > 1e-12), None),
            "identical_scenario_sequence": [b["scenarios"] for b in old_run["batches"]]
                                           == [b["scenarios"] for b in new_run["batches"]],
            "identical_optimizer_step_total": old_run["summary"]["optimizer_steps_total"]
                                              == new_run["summary"]["optimizer_steps_total"],
            "lambda_final_old": old_run["summary"]["lambda_final"],
            "lambda_final_new": new_run["summary"]["lambda_final"],
        }

    out = k.write_json("DUAL_TRAJECTORIES.json", payload)
    print(json.dumps({
        "runs": {cell: {
            "lambda": [run["summary"]["lambda_initial"], run["summary"]["lambda_final"]],
            "g_positive_zero_negative": [run["summary"]["batches_with_positive_g_hat"],
                                         run["summary"]["batches_with_zero_g_hat"],
                                         run["summary"]["batches_with_negative_g_hat"]],
            "mean_paired_task_loss_all_episodes":
                run["summary"]["mean_paired_task_loss_all_training_episodes"],
            "first8_last8_task_loss": [run["summary"]["first8_mean_paired_task_loss"],
                                       run["summary"]["last8_mean_paired_task_loss"]],
            "first8_last8_cost_c_per_episode": [run["summary"]["first8_mean_cost_c_per_episode"],
                                                run["summary"]["last8_mean_cost_c_per_episode"]],
            "optimizer_steps": run["summary"]["optimizer_steps_total"],
            "samples_no_actionable": run["summary"]["actor_samples_without_actionable_total"],
            "chain_closed": [run["summary"]["lambda_chain_closed"],
                             run["summary"]["weight_chain_closed"],
                             run["summary"]["lambda_history_in_checkpoint_matches_batches"]],
            "problems": len(run["problems"]),
        } for cell, run in sorted(runs.items())},
        "cross_run": payload["cross_run"],
    }, ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
