"""ROUND31 §7 真实更新中的任务与成本信号（固定批 0/15/31；0 环境启动、0 optimizer.step）。

样本重建**直接调用冻结训练器自己的 `v2_trainer.batch_samples` / `_unassigned_prefix_totals`**，
输入是从磁盘读回的 tape 与训练行——不复制第二套 GAE/优势数学，也不重新实现奖励归属。
`A_task` 与 `A_cost` 的分量用同一批冻结原语（`cost_math.gae` + `cost_math.actor_advantage`）
分别算出，再断言 `lambda*A_task - A_cost` 逐样本复现训练器记录的合并优势。

梯度诊断的**范围**（执行依据 §7 要求写明）：
- 只有批 0 的**更新前权重**确实落盘（ep0 milestone = 登记初值，`r30_train.save_milestone`
  在首次训练前写一次且永不覆盖），所以只有批 0 能做 `autograd.grad` 分量分解；
- 分解点是该批 epoch 0 / minibatch 0 —— 此时 `old_log_prob == new_log_prob`、`ratio` 恰为 1，
  `torch.min` 两支与 `clamp` 支在该点的 Jacobian 相同，`.mean()` 无样本权重、无 advantage
  标准化，所以 `λ*A_task` 与 `A_cost` 两项可以用**同一组实际缩放**精确分解；
- 批 15 与批 31 的更新前权重未落盘（滚动 checkpoint 每批覆盖），按 §7 记为 missing：
  不用 ep64/ep128 的更新后权重冒充，也不重放 optimizer 恢复；
- 全程只 forward 与 `autograd.grad`，从不 `optimizer.step()`，不写回任何 .pt。
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import r31_common as k
import cost_math as cm
import v2_trainer as trainer
from onpolicy import action_entropy, joint_log_prob

FIXED_BATCHES = (0, 15, 31)


class _CollectorShim:
    """The minimal surface `batch_samples` reads from a collector: `.records` only."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records


def build_episodes(arm: str, seed: int, record: dict[str, Any],
                   rows: dict[tuple[str, int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild the batch's episode dicts exactly as the frozen trainer consumed them."""
    episodes = []
    for episode_index in [int(e) for e in record["episodes"]]:
        row = rows[(arm, seed, episode_index)]
        episodes.append({
            "collector": _CollectorShim(list(k.iter_jsonl(Path(row["artefacts"]["tape"])))),
            "cost_stream": row["cost_stream"],
            "task_stream": row["task_stream"],
            "steps": int(row["steps"]),
            "lambda": float(row["lambda"]),
            "episode_index": int(row["episode_index"]),
            "scenario_seed": int(row["scenario_seed"]),
            "on_time": int(row["on_time"]),
            "spent_bytes": int(row["spent_bytes"]),
            "artefacts": row["artefacts"],
        })
    return episodes


def advantage_components(episodes: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    """A_task and A_cost per sample, from the same frozen primitives the trainer calls."""
    task: list[float] = []
    cost: list[float] = []
    for episode in episodes:
        tapes = sorted(episode["collector"].records, key=lambda tape: int(tape["step"]))
        steps = [int(tape["step"]) for tape in tapes]
        horizon = int(episode["steps"])
        cost_stream = {int(step): float(row["airtime_bytes"])
                       for step, row in episode["cost_stream"].items()}
        task_stream = {int(step): float(value) for step, value in episode["task_stream"].items()}
        bundled_cost, _ = trainer._bundle(cost_stream, steps, horizon)
        bundled_task, _ = trainer._bundle(task_stream, steps, horizon)
        rewards_cost = torch.tensor([bundled_cost[s] / k.B0_BYTES for s in steps],
                                    dtype=torch.float64)
        rewards_task = torch.tensor([bundled_task[s] for s in steps], dtype=torch.float64)
        values_task = torch.tensor([float(tape["value"]) for tape in tapes], dtype=torch.float64)
        values_cost = torch.tensor([float(tape["cost_value"]) for tape in tapes],
                                   dtype=torch.float64)
        terminal = torch.zeros(len(steps), dtype=torch.bool)
        terminal[-1] = True
        bootstrap_task = torch.zeros_like(values_task)
        bootstrap_cost = torch.zeros_like(values_cost)
        if len(steps) > 1:
            bootstrap_task[:-1] = values_task[1:]
            bootstrap_cost[:-1] = values_cost[1:]
        adv_task, _ = cm.gae(rewards_task, values_task, gamma=float(k.v2c.GAMMA),
                             gae_lambda=float(k.v2c.GAE_LAMBDA), terminal=terminal,
                             bootstrap=bootstrap_task)
        adv_cost, _ = cm.gae(rewards_cost, values_cost, gamma=float(k.v2c.GAMMA),
                             gae_lambda=float(k.v2c.GAE_LAMBDA), terminal=terminal,
                             bootstrap=bootstrap_cost)
        task.extend(float(value) for value in adv_task)
        cost.extend(float(value) for value in adv_cost)
    return task, cost


def chunk_groups(n_samples: int, update_index: int) -> list[list[int]]:
    """Reproduce the trainer's per-batch shuffle and its minibatch index groups."""
    shuffle = random.Random(int(hashlib.sha256(
        f"{int(k.c30.DATA_ORDER_SEED)}:{int(update_index)}".encode("utf-8")).hexdigest()[:16],
        16))
    order = list(range(n_samples))
    shuffle.shuffle(order)
    size = int(k.v2c.MINIBATCH_STEPS)
    return [order[start:start + size] for start in range(0, len(order), size)]


def likelihood_recompute(controller: Any, samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Independent re-run of v2_trainer.likelihood_alignment's number (pure forward, no grad)."""
    gaps: list[float] = []
    with torch.no_grad():
        for sample in samples:
            logits, _values = controller.forward(
                torch.as_tensor(sample["features"], dtype=torch.float32))
            recomputed = float(joint_log_prob(
                logits,
                torch.tensor([sample["actions"][t] for t in sample["targets"]], dtype=torch.long),
                torch.tensor([1.0 if t in sample["free"] else 0.0 for t in sample["targets"]],
                             dtype=torch.float32)))
            gaps.append(abs(recomputed - float(sample["log_prob"])))
    return {"samples": len(samples), "max_abs_gap": max(gaps) if gaps else 0.0,
            "tolerance": float(k.v2c.LOG_PROB_TOLERANCE)}


def gradient_report(controller: Any, samples: list[dict[str, Any]], chunk: list[int],
                    lam: float) -> dict[str, Any]:
    """Split the first minibatch's gradient at old=new into the registered components."""
    chunk_samples = [samples[index] for index in chunk]
    stacked = torch.as_tensor(np.concatenate([item["features"] for item in chunk_samples],
                                             axis=0))
    hidden = controller.inner.trunk(stacked)
    logits = controller.inner.action_head(hidden)
    values_task = controller.inner.value_head(hidden).squeeze(-1)
    values_cost = controller.cost_head(hidden).squeeze(-1)

    new_log_prob: list[torch.Tensor] = []
    entropy: list[torch.Tensor] = []
    pooled_task: list[torch.Tensor] = []
    pooled_cost: list[torch.Tensor] = []
    row = 0
    for item in chunk_samples:
        count = len(item["targets"])
        rows = torch.arange(row, row + count)
        row += count
        picked = torch.tensor([item["actions"][t] for t in item["targets"]], dtype=torch.long)
        usable = torch.tensor([1.0 if t in item["free"] else 0.0 for t in item["targets"]],
                              dtype=torch.float32)
        new_log_prob.append(joint_log_prob(logits[rows], picked, usable))
        entropy.append(action_entropy(logits[rows], usable))
        pooled_task.append(values_task[rows].mean())
        pooled_cost.append(values_cost[rows].mean())
    new_log_prob_t = torch.stack(new_log_prob)
    entropy_t = torch.stack(entropy)
    old_log_prob = torch.tensor([item["log_prob"] for item in chunk_samples],
                                dtype=torch.float32)
    advantage_task = torch.tensor([item["advantage_task"] for item in chunk_samples],
                                  dtype=torch.float32)
    advantage_cost = torch.tensor([item["advantage_cost"] for item in chunk_samples],
                                  dtype=torch.float32)
    advantage_combined = torch.tensor([item["advantage"] for item in chunk_samples],
                                      dtype=torch.float32)
    target_task = torch.tensor([item["return_task"] for item in chunk_samples],
                               dtype=torch.float32)
    target_cost = torch.tensor([item["return_cost"] for item in chunk_samples],
                                dtype=torch.float32)

    ratio = torch.exp(new_log_prob_t - old_log_prob)
    combined_from_parts = lam * advantage_task - advantage_cost
    combination_error = float(torch.max((combined_from_parts - advantage_combined).abs()).item())

    surrogate_task = ratio * (lam * advantage_task)
    surrogate_cost = ratio * (-advantage_cost)
    surrogate_as_trainer = torch.min(ratio * advantage_combined, torch.clamp(
        ratio, 1.0 - float(k.v2c.CLIP), 1.0 + float(k.v2c.CLIP)) * advantage_combined)
    value_task_loss = ((torch.stack(pooled_task) - target_task) ** 2).mean()
    value_cost_loss = ((torch.stack(pooled_cost) - target_cost) ** 2).mean()
    loss_task = -surrogate_task.mean()
    loss_cost = -surrogate_cost.mean()
    loss_entropy = -float(k.v2c.ENTROPY_BONUS) * entropy_t.mean()
    loss_critic_task = float(k.v2c.VALUE_COEFFICIENT) * value_task_loss
    loss_critic_cost = float(k.v2c.VALUE_COEFFICIENT) * value_cost_loss
    loss_parts_sum = loss_task + loss_cost + loss_entropy + loss_critic_task + loss_critic_cost
    loss_as_trainer = (-surrogate_as_trainer.mean()
                       + float(k.v2c.VALUE_COEFFICIENT) * (value_task_loss + value_cost_loss)
                       - float(k.v2c.ENTROPY_BONUS) * entropy_t.mean())

    parameters = list(controller.parameters)
    modules = (("trunk", controller.inner.trunk), ("action_head", controller.inner.action_head),
               ("task_value_head", controller.inner.value_head), ("cost_head", controller.cost_head))

    def grads_of(loss: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True))

    parts = {"policy_task_lambda_A_task": loss_task, "policy_cost_minus_A_cost": loss_cost,
             "entropy_bonus": loss_entropy, "critic_task": loss_critic_task,
             "critic_cost": loss_critic_cost, "sum_of_the_five_parts": loss_parts_sum,
             "trainer_expression": loss_as_trainer}
    gradients = {name: grads_of(loss) for name, loss in parts.items()}

    def flatten(name: str) -> torch.Tensor:
        return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().reshape(-1)
                          for g, p in zip(gradients[name], parameters)])

    flat = {name: flatten(name) for name in parts}

    def by_module(name: str) -> dict[str, Any]:
        vector = flat[name]
        out: dict[str, Any] = {}
        offset = 0
        for prefix, module in modules:
            end = offset + sum(int(p.numel()) for p in module.parameters())
            piece = vector[offset:end]
            offset = end
            out[prefix] = {"l1": round(float(piece.abs().sum().item()), 12),
                           "l2": round(float(piece.norm().item()), 12)}
        return out

    closure_error = float((flat["sum_of_the_five_parts"]
                           - (flat["policy_task_lambda_A_task"]
                              + flat["policy_cost_minus_A_cost"] + flat["entropy_bonus"]
                              + flat["critic_task"] + flat["critic_cost"])).abs().max().item())
    trainer_agreement = float((flat["sum_of_the_five_parts"]
                               - flat["trainer_expression"]).abs().max().item())
    policy_total = (flat["policy_task_lambda_A_task"] + flat["policy_cost_minus_A_cost"])
    pre_clip_norm = float(flat["sum_of_the_five_parts"].norm().item())
    max_grad_norm = float(k.v2c.MAX_GRAD_NORM)
    task_norm = float(flat["policy_task_lambda_A_task"].norm().item())
    cost_norm = float(flat["policy_cost_minus_A_cost"].norm().item())
    cosine = (float(torch.dot(flat["policy_task_lambda_A_task"],
                              flat["policy_cost_minus_A_cost"]).item())
              / (task_norm * cost_norm)) if task_norm > 0 and cost_norm > 0 else None
    actionable_in_chunk = sum(1 for item in chunk_samples if item["free"])

    return {
        "scope": "epoch 0 / minibatch 0 of this batch only -- the update start where old == new "
                 "(ratio exactly 1). Later minibatches are NOT reconstructed: that would need "
                 "replaying optimizer.step, which this round forbids",
        "registered_scalars_used": {"minibatch_steps": int(k.v2c.MINIBATCH_STEPS),
                                    "epochs_per_batch": int(k.v2c.EPOCHS_PER_BATCH),
                                    "clip": float(k.v2c.CLIP),
                                    "value_coefficient_per_head": float(k.v2c.VALUE_COEFFICIENT),
                                    "entropy_bonus": float(k.v2c.ENTROPY_BONUS),
                                    "max_grad_norm": max_grad_norm,
                                    "lambda_of_this_batch": lam},
        "chunk_samples": len(chunk_samples),
        "chunk_first_indices": chunk[:5],
        "chunk_samples_with_actionable_candidates": actionable_in_chunk,
        "chunk_samples_with_zero_actor_gradient": len(chunk_samples) - actionable_in_chunk,
        "reduction": "surrogate.mean() and entropy.mean() over the chunk, uniform sample weights, "
                     "no advantage standardisation anywhere (v2_trainer registers this as the "
                     "explicit difference from the old trainer)",
        "ratio_max_abs_deviation_from_one": round(
            float((ratio.detach() - 1.0).abs().max().item()), 12),
        "advantage_combination_max_abs_error": combination_error,
        "advantage_combination_closed": bool(combination_error < 1e-5),
        "component_l2": {name: round(float(vector.norm().item()), 12)
                         for name, vector in flat.items()},
        "component_l1": {name: round(float(vector.abs().sum().item()), 12)
                         for name, vector in flat.items()},
        "component_by_module": {name: by_module(name) for name in parts},
        "closure_error_max_abs": closure_error,
        "parts_sum_vs_trainer_expression_max_abs": trainer_agreement,
        "policy_task_versus_policy_cost": {
            "l2_task": round(task_norm, 12), "l2_cost": round(cost_norm, 12),
            "l2_task_over_l2_cost": (round(task_norm / cost_norm, 9) if cost_norm > 0 else None),
            "cosine": (round(cosine, 9) if cosine is not None else None),
            "policy_l2": round(float(policy_total.norm().item()), 12),
            "note": "magnitudes are pre-clip; the trainer then clips the TOTAL norm and Adam "
                    "rescales per parameter, so these ratios describe the gradient direction at "
                    "the update start, not the parameter step",
        },
        "pre_clip_total_norm": round(pre_clip_norm, 9),
        "clipping_triggered_at_this_minibatch": bool(pre_clip_norm > max_grad_norm),
        "clipping_scale_if_applied": (round(max_grad_norm / pre_clip_norm, 9)
                                      if pre_clip_norm > max_grad_norm else 1.0),
        "critic_share_of_total_l2": round(
            float((flat["critic_task"] + flat["critic_cost"]).norm().item()) / pre_clip_norm, 9)
        if pre_clip_norm > 0 else None,
        "entropy_share_of_total_l2": round(
            float(flat["entropy_bonus"].norm().item()) / pre_clip_norm, 9)
        if pre_clip_norm > 0 else None,
    }


def advantage_block(samples: list[dict[str, Any]], lam: float) -> dict[str, Any]:
    """Distribution of the registered signals actually fed to the actor for this batch."""
    task = [float(s["advantage_task"]) for s in samples]
    cost = [float(s["advantage_cost"]) for s in samples]
    scaled = [lam * value for value in task]
    combined = [float(s["advantage"]) for s in samples]
    combination_error = max((abs(lam * t - c - a) for t, c, a in zip(task, cost, combined)),
                            default=0.0)
    bearing = [index for index, sample in enumerate(samples) if sample["free"]]
    mean_abs_cost = k.mean([abs(value) for value in cost])
    return {
        "A_task": k.summary(task),
        "A_cost_positive_normalised": k.summary(cost),
        "lambda_times_A_task": k.summary(scaled),
        "A_combined": k.summary(combined),
        "combination_closed": bool(combination_error < 1e-5),
        "combination_max_abs_error": round(combination_error, 12),
        "A_task_actor_bearing_only": k.summary([task[i] for i in bearing]) if bearing else None,
        "A_cost_actor_bearing_only": k.summary([cost[i] for i in bearing]) if bearing else None,
        "A_combined_actor_bearing_only": (k.summary([combined[i] for i in bearing])
                                          if bearing else None),
        "samples_total": len(samples),
        "samples_with_actionable_candidates": len(bearing),
        "samples_with_zero_actor_gradient": len(samples) - len(bearing),
        "share_without_actor_gradient": round((len(samples) - len(bearing)) / len(samples), 9),
        "mean_abs_lambda_A_task_over_mean_abs_A_cost": (
            round(k.mean([abs(v) for v in scaled]) / mean_abs_cost, 9)
            if mean_abs_cost else None),
        "returns_task": k.summary([float(s["return_task"]) for s in samples]),
        "returns_cost": k.summary([float(s["return_cost"]) for s in samples]),
        "behaviour_values_task": k.summary([float(s["behaviour_value_task"]) for s in samples]),
        "behaviour_values_cost": k.summary([float(s["behaviour_value_cost"]) for s in samples]),
        "prefix_not_credited_to_any_decision": {
            "task_reward_samples": sum(1 for s in samples
                                       if s["unassigned_prefix_task_reward"] != 0.0),
            "cost_byte_samples": sum(1 for s in samples
                                     if s["unassigned_prefix_cost_bytes"] != 0.0),
        },
    }


def main() -> int:
    built = k.build_row_index()
    rows = {(record["key"]["arm"], int(record["key"]["seed"]), int(record["key"]["episode"])):
            record["row"] for record in built["index"].values()
            if record["key"]["kind"] == "train"}

    results: dict[str, Any] = {}
    for arm in k.ARMS:
        for seed in k.TRAIN_SEEDS:
            cell = f"{arm}_s{seed}"
            controller = None
            for batch_index in FIXED_BATCHES:
                record = k.read_json(k.batch_path(arm, seed, batch_index))
                episodes = build_episodes(arm, seed, record, rows)
                samples = trainer.batch_samples(episodes)
                task, cost = advantage_components(episodes)
                if len(task) != len(samples):
                    raise RuntimeError(f"{cell} batch {batch_index}: component recount "
                                       f"{len(task)} != trainer samples {len(samples)}")
                for index, sample in enumerate(samples):
                    sample["advantage_task"] = task[index]
                    sample["advantage_cost"] = cost[index]
                prefix_task, prefix_cost = trainer._unassigned_prefix_totals(episodes)
                lam_batch = float(record["lambda_at_rollout"])
                update = record.get("update") or {}
                entry: dict[str, Any] = {
                    "cell": cell, "arm": arm, "training_seed": seed, "batch": batch_index,
                    "tape_files": [str(Path(e["artefacts"]["tape"]).relative_to(k.ROOT))
                                   for e in episodes],
                    "availability": {
                        "batch_record": str(k.batch_path(arm, seed, batch_index).relative_to(
                            k.ROOT)),
                        "tapes_saved_for_this_batch": True,
                        "pre_update_weights_saved": batch_index == 0,
                        "pre_update_weights_source": (
                            str(k.milestone_path(arm, seed, 0).relative_to(k.ROOT))
                            if batch_index == 0 else None),
                        "reason_when_absent": (
                            None if batch_index == 0
                            else "r30_train writes the rolling checkpoint every batch (same path, "
                                 "overwritten) plus immutable ep0/64/128 milestones; this batch's "
                                 "pre-update weights are the previous batch's post-update weights "
                                 "and were never written, so they are reported missing -- not "
                                 "approximated by a later milestone, not recovered by replaying "
                                 "the optimizer"),
                    },
                    "reconstruction": {
                        "samples_recomputed": len(samples),
                        "samples_recorded_in_batch": int(update.get("samples", -1)),
                        "samples_match": len(samples) == int(update.get("samples", -1)),
                        "minibatches_recorded": update.get("minibatches"),
                        "unassigned_prefix_task_reward_recomputed": prefix_task,
                        "unassigned_prefix_task_reward_recorded": update.get(
                            "unassigned_prefix_task_reward"),
                        "unassigned_prefix_cost_bytes_recomputed": prefix_cost,
                        "unassigned_prefix_cost_bytes_recorded": update.get(
                            "unassigned_prefix_cost_bytes"),
                        "prefix_recomputation_matches": (
                            abs(prefix_task - float(update.get("unassigned_prefix_task_reward")
                                                    or 0.0)) < 1e-9
                            and abs(prefix_cost - float(update.get("unassigned_prefix_cost_bytes")
                                                        or 0.0)) < 1e-9),
                        "lambda_used_by_this_batch_actor": lam_batch,
                        "lambda_before_recorded": float(record["dual"]["lambda_before"]),
                        "lambda_after_recorded": float(record["dual"]["lambda_next"]),
                        "actor_used_pre_update_lambda": (
                            abs(lam_batch - float(record["dual"]["lambda_before"])) < 1e-12),
                        "episodes": [int(e["episode_index"]) for e in episodes],
                        "scenarios": [int(e["scenario_seed"]) for e in episodes],
                        "optimizer_steps_performed_by_this_round": 0,
                    },
                    "signals": advantage_block(samples, lam_batch),
                    "recorded_update_fields": {
                        "optimizer_steps": update.get("optimizer_steps"),
                        "epochs": update.get("epochs"),
                        "policy_surrogate": update.get("policy_surrogate"),
                        "value_loss_task": update.get("value_loss_task"),
                        "value_loss_cost": update.get("value_loss_cost"),
                        "entropy": update.get("entropy"),
                        "approx_kl": update.get("approx_kl"),
                        "clip_fraction": update.get("clip_fraction"),
                        "gradient_norm_mean": update.get("gradient_norm_mean"),
                        "module_gradient_l1_mean": update.get("module_gradient_l1_mean"),
                        "module_gradient_l1_min": update.get("module_gradient_l1_min"),
                        "module_gradient_scope": update.get("module_gradient_scope"),
                        "likelihood_max_abs_gap_recorded": (update.get("likelihood_alignment")
                                                            or {}).get("max_abs_gap"),
                        "averaging_scope": "each scalar above is the mean over this batch's "
                                           "minibatches as recorded by v2_trainer.ppo_update",
                    },
                }
                if batch_index == 0:
                    if controller is None:
                        import r30_check as check30
                        controller, _optimizer, _state = trainer.load_checkpoint(
                            k.milestone_path(arm, seed, 0), freeze=check30.freeze_record(arm))
                    identity = k.parameter_sha256_from_payload(
                        k.load_payload(k.milestone_path(arm, seed, 0)))
                    groups = chunk_groups(len(samples), batch_index)
                    entry["gradient_at_update_start"] = gradient_report(
                        controller, samples, groups[0], lam_batch)
                    recomputed_gap = likelihood_recompute(controller, samples)
                    entry["likelihood_recomputed_at_pre_update_weights"] = recomputed_gap
                    recorded_gap = entry["recorded_update_fields"][
                        "likelihood_max_abs_gap_recorded"]
                    entry["likelihood_recompute_agrees_with_recorded"] = {
                        "recorded_max_abs_gap": recorded_gap,
                        "recomputed_max_abs_gap": recomputed_gap["max_abs_gap"],
                        "same_sample_count": (recomputed_gap["samples"]
                                              == len(samples)),
                        "within_1e-9": (recorded_gap is not None and abs(
                            float(recorded_gap) - recomputed_gap["max_abs_gap"]) < 1e-9),
                    }
                    entry["pre_update_weights_identity"] = {
                        "ep0_recomputed_parameter_sha256": identity,
                        "batch0_recorded_parameters_before_sha256": record[
                            "controller_parameters_before"]["parameter_sha256"],
                        "match": identity == record["controller_parameters_before"]
                        ["parameter_sha256"],
                    }
                else:
                    entry["gradient_at_update_start"] = {
                        "status": "missing",
                        "reason": entry["availability"]["reason_when_absent"],
                        "what_is_still_computed": "the tape-side advantage decomposition, which "
                                                  "needs no weights because both critic values "
                                                  "were logged at rollout",
                        "optimizer_steps_performed_by_this_round": 0,
                    }
                results.setdefault(cell, {})[f"b{batch_index}"] = entry

    payload = {
        "round": "31",
        "section": "task/cost signal inside the real actor updates, fixed batches 0/15/31 "
                   "(0 launches, 0 optimizer.step; pure-computation forward and autograd.grad on "
                   "an in-memory copy of the saved pre-update weights)",
        "fixed_batch_indices": list(FIXED_BATCHES),
        "selection_rule": "fixed by the execution basis; not chosen from results",
        "path_examined": "tape -> v2_trainer.batch_samples (frozen GAE, reward timing, B0 "
                         "normalisation) -> lambda*A_task - A_cost -> PPO surrogate -> entropy / "
                         "two critic losses -> per-module gradients",
        "availability_matrix": {cell: {f"b{batch}": {
            "advantage_decomposition": True,
            "gradient_decomposition": results[cell][f"b{batch}"]["availability"][
                "pre_update_weights_saved"],
        } for batch in FIXED_BATCHES} for cell in sorted(results)},
        "runs": results,
        "reading_notes": [
            "one sample is one decision-step tape, not one candidate message; a tape whose "
            "candidates are all masked still enters both critic losses and the surrogate mean "
            "denominator while contributing exactly zero actor gradient",
            "A_cost is in normalised units (bytes/B0) and A_task in on-time tasks, so the two are "
            "only commensurable after the multiplication by lambda that the trainer performs",
            "clip_fraction, approx_kl and the module gradient readings are the trainer's own "
            "per-batch means; nothing here substitutes a re-derived number for a missing field",
            "gradient magnitudes describe direction at the update start; Adam is nonlinear in the "
            "gradient, so a component split of the gradient is not a component split of the "
            "parameter step",
        ],
    }
    out = k.write_json("UPDATE_SIGNAL_DIAGNOSIS.json", payload)
    print(json.dumps({
        cell: {f"batch{int(batch[1:])}": {
            "samples": entry["reconstruction"]["samples_recomputed"],
            "samples_match": entry["reconstruction"]["samples_match"],
            "prefix_match": entry["reconstruction"]["prefix_recomputation_matches"],
            "combination_closed": entry["signals"]["combination_closed"],
            "zero_actor_gradient_share": entry["signals"]["share_without_actor_gradient"],
            "A_combined_mean": entry["signals"]["A_combined"]["mean"],
            "A_cost_mean": entry["signals"]["A_cost_positive_normalised"]["mean"],
            "lambda_A_task_mean": entry["signals"]["lambda_times_A_task"]["mean"],
            "ratio_task_over_cost": entry["signals"][
                "mean_abs_lambda_A_task_over_mean_abs_A_cost"],
            "gradient": ("ok" if entry["availability"]["pre_update_weights_saved"]
                         else "missing"),
            "identity_match": (entry.get("pre_update_weights_identity") or {}).get("match"),
            "likelihood_within_1e-9": (entry.get("likelihood_recompute_agrees_with_recorded")
                                       or {}).get("within_1e-9"),
            "clip_fraction": entry["recorded_update_fields"]["clip_fraction"],
            "approx_kl": entry["recorded_update_fields"]["approx_kl"],
        } for batch, entry in sorted(items.items(), key=lambda item: int(item[0][1:]))}
        for cell, items in sorted(results.items())}, ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
