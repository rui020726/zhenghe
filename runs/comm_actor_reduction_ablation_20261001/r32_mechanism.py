"""ROUND32 任务 G 之二：更新机制诊断（0 环境启动、0 新增 optimizer 更新）。

本轮**新具备**的能力：批 0/15/31 的更新前完整 checkpoint 已在启动前接入并落盘，所以六条运行
都能在真正的更新起点上做梯度诊断——不再只有批 0（ROUND31 的限制正是"只保存了更新后权重"）。

两类比较严格分开：

1. **同状态、同 tape 的算术比较**：取某臂某批的更新前权重与该批已落盘 tape，在同一张计算图上
   分别用两种归约求 actor 项梯度。这隔离归约定义的直接影响，比值应与 `N/M` 一致。
2. **训练后各自产生的状态比较**：两臂各自的权重、各自的样本与各自的裁剪情况。这里包含策略与
   数据分布的变化，不能把差异全部当成归约的即时作用。

诊断只在内存副本上用 `torch.autograd.grad`，绝不 `optimizer.step()`，不写回任何 .pt。
分量分解只在 minibatch 0（`old_log_prob == new_log_prob`、`ratio` 恰为 1、`torch.min` 两支
Jacobian 相同、`.mean()` 无样本权重、无任何 advantage 标准化）成立；后续 minibatch 需要重放
optimizer 才能重构，本轮禁止，故明确标为不可得。
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

import r32_common as common  # noqa: F401  (installs the shared sys.path before any archived-round import)
import cost_math as cm
import r32_trainer as trainer
import v2_trainer as frozen
from onpolicy import action_entropy, joint_log_prob
from r32_train import batch_record_path, pre_update_path

FIXED_BATCHES = common.PRE_UPDATE_BATCHES
POLICY_TERMS = ("policy_task_lambda_A_task", "policy_cost_minus_A_cost", "entropy_bonus",
                "critic_task", "critic_cost")


class _CollectorShim:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records


def mean(values):
    """Local descriptive mean (post-run analysis only; no training constant lives here)."""
    return sum(float(v) for v in values) / len(values)


def summary(values):
    """Local descriptive summary; the quantile rule is linear interpolation on sorted values."""
    floats = [float(v) for v in values]
    ordered = sorted(floats)
    n = len(ordered)
    average = mean(floats)

    def quantile(fraction):
        if n == 1:
            return ordered[0]
        position = (n - 1) * fraction
        low = int(position // 1)
        high = min(low + 1, n - 1)
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {"n": n, "mean": round(average, 9), "min": ordered[0], "max": ordered[-1],
            "q25": round(quantile(0.25), 9), "median": round(quantile(0.5), 9),
            "q75": round(quantile(0.75), 9),
            "std_pop": round((sum((v - average) ** 2 for v in floats) / n) ** 0.5, 9),
            "abs_mean": round(mean([abs(v) for v in floats]), 9)}


def load_episode(reduction: str, seed: int, episode: int) -> dict[str, Any]:
    """One archived training row of this round, shaped the way the frozen trainer wants it."""
    matches = sorted(common.ROWS.glob(f"r32_train_{reduction}_s{seed}_e{episode}_s*_a{episode}.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one training row for {reduction} s{seed} "
                           f"e{episode}, found {[path.name for path in matches]}")
    row = common.read_json(matches[0])
    return {
        "collector": _CollectorShim(list(common.iter_jsonl(Path(row["artefacts"]["tape"])))),
        "cost_stream": row["cost_stream"], "task_stream": row["task_stream"],
        "steps": int(row["steps"]), "lambda": float(row["lambda"]),
        "episode_index": int(row["episode_index"]), "scenario_seed": int(row["scenario_seed"]),
        "on_time": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]),
        "row_path": str(matches[0].relative_to(common.ROOT)),
    }


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
        bundled_cost, _ = frozen._bundle(cost_stream, steps, horizon)
        bundled_task, _ = frozen._bundle(task_stream, steps, horizon)
        rewards_cost = torch.tensor([bundled_cost[step] / common.B0_BYTES for step in steps],
                                    dtype=torch.float64)
        rewards_task = torch.tensor([bundled_task[step] for step in steps], dtype=torch.float64)
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
        adv_task, _ = cm.gae(rewards_task, values_task, gamma=float(common.v2c.GAMMA),
                             gae_lambda=float(common.v2c.GAE_LAMBDA), terminal=terminal,
                             bootstrap=bootstrap_task)
        adv_cost, _ = cm.gae(rewards_cost, values_cost, gamma=float(common.v2c.GAMMA),
                             gae_lambda=float(common.v2c.GAE_LAMBDA), terminal=terminal,
                             bootstrap=bootstrap_cost)
        task.extend(float(value) for value in adv_task)
        cost.extend(float(value) for value in adv_cost)
    return task, cost


def chunk_groups(n_samples: int, update_index: int) -> list[list[int]]:
    """Reproduce the frozen trainer's per-batch shuffle and its minibatch index groups."""
    shuffle = random.Random(int(hashlib.sha256(
        f"{int(common.DATA_ORDER_SEED)}:{int(update_index)}".encode("utf-8")).hexdigest()[:16],
        16))
    order = list(range(n_samples))
    shuffle.shuffle(order)
    size = int(common.v2c.MINIBATCH_STEPS)
    return [order[start:start + size] for start in range(0, len(order), size)]


def build_minibatch(controller: Any, samples: list[dict[str, Any]], chunk: list[int],
                    lam: float) -> dict[str, Any]:
    """Rebuild the trainer's first-minibatch terms exactly as `ppo_update` computes them."""
    chunk_samples = [samples[index] for index in chunk]
    stacked = torch.as_tensor(np.concatenate([item["features"] for item in chunk_samples],
                                             axis=0))
    hidden = controller.inner.trunk(stacked)
    logits = controller.inner.action_head(hidden)
    values_task = controller.inner.value_head(hidden).squeeze(-1)
    values_cost = controller.cost_head(hidden).squeeze(-1)
    new_log_prob, entropy, pooled_task, pooled_cost = [], [], [], []
    row = 0
    for item in chunk_samples:
        rows = torch.arange(row, row + len(item["targets"]))
        row += len(item["targets"])
        picked = torch.tensor([item["actions"][t] for t in item["targets"]], dtype=torch.long)
        usable = torch.tensor([1.0 if t in item["free"] else 0.0 for t in item["targets"]],
                              dtype=torch.float32)
        new_log_prob.append(joint_log_prob(logits[rows], picked, usable))
        entropy.append(action_entropy(logits[rows], usable))
        pooled_task.append(values_task[rows].mean())
        pooled_cost.append(values_cost[rows].mean())
    new_log_prob_t = torch.stack(new_log_prob)
    entropy_t = torch.stack(entropy)
    old_log_prob = torch.tensor([item["log_prob"] for item in chunk_samples], dtype=torch.float32)
    advantage = torch.tensor([item["advantage"] for item in chunk_samples], dtype=torch.float32)
    advantage_task = torch.tensor([item["advantage_task"] for item in chunk_samples],
                                  dtype=torch.float32)
    advantage_cost = torch.tensor([item["advantage_cost"] for item in chunk_samples],
                                  dtype=torch.float32)
    target_task = torch.tensor([item["return_task"] for item in chunk_samples],
                                dtype=torch.float32)
    target_cost = torch.tensor([item["return_cost"] for item in chunk_samples],
                               dtype=torch.float32)
    ratio = torch.exp(new_log_prob_t - old_log_prob)
    clip = float(common.v2c.CLIP)
    surrogate = torch.min(ratio * advantage,
                          torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage)
    row_active = trainer.row_active_mask(chunk_samples)
    value_task_loss = ((torch.stack(pooled_task) - target_task) ** 2).mean()
    value_cost_loss = ((torch.stack(pooled_cost) - target_cost) ** 2).mean()
    combination_error = float(torch.max((lam * advantage_task - advantage_cost
                                         - advantage).abs()).item())
    return {
        "chunk_samples": chunk_samples,
        "surrogate": surrogate, "row_active": row_active, "ratio": ratio,
        "advantage": advantage, "advantage_task": advantage_task,
        "advantage_cost": advantage_cost,
        "entropy_mean": entropy_t.mean(), "value_task_loss": value_task_loss,
        "value_cost_loss": value_cost_loss,
        "n_rows": int(surrogate.shape[0]),
        "m_active_rows": int(row_active.sum().item()),
        "active_in_chunk": sum(1 for item in chunk_samples if item["free"]),
        "ratio_max_abs_deviation_from_one": float((ratio.detach() - 1.0).abs().max().item()),
        "combination_max_abs_error": combination_error,
        "clip_fraction": float(((ratio - 1.0).abs() > clip).float().mean().item()),
        "approx_kl": float((old_log_prob - new_log_prob_t).mean().item()),
        "entropy_term": float(entropy_t.mean().detach().item()),
        "value_loss_task": float(value_task_loss.detach().item()),
        "value_loss_cost": float(value_cost_loss.detach().item()),
    }


def term_gradients(controller: Any, built: dict[str, Any], lam: float,
                   reduction: str) -> dict[str, torch.Tensor]:
    """Per-term gradients at the update start, all with the same reduction and scaling."""
    parameters = list(controller.parameters)
    surrogate, row_active, ratio = built["surrogate"], built["row_active"], built["ratio"]
    actor_mean, _, _ = trainer.actor_surrogate_mean(surrogate, row_active, reduction)
    surrogate_task = ratio * (lam * built["advantage_task"])
    surrogate_cost = ratio * (-built["advantage_cost"])
    terms = {
        "policy_task_lambda_A_task": -(surrogate_task.mean()),
        "policy_cost_minus_A_cost": -(surrogate_cost.mean()),
        "policy_total": -(actor_mean),
        "entropy_bonus": -float(common.v2c.ENTROPY_BONUS) * built["entropy_mean"],
        "critic_task": float(common.v2c.VALUE_COEFFICIENT) * built["value_task_loss"],
        "critic_cost": float(common.v2c.VALUE_COEFFICIENT) * built["value_cost_loss"],
    }
    out: dict[str, torch.Tensor] = {}
    for name, term in terms.items():
        grads = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
        out[name] = torch.cat([(g if g is not None else torch.zeros_like(p)).detach()
                               .reshape(-1) for g, p in zip(grads, parameters)])
    out["total_registered_loss"] = (out["policy_total"] + out["entropy_bonus"]
                                   + out["critic_task"] + out["critic_cost"])
    out["sum_of_five_parts"] = sum(out[name] for name in POLICY_TERMS)
    return out


def module_slice(controller: Any, vector: torch.Tensor) -> dict[str, dict[str, float]]:
    modules = (("trunk", controller.inner.trunk), ("action_head", controller.inner.action_head),
               ("task_value_head", controller.inner.value_head), ("cost_head", controller.cost_head))
    out: dict[str, dict[str, float]] = {}
    offset = 0
    for name, module in modules:
        end = offset + sum(int(parameter.numel()) for parameter in module.parameters())
        piece = vector[offset:end]
        offset = end
        out[name] = {"l1": round(float(piece.abs().sum().item()), 12),
                     "l2": round(float(piece.norm().item()), 12)}
    return out


def arm_view(controller: Any, grads: dict[str, torch.Tensor]) -> dict[str, Any]:
    total = float(grads["total_registered_loss"].norm().item())
    clip = float(common.v2c.MAX_GRAD_NORM)
    post = min(total, clip)
    return {
        "term_l2": {name: round(float(vector.norm().item()), 12)
                    for name, vector in grads.items()},
        "term_l1": {name: round(float(vector.abs().sum().item()), 12)
                    for name, vector in grads.items()},
        "term_by_module": {name: module_slice(controller, vector)
                           for name, vector in grads.items()},
        "pre_clip_total_norm": round(total, 9),
        "post_clip_total_norm": round(post, 9),
        "clipping_triggered": bool(total > clip),
        "clip_scale": round(post / total, 9) if total > 0 else 1.0,
        "policy_share_of_pre_clip_l2": (round(float(grads["policy_total"].norm().item()) / total,
                                              9) if total > 0 else None),
        "critic_share_of_pre_clip_l2": (round(
            float((grads["critic_task"] + grads["critic_cost"]).norm().item()) / total, 9)
            if total > 0 else None),
        "closure_error_max_abs": float((grads["sum_of_five_parts"]
                                        - grads["total_registered_loss"]).abs().max().item()),
    }


def freeze_for(reduction: str) -> dict[str, Any]:
    from r32_freeze import freeze_record
    return freeze_record(reduction)


def analyse(reduction: str, seed: int, batch: int) -> dict[str, Any]:
    """Analyse one (run, fixed batch): own arm plus the other reduction on the same state."""
    record = common.read_json(batch_record_path(reduction, seed, batch))
    path = pre_update_path(reduction, seed, batch)
    if not path.exists():
        return {"status": "missing", "batch": batch, "reduction": reduction,
                "training_seed": seed,
                "reason": (f"{path.name} was not written; this round saves the pre-update "
                           "checkpoint before ppo_update for batches "
                           f"{list(FIXED_BATCHES)}, so a missing file means the batch never ran"),
                "optimizer_steps_performed_by_this_round": 0}
    controller, _optimizer, state = trainer.load_checkpoint(path, freeze=freeze_for(reduction),
                                                            reduction=reduction)
    episodes = [load_episode(reduction, seed, int(episode))
                for episode in record["episodes"]]
    samples = frozen.batch_samples(episodes)
    task, cost = advantage_components(episodes)
    if len(task) != len(samples):
        raise RuntimeError(f"{reduction} s{seed} b{batch}: component recount differs")
    lam = float(record["lambda_at_rollout"])
    for index, sample in enumerate(samples):
        sample["advantage_task"] = task[index]
        sample["advantage_cost"] = cost[index]
    groups = chunk_groups(len(samples), batch)
    built = build_minibatch(controller, samples, groups[0], lam)
    own = term_gradients(controller, built, lam, reduction)
    other = next(name for name in common.ARM_ORDER if name != reduction)
    other_view = term_gradients(controller, built, lam, other)
    own_view = arm_view(controller, own)
    other_arm_view = arm_view(controller, other_view)
    views = {reduction: own_view, other: other_arm_view}
    base_view = views[common.ARM_ORDER[0]]
    test_view = views[common.ARM_ORDER[1]]
    actor_base = base_view["term_l2"]["policy_total"]
    actor_test = test_view["term_l2"]["policy_total"]
    expected = (built["n_rows"] / built["m_active_rows"]) if built["m_active_rows"] else None
    observed = (actor_test / actor_base) if actor_base > 0 else None
    return {
        "status": "analysed", "reduction": reduction, "training_seed": seed, "batch": batch,
        "pre_update_checkpoint": str(path.relative_to(common.ROOT)),
        "pre_update_parameter_sha256": record["controller_parameters_before"]
        ["parameter_sha256"],
        "state_at_start": {"episodes_done": int(state["episodes_done"]),
                           "updates": int(state["updates"]),
                           "optimizer_steps": int(state["optimizer_steps"]),
                           "lambda": float(state["lambda"]),
                           "reduction_mode": state["reduction_mode"]},
        "scope": "epoch 0 / minibatch 0 of this batch only (old == new, ratio exactly 1). Later "
                 "minibatches would require replaying optimizer.step, which this round forbids",
        "batch_shape": {"samples_in_batch": len(samples), "minibatches_in_batch": len(groups),
                        "chunk_rows_N": built["n_rows"], "chunk_active_rows_M":
                        built["m_active_rows"],
                        "chunk_masked_rows": built["n_rows"] - built["m_active_rows"],
                        "ratio_max_abs_deviation_from_one": built[
                            "ratio_max_abs_deviation_from_one"]},
        "advantage_combination_closed": bool(built["combination_max_abs_error"] < 1e-5),
        "advantage_combination_max_abs_error": round(built["combination_max_abs_error"], 15),
        "advantages": {
            "A_task": summary(task), "A_cost": summary(cost),
            "lambda_times_A_task": summary([lam * value for value in task]),
            "A_combined": summary([float(sample["advantage"]) for sample in samples]),
            "A_combined_active_rows_only": summary(
                [float(sample["advantage"]) for sample in samples if sample["free"]]),
            "lambda_of_this_batch": lam},
        "untouched_terms_at_this_point": {
            "entropy_term": built["entropy_term"], "value_loss_task": built["value_loss_task"],
            "value_loss_cost": built["value_loss_cost"], "clip_fraction": built["clip_fraction"],
            "approx_kl": built["approx_kl"],
            "note": "these do not depend on the reduction, so both arms see the same numbers at "
                    "this point by construction"},
        "arithmetical_comparison_same_state_same_tape": {
            "base_arm": common.ARM_ORDER[0], "test_arm": common.ARM_ORDER[1],
            "actor_term_l2": {common.ARM_ORDER[0]: round(actor_base, 12),
                              common.ARM_ORDER[1]: round(actor_test, 12)},
            "observed_amplification": round(observed, 9) if observed is not None else None,
            "expected_N_over_M": round(expected, 9) if expected is not None else None,
            "absolute_gap": (round(abs(observed - expected), 12)
                             if observed is not None and expected is not None else None),
            "relative_gap": (round(abs(observed - expected) / expected, 12)
                             if observed is not None and expected else None),
            "tolerance_rule": "relative 1e-5 on the ratio. The ratio itself is O(N/M) and reaches "
                              "51 at these points, so a fixed absolute band would really only be "
                              "testing float32 digit count. Both residuals are recorded.",
            "amplification_matches_within_float32": bool(
                observed is not None and expected
                and abs(observed - expected) / expected < 1e-5),
            "total_pre_clip_l2": {arm: views[arm]["pre_clip_total_norm"]
                                  for arm in common.ARM_ORDER},
            "total_post_clip_l2": {arm: views[arm]["post_clip_total_norm"]
                                   for arm in common.ARM_ORDER},
            "clipping_triggered": {arm: views[arm]["clipping_triggered"]
                                   for arm in common.ARM_ORDER},
            "policy_share_of_pre_clip": {arm: views[arm]["policy_share_of_pre_clip_l2"]
                                         for arm in common.ARM_ORDER},
            "critic_share_of_pre_clip": {arm: views[arm]["critic_share_of_pre_clip_l2"]
                                         for arm in common.ARM_ORDER},
            "note": "the actor term's own L2 scales by N/M; the TOTAL pre-clip norm moves much "
                    "less because the two critics dominate, and when global clipping fires both "
                    "arms are rescaled to the same post-clip magnitude, so what the reduction "
                    "changes is the DIRECTION (the actor's share of the clipped vector), not its "
                    "size. Adam is nonlinear in the gradient, so this is not a step-size claim."},
        "per_arm": views,
        "recorded_during_training": {
            "minibatches": int((record.get("update") or {}).get("minibatches", 0)),
            "optimizer_steps": int((record.get("update") or {}).get("optimizer_steps", 0)),
            "clip_fraction_mean": (record.get("update") or {}).get("clip_fraction"),
            "approx_kl_mean": (record.get("update") or {}).get("approx_kl"),
            "gradient_norm_mean_pre_clip": (record.get("update") or {}).get("gradient_norm_mean"),
            "first_minibatch_read": ((record.get("update") or {}).get("minibatch_reads")
                                     or [{}])[0],
            "samples_recorded_vs_recomputed": {
                "recorded": int((record.get("update") or {}).get("samples", -1)),
                "recomputed": len(samples),
                "match": int((record.get("update") or {}).get("samples", -1)) == len(samples)},
            "lambda_recorded_at_rollout": float(record["lambda_at_rollout"]),
            "lambda_state_at_start": float(state["lambda"])},
        "optimizer_steps_performed_by_this_round": 0,
    }


def main() -> int:
    payload: dict[str, Any] = {
        "round": "32",
        "section": "update mechanism diagnosis on the saved pre-update checkpoints "
                   "(0 launches, 0 optimizer.step, pure-computation forward and autograd.grad on "
                   "an in-memory copy of the saved weights)",
        "fixed_batch_indices": list(FIXED_BATCHES),
        "availability_note": "this round writes the pre-update checkpoint of batches 0, 15 and 31 "
                             "before ppo_update runs, so all six runs are analysable at all three "
                             "indices; ROUND31 could only reach batch 0 because only the initial "
                             "milestone survived",
        "two_kinds_of_comparison": {
            "arithmetical": "same weights, same tape, both reductions in one graph: isolates the "
                            "definition's direct effect",
            "realised": "each arm's own trained weights and own samples: includes policy and data "
                        "distribution changes, so not every difference is the reduction's doing"},
        "runs": {}, "realised_state_comparison": {}}
    for reduction in common.ARM_ORDER:
        for seed in common.TRAIN_SEEDS:
            payload["runs"][f"{reduction}_s{seed}"] = {
                f"b{batch}": analyse(reduction, int(seed), int(batch)) for batch in FIXED_BATCHES}

    for seed in common.TRAIN_SEEDS:
        rows: dict[str, Any] = {}
        for reduction in common.ARM_ORDER:
            blocks = payload["runs"][f"{reduction}_s{seed}"]
            def per_batch(selector: Any) -> dict[str, Any]:
                return {key: (selector(value) if value.get("status") == "analysed" else None)
                        for key, value in sorted(blocks.items())}
            rows[reduction] = {
                "availability": {key: value.get("status") for key, value in
                                 sorted(blocks.items())},
                "samples_in_batch": per_batch(lambda v: v["batch_shape"]["samples_in_batch"]),
                "chunk_active_share": per_batch(
                    lambda v: round(v["batch_shape"]["chunk_active_rows_M"]
                                    / v["batch_shape"]["chunk_rows_N"], 6)),
                "lambda_at_batch_start": per_batch(lambda v: v["state_at_start"]["lambda"]),
                "own_actor_term_l2": per_batch(
                    lambda v: v["per_arm"][reduction]["term_l2"]["policy_total"]),
                "own_total_pre_clip_l2": per_batch(
                    lambda v: v["per_arm"][reduction]["pre_clip_total_norm"]),
                "own_policy_share_of_total": per_batch(
                    lambda v: v["per_arm"][reduction]["policy_share_of_pre_clip_l2"]),
                "own_critic_share_of_total": per_batch(
                    lambda v: v["per_arm"][reduction]["critic_share_of_pre_clip_l2"]),
                "own_clip_scale": per_batch(lambda v: v["per_arm"][reduction]["clip_scale"]),
                "action_head_task_l1": per_batch(
                    lambda v: v["per_arm"][reduction]["term_by_module"]
                    ["policy_task_lambda_A_task"]["action_head"]["l1"]),
                "action_head_cost_l1": per_batch(
                    lambda v: v["per_arm"][reduction]["term_by_module"]
                    ["policy_cost_minus_A_cost"]["action_head"]["l1"]),
                "action_head_entropy_l1": per_batch(
                    lambda v: v["per_arm"][reduction]["term_by_module"]
                    ["entropy_bonus"]["action_head"]["l1"]),
                "action_head_critic_l1_is_exactly_zero": per_batch(
                    lambda v: v["per_arm"][reduction]["term_by_module"]["critic_task"]
                    ["action_head"]["l1"] == 0.0
                    and v["per_arm"][reduction]["term_by_module"]["critic_cost"]
                    ["action_head"]["l1"] == 0.0),
                "trunk_critic_task_l1": per_batch(
                    lambda v: v["per_arm"][reduction]["term_by_module"]["critic_task"]["trunk"]
                    ["l1"]),
                "closure_error_max_abs": per_batch(
                    lambda v: v["per_arm"][reduction]["closure_error_max_abs"]),
            }
        payload["realised_state_comparison"][f"s{seed}"] = rows

    analysed_points = [f"{key}/{batch}" for key, blocks in payload["runs"].items()
                       for batch, value in blocks.items() if value.get("status") == "analysed"]
    missing_points = [f"{key}/{batch}" for key, blocks in payload["runs"].items()
                      for batch, value in blocks.items() if value.get("status") != "analysed"]
    mismatching = [f"{key}/{batch}" for key, blocks in payload["runs"].items()
                   for batch, value in blocks.items()
                   if value.get("status") == "analysed" and not
                   value["arithmetical_comparison_same_state_same_tape"]
                   ["amplification_matches_within_float32"]]
    relative_gaps = [
        value["arithmetical_comparison_same_state_same_tape"]["relative_gap"]
        for blocks in payload["runs"].values() for value in blocks.values()
        if value.get("status") == "analysed"
        and value["arithmetical_comparison_same_state_same_tape"]["relative_gap"] is not None]
    absolute_gaps = [
        value["arithmetical_comparison_same_state_same_tape"]["absolute_gap"]
        for blocks in payload["runs"].values() for value in blocks.values()
        if value.get("status") == "analysed"
        and value["arithmetical_comparison_same_state_same_tape"]["absolute_gap"] is not None]
    not_closed = [f"{key}/{batch}" for key, blocks in payload["runs"].items()
                  for batch, value in blocks.items()
                  if value.get("status") == "analysed" and not value["advantage_combination_closed"]]
    samples_mismatch = [f"{key}/{batch}" for key, blocks in payload["runs"].items()
                        for batch, value in blocks.items()
                        if value.get("status") == "analysed"
                        and not value["recorded_during_training"]["samples_recorded_vs_recomputed"]
                        ["match"]]
    payload["answers"] = {
        "analysed_points": len(analysed_points),
        "expected_points": (len(common.ARM_ORDER) * len(common.TRAIN_SEEDS)
                            * len(FIXED_BATCHES)),
        "missing_points": missing_points,
        "amplification_holds_at_every_analysed_point": bool(analysed_points and not mismatching),
        "amplification_mismatching_points": mismatching,
        "amplification_points_with_a_ratio": len(relative_gaps),
        "amplification_max_relative_gap": max(relative_gaps) if relative_gaps else None,
        "amplification_max_absolute_gap": max(absolute_gaps) if absolute_gaps else None,
        "advantage_combination_closed_everywhere": not not_closed,
        "samples_record_equal_recomputed_everywhere": not samples_mismatch,
        "question_answers": {
            "did_the_expected_actor_amplification_happen":
                "answerable from arithmetical_comparison_same_state_same_tape at all six runs and "
                "three batches; the observed ratio is the actor term's L2 under ACTIVE_ROWS over "
                "ALL_ROWS on identical weights and identical tape",
            "does_it_survive_into_the_total_and_post_clip_gradient":
                "answerable from total_pre_clip_l2, policy_share_of_pre_clip, clipping_triggered "
                "and total_post_clip_l2; a post-clip equality with clipping on both sides means "
                "only the direction changed, not the magnitude",
            "did_constraint_violation_and_policy_behaviour_change":
                "TRAINING_CURVES.json (per-batch g_hat, lambda, cost, active share) plus "
                "PAIRED_COMPARISON.json (behaviour), never inferred from gradient norms alone",
            "if_there_is_no_improvement_what_is_excluded":
                "excludable: that the reduction never reached the actor, or that the actor term "
                "was not amplified. Not excludable by this evidence: whether a different lambda, "
                "more batches, or a different observation would help -- none of those was "
                "intervened on"},
    }
    out = common.OUTPUT_DIR / "UPDATE_MECHANISM.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({
        "analysed": payload["answers"]["analysed_points"],
        "expected": payload["answers"]["expected_points"],
        "missing": missing_points, "mismatching": mismatching,
        "combination_closed": payload["answers"]["advantage_combination_closed_everywhere"],
        "samples_equal": payload["answers"]["samples_record_equal_recomputed_everywhere"],
        "example_ALL_ROWS_s11_b0": payload["runs"]["ALL_ROWS_s11"]["b0"].get(
            "arithmetical_comparison_same_state_same_tape", {}),
    }, ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
