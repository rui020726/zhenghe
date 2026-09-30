"""Stage C correctness-v2 trainer: dual-critic PPO + dual lambda update (ROUND23).

与旧 trainer 的登记差异（spec §2 要求显式记录）：
- **不做任何 advantage 标准化**：旧 `train_onpolicy.flatten`/`train_pilot.update` 逐 episode
  除 `std(advantages)`（数据依赖），本版禁止；task/cost 分量也不分别标准化；
- 正归一化成本优势，组合 `lambda*A_task - A_cost`（cost_math.actor_advantage）；
- **两个 critic**：task 价值头（pilot 既有）+ cost 价值头（Stage B 新增），各自 MSE，
  权重 VALUE_COEFFICIENT=0.5 每头（登记选择）；
- gamma=1、GAE lambda=0.95；每个非终止决策行取下一决策的 task/cost value，真实终止行取0；
- tick 奖励归给该 tick 发生后最近的一次决策，首决策前的前缀收益单独记账；
- 对偶 lambda 每 4 集 PPO 更新后更新一次：`lambda_next = max(0, lambda + eta*g_hat)`，
  同一批 PPO 期间 lambda 固定；g_hat 用配对 reference 缓存标量（无梯度）；
- 熵是额外正则项（ENTROPY_BONUS=0.01，与 pilot 同值），其对纯成本目标的影响在此登记、不隐去；
- minibatch shuffle 用每批独立的 `random.Random(shuffle_seed)`，不碰全局 RNG。

兼容性：trainer/loss/state schema 均为 v2；v1 checkpoint 不允许恢复进修复版。
复用：joint_log_prob/action_entropy/rng_state/restore_rng 取自 train_onpolicy，PPO 超参见
v2_common（全部引用 pilot 轮 train_pilot 的既有值）。
"""

from __future__ import annotations

import hashlib
import json
import random
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import torch

import v2_common as common
import cost_math as cm
from onpolicy import action_entropy, joint_log_prob, restore_rng, rng_state

TRAINER_TAG = "constrained-cost-v2"
LOSS_VERSION = "constrained-lagrangian/2"
STATE_SCHEMA = "constrained-cost-state/2"
GAE_SCHEMA = "per-decision-next-value/1"
OPTIMIZER_SCHEMA = "stage-c-four-module-parameter-set/1"
REWARD_TIMING_SCHEMA = "post-action-most-recent-decision/1"


# ------------------------------------------------------------------ 样本 -------------
def batch_samples(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per decision step, with the combined advantage and both critic targets.

    Each post-action reward is credited to the latest action already sampled at that point.
    Rewards before the first action are unassigned and reported separately; episode totals
    are still reconciled against the executor's ledger.
    """
    samples: list[dict[str, Any]] = []
    for episode in episodes:
        tapes = sorted(episode["collector"].records, key=lambda tape: int(tape["step"]))
        if not tapes:
            continue
        decision_steps = sorted(int(tape["step"]) for tape in tapes)
        horizon = int(episode["steps"])
        cost_stream = {int(step): float(row["airtime_bytes"])
                       for step, row in episode["cost_stream"].items()}
        task_stream = {int(step): float(value) for step, value in episode["task_stream"].items()}
        bundled_cost, prefix_cost = _bundle(cost_stream, decision_steps, horizon)
        bundled_task, prefix_task = _bundle(task_stream, decision_steps, horizon)
        # Keep the cost critic's target in the positive measured-cost convention used by
        # ``lambda * A_task - A_cost``. Negating here and subtracting A_cost again makes
        # higher-cost actions look better to the actor.
        rewards_cost = torch.tensor([bundled_cost[s] / common.B0_BYTES for s in decision_steps],
                                    dtype=torch.float64)
        rewards_task = torch.tensor([bundled_task[s] for s in decision_steps],
                                    dtype=torch.float64)
        values_task = torch.tensor([float(tape["value"]) for tape in tapes], dtype=torch.float64)
        values_cost = torch.tensor([float(tape["cost_value"]) for tape in tapes],
                                   dtype=torch.float64)
        terminal = torch.zeros(len(decision_steps), dtype=torch.bool)
        terminal[-1] = True
        # For a nonterminal decision, delta_t needs V(s_{t+1}); the value captured on the
        # next tape row is that state estimate. The last decision is a true episode terminal
        # in this runner, so its bootstrap remains zero.
        bootstrap_task = torch.zeros_like(values_task)
        bootstrap_cost = torch.zeros_like(values_cost)
        if len(decision_steps) > 1:
            bootstrap_task[:-1] = values_task[1:]
            bootstrap_cost[:-1] = values_cost[1:]
        advantages_task, returns_task = cm.gae(rewards_task, values_task, gamma=common.GAMMA,
                                               gae_lambda=common.GAE_LAMBDA, terminal=terminal,
                                               bootstrap=bootstrap_task)
        advantages_cost, returns_cost = cm.gae(rewards_cost, values_cost, gamma=common.GAMMA,
                                               gae_lambda=common.GAE_LAMBDA, terminal=terminal,
                                               bootstrap=bootstrap_cost)
        combined = cm.actor_advantage(advantages_task, advantages_cost, episode["lambda"])
        for index, tape in enumerate(tapes):
            samples.append({
                "features": np.asarray(tape["features"], dtype=np.float32),
                "targets": list(tape["targets"]),
                "actions": dict(tape["actions"]),
                "free": set(tape["actionable"]),
                "log_prob": float(tape["log_prob"]),
                "advantage": float(combined[index]),
                "return_task": float(returns_task[index]),
                "return_cost": float(returns_cost[index]),
                "behaviour_value_task": float(values_task[index]),
                "behaviour_value_cost": float(values_cost[index]),
                "step": int(tape["step"]),
                "episode_index": int(episode["episode_index"]),
                "unassigned_prefix_task_reward": float(prefix_task if index == 0 else 0.0),
                "unassigned_prefix_cost_bytes": float(prefix_cost if index == 0 else 0.0),
            })
    return samples


def _bundle(stream: dict[int, float], decision_steps: list[int],
            horizon: int) -> tuple[dict[int, float], float]:
    """Assign each tick's post-action reward to the latest decision at or before that tick.

    The relay samples a policy action before ``env.step`` advances the world on a tick. An
    event at that tick therefore follows that action. A prefix before the first decision has
    no causal action to credit and is returned separately.
    """
    bundled = {step: 0.0 for step in decision_steps}
    prefix = 0.0
    # Tick zero is reserved for an event before the first policy action. It remains an
    # unassigned prefix; ordinary relay cost/reward ticks still begin at one.
    for step in range(0, horizon + 1):
        value = float(stream.get(step, 0.0))
        previous = bisect_right(decision_steps, step) - 1
        if previous < 0:
            prefix += value
        else:
            bundled[decision_steps[previous]] += value
    return bundled, prefix


def _unassigned_prefix_totals(episodes: list[dict[str, Any]]) -> tuple[float, float]:
    """Return pre-first-decision task reward and raw cost bytes across a batch."""
    task_total = 0.0
    cost_total = 0.0
    for episode in episodes:
        tapes = episode["collector"].records
        decision_steps = sorted(int(tape["step"]) for tape in tapes)
        horizon = int(episode["steps"])
        task_stream = {int(step): float(value)
                       for step, value in episode["task_stream"].items()}
        cost_stream = {int(step): float(row["airtime_bytes"])
                       for step, row in episode["cost_stream"].items()}
        _task, task_prefix = _bundle(task_stream, decision_steps, horizon)
        _cost, cost_prefix = _bundle(cost_stream, decision_steps, horizon)
        task_total += task_prefix
        cost_total += cost_prefix
    return task_total, cost_total


# ------------------------------------------------------------------ PPO 更新 -------------
def likelihood_alignment(controller: Any, samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute each sample's joint log-probability from features + parameters.

    Run before any weight moves, so a tape that does not describe its own draw fails the
    update instead of biasing it (the check round 2 lacked, now at 1e-6).
    """
    gaps: list[float] = []
    for sample in samples:
        with torch.no_grad():
            logits, _values = controller.forward(
                torch.as_tensor(sample["features"], dtype=torch.float32))
            recomputed = float(joint_log_prob(
                logits,
                torch.tensor([sample["actions"][t] for t in sample["targets"]],
                             dtype=torch.long),
                torch.tensor([1.0 if t in sample["free"] else 0.0 for t in sample["targets"]],
                             dtype=torch.float32)))
        gaps.append(abs(recomputed - float(sample["log_prob"])))
    return {"samples": len(samples),
            "max_abs_gap": max(gaps) if gaps else 0.0,
            "tolerance": common.LOG_PROB_TOLERANCE,
            "passed": (not gaps) or max(gaps) <= common.LOG_PROB_TOLERANCE}


def ppo_update(controller: Any, optimizer: torch.optim.Optimizer,
               episodes: list[dict[str, Any]], *, update_index: int) -> dict[str, Any]:
    """One PPO update over a 4-episode batch: combined advantage, two critics, no scaling."""
    samples = batch_samples(episodes)
    prefix_task, prefix_cost_bytes = _unassigned_prefix_totals(episodes)
    alignment = likelihood_alignment(controller, samples)
    if not alignment["passed"]:
        raise RuntimeError(f"update {update_index}: the rollout's logged probability does not "
                           f"describe the action that was drawn: {alignment}")
    if not samples:
        return {"samples": 0, "minibatches": 0, "optimizer_steps": 0,
                "unassigned_prefix_task_reward": prefix_task,
                "unassigned_prefix_cost_bytes": prefix_cost_bytes,
                "unassigned_prefix_cost_normalized": prefix_cost_bytes / common.B0_BYTES,
                "likelihood_alignment": alignment,
                "note": "no decision step in the batch: nothing to update"}
    shuffle = random.Random(int(hashlib.sha256(
        f"{common.DATA_ORDER_SEED}:{int(update_index)}".encode("utf-8")).hexdigest()[:16], 16))
    order = list(range(len(samples)))
    totals = {"surrogate": 0.0, "value_task": 0.0, "value_cost": 0.0, "entropy": 0.0,
              "kl": 0.0, "clip": 0.0, "grad": 0.0}
    # Registered instrumentation (ROUND24 spec §7): per-minibatch, per-module pre-clip
    # gradient L1, so every later batch can be audited for "first batch green, later batches
    # silently changed path". Reading gradients never changes the update.
    module_gradient_reads: list[dict[str, float]] = []
    modules = (("trunk", controller.inner.trunk),
               ("action_head", controller.inner.action_head),
               ("task_value_head", controller.inner.value_head),
               ("cost_head", controller.cost_head))
    minibatches = 0
    for _epoch in range(common.EPOCHS_PER_BATCH):
        shuffle.shuffle(order)
        for start in range(0, len(order), common.MINIBATCH_STEPS):
            chunk = [samples[index] for index in order[start:start + common.MINIBATCH_STEPS]]
            stacked = torch.as_tensor(np.concatenate([item["features"] for item in chunk], axis=0))
            hidden = controller.inner.trunk(stacked)
            logits = controller.inner.action_head(hidden)
            candidate_values_task = controller.inner.value_head(hidden).squeeze(-1)
            candidate_values_cost = controller.cost_head(hidden).squeeze(-1)
            row = 0
            new_log_prob: list[torch.Tensor] = []
            entropy: list[torch.Tensor] = []
            pooled_task: list[torch.Tensor] = []
            pooled_cost: list[torch.Tensor] = []
            for item in chunk:
                count = len(item["targets"])
                rows = torch.arange(row, row + count)
                row += count
                picked = torch.tensor([item["actions"][t] for t in item["targets"]],
                                      dtype=torch.long)
                usable = torch.tensor([1.0 if t in item["free"] else 0.0
                                       for t in item["targets"]], dtype=torch.float32)
                new_log_prob.append(joint_log_prob(logits[rows], picked, usable))
                entropy.append(action_entropy(logits[rows], usable))
                pooled_task.append(candidate_values_task[rows].mean())
                pooled_cost.append(candidate_values_cost[rows].mean())
            new_log_prob_t = torch.stack(new_log_prob)
            entropy_t = torch.stack(entropy)
            old_log_prob = torch.tensor([item["log_prob"] for item in chunk],
                                        dtype=torch.float32)
            advantage = torch.tensor([item["advantage"] for item in chunk], dtype=torch.float32)
            target_task = torch.tensor([item["return_task"] for item in chunk],
                                       dtype=torch.float32)
            target_cost = torch.tensor([item["return_cost"] for item in chunk],
                                       dtype=torch.float32)
            ratio = torch.exp(new_log_prob_t - old_log_prob)
            surrogate = torch.min(ratio * advantage,
                                  torch.clamp(ratio, 1.0 - common.CLIP, 1.0 + common.CLIP)
                                  * advantage)
            value_task_loss = ((torch.stack(pooled_task) - target_task) ** 2).mean()
            value_cost_loss = ((torch.stack(pooled_cost) - target_cost) ** 2).mean()
            loss = (-surrogate.mean()
                    + common.VALUE_COEFFICIENT * (value_task_loss + value_cost_loss)
                    - common.ENTROPY_BONUS * entropy_t.mean())
            optimizer.zero_grad()
            loss.backward()
            module_gradient_reads.append({
                module_name: sum(float(parameter.grad.detach().abs().sum())
                                 for parameter in module.parameters()
                                 if parameter.grad is not None)
                for module_name, module in modules})
            norm = torch.nn.utils.clip_grad_norm_(controller.parameters, common.MAX_GRAD_NORM)
            optimizer.step()
            if not (torch.isfinite(loss) and torch.isfinite(norm)):
                raise RuntimeError(f"update {update_index}: non-finite loss or gradient norm")
            totals["surrogate"] += float(surrogate.mean().item())
            totals["value_task"] += float(value_task_loss.item())
            totals["value_cost"] += float(value_cost_loss.item())
            totals["entropy"] += float(entropy_t.mean().item())
            totals["kl"] += float((old_log_prob - new_log_prob_t).mean().item())
            totals["clip"] += float(((ratio - 1.0).abs() > common.CLIP).float().mean().item())
            totals["grad"] += float(norm)
            minibatches += 1
    return {
        "samples": len(samples),
        "minibatches": minibatches,
        "optimizer_steps": minibatches,
        "epochs": common.EPOCHS_PER_BATCH,
        "unassigned_prefix_task_reward": prefix_task,
        "unassigned_prefix_cost_bytes": prefix_cost_bytes,
        "unassigned_prefix_cost_normalized": prefix_cost_bytes / common.B0_BYTES,
        "policy_surrogate": round(totals["surrogate"] / minibatches, 6),
        "value_loss_task": round(totals["value_task"] / minibatches, 6),
        "value_loss_cost": round(totals["value_cost"] / minibatches, 6),
        "entropy": round(totals["entropy"] / minibatches, 6),
        "approx_kl": round(totals["kl"] / minibatches, 6),
        "clip_fraction": round(totals["clip"] / minibatches, 4),
        "gradient_norm_mean": round(totals["grad"] / minibatches, 4),
        "module_gradient_l1_mean": {
            module_name: round(sum(read[module_name] for read in module_gradient_reads)
                               / len(module_gradient_reads), 9)
            for module_name, _module in modules} if module_gradient_reads else {},
        "module_gradient_l1_min": {
            module_name: round(min(read[module_name] for read in module_gradient_reads), 9)
            for module_name, _module in modules} if module_gradient_reads else {},
        "module_gradient_scope": "per-minibatch pre-clip L1 of every module's parameters, "
                                 "read after backward and before clipping/step",
        "likelihood_alignment": alignment,
    }


# ------------------------------------------------------------------ 对偶更新 -------------
def dual_update(lam: float, g_hat: float) -> dict[str, Any]:
    """`lambda_next = max(0, lambda + eta*g_hat)` (cost_math.dual_step); reference is a scalar."""
    lam_next = cm.dual_step(lam, g_hat, common.ETA_LAMBDA)
    return {"lambda_before": float(lam), "g_hat": float(g_hat), "eta": common.ETA_LAMBDA,
            "lambda_next": float(lam_next),
            "direction": "increase" if lam_next > float(lam) else (
                "decrease" if lam_next < float(lam) else "unchanged")}


def batch_g_hat(episodes: list[dict[str, Any]], references: dict[int, float]) -> dict[str, Any]:
    """The paired constraint statistic of one batch: mean(Y_ref - Y_policy) - delta.

    ``references`` maps scenario seed -> the cached immediate reference's on-time count; the
    reference is a plain float, so it carries no gradient into the actor.
    """
    pairs = []
    for episode in episodes:
        scenario = int(episode["scenario_seed"])
        if scenario not in references:
            raise RuntimeError(f"no cached immediate reference for scenario {scenario}")
        pairs.append({"scenario": scenario, "y_reference": float(references[scenario]),
                      "y_policy": float(episode["on_time"])})
    g_hat = cm.constraint_statistic([pair["y_reference"] for pair in pairs],
                                    [pair["y_policy"] for pair in pairs], common.DELTA)
    return {"pairs": pairs, "g_hat": float(g_hat), "delta": common.DELTA,
            "note": "reference is the cached immediate run of the same scenario (paired episode "
                    "reference); a plain scalar, never a tensor, so it has no gradient"}


# ------------------------------------------------------------------ checkpoint -------------
def save_checkpoint(controller: Any, optimizer: torch.optim.Optimizer, state: dict[str, Any],
                    path: Path, *, freeze: dict[str, Any]) -> Path:
    payload = {
        "schema": STATE_SCHEMA, "trainer_tag": TRAINER_TAG, "loss_version": LOSS_VERSION,
        "gae_schema": GAE_SCHEMA, "optimizer_schema": OPTIMIZER_SCHEMA,
        "reward_timing_schema": REWARD_TIMING_SCHEMA,
        "freeze": freeze,
        "controller": {key: value for key, value in controller.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "training_state": state,
        "rng": rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, *, freeze: dict[str, Any]) -> tuple[Any, torch.optim.Optimizer,
                                                                    dict[str, Any]]:
    """Resume only a checkpoint this round legitimately produced; refuse anything else."""
    import v2_controller

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != STATE_SCHEMA or payload.get("trainer_tag") != TRAINER_TAG:
        raise RuntimeError(f"{path.name}: not this trainer's checkpoint")
    if payload.get("loss_version") != LOSS_VERSION:
        raise RuntimeError(f"{path.name}: loss version mismatch")
    if (payload.get("gae_schema") != GAE_SCHEMA
            or payload.get("optimizer_schema") != OPTIMIZER_SCHEMA
            or payload.get("reward_timing_schema") != REWARD_TIMING_SCHEMA):
        raise RuntimeError(f"{path.name}: trainer contract schema mismatch")
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict) or (
            training_state.get("training_version") != TRAINER_TAG
            or training_state.get("loss_version") != LOSS_VERSION
            or training_state.get("state_schema") != STATE_SCHEMA
            or training_state.get("gae_schema") != GAE_SCHEMA
            or training_state.get("optimizer_schema") != OPTIMIZER_SCHEMA
            or training_state.get("reward_timing_schema") != REWARD_TIMING_SCHEMA):
        raise RuntimeError(f"{path.name}: training state version mismatch")
    if payload.get("freeze") != freeze:
        raise RuntimeError(f"{path.name}: freeze identity mismatch; refusing to resume")
    controller = v2_controller.fresh_controller(int(training_state["training_seed"]))
    # Loaded sub-module by sub-module: Stage B's controller exposes state_dict(), and touching
    # that frozen file is not necessary to restore it.
    saved = payload["controller"]
    controller.inner.trunk.load_state_dict(saved["trunk"])
    controller.inner.action_head.load_state_dict(saved["action_head"])
    controller.inner.value_head.load_state_dict(saved["value_task_head"])
    controller.cost_head.load_state_dict(saved["value_cost_head"])
    optimizer = torch.optim.Adam(controller.parameters, lr=common.LEARNING_RATE)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload.get("rng"))
    return controller, optimizer, training_state


def fresh_training_state(training_seed: int) -> dict[str, Any]:
    return {"training_version": TRAINER_TAG, "loss_version": LOSS_VERSION,
            "state_schema": STATE_SCHEMA, "gae_schema": GAE_SCHEMA,
            "optimizer_schema": OPTIMIZER_SCHEMA,
            "reward_timing_schema": REWARD_TIMING_SCHEMA,
            "training_seed": int(training_seed), "lambda": common.LAMBDA_INIT,
            "lambda_history": [], "updates": 0, "episodes_done": 0,
            "optimizer_steps": 0, "b0_bytes": common.B0_BYTES, "delta": common.DELTA,
            "eta_lambda": common.ETA_LAMBDA, "gamma": common.GAMMA,
            "gae_lambda": common.GAE_LAMBDA, "data_order_seed": common.DATA_ORDER_SEED}
