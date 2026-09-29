"""ROUND32 的 actor surrogate 归约：本轮**唯一主动改变量**。

设计纪律：
- 优势、GAE、奖励时间归属、likelihood 对齐、λ 更新、batch_g_hat、两个 critic 的样本与归约、
  entropy 的归约与系数、minibatch 构造与顺序、clip、全局梯度裁剪、optimizer 与学习率
  **全部直接引用冻结件**（`v2_trainer` / `cost_math` 的同一函数对象），本文件不复制这些数学；
- `ppo_update` 是冻结实现的逐行副本，**连 import 别名也与父件相同**（`common` 仍是
  `v2_common`），因此 difflib 的差异行只能落在两类：研究变量（actor surrogate 归约）与
  被动读数（N/M、各 loss、KL、裁剪前后范数等）；
- `ALL_ROWS` 分支保留冻结表达式 `-surrogate.mean()` 逐字，所以与旧训练器逐步等价
  （由 `r32_math.py` 的 T9 用权重哈希证明，而不是靠声明）。

归约定义（执行依据 §3.1）：一个 minibatch 里参与 surrogate 归约的记录行数 `N`，其中"在保存的
真实顺序 mask 路径下至少有一个带策略 likelihood 权重的自由候选"的行数 `M`：

    ALL_ROWS:    actor_loss = -sum(s_i, i<N)  / N
    ACTIVE_ROWS: actor_loss = -sum(s_i, i in active) / max(M, 1)

必须分清两件事（本轮实测）：`joint_log_prob` 对无自由候选的行返回 `(... * usable).sum()`
= 精确 0 的**常量**，所以该行 **actor 梯度恒为 0**（在任何权重下都成立，不只是 ratio=1 处），
但它的 `s_i = A_i` **数值不为 0**。因此 `N/M` 关系只对**梯度**成立，对 loss **标量**不成立
（标量还差在被丢掉的常量项上）。两臂的 `policy_surrogate` 读数因此不可直接互比，本文件同时
记录与臂无关的 `policy_surrogate_all_rows_reference`。
"""

from __future__ import annotations

import difflib
import hashlib
import inspect
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

import r32_common as round32
import v2_common as common          # same alias as the frozen trainer, on purpose
import v2_trainer as base
import cost_math as cm              # same alias as the frozen trainer, on purpose
from onpolicy import action_entropy, joint_log_prob, restore_rng, rng_state

REDUCTIONS = tuple(round32.ARM_ORDER)
REDUCTION_TRAINER_TAG = "constrained-cost-v2-reduction/1"

# ---- 引用冻结件：函数对象同一性由 r32_math/r32_preflight 断言，不在此重新实现 -----------
batch_samples = base.batch_samples
_bundle = base._bundle
_unassigned_prefix_totals = base._unassigned_prefix_totals
likelihood_alignment = base.likelihood_alignment
dual_update = base.dual_update
batch_g_hat = base.batch_g_hat
fresh_training_state = base.fresh_training_state
TRAINER_TAG = base.TRAINER_TAG
LOSS_VERSION = base.LOSS_VERSION
STATE_SCHEMA = base.STATE_SCHEMA
GAE_SCHEMA = base.GAE_SCHEMA
OPTIMIZER_SCHEMA = base.OPTIMIZER_SCHEMA
REWARD_TIMING_SCHEMA = base.REWARD_TIMING_SCHEMA


def row_active_mask(chunk: list[dict[str, Any]]) -> torch.Tensor:
    """1.0 for logged rows that carry at least one free (likelihood-weighted) candidate."""
    return torch.tensor([1.0 if item["free"] else 0.0 for item in chunk], dtype=torch.float32)


def actor_surrogate_mean(surrogate: torch.Tensor, row_active: torch.Tensor,
                         reduction: str) -> tuple[torch.Tensor, int, int]:
    """The only research variable: which rows are summed, and by what they are divided."""
    n_rows = int(surrogate.shape[0])
    m_active = int(row_active.sum().item())
    if reduction == "ALL_ROWS":
        return surrogate.mean(), n_rows, m_active
    if reduction == "ACTIVE_ROWS":
        return (surrogate * row_active).sum() / max(m_active, 1), n_rows, m_active
    raise ValueError(f"unknown reduction {reduction!r}; registered: {REDUCTIONS}")


def _finite_or_none(value: Any) -> bool:
    return value is None or bool(np.isfinite(float(value)))


def _module_grad_l1(module: Any) -> float:
    """Passive reading: sum of |grad| over the parameters of one module that carry a grad."""
    return sum(float(parameter.grad.detach().abs().sum())
               for parameter in module.parameters() if parameter.grad is not None)


def _module_grad_l2(module: Any) -> float:
    """Passive reading: L2 norm of one module's concatenated gradient vector."""
    pieces = [parameter.grad.detach().reshape(-1) for parameter in module.parameters()
              if parameter.grad is not None]
    if not pieces:
        return 0.0
    return float(torch.cat(pieces).norm().item())


def ppo_update(controller: Any, optimizer: torch.optim.Optimizer,
               episodes: list[dict[str, Any]], *, update_index: int,
               reduction: str) -> dict[str, Any]:
    """One PPO update over a 4-episode batch: combined advantage, two critics, no scaling."""
    if reduction not in REDUCTIONS:
        raise ValueError(f"update {update_index}: unknown reduction {reduction!r}")
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
                "reduction": reduction, "minibatch_reads": [],
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
    minibatch_reads: list[dict[str, Any]] = []
    surrogate_all_reads: list[float] = []
    active_share_reads: list[float] = []
    clipping_reads: list[int] = []
    non_finite_reads: list[int] = []
    modules = (("trunk", controller.inner.trunk),
               ("action_head", controller.inner.action_head),
               ("task_value_head", controller.inner.value_head),
               ("cost_head", controller.cost_head))
    lambdas = sorted({float(episode["lambda"]) for episode in episodes})
    if len(lambdas) != 1:
        raise RuntimeError(f"update {update_index}: the batch's episodes did not share one "
                           f"lambda: {lambdas}")
    batch_lambda = lambdas[0]
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
            row_active = row_active_mask(chunk)
            actor_surrogate, n_rows, m_active = actor_surrogate_mean(
                surrogate, row_active, reduction)
            loss = (-actor_surrogate
                    + common.VALUE_COEFFICIENT * (value_task_loss + value_cost_loss)
                    - common.ENTROPY_BONUS * entropy_t.mean())
            optimizer.zero_grad()
            loss.backward()
            module_gradient_reads.append({
                module_name: sum(float(parameter.grad.detach().abs().sum())
                                 for parameter in module.parameters()
                                 if parameter.grad is not None)
                for module_name, module in modules})
            # ---- passive readings only: nothing between here and optimizer.step() mutates a
            # tensor, the optimizer or the RNG, so the update itself is untouched
            pre_clip_module_l1 = {module_name: _module_grad_l1(module)
                                  for module_name, module in modules}
            pre_clip_module_l2 = {module_name: _module_grad_l2(module)
                                  for module_name, module in modules}
            surrogate_all = float(surrogate.mean().detach().item())
            read_row: dict[str, Any] = {
                "minibatch": minibatches,
                "epoch": int(_epoch),
                "reduction": reduction,
                "lambda": batch_lambda,
                "n_rows": int(n_rows),
                "m_active_rows": int(m_active),
                "m_over_n": round(m_active / n_rows, 9) if n_rows else None,
                "actor_surrogate_mean": float(actor_surrogate.detach().item()),
                "actor_surrogate_mean_all_rows_reference": surrogate_all,
                "actor_surrogate_mean_active_only_reference": float(
                    surrogate[row_active.bool()].mean().detach().item()) if m_active else None,
                "entropy_term": float(entropy_t.mean().detach().item()),
                "value_loss_task": float(value_task_loss.detach().item()),
                "value_loss_cost": float(value_cost_loss.detach().item()),
                "loss_total": float(loss.detach().item()),
                "approx_kl": float((old_log_prob - new_log_prob_t).mean().item()),
                "clip_fraction": float(((ratio - 1.0).abs() > common.CLIP).float().mean().item()),
                "ratio_min": float(ratio.detach().min().item()),
                "ratio_max": float(ratio.detach().max().item()),
                "advantage_mean": float(advantage.mean().item()),
                "advantage_mean_active_only": float(
                    advantage[row_active.bool()].mean().item()) if m_active else None,
                "pre_clip_module_l1": pre_clip_module_l1,
                "pre_clip_module_l2": pre_clip_module_l2,
            }
            norm = torch.nn.utils.clip_grad_norm_(controller.parameters, common.MAX_GRAD_NORM)
            read_row["pre_clip_total_norm"] = float(norm.detach().item())
            read_row["post_clip_total_norm"] = min(read_row["pre_clip_total_norm"],
                                                   float(common.MAX_GRAD_NORM))
            read_row["clipping_triggered"] = bool(read_row["pre_clip_total_norm"]
                                                  > float(common.MAX_GRAD_NORM))
            read_row["clip_scale"] = (round(read_row["post_clip_total_norm"]
                                            / read_row["pre_clip_total_norm"], 9)
                                      if read_row["pre_clip_total_norm"] > 0 else 1.0)
            read_row["all_finite"] = bool(all(
                _finite_or_none(read_row[key]) for key in (
                    "actor_surrogate_mean", "entropy_term", "value_loss_task", "value_loss_cost",
                    "loss_total", "approx_kl", "clip_fraction", "pre_clip_total_norm",
                    "ratio_min", "ratio_max", "advantage_mean"))
                and all(_finite_or_none(value) for value in pre_clip_module_l1.values())
                and all(_finite_or_none(value) for value in pre_clip_module_l2.values()))
            minibatch_reads.append({
                key: (round(value, 12) if isinstance(value, float) else value)
                if key not in ("pre_clip_module_l1", "pre_clip_module_l2")
                else {name: round(amount, 12) for name, amount in value.items()}
                for key, value in read_row.items()})
            surrogate_all_reads.append(surrogate_all)
            active_share_reads.append(m_active / n_rows if n_rows else 0.0)
            clipping_reads.append(int(read_row["clipping_triggered"]))
            non_finite_reads.append(int(not read_row["all_finite"]))
            optimizer.step()
            if not (torch.isfinite(loss) and torch.isfinite(norm)):
                raise RuntimeError(f"update {update_index}: non-finite loss or gradient norm")
            totals["surrogate"] += float(actor_surrogate.item())
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
        "reduction": reduction,
        "lambda": batch_lambda,
        "unassigned_prefix_task_reward": prefix_task,
        "unassigned_prefix_cost_bytes": prefix_cost_bytes,
        "unassigned_prefix_cost_normalized": prefix_cost_bytes / common.B0_BYTES,
        "policy_surrogate": round(totals["surrogate"] / minibatches, 6),
        "policy_surrogate_all_rows_reference": round(
            sum(surrogate_all_reads) / len(surrogate_all_reads), 6),
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
        "minibatch_reads": minibatch_reads,
        "active_row_share_mean": round(sum(active_share_reads) / len(active_share_reads), 6),
        "clipping_minibatches": sum(clipping_reads),
        "non_finite_minibatches": sum(non_finite_reads),
    }


# ------------------------------------------------------------------ checkpoint 契约 -------------
def save_checkpoint(controller: Any, optimizer: torch.optim.Optimizer, state: dict[str, Any],
                    path: Path, *, freeze: dict[str, Any], reduction: str) -> Path:
    """The frozen v2 payload plus the reduction mode, so cross-reduction resume is impossible."""
    payload = {
        "schema": STATE_SCHEMA, "trainer_tag": TRAINER_TAG, "loss_version": LOSS_VERSION,
        "gae_schema": GAE_SCHEMA, "optimizer_schema": OPTIMIZER_SCHEMA,
        "reward_timing_schema": REWARD_TIMING_SCHEMA,
        "reduction_mode": str(reduction), "reduction_trainer_tag": REDUCTION_TRAINER_TAG,
        "freeze": freeze,
        "controller": {key: value for key, value in controller.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "training_state": {**state, "reduction_mode": str(reduction)},
        "rng": rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, *, freeze: dict[str, Any],
                    reduction: str) -> tuple[Any, torch.optim.Optimizer, dict[str, Any]]:
    """Restore only this round's checkpoint of the same reduction mode; anything else fails."""
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
    if payload.get("reduction_mode") != reduction:
        raise RuntimeError(f"{path.name}: reduction mode is {payload.get('reduction_mode')!r} "
                           f"but {reduction!r} was requested; a checkpoint trained under one "
                           "reduction must never be resumed under the other")
    if payload.get("reduction_trainer_tag") != REDUCTION_TRAINER_TAG:
        raise RuntimeError(f"{path.name}: no ROUND32 reduction tag; an archived round's "
                           "checkpoint cannot be hot-started here")
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict) or (
            training_state.get("training_version") != TRAINER_TAG
            or training_state.get("loss_version") != LOSS_VERSION
            or training_state.get("state_schema") != STATE_SCHEMA
            or training_state.get("reduction_mode") != reduction):
        raise RuntimeError(f"{path.name}: training state version mismatch")
    if payload.get("freeze") != freeze:
        raise RuntimeError(f"{path.name}: freeze identity mismatch; refusing to resume")
    controller = v2_controller.fresh_controller(int(training_state["training_seed"]))
    saved = payload["controller"]
    controller.inner.trunk.load_state_dict(saved["trunk"])
    controller.inner.action_head.load_state_dict(saved["action_head"])
    controller.inner.value_head.load_state_dict(saved["value_task_head"])
    controller.cost_head.load_state_dict(saved["value_cost_head"])
    optimizer = torch.optim.Adam(controller.parameters, lr=common.LEARNING_RATE)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload.get("rng"))
    return controller, optimizer, training_state


# ------------------------------------------------------------------ 登记：逐行差异
_RESEARCH_PATTERN = re.compile(
    r"row_active|actor_surrogate|reduction|m_active|n_rows|max\(m_active|REDUCTIONS|"
    r"actor_surrogate\.item")
_READS_PATTERN = re.compile(
    r"minibatch_reads|read_row|pre_clip|post_clip|clipping|clip_scale|all_finite|"
    r"_finite_or_none|ratio_min|ratio_max|advantage_mean|m_over_n|surrogate_all_reads|"
    r"active_share_reads|clipping_reads|non_finite_reads|active_row_share|"
    r"non_finite_minibatches|all_rows_reference|active_only_reference|lambdas|batch_lambda|"
    r"did not share one|_module_grad|surrogate_all\b|minibatch\"|epoch\"|entropy_term|"
    r"value_loss_task\"|value_loss_cost\"|loss_total|approx_kl\"|clip_fraction\"|for module_name")
_SIGNATURE_PATTERN = re.compile(r"update_index: int,|reduction: str\}|reduction: str,")
_CONTINUATION_PATTERN = re.compile(r"^\}|^\)|^\(|^float\(common\.MAX_GRAD_NORM\)|^key: \(round|"
                                   r"^else \{name: round|^> float\(common\.MAX_GRAD_NORM\)|"
                                   r"^for module_name, module in modules\}")


def _clean(source: str) -> list[str]:
    return [line.strip() for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")]


def _classify_block(lines: list[str]) -> str:
    """One changed block gets one category; a block that matches none is a finding, not noise."""
    joined = " ".join(lines)
    if _RESEARCH_PATTERN.search(joined):
        return "research_variable"
    if _READS_PATTERN.search(joined):
        return "passive_read"
    if _SIGNATURE_PATTERN.search(joined):
        return "reduction_plumbing"
    if all(_CONTINUATION_PATTERN.search(line) for line in lines):
        return "passive_read_continuation"
    return "unexplained"


def source_diff() -> dict[str, Any]:
    """Classify every changed source block of ppo_update as research variable or passive read."""
    parent = _clean(inspect.getsource(base.ppo_update))
    child = _clean(inspect.getsource(ppo_update))
    matcher = difflib.SequenceMatcher(a=parent, b=child)
    blocks: list[dict[str, Any]] = []
    removed: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        removed.extend(parent[i1:i2])
        blocks.append({"opcode": tag, "parent_lines": parent[i1:i2],
                       "child_lines": child[j1:j2],
                       "category": _classify_block(child[j1:j2]) if child[j1:j2] else tag})
    categories: dict[str, int] = {}
    for block in blocks:
        categories[str(block["category"])] = categories.get(str(block["category"]), 0) + 1
    actor_line_removed = [line for line in removed if line.startswith("loss = (-surrogate.mean()")]
    unexplained = [block for block in blocks if block["category"] == "unexplained"]
    unexplained_removed = [
        line for line in removed
        if not (line.startswith("loss = (-surrogate.mean()")
                or line.startswith("episodes: list[dict[str, Any]], *, update_index: int)")
                or line.startswith('totals["surrogate"] += float(surrogate.mean().item())'))]
    return {
        "method": "difflib.SequenceMatcher over comment-stripped source lines of "
                  "v2_trainer.ppo_update and r32_trainer.ppo_update; each changed BLOCK is "
                  "classified as research variable / reduction plumbing / passive read, and any "
                  "block that fits none is reported instead of being waved through",
        "changed_blocks": blocks,
        "block_categories": categories,
        "unexplained_blocks": unexplained,
        "removed_parent_lines": removed,
        "removed_line_categories": {
            line: ("research_variable -- the actor surrogate reduction itself"
                   if line.startswith("loss = (-surrogate.mean()")
                   else "reduction_plumbing -- the new reduction keyword argument"
                   if line.startswith("episodes: list[dict[str, Any]], *, update_index: int)")
                   else "reported_field_only -- the policy_surrogate accumulator now reads the "
                        "arm's own mean; for ALL_ROWS that is the same value (proved by T9)"
                   if line.startswith('totals["surrogate"] += float(surrogate.mean().item())')
                   else "UNEXPLAINED")
            for line in removed},
        "unexplained_removed_lines": unexplained_removed,
        "removed_actor_reduction_line": actor_line_removed,
        "parent_signature_line": parent[0],
        "child_signature_line": child[0],
        "research_variable_blocks": [block for block in blocks
                                     if block["category"] == "research_variable"],
        "passed": bool(not unexplained and not unexplained_removed
                       and len(actor_line_removed) == 1),
        "note": "the ablation deliberately changes the actor's weight relative to entropy and the "
                "two critics; nothing else in the loss expression is touched",
    }


def identity_assertions() -> dict[str, Any]:
    """The unchanged-machinery claim is an object-identity claim, checked rather than asserted."""
    return {
        "batch_samples_is_frozen_object": batch_samples is base.batch_samples,
        "bundle_is_frozen_object": _bundle is base._bundle,
        "prefix_totals_is_frozen_object": (_unassigned_prefix_totals
                                           is base._unassigned_prefix_totals),
        "likelihood_alignment_is_frozen_object": (
            likelihood_alignment is base.likelihood_alignment),
        "dual_update_is_frozen_object": dual_update is base.dual_update,
        "batch_g_hat_is_frozen_object": batch_g_hat is base.batch_g_hat,
        "cost_math_module_is_the_frozen_one": cm is base.cm,
        "gae_is_the_frozen_cost_math_object": cm.gae is base.cm.gae,
        "actor_advantage_is_the_frozen_cost_math_object": (
            cm.actor_advantage is base.cm.actor_advantage),
        "trainer_tag_inherited": TRAINER_TAG == base.TRAINER_TAG,
        "loss_version_inherited": LOSS_VERSION == base.LOSS_VERSION,
        "state_schema_inherited": STATE_SCHEMA == base.STATE_SCHEMA,
        "gae_schema_inherited": GAE_SCHEMA == base.GAE_SCHEMA,
        "registered_scalars_are_the_frozen_module": (
            common is base.common),
        "registered_scalar_values": {name: getattr(common, name) for name in (
            "CLIP", "VALUE_COEFFICIENT", "ENTROPY_BONUS", "MAX_GRAD_NORM", "LEARNING_RATE",
            "EPOCHS_PER_BATCH", "MINIBATCH_STEPS", "GAMMA", "GAE_LAMBDA", "B0_BYTES", "DELTA",
            "LAMBDA_INIT", "ETA_LAMBDA", "LOG_PROB_TOLERANCE")},
    }
