"""ROUND32 任务 B：归约改变了什么（**0 环境启动、0 新流**）。

全部测试穿过**实际 actor loss 路径**：合成 tape 的 likelihood 用冻结件自己的
`onpolicy.joint_log_prob` 生成，样本用冻结的 `v2_trainer.batch_samples` 构造，actor 项用
`r32_trainer.actor_surrogate_mean` —— 即 `ppo_update` 实际放进 loss 的那个表达式。
不测"新写的均值函数"。

覆盖执行依据 §3.3 的 8 项，另加两项：
- `premise_check`：原实现是否真的符合 §3.1 的归约前提（否则按 §3.4 停止）；
- `T9`：`ALL_ROWS` 与冻结 `v2_trainer.ppo_update` 在同一合成批上**逐 optimizer 步权重哈希相同**
  （合成 optimizer.step 单独记账，上限 64）。

合成 tape 只用于数学核验；真实训练不人为追加或删去行（§3.3.6）。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import r32_common as common
import r32_trainer as rtrainer
import v2_trainer as base
from onpolicy import action_entropy, joint_log_prob

FEATURE_DIM = 19
SYNTHETIC_SEED_START = 20261001
SYNTHETIC_SEED_RUNS = 20261002
HORIZON = 60

#: 每个 episode 取 3 行，四种模式轮换 => 12 行里 9 行 active、3 行整行被 mask。
POOL = (
    (("t1", "t2", "t3"), ("t1", "t2", "t3")),   # all free
    (("t1", "t2", "t3"), ("t1",)),              # one free, two masked by the sequential budget
    (("t1", "t2"), ()),                         # fully masked row
    (("t1", "t2", "t3", "t4"), ("t2",)),        # one free
)
#: every row has at least one free candidate, so M == N (T2 uses this instead of mutating rows)
POOL_ALL_ACTIVE = (
    (("t1", "t2", "t3"), ("t1", "t2", "t3")),
    (("t1", "t2"), ("t1", "t2")),
    (("t1", "t2", "t3", "t4"), ("t2",)),
    (("t1",), ("t1",)),
)


def feature_matrix(count: int) -> np.ndarray:
    """A fixed non-degenerate float32 pattern so logits are not all equal."""
    return np.asarray(
        [[(index + 1) * 0.07 * (column % 5 - 2) + 0.01 * column for column in range(FEATURE_DIM)]
         for index in range(count)], dtype=np.float32)


def make_tape(controller: Any, step: int, targets: tuple[str, ...],
              free: tuple[str, ...]) -> dict[str, Any]:
    """One synthetic decision row: saved features, saved actions, saved likelihood."""
    matrix = torch.as_tensor(feature_matrix(len(targets)))
    with torch.no_grad():
        logits, values = controller.forward(matrix)
        probabilities = torch.softmax(logits, dim=-1)
    usable = torch.tensor([1.0 if t in free else 0.0 for t in targets], dtype=torch.float32)
    generator = torch.Generator().manual_seed(step * 977 + len(targets))
    actions: dict[str, int] = {}
    for index, target in enumerate(targets):
        actions[target] = (int(torch.multinomial(probabilities[index], 1, generator=generator)
                               .item()) if target in free else 0)
    picked = torch.tensor([actions[t] for t in targets], dtype=torch.long)
    return {
        "schema": "onpolicy-tape/2", "action_mask_schema": "budget-sequential-mask/1",
        "step": int(step), "targets": list(targets),
        "features": [row.tolist() for row in matrix], "actions": actions,
        "free": list(free), "actionable": list(free), "forced": [],
        "masked": [t for t in targets if t not in free],
        "log_prob": float(joint_log_prob(logits, picked, usable).item()),
        "value": float(values.mean().item()),
        "cost_value": float(controller.cost_values(matrix).mean().item()),
        "send_probabilities": {t: float(probabilities[index, 1].item())
                               for index, t in enumerate(targets) if t in free},
    }


def recompute_log_prob(controller: Any, tape: dict[str, Any]) -> float:
    matrix = torch.as_tensor(np.asarray(tape["features"], dtype=np.float32))
    with torch.no_grad():
        logits, _values = controller.forward(matrix)
    usable = torch.tensor([1.0 if t in tape["actionable"] else 0.0 for t in tape["targets"]],
                          dtype=torch.float32)
    picked = torch.tensor([int(tape["actions"][t]) for t in tape["targets"]], dtype=torch.long)
    return float(joint_log_prob(logits, picked, usable).item())


class _Collector:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records


def build_episodes(controller: Any, *, episodes: int = 4, lam: float = 1.0,
                   pool: tuple = POOL, hold_only: bool = False) -> list[dict[str, Any]]:
    """A synthetic batch with mixed active / fully masked rows and a real sequential-mask shape."""
    out: list[dict[str, Any]] = []
    for episode_index in range(episodes):
        tapes: list[dict[str, Any]] = []
        for offset in range(3):
            targets, free = pool[(episode_index + offset) % len(pool)]
            tape = make_tape(controller, step=1 + 4 * (3 * episode_index + offset),
                             targets=targets, free=free)
            if hold_only:
                for target in tape["actionable"]:
                    tape["actions"][target] = 0
                tape["log_prob"] = recompute_log_prob(controller, tape)
            tapes.append(tape)
        steps = [int(tape["step"]) for tape in tapes]
        out.append({
            "collector": _Collector(tapes),
            "cost_stream": {step: {"airtime_bytes": 1200.0 + 100.0 * index}
                            for index, step in enumerate(steps)},
            "task_stream": {step: float(index % 2) for index, step in enumerate(steps)},
            "steps": HORIZON + 40 * episode_index,
            "lambda": lam, "episode_index": episode_index,
            "scenario_seed": 1300000 + episode_index,
            "on_time": sum(int(step_value) for step_value in (0.0, 1.0, 0.0)),
            "spent_bytes": int(sum(1200.0 + 100.0 * index for index in range(3))),
        })
    return out


def parameter_sha(controller: Any) -> str:
    digest = hashlib.sha256()
    modules = (("trunk", controller.inner.trunk), ("action_head", controller.inner.action_head),
               ("value_task_head", controller.inner.value_head),
               ("value_cost_head", controller.cost_head))
    for prefix, module in modules:
        for name, tensor in sorted(module.named_parameters()):
            digest.update(f"{prefix}.{name}".encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def minibatches_for(n_samples: int) -> int:
    return (int(common.v2c.EPOCHS_PER_BATCH)
            * math.ceil(n_samples / int(common.v2c.MINIBATCH_STEPS)))


def actor_term_gradient(controller: Any, samples: list[dict[str, Any]], chunk: list[int],
                        reduction: str) -> dict[str, Any]:
    """Gradient of the actor term exactly as `ppo_update` builds it, at the current weights."""
    chunk_samples = [samples[index] for index in chunk]
    stacked = torch.as_tensor(np.concatenate([item["features"] for item in chunk_samples],
                                             axis=0))
    hidden = controller.inner.trunk(stacked)
    logits = controller.inner.action_head(hidden)
    new_log_prob: list[torch.Tensor] = []
    row = 0
    for item in chunk_samples:
        rows = torch.arange(row, row + len(item["targets"]))
        row += len(item["targets"])
        picked = torch.tensor([item["actions"][t] for t in item["targets"]], dtype=torch.long)
        usable = torch.tensor([1.0 if t in item["free"] else 0.0 for t in item["targets"]],
                              dtype=torch.float32)
        new_log_prob.append(joint_log_prob(logits[rows], picked, usable))
    new_log_prob_t = torch.stack(new_log_prob)
    old_log_prob = torch.tensor([item["log_prob"] for item in chunk_samples], dtype=torch.float32)
    advantage = torch.tensor([item["advantage"] for item in chunk_samples], dtype=torch.float32)
    ratio = torch.exp(new_log_prob_t - old_log_prob)
    surrogate = torch.min(ratio * advantage,
                          torch.clamp(ratio, 1.0 - float(common.v2c.CLIP),
                                      1.0 + float(common.v2c.CLIP)) * advantage)
    row_active = rtrainer.row_active_mask(chunk_samples)
    mean, n_rows, m_active = rtrainer.actor_surrogate_mean(surrogate, row_active, reduction)
    loss = -mean
    parameters = list(controller.parameters)
    # the actor term never touches either critic head, so those gradients are unused (exactly 0)
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    flat = torch.cat([(g if g is not None else torch.zeros_like(parameter)).detach().reshape(-1)
                      for g, parameter in zip(grads, parameters)])
    return {"loss": float(loss.detach().item()), "n_rows": n_rows, "m_active_rows": m_active,
            "flat": flat, "l1": float(flat.abs().sum().item()), "l2": float(flat.norm().item()),
            "surrogate_mean": float(mean.detach().item()),
            "all_rows_reference_mean": float(surrogate.mean().detach().item())}


def full_loss_gradient(controller: Any, samples: list[dict[str, Any]], chunk: list[int],
                       reduction: str) -> dict[str, Any]:
    """The complete registered loss (actor + entropy + two critics) at the current weights."""
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
    target_task = torch.tensor([item["return_task"] for item in chunk_samples],
                               dtype=torch.float32)
    target_cost = torch.tensor([item["return_cost"] for item in chunk_samples],
                               dtype=torch.float32)
    ratio = torch.exp(new_log_prob_t - old_log_prob)
    surrogate = torch.min(ratio * advantage,
                          torch.clamp(ratio, 1.0 - float(common.v2c.CLIP),
                                      1.0 + float(common.v2c.CLIP)) * advantage)
    value_task_loss = ((torch.stack(pooled_task) - target_task) ** 2).mean()
    value_cost_loss = ((torch.stack(pooled_cost) - target_cost) ** 2).mean()
    actor_surrogate, _, _ = rtrainer.actor_surrogate_mean(
        surrogate, rtrainer.row_active_mask(chunk_samples), reduction)
    loss = (-actor_surrogate
            + float(common.v2c.VALUE_COEFFICIENT) * (value_task_loss + value_cost_loss)
            - float(common.v2c.ENTROPY_BONUS) * entropy_t.mean())
    grads = torch.autograd.grad(loss, list(controller.parameters))
    flat = torch.cat([g.detach().reshape(-1) for g in grads])
    return {"loss": float(loss.detach().item()), "flat": flat,
            "entropy_term": float(entropy_t.mean().detach().item()),
            "value_loss_task": float(value_task_loss.detach().item()),
            "value_loss_cost": float(value_cost_loss.detach().item()),
            "approx_kl": float((old_log_prob - new_log_prob_t).mean().item()),
            "clip_fraction": float(((ratio - 1.0).abs()
                                    > float(common.v2c.CLIP)).float().mean().item())}


def premise_check() -> dict[str, Any]:
    """Do the archived implementation's shapes match the reduction premises of §3.1?"""
    trainer_source = Path(base.__file__).read_text(encoding="utf-8")
    onpolicy_source = Path(common.ROOT / "runs/comm_tight_budget_20260919/onpolicy.py").read_text(
        encoding="utf-8")
    collector_source = Path(common.ROOT / "runs/comm_budget_action_mask_fix_20260922/"
                                            "budget_masked.py").read_text(encoding="utf-8")
    return {
        "parent_actor_term": "-surrogate.mean() == -(1/N) * sum_i s_i over the chunk's rows",
        "surrogate_is_per_row_tensor": "surrogate = torch.min(ratio * advantage," in trainer_source,
        "actor_term_divides_by_row_count": "loss = (-surrogate.mean()" in trainer_source,
        "inactive_row_likelihood_is_identically_zero": (
            "(log_probabilities[rows, picked] * usable).sum()" in onpolicy_source),
        "free_set_is_saved_at_masked_selection": ('"free": list(actionable)' in collector_source),
        "one_sample_is_one_decision_row_not_one_candidate": (
            "for index, tape in enumerate(tapes)" in trainer_source),
        "verdict": ("premises hold. N is the number of logged decision rows in the chunk, every "
                    "row enters the denominator, and a row with no free candidate carries a "
                    "likelihood that is a constant zero, so its actor gradient is zero at any "
                    "weights -- not only where the ratio happens to be 1"),
    }


def real_tape_wiring() -> dict[str, Any]:
    """Recount N and M on saved ROUND30 old-arm tapes: sequential-mask wiring, 0 launches."""
    paths = sorted(common.R30_DIR.glob("rows/r30_train_old_s11_e*_a*.json"))[:6]
    cells: list[dict[str, Any]] = []
    for path in paths:
        row = common.read_json(path)
        tapes = list(common.iter_jsonl(Path(row["artefacts"]["tape"])))
        n_rows = len(tapes)
        m_rows = sum(1 for tape in tapes if tape["actionable"])
        held_only = sum(1 for tape in tapes if tape["actionable"] and all(
            int(tape["actions"][target]) == 0 for target in tape["actionable"]))
        cells.append({
            "row": path.name, "n_rows": n_rows, "m_active_rows": m_rows,
            "m_over_n": round(m_rows / n_rows, 6) if n_rows else None,
            "fully_masked_rows": n_rows - m_rows,
            "rows_with_forced_candidates": sum(1 for tape in tapes if tape.get("forced")),
            "rows_whose_free_candidates_all_chose_HOLD": held_only,
            "actionable_equals_free": all(tape["actionable"] == tape["free"] for tape in tapes),
            "candidate_classes_close": all(
                sorted(set(tape["masked"]) | set(tape["actionable"])
                       | set(tape.get("forced", []))) == sorted(set(tape["targets"]))
                for tape in tapes),
            "masked_rows_never_appear_in_send_probabilities": all(
                target not in tape["send_probabilities"] for tape in tapes
                for target in tape["masked"]),
        })
    return {"source": "ROUND30 old-arm training tapes, already metered (0 new launches)",
            "cells": cells,
            "all_actionable_equals_free": all(c["actionable_equals_free"] for c in cells),
            "all_candidate_classes_close": all(c["candidate_classes_close"] for c in cells),
            "all_masked_rows_have_no_likelihood_entry": all(
                c["masked_rows_never_appear_in_send_probabilities"] for c in cells),
            "fully_masked_rows_present_in_every_cell": all(c["fully_masked_rows"] > 0
                                                           for c in cells)}


def main() -> int:
    import v2_controller

    results: dict[str, Any] = {"round": "32",
                               "section": "reduction mathematics and wiring (0 launches)",
                               "premise_check": premise_check()}
    controller = v2_controller.fresh_controller(SYNTHETIC_SEED_START)
    start_sha = parameter_sha(controller)

    episodes = build_episodes(controller)
    samples = rtrainer.batch_samples(episodes)
    chunk = list(range(len(samples)))
    n_rows = len(samples)
    m_rows = sum(1 for item in samples if item["free"])
    results["batch_shape"] = {"samples": n_rows, "active_rows": m_rows,
                              "inactive_rows": n_rows - m_rows,
                              "reduction_differs_here": 0 < m_rows < n_rows}

    # ---- T1 mixed rows: grad(ACTIVE) == (N / M) * grad(ALL), actor term only
    grad_all = actor_term_gradient(controller, samples, chunk, "ALL_ROWS")
    grad_active = actor_term_gradient(controller, samples, chunk, "ACTIVE_ROWS")
    scaled = grad_all["flat"] * (n_rows / m_rows)
    t1_max_abs = float((grad_active["flat"] - scaled).abs().max().item())
    results["T1_mixed_rows"] = {
        "identity": "grad(ACTIVE_ROWS actor term) == (N/M) * grad(ALL_ROWS actor term)",
        "n_rows": n_rows, "m_active_rows": m_rows, "expected_factor": round(n_rows / m_rows, 9),
        "loss_all": grad_all["loss"], "loss_active": grad_active["loss"],
        "max_abs_difference_after_scaling": t1_max_abs,
        "l2_all": grad_all["l2"], "l2_active": grad_active["l2"],
        "l2_ratio": round(grad_active["l2"] / grad_all["l2"], 9) if grad_all["l2"] else None,
        "tolerance": "float32: the identity is exact in real arithmetic; the residual is "
                     "summation rounding (observed ~2e-08 on a gradient of norm ~1e-01)",
        "passed": bool(t1_max_abs < 1e-6 and grad_all["l2"] > 0
                       and abs(grad_active["l2"] / grad_all["l2"] - n_rows / m_rows) < 1e-6)}

    # ---- T1b an inactive row's likelihood has no gradient at ANY weights
    inactive = next(item for item in samples if not item["free"])
    matrix = torch.as_tensor(inactive["features"])
    probe_grads: list[float] = []
    for shift in (0.0, 0.013, -0.021):
        net = copy.deepcopy(controller)
        with torch.no_grad():
            for parameter in net.parameters:
                parameter.add_(shift)
        logits = net.inner.action_head(net.inner.trunk(matrix))
        usable = torch.tensor([1.0 if t in inactive["free"] else 0.0 for t in inactive["targets"]],
                              dtype=torch.float32)
        picked = torch.tensor([inactive["actions"][t] for t in inactive["targets"]],
                              dtype=torch.long)
        grads = torch.autograd.grad(joint_log_prob(logits, picked, usable),
                                    list(net.parameters), allow_unused=True)
        flat = torch.cat([(g if g is not None else torch.zeros_like(p)).detach().reshape(-1)
                          for g, p in zip(grads, net.parameters)])
        probe_grads.append(float(flat.abs().max().item()))
    results["T1b_inactive_row_zero_gradient"] = {
        "max_abs_actor_gradient_at_weight_shifts": probe_grads,
        "shifts": [0.0, 0.013, -0.021],
        "note": "the archived joint_log_prob multiplies every candidate term by the saved usable "
                "flag, so a row with no free candidate is a constant zero for the actor at any "
                "weights; only its critic and surrogate VALUE still enter the mean denominator",
        "passed": bool(all(value == 0.0 for value in probe_grads))}

    # ---- T2 all rows active: the two reductions agree
    all_active = rtrainer.batch_samples(build_episodes(controller, pool=POOL_ALL_ACTIVE))
    chunk2 = list(range(len(all_active)))
    g2_all = actor_term_gradient(controller, all_active, chunk2, "ALL_ROWS")
    g2_active = actor_term_gradient(controller, all_active, chunk2, "ACTIVE_ROWS")
    results["T2_all_active"] = {
        "n_rows": g2_all["n_rows"], "m_active_rows": g2_active["m_active_rows"],
        "loss_difference": abs(g2_all["loss"] - g2_active["loss"]),
        "gradient_max_abs_difference": float((g2_all["flat"] - g2_active["flat"]).abs().max()),
        "passed": bool(g2_all["m_active_rows"] == g2_all["n_rows"]
                       and abs(g2_all["loss"] - g2_active["loss"]) < 1e-6
                       and float((g2_all["flat"] - g2_active["flat"]).abs().max()) < 1e-6)}

    # ---- T3 fully masked: zero actor gradient, finite loss, critic rows kept
    masked_rows = [(1, ("t1", "t2"), ()), (5, ("t1", "t2", "t3"), ()), (9, ("t1",), ())]
    tapes3 = [make_tape(controller, step, targets, free) for step, targets, free in masked_rows]
    ep3 = {"collector": _Collector(tapes3), "cost_stream": {1: {"airtime_bytes": 800.0},
                                                            5: {"airtime_bytes": 700.0}},
           "task_stream": {1: 1.0, 5: 0.0}, "steps": HORIZON, "lambda": 1.0,
           "episode_index": 0, "scenario_seed": 1300000, "on_time": 1, "spent_bytes": 1500}
    samples3 = base.batch_samples([ep3])
    chunk3 = list(range(len(samples3)))
    g3_all = actor_term_gradient(controller, samples3, chunk3, "ALL_ROWS")
    g3_active = actor_term_gradient(controller, samples3, chunk3, "ACTIVE_ROWS")
    full3 = full_loss_gradient(controller, samples3, chunk3, "ACTIVE_ROWS")
    results["T3_all_masked"] = {
        "n_rows": g3_all["n_rows"], "m_active_rows": g3_active["m_active_rows"],
        "actor_loss_value": g3_active["loss"],
        "actor_gradient_l2_active_rows": g3_active["l2"],
        "actor_gradient_l2_all_rows": g3_all["l2"],
        "critic_rows_retained": len(samples3),
        "value_loss_task": full3["value_loss_task"], "value_loss_cost": full3["value_loss_cost"],
        "full_loss_finite": bool(np.isfinite(full3["loss"])),
        "note": "only the actor term must vanish; the critics keep every masked row and the "
                "shared trunk is still moved by them, so a fully masked minibatch is NOT a no-op",
        "passed": bool(g3_active["m_active_rows"] == 0 and g3_active["l2"] == 0.0
                       and g3_active["loss"] == 0.0 and np.isfinite(full3["loss"])
                       and len(samples3) == 3 and np.isfinite(full3["value_loss_task"]))}

    # ---- T4 a legal HOLD on a free candidate still makes the row active
    # a genuine all-active layout, all actions forced to HOLD: M must still equal N
    hold_samples = rtrainer.batch_samples(
        build_episodes(controller, pool=POOL_ALL_ACTIVE, hold_only=True))
    hold_active = sum(1 for item in hold_samples if item["free"])
    every_action_hold = all(int(item["actions"][t]) == 0 for item in hold_samples
                            for t in item["targets"])
    results["T4_hold_counts_as_active"] = {
        "rows": len(hold_samples), "active_rows": hold_active,
        "all_recorded_actions_are_HOLD": bool(every_action_hold),
        "note": "a sampled HOLD on a free candidate is still a decision the policy took: the row "
                "carries likelihood and must count in M. The comparison below is the sharp form: "
                "M depends on which candidates were free, never on which action was chosen.",
        "M_with_hold_actions": hold_active,
        "M_with_the_sampled_actions": sum(1 for item in rtrainer.batch_samples(
            build_episodes(controller, pool=POOL_ALL_ACTIVE)) if item["free"]),
        "passed": bool(hold_active == len(hold_samples) and every_action_hold
                       and len(hold_samples) > 0)}

    # ---- T5 sequential budget mask, on real saved tapes
    results["T5_real_saved_tapes"] = real_tape_wiring()

    # ---- T6 synthetic zero-gradient padding
    pad_tapes = [make_tape(controller, 200 + index, ("p1", "p2"), ()) for index in range(4)]
    pad_episode = {"collector": _Collector(pad_tapes), "cost_stream": {}, "task_stream": {},
                   "steps": HORIZON, "lambda": 1.0, "episode_index": 99,
                   "scenario_seed": 1300098, "on_time": 0, "spent_bytes": 0}
    pad_samples = base.batch_samples([pad_episode])
    combined = samples + pad_samples
    combined_chunk = list(range(len(combined)))
    c_all = actor_term_gradient(controller, combined, combined_chunk, "ALL_ROWS")
    c_active = actor_term_gradient(controller, combined, combined_chunk, "ACTIVE_ROWS")
    results["T6_zero_gradient_padding"] = {
        "padding_rows": len(pad_samples),
        "all_padding_rows_inactive": all(not item["free"] for item in pad_samples),
        "N_before": n_rows, "N_after": c_all["n_rows"],
        "M_before": m_rows, "M_after": c_active["m_active_rows"],
        "all_rows_l2_ratio_after_over_before": round(c_all["l2"] / grad_all["l2"], 9),
        "expected_all_rows_ratio": round(n_rows / (n_rows + len(pad_samples)), 9),
        "active_rows_gradient_max_abs_difference": float(
            (c_active["flat"] - grad_active["flat"]).abs().max().item()),
        "note": "synthetic-only counterexample. Adding zero-gradient rows shrinks ALL_ROWS' actor "
                "gradient by N/N' and leaves ACTIVE_ROWS' untouched. Real training never adds or "
                "drops rows; this only isolates what the denominator does.",
        "passed": bool(abs(c_all["l2"] / grad_all["l2"] - n_rows / (n_rows + len(pad_samples)))
                       < 1e-5 and c_active["m_active_rows"] == m_rows
                       and float((c_active["flat"] - grad_active["flat"]).abs().max()) < 1e-9)}

    # ---- T7 everything except the actor term is identical between the arms
    f_all = full_loss_gradient(controller, samples, chunk, "ALL_ROWS")
    f_active = full_loss_gradient(controller, samples, chunk, "ACTIVE_ROWS")
    delta_full = f_active["flat"] - f_all["flat"]
    delta_actor = grad_active["flat"] - grad_all["flat"]
    r32_samples = rtrainer.batch_samples(build_episodes(v2_controller.fresh_controller(
        SYNTHETIC_SEED_START)))
    base_samples = base.batch_samples(build_episodes(v2_controller.fresh_controller(
        SYNTHETIC_SEED_START)))
    results["T7_other_losses_unchanged"] = {
        "entropy_term_equal": f_all["entropy_term"] == f_active["entropy_term"],
        "value_loss_task_equal": f_all["value_loss_task"] == f_active["value_loss_task"],
        "value_loss_cost_equal": f_all["value_loss_cost"] == f_active["value_loss_cost"],
        "approx_kl_equal": f_all["approx_kl"] == f_active["approx_kl"],
        "clip_fraction_equal": f_all["clip_fraction"] == f_active["clip_fraction"],
        "advantage_construction_identical_to_frozen_trainer": bool(
            len(r32_samples) == len(base_samples)
            and all(a["advantage"] == b["advantage"] and a["return_task"] == b["return_task"]
                    and a["return_cost"] == b["return_cost"] and a["log_prob"] == b["log_prob"]
                    and np.array_equal(a["features"], b["features"])
                    for a, b in zip(r32_samples, base_samples))),
        "gradient_difference_is_exactly_the_actor_term_max_abs": float(
            (delta_full - delta_actor).abs().max().item()),
        "scope_note": "the ablation deliberately changes the actor's weight RELATIVE to entropy "
                      "and the two critics; it does not change entropy, either critic loss, the "
                      "advantage construction, the existing KL or the clip fields",
    }
    results["T7_other_losses_unchanged"]["passed"] = bool(
        results["T7_other_losses_unchanged"]["entropy_term_equal"]
        and results["T7_other_losses_unchanged"]["value_loss_task_equal"]
        and results["T7_other_losses_unchanged"]["value_loss_cost_equal"]
        and results["T7_other_losses_unchanged"]["approx_kl_equal"]
        and results["T7_other_losses_unchanged"]["clip_fraction_equal"]
        and results["T7_other_losses_unchanged"]["advantage_construction_identical_to_frozen_trainer"]
        and results["T7_other_losses_unchanged"][
            "gradient_difference_is_exactly_the_actor_term_max_abs"] < 1e-6)

    # ---- T8 restore contract: reduction mode is recorded and enforced
    path = common.CHECKPOINTS / "r32_synth_checkpoint_probe.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    net = v2_controller.fresh_controller(SYNTHETIC_SEED_START)
    optimizer = torch.optim.Adam(net.parameters, lr=common.v2c.LEARNING_RATE)
    state = rtrainer.fresh_training_state(SYNTHETIC_SEED_START)
    state["episodes_done"] = 4
    rtrainer.save_checkpoint(net, optimizer, state, path,
                            freeze={"round": "32", "reduction": "ALL_ROWS"},
                            reduction="ALL_ROWS")
    refusals: list[str] = []
    try:
        rtrainer.load_checkpoint(path, freeze={"round": "32", "reduction": "ALL_ROWS"},
                                 reduction="ACTIVE_ROWS")
        refusals.append("NOT REFUSED: cross-reduction resume was allowed")
    except RuntimeError as error:
        refusals.append(f"cross-reduction resume refused: {error}")
    r30_milestone = common.R30_DIR / "checkpoints" / "r30_train_old_s11_ep0.pt"
    try:
        rtrainer.load_checkpoint(r30_milestone, freeze={"any": "freeze"}, reduction="ALL_ROWS")
        refusals.append("NOT REFUSED: an archived ROUND30 checkpoint was loadable")
    except RuntimeError as error:
        refusals.append(f"archived checkpoint refused: {error}")
    reloaded, _opt, reloaded_state = rtrainer.load_checkpoint(
        path, freeze={"round": "32", "reduction": "ALL_ROWS"}, reduction="ALL_ROWS")
    results["T8_restore_contract"] = {
        "refusals": refusals,
        "reduction_mode_in_payload": "reduction_mode" in str(torch.load(
            path, map_location="cpu", weights_only=False).keys()),
        "reduction_mode_in_training_state": reloaded_state.get("reduction_mode"),
        "same_mode_resume_roundtrip_parameter_sha_equal": parameter_sha(reloaded) == start_sha,
        "passed": bool(len(refusals) == 2 and all("refused" in entry for entry in refusals)
                       and reloaded_state.get("reduction_mode") == "ALL_ROWS"
                       and parameter_sha(reloaded) == start_sha)}

    # ---- T9 ALL_ROWS reproduces the frozen trainer step for step
    steps_per_call = minibatches_for(n_rows)
    common.book_synthetic_updates(4 * steps_per_call,
                                  "T9 ALL_ROWS equivalence vs frozen v2_trainer "
                                  "(2 calls per trainer)", key="T9")
    hashes: dict[str, list[str]] = {}
    frozen_reports: dict[str, dict[str, Any]] = {}
    for label, use_r32 in (("r32_trainer", True), ("v2_trainer", False)):
        torch.manual_seed(SYNTHETIC_SEED_RUNS)
        net = v2_controller.fresh_controller(SYNTHETIC_SEED_RUNS)
        opt = torch.optim.Adam(net.parameters, lr=common.v2c.LEARNING_RATE)
        after: list[str] = []
        for _round in range(2):
            # the tape must always describe its own draw, so the batch is re-sampled from the
            # weights that are current at this call -- exactly what real training does between
            # batches. Both trainers see the same deterministic sampler, so any difference in
            # the resulting weights can only come from the code path.
            batch = build_episodes(net)
            if use_r32:
                report = rtrainer.ppo_update(net, opt, batch, update_index=0,
                                             reduction="ALL_ROWS")
            else:
                report = base.ppo_update(net, opt, batch, update_index=0)
            after.append(parameter_sha(net))
            frozen_reports[label] = report
        hashes[label] = after
    frozen_fields = ("policy_surrogate", "value_loss_task", "value_loss_cost", "entropy",
                     "approx_kl", "clip_fraction", "gradient_norm_mean", "samples",
                     "minibatches", "optimizer_steps", "epochs",
                     "unassigned_prefix_task_reward", "unassigned_prefix_cost_bytes")
    scalar_equality = {
        field: frozen_reports["r32_trainer"][field] == frozen_reports["v2_trainer"][field]
        for field in frozen_fields}
    module_gradient_equality = (
        frozen_reports["r32_trainer"]["module_gradient_l1_mean"]
        == frozen_reports["v2_trainer"]["module_gradient_l1_mean"])
    results["T9_all_rows_equals_frozen_trainer"] = {
        "protocol": "same fresh Controller(seed), same synthetic batch, Adam(lr=3e-4), the full "
                    "minibatch sequence run twice, weight hash after every call",
        "minibatches_per_call": steps_per_call,
        "hash_after_call1": hashes,
        "equal_after_call1": hashes["r32_trainer"][0] == hashes["v2_trainer"][0],
        "equal_after_call2": hashes["r32_trainer"][1] == hashes["v2_trainer"][1],
        "frozen_scalar_fields_equal": scalar_equality,
        "module_gradient_l1_mean_equal": bool(module_gradient_equality),
        "weights_actually_moved": hashes["r32_trainer"][0] != start_sha,
        "synthetic_steps_booked_so_far": common.synthetic_updates_used(),
        "synthetic_cap": common.SYNTHETIC_UPDATE_CAP,
        "passed": bool(hashes["r32_trainer"][0] == hashes["v2_trainer"][0]
                       and hashes["r32_trainer"][1] == hashes["v2_trainer"][1]
                       and all(scalar_equality.values()) and module_gradient_equality
                       and hashes["r32_trainer"][0] != start_sha)}

    # ---- T10 the reduction is effective on the real update path, and only there
    common.book_synthetic_updates(2 * steps_per_call,
                                  "T10 per-arm reduction-effect probe (one call per reduction)",
                                  key="T10")
    reports: dict[str, Any] = {}
    for reduction in common.ARM_ORDER:
        torch.manual_seed(SYNTHETIC_SEED_START)
        net = v2_controller.fresh_controller(SYNTHETIC_SEED_START)
        opt = torch.optim.Adam(net.parameters, lr=common.v2c.LEARNING_RATE)
        batch = build_episodes(net)
        reports[reduction] = rtrainer.ppo_update(net, opt, batch, update_index=0,
                                                 reduction=reduction)
    read_all = reports["ALL_ROWS"]["minibatch_reads"]
    read_active = reports["ACTIVE_ROWS"]["minibatch_reads"]
    # only the FIRST minibatch has identical inputs on both arms: from minibatch 1 onward the two
    # nets already differ, and that divergence IS the intervention rather than a failure
    untouched = ("entropy_term", "value_loss_task", "value_loss_cost", "approx_kl",
                 "clip_fraction", "n_rows", "m_active_rows", "lambda", "ratio_min", "ratio_max")
    outcomes = ("actor_surrogate_mean", "pre_clip_total_norm")
    first_match = {key: read_all[0][key] == read_active[0][key] for key in untouched}
    outcome_differ = {key: read_all[0][key] != read_active[0][key] for key in outcomes}
    later_divergence = sorted({
        key for key in untouched + outcomes
        if any(read_all[index][key] != read_active[index][key]
               for index in range(1, len(read_all)))})
    active_definition_gap = max(
        (abs(read["actor_surrogate_mean"]
             - read["actor_surrogate_mean_active_only_reference"])
         for read in read_active if read["m_active_rows"]), default=0.0)
    all_definition_gap = max(
        abs(read["actor_surrogate_mean"] - read["actor_surrogate_mean_all_rows_reference"])
        for read in read_all)
    definitional_active = bool(active_definition_gap < 1e-6)
    definitional_all = bool(all_definition_gap < 1e-12)
    results["T10_reduction_is_effective"] = {
        "minibatches": reports["ALL_ROWS"]["minibatches"],
        "n_rows_identical": [r["n_rows"] for r in read_all] == [r["n_rows"] for r in read_active],
        "m_rows_identical": [r["m_active_rows"] for r in read_all]
                            == [r["m_active_rows"] for r in read_active],
        "first_minibatch_untouched_fields_identical": first_match,
        "first_minibatch_outcome_fields_differ": {
            **outcome_differ,
            "meaning": "the actor surrogate mean and the pre-clip total gradient norm already "
                       "differ at minibatch 0: that IS the intervention being measured, so "
                       "requiring equality there would be wrong"},
        "definitional_checks": {
            "ACTIVE_equals_mean_over_active_rows": definitional_active,
            "ALL_equals_mean_over_all_rows": definitional_all,
            "active_definition_max_abs_gap": active_definition_gap,
            "all_definition_max_abs_gap": all_definition_gap,
            "note": "the loss SCALAR is not the N/M multiple of the other arm's scalar: a masked "
                    "row's surrogate value is a non-zero constant even though its actor gradient "
                    "is exactly zero. The gradient identity is checked separately in T1."},
        "fields_that_differ_after_minibatch_0": later_divergence,
        "divergence_after_minibatch_0_is_expected": bool(later_divergence),
        "all_rows_scalars_match_frozen_report_shape": sorted(
            {"samples", "minibatches", "optimizer_steps", "epochs", "reduction", "lambda",
             "unassigned_prefix_task_reward", "unassigned_prefix_cost_bytes",
             "unassigned_prefix_cost_normalized", "policy_surrogate",
             "policy_surrogate_all_rows_reference", "value_loss_task",
             "value_loss_cost", "entropy", "approx_kl", "clip_fraction", "gradient_norm_mean",
             "module_gradient_l1_mean", "module_gradient_l1_min", "module_gradient_scope",
             "likelihood_alignment", "minibatch_reads", "active_row_share_mean",
             "clipping_minibatches", "non_finite_minibatches"})
        == sorted(reports["ALL_ROWS"].keys()),
        "frozen_scalars_all_rows_vs_base_trainer": "see T9_all_rows_equals_frozen_trainer."
                                                   "frozen_scalar_fields_equal",
        "synthetic_steps_booked_total": common.synthetic_updates_used()}
    results["T10_reduction_is_effective"]["passed"] = bool(
        reports["ALL_ROWS"]["samples"] == reports["ACTIVE_ROWS"]["samples"]
        and results["T10_reduction_is_effective"]["n_rows_identical"]
        and results["T10_reduction_is_effective"]["m_rows_identical"]
        and all(first_match.values())
        and all(outcome_differ.values())
        and definitional_active and definitional_all)

    results["T5_real_saved_tapes"]["passed"] = bool(
        results["T5_real_saved_tapes"]["all_actionable_equals_free"]
        and results["T5_real_saved_tapes"]["all_candidate_classes_close"]
        and results["T5_real_saved_tapes"]["all_masked_rows_have_no_likelihood_entry"]
        and results["T5_real_saved_tapes"]["fully_masked_rows_present_in_every_cell"])
    results["source_diff"] = rtrainer.source_diff()
    results["unchanged_machinery_identity"] = rtrainer.identity_assertions()
    results["synthetic_update_budget"] = {"used": common.synthetic_updates_used(),
                                          "cap": common.SYNTHETIC_UPDATE_CAP}
    results["environment_launches_used"] = 0

    checks = [results["T1_mixed_rows"]["passed"],
              results["T1b_inactive_row_zero_gradient"]["passed"],
              results["T2_all_active"]["passed"], results["T3_all_masked"]["passed"],
              results["T4_hold_counts_as_active"]["passed"],
              results["T5_real_saved_tapes"]["passed"],
              results["T6_zero_gradient_padding"]["passed"],
              results["T7_other_losses_unchanged"]["passed"],
              results["T8_restore_contract"]["passed"],
              results["T9_all_rows_equals_frozen_trainer"]["passed"],
              results["T10_reduction_is_effective"]["passed"],
              results["source_diff"]["passed"]]
    results["premise_consistent"] = bool(
        all(value is True for key, value in results["premise_check"].items()
            if isinstance(value, bool)))
    results["passed"] = bool(all(checks) and results["premise_consistent"])

    target = common.OUTPUT_DIR / "REDUCTION_MATH.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({"passed": results["passed"],
                      "premise_consistent": results["premise_consistent"],
                      "each": {key: (results[key].get("passed") if isinstance(results[key], dict)
                                     else None) for key in sorted(results)
                               if key.startswith("T")},
                      "T1": {k: results["T1_mixed_rows"][k] for k in
                             ("n_rows", "m_active_rows", "expected_factor",
                              "max_abs_difference_after_scaling", "l2_ratio")},
                      "T1b": results["T1b_inactive_row_zero_gradient"]["max_abs_actor_gradient_at_weight_shifts"],
                      "T6": {k: results["T6_zero_gradient_padding"][k] for k in
                             ("all_rows_l2_ratio_after_over_before", "expected_all_rows_ratio",
                              "active_rows_gradient_max_abs_difference")},
                      "T9": {k: results["T9_all_rows_equals_frozen_trainer"][k] for k in
                             ("equal_after_call1", "equal_after_call2", "weights_actually_moved")},
                      "T10": {key: results["T10_reduction_is_effective"][key] for key in
                              ("first_minibatch_untouched_fields_identical",
                               "first_minibatch_outcome_fields_differ",
                               "definitional_checks")},
                      "unexplained_blocks": results["source_diff"]["unexplained_blocks"],
                      "block_categories": results["source_diff"]["block_categories"],
                      "removed_parent_lines": results["source_diff"]["removed_parent_lines"],
                      "synthetic_used": common.synthetic_updates_used()},
                     ensure_ascii=False, indent=1))
    print("wrote", target)
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
