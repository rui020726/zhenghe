"""Stage B collector：预算 mask 采样器 + 规则等价注入 + 双 critic 的 tape 扩展。

- `StageBCollector` 子类化第七轮 `BudgetMaskedCollector`（shadow gate 就是生产
  `CommBudget`，masked 项零似然零熵，tape 记全字段），只在 `decide` 末尾补 cost critic
  读数与 sender-state 摘要 —— 采样/掩码/计费逻辑一律不复制；
- `immediate_equivalent_fn` / `hold_equivalent_fn` 是 B1 的注入谓词：I_send 复现该配置
  immediate 的**实际请求谓词** `pending_event`，并按传输层自己的 `transport_trim_key`
  排序（从 `comm_policy` 导入，不复制公式）；I_hold 恒空。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from budget_masked import BudgetMaskedCollector
from integrated_sar.comm_policy import SendObservation, transport_trim_key


def immediate_equivalent_fn(observation: SendObservation) -> tuple[str, ...]:
    """The native ``immediate`` request predicate, in the transport's own emission order.

    ``ImmediatePolicy`` does not claim an ordering, so the relay trims its selection with
    ``transport_order`` (i.e. ``transport_trim_key``). An injected path that returned the same
    *set* in observation order would be trimmed identically only by luck; sorting here is what
    makes the two paths decide the same emissions in the same order.
    """
    pending = [item for item in observation.items if item.pending_event]
    pending.sort(key=transport_trim_key)
    return tuple(item.target_id for item in pending)


def hold_equivalent_fn(observation: SendObservation) -> tuple[str, ...]:
    """Always-hold: request nothing. The transport speaks for nobody under policy_owned_timing."""
    return ()


class StageBCollector(BudgetMaskedCollector):
    """The masked sampler with a cost-critic reading added to each step's tape."""

    def __init__(self, controller: Any) -> None:
        # forced is identically False: policy_owned_timing with no declared bound means the
        # transport never speaks for the arm, so there is no forced set to mask.
        super().__init__(controller, forced=lambda item: False, mask=False, greedy=False,
                         budget_mask=True)

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        asked = super().decide(observation)
        tape = self.records[-1] if self.records else None
        if tape is not None:
            matrix = torch.as_tensor(np.asarray(tape["features"], dtype=np.float32))
            with torch.no_grad():
                tape["cost_value"] = float(
                    self.controller.cost_values(matrix).mean().item()
                )
            tape["sender_state"] = {
                "step": int(observation.step),
                "candidates": len(observation.items),
                "backlog_targets": int(observation.backlog_targets),
                "remaining_budget_bytes": (
                    None if observation.remaining_budget_bytes == float("inf")
                    else float(observation.remaining_budget_bytes)
                ),
                "budget_bytes_per_episode": observation.budget_bytes_per_episode,
                "capacity_remaining": observation.capacity_remaining,
                "subscribers": int(observation.subscribers),
            }
        return asked

    def tape_summary(self) -> dict[str, Any]:
        """What the rollout recorded, without the per-step bodies."""
        steps = [int(tape["step"]) for tape in self.records]
        masked = sum(len(tape["masked"]) for tape in self.records)
        actionable = sum(len(tape["actionable"]) for tape in self.records)
        forced = sum(len(tape["forced"]) for tape in self.records)
        requested = sum(1 for tape in self.records for value in tape["requested"].values() if value)
        return {
            "decision_steps": len(self.records),
            "first_decision_step": min(steps) if steps else None,
            "last_decision_step": max(steps) if steps else None,
            "candidates_total": int(self.candidates),
            "actionable_total": actionable,
            "masked_total": masked,
            "forced_total": forced,
            "requested_sends_total": requested,
            "log_prob_total": round(sum(float(tape["log_prob"]) for tape in self.records), 9),
            "candidate_classes_disjoint": all(
                not (set(tape["forced"]) & set(tape["masked"]))
                and not (set(tape["forced"]) & set(tape["actionable"]))
                and not (set(tape["masked"]) & set(tape["actionable"]))
                for tape in self.records
            ),
            "candidate_classes_close": all(
                set(tape["forced"]) | set(tape["masked"]) | set(tape["actionable"])
                == set(tape["targets"]) for tape in self.records
            ),
            "masked_carry_no_likelihood": all(
                target not in tape["requested"] or not tape["requested"][target]
                for tape in self.records for target in tape["masked"]
            ),
            "masking_reasons": _masked_reasons(self.records),
            "timing": self.timing(),
        }


def bundle_rewards(stream: dict[int, float], decision_steps: list[int],
                   horizon: int) -> dict[int, float]:
    """Bundle each step's amount into the next decision step (registered wiring convention).

    Rewards that accrue between two decision steps are booked on the later one, so the sum
    over decision steps closes onto the episode totals exactly; the closure itself is still
    checked against the executor's ledger rather than trusted from the bundling.
    """
    bundled = {step: 0.0 for step in decision_steps}
    for step in range(1, horizon + 1):
        target = next((d for d in decision_steps if d >= step), decision_steps[-1])
        bundled[target] += float(stream.get(step, 0.0))
    return bundled


def _masked_reasons(records: list[dict[str, Any]]) -> dict[str, int]:
    reasons: dict[str, int] = {}
    for tape in records:
        for reason in tape.get("masked_reasons", {}).values():
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    return dict(sorted(reasons.items()))
