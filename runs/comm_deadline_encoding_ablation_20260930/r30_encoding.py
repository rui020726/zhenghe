"""ROUND30 §4.4：两臂编码与显式选择编码器的作用域 collector。

接线方式（执行依据 §4.4 要求"不复制两套训练算法、不依赖全局临时替换"）：
`BudgetMaskedCollector._choose_masked` 内只有一行构造特征行
（``rows = [features(observation, item) for item in items]``）。本模块用 ROUND27 先例的
"逐字复制父类实现 + 单行登记差异"子类把这一行换成实例绑定的编码器：

- `ScopedBudgetMaskedCollector._choose_masked` 与父类逐行相同，唯一差异是特征行构造；
  该差异由 `source_diff()` 程序化登记（归一化源码比对，必须恰好一行不同且就是该行）。
- 训练 rollout 用 `ScopedStageBCollector`（先 StageBCollector 后 Scoped 的 MRO：
  `decide` 保留 Stage B 的 cost critic 记账，`_choose_masked` 用本臂编码），
  因此 tape、网络输入、cost critic 输入与 PPO 重算读到的都是同一臂的编码。
- 评估用 `ScopedBudgetMaskedCollector`（greedy/sampled 与父类同语义）。
- 编码器选择经实例属性显式绑定，不依赖模块级全局替换，多进程互不污染。

`URGENCY`/`HORIZON` 两个常量都从 comm_learning 引用，不从数据拟合。
"""

from __future__ import annotations

import difflib
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import r30_common as common  # noqa: E402  (also installs the shared sys.path via v2_common)

from budget_masked import (  # noqa: E402
    ACTION_MASK_SCHEMA,
    TAPE_SCHEMA,
    BudgetMaskedCollector,
    shadow_gate,
)
from sb_collector import StageBCollector  # noqa: E402
from integrated_sar.comm_learning import (  # noqa: E402
    EPISODE_HORIZON_STEPS,
    FEATURE_LAYOUT,
    FEATURE_NAMES,
    URGENCY_SCALE_STEPS,
    features,
)
from integrated_sar.comm_policy import SendObservation  # noqa: E402
from r29_encoding_prototype import (  # noqa: E402
    CANDIDATE_FEATURE_LAYOUT,
    SEMANTIC_REGISTRY,
    candidate_features,
)

SEND, HOLD = 1, 0
DEADLINE_COLUMN = 11

ENCODERS: dict[str, dict[str, Any]] = {
    "old": {
        "layout": FEATURE_LAYOUT,                       # comm-send-state/1
        "encode": features,                             # 生产编码器原样
        "deadline_column": f"clip(remaining_time / {URGENCY_SCALE_STEPS}, -1, 1)",
    },
    "new": {
        "layout": CANDIDATE_FEATURE_LAYOUT,            # comm-send-state/1r29-deadline-over-horizon
        "encode": candidate_features,                   # ROUND29 登记原型
        "deadline_column": f"remaining_time / {EPISODE_HORIZON_STEPS}（不截断）",
    },
}


class ScopedBudgetMaskedCollector(BudgetMaskedCollector):
    """``BudgetMaskedCollector`` with the feature rows built by a bound encoder.

    ``_choose_masked`` below is the parent's implementation verbatim except the single
    registered line that builds the rows. Everything else -- the shadow gate, the mask,
    the greedy/sampled action, the tape schema, the closure assertions -- is the archived
    code, so the archived behaviour is preserved by construction rather than by promise.
    """

    _encoder = staticmethod(features)

    def __init__(self, controller: Any, *, arm: str, forced: Any, mask: bool = True,
                 greedy: bool = False, budget_mask: bool = True) -> None:
        if arm not in ENCODERS:
            raise ValueError(f"unknown encoder arm {arm!r}; registered: {sorted(ENCODERS)}")
        super().__init__(controller, forced=forced, mask=mask, greedy=greedy,
                         budget_mask=budget_mask)
        self._arm = arm
        self._encoder = ENCODERS[arm]["encode"]
        self._layout = ENCODERS[arm]["layout"]

    # ------------------------------------------------------------------ sampling
    def _choose_masked(
        self, observation: SendObservation
    ) -> tuple[list[str], dict[str, int], list[str], float, float, dict[str, Any]]:
        items = list(observation.items)
        must = {item.target_id for item in items if self.mask and self.forced(item)}
        self.forced_sends += len(must)
        gate = shadow_gate(observation)
        budget_at_start = float(gate.remaining_bytes)
        capacity_at_start = observation.capacity_remaining
        rows = [self._encoder(observation, item) for item in items]
        matrix = torch.as_tensor(np.asarray(rows, dtype=np.float32))
        logits, values = self.controller.forward(matrix)
        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=-1)
            log_probabilities = torch.log_softmax(logits, dim=-1)
        actions: dict[str, int] = {}
        requested: dict[str, bool] = {}
        charges: dict[str, int] = {}
        executable_initial: dict[str, bool] = {}
        block_initial: dict[str, str] = {}
        block_effective: dict[str, str] = {}
        candidate_shadow: dict[str, dict[str, Any]] = {}
        actionable: list[str] = []
        masked: list[str] = []
        terms: list[float] = []
        send_probabilities: dict[str, float] = {}
        for index, item in enumerate(items):
            target = item.target_id
            requested[target] = False
            if target in must:
                # The silence bound's decision, not ours: it goes out and it is billed, and it
                # carries no likelihood because nothing chose it.
                actions[target] = SEND
                requested[target] = True
                continue
            charge = item.attempt_charge_bytes
            executable = item.send_executable
            reason = item.send_block_reason
            if charge is None or executable is None or reason is None:
                raise RuntimeError(
                    f"candidate {target!r} has no attempt charge metadata "
                    f"(charge={charge!r}, executable={executable!r}, reason={reason!r}); a "
                    "budget-masked collector cannot know its action set without it, and "
                    "guessing 'affordable' is the defect this class exists to remove"
                )
            charges[target] = int(charge)
            executable_initial[target] = bool(executable)
            block_initial[target] = str(reason)
            before = (float(gate.remaining_bytes), gate.sends_this_step)
            honours = gate.can_send(int(charge))
            block_effective[target] = gate.why_blocked(int(charge))
            candidate_shadow[target] = {
                "budget_before": before[0],
                "capacity_sent_before": before[1],
                "budget_after": float(gate.remaining_bytes),
                "capacity_sent_after": gate.sends_this_step,
                "honours_this_candidate": bool(honours),
            }
            if not honours:
                # Unaffordable in the order the transport would take this candidate: HOLD is not
                # a choice, it is the only legal action, so it must not earn actor gradient.
                actions[target] = HOLD
                masked.append(target)
                self.distribution_reads[
                    "masked_by_credit"
                    if block_effective[target] == "budget_exhausted"
                    else "masked_by_capacity"
                ] += 1
                continue
            actionable.append(target)
            if self.greedy:
                action = SEND if probabilities[index, SEND] >= probabilities[index, HOLD] else HOLD
            else:
                action = int(torch.multinomial(probabilities[index], num_samples=1).item())
            actions[target] = action
            requested[target] = bool(action == SEND)
            terms.append(float(log_probabilities[index, action].item()))
            send_probabilities[target] = float(probabilities[index, SEND].item())
            self.distribution_reads["free_candidates"] += 1
            self.distribution_reads["wanted_send"] += int(action == SEND)
            if action == SEND:
                # Only a sampled send spends the shadow. The gate's own charge() is the rule for
                # how much, so an arithmetic mistake here shows up as a refusal at execution.
                gate.charge(int(charge))
            candidate_shadow[target]["budget_after"] = float(gate.remaining_bytes)
            candidate_shadow[target]["capacity_sent_after"] = gate.sends_this_step
        asked = [target for target in actionable if actions[target] == SEND]
        tape: dict[str, Any] = {
            "schema": TAPE_SCHEMA,
            "action_mask_schema": ACTION_MASK_SCHEMA,
            "step": int(observation.step),
            "targets": [item.target_id for item in items],
            "features": [list(row) for row in rows],
            "actions": actions,
            # ``free`` keeps the parent's key and the parent's meaning as a set of *actionable*
            # targets: the two must not diverge, or the update and the audit read different sets.
            "free": list(actionable),
            "actionable": list(actionable),
            "forced": sorted(must),
            "masked": list(masked),
            "masked_reasons": {target: block_effective[target] for target in masked},
            "attempt_charge_bytes": charges,
            "send_executable_initial": executable_initial,
            "send_block_reason_initial": block_initial,
            "candidate_shadow": candidate_shadow,
            "shadow_budget_before": budget_at_start,
            "shadow_budget_after": float(gate.remaining_bytes),
            "shadow_capacity_before": capacity_at_start,
            "shadow_capacity_after": (
                None
                if capacity_at_start is None
                else int(capacity_at_start) - int(gate.sends_this_step)
            ),
            "requested": requested,
            "log_prob": float(sum(terms)) if terms else 0.0,
            "value": float(values.mean().item()) if items else 0.0,
            "send_probabilities": send_probabilities,
        }
        self._assert_closure(tape)
        return asked, actions, tape["free"], tape["log_prob"], tape["value"], tape


class ScopedStageBCollector(StageBCollector, ScopedBudgetMaskedCollector):
    """The training collector: Stage B's cost-critic tape entry, the bound encoder's rows.

    MRO: ``decide`` resolves to StageBCollector (which calls ``super().decide`` and then reads
    the tape just written, so its ``cost_value`` and ``sender_state`` entries see this arm's
    features); ``choose``/``_choose_masked`` resolve to the scoped implementation.
    """

    def __init__(self, controller: Any, *, arm: str) -> None:
        ScopedBudgetMaskedCollector.__init__(
            self, controller, arm=arm, forced=lambda item: False, mask=False,
            greedy=False, budget_mask=True)


def scoped_training_collector(controller: Any, arm: str) -> ScopedStageBCollector:
    return ScopedStageBCollector(controller, arm=arm)


def scoped_eval_collector(controller: Any, arm: str, *, greedy: bool) -> ScopedBudgetMaskedCollector:
    return ScopedBudgetMaskedCollector(controller, arm=arm, forced=lambda item: False,
                                       mask=False, greedy=greedy, budget_mask=True)


# ------------------------------------------------------------------ 登记：源码差异
def _normalized(source: str) -> list[str]:
    return [line.rstrip() for line in source.splitlines() if line.strip()]


def source_diff() -> dict[str, Any]:
    """Programmatic registration that the scoped ``_choose_masked`` differs from the
    parent's in exactly one line, and that the line is the feature-row construction."""
    parent = inspect.getsource(BudgetMaskedCollector._choose_masked)
    child = inspect.getsource(ScopedBudgetMaskedCollector._choose_masked)
    parent_lines = _normalized(parent)
    child_lines = _normalized(child)
    matcher = difflib.SequenceMatcher(a=parent_lines, b=child_lines)
    changed: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed.append({
            "op": tag,
            "parent_lines": parent_lines[i1:i2],
            "child_lines": child_lines[j1:j2],
        })
    single = len(changed) == 1 and changed[0]["op"] == "replace" and len(
        changed[0]["parent_lines"]) == 1 and len(changed[0]["child_lines"]) == 1
    registered_line_ok = bool(single) and "self._encoder(observation, item)" in changed[0][
        "child_lines"][0] and changed[0]["parent_lines"][0].strip().startswith("rows = [features(")
    return {
        "method": "difflib.SequenceMatcher over blank-stripped source lines of the parent and "
                  "subclass _choose_masked",
        "changed_blocks": changed,
        "exactly_one_changed_line": bool(single),
        "changed_line_is_the_feature_row_construction": registered_line_ok,
        "passed": bool(single and registered_line_ok),
        "note": "the shadow gate, action choice, tape schema and closure assertions are the "
                "parent's code verbatim; only the encoder used to build the rows is bound "
                "per instance",
    }


def arm_declaration() -> dict[str, Any]:
    return {
        "arms": {name: {"feature_layout": spec["layout"], "deadline_column": spec["deadline_column"],
                        "feature_names": list(FEATURE_NAMES), "dim": len(FEATURE_NAMES),
                        "urgency_scale_steps": URGENCY_SCALE_STEPS,
                        "episode_horizon_steps": EPISODE_HORIZON_STEPS}
                 for name, spec in ENCODERS.items()},
        "round29_semantic_registry": SEMANTIC_REGISTRY,
        "only_difference": "the definition of feature column 12 (index 11); the other 18 "
                          "columns, their order, the tape schema, the mask semantics, the "
                          "controller, the trainer and every hyperparameter are shared code",
        "checkpoint_compatibility": "same parameter shape does not make arms interchangeable: "
                                   "a checkpoint trained under one layout must never be "
                                   "resumed or evaluated under the other; r30_train.encode "
                                   "this by carrying the arm and layout inside the freeze "
                                   "record that the save/restore contract compares",
    }
