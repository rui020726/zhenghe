"""Stage B 的 actor 与双 critic：复用 pilot 轮生产网络 + 新增 cost critic 头。

构造契约（STAGE_B_PROTOCOL.md §4）：
- actor + **task critic** = `train_pilot.Controller(ACTOR_INIT_SEED)`，与 pilot/retrain 两轮
  逐字节同架构同初始化顺序（19→64→64 Tanh 主干、动作头 2 路、价值头 1 路）；
- **cost critic** = 同一主干上的 `Linear(64,1)`，在 Controller 构造之后创建，从同一种子流
  继续抽取——Stage A 已登记"新增 cost critic 改变训练状态"，所以参数哈希覆盖四项；
- 只推理：本模块没有任何 optimizer，也没有 `.step()`。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

import sb_common as common


class StageBController:
    """Shared trunk, a send/hold actor head, a task-value head and a cost-value head."""

    def __init__(self, seed: int = common.ACTOR_INIT_SEED) -> None:
        import train_pilot

        self.inner = train_pilot.Controller(int(seed))
        #: The new head the constrained objective adds. Created after the Controller so the
        #: actor/task-critic weights are the ones the pilot round's seed produces, and the cost
        #: head is a documented addition rather than a reshuffle.
        self.cost_head = torch.nn.Linear(train_pilot.HIDDEN, 1)

    # ---------------------------------------------------------------- sampler contract
    def forward(self, batch: torch.Tensor):
        """(action logits, task values) -- exactly the contract BudgetMaskedCollector expects."""
        return self.inner.forward(batch)

    def cost_values(self, batch: torch.Tensor) -> torch.Tensor:
        """The cost critic's per-candidate value of the sender's own to-go cost."""
        return self.cost_head(self.inner.trunk(batch)).squeeze(-1)

    # ------------------------------------------------------------------- parameters
    def parameter_summary(self) -> dict[str, Any]:
        """Names, shapes and a content hash of every parameter; used before and after a run."""
        tensors: dict[str, torch.Tensor] = {}
        for prefix, module in (("trunk", self.inner.trunk), ("action_head", self.inner.action_head),
                               ("value_task_head", self.inner.value_head),
                               ("value_cost_head", self.cost_head)):
            for name, tensor in module.named_parameters():
                tensors[f"{prefix}.{name}"] = tensor.detach().cpu()
        digest = hashlib.sha256()
        for name in sorted(tensors):
            digest.update(name.encode("utf-8"))
            digest.update(tensors[name].contiguous().numpy().tobytes())
        return {
            "parameter_count": int(sum(tensor.numel() for tensor in tensors.values())),
            "parameter_sha256": digest.hexdigest(),
            "shapes": {name: list(tensor.shape) for name, tensor in sorted(tensors.items())},
            "init_seed": common.ACTOR_INIT_SEED,
            "heads": ("action_head", "value_task_head", "value_cost_head"),
            "note": "actor+task critic mirror the pilot round's Controller; the cost head is the "
                    "constrained objective's addition (Stage A: new trainer state)",
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "trunk": self.inner.trunk.state_dict(),
            "action_head": self.inner.action_head.state_dict(),
            "value_task_head": self.inner.value_head.state_dict(),
            "value_cost_head": self.cost_head.state_dict(),
        }


def parameter_hash(summary: dict[str, Any]) -> str:
    return str(summary["parameter_sha256"])


def controller_identity(summary: dict[str, Any]) -> str:
    """A short stable id for logs, from the parameter hash."""
    return json.dumps(summary, sort_keys=True) and summary["parameter_sha256"][:16]
