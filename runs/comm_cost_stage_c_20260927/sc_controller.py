"""Stage C 的通信网络：直接复用 Stage B 的双 critic 控制器（不重写第二套）。

`StageBController`（runs/comm_cost_stage_b_20260927/sb_controller.py）= pilot 轮
`Controller(seed)`（19→64→64 Tanh 主干 + 动作头 + 任务价值头）+ Stage B 登记的 cost 价值头。
Stage C 只从初值训练：每个训练 seed 用 `Controller(training_seed)` 构造，不加载任何旧权重、
不 warm start（与 pilot/retrain 两轮"every arm starts at Controller(training_seed)"同规）。
"""

from __future__ import annotations

from typing import Any

from sb_controller import StageBController  # noqa: F401  (re-exported for the harness)


class StageCController(StageBController):
    """The Stage B dual-critic controller plus the parameter list an optimiser expects.

    Stage B's controller deliberately exposes ``state_dict``/``parameter_summary`` and keeps
    the pilot's ``parameters`` list inside ``inner``; the optimiser in this round needs the
    list itself, so this subclass exposes it without rewriting the frozen Stage B file.
    """

    @property
    def parameters(self) -> list:
        return list(self.inner.parameters)


TRAINER_TAG = "constrained-cost-v1"
LOSS_VERSION = "constrained-lagrangian/1"
STATE_SCHEMA = "constrained-cost-state/1"


def fresh_controller(training_seed: int) -> StageCController:
    """``Controller(training_seed)`` and nothing else -- the only place weights come from."""
    return StageCController(int(training_seed))


def parameter_identity(controller: StageBController) -> dict[str, Any]:
    return controller.parameter_summary()
