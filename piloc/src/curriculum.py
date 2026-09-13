"""课程门控：只按任务指标升级，绝不读取累计奖励。

替换旧的“历史最高累计奖励 + 200 回合未刷新即升级”规则。旧规则有两个问题：

1. 累计回报随回合变长而塌陷（每步隐式成本 × 步数），于是升级变成纯计时器，
   与策略质量无关；
2. 不同长度回合的累计回报不可比，跨阶段比较本身就是错的。

本模块是纯逻辑：输入只能是任务指标（覆盖率、覆盖 AUC、目标发现比例、
碰撞率、重复率），输出当前阶段与是否升级。验证评估（固定地图/seed、
确定性动作）由调用方 ``piloc/train.py`` 执行后把指标喂进来。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

# 门控允许读取的指标名。任何其他键（包括 reward/return）都会被拒绝，
# 这样“门控不读累计回报”是一个可断言的接口约束，而不是约定。
ALLOWED_METRICS = (
    "coverage",
    "coverage_auc",
    "discovery_rate",
    "collision_rate",
    "repeat_rate",
)


@dataclass
class CurriculumStage:
    """一个课程阶段：本阶段允许的回合步数上限与升级阈值。"""

    episode_limit: int
    thresholds: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.episode_limit < 1:
            raise ValueError(f"episode_limit must be >= 1, got {self.episode_limit}")
        unknown = set(self.thresholds) - set(ALLOWED_METRICS)
        if unknown:
            raise ValueError(f"thresholds on unknown metrics: {sorted(unknown)}")


def default_stages(
    limits: Sequence[int],
    *,
    coverage: float = 0.5,
    discovery_rate: float = 4 / 6,
    collision_rate: float = 0.5,
    repeat_rate: float = 0.5,
) -> list[CurriculumStage]:
    """coverage/discovery 阈值按各阶段预算占最终预算的比例缩放。

    上一版把最终阶段的门槛（coverage≥0.5、discovery≥4/6）原样套在每一级上。
    stage 0 只有 10 步，两台机器人在 60×60 地图上几何上不可能覆盖一半格子：
    v2 首轮 5000 回合的验证 coverage 停在 0.06、discovery 停在 1/6，课程永远
    卡在 stage 0。现在阶段 k 的门槛是 ``门槛_最终 × limit_k / max(limits)``——
    E=250 的 0.5 在 E=10 是 0.02，在 E=100 是 0.2。

    collision/repeat 是每步比率，与回合长度无关，不缩放。collision 门槛
    （0.5）特意设得宽松：早期随机策略撞墙/出界是正常现象（实测平均碰撞率
    0.31），避碰是奖励层（V2 的 collision 权重）该管的事，课程只管 E 阶梯；
    coverage/discovery 门槛才是任务能力的正确信号。最后一阶段不再升级，
    其阈值只是阶梯终点的能力描述，不参与任何门控判断。
    """
    if not limits:
        raise ValueError("limits must be non-empty; pass None to disable the curriculum")
    final_limit = max(int(limit) for limit in limits)
    return [
        CurriculumStage(
            int(limit),
            {
                "coverage": coverage * (int(limit) / final_limit),
                "discovery_rate": discovery_rate * (int(limit) / final_limit),
                "collision_rate": collision_rate,
                "repeat_rate": repeat_rate,
            },
        )
        for limit in limits
    ]


@dataclass
class CurriculumGate:
    """每隔 ``eval_every`` 个训练回合吃一次验证指标，连续 ``required_passes``
    次全部阈值达标才升一级。

    - 未达标：连续通过计数清零，阶段保持不变；等待再久也不会自动升级。
    - 升级后：连续通过计数重置，重新积累 ``required_passes`` 次。
    - 已在最后阶段：永远不再升级。
    - ``stages=None``：课程关闭，``episode_limit()`` 恒为 ``fixed_limit``。
    """

    stages: Sequence[CurriculumStage] | None
    eval_every: int = 50
    required_passes: int = 3
    fixed_limit: int = 250
    _stage_index: int = field(default=0, init=False)
    _passes_in_a_row: int = field(default=0, init=False)
    _episodes_since_eval: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.eval_every < 1:
            raise ValueError(f"eval_every must be >= 1, got {self.eval_every}")
        if self.required_passes < 1:
            raise ValueError(f"required_passes must be >= 1, got {self.required_passes}")
        if self.stages is not None and len(self.stages) == 0:
            raise ValueError("stages must be non-empty; pass None to disable the curriculum")

    @property
    def enabled(self) -> bool:
        return self.stages is not None

    def episode_limit(self) -> int:
        if not self.enabled:
            return int(self.fixed_limit)
        return int(self.stages[self._stage_index].episode_limit)

    @property
    def stage_index(self) -> int:
        return self._stage_index if self.enabled else 0

    def should_evaluate(self) -> bool:
        """距上次评估已完成的回合数是否已达到评估边界（在回合结束、
        ``register_episode`` 之后询问）。"""
        return self.enabled and self._episodes_since_eval >= self.eval_every

    def metrics_pass(self, metrics: Mapping[str, float]) -> bool:
        if not self.enabled or self._stage_index >= len(self.stages) - 1:
            return False
        unknown = set(metrics) - set(ALLOWED_METRICS)
        if unknown:
            raise ValueError(
                f"CurriculumGate accepts only task metrics, got {sorted(unknown)}"
            )
        thresholds = self.stages[self._stage_index].thresholds
        missing = set(thresholds) - set(metrics)
        if missing:
            raise ValueError(f"evaluation is missing metrics: {sorted(missing)}")
        for name, bound in thresholds.items():
            value = float(metrics[name])
            if name in ("collision_rate", "repeat_rate"):
                if value > bound:
                    return False
            else:
                if value < bound:
                    return False
        return True

    def register_episode(self) -> None:
        """一个训练回合结束；返回后调用方检查 ``due_for_eval``。"""
        if self.enabled:
            self._episodes_since_eval += 1

    def submit_validation(self, metrics: Mapping[str, float]) -> bool:
        """提交一次验证指标，返回本次是否触发了升级。"""
        if not self.enabled:
            return False
        self._episodes_since_eval = 0
        if self._stage_index >= len(self.stages) - 1:
            return False
        if self.metrics_pass(metrics):
            self._passes_in_a_row += 1
            if self._passes_in_a_row >= self.required_passes:
                self._stage_index += 1
                self._passes_in_a_row = 0  # 阶段切换后重置门控窗口
                return True
        else:
            self._passes_in_a_row = 0
        return False
