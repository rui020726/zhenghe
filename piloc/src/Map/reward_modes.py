"""PILOC 的奖励模式：legacy 对照、v2 默认训练奖励、potential 覆盖塑形。

设计约束（与 ``timemap.Environment.step`` 共同成立）：

- 本模块只做纯算术：输入是标量事件量（新覆盖增量、发现数、碰撞次数等），
  输出是每个 agent 的奖励分量。所有观测量（并集覆盖、目标快照）由环境提供，
  这里不重复定义第二套覆盖率口径。
- 每个分量必须可单独记录，且分量之和恒等于返回的奖励（回归测试锁定）。
- 奖励幅度不得依赖当前课程阶段 E：时间成本按 T_max 归一，
  单个发现事件的奖励幅度与回合长度无关。

legacy 模式不在这里实现：它保留在 ``timemap.Environment.step`` 内，
以保证旧实验对照逐位不变。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping

# 分量名 -> 记录顺序。total 必须等于这些分量之和。
TERM_ORDER = (
    "new_union_coverage",
    "new_target",
    "finish",
    "time_cost",
    "collision",
    "repeat_or_short_cycle",
    "teammate_or_overlap",
    "pheromone",
    "potential",
)


@dataclass
class RewardWeights:
    """V2 / potential 模式的全部可调权重。"""

    target: float = 1.0
    coverage: float = 2.0
    finish: float = 3.0
    time_cost: float = 0.5
    collision: float = 0.2
    repeat: float = 0.0
    pheromone: float = 0.0
    potential_beta: float = 1.0
    t_max: int = 250
    gamma: float = 0.99
    # 信息素分母保护与裁剪：实验项，默认权重为 0 时完全不参与。
    pheromone_epsilon: float = 1e-6
    pheromone_clip: float = 1.0

    @staticmethod
    def from_config(
        weights: Mapping[str, float],
        *,
        t_max: int,
        gamma: float,
        potential_beta: float,
    ) -> "RewardWeights":
        return RewardWeights(
            target=float(weights.get("target", 1.0)),
            coverage=float(weights.get("coverage", 2.0)),
            finish=float(weights.get("finish", 3.0)),
            time_cost=float(weights.get("time_cost", 0.5)),
            collision=float(weights.get("collision", 0.2)),
            repeat=float(weights.get("repeat", 0.0)),
            pheromone=float(weights.get("pheromone", 0.0)),
            potential_beta=float(potential_beta),
            t_max=int(t_max),
            gamma=float(gamma),
        )


@dataclass
class StepEvents:
    """一步之内、与模式无关的事件量（由环境计算，本模块只负责加权）。

    - ``coverage_delta``: 团队并集覆盖率 C_{t+1} - C_t（共享，不重复计数）。
    - ``new_targets``: 本步首次被任意机器人观测到的唯一目标数（共享）。
    - ``all_found``: 本步之后目标是否全部完成。
    - ``collisions``: 每 agent 的碰撞次数（0/1，单独计）。
    - ``repeat_flags``: 每 agent 是否触发短周期重复（默认权重 0）。
    - ``pheromone``: 每 agent 的原始信息素差值（相对+绝对混合，实验项）。
    - ``potential_current`` / ``potential_next``: 势函数模式的 Phi_t 与 Phi_{t+1}。
    """

    coverage_delta: float = 0.0
    new_targets: int = 0
    all_found: bool = False
    collisions: Dict[str, int] = field(default_factory=dict)
    repeat_flags: Dict[str, bool] = field(default_factory=dict)
    pheromone: Dict[str, float] = field(default_factory=dict)
    potential_current: float = 0.0
    potential_next: float = 0.0


def _finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"reward input {name} must be finite, got {value!r}")
    return float(value)


def pheromone_term(raw_delta: float, weight: float, *, clip: float) -> float:
    """信息素实验项：权重为 0 时恒为 0；否则做有限值检查并裁剪到 ±clip。

    ``raw_delta`` 是环境算出的原始混合值（相对+绝对），其分母可能极小，
    环境侧已用 epsilon 保护；这里再裁剪一次，防止单步信息素波动主导团队奖励。
    """
    if weight == 0.0:
        return 0.0
    value = float(raw_delta)
    if not math.isfinite(value):
        return 0.0
    scaled = weight * value
    return max(-clip, min(clip, scaled))


def compute_v2_rewards(
    agent_ids, events: StepEvents, weights: RewardWeights
) -> Dict[str, Dict[str, float]]:
    """V2：团队奖励共享 + 每 agent 碰撞成本。返回 {agent: {term: value}}。

    team_reward = target*N_t + coverage*(C_{t+1}-C_t) + finish*all_found
                  - time_cost/T_max
    agent_reward = team_reward - collision*collision_i + 实验项

    分量 schema 恒定（未激活的项记 0），使日志与断言不依赖某一步恰好发生了什么。
    ``teammate_or_overlap`` 在 V2 中恒为 0：队友邻近不再惩罚（邻近同时是地图
    融合条件），团队覆盖也已在 coverage_delta 里去重，不存在重复计数。
    """
    c_delta = _finite(events.coverage_delta, "coverage_delta")
    n_targets = int(events.new_targets)
    if n_targets < 0:
        raise ValueError("new_targets must be non-negative")

    team_terms: Dict[str, float] = {
        "new_target": weights.target * n_targets,
        "new_union_coverage": weights.coverage * c_delta,
        "finish": weights.finish if events.all_found else 0.0,
        "time_cost": -weights.time_cost / float(weights.t_max),
        "teammate_or_overlap": 0.0,
    }
    result: Dict[str, Dict[str, float]] = {}
    for agent_id in agent_ids:
        terms = dict(team_terms)
        terms["collision"] = -weights.collision * int(events.collisions.get(agent_id, 0))
        terms["repeat_or_short_cycle"] = (
            -weights.repeat if weights.repeat != 0.0 and events.repeat_flags.get(agent_id, False) else 0.0
        )
        if weights.pheromone != 0.0:
            raw = events.pheromone.get(agent_id, 0.0)
            terms["pheromone"] = pheromone_term(
                raw, weights.pheromone, clip=weights.pheromone_clip
            )
        else:
            terms["pheromone"] = 0.0
        result[agent_id] = terms
    return result


def compute_potential_rewards(
    agent_ids, events: StepEvents, weights: RewardWeights
) -> Dict[str, Dict[str, float]]:
    """potential：V2 的覆盖增量项换成势函数塑形项 F。

    F = beta * (gamma * Phi_{t+1} - Phi_t)，Phi = 团队并集覆盖率。

    注意：这使回报对回合长度的依赖减弱但不完全无关（终端与截断的处理
    不同，见 ``terminal_adjustment``）；不要声称它与回合长度完全无关。
    """
    base = compute_v2_rewards(agent_ids, events, weights)
    phi = _finite(events.potential_current, "potential_current")
    phi_next = _finite(events.potential_next, "potential_next")
    shaping = weights.potential_beta * (weights.gamma * phi_next - phi)
    for agent_id in agent_ids:
        terms = base[agent_id]
        # 用塑形项替换原始覆盖增量项，其余分量保持不变。
        terms.pop("new_union_coverage", None)
        terms["potential"] = shaping
    return base


def total_of(terms: Mapping[str, float]) -> float:
    return float(sum(terms.values()))
