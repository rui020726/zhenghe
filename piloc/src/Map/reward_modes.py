"""PILOC 的奖励模式：legacy 对照、v2 默认训练奖励、potential 覆盖塑形。

设计约束（与 ``timemap.Environment.step`` 共同成立）：

- 本模块只做纯算术：输入是标量事件量（整数新格数、发现数、碰撞次数等），
  输出是每个 agent 的奖励分量。所有观测量（并集覆盖、目标快照）由环境提供，
  这里不重复定义第二套覆盖率口径。
- 每个分量必须可单独记录，且分量之和恒等于返回的奖励（回归测试锁定）。
- 奖励幅度不得依赖当前课程阶段 E：覆盖奖励按整数格子计数（不按比例归一），
  单个发现事件的奖励幅度与回合长度无关。

v2 baseline（首轮标定起点，非最终最优参数）::

    R_i = 0.20 * marginal_new_cells_i   # 观察者分摊的整数新格
        + 0.05 * new_union_cells        # 团队并集整数新格（共享）
        + 5.0  * new_targets            # 团队共享
        + 20.0 * all_found              # 团队共享
        - 1.0  * collision_i            # 个体
        - 0.05 * I(union_new_cells == 0)

旧版 v2 把覆盖率增量乘以 10 后直接进奖励，在 60x60 地图上每步只有
0.1~0.2 的绝对信号（均值 0.013），比 legacy 小约两个数量级，且信息素项的
系统性负偏置（实测均值 -0.0128）几乎将其完全抵消。改为整数格子计数后，
单步探索信号在 O(1)~O(10) 量级，与碰撞惩罚（-1）同阶可比。

legacy 模式不在这里实现：它保留在 ``timemap.Environment.step`` 内，
以保证旧实验对照逐位不变。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping

# 分量名 -> 记录顺序。total 必须等于这些分量之和。
TERM_ORDER = (
    "marginal_new_cells",
    "new_union_coverage",
    "new_target",
    "finish",
    "no_new_cell",
    "time_cost",
    "collision",
    "repeat_or_short_cycle",
    "teammate_or_overlap",
    "pheromone",
    "potential",
)


@dataclass
class RewardWeights:
    """V2 / potential 模式的全部可调权重（值为首轮标定起点）。

    - ``marginal_coverage``: 观察者分摊的整数边际新格权重。同一新格被 n 个
      机器人同时看到，每个获得 1/n——既不重复计数又保留个体信用。
    - ``union_coverage``: 团队并集整数新格权重（团队共享）。
    - ``no_new_cell``: 本步团队并集零新增时的惩罚，推动策略离开已探索区。
    """

    target: float = 5.0
    marginal_coverage: float = 0.2
    union_coverage: float = 0.05
    finish: float = 20.0
    collision: float = 1.0
    no_new_cell: float = 0.05
    time_cost: float = 0.0
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
            target=float(weights.get("target", 5.0)),
            marginal_coverage=float(weights.get("marginal_coverage", 0.2)),
            union_coverage=float(weights.get("union_coverage", 0.05)),
            finish=float(weights.get("finish", 20.0)),
            collision=float(weights.get("collision", 1.0)),
            no_new_cell=float(weights.get("no_new_cell", 0.05)),
            time_cost=float(weights.get("time_cost", 0.0)),
            repeat=float(weights.get("repeat", 0.0)),
            pheromone=float(weights.get("pheromone", 0.0)),
            potential_beta=float(potential_beta),
            t_max=int(t_max),
            gamma=float(gamma),
        )


@dataclass
class StepEvents:
    """一步之内、与模式无关的事件量（由环境计算，本模块只负责加权）。

    - ``new_union_cells``: 本步团队并集新覆盖的整数自由格数（共享）。
    - ``marginal_new_cells``: 每 agent 的边际新格数（观察者分摊：同一新格
      被 n 个机器人同时看到，每个获得 1/n）。
    - ``coverage_delta``: 团队并集覆盖率 C_{t+1} - C_t（仅 potential 模式
      的势函数使用；v2 只读整数计数，避免归一化压小信号）。
    - ``new_targets``: 本步首次被任意机器人观测到的唯一目标数（共享）。
    - ``all_found``: 本步之后目标是否全部完成。
    - ``collisions``: 每 agent 的碰撞次数（0/1，单独计）。
    - ``repeat_flags``: 每 agent 是否触发短周期重复（默认权重 0）。
    - ``pheromone``: 每 agent 的原始信息素差值（实验项，默认权重 0）。
    - ``potential_current`` / ``potential_next``: 势函数模式的 Phi_t 与 Phi_{t+1}。
    """

    new_union_cells: int = 0
    marginal_new_cells: Dict[str, float] = field(default_factory=dict)
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

    注意：裁剪在乘权重之后进行，因此 ``weight * |raw| > clip`` 时实际幅度是
    ``clip`` 而不是 ``weight``——这是已知语义，调用方靠 ``pheromone_clip``
    控制最大影响。
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
    """V2 baseline：整数覆盖计数 + 个体碰撞成本。返回 {agent: {term: value}}。

    team_reward = 5*N_targets + 0.05*new_union_cells + 20*all_found
    agent_reward = team_reward + 0.20*marginal_new_cells_i
                   - 1.0*collision_i - 0.05*I(new_union_cells == 0)

    分量 schema 恒定（未激活的项记 0），使日志与断言不依赖某一步恰好发生了什么。
    ``teammate_or_overlap`` 在 V2 中恒为 0：队友邻近不再惩罚（邻近同时是地图
    融合条件），团队覆盖也已在 new_union_cells 里去重，不存在重复计数。
    ``time_cost`` / ``repeat`` / ``pheromone`` 权重默认为 0：首轮标定只验证
    核心覆盖闭环，一次只调一类信号。
    """
    n_union = int(events.new_union_cells)
    if n_union < 0:
        raise ValueError("new_union_cells must be non-negative")
    n_targets = int(events.new_targets)
    if n_targets < 0:
        raise ValueError("new_targets must be non-negative")

    team_terms: Dict[str, float] = {
        "new_target": weights.target * n_targets,
        "new_union_coverage": weights.union_coverage * n_union,
        "finish": weights.finish if events.all_found else 0.0,
        "no_new_cell": -weights.no_new_cell if (weights.no_new_cell != 0.0 and n_union == 0) else 0.0,
        "time_cost": -weights.time_cost / float(weights.t_max) if weights.time_cost != 0.0 else 0.0,
        "teammate_or_overlap": 0.0,
    }
    result: Dict[str, Dict[str, float]] = {}
    for agent_id in agent_ids:
        terms = dict(team_terms)
        marginal = _finite(
            float(events.marginal_new_cells.get(agent_id, 0.0)),
            f"marginal_new_cells[{agent_id}]",
        )
        terms["marginal_new_cells"] = weights.marginal_coverage * marginal
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
    """potential：V2 的覆盖项换成势函数塑形项 F。

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
        # 用塑形项替换覆盖项（并集 + 边际），其余分量保持不变。
        terms.pop("new_union_coverage", None)
        terms.pop("marginal_new_cells", None)
        terms["potential"] = shaping
    return base


def total_of(terms: Mapping[str, float]) -> float:
    return float(sum(terms.values()))
