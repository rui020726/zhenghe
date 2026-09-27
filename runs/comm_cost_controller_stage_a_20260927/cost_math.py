"""Stage A 纯计算模块：成本目标、任务约束、对偶更新、双 critic 与成本账本。

本模块只做数学与记账，**不导入也不启动任何仿真**：没有环境、没有 episode、没有 rollout。
它对应第二十轮指令 §3 的四件事与 §4 的合成测试清单，以及第十九轮设计 spec §4 的
"每条 rollout 字段、优势/return 定义、masked 项处理、成本按实际尝试结算的时点、
HOLD 对后续费用的影响、episode 截断/终止区别、梯度来源"。

符号（指令 §3.1，测试基准）：

    c = C_measured / B0                    已计量尝试字节 ÷ **预锁定**常数 B0
    g = E[Y_ref] - E[Y_policy] - delta     >0 表示约束被违背
    L_max = -E[c] - lambda * g,  lambda >= 0
    lambda_next = project_nonnegative(lambda + eta * g_hat)
    A = lambda * A_task - A_cost           组合优势（推导见 actor_advantage）

三条符号纪律（都有对应测试）：
  - 任务效果相同而字节更多 ⇒ 优化方向必须更差；
  - 任务损失超过 delta ⇒ lambda 必须上升；
  - reference 是**外部固定比较对象**，不参与 actor 求导，其常数与 delta 只影响
    约束估计/乘子更新，不通过伪梯度改变 actor。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch

# --------------------------------------------------------------------------- 版本 -------------

#: 训练器身份。加 cost critic 会改变训练状态、state_dict/optimizer 与 trainer 身份
#: （指令 §3.2：观测不变不等于 checkpoint 架构不变），所以这三项必须显式登记。
TRAINER_TAG = "constrained-cost-v1"
LOSS_VERSION = "constrained-lagrangian/1"
STATE_SCHEMA = "constrained-cost-state/1"
NORM_SCHEMA = "cost-normalization/1"

#: actor 观测 schema 名与字段数：不变（复用现有合法 schema，不偷读未来/真值）。
ACTOR_FEATURE_SCHEMA = "comm-send-state/1"
ACTOR_FEATURE_DIM = 19

# ------------------------------------------------------------------------ 归一化 -------------


@dataclass(frozen=True)
class CostNormalization:
    """成本的**固定**归一化：字节除以预锁定常数 B0。

    B0 是运行身份的一部分，不是每条策略自己的成本（指令 §3.1：
    "B0 是预先锁定的归一化常数，不用每条策略自己的成本作分母"）。
    """

    b0_bytes: float
    schema: str = NORM_SCHEMA

    def __post_init__(self) -> None:
        if not (isinstance(self.b0_bytes, (int, float)) and math.isfinite(self.b0_bytes)
                and self.b0_bytes > 0):
            raise ValueError(f"B0 must be a finite positive byte count, got {self.b0_bytes!r}")

    def normalise(self, billed_bytes: float) -> float:
        """字节 ⇒ 归一化成本 c。"""
        return float(billed_bytes) / float(self.b0_bytes)

    def to_bytes(self, c: float) -> float:
        """归一化成本 c ⇒ 字节。B0 改变只改 c 的数值，不改字节。"""
        return float(c) * float(self.b0_bytes)

    def as_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "b0_bytes": float(self.b0_bytes)}


# ---------------------------------------------------------------------- 成本账本分类 -------------

#: 一次决策的**互斥**结局。计数必须闭合到决策总数，计费必须闭合到账本计数器。
DECISION_BUCKETS = (
    "hold_active",            # 策略主动 HOLD
    "send_executed",          # SEND 被执行且送达
    "send_dropped",           # SEND 被执行但丢包（**照付**）
    "send_refused_budget",    # 被硬额度拒付，未发射 ⇒ 计费 0，且**不算省**
    "send_refused_capacity",  # 被每拍容量拒付，同上
    "send_suppressed_terminal",  # 终态抑制，未发射 ⇒ 计费 0
)
#: 只有真正发射出去的尝试计费。拒付/抑制/主动 HOLD 一律计 0，且不得记成"省下的成本"。
BILLED_BUCKETS = ("send_executed", "send_dropped")
ACK_BUCKETS = ("ack_off_structural_zero", "ack_measured", "ack_expected_but_missing")


@dataclass(frozen=True)
class DecisionOutcome:
    """一次发送决策的记账结果（合成数据即用本结构，真实 rollout 也应当填同一结构）。"""

    bucket: str
    billed_bytes: float = 0.0
    payload_bytes: float | None = None
    subscribers: int | None = None

    def __post_init__(self) -> None:
        if self.bucket not in DECISION_BUCKETS:
            raise ValueError(f"unknown decision bucket {self.bucket!r}")
        if self.bucket in BILLED_BUCKETS and self.billed_bytes <= 0:
            raise ValueError(f"{self.bucket} must carry a positive billed byte count")


def billed_bytes(bucket: str, payload_bytes: float, subscribers: int) -> float:
    """一次尝试的计费：载荷 × 被寻址订阅者；丢包照付；拒付/抑制/HOLD 计 0。

    定价方式与执行器同源（`integrated_sar/comm.py:894-898`），这里只是纯函数版本，
    真实接线时必须由 relay 自己报价，本模块不得成为第二套价格定义。
    """
    if bucket in BILLED_BUCKETS:
        return float(payload_bytes) * float(subscribers)
    return 0.0


def classify_closure(decisions: Sequence[DecisionOutcome]) -> dict[str, Any]:
    """分类闭合：每类计数 + 计费合计 + 与决策总数的闭合断言。"""
    counts = {bucket: 0 for bucket in DECISION_BUCKETS}
    billed = 0.0
    for decision in decisions:
        counts[decision.bucket] += 1
        billed += float(decision.billed_bytes)
    return {"counts": counts, "decisions": len(decisions),
            "counts_sum": sum(counts.values()),
            "billed_bytes_total": billed,
            "counts_close": sum(counts.values()) == len(decisions),
            "bill_matches_decisions": all(
                abs(float(d.billed_bytes)
                    - (float(d.payload_bytes or 0) * int(d.subscribers or 0)
                       if d.bucket in BILLED_BUCKETS else 0.0)) < 1e-9 for d in decisions)}


def ack_status(*, ack_enabled: bool, ack_bytes: float | None) -> dict[str, Any]:
    """"ACK 未启用"是**结构性零**，"ACK 应有但日志缺失"是另一件事，二者不得混记。"""
    if not ack_enabled:
        return {"bucket": "ack_off_structural_zero", "ack_attempt_bytes": 0.0,
                "missing": False}
    if ack_bytes is None:
        return {"bucket": "ack_expected_but_missing", "ack_attempt_bytes": None,
                "missing": True}
    return {"bucket": "ack_measured", "ack_attempt_bytes": float(ack_bytes), "missing": False}


def measured_cost(uplink_bytes: float, ack_bytes: float | None, ack_enabled: bool) -> dict[str, Any]:
    """`C_measured = C_uplink + C_ack`，且只有在账本确实覆盖两项时才使用该名称。"""
    status = ack_status(ack_enabled=ack_enabled, ack_bytes=ack_bytes)
    total = float(uplink_bytes) + (status["ack_attempt_bytes"] or 0.0)
    return {"c_uplink_bytes": float(uplink_bytes), "c_ack_bytes": status["ack_attempt_bytes"],
            "c_measured_bytes": total, "ack": status,
            "fully_metered": not status["missing"]}


# ------------------------------------------------------------------ 目标、约束、对偶 -------------


def constraint_statistic(y_reference: Sequence[float], y_policy: Sequence[float],
                         delta: float) -> float:
    """`g = E[Y_ref] - E[Y_policy] - delta`。

    y 单位是**按时任务/episode**；delta 同单位。>0 = 政策比允许的差得多。
    """
    if len(y_reference) != len(y_policy):
        raise ValueError("reference and policy must be paired seed by seed")
    mean_ref = sum(float(v) for v in y_reference) / len(y_reference)
    mean_pol = sum(float(v) for v in y_policy) / len(y_policy)
    return mean_ref - mean_pol - float(delta)


def dual_objective(c_mean: float, g: float, lam: float) -> float:
    """`L_max = -E[c] - lambda * g`（越大越好）。lambda < 0 直接拒绝。"""
    if lam < 0:
        raise ValueError(f"lambda must be non-negative, got {lam}")
    return -float(c_mean) - float(lam) * float(g)


def project_nonnegative(lam: float) -> float:
    """投影到 lambda >= 0。"""
    return max(0.0, float(lam))


def dual_step(lam: float, g_hat: float, eta: float) -> float:
    """`lambda_next = project_nonnegative(lambda + eta * g_hat)`。

    g_hat 是**测得**的约束违背（配对点估计），不是参考臂失败数、
    不是伪梯度、也不是按消息池化造出来的统计量。
    """
    return project_nonnegative(float(lam) + float(eta) * float(g_hat))


# ------------------------------------------------------------------------ 优势 -------------


def actor_advantage(advantage_task: torch.Tensor, advantage_cost: torch.Tensor,
                    lam: float) -> torch.Tensor:
    """组合优势 `A = lambda * A_task - A_cost`（最小化成本、任务进约束的拉格朗日形式）。

    推导（不做机械替换）：拉格朗日量 `L = E[c] + lambda * g`，
    `g = E[Y_ref] - E[Y_policy] - delta`。对策略参数取**上升**方向即
    `-dL/dtheta = -dE[c]/dtheta + lambda * dE[Y_policy]/dtheta`（`Y_ref` 与 `delta`
    是常数，不进 actor 求导）。逐决策奖励因此是 `r = -c + lambda * y`；优势算子对
    奖励线性、双 critic 各估一项，组合优势即 `lambda * A_task - A_cost`。

    单位：`A_task` 为按时任务数，`A_cost` 为归一化成本；`lambda` 单位为
    任务数 / 归一化成本。**不做逐 batch 分量标准化**：分别标准化两项会改变其有效
    单位、等效改写 lambda，凡引入分量标准化必须作为独立变更重新登记
    （第十九轮 spec §2：训练若归一化，登记固定归一化常数）。
    """
    if lam < 0:
        raise ValueError(f"lambda must be non-negative, got {lam}")
    return float(lam) * advantage_task - advantage_cost


def gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float, gae_lambda: float,
        terminal: torch.Tensor, bootstrap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE，明确区分**真正终止**与**时间截断**。

    - `terminal[i]` 为真 ⇒ 该步之后 episode 真的结束，续接项为 0；
    - `terminal[i]` 为假 ⇒ 只是 horizon 截断，用 `bootstrap[i]`（critic 的估值）继续。

    递推是标准的 advantage 传递 `A_t = delta_t + gamma*gae_lambda*(1-done)*A_{t+1}`，
    因此 `gamma = 1` 且价值为 0 时 `returns[0]` 等于后续无折扣奖励和（有测试锁定）。

    注意：`gamma < 1` 会让 return 变成**折扣**量，与目标里的"episode 总成本/完成数"
    （无折扣）不是同一个数。是否折扣由调用方显式登记，本函数不悄悄替你决定。
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not (0.0 <= gae_lambda <= 1.0):
        raise ValueError(f"gae lambda must be in [0, 1], got {gae_lambda}")
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros_like(rewards[0])
    for step in range(steps - 1, -1, -1):
        # done 用 rewards 的 dtype，避免 float32 中间量把 gamma 折扣的精度悄悄抹掉
        done = 1.0 - terminal[step].to(rewards.dtype)
        delta = rewards[step] + gamma * bootstrap[step].to(rewards.dtype) * done - values[step]
        advantages[step] = delta + gamma * gae_lambda * done * next_advantage
        next_advantage = advantages[step]
    returns = advantages + values
    return advantages, returns


def masked_log_likelihood(logits: torch.Tensor, actions: torch.Tensor,
                          allowed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """被 mask 的项对 actor likelihood **与** entropy 的权重都为零，critic 样本保留。

    整行都被 mask（全 mask 步）时**不产生 actor 样本**：返回 0，而不是 NaN
    （`log_softmax` 在整行 `-inf` 上会给 NaN，所以必须先判空再算）。
    """
    total_ll = torch.zeros(())
    total_entropy = torch.zeros(())
    for decision in range(logits.shape[0]):
        row_mask = allowed[decision]
        if not bool(row_mask.any()):
            continue  # 全 mask：无 Actor 样本，也不是伪样本
        masked_logits = logits[decision].masked_fill(~row_mask, float("-inf"))
        log_prob = torch.log_softmax(masked_logits, dim=-1)
        total_ll = total_ll + log_prob[int(actions[decision])]
        probabilities = log_prob.exp()
        entropy = -(probabilities * log_prob.masked_fill(torch.isinf(log_prob), 0.0)).sum()
        total_entropy = total_entropy + entropy
    return total_ll, total_entropy


# --------------------------------------------------------------------- 训练状态 -------------


def new_cost_state(*, b0_bytes: float, delta: float, task_critic: Any, cost_critic: Any,
                   eta_lambda: float, gamma: float) -> dict[str, Any]:
    """新训练状态：双 critic + lambda + 归一化常数 + 版本号。"""
    return {"schema": STATE_SCHEMA, "trainer_tag": TRAINER_TAG, "loss_version": LOSS_VERSION,
            "normalization": CostNormalization(b0_bytes).as_dict(),
            "delta": float(delta), "lambda": 0.0, "lambda_history": [],
            "eta_lambda": float(eta_lambda), "gamma": float(gamma),
            "actor_features": {"schema": ACTOR_FEATURE_SCHEMA, "dim": ACTOR_FEATURE_DIM},
            "task_critic": task_critic, "cost_critic": cost_critic}


def load_cost_state(payload: dict[str, Any]) -> dict[str, Any]:
    """旧版本 **fail closed**：版本不匹配直接抛错，绝不静默 resume。"""
    schema = payload.get("schema")
    if schema != STATE_SCHEMA:
        raise ValueError(
            f"refusing to resume: state schema {schema!r} is not {STATE_SCHEMA!r} "
            "(old checkpoints must not be silently resumed onto the new objective)")
    for key in ("trainer_tag", "loss_version", "normalization", "delta", "lambda",
                "eta_lambda", "gamma", "task_critic", "cost_critic"):
        if key not in payload:
            raise ValueError(f"refusing to resume: state is missing {key!r}")
    if payload["loss_version"] != LOSS_VERSION:
        raise ValueError("refusing to resume: loss version mismatch")
    return dict(payload)


# ------------------------------------------------------------------- actor 特征边界 -------------

#: 允许出现在 actor 特征里的字段（与现有 19 维 schema 同名；不含任何结局派生量）。
ALLOWED_ACTOR_FEATURE_FIELDS = frozenset({
    "age_steps", "since_last_emission_time", "sent_count", "pending_event", "changed_view",
    "first_emission", "stage_changed", "requirements_changed", "members_changed",
    "progress_delta_abs", "revision_delta", "remaining_time_over_horizon", "urgency_now",
    "has_future_requirements", "queue_backlog_targets", "budget_remaining_fraction",
    "capacity_remaining_fraction", "subscribers", "step_over_horizon",
})
#: 明令禁止：结局派生量、真值、未来信息、参考臂结果。
FORBIDDEN_ACTOR_FEATURE_FIELDS = frozenset({
    "y_episode", "y_reference", "on_time", "expired", "deadline_pass", "future_success",
    "true_goal_state", "reference_failures", "g_hat", "lambda",
})


def build_actor_features(**fields: float) -> torch.Tensor:
    """只接受 19 维 schema 内的字段；出现任何结局/真值/未来字段立即报错。

    这是**结构性**防线：不靠自觉，字段名不在白名单就抛错。
    """
    unknown = set(fields) - ALLOWED_ACTOR_FEATURE_FIELDS
    if unknown:
        raise ValueError(f"actor features may not contain {sorted(unknown)}; "
                         f"allowed: {sorted(ALLOWED_ACTOR_FEATURE_FIELDS)}")
    if len(fields) != ACTOR_FEATURE_DIM:
        raise ValueError(f"actor features must have {ACTOR_FEATURE_DIM} fields, "
                         f"got {len(fields)}")
    return torch.tensor([float(fields[name]) for name in sorted(fields)], dtype=torch.float64)


def report_metric_row(bucket: str, billed_bytes: float) -> dict[str, Any]:
    """报告用的最小行：主动 HOLD / 可执行 SEND / 拒付 / 实际计费**分开列**（指令 §3.4）。"""
    return {"bucket": bucket, "billed_bytes": float(billed_bytes),
            "counts_as_saving": False}
