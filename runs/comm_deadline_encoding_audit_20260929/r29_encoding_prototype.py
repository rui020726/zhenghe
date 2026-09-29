"""ROUND29 §6：发送端期限列的隔离编码原型（纯计算，不触碰生产路径）。

背景（本轮 §4/§5 已证实）：生产编码 ``integrated_sar.comm_learning.features`` 的第 12 列
（0-based index 11，``remaining_time_over_horizon``）实际计算为
``clip(remaining_time / URGENCY_SCALE_STEPS=50, -1, 1)``。在三个冻结来源的全部 102 个
首报决策点上，原始剩余期限合法可见且确实不同（62.9–177.6 步，101 个不同值），但该列
恒等于 1.0：期限差异被完全压缩。

本原型只做一件事：把该列换成 ``remaining_time / configured_episode_horizon``（默认 1000），
不追加任何截断；其余 18 列、顺序、``urgency_now``、输入维数全部与生产编码一致。

严格限定（对应本轮执行依据 §6）：
- 只改变这一列：实现方式就是"调用生产 features() 后原位替换第 12 列"，其余列逐字节相同；
- horizon 必须为正且有限，非有限输入显式报错，不默默填零；
- 允许有限的负剩余期限与超过 horizon 的期限（新列可 < 0 或 > 1）；
- 不从数据拟合尺度；
- 不替换生产编码器、不修改现有 collector 或训练器、不加载旧权重做新编码推理。

观测语义版本登记：维数相同不代表与旧 checkpoint 兼容。
"""

from __future__ import annotations

import math
from typing import Any

from integrated_sar.comm_learning import (
    ACK_FEATURE_NAMES,
    ACK_FEATURE_LAYOUT,
    EPISODE_HORIZON_STEPS,
    FEATURE_LAYOUT,
    FEATURE_NAMES,
    features,
)
from integrated_sar.comm_policy import SendItem, SendObservation

#: 候选编码的观测语义版本。列名仍叫 remaining_time_over_horizon，但语义改为
#: "剩余期限 / 运行前配置的 episode horizon（1000），不截断"。与 ``comm-send-state/1`` 同维
#: （19 维）、同顺序，仅第 12 列定义不同。
CANDIDATE_FEATURE_LAYOUT = "comm-send-state/1r29-deadline-over-horizon"

#: 与生产布局逐列比对时的差异列（0-based）。
DEADLINE_COLUMN = 11

#: 接口要求登记：同一网络在两种语义下不可静默换用。这是登记要求，不是生产拒绝路径的实现——
#: 本轮不声称已完成真实恢复入口的修改与验收。
SEMANTIC_REGISTRY: dict[str, Any] = {
    "layout_unchanged_columns": [name for index, name in enumerate(FEATURE_NAMES)
                                 if index != DEADLINE_COLUMN],
    "changed_column": {"name": FEATURE_NAMES[DEADLINE_COLUMN], "index": DEADLINE_COLUMN},
    "old_semantics": {
        "layout": FEATURE_LAYOUT,
        "formula": "clip(remaining_time / 50.0, -1.0, 1.0)",
        "constants": {"URGENCY_SCALE_STEPS": 50.0},
    },
    "candidate_semantics": {
        "layout": CANDIDATE_FEATURE_LAYOUT,
        "formula": "remaining_time / configured_episode_horizon",
        "constants": {"EPISODE_HORIZON_STEPS": EPISODE_HORIZON_STEPS},
        "no_truncation": True,
        "negative_and_above_horizon_allowed": True,
    },
    "compatibility_rule": (
        "a checkpoint, optimizer state or replay buffer produced under comm-send-state/1 must be "
        "refused rather than silently reused under this layout: the same input dimension does not "
        "make the same weights. This round registers the requirement only; no production "
        "resume/eval entry point was modified and none is claimed to be verified."
    ),
    "ack_note": (
        "layout comm-send-state/2 (23 dims with ACK features) is orthogonal: this prototype is the "
        f"19-dim /1 shape with one changed column; if an ACK build is ever done it would combine the "
        f"two changes under a new layout name, never silently."
    ),
}


def _validate_horizon(horizon: float) -> float:
    value = float(horizon)
    if not math.isfinite(value):
        raise ValueError(f"episode horizon must be finite, got {horizon!r}")
    if value <= 0.0:
        raise ValueError(f"episode horizon must be positive, got {horizon!r}")
    return value


def encoded_deadline(remaining_time: float, *, horizon: float = EPISODE_HORIZON_STEPS) -> float:
    """候选期限列：``remaining_time / horizon``，不截断。

    非有限 ``remaining_time`` 显式报错（不默默填零）；horizon 非正或非有限同样显式报错。
    允许有限的负值与超过 horizon 的值（返回值可 < 0 或 > 1）。
    """
    scale = _validate_horizon(horizon)
    remaining = float(remaining_time)
    if not math.isfinite(remaining):
        raise ValueError(f"remaining_time must be finite, got {remaining_time!r}")
    return remaining / scale


def candidate_features(
    observation: SendObservation,
    item: SendItem,
    *,
    horizon: float = EPISODE_HORIZON_STEPS,
) -> tuple[float, ...]:
    """与生产 19 维特征逐列相同，仅期限列换成 ``remaining_time / horizon``。

    实现方式是调用生产 :func:`integrated_sar.comm_learning.features` 后原位替换第 12 列，
    因此"其余 18 列及顺序不变"由构造保证，而不是由本文件再抄一份公式。
    """
    _validate_horizon(horizon)
    values = list(features(observation, item))
    values[DEADLINE_COLUMN] = encoded_deadline(item.remaining_time, horizon=horizon)
    return tuple(values)


def layout_declaration(horizon: float = EPISODE_HORIZON_STEPS) -> dict[str, Any]:
    """输入维数、顺序与观测版本声明。"""
    return {
        "feature_layout": CANDIDATE_FEATURE_LAYOUT,
        "dim": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "order_note": "identical to comm-send-state/1; only the deadline column's definition changed",
        "horizon": _validate_horizon(horizon),
        "ack_layout_unchanged": ACK_FEATURE_LAYOUT,
        "ack_feature_names": list(ACK_FEATURE_NAMES),
        "actor_visible_inputs_only": (
            "every column is still a property of the message the sender holds, its own ledger and "
            "airtime; nothing receiver-side, nothing outcome-side was added"
        ),
    }
