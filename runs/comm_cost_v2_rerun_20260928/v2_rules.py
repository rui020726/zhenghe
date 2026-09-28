"""Stage C 规则对照：新规则 late_discovery_gate + 六条规则臂的工厂。

`late_discovery_gate`（ROUND22 spec §6）是本轮提出的强对照，**不是学习成果**：
只依据发送端可见的首次发现步号，对晚发现目标首报一次，早发现目标不发；
不把早发现延迟到阈值，不读终局或接收端知识。定义与单位在启动前锁定：

- 谓词只读 `SendItem`：`item.first_emission`（该目标从未发过）且
  `item.ready_step > LATE_CUT_STEPS`；
- `ready_step` 是 relay tick 计的首个未发送变更簿记步；Stage B 归档件核验
  （18/18 首报 `step == ready_step`）表明它等于 immediate 下首报步号，
  与第十九轮开发分析 `step_reported_at`（cut=147、report_when_above）同口径；
- dt=1.0 ⇒ 物理时间与 tick 同刻度，比较就在整数 tick 上做，运算符为**严格大于**；
- 共享 relay 的报价/预算/容量/合并/终态处理：预算拒付后下步重试（first_emission 仍真）。

内建规则经 `build_comm_policy(name, params)` 显式构造：未知参数名直接抛错，不静默默认。
"""

from __future__ import annotations

from typing import Any

from integrated_sar.comm_policy import (
    CommSendPolicy,
    SendObservation,
    build_comm_policy,
    parse_parameters,
)

#:  fixed threshold from the round-19 development analysis (cut=147, report_when_above).
#:  Locked for this round; not re-selected here (spec §6).
LATE_CUT_STEPS = 147.0


class LateDiscoveryGatePolicy(CommSendPolicy):
    """First-report a target only when the sender saw it late (ready_step > 147)."""

    name = "late_discovery_gate"
    orders_selection = False

    def __init__(self, *, cut_steps: float = LATE_CUT_STEPS) -> None:
        cut = float(cut_steps)
        if cut < 0:
            raise ValueError(f"cut_steps must be >= 0, got {cut_steps!r}")
        self.cut_steps = cut

    def parameters(self) -> dict[str, Any]:
        return {"cut_steps": self.cut_steps}

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        return tuple(
            item.target_id
            for item in observation.items
            if item.first_emission and float(item.ready_step) > self.cut_steps
        )


def build_rule(arm: str, scenario_seed: int) -> CommSendPolicy:
    """One rule arm, from its registered parameters; unknown fields are hard errors.

    The built policy's own ``parameters()`` is compared with the registered fields, with the
    single registered substitution (random_send's per-scenario seed) applied on both sides, so
    a rule whose body disagrees with its label fails here rather than in the results.
    """
    import v2_common as common

    if arm not in common.RULE_ARMS:
        raise ValueError(f"unknown rule arm {arm!r}; registered: {sorted(common.RULE_ARMS)}")
    spec = common.RULE_ARMS[arm]
    name = str(spec["policy"])
    if name == "late_discovery_gate":
        policy = LateDiscoveryGatePolicy(cut_steps=float(spec["params"]["cut_steps"]))
        expected = {"cut_steps": float(spec["params"]["cut_steps"])}
    else:
        registered = dict(spec["params"])
        params = dict(registered)
        if params.pop("seed_from_scenario", False):
            params["seed"] = int(scenario_seed)
        policy = build_comm_policy(name, parse_parameters(
            ",".join(f"{key}={value}" for key, value in sorted(params.items()))))
        expected = params
    if dict(policy.parameters()) != dict(expected):
        raise RuntimeError(f"rule {arm}: built {policy.parameters()} != registered {expected}")
    return policy


def rule_identity(arm: str) -> dict[str, Any]:
    import v2_common as common

    spec = common.RULE_ARMS[arm]
    return {"arm": arm, "policy": spec["policy"], "params": dict(spec["params"])}
