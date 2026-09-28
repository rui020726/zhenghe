"""ROUND25 三臂策略构造：immediate 原实现、late_discovery_gate 原阈值 147。

规则实现逐字继承 ROUND24 只读 harness 的 v2_rules.LateDiscoveryGatePolicy（原阈值、原边界、
原运算符），本模块不重写第二份；immediate 是 driver 原生参考策略，不注入。
"""

from __future__ import annotations

from typing import Any

import r25_common as common
from v2_rules import LATE_CUT_STEPS, LateDiscoveryGatePolicy  # original implementation


def build_arm_policy(arm: str, scenario_seed: int):
    """The attached policy of one non-native arm, or None for the native immediate arm."""
    spec = common.ARMS[arm]
    if spec["kind"] == "reference":
        return None
    if spec["kind"] == "rule":
        cut = float(spec["params"]["cut_steps"])
        if cut != LATE_CUT_STEPS:
            raise RuntimeError(f"arm {arm}: cut_steps {cut} != the frozen {LATE_CUT_STEPS}")
        return LateDiscoveryGatePolicy(cut_steps=cut)
    raise ValueError(f"arm {arm!r} has no rule policy; learned arms attach a collector")


def arm_identity(arm: str, policy: Any = None) -> dict[str, Any]:
    spec = common.ARMS[arm]
    identity: dict[str, Any] = {"arm": arm, "kind": spec["kind"], "role": spec["role"]}
    if spec["kind"] == "reference":
        identity["policy"] = str(spec["policy"])
        identity["implementation"] = "driver-native ImmediatePolicy (no injection)"
    elif spec["kind"] == "rule":
        identity["policy"] = str(spec["policy"])
        identity["params"] = dict(spec["params"])
        identity["implementation"] = ("v2_rules.LateDiscoveryGatePolicy, cut_steps=147.0, strict "
                                     "greater-than on ready_step (original boundary)")
        if policy is not None:
            identity["built_parameters"] = dict(policy.parameters())
    else:
        identity["policy"] = "learned"
        identity.update({"training_seed": int(spec["training_seed"]), "mode": str(spec["mode"]),
                         "checkpoint": str(spec["checkpoint"]),
                         "implementation": "BudgetMaskedCollector(greedy=True, budget_mask=True, "
                                          "forced=False, mask=False) on the ep128 milestone; "
                                          "SEND iff p(SEND) >= p(HOLD), ties to SEND"})
    return identity
