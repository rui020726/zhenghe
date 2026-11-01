"""The two registered PPO objectives, as one pure function.

T1 and T2 differ only in the cost coefficient multiplying a cost advantage that stays on the
frozen bytes/10000 scale. Both fix lambda at 1.0 and disable the dual update. Nothing here
normalises advantages.
"""

from __future__ import annotations

import math

from . import constants as C

TASK_COEFFICIENT = C.TASK_MULTIPLIER
COST_COEFFICIENT = dict(C.COST_MULTIPLIER)
#: Test-only neutral pair; it must reproduce the audited lambda=1, c=1 combination exactly.
NEUTRAL = (1.0, 1.0)


def coefficients(arm):
    """(task multiplier, cost multiplier) for one of the two registered arms."""
    if arm == "NEUTRAL":
        return NEUTRAL
    if arm not in COST_COEFFICIENT:
        raise ValueError("unknown objective arm %r; registered arms are %s"
                         % (arm, sorted(COST_COEFFICIENT)))
    return (float(TASK_COEFFICIENT), float(COST_COEFFICIENT[arm]))


def combine(advantage_task, advantage_cost, arm):
    """A = task_multiplier * A_task - cost_multiplier * A_cost, coefficients validated."""
    task_multiplier, cost_multiplier = coefficients(arm)
    for value in (task_multiplier, cost_multiplier):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError("objective coefficients must be finite and non-negative")
    return float(task_multiplier) * advantage_task - float(cost_multiplier) * advantage_cost


def combine_many(advantages_task, advantages_cost, arm):
    """Vector form used by the trainer; the two input sequences must be aligned."""
    if len(advantages_task) != len(advantages_cost):
        raise ValueError("task and cost advantages must be aligned")
    return [combine(task, cost, arm) for task, cost in zip(advantages_task, advantages_cost)]


def describe(arm):
    """Machine-readable description that later reports quote instead of re-deriving."""
    task_multiplier, cost_multiplier = coefficients(arm)
    return {
        "arm": arm,
        "lambda_fixed": float(C.LAMBDA_FIXED),
        "dual_update": "disabled",
        "task_multiplier": task_multiplier,
        "cost_multiplier": cost_multiplier,
        "advantage": "%s * A_task - %s * A_cost" % (task_multiplier, cost_multiplier),
        "cost_scale": "bytes / %d" % int(C.B0_BYTES),
        "advantage_normalisation": "none",
        "critic_loss_weights": {"task": C.CRITIC_LOSS_WEIGHT, "cost": C.CRITIC_LOSS_WEIGHT},
        "cost_critic_aux_gradient_disclosed": True,
    }

