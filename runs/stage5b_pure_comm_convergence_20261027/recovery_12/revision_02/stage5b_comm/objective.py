"""One objective for revision_02: ``A = A_task - lambda * A_cost``.

The only coefficient is lambda, and it is not fixed. ``unlimited_budget.update_lambda`` moves
it at a PPO update boundary from the batch's success rate against the cell's
``normal_immediate`` reference. Everything this module does is the per-step combination, kept
pure so the arithmetic can be checked without a world.

The cost advantage arriving here is already normalised twice over:

* ``trainer.build_samples`` divides each episode's charged bytes by ``N * G`` before forming
  the advantage, so a byte means the same thing in every cell; and
* the critic's own scale is that same quantity, so ``A_cost`` is in "scaled bytes" units.

That is why one lambda can span the whole grid. Multiplying an unscaled byte count by a single
lambda would make a 48-goal/12-subscriber world look twelve times more expensive per byte than
a 12-goal/3-subscriber one purely because of its size.

``cost_pacing`` is not used: it tapered the cost to zero below a budget watermark, and there is
no watermark without a budget. Keeping the switch here means an old caller cannot silently get
the taper back.
"""

from __future__ import annotations

import math

from . import constants as C

#: The registered advantage is exactly this shape; nothing else multiplies either side.
TASK_COEFFICIENT = 1.0


def coefficients(lam, arm=None):
    """``(task_multiplier, cost_multiplier)``; the cost side IS lambda.

    ``arm`` is accepted and ignored so the trainer's existing call shape still works; revision
    02 has one arm, not the old T1/T2 pair, because the cost pressure is now the dual variable
    rather than a hand-chosen constant.
    """
    lam = float(lam)
    if not math.isfinite(lam):
        raise ValueError("lambda must be finite, got %r" % (lam,))
    if lam < 0.0:
        raise ValueError("lambda must be non-negative, got %r" % (lam,))
    return (TASK_COEFFICIENT, lam)


def combine(advantage_task, advantage_cost, lam, arm=None, budget_fraction=None):
    """``A = 1.0 * A_task - lambda * A_cost`` with every term validated finite."""
    task_multiplier, cost_multiplier = coefficients(lam, arm)
    for name, value in (("advantage_task", advantage_task), ("advantage_cost", advantage_cost)):
        if not math.isfinite(float(value)):
            raise ValueError("%s must be finite, got %r" % (name, value))
    if getattr(C, "COST_PACING", "none") != "none":
        raise RuntimeError("revision_02 runs with no budget, so cost pacing has no watermark "
                           "to taper against; COST_PACING must be 'none'")
    return float(task_multiplier) * float(advantage_task) - cost_multiplier * float(advantage_cost)


def combine_many(advantages_task, advantages_cost, lam, arm=None):
    """Vector form; the two sequences must be aligned."""
    if len(advantages_task) != len(advantages_cost):
        raise ValueError("task and cost advantages must be aligned")
    return [combine(task, cost, lam, arm) for task, cost in zip(advantages_task, advantages_cost)]


def describe(lam):
    """What a report should quote instead of re-deriving the objective."""
    task_multiplier, cost_multiplier = coefficients(lam)
    return {
        "arm": "dual",
        "lambda": float(lam),
        "lambda_initial": float(C.LAMBDA_INITIAL),
        "lambda_bounds": [float(C.LAMBDA_MIN), float(C.LAMBDA_MAX)],
        "lambda_step": float(C.LAMBDA_STEP),
        "q_target_fraction": float(C.Q_TARGET_FRACTION),
        "dual_update": str(C.DUAL_UPDATE),
        "task_multiplier": task_multiplier,
        "cost_multiplier": cost_multiplier,
        "advantage": "1.0 * A_task - %.4f * A_cost" % cost_multiplier,
        "cost_scale": "bytes / (N * G)",
        "budget_bytes": int(C.BUDGET_BYTES),
        "budget_is_ceiling": True,
        "advantage_normalisation": str(C.NORMALIZE_ADVANTAGE),
        "credit_mode": str(C.CREDIT_MODE),
        "credit_rule": str(C.CREDIT_RULE),
        "advantage_mode": str(C.ADVANTAGE_MODE),
        "causal_gamma": float(C.CAUSAL_GAMMA),
        "cost_pacing": str(C.COST_PACING),
        "critic_loss_weights": {"task": C.CRITIC_LOSS_WEIGHT, "cost": C.CRITIC_LOSS_WEIGHT},
    }
