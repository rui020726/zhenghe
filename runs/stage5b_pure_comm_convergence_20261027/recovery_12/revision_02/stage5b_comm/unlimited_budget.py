"""Unlimited budget, success-constrained byte cost: the objective of revision_02.

Two ideas, kept separate so each can be tested without a world.

**The budget is a ceiling, not a scarcity.** ``BUDGET_BYTES`` is set to a large finite
integer so the relay never refuses a send and the learner never masks one. The old regime
priced a send by "will this fit in 58,600 B", which conflated two different questions --
"is this message worth sending" and "can we still afford to". With the ceiling lifted, the
only thing that discourages a send is the cost advantage, and the comparison across cells
becomes meaningful because a cell no longer differs merely by how many sends fit.

**The cost is scaled, never the topology.** One byte is priced as ``bytes / (N * G)``. ``N``
is the fan-out that actually bills, ``G`` is how many goals the world offers, so the scale
factor is exactly "one byte per subscriber per target". ``R`` is absent on purpose: it changes
how much work can be executed, not what a message costs.

The lambda rule runs only at a PPO update boundary, never inside an episode:

    lambda <- clip(lambda + 2 * (q - 0.95 * q_ref), 0.25, 8)

with ``q`` the batch's on-time fraction and ``q_ref`` the ``normal_immediate`` rate for the
same cell. Above target raises the price of a byte, below target lowers it. That is the whole
dual: the constraint is "keep at least 95% of the strong rule's success rate", and the
objective under it is bytes.

Every function here is pure. The clamp and the direction are what the unit tests pin.
"""

from __future__ import annotations

#: The frozen cost normaliser every earlier revision used. At the reference cell the scaled
#: cost must reproduce ``bytes / B0_BYTES`` exactly, which is what keeps one lambda
#: interpretable against the archived runs.
B0_BYTES = 10000.0
#: The cell that defines "one unit of scaled cost". N=6 and G=24 is the frozen configuration,
#: so any other cell is priced relative to it by the registered N*G ratio.
REFERENCE_CELL = (6, 24)

#: Large, finite, and safe in the float arithmetic the relay does on it. Not ``inf``: the
#: frozen code divides remaining bytes by the budget and an infinity would make every
#: fraction NaN.
UNLIMITED_BUDGET_BYTES = 1_000_000_000

LAMBDA_INITIAL = 2.0
LAMBDA_STEP = 2.0
LAMBDA_MIN = 0.25
LAMBDA_MAX = 8.0
#: The success target relative to the strong rule for the same cell. This is the *registered
#: verdict* threshold and never changes: a cell improves only if the learned arm keeps at least
#: 95% of ``normal_immediate``'s on-time count and spends fewer bytes.
Q_TARGET_FRACTION = 0.95
#: The dual's own water line, 1.5 points above the verdict. A cell that steers exactly on 0.95
#: is one discrete dropped goal away from failing a verdict that is applied literally, and at
#: G=12 one goal is 8.3% of the cell's total. train_v4g therefore aims at 0.965 * q_ref so the
#: policy stops trimming while it still has margin, not at the edge itself.
Q_TARGET_FRACTION_DUAL = 0.965

#: revision_02 has one lambda for the whole grid. train_v4_cell_lambda keeps one per cell
#: instead, because a single value is driven by whichever cells the batch happened to contain
#: and therefore oscillates between the clamps (measured on train_v3: 191 boundary hits, 122 of
#: them at the ceiling). The per-cell step is 1/8 of the shared step so the dual moves slowly
#: enough for the exponential average to be the signal rather than one batch's luck.
LAMBDA_BY_CELL_INITIAL = 3.0
LAMBDA_STEP_PER_CELL = 0.25
Q_EMA_ALPHA = 0.25


def clamp_lambda(value):
    """The registered clip; refuses a non-finite input instead of propagating NaN."""
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("lambda must be finite, got %r" % (value,))
    return min(LAMBDA_MAX, max(LAMBDA_MIN, value))


def update_lambda(current, q_batch, q_ref):
    """One dual step. Call only at an update boundary, never mid-episode.

    ``q_batch`` is this batch's ``sum(on_time) / sum(generated_goals)``. ``q_ref`` is the same
    quantity for ``normal_immediate`` in the same cell, so the comparison is always like for
    like -- a 48-goal world and a 12-goal world are never compared to each other.

    Returns ``(new_lambda, detail)``; ``detail`` is what the trace records so a reader can
    reconstruct the step rather than trust the number.
    """
    if q_ref is None:
        raise ValueError("q_ref is required; a cell without a normal_immediate rate cannot "
                         "set its own constraint")
    q_batch = float(q_batch)
    q_ref = float(q_ref)
    for name, value in (("q_batch", q_batch), ("q_ref", q_ref)):
        if not 0.0 <= value <= 1.0:
            raise ValueError("%s must be in [0, 1], got %r" % (name, value))
    target = Q_TARGET_FRACTION * q_ref
    error = q_batch - target
    proposed = float(current) + LAMBDA_STEP * error
    new = clamp_lambda(proposed)
    return new, {
        "lambda_before": float(current),
        "lambda_after": new,
        "clamped": new != proposed,
        "q_batch": q_batch,
        "q_ref": q_ref,
        "q_target": target,
        "error": error,
        #: A positive error means the batch beat the target, so bytes got more expensive.
        "direction": "raise_price" if error > 0 else ("lower_price" if error < 0 else "hold"),
    }


LAMBDA_VERDICT_DECAY = 0.25
#: Below this the decayed price is set to exactly zero, so a cell that is short of on-time
#: reads as "cost is free" instead of "cost is cheap".
LAMBDA_VERDICT_FLOOR = 0.05
#: The waste water line: at or above it, useless broadcasts are priced again even when the
#: cell's success rate sits between the on-time target and the immediate reference.
WASTE_WATER_LINE = 0.10


def clamp_lambda_zero(value):
    """``[0, LAMBDA_MAX]``: the v4v actuator, where zero is a legal price."""
    return max(0.0, min(LAMBDA_MAX, float(value)))


def update_lambda_verdict(current, q_ema, q_ref, *, waste_ratio,
                          step=LAMBDA_STEP_PER_CELL, q_target_fraction=None,
                          decay=LAMBDA_VERDICT_DECAY, waste_threshold=WASTE_WATER_LINE):
    """Experiment v4v's per-cell dual: protect on-time first, price only useless broadcasts.

    Three regions, in this order of precedence:

    1. ``q_ema < target`` -- the cell is short of on-time. The price **decays geometrically to
       zero** (``decay`` per scan, snapped to 0 below ``LAMBDA_VERDICT_FLOOR``) and no cost term
       is added at all. The old rule only walked down by ``step * error`` and could never reach
       0.25, so a congested cell kept paying a price while already losing tasks.
    2. ``q_ema >= target`` -- the cell is meeting its constraint, so trimming is allowed:
       ``penalty = max(0, q_ema - q_ref)``, plus ``waste_ratio - waste_threshold`` when the
       cell's broadcast waste is at or above the water line. The price rises by
       ``step * penalty`` and is clamped to ``[0, LAMBDA_MAX]``.
    3. otherwise the price holds.

    ``waste_ratio`` is the paper's definition (broadcast-but-not-on-time over distinct
    broadcasts); ``None`` means it could not be measured, which is treated as "no evidence of
    waste" rather than as zero waste.
    """
    if q_target_fraction is None:
        q_target_fraction = Q_TARGET_FRACTION_DUAL
    q_target_fraction = float(q_target_fraction)
    if not 0.0 < q_target_fraction <= 1.0:
        raise ValueError("q_target_fraction must be in (0, 1], got %r" % (q_target_fraction,))
    if q_ref is None:
        raise ValueError("q_ref is required; a cell without a normal_immediate rate cannot "
                         "set its own constraint")
    q_ema, q_ref = float(q_ema), float(q_ref)
    for name, value in (("q_ema", q_ema), ("q_ref", q_ref)):
        if not 0.0 <= value <= 1.0:
            raise ValueError("%s must be in [0, 1], got %r" % (name, value))
    if decay <= 0.0 or decay >= 1.0:
        raise ValueError("decay must be in (0, 1), got %r" % (decay,))
    current = float(current)
    target = q_target_fraction * q_ref
    waste = None if waste_ratio is None else float(waste_ratio)
    error = q_ema - q_ref

    if q_ema < target:
        proposed = current * float(decay)
        new = 0.0 if proposed < LAMBDA_VERDICT_FLOOR else proposed
        return new, {
            "lambda_before": current, "lambda_after": new,
            "q_ema": q_ema, "q_ref": q_ref, "q_target": target, "error": error,
            "waste_ratio": waste, "waste_threshold": float(waste_threshold),
            "region": "protect_on_time", "direction": "decay_to_zero",
            "penalty": 0.0, "step": float(step), "decay": float(decay),
            "clamped": False,
        }

    penalty = max(0.0, error)
    priced_waste = waste is not None and waste >= float(waste_threshold)
    if priced_waste:
        penalty += waste - float(waste_threshold)
    proposed = current + float(step) * penalty
    new = clamp_lambda_zero(proposed)
    return new, {
        "lambda_before": current, "lambda_after": new,
        "q_ema": q_ema, "q_ref": q_ref, "q_target": target, "error": error,
        "waste_ratio": waste, "waste_threshold": float(waste_threshold),
        "region": "price_waste" if priced_waste else "price_margin",
        "direction": "raise_price" if new > current else "hold",
        "penalty": penalty, "step": float(step), "decay": float(decay),
        "clamped": new != proposed,
    }


def scaled_cost(cost_bytes, subscribers, goals, *, reference=REFERENCE_CELL):
    """``bytes / N``, rescaled so the frozen reference cell keeps the old magnitude.

    History, because the divisor changed and the reason matters. train_v3 divided by ``N * G``
    and then multiplied by ``ref_N * ref_G / B0``, which made one broadcast cost

        (page_bytes * N) * (6 * 24) / (N * G * 10000) = page_bytes * 144 / (G * 10000),

    i.e. ``~6.0 / G`` scaled units for a ~418 B page against a task credit of 1.0. So the
    break-even price was ``lambda* = G / 6``: 2.0 at G=12, 4.0 at G=24, 8.0 at G=48. At G=48
    the break-even sits exactly on the registered ceiling, which is why train_v3's G=48 cells
    only ever stopped sending once lambda was pinned at 8.0 and then collapsed past it.

    Dividing by ``G`` was wrong in the first place: a broadcast costs the same whatever the
    number of goals a world generates, and the charged bytes already carry the fan-out (the
    relay bills each subscriber). So the aligned price is "charged bytes per subscriber":

        bytes / N   ->   one page costs page_bytes / B0 at every cell, ~0.25 with a 418 B page.

    That makes ``lambda* = B0 / page_bytes = 3.99`` independent of G, inside the middle of
    [0.25, 8] rather than at its ceiling. ``G`` is still accepted (callers pass the cell) and
    recorded by the caller, but it no longer prices anything; that is the whole fix.
    """
    subscribers = int(subscribers)
    if subscribers < 1:
        raise ValueError("subscribers must be >= 1, got %d" % (subscribers,))
    if int(goals) < 1:
        raise ValueError("goals must be >= 1, got %d" % (int(goals),))
    ref_subscribers = int(reference[0])
    if ref_subscribers < 1:
        raise ValueError("the reference cell must be positive, got %r" % (reference,))
    return float(cost_bytes) / float(subscribers) * (float(ref_subscribers) / B0_BYTES)


def update_lambda_cell(current, q_ema, q_ref, *, step=LAMBDA_STEP_PER_CELL,
                       q_target_fraction=None):
    """One per-cell dual step, on the cell's own smoothed success rate.

    Same direction and clamp as :func:`update_lambda`; only the signal changes. ``q_ema`` is
    this cell's exponential average of ``on_time / generated``, so one unlucky world cannot
    move the price, and no other cell's batch can either. Returns ``(new_lambda, detail)``.

    The water line defaults to ``Q_TARGET_FRACTION_DUAL`` (0.965) so the dual stops trimming
    with a margin above the 0.95 the verdict is judged on. Passing ``q_target_fraction``
    explicitly is what the tests use to pin both numbers separately.
    """
    if q_target_fraction is None:
        q_target_fraction = Q_TARGET_FRACTION_DUAL
    q_target_fraction = float(q_target_fraction)
    if not 0.0 < q_target_fraction <= 1.0:
        raise ValueError("q_target_fraction must be in (0, 1], got %r" % (q_target_fraction,))
    if q_ref is None:
        raise ValueError("q_ref is required; a cell without a normal_immediate rate cannot "
                         "set its own constraint")
    q_ema = float(q_ema)
    q_ref = float(q_ref)
    for name, value in (("q_ema", q_ema), ("q_ref", q_ref)):
        if not 0.0 <= value <= 1.0:
            raise ValueError("%s must be in [0, 1], got %r" % (name, value))
    target = q_target_fraction * q_ref
    error = q_ema - target
    proposed = float(current) + float(step) * error
    new = clamp_lambda(proposed)
    return new, {
        "lambda_before": float(current),
        "lambda_after": new,
        "clamped": new != proposed,
        "q_ema": q_ema,
        "q_ref": q_ref,
        "q_target": target,
        "q_target_fraction": q_target_fraction,
        "error": error,
        "step": float(step),
        "direction": "raise_price" if error > 0 else ("lower_price" if error < 0 else "hold"),
    }


def ema(previous, value, *, alpha=Q_EMA_ALPHA):
    """Exponential average with the first observation taken as-is (no zero-start bias)."""
    if previous is None:
        return float(value)
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1], got %r" % (alpha,))
    return (1.0 - alpha) * float(previous) + alpha * float(value)


def bytes_per_on_time(cost_bytes, on_time):
    """``bytes / on_time``; ``None`` when nothing completed, never a fake zero or infinity."""
    on_time = int(on_time)
    if on_time <= 0:
        return None
    return float(cost_bytes) / float(on_time)


def improvement_verdict(*, cell, learned_on_time, learned_bytes, immediate_on_time,
                        immediate_bytes):
    """The registered test for one cell, applied literally.

    Improvement requires BOTH: at least 95% of ``normal_immediate``'s on-time count in the
    same cell, and strictly fewer bytes. Anything else is reported as not met, including the
    case where the learned arm is cheaper but completes fewer than 95%.

    ``None`` for the learned on-time rate when the arm has no evaluation point yet; the
    verdict then says "no evaluation" rather than guessing.
    """
    if learned_on_time is None:
        return {"cell": cell, "improved": False, "reason": "no evaluation point yet"}
    if immediate_on_time is None or immediate_on_time <= 0:
        return {"cell": cell, "improved": False,
                "reason": "no immediate reference for this cell"}
    ratio = float(learned_on_time) / float(immediate_on_time)
    fewer = float(learned_bytes) < float(immediate_bytes)
    improved = ratio >= Q_TARGET_FRACTION and fewer
    if improved:
        reason = "success >= 95% of immediate and fewer bytes"
    elif not fewer:
        reason = "bytes not lower than immediate"
    else:
        reason = "success %.3f of immediate, below the 0.95 requirement" % ratio
    return {
        "cell": cell,
        "improved": improved,
        "success_ratio_vs_immediate": ratio,
        "bytes_saved": float(immediate_bytes) - float(learned_bytes),
        "reason": reason,
    }
