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
#: The success target relative to the strong rule for the same cell.
Q_TARGET_FRACTION = 0.95


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


def scaled_cost(cost_bytes, subscribers, goals, *, reference=REFERENCE_CELL):
    """``bytes / (N * G)``, rescaled so the frozen reference cell keeps the old magnitude.

    The task registers ``c = cost_bytes / (N * G)`` for cross-cell comparability, and that
    divisor alone is right for comparability: a byte costs the same in every cell once divided
    by the fan-out and the goal count.

    But it is not right for *magnitude*. With ~2.4 kB per send, ``2400 / (6 * 12) = 33`` scaled
    units per send, while one converted task is worth ``A_task = 1 - V(s) ~= 0.41``. At that
    ratio the registered ``lambda`` floor of 0.25 already prices a send at ~21x a task, so
    "never send" is optimal for every lambda in [0.25, 8] -- which is exactly what the first
    training run did (342 of 366 updates with zero sends, lambda pinned at the floor).

    The fix is a pure rescale, not a re-registration: multiply by the reference cell's own
    ``N*G / B0`` so that at ``(N, G) = (6, 24)`` the reward is exactly ``bytes / B0`` -- the
    magnitude every archived run used -- while any other cell is priced relative to that
    reference by the registered ``N*G`` ratio. One lambda therefore still means the same thing
    across cells, and it also means the same thing as it did before.
    """
    goals = int(goals)
    subscribers = int(subscribers)
    if goals < 1 or subscribers < 1:
        raise ValueError("subscribers and goals must be >= 1, got %d and %d"
                         % (subscribers, goals))
    ref_subscribers, ref_goals = (int(value) for value in reference)
    if ref_subscribers < 1 or ref_goals < 1:
        raise ValueError("the reference cell must be positive, got %r" % (reference,))
    return (float(cost_bytes) / float(subscribers * goals)
            * (float(ref_subscribers * ref_goals) / B0_BYTES))


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
