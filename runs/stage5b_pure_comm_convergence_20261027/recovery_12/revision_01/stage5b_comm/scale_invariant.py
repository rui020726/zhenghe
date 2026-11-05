"""Cost-side scaling algebra for the arbitrary-N pre-experiments (P1).

Pure functions only: no world, no torch, no I/O. Every quantity the fixed-budget
calibre needs, expressed in the subscriber count ``n`` and one measured constant.

Why ``fixed`` is the main calibre and ``proportional`` only an analytic control:
under proportional scaling the budget and B0 both grow as ``n/6``, so the cost of
one send grows as ``n`` and all three decision-relevant ratios cancel --
``share_per_send``, ``affordable_sends`` and ``advantage_cost_per_send`` become
independent of ``n``. Replaying worlds there can only confirm the algebra is
self-consistent. Under ``fixed`` the budget stays at 58,600 B while a send costs
``n`` times as much, so affordability falls as ``23*6/n`` and scarcity -- the thing
a controller has to generalise over -- actually moves with ``n``.

PAYLOAD_BYTES is derived, not asserted: the frozen rule-arm rows give
459,756 paid bytes over 184 emissions at the frozen subscriber count of 6, and
``airtime_bytes == charged_bytes == paid_bytes`` on every one of those rows, i.e.
the recorded figure is already the payload times the fan-out.
"""

from __future__ import annotations

SUBSCRIBERS_REF = 6
BUDGET_BYTES_REF = 58600
B0_BYTES_REF = 10000
MODES = ("fixed", "proportional")

#: Provenance of the one measured constant. Frozen artifacts, read-only.
PAID_BYTES_TOTAL = 459756
EMISSIONS_TOTAL = 184
PAYLOAD_PROVENANCE = {
    "numerator_paid_bytes": PAID_BYTES_TOTAL,
    "denominator_emissions": EMISSIONS_TOTAL,
    "denominator_subscribers": SUBSCRIBERS_REF,
    "source": "runs/stage5b_pure_comm_budget58600_20261026/revision_02/execution/rows/eval/"
              "eval_normal_immediate_none_<scene>.json, scenes 2301901..2301908",
    "fields": "paid_bytes == charged_bytes == airtime_bytes on all 8 rows, so the recorded "
              "byte total is already payload * fan-out; dividing by emissions and by "
              "SUBSCRIBERS_REF recovers the mean payload",
    "cost_per_send_at_ref": PAID_BYTES_TOTAL / EMISSIONS_TOTAL,
    "note": "estimate_message_bytes() prices the canonical JSON of one DetectionMessage, so "
            "the true payload is a distribution (per-scene mean 408.61..423.61 B); the "
            "pooled mean over all 184 frozen sends is what the algebra below uses.",
}

PAYLOAD_BYTES = PAID_BYTES_TOTAL / EMISSIONS_TOTAL / SUBSCRIBERS_REF

#: The rounded figures the P1 acceptance targets were written from, kept only so the
#: report can show where those literals come from. Not used by any function below.
SPEC_LITERAL_COST_PER_SEND = 2500.6
SPEC_LITERAL_PAYLOAD_BYTES = SPEC_LITERAL_COST_PER_SEND / SUBSCRIBERS_REF


def _check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError("mode must be one of %r, got %r" % (MODES, mode))
    return mode


def cost_per_send(n: int) -> float:
    """Bytes one emission costs when ``n`` subscribers are addressed (comm.py:908)."""
    return PAYLOAD_BYTES * int(n)


def budget_bytes(n: int, mode: str = "fixed") -> float:
    """The episode byte credit. ``fixed`` holds 58,600 B at every ``n``."""
    _check_mode(mode)
    return BUDGET_BYTES_REF * int(n) / SUBSCRIBERS_REF if mode == "proportional" \
        else float(BUDGET_BYTES_REF)


def b0_bytes(n: int, mode: str = "fixed") -> float:
    """The advantage normaliser B0 (``A_cost = bytes / B0``)."""
    _check_mode(mode)
    return B0_BYTES_REF * int(n) / SUBSCRIBERS_REF if mode == "proportional" \
        else float(B0_BYTES_REF)


def affordable_sends(n: int, mode: str = "fixed") -> int:
    """How many emissions the credit buys, as an integer count."""
    return int(budget_bytes(n, mode) // cost_per_send(n))


def share_per_send(n: int, mode: str = "fixed") -> float:
    """Fraction of the episode credit one emission consumes (0..1, not percent)."""
    return cost_per_send(n) / budget_bytes(n, mode)


def advantage_cost_per_send(n: int, mode: str = "fixed") -> float:
    """One emission's cost in advantage units, ``cost / B0``."""
    return cost_per_send(n) / b0_bytes(n, mode)


def row(n: int, mode: str = "fixed") -> dict:
    """All five quantities for one (n, mode) cell, with the denominators named."""
    _check_mode(mode)
    return {
        "n": int(n),
        "mode": mode,
        "payload_bytes": PAYLOAD_BYTES,
        "cost_per_send_bytes": cost_per_send(n),
        "budget_bytes": budget_bytes(n, mode),
        "b0_bytes": b0_bytes(n, mode),
        "affordable_sends": affordable_sends(n, mode),
        "share_per_send": share_per_send(n, mode),
        "share_per_send_pct": 100.0 * share_per_send(n, mode),
        "advantage_cost_per_send": advantage_cost_per_send(n, mode),
        "denominators": {
            "affordable_sends": "budget_bytes // cost_per_send_bytes",
            "share_per_send": "cost_per_send_bytes / budget_bytes",
            "advantage_cost_per_send": "cost_per_send_bytes / b0_bytes",
        },
    }


def table(ns=(3, 6, 12), modes=MODES) -> dict:
    """The full (n x mode) grid, keyed by mode then n."""
    return {mode: {int(n): row(n, mode) for n in ns} for mode in modes}
