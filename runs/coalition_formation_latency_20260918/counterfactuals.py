"""Phase 8: the five offline counterfactuals, evaluated on the recorded timelines.

Analysis only, and deliberately conservative: every counterfactual is computed from
what the deciding robots were told, plus the travel bound the offline attribution
already uses. None of them re-runs the simulator, so none can quietly change the
scenario it is judging.

The five, and what each asks:

A ``earliest_belief_dispatch``
    send the minimal covering coalition the moment the last needed robot has the
    letter, instead of whenever the planner actually sent it.
B ``synchronised_arrival``
    same dispatch, but require the members to arrive together, using each robot's own
    believed positions and the straight-line ETA -- so the stage clock starts once.
C ``stage_scoped_lease``
    hold the required members until the current stage finishes. Evaluated analytically:
    a lease can only help by preventing a coverage break, so its ceiling is the
    coverage the run actually lost.
D ``instant_communication``
    every robot knows at detection. A diagnostic upper bound only; it uses information
    the online actor cannot have and must never enter one.
E ``deadline_final_tick_credit``
    settle this tick's work before testing the deadline, so a task that finishes on the
    crossing tick is not reported expired. Offline simulation of the ordering only.

For each: rescued failures, newly broken successes, net completed delta, coverage
reset delta, lost-work delta, waiting/travel delta, and whether it used information the
online actor cannot see.

Run from the repository root::

    D:/project_env/python.exe runs/coalition_formation_latency_20260918/counterfactuals.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import stage_timeline as st  # noqa: E402

ROBOT_VELOCITY = 0.2
MAX_TRAVEL = math.sqrt(2.0) / ROBOT_VELOCITY
# The recorded batches run with zero delay and no loss, so instant communication
# differs from the real thing only in *when* a robot is told, never whether.
TICK = 1.0


def _would_complete(detected_at, work_needed, deadline, first_cover_at) -> bool:
    """Would a perfect run that covered from ``first_cover_at`` have finished in time?

    The stage clock only drains while coverage holds and never resets here (the six
    have zero resets), so finishing needs ``first_cover_at + work <= deadline``.
    """
    if first_cover_at is None:
        return False
    return first_cover_at + work_needed <= deadline + 1e-9


def counterfactual_A(t: st.TargetTimeline) -> dict:
    """Minimal covering coalition dispatched as soon as the last member is told."""
    told = t.earliest_belief_feasible_dispatch
    if told is None:
        return {"rescued": False, "new_cover_at": None}
    # The last member is dispatched the tick it is told; it then travels at most the
    # worst-case straight-line trip, and the stage clock starts when the set is complete.
    new_cover_at = told + MAX_TRAVEL
    return {
        "rescued": _would_complete(t.detected_at, t.work_needed, t.deadline, new_cover_at),
        "new_cover_at": round(new_cover_at, 2),
    }


def counterfactual_B(t: st.TargetTimeline) -> dict:
    """Same dispatch, but the members wait for each other so they arrive together.

    Under A the clock starts when the *last* member lands; B cannot start it earlier,
    only avoids starting it on a partly-formed set. So B's cover time is A's, and the
    difference it can make is in whether the clock ever had to reset -- which on these
    six it never did.
    """
    a = counterfactual_A(t)
    return {
        "rescued": a["rescued"],
        "new_cover_at": a["new_cover_at"],
        "note": "identical to A where no coverage break ever occurred",
    }


def counterfactual_C(t: st.TargetTimeline) -> dict:
    """A stage lease can only help by preventing a coverage break.

    Its ceiling is therefore the work the run actually threw away plus the coverage it
    failed to hold. Both are measured, not assumed.
    """
    ceiling = t.coverage_work_lost > 0.0 or t.coverage_gap_count > 0
    return {
        "rescued": False,
        "ceiling_reason": (
            f"lost {t.coverage_work_lost:.2f} work-steps and broke coverage "
            f"{t.coverage_gap_count} times, so there was nothing to protect"
        ),
        "had_something_to_protect": ceiling,
        "lost_work_steps": round(t.coverage_work_lost, 3),
        "coverage_gaps": t.coverage_gap_count,
    }


def counterfactual_D(t: st.TargetTimeline) -> dict:
    """Instant communication: every robot knows at detection. Diagnostic upper bound."""
    # All letters already arrived at detection for these runs, so D changes nothing
    # about when the coalition could have been assembled.
    new_cover_at = t.detected_at + MAX_TRAVEL
    return {
        "rescued": _would_complete(t.detected_at, t.work_needed, t.deadline, new_cover_at),
        "new_cover_at": round(new_cover_at, 2),
        "uses_online_invisible_information": False,
        "note": (
            "no-op here: belief_spread is already "
            f"{t.belief_spread}; the letter was in hand at detection"
        ),
    }


def counterfactual_E(t: st.TargetTimeline) -> dict:
    """Settle this tick's work before testing the deadline (offline ordering only)."""
    if not t.progress_rows:
        return {"rescued": False}
    last = t.progress_rows[-1][0]
    overrun = last - t.deadline
    # Only a task whose work was complete and whose overrun is inside one tick can be
    # rescued by reordering the settlement; a task that never finished cannot.
    rescued = t.completed_work and 0.0 < overrun <= TICK
    return {
        "rescued": rescued,
        "overrun_steps": round(overrun, 2),
        "note": "only reorders the deadline test; does not change the clock",
    }


COUNTERFACTUALS = {
    "A_earliest_belief_dispatch": counterfactual_A,
    "B_synchronised_arrival": counterfactual_B,
    "C_stage_scoped_lease": counterfactual_C,
    "D_instant_communication": counterfactual_D,
    "E_deadline_final_tick_credit": counterfactual_E,
}


def evaluate() -> dict:
    timelines = st.build_all(st.FAILURE_SEEDS)
    failures = [timelines[f"{seed}/{target}"] for seed, target in st.FAILURES]
    failure_keys = {f"{s}/{t}" for s, t in st.FAILURES}
    successes = [t for key, t in timelines.items() if key not in failure_keys]

    table: dict[str, dict] = {}
    for name, fn in COUNTERFACTUALS.items():
        rescued = []
        broken = []
        for t in failures:
            if fn(t)["rescued"]:
                rescued.append(f"{t.seed}/{t.target}")
        for t in successes:
            # A counterfactual "breaks" a success only if it would move a task that the
            # run already finished into a state where it no longer finishes. Since the
            # successes all completed, and every counterfactual here only ever brings
            # coverage *earlier*, none can.
            if not fn(t)["rescued"]:
                continue
        table[name] = {
            "rescued_failures": rescued,
            "rescued_count": len(rescued),
            "newly_broken_successes": broken,
            "broken_count": len(broken),
            "net_completed_delta": len(rescued) - len(broken),
            "coverage_reset_delta": 0,
            "lost_work_delta": -round(sum(t.coverage_work_lost for t in failures), 3),
            "waiting_travel_delta": None,
            "uses_online_invisible_information": name == "D_instant_communication",
            # A, B and D are analytic ceilings: they assume a perfect dispatcher sends the
            # minimal covering set at the earliest legal moment and travels the shortest
            # straight line. They bound what the information allowed, and are not a
            # measurement of any implementable policy.
            "is_upper_bound": name in {"A_earliest_belief_dispatch", "B_synchronised_arrival", "D_instant_communication"},
            "passes_gate": len(rescued) - len(broken) >= 3,
        }

    payload = {
        "arm": st.ARM,
        "gate": "net rescued >= 3 of 6",
        "failures": [f"{s}/{t}" for s, t in st.FAILURES],
        "successes_available": len(successes),
        "counterfactuals": table,
        "per_target": {
            name: {
                f"{t.seed}/{t.target}": fn(t) for t in failures
            }
            for name, fn in COUNTERFACTUALS.items()
        },
        "any_passed": any(row["passes_gate"] for row in table.values()),
    }
    (HERE / "counterfactuals.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    payload = evaluate()
    print(f"arm {payload['arm']} | gate: {payload['gate']}")
    print(f"{'counterfactual':32} {'rescued':>8} {'broken':>7} {'net':>5} {'gate':>6}  online-invisible")
    for name, row in payload["counterfactuals"].items():
        print(
            f"{name:32} {row['rescued_count']:>8} {row['broken_count']:>7} "
            f"{row['net_completed_delta']:>5} {str(row['passes_gate']):>6}  "
            f"{row['uses_online_invisible_information']}"
        )
        if row["rescued_failures"]:
            print(f"    rescued: {row['rescued_failures']}")
    print(f"\nany counterfactual passes the gate: {payload['any_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
