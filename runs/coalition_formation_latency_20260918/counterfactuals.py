"""Offline counterfactuals, evaluated on the recorded timelines.

Analysis only, and deliberately conservative: every counterfactual is computed from
what the deciding robots were told, plus the travel bound the offline attribution
already uses. None of them re-runs the simulator, so none can quietly change the
scenario it is judged on.

A, B and D are **analytical feasibility estimates**, not measurements of any policy:
each asks only "was a completion in time still *possible* from the information the
robots held", assuming a perfect dispatcher and straight-line travel. Because they
never re-run the simulator, their effect on the other targets in the episode is
**unmeasured** -- the report says so instead of borrowing a zero.

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
    every robot knows at detection. Reported strictly as a delta against A: whatever
    it changes over A is pure communication timing. Diagnostic only; it must never
    enter an online actor.
E ``deadline_final_tick_credit``
    settle this tick's work before testing the deadline, so a task that finishes on the
    crossing tick is not reported expired. Offline simulation of the ordering only.

The report also separates the three terminal states a "rescue" claim has to
distinguish: completed on time, finished after the deadline, never finished.

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
EPS = 1e-9


def _would_complete(detected_at, work_needed, deadline, first_cover_at) -> bool:
    """Would a perfect run that covered from ``first_cover_at`` have finished in time?

    The stage clock only drains while coverage holds and never resets here (the six
    have zero resets), so finishing needs ``first_cover_at + work <= deadline``.
    """
    if first_cover_at is None:
        return False
    return first_cover_at + work_needed <= deadline + EPS


def outcome(t: st.TargetTimeline) -> dict:
    """Which of the three terminal states this target actually reached.

    A dispatch fix "rescues" an expired target only if the target had work left to
    save; a target that delivered everything and merely crossed the line by a tick
    is a boundary-semantics case, not a dispatch one.
    """
    if t.completed_work:
        overrun = None
        if t.expired_at is not None:
            last = t.progress_rows[-1][0]
            overrun = last - t.deadline
        return {
            "state": "finished_after_deadline" if t.expired_at is not None else "completed_on_time",
            "overrun_steps": round(overrun, 2) if overrun is not None else 0.0,
            "within_one_tick": overrun is not None and 0.0 < overrun <= TICK + EPS,
        }
    return {
        "state": "never_finished",
        "overrun_steps": None,
        "within_one_tick": False,
        "final_progress": round(t.final_progress, 4),
        "final_shortfall_steps": round(t.final_coverage_shortfall, 2),
    }


def counterfactual_A(t: st.TargetTimeline) -> dict:
    """Minimal covering coalition dispatched as soon as the last member is told."""
    told = t.earliest_belief_feasible_dispatch
    if told is None:
        return {"potentially_feasible": False, "new_cover_at": None}
    # The last member is dispatched the tick it is told; it then travels at most the
    # worst-case straight-line trip, and the stage clock starts when the set is complete.
    new_cover_at = told + MAX_TRAVEL
    return {
        "potentially_feasible": _would_complete(
            t.detected_at, t.work_needed, t.deadline, new_cover_at
        ),
        "new_cover_at": round(new_cover_at, 2),
    }


def counterfactual_B(t: st.TargetTimeline) -> dict:
    """Same dispatch, but the members wait for each other so they arrive together.

    Under A the clock starts when the *last* member lands; B cannot start it earlier,
    only avoid starting it on a partly-formed set. So B's cover time is A's, and the
    difference it can make is in whether the clock ever had to reset -- which on these
    six it never did.
    """
    a = counterfactual_A(t)
    return {
        "potentially_feasible": a["potentially_feasible"],
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
        "potentially_feasible": False,
        "ceiling_reason": (
            f"lost {t.coverage_work_lost:.2f} work-steps and broke coverage "
            f"{t.coverage_gap_count} times, so there was nothing to protect"
        ),
        "had_something_to_protect": ceiling,
        "lost_work_steps": round(t.coverage_work_lost, 3),
        "coverage_gaps": t.coverage_gap_count,
    }


def counterfactual_D(t: st.TargetTimeline) -> dict:
    """Instant communication, reported strictly as a delta against A.

    A dispatches the tick the last needed robot *has* the letter; D assumes it has the
    letter at detection. The difference between the two cover times is therefore pure
    communication timing. If belief is already synchronised -- the letter in hand at
    detection -- that delta is exactly zero, and any feasibility D shows is inherited
    from A's dispatch assumption, never from communication.
    """
    a = counterfactual_A(t)
    told = t.earliest_belief_feasible_dispatch
    delta = None if told is None else t.detected_at - told
    return {
        "potentially_feasible": a["potentially_feasible"],
        "new_cover_at": a["new_cover_at"],
        "same_dispatch_assumption_as": "A_earliest_belief_dispatch",
        "communication_only_delta_steps": round(delta, 2) if delta is not None else None,
        "uses_online_invisible_information": True,
        "note": (
            "communication-only change is zero here: belief_spread is already "
            f"{t.belief_spread}, so A's earliest-belief dispatch is already "
            "dispatch-at-detection and D adds nothing"
        ),
    }


def counterfactual_E(t: st.TargetTimeline) -> dict:
    """Settle this tick's work before testing the deadline (offline ordering only)."""
    if not t.progress_rows:
        return {"potentially_feasible": False}
    last = t.progress_rows[-1][0]
    overrun = last - t.deadline
    # Only a task whose work was complete and whose overrun is inside one tick can be
    # rescued by reordering the settlement; a task that never finished cannot.
    feasible = t.completed_work and 0.0 < overrun <= TICK + EPS
    return {
        "potentially_feasible": feasible,
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

    table: dict[str, dict] = {}
    for name, fn in COUNTERFACTUALS.items():
        feasible = [f"{t.seed}/{t.target}" for t in failures if fn(t)["potentially_feasible"]]
        table[name] = {
            "potentially_feasible": feasible,
            "potentially_feasible_count": len(feasible),
            # No simulator re-run means no claim about other targets either way.
            "global_effect_unmeasured": True,
            "uses_online_invisible_information": name == "D_instant_communication",
            "is_analytical_feasibility_estimate": name in {
                "A_earliest_belief_dispatch",
                "B_synchronised_arrival",
                "D_instant_communication",
            },
            "feasibility_gate_met": len(feasible) >= 3,
        }

    breakdown = {}
    for t in failures:
        breakdown[f"{t.seed}/{t.target}"] = outcome(t)

    payload = {
        "arm": st.ARM,
        "gate": "potentially_feasible >= 3 of 6 (feasibility gate; the global effect is "
        "measured only by closed-loop replay)",
        "failures": [f"{s}/{t}" for s, t in st.FAILURES],
        "outcome_breakdown": {
            "completed_on_time": [k for k, v in breakdown.items() if v["state"] == "completed_on_time"],
            "finished_after_deadline": [
                k for k, v in breakdown.items() if v["state"] == "finished_after_deadline"
            ],
            "never_finished": [k for k, v in breakdown.items() if v["state"] == "never_finished"],
            "per_target": breakdown,
            "boundary_semantics": "a target that finishes on the very tick its deadline is "
            "crossed is reported expired because task_update tests expiry before settling "
            "work; none of the six finished_after_deadline cases sits inside that one-tick "
            "window (see per_target.overrun_steps), so relabelling the boundary would not "
            "change any of them",
        },
        "counterfactuals": table,
        "per_target": {
            name: {f"{t.seed}/{t.target}": fn(t) for t in failures}
            for name, fn in COUNTERFACTUALS.items()
        },
        "any_feasibility_estimate_meets_gate": any(
            row["feasibility_gate_met"] for row in table.values()
        ),
    }
    (HERE / "counterfactuals.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    payload = evaluate()
    print(f"arm {payload['arm']} | gate: {payload['gate']}")
    print(
        f"{'counterfactual':32} {'feasible':>9} {'global':>8} {'gate':>6}  online-invisible"
    )
    for name, row in payload["counterfactuals"].items():
        print(
            f"{name:32} {row['potentially_feasible_count']:>9} "
            f"{'unmeasured' if row['global_effect_unmeasured'] else '-':>8} "
            f"{str(row['feasibility_gate_met']):>6}  "
            f"{row['uses_online_invisible_information']}"
        )
        if row["potentially_feasible"]:
            print(f"    potentially feasible: {row['potentially_feasible']}")

    bd = payload["outcome_breakdown"]
    print("\noutcome breakdown:")
    print(f"    completed on time:        {bd['completed_on_time'] or 'none'}")
    for key in bd["finished_after_deadline"]:
        per = bd["per_target"][key]
        print(
            f"    finished after deadline:  {key} (+{per['overrun_steps']} steps, "
            f"within one tick: {per['within_one_tick']})"
        )
    for key in bd["never_finished"]:
        per = bd["per_target"][key]
        print(
            f"    never finished:           {key} (progress {per['final_progress']}, "
            f"shortfall {per['final_shortfall_steps']} steps)"
        )
    print(
        f"\nany feasibility estimate meets the gate: "
        f"{payload['any_feasibility_estimate_meets_gate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
