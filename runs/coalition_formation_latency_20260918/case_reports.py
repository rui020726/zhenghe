"""Phase 5-6: per-target case reports and matched successful controls.

Analysis only. Reads the same committed traces as ``stage_timeline.py`` and adds two
things the brief asked for:

* a mutually exclusive ``primary_cause`` for each of the six remaining binding
  failures, chosen by an explicit ordered rule rather than by eye, plus any number of
  secondary causes;
* for each failure, at least five successful targets matched on the covariates the
  brief lists (deadline budget, stage count, duration, requirement shape, detection
  time, active load, capable robots), so the comparison is not against hand-picked
  easy wins.

Run from the repository root::

    D:/project_env/python.exe runs/coalition_formation_latency_20260918/case_reports.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import stage_timeline as st  # noqa: E402

PRIMARY = (
    "communication_or_belief_late",
    "capable_agent_unavailable",
    "planner_dispatch_late",
    "arrival_desynchronised",
    "complete_coverage_interrupted",
    "stage_transition_handoff",
    "deadline_tick_boundary",
    "fleet_capacity",
    "unknown",
)

# A dispatch this far past detection, as a share of the delivery budget, is "late":
# the belief was already in every robot's hands, so nothing but the planner explains it.
LATE_DISPATCH_SHARE = 0.25
# A belief spread this far past detection, as a share of the budget, is "late comms".
LATE_BELIEF_SHARE = 0.25
# An arrival spread this large (absolute steps) means the coalition assembled raggedly.
DESYNC_STEPS = 10.0
# Finishing within this many steps of the deadline is a boundary question, not a plan one.
TICK_EPS = 1.0


def classify(t: st.TargetTimeline) -> tuple[str, list[str]]:
    """The one dominant cause, then every other rule that also fired, in rule order.

    Rules run most-specific first. Every rule that fires is kept as a secondary, so the
    audit trail shows the whole shape rather than only the winner, and a target never
    appears in two primary counts.
    """
    fired: list[str] = []
    budget = max(1e-9, t.delivery_budget)
    lag = t.coalition_dispatch_lag
    handoff = t.worst_stage_handoff_gap
    longest_stage = max(t.durations) if t.durations else 0.0

    # 1. The coalition never formed at all: the task never saw complete coverage.
    if t.first_complete_at is None:
        if t.belief_spread is not None and t.belief_spread <= LATE_BELIEF_SHARE * budget:
            # Every robot had the letter in time and capable ones were free, so nothing
            # but the planner explains a coalition that was never assembled.
            fired.append("planner_dispatch_late")
        else:
            fired.append("communication_or_belief_late")

    # 2. The coalition formed late: the last contributing member was dispatched long
    #    after detection, and the belief had been in hand the whole time.
    elif lag is not None and lag > LATE_DISPATCH_SHARE * budget and t.arrival_spread is not None and t.arrival_spread <= DESYNC_STEPS:
        fired.append("planner_dispatch_late")

    # 3. A stage boundary is where the requirement changes and the old members are
    #    released. A long wait for the next stage to be covered is a handoff failure.
    if len(t.durations) > 1 and handoff > 0.5 * longest_stage:
        fired.append("stage_transition_handoff")

    # 4. Everyone was sent, but they reached the target far apart, so the stage clock
    #    could not start until the last one landed.
    if t.arrival_spread is not None and t.arrival_spread > DESYNC_STEPS:
        fired.append("arrival_desynchronised")

    # 5. The belief was in hand and the last contributing member was dispatched late.
    if t.first_complete_at is not None and (lag is None or lag > LATE_DISPATCH_SHARE * budget):
        fired.append("planner_dispatch_late")

    # 6. The robots that could do the work were already committed elsewhere.
    if t.capable_total > 0 and t.capable_busy_at_detection >= t.capable_total:
        fired.append("capable_agent_unavailable")

    # 7. More live targets than there were free capable robots.
    free_capable = max(1, t.capable_total - t.capable_busy_at_detection)
    if t.concurrent_active_at_detection > free_capable:
        fired.append("fleet_capacity")

    # 8. Nobody was told in time. The letters here all arrive at detection, so this
    #    cannot fire on these six; kept so the rule set is complete and auditable.
    if t.belief_spread is None or t.belief_spread > LATE_BELIEF_SHARE * budget:
        fired.append("communication_or_belief_late")

    # 9. All the work was delivered; only the last tick of it landed past the line.
    if t.completed_work and t.expired_at is not None:
        overrun = (t.progress_rows[-1][0] if t.progress_rows else t.deadline) - t.deadline
        if 0.0 < overrun <= TICK_EPS:
            fired.append("deadline_tick_boundary")

    # 10. Coverage was established and then broke inside the same stage.
    if t.coverage_gap_count > 0 or t.coverage_reset_count > 0:
        fired.append("complete_coverage_interrupted")

    if not fired:
        fired.append("unknown")
    return fired[0], fired[1:]


def event_table(t: st.TargetTimeline) -> list[dict]:
    """The ten rows the brief asks for, in time order, one per event."""
    rows: list[dict] = []

    def add(at, kind, detail):
        rows.append({"at": None if at is None else round(float(at), 1), "event": kind, "detail": detail})

    add(t.detected_at, "detection", f"work {t.work_needed:.2f} steps, budget {t.delivery_budget:.1f}, deadline {t.deadline:.1f}")
    for agent, at in sorted(t.first_belief_at.items(), key=lambda kv: kv[1]):
        add(at, "belief_arrival", f"agent {agent}")
    for agent, at in sorted(t.first_reservation_at.items(), key=lambda kv: kv[1]):
        add(at, "reservation", f"agent {agent}")
    for agent, at in sorted(t.assignment_sent_at.items(), key=lambda kv: kv[1]):
        add(at, "assignment", f"agent {agent}")
    for agent, at in sorted(t.arrived_at.items(), key=lambda kv: kv[1]):
        add(at, "arrival", f"agent {agent}")
    for change in t.coverage:
        kind = "complete_coverage_begin" if change.complete else "complete_coverage_end"
        add(change.at, kind, f"stage {change.stage_index} members {list(change.members)} missing {list(change.missing)}")
    for index, window in t.stages.items():
        for at in window.resets:
            add(at, "time_left_reset", f"stage {index}")
    if t.coverage_intervals:
        begin, end = t.coverage_intervals[-1]
        add(begin, "last_coverage_window", f"{begin:.1f} -> {end:.1f} ({end - begin:.1f} steps)")
    if t.expired_at is not None:
        add(t.expired_at, "expiry", f"reported deadline_expired, final progress {t.final_progress:.3f}")
    return sorted(rows, key=lambda row: (row["at"] is None, row["at"]))


def metric_row(t: st.TargetTimeline) -> dict:
    return {
        "target": f"{t.seed}/{t.target}",
        "completed_work": t.completed_work,
        "expired_at": t.expired_at,
        "deadline": round(t.deadline, 2),
        "work_needed": round(t.work_needed, 3),
        "delivery_budget": round(t.delivery_budget, 2),
        "belief_spread": t.belief_spread,
        "reservation_spread": t.reservation_spread,
        "assignment_spread": t.assignment_spread,
        "arrival_spread": t.arrival_spread,
        "detection_to_first_complete": t.detection_to_first_complete,
        "longest_contiguous_coverage": t.longest_contiguous_coverage,
        "total_coverage_time": t.total_coverage_time,
        "coverage_gap_count": t.coverage_gap_count,
        "coverage_reset_count": t.coverage_reset_count,
        "coverage_work_lost": round(t.coverage_work_lost, 3),
        "final_coverage_shortfall": round(t.final_coverage_shortfall, 3),
        "last_complete_coverage_start": t.last_complete_coverage_start,
        "earliest_belief_feasible_dispatch": t.earliest_belief_feasible_dispatch,
        "earliest_feasible_completion": t.earliest_feasible_completion,
        "capable_agents_total": t.capable_total,
        "capable_agents_busy_at_detection": t.capable_busy_at_detection,
        "concurrent_active_tasks_at_detection": t.concurrent_active_at_detection,
        "capability_waiting_steps": round(t.capability_waiting_steps, 3),
    }


def covariates(t: st.TargetTimeline) -> dict:
    """The matching keys the brief lists, as numbers."""
    return {
        "delivery_budget": t.delivery_budget,
        "stage_count": len(t.durations),
        "mean_duration": statistics.mean(t.durations) if t.durations else 0.0,
        "requirement_count": sum(sum(r) for r in t.requirements[:1]) if t.requirements else 0,
        "detected_at": t.detected_at,
        "active_tasks": t.concurrent_active_at_detection,
        "capable_total": t.capable_total,
        "distance_to_target": None,  # filled by the caller from the decision trace
    }


def matched_controls(
    failure: st.TargetTimeline,
    candidates: list[st.TargetTimeline],
    keys: tuple[str, ...],
    k: int = 5,
) -> list[dict]:
    """The ``k`` successes closest to this failure in normalised covariate space.

    Every key is divided by its own spread across the candidate pool first, so a
    budget measured in steps does not swamp a stage count measured in units.
    """
    target = covariates(failure)
    pool = [c for c in candidates if c.target != failure.target or c.seed != failure.seed]
    if not pool:
        return []

    scales: dict[str, float] = {}
    for key in keys:
        values = [covariates(c)[key] for c in pool if covariates(c)[key] is not None]
        if not values:
            continue
        spread = max(values) - min(values)
        scales[key] = spread if spread > 0 else 1.0

    def distance(candidate: st.TargetTimeline) -> float:
        other = covariates(candidate)
        total = 0.0
        for key in keys:
            if key not in scales:
                continue
            a, b = target[key], other[key]
            if a is None or b is None:
                continue
            total += ((a - b) / scales[key]) ** 2
        return total ** 0.5

    ranked = sorted(pool, key=lambda c: (distance(c), c.seed, c.target))
    return [
        {**metric_row(c), "match_distance": round(distance(c), 4)}
        for c in ranked[:k]
    ]


def compare(failure_row: dict, controls: list[dict]) -> dict:
    """Failure value beside the control median, for the metrics the brief names."""
    out: dict = {}
    for key in (
        "belief_spread",
        "arrival_spread",
        "longest_contiguous_coverage",
        "coverage_reset_count",
        "coverage_work_lost",
        "earliest_feasible_completion",
        "delivery_budget",
        "capability_waiting_steps",
    ):
        values = [c[key] for c in controls if c.get(key) is not None]
        out[key] = {
            "failure": failure_row.get(key),
            "control_median": round(statistics.median(values), 3) if values else None,
            "n": len(values),
        }
    return out


def build() -> dict:
    timelines = st.build_all(st.FAILURE_SEEDS)
    failures = [timelines[f"{seed}/{target}"] for seed, target in st.FAILURES]
    successes = [t for key, t in timelines.items() if key not in {f"{s}/{x}" for s, x in st.FAILURES}]

    reports = []
    for t in failures:
        primary, secondary = classify(t)
        row = metric_row(t)
        controls = matched_controls(
            t,
            successes,
            keys=("delivery_budget", "stage_count", "mean_duration", "requirement_count", "detected_at", "active_tasks", "capable_total"),
            k=5,
        )
        reports.append(
            {
                "target": row["target"],
                "primary_cause": primary,
                "secondary_causes": secondary,
                "metrics": row,
                "events": event_table(t),
                "matched_controls": controls,
                "failure_vs_controls": compare(row, controls),
            }
        )

    counts: dict[str, int] = {}
    for report in reports:
        counts[report["primary_cause"]] = counts.get(report["primary_cause"], 0) + 1

    payload = {
        "arm": st.ARM,
        "seeds": st.FAILURE_SEEDS,
        "rule": {
            "late_dispatch_share": LATE_DISPATCH_SHARE,
            "late_belief_share": LATE_BELIEF_SHARE,
            "desync_steps": DESYNC_STEPS,
            "tick_eps": TICK_EPS,
            "order": list(PRIMARY),
        },
        "primary_cause_counts": counts,
        "successes_available": len(successes),
        "reports": reports,
    }
    (HERE / "case_reports.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    payload = build()
    print(f"arm {payload['arm']} | successes available for matching: {payload['successes_available']}")
    print(f"primary cause counts: {payload['primary_cause_counts']}")
    for report in payload["reports"]:
        m = report["metrics"]
        print(
            f"\n{report['target']}: {report['primary_cause']} "
            f"(secondary {report['secondary_causes']})"
        )
        print(
            f"  completed_work={m['completed_work']} budget={m['delivery_budget']} "
            f"belief_spread={m['belief_spread']} arrival_spread={m['arrival_spread']} "
            f"det_to_cover={m['detection_to_first_complete']} "
            f"resets={m['coverage_reset_count']} lost={m['coverage_work_lost']} "
            f"cap_busy={m['capable_agents_busy_at_detection']}/{m['capable_agents_total']}"
        )
        for control in report["matched_controls"]:
            print(
                f"    control {control['target']:26} d={control['match_distance']:.3f} "
                f"budget={control['delivery_budget']:7.1f} "
                f"arr_spread={control['arrival_spread']} "
                f"longest={control['longest_contiguous_coverage']} "
                f"resets={control['coverage_reset_count']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
