"""Stage-level timeline and coverage metrics for the six remaining binding failures.

Analysis only. Reads the committed ``coalplan0.25sticky`` traces and the already-verified
``deliveries_s*.json`` recordings; never re-runs the simulator, never imports into
``integrated_sar``.

Sources, joined on ``(seed, target, timestamp)``:

* ``rp_<arm>_s<seed>.jsonl`` -- detection, progress rows (which carry the on-site member
  set and the stage's remaining requirement), per-target assignment rows, final verdict;
* ``tr_<arm>_s<seed>.jsonl`` -- per-robot reservations, the mask, the chosen task;
* ``deliveries_s<seed>.json`` -- when each robot was handed the detection letter, the only
  source for a robot's own belief arrival time.

``progress`` rows are written only when the reported state changes, so every interval is
integrated between consecutive rows rather than counted in rows. That is what makes
``longest_contiguous_coverage`` a duration instead of a row count.

Run from the repository root::

    D:/project_env/python.exe runs/coalition_formation_latency_20260918/stage_timeline.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from integrated_sar.contracts import GridPosition  # noqa: E402
from integrated_sar.expired_attribution import ROBOT_VELOCITY, stage_plan_for  # noqa: E402

ARMS = REPO / "runs" / "coordinator_baselines_20260917" / "heldout" / "arms"
DELIVERIES = REPO / "runs" / "duration_relevance_20260917"
ARM = "coalplan0.25sticky"
MAX_TRAVEL = math.sqrt(2.0) / ROBOT_VELOCITY

FAILURES = [
    (20003, "piloc-e1-target-1"),
    (20003, "piloc-e1-target-0"),
    (20008, "piloc-e1-target-2"),
    (20052, "piloc-e1-target-3"),
    (20070, "piloc-e1-target-2"),
    (20083, "piloc-e1-target-4"),
]
FAILURE_SEEDS = sorted({seed for seed, _ in FAILURES})
EPS = 1e-9


def load_records(seed: int) -> list[dict]:
    path = ARMS / ARM / f"rp_{ARM}_s{seed}.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_decisions(seed: int) -> list[dict]:
    path = ARMS / ARM / f"tr_{ARM}_s{seed}.jsonl"
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event_type") == "decision":
            out.append(row)
    return out


def load_deliveries(seed: int) -> dict:
    return json.loads((DELIVERIES / f"deliveries_s{seed}.json").read_text(encoding="utf-8"))


@dataclass
class CoverageChange:
    at: float
    complete: bool
    members: tuple[str, ...]
    covered: tuple[int, ...]
    missing: tuple[int, ...]
    stage_index: int
    progress: float
    committed: tuple[str, ...] = ()
    reservation_ids: tuple[str, ...] = ()
    release_reason: str | None = None
    hard_masked: tuple[str, ...] = ()
    belief_stale: tuple[str, ...] = ()


@dataclass
class StageWindow:
    index: int
    duration: float | None
    requirements: tuple[int, ...]
    first_complete_at: float | None = None
    last_complete_at: float | None = None
    completed_at: float | None = None
    resets: list[float] = field(default_factory=list)


@dataclass
class TargetTimeline:
    seed: int
    target: str
    detected_at: float
    deadline: float
    position: GridPosition | None
    durations: tuple[float, ...]
    requirements: tuple[tuple[int, ...], ...]
    expired_at: float | None = None
    final_progress: float = 0.0
    final_stage: int = 0
    final_status: str | None = None
    first_belief_at: dict[str, float] = field(default_factory=dict)
    belief_revisions: dict[str, list] = field(default_factory=dict)
    first_reservation_at: dict[str, float] = field(default_factory=dict)
    reservation_ids: dict[str, list[str]] = field(default_factory=dict)
    assignment_sent_at: dict[str, float] = field(default_factory=dict)
    departed_at: dict[str, float] = field(default_factory=dict)
    arrived_at: dict[str, float] = field(default_factory=dict)
    left_at: dict[str, float] = field(default_factory=dict)
    on_site: dict[str, list] = field(default_factory=dict)
    coverage: list[CoverageChange] = field(default_factory=list)
    stages: dict[int, StageWindow] = field(default_factory=dict)
    progress_rows: list = field(default_factory=list)
    hard_masked_agents: set[str] = field(default_factory=set)
    concurrent_active_at_detection: int = 0
    capable_total: int = 0
    capable_busy_at_detection: int = 0
    detection_to_first_needs_decision: float | None = None
    capability_waiting_steps: float = 0.0

    @property
    def work_needed(self) -> float:
        return float(sum(self.durations))

    @property
    def delivery_budget(self) -> float:
        return self.deadline - self.detected_at - self.work_needed

    def _spread(self, table: dict[str, float]) -> float | None:
        return None if not table else max(table.values()) - self.detected_at

    @property
    def belief_spread(self) -> float | None:
        return self._spread(self.first_belief_at)

    @property
    def reservation_spread(self) -> float | None:
        return self._spread(self.first_reservation_at)

    @property
    def assignment_spread(self) -> float | None:
        return self._spread(self.assignment_sent_at)

    @property
    def arrival_spread(self) -> float | None:
        return self._spread(self.arrived_at)

    @property
    def first_complete_at(self) -> float | None:
        return next((c.at for c in self.coverage if c.complete), None)

    @property
    def detection_to_first_complete(self) -> float | None:
        first = self.first_complete_at
        return None if first is None else first - self.detected_at

    @property
    def coverage_intervals(self) -> list[tuple[float, float]]:
        """Complete-coverage intervals, integrated between consecutive state changes."""
        intervals: list[tuple[float, float]] = []
        start: float | None = None
        for index, change in enumerate(self.coverage):
            following = self.coverage[index + 1].at if index + 1 < len(self.coverage) else None
            if change.complete:
                start = change.at if start is None else start
                if following is None:
                    intervals.append((start, change.at))
                    start = None
            elif start is not None:
                intervals.append((start, change.at))
                start = None
        return intervals

    @property
    def longest_contiguous_coverage(self) -> float:
        return max((end - begin for begin, end in self.coverage_intervals), default=0.0)

    @property
    def total_coverage_time(self) -> float:
        return float(sum(end - begin for begin, end in self.coverage_intervals))

    @property
    def coverage_gap_count(self) -> int:
        """Coverage breaks *within* a stage, after that stage had been covered.

        A stage transition also drops ``complete`` to False (the next stage's
        requirement is unmet by definition), so counting every False would report one
        gap per stage boundary. Only a same-stage break is an interruption.
        """
        seen: dict[int, bool] = {}
        gaps = 0
        for change in self.coverage:
            if change.complete:
                seen[change.stage_index] = True
            elif seen.get(change.stage_index):
                seen[change.stage_index] = False
                gaps += 1
        return gaps

    @property
    def coverage_reset_count(self) -> int:
        return sum(len(stage.resets) for stage in self.stages.values())

    @property
    def coverage_work_lost(self) -> float:
        """Robot-steps of covered work thrown away by a same-stage ``time_left`` reset.

        The reset restores ``stage["time"]``, so what is lost is exactly the elapsed
        fraction at the moment coverage broke: ``stage_fraction * duration``. Two
        things are *not* losses and are excluded: a stage transition (that stage's work
        was banked, and the next stage's requirement is unmet by definition), and task
        completion (the members are released, so ``complete`` drops to False on the
        final tick although the whole task reached progress 1.0).
        """
        lost = 0.0
        for change, following in zip(self.coverage, self.coverage[1:]):
            if not (change.complete and not following.complete):
                continue
            if following.stage_index != change.stage_index or not self.durations:
                continue
            if following.progress >= 1.0 - EPS:
                continue
            index = min(change.stage_index, len(self.durations) - 1)
            elapsed = change.progress * len(self.durations) - change.stage_index
            lost += max(0.0, min(1.0, elapsed)) * self.durations[index]
        return lost

    @property
    def completed_work(self) -> bool:
        """The whole task reached progress 1.0, i.e. every stage's work was delivered."""
        return self.final_progress >= 1.0 - EPS

    @property
    def finished_after_deadline(self) -> bool:
        """All the work was delivered, but the last tick of it landed past the deadline."""
        return self.completed_work and self.expired_at is not None

    @property
    def final_coverage_shortfall(self) -> float:
        return max(0.0, (1.0 - self.final_progress) * self.work_needed)

    @property
    def last_complete_coverage_start(self) -> float | None:
        intervals = self.coverage_intervals
        return intervals[-1][0] if intervals else None

    @property
    def earliest_belief_feasible_dispatch(self) -> float | None:
        """Latest letter among the robots the target needs; honest lower bound."""
        return max(self.first_belief_at.values()) if self.first_belief_at else None

    @property
    def earliest_feasible_completion(self) -> float | None:
        told = self.earliest_belief_feasible_dispatch
        return None if told is None else told + MAX_TRAVEL + self.work_needed

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "target": self.target,
            "detected_at": self.detected_at,
            "deadline": self.deadline,
            "stage_count": len(self.durations),
            "durations": list(self.durations),
            "requirements": [list(r) for r in self.requirements],
            "work_needed": round(self.work_needed, 3),
            "delivery_budget": round(self.delivery_budget, 3),
            "expired_at": self.expired_at,
            "final_progress": self.final_progress,
            "final_stage": self.final_stage,
            "final_status": self.final_status,
            "completed_work": self.completed_work,
            "finished_after_deadline": self.finished_after_deadline,
            "first_belief_at": self.first_belief_at,
            "belief_revisions": {k: v for k, v in self.belief_revisions.items()},
            "belief_spread": self.belief_spread,
            "first_reservation_at": self.first_reservation_at,
            "reservation_spread": self.reservation_spread,
            "reservation_ids": self.reservation_ids,
            "assignment_sent_at": self.assignment_sent_at,
            "assignment_spread": self.assignment_spread,
            "departed_at": self.departed_at,
            "arrived_at": self.arrived_at,
            "arrival_spread": self.arrival_spread,
            "left_at": self.left_at,
            "on_site": self.on_site,
            "first_complete_at": self.first_complete_at,
            "detection_to_first_complete": self.detection_to_first_complete,
            "coverage_intervals": self.coverage_intervals,
            "longest_contiguous_coverage": self.longest_contiguous_coverage,
            "total_coverage_time": self.total_coverage_time,
            "coverage_gap_count": self.coverage_gap_count,
            "coverage_reset_count": self.coverage_reset_count,
            "coverage_work_lost": round(self.coverage_work_lost, 3),
            "final_coverage_shortfall": round(self.final_coverage_shortfall, 3),
            "last_complete_coverage_start": self.last_complete_coverage_start,
            "earliest_belief_feasible_dispatch": self.earliest_belief_feasible_dispatch,
            "earliest_feasible_completion": self.earliest_feasible_completion,
            "capable_agents_total": self.capable_total,
            "capable_agents_busy_at_detection": self.capable_busy_at_detection,
            "concurrent_active_tasks_at_detection": self.concurrent_active_at_detection,
            "detection_to_first_needs_decision": self.detection_to_first_needs_decision,
            "capability_waiting_steps": round(self.capability_waiting_steps, 3),
            "stage_windows": {
                str(index): {
                    "duration": window.duration,
                    "requirements": list(window.requirements),
                    "first_complete_at": window.first_complete_at,
                    "last_complete_at": window.last_complete_at,
                    "completed_at": window.completed_at,
                    "resets": window.resets,
                }
                for index, window in self.stages.items()
            },
            "coverage_changes": [vars(c) for c in self.coverage],
        }


def _finished_before(records: list[dict], target: str, moment: float) -> bool:
    for row in records:
        if str(row.get("target_id")) != target:
            continue
        if row.get("event_type") == "final" and float(row["timestamp"]) <= moment:
            return True
    return False


def build_timeline(seed: int, target: str) -> TargetTimeline | None:
    records = load_records(seed)
    decisions = load_decisions(seed)
    payload = load_deliveries(seed)

    rows = [r for r in records if str(r.get("target_id")) == target]
    detection = next((r for r in rows if r.get("event_type") == "detection"), None)
    if detection is None:
        return None
    detail = detection.get("detail") or {}
    grid = detail.get("grid_position")
    position = GridPosition(int(grid[0]), int(grid[1])) if grid else None
    detected_at = float(detection["timestamp"])
    plan = stage_plan_for(target, position, detected_at) if position else None
    durations = plan.durations if plan else ()
    requirements = plan.requirements if plan else ()

    timeline = TargetTimeline(
        seed=seed,
        target=target,
        detected_at=detected_at,
        deadline=detected_at + float(detail.get("deadline_offset", 0.0)),
        position=position,
        durations=durations,
        requirements=requirements,
    )

    for row in payload["messages"]:
        if not row["accepted"] or row["message"]["target_id"] != target:
            continue
        agent, at = str(row["agent_id"]), float(row["at"])
        if agent not in timeline.first_belief_at or at < timeline.first_belief_at[agent]:
            timeline.first_belief_at[agent] = at
        timeline.belief_revisions.setdefault(agent, []).append(
            (at, int(row["message"]["revision"]))
        )

    # The recorder timestamps a delivery from inside ``deliver_detection``, which the
    # executor calls before it advances the world, so every letter reads one tick earlier
    # than the lifecycle's detection row. Measured across all five seeds the offset is a
    # constant -1.0 for every accepted delivery. Rebase onto the lifecycle clock so a
    # belief_spread of 0 means "told on the detection tick" rather than "-1".
    if timeline.first_belief_at:
        offset = statistics.median(
            value - detected_at for value in timeline.first_belief_at.values()
        )
        if abs(offset) > EPS:
            timeline.first_belief_at = {
                agent: value - offset for agent, value in timeline.first_belief_at.items()
            }
            timeline.belief_revisions = {
                agent: [(at - offset, revision) for at, revision in values]
                for agent, values in timeline.belief_revisions.items()
            }

    for row in decisions:
        agent, at = str(row["agent_id"]), float(row["timestamp"])
        for reservation in row.get("advice_reservations") or []:
            if str(reservation) == target:
                timeline.first_reservation_at.setdefault(agent, at)
        rid = row.get("reservation_id")
        if rid is not None and row.get("commitment_task") == target:
            timeline.reservation_ids.setdefault(agent, []).append(str(rid))
        if row.get("chosen_task") == target:
            timeline.assignment_sent_at.setdefault(agent, at)
        if row.get("current_task") == target and row.get("current_kind") == "task":
            timeline.departed_at.setdefault(agent, at)
        ids = row.get("action_task_ids")
        mask = row.get("mask")
        if ids and mask and target in ids and mask[list(ids).index(target)]:
            timeline.hard_masked_agents.add(agent)

    progress = sorted(
        (r for r in rows if r.get("event_type") == "progress"),
        key=lambda r: float(r["timestamp"]),
    )
    previous: dict[int, float] = {}
    seen: set[str] = set()
    for row in progress:
        at = float(row["timestamp"])
        stage_index = int(row.get("stage_index") or 0)
        fraction = float(row.get("progress") or 0.0)
        members = tuple(str(m) for m in (row.get("assigned_agents") or ()))
        remaining = tuple(int(v) for v in (row.get("remaining_requirements") or ()))
        timeline.progress_rows.append((at, stage_index, fraction, members, remaining))

        for agent in members:
            timeline.arrived_at.setdefault(agent, at)
            seen.add(agent)
        for agent in seen - set(members):
            timeline.left_at.setdefault(agent, at)

        window = timeline.stages.setdefault(
            stage_index,
            StageWindow(
                index=stage_index,
                duration=durations[stage_index] if stage_index < len(durations) else None,
                requirements=requirements[stage_index] if stage_index < len(requirements) else (),
            ),
        )
        if stage_index in previous and fraction < previous[stage_index] - EPS:
            window.resets.append(at)
        previous[stage_index] = max(previous.get(stage_index, 0.0), fraction)
        for index in list(timeline.stages):
            if index < stage_index and timeline.stages[index].completed_at is None:
                timeline.stages[index].completed_at = at

        complete = bool(members) and bool(remaining) and all(v <= 0 for v in remaining)
        missing = remaining
        covered = tuple(
            int(req) - int(rem) for req, rem in zip(window.requirements, remaining)
        )
        committed = tuple(
            sorted(a for a, sent in timeline.assignment_sent_at.items() if sent <= at and a not in members)
        )
        rids = tuple(
            str(values[-1]) for values in timeline.reservation_ids.values() if values
        )
        if not timeline.coverage or timeline.coverage[-1].complete != complete:
            timeline.coverage.append(
                CoverageChange(
                    at=at,
                    complete=complete,
                    members=members,
                    covered=covered,
                    missing=missing,
                    stage_index=stage_index,
                    progress=fraction,
                    committed=committed,
                    reservation_ids=rids,
                    hard_masked=tuple(sorted(timeline.hard_masked_agents)),
                    belief_stale=tuple(
                        sorted(a for a in members if timeline.first_belief_at.get(a, at) > at)
                    ),
                )
            )
        if complete:
            window.first_complete_at = window.first_complete_at or at
            window.last_complete_at = at

    for row in rows:
        if row.get("event_type") == "final":
            if row.get("failure_reason") == "deadline_expired":
                timeline.expired_at = float(row["timestamp"])
            timeline.final_status = row.get("status")
            timeline.final_progress = float(row.get("progress") or 0.0)
            timeline.final_stage = int(row.get("stage_index") or 0)

    for agent, first in timeline.arrived_at.items():
        end = timeline.left_at.get(agent)
        if end is None:
            end = timeline.expired_at or (
                timeline.progress_rows[-1][0] if timeline.progress_rows else first
            )
        timeline.on_site.setdefault(agent, []).append((first, end))

    waiting = 0.0
    for row, following in zip(progress, progress[1:]):
        members = row.get("assigned_agents") or ()
        remaining = row.get("remaining_requirements") or ()
        if members and any(int(v) > 0 for v in remaining):
            waiting += max(0.0, float(following["timestamp"]) - float(row["timestamp"]))
    timeline.capability_waiting_steps = waiting

    active = {
        str(r["target_id"])
        for r in records
        if r.get("event_type") == "detection"
        and float(r["timestamp"]) <= detected_at
        and not _finished_before(records, str(r["target_id"]), detected_at)
    }
    timeline.concurrent_active_at_detection = len(active)

    first_req = requirements[0] if requirements else ()
    capable = {
        str(row["agent_id"])
        for row in decisions
        if row.get("abilities")
        and any(int(a) > 0 and int(q) > 0 for a, q in zip(row["abilities"], first_req))
    }
    busy = {
        str(row["agent_id"])
        for row in decisions
        if float(row["timestamp"]) <= detected_at and row.get("current_task")
    }
    timeline.capable_total = len(capable)
    timeline.capable_busy_at_detection = len(capable & busy)

    for row in decisions:
        if row.get("needs_decision") and float(row["timestamp"]) >= detected_at:
            timeline.detection_to_first_needs_decision = float(row["timestamp"]) - detected_at
            break

    return timeline


def build_all(seeds: list[int]) -> dict[str, TargetTimeline]:
    timelines: dict[str, TargetTimeline] = {}
    for seed in seeds:
        records = load_records(seed)
        targets = sorted(
            str(r["target_id"])
            for r in records
            if r.get("event_type") == "detection" and str(r.get("target_id")) != "__run__"
        )
        for target in set(targets):
            timeline = build_timeline(seed, target)
            if timeline is not None:
                timelines[f"{seed}/{target}"] = timeline
    return timelines


def write_outputs() -> dict:
    timelines = build_all(FAILURE_SEEDS)
    summary = {
        "arm": ARM,
        "seeds": FAILURE_SEEDS,
        "failure_targets": [f"{s}/{t}" for s, t in FAILURES],
        "targets": [t.as_dict() for t in timelines.values()],
    }
    (HERE / "stage_timelines.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    summary = write_outputs()
    for row in summary["targets"]:
        print(
            f"{row['seed']}/{row['target']}: "
            f"belief_spread={row['belief_spread']} "
            f"arrival_spread={row['arrival_spread']} "
            f"det_to_cover={row['detection_to_first_complete']} "
            f"longest={row['longest_contiguous_coverage']} "
            f"resets={row['coverage_reset_count']} "
            f"lost={row['coverage_work_lost']} "
            f"shortfall={row['final_coverage_shortfall']} "
            f"ebfd={row['earliest_belief_feasible_dispatch']} "
            f"efc={row['earliest_feasible_completion']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
