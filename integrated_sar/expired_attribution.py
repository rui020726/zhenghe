"""Why a joint task expired, and whether *any* controller could have saved it.

The joint ablation left one failure mode: tasks whose deadline passes before the
executors finish. Training a learned coordinator against those tasks is only worth
it if a meaningful share of them was savable -- a task that was already infeasible
at detection time can only be fixed by changing deadlines, task generation or the
robot roster, and no amount of allocation skill will move it.

Everything needed for that verdict is already in the lifecycle log, plus one
reconstruction that is exact rather than inferred:

* the target profile is a pure function of the target's grid position
  (:class:`integrated_sar.profiles.DeterministicTargetProfileProvider`), so the
  stage count, the per-stage ability requirements and the work durations are
  recomputed bit-for-bit from the ``grid_position`` in the detection record;
* the deadline offset comes from the same profile, so ``deadline = detected_at +
  deadline_offset`` needs no assumption;
* the robots move at ``velocity = 0.2`` normalized units per second in a unit
  square, so *no* straight-line trip can exceed ``sqrt(2) / 0.2`` ~= 7.1 steps.
  That bound is what makes the oracle argument airtight: a target whose stage
  work fits inside its deadline minus that worst-case travel was reachable by a
  perfect dispatcher, whatever the allocator did.

The remaining facts are read straight off the log: first assignment, the first
step at which work actually advanced, how many steps of work were delivered, and
where progress stalled. From those the classifier separates

``finished_after_deadline``
    the work was completed, just late (progress reached 1.0);
``infeasible_at_discovery``
    stage work alone (or work plus worst-case travel) exceeds the whole budget;
``never_assigned`` / ``assigned_never_worked``
    nobody was ever sent, or somebody was sent and never produced a step of work;
``capability_blocked``
    members were on site but the stage requirements stayed unmet, so the robots
    present could not do the work;
``stopped_mid_work``
    work started, then stopped well before the deadline;
``ran_out_of_time``
    work was still advancing when the deadline arrived.

Belief staleness is deliberately not estimated here: these runs published with
``comm_delivery_probability = 1.0`` and ``comm_delay_ticks = 0``, which the
run-config record states, so a delivery delay of zero is a fact of the batch
rather than something to infer. :func:`communication_facts` reads it back.

Run as a module for a CLI report over ``rp_<arm>_s<seed>.jsonl`` logs.
"""

from __future__ import annotations

import argparse
import glob as globmodule
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .contracts import GridPosition
from .lifecycle_metrics import FILE_PATTERN, load_records
from .profiles import DeterministicTargetProfileProvider

# ``old_task_env`` gives every rescue robot this speed and moves it by
# ``velocity * dt`` per step; the scene is the unit square.
ROBOT_VELOCITY = 0.2
MAX_TRAVEL_STEPS = math.sqrt(2.0) / ROBOT_VELOCITY
# A stage that shows no advance for this long while members are on site is
# blocked (wrong abilities) rather than merely slow: one stage lasts 3-12 s.
STALL_STEPS = 15.0
PROGRESS_EPS = 1e-9

CLASSES = (
    "finished_after_deadline",
    "infeasible_at_discovery",
    "never_assigned",
    "assigned_never_worked",
    "capability_blocked",
    "stopped_mid_work",
    "ran_out_of_time",
)


@dataclass(frozen=True, slots=True)
class StagePlan:
    """The reconstructed hidden profile of one target."""

    durations: tuple[float, ...]
    requirements: tuple[tuple[int, ...], ...]
    deadline_offset: float
    severity: float

    @property
    def total_work(self) -> float:
        return float(sum(self.durations))

    @property
    def final_requirements(self) -> tuple[int, ...]:
        return self.requirements[-1] if self.requirements else ()


def stage_plan_for(target_id: str, position: GridPosition, detected_at: float) -> StagePlan:
    """Rebuild the exact hidden profile from the target's physical cell."""
    profile = DeterministicTargetProfileProvider().profile_for(target_id, position, detected_at)
    return StagePlan(
        durations=tuple(float(stage.duration) for stage in profile.stages),
        requirements=tuple(tuple(int(v) for v in stage.requirements) for stage in profile.stages),
        deadline_offset=float(profile.deadline_offset),
        severity=float(profile.severity),
    )


@dataclass(frozen=True, slots=True)
class ProgressPoint:
    timestamp: float
    stage_index: int
    progress: float
    members: tuple[str, ...]
    status: str
    expired: bool
    remaining: tuple[int, ...] = ()


@dataclass(slots=True)
class TargetTimeline:
    """Everything one target did, assembled from its lifecycle records."""

    target_id: str
    arm: str
    seed: int
    detected_at: float
    plan: StagePlan
    grid_position: GridPosition | None = None
    assignments: list[tuple[float, str]] = field(default_factory=list)
    progress: list[ProgressPoint] = field(default_factory=list)
    final_status: str | None = None
    final_progress: float | None = None
    final_failure_reason: str | None = None
    rejected: int = 0

    # ---- derived facts -------------------------------------------------
    @property
    def deadline(self) -> float:
        return self.detected_at + self.plan.deadline_offset

    @property
    def work_needed(self) -> float:
        return self.plan.total_work

    @property
    def first_assignment(self) -> float | None:
        return min((ts for ts, _ in self.assignments), default=None)

    @property
    def assignment_latency(self) -> float | None:
        first = self.first_assignment
        return None if first is None else first - self.detected_at

    @property
    def assigned_agents(self) -> tuple[str, ...]:
        return tuple(sorted({agent for _, agent in self.assignments}))

    @property
    def advancing(self) -> list[ProgressPoint]:
        """Progress points that moved the task forward, in order."""
        moved: list[ProgressPoint] = []
        best = 0.0
        for point in self.progress:
            if point.progress > best + PROGRESS_EPS:
                best = point.progress
                moved.append(point)
        return moved

    @property
    def first_work(self) -> float | None:
        moved = self.advancing
        return moved[0].timestamp if moved else None

    @property
    def last_work(self) -> float | None:
        moved = self.advancing
        return moved[-1].timestamp if moved else None

    @property
    def worked_steps(self) -> int:
        return len(self.advancing)

    @property
    def max_progress(self) -> float:
        return max((point.progress for point in self.progress), default=0.0)

    @property
    def progress_setbacks(self) -> int:
        """How often the reported progress went backwards.

        ``old_task_env.task_update`` refills ``time_left`` whenever the stage
        requirements are momentarily unmet, so a member leaving mid-stage throws
        the elapsed work away and the next snapshot reports a smaller number.
        Counting drops between consecutive records -- not drops below a running
        maximum -- keeps one reset from being counted again on the way back up.
        """
        return sum(
            1
            for previous, current in zip(self.progress, self.progress[1:])
            if current.progress < previous.progress - PROGRESS_EPS
        )

    @property
    def discarded_work_steps(self) -> float:
        """Robot-steps of work destroyed by those resets."""
        return sum(
            (previous.progress - current.progress) * self.work_needed
            for previous, current in zip(self.progress, self.progress[1:])
            if current.progress < previous.progress - PROGRESS_EPS
        )

    @property
    def dispatch_to_work(self) -> float | None:
        """Steps between the first dispatch and the first delivered work.

        Non-zero even when travel cannot explain it (a straight-line trip is at
        most ~7 steps): the robot was re-tasked, rejected on arrival, or the task
        sat memberless while commands kept being issued.
        """
        first, work = self.first_assignment, self.first_work
        if first is None or work is None:
            return None
        return work - first

    @property
    def delivery_latency(self) -> float | None:
        """Detection to first delivered step of work -- travel plus queueing."""
        first = self.first_work
        return None if first is None else first - self.detected_at

    @property
    def measured_stage_durations(self) -> dict[int, float]:
        """Stage durations as measured from the progress slope.

        ``progress = (stage_index + elapsed/duration) / total_stages``, so one
        advancing step adds ``1 / (duration * total_stages)``. Cross-checking the
        slope against the reconstructed profile catches a wrong reconstruction.
        """
        total = len(self.plan.durations)
        if total == 0:
            return {}
        per_stage: dict[int, list[float]] = defaultdict(list)
        previous: ProgressPoint | None = None
        for point in self.advancing:
            if previous is not None and point.stage_index == previous.stage_index:
                delta = point.progress - previous.progress
                step = point.timestamp - previous.timestamp
                if delta > PROGRESS_EPS and step > 0:
                    per_stage[point.stage_index].append(step / (delta * total))
            previous = point
        return {
            index: statistics.mean(values) for index, values in per_stage.items() if values
        }

    @property
    def members_waiting_steps(self) -> float:
        """Longest run of steps with members on site but no progress."""
        longest = 0.0
        run_start: float | None = None
        previous = 0.0
        for point in self.progress:
            if point.members and point.progress <= previous + PROGRESS_EPS:
                run_start = point.timestamp if run_start is None else run_start
                longest = max(longest, point.timestamp - run_start)
            else:
                run_start = None
            previous = max(previous, point.progress)
        return longest

    @property
    def short_group_idle_steps(self) -> float:
        """Steps with members on site while the stage requirement was still unmet.

        ``old_task_env`` does not even start the stage clock while a stage's
        requirements are unmet, so a group standing on a target with the wrong
        ability set produces no work at all: it is not slow, it is idle. The
        lifecycle log only writes a progress row when the reported state changes,
        so the duration is integrated between consecutive rows -- counting rows
        would collapse a long stall that writes no row into nothing.
        """
        idle = 0.0
        for point, following in zip(self.progress, self.progress[1:]):
            if point.members and any(value > 0 for value in point.remaining):
                idle += max(0.0, following.timestamp - point.timestamp)
        return idle

    @property
    def latest_start(self) -> float:
        """Latest first-work time that still finishes inside the deadline."""
        return self.deadline - self.work_needed

    @property
    def delivery_budget(self) -> float:
        """Steps a perfect dispatcher could spend getting a robot on site."""
        return self.deadline - self.detected_at - self.work_needed

    @property
    def idle_steps_after_start(self) -> float:
        """Steps between first and last delivered work that produced nothing."""
        first, last = self.first_work, self.last_work
        if first is None or last is None:
            return 0.0
        return max(0.0, (last - first + 1.0) - self.worked_steps)

    @property
    def leftover_work(self) -> float:
        if self.final_progress is None:
            return self.work_needed
        return max(0.0, (1.0 - float(self.final_progress)) * self.work_needed)

    def travel_floor(self) -> float:
        """Worst-case straight-line trip: the strongest feasibility bound we have."""
        return MAX_TRAVEL_STEPS

    def as_row(self) -> dict[str, Any]:
        return {
            "target": self.target_id,
            "arm": self.arm,
            "seed": self.seed,
            "detected_at": self.detected_at,
            "deadline_offset": round(self.plan.deadline_offset, 2),
            "work_needed": round(self.work_needed, 2),
            "stages": len(self.plan.durations),
            "delivery_budget": round(self.delivery_budget, 2),
            "first_assignment_at": self.first_assignment,
            "assignment_latency": self.assignment_latency,
            "first_work_at": self.first_work,
            "delivery_latency": self.delivery_latency,
            "dispatch_to_work": self.dispatch_to_work,
            "worked_steps": self.worked_steps,
            "progress_setbacks": self.progress_setbacks,
            "discarded_work_steps": round(self.discarded_work_steps, 2),
            "max_progress": self.max_progress,
            "idle_steps_after_start": round(self.idle_steps_after_start, 1),
            "members_waiting_steps": round(self.members_waiting_steps, 1),
            "short_group_idle_steps": round(self.short_group_idle_steps, 1),
            "assignments": len(self.assignments),
            "distinct_agents": len(self.assigned_agents),
            "final_progress": self.final_progress,
            "final_status": self.final_status,
        }


def build_timelines(
    records: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    seed: int,
    plan_provider: Callable[[str, GridPosition, float], StagePlan] = stage_plan_for,
) -> dict[str, TargetTimeline]:
    """Assemble one timeline per target from a single run's lifecycle records."""
    timelines: dict[str, TargetTimeline] = {}

    for record in records:
        target = str(record.get("target_id"))
        if target == "__run__":
            continue
        event_type = record.get("event_type")
        timestamp = float(record.get("timestamp") or 0.0)
        detail = record.get("detail") or {}
        if event_type == "detection":
            grid = detail.get("grid_position")
            if grid is None or len(grid) < 2:
                continue
            position = GridPosition(row=int(grid[0]), col=int(grid[1]))
            # One timeline per target, created by the detection that opens it: every
            # later record for this target hangs off it.
            timelines.setdefault(
                target,
                TargetTimeline(
                    target_id=target,
                    arm=arm,
                    seed=seed,
                    detected_at=timestamp,
                    plan=plan_provider(target, position, timestamp),
                    grid_position=position,
                ),
            )
            continue
        timeline = timelines.get(target)
        if timeline is None:
            continue
        if event_type == "assignment":
            agent = (record.get("assigned_agents") or [""])[0]
            timeline.assignments.append((timestamp, str(agent)))
        elif event_type == "assignment_rejected":
            timeline.rejected += 1
        elif event_type == "progress":
            timeline.progress.append(
                ProgressPoint(
                    timestamp=timestamp,
                    stage_index=int(record.get("stage_index") or 0),
                    progress=float(record.get("progress") or 0.0),
                    members=tuple(record.get("assigned_agents") or ()),
                    status=str(record.get("status") or ""),
                    expired=bool(detail.get("expired", False)),
                    remaining=tuple(int(v) for v in (record.get("remaining_requirements") or ())),
                )
            )
        elif event_type == "final":
            timeline.final_status = record.get("status")
            timeline.final_progress = record.get("progress")
            timeline.final_failure_reason = record.get("failure_reason")

    for timeline in timelines.values():
        timeline.progress.sort(key=lambda point: point.timestamp)
        timeline.assignments.sort()
    return timelines


def classify(timeline: TargetTimeline) -> str:
    """Name the single dominant reason this target is not counted as completed."""
    if timeline.final_progress is not None and float(timeline.final_progress) >= 1.0 - PROGRESS_EPS:
        return "finished_after_deadline"
    if timeline.plan.deadline_offset < timeline.work_needed:
        return "infeasible_at_discovery"
    if timeline.first_assignment is None:
        return "never_assigned"
    if timeline.first_work is None:
        if timeline.members_waiting_steps >= STALL_STEPS:
            return "capability_blocked"
        return "assigned_never_worked"
    if timeline.members_waiting_steps >= STALL_STEPS:
        return "capability_blocked"
    if timeline.last_work is not None and timeline.last_work < timeline.deadline - STALL_STEPS:
        return "stopped_mid_work"
    return "ran_out_of_time"


@dataclass(frozen=True, slots=True)
class Attribution:
    """Aggregated verdict for one arm over the targets that expired by a window."""

    arm: str
    window: float
    seeds: tuple[int, ...]
    expired: tuple[TargetTimeline, ...]
    completed: tuple[TargetTimeline, ...]
    counts: Mapping[str, int]

    @property
    def expired_total(self) -> int:
        return len(self.expired)

    @property
    def provably_infeasible(self) -> int:
        """Offset shorter than the stage work: no controller could finish in time."""
        return sum(
            1 for t in self.expired if t.plan.deadline_offset < t.work_needed
        )

    @property
    def infeasible_with_travel(self) -> int:
        return sum(
            1
            for t in self.expired
            if t.plan.deadline_offset < t.work_needed + t.travel_floor()
        )

    def oracle_completion(self, delivery_latency: float) -> float:
        """Share of expired targets a dispatcher with this delivery latency saves.

        A target is saved when a robot is on site ``delivery_latency`` steps after
        detection and the work then runs without interruption: the only
        requirement is ``detected_at + latency + work_needed <= deadline``.
        """
        if not self.expired:
            return float("nan")
        saved = sum(
            1
            for t in self.expired
            if t.plan.deadline_offset >= t.work_needed + delivery_latency
        )
        return saved / len(self.expired)

    def oracle_budget_curve(self, latencies: Sequence[float]) -> dict[float, float]:
        return {latency: self.oracle_completion(latency) for latency in latencies}

    def latency_stats(self) -> dict[str, dict[str, float]]:
        """Median/mean delivery latency for expired versus completed targets."""
        out: dict[str, dict[str, float]] = {}
        for label, group in (("expired", self.expired), ("completed", self.completed)):
            values = [t.delivery_latency for t in group if t.delivery_latency is not None]
            if not values:
                out[label] = {}
                continue
            out[label] = {
                "n": float(len(values)),
                "median": statistics.median(values),
                "mean": statistics.mean(values),
                "p90": sorted(values)[int(0.9 * (len(values) - 1))],
            }
            gaps = [t.dispatch_to_work for t in group if t.dispatch_to_work is not None]
            if gaps:
                out[label]["dispatch_to_work_median"] = statistics.median(gaps)
        return out

    def churn_stats(self) -> dict[str, dict[str, float]]:
        """Command churn and work thrown away, expired versus completed."""
        out: dict[str, dict[str, float]] = {}
        for label, group in (("expired", self.expired), ("completed", self.completed)):
            if not group:
                out[label] = {}
                continue
            out[label] = {
                "n": float(len(group)),
                "assignments_mean": statistics.mean(len(t.assignments) for t in group),
                "distinct_agents_mean": statistics.mean(len(t.assigned_agents) for t in group),
                "setback_share": sum(1 for t in group if t.progress_setbacks > 0) / len(group),
                "discarded_work_mean": statistics.mean(t.discarded_work_steps for t in group),
            }
        return out

    def stall_stats(
        self, thresholds: Sequence[float] = (10.0, 30.0)
    ) -> dict[str, dict[str, float]]:
        """Time spent on site unable to work, expired versus completed targets.

        A group that is on the target with an unmet stage requirement is not making
        slow progress -- it is making none, and the deadline keeps running.
        """
        out: dict[str, dict[str, float]] = {}
        for label, group in (("expired", self.expired), ("completed", self.completed)):
            if not group:
                out[label] = {}
                continue
            values = sorted(t.short_group_idle_steps for t in group)
            entry = {
                "n": float(len(values)),
                "median": statistics.median(values),
                "p90": values[int(0.9 * (len(values) - 1))],
                "max": values[-1],
            }
            for threshold in thresholds:
                entry[f"share_ge_{threshold:g}"] = (
                    sum(1 for value in values if value >= threshold) / len(values)
                )
            out[label] = entry
        return out


def attribute(
    paths: Iterable[str | Path],
    *,
    window: float,
    plan_provider: Callable[[str, GridPosition, float], StagePlan] = stage_plan_for,
) -> dict[str, Attribution]:
    """arm -> attribution over the targets that expired by ``window``."""
    per_arm: dict[str, list[TargetTimeline]] = defaultdict(list)
    seeds: dict[str, set[int]] = defaultdict(set)
    for path in paths:
        match = FILE_PATTERN.match(Path(path).name)
        if match is None:
            raise ValueError(f"not an rp_<arm>_s<seed>.jsonl lifecycle log: {path}")
        arm, seed = match.group("arm"), int(match.group("seed"))
        seeds[arm].add(seed)
        records = load_records(path)
        timelines = build_timelines(records, arm=arm, seed=seed, plan_provider=plan_provider)
        per_arm[arm].extend(timelines.values())

    out: dict[str, Attribution] = {}
    for arm, timelines in per_arm.items():
        expired = [
            t
            for t in timelines
            if not _completed_by(t, window) and t.deadline <= window
        ]
        completed = [t for t in timelines if _completed_by(t, window)]
        counts = Counter(classify(t) for t in expired)
        out[arm] = Attribution(
            arm=arm,
            window=window,
            seeds=tuple(sorted(seeds[arm])),
            expired=tuple(sorted(expired, key=lambda t: (t.seed, t.target_id))),
            completed=tuple(sorted(completed, key=lambda t: (t.seed, t.target_id))),
            counts={name: counts.get(name, 0) for name in CLASSES},
        )
    return out


def _completed_by(timeline: TargetTimeline, window: float) -> bool:
    for point in timeline.progress:
        if point.status == "completed" and point.timestamp <= window:
            return True
    return False


def communication_facts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Read back the batch's communication settings from the run-config record."""
    for record in records:
        if record.get("event_type") == "run_config":
            configuration = (record.get("detail") or {}).get("configuration") or {}
            return {
                "comm_delivery_probability": configuration.get("comm_delivery_probability"),
                "comm_delay_ticks": configuration.get("comm_delay_ticks"),
                "allocation_interval": configuration.get("allocation_interval"),
                "coordination_interval": configuration.get("coordination_interval"),
                "command_ttl": configuration.get("command_ttl"),
            }
    return {}


def report(attributions: Mapping[str, Attribution], latencies: Sequence[float]) -> str:
    lines: list[str] = []
    for arm in sorted(attributions):
        attribution = attributions[arm]
        lines.append(
            f"== {arm} | window {attribution.window:g} | seeds {len(attribution.seeds)} | "
            f"discovered {len(attribution.expired) + len(attribution.completed)} | "
            f"expired {attribution.expired_total} =="
        )
        lines.append("  cause                        n   share")
        for name in CLASSES:
            count = attribution.counts.get(name, 0)
            share = count / attribution.expired_total if attribution.expired_total else 0.0
            lines.append(f"  {name:<26}{count:>4}{share:>8.2f}")
        infeasible = attribution.provably_infeasible
        infeasible_travel = attribution.infeasible_with_travel
        lines.append(
            f"  infeasible at discovery: {infeasible} by stage work alone, "
            f"{infeasible_travel} once worst-case travel ({MAX_TRAVEL_STEPS:.1f} steps) is added"
        )
        lines.append(
            "  oracle completion of the expired set, given a robot on site after "
            "X steps:"
        )
        for latency in latencies:
            share = attribution.oracle_completion(latency)
            lines.append(f"    X={latency:>5.0f}  {share:>6.2f}")
        stats = attribution.latency_stats()
        for label in ("expired", "completed"):
            entry = stats.get(label) or {}
            if entry:
                lines.append(
                    f"  {label:<10} detection->first work: median {entry['median']:.1f} "
                    f"mean {entry['mean']:.1f} p90 {entry['p90']:.1f} (n={int(entry['n'])})"
                )
                if "dispatch_to_work_median" in entry:
                    lines.append(
                        f"  {label:<10} first dispatch->first work: median "
                        f"{entry['dispatch_to_work_median']:.1f} steps "
                        f"(a straight-line trip is at most {MAX_TRAVEL_STEPS:.1f})"
                    )
        for label, entry in attribution.churn_stats().items():
            if entry:
                lines.append(
                    f"  {label:<10} assignments/target {entry['assignments_mean']:.1f}, "
                    f"distinct robots {entry['distinct_agents_mean']:.1f}, "
                    f"share with a stage reset {entry['setback_share']:.2f}, "
                    f"work thrown away by resets {entry['discarded_work_mean']:.1f} steps "
                    f"(n={int(entry['n'])})"
                )
        for label, entry in attribution.stall_stats().items():
            if entry:
                lines.append(
                    f"  {label:<10} on site with the stage requirement unmet: "
                    f"median {entry['median']:.1f} p90 {entry['p90']:.1f} "
                    f"max {entry['max']:.1f} steps; "
                    f">=10 steps in {entry['share_ge_10']:.2f} of targets, "
                    f">=30 in {entry['share_ge_30']:.2f} (n={int(entry['n'])})"
                )
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="integrated_sar.expired_attribution",
        description="Attribute expired joint tasks and bound what a perfect dispatcher could save.",
    )
    parser.add_argument("patterns", nargs="+", help="lifecycle log globs, e.g. rp_*_s*.jsonl")
    parser.add_argument("--window", type=float, default=1000.0)
    parser.add_argument(
        "--latencies", type=float, nargs="+", default=[2, 5, 10, 20, 30, 45, 60],
        help="delivery latencies for the oracle curve",
    )
    parser.add_argument("--json", type=str, default=None, help="write the per-target rows here")
    args = parser.parse_args(argv)

    paths: list[str] = []
    for pattern in args.patterns:
        paths.extend(sorted(globmodule.glob(pattern)))
    if not paths:
        print("no lifecycle logs matched", args.patterns)
        return 1

    attributions = attribute(paths, window=args.window)
    print(report(attributions, args.latencies))

    facts = communication_facts(load_records(paths[0]))
    print(f"batch communication facts (from {Path(paths[0]).name}): {facts}")

    if args.json:
        payload = {
            "window": args.window,
            "communication": facts,
            "arms": {
                arm: {
                    "seeds": list(attribution.seeds),
                    "expired": [t.as_row() for t in attribution.expired],
                    "counts": dict(attribution.counts),
                    "provably_infeasible": attribution.provably_infeasible,
                    "infeasible_with_travel": attribution.infeasible_with_travel,
                    "oracle_curve": {
                        str(latency): attribution.oracle_completion(latency)
                        for latency in args.latencies
                    },
                    "churn": attribution.churn_stats(),
                    "stall": attribution.stall_stats(),
                    "latency": attribution.latency_stats(),
                }
                for arm, attribution in attributions.items()
            },
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"per-target attribution written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())