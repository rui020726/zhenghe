"""Windowed metrics over joint lifecycle JSONL logs.

The lifecycle log is the ground truth: one line per accepted event, so a run can
be re-measured at any observation window without rerunning it. Because the joint
system is causal (nothing at step t depends on step t+1), the metrics of a
500-step run equal the metrics of the first 500 steps of a 1000-step run, and
one long replay per arm serves every window.

Definitions (per window W, over the targets discovered at or before W):

* ``completed``   -- a progress event with status ``completed`` at t <= W;
* ``expired``     -- not completed by W and ``detected_at + deadline_offset <= W``;
* ``unassigned``  -- neither of the above and no assignment command at t <= W;
* ``active``      -- everything else (assigned, unfinished, inside its deadline);

the four categories partition the discovered set, so ``active`` is never dropped
from the denominator:

* ``completion_rate = completed / discovered``;
* ``deadline_rate`` is reported as a micro average ``on_time / completed``
  (a run that completed nothing contributes 0 to both sums, not a fake 0.0
  rate that would drag a macro average down).

Run as a module for a CLI report over ``rp_<arm>_s<seed>.jsonl`` files.
"""

from __future__ import annotations

import argparse
import glob as globmodule
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

FILE_PATTERN = re.compile(r"^(?:rp|ab)_(?P<arm>.+)_s(?P<seed>\d+)\.jsonl$")


@dataclass(frozen=True, slots=True)
class WindowMetrics:
    window: int
    discovered: int
    completed: int
    expired: int
    active: int
    unassigned: int
    on_time: int
    assignment_commands: int
    rejected_assignments: int
    mean_completion_t: float | None
    log_complete: bool

    @property
    def completion_rate(self) -> float | None:
        if self.discovered == 0:
            return None
        return self.completed / self.discovered

    @property
    def deadline_rate(self) -> float | None:
        if self.completed == 0:
            return None
        return self.on_time / self.completed

    def as_row(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "discovered": self.discovered,
            "completed": self.completed,
            "expired": self.expired,
            "active": self.active,
            "unassigned": self.unassigned,
            "on_time": self.on_time,
            "completion_rate": self.completion_rate,
            "deadline_rate": self.deadline_rate,
            "mean_completion_t": self.mean_completion_t,
            "assignment_commands": self.assignment_commands,
            "rejected_assignments": self.rejected_assignments,
            "log_complete": self.log_complete,
        }


def load_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def window_metrics(records: Sequence[dict[str, Any]], window: float) -> WindowMetrics:
    discovered: dict[str, tuple[float, float]] = {}
    completed_at: dict[str, float] = {}
    first_assignment: dict[str, float] = {}
    assignments = 0
    rejections = 0
    log_complete = False

    for record in records:
        event_type = record.get("event_type")
        timestamp = float(record.get("timestamp") or 0.0)
        target = str(record.get("target_id"))
        if event_type == "detection" and timestamp <= window:
            detail = record.get("detail") or {}
            discovered[target] = (
                timestamp,
                timestamp + float(detail.get("deadline_offset", 0.0)),
            )
        elif event_type == "assignment" and timestamp <= window:
            assignments += 1
            if target not in first_assignment or timestamp < first_assignment[target]:
                first_assignment[target] = timestamp
        elif event_type == "assignment_rejected" and timestamp <= window:
            rejections += 1
        elif (
            event_type == "progress"
            and record.get("status") == "completed"
            and timestamp <= window
        ):
            completed_at.setdefault(target, timestamp)
        elif event_type == "final" and target == "__run__":
            log_complete = True

    completed = expired = unassigned = 0
    on_time = 0
    durations: list[float] = []
    for target, (detected_at, deadline) in discovered.items():
        done = completed_at.get(target)
        if done is not None:
            completed += 1
            durations.append(done - detected_at)
            if done <= deadline:
                on_time += 1
        elif deadline <= window:
            expired += 1
        elif target not in first_assignment:
            unassigned += 1
    active = len(discovered) - completed - expired - unassigned

    return WindowMetrics(
        window=int(window),
        discovered=len(discovered),
        completed=completed,
        expired=expired,
        active=active,
        unassigned=unassigned,
        on_time=on_time,
        assignment_commands=assignments,
        rejected_assignments=rejections,
        mean_completion_t=round(statistics.mean(durations), 3) if durations else None,
        log_complete=log_complete,
    )


def parse_run_path(path: str | Path) -> tuple[str, int]:
    match = FILE_PATTERN.match(Path(path).name)
    if match is None:
        raise ValueError(f"not an rp_<arm>_s<seed>.jsonl lifecycle log: {path}")
    return match.group("arm"), int(match.group("seed"))


def collect(
    paths: Iterable[str | Path], windows: Sequence[int]
) -> dict[str, dict[int, dict[int, WindowMetrics]]]:
    """arm -> window -> seed -> metrics."""
    table: dict[str, dict[int, dict[int, WindowMetrics]]] = defaultdict(
        lambda: {int(w): {} for w in windows}
    )
    for path in paths:
        arm, seed = parse_run_path(path)
        records = load_records(path)
        for window in windows:
            table[arm][int(window)][seed] = window_metrics(records, window)
    return dict(table)


def _ci95(values: Sequence[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0
    sem = statistics.stdev(values) / math.sqrt(len(values))
    return mean, 1.96 * sem


def _mean_over(runs: Sequence[WindowMetrics], field: str) -> tuple[float, float]:
    values = [value for value in (getattr(run, field) for run in runs) if value is not None]
    if not values:
        return float("nan"), 0.0
    return _ci95(values)


def report(
    table: dict[str, dict[int, dict[int, WindowMetrics]]],
    windows: Sequence[int],
    baseline: str,
) -> str:
    arms = sorted(table, key=lambda a: (a != baseline, a))
    lines: list[str] = []
    for window in windows:
        seeds = (
            set.intersection(*(set(table[a][int(window)]) for a in arms)) if arms else set()
        )
        lines.append(f"== window {window} | paired seeds: {len(seeds)} ==")
        header = (
            f"{'arm':<10}{'disc':>6}{'compl':>6}{'exp':>5}{'act':>5}{'unas':>5}"
            f"{'completion':>16}{'deadline':>10}{'mean_t':>14}{'asg':>8}{'rej':>6}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for arm in arms:
            runs = [table[arm][int(window)][s] for s in sorted(seeds)]
            disc = sum(r.discovered for r in runs)
            comp = sum(r.completed for r in runs)
            ontime = sum(r.on_time for r in runs)
            completion, completion_ci = _mean_over(runs, "completion_rate")
            mean_t, mean_t_ci = _mean_over(runs, "mean_completion_t")
            deadline_micro = ontime / comp if comp else float("nan")
            lines.append(
                f"{arm:<10}{disc:>6}{comp:>6}"
                f"{sum(r.expired for r in runs):>5}{sum(r.active for r in runs):>5}"
                f"{sum(r.unassigned for r in runs):>5}"
                f"{completion:>10.3f}±{completion_ci:<5.3f}"
                f"{deadline_micro:>10.3f}"
                f"{mean_t:>8.1f}±{mean_t_ci:<5.1f}"
                f"{sum(r.assignment_commands for r in runs):>8}"
                f"{sum(r.rejected_assignments for r in runs):>6}"
            )
        lines.append(f"{'delta vs ' + baseline:<10}{'':>27}{'completion':>16}{'mean_t':>14}{'asg(sum)':>15}")
        for arm in arms:
            if arm == baseline:
                continue
            base_runs = {s: table[baseline][int(window)][s] for s in seeds}
            arm_runs = {s: table[arm][int(window)][s] for s in seeds}
            diffs = [
                arm_runs[s].completion_rate - base_runs[s].completion_rate
                for s in seeds
                if arm_runs[s].completion_rate is not None
                and base_runs[s].completion_rate is not None
            ]
            tdiffs = [
                arm_runs[s].mean_completion_t - base_runs[s].mean_completion_t
                for s in seeds
                if arm_runs[s].mean_completion_t is not None
                and base_runs[s].mean_completion_t is not None
            ]
            adiff = sum(
                arm_runs[s].assignment_commands - base_runs[s].assignment_commands
                for s in seeds
            )
            delta, delta_ci = _ci95(diffs) if diffs else (float("nan"), 0.0)
            tmean, tci = _ci95(tdiffs) if tdiffs else (float("nan"), 0.0)
            lines.append(
                f"{arm:<10}{'':>27}{delta:>10.3f}±{delta_ci:<5.3f}"
                f"{tmean:>8.1f}±{tci:<5.1f}{adiff:>15}"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="integrated_sar.lifecycle_metrics",
        description="Windowed, correctly-normalised metrics over joint lifecycle logs.",
    )
    parser.add_argument("patterns", nargs="+", help="lifecycle log globs, e.g. rp_*_s*.jsonl")
    parser.add_argument("--windows", type=int, nargs="+", default=[500, 750, 1000])
    parser.add_argument("--baseline", default="nocoord")
    args = parser.parse_args(argv)

    paths: list[str] = []
    for pattern in args.patterns:
        paths.extend(sorted(globmodule.glob(pattern)))
    if not paths:
        print("no lifecycle logs matched", args.patterns)
        return 1
    table = collect(paths, args.windows)
    print(f"{len(paths)} logs, arms: {', '.join(sorted(table))}")
    print(report(table, args.windows, args.baseline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
