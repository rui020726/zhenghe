"""Reading a transport transcript: batches, information volume, and how stale a belief was.

Everything here is offline. The transcript and the lifecycle log are the only inputs, so no
number in this file can be an input to a decision, and the age/error figures are allowed to
use the registry's own timeline as ground truth (§4's separation of evaluation from features).

Definitions, stated once because a metric without a unit is a rumour:

* ``emission`` = one task view put on the air; ``packet`` = one emission to one subscribed
  robot; ``payload_bytes`` = the canonical-JSON length of the view, counted per packet.
  A batch is therefore ``emissions``, and the information volume is ``packets``/``bytes``:
  packing three targets into one step would cut the batch count without cutting either of
  those, which is why both are reported.
* ``state_time`` = the registry timestamp carried by a packet (``sent_at``): the world time
  the payload describes.
* information age for one (robot, task) at step ``t`` = ``t - state_time`` of the newest
  packet that robot has had accepted for that task. Undefined before the first accepted
  packet, which is reported separately as time-to-first-knowledge.
* ``belief error`` = the believed stage / remaining requirements differ from the registry's
  at that step, counted in robot-task-steps. Truth comes from the lifecycle log.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


TruthEntry = tuple[float, int, tuple[int, ...]]
"""One registry state of a task: (state time, stage index, remaining requirements)."""


def load_transcript(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def truth_timeline(records: Sequence[dict[str, Any]]) -> dict[str, list[TruthEntry]]:
    """Per target, the registry's ``(state_time, stage, remaining_requirements)`` history."""
    timeline: dict[str, list[TruthEntry]] = defaultdict(list)
    for record in records:
        target = str(record.get("target_id") or "")
        if target == "__run__":
            continue
        if record.get("event_type") not in {"detection", "progress"}:
            continue
        remaining = record.get("remaining_requirements")
        timeline[target].append(
            (
                float(record.get("timestamp") or 0.0),
                int(record.get("stage_index") or 0),
                tuple(int(value) for value in (remaining or ())),
            )
        )
    return {target: sorted(entries) for target, entries in timeline.items()}


def terminal_times(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    """When each task stopped being something a robot could act on.

    A belief about a closed task is not stale information, so age is only counted while the
    task is active; without this cut every condition ages identically after the last
    completion and the comparison measures nothing.
    """
    closed: dict[str, float] = {}
    for record in records:
        target = str(record.get("target_id") or "")
        if target == "__run__" or target in closed:
            continue
        if str(record.get("status") or "").upper() in {"COMPLETED", "EXPIRED", "FAILED", "LOST"}:
            closed[target] = float(record.get("timestamp") or 0.0)
    return closed


def _tr_at(entries: Sequence[TruthEntry], moment: float) -> tuple[int, tuple[int, ...]] | None:
    """The registry's view of a task at a world time: the newest entry at or before it."""
    position = bisect_right([entry[0] for entry in entries], moment)
    if position == 0:
        return None
    _, stage, remaining = entries[position - 1]
    return stage, remaining


def transport_accounting(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Batch and volume numbers, from the relay's own records."""
    sends = [record for record in records if record["kind"] == "send"]
    deliveries = [record for record in records if record["kind"] == "deliver"]
    per_step: dict[int, int] = defaultdict(int)
    for send in sends:
        per_step[int(send["step"])] += 1
    ages = [int(send["step"]) - int(send["ready_step"]) for send in sends]
    flights = [int(delivery["step"]) - int(delivery["sent_step"]) for delivery in deliveries]
    return {
        "emissions": len(sends),
        "first_emissions": sum(1 for send in sends if send["first"]),
        "refresh_emissions": sum(1 for send in sends if not send["first"]),
        "packets_sent": sum(int(send["packets"]) for send in sends),
        "packets_lost": sum(int(send["packets_lost"]) for send in sends),
        "packets_delivered": len(deliveries),
        "deliveries_accepted": sum(1 for delivery in deliveries if delivery["accepted"]),
        "deliveries_duplicate": sum(
            1 for delivery in deliveries if delivery.get("reason") == "delivery_duplicate"
        ),
        "deliveries_superseded": sum(
            1 for delivery in deliveries if delivery.get("reason") == "delivery_superseded"
        ),
        "payload_bytes_total": sum(int(send["bytes_total"]) for send in sends),
        "payload_bytes_mean": (
            round(sum(int(send["payload_bytes"]) for send in sends) / len(sends), 1)
            if sends
            else None
        ),
        "batch_size_mean": round(len(sends) / len(per_step), 3) if sends else 0.0,
        "batch_size_max": max(per_step.values(), default=0),
        "busy_steps": len(per_step),
        "emission_wait_steps_mean": round(sum(ages) / len(ages), 3) if ages else 0.0,
        "emission_wait_steps_max": max(ages, default=0),
        "flight_steps_mean": round(sum(flights) / len(flights), 3) if flights else 0.0,
        "targets_emitted": len({send["target"] for send in sends}),
    }


def belief_age_accounting(
    records: Sequence[dict[str, Any]],
    *,
    horizon: float,
    robots: Iterable[str],
    truth: dict[str, list[TruthEntry]] | None = None,
    active_until: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Information age and belief error, sampled once per step in world time.

    Ages use the clock reading carried by each record, not the relay's step counter: the
    relay ticks before the global clock advances, so the two differ by one step and a
    "step" age would be a half-truth.
    """
    closed_skipped = 0
    sends = {(int(record["step"]), str(record["target"])): record for record in records if record["kind"] == "send"}
    accepted: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for delivery in (record for record in records if record["kind"] == "deliver"):
        if delivery["accepted"]:
            accepted[int(delivery["step"])].append(delivery)

    # (robot, target) -> (state_time, stage, remaining): newest accepted packet wins.
    believed: dict[tuple[str, str], TruthEntry] = {}
    ages: list[float] = []
    scored_steps = 0
    stage_errors = 0
    requirement_errors = 0
    first_knowledge: dict[tuple[str, str], float] = {}
    for sample in range(1, int(horizon) + 1):
        moment = float(sample - 1)
        for delivery in sorted(accepted.get(sample, ()), key=lambda item: str(item["agent"])):
            emission = sends.get((int(delivery["sent_step"]), str(delivery["target"])))
            if emission is None:
                continue
            key = (str(delivery["agent"]), str(delivery["target"]))
            believed[key] = (
                float(emission["now"]),
                int(emission["stage_index"]),
                tuple(int(value) for value in emission["remaining"]),
            )
            first_knowledge.setdefault(key, moment)
        for (robot, target), (state_time, stage, remaining) in believed.items():
            if active_until is not None and moment > active_until.get(target, float("inf")):
                closed_skipped += 1
                continue
            ages.append(moment - state_time)
            if truth is None:
                continue
            actual = _tr_at(truth.get(target, []), moment)
            if actual is None:
                continue
            scored_steps += 1
            stage_errors += int(actual[0] != stage)
            requirement_errors += int(actual[1] != remaining and bool(actual[1]))
    arrivals = sorted(first_knowledge.values())
    return {
        "known_robot_target_pairs": len(first_knowledge),
        "robots_informed": len({robot for robot, _ in first_knowledge}),
        "robots_total": len(list(robots)),
        "time_to_first_knowledge_mean": round(sum(arrivals) / len(arrivals), 3) if arrivals else None,
        "time_to_first_knowledge_max": round(max(arrivals), 3) if arrivals else None,
        "belief_age_steps_mean": round(sum(ages) / len(ages), 3) if ages else None,
        "belief_age_steps_max": round(max(ages), 3) if ages else None,
        "scored_robot_target_steps": scored_steps,
        "stage_error_steps": stage_errors,
        "requirement_error_steps": requirement_errors,
    }


def task_outcomes(timelines: dict[str, Any]) -> dict[str, Any]:
    """Discovery-to-outcome latencies, on the frozen round's definition of on time."""
    rows = list(timelines.values())
    latencies = [
        float(timeline.assignment_latency)
        for timeline in rows
        if getattr(timeline, "assignment_latency", None) is not None
    ]
    work_latencies = []
    for timeline in rows:
        discovered = float(getattr(timeline, "detected_at", 0.0) or 0.0)
        first_work = getattr(timeline, "first_work_at", None)
        if first_work is not None:
            work_latencies.append(round(float(first_work) - discovered, 3))
    return {
        "targets": len(rows),
        "assignment_latency_mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "assignment_latency_max": round(max(latencies), 3) if latencies else None,
        "discovery_to_first_work_mean": (
            round(sum(work_latencies) / len(work_latencies), 3) if work_latencies else None
        ),
        "worked_steps_total": round(
            float(sum(getattr(timeline, "worked_steps", 0.0) for timeline in rows)), 3
        ),
    }
