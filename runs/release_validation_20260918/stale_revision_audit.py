"""Stale-command rejection forensics (release brief section 3).

The 400-seed independent block recorded two candidate-only ``stale_task_revision`` events.
This script reads the two arms' lifecycle logs and answers, per event: what revision the
command carried, when the task moved, what the assignment looked like before and after,
whether the robot's work state changed, how long until it received a command that was
accepted, and whether the same stale command was attempted again. It also derives the
four metrics the brief asks to keep separate: rejections by reason, invalid commands that
were actually executed, recovery latency, and repeated stale attempts.

    D:/project_env/python.exe runs/release_validation_20260918/stale_revision_audit.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
HELDOUT = REPO / "runs" / "coalition_validation_20260918" / "heldout"
ARMS = ("baseline", "candidate")


def read_in_file_order(path: Path) -> list[dict]:
    """The logger's own order: sorting would separate a command from its rejection."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def audit_arm(arm: str) -> tuple[list[dict], dict]:
    events: list[dict] = []
    repeated_pairs = 0
    retry_within_interval = 0
    for path in sorted(HELDOUT.glob(f"{arm}_rp_s*.jsonl")):
        rows = read_in_file_order(path)
        seed = int(path.stem.split("_s")[-1])
        # Registry revision timeline per target, from the log's own revision counter.
        timeline: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for row in rows:
            if row["task_revision"] is not None:
                timeline[row["target_id"]].append((float(row["timestamp"]), int(row["task_revision"])))
        refused = {
            (row.get("detail") or {}).get("command_id")
            for row in rows
            if row["event_type"] == "assignment_rejected"
        }

        def executed(row: dict) -> bool:
            """An assignment the gate let through, not one it dropped in the same tick."""
            return (
                row["event_type"] == "assignment"
                and (row.get("detail") or {}).get("command_id") not in refused
            )

        last_rejection: dict[tuple[str, str], float] = {}
        for index, row in enumerate(rows):
            if row["event_type"] != "assignment_rejected":
                continue
            detail = row.get("detail") or {}
            agent_id = next(iter(row["assigned_agents"]), None)
            target = row["target_id"]
            # Every revision the registry held for this target at this timestamp.
            same_tick = [
                revision
                for stamp, revision in timeline[target]
                if stamp == float(row["timestamp"])
            ]
            # Was this the same stale command the agent had already been refused?
            prior = [row2 for row2 in rows[:index] if row2["event_type"] == "assignment_rejected"
                     and row2["target_id"] == target
                     and next(iter(row2["assigned_agents"]), None) == agent_id]
            consecutive = bool(prior) and (prior[-1].get("detail") or {}).get(
                "command_task_revision"
            ) == detail.get("command_task_revision")
            if consecutive:
                repeated_pairs += 1
            key = (str(agent_id), target)
            previous_at = last_rejection.get(key)
            soon_again = previous_at is not None and float(row["timestamp"]) - previous_at <= 5
            if soon_again:
                retry_within_interval += 1
            last_rejection[key] = float(row["timestamp"])
            # Next command this agent was allowed to carry out after the rejection.
            recovery = None
            for row2 in rows[index + 1 :]:
                if executed(row2) and agent_id in row2["assigned_agents"]:
                    recovery = {
                        "at": float(row2["timestamp"]),
                        "target": row2["target_id"],
                        "latency": round(float(row2["timestamp"]) - float(row["timestamp"]), 2),
                    }
                    break
            # The last command this agent was actually carrying out when the gate refused.
            working_before = None
            for row2 in reversed(rows[:index]):
                if executed(row2) and agent_id in row2["assigned_agents"]:
                    working_before = {"target": row2["target_id"], "at": float(row2["timestamp"])}
                    break
            events.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "target": target,
                    "agent_id": agent_id,
                    "reason": row["failure_reason"],
                    "rejected_at": float(row["timestamp"]),
                    "command_id": detail.get("command_id"),
                    "command_issued_at": detail.get("issued_at"),
                    "command_valid_until": detail.get("valid_until"),
                    "command_task_revision": detail.get("command_task_revision"),
                    "registry_revision_at_rejection": row["task_revision"],
                    "registry_revisions_same_tick": sorted(set(same_tick)),
                    "task_status_at_rejection": row["status"],
                    "task_stage_at_rejection": row["stage_index"],
                    "task_progress_at_rejection": row["progress"],
                    "task_remaining_at_rejection": row["remaining_requirements"],
                    "previous_assignment_for_agent": working_before,
                    "next_accepted_assignment": recovery,
                    "repeat_of_previous_stale_command": consecutive,
                    "retry_within_one_allocation_interval": soon_again,
                }
            )
    by_reason = Counter(event["reason"] for event in events)
    return events, {
        "rejections_by_reason": dict(by_reason),
        "total_rejections": len(events),
        "repeated_stale_command_attempts": repeated_pairs,
        "rejections_within_one_allocation_interval_of_a_previous_one": retry_within_interval,
        "recovery_latencies": sorted(
            event["next_accepted_assignment"]["latency"]
            for event in events
            if event["next_accepted_assignment"]
        ),
        "events_without_a_later_accepted_assignment": sum(
            1 for event in events if not event["next_accepted_assignment"]
        ),
    }


def executed_invalid_check(arm: str) -> dict:
    """Accepted assignments only: an issued command the gate never refused must be current.

    ``assignment`` rows are written when the allocator builds the command, and the gate
    inspects it afterwards, so a row paired with an ``assignment_rejected`` carrying the
    same command id describes a command that was dropped, not one that ran. Counting those
    as executed is what made an earlier revision of this check report 11 and 13 violations
    that were in fact the gate working.
    """
    violations = []
    executed = 0
    rejected_and_never_executed = 0
    for path in sorted(HELDOUT.glob(f"{arm}_rp_s*.jsonl")):
        rows = read_in_file_order(path)
        seed = int(path.stem.split("_s")[-1])
        refused = {
            (row.get("detail") or {}).get("command_id")
            for row in rows
            if row["event_type"] == "assignment_rejected"
        }
        max_revision_at: dict[tuple[float, str], int] = {}
        for row in rows:
            if row["task_revision"] is not None:
                key = (float(row["timestamp"]), row["target_id"])
                max_revision_at[key] = max(max_revision_at.get(key, 0), int(row["task_revision"]))
        for row in rows:
            if row["event_type"] != "assignment":
                continue
            key = (float(row["timestamp"]), row["target_id"])
            command_id = (row.get("detail") or {}).get("command_id")
            if command_id in refused:
                rejected_and_never_executed += 1
                continue
            executed += 1
            current = max_revision_at.get(key)
            if current is not None and int(row["task_revision"]) < current:
                violations.append(
                    {"seed": seed, "target": row["target_id"], "at": row["timestamp"],
                     "stamped": row["task_revision"], "registry_max_this_tick": current,
                     "command_id": command_id}
                )
    return {"assignments_executed": executed, "executed_with_stale_revision": len(violations),
            "commands_rejected_before_execution": rejected_and_never_executed,
            "violations": violations[:10]}


def main() -> int:
    payload: dict = {"source": "runs/coalition_validation_20260918/heldout", "arms": {}}
    all_events: list[dict] = []
    for arm in ARMS:
        events, summary = audit_arm(arm)
        summary["invalid_commands_executed"] = executed_invalid_check(arm)
        payload["arms"][arm] = {"summary": summary, "events": events}
        all_events.extend(events)
        print(f"[{arm}] rejections={summary['total_rejections']} by_reason={summary['rejections_by_reason']} "
              f"repeats={summary['repeated_stale_command_attempts']} "
              f"executed_stale={summary['invalid_commands_executed']['executed_with_stale_revision']} "
              f"recovery_latencies={summary['recovery_latencies']} "
              f"no_later_assignment={summary['events_without_a_later_accepted_assignment']}")
    payload["candidate_only"] = [
        event for event in all_events
        if event["arm"] == "candidate"
        and event["seed"] not in {e["seed"] for e in all_events if e["arm"] == "baseline"}
    ]
    (HERE / "STALE_REVISION_AUDIT.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print()
    print("candidate-only events:")
    for event in payload["candidate_only"]:
        print(
            f"  s{event['seed']} {event['target']} agent {event['agent_id']} @t={event['rejected_at']}: "
            f"cmd rev {event['command_task_revision']} vs registry {event['registry_revision_at_rejection']} "
            f"({event['task_status_at_rejection']}, stage {event['task_stage_at_rejection']}, "
            f"progress {event['task_progress_at_rejection']}) "
            f"recovery={event['next_accepted_assignment']}"
        )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
