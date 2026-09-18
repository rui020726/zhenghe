"""Seed 33011, target-7: where the two arms first part company, and what closed the stage.

The candidate lost a target the binding baseline finished on time. The direct cause was
already traced -- two environment working timeouts took the only dim-4 providers off stage 2
and nobody replaced them. This answers the sharper question: what did the *policy* change
upstream of that, when did the trajectories first differ, and did the policy change the
random stream itself.

The environment draws working timeouts from one generator per episode, sampling once per
arrival. Two arms that dispatch different numbers of robots to different places therefore
consume different numbers of draws, and every later sample lands on a different value. That
is not a measurement error and not a bug to fix; it is why a policy difference shows up as a
different timeout, somewhere, eventually.

    D:/project_env/python.exe runs/membership_consistency_20260918/divergence_33011.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
STRESS = REPO / "runs" / "release_validation_20260918" / "stress" / "B_higher_load"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SEED = 33011
TARGET = "piloc-e1-target-7"


def decisions(arm: str) -> list[dict]:
    path = STRESS / f"{arm}_tr_s{SEED}.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("event_type") == "decision"
    ]


def lifecycle(arm: str) -> list[dict]:
    path = STRESS / f"{arm}_rp_s{SEED}.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main() -> int:
    base, cand = decisions("baseline"), decisions("candidate")
    shared = min(len(base), len(cand))
    first = None
    for index in range(shared):
        left, right = base[index], cand[index]
        keys = ("step", "agent_id", "needs_decision", "replanned", "chosen_task",
                "actual_assignment", "command_id", "current_task", "mask")
        if any(left.get(key) != right.get(key) for key in keys):
            first = (index, left, right)
            break
    print(f"decision rows: baseline={len(base)} candidate={len(cand)}")
    if first is None:
        print("no decision-row difference at all in the shared prefix")
    else:
        index, left, right = first
        keys = ("step", "timestamp", "agent_id", "needs_decision", "replanned", "chosen_task",
                "actual_assignment", "command_id", "current_task")
        print(f"first divergence at row {index + 1}:")
        for key in keys:
            mark = "   =" if left.get(key) == right.get(key) else "   !"
            print(f"  {mark} {key:<18} baseline={left.get(key)!r:<28} candidate={right.get(key)!r}")

    print()
    for arm, rows in (("baseline", base), ("candidate", cand)):
        releases = sum(1 for row in rows if row.get("replanned"))
        commands = sum(1 for row in rows if row.get("command_id"))
        print(f"{arm:<10} re-decided-by-release={releases:>4} commands={commands:>5}")

    print()
    for arm in ("baseline", "candidate"):
        rows = lifecycle(arm)
        arrivals = [row for row in rows if row["event_type"] == "assignment"]
        # Every arrival samples a working timeout, so arrivals are the draw-count proxy.
        before = sum(1 for row in arrivals if float(row["timestamp"]) <= 276.0)
        target_rows = [row for row in rows if row["target_id"] == TARGET]
        stages = sorted({(float(row["timestamp"]), int(row["stage_index"])) for row in target_rows})
        print(f"{arm:<10} assignments issued by t=276: {before}")
        print(f"           target-7 stage timeline: {stages[:3]} ... {stages[-2:]}")
        final = [row for row in target_rows if row["event_type"] == "final"]
        if final:
            row = final[0]
            print(
                f"           target-7 final: stage={row['stage_index']} progress={row['progress']} "
                f"remaining={list(row['remaining_requirements'])} reason={row['failure_reason']}"
            )

    print()
    print("target-7 roster around the loss (candidate):")
    rows = lifecycle("candidate")
    previous = None
    for row in rows:
        if row["target_id"] != TARGET or row["event_type"] not in ("progress", "assignment", "final"):
            continue
        roster = tuple(row["assigned_agents"])
        if roster == previous:
            continue
        previous = roster
        print(
            f"   t={float(row['timestamp']):>6.0f} {row['event_type']:<11} stage={row['stage_index']} "
            f"roster={list(roster)} remaining={list(row['remaining_requirements'])}"
        )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
