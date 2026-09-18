"""Corrected membership measurement: judge presence in the window that uses it.

The four-cell scan read roster presence after the environment step returned. ``task_update``
runs the stage work clock, and ``_check_working_timeouts`` afterwards releases robots that
timed out -- so a tick can show "the clock went down while the remaining bodies cannot cover
it" purely because the robot that covered it left later in the same tick. Seed 20008 tick 132
(task 3, roster [2,4] both on site with distance 0.0 when the clock ran, robot 4 timed out
afterwards) is exactly that shape, and it is the only ghost tick either baseline cell reports.

This script re-measures every cell of both seed blocks with three gates read where they apply:

1. **false credit while working** -- a stage clock ran down while a roster member who was not
   on site (wrong status or a different assignment) was needed to cover the requirement,
   evaluated at the moment ``task_update`` computes it;
2. **dual on-site supply** -- one robot carried on two task rosters in the same tick;
3. **closing task releases a robot assigned elsewhere** -- a task finishing or moving stage
   releases a member whose assignment was another task.

The legacy post-step detector runs in the same replay, so the corrected figure and the
committed one can be reconciled episode by episode instead of replaced. Nothing is patched
into the environment; the probes call the originals and only read.

    D:/project_env/python.exe runs/corrected_env_final_20260918/ghost_recheck.py --seeds dev
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
try:
    import torch

    torch.set_num_threads(1)
except Exception:
    pass

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MEMBERSHIP = REPO / "runs" / "membership_consistency_20260918"
for path in (REPO, HERE, MEMBERSHIP, REPO / "runs" / "release_validation_20260918"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from integrated_sar.coordinator import CoalitionPlannerCoordinator  # noqa: E402
from integrated_sar.joint_driver import build_joint_run  # noqa: E402
from integrated_sar.lifecycle_metrics import load_records  # noqa: E402

import membership_measurement as mm  # noqa: E402

OUT = HERE / "ghost_recheck"
EPS = 1e-9


def probe(cell: str, scenario: str, seed: int, stream: Path) -> dict:
    directory = OUT / cell
    directory.mkdir(parents=True, exist_ok=True)
    configuration, arm = mm.configuration_for(cell, seed, stream, scenario, directory)
    run = build_joint_run(configuration)
    env = run.task_adapter.env
    inner_task_update = env.task_update
    inner_step = run.orchestrator.step
    inner_advise = CoalitionPlannerCoordinator.advise
    apply_commands = run.gate.apply_assignments
    state = {"tick": 0}
    counters = {
        "ticks": 0,
        "false_credit_task_ticks": 0,
        "false_credit_units": 0.0,
        "legacy_ghost_task_ticks": 0,
        "dual_roster_robot_ticks": 0,
        "closing_released_robot_assigned_elsewhere": 0,
        "roster_carried_absent_member_task_ticks": 0,
        "releases_judged": 0,
        "releases_published": 0,
        "releases_executed": 0,
        "releases_without_future_information": 0,
        "departures_that_lowered_real_coverage": 0,
        "departure_into_an_uncovered_stage": 0,
        "executed_releases_that_uncovered_the_stage": 0,
    }
    events: list[dict] = []
    planner_refusals: dict[str, int] = {}
    # (internal task id, agent id) allowed at the execution boundary, waiting to be matched
    # against the next roster read of that task.
    accepted_releases: set[tuple[int, str]] = set()
    # task id -> (roster at the previous tick, was the current stage covered then)
    previous_state: dict[int, tuple[list[int], bool, int]] = {}

    def rows_of(tid, task):
        rows = []
        for member in [int(value) for value in task["members"]]:
            agent = env.agent_dic.get(member)
            if agent is None:
                rows.append({"member": member, "status": "missing-agent", "on_site": False})
                continue
            rows.append(
                {
                    "member": member,
                    "status": str(agent.get("status")),
                    "current_task": int(agent.get("current_task", -1)),
                    "location": [round(float(v), 4) for v in agent["location"]],
                    "distance_to_task": round(
                        float(
                            np.linalg.norm(
                                np.asarray(agent["location"], dtype=float)
                                - np.asarray(task["location"], dtype=float)
                            )
                        ),
                        4,
                    ),
                    "abilities": [int(v) for v in agent["abilities"]],
                    "on_site": bool(
                        agent.get("status") == "working"
                        and int(agent.get("current_task", -1)) == tid
                    ),
                }
            )
        return rows

    def supply_and_gap(rows, requirement):
        present = [row["abilities"] for row in rows if row["on_site"]]
        supply = (
            np.sum(np.array(present, dtype=float), axis=0)
            if present
            else np.zeros(len(requirement))
        )
        gap = np.maximum(np.asarray(requirement, dtype=float) - supply, 0.0)
        return supply, gap

    def task_update():
        """Gate 1 and 3: read presence in the window the work update actually uses."""
        for tid, task in env.task_dic.items():
            if task.get("finished"):
                continue
            roster = [int(value) for value in task["members"]]
            requirement = np.asarray(mm.stage_of(task)["requirements"], dtype=float)
            covered_now = float(np.sum(mm.stage_of(task)["status"])) <= EPS
            stage_now = int(task["current_stage_index"])
            previous = previous_state.get(int(tid))
            if previous is not None:
                gone = [member for member in previous[0] if member not in roster]
                for member in gone:
                    # Would this member have covered strictly more of the stage that is
                    # running now? That is the release's marginal contribution, read from
                    # the world rather than from the belief that authorised it.
                    remaining = [item for item in roster if item != member]
                    with_member = env.get_abilities(remaining + [member])
                    without = (
                        env.get_abilities(remaining)
                        if remaining
                        else np.zeros_like(requirement)
                    )
                    lowered = float(
                        np.sum(np.minimum(requirement, with_member))
                        - np.sum(np.minimum(requirement, without))
                    )
                    if lowered <= EPS:
                        continue
                    counters["departures_that_lowered_real_coverage"] += 1
                    if (int(tid), str(member)) not in accepted_releases:
                        continue
                    counters["executed_releases_that_uncovered_the_stage"] += 1
                    accepted_releases.discard((int(tid), str(member)))
                    events.append(
                        {
                            "kind": "executed_release_uncovered_the_stage",
                            "tick": state["tick"],
                            "time": round(float(env.current_time), 3),
                            "task_id": int(tid),
                            "released_agent": str(member),
                            "roster_before": previous[0],
                            "roster_after": roster,
                            "stage_index": stage_now,
                            "requirement": [int(v) for v in requirement],
                            "covered_units_lost": round(lowered, 3),
                            "remaining_gap": [int(v) for v in mm.stage_of(task)["status"]],
                        }
                    )
                if gone and not covered_now and previous[1] and previous[2] == stage_now:
                    # The same stage read as covered a tick ago and does not now. Kept apart
                    # from the criterion above: a stage that moved on between the two
                    # readings is not a departure effect, and this counter says so.
                    counters["departure_into_an_uncovered_stage"] += 1
            previous_state[int(tid)] = (roster, covered_now, stage_now)
        before = {
            tid: {
                "clock": float(mm.stage_of(task)["time_left"]),
                "stage": int(task["current_stage_index"]),
                "requirement": np.asarray(mm.stage_of(task)["requirements"], dtype=float),
                "rows": rows_of(tid, task),
            }
            for tid, task in env.task_dic.items()
            if not task.get("finished")
        }
        result = inner_task_update()
        for tid, snapshot in before.items():
            task = env.task_dic.get(tid)
            if task is None:
                continue
            now_stage = int(task.get("current_stage_index", 0))
            finished = bool(task.get("finished"))
            now_clock = float(mm.stage_of(task)["time_left"]) if not finished else 0.0
            ran = now_stage == snapshot["stage"] and now_clock < snapshot["clock"] - EPS
            closed_stage = now_stage > snapshot["stage"]
            # Gate 3: this task let go of members during this call (finish or stage move).
            if finished or closed_stage:
                still_listed = {int(value) for value in task["members"]}
                for row in snapshot["rows"]:
                    if finished or row["member"] not in still_listed:
                        if row["current_task"] != int(tid):
                            counters["closing_released_robot_assigned_elsewhere"] += 1
                            events.append(
                                {
                                    "kind": "closing_released_robot_assigned_elsewhere",
                                    "tick": state["tick"],
                                    "time": round(float(env.current_time), 3),
                                    "task_id": int(tid),
                                    "member": row["member"],
                                    "member_status": row["status"],
                                    "member_current_task": row["current_task"],
                                    "task_finished": finished,
                                }
                            )
            if finished or not (ran or closed_stage):
                continue
            # Gate 1: the clock ran on this stage -- did it need a body that was not there?
            supply, gap = supply_and_gap(snapshot["rows"], snapshot["requirement"])
            absent = [row for row in snapshot["rows"] if not row["on_site"]]
            if absent:
                counters["roster_carried_absent_member_task_ticks"] += 1
            if sum(gap) <= EPS:
                continue
            counters["false_credit_task_ticks"] += 1
            counters["false_credit_units"] += float(gap.sum())
            events.append(
                {
                    "kind": "false_credit_while_working",
                    "tick": state["tick"],
                    "time": round(float(env.current_time), 3),
                    "task_id": int(tid),
                    "stage_index": snapshot["stage"],
                    "work_kind": "stage_clock_ran" if ran else "stage_closed",
                    "clock_before": round(snapshot["clock"], 3),
                    "clock_after": round(now_clock, 3),
                    "requirement": [int(v) for v in snapshot["requirement"]],
                    "work_delta": round(snapshot["clock"] - (0.0 if closed_stage else now_clock), 3),
                    "roster_at_entry": [row["member"] for row in snapshot["rows"]],
                    "on_site_at_entry": [row["member"] for row in snapshot["rows"] if row["on_site"]],
                    "on_site_supply": [int(v) for v in supply],
                    "gap_if_only_on_site_counted": [int(v) for v in gap],
                    "absent_members": [
                        {
                            "member": row["member"],
                            "status": row["status"],
                            "current_task": row["current_task"],
                            "distance_to_task": row["distance_to_task"],
                            "abilities": row["abilities"],
                        }
                        for row in absent
                    ],
                }
            )
        return result

    def step():
        state["tick"] += 1
        clocks = {
            tid: float(mm.stage_of(task)["time_left"])
            for tid, task in env.task_dic.items()
            if not task.get("finished")
        }
        report = inner_step()
        counters["ticks"] += 1
        # The legacy post-step detector, kept so the old records stay reconcilable.
        for tid, clock in clocks.items():
            task = env.task_dic.get(tid)
            if task is None or task.get("finished"):
                continue
            if float(mm.stage_of(task)["time_left"]) >= clock - EPS:
                continue
            rows = rows_of(tid, task)
            _, gap = supply_and_gap(rows, mm.stage_of(task)["requirements"])
            if sum(gap) > EPS:
                counters["legacy_ghost_task_ticks"] += 1
        seen: dict[int, int] = {}
        for task in env.task_dic.values():
            for member in task["members"]:
                seen[int(member)] = seen.get(int(member), 0) + 1
        counters["dual_roster_robot_ticks"] += sum(1 for count in seen.values() if count > 1)
        return report

    def advise_with_counters(self, state):
        advice = inner_advise(self, state)
        metadata = advice.metadata or {}
        counters["releases_published"] += int(metadata.get("redundant_releases", 0))
        counters["releases_without_future_information"] += int(
            metadata.get("releases_without_future_information", 0)
        )
        for reason, value in (metadata.get("release_refusals") or {}).items():
            planner_refusals[str(reason)] = planner_refusals.get(str(reason), 0) + int(value)
        return advice

    def apply_with_ledger(commands):
        known = {row.command_id for row in run.gate.rejections}
        apply_commands(commands)
        refused = {row.command_id for row in run.gate.rejections} - known
        for issued in commands:
            metadata = issued.metadata or {}
            source = metadata.get("release_from_task")
            if not source or issued.command_id in refused:
                continue
            counters["releases_executed"] += 1
            internal = run.task_adapter.external_to_internal.get(str(source))
            if internal is not None:
                accepted_releases.add((int(internal), str(issued.agent_id)))

    env.task_update = task_update
    run.orchestrator.step = step
    run.gate.apply_assignments = apply_with_ledger
    CoalitionPlannerCoordinator.advise = advise_with_counters
    try:
        summary = run.run()
    finally:
        CoalitionPlannerCoordinator.advise = inner_advise
        if run.decision_trace is not None:
            run.decision_trace.close()
    by_reason: dict[str, int] = {}
    for row in load_records(directory / f"{cell}_rp_s{seed}.jsonl"):
        if row.get("event_type") != "assignment_rejected":
            continue
        reason = str(row.get("failure_reason"))
        by_reason[reason] = by_reason.get(reason, 0) + 1
    payload = {
        "cell": cell,
        "scenario": scenario,
        "seed": seed,
        "arm": mm.CELLS[cell]["arm"],
        "membership_fix": mm.CELLS[cell]["membership_fix"],
        "source_check": mm.CELLS[cell]["source_check"],
        "ticks": counters["ticks"],
        "false_credit_task_ticks": counters["false_credit_task_ticks"],
        "false_credit_units": round(counters["false_credit_units"], 3),
        "legacy_ghost_task_ticks": counters["legacy_ghost_task_ticks"],
        "dual_roster_robot_ticks": counters["dual_roster_robot_ticks"],
        "closing_released_robot_assigned_elsewhere": counters[
            "closing_released_robot_assigned_elsewhere"
        ],
        "roster_carried_absent_member_task_ticks": counters[
            "roster_carried_absent_member_task_ticks"
        ],
        "rejections_by_reason": dict(sorted(by_reason.items())),
        "releases_published": counters["releases_published"],
        "releases_executed": counters["releases_executed"],
        "releases_refused_by_planner": dict(sorted(planner_refusals.items())),
        "releases_without_future_information": counters["releases_without_future_information"],
        "departures_that_lowered_real_coverage": counters["departures_that_lowered_real_coverage"],
        "departure_into_an_uncovered_stage": counters["departure_into_an_uncovered_stage"],
        "executed_releases_that_uncovered_the_stage": counters[
            "executed_releases_that_uncovered_the_stage"
        ],
        "completed": int(summary.get("status_counts", {}).get("completed", 0)),
        "expired": int(summary.get("status_counts", {}).get("expired", 0)),
        "events": events,
    }
    destination = directory / f"recheck_{cell}_s{seed}.json"
    part = destination.with_suffix(".json.part")
    part.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(part, destination)
    print(
        f"[{cell} {scenario} s{seed}] false_credit={payload['false_credit_task_ticks']} "
        f"legacy_ghost={payload['legacy_ghost_task_ticks']} dual_roster={payload['dual_roster_robot_ticks']} "
        f"closing_release_of_elsewhere_assigned={payload['closing_released_robot_assigned_elsewhere']} "
        f"absent_still_credited_ticks={payload['roster_carried_absent_member_task_ticks']} "
        f"releases={payload['releases_executed']}/{payload['releases_published']} "
        f"release_uncovered={payload['executed_releases_that_uncovered_the_stage']} "
        f"no_next_stage_info={payload['releases_without_future_information']}",
        flush=True,
    )
    return payload


def aggregate(kind: str) -> int:
    rows: dict[str, list[dict]] = {}
    for cell in mm.CELLS:
        for path in sorted((OUT / cell).glob(f"recheck_{cell}_s*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["scenario"] == kind or (kind == "stress" and payload["scenario"] != "dev"):
                rows.setdefault(cell, []).append(payload)
    summary = {}
    for cell, values in sorted(rows.items()):
        if not values:
            continue
        summary[cell] = {
            "episodes": len(values),
            "arm": values[0]["arm"],
            "membership_fix": values[0]["membership_fix"],
            "source_check": values[0]["source_check"],
            "environment_ticks": int(sum(v["ticks"] for v in values)),
            "false_credit_task_ticks": int(sum(v["false_credit_task_ticks"] for v in values)),
            "false_credit_capability_units": round(
                float(sum(v["false_credit_units"] for v in values)), 3
            ),
            "legacy_ghost_task_ticks": int(sum(v["legacy_ghost_task_ticks"] for v in values)),
            "dual_roster_robot_ticks": int(sum(v["dual_roster_robot_ticks"] for v in values)),
            "closing_released_robot_assigned_elsewhere": int(
                sum(v["closing_released_robot_assigned_elsewhere"] for v in values)
            ),
            "roster_carried_absent_member_task_ticks": int(
                sum(v["roster_carried_absent_member_task_ticks"] for v in values)
            ),
            "rejections_by_reason": {
                reason: sum(v["rejections_by_reason"].get(reason, 0) for v in values)
                for reason in sorted({r for v in values for r in v["rejections_by_reason"]})
            },
            "releases_published": int(sum(v["releases_published"] for v in values)),
            "releases_executed": int(sum(v["releases_executed"] for v in values)),
            "release_refusals_by_planner": {
                reason: sum(v["releases_refused_by_planner"].get(reason, 0) for v in values)
                for reason in sorted(
                    {
                        reason
                        for v in values
                        for reason in (v["releases_refused_by_planner"] or {})
                    }
                )
            },
            "releases_without_future_information": int(
                sum(v["releases_without_future_information"] for v in values)
            ),
            "departures_that_lowered_real_coverage": int(
                sum(v["departures_that_lowered_real_coverage"] for v in values)
            ),
            "departure_into_an_uncovered_stage": int(
                sum(v["departure_into_an_uncovered_stage"] for v in values)
            ),
            "executed_releases_that_uncovered_the_stage": int(
                sum(v["executed_releases_that_uncovered_the_stage"] for v in values)
            ),
            "completed": int(sum(v["completed"] for v in values)),
            "expired": int(sum(v["expired"] for v in values)),
        }
    destination = HERE / f"GHOST_RECHECK_{kind}.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default=None)
    parser.add_argument("--seeds", default="dev", choices=("dev", "stress"))
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args(argv)
    if args.aggregate:
        return aggregate(args.seeds)
    jobs = [
        (cell, scenario, seed, stream)
        for cell in ([args.cell] if args.cell else list(mm.CELLS))
        for scenario, seed, stream in mm.scenarios(args.seeds)
    ]
    if args.worker is not None:
        for position, (cell, scenario, seed, stream) in enumerate(jobs):
            if position % args.workers != args.worker:
                continue
            if (OUT / cell / f"recheck_{cell}_s{seed}.json").exists():
                continue
            probe(cell, scenario, seed, stream)
        return 0
    processes = [
        subprocess.Popen(
            [
                sys.executable, str(Path(__file__).resolve()), "--cell", cell,
                "--seeds", args.seeds, "--worker", str(i), "--workers", str(args.workers),
            ],
            cwd=str(REPO),
        )
        for cell in ([args.cell] if args.cell else list(mm.CELLS))
        for i in range(args.workers)
    ]
    codes = [process.wait() for process in processes]
    if any(codes):
        print(f"worker failures: {codes}")
        return 1
    return aggregate(args.seeds)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
