"""Where does the one remaining ghost tick in arm C actually come from?

``membership_measurement.py`` reads presence *after* the environment step returns. That is
one bookkeeping window too late: ``task_update`` runs the work clock, and
``_check_working_timeouts`` afterwards removes robots that timed out. A tick can therefore
look like "the clock ran down while nobody present could cover it" purely because the body
that covered it left at the end of the same tick.

This script walks seed 20008 (the only dev episode with a ghost tick in either baseline cell)
and reads presence at three moments per tick:

* **entry** -- as ``task_update`` begins, which is the state the environment's work decrement
  is computed against (arrival and ``apply_action`` have already run, nothing else moves the
  roster mid-call except the stage-transition/finalisation branches, which run after it);
* **update** -- as ``task_update`` returns;
* **end** -- as the whole step returns, which is what the existing detector reads.

A tick counts as real work if the same stage's clock went down, or if the stage advanced (its
clock reached zero inside this tick). A tick counts as a real ghost only when the robots on
site at *entry* could not cover the requirement the decrement used. Every tick the existing
detector flags is dumped side by side with the entry snapshot, so the two readings can be
attributed instead of asserted.

Nothing is patched into the environment: the probes call the originals and only read.

    D:/project_env/python.exe runs/membership_consistency_20260918/ghost_localize.py
"""

from __future__ import annotations

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
for path in (REPO, HERE, REPO / "runs" / "release_validation_20260918"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from integrated_sar.joint_driver import build_joint_run  # noqa: E402

import membership_measurement as mm  # noqa: E402

OUT = HERE / "ghost_localize"
EPS = 1e-9
SEED = 20008


def probe(cell: str, seed: int) -> dict:
    """Replay one episode, reading presence at task-update entry, exit and end of step."""
    directory = OUT / cell
    directory.mkdir(parents=True, exist_ok=True)
    stream = mm.DEV_STREAMS / f"s{seed}.json"
    configuration, arm = mm.configuration_for(cell, seed, stream, "dev", directory)
    run = build_joint_run(configuration)
    env = run.task_adapter.env
    inner_task_update = env.task_update
    inner_step = run.orchestrator.step
    state = {"tick": 0}
    real: list[dict] = []
    reported: list[dict] = []

    def rows_of(tid, task):
        """Roster members, each marked on-site by the environment's own notion of presence."""
        rows = []
        for member in [int(value) for value in task["members"]]:
            agent = env.agent_dic.get(member)
            if agent is None:
                rows.append({"member": member, "status": "missing-agent", "on_site": False})
                continue
            location = np.asarray(agent["location"], dtype=float)
            rows.append(
                {
                    "member": member,
                    "status": str(agent.get("status")),
                    "current_task": int(agent.get("current_task", -1)),
                    "target_type": str(agent.get("target_type")),
                    "target_id": (
                        None if agent.get("target_id") is None else int(agent.get("target_id"))
                    ),
                    "location": [round(float(value), 4) for value in location],
                    "distance_to_task": round(
                        float(np.linalg.norm(location - np.asarray(task["location"], dtype=float))),
                        4,
                    ),
                    "abilities": [int(value) for value in agent["abilities"]],
                    "on_site": bool(
                        agent.get("status") == "working"
                        and int(agent.get("current_task", -1)) == tid
                    ),
                }
            )
        return rows

    def gap_against(rows, requirement):
        present = [row["abilities"] for row in rows if row["on_site"]]
        supply = np.sum(np.array(present, dtype=float), axis=0) if present else np.zeros(len(requirement))
        gap = np.maximum(np.asarray(requirement, dtype=float) - supply, 0.0)
        return [int(value) for value in supply], [int(value) for value in gap]

    def task_update():
        before = {
            tid: {
                "clock": float(mm.stage_of(task)["time_left"]),
                "stage": int(task["current_stage_index"]),
                "requirement": [int(v) for v in mm.stage_of(task)["requirements"]],
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
            now_clock = (
                float(np.nan) if task.get("finished") else float(mm.stage_of(task)["time_left"])
            )
            same_stage_ran = now_stage == snapshot["stage"] and now_clock < snapshot["clock"] - EPS
            stage_closed = now_stage > snapshot["stage"]
            if not (same_stage_ran or stage_closed) or task.get("finished"):
                continue
            supply, gap = gap_against(snapshot["rows"], snapshot["requirement"])
            tick = {
                "cell": cell,
                "seed": seed,
                "tick": state["tick"],
                "time": round(float(env.current_time), 3),
                "task_id": int(tid),
                "unit": "task-tick (one task's stage clock over one environment tick)",
                "work_kind": "stage_clock_ran" if same_stage_ran else "stage_closed",
                "stage_index": snapshot["stage"],
                "requirement_used_by_the_work_update": snapshot["requirement"],
                "clock_before": round(snapshot["clock"], 3),
                "clock_after": round(now_clock, 3),
                "work_delta": round(snapshot["clock"] - (0.0 if stage_closed else now_clock), 3),
                "roster_at_entry": [row["member"] for row in snapshot["rows"]],
                "env_used_abilities_roster_wide": [
                    int(v) for v in env.get_abilities(
                        [row["member"] for row in snapshot["rows"]]
                    )
                ],
                "on_site_at_entry": [row["member"] for row in snapshot["rows"] if row["on_site"]],
                "on_site_abilities_at_entry": supply,
                "gap_at_entry_if_only_on_site_counted": gap,
                "counted_while_absent_at_entry": [
                    row["member"] for row in snapshot["rows"] if not row["on_site"]
                ],
                "members_at_entry": snapshot["rows"],
                "roster_at_update_exit": [int(v) for v in task["members"]],
            }
            if sum(gap) > EPS:
                real.append(tick)
            state.setdefault("_last", {})[(state["tick"], int(tid))] = tick
        return result

    def step():
        state["tick"] += 1
        clocks = {
            tid: float(mm.stage_of(task)["time_left"])
            for tid, task in env.task_dic.items()
            if not task.get("finished")
        }
        report = inner_step()
        # The existing detector, verbatim: presence read after the whole step.
        for tid, clock in clocks.items():
            task = env.task_dic.get(tid)
            if task is None or task.get("finished"):
                continue
            if float(mm.stage_of(task)["time_left"]) >= clock - EPS:
                continue
            rows = rows_of(tid, task)
            supply, gap = gap_against(rows, mm.stage_of(task)["requirements"])
            if sum(gap) <= EPS:
                continue
            entry = state.get("_last", {}).get((state["tick"], int(tid)))
            reported.append(
                {
                    "tick": state["tick"],
                    "time": round(float(env.current_time), 3),
                    "task_id": int(tid),
                    "requirement_at_end": [int(v) for v in mm.stage_of(task)["requirements"]],
                    "clock_before_step": round(clock, 3),
                    "clock_at_end": round(float(mm.stage_of(task)["time_left"]), 3),
                    "roster_at_end": [int(v) for v in task["members"]],
                    "members_at_end": rows,
                    "on_site_abilities_at_end": supply,
                    "gap_at_end": gap,
                    "counted_while_absent_at_end": [
                        row["member"] for row in rows if not row["on_site"]
                    ],
                    "entry_snapshot_of_the_same_tick": None if entry is None else {
                        key: entry[key]
                        for key in (
                            "work_kind",
                            "stage_index",
                            "requirement_used_by_the_work_update",
                            "clock_before",
                            "clock_after",
                            "roster_at_entry",
                            "on_site_at_entry",
                            "on_site_abilities_at_entry",
                            "gap_at_entry_if_only_on_site_counted",
                            "counted_while_absent_at_entry",
                            "roster_at_update_exit",
                        )
                    },
                    "members_at_entry": None if entry is None else entry["members_at_entry"],
                }
            )
        return report

    env.task_update = task_update
    run.orchestrator.step = step
    try:
        summary = run.run()
    finally:
        if run.decision_trace is not None:
            run.decision_trace.close()
    payload = {
        "cell": cell,
        "seed": seed,
        "arm": arm,
        "membership_fix": bool(mm.CELLS[cell]["membership_fix"]),
        "ticks": state["tick"],
        "final_status": summary.get("final_status"),
        "real_ghost_ticks_at_work_update": len(real),
        "reported_ghost_ticks_after_the_step": len(reported),
        "real_events": real,
        "reported_events": reported,
    }
    destination = directory / f"localize_{cell}_s{seed}.json"
    part = destination.with_suffix(".json.part")
    part.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(part, destination)
    print(
        f"[{cell} s{seed}] ticks={state['tick']} real (at the work update)={len(real)} "
        f"reported (after the step)={len(reported)}",
        flush=True,
    )
    for event in reported:
        entry = event["entry_snapshot_of_the_same_tick"] or {}
        print(
            f"    reported tick={event['tick']} t={event['time']} task={event['task_id']} "
            f"absent_at_end={event['counted_while_absent_at_end']} gap_at_end={event['gap_at_end']} "
            f"| same tick at the update: work={entry.get('work_kind')} "
            f"on_site={entry.get('on_site_at_entry')} gap={entry.get('gap_at_entry_if_only_on_site_counted')} "
            f"absent={entry.get('counted_while_absent_at_entry')}",
            flush=True,
        )
    for event in real:
        print(
            f"    REAL tick={event['tick']} t={event['time']} task={event['task_id']} "
            f"requirement={event['requirement_used_by_the_work_update']} "
            f"roster={event['roster_at_entry']} on_site={event['on_site_at_entry']} "
            f"gap={event['gap_at_entry_if_only_on_site_counted']}",
            flush=True,
        )
    return payload


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--worker":
        probe(argv[1], SEED)
        return 0
    processes = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", cell], cwd=str(REPO)
        )
        for cell in ("A", "C")
    ]
    codes = [process.wait() for process in processes]
    if any(codes):
        print(f"worker failures: {codes}")
        return 1
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
