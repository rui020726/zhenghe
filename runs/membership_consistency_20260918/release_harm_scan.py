"""Were all 19 harmful releases blocked? Measured before and after, same episodes.

The 19 came from the stress block: releases the coordinator acted on where taking the robot
off its task would have lowered the coverage the stage really had. This runs the same 17
episodes twice -- once with the environment and controller as they were when the 19 were
counted, once with roster consistency and the source-stage gate -- and counts the same
quantity in both, so the comparison is on identical episodes rather than a re-selected set.

    D:/project_env/python.exe runs/membership_consistency_20260918/release_harm_scan.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
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

from integrated_sar.coordinator import CoalitionPlannerCoordinator  # noqa: E402
from integrated_sar.expired_attribution import build_timelines  # noqa: E402
from integrated_sar.joint_driver import build_joint_run  # noqa: E402
from integrated_sar.lifecycle_metrics import load_records  # noqa: E402

import membership_measurement as cells  # noqa: E402

OUT = HERE / "harm_scan"
COMPARISON = ("B", "D")


def episodes() -> list[tuple[str, int, Path]]:
    return cells.scenarios("stress")


def run_one(cell: str, scenario: str, seed: int, stream: Path) -> dict:
    directory = OUT / cell
    directory.mkdir(parents=True, exist_ok=True)
    configuration, _arm = cells.configuration_for(cell, seed, stream, scenario, directory)
    run = build_joint_run(configuration)
    adapter = run.task_adapter
    env = adapter.env
    original = CoalitionPlannerCoordinator._joint_redundant
    records: list[dict] = []

    def recording(agents, committed, checks):
        released = original(agents, committed, checks)
        for agent_id in released:
            task_id = agents[agent_id].current_task
            internal = adapter.external_to_internal.get(str(task_id))
            task = env.task_dic.get(int(internal)) if internal is not None else None
            if task is None or task.get("finished"):
                continue
            index = min(int(task.get("current_stage_index", 0)), len(task["stages"]) - 1)
            requirement = np.array(task["stages"][index]["requirements"], dtype=float)
            members = [str(member) for member in task["members"]]
            abilities = {
                member: np.array(env.agent_dic[int(member)]["abilities"], dtype=float)
                for member in members if int(member) in env.agent_dic
            }
            mine = abilities.get(str(agent_id))
            if mine is None:
                continue
            others = [value for member, value in abilities.items() if member != str(agent_id)]
            covered = np.sum(others, axis=0) if others else np.zeros(requirement.shape)
            gap_without = np.maximum(requirement - covered, 0.0)
            records.append(
                {
                    "now": float(env.current_time),
                    "agent_id": str(agent_id),
                    "task_id": task_id,
                    "marginal_units": float(np.minimum(gap_without, mine).sum()),
                }
            )
        return released

    CoalitionPlannerCoordinator._joint_redundant = staticmethod(recording)
    try:
        summary = run.run()
    finally:
        CoalitionPlannerCoordinator._joint_redundant = original
        if run.decision_trace is not None:
            run.decision_trace.close()

    redecided = {
        (str(row["agent_id"]), float(row["timestamp"]))
        for row in (json.loads(line) for line in (directory / f"{cell}_tr_s{seed}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
        if row.get("event_type") == "decision" and row.get("replanned")
    }
    executed = [row for row in records if (row["agent_id"], row["now"]) in redecided]
    counts = summary.get("status_counts", {}) or {}
    timelines = build_timelines(load_records(directory / f"{cell}_rp_s{seed}.jsonl"), arm=cell, seed=seed)
    payload = {
        "cell": cell,
        "seed": seed,
        "on_time": sum(1 for t in timelines.values() if t.final_progress >= 1.0
                       and t.final_failure_reason != "deadline_expired"),
        "detections": len(timelines),
        "completed": int(counts.get("completed", 0)),
        "judged": len(records),
        "executed": len(executed),
        "executed_lowering_real_coverage": sum(1 for row in executed if row["marginal_units"] > 1e-9),
        "worst_units": max((row["marginal_units"] for row in executed), default=0.0),
    }
    destination = directory / f"harm_s{seed}.json"
    part = destination.with_suffix(".json.part")
    part.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(part, destination)
    print(
        f"[{cell} s{seed}] judged={payload['judged']} executed={payload['executed']} "
        f"harmful={payload['executed_lowering_real_coverage']} on_time={payload['on_time']}/{payload['detections']}",
        flush=True,
    )
    return payload


def aggregate() -> int:
    tally = defaultdict(lambda: defaultdict(int))
    for cell in COMPARISON:
        for path in sorted((OUT / cell).glob("harm_s*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key in ("judged", "executed", "executed_lowering_real_coverage"):
                tally[cell][key] += payload[key]
            tally[cell]["episodes"] += 1
            tally[cell]["on_time"] += payload["on_time"]
            tally[cell]["detections"] += payload["detections"]
    summary = {cell: dict(values) for cell, values in sorted(tally.items())}
    for cell, values in summary.items():
        values["harmful_per_executed"] = round(
            values["executed_lowering_real_coverage"] / max(1, values["executed"]), 4
        )
    (HERE / "HARM_SCAN.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=1))
    return 0


def main(argv: list[str]) -> int:
    if "--aggregate" in argv:
        return aggregate()
    if "--worker" in argv:
        cell = argv[argv.index("--cell") + 1]
        index = int(argv[argv.index("--worker") + 1])
        total = int(argv[argv.index("--workers") + 1])
        jobs = episodes()
        for position, (scenario, seed, stream) in enumerate(jobs):
            if position % total != index:
                continue
            if (OUT / cell / f"harm_s{seed}.json").exists():
                continue
            run_one(cell, scenario, seed, stream)
        return 0
    processes = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--cell", cell,
             "--worker", str(i), "--workers", "6"],
            cwd=str(REPO),
        )
        for cell in COMPARISON
        for i in range(6)
    ]
    codes = [process.wait() for process in processes]
    if any(codes):
        print(f"worker failures: {codes}")
        return 1
    return aggregate()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
