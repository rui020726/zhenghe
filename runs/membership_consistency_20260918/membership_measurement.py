"""How much ghost capability does each arm generate, measured on the environment itself?

Watching releases is only half the question: the residue is the environment's, so the same
quantity has to be read in an arm that never releases anything. This harness walks the ticks
and counts, independently of any controller decision:

* a **ghost tick** is a tick where some task's stage work clock ran down while the robots
  actually standing on that task could not have covered its requirement -- the stage
  advanced on credit for an absent body;
* a **dual-roster robot-tick** is one tick where one robot was carried on the member list of
  two tasks at the same time, and so supplied capability to both.

The same script runs the two environment variants (as-is, or with member consistency
enforced) and both controller arms, which is what keeps the environment repair and the
controller effect separable instead of tangled.

    D:/project_env/python.exe runs/membership_consistency_20260918/membership_measurement.py \
        --cell B --seeds dev
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
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

from integrated_sar.expired_attribution import build_timelines  # noqa: E402
from integrated_sar.joint_driver import build_joint_run  # noqa: E402
from integrated_sar.lifecycle_metrics import load_records  # noqa: E402

import stress_replay  # noqa: E402

OUT = HERE / "cells"
EPS = 1e-9
FAILURE_SEEDS = [20003, 20008, 20052, 20070, 20083]
CONTROL_SEEDS = [20000, 20010, 20020, 20030, 20040, 20050, 20060, 20080, 20090, 20001]
DEV_STREAMS = REPO / "runs" / "coordinator_baselines_20260917" / "heldout" / "streams"
# Cells: which environment bookkeeping, which controller.
CELLS = {
    "A": {"arm": "baseline", "membership_fix": False},
    "B": {"arm": "candidate", "membership_fix": False},
    "C": {"arm": "baseline", "membership_fix": True},
    "D": {"arm": "candidate", "membership_fix": True},
}


def stage_of(task):
    return task["stages"][min(int(task["current_stage_index"]), len(task["stages"]) - 1)]


def on_site(env, task, tid):
    return [
        member
        for member in task["members"]
        if env.agent_dic[member].get("status") == "working"
        and int(env.agent_dic[member].get("current_task", -1)) == tid
    ]


def scenarios(kind: str) -> list[tuple[str, int, Path]]:
    if kind == "dev":
        return [("dev", seed, DEV_STREAMS / f"s{seed}.json") for seed in FAILURE_SEEDS + CONTROL_SEEDS]
    found = []
    for path in (REPO / "runs" / "release_validation_20260918" / "stress_safety").glob("*/scan_s*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if any(row["truth_marginal_contribution"] > EPS for row in payload["executed"]):
            seed = int(payload["seed"])
            found.append((payload["condition"], seed, stress_replay.stream_path(payload["condition"], seed)))
    return sorted(set(found))


def configuration_for(cell: str, seed: int, stream: Path, scenario: str, directory: Path):
    spec = CELLS[cell]
    arm = spec["arm"]
    condition = scenario if scenario != "dev" else "A_tighter_deadline"
    configuration = stress_replay.configuration_for(condition, seed, arm, stream)
    overrides = {
        "output": str(directory / f"{cell}_rp_s{seed}.jsonl"),
        "decision_trace": str(directory / f"{cell}_tr_s{seed}.jsonl"),
    }
    if scenario == "dev":
        # The development block was recorded on the default scenario, not a stress one.
        overrides.update({"num_goals": 6, "deadline_tightness": (0.5, 2.0)})
    configuration = replace(configuration, **overrides)
    if spec["membership_fix"]:
        configuration = replace(configuration, task_membership_consistency=True)
    return configuration, arm


def run_one(cell: str, scenario: str, seed: int, stream: Path) -> dict:
    directory = OUT / cell
    directory.mkdir(parents=True, exist_ok=True)
    configuration, arm = configuration_for(cell, seed, stream, scenario, directory)
    run = build_joint_run(configuration)
    env = run.task_adapter.env
    inner_step = run.orchestrator.step
    counters = {
        "ticks": 0,
        "ghost_ticks": 0,
        "ghost_tasks": set(),
        "dual_roster_robot_ticks": 0,
        "finished_with_nobody_on_site": 0,
        "ghost_time_steps": 0.0,
    }

    def step():
        before = {
            tid: (float(stage_of(task)["time_left"]), [str(m) for m in task["members"]])
            for tid, task in env.task_dic.items()
            if not task.get("finished")
        }
        report = inner_step()
        counters["ticks"] += 1
        for tid, (clock, roster) in before.items():
            task = env.task_dic.get(tid)
            if task is None or task.get("finished"):
                continue
            advanced = float(stage_of(task)["time_left"]) < clock - EPS
            if not advanced:
                continue
            requirement = np.array(stage_of(task)["requirements"], dtype=float)
            present = on_site(env, task, tid)
            supply = (
                np.sum([env.agent_dic[int(member)]["abilities"] for member in present], axis=0)
                if present
                else np.zeros(requirement.shape)
            )
            gap = np.maximum(requirement - supply, 0.0)
            if float(gap.sum()) > EPS:
                counters["ghost_ticks"] += 1
                counters["ghost_time_steps"] += float(gap.sum())
                counters["ghost_tasks"].add(tid)
        seen: dict[str, int] = {}
        for task in env.task_dic.values():
            for member in task["members"]:
                seen[str(member)] = seen.get(str(member), 0) + 1
        counters["dual_roster_robot_ticks"] += sum(1 for count in seen.values() if count > 1)
        return report

    run.orchestrator.step = step
    summary = run.run()
    if run.decision_trace is not None:
        run.decision_trace.close()
    counts = summary.get("status_counts", {}) or {}
    timelines = build_timelines(
        load_records(directory / f"{cell}_rp_s{seed}.jsonl"), arm=cell, seed=seed
    )
    payload = {
        "cell": cell,
        "scenario": scenario,
        "seed": seed,
        "arm": arm,
        "membership_fix": CELLS[cell]["membership_fix"],
        "completed": int(counts.get("completed", 0)),
        "expired": int(counts.get("expired", 0)),
        "on_time": sum(1 for t in timelines.values() if t.final_progress >= 1.0
                       and t.final_failure_reason != "deadline_expired"),
        "detections": len(timelines),
        "coverage_resets": int(sum(t.progress_setbacks for t in timelines.values())),
        "lost_work": round(float(sum(t.discarded_work_steps for t in timelines.values())), 3),
        "ticks": counters["ticks"],
        "ghost_ticks": counters["ghost_ticks"],
        "ghost_tasks": len(counters["ghost_tasks"]),
        "ghost_capability_units": round(counters["ghost_time_steps"], 2),
        "dual_roster_robot_ticks": counters["dual_roster_robot_ticks"],
    }
    destination = directory / f"{cell}_s{seed}.json"
    part = destination.with_suffix(".json.part")
    part.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(part, destination)
    print(
        f"[{cell} {scenario} s{seed}] on_time={payload['on_time']}/{payload['detections']} "
        f"expired={payload['expired']} ghost_ticks={payload['ghost_ticks']} "
        f"dual_roster={payload['dual_roster_robot_ticks']} resets={payload['coverage_resets']} "
        f"lost={payload['lost_work']}",
        flush=True,
    )
    return payload


def aggregate(kind: str) -> int:
    rows = {}
    for cell in CELLS:
        for path in sorted((OUT / cell).glob(f"{cell}_s*.json")):
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
            "on_time": int(sum(v["on_time"] for v in values)),
            "detections": int(sum(v["detections"] for v in values)),
            "expired": int(sum(v["expired"] for v in values)),
            "coverage_resets": int(sum(v["coverage_resets"] for v in values)),
            "lost_work": round(float(sum(v["lost_work"] for v in values)), 3),
            "ghost_ticks": int(sum(v["ghost_ticks"] for v in values)),
            "ghost_ticks_per_1k_task_ticks": round(
                1000.0 * sum(v["ghost_ticks"] for v in values) / max(1, sum(v["ticks"] for v in values)), 3
            ),
            "dual_roster_robot_ticks": int(sum(v["dual_roster_robot_ticks"] for v in values)),
        }
    (HERE / f"CELL_SUMMARY_{kind}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
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
    jobs = [(cell, scenario, seed, stream)
            for cell in ([args.cell] if args.cell else list(CELLS))
            for scenario, seed, stream in scenarios(args.seeds)]
    if args.worker is not None:
        for position, (cell, scenario, seed, stream) in enumerate(jobs):
            if position % args.workers != args.worker:
                continue
            if (OUT / cell / f"{cell}_s{seed}.json").exists():
                continue
            run_one(cell, scenario, seed, stream)
        return 0
    processes = [
        subprocess.Popen(
            [
                sys.executable, str(Path(__file__).resolve()), "--cell", cell, "--seeds", args.seeds,
                "--worker", str(i), "--workers", str(args.workers),
            ],
            cwd=str(REPO),
        )
        for cell in ([args.cell] if args.cell else list(CELLS))
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
