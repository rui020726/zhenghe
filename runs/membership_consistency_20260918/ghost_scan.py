"""How much ghost capability did the 19 stale releases actually produce?

Replays the 17 stress episodes that contain one, with a recorder that opens a watch on
every release the coordinator acted on: for each later tick, is the released robot still on
the source task's roster, is it off site, did the source stage's work clock still run down
this tick, and would that stage be uncovered if the roster only counted robots that are
really there. Each tick that answers "credited while absent" is one ghost tick.

Nothing is patched into the environment; the recorder only reads.

    D:/project_env/python.exe runs/membership_consistency_20260918/ghost_scan.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
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

from integrated_sar.coordinator import CoalitionPlannerCoordinator  # noqa: E402
from integrated_sar.joint_driver import build_joint_run  # noqa: E402

import stress_replay  # noqa: E402

OUT = HERE / "ghost_scan"
EPS = 1e-9


def episode_targets() -> list[tuple[str, int]]:
    found: set[tuple[str, int]] = set()
    for path in (REPO / "runs" / "release_validation_20260918" / "stress_safety").glob("*/scan_s*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if any(row["truth_marginal_contribution"] > EPS for row in payload["executed"]):
            found.add((payload["condition"], int(payload["seed"])))
    return sorted(found)


def stage_of(task):
    return task["stages"][min(int(task["current_stage_index"]), len(task["stages"]) - 1)]


def on_site(env, task, tid):
    return [
        member for member in task["members"]
        if env.agent_dic[member].get("status") == "working"
        and int(env.agent_dic[member].get("current_task", -1)) == tid
    ]


def run_one(condition: str, seed: int, arm: str = "candidate") -> dict:
    directory = OUT / arm / condition
    directory.mkdir(parents=True, exist_ok=True)
    rp = directory / f"{arm}_rp_s{seed}.jsonl"
    trace = directory / f"{arm}_tr_s{seed}.jsonl"
    configuration = replace(
        stress_replay.configuration_for(condition, seed, arm, stress_replay.stream_path(condition, seed)),
        output=str(rp),
        decision_trace=str(trace),
    )
    run = build_joint_run(configuration)
    adapter = run.task_adapter
    env = adapter.env
    original = CoalitionPlannerCoordinator._joint_redundant
    watches: dict[tuple[str, int], dict] = {}
    history: list[dict] = []

    def recording(agents, committed, checks):
        released = original(agents, committed, checks)
        for agent_id in released:
            task_id = agents[agent_id].current_task
            internal = adapter.external_to_internal.get(str(task_id))
            if internal is None:
                continue
            key = (str(agent_id), int(internal))
            watches.setdefault(
                key,
                {
                    "opened_at": float(env.current_time),
                    "agent_id": str(agent_id),
                    "internal": int(internal),
                    "task_id": task_id,
                    "ghost_ticks": 0,
                    "ticks_followed": 0,
                    "dual_roster_ticks": 0,
                    "cleared_by": None,
                    "cleared_at": None,
                    "source_finished_while_absent": False,
                    "max_marginal_units": 0.0,
                }
            )
        return released

    CoalitionPlannerCoordinator._joint_redundant = staticmethod(recording)
    inner_step = run.orchestrator.step

    def uncovered_onsite(task, tid):
        """True when nobody standing on the task could cover its believed stage requirement."""
        requirement = np.array(stage_of(task)["requirements"], dtype=float)
        present = on_site(env, task, tid)
        supply = (
            np.sum([env.agent_dic[int(member)]["abilities"] for member in present], axis=0)
            if present
            else np.zeros(requirement.shape)
        )
        return bool(np.maximum(requirement - supply, 0.0).sum() > EPS)

    def step():
        clocks = {
            tid: float(stage_of(task)["time_left"])
            for tid, task in env.task_dic.items()
            if not task.get("finished")
        }
        report = inner_step()
        for key, watch in watches.items():
            if watch["cleared_by"]:
                continue
            tid = watch["internal"]
            task = env.task_dic.get(tid)
            if task is None:
                watch["cleared_by"] = "task_gone"
                watch["cleared_at"] = float(env.current_time)
                continue
            roster = [str(member) for member in task["members"]]
            still_listed = watch["agent_id"] in roster
            agent = env.agent_dic.get(int(watch["agent_id"]))
            on_site_now = watch["agent_id"] in [str(member) for member in on_site(env, task, tid)]
            before = clocks.get(tid)
            advanced = before is not None and float(stage_of(task)["time_left"]) < before - EPS
            watch["ticks_followed"] += 1
            # The clock ran down on this tick while the robots actually present could not
            # have covered the requirement: the stage advanced on credit for an absent body.
            if advanced and not on_site_now and uncovered_onsite(task, tid):
                watch["ghost_ticks"] += 1
            if sum(
                1
                for other in env.task_dic.values()
                if str(watch["agent_id"]) in [str(member) for member in other["members"]]
            ) > 1:
                watch["dual_roster_ticks"] += 1
            if task.get("finished"):
                watch["cleared_by"] = "source_finished"
                watch["cleared_at"] = float(env.current_time)
                watch["source_finished_while_absent"] = watch["ghost_ticks"] > 0
            elif not still_listed or (agent is not None and int(agent.get("current_task", -1)) != tid and not on_site_now):
                # The roster no longer carries the credit, or the robot was put back on site.
                if not still_listed:
                    watch["cleared_by"] = "roster_cleared"
                    watch["cleared_at"] = float(env.current_time)
        return report

    run.orchestrator.step = step
    try:
        run.run()
    finally:
        CoalitionPlannerCoordinator._joint_redundant = original
        if run.decision_trace is not None:
            run.decision_trace.close()
    for watch in watches.values():
        if not watch["cleared_by"]:
            watch["cleared_by"] = "episode_end"
        history.append(watch)
    payload = {"condition": condition, "seed": seed, "arm": arm, "watches": history}
    destination = directory / f"ghost_s{seed}.json"
    part = destination.with_suffix(".json.part")
    part.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(part, destination)
    acted = [w for w in history if w["ghost_ticks"] > 0]
    print(
        f"[{condition} s{seed}] releases watched={len(history)} with ghost ticks={len(acted)} "
        f"max ghost={max((w['ghost_ticks'] for w in history), default=0)}",
        flush=True,
    )
    return payload


def summarise(watches: list[dict]) -> dict:
    ghost = [w for w in watches if w["ghost_ticks"] > 0]
    cleared: dict[str, int] = {}
    for watch in watches:
        cleared[watch["cleared_by"]] = cleared.get(watch["cleared_by"], 0) + 1
    return {
        "episodes": len({(w["condition"], w["seed"]) for w in watches}),
        "releases_watched": len(watches),
        "releases_with_ghost_capability": len(ghost),
        "ghost_ticks_total": int(sum(w["ghost_ticks"] for w in ghost)),
        "ghost_ticks_max": max((w["ghost_ticks"] for w in ghost), default=0),
        "dual_roster_releases": sum(1 for w in watches if w["dual_roster_ticks"] > 0),
        "cleared_by": cleared,
        "source_finished_while_still_credited_absent": sum(
            1 for w in watches if w["source_finished_while_absent"]
        ),
        "longest_residue_ticks": max(
            (int((w["cleared_at"] or 1000.0) - w["opened_at"]) for w in ghost) or [0]
        ),
    }


def aggregate() -> int:
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(OUT.glob("*/*/ghost_s*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for watch in payload["watches"]:
            watch["seed"] = payload["seed"]
            watch["condition"] = payload["condition"]
            by_arm[payload.get("arm", path.parts[-3])].append(watch)
    summary = {arm: summarise(watches) for arm, watches in sorted(by_arm.items())}
    summary["worst_candidate_releases"] = sorted(
        (
            {key: value for key, value in watch.items() if key != "cleared_by"}
            for watch in by_arm.get("candidate", [])
        ),
        key=lambda watch: -watch["ghost_ticks"],
    )[:12]
    (HERE / "GHOST_SCAN.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps({arm: value for arm, value in summary.items() if not arm.startswith("worst")}, indent=1))
    return 0


def main(argv: list[str]) -> int:
    if "--aggregate" in argv:
        return aggregate()
    arm = argv[argv.index("--arm") + 1] if "--arm" in argv else "candidate"
    jobs = episode_targets()
    if "--worker" in argv:
        index = int(argv[argv.index("--worker") + 1])
        total = int(argv[argv.index("--workers") + 1])
        for position, (condition, seed) in enumerate(jobs):
            if position % total != index:
                continue
            if (OUT / arm / condition / f"ghost_s{seed}.json").exists():
                continue
            run_one(condition, seed, arm)
        return 0
    processes = [
        subprocess.Popen(
            [
                sys.executable, str(Path(__file__).resolve()), "--arm", arm,
                "--worker", str(i), "--workers", str(6),
            ],
            cwd=str(REPO),
        )
        for i in range(6)
    ]
    codes = [process.wait() for process in processes]
    if any(codes):
        print(f"worker failures: {codes}")
        return 1
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
