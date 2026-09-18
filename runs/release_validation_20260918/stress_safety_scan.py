"""Does any executed release lower real coverage under stress? (brief sections 2 and 5)

The corrected verdict decides with a robot's own belief. A belief can lag, so the honest
question is not whether the rule looks conservative but what it did: this replays the 180
stress candidate episodes with a recorder wrapped around the joint release verdict -- it
returns the original verdict unchanged -- and for every release the coordinator actually
acted on, compares the registry's real coverage of that stage with the robot in place and
without it. It also checks, seed by seed, that the instrumented run reproduces the stored
stress outcome, which is what makes the recorder admissible as evidence rather than as a
different experiment.

    D:/project_env/python.exe runs/release_validation_20260918/stress_safety_scan.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace as _replace
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
try:
    import torch

    torch.set_num_threads(1)
except Exception:  # torch is optional in dependency-free environments
    pass

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for path in (REPO, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from integrated_sar.coordinator import CoalitionPlannerCoordinator  # noqa: E402
from integrated_sar.expired_attribution import build_timelines  # noqa: E402
from integrated_sar.joint_driver import build_joint_run  # noqa: E402
from integrated_sar.lifecycle_metrics import load_records  # noqa: E402

import stress_replay  # noqa: E402

OUT = HERE / "stress_safety"
WORKERS = 8
JOBS = [
    (condition, seed)
    for condition in stress_replay.CONDITIONS
    for seed in stress_replay.seeds_for(condition)
]


def run_one(condition: str, seed: int) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    directory = OUT / condition
    directory.mkdir(parents=True, exist_ok=True)
    rp = directory / f"candidate_rp_s{seed}.jsonl"
    trace = directory / f"candidate_tr_s{seed}.jsonl"
    records: list[dict] = []

    configuration = _replace(
        stress_replay.configuration_for(condition, seed, "candidate", stress_replay.stream_path(condition, seed)),
        output=str(rp),
        decision_trace=str(trace),
    )
    run = build_joint_run(configuration)
    adapter = run.task_adapter
    env = adapter.env
    original = CoalitionPlannerCoordinator._joint_redundant

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
                for member in members
                if int(member) in env.agent_dic
            }
            if str(agent_id) not in abilities:
                continue
            mine = abilities[str(agent_id)]
            others = [value for member, value in abilities.items() if member != str(agent_id)]
            covered = np.sum(others, axis=0) if others else np.zeros(requirement.shape)
            gap_without = np.maximum(requirement - covered, 0.0)
            records.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "now": float(env.current_time),
                    "agent_id": str(agent_id),
                    "task_id": task_id,
                    "stage": index,
                    "truth_marginal_contribution": float(np.minimum(gap_without, mine).sum()),
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
    counts = summary.get("status_counts", {}) or {}
    timelines = build_timelines(load_records(rp), arm="candidate", seed=seed)
    payload = {
        "condition": condition,
        "seed": seed,
        "completed": int(counts.get("completed", 0)),
        "expired": int(counts.get("expired", 0)),
        "targets": {
            timeline.target_id: {
                "progress": round(timeline.final_progress, 4),
                "reason": timeline.final_failure_reason,
                "on_time": timeline.final_progress >= 1.0
                and timeline.final_failure_reason != "deadline_expired",
            }
            for timeline in timelines.values()
        },
        # Only releases the plan actually re-decided can act; the rest are judgements.
        "executed": [
            row
            for row in records
            if (row["agent_id"], row["now"]) in _redecided(trace)
        ],
        "judged": records,
    }
    destination = directory / f"scan_s{seed}.json"
    part = destination.with_suffix(".json.part")
    part.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(part, destination)
    harmful = sum(1 for row in payload["executed"] if row["truth_marginal_contribution"] > 1e-9)
    print(
        f"[{condition} s{seed}] judged={len(records)} executed={len(payload['executed'])} "
        f"executed_lowering_real_coverage={harmful}",
        flush=True,
    )
    return payload


def _redecided(trace: Path) -> set[tuple[str, float]]:
    out: set[tuple[str, float]] = set()
    for line in trace.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event_type") == "decision" and row.get("replanned"):
            out.add((str(row["agent_id"]), float(row["timestamp"])))
    return out


def aggregate() -> int:
    stored = {}
    for condition in stress_replay.CONDITIONS:
        for path in (HERE / "stress" / condition).glob("candidate_s*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            stored[(condition, int(row["seed"]))] = row
    judged = executed = harmful = 0
    diverged: list[dict] = []
    per_condition: dict[str, dict] = {
        name: {"judged": 0, "executed": 0, "executed_lowering_real_coverage": 0, "episodes": 0}
        for name in stress_replay.CONDITIONS
    }
    worst: list[dict] = []
    for path in sorted(OUT.glob("*/scan_s*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (payload["condition"], int(payload["seed"]))
        reference = stored.get(key)
        if reference is None:
            diverged.append({"key": list(key), "why": "no stored stress artefact"})
            continue
        observed = (payload["completed"], payload["expired"], payload["targets"])
        reference_tuple = (
            int(reference["completed"]),
            int(reference["expired"]),
            reference["targets"],
        )
        if observed != reference_tuple:
            diverged.append({"key": list(key), "why": "outcome differs", "instrumented": observed[:2],
                             "stored": reference_tuple[:2]})
        bucket = per_condition[key[0]]
        bucket["episodes"] += 1
        bucket["judged"] += len(payload["judged"])
        bucket["executed"] += len(payload["executed"])
        bucket["executed_lowering_real_coverage"] += sum(
            1 for row in payload["executed"] if row["truth_marginal_contribution"] > 1e-9
        )
        judged += len(payload["judged"])
        executed += len(payload["executed"])
        harmful += sum(1 for row in payload["executed"] if row["truth_marginal_contribution"] > 1e-9)
        worst.extend(payload["executed"])
    worst.sort(key=lambda row: -row["truth_marginal_contribution"])
    payload = {
        "episodes_instrumented": sum(b["episodes"] for b in per_condition.values()),
        "episodes_reproducing_the_stored_outcome": sum(b["episodes"] for b in per_condition.values()) - len(diverged),
        "identity_divergences": diverged[:10],
        "per_condition": per_condition,
        "totals": {"judged": judged, "executed": executed, "executed_lowering_real_coverage": harmful},
        "worst_executed_releases": worst[:10],
    }
    (HERE / "STRESS_SAFETY.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "worst_executed_releases"}, indent=1))
    return 1 if diverged else 0


def main(argv: list[str]) -> int:
    if "--aggregate" in argv:
        return aggregate()
    if "--worker" in argv:
        index = int(argv[argv.index("--worker") + 1])
        total = int(argv[argv.index("--workers") + 1])
        for position, (condition, seed) in enumerate(JOBS):
            if position % total != index:
                continue
            if (OUT / condition / f"scan_s{seed}.json").exists():
                continue
            run_one(condition, seed)
        return 0
    processes = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(i), "--workers", str(WORKERS)],
            cwd=str(REPO),
        )
        for i in range(WORKERS)
    ]
    codes = [process.wait() for process in processes]
    if any(codes):
        print(f"worker failures: {codes}")
        return 1
    return aggregate()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
