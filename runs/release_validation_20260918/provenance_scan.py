"""Information provenance scan: what the release verdict actually reads, and how fresh it is.

The release decision is made inside ``_joint_redundant``, which receives the team snapshots,
the plan's committed abilities and one belief per candidate. This script wraps that one
function with a recorder -- it changes no verdict -- and replays the 15 pre-registered
development seeds with the shipped candidate switches on. For every candidate the recorder
captures what the coordinator saw (believed requirement, believed stage, revision,
timestamp) and, offline only, what the registry truth was at that same tick: the current
stage's requirement, who was really on site, and whether taking this robot away lowered the
coverage that really existed.

The offline truth is never fed to any decision; it is what makes the audit answerable.

    D:/project_env/python.exe runs/release_validation_20260918/provenance_scan.py --worker 0 --workers 8
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Sharded workers each defaulting to a multi-thread torch pool oversubscribe the box;
# single-thread inference is faster in aggregate and identical within a process.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
try:
    import torch

    torch.set_num_threads(1)
except Exception:  # torch is optional in dependency-free environments
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for path in (REPO, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

sys.path.insert(0, str(REPO / "runs" / "coalition_validation_20260918"))
from ablation_seeds import FAILURE_SEEDS, normal_seeds  # noqa: E402
from integrated_sar.coordinator import CoalitionPlannerCoordinator  # noqa: E402
from integrated_sar.expired_attribution import build_timelines  # noqa: E402
from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402
from integrated_sar.lifecycle_metrics import load_records  # noqa: E402

OUT = HERE / "provenance"
WORKERS = 4
SEEDS = FAILURE_SEEDS + normal_seeds()
INTERNAL_TO_TARGET = {}


def configuration_for(seed: int, output: Path, trace: Path) -> RunConfiguration:
    return RunConfiguration(
        seed=seed,
        steps=1000,
        device="cpu",
        output=str(output),
        decision_trace=str(trace),
        task_checkpoint=str(REPO / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth"),
        mock=False,
        coordinator_policy="coalition_planner",
        sticky_commitments=True,
        use_coordinator=True,
        release_redundant_members=True,
        suppress_recommand_own_task=True,
        residual_scale=0.25,
        allocation_interval=5,
        coordination_interval=5,
        command_ttl=10.0,
        dt=1.0,
        detection_radius=5,
        search_agents=2,
        num_goals=6,
        grid_size=60,
        obstacle_density=0.15,
        rescue_species=3,
        rescue_per_species=2,
        actor_kv_mode="tokens",
        comm_delivery_probability=1.0,
        comm_delay_ticks=0,
        comm_subscribers="all",
        quiet=True,
        replay_stream=str(
            REPO / "runs" / "coordinator_baselines_20260917" / "heldout" / "streams" / f"s{seed}.json"
        ),
    )


def run_one(seed: int) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rp = OUT / f"candidate_rp_s{seed}.jsonl"
    trace = OUT / f"candidate_tr_s{seed}.jsonl"
    records: list[dict] = []
    run = build_joint_run(configuration_for(seed, rp, trace))
    env = run.task_adapter.env
    adapter = run.task_adapter
    original = CoalitionPlannerCoordinator._joint_redundant

    def truth_for(external_task_id: str):
        internal = adapter.external_to_internal.get(str(external_task_id))
        task = env.task_dic.get(int(internal)) if internal is not None else None
        if task is None or task.get("finished"):
            return None
        index = min(int(task.get("current_stage_index", 0)), len(task["stages"]) - 1)
        stage = task["stages"][index]
        return {
            "stage_index": index,
            "requirement": [float(v) for v in stage["requirements"]],
            "status": [float(v) for v in stage["status"]],
            "members": [str(member) for member in task["members"]],
            "location": [float(task["location"][0]), float(task["location"][1])],
        }

    def believed_gross(agent_id: str, external_task_id: str):
        """The stage requirement as this robot was told it, before anyone's abilities."""
        internal = adapter.external_to_internal.get(str(external_task_id))
        if internal is None:
            return None
        knowledge = env.agent_knowledge.get(int(agent_id)) or {}
        value = knowledge.get("task_requirements", {}).get(internal)
        return None if value is None else [float(item) for item in value]

    def believed_status(agent_id: str, external_task_id: str):
        internal = adapter.external_to_internal.get(str(external_task_id))
        if internal is None:
            return None
        knowledge = env.agent_knowledge.get(int(agent_id)) or {}
        value = knowledge.get("task_current_status", {}).get(internal)
        return None if value is None else [float(item) for item in value]

    def recording(agents, committed, checks):
        released = original(agents, committed, checks)
        for agent_id, belief in checks.items():
            task_id = agents[agent_id].current_task
            truth = truth_for(task_id or "")
            entry = {
                "seed": seed,
                "now": float(env.current_time),
                "agent_id": str(agent_id),
                "task_id": task_id,
                "status": agents[agent_id].status,
                "believed_requirement": [float(v) for v in belief.remaining_requirements],
                "believed_stage_index": int(belief.stage_index),
                "believed_future": [float(v) for v in belief.future_requirements],
                "believed_revision": int(belief.revision),
                "believed_updated_at": float(belief.updated_at),
                "believed_members": [str(member) for member in belief.members],
                "believed_finished": bool(belief.finished),
                "candidate_count": len(checks),
                "believed_gross_requirement": believed_gross(agent_id, task_id or ""),
                "pool_abilities": {
                    member: [float(v) for v in agents[member].abilities]
                    for member in sorted(
                        key
                        for key, abilities in committed.get(task_id or "", {}).items()
                        if key in agents and agents[key].status == "working"
                    )
                },
                "onsite_pool": sorted(
                    key
                    for key, abilities in committed.get(task_id or "", {}).items()
                    if key in agents and agents[key].status == "working"
                ),
                "released": agent_id in released,
            }
            if truth is not None:
                requirement = np.array(truth["requirement"], dtype=float)
                pool = [member for member in truth["members"] if member in agents]
                abilities = {
                    member: np.array(agents[member].abilities, dtype=float) for member in pool
                }
                others = [abilities[member] for member in pool if member != agent_id]
                covered = np.sum(others, axis=0) if others else np.zeros(requirement.shape)
                gap_without = np.maximum(requirement - covered, 0.0)
                mine = np.array(agents[agent_id].abilities, dtype=float)
                position = agents[agent_id].position
                entry["truth"] = truth
                # Would taking this robot away lower the coverage that really exists?
                entry["truth_marginal_contribution"] = float(
                    np.minimum(gap_without, mine).sum()
                )
                entry["belief_lag_stages"] = int(entry["believed_stage_index"] - truth["stage_index"])
                entry["believed_current_stage_satisfied"] = bool(
                    sum(entry["believed_requirement"]) <= 0
                )
                entry["candidate_position"] = [float(position.x), float(position.y)]
                entry["distance_to_task"] = float(
                    np.hypot(position.x - truth["location"][0], position.y - truth["location"][1])
                )
                entry["in_truth_members"] = str(agent_id) in truth["members"]
            records.append(entry)
        return released

    CoalitionPlannerCoordinator._joint_redundant = staticmethod(recording)
    try:
        summary = run.run()
    finally:
        CoalitionPlannerCoordinator._joint_redundant = original
    counts = summary.get("status_counts", {}) or {}
    payload = {
        "seed": seed,
        "completed": int(counts.get("completed", 0)),
        "expired": int(counts.get("expired", 0)),
        "active": sum(value for key, value in counts.items() if key not in {"completed", "expired"}),
        "status_counts": counts,
        "checks": records,
    }
    destination = OUT / f"scan_s{seed}.json"
    timelines = build_timelines(load_records(rp), arm="candidate", seed=seed)
    payload["targets"] = {
        timeline.target_id: {
            "progress": round(timeline.final_progress, 4),
            "reason": timeline.final_failure_reason,
            "on_time": timeline.final_progress >= 1.0
            and timeline.final_failure_reason != "deadline_expired",
        }
        for timeline in timelines.values()
    }
    part = destination.with_suffix(".json.part")
    part.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(part, destination)
    releases = sum(1 for row in records if row["released"])
    print(f"[s{seed}] checks={len(records)} releases={releases} completed={payload['completed']}", flush=True)
    return payload


def analyse() -> int:
    ablation = json.loads(
        (REPO / "runs" / "coalition_validation_20260918" / "ABLATION.json").read_text(encoding="utf-8")
    )
    shipped = {
        int(row["seed"]): (int(row["completed"]), int(row["expired"]), row.get("targets"))
        for row in ablation["rows"]
        if row["arm"] == "D_both"
    }
    rows: list[dict] = []
    identity: dict[str, list] = {"matches": [], "differs": []}
    for path in sorted(OUT.glob("scan_s*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        observed = (int(payload["completed"]), int(payload["expired"]), payload.get("targets"))
        bucket = "matches" if shipped.get(seed) == observed else "differs"
        if bucket == "differs":
            identity[bucket].append({"seed": seed, "instrumented": observed[:2], "shipped": (
                shipped.get(seed) or (None, None, None))[:2]})
        else:
            identity["matches"].append(seed)
        # A judgement becomes an action only when the robot is actually re-decided.
        redecided = {
            (str(row["agent_id"]), float(row["timestamp"]))
            for row in (
                json.loads(line)
                for line in (OUT / f"candidate_tr_s{seed}.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("event_type") == "decision"
            )
            if row.get("replanned")
        }
        for record in payload["checks"]:
            record["redecided_this_tick"] = (record["agent_id"], record["now"]) in redecided
            rows.append(record)
    released = [row for row in rows if row["released"]]
    categories = Counter()
    for row in rows:
        if not row.get("truth"):
            categories["belief_without_a_live_task"] += 1
            continue
        if row["believed_current_stage_satisfied"]:
            categories["believed_current_stage_satisfied"] += 1
        if row["belief_lag_stages"] != 0:
            categories["believed_stage_differs_from_truth"] += 1
        if not row["in_truth_members"]:
            categories["working_but_not_a_registry_member"] += 1
        if row.get("distance_to_task", 0.0) > 1e-6:
            categories["working_but_not_at_the_task_location"] += 1
    for row in released:
        executed = row.get("redecided_this_tick")
        prefix = "EXECUTED" if executed else "VERDICT_ONLY"
        if row.get("truth_marginal_contribution", 0.0) > 1e-9:
            categories[f"{prefix}_reduced_real_coverage"] += 1
        if row.get("believed_current_stage_satisfied"):
            categories[f"{prefix}_on_a_satisfied_believed_stage"] += 1
        if row.get("belief_lag_stages", 0) != 0:
            categories[f"{prefix}_with_a_lagged_stage_belief"] += 1
    categories["redecided_this_tick"] = sum(1 for row in released if row.get("redecided_this_tick"))
    summary = {
        "identity_with_the_shipped_arm": identity,
        "seeds": sorted({row["seed"] for row in rows}),
        "release_candidates_examined": len(rows),
        "releases_taken": len(released),
        "categories": dict(categories),
        "release_records": sorted(
            released,
            key=lambda row: (
                bool(row.get("redecided_this_tick")),
                row.get("truth_marginal_contribution", 0.0),
            ),
            reverse=True,
        )[:40],
        "worst_real_coverage_change_on_a_release": min(
            [row.get("truth_marginal_contribution", 0.0) for row in released] or [0.0]
        ),
        "worst_real_coverage_change_on_an_executed_release": min(
            [
                row.get("truth_marginal_contribution", 0.0)
                for row in released
                if row.get("redecided_this_tick")
            ]
            or [0.0]
        ),
    }
    (HERE / "PROVENANCE_SCAN.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "release_records"}, indent=1))
    if identity["differs"]:
        print("IDENTITY FAILURE: the recorder changed the trajectory", identity["differs"])
        return 1
    return 0


def main(argv: list[str]) -> int:
    if "--worker" in argv:
        index = int(argv[argv.index("--worker") + 1])
        total = int(argv[argv.index("--workers") + 1])
        for seed in SEEDS[index::total]:
            if not (OUT / f"scan_s{seed}.json").exists():
                run_one(seed)
        return 0
    if "--analyse" in argv:
        return analyse()
    OUT.mkdir(parents=True, exist_ok=True)
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
    return analyse()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
