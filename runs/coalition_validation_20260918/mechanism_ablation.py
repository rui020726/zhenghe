"""Mechanism ablation on the pre-registered development seed set (brief section two).

Arms, all on one frozen source with binding commitments and the same recorded streams:

  A  binding baseline                    (both switches off)
  B  duplicate-command suppression only  (suppression on, planner release off)
  C  redundant-member replanning only    (release on, suppression off)
  D  release + suppression               (the candidate)

Every run is instrumented read-only, because the brief's metrics -- zero-marginal-
contribution working ticks, reassignment attempts and their latency -- are not in the
committed logs. Outputs land in ``ablation/<arm>_s<seed>.json`` after an atomic rename;
a rerun skips what is already there.

    D:/project_env/python.exe runs/coalition_validation_20260918/mechanism_ablation.py [seeds...]
    ... mechanism_ablation.py --worker 0 --workers 4        (a shard, used by --run-all)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ablation_seeds import FAILURE_SEEDS, normal_seeds  # noqa: E402
from integrated_sar.expired_attribution import build_timelines  # noqa: E402
from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402
from integrated_sar.lifecycle_metrics import load_records  # noqa: E402

ARMS = {
    "A_baseline": {"policy": "coalition_planner", "release": False, "suppress": False},
    "B_suppress": {"policy": "coalition_planner", "release": False, "suppress": True},
    "C_release": {"policy": "coalition_planner", "release": True, "suppress": False},
    "D_both": {"policy": "coalition_planner", "release": True, "suppress": True},
}
OUT = HERE / "ablation"


def run_arm(seed: int, arm: str) -> dict:
    spec = ARMS[arm]
    rp = OUT / f"{arm}_rp_s{seed}.jsonl"
    trace = OUT / f"{arm}_tr_s{seed}.jsonl"
    summary_path = OUT / f"{arm}_s{seed}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    configuration = RunConfiguration(
        seed=seed,
        steps=1000,
        device="cpu",
        output=str(rp),
        decision_trace=str(trace),
        task_checkpoint=str(REPO / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth"),
        mock=False,
        coordinator_policy=spec["policy"],
        sticky_commitments=True,
        use_coordinator=True,
        release_redundant_members=spec["release"],
        suppress_recommand_own_task=spec["suppress"],
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
    run = build_joint_run(configuration)
    env = run.task_adapter.env
    counters = {
        "releases_planned": 0,
        "redundant_working_ticks": 0,
        "working_ticks": 0,
    }
    original_step = run.orchestrator.step

    def step():
        report = original_step()
        abilities = {str(aid): np.array(agent["abilities"], dtype=float) for aid, agent in env.agent_dic.items()}
        for task in env.task_dic.values():
            members = [str(m) for m in task["members"]]
            requirement = np.array(task["stages"][min(int(task["current_stage_index"]), len(task["stages"]) - 1)]["requirements"], dtype=float)
            for member in members:
                if member not in abilities or env.agent_dic.get(int(member)) is None:
                    continue
                if env.agent_dic[int(member)].get("status") != "working":
                    continue
                others = [abilities[m] for m in members if m != member and m in abilities]
                covered = np.sum(others, axis=0) if others else np.zeros(requirement.shape)
                gap_without = np.maximum(requirement - covered, 0.0)
                counters["working_ticks"] += 1
                if float(np.minimum(gap_without, abilities[member]).sum()) <= 1e-9:
                    counters["redundant_working_ticks"] += 1
        counters["releases_planned"] += int(
            (report.advice.metadata or {}).get("redundant_releases", 0) or 0
        )
        return report

    run.orchestrator.step = step
    summary = run.run()
    metrics = measure(seed, arm, rp, trace, summary, counters)
    part = summary_path.with_suffix(".json.part")
    part.write_text(json.dumps(metrics, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(part, summary_path)
    return metrics


def measure(seed: int, arm: str, rp: Path, trace: Path, summary: dict, counters: dict) -> dict:
    rows = [
        json.loads(line)
        for line in trace.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("event_type") == "decision"
    ]
    replan_attempts = sum(1 for r in rows if r.get("replanned"))
    accepted = [
        r for r in rows
        if r.get("replanned") and r.get("command_id") and r.get("actual_assignment") != r.get("current_task")
    ]
    declined = [r for r in rows if r.get("replanned") and not r.get("command_id")]
    same_task_noop = [
        r for r in rows
        if r.get("command_id") is None and r.get("own_task_suppressed")
    ]
    recommands = [
        r for r in rows
        if r.get("command_id")
        and r.get("actual_assignment") is not None
        and r.get("actual_assignment") == r.get("current_task")
    ]
    # replan -> acceptance latency per robot, in ticks.
    latencies: list[float] = []
    waiting: dict[str, float] = {}
    for r in sorted(rows, key=lambda row: (row["agent_id"], row["step"])):
        agent = r["agent_id"]
        if r.get("replanned") and agent not in waiting:
            waiting[agent] = float(r["step"])
        elif agent in waiting:
            if r.get("command_id") and r.get("actual_assignment") != r.get("current_task"):
                latencies.append(float(r["step"]) - waiting.pop(agent))
            elif not r.get("replanned"):
                waiting.pop(agent, None)
    records = load_records(rp)
    timelines = build_timelines(records, arm=arm, seed=seed)
    counts = summary.get("status_counts", {}) or {}
    return {
        "seed": seed,
        "arm": arm,
        "completed": int(counts.get("completed", 0)),
        "expired": int(counts.get("expired", 0)),
        "active": sum(v for k, v in counts.items() if k not in {"completed", "expired"}),
        "coverage_resets": int(sum(t.progress_setbacks for t in timelines.values())),
        "lost_work": round(float(sum(t.discarded_work_steps for t in timelines.values())), 3),
        "executed_override_unjustified": int(summary.get("executed_override_unjustified", 0)),
        "rejected_assignments": int(summary.get("rejected_assignments", 0)),
        "longest_idle_streak": float(summary.get("longest_idle_streak", 0.0)),
        "releases_planned": counters["releases_planned"],
        "working_ticks": counters["working_ticks"],
        "redundant_working_ticks": counters["redundant_working_ticks"],
        "replan_attempts": replan_attempts,
        "accepted_reassignments": len(accepted),
        "replan_declined": len(declined),
        "same_task_noop": len(same_task_noop),
        "recommands": len(recommands),
        "replan_to_accept_latency_mean": round(float(np.mean(latencies)), 2) if latencies else None,
        "targets": {
            t.target_id: {
                "progress": round(t.final_progress, 4),
                "reason": t.final_failure_reason,
                "on_time": t.final_progress >= 1.0 and t.final_failure_reason != "deadline_expired",
            }
            for t in timelines.values()
        },
    }


def main(argv: list[str]) -> int:
    import subprocess

    seeds = FAILURE_SEEDS + normal_seeds()
    if "--worker" in argv:
        index = int(argv[argv.index("--worker") + 1])
        total = int(argv[argv.index("--workers") + 1])
        jobs = [(seed, arm) for seed in seeds for arm in ARMS]
        for seed, arm in jobs[index::total]:
            if (OUT / f"{arm}_s{seed}.json").exists():
                continue
            metrics = run_arm(seed, arm)
            print(
                f"[{arm} s{seed}] completed={metrics['completed']} expired={metrics['expired']} "
                f"redundant_ticks={metrics['redundant_working_ticks']} recommands={metrics['recommands']} "
                f"noop={metrics['same_task_noop']} accepted={metrics['accepted_reassignments']}",
                flush=True,
            )
        return 0

    if "--aggregate" not in argv:
        processes = [
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(i), "--workers", "5"], cwd=str(REPO))
            for i in range(5)
        ]
        codes = [p.wait() for p in processes]
        if any(codes):
            print(f"worker failures: {codes}")
            return 1

    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*_s*.json"))]
    table: dict[str, dict] = {}
    for arm in ARMS:
        arms = [r for r in rows if r["arm"] == arm]
        numeric = (
            "completed", "expired", "active", "coverage_resets", "lost_work",
            "executed_override_unjustified", "rejected_assignments", "recommands",
            "releases_planned", "redundant_working_ticks", "replan_attempts",
            "accepted_reassignments", "replan_declined", "same_task_noop",
        )
        table[arm] = {key: round(float(sum(r[key] for r in arms)), 3) for key in numeric}
        lat = [r["replan_to_accept_latency_mean"] for r in arms if r["replan_to_accept_latency_mean"]]
        table[arm]["replan_to_accept_latency_mean"] = round(float(np.mean(lat)), 2) if lat else None
        table[arm]["max_idle"] = round(max((r["longest_idle_streak"] for r in arms), default=0.0), 2)

    baseline = {r["seed"]: r for r in rows if r["arm"] == "A_baseline"}
    per_arm_outcomes: dict[str, dict] = {}
    for arm in ARMS:
        rescued, regressed = [], []
        for r in [x for x in rows if x["arm"] == arm]:
            base = baseline.get(r["seed"])
            if not base:
                continue
            for target, fact in r["targets"].items():
                before = base["targets"].get(target, {})
                if fact["on_time"] and not before.get("on_time"):
                    rescued.append(f"{r['seed']}/{target}")
                if before.get("on_time") and not fact["on_time"]:
                    regressed.append(f"{r['seed']}/{target}")
        per_arm_outcomes[arm] = {"rescued": sorted(set(rescued)), "regressed": sorted(set(regressed))}

    payload = {"seeds": seeds, "arms": table, "outcomes_vs_A": per_arm_outcomes, "rows": rows}
    (HERE / "ABLATION.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for arm, metrics in table.items():
        print(arm, json.dumps(metrics))
        print("   ", per_arm_outcomes[arm]["rescued"][:4], "regressed:", per_arm_outcomes[arm]["regressed"])
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
