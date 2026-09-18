"""Live-system closed loop (brief section six): real PILOC search, real CTDE allocation.

60 fresh seeds, two arms on one frozen source (baseline vs release + suppression), the
same scenario seed per pair but **no shared detection stream**: the search policy's own
path decides what is discovered and when, so allocation feedback into discovery is part
of what this stage measures. The replay experiment cannot substitute for it.

    D:/project_env/python.exe runs/coalition_validation_20260918/system_closed_loop.py
    ... system_closed_loop.py --worker 0 --workers 6
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
try:
    import torch

    torch.set_num_threads(1)
except Exception:  # torch is optional in dependency-free environments
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from integrated_sar.expired_attribution import build_timelines  # noqa: E402
from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402
from integrated_sar.lifecycle_metrics import load_records  # noqa: E402

import mechanism_ablation as ablation  # noqa: E402
import heldout_record  # noqa: E402

PIL_CHECKPOINT = REPO / "runs" / "cur_v2rep05_s2" / "best" / "shared_mappo_model_21_3100.pth"
CTDE_CHECKPOINT = REPO / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth"
OUT = HERE / "systemloop"
ARMS = {
    "baseline": {"release": False, "suppress": False},
    "candidate": {"release": True, "suppress": True},
}
NUM_GOALS = 6
WORKERS = 6


def seeds() -> list[int]:
    return [int(seed) for seed in json.loads(
        (HERE / "SYSTEM_SEEDS.json").read_text(encoding="utf-8")
    )["seeds"]]


def run_one(seed: int, arm: str) -> dict:
    spec = ARMS[arm]
    OUT.mkdir(parents=True, exist_ok=True)
    rp = OUT / f"{arm}_rp_s{seed}.jsonl"
    trace = OUT / f"{arm}_tr_s{seed}.jsonl"
    configuration = RunConfiguration(
        seed=seed,
        steps=1000,
        device="cpu",
        output=str(rp),
        decision_trace=str(trace),
        piloc_checkpoint=str(PIL_CHECKPOINT),
        task_checkpoint=str(CTDE_CHECKPOINT),
        mock=False,
        coordinator_policy="coalition_planner",
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
        num_goals=NUM_GOALS,
        grid_size=60,
        obstacle_density=0.15,
        rescue_species=3,
        rescue_per_species=2,
        actor_kv_mode="tokens",
        comm_delivery_probability=1.0,
        comm_delay_ticks=0,
        comm_subscribers="all",
        quiet=True,
    )
    counters = {"releases_planned": 0, "redundant_working_ticks": 0, "working_ticks": 0}
    started = time.time()
    run = build_joint_run(configuration)
    env = run.task_adapter.env
    original_step = run.orchestrator.step

    def step():
        report = original_step()
        abilities = {str(aid): np.array(agent["abilities"], dtype=float) for aid, agent in env.agent_dic.items()}
        for task in env.task_dic.values():
            members = [str(m) for m in task["members"]]
            index = min(int(task["current_stage_index"]), len(task["stages"]) - 1)
            requirement = np.array(task["stages"][index]["requirements"], dtype=float)
            for member in members:
                agent = env.agent_dic.get(int(member)) if member.isdigit() else None
                if agent is None or agent.get("status") != "working" or member not in abilities:
                    continue
                others = [abilities[m] for m in members if m != member and m in abilities]
                covered = np.sum(others, axis=0) if others else np.zeros(requirement.shape)
                gap_without = np.maximum(requirement - covered, 0.0)
                counters["working_ticks"] += 1
                if float(np.minimum(gap_without, abilities[member]).sum()) <= 1e-9:
                    counters["redundant_working_ticks"] += 1
        counters["releases_planned"] += int((report.advice.metadata or {}).get("redundant_releases", 0) or 0)
        return report

    run.orchestrator.step = step
    summary = run.run()
    wall = round(time.time() - started, 2)
    metrics = ablation.measure(seed, arm, rp, trace, summary, counters)
    records = load_records(rp)
    timelines = build_timelines(records, arm=arm, seed=seed)
    discovered = sorted({t.target_id for t in timelines.values()})
    on_time = [t.target_id for t in timelines.values() if t.final_progress >= 1.0 and t.final_failure_reason != "deadline_expired"]
    expired = [t.target_id for t in timelines.values() if t.final_failure_reason == "deadline_expired"]
    metrics.update(
        {
            "goals_placed": NUM_GOALS,
            "discovered": len(discovered),
            "discovered_ids": discovered,
            "search_coverage": round(len(discovered) / NUM_GOALS, 4),
            "on_time": len(on_time),
            "expired_ids": expired,
            "completion_rate_of_discovered": round(len(on_time) / len(discovered), 4) if discovered else None,
            "rescue_travel": round(float(sum(agent.get("travel_dist", 0.0) for agent in env.agent_dic.values())), 3),
            "rescue_occupied_ticks": int(metrics["working_ticks"]),
            "wall_seconds": wall,
        }
    )
    part = (OUT / f"{arm}_s{seed}.json").with_suffix(".json.part")
    part.write_text(json.dumps(metrics, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(part, OUT / f"{arm}_s{seed}.json")
    return metrics


def main(argv: list[str]) -> int:
    all_seeds = seeds()
    if "--worker" in argv:
        index = int(argv[argv.index("--worker") + 1])
        total = int(argv[argv.index("--workers") + 1])
        jobs = [(seed, arm) for seed in all_seeds for arm in ARMS]
        for position, (seed, arm) in enumerate(jobs):
            if position % total != index or (OUT / f"{arm}_s{seed}.json").exists():
                continue
            metrics = run_one(seed, arm)
            print(
                f"[{arm} s{seed}] coverage={metrics['search_coverage']} discovered={metrics['discovered']} "
                f"on_time={metrics['on_time']} expired={len(metrics['expired_ids'])} "
                f"redundant={metrics['redundant_working_ticks']} recommands={metrics['recommands']} "
                f"rejected={metrics['rejected_assignments']} {metrics['wall_seconds']}s",
                flush=True,
            )
        return 0

    processes = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(i), "--workers", str(WORKERS)],
            cwd=str(REPO),
        )
        for i in range(WORKERS)
    ]
    codes = [p.wait() for p in processes] if "--aggregate" not in argv else []
    if any(codes):
        print(f"worker failures: {codes}")
        return 1

    rows = {arm: [] for arm in ARMS}
    for path in sorted(OUT.glob("*_s*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        rows[row["arm"]].append(row)
    summary = {}
    for arm, values in rows.items():
        def total(key: str) -> float:
            return float(sum(v[key] for v in values))

        summary[arm] = {
            "episodes": len(values),
            "goals_placed": int(total("goals_placed")),
            "discovered": int(total("discovered")),
            "search_coverage": round(total("discovered") / max(1.0, total("goals_placed")), 4),
            "on_time_completed": int(total("on_time")),
            "expired": int(sum(len(v["expired_ids"]) for v in values)),
            "completion_rate_of_discovered": round(
                total("on_time") / max(1.0, total("discovered")), 4
            ),
            "redundant_working_ticks": int(total("redundant_working_ticks")),
            "releases_planned": int(total("releases_planned")),
            "accepted_reassignments": int(total("accepted_reassignments")),
            "replan_declined": int(total("replan_declined")),
            "same_task_noop": int(total("same_task_noop")),
            "recommands": int(total("recommands")),
            "rejected_assignments": int(total("rejected_assignments")),
            "executed_override_unjustified": int(total("executed_override_unjustified")),
            "coverage_resets": int(total("coverage_resets")),
            "lost_work": round(total("lost_work"), 3),
            "rescue_travel": round(total("rescue_travel"), 1),
            "wall_seconds": round(total("wall_seconds"), 1),
        }
    paired = []
    by_seed = {arm: {row["seed"]: row for row in rows[arm]} for arm in ARMS}
    for seed in all_seeds:
        base = by_seed["baseline"].get(seed)
        cand = by_seed["candidate"].get(seed)
        if not base or not cand:
            continue
        paired.append(
            {
                "seed": seed,
                "discovered_baseline": base["discovered"],
                "discovered_candidate": cand["discovered"],
                "on_time_baseline": base["on_time"],
                "on_time_candidate": cand["on_time"],
                "rate_baseline": base["completion_rate_of_discovered"],
                "rate_candidate": cand["completion_rate_of_discovered"],
            }
        )
    # The same estimator the held-out protocol fixed: seed-level percentile bootstrap.
    usable = [
        row for row in paired
        if row["discovered_baseline"] and row["discovered_candidate"]
        and row["rate_baseline"] is not None and row["rate_candidate"] is not None
    ]
    differences = np.array([row["rate_candidate"] - row["rate_baseline"] for row in usable], dtype=float)
    if differences.size:
        rng = np.random.default_rng(20260918)
        draws = rng.integers(0, differences.size, size=(10_000, differences.size))
        means = differences[draws].mean(axis=1)
        interval = [round(float(np.percentile(means, 2.5)), 6), round(float(np.percentile(means, 97.5)), 6)]
        mean_difference = round(float(differences.mean()), 6)
    else:
        interval, mean_difference = [0.0, 0.0], 0.0
    payload = {
        "seeds": all_seeds,
        "arms": summary,
        "paired": paired,
        "paired_completion_rate_of_discovered": {
            "episodes_in_test": int(differences.size),
            "mean_difference": mean_difference,
            "bootstrap_ci95": interval,
            "bootstrap_resamples": 10_000,
            "bootstrap_rng_seed": 20260918,
            "episodes_where_discovery_differed": sum(
                1 for row in paired if row["discovered_baseline"] != row["discovered_candidate"]
            ),
            "episodes_candidate_better": sum(1 for row in paired if row["on_time_candidate"] > row["on_time_baseline"]),
            "episodes_candidate_worse": sum(1 for row in paired if row["on_time_candidate"] < row["on_time_baseline"]),
        },
    }
    (HERE / "SYSTEM_LOOP.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    for arm, facts in summary.items():
        print(arm, json.dumps(facts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
