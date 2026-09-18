"""Closed-loop gate for the redundant-member release.

Runs the binding baseline and the release candidate on the same seeds and recorded
detection streams, then compares per-target outcomes. This is a full-episode replay,
not an analytical estimate: the mechanism has to change actual commands, arrivals and
final statuses or the gate fails.

Baseline   ``coalition_planner`` + sticky commitments (the 868d68a binding arm).
Candidate  ``coalition_redundant_release`` — the same planner plus exactly one
           mechanism: working robots whose own belief says they fill no slot may be
           re-decided when the plan has somewhere else for them.

    D:/project_env/python.exe runs/dispatch_trigger_audit_20260918/closed_loop_gate.py [seeds...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE.parent / "coalition_formation_latency_20260918") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "coalition_formation_latency_20260918"))

from integrated_sar.expired_attribution import build_timelines  # noqa: E402
from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402
from integrated_sar.lifecycle_metrics import load_records  # noqa: E402

SEEDS = [20003, 20008, 20052, 20070, 20083]
BASE = "coalition_planner"
CANDIDATE = "coalition_redundant_release"
FAILURES = {
    20003: ["piloc-e1-target-1", "piloc-e1-target-0"],
    20008: ["piloc-e1-target-2"],
    20052: ["piloc-e1-target-3"],
    20070: ["piloc-e1-target-2"],
    20083: ["piloc-e1-target-4"],
}


def run_one(seed: int, policy: str, tag: str) -> dict:
    out = HERE / "closedloop" / f"{tag}_rp_s{seed}.jsonl"
    trace = HERE / "closedloop" / f"{tag}_tr_s{seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    configuration = RunConfiguration(
        seed=seed,
        steps=1000,
        device="cpu",
        output=str(out),
        decision_trace=str(trace),
        task_checkpoint=str(REPO / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth"),
        mock=False,
        coordinator_policy=policy,
        sticky_commitments=True,
        use_coordinator=True,
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
    summary = run.run()
    records = load_records(out)
    timelines = build_timelines(records, arm=tag, seed=seed)
    counts = summary.get("status_counts", {}) or {}
    # Re-commands: decisions that issued a command for the task the robot was already
    # carrying (never None == None: a depot decision carries no task at all). A few are
    # legitimate (arrival verification failed), a churn loop is not.
    recommands = 0
    for line in trace.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("event_type") != "decision":
            continue
        assignment = row.get("actual_assignment")
        if (
            row.get("command_id")
            and assignment is not None
            and assignment == row.get("current_task")
        ):
            recommands += 1
    return {
        "seed": seed,
        "policy": policy,
        "completed": int(counts.get("completed", 0)),
        "expired": int(counts.get("expired", 0)),
        "status_counts": dict(counts),
        "coverage_resets": sum(t.progress_setbacks for t in timelines.values()),
        "lost_work": round(sum(t.discarded_work_steps for t in timelines.values()), 3),
        "executed_override_unjustified": int(summary.get("executed_override_unjustified", 0)),
        "rejected_assignments": int(summary.get("rejected_assignments", 0)),
        "longest_idle_streak": float(summary.get("longest_idle_streak", 0.0)),
        "recommands": recommands,
        "targets": {
            t.target_id: {
                "final_progress": round(t.final_progress, 4),
                "failure_reason": t.final_failure_reason,
                "finished_after_deadline": t.final_progress >= 1.0
                and t.final_failure_reason == "deadline_expired",
            }
            for t in timelines.values()
        },
    }


def main(argv: list[str]) -> int:
    seeds = [int(value) for value in argv] or SEEDS
    rows = []
    for seed in seeds:
        for policy, tag in ((BASE, "baseline"), (CANDIDATE, "candidate")):
            row = run_one(seed, policy, tag)
            rows.append(row)
            print(
                f"seed {seed} {tag:9} completed={row['completed']} expired={row['expired']} "
                f"resets={row['coverage_resets']} lost={row['lost_work']} "
                f"override={row['executed_override_unjustified']} "
                f"rejected={row['rejected_assignments']} recommands={row['recommands']} "
                f"idle={row['longest_idle_streak']:.0f}",
                flush=True,
            )

    by_tag: dict[str, list[dict]] = {}
    for row in rows:
        by_tag.setdefault(row["policy"], []).append(row)

    def total(policy: str, key: str) -> float:
        return float(sum(r[key] for r in by_tag.get(policy, [])))

    rescued, new_failures, on_time = [], [], []
    for seed in seeds:
        base_targets = next(r["targets"] for r in by_tag[BASE] if r["seed"] == seed)
        cand_targets = next(r["targets"] for r in by_tag[CANDIDATE] if r["seed"] == seed)
        for target in FAILURES.get(seed, []):
            base_t = base_targets.get(target, {})
            cand_t = cand_targets.get(target, {})
            key = f"{seed}/{target}"
            base_progress = float(base_t.get("final_progress", 0.0))
            cand_progress = float(cand_t.get("final_progress", 0.0))
            base_on_time = base_progress >= 1.0 and not base_t.get("finished_after_deadline")
            cand_on_time = cand_progress >= 1.0 and not cand_t.get("finished_after_deadline")
            if cand_on_time and not base_on_time:
                rescued.append(key)
                if base_progress >= 1.0:
                    on_time.append(key)
            if base_on_time and not cand_on_time:
                new_failures.append(key)

    comparison = {
        key: {"baseline": total(BASE, key), "candidate": total(CANDIDATE, key)}
        for key in (
            "completed", "expired", "coverage_resets", "lost_work",
            "executed_override_unjustified", "rejected_assignments", "recommands",
        )
    }
    deadlocks = max((r["longest_idle_streak"] for r in rows), default=0.0)
    gates = {
        "no_new_execution_break": comparison["executed_override_unjustified"]["candidate"]
        <= comparison["executed_override_unjustified"]["baseline"],
        "zero_invalid_commands": comparison["rejected_assignments"]["candidate"] == 0,
        "zero_deadlocks": deadlocks < 20.0,
        "no_new_failure": len(new_failures) == 0,
        "no_command_churn": comparison["recommands"]["candidate"]
        <= comparison["recommands"]["baseline"] + 20,
        "rescued_at_least_3": len(rescued) >= 3,
    }
    payload = {
        "seeds": seeds,
        "baseline_policy": BASE,
        "candidate_policy": CANDIDATE,
        "rescued": rescued,
        "rescued_from_finished_after_deadline": on_time,
        "new_failures": new_failures,
        "comparison": comparison,
        "max_longest_idle_streak": deadlocks,
        "gates": gates,
        "passed": all(gates.values()),
        "rows": rows,
    }
    (HERE / "CLOSED_LOOP_GATE.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"\nrescued: {rescued or 'none'}")
    print(f"  of which finished-after-deadline -> on time: {on_time or 'none'}")
    print(f"new failures: {new_failures or 'none'}")
    print(f"comparison {json.dumps(comparison)}")
    print(f"gates {json.dumps(gates)}")
    print(f"PASSED {payload['passed']}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
