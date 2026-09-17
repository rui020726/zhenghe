"""Phase 12: the small-scale gate for the assembly-scheduler prototype.

Runs the binding baseline and the prototype on the same seeds and compares completed,
expired, coverage resets, lost work, override justifications, illegal commands and
deadlocks. Analysis only; writes its verdict to ASSEMBLY_GATE.json.

    D:/project_env/python.exe runs/coalition_formation_latency_20260918/assembly_gate.py [seeds...]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from integrated_sar.expired_attribution import build_timelines  # noqa: E402
from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402
from integrated_sar.lifecycle_metrics import load_records  # noqa: E402

SEEDS = [20003, 20008, 20052, 20070, 20083]
BASE = "coalition_planner"
CANDIDATE = "coalition_assembly_scheduler"


def run_one(seed: int, policy: str, tag: str) -> dict:
    out = HERE / "gate" / f"{tag}_s{seed}.jsonl"
    trace = HERE / "gate" / f"{tag}_tr_s{seed}.jsonl"
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
    resets = sum(t.progress_setbacks for t in timelines.values())
    lost = sum(t.discarded_work_steps for t in timelines.values())
    # ``summarize`` reports status counts, not flat completed/expired keys.
    counts = summary.get("status_counts", {}) or {}
    return {
        "seed": seed,
        "policy": policy,
        "completed": int(counts.get("completed", 0)),
        "expired": int(counts.get("expired", 0)),
        "status_counts": dict(counts),
        "coverage_resets": resets,
        "lost_work": round(lost, 3),
        "executed_override_unjustified": int(summary.get("executed_override_unjustified", 0)),
        "rejected_assignments": int(summary.get("rejected_assignments", 0)),
        "longest_idle_streak": float(summary.get("longest_idle_streak", 0.0)),
        "commitment_honour_rate": summary.get("commitment_honour_rate"),
        "commitment_honour_denominator": summary.get("commitment_honour_denominator"),
        "targets": {
            t.target_id: {
                "final_progress": t.final_progress,
                "failure_reason": t.final_failure_reason,
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
                f"rejected={row['rejected_assignments']} idle={row['longest_idle_streak']:.0f}",
                flush=True,
            )

    by_tag: dict[str, list[dict]] = {}
    for row in rows:
        by_tag.setdefault(row["policy"], []).append(row)

    def total(policy: str, key: str) -> float:
        return float(sum(r[key] for r in by_tag.get(policy, [])))

    base, cand = BASE, CANDIDATE
    comparison = {
        "completed": {"baseline": total(base, "completed"), "candidate": total(cand, "completed")},
        "expired": {"baseline": total(base, "expired"), "candidate": total(cand, "expired")},
        "coverage_resets": {"baseline": total(base, "coverage_resets"), "candidate": total(cand, "coverage_resets")},
        "lost_work": {"baseline": total(base, "lost_work"), "candidate": total(cand, "lost_work")},
        "executed_override_unjustified": {
            "baseline": total(base, "executed_override_unjustified"),
            "candidate": total(cand, "executed_override_unjustified"),
        },
        "rejected_assignments": {"baseline": total(base, "rejected_assignments"), "candidate": total(cand, "rejected_assignments")},
    }
    deadlocks = max((r["longest_idle_streak"] for r in rows), default=0.0)
    gates = {
        "no_new_execution_break": comparison["executed_override_unjustified"]["candidate"]
        <= comparison["executed_override_unjustified"]["baseline"],
        "zero_invalid_commands": comparison["rejected_assignments"]["candidate"] == 0,
        "zero_deadlocks": deadlocks < 20.0,
        "no_legacy_regression": comparison["completed"]["candidate"] >= comparison["completed"]["baseline"],
        "rescued_at_least_3": (comparison["completed"]["candidate"] - comparison["completed"]["baseline"]) >= 3,
    }
    payload = {
        "seeds": seeds,
        "comparison": comparison,
        "max_longest_idle_streak": deadlocks,
        "gates": gates,
        "passed": all(gates.values()),
        "rows": rows,
    }
    (HERE / "ASSEMBLY_GATE.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"\ncomparison {json.dumps(comparison)}")
    print(f"gates {json.dumps(gates)}")
    print(f"PASSED {payload['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
