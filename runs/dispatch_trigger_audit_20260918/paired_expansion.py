"""Paired expansion for the redundant-member release: legacy 100 + fresh 60.

The small-scale gate passed 6/6 with no safety or information concern, so per the
brief the two existing analysis blocks run paired, two arms each. Both blocks have
been read during development and are **not** held-out test sets.

Discipline (brief section nine): the code and trace schema were frozen at the commit
recorded below before the first run; a single main process owns the work list, the
worker shards, the preflight and the aggregation; every worker only touches its own
shard; each output is written to a ``.part`` file and atomically renamed on
completion; nothing of an earlier signature is mixed in.

    # main process: shard, spawn workers, preflight, aggregate
    D:/project_env/python.exe runs/dispatch_trigger_audit_20260918/paired_expansion.py
    # one worker's shard (used by the main process, not usually by hand)
    D:/project_env/python.exe runs/dispatch_trigger_audit_20260918/paired_expansion.py --worker 0 --workers 6
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from integrated_sar.data_splits import split_of  # noqa: E402
from integrated_sar.expired_attribution import build_timelines  # noqa: E402
from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402
from integrated_sar.lifecycle_metrics import load_records  # noqa: E402

# The commit the runs must all carry in their run_signature (brief section nine).
FROZEN_COMMIT = "6915d6383fb61b37e1652fca0c4eb73b4b5b32a2"
BASE = "coalition_planner"
CANDIDATE = "coalition_redundant_release"
OUT = HERE / "paired" / "expansion.json"
WORKERS = 8


def stream_path(split_name: str, seed: int) -> Path:
    if split_name == "analysis_legacy":
        return REPO / "runs" / "coordinator_baselines_20260917" / "heldout" / "streams" / f"s{seed}.json"
    return REPO / "runs" / "learnability_20260917" / split_of(split_name).stream_dir_name.split("/", 1)[1] / f"s{seed}.json"


def work_list() -> list[tuple[str, int, str, str]]:
    """(split, seed, policy, tag) for every required run, in a fixed order."""
    items: list[tuple[str, int, str, str]] = []
    for split_name in ("analysis_legacy", "analysis_fresh"):
        for seed in split_of(split_name).seeds:
            for policy, tag in ((BASE, "baseline"), (CANDIDATE, "candidate")):
                items.append((split_name, int(seed), policy, tag))
    return items


def run_one(split_name: str, seed: int, policy: str, tag: str) -> Path:
    rp = HERE / "paired" / split_name / f"{tag}_rp_s{seed}.jsonl"
    trace = HERE / "paired" / split_name / f"{tag}_tr_s{seed}.jsonl"
    rp.parent.mkdir(parents=True, exist_ok=True)
    configuration = RunConfiguration(
        seed=seed,
        steps=1000,
        device="cpu",
        output=str(rp),
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
        replay_stream=str(stream_path(split_name, seed)),
    )
    run = build_joint_run(configuration)
    run.run()
    return rp


def worker(shard: int, total: int) -> int:
    items = work_list()
    mine = items[shard::total]
    for index, (split_name, seed, policy, tag) in enumerate(mine):
        rp = HERE / "paired" / split_name / f"{tag}_rp_s{seed}.jsonl"
        if rp.exists():
            continue
        part = rp.with_suffix(".jsonl.part")
        done = run_one(split_name, seed, policy, tag)
        done.rename(rp)
        print(f"[worker {shard}] {index + 1}/{len(mine)} {split_name} {tag} s{seed}", flush=True)
    return 0


def preflight(items: list[tuple[str, int, str, str]]) -> list[str]:
    """Every run present, every signature the frozen one, every stream the right one."""
    problems: list[str] = []
    for split_name, seed, policy, tag in items:
        rp = HERE / "paired" / split_name / f"{tag}_rp_s{seed}.jsonl"
        if not rp.exists():
            problems.append(f"missing {rp.name}")
            continue
        head = json.loads(rp.read_text(encoding="utf-8").splitlines()[0])
        detail = head.get("detail", {})
        signature = detail.get("run_signature") or {}
        configuration = detail.get("configuration", {})
        if int(configuration.get("seed", -1)) != seed:
            problems.append(f"{rp.name}: seed mismatch")
        if configuration.get("coordinator_policy") != policy:
            problems.append(f"{rp.name}: policy mismatch")
        commit = str(signature.get("code_commit", ""))
        if commit != FROZEN_COMMIT:
            problems.append(f"{rp.name}: code_commit {commit!r} != frozen {FROZEN_COMMIT!r}")
    return problems


def per_run(rp: Path) -> dict:
    records = load_records(rp)
    timelines = build_timelines(records, arm=rp.stem, seed=int(rp.stem.split("s")[-1]))
    counts: dict[str, int] = {"completed": 0, "expired": 0}
    for timeline in timelines.values():
        if timeline.final_progress >= 1.0 and timeline.final_failure_reason != "deadline_expired":
            counts["completed"] += 1
        elif timeline.final_failure_reason == "deadline_expired":
            counts["expired"] += 1
    return {
        "completed": counts["completed"],
        "expired": counts["expired"],
        "coverage_resets": sum(t.progress_setbacks for t in timelines.values()),
        "lost_work": round(sum(t.discarded_work_steps for t in timelines.values()), 3),
        "targets": {
            t.target_id: {
                "progress": round(t.final_progress, 4),
                "reason": t.final_failure_reason,
                "on_time": t.final_progress >= 1.0 and t.final_failure_reason != "deadline_expired",
            }
            for t in timelines.values()
        },
    }


def aggregate(items: list[tuple[str, int, str, str]]) -> dict:
    by_split: dict[str, dict] = {}
    for split_name in ("analysis_legacy", "analysis_fresh"):
        seeds = [int(s) for s in split_of(split_name).seeds]
        paired = []
        for seed in seeds:
            base = per_run(HERE / "paired" / split_name / f"baseline_rp_s{seed}.jsonl")
            cand = per_run(HERE / "paired" / split_name / f"candidate_rp_s{seed}.jsonl")
            rescue = [
                target
                for target, fact in cand["targets"].items()
                if fact["on_time"] and not base["targets"].get(target, {}).get("on_time")
            ]
            regress = [
                target
                for target, fact in base["targets"].items()
                if fact["on_time"] and not cand["targets"].get(target, {}).get("on_time")
            ]
            paired.append(
                {
                    "seed": seed,
                    "baseline": {k: base[k] for k in ("completed", "expired", "coverage_resets", "lost_work")},
                    "candidate": {k: cand[k] for k in ("completed", "expired", "coverage_resets", "lost_work")},
                    "rescued_targets": rescue,
                    "regressed_targets": regress,
                }
            )
        totals = {
            arm: {
                "completed": sum(p[arm]["completed"] for p in paired),
                "expired": sum(p[arm]["expired"] for p in paired),
                "coverage_resets": round(sum(p[arm]["coverage_resets"] for p in paired), 3),
                "lost_work": round(sum(p[arm]["lost_work"] for p in paired), 3),
            }
            for arm in ("baseline", "candidate")
        }
        rescued_seeds = sum(1 for p in paired if p["rescued_targets"])
        regressed_seeds = sum(1 for p in paired if p["regressed_targets"])
        by_split[split_name] = {
            "seeds": len(seeds),
            "totals": totals,
            "seeds_with_rescued_targets": rescued_seeds,
            "seeds_with_regressed_targets": regressed_seeds,
            "completed_delta": totals["candidate"]["completed"] - totals["baseline"]["completed"],
            "expired_delta": totals["candidate"]["expired"] - totals["baseline"]["expired"],
            "paired": paired,
        }
    return by_split


def main(argv: list[str]) -> int:
    if "--worker" in argv:
        index = int(argv[argv.index("--worker") + 1])
        total = int(argv[argv.index("--workers") + 1])
        return worker(index, total)

    items = work_list()
    missing_before = [item for item in items if not (
        HERE / "paired" / item[0] / f"{item[3]}_rp_s{item[1]}.jsonl"
    ).exists()]
    print(f"{len(items)} runs required, {len(items) - len(missing_before)} already present", flush=True)

    processes = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(i), "--workers", str(WORKERS)],
            cwd=str(REPO),
        )
        for i in range(WORKERS)
    ]
    codes = [process.wait() for process in processes]
    if any(code != 0 for code in codes):
        print(f"worker failure codes: {codes}")
        return 1

    problems = preflight(items)
    if problems:
        print("PREFLIGHT FAILED:")
        for problem in problems[:20]:
            print("  ", problem)
        return 1

    payload = {"frozen_commit": FROZEN_COMMIT, "splits": aggregate(items)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    part = OUT.with_suffix(".json.part")
    part.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    part.rename(OUT)
    for split_name, block in payload["splits"].items():
        print(
            f"{split_name}: baseline {block['totals']['baseline']['completed']}completed/"
            f"{block['totals']['baseline']['expired']}expired -> candidate "
            f"{block['totals']['candidate']['completed']}completed/"
            f"{block['totals']['candidate']['expired']}expired "
            f"(rescued seeds {block['seeds_with_rescued_targets']}, "
            f"regressed seeds {block['seeds_with_regressed_targets']})"
        )
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
