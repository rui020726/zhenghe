"""The conditional online-system block: real search, both arms, one corrected environment.

Runs only after the replay block has passed its correctness gates with no unresolved
regression. Nothing here retrains anything: the same PILOC checkpoint searches, the same CTDE
allocation network decides, the two arms differ only in the controller switches frozen in
``arms.py``, and both run on the membership-consistent environment.

The point of the block, stated before it is run: search and allocation are separate entities in
this architecture (invariant 10), so an online run checks system integration under real search
input -- that detections arrive through the search module rather than a stream, and that the
allocation still behaves under that input. It is not evidence of any new feedback path from
allocation into search. That claim is tested here rather than assumed: for every seed, the
ordered detection sequence (target, time, discovering robot) must come out identical in both
arms. If it does not, the two arms did not face the same world and nothing in this block
compares.

    D:/project_env/python.exe runs/corrected_env_final_20260918/system_v2.py --worker 0 --workers 6
    D:/project_env/python.exe runs/corrected_env_final_20260918/system_v2.py --aggregate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
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
for path in (REPO, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import arms  # noqa: E402
from integrated_sar.expired_attribution import build_timelines  # noqa: E402
from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402
from integrated_sar.lifecycle_metrics import load_records  # noqa: E402

OUT = HERE / "system_block"
SEEDS = [int(seed) for seed in json.loads(
    (HERE / "HELDOUT_SEEDS_V2.json").read_text(encoding="utf-8")
)["system_seeds"]]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(part, path)


def online_configuration(label: str, seed: int) -> RunConfiguration:
    arm = arms.ARMS[label]
    values = dict(arm.configuration)  # includes task_membership_consistency: both arms are online here
    directory = OUT / label
    directory.mkdir(parents=True, exist_ok=True)
    values.update(
        seed=seed,
        piloc_checkpoint=str(arms.PIL_CHECKPOINT),
        output=str(directory / f"sys_{label}_s{seed}.jsonl"),
        decision_trace=str(directory / f"sys_tr_{label}_s{seed}.jsonl"),
    )
    # The search side is live: no replay stream, so the guard against two search policies
    # standing in for each other does not apply.
    return RunConfiguration(**values)


def detection_sequence(records: list[dict]) -> list[tuple]:
    """The world both arms must have faced, in the order it arrived."""
    return [
        (
            str(row.get("target_id")),
            round(float(row.get("timestamp", 0.0)), 6),
            str(row.get("discovered_by")),
        )
        for row in records
        if row.get("event_type") == "detection"
    ]


def run_one(label: str, seed: int) -> dict:
    directory = OUT / label
    started = time.time()
    run = build_joint_run(online_configuration(label, seed))
    goals_at_start = int(len(run.search_adapter.env.goals_pos))
    arms.assert_built(run, arms.ARMS[label])
    summary = run.run()
    if run.decision_trace is not None:
        run.decision_trace.close()
    records = load_records(directory / f"sys_{label}_s{seed}.jsonl")
    sequence = detection_sequence(records)
    timelines = build_timelines(records, arm=f"sys{label}", seed=seed)
    on_time = [
        timeline
        for timeline in timelines.values()
        if timeline.final_progress is not None
        and float(timeline.final_progress) >= 1.0
        and timeline.final_failure_reason != "deadline_expired"
    ]
    env = run.task_adapter.env
    payload = {
        "seed": seed,
        "arm": label,
        "code_commit": run.run_signature["code_commit"],
        "signature_arm": run.run_signature["arm"],
        "config_hash": run.run_signature["config_hash"],
        "run_steps": int(summary.get("steps", 0)),
        "run_time": round(float(summary.get("current_time", 0.0)), 3),
        "scenario_goals": goals_at_start,
        "goals_remaining_undiscovered": int(summary.get("remaining_search_targets", 0)),
        "goals_discovered": goals_at_start - int(summary.get("remaining_search_targets", 0)),
        "detection_events": int(summary.get("detection_events", 0)),
        "env_tasks_total": int(len(env.task_dic)),
        "targets_detected_in_lifecycle": len(timelines),
        "on_time": len(on_time),
        "expired": int(summary.get("status_counts", {}).get("expired", 0)),
        "completed_env_tasks": int(summary.get("status_counts", {}).get("completed", 0)),
        "share_of_all_goals_on_time": round(len(on_time) / max(1, goals_at_start), 6),
        "share_of_discovered_on_time": round(len(on_time) / max(1, len(timelines)), 6),
        "rescue_travel_dist_total": round(
            float(sum(float(agent["travel_dist"]) for agent in env.agent_dic.values())), 3
        ),
        "mean_completion_time": (
            round(
                float(
                    np.mean(
                        [
                            float(point.timestamp)
                            for timeline in on_time
                            for point in timeline.progress
                            if float(point.progress) >= 1.0
                        ]
                    )
                ),
                3,
            )
            if on_time
            else None
        ),
        "detection_sequence_sha256": hashlib.sha256(
            json.dumps(sequence).encode("utf-8")
        ).hexdigest(),
        "detection_sequence": sequence,
        "wall_seconds": round(time.time() - started, 1),
    }
    write_json(directory / f"system_{label}_s{seed}.json", payload)
    print(
        f"[{label} s{seed}] goals={payload['scenario_goals']} "
        f"discovered={payload['goals_discovered']} on_time={payload['on_time']} "
        f"det_seq={payload['detection_sequence_sha256'][:10]}",
        flush=True,
    )
    return payload


def aggregate() -> int:
    rows: dict[str, dict[int, dict]] = {
        label: {
            int(row["seed"]): row
            for path in sorted((OUT / label).glob(f"system_{label}_s*.json"))
            for row in [json.loads(path.read_text(encoding="utf-8"))]
        }
        for label in arms.ARMS
    }
    paired = sorted(set(rows["C"]) & set(rows["D"]))
    problems: list[str] = []
    for seed in paired:
        if rows["C"][seed]["detection_sequence_sha256"] != rows["D"][seed]["detection_sequence_sha256"]:
            problems.append(f"s{seed}: the arms searched different worlds")
    for label, values in rows.items():
        for seed, row in values.items():
            if int(row["run_steps"]) != arms.STEPS:
                problems.append(f"{label} s{seed}: steps={row['run_steps']}")
            if row["code_commit"] != arms.freeze_commit():
                problems.append(f"{label} s{seed}: commit {row['code_commit']}")
    summary = {
        "seeds_scheduled": len(SEEDS),
        "episodes": {label: len(values) for label, values in rows.items()},
        "paired_seeds": len(paired),
        "decoupling_invariant_holds": not any(
            "searched different worlds" in problem for problem in problems
        ),
        "technical_problems": problems[:40],
    }
    for label, values in rows.items():
        chosen = [values[seed] for seed in sorted(values)]
        summary[label] = {
            "all_targets": int(sum(row["scenario_goals"] for row in chosen)),
            "discovered": int(sum(row["goals_discovered"] for row in chosen)),
            "on_time": int(sum(row["on_time"] for row in chosen)),
            "completion_rate_of_discovered": round(
                sum(row["on_time"] for row in chosen)
                / max(1, sum(row["targets_detected_in_lifecycle"] for row in chosen)),
                4,
            ),
            "on_time_share_of_all_targets": round(
                sum(row["on_time"] for row in chosen)
                / max(1, sum(row["scenario_goals"] for row in chosen)),
                4,
            ),
            "mean_rescue_travel_dist": round(
                float(np.mean([row["rescue_travel_dist_total"] for row in chosen])), 3
            ),
            "mean_wall_seconds": round(
                float(np.mean([row["wall_seconds"] for row in chosen])), 1
            ),
            "expired": int(sum(row["expired"] for row in chosen)),
        }
    differences = [
        rows["D"][seed]["share_of_all_goals_on_time"] - rows["C"][seed]["share_of_all_goals_on_time"]
        for seed in paired
    ]
    summary["paired_share_of_all_goals_on_time"] = {
        "mean_difference_D_minus_C": round(float(np.mean(differences)), 6) if differences else None,
        "seeds_D_better": int(sum(1 for value in differences if value > 1e-12)),
        "seeds_D_worse": int(sum(1 for value in differences if value < -1e-12)),
        "note": "descriptive: 60 seeds is not powered for an inferential claim",
    }
    write_json(HERE / "SYSTEM_SUMMARY.json", summary)
    print(json.dumps(summary, indent=1))
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--workers", type=int, default=arms.WORKERS)
    parser.add_argument("--arm", choices=tuple(arms.ARMS))
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args(argv)
    if args.aggregate:
        return aggregate()
    jobs = [
        (label, seed)
        for label in ([args.arm] if args.arm else list(arms.ARMS))
        for seed in SEEDS
    ]
    if args.worker is not None:
        for position, (label, seed) in enumerate(jobs):
            if position % args.workers != args.worker:
                continue
            if (OUT / label / f"system_{label}_s{seed}.json").exists():
                continue
            run_one(label, seed)
        return 0
    processes = [
        subprocess.Popen(
            [
                sys.executable, str(Path(__file__).resolve()), "--worker", str(index),
                "--workers", str(args.workers), *(["--arm", args.arm] if args.arm else []),
            ],
            cwd=str(REPO),
        )
        for index in range(args.workers)
    ]
    codes = [process.wait() for process in processes]
    if any(codes):
        print(f"worker failures: {codes}", file=sys.stderr)
        return 1
    return aggregate()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
