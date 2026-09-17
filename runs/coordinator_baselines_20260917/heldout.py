"""Held-out evaluation: fresh streams, one shared event sequence, several arms.

The 30 seeds under ``runs/joint_eval_20260916/streams`` were read repeatedly while
the calibration offset and the coalition term were designed, so they are a
development set: every choice (which offset, which arm) is made there and nowhere
else. This script builds the one-shot test set instead -- seeds recorded *after*
the arms were frozen -- and replays one recorded detection stream through every
arm, so a paired difference between two arms cannot come from a different search
history.

The recording pass writes no arm log: it only produces the stream bundle, so every
arm (including ``nocoord``) is produced by the identical replay code path and the
recorded search run cannot leak its own behaviour into the comparison.

::

    python runs/coordinator_baselines_20260917/heldout.py record --shard 0/3
    python runs/coordinator_baselines_20260917/heldout.py run --shard 0/3
    python runs/coordinator_baselines_20260917/heldout.py audit
    python runs/coordinator_baselines_20260917/heldout.py report

Every subcommand is resume-safe: an output that already exists is skipped, so a
sharded or interrupted run is relaunched rather than repaired.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import dev_sweep  # noqa: E402  (same directory, shares the arm -> CLI mapping)

FROZEN = REPO / "runs" / "joint_eval_20260916"
# The frozen PILOC checkpoint the dev streams were recorded with, so the held-out
# scenes differ from the dev ones by seed only and not by search policy.
PILOC = REPO / "runs" / "cur_v2rep05_s2" / "best" / "shared_mappo_model_21_3100.pth"
HELD = HERE / "heldout"
STREAMS = HELD / "streams"
REC = HELD / "rec"
STEPS = 1000
# Fresh seed block: disjoint from the dev block (10000-10029) and from the older
# joint ablation, so no stream here has ever been measured before.
HELD_SEEDS = list(range(20000, 20100))
# The calibration point the dev set selects: on the 5-seed coarse scan 0.10 is the
# smallest offset at which the depot still won on 94.9% of the legal-task
# decisions; on the full 30 seeds it completes 156 / expires 15 against nocoord's
# 136/35. It is chosen as the smallest offset that removes the failure mode, not as
# the offset with the best score -- 0.25 scores one target higher on the same dev
# set, which is inside the noise, so preferring it here would be tuning on the
# development seeds.
BEST_CAL = 0.10
HEURISTIC_SCALE = dev_sweep.HEURISTIC_SCALE
COALITION_SCALE = dev_sweep.COALITION_SCALE
PLANNER_SCALE = dev_sweep.PLANNER_SCALE

# The arms this dataset is read with. ``(task_logit_offset, residual_scale, policy)``;
# a scale of None means no coordinator at all.
ARMS: tuple[tuple[float, float | None, str], ...] = (
    (0.0, None, "heuristic"),
    (0.0, HEURISTIC_SCALE, "heuristic"),
    (BEST_CAL, None, "heuristic"),
    (0.0, COALITION_SCALE, "coalition"),
    (BEST_CAL, COALITION_SCALE, "coalition"),
    # The reserving planner rides the same residual budget as the two per-pair
    # policies. It is a *later* arm: the dev set selected it (164 / 171 completed and
    # 7 expiries against rs0.25's 152 / 19), so the held-out number this arm produces
    # is a confirmation of a dev decision, not an independent selection. Treat it
    # exactly as the earlier phases treated their own post-hoc arms: the test set can
    # reject the dev choice, and cannot be used to pick a different one.
    (0.0, PLANNER_SCALE, "coalition_planner"),
)
# A further point, added after those were frozen and before any held-out number was
# read: on the dev set the calibration curve is still improving at 0.25 (157
# completed / 14 expired against 156/15 at 0.10, with 0.625 of the remaining
# legal-task decisions still choosing the depot), so the test set has to measure
# the calibration *family* -- 0.00, 0.10, 0.25 -- rather than only the single point
# the dev seeds preferred. Its identity is fixed here, from dev outcomes alone.
EXTRA_ARMS: tuple[tuple[float, float | None, str], ...] = ((0.25, None, "heuristic"),)
ALL_ARMS = ARMS + EXTRA_ARMS


def label_of(offset: float, scale: float | None, policy: str) -> str:
    return dev_sweep.arm_label(
        offset=offset, coordinator_scale=scale, coordinator_policy=policy
    )


def record_command(seed: int) -> list[str]:
    """One live PILOC pass, whose emitted detections are the stream every arm replays."""
    return [
        sys.executable or "python",
        "-m",
        "integrated_sar.joint_driver",
        "--piloc-checkpoint",
        str(PILOC),
        "--task-checkpoint",
        str(dev_sweep.CTDE),
        "--actor-kv-mode",
        "tokens",
        "--steps",
        str(STEPS),
        "--seed",
        str(seed),
        "--no-coordinator",
        "--quiet",
        "--record-stream",
        str(STREAMS / f"s{seed}.json"),
        "--output",
        str(REC / f"rec_s{seed}.jsonl"),
    ]


def _shard(jobs: list, spec: str | None) -> list:
    if not spec:
        return jobs
    index, count = (int(part) for part in spec.split("/"))
    return [job for position, job in enumerate(jobs) if position % count == index]


def record(args: argparse.Namespace) -> int:
    STREAMS.mkdir(parents=True, exist_ok=True)
    REC.mkdir(parents=True, exist_ok=True)
    jobs = [
        seed
        for seed in HELD_SEEDS
        if not (STREAMS / f"s{seed}.json").exists() or not (REC / f"rec_s{seed}.jsonl").exists()
    ]
    jobs = _shard(jobs, args.shard)
    for position, seed in enumerate(jobs, start=1):
        print(f"[{position}/{len(jobs)}] record seed {seed}", flush=True)
        result = subprocess.run(record_command(seed), cwd=REPO)
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})", flush=True)
            return result.returncode
    print(f"{len(jobs)} streams recorded")
    return 0


def run(args: argparse.Namespace) -> int:
    HELD.mkdir(parents=True, exist_ok=True)
    arms = ALL_ARMS
    if args.only:
        arms = tuple(arm for arm in ALL_ARMS if label_of(*arm) == args.only)
        if not arms:
            known = ", ".join(label_of(*arm) for arm in ALL_ARMS)
            print(f"--only {args.only!r} names no arm; known arms: {known}", flush=True)
            return 1
    specs = [
        (seed, offset, scale, policy)
        for offset, scale, policy in arms
        for seed in HELD_SEEDS
    ]
    specs = _shard(specs, args.shard)
    planned = 0
    for position, (seed, offset, scale, policy) in enumerate(specs, start=1):
        label = label_of(offset, scale, policy)
        log = HELD / f"rp_{label}_s{seed}.jsonl"
        trace = HELD / f"tr_{label}_s{seed}.jsonl"
        if log.exists() and trace.exists():
            continue
        stream = STREAMS / f"s{seed}.json"
        if not stream.exists():
            print(f"missing stream {stream}; run the record pass first", flush=True)
            return 1
        planned += 1
        print(f"[{position}/{len(specs)}] {label} seed {seed}", flush=True)
        argv = dev_sweep.command(
            seed, offset, scale, policy, stream=stream, out_dir=HELD
        )
        result = subprocess.run(argv, cwd=REPO)
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})", flush=True)
            return result.returncode
    print(f"{len(specs)} jobs, {planned} executed")
    return 0


def audit(args: argparse.Namespace) -> int:
    """Prove every arm saw the same DetectionEvent sequence as the recorded stream.

    The comparison is by value, not by file identity: each arm's lifecycle log is
    reduced to ``(detected_at, target_id, row, col)`` per detection and must equal
    the bundle's fingerprint, in order. A mismatch means an arm was measuring a
    different world, which invalidates every paired difference drawn from it.
    """
    from integrated_sar.detection_replay import load_detection_stream

    reports: dict[str, dict[str, object]] = {}
    checked = 0
    mismatches: list[str] = []
    for seed in HELD_SEEDS:
        path = STREAMS / f"s{seed}.json"
        if not path.exists():
            mismatches.append(f"seed {seed}: stream missing")
            continue
        stream = load_detection_stream(path)
        expected = stream.fingerprint()
        for offset, scale, policy in ALL_ARMS:
            label = label_of(offset, scale, policy)
            log = HELD / f"rp_{label}_s{seed}.jsonl"
            entry = reports.setdefault(
                label, {"seeds": 0, "detections": 0, "streams": set(), "failures": 0}
            )
            if not log.exists():
                continue
            entry["seeds"] = int(entry["seeds"]) + 1
            observed = _detection_fingerprint(log)
            entry["detections"] = int(entry["detections"]) + len(observed)
            entry["streams"].add(_replayed_stream(log))
            if observed == expected:
                checked += 1
            else:
                entry["failures"] = int(entry["failures"]) + 1
                detail = next(
                    (
                        f"index {i}: log {a} vs stream {b}"
                        for i, (a, b) in enumerate(zip(observed, expected))
                        if a != b
                    ),
                    f"length {len(observed)} vs {len(expected)}",
                )
                mismatches.append(f"seed {seed} {label}: {detail}")

    payload = {
        "checked": checked,
        "mismatches": mismatches[:50],
        "mismatch_count": len(mismatches),
        "arms": {
            label: {
                "runs": entry["seeds"],
                "detections": entry["detections"],
                "failures": entry["failures"],
                "distinct_stream_paths": sorted(entry["streams"]),
            }
            for label, entry in sorted(reports.items())
        },
    }
    out = HELD / "replay_audit.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for label, entry in payload["arms"].items():
        print(
            f"{label:<20} runs {entry['runs']:>4} detections {entry['detections']:>5} "
            f"failures {entry['failures']:>3} streams {entry['distinct_stream_paths']}"
        )
    print(f"{checked} (run, seed) pairs verified event-by-event; {len(mismatches)} mismatches")
    print(f"audit written to {out}")
    return 1 if mismatches else 0


def _detection_fingerprint(path: Path) -> list[tuple[float, str, int, int]]:
    from integrated_sar.lifecycle_metrics import load_records

    observed: list[tuple[float, str, int, int]] = []
    for record in load_records(path):
        if record.get("event_type") != "detection":
            continue
        detail = record.get("detail") or {}
        grid = detail.get("grid_position") or (None, None)
        observed.append(
            (
                float(record.get("timestamp") or 0.0),
                str(record.get("target_id")),
                int(grid[0]),
                int(grid[1]),
            )
        )
    return observed


def _replayed_stream(path: Path) -> str:
    from integrated_sar.lifecycle_metrics import load_records

    for record in load_records(path):
        if record.get("event_type") == "run_config":
            configuration = (record.get("detail") or {}).get("configuration") or {}
            return str(configuration.get("replay_stream"))
    return "<no run_config record>"


def report(args: argparse.Namespace) -> int:
    from integrated_sar.allocation_trace import (
        analyse,
        read_trace,
        report as trace_report,
        summarise,
    )
    from integrated_sar.expired_attribution import attribute, report as expired_report
    from integrated_sar.lifecycle_metrics import collect, report as metrics_report

    logs = sorted(HELD.glob("rp_*_s*.jsonl"))
    traces = sorted(HELD.glob("tr_*_s*.jsonl"))
    if not logs:
        print("no held-out lifecycle logs yet in", HELD)
        return 1

    table = collect([str(path) for path in logs], [STEPS])
    text = metrics_report(table, [STEPS], "nocoord")
    print(text)

    rows = []
    for path in traces:
        meta, decisions = read_trace(path)
        if decisions:
            rows.append(
                analyse(
                    decisions,
                    arm=str(meta.get("arm") or "unknown"),
                    seed=int(meta.get("seed") or 0),
                )
            )
    trace_text = trace_report(summarise(rows)) if rows else "no decision traces yet"
    print(trace_text)

    attributions = attribute([str(path) for path in logs], window=float(STEPS))
    expired_text = expired_report(attributions, (5, 10, 20, 30))
    print(expired_text)

    utilization = dev_sweep.utilization_by_arm(logs)
    print("  fleet utilization (mean of each run's closing record)")
    for arm, block in sorted(utilization.items()):
        print(f"    {arm:<22} {block['mean']:.4f}±{block['ci95']:.4f} ({block['runs']} runs)")

    audit_path = HELD / "replay_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
    payload = {
        "format": "coordinator-baselines/1",
        "stage": "held-out",
        "dataset": {
            "seeds": sorted({int(path.stem.rsplit("_s", 1)[1]) for path in logs}),
            "streams": "runs/coordinator_baselines_20260917/heldout/streams "
            "(recorded after the arms were frozen)",
            "note": "one-shot test set: no arm was selected on these seeds",
        },
        "arms": {
            label_of(offset, scale, policy): {
                "ctde_task_logit_offset": offset,
                "residual_scale": scale,
                "coordinator_policy": None if scale is None else policy,
            }
            for offset, scale, policy in ALL_ARMS
        },
        "lifecycle": {
            arm: {
                str(window): {str(seed): metrics.as_row() for seed, metrics in per_seed.items()}
                for window, per_seed in windows.items()
            }
            for arm, windows in table.items()
        },
        "decision_trace": summarise(rows) if rows else {},
        "resource_utilization": utilization,
        "expired_attribution": {
            arm: {
                "counts": dict(attribution.counts),
                "expired": attribution.expired_total,
                "provably_infeasible": attribution.provably_infeasible,
                "infeasible_with_travel": attribution.infeasible_with_travel,
                "stall": attribution.stall_stats(),
                "latency": attribution.latency_stats(),
            }
            for arm, attribution in attributions.items()
        },
        "replay_audit": audit,
        "lifecycle_report": text,
        "decision_trace_report": trace_text,
        "expired_report": expired_text,
    }
    out = HELD / "heldout_summary.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"summary written to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)

    for name, func, with_shard in (
        ("record", record, True),
        ("run", run, True),
        ("audit", audit, False),
        ("report", report, False),
    ):
        child = sub.add_parser(name)
        if with_shard:
            child.add_argument("--shard", default=None, help="i/n, e.g. 0/3")
        child.set_defaults(func=func)
    sub.choices["run"].add_argument(
        "--only",
        default=None,
        help="execute just this arm label, e.g. nocoord+cal0.25 (a later arm is "
        "added without re-sharding the arms already in flight)",
    )

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    os.environ.setdefault("MKL_NUM_THREADS", "6")
    raise SystemExit(main())