"""Dev-set driver for the two interpretable baselines (calibration, coalition).

Both baselines are tested against the *frozen* 30-seed detection streams recorded
for the joint ablation, so every arm replays byte-identical search events and the
only difference between two runs is the arm itself::

    python runs/coordinator_baselines_20260917/dev_sweep.py run --mode coarse
    python runs/coordinator_baselines_20260917/dev_sweep.py report
    python runs/coordinator_baselines_20260917/dev_sweep.py run --mode full --cal 0.05 0.10
    python runs/coordinator_baselines_20260917/dev_sweep.py run --mode full --cal 0.05 0.10 \\
        --coalition

Those 30 seeds have already been read several times over the course of this work,
so they are a *development* set: arms are chosen and thresholds are set here, and
the winner is then re-measured once on a fresh, larger held-out stream set. The
lifecycle logs, decision traces and summary JSON land in this directory; the
streams stay where they were recorded.

Resume-safe: a run whose lifecycle log and trace already exist is skipped, so a
sharded or interrupted sweep can simply be relaunched. ``--shard i/n`` splits the
work between two processes; each run is single-threaded inside torch, so one
shard per ~6 cores is the useful degree of parallelism.

``compare`` is the trace-level companion used by stage B: given two arms, it pairs
their decisions instant by instant and reports whether the residual policy changed
anything -- not just whether the two policies chose differently, but whether their
per-action residuals differ at all, and how much larger the residual would have to
be before the winner flips. That last number distinguishes "the reward is wrong"
from "this interface cannot reach the decision that matters".
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

FROZEN = REPO / "runs" / "joint_eval_20260916"
STREAMS = FROZEN / "streams"
CTDE = FROZEN / "ctde_final_7_1.pth"
STEPS = 1000
DEV_SEEDS = list(range(10000, 10030))
COARSE_SEEDS = DEV_SEEDS[:5]
# The coarse scan the task fixes: the uncalibrated point plus six non-zero
# offsets, chosen to straddle the 0.03 median |margin| of the uncalibrated actor.
COARSE_OFFSETS = (0.0, 0.02, 0.03, 0.05, 0.10, 0.15, 0.25)
HEURISTIC_SCALE = 0.25
COALITION_SCALE = 0.25


def arm_label(
    *, offset: float, coordinator_scale: float | None, coordinator_policy: str = "heuristic"
) -> str:
    """Identical to ``integrated_sar.allocation_trace.arm_name`` for the arm's meta."""
    from integrated_sar.allocation_trace import arm_name

    return arm_name(
        use_coordinator=coordinator_scale is not None,
        residual_scale=coordinator_scale if coordinator_scale is not None else 0.5,
        task_logit_offset=offset,
        coordinator_policy=coordinator_policy,
    )


def specs(args: argparse.Namespace) -> list[tuple[float, float | None, str]]:
    """The arms to run: ``(task_logit_offset, coordinator_scale or None, policy)``."""
    if args.mode == "coarse":
        arms = [(offset, None, "heuristic") for offset in COARSE_OFFSETS]
    else:
        arms = [(offset, None, "heuristic") for offset in [0.0, *args.cal]]
    arms.append((0.0, HEURISTIC_SCALE, "heuristic"))
    if getattr(args, "coalition", False):
        # Same residual budget as the heuristic arm, so the two differ only in *what*
        # the bias rewards; calibration is stacked on top of the coalition term too.
        arms += [(offset, COALITION_SCALE, "coalition") for offset in [0.0, *args.cal]]
    return arms


def seeds(args: argparse.Namespace) -> list[int]:
    return COARSE_SEEDS if args.mode == "coarse" else DEV_SEEDS


def command(
    seed: int,
    offset: float,
    coordinator_scale: float | None,
    coordinator_policy: str = "heuristic",
    *,
    trace: bool = True,
    stream: Path | None = None,
    out_dir: Path | None = None,
) -> list[str]:
    """The joint_driver argv for one arm on one stream. Shared with ``heldout.py``."""
    stream = STREAMS / f"s{seed}.json" if stream is None else stream
    out_dir = HERE if out_dir is None else out_dir
    label = arm_label(
        offset=offset, coordinator_scale=coordinator_scale, coordinator_policy=coordinator_policy
    )
    argv = [
        sys.executable or "python",
        "-m",
        "integrated_sar.joint_driver",
        "--task-checkpoint",
        str(CTDE),
        "--actor-kv-mode",
        "tokens",
        "--steps",
        str(STEPS),
        "--seed",
        str(seed),
        "--replay-stream",
        str(stream),
        "--quiet",
        "--output",
        str(out_dir / f"rp_{label}_s{seed}.jsonl"),
    ]
    if coordinator_scale is None:
        argv.append("--no-coordinator")
    else:
        argv += ["--residual-scale", f"{coordinator_scale:g}"]
        if coordinator_policy != "heuristic":
            argv += ["--coordinator", coordinator_policy]
    if offset:
        argv += ["--ctde-task-logit-offset", f"{offset:g}"]
    if trace:
        argv += ["--decision-trace", str(out_dir / f"tr_{label}_s{seed}.jsonl")]
    return argv


def run(args: argparse.Namespace) -> int:
    jobs = [
        (seed, offset, scale, policy)
        for offset, scale, policy in specs(args)
        for seed in seeds(args)
    ]
    if args.shard:
        index, count = (int(part) for part in args.shard.split("/"))
        jobs = [job for position, job in enumerate(jobs) if position % count == index]

    planned = 0
    completed = 0
    for seed, offset, scale, policy in jobs:
        label = arm_label(offset=offset, coordinator_scale=scale, coordinator_policy=policy)
        log = HERE / f"rp_{label}_s{seed}.jsonl"
        trace = HERE / f"tr_{label}_s{seed}.jsonl"
        if log.exists() and trace.exists():
            completed += 1
            continue
        planned += 1
        print(f"[{planned}] {label} seed {seed}", flush=True)
        result = subprocess.run(command(seed, offset, scale, policy), cwd=REPO)
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})", flush=True)
            return result.returncode
    print(f"{len(jobs)} jobs, {completed} already done, {planned} executed")
    return 0


def utilization_by_arm(logs) -> dict[str, dict[str, float]]:
    """Per-run fleet utilization, read from each log's closing ``__run__`` record.

    ``resource_utilization`` is part of the run summary (the adapter's own view of how
    much of the fleet was working), not of the per-window metric pass, so it is read
    from the record rather than recomputed here.
    """
    from integrated_sar.lifecycle_metrics import load_records, parse_run_path

    samples: dict[str, list[float]] = {}
    for path in logs:
        arm, _seed = parse_run_path(path)
        for record in load_records(path):
            if record.get("event_type") != "final" or record.get("target_id") != "__run__":
                continue
            metrics = (record.get("detail") or {}).get("metrics") or {}
            value = metrics.get("resource_utilization")
            if value is not None:
                samples.setdefault(arm, []).append(float(value))
    summary: dict[str, dict[str, float]] = {}
    for arm, values in samples.items():
        mean = statistics.mean(values)
        ci = (
            1.96 * statistics.stdev(values) / math.sqrt(len(values))
            if len(values) > 1
            else 0.0
        )
        summary[arm] = {"runs": len(values), "mean": round(mean, 4), "ci95": round(ci, 4)}
    return summary


def report(args: argparse.Namespace) -> int:
    """Measure everything already on disk, paired by seed, and write the JSONs."""
    import json

    from integrated_sar.allocation_trace import (
        analyse,
        read_trace,
        report as trace_report,
        summarise,
    )
    from integrated_sar.expired_attribution import attribute, report as expired_report
    from integrated_sar.lifecycle_metrics import collect, report as metrics_report

    logs = sorted(HERE.glob("rp_*_s*.jsonl"))
    traces = sorted(HERE.glob("tr_*_s*.jsonl"))
    if not logs:
        print("no lifecycle logs yet in", HERE)
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

    # Why the remaining targets expired, and how much of the group's on-site time
    # was spent unable to work: the metric the coalition term is supposed to move.
    attributions = attribute([str(path) for path in logs], window=float(STEPS))
    expired_text = expired_report(attributions, (5, 10, 20, 30))
    print(expired_text)
    stall = {
        arm: {
            "counts": dict(attribution.counts),
            "expired": attribution.expired_total,
            "provably_infeasible": attribution.provably_infeasible,
            "infeasible_with_travel": attribution.infeasible_with_travel,
            "stall": attribution.stall_stats(),
            "latency": attribution.latency_stats(),
        }
        for arm, attribution in attributions.items()
    }

    arms = summarise(rows) if rows else {}
    utilization = utilization_by_arm(logs)
    print("  fleet utilization (mean of each run's closing record)")
    for arm, block in sorted(utilization.items()):
        print(f"    {arm:<22} {block['mean']:.4f}±{block['ci95']:.4f} ({block['runs']} runs)")
    payload = {
        "format": "coordinator-baselines/1",
        "stage": "dev",
        "dataset": {
            "seeds": sorted({int(path.stem.rsplit("_s", 1)[1]) for path in logs}),
            "streams": "runs/joint_eval_20260916/streams (frozen, replayed byte-identical)",
            "note": "development set; the winning arm is re-measured on the held-out set",
        },
        "lifecycle": {
            arm: {
                str(window): {
                    str(seed): metrics.as_row()
                    for seed, metrics in per_seed.items()
                }
                for window, per_seed in windows.items()
            }
            for arm, windows in table.items()
        },
        "decision_trace": arms,
        "resource_utilization": utilization,
        "expired_attribution": stall,
        "lifecycle_report": text,
        "decision_trace_report": trace_text,
        "expired_report": expired_text,
    }
    out = HERE / ("dev_summary.json" if args.mode == "full" else "dev_summary_coarse.json")
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"summary written to {out}")
    return 0


def _seed_of(path: Path) -> int:
    return int(path.stem.rsplit("_s", 1)[1])


def _rows_by_instant(rows: list[dict]) -> dict[tuple[int, str], list[dict]]:
    """Group one run's decisions by ``(step, agent)``; duplicates keep their order.

    The allocator decides per robot per allocation round, so that key is almost
    always unique. Rows are grouped rather than keyed to avoid silently dropping a
    decision if one round ever emits two.
    """
    buckets: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        key = (int(row.get("step") or 0), str(row.get("agent_id")))
        buckets.setdefault(key, []).append(row)
    return buckets


def _counterfactual_choice(row: dict, extra: Sequence[float] | None = None) -> int | None:
    """The argmax of ``ctde + offset(legal tasks) + bias + extra`` under the mask.

    This re-derives the decision from the numbers the trace records. Masked actions
    carry ``None`` in ``ctde_logits`` and can never win, whatever ``extra`` is, so
    the counterfactual is subject to the same hard mask as the live path. Ties break
    to the lowest index, matching ``torch.argmax``.
    """
    from integrated_sar.allocation_trace import legal_task_indices

    logits = row.get("ctde_logits")
    if not logits:
        return None
    legal = set(legal_task_indices(row))
    offset = float(row.get("task_logit_offset") or 0.0)
    bias = row.get("bias") or []
    best_index: int | None = None
    best_score: float | None = None
    for index, base in enumerate(logits):
        if base is None:
            continue
        score = float(base)
        if index in legal:
            score += offset
        if index < len(bias) and bias[index] is not None:
            score += float(bias[index])
        if extra is not None and index < len(extra):
            score += float(extra[index])
        if best_score is None or score > best_score:
            best_index, best_score = index, score
    return best_index


def _is_busy(row: dict) -> bool:
    return row.get("current_kind") is not None or str(row.get("status")) != "idle"


def compare(args: argparse.Namespace) -> int:
    """Pair two arms' decisions and measure whether their residuals change anything."""
    import json

    from integrated_sar.allocation_trace import legal_task_indices, read_trace

    base = Path(args.dir).resolve() if args.dir else HERE

    def rows_by_seed(label: str) -> dict[int, list[dict]]:
        loaded: dict[int, list[dict]] = {}
        for path in sorted(base.glob(f"tr_{label}_s*.jsonl")):
            _meta, decisions = read_trace(path)
            if decisions:
                loaded[_seed_of(path)] = decisions
        return loaded

    left = rows_by_seed(args.left)
    right = rows_by_seed(args.right)
    seeds = sorted(set(left) & set(right))
    if not seeds:
        print(f"no seed has both {args.left} and {args.right} traces", flush=True)
        return 1

    multipliers = [float(value) for value in args.multipliers]
    delta_multipliers = [float(value) for value in args.delta_multipliers]
    totals: dict[str, int] = {
        "paired": 0,
        "selection_diffs": 0,
        "unmatched_left": 0,
        "unmatched_right": 0,
        "legal_left": 0,
        "legal_right": 0,
        "legal_selection_diffs": 0,
        "bias_diff_rows": 0,
        "bias_diff_rows_legal": 0,
        "bias_diff_rows_legal_idle": 0,
        "bias_diff_rows_legal_busy": 0,
        "index_diffs_legal": 0,
        "index_diffs_off_legal": 0,
    }
    # Two counterfactual families. ``absolute`` scales the right arm's whole residual
    # (the shared heuristic part included); ``delta`` scales only what the coalition
    # policy added on top of the heuristic one, which is the term under test. The
    # first delta multiplier that flips a winner is the amplification this interface
    # would need to matter at all.
    flips = {value: {"all": 0, "legal": 0} for value in multipliers}
    delta_flips = {value: {"all": 0, "legal": 0} for value in delta_multipliers}
    legal_deltas: list[float] = []
    per_seed: list[dict] = []

    for seed in seeds:
        left_buckets = _rows_by_instant(left[seed])
        right_buckets = _rows_by_instant(right[seed])
        seed_row = {
            "seed": seed,
            "paired": 0,
            "selection_diffs": 0,
            "bias_diff_rows_legal": 0,
            "flips": {str(value): 0 for value in multipliers},
            "delta_flips": {str(value): 0 for value in delta_multipliers},
        }
        for key, left_rows in left_buckets.items():
            right_rows = right_buckets.get(key) or []
            totals["unmatched_left"] += max(0, len(left_rows) - len(right_rows))
            for left_row, right_row in zip(left_rows, right_rows):
                totals["paired"] += 1
                seed_row["paired"] += 1
                legal_left = bool(legal_task_indices(left_row))
                legal_right = bool(legal_task_indices(right_row))
                totals["legal_left"] += int(legal_left)
                totals["legal_right"] += int(legal_right)
                if int(left_row.get("chosen_index", 0)) != int(
                    right_row.get("chosen_index", 0)
                ):
                    totals["selection_diffs"] += 1
                    seed_row["selection_diffs"] += 1
                    if legal_left or legal_right:
                        totals["legal_selection_diffs"] += 1

                left_bias = left_row.get("bias") or []
                right_bias = right_row.get("bias") or []
                width = max(len(left_bias), len(right_bias))
                diffs = [
                    index
                    for index in range(width)
                    if (left_bias[index] if index < len(left_bias) else None)
                    != (right_bias[index] if index < len(right_bias) else None)
                ]
                if diffs:
                    totals["bias_diff_rows"] += 1
                    legal_right_indices = set(legal_task_indices(right_row))
                    legal_diffs = [index for index in diffs if index in legal_right_indices]
                    totals["index_diffs_legal"] += len(legal_diffs)
                    totals["index_diffs_off_legal"] += len(diffs) - len(legal_diffs)
                    # None is "this arm had no legal action here" (a masked entry, or a
                    # different menu because the two arms diverged), not a zero: it is
                    # handled as a plain 0.0 additive difference below.
                    delta = [
                        (
                            (right_bias[index] if index < len(right_bias) else None) or 0.0
                        )
                        - (
                            (left_bias[index] if index < len(left_bias) else None) or 0.0
                        )
                        for index in range(width)
                    ]
                    if legal_right:
                        totals["bias_diff_rows_legal"] += 1
                        seed_row["bias_diff_rows_legal"] += 1
                        if _is_busy(right_row):
                            totals["bias_diff_rows_legal_busy"] += 1
                        else:
                            totals["bias_diff_rows_legal_idle"] += 1
                        for index in legal_diffs:
                            legal_deltas.append(abs(delta[index]))

                actual = int(right_row.get("chosen_index", 0))
                for value in multipliers:
                    extra = [
                        (value - 1.0) * float(entry)
                        if entry is not None
                        else 0.0
                        for entry in (right_row.get("bias") or [])
                    ]
                    chosen = _counterfactual_choice(right_row, extra)
                    if chosen is not None and chosen != actual:
                        flips[value]["all"] += 1
                        seed_row["flips"][str(value)] += 1
                        if legal_right:
                            flips[value]["legal"] += 1
                if diffs:
                    for value in delta_multipliers:
                        extra = [(value - 1.0) * entry for entry in delta]
                        chosen = _counterfactual_choice(right_row, extra)
                        if chosen is not None and chosen != actual:
                            delta_flips[value]["all"] += 1
                            seed_row["delta_flips"][str(value)] += 1
                            if legal_right:
                                delta_flips[value]["legal"] += 1
        for key, right_rows in right_buckets.items():
            totals["unmatched_right"] += max(
                0, len(right_rows) - len(left_buckets.get(key) or [])
            )
        per_seed.append(seed_row)

    payload = {
        "format": "coordinator-baselines/compare/1",
        "dir": str(base),
        "left": args.left,
        "right": args.right,
        "seeds": seeds,
        "multipliers": multipliers,
        "delta_multipliers": delta_multipliers,
        "totals": totals,
        "flips": {str(value): flips[value] for value in multipliers},
        "delta_flips": {str(value): delta_flips[value] for value in delta_multipliers},
        "first_delta_flip": next(
            (value for value in delta_multipliers if delta_flips[value]["all"]), None
        ),
        "legal_bias_delta": {
            "count": len(legal_deltas),
            "median": round(statistics.median(legal_deltas), 4) if legal_deltas else None,
            "max": round(max(legal_deltas), 4) if legal_deltas else None,
        },
        "per_seed": per_seed,
    }
    out = (
        Path(args.out)
        if args.out
        else base
        / ("dev_coalition_compare.json" if base == HERE else "heldout_coalition_compare.json")
    )
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"== {args.left} vs {args.right} | seeds {len(seeds)} | paired {totals['paired']} "
        f"| unpaired {totals['unmatched_left']}/{totals['unmatched_right']} =="
    )
    print(
        f"  different action chosen           {totals['selection_diffs']} "
        f"(with a legal task {totals['legal_selection_diffs']})"
    )
    print(
        f"  legal-task decisions              left {totals['legal_left']} "
        f"right {totals['legal_right']}"
    )
    print(
        f"  residual vectors differ           {totals['bias_diff_rows']} rows "
        f"(legal {totals['bias_diff_rows_legal']}: idle "
        f"{totals['bias_diff_rows_legal_idle']} busy {totals['bias_diff_rows_legal_busy']}); "
        f"entries on legal actions {totals['index_diffs_legal']}, elsewhere "
        f"{totals['index_diffs_off_legal']}"
    )
    if legal_deltas:
        print(
            f"    |delta| on legal actions        median "
            f"{statistics.median(legal_deltas):.4f} max {max(legal_deltas):.4f}"
        )
    print("  counterfactual A: scale the whole residual of the right arm by m")
    for value in multipliers:
        row = flips[value]
        print(
            f"    m={value:<4g} winner flips {row['all']:>6} "
            f"(of which with a legal task {row['legal']})"
        )
    print("  counterfactual B: scale only what the coalition policy added by m")
    for value in delta_multipliers:
        row = delta_flips[value]
        marker = " <- first flip" if delta_flips[value]["all"] and not any(
            delta_flips[earlier]["all"] for earlier in delta_multipliers
            if earlier < value
        ) else ""
        print(
            f"    m={value:<4g} winner flips {row['all']:>6} "
            f"(of which with a legal task {row['legal']}){marker}"
        )
    print(f"compare written to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)

    runner = sub.add_parser("run", help="execute the missing runs")
    runner.add_argument("--mode", choices=("coarse", "full"), default="coarse")
    runner.add_argument("--cal", type=float, nargs="*", default=[],
                        help="task-logit offsets for --mode full")
    runner.add_argument("--coalition", action="store_true",
                        help="also run the CoalitionAwareCoordinator arms")
    runner.add_argument("--shard", default=None, help="i/n, e.g. 0/2")
    runner.set_defaults(func=run)

    reporter = sub.add_parser("report", help="measure everything on disk")
    reporter.add_argument("--mode", choices=("coarse", "full"), default="coarse")
    reporter.set_defaults(func=report)

    comparer = sub.add_parser(
        "compare", help="pair two arms' traces: did the residual change any choice?"
    )
    comparer.add_argument("--left", default=f"rs{HEURISTIC_SCALE:g}")
    comparer.add_argument("--right", default=f"coal{COALITION_SCALE:g}")
    comparer.add_argument(
        "--multipliers",
        type=float,
        nargs="*",
        default=[0.0, 1.0, 2.0, 4.0],
        help="multipliers for counterfactual A (whole residual) and B (the delta)",
    )
    comparer.add_argument(
        "--delta-multipliers", type=float, nargs="*", default=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    )
    comparer.add_argument(
        "--dir", default=None, help="directory holding the traces (default: this one)"
    )
    comparer.add_argument("--out", default=None, help="output JSON path")
    comparer.set_defaults(func=compare)

    args = parser.parse_args(argv)
    if getattr(args, "mode", None) == "full" and args.action == "run" and not args.cal:
        parser.error("--mode full needs the two chosen offsets, e.g. --cal 0.05 0.10")
    return args.func(args)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    os.environ.setdefault("MKL_NUM_THREADS", "6")
    raise SystemExit(main())