"""Reduce the held-out comparison of the coalition reservation arms to one small JSON.

``heldout.py report`` prints the human report and writes ``heldout_summary.json``, which
carries every per-seed lifecycle row and is therefore a runtime artefact: it lives under
``heldout/`` and is not versioned. What the verdict document quotes -- the arm totals, the
paired differences against ``rs0.25``, the gate table and the reservation columns -- is
reduced here into ``reservation_verdict.json``, a few kilobytes that can be diffed and
carried in git alongside ``verdict.json``.

Reads only artifacts the runs already produced. Writes one file next to this script.

    python runs/coordinator_baselines_20260917/reservation_verdict.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

HELD = HERE / "heldout"
STEPS = 1000
REFERENCE = "rs0.25"
ARMS = ("nocoord", "nocoord+cal0.25", REFERENCE, "coal0.25", "coalplan0.25")
# The gate the phase was set: against the strongest per-pair arm, expiries down by at
# least a fifth, group idling down, completion not worse.
EXPIRY_REDUCTION_TARGET = 0.20


def _paired(deltas: list[float]) -> dict[str, float]:
    from integrated_sar.lifecycle_metrics import _ci95

    mean, ci = _ci95(deltas)
    return {"mean": round(mean, 4), "ci95": round(ci, 4), "targets": round(sum(deltas), 1)}


def collect_lifecycle(seeds: list[int]) -> dict[str, dict[str, object]]:
    from integrated_sar.lifecycle_metrics import collect

    paths = [str(HELD / f"rp_{arm}_s{seed}.jsonl") for arm in ARMS for seed in seeds]
    table = collect(paths, [STEPS])
    rows_by_seed: dict[str, dict[int, object]] = {
        arm: {seed: row for seed, row in table.get(arm, {}).get(STEPS, {}).items()}
        for arm in ARMS
    }
    blocks: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        rows = [rows_by_seed[arm][seed] for seed in seeds if seed in rows_by_seed[arm]]
        discovered = sum(row.discovered for row in rows)
        completed = sum(row.completed for row in rows)
        times = [row.mean_completion_t for row in rows if row.mean_completion_t is not None]
        blocks[arm] = {
            "runs": len(rows),
            "discovered": discovered,
            "completed": completed,
            "expired": sum(row.expired for row in rows),
            "never_assigned": sum(row.unassigned for row in rows),
            "completion_rate": round(completed / discovered, 4) if discovered else None,
            "mean_completion_time": round(statistics.mean(times), 2) if times else None,
            "paired_vs_reference": {},
        }
    # Paired on the seed, not on position: an arm missing a seed must not shift every
    # difference after it onto the wrong pair.
    for arm in ARMS:
        if arm == REFERENCE:
            continue
        completion: list[float] = []
        expired: list[float] = []
        for seed in seeds:
            row = rows_by_seed[arm].get(seed)
            reference = rows_by_seed[REFERENCE].get(seed)
            if row is None or reference is None:
                continue
            if row.completion_rate is not None and reference.completion_rate is not None:
                completion.append(row.completion_rate - reference.completion_rate)
            expired.append(row.expired - reference.expired)
        blocks[arm]["paired_vs_reference"] = {
            "completion": _paired(completion),
            "expired": _paired(expired),
        }
    blocks[REFERENCE]["paired_vs_reference"] = {
        "completion": _paired([0.0]),
        "expired": _paired([0.0]),
    }
    return blocks


def collect_attribution() -> dict[str, dict[str, object]]:
    from integrated_sar.expired_attribution import attribute

    paths = [str(path) for path in sorted(HELD.glob(f"rp_*_s*.jsonl"))]
    attributions = attribute(paths, window=float(STEPS))
    return {
        arm: {
            "expired": attribution.expired_total,
            "classes": {key: value for key, value in attribution.counts.items() if value},
            "stall": attribution.stall_stats(),
            "latency": attribution.latency_stats(),
            "missing_capability_vector": missing_capability_vector(arm),
        }
        for arm, attribution in sorted(attributions.items())
        if arm in ARMS
    }


def missing_capability_vector(arm: str) -> dict[str, object]:
    """What expired targets still needed when their clock ran out, per ability dimension.

    Read from the lifecycle side, where the stage a target was in is tracked, and counted
    only over targets that expired. The decision trace cannot answer this: it only carries
    a ground-truth block for the target a decision was about, so its "last look" is
    usually the pre-arrival one and reports almost everyone as still open.

    The deficit is taken from the last progress report *at or before the deadline*, not
    from the final one. A target that finished late reports a completed stage at the end
    -- all zeros -- and counting that would say nothing was missing about the very targets
    that were missed. What was missing is what the last report before the deadline still
    showed, and an empty vector here means the expiry had no capability component at all.
    """
    from integrated_sar.expired_attribution import build_timelines, load_records

    counts: list[int] = []
    expired = 0
    at_deadline = 0
    for path in sorted(HELD.glob(f"rp_{arm}_s*.jsonl")):
        seed = int(path.stem.rsplit("_s", 1)[1])
        for timeline in build_timelines(load_records(path), arm=arm, seed=seed).values():
            row = timeline.as_row()
            if row.get("final_status") != "expired":
                continue
            expired += 1
            points = [point for point in timeline.progress if point.remaining]
            before = [point for point in points if point.timestamp <= timeline.deadline]
            if not before:
                continue
            at_deadline += 1
            last = before[-1].remaining
            if not counts:
                counts = [0] * len(last)
            for index in range(min(len(counts), len(last))):
                counts[index] += int(last[index] > 0)
    return {
        "expired_targets": expired,
        "targets_with_a_report_before_the_deadline": at_deadline,
        "targets": counts,
        "share": [round(value / at_deadline, 4) if at_deadline else None for value in counts],
    }


def collect_traces() -> dict[str, dict[str, object]]:
    from integrated_sar.allocation_trace import analyse, read_trace, summarise

    rows = []
    for path in sorted(HELD.glob("tr_*_s*.jsonl")):
        meta, decisions = read_trace(path)
        if decisions:
            rows.append(
                analyse(
                    decisions,
                    arm=str(meta.get("arm") or "unknown"),
                    seed=int(meta.get("seed") or 0),
                )
            )
    summary = summarise(rows) if rows else {}
    keep = (
        "reserved_decisions",
        "honoured_reservations",
        "broken_reservations",
        "honoured_share",
        "plan_totals",
        "dispatched_targets",
        "never_reserved",
        "never_reserved_share",
        "reservation_gap_median",
        "reservation_gap_p90",
        "arrival_gap_median",
        "group_arrival_gap_median",
        "progress_gap_median",
        "missing_capability_target_share",
        "oversupplied_share",
        "excess_units_median",
        "switched_before_arrival",
        "task_commitments",
    )
    return {arm: {key: block.get(key) for key in keep} for arm, block in sorted(summary.items())}


def gate(lifecycle: dict[str, dict[str, object]]) -> dict[str, object]:
    reference = lifecycle[REFERENCE]
    candidate = lifecycle["coalplan0.25"]
    expired_now, expired_ref = candidate["expired"], reference["expired"]
    reduction = (expired_ref - expired_now) / expired_ref if expired_ref else 0.0
    completion_delta = candidate["paired_vs_reference"]["completion"]
    return {
        "reference": REFERENCE,
        "expired_reduction_target": EXPIRY_REDUCTION_TARGET,
        "expired_reduction": round(reduction, 4),
        "expiry_gate": "pass" if reduction >= EXPIRY_REDUCTION_TARGET else "fail",
        "completion_delta": completion_delta,
        "completion_gate": (
            "pass"
            if completion_delta["mean"] - completion_delta["ci95"] > -1e-9
            else "fail"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(HERE / "reservation_verdict.json"))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(range(20000, 20100)))
    args = parser.parse_args(argv)

    lifecycle = collect_lifecycle(list(args.seeds))
    payload = {
        "format": "coalition-reservation-verdict/1",
        "dataset": {
            "seeds": [min(args.seeds), max(args.seeds)],
            "count": len(args.seeds),
            "source": "runs/coordinator_baselines_20260917/heldout (fresh seeds, "
            "recorded before this arm existed)",
        },
        "arms": lifecycle,
        "attribution": collect_attribution(),
        "decision_trace": collect_traces(),
        "gate": gate(lifecycle),
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for arm in ARMS:
        block = lifecycle[arm]
        print(
            f"  {arm:16} completed {block['completed']:>4} / {block['discovered']:>4}"
            f"  expired {block['expired']:>4}  rate {block['completion_rate']}"
        )
    print(f"  gate: {payload['gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
