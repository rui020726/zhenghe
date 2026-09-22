"""Run the nine locked arms over one pressure layer's 16 streams.

One cell is one episode: ``pressure_env.run_arm`` builds the frozen configuration, checks the
transport's own declaration of protocol, credit and channel before the first step, and records
the launch whether the cell succeeds or fails. Rows are appended to a per-shard JSONL so an
interrupted shard can resume -- the resume key is **(arm, seed)**, never arm alone: a check that
asks "have I run this arm?" would run the first stream of every arm and skip the rest, loudly
reporting a complete matrix.

    python pressure_matrix.py --goals 6 --shards 4 --shard 0
    python pressure_matrix.py --goals 6 --mode coverage
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pressure_common as common
import pressure_env
import record_layer

#: Every artefact path a row carries is absolute; the report rewrites it under this marker.
ARTEFACT_KEY = "artefacts"


def rows_path(goals: int, shard: int) -> Path:
    return common.layer_rows(goals) / f"rows_g{goals}_shard{shard}.jsonl"


def load_rows(goals: int) -> dict[tuple[str, int], dict[str, Any]]:
    """Read every shard's rows for a layer, de-duplicating by (arm, seed)."""
    seen: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(common.layer_rows(goals).glob(f"rows_g{goals}_shard*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (str(row["arm"]), int(row["seed"]))
                if key in seen and seen[key] != row:
                    raise RuntimeError(
                        f"{path.name}: cell {key} was written twice with different content; "
                        "the same seed cannot belong to two shards"
                    )
                seen[key] = row
    return seen


def cells(goals: int) -> list[tuple[str, int]]:
    return [(arm, seed) for seed in common.LAYERS[goals] for arm in common.ARM_ORDER]


def run(goals: int, shard: int, shards: int) -> dict[str, Any]:
    record_layer.require_clean_audit()
    manifest = common.json_path(f"STREAM_MANIFEST_g{goals}.json")
    if not manifest.exists():
        raise SystemExit(
            f"STREAM_MANIFEST_g{goals}.json is missing: the layer's streams must be recorded and "
            "merged before any arm runs on them"
        )
    registered = json.loads(manifest.read_text(encoding="utf-8"))
    if registered["illegal_streams"]:
        raise SystemExit(
            f"layer {goals} has illegal streams and L0 cannot pass: {registered['illegal_streams']}"
        )
    by_seed = {int(row["seed"]): row for row in registered["streams"]}
    common.layer_rows(goals).mkdir(parents=True, exist_ok=True)
    common.layer_runs(goals).mkdir(parents=True, exist_ok=True)
    done = load_rows(goals)
    path = rows_path(goals, shard)
    path.touch(exist_ok=True)
    mine = common.shard_of(cells(goals), shards, shard)
    started = time.time()
    written = 0
    failures: list[str] = []
    with path.open("a", encoding="utf-8") as handle:
        for arm, seed in mine:
            if (arm, seed) in done:
                continue
            if seed not in by_seed:
                raise SystemExit(f"seed {seed} is in the matrix but not in the layer manifest")
            try:
                row = pressure_env.run_arm(goals, seed, arm)
            except Exception as error:  # noqa: BLE001 - a failed cell is reported, not skipped
                failures.append(f"{arm} s{seed}: {type(error).__name__}: {str(error)[:200]}")
                print(f"[FAIL g{goals} {arm} s{seed}] {error}", file=sys.stderr, flush=True)
                continue
            row["stream_detected"] = int(by_seed[seed]["detected"])
            row["stream_placed"] = int(by_seed[seed]["placed"])
            row["stream_last_detection_step"] = by_seed[seed]["last_detection_step"]
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            written += 1
            done[(arm, seed)] = row
            print(
                f"[g{goals} {arm} s{seed}] on_time={row['on_time']} "
                f"bytes={row['spent_bytes']} credit={row['budget_used_fraction']} "
                f"comp={row['probe']['competition_points'] if row['probe'] else None} "
                f"{row['seconds']}s",
                flush=True,
            )
    return {
        "goals": goals,
        "shard": shard,
        "shards": shards,
        "cells_assigned": len(mine),
        "cells_written": written,
        "failures": failures,
        "seconds": round(time.time() - started, 1),
    }


def coverage(goals: int) -> dict[str, Any]:
    """What the layer actually has: every (arm, seed) present, per arm, plus the gaps."""
    done = load_rows(goals)
    expected = cells(goals)
    missing = [f"{arm} s{seed}" for arm, seed in expected if (arm, seed) not in done]
    by_arm: dict[str, int] = {arm: 0 for arm in common.ARM_ORDER}
    for (arm, _seed) in done:
        by_arm[arm] += 1
    seeds = sorted({seed for _arm, seed in done})
    return {
        "goals": goals,
        "cells_expected": len(expected),
        "cells_present": len(done),
        "per_arm": by_arm,
        "distinct_seeds": len(seeds),
        "missing_cells": missing[:20],
        "missing_count": len(missing),
        "complete": not missing,
        "ledger": common.ledger_totals(),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goals", type=int, required=True, choices=sorted(common.LAYERS))
    parser.add_argument("--mode", choices=("run", "coverage"), default="run")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "coverage":
        report = coverage(args.goals)
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0 if report["complete"] else 1
    if args.shard >= args.shards:
        parser.error(f"--shard {args.shard} is outside --shards {args.shards}")
    report = run(args.goals, args.shard, args.shards)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
