"""Does the refactored relay change what the frozen pipeline does?

Answerable only against the frozen code, so this script runs the same configuration twice --
once from the freeze commit checked out in a temporary worktree, once from this working tree
-- and compares the lifecycle records that carry decisions and outcomes. It ignores only the
run-identity record and the run-level metrics blob, both of which are expected to say more
than they used to.

The seeds are development seeds, and the comparison is a technical identity check on a code
refactor: no result is claimed from it and no per-seed choice is made with it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import common

OLD_ROOT = (common.ROOT.parent / ".zhenghe_frozen_ref").resolve()
PYTHON = Path(sys.executable)
IGNORED_RECORDS = {"run_config"}


def command(seed: int, directory: Path, *, label: str) -> list[str]:
    stream = common.dev_stream(seed)
    return [
        str(PYTHON),
        "-m",
        "integrated_sar.joint_driver",
        "--seed",
        str(seed),
        "--steps",
        "1000",
        "--device",
        "cpu",
        "--task-checkpoint",
        common.CHECKPOINTS["ctde"],
        "--replay-stream",
        str(stream),
        "--coordinator",
        "coalition_planner",
        "--sticky-commitments",
        "--residual-scale",
        "0.25",
        "--task-membership-consistency",
        "--coordination-interval",
        "5",
        "--allocation-interval",
        "5",
        "--command-ttl",
        "10",
        "--dt",
        "1",
        "--detection-radius",
        "5",
        "--search-agents",
        "2",
        "--num-goals",
        "6",
        "--deadline-tightness",
        "0.5",
        "2.0",
        "--grid-size",
        "60",
        "--obstacle-density",
        "0.15",
        "--rescue-species",
        "3",
        "--rescue-per-species",
        "2",
        "--actor-kv-mode",
        "tokens",
        "--comm-delivery-probability",
        "1.0",
        "--comm-delay-ticks",
        "0",
        "--comm-subscribers",
        "all",
        "--output",
        str(directory / f"rp_{label}_s{seed}.jsonl"),
        "--quiet",
    ]


def comparable(path: Path) -> list[dict[str, Any]]:
    """The decision- and outcome-bearing records, with the run identity stripped out."""
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event_type") in IGNORED_RECORDS:
                continue
            detail = record.get("detail")
            if isinstance(detail, dict) and "metrics" in detail:
                detail = {key: value for key, value in detail.items() if key != "metrics"}
                record = {**record, "detail": detail}
            records.append(record)
    return records


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(common.DEV_SEEDS[:8]))
    args = parser.parse_args(argv)

    directory = common.HERE / "identity"
    (directory / "old").mkdir(parents=True, exist_ok=True)
    (directory / "new").mkdir(parents=True, exist_ok=True)
    if not OLD_ROOT.exists():
        raise SystemExit(f"no frozen reference worktree at {OLD_ROOT}")

    rows = []
    for seed in args.seeds:
        for root, label in ((OLD_ROOT, "old"), (common.ROOT, "new")):
            completed = subprocess.run(
                command(seed, directory / label, label="C"),
                cwd=str(root),
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                print(f"{label} seed {seed} failed: {completed.stderr[-800:]}")
                raise SystemExit(1)
        old = comparable(directory / "old" / f"rp_C_s{seed}.jsonl")
        new = comparable(directory / "new" / f"rp_C_s{seed}.jsonl")
        first_difference = next(
            (
                (index, left, right)
                for index, (left, right) in enumerate(zip(old, new))
                if left != right
            ),
            None,
        )
        row = {
            "seed": seed,
            "old_records": len(old),
            "new_records": len(new),
            "identical": old == new,
            "first_difference_timestamp": (
                None if first_difference is None else first_difference[1].get("timestamp")
            ),
            "first_difference": None if first_difference is None else first_difference[1:],
        }
        rows.append(row)
        print(
            f"seed={seed} records={len(old)}/{len(new)} identical={row['identical']}"
            + (
                ""
                if row["identical"]
                else f" first_difference_at={row['first_difference_timestamp']}"
            ),
            flush=True,
        )
    payload = {
        "frozen_reference_commit": "a26e5b54990e2d3b2ead8c542057d6cfa2b23922",
        "seeds": args.seeds,
        "all_identical": all(row["identical"] for row in rows),
        "rows": rows,
    }
    common.write_json(common.HERE / "IDENTITY_CHECK.json", payload)
    print("all_identical:", payload["all_identical"])
    return 0 if payload["all_identical"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
