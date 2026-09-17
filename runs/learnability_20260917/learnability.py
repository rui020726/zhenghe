"""Learnability gate for the learned coalition coordinator: attribute, oracle, decide.

The previous phase left 16 expired targets on the held-out block and a question: is the
rule planner's *ordering* the binding constraint, or is the information it sees not enough
to order better? Training a network before answering that would be spending GPU time on a
hypothesis the data may already refute.

Four subcommands, each writing artefacts under this directory:

* ``manifest``  -- write ``manifest.json``, the split registry with seeds, checkpoints and
  their digests, the code commit and the recording protocol. This is what makes "fresh"
  checkable rather than asserted.
* ``record``    -- record the fresh analysis streams with the frozen PILOC checkpoint.
* ``run``       -- run the rule planner and the same-information oracle arms on a split.
* ``verdict``   -- attribute every expired target of the planner, compare the arms paired
  per seed, and write ``LEARNABILITY_VERDICT.md`` plus ``learnability_verdict.json``.

The hindsight ceiling is ``integrated_sar.expired_attribution``, which no online module
imports; ``test_learnability_gate`` pins that.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from integrated_sar.data_splits import (  # noqa: E402
    CTDE_CHECKPOINT,
    PILOC_CHECKPOINT,
    RECORDING_PROTOCOL,
    SPLITS,
    run_dir,
    split_of,
    stream_path,
)

STEPS = 1000
PLANNER_SCALE = 0.25
# (label, policy, binding). The third entry turns the commitment into something the
# allocator honours; the first two stay advisory so the comparison is against the arm the
# learnability gate measured.
ARMS = (
    ("coalplan0.25", "coalition_planner", False),
    ("coalplan0.25sticky", "coalition_planner", True),
    ("daw0.25", "deadline_aware", False),
)
# The gate the phase was set. The oracle may only reorder legal candidates, so the share
# of the planner's remaining expiries it removes is the measure of learnability.
EXPIRY_FIX_TARGET = 0.30
# Attribution classes, in the order the verdict reports them.
FAILURE_CLASSES = (
    "candidate_ordering_error",
    "task_contention_error",
    "missing_or_late_belief",
    "travel_or_deadline_impossible",
    "reservation_or_execution_break",
    "capacity_conflict",
)


def manifest() -> dict[str, object]:
    """The split registry, with the digests that make a split reproducible."""
    import hashlib

    def digest(path: Path) -> str | None:
        if not path.exists():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "format": "learnability-manifest/1",
        "code_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip(),
        "steps": STEPS,
        "checkpoints": {
            "piloc": {"path": str(PILOC_CHECKPOINT), "sha256": digest(PILOC_CHECKPOINT)},
            "ctde": {"path": str(CTDE_CHECKPOINT), "sha256": digest(CTDE_CHECKPOINT)},
        },
        "splits": {
            name: {
                "purpose": split.purpose,
                "frozen": split.frozen,
                "seeds": list(split.seeds),
                "stream_dir": str(split.stream_dir(REPO)),
                "recorded_with": RECORDING_PROTOCOL,
            }
            for name, split in SPLITS.items()
        },
        "arms": {label: policy for label, policy, _ in ARMS},
        "gate": {
            "same_information_expiry_fix_target": EXPIRY_FIX_TARGET,
            "metric": "share of the rule planner's remaining expired targets removed by "
            "the same-information oracle, paired per seed",
        },
    }


def command(seed: int, label: str, policy: str, split_name: str, binding: bool = False) -> list[str]:
    """The joint_driver argv for one arm on one recorded stream."""
    return [
        sys.executable or "python",
        "-m",
        "integrated_sar.joint_driver",
        "--task-checkpoint",
        str(CTDE_CHECKPOINT),
        "--actor-kv-mode",
        "tokens",
        "--steps",
        str(STEPS),
        "--seed",
        str(seed),
        "--replay-stream",
        str(stream_path(REPO, split_name, seed)),
        "--quiet",
        "--output",
        str(run_dir(REPO, split_name) / f"rp_{label}_s{seed}.jsonl"),
        "--residual-scale",
        f"{PLANNER_SCALE:g}",
        "--coordinator",
        policy,
        "--decision-trace",
        str(run_dir(REPO, split_name) / f"tr_{label}_s{seed}.jsonl"),
    ]
    if binding:
        argv.append("--sticky-commitments")


def record_command(seed: int, split_name: str) -> list[str]:
    """One live PILOC pass; its detections are the stream every arm of the split replays."""
    return [
        sys.executable or "python",
        "-m",
        "integrated_sar.joint_driver",
        "--piloc-checkpoint",
        str(PILOC_CHECKPOINT),
        "--task-checkpoint",
        str(CTDE_CHECKPOINT),
        "--actor-kv-mode",
        "tokens",
        "--steps",
        str(STEPS),
        "--seed",
        str(seed),
        "--no-coordinator",
        "--quiet",
        "--record-stream",
        str(stream_path(REPO, split_name, seed)),
        "--output",
        str(run_dir(REPO, args.split) / f"rec_s{seed}.jsonl"),
    ]


def shard(jobs: list, spec: str | None) -> list:
    if not spec:
        return jobs
    index, count = (int(part) for part in spec.split("/"))
    return [job for position, job in enumerate(jobs) if position % count == index]


def _run_batch(jobs: list[list[str]], args: argparse.Namespace) -> int:
    for position, argv in enumerate(shard(jobs, args.shard), start=1):
        print(f"[{position}/{len(jobs)}] {' '.join(argv[-4:])}", flush=True)
        result = subprocess.run(argv, cwd=REPO)
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})", flush=True)
            return result.returncode
    return 0


def do_record(args: argparse.Namespace) -> int:
    split = split_of(args.split)
    run_dir(REPO, args.split).mkdir(parents=True, exist_ok=True)
    split.stream_dir(REPO).mkdir(parents=True, exist_ok=True)
    jobs = [
        record_command(seed, args.split)
        for seed in split.seeds
        if not stream_path(REPO, args.split, seed).exists()
    ]
    print(f"{len(jobs)} streams to record for split {args.split!r}")
    return _run_batch(jobs, args)


def do_run(args: argparse.Namespace) -> int:
    split = split_of(args.split)
    run_dir(REPO, args.split).mkdir(parents=True, exist_ok=True)
    missing = [
        seed for seed in split.seeds if not stream_path(REPO, args.split, seed).exists()
    ]
    if missing:
        print(f"{len(missing)} streams missing; run the record pass first")
        return 1
    jobs = [
        command(seed, label, policy, args.split, binding)
        for label, policy, binding in ARMS
        for seed in split.seeds
        if not (run_dir(REPO, args.split) / f"rp_{label}_s{seed}.jsonl").exists()
        or args.force
    ]
    print(f"{len(jobs)} arm runs for split {args.split!r}")
    return _run_batch(jobs, args)


def do_manifest(args: argparse.Namespace) -> int:
    payload = manifest()
    out = HERE / "manifest.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for name, split in payload["splits"].items():  # type: ignore[union-attr]
        print(f"  {name:16} {len(split['seeds']):>4} seeds  {split['purpose']}")
    return 0


def do_verdict(args: argparse.Namespace) -> int:
    from integrated_sar.learnability import build_verdict

    payload = build_verdict(REPO, splits=args.splits)
    out = HERE / "learnability_verdict.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    gate = payload["gate"]
    print(f"  expiry fix share {gate['same_information_expiry_fix_share']} "
          f"(target {gate['target']}) -> {gate['decision']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)

    maker = sub.add_parser("manifest", help="write manifest.json")
    maker.set_defaults(func=do_manifest)

    recorder = sub.add_parser("record", help="record the fresh analysis streams")
    recorder.add_argument("--split", default="analysis_fresh")
    recorder.add_argument("--shard", default=None, help="i/n")
    recorder.set_defaults(func=do_record)

    runner = sub.add_parser("run", help="run the planner and oracle arms on a split")
    runner.add_argument("--split", default="analysis_fresh")
    runner.add_argument("--shard", default=None, help="i/n")
    runner.add_argument("--force", action="store_true", help="re-run existing artefacts")
    runner.set_defaults(func=do_run)

    reporter = sub.add_parser("verdict", help="attribute, compare and decide")
    reporter.add_argument("--splits", nargs="*", default=["analysis_legacy", "analysis_fresh"])
    reporter.set_defaults(func=do_verdict)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
