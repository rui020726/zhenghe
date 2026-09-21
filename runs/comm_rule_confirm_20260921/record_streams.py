"""Record this round's own 200 detection streams, once, after a clean block audit.

The engineering confirmation needs data no earlier round has seen, and the brief forbids
swapping the block after looking -- so the block was audited first (``block_audit.py``) and
the recording configuration is the frozen block's own constant, not this round's choice: live
PILOC search, coordinator off, the membership-consistent environment. A stream recorded under
different bookkeeping would carry that bookkeeping's arrivals into every arm of every setting.

The recorder is idempotent (an existing stream is reported, never rewritten) so a technical
failure can be re-run on the *same* seed, which is the only recovery the protocol allows: no
dropping hard seeds, no topping up with prettier ones. Every re-run is registered with its
reason in ``COMM_RULE_STREAMS.json``.

    python record_streams.py --shards 4 --shard 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import confirm_common as common

from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402

#: Modules whose code identity is frozen into the stream registration. Judged by git blob
#: identity, the way the round manifests judge executed code.
RECORDED_MODULES = (
    "integrated_sar/joint_driver.py",
    "integrated_sar/contracts.py",
    "integrated_sar/task_registry.py",
    "integrated_sar/orchestrator.py",
    "integrated_sar/adapters/task03.py",
    "integrated_sar/adapters/piloc.py",
    "integrated_sar/comm.py",
    "integrated_sar/scenarios.py",
    "integrated_sar/piloc_runtime.py",
    "integrated_sar/search_runtime.py",
    "integrated_sar/lifecycle_log.py",
)

#: The only reasons a re-run is allowed. Anything else stops the round.
RERUN_REASONS = (
    "process_killed",
    "disk_full",
    "checkpoint_load_failure",
)


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=str(common.ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def code_identity() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    modules: dict[str, Any] = {}
    for relative in RECORDED_MODULES:
        path = common.ROOT / relative
        blob = git("hash-object", "--", relative)
        try:
            at_head = git("rev-parse", f"HEAD:{relative}")
        except subprocess.CalledProcessError:
            at_head = "<absent-at-head>"
        modules[relative] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "blob": blob,
            "blob_at_head": at_head,
            "matches_head": blob == at_head,
        }
    return {
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": head,
        "comparison": "git blob identity (line-ending filters applied)",
        "modules": modules,
        "all_modules_match_head": all(value["matches_head"] for value in modules.values()),
    }


def record(seed: int, *, reason: str | None = None) -> dict[str, Any]:
    stream = common.stream_path(seed)
    if stream.exists() and reason is None:
        # The counts are read from the file, not assumed: a skipped stream is still part of
        # the block and its registration must carry the same fields as a fresh one.
        payload = json.loads(stream.read_text(encoding="utf-8"))
        return {
            "seed": seed,
            "skipped": True,
            "sha256": common.sha256(stream),
            "frames": len(payload.get("frames", [])),
            "detections": sum(
                len(frame.get("detections", [])) for frame in payload.get("frames", [])
            ),
            "steps_recorded": int(payload.get("steps_recorded", 0)),
            "note": "already recorded; the recorder never rewrites a stream",
        }
    if stream.exists() and reason is not None:
        if reason not in RERUN_REASONS:
            raise RuntimeError(
                f"seed {seed}: re-run reason {reason!r} is not one of the pre-registered "
                f"reasons {RERUN_REASONS}; stop and report instead"
            )
        quarantine = common.HERE / "stream_reruns" / f"s{seed}_{reason}"
        quarantine.mkdir(parents=True, exist_ok=True)
        stream.replace(quarantine / stream.name)
    logs = common.HERE / "record_logs"
    logs.mkdir(parents=True, exist_ok=True)
    configuration = RunConfiguration(
        seed=seed,
        output=str(logs / f"rec_s{seed}.jsonl"),
        decision_trace=str(logs / f"rec_tr_s{seed}.jsonl"),
        record_stream=str(stream),
        **common.RECORD_CONFIGURATION,
    )
    started = time.time()
    summary = build_joint_run(configuration).run()
    if not stream.exists():
        raise AssertionError(f"seed {seed}: the recorder produced no stream")
    payload = json.loads(stream.read_text(encoding="utf-8"))
    return {
        "seed": seed,
        "skipped": False,
        "rerun_reason": reason,
        "sha256": common.sha256(stream),
        "frames": len(payload.get("frames", [])),
        "detections": sum(len(frame.get("detections", [])) for frame in payload.get("frames", [])),
        "steps_recorded": int(payload.get("steps_recorded", 0)),
        "run_steps": int(summary.get("steps", 0)),
        "run_time": round(float(summary.get("current_time", 0.0)), 3),
        "targets_discovered_by_registry": int(summary.get("targets_discovered", 0)),
        "seconds": round(time.time() - started, 1),
    }


def fold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One digest over (seed, stream bytes) -- the fold the manifest and results quote."""
    digest = hashlib.sha256()
    total_bytes = 0
    for row in sorted(rows, key=lambda item: int(item["seed"])):
        payload = common.stream_path(int(row["seed"])).read_bytes()
        total_bytes += len(payload)
        digest.update(str(int(row["seed"])).encode("utf-8"))
        digest.update(payload)
    return {
        "streams": len(rows),
        "bytes": total_bytes,
        "sha256_fold": digest.hexdigest(),
    }


def merge() -> dict[str, Any]:
    """Consolidate the shard registrations into one read-only registration.

    Verifies rather than trusts: exactly the 200 block seeds, one registration each, every
    sha256 recomputed from the file on disk, and one fold over the whole block. A stream that
    is missing or whose bytes moved fails here instead of reaching the matrix.
    """
    shards = sorted(common.HERE.glob("COMM_RULE_STREAMS_shard*.json"))
    if not shards:
        raise SystemExit("no COMM_RULE_STREAMS_shard*.json found; record the streams first")
    rows: list[dict[str, Any]] = []
    for path in shards:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["streams"])
    by_seed: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row["seed"])
        if seed in by_seed:
            raise SystemExit(f"seed {seed} registered twice; the block was recorded twice")
        by_seed[seed] = row
    expected = set(common.BLOCK_SEEDS)
    missing = sorted(expected - set(by_seed))
    extra = sorted(set(by_seed) - expected)
    if missing or extra:
        raise SystemExit(f"block mismatch: missing={missing[:5]} extra={extra[:5]}")
    for seed in sorted(by_seed):
        row = by_seed[seed]
        observed = common.sha256(common.stream_path(seed))
        if observed != row["sha256"]:
            raise SystemExit(
                f"seed {seed}: the file on disk hashes to {observed[:10]}, the registration "
                f"says {str(row['sha256'])[:10]}; a stream changed after it was registered"
            )
    ordered = [by_seed[seed] for seed in sorted(by_seed)]
    # The merged registration describes the block, not one recording pass: the pass-specific
    # "skipped" flag and timing would only say which process happened to see the file first.
    for row in ordered:
        row.pop("skipped", None)
        row.pop("seconds", None)
        row.pop("note", None)
    first = json.loads(shards[0].read_text(encoding="utf-8"))
    payload = {
        "block": [common.BLOCK_START, common.BLOCK_STOP - 1],
        "size": common.BLOCK_STOP - common.BLOCK_START,
        "purpose": first["purpose"],
        "block_audit": first["block_audit"],
        "recording_configuration": first["recording_configuration"],
        "executed_code": first["executed_code"],
        "rerun_policy": first["rerun_policy"],
        "streams": ordered,
        "fold": fold(ordered),
        "shards": [path.name for path in shards],
        "seeds_with_no_detections": [
            int(row["seed"]) for row in ordered if int(row.get("detections", 0)) == 0
        ],
    }
    common.write_json(common.HERE / "COMM_RULE_STREAMS.json", payload)
    return payload


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("record", "merge"), default="record")
    parser.add_argument("--start", type=int, default=common.BLOCK_START)
    parser.add_argument("--stop", type=int, default=common.BLOCK_STOP)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument(
        "--rerun-seed",
        type=int,
        default=None,
        help="re-record one seed for a pre-registered reason (see RERUN_REASONS)",
    )
    parser.add_argument("--rerun-reason", default=None)
    args = parser.parse_args()

    if args.mode == "merge":
        payload = merge()
        print(
            json.dumps(
                {
                    "streams": len(payload["streams"]),
                    "fold": payload["fold"],
                    "seeds_with_no_detections": payload["seeds_with_no_detections"],
                    "output": "COMM_RULE_STREAMS.json",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    if args.rerun_seed is not None and args.rerun_reason is None:
        parser.error("--rerun-seed requires --rerun-reason")
    if args.rerun_seed is None and args.rerun_reason is not None:
        parser.error("--rerun-reason requires --rerun-seed")

    common.STREAMS.mkdir(parents=True, exist_ok=True)
    audit = common.read_json("COMM_RULE_BLOCK_AUDIT.json")
    if not audit.get("clean"):
        raise SystemExit(
            "the block audit is not clean; recording is not allowed. "
            f"verdict: {audit.get('verdict')}"
        )

    if args.rerun_seed is not None:
        seeds = [args.rerun_seed]
    else:
        seeds = [
            seed
            for index, seed in enumerate(range(args.start, args.stop))
            if index % max(1, args.shards) == args.shard % max(1, args.shards)
        ]

    rows: list[dict[str, Any]] = []
    started = time.time()
    for seed in seeds:
        row = record(seed, reason=args.rerun_reason if args.rerun_seed == seed else None)
        rows.append(row)
        print(
            f"[record s{seed}] detections={row.get('detections')} "
            f"sha={str(row.get('sha256'))[:10]} {row.get('seconds')}s"
            + (" (already there)" if row.get("skipped") else "")
            + (f" (rerun: {row.get('rerun_reason')})" if row.get("rerun_reason") else ""),
            flush=True,
        )

    payload = {
        "block": [common.BLOCK_START, common.BLOCK_STOP - 1],
        "size": common.BLOCK_STOP - common.BLOCK_START,
        "purpose": audit["candidate_block"]["purpose"],
        "block_audit": {
            "clean": audit["clean"],
            "verdict": audit["verdict"],
            "sha256": common.sha256(common.HERE / "COMM_RULE_BLOCK_AUDIT.json"),
        },
        "recording_configuration": dict(common.RECORD_CONFIGURATION),
        "executed_code": code_identity(),
        "rerun_policy": {
            "allowed_reasons": list(RERUN_REASONS),
            "rule": (
                "same seed only, for a pre-registered reason; no seed is dropped for being "
                "hard and no replacement sample is drawn"
            ),
        },
        "streams": rows,
        "fold": fold(rows),
        "shard": args.shard,
        "shards": args.shards,
        "seconds": round(time.time() - started, 1),
    }
    common.write_json(common.HERE / f"COMM_RULE_STREAMS_shard{args.shard}.json", payload)
    print(
        json.dumps(
            {
                "recorded": sum(1 for row in rows if not row.get("skipped")),
                "skipped": sum(1 for row in rows if row.get("skipped")),
                "fold": payload["fold"],
                "output": f"COMM_RULE_STREAMS_shard{args.shard}.json",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
