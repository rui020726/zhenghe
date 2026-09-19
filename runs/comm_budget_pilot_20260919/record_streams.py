"""Record this round's own detection streams with the frozen PILOC policy.

The pilot needs data that no earlier round has seen, on both sides of its split, so the
streams are recorded here rather than borrowed. The recording configuration is the frozen
block's own constant -- live PILOC, coordinator off, membership-consistent rescue
environment -- because a stream recorded under the old bookkeeping would carry the old
bookkeeping's arrivals into every arm of a later comparison.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pilot_common as common

from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402


def record(seed: int) -> dict[str, Any]:
    stream = common.stream_path(seed)
    if stream.exists():
        return {"seed": seed, "skipped": True, "sha256": common.sha256(stream)}
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
    import json

    payload = json.loads(stream.read_text(encoding="utf-8"))
    row = {
        "seed": seed,
        "skipped": False,
        "sha256": common.sha256(stream),
        "frames": len(payload.get("frames", [])),
        "detections": sum(len(frame.get("detections", [])) for frame in payload.get("frames", [])),
        "steps_recorded": int(payload.get("steps_recorded", 0)),
        "run_steps": int(summary.get("steps", 0)),
        "seconds": round(time.time() - started, 1),
    }
    (logs / f"rec_s{seed}.json").write_text(
        __import__("json").dumps(row, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return row


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(common.TRAIN_SEEDS) + list(common.VALIDATION_SEEDS))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args(argv)
    common.STREAMS.mkdir(parents=True, exist_ok=True)
    seeds = [
        seed
        for index, seed in enumerate(args.seeds)
        if index % max(1, args.shards) == args.shard % max(1, args.shards)
    ]
    for seed in seeds:
        row = record(seed)
        print(
            f"[record s{seed}] detections={row.get('detections')} sha={str(row.get('sha256'))[:10]} "
            f"{row.get('seconds')}s"
            + (" (already there)" if row.get("skipped") else ""),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
