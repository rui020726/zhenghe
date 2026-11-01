"""Audit replayed joint runs against their recorded detection streams.

For every seed shared by the arms, asserts:
1. each lifecycle log is complete (has a run-final record);
2. the ordered detection fingerprint (timestamp, target_id, grid cell) is
   identical across arms and identical to the recorded stream;
3. the scenario map hash stamped in run_config matches the bundle's map hash.

Any violation exits non-zero, so this gates a metrics report::

    python -m integrated_sar.replay_audit \
        --logs "runs/.../rp_*.jsonl" --bundles "runs/.../streams/*.json"

The log/bundle file names carry the arm and seed as ``<prefix><arm>_s<seed>``
(the replay batches use ``rp_``/``ab_`` prefixes); anything that does not
parse is reported and skipped.
"""

from __future__ import annotations

import argparse
import glob as globmodule
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .errors import ConfigurationError

LOG_PATTERN = re.compile(r"^(?:rp|ab)_(?P<arm>.+)_s(?P<seed>\d+)\.jsonl$")
BUNDLE_PATTERN = re.compile(r"^s(?P<seed>\d+)\.json$")


def log_fingerprint(path: Path) -> tuple[list[tuple[float, str, tuple[int, int]]], str, bool]:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    detections = sorted(
        (float(e["timestamp"]), str(e["target_id"]), tuple(e["detail"]["grid_position"]))
        for e in events
        if e.get("event_type") == "detection"
    )
    run_config = next((e for e in events if e.get("event_type") == "run_config"), None)
    if run_config is None:
        return detections, "", False
    scenario = run_config.get("detail", {}).get("scenario", {})
    complete = any(
        e.get("event_type") == "final" and e.get("target_id") == "__run__" for e in events
    )
    return detections, str(scenario.get("map_sha256", "")), complete


def bundle_fingerprint(path: Path) -> tuple[list[tuple[float, str, tuple[int, int]]], str]:
    from .detection_replay import load_detection_stream

    stream = load_detection_stream(path)
    return (
        sorted(
            (e.detected_at, e.target_id, (e.grid_position.row, e.grid_position.col))
            for e in stream.events
        ),
        stream.map_sha256,
    )


def audit(
    log_paths: list[str], bundle_paths: list[str]
) -> tuple[int, int, list[str]]:
    """Returns (seeds verified, arms checked, failure messages)."""
    by_seed: dict[int, dict[str, Path]] = defaultdict(dict)
    problems: list[str] = []
    for raw in log_paths:
        path = Path(raw)
        match = LOG_PATTERN.match(path.name)
        if match is None:
            problems.append(f"{path.name}: not an <arm>_s<seed> lifecycle log, skipped")
            continue
        by_seed[int(match.group("seed"))][match.group("arm")] = path

    bundles: dict[int, Path] = {}
    for raw in bundle_paths:
        path = Path(raw)
        match = BUNDLE_PATTERN.match(path.name)
        if match is None:
            problems.append(f"{path.name}: not an s<seed>.json stream bundle, skipped")
            continue
        bundles[int(match.group("seed"))] = path

    verified = 0
    arms_checked = 0
    for seed in sorted(by_seed):
        arms = by_seed[seed]
        bundle = bundles.get(seed)
        if bundle is None:
            problems.append(f"seed {seed}: no stream bundle for arms {sorted(arms)}")
            continue
        recorded_detections, recorded_hash = bundle_fingerprint(bundle)
        reference: tuple[list[Any], str] | None = None
        for arm in sorted(arms):
            detections, map_hash, complete = log_fingerprint(arms[arm])
            arms_checked += 1
            if not complete:
                problems.append(f"seed {seed} arm {arm}: log truncated (no run-final record)")
            if detections != recorded_detections:
                problems.append(
                    f"seed {seed} arm {arm}: detections differ from the recorded stream"
                )
            if map_hash and map_hash != recorded_hash:
                problems.append(
                    f"seed {seed} arm {arm}: map hash {map_hash} != bundle {recorded_hash}"
                )
            if reference is None:
                reference = (detections, map_hash)
            elif (detections, map_hash) != reference:
                problems.append(f"seed {seed} arm {arm}: stream differs from arm {sorted(arms)[0]}")
        verified += 1
    return verified, arms_checked, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="integrated_sar.replay_audit",
        description="Assert replayed arms share the recorded detection stream.",
    )
    parser.add_argument("--logs", nargs="+", required=True, help="lifecycle log globs")
    parser.add_argument("--bundles", nargs="+", required=True, help="stream bundle globs")
    args = parser.parse_args(argv)

    log_paths = [p for pattern in args.logs for p in sorted(globmodule.glob(pattern))]
    bundle_paths = [p for pattern in args.bundles for p in sorted(globmodule.glob(pattern))]
    if not log_paths:
        raise ConfigurationError(f"no lifecycle logs matched {args.logs}")
    if not bundle_paths:
        raise ConfigurationError(f"no stream bundles matched {args.bundles}")

    verified, arms_checked, problems = audit(log_paths, bundle_paths)
    for problem in problems:
        print(f"FAIL {problem}")
    print(f"verified {verified} seeds, {arms_checked} arm logs, failures: {len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
