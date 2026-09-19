"""Shared fixtures for the communication-control round.

Everything here *reads* the frozen configuration of ``runs/corrected_env_final_20260918``:
that block's numbers are this round's baseline, so the allocation side must not drift.
The arm configuration is imported from the frozen block's own ``arms.py`` rather than
restated here, and the checkpoints are hash-checked against ``ARMS_CONFIG_TRUTH.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS_BLOCK = ROOT / "runs" / "corrected_env_final_20260918"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Thread caps must be set before torch is imported anywhere in the process.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

if str(PREVIOUS_BLOCK) not in sys.path:
    sys.path.insert(0, str(PREVIOUS_BLOCK))
import arms  # noqa: E402

# Development streams: the 400-seed block 30000..30399 that the membership round already
# used for development. 500000..500399 is the previous round's registered independent test
# set and stays untouched; 600000+ are live-system seeds.
DEV_STREAM_DIR = ROOT / "runs" / "coalition_validation_20260918" / "streams"
DEV_SEEDS = tuple(range(30000, 30040))

TRUTH = json.loads((PREVIOUS_BLOCK / "ARMS_CONFIG_TRUTH.json").read_text(encoding="utf-8"))
FROZEN = TRUTH["frozen"]
CHECKPOINTS = {
    name: str((ROOT / relative).resolve())
    for name, relative in (
        ("piloc", FROZEN["piloc_checkpoint"]),
        ("ctde", FROZEN["ctde_checkpoint"]),
    )
}
CHECKPOINT_SHA256 = {
    name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for name, path in CHECKPOINTS.items()
}
for _name, _digest in CHECKPOINT_SHA256.items():
    if TRUTH["checkpoint_sha256"][_name] != _digest:
        raise RuntimeError(
            f"{_name} checkpoint drifted: recorded {TRUTH['checkpoint_sha256'][_name]}, "
            f"found {_digest}"
        )


def dev_stream(seed: int) -> Path:
    path = DEV_STREAM_DIR / f"s{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"no development stream for seed {seed}: {path}")
    return path


def base_configuration(
    seed: int,
    *,
    directory: Path,
    arm: str = "C",
    **overrides: Any,
) -> dict[str, Any]:
    """The frozen arm's configuration, replaying this seed's recorded detection stream.

    ``arm`` is "C" (corrected environment + binding coalition planner, no release
    mechanism) or "D" (plus the three release mechanisms). C is this round's baseline: the
    communication question is asked where the allocation side is frozen and simplest.
    """
    values = dict(arms.ARMS[arm].configuration)
    values.update(
        task_checkpoint=CHECKPOINTS["ctde"],
        replay_stream=str(dev_stream(seed)),
        seed=int(seed),
        output=str(Path(directory) / f"rp_{arm}_s{seed}.jsonl"),
        decision_trace=str(Path(directory) / f"tr_{arm}_s{seed}.jsonl"),
    )
    values.update(overrides)
    return values


def write_json(path: Path, payload: Any) -> Path:
    """Write atomically: a half-written summary is worse than no summary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


__all__ = [
    "HERE",
    "ROOT",
    "PREVIOUS_BLOCK",
    "DEV_STREAM_DIR",
    "DEV_SEEDS",
    "TRUTH",
    "FROZEN",
    "CHECKPOINTS",
    "CHECKPOINT_SHA256",
    "arms",
    "base_configuration",
    "dev_stream",
    "write_json",
]
