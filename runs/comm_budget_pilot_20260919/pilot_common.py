"""Fixtures for the budget-pilot round, read from the two rounds it builds on.

Nothing here restates a frozen number. The allocation arm configuration, the checkpoint hashes
and the stream format come from the rounds that fixed them; this module only adds what is new
to this round -- its own directory, its seed blocks, and the rule that a budget must be a
constant fixed on calibration data rather than a quantity derived from the episode being
scored.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS_ROUND = ROOT / "runs" / "comm_control_20260919"
FROZEN_BLOCK = ROOT / "runs" / "corrected_env_final_20260918"

for _directory in (str(ROOT), str(PREVIOUS_ROUND), str(FROZEN_BLOCK)):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

# The previous round's fixtures import the frozen block's arms module and refuse to load if a
# checkpoint has drifted, so importing it is itself a verification step.
base = importlib.import_module("common")
_corrected = importlib.import_module("corrected_block")
# Last round's analysis readers, imported rather than restated so the definition of a cost,
# a belief age or a terminal time cannot drift between the two rounds.
metrics = importlib.import_module("comm_metrics")

ROOT = base.ROOT
FROZEN = base.FROZEN
CHECKPOINTS = base.CHECKPOINTS
CHECKPOINT_SHA256 = base.CHECKPOINT_SHA256
RECORD_CONFIGURATION = dict(_corrected.RECORD_CONFIGURATION)
STREAM_STEPS = int(RECORD_CONFIGURATION["steps"])

# Seed blocks. 700000+ are streams recorded for this round and used by nothing before it:
# the training half is what the controller sees, the validation half selects checkpoints.
# 30000..30011 were used to fit last round's protocol and are not evaluation data; the
# previous round's registered independent test set 500000..500399 and the live-system block
# 600000..600059 stay untouched by this round as well.
TRAIN_SEEDS = tuple(range(700000, 700048))
VALIDATION_SEEDS = tuple(range(700048, 700072))
STREAMS = HERE / "streams"


def stream_path(seed: int) -> Path:
    return STREAMS / f"s{seed}.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration_for(
    seed: int,
    *,
    directory: Path,
    label: str,
    arm: str = "C",
    stream: Path | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """The frozen arm's configuration, replaying one stream, plus this run's overrides.

    Built from the frozen block's own arm table rather than the previous round's helper,
    because that one resolves every seed against its own development block.
    """
    resolved = stream if stream is not None else stream_path(seed)
    if not resolved.exists():
        resolved = base.dev_stream(seed)
    values = dict(base.arms.ARMS[arm].configuration)
    values.update(
        task_checkpoint=CHECKPOINTS["ctde"],
        replay_stream=str(resolved),
        seed=int(seed),
    )
    values["output"] = str(Path(directory) / f"rp_{label}_s{seed}.jsonl")
    values["decision_trace"] = str(Path(directory) / f"tr_{label}_s{seed}.jsonl")
    values["comm_transcript"] = str(Path(directory) / f"ct_{label}_s{seed}.jsonl")
    values.update(overrides)
    return values


def write_json(path: Path, payload: Any) -> Path:
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
    "PREVIOUS_ROUND",
    "FROZEN_BLOCK",
    "FROZEN",
    "CHECKPOINTS",
    "CHECKPOINT_SHA256",
    "RECORD_CONFIGURATION",
    "STREAM_STEPS",
    "TRAIN_SEEDS",
    "VALIDATION_SEEDS",
    "STREAMS",
    "base",
    "stream_path",
    "sha256",
    "configuration_for",
    "write_json",
]
