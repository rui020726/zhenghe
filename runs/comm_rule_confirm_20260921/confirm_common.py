"""Fixtures for the rule-confirmation round: one new block, five locked arms, three settings.

This round trains nothing, so the fixtures it needs are narrower than the learning rounds':
the frozen allocation side and locked byte credit come from the rounds that fixed them, the
ACK reverse channel and its rules from round 6, and the episode runner from round 6's copy of
the pilot's. Nothing frozen is restated here -- the only new declarations are this round's own
block, its three information/channel settings, and the five arms the brief names.

The one structural difference from round 6 is the stream: the 200 engineering streams live in
this round's own directory (``streams/``), so ``configuration_for`` resolves them here and
refuses to fall back to another round's development stream. A missing stream is a loud
failure, never a silent replay of someone else's data.

    python confirm_common.py    # self-check: the runner is not a drift, the arms resolve
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
PILOT_ROUND = ROOT / "runs" / "comm_budget_pilot_20260919"
STABILITY_ROUND = ROOT / "runs" / "comm_stability_20260919"
TIGHT_ROUND = ROOT / "runs" / "comm_tight_budget_20260919"
PROTOCOL_ROUND = ROOT / "runs" / "comm_protocol_v2_20260920"
ACK_ROUND = ROOT / "runs" / "comm_ack_value_20260920"

for _directory in (
    str(ROOT),
    str(PILOT_ROUND),
    str(STABILITY_ROUND),
    str(TIGHT_ROUND),
    str(PROTOCOL_ROUND),
    str(ACK_ROUND),
    str(HERE),
):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

# Imported, not restated: the frozen arm configuration, the locked byte tiers, the round-6
# runner and reverse-channel parameters, and round 2's paired bootstrap are shared with the
# archived results this round confirms.
pilot = importlib.import_module("pilot_common")
protocol_common = importlib.import_module("protocol_common")
protocol_env = importlib.import_module("protocol_env")
ack_common = importlib.import_module("ack_common")
tight_common = importlib.import_module("tight_common")

#: The new engineering block. Registered by ``block_audit.py`` (zero overlap, four sources)
#: and recorded once by ``record_streams.py``. Engineering use only.
BLOCK_START = 900_000
BLOCK_STOP = 900_200
BLOCK_SEEDS: tuple[int, ...] = tuple(range(BLOCK_START, BLOCK_STOP))

#: The frozen recording configuration, read from the block that fixed it.
RECORD_CONFIGURATION = dict(pilot.RECORD_CONFIGURATION)
STREAM_STEPS = pilot.STREAM_STEPS

#: The primary byte credit: the locked 25 % tier, read from the calibration file.
PRIMARY_TIER = protocol_common.PRIMARY_TIER
BUDGET_BYTES = protocol_common.BUDGETS_BYTES[PRIMARY_TIER]

#: The two timing protocols stay exactly as rounds 5-6 left them; every arm this round runs
#: owns its own timing.
POLICY_MODE = protocol_common.POLICY_MODE

#: The reverse channel's declared delay, inherited field by field from round 6.
ACK_DELAY_TICKS = ack_common.ACK_DELAY_TICKS

#: The three settings. Uplink and reverse-channel delivery probabilities are the same number
#: inside a setting, which is the brief's rule ("ACK 臂反向信道同参数"): the reverse channel
#: is not a friendlier second uplink. ``isolated`` closes sight refresh and one-hop gossip
#: (``gossip_interval`` above the step count); ``normal`` keeps the project's own 2-tick
#: gossip, i.e. the real information path.
SETTINGS: dict[str, dict[str, Any]] = {
    "isolated-lossless": {
        "information": "isolated",
        "delivery_probability": ack_common.LOSSLESS_DELIVERY_PROBABILITY,
        "role": "negative control: no packet loss on either channel",
    },
    "isolated-lossy": {
        "information": "isolated",
        "delivery_probability": ack_common.PRIMARY_DELIVERY_PROBABILITY,
        "role": "the round-6 primary setting, on fresh streams",
    },
    "normal-lossy": {
        "information": "normal",
        "delivery_probability": ack_common.PRIMARY_DELIVERY_PROBABILITY,
        "role": "the real information path: sight refresh and gossip stay on",
    },
}

#: The five arms, with every parameter taken from the completed 48-stream training matrix.
#: Parameters are NOT re-selected on the new block.
ARMS: dict[str, dict[str, Any]] = {
    "noack_send_once": {
        "policy": "send_once",
        "params": "",
        "ack": False,
        "role": "the cheapest no-ACK engineering baseline",
    },
    "ack_send_once": {
        "policy": "send_once",
        "params": "",
        "ack": True,
        "role": "isolates the ACK protocol overhead alone",
    },
    "ack_retry_2": {
        "policy": "retry_until_acked",
        "params": "max_attempts=2",
        "ack": True,
        "role": "the strongest ACK rule on the training streams",
    },
    "ack_periodic_20": {
        "policy": "periodic",
        "params": "interval=20",
        "ack": True,
        "role": "ACK-rule frontier reference",
    },
    "ack_backoff_16": {
        "policy": "exponential_backoff",
        "params": "base_interval=16,max_interval=32",
        "ack": True,
        "role": "backoff frontier reference / strong lossless negative-control arm",
    },
}

ARM_LABELS: tuple[str, ...] = tuple(ARMS)
SETTING_LABELS: tuple[str, ...] = tuple(SETTINGS)

STREAMS = HERE / "streams"
EVALUATION_OUT = HERE / "evaluation"
TRAIN_OUT = HERE / "train"  # unused this round; kept absent on purpose


def stream_path(seed: int) -> Path:
    return STREAMS / f"s{seed}.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_path(name: str) -> Path:
    return HERE / name


def read_json(name: str) -> Any:
    return json.loads(json_path(name).read_text(encoding="utf-8"))


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


def evaluation_root(setting: str) -> Path:
    """One directory per setting. Round 4's lesson: the writer names files after arm and seed
    only, so two settings sharing a directory overwrite each other."""
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}; this round runs {SETTING_LABELS}")
    return EVALUATION_OUT / setting


def configuration_for(
    seed: int,
    *,
    directory: Path,
    label: str,
    stream: Path | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """The frozen C arm's configuration replaying one of this round's own streams.

    Unlike the pilot's resolver there is no fallback to another round's development stream:
    a seed whose stream was never recorded fails here rather than silently borrowing data.
    """
    resolved = stream if stream is not None else stream_path(seed)
    if not resolved.exists():
        raise FileNotFoundError(
            f"no recorded stream for seed {seed}: {resolved}; record it first with "
            "record_streams.py -- the block may not be substituted after the audit"
        )
    values = dict(pilot.base.arms.ARMS["C"].configuration)
    values.update(
        task_checkpoint=pilot.CHECKPOINTS["ctde"],
        replay_stream=str(resolved),
        seed=int(seed),
    )
    values["output"] = str(Path(directory) / f"rp_{label}_s{seed}.jsonl")
    values["decision_trace"] = str(Path(directory) / f"tr_{label}_s{seed}.jsonl")
    values["comm_transcript"] = str(Path(directory) / f"ct_{label}_s{seed}.jsonl")
    values.update(overrides)
    return values


def paired_bootstrap(rows: list[dict[str, Any]], *, baseline: str, key: str) -> dict[str, Any]:
    """Round 2's seed-level paired bootstrap, reused rather than re-derived."""
    return tight_common.paired_bootstrap(rows, baseline=baseline, key=key)


def shard_of(items: list[Any], shards: int, shard: int) -> list[Any]:
    """The one and only place sharding is applied."""
    return [item for index, item in enumerate(items) if index % max(1, shards) == shard % max(1, shards)]


__all__ = [
    "HERE",
    "ROOT",
    "PILOT_ROUND",
    "STABILITY_ROUND",
    "TIGHT_ROUND",
    "PROTOCOL_ROUND",
    "ACK_ROUND",
    "pilot",
    "protocol_common",
    "protocol_env",
    "ack_common",
    "tight_common",
    "BLOCK_START",
    "BLOCK_STOP",
    "BLOCK_SEEDS",
    "RECORD_CONFIGURATION",
    "STREAM_STEPS",
    "PRIMARY_TIER",
    "BUDGET_BYTES",
    "POLICY_MODE",
    "ACK_DELAY_TICKS",
    "SETTINGS",
    "ARMS",
    "ARM_LABELS",
    "SETTING_LABELS",
    "STREAMS",
    "EVALUATION_OUT",
    "stream_path",
    "sha256",
    "json_path",
    "read_json",
    "write_json",
    "evaluation_root",
    "configuration_for",
    "paired_bootstrap",
    "shard_of",
]
