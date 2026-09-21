"""Fixtures for the evidence-review round: same 3,000 rows, one statistics object.

This round runs no episode. Its inputs are the completed round-7 shards (read-only), its
statistics are round 2's frozen paired bootstrap, and its only new declarations are the
paths that tie the old artefacts to the revised documents. Nothing here re-selects arms,
re-pools settings or rewrites the 3,000-row summary: the aggregate means must come out
identical to the archived ones, and the tests assert exactly that.

The one structural fix this round exists for: the archived report computed the same
paired difference twice -- once over the whole arm set for the comparison tables and once
over a two-arm subset inside the judgement -- and ``evaluate_pilot.paired`` reseeds its RNG
per call but consumes one block of draws per arm, so the two passes drew different bootstrap
samples for the same statistic. Every number in this round is computed once, from one call
per (setting, axis) over the full arm set, and the judgement reads those same blocks.

    python review_common.py    # self-check: the shards load, the runner still resolves
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIRM_ROUND = ROOT / "runs" / "comm_rule_confirm_20260921"
PROTOCOL_ROUND = ROOT / "runs" / "comm_protocol_v2_20260920"
ACK_ROUND = ROOT / "runs" / "comm_ack_value_20260920"

for _directory in (
    str(ROOT),
    str(CONFIRM_ROUND),
    str(PROTOCOL_ROUND),
    str(ACK_ROUND),
    str(HERE),
):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

# Imported, not restated: the frozen arms/settings/bootstrap of the round being revised.
# ``confirm_matrix`` holds the statistics constants and the deterministic helpers; importing
# it is read-only and runs nothing.
confirm_common = importlib.import_module("confirm_common")
confirm_matrix = importlib.import_module("confirm_matrix")

#: The frozen statistics, restated only as pointers.
BOOTSTRAP_REPLICATIONS = confirm_matrix.BOOTSTRAP_REPLICATIONS
BOOTSTRAP_SEED = confirm_matrix.BOOTSTRAP_SEED
SETTING_LABELS: tuple[str, ...] = confirm_common.SETTING_LABELS
ARM_LABELS: tuple[str, ...] = confirm_common.ARM_LABELS
SETTINGS = confirm_common.SETTINGS
ARMS = confirm_common.ARMS
BUDGET_BYTES = confirm_common.BUDGET_BYTES
BLOCK = (confirm_common.BLOCK_START, confirm_common.BLOCK_STOP - 1)
SEALED = "800000..800399"

#: The four pre-registered comparisons, as the archived protocol froze them.
COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("ack_retry_2", "noack_send_once", "协议整体的成本—效果差"),
    ("ack_retry_2", "ack_send_once", "ACK 信息之上的重发规则收益"),
    ("ack_periodic_20", "ack_retry_2", "规则前沿：周期规则对锁定规则"),
    ("ack_backoff_16", "ack_retry_2", "规则前沿：退避规则对锁定规则"),
)

#: Axes, and which of them are cost axes (where "higher" is never "better").
COST_AXES = ("total_bytes", "uplink_bytes", "ack_bytes")
EFFECT_AXES = ("on_time",)

#: The comparison objects the ACK-switch table must list, including round 6's reference.
COMPARISON_OBJECTS: tuple[dict[str, Any], ...] = (
    {
        "arm": "send_once（第六轮矩阵臂）",
        "round": "S3-5（48 条训练流）",
        "ack_switch": "开",
        "consumes_ack_feedback": "否（每条视图只发一次，无从消费）",
        "retry_capability": "无",
        "budget": "29 300 B/episode",
    },
    {
        "arm": "noack_send_once（本轮基线）",
        "round": "S3-6（200 条新流）",
        "ack_switch": "关",
        "consumes_ack_feedback": "否（无反馈信道）",
        "retry_capability": "无",
        "budget": "29 300 B/episode",
    },
    {
        "arm": "ack_send_once（本轮）",
        "round": "S3-6（200 条新流）",
        "ack_switch": "开",
        "consumes_ack_feedback": "否（与第六轮 send_once 同行为，只是显式计费）",
        "retry_capability": "无",
        "budget": "29 300 B/episode",
    },
    {
        "arm": "retry_until_acked_2 / ack_retry_2",
        "round": "S3-5 与 S3-6",
        "ack_switch": "开",
        "consumes_ack_feedback": "是（按未确认订阅者重发，max_attempts=2）",
        "retry_capability": "有（每视图最多 2 次尝试）",
        "budget": "29 300 B/episode",
    },
)


def shard_paths() -> list[Path]:
    """The archived 3,000-row input, in shard order. Read-only: never rewritten here."""
    return sorted(CONFIRM_ROUND.glob("matrix_shard*.json"))


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in shard_paths():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("rejected"):
            raise SystemExit(f"{path.name} carries rejected rows; the input is not the clean one")
        rows.extend(payload["rows"])
    return rows


def load_archived(name: str) -> dict[str, Any]:
    return json.loads((CONFIRM_ROUND / name).read_text(encoding="utf-8"))


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


def paired(rows: list[dict[str, Any]], *, baseline: str, key: str) -> dict[str, Any]:
    """Round 2's frozen bootstrap, called once per (setting, axis) over the full arm set.

    ``evaluate_pilot.paired`` reseeds its RNG per call but consumes one draw block per
    non-baseline arm in row order, so the number of arms in ``rows`` decides where each
    arm's draws sit in the stream. Passing the whole setting every time is what makes the
    comparison tables and the judgement read the same sample.
    """
    return confirm_common.paired_bootstrap(rows, baseline=baseline, key=key)


def channel_label(setting: str) -> str:
    """The setting's channel, named the way the source names it: a success probability."""
    probability = float(SETTINGS[setting]["delivery_probability"])
    return (
        f"上行与反向投递成功率均 {probability:.1f}（丢包率 {1.0 - probability:.1f}）"
    )


__all__ = [
    "HERE",
    "ROOT",
    "CONFIRM_ROUND",
    "ARM_LABELS",
    "SETTING_LABELS",
    "SETTINGS",
    "ARMS",
    "BUDGET_BYTES",
    "BLOCK",
    "SEALED",
    "COMPARISONS",
    "COMPARISON_OBJECTS",
    "COST_AXES",
    "EFFECT_AXES",
    "BOOTSTRAP_REPLICATIONS",
    "BOOTSTRAP_SEED",
    "confirm_common",
    "confirm_matrix",
    "shard_paths",
    "load_rows",
    "load_archived",
    "json_path",
    "read_json",
    "write_json",
    "paired",
    "channel_label",
]
