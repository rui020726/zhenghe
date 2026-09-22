"""Fixtures for multistep diagnostic stage one.

Pure helpers (eligibility, JSON IO, frozen constants) import nothing heavy. Episode
execution pulls the archived runners only when asked, so mechanism tests do not need torch.
Production communication code is not modified; interventions wrap the relay from the outside.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ACK_ROUND = ROOT / "runs" / "comm_ack_value_20260920"
PROTOCOL_ROUND = ROOT / "runs" / "comm_protocol_v2_20260920"
PILOT_ROUND = ROOT / "runs" / "comm_budget_pilot_20260919"

for _directory in (
    str(ROOT),
    str(PILOT_ROUND),
    str(ROOT / "runs" / "comm_stability_20260919"),
    str(ROOT / "runs" / "comm_tight_budget_20260919"),
    str(PROTOCOL_ROUND),
    str(ACK_ROUND),
    str(HERE),
):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

# Frozen constants (also mirrored in FROZEN_CONFIG.json). Values come from the archived
# ACK counterfactual's baseline_rule and ack_common.arm_specs entry for that arm.
BASELINE_RULE = "event_retry_until_acked_5"
POLICY_NAME = "event_retry_until_acked"
POLICY_PARAMS = {"max_attempts": 5, "progress_threshold": 0.2, "urgency_time": 25.0}
POLICY_MODE = "policy_owned_timing"
PRIMARY_TIER = "25pct"
BUDGET_BYTES = 29300
DELIVERY_PROBABILITY = 0.7
ACK_DELAY_TICKS = 0
ACK_DELIVERY_PROBABILITY = 0.7
ISOLATED = "isolated"

TRAIN_SEEDS = tuple(range(700000, 700048))
STREAM_SAMPLE_SEED = 20260921
STREAM_SAMPLE_DRAWN = 16
STREAM_SAMPLE_USED = 8
POINT_SELECTION_SEED = 2026092101
HORIZONS = (1, 20)
MAX_SIMULATION_STARTS = 40

POINTS_INITIAL = HERE / "POINTS_INITIAL.json"
POINTS_JSON = HERE / "POINTS.json"
RESULTS_JSON = HERE / "COMM_MULTISTEP_RESULTS.json"
RESULTS_MD = HERE / "COMM_MULTISTEP_RESULTS.md"
MANIFEST_JSON = HERE / "COMM_MULTISTEP_MANIFEST.json"
FROZEN_CONFIG = HERE / "FROZEN_CONFIG.json"
SOURCE_CANDIDATES_DIGEST = HERE / "SOURCE_CANDIDATES_DIGEST.json"
EPISODES = HERE / "episodes"
POINTS_DIR = HERE / "points"

COUNTERFACTUAL_JSON = ACK_ROUND / "COMM_ACK_VALUE_COUNTERFACTUAL_25pct.json"

_heavy: dict[str, Any] = {}


def _load_heavy() -> dict[str, Any]:
    if _heavy:
        return _heavy
    protocol_common = importlib.import_module("protocol_common")
    protocol_env = importlib.import_module("protocol_env")
    pilot = importlib.import_module("pilot_common")
    # Cross-check frozen constants against the archived fixtures once.
    if int(protocol_common.tier_bytes(PRIMARY_TIER)) != BUDGET_BYTES:
        raise RuntimeError("BUDGET_BYTES drifted from protocol_common.tier_bytes(25pct)")
    if tuple(pilot.TRAIN_SEEDS) != TRAIN_SEEDS:
        raise RuntimeError("TRAIN_SEEDS drifted from pilot_common.TRAIN_SEEDS")
    _heavy.update(
        {
            "protocol_common": protocol_common,
            "protocol_env": protocol_env,
            "pilot": pilot,
        }
    )
    return _heavy


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline_spec() -> dict[str, Any]:
    return {
        "policy": POLICY_NAME,
        "params": dict(POLICY_PARAMS),
        "mode": POLICY_MODE,
        "budget_bytes": BUDGET_BYTES,
        "delivery_probability": DELIVERY_PROBABILITY,
        "ack": True,
        "ack_delay_ticks": ACK_DELAY_TICKS,
        "ack_delivery_probability": ACK_DELIVERY_PROBABILITY,
        "information": ISOLATED,
    }


def policy_params_cli(params: dict[str, Any] | None = None) -> dict[str, Any]:
    return dict(params or POLICY_PARAMS)


def run_baseline_episode(
    seed: int,
    *,
    directory: Path,
    label: str,
    controller: Any | None = None,
    policy_hook: Any | None = None,
    read_timelines: bool = True,
) -> dict[str, Any]:
    """One episode under the frozen baseline rule and channel, optional wrapper."""
    heavy = _load_heavy()
    protocol_env = heavy["protocol_env"]
    return protocol_env.run_episode(
        seed,
        directory=directory,
        label=label,
        policy=POLICY_NAME,
        policy_params=policy_params_cli(),
        budget_bytes=BUDGET_BYTES,
        information=ISOLATED,
        delivery_probability=DELIVERY_PROBABILITY,
        timing_mode=POLICY_MODE,
        controller=controller,
        max_wait_steps=None,
        urgent_wait_steps=None,
        ack=True,
        ack_delay_ticks=ACK_DELAY_TICKS,
        ack_delivery_probability=ACK_DELIVERY_PROBABILITY,
        policy_hook=policy_hook,
        read_timelines=read_timelines,
    )


def eligible(point: dict[str, Any]) -> bool:
    """Pre-intervention eligibility: partial ACK confirmation and a changed view."""
    try:
        confirmed = float(point["confirmed_fraction"])
    except (KeyError, TypeError, ValueError):
        return False
    return 0.0 < confirmed < 1.0 and bool(point.get("changed_view"))


def load_probe_points() -> dict[int, list[dict[str, Any]]]:
    payload = read_json(COUNTERFACTUAL_JSON)
    if payload.get("baseline_rule") != BASELINE_RULE:
        raise RuntimeError(
            f"archived counterfactual baseline is {payload.get('baseline_rule')!r}, "
            f"expected {BASELINE_RULE!r}"
        )
    out: dict[int, list[dict[str, Any]]] = {}
    for entry in payload.get("probe_points", []):
        out[int(entry["seed"])] = list(entry.get("points") or [])
    if len(out) != 48:
        raise RuntimeError(f"expected 48 probe streams, found {len(out)}")
    return out


def strip_outcome_fields(point: dict[str, Any]) -> dict[str, Any]:
    """Keep only the pre-state fields selection is allowed to read."""
    keep = (
        "step",
        "target",
        "confirmed_fraction",
        "unacked",
        "age_steps",
        "changed_view",
        "rule_sent",
    )
    return {key: point[key] for key in keep if key in point}


__all__ = [
    "HERE",
    "ROOT",
    "ACK_ROUND",
    "TRAIN_SEEDS",
    "BASELINE_RULE",
    "POLICY_NAME",
    "POLICY_PARAMS",
    "POLICY_MODE",
    "PRIMARY_TIER",
    "BUDGET_BYTES",
    "DELIVERY_PROBABILITY",
    "ACK_DELAY_TICKS",
    "ACK_DELIVERY_PROBABILITY",
    "ISOLATED",
    "STREAM_SAMPLE_SEED",
    "STREAM_SAMPLE_DRAWN",
    "STREAM_SAMPLE_USED",
    "POINT_SELECTION_SEED",
    "HORIZONS",
    "MAX_SIMULATION_STARTS",
    "POINTS_INITIAL",
    "POINTS_JSON",
    "RESULTS_JSON",
    "RESULTS_MD",
    "MANIFEST_JSON",
    "FROZEN_CONFIG",
    "SOURCE_CANDIDATES_DIGEST",
    "EPISODES",
    "POINTS_DIR",
    "COUNTERFACTUAL_JSON",
    "write_json",
    "read_json",
    "sha256",
    "baseline_spec",
    "policy_params_cli",
    "run_baseline_episode",
    "eligible",
    "load_probe_points",
    "strip_outcome_fields",
]
