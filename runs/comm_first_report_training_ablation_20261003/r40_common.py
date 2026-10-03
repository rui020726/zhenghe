"""ROUND40 paths, immutable launch plan, and process-safe preconstruction claims."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROUND_DIR = Path(__file__).resolve().parent
NORMAL_CAP = 672
RESERVE_CAP = 28
HARD_CAP = NORMAL_CAP + RESERVE_CAP
TRAIN_SEEDS = (11, 22, 33)
TRAIN_EPISODES = 128
EPISODES_PER_UPDATE = 4
TRAIN_SCENARIOS = tuple(range(1_300_000, 1_300_048))
DEV_SCENARIOS = tuple(range(1_301_000, 1_301_016))
RULE_ARMS = ("late_discovery_gate", "always_hold")


def output_dir() -> Path:
    return ROUND_DIR


def bootstrap(input_root: str | Path):
    """Configure archived Stage C imports to read canonical inputs and write only to ROUND40."""
    root = Path(input_root).expanduser().resolve()
    if not (root / ".git").exists() or not (root / "integrated_sar").is_dir():
        raise RuntimeError(f"input root is not the existing project checkout: {root}")
    module_dirs = [
        root / "runs" / "comm_cost_v2_rerun_20260928",
        root / "runs" / "comm_cost_stage_b_20260927",
        root / "runs" / "comm_budget_action_mask_fix_20260922",
        root / "runs" / "comm_budget_pilot_20260919",
        root / "runs" / "comm_cost_controller_stage_a_20260927",
        root / "runs" / "comm_tight_budget_20260919",
        root / "runs" / "search_trace_and_policy_20260925",
    ]
    for directory in reversed(module_dirs):
        if not directory.is_dir():
            raise RuntimeError(f"required archived module directory is missing: {directory}")
        sys.path.insert(0, str(directory))
    sys.path.insert(0, str(root))

    import v2_common as common

    common.ROOT = root
    common.OUTPUT_DIR = ROUND_DIR
    common.STAGE_C_DIR = root / "runs" / "comm_cost_stage_c_20260927"
    common.STAGE_B_DIR = root / "runs" / "comm_cost_stage_b_20260927"
    common.STAGE_A_DIR = root / "runs" / "comm_cost_controller_stage_a_20260927"
    common.PILOT_ROUND = root / "runs" / "comm_budget_pilot_20260919"
    common.MASK_FIX_ROUND = root / "runs" / "comm_budget_action_mask_fix_20260922"
    common.TIGHT_ROUND = root / "runs" / "comm_tight_budget_20260919"
    common.ROUND18_DIR = root / "runs" / "search_trace_and_policy_20260925"
    common.LOGS = ROUND_DIR / "logs"
    common.ROWS = ROUND_DIR / "rows"
    common.REJECTED = ROUND_DIR / "rejected"
    common.CHECKPOINTS = ROUND_DIR / "checkpoints"
    common.CHECKPOINT_FILES = (
        root / "runs" / "cur_v2rep05_s2" / "best" / "shared_mappo_model_21_3100.pth",
        root / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth",
    )
    common.NORMAL_LAUNCHES = NORMAL_CAP
    common.RESERVE_LAUNCHES = RESERVE_CAP
    common.LAUNCH_CAP = HARD_CAP
    for directory in (common.LOGS, common.ROWS, common.REJECTED, common.CHECKPOINTS):
        directory.mkdir(parents=True, exist_ok=True)
    return common


def plan() -> dict[str, Any]:
    return {
        "normal": {
            "immediate_reference": len(TRAIN_SCENARIOS) + len(DEV_SCENARIOS),
            "new_training": len(TRAIN_SEEDS) * TRAIN_EPISODES,
            "old_new_development": 2 * len(TRAIN_SEEDS) * 2 * len(DEV_SCENARIOS),
            "rule_development": len(RULE_ARMS) * len(DEV_SCENARIOS),
        },
        "normal_total": NORMAL_CAP,
        "technical_reserve_max": RESERVE_CAP,
        "hard_cap": HARD_CAP,
    }


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    if not slug:
        raise ValueError(f"empty launch claim key from {value!r}")
    return slug


def _write_new(path: Path, payload: dict[str, Any]) -> bool:
    """Create a JSON file exclusively; False means another process already owns this path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _reserve_slot(category: str, *, logical_job: str, attempt: int) -> int:
    if category not in ("normal", "reserve"):
        raise ValueError(f"unknown launch category {category!r}")
    cap = NORMAL_CAP if category == "normal" else RESERVE_CAP
    slot_dir = ROUND_DIR / "launch_slots" / category
    slot_dir.mkdir(parents=True, exist_ok=True)
    for number in range(1, cap + 1):
        slot_path = slot_dir / f"{number:04d}.json"
        payload = {
            "category": category,
            "slot": number,
            "logical_job": logical_job,
            "attempt": int(attempt),
            "claimed_at_unix": time.time(),
        }
        if _write_new(slot_path, payload):
            return number
    raise RuntimeError(f"ROUND40 {category} launch cap {cap} exhausted before {logical_job}")


def claim_launch(logical_job: str, *, category: str, attempt: int,
                 metadata: dict[str, Any] | None = None) -> Path:
    """Atomically claim a unique job attempt and quota slot before environment construction."""
    slug = _slug(logical_job)
    claim_dir = ROUND_DIR / "launch_claims" / slug
    claim_dir.mkdir(parents=True, exist_ok=True)
    claim_path = claim_dir / f"attempt_{int(attempt):04d}.json"
    # Claim identity first so parallel shards cannot double-launch the same attempt.
    placeholder = {
        "logical_job": logical_job,
        "attempt": int(attempt),
        "category": category,
        "status": "claiming",
        "claimed_at_unix": time.time(),
    }
    if not _write_new(claim_path, placeholder):
        raise RuntimeError(f"ROUND40 launch attempt already claimed: {logical_job} #{attempt}")
    try:
        slot = _reserve_slot(category, logical_job=logical_job, attempt=attempt)
    except Exception:
        _update_claim(claim_path, status="not_started_quota_exhausted")
        raise
    _update_claim(claim_path, status="claimed_before_environment", launch_slot=slot,
                  metadata=metadata or {})
    return claim_path


def _update_claim(path: Path, **updates: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def finish_launch(claim_path: Path, *, ok: bool, detail: str) -> None:
    _update_claim(claim_path, status="complete" if ok else "failed_technical",
                  finished_at_unix=time.time(), detail=str(detail)[:1200])


def next_attempt(logical_job: str) -> int:
    claim_dir = ROUND_DIR / "launch_claims" / _slug(logical_job)
    if not claim_dir.exists():
        return 1
    attempts = []
    for path in claim_dir.glob("attempt_*.json"):
        try:
            attempts.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return max(attempts, default=0) + 1


def ledger_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {"normal": [], "reserve": [], "claims": []}
    for category in ("normal", "reserve"):
        directory = ROUND_DIR / "launch_slots" / category
        if directory.exists():
            result[category] = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(directory.glob("*.json"))
            ]
    claims = ROUND_DIR / "launch_claims"
    if claims.exists():
        result["claims"] = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(claims.glob("*/*.json"))
        ]
    result["normal_used"] = len(result["normal"])
    result["reserve_used"] = len(result["reserve"])
    result["total_used"] = result["normal_used"] + result["reserve_used"]
    result["plan"] = plan()
    if result["normal_used"] > NORMAL_CAP or result["reserve_used"] > RESERVE_CAP:
        raise RuntimeError("ROUND40 ledger exceeds its immutable launch cap")
    return result


__all__ = [
    "DEV_SCENARIOS", "EPISODES_PER_UPDATE", "HARD_CAP", "NORMAL_CAP", "RESERVE_CAP",
    "ROUND_DIR", "RULE_ARMS", "TRAIN_EPISODES", "TRAIN_SCENARIOS", "TRAIN_SEEDS",
    "bootstrap", "claim_launch", "finish_launch", "ledger_snapshot", "next_attempt",
    "output_dir", "plan",
]
