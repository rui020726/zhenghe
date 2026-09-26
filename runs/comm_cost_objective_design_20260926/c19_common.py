"""Shared paths and helpers for round 19 (design-only, zero new episodes)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ROUND_DIR = Path(__file__).resolve().parent
ROUNDS = REPO_ROOT / "runs"

# The two rounds this design is allowed to read. Nothing else is re-run.
R18 = ROUNDS / "search_trace_and_policy_20260925"
R16 = ROUNDS / "comm_info_path_validation_20260923"
R15 = ROUNDS / "comm_research_closure_20260923"

ARMS_R18 = ("A", "B")
SEEDS_R18 = tuple(range(1_200_000, 1_200_060))
ARMS_R16 = ("A1", "A2", "A3", "A4")
SEEDS_R16 = tuple(range(1_100_000, 1_100_060))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(name: str, payload: Any) -> Path:
    """Write one artefact of this round, refusing to touch anything else."""
    if os.sep in name or "/" in name or ".." in name:
        raise ValueError(f"refusing to write outside the round directory: {name!r}")
    target = ROUND_DIR / name
    if target.exists():
        raise RuntimeError(f"refusing to rewrite a delivered artefact: {name}")
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    return target


def ratio(numerator: float, denominator: float, digits: int = 6) -> float | None:
    return None if not denominator else round(numerator / denominator, digits)


def mean(values: list[float]) -> float | None:
    return None if not values else round(sum(values) / len(values), 6)
