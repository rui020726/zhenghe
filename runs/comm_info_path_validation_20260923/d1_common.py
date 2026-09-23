"""Shared machinery for the D1 information-path validation.

Three things are enforced rather than promised:

* every episode launch is booked in a per-process ledger and the cap is judged on the sum,
  because three earlier rounds found that a shared read-modify-write ledger breaks under
  Windows file locks, and "counting other processes' books to decide the cap" turns the
  race back on;
* technical failures are booked too, so the cap cannot be quietly re-opened by retrying;
* comparisons between arms normalise identity fields (paths) through the same helper the
  B0 unit tests use — one implementation of "what counts as behaviour", so the gate and
  the tests cannot drift apart.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
try:
    import torch

    torch.set_num_threads(1)
except Exception:  # torch is optional in dependency-free environments
    pass

ROUND = "comm_info_path_validation_20260923"
ROUND_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROUND_DIR.parent.parent
START_SHA = "b73876013e76a82bbaab2f886eedeb34df7fc039"

ROWS = ROUND_DIR / "rows"
LOGS = ROUND_DIR / "logs"

PIL_CHECKPOINT = REPO_ROOT / "runs" / "cur_v2rep05_s2" / "best" / "shared_mappo_model_21_3100.pth"
CTDE_CHECKPOINT = REPO_ROOT / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth"

SEEDS = list(range(1100000, 1100060))

#: The frozen launch cap from D1 §六, inclusive of technical failures.
LAUNCH_CAP = 296
DATA_ARMS = ("A1", "A2", "A3", "A4")

#: Per-arm overrides on top of the P1 candidate configuration. Each entry is exactly the
#: single difference its contrast is allowed to have; anything else is a bug in this table.
ARM_OVERRIDES: dict[str, dict[str, Any]] = {
    "A1": {},
    "A2": {"comm_policy": "always_hold"},
    "A3": {"comm_delivery_probability": 0.7},
    "A4": {"comm_delivery_probability": 0.7, "gossip_interval": 100000},
}

#: What each contrast is allowed to differ in, as field names of the configuration record.
CONTRAST_FIELDS: dict[str, set[str]] = {
    "C1": {"comm_policy"},
    "C2": {"comm_delivery_probability"},
    "C3": {"gossip_interval"},
}

CONTRASTS = (("C1", "A1", "A2"), ("C2", "A1", "A3"), ("C3", "A3", "A4"))

BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_RNG = 20260923


def base_configuration_fields() -> dict[str, Any]:
    """The P1 candidate configuration, read off that run's own run_config record."""
    return {
        "mock": False,
        "steps": 1000,
        "device": "cpu",
        "quiet": True,
        "piloc_checkpoint": str(PIL_CHECKPOINT),
        "task_checkpoint": str(CTDE_CHECKPOINT),
        "coordinator_policy": "coalition_planner",
        "sticky_commitments": True,
        "use_coordinator": True,
        "release_redundant_members": True,
        "suppress_recommand_own_task": True,
        "residual_scale": 0.25,
        "allocation_interval": 5,
        "coordination_interval": 5,
        "command_ttl": 10.0,
        "dt": 1.0,
        "detection_radius": 5,
        "search_agents": 2,
        "num_goals": 6,
        "grid_size": 60,
        "obstacle_density": 0.15,
        "rescue_species": 3,
        "rescue_per_species": 2,
        "actor_kv_mode": "tokens",
        "comm_delivery_probability": 1.0,
        "comm_delay_ticks": 0,
        "comm_subscribers": "all",
        "comm_policy": "immediate",
        "comm_timing_mode": "legacy_shared_bound",
        "gossip_interval": 2,
        "comm_range": 0.3,
    }


# ------------------------------------------------------------------------------ files --------
def cell_paths(tag: str, arm: str, seed: int) -> dict[str, Path]:
    """Every artefact of one cell. The tag carries each parallel dimension in the name,
    because an earlier round lost data when two dimensions wrote the same file."""
    stem = f"{tag}_{arm}_s{seed}"
    return {
        "lifecycle": LOGS / f"{stem}_rp.jsonl",
        "transcript": LOGS / f"{stem}_ct.jsonl",
        "probe": LOGS / f"{stem}_ip.jsonl",
        "row": ROWS / f"{stem}.json",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(name: str, payload: Any) -> Path:
    """Write one artefact of this round, refusing to touch anything else."""
    if os.sep in name or "/" in name or ".." in name:
        raise ValueError(f"refusing to write outside the round directory: {name!r}")
    target = ROUND_DIR / name
    if target.exists() and target.name != name:
        raise RuntimeError(f"path escapes the round directory: {target}")
    rendered = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"refusing to rewrite a delivered artefact: {name}")
        return target
    target.write_text(rendered, encoding="utf-8")
    return target


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quantiles(values: list[float]) -> dict[str, Any]:
    """min/p25/median/p75/p90/max and mean, or None-filled for an empty list."""
    if not values:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None,
                "p90": None, "max": None, "mean": None}
    ordered = sorted(float(value) for value in values)

    def at(fraction: float) -> float:
        position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[position]

    return {"n": len(ordered), "min": ordered[0], "p25": at(0.25), "median": at(0.50),
            "p75": at(0.75), "p90": at(0.90), "max": ordered[-1],
            "mean": round(sum(ordered) / len(ordered), 5)}


def ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def iter_jsonl(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{path.name}:{number}: unparsable record") from error


# ----------------------------------------------------------------------------- ledger --------
def _ledger_path(tag: str) -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    return LOGS / f"ledger_{tag}_p{os.getpid()}.json"


def ledger_reserve(kind: str, tag: str, detail: str = "") -> int:
    """Book one launch against the round cap. Raises once the cap is spent."""
    path = _ledger_path(kind)
    book = read_json(path) if path.exists() else {"kind": kind, "starts": 0, "entries": []}
    used = ledger_total()
    if used + 1 > LAUNCH_CAP:
        raise RuntimeError(f"launch cap {LAUNCH_CAP} reached at {used}; refusing a {kind} start")
    book["starts"] += 1
    book["entries"].append({"n": book["starts"], "round_total_before_this": used,
                            "detail": detail})
    path.write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
    return used + 1


def ledger_total() -> int:
    total = 0
    for path in sorted(LOGS.glob("ledger_*.json")):
        try:
            total += int(read_json(path).get("starts") or 0)
        except Exception:
            continue
    return total


def ledger_report() -> dict[str, Any]:
    per_kind: dict[str, int] = {}
    for path in sorted(LOGS.glob("ledger_*.json")):
        try:
            book = read_json(path)
        except Exception:
            continue
        per_kind[book.get("kind", "?")] = per_kind.get(book.get("kind", "?"), 0) + int(
            book.get("starts") or 0)
    return {"cap": LAUNCH_CAP, "used": ledger_total(), "by_kind": per_kind}


# ----------------------------------------------------------------------- behaviour view ------
def behaviour_fields(configuration: dict[str, Any]) -> dict[str, Any]:
    """The configuration minus artefact paths — the same split the B0 tests use."""
    from integrated_sar.info_probe import IDENTITY_FIELDS

    return {key: value for key, value in configuration.items()
            if key not in IDENTITY_FIELDS}


def detection_sequence(lifecycle: Path) -> list[tuple[str, float, str]]:
    """Ordered (target, step, discoverer) triples — identity gate I1."""
    out = []
    for record in iter_jsonl(lifecycle):
        if record.get("event_type") == "detection":
            out.append((str(record["target_id"]), float(record["timestamp"]),
                        str(record.get("discovered_by"))))
    return sorted(out)


def emission_sequence(transcript: Path) -> list[tuple[float, str, int, int, int]]:
    """Ordered (step, target, subscribers, cost, per-target ordinal) — identity gate I2."""
    out = []
    per_target: dict[str, int] = {}
    for record in iter_jsonl(transcript):
        if record.get("kind") != "send":
            continue
        target = str(record["target"])
        per_target[target] = per_target.get(target, 0) + 1
        out.append((float(record["step"]), target, int(record.get("subscribers") or 0),
                    int(record.get("cost_bytes") or 0), per_target[target]))
    return sorted(out)


def packet_outcomes(probe_file: Path) -> list[tuple[str, int, str, bool]]:
    """(target, emission ordinal, recipient, arrived) — identity gate I3, per packet."""
    out = []
    for record in iter_jsonl(probe_file):
        if record.get("kind") == "delivery":
            out.append((str(record["target_id"]), int(record.get("emission") or 0),
                        str(record["robot_id"]), bool(record["arrived"])))
    return sorted(out)


def configuration_of(lifecycle: Path) -> dict[str, Any]:
    for record in iter_jsonl(lifecycle):
        if record.get("event_type") == "run_config":
            return dict((record.get("detail") or {}).get("configuration") or {})
    raise AssertionError(f"{lifecycle.name} has no run_config record")


def enter_round() -> None:
    here = str(ROUND_DIR)
    while here in sys.path:
        sys.path.remove(here)
    sys.path.insert(0, here)
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(1, root)


enter_round()
