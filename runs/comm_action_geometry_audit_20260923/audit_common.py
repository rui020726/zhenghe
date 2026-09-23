"""Shared read-only helpers for the communication action-geometry audit.

This round is a retrospective mechanism audit over episodes that already exist on disk. Nothing
here may run an environment: every function opens files, hashes them, or aggregates stored rows.

The tripwire in :func:`install_env_tripwire` is what turns "we promise we did not replay" into a
checkable claim -- it is installed before the frozen round's modules are imported, so any call that
would execute an episode raises instead of running, and the call log is written next to the results.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_ROUND = ROOT / "runs" / "comm_task_pressure_learnability_20260922"

LAYERS = ("g6", "g8", "g10", "g12")
GOALS_OF = {"g6": 6, "g8": 8, "g10": 10, "g12": 12}
ARMS = (
    "send_once",
    "event",
    "event_priority",
    "periodic_5",
    "periodic_20",
    "first_send_plus_bound_3_10",
    "bound_only_3_10",
    "always_hold",
    "event_unbudgeted",
)
BUDGET_BYTES = 29_300

#: The seven competitors in the source protocol's row order (learnability protocol section 3).
#: `best_fixed` is chosen among these, and the row order is the protocol's own tie-break, so the
#: order is part of the definition rather than a listing convenience.
COMPETITION_ORDER = (
    "send_once",
    "event",
    "event_priority",
    "periodic_5",
    "periodic_20",
    "first_send_plus_bound_3_10",
    "bound_only_3_10",
)

#: Anything that could put an episode on the clock. Every name here is a real entry point in the
#: frozen round (``pressure_env.run_episode`` / ``run_arm`` / ``agreement_with_archived_rows``,
#: ``record_layer.record``, ``pressure_matrix.run``); the tripwire refuses to install if one is
#: missing, so a rename cannot silently turn the guarantee into a no-op.
FORBIDDEN_CALLS = (
    ("pressure_env", "run_episode"),
    ("pressure_env", "run_arm"),
    ("pressure_env", "agreement_with_archived_rows"),
    ("record_layer", "record"),
    ("pressure_matrix", "run"),
)

_TRIPPED: list[str] = []
_installed = False


class EnvironmentCallForbidden(RuntimeError):
    """Raised the moment an analysis path tries to execute an episode."""


def _resolve(module_name: str, attr: str) -> Any:
    module = importlib.import_module(module_name)
    if not callable(getattr(module, attr, None)):
        raise RuntimeError(f"tripwire target {module_name}.{attr} is not a callable")
    return module


def install_env_tripwire() -> list[str]:
    """Wrap the episode entry points so calling them raises. Returns the live call log.

    Safe to call more than once; the log object is shared so a caller that installed early and a
    caller that checked late see the same list.
    """
    global _installed
    if _installed:
        return _TRIPPED
    #: The round's modules must be importable before they can be wrapped -- wrapping a module that
    #: failed to import would leave the episode runners untouched and report a clean tripwire.
    enter_source_round()
    for module_name, attr in FORBIDDEN_CALLS:
        module = _resolve(module_name, attr)
        original = getattr(module, attr)
        if getattr(original, "_audit_tripwire", False):
            continue

        def _blocked(*args: Any, _name: str = f"{module_name}.{attr}", **kwargs: Any) -> Any:
            _TRIPPED.append(_name)
            raise EnvironmentCallForbidden(
                f"{_name}() would run an episode; this round is authorised for 0 new episodes"
            )

        _blocked._audit_tripwire = True  # type: ignore[attr-defined]
        _blocked._audit_original = original  # type: ignore[attr-defined]
        setattr(module, attr, _blocked)
    _installed = True
    return _TRIPPED


def enter_source_round() -> None:
    """Put the frozen round's own directory first on sys.path.

    The round's protocols note a real hazard: the per-round fixture directories re-insert
    themselves ahead of everything, so a bare ``import pressure_metrics`` can silently resolve to
    another round's module. Inserting at position 0 after those imports keeps this round's files.
    """
    for path in (SOURCE_ROUND, ROOT):
        text = str(path)
        while text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> Path:
    """Write into *this* audit directory only.

    The frozen round guards its own artefacts with an equivalent rule; the audit inherits it so a
    mistaken path cannot quietly rewrite a published four-layer result.
    """
    resolved = Path(path).resolve()
    if HERE not in resolved.parents and resolved.parent != HERE:
        raise ValueError(f"refusing to write outside the audit directory: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return resolved


def rows_of(layer: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((SOURCE_ROUND / "rows" / layer).glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def transcript_path(layer: str, arm: str, seed: int) -> Path:
    return SOURCE_ROUND / "runs" / layer / f"ct_{arm}_{layer}_s{seed}.jsonl"


def read_transcript(layer: str, arm: str, seed: int) -> list[dict[str, Any]]:
    path = transcript_path(layer, arm, seed)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def quantiles(values: list[float]) -> dict[str, Any]:
    """Linear-interpolation quantiles plus the count, so a small n is visible in the number."""
    if not values:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "p95": None,
                "max": None, "mean": None}
    ordered = sorted(values)

    def _at(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    return {
        "n": len(ordered),
        "min": ordered[0],
        "p25": round(_at(0.25), 6),
        "median": round(_at(0.50), 6),
        "p75": round(_at(0.75), 6),
        "p95": round(_at(0.95), 6),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 6),
    }


def normalize(payload: Any) -> Any:
    """Round-trip through JSON so the two sides are compared as *artefacts*.

    The aggregators key their per-stream maps by ``int(seed)``; JSON turns every key into a string
    on the way to disk. Comparing a loaded file against a live dict without this would report all
    ~100 seed-keyed maps as differences, which is a Python representation detail and not data.
    """
    return json.loads(json.dumps(payload, sort_keys=True, default=str))


def deep_diff(left: Any, right: Any, path: str = "", limit: int = 60) -> list[str]:
    """Field-by-field difference of two decoded JSON trees, as readable strings."""
    a = normalize(left)
    b = normalize(right)
    findings: list[str] = []

    def walk(x: Any, y: Any, at: str) -> None:
        if len(findings) >= limit:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y), key=repr):
                if key not in x:
                    findings.append(f"{at}.{key}: only-in-recomputed {json.dumps(y[key])[:80]}")
                elif key not in y:
                    findings.append(f"{at}.{key}: only-in-published {json.dumps(x[key])[:80]}")
                else:
                    walk(x[key], y[key], f"{at}.{key}")
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                findings.append(f"{at}: length {len(y)} (recomputed) != {len(x)} (published)")
                return
            for index, (item_x, item_y) in enumerate(zip(x, y)):
                walk(item_x, item_y, f"{at}[{index}]")
        elif x != y:
            findings.append(f"{at}: recomputed {json.dumps(y)[:80]} != published {json.dumps(x)[:80]}")

    walk(a, b, path or "$")
    return findings
