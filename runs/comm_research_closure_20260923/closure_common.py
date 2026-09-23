"""Shared machinery for the communication-closure / joint-bottleneck round.

Boundaries this module enforces (the round was authorised as 0 episode / 0 training):

* every environment entry point is wrapped to raise, so an analysis script cannot
  quietly replay an episode to fill a data gap;
* JSON output is sandboxed to this round's own directory, and a file that already
  exists is only rewritten when the new content is byte-identical, so the archived
  result files of other rounds cannot be clobbered;
* reading is streaming: the per-episode logs are hundreds of megabytes and are
  reduced to small records instead of being held in memory.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

ROUND = "comm_research_closure_20260923"

ROUND_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROUND_DIR.parent.parent

#: Rounds whose harness modules the tripwire must import in order to wrap them. They go
#: on the *end* of ``sys.path`` so this round's module names always win: each archived
#: round inserts its own directory at import time, and a silently-shadowed import would
#: mix two harnesses into one aggregate without raising anything.
SOURCE_ROUNDS = ("comm_task_pressure_learnability_20260922",)

# --------------------------------------------------------------------------- trip wire ------
#: Environment entry points that would *make* data rather than read it.
FORBIDDEN_CALLS: tuple[tuple[str, str], ...] = (
    ("pressure_env", "run_episode"),
    ("pressure_env", "run_arm"),
    ("pressure_env", "agreement_with_archived_rows"),
    ("record_layer", "record"),
    ("pressure_matrix", "run"),
)

#: Every entry point that must resolve, or the tripwire is silently useless.
ENV_ENTRY_MODULES = ("integrated_sar.joint_driver", "integrated_sar.orchestrator")

_tripwire_log: list[str] = []
_tripwire_installed = False


def enter_source_round() -> None:
    """Put this round first on ``sys.path``.

    Earlier rounds' harnesses insert their own directory at import time, so a bare
    ``import rule_matrix`` here could resolve to someone else's module and mix two
    harnesses into one aggregate without any error.
    """
    here = str(ROUND_DIR)
    root = str(REPO_ROOT)
    for path in (here, root):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, here)
    sys.path.insert(1, root)
    for round_name in SOURCE_ROUNDS:
        path = str(REPO_ROOT / "runs" / round_name)
        if path not in sys.path:
            sys.path.append(path)


def install_env_tripwire() -> list[str]:
    """Replace each forbidden entry point with a function that raises and logs.

    Returns the resolved target descriptions. Raises if a named target cannot be
    resolved: a tripwire that wraps nothing would report "no calls" truthfully while
    protecting nothing, which is the worst possible failure mode for this round.
    """
    global _tripwire_installed
    if _tripwire_installed:
        return [entry[len("INSTALLED "):] for entry in _tripwire_log
                if entry.startswith("INSTALLED ")]
    enter_source_round()
    resolved: list[str] = []
    missing: list[str] = []
    for module_name, attribute in FORBIDDEN_CALLS:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            missing.append(f"{module_name}: import failed ({type(error).__name__})")
            continue
        if not hasattr(module, attribute):
            missing.append(f"{module_name}.{attribute}: attribute gone")
            continue
        original = getattr(module, attribute)

        def blocked(*args: Any, _m: str = module_name, _a: str = attribute, **kwargs: Any) -> Any:
            _tripwire_log.append(f"BLOCKED {_m}.{_a}")
            raise RuntimeError(f"{ROUND} may not call {_m}.{_a}: this round is read-only")

        setattr(module, attribute, blocked)
        resolved.append(f"{module_name}.{_short(original)}")
    if missing:
        raise RuntimeError("env tripwire could not wrap every entry point: " + "; ".join(missing))
    _tripwire_installed = True
    _tripwire_log.extend(f"INSTALLED {item}" for item in resolved)
    return list(resolved)


def _short(function: Callable[..., Any]) -> str:
    return getattr(function, "__qualname__", repr(function))


def tripwire_calls() -> list[str]:
    return [entry for entry in _tripwire_log if entry.startswith("BLOCKED")]


def tripwire_report() -> dict[str, Any]:
    return {
        "entry_points_wrapped": [entry for entry in _tripwire_log if entry.startswith("INSTALLED")],
        "blocked_calls": tripwire_calls(),
        "blocked_call_count": len(tripwire_calls()),
    }


# Done on import, not inside install_env_tripwire(): a round that imports a project
# module at the top of a file would otherwise fail before the path is arranged.
enter_source_round()


# ------------------------------------------------------------------------------ hashing ------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------------------ writing ------
def write_json(name: str, payload: Any) -> Path:
    """Write one artefact of this round, sandboxed and append-only in effect."""
    if os.sep in name or "/" in name or ".." in name:
        raise ValueError(f"refusing to write outside the round directory: {name!r}")
    target = ROUND_DIR / name
    if target.exists() and target.name != name:
        raise RuntimeError(f"path escapes the round directory: {target}")
    rendered = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"refusing to rewrite a delivered artefact with different content: {name}")
        return target
    target.write_text(rendered, encoding="utf-8")
    return target


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# ----------------------------------------------------------------------------- reading ------
def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Stream one JSONL file. A truncated tail is reported, never silently skipped."""
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{path.name}:{number}: unparsable record ({error})") from error


def quantiles(values: list[float]) -> dict[str, Any]:
    """min/p25/median/p75/p90/max and mean, or ``None`` for an empty list."""
    if not values:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None,
                "p90": None, "max": None, "mean": None}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return float(ordered[position])

    return {
        "n": len(ordered),
        "min": float(ordered[0]),
        "p25": at(0.25),
        "median": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
        "max": float(ordered[-1]),
        "mean": round(sum(ordered) / len(ordered), 4),
    }


def ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def describe_files(paths: Iterable[Path]) -> dict[str, Any]:
    listed = list(paths)
    return {
        "files": len(listed),
        "bytes": sum(path.stat().st_size for path in listed) if listed else 0,
    }


def family_paths(directory: Path, pattern: str) -> list[Path]:
    """Resolve one log family. Shared so the freeze and the re-check cannot disagree.

    Two rounds of this project have been derailed by measuring the same thing twice with
    two code paths; a depth difference here would show up as invented "file drift".
    """
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob(pattern) if path.is_file())


#: The commit this branch was created from, verified against the remote before branching.
START_SHA = "9db4cdf073d21d2730c426dfd8767b92c7a2f117"

#: The round is *instructed* to update these, so "unchanged" is the wrong check for them.
#: Their pre-round hash comes from git, which is authoritative and cannot be edited mid-round.
GOVERNING_DOCS = ["PROJECT_CHARTER.md", "PROJECT_STATUS.md", "README.md", ".gitignore"]


def git_show_hash(rel: str, revision: str = START_SHA) -> str | None:
    """SHA256 of one file's content at ``revision``, or None if it does not exist there."""
    import hashlib
    import subprocess

    result = subprocess.run(("git", "show", f"{revision}:{rel}"), cwd=REPO_ROOT,
                            capture_output=True)
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def governing_state() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rel in GOVERNING_DOCS:
        before = git_show_hash(rel)
        path = REPO_ROOT / rel
        after = sha256_file(path) if path.exists() else None
        out[rel] = {"sha256_at_start_sha": before, "sha256_now": after,
                    "changed": before != after}
    return out
