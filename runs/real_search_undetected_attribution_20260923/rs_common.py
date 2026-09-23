"""Shared machinery for the real-PILOC undetected-goal attribution round.

Boundaries, enforced in code rather than in prose (the round is authorised as
read-only, 0 episode, 0 training, no communication-config change):

* the episode entry points are wrapped so that calling them raises, and the search and
  allocation backends may never be imported at all -- both are checked mechanically;
* an obstacle grid is *rebuilt from the logged seed* only because the runs store the
  grid's hash and not the grid; a rebuild that fails to reproduce that hash is discarded
  instead of trusted, so no geometry in this round rests on an unverified map;
* JSON output is sandboxed to this round's directory, and a file that already exists is
  only rewritten when the new bytes are identical.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable

ROUND = "real_search_undetected_attribution_20260923"

ROUND_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROUND_DIR.parent.parent

START_SHA = "037fde424b3c2c38176699085695d4a51bc8e67d"
START_BRANCH = "exp/comm-info-path-validation"

#: Entry points that would *make* data instead of reading it.
FORBIDDEN_CALLS: tuple[tuple[str, str], ...] = (
    ("integrated_sar.joint_driver", "build_joint_run"),
    ("integrated_sar.joint_driver", "JointRun.run"),
    ("integrated_sar.detection_replay", "record_detection_stream"),
)

#: Backends whose import would mean an episode or a policy forward happened somewhere.
FORBIDDEN_IMPORTS: tuple[str, ...] = ("piloc", "task_allocation", "torch")

_tripwire_log: list[str] = []
_tripwire_installed = False
#: Populated by ``install_env_tripwire`` and read by the verification script.
TRIPWIRE_INSTALLED: list[str] = []


def enter_repo() -> None:
    """This round first, then the repository root, on ``sys.path``.

    Archived rounds insert their own directory at import time; a silently shadowed
    import would mix two harnesses into one aggregate without raising anything.
    """
    here, root = str(ROUND_DIR), str(REPO_ROOT)
    for path in (here, root):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, here)
    sys.path.insert(1, root)


def _make_blocked(label: str, original: Any) -> Any:
    def blocked(*args: Any, **kwargs: Any) -> Any:
        _tripwire_log.append(label)
        raise RuntimeError(f"{label} is blocked in this round: it is read-only")

    blocked.original = original
    return blocked


def install_env_tripwire() -> list[str]:
    """Replace each forbidden entry point with a function that raises and logs.

    Raises if a named target cannot be resolved: a tripwire that wraps nothing would
    report "no calls" truthfully while protecting nothing.
    """
    global _tripwire_installed
    if _tripwire_installed:
        return list(TRIPWIRE_INSTALLED)
    resolved: list[str] = []
    for module_name, dotted in FORBIDDEN_CALLS:
        module = importlib.import_module(module_name)
        target: Any = module
        for part in dotted.split("."):
            target = getattr(target, part)
        label = f"{module_name}.{dotted}"
        blocked = _make_blocked(label, target)
        holder = module
        for part in dotted.split(".")[:-1]:
            holder = getattr(holder, part)
        setattr(holder, dotted.split(".")[-1], blocked)
        resolved.append(label)
    globals()["TRIPWIRE_INSTALLED"] = resolved
    _tripwire_installed = True
    return list(resolved)


def tripwire_calls() -> list[str]:
    return list(_tripwire_log)


def forbidden_imports_seen() -> list[str]:
    """Names of modules that must never appear in ``sys.modules`` for this round."""
    seen = set()
    for name in list(sys.modules):
        root = name.split(".")[0]
        if root in FORBIDDEN_IMPORTS:
            seen.add(root)
    return sorted(seen)


# ---------------------------------------------------------------------------- io -----------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:  # a truncated file must not read as a short one
                raise ValueError(f"{path} line {number} is not JSON: {error}") from error


_WRITTEN: set[Path] = set()


def write_json(name: str, payload: Any) -> Path:
    """Write into this round's directory only, refusing to change a delivered artefact."""
    path = (ROUND_DIR / name).resolve()
    if ROUND_DIR.resolve() not in path.parents:
        raise RuntimeError(f"refusing to write outside the round directory: {name}")
    text = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"refusing to rewrite a delivered artefact with new content: {name}")
        return path
    path.write_text(text, encoding="utf-8")
    _WRITTEN.add(path)
    return path


def register_input(path: Path, manifest: dict[str, Any]) -> Path:
    """Hash a read-only input once and remember it for the manifest."""
    key = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    if key not in manifest:
        manifest[key] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return path


# --------------------------------------------------------------------- the two batches ------
#: Both batches run the real PILOC weights on generated 60x60 scenarios. They are
#: reported separately: different seed blocks, different weeks, different drivers, and
#: no cross-batch pair is a paired comparison of anything.
BATCHES: dict[str, dict[str, Any]] = {
    "B1_d1_20260923": {
        "label": "D1 four arms x 60 seeds (round 16), live PILOC search",
        "dir": REPO_ROOT / "runs" / "comm_info_path_validation_20260923" / "logs",
        "lifecycle": "data_{arm}_s{seed}_rp.jsonl",
        "probe": "data_{arm}_s{seed}_ip.jsonl",
        "arms": ("A1", "A2", "A3", "A4"),
        "seeds": range(1100000, 1100060),
        "search_record_arm": "A1",
        "steps": 1000,
    },
    "B2_systemloop_20260918": {
        "label": "closed-loop validation block, live PILOC search (round 3 era)",
        "dir": REPO_ROOT / "runs" / "coalition_validation_20260918" / "systemloop",
        "lifecycle": "{arm}_rp_s{seed}.jsonl",
        "probe": None,
        "arms": ("baseline", "candidate"),
        "seeds": range(31000, 31060),
        "search_record_arm": "baseline",
        "steps": 1000,
    },
}


def batch_paths(batch: dict[str, Any], arm: str, seed: int) -> dict[str, Path]:
    out = {"lifecycle": batch["dir"] / batch["lifecycle"].format(arm=arm, seed=seed)}
    if batch["probe"]:
        out["probe"] = batch["dir"] / batch["probe"].format(arm=arm, seed=seed)
    return out


# ------------------------------------------------------------------------ grid geometry -----
def neighbours(cell: tuple[int, int], rows: int, cols: int, *, eight: bool) -> list[tuple[int, int]]:
    row, col = cell
    steps = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)) if eight else (
        (-1, 0), (1, 0), (0, -1), (0, 1))
    out = []
    for drow, dcol in steps:
        r, c = row + drow, col + dcol
        if 0 <= r < rows and 0 <= c < cols:
            out.append((r, c))
    return out


def components(obstacles, *, eight: bool = False) -> list[list[tuple[int, int]]]:
    """Connected free regions; ``eight=False`` matches ``scenarios.largest_connected_region``."""
    rows, cols = obstacles.shape
    seen = [[False] * cols for _ in range(rows)]
    out: list[list[tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if int(obstacles[r][c]) == 1 or seen[r][c]:
                continue
            block = [(r, c)]
            seen[r][c] = True
            queue = deque(block)
            while queue:
                cell = queue.popleft()
                for nxt in neighbours(cell, rows, cols, eight=eight):
                    if not seen[nxt[0]][nxt[1]] and int(obstacles[nxt[0]][nxt[1]]) != 1:
                        seen[nxt[0]][nxt[1]] = True
                        block.append(nxt)
                        queue.append(nxt)
            out.append(block)
    return sorted(out, key=len, reverse=True)


def component_index_map(cells: list[list[tuple[int, int]]]) -> dict[tuple[int, int], int]:
    return {cell: index for index, block in enumerate(cells) for cell in block}


def geodesic_distance(obstacles, start: tuple[int, int],
                      targets: Iterable[tuple[int, int]] | None = None) -> dict:
    """Breadth-first cell distance from ``start`` over free cells (4-neighbour).

    With no target list the whole reachable component is measured, so any cell - including a
    visibility-set cell that was never logged - can be looked up. Unreachable targets map to
    ``None`` rather than disappearing, so an absent distance is always an explicit answer.
    """
    rows, cols = obstacles.shape
    if int(obstacles[start[0]][start[1]]) == 1:
        return {tuple(cell): None for cell in (targets or ())}
    distances: dict[tuple[int, int], int] = {start: 0}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        depth = distances[cell] + 1
        for nxt in neighbours(cell, rows, cols, eight=False):
            if nxt in distances or int(obstacles[nxt[0]][nxt[1]]) == 1:
                continue
            distances[nxt] = depth
            queue.append(nxt)
    if targets is None:
        return distances
    return {tuple(cell): distances.get(tuple(cell)) for cell in targets}


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def bresenham(point_a: tuple[int, int], point_b: tuple[int, int]) -> list[tuple[int, int]]:
    """PILOC's own line predicate, re-derived here so both copies can be compared."""
    x1, y1 = point_a
    x2, y2 = point_b
    points = []
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
        points.append((x1, y1))
        if (x1, y1) == (x2, y2):
            return points
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


def piloc_line_of_sight(obstacles, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """``timemap.Environment._is_line_of_sight_blocked`` inverted, same endpoint rule."""
    for point in bresenham(a, b):
        if int(obstacles[point[0]][point[1]]) == 1 and point != b:
            return False
    return True


def goal_label(index: int, episode: int = 1) -> str:
    """The id PILOC mints for the ``index``-th goal of an episode."""
    return f"piloc-e{episode}-target-{index}"


def quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    def at(fraction: float) -> float:
        position = min(len(ordered) - 1, max(0, fraction * (len(ordered) - 1)))
        low = int(position)
        high = min(len(ordered) - 1, low + 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {
        "n": len(ordered), "min": ordered[0], "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 5), "median": round(at(0.5), 5),
        "p25": round(at(0.25), 5), "p75": round(at(0.75), 5), "p90": round(at(0.90), 5),
    }


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 5)
