"""Deterministic scenario construction for the shared search-and-rescue world.

PILOC and task-03 are two different simulators, but architecture invariant 10 requires
them to share one scene and one clock. This module produces that scene: an obstacle
grid plus reproducible search-robot starts and hidden target cells.
"""

from __future__ import annotations

import csv
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .contracts import CoordinateTransform, GridPosition
from .errors import AssetMissingError, DependencyMissingError, ScenarioError


# The PILOC convolutional policy and its timestamp maps are built for a fixed canvas.
PILOC_GRID_SIZE = 60

IMAGE_SUFFIXES = frozenset({".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"})
ARRAY_SUFFIXES = frozenset({".npy"})
TEXT_SUFFIXES = frozenset({".csv", ".txt"})
SUPPORTED_MAP_SUFFIXES = IMAGE_SUFFIXES | ARRAY_SUFFIXES | TEXT_SUFFIXES


@dataclass(frozen=True, slots=True, eq=False)
class SearchScenario:
    """One reproducible world: obstacles, search-robot starts and hidden targets."""

    obstacles: np.ndarray
    agent_positions: tuple[GridPosition, ...]
    goal_positions: tuple[GridPosition, ...]
    seed: int
    source: str
    num_goals_placed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return int(self.obstacles.shape[0])

    @property
    def cols(self) -> int:
        return int(self.obstacles.shape[1])

    @property
    def transform(self) -> CoordinateTransform:
        return CoordinateTransform(rows=self.rows, cols=self.cols)

    @property
    def free_cell_count(self) -> int:
        return int(np.count_nonzero(self.obstacles == 0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "seed": self.seed,
            "source": self.source,
            "obstacle_cells": int(np.count_nonzero(self.obstacles == 1)),
            "free_cells": self.free_cell_count,
            "agent_positions": [[p.row, p.col] for p in self.agent_positions],
            "goal_positions": [[p.row, p.col] for p in self.goal_positions],
            "num_goals_placed": self.num_goals_placed,
        }


def generate_obstacle_map(
    size: int = PILOC_GRID_SIZE,
    seed: int = 0,
    obstacle_density: float = 0.15,
    min_free_cells: int = 2,
) -> np.ndarray:
    """Produce an obstacle grid that is guaranteed to contain one usable region."""
    if size < 8:
        raise ScenarioError(f"generated maps must be at least 8x8, got {size}x{size}")
    if not 0.0 <= obstacle_density < 0.9:
        raise ScenarioError(f"obstacle_density must be in [0, 0.9), got {obstacle_density}")

    rng = np.random.default_rng(seed)
    density = float(obstacle_density)
    for _ in range(12):
        obstacles = (rng.random((size, size)) < density).astype(np.int32)
        if len(largest_connected_region(obstacles)) >= max(min_free_cells, 2):
            return obstacles
        density *= 0.6
    raise ScenarioError(
        f"could not generate a {size}x{size} map with a connected free region "
        f"(seed={seed}, obstacle_density={obstacle_density})"
    )


def load_obstacle_map(path: str | os.PathLike[str], size: int = PILOC_GRID_SIZE) -> np.ndarray:
    """
    Read an obstacle map and normalise it to ``size`` x ``size`` of 0/1 cells.

    Images follow PILOC's own convention: dark pixels are obstacles, and the picture is
    resized to the grid the convolutional policy expects.
    """
    resolved = os.path.abspath(os.fspath(path))
    if not os.path.exists(resolved):
        raise AssetMissingError(f"map file not found: {resolved}")
    if not os.path.isfile(resolved):
        raise AssetMissingError(f"map path is not a file: {resolved}")

    suffix = os.path.splitext(resolved)[1].lower()
    if suffix in ARRAY_SUFFIXES:
        raw = _load_npy(resolved)
    elif suffix in IMAGE_SUFFIXES:
        raw = _load_image(resolved, size)
    elif suffix in TEXT_SUFFIXES:
        raw = _load_text(resolved)
    else:
        raise AssetMissingError(
            f"unsupported map format '{suffix or os.path.basename(resolved)}' for {resolved}; "
            f"supported suffixes: {', '.join(sorted(SUPPORTED_MAP_SUFFIXES))}"
        )

    obstacles = _binarize(raw, resolved)
    if obstacles.shape != (size, size):
        obstacles = _resize_nearest(obstacles, size)
    if obstacles.all():
        raise ScenarioError(f"map {resolved} is entirely obstacle; no free cell to search")
    return obstacles


def build_scenario(
    *,
    map_path: str | os.PathLike[str] | None = None,
    size: int = PILOC_GRID_SIZE,
    seed: int = 0,
    num_agents: int = 2,
    num_goals: int = 6,
    obstacle_density: float = 0.15,
    cluster_radius: int | None = None,
) -> SearchScenario:
    """
    Build one shared scenario.

    Search robots and targets are placed inside a single cluster of the largest
    connected free region, which mirrors how casualties concentrate in one disaster
    zone and keeps short evaluation runs able to discover something.
    """
    if num_agents < 1:
        raise ScenarioError("a scenario needs at least one search robot")
    if num_goals < 0:
        raise ScenarioError("num_goals cannot be negative")

    if map_path is None:
        obstacles = generate_obstacle_map(size, seed, obstacle_density)
        source = f"generated:{size}x{size}:density={obstacle_density}"
    else:
        obstacles = load_obstacle_map(map_path, size)
        source = f"file:{os.path.abspath(os.fspath(map_path))}"

    region = largest_connected_region(obstacles)
    required = num_agents + num_goals
    if len(region) < required:
        raise ScenarioError(
            f"map from {source} has a largest free region of {len(region)} cells, "
            f"but {num_agents} search robots and {num_goals} targets need {required}"
        )

    radius = int(cluster_radius) if cluster_radius is not None else max(4, size // 6)
    rng = np.random.default_rng(seed)
    center = region[int(rng.integers(len(region)))]
    ordered = sorted(
        region,
        key=lambda cell: (_chebyshev(cell, center), cell[0], cell[1]),
    )
    cluster = [cell for cell in ordered if _chebyshev(cell, center) <= radius]
    if len(cluster) < required:
        cluster = ordered

    agent_cells = cluster[:num_agents]
    pool = cluster[num_agents:]
    if num_goals:
        picks = rng.choice(len(pool), size=min(num_goals, len(pool)), replace=False)
        goal_cells = sorted((pool[int(index)] for index in picks))
    else:
        goal_cells = []
    placed = len(goal_cells)
    if placed < num_goals:
        goal_cells = list(goal_cells)
        for cell in pool:
            if placed >= num_goals:
                break
            if cell not in goal_cells:
                goal_cells.append(cell)
                placed += 1

    return SearchScenario(
        obstacles=obstacles,
        agent_positions=tuple(GridPosition(int(r), int(c)) for r, c in agent_cells),
        goal_positions=tuple(GridPosition(int(r), int(c)) for r, c in goal_cells),
        seed=int(seed),
        source=source,
        num_goals_placed=placed,
        metadata={"cluster_radius": radius, "cluster_center": [int(center[0]), int(center[1])]},
    )


def largest_connected_region(obstacles: np.ndarray) -> list[tuple[int, int]]:
    """Return the 4-connected free region with the most cells."""
    grid = np.asarray(obstacles)
    rows, cols = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    best: list[tuple[int, int]] = []
    for start_row in range(rows):
        for start_col in range(cols):
            if grid[start_row, start_col] != 0 or seen[start_row, start_col]:
                continue
            queue = deque([(start_row, start_col)])
            seen[start_row, start_col] = True
            region: list[tuple[int, int]] = []
            while queue:
                row, col = queue.popleft()
                region.append((row, col))
                for next_row, next_col in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ):
                    if not 0 <= next_row < rows or not 0 <= next_col < cols:
                        continue
                    if seen[next_row, next_col] or grid[next_row, next_col] != 0:
                        continue
                    seen[next_row, next_col] = True
                    queue.append((next_row, next_col))
            if len(region) > len(best):
                best = region
    return best


def bresenham_line(row0: int, col0: int, row1: int, col1: int) -> list[tuple[int, int]]:
    """Same rasterisation PILOC uses, so both simulators agree on line of sight."""
    points: list[tuple[int, int]] = []
    dr = abs(row1 - row0)
    dc = abs(col1 - col0)
    sr = 1 if row0 < row1 else -1
    sc = 1 if col0 < col1 else -1
    error = dr - dc
    row, col = row0, col0
    while True:
        points.append((row, col))
        if row == row1 and col == col1:
            break
        doubled = 2 * error
        if doubled > -dc:
            error -= dc
            row += sr
        if doubled < dr:
            error += dr
            col += sc
    return points


def has_line_of_sight(obstacles: np.ndarray, start: Sequence[int], goal: Sequence[int]) -> bool:
    """True when the straight line between two cells crosses no obstacle."""
    grid = np.asarray(obstacles)
    for row, col in bresenham_line(int(start[0]), int(start[1]), int(goal[0]), int(goal[1])):
        if not 0 <= row < grid.shape[0] or not 0 <= col < grid.shape[1]:
            return False
        if grid[row, col] == 1 and (row, col) != (int(goal[0]), int(goal[1])):
            return False
    return True


def _chebyshev(cell: Sequence[int], center: Sequence[int]) -> int:
    return max(abs(int(cell[0]) - int(center[0])), abs(int(cell[1]) - int(center[1])))


def _load_npy(path: str) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - surfaced as an actionable driver error
        raise AssetMissingError(f"cannot read numpy map {path}: {exc}") from exc


def _load_image(path: str, size: int) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise DependencyMissingError(
            f"reading image map {path} requires Pillow ('pip install Pillow')"
        ) from exc
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L").resize((size, size)), dtype=np.int32)
    except Exception as exc:  # noqa: BLE001
        raise AssetMissingError(f"cannot read image map {path}: {exc}") from exc


def _load_text(path: str) -> np.ndarray:
    rows: list[list[int]] = []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            if "," in sample:
                line_iterator: Any = csv.reader(handle, delimiter=",")
            else:
                line_iterator = (line.split() for line in handle)
            for line in line_iterator:
                cells = [token for token in (value.strip() for value in line) if token]
                if not cells:
                    continue
                rows.append([_parse_cell(token, path) for token in cells])
    except AssetMissingError:
        raise
    except OSError as exc:
        raise AssetMissingError(f"cannot read text map {path}: {exc}") from exc

    if not rows:
        raise ScenarioError(f"text map {path} contains no rows")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ScenarioError(f"text map {path} is not rectangular: row widths {sorted(widths)}")
    return np.asarray(rows, dtype=np.int32)


def _parse_cell(token: str, path: str) -> int:
    try:
        value = int(float(token))
    except ValueError as exc:
        raise ScenarioError(f"text map {path} contains non-numeric cell {token!r}") from exc
    if value not in (0, 1):
        raise ScenarioError(f"text map {path} contains cell value {value}; only 0 and 1 are allowed")
    return value


def _binarize(raw: np.ndarray, origin: str) -> np.ndarray:
    array = np.asarray(raw)
    if array.ndim != 2:
        raise ScenarioError(f"map {origin} must be 2-dimensional, got shape {array.shape}")
    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(np.int32)
    if np.issubdtype(array.dtype, np.integer):
        # Image maps already arrive as grey levels, text/npy maps as 0/1 flags.
        unique = np.unique(array)
        if unique.size and unique.max() > 1:
            return (array < 128).astype(np.int32)
        return (array != 0).astype(np.int32)
    if np.issubdtype(array.dtype, np.floating):
        return (array > 0.5).astype(np.int32)
    raise ScenarioError(f"map {origin} has unsupported dtype {array.dtype}")


def _resize_nearest(obstacles: np.ndarray, size: int) -> np.ndarray:
    rows, cols = obstacles.shape
    row_index = np.minimum((np.arange(size) * rows) // size, rows - 1)
    col_index = np.minimum((np.arange(size) * cols) // size, cols - 1)
    return obstacles[np.ix_(row_index, col_index)].astype(np.int32)
