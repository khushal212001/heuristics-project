"""Utility helpers for the CS572 heuristic problem solving project.

This package contains shared types and helpers used by both:
- the original Milestone 2 pipeline, and
- the extended Final Project pipeline.
"""

from __future__ import annotations

import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Coord = Tuple[int, int]
Grid = List[List[int]]  # 0 = free, 1 = blocked


@dataclass(frozen=True)
class SearchResult:
    """Result of a search run."""

    found: bool
    path: List[Coord]
    path_cost: int
    nodes_expanded: int
    execution_time_s: float
    heuristic_evals: int = 0
    tie_breaks: int = 0


def project_root() -> Path:
    """Return the project root directory (folder containing main.py)."""

    return Path(__file__).resolve().parents[2]


def ensure_dirs() -> None:
    """Create required output directories if they don't exist."""

    root = project_root()
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "results" / "plots").mkdir(parents=True, exist_ok=True)

    # Final project outputs.
    (root / "final_results" / "csv").mkdir(parents=True, exist_ok=True)
    (root / "final_results" / "plots").mkdir(parents=True, exist_ok=True)
    (root / "final_results" / "logs").mkdir(parents=True, exist_ok=True)


def manhattan(a: Coord, b: Coord) -> float:
    """Manhattan distance heuristic for 4-neighbor grid movement."""

    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def in_bounds(grid: Grid, p: Coord) -> bool:
    """Return True if coordinate is inside grid."""

    r, c = p
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])


def passable(grid: Grid, p: Coord) -> bool:
    """Return True if coordinate is not blocked."""

    r, c = p
    return grid[r][c] == 0


def neighbors4(grid: Grid, p: Coord) -> Iterable[Coord]:
    """Yield valid 4-connected neighbors."""

    r, c = p
    candidates = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    for q in candidates:
        if in_bounds(grid, q) and passable(grid, q):
            yield q


def reconstruct_path(came_from: Dict[Coord, Coord], start: Coord, goal: Coord) -> List[Coord]:
    """Reconstruct path from predecessor map."""

    if start == goal:
        return [start]
    if goal not in came_from:
        return []

    cur = goal
    path: List[Coord] = [cur]
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def path_cost(path: Sequence[Coord]) -> int:
    """Compute path cost assuming unit step costs."""

    if not path:
        return 0
    return max(0, len(path) - 1)


def generate_grid(
    *,
    size: int,
    obstacle_prob: float,
    rng: random.Random,
    start: Coord = (0, 0),
    goal: Optional[Coord] = None,
) -> Tuple[Grid, Coord, Coord]:
    """Generate a random grid with obstacles.

    Ensures start/goal are free. Does not guarantee solvability.
    """

    if not (0.0 <= obstacle_prob < 1.0):
        raise ValueError("obstacle_prob must be in [0,1)")

    if goal is None:
        goal = (size - 1, size - 1)

    grid: Grid = [[0 for _ in range(size)] for _ in range(size)]
    for r in range(size):
        for c in range(size):
            if (r, c) in (start, goal):
                continue
            grid[r][c] = 1 if rng.random() < obstacle_prob else 0

    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0
    return grid, start, goal


def obstacle_density(grid: Grid) -> float:
    """Return fraction of blocked cells in the grid."""

    if not grid or not grid[0]:
        return 0.0
    total = len(grid) * len(grid[0])
    blocked = sum(1 for r in grid for v in r if v == 1)
    return blocked / total


def write_results_csv(
    csv_path: Path,
    rows: List[Dict[str, object]],
    *,
    required_fields: Optional[Sequence[str]] = None,
) -> None:
    """Write results rows to a CSV file with a stable column order.

    If required_fields is omitted, we default to the Milestone 2 rubric fields.
    """

    ensure_dirs()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to write")

    if required_fields is None:
        required = ["algorithm", "input_size", "execution_time", "nodes_expanded", "path_length", "run_id"]
    else:
        required = list(required_fields)

    all_keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    extras = [k for k in all_keys if k not in required]
    fieldnames = required + extras

    for row in rows:
        for key in required:
            if key not in row:
                raise ValueError(f"Missing required field {key}")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(values: Sequence[float]) -> float:
    """Compute arithmetic mean."""

    if not values:
        return math.nan
    return sum(values) / len(values)


def stdev(values: Sequence[float]) -> float:
    """Compute sample standard deviation (ddof=1)."""

    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return math.sqrt(var)


def ci95_half_width(values: Sequence[float]) -> float:
    """Return 95% CI half-width for the mean (normal approximation)."""

    n = len(values)
    if n < 2:
        return 0.0
    return 1.96 * (stdev(values) / math.sqrt(n))


def pct_improvement(baseline: float, improved: float) -> float:
    """Percentage improvement where smaller is better."""

    if baseline == 0:
        return math.nan
    return 100.0 * (baseline - improved) / baseline


def now_s() -> float:
    """High-resolution wall-clock time."""

    return time.perf_counter()
