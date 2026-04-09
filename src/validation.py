"""Validation utilities for correctness and optimality.

Milestone 2 requires demonstrating baseline correctness.
This module provides:
- path validity checks (no obstacle traversal, 4-connected steps)
- BFS optimal cost for small grids (gold standard for unweighted shortest path)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from .utils import Coord, Grid, neighbors4


@dataclass(frozen=True)
class PathValidity:
    """Details about path validity."""

    valid: bool
    reason: str


def validate_path(grid: Grid, start: Coord, goal: Coord, path: Sequence[Coord]) -> PathValidity:
    """Validate that a path is well-formed and feasible on the grid."""

    if start == goal:
        if list(path) != [start]:
            return PathValidity(False, "start==goal requires path == [start]")
        return PathValidity(True, "ok")

    if not path:
        return PathValidity(False, "empty path")

    if path[0] != start:
        return PathValidity(False, "path does not start at start")

    if path[-1] != goal:
        return PathValidity(False, "path does not end at goal")

    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    for p in path:
        r, c = p
        if not (0 <= r < rows and 0 <= c < cols):
            return PathValidity(False, f"out of bounds: {p}")
        if grid[r][c] != 0:
            return PathValidity(False, f"path crosses obstacle at {p}")

    for a, b in zip(path, path[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return PathValidity(False, "path is not 4-connected")

    return PathValidity(True, "ok")


def bfs_shortest_path_cost(grid: Grid, start: Coord, goal: Coord) -> Optional[int]:
    """Compute optimal shortest-path cost using BFS (unit edge weights).

    Returns:
        Optimal cost if a path exists, else None.
    """

    if start == goal:
        return 0

    q: Deque[Coord] = deque([start])
    dist: Dict[Coord, int] = {start: 0}

    while q:
        cur = q.popleft()
        if cur == goal:
            return dist[cur]
        for nxt in neighbors4(grid, cur):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)

    return None


def assert_optimal_vs_bfs_small(grid: Grid, start: Coord, goal: Coord, path_cost: int) -> None:
    """Assert A* optimality by comparing to BFS optimal cost on small grids."""

    optimal = bfs_shortest_path_cost(grid, start, goal)
    if optimal is None:
        raise AssertionError("BFS found no path; test grid expected solvable")
    if path_cost != optimal:
        raise AssertionError(f"Expected optimal cost {optimal}, got {path_cost}")
