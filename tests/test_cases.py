"""Correctness tests and validation suite.

These tests are executed automatically by `main.py`.

In addition to basic cases, this file validates baseline A* correctness by:
- checking path feasibility (no obstacles, 4-connected)
- proving optimality on small grids by comparing A* cost against BFS
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.baseline import astar_search
from src.enhanced import WeightedAStarConfig, weighted_astar_search
from src.utils import Grid, generate_grid
from src.validation import assert_optimal_vs_bfs_small, validate_path


def _assert_path_valid(grid: Grid, path: List[Tuple[int, int]]) -> None:
    if not path:
        raise AssertionError("Expected non-empty path")
    # Kept for backwards-compat, but primary validation uses src.validation.validate_path.
    for (r, c) in path:
        if grid[r][c] != 0:
            raise AssertionError(f"Path goes through obstacle at {(r, c)}")
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            raise AssertionError("Path is not 4-connected")


def run_validation_suite() -> Dict[str, object]:
    """Run extra validation checks and return a small summary dict."""

    # Validate optimality on multiple small random grids (deterministic seeds).
    seeds = [7, 11, 23, 31, 47]
    validated = 0
    for seed in seeds:
        # Use retry generation to ensure solvable.
        import random

        rng = random.Random(seed)
        for _ in range(50):
            grid, start, goal = generate_grid(size=12, obstacle_prob=0.22, rng=rng)
            res = astar_search(grid, start, goal)
            if res.found:
                pv = validate_path(grid, start, goal, res.path)
                if not pv.valid:
                    raise AssertionError(f"A* returned invalid path: {pv.reason}")
                assert_optimal_vs_bfs_small(grid, start, goal, res.path_cost)
                validated += 1
                break
        else:
            raise AssertionError("Failed to generate a solvable small grid for validation")

    return {"small_grids_validated": validated, "seeds": len(seeds)}


def run_all_tests() -> None:
    """Run all correctness tests (raises AssertionError on failure)."""

    # Test 1: Small grid with a clear corridor; known optimal length.
    grid1: Grid = [
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
    ]
    start1 = (0, 0)
    goal1 = (4, 4)
    res1 = astar_search(grid1, start1, goal1)
    if not res1.found:
        raise AssertionError("A* should find a path in test 1")
    _assert_path_valid(grid1, res1.path)
    pv1 = validate_path(grid1, start1, goal1, res1.path)
    if not pv1.valid:
        raise AssertionError(f"Test 1 path invalid: {pv1.reason}")
    if res1.path_cost != 8:
        raise AssertionError(f"Expected optimal cost 8, got {res1.path_cost}")
    # Prove optimality on this small grid by comparing to BFS.
    assert_optimal_vs_bfs_small(grid1, start1, goal1, res1.path_cost)

    # Test 2: Medium grid; verify WA* finds a path and it's valid.
    # Design: a few obstacle "walls" with gaps, but a guaranteed perimeter path.
    grid2: Grid = [[0 for _ in range(10)] for _ in range(10)]
    # Horizontal wall with a gap.
    for c in range(1, 9):
        grid2[4][c] = 1
    grid2[4][7] = 0
    # Vertical wall with a gap.
    for r in range(1, 9):
        grid2[r][6] = 1
    grid2[7][6] = 0
    # Small block cluster.
    for r, c in [(2, 2), (2, 3), (3, 2)]:
        grid2[r][c] = 1
    start2 = (0, 0)
    goal2 = (9, 9)
    res2 = weighted_astar_search(grid2, start2, goal2, config=WeightedAStarConfig(weight=1.8))
    if not res2.found:
        raise AssertionError("WA* should find a path in test 2")
    _assert_path_valid(grid2, res2.path)
    pv2 = validate_path(grid2, start2, goal2, res2.path)
    if not pv2.valid:
        raise AssertionError(f"Test 2 path invalid: {pv2.reason}")

    # Test 3 (edge case): start == goal.
    grid3: Grid = [[0, 1], [0, 0]]
    start3 = (1, 1)
    goal3 = (1, 1)
    res3 = astar_search(grid3, start3, goal3)
    if not res3.found or res3.path_cost != 0 or res3.path != [start3]:
        raise AssertionError("start==goal should return zero-cost trivial path")

    # Note: additional baseline validation is run by main.py to print a correctness summary.
