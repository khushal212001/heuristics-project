"""Enhanced implementation: Weighted A* (WA*) with heuristic memoization.

WA* uses f(n) = g(n) + w * h(n), w >= 1.
For w>1 this typically expands fewer nodes and runs faster, but may produce
suboptimal (longer) paths; this is acceptable for the "measurable improvement"
criterion since the improvement is in computational efficiency metrics.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .utils import Coord, Grid, SearchResult, manhattan, neighbors4, now_s, path_cost, reconstruct_path


@dataclass(frozen=True)
class WeightedAStarConfig:
    """Configuration for Weighted A*."""

    weight: float = 1.5
    tie_breaker: str = "g"  # "g" prefers deeper nodes; "h" prefers lower heuristic
    memoize_heuristic: bool = True

    # Additional enhancement beyond basic WA*: dynamic weight schedule.
    # When enabled, the effective weight decreases linearly with expansions:
    #   w_eff = max(weight_min, weight - weight_decay_per_1000 * (expanded/1000))
    dynamic_weight: bool = False
    weight_min: float = 1.0
    weight_decay_per_1000: float = 0.25


def weighted_astar_search(
    grid: Grid,
    start: Coord,
    goal: Coord,
    heuristic: Callable[[Coord, Coord], float] = manhattan,
    config: Optional[WeightedAStarConfig] = None,
) -> SearchResult:
    """Run Weighted A* search.

    Args:
        grid: 2D grid, 0 = free, 1 = blocked.
        start: start coordinate.
        goal: goal coordinate.
        heuristic: heuristic function.
        config: WeightedAStarConfig.

    Returns:
        SearchResult including path, expanded nodes, and timing.
    """

    if config is None:
        config = WeightedAStarConfig()

    if config.weight < 1.0:
        raise ValueError("weight must be >= 1.0")
    if config.weight_min < 1.0:
        raise ValueError("weight_min must be >= 1.0")
    if config.weight_min > config.weight:
        raise ValueError("weight_min must be <= weight")
    if config.weight_decay_per_1000 < 0.0:
        raise ValueError("weight_decay_per_1000 must be >= 0")

    t0 = now_s()

    if start == goal:
        t1 = now_s()
        return SearchResult(
            found=True,
            path=[start],
            path_cost=0,
            nodes_expanded=0,
            execution_time_s=t1 - t0,
            heuristic_evals=0,
        )

    # Heuristic evaluation (optionally memoized, for ablation).
    # Memoization is used in the ablation experiment to isolate caching impact.
    # We use a 2D array cache because dict overhead can dominate for cheap heuristics
    # like Manhattan distance.
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    heuristic_evals = 0
    tie_breaks = 0
    h_cache_arr: Optional[List[List[Optional[float]]]] = None
    h_cache_dict: Optional[Dict[Coord, float]] = None
    if config.memoize_heuristic and rows > 0 and cols > 0:
        h_cache_arr = [[None for _ in range(cols)] for _ in range(rows)]
    elif config.memoize_heuristic:
        h_cache_dict = {}

    def h(node: Coord) -> float:
        nonlocal heuristic_evals
        if not config.memoize_heuristic:
            # No-memo variant: every call is a real heuristic computation.
            heuristic_evals += 1
            return float(heuristic(node, goal))

        if h_cache_arr is not None:
            r, c = node
            v = h_cache_arr[r][c]
            if v is None:
                # Cache miss: compute and store.
                heuristic_evals += 1
                v = float(heuristic(node, goal))
                h_cache_arr[r][c] = v
            return v

        assert h_cache_dict is not None
        v2 = h_cache_dict.get(node)
        if v2 is None:
            heuristic_evals += 1
            v2 = float(heuristic(node, goal))
            h_cache_dict[node] = v2
        return v2

    def w_eff(expanded: int) -> float:
        # Effective weight schedule.
        if not config.dynamic_weight:
            return config.weight
        w = config.weight - config.weight_decay_per_1000 * (expanded / 1000.0)
        return max(config.weight_min, w)

    # OPEN set is a heap ordered by f = g + w*h.
    # Each entry: (f_score, secondary_key, node)
    open_heap: List[Tuple[float, float, Coord]] = []
    f_counts: Dict[float, int] = {}
    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, int] = {start: 0}
    # CLOSED set: expanded nodes.
    closed: set[Coord] = set()

    h0 = h(start)
    f0 = 0.0 + w_eff(0) * h0
    secondary0 = h0 if config.tie_breaker == "h" else -0.0
    if f_counts.get(f0, 0) > 0:
        tie_breaks += 1
    f_counts[f0] = f_counts.get(f0, 0) + 1
    heapq.heappush(open_heap, (f0, secondary0, start))

    nodes_expanded = 0

    while open_heap:
        f, _, current = heapq.heappop(open_heap)
        if f in f_counts:
            f_counts[f] -= 1
            if f_counts[f] <= 0:
                del f_counts[f]
        if current in closed:
            # Skip stale heap entries.
            continue
        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            t1 = now_s()
            return SearchResult(
                found=True,
                path=path,
                path_cost=path_cost(path),
                nodes_expanded=nodes_expanded,
                execution_time_s=t1 - t0,
                heuristic_evals=heuristic_evals,
                tie_breaks=tie_breaks,
            )

        cur_g = g_score[current]
        for nxt in neighbors4(grid, current):
            if nxt in closed:
                continue

            tentative_g = cur_g + 1
            best_g = g_score.get(nxt)
            if best_g is None or tentative_g < best_g:
                # Found an improved path to nxt.
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                hn = h(nxt)
                fn = float(tentative_g) + w_eff(nodes_expanded) * hn
                secondary = hn if config.tie_breaker == "h" else -float(tentative_g)
                if f_counts.get(fn, 0) > 0:
                    tie_breaks += 1
                f_counts[fn] = f_counts.get(fn, 0) + 1
                heapq.heappush(open_heap, (fn, secondary, nxt))

    t1 = now_s()
    return SearchResult(
        found=False,
        path=[],
        path_cost=0,
        nodes_expanded=nodes_expanded,
        execution_time_s=t1 - t0,
        heuristic_evals=heuristic_evals,
        tie_breaks=tie_breaks,
    )
