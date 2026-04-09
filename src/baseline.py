"""Baseline implementation: standard A* search on a 2D grid."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .utils import Coord, Grid, SearchResult, manhattan, neighbors4, now_s, path_cost, reconstruct_path


@dataclass(frozen=True)
class AStarConfig:
    """Configuration for A* search."""

    tie_breaker: str = "g"  # "g" prefers deeper nodes on ties; "h" prefers lower heuristic


def astar_search(
    grid: Grid,
    start: Coord,
    goal: Coord,
    heuristic=manhattan,
    config: Optional[AStarConfig] = None,
) -> SearchResult:
    """Run A* search.

    Args:
        grid: 2D grid, 0 = free, 1 = blocked.
        start: start coordinate.
        goal: goal coordinate.
        heuristic: admissible heuristic (default Manhattan).
        config: AStarConfig.

    Returns:
        SearchResult including path, expanded nodes, and timing.
    """

    if config is None:
        config = AStarConfig()

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

    heuristic_evals = 0

    def h(node: Coord) -> int:
        nonlocal heuristic_evals
        heuristic_evals += 1
        return heuristic(node, goal)

    open_heap: List[Tuple[float, float, Coord]] = []
    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, int] = {start: 0}
    closed: set[Coord] = set()

    h0 = h(start)

    # Priority: (f, tie, node). tie is used for deterministic tie-breaking.
    def tie_value(node: Coord) -> float:
        if config.tie_breaker == "h":
            return float(h(node))
        # Default "g": prefer larger g (deeper) to reduce re-expansions.
        return -float(g_score.get(node, 0))

    heapq.heappush(open_heap, (h0, tie_value(start), start))

    nodes_expanded = 0

    while open_heap:
        f, _, current = heapq.heappop(open_heap)
        if current in closed:
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
            )

        cur_g = g_score[current]
        for nxt in neighbors4(grid, current):
            if nxt in closed:
                continue

            tentative_g = cur_g + 1
            best_g = g_score.get(nxt)
            if best_g is None or tentative_g < best_g:
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                hn = h(nxt)
                heapq.heappush(open_heap, (tentative_g + hn, tie_value(nxt), nxt))

    t1 = now_s()
    return SearchResult(
        found=False,
        path=[],
        path_cost=0,
        nodes_expanded=nodes_expanded,
        execution_time_s=t1 - t0,
        heuristic_evals=heuristic_evals,
    )
