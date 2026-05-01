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
    tie_breaks = 0

    def h(node: Coord) -> float:
        nonlocal heuristic_evals
        heuristic_evals += 1
        return float(heuristic(node, goal))

    # OPEN set stored as a priority queue (min-heap) ordered by f = g + h.
    # Each entry: (f_score, secondary_key, node)
    open_heap: List[Tuple[float, float, Coord]] = []

    # Track duplicate f-scores so tie-breaking events are measurable.
    f_counts: Dict[float, int] = {}

    # For path reconstruction.
    came_from: Dict[Coord, Coord] = {}

    # Best known cost-to-come for each discovered node.
    g_score: Dict[Coord, int] = {start: 0}

    # CLOSED set: nodes we've expanded (popped for expansion) already.
    closed: set[Coord] = set()

    h0 = h(start)
    f0 = 0.0 + h0
    secondary0 = h0 if config.tie_breaker == "h" else -0.0
    if f_counts.get(f0, 0) > 0:
        tie_breaks += 1
    f_counts[f0] = f_counts.get(f0, 0) + 1
    heapq.heappush(open_heap, (f0, secondary0, start))

    nodes_expanded = 0

    while open_heap:
        f, _, current = heapq.heappop(open_heap)
        # Maintain f_counts to keep tie-breaking event count meaningful.
        if f in f_counts:
            f_counts[f] -= 1
            if f_counts[f] <= 0:
                del f_counts[f]
        if current in closed:
            # Stale heap entry (a better path to this node was found later).
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
                # Found a strictly better path to nxt.
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                hn = h(nxt)
                fn = float(tentative_g) + hn
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
