"""Heuristics for grid pathfinding.

Final-project requirement: multi-heuristic support.

All heuristics take (node, goal) and return a non-negative estimate of remaining cost.
"""

from __future__ import annotations

import math
from typing import Callable

from ..utils import Coord


HeuristicFn = Callable[[Coord, Coord], float]


def manhattan(node: Coord, goal: Coord) -> float:
    """Manhattan distance (admissible for 4-connected grids)."""

    return float(abs(node[0] - goal[0]) + abs(node[1] - goal[1]))


def euclidean(node: Coord, goal: Coord) -> float:
    """Euclidean distance (also admissible when each step cost is 1)."""

    dx = float(node[0] - goal[0])
    dy = float(node[1] - goal[1])
    return math.sqrt(dx * dx + dy * dy)


def hybrid(alpha: float) -> HeuristicFn:
    """Return hybrid heuristic: alpha*Manhattan + (1-alpha)*Euclidean."""

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0,1]")

    def h(node: Coord, goal: Coord) -> float:
        return alpha * manhattan(node, goal) + (1.0 - alpha) * euclidean(node, goal)

    return h


def get_heuristic(name: str, *, alpha: float = 0.7) -> HeuristicFn:
    """Factory for heuristic functions by name."""

    if name == "manhattan":
        return manhattan
    if name == "euclidean":
        return euclidean
    if name == "hybrid":
        return hybrid(alpha)
    raise ValueError(f"Unknown heuristic: {name}")
