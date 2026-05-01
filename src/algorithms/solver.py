"""Config-driven solver entrypoint.

This keeps the original milestone algorithms intact and provides a single
function (`solve`) that reads from src/config.py.

Enhancements supported here:
- Tie-breaking (prefer lower h on f ties)
- Multi-heuristic selection (Manhattan/Euclidean/Hybrid)
- Density-adaptive strategy (switch algorithm/weight based on obstacle density)
"""

from __future__ import annotations

from dataclasses import replace

from ..baseline import AStarConfig, astar_search
from ..config import Config
from ..enhanced import WeightedAStarConfig, weighted_astar_search
from ..heuristics import get_heuristic
from ..utils import Coord, Grid, SearchResult, obstacle_density


def effective_strategy(grid: Grid, config: Config) -> tuple[str, float]:
    """Return the effective (algorithm, weight) after applying adaptive policies."""

    algorithm = config.algorithm
    weight = float(config.weight)

    if config.use_density_adaptive or algorithm == "adaptive":
        dens = obstacle_density(grid)

        # 3-band policy:
        # - low density: use A*
        # - mid density: WA* with moderate weight
        # - high density: WA* with higher weight (capped)
        if dens < config.density_low_threshold:
            return "astar", 1.0

        if dens < config.density_threshold:
            w = max(weight, float(config.mid_density_weight))
            w = min(w, float(config.density_weight_cap))
            return "wastar", w

        w = max(weight, float(config.high_density_weight))
        w = min(w, float(config.density_weight_cap))
        return "wastar", w

    return algorithm, weight


def solve(grid: Grid, start: Coord, goal: Coord, config: Config) -> SearchResult:
    """Solve a single instance according to Config."""

    config.validate()

    # Choose heuristic function (multi-heuristic enhancement).
    heuristic_fn = get_heuristic(config.heuristic, alpha=config.alpha)

    # Tie-breaking enhancement: when f ties, prefer lower h.
    tie_breaker = "h" if config.use_tiebreaking else "g"

    # Density-adaptive strategy enhancement.
    algorithm, weight = effective_strategy(grid, config)

    if algorithm == "astar":
        return astar_search(grid, start, goal, heuristic=heuristic_fn, config=AStarConfig(tie_breaker=tie_breaker))

    if algorithm == "wastar":
        wa_cfg = WeightedAStarConfig(
            weight=float(weight),
            tie_breaker=tie_breaker,
            memoize_heuristic=bool(config.use_memoization),
            dynamic_weight=bool(config.use_dynamic_weight),
            weight_min=float(config.weight_min),
            weight_decay_per_1000=float(config.weight_decay_per_1000),
        )
        return weighted_astar_search(grid, start, goal, heuristic=heuristic_fn, config=wa_cfg)

    raise ValueError(f"Unsupported algorithm: {algorithm}")
