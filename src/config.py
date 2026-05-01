"""Unified, config-driven interface for the Final Project pipeline.

The milestone code remains intact. The final pipeline consumes this Config and
routes execution through src/algorithms/solver.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Configuration for selecting algorithms and enhancements."""

    # Core selection
    algorithm: str = "astar"  # "astar" | "wastar" | "adaptive"

    # WA* / dynamic weighting
    weight: float = 1.5
    use_dynamic_weight: bool = True
    weight_min: float = 1.0
    weight_decay_per_1000: float = 0.12

    # Heuristic selection
    heuristic: str = "manhattan"  # "manhattan" | "euclidean" | "hybrid"
    alpha: float = 0.7  # hybrid: alpha*manhattan + (1-alpha)*euclidean

    # Enhancements (toggles)
    use_memoization: bool = True
    use_tiebreaking: bool = False  # when f ties, prefer lower h
    use_density_adaptive: bool = False

    # Density-adaptive policy
    density_low_threshold: float = 0.20
    density_threshold: float = 0.35  # high-density cutoff for 3-band policy
    mid_density_weight: float = 1.6
    high_density_weight: float = 2.2
    density_weight_cap: float = 2.5

    def validate(self) -> None:
        if self.algorithm not in {"astar", "wastar", "adaptive"}:
            raise ValueError("algorithm must be one of: astar, wastar, adaptive")
        if self.weight < 1.0:
            raise ValueError("weight must be >= 1.0")
        if self.weight_min < 1.0:
            raise ValueError("weight_min must be >= 1.0")
        if self.weight_min > self.weight:
            raise ValueError("weight_min must be <= weight")
        if self.weight_decay_per_1000 < 0.0:
            raise ValueError("weight_decay_per_1000 must be >= 0")
        if self.heuristic not in {"manhattan", "euclidean", "hybrid"}:
            raise ValueError("heuristic must be one of: manhattan, euclidean, hybrid")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in [0,1]")
        if not (0.0 <= self.density_threshold <= 1.0):
            raise ValueError("density_threshold must be in [0,1]")
        if not (0.0 <= self.density_low_threshold <= 1.0):
            raise ValueError("density_low_threshold must be in [0,1]")
        if self.density_low_threshold > self.density_threshold:
            raise ValueError("density_low_threshold must be <= density_threshold")
        if self.mid_density_weight < 1.0:
            raise ValueError("mid_density_weight must be >= 1.0")
        if self.high_density_weight < 1.0:
            raise ValueError("high_density_weight must be >= 1.0")
        if self.density_weight_cap < 1.0:
            raise ValueError("density_weight_cap must be >= 1.0")
        if self.density_weight_cap < min(self.mid_density_weight, self.high_density_weight):
            raise ValueError("density_weight_cap must be >= mid_density_weight and high_density_weight")
