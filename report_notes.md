# CS572 Milestone 2 — Report Notes (Draft)

## Implementation Summary
- **Problem**: 2D grid pathfinding with obstacles (4-connected movement, unit step cost).
- **Baseline algorithm**: A* search using Manhattan distance heuristic.
- **Enhanced algorithm family**:
  - Weighted A* (WA*): $f(n) = g(n) + w\cdot h(n)$
  - Heuristic memoization (cache $h(n)$ per run)
  - Additional enhancement: **dynamic weight schedule** (effective $w$ decreases with search progress), plus a consistent tie-breaker favoring lower $h(n)$.

## Design Choices (Why These)
- A* is a standard, correct heuristic search baseline with optimality guarantees under an admissible heuristic.
- WA* is a standard enhancement that typically reduces node expansions/time at the cost of potential suboptimality.
- Memoizing $h(n)$ is a clean micro-optimization that reduces repeated heuristic computation (especially when a node enters OPEN multiple times).
- Dynamic weight is a “stronger enhancement” that can improve the speed/solution-quality tradeoff by starting more greedy and becoming less greedy as search progresses.

## Baseline Validation Strategy
To explicitly demonstrate correctness (beyond “it runs”):
- **Path validity** checks:
  - path starts at `S` and ends at `G`
  - path never crosses obstacles
  - each step is 4-connected (Manhattan step of 1)
- **Optimality proof on small grids**:
  - compute true optimal shortest-path cost using BFS
  - assert A* path cost equals BFS cost
- These checks are executed automatically as part of `python main.py`.

## Experiment Design
All experiments are scripted and reproducible (fixed seeds). Each experiment uses **>=10 runs** to support statistical summaries.

### Experiment 1 — Baseline vs Enhanced
- Fixed input size (e.g., 40×40) with a specified obstacle probability.
- Compare A* vs the enhanced configuration.
- Metrics:
  - execution time (seconds)
  - nodes expanded
  - path length (cost)
- Statistics:
  - mean, standard deviation
  - 95% CI half-width

### Experiment 2 — Scaling Analysis
- Sizes: small/medium/large (e.g., 20, 40, 60).
- Compare algorithms across sizes.
- Adds an additional plot of **improvement % vs input size**.

### Experiment 3 — Parameter Sensitivity
- Sweep WA* weights (e.g., 1.0 to 2.5).
- Measures how time/nodes/path length trade off with greediness.
- Used to select a “best” weight (by mean time) and report tradeoffs.

### Experiment 4 — Ablation Study (Memoization)
- Compare WA* **without memoization** vs WA* **with memoization**.
- Same grids, seeds, and weight.
- Quantifies the effect of caching on time and node expansions.

Important: since Manhattan distance is extremely cheap, memoization can reduce the number of heuristic computations (`heuristic_evals`) but may not always reduce wall-clock time due to cache overhead. Reporting `heuristic_evals` makes the ablation result unambiguous.

## Key Observations (What To Look For)
- Enhanced search typically shows:
  - fewer nodes expanded (often the clearest improvement)
  - lower execution time
  - slightly higher average path length for higher weights
- Ablation should show:
  - memoization reduces execution time (usually modest but measurable)

## Challenges Faced / Pitfalls
- Random obstacle grids can be unsolvable; experiments include a bounded retry mechanism to ensure solvable instances.
- WA* does not guarantee optimality for $w>1$; the project reports `path_length` explicitly to show this tradeoff.
- On very small grids, timing can be noisy; statistics (stddev/CI) help communicate uncertainty.

## What Remains for Final Milestone
- Extend the report discussion with:
  - formal problem formulation (state space, actions, costs)
  - complexity discussion and empirical scaling explanation
  - deeper analysis on when WA* helps/hurts (obstacle density, map structure)
- Optionally add:
  - additional obstacle densities (e.g., 0.1, 0.2, 0.3)
  - more input sizes or more runs for tighter confidence intervals
