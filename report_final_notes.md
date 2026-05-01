# CS572 Final Project — Report Notes (Auto-generated)

## Enhancements Implemented (>=3)
1) Tie-breaking: on f(n) ties, prefer lower h(n) (toggle: use_tiebreaking).
2) Multi-heuristic system: Manhattan, Euclidean, Hybrid (toggle: heuristic; hybrid uses alpha).
3) Density-adaptive strategy: switch between A* and WA* / increase WA* weight based on obstacle density (toggle: use_density_adaptive).

## Experiments (>=5, n>=30 per condition)
- Exp1: Baseline vs Enhanced (40x40, p=0.22)
- Exp2: Scaling (20,40,60,80,100)
- Exp3: Weight sensitivity (w ∈ {1.0,1.2,1.4,1.6,2.0,2.5})
- Exp4: Memoization ablation (on/off)
- Exp5: Obstacle density analysis (p ∈ {0.10,0.20,0.30,0.40})
- Exp6: Heuristic comparison (Manhattan vs Euclidean vs Hybrid)
- Exp7: Tie-breaking analysis (on/off)

## Key Findings (populate from final_results/csv/final_results.csv)
- Best configuration (Exp1, normalized time+nodes): **enhanced_combo**
  - time_mean: 0.000610 s
  - nodes_mean: 89.3
- See final_results/logs/summary.txt for the automated trade-off summary.

## Reproducibility
- Deterministic seeds are used for all runs.
- Run the full suite with: `python3 main.py --final`.
- Run a single experiment with: `python3 main.py --experiment exp2` (for example).

## Limitations
- Runtime measurements on very small instances can be noisy; nodes expanded is often a more stable indicator.
- WA* can return longer paths for w>1; this is measured via path_length.

## Future Work
- Add additional map models (maze-like, clustered obstacles) and obstacle densities.
- Add more heuristics or landmark-based heuristics for stronger improvements.
