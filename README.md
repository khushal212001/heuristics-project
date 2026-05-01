# Grid Pathfinding Optimization — CS 57200 Final Project

**Khushal Garg | CS 57200 | Track B: Heuristic Search**

This project implements and evaluates heuristic search optimizations for grid-based pathfinding. It compares a **baseline A\*** solver against an **enhanced solver** combining Weighted A\* (WA\*), memoization, and a density-adaptive heuristic.

## Project Summary

| Item | Detail |
|------|--------|
| Algorithms | Baseline A\*, Weighted A\* (w=1.5), Adaptive Heuristic |
| Enhancements | Memoization, WA\*, Density-adaptive h'(n), Tie-breaking |
| Experiments | 7 experiments, n=30 trials each, 1,110 total runs |
| Key result | ~40% node reduction; path suboptimality <5% |
| Language | Python 3.9+ |

## Repository Structure

```
.
├── src/
│   ├── baseline.py          # Standard A* implementation
│   ├── enhanced.py          # WA* + memoization + tie-breaking
│   ├── config.py            # Config dataclass with all toggles
│   ├── validation.py        # BFS optimality checker
│   ├── algorithms/
│   │   └── solver.py        # Density-adaptive policy + dispatch
│   ├── heuristics/          # Manhattan, Euclidean, Hybrid, Adaptive
│   ├── experiments/
│   │   └── final_runner.py  # 7 experiment definitions
│   └── utils/               # Grid, stats, and I/O helpers
├── tests/
│   └── test_cases.py        # Correctness and regression tests
├── main.py                  # Entry point (--final flag)
├── analysis.py              # Analytics pipeline → CI95 + plots
├── final_results/
│   ├── csv/                 # Raw runs + aggregated results
│   ├── plots/               # 11 experiment plots (PNG)
│   └── logs/                # analysis_report.txt + summary.txt
├── deliverables/
│   ├── Khushal_Final_Presentation.pptx
│   ├── Khushal_Project_Report.docx
│   └── make_amber_style_ppt.py
├── requirements.txt
├── README.md                # This file
└── README_RUN.md            # Step-by-step reproduction guide
```

## Quick Start

```bash
pip install -r requirements.txt
python3 main.py --final     # Run all 7 experiments
python3 analysis.py         # Generate analytics + plots
```

See [README_RUN.md](README_RUN.md) for the full reproduction guide.
