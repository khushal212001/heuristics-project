# CS572 Heuristic Problem Solving — Milestone 2 (Production-Ready)

## Overview
This project is a complete, runnable heuristic problem solving system using **2D grid pathfinding with obstacles**.

It includes:
- **Baseline**: standard A* search
- **Enhancement**: Weighted A* (WA*) + heuristic memoization + dynamic weight schedule
- **Automation**: one command runs tests + experiments + CSV collection + plots

The project is organized to meet the Milestone 2 grading rubric end-to-end with measurable improvement and reproducible experiments.

## Milestone 2 Rubric Coverage (Checklist)
- Baseline implementation: A* in [src/baseline.py](src/baseline.py)
- Enhancement: Weighted A* + caching in [src/enhanced.py](src/enhanced.py)
- Minimum test cases (>=3): [tests/test_cases.py](tests/test_cases.py)
- Experiments (>=3), automated, >=10 runs each, mean/stddev/CI: [src/experiment_runner.py](src/experiment_runner.py)
- Data collection (raw + summary CSV): [data/results.csv](data/results.csv) and [data/summary.csv](data/summary.csv) (generated)
- Visualization (matplotlib, saved PNGs): [results/plots](results/plots) (generated)
- One-command run: [main.py](main.py)

## Problem Definition
- **State**: agent position `(row, col)` on an `N x N` grid
- **Actions**: move in 4 directions (up/down/left/right)
- **Cost**: unit cost per move
- **Constraints**: blocked cells cannot be traversed
- **Objective**: find a path from `S` to `G`

## Algorithms
### Baseline: A* Search
- Uses the standard evaluation function:  
  $f(n) = g(n) + h(n)$
- Heuristic: **Manhattan distance** (admissible for 4-connected grids)

### Enhancement: Weighted A* (WA*) + Heuristic Memoization
- Uses:  
  $f(n) = g(n) + w \cdot h(n)$ where $w \ge 1$
- For `w > 1`, WA* typically **expands fewer nodes** and runs faster, giving a measurable improvement in:
  - execution time
  - nodes expanded
- The implementation also includes **heuristic memoization per run** (cache of $h(n)$ values) to reduce repeated heuristic computations.

Note: WA* may return a slightly longer path than optimal; the project reports `path_length` in the CSV so this tradeoff is visible.

### Additional Enhancement: Dynamic Weight Schedule
Beyond basic WA*, this project includes an optional dynamic weight schedule where the effective $w$ decreases as the search progresses.
This can improve the speed/solution-quality tradeoff by starting more greedy and becoming less greedy over time.

## Project Structure
```
project_root/
├── src/
│   ├── baseline.py
│   ├── enhanced.py
│   ├── utils.py
│   ├── experiment_runner.py
├── data/
│   └── results.csv
├── results/
│   └── plots/
├── tests/
│   └── test_cases.py
├── README.md
├── requirements.txt
└── main.py
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your system does not provide `python` (common on macOS), use `python3` for the commands above.

## Run Everything (Single Command)
```bash
python main.py
```
If `python` is not available:
```bash
python3 main.py
```
This will:
1. Run 3 correctness test cases (small, medium, edge)
2. Run a baseline validation suite (A* optimality vs BFS on small grids)
3. Run all experiments (each with >= 10 runs)
4. Write `data/results.csv` and `data/summary.csv`
5. Save plots to `results/plots/`

## Experiments
### Experiment 1 — Baseline vs Enhanced
- Runs both algorithms on the same medium grid configuration
- Reports average execution time and average nodes expanded

### Experiment 2 — Scaling Analysis
- Runs both algorithms on increasing grid sizes: `20`, `40`, `60`
- Plots time vs size and nodes vs size

### Experiment 3 — Parameter Sensitivity (Weight)
- Varies `w` in Weighted A* over: `1.0, 1.2, 1.4, 1.6, 2.0, 2.5`
- Plots how performance changes with `w`

Implementation detail: experiment runs generate random grids with fixed seeds; a bounded retry mechanism ensures each instance is solvable so timing/node averages are meaningful.

### Experiment 4 — Ablation Study (Memoization)
- Compares WA* **without** heuristic memoization vs WA* **with** memoization
- Quantifies the speed impact of caching while holding the rest constant

## Outputs
After `python main.py` completes:
- Raw CSV results (per run): `data/results.csv`
- Summary CSV (mean/stddev/CI + improvement %): `data/summary.csv`
- Plots:
  - `results/plots/exp1_baseline_vs_enhanced.png`
  - `results/plots/exp2_scaling_time.png` and `results/plots/exp2_scaling_time_nodes.png`
  - `results/plots/exp2_improvement_vs_size.png`
  - `results/plots/exp3_weight_sensitivity.png` and `results/plots/exp3_weight_sensitivity_nodes_path.png`

## Data Format (CSV)
The raw per-run CSV (`data/results.csv`) contains at least the rubric-required columns:
- `algorithm`
- `input_size`
- `execution_time`
- `nodes_expanded`
- `path_length`
- `run_id` (globally unique per run)

It also includes helpful extras for analysis:
- `found`, `obstacle_prob`, `seed`, `attempt`, and WA* configuration fields
- `heuristic_evals`: number of heuristic computations (useful for the memoization ablation)

## Result Summary (Expected)
You should see the enhanced algorithm (WA*) achieve measurable improvement on average:
- lower average nodes expanded
- lower average execution time

The exact numbers will vary slightly depending on random grids, but seeds are fixed so your runs are reproducible.

## Key Findings (What This Project Demonstrates)
- **Correctness**: A* returns valid paths and matches BFS optimal cost on small grids.
- **Efficiency gains**: enhanced variants typically reduce nodes expanded and execution time.
- **Tradeoffs**: higher WA* weights can improve speed but may increase path length; this is quantified in the sensitivity experiment.
- **Ablation**: memoization measurably reduces `heuristic_evals`; whether it reduces wall-clock time depends on heuristic cost vs cache overhead.

## Performance Improvements (How To Read)
- Check console output from `python main.py` for improvement percentages.
- Use `data/summary.csv` to compare mean/stddev/CI across sizes and algorithms.
- See `results/plots/exp2_improvement_vs_size.png` for improvement % trends as problem size increases.

## What Was Built (So Far)
- Implemented A* baseline and WA* enhanced solver with consistent metrics reporting.
- Added 3 correctness tests (small, medium-with-obstacles, and edge case start==goal).
- Built an experiment runner that executes >=10 runs per experiment, computes mean/stddev/CI, and writes both raw + summary CSVs.
- Added publication-quality plots with error bars and an improvement-vs-size plot.
- Added an ablation experiment to isolate the impact of heuristic memoization.
- Verified the end-to-end pipeline locally via `python3 main.py` on macOS.
