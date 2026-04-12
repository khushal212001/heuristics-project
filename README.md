# How to run

## Prerequisites
- macOS/Linux/Windows
- Python 3.9+ (use `python3` on macOS)

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (one command)
```bash
python3 main.py
```

This will automatically:
- run correctness tests
- run baseline validation (A* vs BFS on small grids)
- run all experiments (>=10 runs each)
- write CSV outputs to `data/results.csv` and `data/summary.csv`
- save plots to `results/plots/`

## Outputs
- Raw per-run data: `data/results.csv`
- Summary stats (mean/std/CI95 + improvements): `data/summary.csv`
- Plots: `results/plots/*.png`
