# How to run

## Prerequisites
- Python 3.9+

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Milestone pipeline (default)
```bash
python3 main.py
```

This runs tests + validation + milestone experiments and writes:
- `data/results.csv`, `data/summary.csv`
- `results/plots/*.png`

## Run Final Project suite
Runs the full final rubric suite (>=5 experiments, n>=30 each) and writes outputs under `final_results/`.

```bash
python3 main.py --final
```

Run a single final experiment (examples):
```bash
python3 main.py --experiment scaling
python3 main.py --experiment density
python3 main.py --experiment heuristic
```

Optional reproducibility controls:
```bash
python3 main.py --final --runs 30 --seed 1337
```

## Final outputs
- Raw per-run data: `final_results/csv/final_runs.csv`
- Summary stats (mean/std/CI95 + improvements): `final_results/csv/final_results.csv`
- Plots: `final_results/plots/*.png`
- Auto summary: `final_results/logs/summary.txt`
- Report support notes: `report_final_notes.md`
