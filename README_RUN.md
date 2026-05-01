# Full Reproduction Guide — CS 57200 Grid Pathfinding Project

Follow these steps **in order** to reproduce every result, plot, report, and presentation from scratch.

---

## Step 0 — Prerequisites

- **Python 3.9 or higher** (`python3 --version` to check)
- **macOS / Linux / Windows** (all commands use `python3`; on Windows replace with `python`)
- **Git** (to clone the repo)

---

## Step 1 — Clone and Install

```bash
git clone <repo-url>
cd heuristics_code_khushal

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt
```

**Dependencies installed:**
| Package | Version | Used for |
|---------|---------|----------|
| matplotlib | ≥3.8 | Experiment plots |
| pandas | ≥2.0 | CSV aggregation in analysis.py |
| python-pptx | ≥0.6.23 | Presentation generation |
| python-docx | latest | Report DOCX export |

---

## Step 2 — Run Correctness Tests

Verifies baseline A\* correctness and BFS-optimality on small grids before running experiments.

```bash
python3 -m pytest tests/test_cases.py -v
```

**Expected output:** All tests pass. Zero failures.

---

## Step 3 — Run All 7 Experiments

Runs the full final rubric suite (n=30 grids per condition, 1,110 total trials).

```bash
python3 main.py --final
```

**Optional flags:**
```bash
python3 main.py --final --runs 30 --seed 42   # set trial count and RNG seed
python3 main.py --experiment scaling           # run one experiment only
python3 main.py --experiment density
python3 main.py --experiment heuristic
```

**Outputs written to `final_results/`:**
```
final_results/
├── csv/
│   ├── final_runs.csv       # raw per-trial rows (nodes, path_len, time, algo)
│   └── final_results.csv    # aggregated means + CI95 per experiment condition
└── plots/
    ├── exp1_baseline_vs_enhanced.png
    ├── exp2_improvement_vs_size.png
    ├── exp2_scaling_nodes.png
    ├── exp2_scaling_time.png
    ├── exp3_weight_sensitivity_nodes_path.png
    ├── exp3_weight_sensitivity_time.png
    ├── exp4_ablation_memoization.png
    ├── exp5_density_nodes.png
    ├── exp5_density_time.png
    ├── exp6_heuristic_comparison.png
    └── exp7_tiebreaking.png
```

---

## Step 4 — Run Analytics Pipeline

Reads the CSVs produced in Step 3, computes CI95, and writes a human-readable report.

```bash
python3 analysis.py
```

**Outputs written to `final_results/logs/`:**
```
final_results/logs/
├── analysis_report.txt   # per-experiment narrative with CI95 values
└── summary.txt           # one-line KPI per experiment (quick overview)
```

---

## Step 5 — Rebuild the Presentation (optional)

Regenerates the 26-slide PPTX with all 11 experiment plots embedded.

```bash
python3 deliverables/make_amber_style_ppt.py
```

**Output:** `deliverables/Khushal_Final_Presentation.pptx`

Open with:
```bash
open deliverables/Khushal_Final_Presentation.pptx   # macOS
```

---

## Step 6 — Rebuild the DOCX Report (optional)

```bash
python3 deliverables/markdown_to_docx.py
```

**Output:** `deliverables/Khushal_Project_Report.docx`

---

## Summary: One-liner Full Reproduction

```bash
pip install -r requirements.txt && \
python3 -m pytest tests/test_cases.py -v && \
python3 main.py --final && \
python3 analysis.py
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: pandas` | Run `pip install pandas>=2.0` |
| `ModuleNotFoundError: pptx` | Run `pip install python-pptx` |
| `No such file: final_results/csv/...` | Run Step 3 first (`main.py --final`) |
| `IndentationError` in analysis.py | Ensure you pulled latest commit |
| Tests fail on optimality check | Confirm Python ≥ 3.9 is active |
