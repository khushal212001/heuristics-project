"""Experiment runner for baseline vs enhanced heuristic search.

Generates structured CSV data and matplotlib plots.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .baseline import AStarConfig, astar_search
from .enhanced import WeightedAStarConfig, weighted_astar_search
from .utils import Grid, SearchResult, ci95_half_width, ensure_dirs, generate_grid, mean, pct_improvement, project_root, stdev


@dataclass(frozen=True)
class GridSpec:
    """Grid generation specification."""

    size: int
    obstacle_prob: float


@dataclass(frozen=True)
class RunSpec:
    """A single run spec."""

    algorithm: str
    grid_spec: GridSpec
    run_id: Optional[int]
    seed: int
    weight: Optional[float] = None
    memoize_heuristic: Optional[bool] = None
    dynamic_weight: Optional[bool] = None
    weight_min: Optional[float] = None
    weight_decay_per_1000: Optional[float] = None
    tie_breaker: Optional[str] = None


def _solve_with_retry(
    algorithm: str,
    size: int,
    obstacle_prob: float,
    seed: int,
    weight: Optional[float],
    memoize_heuristic: Optional[bool],
    dynamic_weight: Optional[bool],
    weight_min: Optional[float],
    weight_decay_per_1000: Optional[float],
    tie_breaker: Optional[str],
    max_attempts: int = 50,
) -> Tuple[SearchResult, Dict[str, object]]:
    """Generate a solvable instance and solve it.

    To keep experiments comparable and avoid counting failures, we retry grid generation
    until a path exists (bounded by max_attempts).
    """

    rng = random.Random(seed)
    last_result: Optional[SearchResult] = None
    for attempt in range(max_attempts):
        grid, start, goal = generate_grid(size=size, obstacle_prob=obstacle_prob, rng=rng)

        if algorithm == "baseline_astar":
            result = astar_search(grid, start, goal, config=AStarConfig())
        elif algorithm in {
            "enhanced_weighted_astar",
            "enhanced_weighted_astar_dynamic",
            "enhanced_weighted_astar_nomemo",
        }:
            if weight is None:
                raise ValueError("weight required for weighted A* variants")

            cfg = WeightedAStarConfig(weight=weight)
            if tie_breaker is not None:
                cfg = replace(cfg, tie_breaker=tie_breaker)
            if memoize_heuristic is not None:
                cfg = replace(cfg, memoize_heuristic=memoize_heuristic)

            # Algorithm defaults for ablation.
            if algorithm == "enhanced_weighted_astar_nomemo":
                cfg = replace(cfg, memoize_heuristic=False)
            if algorithm == "enhanced_weighted_astar_dynamic":
                cfg = replace(cfg, dynamic_weight=True)

            if dynamic_weight is not None:
                cfg = replace(cfg, dynamic_weight=dynamic_weight)
            if weight_min is not None:
                cfg = replace(cfg, weight_min=weight_min)
            if weight_decay_per_1000 is not None:
                cfg = replace(cfg, weight_decay_per_1000=weight_decay_per_1000)

            result = weighted_astar_search(grid, start, goal, config=cfg)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        last_result = result
        if result.found:
            meta = {"attempt": attempt + 1}
            return result, meta

        # Re-seed slightly to vary grids but remain deterministic per (seed, attempt).
        rng.seed(seed + 1000 * (attempt + 1))

    # If unsolved, return last result and mark as failure.
    assert last_result is not None
    return last_result, {"attempt": max_attempts, "unsolved": True}


class ExperimentRunner:
    """Runs all milestone experiments and writes outputs."""

    def __init__(self, results_csv: Path):
        self.results_csv = results_csv
        self.rows: List[Dict[str, object]] = []
        self._run_id_counter = 0
        self.summary_rows: List[Dict[str, object]] = []
        ensure_dirs()

    def _record(self, row: Dict[str, object]) -> None:
        self.rows.append(row)

    def run_single(self, run: RunSpec) -> SearchResult:
        """Run one solver execution and record a CSV row."""

        run_id = run.run_id
        if run_id is None:
            run_id = self._run_id_counter
            self._run_id_counter += 1

        result, meta = _solve_with_retry(
            algorithm=run.algorithm,
            size=run.grid_spec.size,
            obstacle_prob=run.grid_spec.obstacle_prob,
            seed=run.seed,
            weight=run.weight,
            memoize_heuristic=run.memoize_heuristic,
            dynamic_weight=run.dynamic_weight,
            weight_min=run.weight_min,
            weight_decay_per_1000=run.weight_decay_per_1000,
            tie_breaker=run.tie_breaker,
        )

        row: Dict[str, object] = {
            # Required rubric fields
            "algorithm": run.algorithm,
            "input_size": run.grid_spec.size,
            "execution_time": result.execution_time_s,
            "nodes_expanded": result.nodes_expanded,
            "run_id": run_id,
            "path_length": result.path_cost,
            # Helpful extras
            "obstacle_prob": run.grid_spec.obstacle_prob,
            "found": int(result.found),
            "seed": run.seed,
            "heuristic_evals": result.heuristic_evals,
        }
        if run.weight is not None:
            row["weight"] = run.weight
        if run.memoize_heuristic is not None:
            row["memoize_heuristic"] = int(bool(run.memoize_heuristic))
        if run.dynamic_weight is not None:
            row["dynamic_weight"] = int(bool(run.dynamic_weight))
        if run.weight_min is not None:
            row["weight_min"] = float(run.weight_min)
        if run.weight_decay_per_1000 is not None:
            row["weight_decay_per_1000"] = float(run.weight_decay_per_1000)
        if run.tie_breaker is not None:
            row["tie_breaker"] = run.tie_breaker
        row.update(meta)

        self._record(row)
        return result

    def _metric_values(self, rows: List[Dict[str, object]], key: str) -> List[float]:
        """Extract a numeric metric column from rows."""

        return [float(r[key]) for r in rows]

    def _metric_stats(self, rows: List[Dict[str, object]], key: str) -> Dict[str, float]:
        """Compute mean, stddev, and CI95 half-width for a metric."""

        vals = self._metric_values(rows, key)
        return {
            "mean": mean(vals),
            "std": stdev(vals),
            "ci95": ci95_half_width(vals),
            "n": float(len(vals)),
        }

    def run_experiment_1_baseline_vs_enhanced(
        self,
        grid_spec: GridSpec,
        runs: int = 10,
        enhanced_weight: float = 2.2,
        use_dynamic_weight: bool = True,
        seed0: int = 1234,
    ) -> Dict[str, float]:
        """Experiment 1: Baseline vs Enhanced comparison on a fixed input size."""

        baseline_times: List[float] = []
        enhanced_times: List[float] = []
        baseline_nodes: List[float] = []
        enhanced_nodes: List[float] = []

        enhanced_algo = "enhanced_weighted_astar_dynamic" if use_dynamic_weight else "enhanced_weighted_astar"

        for i in range(runs):
            seed = seed0 + i
            b = self.run_single(
                RunSpec("baseline_astar", grid_spec=grid_spec, run_id=None, seed=seed)
            )
            e = self.run_single(
                RunSpec(
                    enhanced_algo,
                    grid_spec=grid_spec,
                    run_id=None,
                    seed=seed,
                    weight=enhanced_weight,
                    memoize_heuristic=True,
                    dynamic_weight=use_dynamic_weight,
                    weight_min=1.0,
                    weight_decay_per_1000=0.12,
                    tie_breaker="h",
                )
            )
            baseline_times.append(b.execution_time_s)
            enhanced_times.append(e.execution_time_s)
            baseline_nodes.append(float(b.nodes_expanded))
            enhanced_nodes.append(float(e.nodes_expanded))

        baseline_time_mean = mean(baseline_times)
        enhanced_time_mean = mean(enhanced_times)
        baseline_nodes_mean = mean(baseline_nodes)
        enhanced_nodes_mean = mean(enhanced_nodes)

        return {
            "baseline_time_mean": baseline_time_mean,
            "enhanced_time_mean": enhanced_time_mean,
            "baseline_time_std": stdev(baseline_times),
            "enhanced_time_std": stdev(enhanced_times),
            "baseline_time_ci95": ci95_half_width(baseline_times),
            "enhanced_time_ci95": ci95_half_width(enhanced_times),
            "baseline_nodes_mean": baseline_nodes_mean,
            "enhanced_nodes_mean": enhanced_nodes_mean,
            "baseline_nodes_std": stdev(baseline_nodes),
            "enhanced_nodes_std": stdev(enhanced_nodes),
            "nodes_improvement_pct": pct_improvement(baseline_nodes_mean, enhanced_nodes_mean),
            "time_improvement_pct": pct_improvement(baseline_time_mean, enhanced_time_mean),
        }

    def run_experiment_2_scaling(
        self,
        sizes: Sequence[int],
        obstacle_prob: float,
        runs: int = 10,
        enhanced_weight: float = 2.2,
        use_dynamic_weight: bool = True,
        seed0: int = 2000,
    ) -> None:
        """Experiment 2: Scaling analysis across sizes for both algorithms."""

        enhanced_algo = "enhanced_weighted_astar_dynamic" if use_dynamic_weight else "enhanced_weighted_astar"

        for size in sizes:
            grid_spec = GridSpec(size=size, obstacle_prob=obstacle_prob)
            for i in range(runs):
                seed = seed0 + 100 * size + i
                self.run_single(RunSpec("baseline_astar", grid_spec=grid_spec, run_id=None, seed=seed))
                self.run_single(
                    RunSpec(
                        enhanced_algo,
                        grid_spec=grid_spec,
                        run_id=None,
                        seed=seed,
                        weight=enhanced_weight,
                        memoize_heuristic=True,
                        dynamic_weight=use_dynamic_weight,
                        weight_min=1.0,
                        weight_decay_per_1000=0.12,
                        tie_breaker="h",
                    )
                )

    def run_experiment_3_parameter_sensitivity(
        self,
        grid_spec: GridSpec,
        weights: Sequence[float],
        runs: int = 10,
        seed0: int = 9000,
    ) -> None:
        """Experiment 3: Sensitivity to WA* weight parameter."""

        for w in weights:
            for i in range(runs):
                seed = seed0 + int(1000 * w) + i
                self.run_single(
                    RunSpec(
                        "enhanced_weighted_astar",
                        grid_spec=grid_spec,
                        run_id=None,
                        seed=seed,
                        weight=float(w),
                        memoize_heuristic=True,
                        dynamic_weight=False,
                        tie_breaker="h",
                    )
                )

    def run_experiment_4_ablation_memoization(
        self,
        grid_spec: GridSpec,
        runs: int = 10,
        weight: float = 1.6,
        seed0: int = 12000,
    ) -> None:
        """Experiment 4 (new): ablation study for heuristic memoization."""

        for i in range(runs):
            seed = seed0 + i
            self.run_single(
                RunSpec(
                    "enhanced_weighted_astar_nomemo",
                    grid_spec=grid_spec,
                    run_id=None,
                    seed=seed,
                    weight=weight,
                    memoize_heuristic=False,
                    dynamic_weight=False,
                    tie_breaker="h",
                )
            )
            self.run_single(
                RunSpec(
                    "enhanced_weighted_astar",
                    grid_spec=grid_spec,
                    run_id=None,
                    seed=seed,
                    weight=weight,
                    memoize_heuristic=True,
                    dynamic_weight=False,
                    tie_breaker="h",
                )
            )

    def write_csv(self) -> None:
        """Write all accumulated rows to CSV."""

        from .utils import write_results_csv

        write_results_csv(self.results_csv, self.rows)

    def write_summary_csv(self, summary_csv: Path) -> None:
        """Write per-group summary statistics (mean/std/CI and improvement %)."""

        from .utils import write_results_csv

        write_results_csv(
            summary_csv,
            self.summary_rows,
            required_fields=["algorithm", "input_size", "n"],
        )

    def build_summaries(self, *, baseline_algo: str = "baseline_astar") -> None:
        """Build summary_rows for (algorithm,input_size) groups across key metrics."""

        self.summary_rows = []
        sizes = sorted({int(r["input_size"]) for r in self.rows})
        algos = sorted({str(r["algorithm"]) for r in self.rows})

        for size in sizes:
            baseline_rows = [r for r in self.rows if str(r["algorithm"]) == baseline_algo and int(r["input_size"]) == size]
            if not baseline_rows:
                continue
            b_time = self._metric_stats(baseline_rows, "execution_time")
            b_nodes = self._metric_stats(baseline_rows, "nodes_expanded")
            b_len = self._metric_stats(baseline_rows, "path_length")

            for algo in algos:
                group = [r for r in self.rows if str(r["algorithm"]) == algo and int(r["input_size"]) == size]
                if not group:
                    continue

                t = self._metric_stats(group, "execution_time")
                n = self._metric_stats(group, "nodes_expanded")
                l = self._metric_stats(group, "path_length")

                self.summary_rows.append(
                    {
                        "algorithm": algo,
                        "input_size": size,
                        "execution_time_mean": t["mean"],
                        "execution_time_std": t["std"],
                        "execution_time_ci95": t["ci95"],
                        "nodes_expanded_mean": n["mean"],
                        "nodes_expanded_std": n["std"],
                        "nodes_expanded_ci95": n["ci95"],
                        "path_length_mean": l["mean"],
                        "path_length_std": l["std"],
                        "path_length_ci95": l["ci95"],
                        "n": int(t["n"]),
                        "time_improvement_pct": pct_improvement(b_time["mean"], t["mean"]),
                        "nodes_improvement_pct": pct_improvement(b_nodes["mean"], n["mean"]),
                        "path_length_delta": l["mean"] - b_len["mean"],
                    }
                )

    def _filter_rows(self, *, algorithm: Optional[str] = None, input_size: Optional[int] = None) -> List[Dict[str, object]]:
        """Filter raw rows by algorithm and/or input size."""

        rows = self.rows
        if algorithm is not None:
            rows = [r for r in rows if r.get("algorithm") == algorithm]
        if input_size is not None:
            rows = [r for r in rows if int(r.get("input_size")) == input_size]
        return rows

    def _group_mean(self, rows: List[Dict[str, object]], key: str) -> float:
        """Mean for a metric within a row group."""

        return mean([float(r[key]) for r in rows])

    def _group_std(self, rows: List[Dict[str, object]], key: str) -> float:
        """Stddev for a metric within a row group."""

        return stdev([float(r[key]) for r in rows])

    def plot_experiment_1_bar(self, grid_size: int, out_path: Path) -> None:
        """Bar chart: baseline vs enhanced (avg time and avg nodes)."""

        ensure_dirs()
        b = self._filter_rows(algorithm="baseline_astar", input_size=grid_size)
        # Prefer the strongest enhanced variant if present.
        e = self._filter_rows(algorithm="enhanced_weighted_astar_dynamic", input_size=grid_size)
        if not e:
            e = self._filter_rows(algorithm="enhanced_weighted_astar", input_size=grid_size)

        if not b or not e:
            raise ValueError("Not enough rows to plot experiment 1")

        metrics = [
            (
                "Execution Time (s)",
                self._group_mean(b, "execution_time"),
                self._group_std(b, "execution_time"),
                self._group_mean(e, "execution_time"),
                self._group_std(e, "execution_time"),
            ),
            (
                "Nodes Expanded",
                self._group_mean(b, "nodes_expanded"),
                self._group_std(b, "nodes_expanded"),
                self._group_mean(e, "nodes_expanded"),
                self._group_std(e, "nodes_expanded"),
            ),
        ]

        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(metrics))
        baseline_vals = [m[1] for m in metrics]
        baseline_err = [m[2] for m in metrics]
        enhanced_vals = [m[3] for m in metrics]
        enhanced_err = [m[4] for m in metrics]

        width = 0.35
        ax.bar([i - width / 2 for i in x], baseline_vals, width=width, label="Baseline A*", yerr=baseline_err, capsize=4)
        ax.bar([i + width / 2 for i in x], enhanced_vals, width=width, label="Enhanced", yerr=enhanced_err, capsize=4)

        ax.set_xticks(list(x))
        ax.set_xticklabels([m[0] for m in metrics], rotation=0)
        ax.set_title(f"Baseline vs Enhanced (Grid {grid_size}x{grid_size})")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    def plot_experiment_2_scaling_lines(self, sizes: Sequence[int], out_path: Path) -> None:
        """Line chart: performance vs input size for both algorithms."""

        ensure_dirs()
        fig, ax = plt.subplots(figsize=(8, 4))

        for algorithm, label in [("baseline_astar", "Baseline A*"), ("enhanced_weighted_astar_dynamic", "Enhanced (Dynamic WA*)")]:
            times = []
            times_std = []
            for size in sizes:
                rows = [r for r in self.rows if r.get("algorithm") == algorithm and int(r.get("input_size")) == size]
                times.append(self._group_mean(rows, "execution_time"))
                times_std.append(self._group_std(rows, "execution_time"))

            ax.errorbar(list(sizes), times, yerr=times_std, marker="o", capsize=3, label=f"{label} time (s)")

        ax.set_xlabel("Grid Size (N for NxN)")
        ax.set_ylabel("Average Execution Time (s)")
        ax.set_title("Scaling: Execution Time vs Grid Size")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        # Separate plot for nodes expanded to keep axes readable.
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        for algorithm, label in [("baseline_astar", "Baseline A*"), ("enhanced_weighted_astar_dynamic", "Enhanced (Dynamic WA*)")]:
            nodes = []
            nodes_std = []
            for size in sizes:
                rows = [r for r in self.rows if r.get("algorithm") == algorithm and int(r.get("input_size")) == size]
                nodes.append(self._group_mean(rows, "nodes_expanded"))
                nodes_std.append(self._group_std(rows, "nodes_expanded"))
            ax2.errorbar(list(sizes), nodes, yerr=nodes_std, marker="o", capsize=3, label=f"{label} nodes")

        ax2.set_xlabel("Grid Size (N for NxN)")
        ax2.set_ylabel("Average Nodes Expanded")
        ax2.set_title("Scaling: Nodes Expanded vs Grid Size")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        fig2.tight_layout()
        out_nodes_path = out_path.parent / (out_path.stem + "_nodes.png")
        fig2.savefig(out_nodes_path, dpi=200)
        plt.close(fig2)

    def plot_experiment_3_sensitivity(self, grid_size: int, out_path: Path) -> None:
        """Parameter sensitivity plot for WA* weight vs performance."""

        ensure_dirs()
        rows = [r for r in self.rows if r.get("algorithm") == "enhanced_weighted_astar" and int(r.get("input_size")) == grid_size]
        if not rows:
            raise ValueError("No rows for sensitivity plot")

        weights = sorted({float(r.get("weight", 0.0)) for r in rows if "weight" in r})
        times = []
        times_std = []
        nodes = []
        nodes_std = []
        lengths = []
        lengths_std = []
        for w in weights:
            wr = [r for r in rows if float(r.get("weight")) == w]
            times.append(self._group_mean(wr, "execution_time"))
            nodes.append(self._group_mean(wr, "nodes_expanded"))
            lengths.append(self._group_mean(wr, "path_length"))
            times_std.append(self._group_std(wr, "execution_time"))
            nodes_std.append(self._group_std(wr, "nodes_expanded"))
            lengths_std.append(self._group_std(wr, "path_length"))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.errorbar(weights, times, yerr=times_std, marker="o", capsize=3, label="Avg time (s)")
        ax.set_xlabel("Weight w")
        ax.set_ylabel("Avg time (s)")
        ax.set_title(f"WA* Sensitivity (Grid {grid_size}x{grid_size})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.errorbar(weights, nodes, yerr=nodes_std, marker="o", capsize=3, label="Avg nodes expanded")
        ax2.errorbar(weights, lengths, yerr=lengths_std, marker="o", capsize=3, label="Avg path length")
        ax2.set_xlabel("Weight w")
        ax2.set_ylabel("Average")
        ax2.set_title(f"WA* Sensitivity: Nodes & Path Length (Grid {grid_size}x{grid_size})")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        fig2.tight_layout()
        out2 = out_path.parent / (out_path.stem + "_nodes_path.png")
        fig2.savefig(out2, dpi=200)
        plt.close(fig2)

    def plot_improvement_vs_size(self, sizes: Sequence[int], out_path: Path) -> None:
        """New plot: improvement percentage vs input size (time and nodes)."""

        ensure_dirs()
        fig, ax = plt.subplots(figsize=(8, 4))

        time_impr: List[float] = []
        nodes_impr: List[float] = []

        for size in sizes:
            b = [r for r in self.rows if r.get("algorithm") == "baseline_astar" and int(r.get("input_size")) == size]
            e = [r for r in self.rows if r.get("algorithm") == "enhanced_weighted_astar_dynamic" and int(r.get("input_size")) == size]
            bt = self._group_mean(b, "execution_time")
            et = self._group_mean(e, "execution_time")
            bn = self._group_mean(b, "nodes_expanded")
            en = self._group_mean(e, "nodes_expanded")
            time_impr.append(pct_improvement(bt, et))
            nodes_impr.append(pct_improvement(bn, en))

        ax.plot(list(sizes), time_impr, marker="o", label="Time improvement (%)")
        ax.plot(list(sizes), nodes_impr, marker="o", label="Nodes improvement (%)")
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel("Grid Size (N for NxN)")
        ax.set_ylabel("Improvement vs Baseline (%)")
        ax.set_title("Improvement vs Input Size")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    def best_weight_by_time(self, grid_size: int) -> Dict[str, float]:
        """Pick the WA* weight with the lowest mean time for a given grid size."""

        rows = [
            r
            for r in self.rows
            if r.get("algorithm") == "enhanced_weighted_astar"
            and int(r.get("input_size")) == grid_size
            and "weight" in r
        ]
        if not rows:
            return {}

        weights = sorted({float(r["weight"]) for r in rows})
        best_w = None
        best_time = None
        best_nodes = None
        best_len = None

        for w in weights:
            wr = [r for r in rows if float(r["weight"]) == w]
            t = self._group_mean(wr, "execution_time")
            if best_time is None or t < best_time:
                best_time = t
                best_w = w
                best_nodes = self._group_mean(wr, "nodes_expanded")
                best_len = self._group_mean(wr, "path_length")

        assert best_w is not None and best_time is not None and best_nodes is not None and best_len is not None
        return {
            "best_weight": float(best_w),
            "mean_time": float(best_time),
            "mean_nodes": float(best_nodes),
            "mean_path_length": float(best_len),
        }

    def wa_weight_stats(self, grid_size: int, weight: float) -> Dict[str, float]:
        """Return mean time/nodes/path_length for WA* at a specific weight."""

        rows = [
            r
            for r in self.rows
            if r.get("algorithm") == "enhanced_weighted_astar"
            and int(r.get("input_size")) == grid_size
            and float(r.get("weight", -1.0)) == float(weight)
        ]
        if not rows:
            return {}

        return {
            "weight": float(weight),
            "mean_time": float(self._group_mean(rows, "execution_time")),
            "mean_nodes": float(self._group_mean(rows, "nodes_expanded")),
            "mean_path_length": float(self._group_mean(rows, "path_length")),
        }

    def avg_improvement_across_sizes(self, sizes: Sequence[int]) -> Dict[str, float]:
        """Average improvement (time/nodes) of enhanced dynamic WA* vs baseline across sizes."""

        time_imprs: List[float] = []
        nodes_imprs: List[float] = []
        for size in sizes:
            b = [r for r in self.rows if r.get("algorithm") == "baseline_astar" and int(r.get("input_size")) == size]
            e = [r for r in self.rows if r.get("algorithm") == "enhanced_weighted_astar_dynamic" and int(r.get("input_size")) == size]
            if not b or not e:
                continue
            bt = self._group_mean(b, "execution_time")
            et = self._group_mean(e, "execution_time")
            bn = self._group_mean(b, "nodes_expanded")
            en = self._group_mean(e, "nodes_expanded")
            time_imprs.append(pct_improvement(bt, et))
            nodes_imprs.append(pct_improvement(bn, en))

        return {
            "avg_time_improvement_pct": mean(time_imprs),
            "avg_nodes_improvement_pct": mean(nodes_imprs),
        }

    def ablation_memoization_summary(self, grid_size: int) -> Dict[str, float]:
        """Compare WA* with memoization vs without, on a given grid size."""

        memo = [r for r in self.rows if r.get("algorithm") == "enhanced_weighted_astar" and int(r.get("input_size")) == grid_size]
        nomemo = [r for r in self.rows if r.get("algorithm") == "enhanced_weighted_astar_nomemo" and int(r.get("input_size")) == grid_size]
        if not memo or not nomemo:
            return {}

        memo_time = self._group_mean(memo, "execution_time")
        nomemo_time = self._group_mean(nomemo, "execution_time")
        memo_nodes = self._group_mean(memo, "nodes_expanded")
        nomemo_nodes = self._group_mean(nomemo, "nodes_expanded")

        memo_he = self._group_mean(memo, "heuristic_evals")
        nomemo_he = self._group_mean(nomemo, "heuristic_evals")

        return {
            "memo_time_mean": float(memo_time),
            "nomemo_time_mean": float(nomemo_time),
            "time_improvement_pct": pct_improvement(nomemo_time, memo_time),
            "memo_nodes_mean": float(memo_nodes),
            "nomemo_nodes_mean": float(nomemo_nodes),
            "nodes_improvement_pct": pct_improvement(nomemo_nodes, memo_nodes),
            "memo_heuristic_evals_mean": float(memo_he),
            "nomemo_heuristic_evals_mean": float(nomemo_he),
            "heuristic_evals_reduction_pct": pct_improvement(nomemo_he, memo_he),
        }


def default_experiments(results_csv: Optional[Path] = None) -> ExperimentRunner:
    """Create a runner with default paths."""

    root = project_root()
    if results_csv is None:
        results_csv = root / "data" / "results.csv"
    return ExperimentRunner(results_csv=results_csv)
