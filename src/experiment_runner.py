"""Experiment runner for baseline vs enhanced heuristic search.

This module is imported by main.py and is responsible for:
- running experiments (>=10 runs each)
- collecting raw per-run metrics into data/results.csv
- computing summary statistics into data/summary.csv
- generating plots under results/plots/
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .baseline import AStarConfig, astar_search
from .enhanced import WeightedAStarConfig, weighted_astar_search
from .utils import SearchResult, ci95_half_width, ensure_dirs, generate_grid, mean, pct_improvement, project_root, stdev


@dataclass(frozen=True)
class GridSpec:
	"""Grid generation specification."""

	size: int
	obstacle_prob: float


@dataclass(frozen=True)
class RunSpec:
	"""A single solver run specification."""

	algorithm: str
	grid_spec: GridSpec
	run_id: Optional[int]
	seed: int
	weight: Optional[float] = None
	experiment: str = "unspecified"
	memoize_heuristic: Optional[bool] = None
	dynamic_weight: Optional[bool] = None
	weight_min: Optional[float] = None
	weight_decay_per_1000: Optional[float] = None
	tie_breaker: Optional[str] = None


def _solve_with_retry(
	*,
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
) -> tuple[SearchResult, Dict[str, object]]:
	"""Generate a solvable instance and solve it.

	Random obstacle grids can be unsolvable; to keep experimental datasets meaningful,
	we retry generation (bounded) until a path exists.
	"""

	rng = random.Random(seed)
	last_result: Optional[SearchResult] = None

	for attempt in range(max_attempts):
		grid, start, goal = generate_grid(size=size, obstacle_prob=obstacle_prob, rng=rng)

		if algorithm == "baseline_astar":
			result = astar_search(grid, start, goal, config=AStarConfig(tie_breaker=tie_breaker or "g"))
		elif algorithm in {
			"enhanced_weighted_astar",
			"enhanced_weighted_astar_dynamic",
			"enhanced_weighted_astar_nomemo",
		}:
			if weight is None:
				raise ValueError("weight is required for weighted A* variants")

			cfg = WeightedAStarConfig(weight=float(weight))
			if tie_breaker is not None:
				cfg = replace(cfg, tie_breaker=tie_breaker)
			if memoize_heuristic is not None:
				cfg = replace(cfg, memoize_heuristic=bool(memoize_heuristic))
			if dynamic_weight is not None:
				cfg = replace(cfg, dynamic_weight=bool(dynamic_weight))
			if weight_min is not None:
				cfg = replace(cfg, weight_min=float(weight_min))
			if weight_decay_per_1000 is not None:
				cfg = replace(cfg, weight_decay_per_1000=float(weight_decay_per_1000))

			if algorithm == "enhanced_weighted_astar_nomemo":
				cfg = replace(cfg, memoize_heuristic=False, dynamic_weight=False)
			if algorithm == "enhanced_weighted_astar_dynamic":
				cfg = replace(cfg, dynamic_weight=True)

			result = weighted_astar_search(grid, start, goal, config=cfg)
		else:
			raise ValueError(f"Unknown algorithm: {algorithm}")

		last_result = result
		if result.found:
			return result, {"attempt": attempt + 1}

		# Keep deterministic variability if a particular seed yields unsolvable grids.
		rng.seed(seed + 1000 * (attempt + 1))

	assert last_result is not None
	return last_result, {"attempt": max_attempts, "unsolved": True}


class ExperimentRunner:
	"""Runs milestone experiments, stores raw rows, computes summaries, and plots."""

	def __init__(self, results_csv: Path):
		self.results_csv = results_csv
		self.rows: List[Dict[str, object]] = []
		self.summary_rows: List[Dict[str, object]] = []
		self._run_id_counter = 0
		ensure_dirs()

	def _record(self, row: Dict[str, object]) -> None:
		self.rows.append(row)

	def run_single(self, run: RunSpec) -> SearchResult:
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
			"algorithm": run.algorithm,
			"input_size": run.grid_spec.size,
			"execution_time": result.execution_time_s,
			"nodes_expanded": result.nodes_expanded,
			"path_length": result.path_cost,
			"run_id": run_id,
			"obstacle_prob": run.grid_spec.obstacle_prob,
			"found": int(result.found),
			"seed": run.seed,
			"heuristic_evals": result.heuristic_evals,
			"experiment": run.experiment,
		}
		if run.weight is not None:
			row["weight"] = float(run.weight)
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

	def _filter_rows(
		self,
		*,
		experiment: Optional[str] = None,
		algorithm: Optional[str] = None,
		input_size: Optional[int] = None,
		weight: Optional[float] = None,
	) -> List[Dict[str, object]]:
		out: List[Dict[str, object]] = []
		for r in self.rows:
			if experiment is not None and str(r.get("experiment")) != experiment:
				continue
			if algorithm is not None and str(r.get("algorithm")) != algorithm:
				continue
			if input_size is not None and int(r.get("input_size")) != int(input_size):
				continue
			if weight is not None:
				if "weight" not in r:
					continue
				if float(r["weight"]) != float(weight):
					continue
			out.append(r)
		return out

	def _metric_values(self, rows: List[Dict[str, object]], key: str) -> List[float]:
		return [float(r[key]) for r in rows]

	def _metric_stats(self, rows: List[Dict[str, object]], key: str) -> Dict[str, float]:
		vals = self._metric_values(rows, key)
		return {"mean": mean(vals), "std": stdev(vals), "ci95": ci95_half_width(vals), "n": float(len(vals))}

	def run_experiment_1_baseline_vs_enhanced(
		self,
		*,
		grid_spec: GridSpec,
		runs: int = 10,
		enhanced_weight: float = 2.2,
		use_dynamic_weight: bool = True,
		seed0: int = 111,
	) -> Dict[str, float]:
		enhanced_algo = "enhanced_weighted_astar_dynamic" if use_dynamic_weight else "enhanced_weighted_astar"

		baseline_times: List[float] = []
		enhanced_times: List[float] = []
		baseline_nodes: List[float] = []
		enhanced_nodes: List[float] = []

		for i in range(runs):
			seed = seed0 + i
			b = self.run_single(RunSpec("baseline_astar", grid_spec=grid_spec, run_id=None, seed=seed, experiment="exp1_baseline_vs_enhanced"))
			e = self.run_single(
				RunSpec(
					enhanced_algo,
					grid_spec=grid_spec,
					run_id=None,
					seed=seed,
					weight=enhanced_weight,
					experiment="exp1_baseline_vs_enhanced",
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
		*,
		sizes: Sequence[int],
		obstacle_prob: float,
		runs: int = 10,
		enhanced_weight: float = 2.2,
		use_dynamic_weight: bool = True,
		seed0: int = 222,
	) -> None:
		enhanced_algo = "enhanced_weighted_astar_dynamic" if use_dynamic_weight else "enhanced_weighted_astar"
		for size in sizes:
			grid_spec = GridSpec(size=size, obstacle_prob=obstacle_prob)
			for i in range(runs):
				seed = seed0 + 100 * size + i
				self.run_single(RunSpec("baseline_astar", grid_spec=grid_spec, run_id=None, seed=seed, experiment="exp2_scaling"))
				self.run_single(
					RunSpec(
						enhanced_algo,
						grid_spec=grid_spec,
						run_id=None,
						seed=seed,
						weight=enhanced_weight,
						experiment="exp2_scaling",
						memoize_heuristic=True,
						dynamic_weight=use_dynamic_weight,
						weight_min=1.0,
						weight_decay_per_1000=0.12,
						tie_breaker="h",
					)
				)

	def run_experiment_3_parameter_sensitivity(
		self,
		*,
		grid_spec: GridSpec,
		weights: Sequence[float],
		runs: int = 10,
		seed0: int = 333,
	) -> None:
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
						experiment="exp3_weight_sensitivity",
						memoize_heuristic=True,
						dynamic_weight=False,
						tie_breaker="h",
					)
				)

	def run_experiment_4_ablation_memoization(
		self,
		*,
		grid_spec: GridSpec,
		runs: int = 10,
		weight: float = 1.6,
		seed0: int = 444,
	) -> None:
		for i in range(runs):
			seed = seed0 + i
			self.run_single(
				RunSpec(
					"enhanced_weighted_astar_nomemo",
					grid_spec=grid_spec,
					run_id=None,
					seed=seed,
					weight=weight,
					experiment="exp4_ablation_memoization",
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
					experiment="exp4_ablation_memoization",
					memoize_heuristic=True,
					dynamic_weight=False,
					tie_breaker="h",
				)
			)

	def write_csv(self) -> None:
		from .utils import write_results_csv

		write_results_csv(self.results_csv, self.rows)

	def build_summaries(self, *, baseline_algo: str = "baseline_astar") -> None:
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

	def write_summary_csv(self, summary_csv: Path) -> None:
		from .utils import write_results_csv

		write_results_csv(summary_csv, self.summary_rows, required_fields=["algorithm", "input_size", "n"])

	def _plot_bar_with_error(self, *, labels: list[str], means: list[float], errs: list[float], title: str, ylabel: str, out_path: Path) -> None:
		fig, ax = plt.subplots(figsize=(7, 4))
		ax.bar(labels, means, yerr=errs, capsize=6)
		ax.set_title(title)
		ax.set_ylabel(ylabel)
		ax.grid(axis="y", alpha=0.3)
		fig.tight_layout()
		out_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out_path)
		plt.close(fig)

	def plot_experiment_1_bar(self, *, grid_size: int, out_path: Path) -> None:
		b = self._filter_rows(experiment="exp1_baseline_vs_enhanced", algorithm="baseline_astar", input_size=grid_size)
		e = self._filter_rows(experiment="exp1_baseline_vs_enhanced", algorithm="enhanced_weighted_astar_dynamic", input_size=grid_size)
		if not b or not e:
			return

		bt = self._metric_stats(b, "execution_time")
		et = self._metric_stats(e, "execution_time")
		bn = self._metric_stats(b, "nodes_expanded")
		en = self._metric_stats(e, "nodes_expanded")

		# One figure with two panels.
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
		ax1.bar(["A*", "Enhanced"], [bt["mean"], et["mean"]], yerr=[bt["ci95"], et["ci95"]], capsize=6)
		ax1.set_title("Execution Time (Exp1)")
		ax1.set_ylabel("seconds")
		ax1.grid(axis="y", alpha=0.3)

		ax2.bar(["A*", "Enhanced"], [bn["mean"], en["mean"]], yerr=[bn["ci95"], en["ci95"]], capsize=6)
		ax2.set_title("Nodes Expanded (Exp1)")
		ax2.set_ylabel("nodes")
		ax2.grid(axis="y", alpha=0.3)

		fig.suptitle(f"Experiment 1: Baseline vs Enhanced (size={grid_size})")
		fig.tight_layout()
		out_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out_path)
		plt.close(fig)

	def plot_experiment_2_scaling_lines(self, *, sizes: Sequence[int], out_path: Path) -> None:
		# Time plot
		base_means: List[float] = []
		enh_means: List[float] = []
		base_ci: List[float] = []
		enh_ci: List[float] = []

		base_nodes_means: List[float] = []
		enh_nodes_means: List[float] = []
		base_nodes_ci: List[float] = []
		enh_nodes_ci: List[float] = []

		for s in sizes:
			b = self._filter_rows(experiment="exp2_scaling", algorithm="baseline_astar", input_size=s)
			e = self._filter_rows(experiment="exp2_scaling", algorithm="enhanced_weighted_astar_dynamic", input_size=s)
			if not b or not e:
				return
			bt = self._metric_stats(b, "execution_time")
			et = self._metric_stats(e, "execution_time")
			bn = self._metric_stats(b, "nodes_expanded")
			en = self._metric_stats(e, "nodes_expanded")

			base_means.append(bt["mean"])
			enh_means.append(et["mean"])
			base_ci.append(bt["ci95"])
			enh_ci.append(et["ci95"])

			base_nodes_means.append(bn["mean"])
			enh_nodes_means.append(en["mean"])
			base_nodes_ci.append(bn["ci95"])
			enh_nodes_ci.append(en["ci95"])

		fig, ax = plt.subplots(figsize=(7, 4))
		ax.errorbar(list(sizes), base_means, yerr=base_ci, marker="o", label="A*")
		ax.errorbar(list(sizes), enh_means, yerr=enh_ci, marker="o", label="Enhanced")
		ax.set_title("Scaling: Execution Time vs Size (Exp2)")
		ax.set_xlabel("grid size (N)")
		ax.set_ylabel("seconds")
		ax.grid(alpha=0.3)
		ax.legend()
		fig.tight_layout()
		out_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out_path)
		plt.close(fig)

		# Nodes plot
		if out_path.name == "exp2_scaling_time.png":
			nodes_out = out_path.with_name("exp2_scaling_time_nodes.png")
		else:
			nodes_out = out_path.with_name(out_path.stem + "_nodes.png")
		fig2, ax2 = plt.subplots(figsize=(7, 4))
		ax2.errorbar(list(sizes), base_nodes_means, yerr=base_nodes_ci, marker="o", label="A*")
		ax2.errorbar(list(sizes), enh_nodes_means, yerr=enh_nodes_ci, marker="o", label="Enhanced")
		ax2.set_title("Scaling: Nodes Expanded vs Size (Exp2)")
		ax2.set_xlabel("grid size (N)")
		ax2.set_ylabel("nodes expanded")
		ax2.grid(alpha=0.3)
		ax2.legend()
		fig2.tight_layout()
		fig2.savefig(nodes_out)
		plt.close(fig2)

	def plot_experiment_3_sensitivity(self, *, grid_size: int, out_path: Path) -> None:
		# Time vs weight
		weights = sorted({float(r["weight"]) for r in self._filter_rows(experiment="exp3_weight_sensitivity", input_size=grid_size) if "weight" in r})
		if not weights:
			return

		time_means: List[float] = []
		time_ci: List[float] = []
		nodes_means: List[float] = []
		nodes_ci: List[float] = []
		path_means: List[float] = []

		for w in weights:
			rs = self._filter_rows(experiment="exp3_weight_sensitivity", algorithm="enhanced_weighted_astar", input_size=grid_size, weight=w)
			if not rs:
				continue
			t = self._metric_stats(rs, "execution_time")
			n = self._metric_stats(rs, "nodes_expanded")
			l = self._metric_stats(rs, "path_length")
			time_means.append(t["mean"])
			time_ci.append(t["ci95"])
			nodes_means.append(n["mean"])
			nodes_ci.append(n["ci95"])
			path_means.append(l["mean"])

		fig, ax = plt.subplots(figsize=(7, 4))
		ax.errorbar(weights, time_means, yerr=time_ci, marker="o")
		ax.set_title(f"Weight Sensitivity: Mean Time (size={grid_size})")
		ax.set_xlabel("weight (w)")
		ax.set_ylabel("seconds")
		ax.grid(alpha=0.3)
		fig.tight_layout()
		out_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out_path)
		plt.close(fig)

		# Nodes and path length plot
		if out_path.name == "exp3_weight_sensitivity.png":
			out2 = out_path.with_name("exp3_weight_sensitivity_nodes_path.png")
		else:
			out2 = out_path.with_name(out_path.stem + "_nodes_path.png")

		fig2, ax2 = plt.subplots(figsize=(7, 4))
		ax2.errorbar(weights, nodes_means, yerr=nodes_ci, marker="o", label="nodes expanded")
		ax2.set_xlabel("weight (w)")
		ax2.set_ylabel("nodes expanded")
		ax2.grid(alpha=0.3)
		ax2b = ax2.twinx()
		ax2b.plot(weights, path_means, marker="s", linestyle="--", label="path length")
		ax2b.set_ylabel("path length")
		ax2.set_title(f"Weight Sensitivity: Nodes vs Path Length (size={grid_size})")
		fig2.tight_layout()
		fig2.savefig(out2)
		plt.close(fig2)

	def plot_improvement_vs_size(self, *, sizes: Sequence[int], out_path: Path) -> None:
		time_impr: List[float] = []
		nodes_impr: List[float] = []

		for s in sizes:
			b = self._filter_rows(experiment="exp2_scaling", algorithm="baseline_astar", input_size=s)
			e = self._filter_rows(experiment="exp2_scaling", algorithm="enhanced_weighted_astar_dynamic", input_size=s)
			if not b or not e:
				return
			bt = mean(self._metric_values(b, "execution_time"))
			et = mean(self._metric_values(e, "execution_time"))
			bn = mean(self._metric_values(b, "nodes_expanded"))
			en = mean(self._metric_values(e, "nodes_expanded"))
			time_impr.append(pct_improvement(bt, et))
			nodes_impr.append(pct_improvement(bn, en))

		fig, ax = plt.subplots(figsize=(7, 4))
		ax.plot(list(sizes), time_impr, marker="o", label="time improvement %")
		ax.plot(list(sizes), nodes_impr, marker="o", label="nodes improvement %")
		ax.set_title("Improvement vs Input Size (Exp2)")
		ax.set_xlabel("grid size (N)")
		ax.set_ylabel("improvement % (higher is better)")
		ax.grid(alpha=0.3)
		ax.legend()
		fig.tight_layout()
		out_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out_path)
		plt.close(fig)

	def best_weight_by_time(self, *, grid_size: int) -> Optional[Dict[str, float]]:
		weights = sorted({float(r["weight"]) for r in self._filter_rows(experiment="exp3_weight_sensitivity", input_size=grid_size) if "weight" in r})
		if not weights:
			return None

		best_w: Optional[float] = None
		best_time = float("inf")
		best_nodes = 0.0
		best_len = 0.0

		for w in weights:
			rs = self._filter_rows(experiment="exp3_weight_sensitivity", algorithm="enhanced_weighted_astar", input_size=grid_size, weight=w)
			if not rs:
				continue
			t = mean(self._metric_values(rs, "execution_time"))
			if t < best_time:
				best_time = t
				best_w = w
				best_nodes = mean(self._metric_values(rs, "nodes_expanded"))
				best_len = mean(self._metric_values(rs, "path_length"))

		if best_w is None:
			return None
		return {"best_weight": float(best_w), "mean_time": float(best_time), "mean_nodes": float(best_nodes), "mean_path_length": float(best_len)}

	def wa_weight_stats(self, *, grid_size: int, weight: float) -> Optional[Dict[str, float]]:
		rs = self._filter_rows(experiment="exp3_weight_sensitivity", algorithm="enhanced_weighted_astar", input_size=grid_size, weight=weight)
		if not rs:
			return None
		return {
			"weight": float(weight),
			"mean_time": mean(self._metric_values(rs, "execution_time")),
			"mean_nodes": mean(self._metric_values(rs, "nodes_expanded")),
			"mean_path_length": mean(self._metric_values(rs, "path_length")),
		}

	def avg_improvement_across_sizes(self, *, sizes: Sequence[int]) -> Dict[str, float]:
		time_impr: List[float] = []
		nodes_impr: List[float] = []
		for s in sizes:
			b = self._filter_rows(experiment="exp2_scaling", algorithm="baseline_astar", input_size=s)
			e = self._filter_rows(experiment="exp2_scaling", algorithm="enhanced_weighted_astar_dynamic", input_size=s)
			if not b or not e:
				continue
			bt = mean(self._metric_values(b, "execution_time"))
			et = mean(self._metric_values(e, "execution_time"))
			bn = mean(self._metric_values(b, "nodes_expanded"))
			en = mean(self._metric_values(e, "nodes_expanded"))
			time_impr.append(pct_improvement(bt, et))
			nodes_impr.append(pct_improvement(bn, en))
		return {
			"avg_time_improvement_pct": mean(time_impr) if time_impr else 0.0,
			"avg_nodes_improvement_pct": mean(nodes_impr) if nodes_impr else 0.0,
		}

	def ablation_memoization_summary(self, *, grid_size: int) -> Optional[Dict[str, float]]:
		nom = self._filter_rows(experiment="exp4_ablation_memoization", algorithm="enhanced_weighted_astar_nomemo", input_size=grid_size)
		mem = self._filter_rows(experiment="exp4_ablation_memoization", algorithm="enhanced_weighted_astar", input_size=grid_size)
		if not nom or not mem:
			return None

		nt = mean(self._metric_values(nom, "execution_time"))
		mt = mean(self._metric_values(mem, "execution_time"))
		nn = mean(self._metric_values(nom, "nodes_expanded"))
		mn = mean(self._metric_values(mem, "nodes_expanded"))
		nh = mean(self._metric_values(nom, "heuristic_evals"))
		mh = mean(self._metric_values(mem, "heuristic_evals"))
		return {
			"time_improvement_pct": pct_improvement(nt, mt),
			"nodes_improvement_pct": pct_improvement(nn, mn),
			"heuristic_evals_reduction_pct": pct_improvement(nh, mh),
		}


def default_experiments() -> ExperimentRunner:
	"""Factory for the default experiment runner."""

	ensure_dirs()
	return ExperimentRunner(project_root() / "data" / "results.csv")

