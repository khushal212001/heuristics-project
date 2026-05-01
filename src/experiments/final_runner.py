"""Final Project experiment suite (rubric-driven).

Meets final rubric requirements:
- >= 5 experiments (this suite provides 7)
- n >= 30 runs per condition
- reproducible (deterministic seeds)
- CSV + plots written to final_results/
- automated analysis: improvement %, best configuration, trade-offs

This module is *additive*: it does not remove or alter the Milestone 2 pipeline.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..algorithms.solver import effective_strategy, solve
from ..config import Config
from ..utils import Coord, Grid, SearchResult, ci95_half_width, ensure_dirs, generate_grid, mean, obstacle_density, pct_improvement, project_root, stdev, write_results_csv


@dataclass(frozen=True)
class Condition:
    """A single experimental condition (a named Config + environment params)."""

    name: str
    config: Config


@dataclass(frozen=True)
class GridSpec:
    size: int
    obstacle_prob: float


def _solve_with_retry(
    *,
    grid_spec: GridSpec,
    config: Config,
    seed: int,
    max_attempts: int = 80,
) -> Tuple[SearchResult, Dict[str, object], Grid]:
    """Generate a solvable grid and solve it; retry generation if unsolvable."""

    rng = random.Random(seed)
    last: Optional[SearchResult] = None
    last_grid: Optional[Grid] = None

    for attempt in range(max_attempts):
        grid, start, goal = generate_grid(size=grid_spec.size, obstacle_prob=grid_spec.obstacle_prob, rng=rng)
        last_grid = grid

        dens = obstacle_density(grid)
        effective_algorithm, effective_weight = effective_strategy(grid, config)

        result = solve(grid, start, goal, config)
        last = result
        if result.found:
            meta = {
                "attempt": attempt + 1,
                "obstacle_density": dens,
                "effective_algorithm": effective_algorithm,
                "effective_weight": effective_weight,
            }
            return result, meta, grid

        rng.seed(seed + 1000 * (attempt + 1))

    assert last is not None and last_grid is not None
    dens2 = obstacle_density(last_grid)
    effective_algorithm, effective_weight = effective_strategy(last_grid, config)

    return (
        last,
        {
            "attempt": max_attempts,
            "unsolved": True,
            "obstacle_density": dens2,
            "effective_algorithm": effective_algorithm,
            "effective_weight": effective_weight,
        },
        last_grid,
    )


class FinalExperimentRunner:
    """Runs the final suite and writes results to final_results/."""

    def __init__(self, *, base_seed: int = 1337):
        self.base_seed = base_seed
        self.rows: List[Dict[str, object]] = []
        self.summary_rows: List[Dict[str, object]] = []
        self._run_id_counter = 0

        ensure_dirs()
        root = project_root() / "final_results"
        self.csv_dir = root / "csv"
        self.plots_dir = root / "plots"
        self.logs_dir = root / "logs"

    def _next_run_id(self) -> int:
        rid = self._run_id_counter
        self._run_id_counter += 1
        return rid

    def _record_run(
        self,
        *,
        experiment: str,
        condition: Condition,
        grid_spec: GridSpec,
        seed: int,
        result: SearchResult,
        meta: Dict[str, object],
    ) -> None:
        cfg = condition.config
        row: Dict[str, object] = {
            "experiment": experiment,
            "condition": condition.name,
            "algorithm": cfg.algorithm,
            "input_size": grid_spec.size,
            "obstacle_prob": grid_spec.obstacle_prob,
            "execution_time": result.execution_time_s,
            "nodes_expanded": result.nodes_expanded,
            "path_length": result.path_cost,
            "heuristic_evals": result.heuristic_evals,
            "heuristic_calls": result.heuristic_evals,
            "tie_breaks": result.tie_breaks,
            "run_id": self._next_run_id(),
            "seed": seed,
            "heuristic": cfg.heuristic,
            "alpha": cfg.alpha,
            "weight": cfg.weight,
            "use_memoization": int(cfg.use_memoization),
            "use_dynamic_weight": int(cfg.use_dynamic_weight),
            "use_tiebreaking": int(cfg.use_tiebreaking),
            "use_density_adaptive": int(cfg.use_density_adaptive),
            "density_low_threshold": cfg.density_low_threshold,
            "density_threshold": cfg.density_threshold,
            "mid_density_weight": cfg.mid_density_weight,
            "high_density_weight": cfg.high_density_weight,
            "density_weight_cap": cfg.density_weight_cap,
            "found": int(result.found),
        }
        row.update(meta)
        self.rows.append(row)

    def _generate_shared_instances(
        self,
        *,
        grid_spec: GridSpec,
        runs: int,
        seed_offset: int,
        max_attempts: int = 80,
    ) -> List[Dict[str, object]]:
        """Generate a fixed set of solvable instances for fair, stable comparisons."""

        instances: List[Dict[str, object]] = []
        solvability_cfg = Config(algorithm="astar", heuristic="manhattan", use_memoization=False, use_dynamic_weight=False, use_tiebreaking=False)

        for i in range(runs):
            seed = self.base_seed + seed_offset + i
            rng = random.Random(seed)
            last_grid: Optional[Grid] = None
            last_start: Optional[Coord] = None
            last_goal: Optional[Coord] = None

            for attempt in range(max_attempts):
                grid, start, goal = generate_grid(size=grid_spec.size, obstacle_prob=grid_spec.obstacle_prob, rng=rng)
                last_grid, last_start, last_goal = grid, start, goal

                # Ensure *the instance* is solvable independent of the condition under test.
                res = solve(grid, start, goal, solvability_cfg)
                if res.found:
                    dens = obstacle_density(grid)
                    instances.append(
                        {
                            "seed": seed,
                            "attempt": attempt + 1,
                            "grid": grid,
                            "start": start,
                            "goal": goal,
                            "obstacle_density": dens,
                        }
                    )
                    break

                rng.seed(seed + 1000 * (attempt + 1))

            if len(instances) != i + 1:
                assert last_grid is not None and last_start is not None and last_goal is not None
                dens2 = obstacle_density(last_grid)
                instances.append(
                    {
                        "seed": seed,
                        "attempt": max_attempts,
                        "grid": last_grid,
                        "start": last_start,
                        "goal": last_goal,
                        "obstacle_density": dens2,
                        "unsolved": True,
                    }
                )

        return instances

    def run_condition_on_instances(
        self,
        *,
        experiment: str,
        condition: Condition,
        grid_spec: GridSpec,
        instances: List[Dict[str, object]],
    ) -> None:
        for idx, inst in enumerate(instances):
            grid = inst["grid"]
            start = inst["start"]
            goal = inst["goal"]
            seed = int(inst["seed"])
            dens = float(inst["obstacle_density"])

            res = solve(grid, start, goal, condition.config)
            effective_algorithm, effective_weight = effective_strategy(grid, condition.config)
            meta: Dict[str, object] = {
                "attempt": int(inst.get("attempt", 1)),
                "obstacle_density": dens,
                "effective_algorithm": effective_algorithm,
                "effective_weight": effective_weight,
                "shared_instance_id": idx,
            }
            if "unsolved" in inst:
                meta["unsolved"] = True
            self._record_run(experiment=experiment, condition=condition, grid_spec=grid_spec, seed=seed, result=res, meta=meta)

    def _metric_values(self, rows: List[Dict[str, object]], key: str) -> List[float]:
        return [float(r[key]) for r in rows]

    def _metric_stats(self, rows: List[Dict[str, object]], key: str) -> Dict[str, float]:
        vals = self._metric_values(rows, key)
        return {
            "mean": mean(vals),
            "std": stdev(vals),
            "ci95": ci95_half_width(vals),
            "n": float(len(vals)),
        }

    def _filter(
        self,
        *,
        experiment: str,
        condition: Optional[str] = None,
        input_size: Optional[int] = None,
        obstacle_prob: Optional[float] = None,
        heuristic: Optional[str] = None,
        weight: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for r in self.rows:
            if str(r["experiment"]) != experiment:
                continue
            if condition is not None and str(r["condition"]) != condition:
                continue
            if input_size is not None and int(r["input_size"]) != int(input_size):
                continue
            if obstacle_prob is not None and float(r["obstacle_prob"]) != float(obstacle_prob):
                continue
            if heuristic is not None and str(r["heuristic"]) != heuristic:
                continue
            if weight is not None and float(r["weight"]) != float(weight):
                continue
            out.append(r)
        return out

    def run_condition(
        self,
        *,
        experiment: str,
        condition: Condition,
        grid_spec: GridSpec,
        runs: int,
        seed_offset: int,
    ) -> None:
        for i in range(runs):
            seed = self.base_seed + seed_offset + i
            res, meta, _grid = _solve_with_retry(grid_spec=grid_spec, config=condition.config, seed=seed)
            self._record_run(experiment=experiment, condition=condition, grid_spec=grid_spec, seed=seed, result=res, meta=meta)

    # ----------------------- Experiments -----------------------

    def exp1_baseline_vs_enhanced(self, *, runs: int = 30) -> None:
        """Experiment 1: baseline A* vs enhanced WA* on 40x40."""

        grid = GridSpec(size=40, obstacle_prob=0.22)
        baseline = Condition(
            "baseline_astar",
            Config(algorithm="astar", heuristic="manhattan", use_memoization=False, use_dynamic_weight=False, use_tiebreaking=False),
        )
        enhanced = Condition(
            "enhanced_combo",
            Config(
                algorithm="wastar",
                weight=2.2,
                heuristic="hybrid",
                alpha=0.7,
                use_memoization=True,
                use_dynamic_weight=True,
                use_tiebreaking=True,
                use_density_adaptive=False,
            ),
        )

        self.run_condition(experiment="exp1_baseline_vs_enhanced", condition=baseline, grid_spec=grid, runs=runs, seed_offset=10_000)
        self.run_condition(experiment="exp1_baseline_vs_enhanced", condition=enhanced, grid_spec=grid, runs=runs, seed_offset=20_000)

    def exp2_scaling(self, *, runs: int = 30, sizes: Sequence[int] = (20, 40, 60, 80, 100)) -> None:
        """Experiment 2: scaling analysis across sizes (extended to 80/100)."""

        baseline_cfg = Config(algorithm="astar", heuristic="manhattan", use_memoization=False, use_dynamic_weight=False, use_tiebreaking=False)
        enhanced_cfg = Config(
            algorithm="wastar",
            weight=2.2,
            heuristic="hybrid",
            alpha=0.7,
            use_memoization=True,
            use_dynamic_weight=True,
            use_tiebreaking=True,
        )

        for size in sizes:
            grid = GridSpec(size=int(size), obstacle_prob=0.22)
            self.run_condition(
                experiment="exp2_scaling",
                condition=Condition("baseline_astar", baseline_cfg),
                grid_spec=grid,
                runs=runs,
                seed_offset=30_000 + 100 * int(size),
            )
            self.run_condition(
                experiment="exp2_scaling",
                condition=Condition("enhanced_combo", enhanced_cfg),
                grid_spec=grid,
                runs=runs,
                seed_offset=40_000 + 100 * int(size),
            )

    def exp3_weight_sensitivity(self, *, runs: int = 30, weights: Sequence[float] = (1.0, 1.2, 1.4, 1.6, 2.0, 2.5)) -> None:
        """Experiment 3: WA* weight sensitivity (time/nodes/path length)."""

        grid = GridSpec(size=40, obstacle_prob=0.22)
        # Generate a fixed set of solvable instances and reuse it across all weights.
        instances = self._generate_shared_instances(grid_spec=grid, runs=runs, seed_offset=50_000)
        for w in weights:
            cfg = Config(
                algorithm="wastar",
                weight=float(w),
                heuristic="manhattan",
                use_memoization=True,
                use_dynamic_weight=False,
                use_tiebreaking=True,
            )
            self.run_condition_on_instances(
                experiment="exp3_weight_sensitivity",
                condition=Condition(f"w={w}", cfg),
                grid_spec=grid,
                instances=instances,
            )

    def exp4_ablation_memoization(self, *, runs: int = 30) -> None:
        """Experiment 4: ablation (memoization on/off) at fixed weight."""

        grid = GridSpec(size=40, obstacle_prob=0.22)
        base = Config(algorithm="wastar", weight=1.6, heuristic="manhattan", use_dynamic_weight=False, use_tiebreaking=True)
        instances = self._generate_shared_instances(grid_spec=grid, runs=runs, seed_offset=60_000)
        self.run_condition_on_instances(
            experiment="exp4_ablation_memoization",
            condition=Condition("no_memo", replace(base, use_memoization=False)),
            grid_spec=grid,
            instances=instances,
        )
        self.run_condition_on_instances(
            experiment="exp4_ablation_memoization",
            condition=Condition("memo", replace(base, use_memoization=True)),
            grid_spec=grid,
            instances=instances,
        )

    def exp5_obstacle_density(self, *, runs: int = 30, densities: Sequence[float] = (0.10, 0.20, 0.30, 0.40)) -> None:
        """Experiment 5: obstacle density analysis (0.1 -> 0.4)."""

        size = 60
        baseline_cfg = Config(algorithm="astar", heuristic="manhattan", use_memoization=False, use_dynamic_weight=False, use_tiebreaking=False)
        enhanced_cfg = Config(
            algorithm="wastar",
            weight=2.2,
            heuristic="hybrid",
            alpha=0.7,
            use_memoization=True,
            use_dynamic_weight=True,
            use_tiebreaking=True,
        )

        # Density-adaptive enhancement: choose A* at low density and WA* (higher weight) at high density.
        adaptive_cfg = Config(
            algorithm="adaptive",
            weight=1.5,
            heuristic="manhattan",
            alpha=0.7,
            use_memoization=True,
            use_dynamic_weight=True,
            use_tiebreaking=True,
            use_density_adaptive=True,
            density_low_threshold=0.20,
            density_threshold=0.35,
            mid_density_weight=1.6,
            high_density_weight=2.4,
            density_weight_cap=2.5,
        )

        for p in densities:
            grid = GridSpec(size=size, obstacle_prob=float(p))
            self.run_condition(
                experiment="exp5_obstacle_density",
                condition=Condition("baseline_astar", baseline_cfg),
                grid_spec=grid,
                runs=runs,
                seed_offset=70_000 + int(1000 * float(p)),
            )
            self.run_condition(
                experiment="exp5_obstacle_density",
                condition=Condition("enhanced_combo", enhanced_cfg),
                grid_spec=grid,
                runs=runs,
                seed_offset=80_000 + int(1000 * float(p)),
            )

            self.run_condition(
                experiment="exp5_obstacle_density",
                condition=Condition("density_adaptive", adaptive_cfg),
                grid_spec=grid,
                runs=runs,
                seed_offset=85_000 + int(1000 * float(p)),
            )

    def exp6_heuristic_comparison(self, *, runs: int = 30) -> None:
        """Experiment 6: compare heuristics (Manhattan vs Euclidean vs Hybrid)."""

        grid = GridSpec(size=60, obstacle_prob=0.22)
        alg = "astar"
        base = Config(algorithm=alg, use_memoization=False, use_dynamic_weight=False, use_tiebreaking=True)

        conditions = [
            Condition("manhattan", replace(base, heuristic="manhattan", alpha=0.7)),
            Condition("euclidean", replace(base, heuristic="euclidean", alpha=0.7)),
            Condition("hybrid_a0.7", replace(base, heuristic="hybrid", alpha=0.7)),
        ]
        for idx, c in enumerate(conditions):
            self.run_condition(
                experiment="exp6_heuristic_comparison",
                condition=c,
                grid_spec=grid,
                runs=runs,
                seed_offset=90_000 + 1000 * idx,
            )

    def exp7_tiebreaking(self, *, runs: int = 30) -> None:
        """Experiment 7: tie-breaking (prefer low h on f ties) on/off."""

        grid = GridSpec(size=60, obstacle_prob=0.22)
        base = Config(
            algorithm="wastar",
            weight=2.0,
            heuristic="hybrid",
            alpha=0.7,
            use_memoization=True,
            use_dynamic_weight=True,
            use_density_adaptive=False,
        )

        instances = self._generate_shared_instances(grid_spec=grid, runs=runs, seed_offset=100_000)
        self.run_condition_on_instances(
            experiment="exp7_tiebreaking",
            condition=Condition("tiebreak_off", replace(base, use_tiebreaking=False)),
            grid_spec=grid,
            instances=instances,
        )
        self.run_condition_on_instances(
            experiment="exp7_tiebreaking",
            condition=Condition("tiebreak_on", replace(base, use_tiebreaking=True)),
            grid_spec=grid,
            instances=instances,
        )

    # ----------------------- Summaries & Output -----------------------

    def build_summaries(self) -> None:
        """Build summary rows for each (experiment, condition, input_size, obstacle_prob)."""

        self.summary_rows = []
        experiments = sorted({str(r["experiment"]) for r in self.rows})

        for exp in experiments:
            exp_rows = [r for r in self.rows if str(r["experiment"]) == exp]
            conditions = sorted({str(r["condition"]) for r in exp_rows})
            sizes = sorted({int(r["input_size"]) for r in exp_rows})
            probs = sorted({float(r["obstacle_prob"]) for r in exp_rows})

            # Baseline within each experiment is defined as condition name containing "baseline" if present,
            # otherwise the first condition.
            baseline_cond = next((c for c in conditions if "baseline" in c), conditions[0] if conditions else None)

            for size in sizes:
                for p in probs:
                    base_group = (
                        self._filter(experiment=exp, condition=baseline_cond, input_size=size, obstacle_prob=p)
                        if baseline_cond is not None
                        else []
                    )

                    base_time = self._metric_stats(base_group, "execution_time") if base_group else None
                    base_nodes = self._metric_stats(base_group, "nodes_expanded") if base_group else None

                    for cond in conditions:
                        group = self._filter(experiment=exp, condition=cond, input_size=size, obstacle_prob=p)
                        if not group:
                            continue

                        t = self._metric_stats(group, "execution_time")
                        n = self._metric_stats(group, "nodes_expanded")
                        l = self._metric_stats(group, "path_length")
                        he = self._metric_stats(group, "heuristic_evals")
                        tb = self._metric_stats(group, "tie_breaks")

                        row: Dict[str, object] = {
                            "experiment": exp,
                            "condition": cond,
                            "input_size": size,
                            "obstacle_prob": p,
                            "n": int(t["n"]),
                            "time_mean": t["mean"],
                            "time_std": t["std"],
                            "time_ci95": t["ci95"],
                            "nodes_mean": n["mean"],
                            "nodes_std": n["std"],
                            "nodes_ci95": n["ci95"],
                            "path_len_mean": l["mean"],
                            "path_len_std": l["std"],
                            "path_len_ci95": l["ci95"],
                            "heur_evals_mean": he["mean"],
                            "heur_evals_std": he["std"],
                            "heur_evals_ci95": he["ci95"],
                            "tie_breaks_mean": tb["mean"],
                            "tie_breaks_std": tb["std"],
                            "tie_breaks_ci95": tb["ci95"],
                        }

                        if base_time is not None and base_nodes is not None:
                            row["time_improvement_pct"] = pct_improvement(base_time["mean"], t["mean"])
                            row["nodes_improvement_pct"] = pct_improvement(base_nodes["mean"], n["mean"])
                        self.summary_rows.append(row)

    def write_csvs(self) -> None:
        """Write raw runs and summary CSVs."""

        ensure_dirs()
        runs_csv = self.csv_dir / "final_runs.csv"
        summary_csv = self.csv_dir / "final_results.csv"

        write_results_csv(
            runs_csv,
            self.rows,
            required_fields=[
                "experiment",
                "condition",
                "algorithm",
                "input_size",
                "obstacle_prob",
                "execution_time",
                "nodes_expanded",
                "path_length",
                "heuristic_evals",
                "heuristic_calls",
                "tie_breaks",
                "run_id",
            ],
        )
        write_results_csv(
            summary_csv,
            self.summary_rows,
            required_fields=["experiment", "condition", "input_size", "obstacle_prob", "n"],
        )

    def _save_plot(self, fig, name: str) -> None:
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(self.plots_dir / name)
        plt.close(fig)

    def plot_all(self) -> None:
        """Generate rubric-required plots under final_results/plots."""

        self._plot_exp1()
        self._plot_exp2()
        self._plot_exp3()
        self._plot_exp4()
        self._plot_exp5()
        self._plot_exp6()
        self._plot_exp7()

    def _summary_lookup(self, *, experiment: str) -> List[Dict[str, object]]:
        return [r for r in self.summary_rows if str(r["experiment"]) == experiment]

    def _plot_exp1(self) -> None:
        rows = self._summary_lookup(experiment="exp1_baseline_vs_enhanced")
        rows = [r for r in rows if int(r["input_size"]) == 40 and float(r["obstacle_prob"]) == 0.22]
        if not rows:
            return

        labels = [str(r["condition"]) for r in rows]
        time_means = [float(r["time_mean"]) for r in rows]
        time_err = [float(r["time_std"]) for r in rows]
        nodes_means = [float(r["nodes_mean"]) for r in rows]
        nodes_err = [float(r["nodes_std"]) for r in rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(labels, time_means, yerr=time_err, capsize=6)
        ax1.set_title("Exp1: Execution Time")
        ax1.set_ylabel("seconds")
        ax1.grid(axis="y", alpha=0.3)

        ax2.bar(labels, nodes_means, yerr=nodes_err, capsize=6)
        ax2.set_title("Exp1: Nodes Expanded")
        ax2.set_ylabel("nodes")
        ax2.grid(axis="y", alpha=0.3)

        fig.suptitle("Experiment 1: Baseline vs Enhanced (mean ± std)")
        self._save_plot(fig, "exp1_baseline_vs_enhanced.png")

    def _plot_exp2(self) -> None:
        rows = self._summary_lookup(experiment="exp2_scaling")
        if not rows:
            return

        sizes = sorted({int(r["input_size"]) for r in rows})
        conds = sorted({str(r["condition"]) for r in rows})

        fig, ax = plt.subplots(figsize=(7, 4))
        for cond in conds:
            ys = []
            yerr = []
            for s in sizes:
                r = next((x for x in rows if str(x["condition"]) == cond and int(x["input_size"]) == s), None)
                if r is None:
                    continue
                ys.append(float(r["time_mean"]))
                yerr.append(float(r["time_std"]))
            ax.errorbar(sizes, ys, yerr=yerr, marker="o", label=cond)

        ax.set_title("Experiment 2: Scaling (Time, mean ± std)")
        ax.set_xlabel("grid size (N)")
        ax.set_ylabel("seconds")
        ax.grid(alpha=0.3)
        ax.legend()
        self._save_plot(fig, "exp2_scaling_time.png")

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        for cond in conds:
            ys = []
            yerr = []
            for s in sizes:
                r = next((x for x in rows if str(x["condition"]) == cond and int(x["input_size"]) == s), None)
                if r is None:
                    continue
                ys.append(float(r["nodes_mean"]))
                yerr.append(float(r["nodes_std"]))
            ax2.errorbar(sizes, ys, yerr=yerr, marker="o", label=cond)

        ax2.set_title("Experiment 2: Scaling (Nodes, mean ± std)")
        ax2.set_xlabel("grid size (N)")
        ax2.set_ylabel("nodes expanded")
        ax2.grid(alpha=0.3)
        ax2.legend()
        self._save_plot(fig2, "exp2_scaling_nodes.png")

        # Improvement vs size
        baseline = "baseline_astar"
        enhanced = "enhanced_combo"
        time_impr = []
        nodes_impr = []
        for s in sizes:
            b = next((x for x in rows if str(x["condition"]) == baseline and int(x["input_size"]) == s), None)
            e = next((x for x in rows if str(x["condition"]) == enhanced and int(x["input_size"]) == s), None)
            if b is None or e is None:
                continue
            time_impr.append(pct_improvement(float(b["time_mean"]), float(e["time_mean"])))
            nodes_impr.append(pct_improvement(float(b["nodes_mean"]), float(e["nodes_mean"])))

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.plot(sizes[: len(time_impr)], time_impr, marker="o", label="time improvement %")
        ax3.plot(sizes[: len(nodes_impr)], nodes_impr, marker="o", label="nodes improvement %")
        ax3.set_title("Experiment 2: Improvement vs Size")
        ax3.set_xlabel("grid size (N)")
        ax3.set_ylabel("improvement % (higher is better)")
        ax3.grid(alpha=0.3)
        ax3.legend()
        self._save_plot(fig3, "exp2_improvement_vs_size.png")

    def _plot_exp3(self) -> None:
        rows = self._summary_lookup(experiment="exp3_weight_sensitivity")
        rows = [r for r in rows if int(r["input_size"]) == 40]
        if not rows:
            return

        # Parse weights from condition names like 'w=1.4'
        pairs = []
        for r in rows:
            cond = str(r["condition"])
            if cond.startswith("w="):
                pairs.append((float(cond.split("=")[1]), r))
        pairs.sort(key=lambda t: t[0])

        ws = [w for w, _ in pairs]
        time_means = [float(r["time_mean"]) for _, r in pairs]
        time_err = [float(r["time_std"]) for _, r in pairs]
        nodes_means = [float(r["nodes_mean"]) for _, r in pairs]
        nodes_err = [float(r["nodes_std"]) for _, r in pairs]
        path_means = [float(r["path_len_mean"]) for _, r in pairs]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(ws, time_means, yerr=time_err, marker="o")
        ax.set_title("Experiment 3: Weight Sensitivity (Time)")
        ax.set_xlabel("weight (w)")
        ax.set_ylabel("seconds")
        ax.grid(alpha=0.3)
        self._save_plot(fig, "exp3_weight_sensitivity_time.png")

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.errorbar(ws, nodes_means, yerr=nodes_err, marker="o", label="nodes")
        ax2.set_xlabel("weight (w)")
        ax2.set_ylabel("nodes expanded")
        ax2.grid(alpha=0.3)
        ax2b = ax2.twinx()
        ax2b.plot(ws, path_means, marker="s", linestyle="--", label="path length")
        ax2b.set_ylabel("path length")
        ax2.set_title("Experiment 3: Nodes vs Path Length")
        self._save_plot(fig2, "exp3_weight_sensitivity_nodes_path.png")

    def _plot_exp4(self) -> None:
        rows = self._summary_lookup(experiment="exp4_ablation_memoization")
        rows = [r for r in rows if int(r["input_size"]) == 40]
        if not rows:
            return

        labels = [str(r["condition"]) for r in rows]
        t = [float(r["time_mean"]) for r in rows]
        terr = [float(r["time_std"]) for r in rows]
        he = [float(r["heur_evals_mean"]) for r in rows]
        heerr = [float(r["heur_evals_std"]) for r in rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(labels, t, yerr=terr, capsize=6)
        ax1.set_title("Exp4: Time")
        ax1.set_ylabel("seconds")
        ax1.grid(axis="y", alpha=0.3)

        ax2.bar(labels, he, yerr=heerr, capsize=6)
        ax2.set_title("Exp4: Heuristic Evaluations")
        ax2.set_ylabel("heuristic_evals")
        ax2.grid(axis="y", alpha=0.3)

        fig.suptitle("Experiment 4: Memoization Ablation (mean ± std)")
        self._save_plot(fig, "exp4_ablation_memoization.png")

    def _plot_exp5(self) -> None:
        rows = self._summary_lookup(experiment="exp5_obstacle_density")
        if not rows:
            return

        densities = sorted({float(r["obstacle_prob"]) for r in rows})
        conds = sorted({str(r["condition"]) for r in rows})

        fig, ax = plt.subplots(figsize=(7, 4))
        for cond in conds:
            ys = []
            yerr = []
            for p in densities:
                r = next((x for x in rows if str(x["condition"]) == cond and float(x["obstacle_prob"]) == p), None)
                if r is None:
                    continue
                ys.append(float(r["time_mean"]))
                yerr.append(float(r["time_std"]))
            ax.errorbar(densities, ys, yerr=yerr, marker="o", label=cond)

        ax.set_title("Experiment 5: Density vs Time")
        ax.set_xlabel("obstacle probability")
        ax.set_ylabel("seconds")
        ax.grid(alpha=0.3)
        ax.legend()
        self._save_plot(fig, "exp5_density_time.png")

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        for cond in conds:
            ys = []
            yerr = []
            for p in densities:
                r = next((x for x in rows if str(x["condition"]) == cond and float(x["obstacle_prob"]) == p), None)
                if r is None:
                    continue
                ys.append(float(r["nodes_mean"]))
                yerr.append(float(r["nodes_std"]))
            ax2.errorbar(densities, ys, yerr=yerr, marker="o", label=cond)

        ax2.set_title("Experiment 5: Density vs Nodes")
        ax2.set_xlabel("obstacle probability")
        ax2.set_ylabel("nodes expanded")
        ax2.grid(alpha=0.3)
        ax2.legend()
        self._save_plot(fig2, "exp5_density_nodes.png")

    def _plot_exp6(self) -> None:
        rows = self._summary_lookup(experiment="exp6_heuristic_comparison")
        if not rows:
            return

        labels = [str(r["condition"]) for r in rows]
        t = [float(r["time_mean"]) for r in rows]
        terr = [float(r["time_std"]) for r in rows]
        n = [float(r["nodes_mean"]) for r in rows]
        nerr = [float(r["nodes_std"]) for r in rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(labels, t, yerr=terr, capsize=6)
        ax1.set_title("Exp6: Time")
        ax1.set_ylabel("seconds")
        ax1.grid(axis="y", alpha=0.3)

        ax2.bar(labels, n, yerr=nerr, capsize=6)
        ax2.set_title("Exp6: Nodes")
        ax2.set_ylabel("nodes expanded")
        ax2.grid(axis="y", alpha=0.3)

        fig.suptitle("Experiment 6: Heuristic Comparison (mean ± std)")
        self._save_plot(fig, "exp6_heuristic_comparison.png")

    def _plot_exp7(self) -> None:
        rows = self._summary_lookup(experiment="exp7_tiebreaking")
        if not rows:
            return

        labels = [str(r["condition"]) for r in rows]
        t = [float(r["time_mean"]) for r in rows]
        terr = [float(r["time_std"]) for r in rows]
        n = [float(r["nodes_mean"]) for r in rows]
        nerr = [float(r["nodes_std"]) for r in rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(labels, t, yerr=terr, capsize=6)
        ax1.set_title("Exp7: Time")
        ax1.set_ylabel("seconds")
        ax1.grid(axis="y", alpha=0.3)

        ax2.bar(labels, n, yerr=nerr, capsize=6)
        ax2.set_title("Exp7: Nodes")
        ax2.set_ylabel("nodes expanded")
        ax2.grid(axis="y", alpha=0.3)

        fig.suptitle("Experiment 7: Tie-breaking On/Off (mean ± std)")
        self._save_plot(fig, "exp7_tiebreaking.png")

    # ----------------------- Analysis + Notes -----------------------

    def _best_configuration_from_exp1(self) -> Optional[Dict[str, object]]:
        rows = [r for r in self.summary_rows if str(r["experiment"]) == "exp1_baseline_vs_enhanced"]
        if not rows:
            return None

        # Score = normalized time + normalized nodes (lower is better)
        times = [float(r["time_mean"]) for r in rows]
        nodes = [float(r["nodes_mean"]) for r in rows]
        tmin, tmax = min(times), max(times)
        nmin, nmax = min(nodes), max(nodes)

        def norm(x: float, lo: float, hi: float) -> float:
            if hi == lo:
                return 0.0
            return (x - lo) / (hi - lo)

        best = None
        best_score = float("inf")
        for r in rows:
            score = norm(float(r["time_mean"]), tmin, tmax) + norm(float(r["nodes_mean"]), nmin, nmax)
            if score < best_score:
                best_score = score
                best = r
        return best

    def write_summary_text(self) -> None:
        """Write final_results/logs/summary.txt with key findings."""

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        out = self.logs_dir / "summary.txt"

        best = self._best_configuration_from_exp1()
        lines: List[str] = []
        lines.append("CS572 Final Project — Automated Summary")
        lines.append("")
        lines.append("Outputs:")
        lines.append(f"- CSV runs: {self.csv_dir / 'final_runs.csv'}")
        lines.append(f"- CSV summary: {self.csv_dir / 'final_results.csv'}")
        lines.append(f"- Plots: {self.plots_dir}")
        lines.append("")

        if best is not None:
            lines.append("Best Configuration (from Exp1, normalized time+nodes):")
            lines.append(f"- condition: {best['condition']}")
            lines.append(f"- time_mean: {float(best['time_mean']):.6f} s")
            lines.append(f"- nodes_mean: {float(best['nodes_mean']):.1f}")
            if str(best["condition"]) == "enhanced_combo":
                lines.append("- Best Configuration: WA* + memoization + hybrid heuristic + tie-breaking")
            lines.append("")

        # Trade-off summary for Exp3: compare w=1.0 vs best-by-time
        exp3 = [r for r in self.summary_rows if str(r["experiment"]) == "exp3_weight_sensitivity" and int(r["input_size"]) == 40]
        if exp3:
            def parse_w(cond: str) -> Optional[float]:
                return float(cond.split('=')[1]) if cond.startswith('w=') else None

            exp3_pairs = [(parse_w(str(r["condition"])), r) for r in exp3]
            exp3_pairs = [(w, r) for w, r in exp3_pairs if w is not None]
            if exp3_pairs:
                best_w, best_r = min(exp3_pairs, key=lambda t: float(t[1]["time_mean"]))
                w1 = next((r for w, r in exp3_pairs if w == 1.0), None)
                lines.append("Trade-off (Exp3):")
                lines.append(f"- best weight by time: w={best_w:.2f} (time_mean={float(best_r['time_mean']):.6f}s, path_len_mean={float(best_r['path_len_mean']):.1f})")
                if w1 is not None:
                    lines.append(f"- vs w=1.0: time {float(w1['time_mean']):.6f}s -> {float(best_r['time_mean']):.6f}s, path {float(w1['path_len_mean']):.1f} -> {float(best_r['path_len_mean']):.1f}")
                lines.append("")

        # Simple anomaly checks: highlight conditions that regress vs baseline.
        threshold_pct = 10.0
        anomalies: List[str] = []
        experiments = sorted({str(r["experiment"]) for r in self.summary_rows})
        for exp in experiments:
            exp_rows = [r for r in self.summary_rows if str(r["experiment"]) == exp]
            conditions = sorted({str(r["condition"]) for r in exp_rows})
            sizes = sorted({int(r["input_size"]) for r in exp_rows})
            probs = sorted({float(r["obstacle_prob"]) for r in exp_rows})

            baseline_cond = next((c for c in conditions if "baseline" in c), conditions[0] if conditions else None)
            if baseline_cond is None:
                continue

            for size in sizes:
                for p in probs:
                    base = next(
                        (x for x in exp_rows if str(x["condition"]) == baseline_cond and int(x["input_size"]) == size and float(x["obstacle_prob"]) == p),
                        None,
                    )
                    if base is None:
                        continue
                    base_time = float(base["time_mean"])
                    base_nodes = float(base["nodes_mean"])
                    if base_time <= 0 or base_nodes <= 0:
                        continue

                    for r in exp_rows:
                        if int(r["input_size"]) != size or float(r["obstacle_prob"]) != p:
                            continue
                        cond = str(r["condition"])
                        if cond == baseline_cond:
                            continue
                        tmean = float(r["time_mean"])
                        nmean = float(r["nodes_mean"])
                        time_worse = tmean > base_time * (1.0 + threshold_pct / 100.0)
                        nodes_worse = nmean > base_nodes * (1.0 + threshold_pct / 100.0)
                        if time_worse or nodes_worse:
                            dt = 100.0 * (tmean - base_time) / base_time
                            dn = 100.0 * (nmean - base_nodes) / base_nodes
                            anomalies.append(
                                f"- {exp} size={size} p={p:.2f} cond={cond}: time {dt:+.1f}%, nodes {dn:+.1f}%"
                            )

        lines.append(f"Anomaly checks (>{threshold_pct:.0f}% worse than baseline):")
        if anomalies:
            lines.extend(anomalies[:40])
            if len(anomalies) > 40:
                lines.append(f"- ... ({len(anomalies) - 40} more)")
        else:
            lines.append("- none")

        out.write_text("\n".join(lines))

    def write_report_notes(self) -> None:
        """Generate report_final_notes.md (support file for final report)."""

        best = self._best_configuration_from_exp1()
        md: List[str] = []
        md.append("# CS572 Final Project — Report Notes (Auto-generated)")
        md.append("")
        md.append("## Enhancements Implemented (>=3)")
        md.append("1) Tie-breaking: on f(n) ties, prefer lower h(n) (toggle: use_tiebreaking).")
        md.append("2) Multi-heuristic system: Manhattan, Euclidean, Hybrid (toggle: heuristic; hybrid uses alpha).")
        md.append("3) Density-adaptive strategy: switch between A* and WA* / increase WA* weight based on obstacle density (toggle: use_density_adaptive).")
        md.append("")
        md.append("## Experiments (>=5, n>=30 per condition)")
        md.append("- Exp1: Baseline vs Enhanced (40x40, p=0.22)")
        md.append("- Exp2: Scaling (20,40,60,80,100)")
        md.append("- Exp3: Weight sensitivity (w ∈ {1.0,1.2,1.4,1.6,2.0,2.5})")
        md.append("- Exp4: Memoization ablation (on/off)")
        md.append("- Exp5: Obstacle density analysis (p ∈ {0.10,0.20,0.30,0.40})")
        md.append("- Exp6: Heuristic comparison (Manhattan vs Euclidean vs Hybrid)")
        md.append("- Exp7: Tie-breaking analysis (on/off)")
        md.append("")
        md.append("## Key Findings (populate from final_results/csv/final_results.csv)")
        if best is not None:
            md.append(f"- Best configuration (Exp1, normalized time+nodes): **{best['condition']}**")
            md.append(f"  - time_mean: {float(best['time_mean']):.6f} s")
            md.append(f"  - nodes_mean: {float(best['nodes_mean']):.1f}")
        md.append("- See final_results/logs/summary.txt for the automated trade-off summary.")
        md.append("")
        md.append("## Reproducibility")
        md.append("- Deterministic seeds are used for all runs.")
        md.append("- Run the full suite with: `python3 main.py --final`.")
        md.append("- Run a single experiment with: `python3 main.py --experiment exp2` (for example).")
        md.append("")
        md.append("## Limitations")
        md.append("- Runtime measurements on very small instances can be noisy; nodes expanded is often a more stable indicator.")
        md.append("- WA* can return longer paths for w>1; this is measured via path_length.")
        md.append("")
        md.append("## Future Work")
        md.append("- Add additional map models (maze-like, clustered obstacles) and obstacle densities.")
        md.append("- Add more heuristics or landmark-based heuristics for stronger improvements.")

        (project_root() / "report_final_notes.md").write_text("\n".join(md) + "\n")


def run_final_suite(*, runs: int = 30, base_seed: int = 1337, which: Optional[str] = None) -> FinalExperimentRunner:
    """Run the final suite (or a single experiment) and write outputs."""

    if runs < 30:
        raise ValueError("Final rubric requires runs >= 30")

    runner = FinalExperimentRunner(base_seed=base_seed)

    def run_one(name: str) -> None:
        # Accept both exp ids and friendly aliases.
        aliases = {
            "baseline": "exp1",
            "baseline_vs_enhanced": "exp1",
            "scaling": "exp2",
            "weight": "exp3",
            "sensitivity": "exp3",
            "ablation": "exp4",
            "memoization": "exp4",
            "density": "exp5",
            "obstacle_density": "exp5",
            "heuristic": "exp6",
            "heuristics": "exp6",
            "tiebreak": "exp7",
            "tiebreaking": "exp7",
        }
        name = aliases.get(name, name)

        if name == "exp1":
            runner.exp1_baseline_vs_enhanced(runs=runs)
        elif name == "exp2":
            runner.exp2_scaling(runs=runs)
        elif name == "exp3":
            runner.exp3_weight_sensitivity(runs=runs)
        elif name == "exp4":
            runner.exp4_ablation_memoization(runs=runs)
        elif name == "exp5":
            runner.exp5_obstacle_density(runs=runs)
        elif name == "exp6":
            runner.exp6_heuristic_comparison(runs=runs)
        elif name == "exp7":
            runner.exp7_tiebreaking(runs=runs)
        else:
            raise ValueError("Unknown experiment. Use exp1..exp7")

    if which is None or which in {"all", "final"}:
        runner.exp1_baseline_vs_enhanced(runs=runs)
        runner.exp2_scaling(runs=runs)
        runner.exp3_weight_sensitivity(runs=runs)
        runner.exp4_ablation_memoization(runs=runs)
        runner.exp5_obstacle_density(runs=runs)
        runner.exp6_heuristic_comparison(runs=runs)
        runner.exp7_tiebreaking(runs=runs)
    else:
        run_one(which)

    runner.build_summaries()
    runner.write_csvs()
    runner.plot_all()
    runner.write_summary_text()
    runner.write_report_notes()

    return runner
