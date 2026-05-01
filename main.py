"""CS572 Milestone 2: Heuristic Problem Solving Project (production-ready).

Single-command automation entrypoint:
    python main.py

Runs:
- correctness tests + baseline validation summary
- baseline A* and enhanced Weighted A* variants
- 4 experiments (each >=10 runs)
- raw + summary CSV data collection
- publication-quality matplotlib plots saved under results/plots/
"""

from __future__ import annotations

import argparse

from src.experiment_runner import GridSpec, default_experiments
from src.utils import ensure_dirs, project_root
from tests.test_cases import run_all_tests, run_validation_suite


def run_milestone_pipeline() -> None:
    """Run the original Milestone 2 pipeline (kept for backwards compatibility)."""

    ensure_dirs()

    # 1) Correctness tests
    run_all_tests()
    print("[OK] All tests passed")

    correctness = run_validation_suite()
    print("[OK] Baseline validation:")
    print(f"- Small grids validated (A* == BFS optimal): {correctness['small_grids_validated']}")

    # 2) Experiments
    runner = default_experiments()

    # Experiment 1: baseline vs enhanced on a fixed medium grid size
    exp1_spec = GridSpec(size=40, obstacle_prob=0.22)
    exp1_summary = runner.run_experiment_1_baseline_vs_enhanced(
        grid_spec=exp1_spec,
        runs=10,
        enhanced_weight=2.2,
        use_dynamic_weight=True,
        seed0=111,
    )

    print("\nExperiment 1 (Baseline vs Enhanced) Summary:")
    print(
        f"- Baseline time: {exp1_summary['baseline_time_mean']:.6f} ± {exp1_summary['baseline_time_std']:.6f} s "
        f"(CI95 ± {exp1_summary['baseline_time_ci95']:.6f})"
    )
    print(
        f"- Enhanced time: {exp1_summary['enhanced_time_mean']:.6f} ± {exp1_summary['enhanced_time_std']:.6f} s "
        f"(CI95 ± {exp1_summary['enhanced_time_ci95']:.6f})"
    )
    print(f"- Time improvement: {exp1_summary['time_improvement_pct']:.2f}%")
    print(f"- Baseline nodes:  {exp1_summary['baseline_nodes_mean']:.1f} ± {exp1_summary['baseline_nodes_std']:.1f}")
    print(f"- Enhanced nodes:  {exp1_summary['enhanced_nodes_mean']:.1f} ± {exp1_summary['enhanced_nodes_std']:.1f}")
    print(f"- Nodes improvement: {exp1_summary['nodes_improvement_pct']:.2f}%")

    # Experiment 2: scaling analysis (small/medium/large)
    sizes = [20, 40, 60]
    runner.run_experiment_2_scaling(
        sizes=sizes,
        obstacle_prob=0.22,
        runs=10,
        enhanced_weight=2.2,
        use_dynamic_weight=True,
        seed0=222,
    )

    # Experiment 3: parameter sensitivity (weight)
    runner.run_experiment_3_parameter_sensitivity(
        grid_spec=GridSpec(size=40, obstacle_prob=0.22),
        weights=[1.0, 1.2, 1.4, 1.6, 2.0, 2.5],
        runs=10,
        seed0=333,
    )

    # Experiment 4: ablation study (memoization on/off)
    runner.run_experiment_4_ablation_memoization(
        grid_spec=GridSpec(size=40, obstacle_prob=0.22),
        runs=10,
        weight=1.6,
        seed0=444,
    )

    # 3) Write CSVs (raw + summary)
    runner.write_csv()
    print(f"\n[OK] Wrote results CSV -> {runner.results_csv}")

    runner.build_summaries()
    summary_csv = project_root() / "data" / "summary.csv"
    runner.write_summary_csv(summary_csv)
    print(f"[OK] Wrote summary CSV -> {summary_csv}")

    # 4) Plots
    plots_dir = project_root() / "results" / "plots"
    runner.plot_experiment_1_bar(grid_size=40, out_path=plots_dir / "exp1_baseline_vs_enhanced.png")
    runner.plot_experiment_2_scaling_lines(sizes=sizes, out_path=plots_dir / "exp2_scaling_time.png")
    runner.plot_experiment_3_sensitivity(grid_size=40, out_path=plots_dir / "exp3_weight_sensitivity.png")
    runner.plot_improvement_vs_size(sizes=sizes, out_path=plots_dir / "exp2_improvement_vs_size.png")

    print(f"[OK] Saved plots under -> {plots_dir}")

    # 5) Console insights
    best = runner.best_weight_by_time(grid_size=40)
    if best:
        print("\nPerformance Insights:")
        avg_impr = runner.avg_improvement_across_sizes(sizes=sizes)
        print(
            f"- Avg improvement across sizes (dynamic WA* vs A*): "
            f"time {avg_impr['avg_time_improvement_pct']:.2f}%, nodes {avg_impr['avg_nodes_improvement_pct']:.2f}%"
        )
        print(f"- Best WA* weight by mean time (Exp3): w={best['best_weight']:.2f}")
        print(f"  - mean time: {best['mean_time']:.6f} s")
        print(f"  - mean nodes: {best['mean_nodes']:.1f}")
        print(f"  - mean path length: {best['mean_path_length']:.1f}")

        w1 = runner.wa_weight_stats(grid_size=40, weight=1.0)
        if w1:
            print("  - tradeoff vs w=1.0:")
            print(f"    - time: {w1['mean_time']:.6f} -> {best['mean_time']:.6f} s")
            print(f"    - path length: {w1['mean_path_length']:.1f} -> {best['mean_path_length']:.1f}")

    ab = runner.ablation_memoization_summary(grid_size=40)
    if ab:
        print("- Ablation (memoization):")
        print(f"  - time improvement vs no-memo: {ab['time_improvement_pct']:.2f}%")
        print(f"  - nodes improvement vs no-memo: {ab['nodes_improvement_pct']:.2f}%")
        print(f"  - heuristic eval reduction vs no-memo: {ab['heuristic_evals_reduction_pct']:.2f}%")


def run_final_pipeline(*, runs: int = 30, seed: int = 1337, which: str = "final") -> None:
    """Run the Final Project pipeline (writes outputs under final_results/)."""

    from src.experiments.final_runner import run_final_suite

    run_final_suite(runs=runs, base_seed=seed, which=which)
    root = project_root() / "final_results"
    print("[OK] Final pipeline complete")
    print(f"- CSV summary: {root / 'csv' / 'final_results.csv'}")
    print(f"- CSV runs:    {root / 'csv' / 'final_runs.csv'}")
    print(f"- Plots:       {root / 'plots'}")
    print(f"- Summary:     {root / 'logs' / 'summary.txt'}")
    print(f"- Notes:       {project_root() / 'report_final_notes.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CS572 Heuristic Problem Solving (Milestone 2 + Final Project)")
    parser.add_argument("--final", action="store_true", help="Run the full final-project suite (>=5 experiments, n>=30)")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Run a single final experiment: exp1|exp2|exp3|exp4|exp5|exp6|exp7",
    )
    parser.add_argument("--runs", type=int, default=30, help="Runs per condition (final rubric requires >=30)")
    parser.add_argument("--seed", type=int, default=1337, help="Base seed for reproducibility")
    args = parser.parse_args()

    if args.final or args.experiment is not None:
        which = args.experiment if args.experiment is not None else "final"
        run_final_pipeline(runs=int(args.runs), seed=int(args.seed), which=str(which))
        return

    run_milestone_pipeline()


if __name__ == "__main__":
    main()
