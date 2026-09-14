import sys
import argparse
from pathlib import Path
from easd.evaluation import SCORE_METRICS
from easd.runner import RunConfig, run_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MEASE on a parquet survival dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filepath", type=Path, help="Path to a parquet dataset.")
    parser.add_argument("-time", "--time_col", required=True, help="Survival time column name.")
    parser.add_argument("-event", "--event_col", required=True, help="Event/censoring status column name.")
    parser.add_argument(
        "-label",
        "--subgroup_label_col",
        type=str,
        required=False,
        default=None,
        help="Subgroup label column name. If subgroups are known, the F1-Score is calculated",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results"), help="Directory for generated outputs.")
    parser.add_argument("--dataset_name", default=None, help="Optional name used in output files.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility.")
    parser.add_argument("-th", "--threshold", type=float, default=0.9, help="Jaccard similarity threshold.")
    parser.add_argument("-g", "--generations", type=int, default=500, help="Maximum number of generations.")
    parser.add_argument("-p", "--population", type=int, default=500, help="Population size.")
    parser.add_argument("--restart_gen", type=int, default=3, help="Generation limit without improvement.")
    parser.add_argument("--restart_pop", type=int, default=3, help="Population restart limit.")
    parser.add_argument("--restart_pct", type=float, default=0.10, help="Population percentage restarted each time.")
    parser.add_argument(
        "-comp",
        "--comparacao",
        choices=["complement", "population"],
        default="complement",
        help="Baseline group for the log-rank test.",
    )
    parser.add_argument("-a", "--alpha", type=float, default=0.5, help="Fitness alpha weight.")
    parser.add_argument(
        "--score_metric",
        choices=SCORE_METRICS,
        default="legacy_logrank",
        help=(
            "Fitness discrepancy metric. 'legacy_logrank' uses statsmodels survdiff; "
            "'fast_logrank' uses a precomputed log-rank proxy; 'km_cvm' and 'km_abc' "
            "use weighted Kaplan-Meier curve distances; 'mdir2', 'mdir3' and 'mdir4' "
            "use scalable multiple-direction weighted log-rank proxies."
        ),
    )
    parser.add_argument(
        "--km_time_bins",
        type=int,
        default=512,
        help="Maximum time grid size for km_cvm/km_abc/mdir*. Use 0 for exact event-time grid.",
    )
    parser.add_argument(
        "--redundancy_penalty",
        type=float,
        default=0.0,
        help="Strength of the Top-K survival-curve similarity penalty in [0, 1].",
    )
    parser.add_argument(
        "--redundancy_similarity_threshold",
        type=float,
        default=0.5,
        help="Penalize candidates whose Kaplan-Meier curve similarity exceeds this value.",
    )
    parser.add_argument(
        "--rate_policy",
        choices=["adaptive", "fixed"],
        default="adaptive",
        help="Crossover/mutation policy. Fixed uses crossover=60%% and mutation=40%%.",
    )
    parser.add_argument("-exe", "--executions", type=int, default=1, help="Independent algorithm executions.")
    parser.add_argument("-k", "--ksize", type=int, default=10, help="Top-K rule rank size.")
    parser.add_argument(
        "-plt",
        "--plt_rank",
        type=int,
        default=0,
        help="Save plots for the top-N rules. Only enabled for a single execution.",
    )
    parser.add_argument(
        "-d",
        "--debug_performance",
        choices=["on", "off"],
        default="off",
        help="Collect CPU/RAM measurements.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        filepath=args.filepath,
        time_col=args.time_col,
        event_col=args.event_col,
        label_col=args.subgroup_label_col,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        seed=args.seed,
        executions=args.executions,
        generations=args.generations,
        population=args.population,
        restart_gen=args.restart_gen,
        restart_pop=args.restart_pop,
        restart_pct=args.restart_pct,
        comparacao=args.comparacao,
        alpha=args.alpha,
        score_metric=args.score_metric,
        km_time_bins=None if args.km_time_bins <= 0 else args.km_time_bins,
        redundancy_penalty=args.redundancy_penalty,
        redundancy_similarity_threshold=args.redundancy_similarity_threshold,
        ksize=args.ksize,
        plot_rank=args.plt_rank,
        threshold=args.threshold,
        debug_performance=args.debug_performance == "on",
        rate_policy=args.rate_policy,
    )


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run_dataset(config_from_args(args))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Results saved to {summary.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
