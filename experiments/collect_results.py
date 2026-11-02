from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_DATASETS = ["breast-cancer", "cancer", "carcinoma", "lung", "mgus2", "veteran"]
METRICS = [
    "exceptionality",
    "#sg",
    "length",
    "sgCov",
    "setCov",
    "description redundancy",
    "coverage redundancy",
    "CR",
    "model redundancy",
]
OPTIONAL_METRICS = ["mean_pairwise_rms", "rms_valid_pairs"]


def compile_metric_runs(
    results_dir: Path,
    datasets: list[str],
    executions: int = 30,
    baseline: str = "population",
    output_file: Path | None = None,
) -> pd.DataFrame:
    rows = []

    for dataset in datasets:
        for execution in range(executions):
            metrics_file = (
                results_dir / dataset / baseline / f"{dataset}_{execution}_{baseline}_RulesMetricsResult.csv"
            )
            if not metrics_file.exists():
                print(f"Missing result file: {metrics_file}")
                continue

            metrics_df = pd.read_csv(metrics_file)
            if metrics_df.empty:
                print(f"Empty result file: {metrics_file}")
                continue

            row = {"Dataset": dataset, "Execucao": execution}
            metrics = metrics_df.iloc[-1][METRICS].to_dict()
            # Historical CSVs predate RMS; missing measurements stay missing.
            metrics.update(metrics_df.iloc[-1].reindex(OPTIONAL_METRICS).to_dict())
            for k, v in metrics.items():
                row[str(k)] = v
            rows.append(row)

    output_file = output_file or Path(__file__).resolve().parent / f"resultados_{baseline}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} rows to {output_file}")
    return df


def compile_performance_stats(results_dir: Path, datasets: list[str], baseline: str = "population") -> pd.DataFrame:
    frames = []

    for dataset in datasets:
        stats_file = results_dir / dataset / baseline / f"{dataset}_{baseline}_RulesStatsResult.csv"
        if not stats_file.exists():
            print(f"Missing stats file: {stats_file}")
            continue
        frames.append(pd.read_csv(stats_file).assign(Dataset=dataset))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True).set_index("Dataset")
    print(df.style.to_latex(caption="Performance Results", label="tab:performance", column_format="lc"))
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile per-run MEASE metric CSV files into a single table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--baseline", choices=["complement", "population"], default="population")
    parser.add_argument("-exe", "--executions", type=int, default=30)
    parser.add_argument("--output_file", type=Path, default=None)
    parser.add_argument("--performance", action="store_true", help="Print aggregate performance stats instead.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.performance:
        compile_performance_stats(args.results_dir, args.datasets, args.baseline)
    else:
        compile_metric_runs(args.results_dir, args.datasets, args.executions, args.baseline, args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
