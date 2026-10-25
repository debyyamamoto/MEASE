import os
import argparse
from pathlib import Path
import pandas as pd
from easd.runner import RunConfig, run_dataset

FACTORIAL_PROJECTS_LIST = [
    "experiments/factorial_design/designs/factorial_population.csv",
    "experiments/factorial_design/designs/factorial_generations.csv",
    "experiments/factorial_design/designs/factorial_restart_gen.csv",
    "experiments/factorial_design/designs/factorial_restart_pop.csv",
    "experiments/factorial_design/designs/factorial_restart_pct.csv",
    "experiments/factorial_design/designs/factorial_alpha.csv",
    "experiments/factorial_design/designs/factorial_ksize.csv",
    "experiments/factorial_design/designs/factorial_threshold.csv",
    "experiments/factorial_design/designs/factorial_score_metric.csv",
    "experiments/factorial_design/designs/factorial_redundancy_penalty.csv",
    "experiments/factorial_design/designs/factorial_redundancy_similarity_threshold.csv",
    "experiments/factorial_design/designs/factorial_min_relative_supp.csv",
    "experiments/factorial_design/designs/factorial_max_relative_supp.csv",
]
EXPERIMENT_NAME_COLUMN = "experiment"
RESPONSE_VAR_COLUMN = "response"
FACTOR_COLUMN = "factor"
POPULATION_SIZE_COLUMN = "population"
DATASET_COLUMN = "dataset"
GENERATIONS_COLUMN = "generations"
RATE_POLICY_COLUMN = "rate_policy"
KSIZE_COLUMN = "ksize"
OUTPUT_DIR = "./experiments/factorial_design/factorial_2k"
OPTIONAL_CONFIG_COLUMNS = {
    "generations",
    "population",
    "restart_gen",
    "restart_pop",
    "restart_pct",
    "comparacao",
    "alpha",
    "ksize",
    "plot_rank",
    "threshold",
    "debug_performance",
    "rate_policy",
    "score_metric",
    "km_time_bins",
    "redundancy_penalty",
    "redundancy_similarity_threshold",
    "archive_selection",
    "min_relative_supp",
    "max_relative_supp",
}
DEFAULT_CONFIG = {
    "executions": 30,
    "generations": 500,
    "population": 800,
    "restart_gen": 3,
    "restart_pop": 3,
    "restart_pct": 0.60,
    "comparacao": "complement",
    "alpha": 0.30,
    "ksize": 5,
    "plot_rank": 10,
    "threshold": 0.30,
    "debug_performance": False,
    "rate_policy": "adaptive",
    "score_metric": "fast_logrank",
    "km_time_bins": 512,
    "redundancy_penalty": 0.50,
    "redundancy_similarity_threshold": 0.50,
    "archive_selection": "greedy",
    "min_relative_supp": 0.05,
    "max_relative_supp": 0.45,
}


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def config_from_args(p_args: dict) -> RunConfig:
    if p_args["label_col"] == "":
        p_args["label_col"] = None
    return RunConfig(
        filepath=Path(p_args["filepath"]),
        time_col=p_args["time_col"],
        label_col=p_args["label_col"],
        event_col=p_args["event_col"],
        output_dir=Path(p_args["output_dir"]),
        dataset_name=p_args["dataset_name"],
        executions=p_args["executions"],
        generations=p_args["generations"],
        population=p_args["population"],
        restart_gen=p_args["restart_gen"],
        restart_pop=p_args["restart_pop"],
        restart_pct=p_args["restart_pct"],
        comparacao=p_args["comparacao"],
        alpha=p_args["alpha"],
        ksize=p_args["ksize"],
        plot_rank=p_args["plot_rank"],
        threshold=p_args["threshold"],
        debug_performance=False,
        rate_policy=p_args["rate_policy"],
        score_metric=p_args["score_metric"],
        km_time_bins=p_args["km_time_bins"],
        redundancy_penalty=p_args["redundancy_penalty"],
        redundancy_similarity_threshold=p_args["redundancy_similarity_threshold"],
        archive_selection=p_args["archive_selection"],
        min_relative_supp=p_args["min_relative_supp"],
        max_relative_supp=p_args["max_relative_supp"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MEASE 2^k factorial designs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--designs", nargs="+", type=Path, default=[Path(path) for path in FACTORIAL_PROJECTS_LIST])
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--executions", type=int, default=DEFAULT_CONFIG["executions"])
    return parser


def row_value(experiment: pd.Series, column: str, default):
    if column not in experiment.index or pd.isna(experiment[column]):
        return default
    return experiment[column]


def run_factorial_designs(designs: list[Path], output_dir: str) -> None:
    for project in designs:
        exp_df = pd.read_csv(project)
        response_var = exp_df.loc[0, RESPONSE_VAR_COLUMN]
        experiment_name = exp_df.loc[0, EXPERIMENT_NAME_COLUMN]

        for _, experiment in exp_df.iterrows():
            run_arguments = dict(DEFAULT_CONFIG)
            dataset_name, file_path, time_column, event_column, label_column = str(experiment[DATASET_COLUMN]).split(
                "|"
            )

            if os.path.exists(
                f"{output_dir}/{experiment_name}/{response_var}/{experiment[FACTOR_COLUMN]}/{dataset_name}/complement/{dataset_name}_complement_RulesMetricsResult.csv"
            ):
                continue

            run_arguments["config"] = experiment[FACTOR_COLUMN]
            run_arguments["filepath"] = file_path
            run_arguments["time_col"] = time_column
            run_arguments["event_col"] = event_column
            run_arguments["label_col"] = label_column
            run_arguments["dataset_name"] = dataset_name
            run_arguments["output_dir"] = f"{output_dir}/{experiment_name}/{response_var}/{experiment[FACTOR_COLUMN]}"

            for column in OPTIONAL_CONFIG_COLUMNS:
                run_arguments[column] = row_value(experiment, column, run_arguments[column])

            configs = config_from_args(run_arguments)
            run_dataset(configs)


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_factorial_designs(args.designs, args.output_dir)
