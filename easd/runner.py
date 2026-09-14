from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
from easd.core import MEASE
from easd.metrics import compute_run_metrics, output_metrics, calculate_max_f1_score

matplotlib.use("Agg")

BASELINE = Literal["complement", "population"]
RATEPOLICY = Literal["adaptive", "fixed"]


@dataclass(frozen=True)
class RunConfig:
    filepath: Path
    time_col: str
    event_col: str
    label_col: str | None
    output_dir: Path = Path("results")
    dataset_name: str | None = None
    seed: int | None = None
    executions: int = 1
    generations: int = 500
    population: int = 500
    restart_gen: int = 3
    restart_pop: int = 3
    restart_pct: float = 0.10
    comparacao: BASELINE = "complement"
    alpha: float = 0.5
    ksize: int = 10
    plot_rank: int = 0
    threshold: float = 0.9
    debug_performance: bool = False
    rate_policy: RATEPOLICY = "adaptive"
    score_metric: str = "legacy_logrank"
    km_time_bins: int | None = 512
    redundancy_penalty: float = 0.0
    redundancy_similarity_threshold: float = 0.9


@dataclass(frozen=True)
class RunSummary:
    dataset_name: str
    baseline: BASELINE
    output_dir: Path
    executions: int
    run_metrics: list[dict]
    stats_file: Path | None = None
    metrics_file: Path | None = None


def run_dataset(config: RunConfig) -> RunSummary:
    data = _read_dataset(config.filepath)
    if config.label_col is not None:
        subgroups_labels = data[config.label_col].to_numpy()
        subgroups_labels: np.ndarray
        data = data.drop(columns=[config.label_col])
    else:
        subgroups_labels = np.array([])

    dataset_name = config.dataset_name or config.filepath.stem
    output_dir = config.output_dir / dataset_name / config.comparacao
    output_dir.mkdir(parents=True, exist_ok=True)

    num_plots = config.plot_rank if config.executions == 1 else 0
    state = _AggregateState()
    figures = []
    metrics_list: list[dict] = []

    for run in range(config.executions):
        seed = config.seed if config.seed is not None else run
        print(f"--- Running dataset {dataset_name} ({run + 1}/{config.executions}) ---")

        sd = MEASE(
            data.copy(),
            config.time_col,
            config.event_col,
            max_generations=config.generations,
            population_size=config.population,
            max_generations_no_improve=config.restart_gen,
            max_pop_restarts=config.restart_pop,
            restart_percentage=config.restart_pct,
            seed_val=seed,
            comparacao=config.comparacao,
            alpha=config.alpha,
            ksize=config.ksize,
            plot_n_rules=num_plots,
            coverage_threshold=config.threshold,
            debug_performance=config.debug_performance,
            rate_policy=config.rate_policy,
            score_metric=config.score_metric,
            km_time_bins=config.km_time_bins,
            redundancy_penalty=config.redundancy_penalty,
            redundancy_similarity_threshold=config.redundancy_similarity_threshold,
        )

        _, _, _, runtime, _, info, detailed_rules, top_rules, mean_rule_size, figures = sd.run()
        run_metrics = compute_run_metrics(
            sd.dataset_obj.get_data(),
            top_rules,
            time_col=config.time_col,
            event_col=config.event_col,
            dataset_obj=sd.dataset_obj,
            baseline=config.comparacao,
        ).as_dict()
        run_metrics.update(_topk_rule_score_metrics(detailed_rules))
        metrics_list.append(run_metrics)
        if config.label_col is not None:
            max_f1_score = calculate_max_f1_score(data, top_rules, subgroups_labels, sd.dataset_obj)
            run_metrics.update({"max_f1_score": max_f1_score})

        _save_run_outputs(
            dataset_name=dataset_name,
            baseline=config.comparacao,
            run=run,
            detailed_rules=detailed_rules,
            runtime=float(runtime),
            mean_rule_size=float(mean_rule_size),
            info=info,
            metrics=run_metrics,
            output_dir=output_dir,
            aggregate_state=state,
            debug_performance=config.debug_performance,
        )

    stats_file = None
    metrics_file = None
    if config.executions > 1:
        stats_file, metrics_file = _save_aggregate_outputs(
            dataset_name=dataset_name,
            baseline=config.comparacao,
            metrics_list=metrics_list,
            output_dir=output_dir,
            aggregate_state=state,
            debug_performance=config.debug_performance,
        )

    _save_figures(figures, output_dir, config.plot_rank)

    return RunSummary(
        dataset_name=dataset_name,
        baseline=config.comparacao,
        output_dir=output_dir,
        executions=config.executions,
        run_metrics=metrics_list,
        stats_file=stats_file,
        metrics_file=metrics_file,
    )


@dataclass
class _AggregateState:
    scores: list[np.ndarray] = field(default_factory=list)
    runtimes: list[float] = field(default_factory=list)
    mean_rule_sizes: list[float] = field(default_factory=list)
    cpu_mean_percent: list[float] = field(default_factory=list)
    cpu_peak_percent: list[float] = field(default_factory=list)
    ram_mean_mb: list[float] = field(default_factory=list)
    ram_peak_mb: list[float] = field(default_factory=list)
    ram_incremental_peak_mb: list[float] = field(default_factory=list)
    ram_baseline_mb: list[float] = field(default_factory=list)


def _read_dataset(filepath: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(filepath)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset file not found: {filepath}") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not read dataset {filepath}: {exc}") from exc


def _topk_rule_score_metrics(detailed_rules: pd.DataFrame) -> dict[str, float]:
    if detailed_rules is None or detailed_rules.empty or "Rule_Score" not in detailed_rules.columns:
        return {"mean_rule_score": 0.0}

    scores = pd.to_numeric(detailed_rules["Rule_Score"], errors="coerce").dropna()
    if scores.empty:
        return {"mean_rule_score": 0.0}

    metrics = {"mean_rule_score": float(scores.mean())}
    if "Penalized_Rule_Score" in detailed_rules.columns:
        penalized_scores = pd.to_numeric(detailed_rules["Penalized_Rule_Score"], errors="coerce").dropna()
        if not penalized_scores.empty:
            metrics["mean_penalized_rule_score"] = float(penalized_scores.mean())
    return metrics


def _save_run_outputs(
    *,
    dataset_name: str,
    baseline: BASELINE,
    run: int,
    detailed_rules: pd.DataFrame,
    runtime: float,
    mean_rule_size: float,
    info: pd.DataFrame,
    metrics: dict,
    output_dir: Path,
    aggregate_state: _AggregateState,
    debug_performance: bool,
) -> None:
    aggregate_state.scores.append(np.array(detailed_rules["Rule_Score"].values))
    aggregate_state.runtimes.append(round(runtime, 2))
    aggregate_state.mean_rule_sizes.append(mean_rule_size)

    detailed_rules.to_csv(output_dir / f"{dataset_name}_{run}_{baseline}_DetailedRules.csv", index=False)
    info.to_csv(output_dir / f"{dataset_name}_{run}_{baseline}_Info.csv", index=False)
    pd.DataFrame([metrics]).round(2).to_csv(
        output_dir / f"{dataset_name}_{run}_{baseline}_RulesMetricsResult.csv",
        index=False,
        float_format="%.4f",
    )

    if debug_performance:
        aggregate_state.cpu_mean_percent.append(float(info["cpu_mean_percent"].iloc[0]))
        aggregate_state.cpu_peak_percent.append(float(info["cpu_peak_percent"].iloc[0]))
        aggregate_state.ram_mean_mb.append(float(info["ram_mean_mb"].iloc[0]))
        aggregate_state.ram_peak_mb.append(float(info["ram_peak_mb"].iloc[0]))
        aggregate_state.ram_incremental_peak_mb.append(float(info["ram_incremental_peak_mb"].iloc[0]))
        aggregate_state.ram_baseline_mb.append(float(info["ram_baseline_mb"].iloc[0]))


def _save_aggregate_outputs(
    *,
    dataset_name: str,
    baseline: BASELINE,
    metrics_list: list[dict],
    output_dir: Path,
    aggregate_state: _AggregateState,
    debug_performance: bool,
) -> tuple[Path, Path]:
    stats_path = output_dir / f"{dataset_name}_{baseline}_RulesStatsResult.csv"
    metrics_path = output_dir / f"{dataset_name}_{baseline}_RulesMetricsResult.csv"

    stats_data = {
        "mean_score": [
            _format_mean_std(
                np.mean(aggregate_state.scores, axis=0, dtype=np.float32).mean(),
                np.mean(aggregate_state.scores, axis=0, dtype=np.float32).std(),
            )
        ],
        "mean_runtime": [
            _format_mean_std(
                np.mean(aggregate_state.runtimes, axis=0, dtype=np.float32).mean(),
                np.mean(aggregate_state.runtimes, axis=0, dtype=np.float32).std(),
            )
        ],
        "mean_rule_size": [
            _format_mean_std(
                np.mean(aggregate_state.mean_rule_sizes, axis=0, dtype=np.float32).mean(),
                np.mean(aggregate_state.mean_rule_sizes, axis=0, dtype=np.float32).std(),
            )
        ],
    }
    if debug_performance:
        stats_data.update(
            {
                "cpu_mean_percent": [
                    _format_mean_std(
                        np.mean(aggregate_state.cpu_mean_percent, axis=0, dtype=np.float32).mean(),
                        np.mean(aggregate_state.cpu_mean_percent, axis=0, dtype=np.float32).std(),
                    )
                ],
                "cpu_peak_percent": [
                    _format_mean_std(
                        np.mean(aggregate_state.cpu_peak_percent, axis=0, dtype=np.float32).mean(),
                        np.mean(aggregate_state.cpu_peak_percent, axis=0, dtype=np.float32).std(),
                    )
                ],
                "ram_mean_mb": [
                    _format_mean_std(
                        np.mean(aggregate_state.ram_mean_mb, axis=0, dtype=np.float32).mean(),
                        np.mean(aggregate_state.ram_mean_mb, axis=0, dtype=np.float32).std(),
                    )
                ],
                "ram_peak_mb": [
                    _format_mean_std(
                        np.mean(aggregate_state.ram_peak_mb, axis=0, dtype=np.float32).mean(),
                        np.mean(aggregate_state.ram_peak_mb, axis=0, dtype=np.float32).std(),
                    )
                ],
                "ram_incremental_peak_mb": [
                    _format_mean_std(
                        np.mean(aggregate_state.ram_incremental_peak_mb, axis=0, dtype=np.float32).mean(),
                        np.mean(aggregate_state.ram_incremental_peak_mb, axis=0, dtype=np.float32).std(),
                    )
                ],
                "ram_baseline_mb": [
                    _format_mean_std(
                        np.mean(aggregate_state.ram_baseline_mb, axis=0, dtype=np.float32).mean(),
                        np.mean(aggregate_state.ram_baseline_mb, axis=0, dtype=np.float32).std(),
                    )
                ],
            }
        )

    pd.DataFrame(stats_data).round(2).to_csv(stats_path, index=False)
    output_metrics(metrics_list, str(metrics_path))
    return stats_path, metrics_path


def _save_figures(figures_list: list, output_dir: Path, plot_rank: int) -> None:
    convergency_plot = figures_list.pop(-1)
    convergency_plot.savefig(output_dir / "convergency_score.pdf", dpi=1000, bbox_inches="tight")

    if plot_rank <= 0:
        return
    for i, figure in enumerate(figures_list):
        filename = f"top-{plot_rank}_best_rules.pdf" if i == 0 else f"top-{i}_rule.pdf"
        figure.savefig(output_dir / filename, dpi=1000, bbox_inches="tight")


def _format_mean_std(mean: float, std: float) -> str:
    return f"{round(float(mean), 2)}+/-{round(float(std), 2)}"
