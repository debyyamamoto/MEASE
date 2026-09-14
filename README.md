# MEASE

MEASE is organized in two layers:

- `main.py` runs the algorithm on any parquet survival dataset.
- `experiments/` reproduces the article protocol, result aggregation, and
  hypothesis tests.

## Generic Use

```bash
uv run python main.py datasets/files/cancer.parquet -time time -event status
```

Population baseline with custom parameters:

```bash
uv run python main.py datasets/files/cancer.parquet -time time -event status -comp population -a 0.8 -exe 3
```

Results are written to `results/<dataset>/<baseline>/`.

## Main CLI Parameters

| Argument | Required | Description | Default |
| --- | --- | --- | --- |
| `filepath` | yes | Path to a parquet dataset | - |
| `-time`, `--time_col` | yes | Survival time column | - |
| `-event`, `--event_col` | yes | Event/censoring column | - |
| `--output_dir` | no | Output directory | `results` |
| `--dataset_name` | no | Name used in generated files | file stem |
| `--seed` | no | Reproducibility seed | run index |
| `-g`, `--generations` | no | Maximum generations | `500` |
| `-p`, `--population` | no | Population size | `500` |
| `--restart_gen` | no | Generations without improvement | `3` |
| `--restart_pop` | no | Population restart limit | `3` |
| `--restart_pct` | no | Restarted population percentage | `10` |
| `-comp`, `--comparacao` | no | `complement` or `population` baseline | `complement` |
| `-a`, `--alpha` | no | Fitness alpha weight | `0.5` |
| `--score_metric` | no | `legacy_logrank`, `fast_logrank`, `km_cvm`, `km_abc`, `mdir2`, `mdir3`, or `mdir4` | `legacy_logrank` |
| `--km_time_bins` | no | Maximum time grid size for `km_cvm`/`km_abc`/`mdir*`; use `0` for exact event-time grid | `512` |
| `--redundancy_penalty` | no | Strength of the Top-K curve-similarity penalty in `[0, 1]` | `0.0` |
| `--redundancy_similarity_threshold` | no | Penalize KM curve similarities above this value | `0.5` |
| `-exe`, `--executions` | no | Independent executions | `1` |
| `-k`, `--ksize` | no | Top-K rule rank size | `10` |
| `-plt`, `--plt_rank` | no | Save top-N plots for a single execution | `0` |
| `-d`, `--debug_performance` | no | Collect CPU/RAM metrics (`on`/`off`) | `off` |

When the redundancy penalty is enabled, `Rule_Score` keeps the original fitness
and `Penalized_Rule_Score` records the value used to rank Top-K candidates. Curve
similarity is `1 - sqrt(weighted_cvm_distance)`, so the threshold is expressed on
the Kaplan-Meier survival-probability scale rather than the squared `km_cvm` scale.
The stored penalized score reflects the archive at admission/update time, not a
joint reevaluation of the final Top-K. The paired real-data evaluation and its
diagnostics are documented in [the redundancy benchmark](experiments/redundancy_benchmark.md).

## Article Experiments

Run the full article protocol:

```bash
bash ./experiments/main_stats.sh
```

The script calls `main.py` for each fixed dataset/baseline experiment from the
article and then compiles per-run metrics into:

- `experiments/resultados_complement.csv`
- `experiments/resultados_population.csv`

You can still compile per-run metrics manually:

```bash
uv run python experiments/collect_results.py --baseline complement
uv run python experiments/collect_results.py --baseline population
```

Run Wilcoxon tests:

```bash
uv run python experiments/test_wilcoxon.py
```

See `experiments/README.md` and `AGENT.md` for the reproducibility map.
