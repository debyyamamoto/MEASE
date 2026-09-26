# Experiments

This directory contains aggregation, statistical tests, and generated outputs
for the article. The fixed article protocol lives in `main_stats.sh`, which
calls the generic `main.py` command with the paper datasets and parameters.

## Run Article Experiments

```bash
bash ./experiments/main_stats.sh
```

The shell script is the fixed article protocol. It runs all article datasets
with both `complement` and `population` baselines, 30 executions each, writes
results to `experiments/results/<dataset>/<baseline>/`, and then compiles
`experiments/resultados_complement.csv` and
`experiments/resultados_population.csv`.

For focused debugging runs, call `main.py` directly with fewer executions:

```bash
uv run python main.py datasets/files/cancer.parquet -time time -event status -comp population -exe 3
```

## Compile Metrics

```bash
uv run python experiments/collect_results.py --baseline complement
uv run python experiments/collect_results.py --baseline population
```

These commands generate `experiments/resultados_complement.csv` and
`experiments/resultados_population.csv`.

## Hypothesis Tests

```bash
uv run python experiments/test_wilcoxon.py
```

By default this compares `experiments/resultados_complement.csv` against
`experiments/results_esmam/metrics_baseline-complement.csv` and saves
`experiments/wilcoxon.csv`.

## Quantitative Methods Experiments

The planned factorial experiments for runtime, exceptionality, and MEASE versus
EsmamDS comparison are mapped in:

```bash
experiments/quantitative_methods_plan.md
experiments/designs/factorial_runtime.csv
experiments/designs/factorial_exceptionality.csv
```

These files define the target questions, factor levels, responses, assumptions,
diagnostics, and implementation improvements needed before running the
confirmatory experiments.

After running the factorial treatments with `factorial.py`, build the design
matrix, estimate effects with the manual `2^k r` contrast method, run ANOVA,
and export assumption diagnostics with:

```bash
uv run python experiments/analyze_factorial.py
```

The analysis writes:

- `*_2kr_treatment_table.csv`: one row per treatment, coded signs, replicate
  responses, treatment totals, treatment means, `SSY_i`, and `SSE_i`.
- `*_2kr_steps.csv`: the classroom calculation sequence with `SSY`, `SS0`,
  each `SS_i`, `SSE`, `SST`, `q_i`, `s_qi`, and explained variance.
- `*_effects.csv`, `*_anova.csv`, `*_additive_residuals.csv`, and diagnostic
  plots for interpretation and assumption checks.

With the default `--transform auto`, a log-scale analysis is also written when
the raw response suggests non-additive or multiplicative behavior.
