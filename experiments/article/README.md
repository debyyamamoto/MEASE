# Article protocols

## Reproducible Random Search

With Bash, GNU coreutils and [uv](https://docs.astral.sh/uv/) installed, run:

```bash
bash experiments/article/random_search.sh
```

The script uses `uv run --locked` to install/use the project dependencies from
`uv.lock`. The six parquet datasets must be present under `datasets/files/`.
It resolves repository paths independently of the caller's working directory.

The default search evaluates 300 sampled configurations on each of `cancer`,
`breast-cancer`, `carcinoma`, `lung`, `mgus2` and `veteran` (1,800 MEASE runs).
It preserves all 13 parameter domains from the former root `random_search.py`,
including minimum/maximum rule support. Fixed settings are `complement`,
`greedy`, `adaptive`, 512 time bins, one execution and 10 plots per run.

Sampling uses seed **2026** with an explicitly specified Park-Miller generator
and uniform rejection sampling; MEASE uses seed **0** for every configuration
and dataset. This defines a new reproducible search, not a reconstruction of the
previous unseeded NumPy search. The same seed reproduces the sampled parameters;
numerical reproducibility also depends on the software/hardware environment.

Results default to `experiments/random_search_shell/`, separate from the legacy
results. The layout remains `config_<id>/<dataset>/complement/`, compatible with:

```bash
uv run python experiments/analyze_random_search.py \
  --input experiments/random_search_shell \
  --output experiments/results_random_search_shell_analysis
```

### Preview and customization

```bash
bash experiments/article/random_search.sh --dry-run
bash experiments/article/random_search.sh --num-configs 5 \
  --search-seed 42 --run-seed 7 --output-dir experiments/random_search_pilot
```

`--dry-run` saves/verifies the manifests and prints the commands; it does not
execute MEASE or install dependencies. Reuse the same options without `--dry-run`
to execute that plan. Use a new output directory for a different search or seed.
Use separate directories and distinct `--run-seed` values for independent
algorithm repetitions with the same `--search-seed`.

### Resume and provenance

Repeat the original command to resume. The script:

- Persists `configs.txt` (Python literal dictionaries, compatible with the
  analyzer), `configs.tsv` (ID followed by the 13 parameters in the script's
  `PARAMETERS` order), and `protocol.txt` (seeds, fixed settings, dataset schema
  and SHA-256 fingerprints of code, dependency declarations and dataset files).
- Refuses changed manifests, code or data and refuses legacy output directories
  without its protocol. It never infers completion from directory existence.
- Saves the complete invocation in each run's `command.sh`, plus `run.log`.
  A `.complete` marker is written only after a successful process and the three
  required CSV outputs. It checks their hashes before skipping a completed run.
- Stops on the first error; retrying reruns failed/incomplete or modified runs.
  An exclusive directory lock prevents simultaneous searches in the same output.

Normal exit and interruption release the lock. After a forced kill or machine
failure, remove the empty `.random_search.lock` directory with `rmdir` only after
confirming no search process is using that output directory. A process killed
while initially saving manifests may leave an incomplete plan; use a new output
directory in that case.

Share the manifests along with the source revision, `uv.lock`, dataset versions
and results when publishing the experiment. Files such as CSV/TXT/parquet are
ignored by the repository's current `.gitignore` and need to be distributed
explicitly as research artifacts.

## Repeated evaluation of the best Random Search configurations

Two entrypoints select/freeze the candidates, run fresh seeds, and analyze the
complete panel:

```bash
# Inspect the frozen selection and all commands (does not run MEASE).
bash experiments/article/repeat_random_search.sh --dry-run

# Default: top 10 × six datasets × 30 repetitions = 1,800 new runs.
bash experiments/article/repeat_random_search.sh

# Run after the entire panel has completed.
bash experiments/article/analyze_repeated_search.sh
```

The optional `--mode {repeated,single}` chooses the analysis inputs:

- `repeated` (default): uses the frozen selection in `plan.json` and averages
  the fresh repetitions. Default input: `experiments/repeated_random_search`.
- `single`: selects the best 10 configurations from the original Random Search
  with the same equal-weight mean-rank policy, then runs Friedman/Nemenyi on
  their original `RulesMetricsResult.csv` values (one execution per dataset).
  If the search has fewer than 10 configurations, all are used (minimum 3).
  Default input: `experiments/random_search_shell`. No repetition plan or
  completed repetitions are needed. CSV rounding is preserved.

```bash
bash experiments/article/analyze_repeated_search.sh --mode repeated
bash experiments/article/analyze_repeated_search.sh --mode single

# --input always points to the source directory for the chosen mode.
bash experiments/article/analyze_repeated_search.sh --mode single \
  --input experiments/random_search_shell --output-dir experiments/analysis_single
```

Both modes produce the same statistical tables and record their mode in the
report and `analysis_protocol.json`. By default each writes to `analysis/`
inside its own input directory, so the two default reports are separate.
The default top-10 selection matches between modes for the same original
search; a custom repeated selection (`--config-ids` or `--top-k`) is specific
to that repetition plan. Single mode always selects the original top 10.
Single mode validates the complete original search panel, metadata and saved
output checksums before selection; it does not compare saved code hashes to
the currently installed code. Neither mode launches new MEASE executions.

The shell entrypoints use Python helpers for manifest auditing, execution and
statistics, with the project's locked dependencies. No additional package is
required. Relative paths resolve from the repository root, including when the
shell scripts are invoked from another working directory.

The default source is `experiments/random_search_shell`. All configurations and
datasets must be present and pass metadata, saved-command and completion-hash
checks. Candidates are selected using the existing `rank_panel` policy: average
of the three metric ranks within each dataset, then average across datasets,
with equal weights. Exceptionality and setCov are maximized; model redundancy
is minimized. Ties receive average ranks; configuration ID breaks a tie at the
top-10 selection boundary. The selection is frozen before the repetitions, never
updated based on their results. The current source selects IDs
**55, 172, 150, 9, 43, 103, 79, 234, 21, 182**.

Results go to `experiments/repeated_random_search`, containing:

- `plan.json`: complete parameters, datasets, seeds, shuffled execution order,
  source fingerprints, code/data hashes and environment information.
- `selection_ranking.csv` and `selected_configs.csv`: original ranking and
  selected configurations with their parameters.
- `config_<id>/seed_<seed>/<dataset>/complement/`: normal CLI outputs,
  `command.sh`, `run.log`, unrounded `metrics_full.json`, and `.complete.json`.
- `analysis/relatorio.md`: Portuguese report, statistical CSVs and rank plots
  in PNG/PDF, generated by the analysis entrypoint.

Seeds **1..30** are paired across configurations and exclude the source seed
(0 by default). Each run calls `main.py --executions 1 --seed <seed>`: the current
runner repeats a fixed explicit seed if given `--executions 30 --seed X`.
The new protocol sets `PYTHONHASHSEED=0` and shuffles all jobs with order seed
2026. Plotting defaults to zero to avoid producing thousands of figures; use
`--plot-rank 10` to save them. Other algorithm settings are preserved.

The normal CLI rounds metrics to two decimals. The helper recomputes the metrics
from exported rules using the same metric implementation, checks agreement with
the rounded CSV, and stores full precision for inference. This does not change
the algorithm or the source ranking.

### Custom selection, pilot and resume

```bash
bash experiments/article/repeat_random_search.sh --top-k 5 --repetitions 30 \
  --output-dir experiments/repeated_top5

# Choose explicit original IDs instead of --top-k (at least three).
bash experiments/article/repeat_random_search.sh --config-ids 9 55 172 \
  --repetitions 30 --seed-start 1 --output-dir experiments/repeated_candidates

# Execute only one new job, preserving the full default plan; resume afterwards.
bash experiments/article/repeat_random_search.sh --max-runs 1
bash experiments/article/repeat_random_search.sh

bash experiments/article/analyze_repeated_search.sh \
  --input experiments/repeated_candidates --confidence 0.95
```

Repeat the same experiment options to resume. Valid completed runs are skipped;
incomplete/modified runs are retried. Source selection, seeds, parameters,
implementation, data and environment must match the frozen plan. A changed
implementation requires a new source experiment and new output directory.
Do not delete output folders to resume. An exclusive `.repetitions.lock` prevents
concurrent execution/analysis; after a forced kill, remove that empty directory
only after confirming no experiment process remains active.

The analysis refuses incomplete/unbalanced panels and invalid completion hashes.
It never treats missing runs as zeros or silently discards seeds or datasets.
Use a new `--output-dir` for analysis sensitivity runs to retain previous reports.

### Statistical protocol

Following [Demšar (2006)](https://jmlr.org/papers/v7/demsar06a.html), the independent
blocks are **datasets**, not individual algorithm repetitions. The 30 seeds
estimate one mean per configuration/dataset; Friedman receives an N × K matrix
for each of the three metrics and their combined `mean_metric_rank` score.
With the default panel, **N=6, K=10**, not N=180.
More repetitions improve the estimates on these bases but do not replace more
independent datasets. Independence/representativeness of the six bases must be
assessed separately, particularly if bases might share records or origins.

The descriptive `mean_rank` averages the ranks of exceptionality (higher is
better), setCov (higher is better), and model redundancy (lower is better), with
equal weights across metrics and datasets. Ranks are assigned among the frozen
candidates after averaging repetitions, with average ranks for ties.

Both modes also test the combined score `mean_metric_rank`: within each dataset,
average the three metric ranks with equal weights, then rank these scores again
across configurations (lower is better, average ranks for ties). Friedman and
Nemenyi use these new ranks, with datasets still as blocks. The three metrics
are not treated as additional blocks. This tests the chosen composite score;
it does not imply that every individual metric differs significantly.
`ranks_per_dataset.csv` exposes both `mean_metric_rank` and its test rank,
`rank_mean_metric_rank`. The `mean_metric_rank` entries in `metric_ranks.csv`
average the latter, and can differ from the descriptive `mean_rank` in
`ranking.csv`, which averages the composite scores themselves.

Each metric has a separate tie-corrected Friedman test. Following §3.2.2, its
**Iman–Davenport F approximation** determines rejection at alpha=0.05 by default:
`F = (N−1) chi2 / [N(K−1)−chi2]`, with K−1 and (K−1)(N−1) degrees of freedom.
`friedman.csv` stores this p-value as `p_value`; `p_chi2` is informational only.
The F approximation is not a multiplicity adjustment. Complete within-dataset
ties return statistic=0 and p=1; perfect ordering returns F=infinity and p=0.

After an omnibus rejection, Nemenyi compares all K(K−1)/2 configuration pairs
within that metric at the same alpha, **without additional Holm or Bonferroni**.
Pairwise p-values come from scikit-posthocs' `posthoc_nemenyi_friedman`, using
the same rank matrix as Friedman (rows=datasets, columns=configurations).
A pair is significant when `p_nemenyi < alpha`. The package version is recorded
in `analysis_protocol.json`.
The critical difference is `CD = q_alpha sqrt(K(K+1)/(6N))`, where `q_alpha` is
the studentized-range quantile divided by sqrt(2), as in Table 5(a). It is retained
for tables/plots; exceeding CD is equivalent to the p-value decision. Multiplicity is
controlled within each outcome's pairs, not jointly across the three metrics
and the composite score.
Nemenyi and the omnibus F calibration remain approximations; six datasets and
ten similar candidates may provide little power.

Outputs:

- `ranking.csv`: the three average metric ranks and their combined `mean_rank`.
- `runs.csv`, `dataset_means.csv`, `ranks_per_dataset.csv`, `metric_ranks.csv`:
  observations and intermediate calculations.
- `friedman.csv`: four omnibus results (three metrics plus `mean_metric_rank`).
- `nemenyi.csv`: all 45 pairs per outcome for the default top 10 (180 rows), with p-values,
  critical difference and `decision` (`significant`, `not_significant`, or
  `not_tested` when Friedman does not reject; in that case the post-hoc p is empty).
- `nemenyi_significant.csv`, `nemenyi_not_significant.csv`: the tested pairs split
  by decision. Non-significance does **not** establish equivalence.
- `relatorio.md`, `mean_ranks.png/pdf`, `analysis_protocol.json`: interpretation,
  descriptive plots, and method/input provenance.

Migration from the former analysis: `--permutations` and `--analysis-seed` were
removed. Monte Carlo tests, additional multiplicity adjustments and seed t
intervals are no longer computed. The old `cell_intervals.csv` and
`paired_seed_intervals.csv` are not outputs of this protocol. To retain an old
analysis without mixing artifacts, use a new directory, for example:

```bash
bash experiments/article/analyze_repeated_search.sh \
  --output-dir experiments/repeated_random_search/analysis_friedman_nemenyi
```

Fresh seeds address the original single-run limitation, but selecting candidates
and evaluating them on the same datasets still incurs selection bias. This is a
post-selection analysis; independent validation datasets or nested selection
are needed for confirmatory generalization. No patient-level resampling or
holdout is introduced here.

Existing algorithm limitations remain: support limits are not forwarded to
`RuleEvaluator`, final interval adjustment can change coverage, and model
redundancy includes fallback values for non-testable pairs. Fixes require a
separate version of the experiment; repetitions alone do not fix these issues.

Validate orchestration and statistics without running the full campaign:

```bash
uv run --locked python -m unittest experiments.article.test_repeated_search \
  experiments.article.test_random_search
```
