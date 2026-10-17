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
  --input experiments/repeated_candidates --confidence 0.95 \
  --permutations 99999 --analysis-seed 2026
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
for each of the three metrics. With the default panel, **N=6, K=10**, not N=180.
More repetitions improve the estimates on these bases but do not replace more
independent datasets. Independence/representativeness of the six bases must be
assessed separately, particularly if bases might share records or origins.

The primary omnibus result is a tie-corrected Friedman statistic calibrated by
99,999 within-dataset label permutations (plus-one p-value, fixed analysis seed).
This additional choice avoids relying exclusively on the small-sample chi-square
approximation; it assumes exchangeability under the null. Asymptotic Friedman
and Iman–Davenport values are also reported. Holm adjusts the three omnibus
p-values. Monte Carlo standard errors are exported; a decision close to the
threshold needs more permutations.

Nemenyi compares all configuration pairs using average ranks. It accounts for
all pairs within a metric, with an additional Bonferroni adjustment across the
three metrics. Differences of ranks have simultaneous intervals defined by the
critical difference. A pair is declared significant only after its omnibus test
passes Holm and its Nemenyi interval excludes zero. Nemenyi remains approximate
and can be conservative with ties; the global permutation calibration does not
make the post-hoc test exact. Six datasets and ten similar candidates may provide
little power. Non-significance does **not** establish equivalence.

The report also separates two kinds of uncertainty:

- Pointwise t confidence intervals for each configuration/dataset mean over
  seeds, with SD and zero-variance flags.
- Paired t intervals for raw A−B differences: compute the macro mean across the
  fixed datasets for each seed, then compare the paired seeds. Both pointwise
  and Bonferroni simultaneous intervals over all pairs × metrics are exported.
  These describe algorithm randomness on this fixed panel, not generalization
  to new datasets, and do not replace Friedman/Nemenyi.

The t intervals require a suitable approximation for the mean and independent
repetitions; r=30 does not guarantee normality. Bounds are not clipped. Zero
observed variance yields a degenerate interval, not proof of zero uncertainty.
Use intervals of **differences**, not overlap of two separate mean intervals, to
evaluate a contrast. The multiplicity families for dataset inference and for
conditional seed intervals are separate and explicitly labeled.

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
