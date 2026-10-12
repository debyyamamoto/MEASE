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
