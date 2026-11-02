#!/usr/bin/env bash
# Analyze sensitivity for the article's numerical factors.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$REPO_ROOT"

exec uv run --locked python experiments/factorial_design/factorial_analysis.py \
  --designs \
  experiments/factorial_design/designs/factorial_alpha.csv \
  experiments/factorial_design/designs/factorial_generations.csv \
  experiments/factorial_design/designs/factorial_ksize.csv \
  experiments/factorial_design/designs/factorial_max_relative_supp.csv \
  experiments/factorial_design/designs/factorial_min_relative_supp.csv \
  experiments/factorial_design/designs/factorial_population.csv \
  experiments/factorial_design/designs/factorial_redundancy_penalty.csv \
  experiments/factorial_design/designs/factorial_redundancy_similarity_threshold.csv \
  experiments/factorial_design/designs/factorial_restart_gen.csv \
  experiments/factorial_design/designs/factorial_restart_pct.csv \
  experiments/factorial_design/designs/factorial_restart_pop.csv \
  experiments/factorial_design/designs/factorial_threshold.csv \
  "$@"
