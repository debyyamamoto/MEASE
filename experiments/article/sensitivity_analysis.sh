#!/usr/bin/env bash
# Analyze sensitivity for the article's numerical factors.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$REPO_ROOT"

parameters_list=(
  "experiments/factorial_design/designs/factorial_alpha.csv --response_transforms mean_pairwise_rms=logit"
  "experiments/factorial_design/designs/factorial_generations.csv --response_transforms 'description redundancy'=sqrt mean_pairwise_rms=logit"
  "experiments/factorial_design/designs/factorial_ksize.csv --response_transforms mean_pairwise_rms=logit"
  "experiments/factorial_design/designs/factorial_max_relative_supp.csv"
  "experiments/factorial_design/designs/factorial_min_relative_supp.csv"
  "experiments/factorial_design/designs/factorial_population.csv --response_transforms mean_pairwise_rms=logit"
  "experiments/factorial_design/designs/factorial_redundancy_penalty.csv --response_transforms 'description redundancy'=sqrt 'model redundancy'=sqrt mean_pairwise_rms=logit"
  "experiments/factorial_design/designs/factorial_redundancy_similarity_threshold.csv"
  "experiments/factorial_design/designs/factorial_restart_gen.csv"
  "experiments/factorial_design/designs/factorial_restart_pct.csv"
  "experiments/factorial_design/designs/factorial_restart_pop.csv"
  "experiments/factorial_design/designs/factorial_threshold.csv --response_transforms 'description redundancy'=sqrt CR=sqrt"
)

for param_exec in "${parameters_list[@]}"; do  
  eval "uv run --locked python experiments/factorial_design/factorial_analysis.py --designs $param_exec \"\$@\""
done
