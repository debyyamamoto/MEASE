#!/usr/bin/env bash
set -euo pipefail

# Fixed protocol used for the article experiments.
RESULTS_DIR="experiments/results_mdir2"
EXECUTIONS="30"
GENERATIONS="500"
POPULATION="500"
RESTART_GEN="5"
RESTART_POP="5"
RESTART_PCT="10"
ALPHA="0.10"
KSIZE="10"
THRESHOLD="0.5"
SCORE_METRIC="mdir2"
DEBUG_PERFORMANCE="off"

DATASETS=(
  "cancer|datasets/files/cancer.parquet|time|status"
  "breast-cancer|datasets/files/breast-cancer.parquet|t_tdm|e_tdm"
  "carcinoma|datasets/files/carcinoma.parquet|survival|status"
  "lung|datasets/files/lung.parquet|survival|status"
  "mgus2|datasets/files/mgus2.parquet|futime|status"
  "veteran|datasets/files/veteran.parquet|survival|status"
)

BASELINES=(
  "complement"
)

DATASET_NAMES=()

for dataset_spec in "${DATASETS[@]}"; do
  IFS="|" read -r dataset_name dataset_path time_col event_col <<< "${dataset_spec}"
  DATASET_NAMES+=("${dataset_name}")

  for baseline in "${BASELINES[@]}"; do
    uv run python main.py "${dataset_path}" \
      --dataset_name "${dataset_name}" \
      --time_col "${time_col}" \
      --event_col "${event_col}" \
      --score_metric "${SCORE_METRIC}" \
      --output_dir "${RESULTS_DIR}" \
      --executions "${EXECUTIONS}" \
      --generations "${GENERATIONS}" \
      --population "${POPULATION}" \
      --restart_gen "${RESTART_GEN}" \
      --restart_pop "${RESTART_POP}" \
      --restart_pct "${RESTART_PCT}" \
      --comparacao "${baseline}" \
      --alpha "${ALPHA}" \
      --ksize "${KSIZE}" \
      --threshold "${THRESHOLD}" \
      --debug_performance "${DEBUG_PERFORMANCE}"
  done
done

for baseline in "${BASELINES[@]}"; do
  uv run python experiments/collect_results.py \
    --results_dir "${RESULTS_DIR}" \
    --datasets "${DATASET_NAMES[@]}" \
    --baseline "${baseline}" \
    --executions "${EXECUTIONS}" \
    --output_file "experiments/resultados_${baseline}.csv"
done
