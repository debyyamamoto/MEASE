#!/usr/bin/env bash
# Reproducible Random Search. Requires Bash, coreutils and uv.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash experiments/article/random_search.sh [options]
  --output-dir PATH   Results directory (default: experiments/random_search_shell)
  --num-configs N     Number of configurations (default: 300)
  --search-seed N     Sampling seed, 1..2147483646 (default: 2026)
  --run-seed N        MEASE seed, 0..2147483646 (default: 0)
  --dry-run           Save/verify manifests and print commands without running MEASE
  -h, --help          Show this help

Relative output paths are resolved from the repository root. Repeat the same
command to resume. Existing results without this script's manifests are refused.
EOF
}

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
OUTPUT_DIR=experiments/random_search_shell
NUM_CONFIGS=300
SEARCH_SEED=2026
RUN_SEED=0
DRY_RUN=0
while (($#)); do
  case "$1" in
    --output-dir|--num-configs|--search-seed|--run-seed)
      (($# >= 2)) || die "Missing value for $1"
      case "$1" in
        --output-dir) OUTPUT_DIR=$2 ;;
        --num-configs) NUM_CONFIGS=$2 ;;
        --search-seed) SEARCH_SEED=$2 ;;
        --run-seed) RUN_SEED=$2 ;;
      esac
      shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

for value in "$NUM_CONFIGS" "$SEARCH_SEED" "$RUN_SEED"; do
  [[ $value =~ ^[0-9]{1,10}$ ]] || die "Counts and seeds must be nonnegative integers"
done
NUM_CONFIGS=$((10#$NUM_CONFIGS))
SEARCH_SEED=$((10#$SEARCH_SEED))
RUN_SEED=$((10#$RUN_SEED))
((NUM_CONFIGS > 0 && NUM_CONFIGS <= 2147483646)) || die "Invalid number of configurations"
((SEARCH_SEED > 0 && SEARCH_SEED <= 2147483646)) || die "Invalid sampling seed"
((RUN_SEED <= 2147483646)) || die "Invalid MEASE seed"
[[ -n $OUTPUT_DIR ]] || die "Output directory cannot be empty"

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$REPO_ROOT"
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd)
mkdir -- "$OUTPUT_DIR/.random_search.lock" 2>/dev/null || die "Search already locked: $OUTPUT_DIR/.random_search.lock"
TEMP_DIR=''
cleanup() {
  if [[ -n $TEMP_DIR ]]; then rm -rf -- "$TEMP_DIR"; fi
  rmdir -- "$OUTPUT_DIR/.random_search.lock"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
TEMP_DIR=$(mktemp -d "$OUTPUT_DIR/.plan.XXXXXX")

DATASETS=(
  'cancer|datasets/files/cancer.parquet|time|status'
  'breast-cancer|datasets/files/breast-cancer.parquet|t_tdm|e_tdm'
  'carcinoma|datasets/files/carcinoma.parquet|survival|status'
  'lung|datasets/files/lung.parquet|survival|status'
  'mgus2|datasets/files/mgus2.parquet|futime|status'
  'veteran|datasets/files/veteran.parquet|survival|status'
)
PARAMETERS=(score_metric threshold generations population restart_gen restart_pop
  restart_pct alpha redundancy_penalty redundancy_similarity_threshold ksize
  min_rule_support max_rule_support)

# Park-Miller PRNG: explicit integer arithmetic, independent of Bash's RANDOM.
# Rejection sampling avoids modulo bias. Each call assigns PICK in this shell.
RNG_STATE=$SEARCH_SEED
pick() {
  local options=("$@") count=$# draw limit
  limit=$((2147483646 - 2147483646 % count))
  while :; do
    RNG_STATE=$((RNG_STATE * 16807 % 2147483647))
    draw=$((RNG_STATE - 1))
    if ((draw < limit)); then break; fi
  done
  PICK=${options[draw % count]}
}

# configs.txt stays compatible with experiments/analyze_random_search.py.
: > "$TEMP_DIR/configs.txt"
: > "$TEMP_DIR/configs.tsv"
for ((cid=0; cid<NUM_CONFIGS; cid++)); do
  values=()
  pick fast_logrank km_cvm km_abc mdir2; values+=("$PICK")
  pick 0.10 0.30 0.60 0.70 0.90; values+=("$PICK")
  pick 50 100 250 500; values+=("$PICK")
  pick 100 300 500 800; values+=("$PICK")
  pick 1 3 5 10; values+=("$PICK")
  pick 1 3 5 10; values+=("$PICK")
  pick 0.10 0.30 0.60 0.90; values+=("$PICK")
  pick 0.10 0.30 0.5 0.70 1.0; values+=("$PICK")
  pick 0.3 0.50 0.70 1.0; values+=("$PICK")
  pick 0.10 0.3 0.50 0.70 0.90; values+=("$PICK")
  pick 3 5 7 10; values+=("$PICK")
  pick 0.05 0.10 0.20; values+=("$PICK")
  pick 0.35 0.45 0.55; values+=("$PICK")
  {
    printf "{'score_metric': '%s'" "${values[0]}"
    for ((j=1; j<${#PARAMETERS[@]}; j++)); do
      printf ", '%s': %s" "${PARAMETERS[j]}" "${values[j]}"
    done
    printf '}\n'
  } >> "$TEMP_DIR/configs.txt"
  { printf '%s' "$cid"; printf '\t%s' "${values[@]}"; printf '\n'; } >> "$TEMP_DIR/configs.tsv"
done

# Version/data fingerprints prevent silently combining different implementations.
{
  printf 'protocol=mease-random-search-shell-v1\nsearch_seed=%s\nrun_seed=%s\nnum_configs=%s\n' "$SEARCH_SEED" "$RUN_SEED" "$NUM_CONFIGS"
  printf 'baseline=complement\nexecutions=1\narchive_selection=greedy\nrate_policy=adaptive\nkm_time_bins=512\nplot_rank=10\n'
  sha256sum main.py easd/*.py pyproject.toml uv.lock experiments/article/random_search.sh
  for spec in "${DATASETS[@]}"; do
    IFS='|' read -r name path time_col event_col <<< "$spec"
    printf 'dataset=%s\n' "$spec"
    sha256sum "$path"
  done
} > "$TEMP_DIR/protocol.txt"

if [[ -f $OUTPUT_DIR/protocol.txt ]]; then
  for manifest in protocol.txt configs.txt configs.tsv; do
    cmp -s -- "$TEMP_DIR/$manifest" "$OUTPUT_DIR/$manifest" || die "Manifest differs ($manifest). Use the original settings/code/data, or a new output directory."
  done
else
  # Only our lock and temporary plan may exist before the first initialization.
  shopt -s nullglob dotglob
  for entry in "$OUTPUT_DIR"/*; do
    [[ $entry == "$TEMP_DIR" || $entry == "$OUTPUT_DIR/.random_search.lock" ]] || die "Existing results have no reproducible protocol. Use a new output directory."
  done
  shopt -u nullglob dotglob
  cp -- "$TEMP_DIR/configs.txt" "$TEMP_DIR/configs.tsv" "$OUTPUT_DIR/"
  cp -- "$TEMP_DIR/protocol.txt" "$OUTPUT_DIR/protocol.txt"
fi

if (( ! DRY_RUN )); then
  command -v uv >/dev/null || die "Install uv before executing this protocol"
fi

for spec in "${DATASETS[@]}"; do
  IFS='|' read -r name path time_col event_col <<< "$spec"
  while IFS=$'\t' read -r -a row; do
    cid=${row[0]}
    config_dir="$OUTPUT_DIR/config_$cid"
    run_dir="$config_dir/$name/complement"
    command=(uv run --locked python main.py "$path" --dataset_name "$name"
      --time_col "$time_col" --event_col "$event_col"
      --output_dir "$config_dir" --seed "$RUN_SEED" --executions 1
      --comparacao complement --archive_selection greedy --rate_policy adaptive
      --km_time_bins 512 --plt_rank 10 --debug_performance off)
    for ((j=0; j<${#PARAMETERS[@]}; j++)); do
      command+=("--${PARAMETERS[j]}" "${row[j+1]}")
    done
    if ((DRY_RUN)); then
      printf '%q ' "${command[@]}"; printf '\n'
      continue
    fi
    mkdir -p -- "$run_dir"
    {
      printf '#!/usr/bin/env bash\nset -euo pipefail\ncd -- %q\n' "$REPO_ROOT"
      printf '%q ' "${command[@]}"; printf '\n'
    } > "$TEMP_DIR/command.sh"
    if [[ -f $run_dir/command.sh ]]; then
      cmp -s -- "$TEMP_DIR/command.sh" "$run_dir/command.sh" || die "Command changed for config_$cid/$name"
    else
      cp -- "$TEMP_DIR/command.sh" "$run_dir/command.sh"
    fi
    if [[ -f $run_dir/.complete ]] && (cd -- "$run_dir" && sha256sum --check --status .complete); then
      printf 'Already completed: config_%s / %s\n' "$cid" "$name"
      continue
    fi
    printf 'Running config_%s / %s (seed %s)\n' "$cid" "$name" "$RUN_SEED"
    if "${command[@]}" > "$run_dir/run.log" 2>&1; then
      outputs=("${name}_0_complement_Info.csv" "${name}_0_complement_RulesMetricsResult.csv" "${name}_0_complement_DetailedRules.csv")
      for file in "${outputs[@]}"; do
        [[ -s $run_dir/$file ]] || die "Missing output: $run_dir/$file"
      done
      (cd -- "$run_dir" && sha256sum command.sh "${outputs[@]}") > "$TEMP_DIR/complete"
      mv -- "$TEMP_DIR/complete" "$run_dir/.complete"
    else
      status=$?
      tail -n 30 -- "$run_dir/run.log" >&2
      printf 'Failed config_%s / %s. See %s/run.log\n' "$cid" "$name" "$run_dir" >&2
      exit "$status"
    fi
  done < "$OUTPUT_DIR/configs.tsv"
done
printf 'Manifests and results: %s\n' "$OUTPUT_DIR"
