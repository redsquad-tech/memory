#!/usr/bin/env bash
set -eu

root="$(cd "$(dirname "$0")" && pwd)"
seqroot="${SEQBENCH_ROOT:-$root/../seqbench}"
mode="${1:-pilot}"
data="${2:-$seqroot/data/causal-v2}"
output="${3:-$root/runs/seqbench-causal-v2/$mode}"

case "$mode" in
  pilot)
    seeds="0"
    limits="--train-limit 2000 --eval-limit 200"
    jobs=1
    ;;
  full)
    seeds="0 1 2 3 4"
    limits=""
    jobs="${SEQBENCH_JOBS:-4}"
    ;;
  *)
    echo "usage: $0 [pilot|full] [DATA_DIR] [OUTPUT_DIR]" >&2
    exit 2
    ;;
esac

tasks=""
for file in \
  babi-full.csv babi-oracle.csv babilong-full.csv \
  clutrr-full.csv clutrr-nonce.csv clutrr-nonce-counterfactual.csv clutrr-noise.csv \
  proofwriter-full.csv proofwriter-proof-context.csv \
  recogs-full.csv slog-full.csv
do
  tasks="$tasks --tasks $data/$file"
done

mkdir -p "$output"
runs=""
active_pids=""
active_names=""

wait_oldest() {
  oldest_pid="${active_pids%% *}"
  oldest_name="${active_names%% *}"
  wait "$oldest_pid"
  echo "[$oldest_name] complete"
  active_pids="${active_pids#* }"
  active_names="${active_names#* }"
  if [ "$active_pids" = "$oldest_pid" ]; then
    active_pids=""
    active_names=""
  fi
}

for entry in \
  "learned:algorithm.yaml" \
  "frozen:frozen.yaml" \
  "exact_knn:exact-knn.yaml"
do
  label="${entry%%:*}"
  algorithm="${entry#*:}"
  model_seeds="$seeds"
  if [ "$mode" = "full" ] && [ "$label" = "exact_knn" ]; then
    model_seeds="0"
  fi
  for seed in $model_seeds
  do
    destination="$output/$label/seed-$seed"
    if [ -f "$destination/metrics.json" ]; then
      echo "[$label seed=$seed] complete"
    else
      echo "[$label seed=$seed] running; log=$destination.runner.log"
      (
        # shellcheck disable=SC2086
        "$seqroot/.venv/bin/seqbench" run \
          "$seqroot/specs/runs/causal_v2.yaml" \
          --algorithm "$root/adapters/seqbench/$algorithm" \
          $tasks --output "$destination" --seed "$seed" $limits
      ) >"$destination.runner.log" 2>&1 &
      pid="$!"
      active_pids="${active_pids:+$active_pids }$pid"
      active_names="${active_names:+$active_names }$label-seed-$seed"
      set -- $active_pids
      if [ "$#" -ge "$jobs" ]; then
        wait_oldest
      fi
    fi
    runs="$runs --run $label=$destination"
  done
done

while [ -n "$active_pids" ]; do
  wait_oldest
done

comparison="$output/comparison"
if [ -e "$comparison" ]; then
  echo "$comparison already exists; choose a new output directory" >&2
  exit 2
fi
# shellcheck disable=SC2086
"$seqroot/.venv/bin/seqbench" compare \
  "$seqroot/specs/runs/causal_v2.yaml" \
  $runs --reference learned --output "$comparison"

echo "$comparison/comparison.md"
