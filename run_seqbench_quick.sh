#!/usr/bin/env bash
set -eu

root="$(cd "$(dirname "$0")" && pwd)"
seqroot="${SEQBENCH_ROOT:-$root/../seqbench}"
algorithm="${1:-$root/adapters/seqbench/algorithm.yaml}"
data="${2:-$seqroot/data/screen-v1/tasks.csv}"
output="${3:-$root/runs/seqbench-screen-v1/candidate}"
baseline="${SEQBENCH_SCREEN_BASELINE:-$root/runs/seqbench-screen-v1/exact-knn}"
spec="$seqroot/specs/screens/screen_v1.yaml"

run_seed() {
  seed="$1"
  destination="$2"
  if [ -f "$destination/metrics.json" ]; then
    return
  fi
  "$seqroot/.venv/bin/seqbench" run "$spec" \
    --algorithm "$algorithm" \
    --tasks "$data" \
    --output "$destination" \
    --seed "$seed"
}

if [ ! -f "$baseline/metrics.json" ]; then
  mkdir -p "$(dirname "$baseline")"
  "$seqroot/.venv/bin/seqbench" run "$spec" \
    --algorithm "$root/adapters/seqbench/exact-knn.yaml" \
    --tasks "$data" \
    --output "$baseline" \
    --seed 0
fi

mkdir -p "$output"
run_seed 0 "$output/seed-0"

stage1="$output/screen-stage1"
if [ ! -f "$stage1/screen.json" ]; then
  "$seqroot/.venv/bin/seqbench" screen "$spec" \
    --candidate "$output/seed-0" \
    --baseline "$baseline" \
    --output "$stage1"
fi
decision="$("$seqroot/.venv/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' \
  "$stage1/screen.json")"

if [ "$decision" = "PROMOTE" ] || [ "$decision" = "DROP" ]; then
  echo "$stage1/report.md"
  exit 0
fi

run_seed 1 "$output/seed-1"
run_seed 2 "$output/seed-2"
final="$output/screen-final"
if [ ! -f "$final/screen.json" ]; then
  "$seqroot/.venv/bin/seqbench" screen "$spec" \
    --candidate "$output/seed-0" \
    --candidate "$output/seed-1" \
    --candidate "$output/seed-2" \
    --baseline "$baseline" \
    --output "$final"
fi
echo "$final/report.md"
