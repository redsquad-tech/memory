#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

python_bin="${PYTHON:-$repo_dir/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    python_bin="$(command -v python3)"
fi

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m assocmem.experiments.banking77_paper
