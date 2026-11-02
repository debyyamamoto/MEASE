#!/usr/bin/env bash
# Select and repeat the best configurations from the completed Random Search.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$REPO_ROOT"
exec uv run --locked python experiments/repeat_random_search.py "$@"
