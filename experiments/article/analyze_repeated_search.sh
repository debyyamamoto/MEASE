#!/usr/bin/env bash
# Analyze repeated runs or the original Random Search (--mode single).
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$REPO_ROOT"
exec uv run --locked python experiments/analyze_repeated_search.py "$@"
