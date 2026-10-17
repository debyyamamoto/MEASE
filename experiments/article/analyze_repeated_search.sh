#!/usr/bin/env bash
# Analyze the complete repeated experiment and write a Portuguese report.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$REPO_ROOT"
exec uv run --locked python experiments/analyze_repeated_search.py "$@"
