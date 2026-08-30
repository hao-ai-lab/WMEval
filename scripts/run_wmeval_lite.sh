#!/usr/bin/env bash
# WMEval Lite — thin wrapper around tools/run_wmeval_lite.py.
#
# Usage:
#   bash scripts/run_wmeval_lite.sh run --model-root runs/genie3 --output-dir reports/wmeval_lite/genie3
#   bash scripts/run_wmeval_lite.sh validate
#   bash scripts/run_wmeval_lite.sh list-cases
#
# Environment:
#   PYTHON  python interpreter to use (default: python3)
#   API keys are loaded by the CLI from <repo>/.env when not already exported.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

exec "$PYTHON" "$ROOT/tools/run_wmeval_lite.py" "$@"
