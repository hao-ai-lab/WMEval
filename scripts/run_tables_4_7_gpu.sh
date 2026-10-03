#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_path="${WMEVAL_GPU_VENV:-${repo_root}/.venv-gpu}"
report_suffix="${WMEVAL_REPORT_SUFFIX:-cuda_release}"

cd "${repo_root}"
"${venv_path}/bin/python" tools/preflight_tables_4_7_gpu.py
"${venv_path}/bin/python" tools/run_prepared_table_evaluations.py \
  --device cuda \
  --fail-incomplete \
  --report-suffix "${report_suffix}"
