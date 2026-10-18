#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_path="${WMEVAL_GPU_VENV:-${repo_root}/.venv-gpu}"
uv_cache_path="${UV_CACHE_DIR:-${repo_root}/.uv-cache}"

cd "${repo_root}"
export UV_CACHE_DIR="${uv_cache_path}"
uv venv --python 3.10 "${venv_path}"
uv pip install --python "${venv_path}/bin/python" \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "${venv_path}/bin/python" -r requirements-gpu.lock
"${venv_path}/bin/python" tools/preflight_tables_4_7_gpu.py
