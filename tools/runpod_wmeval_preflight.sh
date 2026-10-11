#!/usr/bin/env bash
# Inspect a mounted Runpod workspace before downloading or running models.
set -euo pipefail
if [[ ! -d /workspace ]] || ! mountpoint -q /workspace; then
  echo 'Refusing to proceed: /workspace is not a mounted volume.' >&2
  exit 2
fi
export WMEVAL_RUNPOD_ROOT=/workspace/WMEval-lighting-orbit
mkdir -p "$WMEVAL_RUNPOD_ROOT"/{cache/{pip,huggingface,torch,cuda,triton,uv},tmp,logs,envs,model_sources,models,outputs}
export TMPDIR="$WMEVAL_RUNPOD_ROOT/tmp"
export PIP_CACHE_DIR="$WMEVAL_RUNPOD_ROOT/cache/pip"
export PIP_REQUIRE_VIRTUALENV=true
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$WMEVAL_RUNPOD_ROOT/cache"
export HF_HOME="$WMEVAL_RUNPOD_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$WMEVAL_RUNPOD_ROOT/cache/torch"
export CUDA_CACHE_PATH="$WMEVAL_RUNPOD_ROOT/cache/cuda"
export TRITON_CACHE_DIR="$WMEVAL_RUNPOD_ROOT/cache/triton"
export UV_CACHE_DIR="$WMEVAL_RUNPOD_ROOT/cache/uv"
export PYTHONUNBUFFERED=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
# Sourcing this mode only prepares paths and environment; skip repeated scans.
if [[ ${1:-} == --env-only ]]; then
  return 0 2>/dev/null || exit 0
fi
df -h /workspace /
command -v python python3 ffmpeg git || true
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
fi
python3 - <<'PY'
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys

names = ('torch', 'torchvision', 'flash-attn', 'numpy', 'transformers',
         'diffusers', 'accelerate', 'huggingface-hub', 'sam2', 'opencv-python-headless')
packages = {}
for name in names:
    try:
        packages[name] = md.version(name)
    except md.PackageNotFoundError:
        packages[name] = None
print(json.dumps({'python': sys.version, 'packages': packages}, indent=2))
try:
    import torch
    print(json.dumps({'torch_cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
                      'compiled_arches': torch.cuda.get_arch_list()}, indent=2))
    for index in range(torch.cuda.device_count()):
        # Check this wheel actually executes on each GPU, especially Blackwell.
        value = torch.ones((32, 32), device=f'cuda:{index}')
        result = (value @ value).sum().item()
        print(json.dumps({'gpu': index, 'capability': torch.cuda.get_device_capability(index),
                          'kernel_check': result == 32768.0}))
except Exception as exc:
    print(json.dumps({'torch_probe_error': str(exc)}))
matches = []
for parent, directories, _ in os.walk('/workspace'):
    rel = Path(parent).relative_to('/workspace')
    if len(rel.parts) >= 4:
        directories[:] = []
        continue
    directories[:] = [n for n in directories if n not in {'.git', '__pycache__', 'node_modules', 'tmp'}]
    for name in directories:
        if any(token in name.lower() for token in ('wmeval', 'lingbot', 'matrix', 'worldplay', 'waypoint', 'sam2', 'grounding')):
            matches.append(str(Path(parent) / name))
print(json.dumps({'existing_directories': matches}, indent=2))
PY
