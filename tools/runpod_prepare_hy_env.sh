#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runpod_wmeval_preflight.sh" --env-only
ENV_PATH="$WMEVAL_RUNPOD_ROOT/envs/hy-worldplay-py312-torch280"
python3 -m venv --system-site-packages "$ENV_PATH"
"$ENV_PATH/bin/python" -m pip install 'numpy==1.26.4' 'opencv-python-headless==4.11.0.86' 'diffusers==0.35.0' 'transformers==4.56.0' 'accelerate==1.12.0' 'huggingface-hub==0.36.0' 'safetensors==0.4.5' 'peft==0.17.0' 'openai==2.8.0' 'einops==0.8.0' 'loguru==0.7.3' 'pillow==11.3.0' 'imageio==2.37.0' 'imageio-ffmpeg==0.6.0' 'omegaconf==2.3.0' 'qwen-vl-utils==0.0.8' 'pandas==2.2.3' 'scipy==1.15.3' 'protobuf' 'moviepy==1.0.3' 'cloudpickle' 'ftfy' 'tiktoken'
"$ENV_PATH/bin/python" - <<'PY'
import torch, json
assert torch.__version__.startswith('2.8.'),torch.__version__
assert torch.version.cuda == '12.8',torch.version.cuda
assert (torch.ones((32,32),device='cuda')@torch.ones((32,32),device='cuda')).sum().item()==32768
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu_kernel_verified':True}))
PY

"$ENV_PATH/bin/python" -m pip freeze > "$ENV_PATH/dependency_freeze.txt"
