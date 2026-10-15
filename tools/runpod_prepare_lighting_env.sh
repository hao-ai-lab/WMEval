#!/usr/bin/env bash
# Run on the cheap preparation Pod; all new files stay on the network volume.
set -euo pipefail
source "$(dirname "$0")/runpod_wmeval_preflight.sh"
ENV_DIR="$WMEVAL_RUNPOD_ROOT/envs/lighting-py312-torch280"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  python3 -m venv --system-site-packages "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
  'numpy==1.26.4' 'opencv-python-headless==4.10.0.84' 'scipy==1.15.3' \
  'diffusers==0.33.1' 'transformers==4.51.3' 'tokenizers==0.21.4' \
  'huggingface-hub==0.36.0' 'accelerate==1.12.0' 'imageio[ffmpeg]==2.37.0' \
  'PyYAML==6.0.2' 'easydict==1.13' 'ftfy==6.3.1' 'einops==0.8.1' \
  'packaging==25.0' 'ninja==1.13.0' 'psutil==7.1.0' 'safetensors==0.6.2'
python - <<'PY'
import torch
import transformers
import numpy
import cv2
from pathlib import Path
import os
assert torch.__version__ == '2.8.0+cu128', torch.__version__
assert torch.version.cuda == '12.8'
assert Path(os.environ['VIRTUAL_ENV']).is_relative_to('/workspace')
print({'torch': torch.__version__, 'transformers': transformers.__version__,
       'numpy': numpy.__version__, 'cv2': cv2.__version__, 'environment_ready': True})
PY
python -m pip freeze > "$WMEVAL_RUNPOD_ROOT/logs/lighting-environment-freeze.txt"
