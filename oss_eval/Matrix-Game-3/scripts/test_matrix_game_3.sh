#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

NUM_GPUS="${NUM_GPUS:-2}"

torchrun --nproc_per_node="$NUM_GPUS" generate.py \
  --size 704*1280 \
  --dit_fsdp \
  --t5_fsdp \
  --ckpt_dir /raid/lah003/models/Matrix-Game-3.0 \
  --fa_version 2 \
  --use_int8 \
  --num_iterations 12 \
  --num_inference_steps 3 \
  --image demo_images/001/image.png \
  --prompt "A colorful, animated cityscape with a gas station and various buildings." \
  --save_name test \
  --seed 42 \
  --compile_vae \
  --lightvae_pruning_rate 0.5 \
  --vae_type mg_lightvae \
  --output_dir /raid/lah003/WMEval/output/Matrix-Game-3.0

# "num_iterations" refers to the number of iterations you want to generate.
# Total frames = 57 + (num_iterations - 1) * 40.
