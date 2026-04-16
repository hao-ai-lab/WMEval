#!/usr/bin/env bash
set -euo pipefail
# Run from this script's directory so generate.py resolves reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

CKPT_DIR="Matrix-Game-3.0"
SYNC_GPU_NUM=8
ASYNC_GPU_NUM=7

ARGS=(
  --size 704*1280
  --dit_fsdp
  --t5_fsdp
  --ckpt_dir ${CKPT_DIR}
  --output_dir ./output
  --fa_version 3
  --use_int8
  --num_iterations 12
  --num_inference_steps 3
  --image demo_images/001/image.png
  --prompt "A colorful, animated cityscape with a gas station and various buildings."
  --save_name test
  --seed 42
  --interactive
  --compile_vae
  --lightvae_pruning_rate 0.5
  --vae_type mg_lightvae
  --use_async_vae
  --async_vae_warmup_iters 1
)

USE_ASYNC_VAE=false
for arg in "${ARGS[@]}"; do
  if [ "$arg" == "--use_async_vae" ]; then
    USE_ASYNC_VAE=true
    break
  fi
done

if [ "$USE_ASYNC_VAE" = true ]; then
  CURRENT_GPU_NUM=${ASYNC_GPU_NUM}
else
  CURRENT_GPU_NUM=${SYNC_GPU_NUM}
fi

ULYSSES_SIZE=${CURRENT_GPU_NUM}
ARGS+=( --ulysses_size "${ULYSSES_SIZE}" )

exec torchrun \
  --nproc_per_node="${CURRENT_GPU_NUM}" \
  "${SCRIPT_DIR}/generate.py" \
  "${ARGS[@]}"
