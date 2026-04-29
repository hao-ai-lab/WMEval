#!/usr/bin/env bash
# Identity sanity check: run all liquid sloshing modules with the same Unity
# video on both sides. Expected outcome:
#   - liquid_velocity: epe ≈ 0, cosine ≈ 1, divergence_residual ≈ 0,
#                      curl_stats.mean_error ≈ 0, temporal_consistency ≈ 1
#   - liquid_volume_conservation: every delta ≈ 0
#   - liquid_surface_orientation: surface_tilt_residual_deg.delta ≈ 0
#   - slosh_spectrum: dominant_frequency_hz.ratio ≈ 1, spectral_l1 ≈ 0
#
# Usage: bash WMEval/scripts/sanity_unity_vs_unity_liquid.sh [VIDEO_PATH]
set -euo pipefail

VIDEO="${1:-/Users/abhilash/Documents/Work/WorldModelBench/WorldModelBenchScenePrivate/GameScenes/LiquidContainer/Recordings/Movie_001.mp4}"
OUT="reports/world_model_liquid_sanity/identity.json"

if [ ! -f "$VIDEO" ]; then
  echo "sanity_unity_vs_unity_liquid: missing video: $VIDEO" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

python tools/evaluate_world_model_diffusionlight.py \
  --video "$VIDEO" \
  --ref_video "$VIDEO" \
  --modules liquid_velocity,liquid_volume_conservation,liquid_surface_orientation,slosh_spectrum \
  --liquid_cache_root runs/liquid_cache \
  --liquid_sam2_model facebook/sam2.1-hiera-small \
  --liquid_sam2_prompt "the liquid inside the bottle" \
  --liquid_piv_window_size 32 \
  --liquid_piv_overlap 16 \
  --liquid_camera_regime c2_static_side \
  --output_json "$OUT" \
  --eval_fps 4 \
  --max_pairs 32

echo "wrote $OUT"
