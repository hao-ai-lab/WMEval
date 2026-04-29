#!/usr/bin/env bash
# Run the full LiquidContainer eval (all 4 sloshing modules + cross-cutting
# perceptual modules) for a single (Unity reference, WM candidate) pair.
#
# Usage: bash WMEval/scripts/evaluate_world_model_liquid_full.sh REF_VIDEO WM_VIDEO OUT_JSON [WM_NAME]
#   REF_VIDEO  Path to the Unity reference recording (e.g. Movie_001.mp4)
#   WM_VIDEO   Path to the world-model candidate video
#   OUT_JSON   Where to write the eval report
#   WM_NAME    Optional label saved into the report's "model" field
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 REF_VIDEO WM_VIDEO OUT_JSON [WM_NAME]" >&2
  exit 1
fi

REF_VIDEO="$1"
WM_VIDEO="$2"
OUT_JSON="$3"
WM_NAME="${4:-$(basename "$WM_VIDEO" .mp4)}"

for f in "$REF_VIDEO" "$WM_VIDEO"; do
  if [ ! -f "$f" ]; then
    echo "evaluate_world_model_liquid_full: missing video: $f" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$OUT_JSON")"

python tools/evaluate_world_model_diffusionlight.py \
  --video "$WM_VIDEO" \
  --ref_video "$REF_VIDEO" \
  --modules liquid_velocity,liquid_volume_conservation,liquid_surface_orientation,slosh_spectrum,lpips,control_motion,clip4clip \
  --liquid_cache_root runs/liquid_cache \
  --liquid_sam2_model facebook/sam2.1-hiera-small \
  --liquid_sam2_prompt "the liquid inside the bottle" \
  --liquid_piv_window_size 32 \
  --liquid_piv_overlap 16 \
  --liquid_camera_regime c2_static_side \
  --lpips_vlm_scene_key liquid_container \
  --output_json "$OUT_JSON" \
  --eval_fps 4 \
  --max_pairs 32

# Tag the report with the model name for downstream leaderboard merging.
python -c "
import json, sys
p = sys.argv[1]
name = sys.argv[2]
with open(p) as f:
    rep = json.load(f)
rep['model'] = name
with open(p, 'w') as f:
    json.dump(rep, f, indent=2)
" "$OUT_JSON" "$WM_NAME"

echo "wrote $OUT_JSON (model=$WM_NAME)"
