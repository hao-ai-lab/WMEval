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

# Resolve OUT_JSON relative to the caller's CWD before we cd into WMEval/.
case "$OUT_JSON" in
  /*) ;;
  *) OUT_JSON="$(pwd)/$OUT_JSON" ;;
esac

# Run from WMEval/ so relative paths (tools/, runs/) resolve consistently.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

for f in "$REF_VIDEO" "$WM_VIDEO"; do
  if [ ! -f "$f" ]; then
    echo "evaluate_world_model_liquid_full: missing video: $f" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$OUT_JSON")"

# Normalize both videos to a common (W, H, fps) before the eval runs.
# WMEval canonical for LiquidContainer is 832x464 @ 16 fps — the lingbot
# native resolution/rate. Unity (1280x720 @ 30 fps) is downsampled to match.
# Without this:
#   - PIV grid shapes differ across WMs and EPE/cosine break.
#   - fps mismatch silently scales pixel velocities and FFT frequencies.
# Override via WMEVAL_CANONICAL_FPS / WMEVAL_CANONICAL_W / WMEVAL_CANONICAL_H.
# shellcheck source=normalize_fps.sh
source "$SCRIPT_DIR/normalize_fps.sh"

WORK_DIR="$(mktemp -d -t wmeval_liquid.XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

REF_VIDEO_NORM="$WORK_DIR/ref_norm.mp4"
WM_VIDEO_NORM="$WORK_DIR/wm_norm.mp4"
normalize_video "$REF_VIDEO" "$REF_VIDEO_NORM"
normalize_video "$WM_VIDEO"  "$WM_VIDEO_NORM"

python tools/evaluate_world_model_liquid.py \
  --video "$WM_VIDEO_NORM" \
  --ref_video "$REF_VIDEO_NORM" \
  --modules liquid_velocity,liquid_volume_conservation,liquid_surface_orientation,slosh_spectrum,lpips,control_motion,clip4clip \
  --liquid_cache_root runs/liquid_cache \
  --liquid_sam2_model facebook/sam2.1-hiera-small \
  --liquid_sam2_prompt "the liquid inside the bottle" \
  --liquid_bottle_prompt "the glass bottle" \
  --liquid_piv_window_size 32 \
  --liquid_piv_overlap 16 \
  --liquid_flow_estimator raft \
  --liquid_camera_compensation \
  --liquid_camera_regime c2_static_side \
  --lpips_vlm_scene_key liquid_container \
  --output_json "$OUT_JSON" \
  --eval_fps 4 \
  --max_pairs 512 \
  --liquid_max_pairs 0

# Tag the report with the model name for downstream leaderboard merging.
# Atomic: write to a temp file then os.replace so an interrupt cannot
# leave OUT_JSON empty/partial.
python - "$OUT_JSON" "$WM_NAME" <<'PY'
import json, os, sys
p, name = sys.argv[1], sys.argv[2]
with open(p) as f:
    rep = json.load(f)
rep["model"] = name
tmp = p + ".tmp"
with open(tmp, "w") as f:
    json.dump(rep, f, indent=2)
os.replace(tmp, p)
PY

echo "wrote $OUT_JSON (model=$WM_NAME)"
