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

# Normalize both videos to the WMEval canonical 24 fps before the eval runs.
# Different world models render at different native rates; without this the
# pipeline would silently misreport velocities and slosh frequencies for any
# WM whose fps disagrees with the reference. See WMEval/scripts/normalize_fps.sh.
# shellcheck source=normalize_fps.sh
source "$SCRIPT_DIR/normalize_fps.sh"

WORK_DIR="$(mktemp -d -t wmeval_liquid.XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

REF_VIDEO_24="$WORK_DIR/ref_24.mp4"
WM_VIDEO_24="$WORK_DIR/wm_24.mp4"
normalize_fps "$REF_VIDEO" "$REF_VIDEO_24"
normalize_fps "$WM_VIDEO"  "$WM_VIDEO_24"

python tools/evaluate_world_model_liquid.py \
  --video "$WM_VIDEO_24" \
  --ref_video "$REF_VIDEO_24" \
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
