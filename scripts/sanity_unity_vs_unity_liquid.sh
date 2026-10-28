#!/usr/bin/env bash
# Floor-finding sanity check: run all four sloshing modules on a Unity
# reference vs Unity reference pair (either the same clip on both sides,
# or two clips from the same Unity scene). The numbers from this run are
# the noise floor — every WM-vs-Unity comparison's deltas should be read
# RELATIVE to these values, not against zero.
#
# Expected outcome for SELF-PAIR (same clip on both sides):
#   - liquid_velocity: epe ≈ 0, cosine ≈ 1, divergence_residual ≈ 0,
#                      curl_stats.mean_error ≈ 0, temporal_consistency ≈ 1
#   - liquid_volume_conservation: every delta ≈ 0
#   - liquid_surface_orientation: surface_tilt_residual_deg.delta ≈ 0
#   - slosh_spectrum: dominant_frequency_hz.ratio ≈ 1, delta_hz ≈ 0,
#                     spectral_l1 ≈ 0
#
# Expected outcome for CROSS-PAIR (Movie_001 vs Movie_002, etc.):
#   non-zero but small deltas — that's the irreducible Unity-vs-Unity
#   noise floor introduced by render seed / stochastic shader state.
#
# Usage:
#   bash WMEval/scripts/sanity_unity_vs_unity_liquid.sh                  # self-pair, default video
#   bash WMEval/scripts/sanity_unity_vs_unity_liquid.sh REF              # self-pair on a custom video
#   bash WMEval/scripts/sanity_unity_vs_unity_liquid.sh REF CAND         # cross-pair
#   bash WMEval/scripts/sanity_unity_vs_unity_liquid.sh REF CAND OUT     # cross-pair, custom output path
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

DEFAULT_VIDEO="/workspace/WMEval/unity/LiquidContainer/Recordings/Movie_001.mp4"
REF_VIDEO="${1:-$DEFAULT_VIDEO}"
CAND_VIDEO="${2:-$REF_VIDEO}"
OUT="${3:-reports/world_model_liquid_sanity/identity.json}"

case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac

for f in "$REF_VIDEO" "$CAND_VIDEO"; do
  if [ ! -f "$f" ]; then
    echo "sanity_unity_vs_unity_liquid: missing video: $f" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$OUT")"

# Same normalize step as the full eval, so floor numbers are computed under
# the exact same preprocessing as WM-vs-Unity comparisons.
# shellcheck source=normalize_fps.sh
source "$SCRIPT_DIR/normalize_fps.sh"

WORK_DIR="$(mktemp -d -t wmeval_liquid_sanity.XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

REF_NORM="$WORK_DIR/ref_norm.mp4"
CAND_NORM="$WORK_DIR/cand_norm.mp4"
normalize_video "$REF_VIDEO" "$REF_NORM"
if [ "$REF_VIDEO" = "$CAND_VIDEO" ]; then
  cp "$REF_NORM" "$CAND_NORM"
else
  normalize_video "$CAND_VIDEO" "$CAND_NORM"
fi

python tools/legacy/evaluate_world_model_liquid.py \
  --video "$CAND_NORM" \
  --ref_video "$REF_NORM" \
  --modules liquid_velocity,liquid_volume_conservation,liquid_surface_orientation,slosh_spectrum \
  --liquid_cache_root runs/liquid_cache \
  --liquid_sam2_model facebook/sam2.1-hiera-small \
  --liquid_sam2_prompt "the liquid inside the bottle" \
  --liquid_bottle_prompt "the glass bottle" \
  --liquid_piv_window_size 32 \
  --liquid_piv_overlap 16 \
  --liquid_flow_estimator raft \
  --liquid_camera_compensation \
  --liquid_camera_regime c2_static_side \
  --output_json "$OUT" \
  --eval_fps 4 \
  --max_pairs 512 \
  --liquid_max_pairs 0

echo "wrote $OUT"
echo "  inspect with: python tools/analysis/inspect_liquid_report.py \"$OUT\""
