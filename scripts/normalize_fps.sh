#!/usr/bin/env bash
# Frame-rate normalization helper for the LiquidContainer eval pipeline.
#
# Provides:
#   normalize_fps INPUT_VIDEO OUTPUT_VIDEO [TARGET_FPS]
#       Probes INPUT_VIDEO's frame rate. If equal to TARGET_FPS within
#       tolerance, copies the file. If above TARGET_FPS, downsamples via
#       nearest-frame selection (no temporal interpolation, no fabricated
#       frames). If below TARGET_FPS, errors out — the WMEval protocol
#       requires every world model to produce at least the canonical fps,
#       so we refuse to fabricate frames into a slower model's output.
#
# Designed to be sourced, not executed directly:
#   source "$(dirname "$0")/normalize_fps.sh"
#   normalize_fps "$WM_VIDEO" "$WORK_DIR/wm_24.mp4"
#
# Default TARGET_FPS is 24, which is the WMEval canonical rate (highest
# fps achievable by every supported world model). Override per call if a
# scene needs a different rate.

CANONICAL_FPS_DEFAULT="${WMEVAL_CANONICAL_FPS:-24}"
FPS_TOLERANCE=0.1

# Internal: parse "30000/1001" or "24/1" rationals into a decimal fps.
_fps_to_decimal() {
  local rational="$1"
  awk -F/ 'BEGIN{ OFMT="%.6f" } { if ($2 == 0) print "0"; else print $1/$2 }' <<<"$rational"
}

normalize_fps() {
  local in="$1"
  local out="$2"
  local target="${3:-$CANONICAL_FPS_DEFAULT}"

  if [ -z "$in" ] || [ -z "$out" ]; then
    echo "normalize_fps: usage: normalize_fps INPUT OUTPUT [TARGET_FPS]" >&2
    return 2
  fi
  if [ ! -f "$in" ]; then
    echo "normalize_fps: input not found: $in" >&2
    return 2
  fi
  if ! command -v ffprobe >/dev/null 2>&1; then
    echo "normalize_fps: ffprobe not on PATH" >&2
    return 3
  fi
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "normalize_fps: ffmpeg not on PATH" >&2
    return 3
  fi

  local rational
  rational=$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=r_frame_rate -of default=nw=1:nk=1 "$in")
  if [ -z "$rational" ]; then
    echo "normalize_fps: could not read frame rate from $in" >&2
    return 4
  fi

  local src_fps
  src_fps=$(_fps_to_decimal "$rational")

  # Reject sources slower than the canonical rate. Up-sampling would force
  # us to either duplicate frames (collapses PIV velocity to zero on the
  # duplicated pairs) or interpolate (fabricates motion the WM never
  # predicted). Either choice corrupts the metric, so we refuse instead.
  local below
  below=$(awk -v s="$src_fps" -v t="$target" -v tol="$FPS_TOLERANCE" \
    'BEGIN { print (s < t - tol) ? 1 : 0 }')
  if [ "$below" = "1" ]; then
    echo "normalize_fps: $in is ${src_fps} fps, below WMEval floor of ${target}" >&2
    echo "  Up-sampling would fabricate frames; refusing." >&2
    return 5
  fi

  mkdir -p "$(dirname "$out")"

  local at_target
  at_target=$(awk -v s="$src_fps" -v t="$target" -v tol="$FPS_TOLERANCE" \
    'BEGIN { print (s >= t - tol && s <= t + tol) ? 1 : 0 }')
  if [ "$at_target" = "1" ]; then
    # Already at target — avoid a re-encode that would change pixel values.
    cp "$in" "$out"
    echo "normalize_fps: $in already at ${src_fps} fps, copied to $out" >&2
    return 0
  fi

  # Downsample. The `fps` filter performs nearest-source-frame selection
  # at the requested output rate; it does NOT temporally interpolate, so
  # every output frame is a genuine frame the source produced.
  echo "normalize_fps: downsampling $in (${src_fps} fps) -> $out (${target} fps)" >&2
  ffmpeg -y -loglevel error -i "$in" -vf "fps=${target}" \
    -c:v libx264 -crf 18 -pix_fmt yuv420p "$out"
}

# normalize_video INPUT OUTPUT [TARGET_FPS] [TARGET_W] [TARGET_H]
#
# Combined fps + spatial resize. Different world models render at different
# resolutions (e.g. lingbot 832x464 vs Unity 1280x720); without spatial
# normalization the PIV grids are incompatible shapes and EPE/cosine across
# Unity and WM is undefined. We resize with bilinear (no aspect-ratio
# adjustment — caller must pick W/H matching the source aspect) and then
# apply the fps filter. Defaults are tuned for the LiquidContainer scene:
# 832x464 (lingbot's native size, the constrained side) at 16 fps (lingbot's
# native rate, the constrained side; Unity is downsampled to match).
normalize_video() {
  local in="$1"
  local out="$2"
  local target_fps="${3:-${WMEVAL_CANONICAL_FPS:-16}}"
  local target_w="${4:-${WMEVAL_CANONICAL_W:-832}}"
  local target_h="${5:-${WMEVAL_CANONICAL_H:-464}}"

  if [ -z "$in" ] || [ -z "$out" ]; then
    echo "normalize_video: usage: normalize_video INPUT OUTPUT [FPS] [W] [H]" >&2
    return 2
  fi
  if [ ! -f "$in" ]; then
    echo "normalize_video: input not found: $in" >&2
    return 2
  fi

  mkdir -p "$(dirname "$out")"
  echo "normalize_video: $in -> $out (${target_w}x${target_h} @ ${target_fps} fps)" >&2
  ffmpeg -y -loglevel error -i "$in" \
    -vf "scale=${target_w}:${target_h}:flags=bilinear,fps=${target_fps}" \
    -c:v libx264 -crf 18 -pix_fmt yuv420p "$out"
}
