#!/usr/bin/env bash
# Shadow-consistency evaluation for world-model videos.
#
# Runs GroundingDINO + SAM 2 object tracking and computes six internal-
# consistency metrics (SRR, SDR, MOSD, SDJ, SAS, SSD) without requiring
# a per-frame Unity ground-truth reference.
#
# Usage:
#   bash scripts/evaluate_world_model_shadow_consistency.sh \
#       --video path/to/generated.mp4 \
#       --text_prompt "table. chair. lamp." \
#       --anchor_label table \
#       --hud_preset genie \
#       --output_json reports/shadow_consistency.json
#
# Requires: torch, transformers, sam2, opencv-python-headless, Pillow

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

python "$REPO_ROOT/tools/evaluate_world_model_shadow_consistency.py" "$@"
