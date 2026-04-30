#!/usr/bin/env bash
# VLM consistency (dual-agent rubrics + batched judge). Run from repo root.
# Requires API keys for the chosen --provider. Defaults use tests/vlm_judge paths.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CSV="tests/vlm_judge/Prompt - Basic_env.csv"

# --- 1) Generate and cache rubrics (once per rubric model / prompt_version) ---
python tools/evaluate_world_model_vlm_consistency.py generate_rubrics \
  --provider openai \
  --model gpt-5.4 \
  --csv "$CSV" \
  --rubric_dir rubrics \
  --ref_dir tests/vlm_judge/ref_img \
  --prompt_version v1

# --- 2) Score all videos under tests/vlm_judge/data/* ---
python tools/evaluate_world_model_vlm_consistency.py evaluate \
  --provider openai \
  --model gpt-5.4 \
  --csv "$CSV" \
  --rubric_dir rubrics \
  --data_dir tests/vlm_judge/data \
  --ref_dir tests/vlm_judge/ref_img \
  --prompt_version v1 \
  --output reports/world_model_vlm_consistency/results.json

# Optional: limit to specific generator folders and/or disable reference image
# python tools/evaluate_world_model_vlm_consistency.py evaluate \
#   --provider openai \
#   --model gpt-5.4 \
#   --model_names Genie Unity \
#   --no_ref \
#   --output reports/world_model_vlm_consistency/results_noref.json

# Offline / CLI smoke tests (same OpenCV dependency chain as other evaluation tests):
# python -m pytest tests/test_vlm_consistency_batch_helpers.py \
#   tests/test_vlm_consistency_batch_cli.py -q
