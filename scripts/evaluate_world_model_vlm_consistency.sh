#!/usr/bin/env bash
# VLM consistency (dual-agent rubrics + batched judge). Run from repo root.
# Requires API keys for the chosen --provider.
#
# Pipeline:
#   1) build_manifest  -- scan {VIEW_DIR}/scenes.csv + {VIEW_DIR}/data/* into manifest.jsonl
#   2) generate_rubrics -- LLM generates env+obj rubrics per scene, cached under {RUBRIC_DIR}
#   3) evaluate         -- VLM judges every video against its scene's cached rubric

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Update VIEW_DIR to point at your evaluation set. The directory must contain:
#   {VIEW_DIR}/scenes.csv         columns: scene_name,prompt[,ref_path]
#   {VIEW_DIR}/data/{scene}/*.mp4 generated videos, one folder per scene
#   {VIEW_DIR}/refs/*.png|jpg     (optional) reference frames pointed at by ref_path
VIEW_DIR="vlm_test_data/test_first_person"
RUBRIC_DIR="${VIEW_DIR}/rubrics"
PROVIDER="openai"
RUBRIC_MODEL="gpt-5.4"
JUDGE_MODEL="gpt-5.4"
PROMPT_VERSION="v1"
OUTPUT="reports/world_model_vlm_consistency/$(basename "${VIEW_DIR}")_results.jsonl"

# --- 1) Build manifest from scenes.csv + data/{scene}/*.mp4 ---
python tools/legacy/evaluate_world_model_vlm_consistency.py build_manifest \
  --view_dir "${VIEW_DIR}"

# --- 2) Generate and cache rubrics (once per rubric model / prompt_version) ---
python tools/legacy/evaluate_world_model_vlm_consistency.py generate_rubrics \
  --manifest "${VIEW_DIR}/manifest.jsonl" \
  --rubric_dir "${RUBRIC_DIR}" \
  --provider "${PROVIDER}" \
  --model "${RUBRIC_MODEL}" \
  --prompt_version "${PROMPT_VERSION}"

# --- 3) Score all videos in the manifest ---
python tools/legacy/evaluate_world_model_vlm_consistency.py evaluate \
  --manifest "${VIEW_DIR}/manifest.jsonl" \
  --rubric_dir "${RUBRIC_DIR}" \
  --provider "${PROVIDER}" \
  --model "${JUDGE_MODEL}" \
  --rubric_provider "${PROVIDER}" \
  --rubric_model "${RUBRIC_MODEL}" \
  --prompt_version "${PROMPT_VERSION}" \
  --output "${OUTPUT}" \
  --resume

# Optional: skip reference image (rubrics + judge run on prompt + frames only)
# python tools/legacy/evaluate_world_model_vlm_consistency.py evaluate \
#   --manifest "${VIEW_DIR}/manifest.jsonl" \
#   --rubric_dir "${RUBRIC_DIR}" \
#   --provider "${PROVIDER}" --model "${JUDGE_MODEL}" \
#   --prompt_version "${PROMPT_VERSION}" \
#   --no_ref \
#   --output "${OUTPUT%.jsonl}_noref.jsonl"

# Offline / CLI smoke tests (require pytest + OpenCV in the active env):
# python -m pytest tests/metrics/test_vlm_consistency_batch_helpers.py \
#   tests/metrics/test_vlm_consistency_batch_cli.py -q
