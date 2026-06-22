#!/usr/bin/env bash
# eval_lite — Prompt-consistency (vlm_consistency) one-command runner.
# Self-contained bundle: scores the bundled Unity groundtruth videos against
# LLM-generated rubrics built from each scene's prompt. Run from anywhere.
#
# Single source of truth: eval_lite/manifest.jsonl (committed, hand/Claude-edited,
# like eval_sets/). There is NO scenes.csv and no build step — add a scene by
# appending one JSON line to the manifest.
#
# Pipeline (reuses the existing, tested batch pipeline):
#   1) generate_rubrics -- LLM generates env+obj rubrics per scene (cached/committed)
#   2) evaluate         -- VLM judges each video against its scene's rubric
#
# Requirements:
#   export OPENAI_API_KEY=...      (rubric generation + judging both call OpenAI)
#
# Usage:
#   bash eval_lite/run_eval_lite.sh                 # reuse committed rubrics
#   OVERWRITE_RUBRICS=1 bash eval_lite/run_eval_lite.sh   # regenerate & overwrite rubrics
#   MODEL=gpt-5.4 bash eval_lite/run_eval_lite.sh   # override OpenAI model

set -euo pipefail

# Resolve repo root (this script lives in <root>/eval_lite/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Load the OpenAI key. Precedence: existing env var > <root>/.env file.
# Users copy .env.example -> .env and paste their key (".env" is git-ignored).
if [[ -z "${OPENAI_API_KEY:-}" && -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set." >&2
  echo "  Fix: cp .env.example .env  &&  edit .env to paste your key" >&2
  echo "  (or: export OPENAI_API_KEY=sk-...)" >&2
  exit 1
fi

VIEW_DIR="eval_lite"
RUBRIC_DIR="${VIEW_DIR}/rubrics"
PROVIDER="openai"
MODEL="${MODEL:-gpt-4.1}"          # override via: MODEL=gpt-5.4 bash ...
PROMPT_VERSION="${PROMPT_VERSION:-v1}"
ALPHA="${ALPHA:-0.6}"
OUTPUT="reports/eval_lite/results.jsonl"

# Python interpreter must have the repo deps (torch, cv2, pydantic, numpy).
# Override if your default `python` lacks them, e.g.:
#   PYTHON=/Users/<you>/anaconda3/envs/ray_env/bin/python bash eval_lite/run_eval_lite.sh
PYTHON="${PYTHON:-python}"
CLI="$PYTHON tools/evaluate_world_model_vlm_consistency.py"

# Optional: OVERWRITE_RUBRICS=1 regenerates rubrics and overwrites the committed ones.
OVERWRITE_FLAG=""
if [[ "${OVERWRITE_RUBRICS:-0}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
  echo ">> OVERWRITE_RUBRICS=1: rubrics will be regenerated and overwritten." >&2
fi

# manifest.jsonl is the committed, hand/Claude-maintained source of truth.
MANIFEST="${VIEW_DIR}/manifest.jsonl"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: ${MANIFEST} not found." >&2
  echo "  This bundle is manifest-driven (no scenes.csv). Add scenes by editing" >&2
  echo "  ${MANIFEST}: one JSON object per line with keys scene_name, prompt," >&2
  echo "  ref_path, video_path, model_name, run_id." >&2
  exit 1
fi

echo ">> [1/2] generate_rubrics (provider=${PROVIDER} model=${MODEL} version=${PROMPT_VERSION})" >&2
$CLI generate_rubrics \
  --manifest "${VIEW_DIR}/manifest.jsonl" \
  --rubric_dir "${RUBRIC_DIR}" \
  --provider "${PROVIDER}" \
  --model "${MODEL}" \
  --prompt_version "${PROMPT_VERSION}" \
  ${OVERWRITE_FLAG}

echo ">> [2/2] evaluate (alpha=${ALPHA}) -> ${OUTPUT}" >&2
mkdir -p "$(dirname "${OUTPUT}")"
$CLI evaluate \
  --manifest "${VIEW_DIR}/manifest.jsonl" \
  --rubric_dir "${RUBRIC_DIR}" \
  --provider "${PROVIDER}" \
  --model "${MODEL}" \
  --rubric_provider "${PROVIDER}" \
  --rubric_model "${MODEL}" \
  --prompt_version "${PROMPT_VERSION}" \
  --alpha "${ALPHA}" \
  --output "${OUTPUT}" \
  --resume

echo ">> Done. Results: ${OUTPUT}" >&2

# Optional: judge without the ref first-frame image (prompt + frames only):
#   $CLI evaluate \
#     --manifest "${VIEW_DIR}/manifest.jsonl" --rubric_dir "${RUBRIC_DIR}" \
#     --provider "${PROVIDER}" --model "${MODEL}" \
#     --prompt_version "${PROMPT_VERSION}" --no_ref \
#     --output "${OUTPUT%.jsonl}_noref.jsonl"
