# CLIP4Clip Evaluation README

This document explains how to run CLIP4Clip evaluation in the current `WMEval` framework.

## 1) Scope

- Main entry: `tools/evaluate_world_model_clip4clip.py`
- Supports:
  - single video + prompt (`--video`, `--text_prompt`)
  - JSONL manifest batch mode (`--manifest_jsonl`, `--output_jsonl`)
- Supports sampling mode switch:
  - `timestamp`
  - `frame_step`

## 2) Environment

At minimum, CLIP4Clip requires:

- `torch`
- `opencv-python`
- `transformers`
- `torchvision`
- `Pillow`

If dependencies are missing, the tool returns an `error` field in output JSON/JSONL.

## 3) Quick Start (Manifest Mode)

Run using the provided script template:

```bash
python tools/evaluate_world_model_clip4clip.py \
  --manifest_jsonl /path/to/world_model_clip4clip_manifest.jsonl \
  --output_jsonl reports/world_model_clip4clip/results.jsonl \
  --sample_mode timestamp \
  --eval_fps 4 \
  --max_pairs 32
```

Equivalent script file:

- `scripts/evaluate_world_model_clip4clip.sh`

## 4) Manifest Format

The tool reads one JSON object per line.

### Required per row

At least one of these video keys must exist:

- `world_model_video`
- `candidate_video`
- `reference_video`
- `video`

The tool will pick the first non-empty key in that order and normalize it as `video`.

### Optional keys

- `sample_id` (or `id`)
- `prompt` (or `text_prompt`)

### Minimal example

```json
{"sample_id":"s1","video":"/abs/path/video_1.mp4","prompt":"A ball rolling"}
{"sample_id":"s2","video":"/abs/path/video_2.mp4","prompt":"A car turning left"}
```


## 5) Single Video Mode

```bash
python tools/evaluate_world_model_clip4clip.py \
  --video /abs/path/video.mp4 \
  --text_prompt "A ball rolls down a ramp" \
  --output_json reports/world_model_clip4clip/single.json \
  --sample_mode timestamp \
  --eval_fps 4 \
  --max_pairs 32
```

## 6) Sampling Modes

- `timestamp`:
  - Time-based sampling with `eval_fps` as sampling FPS.
  - Similar behavior to LPIPS timestamp sampling.
- `frame_step`:
  - Index-step sampling from native video FPS.
  - `eval_fps` is treated as target FPS for step size.

## 8) Output Fields

Each result row includes (typical):

- `module` (`clip4clip`)
- `sample_id`
- `video`
- `prompt`
- `sample_mode`
- `similarity_cosine`
- `num_frames_used`
- `model_id`
- `device`

Some large metadata keys are intentionally dropped in manifest output JSONL for compactness.

## 9) Note on Framework Compatibility

- CLIP4Clip currently runs through the shared evaluation context (`reference` + `candidate`).
- In `tools/evaluate_world_model_clip4clip.py`, single-video use is implemented by assigning the same video to both internal slots.
- This keeps compatibility with the existing framework design and does not require changing LPIPS paths.
