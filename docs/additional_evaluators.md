# Additional Evaluators

Optional evaluators outside the eight released dimensions.

## CLIP4Clip video–text similarity

Optional holistic video–text similarity baseline (`tools/legacy/evaluate_world_model_clip4clip.py`).

*(formerly `README_clip4clip_eval.md`)*

This document explains how to run CLIP4Clip evaluation in the current `WMEval` framework.

### 1) Scope

- Main entry: `tools/legacy/evaluate_world_model_clip4clip.py`
- Supports:
  - single video + prompt (`--video`, `--text_prompt`)
  - JSONL manifest batch mode (`--manifest_jsonl`, `--output_jsonl`)
- Supports sampling mode switch:
  - `timestamp`
  - `frame_step`

### 2) Environment

At minimum, CLIP4Clip requires:

- `torch`
- `opencv-python`
- `transformers`
- `torchvision`
- `Pillow`

If dependencies are missing, the tool returns an `error` field in output JSON/JSONL.

### 3) Quick Start (Manifest Mode)

Run using the provided script template:

```bash
python tools/legacy/evaluate_world_model_clip4clip.py \
  --manifest_jsonl /path/to/world_model_clip4clip_manifest.jsonl \
  --output_jsonl reports/world_model_clip4clip/results.jsonl \
  --sample_mode timestamp \
  --eval_fps 4 \
  --max_pairs 32
```

Equivalent script file:

- `scripts/evaluate_world_model_clip4clip.sh`

### 4) Manifest Format

The tool reads one JSON object per line.

#### Required per row

At least one of these video keys must exist:

- `world_model_video`
- `candidate_video`
- `reference_video`
- `video`

The tool will pick the first non-empty key in that order and normalize it as `video`.

#### Optional keys

- `sample_id` (or `id`)
- `prompt` (or `text_prompt`)

#### Minimal example

```json
{"sample_id":"s1","video":"/abs/path/video_1.mp4","prompt":"A ball rolling"}
{"sample_id":"s2","video":"/abs/path/video_2.mp4","prompt":"A car turning left"}
```


### 5) Single Video Mode

```bash
python tools/legacy/evaluate_world_model_clip4clip.py \
  --video /abs/path/video.mp4 \
  --text_prompt "A ball rolls down a ramp" \
  --output_json reports/world_model_clip4clip/single.json \
  --sample_mode timestamp \
  --eval_fps 4 \
  --max_pairs 32
```

### 6) Sampling Modes

- `timestamp`:
  - Time-based sampling with `eval_fps` as sampling FPS.
  - Similar behavior to LPIPS timestamp sampling.
- `frame_step`:
  - Index-step sampling from native video FPS.
  - `eval_fps` is treated as target FPS for step size.

### 8) Output Fields

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

### 9) Note on Framework Compatibility

- CLIP4Clip currently runs through the shared evaluation context (`reference` + `candidate`).
- In `tools/legacy/evaluate_world_model_clip4clip.py`, single-video use is implemented by assigning the same video to both internal slots.
- This keeps compatibility with the existing framework design and does not require changing LPIPS paths.


## DiffusionLight physics (lighting, legacy)

Earlier light-probe based lighting evaluator kept for reference; the released lighting dimension uses shadow consistency.

*(formerly `README_world_model_diffusionlight_physics_eval.md`)*

This README covers physical-consistency evaluation between a world-model video and
a reference Unity video using:

- `lighting`: DiffusionLight HDR envmap direction consistency
- `motion`: RAFT optical-flow consistency
- `shadow`: SAM2 shadow-mask tracking consistency

### Layout

- `evaluation/modules/lighting.py`: DiffusionLight-based lighting module
- `evaluation/modules/motion.py`: RAFT-based motion module
- `evaluation/modules/shadow.py`: SAM2-based shadow module
- `tools/legacy/evaluate_world_model_diffusionlight.py`: single-pair orchestrator
- `tools/legacy/evaluate_world_model_metrics.py`: standalone alias for the same entrypoint
- `scripts/evaluate_world_model_diffusionlight_physics.sh`: runnable command template

### Dependencies

```bash
pip install numpy opencv-python imageio scipy torch torchvision lpips
pip install huggingface_hub
pip install --no-build-isolation git+https://github.com/facebookresearch/sam2.git
```

You also need DiffusionLight at `/workspace/unity/DiffusionLight` (or pass a
custom path with `--diffusionlight_repo`).

### Usage

```bash
python tools/legacy/evaluate_world_model_diffusionlight.py \
  --video /path/to/world_model_video.mp4 \
  --ref_video /path/to/reference_video.mp4 \
  --modules lighting,motion,shadow \
  --unity_log_json /path/to/ground_truth_log.json \
  --diffusionlight_ref_hdr_dir /path/to/ref_diffusionlight/hdr \
  --diffusionlight_candidate_hdr_dir /path/to/cand_diffusionlight/hdr \
  --sam2_model facebook/sam2.1-hiera-small \
  --device cuda \
  --output_json reports/world_model_physics/report.json \
  --eval_fps 1 \
  --max_pairs 32
```

To run DiffusionLight inside the pipeline (instead of precomputed HDR dirs),
add `--run_diffusionlight` and optionally `--diffusionlight_output_root`.

### Metrics

- `dmc_angular_error_deg_*`: angle between candidate/reference main-light
  direction from DiffusionLight HDR envmaps (lower is better)
- `dmc_score_*`: normalized direction-match score in `[0,1]` (higher is better)
- `unity_json_angular_error_deg_*`: angle between candidate estimated direction
  and Unity log direction (requires coordinate calibration for absolute meaning)
- `epe_all_*`: mean endpoint error between candidate/reference RAFT flows
- `epe_dynamic_*`: EPE on high-motion pixels only
- `flow_angle_error_deg_*`: flow-direction disagreement angle
- `dynamic_motion_ratio_*`: fraction of dynamic pixels in reference flow
- `shadow_iou_*`: IoU between tracked candidate/reference shadow masks
- `shadow_centroid_error_px_*`: centroid displacement error in pixels

### Known Calibration Gap

High `unity_json_angular_error_deg_*` may still occur even when
reference==candidate because Unity light Euler/world convention and
DiffusionLight envmap camera-frame convention are not fully calibrated yet.

Unity exports needed for reviewer-proof alignment:

- camera intrinsics/extrinsics per frame
- ground-truth optical flow maps
- screen-space shadow masks
- instance masks with persistent IDs


## GroundingDINO tracking

Shared open-vocabulary detection, tracking and segmentation helper used by several dimensions.

*(formerly `README_groundingdino.md`)*

This folder contains copied GroundingDINO-related scripts from `VideoScience/judge/cv_tools`.
They are organized as:

- `tools/`: Python pipelines (`groundingdino_bytetrack.py`, `groundingdino_sam.py`)
- `scripts/`: runnable shell examples (`groundingdino_bytetrack*.sh`, `groundingdino_sam*.sh`)

### Requirements

- Python 3.10+
- PyTorch + CUDA (optional but recommended)
- OpenCV (`cv2`), NumPy, Pillow, torchvision
- GroundingDINO package/codebase available in Python path
- `cjm_byte_track` for ByteTrack script
- SAM 2 package (`sam2`) for segmentation script

### Quick usage

ByteTrack pipeline:

```bash
python tools/legacy/groundingdino_bytetrack.py \
  --video /path/to/video.mp4 \
  --text_prompt "ball . pendulum" \
  --grounding_cfg /path/to/GroundingDINO_SwinT_OGC.py \
  --grounding_ckpt /path/to/groundingdino_swint_ogc.pth \
  --output_dir mechanics_tracking_frames/run1 \
  --output_json mechanics_tracking_frames/run1/reports/tracking_report.json
```

SAM 2 pipeline:

```bash
python tools/legacy/groundingdino_sam.py \
  --video /path/to/video.mp4 \
  --text_prompt "ball . car" \
  --grounding_cfg /path/to/GroundingDINO_SwinT_OGC.py \
  --grounding_ckpt /path/to/groundingdino_swint_ogc.pth \
  --output_dir mechanics_tracking_frames/run1/sam \
  --output_json mechanics_tracking_frames/run1/reports/sam_report.json
```

Use the scripts in `scripts/` as ready-to-edit templates for concrete runs, for example:

```bash
bash scripts/groundingdino_bytetrack_wm_car.sh
bash scripts/groundingdino_sam_wm_car.sh
```
