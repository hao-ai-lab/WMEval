# World Model Physics Evaluation

This README covers physical-consistency evaluation between a world-model video and
a reference Unity video using:

- `lighting`: DiffusionLight HDR envmap direction consistency
- `motion`: RAFT optical-flow consistency
- `shadow`: SAM2 shadow-mask tracking consistency

## Layout

- `evaluation/modules/lighting.py`: DiffusionLight-based lighting module
- `evaluation/modules/motion.py`: RAFT-based motion module
- `evaluation/modules/shadow.py`: SAM2-based shadow module
- `tools/evaluate_world_model_metrics.py`: single-pair orchestrator
- `tools/evaluate_world_model_diffusionlight.py`: compatibility wrapper for older commands
- `scripts/evaluate_world_model_diffusionlight_physics.sh`: runnable command template

## Dependencies

```bash
pip install numpy opencv-python imageio scipy torch torchvision lpips
pip install huggingface_hub
pip install --no-build-isolation git+https://github.com/facebookresearch/sam2.git
```

You also need DiffusionLight at `/workspace/unity/DiffusionLight` (or pass a
custom path with `--diffusionlight_repo`).

## Usage

```bash
python tools/evaluate_world_model_metrics.py \
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

## Metrics

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

## Known Calibration Gap

High `unity_json_angular_error_deg_*` may still occur even when
reference==candidate because Unity light Euler/world convention and
DiffusionLight envmap camera-frame convention are not fully calibrated yet.

Unity exports needed for reviewer-proof alignment:

- camera intrinsics/extrinsics per frame
- ground-truth optical flow maps
- screen-space shadow masks
- instance masks with persistent IDs
