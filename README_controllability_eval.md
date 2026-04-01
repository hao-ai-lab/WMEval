# Controllability Evaluation (RAFT + optional synthetic flow)

This document describes the RAFT controllability path in `WMEval`, plus an optional
**synthetic reference flow** branch compatible with the synthetic mode in
[ptlflow `eval_flow_divergence.py`](https://github.com/shaoxiongduan/ptlflow/blob/main/eval_flow_divergence.py).

## Scope

This v1 implementation is intentionally video-only on the Unity side:

- `--ref_video`: Unity reference video
- `--video`: world-model / generated video

The module estimates optical flow from both videos with RAFT, then compares the
resulting motion fields. It does **not** require `GroundingDINO`, `ByteTrack`,
or Unity-exported dense flow artifacts in v1.

## Main Entry Points

- `tools/evaluate_world_model_controllability.py`: dedicated controllability CLI
- `tools/evaluate_world_model_metrics.py --modules control_motion`: generic CLI
- `evaluation/modules/control_motion.py`: core metric implementation

## What The Metric Measures

For each aligned frame pair:

1. Estimate RAFT flow on the Unity reference video.
2. Estimate RAFT flow on the world-model video.
3. Compare the two motion fields.

This is still an estimated-vs-estimated comparison, not true Unity dense-flow
ground truth. It is therefore a controllability-oriented proxy rather than a
final benchmark definition.

## Reported Metrics

The `control_motion` module reports:

- raw frame-space comparison
- optional camera-compensated comparison
- mean / median / p90 / max EPE
- dynamic-region EPE
- dynamic-region angular error
- dynamic-region cosine similarity
- dynamic-region relative EPE
- adaptive dynamic-motion thresholds used for masking
- bad-pixel rates at configurable EPE thresholds
- per-pair camera-motion models for the reference and candidate videos

### Endpoint Error (EPE)

If the reference flow vector at a pixel is `(u_ref, v_ref)` and the candidate
flow vector is `(u_cand, v_cand)`, then:

```text
EPE = sqrt((u_cand - u_ref)^2 + (v_cand - v_ref)^2)
```

Lower EPE means the candidate motion is closer to the Unity reference motion.

### Bad-Pixel Rate

For a chosen threshold `t`, this is the fraction of pixels whose EPE is larger
than `t`. The default thresholds are `1.0` and `3.0` pixels.

### Dynamic Mask

Dynamic-region metrics use a mask derived from the reference-flow magnitude:

- default threshold: `0.5` pixels
- default adaptive percentile: `75`
- the effective threshold is `max(abs_threshold, percentile_threshold)`
- pixels below that threshold are treated as effectively static

## Camera Compensation

Camera drift can dominate dense flow even when the target object behavior is
reasonable. To expose that effect, `control_motion` can also compute a
camera-compensated comparison.

Current v1 strategy:

- estimate camera motion per frame pair from each flow field
- default estimator: `median_translation`
- optional `affine_ransac` mode is available for experimentation
- fall back to translation when affine fitting fails
- subtract the estimated camera-motion field from the corresponding flow field
- recompute EPE / angular error / cosine similarity / bad-pixel rates on the residual flow

This is a lightweight approximation of camera-motion compensation. It is useful
for inspection, but it is not a substitute for object-centric or true
world-frame trajectory analysis.

## Quick Start

```bash
python tools/evaluate_world_model_controllability.py \
  --ref_video /abs/path/unity.mp4 \
  --video /abs/path/world_model.mp4 \
  --output_json reports/world_model_controllability/report.json \
  --eval_fps 4 \
  --max_pairs 32
```

Equivalent generic runner usage:

```bash
python tools/evaluate_world_model_metrics.py \
  --ref_video /abs/path/unity.mp4 \
  --video /abs/path/world_model.mp4 \
  --modules control_motion \
  --output_json reports/world_model_controllability/report.json
```

## Useful Flags

- `--control_global_motion_mode affine_ransac|median_translation|mean_translation`
- `--control_motion_mask_threshold 0.5`
- `--control_motion_mask_percentile 75`
- `--control_bad_pixel_thresholds 1.0,3.0`
- `--disable_control_camera_compensation`

## Interpretation Notes

- Lower `raw.epe_summary.mean` means the world-model motion is closer to the
  Unity reference motion in image space.
- Lower `camera_compensated.epe_summary.mean` suggests the mismatch was partly
  due to global camera drift rather than object-relative motion.
- Higher `camera_compensated.cosine_similarity_dynamic_summary.mean` means the
  candidate motion direction is more aligned with the reference after removing
  estimated camera motion.
- Large gaps between raw and compensated scores are a signal to inspect camera
  motion and framing before drawing strong conclusions about controllability.

## Synthetic flow vs RAFT (action-conditioned proxy)

When you pass a calibration file and either an action recording or **inferred**
actions, `control_motion` additionally reports `synthetic_vs_candidate`:

- **Reference motion**: Longuet-Higgins field from `synthetic_flow.py` (loaded
  from your `ptlflow` checkout: default `--ptlflow_root /workspace/ptlflow`).
- **Candidate motion**: **same RAFT model** as the main block (consistent with
  the RAFT ref vs RAFT gen comparison).
- **Per-pair / temporal metrics**: same families as ptlflow (`mf_epe`, `pixel_epe_mean`,
  `flow_kl_2d`, FOE distance, `fl_all`, divergence onset, etc.) via
  `evaluation/modules/flow_divergence_metrics.py`.

### CLI flags

- `--synthetic_calibration /path/to/calibration.json` — required to enable the block
- `--synthetic_action_file /path/to/actions.npy` — dict with `keyboard` `(T,6)` and
  `mouse` `(T,2)`; rows indexed by sampled candidate frame indices
- `--synthetic_infer_actions` — if no npy is available, fit continuous
  keyboard+mouse so synthetic flow matches **RAFT on the Unity reference** for
  each sampled pair (approximate; slower).
- `--synthetic_use_depth` — translation term uses Depth Anything (optional;
  needs `transformers` + download).
- `--synthetic_grid_size` — grid EPE binning (default 8).

Example:

```bash
python tools/evaluate_world_model_controllability.py \
  --ref_video /abs/unity.mp4 \
  --video /abs/world_model.mp4 \
  --synthetic_calibration /abs/ptlflow/calibration.json \
  --synthetic_infer_actions \
  --output_json reports/world_model_controllability/report_synth.json
```

## Future Extensions

Likely follow-ups after this v1:

- Unity-exported dense flow ground truth
- explicit camera trajectory artifacts
- object-centric controllability regions
- trajectory matching in a stabilized reference frame
