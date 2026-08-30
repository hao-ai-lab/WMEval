# Kinematics and Mechanics (paper appendix B.3)

This document describes the `kinematics_mechanics` registry module, which
reconstructs key-object trajectories from an action-conditioned candidate
rollout with GroundingDINO + ByteTrack, and compares them against Unity-side
references (trajectory fidelity, momentum, kinetic energy) per the NeurIPS
paper's appendix B.3.

## Scope

Like `environment_object_consistency` (B.1), this module is candidate-only
by default: it always reports the reconstructed candidate trajectories
(`trajectories`), useful for QA even without ground truth. Ground-truth
comparison (`trajectory_fidelity`, `mechanical_quantities`) additionally
needs Unity-side artifacts described below.

**Honesty note on what "trajectory fidelity" currently means here:** the
paper's B.3 FDE/DTW are 3D metrics. This module's candidate-side pipeline is
monocular (GroundingDINO + ByteTrack on 2D pixel boxes); without a
per-frame *candidate* camera-pose artifact there is no way to lift that into
a metric 3D trajectory. So when ground truth is available,
`trajectory_fidelity` reports a **2D pixel-space reprojection proxy**:
the candidate's raw pixel trajectory vs. Unity's 3D ground-truth trajectory
reprojected through Unity's *own* ground-truth camera (`status:
"reprojection_proxy_2d"`), not the paper's 3D FDE/DTW. Likewise
`mechanical_quantities` can report mass and a pixel/frame terminal speed
(`status: "pixel_units_only"`) but not momentum/kinetic energy in physical
units, since that also needs metric-scale candidate velocity. See
`eval_sets/wmeval_lite_v1/PENDING.md` for what would close this gap
(candidate-side pose recovery or a metric-depth model).

## Main Entry Points

- `evaluation/modules/kinematics_mechanics.py`: core metric implementation
- `evaluation/registry.py` → `"kinematics_mechanics"`: registry entry
- `tools/evaluate_world_model_metrics.py --modules kinematics_mechanics`: generic CLI
- `tools/groundingdino_bytetrack.py`: the standalone GroundingDINO+ByteTrack
  visualization script this module's detection/tracking step mirrors
  (see `README_groundingdino.md`)

## Pipeline

1. GroundingDINO detects the configured key-object prompt (`kin_text_prompt`)
   independently on every frame of the candidate video.
2. A ByteTrack tracker (`cjm_byte_track.core.BYTETracker`, falling back to
   `bytetracker.BYTETracker`) associates per-frame detections into tracks;
   short tracks (`kin_min_track_len`) are dropped, and each surviving track
   is assigned the majority-vote label across its detections.
3. Objects listed in `kin_static_anchor_labels` (comma-separated) are
   treated as static scene anchors: their mean per-frame centroid shift
   (relative to frame 0) is used as a camera-ego-motion estimate, subtracted
   from every tracked centroid to produce `centroid_uv_compensated` — an
   approximate common static reference frame, in the spirit of the paper's
   "track static anchors to estimate moving reference-frame motion."
4. When ground truth is configured, Unity's 3D trajectory for a matching
   label is reprojected through Unity's own camera
   (`kin_unity_poses_npy`/`kin_unity_intrinsics_npy`) and compared against
   the candidate's **raw** (uncompensated) pixel trajectory with FDE and DTW
   (see the honesty note above for why raw, not compensated, is used here).

## Metrics

| Metric | Formula | Status in this module |
| --- | --- | --- |
| FDE | `\|\| x_hat_T - x_T \|\|_2` | 2D pixel-space proxy (`trajectory_fidelity[label].fde_px`) |
| DTW | `min_pi sum_(i,j) in pi \|\| x_hat_i - x_j \|\|_2` | 2D pixel-space proxy (`trajectory_fidelity[label].dtw_px`) |
| Momentum `p_hat_t = m * v_hat_t` | -- | pending metric-scale velocity |
| Kinetic energy `E_hat_t = 0.5 * m * \|\|v_hat_t\|\|^2` | -- | pending metric-scale velocity |
| ExactMatch / ErrorRate over `p`/`E` | `round2(q_hat_o) == round2(q_o)` / `\|q_hat_o - q_o\| / (\|q_o\| + eps)` | not yet computed (needs the above) |

## Ground truth artifacts

- `kin_unity_object_transforms_json`: same schema as
  `eoc_unity_object_transforms_json` (see
  `README_environment_object_consistency.md`) — per-frame per-object 3D
  world positions.
- `kin_unity_poses_npy` / `kin_unity_intrinsics_npy`: `(T, 4, 4)` /
  `(T, 4)`, same convention as the bundled `poses.npy`/`intrinsics.npy`
  (e.g. `eval_sets/wmeval_lite_v1/cases/BallRoller/ctrl/reference/artifacts/`).
- `kin_unity_object_masses_json`: flat `{"<label>": mass_kg}` dict.

Without `kin_unity_object_transforms_json` + poses + intrinsics, the module
reports `ground_truth_status: "pending"` and leaves `trajectory_fidelity` /
`mechanical_quantities` as `null`, still returning the reconstructed
candidate `trajectories` for inspection.

## Useful Flags

- `--kin_text_prompt "ball . car . pin ."` — GroundingDINO key-object prompt
- `--kin_static_anchor_labels "wall,floor"` — comma-separated static-anchor labels for camera-motion estimation
- `--kin_track_thresh 0.5 --kin_track_buffer 30 --kin_match_thresh 0.8` — ByteTrack knobs
- `--kin_min_track_len 4` — minimum detections for a track to count as a trajectory

## Dependencies

`torch`, `transformers` (GroundingDINO zero-shot detection, same HF path as
`shadow_consistency`), and a ByteTrack implementation:

```bash
pip install cjm-byte-track   # matches tools/groundingdino_bytetrack.py
# or, if unavailable:
pip install bytetracker
```

## Interpretation Notes

- `camera_compensation_applied` is `false` whenever `kin_static_anchor_labels`
  is empty or no anchor was detected in any frame — check this before
  trusting `centroid_uv_compensated`.
- Treat `trajectory_fidelity`/`mechanical_quantities` `status` fields as
  load-bearing: `"reprojection_proxy_2d"` and `"pixel_units_only"` are not
  the paper's final metrics, only diagnostics until candidate-side pose
  recovery exists.
