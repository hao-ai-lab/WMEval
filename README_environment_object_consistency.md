# Environment Object and Character Consistency (paper appendix B.1)

This document describes the `environment_object_consistency` registry module,
which implements the NeurIPS paper's appendix-B.1 metrics: GroundingDINO +
SAM 2 localize and track a fixed set of key objects (and the main character)
through a candidate rollout; a monocular depth model and DINOv2 turn those
tracks into four headline metrics.

## Scope

This is a **reference-free-by-default** module: two of its four metrics
(`object_permanence_pct`, `character_consistency_pct`) only need the
candidate video. The other two (`depth_absrel_pct`,
`pairwise_distance_mae_pct`) additionally need Unity-side **analytical**
ground truth (per-object 3D world transforms + camera pose/intrinsics), per
the paper — not detector output re-run on Unity's own reference video.

As of this writing, no case bundle in `eval_sets/wmeval_lite_v1/` ships that
analytical ground-truth artifact (see `PENDING.md`), so
`depth_absrel_pct`/`pairwise_distance_mae_pct` currently report `null` with
`ground_truth_status: "pending"` until it lands.

## Main Entry Points

- `evaluation/modules/environment_object_consistency.py`: core metric implementation
- `evaluation/registry.py` → `"environment_object_consistency"`: registry entry
- `tools/evaluate_world_model_metrics.py --modules environment_object_consistency`: generic CLI

## Pipeline

1. GroundingDINO detects the configured key-object prompt
   (`eoc_text_prompt`, e.g. `"ball. chair. table."`) and character prompt
   (`eoc_character_prompt`, e.g. `"character. player."`) on frame 0.
2. SAM 2's video predictor tracks every detected box (objects + character)
   through the whole candidate video, giving per-frame masks/centroids.
3. GroundingDINO is **independently re-run on every frame** (not tied to the
   SAM 2 track IDs) to get a genuine per-frame detection confidence
   `s_{t,o}`, since SAM 2's video propagator does not expose one.
4. A monocular depth model (`eoc_depth_model`, default
   `depth-anything/Depth-Anything-V2-Small-hf` via `transformers`) samples
   depth at each tracked object's mask centroid; combined with the frame's
   pinhole intrinsics this back-projects centroids into candidate-side 3D
   points for pairwise distances.
5. DINOv2 (`torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`)
   embeds the cropped character region every frame it is tracked, and its
   cosine similarity to the frame-0 embedding gates Character Consistency.

## Metrics

| Metric | Formula | Ground truth needed |
| --- | --- | --- |
| `depth_absrel_pct` | `(1/\|O\|) * sum_o \|d_hat_o - d_o\| / (d_o + eps) * 100%` | Unity analytical (pending) |
| `pairwise_distance_mae_pct` | `(2/(\|O\|(\|O\|-1))) * sum_{i<j} \|delta_hat_ij - delta_ij\| / (delta_ij + eps) * 100%` | Unity analytical (pending) |
| `object_permanence_pct` | `(1/(T\|O\|)) * sum_t sum_o 1[s_{t,o} >= tau_det] * 100%` | none (candidate-only) |
| `character_consistency_pct` | `(1/T) * sum_t 1[s_t >= tau_det AND a_t >= tau_mask AND cos(phi_t, phi_1) >= tau_id] * 100%` | none (candidate-only) |

`d_o`/`delta_ij` (Unity depth and pairwise 3D distance) are computed
analytically from Unity's own known 3D object transforms and camera pose,
not by re-running detectors on Unity's reference video — see "Ground truth
artifacts" below.

Notes on two formula-vs-implementation choices, documented here rather than
silently baked in:

- The paper's Depth-AbsRel / Pairwise-Distance formulas sum only over
  objects, with no explicit sum over frames `t` (unlike Object Permanence /
  Character Consistency, which are explicit sums over `t`). This
  implementation time-averages `d_hat_o`/`delta_hat_ij` and
  `d_o`/`delta_ij` over the frames each side has data for, then applies the
  formula once on the averages. If the intent was per-frame AbsRel then
  averaged over time, flag it and this will change.
- `a_t` ("mask quality/stability score") is approximated as the tracked
  character mask's area ratio to its frame-0 area, clipped to `[0, 1]`, since
  SAM 2's video predictor does not directly expose a mask-quality score.

## Ground truth artifacts

Configure via `metric_config`/`module_params` (see `METRIC_CONFIG_KEYS` in
`evaluation/lite/adapters/registry_adapter.py`) or the matching
`run_world_model_evaluation(...)` kwargs:

- `eoc_unity_object_transforms_json`: JSON, schema:
  ```json
  {
    "objects": {
      "<label>": {
        "frames": [0, 1, 2],
        "positions_world": [[x, y, z], [x, y, z], [x, y, z]]
      }
    }
  }
  ```
  `<label>` must match (case-insensitively) the noun phrases GroundingDINO
  is asked to detect, so candidate detections and Unity ground truth pair up.
- `eoc_unity_poses_npy`: `(T, 4, 4)` camera-to-world matrices — same
  convention as the `poses.npy` artifact already bundled in e.g.
  `eval_sets/wmeval_lite_v1/cases/BallRoller/ctrl/reference/artifacts/poses.npy`.
- `eoc_unity_intrinsics_npy`: `(T, 4)` `[fx, fy, cx, cy]` per frame — same
  convention as the bundled `intrinsics.npy`.

Without `eoc_unity_object_transforms_json` and `eoc_unity_poses_npy`, the
module still reports `object_permanence_pct`/`character_consistency_pct`
and sets `ground_truth_status: "pending"` with `depth_absrel_pct` /
`pairwise_distance_mae_pct` as `null`.

## Useful Flags

- `--eoc_text_prompt "ball . rack . ramp ."` — GroundingDINO key-object prompt
- `--eoc_character_prompt "player character ."` — GroundingDINO character prompt
- `--eoc_tau_det 0.30` — detection-confidence threshold
- `--eoc_tau_mask 0.50` — mask-quality/stability threshold
- `--eoc_tau_id 0.50` — DINOv2 identity-cosine threshold
- `--eoc_max_objects 6` — cap on tracked key objects
- `--eoc_depth_model depth-anything/Depth-Anything-V2-Small-hf`

## Dependencies

`torch`, `transformers` (GroundingDINO zero-shot detection + depth
estimation), `sam2` (video predictor) — all already required by
`shadow_consistency`. DINOv2 is fetched at runtime via `torch.hub` (needs
network access to github.com on first run; no extra pip package).

## Interpretation Notes

- `object_permanence_pct`/`character_consistency_pct` measure candidate
  self-consistency; they say nothing about whether the candidate matches
  Unity's actual object/character behavior.
- Until `eoc_unity_object_transforms_json` lands for a case,
  `ground_truth_status` will read `"pending"` in every report for that case
  — this is expected, not a bug.
