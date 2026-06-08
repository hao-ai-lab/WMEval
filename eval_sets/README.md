# Evaluation Sets

Curated, Unity-free benchmark bundles. Reference Unity videos and artifacts ship
with the repo so evaluators never need a local Unity checkout.

## WMEval Lite v1

Path: `eval_sets/wmeval_lite_v1/`

Reference videos and artifacts are **shared** across dimensions. Each dimension
applies different metrics to the same third-person scenes. MirrorBall is
**third-person only** (no first-person sample scene entries).

| Dimension | Test cases | Default modules |
| --- | --- | --- |
| `causal_lpips` | MirrorBall, BallRoller, BeamDomino | `lpips` (causal VLM collision) |
| `controllability` | MirrorBall, BallRoller, BeamDomino | `control_motion` |

### Layout

```
eval_sets/wmeval_lite_v1/
  manifest.json
  unity_references/
    mirror_ball_third_person_unity.mp4
    ball_roller_third_person_unity.mp4
    beam_domino_third_person_unity.mp4
  reference_artifacts/
    mirror_ball_third_person/
    ball_roller_third_person/
    beam_domino_third_person/
```

Each test case entry includes:

- `reference_video` — repo-relative Unity MP4 (shared across dimensions)
- `reference_video_sha256` — integrity check
- `reference_artifacts.actions_file` — Unity keyboard-action script (`.txt`) that drove the reference rollout
- `reference_artifacts.synthetic_action_dir` — `action_path/` with `action.npy` (required for controllability `control_motion`; not used by causal LPIPS)
- `reference_metrics` — optional precomputed Unity-side metrics (VBench-style)

### Quick start

Verify bundled references:

```bash
python tools/verify_wmeval_lite_refs.py
```

Dry-run one dimension:

```bash
python tools/evaluate_wmeval_lite.py \
  --dimensions causal_lpips \
  --candidate_jsonl path/to/candidates.jsonl \
  --dry_run
```

Run controllability eval:

```bash
python tools/evaluate_wmeval_lite.py \
  --dimensions controllability \
  --candidate_jsonl path/to/candidates.jsonl \
  --output_dir reports/wmeval_lite/controllability_run
```

Candidate JSONL rows should name the **dimension** and **test case**:

```json
{"dimension":"causal_lpips","test_case_id":"mirror_ball_third_person","world_model_name":"genie","run_id":"run_01","world_model_video":"/path/to/rollout.mp4"}
{"dimension":"controllability","test_case_id":"ball_roller_third_person","world_model_name":"genie","run_id":"run_01","world_model_video":"/path/to/rollout.mp4"}
```

Rows must always name a `dimension`. Within a dimension, a test case's `legacy_ids` (e.g. `sample_scene`, `mirror_ball`) may be used in place of its `test_case_id`.

Aggregate output is labeled **WMEval Lite w/ bundled reference** with per-dimension summaries.
