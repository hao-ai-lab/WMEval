<p align="center">
<img src="assets/img/scenes.jpg" alt="WMEval Unity scenes" width="100%" align="center">
</p>

<div align="center"> <h1>WMEval: Evaluating Interactive World Models Grounded on Physical Simulations</h1> </div>
<p align="center"> <b>📜 Paper</b> (coming soon) | <a href="https://snyhlx.github.io/wmeval-blog/"><b>🌐 Project Page</b></a> | <a href="https://huggingface.co/datasets/lmgame/WMEval-Lite"><b>🤗 Dataset</b></a></p>

## Introduction

WMEval evaluates interactive world models against Unity-based physical simulations. Instead of judging generations on perceptual quality alone, it executes the **same action trajectory** in a Unity scene and in the world model, then compares the two rollouts using simulator-derived ground truth — camera poses, object states, masses, momentum and energy, causal event traces — as the reference. It consists of:

1. **9 Unity scenes** (`BallRoller` · `BeamDomino` · `BilliardGame` · `ChessBoard` · `LiquidContainer` · `MirrorRollingBall` · `RacingCar` · `SunRisesAndSets` · `WanderingGarden`), each paired with deliberately designed action trajectories.
2. **A Unity-free evaluation suite** (WMEval Lite): checksummed case bundles carrying the conditioning frame, prompt, action trajectory, reference rollout and the Unity quantities each metric needs, plus one CLI that validates candidates, dispatches every dimension's evaluator and aggregates over seeds and cases. The paper's Table 4 and Table 7 cells are Lite cases too, so every dimension is evaluated through the same path.
3. **Standardized action trajectories** that compile one semantic trajectory onto each backend's native interface (Unity, Project Genie, Matrix-Game 2.0/3.0, LingBot-World), with explicit action targets, calibration and unsupported-control reporting.

<p align="center">
  <picture>
  <img src="assets/img/overview.jpg" width="90%">
  </picture>
</p>

## Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [WMEval Lite Suite](#wmeval-lite-suite)
  - [Case Bundles](#case-bundles)
  - [Outputs](#outputs)
- [Action Trajectories](#action-trajectories)
  - [Synthetic Trajectory Suite (test_full)](#synthetic-trajectory-suite-test_full)
- [Evaluation Dimensions](#evaluation-dimensions)
- [Additional Evaluators](#additional-evaluators)
- [Citation](#citation)

## Installation

1. Clone this repository:
```
git clone https://github.com/hao-ai-lab/WMEval.git
cd WMEval
```
2. Install dependencies with [`uv`](https://docs.astral.sh/uv/) (Modal jobs pin their own image in `tools/run_wmeval_lite_modal.py`).

   macOS (Apple silicon):
```
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-macos.lock
source .venv/bin/activate
```
   Linux with an NVIDIA GPU (the script creates `.venv-gpu` and checks the environment at the end):
```
bash scripts/bootstrap_tables_4_7_gpu.sh
source .venv-gpu/bin/activate
```
3. Download the case data and validate the suite. The repository holds each case's metadata (`case.yaml` and the checksum manifest); the case data itself (reference videos, conditioning frames, action trajectories and other artifacts) is hosted on the Hugging Face Hub at [`lmgame/WMEval-Lite`](https://huggingface.co/datasets/lmgame/WMEval-Lite):
```
python tools/sync_lite_data.py download        # writes cases/*/reference/ and verifies every checksum
python tools/run_wmeval_lite.py validate --suite eval_sets/wmeval_lite_v1
```
Add `--split full` to also fetch the 220 synthetic action trajectories ([below](#synthetic-trajectory-suite-test_full)). The download fetches the dataset commit this checkout was published with (`eval_sets/wmeval_lite_v1/DATASET_REVISION`), so a newer dataset version never breaks an older branch; `--revision <tag-or-commit>` overrides it. No token is needed (the dataset is public). `python tools/sync_lite_data.py status` reports which cases are present.

VLM-judged dimensions (`prompt_scene_consistency`, `physical_causality`, causal `lpips`) need an `OPENAI_API_KEY`; CV-based dimensions download pinned checkpoints on first use (see `experiments/tables_4_7/evaluator_assets.lock.json`).

## Quick Start

Generate one video per case and seed from the case's conditioning frame, prompt and action script, place them at `<model_root>/<Scene>/<variant>/<seed>.mp4`, then run:

```
# What does the suite contain?
python tools/run_wmeval_lite.py list-cases --suite eval_sets/wmeval_lite_v1

# Evaluate every wired dimension (resumable; --dimensions selects a subset)
python tools/run_wmeval_lite.py run \
  --suite eval_sets/wmeval_lite_v1 \
  --model-root runs/<model> \
  --output-dir reports/wmeval_lite/<model> --resume
```

Instead of a directory tree you can pass a strict JSONL manifest with `--candidate-manifest`:

```json
{"model":"my_model","case_id":"BallRoller/ctrl","seed":"42","candidate_video":"/data/my_model/BallRoller/ctrl/42.mp4"}
{"model":"my_model","case_id":"RacingCar/prompt_first","seed":"42","candidate_video":"/data/other.mp4","params":{"shadow_hud_preset":"genie"}}
```

To compare the Table 4/7 cells with the paper baseline, export the rows and run the comparison:

```
python tools/export_lite_table_results.py --model "<name>" \
  --results reports/wmeval_lite/<model>/per_case_results.jsonl \
  --output reports/wmeval_lite/<model>/table_results.json
python tools/compare_table_results.py compare --allow-incomplete \
  --results reports/wmeval_lite/<model>/table_results.json
```

On Modal GPUs, `modal run tools/run_wmeval_lite_modal.py::main --model-root /cand/<model>` runs one container per candidate; `tools/generate_waypoint15_candidates_modal.py` is a worked example of generating a model's candidates straight from the bundles without Unity.

## WMEval Lite Suite

`eval_sets/wmeval_lite_v1/` is one suite with one CLI. `suite.yaml` declares the dimensions and is the single source of truth for case wiring; each case ships as a checksummed bundle under `cases/<Scene>/<variant>/`. The current release roster is **32 cases across 9 scenes, 8 dimensions and 43 case–dimension assignments**, including the 14 Table 4/7 cells (`<Scene>/scene_<variant>` for Table 7, `<Scene>/kinematics_<view>` for Table 4).

The eight dimensions and their headline metrics are summarized under [Evaluation Dimensions](#evaluation-dimensions).

### Case Bundles

```
eval_sets/wmeval_lite_v1/
  suite.yaml                       # dimension-level manifest (single source of truth)
  cases/<Scene>/<variant>/
    case.yaml                      # prompt, modules, protocol, evaluator_lock      (in this repo)
    reference/                     # case data: downloaded from lmgame/WMEval-Lite, except checksums.sha256
      reference.mp4                # reference video (reference-based dimensions)
      ref.png                      # conditioning frame where it differs from video frame 0
      checksums.sha256             # covers every file under reference/ (feeds the evaluator fingerprint)
      artifacts/
        actions.txt                # action script
        action.npy                 # numeric actions (+ intrinsics.npy / poses.npy / object_transforms.json)
        rubrics/<Scene>/<provider>_<model>_<version>.json   # committed judge rubrics
        qa_dataset.json / *.npz    # dimension-specific payloads (causal QA, liquid masks/flow)
```

Author or refresh a case with `python tools/build_lite_case.py <case_dir>` (writes checksums and the evaluator lock), then publish the data with `python tools/sync_lite_data.py upload` (maintainers); generate prompt rubrics with `python tools/run_wmeval_lite.py generate-rubrics` (requires `OPENAI_API_KEY`). Legacy candidate ids are mapped through `legacy_case_aliases` in `suite.yaml`; convert old files with `tools/convert_legacy_candidates.py`.

### Outputs

`--output-dir` receives `per_case_results.jsonl` (one row per dimension × case × model × seed with status, flat metrics, the module's full report and the case's evaluator lock), `aggregate_summary.json` (two-level means — seeds within a case, then across cases — with `std_across_cases`, case means and per-case seed dispersion) and `aggregate_summary.csv` (headline mean and `_std` per dimension).

## Action Trajectories

Every case ships its action script inside its bundle:

- **Table 4/7 cases** (`*/scene_*`, `*/kinematics_*`) carry the canonical semantic trajectory (`reference/artifacts/trajectory.yaml`, `schema_version: 1`: typed channels, units, time segments, control rate) and its compiled forms (`action.npz` / `action.npy`, the Project Genie browser program `actions.jsonl`, and a readable `actions.txt`). `python tools/build_table_lite_cases.py` rebuilds these bundles from the hash-locked trajectories in `experiments/tables_4_7/trajectories/` and the Unity captures.
- **Other cases** carry `actions.txt` (browser key programs or structured text) and, for most, a numeric `action.npy`; `python tools/audit_trajectory_readiness.py` reports which cases are native, supplemental or underspecified.
- **Backends** are declared once in `experiments/tables_4_7/action_backends.yaml` (read via `evaluation/action_backends.py`): what entity each trajectory actuates (`camera`, `ego_vehicle`, `cue_ball`, `table_objects`), each backend's native timeline, which targets it can express, why it cannot express the others, and the frozen response-side camera calibration. Input preparation refuses unsupported target/backend pairs instead of substituting another control.

| Backend | Native input | Camera trajectories | Vehicle / ball trajectories |
| --- | --- | --- | --- |
| Unity | runtime arguments | reference | reference |
| Project Genie | browser keys | nominal (left turn 51.47 °/s calibrated) | nominal key rules |
| Matrix-Game 2.0 | 129 frames @ 12 fps, keys + mouse | pooled 10.1 °/mouse unit/frame | thresholded WASD (target validation required) |
| Matrix-Game 3.0 | 177 frames @ 17 fps, keys + mouse | regime/direction-specific gains | out of scope |
| LingBot-World Base Cam | 161 camera poses @ 16 fps | nominal poses, no response correction | unsupported (camera input only) |

```
python tools/prepare_table_trajectories.py        # validate + compile semantic trajectories
python tools/prepare_matrix_game_inputs.py        # Matrix-Game 2.0 / 3.0 schedules + queues
python tools/prepare_lingbot_table7_inputs.py     # LingBot Base Cam poses/intrinsics + queue
```

### Synthetic Trajectory Suite (test_full)

The curated cases form the `test_core` split (33 rows). The `test_full` split adds **220 synthetic action trajectories** (253 rows in total), built text first: each starts as an action description (a short sequence of steps from a per-target vocabulary, e.g. *"the camera turns left at 30 degrees per second for 2 seconds (60 degrees in total), then holds still until the end of the 10-second clip"*), which becomes the second half of its prompt and is compiled into the action sequence (a schema-v1 `trajectory.yaml`). The first half of the prompt is the scene description of a core case, which also supplies the conditioning frame and action target; steps start from that case's initial state, so the core conditioning frame stays valid. Steps are sampled with a fixed seed, and each family keeps only distinct, mutually far-apart action sequences, balanced across step types and step counts.

| Family | Target | Trajectories | Step vocabulary (core case) |
| --- | --- | --- | --- |
| `camera_yaw` | camera | 45 | turn, hold, sweep and back, look behind (5 on each of 9 scene views) |
| `camera_orbit` | camera | 20 | circle clockwise / counter-clockwise, pause (`SunRisesAndSets/lighting_orbit_v2`) |
| `ego_vehicle` | ego_vehicle | 28 | drive, steer, lane change, brake, coast (`RacingCar/kinematics_third`) |
| `cue_ball` | cue_ball | 25 | single or repeated strikes in 8 directions at 3 strengths, waits (`BilliardGame/kinematics_first`) |
| `ball_force` | ball | 28 | pushes in 8 directions at 2 strengths, free rolling (14 each on `BallRoller`, `MirrorRollingBall`) |
| `beam_domino` | rolling_ball | 25 | rolls forward / angled / sideways / back, waits (`BeamDomino/causal`) |
| `liquid_container` | liquid_container | 25 | jolts, slides, fixed-rate shakes, rising-rate shakes, rests (`LiquidContainer/fluid_rest`) |
| `character` | character | 24 | walk in 8 directions, sprint, jump, stand (`ChessBoard/walk_third`) |

```
python tools/sync_lite_data.py download --split full     # cases + synthetic/, checksum-verified
python tools/expected_jobs.py --model "<backend>" --split full --output runs/<model>
python tools/generate_synthetic_trajectories.py --verify  # rebuild locally and check against synthetic/manifest.json
```

Unity references (maintainers; needs the Unity editor and WorldModelBenchScenePrivate with its LFS assets):

```
python tools/render_synthetic_references.py build             # replay players (variant scene revisions included)
python tools/render_synthetic_references.py run --detach      # render every trajectory, resumable, in the background
python tools/render_synthetic_references.py publish           # copy into synthetic/*/reference/ + synthetic/references.json
python tools/generate_synthetic_trajectories.py               # fold reference status into meta.json / manifest
```

Each trajectory directory `synthetic/<family>/<Scene>__<variant>/<pattern>/` holds `trajectory.yaml`, the compiled `action.npz` (plus `poses.npy` for orbits), the Project Genie program `actions.jsonl` and `meta.json` (prompt, action description and steps, first frame, target dimensions). `expected_jobs.py` applies the same action-target gate as the core cases and writes candidates to `<model_root>/synthetic/<family>/<Scene>__<variant>/<pattern>/<seed>.mp4`. Each trajectory also has a Unity reference in its `reference/` folder (video, frame 0, Unity ground truth, capture record), rendered by replaying the same actions in the core case's Unity scene; `synthetic/references.json` lists every reference file's checksum, how closely its frame 0 matches the core conditioning frame, and which scene revision and assets it was rendered from. References whose frame 0 differs visibly from the core frame (below 20 dB PSNR) are marked `reference_quality: weak_first_frame`. Scoring synthetic rows is not wired into `run_wmeval_lite.py` yet: it scores `test_core` only.

## Evaluation Dimensions

Each dimension compares what is recovered from the candidate video with quantities logged by the Unity engine twin. Run any subset with `--dimensions`. The pipeline, flags, outputs and interpretation notes of every evaluator are in [docs/evaluation_dimensions.md](docs/evaluation_dimensions.md); *Needs* says whether a dimension uses pinned CV checkpoints (GPU recommended) or a VLM judge (`OPENAI_API_KEY`).

| Dimension | Question | Headline metrics | Cases | Needs |
| --- | --- | --- | --- | --- |
| [Environment Object & Character Consistency](docs/evaluation_dimensions.md#environment-object--character-consistency) | Do objects, spatial relations and character identity persist as the camera moves away and returns? | Depth AbsRel ↓, pairwise-distance MAE ↓ (against Unity-analytic geometry), object permanence ↑, DINOv2 character consistency ↑ | 16 | CV models |
| [Action Controllability](docs/evaluation_dimensions.md#action-controllability) | Does generated motion follow the commanded action or camera control? | Ground-Truth RAFT (relative EPE ↓, cosine ↑ vs. Unity flow); synthetic-flow cosine ↑ / pixel EPE ↓ vs. commanded motion | 3 | CV models |
| [Kinematics & Mechanics](docs/evaluation_dimensions.md#kinematics--mechanics) | Do matched controls produce simulator-consistent trajectories and mechanical outcomes? | DTW ↓, FDE ↓ (trajectories); momentum / kinetic-energy Error Rate ↓ and Exact Match ↑ | 3 | CV models |
| [Fluid Mechanics](docs/evaluation_dimensions.md#fluid-mechanics) | Does the liquid in a sealed, moving bottle slosh like the reference? | flow cosine (diagnostic), area jitter Δσ_A ↓ / drift ↓, surface angle ↓, slosh-spectrum L1 ↓, LPIPS ↓ | 4 | CV models |
| [Physical Causality](docs/evaluation_dimensions.md#physical-causality) | Does an action propagate through a multi-stage causal chain (rolling → impact → topple → cascade)? | Causal Success Rate ↑, AMCL/MCL, weighted score, DUA, ICRA — reported as deltas from the Unity reference judged the same way | 3 | VLM judge |
| [Lighting & Shading](docs/evaluation_dimensions.md#lighting--shading) | Do cast shadows stay geometrically and temporally coherent with one light source as the camera moves? | Scene Retention ↑, Shadow Detection ↑, Multi-Object Sun Disagreement ↓, Shadow Direction Jitter ↓, Shadow Area Stability ↓, Shadow Shape Drift ↓ | 1 | CV models |
| [Prompt-Scene Consistency](docs/evaluation_dimensions.md#prompt-scene-consistency) | Does the rollout match its scene description, for both environment and foreground objects? | Environment ↑, Object ↑, Overall ↑ rubric scores (ρ = 0.90 with human ratings) | 6 | VLM judge |
| [Perceptual Similarity (LPIPS)](docs/evaluation_dimensions.md#perceptual-similarity-lpips) | How perceptually close is the rollout to the reference around matched contact events? | Causal LPIPS ↓ (offsets −2…+2 around chronologically matched events), temporal LPIPS ↓ as a separate fallback | 7 | CV + VLM judge |

Rows report `metric_status`, `action_support` and `input_contract`; a dimension headline needs `repeat_policy.required_repeats` (5) seeds per applicable case. Unity self-match error floors per scene are in [`SELF_MATCH_BOUNDS.md`](eval_sets/wmeval_lite_v1/SELF_MATCH_BOUNDS.md), and differences from the paper in [`DIFFERENCES_FROM_PAPER.md`](eval_sets/wmeval_lite_v1/DIFFERENCES_FROM_PAPER.md).

### Release tooling

| Task | Command / file |
| --- | --- |
| New model: jobs and N/A cases | `python tools/expected_jobs.py --model <name> --output runs/<name>` · `tools/model_adapters/TEMPLATE.md` |
| Check the environment before a run | `python tools/preflight_lite.py --device <cuda\|mps\|cpu>` |
| No-cost candidates for tests | `python tools/fixture_backend.py --output-root <dir>` |
| Hosted judge through a gateway | `python tools/judge_gateway.py --key-file <key> [--script <tool>] -- <args>` |
| Export / verify the HF package | `python tools/export_hf_package.py export --output <dir>` · `... verify <dir>` |
| Case dependency locks | `python tools/lock_case_dependencies.py [--check]` |
| Differences from the paper | `eval_sets/wmeval_lite_v1/DIFFERENCES_FROM_PAPER.md` |
| Frozen judge outputs, re-baselining | `eval_sets/wmeval_lite_v1/FROZEN_JUDGE_OUTPUTS.md` |

Rows carry `input_contract`, `metric_status` and `action_support`; a dimension
headline needs `repeat_policy.required_repeats` (5) seeds per applicable case,
otherwise the aggregate keeps it as `provisional_value`.

## Additional Evaluators

### CLIP4Clip video–text similarity

Optional holistic video–text similarity baseline (`tools/evaluate_world_model_clip4clip.py`).

<details>
<summary><b>Full reference</b> (formerly <code>README_clip4clip_eval.md</code>)</summary>

This document explains how to run CLIP4Clip evaluation in the current `WMEval` framework.

##### 1) Scope

- Main entry: `tools/evaluate_world_model_clip4clip.py`
- Supports:
  - single video + prompt (`--video`, `--text_prompt`)
  - JSONL manifest batch mode (`--manifest_jsonl`, `--output_jsonl`)
- Supports sampling mode switch:
  - `timestamp`
  - `frame_step`

##### 2) Environment

At minimum, CLIP4Clip requires:

- `torch`
- `opencv-python`
- `transformers`
- `torchvision`
- `Pillow`

If dependencies are missing, the tool returns an `error` field in output JSON/JSONL.

##### 3) Quick Start (Manifest Mode)

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

##### 4) Manifest Format

The tool reads one JSON object per line.

###### Required per row

At least one of these video keys must exist:

- `world_model_video`
- `candidate_video`
- `reference_video`
- `video`

The tool will pick the first non-empty key in that order and normalize it as `video`.

###### Optional keys

- `sample_id` (or `id`)
- `prompt` (or `text_prompt`)

###### Minimal example

```json
{"sample_id":"s1","video":"/abs/path/video_1.mp4","prompt":"A ball rolling"}
{"sample_id":"s2","video":"/abs/path/video_2.mp4","prompt":"A car turning left"}
```


##### 5) Single Video Mode

```bash
python tools/evaluate_world_model_clip4clip.py \
  --video /abs/path/video.mp4 \
  --text_prompt "A ball rolls down a ramp" \
  --output_json reports/world_model_clip4clip/single.json \
  --sample_mode timestamp \
  --eval_fps 4 \
  --max_pairs 32
```

##### 6) Sampling Modes

- `timestamp`:
  - Time-based sampling with `eval_fps` as sampling FPS.
  - Similar behavior to LPIPS timestamp sampling.
- `frame_step`:
  - Index-step sampling from native video FPS.
  - `eval_fps` is treated as target FPS for step size.

##### 8) Output Fields

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

##### 9) Note on Framework Compatibility

- CLIP4Clip currently runs through the shared evaluation context (`reference` + `candidate`).
- In `tools/evaluate_world_model_clip4clip.py`, single-video use is implemented by assigning the same video to both internal slots.
- This keeps compatibility with the existing framework design and does not require changing LPIPS paths.

</details>

### DiffusionLight physics (lighting, legacy)

Earlier light-probe based lighting evaluator kept for reference; the released lighting dimension uses shadow consistency.

<details>
<summary><b>Full reference</b> (formerly <code>README_world_model_diffusionlight_physics_eval.md</code>)</summary>

This README covers physical-consistency evaluation between a world-model video and
a reference Unity video using:

- `lighting`: DiffusionLight HDR envmap direction consistency
- `motion`: RAFT optical-flow consistency
- `shadow`: SAM2 shadow-mask tracking consistency

##### Layout

- `evaluation/modules/lighting.py`: DiffusionLight-based lighting module
- `evaluation/modules/motion.py`: RAFT-based motion module
- `evaluation/modules/shadow.py`: SAM2-based shadow module
- `tools/evaluate_world_model_diffusionlight.py`: single-pair orchestrator
- `tools/evaluate_world_model_metrics.py`: standalone alias for the same entrypoint
- `scripts/evaluate_world_model_diffusionlight_physics.sh`: runnable command template

##### Dependencies

```bash
pip install numpy opencv-python imageio scipy torch torchvision lpips
pip install huggingface_hub
pip install --no-build-isolation git+https://github.com/facebookresearch/sam2.git
```

You also need DiffusionLight at `/workspace/unity/DiffusionLight` (or pass a
custom path with `--diffusionlight_repo`).

##### Usage

```bash
python tools/evaluate_world_model_diffusionlight.py \
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

##### Metrics

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

##### Known Calibration Gap

High `unity_json_angular_error_deg_*` may still occur even when
reference==candidate because Unity light Euler/world convention and
DiffusionLight envmap camera-frame convention are not fully calibrated yet.

Unity exports needed for reviewer-proof alignment:

- camera intrinsics/extrinsics per frame
- ground-truth optical flow maps
- screen-space shadow masks
- instance masks with persistent IDs

</details>

### GroundingDINO tracking

Shared open-vocabulary detection, tracking and segmentation helper used by several dimensions.

<details>
<summary><b>Full reference</b> (formerly <code>README_groundingdino.md</code>)</summary>

This folder contains copied GroundingDINO-related scripts from `VideoScience/judge/cv_tools`.
They are organized as:

- `tools/`: Python pipelines (`groundingdino_bytetrack.py`, `groundingdino_sam.py`)
- `scripts/`: runnable shell examples (`groundingdino_bytetrack*.sh`, `groundingdino_sam*.sh`)

##### Requirements

- Python 3.10+
- PyTorch + CUDA (optional but recommended)
- OpenCV (`cv2`), NumPy, Pillow, torchvision
- GroundingDINO package/codebase available in Python path
- `cjm_byte_track` for ByteTrack script
- SAM 2 package (`sam2`) for segmentation script

##### Quick usage

ByteTrack pipeline:

```bash
python tools/groundingdino_bytetrack.py \
  --video /path/to/video.mp4 \
  --text_prompt "ball . pendulum" \
  --grounding_cfg /path/to/GroundingDINO_SwinT_OGC.py \
  --grounding_ckpt /path/to/groundingdino_swint_ogc.pth \
  --output_dir mechanics_tracking_frames/run1 \
  --output_json mechanics_tracking_frames/run1/reports/tracking_report.json
```

SAM 2 pipeline:

```bash
python tools/groundingdino_sam.py \
  --video /path/to/video.mp4 \
  --text_prompt "ball . car" \
  --grounding_cfg /path/to/GroundingDINO_SwinT_OGC.py \
  --grounding_ckpt /path/to/groundingdino_swint_ogc.pth \
  --output_dir mechanics_tracking_frames/run1/sam \
  --output_json mechanics_tracking_frames/run1/reports/sam_report.json
```

Use the scripts in `scripts/` as ready-to-edit templates for concrete runs, for example:

```bash
bash scripts/groundingdino_bytetrack.sh
bash scripts/groundingdino_sam.sh
```

</details>


## Citation

If you find this repository helpful, please kindly cite:
```
@article{hu2026wmeval,
  title   = {WMEval: Evaluating Interactive World Models Grounded on Physical Simulations},
  author  = {Hu, Lanxiang and Huang, Yixin and Shankarampeta, Abhilash and Dai, Zilin and
             Wu, Yuyuan and Qian, Kun and Liu, Junze and Srinivasan, Arvind and Huo, Mingjia and
             Lin, Will and Zhong, Kai and Zhang, Weiqi and Wang, Tian and Cui, Qingjun and
             Rosing, Tajana and Zhang, Hao},
  journal = {arXiv preprint},
  year    = {2026}
}
```
