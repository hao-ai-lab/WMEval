---
pretty_name: WMEval Lite
license: other
license_name: unity-benchmark-scene-license-v1.0
license_link: https://huggingface.co/datasets/lmgame/WMEval-Lite/blob/main/LICENSE-UNITY.txt
language:
  - en
tags:
  - world-models
  - video-generation
  - interactive-video
  - evaluation
  - benchmark
  - unity
size_categories:
  - n<1K
dataset_info:
  - config_name: default
    features:
      - name: case_id
        dtype: string
      - name: scene
        dtype: string
      - name: variant
        dtype: string
      - name: group
        dtype: string
      - name: family
        dtype: string
      - name: action_target
        dtype: string
      - name: suites
        sequence: string
      - name: dimensions
        sequence: string
      - name: perspective
        dtype: string
      - name: prompt
        dtype: string
      - name: conditioning_case
        dtype: string
      - name: reference_video
        dtype: string
      - name: first_frame
        dtype: string
      - name: trajectory
        dtype: string
      - name: candidate_normalization
        dtype: string
      - name: reference_status
        dtype: string
configs:
  - config_name: default
    data_files:
      - split: test_full
        path: test_full.jsonl
      - split: test_core
        path: test_core.jsonl
---

# WMEval Lite (v1)

## Summary

WMEval Lite is the Unity-free evaluation set of WMEval, a benchmark for
**interactive world models**: video generators that take a starting frame, a
scene description and a control signal (keyboard / mouse / camera actions) and
roll the world forward. Each case is a scene rendered in Unity under a known
action script. This dataset ships the **generation conditions** (prompt,
conditioning frame, action script) and the **frozen Unity references** (videos,
ground-truth geometry and frozen judge outputs). You generate videos from the
conditions with your model, then score them against the frozen references with
the open-source evaluator. You do not need Unity, the private scene projects,
or help from the maintainers.

- Suite: `WMEval Lite`, version `v1`, release id `wmeval-lite-v1`
- 32 cases across 9 Unity scenes, 8 evaluation dimensions, 43 case-dimension
  assignments (one case can be scored by several dimensions)
- Lighting is scored on one case, `SunRisesAndSets/lighting`, with the paper's
  own (native) protocol; the case also ships a Unity reference orbit from the
  same first frame.

## Dataset structure

The viewer shows two splits and opens on `test_full`:

- `test_full` (252 rows): `test_core` plus 220 synthetic action trajectories
  under `synthetic/` (camera yaw, look-at orbit, vehicle, cue ball, rolling ball,
  beam-domino ball, bottle motion, character). Each is built text first: an action
  description (a short sequence of steps such as "turns left at 30 degrees per
  second for 2 seconds") forms the second half of its prompt and is compiled into
  its action sequence; the first half is the scene description of the core case
  whose conditioning frame and initial state it reuses. `meta.json` stores the
  description and its steps. Each also has a Unity reference in its
  `reference/` folder, rendered by replaying the same actions in the core case's
  Unity scene (`reference_status: unity_rendered`); `synthetic/references.json`
  in the code repository records checksums, render provenance and how closely
  frame 0 matches the core frame (`reference_quality: weak_first_frame` below
  20 dB). Their `suites` value is `wmeval_v1`; core rows carry the
  Lite suite they belong to (`wmeval_lite_v1`). `run_wmeval_lite.py run
  --split synthetic` (or `full`) scores them on their declared dimensions with
  the conditioning case's protocol, against their own reference and its
  `reference/artifacts/` (Unity ground truth plus the same pre-computed inputs
  as the core cases: LPIPS contact events, CausalStep QA and recorded reference
  answers, liquid masks and flow, rubrics); their aggregate is reported apart
  from the core headline, with `reference_quality` on every row.
- `test_core` (32 rows): the curated WMEval Lite v1 cases, each with a Unity
  reference rendering. The paper's numbers come from this split.

Each row links to its first frame, trajectory and reference video.

This Hub dataset holds the case data: every file under each case's `reference/`
directory (reference videos, conditioning frames, action trajectories, Unity
ground-truth arrays and frozen judge outputs), at the same paths as in the code
repository [hao-ai-lab/WMEval](https://github.com/hao-ai-lab/WMEval). The code
repository keeps `suite.yaml`, each `case.yaml` and the `reference/checksums.sha256`
manifests; `python tools/sync_lite_data.py download` places the files and verifies
every checksum.


```
README.md                         # this card
MANIFEST.json                     # sha256 + size of every file, dataset id/version,
                                  # evaluator code revision, schema versions
suite.yaml                        # dimension-level manifest: which cases each
                                  # dimension scores, metric config, aggregation
cases/<Scene>/<variant>/
  case.yaml                       # prompt, perspective, modules, protocol
                                  # (duration rules), evaluator_lock, per-module params
  reference/
    reference.mp4 | unity_groundtruth.mp4 | unity_orbit.mp4
                                  # the frozen Unity reference video
    ref.png                       # conditioning frame, where it differs from frame 0
    checksums.sha256              # covers every file under reference/
    artifacts/
      actions.txt                 # human-readable action script
      action.npy / action.npz     # numeric actions
      trajectory.yaml             # semantic trajectory (Table 4/7 and newer cases)
      actions.jsonl               # browser key program / camera pose samples
      poses.npy, intrinsics.npy,  # Unity camera and object ground truth
      object_transforms.json, object_masses.json, plane_calibration.json, ...
      rubrics/<Scene>/<provider>_<model>_<version>.json   # prompt-judge rubrics
      judge_answers.json          # frozen causality judge passes (BeamDomino)
      reference_scores.json       # frozen prompt-judge scores of the Unity video
      contact_events.json         # frozen LPIPS reference contact events
      liquid_mask.npz, bottle_mask.npz, raft_pooled_flow.npz   # LiquidContainer
human_alignment/
  prompt_scene_consistency.json   # the human side of the judge-human study
baselines/
  lighting_paper_native.json      # the paper's lighting table: per-run scores of the
                                  # six evaluated models (5 seeds each) on SunRisesAndSets/lighting
```

Every case bundle can be checked on its own: `checksums.sha256` covers its
`reference/` tree, and `case.yaml`'s `evaluator_lock` pins the evaluator
fingerprint, module configuration and dependency versions it was built with.

## Dimensions and case counts

Taken from `suite.yaml` (`dimensions.<id>.cases`):

| Dimension (`suite.yaml` id) | Cases | Needs reference video | Headline metric (direction) |
| --- | ---: | :---: | --- |
| LPIPS (causal) (`lpips`) | 7 | yes | `lpips_mean` (lower) |
| Action Controllability (`action_controllability`) | 3 | yes | `relative_epe_dynamic_mean_raw` (lower) |
| Prompt-Scene Consistency (`prompt_scene_consistency`) | 6 | no (first frame only) | `overall_score` (higher) |
| Lighting & Shading (`lighting_shading`) | 1 | no | `shadow_direction_jitter_deg_per_frame` (lower) |
| Environment Object Consistency (`environment_object_consistency`) | 16 | yes | `object_permanence_pct` (higher) |
| Kinematics & Mechanics (`kinematics_mechanics`) | 3 | yes | `fde_m_mean` (lower) |
| Physical Causality (`physical_causality`) | 3 | yes | `csr_delta` (lower) |
| Fluid Mechanics (`fluid_mechanics`) | 4 | yes | `velocity_cosine_similarity` (higher) |
| **Total assignments** | **43** | | |

Cases per scene: MirrorRollingBall 7, BilliardGame 5, WanderingGarden 5,
LiquidContainer 4, BeamDomino 3, ChessBoard 3, RacingCar 3, BallRoller 1,
SunRisesAndSets 1 (`SunRisesAndSets/lighting`, the paper's native lighting
case, scored reference-free exactly as in the paper; its `reference/unity_orbit.mp4`
is a Unity rendering of the same walk from the same first frame, which the score
does not read).

Each dimension's `extra_means` in `suite.yaml` lists the secondary metrics
that are aggregated alongside the headline. Several dimensions carry caveats
on how to read their headline metric, documented in their `description` field.

## How to score a model

The evaluator lives in the WMEval code repository. `MANIFEST.json` records
the code revision (`evaluator_code.revision`) this package was exported from.
Use that revision for results you can compare against published numbers.

1. **Check the download.** From a checkout of the evaluator code:

   ```sh
   python tools/dataset/export_hf_package.py verify <dataset_dir>
   python tools/run_wmeval_lite.py validate --suite <dataset_dir>
   python tools/run_wmeval_lite.py list-cases --suite <dataset_dir>
   ```

2. **Generate one video per case and seed** from the case's conditions:
   - the prompt: `case.yaml` `prompt`;
   - the conditioning frame: `reference/ref.png` if present, otherwise frame 0
     of the reference video;
   - the controls: `reference/artifacts/actions.txt` (and the numeric
     `action.npy` / `action.npz`, or `trajectory.yaml` where present).

   Reference-based dimensions compare candidate frame *f* with Unity frame
   *f*. Match the reference's fps, resolution and duration, or follow the
   case's `protocol` block (`candidate_normalization`, `min_duration_sec`,
   `min_duration_ratio`). The case's `protocol` decides when a clip counts as
   truncated.

3. **Lay the videos out** as `<model_root>/<model>/<Scene>/<variant>/<seed>.mp4`,
   or write a JSONL manifest with `model`, `case_id`, `seed`,
   `candidate_video[, params]`.

4. **Run the suite:**

   ```sh
   python tools/run_wmeval_lite.py run \
     --suite <dataset_dir> \
     --model-root <model_root>/<model> \
     --output-dir reports/<model> --resume
   # --dimensions lpips,fluid_mechanics   to run a subset
   # --candidate-manifest rows.jsonl      instead of --model-root
   ```

   The output directory receives `per_case_results.jsonl`,
   `aggregate_summary.json` and `aggregate_summary.csv`. To check a prompt-judge run
   against the human study, use
   `python tools/dataset/check_prompt_human_alignment.py --fixture <dataset_dir>/human_alignment/prompt_scene_consistency.json ...`.

Some `metric_config` entries in `suite.yaml` (`runs/causal_step_cache`,
`runs/liquid_cache`) are cache directories that the evaluator *writes* to.
They resolve against your working directory and are not part of the dataset.

## What is frozen

The following are recorded once and shipped. Scoring never recomputes them
from Unity:

- **Reference videos and arrays.** These are the Unity reference videos and
  the conditioning frames. They also include Unity camera poses and
  intrinsics, object transforms, masses and diameters, plane calibrations,
  and the compiled action arrays. Every file is covered by the bundle's
  `checksums.sha256` and by `MANIFEST.json`.
- **Frozen judge outputs.** Wherever a reference side needs a hosted
  VLM judge, the reference answers are frozen so that the reference side does
  not drift between runs or cost API calls:
  - *Physical causality*: `reference/artifacts/judge_answers.json` in the three
    BeamDomino cases holds five recorded judge passes on the Unity reference.
    A live judge whose model, frame count or prompt version differs from the
    recording is refused.
  - *Prompt-scene consistency*: `reference/artifacts/reference_scores.json` in
    the six prompt cases holds the Unity video's frozen scores, reported as
    `*_unity`. The judge rubrics ship in `reference/artifacts/rubrics/`.
  - *LPIPS contact events*: `reference/artifacts/contact_events.json` holds the
    frozen reference-side contact events for `MirrorRollingBall/lpips_collision`,
    `BeamDomino/causal` and `BallRoller/ctrl`. The four LiquidContainer cases
    ship it too, with 0 consensus events (no candidate contact reached 3 of 5
    passes), so LPIPS uses its temporal fallback there.
- **Reference-side CV outputs** for fluid mechanics: the SAM2 liquid and
  bottle masks and the pooled RAFT flow of each LiquidContainer reference
  (`*.npz`). Only the candidate side runs SAM2 and RAFT.
- **Human study**: `human_alignment/prompt_scene_consistency.json`.
- **Baselines**: `baselines/lighting_paper_native.json` holds the paper's
  lighting scores (six models, five seeds, every per-run metric), which the
  evaluator reproduces exactly.


### Frozen judge outputs

Hosted VLM judges are sampled, so judging the Unity reference on every run would
make the reference a moving target. The reference side is judged once, every
pass is stored in the case bundle, and runtime reads the stored result.

| Dimension | File (per case) | Tool | Passes | Stale when |
|---|---|---|---|---|
| Physical causality | `reference/artifacts/judge_answers.json` | `tools/dataset/record_causal_reference_answers.py` | 5 | judge model, frame count `K`, or prompt version differ from the live judge (runtime refuses the replay) |
| Prompt consistency | `reference/artifacts/reference_scores.json` | `tools/dataset/freeze_prompt_reference_scores.py` | 5 | judge model, prompt version, alpha, rubric, system prompts, sampling, `ref.png` or the Unity video change (rows mark the reference `stale`) |
| LPIPS | `reference/artifacts/contact_events.json` | `tools/dataset/freeze_reference_events.py` | 5, majority (3/5) | the reference video's SHA-256 changes (the loader rejects it) |

### Re-baselining

Re-baseline only when the judge or the reference changes, and re-baseline every
case of the dimension together.

```bash
# 1. Re-judge (hosted judge; explicit opt-in via OPENAI_API_KEY)
python tools/dataset/record_causal_reference_answers.py lite/cases/BeamDomino/causal --passes 5
python tools/dataset/freeze_prompt_reference_scores.py --passes 5 --register
python tools/dataset/freeze_reference_events.py MirrorRollingBall/lpips_collision --passes 5 --register

# Same, through an OpenAI-compatible gateway serving the same model:
python tools/judge_gateway.py --key-file <key> --script tools/dataset/freeze_reference_events.py -- \
    LiquidContainer/fluid_rest --passes 5 --register --allow-empty

# 2. Re-lock each touched case (checksums + evaluator fingerprint)
python tools/dataset/build_lite_case.py lite/cases/<Scene>/<variant>

# 3. Verify
python tools/run_wmeval_lite.py validate --suite lite
python -m pytest -q tests
```

Record in the commit message: judge model and snapshot, pass count, transport,
and the before/after reference numbers. Results scored against an older baseline
are not comparable with results scored against the new one; re-score both sides.

## External requirements

Scoring a new model still needs the following, none of which ship in this
dataset:

- **The evaluator code** (WMEval repository) at the revision in
  `MANIFEST.json`, with its Python dependencies (`requirements/*.lock`).
- **A GPU** for the CV-based dimensions: SAM2, GroundingDINO, Depth Anything,
  DINOv2, RAFT and LPIPS. Suite validation and dry runs need only PyYAML.
- **CV checkpoints**, downloaded on first use from their upstream hosts. The
  pinned revisions and weight SHA-256s are listed in the evaluator repository's
  `experiments/protocol/evaluator_assets.lock.json`.
  DINOv2 is loaded through `torch.hub`, which needs network access to GitHub on
  first use.
- **A hosted VLM judge** (OpenAI API, `OPENAI_API_KEY`) for the candidate side
  of `prompt_scene_consistency`, `physical_causality` and causal `lpips`. The
  judge models are pinned in `suite.yaml` (`gpt-5.4`, plus `gpt-5.4-mini` for
  the LPIPS coarse pass). These calls cost money, and the judge is sampled, not
  deterministic.

## Differences from the paper

See [`DIFFERENCES_FROM_PAPER.md`](DIFFERENCES_FROM_PAPER.md): roster and
prompt-driven rows dropped as N/A, per-dimension action trajectories (including
the prompt cases' as-run inputs), frozen judge outputs and their effect on the
Unity reference numbers, and evaluation behavior added after the paper. How the
frozen judge outputs are produced and re-baselined: "Frozen judge outputs" above.

## License

The Unity benchmark scenes, and any output they produce, including the reference rollouts and Unity-logged
quantities in this dataset, are licensed under the Unity Benchmark Scene License v1.0
([`LICENSE-UNITY.txt`](LICENSE-UNITY.txt)), reproduced below.

> Unity Benchmark Scene License v1.0
>
> Copyright (c) 2026 Unity Technologies SF. All rights reserved.
>
> Permission to execute and redistribute this software in unmodified binary form is granted, free of charge, solely to run the [UCSD AI World Model Benchmarking Protocol] (the "Protocol") in connection with this software for the purpose of producing or validating the Protocol's benchmarking metrics (the "Permitted Use"), subject in all instances to the following conditions:
>
> The software, and any output produced by it, may not be used for any purpose other than the Permitted Use. In particular, neither the software nor its output may be used as training data, fine-tuning data, evaluation input, retrieval corpus, embedding source, or any other input to any AI or machine learning model, except solely as the reference standard against which a model's independently generated output is compared.
>
> Embedded assets, models, textures, animations, audio, shaders, and scripts may not be extracted, separated, re-rendered, or otherwise obtained from the software.
>
> The software may not be modified, decompiled, or reverse engineered, nor incorporated into other software, except to the extent applicable law prohibits such restriction.
>
> Redistributions must be unmodified, must retain this license and any accompanying NOTICE file, and must impose this license on each recipient.
>
> No trademark rights are granted.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL UNITY TECHNOLOGIES SF OR ITS AFFILIATES BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR ITS USE. ANY BREACH OF THIS LICENSE TERMINATES ALL RIGHTS GRANTED UNDER IT.

Third-party code and model weights used by the evaluator are listed in
`THIRD_PARTY_NOTICES.md` in the evaluator repository.

## Citation

See `CITATION.cff` in the evaluator repository.
