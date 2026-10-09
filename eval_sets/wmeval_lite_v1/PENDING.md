# WMEval Lite release — pending work by evaluation dimension

Updated 2026-10-09 · branch `lanxiang/release-cleanup` · paper: ready · project
page preview: https://snyhlx.github.io/wmeval-blog

**Release goal.** A new user downloads the dataset from Hugging Face, generates
their model's videos from the supplied conditions, and scores them against
frozen Unity references — without Unity, private repos, or maintainer help.
GPUs, CV checkpoints and a hosted judge (OpenAI) may still be required; each
must be declared, never hidden.

**The three gaps that block "out of the box" use** (details per dimension below):

1. **Actions** — not every case has an executable, target-aware action script,
   and several paper models have no generation path in this repo.
2. **Frozen references** — two dimensions still re-judge or omit the Unity
   reference at evaluation time.
3. **Normalization** — 12 of 19 cases do not resample candidate videos to the
   reference fps / resolution / duration.

---

## How to use this file

- **Owners:** find your dimension section. Fill in `Owner:`, then work the
  checklist top to bottom. Tick `[x]` only with evidence (commit hash, output
  path, or command output).
- **Agents:** each task has **Do / Files / Check / Done when**. Execute tasks in
  order, run the **Check** command after each change, and stop and report if a
  check fails. Follow the ground rules below without exception.
- **Status legend:** ✅ ready · ⚠️ partial · ❌ missing · — out of scope / not in
  the paper for this dimension.

### Ground rules (all owners and agents)

1. **Never edit hash-locked inputs in place.** This covers
   `experiments/tables_4_7/trajectories/*.yaml`, `experiments/tables_4_7/compiled/*`,
   and every file under `cases/*/*/reference/`. Change a case by re-authoring
   it (see *Adding or re-locking a case*), never by hand-editing its files.
2. **Unsupported ≠ zero.** If a model cannot express a case's control (e.g. a
   camera-only model asked to steer a car), record `unsupported` with a reason.
   Never substitute another control (moving the camera instead of the car).
3. **Paper tables are frozen.** Five-run paper selections stay as published;
   30-run robustness results go to `reports/tables_4_7/final/` only.
   Reconstructed (orange) rows stay labeled as non-parity.
4. **Verify after every change:**
   ```bash
   python tools/run_wmeval_lite.py validate --suite eval_sets/wmeval_lite_v1
   python -m pytest -q tests
   ```
5. **No hidden paid calls.** Anything that calls a hosted judge needs an explicit
   opt-in and must record model identity, requests and responses.

### Adding or re-locking a case

1. `mkdir -p cases/<Scene>/<variant>/reference/artifacts` and add the files.
2. Write `case.yaml` (copy a live case of the same kind).
3. `python tools/build_lite_case.py cases/<Scene>/<variant>` (checksums + lock).
4. Add the case id to the dimension's `cases:` list in `suite.yaml`.
5. `python tools/run_wmeval_lite.py validate` (+ `generate-rubrics` for prompt cases).

---

## Shared contract every dimension must meet

Each case in the HF release ships these, keyed by hashes:

| Item | File(s) | Required content |
| --- | --- | --- |
| Conditioning | `conditioning/first_frame.png`, prompt in `generation.json` | Exact RGB pixels (no "frame 0 of the video" fallback); frame-0 convention (supplied image vs. first prediction) |
| Actions | `conditioning/actions.json` | Versioned schema: target entity (`camera` / `ego_vehicle` / `cue_ball` / `table_objects` / `bottle` / `avatar`), channels with units and range, coordinate frame, absolute vs. relative, timestamps, duration, control rate, action-to-frame latency |
| Reference | `reference/manifest.json`, `reference/frames/`, `frame_index.json`, `artifacts/` | Every array a metric reads, with units, shapes, frames, timebase and provenance (`unity_export` / `calibration` / `cv_pseudo_label` / `vlm_annotation`) |
| Frozen judgments | in `reference/artifacts/` | Any reference-side VLM output (events, answers, scores), keyed by judge model + prompt version + sampled frames |
| Normalization | `case.yaml: candidate_normalization` | Explicit fps, resolution, duration, crop/pad and time origin; the transform is recorded per result row |

Backend capabilities and calibration for Table 4/7 already live in
`experiments/tables_4_7/action_backends.yaml` (read by
`evaluation/action_backends.py`); extend that registry rather than hard-coding
model constants.

### Model roster

| Model | Interface | Generation code in this branch |
| --- | --- | --- |
| Unity (reference) | runtime args | `tools/run_unity_table_scenes.py` (Table 4/7); Lite references are pre-baked |
| Project Genie | browser keys (rule-based, uncalibrated; left turn calibrated 51.47 °/s) | `tools/project_genie_agent.py`, `tools/run_project_genie_tables.py` |
| Matrix-Game 2.0 | keys + mouse, 129 frames @ 12 fps | `tools/prepare_matrix_game_inputs.py`, `tools/run_matrix_game2.py` (Table 4/7 only) |
| Matrix-Game 3.0 | keys + mouse, 177 frames @ 17 fps | `tools/prepare_matrix_game_inputs.py`, `tools/run_matrix_game3.py` (Table 7 only) |
| LingBot-World (Cam) | camera poses, 161 @ 16 fps | `tools/prepare_lingbot_table7_inputs.py`, `tools/run_lingbot_table7*.py` (Table 7 only) |
| LingBot-World (Act) | action-conditioned | ❌ none |
| HY-WorldPlay | — | ❌ none (licensing to confirm) |
| Waypoint-1.5 | keys + mouse, first person | `tools/generate_waypoint15_candidates_modal.py` (Lite, Modal; hand-written overrides for 3 cases) |

---

## 1. Environment object & character consistency — `environment_object_consistency`

**Owner:** _TBD_ · **Paper:** Table 7 · **Lite cases (6):** BallRoller/ctrl,
BilliardGame/prompt_first, ChessBoard/walk_third, MirrorRollingBall/prompt_third,
RacingCar/prompt_third, WanderingGarden/scene_first

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| BallRoller/ctrl | ✅ keys + `action.npy` | ✅ poses, transforms, masses | ❌ none |
| MirrorRollingBall/prompt_third | ✅ camera rotation | ✅ poses, diameters (depth via photogrammetric workaround) | ❌ none |
| RacingCar/prompt_third | ✅ camera rotation | ✅ poses, transforms | ❌ none |
| BilliardGame/prompt_first | ❌ no action file | ✅ poses, transforms | ❌ none |
| ChessBoard/walk_third | ❌ comment-only table | ✅ poses, transforms | ❌ none |
| WanderingGarden/scene_first | ❌ no action file | ✅ poses, transforms | ❌ none |

| Model | In paper | Generation path | Status |
| --- | --- | --- | --- |
| Project Genie | ✅ (reconstructed row) | Table 7 browser runs | ✅ 5-run Table 7 complete |
| LingBot-World (Cam) | ✅ | Table 7 pose inputs | ✅ 5-run complete; adaptive 30-run expansion 100/360 |
| Matrix-Game 3.0 | ✅ | Table 7 schedules | ✅ 5-run complete |
| Matrix-Game 2.0 | ✅ | Table 7 schedules | ✅ 5-run complete |
| Waypoint-1.5 | — | Lite (Modal) | ✅ scored on all 6 Lite cases (overrides for Billiard/ChessBoard/Garden) |

- [ ] **Write executable actions for the 3 missing cases.**
  Do: author `actions.json` (target `camera` or `avatar`) for BilliardGame/prompt_first,
  ChessBoard/walk_third and WanderingGarden/scene_first from their capture logs; mark each
  as `exact` or `reconstructed`. Files: `cases/<case>/reference/artifacts/`. Check:
  `python tools/audit_trajectory_readiness.py --output reports/tables_4_7/calibration/trajectory_readiness.json`.
  Done when: all 6 cases are `native_machine_readable_ready` and Waypoint's
  `CASE_OVERRIDES` are no longer needed.
- [ ] **Lock normalization for all 6 cases.** Do: add a resampling token (reference
  fps/resolution/duration) to each `case.yaml` and record the transform per row.
  Done when: a 60 fps 1280×720 candidate scores the same as its pre-resampled copy.
- [ ] **Add the anchor validity gate.** Do: require ≥2 visible, confidently assigned
  anchors and a minimum paired-frame coverage before reporting pairwise-distance MAE;
  otherwise emit `unmeasurable`. Done when: single-anchor cases (Mirror, RacingCar,
  Garden) report `unmeasurable` instead of null-mixed means.
- [ ] **Report realized camera motion** next to every Table 7 score (LingBot realized
  −17.7/+41.4° for ±45° and +54.9/+277.9° for full turns). Done when: realized yaw is a
  column in the exported tables.
- [ ] **Publish per-scene self-match bounds** (Unity vs. Unity) and the Mirror
  photogrammetric-depth caveat in the case manifest.

## 2. Action controllability — `action_controllability`

**Owner:** _TBD_ · **Paper:** controllability table · **Lite cases (2):**
BallRoller/ctrl, BeamDomino/causal

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| BallRoller/ctrl | ✅ keys + `action.npy` (target not declared) | ✅ video (flow computed at eval time) | ❌ none |
| BeamDomino/causal | ⚠️ 2-channel ball force, no key program | ✅ video | ✅ `beamdomino_native` |

| Model | In paper | Generation path | Status |
| --- | --- | --- | --- |
| Project Genie | ✅ | Lite key program (BallRoller); none for BeamDomino force | ⚠️ paper rows not reproducible from this branch for BeamDomino |
| LingBot-World (Act) | ✅ | ❌ no adapter | ❌ |
| Waypoint-1.5 | ✅ (GT-RAFT only) | Lite (Modal) | ⚠️ scored; WASD moves the ego camera, not the ball (expected ambiguity) |
| Matrix-Game 2.0/3.0, LingBot (Cam), HY-WorldPlay | — | — | — |

- [ ] **Declare the action target** (`cue_ball` for BallRoller, `ball` for BeamDomino) in
  `actions.json`; models without an object channel report `unsupported` instead of a
  flow error. Done when: Waypoint rows show `unsupported`, not 755 px EPE.
- [ ] **Add a BeamDomino key/browser adapter or mark it unsupported** per backend in
  `action_backends.yaml`.
- [ ] **Bake the reference flow** (RAFT on the Unity video) into the case so it is not
  recomputed per candidate.
- [ ] **Vendor or package PTLFlow** for the synthetic-flow target; pin its checkpoint.
- [ ] **Lock normalization** for BallRoller/ctrl; promote relative EPE (scale-free) over
  pixel EPE as the headline.
- [ ] **Add a MirrorRollingBall controllability case** (in the paper, not in Lite).

## 3. Kinematics & mechanics — `kinematics_mechanics`

**Owner:** _TBD_ · **Paper:** Table 4 · **Lite cases (3):** BallRoller/ctrl,
RacingCar/prompt_third, BilliardGame/prompt_first

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| BallRoller/ctrl | ✅ keys + `action.npy` | ✅ transforms, masses | ❌ none |
| RacingCar/prompt_third | ❌ **camera-only rotation around a parked car** (no driving) | ✅ transforms, masses | ❌ none |
| BilliardGame/prompt_first | ❌ no action file (cue strike is a scene event) | ✅ raw capture GT | ❌ none |

| Model | In paper | Generation path | Status |
| --- | --- | --- | --- |
| Project Genie | ✅ (reconstructed row) | Table 4 browser runs (RacingCar third, Billiard first) | ✅ 5-run complete |
| Matrix-Game 2.0 | ✅ | Table 4 schedules (thresholded WASD) | ⚠️ generated; car/ball target not validated |
| LingBot-World (Cam) | — | — | — unsupported (camera input only) |
| Waypoint-1.5 | — | Lite (Modal) | ⚠️ scored on wrong-action cases |

- [ ] **Replace the RacingCar action.** Do: create a RacingCar kinematics case driven by
  `experiments/tables_4_7/trajectories/racing_car_drive.yaml` (steer/throttle/brake), or
  drop `RacingCar/prompt_third` from this dimension in `suite.yaml`. Done when: the
  kinematics case's action target is `ego_vehicle`.
- [ ] **Give BilliardGame an executable action** (ball control from
  `billiard_first_person_ball_control.yaml`) or remove it from this dimension.
- [ ] **Validate target control for Matrix-Game 2.0.** Do: probe steering/throttle/brake
  and ball forward/lateral; reject outputs where the camera or an avatar moved instead
  of the named target. Done when: a target-success rate is recorded per run.
- [ ] **Lock normalization** for all 3 cases.
- [ ] **Units and floors.** Add held-out reprojection checks, label exact-match as
  diagnostic, and add an absolute-error floor for near-zero motion (momentum/energy).

## 4. Fluid mechanics — `fluid_mechanics`

**Owner:** _TBD_ · **Paper:** LiquidContainer table · **Lite cases (4):** rest,
step_hold, impulse_decay, chirp_strafe

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| all 4 profiles | ⚠️ 3-channel bottle motion (forward/right/up); no key/browser adapter | ✅ liquid/bottle masks, pooled flow | ✅ `liquid_832x464_16fps` |

| Model | In paper | Generation path | Status |
| --- | --- | --- | --- |
| Project Genie | ✅ | ❌ no bottle adapter (not machine-executable) | ❌ not reproducible from this branch |
| LingBot-World | ✅ | ❌ | ❌ |
| Matrix-Game 2.0 | ✅ | ❌ | ❌ |
| Matrix-Game 3.0 | ✅ | ❌ | ❌ |
| HY-WorldPlay | ✅ | ❌ | ❌ |
| Waypoint-1.5 | — | not run | — |

- [ ] **Define the bottle action target.** Do: `actions.json` with target `bottle`, units
  m/s² in the world frame; add a `bottle` row to `action_backends.yaml` with each model's
  support status. Done when: keyboard-only models report `unsupported` or use a
  documented, calibrated mapping.
- [ ] **Recover or regenerate the paper's fluid rollouts** for the 5 paper models with a
  recorded action receipt each (they were produced outside this branch).
- [ ] **Pin SAM 2 and RAFT checkpoints** used on the candidate side.
- [ ] **Keep the per-profile, multi-axis panel canonical**; label any scalar headline as
  navigation-only.

## 5. Physical causality — `physical_causality`

**Owner:** _TBD_ · **Paper:** BeamDomino causality table · **Lite cases (3):**
BeamDomino causal / causal_glancing / causal_direct_strike

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| all 3 | ⚠️ 2-channel ball force; no key/browser program | ✅ **frozen** QA + `judge_answers.json` (5 passes) | ✅ `beamdomino_native` |

| Model | In paper | Generation path | Status |
| --- | --- | --- | --- |
| Project Genie | ✅ | ❌ no browser program for a force action | ⚠️ paper rows not reproducible from this branch |
| LingBot-World (Cam) | ✅ | ❌ camera-only interface cannot strike the ball | ⚠️ confirm what was executed for the paper row |
| Waypoint-1.5 | — | not run | — |

- [ ] **Define the strike action** (target `ball`, force units) and each backend's support
  in `action_backends.yaml`.
- [ ] **Audit the LingBot (Cam) paper row**: a camera-pose model cannot apply the strike;
  document the executed condition or mark the row as a different protocol.
- [ ] **Record every candidate judge pass** (`causal_step_judge_repeats: 5` is
  load-bearing: CSR from one pass can only be 0 or 100).
- [ ] Re-baseline the frozen reference answers whenever the judge model or prompt changes.

## 6. Lighting & shading — `lighting_shading`

**Owner:** _TBD_ · **Paper:** SunRisesAndSets table · **Lite case (1):**
SunRisesAndSets/lighting (first person, fixed sun, walking orbit)

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| SunRisesAndSets/lighting | ⚠️ per-model `actions_per_model.json`; orbit reconstructed | — reference-free (analytic geometry floor) | ❌ none (shadow jitter is per frame → fps-dependent) |

| Model | In paper | Generation path | Status |
| --- | --- | --- | --- |
| Project Genie | ✅ | existing rollouts (lighting repro volume) | ✅ reproduces paper |
| LingBot-World | ✅ | existing rollouts | ✅ reproduces paper |
| Matrix-Game 2.0 | ✅ | existing rollouts | ⚠️ reproduces paper except SRR |
| Matrix-Game 3.0 | ✅ | existing rollouts | ✅ reproduces paper |
| HY-WorldPlay | ✅ | existing rollouts | ✅ reproduces paper |
| Waypoint-1.5 | ✅ | Lite (Modal) | ✅ identical to paper row |

- [ ] **Replace the per-model action file with one semantic orbit trajectory** and
  compile it per backend; record the exact Unity controller trace if recoverable.
- [ ] **Normalize to a fixed fps** (or report jitter per second) so models at different
  frame rates are comparable.
- [ ] **Bundle or pin the rollouts** in the HF baseline set (they live on a Modal volume).
- [ ] Optional, after release: `ChessBoard/lighting` (needs a team scope decision) and a
  separate moving-sun `SunRisesAndSets/lighting_sunsweep` protocol.

## 7. Prompt-scene consistency — `prompt_scene_consistency`

**Owner:** _TBD_ · **Paper:** prompt-consistency table · **Lite cases (6):**
ChessBoard, MirrorRollingBall, RacingCar × prompt_first / prompt_third

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| all 6 | ✅ 10 s camera rotation | ⚠️ rubrics frozen; **Unity reference scores not stored**; candidate judge is a mutable hosted `gpt-5.4` alias | — low risk (sampled frames) |

| Model | In paper | Generation path | Status |
| --- | --- | --- | --- |
| Project Genie | ✅ | Lite key programs | ⚠️ paper rows; rerun needs judge snapshot |
| LingBot-World (Cam) | ✅ | ❌ no Lite path | ❌ |
| Matrix-Game 3.0 | ✅ | ❌ no Lite path | ❌ |
| HY-WorldPlay | ✅ | ❌ | ❌ |
| Waypoint-1.5 | ✅ | Lite (Modal) | ✅ scored 30/30 rows |

- [ ] **Freeze the Unity reference scores** per case, keyed by judge model + prompt
  version + sampled frames. Done when: reference rows load from the bundle, not an API.
- [ ] **Record judge identity, requests and responses** for every candidate row; publish
  the human-alignment fixture (ρ = 0.90) as a regression check with tolerance bands.
- [ ] **Add Lite generation paths** for LingBot (Cam) and Matrix-Game 3.0 (camera
  rotation is supported by both via `action_backends.yaml`).

## 8. Perceptual similarity — `lpips`

**Owner:** _TBD_ · **Paper:** causal-LPIPS table · **Lite cases (6):**
MirrorRollingBall/lpips_collision, BeamDomino/causal, LiquidContainer × 4

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| MirrorRollingBall/lpips_collision | ✅ keys + `action.npy` | ⚠️ **contact events re-judged per candidate** | ❌ none |
| BeamDomino/causal | ⚠️ ball force (see §5) | ⚠️ events re-judged (0/1/3/0/0 contacts on one reference) | ✅ native |
| LiquidContainer × 4 | ⚠️ bottle motion (see §4) | ⚠️ events re-judged | ✅ liquid |

| Model | In paper | Generation path | Status |
| --- | --- | --- | --- |
| Project Genie | ✅ | Lite key program (Mirror) | ⚠️ |
| LingBot-World (Act) | ✅ | ❌ no adapter | ❌ |
| Waypoint-1.5 | — | Lite (Modal) | ⚠️ BeamDomino fell back to temporal LPIPS (no candidate contacts) |

- [ ] **Freeze reference contact events** once per case (judge model + prompt version +
  frame sampling). Done when: every candidate row reuses the same reference events.
- [ ] **Flag fallbacks per row**: `causal` vs `temporal_only` must never be averaged
  together in one headline.
- [ ] **Lock normalization** for MirrorRollingBall/lpips_collision.

---

## Release owner — cross-cutting items

- [ ] **Scope and rights (P0-01).** Freeze the 19-case / 8-dimension roster and the
  repeat policy; clear redistribution of Unity renders, quantities and annotations;
  add LICENSE files, attribution, citation and a dataset card. Confirm HY-WorldPlay.
- [ ] **HF package (P0-02).** Export → download → verify into a clean directory with
  per-file hashes, dataset/evaluator/schema versions; nothing may resolve into `runs/`,
  a private repo or a Modal volume.
- [ ] **Adapter template (P0-05).** A model-neutral adapter (capability check +
  generation) with a no-cost fixture backend, an expected-job export
  (case × repeat × seed) and generation receipts accepted by the Lite runner.
- [ ] **Environment (P0-06).** Extend `experiments/tables_4_7/evaluator_assets.lock.json`
  (GroundingDINO, Depth Anything, SAM 2.1, DINOv2 pinned) to RAFT, LPIPS/VGG, PTLFlow,
  OpenPIV and indoor depth; fill the 19 empty case locks; add a preflight.
- [ ] **Results (P0-07/09).** Per-metric status (`ok`, `unsupported`, `missing_candidate`,
  `invalid_candidate`, `not_applicable`, `unmeasurable`, `error`) with reasons; a
  coverage gate so partial runs cannot pose as full leaderboards; one repeat policy
  (LingBot's adaptive 5→30 vs. five runs elsewhere).
- [ ] **Acceptance (P0-10).** Make tests hermetic (no reads from `runs/` or the sibling
  paper checkout), add CI, then run the full workflow from the real HF download on a
  clean machine with one new model.

## Release order

1. Close scope and rights; freeze schemas (`actions.json`, reference manifest, result status).
2. Per dimension: actions → frozen references → normalization (sections 1–8).
3. Adapter template, environment lock, coverage gate.
4. Clean-machine acceptance run from the HF package; record commands and expected scores.
5. Ship only when every P0 item above is closed.

Related: open-model Table 4/7 status and calibration →
[`TABLE4_TABLE7_REMAINING_MODELS_TODO.md`](../../TABLE4_TABLE7_REMAINING_MODELS_TODO.md);
generated readiness report → `PRODUCTION_READINESS.md`
(`python tools/audit_production_readiness.py`).
