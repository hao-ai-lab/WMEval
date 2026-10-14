# WMEval Lite release — pending work by evaluation dimension

Updated 2026-10-09 · branch `lanxiang/release-cleanup` · paper: ready · project
page preview: https://snyhlx.github.io/wmeval-blog

**Release goal.** A new user downloads the dataset from Hugging Face, generates
their model's videos from the supplied conditions, and scores them against
frozen Unity references — without Unity, private repos, or maintainer help.
GPUs, CV checkpoints and a hosted judge (OpenAI) may still be required; each
must be declared, never hidden.

**Where each dimension stands** (verified 2026-10-09):

- **Actions.** Every dimension now runs through the Lite suite. The 14 Table 4/7
  cells are Lite cases (`<Scene>/scene_<variant>` for Table 7,
  `<Scene>/kinematics_<view>` for Table 4) carrying their conditioning frame,
  compiled action trajectory and Unity ground truth (`tools/build_table_lite_cases.py`);
  a camera trajectory is just an action trajectory whose target is `camera`.
  Causality (ball strike), fluid (bottle motion) and lighting (per-model orbit
  file) still lack a standardized, target-aware action.
- **Frozen references.** LPIPS re-judges Unity contact events per candidate;
  prompt consistency does not store the Unity reference scores and judges with a
  mutable hosted model. Causality's reference answers are already frozen.
- **Normalization.** The 14 Table 4/7 cases use `match_reference` (resample to the
  reference fps / resolution / frame count) and the 4 LiquidContainer cases their
  locked liquid token; the other 15 cases (including `beamdomino_native`, an
  identity token) still use the candidate as-is.

---

## How to use this file

- **Owners:** find your dimension section, fill in `Owner:`, follow *Run it for a
  new model* to reproduce your dimension end to end, then work the checklist. Tick
  `[x]` only with evidence (commit hash, output path, or command output).
- **Agents:** run the *Run it* steps exactly; each checklist task has **Do / Files /
  Check / Done when**. Run the **Check** after each change; stop and report if it
  fails. Follow the ground rules without exception.
- **Legend:** ✅ ready · ⚠️ partial · ❌ missing · — out of scope / not in the paper.

### Ground rules (all owners and agents)

1. **Never edit hash-locked inputs in place:** `experiments/tables_4_7/trajectories/*.yaml`,
   `experiments/tables_4_7/compiled/*`, and every file under `cases/*/*/reference/`.
   Re-author a case instead (see *Adding or re-locking a case*).
2. **Unsupported ≠ zero.** If a model cannot express a case's action target (e.g. a
   camera-only model asked to steer the car), record `unsupported` with a reason.
   Never substitute another control.
3. **Paper tables are frozen.** Five-run paper selections stay as published; 30-run
   robustness output goes to `reports/tables_4_7/final/` only. Reconstructed
   (orange) rows stay labeled as non-parity.
4. **Verify after every change:**
   ```bash
   python tools/run_wmeval_lite.py validate --suite eval_sets/wmeval_lite_v1
   python -m pytest -q tests
   ```
5. **No hidden paid calls.** Hosted-judge use needs explicit opt-in
   (`OPENAI_API_KEY`) and must record judge identity, requests and responses.

### Adding or re-locking a case

1. `mkdir -p cases/<Scene>/<variant>/reference/artifacts` and add the files.
2. Write `case.yaml` (copy a live case of the same kind).
3. `python tools/build_lite_case.py cases/<Scene>/<variant>` (checksums + lock).
4. Add the case id to the dimension's `cases:` list in `suite.yaml`.
5. `python tools/run_wmeval_lite.py validate` (+ `generate-rubrics` for prompt cases).

---

## Evaluating a new model — one path for every dimension

Every dimension, including the Table 4/7 reproduction, is evaluated the same way:
fixed conditions (image + prompt + action trajectory) → **five independent
generations per case** → `tools/run_wmeval_lite.py`.

```bash
# 1. Conditions for each case listed under the dimension in suite.yaml:
#      image   = cases/<Scene>/<variant>/reference/ref.png  (else frame 0 of the reference video)
#      prompt  = cases/<Scene>/<variant>/case.yaml : prompt
#      actions = cases/<Scene>/<variant>/reference/artifacts/
#                  trajectory.yaml + action.npz      (Table 4/7 cases: canonical semantic trajectory)
#                  actions.jsonl                     (Project Genie browser program, where supported)
#                  actions.txt / action.npy          (other cases)
#    Compile the actions onto your model's native controls. Register the backend in
#    experiments/tables_4_7/action_backends.yaml (native frames/fps, supported targets
#    + reasons, calibration); see tools/prepare_matrix_game_inputs.py and
#    tools/prepare_lingbot_table7_inputs.py. Never alter the trajectory itself.
# 2. Generate seeds 42 1 2 3 4 → runs/<model>/<Scene>/<variant>/<seed>.mp4
#    (cases with candidate_normalization: match_reference or liquid are resampled by
#     the runner; for the others, match the reference fps/resolution/duration yourself)
# 3. Evaluate (resumable; OPENAI_API_KEY for judge-based dimensions):
python tools/run_wmeval_lite.py run --suite eval_sets/wmeval_lite_v1 \
  --model-root runs/<model> --dimensions <dimension> \
  --output-dir reports/wmeval_lite/<model> --resume
# 4. Read reports/wmeval_lite/<model>/aggregate_summary.{json,csv}
# 5. Table 4/7 only — compare with the paper baseline:
python tools/export_lite_table_results.py --model "<name>" \
  --results reports/wmeval_lite/<model>/per_case_results.jsonl \
  --output reports/wmeval_lite/<model>/table_results.json
python tools/compare_table_results.py compare --allow-incomplete \
  --results reports/wmeval_lite/<model>/table_results.json
```

Backends whose accepted action timing differs from the commanded trajectory (e.g. a
browser model recording when a key press took effect) may pass per-run ground-truth
overrides (`eoc_unity_poses_npy`, …) in their candidate-manifest `params`; the baked
case is the nominal protocol. Worked generator: `tools/generate_waypoint15_candidates_modal.py`.

The earlier orchestration that produced the published Table 4/7 rows
(`tools/run_prepared_table_evaluations.py`, `tools/run_open_model_table_evaluations.py`,
`tools/prepare_project_genie_case_eval.py`) remains for provenance; new models use the
Lite cases. Their baked ground truth is identical to that orchestration's for the
full-turn and captured cells.

### Model roster

| Model | Interface | Generation code in this branch |
| --- | --- | --- |
| Unity (reference) | runtime args | `tools/run_unity_table_scenes.py` (Table 4/7); Lite references pre-baked |
| Project Genie | browser keys (rule-based, uncalibrated; left turn 51.47 °/s calibrated) | `tools/project_genie_agent.py`, `tools/run_project_genie_tables.py` |
| Matrix-Game 2.0 | keys + mouse, 129 frames @ 12 fps | `tools/prepare_matrix_game_inputs.py`, `tools/run_matrix_game2.py` (Table 4/7) |
| Matrix-Game 3.0 | keys + mouse, 177 frames @ 17 fps | `tools/prepare_matrix_game_inputs.py`, `tools/run_matrix_game3.py` (Table 7) |
| LingBot-World (Cam) | camera poses, 161 @ 16 fps | `tools/prepare_lingbot_table7_inputs.py`, `tools/run_lingbot_table7*.py` (Table 7) |
| LingBot-World (Act) | action-conditioned | ❌ none |
| HY-WorldPlay | — | ❌ none (licensing to confirm) |
| Waypoint-1.5 | keys + mouse, first person | `tools/generate_waypoint15_candidates_modal.py` (Lite, Modal) |

---

## 1. Environment object & character consistency — `environment_object_consistency`

**Owner:** _TBD_ · **Paper:** Table 7 · **Lite cases (16):** the 12 Table 7 cells
`{WanderingGarden, BilliardGame, MirrorRollingBall}/scene_{perturb_in, perturb_out, 360_in, 360_out}`
(camera trajectories `camera_*.yaml`, target `camera`) + BallRoller/ctrl,
RacingCar/prompt_third, ChessBoard/walk_third, WanderingGarden/scene_first (indoor)

**Run it for a new model:** the single path with `--dimensions environment_object_consistency`,
then `tools/export_lite_table_results.py` + `tools/compare_table_results.py` for Table 7.
Key files: `cases/<Scene>/scene_*/` (`ref.png`, `reference.mp4`,
`artifacts/{trajectory.yaml, action.npz, actions.jsonl, poses.npy, intrinsics.npy, object_transforms.json, table_cell.json}`),
`experiments/tables_4_7/action_backends.yaml`, `evaluation/modules/environment_object_consistency.py`,
`tools/build_table_lite_cases.py` (rebuilds the 12 cells from their Unity sources).

| Cases | Actions | Unity reference | Normalization |
| --- | --- | --- | --- |
| WanderingGarden scene_* × 4 | ✅ camera trajectories | ✅ GardenBridge orbit re-indexed on the commanded yaw (outward anchor reconstructed) | ✅ `match_reference` |
| BilliardGame scene_* × 4 | ✅ | ✅ dedicated Unity captures | ✅ `match_reference` |
| MirrorRollingBall scene_* × 4 | ✅ | ✅ Lite orbit re-indexed (photogrammetric depth caveat) | ✅ `match_reference` |
| BallRoller/ctrl, RacingCar/prompt_third | ✅ | ✅ bundled | ❌ none |
| ChessBoard/walk_third, WanderingGarden/scene_first | ❌ comment-only / no action file | ✅ bundled | ❌ none |

| Model | In paper | Status |
| --- | --- | --- |
| Project Genie | ✅ (reconstructed row) | ✅ 5-run results exist (produced by the earlier orchestration; full-turn and captured cells have identical baked ground truth) |
| LingBot-World (Cam) | ✅ | ✅ 5-run; adaptive 30-run expansion 100/360 |
| Matrix-Game 3.0 / 2.0 | ✅ | ✅ 5-run |
| Waypoint-1.5 | — | ⚠️ scored only on the legacy cases; regenerate on `scene_*` |

- [ ] **Re-score one existing model through the Lite cases** (e.g. Matrix-Game 3.0, whose
  schedules follow the commanded timing) and confirm the Table 7 means match the published
  five-run values. Check: `tools/export_lite_table_results.py` → `tools/compare_table_results.py compare`.
- [ ] **Distribution clearance** for the baked Unity renders and arrays (they now live in
  the case bundles); release them with the HF package.
- [ ] **Report realized camera motion** beside every score (LingBot realized −17.7/+41.4°
  for ±45° and +54.9/+277.9° for full turns). Done when: realized yaw is a column in the
  exported tables.
- [ ] **Anchor validity gate:** require ≥2 visible, confidently assigned anchors and a
  minimum paired-frame coverage for pairwise-distance MAE; otherwise `unmeasurable`.
- [ ] **Legacy cases:** give ChessBoard/walk_third and WanderingGarden/scene_first
  executable actions (or drop them) and switch BallRoller/ctrl and RacingCar/prompt_third
  to `match_reference`.
- [ ] **Publish per-scene self-match bounds** and the Mirror photogrammetric caveat.

## 2. Action controllability — `action_controllability`

**Owner:** _TBD_ · **Paper:** controllability table · **Lite cases:**
BallRoller/ctrl, BeamDomino/causal

**Run it for a new model:** the single path with `--dimensions action_controllability`. Key
files: `cases/BallRoller/ctrl/`, `cases/BeamDomino/causal/` (`actions.txt`,
`action.npy`, `reference.mp4`), `evaluation/modules/control_motion.py`,
`tools/evaluate_world_model_controllability.py`.

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| BallRoller/ctrl | ✅ keys + `action.npy` (target not declared) | ✅ video; flow computed at eval time | ❌ none |
| BeamDomino/causal | ⚠️ 2-channel ball force, no key program | ✅ video | ❌ identity (`beamdomino_native`) |

| Model | In paper | Status |
| --- | --- | --- |
| Project Genie | ✅ | ⚠️ BallRoller runnable via keys; BeamDomino force not expressible from this branch |
| LingBot-World (Act) | ✅ | ❌ no adapter |
| Waypoint-1.5 | ✅ (GT-RAFT only) | ⚠️ scored; WASD moves the ego camera, not the ball |

- [ ] **Declare the action target** (`cue_ball` / `ball`) in each case's `actions.json`;
  models without an object channel report `unsupported`. Done when: Waypoint rows show
  `unsupported`, not a 755 px EPE.
- [ ] **BeamDomino adapter or explicit `unsupported`** per backend in `action_backends.yaml`.
- [ ] **Bake the reference flow** into the case; **vendor PTLFlow** and pin its checkpoint.
- [ ] **Resample both cases** to the reference fps/resolution; headline relative EPE.
- [ ] **Add a MirrorRollingBall controllability case** (in the paper, not in Lite).

## 3. Kinematics & mechanics — `kinematics_mechanics`

**Owner:** _TBD_ · **Paper:** Table 4 · **Lite cases (3):** RacingCar/kinematics_third
(`racing_car_drive.yaml`, steer/throttle/brake, target `ego_vehicle`),
BilliardGame/kinematics_first (`billiard_first_person_ball_control.yaml`, forward/lateral,
target `cue_ball`), BallRoller/ctrl

**Run it for a new model:** the single path with `--dimensions kinematics_mechanics`,
then `tools/export_lite_table_results.py` + `tools/compare_table_results.py` for Table 4.
Key files: `cases/RacingCar/kinematics_third/`, `cases/BilliardGame/kinematics_first/`
(`artifacts/{trajectory.yaml, action.npz, actions.jsonl, poses.npy, intrinsics.npy,
object_transforms.json, object_masses.json, plane_calibration.json}`),
`evaluation/modules/kinematics_mechanics.py`.

| Case | Actions | Unity reference | Normalization |
| --- | --- | --- | --- |
| RacingCar/kinematics_third | ✅ drive trajectory | ✅ Unity capture, masses, plane calibration | ✅ `match_reference` |
| BilliardGame/kinematics_first | ✅ ball-control trajectory | ✅ Unity capture, masses, plane calibration | ✅ `match_reference` |
| BallRoller/ctrl | ✅ keys + `action.npy` | ✅ bundled | ❌ none |
| Excluded: RacingCar first, BilliardGame third | — | — | — |

| Model | In paper | Status |
| --- | --- | --- |
| Project Genie | ✅ (reconstructed row) | ✅ 5-run, both cells |
| Matrix-Game 2.0 | ✅ | ⚠️ 5-run generated; car/ball target control not validated (thresholded WASD) |
| LingBot-World (Cam) | — | — unsupported (camera input only) |
| Matrix-Game 3.0 | — | — out of scope (Table 7 only) |
| Waypoint-1.5 | — | ⚠️ scored only on legacy cases; regenerate on `kinematics_*` |

- [ ] **Re-score one existing model through the Lite cases** and confirm the Table 4
  means match the published values.
- [ ] **Validate target control per backend:** reject outputs where the camera or an
  avatar moved instead of the car/ball; record a target-success rate per run (Matrix-Game
  2.0 first).
- [ ] **Units and floors:** held-out reprojection checks, exact-match as diagnostic,
  absolute-error floor for near-zero motion.

## 4. Fluid mechanics — `fluid_mechanics`

**Owner:** _TBD_ · **Paper:** LiquidContainer table · **Lite cases:** 4 profiles
(rest, step_hold, impulse_decay, chirp_strafe)

**Run it for a new model:** the single path with `--dimensions fluid_mechanics` (the runner
resamples to 832×464 @ 16 fps). Key files: `cases/LiquidContainer/*/reference/artifacts/`
(`action.npy` forward/right/up, `liquid_mask.npz`, `bottle_mask.npz`,
`raft_pooled_flow.npz`), `evaluation/modules/liquid_*.py`, `slosh_spectrum.py`,
`tools/evaluate_world_model_liquid.py`.

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| all 4 | ⚠️ bottle motion; no key/browser adapter | ✅ masks + pooled flow baked | ✅ resampled |

| Model | In paper | Status |
| --- | --- | --- |
| Project Genie, LingBot-World, Matrix-Game 2.0, Matrix-Game 3.0, HY-WorldPlay | ✅ | ❌ no bottle adapter in this branch; paper rollouts produced elsewhere |
| Waypoint-1.5 | — | not run |

- [ ] **Define the bottle action target** (`bottle`, m/s², world frame) in `actions.json`
  and each backend's support in `action_backends.yaml`.
- [ ] **Recover or regenerate the 5 paper models' rollouts** with action receipts.
- [ ] **Pin SAM 2 and RAFT checkpoints**; keep the per-profile multi-axis panel canonical.

## 5. Physical causality — `physical_causality`

**Owner:** _TBD_ · **Paper:** BeamDomino causality table · **Lite cases:**
BeamDomino causal / causal_glancing / causal_direct_strike

**Run it for a new model:** the single path with `--dimensions physical_causality`
(`OPENAI_API_KEY`; 5 judge passes per candidate). Key files:
`cases/BeamDomino/*/reference/artifacts/` (`action.npy`, `qa_dataset.json`,
`judge_answers.json`), `evaluation/modules/causal_step*.py`,
`tools/evaluate_world_model_causal_step.py`.

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| all 3 | ⚠️ 2-channel ball force; no key/browser program | ✅ frozen QA + `judge_answers.json` (5 passes) | ❌ identity (`beamdomino_native`) |

| Model | In paper | Status |
| --- | --- | --- |
| Project Genie | ✅ | ⚠️ force action not expressible from this branch |
| LingBot-World (Cam) | ✅ | ⚠️ camera-only interface cannot strike the ball — confirm the executed condition |
| Waypoint-1.5 | — | not run |

- [ ] **Define the strike action** (target `ball`, force units) and backend support.
- [ ] **Audit the LingBot (Cam) paper row** and label it if it used a different protocol.
- [ ] **Keep `causal_step_judge_repeats: 5`** and record every candidate judge pass;
  re-baseline the frozen reference answers when the judge or prompt changes.

## 6. Lighting & shading — `lighting_shading`

**Owner:** _TBD_ · **Paper:** SunRisesAndSets table · **Lite cases:**
SunRisesAndSets/lighting (first person, fixed sun, walking orbit; reference-free)

**Run it for a new model:** the single path with `--dimensions lighting_shading`. Key files:
`cases/SunRisesAndSets/lighting/reference/` (`ref.png`, `actions.txt`,
`actions_per_model.json`), `evaluation/modules/shadow_consistency.py`,
`tools/analytic_lighting_floor.py`.

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| SunRisesAndSets/lighting | ⚠️ per-model file; orbit reconstructed | — reference-free (analytic floor) | ❌ none (jitter is per frame) |

| Model | In paper | Status |
| --- | --- | --- |
| Project Genie, LingBot-World, Matrix-Game 3.0, HY-WorldPlay, Waypoint-1.5 | ✅ | ✅ reproduces the paper (rollouts on a Modal volume) |
| Matrix-Game 2.0 | ✅ | ⚠️ reproduces the paper except Scene Retention |

- [ ] **One semantic orbit trajectory** compiled per backend, replacing `actions_per_model.json`.
- [ ] **Normalize fps** (or report jitter per second).
- [ ] **Bundle or pin the six rollouts** in the HF baseline set.
- [ ] After release (optional): `ChessBoard/lighting`, moving-sun `lighting_sunsweep`.

## 7. Prompt-scene consistency — `prompt_scene_consistency`

**Owner:** Rita Wu · **Paper:** prompt-consistency table · **Lite cases:**
ChessBoard, MirrorRollingBall, RacingCar × prompt_first / prompt_third

**Run it for a new model:** the single path with `--dimensions prompt_scene_consistency`
(`OPENAI_API_KEY`). Key files: `cases/*/prompt_*/case.yaml` (prompt),
`reference/ref.png`, `reference/artifacts/rubrics/`,
`tools/evaluate_world_model_vlm_consistency.py`.

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| all 6 | ✅ 10 s camera rotation | ✅ rubrics and Unity reference scores frozen in the bundle; hosted `gpt-5.4` judge, snapshot recorded per call | — low risk |

| Model | In paper | Status |
| --- | --- | --- |
| Project Genie | ✅ | ✅ 5-run reproduced through Lite; judge snapshot recorded per call |
| LingBot-World (Cam), Matrix-Game 3.0 | ✅ | ✅ 5-run reproduced; Lite generation path (`tools/prepare_prompt_case_inputs.py`) |
| HY-WorldPlay | ✅ | ⚠️ 5-run reproduced from the existing rollouts; no Lite generation path |
| Waypoint-1.5 | ✅ | ✅ 30/30 rows |

- [x] **Freeze the Unity reference scores** per case (judge model + prompt version +
  sampled frames). Done when: reference rows load from the bundle. Evidence:
  `tools/freeze_prompt_reference_scores.py --passes 5` →
  `cases/*/prompt_*/reference/artifacts/reference_scores.json` (5 passes each,
  `gpt-5.4-2026-03-05`). Rows report `overall_score_unity` / `env_score_unity` /
  `obj_score_unity` from that file and mark it `stale` instead when the judge settings or
  the Unity video change. Unity overall, first / third: ChessBoard 1.000 / 1.000,
  MirrorRollingBall 0.953 / 1.000, RacingCar 0.840 / 0.837.
- [x] **Record judge identity, requests and responses.** Every row's `full_report` has
  `judge` (settings, rubric and first-frame hashes) and `judge_calls` (resolved snapshot,
  response id, usage, raw output, request text and per-image hashes).
- [x] **Ship the ρ = 0.90 human-alignment fixture** as a regression check. Evidence:
  `human_alignment/prompt_scene_consistency.json` (human consensus per case and model:
  3 raters, 126 videos; the paper's judge scores reproduce ρ 0.90 / τ 0.80 over the 5
  models and 0.80 / 0.667 over the 4 generative ones).
  `tools/check_prompt_human_alignment.py --results <per_case_results.jsonl>` recomputes it
  for any run and exits non-zero below the paper value.
- [x] **Add Lite generation paths** for LingBot (Cam) and Matrix-Game 3.0. Evidence: the
  cases' 10 s left turn is the canonical `trajectories/prompt_camera_left_turn.yaml`
  (`prompt.camera.left_turn_v1`, target `camera`, 40 deg/s as in the Unity reference),
  baked into every case (`trajectory.yaml`, `action.npz`, `actions.jsonl`).
  `tools/prepare_prompt_case_inputs.py` compiles it to LingBot poses (the Unity camera path,
  161 frames at 16 fps) and to the Matrix-Game 3.0 key recipe its rollouts were generated
  with (177 frames at 17 fps: `j` every frame, plus `d` in third person to orbit the
  character; seeds 0–4), and writes the five-repeat queues for
  `run_lingbot_table7.py --queue` / `run_matrix_game3.py --queue` plus the Lite candidate
  manifest.
- [x] **Reproduced through the Lite path** (case bundle → 5 seeds →
  `run_wmeval_lite.py --dimensions prompt_scene_consistency`): Project Genie, LingBot (Cam),
  Matrix-Game 3.0 and HY-WorldPlay, 120/120 rows. Against the paper's values, per-cell mean
  |Δ| is 0.023 (max 0.067) and the model ranking is unchanged
  (Unity 0.938 > Genie 0.905 > LingBot 0.891 > HY 0.792 > Matrix-Game 3.0 0.745);
  `tools/check_prompt_human_alignment.py` gives ρ 0.90 / 0.80, as in the paper.

## 8. Perceptual similarity — `lpips`

**Owner:** _TBD_ · **Paper:** causal-LPIPS table · **Lite cases:**
MirrorRollingBall/lpips_collision, BeamDomino/causal, LiquidContainer × 4

**Run it for a new model:** the single path with `--dimensions lpips` (`OPENAI_API_KEY` for
event detection). Key files: `cases/MirrorRollingBall/lpips_collision/`,
`evaluation/modules/lpips*.py`, `tools/evaluate_world_model_lpips.py`.

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| MirrorRollingBall/lpips_collision | ✅ keys + `action.npy` | ⚠️ contact events re-judged per candidate | ❌ none |
| BeamDomino/causal | ⚠️ ball force (§5) | ⚠️ re-judged (0/1/3/0/0 contacts on one reference) | ❌ identity |
| LiquidContainer × 4 | ⚠️ bottle motion (§4) | ⚠️ re-judged | ✅ resampled |

| Model | In paper | Status |
| --- | --- | --- |
| Project Genie | ✅ | ⚠️ Mirror runnable via keys |
| LingBot-World (Act) | ✅ | ❌ no adapter |
| Waypoint-1.5 | — | ⚠️ BeamDomino fell back to temporal LPIPS |

- [ ] **Freeze reference contact events** once per case. Done when: every candidate row
  reuses the same reference events.
- [ ] **Flag `causal` vs `temporal_only` per row**; never average them in one headline.
- [ ] **Resample** MirrorRollingBall/lpips_collision and BeamDomino/causal.

---

## Release owner — cross-cutting items

- [ ] **Scope and rights (P0-01).** Freeze the case roster (33 Lite case bundles, including the 14 Table 4/7
  cells) and repeat policy; clear redistribution of Unity renders, quantities and
  annotations; add LICENSE files, attribution, citation, dataset card; confirm HY-WorldPlay.
- [ ] **HF package (P0-02).** Export → download → verify into a clean directory with
  per-file hashes and dataset/evaluator/schema versions; nothing may resolve into `runs/`,
  a private repo or a Modal volume.
- [ ] **Adapter template (P0-05).** A model-neutral adapter (capability check +
  generation) driven by `action_backends.yaml`, a no-cost fixture backend, an expected-job
  export (case × repeat × seed) and receipts accepted by the Lite runner.
- [ ] **Environment (P0-06).** Extend `experiments/tables_4_7/evaluator_assets.lock.json`
  (GroundingDINO, Depth Anything, SAM 2.1, DINOv2 pinned) to RAFT, LPIPS/VGG, PTLFlow,
  OpenPIV and indoor depth; fill the 19 empty case locks; add a preflight.
- [ ] **Results (P0-07/09).** Per-metric status (`ok`, `unsupported`, `missing_candidate`,
  `invalid_candidate`, `not_applicable`, `unmeasurable`, `error`); a coverage gate; one
  repeat policy (LingBot's adaptive 5→30 vs. five runs elsewhere).
- [ ] **Acceptance (P0-10).** Hermetic tests (no reads from `runs/` or the sibling paper
  checkout), CI, and a clean-machine run of the Lite path from the HF download with one new
  model.

## Release order

1. Scope and rights; freeze schemas (`actions.json`, reference manifest, result status).
2. Re-score one model through the Table 4/7 Lite cases (§1, §3); standardize the remaining actions (§2, §4–§6).
3. Freeze reference judgments (§7, §8); lock normalization everywhere.
4. Adapter template, environment lock, coverage gate.
5. Clean-machine acceptance from the HF package; ship when every item above is closed.

Related: open-model Table 4/7 status and calibration →
[`TABLE4_TABLE7_REMAINING_MODELS_TODO.md`](../../TABLE4_TABLE7_REMAINING_MODELS_TODO.md);
generated readiness report → `PRODUCTION_READINESS.md`
(`python tools/audit_production_readiness.py`).
