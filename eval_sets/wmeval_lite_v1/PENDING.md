# WMEval Lite release — pending work by evaluation dimension

Updated 2026-10-09 · branch `lanxiang/release-cleanup` · paper: ready · project
page preview: https://snyhlx.github.io/wmeval-blog

**Release goal.** A new user downloads the dataset from Hugging Face, generates
their model's videos from the supplied conditions, and scores them against
frozen Unity references — without Unity, private repos, or maintainer help.
GPUs, CV checkpoints and a hosted judge (OpenAI) may still be required; each
must be declared, never hidden.

**Where each dimension stands** (verified 2026-10-09):

- **Actions.** Object consistency and kinematics already run from standardized
  action trajectories: the 14 Table 4/7 cells (`experiments/tables_4_7/`), where a
  camera trajectory is just an action trajectory whose target is `camera`. Their gap
  is packaging (ground truth in a private ZIP + `runs/`, model lists hard-coded in
  two orchestration scripts). Causality (ball strike), fluid (bottle motion) and
  lighting (per-model orbit file) still lack a standardized, target-aware action.
- **Frozen references.** LPIPS re-judges Unity contact events per candidate;
  prompt consistency does not store the Unity reference scores and judges with a
  mutable hosted model. Causality's reference answers are already frozen.
- **Normalization.** The Lite runner resamples only the 4 LiquidContainer cases;
  the other 15 (including `beamdomino_native`, which is an identity token) use the
  candidate as-is. The Table 4/7 path resamples to 150 frames @ 15 fps itself.

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

## Evaluating a new model — the two paths

Every dimension uses one of two paths. Both start from fixed conditions (image +
prompt + action trajectory), produce **five independent generations per case**, and
end in `tools/run_wmeval_lite.py`.

### Path A — Lite cases (controllability, fluid, causality, lighting, prompt, LPIPS)

```bash
# 1. Conditions for each case listed under the dimension in suite.yaml:
#      image   = cases/<Scene>/<variant>/reference/ref.png  (else frame 0 of reference/reference.mp4)
#      prompt  = cases/<Scene>/<variant>/case.yaml : prompt
#      actions = cases/<Scene>/<variant>/reference/artifacts/actions.txt (+ action.npy)
# 2. Generate seeds 42 1 2 3 4 → runs/<model>/<Scene>/<variant>/<seed>.mp4
#    Match the reference fps / resolution / duration yourself unless the case's
#    candidate_normalization token resamples it (today: LiquidContainer only).
# 3. Evaluate (resumable; OPENAI_API_KEY for judge-based dimensions):
python tools/run_wmeval_lite.py run --suite eval_sets/wmeval_lite_v1 \
  --model-root runs/<model> --dimensions <dimension> \
  --output-dir reports/wmeval_lite/<model> --resume
# 4. Read reports/wmeval_lite/<model>/aggregate_summary.{json,csv}
```

Worked generator: `tools/generate_waypoint15_candidates_modal.py` (Waypoint-1.5).

### Path B — Table 4/7 cells (object consistency, kinematics)

```bash
# 0. Install + verify the reference package (experiments/tables_4_7/REFERENCE_PACKAGE.md)
python tools/build_table_reference_package.py verify
# 1. Conditions per active cell in experiments/tables_4_7/cases.yaml:
#      image   = first_frame
#      actions = trajectory  (experiments/tables_4_7/trajectories/*.yaml,
#                compiled to experiments/tables_4_7/compiled/*.action.npz)
# 2. Register your backend in experiments/tables_4_7/action_backends.yaml
#    (native frames/fps, supported targets + reasons, calibration) and compile the
#    trajectory onto it — see tools/prepare_matrix_game_inputs.py and
#    tools/prepare_lingbot_table7_inputs.py. Calibrate camera gain/latency on
#    calibration_samples/; never alter the trajectory.
# 3. Generate run0..run4 per cell → runs/<model>/<table>/<Scene>/<variant>/runN/
#    with a receipt.json (status, repeat_id, first_frame_sha256, actions_sha256, output hash).
# 4. Standardize (150 frames @ 15 fps) and attach the cell's Unity ground truth as
#    per-run params on a Lite evaluator template, e.g. a Garden full turn:
python tools/prepare_project_genie_case_eval.py --model <model> \
  --run-root runs/<model>/table7/WanderingGarden/360_in \
  --lite-case-id WanderingGarden/scene_first \
  --camera-trajectory-mode full_turn --reference-start-frame 0 --reference-target-yaw-deg -360 \
  --output-dir runs/<model>/table7/WanderingGarden/360_in/evaluation_prepared
#    (BilliardGame cells and Table 4 need extra Unity overrides: today only
#     tools/run_prepared_table_evaluations.py [Genie] and
#     tools/run_open_model_table_evaluations.py [MODELS list] provide them.)
# 5. Evaluate the prepared manifest:
python tools/run_wmeval_lite.py run --suite eval_sets/wmeval_lite_v1 \
  --candidate-manifest runs/<model>/table7/WanderingGarden/360_in/evaluation_prepared/candidate_manifest.jsonl \
  --dimensions environment_object_consistency --output-dir reports/tables_4_7/<model>/WanderingGarden__360_in
# 6. Build tables and compare with the paper baseline:
python tools/build_table7_rerun_results.py      # Table 7
python tools/build_table4_calibrated_results.py # Table 4
python tools/compare_table_results.py --baseline experiments/tables_4_7/paper_baseline.yaml
```

Template mapping (Table cell → Lite evaluator case): WanderingGarden →
`WanderingGarden/scene_first`; BilliardGame → `BilliardGame/prompt_first`;
MirrorRollingBall → `MirrorRollingBall/prompt_third`; RacingCar →
`RacingCar/prompt_third`. Running those Lite cases *without* Path B uses their
legacy bundled actions and references, which are not the standardized protocol.

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

**Owner:** _TBD_ · **Paper:** Table 7 · **Protocol:** Path B, 12 cells —
WanderingGarden / BilliardGame / MirrorRollingBall × perturb_in, perturb_out,
360_in, 360_out (camera trajectories `camera_*.yaml`, target `camera`)

**Run it for a new model:** Path B with `--dimensions environment_object_consistency`.
Key files: `experiments/tables_4_7/cases.yaml`, `trajectories/camera_*.yaml`,
`action_backends.yaml`, `evaluator_protocol.yaml`, `evaluator_assets.lock.json`,
`evaluation/modules/environment_object_consistency.py`.

| Cell group | Actions | Unity reference | Normalization |
| --- | --- | --- | --- |
| WanderingGarden × 4 | ✅ camera trajectories | ⚠️ GardenBridge capture in `runs/unity/` (ignored); outward anchor reconstructed | ✅ Path B (150 f @ 15 fps) |
| BilliardGame × 4 | ✅ | ⚠️ Unity captures in `runs/tables_4_7/` (ignored) | ✅ Path B |
| MirrorRollingBall × 4 | ✅ | ⚠️ Lite poses + derived outward frame; photogrammetric depth | ✅ Path B |
| Lite-only: BallRoller/ctrl, ChessBoard/walk_third | ✅ / ❌ comment-only | ✅ bundled | ❌ none |

| Model | In paper | Status |
| --- | --- | --- |
| Project Genie | ✅ (reconstructed row) | ✅ 5-run, 12 cells |
| LingBot-World (Cam) | ✅ | ✅ 5-run; adaptive 30-run expansion 100/360 |
| Matrix-Game 3.0 | ✅ | ✅ 5-run |
| Matrix-Game 2.0 | ✅ | ✅ 5-run |
| Waypoint-1.5 | — | ⚠️ scored only on the legacy Lite cases (not the Table 7 protocol) |

- [ ] **Package the 12 cells as self-contained HF cases.** Do: bake each cell's
  conditioning image, compiled trajectory, Unity poses/intrinsics/object transforms and
  standardized reference frames into a case bundle, so Path B needs nothing from `runs/`
  or the private ZIP. Check: `python tools/build_table_reference_package.py verify`.
  Done when: Path B runs from a clean clone + HF download.
- [ ] **Make Path B model-agnostic.** Do: move the BilliardGame Unity overrides out of
  `run_prepared_table_evaluations.py` / `run_open_model_table_evaluations.py` into case
  data, and drop the hard-coded `MODELS` list. Done when: `prepare_project_genie_case_eval.py
  --model <any>` prepares all 12 cells.
- [ ] **Report realized camera motion** beside every score (LingBot realized −17.7/+41.4°
  for ±45° and +54.9/+277.9° for full turns). Done when: realized yaw is a column in the
  exported tables.
- [ ] **Anchor validity gate:** require ≥2 visible, confidently assigned anchors and a
  minimum paired-frame coverage for pairwise-distance MAE; otherwise `unmeasurable`.
- [ ] **Resolve the Lite-only cases:** give ChessBoard/walk_third an executable
  `actions.json` (or drop it) and lock normalization for BallRoller/ctrl.
- [ ] **Publish per-scene self-match bounds** and the Mirror photogrammetric caveat.

## 2. Action controllability — `action_controllability`

**Owner:** _TBD_ · **Paper:** controllability table · **Protocol:** Path A —
BallRoller/ctrl, BeamDomino/causal

**Run it for a new model:** Path A with `--dimensions action_controllability`. Key
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

**Owner:** _TBD_ · **Paper:** Table 4 · **Protocol:** Path B, 2 active cells —
`table4/RacingCar/third` (`racing_car_drive.yaml`, steer/throttle/brake, target
`ego_vehicle`) and `table4/BilliardGame/first` (`billiard_first_person_ball_control.yaml`,
forward/lateral, target `cue_ball`)

**Run it for a new model:** Path B with `--dimensions kinematics_mechanics`; Table 4
preparation needs the Unity overrides in `tools/run_open_model_table_evaluations.py`
(add the model to `MODELS`) until they move into case data. Key files:
`experiments/tables_4_7/cases.yaml`, `trajectories/racing_car_drive.yaml`,
`trajectories/billiard_first_person_ball_control.yaml`, `action_backends.yaml`,
`evaluation/modules/kinematics_mechanics.py`, `tools/build_table4_calibrated_results.py`.

| Cell | Actions | Unity reference | Normalization |
| --- | --- | --- | --- |
| RacingCar third | ✅ drive trajectory | ⚠️ capture + `gt.json` in `runs/` (ignored) | ✅ Path B |
| BilliardGame first | ✅ ball-control trajectory | ⚠️ capture in `runs/` (ignored) | ✅ Path B |
| Excluded: RacingCar first, BilliardGame third | — | — | — |
| Lite-only: BallRoller/ctrl | ✅ keys | ✅ bundled | ❌ none |

| Model | In paper | Status |
| --- | --- | --- |
| Project Genie | ✅ (reconstructed row) | ✅ 5-run, both cells |
| Matrix-Game 2.0 | ✅ | ⚠️ 5-run generated; car/ball target control not validated (thresholded WASD) |
| LingBot-World (Cam) | — | — unsupported (camera input only) |
| Matrix-Game 3.0 | — | — out of scope (Table 7 only) |
| Waypoint-1.5 | — | ⚠️ scored only on legacy Lite cases |

- [ ] **Package both cells as self-contained HF cases** (as in §1) and move the Racing /
  Billiard Unity overrides into case data. Done when: Path B runs for a new model with no
  edits to orchestration code.
- [ ] **Validate target control per backend:** reject outputs where the camera or an
  avatar moved instead of the car/ball; record a target-success rate per run (Matrix-Game
  2.0 first).
- [ ] **Retire the legacy Lite kinematics rows** (`RacingCar/prompt_third` is a camera
  rotation around a parked car; `BilliardGame/prompt_first` has no action) or point them
  at the Table 4 trajectories.
- [ ] **Units and floors:** held-out reprojection checks, exact-match as diagnostic,
  absolute-error floor for near-zero motion.

## 4. Fluid mechanics — `fluid_mechanics`

**Owner:** _TBD_ · **Paper:** LiquidContainer table · **Protocol:** Path A — 4 profiles
(rest, step_hold, impulse_decay, chirp_strafe)

**Run it for a new model:** Path A with `--dimensions fluid_mechanics` (the runner
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

**Owner:** _TBD_ · **Paper:** BeamDomino causality table · **Protocol:** Path A —
BeamDomino causal / causal_glancing / causal_direct_strike

**Run it for a new model:** Path A with `--dimensions physical_causality`
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

**Owner:** _TBD_ · **Paper:** SunRisesAndSets table · **Protocol:** Path A —
SunRisesAndSets/lighting (first person, fixed sun, walking orbit; reference-free)

**Run it for a new model:** Path A with `--dimensions lighting_shading`. Key files:
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

**Owner:** _TBD_ · **Paper:** prompt-consistency table · **Protocol:** Path A —
ChessBoard, MirrorRollingBall, RacingCar × prompt_first / prompt_third

**Run it for a new model:** Path A with `--dimensions prompt_scene_consistency`
(`OPENAI_API_KEY`). Key files: `cases/*/prompt_*/case.yaml` (prompt),
`reference/ref.png`, `reference/artifacts/rubrics/`,
`tools/evaluate_world_model_vlm_consistency.py`.

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| all 6 | ✅ 10 s camera rotation | ⚠️ rubrics frozen; Unity reference scores not stored; hosted `gpt-5.4` judge | — low risk |

| Model | In paper | Status |
| --- | --- | --- |
| Project Genie | ✅ | ⚠️ runnable; needs judge snapshot for reruns |
| LingBot-World (Cam), Matrix-Game 3.0 | ✅ | ❌ no Lite generation path (camera rotation is supported by both) |
| HY-WorldPlay | ✅ | ❌ |
| Waypoint-1.5 | ✅ | ✅ 30/30 rows |

- [ ] **Freeze the Unity reference scores** per case (judge model + prompt version +
  sampled frames). Done when: reference rows load from the bundle.
- [ ] **Record judge identity, requests and responses**; ship the ρ = 0.90 human-alignment
  fixture as a regression check.
- [ ] **Add Lite generation paths** for LingBot (Cam) and Matrix-Game 3.0.

## 8. Perceptual similarity — `lpips`

**Owner:** _TBD_ · **Paper:** causal-LPIPS table · **Protocol:** Path A —
MirrorRollingBall/lpips_collision, BeamDomino/causal, LiquidContainer × 4

**Run it for a new model:** Path A with `--dimensions lpips` (`OPENAI_API_KEY` for
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

- [ ] **Scope and rights (P0-01).** Freeze the case roster (Lite cases + the 14 Table 4/7
  cells) and repeat policy; clear redistribution of Unity renders, quantities and
  annotations; add LICENSE files, attribution, citation, dataset card; confirm HY-WorldPlay.
- [ ] **HF package (P0-02).** Export → download → verify into a clean directory with
  per-file hashes and dataset/evaluator/schema versions; nothing may resolve into `runs/`,
  a private repo or a Modal volume.
- [ ] **Adapter template (P0-05).** A model-neutral adapter (capability check +
  generation) driven by `action_backends.yaml`, a no-cost fixture backend, an expected-job
  export (case × repeat × seed) and receipts accepted by both paths.
- [ ] **Environment (P0-06).** Extend `experiments/tables_4_7/evaluator_assets.lock.json`
  (GroundingDINO, Depth Anything, SAM 2.1, DINOv2 pinned) to RAFT, LPIPS/VGG, PTLFlow,
  OpenPIV and indoor depth; fill the 19 empty case locks; add a preflight.
- [ ] **Results (P0-07/09).** Per-metric status (`ok`, `unsupported`, `missing_candidate`,
  `invalid_candidate`, `not_applicable`, `unmeasurable`, `error`); a coverage gate; one
  repeat policy (LingBot's adaptive 5→30 vs. five runs elsewhere).
- [ ] **Acceptance (P0-10).** Hermetic tests (no reads from `runs/` or the sibling paper
  checkout), CI, and a clean-machine run of both paths from the HF download with one new
  model.

## Release order

1. Scope and rights; freeze schemas (`actions.json`, reference manifest, result status).
2. Package the Table 4/7 cells (§1, §3); standardize the remaining actions (§2, §4–§6).
3. Freeze reference judgments (§7, §8); lock normalization everywhere.
4. Adapter template, environment lock, coverage gate.
5. Clean-machine acceptance from the HF package; ship when every item above is closed.

Related: open-model Table 4/7 status and calibration →
[`TABLE4_TABLE7_REMAINING_MODELS_TODO.md`](../../TABLE4_TABLE7_REMAINING_MODELS_TODO.md);
generated readiness report → `PRODUCTION_READINESS.md`
(`python tools/audit_production_readiness.py`).
