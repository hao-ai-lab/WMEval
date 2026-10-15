# WMEval Lite release — pending work by evaluation dimension

Updated 2026-10-15 (after #33–#36) · branch `lanxiang/release-cleanup` · paper: ready · project
page preview: https://snyhlx.github.io/wmeval-blog

**Release goal.** A new user downloads the dataset from Hugging Face, generates
their model's videos from the supplied conditions, and scores them against
frozen Unity references — without Unity, private repos, or maintainer help.
GPUs, CV checkpoints and a hosted judge (OpenAI) may still be required; each
must be declared, never hidden.

**Where each dimension stands** (verified 2026-10-15, after #33–#36):

- **Actions.** 30 of the 32 live cases carry a standardized, target-aware action
  (`trajectory.yaml` / `action.npz` / `actions.jsonl`; target declared in
  `action_backends.yaml`, and in `actions.json` where backends must be gated):
  Table 4/7 cells (`camera`, `ego_vehicle`, `cue_ball`), lighting orbit (`camera`, #33),
  causality (`rolling_ball`, exact) and fluid (`liquid_container`, reconstructed) (#34),
  ball control (`ball`, #35), prompt left turn (`camera`, #36). Gaps: ChessBoard/walk_third
  and WanderingGarden/scene_first (§1) have no executable action; prompt consistency's turn is
  nominal, since generation still replays each model's original camera motion (§7).
- **Frozen references.** Causality answers, prompt Unity scores (#36) and LPIPS contact
  events for BallRoller / BeamDomino / MirrorRollingBall (#35) are frozen in the bundles.
  LiquidContainer contact events are not.
- **Normalization.** 14 Table 4/7 cases `match_reference`, 4 liquid and 1 lighting token,
  3 `beamdomino_native` (identity). 10 cases use the candidate as-is: BallRoller/ctrl and
  MirrorRollingBall/lpips_collision by decision (paper parity, §2), the 6 prompt cases (judge
  input, low risk), and the 2 legacy §1 cases.
- **Unsupported is reported, not scored.** Backends that cannot express a case's target
  return `unsupported` with the registry reason; aggregates carry `num_unsupported`.

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
| BallRoller/ctrl | ✅ ball trajectory, target `ball` (§2) | ✅ bundled | ❌ none |
| RacingCar/prompt_third | ⚠️ left-turn trajectory, target `camera` (§7; nominal) | ✅ bundled | ❌ none |
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
BallRoller/ctrl, BeamDomino/causal, MirrorRollingBall/lpips_collision

**Run it for a new model:** the single path with `--dimensions action_controllability`. Key
files: `cases/<Scene>/<variant>/reference/artifacts/` (`trajectory.yaml`, `action.npz`,
`actions.jsonl`, `actions.json` = declared target), `evaluation/modules/control_motion.py`,
`experiments/tables_4_7/action_backends.yaml`. A model registered with a `real_fps` there
(e.g. `lingbot_world_act`) needs no `model.yaml`.

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| BallRoller/ctrl | ✅ `controllability.ball_roller.collision_path_v1`, target `ball` | ✅ video; flow computed at eval time | `none` (kept, see below) |
| BeamDomino/causal | ✅ `wmeval_lite.beam_domino.causal.ball_force_v1`, target `rolling_ball` (the §5 trajectory, exact to `action.npy`) | ✅ video (726-frame, replaced in #32) | `beamdomino_native` (identity) |
| MirrorRollingBall/lpips_collision | ✅ `controllability.mirror_rolling_ball.axis_cycle_v1`, target `ball` | ✅ video | `none` |

| Model | In paper | Status |
| --- | --- | --- |
| Project Genie | ✅ | ✅ W/A/S/D program compiled from each trajectory (`actions.jsonl`) |
| LingBot-World (Act) | ✅ | ⚠️ scored from existing rollouts; no generation adapter in this repo |
| Waypoint-1.5 | ✅ (GT-RAFT only) | ✅ reported `unsupported` for target `ball` (no object channel) |

- [x] **Declare the action target** in each case's `actions.json`; models without an object
  channel report `unsupported`. Evidence: `728228b`, `2220da2`; a Waypoint candidate on
  `BallRoller/ctrl` through `run_wmeval_lite.py` yields `status: unsupported` with the
  recorded reason and empty metrics; aggregates carry `num_unsupported`, the CSV prints
  `unsupported`.
- [x] **BeamDomino adapter or explicit `unsupported`** per backend in `action_backends.yaml`
  (new `ball` target classified for every backend). Evidence: `2220da2`. BeamDomino/causal
  uses the §5 trajectory (`ball_force_v1`, target `rolling_ball`, 122 steps, bit-for-bit
  equal to `action.npy`) rather than a second controllability copy; LingBot (Act) supports
  `rolling_ball`, Waypoint-1.5 reports it `unsupported`.
- [x] **Bake the reference flow** (decided against) / **vendor PTLFlow** (done). RAFT's checkpoint is now
  pinned (explicit weights enum + SHA-256 in `evaluator_assets.lock.json`, tested). Still open:
  - [x] Synthetic RAFT (Table 3 "Mean Flow Cosine" / "Pixel EPE"): `synthetic_flow.py` and
    `calibration.json` are vendored unmodified from `github.com/shaoxiongduan/ptlflow` (Apache-2.0
    upstream) under `evaluation/modules/vendor/ptlflow_synthetic/` with a NOTICE; `ptlflow_root` and
    the calibration default to them. Actions are inferred per pair from the reference RAFT flow, no
    depth model. Surfaced as `synthetic_mean_flow_cosine` / `synthetic_pixel_epe_mean`.
    Verified on Modal (5 seeds; paper in parentheses):

    | Model | BallRoller | BeamDomino | MirrorRollingBall |
    |---|---|---|---|
    | Genie cos / EPE | -0.33 / 665 (-0.33 / 665) | 0.99 / 839 (0.99 / 841) | 0.98 / 271 (0.98 / 271) |
    | LingBot Act cos / EPE | 1.00 / 612 (1.00 / 621) | 0.99 / 967 (0.99 / 970) | 1.00 / 668 (1.00 / 678) |
  - Reference flow is not baked: ~1.8 MB per pair at 1280x720 float16 (24-32 pairs, ~45-60 MB per
    case) for determinism/speed only, since RAFT's weights are pinned and the candidate side
    still runs RAFT.
- [x] **Headline relative EPE** (done) / **resample to the reference** (decided against).
  Measured on BallRoller (Modal, 5 seeds), relative EPE: Genie `none` 0.55 vs
  `match_reference` 0.46 (paper 0.55 ± 0.11); LingBot `none` 0.16 vs `match_reference` 0.13
  (paper 0.16 ± 0.01). Time-stretching the whole clip onto the reference duration breaks
  paper parity (and doubles LingBot's compared pairs, 11 -> 23); the module already handles
  the 832x464 / 16 fps-tagged clips against the 1280x720 reference, so `none` is kept.
- [x] **MirrorRollingBall controllability** is served by `MirrorRollingBall/lpips_collision`
  (its reference is byte-identical to the paper's third-person capture). Evidence: below.

**Verified 2026-10-10** (Modal, 5 seeds, `candidate_normalization: none`; relative EPE / cosine,
paper Table 3 in parentheses):

| | BallRoller | BeamDomino | MirrorRollingBall |
| --- | --- | --- | --- |
| Project Genie | 0.55 / 0.71 (0.55 / 0.71) | 0.43 / 0.99 (0.44 / 0.98) | 0.77 / 0.83 (0.77 / 0.84) |
| LingBot-World (Act) | 0.16 / 0.99 (0.16 / 0.99) | 0.34 / 0.99 (0.34 / 0.98) | 0.49 / 0.94 (0.49 / 0.94) |

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
| BallRoller/ctrl | ✅ `controllability.ball_roller.collision_path_v1`, target `ball` (§2) | ✅ bundled | ❌ none |
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
| all 4 | ✅ `liquid_container` trajectory (nominal 12 Hz profile, `reconstructed`) + Genie key program | ✅ masks + pooled flow baked | ✅ resampled |

| Model | In paper | Status |
| --- | --- | --- |
| Project Genie | ✅ | ⚠️ supported (`nominal` keys); paper rollouts produced elsewhere, rerun with `actions.jsonl` |
| LingBot-World, Matrix-Game 2.0, Matrix-Game 3.0 | ✅ | ❌ `unsupported` → excluded (incompatible interface, per the experiment setup); drop the prompt-driven paper rows |
| HY-WorldPlay | ✅ | ❌ not registered in `action_backends.yaml` |
| Waypoint-1.5 | — | not run |

- [x] **Define the bottle action target**: `liquid_container`, normalized
  forward/right/up in the container frame, sampled from
  `bottle_controller_genie_profiles.py`; backend support in `action_backends.yaml`.
  The capture itself stepped on wall-clock (~10 Hz); the legacy `action.npy` keeps that record.
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
| all 3 | ✅ `rolling_ball` trajectory, `exact` (compiles bit-for-bit to the legacy `action.npy`) + Genie key program | ✅ frozen QA + `judge_answers.json` (5 passes) | ❌ identity (`beamdomino_native`) |

| Model | In paper | Status |
| --- | --- | --- |
| Project Genie | ✅ | ⚠️ supported via binary keys: `glancing` 0.4 lateral and `direct_strike` 0.6 forward are held as full presses |
| LingBot-World (Cam) | ✅ | ❌ `unsupported` → excluded (incompatible interface, per the experiment setup); drop the prompt-driven paper row |
| Waypoint-1.5 | — | not run |

- [x] **Define the strike action**: target `rolling_ball`, normalized z/x force on the
  floor plane, from `controller.py` `TRAJECTORIES`; backend support in `action_backends.yaml`.
- [x] **Audit the LingBot (Cam) paper row**: it was prompt-driven (static camera, no ball
  action). Decision: exclude it, like Matrix-Game, as an incompatible interface.
- [ ] **Remove the excluded rows from the paper**: LingBot (Cam) from the causality table
  and text; LingBot / Matrix-Game 2.0 / 3.0 from the fluid table and text.
- [ ] **Keep `causal_step_judge_repeats: 5`** and record every candidate judge pass;
  re-baseline the frozen reference answers when the judge or prompt changes.
  Five judge passes over one rollout is a different estimator from the paper's
  five rollouts: it makes fractional CSR expressible but does not reproduce
  Table 12. gpt-5.4 sends no temperature, so passes are sampled.
- [x] **Refuse mismatched recordings.** `causal_step` now rejects a
  `judge_answers.json` recorded with a different judge model, frame count or
  prompt version (fef83e8).
- [x] **Repair the four reference-side descriptive misses** (`direct_strike` seg1
  stale distractor; `causal` seg2; `glancing` seg1 and seg2), checked against the
  Unity logs and the judge's K=8 frames. Re-recorded and re-locked: `csr_unity`
  80/20/0 → 100/80/100 (Table 12: 100/60/100); only `glancing` seg1 is below 5/5
  (4/5). The Lite QA now differs from the paper's; see the suite.yaml banner (e4e1c9f).

## 6. Lighting & shading — `lighting_shading`

**Owner:** Forrest Dai · **Active case:** SunRisesAndSets/lighting_orbit_v2.

The case ships a complete ten-second, two-metre camera orbit, explicit tabletop
look-at target and fixed sun, with trajectory.yaml, action.npz, actions.jsonl,
table_cell.json and verified Unity poses/closure. Native videos are timestamp
sampled into exactly 120 frames at 12fps without padding or retiming.

LingBot, Matrix-Game3 and HY-WorldPlay are supported and each ships five first
valid native rollouts (42,1,2,3,4), original generation receipts and relative
candidate paths in baselines/lighting_orbit_v2. MG2/Waypoint lack a documented
metric input conversion; Genie's hosted UI lacks native-frame action submission.
These interfaces are explicitly N/A for the strict protocol, with source/timing
evidence in experiments/lighting_orbit_v2/results/capability_audit.json.
Generated appearance is not a support criterion. Five executed Genie exports
and scores remain diagnostic and never enter the formal five-seed aggregate.

2026-10-11 acceptance: on the pinned RTX PRO 6000/CUDA12.8 environment, all 15
candidates recomputed through run_wmeval_lite.py, all 90 metric values exactly
matched the original scores at 1e-6, and actual --resume preserved row hashes.
Two independent LingBot/42 scores also matched the original. Correcting a stale
derived lock stamp changed no evaluator input or score; the corrected stamp also
passed actual resume. Reports: experiments/lighting_orbit_v2/results/same_gpu_verification/.
The separate PRO 4500 audit records 21/90 small numerical differences and remains
visible; exact cross-GPU equality is not claimed. Use the pinned hardware for
strict fresh verification, or the bundled cached scores for portable exact replay.

The portable archive contains the full case, fifteen videos, public scorer,
relative candidate manifest, immutable asset/source/dependency locks and explicit
fifteen expected jobs plus three N/A entries. Its builder runs from the public
release_sources.json with no private scratch directory or old Pod. The source
fingerprint and cached input hashes invalidate changed cases/videos/parameters.

Historical SunRisesAndSets/lighting and its 30 native-paper results remain
separate. Run the old case with --suite eval_sets/wmeval_lite_v1/lighting_paper_native;
baselines/lighting_paper_native.json retains native metadata and all 180 metric
values. Hosted historical Genie runs have no recorded numeric generation seeds.
HF publication remains a separate action, not a release acceptance blocker here.

- [x] **Standardized action:** one semantic metric look-at trajectory with explicit camera target; native pose compilers and audited N/A interfaces.
- [x] **Common fps:** exactly ten native seconds / 120 timestamp-sampled frames at 12fps; no padded or retimed video.
- [x] **Five-seed reproduction:**15/15 supported runs,90/90 strict pinned-GPU metrics, unchanged actual resume, portable data/scorer archive and source/asset/version pins.
- [ ] Optional after release: ChessBoard/lighting and a separately defined moving-sun lighting_sunsweep case.

## 7. Prompt-scene consistency — `prompt_scene_consistency`

**Owner:** Rita Wu · **Paper:** prompt-consistency table · **Lite cases:**
ChessBoard, MirrorRollingBall, RacingCar × prompt_first / prompt_third

**Run it for a new model:** the single path with `--dimensions prompt_scene_consistency`
(`OPENAI_API_KEY`). Key files: `cases/*/prompt_*/case.yaml` (prompt),
`reference/ref.png`, `reference/artifacts/rubrics/`,
`tools/evaluate_world_model_vlm_consistency.py`.

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| all 6 | ⚠️ canonical 10 s left turn baked (`prompt.camera.left_turn_v1`, 40 °/s, target `camera`); generation inputs replay each model's original rollouts instead | ✅ rubrics and Unity reference scores frozen in the bundle; hosted `gpt-5.4` judge, snapshot recorded per call | — low risk |

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
  `tools/prepare_prompt_case_inputs.py` writes the inputs the rollouts were generated with
  (seeds 0–4): LingBot poses, 161 frames at 16 fps (one counter-clockwise turn in place in
  first person, two orbits around the character in third person); Matrix-Game 3.0 keys,
  177 frames at 17 fps (`j` every frame, plus `d` in third person to orbit). It writes the
  five-repeat queues for
  `run_lingbot_table7.py --queue` / `run_matrix_game3.py --queue` plus the Lite candidate
  manifest.
- [x] **Reproduced through the Lite path** (case bundle → 5 seeds →
  `run_wmeval_lite.py --dimensions prompt_scene_consistency`): Project Genie, LingBot (Cam),
  Matrix-Game 3.0 and HY-WorldPlay, 120/120 rows. Against the paper's values, per-cell mean
  |Δ| is 0.023 (max 0.067) and the model ranking is unchanged
  (Unity 0.938 > Genie 0.905 > LingBot 0.891 > HY 0.792 > Matrix-Game 3.0 0.745);
  `tools/check_prompt_human_alignment.py` gives ρ 0.90 / 0.80, as in the paper.
- [ ] **Generate from the canonical turn.** The baked trajectory is 40 °/s for 10 s, but
  `prepare_prompt_case_inputs.py` reproduces the original rollouts (LingBot one turn in place
  / two orbits; Matrix-Game 3.0 `j`, plus `d` in third person), so models still receive
  different camera motion. Do: compile `prompt.camera.left_turn_v1` per backend (as §1 does
  for Table 7) or record the per-model input as an explicit, documented deviation. Done when:
  every backend's prompt inputs are derived from the baked trajectory or labeled as such.

## 8. Perceptual similarity — `lpips`

**Owner:** _TBD_ · **Paper:** causal-LPIPS table · **Lite cases:**
MirrorRollingBall/lpips_collision, BeamDomino/causal, BallRoller/ctrl, LiquidContainer x 4

**Run it for a new model:** the single path with `--dimensions lpips` (`OPENAI_API_KEY` for
the candidate's event detection; the reference is no longer judged). Key files:
`cases/<Scene>/<variant>/reference/artifacts/contact_events.json`,
`evaluation/modules/lpips*.py`, `tools/freeze_reference_events.py` (+ `_modal`).

| Case | Actions | Reference | Normalization |
| --- | --- | --- | --- |
| BallRoller/ctrl | ✅ (see §2) | ✅ frozen (5/5 passes agree on 1 collision) | `none` |
| MirrorRollingBall/lpips_collision | ✅ (see §2) | ✅ frozen (judge found 2-6 events/pass; 1 kept by majority) | `none` |
| BeamDomino/causal | ✅ (see §2) | ✅ frozen (2 events/pass at differing frames; 1 kept) | `beamdomino_native` |
| LiquidContainer x 4 | ✅ `liquid_container` trajectory (§4) | ❌ not frozen (§4's action is settled; run `tools/freeze_reference_events.py`) | ✅ resampled |

| Model | In paper | Status |
| --- | --- | --- |
| Project Genie | ✅ | ✅ scored on all three paper scenes |
| LingBot-World (Act) | ✅ | ⚠️ scored from existing rollouts; no generation adapter in this repo |
| Waypoint-1.5 | — | ⚠️ reports `unsupported` where the case target is `ball` |

- [x] **Freeze reference contact events** once per case. Evidence: `2220da2`. Every pass is
  stored; events are those a majority (3 of 5) of passes agree on, so every candidate row
  reuses the same events. The file records the reference video's SHA-256 and the loader
  rejects a mismatch. BallRoller was missing from the `lpips` dimension although the
  paper's cLPIPS table has it; it is added here.
  - Re-validated with the default `lpips_causal_event_source: both` (candidate judged live,
    Modal, 5 seeds, 30/30 rows completed). Causal LPIPS, paper in parentheses:

    | Model | BallRoller | BeamDomino | MirrorRollingBall |
    |---|---|---|---|
    | Project Genie | 0.56 (0.56) | 0.60 (0.60) | 0.73 (0.72) |
    | LingBot Act | 0.38 (0.38) | 0.58 (0.61) | 0.59 (0.62) |

    Genie falls back to `temporal_only` on 1 of 15 rows, LingBot on 4 of 15, identically across two
    runs: the candidate judge reports no contact in those videos (0 candidate events; e.g. the
    objects stay visibly apart), so there is nothing to pair with the frozen reference event. That
    is a model/judge outcome, not a pipeline fault. Offline (`event_source: reference`, zero API
    calls) the
    frozen events reproduce the paper except LingBot on BeamDomino (table below).
  - Judge bug (fixed): side relabeling overwrote an event's `evidence_frame_indices` with its
    own local window, which the grid sanitizer then rejected as `evidence_not_in_grid` (one
    reference gave 0/1/3/0/0 contacts). The relabel window now goes to
    `side_evidence_frame_indices` and the grid evidence is kept; covered by a test. Scores were
    unchanged on re-run (the fallbacks above are judge no-contact outcomes, not this bug).
- [x] **Flag `causal` vs `temporal_only` per row** — already implemented (`lpips_causal_mean`,
  `lpips_temporal_fallback_mean`, matched-event counts); the dimension summary now also
  carries `lpips_primary_source_counts`.
- [x] **Resample** — decided against `match_reference` for these cases (see §2 evidence); the
  paper values above are reproduced with `candidate_normalization: none`.

**Verified 2026-10-10** (Modal, 5 seeds, frozen reference events, `event_source: reference`,
no hosted judge; paper Table 8 in parentheses):

| | BallRoller | BeamDomino | MirrorRollingBall |
| --- | --- | --- | --- |
| Project Genie | 0.547 (0.56 ± 0.02) | 0.620 (0.60 ± 0.02) | 0.729 (0.72 ± 0.02) |
| LingBot-World (Act) | 0.382 (0.38 ± 0.02) | **0.547** (0.61 ± 0.02) | 0.632 (0.62 ± 0.05) |

LingBot on BeamDomino is low because its event timing drifts from the reference's: mapping the
reference's timing onto it misses its own contact. `event_source: both` (judging the candidate)
gave 0.615 before the freeze; re-checking it needs API credits.

---

## Release owner — cross-cutting items

- [ ] **Scope and rights (P0-01).** Freeze the case roster (32 live Lite case bundles, including the 14 Table 4/7
  cells) and repeat policy; clear redistribution of Unity renders, quantities and
  annotations; add LICENSE files, attribution, citation, dataset card; confirm HY-WorldPlay.
- [ ] **HF package (P0-02).** Export → download → verify into a clean directory with
  per-file hashes and dataset/evaluator/schema versions; nothing may resolve into `runs/`,
  a private repo or a Modal volume.
- [ ] **Adapter template (P0-05).** A model-neutral adapter (capability check +
  generation) driven by `action_backends.yaml`, a no-cost fixture backend, an expected-job
  export (case × repeat × seed) and receipts accepted by the Lite runner.
- [ ] **Environment (P0-06).** Extend `experiments/tables_4_7/evaluator_assets.lock.json`
  (GroundingDINO, Depth Anything, SAM 2.1, DINOv2, RAFT, LPIPS/VGG pinned) to PTLFlow,
  OpenPIV and indoor depth; fill the 31 empty case dependency locks (only lighting is
  pinned); add a preflight.
- [ ] **Results (P0-07/09).** `unsupported` rows and `num_unsupported` exist (#35). Per-metric status (`ok`, `unsupported`, `missing_candidate`,
  `invalid_candidate`, `not_applicable`, `unmeasurable`, `error`); a coverage gate; one
  repeat policy (LingBot's adaptive 5→30 vs. five runs elsewhere).
- [ ] **Acceptance (P0-10).** Hermetic tests (no reads from `runs/` or the sibling paper
  checkout), CI, and a clean-machine run of the Lite path from the HF download with one new
  model.

## Release order

1. Scope and rights; freeze schemas (`actions.json`, reference manifest, result status).
2. Re-score one model through the Table 4/7 Lite cases (§1, §3); close the last action gaps (§1 legacy cases, §7 generation).
3. Freeze LiquidContainer reference events (§8); lock normalization everywhere.
4. Adapter template, environment lock, coverage gate.
5. Clean-machine acceptance from the HF package; ship when every item above is closed.

Related: open-model Table 4/7 status and calibration →
[`TABLE4_TABLE7_REMAINING_MODELS_TODO.md`](../../TABLE4_TABLE7_REMAINING_MODELS_TODO.md);
generated readiness report → `PRODUCTION_READINESS.md`
(`python tools/audit_production_readiness.py`).
