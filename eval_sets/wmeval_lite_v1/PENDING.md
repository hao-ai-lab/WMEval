# Pending case data — intake checklist

Deliverables for each planned-but-not-yet-bundled case. Case folders are
created **only when the data lands** (the trust layer rejects placeholders:
everything under `cases/` must validate). Wiring steps for every case:

1. `mkdir -p cases/<Scene>/<variant>/reference/artifacts` and drop the files
2. write `case.yaml` (copy a live case of the same kind as template)
3. `python tools/build_lite_case.py cases/<Scene>/<variant>` (checksums + lock)
4. add the case id to the dimension's `cases:` list in `suite.yaml`
5. `python tools/run_wmeval_lite.py validate` (+ `generate-rubrics` where noted)

Delete each section (and finally this file) as data lands.

---

## ChessBoard — prompt_scene_consistency + lighting_shading

**`ChessBoard/prompt_first`, `ChessBoard/prompt_third`**

| file | notes |
| --- | --- |
| `reference/ref.png` | first frame |
| `reference/unity_groundtruth.mp4` | GT rollout (smoke-test candidate) |
| `reference/artifacts/actions.txt` | keyboard script that drove the rollout (camera moves through the scene) |
| `reference/artifacts/action.npy` (+ `intrinsics.npy`, `poses.npy`, `meta.json`) | numeric action data |
| case.yaml `prompt:` | scene description text |
| case.yaml `rubric_scene_name:` | `ChessBoard_first` / `ChessBoard_third` |

After bundling: `run_wmeval_lite.py generate-rubrics` (needs `OPENAI_API_KEY`) — writes the rubric into `reference/artifacts/rubrics/` and re-authors the bundle; commit the result.

**`ChessBoard/lighting`** — team decision first: if lighting judges the same
video as `prompt_third`, do NOT create a case — just list
`ChessBoard/prompt_third` under `lighting_shading.cases` (borrow). If lighting
needs its own rollout, create the case with
`module_params.shadow_consistency.{shadow_text_prompt, shadow_anchor_label}`
(e.g. chess pieces as shadow casters).

## SunRisesAndSets — lighting_shading

**`SunRisesAndSets/lighting`** (third person)

Same bundle shape as a prompt case (prompt + ref.png + GT video + actions) plus
`module_params.shadow_consistency` detection prompts. Note: the sun moving is
the point of this scene — shadow-direction metrics (MOSD/SDJ) assume a static
sun; confirm with the dimension owner how the moving-sun protocol is scored
before authoring.

## BallRoller/ctrl — environment_object_consistency (LIVE, object half only)

Resolved and wired into suite.yaml (2026-08). Scoped to the case's static
`ramp`/`dome` obstacles only — ground truth (`reference/artifacts/
object_transforms.json`) was computed by reading their world positions
directly from the Unity scene file in Edit mode (no rendering needed for
static objects), combined with the case's already-shipped `poses.npy`/
`intrinsics.npy`. `object_permanence_pct` self-validates clean (100%).
`depth_absrel_pct`/`pairwise_distance_mae_pct` are real and correctly wired
but show large (~70-80%) self-match error — see suite.yaml's header comment
for the root cause (a real-world-trained monocular depth model has a
systematic scale bias on synthetic content; not a wiring bug). The
`character_consistency_pct` half is explicitly disabled for this case
(`eoc_character_prompt` set to a non-matching string) — the ball ("player")
was never reviewed as a character-identity subject; see the WanderingGarden
section below for the actual paper-mapped character-consistency dimension.

## WanderingGarden — scene_character_consistency

**`WanderingGarden/scene_first`, `WanderingGarden/scene_third`**

Bundle shape as prompt cases. Blocked on defining the dimension itself:
which judge configuration (the PR #14 vlm_consistency pipeline is the
candidate), what the rubrics are generated from, and its suite.yaml
`vlm:`/aggregate spec.

**Open conflict to resolve (flagging for a human decision, not silently
picking one):** this note predates
`evaluation/modules/environment_object_consistency.py`, which implements the
paper's appendix-B.1 spec verbatim (GroundingDINO + SAM2 + DINOv2, registry
evaluator, `requires_reference: false` for its two candidate-only metrics).
That is a different evaluator family than the `vlm_consistency` judge this
note assumed. Before wiring `scene_character_consistency` in suite.yaml,
decide: (a) `evaluator: registry`, `modules: [environment_object_consistency]`
per the paper, (b) the VLM judge as originally planned here, or (c) both,
reported side by side. Once decided, this dimension also needs:

| file | notes |
| --- | --- |
| `reference/artifacts/unity_object_transforms.json` | per-frame per-object 3D world positions (schema in the module docstring) — only needed for `depth_absrel_pct`/`pairwise_distance_mae_pct`, not for `object_permanence_pct`/`character_consistency_pct` |
| `reference/artifacts/poses.npy`, `intrinsics.npy` | camera-to-world + intrinsics, same convention already bundled in `BallRoller/ctrl` |

## BeamDomino — physical_causality

**`BeamDomino/causal`** — reference video + actions are ALREADY bundled
(earmarked). Missing only:

| file | notes |
| --- | --- |
| `reference/artifacts/qa_dataset.json` | CausalStep stepwise QA over the rollout |

Then: re-run `build_lite_case.py`, add the `physical_causality` dimension to
suite.yaml (registry, modules `[causal_step]`, ref-based, headline from the
csr/delta family). More strike variants (`causal_glancing`,
`causal_direct_strike`) each need their own reference.mp4 + actions + QA.

## LiquidContainer — fluid_mechanics

**`LiquidContainer/fluid_rest`, `fluid_step_hold`, `fluid_impulse_decay`, `fluid_chirp_strafe`**

| file | notes |
| --- | --- |
| `reference/reference.mp4` | sealed-bottle rollout, protocol-locked geometry |
| `reference/artifacts/liquid_mask.npz` | SAM2 per-frame liquid mask |
| `reference/artifacts/bottle_mask.npz` | SAM2 bottle mask |
| `reference/artifacts/raft_pooled_flow.npz` | pooled RAFT flow field |

The `.npz` artifacts MUST be produced with
`evaluation/lite/reference_artifacts.write_*` (PR #19's API) or the
`evaluator_lock` consistency check refuses them — coordinate with
@abhilashreddys. case.yaml `protocol.candidate_normalization:
liquid_832x464_16fps`. suite.yaml gains the `fluid_mechanics` dimension
(4 liquid modules).

## BallRoller / RacingCar — kinematics_mechanics (still NOT live — new root cause found)

Module confirmed working end-to-end: `evaluation/modules/kinematics_mechanics.py`
(paper appendix B.3: GroundingDINO + ByteTrack trajectory reconstruction,
camera-motion compensation via static anchors). `BallRoller/ctrl`'s
`case.yaml` already has `module_params.kinematics_mechanics` fully
configured (text prompt, GT artifacts, a tuned `kin_track_thresh`) and
`reference/artifacts/object_masses.json` is bundled — this is ready
infrastructure, kept even though the dimension isn't wired live, because
the blocker below is about ground-truth data quality, not this module.

Three real bugs were found and fixed while getting this far (all in
`evaluation/modules/kinematics_mechanics.py`):
1. `_compensate_camera_motion` omitted the `centroid_compensated` key
   entirely when no static anchors were configured, crashing downstream
   with `KeyError`. Now defaults to the raw (uncompensated) centroid.
2. `_detect_all_frames` had no defense against the same degenerate/
   duplicate GroundingDINO detection issue found in
   `environment_object_consistency.py` (see below) — a completely separate
   detection pipeline that didn't inherit that fix. Now reuses
   `_dedupe_seed_detections`.
3. The default `kin_track_thresh=0.5` sat above GroundingDINO's actual
   detection confidence on this content (~0.37-0.43), so ByteTrack's
   two-stage association never had a high-confidence detection to *start* a
   new track — zero tracks formed despite clean per-frame detection. Fixed
   via a per-case `kin_track_thresh: 0.3` override (not a module-default
   change, since that threshold should be tuned per detector/content, not
   hardcoded repo-wide).

With all three fixed, tracking genuinely works (`dome`: 171/173 frames,
`ramp`: 17/173 — real, non-fabricated ByteTrack output on `BallRoller/ctrl`'s
static ramp/dome, scoped the same way as `environment_object_consistency`
above; the moving ball remains excluded for the data-provenance reason
already documented).

**Why it's still not wired live: `poses.npy`'s rotation component is not
real camera-orientation data.** `trajectory_fidelity` needs to reproject the
Unity 3D ground truth into 2D pixels through `poses.npy`'s camera-to-world
matrix. Doing that for `dome` (a perfectly static object, per its own
detected pixel trajectory) lands at `v≈-1151.85`, while the real detected
pixel is `v≈548.7` — over 1700px off. `poses.npy`'s rotation block is
identity at every single frame of this case, despite the case's own prompt
describing "a fixed side-table camera looking **diagonally** across the
arena" — i.e. whatever produced `poses.npy` only encoded real translation,
not real orientation. `environment_object_consistency`'s depth/distance
metrics were unaffected by this because Euclidean distance is
rotation-invariant; `kinematics_mechanics`'s pixel reprojection is not.

This likely affects every case built on the same external capture
pipeline, not just `BallRoller/ctrl` — worth checking before trusting
`poses.npy` for reprojection-dependent metrics anywhere else in the suite.
Fixing this needs either real per-frame camera rotation captured alongside
translation, or a different metric formulation that doesn't require it.
`mechanical_quantities` is separately still capped at pixel units only
(no per-frame *candidate* camera-pose artifact exists to lift the
candidate's monocular trajectory into a metric 3D frame) — a distinct,
already-documented paper-blueprint gap (see suite.yaml comments).

**Attempted and rejected: empirically recovering the missing rotation.**
Since we know both objects' real 3D positions and can detect their real 2D
pixel positions, tried solving for the camera's 3-DOF rotation (translation
held fixed at the known value) via least-squares reprojection-error
minimization over the 2 known correspondences (`ramp`, `dome`; 200 random
restarts, Nelder-Mead). Best fit: **~77px residual per point** — nowhere
near the sub-pixel-to-few-pixel residual a correct model should produce.
Real camera-pose recovery (PnP) needs ≥3 non-collinear correspondences to
be verifiable; with exactly 2, this is fundamentally underdetermined —
the "best fit" cannot be distinguished from an equally-plausible wrong
one. **Empirically confirmed, not just theoretical**: tested the
recovered rotation against a genuine held-out 3rd point (the case's
"player" ball, whose known Edit-mode initial position is `(-7, 0.3, 0)`
and whose real frame-0 pixel position was independently detected at
`(729.34, 409.86)`, not used in the original 2-point fit). The recovered
rotation predicts `(545.85, 785.62)` for it — a **418px residual**, 5x
worse than the ~77px training-set fit. That's the textbook signature of
overfitting to too few points, not a real generalizable camera pose. Do
not ship this (or repeat this exact attempt) as ground truth without at
least one more independent static anchor object in the scene.
The only real ways forward: (a) a 3rd+ static anchor object captured in a
future case-authoring pass, so PnP becomes well-posed, or (b) the original
capture harness's actual camera rotation, if it exists in a log somewhere.

## BilliardGame

In the scene roster but unmapped to any dimension — confirm intent before
authoring anything.
