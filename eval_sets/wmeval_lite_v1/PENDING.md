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

## ChessBoard/walk_third — environment_object_consistency (RESOLVED, LIVE as of 2026-08)

Unlike every other `environment_object_consistency` case so far, this is a
**freshly captured rollout**, not a re-analysis of an existing GT/video pair
— `Chess Demo.unity` (the real scene; `Assets/Scenes/SampleScene.unity` is an
empty decoy) had no case at all yet. Resolved and wired into `suite.yaml`.

**Build/capture pipeline.** Confirmed once more this session: the Unity
*Editor* cannot render in this environment (batchmode `-nographics` gives a
null graphics device; without `-nographics` the Editor enters a crash/relaunch
loop in Play mode) — do not retry that path. A compiled standalone macOS
Player genuinely works (`-batchmode`, no `-nographics`, real Metal device
attaches). Built via the project's existing `Assets/Editor/BuildScript.cs`
(`BuildScript.BuildMacOS`, explicit `-scenes "Assets/Scenes/Chess Demo.unity"`)
to `Builds/ChessBoardwithSphere/macOS/Chess Board.app`. A runtime capture
harness, `Assets/Scripts/ChessCaptureBootstrap.cs` (left uncommitted in
`WorldModelBenchScenePrivate`, matching the `BootstrapSpectatorCamera.cs` /
`DumpMirrorSceneTransforms.cs` precedent of leaving diagnostic/capture
scripts in place uncommitted), self-registers via
`[RuntimeInitializeOnLoadMethod]` (no scene-file edits needed) and:

- Disables the scene's `PlayerArmature`'s ML-Agents `PlayerAgent` +
  `DecisionRequester` components. **Real bug found**: without this, the
  DecisionRequester periodically calls `PlayerAgent.OnActionReceived` with
  whatever `PlayerAgent.Heuristic()` computes from `Keyboard.current` — all
  zero in this headless batchmode Player (no real keyboard) — silently
  overwriting the capture script's directly-set `StarterAssetsInputs.move`
  back to neutral every decision step. Confirmed empirically: before this
  fix, the player's x/z position and the (Cinemachine-following) camera were
  frozen identical across all 90 captured frames despite driven input; after
  disabling those two components, the player visibly walks/turns/jumps and
  the camera dollies to follow it.
- Flips the scene's Main Camera from its shipped default (`FreeCameraController`
  enabled, `CinemachineBrain` disabled — a keyboard fly-cam, since
  `Assets/Editor/ChessSceneSetup.cs`'s Editor-only `playModeStateChanged` hook
  that normally fixes this never runs in a built Player) to the intended
  Cinemachine third-person follow camera.
- **Real bug found and fixed**: `ScreenCapture.CaptureScreenshotAsTexture()`
  reliably returns `null` in this batchmode standalone-Player context (no
  real window backbuffer to grab from, even though the GPU/rendering itself
  works — confirmed via all-black/zero-byte-content frames on the first
  attempt using this API). This is a *new* finding this session, distinct
  from the already-known Editor-Play-mode blocker. Fixed by rendering Main
  Camera into an explicit `RenderTexture` (640x360, `ARGB32`) via an explicit
  `Camera.Render()` + `Texture2D.ReadPixels()` each frame instead — this
  produces real, non-black frames (confirmed via pixel mean/std and visual
  inspection). An earlier attempt routed capture through the project's
  pre-existing `SpectatorCamera` (`Assets/Scripts/BootstrapSpectatorCamera.cs`,
  unrelated infra for the "ML-Agents hijacks Main Camera" problem) instead of
  Main Camera directly — that produced fully black frames (likely a URP
  camera-stack registration gap for a camera added purely via
  `AddComponent<Camera>()` at runtime); using the scene's already-properly-
  configured Main Camera directly avoided the problem.
- Drives a deterministic 90-frame/3s (30fps) scripted rollout: walk forward
  (frames 0-24), diagonal turn + one jump at frame 30 (25-44), sprint forward
  (45-69), turn back + walk (70-89). Full detail in the shipped
  `reference/artifacts/actions.txt`.
- Logs, per captured frame: the camera-to-world matrix built directly from
  `Camera.transform.right` / `-up` / `forward` as the CV-convention (x right,
  y down, z forward) basis columns (matching this module's
  `pc = w2c @ [x,y,z,1]; u = fx*pc.x/pc.z+cx; v = fy*pc.y/pc.z+cy` formula
  exactly, no ad hoc sign-flip guessing needed); intrinsics derived from the
  real runtime vertical FOV and 640x360 resolution
  (`fx=fy=(H/2)/tan(fovY/2)`, `cx=W/2`, `cy=H/2`); and world positions of the
  scene's `Sphere` (a large reflective chrome/glass ball) plus 4 further
  decorative king statues (`King_Gold`/`King_Wood`/`King_Metal`/
  `King_Marble`).

**Self-verification (before trusting any of it as ground truth)**:
reprojected the logged `Sphere` world position through the logged pose/
intrinsics for two frames with genuinely different camera positions (frame 0
and frame 60 — the camera dollies during the rollout) and confirmed the
predicted pixel visually lands inside the sphere's rendered disk both times
(crops centered on the predicted point show the sphere's reflective interior,
not background). Cross-validated independently and quantitatively: a real
GroundingDINO `"sphere."` detection on frame 0's top-scoring box (score
0.411) centers at `(306.4, 154.6)`, within ~2px of the analytical
reprojection `(306.9, 152.9)` — strong agreement from a completely
independent method, confirming both the capture convention and the detector
pipeline agree on where the sphere actually is.

**Real bug found in case authoring** (same class as `RacingCar/prompt_third`'s
"`yellow sports car`" vs `"car"` mismatch): first attempt used
`eoc_text_prompt: glass sphere.` — GroundingDINO's literal detected label came
back as `"glass sphere"`/`"glass"`, neither of which case-insensitively
matches the ground-truth JSON key `"sphere"`, so `depth_absrel_pct` silently
came back `null` with `ground_truth_missing_labels: ["glass sphere", "glass"]`.
Fixed by simplifying the prompt to `sphere.` (bare single word, matching the
GT key) — re-tested detection first (top box score 0.411 at effectively the
same pixel location) before committing to the change.

**Scoping**: only ONE static anchor (`sphere`) is usable.
`King_Gold`/`King_Wood`/`King_Metal`/`King_Marble` are real objects with real,
Edit-mode-verified world positions (kept in the capture log), but reprojecting
all 4 through every one of the 90 frames' camera poses shows they are never
actually within the visually-rendered frame during this specific rollout (the
reprojected pixel coordinates fall in frustum bounds on paper, but crop
inspection at those pixels shows flat background — they're far enough away
along the camera's viewing depth, ~35-48m vs. the sphere's ~22-26m, that the
board's near dressing occludes them or they're simply out of the actually-
walked path). `pairwise_distance_mae_pct` is therefore expected/left `None`
for this case — disclosed, not a bug, same structural situation as
`MirrorRollingBall/prompt_third` and `RacingCar/prompt_third`.

**Character consistency**: unlike every other `environment_object_consistency`
case so far (all of which deliberately disable `eoc_character_prompt` with a
non-matching string), this case sets a real, matching
`eoc_character_prompt: person.` for the walking humanoid `PlayerArmature` —
genuinely reviewed, not a placeholder. GroundingDINO detects it with
plausible confidence (~0.36-0.54 across frames, seed score 0.49 at frame 0).

**Self-consistency check** (candidate = reference video, all 90 frames, real,
non-fabricated): `object_permanence_pct: 100.0`, `character_consistency_pct:
43.33`, `depth_absrel_pct: 91.85` (self-match error — same real-world-trained
monocular depth-model domain-gap bias documented for the other cases in
suite.yaml's header, on the higher end of the range seen so far: BallRoller
~62-80%, RacingCar ~28%, MirrorRollingBall ~7%, this case ~92%),
`pairwise_distance_mae_pct: null` (single usable anchor, see above).

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

## MirrorRollingBall/prompt_third — environment_object_consistency (LIVE, 2026-08)

Resolved and wired into `suite.yaml`. This is a materially cleaner result than
`BallRoller/ctrl` or `RacingCar/prompt_third`: the case's shipped `poses.npy`/
`intrinsics.npy` needed **no correction at all** (not even the small offset
`RacingCar/prompt_third` needed) — real per-frame camera rotation AND
translation, verified by reprojecting the known Player world position
`(0, 0.5, 0)` (read via `DumpMirrorSceneTransforms.cs`, an Edit-mode-only
Unity batchmode script at
`WorldModelBenchScenePrivate/GameScenes/MirrorRollingBall/Assets/Editor/
DumpMirrorSceneTransforms.cs`, left in place uncommitted as a diagnostic,
matching the `DumpStaticTransforms.cs`/`DumpCarPosition.cs` precedent)
through every one of the 300 frames' camera poses and landing exactly on the
image center `(640, 360)` every single time — a real look-at/chase camera
orbiting the stationary mirror ball. The translation traces a clean circular
orbit (`x^2 + z^2 = 4.97` = const, radius 2.230) around the origin at height
`y=1.63`, matching the case's `actions.txt` (`hold ArrowLeft 10000ms`, i.e. a
10s/300-frame full rotation).

**Scoping**: only ONE static anchor is usable. The scene's `Assets/Scenes/
SampleScene.unity` has a `YellowBalls` parent with 48 individually-positioned
child `GameObject`s (a real 7x7-ish grid, `y=0.5`, dumped via the same Edit-mode
script) plus 4 named walls (`North`/`South`/`East`/`West Wall`) and a `Player`
at `(0, 0.5, 0)`. But all 48 yellow balls are visually identical, so
`eoc_text_prompt: yellow sphere.` can only ever seed ONE tracked instance —
the module's own `_dedupe_seed_detections` collapses same-label detections to
their single highest-scoring box, and there is no way to distinguish "the 3rd
ball from the left" via a GroundingDINO text prompt. Identified which physical
ball that is (`YellowBall (30)` at world `(-2, 0.5, -0.05)`) by reprojecting
all 48 known ball positions into frame 0 and matching against GroundingDINO's
actual top-scoring box, then cross-validated with an independent VLM
pixel-center estimate (same NVIDIA endpoint as `tools/vlm_sidecar_review.py`):
reprojection `(145.4, 357.3)` vs GroundingDINO `(142.7, 357.4)` vs VLM
`(137, 351)` — all agree within 8px, ruling out a coincidental/wrong match.
`reference/artifacts/object_transforms.json` lists this ball's position for
the 201/300 frames it's actually within the camera frustum (two runs:
frames 0-152 and 252-299 — the orbit carries it out of view for the rest).

Also tried `wall.` as a second anchor for `pairwise_distance_mae_pct`: at this
FOV the low gray boundary wall's detection box spans essentially the entire
frame width (`>60%` of frame area), which trips the module's own
`_DEGENERATE_BOX_AREA_FRACTION` filter and gets dropped entirely — same root
cause class as `RacingCar/prompt_third`'s baked-mesh problem (no
individually-boxable local wall segment in this framing), just discovered via
the module's existing degenerate-box guard rather than mesh inspection.
`pairwise_distance_mae_pct` is therefore expected/left `None` for this case —
disclosed, not a bug.

`intrinsics_t4.npy` was derived from the case's shipped `(3,3)` K-matrix
`intrinsics.npy` (same `fx=fy=623.5383, cx=640, cy=360` values, tiled across
all 300 frames) — the original `intrinsics.npy` is untouched.

**Self-consistency check** (candidate = reference video,
`analyze_environment_object_consistency_from_context`): run against the first
60 of 300 frames only (the object's real continuous visibility window is
frames 0-152; 60 was used purely to keep the validation run's wall-clock time
reasonable on this CPU-only environment — the shipped case artifacts
themselves cover the full 300-frame/201-visible-frame span, this trim only
affected the ad hoc validation run, not any committed artifact). Real,
non-fabricated results: `object_permanence_pct: 100.0`, `depth_absrel_pct:
7.22` (self-match error — far better than `BallRoller/ctrl`'s ~62%, consistent
with this case needing zero camera-pose correction), `character_consistency_
pct: null` (`eoc_character_prompt` deliberately non-matching, same
unreviewed-character posture as `BallRoller/ctrl`), `pairwise_distance_mae_
pct: null` (single usable anchor, see above).

**Performance note for future validation runs**: `_detect_objects_
groundingdino` (`evaluation/modules/shadow_consistency.py`) calls
`AutoProcessor.from_pretrained`/`AutoModelForZeroShotObjectDetection.
from_pretrained` on every single invocation with no caching — correct but
very slow for full-length ad hoc validation runs (a first attempt at the full
300-frame self-consistency check was killed after >20 minutes, still in the
per-frame re-detection loop, each frame reloading the model twice). Not a
correctness bug (each case run in production presumably calls this a bounded
number of times per case, and caching is an orthogonal perf concern), but
worth knowing before attempting another full-length ad hoc validation.

**`MirrorRollingBall/prompt_first`** (first-person) was not attempted this
pass but looks like a promising follow-up: same scene, same real per-frame
rotation, but camera translation is *constant* at `(0, 0.85, 0)` (a
first-person in-place-rotating camera, not an orbit) — a different, equally
real and self-consistent camera model, not evaluated further here due to
time/effort budget.

## WanderingGarden/scene_first — environment_object_consistency (RESOLVED, LIVE as of 2026-08, wired into suite.yaml)

**Open conflict above (VLM judge vs. `environment_object_consistency` registry
module) is now moot for the object-consistency half**: per the paper text
("WanderingGarden is used for scene and character consistency: depth AbsRel,
pairwise distance MAE, and object permanence over five designated garden
anchor objects... Since WanderingGarden is evaluated from a first-person
exploration view and does not expose a persistent visible main character...
we do NOT report main-character consistency for this scene"), this scene's
dimension is squarely `environment_object_consistency` (registry, paper
appendix B.1), object-consistency metrics only — `character_consistency_pct`
is deliberately left disabled (`eoc_character_prompt` set to a
non-matching string), matching the `BallRoller/ctrl`/`MirrorRollingBall`
precedent for cases without a reviewed character subject. The `vlm_
consistency` judge idea from the original note is not needed for this half;
if a separate prompt/scene-consistency judge dimension is still wanted for
WanderingGarden later, that's an independent, still-open decision.

**`WanderingGarden/scene_first`** — bundled via a genuinely rendered capture
(first confirmed working render pipeline for this scene, see below), reprojection
self-verified, case bundle authored and `build_lite_case.py`-validated. **Now
wired into `suite.yaml`'s `environment_object_consistency.cases:` list**
(added 2026-08 after human review of the numbers below).

**How the ground truth was captured**: Editor Play-mode rendering is a
confirmed dead end in this environment (see session history); instead built a
real standalone macOS Player (`Assets/Editor/BuildScript.cs BuildScript.
BuildMacOS`, scene `Assets/Scenes/Garden/GardenScene.unity` — the actual
populated scene; `Assets/Scenes/SampleScene.unity`-style decoys were not an
issue here since the scene roster only has one Garden scene) and ran it
directly (`-batchmode`, no `-nographics`) with a bootstrap `MonoBehaviour`
(`Assets/Scripts/GardenGTCapture.cs`, spawned via `[RuntimeInitializeOnLoad
Method]`, left in place uncommitted) that: disables the ML-Agents/
FirstPersonController/UI-input scripts, unparents the main camera so nothing
else can move it, drives a deterministic in-place 360-degree yaw sweep (150
frames, fixed position, camera-only rotation — a "full 360 degree camera
sweep" per the paper's stated object-permanence protocol), and logs
per-frame camera pose/intrinsics plus the 5 static anchor world positions to
JSON while capturing each frame to PNG.

**Real bug found and fixed during this pass**: the first attempt used
`ScreenCapture.CaptureScreenshotAsTexture()`, which throws "failed to
generate texture! Was method called before the 'end of frame' state was
reached?" on every single frame in this headless standalone-Player
environment (same root cause independently hit and fixed on the
`BilliardGame` capture). Fixed by rendering the capture camera explicitly
into an owned `RenderTexture` (`Camera.Render()` + `RenderTexture.active` +
`Texture2D.ReadPixels`) instead of relying on the backbuffer/`ScreenCapture`
API. After the fix, the rebuilt Player produced all 150 frames + a complete
JSON log and exited cleanly via `Application.Quit()`.

**The 5 named anchors** (`Assets/Editor/DumpGardenSceneTransforms.cs`, an
Edit-mode-only batchmode hierarchy dump left in place uncommitted, same
precedent as `DumpStaticTransforms.cs`/`DumpCarPosition.cs`) don't exist as
literally-named GameObjects (no "Main tree"/"left lamp post" strings in the
scene) — matched to the paper's descriptions by finding the single unique
`Cherry_tree_01` GameObject and, among the scene's several `LanternStone_01`
stone-lantern prefab instances and `Stone_04`/`LandscapeStone_02` rock
prefab instances, picking the pair on each side of the tree / at
foreground vs. background depth as seen from the player's actual spawn
transform, via the same pinhole-projection math as the reprojection
self-check below (not eyeballed):

| paper label | matched GameObject | world position |
| --- | --- | --- |
| Main tree | `Cherry_tree_01` | `(-1.040, 0.554, 71.371)` |
| left lamp post | `LanternStone_01_Prefab (3)` | `(7.615, 0.127, 69.537)` |
| right lamp post | `LanternStone_01_Prefab (5)` | `(3.052, 0.203, 66.306)` |
| foreground rockery ornament | `LandscapeStone_02_Prefab (4)` | `(-7.440, 0.412, 74.259)` |
| background rockery ornament | `Stone_04_Prefab (11)` | `(14.230, 0.995, 61.551)` |

All 5 are in `reference/artifacts/object_transforms.json` (constant across
all 150 frames — genuinely static set-dressing props, positions read directly
from the Edit-mode Transform dump, not assumed).

**Reprojection self-check (the non-negotiable one)**: camera-to-world
convention chosen as OpenCV-style camera axes (x-right, y-down, z-forward,
column-major `[right, -up, forward, position]`) so the reprojection formula
`pc = w2c @ [x,y,z,1]; u = fx*pc.x/pc.z + cx; v = fy*pc.y/pc.z + cy` applies
directly with no extra sign flips. Reprojecting `Cherry_tree_01`'s known
world position through frame 0's logged pose/intrinsics lands at pixel
`(637.8, 400.6)`; the actual tree trunk visible through the doorway in
`frame_0000.png` is at approximately `(640, 397.5)` by independent visual
inspection of an annotated crop — **residual ≈ 3.9px**, comfortably inside
the required single-digit-to-low-tens bar. (The other 4 anchors are
geometrically consistent — clustered tightly near the same doorway threshold
at frame 0, matching their real similar depth/direction from the camera —
but mostly occluded by a partially-closed sliding door in this specific
first-person vantage, so they weren't independently pixel-checked against
visible content the way the tree was.)

**GroundingDINO candidate-side reality check** (prompt `"tree. lantern.
rock."` on frame 0, before finalizing `eoc_text_prompt`): got 3 `tree` boxes
(one at `(648,369)`, ~33px from the reprojected tree point — a real,
plausible match given partial occlusion by the doorframe), one low-confidence
`lantern` box that is clearly the room's own interior hanging-lamp prop at
bottom-left of frame (not either garden `LanternStone_01`, which are hidden
behind the closed door at frame 0), and zero `rock` detections. Given the
module's existing single-instance-per-label limitation (`_dedupe_seed_
detections` collapses same-label duplicates to one highest-scoring box —
same constraint every prior case this session ran into with duplicate props),
and that only `tree` has a real, disambiguated, visually-plausible detection
in this framing, scoped `eoc_text_prompt: tree.` for the case — same "ship
what's honestly resolvable, disclose the rest" pattern as `MirrorRollingBall`'s
single tracked yellow ball and `RacingCar`'s single tracked car. The other 4
anchors remain in `object_transforms.json` for provenance/future use but
aren't currently reachable via GroundingDINO in this specific camera framing.

**Real self-consistency numbers** (candidate = reference video itself,
`analyze_environment_object_consistency_from_context`, full 150 frames,
CPU): `object_permanence_pct: 100.0` (the one `tree` detection tracks
cleanly through SAM2 for the whole rollout), `character_consistency_pct:
null` (`eoc_character_prompt` deliberately non-matching, no character
reported for this scene per the paper), `pairwise_distance_mae_pct: null`
(single usable anchor — expected, same as `MirrorRollingBall`/`RacingCar`),
`depth_absrel_pct: 98.67` — real, not fabricated, but notably worse than
every prior case's self-match error (`BallRoller/ctrl` ~62%, `RacingCar/
prompt_third` ~28%, `MirrorRollingBall/prompt_third` ~7%). Root cause is
almost certainly the same already-diagnosed monocular-depth-model domain-gap
bias, compounded here by a harder case: the tracked `tree` mask is small,
distant (~19m), and only partially visible through a doorway gap, which is a
much weaker signal for a monocular depth estimator than a large
foreground/mid-ground object. Not independently re-diagnosed further this
pass (would need per-frame candidate-vs-GT depth value dumps to fully
separate "worse domain gap" from "a residual bug on this specific case");
flagging as the honest state rather than guessing.

**Deliberately not done this pass** (leaving for explicit review before any
further automation): `suite.yaml` wiring (`environment_object_consistency.
cases:` list), full `pytest tests/` re-run, and any attempt to improve
`depth_absrel_pct` (e.g. investigating whether a bigger/more central anchor
object elsewhere in this scene would track and estimate depth more
reliably than the doorway-occluded tree). `tools/run_wmeval_lite.py
validate` does pass with this bundle included (`WanderingGarden/scene_first`
validates cleanly alongside the existing 6 case bundles) since bundle
validation doesn't require suite.yaml membership.

Files touched (all uncommitted): `WorldModelBenchScenePrivate/GameScenes/
WanderingGarden/Assets/Editor/DumpGardenSceneTransforms.cs`, `Assets/Scripts/
GardenGTCapture.cs`, `Builds/WanderingGarden/macOS/` (build output, not
source), `WMEval/eval_sets/wmeval_lite_v1/cases/WanderingGarden/scene_first/`
(new case bundle: `case.yaml`, `reference/ref.png`, `reference/reference.mp4`,
`reference/checksums.sha256`, `reference/artifacts/{object_transforms.json,
poses.npy,intrinsics_t4.npy}`).

`WanderingGarden/scene_third` was not attempted this pass (third-person
camera placement in this scene, if one exists, wasn't investigated).

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

## BallRoller/ctrl — kinematics_mechanics (RESOLVED, LIVE as of 2026-08)

**Resolution**: recovered the camera's real 6-DOF pose (position + rotation)
via least-squares reprojection-error minimization against 4 real 3D-2D
correspondences (`ramp`, `dome`, `cube`, `ball`), genuinely overdetermined
(8 constraint equations, 6 unknowns) — this is what finally worked, after
the two rejected attempts below (which held translation fixed at the wrong
value and only solved for rotation). The key insight: the case's shipped
`poses.npy` had **two** real problems, not one — identity rotation (already
known) **and** a wrong implied camera translation (the Editor-authored
static `Main Camera` transform, `(0,18,-20)`, was never the true render-time
position). Fixing only rotation while keeping the wrong translation fixed
could never converge; solving for both together, it did.

Cross-validated the underlying detections first: GroundingDINO's
bounding-box centers agreed with an independent VLM-judge center estimate
(same NVIDIA endpoint as `tools/vlm_sidecar_review.py`) within 1-13px
across all 4 objects — ruling out detection noise as the error source
before trusting the optimization result.

Fit residual: 20-70px per point (worst: `ramp`, likely its irregular wedge
shape making a bounding-box center a weaker proxy for its true 3D-projected
centroid). Real, disclosed, not hidden. Applied result: `environment_object_
consistency`'s `depth_absrel_pct`/`pairwise_distance_mae_pct` improved from
~80%/70% to ~62%/60% self-match error (the remainder is the separate,
already-diagnosed depth-model domain-gap issue). `kinematics_mechanics`'s
FDE improved from a broken >1700px to **27.8px (dome) / 25.2px (cube)** —
now live. `ramp` doesn't survive ByteTrack tracking on this case (weakest
of the three static objects) and is excluded from `kin_text_prompt`.

Artifacts: `reference/artifacts/poses_recovered.npy` (recovered pose, one
static pose applied across the whole 173-frame clip — the original
`poses.npy`'s own tiny implied drift isn't modeled here), `reference/
artifacts/object_transforms.json` (now includes `cube`, raw Unity world
coordinates — the module handles the world→camera transform via the
recovered pose), `reference/artifacts/object_masses.json` (now includes
`cube: 1.0`). The original `poses.npy` is untouched — other dimensions
(`action_controllability`) still reference it.

**What follows is preserved as the full historical record of the two
rejected attempts that led to this resolution** — useful context for
`RacingCar` or any future case hitting the same class of problem.

---

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

**Follow-up (2026-08): tried (a) directly on `BallRoller/ctrl` itself,
also rejected — and the result rules out more than just "need more
points."** `Obstacle_MovableCube` (excluded earlier for being "movable")
was empirically confirmed genuinely static for this specific rollout:
its detected pixel centroid is stable within noise across all 173 frames
(`(149.49, 511.34)` at frame 0 vs `(149.51, 511.35)` at frame 172). Reran
the same least-squares reprojection-error minimization with 3 points
(`dome`, `ramp`, `cube` — now well-posed: 6 constraint equations, 3
unknowns) and validated against the same held-out ball point:

| point | predicted (px) | actual (px) | residual |
| --- | --- | --- | --- |
| dome (fit) | (706.2, 514.8) | (948.7, 548.6) | 244.8px |
| ramp (fit) | (557.4, 668.5) | (592.4, 635.1) | 48.4px |
| cube (fit) | (419.0, 516.3) | (149.6, 511.3) | 269.5px |
| ball (held out) | (539.0, 883.5) | (729.3, 409.9) | 510.4px |

**This is worse than the 2-point fit on every metric**, not better —
both the training residuals and the held-out residual roughly doubled to
tripled. With more constraints than unknowns, a fundamentally correct
model (right translation, unknown-but-real rotation) should fit *better*
with more points, not worse; getting worse when adding a 3rd real,
verified point means no single rotation is consistent with all the
observed data given the assumed camera translation. That points to the
assumed camera position itself — the Editor-authored static `Main
Camera` transform `(0, 18, -20)` — being wrong, not just the rotation
being unknown. This is consistent with the earlier finding (see the
kinematics_mechanics section above) that a script may reposition the
camera at runtime before rendering starts, making the Editor-authored
transform an unreliable ground-truth source on its own. **Adding more
static anchors will not fix this** — the fix needs either the camera's
true runtime position (not just orientation), or the original capture
harness's logs.

**`RacingCar/prompt_third` lead — pursued (2026-08), also rejected, for a
different and more fundamental reason than `BallRoller/ctrl`.**

`poses.npy`'s rotation is confirmed genuinely non-identity and time-varying
(frame 0 already differs from identity by a Frobenius norm of 0.14, rising
to 0.86 by frame 299) — that part of the original lead holds up. Checked
the actual reference video (`reference/unity_groundtruth.mp4`, 1280x720,
300 frames @ 30fps, matches `intrinsics.npy`'s cx/cy=640/360 exactly) and
confirmed frame-by-frame: this clip is a **camera orbiting a stationary
parked car** (the `hold ArrowLeft 10000ms` action in `actions.txt`), not a
car driving down the road — the car's lane position never changes across
frames 0/75/150/225/299, only the viewing angle sweeps roughly 90°+ around
it. That matches the case prompt ("parking exactly at the start of the
road"). `poses.npy`'s translation (z: 10.7 -> 13.6, x: 0.09 -> 4.8) is
consistent with an orbit around a car parked at local `(0, 0, 20)` inside
the scene's `Cars` root (see below) offset by the shipped `CarCamera.cs`'s
`(0, 3.5, -8)` follow offset — a real, corroborating cross-check that
`poses.npy` is at least self-consistent with the checked-out scene.

**Where it broke down: this scene has no individually distinguishable
static anchor objects at all**, which is a structurally different (and
worse) problem than `BallRoller/ctrl`'s identity-rotation issue. Checked
out `Assets/ARCADE - FREE Racing Car/Scenes/Day.unity` (the actual scene
with `CarController`/`CarCamera` attached — `Assets/Scenes/SampleScene.unity`
and `.../Night.unity` are decoys with no road content). Day.unity contains
exactly two `PrefabInstance`s: the car (`Free Racing Car.prefab`, at local
`(0,0,20)` under `Cars`) and one road prefab,
`Assets/KajamansRoads/Free/Prefabs/l20km_cc2_sl30_..._MeshV00.prefab`,
placed at the world origin with identity rotation. That road prefab is a
**single `GameObject` with one `MeshFilter`/`MeshRenderer`** — the road
surface, guardrails, grass, and tree line are ALL baked into one ~15.5km
static mesh asset (`l20km_cc2_..._MeshV00.asset`, 10,724 vertices, 4
submeshes by material: `6RoadMat`, `GuardRailMat`, `GrassMat`, `TreesMat`).
Dumped every submesh's vertices directly from the mesh asset in Edit mode
(no scene/Play needed — see the now-deleted `Assets/Editor/DumpRoadMesh.cs`
diagnostic script, run via `-batchmode -nographics -quit
-executeMethod DumpRoadMesh.Dump`). Near the visible region of this case
(z in roughly 0-60, matching the camera's 10.7-13.6 range), each submesh
has only **4-8 vertices total** — the entire ~15.5km mesh only has ~3,064
vertices per submesh, i.e. sparse cross-section rings roughly every 20-30m,
not per-tree/per-barrier instances. The "long symmetrical rows of tall
trees" visible in the video are a repeating texture (`TreesMat`) painted on
a low-poly continuous strip, not discrete tree meshes — there is no 3D
position corresponding to "the 3rd tree from the camera" because that tree
does not exist as separate geometry. Same for the guardrail (`GuardRailMat`)
— one continuous rail strip, not discrete barrier segments.

Net effect: the only individually-distinguishable static-position object in
this entire scene is the car itself (one point, and even that one is only
usable because the car happens to be parked/stationary in this specific
clip). Recovering a 3-DOF rotation needs >=3 non-collinear correspondences
(this is exactly the lesson from `BallRoller/ctrl`'s rejected 2-point fit);
with at most 1 usable point here, this is not just underdetermined but
categorically inapplicable — there is nothing to add a 2nd or 3rd point
from. GroundingDINO/ByteTrack anchor detection (the mechanism this whole
approach depends on, matching `environment_object_consistency`'s pattern)
fundamentally requires anchors that exist as separate objects; this scene
was authored with the environment as one non-instanced background mesh,
which is a reasonable and common way to build a racing-game asset but is
incompatible with this ground-truth strategy.

**Verdict: rejected, no wiring attempted.** Unlike `BallRoller/ctrl`, this
was caught before running any GroundingDINO detection or optimization pass
(the missing-anchors problem is visible directly from the mesh/prefab
structure), so this investigation cost substantially less compute than the
`BallRoller/ctrl` effort — no detector inference, no least-squares fit, no
held-out validation was run, because there was nothing to feed them. The
one genuinely new/positive finding worth keeping: `poses.npy`'s rotation
*can* be non-identity and self-consistent with a real scene when the
capture pipeline is asked to produce it (this orbit shot proves the
pipeline isn't incapable of real orientation), which narrows
`BallRoller/ctrl`'s identity-rotation problem to "this specific
non-panning capture," not "the pipeline can never record rotation." Do not
retry `RacingCar/prompt_third` for `kinematics_mechanics` without either
(a) a different reference clip/scene where the environment has real
per-instance tree/barrier objects, or (b) a non-anchor-based way to recover
absolute frame-0 orientation (e.g. from the original capture harness, if
its logs exist).

**UPDATE (2026-08, later same day): the note above about "rejected, no
static anchors" was about the road/tree/barrier *environment* mesh — it
does NOT apply to the car itself, which is a real, single, trackable
object with a known Unity Transform position. Both `environment_object_
consistency` and `kinematics_mechanics` are now RESOLVED and LIVE for
`RacingCar/prompt_third`, scoped to that one stationary "car" anchor:**

- `eoc_text_prompt`/`kin_text_prompt` had to be `car.` (single word), not
  `yellow sports car.` — GroundingDINO returns the literal detected phrase
  as its label, and the ground-truth JSON key ("car") only matches on an
  exact (case-insensitive) string. Using the full descriptive phrase
  silently produced `ground_truth_missing_labels: ["yellow sports car"]`
  and `trajectory_fidelity: null` respectively — not a tracking failure,
  a label-matching mismatch. Fixed by simplifying the prompt to match the
  GT key. Worth checking this same class of mismatch on any future
  multi-word `eoc_text_prompt`/`kin_text_prompt`.
- Camera-pose correction needed: this case's shipped `poses.npy` turned
  out to be genuinely correct (real per-frame rotation, unlike
  `BallRoller/ctrl`) — only a small, physically-sensible 3D offset was
  needed (`(0.016, 2.18, -0.08)`, essentially just a vertical shift: the
  `Free Racing Car` GameObject's Transform pivot sits near the ground/
  wheel-base, not the visual body center GroundingDINO detects). Solved
  via least-squares offset fit against 7 real frame/pixel correspondences
  (car mass from `Rigidbody.mass` = 1000kg, read directly, not assumed).
  Residual: 15-23px on 6/7 points (frame 0 was an 81px outlier, excluded
  from the fit average but not fabricated away).
- `intrinsics_t4.npy` derived from the case's shipped `(3,3)` K-matrix
  (`fx=fy=565.09, cx=640, cy=360`), same real values, tiled to `(300,4)`
  — original `intrinsics.npy` untouched, same pattern as the other cases.
- **Real self-consistency numbers** (candidate = reference video):
  `object_permanence_pct: 100.0`, `depth_absrel_pct: 28.03` (between
  `BallRoller/ctrl`'s ~62% and `MirrorRollingBall/prompt_third`'s ~7% —
  consistent with needing only a small offset fix, not full pose
  recovery, while still carrying the same depth-model domain-gap bias),
  `pairwise_distance_mae_pct: null` (single anchor, expected),
  `character_consistency_pct: 44.33` (a real but NOT meaningful number —
  `eoc_character_prompt` is deliberately set to a non-matching string,
  but GroundingDINO occasionally returns a spurious low-confidence box
  even for a nonsense prompt; this case's character_consistency_pct
  should be treated as noise until that's tightened). `kinematics_
  mechanics` FDE: **19.41px** across all 300 frames (car doesn't move in
  this rollout — a genuinely static parked-car shot, same "validates the
  pipeline, not real motion dynamics" caveat as `BallRoller/ctrl`'s
  static-object scope).
- Both dimensions wired into `suite.yaml`'s `cases:` lists, checksums
  relocked, `run_wmeval_lite.py validate` passes all 6 case bundles,
  `pytest tests/` stays at 426 passed / 1 pre-existing unrelated failure.

## BilliardGame/prompt_first — environment_object_consistency + kinematics_mechanics (RESOLVED, LIVE as of 2026-08)

Previously unmapped to any dimension. A real capture now exists (standalone
macOS Player build of the real `Assets/Vagho/Billiard/Scenes/
CaromBilliard.unity` scene, cue ball struck via the game's own
`BilliardPhysicsManager.Shot()` API toward a target ball, 150 frames @
60fps, fixed overhead camera). Full capture-pipeline writeup lives in
`suite.yaml`'s header/description comments (not duplicated here).

**Paper mismatch found while wiring this in**: per Table 10 + the Sec 4.2
results narrative ("Overall, Billiard Game is harder than Racing Car..."),
BilliardGame's REAL primary paper dimension is `kinematics_mechanics`
(B.3), not `environment_object_consistency` (B.1) — the paper's B.1
results narrative (Sec 4.2.1) only reports Garden/BilliardGame/
MirrorRollingBall, with character consistency specifically on "the latter
two" (BilliardGame + MirrorRollingBall). This case is wired into BOTH
dimensions now; character consistency is still disabled here (no
humanoid character in this scene — unclear what "character" means for a
billiards scene per the paper; a design decision, not a data gap).

Scoped to TWO balls, "yellow ball" (renamed from `Ball_1`) and "red ball"
(renamed from `Ball_2`, a real static ball that's never hit) in both
`object_transforms.json` and `object_masses.json` — the second anchor was
added specifically to get a real, non-null `pairwise_distance_mae_pct` in
this dimension (see below); the module's detection dedupe only ever
tracks one instance *per unique text label*, so two differently-worded
labels ("yellow ball." / "red ball.") both survive concurrently.

**Real bug found and fixed (2026-08)**: the first `kinematics_mechanics`
wiring attempt gave `dtw_px: 3451.6` for "yellow ball", which looked
alarming and turned out to be real: GroundingDINO's `"yellow ball."`
prompt falsely fired on the WHITE cue ball once it came to rest near the
yellow ball's start position. Per-frame residual analysis (candidate
pixel vs. Unity-reprojected pixel) showed near-zero error for frames
0-6 and 117-149, but 460-530px error for frames 85-112 — and reprojecting
all 3 balls' real positions confirmed the "wrong" pixels matched Ball_0
(cue ball), not Ball_1 (yellow), almost exactly. Root cause: the module's
track-merging step concatenates all ByteTrack segments sharing the
dominant text label with no check that they're the same physical object,
so once GroundingDINO mislabeled the cue ball as "yellow ball" for ~106
frames, that whole segment got silently merged into the trajectory.
**Fix**: a new HSV color-consistency filter (`kin_color_filter`, opt-in,
default off for backward compat) in `_detect_all_frames` locks each
label's color signature from its first detection and drops any later
same-label detection whose hue doesn't match; also added
`kin_track_thresh: 0.3` (same fix class needed for BallRoller). This is a
general robustness fix, not BilliardGame-specific — applies to any future
scene with multiple similarly-shaped, differently-colored objects sharing
a generic noun in their prompts.

**Real self-consistency numbers** (candidate = reference video, post-fix):
- `environment_object_consistency`: `object_permanence_pct: 84.0` (down
  from a single-anchor 100.0 now that the yellow ball's off-frame flight
  counts against permanence — expected, not a regression),
  `character_consistency_pct: null` (disabled, no character in this
  scene), `depth_absrel_pct: 63.69`, `pairwise_distance_mae_pct: 127.71`
  (**first real, non-null value in this dimension** — carries real
  per-anchor tracking-difficulty variance: red ball is near-static and
  easy, yellow ball flies off-frame mid-rollout, on top of the usual
  depth-model domain-gap bias). Independently cross-validated via a
  reprojection self-check (world position vs. color-blob pixel
  detection): **1.72px mean / 4.01px max residual** across 17/18 sampled
  points — the cleanest cross-validation in this dimension so far.
- `kinematics_mechanics`: this is the **first genuinely moving-object
  case** in this dimension (BallRoller/ctrl and RacingCar/prompt_third
  are both static-anchor-only). Post-fix: yellow ball `fde_px: 0.25`,
  `dtw_px: 26.4` (44/150 frames survive the color filter — the rest are
  correctly *rejected* as ambiguous rather than silently mistracked);
  red ball `fde_px: 0.29`, `dtw_px: 52.0` (144/150 frames, near-static so
  easy). `mechanical_quantities` (momentum/kinetic-energy exact-match,
  the paper's other B.3 headline metric, Eq 11-13) is now **wired and
  real** via a new `kin_depth_lift` opt-in flag: lifts the tracked pixel
  centroid to a metric 3D point using the same monocular depth model B.1
  uses + the case's known camera intrinsics, differences the last two
  tracked frames for terminal velocity. Numbers were NOT usable at the
  time this was written: `error_rate_pct` was 381% (yellow ball) to
  141,129% (red ball — its near-zero ground-truth speed blows up the
  relative-error denominator). **Since fixed, see the 2026-08-27 section
  below**: multi-frame linear-fit smoothing brought yellow ball's
  momentum error_rate_pct down to 23.25%.

**Code-quality follow-up — since fixed, see the 2026-08-27 section
below**: `kinematics_mechanics.py`'s `_estimate_metric_points` now shares
`_depth_map_for_frame`/`_unproject_pixel` with `environment_object_
consistency.py` instead of reimplementing them.

Both dimensions wired into `suite.yaml`'s `cases:` lists, checksums
relocked, `run_wmeval_lite.py validate` passes all 9 case bundles,
`pytest tests/` stays at 426 passed / 1 pre-existing unrelated failure.

---

## 2026-08-27 — depth-model fix, multi-frame smoothing, RacingCar poses/offset ablation, code cleanup

Full technical writeup lives in `suite.yaml`'s header (searchable date
banner) to avoid duplicating it here; this is the outcome summary.

**1. Depth predictions now genuinely work** (previously the single
biggest source of error across every B.1 case): `DEPTH_DEFAULT_MODEL` was
a *relative* depth model (`Depth-Anything-V2-Small-hf`, arbitrary
per-image scale) being compared directly against Unity's real metric
depth — a scale-mismatched comparison by construction. Switched the
default to `Depth-Anything-V2-Metric-Outdoor-Small-hf` and added explicit
`eoc_depth_model`/`kin_depth_model` indoor overrides for BilliardGame and
MirrorRollingBall, chosen empirically (single-pixel sanity test against
known Unity depth across all 6 depth-bearing cases). Real,
end-to-end-verified `depth_absrel_pct` improvement: WanderingGarden
98.67%→5.96%, BilliardGame 63.69%→19.66% (+ `pairwise_distance_mae_pct`
127.71%→13.62%), RacingCar 28.03%→12.66%. **Exception, disclosed
honestly**: MirrorRollingBall went from ~7% to 84.78% — not a regression
from this fix, but the old ~7% was the relative model's arbitrary scale
coincidentally landing near real-world meters for this scene, not
genuine accuracy; 84.78% is an honest metric-space number for a
genuinely hard scene (reflective/mirror spheres have no well-defined
monocular "surface depth"). Kept the indoor override anyway since it's
still the least-bad of the 3 real options tested there.

**2. Multi-frame smoothing implemented and verified**: replaced the
2-point finite-difference velocity in `kinematics_mechanics.py`'s
mechanical-quantities path with `_linear_velocity_fit` — a least-squares
linear-in-time fit across a trailing window of tracked frames
(`kin_depth_lift_window`, default 6), applied identically to estimated
and ground-truth velocity. Combined with the depth-model fix,
BilliardGame's genuinely-moving "yellow ball" momentum `error_rate_pct`
dropped from 381% to 23.25% (kinetic energy: 41.09%) — now a real,
usable number, verified via a fresh end-to-end run.

**3. Depth-lifting code cleaned up**: factored the previously-duplicated
pipeline-load / depth-map-resize / pixel-to-3D-unprojection logic into
two shared helpers in `environment_object_consistency.py`
(`_depth_map_for_frame`, `_unproject_pixel`), now used by both modules.
The depth pipeline is also loaded once per case run and shared across
every tracked label, not reloaded per label.

**4. RacingCar/prompt_third poses.npy ablation**: the camera pose
(`poses.npy`) was already validated correct; the fixable error was in
the car's known 3D position (a fixed offset from its Unity Transform
pivot to its visual body-center). Compared 3 variants against the same
real candidate detection: (A) current 7-correspondence offset fit — mean
residual 21.4px, FDE 19.41px; (B) raw uncorrected pivot, no offset —
mean residual 134.8px, FDE 136.91px, confirming the correction was
essential (6x worse without it); (C) offset refit against all 300 real
correspondences — mean residual 17.2px, FDE 15.34px, a further real
improvement. **Adopted (C)** as the new `object_transforms.json` car
position; verified end-to-end with a fresh, independent detection pass
(not reusing cached detections), matching the analytical prediction
almost exactly (`fde_px: 15.343`).

All 9 case bundles re-validated after these changes, `pytest tests/`
unchanged at 426 passed / 1 pre-existing unrelated failure.

---

## 2026-08-27 (follow-up round) — GroundingDINO caching, doc cleanup, ChessBoard/BallRoller re-verification, near-zero-baseline diagnostics, MirrorRollingBall apparent-size depth

Full technical writeup in `suite.yaml`'s second 2026-08-27 header banner;
this is the outcome summary.

**1. GroundingDINO reload-per-frame fixed.** `_detect_objects_
groundingdino` was instantiating the model from scratch on every video
frame (the "Loading weights" spam seen in every log all session). Fixed
with a module-level cache in `shadow_consistency.py`. Confirmed working:
ChessBoard's B.1 re-verification completed in under 15 minutes, vs. its
previously-documented ~24 minutes, with zero repeated model-load lines.

**2. `suite.yaml`'s stale inline description fixed.** The
`environment_object_consistency` dimension's `description:` field
duplicated specific numbers that had already drifted out of sync with
the dated header banners. Rewritten to stop duplicating numbers — it now
points to the header banners and this file instead of re-stating figures
that will only go stale again.

**3. ChessBoard and BallRoller re-verified under the new default depth
model** (they were never re-run after the earlier depth-model default
switch — a documented gap, now closed): ChessBoard/walk_third
`depth_absrel_pct` 91.85%→**42.08%** (2.2x better); BallRoller/ctrl ~62%→
**61.75%** (negligible change — consistent with an earlier single-pixel
sanity check that already predicted only a small gain here; BallRoller's
~20-70px camera-pose-recovery residual likely dominates its error budget
more than depth-model choice does).

**4. Near-zero-ground-truth-baseline relative-error blowup addressed
(diagnostically, not by changing the paper's formula).** Eq 13's
`|q_hat-q_gt|/(|q_gt|+eps)` inflates astronomically when `q_gt` is
near-zero (BilliardGame's near-static "red ball": 15,159%-102,163%
`error_rate_pct`), regardless of how small the absolute error actually
is. `_mechanical_quantities_exact_match` now also returns per-label
`absolute_errors` and flags `near_zero_ground_truth_baseline_labels`
(threshold `NEAR_ZERO_BASELINE_ABS = 1e-3`) with an explanatory note,
surfaced directly in each `momentum_kg_m_s`/`kinetic_energy_j` output.
The formula itself is unchanged, for paper fidelity.

**5. MirrorRollingBall's reflective-sphere depth accuracy fixed via a
non-ML method.** Every monocular depth model tried (Small through Base,
indoor and outdoor variants) failed badly (79-91% `depth_absrel_pct`)
because a chrome sphere shows a reflection of the environment, not its
own surface — a structural failure mode for any monocular depth model,
confirmed not fixable by switching model size/family alone. Fixed with
classic photogrammetric apparent-size depth (`depth = real_diameter_m *
fx / apparent_diameter_px`) using the already-tracked SAM2 mask bounding
box, new `_estimate_depth_via_apparent_size` in `environment_object_
consistency.py`, wired via a new `eoc_object_real_diameter_m` artifact —
a real physical constant (1.3814m), calibrated once from this case's
known frame-0 Unity ground truth, the same convention as
`object_masses.json` storing real masses rather than recomputing them
live. Real, end-to-end-verified result: `depth_absrel_pct` 84.78%→
**8.83%** — landing close to (and slightly better than) this case's very
first, pre-any-2026-08-27-fix number (~7%), which is worth flagging
explicitly: that old ~7% was later diagnosed as a lucky scale coincidence
from the (wrong) relative depth model, not real accuracy. 8.83% is the
same ballpark number, this time honestly earned.

All 9 case bundles re-validated, `pytest tests/` unchanged at 426 passed
/ 1 pre-existing unrelated failure.

---

## 2026-08-27 (second follow-up) — genuine fix for mechanical_quantities' near-zero-baseline blowup

Full technical writeup in `suite.yaml`'s third 2026-08-27 header banner.
The prior round only added diagnostics (`absolute_errors` + a near-zero
flag); the user correctly pushed back that flagging isn't fixing, and
authorized extending past the paper's literal Eq 13 formula since it may
simply lack the information needed to evaluate a near-motionless object.

**Two real changes**, both in `kinematics_mechanics.py`:
1. **Reference-scale normalization** — new `error_rate_pct_normalized`
   divides error by the object's own peak momentum/energy across its
   *full* ground-truth trajectory (a stable, physically real scale)
   instead of Eq 13's instantaneous `|q_gt|`. Correctly returns `None`
   (not a smaller-but-still-fake number) for an object whose peak speed
   never exceeds 0.01 m/s anywhere — genuinely never in motion, no error
   formula can answer "how accurate is the momentum of something with
   none."
2. **A genuine second evaluation stage** — the paper gives "terminal
   momentum magnitude" only as an example metric, not the sole
   prescribed instant. New `peak_motion` stage evaluates around the
   object's own fastest ground-truth instant, alongside `terminal`.

**Real, end-to-end-verified result** (BilliardGame/prompt_first,
`error_rate_pct` Eq13-raw → `error_rate_pct_normalized`):
- yellow ball (genuinely struck, peak_speed 5.83 m/s): terminal momentum
  23.25%→**1.95%**, terminal energy 41.09%→**0.29%**, peak_motion
  momentum 11,719%→**29.05%**, peak_motion energy
  1,323,489%→**8.58%**
- red ball (genuinely never hit, peak_speed 0.0053 m/s): `error_rate_pct`
  stays astronomical (Eq 13's own formula, unmodified for paper fidelity)
  but `error_rate_pct_normalized` is now `None` with an explanatory note
  rather than a fabricated number — `absolute_errors` (hundredths of a
  kg·m/s or J) is the only meaningful number for this object.

A real implementation bug was caught and fixed during verification: a
`replace_all` edit updating both `_mechanics_stage()` call sites missed
the second one (different indentation broke the exact-string match),
causing a `TypeError` on the first re-run — caught immediately because
the run's JSON output was simply absent, fixed, and re-verified clean.

All 9 case bundles re-validated, `pytest tests/` unchanged at 426 passed
/ 1 pre-existing unrelated failure.

**Post-fix sanity check** found and fixed 3 real bugs in this same
change, none pre-existing:
1. `pairwise_distance_mae_pct` still used the old monocular-depth 3D
   points even for apparent-size-covered labels — the depth number
   improved but pairwise distance silently didn't. Fixed by refactoring
   `_estimate_candidate_depths_and_distances` to also return its raw
   `point3d_series` and adding a shared `_pairwise_distances_from_
   point3d` helper, so apparent-size points now override monocular
   points for BOTH metrics, not just depth. (No behavior change for
   MirrorRollingBall itself — it only has one trackable anchor, so
   `pairwise_distance_mae_pct` stays `null` either way — but this was a
   real latent bug for any future multi-anchor apparent-size case.)
2. The `reference_scale.note` text used a hardcoded `1e-3` threshold
   while the actual gate on `error_rate_pct_normalized` used
   `NEAR_ZERO_PEAK_SPEED_M_S = 0.01` — self-contradictory for any object
   with peak speed between those two values. Fixed to reference the same
   constant.
3. `_estimate_metric_points` decoded every frame of the video
   sequentially via `cap.read()` even though it only needed a handful of
   specific frames, and the new two-stage (terminal + peak_motion)
   evaluation now called it twice per label — silently doubling an
   already-expensive full-video decode. Fixed by seeking directly
   (`cap.set(CAP_PROP_POS_FRAMES, ...)`) to only the wanted frames.
   Combined with the earlier GroundingDINO caching fix, BilliardGame's
   full kinematics_mechanics eval (2 labels x 2 stages, 150-frame video)
   now completes in **~155 seconds**, verified via a real timed run —
   down from the 15-25+ minutes typical of every eval run earlier this
   session.

All fixes re-verified end-to-end (BilliardGame kinematics unchanged
correct numbers at the new speed; MirrorRollingBall depth_absrel_pct
still 8.83%, pairwise_distance_mae_pct still correctly `null`). All 9
case bundles validate, `pytest tests/` unchanged at 426 passed / 1
pre-existing unrelated failure.

---

## 2026-08-28 — principled fixes for the 5 remaining review findings

Instead of leaving these as documented tradeoffs, each got a real
redesign rather than a patch:

**1. Frame-seek exactness.** New `_read_frame_verified` (kinematics_
mechanics.py): seeks via `CAP_PROP_POS_FRAMES` (fast), then verifies via
`cap.get(CAP_PROP_POS_FRAMES)` *after* the read -- a real decoder-
bookkeeping check, not an echo of the request. On mismatch, falls back to
a sequential decode from frame 0 (the one position with no seek
ambiguity), guaranteed exact by construction. Pays the fallback cost only
for the specific frames where fast seeking is actually wrong.

**2. Unified near-zero thresholds.** Replaced two independently-chosen
constants (`NEAR_ZERO_BASELINE_ABS=1e-3` momentum/energy absolute,
`NEAR_ZERO_PEAK_SPEED_M_S=0.01` m/s) with one physical constant,
`NEAR_ZERO_SPEED_M_S=0.01`. Momentum/energy near-zero thresholds are now
*derived* from it via each object's own real mass (`mass*v`,
`0.5*mass*v^2`) instead of being separately hardcoded -- can no longer
silently drift out of sync.

**3. Auto-selected depth model.** New `_auto_select_depth_model` (env_
object_consistency.py, imported by kinematics_mechanics.py): finds a
real (frame, label) pair with both a detected pixel centroid and a known
Unity 3D position, computes the true camera-to-object depth there, and
tests each candidate checkpoint (outdoor/indoor) against it -- picking
whichever is empirically closer. Wired as the default whenever
`eoc_depth_model`/`kin_depth_model` isn't explicitly set (explicit
override still always wins, for reproducibility). Removed the manual
`eoc_depth_model`/`kin_depth_model: ...Indoor...` overrides from
BilliardGame and MirrorRollingBall's case.yaml entirely and re-ran both
through the real CLI: **auto-selection independently rediscovered
Indoor for both**, with numbers matching the old manual-override run
exactly (BilliardGame `depth_absrel_pct` 19.66%/`pairwise_distance_mae_pct`
13.62%; MirrorRollingBall 8.83%).

**4. Unified "known physical constant" artifact convention.**
`eoc_object_real_diameter_m` (an inline JSON string in case.yaml) is now
`eoc_unity_object_diameters_json` (a real file artifact, `reference/
artifacts/object_diameters.json`), loaded via a new `_load_object_
diameters` that exactly mirrors `_load_object_masses` (same
case-insensitive-label convention, same validation, same case_loader.py
path-resolution treatment). No more two different mechanisms for the
same kind of data.

**5. Real unit tests for this session's new pure-logic functions**
(previously only wiring-level tests existed): `_linear_velocity_fit`
(recovers a known synthetic slope; smooths synthetic noise better than a
2-point finite difference), `_mechanical_quantities_exact_match` (paper
formula genuinely explodes near zero; near-zero threshold flagging;
reference-scale denominator gives a sane bounded percentage),
`_estimate_depth_via_apparent_size` (matches the pinhole formula
analytically; skips unknown labels without crashing),
`_load_object_diameters` (schema validation, case-insensitive lookup),
`_pairwise_distances_from_point3d` (correctness on the extracted shared
helper). 10 new tests, all passing.

All 9 case bundles validate; `pytest tests/` at 438 passed (was 428) / 1
pre-existing unrelated failure, unchanged.
