# Tables 4 and 7 execution assets

This directory is the machine-readable execution contract for the paper's
Table 4 and Table 7 rerun. `cases.yaml` contains all 16 table cells: four
kinematics/mechanics cells and twelve consistency cells. The seven semantic
trajectory YAML files are the authoritative controls; generated repo-native
`action.npy`, human-readable `actions.txt`, self-describing `action.npz`, and
browser `actions.jsonl` files are derivatives.

Remaining-model scope and readiness are tracked in [`remaining_models.yaml`](remaining_models.yaml).

## Alignment status

The scene roster, views, action-space dimensions, metrics, and five-generation
aggregation match the paper. The paper source does **not** publish the numeric
control timeline. These trajectories are consequently marked `reconstructed`,
not `exact`. Strict preflight fails until the original timelines are recovered
and their hashes recorded.

The existing Lite `action.npy` files are model-adapter controls, not always the
native Unity action vectors. For example, both RacingCar Lite cases contain a
constant five-channel array and describe a camera turn, whereas the paper and
`CarAgent.cs` define the scene action as continuous steering and throttle plus
a discrete brake branch. The Table 4 semantic trajectory therefore uses the
three native scene channels and compiles a separate keyboard derivative for
Project Genie; it does not reinterpret the five-column Lite adapter array as
Unity ground truth.

The paper has an additional internal conflict: Table 4 reports first- and
third-person BilliardGame cells while the scene appendix specifies one fixed
overhead camera. The reconstructed suite now covers both published view rows,
using one browser-expressible W/A/D direct-ball trajectory in both views so
viewpoint is the only intended independent variable. Unity replays the same
semantic impulses through its real billiard physics API. This is a user-selected
reconstructed protocol, not a claim that the paper used direct ball control
instead of its documented spin-aware cue shot. The hosted Genie interface still
exposes navigation keys rather than a guaranteed ball-force API, so action-target
ambiguity is measured and reported rather than treated as successful ball control.

Table-4 Unity exports now include `plane_calibration.json`: static visual
anchors, the horizontal motion plane, camera poses/intrinsics, and metric X/Z
coordinates. Generated-image rays are intersected with that plane using the
timestamp-matched Unity camera, while optical-flow anchor survival is reported
as camera-divergence QA. This yields DTW/FDE in metres. Exact paper parity is
still unavailable because the paper does not publish its scale transform
`S` or exact DTW normalization.

Other WMEval dimensions are only partially native-trajectory-ready. The current
audit finds 15 of 19 Lite cases natively machine-readable. The locked Billiard
and Garden bundles are underspecified but now have compiled Table 4/7
supplemental reconstructions. `ChessBoard/walk_third` now has an exact
supplemental 90-frame trajectory compiled from its documented Unity capture
schedule. `SunRisesAndSets/lighting` now has a compiled one-channel semantic
orbit derived from its locked 10-second browser schedule; the unpublished
Unity controller gains remain explicitly reconstructed. Thus every case has a
numeric trajectory source, but only 15 are native to the locked Lite bundles
and exact paper parity is not implied for the four supplements.

Regenerate the repository-wide readiness inventory with:

```bash
uv run python tools/trajectories/audit_trajectory_readiness.py \
  --output experiments/protocol/other_trajectory_inventory.json

uv run python tools/trajectories/prepare_other_trajectories.py
```

## Validate and compile

```bash
uv run python tools/trajectories/prepare_table_trajectories.py audit
uv run python tools/trajectories/prepare_table_trajectories.py audit --strict
uv run python tools/trajectories/prepare_table_trajectories.py compile
```

The ordinary audit proves structural coverage and reports parity caveats as
warnings. Strict audit converts every unrecovered timeline, missing frame, or
uncalibrated/unsupported adapter into an error.

`paper_baseline.yaml` locks the exact TeX source hashes, five-run/sample-std
aggregation, all published Project Genie values, metric directions, N/A cells,
and the paper's unresolved internal conflicts. Create the result-ingestion
template and, later, the JSON/CSV/Markdown cross-comparison with:

```bash
uv run python tools/paper_tables/compare_table_results.py template
uv run python tools/paper_tables/compare_table_results.py compare \
  --results experiments/protocol/rerun_results.template.json \
  --output runs/tables_4_7/comparison
```

Comparison fails closed unless every applicable metric has numeric `run0`
through `run4` values. `--allow-incomplete` is available only for rendering a
clearly marked pending preview.

## Project Genie computer-use controller

The controller uses a dedicated Chrome profile and attaches over the Chrome
DevTools Protocol. It uses accessible labels/roles rather than private APIs,
keeps authentication user-owned, fills both Project Genie prompt fields,
uploads the exact Unity frame-zero PNG, replays the compiled keyboard program,
downloads the native video, and writes screenshots plus a timestamped
provenance receipt.

The table runner uses one calibrated, fixed-horizon yaw policy. A 45-degree
turn holds the requested arrow key for 874 ms and then rests; a full turn holds
it for the validated 6994 ms visual period and then rests for 3006 ms. Both
therefore cover the same 10-second horizon, and right turns use the documented
opposite-key symmetric-period assumption. `--closed-loop-yaw` remains a
diagnostic controller option, but it is not used for table runs because noisy
visual return detections can stop early or span multiple revolutions. The
downloaded UI-free video is aligned automatically to nine ordered browser
checkpoints. Only a chronological, high-confidence mapping is trimmed to
`trajectory_native.mp4` and marked `native_action_aligned_ready`.

```bash
# Environment and Chrome/CDP readiness
uv run python tools/generation/project_genie_agent.py doctor
uv run python tools/generation/project_genie_agent.py launch

# Compile controls first
uv run python tools/trajectories/prepare_table_trajectories.py compile

# Launch the dedicated Chrome profile, then execute one independent run
uv run python tools/generation/project_genie_agent.py run --launch \
  --first-frame lite/cases/RacingCar/prompt_third/reference/ref.png \
  --actions experiments/protocol/compiled/racing_car_drive.actions.jsonl \
  --environment-prompt "A wide straight highway lined with trees." \
  --character-prompt "A yellow sports car with two black racing stripes." \
  --view third_person --repeat-id run0 \
  --output-dir runs/project_genie/table4/RacingCar/third/run0
```

On first use, sign in and satisfy any subscription/consent screen in the
opened browser, then rerun with `--resume`. The controller does not automate
credentials or bypass access controls. Project Genie does not expose RNG seed
selection in this web UI, so `run0` through `run4` are independent generation
IDs rather than claimed seeds.

When the operator has explicitly authorized automatic acknowledgement of the
image-upload notice, pass `--auto-accept-upload-policy`. The controller clicks
only when both the exact `Prohibited Use Policy` link and exact `I agree`
button are visible, and records the acceptance time and mode in `receipt.json`.
The flag is available on both `project_genie_agent.py run` and
`run_project_genie_tables.py run`; it does not automate sign-in, subscription,
age, or any other dialog.

Project Genie is an experimental UI and selectors can change. On an unknown
screen the controller stops, saves `blocked.png` and a sanitized accessible
control inventory in `receipt.json`, and leaves the dedicated Chrome session
available for inspection.

`--reuse-current-world-from` is for controller calibration and debugging. Its
receipt is explicitly marked as a revisited world and the queue refuses to
count it as one of the paper's five independent generations.

Resume is fail-closed on both the encoded live-world prompt and perspective;
an unrelated open world is never reused merely because it has the same view.
If action replay completed but native export failed, `--allow-missing-native`
revalidates every captured-frame hash and the full action trace, then promotes
the action-aligned browser video. A
receipt that was already complete but later fails provenance validation is
always regenerated as a new world rather than resumed from the currently open
generation.

Chrome can occasionally finish a valid `Genie*.mp4` in `~/Downloads` just
after the UI reports `Download failed`. The controller now watches that folder
for a new, stable file after an export click and copies it into the run before
alignment. A file left by an older session can be recovered explicitly only
after nine-checkpoint visual validation:

```bash
uv run python tools/generation/recover_project_genie_download.py \
  --source ~/Downloads/Genie….mp4 \
  --run-dir runs/project_genie/table7/WanderingGarden/360_out/run0 \
  --endpoint-mode calibrated_visual_periodicity
```

Recovery copies rather than moves the source and records its path and SHA-256
in `receipt.json`.

Plan or resume the full five-run queue with:

```bash
uv run python tools/paper_tables/run_project_genie_tables.py plan
uv run python tools/paper_tables/run_project_genie_tables.py run --resume
```

The authenticated September 2026 UI has been exercised end to end: creation
entry, exact image upload, environment and character prompt entry, perspective
selection, generation, keyboard control, revisit, and native download. Its
central creation card is exposed only through the full-page canvas. Pass
`--allow-canvas-entry` to explicitly enable the checkpointed,
viewport-relative entry sequence; ordinary form controls remain role/label
selected.

The manifest contains 80 provenance slots; the active plan contains 70 after
the user-requested exclusions of Table 4 BilliardGame third-person and
RacingCar first-person. Reconstructed orientation-specific Unity frames
now exist for every Table 7 cell, including all four Billiard cells. They
unlock reconstructed-setting execution while retaining an explicit parity
caveat because the paper's exact In/Out anchors are unpublished. A scene- or
case-level `--first-frame-override KEY=PATH` can replace a reconstructed frame.
Both active Table 4 cells have five independent, hash-matched Project Genie
runs. BilliardGame uses only the active first-person reconstructed WASD
direct-ball trajectory, and RacingCar uses the third-person reconstructed
steer/throttle/brake trajectory. RacingCar first-person generation is archived,
but the cell is excluded from active evaluation because the camera-fixed ego car
requires unavailable video-derived ego motion.

## Private Unity scene captures

The eight reconstructed scene/view cells are described in
`unity_scene_runs.yaml` and can be rebuilt, captured, and exported from
`WorldModelBenchScenePrivate` with one command:

```bash
uv run python tools/trajectories/run_unity_table_scenes.py all
```

This emits per-cell precompiled shell scripts, deterministic players, frame
sequences, frame-zero conditioning PNGs, videos, poses, intrinsics, object
transforms, and SHA256 receipts under
`runs/tables_4_7/unity_scene_runs/`. Flat or empty captures fail the export
gate. The eight cells are RacingCar first and third person; BilliardGame Table 4 first
and third person; and BilliardGame's four Table 7 trajectories.

## Unity Garden ground truth and first-frame conditioning

The checked local Garden player includes a deterministic 150-frame, 15 fps,
360-degree camera capture. Capture and validate its camera matrices, zero
translation, yaw rate, frame count, and exact frame-zero hash with:

```bash
.venv-unity/bin/python tools/trajectories/unity_trajectory_runner.py \
  --output-dir runs/unity/garden_bridge_gt capture-ground-truth
```

The validated frame-zero PNG is now the manifest default for inward
WanderingGarden cases (no JPEG conversion or crop):

```bash
uv run python tools/paper_tables/run_project_genie_tables.py run --resume \
  --case table7/WanderingGarden/360_in \
  --turn-calibration table7/WanderingGarden/360_in=runs/project_genie/calibration/garden_bridge_arrow_left_attempt4/turn_calibration.json
```

The validated controller smoke run used that exact PNG (SHA256
`195a2384dac3aad0aea93b508d381b8162b37b3e7a60b8e68a047698593ae742`),
an empty character prompt, and first-person mode. Its nine native/browser
matches are strictly ordered with timeline-fit `R^2 = 0.9991`; the derived
UI-free clip contains 129 frames at 24 fps. Because that run deliberately
revisited a generated world during calibration, it is not table-repeat
eligible.

The exact Unity-rendered generation inputs are pinned under
`experiments/protocol/conditioning_frames/`: `garden_inward.png` has the
hash above and `garden_outward.png` has SHA256
`60321e3b5bd810f12087f2029ba3c26757a4883df0d2cbc40dcb316fc12b89a3`.
They remain immutable because the completed browser-run receipts bind to those
bytes. The Unity evaluator capture may be rebuilt independently at the same
camera poses; its export gate requires square-pixel intrinsics (`fx == fy`).
This separation prevents a harmless renderer rebuild from changing generation
provenance while still allowing corrected pose/intrinsics ground truth.

Five fresh independent generations (`run0`--`run4`) are now validated for this
cell. For example, `run1` returned at 8.643 seconds on the browser clock and
its nine ordered checkpoints align to native frames 35--197 (`R^2 = 0.99948`). Its
UI-free evaluation clip contains 163 frames at 24 fps. `run3` exercises the
non-affine-clock case: all nine confidence-qualified matches are strictly
ordered, with bounded local native/browser speed ratios of 0.147--1.184, while
the diagnostic linear fit is only `R^2 = 0.96754`. The v2 aligner therefore
uses strict chronology, local time-warp bounds, match confidence/margins, and
semantic endpoints as gates; it reports linear `R^2` but does not assume that
Playwright screenshot time and native export time are affine. Each accepted
clip is marked `native_action_aligned_ready`. These are valid
reconstructed-setting repeats, not exact-paper-parity claims, because the
original prompt and numeric action timeline are unpublished.

The earlier calibration-only `run0` is preserved under
`run0_calibration_revisit`; `tools/generation/relocate_project_genie_run.py` rewrites its
internal absolute paths and refreshes dependent hashes. The canonical `run0`
slot was regenerated independently, so calibration evidence is retained
without contaminating the five table repeats.

The capture receipt and manifest also record frame 75 as a 180-degree
opposite-yaw candidate for the outward orientation. The automated queue uses
that frame for the reconstructed-setting outward cells and records
`paper_orientation_anchor_unpublished` as a caveat, because the paper does not
identify the orientation anchor explicitly. An operator can replace it with a
case-specific `--first-frame-override` if the authors' original anchor becomes
available.

The paper does not publish the exact Project Genie environment/character
prompt strings. The prompt components in this directory are therefore
reconstructed settings and remain a paper-parity caveat even though the web
fields and their receipt provenance are fully automated.

## Prepare every automatable asset

Materialize a self-contained, hash-locked asset bundle and an 80-slot
generation queue with:

```bash
uv run python tools/paper_tables/prepare_table_assets.py
```

The command copies every source-backed conditioning frame and all five forms
of each compiled trajectory, derives MirrorRollingBall's outward image from
frame 135 of its Unity reference orbit, and validates that frame against the
recorded camera poses. It writes `asset_manifest.json`, `asset_status.csv`,
`generation_queue.jsonl`, `evaluation_queue.jsonl`, per-case JSON receipts, and an executable
`run_ready_project_genie.sh` under `runs/tables_4_7/prepared_assets`.

The generated runner opts into reconstructed controls with
`--allow-nominal-adapter`; this changes the claim scope, not the underlying
paper-parity status. All Table 7 turns use the same recorded browser timing
calibration and fixed 10-second horizon. The current queue has 70 ready repeats
and 10 explicitly excluded repeats across all 16 provenance cells;
the four Billiard Table 7 cells now use reconstructed private-Unity views.
The evaluation queue separately records paper-parity caveats while keeping all
14 reconstructed-protocol cells generation-runnable. Ten Table 4 repeats are
metric-runnable; RacingCar first-person and BilliardGame third-person are
excluded. Table 4 uses Unity camera
poses, intrinsics, motion planes, object trajectories, and simulator masses for
metric DTW/FDE plus mechanical Error Rate; Exact Match is intentionally omitted
from the reported contract. Mechanics uses a bounded, persistence-gated peak
event finite difference so small action/event offsets do not mix moving and
stopped samples. Visual-anchor flow is retained as camera-divergence QA.
Five-run preparation enforces one video source type per
cell so native exports and browser captures can never be mixed in a variance
calculation. MirrorRollingBall now locks two foreground yellow
spheres (`YellowBall (30)` and `YellowBall (32)`) at a 4.0 m separation, so
pairwise-distance MAE has the required fixed-anchor source. All four Mirror
five-run cells have been reevaluated against that anchor and now contain
numeric distance results for every repeat.

Audit every completed browser trace against this policy, including dispatched
key timing, checkpoint coverage, artifact hashes, repeat independence, and
cross-run consistency, with:

```bash
uv run python tools/paper_tables/audit_project_genie_trajectories.py \
  --output runs/tables_4_7/project_genie_trajectory_audit.json
```

Invalid runs are listed as selective regeneration candidates; the audit itself
does not modify source runs.

Audit Table 4 channel states, timing, cross-view action hashes, Unity inputs,
conditioning-frame hashes, five-repeat execution, and semantic action-target
gaps with:

```bash
.venv/bin/python tools/paper_tables/audit_table4_action_coverage.py
```

The report is written to
`reports/tables_4_7/project_genie/table4_action_coverage/`.

The frozen Table 4/Table 7 metric and aggregation contract is
`evaluator_protocol.yaml`. Repository-wide conventions and remaining gaps for
all eight evaluation dimensions are recorded in
`../evaluation_dimension_standardization.yaml`; missing metrics are always
explicitly null with a reason and are never zero-filled.

When detections and tracks are already cached, mechanics-only iterations can be
reproduced without rerunning GroundingDINO:

```bash
.venv/bin/python tools/paper_tables/recompute_table4_event_mechanics.py
.venv/bin/python tools/paper_tables/build_table4_calibrated_results.py
```

The first command writes new provenance-linked reports and never mutates the
source evaluation rows.

After (or while) the browser runner fills the queue, one resumable command
automatically detects complete five-run groups, validates their receipts,
standardizes the videos, remaps the trajectory-specific camera ground truth,
and runs the available diagnostic evaluator:

```bash
.venv/bin/python tools/paper_tables/run_prepared_table_evaluations.py --device auto
```

`auto` uses CUDA, then MPS, then CPU according to the installed PyTorch runtime.
Use `--prepare-only` to materialize all currently ready inputs without running
the GPU evaluator. Incomplete groups are recorded as waiting and unsupported
cells remain explicit blockers in
`runs/tables_4_7/prepared_assets/evaluation_automation_status.json`.
For a single resumable entrypoint, run
`runs/tables_4_7/prepared_assets/run_full_pipeline.sh`; it executes the browser
queue first, then evaluates every completed group, and returns nonzero while
any prepared cell remains incomplete.
Every successful Table 7 cell also refreshes the cumulative published-versus-
rerun table under `reports/tables_4_7/project_genie/consolidated/`; only
pose-corrected, five-row-complete result files are eligible for that merge.

At any point, rebuild the requirement-level status artifact with:

```bash
uv run python tools/paper_tables/audit_table_reproduction.py
```

This writes `runs/tables_4_7/reproduction_status.json`. It verifies the paper
source hashes, all 19 native-or-supplemental trajectory sources, all 14 active
five-generation groups, exact guarded upload-policy selectors, and result rows
against the current evaluator fingerprints. The reconstructed-protocol audit
is complete when all 14 runnable cells have five current rows; strict paper
parity is reported independently and remains blocked when unpublished or
internally conflicting paper inputs are required.

The earlier Garden 360-in checkpoint (`41.302 +/- 6.809%` Depth AbsRel,
`78.2 +/- 12.233%` Object Permanence) is retained only as superseded evidence:
its open-vocabulary prompt collapsed the paper's five named anchors to a
mostly single-tree track. Current preparation supplies the exact five-anchor
set, assigns generic tree/lamp/rock detections to canonical identities using
the aligned Unity projection, and preserves all five objects in the permanence
denominator. Evaluator fingerprints cover this configuration, and resume
automatically discards stale rows before recomputing the corrected table.

### Standardized Garden Table 7 evaluation

The evaluator bridge verifies that every generated run used the exact Unity
conditioning-frame hash, locks the five Garden anchors published in Appendix
B.1, and derives a uniform evaluation copy for each native export: 150 frames,
15 fps, 10 seconds, and 1280x720. Temporal normalization drops or duplicates
frames without optical-flow interpolation; the native downloads remain
unchanged and both hashes are recorded.

```bash
.venv/bin/python tools/paper_tables/prepare_project_genie_table7_eval.py \
  --validation \
    runs/project_genie/table7/WanderingGarden/360_in/five_run_validation.json \
  --unity-receipt runs/unity/garden_bridge_gt/capture_receipt.json \
  --trajectory-mode full_turn \
  --reference-start-frame 0 \
  --reference-target-yaw-deg=-360 \
  --output-dir \
    runs/project_genie/table7/WanderingGarden/360_in/evaluation_prepared_pose_corrected

PYTORCH_ENABLE_MPS_FALLBACK=1 \
.venv/bin/python tools/run_wmeval_lite.py run \
  --suite lite \
  --candidate-manifest \
    runs/project_genie/table7/WanderingGarden/360_in/evaluation_prepared_pose_corrected/candidate_manifest.jsonl \
  --dimensions environment_object_consistency \
  --device mps --resume \
  --output-dir \
    reports/table7/project_genie/WanderingGarden/360_in/five_run_reconstructed_pose_corrected
```

Convert the five evaluator rows to the paper comparison schema and render the
cross-comparison:

```bash
.venv/bin/python tools/paper_tables/build_table7_rerun_results.py \
  --results-jsonl \
    reports/table7/project_genie/WanderingGarden/360_in/five_run_reconstructed_pose_corrected/per_case_results.jsonl \
  --evaluation-inputs \
    runs/project_genie/table7/WanderingGarden/360_in/evaluation_prepared_pose_corrected/evaluation_inputs.json \
  --output \
    reports/table7/project_genie/WanderingGarden/360_in/rerun_results.json

.venv/bin/python tools/paper_tables/compare_table_results.py compare --allow-incomplete \
  --results \
    reports/table7/project_genie/WanderingGarden/360_in/rerun_results.json \
  --output \
    reports/table7/project_genie/WanderingGarden/360_in/comparison
```

`--allow-incomplete` is intentional until every Table 4/7 cell is generated.
It does not turn missing or non-numeric metrics into successful results.

`probe` and `calibrate-yaw` exercise the scene's live four-channel ML-Agents
controller. They require a player rebuilt after the capture bootstrap's
`-wmeval-mlagents` opt-out was added. The current editor installation cannot
perform that rebuild until its Unity license is activated; the existing
standalone ground-truth capture remains runnable and independently validated.
