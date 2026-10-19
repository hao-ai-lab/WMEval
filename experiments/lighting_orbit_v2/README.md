# Complete lighting orbit protocol

The canonical action is a 10-second counter-clockwise circle of radius 2 m,
with eye height 1.6 m and a continuous look-at at the measured tabletop centre
(1.2735698223114014 m). The fixed sun and static props come from the original
SunRisesAndSets scene at revision 84a06a422ae0a0d04153596e881c85be8d4ff3f1.
The portable candidate bundle is `SunRisesAndSets/lighting_orbit_v2`.
The historical `SunRisesAndSets/lighting` case remains the paper rescoring path via `--suite eval_sets/wmeval_lite_v1/lighting_paper_native`
until the new generation and scoring coverage is complete.

Unity capture supplies 121 camera samples at 12 Hz, including the closure at
10 seconds. Its scoring video contains the first 120 frames, [0,10).
The case includes the analytic 601-sample 60-Hz trajectory, actions.npz,
camera-pose JSONL, intrinsics, calibration, fixed-state hashes and capture QA.
Camera poses are right-handed +Y-up OpenCV camera-to-world matrices. These
JSONL records are geometry, not browser key actions.

## Native interfaces

`tools/prepare_lighting_backend_inputs.py` derives inputs from the case's
trajectory.yaml, without substituting a yaw-only turn or stretching time:

- LingBot Base Cam: 161 analytic poses at 16 fps, t=0 through 10. Its nominal
  832x480 intrinsic input is compensated for the square seed before upstream
  resizes it. Both output focal lengths therefore encode the same 60° FOV.
  The fixed source is a43bec7f8091c83e9b30b16b912f6fc906236fa6; the fixed
  model revision is 6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd. Run 20 steps,
  full BF16, two GPUs with DiT/T5 FSDP and Ulysses size 2. The no-UI wrapper
  only disables the keyboard/camera diagnostic overlay.
  Upstream converts the sampled poses to frame-relative transforms and
  normalizes translation by the maximum step norm. The bundle records the
  intended metre-scale geometry; model conditioning uses those native
  normalized units. This native processing is retained and recorded, rather
  than claiming that the model enforces an absolute two-metre radius.
- HY WorldPlay: 64 latent poses spaced at 4/24 seconds for 253 native frames
  at 24 fps. Inputs are OpenCV C2W and seed intrinsics, consumed by native
  pose_to_input. Pose samples after t=10 hold the endpoint and are excluded
  from scoring. The pinned source is 1588e1336e842b03b0a7860c654ebd7c46bb065e
  and action model is f4c29235647707b571479a69b569e4166f9f5bf8. All five
  formal seeds completed native four-step AR generation and Lite scoring.
  Strict loading verified all 1,905 tensors and 8,557,933,856 parameters;
  use native BF16, two GPUs with sequence parallelism, torch_causal attention,
  no quantization, super-resolution, prompt rewriting or compilation.
  `tools/prepare_hy_runtime.py` records the pinned text/vision/Glyph assets;
  the native vision path must include `vision_encoder/siglip/`.
  `tools/run_lighting_hy_runpod.py` enforces the worker deadline and caches
  the native pipeline across the five seeds, resetting inference state each time.
- MG3: 177 native frames at 17 fps, four clips. A fixed reflected world basis
  maps canonical +Y-up into native XY-floor/+Z-up C2W, preserving every
  camera-relative transform. The pose pipeline receives the exact orbit;
  its auxiliary D keys use the native 0.1235 m/frame duty cycle and mouse
  yaw uses 15 degrees/unit. Native normalization retains the nominal geometry
  without claiming literal metre-scale output. The square seed stays square
  (928x928 for the 704x1280 area limit), with K transformed to preserve 60°.
  Use the pinned Matrix source 756776d516233027631008761d4d40ddaea3e23a and
  checkpoint 382aa382a03cec057360755aa72c7735025770bb. All five fixed seeds
  completed native generation and Lite scoring. Verified video hashes and
  per-seed metrics are in `results/run_summary.json`.
- MG2 and Waypoint are N/A for this release's metric look-at trajectory.
  Their pinned native interfaces expose keyboard/mouse tokens, not a camera
  pose/target or a documented key-to-metre conversion. The independent probes
  remain diagnostic evidence and do not establish a verified physical mapping.
  This classification concerns the audited input interface, not scene quality
  or the models' ability to navigate freely. Source hashes, symbols and the
  exact scope are in results/capability_audit.json.

Hosted Genie has five completed independent diagnostic repeats. The frozen
program uses forty 323-ms wall-clock cycles: simultaneous D+Left for 114 ms,
D for 112 ms, then release until the absolute cycle deadline. It targets
0.25 canonical seconds and 9 degrees per cycle, with a fixed 2.25-second
released tail outside the action. Compile it with
`tools/prepare_lighting_genie_controls.py --output <directory>`.
All runs use the same prompt, first-person mode and 1280x704 letterboxed seed;
numeric generation seeds are unavailable, so actual hosted export IDs are recorded.
Five ordered screenshot checkpoints verify each video association. The same
input spans 8.9583-10.1667 native seconds across repeats: HTML media currentTime
and the remaining-time bar include waiting and cannot supply a native frame clock.
The generated camera also approaches the tabletop; a complete two-metre
look-at closure is not established. These are completed diagnostic runs, with
first-ten-native-second Lite scores in `results/genie_diagnostic_summary.json`,
excluded from formally qualified protocol coverage; the hosted UI is N/A
for the required native-frame-indexed metric input. The first frozen repeat is
shown in `results/genie_first_repeat_contact_sheet.jpg` without output selection.
No output retiming,
quality-based seed replacement or post-hoc trajectory correction was applied.

Independent key/mouse calibration uses `tools/prepare_lighting_key_controls.py`
with fixed calibration seeds 1001 and 1002, before formal seeds are generated.
MG2 preserves the complete square seed as a 352x352 image centred in its native
640x352 canvas; K is transformed by the same scale and 144-pixel horizontal
offset. The probes apply D or mouse-yaw separately for two seconds. The native
model exports at 12 fps, despite a different framerate advertised elsewhere.
`tools/analyze_lighting_control_probe.py` measures metric camera response using
seed mesh points, forward/backward optical flow and robust PnP. It rejects
insufficient correspondence fits instead of assigning unsupported status.
The portable mesh points in `calibration_assets/` were obtained by
`tools/unity/LightingSeedGeometry.cs` from the original static scene's exact
MeshFilter triangles. Their seed rerender has MAE=0, and maximum reprojection
error is 0.00252 pixels. Alpha texture/skinned-mesh mismatches must be rejected.
These assets are independent calibration evidence, not formal output correction.
Keep unresolved response measurements separate from input-interface support; only accepted independent gains
can compile a full D-plus-yaw schedule for formal generation.

`experiments/tables_4_7/action_backends.yaml` tracks capability for this exact
trajectory. Pending adapter/calibration/access is incomplete work, not N/A.
Only a demonstrated interface inability qualifies as unsupported.

## Runpod preparation and generation

Use the official `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` image and a
persistent network volume mounted at /workspace. Connect with the console's
**SSH over exposed TCP** command. Every new file, virtual environment, pip,
Hugging Face, Torch, CUDA, Triton and temporary cache stays below
`/workspace/WMEval-lighting-orbit`. Use a cheap preparation GPU first.

Run `tools/runpod_prepare_lighting_env.sh`. Install the official FlashAttention
2.8.3 wheel for CUDA 12 / Torch 2.8 / CXX11 ABI TRUE / CPython 3.12; its SHA-256
is f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb.
Verify both Torch and FlashAttention execute on the selected GPU before
switching to the two-GPU generation machine.

Materialize the fixed LingBot source under `model_sources/lingbot-world` and
write source_provenance.json with repository, full revision and a `files`
mapping from relative filenames to SHA-256. Include generate.py and wan/.
The runner verifies every listed source file before loading the model.

`tools/download_lighting_checkpoint.py` accepts an HF inventory of model
revision, filenames, byte counts and LFS SHA-256 values. Download one shard
at a time; verify the published original hash. `--native-bf16-dit` stores
FP32 DiT tensors as the exact BF16 values upstream loads, records both
original and derived hashes, and fixes shard-index total_size. This is not
an additional quantization setting. Retain checkpoint_provenance.json.

Prepare the case with `tools/build_lite_case.py --relock` after committing
its evaluator implementation. Do not relock historical cases. Use a fresh
output directory; preserve failed run logs and receipts instead of silently
replacing a run. The formal seeds are 42, 1, 2, 3 and 4.

```sh
source tools/runpod_wmeval_preflight.sh --env-only
source "$WMEVAL_RUNPOD_ROOT/envs/lighting-py312-torch280/bin/activate"
python tools/run_lighting_lingbot_runpod.py \
  --case eval_sets/wmeval_lite_v1/cases/SunRisesAndSets/lighting_orbit_v2 \
  --deadline-utc "$WMEVAL_GENERATION_DEADLINE_UTC" --seeds 42 1 2 3 4
```

The runner records fixed source/model revisions, conditioning and input
hashes, dependency versions, hardware, command, elapsed time and video hash.
Multi-seed runs keep one immutable model allocation and distributed group.
Each native generate call recreates the seeded noise generator and scheduler;
completed receipts remain intact if a later seed fails. Shared generation
logs and the batch specification are linked from every receipt. On network
storage, the batch wrapper uses the official Diffusers disable_mmap load path.
Diffusers 0.33.1 omits this flag in its sharded dispatch, so a scoped wrapper
for this checkpoint forwards it. Stored tensors, model settings and sampling
are unchanged; record the full shard hash and tensor-equivalence check.
It kills the complete torchrun process group when the supplied deadline
expires. Pod lifecycle is managed separately: preserve /workspace and
terminate the generation Pod immediately on completion, failure or budget
expiry. Do not leave an idle GPU attached while doing local work.

## Scoring and release checks

LingBot, MG3 and HY have complete five-seed results in
results/run_summary.json. The other three pinned interfaces are explicitly
N/A for the strict trajectory, with source-level evidence; no zero scores or
missing-seed aggregates are substituted. The report records
the first valid native video and its normalized hash for each seed, including
poor model outcomes. Calibration probes are excluded from formal aggregation.

Install SAM2 source at 2b90b9f5ceec907a1c18123530e92e794ad901a4 in the
workspace environment. `tools/prepare_lighting_evaluator.py` downloads the
case's locked GroundingDINO and SAM2 large weights, verifies SHA-256 and binds
this task's dedicated HF cache to those revisions. Set HF_HUB_OFFLINE=1 and
TRANSFORMERS_OFFLINE=1 for all scoring calls. Preserve evaluator provenance.
When making a small source archive, include sam2/, setup.py, pyproject.toml,
README.md, MANIFEST.in and LICENSE. MANIFEST.in includes the YAML configs in
the installed wheel. Check the large-model Hydra config and execute the CUDA
connected-components kernel before accepting the installation.

Each candidate's receipt supplies `lighting_time_mapping`: schema version 1,
native fps, canonical_zero_frame and video_sha256. A candidate without one
(e.g. a model-root directory) gets a mapping derived from the video: its
decoded fps, canonical_zero_frame 0 and its own sha256; rows record
`input_contract.lighting_time_mapping` as `receipt` or `derived`. A model whose
orbit does not start at frame 0 passes the mapping in the candidate manifest.
Lite verifies CFR timestamps and at least 10 real seconds after that offset,
then samples exactly 120 frames at 12 fps before tracking. It keeps image
geometry and playback speed.
Short, VFR and mismatched-hash candidates fail instead of being padded.

Generation coverage is not complete until all required backends have five
valid runs (Genie has five independent hosted run IDs rather than invented
numeric seeds), or a demonstrated unsupported interface is explicitly N/A.
Run the same bundles with `tools/run_wmeval_lite.py run --dimensions lighting_shading`.
Compare a fresh result and a resume; invalidate stale candidate/evaluator hashes.
Do not calculate a complete-model aggregate from fewer than five successful
seeds, or represent missing results as zero. The active lighting case is now lighting_orbit_v2 after pinned-hardware fresh-score
and resume acceptance. The fifteen formal native videos are checked into the
repository. Merging and HF publication are separate actions.

## Standalone release package

tools/build_lighting_release_package.py builds a self-contained scorer and
dataset from the fifteen verified generation receipts. All candidate paths
are relative; the package contains the full Unity/action case, table_cell.json,
the public evaluation sources, asset/version locks and the six support decisions.
Genie diagnostic media are kept outside the formal candidate manifest.
The fifteen expected jobs and three N/A entries are explicit.

The archive is byte-reproducible. Verification rejects changed or missing
data/code, but ignores Python's runtime bytecode and the documented results/cache
directories. Runtime files are never silently incorporated into the archive.
A source generation_case.yaml retains the original authoring identity; adding
transport metadata does not alter any frozen pose, action or video bytes.

Fresh scoring uses the same tools/run_wmeval_lite.py path on an extracted clean
directory. tools/verify_lighting_release_run.py splits the fifteen jobs between
two GPUs, compares every one of the ninety metric values with tolerance 1e-6,
runs the real Lite resume command, and requires unchanged result-row hashes.
The initial bytecode-inventory bug was fixed and regression tested. A PRO 4500
run completed 15/15 with identical normalized inputs but 21/90 numerical differences;
that complete audit remains in results/release_verification/.

The subsequent original-model RTX PRO 6000 run completed 15/15 with all 90 metrics
exactly equal at 1e-6, maximum absolute error 0, and actual resume preserved the row
hash. Two independent LingBot/42 scores also matched the original. Its GPU/driver,
environment, raw scores, commands and reports are in results/same_gpu_verification/.
Strict verification requires the case-locked PRO 6000 model and driver; the checker
rejects a different GPU before inference. Other hardware can use ordinary Lite
scoring as a separately reported numerical-variation run.

Adding transport metadata had left a stale derived case stamp. The stamp was
corrected from the actual scored case and evaluator sources, with byte-identical
reference assets, case parameters and metrics. Raw original rows remain unchanged;
cached rows retain source_evaluator_lock and an explicit correction audit. Actual
resume on the corrected identity also passed. Future metadata authoring refreshes
this stamp automatically.

Build from eval_sets/wmeval_lite_v1/baselines/lighting_orbit_v2/release_sources.json.
The published baseline README provides full build, fresh-scoring and cached-resume
commands. A copied cache replays all 15 rows on CPU without GPU or model downloads;
changing candidate/evaluator fingerprints invalidates stale rows. Exact cross-GPU
fresh-score equality is not claimed. No GPT API key is used.

Model generation sources and checkpoint inventories are in source_provenance/,
generation_model_inventory.json and the two HY inventories. Clone the listed
official source revisions, verify the listed source hashes, and install their
manifest at the native wrapper's expected source_provenance.json (LingBot/Matrix)
or source_manifest.json (HY). Checkpoints are fetched at immutable revisions
with tools/download_lighting_checkpoint.py / tools/prepare_hy_runtime.py.
No private scratch directory, old Pod or model weights inside this package are
needed for evaluating its bundled videos. HF publication is a separate action.
