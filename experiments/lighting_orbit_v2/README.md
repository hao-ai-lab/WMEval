# Complete lighting orbit protocol

The canonical action is a 10-second counter-clockwise circle of radius 2 m,
with eye height 1.6 m and a continuous look-at at the measured tabletop centre
(1.2735698223114014 m). The fixed sun and static props come from the original
SunRisesAndSets scene at revision 84a06a422ae0a0d04153596e881c85be8d4ff3f1.
The portable candidate bundle is `SunRisesAndSets/lighting_orbit_v2`.
The historical `SunRisesAndSets/lighting` case remains the paper rescoring path
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
  from scoring. Runtime/checkpoint verification is still pending.
- MG3: 177 native frames at 17 fps, four clips. A fixed reflected world basis
  maps canonical +Y-up into native XY-floor/+Z-up C2W, preserving every
  camera-relative transform. The pose pipeline receives the exact orbit;
  its auxiliary D keys use the native 0.1235 m/frame duty cycle and mouse
  yaw uses 15 degrees/unit. Native normalization retains the nominal geometry
  without claiming literal metre-scale output. The square seed stays square
  (928x928 for the 704x1280 area limit), with K transformed to preserve 60°.
  Use the pinned Matrix source 756776d516233027631008761d4d40ddaea3e23a and
  checkpoint 382aa382a03cec057360755aa72c7735025770bb. Runtime verification
  is pending; the numeric adapter's geometry and timing tests pass.
- MG2, Waypoint and hosted Genie: full translation plus look-at adapters
  are still pending. Existing Table 7 yaw calibrations do not establish a
  metric-space orbit. An input interface's poor generated video remains a
  measured outcome; it does not make that backend unsupported.

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
native fps, canonical_zero_frame and video_sha256. Lite verifies CFR timestamps
and at least 10 real seconds after that offset, then samples exactly 120 frames
at 12 fps before tracking. It keeps image geometry and playback speed.
Short, VFR and mismatched-hash candidates fail instead of being padded.

Generation coverage is not complete until all required backends have five
valid runs (Genie has five independent hosted run IDs rather than invented
numeric seeds), or a demonstrated unsupported interface is explicitly N/A.
Run the same bundles with `tools/run_wmeval_lite.py --dimensions lighting_shading`.
Compare a fresh result and a resume; invalidate stale candidate/evaluator hashes.
Do not calculate a complete-model aggregate from fewer than five successful
seeds, or represent missing results as zero. Switch the active lighting case
only after generation, normalization, six metrics and portable bundle checks
pass. Update PENDING.md and open a PR against lanxiang/release-cleanup; publishing
baseline media and merging are separate actions.
