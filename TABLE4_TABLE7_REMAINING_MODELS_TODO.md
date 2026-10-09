# Table 4/7 open-model checklist

Status date: 2026-10-08. This is the checklist for adding LingBot-World Base
Cam, Matrix-Game 2.0, and Matrix-Game 3.0 to the reconstructed Table 4/7
protocol. It does not change the paper-parity caveats in
`experiments/tables_4_7/README.md`. The minute-by-minute GPU production log
that accompanied the original runner commit (`d913fb9`, branch
`lanxiang/release-cleanup-oss-models`) is preserved in git history and is
summarized below rather than repeated.

## Paper reporting decision (2026-10-05)

- Overleaf commit `9eaa0ed` is the Table 4 baseline;
  `table/main-physical_laws.tex` stays byte-for-byte unchanged. The preserved
  Table 4 RacingCar Matrix-Game 2.0 values originate from 30 runs even though
  the caption says five; keep that provenance and do not alter the frozen
  table silently.
- Commit `6a87b3a` installed runs 0--4 for all three open-source Table 7
  models; follow-up `6b29a84` corrected the Mirror wording, the nonzero Garden
  ranges, and the geometry-versus-permanence scoring note.
- Geometry cells need five numeric anchor-supported measurements; missing
  cells are N/A, never replaced by extra seeds or partial-cohort averages.
  Permanence keeps missing anchors as failures; paired character scores average
  In/Out per repeat before computing sample statistics.
- The separately authorized 30-run robustness work writes to
  `reports/tables_4_7/final/` and must not overwrite the five-run paper tables.

## Production status

- Base scope complete and receipt-verified: Matrix-Game 2.0 70/70,
  Matrix-Game 3.0 60/60, LingBot Base Cam 60/60 (fail-closed verifier, zero
  hash/identity/repeat errors).
- Each Table 7 model has 12/12 five-run variance-gate artifacts backed by
  exactly five per-run result rows; the two Matrix-Game 2.0 Table 4 groups are
  summarized separately.
- Adaptive LingBot expansion: frozen queue of 360 case/repeat rows; 100/360
  receipts verified at the last production checkpoint (2026-10-05).

## Action standardization

All open-model inputs are compiled from the hash-locked semantic trajectories
in `experiments/tables_4_7/trajectories/`. Each backend's native timeline,
supported action targets (camera / ego_vehicle / cue_ball / table_objects),
unsupported reasons and frozen camera gains live in
`experiments/tables_4_7/action_backends.yaml` and are read through
`evaluation/action_backends.py`; rate integration and zero-order-hold
resampling are shared helpers in `evaluation/trajectories.py`. A backend that
cannot express a trajectory's target is skipped and reported as unsupported,
never fed a substitute action.

| Backend | Native timeline | Camera (Table 7) | Car / ball (Table 4) |
| --- | --- | --- | --- |
| Project Genie | browser keys | nominal; left turn 51.47 deg/s, right turn not frozen | nominal key rules |
| Matrix-Game 2.0 | 129 frames @ 12 fps, 4 keys + mouse | pooled 10.1 deg/mouse unit/frame | thresholded WASD, target validation required |
| Matrix-Game 3.0 | 177 frames @ 17 fps, 6 keys + mouse | regime/direction gains 10.511--11.434 | out of scope |
| LingBot Base Cam | 161 poses @ 16 fps | nominal poses, no response correction | unsupported (camera input only) |

## Scope and definition of done

The repository contains 16 paper-shaped cells, but the active execution scope
is 14 cells: two Table 4 cells and all twelve Table 7 cells. The supplied
reference package covers those 14. `table4/RacingCar/first` and
`table4/BilliardGame/third` remain excluded and their packaged reference frames
are absent. Completing the *active* scope means:

- Matrix-Game 2.0: 2 Table 4 + 12 Table 7 groups, five generations each (70).
- LingBot-World Base Cam: 12 Table 7 groups, five generations each (60).
- Matrix-Game 3.0: 12 Table 7 groups, five generations each (60).
- 190 claimable generations in 38 model/cell groups.
- Six primary Table 4 metric entries per Table 4 model (2 cells x DTW, FDE,
  Error Rate), two legacy Exact Match diagnostics, and 32 applicable Table 7
  metric entries per Table 7 model. A literal paper-shaped Table 4 therefore
  has eight numeric entries per model.

LingBot Base Cam is not a valid Table 4 backend. Its input is a camera path,
not the target-entity controls required by RacingCar or BilliardGame. Do not
move the camera and report that as car/ball controllability. To add LingBot to
Table 4, explicitly choose a target-aware action variant, calibrate it, and add
10 generations; otherwise publish those two cells as unsupported.

Reopening all 16 cells is a separate scope change. It requires reference assets
for both excluded cells plus a decision on unavailable RacingCar ego motion and
the deliberately excluded Billiard third-person protocol.

## Completed work

- [x] Verified the ZIP container and all 242 manifest-listed files by SHA-256.
- [x] Installed its `runs/` and `calibration_samples/` assets without replacing
  repository configuration files.
- [x] Audited all semantic trajectories: 19 cases have numeric sources; 15 are
  native and four are supplemental reconstructions.
- [x] Audited active Table 4 action coverage. RacingCar third-person is
  metric-ready, with reverse throttle and the exact paper timing still absent.
  Billiard first-person remains diagnostic until target-action ambiguity and
  camera leakage are resolved.
- [x] Made the Project Genie turn analyzer consume the ZIP's portable JPEG
  samples when the original absolute PNG path is unavailable.
- [x] Recomputed the left-turn visual calibration from portable samples:
  6.99389 s/full turn, 51.4735 deg/s, 0.874236 s/45 degrees.
- [x] Exercised the right-turn samples fail-closed. They have no reliable
  full-yaw correlation peak, so no independent right-turn gain is frozen.
- [x] Prepared twelve 161-frame LingBot Base Cam Table 7 pose/intrinsics inputs
  and a 60-generation launch queue with hashes and source receipts. These are
  fixed-side inputs, not response-side model calibration.
- [x] Inspected GPU occupancy without changing it. Eight vLLM tensor-parallel
  workers span all eight A100s. The final snapshot showed 22,116 MiB free on
  GPUs 0-6 and 5,770 MiB on GPU 7; no process was killed, paused, or
  reconfigured.
- [x] Acquired the official LingBot and Matrix-Game source trees plus immutable
  checkpoint revisions in isolated environments; revisions are recorded in
  `remaining_models.yaml`.
- [x] Added deterministic, HUD-free adapters for LingBot Base Cam,
  Matrix-Game 2.0, and Matrix-Game 3.0, including eight-way Matrix sharding.
- [x] Response-calibrated Matrix-Game 2.0 to 10.1 visual degrees per mouse unit
  per frame and Matrix-Game 3.0 with direction/regime-specific gains; exact
  production-shaped perturbation checks landed within 0.9 degrees of target.
- [x] Completed LingBot's signed short-horizon sweep. It exposes nonlinear,
  horizon-sensitive response and autonomous hold drift, so nominal native pose
  paths are retained rather than applying an invalid scalar compensation.
- [x] Pinned and cached the Table 4/7 evaluator assets in
  `evaluator_assets.lock.json` (GroundingDINO, SAM2, Depth Anything, DINOv2).

## Blockers before any GPU launch

- [x] Acquire each official source tree and checkpoint under an immutable
  revision; record commit, checkpoint revision, and file hashes in
  `experiments/tables_4_7/remaining_models.yaml`.
- [x] Build isolated, locked environments. Do not install into or alter the
  live vLLM environment.
- [x] Obtain an operator-approved GPU window. Matrix-Game 2.0 documents at
  least 24 GiB GPU memory, which exceeds the currently free memory on every
  GPU. The official LingBot 14B example uses all eight GPUs. Recheck with
  `nvidia-smi` immediately before launch; never terminate the vLLM workers.
- [x] Run one minimal smoke generation per backend, retain command/stdout/stderr,
  record peak memory, and abort on OOM rather than reclaiming another service's
  memory.
- [x] Decide whether the original, now-unmaintained LingBot Base Cam is the
  required artifact (this checklist assumes yes) and not a silent substitution
  with LingBot-World v2.

## Required deterministic adapters

- [x] **LingBot Base Cam:** consume the prepared `[161,4,4]` OpenCV-style
  camera-to-world poses and `[161,4]` `(fx, fy, cx, cy)` intrinsics. Confirm the
  first generated view against the conditioning image and confirm yaw sign on
  one inward and one outward probe before queueing all cells.
- [x] **Matrix-Game 2.0:** replace randomized/hard-coded benchmark scheduling
  with a deterministic schedule input. Log every accepted keyboard/mouse
  command with model-frame index and time. The upstream mouse amplitude `0.1`
  is only a probe value, not a calibrated degree mapping.
- [x] **Matrix-Game 3.0:** bypass interactive stdin with a deterministic
  per-chunk schedule and log the six-component action vector accepted by each
  chunk. Its upstream camera amplitude `0.1` is likewise uncalibrated.
- [ ] Every adapter must record conditioning hash, semantic trajectory hash,
  backend source/checkpoint revisions, seed or generation-repeat ID, action
  acceptance trace, raw output hash, normalization receipt, and failures.
- [ ] Reject a Table 4 output when camera motion or generic avatar navigation
  occurred instead of motion of the named car/ball target.

## Calibration protocol (run before scored generations)

For each action-based backend, use a disposable calibration output directory;
five independent repeats per probe; no scored case as fitting data.

Current evidence does not satisfy that stronger repeat requirement, so the
boxes below intentionally remain open even though response gains were frozen
for the requested production run. Matrix-Game 2.0 has 11 disposable outputs
across nine settings: one run for no-action and each signed 0.05/0.1/0.2
impulse, plus two runs for each signed 0.3 impulse. LingBot has one output for
each of nine probes (no-action and signed 45/90/115/135-degree paths).
Matrix-Game 3.0's direct UI attempt is retained as rejected; its production
direction/regime gains instead come from the exact-shape response checks
described in the completed-work section. These artifacts support the scoped
launch but cannot establish five-repeat calibration variance or universal
multi-model acceptance bands.

- [ ] Reset/no-action probe: measure autonomous drift and verify reset identity.
- [ ] Signed camera impulses: left/right at at least three durations or chunks.
- [ ] Endpoint probes: -45, +45, -360, +360 degrees on a textured panorama.
- [ ] Fit accepted-action latency, signed yaw gain, endpoint error, hold drift,
  repeat variance, and dropped/rejected commands.
- [ ] Validate on a held-out mixed left/hold/right trajectory.
- [ ] Freeze gain, timing, command quantization, thresholds, source/checkpoint
  hashes, and calibration artifact hashes before scoring.
- [ ] For Matrix-Game 2.0 Table 4, separately probe Racing steering/throttle/
  brake and Billiard forward/lateral controls. Require target-entity success;
  do not infer it from key names.
- [ ] Freeze multi-model acceptance bands required by `PENDING.md`: yaw endpoint
  tolerance, latency, repeat variance, hold drift, and target-action success.

## Generation checklist

For each of the 38 claimable model/cell groups below, produce `run0` through
`run4`, then normalize to the locked reference fps, duration, resolution,
crop/pad policy, and timebase before evaluation.

- [x] LingBot Cam — WanderingGarden: `perturb_in`, `perturb_out`, `360_in`,
  `360_out` (20 generations).
- [x] LingBot Cam — BilliardGame: `perturb_in`, `perturb_out`, `360_in`,
  `360_out` (20 generations).
- [x] LingBot Cam — MirrorRollingBall: `perturb_in`, `perturb_out`, `360_in`,
  `360_out` (20 generations).
- [x] Matrix-Game 2.0 — Table 4: RacingCar `third`, BilliardGame `first`
  (10 generations).
- [x] Matrix-Game 2.0 — each of the twelve Table 7 cells (60 generations).
- [x] Matrix-Game 3.0 — each of the twelve Table 7 cells (60 generations).

For every completed group:

- [x] Validate five distinct, hash-addressed outputs and action receipts for
  every requested base group. The fail-closed verifier reports 70/70 MG2,
  60/60 MG3, and 60/60 LingBot generations with zero errors.
- [ ] Run target-action success and camera-path QA before the expensive metrics.
- [x] Run only applicable metrics; preserve paper `N/A` cells as null, not zero.
- [x] Emit per-run values plus arithmetic mean and sample standard deviation
  (`ddof=1`). A comparison is incomplete until all five numeric values exist.
  Each Table 7 model has 12/12 five-run variance-gate artifacts backed by
  exactly five per-run result rows; the two requested MG2 Table 4 groups are
  included in the separate Table 4 summaries.
- [ ] Retain failure statuses rather than replacing rejected generations.

### Adaptive 30-run variance gate

The five-run result is screened before looking at any additional seeds. Expand
an entire model/cell group from 5 to 30 independent generations when at least
one applicable metric crosses its pre-registered dispersion threshold:

- percentage metrics: sample standard deviation greater than 5 percentage
  points;
- positive unbounded metrics (DTW/FDE): coefficient of variation greater than
  0.20, with a near-zero safeguard that also requires standard deviation above
  0.05.

For every expanded group, keep runs 0--4 as the original five-run cohort and
runs 5--29 as an independent confirmation cohort. Report the 30-run mean,
sample standard deviation, and 95% confidence interval. Compare the original
5 against the independent 25 with a two-sided Welch t-test for location and a
median-centered Levene/Brown--Forsythe test for dispersion; apply Holm's
correction across the metrics in that model/cell group. Do not compare the
first five against the containing set of all thirty as if they were
independent.

## Metric and release gates inherited from `PENDING.md`

- [x] Resolve the Table 4 schema mismatch before final export. The evaluator
  already emitted the legacy `exact_match_pct`; the open-model summarizer and
  LaTeX exporter retain it explicitly while the locked protocol keeps Error
  Rate as the headline mechanics metric, so no paper-shaped cell is silently
  null.
- [ ] Add target entity, coordinate frame, and unsupported-channel semantics to
  a new version of the semantic trajectory contract, then rebuild and re-lock
  the reference package. Do not mutate the supplied v1 hash-locked assets in
  place; the broader Lite-suite migration in `PENDING.md` remains separate.
  Interim (done): target entity, frame and per-backend unsupported reasons are
  declared in `action_backends.yaml` alongside the untouched v1 trajectories
  and enforced at input-preparation time; folding them into a v2 trajectory
  schema and re-locking the package is still open.
- [ ] Lock candidate normalization and emit its provenance per result row.
- [ ] Add the uniform metric validity envelope: status, reason, source, units,
  coverage, normalization, and evaluator fingerprint.
- [ ] Freeze the EOC anchor visibility/assignment/coverage failure gate on
  Mirror, Billiard, and Garden candidates from more than one model.
- [ ] Promote verified Billiard/Garden supplemental controls into locked bundles
  while retaining reconstructed-vs-exact provenance.
- [ ] Pin evaluator dependencies and model assets; refresh evaluator locks and
  make stale locks fail CI.
- [ ] Keep the known caveats visible: unpublished exact action timelines,
  reconstructed In/Out anchors, Billiard paper/appendix view conflict, and
  Mirror photogrammetric depth workaround.

## Commands and artifacts

Prepare or revalidate the open-model fixed-side inputs (both read
`action_backends.yaml`):

```bash
python tools/prepare_lingbot_table7_inputs.py
python tools/prepare_matrix_game_inputs.py
```

Outputs are intentionally ignored run artifacts under
`runs/tables_4_7/model_inputs/lingbot_world_cam/`, including `summary.json` and
`generation_queue.jsonl`. The durable model/scope status is
`experiments/tables_4_7/remaining_models.yaml`.

Reference-side audits used in this pass:

```bash
python tools/analyze_project_genie_turn.py \
  calibration_samples/arrow_left/turn_samples.json \
  --output reports/tables_4_7/calibration/arrow_left.json
python tools/audit_trajectory_readiness.py \
  --output reports/tables_4_7/calibration/trajectory_readiness.json
python tools/audit_table4_action_coverage.py \
  --output reports/tables_4_7/calibration/table4_action_coverage
```

The right-turn analyzer is expected to fail until better samples are captured;
that failure is an evidence-preserving calibration result, not a reason to copy
the left gain without labeling a symmetry assumption.
