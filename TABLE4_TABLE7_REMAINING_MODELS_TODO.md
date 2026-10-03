# Table 4/7 remaining-model launch checklist

Status date: 2026-10-03. This is the operational checklist for adding
LingBot-World Base Cam, Matrix-Game 2.0, and Matrix-Game 3.0 to the reconstructed
Table 4/7 protocol. It does not change the paper-parity caveats in
`experiments/tables_4_7/README.md`.

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

## Work completed in this preparation pass

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

## Blockers before any GPU launch

- [ ] Acquire each official source tree and checkpoint under an immutable
  revision; record commit, checkpoint revision, and file hashes in
  `experiments/tables_4_7/remaining_models.yaml`.
- [ ] Build isolated, locked environments. Do not install into or alter the
  live vLLM environment.
- [ ] Obtain an operator-approved GPU window. Matrix-Game 2.0 documents at
  least 24 GiB GPU memory, which exceeds the currently free memory on every
  GPU. The official LingBot 14B example uses all eight GPUs. Recheck with
  `nvidia-smi` immediately before launch; never terminate the vLLM workers.
- [ ] Run one minimal smoke generation per backend, retain command/stdout/stderr,
  record peak memory, and abort on OOM rather than reclaiming another service's
  memory.
- [ ] Decide whether the original, now-unmaintained LingBot Base Cam is the
  required artifact (this checklist assumes yes) and not a silent substitution
  with LingBot-World v2.

## Required deterministic adapters

- [ ] **LingBot Base Cam:** consume the prepared `[161,4,4]` OpenCV-style
  camera-to-world poses and `[161,4]` `(fx, fy, cx, cy)` intrinsics. Confirm the
  first generated view against the conditioning image and confirm yaw sign on
  one inward and one outward probe before queueing all cells.
- [ ] **Matrix-Game 2.0:** replace randomized/hard-coded benchmark scheduling
  with a deterministic schedule input. Log every accepted keyboard/mouse
  command with model-frame index and time. The upstream mouse amplitude `0.1`
  is only a probe value, not a calibrated degree mapping.
- [ ] **Matrix-Game 3.0:** bypass interactive stdin with a deterministic
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

- [ ] LingBot Cam — WanderingGarden: `perturb_in`, `perturb_out`, `360_in`,
  `360_out` (20 generations).
- [ ] LingBot Cam — BilliardGame: `perturb_in`, `perturb_out`, `360_in`,
  `360_out` (20 generations).
- [ ] LingBot Cam — MirrorRollingBall: `perturb_in`, `perturb_out`, `360_in`,
  `360_out` (20 generations).
- [ ] Matrix-Game 2.0 — Table 4: RacingCar `third`, BilliardGame `first`
  (10 generations).
- [ ] Matrix-Game 2.0 — each of the twelve Table 7 cells (60 generations).
- [ ] Matrix-Game 3.0 — each of the twelve Table 7 cells (60 generations).

For every completed group:

- [ ] Validate five distinct, hash-addressed outputs and action receipts.
- [ ] Run target-action success and camera-path QA before the expensive metrics.
- [ ] Run only applicable metrics; preserve paper `N/A` cells as null, not zero.
- [ ] Emit per-run values plus arithmetic mean and sample standard deviation
  (`ddof=1`). A comparison is incomplete until all five numeric values exist.
- [ ] Retain failure statuses rather than replacing rejected generations.

## Metric and release gates inherited from `PENDING.md`

- [ ] Resolve the Table 4 schema mismatch before final export:
  `paper_baseline.yaml` and the result template require Exact Match, while
  `evaluator_protocol.yaml` intentionally reports Error Rate only. Either
  implement the legacy diagnostic or version the comparison schema so it is
  explicitly N/A; do not leave a supposedly complete table with silent nulls.
- [ ] Add target entity, coordinate frame, and unsupported-channel semantics to
  the versioned action contract.
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

Prepare or revalidate the LingBot fixed-side inputs:

```bash
python tools/prepare_lingbot_table7_inputs.py
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
