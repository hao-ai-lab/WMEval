# Table 4/7 remaining-model launch checklist

Status date: 2026-10-05. This is the operational checklist for adding
LingBot-World Base Cam, Matrix-Game 2.0, and Matrix-Game 3.0 to the reconstructed
Table 4/7 protocol. It does not change the paper-parity caveats in
`experiments/tables_4_7/README.md`.

## Current paper reporting decision (2026-10-05)

The latest user instruction supersedes the earlier plan to install adaptive
30-run means in the paper. Overleaf commit `9eaa0ed` is the Table 4 baseline;
keep `table/main-physical_laws.tex` byte-for-byte unchanged. Commit `6a87b3a`
installed the original runs 0--4 for all three open-source Table 7 models and
was pushed to GitHub. Geometry cells require five numeric anchor-supported
measurements; missing cells are unreported as N/A, with no replacement seeds
or partial-cohort averages. Permanence retains missing anchors as failures;
paired character scores average In/Out per repeat before computing sample
statistics. Keep visible paper edits orange via `lx_validation`.
Follow-up commit `6b29a84` was pushed and verified on `origin/master`; it
corrects the Mirror comparison to say "among open-source models", corrects
the nonzero Garden-view ranges, and clarifies geometry versus permanence
scoring. The Table 7 numerical body and the frozen Table 4 file are unchanged.

The already authorized 30-run robustness work remains in progress separately;
its completion does not authorize overwriting this five-run paper selection.
The current finalizer writes to `reports/tables_4_7/final/`, not the paper
checkout. Its adaptive exports must not be substituted for the latest paper
tables during handoff. The preserved Table 4 RacingCar Matrix-Game 2 values
originate from 30 runs even though the latest Overleaf caption says five;
retain that provenance here and do not alter the user-frozen table silently.

Latest generation checkpoint: 100/360 LingBot receipts, leaving 260 runs.
`BilliardGame/perturb_in` and `BilliardGame/perturb_out` runs 0--8,
and `BilliardGame/360_in` runs 0--10
pass the hash, model/case/repeat identity and distinct-output verifier.
The newest videos in all three groups decode as H.264, 832x464, 161 frames at
16 fps (10.0625 seconds). GPU 4 has handed off to its full adaptive shard;
its adapter and the GPU 1 adapter are live during CPU-side work, while the
other six GPUs were sampled at 100% utilization. The existing finalizer and
adaptive evaluation gate remain live; no jobs were restarted.

- [x] Replaced brittle paper-caption string substitution in the separate
  robustness exporter with cohort-aware captions and explicit reporting-scope
  metadata. A 30-run export cannot silently inherit the five-run paper caption.
  All 10 focused exporter/finalizer tests pass; caption replacement also passes
  against both current paper templates without changing their table bodies.

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

## Live production status (2026-10-05 00:06 UTC snapshot)

- [x] Re-ran the fail-closed 38-case completion audit during LingBot
  production. All 26 Matrix-Game cases are complete, and the Matrix-Game 2.0
  Table 4 plus both Matrix Table 7 LaTeX blocks validate. All 12 remaining
  incomplete cases and the sole unvalidated paper block are confined to the
  expected LingBot Table 7 work.
- [x] Completed and receipt-verified the full five-run Matrix-Game 2.0 base
  scope: 70/70 outputs (two Table 4 groups and twelve Table 7 groups).
- [x] Completed and receipt-verified the full five-run Matrix-Game 3.0 base
  scope: 60/60 Table 7 outputs.
- [x] Complete LingBot Base Cam Table 7 base scope: 60/60 outputs have complete
  receipts. The original GPU-7 shard completed all six assignments; its
  delayed receipts were recovered only after each video became stable and
  ffprobe-valid. Parent-aware handoff watchers started LingBot shards 0--4
  after their Matrix-Game 3 parents exited, and GPUs 5--6 handed over after
  completing Matrix-Game 2. All eight LingBot generation slots remain
  assigned, avoiding transient between-sample GPU gaps that do not have enough
  memory for LingBot.
  WanderingGarden `perturb_in` has reached 5/5 receipt-verified generations,
  and its first five-run evaluation completed alongside generation on GPU 3
  within the measured memory envelope. Its preregistered gate selected a
  30-run expansion: depth AbsRel had $6.17$ percentage-point sample SD and
  pairwise-distance MAE had $11.42$ percentage-point sample SD; object
  permanence alone stayed below the expansion threshold. WanderingGarden
  `perturb_out` has also completed its five-run gate and remains at five runs:
  object permanence was exactly zero in every repeat, while depth AbsRel and
  pairwise-distance MAE were explicitly undefined in all five rather than
  being fabricated; these cells export as `N/D`.
- [ ] Complete the known LingBot adaptive pairs. The final two Garden
  perturbation gates select both In and Out for 30 runs under the paired-cell
  policy. Garden `360_out` also triggered on depth AbsRel ($7.02$ points SD)
  and pairwise-distance MAE ($11.54$ points SD), selecting both 360-degree
  directions; `360_in` independently confirmed the same decision ($7.04$ and
  $16.29$ points SD, respectively). BilliardGame `perturb_in` is the fifth
  completed gate and selected expansion on all three reported metrics: depth
  AbsRel had $45.02$ percentage-point sample SD, pairwise-distance MAE had
  $37.97$ points SD, and character consistency had $42.19$ points SD.
  `perturb_out` is the sixth completed gate and independently confirms the
  expansion: depth AbsRel had $5.78$ percentage-point sample SD,
  pairwise-distance MAE had $42.88$ points SD, and character consistency had
  $45.39$ points SD. The paired-cell decision therefore extends both
  BilliardGame perturbation directions. BilliardGame `360_in` is the seventh
  completed gate and also selected expansion: pairwise-distance MAE had
  $19.83$ percentage-point sample SD and character consistency had $15.21$
  points SD, while depth AbsRel remained below threshold at $3.56$ points.
  Its paired-cell decision extends both BilliardGame 360-degree directions.
  `360_out` is now the eighth completed gate and independently confirms that
  decision: depth AbsRel, pairwise-distance MAE, and character consistency had
  sample SDs of $6.50$, $36.29$, and $16.15$ percentage points, respectively.
  Replaying the exact pair extension added zero rows and retained 260 total,
  proving the queue update remains idempotent. MirrorRollingBall `perturb_in`
  is the ninth completed gate and selected expansion on character consistency
  ($27.88$ percentage-point sample SD); pairwise-distance MAE stayed stable at
  $1.26$ points SD. The paired-cell policy immediately extended both Mirror
  perturbation directions by 50 rows.
  Ten selected cells have been idempotently extended by 250 rows (310 total),
  and GPU 7 is prelaunching exact confirmation runs 5--10 for
  `perturb_in`. Its later full-plan handoff is reversibly paused and is
  automatically resumed by the prelaunch wrapper on exit. The isolated batch
  label and queue idempotency are covered by focused checks (6 passed). The
  early runner completed model preparation and is sampling on GPU 7, so all
  eight GPUs are performing LingBot generation work. Its first four adaptive
  outputs, `WanderingGarden/perturb_in/run5` through `run8`, are
  receipt-complete; a nine-repeat verifier pass over runs 0--8 found zero
  hash, provenance, or repeat-ID errors. GPU 4 completed its six-row base
  shard with all finalized receipts hash-matching, then was immediately reused
  for exact BilliardGame `360_in` confirmation runs 5--10. Its isolated batch
  pauses only that GPU's future full-plan handoff and automatically resumes the
  handoff on exit, matching the guarded GPU-7 prelaunch pattern. Its first
  adaptive output, `BilliardGame/360_in/run5`, passed the contiguous six-repeat
  cohort verifier and media checks. GPU 2 then completed its seven-row base
  shard with every receipt hash-matching. Its idle slot was safely reassigned
  to exact `MirrorRollingBall/perturb_in` confirmation runs 5--10 after a dry
  run proved the six outputs were non-overlapping. The guarded wrapper pauses
  only GPU 2's future full-plan handoff and resumes it automatically on exit.
  The newest
  base output, `MirrorRollingBall/360_in/run1`, passed receipt hash/byte checks,
  the three-repeat fail-closed cohort verifier, and ffprobe validation (H.264,
  832x464, 161 frames at 16 fps). Two more base outputs then completed:
  `BilliardGame/360_out/run3` and `MirrorRollingBall/perturb_in/run1`. Each
  completed its five-repeat cohort and passed the full receipt/hash/ffprobe
  verifier and automatically started both pending five-run evaluators. The
  subsequent `MirrorRollingBall/perturb_out/run3` receipt also passed its
  provenance, SHA-256, byte-size, and ffprobe checks (H.264, 832x464, 161
  frames at 16 fps). `MirrorRollingBall/perturb_out/run0` then passed the same
  checks, bringing that base cohort to 3/5. The next adaptive output,
  `WanderingGarden/perturb_in/run8`, passed the contiguous nine-repeat cohort
  verifier and media checks. The subsequent base output,
  `MirrorRollingBall/360_out/run3`, also passed provenance, SHA-256, byte-size,
  and media checks. `MirrorRollingBall/perturb_out/run1` then passed the same
  checks. `MirrorRollingBall/perturb_out/run4` completed that five-repeat
  cohort; the full cohort verifier and media checks passed, its five-run
  evaluator started automatically on GPU 4, and all LingBot receipts reached
  60. The immediately following `MirrorRollingBall/360_out/run4` receipt also
  passed provenance, SHA-256, byte-size, and media checks, bringing that cohort
  to 3/5. `MirrorRollingBall/360_out/run1` then passed the same checks, bringing
  that cohort to 4/5 and all LingBot receipts to 62.
- [x] Completed the tenth LingBot five-run variance gate.
  `MirrorRollingBall/perturb_out` independently confirmed the paired
  perturbation expansion: character-consistency sample SD was $24.89$
  percentage points, while pairwise-distance MAE remained stable at $1.00$
  point. Replaying the exact pair extension added zero rows and retained the
  frozen 310-row queue. The next base receipt,
  `MirrorRollingBall/360_in/run3`, passed the four-repeat cohort verifier and
  media validation (H.264, 832x464, 161 frames at 16 fps), bringing the base
  scope to 58/60 and all LingBot receipts to 63. GPUs 0, 3, and 5 were then
  assigned six non-overlapping confirmation rows each from already selected
  Garden cells; their later full-plan handoffs are paused only for the duration
  of these isolated batches and resume automatically. Together with the five
  existing workers, this restores an eight-GPU generation allocation.
  `WanderingGarden/perturb_in/run9` subsequently completed, bringing that
  adaptive cohort to ten contiguous, receipt-verified repeats and total
  LingBot receipts to 64. The full cohort verifier reported zero errors, and
  the new video independently decoded as H.264, 832x464, 161 frames at 16 fps.
  GPU 3's final five-run evaluator watcher remains active alongside generation
  (within the previously measured memory envelope), while only its future
  generation handoff stays paused; the last Mirror 360-degree gate therefore
  will not wait for the six-row prelaunch batch to finish.
  The next completed output was the first Mirror confirmation row,
  `MirrorRollingBall/perturb_in/run5`. It passed the complete contiguous
  six-repeat cohort verifier and independent media validation (H.264,
  832x464, 161 frames at 16 fps), bringing total LingBot receipts to 65.
  `BilliardGame/360_in/run6` followed and passed the same fail-closed checks
  over its complete contiguous seven-repeat cohort, bringing total LingBot
  receipts to 66.
  The fifth `MirrorRollingBall/360_in` base output (`run4`) then completed.
  The full five-repeat cohort passed provenance/hash verification, and the new
  video independently decoded as H.264, 832x464, 161 frames at 16 fps. This
  brings the base scope to 59/60 and total LingBot receipts to 67. Its pre-armed
  GPU-4 evaluator started automatically and is producing the eleventh
  preregistered variance gate while generation continues.
  The final base output, `MirrorRollingBall/360_out/run2`, then completed and
  passed the full five-repeat cohort verifier plus independent media checks
  (H.264, 832x464, 161 frames at 16 fps). LingBot's requested five-run base
  scope is therefore complete at 60/60, with 68 total receipts including the
  eight confirmation rows produced so far. Both final Mirror 360-degree
  evaluators started automatically on GPUs 3 and 4; their gates will freeze
  the final paired expansion plan.
  The freed GPU 1 and GPU 6 slots were immediately reassigned to six
  non-overlapping confirmation rows each for the already-selected
  `BilliardGame/perturb_in` and `perturb_out` cells. Dry runs verified exact
  runs 5--10 before launch; only those GPUs' later full-plan handoffs are
  paused, with automatic resume on wrapper exit. Both workers completed model
  assembly and entered full GPU sampling, restoring simultaneous generation
  load on all eight GPUs while the last two five-run evaluators run safely
  alongside generation on GPUs 3 and 4.
  The eleventh gate, `MirrorRollingBall/360_in`, then selected expansion:
  character-consistency SD was $9.38$ percentage points and
  pairwise-distance-MAE SD was $12.63$ points. The preregistered paired-cell
  policy therefore extended both Mirror 360-degree directions by 50 rows.
  Replaying the same extension added zero rows, proving idempotency, and the
  complete LingBot queue is now frozen at 360 unique case/repeat rows pending
  the independent `360_out` confirmation gate.
  The twelfth and final gate, `MirrorRollingBall/360_out`, independently
  confirmed expansion: character-consistency SD was $20.70$ points and
  pairwise-distance-MAE SD was $83.02$ points. Replaying the paired extension
  again added zero rows. The original planner watcher then exposed a direct-
  script import-path defect after all gates landed; the planner was hardened
  with the repository-root bootstrap used by the other operational tools, its
  focused tests pass (2 passed), and both dry-run and production CLI paths now
  succeed. The frozen plan records all 12 gates, selects all 12 Table 7 cells,
  and preserves the 360-row duplicate-free queue with `new_rows: 0`.
  Its SHA-256 is
  `ed516f935449af845e48e33674c776e794a88469a9ec5a7861e0f9ebd37d66f7`;
  the corresponding queue SHA-256 is
  `20a20231567c100945989150b7ce7ff77dca79abd7ee8e1b9f1bc61fb31dd436`.
  All eight full-plan handoff processes remain reversibly stopped behind their
  active isolated batches and will resume automatically, preventing duplicate
  generation while preserving continuous eight-GPU work.
  The first post-plan receipts arrived together:
  `WanderingGarden/360_out/run5` and
  `WanderingGarden/perturb_out/run5`. Both complete six-repeat cohorts passed
  provenance/hash verification, and both new videos independently decoded as
  H.264, 832x464, 161 frames at 16 fps. Total LingBot receipts are now 70.
  `WanderingGarden/perturb_in/run10` then completed its isolated six-row
  batch. The full contiguous 11-repeat cohort and new media passed the same
  checks, bringing total receipts to 71. The batch exited cleanly with zero
  failures, its trap resumed the guarded GPU-7 handoff, and that watcher
  successfully exec'd the frozen full-plan shard runner.
  Auditing that first staggered handoff exposed a cross-shard reservation
  hazard: stable shard 7 included one row still owned by GPU 0's opportunistic
  batch. The runner was paused before producing any candidate, then hardened
  so unlabeled full-plan shards automatically exclude every row in all
  `early_adaptive_*` manifests; an explicit recovery override remains for a
  failed reservation. Focused runner/resume tests pass (7 passed). GPU 7's
  replacement dry run selected 33 jobs, excluded six stable-shard reservations,
  and had zero overlap with all 48 reserved keys. The unsafe process was
  terminated before output, and the audited 33-job replacement runner plus
  its waiting adaptive evaluator are now live. All later staggered handoffs
  inherit the same reservation guard.
  `WanderingGarden/360_in/run5` then completed and passed its full contiguous
  six-repeat cohort verifier plus media checks (H.264, 832x464, 161 frames at
  16 fps), bringing total LingBot receipts to 72.
  `MirrorRollingBall/perturb_in/run6` completed next. Its receipt SHA-256
  matches the candidate bytes, and the video independently probes as H.264,
  832x464, 161 frames at 16 fps. Total LingBot progress is therefore 73/360,
  with 287 generations outstanding; all eight host GPUs remain assigned and
  the transient zero-utilization interval on GPU 2 is the expected handoff
  between consecutive samples.
  A read-only audit of all eight stable full-plan shards finds 252 pending
  full-shard jobs with 252 unique keys, zero cross-shard duplicates, and zero
  overlap with the 48 early reservations. The remaining 36 incomplete reserved
  rows plus these 252 jobs accounted for all 288 outstanding generations at
  the preceding 72-receipt snapshot; the newly completed reserved row reduces
  the live outstanding total to 287 without changing either frozen manifest.
  Two Billiard outputs then completed nearly together:
  `perturb_in/run5` and `360_in/run7`. Both receipts match the candidate
  SHA-256 values, and both files independently probe as H.264, 832x464,
  161 frames at 16 fps. `BilliardGame/perturb_out/run5` followed immediately
  and passed the same independent hash and media checks. LingBot progress is
  now 76/360, leaving 284 frozen-queue generations outstanding. GPUs 1, 4,
  and 6 entered their expected low-memory decode/save interval together after
  finishing diffusion. GPU 2 completed its own CPU-heavy decode/save
  transition and resumed full sampling; the next host snapshot again showed
  all eight GPUs saturated during sampling, with no error signatures in any
  live LingBot batch log.
  A fresh fail-closed audit at this 76-receipt point reports 26/38 cases
  complete: all 14 Matrix-Game 2.0 and all 12 Matrix-Game 3.0 cases pass, and
  the only 12 incomplete cases are the expected LingBot adaptive cohorts. The
  MG2 Table 4, MG2 Table 7, and MG3 Table 7 paper blocks all reconstruct and
  validate, so no hidden Matrix or downstream schema blocker remains.
  A focused rerun of the statistical-summary, adaptive-plan/evaluator,
  completion-audit, finalizer, and LaTeX-export tests passes 14/14 with
  third-party pytest plugin autoload disabled.
  Host-capacity preflight during the live run found 4.9 TiB filesystem space
  free, 99% of inodes free, and 1.3 TiB RAM available; the current LingBot
  model-output tree uses 802 MiB. Disk, inode, and host-memory exhaustion are
  therefore not expected for the remaining frozen queue.
  `WanderingGarden/perturb_out/run6` completed next. Its receipt SHA-256
  matches the candidate bytes, and the video independently probes as H.264,
  832x464, 161 frames at 16 fps. LingBot progress is now 77/360, leaving 283
  frozen-queue generations outstanding; no duplicate or off-queue receipt key
  has been observed.
  `WanderingGarden/360_out/run6` followed and passed the same independent hash
  and media checks. LingBot progress is now 78/360, leaving 282 frozen-queue
  generations outstanding.
  `WanderingGarden/perturb_in/run15` is the next completed frozen-queue row.
  Its receipt SHA-256 matches the candidate bytes; independent decoding reports
  H.264 at 832x464, 161 frames, 16 fps, and 10.0625 seconds. LingBot progress
  is now 79/360, leaving 281 generations outstanding. The refreshed
  fail-closed audit remains at the expected 26/38 complete cases, with all 12
  incomplete cases confined to the in-flight LingBot 30-run cohorts.
  `WanderingGarden/360_in/run6` then completed with a matching receipt and
  independent H.264 decode at 832x464, 161 frames, 16 fps, and 10.0625
  seconds. LingBot progress is now 80/360, leaving 280 frozen-queue
  generations outstanding. A full incremental sweep of all 80 production
  receipts recomputed every candidate SHA-256 and probed codec, dimensions,
  frame rate, and frame count; all 80 passed with zero errors.
  `MirrorRollingBall/perturb_in/run7` completed next. Its receipt hash matches
  the candidate, and a full independent decode reports H.264 at 832x464, 161
  frames, 16 fps, and 10.0625 seconds. LingBot progress is now 81/360, leaving
  279 frozen-queue generations outstanding.
  Three BilliardGame rows then completed in a closely spaced burst:
  `perturb_in/run6`, `360_in/run8`, and `perturb_out/run6`. Every receipt hash
  matches its candidate and every full decode reports H.264 at 832x464, 161
  frames, 16 fps, and 10.0625 seconds. LingBot progress is now 84/360, leaving
  276 frozen-queue generations outstanding.
  `WanderingGarden/perturb_out/run7` and `360_out/run7` completed next. Both
  receipt hashes match their candidates, and both full decodes report H.264
  at 832x464, 161 frames, 16 fps, and 10.0625 seconds. LingBot progress is now
  86/360, leaving 274 frozen-queue generations outstanding.
  `WanderingGarden/perturb_in/run23` and `360_in/run7` followed. Both receipt
  hashes match their candidates, and both fully decode as H.264 at 832x464,
  161 frames, 16 fps, and 10.0625 seconds. LingBot progress is now 88/360,
  leaving 272 frozen-queue generations outstanding.
  A fresh fail-closed audit at this checkpoint still reports the expected
  26/38 complete cases (Matrix-Game 2: 14/14; Matrix-Game 3: 12/12), with all
  12 incomplete cases confined to the in-flight LingBot cohorts. No hash,
  identity, duplicate, unexpected-row, or schema error was found; the three
  completed Matrix paper blocks still reconstruct successfully.
  `MirrorRollingBall/perturb_in/run8` completed next. Its receipt SHA-256
  matches the candidate, and a full decode reports H.264 at 832x464, 161
  frames, 16 fps, and 10.0625 seconds. LingBot progress is now 89/360, leaving
  271 frozen-queue generations outstanding.
  `BilliardGame/perturb_in/run7` then completed. Its frozen identity and
  provenance passed the contiguous eight-repeat cohort verifier, its receipt
  SHA-256 matches the candidate bytes, and an independent media probe reports
  H.264 at 832x464, 161 frames, 16 fps, and 10.0625 seconds. LingBot progress
  is now 90/360, leaving 270 frozen-queue generations outstanding.
  `BilliardGame/360_in/run9` followed. Its contiguous ten-repeat cohort passes
  the frozen identity, provenance, and SHA-256 verifier, and its candidate
  independently probes as H.264 at 832x464, 161 frames, 16 fps, and 10.0625
  seconds. LingBot progress is now 91/360, leaving 269 frozen-queue
  generations outstanding.
  `BilliardGame/perturb_out/run7` completed next. The contiguous eight-repeat
  cohort passes the frozen identity, provenance, and SHA-256 verifier, and the
  new candidate independently probes as H.264 at 832x464, 161 frames, 16 fps,
  and 10.0625 seconds. LingBot progress is now 92/360, leaving 268 frozen-queue
  generations outstanding.
  `WanderingGarden/perturb_out/run8` followed. Its receipt SHA-256 matches the
  candidate, and an independent full decode reports H.264 at 832x464, 161
  frames, 16 fps, and 10.0625 seconds. A refreshed fail-closed completion audit
  still finds exactly the expected 26/38 complete cases, with every incomplete
  case confined to the 12 in-flight LingBot groups and all three completed
  Matrix paper blocks reconstructing successfully. LingBot progress is now
  93/360, leaving 267 frozen-queue generations outstanding.
  `WanderingGarden/360_out/run8` completed next. Its frozen trajectory
  identity and provenance are intact, its receipt SHA-256 matches the
  candidate, and an independent full decode reports H.264 at 832x464, 161
  frames, 16 fps, and 10.0625 seconds. LingBot progress is now 94/360,
  leaving 266 frozen-queue generations outstanding.
  GPU 7's full-plan shard then completed the non-contiguous but frozen
  `WanderingGarden/perturb_out/run17` assignment. Its queue identity, seed,
  model/source/checkpoint revisions, conditioning/pose/intrinsics hashes,
  candidate SHA-256, and byte size all match the receipt; an independent full
  decode reports H.264 at 832x464, 161 frames, 16 fps, and 10.0625 seconds.
  LingBot progress is now 95/360, leaving 265 frozen-queue generations
  outstanding.
  `WanderingGarden/360_in/run8` followed from GPU 3's isolated confirmation
  batch. Its frozen queue and batch identities match exactly, the receipt
  SHA-256 matches the candidate, and an independent full decode reports H.264
  at 832x464, 161 frames, 16 fps, and 10.0625 seconds. LingBot progress is
  now 96/360, leaving 264 frozen-queue generations outstanding.
- [x] Applied the preregistered five-run variance screen to every completed
  Matrix group. Matrix-Game 2.0 expanded 10 Table 7 cells plus RacingCar;
  Matrix-Game 3.0 expanded all 12 Table 7 cells.
- [x] Completed Matrix-Game 2.0 RacingCar's 30-run generation, evaluation,
  confidence intervals, Welch tests, Brown--Forsythe tests, and Holm
  correction. BilliardGame Table 4 remained at five runs because its
  dispersion stayed below the gate.
- [x] Completed all four Matrix-Game 3.0 WanderingGarden 30-run generations,
  evaluations, confidence intervals, and preregistered statistical tests.
- [x] Finished all adaptive Matrix cohorts. Generation and evaluation
  are running concurrently with LingBot; completed cohorts are evaluated as
  soon as their 30th receipt is verified. Matrix-Game 2 generation is now
  complete and passed a full mixed-cohort audit: 345/345 distinct,
  hash-matching outputs (15 across three five-run groups and 330 across eleven
  30-run groups). Matrix-Game 3 has completed and
  passed a full fail-closed audit of all 360 adaptive generations. Its four
  BilliardGame 30-run confidence intervals and corrected tests are complete;
  none of their first-five versus confirmation-25 comparisons is
  Holm-significant. Matrix-Game 3 is now complete at 12/12 directional 30-run
  comparison artifacts plus 4/4 paired-character comparison artifacts. Its
  final MirrorRollingBall 360-degree direction and pair had no corrected mean
  or variance shift. Matrix-Game 2 is complete at 11/11 directional comparison
  artifacts plus 4/4 paired-character comparison artifacts. Its BilliardGame
  360-degree pair is complete, with no Holm-significant first-five versus
  confirmation-25 mean or variance shift.
  MirrorRollingBall `perturb_in` is also complete: character consistency
  changed from $94.334$ for the first five to $72.280$ for the independent 25
  confirmations (Holm-adjusted Welch $p=0.0291$), while its Brown--Forsythe
  variance result and both pairwise-distance tests were not significant.
  MirrorRollingBall `360_in`, `360_out`, and their paired 360-degree character
  statistic are also complete without a significant corrected mean or
  variance shift. MirrorRollingBall `perturb_out` likewise had no significant
  directional test, but the paired perturbation character mean confirmed the
  first-five shift ($48.332$ versus $37.305$ for the independent 25;
  Holm-adjusted Welch $p=0.0146$). A fresh fail-closed readiness pass reports
  no remaining Matrix-Game 2 or Matrix-Game 3 deficiency.
- [x] Armed sequential, fail-closed LingBot five-run evaluation and variance-
  screen chains for all twelve Table 7 cells. Each chain waits for five
  receipt-verified generations and its assigned Matrix evaluator, maintains at
  most one evaluator beside LingBot on a GPU, and writes the preregistered
  metric-specific variance gate before advancing to its next cell.
- [x] Armed all eight LingBot adaptive-generation handoffs and added a sharded,
  fail-closed adaptive evaluation driver. It consumes the paired expansion
  plan, waits for 30 receipt-complete generations, hash-verifies every cohort,
  resumes the evaluator from the frozen original five rows, then writes 95%
  confidence intervals plus Welch, Brown--Forsythe, and Holm results.
- [x] Attached the eight adaptive evaluation/statistics workers behind their
  corresponding generation processes, plus a stable-video receipt monitor for
  the one legacy GPU-7 process. The full repository suite passes after the
  receipt-resume, exact-row tail-helper, and evaluation-concurrency hardening
  (664 passed, 1 skipped).
  A newly emitted native LingBot receipt was
  independently hash-matched and decoded as 161 frames at 832x464/16 fps with
  complete source, checkpoint, conditioning, pose, and intrinsics provenance.
  The formerly direct GPU-7 evaluation poller was replaced in place with an
  all-360-generation receipt gate followed by a 15-minute decode/offload
  quiescence window. It remains CPU-only while waiting, so it cannot contend
  with either the active generator or a later tail helper, including during
  cleanup after the final receipt; the other seven evaluators remain chained
  behind their generation processes.
  The final LaTeX exporter now fails closed
  until every variance-selected 30-run cohort and its corrected statistical
  comparison are complete; it cannot silently fall back to five-run values.
- [x] Rebased paper integration onto the clean Git-backed Overleaf checkout at
  `/workspace/-arxiv-WorldModel-Eval` rather than the historical extracted
  ZIP directory. Added the requested logical `lx_validation` macro with an
  orange `\lxvalidation{...}` alias, installed the final Matrix-Game 2.0 Table
  4 and Matrix-Game 2.0/3.0 Table 7 cells, and corrected the corresponding
  active experiment and implementation text. A canonical comparison confirms
  that all three installed model blocks exactly match the WMEval exporter after
  removing only the orange wrappers. This validated intermediate paper state
  was committed as `3ff9859` and pushed to GitHub `origin/master`; the final
  LingBot row and statistical discussion remain pending until that cohort is
  frozen.
- [x] Added a 38-case final completion auditor covering the exact requested
  model/table scope. It hash-verifies every required generation, requires the
  gate-selected evaluation cohort and statistics, validates paired character
  cells, and reconstructs all four paper-shaped model blocks before it can
  return `status: complete`.
- [x] Armed a fail-closed finalization worker. Once readiness is satisfied, it
  runs the 38-case hash audit exactly once, exports the complete paper-shaped
  Table 4 and Table 7 LaTeX files, and records hashes for the audit, tables,
  and export receipt under `reports/tables_4_7/final/`.
- [x] Added final JSON and CSV statistical summaries covering all directional
  groups and the paired character cells actually reported in Table 7. They
  preserve 95% intervals, original-five and confirmation-25 means, raw and
  Holm-adjusted Welch/Brown--Forsythe p-values, and corrected significance.
- [x] Added reported-cell statistical assessment for Table 7 character
  consistency: pair In/Out by repeat ID, average each pair, then apply the
  same five-run gate and 5-vs-25 Welch/Brown--Forsythe/Holm protocol. Persistent
  workers are armed for all three models; completed Matrix-Game 3.0 Billiard
  paired cells show no corrected mean or dispersion shift.
- [x] Hardened generation resume to require a complete, hash-matching receipt;
  an orphaned candidate can no longer be mistaken for a completed run.
- [x] Hardened the persistent LingBot adapter to recheck each job immediately
  before generation and skip it only when both the candidate hash and all
  frozen job identity/provenance fields match its complete receipt. This
  enables safe late-stage work redistribution from a finished GPU to a tail
  row of a heavier shard. The launcher now also accepts repeated, fail-closed
  `--job-key CASE_ID::RUN` selectors only with an isolated batch label, so a
  helper can keep one model load while claiming exact tail rows. Focused
  resume/batch/evaluator tests pass 17/17. Future adapters also acquire an
  exclusive OS file lock per output and recheck the receipt under that lock,
  making concurrent helper/donor duplication impossible and automatically
  releasing ownership if a process exits. Already-running adapters are intentionally
  unaffected, while every later full-shard or tail-helper process loads the
  hardened behavior. The first helper is armed behind GPU 7's current shard
  and will claim only `MirrorRollingBall/360_out/run23` and `run25`, the final
  non-reserved rows of the two heaviest shards, if they are still incomplete.
  Their donor adapters load the receipt-aware skip before those full shards
  start, preventing duplicate work.
- [x] Hardened evaluator resume to remove and retry transient error rows,
  force pinned assets into offline mode, and fail before writing rows when a
  requested CUDA device is unavailable.

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
