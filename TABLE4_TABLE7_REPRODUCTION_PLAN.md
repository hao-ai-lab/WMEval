# Table 4 and Table 7 reproduction plan

## Goal

Re-run the paper's Table 4 (kinematics and mechanics) and Table 7
(environment/object/character consistency) with five Project Genie generations
per cell, replaying the same semantic action trajectories in Unity and Project
Genie, then emit paper-shaped tables containing the published value, rerun
value, and delta.

This plan distinguishes a paper-parity run from the currently runnable WMEval
Lite proxies. A result is not labelled a reproduction unless the scene,
viewpoint, action trajectory, duration, seed frame, evaluator definition, and
five-run aggregation all pass the preflight checks below.

## Paper scope

### Table 4: kinematics and mechanics

- Scenes: `RacingCar`, `BilliardGame`.
- Views: first person and third person for both scenes.
- Metrics: DTW, FDE, and mechanical Error Rate. Exact Match is retained only in
  historical/raw inputs and is omitted from the active reporting contract.
- Paper aggregation: mean and sample standard deviation over five generations
  per model/scene/view cell.
- Project Genie is the first target. Other model rows can use the same protocol
  after their action adapters are calibrated.

### Table 7: environment/object/character consistency

- Scenes: `WanderingGarden`, `BilliardGame`, `MirrorRollingBall`.
- Camera trajectories per scene: perturbation-in, perturbation-out, 360-in,
  and 360-out.
- Metrics: Depth AbsRel, pairwise-distance MAE, object permanence, and
  character consistency, with N/A preserved where the paper marks a metric as
  inapplicable.
- Paper aggregation: mean and sample standard deviation over five generations
  per model/scene/trajectory cell.

## Current repository readiness

The repository can validate bundles, discover candidates, run the wired
dimensions, resume interrupted work, and write per-case JSONL plus aggregate
JSON/CSV. The local `uv` environment passes the full test suite.

The current Lite suite is not sufficient for exact paper parity:

| Requirement | Current state | Required action |
| --- | --- | --- |
| Table 4 `RacingCar` first/third driving controls | A validated semantic steering/throttle/brake trajectory now compiles to repo-native and browser artifacts; the paper's numeric timeline is unpublished | Recover the author timeline or approve the reconstructed trajectory, then capture matching Unity state |
| Table 4 `BilliardGame` | Active rerun is now first-person only with a compiled W/A/D ball-control trajectory, matched Unity impulses, a 10-second/300-frame reference, and metric felt-plane anchors. The third-person row is retained only in the published baseline and explicitly excluded. | Generate five new first-person Genie traces, then run the metric evaluator. Exact paper parity still requires the unpublished cue-shot timeline, scale transform `S`, and camera definition. |
| Table 4 3D FDE/DTW | Lite reports a 2D reprojection proxy | Recover candidate camera pose/metric trajectory or implement the paper's documented static-frame mapping |
| Table 4 physical Error Rate | Available from calibrated planar trajectories and simulator mass | Validate metric scale and bounded event alignment against Unity calibration clips before reporting paper parity |
| Table 7 four camera trajectories | All 12 cells now have orientation-specific Unity frame-zero assets and four semantic yaw proxies; exact paper anchors and rate/duration remain unpublished | Run the reconstructed setting while preserving its caveat; replace assets if author-native anchors/timelines are recovered |
| Table 7 action scripts | Canonical YAML now compiles to `action.npy`, `actions.txt`, audited NPZ, and Project Genie JSONL | Calibrate each scene/backend adapter and preserve accepted-action traces |
| Table 7 `MirrorRollingBall` distance MAE | Two fixed foreground sphere anchors are now locked to `YellowBall (30)` and `(32)` with 4.0 m separation | Reevaluate the five existing runs per trajectory; retain the projection-assisted identity diagnostics |
| Other-dimension trajectories | 15/19 Lite cases have native numeric and semantic controls. Billiard and Garden have compiled Table 4/7 supplements; Chess walk has an exact compiled 90-frame supplement; SunRisesAndSets has a compiled 160-step semantic orbit matching its locked 10-second browser schedule | Keep the inventory current and do not label the four supplemental controls as native locked-bundle assets |
| Project Genie generation | The authenticated web UI is automated end to end. The prepared queue now has 70 runnable repeats: 50 existing validated repeats and 20 pending Billiard Table 7 repeats. Ten Table 4 Billiard repeats remain unsupported | Run the 20 new Billiard Table 7 generations; retain the hosted UI's lack of controllable RNG seeds and reconstructed timing scope |
| Local Unity execution | Unity 6000.3.7f1 players and validated captures now exist for all seven previously missing cells, with frame-zero PNG, video, poses, intrinsics, transforms, and hashes | Preserve private-scene revision provenance and rerun the capture gate after any Unity change |

## Canonical action-trajectory contract

Every paper-parity case carries a versioned `trajectory.yaml` plus repo-native
`action.npy`/`actions.txt`, a self-describing `action.npz`, and a browser
`actions.jsonl` where the web action space is representable. The semantic file
is authoritative; Unity and model adapters are derived products.

Required metadata:

- `trajectory_id`, schema version, scene revision, Unity project commit, and
  view (`first`, `third`, `top_down`).
- fixed reset seed, physics timestep, control rate, render rate, duration, and
  warm-up duration.
- coordinate conventions and units.
- named control channels with type/range, for example
  `steer [-1,1]`, `throttle [0,1]`, `brake {0,1}`, `yaw_rate deg/s`, or
  billiards `hit_x`, `hit_y`, `strength`, `spin_x`, `spin_y`.
- timestamped, piecewise-constant or piecewise-linear segments. Timestamps,
  not frame indices, are the source of truth.
- reset pose, camera pose, tracked-object IDs, and expected terminal event.
- SHA256 hashes for the semantic trajectory, compiled array, Unity first
  frame, Unity video, and Unity state log.

Compilation rules:

1. Use a canonical timeline in seconds.
2. Resample continuous controls with zero-order hold at each backend's control
   rate; emit discrete edges exactly once.
3. Preserve action dose (`integral |u(t)| dt`) and segment boundaries during
   resampling.
4. Record the actions actually accepted by the backend, not only the requested
   actions.
5. Store adapter name/version and a reverse-decoded semantic trace so the
   compiled trajectory can be audited.

## Calibration protocol

Calibrate each action backend before generating table candidates.

1. **Reset calibration:** run five resets with no action. Verify first-frame
   pixel stability, camera pose, object pose, and absence of hidden warm-up
   motion.
2. **Per-channel excitation:** run positive/negative impulses, step holds, and
   ramps independently for each continuous channel; toggle each discrete
   channel once.
3. **Measure response:** use Unity state directly and candidate optical
   flow/object tracks to estimate latency, dead zone, gain, saturation, and
   sign/axis convention.
4. **Fit adapter:** map canonical controls to backend controls with a monotone
   piecewise-linear gain plus integer-step latency. Do not time-warp individual
   evaluation rollouts after observing their outcomes.
5. **Hold-out validation:** replay a mixed trajectory not used in fitting.
   Gate on action timing, total yaw/displacement, endpoint, and event order.
6. **Freeze calibration:** hash the calibration clips, fit parameters, adapter
   code, and acceptance report. All five generation seeds use the same frozen
   adapter.

Suggested trajectory gates:

- start-frame camera/object pose within the Unity capture tolerance;
- duration within one target frame;
- every action boundary within one control step;
- sign agreement on every excited axis;
- integrated yaw/translation within 5% for camera paths;
- correct collision/event order for billiards;
- no post-hoc per-seed tuning.

Thresholds should be finalized from repeated Unity captures before candidate
generation. The same frozen thresholds apply to all models.

## Unity-to-Project-Genie pipeline

1. **Unity capture**
   - reset the exact scene revision and viewpoint;
   - render a lossless pre-action frame (`ref.png`);
   - begin state/video recording before the first control edge;
   - replay the compiled Unity trajectory;
   - write video, accepted controls, camera matrices, object transforms,
     masses, collision/event log, and capture metadata atomically.

2. **Initial conditioning**
   - use the exact Unity `ref.png`, without JPEG conversion or model-specific
     crop;
   - if a backend requires resizing, preserve the original and record the
     deterministic resize/pad transform;
   - verify the decoded candidate frame 0 against the actual conditioning
     image and report the transform/error.

3. **Project Genie generation**
   - supply the case prompt, conditioned Unity frame, frozen model adapter,
     canonical trajectory, and five independent run IDs `run0`--`run4`;
   - do not label these runs as RNG seeds: the hosted web UI exposes no seed
     control;
   - capture accepted action events and timestamps from the session;
   - store the native output and a provenance JSON before normalization.

4. **Normalization**
   - create a derived evaluation copy at the reference resolution, fps, and
     duration using one pinned ffmpeg command;
   - retain the native file; never evaluate a silently overwritten video;
   - record frame mapping and hashes for both files.

5. **Evaluation**
   - run only the table's frozen metric revision;
   - save per-object/per-frame diagnostics, not just the headline mean;
   - fail a row when a required object is untracked instead of dropping that
     object from the mean;
   - run Unity-vs-Unity identity checks and known perturbation controls before
     scoring Project Genie.

6. **Aggregation and cross-comparison**
   - aggregate exactly five seeds with sample standard deviation (`ddof=1`);
   - preserve N/A rather than coercing to zero;
   - emit JSON, CSV, Markdown, and LaTeX from the same reviewed result object;
   - for every paper cell emit `published`, `rerun`, `delta`, relative delta
     where meaningful, seed count, and validation status.

## Automatic command surface

The implemented single-device preparation and generation interface is:

```bash
uv run python tools/prepare_table_trajectories.py audit --strict
uv run python tools/prepare_table_trajectories.py compile
uv run python tools/run_unity_table_scenes.py all
.venv-unity/bin/python tools/unity_trajectory_runner.py \
  --output-dir runs/unity/garden_bridge_gt capture-ground-truth
uv run python tools/project_genie_agent.py doctor
uv run python tools/run_project_genie_tables.py plan
uv run python tools/run_project_genie_tables.py run --resume \
  --turn-calibration table7/WanderingGarden/360_in=runs/project_genie/calibration/garden_bridge_arrow_left_attempt4/turn_calibration.json
uv run python tools/align_project_genie_native.py --run-dir <run-directory>
uv run python tools/prepare_table_assets.py
uv run python tools/run_prepared_table_evaluations.py --prepare-only
PYTORCH_ENABLE_MPS_FALLBACK=1 \
  uv run python tools/run_prepared_table_evaluations.py --device mps
uv run python tools/compare_table_results.py template
uv run python tools/compare_table_results.py compare \
  --results experiments/tables_4_7/rerun_results.template.json
```

Generation is resumable and receipts include input/output hashes. Revisited
worlds are calibration-only and cannot satisfy an independent table repeat.
Table runs use the validated turn period and a fixed 10-second horizon;
closed-loop visual return remains diagnostic-only. Successful runs
automatically emit a validated, UI-free native trajectory clip. Strict paper-parity preflight
intentionally remains red while author-native timelines, orientation anchors,
adapter calibration, and Billiard controls/views are unresolved. The
reconstructed asset audit is independently green.
Normalization, trajectory-specific ground-truth remapping, metric execution,
and incomplete cross-comparison rendering are now automated for every completed
five-run group. Garden uses its periodic Unity orbit in the commanded yaw
direction (negative Project Genie yaw walks the positive Unity capture
backward); perturbations use each run's measured accepted/completed action
times. Mirror pose selection is derived from camera-forward yaw because its
recorded 300-frame orbit covers about 399 degrees rather than one uniform turn.
The comparison renderer still fails closed for a complete paper table until all
five numeric runs are present for every applicable metric.
The evaluation driver now applies the same rule at process level:
`--fail-incomplete` exits with status 2 when any selected cell is blocked, a
five-run generation group is incomplete, or an applicable metric is null in
any repeat, even when all five evaluator rows themselves completed.

The latest device run completed all nine runnable diagnostic cells and reused
the locked MPS evaluator fingerprints on validation. Four cells have every
applicable metric, while five have completed rows but incomplete metrics (one
Garden distance value and the source-blocked Mirror distance metric). The
paper-shaped output is
`reports/tables_4_7/project_genie/consolidated/comparison.md`; its overall
`Complete` value intentionally remains false.

## Validation controls

Run these before accepting a new table:

- reference bundle checksums and evaluator tests;
- Unity-vs-Unity identity evaluation;
- identical-video, one-frame-shift, temporal-scale, horizontal-flip, frozen
  video, and object-removal controls;
- first-frame hash/transform audit;
- action accepted-vs-requested trace audit;
- object-label coverage for every required metric cell;
- all 10 three-of-five seed subsets to check rank/order stability;
- bootstrap confidence intervals as a supplement, while retaining the paper's
  mean +/- standard deviation in the primary table;
- manual inspection packets for the worst delta and every failed gate.

## Result-table format

Keep the paper's row/column layout. Each cell in the comparison artifact has
three lines:

```text
published: mean +/- std
rerun:     mean +/- std
delta:     rerun - published
```

The Markdown/CSV companion also includes `n`, action-trajectory hash,
conditioning-frame hash, Unity capture hash, evaluator fingerprint, and a
status in `{paper_parity, proxy_only, invalid}`. Only `paper_parity` cells may
be used to claim reproduction.

## Immediate execution order

1. Use `run_ready_project_genie.sh` to retain the completed Garden, Mirror, and
   RacingCar groups and generate the 20 pending Billiard Table 7 repeats. The
   runner is resumable and automatically handles the guarded upload consent.
2. Run `run_prepared_table_evaluations.py`; it detects those complete groups,
   validates receipts, remaps camera ground truth, evaluates, and writes the
   partial comparison automatically.
   Earlier Garden checkpoints are superseded because the detector
   configuration collapsed five paper anchors into a mostly single-tree
   track. The corrected evaluator keeps the exact five-anchor denominator,
   assigns identities from aligned Unity projections, fingerprints the full
   case contract, and replaces stale resume rows before rebuilding the table.
   `tools/audit_table_reproduction.py` provides the machine-readable live
   completion view in `runs/tables_4_7/reproduction_status.json` and ignores
   result rows produced by stale evaluator fingerprints.
3. Treat RacingCar third-person's current five-run result only as a 2-D pixel
   proxy (`FDE 136.7864 ± 52.0648 px`; `DTW 36897.5904 ± 17961.5102 px`), not
   as a numeric cross-comparison with the paper's differently defined metrics.
4. Regenerate Garden `perturb_out/run1`, whose generated world exposes fewer
   than two configured anchors, then reevaluate the repaired Mirror anchors.
5. Resolve the remaining source blockers: unpublished exact prompts/timelines
   and orientation anchors, Table 4 RacingCar first-person evaluator wiring,
   the Table 4 Billiard control contradiction, and Table 7 dynamic-scene
   paper-metric parity.
