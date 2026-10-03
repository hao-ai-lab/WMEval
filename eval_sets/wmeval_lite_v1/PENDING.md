# Pending case data — intake checklist

Deliverables for each planned-but-not-yet-bundled case. Case folders are
created **only when the data lands** (the trust layer rejects placeholders:
everything under `cases/` must validate). Wiring steps for every case:

1. `mkdir -p cases/<Scene>/<variant>/reference/artifacts` and drop the files
2. write `case.yaml` (copy a live case of the same kind as template)
3. `python tools/build_lite_case.py cases/<Scene>/<variant>` (checksums + lock)
4. add the case id to the dimension's `cases:` list in `suite.yaml`
5. `python tools/run_wmeval_lite.py validate` (+ `generate-rubrics` where noted)

Delete each bullet below as its data lands. See `suite.yaml`'s dated header
banners for the history of everything already resolved.

---

## Release cleanup pending register

This is the canonical release backlog as of `lanxiang/release-cleanup`.
Checkboxes are deliberately separated into release-blocking P0 work,
post-release rigor P1 work, and P2 optimizations. An owner label identifies
work that belongs to another evaluation-dimension maintainer; `multi-model`
means the policy must be tuned and validated on more than Project Genie before
it can be frozen.

### P0 — required for a general production-ready suite

- [ ] **Release infrastructure — dependency and artifact locks.** Populate
  every case's `evaluator_lock.dependency_versions` and
  `artifact_schema_versions`, refresh evaluator fingerprints, and make stale
  locks fail CI. The CUDA Table 4/7 environment is now pinned separately, but
  the 19 general Lite case locks remain incomplete.
- [ ] **Release infrastructure + dimension owners — immutable model assets.**
  Pin revisions and checksums for GroundingDINO, SAM2, Depth Anything, DINOv2,
  RAFT/OpenPIV, LPIPS weights, and hosted-judge model snapshots. Preflight now
  names the Table 4/7 runtime aliases, but it cannot certify mutable remote
  aliases or an absent local cache.
- [ ] **Action controllability owner — action target schema (`multi-model`).**
  Extend the versioned semantic trajectory contract with target entity,
  coordinate frame, and unsupported-channel semantics; migrate native arrays
  and supplemental trajectories without changing their commands. Reject an
  object-control score when a keyboard-only backend actually moved the camera.
- [ ] **Runner + kinematics/EOC owners — suite-level candidate normalization
  (`multi-model`).** Add a locked transform for fps, duration, resolution,
  crop/pad, and timebase to non-liquid case contracts and record it per result
  row. Table 4/7 preparers already do this explicitly, but arbitrary Lite
  candidates still use `candidate_normalization: none`.
- [ ] **Result-schema owner — uniform metric validity envelope.** Add per-metric
  status, reason, primary source, units, coverage, normalization provenance,
  and evaluator fingerprint. LPIPS source/event coverage and kinematics label
  coverage are now surfaced, but the contract is not yet uniform across all
  dimensions.
- [ ] **LPIPS owner — deterministic cached reference events.** Cache one
  reference judgment per case + judge model/snapshot + prompt version + frame
  sampling contract, then reuse it for every candidate. Causal and temporal
  fallback rows are now separated, but reference re-judging can still vary.
- [ ] **EOC owner — move supplemental controls into locked bundles.** Promote
  verified BilliardGame, ChessBoard walk, and WanderingGarden controls from
  supplemental manifests into their case bundles and refresh checksums/locks.
  Preserve reconstructed-vs-exact provenance.
- [ ] **EOC owner — anchor validity gate (`multi-model`).** Freeze minimum
  per-anchor visibility, assignment confidence, paired-frame coverage, and
  failure behavior before emitting pairwise-distance MAE. Validate on
  reflective Mirror, indoor Billiard, and outdoor Garden candidates.
- [ ] **Fluid mechanics owner — checkpoint provenance.** Pin and checksum the
  SAM2/RAFT artifacts used by candidate-side fluid evaluation and emit them in
  result provenance.

Completed in this branch: RacingCar first-person is excluded from active Table
4 scoring; the CUDA handoff is hashed and relocatable; LPIPS causal vs temporal
fallback values are separated at row/aggregate level; and kinematics FDE/DTW
become null when any required label is unscored instead of averaging a
favorable subset.

### P1 — post-release rigor and multi-model calibration

- [ ] **Table 4/7 maintainers (`multi-model`).** Validate the reconstructed
  RacingCar third-person, Billiard first-person, and shared Table 7 camera
  controls on additional world models; freeze acceptable gain/timing bands and
  action-target success criteria. Current Genie controls are nominal, not
  universally tuned.
- [ ] **Kinematics owner.** Add held-out reprojection checks and explicit SI vs
  pixel/depth units for every scene; retain error rate as the Table 4 headline
  and significant-digit exact match only as a diagnostic.
- [ ] **EOC owner.** Record depth source per metric and maintain scene-specific
  self-match calibration bounds, especially Mirror's photogrammetric path.
- [ ] **LPIPS owner.** Add target-aware browser adapters for BeamDomino/liquid,
  or mark those backend/case pairs unsupported rather than guessing controls.
- [ ] **Action controllability owner.** Package the PTLFlow synthetic-flow
  dependency and calibration artifact; freeze resolution normalization or
  promote relative EPE.
- [ ] **Prompt-consistency owner.** Mark reconstructed prompts in every receipt,
  retain hashes, capture hosted-model snapshot metadata, and maintain frozen
  regression fixtures with tolerance bands.
- [ ] **Lighting owner.** Capture the original Unity controller trace for the
  fixed-sun orbit with poses/timestamps; keep the optional moving-sun protocol
  separate.
- [ ] **Physical-causality owner.** Preserve every judge pass, snapshot metadata,
  and uncertainty; implement an object-strike browser adapter or declare the
  backend unsupported.
- [ ] **Fluid owner.** Define a target-aware bottle/camera browser adapter and
  reject backends that cannot express it.
- [ ] **Release infrastructure.** Standardize generation repeats vs judge
  repeats (`generation_seed`, `generation_repeat`, `judge_repeat`, aggregation
  axis, `ddof`, effective sample size) and add lightweight CI plus opt-in
  GPU/API integration gates.

### P2 — optimizations and optional extensions

- [ ] Replace the separate macOS/CUDA requirement files with supported uv
  project extras and platform locks; add a fully offline checkpoint preflight.
- [ ] Define canonical evidence retention, archival, and dry-run cleanup for
  local `runs/` caches and superseded reports.
- [ ] Keep the fluid multi-axis panel canonical and label any scalar headline
  as navigation-only.
- [ ] Decide whether ChessBoard lighting is in release scope before bundling it.
- [ ] Implement the optional moving-sun case only with a separate sweep-aware
  metric contract.

---

## Not yet implemented

- **`ChessBoard/lighting`** → `lighting_shading`
  Team decision first: `ChessBoard/walk_third` (a real capture of the
  ChessBoardwithSphere scene, live under `environment_object_consistency`)
  can simply be listed under `lighting_shading.cases` as a borrow if judging
  that rollout is acceptable — it then needs
  `module_params.shadow_consistency.{shadow_text_prompt, shadow_anchor_label}`
  (e.g. `chess piece. sphere.` / `sphere`) on its case.yaml. If lighting
  needs its own first-person fixed-sun orbit like SunRisesAndSets, author a
  new `ChessBoard/lighting` bundle with the same shape as
  `SunRisesAndSets/lighting`. The paper reports lighting numbers on
  SunRisesAndSets only, so this is an extension, not a paper-parity blocker.
  Note: the paper's ChessBoard Unity render was used as the cross-scene
  *measurement* reference for the shadow pipeline (MOSD 73.85, SDJ 3.02,
  SAS 18.16, SSD 4.90; `WMEval-lighting-repro:eval/metrics/unity_chessboard__metrics.json`).

- **`SunRisesAndSets/lighting_sunsweep`** (OPTIONAL, not in the paper) →
  `lighting_shading`
  `SunRisesAndSets/lighting` landed 2026-09-15 with the paper's protocol
  (first-person, FIXED sun, walking orbit; see the suite.yaml banner). The
  moving-sun reading of the scene name would be a *separate* case: static
  camera, prompt-driven sun sweep, scored against the expected azimuth
  sweep (monotone, smooth; MOSD still applies) rather than against a fixed
  direction — the current SDJ/MOSD formulas would penalise the sweep itself.
  Unity is not needed at eval time (the Unity light log from the
  rotating-sun capture already exists for calibration), but no world-model
  rollouts exist for that prompt yet, and the shadow module needs a
  sweep-aware scoring rule first. Only author it if the team wants a second
  lighting protocol.

## Known limitations on live dimensions

- **Real-candidate coverage.** `lighting_shading` was run on the paper's
  6 models x 5 seeds (2026-09-15, reproduces the paper exactly), and
  Waypoint-1.5 has been run through every live dimension: the non-VLM ones
  on 2026-09-22 (55 rows, no errors) and `prompt_scene_consistency` +
  `lpips` on 2026-09-24 (30 + 10 rows, all completed, ~3M gpt-5.4 input
  tokens in total). Numbers are in the suite.yaml banners. Waypoint is a
  first-person, keyboard/mouse-only model, so third-person and object-control
  cases measure its action-target ambiguity rather than a strong result.
- **The reference-side causal judge is re-run per candidate and is not
  deterministic.** With `lpips_causal_event_source: both` each row re-judges
  the reference video (~55k input tokens); on the identical BeamDomino
  reference it reported 0, 1, 3, 0 and 0 contacts across five rows. Cache the
  reference judgement per case (and per judge model/prompt version) so every
  candidate is scored against the same reference events.
- **Candidate normalisation is undefined for the non-liquid cases.**
  `kinematics_mechanics` and `environment_object_consistency` read every
  frame and compare candidate frame *f* with Unity frame *f* in the
  reference pixel frame, but `candidate_normalization: none` does no
  resampling, so a candidate at another fps/resolution (Waypoint: 60 fps
  1280x720) is silently misaligned. `tools/generate_waypoint15_candidates_modal.py`
  works around it by writing a copy resampled to the reference's fps,
  resolution and duration; the suite should either add a normalisation
  token for these cases or resample inside the adapter.
- **Three cases under-specify the generation condition.**
  `BilliardGame/prompt_first` and `WanderingGarden/scene_first` ship no
  `reference/artifacts/actions.txt`, and `ChessBoard/walk_third`'s is a
  comment-only frame table. Any candidate model has to guess; the Waypoint
  generator hard-codes the bundles' own notes (idle for 2.5 s / 360-degree
  yaw over 10 s / the walk-diagonal-sprint-turn table). Author real action
  scripts and re-lock the three bundles.
- **`kinematics_mechanics`'s mechanical quantities** (momentum/kinetic-energy
  exact-match) inherit the monocular depth model's domain-gap bias and are
  only meaningfully accurate for objects with real, non-trivial motion.
- **`MirrorRollingBall`'s depth accuracy** relies on an apparent-size
  (photogrammetric) workaround, not a monocular depth model — no monocular
  depth model has been found that handles its reflective sphere correctly.

- **`physical_causality`'s CSR needs the repeated passes to mean anything.**
  CSR is all-or-nothing per video (`max_chain_length >= num_segments`,
  `causalstep/metrics.py`) and each lite case bundles ONE video, so a single
  judge pass can only ever report 0 or 100 — and the judge is not reproducible
  at `temperature=0` (two runs of the identical config disagree on 4 of the 19
  bundled QA items). `causal_step_judge_repeats: 5` is therefore load-bearing,
  not a tuning knob: it restores the paper's cell semantics, where CSR over five
  passes is successes/5. Lowering it silently turns `csr_delta` back into a coin
  flip. `amcl_delta` is carried in `extra_means` as the ungated companion.
